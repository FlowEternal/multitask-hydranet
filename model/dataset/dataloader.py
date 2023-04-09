# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is the class for CurveLane dataset."""

import os
import json
import cv2
import numpy as np
import warnings
import yaml

from torch.utils.data import Dataset
import torch.utils.data.dataloader
from dataset.utility import get_img_whc, imread, create_subset, resize_by_wh, bgr2rgb, imagenet_normalize, load_json

# 车道线encoder / decoder
from head_lane.lane_codec import LaneCodec

#---------------------------------------------------#
#  数据增强函数
#---------------------------------------------------#
import imgaug as ia
import imgaug.augmenters as iaa

# 车道线
from imgaug.augmentables.lines import LineStringsOnImage
from imgaug.augmentables.lines import LineString as ia_LineString

# 目标检测
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmentables.bbs import BoundingBox as ia_Bbox

#  语义分割增强函数
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

def _lane_argue(*,
                image,
                lane_label=None,
                det_label=None,
                seg_label=None,
                do_flip=False,
                do_split=False,
                split_ratio=None,
                ):

    #---------------------------------------------------#
    #  定义增强序列
    #---------------------------------------------------#
    color_shift = iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.5, 1.5)),
        iaa.LinearContrast((1.5, 1.5), per_channel=False),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(0, iaa.Multiply((0.7, 1.3)))),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(1, iaa.Multiply((0.1, 2)))),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(2, iaa.Multiply((0.5, 1.5)))),
    ])

    geometry_trans_list = [
            iaa.Fliplr(),
            iaa.TranslateX(px=(-16, 16)),
            iaa.ShearX(shear=(-15, 15)),
            iaa.Rotate(rotate=(-15, 15))
        ]

    if do_flip:
        geometry_trans_list.append(iaa.Flipud())

    if do_split:
        # top right down left
        split_one = iaa.Crop(percent=([0, 0.2], [1 - split_ratio], [0, 0], [0, 0.15]), keep_size=True)# 右边是1-ratio
        split_two = iaa.Crop(percent=([0, 0.2], [0, 0.15], [0, 0], [split_ratio]), keep_size=True)
        split_shift = iaa.OneOf([split_one, split_two])

    else:
        geometry_trans_list.append(iaa.Crop(percent=([0, 0.2], [0, 0.15], [0, 0], [0, 0.15]), keep_size=True))
        split_shift = None

    posion_shift = iaa.SomeOf(4, geometry_trans_list)

    if do_split:
        aug = iaa.Sequential([
            iaa.Sometimes(p=0.6, then_list=color_shift),
            iaa.Sometimes(p=0.6, then_list=split_shift), # 以0.5概率去split debug时候1.0
            iaa.Sometimes(p=0.6, then_list=posion_shift)
        ], random_order=True)

    else:
        aug = iaa.Sequential([
            iaa.Sometimes(p=0.6, then_list=color_shift),
            iaa.Sometimes(p=0.6, then_list=posion_shift)], random_order=True)

    # =========================================
    # 开始数据增强
    # =========================================
    args = dict(images=[image])
    if lane_label is not None:
        lines_tuple = [[(float(pt['x']), float(pt['y'])) for pt in line_spec] for line_spec in lane_label['Lines']]
        lss = [ia_LineString(line_tuple_spec) for line_tuple_spec in lines_tuple]
        lsoi = LineStringsOnImage(lss, shape=image.shape)
        args.update({"line_strings":[lsoi]})

    # =========================================
    # 做语义分割增强
    # =========================================
    if seg_label is not None:
        segmap = SegmentationMapsOnImage(seg_label , shape=image.shape)
        args.update({"segmentation_maps":[segmap]})

    # =========================================
    # 做目标检测增强
    # =========================================
    if det_label is not None:
        bbox_list = [ia_Bbox( *list(one_det_poly[:5])) for one_det_poly in det_label]
        deoi = BoundingBoxesOnImage(bbox_list, shape=image.shape)
        args.update({"bounding_boxes": [deoi]})

    # =========================================
    # 开始增强
    # =========================================
    batch = ia.Batch(**args)
    batch_aug = list(aug.augment_batches([batch]))[0]  # augment_batches returns a generator
    image_aug = batch_aug.images_aug[0]

    # 增强line
    aug_result = dict(images=image_aug)
    if lane_label is not None:
        # lsoi_aug = batch_aug.line_strings_aug[0].clip_out_of_image() # 这个很重要
        lsoi_aug = batch_aug.line_strings_aug[0] # 这里不clip_out_of_image()

        lane_aug = [[dict(x= float(int(kpt.x)), y=float(int(kpt.y))) for kpt in shapely_line.to_keypoints()]
                    for shapely_line in lsoi_aug]

        aug_result.update({"lane_aug": dict(Lines=lane_aug,Labels=None)})

    # 增强detection
    if det_label is not None:
        # 这里clip out of image 会有问题，所以不clip
        deoi_aug = batch_aug.bounding_boxes_aug[0].clip_out_of_image()
        if len(deoi_aug) ==0:
            det_label_aug = np.zeros((0, 5)) # clip后没有目标的情况
        else:
            det_label_aug = np.vstack([np.hstack([det_bbox.coords.reshape(1,4),det_bbox.label.reshape(1,1)]) for det_bbox in deoi_aug])
        aug_result.update({"det_aug":det_label_aug})

    # 增强分割掩码
    if seg_label is not None:
        segmap_aug = batch_aug.segmentation_maps_aug[0]
        aug_result.update({"seg_aug":segmap_aug})

    return aug_result

def _read_curvelane_type_annot(annot_path):
    return load_json(annot_path)

class MultitaskData(Dataset):
    def __init__(self, cfgs, mode):
        """Construct the dataset."""
        super().__init__()
        self.cfgs = cfgs
        self.mode = mode

        # 加载通用
        self.network_input_width = self.cfgs["dataloader"]["network_input_width"]
        self.network_input_height = self.cfgs["dataloader"]["network_input_height"]

        # 数据增强配置
        self.with_aug = self.cfgs["dataloader"]["with_aug"]
        if self.mode == "val": self.with_aug = False

        self.do_split_img = self.cfgs["dataloader"]["do_split"]
        self.do_flip_img = self.cfgs["dataloader"]["do_flip"]

        # 加载检测相关
        self.train_detect = self.cfgs["train"]["train_detect"]
        self.det_class_list = self.cfgs["detection"]["class_list"]
        self.det_num_classes = len(self.det_class_list)
        self.det_class_to_ind = dict(zip(self.det_class_list, range(self.det_num_classes)))

        # 加载分割相关
        self.train_seg = self.cfgs["train"]["train_seg"]
        self.seg_class_list = self.cfgs["segment"]["class_list"]

        # 加载车道线相关
        self.train_lane = self.cfgs["train"]["train_lane"]
        self.anchor_stride = self.cfgs["lane"]["anchor_stride"]
        self.interval = self.cfgs["lane"]["interval"]
        self.anchor_lane_num = self.cfgs["lane"]["anchor_lane_num"]
        self.interpolate = self.cfgs["lane"]["interpolate"]  # 是否插值 如果插值 即为沿长到版边 和原代码就完全一致
        self.scale_invariance = self.cfgs["lane"]["scale_invariance"]
        self.points_per_line = int(self.network_input_height / self.interval)

        if not (self.train_lane or self.train_seg or self.train_detect):
            print("must train at least one header")
            exit()

        self.lane_coder = LaneCodec(input_width=self.network_input_width,
                                        input_height=self.network_input_height,
                                        anchor_stride=self.anchor_stride,
                                        points_per_line=self.points_per_line,
                                        do_interpolate=self.interpolate,
                                        anchor_lane_num=self.anchor_lane_num,
                                        scale_invariance=self.scale_invariance,
                                        )

        self.encode_lane = self.lane_coder.encode_lane

        # 准备数据集
        self.data_list = self.cfgs["dataloader"]["data_list"]
        self.data_list_train = os.path.join(self.data_list, "train.txt")
        self.data_list_valid = os.path.join(self.data_list, "valid.txt")

        dataset_pairs = dict(
            train=create_subset(self.data_list_train,with_lane=self.train_lane,with_seg=self.train_seg,with_detect=self.train_detect),
            val=create_subset(self.data_list_valid,with_lane=self.train_lane,with_seg=self.train_seg,with_detect=self.train_detect)
        )

        if self.mode not in dataset_pairs.keys():
            raise NotImplementedError(f'mode should be one of {dataset_pairs.keys()}')
        self.image_annot_path_pairs = dataset_pairs.get(self.mode)

        # 收集函数
        self.collate_fn = Collater(target_height=self.network_input_height,
                                   target_width=self.network_input_width,
                                   is_lane = self.train_lane,
                                   is_seg=self.train_seg,
                                   is_det=self.train_detect)

    def __len__(self):
        """Get the length.

        :return: the length of the returned iterators.
        :rtype: int
        """
        return len(self.image_annot_path_pairs)

    def __getitem__(self, idx):
        """Get an item of the dataset according to the index.

        :param idx: index
        :type idx: int
        :return: an item of the dataset according to the index
        :rtype: dict
        """
        return self.prepare_img(idx)

    def prepare_img(self, idx):
        """Prepare an image for training.

        :param idx:index
        :type idx: int
        :return: an item of data according to the index
        :rtype: dict
        """
        target_pair = self.image_annot_path_pairs[idx]
        image_arr = imread(target_pair['image_path'])
        whc = get_img_whc(image_arr)

        #---------------------------------------------------#
        #  加入车道线标签
        #---------------------------------------------------#
        if self.train_lane:
            lane_label = self.parse_own_label(load_json(target_pair['annot_path_lane']))
            annot_lane_path = target_pair['annot_path_lane']
        else:
            lane_label = None
            annot_lane_path = None

        # ---------------------------------------------------#
        #  加入语义分割
        # ---------------------------------------------------#
        if self.train_seg:
            seg_label = cv2.imread(target_pair['annot_path_seg'], cv2.IMREAD_UNCHANGED)
        else:
            seg_label = None

        # ---------------------------------------------------#
        #  加入目标检测
        # ---------------------------------------------------#
        if self.train_detect:
            obj_label = self.load_detect_annot(target_pair['annot_path_detect'])
        else:
            obj_label = None

        if DEBUG:
            self.draw_label_on_image(image_arr, lane_label, obj_label, seg_label, "img_org.png")

        # =========================================
        # 数据增强
        # =========================================
        if self.with_aug:
            if self.do_split_img:
                do_split, split_ratio = self.cal_split(image_arr, lane_label)
            else:
                do_split = False
                split_ratio = None

            aug_dict = _lane_argue(image=image_arr,
                                   lane_label=lane_label,
                                   det_label=obj_label,
                                   seg_label=seg_label,
                                   do_flip=self.do_flip_img,
                                   do_split=do_split,
                                   split_ratio=split_ratio
                                   )

            # =========================================
            # 取出数据
            # =========================================
            # 1.图像
            image_arr = aug_dict["images"]

            # 2.车道线
            if self.train_lane:
                lane_label = aug_dict["lane_aug"]

            # 3.覆盖分割
            if self.train_seg:
                seg_label = aug_dict["seg_aug"].arr[:, :, 0].astype(np.uint8)  # 分割标签 原图尺度

            # 4.检测标签
            if self.train_detect:
                obj_label = aug_dict["det_aug"]

            if DEBUG:
                self.draw_label_on_image(image_arr, lane_label, obj_label, seg_label, "img_aug.png")

        # =========================================
        # lane进一步进行decode
        # =========================================
        if self.train_lane:
            encode_type, encode_loc = self.encode_lane(lane_object=lane_label,
                                                                   org_width=whc['width'],
                                                                   org_height=whc['height'])

            if self.scale_invariance:
                # up anchor
                encode_loc[:, self.points_per_line + 2: 2 * self.points_per_line + 2] /= self.interval
                # down anchor
                encode_loc[:, :self.points_per_line] /= self.interval

            encode_loc = encode_loc.astype(np.float32)
            encode_type = encode_type.astype(np.float32)

        else:
            encode_type, encode_loc, encode_cls = None, None, None


        # 图像resize
        network_input_image = bgr2rgb(resize_by_wh(img=image_arr,
                                                   width= self.network_input_width,
                                                   height=self.network_input_height))

        net_input_image_shape = json.dumps(dict(width=self.network_input_width, height=self.network_input_height, channel=3))

        result = dict(
                      image=np.transpose(imagenet_normalize(img=network_input_image), (2, 0, 1)).astype('float32'),
                      src_image_shape=whc,
                      net_input_image_shape=net_input_image_shape,
                      src_image_path=target_pair['image_path'],
                      annot_lane=json.dumps(lane_label),
                      annot_lane_path=annot_lane_path,
                      gt_loc=encode_loc,
                      gt_cls=encode_type,
                      gt_seg=seg_label,
                      gt_det=obj_label
                      )

        return result

    @staticmethod
    def parse_own_label(labels):
        lane_list = {"Lines": [], "Labels": []}
        for one_line in labels["shapes"]:
            labels = one_line["label"]
            pts = one_line["points"]
            one_line_list = [{"x": pt[0], "y": pt[1]} for pt in pts]
            lane_list["Lines"].append(one_line_list)
            lane_list["Labels"].append(labels)
        assert len(lane_list["Lines"])==len(lane_list["Labels"])
        return lane_list

    @staticmethod
    def load_detect_annot(labels_txt):
        annotations_list = open(labels_txt).readlines()
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_list) == 0:
            return annotations

        for idx, one_label in enumerate(annotations_list):
            one_label = one_label.strip("\n").split(",")
            x1 = int(one_label[0])
            y1 = int(one_label[1])
            x2 = int(one_label[2])
            y2 = int(one_label[3])
            category_id = int(one_label[4])  # 这里0为背景，因此所有非背景目标都从1开始
            width = x2 - x1
            height = y2 - y1

            # some annotations have basically no width / height, skip them
            if width < 1 or height < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = [x1, y1, width, height]
            annotation[0, 4] = category_id - 1  # 这里减一为去掉背景
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    @staticmethod
    def cal_split(image,lane_object):
        height , width = image.shape[0], image.shape[1]
        k0_list = []
        k1_list = []
        all_lines = []
        for one_lane in lane_object["Lines"]:
            x_list = []
            y_list = []
            one_line_pts = []
            for pt_index in range(len(one_lane)):
                one_pt = (int(float(one_lane[pt_index]["x"])), height - int(float(one_lane[pt_index]["y"])))
                x_list.append(one_pt[0])
                y_list.append(one_pt[1])
                one_line_pts.append(one_pt)
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    coeff = np.polyfit(x_list, y_list, 1)
                except np.RankWarning:
                    return False,None
                except:
                    return False, None
            k0 = coeff[1]
            k1 = coeff[0]
            k0_list.append(k0)
            k1_list.append(k1)
            all_lines.append(one_line_pts)

        # 进行逻辑判断
        k1_list = np.array(k1_list)
        sorted_k1 = np.sort(k1_list)
        index = np.argsort(k1_list)
        if np.all(sorted_k1>=0) or np.all(sorted_k1) <=0:
            do_split_possible = False
            split_ratio = None
        else:
            index_left_lane = np.where(sorted_k1 <=0)[0][0] # 负得最大的那个为左
            left_lane_index = index[ index_left_lane ]
            right_lane_index = index[-1] # 正得最大的那个为左

            left_lane_pts = np.array(all_lines[left_lane_index])
            right_lane_pts = np.array(all_lines[right_lane_index])

            left_lane_pts_sort = left_lane_pts[ np.argsort((left_lane_pts)[:,1],axis=0)  ]
            right_lane_pts_sort = right_lane_pts[ np.argsort((right_lane_pts)[:,1],axis=0)  ]

            left_x_ = left_lane_pts_sort[0,0]
            right_x_ = right_lane_pts_sort[0,0]
            do_split_possible = True
            split_ratio = (left_x_ + right_x_) / 2.0 / width

        return do_split_possible, split_ratio

    @staticmethod
    def draw_line_on_image(image, lane_object, save_name):
        im_vis_org = image.copy()
        for one_lane in lane_object["Lines"]:
            rd_color = (int(np.random.randint(0, 255)),
                        int(np.random.randint(0, 255)),
                        int(np.random.randint(0, 255)))
            for pt_index in range(len(one_lane) - 1):
                one_pt = one_lane[pt_index]
                one_pt_next = one_lane[pt_index + 1]
                one_pt = (int(float(one_pt["x"])), int(float(one_pt["y"])))
                one_pt_ = (int(float(one_pt_next["x"])), int(float(one_pt_next["y"])))
                print(one_pt)
                cv2.line(im_vis_org, one_pt, one_pt_, rd_color, 3)
        cv2.imwrite(save_name, im_vis_org)

    def draw_label_on_image(self, image, lane_label, obj_label ,seg_label, save_name):
        im_vis_org = image.copy()

        # 语义分割
        np.random.seed(1991)
        seg_arr_vis = np.zeros_like(image)
        for idx in range(len(self.seg_class_list)):
            value = (int(np.random.randint(128,255)),int(np.random.randint(128,255)),int(np.random.randint(128,255)))
            seg_arr_vis[seg_label==idx]=value
        im_vis_org = cv2.addWeighted(im_vis_org,0.5,seg_arr_vis,0.5,0.0)

        # 车道线
        for one_lane in lane_label["Lines"]:
            rd_color = (0,255,0)
            for pt_index in range(len(one_lane) - 1):
                one_pt = one_lane[pt_index]
                one_pt_next = one_lane[pt_index + 1]
                one_pt = (int(float(one_pt["x"])), int(float(one_pt["y"])))
                one_pt_ = (int(float(one_pt_next["x"])), int(float(one_pt_next["y"])))
                print(one_pt)
                cv2.line(im_vis_org, one_pt, one_pt_, rd_color, 3)

        # 目标检测
        for idx,one_box in enumerate(obj_label):
            x1, y1, x2, y2 = int(one_box[0]),int(one_box[1]),int(one_box[2]),int(one_box[3])
            class_category = int(one_box[4]) # 这里非背景类从1开始
            pt1=(x1,y1)
            pt2=(x2,y1)
            pt3=(x2,y2)
            pt4=(x1,y2)
            cv2.line(im_vis_org,pt1,pt2,(0,255,0),2)
            cv2.line(im_vis_org,pt2,pt3,(0,0,255),2)
            cv2.line(im_vis_org,pt3,pt4,(0,0,255),2)
            cv2.line(im_vis_org,pt4,pt1,(0,0,255),2)

            fontScale = 0.5
            thickness = 1
            font = cv2.FONT_HERSHEY_COMPLEX
            pt_txt = (pt1[0], pt1[1] - 5)

            cv2.putText(im_vis_org, str(self.det_class_list[class_category + 1]), pt_txt, font, fontScale,
                        [0, 0, 0], thickness=thickness,lineType=cv2.LINE_AA)

        cv2.imwrite(save_name, im_vis_org)


class Collater(object):
    def __init__(self,
                 target_width,
                 target_height,
                 is_lane=True,
                 is_det=True,
                 is_seg=True):
        self.target_width = target_width
        self.target_height = target_height
        self.is_lane = is_lane
        self.is_det = is_det
        self.is_seg = is_seg

    def __call__(self, batch):
        image_data = np.stack([item["image"] for item in batch]) # images
        image_data = torch.from_numpy(image_data)
        img_shape_list = np.stack([item["src_image_shape"] for item in batch])  # cls

        # =========================================
        # 处理车道线
        # =========================================
        if self.is_lane:
            gt_loc = np.stack([item["gt_loc"] for item in batch]) # location
            gt_loc = torch.from_numpy(gt_loc)
            gt_cls = np.stack([item["gt_cls"] for item in batch]) # cls
            gt_cls = torch.from_numpy(gt_cls)
        else:
            gt_loc = None
            gt_cls = None

        # =========================================
        # 处理分割
        # =========================================
        if self.is_seg:
            gt_seg = np.stack([cv2.resize(item["gt_seg"],(self.target_width,self.target_height),cv2.INTER_NEAREST)
                               for item in batch]) # seg
            gt_seg = torch.from_numpy(gt_seg)
        else:
            gt_seg=None

        # =========================================
        # 处理检测
        # =========================================
        if self.is_det:
            annots = [item["gt_det"] for item in batch] # det
            max_num_annots = max(annot.shape[0] for annot in annots)

            if max_num_annots > 0:

                annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

                for idx, annot in enumerate(annots):

                    # scale to target size
                    img_shape_dict = img_shape_list[idx]

                    org_width, org_height = img_shape_dict["width"], img_shape_dict["height"]
                    im_scale_x = int(self.target_width) / float(org_width)
                    im_scale_y = int(self.target_height) / float(org_height)
                    im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
                    annot[:, :4] *= im_scale

                    if annot.shape[0] > 0:
                        annot_padded[idx, :annot.shape[0], :] = torch.tensor(annot)
            else:
                annot_padded = torch.ones((len(annots), 1, 5)) * -1
        else:
            annot_padded = None

        # =========================================
        # 输入
        # =========================================
        output_dict = dict()
        output_dict["image"] = image_data
        output_dict["net_input_image_shape"] = np.stack([item["net_input_image_shape"] for item in batch])
        output_dict["src_image_shape"] = img_shape_list

        if self.is_lane:
            output_dict["annot_lane_path"] = np.stack([item["annot_lane_path"] for item in batch])
            output_dict["annot_lane"] = np.stack([item["annot_lane"] for item in batch])
            output_dict["gt_loc"] = gt_loc
            output_dict["gt_cls"] = gt_cls

        if self.is_seg:
            output_dict["gt_seg"]=gt_seg

        if self.is_det:
            output_dict["gt_det"]=annot_padded

        return output_dict

DEBUG = False
if __name__ == '__main__':
    # 输入测试参数
    CFG_PATH = "../cfgs/hydranet_joint_big_backbone.yml"
    MODE = "val"

    cfgs = yaml.safe_load(open(CFG_PATH))
    mt_data = MultitaskData(cfgs=cfgs,mode=MODE)

    trainloader = torch.utils.data.dataloader.DataLoader(mt_data,
                                                              batch_size=4,
                                                              num_workers=0,
                                                              shuffle=False,
                                                              drop_last=False,
                                                              pin_memory=True,
                                                              collate_fn=mt_data.collate_fn,
                                                              )

    one_data = iter(trainloader).__next__()

    for key,value in one_data.items():
        if not isinstance(value,list):
            if value is not None:
                print(key, value.shape)
            else:
                print(key,"None")
        else:
            print(key)
            for elem in value:
                print(elem)