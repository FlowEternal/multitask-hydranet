import numpy as np
import itertools
import torch
import torch.nn as nn
from typing import Union
from net.common import SeparableConvBlock, MemoryEfficientSwish, Swish
from head_detect.detection_loss import BBoxTransform, ClipBoxes, postprocess
from head_detect.display import display_
from dataset.utility import imagenet_denormalize

class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers, pyramid_levels=5, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for _ in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for _ in range(num_layers)]) for _ in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats


class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, pyramid_levels=5, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for _ in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for _ in range(num_layers)]) for _ in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()
        return feats


class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    """

    def __init__(self,
                 anchor_scale, pyramid_levels, scales, ratio):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        self.strides = [2 ** x for x in self.pyramid_levels]
        self.scales = scales
        self.ratios = ratio

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        """Generates multiscale anchor boxes.

        Args:
          image: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          dtype: data type for anchors.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """

        image_shape = image.shape[2:]

        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0 or image_shape[0] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')

                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))

            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        # save it for later use to reduce overhead
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes


class DetectionHeader(nn.Module):
    def __init__(self,
                 num_classes,
                 fpn_num_filters_detect,
                 aspect_ratios,
                 scales,
                 box_class_repeats,
                 pyramid_levels,
                 anchor_scale,
                 onnx_export=False):
        super(DetectionHeader, self).__init__()
        self.num_classes = num_classes
        self.fpn_num_filters_detect = fpn_num_filters_detect
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.num_anchors = len(aspect_ratios) * len(self.scales)
        self.box_class_repeats = box_class_repeats
        self.pyramid_levels = pyramid_levels
        self.anchor_scale = anchor_scale

        self.regressor = Regressor(in_channels=self.fpn_num_filters_detect,
                              num_anchors=self.num_anchors,
                              num_layers=self.box_class_repeats,
                              pyramid_levels=self.pyramid_levels,
                              onnx_export=onnx_export)

        self.classifier = Classifier(in_channels=self.fpn_num_filters_detect,
                                num_anchors=self.num_anchors,
                                num_classes=self.num_classes,
                                num_layers=self.box_class_repeats,
                                pyramid_levels=self.pyramid_levels,
                                onnx_export=onnx_export)

        self.anchors = Anchors(anchor_scale=self.anchor_scale,
                               pyramid_levels=(torch.arange(self.pyramid_levels) + 3).tolist(),
                               scales=scales, ratio=aspect_ratios)


    def forward(self, x, fused_feats):
        anchors = self.anchors(x, x.dtype)
        regression = self.regressor(fused_feats)
        classification = self.classifier(fused_feats)
        return anchors, regression, classification

    @staticmethod
    def invert_affine(metas: Union[float, list, tuple], preds):
        for i in range(len(preds)):
            if len(preds[i]['rois']) == 0:
                continue
            else:
                if metas is float:
                    preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                    preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
                else:
                    new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                    preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                    preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
        return preds

    @staticmethod
    def decode(imgs, regressions, classifications, anchors, conf_thres=0.6, iou_thres=0.3):
        if imgs is not None:
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(imgs.detach(),
                              torch.stack([anchors[0]] * imgs.shape[0], 0).detach(),
                              regressions.detach(),
                              classifications.detach(),
                              regressBoxes, clipBoxes,
                              conf_thres, iou_thres)

            return out

    @staticmethod
    def display(decode, imgs, obj_list, org_size, target_size):
        imgs_ = display_(decode, imgs, obj_list, org_size, target_size)
        if imgs_ is None:
            return imgs
        return imgs_

if __name__ == '__main__':
    # =========================================
    # 参数加载
    # =========================================
    import yaml
    CFG_PATH = "../cfgs/hydranet_joint_small_backbone.yml"
    cfgs = yaml.safe_load(open(CFG_PATH))

    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    batch_size = 2

    # 1.create input
    dummy_input = torch.randn((batch_size, 3, net_input_height, net_input_width)).to("cuda:0")

    # 2.backbone regular network
    from net.regnet import RegNetY

    # parameters setting
    initial_width = cfgs["backbone"]["initial_width"]
    slope = cfgs["backbone"]["slope"]
    quantized_param = cfgs["backbone"]["quantized_param"]
    network_depth = cfgs["backbone"]["network_depth"]
    bottleneck_ratio = cfgs["backbone"]["bottleneck_ratio"]
    group_width = cfgs["backbone"]["group_width"]
    stride = cfgs["backbone"]["stride"]
    se_ratio = cfgs["backbone"]["se_ratio"]

    # create backbone
    backbone = RegNetY(initial_width,
                        slope,
                        quantized_param,
                        network_depth,
                        bottleneck_ratio,
                        group_width,
                        stride,
                        se_ratio).to("cuda:0")

    # 3.create fusion neck
    from net.bifpn import stackBiFPN

    fpn_num_filters = cfgs["backbone"]["fpn_num_filters"]
    fpn_cell_repeats = cfgs["backbone"]["fpn_cell_repeats"]
    conv_channel_coef = cfgs["backbone"]["conv_channel_coef"]  # the channels of P3/P4/P5

    neck = stackBiFPN(fpn_num_filters=fpn_num_filters,
                      fpn_cell_repeats=fpn_cell_repeats,
                      conv_channel_coef=conv_channel_coef).to("cuda:0")

    # =========================================
    # detection branch
    # =========================================
    # 4.create detection header
    num_classes = cfgs["detection"]["num_classes"]
    fpn_num_filters_detect = cfgs["detection"]["fpn_num_filters_detect"]
    aspect_ratios_factor = cfgs["detection"]["aspect_ratios_factor"]
    scales_factor = cfgs["detection"]["scales_factor"]
    box_class_repeats = cfgs["detection"]["box_class_repeats"]
    pyramid_levels = cfgs["detection"]["pyramid_levels"]
    anchor_scale = cfgs["detection"]["anchor_scale"]

    # 导出参数
    ratio_one = aspect_ratios_factor[0]
    ratio_two = aspect_ratios_factor[1]
    aspect_ratios = [(1.0, 1.0),(ratio_one,ratio_two),(ratio_two,ratio_one)]
    scales = [2 ** scales_factor[0], 2 ** scales_factor[1], 2 ** scales_factor[2]]

    header_detect = DetectionHeader(num_classes = num_classes,
                                    fpn_num_filters_detect=fpn_num_filters_detect,
                                    aspect_ratios=aspect_ratios,
                                    scales=scales,
                                    box_class_repeats=box_class_repeats,
                                    pyramid_levels=pyramid_levels,
                                    anchor_scale=anchor_scale).to("cuda:0")

    # inference test
    feats = backbone(dummy_input)   # backbone
    fused_feats = neck(feats)       # neck
    anchors, regression, classification = header_detect(dummy_input, fused_feats)
    print(anchors.shape)
    print(regression.shape)
    print(classification.shape)

    # =========================================
    # training
    # =========================================
    from head_detect.detection_loss import FocalLoss
    criterion = FocalLoss().to("cuda:0")

    # optimizer defination
    train_param = list()
    train_param += backbone.parameters()
    train_param += neck.parameters()
    train_param += header_detect.parameters()
    optimizer = torch.optim.Adam(train_param, 0.01, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100,eta_min=1e-8)

    annotations = torch.ones([batch_size,16,5], dtype= torch.float32).to("cuda:0")
    cls_loss, reg_loss = criterion(classification, regression, anchors, annotations)

    cls_loss = cls_loss.mean()
    reg_loss = reg_loss.mean()
    loss = cls_loss + reg_loss

    if loss == 0 or not torch.isfinite(loss):exit()

    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()

    # =========================================
    # inference + visual
    # =========================================
    org_size = (1920, 1080)
    target_size = (net_input_width, net_input_height)
    obj_list = cfgs["detection"]["class_list"]

    # display
    imgs = imagenet_denormalize(dummy_input)
    preds = header_detect.decode(dummy_input,
                                 regression,
                                 classification,anchors)

    imgs_vis = [np.ones([org_size[1], org_size[0],3],dtype=np.uint8) for _ in range(batch_size)]

    imgs_vis = header_detect.display(preds, imgs_vis, obj_list, org_size, target_size)

    # =========================================
    # 验证mAP
    # =========================================
    import os
    import json
    from pycocotools.coco import COCO
    from head_detect.detect_eval import _eval
    from head_detect.gen_val_json import gen_coco_label

    MAX_IMAGES = cfgs["detection"]["max_images"]
    root_dir = cfgs["dataloader"]["data_list"].replace("list","")

    # 准备gt_detect_label.json
    eval_dir = os.path.join(root_dir, "eval_detect")
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)
    val_gt_json = gen_coco_label(root_dir) # 产生真值json
    coco_gt = COCO(val_gt_json)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    results = list() # 检测分支
    for idx in image_ids:
        feats = backbone(dummy_input[0:1])  # backbone
        fused_feats = neck(feats)  # neck

        # =========================================
        # 检测分支
        # =========================================
        anchors, regression, classification = header_detect(dummy_input, fused_feats)
        preds = header_detect.decode(dummy_input, regression, classification,anchors)

        if not preds:
            continue

        # new_w, new_h, old_w, old_h, padding_w, padding_h,
        framed_meta = [org_size[0], org_size[1], target_size[0], target_size[1], 0, 0]
        batch_size = dummy_input.shape[0]
        framed_metas = [framed_meta for _ in range(batch_size)]

        # predict inverse affine transform
        preds = header_detect.invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': idx,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)


    # =========================================
    # 循环完后检测分支
    # =========================================
    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = os.path.join(eval_dir,'val_bbox_results.json')
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

    _eval(coco_gt, image_ids, filepath)


