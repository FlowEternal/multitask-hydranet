import yaml, time, shutil
import numpy as np
from dataset.dataloader import MultitaskData

import torch.distributed
import torch.utils.data.dataloader
import warnings
warnings.filterwarnings("ignore")

from model import HydraNet

#---------------------------------------------------#
#  validation packages
#---------------------------------------------------#
# segmentation
from head_seg.seg_metrics import IntersectionOverUnion

# detection
import os
import json
from pycocotools.coco import COCO
from head_detect.detect_eval import _eval
from head_detect.gen_val_json import gen_coco_label

# lane
from head_lane.lane_metric import LaneMetric

# others
import prettytable as pt


class HydraTrainer(object):
    def __init__(self, cfgs, cfg_path):
        self.cfgs = cfgs
        self.tag = self.cfgs["tag"]
        self.logs = self.cfgs["train"]["logs"]
        self.print_interval = self.cfgs["train"]["print_interval"]

        self.save_dir = os.path.join(self.logs, time.strftime('%d_%B_%Y_%H_%M_%S') + "_" + self.tag)
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

        self.cfg_path = cfg_path
        self.cfg_path_backup = os.path.join(self.save_dir,"config.yml")
        shutil.copy(self.cfg_path, self.cfg_path_backup)

        self.model_save_dir = os.path.join(self.save_dir,"model")
        if not os.path.exists(self.model_save_dir): os.makedirs(self.model_save_dir)


        self.train_detect = self.cfgs["train"]["train_detect"]
        self.train_seg = self.cfgs["train"]["train_seg"]
        self.train_lane = self.cfgs["train"]["train_lane"]

        #---------------------------------------------------#
        #  数据
        #---------------------------------------------------#
        self.batch_size_train = self.cfgs["train"]["batch_size_train"]
        self.num_worker_train = self.cfgs["train"]["num_worker_train"]

        self.batch_size_valid = self.cfgs["train"]["batch_size_valid"]
        self.num_worker_valid = self.cfgs["train"]["num_worker_valid"]

        self.net_input_width = self.cfgs["dataloader"]["network_input_width"]
        self.net_input_height = self.cfgs["dataloader"]["network_input_height"]

        self.train_data = MultitaskData(cfgs=cfgs, mode="train")
        self.trainloader = torch.utils.data.dataloader.DataLoader(self.train_data,
                                                             batch_size=self.batch_size_train,
                                                             num_workers=self.num_worker_train,
                                                             shuffle=True,
                                                             drop_last=False,
                                                             pin_memory=True,
                                                             collate_fn=self.train_data.collate_fn,
                                                             )

        self.valid_data = MultitaskData(cfgs=cfgs, mode="val")
        self.validloader = torch.utils.data.dataloader.DataLoader(self.valid_data,
                                                             batch_size=self.batch_size_valid,
                                                             num_workers=self.num_worker_valid,
                                                             shuffle=False,
                                                             drop_last=False,
                                                             pin_memory=True,
                                                             collate_fn=self.valid_data.collate_fn,
                                                             )

        
        #---------------------------------------------------#
        #  模型
        #---------------------------------------------------#
        self.hydranet = HydraNet(cfgs=cfgs).cuda()
        self.continue_train = self.cfgs["train"]["continue_train"]
        self.weight_file = self.cfgs["train"]["weight_file"]
        flag_joint = self.train_detect and self.train_seg and self.train_lane and (self.weight_file == "")

        if self.continue_train:
            def deparallel_model(dict_param):
                ck_dict_new = dict()
                for key, value in dict_param.items():
                    temp_list = key.split(".")[1:]
                    new_key = ""
                    for tmp in temp_list:
                        new_key += tmp + "."
                    ck_dict_new[new_key[0:-1]] = value
                return ck_dict_new

            if not flag_joint:
                dict_old = torch.load(self.weight_file)
                dict_new = deparallel_model(dict_old)
                self.hydranet.load_state_dict(dict_new)

            else:
                # 依次加载子任务网络覆盖参数 (这段代码其实不会常用，可能后面会去掉)
                # 最后一个加载的子任务，比如最后一个加载检测，那么主干网络的初始化参数就是检测的

                # 加载车道线
                dict_lane = deparallel_model(torch.load(self.cfgs["train"]["weight_file_lane"]))
                self.hydranet.load_state_dict(dict_lane,strict=False)

                # 加载语义分割
                dict_seg = deparallel_model(torch.load(self.cfgs["train"]["weight_file_seg"]))
                self.hydranet.load_state_dict(dict_seg,strict=False)

                # 加载目标检测
                # 这里把检测放在最后，因为检测最重要
                dict_det = deparallel_model(torch.load(self.cfgs["train"]["weight_file_det"]))
                self.hydranet.load_state_dict(dict_det,strict=False)

        # 并行训练开启与否
        self.use_distribute = self.cfgs["train"]["use_distribute"]
        if self.use_distribute:

            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='tcp://localhost:23457',
                                                 rank=0,
                                                 world_size=1)

            self.hydranet = torch.nn.parallel.DistributedDataParallel(self.hydranet,find_unused_parameters=True)

        #---------------------------------------------------#
        #  优化器
        #---------------------------------------------------#
        self.lr = self.cfgs["train"]["lr"]
        self.weight_decay = self.cfgs["train"]["weight_decay"]
        self.epoch = self.cfgs["train"]["epoch"]
        self.total_iters = len(self.trainloader) * self.epoch

        self.optimizer = torch.optim.Adam(self.hydranet.parameters(), self.lr, weight_decay= self.weight_decay)

        # iteration based
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_iters, eta_min=1e-8)

        #---------------------------------------------------#
        #  loss权重
        #---------------------------------------------------#
        # loss权重 seg
        self.segment_weight = self.cfgs["segment"]["segment_weight"]

        # loss权重 detection
        self.loss_cls_weight = self.cfgs["detection"]["loss_cls_weight"]
        self.loss_reg_weight = self.cfgs["detection"]["loss_reg_weight"]
        self.detection_weight = self.cfgs["detection"]["detection_weight"]

        # loss权重 lane
        self.loss_cls_pos_weight = self.cfgs["lane"]["loss_cls_pos_weight"]
        self.loss_cls_neg_weight = self.cfgs["lane"]["loss_cls_neg_weight"]
        self.loss_loc_weight = self.cfgs["lane"]["loss_loc_weight"]
        self.lane_weight = self.cfgs["lane"]["lane_weight"]

        #---------------------------------------------------#
        #  模型验证 -- 关键
        #---------------------------------------------------#
        if self.train_seg:
            self.segment_class_list = self.cfgs["segment"]["class_list"]
            self.metric_evaluator_iou = IntersectionOverUnion(n_classes=len(self.segment_class_list))

        if self.train_detect:
            self.MAX_IMAGES = self.cfgs["detection"]["max_images"]
            self.root_dir = self.cfgs["dataloader"]["data_list"].replace("/list", "")

            # 准备gt_detect_label.json
            self.eval_dir = os.path.join(self.root_dir, "eval_detect")
            if not os.path.exists(self.eval_dir): os.makedirs(self.eval_dir)
            self.val_gt_json = gen_coco_label(self.root_dir)  # 产生真值json
            self.coco_gt = COCO(self.val_gt_json)
            self.image_ids = self.coco_gt.getImgIds()[:self.MAX_IMAGES]

        if self.train_lane:
            self.lane_metric = LaneMetric(method="f1_measure", iou_thresh=0.5, lane_width=30, thresh_list=[0.5])
            [metric_handle.reset() for metric_handle in self.lane_metric.metric_handlers]


    def cal_total_loss(self, loss_dict):
        loss_total = 0.0
        if self.train_seg:
            loss_total += loss_dict["loss_seg"] * self.segment_weight
        if self.train_detect:
            loss_total += (loss_dict["loss_det_cls"] * self.loss_cls_weight + loss_dict["loss_det_reg"] * self.loss_reg_weight) * self.detection_weight
        if self.train_lane:
            loss_total += (loss_dict["loss_lane_cls_pos"] * self.loss_cls_pos_weight
                           + loss_dict["loss_lane_cls_neg"] * self.loss_cls_neg_weight
                           + loss_dict["loss_lane_loc"] * self.loss_loc_weight) * self.lane_weight

        return loss_total

    def print_loss_info(self, loss_dict, epoch, batch_idx, mode="train"):
        if mode == "train":
            print("TRAIN Epoch [%i|%i] Iter [%i|%i] Lr %.5f" % (epoch , self.epoch,
                                                                batch_idx , len(self.trainloader),
                                                                self.optimizer.param_groups[0]["lr"]))
        else:
            print("VALID Epoch [%i|%i] Iter [%i|%i] Lr %.5f" % (epoch , self.epoch,
                                                                batch_idx , len(self.validloader),
                                                                self.optimizer.param_groups[0]["lr"]))

        tb = pt.PrettyTable()
        row_list = list()
        key_list = list()
        for key, value in loss_dict.items():
            value_str = float("%.3f" %value.item() )
            row_list.append(value_str)
            key_list.append(key)

        tb.field_names = key_list
        tb.add_row(row_list)
        print(tb)
        print()

    def to_gpu(self,batch_data):
        batch_data["image"] = batch_data["image"].cuda().float()
        if self.train_lane:
            batch_data["gt_loc"] = batch_data["gt_loc"].cuda().float()
            batch_data["gt_cls"] = batch_data["gt_cls"].cuda().float()

        if self.train_seg:
            batch_data["gt_seg"] = batch_data["gt_seg"].cuda().float()

        if self.train_detect:
            batch_data["gt_det"] = batch_data["gt_det"].cuda().float()
        return batch_data

    def train_one_epoch(self, epoch):
        self.hydranet.train()
        for iter_idx, batch_data in enumerate(self.trainloader):

            # forward pass
            batch_data = self.to_gpu(batch_data)
            inputs = batch_data["image"]
            outputs = self.hydranet(inputs)
            if self.use_distribute:
                loss_dict = self.hydranet.module.cal_loss(outputs, batch_data)
            else:
                loss_dict = self.hydranet.cal_loss(outputs, batch_data)

            loss_total = self.cal_total_loss(loss_dict)
            loss_dict.update({"total_loss": loss_total})

            # 打印结果
            if iter_idx % self.print_interval ==0:
                self.print_loss_info(loss_dict, epoch, iter_idx,mode="train")

            self.optimizer.zero_grad()
            loss_total.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            self.optimizer.step()

            # 调整学习率
            self.scheduler.step()

        return

    def valid(self, epoch):
        self.hydranet.eval()
        detect_result = list()
        lane_result = list()
        for iter_idx , batch_data in enumerate(self.validloader):


            # forward pass
            batch_data = self.to_gpu(batch_data)
            inputs = batch_data["image"]
            batch_size_tmp = inputs.shape[0]
            outputs = self.hydranet(inputs)
            if self.use_distribute:
                loss_dict = self.hydranet.module.cal_loss(outputs, batch_data)
            else:
                loss_dict = self.hydranet.cal_loss(outputs, batch_data)

            loss_total = self.cal_total_loss(loss_dict)
            loss_dict.update({"total_loss": loss_total})

            # 打印结果
            self.print_loss_info(loss_dict, epoch, iter_idx, mode="valid")

            if self.train_seg:
                #---------------------------------------------------#
                #  分割部分metric
                #---------------------------------------------------#
                output_seg = outputs["seg"]
                gt_seg = batch_data["gt_seg"]

                for batch_idx in range(batch_size_tmp):
                    gt_seg_one = gt_seg[batch_idx].unsqueeze(0).detach().cpu()
                    predict_seg = output_seg[batch_idx].unsqueeze(0).detach().cpu()

                    seg_prediction = torch.argmax(predict_seg, dim=1)
                    self.metric_evaluator_iou.update(seg_prediction, gt_seg_one)

            if self.train_detect:
                #---------------------------------------------------#
                #  检测部分metric
                #---------------------------------------------------#
                regression = outputs["detection"]["regression"]
                classification = outputs["detection"]["classification"]
                anchors = outputs["detection"]["anchors"]
                src_image_shape = batch_data["src_image_shape"]

                if self.use_distribute:
                    preds = self.hydranet.module.detectheader.decode(inputs, regression, classification, anchors, conf_thres=0.3, iou_thres=0.3)
                else:
                    preds = self.hydranet.detectheader.decode(inputs, regression, classification, anchors, conf_thres=0.3, iou_thres=0.3)

                if not preds: continue

                # new_w, new_h, old_w, old_h, padding_w, padding_h,
                framed_metas = list()

                for idx in range(batch_size_tmp):
                    src_image_shape_ = src_image_shape[idx]
                    framed_meta = [self.net_input_width, self.net_input_height,src_image_shape_["width"], src_image_shape_["height"],  0, 0]
                    framed_metas.append(framed_meta)

                # predict inverse affine transform
                if self.use_distribute:
                    preds = self.hydranet.module.detectheader.invert_affine(framed_metas, preds)
                else:
                    preds = self.hydranet.detectheader.invert_affine(framed_metas, preds)

                for batch_inner_idx in range(batch_size_tmp):
                    scores = preds[batch_inner_idx]['scores']
                    class_ids = preds[batch_inner_idx]['class_ids']
                    rois = preds[batch_inner_idx]['rois']

                    image_id = iter_idx * self.batch_size_valid + batch_inner_idx + 1
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
                                'image_id': image_id,
                                'category_id': label + 1,
                                'score': float(score),
                                'bbox': box.tolist(),
                            }

                            detect_result.append(image_result)

            if self.train_lane:
                cls_preds = outputs["lane"]["predict_cls"]
                loc_preds = outputs["lane"]["predict_loc"]
                src_image_shape = batch_data["src_image_shape"]
                annot_lanes = batch_data["annot_lane_path"]
                for batch_idx in range(batch_size_tmp):
                    cls_pred = cls_preds[batch_idx]
                    loc_pred = loc_preds[batch_idx]
                    annot_lane = annot_lanes[batch_idx]
                    src_image_shape_ = src_image_shape[batch_idx]

                    if self.use_distribute:
                        lane_nms_set = self.hydranet.module.laneheader.decode(cls_pred, loc_pred, self.valid_data.lane_coder, 0.5, 100, False)
                        predict_json = self.hydranet.module.laneheader.scale_to_org(lane_nms_set,
                                                                             self.net_input_width,
                                                                             self.net_input_height,
                                                                             src_image_shape_["width"],
                                                                             src_image_shape_["height"])
                    else:
                        lane_nms_set = self.hydranet.laneheader.decode(cls_pred, loc_pred, self.valid_data.lane_coder, 0.5, 100, False)
                        predict_json = self.hydranet.laneheader.scale_to_org(lane_nms_set,
                                                                             self.net_input_width,
                                                                             self.net_input_height,
                                                                             src_image_shape_["width"],
                                                                             src_image_shape_["height"])

                    target_json = self.valid_data.parse_own_label(json.load(open(annot_lane)))

                    lane_result.append(dict(pr_result={**predict_json, **dict(Shape=src_image_shape_)},
                                        gt_result={**target_json, **dict(Shape=src_image_shape_)}))

                # lane metric
                self.lane_metric(output=lane_result)

        # ---------------------------------------------------#
        #  分割部分metric summary
        # ---------------------------------------------------#
        if self.train_seg:
            print("=========================== metric segmentation %i ===========================" % epoch)
            scores = self.metric_evaluator_iou.compute()
            mIOU = list()
            for key, value in zip(self.segment_class_list, scores):
                print(key + ", " + "%.3f" % value)
                mIOU.append(value)
            mIOU = np.array(mIOU).mean()
            print("mIOU" + ", " + "%.3f" % mIOU)

        # ---------------------------------------------------#
        #  检测部分metric summary
        # ---------------------------------------------------#
        if self.train_detect:
            print("=========================== metric detection %i ===========================" % epoch)
            if len(detect_result) !=0:
                # raise Exception('the model does not provide any valid output, check model architecture and the data input')

                # write output
                filepath = os.path.join(self.eval_dir, 'val_bbox_results.json')
                if os.path.exists(filepath):
                    os.remove(filepath)
                json.dump(detect_result, open(filepath, 'w'), indent=4)

                _eval(self.coco_gt, self.image_ids, filepath)

        # ---------------------------------------------------#
        #  车道线部分metric summary
        # ---------------------------------------------------#
        if self.train_lane:
            print("=========================== metric lane %i ===========================" % epoch)
            metric = self.lane_metric.summary()
            print(metric)
        print()

        torch.save(self.hydranet.state_dict(), os.path.join(self.model_save_dir,"epoch_%i.pth" % epoch))
        return


def main(cfg_path):
    cfgs = yaml.safe_load(open(cfg_path))
    trainer = HydraTrainer(cfgs, cfg_path)
    epoch_all = cfgs["train"]["epoch"]

    fine_tuning = cfgs["train"]["fine_tuning"]
    if fine_tuning:
        epoch_tunning = cfgs["train"]["epoch_tuning"]

        tuning_turn = cfgs["train"]["tuning_turn"]

        assert (3 * epoch_tunning * tuning_turn <= epoch_all)

        epoch_joint = int(epoch_all / tuning_turn) - epoch_tunning * 3

    else:
        epoch_joint = None
        epoch_tunning = None

    for epoch in range(epoch_all):
        # 是否单独tuning每一个分支头
        if fine_tuning:
            curr_turn = int(epoch / (epoch_joint + epoch_tunning * 3) )
            epoch_this_turn = epoch % (epoch_joint + epoch_tunning * 3)

            if epoch_this_turn < epoch_joint:
                print("======= TURN %i JOINT TRAINING =======" % curr_turn)
                if trainer.use_distribute:
                    trainer.optimizer.param_groups[0]['params'] = list(trainer.hydranet.module.parameters())
                else:
                    trainer.optimizer.param_groups[0]['params'] = list(trainer.hydranet.parameters())

                # trainer.optimizer = torch.optim.Adam(trainer.hydraNET.module.parameters(), trainer.lr, weight_decay=trainer.weight_decay)
                # trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, trainer.total_iters, eta_min=1e-8)

            # lane tuning
            elif epoch_joint <= epoch_this_turn < epoch_joint + epoch_tunning:
                print("======= TURN %i LANE TRAINING =======" % curr_turn)
                if trainer.use_distribute:
                    trainer.optimizer.param_groups[0]['params'] = list(trainer.hydranet.module.laneheader.parameters())
                else:
                    trainer.optimizer.param_groups[0]['params'] = list(trainer.hydranet.laneheader.parameters())

                # trainer.optimizer = torch.optim.Adam(trainer.hydraNET.module.segheader.parameters(), trainer.lr, weight_decay=trainer.weight_decay)
                # trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, trainer.total_iters, eta_min=1e-8)


            # det tuning
            elif epoch_joint + epoch_tunning <= epoch_this_turn < epoch_joint + 2 * epoch_tunning:
                print("======= TURN %i DET TRAINING =======" % curr_turn)
                if trainer.use_distribute:
                    trainer.optimizer.param_groups[0]['params'] = list(trainer.hydranet.module.detectheader.parameters())
                else:
                    trainer.optimizer.param_groups[0]['params'] = list(trainer.hydranet.detectheader.parameters())

                # trainer.optimizer = torch.optim.Adam(trainer.hydraNET.module.detectheader.parameters(), trainer.lr, weight_decay=trainer.weight_decay)
                # trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, trainer.total_iters, eta_min=1e-8)

            # seg tuning
            else:
                print("======= TURN %i SEG TRAINING =======" % curr_turn)
                if trainer.use_distribute:
                    trainer.optimizer.param_groups[0]['params'] = list(trainer.hydranet.module.segheader.parameters())
                else:
                    trainer.optimizer.param_groups[0]['params'] = list(trainer.hydranet.segheader.parameters())

                # trainer.optimizer = torch.optim.Adam(trainer.hydraNET.module.laneheader.parameters(), trainer.lr, weight_decay=trainer.weight_decay)
                # trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, trainer.total_iters, eta_min=1e-8)

        trainer.train_one_epoch(epoch)
        print("=========================== VALIDATION %i ===========================" %epoch)
        trainer.valid(epoch)

    print("============== finish training ==============")
    return


def gpu_set(gpu_begin, gpu_number):
    gpu_id_str = ""
    for i in range(gpu_begin, gpu_number + gpu_begin):
        if i != gpu_begin + gpu_number - 1:
            gpu_id_str = gpu_id_str + str(i) + ","
        else:
            gpu_id_str = gpu_id_str + str(i)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str


if __name__ == '__main__':
    gpu_set(0, 1)
    # cfg_path = "cfgs/hydranet_joint_small_backbone.yml"
    # cfg_path = "cfgs/hydranet_joint_big_backbone.yml"
    cfg_path = "cfgs/hydranet_joint_big_backbone_interview.yml"

    main(cfg_path)

    # 杀死所有进程号从2开始的进程
    # fuser -v /dev/nvidia* | grep 2* | xargs kill -9
