import torch.nn
import torch.nn.functional as F

from net.regnet import RegNetY
from net.bifpn import stackBiFPN

from head_seg.segmentation import SegmentHeader

from head_detect.detection import DetectionHeader

from head_lane.lanedetect import LaneHeader

#---------------------------------------------------#
#  loss function
#---------------------------------------------------#
# 分割
from head_seg.segmentation_loss import CrossEntropyLoss

# 检测
from head_detect.detection_loss import FocalLoss

# 车道线
from head_lane.lanedetect_loss import cal_loss_cls, cal_loss_regress


class HydraNet(torch.nn.Module):
    def __init__(self,cfgs,onnx_export = False):
        super(HydraNet, self).__init__()

        self.cfgs = cfgs
        self.onnx_export = onnx_export

        # 1.backbone
        self.net_input_width = self.cfgs["dataloader"]["network_input_width"]
        self.net_input_height = self.cfgs["dataloader"]["network_input_height"]

        # parameters setting
        self.initial_width = self.cfgs["backbone"]["initial_width"]
        self.slope = self.cfgs["backbone"]["slope"]
        self.quantized_param = self.cfgs["backbone"]["quantized_param"]
        self.network_depth = self.cfgs["backbone"]["network_depth"]
        self.bottleneck_ratio = self.cfgs["backbone"]["bottleneck_ratio"]
        self.group_width = self.cfgs["backbone"]["group_width"]
        self.stride = self.cfgs["backbone"]["stride"]
        self.se_ratio = self.cfgs["backbone"]["se_ratio"]

        # create backbone
        self.backbone = RegNetY(self.initial_width,
                               self.slope,
                               self.quantized_param,
                               self.network_depth,
                               self.bottleneck_ratio,
                               self.group_width,
                               self.stride,
                               self.se_ratio)

        # 2.create fusion neck
        self.fpn_num_filters = self.cfgs["backbone"]["fpn_num_filters"]
        self.fpn_cell_repeats = self.cfgs["backbone"]["fpn_cell_repeats"]
        self.conv_channel_coef = self.cfgs["backbone"]["conv_channel_coef"]  # the channels of P3/P4/P5

        self.neck = stackBiFPN(fpn_num_filters=self.fpn_num_filters,
                          fpn_cell_repeats=self.fpn_cell_repeats,
                          conv_channel_coef=self.conv_channel_coef,
                          onnx_export=self.onnx_export)

        # 加载检测相关
        self.train_detect = self.cfgs["train"]["train_detect"]
        if self.train_detect:
            self.num_classes = self.cfgs["detection"]["num_classes"]
            self.fpn_num_filters_detect = self.cfgs["detection"]["fpn_num_filters_detect"]
            self.aspect_ratios_factor = self.cfgs["detection"]["aspect_ratios_factor"]
            self.scales_factor = self.cfgs["detection"]["scales_factor"]
            self.box_class_repeats = self.cfgs["detection"]["box_class_repeats"]
            self.pyramid_levels = self.cfgs["detection"]["pyramid_levels"]
            self.anchor_scale = self.cfgs["detection"]["anchor_scale"]

            # 导出参数
            ratio_one = self.aspect_ratios_factor[0]
            ratio_two = self.aspect_ratios_factor[1]
            self.aspect_ratios = [(1.0, 1.0), (ratio_one, ratio_two), (ratio_two, ratio_one)]
            self.scales = [2 ** self.scales_factor[0], 2 ** self.scales_factor[1], 2 ** self.scales_factor[2]]

            self.detectheader = DetectionHeader(num_classes=self.num_classes,
                                            fpn_num_filters_detect=self.fpn_num_filters_detect,
                                            aspect_ratios=self.aspect_ratios,
                                            scales=self.scales,
                                            box_class_repeats=self.box_class_repeats,
                                            pyramid_levels=self.pyramid_levels,
                                            anchor_scale=self.anchor_scale,
                                            onnx_export=self.onnx_export
                                            )

            self.loss_detect = FocalLoss()

        else:
            self.detectheader = None
            self.loss_detect = None

        # 加载分割相关
        self.train_seg = self.cfgs["train"]["train_seg"]
        if self.train_seg:
            self.segment_class_list = self.cfgs["segment"]["class_list"]

            self.channel_dimension_seg_encode = self.cfgs["segment"]["channel_dimension_seg_encode"]
            self.channel_dimension_seg_decode = self.cfgs["segment"]["channel_dimension_seg_decode"]

            self.segheader = SegmentHeader(num_ch_enc=self.channel_dimension_seg_encode,
                                      num_ch_dec=self.channel_dimension_seg_decode,
                                      num_output_channels=len(self.segment_class_list))

            self.use_lovasz = self.cfgs["segment"]["use_lovasz"]
            self.weight_seg = self.cfgs["segment"]["class_weight"]
            self.use_top_k = self.cfgs["segment"]["use_top_k"]
            self.top_k_ratio = self.cfgs["segment"]["top_k_ratio"]
            self.use_focal = self.cfgs["segment"]["use_focal"]

            if not self.use_lovasz:
                self.loss_seg = CrossEntropyLoss(
                    class_weights=torch.tensor(self.weight_seg),
                    use_top_k=self.use_top_k,
                    top_k_ratio=self.top_k_ratio,
                    use_focal=self.use_focal).cuda()
            else:
                from head_seg.loss_lovasz import lovasz_softmax
                self.loss_seg = lovasz_softmax

        else:
            self.segheader = None
            self.loss_seg = None

        # 加载车道线相关
        self.train_lane = self.cfgs["train"]["train_lane"]
        if self.train_lane:
            self.base_channel = self.cfgs["lane"]["base_channel"]
            self.num_classes = self.cfgs["lane"]["num_classes"]
            self.stride = self.cfgs["lane"]["anchor_stride"]
            self.interval = self.cfgs["lane"]["interval"]

            self.laneheader = LaneHeader(base_channel=self.base_channel,
                                    num_classes=self.num_classes,
                                    stride=self.stride,
                                    input_width=self.net_input_width,
                                    input_height=self.net_input_height,
                                    interval=self.interval)

            self.loss_cls = cal_loss_cls
            self.loss_reg = cal_loss_regress


        else:
            self.laneheader = None
            self.loss_cls = None
            self.loss_reg = None


        return

    def forward(self, x, mode = "train"):
        # feature abstraction
        feats = self.backbone(x)  # backbone
        fused_feats = self.neck(feats)  # neck

        output_dict = {}
        # segment
        if self.train_seg:
            feats_seg = list()
            feats_seg.append(feats[0])
            feats_seg.append(fused_feats[0])
            feats_seg.append(fused_feats[1])
            feats_seg.append(fused_feats[2])
            output_seg = self.segheader(feats_seg)
            output_dict.update({"seg":output_seg})
        else:
            output_seg = None

        # detection
        if self.train_detect:
            anchors, regression, classification = self.detectheader(x, fused_feats)
            detect_dict = {"anchors":anchors, "regression":regression, "classification":classification}
            output_dict.update({"detection":detect_dict})
        else:
            anchors, regression, classification = None, None, None
        # lane
        if self.train_lane:
            lane_result = self.laneheader(fused_feats)
            output_dict.update({"lane":lane_result})
            lane_cls = lane_result["predict_cls"]
            lane_reg = lane_result["predict_loc"]
        else:
            lane_cls ,lane_reg = None, None

        if mode !="deploy":
            return output_dict
        else:
            # joint 模型
            output_seg_ = torch.argmax(output_seg, dim=1)
            return output_seg_, anchors, regression, classification, lane_cls, lane_reg

    # 关键函数 -- loss计算
    def cal_loss(self, pred_dict, gt_dict):
        loss_dict = {}
        if self.train_seg:
            gt_seg = gt_dict["gt_seg"]
            output_seg = pred_dict["seg"]

            if not self.use_lovasz:
                loss_seg = self.loss_seg(output_seg, gt_seg.long())
            else:
                loss_seg = self.loss_seg(F.softmax(output_seg, dim=1), gt_seg.long(), ignore=255)

            if loss_seg == 0 or not torch.isfinite(loss_seg):
                print("cal segment loss diverge!")
                exit()

            loss_dict.update({"loss_seg":loss_seg})

        if self.train_detect:
            annotations = gt_dict["gt_det"]
            classification = pred_dict["detection"]["classification"]
            regression = pred_dict["detection"]["regression"]
            anchors = pred_dict["detection"]["anchors"]
            cls_loss, reg_loss = self.loss_detect(classification, regression, anchors, annotations)

            cls_loss = cls_loss.mean()
            if not torch.isfinite(cls_loss):
                print("cal det cls loss diverge!")
                exit()

            reg_loss = reg_loss.mean()
            if not torch.isfinite(reg_loss):
                print("cal det reg loss diverge!")
                exit()


            loss_dict.update({"loss_det_cls":cls_loss})
            loss_dict.update({"loss_det_reg":reg_loss})

        if self.train_lane:
            cls_preds = pred_dict["lane"]['predict_cls']
            loc_preds = pred_dict["lane"]['predict_loc']

            cls_targets = gt_dict["gt_cls"]
            loc_targets = gt_dict["gt_loc"]
            total_cross_pos, total_cross_neg, pmask, positive_num = self.loss_cls(cls_targets, cls_preds)
            total_loc = self.loss_reg(pmask, positive_num, loc_targets, loc_preds)

            if total_cross_pos == 0 or not torch.isfinite(total_cross_pos):
                print("cal lane pos loss diverge!")
                exit()

            if total_cross_neg == 0 or not torch.isfinite(total_cross_neg):
                print("cal lane neg loss diverge!")
                exit()

            if total_loc == 0 or not torch.isfinite(total_loc):
                print("cal lane loc loss diverge!")
                exit()

            loss_dict.update({"loss_lane_cls_pos":total_cross_pos})
            loss_dict.update({"loss_lane_cls_neg":total_cross_neg})
            loss_dict.update({"loss_lane_loc":total_loc})

        return loss_dict

if __name__ == '__main__':
    import yaml
    CFG_PATH = "cfgs/hydranet_joint_big_backbone.yml"
    # CFG_PATH = "cfgs/hydranet_joint_small_backbone.yml"

    cfgs = yaml.safe_load(open(CFG_PATH))
    hydranet = HydraNet(cfgs=cfgs).cuda()

    # inference
    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    batch_size = 1
    dummy_input = torch.randn((batch_size, 3, net_input_height, net_input_width)).cuda()

    import time
    while True:
        tic = time.time()
        ouptut = hydranet(dummy_input)
        torch.cuda.synchronize()
        print("inference time: %i" %(1000*(time.time() - tic)))
