""" lane detector header"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from head_lane.lane_codec import LaneCodec
from head_lane.lane_codec_utils import nms_with_pos,order_lane_x_axis,convert_lane_to_dict
import cv2
import numpy as np

def upsample(x, factor = 2):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=factor, mode="nearest")

class LaneHeader(nn.Module):
    """CurveLaneHead."""

    def __init__(self,
                 base_channel,
                 num_classes,
                 stride,
                 input_width,
                 input_height,
                 interval):
        super(LaneHeader, self).__init__()

        # feature dimension
        self.base_channel = base_channel
        self.num_classes = num_classes
        self.stride = stride
        self.input_width = input_width
        self.input_height = input_height
        self.interval = interval
        self.feat_width = int(self.input_width / self.stride)
        self.feat_height = int(self.input_height / self.stride)
        self.points_per_line = int(self.input_height / self.interval)
        self.lane_up_pts_num = self.points_per_line + 1
        self.lane_down_pts_num = self.points_per_line + 1

        # construct fusion layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.upsample = upsample

        # construct lane header
        self.conv_cls_conv = nn.Sequential(
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.base_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channel, num_classes, kernel_size=1, stride=1)
        )

        self.conv_up_conv = nn.Sequential(
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.base_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channel, self.lane_up_pts_num, kernel_size=1, stride=1)
        )

        self.conv_down_conv = nn.Sequential(
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.base_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channel, self.lane_down_pts_num, kernel_size=1, stride=1)
        )

    def forward(self, fused_x):
        """Forward method of this head."""
        # multi-scale feature fusion for stride 16
        # 80, 40 ,20 ,10
        if self.stride ==16:
            fused_80_down = self.maxpool(fused_x[0])
            fused_20_up = self.upsample(fused_x[2], factor = 2)
            fused_10_up = self.upsample(fused_x[3], factor = 4)
            fused_lane = torch.cat([fused_80_down, fused_20_up, fused_x[1], fused_10_up], dim=1)

        elif self.stride ==32:
            fused_80_down = self.maxpool(self.maxpool(fused_x[0]))
            fused_40_down = self.maxpool(fused_x[1])
            fused_20_up = self.upsample(fused_x[3], factor = 2)
            fused_lane = torch.cat([fused_80_down, fused_40_down, fused_x[2], fused_20_up],dim=1)

        else:
            fused_lane = None

        # prediction
        predict_cls = self.conv_cls_conv(fused_lane).permute((0, 2, 3, 1)).contiguous()
        predict_cls = predict_cls.view(predict_cls.shape[0], -1, self.num_classes)

        predict_up = self.conv_up_conv(fused_lane).permute((0, 2, 3, 1))
        predict_down = self.conv_down_conv(fused_lane).permute((0, 2, 3, 1))
        predict_loc = torch.cat([predict_down, predict_up], -1).contiguous()
        predict_loc = predict_loc.view(predict_loc.shape[0], -1, self.lane_up_pts_num + self.lane_down_pts_num)

        result = dict(predict_cls=predict_cls,predict_loc=predict_loc)

        return result

    @property
    def input_shape(self):
        """Output of backbone."""
        return self.feat_height, self.feat_width

    @staticmethod
    def decode(predict_cls,predict_loc, pointlane, conf_thres = 0.5, nms_line_thres = 100, use_mean = False):
        regression = predict_loc
        classfication = F.softmax(predict_cls, -1)

        lane_set = pointlane.decode_lane(predict_type=classfication,
                                         predict_loc=regression,
                                         exist_threshold=conf_thres,
                                         )

        lane_nms_set = nms_with_pos(lane_set,
                                    thresh=nms_line_thres,
                                    use_mean_dist=use_mean)
        return lane_nms_set

    @staticmethod
    def scale_to_org(lane_nms_set,net_input_width,net_input_height,org_width,org_height):
        lane_order_set = order_lane_x_axis(list(lane_nms_set), net_input_height)
        scale_x = org_width/ net_input_width
        scale_y = org_height / net_input_height
        predict_json = convert_lane_to_dict(lane_order_set, scale_x, scale_y)
        return predict_json

    @staticmethod
    def visual(imgs, predict_jsons,org_width = 1920, min_length = 2, filter_vertical = True, filter_thres = 65):
        batch_size = len(imgs)
        vis_list = list()
        for batch_idx in range(batch_size):
            im_vis = imgs[batch_idx]
            predict_json = predict_jsons[batch_idx]
            for tmp_line in predict_json:
                score = tmp_line["score"]
                pts = tmp_line["points"]
                lane_type = "Lane"
                length = len(pts)
                if length < min_length:
                    continue

                if filter_vertical:
                    # 过滤掉太竖直的线 TODO
                    pt_list_ = np.array( [ [int(pt["x"]), int(pt["y"])] for pt in pts] )
                    coeff = np.polyfit(pt_list_[:,0], pt_list_[:,1], 1)
                    theta = abs(np.arctan(coeff[0])) / 3.1415 * 180
                    if theta > filter_thres:
                        continue

                for idx, pt in enumerate(pts):
                    point = (int(pt["x"]), int(pt["y"]))

                    if idx + 1 == length:
                        break
                    else:
                        points_after = (int(pts[idx + 1]["x"]), int(pts[idx + 1]["y"]))

                    im_vis = cv2.line(im_vis,
                                      tuple(point),
                                      tuple(points_after),
                                      color=(255, 255, 0),
                                      thickness=15)

                # 置信度文本显示
                pt_ = pts[min_length-1]
                pt_txt = [int(pt_["x"]), int(pt_["y"])]
                if pt_txt[0] < 0:
                    pt_txt[0] = 30

                if pt_txt[0] > org_width:
                    pt_txt[0] = org_width  - 300
                    pt_txt[1] = pt_txt[1] - 60

                cv2.putText(im_vis, "%s: %.2f" % (lane_type, float(score)),
                            (pt_txt[0], pt_txt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0, (255, 255, 0), 7)
            vis_list.append(im_vis)
        return vis_list

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
    # lane detect branch
    # =========================================
    base_channel = cfgs["lane"]["base_channel"]
    num_classes = cfgs["lane"]["num_classes"]
    stride = cfgs["lane"]["anchor_stride"]
    interval = cfgs["lane"]["interval"]

    laneheader = LaneHeader(base_channel=base_channel,
                            num_classes=num_classes,
                            stride=stride,
                            input_width=net_input_width,
                            input_height=net_input_height,
                            interval=interval).to("cuda:0")

    # inference
    feats = backbone(dummy_input)   # backbone
    fused_feats = neck(feats)       # neck
    result = laneheader(fused_feats)
    print(result["predict_cls"].shape)
    print(result["predict_loc"].shape)

    # =========================================
    # training
    # =========================================
    from head_lane.lanedetect_loss import cal_loss_cls, cal_loss_regress

    # optimizer defination
    train_param = list()
    train_param += backbone.parameters()
    train_param += neck.parameters()
    train_param += laneheader.parameters()
    optimizer = torch.optim.Adam(train_param, 0.01, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100,eta_min=1e-8)

    # cal loss
    cls_preds = result['predict_cls']
    loc_preds = result['predict_loc']

    cls_targets = torch.ones_like(cls_preds)
    cls_targets[:,0:40, 1] = 0
    loc_targets = torch.ones_like(loc_preds)

    total_cross_pos, total_cross_neg, pmask, positive_num = cal_loss_cls(cls_targets, cls_preds)
    total_loc = cal_loss_regress(pmask, positive_num, loc_targets, loc_preds)

    loss_lane = total_cross_neg + total_cross_pos + total_loc

    if loss_lane == 0 or not torch.isfinite(loss_lane):
        exit()

    optimizer.zero_grad()
    loss_lane.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()

    # =========================================
    # point lane coder
    # =========================================
    # 加载车道线相关
    train_lane = cfgs["train"]["train_lane"]
    anchor_stride = cfgs["lane"]["anchor_stride"]
    interval = cfgs["lane"]["interval"]
    anchor_lane_num = cfgs["lane"]["anchor_lane_num"]
    interpolate = cfgs["lane"]["interpolate"]  # 是否插值 如果插值 即为沿长到版边 和原代码就完全一致
    scale_invariance = cfgs["lane"]["scale_invariance"]
    points_per_line = int(net_input_height / interval)

    lane_coder = LaneCodec( input_width=net_input_width,
                            input_height=net_input_height,
                            anchor_stride=anchor_stride,
                            points_per_line=points_per_line,
                            do_interpolate=interpolate,
                            anchor_lane_num=anchor_lane_num,
                            scale_invariance=scale_invariance )

    # =========================================
    # inference + visual
    # =========================================
    import numpy as np
    org_size = (1920,1080)
    conf_thres = cfgs["lane"]["conf_thres"]
    nms_thres = cfgs["lane"]["nms_thres"]

    imgs_vis = [np.ones([org_size[1], org_size[0],3],dtype=np.uint8) for _ in range(batch_size)]
    cls_preds = result['predict_cls']
    loc_preds = result['predict_loc']
    batch_size = len(imgs_vis)

    predict_jsons = list()
    for batch_idx in range(batch_size):
        cls_pred = cls_preds[batch_idx]
        loc_pred = loc_preds[batch_idx]
        lane_nms_set = laneheader.decode(cls_pred,loc_pred,lane_coder,conf_thres,nms_thres,False)
        predict_json = laneheader.scale_to_org(lane_nms_set,net_input_width,net_input_height,org_size[1],org_size[0])["Lines"]
        predict_jsons.append(predict_json)

    imgs_vis = laneheader.visual(imgs_vis, predict_jsons,org_size[1])

    # =========================================
    # 验证
    # =========================================
    import json
    from head_lane.lane_metric import LaneMetric

    annot_lane = "/data/zdx/Data/MULTITASK_8/labels_lane/0a0ae73285e382312041a4bc06fc5e05.json"
    src_image_shape = dict(width=org_size[0], height=org_size[1], channel=3)

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

    results = []
    lane_metric = LaneMetric(method="f1_measure",iou_thresh=0.5,lane_width=30,thresh_list=[0.5])
    # reset
    [metric_handle.reset() for metric_handle in lane_metric.metric_handlers]
    for _ in range(4):
        for batch_idx in range(batch_size):
            cls_pred = cls_preds[batch_idx]
            loc_pred = loc_preds[batch_idx]
            lane_nms_set = laneheader.decode(cls_pred, loc_pred, lane_coder, 0.5, 100, False)
            predict_json = laneheader.scale_to_org(lane_nms_set, net_input_width, net_input_height, org_size[0], org_size[1])

            target_json = parse_own_label(json.load(open(annot_lane)))

            results.append(dict(pr_result={**predict_json, **dict(Shape=src_image_shape)},
                                gt_result={**target_json, **dict(Shape=src_image_shape)}))

        # lane metric
        lane_metric(output=results)

    metric = lane_metric.summary()
    print(metric)