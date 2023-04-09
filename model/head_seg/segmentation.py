import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class SegmentHeader(nn.Module):
    def __init__(self,
                 num_ch_enc,
                 num_ch_dec=None,
                 num_output_channels=10,
                 use_skips=True,
                 ):
        super(SegmentHeader, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec

        # decoder
        self.convs = OrderedDict()
        for i in range(len(num_ch_enc)-1, -1, -1):
            
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == len(num_ch_enc)-1 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.convs[("output")] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)

        self.decoder = nn.ModuleList(self.convs.values())

    def forward(self, input_features):
        # output_guide = list()
        # decoder
        x = input_features[-1]
        for i in range(0,len(input_features)):
            
            # up 1
            x = self.decoder[2*i](x)
            x = [upsample(x)]
            if self.use_skips and i < len(input_features)-1:
                x += [input_features[ len(input_features)-2 - i]]
            x = torch.cat(x, 1)
            
            # up 2
            x = self.decoder[2*i + 1](x)

            # for semantic guide
            # output_guide.append(x)

        output_seg = self.decoder[-1](upsample(x))

        return output_seg

    @staticmethod
    def decode(imgs, masks, org_size, vis_color_id):
        seg_predictions = torch.argmax(masks, dim=1).detach().cpu().numpy()

        batch_size = len(imgs)
        visual_imgs = list()
        for batch_idx in range(batch_size):
            seg_prediction = seg_predictions[batch_idx]
            im_vis = imgs[batch_idx]

            # vis
            vis_seg = np.zeros([seg_prediction.shape[0], seg_prediction.shape[1], 3], dtype=np.uint8)
            for cls_id, color in vis_color_id.items():
                vis_seg[seg_prediction == cls_id] = color
            vis_seg = cv2.resize(vis_seg, org_size, cv2.INTER_NEAREST)
            im_vis = cv2.addWeighted(im_vis, 0.8, vis_seg, 0.5, 0.0)
            visual_imgs.append(im_vis)

        return visual_imgs

if __name__ == '__main__':
    # =========================================
    # 参数加载
    # =========================================
    import yaml
    CFG_PATH = "../cfgs/hydranet_joint_small_backbone.yml"
    cfgs = yaml.safe_load(open(CFG_PATH))

    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    batch_size = cfgs["train"]["batch_size_train"]

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
    # segmentation branch
    # =========================================
    # 4.create segmentation head
    segment_class_list = cfgs["segment"]["class_list"]
    channel_dimension_seg_encode = cfgs["segment"]["channel_dimension_seg_encode"]
    channel_dimension_seg_decode = cfgs["segment"]["channel_dimension_seg_decode"]

    segheader = SegmentHeader(num_ch_enc=channel_dimension_seg_encode,
                                 num_ch_dec=channel_dimension_seg_decode,
                                 num_output_channels=len(segment_class_list)).to("cuda:0")

    # 5.inference test
    feats = backbone(dummy_input)   # backbone
    fused_feats = neck(feats)       # neck
    feats_seg = list()
    feats_seg.append(feats[0])
    feats_seg.append(fused_feats[0])
    feats_seg.append(fused_feats[1])
    feats_seg.append(fused_feats[2])
    output_seg = segheader(feats_seg)
    print(output_seg.shape)

    # 6.train
    from head_seg.segmentation_loss import CrossEntropyLoss

    weight_seg = cfgs["segment"]["class_weight"]
    use_top_k = cfgs["segment"]["use_top_k"]
    top_k_ratio = cfgs["segment"]["top_k_ratio"]
    use_focal = cfgs["segment"]["use_focal"]

    # loss function
    from head_seg.loss_lovasz import lovasz_softmax
    weight = torch.tensor(weight_seg)
    # loss_seg_fun = CrossEntropyLoss(
    #     class_weights=weight,
    #     use_top_k=use_top_k,
    #     top_k_ratio=top_k_ratio,
    #     use_focal=use_focal).to("cuda:0")

    loss_seg_fun = lovasz_softmax

    # optimizer defination
    train_param = list()
    train_param += backbone.parameters()
    train_param += neck.parameters()
    train_param += segheader.parameters()
    optimizer = torch.optim.Adam(train_param, 0.001, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-8)

    gt_seg = torch.ones([batch_size, net_input_height, net_input_width], dtype=torch.float32).to("cuda:0")
    # loss_seg = loss_seg(output_seg, gt_seg.long())
    loss_seg = loss_seg_fun(F.softmax(output_seg, dim=1), gt_seg.long(), ignore=255)

    if loss_seg == 0 or not torch.isfinite(loss_seg):exit()

    optimizer.zero_grad()
    loss_seg.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()

    # =========================================
    # inference + visualization
    # =========================================
    org_size = (1920, 1080)

    seg_class_color_id = dict()
    for idx in range(len(segment_class_list)):
        color = (int(np.random.randint(128,255)), int(np.random.randint(128,255)), int(np.random.randint(128,255)) )
        seg_class_color_id.update({idx:color})

    # display
    imgs = [np.ones([org_size[1], org_size[0],3],dtype=np.uint8) for _ in range(batch_size)]
    imgs = segheader.decode(imgs, output_seg, org_size, seg_class_color_id)

    # =========================================
    # validation
    # =========================================
    from head_seg.seg_metrics import IntersectionOverUnion
    metric_evaluator_iou = IntersectionOverUnion(n_classes=len(segment_class_list))

    for _ in range(20):
        # =========================================
        # 分割头
        # =========================================
        output_seg = segheader(feats_seg)
        gt_seg = torch.ones([batch_size, net_input_height, net_input_width])

        for batch_idx in range(batch_size):
            gt_seg_one = gt_seg[batch_idx].unsqueeze(0)
            predict_seg = output_seg[batch_idx].unsqueeze(0).detach().cpu()

            seg_prediction = torch.argmax(predict_seg, dim=1)
            metric_evaluator_iou.update(seg_prediction, gt_seg_one)

    # =========================================
    # 循环后分割头
    # =========================================
    scores = metric_evaluator_iou.compute()
    mIOU = list()
    for key, value in zip(segment_class_list, scores):
        print(key + ", " + "%.3f" % value)
        mIOU.append(value)
    mIOU = np.array(mIOU).mean()
    print("mIOU" + ", " + "%.3f" % mIOU)
