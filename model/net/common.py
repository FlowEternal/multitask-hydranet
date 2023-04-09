import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops.boxes import nms as nms_torch

def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

        self.padding_size = [0,0,0,0] if self.kernel_size[0] ==1 else [1,1,1,1]

    def forward(self, x):
        # h, w = x.shape[-2:]
        # extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        # extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        # left = extra_h // 2
        # right = extra_h - left
        # top = extra_v // 2
        # bottom = extra_v - top
        x = F.pad(x, self.padding_size)

        x = self.conv(x)
        return x


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2
        self.padding_size = [0, 1, 0, 1]

    def forward(self, x):
        # h, w = x.shape[-2:]
        # extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        # extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        # left = extra_h // 2
        # right = extra_h - left
        # top = extra_v // 2
        # bottom = extra_v - top
        # print([left, right, top, bottom],[h,w,self.stride[0],self.kernel_size[0]])
        x = F.pad(x, self.padding_size)

        x = self.pool(x)
        return x