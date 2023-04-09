"""
Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)
"""

import numpy as np
from net.anynet import AnyNetXe


class RegNetX(AnyNetXe):
    def __init__(self,
                 initial_width,
                 slope,
                 quantized_param,
                 network_depth,
                 bottleneck_ratio,
                 group_width,
                 stride,
                 se_ratio=None
                 ):

        # We need to derive block width and number of blocks from initial parameters.
        parameterized_width = initial_width + slope * np.arange(network_depth)  # From equation 2
        parameterized_block = np.log(parameterized_width / initial_width) / np.log(quantized_param)  # From equation 3
        parameterized_block = np.round(parameterized_block)
        quantized_width = initial_width * np.power(quantized_param, parameterized_block)

        # We need to convert quantized_width to make sure that it is divisible by 8
        quantized_width = 8 * np.round(quantized_width / 8)
        ls_block_width, ls_num_blocks = np.unique(quantized_width.astype(np.int32), return_counts=True)

        # At this points, for each stage, the above-calculated block width could be incompatible to group width
        # due to bottleneck ratio. Hence, we need to adjust the formers.
        # Group width could be swapped to number of groups, since their multiplication is block width
        ls_group_width = np.array([min(group_width, block_width // bottleneck_ratio) for block_width in ls_block_width])
        ls_block_width = np.round(ls_block_width // bottleneck_ratio / group_width) * group_width
        ls_group_width = ls_group_width.astype(np.int32) * bottleneck_ratio
        ls_bottleneck_ratio = [bottleneck_ratio for _ in range(len(ls_block_width))]
        # print (ls_num_blocks)
        # print (ls_block_width)
        # print (ls_bottleneck_ratio)
        # print (ls_group_width)

        super(RegNetX, self).__init__(ls_num_blocks, ls_block_width.astype(np.int32).tolist(), ls_bottleneck_ratio,
                                       ls_group_width.tolist(), stride, se_ratio)


class RegNetY(RegNetX):
    # RegNetY = RegNetX + SE
    def __init__(self, initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride,
                 se_ratio):
        super(RegNetY, self).__init__(initial_width, slope, quantized_param, network_depth, bottleneck_ratio,
                                      group_width, stride, se_ratio)


if __name__ == '__main__':
    # parameters setting
    bottleneck_ratio = 1
    group_width = 8
    initial_width = 24
    slope = 36
    quantized_param = 2.5
    network_depth = 30
    stride = 2
    se_ratio = 4

    # create models
    model = RegNetY(initial_width,
                    slope,
                    quantized_param,
                    network_depth,
                    bottleneck_ratio,
                    group_width,
                    stride,
                    se_ratio)

    model.to("cuda:0")
    model.eval()

    # inference
    import torch
    dummy_input = torch.randn((1, 3, 640, 640)).to("cuda:0")
    dummy_output = model(dummy_input)
    print(len(dummy_output))
    for output in dummy_output:
        print(output.shape)

    # warm-up GPU
    import time
    for _ in range(5):
        torch.cuda.synchronize(0)
        tic = time.time()
        _ = model(dummy_input)
        torch.cuda.synchronize(0)
        tac = time.time()
        print("one batch inference time is %i" % ((tac - tic)*1000))

    # inference speed
    acc_time = 0.0
    for _ in range(5):
        torch.cuda.synchronize(0)
        tic = time.time()
        _ = model(dummy_input)
        torch.cuda.synchronize(0)
        tac = time.time()
        acc_time += (tac - tic) * 1000
    print("average one batch inference time is %i" % (acc_time / 5.0) )

    # export to onnx
    # import torch.onnx
    # torch.onnx.export(
    #     model,dummy_input,
    #     "backbone.onnx",
    #     export_params=True,
    #     input_names=["input"],
    #     output_names=["features"],
    #     opset_version=11,
    #     verbose=False,
    # )


