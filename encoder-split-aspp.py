# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import paddle
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d
from .carafe import CARAFE

# from sparseinst.backbones.intern_image import InternImage

SPARSE_INST_ENCODER_REGISTRY = Registry("SPARSE_INST_ENCODER")
SPARSE_INST_ENCODER_REGISTRY.__doc__ = "registry for SparseInst decoder"


class S_ASPP(nn.Module):
    """
    ASPP特征提取模块
    利用不同膨胀率的膨胀卷积进行特征提取
    """

    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, branch_ratio=0.25):
        super(S_ASPP, self).__init__()
        # gc = int(dim_in * branch_ratio)  # channel numbers of a convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=1 * rate, dilation=1 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=5 * rate, dilation=5 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        # self.split_indexes = (dim_in - 3 * gc, gc, gc, gc)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_in * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x_conv1x1, x_conv3x3_1, x_conv3x3_2, x_conv3x3_3 = torch.split(x, self.split_indexes, dim=1)

        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class CARAFE_FPN(nn.Module):
    """
    in_channel_list相当于self.in_features
    out_channel相当于self.num_channels
    in_channel_self相当于self.in_channels
    """

    def __init__(self, in_channel_list, out_channel, in_channel_self):
        super(CARAFE_FPN, self).__init__()
        fpn_laterals = []
        fpn_outputs = []

        self.fpn_features = in_channel_list
        # in_channel_self :In2,In3,In4
        for in_channel in reversed(in_channel_self):
            lateral_conv = Conv2d(in_channel, out_channel, 1)
            output_conv = Conv2d(out_channel, out_channel, 3, padding=1)
            # 初始化lateral层
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            fpn_laterals.append(lateral_conv)
            fpn_outputs.append(output_conv)
        self.fpn_laterals = nn.ModuleList(fpn_laterals)
        self.s_aspp = S_ASPP(out_channel, out_channel)
        self.carafe = CARAFE(out_channel, out_channel)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)

    def forward(self, x):
        x = [x[f] for f in self.fpn_features]
        x = x[::-1]
        prev_features = self.s_aspp(self.fpn_laterals[0](x[0]))  # C5 32,2048,W,H
        outputs = [self.fpn_outputs[0](prev_features)]
        # PAFPN自顶向下部分==FPN部分
        for feature, lat_conv, output_conv in zip(x[1:], self.fpn_laterals[1:], self.fpn_outputs[1:]):
            lat_features = lat_conv(feature)
            # 上采样操作
            # top_down_features = F.interpolate(self.cbam(prev_features), scale_factor=2.0, mode='nearest')
            top_down_features = self.carafe(prev_features)
            prev_features = lat_features + top_down_features
            # 与原码中的操作（进栈）不同，这里是进队列
            # outputs.insert(0, output_conv(prev_features))
            outputs.append(output_conv(prev_features))
            # print("aspp_fpn_output_conv(prev_features).shape):", output_conv(prev_features).shape)

        return outputs


# 构建一个用于下采样的卷积池化模块
class ConvNormLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=1):
        super(ConvNormLayer, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channel)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        return out


class PAFPN_bottom_to_up(nn.Module):
    def __init__(self, in_channel_list, out_channel, cfg):
        super(PAFPN_bottom_to_up, self).__init__()

        # move to target device
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.bottom_up = ConvNormLayer(out_channel, out_channel, 3, 2)  # 2倍下采样模块
        inner_layer = []  # 1x1卷积，统一通道数，处理P3-P5的输出
        out_layer = []  # 3x3卷积，对concat后的特征图进一步融合

        for i in range(len(in_channel_list)):
            if i == 0:
                inner_layer.append(
                    nn.Conv2d(out_channel, out_channel, 1))  # 处理P3,（原用于处理P2，不过此处FPN输出的第一个是P3，类似P2步骤）
            else:
                inner_layer.append(nn.Conv2d(out_channel * 2, out_channel, 1))  # 处理P4-P5，多一个concat操作，故输入层数加倍
            out_layer.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        self.inner_layer = nn.ModuleList(inner_layer)  # 1x1卷积，统一通道数，处理P3-P5的输出
        self.out_layer = nn.ModuleList(out_layer)  # 3x3卷积，对concat后的特征图进一步融合

    # 自底向上部分
    def forward(self, x):
        head_output = []  # 存放最终输出特征图
        corent_inner = self.inner_layer[0](x[-1])  # 过1x1卷积，对P3统一通道数操作
        head_output.append(self.out_layer[0](corent_inner))  # 过3x3卷积，对统一通道后过的特征进一步融合，加入head_output列表
        # print(self.out_layer[0](corent_inner).shape)
        for i in range(1, len(x), 1):
            pre_bottom_up = corent_inner
            pre_concat = self.bottom_up(pre_bottom_up)  # 下采样
            pre_inner = torch.cat([x[-1 - i], pre_concat], dim=1).to(self.device)
            corent_inner = self.inner_layer[i](pre_inner)  # 1x1卷积压缩通道
            head_output.append(self.out_layer[i](corent_inner))  # 3x3卷积进一步融合
            # head_output.insert(0, self.out_layer[i](corent_inner))  # 3x3卷积进一步融合，进栈操作，将N3的顺序置为0
            # outputs.insert(0, output_conv(prev_features))
            # print("pa_fpn_out.shape:", head_output[0].shape)

        return head_output


@SPARSE_INST_ENCODER_REGISTRY.register()
class InstanceContextEncoder(nn.Module):
    """
    Instance Context Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)->s_aspp
    3. multi-scale fusion
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        self.num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
        self.in_features = cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES
        self.in_channels = [input_shape[f].channels for f in self.in_features]

        self.fpn = CARAFE_FPN(self.in_features, self.num_channels, self.in_channels)
        self.pa_fpn_in_channel = self.in_features
        self.pa_fpn = PAFPN_bottom_to_up(self.pa_fpn_in_channel, self.num_channels, cfg)

        # final fusion
        self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
        # self.aspp = ASPP(self.num_channels, self.num_channels)
        # 初始化融合权重
        c2_msra_fill(self.fusion)

    def forward(self, x):
        # FPN输出部分
        fpn_outputs = self.fpn(x)  # list类型
        pa_fpn_outputs = self.pa_fpn(fpn_outputs)  # list类型

        # 特征融合部分，输出大小与P3保持一致
        size = pa_fpn_outputs[0].shape[2:]
        features = [
                       pa_fpn_outputs[0]] + [F.interpolate(x, size, mode='bilinear', align_corners=False) for x in
                                             pa_fpn_outputs[1:]]
        # 替换上采样算子：由双线性插值转化为carafe
        # features = [
        #                pa_fpn_outputs[0]] + [self.carafe(x, delta=size[0] // x.shape[2]) for x in
        #                                      pa_fpn_outputs[1:]]
        features = self.fusion(torch.cat(features, dim=1))
        # print("final_fusion.size:", features.shape)
        # return self.aspp(features)
        return features


def build_sparse_inst_encoder(cfg, input_shape):
    name = cfg.MODEL.SPARSE_INST.ENCODER.NAME
    return SPARSE_INST_ENCODER_REGISTRY.get(name)(cfg, input_shape)


if __name__ == '__main__':
    data = torch.rand(4, 160, 10, 10)
    # carafe = CARAFE(160, 10)
    # print(carafe(data).size())  # 4,10,20,20
    # in_features = ["In2", "In3", "In4"]
    # num_channels = 256
    # model = InternImage(channels=64).cuda()
    # in_features = [torch.rand(4, 256, 80, 80), torch.rand(4, 256, 40, 40), torch.rand(4, 256, 20, 20)]
    # module = InstanceContextEncoder(in_features, input_shape=data.size())
    # print(module)
