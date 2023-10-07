# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import paddle
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

from operator import length_hint
import numpy as np
from timm.models.layers.create_act import create_act_layer
from attentions.SimAM import SimAM

SPARSE_INST_ENCODER_REGISTRY = Registry("SPARSE_INST_ENCODER")
SPARSE_INST_ENCODER_REGISTRY.__doc__ = "registry for SparseInst decoder"


# 深度可分离卷积块
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, bn_mom=0.1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation,
                      groups=in_channels),
            nn.BatchNorm2d(in_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),  # inplace=True节省空间
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class S_ASPP_SimAM(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(S_ASPP_SimAM, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size=3, padding=6 * rate, dilation=6 * rate)
        self.branch3 = DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size=3, padding=18 * rate, dilation=18 * rate)
        self.branch4 = DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size=3, padding=24 * rate, dilation=24 * rate)

        self.branch5 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.attention = SimAM()
        # self.attention = EfficientCBAM(channels=dim_out * 5)
        # self.cbam = EfficientCBAM(channels=dim_out * 5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        # 第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)

        # 加入CBAM注意力机制
        aspp_simam = self.attention(feature_cat)
        result = self.conv_cat(aspp_simam)
        # print('aspp_cbam.shape:', result[0].shape)  # 32,256,20,27(通道由1280变为256，图片大小不变）
        return result


class EffectiveSEModule(nn.Module):
    def __init__(self, channels=512, add_maxpool=False, gate_layer='hard_sigmoid'):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        out = x * self.gate(x_se)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class ASPP_BiFPN(nn.Module):

    # in_channel_list相当于self.in_features
    # out_channel相当于self.num_channels
    # in_channel_self相当于self.in_channels
    def __init__(self, in_channel_list, out_channel, in_channel_self):
        super(ASPP_BiFPN, self).__init__()
        fpn_laterals = []
        fpn_outputs = []
        self.fpn_features = in_channel_list
        for in_channel in reversed(in_channel_self):
            lateral_conv = Conv2d(in_channel, out_channel, 1)
            output_conv = Conv2d(out_channel, out_channel, 3, padding=1)
            # 初始化lateral层
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            fpn_laterals.append(lateral_conv)
            fpn_outputs.append(output_conv)
        self.fpn_laterals = nn.ModuleList(fpn_laterals)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)
        # ASPP
        self.aspp_simam = S_ASPP_SimAM(out_channel, out_channel)

        # bottom_to_up part
        # 2倍下采样模块
        self.bottom_up = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel, momentum=0.1),
        )
        inner_layer = []  # 1x1卷积，统一通道数，处理P3-P5的输出
        out_layer = []  # 3x3卷积，对concat后的特征图进一步融合

        for i in range(len(in_channel_list)):
            if i == 0:
                inner_layer.append(
                    nn.Conv2d(out_channel, out_channel, 1))  # 用于处理P2
            elif i == 1:
                inner_layer.append(nn.Conv2d(out_channel * 4, out_channel, 1))  # 处理P3，3个concat操作(
                # 加一个C3、C4残差边），故输入层数加倍
            elif i == 2:
                inner_layer.append(nn.Conv2d(out_channel * 6, out_channel, 1))  # 处理P4，3个concat操作(
                # 加一个C3、C4残差边），故输入层数加倍
            else:  # i=3
                inner_layer.append(nn.Conv2d(out_channel * 2, out_channel, 1))  # 处理p5,2个concat操作
            out_layer.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        self.inner_layer = nn.ModuleList(inner_layer)  # 1x1卷积，统一通道数，处理P3-P5的输出
        self.out_layer = nn.ModuleList(out_layer)  # 3x3卷积，对concat后的特征图进一步融合

    def forward(self, x):
        x = [x[f] for f in self.fpn_features]
        c3 = x[1]
        c4 = x[2]
        x = x[::-1]

        # fpn part
        prev_features = self.aspp_simam(self.fpn_laterals[0](x[0]))
        outputs = [self.fpn_outputs[0](prev_features)]
        # PAFPN自顶向下部分==FPN部分
        for feature, lat_conv, output_conv in zip(x[1:], self.fpn_laterals[1:], self.fpn_outputs[1:]):
            lat_features = lat_conv(feature)
            # 上采样操作
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode='nearest')
            prev_features = lat_features + top_down_features
            # 与原码中的操作（进栈）不同，这里是进队列
            # outputs.insert(0, output_conv(prev_features))
            outputs.append(output_conv(prev_features))

        # bottom_to_up part
        x = outputs
        head_output = []  # 存放最终输出特征图
        corent_inner = self.inner_layer[0](x[-1])  # 过1x1卷积，对P2统一通道数操作
        # p2不加入输出列表
        head_output.append(self.out_layer[0](corent_inner))  # 过3x3卷积，对统一通道后过的特征进一步融合，加入head_output列表
        for i in range(1, len(x), 1):
            pre_bottom_up = corent_inner
            pre_concat = self.bottom_up(pre_bottom_up)  # 下采样
            if i == 1:
                pre_inner = torch.cat([c3, x[-1 - i], pre_concat], dim=1)
            elif i == 2:
                pre_inner = torch.cat([c4, x[-1 - i], pre_concat], dim=1)
            else:
                pre_inner = torch.cat([x[-1 - i], pre_concat], dim=1)
            corent_inner = self.inner_layer[i](pre_inner)  # 1x1卷积压缩通道
            head_output.append(self.out_layer[i](corent_inner))  # 3x3卷积进一步融合

        return head_output


@SPARSE_INST_ENCODER_REGISTRY.register()
class InstanceContextEncoder(nn.Module):
    """
    Instance Context Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)->aspp
    3. multi-scale fusion
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        self.num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
        self.in_features = cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES
        self.in_channels = [input_shape[f].channels for f in self.in_features]
        self.bifpn = ASPP_BiFPN(self.in_features, self.num_channels, self.in_channels)

        # final fusion
        self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
        self.ca = EffectiveSEModule(channels=self.num_channels)
        self.sa = SpatialAttention(kernel_size=5)
        self.bottom_up = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.num_channels, momentum=0.1),
        )
        # 初始化融合权重
        c2_msra_fill(self.fusion)

    def forward(self, x):
        pa_fpn_outputs = self.bifpn(x)  # list类型
        # 特征融合部分
        # 经过自顶向下和自底向上部分的list顺序是否一致，箭头顺序是从小到大方向
        # 从N3开始都进行上采样并与0融合
        size = pa_fpn_outputs[1].shape[2:]
        ca_feature = F.interpolate(pa_fpn_outputs[2], size, mode='bilinear', align_corners=False)
        sa_feature = self.bottom_up(pa_fpn_outputs[0])
        features = [
                       pa_fpn_outputs[1]] + [F.interpolate(x, size, mode='bilinear', align_corners=False) for x in
                                             pa_fpn_outputs[2:]]
        features = self.fusion(torch.cat(features, dim=1))
        features = ca_feature * self.ca(features) + sa_feature * self.sa(features)
        return features


def build_sparse_inst_encoder(cfg, input_shape):
    name = cfg.MODEL.SPARSE_INST.ENCODER.NAME
    return SPARSE_INST_ENCODER_REGISTRY.get(name)(cfg, input_shape)
