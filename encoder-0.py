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

SPARSE_INST_ENCODER_REGISTRY = Registry("SPARSE_INST_ENCODER")
SPARSE_INST_ENCODER_REGISTRY.__doc__ = "registry for SparseInst decoder"


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, channels=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes]
        )
        self.bottleneck = Conv2d(
            in_channels + len(sizes) * channels, in_channels, 1)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=F.relu_(stage(feats)), size=(
            h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out


class CAM(nn.Module):
    def __init__(self, inc, fusion='weight'):
        super().__init__()

        assert fusion in ['weight', 'adaptive', 'concat']
        self.fusion = fusion

        self.conv1 = Conv2d(inc, inc, 3, 1, None, 1, 1)
        self.conv2 = Conv2d(inc, inc, 3, 1, None, 1, 3)
        self.conv3 = Conv2d(inc, inc, 3, 1, None, 1, 5)

        self.fusion_1 = Conv2d(inc, inc, 1)
        self.fusion_2 = Conv2d(inc, inc, 1)
        self.fusion_3 = Conv2d(inc, inc, 1)

        if self.fusion == 'adaptive':
            self.fusion_4 = Conv2d(inc * 3, 3, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        if self.fusion == 'weight':
            return self.fusion_1(x1) + self.fusion_2(x2) + self.fusion_3(x3)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(
                self.fusion_4(torch.cat([self.fusion_1(x1), self.fusion_2(x2), self.fusion_3(x3)], dim=1)), dim=1)
            x1_weight, x2_weight, x3_weight = torch.split(fusion, [1, 1, 1], dim=1)
            return x1 * x1_weight + x2 * x2_weight + x3 * x3_weight
        else:
            return torch.cat([self.fusion_1(x1), self.fusion_2(x2), self.fusion_3(x3)], dim=1)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        # 分别基于宽度和高度分别进行最大池化和平局池化
        # self.max_pool = nn.AdaptiveMaxPool2d(kernel_size=(1, None), output_size=(1, 1))
        # self.avg_pool = nn.AdaptiveAvgPool2d(kernel_size=(None, 1), output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        # spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print('max_out:', max_out.shape)  # 32,1
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print('avg_out:', avg_out.shape)  # 32,1
        a = torch.cat([max_out, avg_out], dim=1)
        # print('a:', a.shape)  # 32,2

        spatial_out = self.sigmoid(a)  # 32,2
        # print('spatial_out:', spatial_out.shape)  # 32,2

        spatial_out = self.conv(spatial_out)  # 调整通道数
        # print('spatial_out:', spatial_out.shape)  # 32,1

        spatial_out = self.sigmoid(spatial_out)
        x = spatial_out * x

        # print('x:', x.shape)#32,1280
        return x


# -----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP_CBAM(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP_CBAM, self).__init__()
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

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAMLayer(channel=dim_out * 5)

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

        # 加入CBAM注意力机制
        cbamaspp = self.cbam(feature_cat)
        result = self.conv_cat(cbamaspp)
        # print('aspp_cbam.shape:', result[0].shape)  # 32,256,20,27(通道由1280变为256，图片大小不变）
        return result


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class ASPP_FPN(nn.Module):

    # in_channel_list相当于self.in_features
    # out_channel相当于self.num_channels
    # in_channel_self相当于self.in_channels
    def __init__(self, in_channel_list, out_channel, in_channel_self):
        super(ASPP_FPN, self).__init__()
        fpn_laterals = []
        fpn_outputs = []
        # print("aspp_fpn_in_channel_list.type:", type(in_channel_list))  # list类型
        self.fpn_features = in_channel_list
        # print("aspp_fpn_in_channel_list.shape:", len(in_channel_list))  # aspp_cbam.shape图片大小的2倍
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
        # # ppm
        # self.ppm = PyramidPoolingModule(self.num_channels, self.num_channels // 4)
        # ASPP
        self.asppcbam = ASPP_CBAM(out_channel, out_channel)


    def forward(self, x):
        x = [x[f] for f in self.fpn_features]
        x = x[::-1]
        # 对C5（通道数：2048）进行ASPP_CBAM操作
        # print('x[0].shape:', x[0].shape)  # 32,2048,27,20
        # print('self.fpn_laterals[0](x[0]).shape:', self.fpn_laterals[0](x[0]).shape)
        prev_features = self.asppcbam(self.fpn_laterals[0](x[0]))
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
    2. enlarge receptive fields (ppm)->aspp
    3. multi-scale fusion
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        self.num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
        self.in_features = cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES
        self.in_channels = [input_shape[f].channels for f in self.in_features]

        self.fpn = ASPP_FPN(self.in_features, self.num_channels, self.in_channels)
        self.pa_fpn_in_channel = self.in_features

        self.pa_fpn = PAFPN_bottom_to_up(self.pa_fpn_in_channel, self.num_channels, cfg)

        # final fusion
        self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
        # 初始化融合权重
        c2_msra_fill(self.fusion)

    def forward(self, x):
        # print("输入图片大小为：", len(x))
        # FPN输出部分
        fpn_outputs = self.fpn(x)
        # print("fpn_outputs.type:", type(fpn_outputs))  # list类型
        pa_fpn_outputs = self.pa_fpn(fpn_outputs)  # list类型

        # for i in pa_fpn_outputs:
        #     print("pa_fpn_outputs.size:", i[0].shape)
        # 特征融合部分
        # 经过自顶向下和自底向上部分的list顺序是否一致，箭头顺序是从小到大方向
        # 从N3开始都进行上采样并与0融合
        size = pa_fpn_outputs[0].shape[2:]
        # print("pa_fpn_outputs[0].shape[2:]:", size)
        features = [
                       pa_fpn_outputs[0]] + [F.interpolate(x, size, mode='bilinear', align_corners=False) for x in
                                             pa_fpn_outputs[1:]]
        features = self.fusion(torch.cat(features, dim=1))
        # print("final_fusion.size:", features.shape)
        return features


def build_sparse_inst_encoder(cfg, input_shape):
    name = cfg.MODEL.SPARSE_INST.ENCODER.NAME
    return SPARSE_INST_ENCODER_REGISTRY.get(name)(cfg, input_shape)
