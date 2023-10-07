import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

SPARSE_INST_ENCODER_REGISTRY = Registry("SPARSE_INST_ENCODER")
SPARSE_INST_ENCODER_REGISTRY.__doc__ = "registry for SparseInst decoder"

from attentions.SimAM import SimAM


# 深度可分离卷积块
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class S_ASPP_SimAM(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(S_ASPP_SimAM, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size=3, padding=1 * rate, dilation=1 * rate)
        self.branch3 = DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size=3, padding=3 * rate, dilation=3 * rate)
        self.branch4 = DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size=3, padding=5 * rate, dilation=5 * rate)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.attention = SimAM()
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
        aspp_simam = self.attention(feature_cat)
        result = self.conv_cat(aspp_simam)
        # print('aspp_cbam.shape:', result[0].shape)  # 32,256,20,27(通道由1280变为256，图片大小不变）
        return result


@SPARSE_INST_ENCODER_REGISTRY.register()
class InstanceContextEncoder(nn.Module):
    """
    Instance Context Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)
    3. multi-scale fusion
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
        self.in_features = cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES
        self.in_channels = [input_shape[f].channels for f in self.in_features]
        fpn_laterals = []
        fpn_outputs = []
        for in_channel in reversed(self.in_channels):
            lateral_conv = Conv2d(in_channel, self.num_channels, 1)
            output_conv = Conv2d(self.num_channels, self.num_channels, 3, padding=1)
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            fpn_laterals.append(lateral_conv)
            fpn_outputs.append(output_conv)
        self.fpn_laterals = nn.ModuleList(fpn_laterals)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)
        # ppm
        # self.ppm = PyramidPoolingModule(self.num_channels, self.num_channels // 4)
        self.aspp_simam = S_ASPP_SimAM(self.num_channels, self.num_channels)
        # final fusion
        self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
        c2_msra_fill(self.fusion)

    def forward(self, features):
        features = [features[f] for f in self.in_features]
        features = features[::-1]
        prev_features = self.aspp_simam(self.fpn_laterals[0](features[0]))
        outputs = [self.fpn_outputs[0](prev_features)]
        for feature, lat_conv, output_conv in zip(features[1:], self.fpn_laterals[1:], self.fpn_outputs[1:]):
            lat_features = self.aspp_simam(lat_conv(feature))
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode='nearest')
            prev_features = lat_features + top_down_features
            outputs.insert(0, output_conv(prev_features))
        size = outputs[0].shape[2:]
        features = [
                       outputs[0]] + [F.interpolate(x, size, mode='bilinear', align_corners=False) for x in outputs[1:]]
        features = self.fusion(torch.cat(features, dim=1))
        return features


def build_sparse_inst_encoder(cfg, input_shape):
    name = cfg.MODEL.SPARSE_INST.ENCODER.NAME
    return SPARSE_INST_ENCODER_REGISTRY.get(name)(cfg, input_shape)
