import torch
import torch.nn as nn
import torch.nn.functional as F
from attentions.EfficientCBAM import EfficientCBAM
from attentions.SimAM import SimAM


# 深度可分离卷积块
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation,
                      groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),  # inplace=True节省空间
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class S_ASPP_CBAM(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(S_ASPP_CBAM, self).__init__()
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
        conv3x3_1 = x + self.branch2(x)
        conv3x3_2 = x + self.branch3(x)
        conv3x3_3 = x + self.branch4(x)
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


if __name__ == '__main__':
    inputs = torch.randn(4, 256, 13, 13)  # b, c,h, w
    # torch.randn(2, 3, 640, 640)
    aspp_module = S_ASPP_CBAM(dim_in=256, dim_out=256)
    outputs = aspp_module(inputs)
    print(aspp_module)
    # 计算模型参数数量
    num_params = sum(p.numel() for p in aspp_module.parameters() if p.requires_grad)
    # 参数量是模型中可训练的参数数量，不包括不需要学习的参数（如 Batch Normalization 层中的参数）
    print("S_ASPP_SimAM 模型的参数数量：", num_params)
