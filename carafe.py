# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:35:40 2020
@author: ZZL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    """
    Kup is the size of the reassembly kernel.
    Cm is the channel number after compression.
    delta is the expected upsample scale.
    Kencoder is  the kernel size of the encoder.
    """

    def __init__(self, inC, outC, Kencoder=3, delta=2, Kup=5, Cm=64):  # Kup = Kencoder + 2
        super(CARAFE, self).__init__()
        self.Kencoder = Kencoder
        self.delta = delta
        self.Kup = Kup
        self.down = nn.Conv2d(in_channels=inC, out_channels=Cm, kernel_size=1)  #
        self.encoder = nn.Conv2d(64, self.delta ** 2 * self.Kup ** 2,
                                 self.Kencoder, 1, self.Kencoder // 2)
        self.out = nn.Conv2d(inC, outC, 1)

    # delta 与上述的delta需要分开传入
    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(in_tensor)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, delta^2 * Kup^2, H, W)
        # reshape
        kernel_tensor = F.pixel_shuffle(kernel_tensor,
                                        upscale_factor=self.delta)  # (N, delta^2 * Kup^2, H, W)->(N, Kup^2, delta*H,
        # delta*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, delta*H, delta*W)
        kernel_tensor = kernel_tensor.unfold(2, self.delta, step=self.delta)  # (N, Kup^2, H, W*delta, delta)
        kernel_tensor = kernel_tensor.unfold(3, self.delta, step=self.delta)  # (N, Kup^2, H, W, delta, delta)
        kernel_tensor = kernel_tensor.reshape(N, self.Kup ** 2, H, W, self.delta ** 2)  # (N, Kup^2, H, W, delta^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, delta^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        in_tensor = F.pad(in_tensor, pad=(self.Kup // 2, self.Kup // 2,
                                          self.Kup // 2, self.Kup // 2),
                          mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        in_tensor = in_tensor.unfold(dimension=2, size=self.Kup, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        in_tensor = in_tensor.unfold(3, self.Kup, step=1)  # (N, C, H, W, Kup, Kup)
        in_tensor = in_tensor.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(in_tensor, kernel_tensor)  # (N, H, W, C, delta^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.delta)  # 4，160，20，20
        out_tensor = self.out(out_tensor)
        return out_tensor


if __name__ == '__main__':
    data = torch.rand(4, 160, 10, 10)
    carafe = CARAFE(160, 10)
    print(carafe(data).size())  # 4,10,20,20
