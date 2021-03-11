# MIT License
#
# Copyright (c) 2018 Jonathan Lorraine, Google LLC

# Adapted from https://discuss.pytorch.org/t/unet-implementation/426
# This file is copied from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py
import ipdb

import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        do_noise_channel=False,
        use_identity_residual=False,
        up_mode='upconv'
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()

        self.padding = padding
        self.depth = depth
        self.do_noise_channel = do_noise_channel
        self.use_identity_residual = use_identity_residual

        if self.do_noise_channel:
            in_channels += 1

        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x, use_zero_noise=False, class_label=None):
        blocks = []

        do_class_generation = False
        if self.do_noise_channel:
            if do_class_generation:
                x = x * 0 + class_label.float().reshape(-1, 1, 1, 1)

            noise_channel = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3])).cuda()*0 + torch.randn((x.shape[0], 1, 1, 1)).cuda()
            if use_zero_noise:
                noise_channel = noise_channel * 0

            out = torch.cat([x, noise_channel], dim=1)
        else:
            out = x

        for i, down in enumerate(self.down_path):
            out = down(out)
            if i != len(self.down_path) - 1:
                blocks.append(out)
                out = F.max_pool2d(out, 2)

        for i, up in enumerate(self.up_path):
            out = up(out, blocks[-i - 1])

        if self.use_identity_residual:
            res = self.last(out)
            # normer = 2.0  # (x.norm(dim=(2, 3)) / res.norm(dim=(2, 3))).reshape(x.shape[0], x.shape[1], 1, 1)
            # res = res * normer
            # mixer = 0.75  # torch.rand((x.shape[0], 1, 1, 1)).cuda()  # 0.5
            if not do_class_generation:
                res = torch.tanh(res)
                return x + res  # mixer * x + (1.0 - mixer) * res
            else:
                return res
        else:
            return self.last(out)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out