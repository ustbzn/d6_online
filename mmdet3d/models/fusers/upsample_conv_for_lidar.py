import random
from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["UpsampleConvForLidarFuser"]


@FUSERS.register_module()
class UpsampleConvForLidarFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fuser = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.upsample_lidar = nn.Sequential(
                    nn.ConvTranspose2d(
                    256, 256,
                    2,
                    stride=2, bias=False
                    ),
                    nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                    nn.ReLU(True),
                )


    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        features = []
        for input in inputs:
            channel = input.shape[1]
            if channel == 80:
                features.append(input)
            else:
                features.append(self.upsample_lidar(input))

        return self.fuser(torch.cat(features, dim=1))
