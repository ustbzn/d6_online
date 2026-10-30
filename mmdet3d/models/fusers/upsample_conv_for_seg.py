import random
from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["UpsampleConvForSegFuser"]


@FUSERS.register_module()
class UpsampleConvForSegFuser(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        self.fuser = nn.Sequential(
            # nn.Conv2d(256, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(True),
            #nn.Conv2d(mid_channels, out_channels, 1, padding=0, bias=False),
            nn.Conv2d(mid_channels, out_channels,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.upsample_feature = nn.Sequential(
                    nn.ConvTranspose2d(
                    in_channels[0], mid_channels,
                    2,
                    stride=2, bias=False
                    ),
                    nn.BatchNorm2d(mid_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU(True),
                )


    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        #features = []
        #for input in inputs:
        #    channel = input.shape[1]
        #    if channel == 320: # TODO: modify
        #        features.append(input)
        #    else:
        #        features.append(self.upsample_feature(input))
        x1=self.upsample_feature(inputs)
        return self.fuser(x1)
        #return self.fuser(torch.cat(features, dim=1))
