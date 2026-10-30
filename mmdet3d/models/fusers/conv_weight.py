from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvWeightFuser"]


@FUSERS.register_module()
class ConvWeightFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, weight: float) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = weight
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat([self.weight[0]*inputs[0], self.weight[1]*inputs[1]], dim=1))
