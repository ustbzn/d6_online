from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["SENetFuser"]


class SELayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,1,0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction,channel,1,1,0),
            nn.Sigmoid()
        )

    def forward(self,x):
        out = self.avg_pool(x)
        out = self.fc(out)
        out = x*out
        return out

@FUSERS.register_module()
class SENetFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels

        # selayers = [SELayer(in_channel) for in_channel in in_channels]
        selayers = SELayer(sum(in_channels))
        # import ipdb; ipdb.set_trace()

        super().__init__(
            # *selayers,
            selayers,
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))

