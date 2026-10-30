import random
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSERS

__all__ = ["UpsampleConvForOccFuser", "UpsampleInterpolationForOccFuser"]

class OccBEVAdapter(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),

            nn.Conv2d(C, C, 3, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


@FUSERS.register_module()
class UpsampleConvForOccFuser(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 layer_num: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_num = 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.inconv = nn.Sequential(
                    nn.ConvTranspose2d(
                    in_channels, in_channels/2,
                    2,
                    stride=2, bias=False
                    ),
                    nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU(True),
                )


    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        #features = []
        #for input in inputs:
        #    channel = input.shape[1]
        #    if channel == 320: 
        #        features.append(input)
        #    else:
        #        features.append(self.upsample_feature(input))
        x1=self.inconv(inputs)
        return self.conv(x1)
        #return self.fuser(torch.cat(features, dim=1))
    
@FUSERS.register_module()
class UpsampleInterpolationForOccFuser(nn.Module):
    def __init__(self,
                 inplanes: int,
                 interplanes: int,
                 outplanes: int,
                 scale_factor: int,
                 pc_range: list,
                 occ_range: list,
                 mode: str):
        super().__init__()

        self.occ_adapter = OccBEVAdapter(inplanes)
        # self.conv1 = nn.Conv2d(
        #     inplanes,
        #     interplanes,
        #     kernel_size=3,
        #     padding=2,
        #     dilation=2,
        #     bias=False
        # )
        # self.bn1 = nn.BatchNorm2d(interplanes)
        # self.relu1 = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv2d(interplanes, outplanes, 1)
        
        self.conv1 = nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(interplanes, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, 1, bias=True)
        
        self.scale_factor = scale_factor
        self.pc_range = pc_range
        self.occ_range = occ_range
        self.mode = mode 
        
    def forward(self, bev_feat):
        bev_feat = self.occ_adapter(bev_feat)
        
        B, C, H, W = bev_feat.shape
        # device = bev_feat.device
        # dtype = bev_feat.dtype

        dx = (self.pc_range[3] - self.pc_range[0]) / H
        start = int(round((self.occ_range[0] - self.pc_range[0]) / dx))
        end   = int(round((self.occ_range[3] - self.pc_range[0]) / dx))

        # assert end > start

        # xs = torch.arange(H, device=device, dtype=dtype)
        # # 平滑边界宽度（超参数，2~4 个 voxel 很稳）
        # tau = 2.0

        # left  = torch.sigmoid((xs - start) / tau)
        # right = torch.sigmoid((end - xs) / tau)
        # mask_1d = left * right                       

        # mask = mask_1d.view(1, 1, H, 1)              
        # mask = mask.expand(B, 1, H, W)
        
        # occ_feat = bev_feat * mask.detach()
        occ_feat = bev_feat[:, :, start:end, :]
        occ_feat = self.relu1(self.bn1(self.conv1(occ_feat)))
        # occ_feat = self.relu2(self.bn2(self.conv2(occ_feat)))

        # occ_feat = occ_feat * mask
        # occ_feat = occ_feat[:, :, start:end, :]
        out = self.conv2(occ_feat)

        if self.scale_factor is not None:
            out = F.interpolate(
                out,
                scale_factor=self.scale_factor,
                mode=self.mode,
                align_corners=False if self.mode != 'nearest' else None
            )

        return out

# @FUSERS.register_module()
# class UpsampleInterpolationForOccFuser(nn.Module):
#     def __init__(self,
#                  inplanes: int,
#                  interplanes: int,
#                  outplanes: int,
#                  scale_factor: int,
#                  pc_range: list,
#                  occ_range: list,
#                  mode: str):
#         super().__init__()
#         self.inplanes = inplanes
#         self.interplanes = interplanes
#         self.outplanes = outplanes
#         self.bn1 = nn.BatchNorm2d(inplanes, momentum=0.1)
#         self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(interplanes, momentum=0.1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
#         self.scale_factor = scale_factor

#         self.pc_range = pc_range
#         self.occ_range = occ_range
#         self.mode = mode

#     def forward(self, bev_feat):
#         B, C, H, W = bev_feat.shape
#         device = bev_feat.device
#         dtype = bev_feat.dtype

#         dx = (self.pc_range[3] - self.pc_range[0]) / H
#         start = int(round((self.occ_range[0] - self.pc_range[0]) / dx))
#         end   = int(round((self.occ_range[3] - self.pc_range[0]) / dx))

#         assert end > start

#         # xs = torch.arange(H, device=device, dtype=dtype)
#         # # 平滑边界宽度（超参数，2~4 个 voxel 很稳）
#         # tau = 2.0

#         # left  = torch.sigmoid((xs - start) / tau)
#         # right = torch.sigmoid((end - xs) / tau)
#         # mask_1d = left * right                       

#         # mask = mask_1d.view(1, 1, H, 1)              
#         # mask = mask.expand(B, 1, H, W)
        
#         # occ_feat = bev_feat * mask.detach()  
#         x = bev_feat[:, :, start:end, :]
#         x = self.conv1(self.relu(self.bn1(x)))
#         out = self.conv2(self.relu(self.bn2(x)))

#         if self.scale_factor is not None:
#             height = out.shape[-2] * self.scale_factor
#             width = out.shape[-1] * self.scale_factor
#             out = F.interpolate(out,
#                         size=[height, width],
#                         mode='bilinear')
#         return out