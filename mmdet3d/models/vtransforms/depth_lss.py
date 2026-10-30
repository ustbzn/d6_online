from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform

__all__ = ["DepthLSSTransform"]


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(#构造函数
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(#调用父类构造函数
            in_channels=in_channels,#256
            out_channels=out_channels,#80
            image_size=image_size,#[256, 704]
            feature_size=feature_size,#[32, 88]
            xbound=xbound,#前后各54m,分辨率0.3
            ybound=ybound,#左右各54m,分辨率0.3
            zbound=zbound,#pillar，-10到+10，分辨率20m
            dbound=dbound,#[1.0, 60.0, 0.5]，深度分辨率
        )
        self.dtransform = nn.Sequential(#输入[4,254,704]
            nn.Conv2d(1, 8, 1),#[4,8,254,704]
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),#[4,32,64,176]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),#[4,64,32,88]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(#输入[4, 256+64, 32, 88]
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),#[4, 256, 32, 88]
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),#[4, 256, 32, 88]
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),#[4, self.D+80, 32, 88]
        )
        if downsample > 1:#2
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,                 
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, d):#通过dtransform，一系列的卷积，进行深度的特征提取，同时resize到特征图的尺度；和前序的特征concate到一起；通过depthnet，预估出新的特征，以及深度对应的权重；
        B, N, C, fH, fW = x.shape#batch_size、camera numbers、feature_chanel-256、feature_h、feature_w[4, 1, 256, 32, 88]

        d = d.view(B * N, *d.shape[2:])#[4, 256, 704]
        x = x.view(B * N, C, fH, fW)#[4, 256, 32, 88]

        d = self.dtransform(d)#使用定义好的dtransform卷积网络进行深度估计,得到64维度深度特征[4,64,32,88]
        x = torch.cat([d, x], dim=1)#将img特征和深度cat起来[4,256+64,32,88]
        x = self.depthnet(x)#256+64->self.D+self.C==?+80,[1.0, 90.0, 0.5]->self.D=178,x.shape = [4,258,32,88]

        depth = x[:, : self.D].softmax(dim=1)#得到深度概率分布

        # import matplotlib.pyplot as plt
        # import os
        # import numpy as np
        # import torch.nn.functional as F
        
        # # 获取最大概率深度索引 [B*N, fH, fW]
        # depth_indices = torch.argmax(depth, dim=1)
        
        # # 上采样到704x256（最近邻插值保持离散值）
        # upsampled = F.interpolate(depth_indices.unsqueeze(1).float(), 
        #                         size=(256, 704), mode='nearest')  # 尺寸顺序为（高度，宽度）
        # upsampled = upsampled.squeeze(1)  # [B*N, 256, 704]
        # # print("upsampled.shape: ", upsampled.shape)
        
        # # 创建保存目录
        # save_dir = "depth_vis"
        # os.makedirs(save_dir, exist_ok=True)
        
        # # 遍历每个样本/相机
        # for i in range(upsampled.shape[0]):
        #     depth_map = upsampled[i].detach().cpu().numpy()
            
        #     # 归一化到0-1并应用颜色映射
        #     depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8)
        #     plt.imsave(
        #         os.path.join(save_dir, f'depth_b{i//N}_cam{i%N}.png'), 
        #         depth_map, 
        #         cmap='jet'
        #     )
        #     plt.close()

        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)#[5, 80, 178, 32, 88]
        # print("x.shape: ", x.shape)

        # selected_features = torch.gather(
        #     x, 
        #     dim=2, 
        #     index=depth_indices.unsqueeze(1).unsqueeze(1).expand(-1, self.C, -1, fH, fW)
        # ).squeeze(2)
        
        # upsampled_features = F.interpolate(
        #     selected_features, 
        #     size=(256, 704), 
        #     mode='bilinear'
        # )
        # # print("upsampled_features.shape: ", upsampled_features[0].shape)

        # # 创建保存目录
        # feature_save_dir = "feature_vis"
        # os.makedirs(feature_save_dir, exist_ok=True)

        # for i in range(upsampled_features.shape[0]):
        #     feature_map = upsampled_features[i, 0].detach().cpu().numpy()  # 形状变为 [H, W]
            
        #     # 归一化到0-1并应用颜色映射
        #     feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-8)
        #     plt.imsave(
        #         os.path.join(feature_save_dir, f'depth_b{i//N}_cam{i%N}.png'), 
        #         feature_map, 
        #         cmap='jet'
        #     )
        #     plt.close()

        x = x.view(B, N, self.C, self.D, fH, fW)#[4,1,80,178,32,88]
        x = x.permute(0, 1, 3, 4, 5, 2)#[4,1,178,32,88,80]
        return x

    def forward(self, *args, **kwargs):#前向函数
        x = super().forward(*args, **kwargs)#调用基类BaseDepthTransform的forward函数 # torch.Size([4, 80, 328, 280])
        x = self.downsample(x)#降采样 torch.Size([4, 80, 164, 140])
        return x