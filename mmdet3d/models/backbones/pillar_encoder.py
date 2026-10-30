"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

from typing import Any, Dict

import torch
from mmcv.cnn import build_norm_layer
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import build_backbone
from mmdet.models import BACKBONES

import ast
import sys

__all__ = ["PillarFeatureNet", "PointPillarsScatter", "PointPillarsEncoder"]


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]
    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = "PFNLayer"
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):

        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            #x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            # trt not repeat, change by kh
            x_repeat = x_max.permute(0,2,1)
            val = torch.ones(1,inputs.shape[1]).cuda()
            x_repeat = x_repeat * val

            x_repeat = x_repeat.permute(0,2,1)

            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PFNLayerV2(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False, first_layer=True):
        """Implemet PFNLayer with conv.

        Implement PFNLayer by conv2d inplace of linear layer, to reduce the time-consuming
        transpose op for large size data tensor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            norm_cfg (optional): Batchnorm config. Defaults to None.
            last_layer (bool): If last_layer, there is no concatenation of features.
            first_layer (bool): If first_layer, there need a permute adjust features.

        """
        
        super().__init__()
        self.name = "PFNLayer"
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN2d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.conv = nn.Conv2d(in_channels, self.units, 1, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]
        self.first_layer = first_layer
    
    def forward(self, inputs):
        if self.first_layer:
            # inputs: (N, P, C) -> (1, C, N, P)
            x = inputs.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        else:
            x = inputs
        x = self.conv(x)
        torch.backends.cudnn.enabled = False
        x = self.norm(x)
        torch.backends.cudnn.enabled = True
        x = F.relu(x)  # (1, C, N, P)

        if self.last_vfe:
            x_max = torch.max(x, dim=-1, keepdim=False)[0]  # (1, C, N)
            return x_max.squeeze(0).permute(1, 0).contiguous()
        else:
            x_max = torch.max(x, dim=-1, keepdim=True)[0]
            # trt not support repeat
            val = torch.ones(1, 1, 1, inputs.shape[1]).cuda()
            x_repeat = x_max * val
            x_concatenated = torch.concat([x, x_repeat], dim=1)  # (1, C2, N, P)
            return x_concatenated


@BACKBONES.register_module()
class PillarFeatureNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        feat_channels=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
        conv_as_fc=False,
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        :param conv_as_fc: (<bool>). Replace fc with conv to reduce deploy latency.
        """

        super().__init__()
        self.name = "PillarFeatureNet"
        assert len(feat_channels) > 0

        self.in_channels = in_channels
        in_channels += 5
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self.conv_as_fc = conv_as_fc

        # Create PillarFeatureNet layers
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            if i == 0:
                first_layer = True
            else:
                first_layer = False
            if not self.conv_as_fc:
                pfn_layers.append(
                    PFNLayer(
                        in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                    )
                )
            else:
                pfn_layers.append(
                    PFNLayerV2(
                        in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer, first_layer=first_layer
                    )
                )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device

        dtype = features.dtype

        # Find distance of x, y, and z from cluster center
        # features = features[:, :, :self.num_input]
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
            features
        ).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        # f_center = features[:, :, :2]
        # modified according to xyz coords
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 1].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # print("1111111111111111111111111111111111111111")
        # print("type(features) = ", type(features))
        # print("features.shape = ", features.shape)

        # with open('/pcpt/pcpt/project/atlb/model_online/1.bin', 'wb') as f:
        #     f.write(features.cpu().numpy().tobytes())   # 二进制块
        # # 把形状/dtype 记到同名文本文件，方便恢复
        # with open('/pcpt/pcpt/project/atlb/model_online/1.meta', 'w') as f:
        #     f.write(f"{tuple(features.shape)}\n{features.dtype}\n")

        # # 1. 读回 meta
        # with open('/pcpt/pcpt/project/atlb/model_online/1.meta', 'r') as f:
        #     test_shape = ast.literal_eval(f.readline().strip())
        #     # test_dtype = getattr(torch, f.readline().strip())

        # # 2. 读二进制并 reshape
        # with open('/pcpt/pcpt/project/atlb/model_online/1.bin', 'rb') as f:
        #     swpld = torch.frombuffer(f.read(), dtype=torch.float32).view(test_shape).to(features.device)

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            # swpld = pfn(swpld)
            features = pfn(features)

        # print('swpld = ', swpld)

        # print(swpld.flatten()[:100])

        # print("222222222222222222222222")
        # print('features = ', features)
        # sys.exit(1)

        return features.squeeze()


@BACKBONES.register_module()
class PointPillarsScatter(nn.Module):
    def __init__(self, in_channels=64, output_shape=(512, 512), **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.in_channels = in_channels
        self.output_shape = output_shape
        self.nx = output_shape[0]
        self.ny = output_shape[1]

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, output_shape={tuple(self.output_shape)}"
        )

    def forward(self, voxel_features, coords, batch_size):
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt

            this_coords = coords[batch_mask, :]
            # modified -> xyz coords
            indices = this_coords[:, 1] * self.ny + this_coords[:, 2]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.nx, self.ny)
        return batch_canvas


@BACKBONES.register_module()
class PointPillarsEncoder(nn.Module):
    def __init__(
        self,
        pts_voxel_encoder: Dict[str, Any],
        pts_middle_encoder: Dict[str, Any],
        scale = None,
        **kwargs,
    ):
        super().__init__()
        self.pts_voxel_encoder = build_backbone(pts_voxel_encoder)
        self.pts_middle_encoder = build_backbone(pts_middle_encoder)
        self.scale = scale
        if self.scale is not None:
            if 'type' not in self.scale:
                block = []
                channels = [self.scale['in_channels'], *self.scale['out_channels']]
                # for i in range(self.scale['layer_num']):
                #     block.append(
                #         nn.Conv2d(
                #             channels[i], channels[i+1], 3, stride=self.scale['stride'][i], padding=1
                #         )
                #     )
                #     block.append(nn.BatchNorm2d(channels[i+1]))  
                norm_cfg = dict(type="BN2d", eps=1e-3, momentum=0.01)
                for i in range(self.scale['layer_num']):
                    block.append(
                        nn.Conv2d(
                            channels[i], channels[i+1], 3, stride=self.scale['stride'][i], padding=1
                        )
                    )
                    # block.append(nn.BatchNorm2d(channels[i+1]))
                    block.append(build_norm_layer(norm_cfg, channels[i+1])[1])
        
                    block.append(nn.ReLU(inplace=True))
                self.scale_module = nn.Sequential(*block)
            else:
                self.scale_module = build_backbone(self.scale)
        else:
            self.scale_module = None

    def forward(self, feats, coords, batch_size, sizes):
        x = self.pts_voxel_encoder(feats, sizes, coords)
        x = self.pts_middle_encoder(x, coords, batch_size)
        
        if self.scale_module is not None:
            if 'type' in self.scale:
                x = self.scale_module(x)[0]
            else:
                x = self.scale_module(x)

        return x
