from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel
from rtx_deep_plugin import Grid2Point, VoxelMaxPool, max_op
import ipdb
import numpy as np
import os

__all__ = ["BEVFusion"]


def conv1x1_bn_act(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

import torch
import torch.nn as nn
from torchvision.models import resnet18

class HDMapBEVResNet18(nn.Module):
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        
        # -----------------------------
        # 1. ResNet18 Backbone
        # -----------------------------
        resnet = resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # stride=2 → total stride=4
            resnet.layer1,   # stride=1
            resnet.layer2,   # stride=2 → total stride=8
        )
        self.out_conv = nn.Conv2d(128, out_channels, kernel_size=1)
        
        # -----------------------------
        # 2. Gated Fusion
        # -----------------------------
        self.map_proj = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.alpha_conv = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)
        nn.init.constant_(self.alpha_conv.weight, 0.0)
        nn.init.constant_(self.alpha_conv.bias, 0.1)
        
    def forward(self, bev_feat, hdmap_bev):
        """
        bev_feat: (B, C_bev, H, W) -> 感知 BEV
        hdmap_bev: (B, 4, H_hdmap, W_hdmap) -> HDMap BEV (原始分辨率)
        """
        # -----------------------------
        # 1. ResNet 提取 HDMap 特征
        # -----------------------------
        hdmap_feat = self.backbone(hdmap_bev)  # [B, 128, H/8, W/8]
        hdmap_feat = self.out_conv(hdmap_feat)  # [B, out_channels, H/8, W/8]

        # 如果 bev_feat 分辨率不同，需要上采样到 hdmap_feat 的大小
        if bev_feat.shape[2:] != hdmap_feat.shape[2:]:
            bev_feat = nn.functional.interpolate(bev_feat, size=hdmap_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # -----------------------------
        # 2. 计算 alpha gate
        # -----------------------------
        alpha = torch.sigmoid(self.alpha_conv(hdmap_feat))  # [B,1,H/8,W/8]

        # -----------------------------
        # 3. map_proj
        # -----------------------------
        hdmap_proj = self.map_proj(hdmap_feat)              # [B, C_bev, H/8, W/8]

        # -----------------------------
        # 4. gated 融合
        # -----------------------------
        hdmap_feat_gated = alpha * hdmap_proj
        bev_fused = torch.cat([bev_feat, hdmap_feat_gated], dim=1)       # [B, C_bev, H/8, W/8]
        
        return bev_fused



@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        fuser_for_seg: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        fuser_for_occ: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),#SwinTransformer
                    "neck": build_neck(encoders["camera"]["neck"]),#GeneralizedLSSFPN
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),#DepthLSSTransform
                }
            )#从参数文件中读取camera pipeline的模型
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:#10
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])#构建体素栅格
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),#SparseEncoder
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)#体素降采样参数，一般是false
            
        self.hdmap = HDMapBEVResNet18(in_channels=4, out_channels=64)

        if fuser is not None:
            self.fuser = build_fuser(fuser)#ConvFuser
        else:
            self.fuser = None

        if fuser_for_seg is not None:
            self.fuser_for_seg = build_fuser(fuser_for_seg)#ConvFuser
        else:
            self.fuser_for_seg = None
            
        if fuser_for_occ is not None:
            self.fuser_for_occ = build_fuser(fuser_for_occ)#ConvFuser
        else:
            self.fuser_for_occ = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),#SECOND
                "neck": build_neck(decoder["neck"]),#SECONDFPN
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:#包含map和detect两种，即语义分割和目标检测
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])
        self.testsavepcd=False
        # self.testsavepcd=kwargs["testsavepcd"]
        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # self.conv1x1 = conv1x1_bn_act(704, 256)  # compress mid_lidar_feature
        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()#camera主干网络使用预训练模型

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()#B-batch size, N-camera numbers, C-rgb为3, H高度900， W宽度1600
        x = x.view(B * N, C, H, W)#降维

        # with torch.no_grad():
        x = self.encoders["camera"]["backbone"](x)#tuple, len(x)=3, [B * N, 512, 32, 88],[B * N, 1024, 16, 44],[B * N, 2048, 8, 22]，8倍、16倍、32倍的降采样
        x = self.encoders["camera"]["neck"](x)#tuple, len(x)=2, [B * N, 256, 32, 88],[B * N, 256, 16, 44]

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,#points per samples,但是图像pipeline点云字段是空的
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    # @torch.no_grad()
    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x) # torch.Size([186019, 4]) torch.Size([186019, 4]) torch.Size([186019])
        batch_size = coords[-1, 0] + 1
        # x, mid_feature = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

        # return x, mid_feature
    def extract_hdmap_features(self, bev_feat, hdmap_feat):
        return self.hdmap(bev_feat, hdmap_feat)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                '''
                  voxels: [M, max_points, ndim] float tensor. only contain points
                  and returned when max_points != -1.     //ndim=4  =point.size(1)
                  coordinates: [M, 3] int32 tensor, always returned.
                  num_points_per_voxel: [M] int32 tensor. Only returned when
                  max_points != -1.
                '''
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0) # [bs_id, h, w, z]
        # ipdb.set_trace()
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce: # 将voxel内的10个点求mean
                feats1 = feats[:,:,:3].sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats2 = torch.max(feats[:,:,3], dim=1, keepdim=False)[0]
                feats = torch.cat([feats1, feats2.unsqueeze(1)], dim=-1)
                feats = feats.contiguous() # torch.Size([147059, 5])

        return feats, coords, sizes
    
    # @torch.no_grad()
    # @force_fp32()
    # def voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret = self.encoders["lidar"]["voxelize"](res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode="constant", value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
    #                 -1, 1
    #             )
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(#代码从这里开始，参数从参数文件中读取
        self,
        points,
        hdmap_bev,
        lidar2ego,
        lidar_aug_matrix,
        metas,
        img=None,
        camera2ego=None,
        lidar2camera=None,
        lidar2image=None,
        camera_intrinsics=None,
        camera2lidar=None,
        img_aug_matrix=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        points_labels=None,
        points_grid_ind=None,
        coords_xyz=None,
        scale_rate=None,
        voxel_semantics=None,
        mask_camera=None,
        **kwargs,
    ):
        # ipdb.set_trace()
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                hdmap_bev,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                points_labels,
                points_grid_ind,
                coords_xyz,
                scale_rate,
                voxel_semantics,   
                mask_camera,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(#调用这里
        self,
        img,
        points,
        hdmap_bev,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        points_labels=None,
        points_grid_ind=None,
        coords_xyz=None,
        scale_rate=None,
        voxel_semantics=None,
        mask_camera=None,
        **kwargs,
    ):
        # breakpoint()
        # with open('/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/camera_input.bin', 'wb') as f:
        #     f.write(img[0].to(torch.float16).cpu().numpy().tobytes())   # 二进制块
        # # 把形状/dtype 记到同名文本文件，方便恢复
        # with open('/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/camera_input.meta', 'w') as f:
        #     f.write(f"{tuple(img[0].shape)}\n{img[0].dtype}\n")
        # breakpoint()
        # with open('/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/lidar_input.bin', 'wb') as f:
        #     f.write(points[0].to(torch.float16).cpu().numpy().tobytes())   # 二进制块
        # # 把形状/dtype 记到同名文本文件，方便恢复
        # with open('/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/lidar_input.meta', 'w') as f:
        #     f.write(f"{tuple(points[0].shape)}\n{points[0].dtype}\n")
        # breakpoint()
        # for save orin_img  zhuning
        # self.save_img_and_img_aug(img, lidar2camera, camera_intrinsics, img_aug_matrix, gt_bboxes_3d, gt_labels_3d, metas)
        features = []
        mid_lidar_feature=None
        # points_labels = points_labels.unsqueeze(2)
        
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]#如果self.training（这个变量在哪里定义的？）为true，则返回encoders字典，如果为false,则返回encoders的keys倒序遍历
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(#encoder-camera pipeline torch.Size([4, 80, 164, 140])
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            elif sensor == "lidar":
                # feature, x_for_seg = self.extract_lidar_features(points)#encoder-lidar pipeline
                feature = self.extract_lidar_features(points)#encoder-lidar pipeline

                # split feature_concat
                # N, C, H, W = feature.shape
                # mid_lidar_feature = feature[:, 256:, :, :].view(N, -1, 2*H, 2*W)
                # feature = feature[:, :256, :, :]
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]
        # breakpoint()
        # with open('/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/fuser_camera_input.bin', 'wb') as f:
        #     f.write(features[0].to(torch.float16).cpu().numpy().tobytes())   # 二进制块
        # # 把形状/dtype 记到同名文本文件，方便恢复
        # with open('/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/fuser_camera_input.meta', 'w') as f:
        #     f.write(f"{tuple(features[0].shape)}\n{features[0].dtype}\n")

        # with open('/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/fuser_lidar_input.bin', 'wb') as f:
        #     f.write(features[1].to(torch.float16).cpu().numpy().tobytes())   # 二进制块
        # # 把形状/dtype 记到同名文本文件，方便恢复
        # with open('/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/fuser_lidar_input.meta', 'w') as f:
        #     f.write(f"{tuple(features[1].shape)}\n{features[1].dtype}\n")
        # breakpoint()
        if self.fuser is not None:
            x = self.fuser(features)#特征融合
        else:
            assert len(features) == 1, features
            x = features[0]
            
        hdmap_bev_encoded = None
        if hdmap_bev != None:
            hdmap_bev_encoded = self.extract_hdmap_features(x, hdmap_bev)
        x = hdmap_bev_encoded
        
        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)#decoder
        x = self.decoder["neck"](x)
        # TODO: 3D decoder

        ##### fuser fusion_feature with mid_lidar_feature for seg ###
        if self.fuser_for_seg is not None:
            # mid_lidar_feature 704->256
            # x_for_seg = self.conv1x1(x_for_seg)
            coords_wl = coords_xyz[:, :, :2].contiguous()
            x_for_seg = self.fuser_for_seg(x[0])
            
            g2p_points = Grid2Point(x_for_seg, coords_wl, scale_rate=(0.25, 0.25))
        ##### fuser fusion_feature with mid_lidar_feature for seg end###
        
        if self.fuser_for_occ is not None:
            x_for_occ = self.fuser_for_occ(x[0]) # × 4

        if self.training:
            outputs = {}
            for type, head in self.heads.items():#遍历所有的head
                if type == "object":#检测头
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)#loss
                    for name, val in losses.items():
                        if val.requires_grad:
                            outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]#反向传播
                        else:
                            outputs[f"stats/{type}/{name}"] = val
                elif type == "map":#分割头
                    # losses = head(x_for_seg, gt_masks_bev)
                    expanded_tensors = [tensor.unsqueeze(-1) for tensor in points]

                    stacked_tensor = torch.stack(expanded_tensors)

                    final_tensor = stacked_tensor.permute(0, 2, 1, 3)
                    losses = head(final_tensor,g2p_points, points_labels) # 点分割
                    if losses.requires_grad:
                        outputs[f"loss/{type}"] = losses * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}"] = losses
                        
                elif type == 'occ':
                    occ_outs = head(x_for_occ) # (B, 128, 496, 496)
                    losses = head.loss(
                        occ_outs,  # (B, Dx, Dy, Dz, n_cls)
                        voxel_semantics,  # (B, Dx, Dy, Dz)
                        mask_camera,  # (B, Dx, Dy, Dz)
                    )

                    for name, val in losses.items():
                        if val.requires_grad:
                            outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]#反向传播
                        else:
                            outputs[f"stats/{type}/{name}"] = val
                else:#nothing
                    raise ValueError(f"unsupported head: {type}")
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                                "gt_bboxes_3d": gt_bboxes_3d[k].to("cpu"),
                                "gt_labels_3d": gt_labels_3d[k].cpu(),
                            }
                        )
                elif type == "map":
                    expanded_tensors = [tensor.unsqueeze(-1) for tensor in points]

                    stacked_tensor = torch.stack(expanded_tensors)

                    final_tensor = stacked_tensor.permute(0, 2, 1, 3)

                    logits = head(final_tensor,g2p_points)
                    one_of = logits[0, :, 10000]
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "seg_logits_bev": True,
                                "pred_points_labels": logits[k].cpu().argmax(dim=0),
                                # "gt_masks_bev": gt_masks_bev[k].cpu(),
                                "gt_points_labels": points_labels[k].cpu(),
                                # "points_grid_ind": points_grid_ind[k].cpu()
                            }
                        )
                        if self.testsavepcd:
                            outputs[k].update(
                                {
                                    "pcds": final_tensor[k].cpu()
                                }
                            )
                elif type == 'occ':
                    occ_outs = head(x_for_occ) # (B, 128, 496, 496)
                    occ_preds = head.get_occ(occ_outs)

                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "occ_pred_labels": occ_preds.cpu(),
                                "voxel_semantics": voxel_semantics[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
    def infer(#调用这里
            self,
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            points_labels=None,
            points_grid_ind=None,
            coords_xyz=None,
            scale_rate=None,
            **kwargs,
        ):
            features = []
            mid_lidar_feature=None
            # points_labels = points_labels.unsqueeze(2)
            for sensor in (
                self.encoders if self.training else list(self.encoders.keys())[::-1]#如果self.training（这个变量在哪里定义的？）为true，则返回encoders字典，如果为false,则返回encoders的keys倒序遍历
            ):
                if sensor == "camera":
                    feature = self.extract_camera_features(#encoder-camera pipeline torch.Size([4, 80, 164, 140])
                        img,
                        points,
                        camera2ego,
                        lidar2ego,
                        lidar2camera,
                        lidar2image,
                        camera_intrinsics,
                        camera2lidar,
                        img_aug_matrix,
                        lidar_aug_matrix,
                        metas,
                    )
                elif sensor == "lidar":
                    feature = self.extract_lidar_features(points)#encoder-lidar pipeline
                    # split feature_concat
                    # N, C, H, W = feature.shape
                    # mid_lidar_feature = feature[:, 256:, :, :].view(N, -1, 2*H, 2*W)
                    # feature = feature[:, :256, :, :][:,256,222,120]
                else:
                    raise ValueError(f"unsupported sensor: {sensor}")
                features.append(feature)

            if not self.training:
                # avoid OOM
                features = features[::-1]

            if self.fuser is not None:
                x = self.fuser(features)#特征融合
            else:
                assert len(features) == 1, features
                x = features[0]

            batch_size = x.shape[0]

            x = self.decoder["backbone"](x)#decoder
            x = self.decoder["neck"](x)
            # TODO: 3D decoder

            ##### fuser fusion_feature with mid_lidar_feature for seg ###
            if self.fuser_for_seg is not None:
                # mid_lidar_feature 704->256
                # x_for_seg = self.conv1x1(x_for_seg)

                coords_wl = coords_xyz[:, :, :2].contiguous()
                x_for_seg = self.fuser_for_seg(x[0])
                g2p_points = Grid2Point(x_for_seg, coords_wl, scale_rate=(0.25, 0.25))
            ##### fuser fusion_feature with mid_lidar_feature for seg end###

            pred_dict = self.heads['object'](x, metas)
            bboxes = self.heads['object'].get_bboxes(pred_dict, metas)

            expanded_tensors = [tensor.unsqueeze(-1) for tensor in points]
            stacked_tensor = torch.stack(expanded_tensors)
            final_tensor = stacked_tensor.permute(0, 2, 1, 3)

            logits = self.heads['map'](final_tensor,g2p_points)
            outputs = {
                'det_ret': pred_dict,
                'seg_ret': logits
            }
            return outputs
    
    def save_img_and_img_aug(self, img, lidar2camera, camera_intrinsics, img_aug_matrix, gt_bboxes_3d, gt_labels_3d,metas):
        img = np.array([np.array(rgb_img) for rgb_img in metas[0]['rgb_img']])
        origin_img = np.array([np.array(origin_rgb_img) for origin_rgb_img in metas[0]['origin_rgb_img']])
        lidar_path = metas[0]['lidar_path']
        # img = img[0].cpu().numpy()
        # breakpoint()
        lidar2camera = lidar2camera[0].cpu().numpy()
        camera_intrinsics = camera_intrinsics[0].cpu().numpy()
        img_aug_matrix = img_aug_matrix[0].cpu().numpy()
        gt_bboxes_3d = gt_bboxes_3d[0].tensor.cpu().numpy()
        gt_labels_3d = gt_labels_3d[0].cpu().numpy()

        npz_name = lidar_path.split('/')[-1]
        scenes_name = lidar_path.split('/')[-3]
        save_dir = os.path.join('/pcpt/pcpt/project/atlb/model2_camera/multimodal_bevfusion','debug_camera_intrinsics/debug_pose_matrix', scenes_name)

        # 创建目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, npz_name)
        
        npz_dict = {
            'img': img,
            'origin_img': origin_img,
            'lidar2camera': lidar2camera,
            'camera_intrinsics': camera_intrinsics,
            'img_aug_matrix': img_aug_matrix,
            'lidar_path': lidar_path,
            'gt_bboxes_3d' : gt_bboxes_3d,
            'gt_labels_3d' : gt_labels_3d,
        }
        
        # 保存npz文件
        np.savez(save_path, **npz_dict)
        print(f"Saved debug data to: {save_path}") 