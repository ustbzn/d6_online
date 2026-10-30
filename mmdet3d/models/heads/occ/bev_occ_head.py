# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
import math
# from ..losses.semkitti_loss import sem_scal_loss, geo_scal_loss
# from ..losses.lovasz_softmax import lovasz_softmax

__all__ = ["BEVOCCHead2D"]

nusc_class_frequencies = np.array([
    3508849,
    1355,
    1347,
    1407,
    35746,
    1776,
    1024,
    1745,
    36178,
    25241,
    2614,
    4065,
    1104,
    86165,
])

@HEADS.register_module()
class BEVOCCHead2D(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=21,
                 use_predicter=True,
                 class_balance=False,
                 ohem = None,
                 loss_ce = None,
                 loss_lovasz = None,
                 loss_geo = None,
                 loss_sem = None,
                 ):
        
        super(BEVOCCHead2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                # nn.Softplus(), 
                nn.ReLU(inplace=True),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.class_balance = class_balance
        self.ohem = ohem
        
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            loss_ce['class_weight'] = class_weights
            self.cls_weights = class_weights
        # else:
        #     class_weights = torch.ones(num_classes)/num_classes  # FIXME hardcode 17
        #     loss_ce['class_weight'] = class_weights
        #     self.cls_weights = class_weights
        
        self.loss_ce = build_loss(loss_ce)
        self.loss_lovasz = build_loss(loss_lovasz)
        self.loss_geo = build_loss(loss_geo)
        self.loss_sem = build_loss(loss_sem)

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:

        """
        # (B, C, Dx, Dy) --> (B, C, Dx, Dy) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 2, 3, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss_occ = dict()
        voxel_semantics = voxel_semantics.long()
        
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)

            loss_ce = self.loss_ce(preds, voxel_semantics)
            if self.ohem is not None:
                loss_ce = self.get_ohem_loss(
                    loss_mat=loss_ce,
                    # valid_mask=(voxel_semantics >= 0),  # 或你的有效 voxel mask
                    top_ratio=self.ohem.top_ratio,
                    top_weight=self.ohem.top_weight
                ) * self.ohem.loss_weight_ohem
            
            loss_lovasz = self.loss_lovasz(torch.softmax(preds, dim=1), voxel_semantics)
            loss_geo = self.loss_geo(preds, voxel_semantics)
            loss_sem = self.loss_sem(preds, voxel_semantics)
            # ## grid_required = false
            occ_iou = self.compute_occupancy_iou(preds, voxel_semantics, mask_camera)
            
            loss_occ.update({"loss_ce": loss_ce})
            loss_occ.update({"loss_lovasz": loss_lovasz})
            loss_occ.update({"loss_geo": loss_geo})
            loss_occ.update({"loss_sem": loss_sem})
            loss_occ.update({"occ_iou": occ_iou})

        else:   # invalid#####################################################
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = len(voxel_semantics)

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                avg_factor=num_total_samples
            )
            occ_iou = self.compute_occupancy_iou(preds, voxel_semantics)
                # invalid#####################################################
        return loss_occ

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        # occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return occ_res

    @torch.no_grad()
    def compute_occupancy_iou(self, occ_pred, voxel_semantics, mask_camera=None):
        """
        计算每个类别 IoU 以及 mIoU
        occ_pred: (B, Dx, Dy, Dz, n_cls) logits
        voxel_semantics: (B, Dx, Dy, Dz)
        mask_camera: optional (B, Dx, Dy, Dz)
        """
        pred_labels = occ_pred.argmax(dim=-1)  # (B, Dx, Dy, Dz)

        voxel_semantics = voxel_semantics.long()

        if mask_camera is not None:
            mask = mask_camera.bool()
            pred_labels = pred_labels[mask]
            voxel_semantics = voxel_semantics[mask]
        else:
            pred_labels = pred_labels.reshape(-1)
            voxel_semantics = voxel_semantics.reshape(-1)

        ious = []
        for cls_id in range(self.num_classes):
            pred_cls = (pred_labels == cls_id)
            gt_cls = (voxel_semantics == cls_id)

            intersection = (pred_cls & gt_cls).sum().float()
            union = (pred_cls | gt_cls).sum().float()

            if union == 0:
                ious.append(float('nan'))  # 该类不存在，忽略
            else:
                ious.append((intersection / union).item())

        # 计算 mIoU（忽略 nan）
        valid_ious = [x for x in ious if not math.isnan(x)]
        miou = sum(valid_ious) / len(valid_ious) if len(valid_ious) > 0 else 0.0
        
        miou = torch.tensor(miou,device=occ_pred.device)
        return miou
    
    # valid_mask ignore_lable 要注意使用
    def get_ohem_loss(self, loss_mat, valid_mask=None, top_ratio=0.2, top_weight=3.0):
        if valid_mask is not None:
            loss_flat = loss_mat[valid_mask.bool()]
        else:
            loss_flat = loss_mat.view(-1)

        valid_num = loss_flat.numel()
        if valid_num == 0:
            return loss_mat.sum() * 0.0  # safe zero

        loss_mean = loss_flat.mean()

        topk_num = int(valid_num * top_ratio)
        if topk_num <= 0:
            return loss_mean

        loss_topk = torch.topk(
            loss_flat,
            k=topk_num,
            largest=True,
            sorted=False
        )[0]

        return loss_mean + top_weight * loss_topk.mean()