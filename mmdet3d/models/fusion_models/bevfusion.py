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

__all__ = ["BEVFusion"]


def conv1x1_bn_act(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        fuser_for_seg: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
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

        if fuser is not None:
            self.fuser = build_fuser(fuser)#ConvFuser
        else:
            self.fuser = None

        if fuser_for_seg is not None:
            self.fuser_for_seg = build_fuser(fuser_for_seg)#ConvFuser
        else:
            self.fuser_for_seg = None

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
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x, mid_feature = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x, mid_feature

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
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
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce: # 将voxel内的10个点求mean
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous() # torch.Size([147059, 5])

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(#代码从这里开始，参数从参数文件中读取
        self,
        points,
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
        # points_labels=None,
        points_grid_ind=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
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
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                # points_labels,
                points_grid_ind,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(#调用这里
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
        # points_labels=None,
        points_grid_ind=None,
        **kwargs,
    ):
        features = []
        mid_lidar_feature=None
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
                feature, x_for_seg = self.extract_lidar_features(points)#encoder-lidar pipeline
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
            x_for_seg = self.fuser_for_seg([x_for_seg, x[0]])
        ##### fuser fusion_feature with mid_lidar_feature for seg end###

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
                    losses = head(x_for_seg, gt_masks_bev)
                    if losses.requires_grad:
                        outputs[f"loss/{type}"] = losses * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}"] = losses
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
                            }
                        )
                elif type == "map":
                    logits = head(x_for_seg)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "seg_logits_bev": logits[k].cpu().argmax(dim=0),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                                # "gt_points_labels": points_labels[k].cpu()
                                # "points_grid_ind": points_grid_ind[k].cpu()
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
            img_aug_matrix=None,
            lidar_aug_matrix=None,
            metas=None,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            # points_labels=None,
            points_grid_ind=None,
            **kwargs,
        ):
            features = []
            mid_lidar_feature=None
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
                    feature, x_for_seg = self.extract_lidar_features(points)#encoder-lidar pipeline
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
                x_for_seg = self.fuser_for_seg([x_for_seg, x[0]])
            ##### fuser fusion_feature with mid_lidar_feature for seg end###
            pred_dict = self.heads['object'](x, metas)
            logits = self.heads['map'](x_for_seg)
            outputs = {
                'det_ret': pred_dict,
                'seg_ret': logits
            }
            return outputs