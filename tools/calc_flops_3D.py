# coding=utf-8
'''
Author: tuchaoping
License: Apache Licence
Software: VSCode
Date: 2024-08-01 09:54:29
LastEditors: tuchaoping
LastEditTime: 2024-08-08 07:52:41
'''
import argparse
from thop import profile
from thop import clever_format
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmdet3d.utils import recursive_eval
from torchpack.utils.config import configs
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.cnn import fuse_conv_bn
from mmdet.datasets import replace_ImageToTensor
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from torch.nn.parallel import DataParallel
import pdb

if __name__ == '__main__':
    device = 'cuda:0'
    parser = argparse.ArgumentParser(description="MMDet test flops a model")
    parser.add_argument("--config", help="test config file path")

    args = parser.parse_args()
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    cfg.model.pretrained = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    cfg.data.test.test_mode = True
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    # fp16_cfg = cfg.get("fp16", None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    # print(checkpoint.get('meta'))

    model = fuse_conv_bn(model)

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    model.CLASSES = dataset.CLASSES
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for data in data_loader:
            # print(data.keys())
            if 'img' in data.keys():
                img = data["img"]._data[0].to(device)
                camera2ego = data["camera2ego"]._data[0].to(device)
                lidar2camera = data["lidar2camera"]._data[0].to(device)
                lidar2image = data["lidar2image"]._data[0].to(device)
                camera_intrinsics = data["camera_intrinsics"]._data[0].to(device)
                camera2lidar = data["camera2lidar"]._data[0].to(device)
                img_aug_matrix = data["img_aug_matrix"]._data[0].to(device)
            # pdb.set_trace()
            gt_bboxes_3d = data["gt_bboxes_3d"]._data[0][0].tensor.to(device)
            gt_labels_3d = data["gt_labels_3d"]._data[0][0].to(device)
            gt_masks_bev = data["gt_masks_bev"].to(device)
            points_labels = data["points_labels"].to(device)
            coords_xyz = data["coords_xyz"].to(device)
            points_grid_ind = data["points_grid_ind"]._data[0][0].to(device)
            scale_rate = data["scale_rate"][0].to(device)

            points = torch.stack(data["points"]._data[0], dim=0).to(device)
            lidar2ego = data["lidar2ego"]._data[0].to(device)
            lidar_aug_matrix = data["lidar_aug_matrix"]._data[0].to(device)
            metas = data["metas"]._data[0]
            from functools import partial
            model.forward = partial(
                model.infer,
            )
            # pdb.set_trace()
            if 'img' in data.keys():
                inputs = (img,points,camera2ego,lidar2ego,lidar2camera,lidar2image,camera_intrinsics,camera2lidar,img_aug_matrix,lidar_aug_matrix, metas,
                                    gt_masks_bev,gt_bboxes_3d,gt_labels_3d,points_labels,points_grid_ind,coords_xyz,scale_rate)
            else:
                img=None
                camera2ego=None
                lidar2camera=None
                lidar2image=None
                camera_intrinsics=None
                camera2lidar=None
                img_aug_matrix=None
                inputs = (img,points,camera2ego,lidar2ego,lidar2camera,lidar2image,camera_intrinsics,camera2lidar,img_aug_matrix,lidar_aug_matrix, metas,
                                    gt_masks_bev,gt_bboxes_3d,gt_labels_3d,points_labels,points_grid_ind,coords_xyz,scale_rate)

            # from torchstat import stat
            # # 计算模型的 FLOPs 和参数量
            # stat(model, inputs)
            flops, params = profile(model, inputs=inputs)
            flops, params = clever_format([flops, params], '%.3f')
            print("flops:{}, params:{}".format(flops, params))
            print('done')
            break

    # for data in data_loader:
    #     # print(data)
        
    #     inputs = data
   
    #     flops, params = profile(model, inputs=inputs)
    #     flops, params = clever_format([flops, params], '%.3f')
    #     print("flops:{}, params:{}".format(flops, params))
    #     print('done')
    #     break
