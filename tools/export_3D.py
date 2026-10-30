import argparse
import os
import time
import warnings

import mmcv
import onnx
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmcv.cnn import fuse_conv_bn

from mmdet.datasets import replace_ImageToTensor
from onnxsim import simplify
from tqdm import tqdm
from mmdet3d.utils import recursive_eval
from torchpack.utils.config import configs
from rtx_deep.graph_tracer import graph_utils
import numpy as np
import pdb
from torch.cuda.amp import autocast


def test_time(model, *args, **kwargs):
    time_cost = []
    with torch.no_grad():
        for i in range(100):
            start = time.time()
            torch.cuda.synchronize()

            
            with autocast():
                model(*args, **kwargs)

            torch.cuda.synchronize()
            end = time.time()
            time_cost.append(end - start)
    tc = float(np.array(time_cost[20:]).mean())
    return tc

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("--config", help="test config file path")
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args = parser.parse_args()
    return args


def main():
    device = 'cuda:0'
    args = parse_args()

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # cfg = Config.fromfile(args.config)
    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True

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
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
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
            # print(metas)
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
            # inputs = (points, gt_bboxes_3d,gt_labels_3d,gt_masks_bev,points_labels,coords_xyz,lidar2ego, points_grid_ind,scale_rate,lidar_aug_matrix, metas, img, camera2ego,lidar2camera, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix)
            input_namesin=["img","points","camera2ego","'lidar2ego","lidar2camera","lidar2image","camera_intrinsics","camera2lidar","img_aug_matrix","lidar_aug_matrix", "metas",
             "gt_masks_bev","gt_bboxes_3d","gt_labels_3d","points_labels","points_grid_ind","coords_xyz","scale_rate"]
            
            # for name, item in zip(input_names, inputs):
            #     try:
            #         print(name, item.shape)
            #     except:
            #         print(name)

            from functools import partial
            model.forward = partial(
                model.infer,
            )
            # pdb.set_trace()
            if 'img' in data.keys():
                pytorch_time = test_time(model,*inputs)
            else:
                pytorch_time = test_time(model,*inputs)
            print('PyTorch Model: {0} s;'.format(pytorch_time))

            with autocast():
                # print(model(*inputs))
                model(*inputs)
            return

            torch.onnx.export(
                model,
                *inputs,
                "model.onnx",
                input_names=input_namesin,
                opset_version=13
            )
            model = onnx.load("model.onnx")
            model, _ = simplify(model)
            onnx.save(model, "model.onnx")
            return


if __name__ == "__main__":
    main()
