import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
# from mmdet3d.core.visualizer import show_result
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):#递归读取参数
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")#参数文件，是个位置参数，必须有的
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])#默认模式是绘制groundtruth
    parser.add_argument("--checkpoint", type=str, default=None)#没有默认checkpoint,因为默认绘制的是真值
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=0.1)#
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")#可视化文件保存地址，默认是在bevfusion文件夹下面的viz
    args, opts = parser.parse_known_args()#opt里面存储没有定义的参数

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)#递归读取参数

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark#false
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])#加载val参数
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )
    # import ipdb; ipdb.set_trace()
    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    for data in tqdm(dataflow):
        # import ipdb; ipdb.set_trace()
        metas = data["metas"].data[0][0]
        name = "{}".format(metas["timestamp"])

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2#0.5, 0.5, 0.5 -> 0.5, 0.5, 0
            # print("bboxes.size() = ", bboxes.shape)#8*7,应该维度为9,差了速度
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)#mogo数据没有速度，所以维度是7
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )

        # open3d viz
        gt_bboxes = None
        pred_bboxes = None
        if args.mode == "gt" and "gt_bboxes_3d" in data:
            gt_bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            pred_bboxes = outputs[0]["boxes_3d"].tensor.numpy()
                
        # show_result(
        #     lidar,
        #     gt_bboxes = gt_bboxes,
        #     pred_bboxes = pred_bboxes,
        #     out_dir = os.path.join(args.out_dir, "lidar_m"),
        #     filename = name,
        #     show = True
        #     )

if __name__ == "__main__":
    main()
