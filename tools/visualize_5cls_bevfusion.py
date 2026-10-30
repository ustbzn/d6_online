import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
# from mmdet3d.core.visualizer import show_result
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from PIL import Image

def read_image_preserve_color(image_path):
    """读取图片并尽量保持原始颜色"""
    img = Image.open(image_path)
    
    # 查看原始信息
    print(f"原始模式: {img.mode}")
    print(f"原始格式: {img.format}")
    
    # 根据模式处理
    if img.mode == 'CMYK':
        # CMYK 转 RGB
        img = img.convert('RGB')
    elif img.mode == 'RGBA':
        # 如果需要保留透明度
        img = img.convert('RGBA')
    elif img.mode in ('P', 'L'):
        # 调色板或灰度图
        img = img.convert('RGB')
    
    return img


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
    # dist.init()

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
    # torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])#加载val参数
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        shuffle=False,
    )
    # import ipdb; ipdb.set_trace()
    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        # dist
        # model = MMDistributedDataParallel(
        #     model.cuda(),
        #     device_ids=[torch.cuda.current_device()],
        #     broadcast_buffers=False,
        # )

        model = MMDataParallel(
            model.cuda(),
            device_ids=[0],
        )
        model.eval()

    for data in tqdm(dataflow):
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

            # labels[labels==2] = 1
            # labels[labels==3] = 1
            # labels[labels==4] = 2
            # labels[labels==5] = 3
            # labels[labels==6] = 4

            bboxes = bboxes[labels > -1]
            labels = labels[labels > -1]

            # bboxes[..., 2] -= bboxes[..., 5] / 2#0.5, 0.5, 0.5 -> 0.5, 0.5, 0
            # print("bboxes.size() = ", bboxes.shape)#8*7,应该维度为9,差了速度
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)#mogo数据没有速度，所以维度是7

        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()
            
            gt_bboxes = outputs[0]["gt_bboxes_3d"].tensor.numpy()
            gt_labels = outputs[0]["gt_labels_3d"].numpy()
            
            gt_labels = np.full_like(gt_labels, 6)

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

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            
            labels = np.concatenate([labels, gt_labels])
            bboxes = np.concatenate([bboxes, gt_bboxes])
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
        visual_class = cfg.object_classes
        visual_class.append('gt')
        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                # image = mmcv.imread(image_path)
                image = Image.open(image_path)
                # breakpoint()
                if k == 0:
                    image = image.resize((3840, 2160))
                else:
                    image = image.resize((1920, 1536))
                    
                # # ========== 调试代码开始 ==========
                # print(f"=== 图像 {k} 调试 ===")
                # print(f"原始路径: {image_path}")
                # print(f"PIL 模式: {image.mode}")
                # print(f"PIL 大小: {image.size}")
                
                # # 转换为 numpy 检查
                # img_np = np.array(image)
                # print(f"Numpy 形状: {img_np.shape}")
                # print(f"Numpy 类型: {img_np.dtype}")
                # print(f"值范围: [{img_np.min()}, {img_np.max()}]")
                
                # # 检查前几个像素
                # print(f"左上角像素 (5,5): {img_np[5,5]}")
                # print(f"中心像素: {img_np[img_np.shape[0]//2, img_np.shape[1]//2]}")
                
                # # 保存原始图像用于对比
                # temp_path = f"/tmp/original_{k}.png"
                # image.save(temp_path)
                # print(f"原始图像已保存到: {temp_path}")
                # # ========== 调试代码结束 ==========
                # breakpoint()
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    np.array(image),
                    bboxes=bboxes,
                    labels=labels,
                    lidar2camera_single = data['lidar2camera'].data[0][0][k].numpy(),
                    camera_intrinsics_single = data['camera_intrinsics'].data[0][0][k].numpy(),
                    img_aug_matrix_single = None,
                    classes=visual_class,
                    # thickness=1,
                )
    
                # visualize_camera(
                #     os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                #     np.array(image),
                #     bboxes=bboxes,
                #     labels=labels,
                #     transform=metas["lidar2image"][k],
                #     classes=visual_class,
                #     # thickness=1,
                # )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=visual_class,
            )

        # if masks is not None:
        #     visualize_map(
        #         os.path.join(args.out_dir, "map", f"{name}.png"),
        #         masks,
        #         classes=cfg.map_classes,
        #     )

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
