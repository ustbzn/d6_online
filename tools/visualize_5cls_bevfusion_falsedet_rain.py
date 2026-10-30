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


NAME_MAPPING = {
    # 人
    "Pedestrian": 0,
    "Bicycle": 1,
    # 骑行类
    "Rider": 2,
    # 车
    "Car": 3,
    # 大车（如果你后面能细分再扩展）
    "Truck": 4,
    "Bus": 5,
    # 忽略类（直接过滤）
    "Unknown": 6,
    "Fence": None,
}

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

    # torch.backends.cudnn.benchmark = cfg.cudnn_benchmark#false
    # torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    # dataset = build_dataset(cfg.data[args.split])#加载val参数
    # dataflow = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     shuffle=False,
    # )
    # import ipdb; ipdb.set_trace()
    # build the model and load checkpoint
    # if args.mode == "pred":
        # model = build_model(cfg.model)
        # load_checkpoint(model, args.checkpoint, map_location="cpu")
        # dist
        # model = MMDistributedDataParallel(
        #     model.cuda(),
        #     device_ids=[torch.cuda.current_device()],
        #     broadcast_buffers=False,
        # )

        # model = MMDataParallel(
        #     model.cuda(),
        #     device_ids=[0],
        # )
        # model.eval()

    import pickle as pkl
    # dataflow = pkl.load(open('/pcpt/pcpt/project/atlb/ground_sam3/case_data/20250827142900/log/case.pkl','rb'))['infos']
    dataflow = pkl.load(open('/pcpt/pcpt/project/atlb/ground_sam3/Grounded-SAM-2/single_frame.pkl','rb'))['infos']
    
    # /pcpt/pcpt/project/atlb/ground_sam3/Grounded-SAM-2/single_frame.pkl
    for data in tqdm(dataflow):
        # metas = data["metas"].data[0][0]
        name = "{}".format(data["timestamp"])

        # if args.mode == "pred":
        #     with torch.inference_mode():
        #         outputs = model(**data)

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

        # elif args.mode == "pred" and "boxes_3d" in outputs[0]:
        elif args.mode == "pred" :
            
            orin_bboxes = data["det_boxes"]
            # breakpoint()
            # orin_bboxes[:,2] -= orin_bboxes[:,5] * 0.5
            # scores = data["scores_3d"].numpy()
            orin_labels = data["det_names"]
            bboxes = []
            labels = []
            
            for index, label in enumerate(orin_labels):
                # breakpoint()
                if isinstance(label, np.integer):
                    bboxes.append(orin_bboxes[index])
                    labels.append(label)
                    continue

                if NAME_MAPPING[label] is not None:
                    try:
                        bboxes.append(orin_bboxes[index])
                        labels.append(NAME_MAPPING[label])
                    except:
                        breakpoint()
            # gt_bboxes = outputs[0]["gt_bboxes_3d"].tensor.numpy()
            # gt_labels = outputs[0]["gt_labels_3d"].numpy()
            
            # gt_labels = np.full_like(gt_labels, 6)

            # if args.bbox_classes is not None:
            #     indices = np.isin(labels, args.bbox_classes)
            #     bboxes = bboxes[indices]
            #     scores = scores[indices]
            #     labels = labels[indices]

            # if args.bbox_score is not None:
            #     indices = scores >= args.bbox_score
            #     bboxes = bboxes[indices]
            #     scores = scores[indices]
            #     labels = labels[indices]

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            
            # labels = np.concatenate([labels, gt_labels])
            # bboxes = np.concatenate([bboxes, gt_bboxes])
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)
            
        else:
            bboxes = None
            labels = None
        # breakpoint()
        labels = np.array(labels)

        # if args.mode == "gt" and "gt_masks_bev" in data:
        #     masks = data["gt_masks_bev"].data[0].numpy()
        #     masks = masks.astype(np.bool)
        # elif args.mode == "pred" and "masks_bev" in outputs[0]:
        #     masks = outputs[0]["masks_bev"].numpy()
        #     masks = masks >= args.map_score
        # else:
        masks = None
        visual_class = cfg.object_classes
        visual_class.append('gt')
        import json

        cal_data = json.load(open(data['calib_json_name'], 'r'))

        cam_list = ['60_front', '120_front', '120_back', '120_left', '120_right','left_back', 'right_back']

        for cam_name in cam_list:

            # === 1. 构造外参 ===
            ext = np.array(cal_data[cam_name]['extrinsic'], dtype=np.float32).reshape(3, 4)

            lidar2camera = np.eye(4, dtype=np.float32)
            lidar2camera[:3, :4] = ext

            # ⭐ 关键：inverse
            lidar2camera = np.linalg.inv(lidar2camera)

            # === 2. 内参 ===
            intrinsic = np.array(cal_data[cam_name]['intrinsic'], dtype=np.float32).reshape(3, 3)

            # === 3. 找对应图片 ===
            image_path = data["data_path"].replace('3d_npz', cam_name).replace(".npz", '.jpg')

            if not os.path.exists(image_path):
                print(f"[WARN] image not found: {image_path}")
                continue

            # === 4. 读图 ===
            image = Image.open(image_path)

            # === 5. 可视化 ===
            visualize_camera(
                os.path.join(args.out_dir, cam_name, f"{name}.png"),
                np.array(image),
                bboxes=bboxes,
                labels=labels,
                lidar2camera_single=lidar2camera,
                camera_intrinsics_single=intrinsic,
                img_aug_matrix_single=None,
                classes=visual_class,
            )
            
        # import json
        # cal_data = json.load(open(data['calib_json_name'], 'r'))
        # cam_name = '60_front'   # 或循环多个相机

        # ext = np.array(cal_data[cam_name]['extrinsic'], dtype=np.float32).reshape(3, 4)
        # # lidar2camera = np.eye(4, dtype=np.float32)
        # # lidar2camera[:3, :4] = ext
        # lidar2camera = np.eye(4, dtype=np.float32)
        # lidar2camera[:3, :4] = ext

        # # ⭐ 核心修复
        # lidar2camera = np.linalg.inv(lidar2camera)

        # intrinsic = np.array(cal_data[cam_name]['intrinsic'], dtype=np.float32).reshape(3, 3)
        # # breakpoint()
        
        # front_60_img = [data["data_path"].replace('3d_npz', cam_name).replace(".npz",'.jpg')]
        # if  True :
        #     for k, image_path in enumerate(front_60_img):
        #         image = Image.open(image_path)
        #         visualize_camera(
        #             os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
        #             np.array(image),
        #             bboxes=bboxes,
        #             labels=labels,
        #             lidar2camera_single = lidar2camera,
        #             camera_intrinsics_single = intrinsic,
        #             img_aug_matrix_single = None,
        #             classes=visual_class,
        #             # thickness=1,
        #         )

        # if "points" in data:
        if True:
            lidar_name = data['data_path']
            lidar = np.load(lidar_name)['pcds']
            # lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=visual_class,
            )
        # breakpoint()

        # if masks is not None:
        #     visualize_map(
        #         os.path.join(args.out_dir, "map", f"{name}.png"),
        #         masks,
        #         classes=cfg.map_classes,
        #     )

        # open3d viz
        # gt_bboxes = None
        # pred_bboxes = None
        # if args.mode == "gt" and "gt_bboxes_3d" in data:
        #     gt_bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
        # elif args.mode == "pred" and "boxes_3d" in outputs[0]:
        #     pred_bboxes = outputs[0]["boxes_3d"].tensor.numpy()
                
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
