import argparse
import copy
import os
import pickle

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

class_id = ['person', 'bike', 'car', 'truck', 'bus']

lidar2image = np.array([[ 1.0218831e+03, -9.5876221e+02,  8.0785286e-01, -5.3158140e+03],
                        [ 5.2347321e+02,  1.2985085e+01, -9.4400433e+02, -6.1988336e+02],
                        [ 9.9984211e-01, -6.2910425e-03,  1.6618235e-02, -5.1999998e+00],
                        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])


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
    # parser.add_argument("config", metavar="FILE")#参数文件，是个位置参数，必须有的
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])#默认模式是绘制groundtruth
    parser.add_argument("--checkpoint", type=str, default=None)#没有默认checkpoint,因为默认绘制的是真值
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=0.1)#
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")#可视化文件保存地址，默认是在bevfusion文件夹下面的viz
    args, opts = parser.parse_known_args()#opt里面存储没有定义的参数

    # configs.load(args.config, recursive=True)
    # configs.update(opts)

    # cfg = Config(recursive_eval(configs), filename=args.config)#递归读取参数

    # torch.backends.cudnn.benchmark = cfg.cudnn_benchmark#false
    # torch.cuda.set_device(dist.local_rank())

    # # build the dataloader
    # dataset = build_dataset(cfg.data[args.split])#加载val参数
    # dataflow = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=True,
    #     shuffle=False,
    # )
    import ipdb; ipdb.set_trace()

    #### lidar centerpoint 
    gt_out_range_pkl = "/rbs/houjiayue/code/BEVFusion/hrn_5cls/data/3_L_Y_A_v1.0_for_lidar/mogo_infos_val.pkl"
    v56_2_pred_pkl = "../bevfusion/runs/v56_2_centerpoint_senet_pointpillar_falcon3LYA_2x_yawfocalcls_softmaxCELoss1_resmiddle_smoothL1Loss10_cornerdist2/eval/epoch_60/val/default/result.pkl"
    # v56_2_pred_pkl = "../bevfusion/runs/v56_2_centerpoint_senet_pointpillar_falcon3LYA_2x_yawfocalcls_softmaxCELoss1_resmiddle_smoothL1Loss10_cornerdist2/kitti_models/v56_2_centerpoint_pointpillar_falcon3LYA_2x_senet_vfe1fc_yawfocalcls_softmaxCEloss10_cornerdist2/default/eval/eval_with_train/epoch_200/val/result.pkl"
    
    if args.mode == "pred": 
        with open(v56_2_pred_pkl, 'rb') as det_f:
            datas = pickle.load(det_f)
    if args.mode == "gt":
        with open(gt_out_range_pkl, 'rb') as gt_f:
            datas = pickle.load(gt_f)

    
    #### end 

    
    for data in tqdm(datas):
        if args.mode == "gt" and "gt_boxes" in data:
            bboxes = data["gt_boxes"]
            labels = np.array([class_id.index(name) for name in data['name']])
            lidar_path = data['single_pcd_path']
            

            
        if args.mode == 'pred':
            bboxes = data["pred_lidar"]
            labels = data['pred_labels'] - 1
            lidar_path = data['frame_id']

        # bboxes[..., 2] -= bboxes[..., 5] / 2#0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        bboxes[..., 6] *= -1
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)#mogo数据没有速度，所以维度是7

        # if "img" in data:
        #     for k, image_path in enumerate(metas["filename"]):
        #         image = mmcv.imread(image_path)
        #         visualize_camera(
        #             os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
        #             image,
        #             bboxes=bboxes,
        #             labels=labels,
        #             transform=metas["lidar2image"][k],
        #             classes=cfg.object_classes,
        #         )

        if 1:
            lidar = np.load(lidar_path)
            lidar=lidar[lidar[:,2]<3]
            lidar=lidar[lidar[:,2]>-1]
            lidar=lidar[lidar[:,0]<98.4]
            lidar=lidar[lidar[:,0]>0]
            lidar=lidar[lidar[:,1]<42]
            lidar=lidar[lidar[:,1]>-42]
            name = os.path.basename(lidar_path)[:-4]
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[0,98.4],
                ylim=[-42,42],
                classes=['person', 'bike', 'car', 'truck', 'bus'],
            )


if __name__ == "__main__":
    main()
