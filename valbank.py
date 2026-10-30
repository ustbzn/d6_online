# coding=utf-8
'''
Author: tuchaoping
License: Apache Licence
Software: VSCode
Date: 2024-10-08 06:55:52
LastEditors: tuchaoping
LastEditTime: 2024-10-17 07:26:22
'''
import os
import yaml

import functools
import multiprocessing
from tqdm import tqdm
import pdb
import numpy as np
from scipy.spatial import cKDTree, Delaunay
import torch
import sys
import glob
sys.path.insert(0, sys.path[0]+"../detsegflow")

def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0
def compute_box_3d(center, size, yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    
    # 3d bounding box dimensions
    l = size[0]
    w = size[1]
    h = size[2]
    
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    return corners_3d.T

def filter_box(box3d, pcd, point_cloud_range=[-102.4, -51.2, -3.0, 102.4, 51.2, 5.0], filter_nums=10):
    box_mask_list = []
    for box in box3d:
        center = box[:3]
        size = box[3:6]
        if center[0] < point_cloud_range[0] or \
           center[1] < point_cloud_range[1] or \
           center[2] < point_cloud_range[2]:
            box_mask_list.append(False)
            continue
        if center[0] > point_cloud_range[3] or \
           center[1] > point_cloud_range[4] or \
           center[2] > point_cloud_range[5]:
            box_mask_list.append(False)
            continue
        box_corner = compute_box_3d(box[:3], box[3:6], box[8])
        obj_mask = in_hull(pcd[:, :3], box_corner)
        pts_nums = obj_mask.sum()
        if pts_nums >= filter_nums:
            box_mask_list.append(True)
        else:
            box_mask_list.append(False)
    box_mask = np.array(box_mask_list).astype(bool)
    return box_mask

def calc_iou(gt_boxes, pt_boxes):
    from utils import nms
    '''
    calc iou in cuda and tensor!!
    '''
    if isinstance(gt_boxes, np.ndarray):
        gt_boxes = torch.from_numpy(gt_boxes)
    if isinstance(pt_boxes, np.ndarray):
        pt_boxes = torch.from_numpy(pt_boxes)
    ious = nms.boxes_iou_bev(gt_boxes.cuda(), pt_boxes.cuda())
    ious = ious.cpu().data.numpy()
    return ious


def save_boxes_bank(gt_box3d, gt_label, pcd, pcd_label, task_cfg, token, fpath_mogo_bank):
    for i in range(gt_box3d.shape[0]):
        box = gt_box3d[i]
        sem_label = int(gt_label[i])
        sem_label_name = task_cfg['label_map_inv'][sem_label]
        if sem_label_name in class_set:
            box_corner = compute_box_3d(box[:3], box[3:6], box[6])
            obj_mask = in_hull(pcd[:, :3], box_corner)
            pcds_obj = pcd[obj_mask]
            pcds_label_obj = pcd_label[obj_mask]
            label_list = np.unique(pcds_label_obj).tolist()

            for label in label_list:
                if sem_label == label:
                    obj_refine_mask = (pcds_label_obj == label)
                    pcds_obj_refine = pcds_obj[obj_refine_mask]
                    pcds_label_obj_refine = pcds_label_obj[obj_refine_mask]
                    pcds_lidarid = np.zeros_like(pcds_label_obj_refine)
                    fname_obj = os.path.join(fpath_mogo_bank, sem_label_name, "{0}##{1}.npz".format(token, i))

                    np.savez_compressed(fname_obj, 
                                        pcds=pcds_obj_refine, 
                                        pcds_lidarid=pcds_lidarid, 
                                        cate_id=sem_label, 
                                        cate=sem_label_name,
                                        center=gt_box3d[i, :3], 
                                        size=gt_box3d[i, 3:6], 
                                        yaw=gt_box3d[i, 6])
                    
def process_single(fname_npz, common_infos=None):
    class_set = common_infos['class_set']
    fpath_mogo_bank = common_infos['fpath_mogo_bank']
    task_cfg = common_infos['task_cfg']
    token = fname_npz.split('/')[-1][:-4]

    data_dic = np.load(fname_npz)
    pcds_xyzi = data_dic['pcd']
    pcd_label = data_dic['pcd_label']
    pt_box3d = data_dic['pt_boxes']
    pt_label = data_dic['pt_labels']
    pt_score = data_dic['pt_scores']
    gt_box3d = data_dic['gt_boxes']
    gt_label = data_dic['gt_labels']
    gt_score = data_dic['gt_scores']

    if gt_box3d.shape[0] == 0:
        return

    pt_mask = filter_box(pt_box3d, pcds_xyzi)
    pt_box3d = pt_box3d[:, [0,1,2,3,4,5,8]]
    pt_box3d = pt_box3d[pt_mask]
    pt_label = pt_label[pt_mask]
    pt_score = pt_score[pt_mask]
    pt_label = pt_label.reshape(1,-1)

    gt_mask = filter_box(gt_box3d, pcds_xyzi)
    gt_box3d = gt_box3d[:, [0,1,2,3,4,5,8]]
    gt_box3d = gt_box3d[gt_mask]
    gt_label = gt_label[gt_mask]
    gt_score = gt_score[gt_mask]
    gt_label = gt_label.reshape(-1,1)
    if gt_box3d.shape[0] == 0:
        return

    label_mask = (gt_label==pt_label)
    all_ious = calc_iou(gt_box3d, pt_box3d) ## M x N

    all_ious = all_ious*label_mask
    gt_ious = np.max(all_ious, axis=1)
    gt_idx = np.argmax(all_ious, axis=1)
    gt_bank_mask = (gt_ious<0.1)|(pt_score[gt_idx]<0.1)

    hard_box3d = gt_box3d[gt_bank_mask]
    hard_label = gt_label[gt_bank_mask]

    for i in range(hard_box3d.shape[0]):
        box = hard_box3d[i]
        sem_label = int(hard_label[i])+1
        sem_label_name = task_cfg['label_map_inv'][sem_label]
        if sem_label_name in class_set:
            box_corner = compute_box_3d(box[:3], box[3:6], box[6])
            obj_mask = in_hull(pcds_xyzi[:, :3], box_corner)
            pcds_obj = pcds_xyzi[obj_mask]
            pcds_label_obj = pcd_label[obj_mask]
            label_list = np.unique(pcds_label_obj).tolist()

            for label in label_list:
                # print(sem_label_name, sem_label, label)
                if sem_label == label:
                    obj_refine_mask = (pcds_label_obj == label)
                    pcds_obj_refine = pcds_obj[obj_refine_mask]
                    pcds_label_obj_refine = pcds_label_obj[obj_refine_mask]
                    pcds_lidarid = np.zeros_like(pcds_label_obj_refine)
                    fname_obj = os.path.join(fpath_mogo_bank, sem_label_name, "{0}##{1}.npz".format(token, i))

                    np.savez_compressed(fname_obj, 
                                        pcds=pcds_obj_refine, 
                                        pcds_lidarid=pcds_lidarid, 
                                        cate_id=sem_label, 
                                        cate=sem_label_name,
                                        center=hard_box3d[i, :3], 
                                        size=hard_box3d[i, 3:6], 
                                        yaw=hard_box3d[i, 6])

if __name__ == '__main__':
    class_set = set(("person", "bike", "rider", "car", "truck", "bus"))
    fpath_mogo = '/rtx/lijun/datasets/mogoB2/bankB22024'
    fpath_mogo_bank = os.path.join(fpath_mogo, 'mogo_bank1203')

    for cate in class_set:
        fpath_cate = os.path.join(fpath_mogo_bank, cate)
        os.system("mkdir -p {}".format(fpath_cate))

    with open('configs/mogo/mogo_21.yaml', 'r') as f:
        task_cfg = yaml.load(f, Loader=yaml.Loader)
        
    data_root = "/rtx/lijun/datasets/mogoB2/dataB22024"
    flist = []
    for fn in os.listdir(data_root):
        print('root:{}'.format(fn))
        fn=os.path.join(data_root,fn,'3d_npz')
        npz_files_in_dir = glob.glob(os.path.join(fn, '*.npz'))
        flist.extend(npz_files_in_dir)

    common_infos=dict(class_set=class_set, fpath_mogo_bank=fpath_mogo_bank, task_cfg=task_cfg)

    print("Total Length: ", len(flist))
    # multi-process
    pbar = tqdm(total=len(flist))
    pbar.set_description("processing")
    update = lambda *args: pbar.update()
    pool1 = multiprocessing.Pool(processes = 32)

    pool1.map(functools.partial(process_single, common_infos=common_infos), flist)
    pool1.close()
    pool1.join()