# coding=utf-8
'''
Author: husserl
License: Apache Licence
Software: VSCode
Date: 2023-12-07 07:00:34
LastEditors: jiangjunyuan jiangjunyuan@zhidaoauto.com
LastEditTime: 2025-06-13 06:40:11
'''
import numpy as np
import os
import json
import glob
import tqdm
from scipy.spatial import Delaunay

label_map = {
    # 0: "ignore",
    0: "person",
    1: "bike",
    2: "rider",
    3: "car",
    4: "truck",
    5: "bus"
}

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

def filter_box(box3d, pcd): # 基于点云过滤框，TODO
    box_mask_list = []
    for box in box3d:
        box_corner = compute_box_3d(box[:3], box[3:6], box[6])
        obj_mask = in_hull(pcd[:, :3], box_corner)
        pts_nums = obj_mask.sum()
        if pts_nums >= 10:
            box_mask_list.append(True)
        else:
            box_mask_list.append(False)
    box_mask = np.array(box_mask_list).astype(bool)
    return box_mask

def get_json_obj(box3d, labels, scores, token):
    annos = []
    for i, box in enumerate(box3d):
        det_anno = {
            "sample_token": token,
            "translation": box[:3].tolist(),
            "size": box[3:6].tolist(),
            "rotation": box[6:].tolist(),
            "velocity": [0.0, 0.0],
            "detection_name": label_map[labels[i]],
            "detection_score": float(scores[i]),
        }
        annos.append(det_anno)    
    return annos
import pickle
import mmcv
import orjson
from datetime import date

if __name__ == '__main__':
    data_flag = 'bevfusion_eval'
    version_flag = 'v3_mixB2C1_0724'
    # data_root = '/rtx/lijun/data/{}/{}'.format(data_flag, version_flag)
    ann_file = '/pcpt/pcpt/project/jiangjunyuan/code/BEVFusion/multimodal_bevfusion/runs/v3_mixB2C1_0724/best_bbox_mAP_epoch_24.pth_results.pkl'
    with open(ann_file, 'rb') as f:  
        datapkl = pickle.load(f)
    gt_res_path = '/pcpt/pcpt/project/jiangjunyuan/code/BEVFusion/{}/{}_gt_filter.json'.format(data_flag, version_flag)
    pt_res_path = '/pcpt/pcpt/project/jiangjunyuan/code/BEVFusion/{}/{}_pt.json'.format(data_flag, version_flag)
    fpath_record='/pcpt/pcpt/project/jiangjunyuan/code/BEVFusion/{}/{}'.format(data_flag, version_flag)
    if not os.path.exists(fpath_record):
        os.system("mkdir -p {}".format(fpath_record))
    pt_annos, gt_annos = {}, {}
    for sample_id, npz_data in enumerate(mmcv.track_iter_progress(datapkl)):
        # npz_data = np.load(npz_path)
        # pcds_xyzi = npz_data['pcds_xyzi'][0,...,0].transpose((1,0))[:,:3]
        # print(npz_data.keys())

        pred_bbox3d = npz_data['boxes_3d'].tensor.numpy()
        pred_bbox3d_class = npz_data['labels_3d'].numpy()
        pred_bbox3d_score = npz_data['scores_3d'].numpy()
        gt_bbox3d = npz_data['gt_bboxes_3d'].tensor.numpy()
        gt_label = npz_data['gt_labels_3d'].numpy()
        # pcds = npz_data['pcds'].numpy()
        pcds = npz_data['gt_points_labels'].numpy()
        pred_points_labels = npz_data['pred_points_labels'].numpy()
        try:
            pcds = npz_data['pcds'].numpy()
            fname_record = os.path.join(fpath_record, "pcd####{}.npz".format(sample_id))
            np.savez_compressed(fname_record, gt=gt_label, pred=pred_points_labels, pcds_xyzi=pcds,
                pred_bbox3d=pred_bbox3d, pred_bbox3d_score=pred_bbox3d_score,pred_bbox3d_class=pred_bbox3d_class.astype(np.int64), gt_bbox3d=gt_bbox3d, gt_label=gt_label)
        except:
            pass
        sample_token = '{}'.format(sample_id)
        pt_box3d = pred_bbox3d[:, :7]
        pt_box3d = pt_box3d
        pt_label = pred_bbox3d_class
        pt_score = pred_bbox3d_score
        pt_anno = get_json_obj(pt_box3d, pt_label, pt_score, sample_token)
        pt_annos.update({sample_token: pt_anno})

        gt_box3d = gt_bbox3d[:, :7]
        gt_box3d = gt_box3d
        gt_score = np.ones_like(gt_label, dtype=np.float32)
        gt_anno = get_json_obj(gt_box3d, gt_label, gt_score, sample_token)
        gt_annos.update({sample_token: gt_anno})
        # break
    # with open(gt_res_path, "w") as f:
    #     json.dump(gt_annos, f)

    # with open(pt_res_path, "w") as f:
    #     json.dump(pt_annos, f)
    print(gt_annos['2'])

    # 写入 JSON 文件
    with open(gt_res_path, 'wb') as f:
        f.write(orjson.dumps(gt_annos))
    with open(pt_res_path, 'wb') as f:
        f.write(orjson.dumps(pt_annos))