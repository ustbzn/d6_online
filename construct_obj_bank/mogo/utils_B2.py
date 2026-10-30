import numpy as np
import random
import yaml
import os
import cv2
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

import pdb


def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def csr2corners(center, size, yaw):
    '''
        0 -------- 1
       /|         /|
      3 -------- 2 .
      | |        | |
      . 4 -------- 5
      |/         |/
      7 -------- 6
    Input: 
        center, size, 3
        yaw, 1
    Output:
        corners_3d, (8, 3)
    '''
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
    corners_3d = np.dot(R, np.stack((x_corners, y_corners, z_corners), axis=0))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = corners_3d.T
    return corners_3d


def csr2corners_batch(gt_3d_box):
    '''
    Input:
        gt_3d_box, (N, 7), 7 -> (cx, cy, cz, l, w, h, yaw)
    Output:
        gt_3d_box_corners, (N, 8, 3)
    '''
    gt_3d_box_corners = []
    for i in range(gt_3d_box.shape[0]):
        corners_3d_tmp = csr2corners(center=gt_3d_box[i, :3], size=gt_3d_box[i, 3:6], yaw=gt_3d_box[i, 6])
        gt_3d_box_corners.append(corners_3d_tmp)
    
    gt_3d_box_corners = np.stack(gt_3d_box_corners, axis=0)
    return gt_3d_box_corners


def corners2csr_batch(gt_3d_box_corners):
    '''
        0 -------- 1
       /|         /|
      3 -------- 2 .
      | |        | |
      . 4 -------- 5
      |/         |/
      7 -------- 6
    Input:
        gt_3d_box_corners, (N, 8, 3)
    Output:
        gt_3d_box, (N, 7), 7 -> (cx, cy, cz, l, w, h, yaw)
    '''
    center = gt_3d_box_corners.mean(axis=1) #(N, 3)
    l = np.sqrt(np.power(gt_3d_box_corners[:, [0, 1, 4, 5]] - gt_3d_box_corners[:, [3, 2, 7, 6]], 2).sum(axis=2)).mean(axis=1, keepdims=True)
    w = np.sqrt(np.power(gt_3d_box_corners[:, [0, 3, 4, 7]] - gt_3d_box_corners[:, [1, 2, 5, 6]], 2).sum(axis=2)).mean(axis=1, keepdims=True)
    h = np.sqrt(np.power(gt_3d_box_corners[:, [0, 1, 2, 3]] - gt_3d_box_corners[:, [4, 5, 6, 7]], 2).sum(axis=2)).mean(axis=1, keepdims=True)
    r_tmp = gt_3d_box_corners[:, [0, 1, 4, 5], :2] - gt_3d_box_corners[:, [3, 2, 7, 6], :2]
    r = np.arctan2(r_tmp[:, :, 1], r_tmp[:, :, 0]).mean(axis=1, keepdims=True)
    gt_3d_box = np.concatenate((center, l, w, h, r), axis=1)
    return gt_3d_box


def random_float(v_range):
    v = random.random()
    v = v * (v_range[1] - v_range[0]) + v_range[0]
    return v


def in_range(v, r):
    return (v >= r[0]) * (v < r[1])


def filter_pcds(pcds, range_x=(-40, 60), range_y=(-40, 40), range_z=(-3, 5)):
    valid_x = (pcds[:, 0] >= range_x[0]) * (pcds[:, 0] < range_x[1])
    valid_y = (pcds[:, 1] >= range_y[0]) * (pcds[:, 1] < range_y[1])
    valid_z = (pcds[:, 2] >= range_z[0]) * (pcds[:, 2] < range_z[1])
    
    pcds_filter = pcds[valid_x * valid_y * valid_z]
    return pcds_filter


def filter_pcds_mask(pcds, range_x=(-40, 60), range_y=(-40, 40), range_z=(-3, 5)):
    valid_x = (pcds[:, 0] >= range_x[0]) * (pcds[:, 0] < range_x[1])
    valid_y = (pcds[:, 1] >= range_y[0]) * (pcds[:, 1] < range_y[1])
    valid_z = (pcds[:, 2] >= range_z[0]) * (pcds[:, 2] < range_z[1])
    
    valid_mask = valid_x * valid_y * valid_z
    return valid_mask


def Trans(pcds, mat):
    pcds_out = pcds.copy()
    
    pcds_tmp = pcds_out[:, :4].T
    pcds_tmp[-1] = 1
    pcds_tmp = mat.dot(pcds_tmp)
    pcds_tmp = pcds_tmp.T
    
    pcds_out[..., :3] = pcds_tmp[..., :3]
    pcds_out[..., 3:] = pcds[..., 3:]
    return pcds_out


def relabel(pcds_labels, label_map):
    result_labels = np.zeros((pcds_labels.shape[0],), dtype=pcds_labels.dtype)
    for key in label_map:
        value = label_map[key]
        mask = (pcds_labels == key)
        result_labels[mask] = value
    
    return result_labels


def compute_3dbox_iou(box1_corners, box2_corners):
    '''
    Input:
        box1_corners, box2_corners, (8, 3)
    '''
    box1_bev_poly = Polygon(box1_corners[[0,1,2,3], :2])
    box2_bev_poly = Polygon(box2_corners[[0,1,2,3], :2])

    # bev IoU
    box_12_bev_intersec = box1_bev_poly.intersection(box2_bev_poly).area

    # 3D IoU
    box_12_h_intersec = min(box1_corners[0, 2], box2_corners[0, 2]) - max(box1_corners[7, 2], box2_corners[7, 2])
    box_12_h_intersec = max(0, box_12_h_intersec)
    box1_h = box1_corners[0, 2] - box1_corners[7, 2]
    box2_h = box2_corners[0, 2] - box2_corners[7, 2]
    box_3d_intersec = box_12_bev_intersec * box_12_h_intersec
    iou_3d = box_3d_intersec / (box1_bev_poly.area * box1_h + box2_bev_poly.area * box2_h - box_3d_intersec + 1e-12)
    return iou_3d


def compute_3dbox_iou_self(box_corners):
    '''
    Input:
        box_corners, (N, 8, 3)
    '''
    N = len(box_corners)
    iou_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            iou_mat[i, j] = compute_3dbox_iou(box_corners[i], box_corners[j])
    return iou_mat


def process_single(meta_data, common_infos=None):
    fname_npz = meta_data

    class_set = common_infos['class_set']
    fpath_mogo_bank = common_infos['fpath_mogo_bank']
    task_cfg = common_infos['task_cfg']

    data_dic = np.load(fname_npz, allow_pickle=True)
    #print(fname_npz, data_dic.files)
    try:
        pcds_xyzi = data_dic['pcd'].astype(np.float32)
        pcds_label = data_dic['pcd_label'].astype(np.int64)
        pcds_label_use = relabel(pcds_label, task_cfg['learning_map'])
        gt_bbox3d = data_dic['bboxes'].astype(np.float32)
        gt_cls = data_dic['bboxes_label'].astype(np.int64)
    except: 
        pcds_xyzi = data_dic['pcds'].astype(np.float32)
        pcds_label = data_dic['pcds_label'].astype(np.int64)
        pcds_label_use = relabel(pcds_label, task_cfg['learning_map'])
        gt_bbox3d = data_dic['boxes'].astype(np.float32)
        gt_cls = data_dic['boxes_label'].astype(np.int64)
    try:
        pcds_lidarid = data_dic['pcds_lidarid'].astype(np.uint8)
    except:
        pcds_lidarid = np.zeros_like(pcds_label_use)

    try:
        groups = data_dic['groups']
        is_excessive_layering_list = data_dic['is_excessive_layering_list']
    except:
        groups = np.zeros((gt_bbox3d.shape[0],), dtype=np.uint8)
        is_excessive_layering_list = np.zeros((gt_bbox3d.shape[0],), dtype=np.uint8)

    if gt_bbox3d.shape[0] >= 1:
        valid_mask = (groups == 0) * (is_excessive_layering_list == 0)
        # valid_mask = (groups == 0) * (is_excessive_layering_list == 'False')
        try:
            gt_bbox3d = gt_bbox3d[valid_mask]
            gt_cls = gt_cls[valid_mask]
        except:
            pass
            # print(fname_npz, valid_mask.shape, gt_bbox3d.shape)
    
    if gt_bbox3d.shape[0] >= 1:
        gt_cls_use = relabel(gt_cls, task_cfg['learning_map'])

        gt_boxes_corners = csr2corners_batch(gt_bbox3d)
        iou_mat = compute_3dbox_iou_self(gt_boxes_corners)
        delete_index_set = set(np.where(iou_mat > 0)[0]) | set(np.where(iou_mat > 0)[1])
        for i in range(gt_bbox3d.shape[0]):
            sem_label = int(gt_cls_use[i])
            sem_label_name = task_cfg['label_map_inv'][sem_label]
            if (i not in delete_index_set) and (sem_label_name in class_set):
                obj_mask = in_hull(pcds_xyzi[:, :3], gt_boxes_corners[i])
                pcds_obj = pcds_xyzi[obj_mask]
                pcds_label_use_obj = pcds_label_use[obj_mask]
                pcds_lidarid_obj = pcds_lidarid[obj_mask]

                if pcds_obj.shape[0] > 5:
                    label_list = np.unique(pcds_label_use_obj).tolist()
                    for label in label_list:
                        if sem_label == label:
                            obj_refine_mask = (pcds_label_use_obj == label)
                            pcds_obj_refine = pcds_obj[obj_refine_mask]
                            pcds_label_use_obj_refine = pcds_label_use_obj[obj_refine_mask]
                            pcds_lidarid_obj_refine = pcds_lidarid_obj[obj_refine_mask]
                            if pcds_obj_refine.shape[0] > 5:
                                bag_name, timestamp = fname_npz.split('/')[-2:]
                                fname_obj = os.path.join(fpath_mogo_bank, sem_label_name, "{0}##{1}##{2}.npz".format(bag_name, timestamp, i))

                                np.savez_compressed(fname_obj, pcds=pcds_obj_refine, pcds_lidarid=pcds_lidarid_obj_refine, cate_id=sem_label, cate=sem_label_name,
                                center=gt_bbox3d[i, :3], size=gt_bbox3d[i, 3:6], yaw=gt_bbox3d[i, 6])
