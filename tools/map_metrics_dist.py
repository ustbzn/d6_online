import numpy as np
import torch
import collections
import pdb
import numba
import os
import json
import sys 
import tqdm
import matplotlib.pyplot as plt
from datetime import date
from scipy.spatial import Delaunay
from types import SimpleNamespace
from collections import defaultdict, OrderedDict

sys.path.insert(0, "/pcpt/pcpt/project/lijun/BigFusion")

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


class SceneFlowMetric:
    def __init__(self, ignore_index=0, Classes=[], static_thresh=0.2):
        self.ignore_index = ignore_index
        self.Classes = Classes
        self.static_thresh = static_thresh
        self.reset()
    
    def reset(self):
        self.error_sum = np.zeros((len(self.Classes), 2), dtype=np.double)
        self.num = np.zeros((len(self.Classes), 2), dtype=np.double)
    
    def addBatch(self, gt_cls, gt_motion, pred_motion):
        # gt_cls (N)
        # pred_motion, gt_motion (N, 4), 4 -> (vx, vy, vz, flag)
        gt_motion_v = np.linalg.norm(gt_motion[:, :3], ord=2, axis=1)
        valid_motion_mask = (gt_motion[:, 3] >= 0)
        valid_moving_mask = valid_motion_mask * (gt_motion_v >= self.static_thresh)
        valid_static_mask = valid_motion_mask * (gt_motion_v < self.static_thresh)

        error_motion_v = np.linalg.norm((gt_motion[:, :3] - pred_motion[:, :3]), ord=2, axis=1)
        for i, cate in enumerate(self.Classes):
            if i != self.ignore_index:
                gt_cls_mask_i = (gt_cls == i)

                # moving
                moving_mask_i = gt_cls_mask_i * valid_moving_mask
                moving_mask_i_sum = moving_mask_i.sum()
                if(moving_mask_i_sum > 0):
                    self.error_sum[i, 0] += error_motion_v[moving_mask_i].sum()
                    self.num[i, 0] += moving_mask_i_sum
                
                # static
                static_mask_i = gt_cls_mask_i * valid_static_mask
                static_mask_i_sum = static_mask_i.sum()
                if(static_mask_i_sum > 0):
                    self.error_sum[i, 1] += error_motion_v[static_mask_i].sum()
                    self.num[i, 1] += static_mask_i_sum
    
    def get_metric(self):
        result_dic = collections.OrderedDict()
        error_avg = self.error_sum / (self.num + 1e-12)
        for i, cate in enumerate(self.Classes):
            if i != self.ignore_index:
                result_dic[cate + '_L2_moving'] = float(error_avg[i, 0])
                result_dic[cate + '_L2_static'] = float(error_avg[i, 1])
        
        self.reset()
        return result_dic
    
def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

class Det3DMetric:
    def __init__(self, iou_type="3d", Classes=[], iou_threshs=[], point_cloud_range=[-102.4, -102.4, -5.0, 102.4, 102.4, 5.0], filter_nums=0, metric_key='mAP', metric_type='macro', range_version="0", metric_classes = []):
        self.iou_type = iou_type
        self.det_classes = Classes
        if len(metric_classes) <= 0:
            self.metric_classes = Classes
        else:
            self.metric_classes = metric_classes
        self.iou_threshs = iou_threshs
        self.metric_key = metric_key
        self.metric_type = metric_type
        self.class_nums = len(Classes)
        self.point_cloud_range = {}
        if isinstance(point_cloud_range, list):
            for cls_label in Classes:
                self.point_cloud_range.update({cls_label: point_cloud_range})
        elif isinstance(point_cloud_range, dict):
            self.point_cloud_range = point_cloud_range
        else:
            for cls_label in Classes:
                self.point_cloud_range.update({cls_label: point_cloud_range})

        self.filter_nums = filter_nums
        self.pt_detections = dict()
        self.gt_detections = dict()
        self.pcds_result = dict()
        self.N_SAMPLE_PTS = 101
        self.epoch = range_version
        self.beta = 1.0
        # first and second bins
        # self.first_bins = [0, 10, 30, 50, 80]     # 0-10, 10-30, 30-50, 50-80, >80
        # self.n_first = len(self.first_bins)
        # self.second_bins = [45, 135, 225, 315, 405]   # 4 类角度
        # self.n_second = len(self.second_bins) - 1
        # x and y bins
        self.first_bins = [-80, -50, -30, -10, 0, 10, 30, 50, 80]  # 示例，可根据实际场景调整
        self.n_first = len(self.first_bins) - 1
        self.second_bins = [-50, -30, -10, 0, 10, 30, 50]
        self.n_second = len(self.second_bins) - 1

    def reset(self):
        self.pt_detections = dict()
        self.gt_detections = dict()
        self.pcds_result = dict()

    def addBatch(self, batch_result):  # type: ignore
        gt_results = batch_result['gt_results']
        pt_results = batch_result['pt_results']
        self.pt_detections.update(pt_results)
        self.gt_detections.update(gt_results)
        self.pcds_result.update(batch_result['pts_filename'])

    def get_boxs_pts_num(self, box, pcd):
        corner_box = csr2corners(box[:3],box[3:6],box[6])
        box_pcd_flag = in_hull(pcd[:,:3], corner_box)
        pts_nums = box_pcd_flag.sum()
        return pts_nums

    def get_thresholds(self, scores: np.ndarray, num_gt, num_sample_pts=41):
        scores.sort()
        scores = scores[::-1]
        current_recall = 0
        thresholds = []
        for i, score in enumerate(scores):
            l_recall = (i + 1) / num_gt
            if i < (len(scores) - 1):
                r_recall = (i + 2) / num_gt
            else:
                r_recall = l_recall
            # print(score, r_recall, l_recall, current_recall)
            if (((r_recall - current_recall) < (current_recall - l_recall))
                    and (i < (len(scores) - 1))):
                continue
            # recall = l_recall
            thresholds.append(score)
            current_recall += 1 / (num_sample_pts - 1.0)
        return thresholds

    def compute_statistics_jit(self,
                               overlaps,
                               dt_scores,
                               ignored_gt,
                               ignored_det,
                               min_overlap,
                               thresh=0,
                               compute_fp=False):
        """
        单帧内统计 TP / FP / FN 及 TP 置信度列表。
        ----
        overlaps      : (N_det, N_gt)  IoU 矩阵
        dt_scores     : (N_det,)       预测框置信度
        ignored_gt    : (N_gt,)  -1 忽略, 0 参与
        ignored_det   : (N_det,) -1 忽略, 0 参与
        min_overlap   : float        IoU 阈值
        thresh        : float        置信度阈值（仅 compute_fp=True 时生效）
        compute_fp    : bool         是否计算 FP
        ----
        tp, fp, fn, thresholds
        """
        n_det = overlaps.shape[0]
        n_gt  = overlaps.shape[1]

        # 标记：预测框是否已匹配 / 是否因低置信度被忽略
        assigned_det   = [False] * n_det
        ignored_thresh = [False] * n_det
        if compute_fp:
            for i in range(n_det):
                if dt_scores[i] < thresh:
                    ignored_thresh[i] = True
        NO_DETECTION = -1e8          # 改为更安全的极小值
        tp = fp = fn = 0
        thresholds = np.zeros(n_gt, dtype=np.float32)
        thresh_idx = 0
        # ---------- 遍历每个 GT ----------
        for g in range(n_gt):
            if ignored_gt[g] == -1:
                continue                      # 忽略的背景 GT
            best_det_idx = -1
            best_score   = NO_DETECTION
            max_iou      = 0.0
            # 找最佳匹配预测框
            for d in range(n_det):
                if ignored_det[d] == -1 or assigned_det[d] or ignored_thresh[d]:
                    continue
                iou = overlaps[d, g]
                score = dt_scores[d]
                # 未 compute_fp：选最高置信度且 IoU 达标的
                if not compute_fp and iou >= min_overlap and score > best_score:
                    best_det_idx = d
                    best_score   = score
                # compute_fp：选最高 IoU 且达标的
                elif compute_fp and iou >= min_overlap and iou > max_iou:
                    max_iou = iou
                    best_det_idx = d
                    best_score = 1.0  # 占位即可
            # 根据匹配结果计数
            if best_score == NO_DETECTION:
                fn += 1
            else:
                tp += 1
                thresholds[thresh_idx] = dt_scores[best_det_idx]
                thresh_idx += 1
                assigned_det[best_det_idx] = True
        # ---------- 剩下未匹配且未被忽略的预测框 → FP ----------
        if compute_fp:
            for d in range(n_det):
                if (not assigned_det[d] and
                    ignored_det[d] != -1 and
                    not ignored_thresh[d]):
                    fp += 1
        return tp, fp, fn, thresholds[:thresh_idx]

    def calc_iou(self, gt_boxes, pt_boxes):
        from nms_lib import boxes_iou_bev, boxes_iou3d_gpu
        '''
        calc iou in cuda and tensor!!
        '''
        if isinstance(gt_boxes, np.ndarray):
            gt_boxes = torch.from_numpy(gt_boxes)

        if isinstance(pt_boxes, np.ndarray):
            pt_boxes = torch.from_numpy(pt_boxes)

        if self.iou_type == 'bev':
            ious = boxes_iou_bev(pt_boxes.cuda(), gt_boxes.cuda())
        elif self.iou_type == '3d':
            ious = boxes_iou3d_gpu(pt_boxes.cuda(), gt_boxes.cuda())
        else:
            ious = boxes_iou_bev(pt_boxes.cuda(), gt_boxes.cuda())
        ious = ious.cpu().data.numpy()
        return ious
    
    def process_annos(self, pt_detections, gt_detections=None, epoch=0, save_path=''):
        print('pred detection nums:', len(pt_detections.keys()))
        self.epoch = epoch
        # assert len(pt_detections.keys()) == len(gt_detections.keys())
        pt_annos = {}
        for token, det in pt_detections.items():
            pts_filename = self.pcds_result[token]
            box3d = det["box3d_lidar"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            labels = det["label_preds"].detach().cpu().numpy()
            box3d = box3d.astype(np.float32)
            scores = scores.astype(np.float32)
            labels = labels.astype(np.uint8)
            annos = []
            for i, box in enumerate(box3d):
                det_anno = {
                    "sample_token": token,
                    "pts_filename": pts_filename,
                    "translation": box[:3].tolist(),
                    "size": box[3:6].tolist(),
                    "rotation": box[8:].tolist(),
                    "velocity": box[6:8].tolist(),
                    "detection_name": self.det_classes[labels[i]],
                    "detection_score": float(scores[i]),
                }
                annos.append(det_anno)
            pt_annos.update({token: annos})

        self.pt_res_path = os.path.join(save_path, 'pt_result_{}.json'.format(epoch))
        with open(self.pt_res_path, "w") as f:
            json.dump(pt_annos, f)
        print(f"Finish generate predictions for testset, save to {self.pt_res_path}")

        self.gt_res_path = os.path.join(save_path, 'gt_result.json')
        if not os.path.exists(self.gt_res_path) and gt_detections is not None:
            print('gt detection nums:', len(gt_detections.keys()))
            gt_annos = {}
            for token, det in gt_detections.items():
                pts_filename = self.pcds_result[token]
                box3d = det["box3d_lidar"].detach().cpu().numpy()
                scores = det["scores"].detach().cpu().numpy()
                labels = det["label_preds"].detach().cpu().numpy()
                box3d = box3d.astype(np.float32)
                scores = scores.astype(np.float32)
                labels = labels.astype(np.uint8)
                annos = []
                for i, box in enumerate(box3d):
                    det_anno = {
                        "sample_token": token,
                        "pts_filename": pts_filename,
                        "translation": box[:3].tolist(),
                        "size": box[3:6].tolist(),
                        "rotation": box[8:].tolist(),
                        "velocity": box[6:8].tolist(),
                        "detection_name": self.det_classes[labels[i]],
                        "detection_score": float(scores[i]),
                    }
                    annos.append(det_anno)
                gt_annos.update({token: annos})
            with open(self.gt_res_path, "w") as f:
                json.dump(gt_annos, f)
            print(f"Finish generate gt for testset, save to {self.gt_res_path}")

    def get_mAP(self, prec):
        sums = 0
        for i in range(1, prec.shape[-1]):
            sums = sums + prec[..., i]
        return sums / (self.N_SAMPLE_PTS-1) * 100

    def merge_list(self,annos, pcd=None):
        box3d, scores, labels = [], [], []
        # pcd = np.load(annos[0]['pts_filename'])['pcd']
        for i, anno in enumerate(annos):
            sample_token = anno['sample_token']
            center = anno['translation']
            cls_name = anno['detection_name']

            # TODO check this!!
            if center[0] < self.point_cloud_range[cls_name][0] or \
               center[1] < self.point_cloud_range[cls_name][1] or \
               center[2] < self.point_cloud_range[cls_name][2]:
                continue
            if center[0] > self.point_cloud_range[cls_name][3] or \
               center[1] > self.point_cloud_range[cls_name][4] or \
               center[2] > self.point_cloud_range[cls_name][5]:
                continue

            size = anno['size']
            yaw = anno['rotation']

            box = center+size+yaw
            if pcd is not None and self.get_boxs_pts_num(box, pcd) <= self.filter_nums:
                continue

            score = anno['detection_score']
            label = self.det_classes.index(cls_name)

            box3d.append(box)
            scores.append(score)
            labels.append(label)

        box3d = np.array(box3d).astype(np.float32)
        scores = np.array(scores).astype(np.float32)
        labels = np.array(labels).astype(np.uint8)
        return box3d, scores, labels

    # ''' ---------- 1. 读 JSON ----------'''
    def _load_results(self):
        with open(self.pt_res_path, 'r') as f:
            pt = json.load(f)
        with open(self.gt_res_path, 'r') as f:
            gt = json.load(f)
        print('pred frames:', len(pt), 'gt frames:', len(gt))
        return pt, gt

    # ---------- 2. 初始化容器 ----------
    def _init_stats(self):
        stats = SimpleNamespace()   # 用属性访问更直观
        stats.thresholdss   = defaultdict(list)   # key: (cls, iou, first, second)
        stats.valid_gt_nums = defaultdict(int)    # key: (cls, first, second)
        stats.valid_pt_nums = defaultdict(int)    # key: (cls, first, second)
        stats.aoe = defaultdict(list)             # key: (cls, first, second)
        stats.ate_x = defaultdict(list)
        stats.ate_y = defaultdict(list)
        stats.ase_x = defaultdict(list)
        stats.ase_y = defaultdict(list)
        stats.all_ious = []
        stats.pt_scores_list = []
        stats.pt_labels_list = []
        stats.gt_labels_list = []
        return stats

    # ---------- 加载pcd点云 ----------
    def _load_pcd(self, annos):
        try:
            if self.filter_nums > 0:
                try:
                    pcd = np.load(annos[0]['pts_filename'])['pcd']
                except:
                    pcd = np.load(annos[0]['pts_filename'])['pcds']
            else:
                pcd = None
        except:
            # print('pcd is None')
            pcd = None
        return pcd

    # ---------- 3. 处理单帧 ----------
    def _process_frame(self, token, annos, pt_results, stats):
        # 3.1 读点云
        pcd = self._load_pcd(annos)

        # 3.2 合并 box
        gt_boxes, _, gt_labels = self.merge_list(annos, pcd)
        pt_boxes, pt_scores, pt_labels = self.merge_list(pt_results[token], pcd)

        if gt_boxes.shape[0] == 0 or pt_boxes.shape[0] == 0:
            stats.all_ious.append(None)
            stats.pt_scores_list.append(pt_scores)
            stats.pt_labels_list.append(pt_labels)
            stats.gt_labels_list.append(gt_labels)
            return

        # 3.3 计算距离并分 bin
        gt_first, gt_second = self._dist_xy(gt_boxes)
        pt_first, pt_seconde = self._dist_xy(pt_boxes)

        # 3.4 逐 bin 统计
        frame_ious = [[None] * self.n_second for _ in range(self.n_first)]
        frame_pt_s = [[None] * self.n_second for _ in range(self.n_first)]
        frame_pt_l = [[None] * self.n_second for _ in range(self.n_first)]
        frame_gt_l = [[None] * self.n_second for _ in range(self.n_first)]

        for first_idx in range(self.n_first):
            g_mask_first = gt_first == first_idx
            p_mask_first = pt_first == first_idx
            for second_idx in range(self.n_second):
                # 角度索引过滤
                g_mask_second = gt_second == second_idx
                p_mask_second = pt_seconde == second_idx
                g_mask = g_mask_first & g_mask_second
                p_mask = p_mask_first & p_mask_second
                g_b = gt_boxes[g_mask];   p_b = pt_boxes[p_mask]
                g_l = gt_labels[g_mask];  p_l = pt_labels[p_mask]
                p_s = pt_scores[p_mask]
                if g_b.shape[0] == 0 and p_b.shape[0] == 0:
                    continue
                frame_pt_s[first_idx][second_idx] = p_s
                frame_pt_l[first_idx][second_idx] = p_l
                frame_gt_l[first_idx][second_idx] = g_l
                # IoU
                if g_b.shape[0] and p_b.shape[0]:
                    ious = self.calc_iou(g_b, p_b)
                else:
                    ious = None
                    continue
                
                frame_ious[first_idx][second_idx] = ious

                # 匹配 & 误差
                self._update_error(ious, g_b, p_b, g_l, p_l, p_s, first_idx, second_idx, stats)

                # PR 统计
                for cls_idx, cls in enumerate(self.det_classes):
                    self._update_pr(ious, g_l, p_l, p_s, cls_idx, first_idx, second_idx, stats)

        stats.pt_scores_list.append(frame_pt_s)
        stats.pt_labels_list.append(frame_pt_l)
        stats.gt_labels_list.append(frame_gt_l)
        stats.all_ious.append(frame_ious)

    # ---------- 5. 距离计算和角度索引 ----------
    def _dist_angle(self, boxes):
        """
        返回 (距离 bin, 角度 bin)
        角度 = yaw 在 [0, 360)
        """
        # 1. 距离
        d = np.sqrt(boxes[:, 0]**2 + boxes[:, 1]**2)
        d_bin = np.clip(np.digitize(d, self.first_bins) - 1, 0, self.n_first - 1)

        # 2. 角度
        yaw = self._angle_from_xy(boxes[:, 0], boxes[:, 1]) 
        yaw = np.where(yaw < 45, yaw + 360, yaw)
        a_bin = np.clip(np.digitize(yaw, self.second_bins) - 1, 0, self.n_second - 1)

        return d_bin, a_bin
    
    def _dist_xy(self, boxes):
        """
        返回 (x_bin, y_bin)
        """
        x_bin = np.clip(np.digitize(boxes[:, 0], self.first_bins) - 1, 0, self.n_first - 1)
        y_bin = np.clip(np.digitize(boxes[:, 1], self.second_bins) - 1, 0, self.n_second - 1)
        return x_bin, y_bin

    def _angle_from_xy(self, x, y):
        """
        输入: x, y 可以是标量或 ndarray
        输出: 角度(°)，范围 [0,360)，顺时针为正
        """
        # 1) 用 arctan2 得到 [-π,π] 的弧度
        yaw = np.arctan2(-y, x)        # -y 让顺时针为正
        # 2) 转到 [0,360)
        deg = (np.degrees(yaw) + 360) % 360
        return deg

    # ---------- 6. 误差计算 ----------
    def _update_error(self, ious, g_b, p_b, g_l, p_l, p_s, d, second_idx, st):
        if ious is None: return
        ## metric yaw error
        iou_idxs = np.argmax(ious, axis=0)
        iou_values = np.max(ious, axis=0)
        iou_mask = iou_values >= self.iou_threshs[0]
        score_mask = p_s[iou_idxs] >= 0.1
        label_mask = (g_l == p_l[iou_idxs])
        match_mask = iou_mask & score_mask & label_mask
        gt_match_boxes = g_b[match_mask,:]
        pt_match_boxes = p_b[iou_idxs,:][match_mask,:]
        gt_match_labels = g_l[match_mask]
        pt_match_labels = p_l[iou_idxs][match_mask]
        ## no detection
        if pt_match_boxes.shape[0] <= 0 or gt_match_boxes.shape[0] <=0 :
            return

        ## 平均角度误差-aoe
        gt_yaw = limit_period(gt_match_boxes[:,6], 0.5, 2*np.pi) #归一到【-π，π】
        pt_yaw = limit_period(pt_match_boxes[:,6], 0.5, 2*np.pi)
        aoe = np.fabs(gt_yaw - pt_yaw)
        aoe = np.minimum(aoe, 2*np.pi - aoe)  # Limit angle difference to 0-180 degrees
        aoe = np.minimum(aoe, np.pi - aoe)  # Limit angle difference to 0-90 degrees

        ## 平均横向误差-x
        gt_x_center = gt_match_boxes[:,0]
        pt_x_center = pt_match_boxes[:,0]
        ate_x = np.fabs(gt_x_center - pt_x_center)
        # print(pt_scores[iou_idxs], iou_values, iou_idxs, ate_x)
        ## 平均纵向误差-y
        gt_y_center = gt_match_boxes[:,1]
        pt_y_center = pt_match_boxes[:,1]
        ate_y = np.fabs(gt_y_center - pt_y_center)
        ## 平均横向size误差
        gt_x_size = gt_match_boxes[:,3]
        pt_x_size = pt_match_boxes[:,3]
        ase_x = np.fabs(gt_x_size - pt_x_size)
        ## 平均纵向size误差
        gt_y_size = gt_match_boxes[:,4]
        pt_y_size = pt_match_boxes[:,4]
        ase_y = np.fabs(gt_y_size - pt_y_size)

        for m, current_class in enumerate(self.det_classes):
            st.aoe[(m, d, second_idx)].append(aoe[gt_match_labels==m])
            st.ate_x[(m, d, second_idx)].append(ate_x[gt_match_labels==m])
            st.ate_y[(m, d, second_idx)].append(ate_y[gt_match_labels==m])
            st.ase_x[(m, d, second_idx)].append(ase_x[gt_match_labels==m])
            st.ase_y[(m, d, second_idx)].append(ase_y[gt_match_labels==m])

    # ---------- 7. PR 统计 ----------
    def _update_pr(self, ious, g_l, p_l, p_s, cls_idx, d, second_idx, st):
        ignored_gt = np.where(g_l==cls_idx, 0, -1)
        ignored_det = np.where(p_l==cls_idx, 0, -1)
        st.valid_gt_nums[(cls_idx, d, second_idx)] += (ignored_gt+1).sum()
        st.valid_pt_nums[(cls_idx, d, second_idx)] += (ignored_det+1).sum()
        for k, min_overlap in enumerate(self.iou_threshs):
            rets = self.compute_statistics_jit(
                    ious,
                    p_s,
                    ignored_gt,
                    ignored_det,
                    min_overlap,
                    thresh=0.0,
                    compute_fp=False
            )
            tp, fp, fn, thresholds = rets
            st.thresholdss[(cls_idx,k,d,second_idx)] += thresholds.tolist()

    # ---------- 8. 汇总MAP、TP95 ----------
    def _summarize(self, save_path, stats):
        """
        汇总所有统计结果并生成 JSON / 打印
        stats 结构见 _init_stats()
        """
        # ---------- 0. 预定义 ----------
        iou_nums  = len(self.iou_threshs)
        cls_nums  = len(self.det_classes)
        first_nums = self.n_first
        second_nums = self.n_second
        beta      = self.beta

        # ---------- 1. 计算 mAP ----------
        # precision / recall / scores 形状 (cls, iou, first, second, N_SAMPLE_PTS)
        precision = np.zeros([cls_nums, iou_nums, first_nums, second_nums, self.N_SAMPLE_PTS])
        recall    = np.zeros_like(precision)
        scores    = np.zeros_like(precision)
        det_nums  = np.zeros_like(precision)

        for cls_idx in range(cls_nums):
            for iou_idx, min_iou in enumerate(self.iou_threshs):
                for first_idx in range(first_nums):
                    for second_idx in range(second_nums):
                        key = (cls_idx, iou_idx, first_idx, second_idx)
                        ths = np.array(stats.thresholdss[key])
                        n_gt = stats.valid_gt_nums[(cls_idx, first_idx, second_idx)]
                        n_pt = stats.valid_pt_nums[(cls_idx, first_idx, second_idx)]
                        if n_gt == 0:
                            # 修正：没有gt 召回率为1.0 准确率看n_pt
                            if n_pt == 0:
                                precision[cls_idx, iou_idx, first_idx, second_idx, :] = 1.0
                            else:
                                precision[cls_idx, iou_idx, first_idx, second_idx, :] = 0.0
                            recall[cls_idx, iou_idx, first_idx, second_idx, :] = 1.0
                            continue
                        elif len(ths) == 0:
                            precision[cls_idx, iou_idx, first_idx, second_idx, :] = 1.0
                            recall[cls_idx, iou_idx, first_idx, second_idx, :] = 0.0
                            continue
                        thresholds = self.get_thresholds(ths, n_gt, self.N_SAMPLE_PTS)
                        pr = np.zeros([len(thresholds), 3])  # TP, FP, FN
                        for t, th in enumerate(thresholds):
                            for all_ious, gt_l, pt_l, pt_s in zip(
                                    stats.all_ious,
                                    stats.gt_labels_list,
                                    stats.pt_labels_list,
                                    stats.pt_scores_list):
                                if gt_l[first_idx][second_idx] is None:
                                    # 没有gt和pt
                                    tp = 0
                                    fp = 0
                                    fn = 0
                                elif all_ious[first_idx][second_idx] is None:
                                    # 没有匹配的 IoU
                                    if pt_l[first_idx][second_idx].shape[0] == 0:
                                        tp = 0
                                        fp = 0
                                        fn = np.where(gt_l[first_idx][second_idx] == cls_idx, 1, 0).sum()
                                    elif gt_l[first_idx][second_idx].shape[0] == 0:
                                        tp = 0
                                        fp = np.where(pt_l[first_idx][second_idx] == cls_idx, 1, 0).sum()
                                        fn = 0
                                else:
                                    ignored_gt  = np.where(gt_l[first_idx][second_idx] == cls_idx, 0, -1)
                                    ignored_det = np.where(pt_l[first_idx][second_idx] == cls_idx, 0, -1)
                                    ious = all_ious[first_idx][second_idx]
                                    if ious is None:
                                        tp = fp = fn = 0
                                    else:
                                        p_s_d = pt_s[first_idx][second_idx]
                                        tp, fp, fn, _ = self.compute_statistics_jit(
                                            ious, p_s_d, ignored_gt, ignored_det,
                                            min_iou, th, True)
                                pr[t, 0] += tp
                                pr[t, 1] += fp
                                pr[t, 2] += fn 
                                if (all_ious is not None) and (all_ious[first_idx][second_idx] is not None):
                                    det_nums[cls_idx, iou_idx, first_idx, second_idx, t] += tp + fp

                        # 计算 PR
                        scores[cls_idx, iou_idx, first_idx, second_idx, :len(thresholds)] = thresholds
                        mask = (pr[:, 0] + pr[:, 1]) > 0
                        precision[cls_idx, iou_idx, first_idx, second_idx, :len(thresholds)][mask] = \
                            pr[mask, 0] / (pr[mask, 0] + pr[mask, 1])
                        mask = (pr[:, 0] + pr[:, 2]) > 0
                        recall[cls_idx, iou_idx, first_idx, second_idx, :len(thresholds)][mask] = \
                            pr[mask, 0] / (pr[mask, 0] + pr[mask, 2])

                        # 单调化
                        for t in range(len(thresholds)):
                            precision[cls_idx, iou_idx, first_idx, second_idx, t] = \
                                np.max(precision[cls_idx, iou_idx, first_idx, second_idx, t:])
                            recall[cls_idx, iou_idx, first_idx, second_idx, t] = \
                                np.max(recall[cls_idx, iou_idx, first_idx, second_idx, :t+1])

        mAP = self.get_mAP(precision)  # (cls, iou, dist)

        # ---------- 2. 误差 TP95 / Mean ----------
        details = OrderedDict()
        summary = OrderedDict()
        metric_cls_cnt = len(self.metric_classes)
        for first_idx in range(first_nums):
            for second_idx in range(second_nums):
                first_tag = f'{self.first_bins[first_idx]}~{self.first_bins[first_idx+1]}m' \
                            if first_idx < first_nums-1 else f'>{self.first_bins[-1]}m'
                second_tag = f'{self.second_bins[second_idx]}~{self.second_bins[second_idx+1]}°'
                aoe_all_classes = None
                ate_y_classes = None
                for cls_idx, cls_name in enumerate(self.det_classes):
                    if cls_name not in self.metric_classes:
                        continue
                    # AOE
                    aoe = np.rad2deg(np.concatenate(stats.aoe[(cls_idx, first_idx, second_idx)] or [[]]))
                    if aoe_all_classes is None:
                        aoe_all_classes = aoe
                    else:
                        aoe_all_classes = np.concatenate([aoe_all_classes, aoe])
                    aoe_tp95 = np.percentile(aoe, 95) if len(aoe) else 0
                    aoe_mean = aoe.mean() if len(aoe) else 0
                    details[f'aoe_tp95_{cls_name}_{first_tag}_{second_tag}'] = f'{aoe_tp95:.2f}'
                    details[f'aoe_{cls_name}_{first_tag}_{second_tag}'] = f'{aoe_mean:.2f}'

                    # ATE
                    ate_x = np.concatenate(stats.ate_x[(cls_idx, first_idx, second_idx)] or [[]])
                    ate_x_tp95 = np.percentile(ate_x, 95) if len(ate_x) else 0
                    ate_x_mean = ate_x.mean() if len(ate_x) else 0
                    details[f'ate_x_tp95_{cls_name}_{first_tag}_{second_tag}'] = f'{ate_x_tp95:.2f}'
                    details[f'ate_x_{cls_name}_{first_tag}_{second_tag}'] = f'{ate_x_mean:.2f}'

                    ate_y = np.concatenate(stats.ate_y[(cls_idx, first_idx, second_idx)] or [[]])
                    if ate_y_classes is None:
                        ate_y_classes = ate_y
                    else:
                        ate_y_classes = np.concatenate([ate_y_classes, ate_y])
                    ate_y_tp95 = np.percentile(ate_y, 95) if len(ate_y) else 0
                    ate_y_mean = ate_y.mean() if len(ate_y) else 0
                    details[f'ate_y_tp95_{cls_name}_{first_tag}_{second_tag}'] = f'{ate_y_tp95:.2f}'
                    details[f'ate_y_{cls_name}_{first_tag}_{second_tag}'] = f'{ate_y_mean:.2f}'

                    # ASE
                    ase_x = np.concatenate(stats.ase_x[(cls_idx, first_idx, second_idx)] or [[]])
                    ase_x_tp95 = np.percentile(ase_x, 95) if len(ase_x) else 0
                    ase_x_mean = ase_x.mean() if len(ase_x) else 0
                    details[f'ase_x_tp95_{cls_name}_{first_tag}_{second_tag}'] = f'{ase_x_tp95:.2f}'
                    details[f'ase_x_{cls_name}_{first_tag}_{second_tag}'] = f'{ase_x_mean:.2f}'

                    ase_y = np.concatenate(stats.ase_y[(cls_idx, first_idx, second_idx)] or [[]])
                    ase_y_tp95 = np.percentile(ase_y, 95) if len(ase_y) else 0
                    ase_y_mean = ase_y.mean() if len(ase_y) else 0
                    details[f'ase_y_tp95_{cls_name}_{first_tag}_{second_tag}'] = f'{ase_y_tp95:.2f}'
                    details[f'ase_y_{cls_name}_{first_tag}_{second_tag}'] = f'{ase_y_mean:.2f}'
            
                # AOE no classes
                aoe_tp95_all_classes = np.percentile(aoe_all_classes, 95) if len(aoe_all_classes) else 0
                aoe_mean_all_classes = aoe_all_classes.mean() if len(aoe_all_classes) else 0
                details[f'aoe_tp95_{first_idx}_{second_idx}'] = f'{aoe_tp95_all_classes:.2f}'
                details[f'aoe_{first_idx}_{second_idx}'] = f'{aoe_mean_all_classes:.2f}'
                # ATE_Y no classes
                ate_y_tp95_classes = np.percentile(ate_y_classes, 95) if len(ate_y_classes) else 0
                ate_y_mean_classes = ate_y_classes.mean() if len(ate_y_classes) else 0
                details[f'ate_y_tp95_{first_idx}_{second_idx}'] = f'{ate_y_tp95_classes:.2f}'
                details[f'ate_y_{first_idx}_{second_idx}'] = f'{ate_y_mean_classes:.2f}'

        # ---------- 3. macro / micro ----------
        mpre_all = [[[0.0] * iou_nums for _ in range(self.n_second)] for _ in range(self.n_first)]
        mrec_all = [[[0.0] * iou_nums for _ in range(self.n_second)] for _ in range(self.n_first)]
        total_gt = [[[0.0] * iou_nums for _ in range(self.n_second)] for _ in range(self.n_first)]
        total_pt = [[[0.0] * iou_nums for _ in range(self.n_second)] for _ in range(self.n_first)]

        metric_details = {}
        score_list = []
        for cls_idx, cls_name in enumerate(self.det_classes):
            if cls_name not in self.metric_classes:
                continue
            for iou_idx, min_iou in enumerate(self.iou_threshs):
                for first_idx in range(first_nums):
                    first_tag = f'{self.first_bins[first_idx]}~{self.first_bins[first_idx+1]}m' \
                            if first_idx < first_nums-1 else f'>{self.first_bins[-1]}m'
                    for second_idx in range(second_nums):
                        # 计算 F-score
                        prec = precision[cls_idx, iou_idx, first_idx, second_idx]
                        rec  = recall[cls_idx, iou_idx, first_idx, second_idx]
                        sc   = scores[cls_idx, iou_idx, first_idx, second_idx]
                        mask = prec > 0
                        if not mask.any():
                            continue
                        prec = prec[mask]; rec = rec[mask]; sc = sc[mask]
                        f = (1+beta)*prec*rec / (beta*prec+rec+1e-12)
                        best = np.argmax(f)
                        best_pre = prec[best]; best_rec = rec[best]; best_score = sc[best]
                        if iou_idx == 0:
                            score_list.append(best_score)

                        # # 记录 PR 曲线
                        # metric_details[f'{cls_name}_{int(min_iou*100)}_{first_tag}'] = {
                        #     'recall': rec.tolist(),
                        #     'precision': prec.tolist(),
                        #     'scores': sc.tolist(),
                        #     'f-score': f.tolist()
                        # }

                        # macro / micro
                        n_gt = stats.valid_gt_nums[(cls_idx, first_idx, second_idx)]
                        n_pt = det_nums[cls_idx, iou_idx, first_idx, second_idx, best]
                        if self.metric_type == 'macro':
                            mpre_all[first_idx][second_idx][iou_idx] += best_pre
                            mrec_all[first_idx][second_idx][iou_idx] += best_rec
                        else:  # micro
                            total_gt[first_idx][second_idx][iou_idx] += n_gt
                            mpre_all[first_idx][second_idx][iou_idx] += best_pre * n_pt
                            mrec_all[first_idx][second_idx][iou_idx] += best_rec * n_gt
                            total_pt[first_idx][second_idx][iou_idx] += n_pt

        # ---------- 4. 写 JSON ----------
        # os.makedirs(os.path.join(save_path, 'plots'), exist_ok=True)
        # metric_path = os.path.join(save_path, f'metric_details_{self.epoch}.json')
        # with open(metric_path, 'w') as f:
        #     json.dump(metric_details, f, indent=2)

        # ---------- 5. summary ----------
        summary['score_list'] = [float(s) for s in score_list]
        # summary.update(details)
        for first_idx in range(first_nums):
            for second_idx in range(second_nums):
                for k, min_iou in enumerate(self.iou_threshs):
                    if self.metric_type == 'micro':
                        denom_pt = total_pt[first_idx][second_idx][k] or 1
                        denom_gt = total_gt[first_idx][second_idx][k] or 1
                    else:
                        denom_pt = denom_gt = metric_cls_cnt
                    summary[f'mAP@{int(min_iou*100)}@{first_idx}@{second_idx}'] = f'{mAP[:,k, first_idx, second_idx].mean():.2f}'
                    summary[f'FalseDet@{int(min_iou*100)}@{first_idx}@{second_idx}'] = f'{100 - mpre_all[first_idx][second_idx][k]/denom_pt*100:.2f}'
                    summary[f'Missing@{int(min_iou*100)}@{first_idx}@{second_idx}'] = f'{100 - mrec_all[first_idx][second_idx][k]/denom_gt*100:.2f}'

        result = OrderedDict()
        result['details'] = details
        result['summary'] = summary
        result[self.metric_key] = f'{mAP[:, -1, :].mean():.2f}'

        # # 控制台打印
        # for k, v in summary.items():
        #     print(f'{k}: {v}')
        return result

    def get_metric_new(self, save_path=''):
        """计算 mAP、AOE/ATE/ASE，并支持按距离分段"""
        # 1. 读取结果
        pt_results, gt_results = self._load_results()

        # 2. 预分配容器（距离维度 + 类别维度）
        stats = self._init_stats()

        # 3. 逐帧统计
        for token, annos in tqdm.tqdm(gt_results.items()):
            self._process_frame(token, annos, pt_results, stats)

        # 4. 汇总指标
        result_dic = self._summarize(save_path, stats)
        return result_dic



if __name__ == '__main__':
    today = date.today()
    date_str = today.strftime("%Y-%m-%d")
    data_flag = 'bevfusion_eval'
    version_flag = 'v1_D6_c3_1.6k_0.4k_cam5lidar_voxelpoint50_250808'
    metric_type = 'macro'    ## "macro" or "micro"
    Classes=("person", "bike", "rider", "car", "truck", "bus")
    car_classes = ("car", "truck", "bus")
    person_classes = ("person", "bike", "rider")
    point_cloud_range_v1 = [-81.6, -48, -1, 97.6, 48, 3.0]
    point_cloud_range_v2 = [-60.0, -10.0, -1.0, 70.0, 10.0, 3.0]
    point_cloud_range_v3 = [-20.0, -10.0, -1.0, 20.0, 10.0, 3.0]
    range_dic = {
        "v1": {"point_cloud_range": point_cloud_range_v1, "metric_classes": Classes},
        # "v2": {"point_cloud_range": point_cloud_range_v2, "metric_classes": Classes},
        # "carv2":{"point_cloud_range": point_cloud_range_v2, "metric_classes": car_classes},
        # "personv3":{"point_cloud_range": point_cloud_range_v3, "metric_classes": person_classes},
    }
    save_path = '/pcpt/pcpt/project/liqinbu/D6/BEVFusion/{}/{}_metric_first_second{}'.format(data_flag, version_flag, metric_type)
    if not os.path.exists(save_path):
        os.system("mkdir -p {}".format(save_path))
    gt_res_path = '/pcpt/pcpt/project/liqinbu/D6/BEVFusion/{}/{}_gt_filter.json'.format(data_flag, version_flag)
    pt_res_path = '/pcpt/pcpt/project/liqinbu/D6/BEVFusion/{}/{}_pt.json'.format(data_flag, version_flag)
    
    for k, v in range_dic.items():
        print('===================== metric version: {}-{} ====================='.format(metric_type, k))
        det_metric = Det3DMetric(iou_type='bev', 
                                 Classes=Classes,
                                 iou_threshs=[0.1, 0.25, 0.5],
                                #  point_cloud_range=[-20, -5.5, -1.0, 45, 5.5, 3.0],
                                #  point_cloud_range={
                                #      "person":[-10, -5.5, -1.0, 20, 5.5, 3.0],
                                #      "bike":[-10, -5.5, -1.0, 20, 5.5, 3.0],
                                #      "rider":[-10, -5.5, -1.0, 20, 5.5, 3.0],
                                #      "car":[-20, -5.5, -1.0, 45, 5.5, 3.0],
                                #      "truck":[-20, -5.5, -1.0, 45, 5.5, 3.0],
                                #      "bus":[-20, -5.5, -1.0, 45, 5.5, 3.0],
                                # },
                                #  point_cloud_range=[-81.6, -48, -1, 97.6, 48, 3.0],
                                #  point_cloud_range=[-60.0, -10.0, -1.0, 70.0, 10.0, 3.0],
                                 point_cloud_range=v['point_cloud_range'],
                                 filter_nums=0,
                                 metric_key='mAP',
                                 metric_type = metric_type,
                                 range_version=k,
                                 metric_classes=v['metric_classes'],)
        det_metric.pt_res_path = pt_res_path
        det_metric.gt_res_path = gt_res_path
        result_dic = det_metric.get_metric_new(save_path)
        with open(os.path.join(save_path, 'metric_{}.json'.format(k)), 'w') as f:
            json.dump(result_dic, f)
