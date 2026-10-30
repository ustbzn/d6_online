import numpy as np
import torch
import os
import json
import sys
import tqdm
from datetime import date
from scipy.spatial import Delaunay
from types import SimpleNamespace
from collections import defaultdict, OrderedDict

sys.path.insert(0, "/pcpt/pcpt/project/liqinbu/D6/BEVFusion/multimodal_bevfusion/tools")
from nms_lib import boxes_iou_bev, boxes_iou3d_gpu
from shapely.geometry import Polygon



class Det3DMetric:
    """
    3D目标检测评测类，支持mAP、AOE、ATE、ASE等指标，按空间分bin统计。
    """
    def __init__(self, iou_type="3d", Classes=[], iou_threshs=[], point_cloud_range=None, filter_nums=0, metric_key='mAP', metric_type='macro', range_version="0", metric_classes=None):
        self.iou_type = iou_type
        self.det_classes = Classes
        self.metric_classes = metric_classes if metric_classes else Classes
        self.iou_threshs = iou_threshs
        self.metric_key = metric_key
        self.metric_type = metric_type
        self.class_nums = len(Classes)
        self.point_cloud_range = self._init_point_cloud_range(point_cloud_range, Classes)
        self.filter_nums = filter_nums
        self.pt_detections = dict()
        self.gt_detections = dict()
        self.pcds_result = dict()
        self.N_SAMPLE_PTS = 41
        self.epoch = range_version
        self.beta = 1.0
        # 空间分bin
        self.first_bins = [-80, -50, -30, -10, 0, 10, 30, 50, 80]
        self.n_first = len(self.first_bins) - 1
        self.second_bins = [-50, -30, -10, 0, 10, 30, 50]
        self.n_second = len(self.second_bins) - 1

    def _init_point_cloud_range(self, point_cloud_range, Classes):
        """初始化点云范围字典"""
        if isinstance(point_cloud_range, list):
            return {cls_label: point_cloud_range for cls_label in Classes}
        elif isinstance(point_cloud_range, dict):
            return point_cloud_range
        else:
            return {cls_label: point_cloud_range for cls_label in Classes}

    def reset(self):
        """重置检测结果容器"""
        self.pt_detections = dict()
        self.gt_detections = dict()
        self.pcds_result = dict()

    def addBatch(self, batch_result):
        """批量添加检测结果"""
        gt_results = batch_result['gt_results']
        pt_results = batch_result['pt_results']
        self.pt_detections.update(pt_results)
        self.gt_detections.update(gt_results)
        self.pcds_result.update(batch_result['pts_filename'])

    def csr2corners(self, center, size, yaw):
        """计算3D框的8个角点"""
        c = np.cos(yaw)
        s = np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        l, w, h = size
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        corners_3d = np.dot(R, np.stack((x_corners, y_corners, z_corners), axis=0))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return corners_3d.T

    def in_hull(self, p, hull):
        """判断点是否在凸包内"""
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        return hull.find_simplex(p) >= 0

    def get_boxs_pts_num(self, box, pcd):
        """统计点云中落入box的点数"""
        corner_box = self.csr2corners(box[:3], box[3:6], box[6])
        box_pcd_flag = self.in_hull(pcd[:, :3], corner_box)
        pts_nums = box_pcd_flag.sum()
        return pts_nums

    def get_thresholds(self, scores: np.ndarray, num_gt, num_sample_pts=41):
        """根据召回率采样置信度阈值"""
        scores.sort()
        scores = scores[::-1]
        current_recall = 0
        thresholds = []
        for i, score in enumerate(scores):
            l_recall = (i + 1) / num_gt
            r_recall = (i + 2) / num_gt if i < (len(scores) - 1) else l_recall
            if (((r_recall - current_recall) < (current_recall - l_recall)) and (i < (len(scores) - 1))):
                continue
            thresholds.append(score)
            current_recall += 1 / (num_sample_pts - 1.0)
        return thresholds

    def compute_statistics_jit(self, overlaps, dt_scores, ignored_gt, ignored_det, min_overlap, thresh=0, compute_fp=False):
        """
        单帧内统计 TP / FP / FN 及 TP 置信度列表。
        overlaps: (N_det, N_gt)  IoU 矩阵
        dt_scores: (N_det,)      预测框置信度
        ignored_gt: (N_gt,)      -1 忽略, 0 参与
        ignored_det: (N_det,)    -1 忽略, 0 参与
        min_overlap: float       IoU 阈值
        thresh: float            置信度阈值（仅 compute_fp=True 时生效）
        compute_fp: bool         是否计算 FP
        """
        n_det = overlaps.shape[0]
        n_gt = overlaps.shape[1]
        assigned_det = [False] * n_det
        ignored_thresh = [False] * n_det
        if compute_fp:
            for i in range(n_det):
                if dt_scores[i] < thresh:
                    ignored_thresh[i] = True
        NO_DETECTION = -1e8
        tp = fp = fn = 0
        thresholds = np.zeros(n_gt, dtype=np.float32)
        thresh_idx = 0
        for g in range(n_gt):
            if ignored_gt[g] == -1:
                continue
            best_det_idx = -1
            best_score = NO_DETECTION
            max_iou = 0.0
            for d in range(n_det):
                if ignored_det[d] == -1 or assigned_det[d] or ignored_thresh[d]:
                    continue
                iou = overlaps[d, g]
                score = dt_scores[d]
                if not compute_fp and iou >= min_overlap and score > best_score:
                    best_det_idx = d
                    best_score = score
                elif compute_fp and iou >= min_overlap and iou > max_iou:
                    max_iou = iou
                    best_det_idx = d
                    best_score = 1.0
            if best_score == NO_DETECTION:
                fn += 1
            else:
                tp += 1
                thresholds[thresh_idx] = dt_scores[best_det_idx]
                thresh_idx += 1
                assigned_det[best_det_idx] = True
        if compute_fp:
            for d in range(n_det):
                if (not assigned_det[d] and ignored_det[d] != -1 and not ignored_thresh[d]):
                    fp += 1
        return tp, fp, fn, thresholds[:thresh_idx]

    def calc_iou(self, gt_boxes, pt_boxes):
        """计算IoU，支持BEV和3D"""
        if isinstance(gt_boxes, np.ndarray):
            gt_boxes = torch.from_numpy(gt_boxes)
        if isinstance(pt_boxes, np.ndarray):
            pt_boxes = torch.from_numpy(pt_boxes)
        if self.iou_type == 'bev':
            #ious = boxes_iou_bev(pt_boxes.cuda(), gt_boxes.cuda())
            ious = self.boxes_iou_bev_cpu(pt_boxes.numpy(), gt_boxes.numpy())
        elif self.iou_type == '3d':
            ious = boxes_iou3d_gpu(pt_boxes.cuda(), gt_boxes.cuda())
            ious = ious.cpu().data.numpy()
        else:
            ious = boxes_iou_bev(pt_boxes.cuda(), gt_boxes.cuda())
            ious = ious.cpu().data.numpy()
        return ious

    def boxes_iou_bev_cpu(self, pt_boxes, gt_boxes):
        """
        计算BEV（Bird's Eye View）IoU，CPU实现。
        输入:
            pt_boxes: (N, 7) 或 (N, 5) [x, y, z, l, w, h, yaw] 或 [x, y, l, w, yaw]
            gt_boxes: (M, 7) 或 (M, 5)
        输出:
            ious: (N, M) IoU矩阵
        """
        def get_corners(box):
            # box: [x, y, l, w, yaw]
            x, y, l, w, yaw = box
            c, s = np.cos(yaw), np.sin(yaw)
            dx = l / 2
            dy = w / 2
            corners = np.array([
                [dx, dy],
                [dx, -dy],
                [-dx, -dy],
                [-dx, dy]
            ])
            rot = np.array([[c, -s], [s, c]])
            corners = corners @ rot.T
            corners += np.array([x, y])
            return corners

        def poly_area(corners):
            x = corners[:, 0]
            y = corners[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        def poly_intersection(c1, c2):
            try:
                poly1 = Polygon(c1)
                poly2 = Polygon(c2)
                if not poly1.is_valid or not poly2.is_valid:
                    return 0.0
                inter = poly1.intersection(poly2)
                return inter.area
            except ImportError:
                # fallback: no intersection
                return 0.0

        # 只取BEV部分
        if pt_boxes.shape[1] == 7:
            pt_boxes_bev = pt_boxes[:, [0, 1, 3, 4, 6]]
        else:
            pt_boxes_bev = pt_boxes
        if gt_boxes.shape[1] == 7:
            gt_boxes_bev = gt_boxes[:, [0, 1, 3, 4, 6]]
        else:
            gt_boxes_bev = gt_boxes

        N = pt_boxes_bev.shape[0]
        M = gt_boxes_bev.shape[0]
        ious = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            corners1 = get_corners(pt_boxes_bev[i])
            area1 = poly_area(corners1)
            for j in range(M):
                corners2 = get_corners(gt_boxes_bev[j])
                area2 = poly_area(corners2)
                inter = poly_intersection(corners1, corners2)
                union = area1 + area2 - inter
                ious[i, j] = inter / union if union > 0 else 0.0
        return ious


    def limit_period(self, val, offset=0.5, period=np.pi):
        """角度归一化到[-π, π]"""
        return val - np.floor(val / period + offset) * period

    def merge_list(self, annos, pcd=None):
        """合并标注列表为box、score、label数组，并过滤无效框"""
        box3d, scores, labels = [], [], []
        for anno in annos:
            center = anno['translation']
            cls_name = anno['detection_name']
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
            box = center + size + yaw
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

    def _load_results(self):
        """读取预测和GT结果json"""
        with open(self.pt_res_path, 'r') as f:
            pt = json.load(f)
        with open(self.gt_res_path, 'r') as f:
            gt = json.load(f)
        print('pred frames:', len(pt), 'gt frames:', len(gt))
        return pt, gt

    def _init_stats(self):
        """初始化统计容器"""
        stats = SimpleNamespace()
        stats.thresholdss = defaultdict(list)
        stats.valid_gt_nums = defaultdict(int)
        stats.valid_pt_nums = defaultdict(int)
        stats.det_points_info = defaultdict(list)
        stats.gt_points_info = defaultdict(list)
        stats.aoe = defaultdict(list)
        stats.ate_x = defaultdict(list)
        stats.ate_y = defaultdict(list)
        stats.ase_x = defaultdict(list)
        stats.ase_y = defaultdict(list)
        stats.all_ious = []
        stats.pt_scores_list = []
        stats.pt_labels_list = []
        stats.gt_labels_list = []
        return stats

    def _load_pcd(self, annos):
        """加载点云数据"""
        try:
            if self.filter_nums > 0:
                try:
                    pcd = np.load(annos[0]['pts_filename'])['pcd']
                except:
                    pcd = np.load(annos[0]['pts_filename'])['pcds']
            else:
                pcd = None
        except:
            pcd = None
        return pcd

    def _dist_xy(self, boxes):
        """按x/y坐标分bin"""
        x_bin = np.clip(np.digitize(boxes[:, 0], self.first_bins) - 1, 0, self.n_first - 1)
        y_bin = np.clip(np.digitize(boxes[:, 1], self.second_bins) - 1, 0, self.n_second - 1)
        return x_bin, y_bin

    def _angle_from_xy(self, x, y):
        """计算xy对应的角度，顺时针为正，范围[0,360)"""
        yaw = np.arctan2(-y, x)
        deg = (np.degrees(yaw) + 360) % 360
        return deg

    def _process_frame(self, token, annos, pt_results, stats):
        """处理单帧，统计各类指标"""
        pcd = self._load_pcd(annos)
        gt_boxes, _, gt_labels = self.merge_list(annos, pcd)
        pt_boxes, pt_scores, pt_labels = self.merge_list(pt_results[token], pcd)
        if gt_boxes.shape[0] == 0 or pt_boxes.shape[0] == 0:
            stats.all_ious.append(None)
            stats.pt_scores_list.append(pt_scores)
            stats.pt_labels_list.append(pt_labels)
            stats.gt_labels_list.append(gt_labels)
            return
        gt_first, gt_second = self._dist_xy(gt_boxes)
        pt_first, pt_seconde = self._dist_xy(pt_boxes)
        frame_ious = [[None] * self.n_second for _ in range(self.n_first)]
        frame_pt_s = [[None] * self.n_second for _ in range(self.n_first)]
        frame_pt_l = [[None] * self.n_second for _ in range(self.n_first)]
        frame_gt_l = [[None] * self.n_second for _ in range(self.n_first)]
        for first_idx in range(self.n_first):
            g_mask_first = gt_first == first_idx
            p_mask_first = pt_first == first_idx
            for second_idx in range(self.n_second):
                g_mask_second = gt_second == second_idx
                p_mask_second = pt_seconde == second_idx
                g_mask = g_mask_first & g_mask_second
                p_mask = p_mask_first & p_mask_second
                g_b = gt_boxes[g_mask]
                p_b = pt_boxes[p_mask]
                g_l = gt_labels[g_mask]
                p_l = pt_labels[p_mask]
                p_s = pt_scores[p_mask]
                if g_b.shape[0] == 0 and p_b.shape[0] == 0:
                    continue
                frame_pt_s[first_idx][second_idx] = p_s
                frame_pt_l[first_idx][second_idx] = p_l
                frame_gt_l[first_idx][second_idx] = g_l
                if g_b.shape[0] and p_b.shape[0]:
                    ious = self.calc_iou(g_b, p_b)
                else:
                    ious = None
                    # 漏检点（FN）
                    if g_b.shape[0] > 0 and (p_b.shape[0] == 0 or ious is None):
                        for gt_box, gt_lable in zip(g_b, g_l):
                            stats.gt_points_info['miss'].append({
                                'coord': gt_box[:3].tolist(),
                                'class': self.det_classes[gt_lable],
                                'bin': (first_idx, second_idx),
                                'error_type': 'miss'
                            })
                    # 误检点（FP）
                    if p_b.shape[0] > 0 and (g_b.shape[0] == 0 or ious is None):
                        for pt_box, pt_label in zip(p_b, p_l):
                            stats.det_points_info['false'].append({
                                'coord': pt_box[:3].tolist(),
                                'class': self.det_classes[pt_label],
                                'bin': (first_idx, second_idx),
                                'error_type': 'false'
                            })
                    continue
                frame_ious[first_idx][second_idx] = ious
                self._update_error(ious, g_b, p_b, g_l, p_l, p_s, first_idx, second_idx, stats)
                for cls_idx, cls in enumerate(self.det_classes):
                    self._update_pr(ious, g_l, p_l, p_s, cls_idx, first_idx, second_idx, stats)
                # 匹配点（TP）
                if ious is not None and g_b.shape[0] > 0 and p_b.shape[0] > 0:
                    iou_idxs = np.argmax(ious, axis=0)
                    iou_values = np.max(ious, axis=0)
                    iou_mask = iou_values >= self.iou_threshs[0]
                    score_mask = p_s[iou_idxs] >= 0.1
                    label_mask = (g_l == p_l[iou_idxs])
                    match_mask = iou_mask & score_mask & label_mask
                    for i, matched in enumerate(match_mask):
                        if matched:
                            # stats.gt_points_info['match'].append({
                            #     'coord': g_b[i][:3].tolist(),
                            #     'bin': (first_idx, second_idx),
                            #     'error_type': 'match',
                            #     'det_coord': p_b[iou_idxs[i]][:3].tolist(),
                            #     'error': np.abs(g_b[i][:] - p_b[iou_idxs[i]][:]).tolist()
                            # })
                            stats.det_points_info['match'].append({
                                'coord': p_b[iou_idxs[i]][:3].tolist(),
                                'class': self.det_classes[p_l[iou_idxs[i]]],
                                'bin': (first_idx, second_idx),
                                'error_type': 'match',
                                'gt_coord': g_b[i][:3].tolist(),
                                'error': np.abs(g_b[i][:] - p_b[iou_idxs[i]][:]).tolist()
                            })
        stats.pt_scores_list.append(frame_pt_s)
        stats.pt_labels_list.append(frame_pt_l)
        stats.gt_labels_list.append(frame_gt_l)
        stats.all_ious.append(frame_ious)

    def _update_error(self, ious, g_b, p_b, g_l, p_l, p_s, d, second_idx, st):
        """误差统计（AOE、ATE、ASE）"""
        if ious is None: return
        iou_idxs = np.argmax(ious, axis=0)
        iou_values = np.max(ious, axis=0)
        iou_mask = iou_values >= self.iou_threshs[0]
        score_mask = p_s[iou_idxs] >= 0.1
        label_mask = (g_l == p_l[iou_idxs])
        match_mask = iou_mask & score_mask & label_mask
        gt_match_boxes = g_b[match_mask, :]
        pt_match_boxes = p_b[iou_idxs, :][match_mask, :]
        gt_match_labels = g_l[match_mask]
        if pt_match_boxes.shape[0] <= 0 or gt_match_boxes.shape[0] <= 0:
            return
        gt_yaw = self.limit_period(gt_match_boxes[:, 6], 0.5, 2 * np.pi)
        pt_yaw = self.limit_period(pt_match_boxes[:, 6], 0.5, 2 * np.pi)
        aoe = np.fabs(gt_yaw - pt_yaw)
        aoe = np.minimum(aoe, 2 * np.pi - aoe)
        aoe = np.minimum(aoe, np.pi - aoe)
        gt_x_center = gt_match_boxes[:, 0]
        pt_x_center = pt_match_boxes[:, 0]
        ate_x = np.fabs(gt_x_center - pt_x_center)
        gt_y_center = gt_match_boxes[:, 1]
        pt_y_center = pt_match_boxes[:, 1]
        ate_y = np.fabs(gt_y_center - pt_y_center)
        gt_x_size = gt_match_boxes[:, 3]
        pt_x_size = pt_match_boxes[:, 3]
        ase_x = np.fabs(gt_x_size - pt_x_size)
        gt_y_size = gt_match_boxes[:, 4]
        pt_y_size = pt_match_boxes[:, 4]
        ase_y = np.fabs(gt_y_size - pt_y_size)
        for m, current_class in enumerate(self.det_classes):
            st.aoe[(m, d, second_idx)].append(aoe[gt_match_labels == m])
            st.ate_x[(m, d, second_idx)].append(ate_x[gt_match_labels == m])
            st.ate_y[(m, d, second_idx)].append(ate_y[gt_match_labels == m])
            st.ase_x[(m, d, second_idx)].append(ase_x[gt_match_labels == m])
            st.ase_y[(m, d, second_idx)].append(ase_y[gt_match_labels == m])

    def _update_pr(self, ious, g_l, p_l, p_s, cls_idx, d, second_idx, st):
        """PR统计"""
        ignored_gt = np.where(g_l == cls_idx, 0, -1)
        ignored_det = np.where(p_l == cls_idx, 0, -1)
        st.valid_gt_nums[(cls_idx, d, second_idx)] += (ignored_gt + 1).sum()
        st.valid_pt_nums[(cls_idx, d, second_idx)] += (ignored_det + 1).sum()
        for k, min_overlap in enumerate(self.iou_threshs):
            tp, fp, fn, thresholds = self.compute_statistics_jit(
                ious, p_s, ignored_gt, ignored_det, min_overlap, thresh=0.0, compute_fp=False
            )
            st.thresholdss[(cls_idx, k, d, second_idx)] += thresholds.tolist()

    def get_mAP(self, prec):
        """计算mAP"""
        sums = 0
        for i in range(1, prec.shape[-1]):
            sums = sums + prec[..., i]
        return sums / (self.N_SAMPLE_PTS - 1) * 100

    def _summarize(self, save_path, stats):
        """汇总所有统计结果并生成JSON"""
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
                        elif n_pt == 0:
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
                                if all_ious is None:
                                    tp = 0
                                    fp = 0
                                    fn = 0
                                elif gt_l[first_idx][second_idx] is None:
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
                ate_x_classes = None
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
                    if ate_x_classes is None:
                        ate_x_classes = ate_x
                    else:
                        ate_x_classes = np.concatenate([ate_x_classes, ate_x])
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
                # ATE_X no classes
                ate_x_tp95_classes = np.percentile(ate_x_classes, 95) if len(ate_x_classes) else 0
                ate_x_mean_classes = ate_x_classes.mean() if len(ate_x_classes) else 0
                details[f'ate_x_tp95_{first_idx}_{second_idx}'] = f'{ate_x_tp95_classes:.2f}'
                details[f'ate_x_{first_idx}_{second_idx}'] = f'{ate_x_mean_classes:.2f}'
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

        # 全局统计
        global_mpre = [0.0] * cls_nums
        global_mrec = [0.0] * cls_nums
        global_denom_pt = [0.0] * cls_nums
        global_denom_gt = [0.0] * cls_nums

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
                            # 全局
                            global_mpre[cls_idx] += best_pre
                            global_mrec[cls_idx] += best_rec
                            global_denom_pt[cls_idx] += 1  # macro, denom = number of bins or classes? Wait, for macro, denom = metric_cls_cnt
                            global_denom_gt[cls_idx] += 1
                        else:  # micro
                            total_gt[first_idx][second_idx][iou_idx] += n_gt
                            mpre_all[first_idx][second_idx][iou_idx] += best_pre * n_pt
                            mrec_all[first_idx][second_idx][iou_idx] += best_rec * n_gt
                            total_pt[first_idx][second_idx][iou_idx] += n_pt
                            # 全局 micro 需要累加 n_pt, n_gt
                            global_mpre[cls_idx] += best_pre * n_pt
                            global_mrec[cls_idx] += best_rec * n_gt
                            global_denom_pt[cls_idx] += n_pt
                            global_denom_gt[cls_idx] += n_gt

        # ---------- 4. 写 JSON ----------
        # os.makedirs(os.path.join(save_path, 'plots'), exist_ok=True)
        # metric_path = os.path.join(save_path, f'metric_details_{self.epoch}.json')
        # with open(metric_path, 'w') as f:
        #     json.dump(metric_details, f, indent=2)

        # ---------- 5. summary ----------
        summary['score_list'] = [float(s) for s in score_list]
        # summary.update(details)

        # 全局每个类别的指标
        global_miss_list = []
        global_falsedet_list = []
        global_ate_x_tp95_list = []
        global_ate_x_mean_list = []
        global_ate_y_tp95_list = []
        global_ate_y_mean_list = []
        global_aoe_tp95_list = []
        global_aoe_mean_list = []
        for cls_idx, cls_name in enumerate(self.det_classes):
            if cls_name not in self.metric_classes:
                continue
            # 全局 miss 和 falsedet
            if self.metric_type == 'macro':
                denom_pt = global_denom_pt[cls_idx] if global_denom_pt[cls_idx] > 0 else 1
                denom_gt = global_denom_gt[cls_idx] if global_denom_gt[cls_idx] > 0 else 1
            else:
                denom_pt = global_denom_pt[cls_idx] or 1
                denom_gt = global_denom_gt[cls_idx] or 1
            global_miss = 100 - global_mrec[cls_idx] / denom_gt * 100
            global_falsedet = 100 - global_mpre[cls_idx] / denom_pt * 100
            summary[f'global_miss_{cls_name}'] = f'{global_miss:.2f}'
            summary[f'global_falsedet_{cls_name}'] = f'{global_falsedet:.2f}'
            global_miss_list.append(global_miss)
            global_falsedet_list.append(global_falsedet)

            # 全局误差
            all_ate_x = []
            all_ate_y = []
            all_aoe = []
            for first_idx in range(first_nums):
                for second_idx in range(second_nums):
                    all_ate_x.extend(np.concatenate(stats.ate_x[(cls_idx, first_idx, second_idx)] or [[]]))
                    all_ate_y.extend(np.concatenate(stats.ate_y[(cls_idx, first_idx, second_idx)] or [[]]))
                    all_aoe.extend(np.concatenate(stats.aoe[(cls_idx, first_idx, second_idx)] or [[]]))
            if all_ate_x:
                global_ate_x_tp95 = np.percentile(all_ate_x, 95)
                global_ate_x_mean = np.mean(all_ate_x)
                summary[f'global_ate_x_tp95_{cls_name}'] = f'{global_ate_x_tp95:.2f}'
                summary[f'global_ate_x_mean_{cls_name}'] = f'{global_ate_x_mean:.2f}'
                global_ate_x_tp95_list.append(global_ate_x_tp95)
                global_ate_x_mean_list.append(global_ate_x_mean)
            if all_ate_y:
                global_ate_y_tp95 = np.percentile(all_ate_y, 95)
                global_ate_y_mean = np.mean(all_ate_y)
                summary[f'global_ate_y_tp95_{cls_name}'] = f'{global_ate_y_tp95:.2f}'
                summary[f'global_ate_y_mean_{cls_name}'] = f'{global_ate_y_mean:.2f}'
                global_ate_y_tp95_list.append(global_ate_y_tp95)
                global_ate_y_mean_list.append(global_ate_y_mean)
            if all_aoe:
                global_aoe_tp95 = np.percentile(np.rad2deg(all_aoe), 95)
                global_aoe_mean = np.mean(np.rad2deg(all_aoe))
                summary[f'global_aoe_tp95_{cls_name}'] = f'{global_aoe_tp95:.2f}'
                summary[f'global_aoe_mean_{cls_name}'] = f'{global_aoe_mean:.2f}'
                global_aoe_tp95_list.append(global_aoe_tp95)
                global_aoe_mean_list.append(global_aoe_mean)

        # 六个类别的均值
        if global_miss_list:
            summary['global_miss_all_classes'] = f'{np.mean(global_miss_list):.2f}'
        if global_falsedet_list:
            summary['global_falsedet_all_classes'] = f'{np.mean(global_falsedet_list):.2f}'
        if global_ate_x_tp95_list:
            summary['global_ate_x_tp95_all_classes'] = f'{np.mean(global_ate_x_tp95_list):.2f}'
        if global_ate_x_mean_list:
            summary['global_ate_x_mean_all_classes'] = f'{np.mean(global_ate_x_mean_list):.2f}'
        if global_ate_y_tp95_list:
            summary['global_ate_y_tp95_all_classes'] = f'{np.mean(global_ate_y_tp95_list):.2f}'
        if global_ate_y_mean_list:
            summary['global_ate_y_mean_all_classes'] = f'{np.mean(global_ate_y_mean_list):.2f}'
        if global_aoe_tp95_list:
            summary['global_aoe_tp95_all_classes'] = f'{np.mean(global_aoe_tp95_list):.2f}'
        if global_aoe_mean_list:
            summary['global_aoe_mean_all_classes'] = f'{np.mean(global_aoe_mean_list):.2f}'

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
        result['det_points_info'] = stats.det_points_info
        result['gt_points_info'] = stats.gt_points_info
        result[self.metric_key] = f'{mAP[:, -1, :].mean():.2f}'

        # # 控制台打印
        # for k, v in summary.items():
        #     print(f'{k}: {v}')
        return result
    
    def get_metric_new(self, save_path=''):
        """主入口：计算所有指标并保存"""
        pt_results, gt_results = self._load_results()
        stats = self._init_stats()
        for token, annos in tqdm.tqdm(gt_results.items()):
            self._process_frame(token, annos, pt_results, stats)
        result_dic = self._summarize(save_path, stats)
        # 保存加权均值到txt文件
        txt_path = os.path.join(save_path, f'weighted_metrics.txt')
        with open(txt_path, 'w') as f:
            f.write("\nGlobal Metrics per Class:\n")
            for cls_name in self.metric_classes:
                f.write(f"Global Miss {cls_name}: {result_dic['summary'].get(f'global_miss_{cls_name}', 'N/A')}\n")
                f.write(f"Global Falsedet {cls_name}: {result_dic['summary'].get(f'global_falsedet_{cls_name}', 'N/A')}\n")
                f.write(f"Global ATE_X TP95 {cls_name}: {result_dic['summary'].get(f'global_ate_x_tp95_{cls_name}', 'N/A')}\n")
                f.write(f"Global ATE_X Mean {cls_name}: {result_dic['summary'].get(f'global_ate_x_mean_{cls_name}', 'N/A')}\n")
                f.write(f"Global ATE_Y TP95 {cls_name}: {result_dic['summary'].get(f'global_ate_y_tp95_{cls_name}', 'N/A')}\n")
                f.write(f"Global ATE_Y Mean {cls_name}: {result_dic['summary'].get(f'global_ate_y_mean_{cls_name}', 'N/A')}\n")
                f.write(f"Global AOE TP95 {cls_name}: {result_dic['summary'].get(f'global_aoe_tp95_{cls_name}', 'N/A')}\n")
                f.write(f"Global AOE Mean {cls_name}: {result_dic['summary'].get(f'global_aoe_mean_{cls_name}', 'N/A')}\n")
                f.write("\n")
            f.write("Global Metrics All Classes Mean:\n")
            f.write(f"Global Miss All Classes: {result_dic['summary'].get('global_miss_all_classes', 'N/A')}\n")
            f.write(f"Global Falsedet All Classes: {result_dic['summary'].get('global_falsedet_all_classes', 'N/A')}\n")
            f.write(f"Global ATE_X TP95 All Classes: {result_dic['summary'].get('global_ate_x_tp95_all_classes', 'N/A')}\n")
            f.write(f"Global ATE_X Mean All Classes: {result_dic['summary'].get('global_ate_x_mean_all_classes', 'N/A')}\n")
            f.write(f"Global ATE_Y TP95 All Classes: {result_dic['summary'].get('global_ate_y_tp95_all_classes', 'N/A')}\n")
            f.write(f"Global ATE_Y Mean All Classes: {result_dic['summary'].get('global_ate_y_mean_all_classes', 'N/A')}\n")
            f.write(f"Global AOE TP95 All Classes: {result_dic['summary'].get('global_aoe_tp95_all_classes', 'N/A')}\n")
            f.write(f"Global AOE Mean All Classes: {result_dic['summary'].get('global_aoe_mean_all_classes', 'N/A')}\n")
        return result_dic

if __name__ == '__main__':
    # 主流程入口
    today = date.today()
    date_str = today.strftime("%Y-%m-%d")
    data_flag = 'bevfusion'
    version_flag = 'v32_B2_second1231_3dseg_cls21_cam5lidar_intensityraw_fromt12025_42wmax50_newdata_v2'
    metric_type = 'macro'
    Classes = ("person", "bike", "rider", "car", "truck", "bus")
    point_cloud_range_v1 = [-81.6, -48, -1, 97.6, 48, 3.0]
    range_dic = {
        "v1": {"point_cloud_range": point_cloud_range_v1, "metric_classes": Classes},
    }
    save_path = f'/pcpt/pcpt/project/liqinbu/data/{data_flag}/{version_flag}_metric_first_second{metric_type}'
    if not os.path.exists(save_path):
        os.system(f"mkdir -p {save_path}")
    gt_res_path = f'/pcpt/pcpt/project/liqinbu/data/{data_flag}/{version_flag}_{date_str}gt_filter.json'
    pt_res_path = f'/pcpt/pcpt/project/liqinbu/data/{data_flag}/{version_flag}_{date_str}pt.json'
    for k, v in range_dic.items():
        print(f'===================== metric version: {metric_type}-{k} =====================')
        det_metric = Det3DMetric(
            iou_type='bev',
            Classes=Classes,
            iou_threshs=[0.1, 0.25, 0.5],
            point_cloud_range=v['point_cloud_range'],
            filter_nums=0,
            metric_key='mAP',
            metric_type=metric_type,
            range_version=k,
            metric_classes=v['metric_classes'],
        )
        det_metric.pt_res_path = pt_res_path
        det_metric.gt_res_path = gt_res_path
        result_dic = det_metric.get_metric_new(save_path)
        with open(os.path.join(save_path, f'metric_{k}.json'), 'w') as f:
            json.dump(result_dic, f)
