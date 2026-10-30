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
from scipy.spatial import Delaunay
# sys.path.insert(0, "/rbs/jiangjunyuan/code/BigFusion/camera_e2e")

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
    def __init__(self, iou_type="3d", Classes=[], iou_threshs=[], point_cloud_range=[-102.4, -102.4, -5.0, 102.4, 102.4, 5.0], filter_nums=0, metric_key='mAP'):
        self.iou_type = iou_type
        self.det_classes = Classes
        self.iou_threshs = iou_threshs
        self.metric_key = metric_key
        self.class_nums = len(Classes)
        self.point_cloud_range = point_cloud_range
        self.filter_nums = filter_nums
        self.pt_detections = dict()
        self.gt_detections = dict()
        self.pcds_result = dict()
        self.N_SAMPLE_PTS = 41
        self.epoch = 0
        self.beta = 1.0

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
        det_size = overlaps.shape[0]
        gt_size = overlaps.shape[1]

        assigned_detection = [False] * det_size
        ignored_threshold = [False] * det_size
        if compute_fp:
            for i in range(det_size):
                if (dt_scores[i] < thresh):
                    ignored_threshold[i] = True

        NO_DETECTION = -10000000
        tp, fp, fn = 0, 0, 0
        thresholds = np.zeros((gt_size, ))
        thresh_idx = 0
        for i in range(gt_size):
            if ignored_gt[i] == -1:
                continue
            det_idx = -1
            valid_detection = NO_DETECTION
            max_overlap = 0
            assigned_ignored_det = False

            for j in range(det_size):
                if (ignored_det[j] == -1):
                    continue
                if (assigned_detection[j]):
                    continue
                if (ignored_threshold[j]):
                    continue
                overlap = overlaps[j, i]
                dt_score = dt_scores[j]
                if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                    det_idx = j
                    valid_detection = dt_score
                elif (compute_fp and (overlap > min_overlap)
                      and (overlap > max_overlap or assigned_ignored_det)
                      and ignored_det[j] == 0):
                    max_overlap = overlap
                    det_idx = j
                    valid_detection = 1
                    assigned_ignored_det = False
                # elif (compute_fp and (overlap > min_overlap)
                #       and (valid_detection == NO_DETECTION)
                #       and ignored_det[j] == 1):
                #     det_idx = j
                #     valid_detection = 1
                #     assigned_ignored_det = True

            if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
                fn += 1
            elif valid_detection != NO_DETECTION:
                tp += 1
                thresholds[thresh_idx] = dt_scores[det_idx]
                thresh_idx += 1
                assigned_detection[det_idx] = True
        
        if compute_fp:
            for j in range(det_size):
                if (not (assigned_detection[j] or ignored_det[j] == -1 or ignored_threshold[j])):
                    fp += 1

        return tp, fp, fn, thresholds[:thresh_idx]

    def calc_iou(self, gt_boxes, pt_boxes):
        from utils import nms
        '''
        calc iou in cuda and tensor!!
        '''
        if isinstance(gt_boxes, np.ndarray):
            gt_boxes = torch.from_numpy(gt_boxes)

        if isinstance(pt_boxes, np.ndarray):
            pt_boxes = torch.from_numpy(pt_boxes)

        if self.iou_type == 'bev':
            ious = nms.boxes_iou_bev(pt_boxes.cuda(), gt_boxes.cuda())
        elif self.iou_type == '3d':
            ious = nms.boxes_iou3d_gpu(pt_boxes.cuda(), gt_boxes.cuda())
        else:
            ious = nms.boxes_iou_bev(pt_boxes.cuda(), gt_boxes.cuda())
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
            # TODO check this!!
            if center[0] < self.point_cloud_range[0] or \
               center[1] < self.point_cloud_range[1] or \
               center[2] < self.point_cloud_range[2]:
                continue
            if center[0] > self.point_cloud_range[3] or \
               center[1] > self.point_cloud_range[4] or \
               center[2] > self.point_cloud_range[5]:
                continue

            size = anno['size']
            yaw = anno['rotation']

            box = center+size+yaw
            if pcd is not None and self.get_boxs_pts_num(box, pcd) <= self.filter_nums:
                continue

            score = anno['detection_score']
            label = self.det_classes.index(anno['detection_name'])

            box3d.append(box)
            scores.append(score)
            labels.append(label)

        box3d = np.array(box3d).astype(np.float32)
        scores = np.array(scores).astype(np.float32)
        labels = np.array(labels).astype(np.uint8)
        return box3d, scores, labels

    def get_metric(self, save_path=''):
        """Update state with predictions and targets."""
        '''
        preds_boxes, target_boxes: 'box3d_lidar', 'label_preds', 'scores'
        box3d_lidar: N x 7
        label_preds: N 
        scores: N
        '''
        # self.pt_res_path = os.path.join(save_path, 'pt_result_{}.json'.format(23))
        # self.gt_res_path = os.path.join(save_path, 'gt_result.json')

        with open(self.pt_res_path, 'r') as f:
            pt_results = json.load(f)

        with open(self.gt_res_path, 'r') as f:
            gt_results = json.load(f)

        print('pred detection nums:', len(pt_results.keys()))
        print('gt detection nums:', len(gt_results.keys()))
        iou_nums = len(self.iou_threshs)
        thresholdss = [[[] for i in range(iou_nums)] for j in range(self.class_nums)]

        pt_scores_list, pt_labels_list, gt_labels_list = [], [], []
        all_ious_list = []
        total_num_valid_gt = [0 for i in range(self.class_nums)]

        aoe_list = [[] for j in range(self.class_nums)]
        ate_x_list = [[] for j in range(self.class_nums)]
        ate_y_list = [[] for j in range(self.class_nums)]
        ase_x_list = [[] for j in range(self.class_nums)]
        ase_y_list = [[] for j in range(self.class_nums)]
        for token, annos in tqdm.tqdm(gt_results.items()):
            ## get pcd
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

            gt_boxes,gt_scores,gt_labels = self.merge_list(annos, pcd)
            pt_boxes,pt_scores,pt_labels = self.merge_list(pt_results[token], pcd)

            pt_scores_list.append(pt_scores)
            pt_labels_list.append(pt_labels)
            gt_labels_list.append(gt_labels)

            ## no detection
            if pt_boxes.shape[0] <= 0 or gt_boxes.shape[0] <=0 :
                all_ious_list.append(None)
                continue
            all_ious = self.calc_iou(gt_boxes, pt_boxes) ## M x N
            ## metric yaw error
            iou_idxs = np.argmax(all_ious, axis=0)
            iou_values = np.max(all_ious, axis=0)
            iou_mask = iou_values >= self.iou_threshs[0]
            score_mask = pt_scores[iou_idxs] >= 0.1
            match_mask = iou_mask & score_mask
            gt_match_boxes = gt_boxes[match_mask,:]
            pt_match_boxes = pt_boxes[iou_idxs,:][match_mask,:]
            gt_match_labels = gt_labels[match_mask]

            ## 平均角度误差-aoe
            gt_yaw = limit_period(gt_match_boxes[:,6], 0.5, 2*np.pi)
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
                aoe_list[m].append(aoe[gt_match_labels==m])
                ate_x_list[m].append(ate_x[gt_match_labels==m])
                ate_y_list[m].append(ate_y[gt_match_labels==m])
                ase_x_list[m].append(ase_x[gt_match_labels==m])
                ase_y_list[m].append(ase_y[gt_match_labels==m])
                
            all_ious_list.append(all_ious)

            for m, current_class in enumerate(self.det_classes):
                ignored_gt = np.where(gt_labels==m, 0, -1)
                ignored_det = np.where(pt_labels==m, 0, -1)
                total_num_valid_gt[m] += (ignored_gt+1).sum()
                for k, min_overlap in enumerate(self.iou_threshs):
                    rets = self.compute_statistics_jit(
                            all_ious,
                            pt_scores,
                            ignored_gt,
                            ignored_det,
                            min_overlap,
                            thresh=0.0,
                            compute_fp=False
                    )
                    tp, fp, fn, thresholds = rets
                    thresholdss[m][k] += thresholds.tolist()

        precision = np.zeros([self.class_nums, iou_nums, self.N_SAMPLE_PTS])
        recall = np.zeros([self.class_nums, iou_nums, self.N_SAMPLE_PTS])
        scores = np.zeros([self.class_nums, iou_nums, self.N_SAMPLE_PTS])
        det_nums = np.zeros([self.class_nums, iou_nums, self.N_SAMPLE_PTS])
        for m, current_class in enumerate(self.det_classes):
            for k, min_overlap in enumerate(self.iou_threshs):
                np_thresholdss = np.array(thresholdss[m][k])
                # np.savez('th.npz', th=np_thresholdss)
                # print(total_num_valid_gt[m], len(np_thresholdss))
                thresholds = self.get_thresholds(np_thresholdss, total_num_valid_gt[m], self.N_SAMPLE_PTS)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 3])
                for t, thresh in enumerate(thresholds):
                    for idx, all_ious in enumerate(all_ious_list):
                        gt_labels = gt_labels_list[idx]
                        pt_labels = pt_labels_list[idx]
                        if all_ious is None:
                            tp, fp, fn = 0,0,0
                            if gt_labels.shape[0] > 0:
                                ignored_gt = np.where(gt_labels == m, 0, -1)
                                fn = (ignored_gt+1).sum()
                            if pt_labels.shape[0] > 0:
                                ignored_det = np.where(pt_labels == m, 0, -1)
                                fp = (ignored_det+1).sum()
                        else:
                            ignored_gt = np.where(gt_labels == m, 0, -1)
                            ignored_det = np.where(pt_labels == m, 0, -1)
                            # print(all_ious.shape, ignored_det.shape, ignored_gt.shape)
                            tp, fp, fn, _ = self.compute_statistics_jit(
                                all_ious,
                                pt_scores_list[idx],
                                ignored_gt,
                                ignored_det,
                                min_overlap,
                                thresh=thresh,
                                compute_fp=True
                            )
                        pr[t, 0] += tp
                        pr[t, 1] += fp
                        pr[t, 2] += fn
                        det_nums[m, k, t] += tp + fp
                scores[m, k, :len(thresholds)] = thresholds
                for t, thresh in enumerate(thresholds):
                    recall[m, k, t] = pr[t, 0] / (pr[t, 0] + pr[t, 2])
                    precision[m, k, t] = pr[t, 0] / (pr[t, 0] + pr[t, 1])

                for t, thresh in enumerate(thresholds):
                    precision[m, k, t] = np.max(precision[m, k, t:], axis=-1)
                    recall[m, k, t] = np.max(recall[m, k, :(t + 1)], axis=-1)

        ## summary detail
        result_dic = collections.OrderedDict()
        metric_details = {}

        save_plots_path = os.path.join(save_path, 'plots')
        if not os.path.exists(save_plots_path):
            os.system("mkdir -p {}".format(save_plots_path))

        all_aoe_mean = 0
        all_aoe_tp95 = 0
        all_ate_x_mean, all_ate_y_mean = 0, 0
        all_ate_x_tp95, all_ate_y_tp95 = 0, 0
        all_ase_x_mean, all_ase_y_mean = 0, 0
        all_ase_x_tp95, all_ase_y_tp95 = 0, 0

        details_dic = {}
        summary_dic = {}
        for m, current_class in enumerate(self.det_classes):
            aoe_e_l = []
            for aoes in aoe_list[m]:
                aoe_e_l.extend(aoes)
            aoe_class = np.rad2deg(np.array(aoe_e_l))
            # ## 绘制数据分布图
            # plt.figure("AOE Distribution")
            # plt.title('AOE Distribution')
            # plt.xlabel('AOE')
            # plt.ylabel('Frequency')
            # plt.hist(aoe_class, bins=30, alpha=0.5, color='g', edgecolor='black')
            # plt.savefig(os.path.join(save_plots_path, 'aoe_{}.png'.format(current_class)))

            # Calculate TP95 angle error
            aoe_tp95 = np.percentile(aoe_class, 95) if len(aoe_class) > 0 else 0
            all_aoe_tp95 += aoe_tp95
            aoe_mean = aoe_class.mean()
            all_aoe_mean += aoe_mean
            details_dic.update({
                "aoe_tp95_{}".format(current_class): '{:.2f}'.format(float(aoe_tp95)),
                "aoe_{}".format(current_class): '{:.2f}'.format(float(aoe_mean))
            })

            ## 平均横向误差-x
            ate_x_l = []
            for ates in ate_x_list[m]:
                ate_x_l.extend(ates)
            ate_x_class = np.array(ate_x_l)
            # ## 绘制数据分布图
            # plt.figure("ATE-X Distribution")
            # plt.title('ATE-X Distribution')
            # plt.xlabel('ATE-X')
            # plt.ylabel('Frequency')
            # plt.hist(ate_x_class, bins=30, alpha=0.5, color='g', edgecolor='black')
            # plt.savefig(os.path.join(save_plots_path, 'ate_x_{}.png'.format(current_class)))

            # Calculate TP95 trans error
            ate_x_tp95 = np.percentile(ate_x_class, 95) if len(ate_x_class) > 0 else 0
            all_ate_x_tp95 += ate_x_tp95
            ate_x_mean = ate_x_class.mean()
            all_ate_x_mean += ate_x_mean
            details_dic.update({
                "ate_x_tp95_{}".format(current_class): '{:.2f}'.format(float(ate_x_tp95)),
                "ate_x_{}".format(current_class): '{:.2f}'.format(float(ate_x_mean))
            })

            ## 平均纵向误差-y
            ate_y_l = []
            for ates in ate_y_list[m]:
                ate_y_l.extend(ates)
            ate_y_class = np.array(ate_y_l)
            # ## 绘制数据分布图
            # plt.figure("ATE-Y Distribution")
            # plt.title('ATE-Y Distribution')
            # plt.xlabel('ATE-Y')
            # plt.ylabel('Frequency')
            # plt.hist(ate_y_class, bins=30, alpha=0.5, color='g', edgecolor='black')
            # plt.savefig(os.path.join(save_plots_path, 'ate_y_{}.png'.format(current_class)))

            ate_y_tp95 = np.percentile(ate_y_class, 95) if len(ate_y_class) > 0 else 0
            all_ate_y_tp95 += ate_y_tp95
            ate_y_mean = ate_y_class.mean()
            all_ate_y_mean += ate_y_mean
            details_dic.update({
                "ate_y_tp95_{}".format(current_class): '{:.2f}'.format(float(ate_y_tp95)),
                "ate_y_{}".format(current_class): '{:.2f}'.format(float(ate_y_mean))
            })

            ## 平均横向size误差
            ase_x_l = []
            for ases in ase_x_list[m]:
                ase_x_l.extend(ases)
            ase_x_class = np.array(ase_x_l)
            # ## 绘制数据分布图
            # plt.figure("ASE-X Distribution")
            # plt.title('ASE-X Distribution')
            # plt.xlabel('ASE-X')
            # plt.ylabel('Frequency')
            # plt.hist(ase_x_class, bins=30, alpha=0.5, color='g', edgecolor='black')
            # plt.savefig(os.path.join(save_plots_path, 'ase_x_{}.png'.format(current_class)))

            ase_x_tp95 = np.percentile(ase_x_class, 95) if len(ase_x_class) > 0 else 0
            all_ase_x_tp95 += ase_x_tp95
            ase_x_mean = ase_x_class.mean()
            all_ase_x_mean += ase_x_mean
            details_dic.update({
                "ase_x_tp95_{}".format(current_class): '{:.2f}'.format(float(ase_x_tp95)),
                "ase_x_{}".format(current_class): '{:.2f}'.format(float(ase_x_mean))
            })

            ## 平均纵向size误差
            ase_y_l = []
            for ases in ase_y_list[m]:
                ase_y_l.extend(ases)
            ase_y_class = np.array(ase_y_l)
            # ## 绘制数据分布图
            # plt.figure("ASE-Y Distribution")
            # plt.title('ASE-Y Distribution')
            # plt.xlabel('ASE-Y')
            # plt.ylabel('Frequency')
            # plt.hist(ase_y_class, bins=30, alpha=0.5, color='g', edgecolor='black')
            # plt.savefig(os.path.join(save_plots_path, 'ase_y_{}.png'.format(current_class)))

            ase_y_tp95 = np.percentile(ase_y_class, 95) if len(ase_y_class) > 0 else 0
            all_ase_y_tp95 += ase_y_tp95
            ase_y_mean = ase_y_class.mean() 
            all_ase_y_mean += ase_y_mean
            details_dic.update({
                "ase_y_tp95_{}".format(current_class): '{:.2f}'.format(float(ase_y_tp95)),
                "ase_y_{}".format(current_class): '{:.2f}'.format(float(ase_y_mean))
            })

        all_aoe_mean = all_aoe_mean / len(self.det_classes)
        all_ate_x_mean = all_ate_x_mean / len(self.det_classes)
        all_ate_y_mean = all_ate_y_mean / len(self.det_classes)
        all_ase_x_mean = all_ase_x_mean / len(self.det_classes)
        all_ase_y_mean = all_ase_y_mean / len(self.det_classes)
        all_aoe_tp95 = all_aoe_tp95 / len(self.det_classes)
        all_ate_x_tp95 = all_ate_x_tp95 / len(self.det_classes)
        all_ate_y_tp95 = all_ate_y_tp95 / len(self.det_classes)
        all_ase_x_tp95 = all_ase_x_tp95 / len(self.det_classes)
        all_ase_y_tp95 = all_ase_y_tp95 / len(self.det_classes)
        summary_dic.update({
            "AOE_tp95_all": '{:.2f}'.format(float(all_aoe_tp95)),
            "ATE_x_tp95_all": '{:.2f}'.format(float(all_ate_x_tp95)),
            "ATE_y_tp95_all": '{:.2f}'.format(float(all_ate_y_tp95)),
            "ASE_x_tp95_all": '{:.2f}'.format(float(all_ase_x_tp95)),
            "ASE_y_tp95_all": '{:.2f}'.format(float(all_ase_y_tp95))
        })

        summary_dic.update({
            "AOE_all": '{:.2f}'.format(float(all_aoe_mean)),
            "ATE_x_all": '{:.2f}'.format(float(all_ate_x_mean)),
            "ATE_y_all": '{:.2f}'.format(float(all_ate_y_mean)),
            "ASE_x_all": '{:.2f}'.format(float(all_ase_x_mean)),
            "ASE_y_all": '{:.2f}'.format(float(all_ase_y_mean))
        })

        mpre_all = [0.0 for i in range(iou_nums)]
        mrec_all = [0.0 for i in range(iou_nums)]
        total_num_valid_pt = [0.0 for i in range(iou_nums)]
        for m, current_class in enumerate(self.det_classes):
            for k, min_overlap in enumerate(self.iou_threshs):
                mop = int(min_overlap*100)
                name_key = '{}_{}'.format(current_class,mop)
                prec = precision[m, k, :]
                rec = recall[m, k, :]
                score = scores[m, k, :]
                pt_dets = det_nums[m, k, :]
                ## save pr png
                mask = prec > 0.0
                prec = prec[mask]
                rec = rec[mask]
                score = score[mask]
                pt_dets = pt_dets[mask]

                # 计算F-SCORE
                f_score = (1+self.beta) * prec * rec / (self.beta*prec + rec + 1e-12)
                max_f_idx = np.argmax(f_score)
                max_pre = prec[max_f_idx]
                max_rec = rec[max_f_idx]
                max_score = score[max_f_idx]
                max_pt_dets = pt_dets[max_f_idx]
                # mpre_all[k] += max_pre*max_pt_dets
                # mrec_all[k] += max_rec*total_num_valid_gt[m]
                # total_num_valid_pt[k] += max_pt_dets
                mpre_all[k] += max_pre
                mrec_all[k] += max_rec
                # max_f = f_score[max_f_idx]
                details_dic.update({     
                    'pre@{}_{}'.format(int(min_overlap*100), current_class): '{: .2f}'.format(float(max_pre)*100.0),
                    'rec@{}_{}'.format(int(min_overlap*100), current_class): '{: .2f}'.format(float(max_rec)*100.0),
                    'score@{}_{}'.format(int(min_overlap*100), current_class): '{: .2f}'.format(float(max_score))
                })
                
                metric_details.update({
                    name_key: 
                    {
                        'recall': rec.tolist(),
                        "precision": prec.tolist(),
                        "scores": score.tolist(),
                        "f-score": f_score.tolist()
                    }
                })

                # plt.figure("{}-PR Curve".format(name_key))
                # plt.title('{}-PR Curve'.format(name_key))
                # plt.xlabel('Recall')
                # plt.ylabel('Precision')
                # plt.plot(rec,prec)
                # plt.savefig(os.path.join(save_plots_path,'{}.png'.format(name_key)))

        mAP = self.get_mAP(precision)
        mAP_mean = mAP.mean(axis=0, keepdims=False)
        result_dic.update({
            'details': details_dic
        })
        # total_nums_all_gt = np.array(total_num_valid_gt).sum()
        # total_nums_all_pt = np.array(total_num_valid_pt).sum()
        total_nums_all_gt = len(self.det_classes)
        total_nums_all_pt = len(self.det_classes)
        for k, min_overlap in enumerate(self.iou_threshs):
            summary_dic.update({
                'mAP@{}'.format(int(min_overlap*100)): '{: .2f}'.format(float(mAP_mean[k])),
                'FalseDet@{}'.format(int(min_overlap*100)): '{: .2f}'.format(100.0-float(mpre_all[k]*100.0/total_nums_all_pt)),
                'Missing@{}'.format(int(min_overlap*100)): '{: .2f}'.format(100.0-float(mrec_all[k]*100.0/total_nums_all_gt))
            })
            for m, current_class in enumerate(self.det_classes):
                details_dic.update({
                    'ap@{}_{}'.format(int(min_overlap*100), current_class): '{: .2f}'.format(float(mAP[m,k]))
                })
        result_dic.update({
            self.metric_key: '{: .2f}'.format(float(mAP_mean[-1]))
        })
        result_dic.update({
            'summary': summary_dic
        })
        # metric_details.update(result_dic)
        metric_res_path = os.path.join(save_path, 'metric_details_{}.json'.format(self.epoch))
        with open(metric_res_path, "w") as f:
            json.dump(metric_details, f)
        print(f"Finish generate predictions for testset, save to {metric_res_path}")

        for key, value in summary_dic.items():
            print('{}:{}'.format(key, value))
        self.reset()
        return result_dic


from datetime import date

if __name__ == '__main__':
    today = date.today()
    date_str = today.strftime("%Y-%m-%d")
    # save_path = 'experiments/mogo/config_segdet_adam_cpaug_rs80_bevl_singleframe_int8_qat/rs80_L32_1242_det_int8_12'
    # pt_res_path = os.path.join(save_path, 'pt_{}_filter.json'.format(0))
    # gt_res_path = os.path.join(save_path, 'gt_filter.json')
    data_flag = 'bevfusion_eval'
    version_flag = 'v1_c3_0418'
    save_path = '/pcpt/pcpt/project/jiangjunyuan/code/BEVFusion/{}/{}_metric'.format(data_flag, version_flag)
    if not os.path.exists(save_path):
        os.system("mkdir -p {}".format(save_path))
    gt_res_path = '/pcpt/pcpt/project/jiangjunyuan/code/BEVFusion/{}/{}_gt_filter.json'.format(data_flag, version_flag)
    pt_res_path = '/pcpt/pcpt/project/jiangjunyuan/code/BEVFusion/{}/{}_pt.json'.format(data_flag, version_flag)
    
    det_metric = Det3DMetric(iou_type='bev', 
                             Classes=("person", "bike", "rider", "car", "truck", "bus"),
                             iou_threshs=[0.1, 0.25, 0.5],
                            #  iou_threshs=[0.25, 0.5, 0.75],
                            #  point_cloud_range=[4, -40, -1.0, 104, 40, 5.0],
                             point_cloud_range=[-81.6, -48, -1, 97.6, 48, 3.0],
                            #  point_cloud_range=[-10.0, -40.0, -1.0, 90.0, 40.0, 7.0],
                             filter_nums=0,
                             metric_key='mAP')
    
    det_metric.pt_res_path = pt_res_path
    det_metric.gt_res_path = gt_res_path
    # metric_path = os.path.join(save_path, "metric")
    result_dic = det_metric.get_metric(save_path)

    with open(os.path.join(save_path, 'metric.json'), 'w') as f:
        json.dump(result_dic, f)
