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
                print('pcd is None')
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

        for m, current_class in enumerate(self.det_classes):
            for k, min_overlap in enumerate(self.iou_threshs):
                mop = int(min_overlap*100)
                name_key = '{}_{}'.format(current_class,mop)
                prec = precision[m, k, :]
                rec = recall[m, k, :]
                score = scores[m, k, :]
                metric_details.update({
                    name_key: 
                    {
                        'recall': rec.tolist(),
                        "precision": prec.tolist(),
                        "scores": score.tolist()
                    }
                })
                ## save pr png
                mask = prec > 0.0
                prec = prec[mask]
                rec = rec[mask]
                plt.figure("{}-PR Curve".format(name_key))
                plt.title('{}-PR Curve'.format(name_key))
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.plot(rec,prec)
                plt.savefig(os.path.join(save_plots_path,'{}.png'.format(name_key)))

        mAP = self.get_mAP(precision)
        mAP_mean = mAP.mean(axis=0, keepdims=False)
        for k, min_overlap in enumerate(self.iou_threshs):
            result_dic.update({
                'mAP@{}'.format(int(min_overlap*100)): '{:.2f}'.format(float(mAP_mean[k]))
            })
            for m, current_class in enumerate(self.det_classes):
                result_dic.update({
                    'AP@{}_{}'.format(int(min_overlap*100), current_class): '{:.2f}'.format(float(mAP[m,k]))
                })
        result_dic.update({
            self.metric_key: '{:.2f}'.format(float(mAP_mean[-1]))
        })
        metric_details.update(result_dic)
        metric_res_path = os.path.join(save_path, 'metric_details_{}.json'.format(self.epoch))
        with open(metric_res_path, "w") as f:
            json.dump(metric_details, f)
        print(f"Finish generate predictions for testset, save to {metric_res_path}")

        for key, value in result_dic.items():
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
    version_flag = 'v1_c3'
    save_path = '/rbs/jiangjunyuan/code/BEVFusion/dataset/{}/{}_metric'.format(data_flag, version_flag)
    if not os.path.exists(save_path):
        os.system("mkdir -p {}".format(save_path))
    gt_res_path = '/pcpt/pcpt/project/lijun/data/{}/{}_{}gt_filter.json'.format(data_flag, version_flag,date_str)
    pt_res_path = '/pcpt/pcpt/project/lijun/data/{}/{}_{}pt.json'.format(data_flag, version_flag,date_str)
    
    det_metric = Det3DMetric(iou_type='bev', 
                             Classes=("person", "bike", "rider", "car", "truck", "bus"),
                             iou_threshs=[0.25, 0.5, 0.75],
                            #  point_cloud_range=[-102.4, -102.4, -3.0, 102.4, 102.4, 5.0],
                             point_cloud_range=[-81.6, -48, -1, 97.6, 48, 3.0],
                             filter_nums=0,
                             metric_key='mAP')
    
    det_metric.pt_res_path = pt_res_path
    det_metric.gt_res_path = gt_res_path
    result_dic = det_metric.get_metric(save_path)

    with open(os.path.join(save_path, 'metric.json'), 'w') as f:
        json.dump(result_dic, f)

    # th = np.load('th.npz')['th']
    # scores = det_metric.get_thresholds(th, 8656)
    # print(scores)
    # print(len(scores))
