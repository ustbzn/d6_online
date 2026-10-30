import os
import pickle
import numpy as np
import json
from datetime import date
import mmcv
import orjson

class DetResultParser:
    def __init__(self, ann_file, save_dir, label_map):
        self.ann_file = ann_file
        self.save_dir = save_dir
        self.label_map = label_map

    def get_json_obj(self, box3d, labels, scores, token):
        annos = []
        for i, box in enumerate(box3d):
            det_anno = {
                "sample_token": token,
                "translation": box[:3].tolist(),
                "size": box[3:6].tolist(),
                "rotation": box[6:].tolist(),
                "velocity": [0.0, 0.0],
                "detection_name": self.label_map.get(labels[i], "unknown"),
                "detection_score": float(scores[i]),
            }
            annos.append(det_anno)
        return annos

    def process_sample(self, sample_id, npz_data, fpath_record):
        pt_annos, gt_annos = {}, {}
        pred_bbox3d = npz_data['boxes_3d'].tensor.numpy()
        pred_bbox3d_class = npz_data['labels_3d'].numpy()
        pred_bbox3d_score = npz_data['scores_3d'].numpy()
        gt_bbox3d = npz_data['gt_bboxes_3d'].tensor.numpy()
        gt_label = npz_data['gt_labels_3d'].numpy()
        try:
            pred_points_labels = npz_data['pred_points_labels'].numpy()
        except KeyError:
            pred_points_labels = np.array([])  # Default to empty array if key not found
        try:
            # import pdb; pdb.set_trace()
            pcds = npz_data['pcds'].numpy()
            fname_record = os.path.join(fpath_record, f"pcd_{sample_id}.npz")
            np.savez_compressed(fname_record, gt=gt_label, pred=pred_points_labels, pcds_xyzi=pcds,
                pred_bbox3d=pred_bbox3d, pred_bbox3d_score=pred_bbox3d_score,
                pred_bbox3d_class=pred_bbox3d_class.astype(np.int64), gt_bbox3d=gt_bbox3d, gt_label=gt_label)
        except Exception as e:
            print(f"Sample {sample_id} pcd save failed: {e}")
        sample_token = str(sample_id)
        pt_box3d = pred_bbox3d[:, :7]
        pt_label = pred_bbox3d_class
        pt_score = pred_bbox3d_score
        pt_anno = self.get_json_obj(pt_box3d, pt_label, pt_score, sample_token)
        pt_annos[sample_token] = pt_anno

        gt_box3d = gt_bbox3d[:, :7]
        gt_score = np.ones_like(gt_label, dtype=np.float32)
        gt_anno = self.get_json_obj(gt_box3d, gt_label, gt_score, sample_token)
        gt_annos[sample_token] = gt_anno
        return pt_annos, gt_annos

    def parse(self):
        today = date.today()
        date_str = today.strftime("%Y-%m-%d")
        fpath_record = os.path.join(self.save_dir, date_str)
        # os.makedirs(fpath_record, exist_ok=True)
        pt_annos, gt_annos = {}, {}
        with open(self.ann_file, 'rb') as f:
            datapkl = pickle.load(f)
        for sample_id, npz_data in enumerate(mmcv.track_iter_progress(datapkl)):
            pt_anno, gt_anno = self.process_sample(sample_id, npz_data, fpath_record)
            pt_annos.update(pt_anno)
            gt_annos.update(gt_anno)
        pt_res_path = os.path.join(self.save_dir, 'pt.json')
        gt_res_path = os.path.join(self.save_dir, 'gt.json')
        with open(pt_res_path, 'w') as f:
            json.dump(pt_annos, f)
        with open(gt_res_path, 'w') as f:
            json.dump(gt_annos, f)
        print(f"Saved {len(gt_annos)} gt and {len(pt_annos)} pt annotations.")
        return pt_res_path, gt_res_path