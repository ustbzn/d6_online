import json
import os
from collections import OrderedDict
from Det3DMetric import Det3DMetric

class DetMetricEvaluator:
    def __init__(self, pt_res_path, gt_res_path, save_path, metric_config):
        self.pt_res_path = pt_res_path
        self.gt_res_path = gt_res_path
        self.save_path = save_path
        self.metric_config = metric_config

    def evaluate(self):
        # 这里直接调用 Det3DMetric 类（你的原始评测类）
        det_metric = Det3DMetric(
            iou_type=self.metric_config.get('iou_type', 'bev'),
            Classes=self.metric_config['Classes'],
            iou_threshs=self.metric_config['iou_threshs'],
            point_cloud_range=self.metric_config['point_cloud_range'],
            filter_nums=self.metric_config.get('filter_nums', 0),
            metric_key=self.metric_config.get('metric_key', 'mAP'),
            metric_type=self.metric_config.get('metric_type', 'macro'),
            range_version=self.metric_config.get('range_version', 'v1'),
            metric_classes=self.metric_config.get('metric_classes', self.metric_config['Classes']),
        )
        det_metric.pt_res_path = self.pt_res_path
        det_metric.gt_res_path = self.gt_res_path
        result_dic = det_metric.get_metric_new(self.save_path)
        metric_file = os.path.join(self.save_path, f'metric_{self.metric_config.get("range_version", "v1")}.json')
        with open(metric_file, 'w') as f:
            json.dump(result_dic, f, indent=2)
        print(f"Metric results saved to {metric_file}")