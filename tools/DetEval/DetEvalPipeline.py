#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import argparse
from DetResultParser import DetResultParser
from DetMetricEvaluator import DetMetricEvaluator

class DetEvalPipeline:
    def __init__(self, ann_file, save_dir, label_map, metric_config):
        self.parser = DetResultParser(ann_file, save_dir, label_map)
        self.metric_config = metric_config
        self.save_dir = save_dir

    def run(self):
        pt_res_path, gt_res_path = self.parser.parse()
        evaluator = DetMetricEvaluator(pt_res_path, gt_res_path, self.save_dir, self.metric_config)
        evaluator.evaluate()

def main():
    parser = argparse.ArgumentParser(description='DetEval Pipeline')
    parser.add_argument('--ann_file', type=str, default='', help='Path to the annotation file')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save results')
    args = parser.parse_args()

    ann_file = args.ann_file
    save_dir = args.save_dir
    label_map = {
        0: "person",
        1: "bike",
        2: "rider",
        3: "car",
        4: "truck",
        5: "bus"
    }
    metric_config = {
        'iou_type': 'bev',
        'Classes': ("person", "bike", "rider", "car", "truck", "bus"),
        'iou_threshs': [0.1, 0.25, 0.5],
        'point_cloud_range': [-81.6, -48, -1, 97.6, 48, 3.0],
        'filter_nums': 0,
        'metric_key': 'mAP',
        'metric_type': 'macro',
        'range_version': 'v1',
        'metric_classes': ("person", "bike", "rider", "car", "truck", "bus"),
    }
    pipeline = DetEvalPipeline(ann_file, save_dir, label_map, metric_config)
    pipeline.run()

if __name__ == '__main__':
    main()