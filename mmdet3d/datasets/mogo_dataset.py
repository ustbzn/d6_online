import mmcv
import numpy as np
import os
import copy
import tempfile
import torch
from mmcv.utils import print_log
from os import path as osp
from typing import Any, Dict

from mmdet.datasets import DATASETS
from ..core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset

import pyquaternion
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox


@DATASETS.register_module()
class MogoDataset(Custom3DDataset):
    """Mogo Dataset.

    Args:
        dataset_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        object_classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
    """

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        gt_names = info["gt_names"].copy()
        for idx, name in enumerate(gt_names):
            if name in self.bike_list:
                gt_names[idx] = 'bike'

        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(gt_names[mask])
        else:
            gt_names = set(gt_names)

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        return data_infos

    def __init__(self,
                 dataset_root,
                 ann_file,#pkl文件路径
                 pipeline=None,
                 object_classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 post_center_range=None,
                 eval_version="detection_cvpr_2019",
                 use_valid_flag=False,
                 load_interval=1):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        self.version = "v1.0-trainval"
        self.bike_list = ['pushing', 'bike', 'rider']
        self.post_center_range = post_center_range

        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]

        data = dict(
            timestamp=info['timestamp'],
            lidar_path=info["lidar_path"],
            #sweeps=info["sweeps"],
        )

        # # ego to global transform
        # ego2global = np.eye(4).astype(np.float32)
        # ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        # ego2global[:3, 3] = info["ego2global_translation"]
        # data["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            # for _, camera_info in info["cams"].items():#'120_front','30_front','60_front',pkl中camera的顺序是根据calibrated_sensor.json中的顺序决定的
            for camera_name in ['120_front', '60_front','30_front']:#强制按照'120_front','60_front','30_front'的顺序
                find_camera = False
                for _, camera_info in info["cams"].items():
                    if camera_name == camera_info["type"]:
                        find_camera = True
                        break
                if not find_camera:
                    continue
                # print("camera_info in infoitems()")
                data["image_paths"].append(camera_info["data_path"])

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)

        annos = self.get_ann_info(index)
        data["ann_info"] = annos
        return data

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:#true
            mask = info["valid_flag"]#填充的全是true
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        
        for idx, cat in enumerate(gt_names_3d):
            if cat in self.bike_list:
                cat = 'bike'
                gt_names_3d[idx] = cat
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=7, origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
            """Convert the results to the standard format.

            Args:
                results (list[dict]): Testing results of the dataset.
                jsonfile_prefix (str): The prefix of the output jsonfile.
                    You can specify the output directory/filename by
                    modifying the jsonfile_prefix. Default: None.

            Returns:
                str: Path of the output json file.
            """
            dt_annos = []
            mapped_class_names = self.CLASSES

            print("Start to convert detection format...")
            for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
                annos = {}
                bboxes = det["boxes_3d"].tensor.numpy()
                scores = det["scores_3d"].numpy()
                labels = det["labels_3d"].numpy()

                indices = scores >= 0.1
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

                annos['box_center'] = bboxes[:, :3]
                annos['box_center'][:, 2] += bboxes[:, 5] * 0.5
                annos['box_size'] = bboxes[:, 3:6]
                annos['box_rotation'] = np.zeros((bboxes.shape[0], 3))
                annos['box_rotation'][:,2] = bboxes[:, 6]
                annos['scores'] = scores

                names = []
                for i, box in enumerate(bboxes):
                    name = mapped_class_names[labels[i]]#根据label的序号查找对应的类型
                    names.append(name)
                annos['name'] = np.array(names)
                dt_annos.append(annos)

            mmcv.mkdir_or_exist(jsonfile_prefix)
            res_path = osp.join(jsonfile_prefix, "results_mogo.json")
            print("Results writes to", res_path)
            mmcv.dump(dt_annos, res_path)#保存成json文件
            return dt_annos
    
    def format_results(self, results, jsonfile_prefix=None):
            """Format the results to json (standard format for COCO evaluation).

            Args:
                results (list[dict]): Testing results of the dataset.
                jsonfile_prefix (str | None): The prefix of json files. It includes
                    the file path and the prefix of filename, e.g., "a/b/prefix".
                    If not specified, a temp file will be created. Default: None.

            Returns:
                tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                    dict containing the json filepaths, `tmp_dir` is the temporal \
                    directory created for saving json files when \
                    `jsonfile_prefix` is not specified.
            """
            assert isinstance(results, list), "results must be a list"
            assert len(results) == len(
                self
            ), "The length of results is not equal to the dataset len: {} != {}".format(
                len(results), len(self)
            )

            if jsonfile_prefix is None:
                tmp_dir = tempfile.TemporaryDirectory()
                jsonfile_prefix = osp.join(tmp_dir.name, "results")
            else:
                tmp_dir = None

            result_files = self._format_bbox(results, jsonfile_prefix)
            return result_files, tmp_dir

    def format_gtinfos(self):
        gt_annos = copy.deepcopy(self.data_infos)
        for anno in gt_annos:
            mask = (anno['gt_boxes'][..., :3] >= self.post_center_range[:3]).all(1)
            mask &= (anno['gt_boxes'][..., :3] <= self.post_center_range[3:]).all(1)

            anno['name'] = anno['gt_names'][mask]
            for idx, cat in enumerate(anno['name']):
                if cat in self.bike_list:
                    cat = 'bike'
                    anno['name'][idx] = cat

            anno['box_center'] = anno['gt_boxes'][mask][:, :3]
            anno['box_size'] = anno['gt_boxes'][mask][:, 3:6]
            anno['box_rotation'] = np.zeros((anno['gt_boxes'][mask].shape[0], 3))
            anno['box_rotation'][:, 2] = anno['gt_boxes'][mask][:, 6]

        return gt_annos
    
    def evaluate(#eva指标评估函数
        self,
        results,
        metric="bbox",
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        **kwargs,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        # import ipdb; ipdb.set_trace()
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)#将output转换格式输出到results_mogo.json文件中
        from .eval import get_official_eval_result
        gt_annos = self.format_gtinfos()
        # import ipdb; ipdb.set_trace()
        ap_result_str, ap_dict = get_official_eval_result(gt_annos, result_files, self.CLASSES)


        return ap_result_str, ap_dict 

