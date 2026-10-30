import mmcv
import numpy as np
import os
import copy
import tempfile
import torch
import json
import yaml
import pickle
import glob
from mmcv.utils import print_log
from os import path as osp
from typing import Any, Dict
from pyquaternion import Quaternion
from tabulate import tabulate

from mmdet.datasets import DATASETS
from ..core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
import random
import glob
from PIL import Image
import math

def update_vehicle_calib(vehicle_config_path, cam_list):
    def inverse_rigid_trans(Tr):
        """ Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        """
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    with open(vehicle_config_path, 'r') as f:
        calib_objs = json.load(f)

    for calib_type, calib_dic in calib_objs.items():
        for sensor_type, sensor_dic in calib_dic.items():
            if sensor_type not in cam_list:
                continue
            H = sensor_dic['height']
            W = sensor_dic['width']
            intrinsic = np.array(sensor_dic['intrinsic'], dtype=np.float32).reshape(3, 3)
            distort = np.array(sensor_dic['distort'], dtype=np.float32).reshape(-1)

            # get extrinsic matrix
            mat_cam2imu = np.array(sensor_dic['extrinsic'], dtype=np.float32).reshape(3, 4)
            # mat_imu2cam = inverse_rigid_trans(mat_cam2imu).astype(np.float32).reshape(3, 4) #(3,4)
            sensor_dic.update({
                'intrinsic': intrinsic,
                'distort': distort,
                'sensor2lidar_rotation': mat_cam2imu[:3, :3],
                'sensor2lidar_translation': mat_cam2imu[:, -1],
            })

    """ 
    calib_objs['DFHYD00908']['30_front']:{
        'height':2160
        'width':3840
        'intrinsic':3*3array
        'distort':array,len=5
        'sensor2lidar_rotation':3*3array,旋转矩阵
        'sensor2lidar_translation':3array,平移
    }
    """ 
    return calib_objs
def only_update_vehicle_calib(vehicle_config_path, cam_list):
    def inverse_rigid_trans(Tr):
        """ Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        """
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    with open(vehicle_config_path, 'r') as f:
        calib_objs = json.load(f)

    # for calib_type, calib_dic in calib_objs.items():
    for sensor_type, sensor_dic in calib_objs.items():
        if sensor_type not in cam_list:
            continue
        H = sensor_dic['height']
        W = sensor_dic['width']
        intrinsic = np.array(sensor_dic['intrinsic'], dtype=np.float32).reshape(3, 3)
        distort = np.array(sensor_dic['distort'], dtype=np.float32).reshape(-1)

        # get extrinsic matrix
        mat_cam2imu = np.array(sensor_dic['extrinsic'], dtype=np.float32).reshape(3, 4)
        # mat_imu2cam = inverse_rigid_trans(mat_cam2imu).astype(np.float32).reshape(3, 4) #(3,4)
        sensor_dic.update({
            'intrinsic': intrinsic,
            'distort': distort,
            'sensor2lidar_rotation': mat_cam2imu[:3, :3],
            'sensor2lidar_translation': mat_cam2imu[:, -1],
        })

    """ 
    calib_objs['DFHYD00908']['30_front']:{
        'height':2160
        'width':3840
        'intrinsic':3*3array
        'distort':array,len=5
        'sensor2lidar_rotation':3*3array,旋转矩阵
        'sensor2lidar_translation':3array,平移
    }
    """ 
    return calib_objs
@DATASETS.register_module()
class MogoAT128Dataset(Custom3DDataset):
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
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
    """

    def get_cat_ids(self, idx):#CBGS时候需要用
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        gt_names = info['gt_names']
        
        if 'num_lidar_pts' in info:
            mask = info['num_lidar_pts'] > self.min_pts_num
        else:
            mask = np.ones_like(gt_names).astype(np.bool_)

        gt_names_3d = gt_names[mask]
        cat_ids = []
        for name in gt_names_3d:
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
        # data = mmcv.load(ann_file)
        with open(ann_file, 'rb') as f:  
            data = pickle.load(f)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        return data_infos

    def __init__(self,
                 dataset_root,
                 ann_file,#pkl文件路径
                 pipeline=None,
                 object_classes=None,
                 occ_classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 post_center_range=None,
                 load_interval=1,
                 calib_path=None,
                 cam_list=None,
                 cam_random_num=-1,
                 label_map_pth=None,
                 min_pts_num=0,
                 ignore_label=0,
                 evaluate_by_points=False,
                 point_cloud_range=None,
                 save_best=None,
                 metric_split = None,
                 metric_name = None,):
        self.dataset_root = dataset_root
        self.load_interval = load_interval
        self.bike_list = ['pushing', 'bike', 'rider']
        self.post_center_range = post_center_range
        self.cam_list = cam_list
        # self.calib_path = calib_path
        # self.calib_dict = update_vehicle_calib(self.calib_path, self.cam_list)
        self.min_pts_num = min_pts_num
        with open(label_map_pth, 'r') as f:
            self.label_mapping = yaml.load(f, Loader=yaml.Loader)
        self.ignore_label = ignore_label
        self.evaluate_by_points = evaluate_by_points
        self.cam_random_num = cam_random_num
        self.things =  ["person", "bike", "rider", "car", "truck", "bus"]
        self.save_best=save_best
        self.occ_classes = occ_classes
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            point_cloud_range=point_cloud_range,
            metric_split = metric_split,
            metric_name = metric_name,
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

        data_path = info['data_path']
        if not os.path.isabs(data_path):
            # if 'B22024' in data_path:
                # 当data_path包含B22024时，拼接self.dataset_root/B22024/data_path
                # data_path = os.path.join(self.dataset_root, 'B22024', data_path)
            data_path = os.path.join(self.dataset_root, data_path)
            # else:
            #     # 普通相对路径直接拼接self.dataset_root
            #     data_path = os.path.join(self.dataset_root, data_path)

        data = dict(
            timestamp=info['timestamp'],
            lidar_path=data_path,
        )
        if not os.path.exists(data_path):
            print("lidar path not exist: ", data_path)
            return None
        else:
            try:
                np.load(data_path)
            except Exception as e:
                print(f"Error loading npz file {data_path}: {e}")
                return None
        
        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion([1.0, 0.0, 0.0, 0.0]).rotation_matrix#需要check下
        lidar2ego[:3, 3] = [0.0, 0.0, 0.0]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            if 'calib_json_name' in info:
                # calib_json_name = os.path.join(self.dataset_root, 'B22024', info['calib_json_name'])
                calib_json_name = os.path.join(self.dataset_root, info['calib_json_name'])
                calib_json_name_front = os.path.dirname(calib_json_name)
                if not os.path.exists(calib_json_name_front):
                    print("currrent path not exist! calib_json_name_front: ", calib_json_name_front)
                    return None
                calib_json_name_last = glob.glob(os.path.join(calib_json_name_front, "*.json"))
                for calib in calib_json_name_last:
                    if'radar' in calib:
                        continue
                    calib_file=calib
                # 当前帧数据缺少标定文件,return
                try:
                    calib_dic=only_update_vehicle_calib(calib_file, self.cam_list)
                    # calib_dic=only_update_vehicle_calib(calib_json_name_last[0], self.cam_list)
                except Exception as e:
                    print(f"Error loading calibration file {calib_json_name_last}: {e}")
                    print("calib_json_name: ", calib_json_name, " calib_json_name_last: ", calib_json_name_last)
                    return None
            elif 'calib_data' in info:
                # 新模式：直接读取数据
                calib_dic = info['calib_data']

            # with open(calib_json_name, 'r') as f:
            #     calib_dic = json.load(f)
            if self.cam_random_num > 0:
                all_cam_list = random.sample(self.cam_list, self.cam_random_num)
            else:
                all_cam_list = self.cam_list

            cam_list = list(set(list(calib_dic.keys())) & set(all_cam_list))
            if len(cam_list) != len(all_cam_list):
                print( "cam_random_num {} cam_list {} != all_cam_list {}  calib_dic.key {}".format(self.cam_random_num,cam_list,all_cam_list,set(list(calib_dic.keys()))))
                print(calib_file)
                return None

            # 6个相机只要有一个没有数据,return
            for cam_type in all_cam_list:
                if 'calib_json_name' in info:
                    cam_token = info['data_path'].replace("3d_npz", cam_type).replace("npz", "jpg")
                    # cam_path = os.path.join(self.dataset_root, 'B22024', cam_token)
                    cam_path = os.path.join(self.dataset_root, cam_token)
                elif 'calib_data' in info:
                    cam_path = calib_dic[cam_type]['cam_token']
                if not os.path.exists(cam_path):
                    print("cam path not exist: ", cam_path)
                    return None
                else:
                    try:
                        Image.open(cam_path)
                    except:
                        print('jpg file invalid : ', cam_path)
                        return None

            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            # 填充6个相机的路径以及坐标转换,按照self.cam_list的顺序
            for cam_type in all_cam_list:
                if 'calib_json_name' in info:
                    cam_token = info['data_path'].replace("3d_npz", cam_type).replace("npz", "jpg")
                    # cam_path = os.path.join(self.dataset_root, 'B22024', cam_token)
                    cam_path = os.path.join(self.dataset_root, cam_token)
                elif 'calib_data' in info:
                    cam_path = calib_dic[cam_type]['cam_token']
                data["image_paths"].append(cam_path)#绝对路径

                sensor_dic = calib_dic[cam_type]

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(sensor_dic['sensor2lidar_rotation'])
                lidar2camera_t = (
                    sensor_dic['sensor2lidar_translation'] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = sensor_dic['intrinsic']
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = sensor_dic['sensor2lidar_rotation']
                camera2ego[:3, 3] = sensor_dic['sensor2lidar_translation']
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = sensor_dic['sensor2lidar_rotation']
                camera2lidar[:3, 3] = sensor_dic['sensor2lidar_translation']
                data["camera2lidar"].append(camera2lidar)

        annos = self.get_ann_info(index)
        data["ann_info"] = annos
        data['label_mapping'] = self.label_mapping
        if 'occ_path' in info:
            data['occ_gt_path'] = info['occ_path']
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
        gt_names = info['gt_names']
        if 'num_lidar_pts' in info:
            mask = info['num_lidar_pts'] > self.min_pts_num
        else:
            mask = np.ones_like(gt_names).astype(np.bool_)

        # import ipdb; ipdb.set_trace()
        if(mask.shape != gt_names.shape):
            print("index: ", index, "mask.shape: ", mask.shape, "gt_names: ", gt_names.shape, gt_names)
            print("info: ", info)
            print("self.min_pts_num: ", self.min_pts_num)
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = gt_names[mask]
        
        for index, cat in enumerate(gt_names_3d):
            if cat in  ["conebucket", "bullbarrel"]:
                gt_bboxes_3d[index][-1] = 0.0
                
            if cat in ["waterhorse","construction","warning"]:
                angle = gt_bboxes_3d[index][-1]
                angle = (angle + np.pi) % (2 * np.pi) - np.pi
                gt_bboxes_3d[index][-1] = angle if angle > 0 else np.pi +  angle
                
        # heading要取反
        try:
            gt_bboxes_3d[:, -1] *= -1
        except:
            print(gt_bboxes_3d.shape)
            pass
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)

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
            mask &= anno['num_lidar_pts'] > self.min_pts_num
            
            mask &= np.array([name in self.CLASSES for name in anno['gt_names']]) #zhuning

            anno['name'] = anno['gt_names'][mask]

            anno['box_center'] = anno['gt_boxes'][mask][:, :3]
            anno['box_size'] = anno['gt_boxes'][mask][:, 3:6]
            anno['box_rotation'] = np.zeros((anno['gt_boxes'][mask].shape[0], 3))
            anno['box_rotation'][:, 2] = anno['gt_boxes'][mask][:, 6] * -1

        return gt_annos

    def evaluate_map(self, results):
        num_classes = len(self.label_mapping['label_map_inv'])

        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)

        for result in results:
            pred = result["seg_logits_bev"]
            label = result["gt_masks_bev"]

            pred = pred.view(-1)
            label = label.view(-1)

            # mask ignore
            nonzero_idx = label != self.ignore_label
            label = label[nonzero_idx]
            pred = pred[nonzero_idx]

            for class_idx in range(num_classes):
                pred_class = (pred == class_idx)
                label_class = (label == class_idx)

                tp[class_idx] += (pred_class & label_class).sum().item()
                fp[class_idx] += (pred_class & ~label_class).sum().item()
                fn[class_idx] += (~pred_class & label_class).sum().item()

        ious = tp / (tp + fp + fn + 1e-7)
        precisions = tp / (tp + fp + 1e-7)
        recalls = tp / (tp + fn + 1e-7)

        metrics = {}
        valid_ious = []
        valid_precisions = []
        valid_recalls = []

        for index, label_index in enumerate(self.label_mapping['label_map_inv']):
            name = self.label_mapping['label_map_inv'][label_index]
            metrics[f"{name}/iou"] = ious[index].item()
            metrics[f"{name}/precision"] = precisions[index].item()
            metrics[f"{name}/recall"] = recalls[index].item()
            if label_index != self.ignore_label:
                valid_ious.append(ious[index].item())
                valid_precisions.append(precisions[index].item())
                valid_recalls.append(recalls[index].item())
        
        metrics["mean/iou"] = sum(valid_ious) / len(valid_ious)
        metrics["mean/precision"] = sum(valid_precisions) / len(valid_precisions)
        metrics["mean/recall"] = sum(valid_recalls) / len(valid_recalls)
        all_tp = tp.sum().item()
        all_fp = fp.sum().item()
        all_fn = fn.sum().item()
        metrics["all/iou"] = all_tp / (all_tp + all_fn + all_fp)
        metrics["all/precision"] = all_tp / (all_tp + all_fp)
        metrics["all/recall"] = all_tp / (all_tp + all_fn)

        return metrics


    def evaluate_map_points_new(self, results):
        num_classes = len(self.label_mapping['label_map_inv'])

        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)

        for result in results:
            
            pred = torch.tensor(result["pred_points_labels"], dtype=torch.long)
            label = torch.tensor(result["gt_points_labels"], dtype=torch.long)
            # Flatten the predictions and labels
            pred = pred.view(-1)
            label = label.view(-1)

            # mask ignore
            nonzero_idx = label != self.ignore_label
            label = label[nonzero_idx]
            pred = pred[nonzero_idx]

            for class_idx in range(num_classes):
                pred_class = (pred == class_idx)
                label_class = (label == class_idx)

                tp[class_idx] += (pred_class & label_class).sum().item()
                fp[class_idx] += (pred_class & ~label_class).sum().item()
                fn[class_idx] += (~pred_class & label_class).sum().item()

        ious = tp / (tp + fp + fn + 1e-7)
        precisions = tp / (tp + fp + 1e-7)
        recalls = tp / (tp + fn + 1e-7)

        metrics = {}
        valid_ious = []
        valid_precisions = []
        valid_recalls = []
        iou_things_mean, iou_stuff_mean = [], []
        pre_things_mean, pre_stuff_mean = [], []
        rec_things_mean, rec_stuff_mean = [], []
        for index, label_index in enumerate(self.label_mapping['label_map_inv']):
            name = self.label_mapping['label_map_inv'][label_index]
            metrics[f"{name}/iou"] = ious[index].item()
            metrics[f"{name}/precision"] = precisions[index].item()
            metrics[f"{name}/recall"] = recalls[index].item()
            if label_index != self.ignore_label:
                valid_ious.append(ious[index].item())
                valid_precisions.append(precisions[index].item())
                valid_recalls.append(recalls[index].item())
                if name in self.things:
                    iou_things_mean.append(ious[index].item())
                    pre_things_mean.append(precisions[index].item())
                    rec_things_mean.append(recalls[index].item())
                else:
                    iou_stuff_mean.append(ious[index].item())
                    pre_stuff_mean.append(precisions[index].item())
                    rec_stuff_mean.append(recalls[index].item())
        
        metrics["mean/iou"] = sum(valid_ious) / len(valid_ious)
        metrics["mean/precision"] = sum(valid_precisions) / len(valid_precisions)
        metrics["mean/recall"] = sum(valid_recalls) / len(valid_recalls)
        all_tp = tp.sum().item()
        all_fp = fp.sum().item()
        all_fn = fn.sum().item()
        metrics["all/iou"] = all_tp / (all_tp + all_fn + all_fp)
        metrics["all/precision"] = all_tp / (all_tp + all_fp)
        metrics["all/recall"] = all_tp / (all_tp + all_fn)
        metrics["things_mean/iou"]  = sum(iou_things_mean) / len(iou_things_mean)
        metrics["tuff_mean/iou"]  = sum(iou_stuff_mean) / len(iou_stuff_mean)
        metrics["things_mean/precision"] = sum(pre_things_mean) / len(pre_things_mean)
        metrics["tuff_mean/precision"] = sum(pre_stuff_mean) / len(pre_stuff_mean)
        metrics["things_mean/recall"] = sum(rec_things_mean) / len(rec_things_mean)
        metrics["tuff_mean/recall"] = sum(rec_stuff_mean) / len(rec_stuff_mean)

        return metrics
    
    def evaluate_map_points(self, results):
        num_classes = len(self.label_mapping['label_map_inv'])

        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)

        for result in results:
            
            pred = torch.tensor(result["pred_points_labels"], dtype=torch.long)
            label = torch.tensor(result["gt_points_labels"], dtype=torch.long)
            # Flatten the predictions and labels
            pred = pred.view(-1)
            label = label.view(-1)

            # mask ignore
            nonzero_idx = label != self.ignore_label
            label = label[nonzero_idx]
            pred = pred[nonzero_idx]

            for class_idx in range(num_classes):
                pred_class = (pred == class_idx)
                label_class = (label == class_idx)

                tp[class_idx] += (pred_class & label_class).sum().item()
                fp[class_idx] += (pred_class & ~label_class).sum().item()
                fn[class_idx] += (~pred_class & label_class).sum().item()

        ious = tp / (tp + fp + fn + 1e-7)
        precisions = tp / (tp + fp + 1e-7)
        recalls = tp / (tp + fn + 1e-7)

        metrics = {}
        valid_ious = []
        valid_precisions = []
        valid_recalls = []

        for index, label_index in enumerate(self.label_mapping['label_map_inv']):
            name = self.label_mapping['label_map_inv'][label_index]
            metrics[f"{name}/iou"] = ious[index].item()
            metrics[f"{name}/precision"] = precisions[index].item()
            metrics[f"{name}/recall"] = recalls[index].item()
            if label_index != self.ignore_label:
                valid_ious.append(ious[index].item())
                valid_precisions.append(precisions[index].item())
                valid_recalls.append(recalls[index].item())
        
        metrics["mean/iou"] = sum(valid_ious) / len(valid_ious)
        metrics["mean/precision"] = sum(valid_precisions) / len(valid_precisions)
        metrics["mean/recall"] = sum(valid_recalls) / len(valid_recalls)
        all_tp = tp.sum().item()
        all_fp = fp.sum().item()
        all_fn = fn.sum().item()
        metrics["all/iou"] = all_tp / (all_tp + all_fn + all_fp)
        metrics["all/precision"] = all_tp / (all_tp + all_fp)
        metrics["all/recall"] = all_tp / (all_tp + all_fn)

        return metrics
    
    def evaluate(
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
        result_dic = {}
        if self.evaluate_by_points == False and self.save_best=='bbox_mAP':
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
            from .eval import get_official_eval_result
            gt_annos = self.format_gtinfos()
            ap_result_str, ap_dict = get_official_eval_result(gt_annos, result_files, self.CLASSES)
            return {self.save_best: ap_dict['map']['map_bev/easy_R40'] }
        
        if self.evaluate_by_points == False and self.save_best=='occ_mIOU':
            occ_metric = self.compute_occupancy_iou(results)
            return {self.save_best: occ_metric['miou'] }
        
        if "seg_logits_bev" in results[0]:
            if self.evaluate_by_points == True:
                seg_eval_result = self.evaluate_map_points_new(results)
                # seg_eval_result = self.evaluate_map_points(results)
            else:
                # seg_eval_result = self.evaluate_map(results)
                seg_eval_result = self.evaluate_map_points(results)
                return {self.save_best: seg_eval_result["mean/iou"] }

            class_names = [self.label_mapping['label_map_inv'][i] for i in range(len(self.label_mapping['label_map_inv']))]
            ious = [f"{seg_eval_result[f'{name}/iou']:.3f}" for name in class_names]
            precisions = [f"{seg_eval_result[f'{name}/precision']:.3f}" for name in class_names]
            recalls = [f"{seg_eval_result[f'{name}/recall']:.3f}" for name in class_names]

            ious.append(f"{seg_eval_result['mean/iou']:.3f}")
            precisions.append(f"{seg_eval_result['mean/precision']:.3f}")
            recalls.append(f"{seg_eval_result['mean/recall']:.3f}")

            class_names.append("Mean")

            ious.append(f"{seg_eval_result['all/iou']:.3f}")
            precisions.append(f"{seg_eval_result['all/precision']:.3f}")
            recalls.append(f"{seg_eval_result['all/recall']:.3f}")

            class_names.append("All")

            ious.append(f"{seg_eval_result['things_mean/iou']:.3f}")
            precisions.append(f"{seg_eval_result['things_mean/precision']:.3f}")
            recalls.append(f"{seg_eval_result['things_mean/recall']:.3f}")

            class_names.append("things_mean")

            ious.append(f"{seg_eval_result['tuff_mean/iou']:.3f}")
            precisions.append(f"{seg_eval_result['tuff_mean/precision']:.3f}")
            recalls.append(f"{seg_eval_result['tuff_mean/recall']:.3f}")

            class_names.append("tuff_mean")
            table_data = [
                ["Class"] + class_names,
                ["IoU"] + ious,
                ["Precision"] + precisions,
                ["Recall"] + recalls,
            ]

            table_str = tabulate(table_data, headers="firstrow", tablefmt="grid")
            # print()
            # print(table_str)
            result_dic.update({
                'seg': table_str
            })
            
        if "voxel_semantics" in results[0]:
            occ_metric = self.compute_occupancy_iou(results)
            
            occ_class_names = self.occ_classes

            occ_ious = []
            for iou in occ_metric['per_class_iou']:
                if iou is None or (isinstance(iou, float) and math.isnan(iou)):
                    occ_ious.append('nan')
                else:
                    occ_ious.append(f"{iou:.3f}")

            # Mean IoU
            occ_ious.append(f"{occ_metric['miou']:.3f}")
            occ_class_names = list(occ_class_names) + ['Mean']

            table_data = [
                ['Class'] + occ_class_names,
                ['IoU'] + occ_ious
            ]

            table_str = tabulate(
                table_data,
                headers="firstrow",
                tablefmt="grid"
            )

            result_dic.update({
                'occ': table_str
            })
        
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        from .eval import get_official_eval_result
        gt_annos = self.format_gtinfos()
        ap_result_str, ap_dict = get_official_eval_result(gt_annos, result_files, self.CLASSES)

        result_dic.update({
            'ap_result_str': ap_result_str,
            'ap_dict': ap_dict,
        })
        return result_dic
    
    def compute_occupancy_iou(self, results, mask_camera=None):
        """
        计算所有数据的 per-class IoU 和 mIoU
        results: list of dict
            - occ_pred_labels: (B, Dx, Dy, Dz, num_classes) or logits
            - voxel_semantics: (B, Dx, Dy, Dz)
        mask_camera: optional (B, Dx, Dy, Dz)
        """
        device = results[0]['occ_pred_labels'].device

        total_intersections = torch.zeros(len(self.occ_classes), device=device)
        total_unions = torch.zeros(len(self.occ_classes), device=device)

        for result in results:
            pred_labels = result['occ_pred_labels']
            voxel_semantics = result['voxel_semantics'].long()

            # (B, Dx, Dy, Dz)
            # pred_labels = occ_pred.argmax(dim=-1)

            if mask_camera is not None:
                mask = mask_camera.bool()
                pred_labels = pred_labels[mask]
                voxel_semantics = voxel_semantics[mask]
            else:
                pred_labels = pred_labels.reshape(-1)
                voxel_semantics = voxel_semantics.reshape(-1)

            for cls_id in range(len(self.occ_classes)):
                pred_cls = (pred_labels == cls_id)
                gt_cls = (voxel_semantics == cls_id)

                intersection = (pred_cls & gt_cls).sum()
                union = (pred_cls | gt_cls).sum()

                total_intersections[cls_id] += intersection
                total_unions[cls_id] += union

        per_class_iou = []
        for cls_id in range(len(self.occ_classes)):
            if total_unions[cls_id] == 0:
                per_class_iou.append(float('nan'))
            else:
                per_class_iou.append(
                    (total_intersections[cls_id] / total_unions[cls_id]).item()
                )
                
        valid_ious = [iou for iou in per_class_iou if not math.isnan(iou)]
        miou = sum(valid_ious) / len(valid_ious) if len(valid_ious) > 0 else 0.0
        return {
            'per_class_iou': per_class_iou,
            'miou': miou
        }
