import os
from pathlib import Path
from typing import Any, Dict, Tuple

import mmcv
import numpy as np
import numba as nb
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from PIL import Image
import torch
import cv2

from extract_hdmap_featrue.commonUtils import (
    DBManager,
    load_hdmap_around_ego,
    utm_to_wgs84,
    visual_map_featrue,
    points_to_bev_indices,
    draw_polygon_boundary_bev,
)

from skimage.draw import polygon_perimeter

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams
from ...core.bbox import LiDARInstance3DBoxes
    
    
@nb.jit('u1[:,:](u1[:,:],i8[:,:],i8)', nopython=True, cache=True, parallel=False)
def numba_process_label(grid_label, sorted_label_voxel_pair, label_size):
    # label_size = len(self.classes)
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 2]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :2]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :2]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            grid_label[cur_sear_ind[0], cur_sear_ind[1]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 2]] += 1
    grid_label[cur_sear_ind[0], cur_sear_ind[1]] = np.argmax(counter)
    return grid_label

@nb.jit(nopython=True, cache=True, parallel=False)
def numba_process_label_3d(grid_label, sorted_label_voxel_pair, label_size):
    # label_size = len(self.classes)
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            grid_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    grid_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return grid_label

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles:
    """Load multi channel images from a list of separate channel files.

    Expects results['image_paths'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged", is_bag=False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.is_bag = is_bag

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["image_paths"]
        # img is of shape (h, w, c, num_views)
        # modified for waymo
        images = []
        images_size = []
        h, w = 0, 0
        for index, name in enumerate(filename):
            img = Image.open(name)
            if self.is_bag:
                if index == 0:
                    img = img.resize((3840, 2160))
                else:
                    img = img.resize((1920, 1536))     
            # img = Image.open(name)
            images.append(img)
            images_size.append(img.size)
        
        #TODO: consider image padding in waymo

        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = images
        # [1600, 900]
        results["img_shape"] = images_size
        results["ori_shape"] = images_size
        # Set initial values for default meta_keys
        results["pad_shape"] = images_size
        results["scale_factor"] = 1.0
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


@PIPELINES.register_module()
class LoadBEVSegmentation:
    def __init__(
        self,
        dataset_root: str,
        # xbound: Tuple[float, float, float],
        # ybound: Tuple[float, float, float],
        scale: Tuple[float, float, float, float, float, float],
        resolution: Tuple[float, float, float],
        classes: Tuple[str, ...],
        ignore_label: int,
        downsample_rate=8,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.resolution = resolution 
        self.downsample_rate = downsample_rate
        dim_x = round((scale[3] - scale[0]) / resolution[0] / self.downsample_rate)  # int((max-min)/interval)
        dim_y = round((scale[4] - scale[1]) / resolution[1] / self.downsample_rate)  
        dim_z = round((scale[5] - scale[2]) / resolution[2] / self.downsample_rate) 
        self.voxel_size = (dim_x, dim_y, dim_z)
        self.classes = classes
        self.ignore_label = ignore_label


    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        '''
        lidar2point = data["lidar_aug_matrix"]
        point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        lidar2global = ego2global @ lidar2ego @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        location = data["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        ) # np.array
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)
        masks = masks.astype(np.bool)

        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        '''
        # temp set all 0
        # labels = np.zeros((len(self.classes), *self.voxel_size[:2]), dtype=np.long)
        # data["gt_masks_bev"] = labels
        points = data['points'].tensor.numpy()[:,:2].astype(np.float32)
        grid_ind = (np.floor((np.clip(points, self.scale[:2], [x-0.001 for x in self.scale[3:5]]) - self.scale[:2]) 
                             / (np.array(self.resolution[:2]) * self.downsample_rate))).astype(np.int64)
        grid_label = np.ones(self.voxel_size[:2], dtype=np.uint8) * self.ignore_label
        if "points_labels" in data.keys():
            points_labels = data['points_labels'].astype(np.int64)
            label_voxel_pair = np.concatenate([grid_ind, data['points_labels'].reshape(-1,1)], axis=1).astype(np.int64)
            label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1])), :]
            grid_label = numba_process_label(np.copy(grid_label), label_voxel_pair, len(self.classes))
        data['gt_masks_bev'] = grid_label

        data['points_grid_ind'] = grid_ind
        return data
    
@PIPELINES.register_module()
class LoadHdMapFromSQL:
    def __init__(
        self,
        point_cloud_range,
        voxel_size,
        hdmap_path,
        is_visual,
        coord_type,
    ):
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.hdmap_path = hdmap_path
        self.is_visual = is_visual
        self.coord_type = coord_type
        
        self.CH = {
            'lanes': 0,
            'lane_links': 1,
            'lane_boundaries': 2,
            'junctions': 3,
        }
        
    def load_map(self):
        conn = DBManager.initialize_database(self.hdmap_path)
        return conn
    
    def pointUtm_list_to_numpy(self, pointUtm_list):
        # if len(pointUtm_list) == 0:
        #     return np.array([])
        
        # geom_pts = np.array([[p.x, p.y] for p in pointUtm_list], dtype=np.float32)
        # pad_zeros = np.zeros((geom_pts.shape[0],1))
        # geom_pts = np.concatenate((geom_pts, pad_zeros),axis=-1)
        
        # points_class = get_points_type(self.coord_type)
        # geom_pts = points_class(geom_pts, points_dim=geom_pts.shape[-1])
        # return geom_pts
        return np.array([[p.x, p.y] for p in pointUtm_list], dtype=np.float32)
    
    def translate_ego(self, data, pose_matrix):
        pose_inv = np.linalg.inv(pose_matrix)
        out = {}

        for k, polylines in data.items():
            out[k] = []
            for pts in polylines:
                N = pts.shape[0]
                pts_h = np.ones((N, 4), dtype=np.float32)
                pts_h[:, 0] = pts[:, 0]
                pts_h[:, 1] = pts[:, 1]
                pts_h[:, 2] = 0.0

                pts_ego = (pose_inv @ pts_h.T).T
                
                # pad_zeros = np.zeros((pts_ego.shape[0],1))
                # geom_pts = np.concatenate((pts_ego[:, :2], pad_zeros),axis=-1)
                
                # points_class = get_points_type(self.coord_type)
                # geom_pts = points_class(geom_pts, points_dim=geom_pts.shape[-1])

                out[k].append(pts_ego[:, :2])
        return out
    
    def densify_polyline(self, pts, step=0.1):
        
        dense = [pts[0]]
        for i in range(1, len(pts)):
            p0, p1 = pts[i - 1], pts[i]
            dist = np.linalg.norm(p1 - p0)
            if dist < step:
                dense.append(p1)
                continue
            num = max(1, int(dist / step))
            for j in range(1, num + 1):
                dense.append(p0 + (p1 - p0) * j / num)
                
        # return np.asarray(dense, dtype=np.float32)
        
        dense = np.asarray(dense, dtype=np.float32)
        pad_zeros = np.zeros((dense.shape[0],1))
        geom_pts = np.concatenate((dense, pad_zeros),axis=-1)
        
        points_class = get_points_type(self.coord_type)
        geom_pts = points_class(geom_pts, points_dim=geom_pts.shape[-1])
        
        return geom_pts
    
    def densify_ego_points(self, ego_points, step=0.1):
        LINE_KEYS = {
            'lanes',
            'lane_links',
            'lane_boundaries',
            'roads',
        }
        POLYGON_KEYS = {
            'junctions',
            'greenbelts',
            'safetyislands',
        }

        out = {}
        for k, geometries in ego_points.items():
            out[k] = []

            if k in LINE_KEYS:
                for pts in geometries:
                    if len(pts) >= 2:
                        out[k].append(self.densify_polyline(pts, step))
            elif k in POLYGON_KEYS:
                for pts in geometries:
                    pts = pts.astype(np.float32)
                    pad_zeros = np.zeros((pts.shape[0],1))
                    geom_pts = np.concatenate((pts, pad_zeros),axis=-1)
                    
                    points_class = get_points_type(self.coord_type)
                    geom_pts = points_class(geom_pts, points_dim=geom_pts.shape[-1])
                    if len(pts) >= 3:
                        out[k].append(geom_pts)
            else:
                print(geometries, 'no exsists!!!!!!')

        return out
    
    def hdmap_data_to_numpy_raw(self, data):
        hdmap_np = {
            'lanes': [],
            'lane_links': [],
            'lane_boundaries': [],
            'junctions': [],
        }
        if 'lanes' in data:
            for lane in data['lanes']:
                pts = self.pointUtm_list_to_numpy(lane.pointUtm)
                if len(pts) >= 2:
                    hdmap_np['lanes'].append(pts)

        if 'lane_links' in data:
            for link in data['lane_links']:
                pts = self.pointUtm_list_to_numpy(link.pointUtm)
                if len(pts) >= 2:
                    hdmap_np['lane_links'].append(pts)

        if 'lane_boundaries' in data:
            for bd in data['lane_boundaries']:
                pts = self.pointUtm_list_to_numpy(bd.pointUtm)
                if len(pts) >= 2:
                    hdmap_np['lane_boundaries'].append(pts)

        if 'junctions' in data:
            for junc in data['junctions']:
                pts = self.pointUtm_list_to_numpy(junc.pointUtm)
                if len(pts) >= 3:
                    hdmap_np['junctions'].append(pts)

        # for k, v in hdmap_np.items():
        #     print(f'{k}: {len(v)}')
            
        return hdmap_np
    
    def get_hd_geom(self, hd_map, results):
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        vx, vy, _ = self.voxel_size
        
        npz_info = np.load(results['lidar_path'],allow_pickle=True)
        
        pose = npz_info['transform_mat']
        lon, lat = utm_to_wgs84(position=pose[:, 3][:3])

        # -------- load hdmap --------
        width  = x_max - x_min
        height = y_max - y_min
        radius_m = int(np.sqrt(width ** 2 + height ** 2) / 2) 
        ego_range_post = [-radius_m, -radius_m, radius_m, radius_m]

        hdmap_geom = load_hdmap_around_ego(
            hd_map, lon, lat, ego_range_post
        )
        hdmap_np = self.hdmap_data_to_numpy_raw(hdmap_geom)

        ego_points = self.translate_ego(hdmap_np, pose)
        ego_points = self.densify_ego_points(
            ego_points, step = self.voxel_size[0] # 0.1
        )
        
        extrinsic = npz_info['sensing_120_front'].item()['extrinsic']
        intrinsic = npz_info['sensing_120_front'].item()['intrinsic']
        cam_token = results['image_paths'][1]
        
        if self.is_visual:
            H = int(round((x_max - x_min) / vx))  # 1792
            W = int(round((y_max - y_min) / vy))  # 960
            
            hdmap_bev = torch.zeros((4, H, W),dtype=torch.float32,device='cpu')

            for k, polylines in ego_points.items():
                ch = self.CH[k]

                if k in ['lanes', 'lane_links', 'lane_boundaries']:
                    for pts in polylines:
                        pts = np.array(pts.tensor.copy())
                        rows, cols = points_to_bev_indices(
                            pts, self.point_cloud_range, self.voxel_size
                        )
                        if rows is None:
                            continue
                        # hdmap_bev[ch, cols, rows] = 1.0
                        try:
                            hdmap_bev[ch, rows, cols] = 1.0
                        except:
                            breakpoint()

                elif k == 'junctions':
                    for poly in polylines:
                        if len(poly) >= 3:
                            draw_polygon_boundary_bev(
                                poly,
                                hdmap_bev,
                                ch,
                                self.point_cloud_range,
                                self.voxel_size
                        )
                            
            visual_map_featrue(hdmap_bev, extrinsic, cam_token, self.point_cloud_range, self.voxel_size, intrinsic)
        return ego_points, extrinsic, intrinsic, cam_token
        
    def __call__(self, results):
        hd_map= self.load_map()
        
        hdmap_geom, extrinsic, intrinsic, cam_token = self.get_hd_geom(hd_map, results)
        
        results['hdmap_geom'] = hdmap_geom
        results['extrinsic_zn'] = extrinsic
        results['intrinsic_zn'] = intrinsic
        results['cam_token_zn'] = cam_token
        
        # breakpoint()
        return results

@PIPELINES.register_module()
class LoadPointsFromFile:
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
        label_mapping=None,
        its_mean=[0.0],
        its_std=[256.0],
        # with_lidarid=False
        lidar_idx = None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams
        self.label_mapping = label_mapping
        self.its_mean=its_mean
        self.its_std=its_std
        # self.with_lidarid=with_lidarid
        self.lidar_idx = lidar_idx

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        points_labels = []
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        elif lidar_path.endswith(".npz"):
            if 'B22025data_v1' in lidar_path or 'B22025data_v2' in lidar_path:
                lidar_path = lidar_path.replace('3d_npz', '3d_npz_new')
                if Path(lidar_path).exists() == False:
                    lidar_path = lidar_path.replace('3d_npz_new', '3d_npz')
            npz =  np.load(lidar_path)
            pcds_lidarid = npz['pcds_lidarid']
            try:
                points = npz['pcds']
                # points_labels = npz['pcds_label']
            except:
                points = npz['pcd']
                # points_labels = npz['pcd_label']

            # if isinstance(points_labels, list):
            #     points_labels=np.array(points_labels).astype(np.int32)
            
            # D6 dataset special handling for front and rear lidars
            # if npz.keys().__contains__('pcds_lidarid'): # disable for now zhuning
            #     breakpoint()
            #     lidarid = npz['pcds_lidarid'].astype(np.uint8)
            #     if points.shape[0] == lidarid.shape[0]:
            #         if lidarid.ndim == 2:
            #             points_valid_idx = np.isin(lidarid[:,0], np.array([2])) & np.isin(lidarid[:,1], np.array([1,2,3,4])) # 4lidar RobinW
            #             points = points[points_valid_idx]
            #             # points_labels = points_labels[points_valid_idx]
                        
            #         else:
            #             points_valid_idx = np.isin(lidarid, np.array([4, 5, 6, 7])) # 4lidar 
            #             points = points[points_valid_idx]
            #             # points_labels = points_labels[points_valid_idx]

            if 'at128_seg' in lidar_path:
                pointid=npz['pcds_lidarid']
                mask = np.ones(pointid.shape[0], dtype=bool)
                mask = np.logical_and(mask, pointid < 4)
                points=points[mask]
                points_labels=points_labels[mask]
                # print("at128_seg into  {}   {}".format(pointid.shape[0],points.shape[0]))
            # assert(points.shape[0] == points_labels.shape[0]), "data labels wrong!"
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
            
        ## filter at128 points
        rb_mask = (pcds_lidarid[:,0] == 2)
        points = points[rb_mask]
        pcds_lidarid = pcds_lidarid[rb_mask]
        
        # if self.with_lidarid:
        if self.lidar_idx != None:
            lidarid = pcds_lidarid
            assert (lidarid[:,0]==2).sum() == lidarid.shape[0]
            robinw_mask = (lidarid[:,1] == self.lidar_idx[0]) | (lidarid[:,1] == self.lidar_idx[1])
            points = points[robinw_mask]

            # for idx in range(len(self.its_mean)):
            #     mean_idx = self.its_mean[idx]
            #     std_idx = self.its_std[idx]
            #     mask = (lidarid==idx)
            #     intensity[mask] = (intensity[mask]-mean_idx)/std_idx
            #     points[:, 3] = intensity
                
        points[:, 3] = (points[:, 3]-self.its_mean[0])/self.its_std[0]
        
        # else:
            # points[:, 3] = (points[:, 3]-self.its_mean[0])/self.its_std[0]
            # if points.shape[1] == 5:
            #     points_id = points[:,4].astype(np.uint8)
            #     points_valid = np.isin(points_id, np.array([4,6]))
            #     points = points[points_valid]
            #     points_labels = points_labels[points_valid]
            #     points = points[:, :4]
        return points, points_labels
    def relabel(self, pcds_labels, label_map):
        result_labels = np.zeros((pcds_labels.shape[0],), dtype=pcds_labels.dtype)
        for key in label_map:
            value = label_map[key]
            mask = (pcds_labels == key)
            result_labels[mask] = value
        
        return result_labels
    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        points, points_labels = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        # TODO: make it more general
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        # relabel according label mapping
        if points_labels != []:
            # points_labels = np.vectorize(results['label_mapping']['learning_map'].__getitem__)(points_labels)
            points_labels = self.relabel(points_labels, results['label_mapping']['learning_map'])
            results["points_labels"] = points_labels

        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
    """

    def __init__(
        self,
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_bbox=False,
        with_label=False,
        with_mask=False,
        with_seg=False,
        with_bbox_depth=False,
        poly2mask=True,
    ):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
        )
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results["gt_bboxes_3d"] = results["ann_info"]["gt_bboxes_3d"]
        results["bbox3d_fields"].append("gt_bboxes_3d")
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results["centers2d"] = results["ann_info"]["centers2d"]
        results["depths"] = results["ann_info"]["depths"]
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["gt_labels_3d"] = results["ann_info"]["gt_labels_3d"]
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["attr_labels"] = results["ann_info"]["attr_labels"]
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)

        return results
    
@PIPELINES.register_module()
class OccGTRegroup(object):
    def __init__(
        self,
        gt_range=[-50., -50., -1., 50., 50., 5.4],
        gt_voxel_size=0.2,
    ) -> None:
        self.pc_range = gt_range
        self.voxel_size = gt_voxel_size
        
    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_labels = np.load(occ_gt_path)

        coords = occ_labels[:, :3].astype(np.float32)
        labels = occ_labels[:, 3].astype(np.int64)

        centers = (
            coords * self.voxel_size
            + self.pc_range[:3]
            + self.voxel_size * 0.5
        )
        
        num = centers.shape[0]
        sizes = np.full((num, 3), self.voxel_size, dtype=np.float32)
        yaw = np.zeros((num, 1), dtype=np.float32)

        # (N, 7): [x, y, z, w, l, h, yaw]
        boxes = np.concatenate([centers, sizes, yaw], axis=1)

        results["voxel_in_LiDARInstance3DBoxes"] = LiDARInstance3DBoxes(boxes)
        results["voxel_in_semantics"] = labels

        return results
    
@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    def __init__(
        self,
        ignore_label=0,
        desired_range=[-50, -50, -1, 50, 50, 5.4],
        desired_voxel_size=[0.1, 0.1, 0.1],
        downsample_rate=1,
        classes=None,
    ) -> None:
        self.ignore_label = ignore_label
        self.desired_range = np.array(desired_range, dtype=np.float32)
        self.desired_voxel_size = np.array(desired_voxel_size, dtype=np.float32)
        self.downsample_rate = downsample_rate
        self.classes = classes

        dim_x = round((desired_range[3] - desired_range[0]) / desired_voxel_size[0] / downsample_rate)
        dim_y = round((desired_range[4] - desired_range[1]) / desired_voxel_size[1] / downsample_rate)
        dim_z = round((desired_range[5] - desired_range[2]) / desired_voxel_size[2] / downsample_rate)
        self.desired_occ_size = (dim_x, dim_y, dim_z)

    def __call__(self, results):
        boxes = results["voxel_in_LiDARInstance3DBoxes"].tensor.numpy()
        labels = results["voxel_in_semantics"].astype(np.int64)

        centers = boxes[:, :3]

        voxel_coords = (
            (centers - self.desired_range[:3])
            / self.desired_voxel_size
            / self.downsample_rate
        ).astype(np.int32)

        valid_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < self.desired_occ_size[0]) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < self.desired_occ_size[1]) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < self.desired_occ_size[2])
        )

        voxel_coords = voxel_coords[valid_mask]
        labels = labels[valid_mask]

        if voxel_coords.shape[0] > 0:
            coords = np.concatenate(
                [voxel_coords, labels[:, None]],
                axis=1
            )
        else:
            coords = np.empty((0, 4), dtype=np.int32)

        voxel_label = np.full(
            self.desired_occ_size,
            self.ignore_label,
            dtype=np.int64
        )

        if coords.shape[0] > 0:
            # sort by (z, y, x) for numba vote
            sorted_coords = coords[np.lexsort(
                (coords[:, 0], coords[:, 1], coords[:, 2])
            )]
            voxel_label = numba_process_label_3d(
                voxel_label,
                sorted_coords,
                len(self.classes)
            )

        mask_lidar = np.ones(self.desired_occ_size, dtype=np.int64)
        mask_camera = np.ones(self.desired_occ_size, dtype=np.int64)

        results['voxel_semantics'] = voxel_label
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera
        return results