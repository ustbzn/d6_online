from typing import Any, Dict

import os
import mmcv
import numpy as np
import torch
import torchvision
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg
from numpy import random
from PIL import Image

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import (
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
    box_np_ops,
)
from mmdet.datasets.builder import PIPELINES

from ..builder import OBJECTSAMPLERS
from .utils import noise_per_object_v3_, in_hull, in_range, rotate_along_z, csr2corners, corners2csr_batch
import os

from extract_hdmap_featrue.commonUtils import (
    visual_map_featrue,
    points_to_bev_indices,
    draw_polygon_boundary_bev,
)

@PIPELINES.register_module()
class Build_hdmap_bev:
    def __init__(
        self, point_cloud_range, voxel_size, is_visual
    ):
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.is_visual = is_visual
                
        self.CH = {
            'lanes': 0,
            'lane_links': 1,
            'lane_boundaries': 2,
            'junctions': 3,
        }
        
    def __call__(self, data):
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        vx, vy, _ = self.voxel_size
        
        H = int(round((x_max - x_min) / vx))  # 1792
        W = int(round((y_max - y_min) / vy))  # 960
        
        hdmap_bev = torch.zeros((4, H, W),dtype=torch.float32,device='cpu')
        
        aug_matrix = data['lidar_aug_matrix']

        for k, polylines in data['hdmap_geom'].items():
            ch = self.CH[k]

            if k in ['lanes', 'lane_links', 'lane_boundaries']:
                for pts in polylines:
                    pts = np.array(pts.tensor)
                    
                    ##################
                    # pts -= aug_matrix[:3, 3]
                    # R_inv = np.linalg.inv(aug_matrix[:3, :3])   # (3,3)
                    # pts = (R_inv @ pts.T).T
                    ##################
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
                    poly = np.array(poly.tensor)
                    ##################
                    # poly -= aug_matrix[:3, 3]
                    # R_inv = np.linalg.inv(aug_matrix[:3, :3])   # (3,3)
                    # poly = poly @ R_inv.T
                    # poly = (R_inv @ poly.T).T
                    ##################
                    if len(poly) >= 3:
                        draw_polygon_boundary_bev(
                            poly,
                            hdmap_bev,
                            ch,
                            self.point_cloud_range,
                            self.voxel_size
                    )
                        
        if self.is_visual:  
            
            extrinsic = data['extrinsic_zn']
            intrinsic = data['intrinsic_zn']
            cam_token = data['cam_token_zn']
            visual_map_featrue(hdmap_bev, extrinsic, cam_token, self.point_cloud_range, self.voxel_size, intrinsic)
            breakpoint()
        data['hdmap_bev'] = hdmap_bev
        return data

@PIPELINES.register_module()
class Quantize:
    def __init__(
        self, points_cloud_range, grid_size, downsample_rate
    ):
        self.range = points_cloud_range
        self.size = grid_size
        self.downsample_rate = downsample_rate
    
    def __call__(self, data):
        x = data["points"].tensor[:, 0]
        y = data["points"].tensor[:, 1]
        z = data["points"].tensor[:, 2]
        
        size_x = self.size[0]
        size_y = self.size[1]
        size_z = self.size[2]
        
        dx = (self.range[3] - self.range[0]) / size_x
        dy = (self.range[4] - self.range[1]) / size_y
        dz = (self.range[5] - self.range[2]) / size_z
        
        x_quan = ((x - self.range[0]) / dx)
        y_quan = ((y - self.range[1]) / dy)
        z_quan = ((z - self.range[2]) / dz)
        
        # pcds_quan = np.stack((x_quan, y_quan, z_quan), axis=-1)
        data['scale_rate'] = (1/self.downsample_rate, 1/self.downsample_rate)
        data['coords_xyz'] = np.expand_dims(np.stack((x_quan, y_quan, z_quan), axis=-1), -1)
        return data
    
@PIPELINES.register_module()
class ResamplePoints:
    def __init__(self, point_sample_num):
        self.point_sample_num = point_sample_num

    def __call__(self, data):
        pcds = data['points']
        points_num = pcds.shape[0]
        # print(points_num)
        if points_num >= self.point_sample_num:
            choice_idxs = np.random.choice(points_num, self.point_sample_num, replace=False)
        else:
            idx1 = np.arange(0,points_num,1)
            idx2 = np.random.choice(points_num, self.point_sample_num-points_num, replace=True)
            choice_idxs = np.concatenate([idx1, idx2])
        # print(choice_idxs.shape)
        pcds = pcds[choice_idxs]
        if 'pts_semantic_mask' in data:
            data['pts_semantic_mask'] = data['pts_semantic_mask'][choice_idxs]

        if 'points_raw' in data:
            data['points_raw'] = data['points_raw'][choice_idxs]

        point_sample_num = pcds.shape[0]
        data['point_choice_idxs'] = choice_idxs
        data['point_sample_num'] = point_sample_num
        data['points'] = pcds
        # data['points_grid_ind'] = data['points_grid_ind'][choice_idxs]
        # data['points_labels'] = data['points_labels'][choice_idxs]
        return data
    
@PIPELINES.register_module()
class ImageAug3D:
    def __init__(
        self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip, is_train, camera_num, resize_dim=None
    ):
        assert camera_num == len(resize_lim) and camera_num == len(bot_pct_lim)
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train
        self.camera_num = camera_num
        self.resize_dim = resize_dim

    def sample_augmentation(self, results, index):
        W, H = results["ori_shape"][index]
        fH, fW = self.final_dim #256*704
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim[index])#[0.38, 0.55]，随机采样，那每次得到的这个值一样吗？
            resize_dims = (int(W * resize), int(H * resize))#922,518，每次不一定一样
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim[index])) * newH) - fH#如果self.bot_pct_lim是[0,0]表示直接用从最下面往上的fH，即self.bot_pct_lim表示最下面裁掉的比例
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))#左右随机裁剪到fW
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim[index])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim[index])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def sample_augmentation_v2(self, results, index):
        W, H = results["ori_shape"][index]#1920*1080
        rH, rW = self.resize_dim  #432x768
        fH, fW = self.final_dim   #256*704
        if self.is_train:
            resize = rW*1.0/W
            resize_dims = self.resize_dim
            crop_h = 0
            crop_w = int(np.random.uniform(0, max(0, rH - fW)))#左右随机裁剪到fW
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = rW*1.0/W
            resize_dims = self.resize_dim
            crop_h = 0
            crop_w = int(max(0, rW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip, rotate
    ):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["img"]
        # data['origin_rgb_img'] = imgs  # for debug zhuning
        # assert len(imgs) == self.camera_num
        new_imgs = []
        transforms = []
        for i, img in enumerate(imgs):
            if self.resize_dim is not None:
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation_v2(data, i)
            else:
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data, i)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(new_img)
            # transform=torch.inverse(transform)
            transforms.append(transform.numpy())
        data["img"] = new_imgs
        # data['rgb_img'] = new_imgs  # for debug zhuning
        # update the calibration matrices
        data["img_aug_matrix"] = transforms
        # print("img_aug_matrix", data["img_aug_matrix"])
        return data


@PIPELINES.register_module()
class GlobalRotScaleTrans: #参考blog：https://blog.csdn.net/huang_victor/article/details/130558625#t2
    def __init__(self, resize_lim, rot_lim, trans_lim, is_train):
        self.resize_lim = resize_lim#[0.9, 1.1]
        self.rot_lim = rot_lim#rad:[-0.78539816, 0.78539816] degree:-45,45
        self.trans_lim = trans_lim#0.5
        self.is_train = is_train

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transform = np.eye(4).astype(np.float32)#对角阵

        if self.is_train:#如果是训练过程的数据处理的话
            scale = random.uniform(*self.resize_lim)#均匀分布随机采样
            theta = random.uniform(*self.rot_lim)
            translation = np.array([random.normal(0, self.trans_lim) for i in range(3)])#xyz三个方向的随机平移，在0-0.5之间随机采样
            rotation = np.eye(3)

            #data["points"]是BasePoints类型
            if "points" in data:
                data["points"].rotate(-theta)#进行旋转操作，逆时针-theta即顺时针theta
                data["points"].translate(translation)#平移
                data["points"].scale(scale)#缩放
            
            if 'hdmap_geom' in data:
                for k, polylines in data['hdmap_geom'].items():
                    for pts in polylines:
                        pts.rotate(-theta)
                        pts.translate(translation)
                        pts.scale(scale)
 
            # 使用LiDARInstance3DBoxes类的rotate,translate,scale对box进行相应的变换
            # 注意这里用的是theta，顺时针theta
            gt_boxes = data["gt_bboxes_3d"]
            rotation = rotation @ gt_boxes.rotate(theta).numpy()
            gt_boxes.translate(translation)
            gt_boxes.scale(scale)
            data["gt_bboxes_3d"] = gt_boxes
            
            if "voxel_in_LiDARInstance3DBoxes" in data:
                gt_boxes_in_voxel = data["voxel_in_LiDARInstance3DBoxes"]
                rotation_in_voxel = rotation @ gt_boxes_in_voxel.rotate(theta).numpy()
                gt_boxes_in_voxel.translate(translation)
                gt_boxes_in_voxel.scale(scale)
                data["voxel_in_LiDARInstance3DBoxes"] = gt_boxes_in_voxel

            transform[:3, :3] = rotation.T * scale
            transform[:3, 3] = translation * scale

        data["lidar_aug_matrix"] = transform #填充转换矩阵T
        return data

@PIPELINES.register_module()
class CutPasteObj:
    def __init__(self, 
                 ObjBankDir, 
                 category_list=[], 
                 sem_map={}, 
                 det_list=[],
                 paste_max_obj_num=20, 
                 paste_min_obj_num=0,
                 road_idx=[],
                 things_label_range=[],
                 min_pts_num=5,
                 time_interval=0.1,
                 with_yaw_fix=False,
                 max_epochs=20,
                 use_CutPasteObj=True):
        self.max_epoch=max_epochs
        self.epoch = -1
        self.use_CutPasteObj=use_CutPasteObj
        self.object_dir = ObjBankDir
        self.sem_map = sem_map
        self.det_list = det_list
        self.time_interval = time_interval
        self.with_yaw_fix = with_yaw_fix

        self.class_list = []
        self.sub_dirs_dic = {}
        self.velo_range_dic = {}
        for item in category_list:
            class_value = 1
            velo_range = (0,0)
            if isinstance(item, str):
                class_name = item
            elif isinstance(item, list) or isinstance(item, tuple):
                class_name = item[0]
                if len(item) >= 2:
                    class_value = item[1]
                if len(item) >= 3:
                    velo_range = item[2]
            else:
                class_name = item
            self.class_list += [class_name]*class_value
            fpath = os.path.join(self.object_dir, class_name)
            fname_list = [os.path.join(fpath, x) for x in os.listdir(fpath) if x.endswith('.npz')]
            print('Load {0}: {1}'.format(class_name, len(fname_list)))
            self.sub_dirs_dic[class_name] = fname_list
            self.velo_range_dic[class_name] = velo_range
        
        self.paste_min_obj_num = paste_min_obj_num
        self.paste_max_obj_num = paste_max_obj_num
        self.road_idx = road_idx
        self.things_label_range = things_label_range
        self.min_pts_num = min_pts_num
    
    def get_random_rotate_along_z_obj(self, pcds_obj, bbox_corners, theta):
        pcds_obj_result = rotate_along_z(pcds_obj, theta)
        bbox_corners_result = rotate_along_z(bbox_corners, theta)
        return pcds_obj_result, bbox_corners_result
    
    def get_fov(self, pcds_obj):
        x, y, z = pcds_obj[:, 0], pcds_obj[:, 1], pcds_obj[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
        u = np.sqrt(x ** 2 + y ** 2) + 1e-12

        phi = np.arctan2(x, y)
        theta = np.arcsin(z / d)

        u_fov = (u.min(), u.max())
        phi_fov = (phi.min(), phi.max())
        theta_fov = (theta.min(), theta.max())
        return u_fov, phi_fov, theta_fov
    
    def no_occlusion_check(self, pcds, pcds_label, phi_fov, theta_fov):
        x, y, z = pcds[:, 0], pcds[:, 1], pcds[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
        u = np.sqrt(x ** 2 + y ** 2) + 1e-12

        phi = np.arctan2(x, y)
        theta = np.arcsin(z / d)

        fov_mask = in_range(phi, phi_fov) * in_range(theta, theta_fov)
        pcds_label_in_fov = pcds_label[fov_mask]
        in_fov_obj_mask = (pcds_label_in_fov >= self.things_label_range[0]) * (pcds_label_in_fov <= self.things_label_range[1])
        if in_fov_obj_mask.sum() < 3:
            return True, fov_mask
        else:
            return False, fov_mask
    
    def no_collision_check(self, pcds, pcds_label, bbox_corners):
        in_box3d_mask = in_hull(pcds[:, :3], bbox_corners)
        pcds_label_in_box = pcds_label[in_box3d_mask]
        in_box_obj_mask = (pcds_label_in_box >= self.things_label_range[0]) * (pcds_label_in_box <= self.things_label_range[1])
        if in_box_obj_mask.sum() < 3:
            return True
        else:
            return False
        
    def gen_fov_mask(self, pcds, phi_fov, theta_fov):
        x, y, z = pcds[:, 0], pcds[:, 1], pcds[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
        u = np.sqrt(x ** 2 + y ** 2) + 1e-12

        phi = np.arctan2(x, y)
        theta = np.arcsin(z / d)
        fov_mask = in_range(phi, phi_fov) * in_range(theta, theta_fov)
        return fov_mask
    
    def paste_single_obj(self, pcds, pcds_road, pcds_label, gt_3d_box=None, gt_3d_label=None):
        '''
        Input:
            pcds, (N, 4), 4 -> x, y, z, intensity
            pcds_road, (M, 4)
            pcds_label, (N,)
        Output:
            pcds, (N1, 4)
            pcds_label, (N1,)
        '''
        # pcds (N, 4), 4 contains x, y, z, intensity
        # pcds_label(N)
        cate = random.choice(self.class_list)
        # fname_npz = random.choice(self.sub_dirs_dic[cate])
        # npkl = np.load(fname_npz)
        max_retry = 10  # 最大重试次数
        npkl = None
        
        for _ in range(max_retry):
            try:
                fname_npz = random.choice(self.sub_dirs_dic[cate])
                npkl = np.load(fname_npz)
                break  # 文件加载成功，跳出循环
            except FileNotFoundError:
                continue  # 文件不存在，继续尝试
        
        if npkl is None:  # 所有尝试都失败
            return pcds, pcds_label, gt_3d_box, gt_3d_label, 0

        pcds_obj = npkl['pcds']
        # sem_cate_id = int(npkl['cate_id'])
        semantic_cate = str(npkl['cate'])

        sem_cate_id = 0
        if semantic_cate in self.sem_map:
            sem_cate_id = self.sem_map.index(semantic_cate)

        det_cate_id = -1
        if semantic_cate in self.det_list:
            det_cate_id = self.det_list.index(semantic_cate)

        if 'gt_3d_box' in npkl: 
            obj_3d_box = npkl['gt_3d_box']
            bbox_corners = csr2corners(obj_3d_box[:3], obj_3d_box[3:6], obj_3d_box[6])
        else:  
            bbox_center = npkl['center']
            bbox_size = npkl['size']
            bbox_yaw = npkl['yaw']
            if self.with_yaw_fix: ## this for nuscenes
                bbox_yaw = bbox_yaw+np.pi/2
                bbox_size = np.array([bbox_size[1], bbox_size[0], bbox_size[2]])
            bbox_corners = csr2corners(bbox_center, bbox_size, bbox_yaw)

        if(len(pcds_obj) < self.min_pts_num):
            return pcds, pcds_label, gt_3d_box, gt_3d_label, 0
        
        theta_list = np.arange(0, 360, 18).tolist()
        np.random.shuffle(theta_list)
        for theta in theta_list:
            # global rotate object
            pcds_obj_aug, bbox_corners_aug = self.get_random_rotate_along_z_obj(pcds_obj, bbox_corners, theta)

            # get local road height
            valid_road_mask = in_hull(pcds_road[:, :2], bbox_corners_aug[:4, :2])
            pcds_local_road = pcds_road[valid_road_mask]
            if pcds_local_road.shape[0] > 5:
                road_mean_height = float(pcds_local_road[:, 2].mean())
                z_shift_value_box = road_mean_height - bbox_corners_aug[:, 2].min()
                z_shift_value_pcd = road_mean_height - pcds_obj_aug[:, 2].min()
                pcds_obj_aug[:, 2] += z_shift_value_box
                bbox_corners_aug[:, 2] += z_shift_value_box
            else:
                # object is not on road
                continue
            
            # get object fov
            u_fov, phi_fov, theta_fov = self.get_fov(pcds_obj_aug)
            if (abs(u_fov[1] - u_fov[0]) < 8) and (abs(phi_fov[1] - phi_fov[0]) < 1) and (abs(theta_fov[1] - theta_fov[0]) < 1):
                # if it is occlusion with the existing objects
                no_occlusion_flag, fov_mask = self.no_occlusion_check(pcds, pcds_label, phi_fov, theta_fov)
                # if it collides with existing objects
                no_collision_flag = self.no_collision_check(pcds, pcds_label, bbox_corners_aug)

                if no_occlusion_flag and no_collision_flag:
                    assert pcds.shape[0] == pcds_label.shape[0]

                    # add object back
                    pcds_filter = pcds[~fov_mask]
                    pcds_label_filter = pcds_label[~fov_mask]

                    if pcds_obj_aug.shape[1] < pcds_filter.shape[1]:
                        pcds_obj_aug = np.pad(pcds_obj_aug, ((0,0),(0,1)), 'constant', constant_values=0)
                    else:
                        pcds_obj_aug = pcds_obj_aug[:, :pcds_filter.shape[1]]
                    pcds = np.concatenate((pcds_filter, pcds_obj_aug), axis=0)

                    pcds_addobj_label = np.full((pcds_obj_aug.shape[0],), fill_value=sem_cate_id, dtype=pcds_label.dtype)
                    pcds_label = np.concatenate((pcds_label_filter, pcds_addobj_label), axis=0)

                    if gt_3d_box is not None:
                        gt_3d_box = np.pad(gt_3d_box, ((0,1),(0,0)), 'constant', constant_values=0)
                        obj_box_aug = corners2csr_batch(bbox_corners_aug.reshape(1, 8, 3))
                        obj_box_aug[:, -1] *= -1 ##角度翻转
                        obj_box_aug[:, 2] -= obj_box_aug[:, 5] * 0.5  ## Z点由center变为bottom

                        gt_3d_box[-1, :7] = obj_box_aug[0]
                        gt_3d_label = np.pad(gt_3d_label, (0,1), 'constant', constant_values=-1)
                        gt_3d_label[-1] = det_cate_id

                    break
                else:
                    # invalid heading
                    continue
            else:
                break
        
        return pcds, pcds_label, gt_3d_box, gt_3d_label, 1
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Input:
            pcds, (N, 4), 4 -> x, y, z, intensity
            pcds_label, (N,)
            gt_3d_box, (K, 8), 8 -> (cx, cy, cz, sx, sy, sz, yaw, cls)
        '''
        # print(self.epoch)
        if  not self.use_CutPasteObj:
            if  self.max_epoch-self.epoch<6:
                return data
        paste_obj_num = random.randint(self.paste_min_obj_num, self.paste_max_obj_num)
        if paste_obj_num == 0:
            return data
        else:
            pcds = data['points']
            pcds_label = data['points_labels']
            gt_bboxes_3d = data['gt_bboxes_3d']
            gt_labels_3d = data['gt_labels_3d']

            pcds_new = pcds.tensor.numpy().copy()
            pcds_label_new = pcds_label.copy()

            pcds_road = [pcds_new[pcds_label == i] for i in self.road_idx]
            pcds_road = np.concatenate(pcds_road, axis=0)

            gt_3d_box_new = gt_bboxes_3d.tensor.numpy().copy()
            gt_3d_label_new = gt_labels_3d.copy()

            paste_num = 0
            while paste_num <= paste_obj_num:
                pcds_new, pcds_label_new, gt_3d_box_new, gt_3d_label_new, paste_flag = self.paste_single_obj(pcds_new, 
                                                                                                             pcds_road, 
                                                                                                             pcds_label_new, 
                                                                                                             gt_3d_box_new, 
                                                                                                             gt_3d_label_new)
                paste_num += paste_flag
            data['points'] = pcds.new_point(pcds_new)
            data['points_labels'] = pcds_label_new
            data['gt_bboxes_3d'] = gt_bboxes_3d.new_box(gt_3d_box_new)
            data['gt_labels_3d'] = gt_3d_label_new
            return data

@PIPELINES.register_module()
class CutPasteNoise:
    def __init__(self, 
                 ObjBankDir,
                 category_list,
                 paste_max_obj_num,
                 min_pts_num):
        self.object_dir = ObjBankDir
        self.sub_dirs = category_list
        self.sub_dirs_dic = {}
        self.class_list = []
        for fp, class_value in self.sub_dirs:
            fpath = os.path.join(self.object_dir, fp)
            fname_list = [os.path.join(fpath, x) for x in os.listdir(fpath) if x.endswith('.npz')]
            print('Load {0}: {1}'.format(fp, len(fname_list)))
            self.sub_dirs_dic[fp] = fname_list
            self.class_list += [fp]*class_value

        self.paste_max_obj_num = paste_max_obj_num
        self.min_pts_num = min_pts_num

    def make_sequential_obj(self, fname_npz):
        npkl = np.load(fname_npz)

        pcds_obj = npkl['pcds']
        cate_id = int(npkl['cate_id'])
        semantic_cate = str(npkl['cate'])

        if(len(pcds_obj) < self.min_pts_num):
            return None, None, None
        
        pcds_obj[:, :3] += np.random.normal(0, 0.001, size=(pcds_obj.shape[0], 3))
        return pcds_obj, cate_id, semantic_cate

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # import ipdb; ipdb.set_trace()
        paste_obj_num = random.randint(0, self.paste_max_obj_num)
        if paste_obj_num == 0:
            return data
        else:
            pcds = data['points']
            pcds_label = data['points_labels']

            pcds_new = pcds.tensor.numpy().copy()
            pcds_label_new = pcds_label.copy()
            for i in range(paste_obj_num):
                cate = random.choice(self.class_list)
                fname_npz = random.choice(self.sub_dirs_dic[cate])
                pc_object, cate_id, semantic_cate = self.make_sequential_obj(fname_npz)
                if pc_object is None:
                    continue

                pcds_new = np.concatenate((pcds_new, pc_object), axis=0)
                pcds_addobj_label = np.full((pc_object.shape[0],), fill_value=cate_id, dtype=pcds_label_new.dtype)
                pcds_label_new = np.concatenate((pcds_label_new, pcds_addobj_label), axis=0)

            data['points'] = pcds.new_point(pcds_new)
            data['points_labels'] = pcds_label_new
            return data

@PIPELINES.register_module()
class RandomModalityDrop:
    def __init__(self, only_camera_drop: bool = False, only_lidar_drop: bool = False, drop_rate: float = 0.0) -> None:
        self.drop_rate = drop_rate
        self.only_camera_drop = only_camera_drop
        self.only_lidar_drop = only_lidar_drop

    def camera_drop(self, imgs):
        drop_arr = np.random.randint(0, 2, len(imgs))
        for drop_idx, flag in enumerate(drop_arr):
            if flag:
                imgs[drop_idx] = Image.new('RGB', imgs[drop_idx].size, (0, 0, 0))

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        drop = np.random.choice([False, True], replace=False, p=[1-self.drop_rate, self.drop_rate])
        if drop:
            if self.only_camera_drop:
                self.camera_drop(data["img"])
            elif self.only_lidar_drop:
                data["points"].value = torch.ones(data["points"].shape[0], 4)
            else:
                drop_modality = np.random.choice([0, 1], replace=False, p=[0.5, 0.5])
                if drop_modality == 0:
                    self.camera_drop(data["img"])
                else:
                    data["points"].value = torch.ones(data["points"].shape[0], 4)
            
        return data

@PIPELINES.register_module()
class GridMask:
    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results["img"]
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results.update(img=imgs)
        return results


@PIPELINES.register_module()
class RandomFlip3D:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = random.choice([0, 1])
        flip_vertical = random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("horizontal")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("horizontal")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, ::-1].copy()
            if "voxel_in_LiDARInstance3DBoxes" in data:
                data["voxel_in_LiDARInstance3DBoxes"].flip("horizontal")

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("vertical")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("vertical")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][::-1, :].copy()
            if "voxel_in_LiDARInstance3DBoxes" in data:
                data["voxel_in_LiDARInstance3DBoxes"].flip("vertical")

        data["lidar_aug_matrix"][:3, :] = rotation @ data["lidar_aug_matrix"][:3, :]
        return data

@PIPELINES.register_module()
class RandomFlipHorizontal3D:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = random.choice([0, 1])
        # flip_vertical = random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("horizontal")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("horizontal")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, ::-1].copy()
            if "voxel_in_LiDARInstance3DBoxes" in data:
                data["voxel_in_LiDARInstance3DBoxes"].flip("horizontal")
            if 'hdmap_geom' in data:
                for k, polylines in data['hdmap_geom'].items():
                    for pts in polylines:
                        pts.flip("horizontal")

        # if flip_vertical:
        #     rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
        #     if "points" in data:
        #         data["points"].flip("vertical")
        #     if "gt_bboxes_3d" in data:
        #         data["gt_bboxes_3d"].flip("vertical")
        #     if "gt_masks_bev" in data:
        #         data["gt_masks_bev"] = data["gt_masks_bev"][::-1, :].copy()

        data["lidar_aug_matrix"][:3, :] = rotation @ data["lidar_aug_matrix"][:3, :]
        return data


@PIPELINES.register_module()
class ObjectPaste:
    """Sample GT objects to the data.
    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False, stop_epoch=None):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if "type" not in db_sampler.keys():
            db_sampler["type"] = "DataBaseSampler"
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.epoch = -1
        self.stop_epoch = stop_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.
        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.
        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, data):
        """Call function to sample ground truth objects to the data.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
            return data
        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]

        # change to float for blending operation
        points = data["points"]
        if self.sample_2d:
            img = data["img"]
            gt_bboxes_2d = data["gt_bboxes"]
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img,
            )
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None
            )

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict["gt_bboxes_3d"]
            sampled_points = sampled_dict["points"]
            sampled_gt_labels = sampled_dict["gt_labels_3d"]

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels], axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d])
            )

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict["gt_bboxes_2d"]
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]
                ).astype(np.float32)

                data["gt_bboxes"] = gt_bboxes_2d
                data["img"] = sampled_dict["img"]

        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d.astype(np.long)
        data["points"] = points

        return data


@PIPELINES.register_module()
class ObjectNoise:
    """Apply noise to each GT objects in the scene.
    Args:
        translation_std (list[float], optional): Standard deviation of the
            distribution where translation noise are sampled from.
            Defaults to [0.25, 0.25, 0.25].
        global_rot_range (list[float], optional): Global rotation to the scene.
            Defaults to [0.0, 0.0].
        rot_range (list[float], optional): Object rotation range.
            Defaults to [-0.15707963267, 0.15707963267].
        num_try (int, optional): Number of times to try if the noise applied is
            invalid. Defaults to 100.
    """

    def __init__(
        self,
        translation_std=[0.25, 0.25, 0.25],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267],
        num_try=100,
    ):
        self.translation_std = translation_std
        self.global_rot_range = global_rot_range
        self.rot_range = rot_range
        self.num_try = num_try

    def __call__(self, data):
        """Call function to apply noise to each ground truth in the scene.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after adding noise to each object, \
                'points', 'gt_bboxes_3d' keys are updated in the result dict.
        """
        gt_bboxes_3d = data["gt_bboxes_3d"]
        points = data["points"]

        # TODO: check this inplace function
        numpy_box = gt_bboxes_3d.tensor.numpy()
        numpy_points = points.tensor.numpy()

        noise_per_object_v3_(
            numpy_box,
            numpy_points,
            rotation_perturb=self.rot_range,
            center_noise_std=self.translation_std,
            global_random_rot_range=self.global_rot_range,
            num_try=self.num_try,
        )

        data["gt_bboxes_3d"] = gt_bboxes_3d.new_box(numpy_box)
        data["points"] = points.new_point(numpy_points)
        return data


@PIPELINES.register_module()
class FrameDropout:
    def __init__(self, prob: float = 0.5, time_dim: int = -1) -> None:
        self.prob = prob
        self.time_dim = time_dim

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        offsets = []
        for offset in torch.unique(data["points"].tensor[:, self.time_dim]):
            if offset == 0 or random.random() > self.prob:
                offsets.append(offset)
        offsets = torch.tensor(offsets)

        points = data["points"].tensor
        indices = torch.isin(points[:, self.time_dim], offsets)
        data["points"].tensor = points[indices]
        return data


@PIPELINES.register_module()
class PointShuffle:
    def __call__(self, data):
        data["points"].shuffle()
        return data


@PIPELINES.register_module()
class ObjectRangeFilter:
    """Filter objects by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        """Call function to filter objects by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(
            data["gt_bboxes_3d"], (LiDARInstance3DBoxes, DepthInstance3DBoxes)
        ):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(data["gt_bboxes_3d"], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d
        
        # if ('voxel_in_LiDARInstance3DBoxes' in data) and ('voxel_in_semantics' in data):
        #     gt_bboxes_3d_voxel = data["voxel_in_LiDARInstance3DBoxes"]
        #     gt_labels_3d_voxel = data["voxel_in_semantics"]
            
        #     if gt_bboxes_3d_voxel.tensor.shape[0] != gt_labels_3d_voxel.shape[0]:
        #         breakpoint()
        #     mask_voxel = gt_bboxes_3d_voxel.in_range_bev(bev_range)
            
        #     gt_bboxes_3d_voxel = gt_bboxes_3d_voxel[mask_voxel]
        #     gt_labels_3d_voxel = gt_labels_3d_voxel[mask_voxel.numpy().astype(np.bool)]
            
        #     data["voxel_in_LiDARInstance3DBoxes"] = gt_bboxes_3d_voxel
        #     data["voxel_in_semantics"] = gt_labels_3d_voxel

        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(point_cloud_range={self.pcd_range.tolist()})"
        return repr_str


@PIPELINES.register_module()
class PointsRangeFilter:
    """Filter points by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        """Call function to filter points by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = data["points"]
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        data["points"] = clean_points
        # data["points_labels"] = data["points_labels"][points_mask]
        # data["points_grid_ind"] = data["points_grid_ind"][points_mask]
        return data


@PIPELINES.register_module()
class ObjectNameFilter:
    """Filter GT objects by their names.
    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, data):
        gt_labels_3d = data["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        data["gt_bboxes_3d"] = data["gt_bboxes_3d"][gt_bboxes_mask]
        data["gt_labels_3d"] = data["gt_labels_3d"][gt_bboxes_mask]
        return data


@PIPELINES.register_module()
class PointSample:
    """Point sample.
    Sampling data to a certain number.
    Args:
        num_points (int): Number of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool, optional): Whether the sampling is with or without
            replacement. Defaults to False.
    """

    def __init__(self, num_points, sample_range=None, replace=False):
        self.num_points = num_points
        self.sample_range = sample_range
        self.replace = replace

    def _points_random_sampling(
        self,
        points,
        num_samples,
        sample_range=None,
        replace=False,
        return_choices=False,
    ):
        """Points random sampling.
        Sample points to a certain number.
        Args:
            points (np.ndarray | :obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.
            sample_range (float, optional): Indicating the range where the
                points will be sampled. Defaults to None.
            replace (bool, optional): Sampling with or without replacement.
                Defaults to None.
            return_choices (bool, optional): Whether return choice.
                Defaults to False.
        Returns:
            tuple[np.ndarray] | np.ndarray:
                - points (np.ndarray | :obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if not replace:
            replace = points.shape[0] < num_samples
        point_range = range(len(points))
        if sample_range is not None and not replace:
            # Only sampling the near points when len(points) >= num_samples
            depth = np.linalg.norm(points.tensor, axis=1)
            far_inds = np.where(depth > sample_range)[0]
            near_inds = np.where(depth <= sample_range)[0]
            # in case there are too many far points
            if len(far_inds) > num_samples:
                far_inds = np.random.choice(far_inds, num_samples, replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            # Shuffle points after sampling
            np.random.shuffle(choices)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def __call__(self, data):
        """Call function to sample points to in indoor scenes.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = data["points"]
        # Points in Camera coord can provide the depth information.
        # TODO: Need to suport distance-based sampling for other coord system.
        if self.sample_range is not None:
            from mmdet3d.core.points import CameraPoints

            assert isinstance(
                points, CameraPoints
            ), "Sampling based on distance is only appliable for CAMERA coord"
        points, choices = self._points_random_sampling(
            points,
            self.num_points,
            self.sample_range,
            self.replace,
            return_choices=True,
        )
        data["points"] = points
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(num_points={self.num_points},"
        repr_str += f" sample_range={self.sample_range},"
        repr_str += f" replace={self.replace})"

        return repr_str


@PIPELINES.register_module()
class BackgroundPointsFilter:
    """Filter background points near the bounding box.
    Args:
        bbox_enlarge_range (tuple[float], float): Bbox enlarge range.
    """

    def __init__(self, bbox_enlarge_range):
        assert (
            is_tuple_of(bbox_enlarge_range, float) and len(bbox_enlarge_range) == 3
        ) or isinstance(
            bbox_enlarge_range, float
        ), f"Invalid arguments bbox_enlarge_range {bbox_enlarge_range}"

        if isinstance(bbox_enlarge_range, float):
            bbox_enlarge_range = [bbox_enlarge_range] * 3
        self.bbox_enlarge_range = np.array(bbox_enlarge_range, dtype=np.float32)[
            np.newaxis, :
        ]

    def __call__(self, data):
        """Call function to filter points by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = data["points"]
        gt_bboxes_3d = data["gt_bboxes_3d"]

        # avoid groundtruth being modified
        gt_bboxes_3d_np = gt_bboxes_3d.tensor.clone().numpy()
        gt_bboxes_3d_np[:, :3] = gt_bboxes_3d.gravity_center.clone().numpy()

        enlarged_gt_bboxes_3d = gt_bboxes_3d_np.copy()
        enlarged_gt_bboxes_3d[:, 3:6] += self.bbox_enlarge_range
        points_numpy = points.tensor.clone().numpy()
        foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, gt_bboxes_3d_np, origin=(0.5, 0.5, 0.5)
        )
        enlarge_foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, enlarged_gt_bboxes_3d, origin=(0.5, 0.5, 0.5)
        )
        foreground_masks = foreground_masks.max(1)
        enlarge_foreground_masks = enlarge_foreground_masks.max(1)
        valid_masks = ~np.logical_and(~foreground_masks, enlarge_foreground_masks)

        data["points"] = points[valid_masks]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(bbox_enlarge_range={self.bbox_enlarge_range.tolist()})"
        return repr_str


@PIPELINES.register_module()
class VoxelBasedPointSampler:
    """Voxel based point sampler.
    Apply voxel sampling to multiple sweep points.
    Args:
        cur_sweep_cfg (dict): Config for sampling current points.
        prev_sweep_cfg (dict): Config for sampling previous points.
        time_dim (int): Index that indicate the time dimention
            for input points.
    """

    def __init__(self, cur_sweep_cfg, prev_sweep_cfg=None, time_dim=3):
        self.cur_voxel_generator = VoxelGenerator(**cur_sweep_cfg)
        self.cur_voxel_num = self.cur_voxel_generator._max_voxels
        self.time_dim = time_dim
        if prev_sweep_cfg is not None:
            assert prev_sweep_cfg["max_num_points"] == cur_sweep_cfg["max_num_points"]
            self.prev_voxel_generator = VoxelGenerator(**prev_sweep_cfg)
            self.prev_voxel_num = self.prev_voxel_generator._max_voxels
        else:
            self.prev_voxel_generator = None
            self.prev_voxel_num = 0

    def _sample_points(self, points, sampler, point_dim):
        """Sample points for each points subset.
        Args:
            points (np.ndarray): Points subset to be sampled.
            sampler (VoxelGenerator): Voxel based sampler for
                each points subset.
            point_dim (int): The dimention of each points
        Returns:
            np.ndarray: Sampled points.
        """
        voxels, coors, num_points_per_voxel = sampler.generate(points)
        if voxels.shape[0] < sampler._max_voxels:
            padding_points = np.zeros(
                [
                    sampler._max_voxels - voxels.shape[0],
                    sampler._max_num_points,
                    point_dim,
                ],
                dtype=points.dtype,
            )
            padding_points[:] = voxels[0]
            sample_points = np.concatenate([voxels, padding_points], axis=0)
        else:
            sample_points = voxels

        return sample_points

    def __call__(self, results):
        """Call function to sample points from multiple sweeps.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results["points"]
        original_dim = points.shape[1]

        # TODO: process instance and semantic mask while _max_num_points
        # is larger than 1
        # Extend points with seg and mask fields
        map_fields2dim = []
        start_dim = original_dim
        points_numpy = points.tensor.numpy()
        extra_channel = [points_numpy]
        for idx, key in enumerate(results["pts_mask_fields"]):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        start_dim += len(results["pts_mask_fields"])
        for idx, key in enumerate(results["pts_seg_fields"]):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        points_numpy = np.concatenate(extra_channel, axis=-1)

        # Split points into two part, current sweep points and
        # previous sweeps points.
        # TODO: support different sampling methods for next sweeps points
        # and previous sweeps points.
        cur_points_flag = points_numpy[:, self.time_dim] == 0
        cur_sweep_points = points_numpy[cur_points_flag]
        prev_sweeps_points = points_numpy[~cur_points_flag]
        if prev_sweeps_points.shape[0] == 0:
            prev_sweeps_points = cur_sweep_points

        # Shuffle points before sampling
        np.random.shuffle(cur_sweep_points)
        np.random.shuffle(prev_sweeps_points)

        cur_sweep_points = self._sample_points(
            cur_sweep_points, self.cur_voxel_generator, points_numpy.shape[1]
        )
        if self.prev_voxel_generator is not None:
            prev_sweeps_points = self._sample_points(
                prev_sweeps_points, self.prev_voxel_generator, points_numpy.shape[1]
            )

            points_numpy = np.concatenate([cur_sweep_points, prev_sweeps_points], 0)
        else:
            points_numpy = cur_sweep_points

        if self.cur_voxel_generator._max_num_points == 1:
            points_numpy = points_numpy.squeeze(1)
        results["points"] = points.new_point(points_numpy[..., :original_dim])

        # Restore the correspoinding seg and mask fields
        for key, dim_index in map_fields2dim:
            results[key] = points_numpy[..., dim_index]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""

        def _auto_indent(repr_str, indent):
            repr_str = repr_str.split("\n")
            repr_str = [" " * indent + t + "\n" for t in repr_str]
            repr_str = "".join(repr_str)[:-1]
            return repr_str

        repr_str = self.__class__.__name__
        indent = 4
        repr_str += "(\n"
        repr_str += " " * indent + f"num_cur_sweep={self.cur_voxel_num},\n"
        repr_str += " " * indent + f"num_prev_sweep={self.prev_voxel_num},\n"
        repr_str += " " * indent + f"time_dim={self.time_dim},\n"
        repr_str += " " * indent + "cur_voxel_generator=\n"
        repr_str += f"{_auto_indent(repr(self.cur_voxel_generator), 8)},\n"
        repr_str += " " * indent + "prev_voxel_generator=\n"
        repr_str += f"{_auto_indent(repr(self.prev_voxel_generator), 8)})"
        return repr_str


@PIPELINES.register_module()
class ImagePad:
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results["img"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in results["img"]
            ]
        results["img"] = padded_img
        results["img_shape"] = [img.shape for img in padded_img]
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class ImageNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["img"] = [self.compose(img) for img in data["img"]]
        data["img_norm_cfg"] = dict(mean=self.mean, std=self.std)
        return data


@PIPELINES.register_module()
class ImageDistort:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        data["img"] = new_imgs
        return data
