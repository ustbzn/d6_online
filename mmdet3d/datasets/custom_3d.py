import tempfile
import os
from os import path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.datasets import DATASETS

from ..core.bbox import get_box_type
from .pipelines import Compose
from .utils import extract_result_dict
import matplotlib.cm as cmx

NUM_COLORS = 30

def getcolors(number=NUM_COLORS):
    colorlist = list(cmx.rainbow(np.linspace(0, 1, number)))
    colorlist = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in colorlist]
    return colorlist

def draw_2d_points_to_image(image, draw_points_2d, depth):
    import cv2
    h, w, _ = image.shape
    pts = np.array(draw_points_2d)
    depth = np.array(depth)
    # print('pts', pts.shape, 'depth', depth.shape)
    # generate color map from depth
    colorlist = getcolors(NUM_COLORS)
    max_x = np.max(depth[:, 0])
    min_x = np.min(depth[:, 0])
    step_x = (max_x - min_x) / NUM_COLORS
    # draw points
    n = pts.shape[0]
    show_image = image.copy()
    count = 0
    for i in range(n):
        c = int((depth[i] - min_x) / step_x)
        c = c if c <= NUM_COLORS - 1 else NUM_COLORS - 1
        if (0 < int(pts[i][0]) < w) and (0 < int(pts[i][1]) < h):
            cv2.circle(show_image, (int(pts[i][0]), int(pts[i][1])), radius=1, color=colorlist[c], thickness=-1)
            count += 1
    return show_image


@DATASETS.register_module()
class Custom3DDataset(Dataset):
    """Customized 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.

    Args:
        dataset_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(
        self,
        dataset_root,
        ann_file,
        pipeline=None,
        classes=None,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        point_cloud_range=None,
        metric_split = None,
        metric_name = None,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if metric_split != None:
            filtered_infos = []
            for info in self.data_infos:
                value  = info['scene_tags']['raw_ids'].get(metric_split, False)
                if value  == metric_name:
                    filtered_infos.append(info)

            self.data_infos = filtered_infos

        self.point_cloud_range = point_cloud_range

        if pipeline is not None:
            self.pipeline = Compose(pipeline)#transform compose

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        self.epoch = -1
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self, "pipeline"):
            for transform in self.pipeline.transforms:
                if hasattr(transform, "set_epoch"):
                    transform.set_epoch(epoch)
        
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        return mmcv.load(ann_file)

    def get_data_info(self, index):#不执行
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - lidar_path (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info["point_cloud"]["lidar_idx"]
        lidar_path = osp.join(self.dataset_root, info["pts_path"])

        input_dict = dict(
            lidar_path=lidar_path, sample_idx=sample_idx, file_name=lidar_path
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos
            if self.filter_empty_gt and ~(annos["gt_labels_3d"] != -1).any():
                return None
        return input_dict

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d
        results["points_labels"] = []
        # results["gt_masks_bev"] = []
        results["points_grid_ind"] = []

    def prepare_train_data(self, index):#!!!
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and (
            example is None or ~(example["gt_labels_3d"]._data != -1).any()
        ):
            return None
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        return class_names

    def format_results(self, outputs, pklfile_prefix=None, submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results, \
                tmp_dir is the temporal directory created for saving json \
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, "results")
            out = f"{pklfile_prefix}.pkl"
        mmcv.dump(outputs, out)
        return outputs, tmp_dir

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, "data loading pipeline is not provided"
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            data = self.prepare_test_data(idx)
            return data
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            # ### 1、visualize lidar aug start
            # # import ipdb;ipdb.set_trace()
            # import os
            # import copy
            # from mmdet3d.core import LiDARInstance3DBoxes
            # from mmdet3d.core.utils import visualize_camera, visualize_lidar
            # data_vis = copy.deepcopy(data)
            # metas = data_vis["metas"].data
            # name = "{}".format(metas["timestamp"])
            # name = '_'.join([metas['lidar_path'].split('/')[6], metas['lidar_path'].split('/')[8]])

            # bboxes = data_vis["gt_bboxes_3d"].data.tensor.numpy()
            # labels = data_vis["gt_labels_3d"].data.numpy()

            # bboxes = bboxes[labels > -1]
            # labels = labels[labels > -1]

            # bboxes[..., 2] -= bboxes[..., 5] / 2 #0.5, 0.5, 0.5 -> 0.5, 0.5, 0
            # bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)

            # lidar = data_vis["points"].data.numpy()
            # visualize_lidar(
            #     f"lidar_debug_vis/{name}_0.png",
            #     lidar,
            #     bboxes=bboxes,
            #     labels=labels,
            #     xlim=[self.point_cloud_range[0], self.point_cloud_range[3]],
            #     ylim=[self.point_cloud_range[1], self.point_cloud_range[4]],
            #     classes=self.CLASSES,
            # )
            # ## visualize lidar aug end
            # ### 2、visualize image aug start
            # # import ipdb; ipdb.set_trace()
            # import cv2
            # import torch
            # metas = data["metas"].data
            # folder = metas['lidar_path'].split('/')[5]
            # os.makedirs('train_vis_undist/%s'%folder, exist_ok=True)
            # name = "{}".format(metas["timestamp"])
            # print(name)
            # img = data["img"].data.numpy()#(3, 3, 256, 704) N*C*H*W
            # img_hwc = img.transpose(0,2,3,1)#(3, 256, 704, 3)
            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225]
            # img_hwc *= std
            # img_hwc += mean#反ImageNormalize操作
            # img_hwc *= 255#从0~1转换为0~255

            # cur_lidar_aug_matrix = data["lidar_aug_matrix"].data
            # cur_coords_ori = data["points"].data[:, :3]
            # cur_coords_ori -= cur_lidar_aug_matrix[:3, 3]
            # cur_coords_ori = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
            #     cur_coords_ori.transpose(1, 0)
            # )

            # vis_imgs = []
            # for img_idx in range(img_hwc.shape[0]):
            #     cur_lidar2image = torch.tensor(metas["lidar2image"][img_idx])
            #     cur_img_aug_matrix = data["img_aug_matrix"].data[img_idx]

            #     cur_coords = cur_lidar2image[:3, :3].matmul(cur_coords_ori)
            #     cur_coords += cur_lidar2image[:3, 3].reshape(3, 1) # torch.Size([1, 3, n]),将点云转换到图片坐标系
            #     # get 2d coords
            #     dist = cur_coords[2, :]
            #     cur_coords[2, :] = torch.clamp(cur_coords[2, :], 1e-5, 1e5)
            #     cur_coords[:2, :] /= cur_coords[2:3, :]

            #     cur_coords = cur_img_aug_matrix[:3, :3].matmul(cur_coords)
            #     cur_coords += cur_img_aug_matrix[:3, 3].reshape(3, 1)#图像坐标系下的点云进行图像的数据增强转换

            #     draw_points_2d = cur_coords[:2, :].transpose(1, 0).cpu().numpy()
            #     depth = cur_coords[2:3, :].transpose(1, 0).cpu().numpy()

            #     draw_points_2d[np.isnan(draw_points_2d)] = 0
            #     del_xindx, del_yindx = np.where(draw_points_2d[:, :] == 0)
            #     draw_points_2d = np.delete(draw_points_2d, del_xindx, axis=0)
            #     depth = np.delete(depth, del_xindx, axis=0)
            #     show_img = draw_2d_points_to_image(img_hwc[img_idx][:,:,::-1], draw_points_2d, depth)
            #     vis_imgs.append(np.concatenate((img_hwc[img_idx][:,:,::-1], show_img), axis=1))

            # visres = vis_imgs[0]
            # for idx in range(1, len(vis_imgs)):
            #     visres = np.concatenate((visres, vis_imgs[idx]), axis=0)
            # cv2.imwrite(f"train_vis_undist/{folder}/{name}_img.png", visres)
            # ### visualize image aug end

            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
