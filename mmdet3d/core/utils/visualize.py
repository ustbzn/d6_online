import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from ..bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]


# OBJECT_PALETTE = {
#     "person": (255, 158, 0),
#     "pushing": (255, 99, 71),
#     "bike": (233, 150, 70),
#     "rider": (255, 69, 0),
#     "car": (255, 140, 0),
#     "truck": (112, 128, 144),
#     "bus": (255, 61, 99),
#     "obstacle": (220, 20, 60),
#     "sign": (0, 0, 230),
#     "noise": (47, 79, 79),
# }
OBJECT_PALETTE = {
    "person": (255, 255, 0),
    "bike": (0, 255, 0),
    "car": (0, 255, 255),
    "truck": (255, 0, 0),
    "bus": (255, 0, 255),
    "rider": (0, 0, 230),
    "gt": (255, 165, 0),
}

OBJECT_PALETTE_BGR = {
    "person": (0, 255, 255),    # 黄色
    "bike": (0, 255, 0),        # 绿色
    "car": (255, 255, 0),       # 青色
    "truck": (0, 0, 255),       # 红色
    "bus": (255, 0, 255),       # 紫色
    "rider": (230, 0, 0),       # 蓝色
    "gt": (0, 165, 255),        # 橙色
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


# def visualize_camera(
#     fpath: str,
#     image: np.ndarray,
#     *,
#     bboxes: Optional[LiDARInstance3DBoxes] = None,
#     labels: Optional[np.ndarray] = None,
#     transform: Optional[np.ndarray] = None,
#     classes: Optional[List[str]] = None,
#     color: Optional[Tuple[int, int, int]] = None,
#     thickness: float = 4,
# ) -> None:
#     canvas = image.copy()
#     canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

#     if bboxes is not None and len(bboxes) > 0:
#         corners = bboxes.corners
#         num_bboxes = corners.shape[0]

#         coords = np.concatenate(
#             [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
#         )
#         transform = copy.deepcopy(transform).reshape(4, 4)
#         coords = coords @ transform.T
#         coords = coords.reshape(-1, 8, 4)

#         indices = np.all(coords[..., 2] > 0, axis=1)
#         coords = coords[indices]
#         labels = labels[indices]

#         indices = np.argsort(-np.min(coords[..., 2], axis=1))
#         coords = coords[indices]
#         labels = labels[indices]

#         coords = coords.reshape(-1, 4)
#         coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
#         coords[:, 0] /= coords[:, 2]
#         coords[:, 1] /= coords[:, 2]

#         coords = coords[..., :2].reshape(-1, 8, 2)
#         for index in range(coords.shape[0]):
#             name = classes[labels[index]]
#             for start, end in [
#                 (0, 1),
#                 (0, 3),
#                 (0, 4),
#                 (1, 2),
#                 (1, 5),
#                 (3, 2),
#                 (3, 7),
#                 (4, 5),
#                 (4, 7),
#                 (2, 6),
#                 (5, 6),
#                 (6, 7),
#             ]:
#                 cv2.line(
#                     canvas,
#                     coords[index, start].astype(np.int),
#                     coords[index, end].astype(np.int),
#                     color or OBJECT_PALETTE[name],
#                     thickness,
#                     cv2.LINE_AA,
#                 )
#         canvas = canvas.astype(np.uint8)
#     canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

#     mmcv.mkdir_or_exist(os.path.dirname(fpath))
#     mmcv.imwrite(canvas, fpath)


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 10,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)

def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    lidar2camera_single: Optional[np.ndarray] = None,
    camera_intrinsics_single: Optional[np.ndarray] = None,
    img_aug_matrix_single: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()  
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is None or len(bboxes) == 0:
        mmcv.mkdir_or_exist(os.path.dirname(fpath))
        mmcv.imwrite(canvas, fpath)
        return
    corners = bboxes.corners  # (N, 8, 3)
    num_bboxes = corners.shape[0]

    coords = np.concatenate([corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1)  # (N*8,4)
    # LiDAR -> Camera
    lidar2camera_single = copy.deepcopy(lidar2camera_single).reshape(4, 4)

    coords = coords @ lidar2camera_single.T
    coords = coords.reshape(-1, 8, 4)

    # 过滤：只要 bbox 至少有一个角点在相机前就保留
    mask_front = np.any(coords[..., 2] > 1e-5, axis=1)
    coords = coords[mask_front]
    if labels is not None:
        labels = labels[mask_front]

    if len(coords) == 0:
        mmcv.mkdir_or_exist(os.path.dirname(fpath))
        mmcv.imwrite(canvas, fpath)
        return

    # 按深度排序（近的后画）
    sort_idx = np.argsort(-np.min(coords[..., 2], axis=1))
    coords = coords[sort_idx]
    if labels is not None:
        labels = labels[sort_idx]

    # 投影到相机像素
    coords_reshape = coords.reshape(-1, 4)
    coords_reshape[:, 2] = np.clip(coords_reshape[:, 2], 1e-5, 1e5)

    K = copy.deepcopy(camera_intrinsics_single)
    if K.shape == (4, 4):
        K = K[:3, :3]
    else:
        K = K.reshape(3, 3)

    coords_proj = (K @ coords_reshape[:, :3].T).T
    coords_proj[:, 0] /= (coords_proj[:, 2] + 1e-5)
    coords_proj[:, 1] /= (coords_proj[:, 2] + 1e-5)
    coords_proj = coords_proj[:, :2].reshape(-1, 8, 2)

    # 图像增强矩阵
    if img_aug_matrix_single is not None:
        coords_aug = []
        for i in range(coords_proj.shape[0]):
            pts = coords_proj[i]  # (8,2)
            if img_aug_matrix_single.shape == (3, 3):
                pts_h = np.concatenate([pts, np.ones((8, 1))], axis=-1)
                pts_aug = (img_aug_matrix_single @ pts_h.T).T
                pts_aug[:, :2] /= pts_aug[:, 2:3]
            elif img_aug_matrix_single.shape == (4, 4):
                pts_h = np.concatenate([pts, np.zeros((8, 1))], axis=-1)
                pts_aug = (img_aug_matrix_single[:3,:3] @ pts_h.T).T
                pts_aug += img_aug_matrix_single[:3, 3:4].T
            else:
                raise ValueError(f"Unsupported img_aug_matrix_single shape: {img_aug_matrix_single.shape}")
            coords_aug.append(pts_aug[:, :2])
        coords_proj = np.stack(coords_aug, axis=0)

    H, W = image.shape[0], image.shape[1]
    # breakpoint()
    # 过滤 NaN / inf
    valid_mask = np.all(np.isfinite(coords_proj), axis=(1, 2))
    coords_proj = coords_proj[valid_mask]
    if labels is not None:
        labels = labels[valid_mask]

    # 过滤超出图像范围的 bbox（至少一个角点在图像内就保留）
    in_range_mask = np.any(
        (coords_proj[..., 0] >= 0) & (coords_proj[..., 0] < W) &
        (coords_proj[..., 1] >= 0) & (coords_proj[..., 1] < H),
        axis=1
    )
    coords_proj = coords_proj[in_range_mask]
    if labels is not None:
        labels = labels[in_range_mask]
    
    # 画 3D bbox
    for i in range(coords_proj.shape[0]):
        name = classes[labels[i]] if classes is not None else "object"
        for start, end in [
            (0, 1), (0, 3), (0, 4),
            (1, 2), (1, 5), (3, 2),
            (3, 7), (4, 5), (4, 7),
            (2, 6), (5, 6), (6, 7)
        ]:
            pt1 = tuple(map(int, np.round(coords_proj[i, start])))
            pt2 = tuple(map(int, np.round(coords_proj[i, end])))
            cv2.line(canvas, pt1, pt2, color or OBJECT_PALETTE_BGR.get(name, (0, 255, 0)), thickness, cv2.LINE_AA)

    # ========== 修复：删除或注释这行 ==========
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  # 删除或注释这行
    # ========== 修复结束 ==========
    
    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)  # 直接保存 BGR 格式

# def visualize_camera(
#     fpath: str,
#     image: np.ndarray,
#     *,
#     bboxes: Optional[LiDARInstance3DBoxes] = None,
#     labels: Optional[np.ndarray] = None,
#     lidar2camera_single: Optional[np.ndarray] = None,
#     camera_intrinsics_single: Optional[np.ndarray] = None,
#     img_aug_matrix_single: Optional[np.ndarray] = None,
#     classes: Optional[List[str]] = None,
#     color: Optional[Tuple[int, int, int]] = None,
#     thickness: float = 4,
# ) -> None:
#     canvas = image.copy()  
#     canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

#     if bboxes is None or len(bboxes) == 0:
#         mmcv.mkdir_or_exist(os.path.dirname(fpath))
#         mmcv.imwrite(canvas, fpath)
#         return
#     corners = bboxes.corners  # (N, 8, 3)
#     num_bboxes = corners.shape[0]

#     coords = np.concatenate([corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1)  # (N*8,4)

#     # LiDAR -> Camera
#     lidar2camera_single = copy.deepcopy(lidar2camera_single).reshape(4, 4)

#     coords = coords @ lidar2camera_single.T
#     coords = coords.reshape(-1, 8, 4)

#     # 过滤：只要 bbox 至少有一个角点在相机前就保留
#     mask_front = np.any(coords[..., 2] > 1e-5, axis=1)
#     coords = coords[mask_front]
#     if labels is not None:
#         labels = labels[mask_front]

#     if len(coords) == 0:
#         mmcv.mkdir_or_exist(os.path.dirname(fpath))
#         mmcv.imwrite(canvas, fpath)
#         return

#     # 按深度排序（近的后画）
#     sort_idx = np.argsort(-np.min(coords[..., 2], axis=1))
#     coords = coords[sort_idx]
#     if labels is not None:
#         labels = labels[sort_idx]

#     # 投影到相机像素
#     coords_reshape = coords.reshape(-1, 4)
#     coords_reshape[:, 2] = np.clip(coords_reshape[:, 2], 1e-5, 1e5)

#     K = copy.deepcopy(camera_intrinsics_single)
#     if K.shape == (4, 4):
#         K = K[:3, :3]
#     else:
#         K = K.reshape(3, 3)

#     coords_proj = (K @ coords_reshape[:, :3].T).T
#     coords_proj[:, 0] /= (coords_proj[:, 2] + 1e-5)
#     coords_proj[:, 1] /= (coords_proj[:, 2] + 1e-5)
#     coords_proj = coords_proj[:, :2].reshape(-1, 8, 2)

#     # 图像增强矩阵
#     if img_aug_matrix_single is not None:
#         coords_aug = []
#         for i in range(coords_proj.shape[0]):
#             pts = coords_proj[i]  # (8,2)
#             if img_aug_matrix_single.shape == (3, 3):
#                 pts_h = np.concatenate([pts, np.ones((8, 1))], axis=-1)
#                 pts_aug = (img_aug_matrix_single @ pts_h.T).T
#                 pts_aug[:, :2] /= pts_aug[:, 2:3]
#             elif img_aug_matrix_single.shape == (4, 4):
#                 pts_h = np.concatenate([pts, np.zeros((8, 1))], axis=-1)
#                 pts_aug = (img_aug_matrix_single[:3,:3] @ pts_h.T).T
#                 pts_aug += img_aug_matrix_single[:3, 3:4].T
#             else:
#                 raise ValueError(f"Unsupported img_aug_matrix_single shape: {img_aug_matrix_single.shape}")
#             coords_aug.append(pts_aug[:, :2])
#         coords_proj = np.stack(coords_aug, axis=0)
#     # breakpoint()

#     # # 过滤 NaN 或 inf
#     # valid_mask = np.all(np.isfinite(coords_proj), axis=(1, 2))
#     # coords_proj = coords_proj[valid_mask]
#     # H, W = 256, 704  # 图像尺寸
#     # breakpoint()
#     H, W = image.shape[0],image.shape[1]
#     # H, W = 512, 704  # 图像尺寸
#     # breakpoint()

#     # 过滤 NaN / inf
#     valid_mask = np.all(np.isfinite(coords_proj), axis=(1, 2))
#     coords_proj = coords_proj[valid_mask]
#     if labels is not None:
#         labels = labels[valid_mask]

#     # 过滤超出图像范围的 bbox（至少一个角点在图像内就保留）
#     in_range_mask = np.any(
#         (coords_proj[..., 0] >= 0) & (coords_proj[..., 0] < W) &
#         (coords_proj[..., 1] >= 0) & (coords_proj[..., 1] < H),
#         axis=1
#     )
#     coords_proj = coords_proj[in_range_mask]
#     if labels is not None:
#         labels = labels[in_range_mask]
    
#     # 画 3D bbox
#     for i in range(coords_proj.shape[0]):
#         name = classes[labels[i]] if classes is not None else "object"
#         for start, end in [
#             (0, 1), (0, 3), (0, 4),
#             (1, 2), (1, 5), (3, 2),
#             (3, 7), (4, 5), (4, 7),
#             (2, 6), (5, 6), (6, 7)
#         ]:
#             pt1 = tuple(map(int, np.round(coords_proj[i, start])))
#             pt2 = tuple(map(int, np.round(coords_proj[i, end])))
#             # breakpoint()
#             # print(f"pt1 (rounded): {pt1}, pt2 (rounded): {pt2}")  # 打印调试信息
#             cv2.line(canvas, pt1, pt2, color or OBJECT_PALETTE.get(name, (0, 255, 0)), thickness, cv2.LINE_AA)

#     canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
#     mmcv.mkdir_or_exist(os.path.dirname(fpath))
#     mmcv.imwrite(canvas, fpath)
