# ==============================
# Imports
# ==============================
import math
import numpy as np
import torch
import sqlite3
import pyproj
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import cv2
from PIL import Image



from commonUtils import (
    DBManager,
    load_hdmap_around_ego,
    wgs84_to_utm,
)

# ==============================
# Channel definition
# ==============================
CH = {
    'lanes': 0,
    'lane_links': 1,
    'lane_boundaries': 2,
    'junctions': 3,
    # 'greenbelts': 4,
}

# ==============================
# Utils
# ==============================
def utm_to_wgs84(position, zone=50, south=False):
    proj = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84', south=south)
    x, y, z = position
    lon, lat = proj(x, y, inverse=True)
    return lon, lat


def pointUtm_list_to_numpy(pointUtm_list):
    return np.array([[p.x, p.y] for p in pointUtm_list], dtype=np.float32)


def hdmap_data_to_numpy_raw(data):
    hdmap_np = {
        'lanes': [],
        'lane_links': [],
        'lane_boundaries': [],
        'junctions': [],
        # 'greenbelts': [],
    }

    if 'lanes' in data:
        for lane in data['lanes']:
            pts = pointUtm_list_to_numpy(lane.pointUtm)
            if len(pts) >= 2:
                hdmap_np['lanes'].append(pts)

    if 'lane_links' in data:
        for link in data['lane_links']:
            pts = pointUtm_list_to_numpy(link.pointUtm)
            if len(pts) >= 2:
                hdmap_np['lane_links'].append(pts)

    if 'lane_boundaries' in data:
        for bd in data['lane_boundaries']:
            pts = pointUtm_list_to_numpy(bd.pointUtm)
            if len(pts) >= 2:
                hdmap_np['lane_boundaries'].append(pts)

    if 'junctions' in data:
        for junc in data['junctions']:
            pts = pointUtm_list_to_numpy(junc.pointUtm)
            if len(pts) >= 3:
                hdmap_np['junctions'].append(pts)
                
    # if 'greenbelts' in data:
    #     for gb in data['greenbelts']:
    #         pts = pointUtm_list_to_numpy(gb.pointUtm)
    #         if len(pts) >= 3:
    #             hdmap_np['greenbelts'].append(pts)

    for k, v in hdmap_np.items():
        print(f'{k}: {len(v)}')
        
    return hdmap_np


def translate_ego(data, pose_matrix):
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
            out[k].append(pts_ego[:, :2])

    return out


# ==============================
# Geometry processing
# ==============================
def densify_polyline(pts, step=0.1):
    if len(pts) < 2:
        return pts.astype(np.float32)

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

    return np.asarray(dense, dtype=np.float32)


# def densify_ego_points(ego_points, step=0.1):
#     out = {}
#     for k, polylines in ego_points.items():
#         out[k] = []
#         for pts in polylines:
#             if k == 'junctions':
#                 out[k].append(pts.astype(np.float32))
#             else:
#                 out[k].append(densify_polyline(pts, step))
#     return out

def densify_ego_points(ego_points, step=0.1):
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
                    out[k].append(densify_polyline(pts, step))

        elif k in POLYGON_KEYS:
            for pts in geometries:
                if len(pts) >= 3:
                    out[k].append(pts.astype(np.float32))

        else:
            print(geometries, 'no exsists!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return out

# ==============================
# Rasterization
# ==============================
def points_to_bev_indices(pts, ego_range, voxel_size):
    x_min, y_min = ego_range[0], ego_range[1]
    x_max, y_max = ego_range[3], ego_range[4]
    vx, vy = voxel_size[0], voxel_size[1]

    mask = (
        (pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
        (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
    )
    pts = pts[mask]
    if len(pts) == 0:
        return None, None

    rows = ((pts[:, 0] - x_min) / vx).astype(np.int32)
    cols = ((pts[:, 1] - y_min) / vy).astype(np.int32)
    return rows, cols


def fill_polygon_bev(pts, bev_feat, channel, ego_range, voxel_size):
    """
    pts: (N, 2) array, [x, y] in ego / UTM coord
    bev_feat: [C, H, W]
    """

    x_min, y_min = ego_range[0], ego_range[1]
    vx, vy = voxel_size[0], voxel_size[1]
    _, H, W = bev_feat.shape

    # x -> H (row), y -> W (col)
    rows = (pts[:, 0] - x_min) / vx   # x
    cols = (pts[:, 1] - y_min) / vy   # y

    poly = Path(np.stack([rows, cols], axis=1))

    r0 = max(int(np.floor(rows.min())), 0)
    r1 = min(int(np.ceil(rows.max())) + 1, H)
    c0 = max(int(np.floor(cols.min())), 0)
    c1 = min(int(np.ceil(cols.max())) + 1, W)

    for r in range(r0, r1):
        for c in range(c0, c1):
            if poly.contains_point((r, c)):
                bev_feat[channel, r, c] = 1.0
                
from skimage.draw import polygon_perimeter

def draw_polygon_boundary_bev(pts, bev_feat, channel, ego_range, voxel_size):
    """
    只画 polygon 的边界到 BEV tensor
    pts: (N,2) numpy array, UTM/ego 坐标
    bev_feat: [C,H,W] tensor/array
    channel: 要绘制的 channel
    ego_range: [x_min, y_min, x_max, y_max] 米
    voxel_size: [vx, vy] 每个像素对应的米
    """
    x_min, y_min = ego_range[0], ego_range[1]
    vx, vy = voxel_size[0], voxel_size[1]
    _, H, W = bev_feat.shape

    rows = (pts[:,0] - x_min) / vx
    cols = (pts[:,1] - y_min) / vy

    rows = np.clip(rows, 0, H-1)
    cols = np.clip(cols, 0, W-1)

    rr, cc = polygon_perimeter(rows, cols, shape=(H,W))
    bev_feat[channel, rr, cc] = 1.0
                
def save_bev_rgb_png(
    bev_feat,
    save_path
):
    """
    bev_feat: (4, H, W)
    """
    if isinstance(bev_feat, torch.Tensor):
        bev = bev_feat.cpu().numpy()
    else:
        bev = bev_feat

    H, W = bev.shape[1:]

    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # R: junction
    rgb[..., 0] = bev[3]

    # G: lane_link
    rgb[..., 1] = bev[1]

    # B: lane_boundary
    rgb[..., 2] = bev[2]

    # lane -> white (加到 RGB)
    rgb += bev[0][..., None]

    rgb = np.clip(rgb, 0, 1)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        rgb.transpose(1, 0, 2),  # (W,H,3) -> (H,W,3)
        origin='lower'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def save_bev_channel_png(
    bev_feat,
    channel,
    save_path,
    title=None
):
    """
    bev_feat: (C, H, W) torch.Tensor or np.ndarray
    channel: int
    """
    if isinstance(bev_feat, torch.Tensor):
        img = bev_feat[channel].cpu().numpy()
    else:
        img = bev_feat[channel]

    plt.figure(figsize=(8, 6))
    plt.imshow(
        img.T,           # 关键：转置
        origin='lower',
        cmap='gray'
    )
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    
def project_bev_nonzero_to_image(
    bev_feat,        # torch.Tensor [H, W] or [C, H, W]
    ego_range,       # [-81.6, -48.0, -1.0, 97.6, 48.0, -3.0]
    voxel_size,      # [0.1, 0.1, 0.2]
    T_cam_ego,       # 4x4 ego -> camera (numpy)
    K,               # 3x3 camera intrinsic (numpy)
    image,           # HxWx3 (numpy, BGR)
    z_plane=0.0,
    min_depth=0.1
):
    """
    BEV 非 0 网格 -> 图像投影
    Returns:
        image_vis: 画好点的图像
        uv: [N, 2] 投影像素坐标
    """

    # ----------------------------------
    # 1. 非 0 mask（✅ torch 正确写法）
    # ----------------------------------
    if not isinstance(bev_feat, torch.Tensor):
        raise TypeError(f'bev_feat must be torch.Tensor, got {type(bev_feat)}')
    # breakpoint()
    if bev_feat.dim() == 3:
        # [C, H, W] -> [H, W]
        mask = torch.any(bev_feat != 0, dim=0)
    else:
        mask = bev_feat != 0

    idx = torch.nonzero(mask, as_tuple=False)  # [N, 2] (row, col)

    if idx.numel() == 0:
        return image.copy(), np.zeros((0, 2), dtype=np.int32)

    # 转 numpy（后面全是 numpy 几何）
    rows = idx[:, 0].cpu().numpy()
    cols = idx[:, 1].cpu().numpy()

    # ----------------------------------
    # 2. BEV index -> ego 坐标
    # ----------------------------------
    x_min, y_min = ego_range[0], ego_range[1]
    vx, vy = voxel_size[0], voxel_size[1]

    # col -> x (forward), row -> y (left)
    x_ego = x_min + (rows + 0.5) * vx
    y_ego = y_min + (cols + 0.5) * vy
    z_ego = np.full_like(x_ego, z_plane)

    pts_ego = np.stack(
        [x_ego, y_ego, z_ego, np.ones_like(x_ego)],
        axis=1
    )  # [N, 4]

    # ----------------------------------
    # 3. ego -> camera
    # ----------------------------------
    pts_cam = (T_cam_ego @ pts_ego.T).T
    depth = pts_cam[:, 2]

    valid = depth > min_depth
    pts_cam = pts_cam[valid]

    if pts_cam.shape[0] == 0:
        return image.copy(), np.zeros((0, 2), dtype=np.int32)

    # ----------------------------------
    # 4. camera -> image
    # ----------------------------------
    xyz = pts_cam[:, :3]
    uvw = (K @ xyz.T).T

    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]

    # ----------------------------------
    # 5. 图像范围过滤
    # ----------------------------------
    H_img, W_img = image.shape[:2]
    mask_img = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)

    u = u[mask_img].astype(np.int32)
    v = v[mask_img].astype(np.int32)

    uv = np.stack([u, v], axis=1)

    # ----------------------------------
    # 6. 可视化
    # ----------------------------------
    image_vis = image.copy()
    for ui, vi in uv:
        cv2.circle(image_vis, (ui, vi), 1, (0, 0, 255), -1)

    return image_vis, uv


# ==============================
# Main
# ==============================
if __name__ == '__main__':
    device = 'cpu'

    pkl_path = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/train_bj_3d_d6_20260104_train_all_13k_for_occ.pkl'
    infos = pkl.load(open(pkl_path, 'rb'))['infos'][::100]

    conn = DBManager.initialize_database(
        "hm_beijing_3.3.2_20251231.sqlite"
    )

    ego_range = [-81.6, -48.0, -1.0, 97.6, 48.0, -3.0]
    voxel_size = [0.1, 0.1, 0.2]

    x_min, y_min, z_min, x_max, y_max, z_max = ego_range
    vx, vy, vz = voxel_size

    H = int(round((x_max - x_min) / vx))  # 1792
    W = int(round((y_max - y_min) / vy))  # 960
    # W = int(round(ego_range[4] - ego_range[1]) / voxel_size[1])
    # H = int(round(ego_range[3] - ego_range[0]) / voxel_size[0])
    for idx, info in enumerate(infos):
        pose = info['pose']
        extrinsic = info['calib_data']['120_front']['extrinsic']
        intrinsic = info['calib_data']['120_front']['intrinsic']
        sensor2lidar_rotation = info['calib_data']['120_front']['sensor2lidar_rotation']
        sensor2lidar_translation = info['calib_data']['120_front']['sensor2lidar_translation']
        cam_token = info['calib_data']['120_front']['cam_token']
        
        lon, lat = utm_to_wgs84(position=pose[:, 3][:3])

        # -------- load hdmap --------
        width = ego_range[3] - ego_range[0]
        height = ego_range[4] - ego_range[1]
        radius_m = int(np.sqrt(width ** 2 + height ** 2) / 2)
        ego_range_post = [-radius_m, -radius_m, radius_m, radius_m]

        hdmap = load_hdmap_around_ego(
            conn, lon, lat, ego_range_post
        )

        # -------- processing --------
        hdmap_np = hdmap_data_to_numpy_raw(hdmap)
        ego_points = translate_ego(hdmap_np, pose)
        ego_points = densify_ego_points(
            ego_points, step=voxel_size[0]
        )
        # -------- BEV --------
        hdmap_bev = torch.zeros(
            (4, H, W),
            dtype=torch.float32,
            device=device
        )

        for k, polylines in ego_points.items():
            ch = CH[k]

            if k in ['lanes', 'lane_links', 'lane_boundaries']:
                for pts in polylines:
                    rows, cols = points_to_bev_indices(
                        pts, ego_range, voxel_size
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
                            ego_range,
                            voxel_size
                        )
                        # fill_polygon_bev(
                        #     poly,
                        #     hdmap_bev,
                        #     ch,
                        #     ego_range,
                        #     voxel_size
                        # )

        print(f'Frame {idx} HDMap BEV sum:',
              hdmap_bev.sum(dim=(1, 2)))
        
        import os
        out_dir = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/extract_hdmap_featrue/tensor_vis/'
        out_dir_img = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/extract_hdmap_featrue/img_vis/'

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir_img, exist_ok=True)
        

        frame_id = idx

        # # 单通道
        # save_bev_channel_png(
        #     hdmap_bev,
        #     CH['lanes'],
        #     f'{out_dir}/frame_{frame_id}_lane.png',
        #     title='Lane'
        # )

        # save_bev_channel_png(
        #     hdmap_bev,
        #     CH['lane_links'],
        #     f'{out_dir}/frame_{frame_id}_lane_link.png',
        #     title='Lane Link'
        # )

        # save_bev_channel_png(
        #     hdmap_bev,
        #     CH['lane_boundaries'],
        #     f'{out_dir}/frame_{frame_id}_lane_boundary.png',
        #     title='Lane Boundary'
        # )

        # save_bev_channel_png(
        #     hdmap_bev,
        #     CH['junctions'],
        #     f'{out_dir}/frame_{frame_id}_junction.png',
        #     title='Junction'
        # )

        # RGB 叠加
        save_bev_rgb_png(
            hdmap_bev,
            f'{out_dir}/frame_{frame_id}_hdmap_rgb.png'
        )
        
        T_inv = np.linalg.inv(extrinsic)

        image = np.array(Image.open(cam_token).convert('RGB'))
        # image = 
        image_vis, uv = project_bev_nonzero_to_image(
        bev_feat=hdmap_bev,
        ego_range=ego_range,
        voxel_size=voxel_size,
        T_cam_ego=T_inv,
        K=intrinsic,
        image=image,
        z_plane=0.0
        )

        save_path = os.path.join(
            out_dir_img,
            f'{frame_id}_{frame_id}_bev_proj.png'
        )

        cv2.imwrite(save_path, image_vis)
        print(f'Saved: {save_path}, num_pts={len(uv)}')
