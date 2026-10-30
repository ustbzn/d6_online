import math

import numpy as np
from flask import Flask, request, jsonify
import sqlite3
import pyproj
import heapq
from shapely.geometry import LineString, Point
from shapely import wkt

from commonUtils import DBManager, load_hdmap_around_ego, wgs84_to_utm

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

def draw_hdmap_centered(hdmap, center_utm, figsize=(10, 10),
                        save_path=None):
    """
    hdmap: load_hdmap_around_ego() 返回
    center_utm: (cx, cy) = wgs84_to_utm(lon, lat)
    save_path: str, 如果不为 None 则保存图片到指定路径
    """

    cx, cy = center_utm
    plt.figure(figsize=figsize)

    # ========= Lane =========
    if 'lanes' in hdmap:
        for lane in hdmap["lanes"]:
            pts = np.array([[p.x - cx, p.y - cy] for p in lane.pointUtm])
            if len(pts) >= 2:
                plt.plot(pts[:, 0], pts[:, 1], color="blue", linewidth=1.5)

    # ========= LaneLink =========
    if 'lane_links' in hdmap:
        for link in hdmap["lane_links"]:
            pts = np.array([[p.x - cx, p.y - cy] for p in link.pointUtm])
            if len(pts) >= 2:
                plt.plot(pts[:, 0], pts[:, 1],
                        color="orange", linewidth=1.2, linestyle="--")

    # ========= LaneBoundary =========
    if 'lane_boundaries' in hdmap:
        for bd in hdmap["lane_boundaries"]:
            pts = np.array([[p.x - cx, p.y - cy] for p in bd.pointUtm])
            if len(pts) >= 2:
                plt.plot(pts[:, 0], pts[:, 1],
                        color="black", linewidth=1.0, alpha=0.7)

    # ========= Junction =========
    if 'junctions' in hdmap:
        for junc in hdmap["junctions"]:
            pts = np.array([[p.x - cx, p.y - cy] for p in junc.pointUtm])
            if len(pts) >= 3:
                plt.fill(pts[:, 0], pts[:, 1], color="red", alpha=0.2)

    # ========= Center (Ego) =========
    plt.scatter(0.0, 0.0, color="green", s=80, marker="x", label="Center")

    # ========= 图形设置 =========
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("HDMap centered at ego (0,0)")
    plt.grid(True, linestyle="--", alpha=0.3)

    # ========= 保存或显示 =========
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"HDMap saved to: {save_path}")
    else:
        plt.show()

    plt.close()
    
def utm_to_wgs84(position, zone=50, south=False):
    import pyproj
    proj = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84', south=south)
    x, y, z = position
    lon, lat = proj(x, y, inverse=True)
    return lon, lat


def densify_polyline(pts, step=0.1):
    """
    pts: (N, 2) ego-local 坐标
    step: 插值间距（建议 = voxel_size）
    """
    if len(pts) < 2:
        return pts

    dense_pts = [pts[0]]
    for i in range(1, len(pts)):
        p0 = pts[i - 1]
        p1 = pts[i]
        dist = np.linalg.norm(p1 - p0)

        if dist < step:
            dense_pts.append(p1)
            continue

        num = int(dist / step)
        for j in range(1, num + 1):
            dense_pts.append(p0 + (p1 - p0) * j / num)

    return np.array(dense_pts)

def densify_ego_points(ego_points, step=0.1):
    """
    只对 ego_points 里的点做插值
    不改坐标、不 rasterize
    """
    out = {}

    for k, polylines in ego_points.items():
        dense_list = []
        for pts in polylines:
            dense_pts = densify_polyline(pts, step=step)
            dense_list.append(dense_pts)
        out[k] = dense_list

    return out

from matplotlib.path import Path

def fill_polygon_bev(pts, bev_feat, channel, ego_range, voxel_size):
    x_min, y_min = ego_range[0], ego_range[1]
    vx, vy = voxel_size[0], voxel_size[1]

    rows = ((pts[:,0] - x_min) / vx).astype(int)
    cols = ((pts[:,1] - y_min) / vy).astype(int)
    poly = Path(np.stack([rows, cols], axis=1))

    H, W = bev_feat.shape[1:]
    for r in range(min(rows), max(rows)):
        for c in range(min(cols), max(cols)):
            if poly.contains_point((r, c)):
                bev_feat[channel, r, c] = 1.0
                
def pointUtm_list_to_numpy(pointUtm_list):
    """
    pointUtm_list: [Point(x, y), ...]
    return: (N, 2) np.array
    """
    return np.array([[p.x, p.y] for p in pointUtm_list],
                    dtype=np.float32)

def hdmap_data_to_numpy_raw(data):
    """
    data: load_hdmap_around_ego() 返回的 hdmap dict
    只做数据类型转换，不做任何坐标处理
    """
    hdmap_np = {
        'lanes': [],
        'lane_links': [],
        'lane_boundaries': [],
        'junctions': [],
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

    return hdmap_np

def translate_ego(data, pose_matrix):
    """
    data: hdmap_np (四类, 每类是 [np.array(N,2), ...])
    pose_matrix: (4,4) ego -> world
    return: 同结构，但点已经是 ego 坐标系
    """
    pose_inv = np.linalg.inv(pose_matrix)

    new_dict = {}

    for k, v in data.items():
        new_dict[k] = []

        for pts in v:
            # pts: (N, 2) world / UTM
            N = pts.shape[0]

            # 构造齐次坐标 (N, 4)
            pts_homo = np.zeros((N, 4), dtype=np.float32)
            pts_homo[:, 0] = pts[:, 0]
            pts_homo[:, 1] = pts[:, 1]
            pts_homo[:, 2] = 0.0
            pts_homo[:, 3] = 1.0

            # world -> ego
            pts_ego = (pose_inv @ pts_homo.T).T   # (N,4)

            # 只保留 x, y
            new_dict[k].append(pts_ego[:, :2])

    return new_dict



if __name__ == '__main__':
    visual = False
    # conn = sqlite3.connect("hm_beijing_3.3.2_20251231.sqlite")
    # cursor = conn.cursor()

    # cursor.execute("""
    # SELECT name FROM sqlite_master
    # WHERE type='table';
    # """)

    # tables = cursor.fetchall()
    # for t in tables:
    #     print(t[0])
    # # breakpoint()

    # utm_start_X: 432343.38092485664
    # utm_start_Y: 4396792.396297588
    
    import torch    
    import numpy as np

    device = 'cpu'  # 或 cuda
    
    pkl_path = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/train_bj_3d_d6_20260104_train_all_13k_for_occ.pkl'
    pkl_data = pkl.load(open(pkl_path,'rb'))['infos']
    
    for index, data in enumerate(pkl_data):
        pose = data['pose']
        data_path = data['data_path']
        calib_data = data['calib_data']
        
        npz_data = np.load(data_path, allow_pickle=True)
        
        conn = DBManager.initialize_database("hm_beijing_3.3.2_20251231.sqlite")
        
        lon, lat = utm_to_wgs84(position=pose[:,3][:3])
        # 经度: 116.6467321714981
        # 纬度: 40.1758831781218
        
        ego_range = [-81.6, -48, -1, 97.6, 48, -3]
        voxel_size = [0.1, 0.1, 0.2]

        # 宽度和高度
        width = ego_range[3] - ego_range[0]   # x_max - x_min
        height = ego_range[4] - ego_range[1]  # y_max - y_min

        # 半径 = 对角线的一半
        radius_m = int(np.sqrt(width**2 + height**2) / 2)
        ego_range_post = [-radius_m, -radius_m, radius_m, radius_m]
        
        # 加载 ego 附近 Lane
        # ego_lanes = load_lanes_around_ego(conn, lon, lat, ego_range_post)
        data = load_hdmap_around_ego(conn, lon, lat, ego_range_post)
        array_data = hdmap_data_to_numpy_raw(data)
        ego_points = translate_ego(array_data, pose)
        breakpoint()

        # print(f"裁剪后 Lane 数量: {len(ego_lanes)}")
            
        if visual:
            center_utm = wgs84_to_utm(lon, lat)
            draw_hdmap_centered(
                data,
                center_utm,
                save_path="/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/extract_hdmap_featrue/hd_map_{}.png".format(index)
            )
        
        # hdmap_bevfeatrue = torch.tensor(([1,4,1792, 960])) # b c h w
        
        C, H, W = 4, 1792, 960
        hdmap_bevfeature = torch.zeros((C, H, W), dtype=torch.float32, device=device)
    
        for k, v in data.items():
            breakpoint()