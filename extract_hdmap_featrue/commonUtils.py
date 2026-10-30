import sqlite3
import pyproj
from shapely.geometry import LineString, Point
from shapely import wkt

import os
import numpy as np
import cv2
import torch
# from skimage.draw import polygon_perimeter
from skimage.draw import polygon
from PIL import Image

class HDPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Link:
    def __init__(self, fromRoadPkid, fromLaneId, toRoadPkid, toLaneId, points84, pointUtm, geom):
        self.fromRoadPkid = fromRoadPkid
        self.fromLaneId = fromLaneId
        self.toRoadPkid = toRoadPkid
        self.toLaneId = toLaneId
        self.geom = geom
        self.points84 = points84
        self.pointUtm = pointUtm

class Lane:
    def __init__(self, roadPkid, laneId, points84, pointUtm, geom):
        self.roadPkid = roadPkid
        self.laneId = laneId
        self.geom = geom
        self.points84 = points84
        self.pointUtm = pointUtm
        
class LaneBoundary:
    def __init__(self, roadPkid, points84, pointUtm, geom):
        self.roadPkid = roadPkid
        self.geom = geom
        self.points84 = points84
        self.pointUtm = pointUtm
        
class Junction:
    def __init__(self, pkid, points84, pointUtm, geom):
        self.pkid = pkid
        self.geom = geom            # Polygon
        self.points84 = points84    # exterior ring
        self.pointUtm = pointUtm


zone_transform = {}

def utm_to_wgs84(position, zone=50, south=False):
    proj = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84', south=south)
    x, y, z = position
    lon, lat = proj(x, y, inverse=True)
    return lon, lat

def wgs84_to_utm(wgs_x, wgs_y):
    zone_number = int((wgs_x + 186.0) / 6.0)
    if zone_number in zone_transform:
        transformer = zone_transform[zone_number]
    else:
        utm_params = {
            'proj': 'utm',
            'zone': zone_number,
            'datum': 'WGS84',
            'ellps': 'WGS84',
            'units': 'm',
            'no_defs': True
        }
        transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_params, always_xy=True)
        zone_transform[zone_number] = transformer
    # 将WGS84坐标转换为UTM坐标
    utm_x, utm_y = transformer.transform(wgs_x, wgs_y)
    return utm_x, utm_y


class DBManager:
    @staticmethod
    def initialize_database(db_path, load_extension=True):
        conn = sqlite3.connect(db_path)
        if load_extension:
            conn.enable_load_extension(True)
            conn.load_extension("mod_spatialite")

        # 执行PRAGMA配置
        DBManager.exec_sql(conn, "PRAGMA foreign_keys = ON;")
        DBManager.exec_sql(conn, "PRAGMA page_size = 8192;")
        DBManager.exec_sql(conn, "PRAGMA cache_size = 82000;")
        DBManager.exec_sql(conn, "PRAGMA temp_store = memory;")
        DBManager.exec_sql(conn, "PRAGMA synchronous = off;")
        DBManager.exec_sql(conn, "PRAGMA locking_mode = normal ;")
        DBManager.exec_sql(conn, "PRAGMA journal_mode = off;")
        DBManager.exec_sql(conn, "PRAGMA auto_vacuum = 0;")

        # 检查并初始化空间元数据
        if not DBManager.table_exists(conn, "spatial_ref_sys"):
            DBManager.exec_sql(conn, "SELECT InitSpatialMetadata(1);")

        DBManager.exec_sql(conn, "PRAGMA encoding = 'UTF-8';")
        return conn

    @staticmethod
    def exec_sql(connection, sql):
        cursor = connection.cursor()
        cursor.execute(sql)
        connection.commit()

    @staticmethod
    def table_exists(db_ptr, table_name):
        cursor = db_ptr.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cursor.fetchone() is not None

    @staticmethod
    def select_geometry(conn, sql, object_type):
        objects = []
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                if object_type == 'Lane':
                    roadPkid, laneId, row_geom = row
                elif object_type == 'Link':
                    pkid, fromRoadPkid, fromLaneId, toRoadPkid, toLaneId, row_geom = row

                if row_geom:
                    geom = wkt.loads(row_geom)
                else:
                    geom = LineString()
                points = list(geom.coords)
                points84 = [HDPoint(x, y, z) for x, y, z in points]
                pointsUtm = [HDPoint(*wgs84_to_utm(x, y), z) for x, y, z in points]

                if object_type == 'Lane':
                    obj = Lane(roadPkid, laneId, points84, pointsUtm, geom)
                elif object_type == 'Link':
                    obj = Link(pkid, fromRoadPkid, fromLaneId, toRoadPkid, toLaneId, points84, pointsUtm, geom)

                objects.append(obj)
        except sqlite3.Error as e:
            print(f"Error: {e}")
        return objects
    
    @staticmethod
    def select_geometry_from_rows(rows, object_type):
        objects = []
        for row in rows:
            if object_type == 'Lane':
                roadPkid, laneId, row_geom = row
            elif object_type == 'Link':
                pkid, fromRoadPkid, fromLaneId, toRoadPkid, toLaneId, row_geom = row

            if row_geom:
                geom = wkt.loads(row_geom)
            else:
                geom = LineString()
            points = list(geom.coords)
            points84 = [HDPoint(x, y, z) for x, y, z in points]
            pointsUtm = [HDPoint(*wgs84_to_utm(x, y), z) for x, y, z in points]

            if object_type == 'Lane':
                obj = Lane(roadPkid, laneId, points84, pointsUtm, geom)
            elif object_type == 'Link':
                obj = Link(pkid, fromRoadPkid, fromLaneId, toRoadPkid, toLaneId, points84, pointsUtm, geom)

            objects.append(obj)
        return objects


    @staticmethod
    def select_div(conn, sql):
        objects = []
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                roadId, row_geom = row

                if row_geom:
                    geom = wkt.loads(row_geom)
                else:
                    geom = LineString()
                points = list(geom.coords)
                points84 = [HDPoint(x, y, z) for x, y, z in points]
                print(points84)


        except sqlite3.Error as e:
            print(f"Error: {e}")
        return objects
    
from shapely import wkt
from shapely.geometry import LineString
from shapely.geometry import Polygon

def load_lanes_around_ego(conn, lon, lat, ego_range=[-81.6, -48, 97.6, 48]):
    """
    conn: Spatialite 连接
    lon, lat: 自车 WGS84 坐标（度）
    ego_range: ego 周围裁剪范围（米）
    """
    # 如果想用米裁剪，需要先近似把米转换成经纬度
    # 简单近似：1度经度 ≈ 111320米，1度纬度 ≈ 110540米（北京附近）
    deg_per_meter_lon = 1 / 111320.0
    deg_per_meter_lat = 1 / 110540.0

    x_min, y_min, x_max, y_max = ego_range
    lon_min = lon + x_min * deg_per_meter_lon
    lon_max = lon + x_max * deg_per_meter_lon
    lat_min = lat + y_min * deg_per_meter_lat
    lat_max = lat + y_max * deg_per_meter_lat

    # SQL 裁剪
    lane_sql = """
    SELECT road_pkid, lane_id, AsText(geom)
    FROM TX_LANE
    WHERE ST_Intersects(
        geom,
        BuildMBR(?, ?, ?, ?, 4326)
    )
    """
    cursor = conn.cursor()
    cursor.execute(lane_sql, (lon_min, lat_min, lon_max, lat_max))
    rows = cursor.fetchall()

    # 转 Lane 对象
    lanes = []
    for row in rows:
        roadPkid, laneId, row_geom = row
        # geom = wkt.loads(row_geom) if row_geom else LineString()
        if row_geom:
            geom = wkt.loads(row_geom)
        else:
            geom = LineString()
        points84 = [HDPoint(x, y, 0.0) for x, y, *rest in geom.coords]
        # 经纬度直接转成 UTM
        pointsUtm = [HDPoint(*wgs84_to_utm(p.x, p.y), p.z) for p in points84]
        lane = Lane(roadPkid, laneId, points84, pointsUtm, geom)
        lanes.append(lane)

    return lanes


def load_lane_links_around_ego(conn, lon, lat, ego_range=[-81.6, -48, 97.6, 48]):
    deg_per_meter_lon = 1 / 111320.0
    deg_per_meter_lat = 1 / 110540.0

    x_min, y_min, x_max, y_max = ego_range
    lon_min = lon + x_min * deg_per_meter_lon
    lon_max = lon + x_max * deg_per_meter_lon
    lat_min = lat + y_min * deg_per_meter_lat
    lat_max = lat + y_max * deg_per_meter_lat

    link_sql = """
    SELECT
        from_road_pkid,
        from_lane_id,
        to_road_pkid,
        to_lane_id,
        AsText(geom)
    FROM TX_LANE_LINK
    WHERE ST_Intersects(
        geom,
        BuildMBR(?, ?, ?, ?, 4326)
    )
    """

    cursor = conn.cursor()
    cursor.execute(link_sql, (lon_min, lat_min, lon_max, lat_max))
    rows = cursor.fetchall()

    links = []
    for row in rows:
        fr_road, fr_lane, to_road, to_lane, row_geom = row
        geom = wkt.loads(row_geom) if row_geom else LineString()

        points84 = [HDPoint(x, y, 0.0) for x, y, *rest in geom.coords]
        pointsUtm = [HDPoint(*wgs84_to_utm(p.x, p.y), p.z) for p in points84]
        # class Link:
        #     def __init__(self, pkid, fromRoadPkid, fromLaneId, toRoadPkid, toLaneId, points84, pointUtm, geom):
        #         self.fromRoadPkid = fromRoadPkid
        #         self.fromLaneId = fromLaneId
        #         self.toRoadPkid = toRoadPkid
        #         self.toLaneId = toLaneId
        #         self.geom = geom
        #         self.points84 = points84
        #         self.pointUtm = pointUtm
        link = Link(
            fr_road, fr_lane,
            to_road, to_lane,
            points84, pointsUtm, geom,
        )
        links.append(link)

    return links

def load_lane_boundaries_around_ego(conn, lon, lat, ego_range=[-81.6, -48, 97.6, 48]):
    deg_per_meter_lon = 1 / 111320.0
    deg_per_meter_lat = 1 / 110540.0

    x_min, y_min, x_max, y_max = ego_range
    lon_min = lon + x_min * deg_per_meter_lon
    lon_max = lon + x_max * deg_per_meter_lon
    lat_min = lat + y_min * deg_per_meter_lat
    lat_max = lat + y_max * deg_per_meter_lat

    boundary_sql = """
    SELECT
        pkid,
        AsText(geom)
    FROM TX_LANE_BOUNDARY
    WHERE ST_Intersects(
        geom,
        BuildMBR(?, ?, ?, ?, 4326)
    )
    """
    cursor = conn.cursor()
    cursor.execute(boundary_sql, (lon_min, lat_min, lon_max, lat_max))
    rows = cursor.fetchall()

    boundaries = []
    for row in rows:
        roadPkid, row_geom = row
        geom = wkt.loads(row_geom) if row_geom else LineString()

        points84 = [HDPoint(x, y, 0.0) for x, y, *rest in geom.coords]
        pointsUtm = [HDPoint(*wgs84_to_utm(p.x, p.y), p.z) for p in points84]
        
        # class LaneBoundary:
        #     def __init__(self, pkid, roadPkid, laneId, boundaryType,
        #                 points84, pointUtm, geom):
        #         self.pkid = pkid
        #         self.roadPkid = roadPkid
        #         self.laneId = laneId
        #         self.boundaryType = boundaryType  # 实线 / 虚线 / 路缘等
        #         self.geom = geom
        #         self.points84 = points84
        #         self.pointUtm = pointUtm

        boundary = LaneBoundary(
            roadPkid, points84, pointsUtm, geom
        )
        boundaries.append(boundary)

    return boundaries

def load_junctions_around_ego(conn, lon, lat, ego_range=[-81.6, -48, 97.6, 48]):
    deg_per_meter_lon = 1 / 111320.0
    deg_per_meter_lat = 1 / 110540.0

    x_min, y_min, x_max, y_max = ego_range
    lon_min = lon + x_min * deg_per_meter_lon
    lon_max = lon + x_max * deg_per_meter_lon
    lat_min = lat + y_min * deg_per_meter_lat
    lat_max = lat + y_max * deg_per_meter_lat
    
    junction_sql = """
    SELECT
        pkid,
        AsText(geom)
    FROM TX_JUNCTION
    WHERE ST_Intersects(
        geom,
        BuildMBR(?, ?, ?, ?, 4326)
    )
    """

    cursor = conn.cursor()
    cursor.execute(junction_sql, (lon_min, lat_min, lon_max, lat_max))
    rows = cursor.fetchall()

    junctions = []
    for row in rows:
        pkid, row_geom = row
        geom = wkt.loads(row_geom) if row_geom else Polygon()

        points84 = [HDPoint(x, y, 0.0) for x, y in geom.exterior.coords]
        pointsUtm = [HDPoint(*wgs84_to_utm(p.x, p.y), p.z) for p in points84]
        
        # class Junction:
        #     def __init__(self, pkid, points84, pointUtm, geom):
        #         self.pkid = pkid
        #         self.geom = geom            # Polygon
        #         self.points84 = points84    # exterior ring
        #         self.pointUtm = pointUtm

        junction = Junction(pkid, points84, pointsUtm, geom)
        junctions.append(junction)

    return junctions
 
def load_hdmap_around_ego(conn, lon, lat, ego_range):
    return {
        "lanes": load_lanes_around_ego(conn, lon, lat, ego_range),
        "lane_links": load_lane_links_around_ego(conn, lon, lat, ego_range),
        "lane_boundaries": load_lane_boundaries_around_ego(conn, lon, lat, ego_range),
        "junctions": load_junctions_around_ego(conn, lon, lat, ego_range),
        # "roads": load_roads_around_ego(conn, lon, lat, ego_range),
        # "greenbelts": load_greenbelts_around_ego(conn, lon, lat, ego_range),
        # "safetyislands": load_safetyislands_around_ego(conn, lon, lat, ego_range),
    }
    
def save_bev_rgb_png(bev_feat, save_path):
    """
    bev_feat: (4, H, W)
    """
    import matplotlib.pyplot as plt
    
    if isinstance(bev_feat, torch.Tensor):
        bev = bev_feat.cpu().numpy()
    else:
        bev = bev_feat

    H, W = bev.shape[1:]

    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # R: junction
    # G: lane_link
    # B: lane_boundary
    rgb[..., 0] = bev[3]
    rgb[..., 1] = bev[1]
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
    if not isinstance(bev_feat, torch.Tensor):
        raise TypeError(f'bev_feat must be torch.Tensor, got {type(bev_feat)}')
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

def visual_map_featrue(hdmap_bev, extrinsic,
    cam_token, ego_range, voxel_size, intrinsic):
    
    out_dir = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/extract_hdmap_featrue/tensor_vis/'
    out_dir_img = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/extract_hdmap_featrue/img_vis/'

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_img, exist_ok=True)

    # RGB 叠加
    save_bev_rgb_png(
        hdmap_bev,
        f'{out_dir}/frame_hdmap_rgb.png'
    )
    
    T_inv = np.linalg.inv(extrinsic)

    image = np.array(Image.open(cam_token).convert('RGB'))
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
        f'bev_proj.png'
    )

    cv2.imwrite(save_path, image_vis)
    print(f'Saved: {save_path}, num_pts={len(uv)}')

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

    # rows = ((pts[:, 0] - x_min) / vx).astype(np.int32)
    # cols = ((pts[:, 1] - y_min) / vy).astype(np.int32)
    
    rows = np.clip(((pts[:,0] - x_min) / vx).astype(np.int32), 0, 1792-1)
    cols = np.clip(((pts[:,1] - y_min) / vy).astype(np.int32), 0, 960-1)
    return rows, cols

def draw_polygon_boundary_bev(pts, bev_feat, channel, ego_range, voxel_size):
    x_min, y_min = ego_range[0], ego_range[1]
    vx, vy = voxel_size[0], voxel_size[1]
    _, H, W = bev_feat.shape

    rows = (pts[:,0] - x_min) / vx
    cols = (pts[:,1] - y_min) / vy

    rows = np.clip(rows, 0, H-1)
    cols = np.clip(cols, 0, W-1)

    # rr, cc = polygon_perimeter(rows, cols, shape=(H,W))
    # bev_feat[channel, rr, cc] = 1.0
    rr, cc = polygon(rows, cols, shape=(H, W))
    bev_feat[channel, rr, cc] = 1.0

# def draw_polygon_boundary_bev(pts, bev_feat, channel, ego_range, voxel_size):
#     """
#     pts: (N,2) BEV 坐标
#     bev_feat: [C,H,W]
#     channel: int
#     ego_range: [x_min, y_min, x_max, y_max]
#     voxel_size: [vx, vy]
#     """
#     x_min, y_min = ego_range[0], ego_range[1]
#     vx, vy = voxel_size[0], voxel_size[1]
#     _, H, W = bev_feat.shape

#     rows = np.floor((pts[:,0] - x_min) / vx).astype(np.int32)
#     cols = np.floor((pts[:,1] - y_min) / vy).astype(np.int32)

#     rows = np.clip(rows, 0, H-1)
#     cols = np.clip(cols, 0, W-1)
    
#     if len(rows) < 3 or len(cols) < 3:
#         return
#     if (rows.max() < 0 or rows.min() >= H or cols.max() < 0 or cols.min() >= W):
#         return
#     try:
#         # rr, cc = polygon_perimeter(rows, cols, shape=(H,W))
#         # bev_feat[channel, rr, cc] = 1.0
#         rr, cc = polygon(rows, cols, shape=(H, W))
#         bev_feat[channel, rr, cc] = 1.0
#     except:
#         breakpoint()


