import math

import numpy as np
from flask import Flask, request, jsonify
import sqlite3
import pyproj
import heapq
from shapely.geometry import LineString, Point
from shapely import wkt

from commonUtils import DBManager, load_lanes_around_ego, load_hdmap_around_ego, wgs84_to_utm
app = Flask(__name__)


class Node:
    def __init__(self, lane, g=0, h=0, parent=None, link=None, transition_geom=None, is_transition=False):
        self.lane = lane
        self.g = g  # 从起点到该节点的实际代价
        self.h = h  # 从该节点到终点的启发式代价
        self.parent = parent
        self.link = link  # 当前节点和父节点之间的Link几何图形
        self.transition_geom = transition_geom
        self.is_transition = is_transition

    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f() < other.f()


links = []
lanes = []
lane_dict = {}
link_dict = {}


# def heuristic(a, b):
#     # a 和 b 都是坐标元组，因此需要将它们转换为 Point 对象
#     point_a = Point(a)
#     point_b = Point(b)
#     return point_a.distance(point_b)

def heuristic(a, b):
    # 计算曼哈顿距离
    return abs(a.x - b.x) + abs(a.y - b.y)


def create_transition_geom(start_lane, end_lane, link_dict, transition_distance=5):
    # 选取当前 lane 的起点
    start_point = Point(start_lane.geom.coords[0])

    # 选取临近 lane 或其后续 lane 上合适的终点
    end_point = find_transition_end_point(end_lane, transition_distance, link_dict)

    # 生成贝塞尔曲线
    control_point = Point((start_point.x + end_point.x) / 2, (start_point.y + end_point.y) / 2)
    bezier_points = [start_point, control_point, end_point]
    bezier_line = bezier_curve(bezier_points)

    return LineString(bezier_line)


def find_transition_end_point(lane, distance, link_dict):
    # 如果当前 lane 的长度大于等于指定距离，返回该点
    if lane.geom.length >= distance:
        return lane.geom.interpolate(distance)
    else:
        # 找到下一个 lane
        next_lane = find_next_lane(lane, link_dict)
        if next_lane:
            # 计算从起点到下一个 lane 的过渡点
            remaining_distance = distance - lane.geom.length
            return next_lane.geom.interpolate(remaining_distance)
        else:
            # 如果没有下一个 lane，返回当前 lane 的终点
            return lane.geom.interpolate(lane.geom.length)


def find_next_lane(current_lane, link_dict):
    key = (current_lane.roadPkid, current_lane.laneId)
    links = link_dict.get(key, [])
    for link in links:
        next_lane = lane_dict.get((link.toRoadPkid, link.toLaneId))
        if next_lane and link.geom.length == 0:  # 确保 link 的长度为 0
            return next_lane
    return None


def bezier_curve(points, n=100):
    return np.array([bezier_interpolation(t, points) for t in np.linspace(0, 1, n)])


def bezier_interpolation(t, points):
    n = len(points) - 1
    return sum(comb(n, i) * (1 - t) ** (n - i) * t ** i * np.array([p.x, p.y]) for i, p in enumerate(points))


def comb(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def find_path(lane_dict, start_lane, end_lane, links):
    open_list = []
    closed_list = set()

    start_node = Node(start_lane)
    end_point = Point(end_lane.geom.coords[-1])  # 将终点坐标元组转换为 Point 对象

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.lane == end_lane:
            return reconstruct_path(current_node)

        closed_list.add(current_node.lane)

        # 查找从当前 lane 的末端出发的 link
        key = (current_node.lane.roadPkid, current_node.lane.laneId)
        links = link_dict.get(key, [])
        for link in links:
            next_lane = lane_dict.get((link.toRoadPkid, link.toLaneId))
            if next_lane in closed_list:
                continue
            # 计算新的实际代价
            g = current_node.g + current_node.lane.geom.length + link.geom.length + next_lane.geom.length
            h = heuristic(Point(next_lane.geom.coords[-1]), end_point)  # 终点距离
            next_node = Node(next_lane, g, h, current_node, link)

            heapq.heappush(open_list, next_node)

        # 变道逻辑：检查左车道
        left_lane = lane_dict.get((current_node.lane.roadPkid, current_node.lane.laneId + 1))
        if left_lane and left_lane not in closed_list:
            g = current_node.g + current_node.lane.geom.length + left_lane.geom.length
            h = heuristic(Point(left_lane.geom.coords[-1]), end_point)
            left_node = Node(left_lane, g, h, current_node)

            # 生成平滑的变道曲线
            transition_geom = create_transition_geom(current_node.lane, left_lane, link_dict)
            left_node.transition_geom = transition_geom
            left_node.is_transition = True

            heapq.heappush(open_list, left_node)

        # 变道逻辑：检查右车道
        right_lane = lane_dict.get((current_node.lane.roadPkid, current_node.lane.laneId - 1))
        if right_lane and right_lane not in closed_list:
            g = current_node.g + current_node.lane.geom.length + right_lane.geom.length
            h = heuristic(Point(right_lane.geom.coords[-1]), end_point)
            right_node = Node(right_lane, g, h, current_node)
            right_node.is_transition = True

            # 生成平滑的变道曲线
            transition_geom = create_transition_geom(current_node.lane, right_lane, link_dict)
            right_node.transition_geom = transition_geom

            heapq.heappush(open_list, right_node)

    return None  # 没有找到路径


def reconstruct_path(node):
    path = []
    while node:
        path.append(node)
        node = node.parent
    return path[::-1]  # 反转路径


@app.route('/route', methods=['GET'])
def calculate_route():
    start_road_pkid = int(request.args.get('start_road_pkid'))
    start_lane_id = int(request.args.get('start_lane_id'))
    end_road_pkid = int(request.args.get('end_road_pkid'))
    end_lane_id = int(request.args.get('end_lane_id'))

    start_lane = lane_dict.get((start_road_pkid, start_lane_id))  # 起点 Lane
    end_lane = lane_dict.get((end_road_pkid, end_lane_id))  # 终点 Lane

    path = find_path(lane_dict, start_lane, end_lane, links)
    if path:
        combined_coords = []
        # for node in path:
        #     # 如果有 link_geom，先拼接 link_geom 的坐标
        #     if node.link:
        #         combined_coords.extend([(point.x, point.y) for point in node.link.points84])
        #     combined_coords.extend([(point.x, point.y) for point in node.lane.points84])

        for node in path:
            # 如果有 transition_geom，先拼接 transition_geom 的坐标
            if node.transition_geom:
                combined_coords.extend([(point[0], point[1]) for point in node.transition_geom.coords])
            # 如果有 link_geom，拼接 link_geom 的坐标
            elif node.link:
                combined_coords.extend([(point.x, point.y) for point in node.link.points84])
            if not node.is_transition:
                combined_coords.extend([(point.x, point.y) for point in node.lane.points84])

        combined_line = LineString(combined_coords)
        wkt_string = combined_line.wkt

        return wkt_string
    else:
        return jsonify({"error": "No path found"}), 404


import matplotlib.pyplot as plt
import numpy as np

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


if __name__ == '__main__':
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
        
    # conn = sqlite3.connect("hm_beijing_3.3.2_20251231.sqlite")
    # conn.enable_load_extension(True)

    # try:
    #     conn.load_extension("mod_spatialite")
    #     print("Spatialite loaded OK")
    # except Exception as e:
    #     print("Failed to load spatialite:", e)

    # cursor = conn.cursor()
    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='TX_LANE'")
    # print("TX_LANE:", cursor.fetchall())
    
    # breakpoint()
    
    # utm_start_X: 432343.38092485664
    # utm_start_Y: 4396792.396297588
    try:
        conn = DBManager.initialize_database("hm_beijing_3.3.2_20251231.sqlite")
        
        lat = 40.1758831781218
        lon = 116.6467321714981
        
        # 经度: 116.6467321714981
        # 纬度: 40.1758831781218
        # lat = 40.09027083   # 北纬
        # lon = 116.53316961  # 东经
        
        ego_range = [-81.6, -48, 97.6, 48]

        # 宽度和高度
        width = ego_range[2] - ego_range[0]   # x_max - x_min
        height = ego_range[3] - ego_range[1]  # y_max - y_min

        # 半径 = 对角线的一半
        radius_m = int(np.sqrt(width**2 + height**2) / 2)
        ego_range_post = [-radius_m, -radius_m, radius_m, radius_m]
        
        # 加载 ego 附近 Lane
        # ego_lanes = load_lanes_around_ego(conn, lon, lat, ego_range_post)
        data = load_hdmap_around_ego(conn, lon, lat, ego_range_post)

        # print(f"裁剪后 Lane 数量: {len(ego_lanes)}")
        
        # lane_sql = '''SELECT road_pkid, lane_id, AsText(geom) FROM `TX_LANE`;'''
        # lanes = DBManager.select_geometry(conn, lane_sql, 'Lane')
        # breakpoint()

        # link_sql = '''SELECT pkid, from_road_pkid, from_lane_id, to_road_pkid, to_lane_id, AsText(geom) FROM `TX_LANE_LINK`;'''
        # links = DBManager.select_geometry(conn, link_sql, 'Link')

        # lane_dict = {(lane.roadPkid, lane.laneId): lane for lane in lanes}

        # for link in links:
        #     key = (link.fromRoadPkid, link.fromLaneId)
        #     if key not in link_dict:
        #         link_dict[key] = []
        #     link_dict[key].append(link)

    except sqlite3.Error as e:
        print("SQLite错误:", e)
    # app.run(host='0.0.0.0', port=5000)

    if True:
        center_utm = wgs84_to_utm(lon, lat)
        draw_hdmap_centered(
            data,
            center_utm,
            save_path="/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/extract_hdmap_featrue/hd_map.png"
        )
        
