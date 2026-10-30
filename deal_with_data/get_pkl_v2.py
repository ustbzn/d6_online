import os
import glob
import numpy as np
from scipy.spatial import Delaunay
import tqdm
import pickle as pkl
from multiprocessing import Pool, cpu_count

# ================= 相机配置 =================
cam_list = [
    '60_front', '120_front', '120_back',
    'left_back', 'right_back',
    '120_left', '120_right',
    '60_left_front', '60_right_front'
]

cam_mapping = {
    'sensing_60_front': '60_front',
    'sensing_120_front': '120_front',
    'sensing_120_rear': '120_back',
    'sensing_120_left_backward': 'left_back',
    'sensing_120_right_backward': 'right_back',
    'sensing_120_left': '120_left',
    'sensing_120_right': '120_right',
    'sensing_60_left_front': '60_left_front',
    'sensing_60_right_front': '60_right_front'
}

cam_file_mapping = {
    'sensing_60_front': '60_front',
    'sensing_120_front': '120_front',
    'sensing_120_rear': '120_back',
    'sensing_120_left_backward': 'left_back',
    'sensing_120_right_backward': 'right_back',
    'sensing_120_left': 'left_front',
    'sensing_120_right': 'right_front',
    'sensing_60_left_front': 'left_front_60',
    'sensing_60_right_front': 'right_front_60'
}

# ================= label =================
label_id_mapping = {
    '1000': 'person','1001': 'person','1002': 'person','1003': 'person','1004': 'person',
    '2000': 'person',
    '3001': 'bike','3002': 'bike','3003': 'bike','3004': 'bike',
    '4000': 'rider','4001': 'rider','4002': 'rider','4003': 'rider','4004': 'rider',
    '5000': 'car','5001': 'car','5002': 'car','5003': 'car','5004': 'car',
    '6000': 'truck','6001': 'truck','6002': 'truck','6003': 'truck',
    '7000': 'bus','7001': 'bus','7002': 'bus',
    '9001': 'conebucket','9002': 'waterhorse','9003': 'bullbarrel',
    '9004': 'construction','9005': 'warning'
}

label_id_remove = [11001, 11002, 11003]

# ================= 几何工具 =================
def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

def csr2corners(center, size, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]])

    l, w, h = size
    x = [ l/2, l/2,-l/2,-l/2, l/2, l/2,-l/2,-l/2]
    y = [ w/2,-w/2,-w/2, w/2, w/2,-w/2,-w/2, w/2]
    z = [ h/2]*4 + [-h/2]*4

    corners = np.dot(R, np.stack([x,y,z],0))
    corners += np.array(center).reshape(3,1)
    return corners.T

# ================= 标定 =================
def adapt_npz_to_calib(npz_data, npz_path, cam_list):
    calib_objs = {}
    for key in npz_data.files:
        if not key.startswith('sensing_'):
            continue

        calib_key = cam_mapping.get(key)
        npz_key = cam_file_mapping.get(key)

        if calib_key not in cam_list:
            continue

        sensing_data = npz_data[key].item()
        cam_token = npz_data[npz_key].item() if npz_key in npz_data else None
        
        if cam_token == None:
            print(npz_key,'no find ==========================================')

        calib_objs[calib_key] = {
            'height': sensing_data['height'],
            'width': sensing_data['width'],
            'intrinsic': sensing_data['intrinsic'],
            'distort': sensing_data['distort'],
            'extrinsic': sensing_data['extrinsic'],
            'sensor2lidar_rotation': sensing_data['sensor2lidar_rotation'],
            'sensor2lidar_translation': sensing_data['sensor2lidar_translation'],
            'cam_token': cam_token
        }
    return calib_objs

# ================= 核心处理（不改逻辑） =================
def process_box(bag_dir, bag_id):
    npz_list = sorted(glob.glob(bag_dir + '/*.npz'))
    bag_info_list = []

    for npz_path in npz_list:
        npz_data = np.load(npz_path, allow_pickle=True)
        ts = os.path.basename(npz_path).replace('.npz', '')

        try:
            box_list = npz_data['boxes']
            box_label_list = npz_data['boxes_label']
            pcd = npz_data['pcds']
        except:
            box_list = npz_data['bboxes']
            box_label_list = npz_data['bboxes_sublabel']
            pcd = npz_data['pcd']

        if pcd.shape[0] <= 1000:
            continue

        excess = npz_data.get('is_excessive_layering', np.zeros(len(box_list)))
        group = npz_data.get('group', np.zeros(len(box_list)))
        cam_based = npz_data.get('is_camera_based', np.zeros(len(box_list)))

        valid = (excess + group + cam_based) == 0
        box_list = box_list[valid]
        box_label_list = box_label_list[valid]

        keep = [l not in label_id_remove for l in box_label_list]
        box_list = box_list[keep]
        box_label_list = box_label_list[keep]

        box_num_pts = []
        for box in box_list:
            x,y,z,w,h,d,a = box
            corners = csr2corners([x,y,z],[w,h,d],a)
            mask = in_hull(pcd[:,:3], corners)
            box_num_pts.append(mask.sum())

        cur_info = {
            'lidar_token': ts,
            'data_path': npz_path,
            'sample_token': bag_id,
            'timestamp': float(ts),
            'gt_boxes': box_list.astype(np.float32),
            'gt_labels': box_label_list.astype(np.int64),
            'gt_names': np.array([label_id_mapping[str(i)] for i in box_label_list]),
            'gt_velocity': np.zeros((len(box_list),3),np.float32),
            'pose': npz_data.get('transform_mat', np.eye(4)),
            'num_lidar_pts': np.array(box_num_pts),
            'calib_data': adapt_npz_to_calib(npz_data, npz_path, cam_list)
        }

        if len(cur_info['calib_data']) == 9:
            bag_info_list.append(cur_info)

    return bag_info_list

# ================= 多进程 wrapper =================
def process_box_wrapper(args):
    return process_box(*args)

# ================= bag 级并行 =================
def process_bag_data(npz_path, bag_list):
    tasks = [(os.path.join(npz_path, bag), bag) for bag in bag_list]
    all_infos = []

    workers = max(1, cpu_count() // 2)
    with Pool(workers) as pool:
        for infos in tqdm.tqdm(
            pool.imap_unordered(process_box_wrapper, tasks),
            total=len(tasks),
            desc='Processing Bags (MP)'
        ):
            all_infos.extend(infos)

    return all_infos

# ================= 保存 =================
def save_to_pkl(data, out_pkl):
    with open(out_pkl, 'wb') as f:
        pkl.dump({'infos': data}, f)

# ================= main =================
def main():
    data_root = '/pcpt/pcpt/D6_1015/data/temporal_data_251224/keep_filter_bboxes_allobj_new'
    flags = ['train_bj_3d_d6_s1001779_20251209']
    out_pkl = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/all_data/train_bj_3d_d6_s1001779_20251209_trainval_batch_11k_temporal.pkl'

    all_infos = []
    for flag in flags:
        folder = os.path.join(data_root, flag)
        bag_list = os.listdir(folder)
        infos = process_bag_data(folder, bag_list)
        all_infos.extend(infos)
        
    all_infos.sort(key=lambda x: x['timestamp'])
    save_to_pkl(all_infos, out_pkl)
    print('Saved to', out_pkl)

if __name__ == '__main__':
    main()
