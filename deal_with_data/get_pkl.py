import os
import glob
import numpy as np
import glob
from scipy.spatial import Delaunay
import tqdm
import pickle as pkl
from multiprocessing import Manager, Pool, Process

# cam_list = ['30_front','60_front', '120_front', '120_left', '120_right', '120_back']
cam_list = ['60_front', '120_front', '120_back', 'left_back', 'right_back', '120_left', '120_right', '60_left_front', '60_right_front']

cam_mapping = {'sensing_60_front': '60_front',
               'sensing_120_front': '120_front',
               'sensing_120_rear': '120_back',
               'sensing_120_left_backward': 'left_back',
               'sensing_120_right_backward': 'right_back',
               'sensing_120_left': '120_left',
               'sensing_120_right': '120_right',
               'sensing_60_left_front': '60_left_front',
               'sensing_60_right_front': '60_right_front'
               }

cam_file_mapping = {'sensing_60_front': '60_front',
               'sensing_120_front': '120_front',
               'sensing_120_rear': '120_back',
               'sensing_120_left_backward': 'left_back',
               'sensing_120_right_backward': 'right_back',
               'sensing_120_left': 'left_front',
               'sensing_120_right': 'right_front',
               'sensing_60_left_front': 'left_front_60',
               'sensing_60_right_front': 'right_front_60'
               }
# all label id {5001, 5002, 5003, 5004, 4001, 4002, 4003, 9001, 9002, 3001, 3002, 3003, 7001, 7002, 1001, 1002, 1003, 1004, 6001, 6002, 6003}
label_id_mapping = {'1000': 'person',
                    '1001': 'person',
                    '1002': 'person',
                    '1003': 'person',
                    '1004': 'person',
                    '2000': 'person',
                    '3001': 'bike',
                    '3002': 'bike',
                    '3003': 'bike',
                    '3004': 'bike',
                    '4000': 'rider',
                    '4001': 'rider',
                    '4002': 'rider',
                    '4003': 'rider',
                    '4004': 'rider',
                    '5000': 'car',
                    '5001': 'car',
                    '5002': 'car',
                    '5003': 'car',
                    '5004': 'car',
                    '6000': 'truck',
                    '6001': 'truck',
                    '6002': 'truck',
                    '6003': 'truck',
                    '7000': 'bus',
                    '7001': 'bus',
                    '7002': 'bus',
                    '9001': 'conebucket',
                    '9002': 'waterhorse',
                    '9003': 'bullbarrel',
                    '9004': 'construction',
                    '9005': 'warning',
                    }

# 平台的残影框、偏差框、后车框
label_id_remove = [11001, 11002, 11003]

def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

def csr2corners(center, size, yaw):
    '''
        0 -------- 1
       /|         /|
      3 -------- 2 .
      | |        | |
      . 4 -------- 5
      |/         |/
      7 -------- 6
    Input:
        center, size, 3
        yaw, 1
    Output:
        corners_3d, (8, 3)
    '''
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])

    # 3d bounding box dimensions
    l = size[0]
    w = size[1]
    h = size[2]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.stack((x_corners, y_corners, z_corners), axis=0))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = corners_3d.T
    return corners_3d

def process_box(bag_dir, bag_id, mgr_list):
    npz_list = glob.glob(bag_dir+'/*.npz')
    npz_list.sort()
    ts_list = [_.strip('.npz').split('/')[-1] for _ in npz_list]
    ts_float_list = [float(_) for _ in ts_list]
    bag_info_list = []
    # print(ts_list[1])
    for i in range(len(ts_list)):
        npz_data = np.load(npz_list[i], allow_pickle=True)
        # if npz_list[i] != "/pcpt/pcpt/D6_1015/data/npz/train_bj_3d_d6_S1001779_20251022/20251013162854-1760344134921892046-full-S1001779/1760344127.200437307.npz":
        #     continue

        print(f"npz_path: {npz_list[i]}")
        # print(npz_data.files)
        # breakpoint()
        try:
            box_list = npz_data['boxes']
            box_label_list=npz_data['boxes_label']
            pcd = npz_data['pcds']
        except:
            box_list = npz_data['bboxes']
            box_label_list=npz_data['bboxes_sublabel']
            # print(box_label_list)
            pcd = npz_data['pcd']
        
        if "is_excessive_layering" in npz_data and len(npz_data["is_excessive_layering"]) > 0:
            excess_list = npz_data["is_excessive_layering"]
        else:
            excess_list =  np.zeros_like(box_list)
        
        if "group" in npz_data and len(npz_data["group"]) > 0:
            groups_id = npz_data["group"]
        else:
            groups_id = np.zeros_like(box_list)

        if "is_camera_based" in npz_data and len(npz_data["is_camera_based"]) > 0:
            is_camera_based_id = npz_data["is_camera_based"]
        else:
            is_camera_based_id = np.zeros_like(box_list)
               
        assert(excess_list.shape[0] == groups_id.shape[0] == is_camera_based_id.shape[0] == box_list.shape[0])
        # print('excess_list',excess_list.shape,' groups_id',groups_id.shape,' is_camera_based_id',is_camera_based_id.shape,' box_list',box_list.shape, )

        filter_mask = excess_list + groups_id + is_camera_based_id
        # filter_mask = groups_id

        mask_sum = np.sum(filter_mask)

        if mask_sum > 0:
            # print("npz name is {}".format(npz_list[i]))
            # print("filter_mask  is {}".format(filter_mask))
        
            # 既不是残影，也不是 group 的 box 留下, 也不是视觉框的留下
            valid_mask = np.where(filter_mask == 0)
            # print("raw box list len is {}".format(len(box_list)))
            box_list = box_list[valid_mask] 
            box_label_list = box_label_list[valid_mask]     
            assert len(box_label_list) == len(box_list)    
            # print("filter box list len is {}".format(len(box_list)))

        # 11001,11003 11002残影框 偏差框 后车框过滤掉, 平台错误没有绑定关系
        anno_error_mask = [False if label in label_id_remove else True for label in box_label_list]
        box_label_list = box_label_list[anno_error_mask]
        box_list = box_list[anno_error_mask]

        assert len(box_label_list) == len(box_list)
        
        if pcd.shape[0] <= 1000:
            # print(npz_list[i])
            continue

        try:
            cur_pose = npz_data['transform_mat']
        except:
            cur_pose = np.eye(4,dtype=np.float32)

        box_num_pts_list = []
        for bidx, box in enumerate (box_list):
            x,y,z,w,h,d,a = box
            corner_box = csr2corners([x,y,z],[w,h,d],a)
            box_pcd_flag = in_hull(pcd[:,:3], corner_box)
            box_num_pts_list.append(box_pcd_flag.sum())
       
        gt_boxes = np.array(box_list).astype(np.float32)

        assert len(gt_boxes) == len(box_list) == len(box_label_list) == len(box_num_pts_list)
        cur_info_dict = {
            'lidar_token': ts_list[i],
            'data_path': npz_list[i],
            'sample_token': bag_id,
            'timestamp': ts_float_list[i],
            'gt_boxes': gt_boxes,
            'gt_labels': np.array(box_label_list).astype(np.int64),
            'gt_names': np.array([label_id_mapping[str(label_id)] for label_id in box_label_list]),
            'gt_velocity': np.zeros((box_list.shape[0], 3), dtype=np.float32),
            'pose': np.array(cur_pose).astype(np.double),#4*4
            'num_lidar_pts': np.array(box_num_pts_list).astype(np.float32),
            'calib_data': adapt_npz_to_calib(npz_data, npz_list[i], cam_list)  # 直接存储标定数据
        }
        print(len(cur_info_dict['gt_boxes']), len(cur_info_dict['gt_labels']),len(cur_info_dict['gt_names']), len(box_num_pts_list))
        assert len(cur_info_dict['gt_boxes']) == len(cur_info_dict['gt_labels']) == len(cur_info_dict['gt_names']) == len(box_num_pts_list)
        # if len(cur_info_dict['calib_data']) != 9:
        #     print("camera num error___________:", npz_list[i])
        
        if len(cur_info_dict['calib_data']) == 9:
            bag_info_list.append(cur_info_dict)
    mgr_list.extend(bag_info_list)

# def adapt_npz_to_calib(npz_data, npz_path, cam_list):s
#     """将npz数据结构转换为原代码可处理的calib格式"""
    
#     def quaternion_to_rotation_matrix(q):
#         """将四元数转换为旋转矩阵"""
#         x, y, z, w = q['x'], q['y'], q['z'], q['w']
#         return np.array([
#             [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
#             [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
#             [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
#         ], dtype=np.float32)

#     calib_objs = {}
    
#     # 遍历所有sensing键
#     for key in npz_data.files:
#         if not key.startswith('sensing_'):
#             continue
            
#         # 解析传感器类型和方向
#         sensing_data = npz_data[key].item()
#         # print(sensing_data)
#         sensor_type = sensing_data['type']
#         if sensor_type == 'SENSING_60_3d':
#             continue
#         orientation = sensing_data['orientation'].lower()
#         if orientation == 'rear':
#             orientation = 'back'
        
#         # 生成标准键名 (例如: "30_front")
#         calib_key = f"{sensor_type.split('_')[-1]}_{orientation}"
#         if orientation == 'right_backward':
#             calib_key = 'right_back'
#         elif orientation == 'left_backward':
#             calib_key = 'left_back'
#         # print(f"Processing calibration key: {sensor_type.split('_')}, {calib_key}")
        
#         if calib_key not in cam_list:
#             continue
            
#         # 构建参数结构
#         extrinsic = sensing_data['extrinsic']
#         intrinsic = sensing_data['intrinsic']
#         distcoeff = sensing_data['distcoeff']
        
#         # 转换外参矩阵
#         rotation = quaternion_to_rotation_matrix(extrinsic['rotation'])
#         translation = np.array([
#             extrinsic['translation']['x'],
#             extrinsic['translation']['y'],
#             extrinsic['translation']['z']
#         ])
#         mat_cam2imu = np.column_stack([rotation, translation])
        
#         # 转换内参矩阵
#         intrinsic_matrix = np.array([
#             [intrinsic['matrix'], intrinsic['matrix_1'], intrinsic['matrix_2']],
#             [intrinsic['matrix_3'], intrinsic['matrix_4'], intrinsic['matrix_5']],
#             [intrinsic['matrix_6'], intrinsic['matrix_7'], intrinsic['matrix_8']]
#         ], dtype=np.float32)
        
#         # 转换畸变参数（调整顺序匹配原格式）
#         distort_params = np.array([
#             distcoeff['distort_matrix'],
#             distcoeff['distort_matrix_1'],
#             distcoeff['distort_matrix_2'],
#             distcoeff['distort_matrix_3'],
#             distcoeff['distort_matrix_4']
#         ], dtype=np.float32)

#         cam_token = npz_path
#         # import ipdb; ipdb.set_trace()
#         # ================ 5帧 ================
#         # cam_token = cam_token.replace("keep_filter_bboxes_box_seg_f2000", "data_orin")  # 替换父目录 data_path = '/rbs/bus_data/c1/data_20250305/data_orin/train_beijing_C1_7105_box_20250214_data_new/20250214163114-20250214163144-1739519177768807288-full-DFHYDB7105_09/1739521892.301194906.npz'
#         # path_parts = cam_token.split("/")
#         # data_orin_index = path_parts.index("data_orin")
#         # path_parts.insert(data_orin_index + 2, "all_frames")  # 在data_orin的下两级插入
#         # ================ 5帧 ================

#         # ================ 20帧 ================
#         # /pcpt/pcpt/data/C3/npz/train_bj_C3_1001386_seg_20250425/20250425143006-20250425143036-1745553791516623912-full-R1001386_03/1745562612.800450801.npz
#         path_parts = cam_token.split("/")
#         # id_pos = path_parts.index('id')
#         # del path_parts[id_pos]
#         c3_pos = path_parts.index('C3')
#         if path_parts[c3_pos + 1] == 'npz':
#             del path_parts[c3_pos + 1]
#         path_parts.insert(c3_pos + 2, "send")
#         # ================ 20帧 ================

#         # 通用
#         path_parts.insert(-1, calib_key)
#         cam_token = os.path.join("/",*path_parts)
#         cam_token = cam_token.replace("npz", "jpg")
#         if not os.path.isfile(cam_token):
#             print(cam_token)
        
#         sensor2lidar_t = np.array(mat_cam2imu.flatten().tolist(), dtype=np.float32).reshape(3, 4)
        
#         calib_objs[calib_key] = {
#             'height': intrinsic['height'],
#             'width': intrinsic['width'],
#             'intrinsic': intrinsic_matrix,
#             'distort': distort_params,
#             'extrinsic': mat_cam2imu.flatten().tolist(),  # 保持与原JSON兼容
#             'sensor2lidar_rotation': sensor2lidar_t[:3, :3],
#             'sensor2lidar_translation': sensor2lidar_t[:, -1],
#             'cam_token': cam_token
#         }
    
#     return calib_objs


def adapt_npz_to_calib(npz_data, npz_path, cam_list):
    """将npz数据结构转换为原代码可处理的calib格式"""
    calib_objs = {}
    
    # 遍历所有sensing键
    # for key in cam_list:
    #     calib_objs[key] = npz_data[key]


    for key in npz_data.files:
        if not key.startswith('sensing_'):
            continue
            
        # 解析传感器类型和方向
        sensing_data = npz_data[key].item()
        calib_key = cam_mapping[key]
        npz_key = cam_file_mapping[key]
        
        # print(sensing_data)
        # sensor_type = sensing_data['type']
        # if sensor_type == 'SENSING_60_3d':
        #     continue
        # orientation = sensing_data['orientation'].lower()
        # if orientation == 'rear':
        #     orientation = 'back'
        
        # # 生成标准键名 (例如: "30_front")
        # calib_key = f"{sensor_type.split('_')[-1]}_{orientation}"
        # if orientation == 'right_backward':
        #     calib_key = 'right_back'
        # elif orientation == 'left_backward':
        #     calib_key = 'left_back'
        # # print(f"Processing calibration key: {sensor_type.split('_')}, {calib_key}")
        
        # # print("calib_key", calib_key)
        if calib_key not in cam_list:
            continue
            
        # # 构建参数结构
        # extrinsic = sensing_data['extrinsic']
        # intrinsic = sensing_data['intrinsic']
        # distcoeff = sensing_data['distcoeff']
        
        # # 转换外参矩阵
        # rotation = quaternion_to_rotation_matrix(extrinsic['rotation'])
        # translation = np.array([
        #     extrinsic['translation']['x'],
        #     extrinsic['translation']['y'],
        #     extrinsic['translation']['z']
        # ])
        # mat_cam2imu = np.column_stack([rotation, translation])
        
        # # 转换内参矩阵
        # intrinsic_matrix = np.array([
        #     [intrinsic['matrix'], intrinsic['matrix_1'], intrinsic['matrix_2']],
        #     [intrinsic['matrix_3'], intrinsic['matrix_4'], intrinsic['matrix_5']],
        #     [intrinsic['matrix_6'], intrinsic['matrix_7'], intrinsic['matrix_8']]
        # ], dtype=np.float32)
        
        # # 转换畸变参数（调整顺序匹配原格式）
        # distort_params = np.array([
        #     distcoeff['distort_matrix'],
        #     distcoeff['distort_matrix_1'],
        #     distcoeff['distort_matrix_2'],
        #     distcoeff['distort_matrix_3'],
        #     distcoeff['distort_matrix_4']
        # ], dtype=np.float32)

        # cam_token = npz_path
        # npz = np.load(npz_path)
        # import ipdb; ipdb.set_trace()
        # ================ 5帧 ================
        # cam_token = cam_token.replace("keep_filter_bboxes_box_seg_f2000", "data_orin")  # 替换父目录 data_path = '/rbs/bus_data/c1/data_20250305/data_orin/train_beijing_C1_7105_box_20250214_data_new/20250214163114-20250214163144-1739519177768807288-full-DFHYDB7105_09/1739521892.301194906.npz'
        # path_parts = cam_token.split("/")
        # data_orin_index = path_parts.index("data_orin")
        # path_parts.insert(data_orin_index + 2, "all_frames")  # 在data_orin的下两级插入
        # ================ 5帧 ================

        # ================ 20帧 ================
        # /pcpt/pcpt/data/C3/npz/train_bj_C3_1001386_seg_20250425/20250425143006-20250425143036-1745553791516623912-full-R1001386_03/1745562612.800450801.npz
        # path_parts = cam_token.split("/")
        # # id_pos = path_parts.index('id')
        # # del path_parts[id_pos]
        # c3_pos = path_parts.index('data')
        # if path_parts[c3_pos + 1] == 'npz':
        #     del path_parts[c3_pos + 1]
        # path_parts.insert(c3_pos + 2, "send")
        # # ================ 20帧 ================

        # # 通用
        # breakpoint()
        # path_parts.insert(-1, cam_file_mapping[key])
        # cam_token = os.path.join("/",*path_parts)
        # cam_token = cam_token.replace("npz", "jpg")
        # if not os.path.isfile(cam_token):
        #     print(cam_token)
            
        # calib_objs['cam_token'] = cam_token
        if npz_key in npz_data:
            cam_token = npz_data[npz_key].item()
            breakpoint()
        else:
            print('############################# npz_key no find:  ',npz_key,'  #############################')
        
        calib_objs[calib_key] = {
            'height': sensing_data['height'],
            'width': sensing_data['width'],
            'intrinsic': sensing_data['intrinsic'],
            'distort': sensing_data['distort'],
            'extrinsic': sensing_data['extrinsic'],  # 保持与原JSON兼容
            'sensor2lidar_rotation': sensing_data['sensor2lidar_rotation'],
            'sensor2lidar_translation': sensing_data['sensor2lidar_translation'],
            'cam_token': cam_token
        }

    return calib_objs

def process_bag_data(npz_path, bag_list):
    mgr_list = []
    for bag_id in tqdm.tqdm(bag_list, desc="Processing Bags"):
        bag_dir = os.path.join(npz_path, bag_id)
        process_box(bag_dir, bag_id, mgr_list)
    #     break  # 只处理一个包，便于调试
    # print(f"Processed {len(mgr_list)} items from {npz_path}.")
    return mgr_list

def save_to_pkl(mgr_list, out_pkl_flag):
    with open(out_pkl_flag, 'wb') as f:
        pkl.dump({'infos': list(mgr_list)}, f)

def main():
    data_root = '/pcpt/pcpt/D6_1015/data/temporal_data_251224/keep_filter_bboxes_allobj_new'
    flags = ['train_bj_3d_d6_s1001779_20251209']

    out_pkl = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/all_data/train_bj_3d_d6_s1001779_20251209_trainval_batch_11k_temporal.pkl'
    # 结果存储列表
    all_mgr_list = []

    # 遍历每个flag指定的子文件夹
    nums = 0
    for item in flags:
        print(f"========== process {item} ==========")
        data_folder = os.path.join(data_root, item)

        # 遍历子文件夹中的所有目录
        sub_dirs = os.listdir(data_folder)
        # for bag in sub_dirs:
        #     nums += len(os.listdir(os.path.join(data_folder, bag)))

        #     break  # 只处理一个包，便于调试
    # print(f"Processed {len(mgr_list)} items from {npz_path}.")          
        
        # =========== 5帧 =================
        # for sub_dir in sub_dirs:
        #     npz_path = os.path.join(data_folder, sub_dir) # train_beijing_C1_7105_box_20250214_data_new
        #     bag_list = os.listdir(npz_path)  # 直接获取文件夹中的文件列表 0:20250214163941-20250214164011-1739519177768807294-full-DFHYDB7105_11
        #     import ipdb; ipdb.set_trace()

        #     # 处理每个包文件并将结果添加到 all_mgr_list 中
        #     # mgr_list = process_bag_data(npz_path, bag_list)
        #     print(f"Processed {len(mgr_list)} items from {sub_dir}.")

        #     # 将当前sub_dir的结果添加到所有结果列表中
        #     all_mgr_list.extend(mgr_list)
        # =========== 5帧 =================

        # import ipdb; ipdb.set_trace()
        # =========== 20帧 =================
        mgr_list = process_bag_data(data_folder, sub_dirs)
        print(f"Processed {len(mgr_list)} items from {item}.")

        # 将当前sub_dir的结果添加到所有结果列表中
        all_mgr_list.extend(mgr_list)
        # =========== 20帧 =================

    # 保存所有数据到一个pkl文件


    out_pkl_flag = out_pkl.format(item)  # 最后一个flag值作为文件名
    save_to_pkl(all_mgr_list, out_pkl_flag)
    print(f"All data saved to {out_pkl_flag}")

if __name__ == '__main__':
    main()
