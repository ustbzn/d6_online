# import pickle
# from collections import defaultdict
# import random
# import os
# from PIL import Image

# # Load data
# # data = pickle.load(open('/pcpt/pcpt/D6_1015/d6_10k_multibalance_train.pkl', 'rb'))
# path_pkl = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/all_data/train_bj_3d_d6_s1001779_20251209_trainval_batch_11k_temporal.pkl'
# path_pkl_new = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/all_data/train_bj_3d_d6_s1001779_20251209_trainval_batch_11k_temporal_invalid.pkl'
# data = pickle.load(open(path_pkl, 'rb'))
# all_cam_list = ['60_front', '120_front', '120_back', 'left_back', 'right_back', '120_left', '120_right', '60_left_front', '60_right_front']

# infos = data['infos']
# print(f"Original number of infos: {len(infos)}")

# # Assuming dataset_root is the base path, you may need to set it appropriately
# dataset_root = '/pcpt/pcpt/D6_1015'  # Adjust if necessary

# valid_infos = []
# for info in infos:
#     calib_dic = info['calib_data']
#     cam_list = list(set(list(calib_dic.keys())) & set(all_cam_list))
#     if len(cam_list) != len(all_cam_list):
#         print(f"Skipping info due to missing cameras: cam_list {cam_list} != all_cam_list {all_cam_list}")
#         continue

#     is_valid = True
#     for cam_type in all_cam_list:
#         if 'calib_json_name' in info:
#             cam_token = info['data_path'].replace("3d_npz", cam_type).replace("npz", "jpg")
#             cam_path = os.path.join(dataset_root, cam_token)
#         elif 'calib_data' in info:
#             cam_path = calib_dic[cam_type]['cam_token']
#         if not os.path.exists(cam_path):
#             print(f"Cam path does not exist: {cam_path}")
#             is_valid = False
#             break
#         else:
#             try:
#                 Image.open(cam_path)
#             except Exception as e:
#                 print(f"JPG file invalid: {cam_path}, error: {e}")
#                 is_valid = False
#                 break
#     if is_valid:
#         valid_infos.append(info)

# print(f"Number of valid infos: {len(valid_infos)}")

# # Save new data
# new_data = data.copy()
# new_data['infos'] = valid_infos
# new_pkl_path = path_pkl_new
# with open(new_pkl_path, 'wb') as f:
#     pickle.dump(new_data, f)

# print(f"New PKL saved to: {new_pkl_path}")


import pickle
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================== 路径 ==================
path_pkl = '/pcpt/pcpt/D6/d6_bj_3d_d6_S1001779_20251203_val_all_1k_0116_manlabel_model_np123.pkl'
path_pkl_new = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/d6_bj_3d_d6_S1001779_20251203_val_all_1k_0116_manlabel_model_np123.pkl'

no_include = ['1764819709.100200891','1764819708.600441933']

dataset_root = '/pcpt/pcpt/D6_1015'

all_cam_list = [
    '60_front', '120_front', '120_back',
    'left_back', 'right_back',
    '120_left', '120_right',
    '60_left_front', '60_right_front'
]

# ================== Load ==================
data = pickle.load(open(path_pkl, 'rb'))
infos = data['infos']
print(f"Original number of infos: {len(infos)}")

# ================== 单条校验函数 ==================
def check_one_info(info):
    calib_dic = info['calib_data']
    cam_list = list(set(calib_dic.keys()) & set(all_cam_list))
    if len(cam_list) != len(all_cam_list):
        return None
    
    # if os.path.basename(info['data_path']).replace(".npz","") in no_include:
    #     print(os.path.basename(info['data_path']).replace(".npz",""))
    #     return None
    
    for cam_type in all_cam_list:
        if 'calib_json_name' in info:
            cam_token = info['data_path'].replace("3d_npz", cam_type).replace("npz", "jpg")
            cam_path = os.path.join(dataset_root, cam_token)
        else:
            cam_path = calib_dic[cam_type]['cam_token']

        if not os.path.exists(cam_path):
            return None

        try:
            Image.open(cam_path)
        except Exception:
            return None

    return info  # valid

# ================== 多线程校验 ==================
valid_infos = []

# IO 密集型，线程数可以大一点
num_workers = min(32, os.cpu_count() * 4)

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(check_one_info, info) for info in infos]

    for future in as_completed(futures):
        res = future.result()
        if res is not None:
            valid_infos.append(res)

print(f"Number of valid infos: {len(valid_infos)}")

# ================== 保存 ==================
valid_infos.sort(key=lambda x: x['timestamp'])
new_data = data.copy()
new_data['infos'] = valid_infos

with open(path_pkl_new, 'wb') as f:
    pickle.dump(new_data, f)

print(f"New PKL saved to: {path_pkl_new}")
