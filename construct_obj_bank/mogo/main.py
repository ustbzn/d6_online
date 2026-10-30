import os
import yaml
import pickle as pkl
import numpy as np

import utils
from multi_process import MultiProcess

import pdb
import glob
import tqdm
fpath_mogo = '/pcpt/pcpt/data/B22024'
fpath_mogo_bank = os.path.join(fpath_mogo, 'bankB22024_toT12025')
class_set = set(("person", "bike", "rider", "car", "truck", "bus", "obstacle", "traffic_sign", "noise"))

for cate in class_set:
    fpath_cate = os.path.join(fpath_mogo_bank, cate)
    if not os.path.exists(fpath_cate):
        os.system("mkdir -p {}".format(fpath_cate))


with open('/pcpt/pcpt/project/lijun/multimodal_bevfusion/configs/mogo/mogo_21.yaml', 'r') as f:
    task_cfg = yaml.load(f, Loader=yaml.Loader)


common_infos=dict(class_set=class_set, fpath_mogo_bank=fpath_mogo_bank, task_cfg=task_cfg)

# flist = []
# for bag_id in os.listdir(os.path.join(fpath_mogo, "dataB22024_250220")):
#     fpath_bag = os.path.join(fpath_mogo, "dataB22024_250220", bag_id)
#     for fn in os.listdir(fpath_bag):
#         if fn.endswith('.npz'):
#             fname_npz = os.path.join(fpath_bag, fn)
#             flist.append(fname_npz)

file_pkl = os.path.join(fpath_mogo, "filter_B2_infos_20250324_250701allmerge.pkl")
with open(file_pkl, 'rb') as f:
        pkl_dict = pkl.load(f)
flist = []
# print(len(pkl_dict['infos']))
for fn in pkl_dict['infos']:
    # print("fn      :    {}".format(fn))
    # print('root:{}'.format(fn))
    lidar_path=os.path.join(fpath_mogo, fn['data_path'])
    # npz_files_in_dir = glob.glob(os.path.join(fn, '*.npz'))
    flist.append(lidar_path)

# file_pkl = os.path.join(fpath_mogo, "filter_B2_infos_20250324_250612allmerge_val.pkl")
# with open(file_pkl, 'rb') as f:
#         pkl_dict2 = pkl.load(f)
# # print(len(pkl_dict2['infos']))

# for fn in pkl_dict2['infos']:
#     # print('root:{}'.format(fn))
#     lidar_path=os.path.join(fpath_mogo, fn['data_path'])
#     # npz_files_in_dir = glob.glob(os.path.join(fn, '*.npz'))
#     flist.append(lidar_path)
print(len(flist))
for fname_npz in tqdm.tqdm(flist):
    utils.process_single(fname_npz, common_infos=common_infos)
# # multi-process
# multi_obj = MultiProcess(flist, utils.process_single, num_workers=40, common_infos=common_infos)
# for i, data in enumerate(multi_obj.run()):
#     if i % 5000 == 0:
#         print("Samples: {}".format(i))