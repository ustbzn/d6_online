import numpy as np
import os

from multi_process import MultiProcess

import pdb


fpath_mogo = '/rtx/miaochunle/datasets/rs80/mogo_all'
fpath_mogo_motion = '/rtx/miaochunle/datasets/rs80/mogo_all_motion'

fpath_target = '/rtx/andy/datasets/mogo_final'

if os.path.exists(fpath_target):
    os.system("rm -rf {}".format(fpath_target))

os.system("mkdir -p {}".format(fpath_target))

flist = []
key_list = ['train', 'val']
for key in key_list:
    fpath_mogo_key = os.path.join(fpath_mogo, key)
    fpath_mogo_motion_key = os.path.join(fpath_mogo_motion, key)

    for bag_name in os.listdir(fpath_mogo_key):
        fpath_mogo_key_bag = os.path.join(fpath_mogo_key, bag_name)
        fpath_mogo_motion_key_bag = os.path.join(fpath_mogo_motion_key, bag_name)
        fpath_target_key_bag = os.path.join(fpath_target, key, bag_name)

        assert(os.path.exists(fpath_mogo_key_bag))
        assert(os.path.exists(fpath_mogo_motion_key_bag))
        flist.append((fpath_mogo_key_bag, fpath_mogo_motion_key_bag, fpath_target_key_bag))


print("Total bags: {}".format(len(flist)))


def process_single(meta_data, common_infos=None):
    fpath_mogo_key_bag, fpath_mogo_motion_key_bag, fpath_target_key_bag = meta_data
    if not os.path.exists(fpath_target_key_bag):
        os.system("mkdir -p {}".format(fpath_target_key_bag))
    
    fn_list = [x for x in os.listdir(fpath_mogo_key_bag) if x.endswith('.npz')]
    fn_list.sort()
    for t, fn in enumerate(fn_list):
        if fn.endswith('.npz'):
            fname_mogo_npz = os.path.join(fpath_mogo_key_bag, fn)
            fname_mogo_motion_npz = os.path.join(fpath_mogo_motion_key_bag, fn)
            fname_target_npz = os.path.join(fpath_target_key_bag, fn)

            data_dic = dict(np.load(fname_mogo_npz))
            if t >= 1:
                data_dic_motion = dict(np.load(fname_mogo_motion_npz))

                motion_v = data_dic_motion['motion_v']
                motion_label = data_dic_motion['motion_label']
                box_motion = data_dic_motion['box_motion']
                box_motion_label = data_dic_motion['box_motion_label']
                pc_cur_num = data_dic_motion['point_num'][-1]
                pc_cur = data_dic_motion['pcd'][-1, :pc_cur_num]

                error1 = np.abs(data_dic['bboxes'] - data_dic_motion['boxes'])
                error2 = np.abs(data_dic['pcd'] - pc_cur)
                assert(error1.max() <= 0.01)
                assert(error2.max() <= 0.01)

                # record data
                data_dic['motion_v'] = motion_v[:pc_cur_num]
                data_dic['motion_label'] = motion_label[:pc_cur_num]
                data_dic['bboxes_motion'] = box_motion
                data_dic['bboxes_motion_label'] = box_motion_label
            else:
                data_dic['motion_v'] = np.zeros((data_dic['pcd'].shape[0], 3), dtype=np.float16)
                data_dic['motion_label'] = np.zeros((data_dic['pcd'].shape[0], 1), dtype=np.int8) - 1
                data_dic['bboxes_motion'] = np.zeros((data_dic['bboxes'].shape[0], 3), dtype=np.float16)
                data_dic['bboxes_motion_label'] = np.zeros((data_dic['bboxes'].shape[0],), dtype=np.int8) - 1
            
            np.savez_compressed(fname_target_npz, **data_dic)

            yield 1


# multi-process
multi_obj = MultiProcess(flist, process_single, num_workers=40, common_infos=None)
for i, data in enumerate(multi_obj.run()):
    if i % 5000 == 0:
        print("Samples: {}".format(i))