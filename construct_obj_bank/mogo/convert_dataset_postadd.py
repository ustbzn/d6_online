import numpy as np
import os

import multiprocessing
import functools
import random

import pdb


fpath_mogo_motion = '/rtx/miaochunle/datasets/rs80/mogo_all_oneframe_motion'
fpath_target = '/rtx/andy/datasets/mogo_final_v2'

fname_train_txt = '/rtx/miaochunle/datasets/rs80/train_20231023.txt'

bag_new_list = []
with open(fname_train_txt, 'r') as f:
    for line in f:
        bag_name = line.strip()
        fpath_bag_target = os.path.join(fpath_target, bag_name)
        if not os.path.exists(fpath_bag_target):
            fpath_bag = os.path.join(fpath_mogo_motion, bag_name)
            bag_new_list.append(fpath_bag)


def get_baginfo(fpath_bag):
    fn_list = [fn for fn in os.listdir(fpath_bag) if fn.endswith('.npz')]
    time_list = np.array([fn.replace('.npz', '') for fn in fn_list], dtype=np.double)

    if len(time_list) <= 1:
        return (fpath_bag, fn_list, False, len(time_list), True)
    
    # sort
    sorted_index = np.argsort(time_list)
    time_list = time_list[sorted_index]
    fn_list = np.array(fn_list)[sorted_index]

    time_diff = time_list[1:] - time_list[:-1]

    # assign labels
    valid_seq = True
    file_length = 0
    only_det = False
    if(time_diff.max() > 0.15):
        valid_seq = False
    
    file_length = len(fn_list)
    data_dic = np.load(os.path.join(fpath_bag, fn_list[0]), allow_pickle=True)
    pcds_label = data_dic['pcds_label']
    valid_stuff_mask = (pcds_label < 1000) * (pcds_label > 0)
    if(valid_stuff_mask.sum() < 5):
        only_det = True
    
    return (fpath_bag, fn_list.tolist(), valid_seq, file_length, only_det)


def convert_bag(bag_info, mode):
    bag_name = bag_info[0].split('/')[-1]
    fpath_target_bag = os.path.join(fpath_target, mode, bag_name)
    os.system("mkdir -p {}".format(fpath_target_bag))
    for t, fn in enumerate(bag_info[1]):
        data_dic_src = np.load(os.path.join(bag_info[0], fn), allow_pickle=True)
        fname_tgt_npz = os.path.join(fpath_target_bag, "{}.npz".format(str(t).rjust(4, '0')))

        saved_dic = {}
        saved_dic['pcd'] = data_dic_src['pcds'].astype(np.float16)
        saved_dic['pcd_label'] = data_dic_src['pcds_label'].astype(np.int64)
        saved_dic['bboxes'] = data_dic_src['boxes'].astype(np.float32)
        saved_dic['bboxes_label'] = data_dic_src['boxes_label'].astype(np.int64)
        saved_dic['objects_id'] = data_dic_src['oids'].astype(np.int64)
        saved_dic['pose'] = data_dic_src['pose'].astype(np.double)
        saved_dic['motion_v'] = data_dic_src['motion_v'].astype(np.float16)
        saved_dic['motion_label'] = data_dic_src['motion_label'].astype(np.int64)
        saved_dic['bboxes_motion'] = data_dic_src['box_motion'].astype(np.float32)
        saved_dic['bboxes_motion_label'] = data_dic_src['boxes_label'].astype(np.int64)

        groups = data_dic_src['groups']
        if len(groups.shape) == 0:
            saved_dic['groups'] = np.zeros((saved_dic['bboxes'].shape[0],), dtype=np.int8)
        else:
            saved_dic['groups'] = data_dic_src['groups'].astype(np.int8)
        
        is_excessive_layering_list = data_dic_src['is_excessive_layering_list']
        if len(is_excessive_layering_list.shape) == 0:
            saved_dic['is_excessive_layering_list'] = np.zeros((saved_dic['bboxes'].shape[0],), dtype=np.int8)
        else:
            saved_dic['is_excessive_layering_list'] = data_dic_src['is_excessive_layering_list'].astype(np.int8)
        
        np.savez_compressed(fname_tgt_npz, **saved_dic)


# assign label to each bag
pool1 = multiprocessing.Pool(processes = 32)
bag_info_list = pool1.map(get_baginfo, bag_new_list)
pool1.close()
pool1.join()

bag_info_list = [meta_data for meta_data in bag_info_list if meta_data[2]]

# convert data
pool_train = multiprocessing.Pool(processes = 32)
pool_train.map(functools.partial(convert_bag, mode="train"), bag_info_list)
pool_train.close()
pool_train.join()