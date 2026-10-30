#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle


def merge_pkl(data, loaded_data):
    data = data + loaded_data['infos']
    return data



if __name__ == '__main__':

    pkl_list = ['train_yt_rsp_falcon_230616_7587D_L', 
                'train_yt_rsp_sweeper_falcon_230613_758D_Y',
                'train_yt_rsp_sweeper_falcon_230615_7587D_A_3926']
    merged_name = '../data/3_L_Y_A_v1.0.pkl'
    data = []
    for pkl_name in pkl_list:
        pkl_path = '../data/' + pkl_name + '/mogo_infos_train.pkl'
        with open(pkl_path, 'rb') as file:
            loaded_data = pickle.load(file)
            data = merge_pkl(data, loaded_data)

    all_db_infos = dict()
    all_db_infos['infos'] = data
    with open(merged_name, 'wb') as file:
        pickle.dump(all_db_infos, file)
