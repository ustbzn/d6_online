import pickle as pkl 
import numpy as np
import os 

def save_to_pkl(mgr_list, out_pkl_flag):
    with open(out_pkl_flag, 'wb') as f:
        pkl.dump({'infos': list(mgr_list)}, f)
        
def fileter_exists_pkl(ann_file, out_val_path, val_pkl):
    # with open(train_pkl,'rb') as f:
    #     pkl_data = pkl.load(f)
    
    # train_list = [info['timestamp'] for info in pkl_data['infos']]
    # print('train_list  :',len(train_list))
        
    with open(val_pkl,'rb') as f:
        pkl_data = pkl.load(f)
    
    val_list = [info['timestamp'] for info in pkl_data['infos']]
    print('val_list  :',len(val_list))
    
    
    # for val in val_list:
    #     if val in train_list:
    #         print('###################################')
    #         print(val)
    
    with open(ann_file,'rb') as f:
        pkl_data = pkl.load(f)
    
    batch_data = []
    train_data = []
    val_data = []
    flag_name = set()
    print('ann_file  :',len(pkl_data['infos']))
    
    for info in pkl_data['infos']:
        data_path = info['timestamp']
        # if data_path in train_list:
        #     train_data.append(info)
            
        if data_path in val_list:
            val_data.append(info)
            flag_name.add(info['data_path'].split('/')[-3])
            
        # else:
        #     batch_data.append(info)
    
    # pkl_data['infos'] = batch_data

    # dir_name = os.path.dirname(out_path)
    # train_dirname = os.path.dirname(out_train_path)
    val_dirname = os.path.dirname(out_val_path)

    # if dir_name != "" and not os.path.exists(dir_name):
        # os.makedirs(dir_name, exist_ok=True)
    # save_to_pkl(batch_data, out_path)
    # print('batch_all_pkl  :',len(batch_data))
    # print('save_path      :',out_path)
    
    # if train_dirname != "" and not os.path.exists(train_dirname):
    #     os.makedirs(train_dirname, exist_ok=True)
    # save_to_pkl(train_data, out_train_path)
    # print('batch_train_pkl  :',len(train_data))
    # print('save_path      :',out_train_path)
    
    if val_dirname != "" and not os.path.exists(val_dirname):
        os.makedirs(val_dirname, exist_ok=True)
    save_to_pkl(val_data, out_val_path)
    print('batch_val_pkl  :',len(val_data))
    print('save_path      :',out_val_path)


if __name__ == "__main__":
    # ann_file = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval/d6_18k_train_bj_3d_d6_S1001779_20251129.pkl'
    ann_file = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval/batch/d6_hy_3d_d6_20123_20251210_val_batch_3k.pkl'
    
    # train_pkl = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/train/d6_bj_3d_d6_S1001779_20251107_train_batch_10k.pkl'
    val_pkl = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_hy_3d_d6_20123_20251210_val_batch_3k_zn_visibility.pkl'
    
    # out_all_path = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval/d6_bj_3d_d6_S1001779_20251129_trainval_batch_7k_filter_invalid.pkl'
    # out_train_path = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/train/d6_bj_3d_d6_S1001779_20251129_train_batch_10k_filter_invalid.pkl'
    out_val_path = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_hy_3d_d6_20123_20251229_val_batch_6k.pkl'

    fileter_exists_pkl(ann_file, out_val_path, val_pkl)