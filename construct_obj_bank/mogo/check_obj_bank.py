import os
import numpy as np

import pdb

fpath = '/data1/turbo_data/grain/datasets/mogo_final/mogo_bank'

def get_cate_dic(fpath):
    cate_dic = {}
    for cate in os.listdir(fpath):
        fpath_cate = os.path.join(fpath, cate)
        print(cate)
        for fn in os.listdir(fpath_cate):
            if fn.endswith('.npz'):
                fname_npz = os.path.join(fpath_cate, fn)
                data_dic = np.load(fname_npz)

                if cate not in cate_dic:
                    cate_dic[cate] = data_dic['size']
                else:
                    cate_dic[cate] = np.maximum(cate_dic[cate], data_dic['size'])
    
    return cate_dic


def get_big_truck_bus(fpath):
    for cate in ['truck', 'bus']:
        fpath_cate = os.path.join(fpath, cate)
        print(cate)
        for fn in os.listdir(fpath_cate):
            if fn.endswith('.npz'):
                fname_npz = os.path.join(fpath_cate, fn)
                data_dic = np.load(fname_npz)

                size = data_dic['size']
                if size.max() > 15:
                    print(size)
                    fpath_new_cate = os.path.join(fpath, "big_{}".format(cate))
                    if not os.path.exists(fpath_new_cate):
                        os.system("mkdir -p {}".format(fpath_new_cate))
                    
                    fname_new_npz = os.path.join(fpath_new_cate, fn)
                    os.system("cp {} {}".format(fname_npz, fname_new_npz))

#print(get_cate_dic(fpath))
get_big_truck_bus(fpath)