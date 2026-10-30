import os
import yaml

import utils
import functools
import multiprocessing
from tqdm import tqdm
import pdb


bag_flag = True
if bag_flag:
    fpath_mogo = '/rtx/tuchaoping/data/bus_data/zvision_v1/'
    fpath_mogo_bank = os.path.join(fpath_mogo, 'mogo_bank')
else:
    fpath_mogo = '/rtx/tuchaoping/data/mogo2/noisedata/noise420'
    fpath_mogo_bank = '/rtx/tuchaoping/data/mogo2/noise_jira'
#class_set = set(("person", "bike", "rider", "car", "truck", "bus", "obstacle", "traffic_sign", "noise"))
class_set = set(("person", "bike", "rider", "car", "truck", "bus", "obstacle", "noise"))

for cate in class_set:
    fpath_cate = os.path.join(fpath_mogo_bank, cate)
    os.system("mkdir -p {}".format(fpath_cate))


with open('../../datasets/mogo/falcon.yaml', 'r') as f:
    task_cfg = yaml.load(f, Loader=yaml.Loader)


common_infos=dict(class_set=class_set, fpath_mogo_bank=fpath_mogo_bank, task_cfg=task_cfg)

flist = []
if bag_flag:
    bag_folder = os.path.join(fpath_mogo, "train")
    for bag_id in os.listdir(bag_folder):
        # if not bag_id.startswith('Det'):
        #     continue
        fpath_bag = os.path.join(bag_folder, bag_id)
        for fn in os.listdir(fpath_bag):
            if fn.endswith('.npz'):
                fname_npz = os.path.join(fpath_bag, fn)
                flist.append(fname_npz)
else:
    bag_folder = fpath_mogo
    for fn in os.listdir(fpath_mogo):
        if fn.endswith('.npz'):
            fname_npz = os.path.join(fpath_mogo, fn)
            flist.append(fname_npz)


print("Total Length: ", len(flist))
# multi-process
pbar = tqdm(total=len(flist))
pbar.set_description("processing")
update = lambda *args: pbar.update()
pool1 = multiprocessing.Pool(processes = 32)

#for fname_npz in flist:
#    pool1.apply_async(utils.process_single, (fname_npz, common_infos), callback=update)
pool1.map(functools.partial(utils.process_single, common_infos=common_infos), flist)
pool1.close()
pool1.join()
