#!/usr/bin/python
# -*- coding: UTF-8 -*-
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle
import numpy as np


npz_path = '/pcpt/pcpt/data/B22024/0324_0701/B22025data_v1/val/20073D|20241206144558-20241206144608-1733455227220346176-full-JLBJA20073D_01/3d_npz_new/1733467560.300474166.npz'
obj = np.load(npz_path, allow_pickle=True)
import pdb; pdb.set_trace()

def read_pcd(file_path):
    with open(file_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line.startswith('DATA'):
                break

        metadata = {}
        for line in header:
            if line.startswith('#') or line == '':
                continue
            key, value = line.split(' ', 1)
            metadata[key] = value

        if metadata['DATA'].lower() not in ['ascii', 'binary']:
            raise ValueError(f"type is error with: {metadata['DATA']}")

        points_num = int(metadata['POINTS'])
        fields = metadata['FIELDS'].split()
        size = int(metadata['WIDTH']) * int(metadata['HEIGHT'])

        if metadata['DATA'].lower() == 'ascii':
            data = np.loadtxt(f, dtype=np.float64, max_rows=points_num)
        else:
            dtype = np.dtype([(f, np.float64) for f in fields])
            data = np.fromfile(f, dtype=dtype, count=points_num)
            data = np.array([list(row) for row in data])

        return data, fields

def process_info(value):
    for k, v in value.items():
        if k == 'data_path':
            if v.endswith('.npz') and ('B22025data_v1' in v or 'B22025data_v2' in v):
                pcd_path = v.replace('.npz', '.pcd').replace('3d_npz', '3d_url')
                npz_new_path = v.replace('3d_npz', '3d_npz_new')
                npz_new_dir = Path(npz_new_path).parent
                npz_new_dir.mkdir(parents=True, exist_ok=True)
                if Path(pcd_path).exists():
                    points_pcd, fields = read_pcd(pcd_path)
                    points_pcd_lidarid = None 
                    if points_pcd.shape[1] == 6:
                        points_pcd_lidarid = points_pcd[:, 5]
                    elif points_pcd.shape[1] == 7:
                        points_pcd_lidarid = points_pcd[:, 5:7]
                    if not Path(v).exists():
                        print(f"NPZ file {v} does not exist, skipping.")
                        return
                    points_npz = np.load(v, allow_pickle=True)
                    points_pcd_lidarid = points_pcd_lidarid.astype(np.uint8)
                    points_npz = dict(points_npz)
                    points_npz['pcds_lidarid'] = points_pcd_lidarid
                    with open(npz_new_path, 'wb') as f_new:
                        np.savez(f_new, **points_npz)
                else:
                    print(f"PCD file {pcd_path} does not exist, skipping.")

pkl_path = Path('/pcpt/pcpt/data/B22024/0324_0701/B22025data_v3/filter_B2_infos_250731allmerge_val.pkl')
with pkl_path.open('rb') as f:
    obj = pickle.load(f)
    import pdb; pdb.set_trace()
    if 'infos' in obj:
        values = obj['infos']
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(process_info, value) for value in values]
            for future in as_completed(futures):
                future.result()