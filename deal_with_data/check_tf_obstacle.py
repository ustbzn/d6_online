import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

path_pkl = '/pcpt/pcpt/D6_1015/pkls/detection//train_hy_3d_d6_20123_20260114_train.pkl'

# ================== Load ==================
data = pickle.load(open(path_pkl, 'rb'))
infos = data['infos']

flag_name = set()
lock = Lock()   # 保护共享 set

def process_info(info):
    filter_name = info['data_path'].split('/')[-3]
    if 'tfc_obstacle' in info['gt_names']:
        return filter_name
    return None

num_workers = 8  # 可按 CPU 调整

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(process_info, info) for info in infos]

    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            with lock:
                flag_name.add(result)

print(flag_name)
print(f"Original number of infos: {len(infos)}")
