import pickle as pkl
from collections import defaultdict, Counter

def pretty_print_scene_counter(scene_counter, max_scenes=5):
    """
    打印 scene_counter：
    scene_id
      └── category
            └── sub_id: count
    """
    print('\n========== Scene Counter ==========')

    for i, (scene_id, cat_dict) in enumerate(scene_counter.items()):
        print(f'\n🟦 Scene: {scene_id}')
        for category, counter in cat_dict.items():
            total = sum(counter.values())
            print(f'  ├─ {category} (total={total})')

            for sid, cnt in counter.most_common():
                print(f'  │    └─ id={sid}: {cnt}')

        if max_scenes and i + 1 >= max_scenes:
            print('\n... (truncated)')
            break

    print('\n==================================\n')


data_path = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/d6_bj_3d_d6_S1001779_20251203_val_all_1k_0116_manlabel_model_np123.pkl'

with open(data_path, 'rb') as f:
    data_list = pkl.load(f)

# scene_id -> category -> Counter(id -> count)
scene_counter = defaultdict(lambda: defaultdict(Counter))

for data in data_list['infos']:
    scene_id = data.get('scene_id', data.get('token', 'unknown_scene'))
    raw_ids = data['scene_tags'].get('raw_ids', {})

    if not isinstance(raw_ids, dict):
        continue

    for category, values in raw_ids.items():
        if values is None:
            continue

        if isinstance(values, (list, tuple, set)):
            sub_ids = values
        else:
            sub_ids = [values]

        for sid in sub_ids:
            scene_counter[scene_id][category][sid] += 1
            
pretty_print_scene_counter(scene_counter, max_scenes=10)
# ========== 汇总成 category -> set(ids) ==========
metric_ids = defaultdict(set)

for _, cat_dict in scene_counter.items():
    for category, counter in cat_dict.items():
        for sid, cnt in counter.items():
            if cnt > 0:   # 可改成 >=10 做过滤
                metric_ids[category].add(sid)

metric_ids = {k: sorted(list(v)) for k, v in metric_ids.items()}

# ========== 生成 bash 配置 ==========
out_file = 'metric_splits.sh'
with open(out_file, 'w') as f:
    f.write('#!/usr/bin/env bash\n\n')
    for metric, ids in metric_ids.items():
        ids_str = ' '.join(str(i) for i in ids)
        f.write(f'{metric.upper()}_IDS=({ids_str})\n')

print('Generated metric_splits.sh')
for k, v in metric_ids.items():
    print(f'{k}: {v}')
