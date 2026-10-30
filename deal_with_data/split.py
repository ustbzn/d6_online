# import pickle
# import random
# from collections import defaultdict

# def multi_objective_split(ann_file, out_train, out_val, ratio=0.9, seed=2025):
#     """
#     三重约束的 9:1 分割：
#     1) 各类别样本数量接近 9:1
#     2) 各类别出现的帧数接近 9:1
#     3) 总体 Train/Val 帧数接近 9:1
#     """

#     random.seed(seed)

#     # ============================
#     # 1) Load infos & 按时间排序
#     # ============================
#     with open(ann_file, "rb") as f:
#         data = pickle.load(f)

#     infos = sorted(data["infos"], key=lambda x: x["timestamp"])
#     total_frames = len(infos)

#     print(f"\nTotal frames: {total_frames}")

#     # =====================================================
#     # 收集类别 -> 帧索引列表（保证帧级别划分）
#     # =====================================================
#     cls_to_frames = defaultdict(list)
#     for idx, info in enumerate(infos):
#         for cls in set(info["gt_names"]):
#             cls_to_frames[cls].append(idx)

#     # =====================================================
#     # 2) 初步按类别切分 9:1 (帧级别)
#     # =====================================================
#     train_frames = set()
#     val_frames = set()

#     for cls, frames in cls_to_frames.items():
#         frames = list(frames)
#         random.shuffle(frames)

#         n_total = len(frames)
#         n_train = int(n_total * ratio)

#         train_part = frames[:n_train]
#         val_part = frames[n_train:]

#         # 帧级别 union
#         for idx in train_part:
#             train_frames.add(idx)
#         for idx in val_part:
#             val_frames.add(idx)

#         print(f"[{cls}] frames={n_total}, train={len(train_part)}, val={len(val_part)}")

#     # =====================================================
#     # 3) 解决 train/val 交叉冲突
#     # （若某帧同时出现在 train_frames 和 val_frames，强制归 train）
#     # =====================================================
#     val_frames = val_frames - train_frames

#     # =====================================================
#     # 4) 调整总体比例到接近 9:1
#     # =====================================================
#     desired_train = int(total_frames * ratio)
#     desired_val = total_frames - desired_train

#     print("\nBefore global balance:")
#     print(f"train frames = {len(train_frames)}, val frames = {len(val_frames)}")

#     # 如果 train 太少 —— 从 val 补到 train
#     while len(train_frames) < desired_train:
#         idx = val_frames.pop()
#         train_frames.add(idx)

#     # 如果 train 太多 —— 移回 val
#     while len(train_frames) > desired_train:
#         idx = next(iter(train_frames))
#         train_frames.remove(idx)
#         val_frames.add(idx)

#     print("\nAfter global balance:")
#     print(f"train frames = {len(train_frames)}, val frames = {len(val_frames)}")

#     # =====================================================
#     # 5) 构建最终的 train_infos / val_infos
#     # =====================================================
#     train_infos = [infos[i] for i in sorted(train_frames)]
#     val_infos   = [infos[i] for i in sorted(val_frames)]

#     print("\n===== Final =====")
#     print(f"Train frames: {len(train_infos)}")
#     print(f"Val frames:   {len(val_infos)}")
#     print("=================\n")

#     # 保存
#     with open(out_train, "wb") as f:
#         pickle.dump({"infos": train_infos, "nums": len(train_infos)}, f)
#     with open(out_val, "wb") as f:
#         pickle.dump({"infos": val_infos, "nums": len(val_infos)}, f)

#     print("Saved:")
#     print(out_train)
#     print(out_val)


# if __name__ == "__main__":
#     ann_file = '/pcpt/pcpt/D6_1015/d6_12k_train_bj_3d_d6_S1001779_20251123.pkl'

#     out_train = '/pcpt/pcpt/D6_1015/d6_12k_multibalance_train.pkl'
#     out_val   = '/pcpt/pcpt/D6_1015/d6_12k_multibalance_val.pkl'

#     multi_objective_split(ann_file, out_train, out_val)

import pickle
import random
from collections import defaultdict, Counter

def multi_objective_split(ann_file, out_train, out_val, ratio=0.9, seed=2025):
    """
    三重约束的 9:1 分割：
    1) 各类别样本数量接近 9:1
    2) 各类别出现的帧数接近 9:1
    3) 总体 Train/Val 帧数接近 9:1
    """

    random.seed(seed)

    # ============================
    # 1) Load infos & 按时间排序
    # ============================
    with open(ann_file, "rb") as f:
        data = pickle.load(f)

    
    print('================pkl_len================   :',len(data['infos']))
    infos = sorted(data["infos"], key=lambda x: x["timestamp"])
    total_frames = len(infos)

    print(f"\nTotal frames: {total_frames}")

    # =====================================================
    # 收集类别 -> 帧索引列表（保证帧级别划分）
    # =====================================================
    cls_to_frames = defaultdict(list)
    for idx, info in enumerate(infos):
        for cls in set(info["gt_names"]):
            cls_to_frames[cls].append(idx)

    # =====================================================
    # 2) 初步按类别切分 9:1 (帧级别)
    # =====================================================
    train_frames = set()
    val_frames = set()

    for cls, frames in cls_to_frames.items():
        frames = list(frames)
        random.shuffle(frames)

        n_total = len(frames)
        n_train = int(n_total * ratio)

        train_part = frames[:n_train]
        val_part = frames[n_train:]

        # 帧级别 union
        for idx in train_part:
            train_frames.add(idx)
        for idx in val_part:
            val_frames.add(idx)

        print(f"[{cls}] frames={n_total}, train={len(train_part)}, val={len(val_part)}")

    # =====================================================
    # 3) 解决 train/val 交叉冲突
    # （若某帧同时出现在 train_frames 和 val_frames，强制归 train）
    # =====================================================
    val_frames = val_frames - train_frames

    # =====================================================
    # 4) 调整总体比例到接近 9:1
    # =====================================================
    desired_train = int(total_frames * ratio)
    desired_val = total_frames - desired_train

    print("\nBefore global balance:")
    print(f"train frames = {len(train_frames)}, val frames = {len(val_frames)}")

    # 如果 train 太少 —— 从 val 补到 train
    while len(train_frames) < desired_train:
        idx = val_frames.pop()
        train_frames.add(idx)

    # 如果 train 太多 —— 移回 val
    while len(train_frames) > desired_train:
        idx = next(iter(train_frames))
        train_frames.remove(idx)
        val_frames.add(idx)

    print("\nAfter global balance:")
    print(f"train frames = {len(train_frames)}, val frames = {len(val_frames)}")

    # =====================================================
    # 5) 构建最终的 train_infos / val_infos
    # =====================================================
    train_infos = [infos[i] for i in sorted(train_frames)]
    val_infos   = [infos[i] for i in sorted(val_frames)]

    print("\n===== Final =====")
    print(f"Train frames: {len(train_infos)}")
    print(f"Val frames:   {len(val_infos)}")
    print("=================\n")

    # 保存
    with open(out_train, "wb") as f:
        pickle.dump({"infos": train_infos, "nums": len(train_infos)}, f)
    with open(out_val, "wb") as f:
        pickle.dump({"infos": val_infos, "nums": len(val_infos)}, f)

    print("Saved:")
    print(out_train)
    print(out_val)

    # 调用 count_gt_distribution 进行类别统计
    print("\n===== Train Set GT Distribution =====")
    count_gt_distribution(out_train)

    print("\n===== Validation Set GT Distribution =====")
    count_gt_distribution(out_val)


def count_gt_distribution(ann_file):
    """
    统计：
    1. 所有 gt_names 的数量与百分比
    2. 每个类别出现的帧数（样本数）
    """
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)

    data_infos = sorted(data["infos"], key=lambda e: e["timestamp"])
    count_all = Counter()           # 累计目标总数
    appear_frames = defaultdict(int) # 每类出现的帧数

    # 遍历所有帧
    for data_info in data_infos:
        gt_names = data_info.get('gt_names', [])
        count_all.update(gt_names)
        for cls in set(gt_names):  # 每类只算一次帧出现
            appear_frames[cls] += 1

    total = sum(count_all.values())

    print("===== GT 类别统计 (数量 + 占比 + 帧数) =====")
    print(f"{'类别':<8} {'数量':>8} {'百分比':>10} {'帧数':>8}")
    print("---------------------------------------------")
    for k, v in count_all.most_common():
        ratio = v / total * 100
        frame_cnt = appear_frames.get(k, 0)
        print(f"{k:<15} {v:>8} {ratio:9.2f}% {frame_cnt:10d}")
    print("---------------------------------------------")
    print(f"总计: {total}  帧总数: {len(data_infos)}")
    print("=============================================")
    return count_all, appear_frames

def clip_split(filter_path, ann_file, out_train, out_val):
    with open(filter_path, "rb") as f:
        data = pickle.load(f)['infos']
        
    exsits_scenes = set([da['sample_token'] for da in data])
    exsits_scenes = list(exsits_scenes)

    with open(ann_file, "rb") as f:
        data = pickle.load(f)
    
    pkl_data = data
    scene_name = set()

    for data_i in pkl_data['infos']:
        scene_name.add(data_i['sample_token'])
    
    scene_name = list(scene_name)
    # random.shuffle(scene_name)
    
    # split_index = int(0.9 * len(scene_name))
    
    # train_sample_tokens = scene_name[:split_index]
    # val_sample_tokens = scene_name[split_index:]
    
    train_sample_tokens = [i for i in scene_name if i not in exsits_scenes]
    val_sample_tokens = [i for i in scene_name if i in exsits_scenes]
    breakpoint()

    train_data = []
    val_data = []

    for data_i in pkl_data['infos']:
        if data_i['sample_token'] in train_sample_tokens:
            train_data.append(data_i)
        elif data_i['sample_token'] in val_sample_tokens:
            val_data.append(data_i)
    
    # with open(out_train, 'wb') as f_train:
    #     pickle.dump(train_data, f_train)

    # with open(out_val, 'wb') as f_val:
    #     pickle.dump(val_data, f_val)
        
    with open(out_train, "wb") as f:
        pickle.dump({"infos": train_data}, f)
        
    with open(out_val, "wb") as f:
        pickle.dump({"infos": val_data}, f)

    print(f"数据切分完成：训练集 {len(train_data)} 条，验证集 {len(val_data)} 条。")


if __name__ == "__main__":
    filter_path = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval/batch/d6_hy_3d_d6_20123_20251210_val_batch_3k.pkl'
    ann_file = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/all_data/train_hy_3d_d6_20123_20251210_trainval_batch_15k_temporal.pkl'

    out_train = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval/batch/train_hy_3d_d6_20123_20251210_train_batch_15k_temporal.pkl'
    out_val   = '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval/batch/train_hy_3d_d6_20123_20251210_val_batch_15k_temporal.pkl'

    # multi_objective_split(ann_file, out_train, out_val)
    
    # temporal data
    clip_split(filter_path, ann_file, out_train, out_val)
