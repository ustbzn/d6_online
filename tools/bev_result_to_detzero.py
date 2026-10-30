import numpy as np
import pickle as pkl
import os

save_path = '/pcpt/pcpt/project/atlb/data/'
#data_path = '/pcpt/pcpt/project/atlb/model/multimodal_bevfusion/experiments/v30_B2_second1231_3dseg_cls21_cam5lidar_intensityraw_for_lyp_test/results.pkl'
data_path = '/pcpt/pcpt/project/atlb/model/multimodal_bevfusion/experiments/v30_B2_second1231_3dseg_cls21_cam5lidar_intensityraw/latest.pth_results.pkl'
point_cloud_range = [-81.6, -48, -1, 97.6, 48, 3.0]


def save_npz(npz_path, output):
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, **output)
    
with open(data_path,'rb') as f:
    dataes = pkl.load(f)
    
for index, data in enumerate(dataes):
    output = {
        'pred_points_labels': data['pred_points_labels'].numpy(),
        'seg_logits': data['seg_logits'].numpy(),
        'gt_points_labels': data['gt_points_labels'].numpy(),
        'pcds': data['pcds'].numpy(),
        'boxes_3d': data['boxes_3d'].tensor.numpy(),
        'scores_3d': data['scores_3d'].numpy(),
        'labels_3d': data['labels_3d'].numpy(),
        'gt_bboxes_3d': data['gt_bboxes_3d'].tensor.numpy(),
        'gt_labels_3d': data['gt_labels_3d'].numpy(),
        'gt_ids': data['gt_ids'].numpy(),
    }
    npz_data = np.load(data['lidar_path'])
    pose = npz_data['pose']
    output['pose'] = pose
    
    if len(output['gt_bboxes_3d'])==0:
        breakpoint()
    
    lidar_path = data['lidar_path']
    dir_name = lidar_path.split('/')[-3].split('|')[-1]
    file_name = lidar_path.split('/')[-1]

    save_npz_name = os.path.join(save_path, dir_name, file_name)

    # np_output = {k: v.cpu().numpy() if hasattr(v, 'numpy') else v 
    #         for k, v in output.items()}
    save_npz(save_npz_name, output)
