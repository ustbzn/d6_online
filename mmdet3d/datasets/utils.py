import mmcv
import numpy as np


def extract_result_dict(results, key):
    """Extract and return the data corresponding to key in result dict.

    ``results`` is a dict output from `pipeline(input_dict)`, which is the
        loaded data from ``Dataset`` class.
    The data terms inside may be wrapped in list, tuple and DataContainer, so
        this function essentially extracts data from these wrappers.

    Args:
        results (dict): Data loaded using pipeline.
        key (str): Key of the desired data.

    Returns:
        np.ndarray | torch.Tensor | None: Data term.
    """
    if key not in results.keys():
        return None
    # results[key] may be data or list[data] or tuple[data]
    # data may be wrapped inside DataContainer
    data = results[key]
    if isinstance(data, (list, tuple)):
        data = data[0]
    if isinstance(data, mmcv.parallel.DataContainer):
        data = data._data
    return data

def compute_box_3d(center, size, yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    '''
        0 -------- 1
       /|         /|
      3 -------- 2 .
      | |        | |
      . 4 -------- 5
      |/         |/
      7 -------- 6
    '''
    # 3d bounding box dimensions
    l = size[0]
    w = size[1]
    h = size[2]
    
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    return corners_3d.T

def csr2corners_batch(gt_3d_box):
    '''
    Input:
        gt_3d_box, (N, 7), 7 -> (cx, cy, cz, l, w, h, yaw)
    Output:
        gt_3d_box_corners, (N, 8, 3)
    '''
    gt_3d_box_corners = []
    for i in range(gt_3d_box.shape[0]):
        corners_3d_tmp = compute_box_3d(center=gt_3d_box[i, :3], size=gt_3d_box[i, 3:6], yaw=gt_3d_box[i, 6])
        gt_3d_box_corners.append(corners_3d_tmp)
    
    gt_3d_box_corners = np.stack(gt_3d_box_corners, axis=0)
    return gt_3d_box_corners

def quaternion2matrix(q):
    #q:list
    w,x,y,z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
             [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
             [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])
