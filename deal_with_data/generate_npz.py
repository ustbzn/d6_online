import os
import numpy as np
import struct
import argparse
import json
import glob
import re
from collections import defaultdict
from tqdm import tqdm

# def parse_sensor_config_improved(config_text):
#     config_dict = {}
#     lines = [line.strip() for line in config_text.split('\n') if line.strip()]
#     if lines[0].startswith('base:'):
#         config_dict['base'] = lines[0].split('"')[1]
#         idx = 1
#     else:
#         idx = 0

#     config_dict['sensor_info'] = []

#     while idx < len(lines):
#         if lines[idx] == 'sensor_info {':
#             sensor_dict, idx = parse_sensor_block(lines, idx + 1)
#             config_dict['sensor_info'].append(sensor_dict)
#         else:
#             idx += 1

#     return config_dict


def parse_sensor_config_improved(config_text, calibrated_sensor_dict):
    config_dict = {}
    for sensor in config_text:
        if "fish" in sensor:
            continue

        sensor_name = calibrated_sensor_dict[sensor]['npz_key']
        type = calibrated_sensor_dict[sensor]['type']
        orientation = calibrated_sensor_dict[sensor]['orientation']
        height = calibrated_sensor_dict[sensor]['height']
        width = calibrated_sensor_dict[sensor]['width']
        extrinsic_matrix = np.array(config_text[sensor]['cam_to_lidar'], dtype=np.float64).reshape(4, 4)
        fx, fy, cx, cy = config_text[sensor]['fx'], config_text[sensor]['fy'], config_text[sensor]['cx'], config_text[sensor]['cy']
        intrinsic_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float64) 

        config_dict[sensor_name] = {
            'type': type,
            'orientation': orientation,
            'height': height,
            'width': width,
            'intrinsic': intrinsic_matrix,
            'distort': np.array([]),
            'extrinsic': extrinsic_matrix, 
            'sensor2lidar_rotation': extrinsic_matrix[:3, :3] ,
            'sensor2lidar_translation': extrinsic_matrix[:3, 3]
        }

    return config_dict


def parse_sensor_block(lines, start_idx):
    sensor_dict = {}
    idx = start_idx

    while idx < len(lines) and lines[idx] != '}':
        line = lines[idx].strip()
        if '{' in line and '}' not in line:
            field_name = line.split('{')[0].strip()
            nested_dict, idx = parse_nested_block(lines, idx + 1)
            sensor_dict[field_name] = nested_dict
            continue

        if ':' in line:
            key, value = [part.strip() for part in line.split(':', 1)]
            sensor_dict[key] = parse_value(value)

        idx += 1

    return sensor_dict, idx + 1 

def parse_nested_block(lines, start_idx):
    nested_dict = {}
    idx = start_idx
    
    matrix_i = 0
    distort_matrix_i = 0

    while idx < len(lines) and lines[idx] != '}':
        line = lines[idx].strip()

        if '{' in line and '}' not in line:
            field_name = line.split('{')[0].strip()
            deeper_dict, idx = parse_nested_block(lines, idx + 1)
            nested_dict[field_name] = deeper_dict
            continue

        if ':' in line:
            key, value = [part.strip() for part in line.split(':', 1)]

            if key in nested_dict and isinstance(nested_dict[key], list):
                nested_dict[key].append(parse_value(value))
            # elif key in ['matrix', 'distort_matrix']:
            #     breakpoint()
            #     nested_dict[key] = [parse_value(value)]
            elif key == 'matrix':
                if matrix_i != 0:
                    key = key + '_{}'.format(matrix_i)
                nested_dict[key] = parse_value(value)
                matrix_i += 1
                
            elif key == 'distort_matrix':
                if distort_matrix_i != 0:
                    key = key + '_{}'.format(distort_matrix_i)
                nested_dict[key] = parse_value(value)
                distort_matrix_i += 1

            else:
                nested_dict[key] = parse_value(value)

        idx += 1

    return nested_dict, idx + 1

def parse_value(value):
    if '#' in value:
        value = value.split('#')[0].strip()

    if value.startswith('"') and value.endswith('"'):
        return value.strip('"')

    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass

    return value

def convert_to_structured_bboxes(objects_list):
        bboxes = []
        group = []
        ObjectID = []
        label = []
        sublabel = []
        pointnum = []
        is_excessive_layering = []
        is_camera_based = []
        
        for obj in objects_list:
            center = obj['3Dcenter']
            size = obj['3Dsize']
            bbox = [
                center['x'], center['y'], center['z'],
                size['length'], size['width'], size['height'],
                size.get('alpha', 0)
            ]
            bboxes.append(bbox)
            group.append(obj['group'])
            ObjectID.append(obj['ObjectID'])
            label.append(obj['label'])
            sublabel.append(obj['sublabel'])
            pointnum.append(obj['pointnum'])
            is_excessive_layering.append(obj['is_excessive_layering'])
            is_camera_based.append(obj['is_camera_based'])
    
        return (bboxes, group, ObjectID, label, sublabel, pointnum, is_excessive_layering, is_camera_based)

def read_json(file_path):
    with open(file_path, 'rb') as f:
        data = json.load(f)
        json_data = dict()
        for k, v in data.items():
            if k == 'data':
                json_data['3d_url'] = v['3d_url']
            if k == 'nCloud':
                json_data[k] = v
            if k == 'result':
                obj_list = convert_to_structured_bboxes(v['data'])
                json_data['bboxes'] = obj_list[0]
                json_data['group'] = obj_list[1]
                json_data['ObjectID'] = obj_list[2]
                json_data['label'] = obj_list[3]
                json_data['sublabel'] = obj_list[4]
                json_data['pointnum'] = obj_list[5]
                json_data['is_excessive_layering'] = obj_list[6]
                json_data['is_camera_based'] = obj_list[7]
                
    return json_data   
        
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
            raise ValueError(f"不支持的DATA格式: {metadata['DATA']}")

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
    
def read_pose_txt(txt_pth):
    matrixes = dict()
    with open(txt_pth, 'rb') as f:
        for line in f:
            numbers = line.decode('utf-8').strip().split()
            time_stamp = numbers[0]
            pose_matrix = np.array([float(num) for num in numbers[1:]]).reshape(4,4)
            matrixes[time_stamp] = pose_matrix
    
    return matrixes

def save_npz(path, f_dict):
    try:
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        np.savez(path, **f_dict)
        return True
    except Exception as e:
        print(f"保存失败: {str(e)}")
        print(f"【ERROR】: {path} 文件保存失败！】")
        assert 0
        return False
    
def print_tqdm(total_bags_num, path):
    if total_bags_num % 100 == 0:
        print('bag_num: ', total_bags_num, ' bag_name: ', path)

def sort_by_timestamp(jpg_list):
    """按文件名中的时间戳数值排序路径列表"""
    def get_timestamp(path):
        # 提取文件名（如 "1754548662.200383186.jpg"）
        filename = os.path.basename(path)
        # 提取时间戳部分（去除 ".jpg"）并转换为浮点数
        timestamp_str = filename.split('.jpg')[0]
        return float(timestamp_str)  # 按数值排序
    
    # 按时间戳数值升序排序
    return sorted(jpg_list, key=get_timestamp)
        
class RUN:
    def __init__(self,is_manual_data, src_path, dst_path, is_split, refresh):
        self.is_manual_data = is_manual_data
        self.is_split = is_split
        self.refresh = refresh
        self.src_path = src_path
        self.dst_path = dst_path
        self.manual_filename = ['all_data', 'train', 'val']
        self.orin_filename = ['all_frames', 'result', 'send']
        # self.calibrated_sensor_dict = {
        #     'image_30':        {'npz_key':'sensing_30_front',           'type':'SENSING_30', 'orientation':'FRONT',        },
        #     'image_60':        {'npz_key':'sensing_60_front',           'type':'SENSING_60', 'orientation':'FRONT'},
        #     'image_120':       {'npz_key':'sensing_120_front',          'type':'SENSING_120','orientation':'FRONT'},
        #     'image_back':      {'npz_key':'sensing_120_rear',           'type':'SENSING_120','orientation':'REAR'},
        #     'image_left':      {'npz_key':'sensing_120_left',           'type':'SENSING_120','orientation':'LEFT'},
        #     'image_right':     {'npz_key':'sensing_120_right',          'type':'SENSING_120','orientation':'RIGHT'},
        #     'image_left_back': {'npz_key':'sensing_120_left_backward',  'type':'SENSING_120','orientation':'LEFT_BACKWARD'},
        #     'image_right_back':{'npz_key':'sensing_120_right_backward', 'type':'SENSING_120','orientation':'RIGHT_BACKWARD'}
        # }
        self.calibrated_sensor_dict = {
            '60_front' :       {'npz_key':'sensing_60_front',              'type':'SENSING_60', 'orientation':'FRONT',          'height' :2160, 'width' :3840},
            '120_front' :      {'npz_key':'sensing_120_front',             'type':'SENSING_120','orientation':'FRONT',          'height' :1536, 'width' :1920},
            '120_back' :       {'npz_key':'sensing_120_rear',              'type':'SENSING_120','orientation':'REAR',           'height' :1536, 'width' :1920},
            'left_back' :      {'npz_key':'sensing_120_left_backward',     'type':'SENSING_120','orientation':'LEFT_BACKWARD',  'height' :1536, 'width' :1920},
            'right_back' :     {'npz_key':'sensing_120_right_backward',    'type':'SENSING_120','orientation':'RIGHT_BACKWARD', 'height' :1536, 'width' :1920},
            'left_front' :     {'npz_key':'sensing_120_left',              'type':'SENSING_120','orientation':'LEFT',           'height' :1536, 'width' :1920},
            'right_front' :    {'npz_key':'sensing_120_right',             'type':'SENSING_120','orientation':'RIGHT',          'height' :1536, 'width' :1920},
            'left_front_60' :  {'npz_key':'sensing_60_left_front',         'type':'SENSING_60','orientation':'LEFT_FRONT',      'height' :1536, 'width' :1920},
            'right_front_60' : {'npz_key':'sensing_60_right_front',        'type':'SENSING_60','orientation':'RIGHT_FRONT',     'height' :1536, 'width' :1920}

        }
        # self.camera_dict = {
        #     '30_front'  :'image_30',        
        #     '60_front'  :'image_60',       
        #     '120_front' :'image_120',     
        #     '120_back'  :'image_back', 
        #     '120_left'  :'image_left', 
        #     '120_right' :'image_right',
        #     'left_back' :'image_left_back',
        #     'right_back':'image_right_back'   
        # }
    
        self.camera_dict = {
            '60_front' : '60_front',
            '120_front' : '120_front',
            '120_back' : '120_back',
            'left_back' : 'left_back',
            'right_back' : 'right_back',
            'left_front' : 'left_front',
            'right_front' : 'right_front',
            'left_front_60' : 'left_front_60',
            'right_front_60' : 'right_front_60'
        }
       
        # if is_manual_data:
        #     # self.dst_path = os.path.join(dst_path, 'data_manual_annotation_npz')
        #     self.create_result_path(self.dst_path, self.manual_filename)
        # else:
        #     self.dst_path = os.path.join(dst_path, 'data_orin_npz')

    def create_result_path(self, dst_path, manual_filename):
        for dirname in manual_filename:
            current_path = os.path.join(dst_path, dirname)
            try:
                if not os.path.exists(current_path):
                    os.makedirs(current_path)
            except OSError as e:
                raise OSError(f"Failed to create directory '{current_path}': {e}")
            
    def run(self):
        total_bags_num = 0
        # data_250408/data_orin/train_bj_20241206_JLBJA20073D_b2_2024_Z_V4
        # orin_batch_list = os.listdir(self.src_path)
        # for batch_name in orin_batch_list:
        #     orin_batch_path = os.path.join(self.src_path, batch_name)
        # orin_all_frames_path = os.path.join(orin_batch_path, 'all_frames')
        orin_result_path = os.path.join(self.src_path, 'result')
        orin_send_path = os.path.join(self.src_path, 'send')
        batch = self.src_path.split('/')[-2]  # train_bj_C3_1001386_seg_20250425
        # train_bj_C3_1001386_seg_20250425_1
        orin_subbatch_list = os.listdir(orin_result_path)
        for orin_subbatch_name in orin_subbatch_list:
            orin_subbatch_path = os.path.join(orin_result_path, orin_subbatch_name)
            # 20250423163627-20250423163657-1745397001035514395-full-R1001386_00
            
            for bagname in os.listdir(orin_subbatch_path):
                # print(orin_subbatch_path)
                current_npz_path = os.path.join(self.dst_path, batch, bagname)
                # if orin_subbatch_path != "/pcpt/pcpt/D6_1015/data/train_bj_3d_d6_S1001779_20251024/result/train_bj_3d_d6_s1001779_20251014_01_2_1" 
                # or bagname != "20251014131848-1760419128117913313-full-S1001779":
                #     continue
                print(f"bagname {bagname}")
                print(f"ori_subbatch_path: {orin_subbatch_path} ")

                if self.refresh and os.path.exists(current_npz_path):
                    print(f"[DEBUG] 跳过已存在的文件夹: {current_npz_path}")
                    continue
                result_bag_path = os.path.join(orin_subbatch_path, bagname)
                send_bag_path  = os.path.join(orin_send_path, bagname)
                if not os.path.exists(send_bag_path):
                    continue

                # all_frames_path = os.path.join(orin_all_frames_path, bagname)
                # all json pcd list

                # 保证标注的json和pcd的一致性，通过时间戳过滤一遍，保证能够按url的索引找到对应的json
                # all_json = sorted(glob.glob(result_bag_path+'/result_json/*.json'))
                # all_json_ts = [os.path.basename(json).split(".json")[0] for json in all_json]
                # all_pcd = sorted(glob.glob(result_bag_path+'/result_pcd/*.pcd'))
                all_3d_url = sorted(glob.glob(send_bag_path+'/3d_url/*.pcd'))
                all_3d_url_ts =  [os.path.basename(url).split(".pcd")[0] for url in all_3d_url]

                # url_json_common_ts = sorted(set(all_json_ts) & set(all_3d_url_ts))
                # all_json = [json for json in all_json if os.path.basename(json).split(".json")[0] in url_json_common_ts]
                # all_3d_url = [url for url in all_3d_url if os.path.basename(url).split(".pcd")[0] in url_json_common_ts]
                # assert len(all_json) == len(all_3d_url)
                
                # if len(all_json) != len(all_pcd):
                #     print(f"[DEBUG] 数量不一致: all_json={len(all_json)}, all_pcd={len(all_pcd)}")
                    
                #     # 提取jpg时间戳集合
                #     json_stamps = {path.split('/')[-1].replace('.json', '') for path in all_json}
                #     # 提取pcd时间戳集合
                #     pcd_stamps = {path.split('/')[-1].replace('.pcd', '') for path in all_pcd}
                    
                #     # 找出缺失的时间戳
                #     missing_in_pcd = json_stamps - pcd_stamps  # 在json中存在但pcd中缺失的
                #     missing_in_json = pcd_stamps - json_stamps  # 在pcd中存在但json中缺失的
                    
                #     # 打印详细调试信息
                #     if missing_in_pcd:
                #         print(f"[DEBUG] {len(missing_in_pcd)}个时间戳在all_pcd中缺失:")
                #         # 创建新的json_list，只保留在pcd中有对应项的时间戳
                #         new_json_list = []
                #         for stamp in missing_in_pcd:
                #             # 查找对应的json路径
                #             new_json_list = [item for item in all_json if stamp not in item]
                #             json_path = next(p for p in all_json if stamp in p)
                #             print(f"  - 时间戳: {stamp}")
                #             print(f"    JSON路径: {json_path}")
                #         all_json = new_json_list
                        
                #     if missing_in_json:
                #         print(f"[DEBUG] {len(missing_in_json)}个时间戳在all_json中缺失:")
                #         # 创建新的pcd_list，只保留在json中有对应项的时间戳
                #         new_pcd_list = []
                #         for stamp in missing_in_json:
                #             # 查找对应的pcd路径
                #             new_pcd_list = [item for item in all_pcd if stamp not in item]
                #             pcd_path = next(p for p in all_pcd if stamp in p)
                #             print(f"  - 时间戳: {stamp}")
                #             print(f"    PCD路径: {pcd_path}")
                #         all_pcd = new_pcd_list
                
                # # if manual files num < 2
                # if len(all_pcd) < 2:
                #     print(result_bag_path)
                #     continue

                if self.is_manual_data:
                    # read_pose_txt 形成每帧的pose
                    camera_name_list = [name for name in os.listdir(send_bag_path) if name.split('.')[-1] != 'txt' and name.split('.')[-1] != 'json' and name != '3d_url']
                    pose_txt = os.path.join(send_bag_path, 'pose.txt')
                    pose_data_dict = read_pose_txt(pose_txt)
                    
                    # # read_calibrated_sensor_txt
                    # calibrated_sensor_pb_txt = os.path.join(send_bag_path, 'calibrated_sensor.pb.txt')
                    # with open(calibrated_sensor_pb_txt, "r") as f:
                    #     config_text = f.read()
                    # sensor_dict = parse_sensor_config_improved(config_text)

                    # calibrated_sensor_pb_txt = os.path.join(send_bag_path, 'calibrated_sensor.pb.txt')
                    # with open(calibrated_sensor_pb_txt, "r") as f:
                    #     config_text = f.read()
                    # sensor_dict = parse_sensor_config_improved(config_text)
                    calibrated_sensor_pb_txt = os.path.join(send_bag_path, 'calib.json')
                    with open(calibrated_sensor_pb_txt, "r") as f:
                        config_text = json.load(f)
                    sensor_dict = parse_sensor_config_improved(config_text, self.calibrated_sensor_dict)
                    
                    # camera_jpg_list 这里也要保证和标注文件中的时间戳保持一致
                    camera_path_dict = dict()
                    for camera in camera_name_list:
                        camera_i = os.path.join(send_bag_path, camera)
                        valid_cameras = [camera_path for camera_path in sorted(glob.glob(camera_i + '/*.jpg')) if os.path.basename(camera_path).split(".jpg")[0] in all_3d_url_ts]
                        assert [os.path.basename(url).split(".pcd")[0] for url in all_3d_url] == [os.path.basename(valid_cameras).split(".jpg")[0] for valid_cameras in valid_cameras]
                        camera_path_dict[camera] = valid_cameras
                    
                    # calibrated_data_dict 这里就是找每个相机的内外参及畸变系数即可，后续可以根据需求进行修改
                    # calibrated_data_dict = dict()
                    # for camera_name in camera_path_dict.keys():
                    #     calibrated_data = self.calibrated_sensor_dict[self.camera_dict[camera_name]]
                    #     for single_sensor in sensor_dict['sensor_info']:
                    #         if calibrated_data['type'] == single_sensor['type'] and calibrated_data['orientation'] == single_sensor['orientation']:
                    #             calibrated_data_dict[calibrated_data['npz_key']] = single_sensor
                                
                    # assert len(calibrated_data_dict) == len(camera_path_dict)
                    
                    for camera_name in camera_path_dict.keys():
                        jpg_list = camera_path_dict[camera_name]
                            
                        # if len(jpg_list) != len(all_pcd):
                        #     print(f"[DEBUG] 数量不一致: jpg_list={len(jpg_list)}, all_pcd={len(all_pcd)}")
                            
                        #     # 提取jpg时间戳集合
                        #     jpg_stamps = {path.split('/')[-1].replace('.jpg', '') for path in jpg_list}
                        #     # 提取pcd时间戳集合
                        #     pcd_stamps = {path.split('/')[-1].replace('.pcd', '') for path in all_pcd}
                            
                        #     # 找出缺失的时间戳
                        #     missing_in_pcd = jpg_stamps - pcd_stamps  # 在jpg中存在但pcd中缺失的
                        #     missing_in_jpg = pcd_stamps - jpg_stamps  # 在pcd中存在但jpg中缺失的
                            
                        #     # 打印详细调试信息
                        #     if missing_in_pcd:
                        #         print(f"[DEBUG] {len(missing_in_pcd)}个时间戳在all_pcd中缺失:")
                        #         # 创建新的jpg_list，只保留在pcd中有对应项的时间戳
                        #         new_jpg_list = []
                        #         for stamp in missing_in_pcd:
                        #             # 查找对应的jpg路径
                        #             new_jpg_list = [item for item in jpg_list if stamp not in item]
                        #             jpg_path = next(p for p in jpg_list if stamp in p)
                        #             print(f"  - 时间戳: {stamp}")
                        #             print(f"    JPG路径: {jpg_path}")
                                
                        #         # print(f"[DEBUG] 新的jpg_list长度: {len(new_jpg_list)}")
                        #         # print(f"[DEBUG] 原jpg_list: {jpg_list}")
                        #         jpg_list = new_jpg_list  # 更新jpg_list为新的列表
                            
                        #     if missing_in_jpg:
                        #         print(f"[DEBUG] {len(missing_in_jpg)}个时间戳在jpg_list中缺失:")
                        #         for stamp in missing_in_jpg:
                        #             # 查找对应的pcd路径
                        #             pcd_path = next(p for p in all_pcd if stamp in p)
                        #             print(f"  - 时间戳: {stamp}")
                        #             print(f"    PCD路径: {pcd_path}")
                        #             new_jpg_stamp_parts = jpg_list[0].split('/')
                        #             new_jpg_stamp_parts[-1] = stamp + '.jpg'
                        #             new_jpg_path = '/'.join(new_jpg_stamp_parts)
                        #             jpg_list.append(new_jpg_path)  # 添加新的jpg路径
                        #             jpg_list = sort_by_timestamp(jpg_list)  # 重新排序
                        #             for idx, jpg_path in enumerate(jpg_list):
                        #                 print(f"    新的JPG路径: {jpg_path}")

                    # save per npz files


                    for index, pcd_path in enumerate(all_3d_url):
                        final_npz_dict = dict()
                        # print(f"[DEBUG] 处理第 {index+1}/{len(all_pcd)} 个 PCD 文件: {pcd_path}")
                        # print(f"[DEBUG] 对应的 ori_pcd 文件: {all_3d_url[index]}")

                        time_stamp = pcd_path.split('/')[-1].replace('.pcd','')
                        pcd_data = read_pcd(pcd_path)
                        pcd_attribute = pcd_data[-1]
                        # print(pcd_data[-1])
                        
                        # 3d_url_path原始点云获取lidar id
                        # ori_pcd_path = all_3d_url[index]
                        # ori_pcd_data = read_pcd(ori_pcd_path)
                        # ori_pcd_data_attribute = ori_pcd_data[-1]
                        # id_idx = ori_pcd_data_attribute.index('id')
                        # lidar_id = ori_pcd_data[0][:, id_idx].astype(np.uint8)
                        # lidar_id = lidar_id.reshape(-1, 1)  # -1表示自动计算行数
                        
                        x_idx = pcd_attribute.index('x')
                        y_idx = pcd_attribute.index('y')
                        z_idx = pcd_attribute.index('z')
                        intensity_idx = pcd_attribute.index('intensity')
                        # label_idx = pcd_attribute.index('label')
                        
                        if 'id' in pcd_attribute:
                            id_idx = pcd_attribute.index('id')
                            lidar_id = pcd_data[0][:, id_idx].astype(np.uint8)
                        else:
                            lidar_id_idx = pcd_attribute.index('lidar_id')
                            lidar_direction_idx = pcd_attribute.index('lidar_direction')
                            lidar_id = pcd_data[0][:, lidar_id_idx:lidar_direction_idx+1].astype(np.uint8)
                        
                        pcdi = pcd_data[0][:,[x_idx,y_idx,z_idx,intensity_idx]]
                        # pcd_label = pcd_data[0][:, label_idx].astype(np.int64)
                        final_npz_dict['pcd'] = pcdi
                        final_npz_dict['pcds_timestamp'] = time_stamp
                        # final_npz_dict['pcd_label'] = pcd_label
                        final_npz_dict['pcds_lidarid'] = lidar_id
                        
                        for camera_name in camera_path_dict.keys():
                            npz_camera_name = self.camera_dict[camera_name]
                            
                            # assert len(jpg_list) == len(all_pcd)
                            jpg_name = jpg_list[index]
                            jpg_time_stamp = jpg_name.split('/')[-1].replace('.jpg','')
                            
                            if jpg_time_stamp != time_stamp:
                                print(f"[DEBUG] 时间戳不匹配: jpg_time_stamp={jpg_time_stamp}, time_stamp={time_stamp}")
                            # assert jpg_time_stamp == time_stamp
                            final_npz_dict[npz_camera_name] = jpg_name
                                
                        # assert len(all_pcd) == len(all_json)
                        # json_data_dict = read_json(all_json[index])
                        final_npz_dict['bboxes'] = np.array([], dtype=np.float32)
                        final_npz_dict['bboxes_label'] = np.array([], dtype=np.int64) // 1000
                        final_npz_dict['bboxes_sublabel'] = np.array([], dtype=np.int64)
                        final_npz_dict['objects_id'] = np.array([], dtype=np.int64)
                        final_npz_dict['group'] = np.array([], dtype=np.int64)
                        
                        # # updata calibrated_data_dict in final_npz_dict
                        final_npz_dict.update(sensor_dict)
                        
                        assert time_stamp in pose_data_dict.keys()
                        final_npz_dict['transform_mat'] = pose_data_dict[time_stamp]
                        final_npz_dict['position'] = pose_data_dict[time_stamp][:,-1].reshape(-1)[:3]
                        
                        batch_name = pcd_path.split('/')[-5] # train_bj_20241206_JLBJA20073D_b2_2024_Z_V4
                        bag_name =  pcd_path.split('/')[-3]  # 20241205125413-20241205125423-1733368729087458033-full-JLBJA20073D_04
                        npz_name = pcd_path.split('/')[-1].replace('.pcd','.npz') # 1733374462.700472593.npz
                        
                        # all_data_save_path = os.path.join(self.dst_path, 'all_data', batch_name, bag_name, npz_name)
                        all_data_save_path = os.path.join(self.dst_path, batch_name, bag_name, npz_name)
                        save_npz(all_data_save_path, final_npz_dict)

                        if self.is_split:
                            if index == 0:
                                train_val_save_path = os.path.join(self.dst_path, 'val', batch_name, bag_name, npz_name)
                            else:
                                train_val_save_path = os.path.join(self.dst_path, 'train', batch_name, bag_name, npz_name)
                            save_npz(train_val_save_path, final_npz_dict)
                        
                        total_bags_num += 1 
                        print_tqdm(total_bags_num, bag_name)


def parse_args():
    parser = argparse.ArgumentParser(description='data_process')
    # parser.add_argument('--is_manual_data', type=bool, default=store)
    # parser.add_argument('--is_split', type=bool, default=store)
    parser.add_argument('--is_manual_data', action='store_true', default=True)
    parser.add_argument('--is_split', action='store_true', default=False)
    parser.add_argument('--refresh', action='store_true', default=False)
    
    parser.add_argument('--src_path', type=str, default='/pcpt/pcpt/D6_1015/data/test_1025_1027_bag')
    parser.add_argument('--dst_path', type=str, default='/pcpt/pcpt/D6_1015/data/npz_test')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    import time
    start_time = time.time()  # 记录开始时间
    
    args = parse_args()
    runner = RUN(args.is_manual_data, args.src_path, args.dst_path, args.is_split, args.refresh)
    runner.run()
    
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算耗时
    print(f"代码执行耗时: {elapsed_time:.6f} 秒")
