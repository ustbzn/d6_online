import mmcv
import numpy as np
from glob import glob
import os
from os.path import join
from os import path as osp
import shutil
import json
import copy
import collections
from pypcd import pypcd
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from mmdet3d.datasets import csr2corners_batch, quaternion2matrix
from .pick_data import get_pointnums


class Mogo2KITTI(object):
    """
    Args:
        load_dir (str): Directory to load waymo raw data.
        prefix (str): Prefix of filename. 'training', 'validation', 'testing'.
        max_sweeps (int): Number of sweeps.
    """

    def __init__(self,
                 load_dir,
                 prefix,
                 max_sweeps=10):
        self.class_names = ['person', 'bike', 'car', 'truck', 'bus']

        self.load_dir = load_dir
        self.prefix = prefix
        self.test = False

        # 从config文件中读取6个camera的内外参
        json_label_name = f'{self.load_dir}/calibrated_sensor.json'
        with open(json_label_name, "r") as f:
            camera_cali_json = json.load(f)
        self.camera_calibrations = self.ParseCalibrationsJson(camera_cali_json)

    def convert(self):#调用这里
        """Convert action."""
        print('Start converting ...')
        train_list = []
        count = 0
        file_names = os.listdir(self.load_dir)
        file_names.sort()
        for file_name in file_names:
            full_path = os.path.join(self.load_dir, file_name, '3d_url')
            if os.path.isdir(full_path):
                print(count, full_path)
                train_info = self.convertOne(full_path)
                count = count + 1
                print("sample: {}".format(len(train_info)))
                train_list = train_list + train_info
               
        info_path = osp.join(self.load_dir, "{}_infos_train_3060undist_clsnamev4_ptsnum.pkl".format(self.prefix))#目录以及文件名称拼接
        data = dict(infos=train_list)
        mmcv.dump(data, info_path)#保存文件

        print('\nFinished ...')
    
    def convertOne(self, file_path):
        label_num = 0
        points_data_num = 0
        c0_data_num = c1_data_num = c2_data_num = c3_data_num = c4_data_num = c5_data_num = 0
        multi_image_names = []
        # 检查lidar数据、camera数据个数是否一致
        for dir_name in os.listdir(file_path):
            if dir_name == 'json':
    
                label_names = os.listdir(file_path + '/json/')
                label_names.sort()
                label_num = len(label_names)
                # print("=====label==========")
                # print("Label_nums: ", label_num)
                # print("Firstlabel: ", label_names[0])
                # print("Lastlabel : ", label_names[label_num - 1])
    
            elif dir_name == 'pcd':

                points_data_names = os.listdir(file_path + '/pcd/')
                points_data_names.sort()
                points_data_num = len(points_data_names)
                # print("=====points data==========")
                # print("data_nums: ", points_data_num)
                # print("Firstdata: ", points_data_names[0])
                # print("Lastdata : ", points_data_names[points_data_num - 1])

            elif dir_name == 'raw_data':

                c0_data_path = file_path + '/raw_data/30_front'
                if os.path.exists(c0_data_path):
                    c0_data_names = os.listdir(c0_data_path)
                    c0_data_names.sort()
                    c0_data_num = len(c0_data_names)
                    multi_image_names.append(c0_data_names)
                else:
                    multi_image_names.append([])

                c1_data_path = file_path + '/raw_data/60_front'
                if os.path.exists(c1_data_path):
                    c1_data_names = os.listdir(c1_data_path)
                    c1_data_names.sort()
                    c1_data_num = len(c1_data_names)
                    multi_image_names.append(c1_data_names)
                else:
                    multi_image_names.append([])

                c2_data_path = file_path + '/raw_data/120_front'
                if os.path.exists(c2_data_path):
                    c2_data_names = os.listdir(c2_data_path)
                    c2_data_names.sort()
                    c2_data_num = len(c2_data_names)
                    multi_image_names.append(c2_data_names)
                else:
                    multi_image_names.append([])

                c3_data_path = file_path + '/raw_data/120_back'
                if os.path.exists(c3_data_path):
                    c3_data_names = os.listdir(c3_data_path)
                    c3_data_names.sort()
                    c3_data_num = len(c3_data_names)
                    multi_image_names.append(c3_data_names)
                else:
                    multi_image_names.append([])

                c4_data_path = file_path + '/raw_data/120_left'
                if os.path.exists(c4_data_path):
                    c4_data_names = os.listdir(c4_data_path)
                    c4_data_names.sort()
                    c4_data_num = len(c4_data_names)
                    multi_image_names.append(c4_data_names)
                else:
                    multi_image_names.append([])

                c5_data_path = file_path + '/raw_data/120_right'
                if os.path.exists(c5_data_path):
                    c5_data_names = os.listdir(c5_data_path)
                    c5_data_names.sort()
                    c5_data_num = len(c5_data_names)
                    multi_image_names.append(c5_data_names)
                else:
                    multi_image_names.append([])

                print("=====cameras data==========")
                print("30_front_nums: ", c0_data_num)
                print("60_front_nums: ", c1_data_num)
                print("120_front_nums: ", c2_data_num)
                print("120_back_nums: ", c3_data_num)
                print("120_left_nums: ", c4_data_num)
                print("120_right_nums: ", c5_data_num)

            else:
                continue

        train_infos = []
        if points_data_num == 0:
            if not os.path.exists(file_path + '/bin/'):
                print("Error!   can not find 'bin'  in ", file_path)
                return train_infos
            points_data_names = os.listdir(file_path + '/bin/')
            points_data_names.sort()
            points_data_num = len(points_data_names)
            print("\nError!   can not find 'pcd'  in ", file_path)
            if points_data_num == 0:
                print("Error!   can not find 'bin'  in ", file_path)
                return train_infos
        if label_num == 0:
            print("\nError!   can not find 'json' in ", file_path)
            return train_infos
        if points_data_num > label_num:
            print("\nWarning！ data_num>points_data_num ")
            return train_infos

        '''
        # 检查数据与标签名是否一致
        j = 0
        for i in range(len(label_names)):
            templabel = label_names[i][:-5]#后4位是.json不用比较
            tempdata = points_data_names[i][:-4]
            c0_data = multi_image_names[0][i][:-4]
            c1_data = multi_image_names[1][i][:-4]
            c2_data = multi_image_names[2][i][:-4]
            c3_data = multi_image_names[3][i][:-4]
            c4_data = multi_image_names[4][i][:-4]
            c5_data = multi_image_names[5][i][:-4]
            if (templabel != tempdata or templabel != c0_data or 
                templabel != c1_data or templabel != c2_data or 
                templabel != c3_data or templabel != c4_data or 
                templabel != c5_data):
                print("label and data not match", templabel, tempdata, i)
                j = j + 1
        print(" not match  num :", j)
        '''

        # 转换pcd转为bin格式
        mmcv.mkdir_or_exist(f'{file_path}/bin')
        for data_name in points_data_names:
            self.save_lidar(file_path, data_name[:-4])

        # 生成pkl文件
        data_names = list(map(lambda n: n[:-4], points_data_names))
        train_infos = self.fill_trainval_infos(file_path, data_names)
        return train_infos

    def save_lidar(self, file_path, data_name):
        src = f'{file_path}/pcd/{data_name}.pcd'
        dst = f'{file_path}/bin/{data_name}.bin'
        if os.path.exists(dst):
            return
        else:
            self.pcd2bin(src, dst)

    def pcd2bin(self, pcd_fullname, bin_fullname):
        pc = pypcd.PointCloud.from_path(pcd_fullname)

        ## Get data from pcd (x, y, z, intensity, label)
        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_i = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32)/256
        np_l = (np.array(pc.pc_data['label'], dtype=np.int16)).astype(np.int16)
  
        points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i, np_l)))

        ## Save bin file                                    
        points_32.tofile(bin_fullname)
    
    def get_class_name_v_4(self, label=None):
        parent_label = label['label']
        sublabel_label = label['sublabel']

        # person without person_inside
        if (parent_label == 'person' or parent_label == '1000') and sublabel_label != '1004': # 1004 person_inside
            cur_class = 'person'

        # pushing -> bike
        elif parent_label == 'pushing' or parent_label == '2000' or sublabel_label == '8003':
            cur_class = 'bike'

        # rider -> bike
        elif (parent_label == 'rider' or \
                (parent_label == 'bike' and sublabel_label != 'crowds')) or \
                    (parent_label == '4000' or (parent_label == '3000')):
            cur_class = 'bike'
            # bike group but group=0
            width = label['3Dsize']['width']
            if "deep" in label["3Dsize"]:
                length = label["3Dsize"]["deep"]
            elif "length" in label["3Dsize"]:
                length = label["3Dsize"]["length"]
            heigth = label['3Dsize']['height']
            if width > 3.5 or heigth > 3.5:
                cur_class = None

        # car
        elif parent_label == 'car' or parent_label == '5000':
            width = label['3Dsize']['width']
            if "deep" in label["3Dsize"]:
                length = label["3Dsize"]["deep"]
            elif "length" in label["3Dsize"]:
                length = label["3Dsize"]["length"]
            heigth = label['3Dsize']['height']

            # minicar-> bike, shed_car->bike
            if ((sublabel_label == 'shed' or sublabel_label in ['5001', '5002']) and \
                    (width < 1.1 and heigth < 2.2)):
                cur_class = 'bike'
            else:
                cur_class = 'car'

        # truck
        elif parent_label == 'truck' or parent_label == '6000':
            cur_class = 'truck'

        # bus
        elif parent_label == 'bus' or parent_label == '7000':
            cur_class = 'bus'

        else:
            cur_class = None

        if label['sublabel'] == '1004':
            print('person_inside cur_class: ', cur_class)

        return cur_class

    def ParseAnnoJson(self, data_name):
        json_label_name = f'{data_name}.json'
        with open(json_label_name, "r") as f:
            label_json_dict = json.load(f)

        gt_cate = []
        gt_2d_box = []
        gt_3d_box = []
        num_pits = []
        
        for item in label_json_dict["result"]["data"]:
            # mapping: label->class_name
            class_name = self.get_class_name_v_4(item)
            if class_name not in self.class_names:
                # print('Other class_name: ', data_name, item)
                continue
            
            #剔除残影
            if 'is_excessive_layering' in label_json_dict['result']:
                if label_json_dict['result']['is_excessive_layering'] == 'True':
                    print('Excessive_layering 1: ', data_name, item)
                    continue
            else:
                if item['is_excessive_layering'] == 'True':
                    print('Excessive_layering 2: ', data_name, item)
                    continue

            #剔除群目标
            if item['group'] == 1:
                print('Group: ', data_name, item)
                continue

            if item.get('3DCenter', None) != None:
                item['3Dcenter'] = item['3DCenter']

            cx = item['3Dcenter']['x']
            cy = item['3Dcenter']['y']
            cz = item['3Dcenter']['z']

            if "deep" in item["3Dsize"]:
                length = item["3Dsize"]["deep"]
            elif "length" in item["3Dsize"]:
                length = item["3Dsize"]["length"]
            
            width = item["3Dsize"]["width"]
            height = item["3Dsize"]["height"]
            yaw = item["3Dsize"]["alpha"]

            if 'pointLength' in item:
                points_num=item['pointLength']
            else:
                points_num=item['pointnum'] 

            if points_num == 0: #由于平台导出错误,有的json文件中所有的目标点数属性个数都为0,需要重新计算
                self.load_dim, self.use_dim = 5, [0,1,2,3]
                lidar_path = data_name.replace('/json/', '/bin/') + '.bin'
                points = np.fromfile(lidar_path, dtype=np.float32)
                points = points.reshape(-1, self.load_dim)
                points = points[:, self.use_dim]

                if 'is_excessive_layering' in label_json_dict['result']:
                    rbox_i = [item['3Dcenter']['x'],   item['3Dcenter']['y'],  item['3Dcenter']['z'],
                              item['3Dsize']['height'],item['3Dsize']['width'],item['3Dsize']['deep'],
                              item['3Dsize']['alpha']]
                else:
                    rbox_i = [item['3Dcenter']['x'],   item['3Dcenter']['y'],  item['3Dcenter']['z'],
                              item['3Dsize']['length'],item['3Dsize']['width'],item['3Dsize']['height'],
                            # item['3Dsize']['deep'],item['3Dsize']['width'],item['3Dsize']['height'],
                              item['3Dsize']['alpha']]
                rbox_np = np.array([rbox_i])
                points_num = get_pointnums(points, rbox_np)[0]
            
            if points_num < 1:
                print('points_num < 1: ', data_name, item)
                continue

            float_list = [cx, cy, cz, length, width, height, yaw]
            float_flag = True
            for tmp in float_list:
                if not isinstance(tmp, float) and not isinstance(tmp, int):
                    float_flag = False
                    break
            if not float_flag:
                print('Not float_flag: ', data_name, item)
                continue

            gt_cate.append(class_name)
            gt_2d_box.append([0,0,0,0])
            gt_3d_box.append([cx, cy, cz, length, width, height, yaw])
            num_pits.append(points_num)

        gt_cate = np.array(gt_cate, dtype=str)
        gt_2d_box = np.array(gt_2d_box, dtype=np.float32)
        gt_3d_box = np.array(gt_3d_box, dtype=np.float32)
        gt_3d_box_corners = None
        num_pits = np.array(num_pits, dtype=np.int)
        return gt_cate, gt_2d_box, gt_3d_box, gt_3d_box_corners, num_pits
    
    class SensorCali:
        def __init__(self,
                 name,
                 topic,
                 extrinsic_tran,
                 extrinsic_rot,
                 intrinsic):
            self.name = name
            self.topic = topic
            self.extrinsic_tran = extrinsic_tran
            self.extrinsic_rot = extrinsic_rot
            self.intrinsic = intrinsic

    def ParseCalibrationsJson(self, camera_cali_json):
        camera_calibrations = []
        sensor_info_list = camera_cali_json["sensor_info"] if "sensor_info" in camera_cali_json.keys() else camera_cali_json["sensorInfo"]
        for item in sensor_info_list:
            if "topic" not in item.keys():
                continue

            topic = item["topic"] 
            if topic not in ["/sensor/camera/sensing/image_raw_120", "/sensor/camera/sensing/image_raw_60", "/sensor/camera/sensing/image_raw_30"]:
                continue

            name = item["name"]
            trans_json = item["extrinsic"]["translation"]
            extrinsic_tran = [trans_json["x"], trans_json["y"], trans_json["z"]]
            rot_json = item["extrinsic"]["rotation"]
            extrinsic_rot = [rot_json["w"], rot_json["x"], rot_json["y"], rot_json["z"]]#(w, x, y, z)

            int_json = item["intrinsic"]
            if "matrix" in int_json:
                intrinsic = int_json["matrix"]
            else:
                intrinsic = [int_json["matrix0"], int_json["matrix1"], int_json["matrix2"], 
                             int_json["matrix3"], int_json["matrix4"], int_json["matrix5"], 
                             int_json["matrix6"], int_json["matrix7"], int_json["matrix8"]]

            cali = Mogo2KITTI.SensorCali(name, topic, extrinsic_tran, extrinsic_rot, intrinsic)
            camera_calibrations.append(cali)

        return camera_calibrations

    def parse_pose_file(self, fname_pose):
        pose_dict = collections.OrderedDict()
        with open(fname_pose, 'r') as f:
            for line in f:
                s = line.strip().split(' ')
                pose_dict[s[0]] = np.array(s[1:], dtype=np.float32).reshape((4, 4))
        return pose_dict

    def fill_trainval_infos(self, file_path, data_names):
        """Generate the train/val infos from the raw data.
        """

        camera_map = {
            '/sensor/camera/sensing/image_raw_60': '60_front',
            '/sensor/camera/sensing/image_raw_30': '30_front',
            '/sensor/camera/sensing/image_raw_120': '120_front',
            '/sensor/camera/sensing/image_raw_back': '120_back',
            '/sensor/camera/sensing/image_raw_left': '120_left',
            '/sensor/camera/sensing/image_raw_right': '120_right'
        }
        cameras_calib = []
        for camera in self.camera_calibrations:

            # extrinsic parameters
            sensor2ego_rotation = camera.extrinsic_rot#四元数转旋转矩阵,(w, x, y, z)
            sensor2ego_translation = camera.extrinsic_tran

            # 这两个矩阵本来应该是按照lidar和camera的时间戳，转换到utm坐标系再转换回来的，但是我们是强制对齐的，所以是固定转换
            sensor2lidar_rotation = Quaternion(sensor2ego_rotation).rotation_matrix # 这里要求输入的四元数是 [w,x,y,z]
            sensor2lidar_translation = sensor2ego_translation

            info = {
                "topic": camera.topic,
                "type": camera_map[camera.topic],
                "sensor2ego_translation": sensor2ego_translation,
                "sensor2ego_rotation": sensor2ego_rotation,
                "camera_intrinsics": np.array(camera.intrinsic).reshape(3, 3),
                "sensor2lidar_rotation": np.array(sensor2lidar_rotation),
                "sensor2lidar_translation": np.array(sensor2lidar_translation)
            }
            cameras_calib.append(info)

        #因为我们的点云和真值都是base_link坐标系下，所以lidar2ego旋转矩阵设置为单位阵，平移为0
        lidar2ego_rotation = [1.0, 0.0, 0.0, 0.0]#保存为(w, x, y, z)格式
        lidar2ego_translation = [0.0, 0.0, 0.0]
        
        # # 读取base_link到utm的坐标转换
        # pose_txt_name = f'{file_path}/raw_data/pose.txt'
        # pose = self.parse_pose_file(pose_txt_name)

        # 组织数据
        train_infos = []
        for idx, data_name in enumerate(data_names):
            lidar_path = f'{file_path}/bin/{data_name}.bin'
            # # 从pose.txt中读取
            # ego2global_rotation = Rotation.from_matrix(pose[data_name][:3, :3]).as_quat().tolist()#scipy输出为(x, y, z, w)
            # ego2global_rotation[0], ego2global_rotation[1], ego2global_rotation[2], ego2global_rotation[3] = ego2global_rotation[3], ego2global_rotation[0], ego2global_rotation[1], ego2global_rotation[2]#转换为(w, x, y, z)
            # ego2global_translation= pose[data_name][:3, 3].tolist()
            # 时间戳字符串转int
            timestamp = data_name.replace('.', '')
            timestamp = int(timestamp)

            info = {
                "lidar_path": lidar_path,
                "sweeps": [],
                "cams": dict(),
                "lidar2ego_translation": lidar2ego_translation,
                "lidar2ego_rotation": lidar2ego_rotation,
                # "ego2global_translation": ego2global_translation,
                # "ego2global_rotation": ego2global_rotation,
                "timestamp": timestamp
            }

            # obtain 6 image's information per frame
            no_img = False
            for cam_info in cameras_calib:
                cam_info_copy = copy.deepcopy(cam_info)

                if cam_info_copy["type"] in ['30_front', '60_front']:
                    data_path = f'{file_path}/raw_data/{cam_info_copy["type"]}/{data_name}.jpg'
                    # data_path = f'{file_path}/raw_data/{cam_info_copy["type"]}/{data_name}_undist.jpg'
                else:
                    data_path = f'{file_path}/raw_data/{cam_info_copy["type"]}/{data_name}.jpg'

                if not os.path.exists(data_path):
                    print('Not exist img: ', data_path)
                    continue
                    # no_img = True
                    # break

                cam_info_copy.update(data_path=data_path)
                # cam_info_copy.update(ego2global_translation=ego2global_translation)
                # cam_info_copy.update(ego2global_rotation=ego2global_rotation)
                cam_info_copy.update(timestamp=timestamp)
                info["cams"].update({cam_info_copy["type"]: cam_info_copy})
            # if no_img:
            #     print(data_name, "no img matched")
            #     continue
            # obtain annotation
            if not self.test:
                gt_cate, gt_2d_box, gt_3d_box, t_3d_box_corners, num_pits = self.ParseAnnoJson(f'{file_path}/json/{data_name}')

                gt_boxes = []
                for bbox3 in gt_3d_box:
                    bbox3[6] = -bbox3[6] # https://github.com/mit-han-lab/bevfusion/issues/382 heading需要取反，不知道为啥
                    gt_boxes.append(bbox3)

                valid_flag = np.array(
                    [
                        num > 0
                        for num in num_pits
                    ],
                    dtype=bool,
                ).reshape(-1)

                if len(gt_boxes) < 1:
                    print(lidar_path,' : fail ! no object ',  f'{idx} / {len(data_names)}')
                    continue

                info["gt_boxes"] = np.array(gt_boxes)#存放的是base_link坐标系下的标注
                info["gt_names"] = gt_cate
                info["num_lidar_pts"] = num_pits
                info["valid_flag"] = valid_flag
                # we need to convert rot to SECOND format.
                # gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

            train_infos.append(info)

        return train_infos


