import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

nus_categories = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

nus_attributes = (
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
    "None",
)


def create_nuscenes_infos(
    root_path, info_prefix, version="v1.0-trainval", max_sweeps=10
):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits

    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers #version在这三类之一，如果不是的话，直接触发异常结束
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes] #得到一个有效场景list
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes)) #train_scenes中在available_scene_names中的筛选出一个list出来
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set( #list去重
        [
            available_scenes[available_scene_names.index(s)]["token"] #token是场景和标注的唯一标识
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )

    test = "test" in version#是test的话
    if test:
        print("test scene: {}".format(len(train_scenes)))#打印test场景的长度
    else:
        print(
            "train scene: {}, val scene: {}".format(len(train_scenes), len(val_scenes))#是trainval的话打印训练集和验证集的长度
        )
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos( #读取数据
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps
    )

    metadata = dict(version=version)#创建一个字典
    if test:
        print("test sample: {}".format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, "{}_infos_test.pkl".format(info_prefix))#目录以及文件名称拼接
        mmcv.dump(data, info_path)#保存文件
    else:
        print(
            "train sample: {}, val sample: {}".format(
                len(train_nusc_infos), len(val_nusc_infos)
            )
        )
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, "{}_infos_train.pkl".format(info_prefix))
        mmcv.dump(data, info_path)
        data["infos"] = val_nusc_infos
        info_val_path = osp.join(root_path, "{}_infos_val.pkl".format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):#获取场景，总共有1110个
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    sample_rec：
    { 'token': 'ca9a282c9e77460f8360f564131a8af5', 
      'timestamp': 1532402927647951, 
      'prev': '', 
      'next': '39586f9d59004284a7114a68825e8eec', 
      'scene_token': 'cc8c0bf57f984915a77078b10eb33198', 
      'data': {
        'RADAR_FRONT': '37091c75b9704e0daa829ba56dfa0906', 
        'RADAR_FRONT_LEFT': '11946c1461d14016a322916157da3c7d', 
        'RADAR_FRONT_RIGHT': '491209956ee3435a9ec173dad3aaf58b', 
        'RADAR_BACK_LEFT': '312aa38d0e3e4f01b3124c523e6f9776', 
        'RADAR_BACK_RIGHT': '07b30d5eb6104e79be58eadf94382bc1', 
        'LIDAR_TOP': '9d9bf11fb0e144c8b446d54a8a00184f', 
        'CAM_FRONT': 'e3d495d4ac534d54b321f50006683844', 
        'CAM_FRONT_RIGHT': 'aac7867ebf4f446395d29fbd60b63b3b', 
        'CAM_BACK_RIGHT': '79dbb4460a6b40f49f9c150cb118247e', 
        'CAM_BACK': '03bea5763f0f4722933508d5999c5fd8', 
        'CAM_BACK_LEFT': '43893a033f9c46d4a51b5e08a67a1eb7', 
        'CAM_FRONT_LEFT': 'fe5422747a7d4268a4b07fc396707b23'
        }, 
      'anns': ['ef63a697930c4b20a6b9791f423351da', '6b89da9bf1f84fd6a5fbe1c3b236f809', 
               '924ee6ac1fed440a9d9e3720aac635a0', '91e3608f55174a319246f361690906ba', 
               'cd051723ed9c40f692b9266359f547af', '36d52dfedd764b27863375543c965376', 
               '70af124fceeb433ea73a79537e4bea9e', '63b89fe17f3e41ecbe28337e0e35db8e', 
               'e4a3582721c34f528e3367f0bda9485d', 'fcb2332977ed4203aa4b7e04a538e309', 
               'a0cac1c12246451684116067ae2611f6', '02248ff567e3497c957c369dc9a1bd5c', 
               '9db977e264964c2887db1e37113cddaa', 'ca9c5dd6cf374aa980fdd81022f016fd', 
               '179b8b54ee74425893387ebc09ee133d', '5b990ac640bf498ca7fd55eaf85d3e12', 
               '16140fbf143d4e26a4a7613cbd3aa0e8', '54939f11a73d4398b14aeef500bf0c23', 
               '83d881a6b3d94ef3a3bc3b585cc514f8', '74986f1604f047b6925d409915265bf7', 
               'e86330c5538c4858b8d3ffe874556cc5', 'a7bd5bb89e27455bbb3dba89a576b6a1', 
               'fbd9d8c939b24f0eb6496243a41e8c41', '198023a1fb5343a5b6fad033ab8b7057', 
               'ffeafb90ecd5429cba23d0be9a5b54ee', 'cc636a58e27e446cbdd030c14f3718fd', 
               '076a7e3ec6244d3b84e7df5ebcbac637', '0603fbaef1234c6c86424b163d2e3141', 
               'd76bd5dcc62f4c57b9cece1c7bcfabc5', '5acb6c71bcd64aa188804411b28c4c8f', 
               '49b74a5f193c4759b203123b58ca176d', '77519174b48f4853a895f58bb8f98661', 
               'c5e9455e98bb42c0af7d1990db1df0c9', 'fcc5b4b5c4724179ab24962a39ca6d65', 
               '791d1ca7e228433fa50b01778c32449a', '316d20eb238c43ef9ee195642dd6e3fe', 
               'cda0a9085607438c9b1ea87f4360dd64', 'e865152aaa194f22b97ad0078c012b21', 
               '7962506dbc24423aa540a5e4c7083dad', '29cca6a580924b72a90b9dd6e7710d3e', 
               'a6f7d4bb60374f868144c5ba4431bf4c', 'f1ae3f713ba946069fa084a6b8626fbf', 
               'd7af8ede316546f68d4ab4f3dbf03f88', '91cb8f15ed4444e99470d43515e50c1d', 
               'bc638d33e89848f58c0b3ccf3900c8bb', '26fb370c13f844de9d1830f6176ebab6', 
               '7e66fdf908d84237943c833e6c1b317a', '67c5dbb3ddcc4aff8ec5140930723c37', 
               'eaf2532c820740ae905bb7ed78fb1037', '3e2d17fa9aa5484d9cabc1dfca532193', 
               'de6bd5ffbed24aa59c8891f8d9c32c44', '9d51d699f635478fbbcd82a70396dd62', 
               'b7cbc6d0e80e4dfda7164871ece6cb71', '563a3f547bd64a2f9969278c5ef447fd', 
               'df8917888b81424f8c0670939e61d885', 'bb3ef5ced8854640910132b11b597348', 
               'a522ce1d7f6545d7955779f25d01783b', '1fafb2468af5481ca9967407af219c32', 
               '05de82bdb8484623906bb9d97ae87542', 'bfedb0d85e164b7697d1e72dd971fb72', 
               'ca0f85b4f0d44beb9b7ff87b1ab37ff5', 'bca4bbfdef3d4de980842f28be80b3ca', 
               'a834fb0389a8453c810c3330e3503e16', '6c804cb7d78943b195045082c5c2d7fa', 
               'adf1594def9e4722b952fea33b307937', '49f76277d07541c5a584aa14c9d28754', 
               '15a3b4d60b514db5a3468e2aef72a90c', '18cc2837f2b9457c80af0761a0b83ccc', 
               '2bfcc693ae9946daba1d9f2724478fd4']
    }
    sd_rec:
    { 'token': '9d9bf11fb0e144c8b446d54a8a00184f', 
      'sample_token': 'ca9a282c9e77460f8360f564131a8af5', 
      'ego_pose_token': '9d9bf11fb0e144c8b446d54a8a00184f', 
      'calibrated_sensor_token': 'a183049901c24361a6b0b11b8013137c', 
      'timestamp': 1532402927647951, 
      'fileformat': 'pcd', 
      'is_key_frame': True, 
      'height': 0, 
      'width': 0, 
      'filename': 'samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin', 
      'prev': '', 'next': '0cedf1d2d652468d92d23491136b5d15', 
      'sensor_modality': 'lidar', 
      'channel': 'LIDAR_TOP'
    }
    """
    available_scenes = []
    print("total scene num: {}".format(len(nusc.scene)))
    for scene in nusc.scene:#遍历1110个场景
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])#获取当前sample的激光数据
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])#获取对应的激光数据、标注框
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc, train_scenes, val_scenes, test=False, max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    
    cs_record:#sensor-ego(calibrated_sensor.json)
    {  "token": "a183049901c24361a6b0b11b8013137c",
       "sensor_token": "dc8b396651c05aedbb9cdaae573bb567",
       "translation": [0.943713,0.0,1.84023],
       "rotation": [0.7077955119163518,0.006492242056004365,0.010646214713995808,-0.7063073142877817],
       "camera_intrinsic": []
    }
    pose_record:#ego-global
    {  'token': '9d9bf11fb0e144c8b446d54a8a00184f',
       'timestamp': ' ',
       'translation': ' ',
       'rotation': ' ',
    }
    """
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        location = nusc.get(
            "log", nusc.get("scene", sample["scene_token"])["log_token"]
        )["location"]
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)#boxes-lidar box(标注文件中是世界坐标系下的标注，需要转换到lidar frame,并且按照一定的格式进行转换)

        mmcv.check_file_exist(lidar_path)

        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
            "location": location,
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix#四元数转旋转矩阵
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        for cam in camera_types:
            cam_token = sample["data"][cam]#依次取camera数据
            cam_path, _, camera_intrinsics = nusc.get_sample_data(cam_token)#获取图片路径和内参,box不要
            """
            cam_info举个例子:
            'CAM_FRONT_RIGHT': 
            {   'data_path': './data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-11-21-19-58-31+0800__CAM_FRONT_RIGHT__1542801729870339.jpg', 
                'type': 'CAM_FRONT_RIGHT', 
                'sample_data_token': '93fb7424292d4d48bc6ef2ae9c7006bd', 
                'sensor2ego_translation': [1.5508477543, -0.493404796419, 1.49574800619], 
                'sensor2ego_rotation': [0.2060347966337182, -0.2026940577919598, 0.6824507824531167, -0.6713610884174485], 
                'ego2global_translation': [1533.066845615666, 1336.0792883113154, 0.0], 
                'ego2global_rotation': [0.656367902638072, -0.005911921657241622, 0.015944576065496593, -0.7542492930476692], 
                'timestamp': 1542801729870339, 
                'sensor2lidar_rotation': array([[ 0.55051603, -0.01091146,  0.83475328],[-0.83441872,  0.02398423,  0.55060889],[-0.02602887, -0.99965279,  0.00409898]]),#lidar先按照lidar时间戳转到global，再根据camera时间戳转到相机坐标系
                'sensor2lidar_translation': array([ 0.49492732,  0.27486624, -0.33954398]), 
                'camera_intrinsics': array([[1.26084744e+03, 0.00000000e+00, 8.07968245e+02],[0.00000000e+00, 1.26084744e+03, 4.95334427e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            }, 
            """
            cam_info = obtain_sensor2top(#camera到lidar坐标系的坐标转换,因为camera和lidar数据获取时候的时间戳不一致，所以需要先转到utm坐标系，再转回到lidar坐标系
                nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
            )
            cam_info.update(camera_intrinsics=camera_intrinsics)#把相机内参放进去
            info["cams"].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec["prev"] == "":
                sweep = obtain_sensor2top(
                    nusc, sd_rec["prev"], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar"
                )
                sweeps.append(sweep)
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)#中心点，需要确认这里的中心是地面中心还是重心-已经确认了是重心
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)#需要确认时lwh还是wlh-确认了是wlh
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(#需要确认从x正轴开始，还是y负轴开始，应该是x轴逆时针
                -1, 1
            )
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample["anns"]]
            )
            valid_flag = np.array(
                [
                    (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                    for anno in annotations
                ],
                dtype=bool,
            ).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)#heading为什么转换成这个样子?
            assert len(gt_boxes) == len(
                annotations
            ), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
            info["valid_flag"] = valid_flag

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)#camera
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]#相机时间戳下的ego-global
    #e2g_r_mat lidar时间戳下，ego-global

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    nusc_infos = mmcv.load(info_path)["infos"]
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info["cams"][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info["sample_data_token"],
                visibilities=["", "1", "2", "3", "4"],
                mono3d=mono3d,
            )
            (height, width, _) = mmcv.imread(cam_info["data_path"]).shape
            coco_2d_dict["images"].append(
                dict(
                    file_name=cam_info["data_path"].split("data/nuscenes/")[-1],
                    id=cam_info["sample_data_token"],
                    token=info["token"],
                    cam2ego_rotation=cam_info["sensor2ego_rotation"],
                    cam2ego_translation=cam_info["sensor2ego_translation"],
                    ego2global_rotation=info["ego2global_rotation"],
                    ego2global_translation=info["ego2global_translation"],
                    camera_intrinsics=cam_info["camera_intrinsics"],
                    width=width,
                    height=height,
                )
            )
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info["segmentation"] = []
                coco_info["id"] = coco_ann_id
                coco_2d_dict["annotations"].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f"{info_path[:-4]}_mono3d"
    else:
        json_prefix = f"{info_path[:-4]}"
    mmcv.dump(coco_2d_dict, f"{json_prefix}.coco.json")


def get_2d_boxes(nusc, sample_data_token: str, visibilities: List[str], mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get("sample_data", sample_data_token)

    assert sd_rec["sensor_modality"] == "camera", (
        "Error: get_2d_boxes only works" " for camera sample_data!"
    )
    if not sd_rec["is_key_frame"]:
        raise ValueError("The 2D re-projections are available only for keyframes.")

    s_rec = nusc.get("sample", sd_rec["sample_token"])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    camera_intrinsic = np.array(cs_rec["camera_intrinsic"])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get("sample_annotation", token) for token in s_rec["anns"]]
    ann_recs = [
        ann_rec for ann_rec in ann_recs if (ann_rec["visibility_token"] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec["sample_annotation_token"] = ann_rec["token"]
        ann_rec["sample_data_token"] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec["token"])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec["translation"]))
        box.rotate(Quaternion(pose_rec["rotation"]).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec["translation"]))
        box.rotate(Quaternion(cs_rec["rotation"]).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = (
            view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
        )

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(
            ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec["filename"]
        )

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec["rotation"]).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec["rotation"]).rotation_matrix
            cam_velo3d = (
                global_velo3d @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            )
            velo = cam_velo3d[0::2].tolist()

            repro_rec["bbox_cam3d"] = loc + dim + rot
            repro_rec["velo_cam3d"] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(center3d, camera_intrinsic, with_depth=True)
            repro_rec["center2d"] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec["center2d"][2] <= 0:
                continue

            ann_token = nusc.get("sample_annotation", box.token)["attribute_tokens"]
            if len(ann_token) == 0:
                attr_name = "None"
            else:
                attr_name = nusc.get("attribute", ann_token[0])["name"]
            attr_id = nus_attributes.index(attr_name)
            repro_rec["attribute_name"] = attr_name
            repro_rec["attribute_id"] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords]
        )

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(
    ann_rec: dict,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    sample_data_token: str,
    filename: str,
) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec["sample_data_token"] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        "attribute_tokens",
        "category_name",
        "instance_token",
        "next",
        "num_lidar_pts",
        "num_radar_pts",
        "prev",
        "sample_annotation_token",
        "sample_data_token",
        "visibility_token",
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec["bbox_corners"] = [x1, y1, x2, y2]
    repro_rec["filename"] = filename

    coco_rec["file_name"] = filename
    coco_rec["image_id"] = sample_data_token
    coco_rec["area"] = (y2 - y1) * (x2 - x1)

    if repro_rec["category_name"] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec["category_name"]]
    coco_rec["category_name"] = cat_name
    coco_rec["category_id"] = nus_categories.index(cat_name)
    coco_rec["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec["iscrowd"] = 0

    return coco_rec
