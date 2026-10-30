import argparse
from os import path as osp


from data_converter import nuscenes_converter as nuscenes_converter
from data_converter import mogo_converter as mogo_converter
from data_converter.create_gt_database import create_groundtruth_database

# 数据预处理代码

def nuscenes_data_prep(
    root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    max_sweeps=10,
    load_augmented=None,
):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    if load_augmented is None:
        # otherwise, infos must have been created, we just skip.
        nuscenes_converter.create_nuscenes_infos(#预处理数据保存为pkl文件
            root_path, info_prefix, version=version, max_sweeps=max_sweeps
        )

        # if version == "v1.0-test":
        #     info_test_path = osp.join(root_path, f"{info_prefix}_infos_test.pkl")
        #     nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
        #     return

        # info_train_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
        # info_val_path = osp.join(root_path, f"{info_prefix}_infos_val.pkl")
        # nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        # nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)

    create_groundtruth_database(
        dataset_name,
        root_path,
        info_prefix,
        f"{out_dir}/{info_prefix}_infos_train.pkl",
        load_augmented=load_augmented,
    )

def mogo_data_prep(root_path,
                    info_prefix,
                    workers,
                    max_sweeps=0):
    
    # Generate mogo infos to .pkl
    converter = mogo_converter.Mogo2KITTI(
        root_path,
        info_prefix,
        max_sweeps=0)
    converter.convert()


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="kitti", help="name of the dataset") #数据集名称,用nuscenes
parser.add_argument( #数据一般保存在./data/nuscenes
    "--root-path",
    type=str,
    default="./data/kitti",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--version",
    type=str,
    default="v1.0",
    required=False,
    help="specify the dataset version, no need for kitti",
)
parser.add_argument( #激光频率，默认10hz
    "--max-sweeps",
    type=int,
    default=10,
    required=False,
    help="specify sweeps of lidar per example",
)
parser.add_argument( #pkl文件输出地址，一般设置同--root-path
    "--out-dir",
    type=str,
    default="./data/kitti",
    required=False,
    help="name of info pkl",
)
parser.add_argument("--extra-tag", type=str, default="kitti") #额外的标签，用于pkl文件前缀
parser.add_argument("--painted", default=False, action="store_true")
parser.add_argument("--virtual", default=False, action="store_true")
parser.add_argument( #使用的线程个数，默认为4
    "--workers", type=int, default=4, help="number of threads to be used"
)
args = parser.parse_args()

if __name__ == "__main__":
    load_augmented = None
    if args.virtual: #False
        if args.painted:
            load_augmented = "mvp"
        else:
            load_augmented = "pointpainting"

    if args.dataset == "nuscenes" and args.version != "v1.0-mini": #True
        train_version = f"{args.version}-trainval" #训练、验证数据集文件夹名称string
        nuscenes_data_prep( #数据预处理
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
        test_version = f"{args.version}-test"#测试数据集文件夹名称
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
    elif args.dataset == "nuscenes" and args.version == "v1.0-mini": #如果使用的是mini数据集的话
        train_version = f"{args.version}"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
    elif args.dataset == 'mogo':
        mogo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
