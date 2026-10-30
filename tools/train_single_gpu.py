import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


def main():
    # dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")#config文件
    parser.add_argument("run_dir", metavar="DIR", help="run directory")#输出路径，output/bev_result/
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)#递归加载参数
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)#使用mmcv中的Config构造对象

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark#false
    torch.cuda.set_device(0)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)#用于log日志的输出
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))#保存config文件

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)#log

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)# Python的随机性
        np.random.seed(cfg.seed)#numpy随机性
        torch.manual_seed(cfg.seed)#torch的cpu随机性，为cpu随机设置种子
        if cfg.deterministic:#参数中是false
            torch.backends.cudnn.deterministic = True#选择确定算法
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]#dataset实例化

    model = build_model(cfg.model)#构造模型-BEVFusion
    model.init_weights()#初始化的时候已经调用了，不用在这里再调用一次-camera主干网络使用预训练模型
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])#?

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=timestamp,
    )#开始训练步骤


if __name__ == "__main__":
    main()
