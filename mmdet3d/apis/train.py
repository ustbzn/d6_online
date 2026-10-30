import torch
from torch import nn
from mmcv.parallel import MMDistributedDataParallel,MMDataParallel
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    GradientCumulativeFp16OptimizerHook,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from mmdet3d.runner import CustomEpochBasedRunner

from mmdet3d.utils import get_root_logger
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet3d.hooks import *

import os.path as osp
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm


def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]

    # put model on gpus
    find_unused_parameters = cfg.get("find_unused_parameters", False)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(
            model.cuda(),
            device_ids=[0],
        )
    # #freeze_model_param
    # freeze_except_occ(model)
    # #freeze_bn
    # freeze_bn(model)
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.run_dir,
            logger=logger,
            meta={},
        ),
    )
    
    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if "cumulative_iters" in cfg.optimizer_config:
            optimizer_config = GradientCumulativeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
        custom_hooks_config=cfg.get('custom_hooks', None)
    )
    if isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
            # 保存间隔（单位：epoch），这里设置为每个 epoch 保存一次
        eval_cfg["interval"] = cfg.data.interval
            # 根据指定指标保存最佳模型（如 mAP、accuracy）
            # 根据 COCO 评估的 mAP 保存最佳模型
            # 监控的指标名称，可以是 'bbox_mAP', 'segm_mAP', 'acc' 等
        eval_cfg["save_best"] = cfg.data.save_best
        # 指标越大越好（如果是 loss，用 'less'）
        eval_cfg["rule"] = 'greater'
        # 最多保留的模型数量（避免磁盘爆炸）
        eval_cfg["max_keep_ckpts"] = cfg.data.max_keep_ckpts    
        # ====== 原有的评测钩子 ======   
        # eval_hook = DistEvalHook if distributed else EvalHook
        # ###主要是这一步
        # runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
        
        # ====== 新增自定义评测钩子 ======
        total_epochs = cfg.runner['max_epochs']
        
        if distributed:
            BaseEvalHook = DistEvalHook
        else:
            BaseEvalHook = EvalHook

        class CustomEvalHook(BaseEvalHook):
            def __init__(self, *args, start_epoch=None, **kwargs):
                super().__init__(*args, **kwargs)
                # 默认最后10个epoch开始评测
                if start_epoch is None:
                    self.start_epoch = max(0, total_epochs - 10)
                else:
                    self.start_epoch = start_epoch - 1  # 1-based转0-based
            
            def after_train_epoch(self, runner):
                if runner.epoch >= self.start_epoch:
                    super().after_train_epoch(runner)
            
            def _do_evaluate(self, runner):
                """perform evaluation and save ckpt."""
                # Synchronization of BatchNorm's buffer (running_mean
                # and running_var) is not supported in the DDP of pytorch,
                # which may cause the inconsistent performance of models in
                # different ranks, so we broadcast BatchNorm's buffers
                # of rank 0 to other ranks to avoid this.
                if self.broadcast_bn_buffer:
                    model = runner.model
                    for name, module in model.named_modules():
                        if isinstance(module,
                                    _BatchNorm) and module.track_running_stats:
                            dist.broadcast(module.running_var, 0)
                            dist.broadcast(module.running_mean, 0)

                if not self._should_evaluate(runner):
                    return

                tmpdir = self.tmpdir
                if tmpdir is None:
                    tmpdir = osp.join(runner.work_dir, '.eval_hook')

                from mmdet3d.apis import multi_gpu_test

                # Changed results to self.results so that MMDetWandbHook can access
                # the evaluation results and log them to wandb.
                results = multi_gpu_test(
                    runner.model,
                    self.dataloader,
                    tmpdir=tmpdir,
                    gpu_collect=self.gpu_collect)
                self.latest_results = results
                if runner.rank == 0:
                    print('\n')
                    runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
                    key_score = self.evaluate(runner, results)

                    # the key_score may be `None` so it needs to skip
                    # the action to save the best checkpoint
                    if self.save_best and key_score:
                        self._save_ckpt(runner, key_score)
        
        # 使用最后10个epoch方案（方案1）
        # 或使用：start_epoch=10 表示从第10个epoch开始（方案2）
        eval_hook = CustomEvalHook(
            val_dataloader, 
            start_epoch=None,  # None表示最后10个epoch
            **eval_cfg
        )
        
        runner.register_hook(eval_hook)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        if cfg.get("load_from_camera", False):
            runner.load_checkpoint(cfg.load_from_camera)
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, [("train", 1)])
