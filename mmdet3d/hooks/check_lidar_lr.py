from mmcv.runner import Hook,HOOKS

@HOOKS.register_module()
class LidarBackboneLRHook(Hook):  # zhuning
    def __init__(self, interval=1):
        self.interval = interval

    # def after_train_iter(self, runner):
    #     if self.every_n_iters(runner, self.interval):
    #         for param_group in runner.optimizer.param_groups:
    #             breakpoint()
    #             if 'encoders.lidar' in param_group.get('name', ''):
    #                 runner.logger.info(
    #                     f'encoders.lidar LR: {param_group["lr"]:.2e}'
    #                 )
    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            for param_group in runner.optimizer.param_groups:
                # 获取原始模型（去掉DP/DDP封装）
                model = runner.model.module if hasattr(runner.model, 'module') else runner.model
                
                # 获取第一个参数的完整名称
                param = param_group['params'][0]
                for name, p in model.named_parameters():
                    if p is param:  # 通过内存地址匹配
                        if 'encoders.lidar' in name:
                            runner.logger.info(f'encoders.lidar LR: {param_group["lr"]:.2e}')
                        break