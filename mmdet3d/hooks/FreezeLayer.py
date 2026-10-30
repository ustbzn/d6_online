from mmcv.runner import Hook, HOOKS

@HOOKS.register_module()
class FreezeLayerHook(Hook):
    def __init__(self, freeze_epochs=10, frozen_layers=None):
        self.freeze_epochs = freeze_epochs
        self.frozen_layers = frozen_layers or ['backbone']
    
    def before_train_epoch(self, runner):
        if runner.epoch < self.freeze_epochs:
            self._freeze_layers(runner.model)
        else:
            self._unfreeze_layers(runner.model)
    
    def _freeze_layers(self, model):
        for name, param in model.named_parameters():
            for layer in self.frozen_layers:
                if layer in name:
                    # print(name)
                    param.requires_grad = False
    
    def _unfreeze_layers(self, model):
        for param in model.parameters():
            param.requires_grad = True