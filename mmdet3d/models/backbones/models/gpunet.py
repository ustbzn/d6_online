import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from mmcv.runner import BaseModule
from mmdet.models import BACKBONES
from .gpunet_builder import GPUNet_Builder # 确保这些导入是正确的
from ..configs.model_hub import get_configs, get_model_list
from ..configs.gpunet_torchhub import nvidia_gpunet
import torch
from collections import OrderedDict
from timm.models import (
    load_checkpoint,
)

@BACKBONES.register_module()
class GPUNetWrapper(BaseModule):
    def __init__(self, model_type, precision, img_height, img_width, out_indices, latency, gpuType, batch=1, init_cfg=None, pretrained=None):
        super().__init__(init_cfg)
        self.model_type = model_type
        self.precision = precision
        self.img_height = img_height
        self.img_width = img_width
        self.out_indices = out_indices
        self.latency = latency
        self.gpuType = gpuType
        self.batch = batch
        self.pretrained = pretrained
        
        modelJSON, _ = get_configs(batch=self.batch, latency=self.latency, gpuType=self.gpuType)#读取json参数文件
        builder = GPUNet_Builder()
        self.model = builder.get_model(modelJSON)#根据参数构建模型
        pretrain_modelJSON, _ = get_configs(batch=self.batch, latency=self.latency, gpuType="GV100")#这里为什么加载V100的上的预训练模型参数？因为官网只提供了V100上的预训练模型
        pretrain_model = builder.get_model(pretrain_modelJSON)
        if pretrained:
            load_checkpoint(pretrain_model, rootPath+pretrained, use_ema=True)#加载预训练权重
            pretrained_state_dict = pretrain_model.state_dict()
            model_state_dict = self.model.state_dict()
            new_state_dict = OrderedDict()
            for k, v in pretrained_state_dict.items():
                # 检查你的模型是否有相应的层
                if k in model_state_dict and v.size() == model_state_dict[k].size():
                    # 如果有，并且形状匹配，则保留这部分权重
                    new_state_dict[k] = v
            # 更新你的模型的状态字典
            model_state_dict.update(new_state_dict)

            # 加载新的状态字典到你的模型
            self.model.load_state_dict(model_state_dict, strict=False)
            
        


    def forward(self, x):
        features = []
        for i, layer in enumerate(self.model.network):
            x = layer(x)
            if i in self.out_indices:
                features.append(x)
        return tuple(features)


def main():
    modelJSON, cpkPath = get_configs(batch=1, latency="0.85ms", gpuType="orin",download=False,config_root_dir="../configs")

    #print(get_model_list(1))

    builder = GPUNet_Builder()
    model = builder.get_model(modelJSON)
    builder.export_onnx(model)
    print(model, model.img_height, model.img_width)
    random_input = torch.randn(5, 3, 672, 1152)  # (batch_size, channels, height, width)
    model_output = model(random_input)
    print("Model output shape:", [output.shape for output in model_output])
    print("Model output shape:", model_output.shape)

if __name__ == '__main__':
    main()