# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS

import inspect

__all__ = ["SECONDFPN"]
def class2dic(config):
    assert inspect.isclass(config)
    config_dic = dict(config.__dict__)
    del_key_list = []
    for key in config_dic:
        if key.startswith('__') and key.endswith('__'):
            del_key_list.append(key)
    
    for key in del_key_list:
        config_dic.pop(key)
    return config_dic
def get_module(config=None, *args, **kwargs):
    if config != None:
        if inspect.isclass(config):
            config = class2dic(config)
        elif isinstance(config, str):
            config = dict(type=config)
        else:
            assert isinstance(config, dict)
        
        for key in config:
            kwargs[key] = config[key]
    
    assert 'type' in kwargs
    method_code = eval(kwargs['type'])

    args_count = method_code.__init__.__code__.co_argcount
    input_params = method_code.__init__.__code__.co_varnames[1:args_count]

    new_kwargs = {}
    for i, value in enumerate(args):
        new_kwargs[input_params[i]] = value
    
    for key in kwargs:
        if key in input_params:
            new_kwargs[key] = kwargs[key]
    
    result_module = q(**new_kwargs)
    return result_module

class conv3x3_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, groups=1):
        super(conv3x3_bn_relu, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.net(x)
        return x1
class conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, groups=1, bias=False):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=bias)
    
    def forward(self, x):
        return self.conv(x)

class AttMerge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(AttMerge, self).__init__()
        self.scale_factor = scale_factor
        self.cout = cout

        # self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        self.upsample_layer = build_upsample_layer(
                    dict(type="deconv", bias=False),
                    in_channels=cin_high,
                    out_channels=cin_high,
                    kernel_size=2,
                    stride=2,
        )
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.att_layer = nn.Sequential(
            conv3x3_bn_relu(2 * cout, cout // 2, stride=1, dilation=1),
            conv3x3(cout // 2, 1, stride=1, dilation=1, bias=True),
            nn.Sigmoid()
        )
        self.conv_high = conv3x3_bn_relu(cin_high, cout, stride=1, dilation=1)
        self.conv_low = conv3x3_bn_relu(cin_low, cout, stride=1, dilation=1)
    
    def forward(self, x_low, x_high):
        #pdb.set_trace()
        # x_high_up = self.upsample(x_high)
        x_high_up = self.upsample_layer(x_high)
        x_low_feat = self.conv_low(x_low)
        x_high_up_feat = self.conv_high(x_high_up)

        x_merge = torch.cat((x_low_feat, x_high_up_feat), dim=1) #(BS, 2*channels, H, W)
        x_merge = self.dropout(x_merge)

        # attention fusion
        ca_map = self.att_layer(x_merge)
        x_out = x_low_feat * ca_map + x_high_up_feat * (1 - ca_map)
        return x_out
@NECKS.register_module()
class SECONDFPN(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(
        self,
        in_channels=[128, 128, 256],
        out_channels=[256, 256, 256],
        out_channel=256,
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        conv_cfg=dict(type="Conv2d", bias=False),
        use_conv_for_no_stride=False,
        init_cfg=None,
    ):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.out_channel = out_channel

        self.decoder_neck_blocks = nn.ModuleList()
        L = len(self.in_channels)
        for i in range(1, L-1):
                self.decoder_neck_blocks.append(AttMerge(cin_low=self.in_channels[  L - i - 1],cin_high=self.in_channels[L-i],cout= self.out_channel, scale_factor=upsample_strides[ L - i ]))
        self.conv3x3_bn_relu = conv3x3_bn_relu(self.in_channels[ 0 ], self.out_channel, stride=1, dilation=1)

        # deblocks = []
        # for i, out_channel in enumerate(out_channels):
        #     stride = upsample_strides[i]
        #     if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
        #         upsample_layer = build_upsample_layer(
        #             upsample_cfg,
        #             in_channels=in_channels[i],
        #             out_channels=out_channel,
        #             kernel_size=upsample_strides[i],
        #             stride=upsample_strides[i],
        #         )
        #     else:
        #         stride = np.round(1 / stride).astype(np.int64)
        #         upsample_layer = build_conv_layer(
        #             conv_cfg,
        #             in_channels=in_channels[i],
        #             out_channels=out_channel,
        #             kernel_size=stride,
        #             stride=stride,
        #         )

        #     deblock = nn.Sequential(
        #         upsample_layer,
        #         build_norm_layer(norm_cfg, out_channel)[1],
        #         nn.ReLU(inplace=True),
        #     )
        #     deblocks.append(deblock)
        # self.deblocks = nn.ModuleList(deblocks)

        # if init_cfg is None:
        #     self.init_cfg = [
        #         dict(type="Kaiming", layer="ConvTranspose2d"),
        #         dict(type="Constant", layer="NaiveSyncBatchNorm2d", val=1.0),
        #     ]

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        # assert len(x) == len(self.in_channels)
        # ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        # if len(ups) > 1:
        #     out = torch.cat(ups, dim=1)
        # else:
        #     out = ups[0]
        # return [out]
        L = len(x)
        x_merge = self.decoder_neck_blocks[0](x[L-2], x[ L-1])
        for i in range(1,len(self.decoder_neck_blocks)):
            x_merge = self.decoder_neck_blocks[i](x[ L- i - 2],x_merge)
        # x_merge=self.conv3x3_bn_relu(x[ 0 ])+ x_merge
        out = torch.cat((self.conv3x3_bn_relu(x[ 0 ]),x_merge), dim=1)
        return [out]

