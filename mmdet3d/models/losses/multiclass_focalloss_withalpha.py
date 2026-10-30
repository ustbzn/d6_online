# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmdet.models import LOSSES

__all__ = ["MultiClassFocalLossWithAlpha"]

@LOSSES.register_module()
class MultiClassFocalLossWithAlpha(nn.Module):
    """
    Refer to https://zhuanlan.zhihu.com/p/562641889
    """
    
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean', loss_weight=1.0):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        # import ipdb;ipdb.set_trace()
        alpha = self.alpha.cuda()[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        #print('alpha: ', alpha, ' gamma: ', self.gamma, ' focal loss: ', torch.mean(focal_loss))
        if self.reduction == "mean":
            return self.loss_weight * torch.mean(focal_loss)
        if self.reduction == "sum":
            return self.loss_weight * torch.sum(focal_loss)
        return self.loss_weight * focal_loss
