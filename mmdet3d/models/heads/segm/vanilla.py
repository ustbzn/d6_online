from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import HEADS

__all__ = ["BEVSegmentationHead", "PredBranch"]

def lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.mean(torch.stack(losses))

def flatten_probas(probas, labels, ignore=None):
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 5:
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H*W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

def get_ohem_loss(loss_mat, valid_mask=None, top_ratio=0, top_weight=1):
    loss_mat_valid = None
    valid_num = None
    topk_num = None
    if valid_mask is not None:
        loss_mat_valid = (loss_mat * valid_mask.float()).view(-1)
        valid_num = int(valid_mask.float().sum())
        topk_num = int(valid_num * top_ratio)
    else:
        loss_mat_valid = loss_mat.view(-1)
        valid_num = loss_mat_valid.shape[0]
        topk_num = int(valid_num * top_ratio)
    
    loss_total = loss_mat_valid.sum() / (valid_num + 1e-12)
    if topk_num == 0:
        return loss_total
    else:
        loss_topk = torch.topk(loss_mat_valid, k=topk_num, dim=0, largest=True, sorted=False)[0]
        loss_total = loss_total + top_weight * loss_topk.mean()
        return loss_total

class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x

@HEADS.register_module()
class PredBranch(nn.Module):
    def __init__(
        self, 
        cin,
        cmid, 
        classes: List[str],
        loss: str,
        loss_weight_lovasz: float = 1.0,
        loss_weight_ohem: float = 1.0,
        loss_weight: float = 1.0,
        ignore_label: int = -1,
        top_ratio: float = 0.2,
        top_weight: float = 3.0
        ):
        super(PredBranch, self).__init__()

        self.pointnet = nn.Sequential(
            nn.BatchNorm2d(4),
            nn.Conv2d(4, cin, kernel_size=1, bias=False),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True)
        )

        self.layer0 = nn.Sequential(
            nn.Conv2d(2*cin, cin, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            nn.Sigmoid()
        )
        self.pred_layer = nn.Sequential(
            nn.BatchNorm2d(cmid),
            nn.Dropout(p=0.2, inplace=False),
            nn.Conv2d(cmid, len(classes), kernel_size=1, stride=1, padding=0, dilation=1)
        )
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.loss_cross = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        self.loss_lovasz = lovasz_softmax
        self.loss = loss
        self.top_ratio = top_ratio
        self.top_weight = top_weight
        self.loss_weight_lovasz = loss_weight_lovasz
        self.loss_weight_ohem = loss_weight_ohem
    def forward(self,points, x, target: Optional[torch.Tensor] = None):
        # expanded_tensors = [tensor.unsqueeze(-1) for tensor in points]

        # stacked_tensor = torch.stack(expanded_tensors)

        # final_tensor = stacked_tensor.permute(0, 2, 1, 3)
        x0=self.pointnet(points)
        x64 = torch.cat([x0,x],dim=1)
        x32 = self.layer0(x64)
        x1 = self.layer1(x32)
        x = x1 * x32
        x = self.pred_layer(x)
        if self.training:
            if self.loss == "combine":
                target = target.long()
                loss1 = self.loss_cross(x, target)
                loss1_5 = get_ohem_loss(loss1, None, top_ratio=self.top_ratio, top_weight=self.top_weight)
                loss1_5 = loss1_5*self.loss_weight_ohem
                loss2 = self.loss_lovasz(F.softmax(x, dim=1), target, ignore=self.ignore_label)
                loss2 = loss2*self.loss_weight_lovasz
                loss = self.loss_weight * (loss1_5 + loss2)
            else:
                raise ValueError(f"unsupported loss: {self.loss}")
        
            return loss
        else:
            # return torch.sigmoid(x)
            return F.softmax(x, dim=1)

@HEADS.register_module()
class BEVSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_transform: Dict[str, Any],
        classes: List[str],
        loss: str,
        loss_weight: float = 1.0,
        ignore_label: int = -1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.loss = loss
        self.ignore_label = ignore_label
        self.loss_xent = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        self.loss_lovasz = lovasz_softmax
        self.loss_weight = loss_weight

        self.transform = BEVGridTransform(**grid_transform)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True), # TODO: 2*conv -> 1*conv
            nn.Conv2d(in_channels, len(classes), 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(x, (list, tuple)):
            x = x[0]

        # x = self.transform(x)  # bn,512,328,160
        x = self.classifier(x)  # bn,512,328,160 -> bn,21,328,160

        if self.training:
            if self.loss == "combine":
                target = target.long()
                loss1 = self.loss_xent(x, target)
                loss2 = self.loss_lovasz(F.softmax(x, dim=1), target, ignore=self.ignore_label)
                loss = self.loss_weight * (loss1 + loss2)
            else:
                raise ValueError(f"unsupported loss: {self.loss}")
        
            return loss
        else:
            # return torch.sigmoid(x)
            return F.softmax(x, dim=1)
