# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.amp as amp
try:
    from itertools import  ifilterfalse
except ImportError:
    from itertools import  filterfalse as ifilterfalse

from mmdet.models import LOSSES

__all__ = ["LovaszSoftmaxV1","GeometryAffinityLoss","SemanticAffinityLoss"]

def isnan(x):
    return x != x

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


@LOSSES.register_module()
class LovaszSoftmaxV1(nn.Module):
    '''
    This is the autograd version, used in the multi-category classification case
    '''
    def __init__(self, loss_weight=1.0, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmaxV1, self).__init__()
        self.loss_weight = loss_weight
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore_index
        
    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1: # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
        
    def lovasz_softmax_flat(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float() # foreground for class c
            if (classes is 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted))))
        return mean(losses)

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 2:
            if ignore is not None:
                valid = (labels != ignore)
                probas = probas[valid]
                labels = labels[valid]
            return probas, labels

        elif probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        elif probas.dim() == 5:
            #3D segmentation
            B, C, L, H, W = probas.size()
            probas = probas.contiguous().view(B, C, L, H*W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels
    
    # def forward(self, logits, label):
    def forward(self, probas, labels):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """
        if self.per_image:
            loss = mean(self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), self.ignore), classes=self.classes)
                            for prob, lab in zip(probas, labels))
        else:
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, self.ignore), classes=self.classes)
        return loss * self.loss_weight
    

@LOSSES.register_module()
class GeometryAffinityLoss(nn.Module):
    def __init__(self, loss_weight=1.0, ignore_index=255, non_empty_idx=0):
        super(GeometryAffinityLoss, self).__init__()
        self.ignore_index = ignore_index
        self.non_empty_idx = non_empty_idx
        self.loss_weight = loss_weight

    # def forward(self, logits, label):
    def forward(self, pred, ssc_target):

        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, self.non_empty_idx]
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = ssc_target != self.ignore_index
        nonempty_target = ssc_target != self.non_empty_idx
        nonempty_target = nonempty_target[mask].float()
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        eps = 1e-5
        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / (nonempty_probs.sum()+eps)
        recall = intersection / (nonempty_target.sum()+eps)
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum()+eps)
        
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                F.binary_cross_entropy(precision.float(), torch.ones_like(precision).float())
                + F.binary_cross_entropy(recall.float(), torch.ones_like(recall).float())
                + F.binary_cross_entropy(spec.float(), torch.ones_like(spec).float())
            )
        return loss * self.loss_weight
            
        # return (
        #     F.binary_cross_entropy(precision, torch.ones_like(precision))
        #     + F.binary_cross_entropy(recall, torch.ones_like(recall))
        #     + F.binary_cross_entropy(spec, torch.ones_like(spec))
        #     ) * self.loss_weight
        
    
@LOSSES.register_module()
class SemanticAffinityLoss(nn.Module):
    def __init__(self, loss_weight=1.0, ignore_index=255):
        super(SemanticAffinityLoss, self).__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    # def forward(self, logits, label):
    def forward(self, pred, ssc_target):
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)
        loss = 0
        count = 0
        mask = ssc_target != self.ignore_index
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i]

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                with torch.cuda.amp.autocast(enabled=False):
                    if torch.sum(p) > 0:
                        precision = nominator / (torch.sum(p))
                        loss_precision = F.binary_cross_entropy(
                            precision.float(), torch.ones_like(precision).float()
                        )
                        loss_class += loss_precision
                    if torch.sum(completion_target) > 0:
                        recall = nominator / (torch.sum(completion_target))
                        loss_recall = F.binary_cross_entropy(recall.float(), torch.ones_like(recall).float())
                        loss_class += loss_recall
                    if torch.sum(1 - completion_target) > 0:
                        specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                            torch.sum(1 - completion_target)
                        )
                        loss_specificity = F.binary_cross_entropy(
                            specificity.float(), torch.ones_like(specificity).float()
                        )
                        loss_class += loss_specificity
                loss += loss_class
                
        return (loss / count) * self.loss_weight