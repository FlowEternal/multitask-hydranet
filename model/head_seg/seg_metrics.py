# ==================================================================
# Author    : Dongxu Zhan
# Time      : 2021/7/28 10:57
# File      : metric.py
# Function  : validation metrics
# ==================================================================

import torch
import torch.nn
from typing import Optional

def stat_scores_multiple_classes(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None):

    if pred.dtype != torch.bool:
        pred = pred.clamp_max(max=num_classes)
    if target.dtype != torch.bool:
        target = target.clamp_max(max=num_classes)

    pred = pred.view((-1, )).long()
    target = target.view((-1, )).long()

    tps = torch.zeros((num_classes + 1, ), device=pred.device)
    fps = torch.zeros((num_classes + 1, ), device=pred.device)
    fns = torch.zeros((num_classes + 1, ), device=pred.device)
    sups = torch.zeros((num_classes + 1, ), device=pred.device)

    match_true = (pred == target).float()
    match_false = 1 - match_true

    tps.scatter_add_(0, pred, match_true)
    fps.scatter_add_(0, pred, match_false)
    fns.scatter_add_(0, target, match_false)
    tns = pred.size(0) - (tps + fps + fns)
    sups.scatter_add_(0, target, torch.ones_like(match_true))

    tps = tps[:num_classes]
    fps = fps[:num_classes]
    tns = tns[:num_classes]
    fns = fns[:num_classes]
    sups = sups[:num_classes]
    return tps.float(), fps.float(), tns.float(), fns.float(), sups.float()

#---------------------------------------------------#
#  semantic segmentation
#---------------------------------------------------#
class IntersectionOverUnion(object):
    """Computes intersection-over-union."""
    def __init__(
        self,
        n_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        reduction: str = 'none'):

        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.absent_score = absent_score
        self.reduction = reduction

        self.true_positive =torch.zeros(n_classes)
        self.false_positive =torch.zeros(n_classes)
        self.false_negative =torch.zeros(n_classes)
        self.support =torch.zeros(n_classes)

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        tps, fps, _, fns, sups = stat_scores_multiple_classes(prediction, target,self.n_classes)
        self.true_positive += tps
        self.false_positive += fps
        self.false_negative += fns
        self.support += sups

    def compute(self):
        scores = torch.zeros(self.n_classes, device=self.true_positive.device, dtype=torch.float32)

        for class_idx in range(self.n_classes):
            if class_idx == self.ignore_index:
                continue

            tp = self.true_positive[class_idx]
            fp = self.false_positive[class_idx]
            fn = self.false_negative[class_idx]
            sup = self.support[class_idx]

            # If this class is absent in the target (no support) AND absent in the pred (no true or false
            # positives), then use the absent_score for this class.
            if sup + tp + fp == 0:
                scores[class_idx] = self.absent_score
                continue

            denominator = tp + fp + fn
            score = tp.to(torch.float) / denominator
            scores[class_idx] = score

        # Remove the ignored class index from the scores.
        if (self.ignore_index is not None) and (0 <= self.ignore_index < self.n_classes):
            scores = torch.cat([scores[:self.ignore_index], scores[self.ignore_index+1:]])

        return scores
