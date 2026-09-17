import torch
import torch.nn as nn


def _reduce(div, reduction):
    if div.ndim > 1:
        div = div.sum(dim=-1)
    if reduction == "mean":
        return div.mean()
    if reduction == "sum":
        return div.sum()
    return div


class BregmanLoss(nn.Module):

    def __init__(self, reduction="mean", eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        q = pred.clamp_min(self.eps)
        p = target.clamp_min(self.eps)
        div = p * (p / q).log() - p + q
        return _reduce(div, self.reduction)
