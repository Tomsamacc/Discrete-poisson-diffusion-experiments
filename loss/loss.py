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


def normalized_ratio_targets(x, z, k_max=3):
    """T_k = X^k / [(z+1)...(z+k)], shape [B, k_max]. Recurrence, float64."""
    x = x.reshape(-1).double()
    z = z.reshape(-1).double()
    T = torch.empty(x.shape[0], int(k_max), device=x.device, dtype=torch.float64)
    running = torch.ones_like(x)
    for k in range(1, int(k_max) + 1):
        running = running * x / (z + float(k))
        T[:, k - 1] = running
    return T


def posterior_moments_from_logB(log_B, z):
    """m_k = (z+1)^↑k exp(b_k) = E[X^k | z]. Returns list of [B, 1] float32."""
    z = z.reshape(-1).double()
    B = torch.exp(log_B.double())
    rising = torch.ones_like(z)
    moms = []
    for k in range(1, int(log_B.shape[-1]) + 1):
        rising = rising * (z + float(k))
        moms.append((rising * B[:, k - 1]).float().view(-1, 1))
    return moms


class NormalizedRatioLoss(nn.Module):
    """ℓ_k = exp(b_k) - T_k b_k,  b_k = log B_k."""

    def __init__(self, weights=None, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        if weights is None:
            self.weights = None
        else:
            self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float64))

    def forward(self, log_B, target_T):
        log_B = log_B.double()
        target_T = target_T.double()
        per_k = torch.exp(log_B) - target_T * log_B
        w = self.weights
        if w is None:
            w = torch.ones(log_B.shape[-1], device=log_B.device, dtype=log_B.dtype)
        else:
            w = w.to(device=log_B.device, dtype=log_B.dtype)
        div = (per_k * w).sum(dim=-1)
        return _reduce(div, self.reduction)
