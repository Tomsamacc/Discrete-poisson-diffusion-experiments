"""Ratio-consistency residuals, penalties, and Euclidean projection.

The identity is log S_2(z) = log S_1(z) + log S_1(z+1) and
log S_3(z) = log S_1(z) + log S_1(z+1) + log S_1(z+2).
Data loss stays outside this file.
"""
from __future__ import annotations

import torch


def constraint_A(device=None, dtype=torch.float64):
    """Rows are (delta_2, delta_3) on [a1(z), a1(z+1), a1(z+2), a2(z), a3(z)]."""
    return torch.tensor(
        [
            [-1.0, -1.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def _as_cols(a):
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a.reshape(a.shape[0], -1)


def _col0(a):
    if a.ndim == 1:
        return a.reshape(-1)
    return a.reshape(a.shape[0], -1)[:, 0]


def raw_deltas(a_z, a1, a2):
    """delta_2, delta_3 from network logs. a_z is [B, 3] or more."""
    cols = _as_cols(a_z)
    b0 = cols[:, 0]
    b1 = _col0(a1)
    b2 = _col0(a2)
    d2 = cols[:, 1] - b0 - b1
    d3 = cols[:, 2] - b0 - b1 - b2
    return d2, d3


def expand_hard(a_z, a1, a2):
    """a2 = a1(z)+a1(z+1), a3 = a2+a1(z+2). Returns [B, 3]."""
    b0 = _col0(a_z)
    b1 = _col0(a1)
    b2 = _col0(a2)
    return torch.stack([b0, b0 + b1, b0 + b1 + b2], dim=-1)


def project_u(u):
    """Euclidean projection onto A u = 0. u is [B, 5]. Returns u_proj, A u_proj."""
    A = constraint_A(u.device, torch.float64)
    x = u.reshape(u.shape[0], 5).double()
    Au = x @ A.T
    w = torch.linalg.solve(A @ A.T, Au.T).T
    up = (x - w @ A).to(dtype=u.dtype)
    resid = up.double() @ A.T
    return up, resid.to(dtype=u.dtype)


def project_heads(a_z, a1, a2):
    """Project the local 5-vector and return log S at z plus the projected residual."""
    cols = _as_cols(a_z)
    if cols.shape[1] < 3:
        raise ValueError("projection needs three log-ratio heads at z")
    u = torch.stack(
        [cols[:, 0], _col0(a1), _col0(a2), cols[:, 1], cols[:, 2]],
        dim=-1,
    )
    up, resid = project_u(u)
    log_s = torch.stack([up[:, 0], up[:, 3], up[:, 4]], dim=-1)
    return log_s, resid[:, 0], resid[:, 1]


def assemble_log_s(a_z, a1, a2, mode):
    """Log-ratios used for moments, their residuals, and the pre-projection residuals.

    mode hard builds S2, S3 from S1. mode project replaces the 5-vector.
    soft, anchored, and none use the three heads at z.
    Returns log_s [B, 3], delta2, delta3, delta2_in, delta3_in.
    """
    mode = str(mode or "none")
    if mode == "hard":
        log_s = expand_hard(a_z, a1, a2)
        zero = log_s.new_zeros(log_s.shape[0])
        return log_s, zero, zero, zero, zero
    d2, d3 = raw_deltas(a_z, a1, a2)
    if mode == "project":
        log_s, r2, r3 = project_heads(a_z, a1, a2)
        return log_s, r2, r3, d2, d3
    cols = _as_cols(a_z)
    if cols.shape[1] < 3:
        raise ValueError(f"consistency mode {mode} needs three log-ratio heads")
    return cols[:, :3], d2, d3, d2, d3


def delta_squares(pred, a1, a2, mode):
    """Per-example delta_2^2 and delta_3^2. anchored stop-grads the a1 sum."""
    mode = str(mode)
    base = _as_cols(pred)
    s1 = _col0(a1)
    s2 = _col0(a2)
    if mode == "anchored":
        target2 = (base[:, 0] + s1).detach()
        target3 = (base[:, 0] + s1 + s2).detach()
        d2 = base[:, 1] - target2
        d3 = base[:, 2] - target3
    elif mode == "soft":
        d2 = base[:, 1] - base[:, 0] - s1
        d3 = base[:, 2] - base[:, 0] - s1 - s2
    else:
        raise ValueError(mode)
    return d2.square(), d3.square()


def penalty_from_logs(pred, a1, a2, mode, weights=None):
    """Mean consistency penalty. weights [B, 3] uses columns w2, w3."""
    d2, d3 = delta_squares(pred, a1, a2, mode)
    if weights is None:
        return (d2 + d3).mean()
    w = weights.reshape(d2.shape[0], -1).to(dtype=d2.dtype)
    return (w[:, 1] * d2 + w[:, 2] * d3).mean()


def consistency_squares(model, z, t, alpha, pred, mode):
    """Per-example delta_2^2, delta_3^2. Same t and gamma at z, z+1, z+2."""
    mode = str(mode)
    if mode == "anchored":
        with torch.no_grad():
            a1 = model(z + 1.0, t, alpha=alpha)
            a2 = model(z + 2.0, t, alpha=alpha)
    elif mode == "soft":
        a1 = model(z + 1.0, t, alpha=alpha)
        a2 = model(z + 2.0, t, alpha=alpha)
    else:
        raise ValueError(mode)
    return delta_squares(pred, a1, a2, mode)


def consistency_penalty(model, z, t, alpha, pred, mode, weights=None):
    """Mean of delta_2^2 + delta_3^2, or of w2*delta_2^2 + w3*delta_3^2."""
    d2, d3 = consistency_squares(model, z, t, alpha, pred, mode)
    if weights is None:
        return (d2 + d3).mean()
    w = weights.reshape(d2.shape[0], -1).to(dtype=d2.dtype)
    return (w[:, 1] * d2 + w[:, 2] * d3).mean()
