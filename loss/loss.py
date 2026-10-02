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


def rising_factorial(z, k_max):
    """D_k = (z+1)^↑k, shape [B, k_max], float64."""
    z = z.reshape(-1).double()
    D = torch.empty(z.shape[0], int(k_max), device=z.device, dtype=torch.float64)
    running = torch.ones_like(z)
    for k in range(1, int(k_max) + 1):
        running = running * (z + float(k))
        D[:, k - 1] = running
    return D


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


def raw_ratio_targets(x, z, gamma, k_max=3):
    """R_k = (γX)^k / D_k(z),  E[R_k|z] = S_k = p(z+k)/p(z)."""
    x = x.reshape(-1).double()
    z = z.reshape(-1).double()
    g = torch.as_tensor(gamma, device=x.device, dtype=torch.float64).reshape(-1)
    if g.numel() == 1:
        g = g.expand(x.shape[0])
    gx = g * x
    R = torch.empty(x.shape[0], int(k_max), device=x.device, dtype=torch.float64)
    running = torch.ones_like(x)
    for k in range(1, int(k_max) + 1):
        running = running * gx / (z + float(k))
        R[:, k - 1] = running
    return R


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


def posterior_moments_from_logS(log_S, z, gamma, gmin=1e-2, prior=None):
    """log m_k = a_k + log D_k - k log γ. γ<gmin uses prior E[X^k] if given."""
    z = z.reshape(-1).double()
    a = log_S.double()
    g = torch.as_tensor(gamma, device=z.device, dtype=torch.float64).reshape(-1)
    if g.numel() == 1:
        g = g.expand(z.shape[0])
    # Float32 quadratic node at t_eps can sit just under gmin. Only gamma=0 uses the prior.
    tiny = g < float(gmin) * (1.0 - 1e-3)
    log_D = torch.zeros_like(z)
    log_g = g.clamp_min(float(gmin)).log()
    moms = []
    for k in range(1, int(a.shape[-1]) + 1):
        log_D = log_D + (z + float(k)).clamp_min(1e-12).log()
        log_m = a[:, k - 1] + log_D - float(k) * log_g
        m = log_m.exp()
        if prior is not None and tiny.any():
            m = torch.where(tiny, torch.full_like(m, float(prior[k - 1])), m)
        moms.append(m.float().view(-1, 1))
    return moms


def uses_raw_log_s(args):
    """Network output is log S_k, and m_k = D_k exp(a_k) / gamma^k."""
    if str(getattr(args, "ratio_loss", "normalized")) == "raw":
        return True
    if str(getattr(args, "moment_param", "")) == "raw":
        return True
    return str(getattr(args, "moment_loss", "")) in ("raw_ratio", "raw_multi", "hybrid", "raw_bal")


def moments_from_ratio_pred(pred, z, gamma, args):
    if uses_raw_log_s(args):
        gmin = float(getattr(args, "t_eps", 1e-4)) * float(getattr(args, "lbd", 100.0))
        return posterior_moments_from_logS(
            pred, z, gamma, gmin=gmin, prior=getattr(args, "prior_moments", None)
        )
    return posterior_moments_from_logB(pred, z)


def full_bregman(y, mu):
    """D(y, μ) = y log(y/μ) - y + μ. y = 0 uses 0 log 0 = 0. Nonnegative."""
    y = y.reshape(-1).double().clamp_min(0)
    mu = mu.reshape(-1).double().clamp_min(1e-12)
    ylogy = torch.where(y > 0, torch.special.xlogy(y, y), torch.zeros_like(y))
    return ylogy - torch.special.xlogy(y, mu) - y + mu


def reduced_bregman(y, mu):
    """μ - y log μ. Same μ-gradient as full_bregman. The gap is y log y - y."""
    y = y.reshape(-1).double().clamp_min(0)
    mu = mu.reshape(-1).double().clamp_min(1e-12)
    return mu - torch.special.xlogy(y, mu)


class MomentPRLLoss(nn.Module):
    """Moment-space full Bregman D(X^k, m̂). Zero at m̂ = X^k."""

    def forward(self, m_hat, x, k):
        target = x.reshape(-1).double().clamp_min(0).pow(int(k))
        return full_bregman(target, m_hat).mean()


class RawRatioBregmanLoss(nn.Module):
    """Full Bregman D(R_k, exp(log S)). Single head, log S unconstrained."""

    def forward(self, log_s, target_r):
        s = log_s.reshape(-1).double().exp()
        r = target_r.reshape(-1).double().clamp_min(0)
        return full_bregman(r, s).mean()


def raw_multi_bregman(log_s, target_r):
    """Per-example full Bregman D(R_k, exp(a_k)). Shape [B, K]."""
    log_s = log_s.double()
    target_r = target_r.double()
    if log_s.ndim == 1:
        log_s = log_s.view(-1, 1)
        target_r = target_r.view(-1, 1)
    cols = []
    for k in range(log_s.shape[-1]):
        cols.append(full_bregman(target_r[:, k].clamp_min(0), log_s[:, k].exp()))
    return torch.stack(cols, dim=-1)


def raw_multi_head_losses(log_s, target_r):
    """Mean full Bregman per head. S = exp(a)."""
    per = raw_multi_bregman(log_s, target_r)
    return [per[:, k].mean() for k in range(per.shape[-1])]


def quad_reverse_h(t, lbd, steps):
    """Reverse step h at gamma=t*lbd on gamma(u)=lbd*u^2, T=steps.

    u=sqrt(t) is the state before the jump. u is capped at 1-1/T.
    """
    du = 1.0 / float(steps)
    u = t.reshape(-1).double().clamp_min(0).sqrt().clamp(max=1.0 - du)
    return float(lbd) * ((u + du).pow(2) - u.pow(2))


def raw_multi_weights(t, mode, lbd, steps, k_max=3):
    """Per-example head weights, shape [B, K].

    equal: 1, 1, 1
    dyn: h, h^2/2, h^3/6
    dyn_norm: those divided by their sum
    """
    mode = str(mode)
    n = t.reshape(-1).shape[0]
    if mode == "equal":
        return torch.ones(n, int(k_max), dtype=torch.float64, device=t.device)
    h = quad_reverse_h(t, lbd, steps)
    fact = 1.0
    hk = torch.ones_like(h)
    cols = []
    for k in range(1, int(k_max) + 1):
        fact *= float(k)
        hk = hk * h
        cols.append(hk / fact)
    w = torch.stack(cols, dim=-1)
    if mode == "dyn_norm":
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    elif mode != "dyn":
        raise ValueError(mode)
    return w


HIGHER_PARAMS = ("raw_log", "direct_ratio", "log_moment", "root_moment", "direct_moment")


def higher_moment_maps(pred, z, gamma, param, gmin=1e-2, prior=None):
    """Three heads. m1 is always softplus(f1).

    k=2,3 follow param. Returns m, s with shape [B, 3], float64.
    raw_log:      S=exp(a),           m=c S
    direct_ratio: S=softplus(u),      m=c S
    log_moment:   m=exp(b),           S=m/c
    root_moment:  m=softplus(r)^k,    S=m/c
    direct_moment: m=softplus(v),     S=m/c
    c_k = (z+1)...(z+k) / gamma^k.
    gamma below gmin uses that floor. raw_log and direct_ratio replace
    m2, m3 with prior moments when gamma is below the floor.
    """
    if param not in HIGHER_PARAMS:
        raise ValueError(param)
    pred = pred.double()
    if pred.ndim == 1:
        pred = pred.view(-1, 1)
    z = z.reshape(-1).double()
    g = torch.as_tensor(gamma, device=z.device, dtype=torch.float64).reshape(-1)
    if g.numel() == 1:
        g = g.expand(z.shape[0])
    tiny = g < float(gmin) * (1.0 - 1e-3)
    g_use = torch.where(g < float(gmin), torch.full_like(g, float(gmin)), g)
    log_g = g_use.log()
    m = torch.empty(pred.shape[0], 3, dtype=torch.float64, device=pred.device)
    s = torch.empty_like(m)
    m[:, 0] = torch.nn.functional.softplus(pred[:, 0])
    s[:, 0] = m[:, 0]
    log_D = torch.zeros_like(z)
    for k in range(1, 4):
        log_D = log_D + (z + float(k)).clamp_min(1e-12).log()
        if k == 1:
            continue
        log_c = log_D - float(k) * log_g
        a = pred[:, k - 1]
        if param == "raw_log":
            sk = a.exp()
            mk = (a + log_c).exp()
        elif param == "direct_ratio":
            sk = torch.nn.functional.softplus(a)
            mk = (sk.clamp_min(1e-12).log() + log_c).exp()
        elif param == "log_moment":
            mk = a.exp()
            sk = (a - log_c).exp()
        elif param == "root_moment":
            sp = torch.nn.functional.softplus(a).clamp_min(1e-12)
            mk = sp.pow(float(k))
            sk = (float(k) * sp.log() - log_c).exp()
        else:
            mk = torch.nn.functional.softplus(a)
            sk = (mk.clamp_min(1e-12).log() - log_c).exp()
        if prior is not None and param in ("raw_log", "direct_ratio") and bool(tiny.any()):
            mk = torch.where(tiny, torch.full_like(mk, float(prior[k - 1])), mk)
        m[:, k - 1] = mk
        s[:, k - 1] = sk
    return m, s


def linked_mean(pred, z, gamma, kind, gmin=1e-2):
    """soft_ratio: (z+1)/gamma * softplus(a). offset: exp(b)."""
    a = pred.reshape(-1).double()
    if kind == "offset":
        return a.exp()
    if kind != "soft_ratio":
        raise ValueError(kind)
    g = torch.as_tensor(gamma, device=a.device, dtype=torch.float64).reshape(-1)
    if g.numel() == 1:
        g = g.expand(a.shape[0])
    g = torch.where(g < float(gmin), torch.full_like(g, float(gmin)), g)
    s = torch.nn.functional.softplus(a)
    return (z.reshape(-1).double() + 1.0) / g * s


def moment_from_param(pred, z, gamma, k, param):
    """direct: softplus m̂. ratio: D_k e^b. raw: D_k e^a / γ^k."""
    if param == "direct":
        return pred.reshape(-1, 1)
    b = pred.reshape(-1).double()
    log_D = rising_factorial(z, int(k))[:, int(k) - 1].clamp_min(1e-12).log()
    log_m = b + log_D
    if param == "raw":
        g = torch.as_tensor(gamma, device=b.device, dtype=torch.float64).reshape(-1)
        if g.numel() == 1:
            g = g.expand(b.shape[0])
        log_m = log_m - float(k) * g.clamp_min(1e-12).log()
    elif param != "ratio":
        raise ValueError(param)
    return log_m.exp().float().view(-1, 1)


class NormalizedRatioLoss(nn.Module):
    """ℓ_k = exp(b_k) - T_k b_k. If balanced, multiply by D_k = (z+1)^↑k."""

    def __init__(self, weights=None, reduction="mean", balanced=False):
        super().__init__()
        self.reduction = reduction
        self.balanced = bool(balanced)
        if weights is None:
            self.weights = None
        else:
            self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float64))

    def forward(self, log_B, target_T, z=None):
        log_B = log_B.double()
        target_T = target_T.double()
        per_k = torch.exp(log_B) - target_T * log_B
        if self.balanced:
            if z is None:
                raise ValueError("balanced ratio loss needs z")
            per_k = per_k * rising_factorial(z, log_B.shape[-1])
        w = self.weights
        if w is None:
            w = torch.ones(log_B.shape[-1], device=log_B.device, dtype=log_B.dtype)
        else:
            w = w.to(device=log_B.device, dtype=log_B.dtype)
        div = (per_k * w).sum(dim=-1)
        return _reduce(div, self.reduction)
