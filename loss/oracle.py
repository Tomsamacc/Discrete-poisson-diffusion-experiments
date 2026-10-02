"""Exact / quadrature posterior moments E[X^k | Z=z, γ] for the 1D toys."""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats


GAMMA_EDGES = (0.1, 1.0, 10.0, 100.0)
BIN_LABELS = ("[0,0.1]", "[0.1,1]", "[1,10]", "[10,100]")
DISCRETE_BOUNDS = {
    "pois20": (0, 50),
    "nb": (0, 120),
    "poismix_mod": (0, 90),
    "poissmix": (0, 160),
    "poissmix3": (0, 150),
    "zip": (0, 25),
    "yule_simon": (1, 2000),
}


def discrete_pmf(name, ks):
    ks = np.asarray(ks)
    if name == "nb":
        return stats.nbinom.pmf(ks, n=5, p=0.2)
    if name == "pois20":
        return stats.poisson.pmf(ks, 20)
    if name == "poismix_mod":
        return 0.5 * stats.poisson.pmf(ks, 5) + 0.5 * stats.poisson.pmf(ks, 50)
    if name == "poissmix":
        return 0.1 * stats.poisson.pmf(ks, 1) + 0.9 * stats.poisson.pmf(ks, 100)
    if name == "poissmix3":
        return (
            (1.0 / 3.0) * stats.poisson.pmf(ks, 1)
            + (1.0 / 3.0) * stats.poisson.pmf(ks, 50)
            + (1.0 / 3.0) * stats.poisson.pmf(ks, 100)
        )
    if name == "zip":
        p = 0.3 * stats.poisson.pmf(ks, 5.0)
        return np.where(ks == 0, p + 0.7, p)
    if name == "yule_simon":
        return stats.yulesimon.pmf(ks, 2.0)
    raise ValueError(name)


def support_logp(name, device):
    """xs [S], log p(x) [S] on `device`."""
    if name == "gamma_ltj":
        xs = np.linspace(1e-4, float(stats.gamma.ppf(0.9999, a=1.0, scale=10.0)), 1024)
        logp = stats.gamma.logpdf(xs, a=1.0, scale=10.0)
        logp = np.where(np.isfinite(logp), logp, -1e20)
        return (
            torch.tensor(xs, device=device, dtype=torch.float32),
            torch.tensor(logp, device=device, dtype=torch.float32),
        )
    lo, hi = DISCRETE_BOUNDS[name]
    ks = np.arange(lo, hi + 1)
    p = np.clip(discrete_pmf(name, ks), 0.0, None)
    p = p / max(float(p.sum()), 1e-12)
    logp = np.log(np.clip(p, 1e-300, 1.0))
    return (
        torch.tensor(ks, device=device, dtype=torch.float32),
        torch.tensor(logp, device=device, dtype=torch.float32),
    )


def posterior_weights(z, gamma, xs, logp):
    z = z.reshape(-1, 1).float()
    g = torch.as_tensor(gamma, device=z.device, dtype=torch.float32).reshape(-1)
    if g.numel() == 1:
        g = g.expand(z.shape[0])
    g = g.view(-1, 1).clamp_min(1e-12)
    rate = (g * xs.view(1, -1).clamp_min(1e-12)).clamp_min(1e-12)
    log_lik = z * rate.log() - rate - torch.lgamma(z + 1.0)
    return torch.softmax(log_lik + logp.view(1, -1), dim=-1)


def poisson_mixture_pmf(w, xs, h, kmax=3):
    """q(k) = sum_x w(x) Pois(k; h x) for k = 0..kmax. tail = P(K > kmax).

    w: [B, S], xs: [S], h: scalar. Returns q [B, kmax+1], tail [B], float64.
    """
    w64 = w.double()
    rate = (float(h) * xs.double().clamp_min(0)).view(1, -1)
    ks = torch.arange(int(kmax) + 1, device=w.device, dtype=torch.float64)
    logp = ks[:, None] * rate.clamp_min(1e-30).log() - rate - torch.lgamma(ks[:, None] + 1.0)
    pmf = logp.exp()
    zero = rate.view(-1) <= 0
    if bool(zero.any()):
        pmf = pmf.clone()
        pmf[:, zero] = 0
        pmf[0, zero] = 1
    q = w64 @ pmf.transpose(0, 1)
    q = q.clamp_min(0)
    tail = (1.0 - q.sum(dim=-1)).clamp_min(0)
    return q, tail


def oracle_moments(z, gamma, xs, logp, k_max=3):
    """m_k* = E[X^k | Z_γ=z], list of [B, 1]."""
    w = posterior_weights(z, gamma, xs, logp)
    x = xs.view(1, -1).to(dtype=w.dtype)
    xk = torch.ones_like(w)
    moms = []
    for _ in range(int(k_max)):
        xk = xk * x
        moms.append((w * xk).sum(-1, keepdim=True))
    return moms


def gamma_bin_index(gamma, edges=GAMMA_EDGES):
    g = torch.as_tensor(gamma, dtype=torch.float32).reshape(-1)
    bounds = torch.tensor(list(edges), device=g.device, dtype=g.dtype)
    return torch.bucketize(g, bounds, right=False).clamp(max=len(edges) - 1)


def signed_log_summary(hat, star, eps=1e-12):
    """e = log mhat - log m*. Returns median / p90 / p99 / max and positive-tail stats."""
    h = np.asarray(hat, dtype=np.float64).reshape(-1)
    s = np.asarray(star, dtype=np.float64).reshape(-1)
    e = np.log(np.clip(h, eps, None)) - np.log(np.clip(s, eps, None))
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "pos_frac": float("nan"),
            "pos_p90": float("nan"),
        }
    pos = e[e > 0]
    return {
        "n": int(e.size),
        "median": float(np.median(e)),
        "p90": float(np.quantile(e, 0.90)),
        "p99": float(np.quantile(e, 0.99)),
        "max": float(np.max(e)),
        "pos_frac": float((e > 0).mean()),
        "pos_p90": float(np.quantile(pos, 0.90)) if pos.size else 0.0,
    }


def format_mom_line(means, counts=None):
    """means: [k_max, n_bins] tensor/ndarray."""
    arr = np.asarray(means, dtype=np.float64)
    parts = []
    for k in range(arr.shape[0]):
        bits = " ".join(f"{lab}={arr[k, i]:.3g}" for i, lab in enumerate(BIN_LABELS))
        parts.append(f"|log m{k + 1}| {bits}")
    return "  ".join(parts)
