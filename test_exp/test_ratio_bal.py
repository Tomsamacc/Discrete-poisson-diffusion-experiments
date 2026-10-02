"""CPU check: (z+1)/gamma * D(R, exp(a)) equals D(X, m) with m = (z+1)/gamma * exp(a)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.loss import full_bregman  # noqa: E402
from model.small_diffusion import PoissonDiscreteDiffusionModel  # noqa: E402
from train import _raw_bal_batch  # noqa: E402

RatioMLP = __import__("model.ratio_mlp", fromlist=["RatioMLP"]).RatioMLP


def main():
    x = torch.tensor([0.0, 2.0, 5.0, 9.0])
    z = torch.tensor([0.0, 3.0, 1.0, 4.0])
    g = torch.tensor([0.2, 1.5, 10.0, 0.05])
    a = torch.tensor([-0.4, 0.2, 1.1, -1.5])
    s = a.exp()
    r = g * x / (z + 1.0)
    m = (z + 1.0) / g * s
    left = (z + 1.0) / g * full_bregman(r, s)
    right = full_bregman(x, m)
    if not torch.allclose(left, right, rtol=1e-5, atol=1e-6):
        raise SystemExit(f"balance identity failed\n{left}\n{right}")

    net = RatioMLP(in_dim=1, hidden=16, out_dim=1, layers=2, continuous_t=True)
    for p in net.parameters():
        p.data.zero_()
    model = PoissonDiscreteDiffusionModel(net, lbd=100.0, z_rescale=True, clip_z=True, ratio=True)
    xb = torch.tensor([[2.0], [0.0], [5.0], [1.0]])
    t = torch.tensor([0.04, 0.25, 0.81, 0.16])
    z = torch.tensor([[1.0], [0.0], [4.0], [2.0]])
    eps = 1e-4
    u = (t - eps) / (1.0 - eps)
    real_rand, real_pois = torch.rand, torch.poisson
    torch.rand = lambda *_a, **_k: u.clone()
    torch.poisson = lambda _rate: z.clone()
    args = SimpleNamespace(t_eps=eps, lbd=100.0, z_rescale=True)
    try:
        loss, pred, heads, info = _raw_bal_batch(model, xb, args)
    finally:
        torch.rand, torch.poisson = real_rand, real_pois
    g = t * 100.0
    # zero network => a = 0, S = 1
    r = g * xb.reshape(-1) / (z.reshape(-1) + 1.0)
    expect = (((z.reshape(-1) + 1.0) / g) * full_bregman(r, torch.ones_like(g))).mean()
    if not torch.allclose(loss, expect, rtol=1e-5, atol=1e-6):
        raise SystemExit(f"batch loss {float(loss)} != {float(expect)}")
    if heads is not None or info["cons"] is not None:
        raise SystemExit("raw_bal is a single term")
    loss.backward()
    bias = model.net.out_fc[-1].bias.grad
    if bias is None or not torch.isfinite(bias).all() or float(bias.abs().sum()) == 0.0:
        raise SystemExit(f"no gradient {bias}")
    print("ok", flush=True)


if __name__ == "__main__":
    main()
