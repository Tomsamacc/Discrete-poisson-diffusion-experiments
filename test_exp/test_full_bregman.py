"""H. Full Bregman and reduced loss have the same parameter gradient."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.loss import (  # noqa: E402
    MomentPRLLoss,
    RawRatioBregmanLoss,
    full_bregman,
    raw_ratio_targets,
    reduced_bregman,
)
from loss.oracle import poisson_mixture_pmf  # noqa: E402
from model.small_diffusion import PoissonDiscreteDiffusionModel  # noqa: E402
from train import _batch_loss  # noqa: E402


def _param_grads(layer, xb, y, fn):
    layer.zero_grad(set_to_none=True)
    mu = layer(xb).squeeze(-1).double().exp()
    loss = fn(y, mu).mean()
    loss.backward()
    return (
        loss.detach().clone(),
        layer.weight.grad.detach().clone(),
        layer.bias.grad.detach().clone(),
    )


def test_same_parameter_gradient():
    torch.manual_seed(0)
    layer = nn.Linear(3, 1)
    xb = torch.randn(64, 3)
    y = torch.rand(64) * 5
    y[:8] = 0
    full_loss, gw, gb = _param_grads(layer, xb, y, full_bregman)
    red_loss, rw, rb = _param_grads(layer, xb, y, reduced_bregman)
    werr = (gw - rw).abs().max().item()
    berr = (gb - rb).abs().max().item()
    print(f"grad max |full-reduced| weight={werr:.3e} bias={berr:.3e}")
    assert werr < 1e-6 and berr < 1e-6
    y64 = y.double().clamp_min(0)
    gap = (torch.special.xlogy(y64, y64) - y64).mean()
    assert torch.allclose(full_loss - red_loss, gap, rtol=1e-6, atol=1e-8)
    assert float(full_loss) >= -1e-8


def test_zero_target_and_match():
    y = torch.tensor([0.0, 0.5, 2.0, 10.0])
    mu = torch.tensor([0.3, 0.5, 2.0, 4.0])
    full = full_bregman(y, mu)
    assert torch.all(full >= -1e-10)
    assert abs(float(full[0]) - float(mu.double()[0])) < 1e-8
    assert float(full[1]) < 1e-8
    assert float(full[2]) < 1e-8
    assert float(full[3]) > 0


def test_moment_and_raw_losses_match_reduced():
    torch.manual_seed(1)
    layer = nn.Linear(4, 1)
    xb = torch.randn(48, 4)
    x = torch.rand(48) * 6
    x[:6] = 0

    layer.zero_grad(set_to_none=True)
    m = torch.nn.functional.softplus(layer(xb).squeeze(-1))
    MomentPRLLoss()(m, x, 2).backward()
    g_full = layer.weight.grad.detach().clone()

    layer.zero_grad(set_to_none=True)
    m = torch.nn.functional.softplus(layer(xb).squeeze(-1))
    reduced_bregman(x.double().pow(2), m).mean().backward()
    g_red = layer.weight.grad.detach().clone()
    assert (g_full - g_red).abs().max().item() < 1e-6

    target = torch.rand(48) * 3
    target[:5] = 0
    layer.zero_grad(set_to_none=True)
    log_s = layer(xb).squeeze(-1)
    RawRatioBregmanLoss()(log_s, target).backward()
    g_full = layer.weight.grad.detach().clone()
    layer.zero_grad(set_to_none=True)
    log_s = layer(xb).squeeze(-1)
    reduced_bregman(target, log_s.double().exp()).mean().backward()
    g_red = layer.weight.grad.detach().clone()
    assert (g_full - g_red).abs().max().item() < 1e-6


def test_raw_target_identity():
    x = torch.tensor([0.0, 1.5, 4.0])
    z = torch.tensor([0.0, 2.0, 5.0])
    gamma = torch.tensor([0.2, 1.0, 3.0])
    r = raw_ratio_targets(x, z, gamma, 3)
    r1 = gamma * x / (z + 1)
    r2 = r1 * gamma * x / (z + 2)
    r3 = r2 * gamma * x / (z + 3)
    assert torch.allclose(r[:, 0], r1.double())
    assert torch.allclose(r[:, 1], r2.double())
    assert torch.allclose(r[:, 2], r3.double())


def test_point_mass_jump():
    from scipy.stats import poisson

    xs = torch.tensor([0.0, 2.0, 5.0])
    w = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.25, 0.75]])
    h = 1.5
    q, tail = poisson_mixture_pmf(w, xs, h, kmax=3)
    rate = h * 2.0
    for k in range(4):
        assert abs(float(q[0, k]) - float(poisson.pmf(k, rate))) < 1e-6
    assert abs(float(tail[0]) - float(poisson.sf(3, rate))) < 1e-6
    for k in range(4):
        mix = 0.25 * poisson.pmf(k, h * 2.0) + 0.75 * poisson.pmf(k, h * 5.0)
        assert abs(float(q[1, k]) - float(mix)) < 1e-6
    q0, tail0 = poisson_mixture_pmf(w, xs, 0.0, kmax=3)
    assert torch.allclose(q0[:, 0], torch.ones(2, dtype=torch.float64))
    assert torch.allclose(tail0, torch.zeros(2, dtype=torch.float64))


def test_raw_batch_backward():
    torch.manual_seed(2)
    RatioMLP = importlib.import_module("model.ratio_mlp").RatioMLP
    net = RatioMLP(in_dim=1, hidden=16, out_dim=1, layers=1, continuous_t=True)
    model = PoissonDiscreteDiffusionModel(net, lbd=100.0, z_rescale=True, clip_z=True, ratio=True)
    xb = torch.rand(32, 1) * 8
    for k in (1, 2, 3):
        args = SimpleNamespace(
            t_eps=1e-4,
            lbd=100.0,
            z_rescale=True,
            moment_k=k,
            moment_loss="raw_ratio",
            moment_param="raw",
            ratio=True,
            k_max=1,
        )
        model.zero_grad(set_to_none=True)
        loss, pred, _heads = _batch_loss(model, xb, RawRatioBregmanLoss(), args)
        assert loss.ndim == 0
        assert float(loss.detach()) >= -1e-6
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
        print(f"raw batch k={k} loss={float(loss):.4f} pred={tuple(pred.shape)}")


if __name__ == "__main__":
    test_same_parameter_gradient()
    test_zero_target_and_match()
    test_moment_and_raw_losses_match_reduced()
    test_raw_target_identity()
    test_point_mass_jump()
    test_raw_batch_backward()
    print("ok")
