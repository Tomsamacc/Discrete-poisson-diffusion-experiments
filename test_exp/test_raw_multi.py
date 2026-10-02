"""Equal-weight raw multi-output loss and trunk gradients."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.loss import (  # noqa: E402
    RawRatioBregmanLoss,
    quad_reverse_h,
    raw_multi_head_losses,
    raw_multi_weights,
)
from model.small_diffusion import PoissonDiscreteDiffusionModel  # noqa: E402
from train import _batch_loss, _record_trunk, trunk_parameters  # noqa: E402


def test_sum_matches_single_heads():
    torch.manual_seed(0)
    log_s = torch.randn(16, 3, requires_grad=True)
    target = torch.rand(16, 3)
    target[0, :] = 0
    heads = raw_multi_head_losses(log_s, target)
    singles = [
        RawRatioBregmanLoss()(log_s[:, k], target[:, k])
        for k in range(3)
    ]
    for k in range(3):
        assert torch.allclose(heads[k], singles[k], rtol=1e-6, atol=1e-8)
    total = heads[0] + heads[1] + heads[2]
    assert float(total.detach()) >= -1e-8
    assert torch.isfinite(total)


def test_quad_h_and_weights():
    t = torch.tensor([1e-4, 0.99 ** 2, 1.0])
    h = quad_reverse_h(t, 100.0, 100)
    assert abs(float(h[0]) - 0.03) < 1e-8
    assert abs(float(h[1]) - 1.99) < 1e-6
    assert abs(float(h[2]) - 1.99) < 1e-6
    dyn = raw_multi_weights(t, "dyn", 100.0, 100, 3)
    assert torch.allclose(dyn[:, 0], h)
    assert torch.allclose(dyn[:, 1], h.pow(2) / 2)
    assert torch.allclose(dyn[:, 2], h.pow(3) / 6)
    norm = raw_multi_weights(t, "dyn_norm", 100.0, 100, 3)
    assert torch.allclose(norm.sum(-1), torch.ones(3, dtype=torch.float64))
    eq = raw_multi_weights(t, "equal", 100.0, 100, 3)
    assert torch.allclose(eq, torch.ones_like(eq))
    print("h", [float(v) for v in h])


def test_trunk_grad_adds():
    torch.manual_seed(1)
    RatioMLP = importlib.import_module("model.ratio_mlp").RatioMLP
    net = RatioMLP(in_dim=1, hidden=16, out_dim=3, layers=1, continuous_t=True)
    model = PoissonDiscreteDiffusionModel(net, lbd=100.0, z_rescale=True, clip_z=True, ratio=True)
    xb = torch.rand(24, 1) * 6
    args = SimpleNamespace(
        t_eps=1e-4,
        lbd=100.0,
        z_rescale=True,
        moment_loss="raw_multi",
        k_max=3,
        ratio=True,
    )
    loss, pred, heads = _batch_loss(model, xb, None, args)
    assert pred.shape[-1] == 3
    assert len(heads) == 3
    params = trunk_parameters(model)
    last = {id(p) for p in model.net.out_fc[-1].parameters()}
    assert last.isdisjoint({id(p) for p in params})
    per = []
    for head in heads:
        gs = torch.autograd.grad(head, params, retain_graph=True, allow_unused=True)
        per.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, gs)]))
    gs = torch.autograd.grad(loss, params, retain_graph=True)
    total = torch.cat([g.reshape(-1) for g in gs])
    added = per[0] + per[1] + per[2]
    err = (total - added).abs().max().item()
    print(f"trunk grad |sum heads - total| {err:.3e}")
    assert err < 1e-5
    trace = {"norms": [], "cos": []}
    _record_trunk(model, heads, trace)
    assert len(trace["norms"][0]) == 3
    assert len(trace["cos"][0]) == 3
    assert all(torch.isfinite(torch.tensor(trace["norms"][0])))
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"raw multi loss={float(loss.detach()):.4f} trunk p={len(params)}")


if __name__ == "__main__":
    test_sum_matches_single_heads()
    test_quad_h_and_weights()
    test_trunk_grad_adds()
    print("ok")
