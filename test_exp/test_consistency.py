"""CPU checks for ratio consistency, projection, and the hard identity."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.consistency import (  # noqa: E402
    assemble_log_s,
    consistency_penalty,
    constraint_A,
    expand_hard,
    penalty_from_logs,
    project_u,
)
from loss.loss import RawRatioBregmanLoss, moments_from_ratio_pred  # noqa: E402
from sample import (  # noqa: E402
    log_s_eval,
    posterior_moments,
    predict_log_B,
    two_pois_gap,
    two_pois_moment_flags,
)
from train import _batch_loss, apply_consistency  # noqa: E402


def test_constraint_and_projection():
    A = constraint_A()
    expect = torch.tensor(
        [
            [-1.0, -1.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0, 0.0, 1.0],
        ]
    )
    assert torch.equal(A, expect)
    u = torch.randn(6, 5)
    up, resid = project_u(u)
    assert resid.abs().max() < 1e-5
    nulls = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ]
    )
    diff = (u - up).double()
    for vec in nulls:
        dots = diff @ vec.double()
        assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5)
    ident = torch.zeros(2, 5)
    ident[:, 0] = 0.2
    ident[:, 1] = -0.1
    ident[:, 2] = 0.4
    ident[:, 3] = ident[:, 0] + ident[:, 1]
    ident[:, 4] = ident[:, 3] + ident[:, 2]
    up2, resid2 = project_u(ident)
    assert torch.allclose(up2, ident, atol=1e-6)
    assert resid2.abs().max() < 1e-6


def test_expand_and_penalty():
    a0 = torch.tensor([0.2, -0.1])
    a1 = torch.tensor([0.3, 0.4])
    a2 = torch.tensor([-0.2, 0.1])
    log_s = expand_hard(a0, a1, a2)
    assert torch.allclose(log_s[:, 1], a0 + a1)
    assert torch.allclose(log_s[:, 2], a0 + a1 + a2)
    pred = torch.tensor([[0.2, 0.5, -0.1], [0.0, 1.0, 0.3]], requires_grad=True)
    s1 = torch.tensor([0.1, -0.2], requires_grad=True)
    s2 = torch.tensor([0.4, 0.2], requires_grad=True)
    soft = penalty_from_logs(pred, s1, s2, "soft")
    anchored = penalty_from_logs(pred, s1, s2, "anchored")
    assert torch.allclose(soft, anchored)
    g_soft = torch.autograd.grad(soft, [pred, s1, s2], retain_graph=True)
    g_anchor = torch.autograd.grad(anchored, [pred, s1, s2], allow_unused=True)
    assert g_anchor[1] is None
    assert g_anchor[2] is None
    assert torch.allclose(g_anchor[0][:, 0], torch.zeros(2))
    assert g_soft[1].abs().sum() > 0
    assert g_soft[0][:, 0].abs().sum() > 0
    fresh = torch.tensor([[0.2, 0.5, -0.1], [0.0, 1.0, 0.3]])
    s1b = torch.tensor([0.1, -0.2])
    s2b = torch.tensor([0.4, 0.2])
    plain = penalty_from_logs(fresh, s1b, s2b, "soft")
    only_w1 = torch.zeros(2, 3)
    only_w1[:, 0] = 1
    assert penalty_from_logs(fresh, s1b, s2b, "soft", weights=only_w1).abs() < 1e-8
    w23 = torch.tensor([[0.2, 0.5, 0.3], [0.2, 0.5, 0.3]])
    assert not torch.allclose(penalty_from_logs(fresh, s1b, s2b, "soft", weights=w23), plain)


def test_module_backward():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(1, 3)

        def forward(self, z, t, alpha=None):
            return self.lin(z.reshape(-1, 1))

    z = torch.rand(8, 1)
    t = torch.rand(8)
    for mode in ("soft", "anchored"):
        model = Tiny()
        pred = model(z, t, alpha=t)
        loss = pred.square().mean() + consistency_penalty(model, z, t, t, pred, mode)
        loss.backward()
        assert model.lin.weight.grad is not None
        assert torch.isfinite(model.lin.weight.grad).all()


def test_batch_loss_modes():
    class Heads(nn.Module):
        def __init__(self, out_dim):
            super().__init__()
            self.lin = nn.Linear(1, out_dim)

        def forward(self, z, t, alpha=None):
            return self.lin(z.reshape(-1, 1))

    xb = torch.rand(16, 1) + 0.1
    for mode in ("soft", "anchored"):
        args = SimpleNamespace(
            t_eps=1e-4,
            lbd=100.0,
            z_rescale=False,
            k_max=3,
            moment_loss="raw_multi",
            raw_weight="equal",
            weight_steps=100,
            consistency=mode,
            lambda_cons=1.0,
        )
        model = Heads(3)
        nn.init.zeros_(model.lin.weight)
        nn.init.zeros_(model.lin.bias)
        loss, pred, heads, info = _batch_loss(model, xb, None, args)
        assert info["cons"] is not None
        assert pred.shape[-1] == 3
        assert heads is not None and len(heads) == 3
        params = [p for p in model.parameters() if p.requires_grad]
        for head in heads:
            torch.autograd.grad(head, params, retain_graph=True, allow_unused=True)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(model.lin.weight.grad).all()
    args_w = SimpleNamespace(
        t_eps=1e-4,
        lbd=100.0,
        z_rescale=False,
        k_max=3,
        moment_loss="raw_multi",
        raw_weight="dyn_norm",
        weight_steps=100,
        consistency="soft",
        lambda_cons=1.0,
        cons_weight="dyn_norm",
    )
    model = Heads(3)
    nn.init.zeros_(model.lin.weight)
    nn.init.zeros_(model.lin.bias)
    loss, pred, heads, info = _batch_loss(model, xb, None, args_w)
    assert info["cons"] is not None
    loss.backward()
    assert torch.isfinite(loss)
    args = SimpleNamespace(
        t_eps=1e-4,
        lbd=100.0,
        z_rescale=False,
        k_max=1,
        moment_k=1,
        moment_param="raw",
        moment_loss="raw_ratio",
        consistency="hard",
        lambda_cons=1.0,
    )
    model = Heads(1)
    nn.init.zeros_(model.lin.weight)
    nn.init.zeros_(model.lin.bias)
    loss, pred, heads, info = _batch_loss(model, xb, RawRatioBregmanLoss(), args)
    assert info["cons"] is None
    assert pred.shape[-1] == 1
    assert heads is None
    loss.backward()
    assert torch.isfinite(loss)


def test_apply_consistency_keeps_requested_weight():
    soft = SimpleNamespace(
        consistency="soft",
        lambda_cons=1.0,
        raw_weight="dyn_norm",
        moment_loss="moment",
        moment_k=2,
        k_max=1,
        ratio=False,
        moment_param="direct",
    )
    apply_consistency(soft)
    assert soft.raw_weight == "dyn_norm"
    assert soft.moment_loss == "raw_multi"
    assert soft.k_max == 3
    assert soft.moment_k == 0
    hard = SimpleNamespace(
        consistency="hard",
        lambda_cons=1.0,
        moment_k=0,
        moment_loss="moment",
        k_max=3,
        ratio=False,
        moment_param="direct",
    )
    apply_consistency(hard)
    assert hard.moment_k == 1
    assert hard.k_max == 1
    assert hard.moment_loss == "raw_ratio"
    plain = SimpleNamespace(
        consistency="none",
        lambda_cons=1.0,
        raw_weight="dyn",
        moment_loss="raw_multi",
        moment_k=0,
        k_max=3,
        ratio=True,
        moment_param="raw",
    )
    apply_consistency(plain)
    assert plain.raw_weight == "dyn"


def test_two_pois_flags_match_sampler():
    m1 = torch.tensor([[1.0], [0.0], [-0.2], [3.0], [2.0]])
    m2 = torch.tensor([[0.5], [1.0], [0.1], [20.0], [4.0]])
    m3 = torch.tensor([[0.2], [0.0], [0.0], [100.0], [8.0]])
    flags = two_pois_moment_flags(m1, m2, m3)
    stats = {}
    torch.manual_seed(1)
    k1 = two_pois_gap(m1, m2, m3, torch.tensor(0.2), stats=stats)
    torch.manual_seed(1)
    k2 = two_pois_gap(m1, m2, m3, torch.tensor(0.2), stats=None)
    assert torch.equal(k1, k2)
    assert stats["invalid_v"] == int(flags["thin"].sum())
    assert stats["invalid_atom"] == int(flags["bad_atom"].sum())
    assert stats["extreme_w"] == int(flags["extreme"].sum())
    assert stats["fallback"] == int(flags["fallback"].sum())
    assert int(flags["v_le_0"].sum()) >= 1


def _moment_args(mode):
    return SimpleNamespace(
        lbd=100.0,
        t_eps=1e-4,
        z_rescale=False,
        ratio=True,
        consistency=mode,
        moment_param="raw",
        moment_loss="raw_multi" if mode != "hard" else "raw_ratio",
        ratio_loss="raw",
        k_max=1 if mode == "hard" else 3,
        prior_moments=[1.0, 2.0, 6.0],
    )


class _ShiftNet(nn.Module):
    def forward(self, z, t, alpha=None):
        return (0.01 * z.reshape(-1, 1))


class _Three(nn.Module):
    def forward(self, z, t, alpha=None):
        return z.reshape(-1, 1) * torch.tensor([0.01, -0.02, 0.03])


def test_posterior_paths():
    z = torch.tensor([[3.0], [4.0]])
    gamma = torch.full((2,), 10.0)
    hard_args = _moment_args("hard")
    moms = posterior_moments(_ShiftNet(), z, gamma, hard_args, order=3)
    log_s = torch.tensor([[0.03, 0.07, 0.12], [0.04, 0.09, 0.15]])
    expect = moments_from_ratio_pred(log_s, z, gamma, hard_args)
    for got, ref in zip(moms, expect):
        assert torch.allclose(got, ref, atol=1e-5)
    moms0 = posterior_moments(_ShiftNet(), z, torch.zeros(2), hard_args, order=3)
    assert torch.allclose(moms0[0], torch.ones(2, 1))
    assert torch.allclose(moms0[1], torch.full((2, 1), 2.0))
    assert torch.allclose(moms0[2], torch.full((2, 1), 6.0))
    _, d2h, d3h, _, _ = log_s_eval(_ShiftNet(), z, gamma, hard_args)
    assert float(d2h.abs().max()) == 0.0
    assert float(d3h.abs().max()) == 0.0
    none_args = _moment_args("none")
    direct = moments_from_ratio_pred(predict_log_B(_Three(), z, gamma, none_args), z, gamma, none_args)
    got = posterior_moments(_Three(), z, gamma, none_args, order=3)
    for a, b in zip(got, direct):
        assert torch.allclose(a, b, atol=1e-6)
    proj_args = _moment_args("project")
    log_p, d2, d3, d2_in, d3_in = log_s_eval(_Three(), z, gamma, proj_args)
    assert log_p.shape == (2, 3)
    assert d2.abs().max() < 1e-4
    assert d3.abs().max() < 1e-4
    assert d2_in.abs().max() > 1e-3
    raw, _, _, raw_d2, _ = assemble_log_s(*[
        _Three()(z, torch.full((2,), 0.1), alpha=None),
        _Three()(z + 1, torch.full((2,), 0.1), alpha=None),
        _Three()(z + 2, torch.full((2,), 0.1), alpha=None),
    ], "none")
    assert not torch.allclose(log_p, raw[:, :3])
    assert raw_d2.abs().max() > 1e-3


def test_real_equal_projection():
    ckpt = ROOT / "experiments" / "raw_multi" / "equal" / "gamma_ltj" / "best.pt"
    if not ckpt.is_file():
        return
    from sample_tests import load_model

    model, args = load_model(ckpt, torch.device("cpu"))
    args.moment_param = "raw"
    args.ratio_loss = "raw"
    args.consistency = "none"
    z = torch.tensor([[0.0], [3.0], [10.0], [40.0]])
    gamma = torch.tensor([1.0, 1.0, 10.0, 50.0])
    direct = predict_log_B(model, z, gamma, args)
    packed, _, _, _, _ = log_s_eval(model, z, gamma, args)
    assert torch.allclose(packed, direct, atol=1e-6)
    args.consistency = "project"
    log_s, d2, d3, d2_in, d3_in = log_s_eval(model, z, gamma, args)
    assert log_s.shape == (4, 3)
    assert d2.abs().max() < 1e-4
    assert d3.abs().max() < 1e-4
    assert torch.isfinite(d2_in).all()
    assert torch.isfinite(d3_in).all()


if __name__ == "__main__":
    test_constraint_and_projection()
    test_expand_and_penalty()
    test_module_backward()
    test_batch_loss_modes()
    test_apply_consistency_keeps_requested_weight()
    test_two_pois_flags_match_sampler()
    test_posterior_paths()
    test_real_equal_projection()
    print("ok")
