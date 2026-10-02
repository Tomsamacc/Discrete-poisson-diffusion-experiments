"""CPU checks for the hybrid m1 loss. No checkpoints."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.loss import full_bregman, quad_reverse_h, raw_ratio_targets, uses_raw_log_s  # noqa: E402
from loss.oracle import support_logp  # noqa: E402
from model.small_diffusion import PoissonDiscreteDiffusionModel  # noqa: E402
from sample import generate, make_gammas  # noqa: E402
from train import _hybrid_batch  # noqa: E402

RatioMLP = __import__("model.ratio_mlp", fromlist=["RatioMLP"]).RatioMLP


def zero_model():
    net = RatioMLP(in_dim=1, hidden=16, out_dim=3, layers=2, continuous_t=True)
    for p in net.parameters():
        p.data.zero_()
    return PoissonDiscreteDiffusionModel(
        net, lbd=100.0, z_rescale=True, clip_z=True, ratio=True
    )


def args_for(weight):
    return SimpleNamespace(
        t_eps=1e-4,
        lbd=100.0,
        z_rescale=True,
        hybrid_weight=weight,
        weight_steps=100,
        k_max=3,
        moment_loss="hybrid",
        moment_param="raw",
        ratio_loss="normalized",
        ratio=True,
        consistency="none",
        sample_batch=4,
        normalize=None,
        prior_moments=[2.0, 8.0, 40.0],
    )


def reference(xb, t, z, weight):
    gamma = t * 100.0
    target = raw_ratio_targets(xb, z, gamma, 3)
    log_m1 = (z.reshape(-1) + 1.0).clamp_min(1e-12).log() - gamma.reshape(-1).clamp_min(1e-12).log()
    l1 = full_bregman(xb.reshape(-1), log_m1.exp())
    l2 = full_bregman(target[:, 1], torch.ones(xb.shape[0], dtype=torch.float64))
    l3 = full_bregman(target[:, 2], torch.ones(xb.shape[0], dtype=torch.float64))
    if weight == "dyn":
        h = quad_reverse_h(t, 100.0, 100)
        l2 = l2 * (h / 2.0)
        l3 = l3 * (h.square() / 6.0)
    return (l1 + l2 + l3).mean(), l1.mean(), l2.mean(), l3.mean()


def run_fixed(weight, xb, t, z):
    model = zero_model()
    real_rand, real_pois = torch.rand, torch.poisson
    eps = 1e-4
    u = (t - eps) / (1.0 - eps)

    def fake_rand(*_a, **_k):
        return u.clone()

    def fake_pois(_rate):
        return z.clone()

    torch.rand = fake_rand
    torch.poisson = fake_pois
    try:
        loss, _pred, heads, info = _hybrid_batch(model, xb, args_for(weight))
    finally:
        torch.rand, torch.poisson = real_rand, real_pois
    return model, loss, heads, info


def main():
    x = torch.tensor([1.0, 4.0, 0.0, 9.0])
    z = torch.tensor([2.0, 5.0, 0.0, 3.0])
    g = torch.tensor([0.5, 2.0, 10.0, 0.2])
    m = torch.tensor([1.2, 3.0, 0.4, 8.0])
    left = full_bregman(g * x / (z + 1.0), g * m / (z + 1.0))
    right = (g / (z + 1.0)) * full_bregman(x, m)
    if not torch.allclose(left, right, rtol=1e-5, atol=1e-6):
        raise SystemExit("weight identity failed")

    xb = torch.tensor([[2.0], [0.0], [5.0], [1.0], [8.0], [3.0]])
    t = torch.tensor([0.04, 0.25, 0.81, 0.09, 0.36, 0.64])
    z = torch.tensor([[1.0], [2.0], [4.0], [0.0], [3.0], [5.0]])
    exp_eq, e1, e2, e3 = reference(xb, t, z, "equal")
    _model, loss_eq, heads_eq, info = run_fixed("equal", xb, t, z)
    if not torch.allclose(loss_eq, exp_eq, rtol=1e-5, atol=1e-6):
        raise SystemExit(f"equal loss {float(loss_eq)} != {float(exp_eq)}")
    for head, exp in zip(heads_eq, (e1, e2, e3)):
        if not torch.allclose(head, exp, rtol=1e-5, atol=1e-6):
            raise SystemExit(f"equal head {float(head)} != {float(exp)}")
    if info["cons"] is not None:
        raise SystemExit("hybrid cons field should stay empty")
    loss_eq.backward()
    last = _model.net.out_fc[-1].bias.grad
    if last is None or not torch.isfinite(last).all() or float(last.abs().sum()) == 0.0:
        raise SystemExit(f"m1 head got no gradient {last}")

    exp_dyn, d1, d2, d3 = reference(xb, t, z, "dyn")
    model_d, loss_dyn, heads_dyn, _ = run_fixed("dyn", xb, t, z)
    if not torch.allclose(loss_dyn, exp_dyn, rtol=1e-5, atol=1e-6):
        raise SystemExit(f"dyn loss {float(loss_dyn)} != {float(exp_dyn)}")
    for head, exp in zip(heads_dyn, (d1, d2, d3)):
        if not torch.allclose(head, exp, rtol=1e-5, atol=1e-6):
            raise SystemExit(f"dyn head {float(head)} != {float(exp)}")
    if torch.allclose(loss_dyn, loss_eq):
        raise SystemExit("dyn weights did not change the loss")
    loss_dyn.backward()

    ns = args_for("dyn")
    if not uses_raw_log_s(ns):
        raise SystemExit("hybrid must use raw log S")

    xs, logp = support_logp("gamma_ltj", torch.device("cpu"))
    gammas = make_gammas("quad", 2, 0.0, 100.0, torch.device("cpu"), power=2.0)
    on_policy = {"xs": xs, "logp": logp, "n": 4, "seen": 0, "chunks": []}
    torch.manual_seed(0)
    out = generate(
        zero_model(),
        4,
        "poisson",
        args_for("equal"),
        torch.device("cpu"),
        gammas=gammas,
        on_policy=on_policy,
    )
    if out.shape != (4,) or not on_policy["chunks"] or int(on_policy["seen"]) != 4:
        raise SystemExit(f"on-policy generate failed shape={out.shape} seen={on_policy['seen']}")
    if not torch.isfinite(on_policy["chunks"][0]["kl"]).all():
        raise SystemExit("on-policy KL not finite")
    print("ok", flush=True)


if __name__ == "__main__":
    main()
