"""Same R_k and same S_k give the same loss under every higher-head link."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.loss import (  # noqa: E402
    HIGHER_PARAMS,
    full_bregman,
    higher_moment_maps,
    raw_ratio_targets,
    rising_factorial,
)
from model.small_diffusion import PoissonDiscreteDiffusionModel  # noqa: E402
from loss.oracle import support_logp  # noqa: E402
from train import _hm_batch, eval_moment_bins, make_loader  # noqa: E402

RatioMLP = __import__("model.ratio_mlp", fromlist=["RatioMLP"]).RatioMLP


def softplus_inv(s):
    return torch.log(torch.expm1(s))


def main():
    z = torch.tensor([0.0, 3.0, 8.0])
    g = torch.tensor([0.4, 2.0, 25.0])
    x = torch.tensor([1.0, 4.0, 0.0])
    f1 = torch.tensor([0.2, -0.5, 1.1])
    s2 = torch.tensor([0.7, 1.4, 0.3])
    s3 = torch.tensor([1.1, 0.5, 2.0])
    c = rising_factorial(z, 3)
    c2 = c[:, 1] / g.double().pow(2)
    c3 = c[:, 2] / g.double().pow(3)
    m2 = c2 * s2.double()
    m3 = c3 * s3.double()
    preds = {
        "raw_log": torch.stack([f1, s2.log(), s3.log()], dim=-1),
        "direct_ratio": torch.stack([f1, softplus_inv(s2), softplus_inv(s3)], dim=-1),
        "log_moment": torch.stack([f1, m2.log(), m3.log()], dim=-1),
        "root_moment": torch.stack(
            [f1, softplus_inv(m2.sqrt()), softplus_inv(m3.pow(1.0 / 3.0))], dim=-1
        ),
        "direct_moment": torch.stack([f1, softplus_inv(m2), softplus_inv(m3)], dim=-1),
    }
    ref_m1 = torch.nn.functional.softplus(f1).double()
    r = raw_ratio_targets(x, z, g, 3)
    ref_loss = full_bregman(r[:, 1], s2) + full_bregman(r[:, 2], s3)
    for name, pred in preds.items():
        m, s = higher_moment_maps(pred, z, g, name, gmin=1e-8)
        if not torch.allclose(m[:, 0], ref_m1, rtol=1e-5, atol=1e-5):
            raise SystemExit(f"{name} m1")
        if not torch.allclose(s[:, 1], s2.double(), rtol=1e-4, atol=1e-4):
            raise SystemExit(f"{name} S2 {s[:, 1]} {s2}")
        if not torch.allclose(s[:, 2], s3.double(), rtol=1e-4, atol=1e-4):
            raise SystemExit(f"{name} S3")
        if not torch.allclose(m[:, 1], c2 * s[:, 1], rtol=1e-4, atol=1e-4):
            raise SystemExit(f"{name} m2")
        if not torch.allclose(m[:, 2], c3 * s[:, 2], rtol=1e-4, atol=1e-4):
            raise SystemExit(f"{name} m3")
        got = full_bregman(r[:, 1], s[:, 1]) + full_bregman(r[:, 2], s[:, 2])
        if not torch.allclose(got, ref_loss, rtol=1e-4, atol=1e-4):
            raise SystemExit(f"{name} loss")

    net = RatioMLP(in_dim=1, hidden=8, out_dim=3, layers=1, continuous_t=True)
    for p in net.parameters():
        p.data.zero_()
    model = PoissonDiscreteDiffusionModel(net, lbd=100.0, z_rescale=True, clip_z=True, ratio=True)
    xb = torch.tensor([[1.0], [0.0], [3.0], [2.0]])
    base = dict(t_eps=1e-4, lbd=100.0, z_rescale=True, weight_steps=100, k_max=3)
    for param in HIGHER_PARAMS:
        for weight in ("equal", "dyn"):
            args = SimpleNamespace(hm_param=param, hybrid_weight=weight, **base)
            torch.manual_seed(0)
            loss, _pred, heads, info = _hm_batch(model, xb, args)
            if not torch.isfinite(loss) or info["cons"] is not None or len(heads) != 3:
                raise SystemExit(f"{param} {weight} batch")
            loss.backward()
            grad = model.net.out_fc[-1].bias.grad
            if grad is None or grad.shape != (3,) or not torch.isfinite(grad).all():
                raise SystemExit(f"{param} {weight} grad")
            if float(grad.abs().min()) == 0.0:
                raise SystemExit(f"{param} {weight} a head got no gradient {grad}")
            model.zero_grad(set_to_none=True)

    xs, logp = support_logp("poissmix3", torch.device("cpu"))
    xb = torch.tensor([[1.0], [50.0], [100.0], [0.0], [50.0], [1.0], [100.0], [50.0]])
    loader = make_loader(xb, xb.shape[0], shuffle=False)
    for param in HIGHER_PARAMS:
        ev = SimpleNamespace(
            hm_param=param,
            moment_loss="hm",
            moment_k=0,
            moment_param="direct",
            ratio=True,
            k_max=3,
            t_eps=1e-4,
            lbd=100.0,
            z_rescale=True,
        )
        means, counts = eval_moment_bins(model, loader, ev, torch.device("cpu"), xs, logp)
        filled = counts > 0
        if int(counts.sum()) != xb.shape[0] or not torch.isfinite(means[:, filled]).all():
            raise SystemExit(f"{param} val log-moment is not finite {means}")
    print("ok", flush=True)


if __name__ == "__main__":
    main()
