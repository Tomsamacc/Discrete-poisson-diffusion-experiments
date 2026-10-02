"""CPU check for softplus-ratio and offset means."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.loss import full_bregman, linked_mean  # noqa: E402
from model.small_diffusion import PoissonDiscreteDiffusionModel  # noqa: E402
from train import _link_batch  # noqa: E402

RatioMLP = __import__("model.ratio_mlp", fromlist=["RatioMLP"]).RatioMLP


def main():
    a = torch.tensor([-2.0, 0.0, 1.5])
    z = torch.tensor([0.0, 4.0, 2.0])
    g = torch.tensor([0.5, 2.0, 10.0])
    s = torch.nn.functional.softplus(a)
    m = (z + 1.0) / g * s
    got = linked_mean(a, z, g, "soft_ratio", gmin=1e-6)
    if not torch.allclose(got, m.double()):
        raise SystemExit("softplus link failed")
    b = torch.tensor([-1.0, 0.2, 2.0])
    if not torch.allclose(linked_mean(b, z, g, "offset"), b.double().exp()):
        raise SystemExit("offset link failed")
    x = torch.tensor([1.0, 0.0, 4.0])
    loss = full_bregman(x, m)
    if not torch.isfinite(loss).all():
        raise SystemExit("loss not finite")

    net = RatioMLP(in_dim=1, hidden=8, out_dim=1, layers=1, continuous_t=True)
    for p in net.parameters():
        p.data.zero_()
    model = PoissonDiscreteDiffusionModel(net, lbd=100.0, z_rescale=True, clip_z=True, ratio=True)
    xb = torch.tensor([[1.0], [3.0], [0.0], [2.0]])
    args = SimpleNamespace(t_eps=1e-4, lbd=100.0, z_rescale=True)
    for kind in ("soft_ratio", "offset"):
        torch.manual_seed(0)
        loss, _pred, heads, info = _link_batch(model, xb, args, kind)
        if not torch.isfinite(loss) or heads is not None or info["cons"] is not None:
            raise SystemExit(f"{kind} batch failed {loss}")
        loss.backward()
        grad = model.net.out_fc[-1].bias.grad
        if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) == 0.0:
            raise SystemExit(f"{kind} got no gradient")
        model.zero_grad(set_to_none=True)
    print("ok", flush=True)


if __name__ == "__main__":
    main()
