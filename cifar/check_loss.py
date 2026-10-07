import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cifar.losses import cifar_maps, consistency_penalty, data_terms, load_split, q_sample
from cifar.model import net_input
from cifar.sample import final_image
from cifar.variants import PARAMS, rows_for
from loss.loss import full_bregman, higher_moment_maps


def fail(msg):
    raise SystemExit(msg)


def check_rows():
    counts = {
        "ab": 4,
        "c_ub": 10,
        "c_bd": 10,
        "d_ub": 10,
        "d_bd": 10,
        "cons": 20,
    }
    for name, n in counts.items():
        rows = rows_for(name)
        if len(rows) != n:
            fail(f"{name} {len(rows)} != {n}")


def check_input():
    z = torch.zeros(1, 3, 2, 2)
    z[0, 0, 0, 0] = 5.0
    z[0, 0, 0, 1] = -3.0
    zin, logg = net_input(z, torch.ones(1), 0.01)
    if abs(float(zin[0, 0, 0, 0]) - 5.0) > 1e-6:
        fail(f"input above 1 was changed {float(zin[0, 0, 0, 0])}")
    if float(zin[0, 0, 0, 1]) != 0.0:
        fail("negative input was not removed")
    if not torch.isfinite(logg).all():
        fail("log gamma")


def check_loss_keeps_large_mean():
    pred = torch.full((2, 1, 3, 8, 8), 5.0)
    x = torch.full((2, 3, 8, 8), 0.4)
    z = torch.ones_like(x)
    gamma = torch.full((2,), 10.0)
    t = torch.full((2,), 0.1)
    spec = {"family": "mean", "param": "direct", "weight": "equal", "heads": 1, "cons": "none", "lam": 0.0, "link": "unbounded"}
    terms = data_terms(pred, x, z, gamma, t, spec, 100.0, 100, 0.01)
    m = F.softplus(pred[:, 0]).reshape(-1).double()
    if float(m.min()) <= 1.0:
        fail("softplus target did not stay above 1")
    ref = full_bregman(x.reshape(-1), m).mean()
    if not torch.allclose(terms["loss"], ref):
        fail(f"loss changed a mean above 1 {float(terms['loss'])} {float(ref)}")


def check_data():
    path = "/home/tomsama/scratch/itdpdm/datasets/cifar10/split_80_20.npz"
    xtr, xva = load_split(path)
    if float(xtr.min()) < 0 or float(xtr.max()) > 1:
        fail(f"split not in [0,1] {float(xtr.min())} {float(xtr.max())}")
    if xtr.shape[1:] != (3, 32, 32) or xva.shape[0] < 1000:
        fail(f"shape {tuple(xtr.shape)} {tuple(xva.shape)}")
    print(f"data n={xtr.shape[0]}/{xva.shape[0]} mean={float(xtr.mean()):.4g} max={float(xtr.max()):.4g}")


class ZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, z, gamma):
        return z.new_zeros(z.shape[0], 3, z.shape[1], z.shape[2], z.shape[3]) + self.anchor * 0 + gamma.reshape(-1).mean() * 0


class ScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, z, gamma):
        m = z.mean(dim=(1, 2, 3), keepdim=True)
        heads = [(self.scale * float(k + 1)) * m.expand_as(z) + gamma.reshape(-1).view(-1, 1, 1, 1) * 0 for k in range(3)]
        return torch.stack(heads, dim=1)


def check_step_weight():
    pred = torch.randn(2, 3, 3, 4, 4)
    x = torch.rand(2, 3, 4, 4) * 0.5 + 0.05
    z = torch.ones_like(x)
    gamma = torch.tensor([10.0, 40.0])
    t = torch.tensor([0.05, 0.8])
    base = {"family": "hybrid_ratio", "param": "raw_log", "heads": 3, "cons": "none", "lam": 0.0, "link": "unbounded"}
    eq = dict(base)
    eq["weight"] = "equal"
    st = dict(base)
    st["weight"] = "step"
    a = data_terms(pred, x, z, gamma, t, eq, 100.0, 100, 0.01)["loss"]
    b = data_terms(pred, x, z, gamma, t, st, 100.0, 100, 0.01)["loss"]
    if torch.allclose(a, b):
        fail("step weights did not change the loss")


def check_links():
    pred = torch.full((4, 3), 8.0)
    z = torch.full((4,), 3.0)
    g = torch.full((4,), 5.0)
    bounded, _ = cifar_maps(pred, z, g, "root_moment", "bounded", 0.01)
    unbounded, _ = cifar_maps(pred, z, g, "root_moment", "unbounded", 0.01)
    if float(bounded.max()) > 1.0:
        fail(f"bounded moment left (0,1) {float(bounded.max())}")
    if float(unbounded[:, 1].max()) <= 1.0:
        fail("unbounded root moment did not exceed 1")
    rnd = torch.randn(6, 3)
    zz = torch.rand(6) * 4 + 0.2
    gg = torch.rand(6) * 10 + 0.2
    for param in PARAMS:
        left = cifar_maps(rnd, zz, gg, param, "unbounded", 0.01)
        right = higher_moment_maps(rnd, zz, gg, param, gmin=0.01)
        if not torch.allclose(left[0], right[0], rtol=1e-5, atol=1e-5):
            fail(f"unbounded map drifted for {param}")
        if not torch.allclose(left[1], right[1], rtol=1e-5, atol=1e-5):
            fail(f"unbounded ratio drifted for {param}")


class _ZeroHeads(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, z, gamma):
        return z.new_zeros(z.shape[0], self.k, z.shape[1], z.shape[2], z.shape[3]) + self.w * 0


def check_final_image():
    z = torch.full((1, 3, 2, 2), 14.0)
    g = torch.tensor([10.0])
    ratio = {
        "family": "ratio",
        "param": "raw_log",
        "weight": "equal",
        "heads": 3,
        "cons": "none",
        "lam": 0.0,
        "link": "unbounded",
    }
    x = final_image(_ZeroHeads(3), z, g, ratio, 0.01)
    if abs(float(x.detach().reshape(-1)[0]) - 1.5) > 1e-4:
        fail(f"ratio final image {float(x.reshape(-1)[0])}")
    if float(x.max()) <= 1.0:
        fail("ratio posterior mean was clamped to 1")
    bounded = {
        "family": "mean",
        "param": "direct",
        "weight": "equal",
        "heads": 1,
        "cons": "none",
        "lam": 0.0,
        "link": "bounded",
    }
    y = final_image(_ZeroHeads(1), z, g, bounded, 0.01)
    if abs(float(y.detach().reshape(-1)[0]) - 0.5) > 1e-4:
        fail(f"sigmoid final image {float(y.reshape(-1)[0])}")
    if torch.allclose(y, z / g.view(1, 1, 1, 1)):
        fail("final image fell back to z/gamma")


def check_consistency():
    z = torch.rand(2, 3, 4, 4)
    gamma = torch.full((2,), 4.0)
    t = torch.full((2,), 0.2)
    index = torch.tensor([1, 3])
    spec = {
        "family": "ratio",
        "param": "raw_log",
        "weight": "equal",
        "heads": 3,
        "cons": "sym",
        "lam": 1.0,
        "link": "unbounded",
    }
    pen = consistency_penalty(ZeroNet(), z, gamma, t, index, spec, 100.0, 100, 0.01)
    if float(pen.detach().abs()) > 1e-8:
        fail(f"constant log S penalty {float(pen)}")
    net = ScaleNet()
    spec = dict(spec)
    spec["cons"] = "sym"
    pen = consistency_penalty(net, z, gamma, t, index, spec, 100.0, 100, 0.01)
    pen.backward()
    if net.scale.grad is None or float(net.scale.grad.abs()) == 0.0:
        fail("consistency did not reach the trunk")


def main():
    check_rows()
    check_input()
    check_loss_keeps_large_mean()
    check_step_weight()
    check_links()
    check_final_image()
    check_consistency()
    check_data()
    print("ok")


if __name__ == "__main__":
    main()
