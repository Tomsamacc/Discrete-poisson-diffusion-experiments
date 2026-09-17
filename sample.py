import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from model.small_diffusion import PoissonDiscreteDiffusionModel
from train import ROOT, load_yaml, pair_or_none, resolve, str2bool

MLP = importlib.import_module("model.1d_mlp").MLP
KERNELS = ("poisson", "nb", "repoisson", "twopois")


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", default=str(ROOT / "config.yml"))
    pre.add_argument("--ckpt", default=None, help="best.pt / latest.pt")
    pre_args, _ = pre.parse_known_args()
    cfg = load_yaml(pre_args.config)
    ckpt_args = {}
    if pre_args.ckpt:
        blob = torch.load(pre_args.ckpt, map_location="cpu", weights_only=False)
        ckpt_args = blob.get("args") or {}

    def pick(key, default):
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        return ckpt_args.get(key, default)

    p = argparse.ArgumentParser(
        parents=[pre],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", default=None, help="defaults to ckpt folder")
    p.add_argument("--device", default=pick("device", "cuda"))
    p.add_argument("--lbd", type=float, default=float(pick("lbd", 100.0)))
    p.add_argument("--snr_min", type=float, default=float(cfg.get("snr_min", 0.0)))
    p.add_argument("--snr_max", type=float, default=float(cfg.get("snr_max", pick("lbd", 100.0))))
    p.add_argument("--sample_steps", type=int, default=int(cfg.get("sample_steps", 100)))
    p.add_argument("--n_sample", type=int, default=int(cfg.get("n_sample", 50000)))
    p.add_argument("--sample_batch", type=int, default=int(cfg.get("sample_batch", 4096)))
    p.add_argument("--t_eps", type=float, default=float(pick("t_eps", 1e-4)))
    p.add_argument(
        "--kernels",
        default=cfg.get("kernels", "poisson,nb,repoisson"),
        help="comma list: poisson,nb,repoisson,twopois",
    )
    p.add_argument("--hidden", type=int, default=int(pick("hidden", 128)))
    p.add_argument("--layers", type=int, default=int(pick("layers", 3)))
    p.add_argument("--continuous_t", type=str2bool, default=pick("continuous_t", True))
    p.add_argument(
        "--scale",
        "--z_rescale",
        dest="z_rescale",
        type=str2bool,
        default=bool(ckpt_args.get("z_rescale", ckpt_args.get("scale", cfg.get("scale", False)))),
        help="MLP sees z/(λ α_t)=z/γ; match the ckpt",
    )
    p.add_argument("--clip_z", type=str2bool, default=pick("clip_z", True))
    p.add_argument("--clip_range", nargs=2, type=float, default=pair_or_none(pick("clip_range", None)))
    p.add_argument("--normalize", nargs=2, type=float, default=pair_or_none(pick("normalize", None)))
    p.add_argument("--seed", type=int, default=int(pick("seed", 0)))
    args = p.parse_args()
    if args.ckpt is None:
        p.error("--ckpt is required")
    if args.out_dir is None:
        args.out_dir = str(Path(args.ckpt).resolve().parent)
    return args


def build_model(args, device):
    net = MLP(
        in_dim=1,
        hidden=args.hidden,
        out_dim=1,
        layers=args.layers,
        continuous_t=args.continuous_t,
    )
    model = PoissonDiscreteDiffusionModel(
        net,
        lbd=args.lbd,
        z_rescale=args.z_rescale,
        clip_z=args.clip_z,
        clip_range=args.clip_range,
        normalize=args.normalize,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def time_of(gamma, lbd, t_eps, n, device):
    t = (float(gamma) / float(lbd)) if not torch.is_tensor(gamma) else (gamma / lbd)
    t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
    t = t.clamp_min(t_eps)
    if t.numel() == 1:
        t = t.expand(n)
    return t


def predict_x(model, z, gamma, args):
    t = time_of(gamma, args.lbd, args.t_eps, z.shape[0], z.device)
    alpha = t if args.z_rescale else None
    return model(z, t, alpha=alpha)


def posterior_moments(model, z, gamma, args, order=3):
    acc = None
    moms = []
    for i in range(order):
        mi = predict_x(model, z + float(i), gamma, args).clamp_min(0.0)
        acc = mi if acc is None else acc * mi
        moms.append(acc)
    return moms


def two_pois_gap(m1, m2, m3, h):
    """3-moment two-point mix: K ~ w Pois(h x1) + (1-w) Pois(h x2)."""
    v = (m2 - m1 * m1).clamp_min(0.0)
    c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1.pow(3)
    m_safe = m1.clamp_min(1e-8)
    thin = (m1 <= 0) | (v <= 1e-8 * m_safe)
    sigma = v.sqrt()
    g = c3 / sigma.pow(3).clamp_min(1e-12)
    w = 0.5 * (1.0 + g / (g.square() + 4.0).sqrt())
    w = w.clamp(1e-4, 1.0 - 1e-4)
    x1 = m1 - sigma * ((1.0 - w) / w).sqrt()
    x2 = m1 + sigma * (w / (1.0 - w)).sqrt()
    k_p = torch.poisson((h * m1).clamp_min(0.0))
    pick = torch.rand_like(m1)
    k_mix = torch.where(
        pick < w,
        torch.poisson((h * x1).clamp_min(0.0)),
        torch.poisson((h * x2).clamp_min(0.0)),
    )
    k_mix = torch.where(x1 < 0, k_p, k_mix)
    return torch.where(thin, k_p, k_mix)


def nb_gap(m, v, h):
    mean_k = (h * m).clamp_min(0.0)
    pois = torch.poisson(mean_k)
    m_safe = m.clamp_min(1e-8)
    v_safe = v.clamp_min(0.0)
    thin = (m <= 0) | (v_safe <= 1e-8 * m_safe)
    r = (m_safe * m_safe / v_safe.clamp_min(1e-12)).clamp(1e-4, 1e6)
    scale = (v_safe / m_safe).clamp_min(1e-12)
    xg = torch.distributions.Gamma(r, 1.0 / scale).sample()
    nb = torch.poisson((h * xg).clamp_min(0.0))
    return torch.where(thin, pois, nb)


@torch.no_grad()
def reverse_step(model, z, gamma, h, kernel, args):
    m = predict_x(model, z, gamma, args)
    if kernel == "poisson":
        return z + torch.poisson((h * m).clamp_min(0.0))
    if kernel == "repoisson":
        m = m.clamp_min(0.0).round()
        return torch.poisson(((gamma + h) * m).clamp_min(0.0))
    if kernel == "nb":
        m1 = predict_x(model, z + 1.0, gamma, args)
        v = (m * m1 - m * m).clamp_min(0.0)
        return z + nb_gap(m, v, h)
    if kernel in ("twopois", "2pois", "two_poisson"):
        m1, m2, m3 = posterior_moments(model, z, gamma, args, order=3)
        return z + two_pois_gap(m1, m2, m3, h)
    raise ValueError(kernel)


@torch.no_grad()
def generate(model, n, kernel, args, device):
    g0, g1 = float(args.snr_min), float(args.snr_max)
    steps = int(args.sample_steps)
    h = (g1 - g0) / steps
    out = []
    left = n
    while left > 0:
        b = min(int(args.sample_batch), left)
        z = torch.zeros(b, 1, device=device)
        gamma = g0
        for _ in range(steps):
            z = reverse_step(model, z, gamma, h, kernel, args)
            gamma = gamma + h
        x = (z / max(float(gamma), 1e-12)).clamp_min(0.0)
        if args.normalize is not None:
            x = model.decode_x(x)
        out.append(x.cpu())
        left -= b
    return torch.cat(out, 0).numpy().reshape(-1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args, device)
    kernels = [k.strip() for k in str(args.kernels).split(",") if k.strip()]
    for k in kernels:
        if k not in KERNELS and k not in ("2pois", "two_poisson"):
            raise ValueError(f"unknown kernel {k}")
        key = "twopois" if k in ("twopois", "2pois", "two_poisson") else k
        print(
            f"sample {key} n={args.n_sample} snr={args.snr_min:g}->{args.snr_max:g} "
            f"steps={args.sample_steps} z_rescale={args.z_rescale} ckpt={args.ckpt}",
            flush=True,
        )
        x = generate(model, args.n_sample, key, args, device)
        path = out_dir / f"samples_{key}.npy"
        np.save(path, x)
        print(f"  saved {path} E[z/γ]={x.mean():.4g}", flush=True)
    with open(out_dir / "sample_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    from test import plot_experiment

    plot_keys = tuple(
        k for k in ("poisson", "nb", "repoisson", "twopois") if (out_dir / f"samples_{k}.npy").is_file()
    )
    plot_experiment(out_dir, kernels=plot_keys)


if __name__ == "__main__":
    main()
