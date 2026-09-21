"""Sample PMF-ratio ckpts: uniform h=1 and quadratic γ_t=100(t/100)^2, T=100."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sample import build_model, generate, make_gammas
from test import plot_experiment
from train import ROOT


NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
KERNELS_USE = ("poisson", "nb", "twopois")


def split_csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_root", default=str(ROOT / "experiments" / "pmfratio"))
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--kernels", default=",".join(KERNELS_USE))
    p.add_argument("--n_sample", type=int, default=50000)
    p.add_argument("--sample_batch", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_pair(ckpt, device):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    from types import SimpleNamespace

    a = blob.get("args") or {}
    args = SimpleNamespace(
        ckpt=str(ckpt),
        hidden=int(a.get("hidden", 128)),
        layers=int(a.get("layers", 3)),
        continuous_t=bool(a.get("continuous_t", True)),
        lbd=float(a.get("lbd", 100.0)),
        z_rescale=bool(a.get("z_rescale", a.get("scale", False))),
        clip_z=bool(a.get("clip_z", True)),
        clip_range=a.get("clip_range"),
        normalize=a.get("normalize"),
        t_eps=float(a.get("t_eps", 1e-4)),
        ratio=bool(a.get("ratio", True)),
        k_max=int(a.get("k_max", 3)),
        sample_batch=4096,
        snr_min=0.0,
        snr_max=float(a.get("lbd", 100.0)),
        sample_steps=100,
        n_sample=50000,
        kernels="poisson,nb,twopois",
        seed=0,
        device=str(device),
    )
    model = build_model(args, device)
    return model, args


def main():
    cli = parse_args()
    names = split_csv(cli.names)
    kernels = tuple(k for k in split_csv(cli.kernels) if k != "repoisson")
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    device = torch.device(cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu")
    ckpt_root = Path(cli.ckpt_root) if Path(cli.ckpt_root).is_absolute() else ROOT / cli.ckpt_root
    rows = []
    for name in names:
        ckpt = ckpt_root / name / "best.pt"
        if not ckpt.is_file():
            print(f"skip {name}: no {ckpt}", flush=True)
            continue
        model, args = load_pair(ckpt, device)
        args.sample_batch = cli.sample_batch
        g1 = float(args.lbd)
        for sched, power, tag in (("uniform", 1.0, "unif_T100"), ("quad", 2.0, "quad_T100")):
            dest = ckpt_root / name / tag
            dest.mkdir(parents=True, exist_ok=True)
            gammas = make_gammas("quad", 100, 0.0, g1, device, power=power)
            print(
                f"  {dest} ratio={args.ratio} k_max={args.k_max} "
                f"steps=100 γ0=0 γT={g1:g} power={power}",
                flush=True,
            )
            for k in kernels:
                path = dest / f"samples_{k}.npy"
                if path.is_file():
                    print(f"    skip {k}", flush=True)
                    continue
                x = generate(model, cli.n_sample, k, args, device, gammas=gammas)
                np.save(path, x)
                print(f"    saved {path.name} E[z/γ]={x.mean():.4g}", flush=True)
            (dest / "meta.json").write_text(
                json.dumps(
                    {
                        "name": name,
                        "tag": tag,
                        "schedule": sched,
                        "power": power,
                        "ratio": True,
                        "k_max": args.k_max,
                        "ckpt": str(ckpt),
                    },
                    indent=2,
                )
                + "\n"
            )
            table = plot_experiment(dest, name=name, kernels=kernels)
            for r in table:
                r["tag"] = tag
                rows.append(r)
    if rows:
        keys = ("tag", "dist", "kernel", "wd", "tv")
        lines = [" ".join(f"{k:>12}" if k in ("wd", "tv") else f"{k:14}" for k in keys)]
        for r in rows:
            lines.append(
                f"{r['tag']:14} {r['dist']:14} {r['kernel']:12} {r['wd']:12.4f} {r['tv']:12.4f}"
            )
        text = "\n".join(lines) + "\n"
        (ckpt_root / "metrics.txt").write_text(text)
        print(text, end="", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
