"""Poisson-only sampling for the single-head parameterization check.

Both checkpoints already use L = D(X, m1), config_scale, k=1.
  direct: network output is m1 (softplus)
  raw:     network output is a, m1 = (z+1)/gamma * exp(a)

Quadratic T=100, lambda=100, n=50000. No NB or TwoPois.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from sample import generate, make_gammas  # noqa: E402
from sample_tests import load_model  # noqa: E402
from test import plot_experiment  # noqa: E402

NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
TAGS = ("direct", "raw")
RESULTS = HERE / "results" / "23_param_poisson"


def ckpt_of(tag, name):
    return ROOT / "experiments" / "direct_prl" / tag / "k1" / name / "best.pt"


def prepare(args, tag):
    if tag == "raw":
        args.ratio = True
        args.k_max = 1
        args.moment_param = "raw"
        args.moment_loss = "moment"
        args.moment_k = 1
        args.ratio_loss = "normalized"
        args.consistency = "none"
        args.prior_moments = None
    else:
        args.ratio = False
        args.moment_param = "direct"
        args.moment_loss = "moment"
        args.moment_k = 1
        args.consistency = "none"
    return args


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--n-sample", type=int, default=50000)
    p.add_argument("--sample-batch", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    cli = p.parse_args()
    names = [x.strip() for x in cli.names.split(",") if x.strip()]
    device = torch.device(cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu")
    out = Path(cli.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    missing = [str(ckpt_of(tag, name)) for tag in TAGS for name in names if not ckpt_of(tag, name).is_file()]
    if missing:
        for path in missing:
            print(f"missing {path}", flush=True)
        raise SystemExit("missing single-head checkpoints")

    rows = []
    for tag in TAGS:
        for name in names:
            ckpt = ckpt_of(tag, name)
            dest = out / "sample" / tag / name
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / "samples_poisson.npy"
            if not path.is_file() or cli.force:
                model, args = load_model(ckpt, device)
                prepare(args, tag)
                args.sample_batch = cli.sample_batch
                gammas = make_gammas("quad", 100, 0.0, float(args.lbd), device, power=2.0)
                torch.manual_seed(cli.seed)
                np.random.seed(cli.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(cli.seed)
                print(f"sample {tag} {name} poisson ckpt={ckpt}", flush=True)
                x = generate(model, cli.n_sample, "poisson", args, device, gammas=gammas, counters={})
                np.save(path, x)
                print(f"  E[z/gamma]={float(np.mean(x)):.4g}", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                print(f"skip {tag} {name}", flush=True)
            side = dest / "metrics.json"
            if side.is_file() and not cli.force:
                table = json.loads(side.read_text())
            else:
                table = plot_experiment(dest, name=name, kernels=("poisson",))
                side.write_text(json.dumps(table, indent=2) + "\n")
            rec = next(r for r in table if r["kernel"] == "poisson")
            rows.append(
                {
                    "tag": tag,
                    "dist": name,
                    "kernel": "poisson",
                    "wd": rec["wd"],
                    "tv": rec["tv"],
                    "overflow": rec.get("overflow"),
                    "ckpt": str(ckpt),
                }
            )
    (out / "sampling_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["tag dist wd tv overflow"]
    for r in rows:
        lines.append(f"{r['tag']} {r['dist']} {r['wd']:.6g} {r['tv']:.6g} {r['overflow']:.6g}")
    (out / "sampling_metrics.txt").write_text("\n".join(lines) + "\n")
    (out / "config.txt").write_text(
        "\n".join(
            [
                "single head, L = D(X, m1) on both sides",
                "direct: m1 = softplus(network)",
                "raw: m1 = (z+1)/gamma * exp(a)",
                "same data, gamma=t*100, hidden 128, layers 3, Adam 1e-3, 200 epochs, batch 256, seed 0",
                "sampler: Poisson only, quadratic T=100, lambda=100, n=50000, seed 0",
                "checkpoints: experiments/direct_prl/{direct,raw}/k1/",
                "",
            ]
        )
    )
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
