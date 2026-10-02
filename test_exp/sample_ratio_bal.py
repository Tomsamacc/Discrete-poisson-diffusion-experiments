"""Poisson sampling for the explicit balanced ratio loss.

direct rows are copied from 23_param_poisson. Those models use D(X, m).
ratio_bal uses (z+1)/gamma * D(R, exp(a)).
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
RESULTS = HERE / "results" / "24_ratio_bal"
DIRECT = HERE / "results" / "23_param_poisson" / "sampling_metrics.txt"


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-sample", type=int, default=50000)
    p.add_argument("--sample-batch", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    cli = p.parse_args()
    device = torch.device(cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu")
    out = RESULTS
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in NAMES:
        ckpt = ROOT / "experiments" / "ratio_bal" / name / "best.pt"
        if not ckpt.is_file():
            raise SystemExit(f"missing {ckpt}")
        dest = out / "sample" / name
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "samples_poisson.npy"
        if not path.is_file() or cli.force:
            model, args = load_model(ckpt, device)
            args.ratio = True
            args.k_max = 1
            args.moment_param = "raw"
            args.moment_loss = "raw_bal"
            args.moment_k = 1
            args.ratio_loss = "normalized"
            args.consistency = "none"
            args.prior_moments = None
            args.sample_batch = cli.sample_batch
            gammas = make_gammas("quad", 100, 0.0, float(args.lbd), device, power=2.0)
            torch.manual_seed(cli.seed)
            np.random.seed(cli.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(cli.seed)
            print(f"sample {name} poisson ckpt={ckpt}", flush=True)
            x = generate(model, cli.n_sample, "poisson", args, device, gammas=gammas, counters={})
            np.save(path, x)
            print(f"  E[z/gamma]={float(np.mean(x)):.4g}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        side = dest / "metrics.json"
        if side.is_file() and not cli.force and path.is_file():
            table = json.loads(side.read_text())
        else:
            table = plot_experiment(dest, name=name, kernels=("poisson",))
            side.write_text(json.dumps(table, indent=2) + "\n")
        rec = next(r for r in table if r["kernel"] == "poisson")
        rows.append({"tag": "ratio_bal", "dist": name, "wd": rec["wd"], "tv": rec["tv"], "overflow": rec.get("overflow")})

    if DIRECT.is_file():
        for line in DIRECT.read_text().splitlines()[1:]:
            tag, dist, wd, tv, overflow = line.split()
            if tag != "direct" or dist not in NAMES:
                continue
            rows.append({"tag": "direct", "dist": dist, "wd": float(wd), "tv": float(tv), "overflow": float(overflow)})
    (out / "sampling_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["tag dist wd tv overflow"]
    for r in rows:
        lines.append(f"{r['tag']} {r['dist']} {float(r['wd']):.6g} {float(r['tv']):.6g} {float(r['overflow']):.6g}")
    (out / "sampling_metrics.txt").write_text("\n".join(lines) + "\n")
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
