"""Poisson, NB, and TwoPois sampling for the higher-head parameterizations.

m1 is softplus in every run. k=2,3 change only the link into the same D(R, S).
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
WEIGHTS = ("equal", "dyn")
PARAMS = ("raw_log", "direct_ratio", "log_moment", "root_moment", "direct_moment")
KERNELS = ("poisson", "nb", "twopois")
RESULTS = HERE / "results" / "26_hm"


def frac(num, den):
    if not den:
        return float("nan")
    return float(num) / float(den)


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
    for weight in WEIGHTS:
        for param in PARAMS:
            for name in NAMES:
                ckpt = ROOT / "experiments" / "hm" / weight / param / name / "best.pt"
                if not ckpt.is_file():
                    print(f"missing {ckpt}", flush=True)
                    continue
                dest = out / "sample" / weight / param / name
                dest.mkdir(parents=True, exist_ok=True)
                model = None
                args = None
                for kernel in KERNELS:
                    path = dest / f"samples_{kernel}.npy"
                    cpath = dest / f"counters_{kernel}.json"
                    if not path.is_file() or not cpath.is_file() or cli.force:
                        if model is None:
                            model, args = load_model(ckpt, device)
                            args.ratio = True
                            args.k_max = 3
                            args.moment_loss = "hm"
                            args.hm_param = param
                            args.hybrid_weight = weight
                            args.moment_param = "direct"
                            args.consistency = "none"
                            args.sample_batch = cli.sample_batch
                        gammas = make_gammas("quad", 100, 0.0, float(args.lbd), device, power=2.0)
                        torch.manual_seed(cli.seed)
                        np.random.seed(cli.seed)
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(cli.seed)
                        counters = {}
                        print(f"sample {weight} {param} {name} {kernel}", flush=True)
                        x = generate(
                            model, cli.n_sample, kernel, args, device, gammas=gammas, counters=counters
                        )
                        np.save(path, x)
                        cpath.write_text(json.dumps(counters) + "\n")
                        line = f"  E[z/gamma]={float(np.mean(x)):.4g}"
                        if counters.get("traj"):
                            line += f" runaway={frac(counters.get('runaway_traj', 0), counters['traj']):.4g}"
                        if kernel == "twopois" and counters.get("n"):
                            line += f" fallback={frac(counters.get('fallback', 0), counters['n']):.4g}"
                        print(line, flush=True)
                if model is not None:
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                side = dest / "metrics.json"
                have = all((dest / f"samples_{k}.npy").is_file() for k in KERNELS)
                if not have:
                    continue
                if side.is_file() and not cli.force:
                    table = json.loads(side.read_text())
                else:
                    table = plot_experiment(dest, name=name, kernels=KERNELS)
                    side.write_text(json.dumps(table, indent=2) + "\n")
                by_k = {r["kernel"]: r for r in table}
                for kernel in KERNELS:
                    rec = by_k[kernel]
                    counters = json.loads((dest / f"counters_{kernel}.json").read_text())
                    rows.append(
                        {
                            "weight": weight,
                            "param": param,
                            "dist": name,
                            "kernel": kernel,
                            "wd": rec["wd"],
                            "tv": rec["tv"],
                            "overflow": rec.get("overflow"),
                            "runaway": frac(counters.get("runaway_traj", 0), counters.get("traj", 0)),
                            "fallback": frac(counters.get("fallback", 0), counters.get("n", 0))
                            if kernel == "twopois"
                            else 0.0,
                        }
                    )
    (out / "sampling_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["weight param dist kernel wd tv overflow runaway fallback"]
    for r in rows:
        lines.append(
            f"{r['weight']} {r['param']} {r['dist']} {r['kernel']} "
            f"{float(r['wd']):.6g} {float(r['tv']):.6g} {float(r['overflow']):.6g} "
            f"{float(r['runaway']):.6g} {float(r['fallback']):.6g}"
        )
    (out / "sampling_metrics.txt").write_text("\n".join(lines) + "\n")
    print(f"done -> {out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
