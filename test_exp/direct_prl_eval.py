"""Aggregate single-head direct / ratio / raw moment-PRL checkpoints. No sampling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from loss.oracle import BIN_LABELS  # noqa: E402
from sample_tests import load_model  # noqa: E402
from train import eval_moment_quantiles, load_counts  # noqa: E402
from loss.oracle import support_logp  # noqa: E402

RESULTS = HERE / "results" / "14_direct"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
JOBS = [("direct", k) for k in (1, 2, 3)] + [("ratio", k) for k in (1, 2, 3)] + [("raw", 1)]


def ckpt_of(param, k, name):
    return ROOT / "experiments" / "direct_prl" / param / f"k{k}" / name / "best.pt"


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    names = [x.strip() for x in args.names.split(",") if x.strip()]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    grads = []
    for name in names:
        x = load_counts(ROOT / "data" / name / "val.npy")
        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(TensorDataset(x), batch_size=1024, shuffle=False)
        xs, logp = support_logp(name, device)
        for param, k in JOBS:
            ckpt = ckpt_of(param, k, name)
            if not ckpt.is_file():
                print(f"skip {param} k={k} {name}", flush=True)
                continue
            model, margs = load_model(ckpt, device)
            margs.moment_k = k
            margs.moment_param = param
            print(f"eval {param} k={k} {name}", flush=True)
            table = eval_moment_quantiles(model, loader, margs, device, xs, logp)
            for row in table:
                rows.append(
                    {
                        "dist": name,
                        "param": param,
                        "k": k,
                        "bin": row["bin"],
                        "n": row["log"]["n"],
                        "log_median": row["log"]["median"],
                        "log_p90": row["log"]["p90"],
                        "log_p99": row["log"]["p99"],
                        "log_max": row["log"]["max"],
                        "abs_median": row["abs"]["median"],
                        "abs_p90": row["abs"]["p90"],
                        "abs_p99": row["abs"]["p99"],
                        "abs_max": row["abs"]["max"],
                    }
                )
            gpath = ckpt.parent / "grad_summary.json"
            if gpath.is_file():
                g = json.loads(gpath.read_text())
                grads.append({"dist": name, "param": param, "k": k, **g})
    (RESULTS / "moments.json").write_text(json.dumps(rows, indent=2) + "\n")
    (RESULTS / "grads.json").write_text(json.dumps(grads, indent=2) + "\n")
    keys = (
        "dist", "param", "k", "bin", "n",
        "log_median", "log_p90", "log_p99", "log_max",
        "abs_median", "abs_p99", "abs_max",
    )
    lines = [" ".join(f"{k:14}" for k in keys)]
    for r in rows:
        bits = []
        for k in keys:
            v = r.get(k)
            bits.append(f"{v:14.4f}" if isinstance(v, float) else f"{str(v):14}")
        lines.append(" ".join(bits))
    text = "\n".join(lines) + "\n"
    (RESULTS / "moments.txt").write_text(text)
    print(text, end="")
    if grads:
        glines = ["dist           param          k              p50            p90            p99            max"]
        for g in grads:
            glines.append(
                f"{g['dist']:14} {g['param']:14} {g['k']:<14} {g['median']:14.4g} {g['p90']:14.4g} {g['p99']:14.4g} {g['max']:14.4g}"
            )
        (RESULTS / "grads.txt").write_text("\n".join(glines) + "\n")
        print("\n".join(glines))
    print("bins", BIN_LABELS)
    print("done", RESULTS)


if __name__ == "__main__":
    main()
