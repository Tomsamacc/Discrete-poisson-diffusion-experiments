"""Collect raw multi-output checkpoints for equal, dyn, and dyn_norm. No sampling.

Writes ratio log error, moment log error, absolute moment error,
per-head trunk gradient norms, and pairwise gradient cosine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from loss.oracle import support_logp  # noqa: E402
from sample_tests import load_model  # noqa: E402
from train import _write_table, eval_raw_multi_quantiles, load_counts  # noqa: E402

RESULTS = HERE / "results" / "16_raw_multi"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
WEIGHTS = ("equal", "dyn", "dyn_norm")


def ckpt_of(weight, name):
    return ROOT / "experiments" / "raw_multi" / weight / name / "best.pt"


def qrow(name, k, bin_name, stats, prefix):
    return {
        "dist": name,
        "k": k,
        "bin": bin_name,
        "n": stats["n"],
        f"{prefix}median": stats["median"],
        f"{prefix}p90": stats["p90"],
        f"{prefix}p99": stats["p99"],
        f"{prefix}max": stats["max"],
    }


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(RESULTS))
    args = p.parse_args()
    names = [x.strip() for x in args.names.split(",") if x.strip()]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    ratio_rows, mom_rows, grad_rows, cos_rows = [], [], [], []
    for weight in WEIGHTS:
        for name in names:
            ckpt = ckpt_of(weight, name)
            run = ckpt.parent
            if not ckpt.is_file():
                print(f"skip {weight} {name}", flush=True)
                continue
            model, margs = load_model(ckpt, device)
            margs.moment_loss = "raw_multi"
            margs.k_max = 3
            x = load_counts(ROOT / "data" / name / "val.npy")
            loader = DataLoader(TensorDataset(x), batch_size=1024, shuffle=False)
            xs, logp = support_logp(name, device)
            print(f"eval raw multi {weight} {name}", flush=True)
            table = eval_raw_multi_quantiles(model, loader, margs, device, xs, logp)
            local_ratio, local_mom = [], []
            for row in table:
                r = {"weight": weight, **qrow(name, row["k"], row["bin"], row["ratio"], "")}
                m = {
                    "weight": weight,
                    "dist": name,
                    "k": row["k"],
                    "bin": row["bin"],
                    "n": row["mom"]["n"],
                    "log_median": row["mom"]["median"],
                    "log_p90": row["mom"]["p90"],
                    "log_p99": row["mom"]["p99"],
                    "log_max": row["mom"]["max"],
                    "abs_median": row["abs"]["median"],
                    "abs_p90": row["abs"]["p90"],
                    "abs_p99": row["abs"]["p99"],
                    "abs_max": row["abs"]["max"],
                }
                ratio_rows.append(r)
                mom_rows.append(m)
                local_ratio.append({k: v for k, v in r.items() if k not in ("dist", "weight")})
                local_mom.append({k: v for k, v in m.items() if k not in ("dist", "weight")})
            _write_table(run / "ratio_errors.txt", local_ratio)
            _write_table(run / "moment_errors.txt", local_mom)
            gpath = run / "grad_heads.json"
            if gpath.is_file():
                for row in json.loads(gpath.read_text()):
                    grad_rows.append({"weight": weight, "dist": name, **row})
            cpath = run / "grad_cosine.json"
            if cpath.is_file():
                for row in json.loads(cpath.read_text()):
                    cos_rows.append({"weight": weight, "dist": name, **row})
    _write_table(out / "ratio_errors.txt", ratio_rows)
    _write_table(out / "moment_errors.txt", mom_rows)
    _write_table(out / "grads.txt", grad_rows)
    _write_table(out / "grad_cosine.txt", cos_rows)
    (out / "ratio_errors.json").write_text(json.dumps(ratio_rows, indent=2) + "\n")
    (out / "moment_errors.json").write_text(json.dumps(mom_rows, indent=2) + "\n")
    (out / "grads.json").write_text(json.dumps(grad_rows, indent=2) + "\n")
    (out / "grad_cosine.json").write_text(json.dumps(cos_rows, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
