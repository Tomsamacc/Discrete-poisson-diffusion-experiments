"""Collect single-head raw-ratio checkpoints. No sampling.

Reads experiments/raw_ratio/k{k}/{name}/best.pt.
Writes ratio log error, recovered-moment log error, |m-m*|, target R scale, grad norms.
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
from train import eval_raw_ratio_quantiles, load_counts  # noqa: E402

RESULTS = HERE / "results" / "15_raw"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")


def ckpt_of(k, name):
    return ROOT / "experiments" / "raw_ratio" / f"k{k}" / name / "best.pt"


def flat(name, k, row):
    out = {"dist": name, "k": k, "bin": row["bin"]}
    for key, prefix in (("ratio", "ratio"), ("mom", "mom"), ("abs", "abs"), ("target", "R")):
        st = row[key]
        out[f"{prefix}_n"] = st["n"]
        out[f"{prefix}_p50"] = st["median"]
        out[f"{prefix}_p90"] = st["p90"]
        out[f"{prefix}_p99"] = st["p99"]
        out[f"{prefix}_max"] = st["max"]
    return out


def write_table(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    lines = [" ".join(keys)]
    for row in rows:
        bits = []
        for key in keys:
            val = row[key]
            if isinstance(val, float):
                bits.append(f"{val:.6g}")
            else:
                bits.append(str(val))
        lines.append(" ".join(bits))
    path.write_text("\n".join(lines) + "\n")


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
        loader = DataLoader(TensorDataset(x), batch_size=1024, shuffle=False)
        xs, logp = support_logp(name, device)
        for k in (1, 2, 3):
            ckpt = ckpt_of(k, name)
            if not ckpt.is_file():
                print(f"skip k={k} {name}", flush=True)
                continue
            model, margs = load_model(ckpt, device)
            margs.moment_k = k
            margs.moment_loss = "raw_ratio"
            margs.moment_param = "raw"
            print(f"eval raw k={k} {name}", flush=True)
            table = eval_raw_ratio_quantiles(model, loader, margs, device, xs, logp)
            for row in table:
                rows.append(flat(name, k, row))
            gpath = ckpt.parent / "grad_summary.json"
            if gpath.is_file():
                g = json.loads(gpath.read_text())
                grads.append({"dist": name, "k": k, **g})
    write_table(RESULTS / "errors.txt", rows)
    (RESULTS / "errors.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_table(RESULTS / "grads.txt", grads)
    (RESULTS / "grads.json").write_text(json.dumps(grads, indent=2) + "\n")
    print(f"wrote {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
