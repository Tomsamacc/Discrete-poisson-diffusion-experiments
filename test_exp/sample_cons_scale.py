"""Quadratic T=100 sampling for the lambda sweep and weighted-consistency runs.

New checkpoints:
  l0.1, l1o3   unweighted consistency, lambda 0.1 and 1/3
  w1, w0.3     dyn_norm-weighted consistency, lambda 1 and 0.3

Copied, not resampled:
  dyn_norm     lambda 0, experiment 17
  l1_unw       unweighted lambda 1, experiment 19 tag soft
"""
from __future__ import annotations

import json
import subprocess
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
from train import load_counts  # noqa: E402

NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
KERNELS = ("poisson", "nb", "twopois")
RESULTS = HERE / "results" / "22_cons_scale"
RUNS = (
    ("l0.1", ROOT / "experiments" / "raw_cons_scale" / "l0.1"),
    ("l1o3", ROOT / "experiments" / "raw_cons_scale" / "l1o3"),
    ("w1", ROOT / "experiments" / "raw_cons_w" / "l1"),
    ("w0.3", ROOT / "experiments" / "raw_cons_w" / "l0.3"),
)
DYN = HERE / "results" / "17_raw_multi_sampling" / "sampling_metrics.json"
L1 = HERE / "results" / "19_ratio_cons_dyn_norm" / "sampling_metrics.json"


def ckpt_of(tag, name):
    for t, root in RUNS:
        if t == tag:
            return root / name / "best.pt"
    raise KeyError(tag)


def prepare(args, name):
    args.ratio = True
    args.moment_param = "raw"
    args.ratio_loss = "raw"
    args.moment_loss = "raw_multi"
    args.consistency = "none"
    args.k_max = 3
    prior = getattr(args, "prior_moments", None)
    if prior is None or len(list(prior)) < 3:
        x = load_counts(ROOT / "data" / name / "train.npy").reshape(-1).double()
        args.prior_moments = [float(x.pow(k).mean()) for k in range(1, 4)]
    return args


def frac(num, den):
    if not den:
        return float("nan")
    return float(num) / float(den)


def run_sample(names, kernels, cli, device, out):
    for tag, _root in RUNS:
        for name in names:
            ckpt = ckpt_of(tag, name)
            dest = out / "sample" / tag / name
            dest.mkdir(parents=True, exist_ok=True)
            model, args = load_model(ckpt, device)
            prepare(args, name)
            args.sample_batch = cli.sample_batch
            gammas = make_gammas("quad", 100, 0.0, float(args.lbd), device, power=2.0)
            print(f"sample {tag} {name} quad T=100 ckpt={ckpt}", flush=True)
            for kernel in kernels:
                path = dest / f"samples_{kernel}.npy"
                cpath = dest / f"counters_{kernel}.json"
                if path.is_file() and cpath.is_file() and not cli.force:
                    print(f"  skip {kernel}", flush=True)
                    continue
                torch.manual_seed(cli.seed)
                np.random.seed(cli.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(cli.seed)
                counters = {}
                x = generate(
                    model, cli.n_sample, kernel, args, device, gammas=gammas, counters=counters,
                )
                np.save(path, x)
                cpath.write_text(json.dumps(counters) + "\n")
                line = f"  saved {kernel} E[z/gamma]={float(np.mean(x)):.4g}"
                if counters.get("traj"):
                    line += f" runaway={frac(counters.get('runaway_traj', 0), counters['traj']):.4g}"
                if kernel == "twopois" and counters.get("n"):
                    line += f" fallback={frac(counters.get('fallback', 0), counters['n']):.4g}"
                print(line, flush=True)
            (dest / "meta.json").write_text(
                json.dumps({"tag": tag, "name": name, "schedule": "quad", "steps": 100, "ckpt": str(ckpt)}, indent=2)
                + "\n"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


def copy_rows(path, names, kernels, src_tag, dst_tag, source):
    rows = []
    if not Path(path).is_file():
        print(f"missing {path}", flush=True)
        return rows
    for row in json.loads(Path(path).read_text()):
        if row.get("tag") != src_tag:
            continue
        if row.get("dist") not in names or row.get("kernel") not in kernels:
            continue
        item = dict(row)
        item["tag"] = dst_tag
        item["source"] = source
        rows.append(item)
    return rows


def collect(names, kernels, out, force_plot):
    rows = []
    for tag, _root in RUNS:
        for name in names:
            dest = out / "sample" / tag / name
            gens = [k for k in kernels if (dest / f"samples_{k}.npy").is_file()]
            if not gens:
                continue
            side = dest / "metrics.json"
            table = json.loads(side.read_text()) if side.is_file() and not force_plot else None
            if table is None:
                table = plot_experiment(dest, name=name, kernels=tuple(gens))
                side.write_text(json.dumps(table, indent=2) + "\n")
            by_k = {r["kernel"]: r for r in table}
            for kernel in gens:
                rec = by_k.get(kernel)
                if rec is None:
                    continue
                counters = {}
                cpath = dest / f"counters_{kernel}.json"
                if cpath.is_file():
                    counters = json.loads(cpath.read_text())
                n_tp = int(counters.get("n") or 0)
                n_tr = int(counters.get("traj") or 0)
                rows.append(
                    {
                        "tag": tag,
                        "dist": name,
                        "kernel": kernel,
                        "wd": rec["wd"],
                        "tv": rec["tv"],
                        "overflow": rec.get("overflow", float("nan")),
                        "runaway": frac(counters.get("runaway_traj", 0), n_tr),
                        "fallback": frac(counters.get("fallback", 0), n_tp) if kernel == "twopois" else float("nan"),
                        "invalid_v": frac(counters.get("invalid_v", 0), n_tp) if kernel == "twopois" else float("nan"),
                        "invalid_atom": frac(counters.get("invalid_atom", 0), n_tp) if kernel == "twopois" else float("nan"),
                        "extreme_w": frac(counters.get("extreme_w", 0), n_tp) if kernel == "twopois" else float("nan"),
                        "source": "22_cons_scale",
                    }
                )
    rows.extend(copy_rows(DYN, names, kernels, "dyn_norm", "dyn_norm", "17_raw_multi_sampling"))
    rows.extend(copy_rows(L1, names, kernels, "soft", "l1_unw", "19_ratio_cons_dyn_norm"))
    return rows


def write_table(path, rows):
    keys = ["tag", "dist", "kernel", "wd", "tv", "overflow", "runaway", "fallback", "invalid_v", "invalid_atom", "extreme_w", "source"]
    lines = [" ".join(keys)]
    for row in rows:
        bits = []
        for key in keys:
            val = row.get(key)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                bits.append("nan")
            elif isinstance(val, float):
                bits.append(f"{val:.6g}")
            else:
                bits.append(str(val))
        lines.append(" ".join(bits))
    path.write_text("\n".join(lines) + "\n")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--kernels", default=",".join(KERNELS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--n-sample", type=int, default=50000)
    p.add_argument("--sample-batch", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", default="all", choices=("all", "sample", "tables"))
    p.add_argument("--force", action="store_true")
    cli = p.parse_args()
    names = [x.strip() for x in cli.names.split(",") if x.strip()]
    kernels = [x.strip() for x in cli.kernels.split(",") if x.strip()]
    device = torch.device(cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu")
    out = Path(cli.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    if cli.stage in ("all", "sample"):
        missing = [str(ckpt_of(tag, name)) for tag, _r in RUNS for name in names if not ckpt_of(tag, name).is_file()]
        if missing:
            for path in missing:
                print(f"missing {path}", flush=True)
            raise SystemExit("finish the consistency runs before sampling")
        run_sample(names, kernels, cli, device, out)
    if cli.stage in ("all", "tables"):
        rows = collect(names, kernels, out, cli.force)
        (out / "sampling_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        write_table(out / "sampling_metrics.txt", rows)
        try:
            rev = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            rev = ""
        (out / "git_commit.txt").write_text(rev + "\n")
        (out / "config.txt").write_text(
            "\n".join(
                [
                    "consistency lambda sweep and weighted consistency",
                    "schedule: quadratic T=100 lambda=100 p=2",
                    "l0.1 l1o3: L_data + lambda (delta2^2+delta3^2), lambda 0.1 and 1/3",
                    "w1 w0.3: L_data + lambda (w2 delta2^2 + w3 delta3^2), lambda 1 and 0.3",
                    "dyn_norm copied from experiment 17 (lambda 0)",
                    "l1_unw copied from experiment 19 soft (unweighted lambda 1)",
                    f"n_sample: {cli.n_sample}",
                    f"seed: {cli.seed}",
                    "",
                ]
            )
        )
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
