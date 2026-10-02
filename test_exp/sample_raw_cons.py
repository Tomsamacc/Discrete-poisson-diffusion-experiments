"""Sample the new consistency models and the post-hoc projections.

equal and dyn_norm WD/TV are copied from experiment 17.
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
sys.path.insert(0, str(HERE))

from cons_tables import (  # noqa: E402
    NAMES,
    ckpt_of,
    missing_ckpts,
    tag_specs,
    write_comparison,
)
from sample import generate, make_gammas  # noqa: E402
from sample_tests import load_model  # noqa: E402
from test import plot_experiment  # noqa: E402
from train import load_counts  # noqa: E402

RESULTS = HERE / "results" / "18_ratio_consistency"
BASELINE = HERE / "results" / "17_raw_multi_sampling" / "sampling_metrics.json"
KERNELS = ("poisson", "nb", "twopois")


def parse_args():
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
    p.add_argument("--skip-dyn-norm", action="store_true")
    return p.parse_args()


def split_csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def prepare(kind, mode, args, name):
    args.ratio = True
    args.moment_param = "raw"
    args.ratio_loss = "raw"
    args.consistency = mode
    prior = getattr(args, "prior_moments", None)
    if prior is None or len(list(prior)) < 3:
        x = load_counts(ROOT / "data" / name / "train.npy").reshape(-1).double()
        args.prior_moments = [float(x.pow(k).mean()) for k in range(1, 4)]
    if kind == "hard":
        args.moment_loss = "raw_ratio"
    else:
        args.moment_loss = "raw_multi"
    return args


def sample_dir(out, tag, name):
    return Path(out) / "sample" / tag / name


def frac(num, den):
    if not den:
        return float("nan")
    return float(num) / float(den)


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def run_sample(names, kernels, specs, cli, device, out):
    for spec in specs:
        if not spec["sample"]:
            continue
        for name in names:
            ckpt = ckpt_of(spec["kind"], name, spec.get("family"))
            dest = sample_dir(out, spec["tag"], name)
            dest.mkdir(parents=True, exist_ok=True)
            model, args = load_model(ckpt, device)
            prepare(spec["kind"], spec["mode"], args, name)
            args.sample_batch = cli.sample_batch
            gammas = make_gammas("quad", 100, 0.0, float(args.lbd), device, power=2.0)
            print(
                f"sample {spec['tag']} {name} mode={spec['mode']} quad T=100 ckpt={ckpt}",
                flush=True,
            )
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
                    model,
                    cli.n_sample,
                    kernel,
                    args,
                    device,
                    gammas=gammas,
                    counters=counters,
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
                json.dumps(
                    {
                        "tag": spec["tag"],
                        "kind": spec["kind"],
                        "mode": spec["mode"],
                        "name": name,
                        "schedule": "quad",
                        "steps": 100,
                        "lbd": float(args.lbd),
                        "ckpt": str(ckpt),
                    },
                    indent=2,
                )
                + "\n"
            )
            del model


def baseline_rows(names, kernels):
    if not BASELINE.is_file():
        print(f"missing baseline {BASELINE}", flush=True)
        return []
    rows = []
    for row in json.loads(BASELINE.read_text()):
        if row.get("weight") not in ("equal", "dyn_norm"):
            continue
        if row.get("dist") not in names or row.get("kernel") not in kernels:
            continue
        item = dict(row)
        item["tag"] = row["weight"]
        item["source"] = "17_raw_multi_sampling"
        rows.append(item)
    return rows


def collect_metrics(names, kernels, specs, out, force_plot, source="18_ratio_consistency"):
    rows = []
    for spec in specs:
        if not spec["sample"]:
            continue
        for name in names:
            dest = sample_dir(out, spec["tag"], name)
            gens = [k for k in kernels if (dest / f"samples_{k}.npy").is_file()]
            if not gens:
                continue
            side = dest / "metrics.json"
            table = None
            if side.is_file() and not force_plot:
                table = json.loads(side.read_text())
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
                        "tag": spec["tag"],
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
                        "source": source,
                    }
                )
    if any(s["tag"] == "equal" for s in specs):
        rows.extend([r for r in baseline_rows(names, kernels) if r["tag"] == "equal"])
    if any(s["tag"] == "dyn_norm" for s in specs):
        rows.extend([r for r in baseline_rows(names, kernels) if r["tag"] == "dyn_norm"])
    return rows


def write_table(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
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
    cli = parse_args()
    names = split_csv(cli.names)
    kernels = split_csv(cli.kernels)
    device = torch.device(cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu")
    out = Path(cli.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    specs = tag_specs(skip_dyn_norm=cli.skip_dyn_norm)
    if cli.stage in ("all", "sample"):
        missing = missing_ckpts(names, skip_dyn_norm=cli.skip_dyn_norm)
        if missing:
            for path in missing:
                print(f"missing {path}", flush=True)
            raise SystemExit("train the missing consistency runs, then rerun sampling")
        run_sample(names, kernels, specs, cli, device, out)
    if cli.stage in ("all", "tables"):
        rows = collect_metrics(names, kernels, specs, out, cli.force)
        (out / "sampling_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        write_table(out / "sampling_metrics.txt", rows)
        (out / "config.txt").write_text(
            "\n".join(
                [
                    "ratio consistency sampling",
                    "schedule: quadratic gamma(u)=lambda*u^2, T=100, lambda=100, gamma from 0 to 100",
                    "new models: soft, anchored, hard",
                    "projections: equal_proj, dyn_norm_proj",
                    "equal and dyn_norm rows are copied from experiment 17",
                    "hard: S2=S1(z)S1(z+1), S3=S2*S1(z+2)",
                    "project: Euclidean projection of [a1(z), a1(z+1), a1(z+2), a2(z), a3(z)]",
                    f"names: {','.join(names)}",
                    f"kernels: {','.join(kernels)}",
                    f"n_sample: {cli.n_sample}",
                    f"seed: {cli.seed}",
                    "",
                ]
            )
        )
        (out / "git_commit.txt").write_text(git_rev() + "\n")
        path = write_comparison(out)
        print(f"tables -> {path}", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
