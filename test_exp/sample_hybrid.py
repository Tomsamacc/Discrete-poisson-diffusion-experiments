"""Quadratic T=100 sampling for hybrid_eq and hybrid_dyn.

Poisson, NB, and TwoPois. On-policy m1 diagnostics use the first
8192 trajectories of the same seed. Direct learned-mean rows are
copied from 02_exponent p=2. dyn_norm rows are copied from experiment 17.
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

from loss.oracle import BIN_LABELS, support_logp  # noqa: E402
from sample import generate, make_gammas  # noqa: E402
from sample_tests import load_model  # noqa: E402
from test import plot_experiment  # noqa: E402
from train import load_counts  # noqa: E402

NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
TAGS = ("hybrid_eq", "hybrid_dyn")
KERNELS = ("poisson", "nb", "twopois")
RESULTS = HERE / "results" / "21_hybrid"
MEAN_METRICS = HERE / "results" / "02_exponent" / "metrics.txt"
DYN_METRICS = HERE / "results" / "17_raw_multi_sampling" / "sampling_metrics.json"


def qstats(arr):
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "median": None, "p90": None, "p99": None, "max": None}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


def ckpt_of(tag, name):
    return ROOT / "experiments" / "hybrid" / tag / name / "best.pt"


def prepare(args, name):
    args.ratio = True
    args.moment_param = "raw"
    args.ratio_loss = "raw"
    args.moment_loss = "hybrid"
    args.k_max = 3
    args.consistency = "none"
    prior = getattr(args, "prior_moments", None)
    if prior is None or len(list(prior)) < 3:
        x = load_counts(ROOT / "data" / name / "train.npy").reshape(-1).double()
        args.prior_moments = [float(x.pow(k).mean()) for k in range(1, 4)]
    return args


def frac(num, den):
    if not den:
        return float("nan")
    return float(num) / float(den)


def summarize_on_policy(store):
    chunks = store.get("chunks") or []
    if not chunks:
        return {"seen": int(store.get("seen", 0)), "bins": []}
    idx = torch.cat([c["idx"] for c in chunks]).numpy()
    packs = {
        "log1": torch.cat([c["log"][0] for c in chunks]).numpy(),
        "log2": torch.cat([c["log"][1] for c in chunks]).numpy(),
        "log3": torch.cat([c["log"][2] for c in chunks]).numpy(),
        "habs1": torch.cat([c["habs"] for c in chunks]).numpy(),
        "kl": torch.cat([c["kl"] for c in chunks]).numpy(),
        "eta2": torch.cat([c["eta2"] for c in chunks]).numpy(),
        "eta3": torch.cat([c["eta3"] for c in chunks]).numpy(),
    }
    rows = []
    masks = [("all", np.ones(idx.shape[0], dtype=bool))]
    for b, lab in enumerate(BIN_LABELS):
        masks.append((lab, idx == b))
    for lab, m in masks:
        row = {"bin": lab, "n": int(m.sum())}
        for key, arr in packs.items():
            row[key] = qstats(arr[m])
        rows.append(row)
    store["chunks"].clear()
    return {"seen": int(store.get("seen", 0)), "bins": rows}


def run_sample(names, kernels, cli, device, out):
    for tag in TAGS:
        for name in names:
            ckpt = ckpt_of(tag, name)
            dest = out / "sample" / tag / name
            dest.mkdir(parents=True, exist_ok=True)
            model, args = load_model(ckpt, device)
            prepare(args, name)
            args.sample_batch = cli.sample_batch
            gammas = make_gammas("quad", 100, 0.0, float(args.lbd), device, power=2.0)
            xs, logp = support_logp(name, device)
            print(f"sample {tag} {name} quad T=100 ckpt={ckpt}", flush=True)
            on_policy_out = {}
            for kernel in kernels:
                path = dest / f"samples_{kernel}.npy"
                cpath = dest / f"counters_{kernel}.json"
                opath = dest / f"on_policy_{kernel}.json"
                if path.is_file() and cpath.is_file() and opath.is_file() and not cli.force:
                    print(f"  skip {kernel}", flush=True)
                    continue
                torch.manual_seed(cli.seed)
                np.random.seed(cli.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(cli.seed)
                counters = {}
                on_policy = {"xs": xs, "logp": logp, "n": int(cli.diag_n), "seen": 0, "chunks": []}
                x = generate(
                    model,
                    cli.n_sample,
                    kernel,
                    args,
                    device,
                    gammas=gammas,
                    counters=counters,
                    on_policy=on_policy,
                )
                np.save(path, x)
                cpath.write_text(json.dumps(counters) + "\n")
                summary = summarize_on_policy(on_policy)
                opath.write_text(json.dumps(summary) + "\n")
                on_policy_out[kernel] = summary
                line = f"  saved {kernel} E[z/gamma]={float(np.mean(x)):.4g}"
                if counters.get("traj"):
                    line += f" runaway={frac(counters.get('runaway_traj', 0), counters['traj']):.4g}"
                if kernel == "twopois" and counters.get("n"):
                    line += f" fallback={frac(counters.get('fallback', 0), counters['n']):.4g}"
                print(line, flush=True)
            (dest / "meta.json").write_text(
                json.dumps(
                    {
                        "tag": tag,
                        "name": name,
                        "schedule": "quad",
                        "steps": 100,
                        "power": 2,
                        "lbd": float(args.lbd),
                        "ckpt": str(ckpt),
                        "diag_n": int(cli.diag_n),
                    },
                    indent=2,
                )
                + "\n"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


def parse_mean_rows(names, kernels):
    rows = []
    if not MEAN_METRICS.is_file():
        print(f"missing {MEAN_METRICS}", flush=True)
        return rows
    for line in MEAN_METRICS.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] == "power":
            continue
        power, dist, kernel, wd, tv = parts
        if abs(float(power) - 2.0) > 1e-6:
            continue
        if dist not in names or kernel not in kernels:
            continue
        rows.append(
            {
                "tag": "mean",
                "dist": dist,
                "kernel": kernel,
                "wd": float(wd),
                "tv": float(tv),
                "overflow": float("nan"),
                "runaway": float("nan"),
                "fallback": float("nan"),
                "invalid_v": float("nan"),
                "invalid_atom": float("nan"),
                "source": "02_exponent_p2",
            }
        )
    return rows


def parse_dyn_rows(names, kernels):
    rows = []
    if not DYN_METRICS.is_file():
        print(f"missing {DYN_METRICS}", flush=True)
        return rows
    for row in json.loads(DYN_METRICS.read_text()):
        if row.get("weight") != "dyn_norm":
            continue
        if row.get("dist") not in names or row.get("kernel") not in kernels:
            continue
        item = dict(row)
        item["tag"] = "dyn_norm"
        item["source"] = "17_raw_multi_sampling"
        rows.append(item)
    return rows


def collect(names, kernels, out, force_plot):
    rows = []
    for tag in TAGS:
        for name in names:
            dest = out / "sample" / tag / name
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
                        "source": "21_hybrid",
                    }
                )
    rows.extend(parse_mean_rows(names, kernels))
    rows.extend(parse_dyn_rows(names, kernels))
    return rows


def write_table(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = ["tag", "dist", "kernel", "wd", "tv", "overflow", "runaway", "fallback", "invalid_v", "invalid_atom", "source"]
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


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--kernels", default=",".join(KERNELS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--n-sample", type=int, default=50000)
    p.add_argument("--diag-n", type=int, default=8192)
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
        missing = [str(ckpt_of(tag, name)) for tag in TAGS for name in names if not ckpt_of(tag, name).is_file()]
        if missing:
            for path in missing:
                print(f"missing {path}", flush=True)
            raise SystemExit("train hybrid runs first: ./script/train_hybrid.sh")
        run_sample(names, kernels, cli, device, out)
    if cli.stage in ("all", "tables"):
        rows = collect(names, kernels, out, cli.force)
        (out / "sampling_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        write_table(out / "sampling_metrics.txt", rows)
        (out / "config.txt").write_text(
            "\n".join(
                [
                    "hybrid m1-protected sampling",
                    "schedule: quadratic gamma(u)=lambda*u^2, T=100, lambda=100, p=2",
                    "hybrid_eq: D(X, m1) + D(R2, S2) + D(R3, S3)",
                    "hybrid_dyn: D(X, m1) + (h/2) D(R2, S2) + (h^2/6) D(R3, S3)",
                    "mean rows: direct learned mean, copied from 02_exponent p=2",
                    "dyn_norm rows: raw multi-head, copied from experiment 17",
                    "on-policy diagnostics: first diag_n trajectories, every reverse step",
                    f"names: {','.join(names)}",
                    f"kernels: {','.join(kernels)}",
                    f"n_sample: {cli.n_sample}",
                    f"diag_n: {cli.diag_n}",
                    f"seed: {cli.seed}",
                    "",
                ]
            )
        )
        (out / "git_commit.txt").write_text(git_rev() + "\n")
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
