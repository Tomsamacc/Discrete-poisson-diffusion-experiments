"""k_max=1→2→3 vs learned-mean. Poisson kernel only. Oracle |log m| by γ.

Ckpts (no retrain here)
-----------------------
mean   experiments/scale/{name}/best.pt
k1     experiments/pmfratio/kmax1/{name}/best.pt
k2     experiments/pmfratio/kmax2/{name}/best.pt
k3     experiments/pmfratio/{name}/best.pt

Outputs: test_exp/results/12_kmax/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from loss.oracle import BIN_LABELS, format_mom_line, gamma_bin_index, oracle_moments, support_logp  # noqa: E402
from loss.loss import posterior_moments_from_logB  # noqa: E402
from sample import generate, make_gammas, predict_x  # noqa: E402
from sample_tests import load_model  # noqa: E402
from test import plot_experiment  # noqa: E402
from train import load_counts, q_sample  # noqa: E402

RESULTS = HERE / "results" / "12_kmax"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
MODELS = ("mean", "k1", "k2", "k3")
MODEL_LABEL = {
    "mean": "learned mean",
    "k1": r"$k_{\max}=1$",
    "k2": r"$k_{\max}=2$",
    "k3": r"$k_{\max}=3$",
}
COLORS = {"mean": "0.25", "k1": "#1f77b4", "k2": "#ff7f0e", "k3": "#d62728"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--n_sample", type=int, default=50000)
    p.add_argument("--n_mom", type=int, default=16384)
    p.add_argument("--sample_batch", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--stage",
        default="all",
        choices=("all", "moments", "sample", "plot"),
    )
    return p.parse_args()


def split_csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def ckpt_of(tag, name):
    if tag == "mean":
        return ROOT / "experiments" / "scale" / name / "best.pt"
    if tag == "k1":
        return ROOT / "experiments" / "pmfratio" / "kmax1" / name / "best.pt"
    if tag == "k2":
        return ROOT / "experiments" / "pmfratio" / "kmax2" / name / "best.pt"
    if tag == "k3":
        return ROOT / "experiments" / "pmfratio" / name / "best.pt"
    raise ValueError(tag)


@torch.no_grad()
def predicted_moments(model, z, gamma, args):
    if bool(getattr(args, "ratio", False)):
        from sample import predict_log_B

        log_B = predict_log_B(model, z, gamma, args)
        return posterior_moments_from_logB(log_B, z)
    return [predict_x(model, z, gamma, args)]


@torch.no_grad()
def moment_table(model, args, x, device, xs, logp, n_mom, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    n = min(int(n_mom), int(x.shape[0]))
    idx = torch.randperm(x.shape[0], generator=g)[:n]
    xb = x[idx].to(device)
    t = torch.rand(n, device=device) * (1.0 - args.t_eps) + args.t_eps
    z = q_sample(xb, t, args.lbd)
    gamma = t * args.lbd
    hats = predicted_moments(model, z, gamma, args)
    k_use = len(hats)
    stars = oracle_moments(z, gamma, xs, logp, k_max=k_use)
    bins = gamma_bin_index(gamma).cpu().numpy()
    rows = []
    summary = np.full((k_use, 4), np.nan)
    counts = np.zeros(4, dtype=np.int64)
    for b in range(4):
        mask = bins == b
        counts[b] = int(mask.sum())
        if not mask.any():
            continue
        for k in range(k_use):
            h = hats[k].reshape(-1).double().clamp_min(1e-12).cpu().numpy()
            s = stars[k].reshape(-1).double().clamp_min(1e-12).cpu().numpy()
            err = np.abs(np.log(h[mask]) - np.log(s[mask]))
            summary[k, b] = float(err.mean())
            rows.append(
                {
                    "k": k + 1,
                    "bin": BIN_LABELS[b],
                    "n": int(mask.sum()),
                    "mean_abs_log": float(err.mean()),
                    "median_abs_log": float(np.median(err)),
                    "p90_abs_log": float(np.quantile(err, 0.9)),
                }
            )
    return rows, summary, counts


def write_text_table(path, rows, keys):
    path = Path(path)
    lines = [" ".join(f"{k:16}" for k in keys)]
    for r in rows:
        bits = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                bits.append(f"{v:16.4f}")
            else:
                bits.append(f"{str(v):16}")
        lines.append(" ".join(bits))
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text(), end="", flush=True)


def plot_wd(metrics, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for r in metrics:
        if r["dist"] not in names:
            names.append(r["dist"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    x = np.arange(len(MODELS))
    for ax, metric, title in zip(axes, ("wd", "tv"), ("WD", "TV")):
        for sched, ls, marker in (("unif_T100", "-", "o"), ("quad_T100", "--", "s")):
            for name in names:
                ys = []
                for tag in MODELS:
                    hit = [
                        r[metric]
                        for r in metrics
                        if r["dist"] == name and r["tag"] == tag and r["sched"] == sched
                    ]
                    ys.append(hit[0] if hit else np.nan)
                ax.plot(x, ys, ls=ls, marker=marker, label=f"{name} {sched}", lw=1.4, ms=5)
        ax.set_xticks(x, [MODEL_LABEL[t] for t in MODELS], fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f"Poisson T=100  {title} vs $k_{{\\max}}$")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(dest / "wd_tv_vs_kmax.png", dpi=160, bbox_inches="tight")
    fig.savefig(dest / "wd_tv_vs_kmax.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_mom(summary, dest):
    """summary[(name, tag)] = array [k_max, 4]."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for name, _ in summary:
        if name not in names:
            names.append(name)
    for k_plot, fname, title in (
        (1, "m1_err_by_gamma", r"$|\log\hat m_1-\log m_1^*|$"),
        (2, "m2_err_by_gamma", r"$|\log\hat m_2-\log m_2^*|$"),
        (3, "m3_err_by_gamma", r"$|\log\hat m_3-\log m_3^*|$"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharey=False)
        axes = axes.ravel()
        x = np.arange(4)
        width = 0.18
        tags_k = [t for t in MODELS if t == "mean" or int(t[1:]) >= k_plot]
        if k_plot > 1:
            tags_k = [t for t in MODELS if t != "mean" and int(t[1:]) >= k_plot]
        for ax, name in zip(axes, names):
            for i, tag in enumerate(tags_k):
                arr = summary.get((name, tag))
                if arr is None or arr.shape[0] < k_plot:
                    continue
                ax.bar(
                    x + (i - 0.5 * (len(tags_k) - 1)) * width,
                    arr[k_plot - 1],
                    width=width,
                    color=COLORS[tag],
                    label=MODEL_LABEL[tag],
                )
            ax.set_xticks(x, BIN_LABELS, fontsize=7)
            ax.set_title(name, fontsize=9)
            ax.set_ylabel(title, fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=6, frameon=False)
        fig.suptitle(f"oracle moment error by γ  {title}", fontsize=11)
        fig.tight_layout()
        fig.savefig(dest / f"{fname}.png", dpi=160, bbox_inches="tight")
        fig.savefig(dest / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    cli = parse_args()
    names = split_csv(cli.names)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    device = torch.device(
        cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu"
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    do_mom = cli.stage in ("all", "moments")
    do_sample = cli.stage in ("all", "sample")
    do_plot = cli.stage in ("all", "plot")

    loaded = {}
    for name in names:
        for tag in MODELS:
            ckpt = ckpt_of(tag, name)
            if not ckpt.is_file():
                print(f"skip model {tag} {name}: no {ckpt}", flush=True)
                continue
            print(f"load {tag} {name} <- {ckpt}", flush=True)
            loaded[(name, tag)] = load_model(ckpt, device)

    mom_rows = []
    mom_summary = {}
    if do_mom:
        for name in names:
            xs, logp = support_logp(name, device)
            xval = load_counts(ROOT / "data" / name / "val.npy")
            for tag in MODELS:
                pair = loaded.get((name, tag))
                if pair is None:
                    continue
                model, args = pair
                args.sample_batch = cli.sample_batch
                rows, summary, counts = moment_table(
                    model, args, xval, device, xs, logp, cli.n_mom, cli.seed
                )
                mom_summary[(name, tag)] = summary
                print(
                    f"  mom {name:12} {tag:4} n={int(counts.sum())}  {format_mom_line(summary)}",
                    flush=True,
                )
                for r in rows:
                    mom_rows.append({"dist": name, "model": tag, **r})
        if mom_rows:
            write_text_table(
                RESULTS / "moments.txt",
                mom_rows,
                ("dist", "model", "k", "bin", "n", "mean_abs_log", "median_abs_log", "p90_abs_log"),
            )
            (RESULTS / "moments.json").write_text(json.dumps(mom_rows, indent=2) + "\n")

    metrics = []
    if do_sample:
        for name in names:
            for tag in MODELS:
                pair = loaded.get((name, tag))
                if pair is None:
                    continue
                model, args = pair
                args.sample_batch = cli.sample_batch
                g1 = float(args.lbd)
                for sched, power, stag in (("uniform", 1.0, "unif_T100"), ("quad", 2.0, "quad_T100")):
                    dest = RESULTS / "sample" / tag / name / stag
                    dest.mkdir(parents=True, exist_ok=True)
                    path = dest / "samples_poisson.npy"
                    gammas = make_gammas("quad", 100, 0.0, g1, device, power=power)
                    if path.is_file():
                        print(f"    skip sample {tag} {name} {stag}", flush=True)
                    else:
                        print(
                            f"    sample Poisson {tag} {name} {stag} T=100 power={power}",
                            flush=True,
                        )
                        x = generate(model, cli.n_sample, "poisson", args, device, gammas=gammas)
                        np.save(path, x)
                    (dest / "meta.json").write_text(
                        json.dumps(
                            {
                                "name": name,
                                "model": tag,
                                "tag": stag,
                                "kernel": "poisson",
                                "k_max": int(getattr(args, "k_max", 1)),
                                "ratio": bool(getattr(args, "ratio", False)),
                                "ckpt": str(ckpt_of(tag, name)),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    table = plot_experiment(dest, name=name, kernels=("poisson",))
                    for r in table:
                        metrics.append(
                            {
                                "dist": name,
                                "tag": tag,
                                "sched": stag,
                                "kernel": "poisson",
                                "wd": r["wd"],
                                "tv": r["tv"],
                            }
                        )
        if metrics:
            write_text_table(
                RESULTS / "metrics.txt",
                metrics,
                ("dist", "tag", "sched", "kernel", "wd", "tv"),
            )

    if do_plot:
        if not metrics and (RESULTS / "metrics.txt").is_file():
            pass
        if metrics:
            plot_wd(metrics, RESULTS)
        if mom_summary:
            plot_mom(mom_summary, RESULTS)
    print("done ->", RESULTS, flush=True)


if __name__ == "__main__":
    main()
