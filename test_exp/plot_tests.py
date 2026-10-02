"""Summary PMF grids + WD/TV for test_exp sampling ablations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test import (  # noqa: E402
    dist_kind,
    load_gens,
    metrics_discrete,
    plot_discrete_ax,
    plot_gamma_ax,
    wd1,
    tv_pdf_hist,
    true_pdf_gamma,
)
from scipy import stats  # noqa: E402

SWEEP_TAGS = ("T100_h1", "T200_h05", "T500_h02", "T1000_h01")
SWEEP_H = {"T100_h1": 1.0, "T200_h05": 0.5, "T500_h02": 0.2, "T1000_h01": 0.1}
ORACLE_TAGS = ("A_baseline", "B_true_mean", "C_exact_mix")
ORACLE_TITLE = {
    "A_baseline": "A current",
    "B_true_mean": r"B $m=E[X]$",
    "C_exact_mix": r"C $X\sim p_X$",
}
DEFAULT_NAMES = "gamma_ltj,nb,poismix_mod,pois20,poissmix,poissmix3,zip,yule_simon"
DEFAULT_KERNELS = "poisson,nb,twopois"


def split_csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default=str(HERE))
    p.add_argument("--names", default=DEFAULT_NAMES)
    p.add_argument("--kernels", default=DEFAULT_KERNELS)
    p.add_argument("--logy", action="store_true", default=True)
    p.add_argument("--nolog", action="store_true")
    p.add_argument("--stage", default="all", choices=("all", "sweep", "quad", "quad01", "unif01", "oracle"))
    return p.parse_args()


def draw_panel(ax, name, gens, logy):
    if not gens:
        ax.set_title(f"{name} (missing)")
        ax.axis("off")
        return None, None
    n_gen = max(len(v) for v in gens.values())
    val = ROOT / "data" / name / "val.npy"
    true_x = np.load(val) if val.is_file() else None
    if dist_kind(name) == "discrete":
        pmfs, overflow, ks, true_p = plot_discrete_ax(ax, name, gens, logy, n_gen)
        rows = []
        if true_x is not None:
            rows = metrics_discrete(name, gens, true_x, pmfs, overflow, true_p)
        return rows, n_gen
    plot_gamma_ax(ax, gens, logy, n_gen, true_x)
    rows = []
    if true_x is not None:
        lo = 0.0
        hi = float(stats.gamma.ppf(0.999, a=1.0, scale=10.0))
        for g in list(gens.values()) + [true_x]:
            x = np.asarray(g, dtype=np.float64).reshape(-1)
            x = x[np.isfinite(x)]
            if x.size:
                hi = max(hi, float(np.quantile(x, 0.995)))
        for k, gen in gens.items():
            rows.append(
                {
                    "dist": name,
                    "kernel": k,
                    "wd": wd1(gen, true_x),
                    "tv": tv_pdf_hist(gen, true_pdf_gamma, lo, hi),
                    "overflow": 0.0,
                }
            )
    return rows, n_gen


def write_metrics(table, path, extra_keys=("test", "tag")):
    keys = list(extra_keys) + ["dist", "kernel", "wd", "tv"]
    widths = {k: max(len(k), 10) for k in keys}
    widths["dist"] = 14
    widths["kernel"] = 16
    widths["tag"] = 16
    widths["test"] = 10
    lines = [" ".join(f"{k:{widths[k]}}" if k not in ("wd", "tv") else f"{k:>10}" for k in keys)]
    for r in table:
        parts = []
        for k in keys:
            if k in ("wd", "tv"):
                v = r.get(k)
                parts.append(f"{v:10.4f}" if v is not None and np.isfinite(v) else f"{'nan':>10}")
            else:
                parts.append(f"{str(r.get(k, '')):{widths[k]}}")
        lines.append(" ".join(parts))
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text)
    print(text, end="", flush=True)


def plot_named_grid(names, col_tags, folder_fn, titles, kernels, logy, path, test_name):
    if len(col_tags) == 1:
        n = len(names)
        ncols = min(4, max(n, 1))
        nrows = int(np.ceil(n / ncols)) if n else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
        table = []
        axes_flat = axes.ravel()
        for i, name in enumerate(names):
            ax = axes_flat[i]
            tag = col_tags[0]
            gens = load_gens(folder_fn(name, tag), kernels)
            rows, _ = draw_panel(ax, name, gens, logy)
            ax.set_title(f"{name}  {titles[0]}", fontsize=9)
            if rows:
                for r in rows:
                    r = dict(r)
                    r["test"] = test_name
                    r["tag"] = tag
                    table.append(r)
        for ax in axes_flat[n:]:
            ax.axis("off")
        axes_flat[0].legend(loc="best", fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}", flush=True)
        return table
    nrows, ncols = len(names), len(col_tags)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
    table = []
    for i, name in enumerate(names):
        for j, tag in enumerate(col_tags):
            ax = axes[i][j]
            folder = folder_fn(name, tag)
            gens = load_gens(folder, kernels)
            rows, _ = draw_panel(ax, name, gens, logy)
            ax.set_title(f"{name}  {titles[j]}", fontsize=9)
            if rows:
                for r in rows:
                    r = dict(r)
                    r["test"] = test_name
                    r["tag"] = tag
                    table.append(r)
        axes[i][0].set_ylabel(name)
    axes[0][0].legend(loc="best", fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)
    return table


def main():
    cli = parse_args()
    names = split_csv(cli.names)
    kernels = split_csv(cli.kernels)
    logy = not cli.nolog
    root = Path(cli.out_root)
    if not root.is_absolute():
        root = ROOT / root
    root = root.resolve()
    table = []
    stages = ("sweep", "quad", "quad01", "oracle") if cli.stage == "all" else (cli.stage,)

    if "sweep" in stages:
        sweep_titles = [f"T={t.split('_')[0][1:]}  h={SWEEP_H[t]:g}" for t in SWEEP_TAGS]
        table.extend(
            plot_named_grid(
                names,
                SWEEP_TAGS,
                lambda name, tag: root / "sweep" / tag / name,
                sweep_titles,
                kernels,
                logy,
                root / "sweep_log.png" if logy else root / "sweep.png",
                "sweep",
            )
        )
    if "quad" in stages:
        table.extend(
            plot_named_grid(
                names,
                ("T100_h1", "T100_quad"),
                lambda name, tag: (
                    root / "sweep" / "T100_h1" / name if tag == "T100_h1" else root / "quad" / tag / name
                ),
                ("uniform T=100  h=1", r"quad T=100  $\gamma_0$=0"),
                kernels,
                logy,
                root / "quad_vs_uniform.png",
                "quad",
            )
        )
    if "quad01" in stages:
        table.extend(
            plot_named_grid(
                names,
                ("T100_quad_g01_zx",),
                lambda name, tag: root / "quad" / tag / name,
                (r"$z\sim\mathrm{Pois}(0.1X)$, $\gamma$:0.1$\to$100 quad",),
                kernels,
                logy,
                root / "quad_g01.png",
                "quad01",
            )
        )
    if "unif01" in stages:
        table.extend(
            plot_named_grid(
                names,
                ("T100_g01_zx",),
                lambda name, tag: root / "unif" / tag / name,
                (r"uniform T=100, $z\sim\mathrm{Pois}(0.1X)$, $\gamma$:0.1$\to$100",),
                kernels,
                logy,
                root / "unif_g01.png",
                "unif01",
            )
        )
    if "oracle" in stages:
        table.extend(
            plot_named_grid(
                names,
                ORACLE_TAGS,
                lambda name, tag: root / "oracle" / tag / name,
                [ORACLE_TITLE[t] for t in ORACLE_TAGS],
                kernels,
                logy,
                root / "oracle.png",
                "oracle",
            )
        )
    dest = root / f"metrics_{cli.stage}.txt" if cli.stage in ("quad01", "unif01") else root / "metrics.txt"
    write_metrics(table, dest)


if __name__ == "__main__":
    main()
