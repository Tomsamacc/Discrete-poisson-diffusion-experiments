"""PMF line overlays (same style as the old kernel_bench plots)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from train import ROOT, load_yaml, resolve

KERNEL_LABELS = {
    "True": "True",
    "poisson": "Poisson",
    "nb": "NB",
    "repoisson": r"RePoisson ($z\sim\mathrm{Pois}(\gamma^+ \hat x)$)",
    "twopois": r"Two-Poisson ($w$Pois$+(1-w)$Pois)",
}
COLORS = {
    "True": "black",
    "poisson": "#ff7f0e",
    "nb": "#2ca02c",
    "repoisson": "#9467bd",
    "twopois": "#d62728",
}
MASS = 0.999
PLOT_CAP = 400


def emp_pmf_window(samples, lo, hi):
    x = np.round(np.asarray(samples)).astype(np.int64).reshape(-1)
    n = max(x.size, 1)
    in_win = (x >= lo) & (x <= hi)
    counts = np.bincount(x[in_win] - lo, minlength=hi - lo + 1).astype(np.float64)
    overflow = float((~in_win).mean()) if x.size else 0.0
    return counts / n, overflow


def wd1(a, b):
    a = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    n = min(a.size, b.size)
    return float(np.abs(a[:n] - b[:n]).mean())


def l1_window(true_p, emp):
    return 0.5 * float(np.abs(true_p - emp).sum())


def dist_kind(name):
    return "continuous" if name == "gamma_ltj" else "discrete"


def true_pmf(name, ks):
    ks = np.asarray(ks)
    if name == "nb":
        return stats.nbinom.pmf(ks, n=5, p=0.2)
    if name == "pois20":
        return stats.poisson.pmf(ks, 20)
    if name == "poismix_mod":
        return 0.5 * stats.poisson.pmf(ks, 5) + 0.5 * stats.poisson.pmf(ks, 50)
    if name == "poissmix":
        return 0.1 * stats.poisson.pmf(ks, 1) + 0.9 * stats.poisson.pmf(ks, 100)
    if name == "poissmix3":
        return (
            (1.0 / 3.0) * stats.poisson.pmf(ks, 1)
            + (1.0 / 3.0) * stats.poisson.pmf(ks, 50)
            + (1.0 / 3.0) * stats.poisson.pmf(ks, 100)
        )
    if name == "zip":
        p = 0.3 * stats.poisson.pmf(ks, 5.0)
        return np.where(ks == 0, p + 0.7, p)
    if name == "yule_simon":
        return stats.yulesimon.pmf(ks, 2.0)
    raise ValueError(name)


def true_cdf(name, k):
    if name == "nb":
        return float(stats.nbinom.cdf(k, n=5, p=0.2))
    if name == "pois20":
        return float(stats.poisson.cdf(k, 20))
    if name == "poismix_mod":
        return float(0.5 * stats.poisson.cdf(k, 5) + 0.5 * stats.poisson.cdf(k, 50))
    if name == "poissmix":
        return float(0.1 * stats.poisson.cdf(k, 1) + 0.9 * stats.poisson.cdf(k, 100))
    if name == "poissmix3":
        return float(
            (1.0 / 3.0) * stats.poisson.cdf(k, 1)
            + (1.0 / 3.0) * stats.poisson.cdf(k, 50)
            + (1.0 / 3.0) * stats.poisson.cdf(k, 100)
        )
    if name == "zip":
        return float(0.7 + 0.3 * stats.poisson.cdf(k, 5.0)) if k >= 0 else 0.0
    if name == "yule_simon":
        return float(stats.yulesimon.cdf(k, 2.0))
    raise ValueError(name)


def k_plot_of(name):
    for k in range(0, PLOT_CAP + 1):
        if true_cdf(name, k) >= MASS:
            return k
    return PLOT_CAP


def true_pdf_gamma(xs):
    return stats.gamma.pdf(xs, a=1.0, scale=10.0)


def plot_pmf_curve(ax, ks, p, name, logy, n_gen):
    p = np.asarray(p, dtype=np.float64)
    color = COLORS.get(name, "gray")
    label = KERNEL_LABELS.get(name, name)
    lw = 2.0 if name == "True" else 1.35
    z = 5 if name == "True" else 3
    if name == "True":
        z, lw = 5, 2.0
    elif name in ("nb", "twopois"):
        z, lw = 4, 1.7
    ls = "-"
    if logy:
        floor = 0.5 / max(n_gen, 1)
        pos = p > 0
        if not pos.any():
            return
        ax.plot(
            ks[pos],
            np.clip(p[pos], floor, None),
            ls=ls,
            lw=lw,
            color=color,
            marker="o",
            ms=2.2 if name == "True" else 1.6,
            mew=0,
            zorder=z,
            label=label,
        )
        return
    if name != "True":
        p = gaussian_filter1d(p, sigma=1.0)
    ax.plot(ks, p, ls=ls, lw=lw, color=color, zorder=z, label=label)


def style_ax(ax, logy, pmax, n_gen, ks=None, lo=None, hi=None):
    if ks is not None:
        ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
    elif lo is not None:
        ax.set_xlim(lo, hi)
    if logy:
        floor = 0.5 / max(n_gen, 1)
        ymin = max(floor * 0.4, 1e-6)
        ymax = min(max(pmax * 2.8, 3e-2), 1.05)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.axhline(floor, color="0.55", ls=":", lw=0.8, zorder=1)
    else:
        ax.set_ylim(bottom=0, top=pmax * 1.15 if pmax > 0 else 1.0)
    ax.grid(True, alpha=0.25, which="both")


def tv_pdf_hist(gen, pdf_fn, lo, hi, bins=256):
    x = np.asarray(gen, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    hist, edges = np.histogram(x, bins=int(bins), range=(float(lo), float(hi)), density=True)
    xc = 0.5 * (edges[:-1] + edges[1:])
    dx = float(edges[1] - edges[0])
    p = np.asarray(pdf_fn(xc), dtype=np.float64)
    p = np.where(np.isfinite(p), p, 0.0)
    return 0.5 * float(np.abs(hist - p).sum() * dx)


def load_gens(out_dir, kernels):
    gens = {}
    out_dir = Path(out_dir)
    for k in kernels:
        path = out_dir / f"samples_{k}.npy"
        if path.is_file():
            gens[k] = np.load(path)
    return gens


def infer_name(out_dir):
    return Path(out_dir).name


def plot_discrete_ax(ax, name, gens, logy, n_gen):
    k_plot = k_plot_of(name)
    ks = np.arange(0, k_plot + 1)
    true_p = true_pmf(name, ks)
    pmfs = {"True": true_p}
    overflow = {}
    for k, gen in gens.items():
        emp, ov = emp_pmf_window(gen, 0, k_plot)
        pmfs[k] = emp
        overflow[k] = ov
    pmax = 0.0
    for key in ["True", *gens]:
        p = np.asarray(pmfs[key], dtype=np.float64)
        plot_pmf_curve(ax, ks, p, key, logy, n_gen)
        pmax = max(pmax, float(np.nanmax(p)) if p.size else 0.0)
    style_ax(ax, logy, pmax, n_gen, ks=ks)
    ov = ", ".join(f"{k}={overflow[k]:.3f}" for k in gens)
    if ov:
        ax.text(0.02, 0.02, "overflow " + ov, transform=ax.transAxes, va="bottom", fontsize=6)
    ax.set_title(f"{name}  $K_{{0.999}}={k_plot}$", fontsize=9)
    return pmfs, overflow, ks, true_p


def plot_gamma_ax(ax, gens, logy, n_gen, true_x):
    hi = float(stats.gamma.ppf(MASS, a=1.0, scale=10.0))
    extras = [] if true_x is None else [true_x]
    for g in list(gens.values()) + extras:
        x = np.asarray(g, dtype=np.float64).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size:
            hi = max(hi, float(np.quantile(x, 0.995)))
    lo, hi = 0.0, max(hi, 1e-3)
    xs = np.linspace(lo, hi, 400)
    ys = true_pdf_gamma(xs)
    pmax = float(np.nanmax(ys[np.isfinite(ys)])) if ys.size else 1.0
    plot_pmf_curve(ax, xs, ys, "True", logy, n_gen)
    bins = 80
    for k, gen in gens.items():
        x = np.asarray(gen, dtype=np.float64).reshape(-1)
        hist, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
        xc = 0.5 * (edges[:-1] + edges[1:])
        plot_pmf_curve(ax, xc, hist, k, logy, n_gen)
        if hist.size:
            pmax = max(pmax, float(np.nanmax(hist)))
    style_ax(ax, logy, pmax, n_gen, lo=lo, hi=hi)
    ax.set_title("gamma_ltj  PDF", fontsize=9)


def metrics_discrete(name, gens, true_x, pmfs, overflow, true_p):
    rows = []
    for k, gen in gens.items():
        rows.append(
            {
                "dist": name,
                "kernel": k,
                "wd": wd1(gen, true_x),
                "tv": l1_window(true_p, pmfs[k]),
                "overflow": overflow[k],
            }
        )
    return rows


def plot_experiment(out_dir, name=None, kernels=("poisson", "nb", "repoisson"), data_root=None):
    out_dir = Path(out_dir)
    name = name or infer_name(out_dir)
    data_root = Path(data_root) if data_root else ROOT / "data" / name
    gens = load_gens(out_dir, kernels)
    if not gens:
        print(f"no samples in {out_dir}", flush=True)
        return []
    val_path = data_root / "val.npy"
    true_x = np.load(val_path) if val_path.is_file() else None
    n_gen = max(len(v) for v in gens.values())
    table = []

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    if dist_kind(name) == "discrete":
        pmfs, overflow, ks, true_p = plot_discrete_ax(axes[0], name, gens, False, n_gen)
        plot_discrete_ax(axes[1], name, gens, True, n_gen)
        if true_x is not None:
            table = metrics_discrete(name, gens, true_x, pmfs, overflow, true_p)
    else:
        plot_gamma_ax(axes[0], gens, False, n_gen, true_x)
        plot_gamma_ax(axes[1], gens, True, n_gen, true_x)
        if true_x is not None:
            lo = 0.0
            hi = float(stats.gamma.ppf(MASS, a=1.0, scale=10.0))
            for g in list(gens.values()) + [true_x]:
                x = np.asarray(g, dtype=np.float64).reshape(-1)
                x = x[np.isfinite(x)]
                if x.size:
                    hi = max(hi, float(np.quantile(x, 0.995)))
            for k, gen in gens.items():
                table.append(
                    {
                        "dist": name,
                        "kernel": k,
                        "wd": wd1(gen, true_x),
                        "tv": tv_pdf_hist(gen, true_pdf_gamma, lo, hi),
                        "overflow": 0.0,
                    }
                )
    axes[0].legend(loc="best", fontsize=7, frameon=False)
    axes[0].set_title(axes[0].get_title() + "  lin")
    axes[1].set_title(axes[1].get_title() + "  log")
    fig.tight_layout()
    png = out_dir / "pmf.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "pmf.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png}", flush=True)
    return table


def plot_grid(exp_root, names, kernels, logy, path):
    n = len(names)
    ncols = min(n, 4) if n else 1
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.5 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()
    n_gen = 50000
    for ax, name in zip(axes, names):
        out_dir = Path(exp_root) / name
        gens = load_gens(out_dir, kernels)
        if not gens:
            ax.set_title(f"{name} (missing)")
            continue
        val = ROOT / "data" / name / "val.npy"
        true_x = np.load(val) if val.is_file() else None
        n_gen = max(len(v) for v in gens.values())
        if dist_kind(name) == "discrete":
            plot_discrete_ax(ax, name, gens, logy, n_gen)
        else:
            plot_gamma_ax(ax, gens, logy, n_gen, true_x)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(loc="best", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def write_metrics(table, path):
    lines = [f"{'dist':14} {'kernel':16} {'wd':>10} {'tv':>10}"]
    for r in table:
        tv = r.get("tv")
        tv_s = f"{tv:10.4f}" if tv is not None and np.isfinite(tv) else f"{'nan':>10}"
        lines.append(f"{r['dist']:14} {r['kernel']:16} {r['wd']:10.4f} {tv_s}")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text)
    print(text, end="", flush=True)


def parse_args():
    cfg = load_yaml(ROOT / "config.yml")
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out_dir", default=None, help="one experiment folder; default = all")
    p.add_argument("--exp_root", default=str(ROOT / "experiments"))
    p.add_argument("--kernels", default=cfg.get("kernels", "poisson,nb,repoisson,twopois"))
    p.add_argument(
        "--names",
        default="gamma_ltj,nb,poismix_mod,pois20,poissmix,poissmix3,zip,yule_simon",
    )
    return p.parse_args()


def main():
    args = parse_args()
    kernels = tuple(k.strip() for k in args.kernels.split(",") if k.strip())
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    table = []
    if args.out_dir:
        table.extend(plot_experiment(resolve(args.out_dir), kernels=kernels))
    else:
        root = resolve(args.exp_root)
        for name in names:
            table.extend(plot_experiment(root / name, name=name, kernels=kernels))
        plot_grid(root, names, kernels, False, root / "pmf.png")
        plot_grid(root, names, kernels, True, root / "pmf_log.png")
    if table:
        dest = resolve(args.out_dir) if args.out_dir else resolve(args.exp_root)
        write_metrics(table, Path(dest) / "metrics.txt")


if __name__ == "__main__":
    main()
