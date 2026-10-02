"""Write test_exp/results/12_kmax/report.md from existing metrics/moments/samples."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from test import dist_kind, plot_discrete_ax, plot_gamma_ax, load_gens  # noqa: E402

RES = HERE / "results" / "12_kmax"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
MODELS = ("mean", "k1", "k2", "k3")
MODEL_LAB = {"mean": "learned mean", "k1": r"$k_{\max}=1$", "k2": r"$k_{\max}=2$", "k3": r"$k_{\max}=3$"}
COLORS = {"mean": "0.25", "k1": "#1f77b4", "k2": "#ff7f0e", "k3": "#d62728"}
BINS = ("[0,0.1]", "[0.1,1]", "[1,10]", "[10,100]")
SCHED = ("unif_T100", "quad_T100")


def parse_metrics(path):
    rows = []
    for line in Path(path).read_text().splitlines()[1:]:
        p = line.split()
        if len(p) < 6:
            continue
        rows.append(
            {
                "dist": p[0],
                "tag": p[1],
                "sched": p[2],
                "kernel": p[3],
                "wd": float(p[4]),
                "tv": float(p[5]),
            }
        )
    return rows


def parse_moments(path):
    rows = []
    for line in Path(path).read_text().splitlines()[1:]:
        p = line.split()
        if len(p) < 8:
            continue
        rows.append(
            {
                "dist": p[0],
                "model": p[1],
                "k": int(p[2]),
                "bin": p[3],
                "n": int(p[4]),
                "mean": float(p[5]),
                "median": float(p[6]),
                "p90": float(p[7]),
            }
        )
    return rows


def get(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def fmt(x, nd=4):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def plot_wd(metrics, dest):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    x = np.arange(len(MODELS))
    for ax, metric, title in zip(axes, ("wd", "tv"), ("WD", "TV")):
        for name in NAMES:
            for sched, ls, mk in (("unif_T100", "-", "o"), ("quad_T100", "--", "s")):
                ys = []
                for tag in MODELS:
                    hit = get(metrics, dist=name, tag=tag, sched=sched)
                    ys.append(hit[metric] if hit else np.nan)
                ax.plot(
                    x,
                    ys,
                    ls=ls,
                    marker=mk,
                    lw=1.4,
                    ms=5,
                    label=f"{name} {sched.replace('_T100','')}",
                )
        ax.set_xticks(x, [MODEL_LAB[t] for t in MODELS], fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f"Poisson T=100  {title}")
        if metric == "wd":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(dest / "wd_tv_vs_kmax.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_mom(moments, dest):
    lookup = {}
    for r in moments:
        lookup[(r["dist"], r["model"], r["k"], r["bin"])] = r["mean"]
    for k_plot, fname, title in (
        (1, "m1_err_by_gamma", r"$|\log\hat m_1-\log m_1^*|$"),
        (2, "m2_err_by_gamma", r"$|\log\hat m_2-\log m_2^*|$"),
        (3, "m3_err_by_gamma", r"$|\log\hat m_3-\log m_3^*|$"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
        axes = axes.ravel()
        x = np.arange(4)
        width = 0.18
        if k_plot == 1:
            tags = list(MODELS)
        else:
            tags = [t for t in MODELS if t != "mean" and int(t[1:]) >= k_plot]
        for ax, name in zip(axes, NAMES):
            plotted = False
            for i, tag in enumerate(tags):
                ys = [lookup.get((name, tag, k_plot, b), np.nan) for b in BINS]
                if np.all(np.isnan(ys)):
                    continue
                ax.bar(
                    x + (i - 0.5 * (len(tags) - 1)) * width,
                    ys,
                    width=width,
                    color=COLORS[tag],
                    label=MODEL_LAB[tag],
                )
                plotted = True
            ax.set_xticks(x, BINS, fontsize=7)
            ax.set_title(name, fontsize=9)
            ax.set_ylabel(title, fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            if plotted:
                ax.legend(fontsize=6, frameon=False)
        fig.suptitle(f"oracle moment error by γ   {title}", fontsize=11)
        fig.tight_layout()
        fig.savefig(dest / f"{fname}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_pmf_grid(dest, sched):
    fig, axes = plt.subplots(len(NAMES), len(MODELS), figsize=(14.5, 11.5), squeeze=False)
    for i, name in enumerate(NAMES):
        val = ROOT / "data" / name / "val.npy"
        true_x = np.load(val) if val.is_file() else None
        for j, tag in enumerate(MODELS):
            ax = axes[i, j]
            folder = dest / "sample" / tag / name / sched
            gens = load_gens(folder, ("poisson",))
            if not gens:
                ax.set_axis_off()
                ax.set_title(f"{name} {tag} missing", fontsize=8)
                continue
            n_gen = max(len(v) for v in gens.values())
            if dist_kind(name) == "discrete":
                plot_discrete_ax(ax, name, gens, True, n_gen)
            else:
                plot_gamma_ax(ax, gens, True, n_gen, true_x)
            ax.set_title(f"{name}  {tag}", fontsize=8)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(f"Poisson  {sched}  log PMF / PDF", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = dest / f"pmf_grid_{sched}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def md_wd(metrics, sched):
    lines = [
        f"| dataset | learned mean | $k_{{\\max}}=1$ | $k_{{\\max}}=2$ | $k_{{\\max}}=3$ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        cells = [name]
        for tag in MODELS:
            r = get(metrics, dist=name, tag=tag, sched=sched)
            if r is None:
                cells.append("—")
            else:
                cells.append(f"{r['wd']:.4f} / {r['tv']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_mom(moments, k):
    header = "| dataset | model | [0,0.1] | [0.1,1] | [1,10] | [10,100] |"
    sep = "|---|---|---:|---:|---:|---:|"
    lines = [header, sep]
    models = MODELS if k == 1 else tuple(t for t in MODELS if t != "mean" and int(t[1:]) >= k)
    for name in NAMES:
        for tag in models:
            cells = [name, tag]
            any_hit = False
            for b in BINS:
                r = get(moments, dist=name, model=tag, k=k, bin=b)
                if r is None:
                    cells.append("—")
                else:
                    any_hit = True
                    cells.append(f"{r['mean']:.4f}")
            if any_hit:
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_n_bins(moments):
    # n is the same for all k of a (dist, model); use k=1 mean/k1
    lines = ["| dataset | [0,0.1] | [0.1,1] | [1,10] | [10,100] |", "|---|---:|---:|---:|---:|"]
    for name in NAMES:
        cells = [name]
        for b in BINS:
            r = get(moments, dist=name, model="mean", k=1, bin=b)
            cells.append(str(r["n"]) if r else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(metrics, moments):
    md = f"""# PMF-ratio  $k_{{\\max}}=1\\to 2\\to 3$  vs learned-mean

Poisson kernel only. T=100. Config otherwise unchanged (λ=100, z_rescale, 200 epoch, Adam 1e-3).

| tag | ckpt | what it predicts |
|---|---|---|
| learned mean | `experiments/scale/{{name}}/best.pt` | posterior mean MLP |
| $k_{{\\max}}=1$ | `experiments/pmfratio/kmax1/{{name}}/best.pt` | $b_1=\\log B_1$, $\\hat m_1=(z+1)e^{{b_1}}$ |
| $k_{{\\max}}=2$ | `experiments/pmfratio/kmax2/{{name}}/best.pt` | same + second head |
| $k_{{\\max}}=3$ | `experiments/pmfratio/{{name}}/best.pt` | three heads (previous run) |

Schedules:

- `unif_T100`: $h=1$, $\\gamma=0,1,\\ldots,100$
- `quad_T100`: $\\gamma_t=100(t/100)^2$

Oracle error (val, n=16384, $t\\sim U[10^{{-4}},1]$, $\\gamma=t\\lambda$):

$$
|\\log\\hat m_k-\\log m_k^*|,\\qquad
m_k^*(z,\\gamma)=\\mathbb{{E}}[X^k\\mid Z_\\gamma=z]
$$

γ bins: `[0,0.1]`, `[0.1,1]`, `[1,10]`, `[10,100]`. Low-γ bins have few points (table below).

Raw files: `metrics.txt`, `moments.txt`. Samples: `sample/{{mean,k1,k2,k3}}/{{name}}/{{unif,quad}}_T100/`.

---

## 1. WD / TV  (Poisson)

Each cell is **WD / TV**.

### uniform T=100

{md_wd(metrics, "unif_T100")}

### quadratic T=100

{md_wd(metrics, "quad_T100")}

WD panel is log-scale because `poissmix3` / $k_{{\\max}}=2$ / quad WD = 153.3 (that run also has overflow 0.619).

![WD/TV vs kmax](wd_tv_vs_kmax.png)

---

## 2. Oracle $|\\log\\hat m_k-\\log m_k^*|$ by γ

Val subsample sizes for the mean model (other models are the same order):

{md_n_bins(moments)}

### $m_1$

{md_mom(moments, 1)}

![m1 error](m1_err_by_gamma.png)

### $m_2$  (only $k_{{\\max}}\\ge 2$)

{md_mom(moments, 2)}

![m2 error](m2_err_by_gamma.png)

### $m_3$  (only $k_{{\\max}}=3$)

{md_mom(moments, 3)}

![m3 error](m3_err_by_gamma.png)

---

## 3. PMF / PDF  (Poisson, log)

### quadratic T=100

![quad grid](pmf_grid_quad_T100.png)

### uniform T=100

![unif grid](pmf_grid_unif_T100.png)

Per-run plots: `sample/<model>/<name>/<sched>/pmf.png`.

---

## 4. Extra mean-only datasets (not trained at k=1,2,3)

`run_12_kmax.sh` inherited `_common.sh` names, so learned-mean was also sampled on pois20 / poissmix / zip / yule_simon.

| dataset | unif WD / TV | quad WD / TV |
|---|---:|---:|
| pois20 | {fmt(get(metrics, dist="pois20", tag="mean", sched="unif_T100")["wd"])} / {fmt(get(metrics, dist="pois20", tag="mean", sched="unif_T100")["tv"])} | {fmt(get(metrics, dist="pois20", tag="mean", sched="quad_T100")["wd"])} / {fmt(get(metrics, dist="pois20", tag="mean", sched="quad_T100")["tv"])} |
| poissmix | {fmt(get(metrics, dist="poissmix", tag="mean", sched="unif_T100")["wd"])} / {fmt(get(metrics, dist="poissmix", tag="mean", sched="unif_T100")["tv"])} | {fmt(get(metrics, dist="poissmix", tag="mean", sched="quad_T100")["wd"])} / {fmt(get(metrics, dist="poissmix", tag="mean", sched="quad_T100")["tv"])} |
| zip | {fmt(get(metrics, dist="zip", tag="mean", sched="unif_T100")["wd"])} / {fmt(get(metrics, dist="zip", tag="mean", sched="unif_T100")["tv"])} | {fmt(get(metrics, dist="zip", tag="mean", sched="quad_T100")["wd"])} / {fmt(get(metrics, dist="zip", tag="mean", sched="quad_T100")["tv"])} |
| yule_simon | {fmt(get(metrics, dist="yule_simon", tag="mean", sched="unif_T100")["wd"])} / {fmt(get(metrics, dist="yule_simon", tag="mean", sched="unif_T100")["tv"])} | {fmt(get(metrics, dist="yule_simon", tag="mean", sched="quad_T100")["wd"])} / {fmt(get(metrics, dist="yule_simon", tag="mean", sched="quad_T100")["tv"])} |
"""
    (RES / "report.md").write_text(md)
    print("wrote", RES / "report.md")


def main():
    metrics = parse_metrics(RES / "metrics.txt")
    moments = parse_moments(RES / "moments.txt")
    plot_wd(metrics, RES)
    plot_mom(moments, RES)
    plot_pmf_grid(RES, "quad_T100")
    plot_pmf_grid(RES, "unif_T100")
    write_report(metrics, moments)


if __name__ == "__main__":
    main()
