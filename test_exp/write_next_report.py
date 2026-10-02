"""Write test_exp/results/13_next/report.md from metrics, tails, schedule, samples."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from test import dist_kind, load_gens, plot_discrete_ax, plot_gamma_ax  # noqa: E402

RES = HERE / "results" / "13_next"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
TAGS = ("A_mean", "B_norm_k1", "C_bal_k1", "D_raw_k1", "E_bal_k2", "F_bal_k3")
TAG_LAB = {
    "A_mean": "A mean",
    "B_norm_k1": "B norm k1",
    "C_bal_k1": "C bal k1",
    "D_raw_k1": "D raw k1",
    "E_bal_k2": "E bal k2",
    "F_bal_k3": "F bal k3",
    "G_oracle": "G oracle",
}
BINS = ("[0,0.1]", "[0.1,1]", "[1,10]", "[10,100]")
EXTRA = ("pois20", "poissmix", "zip", "yule_simon")


def rows_of(path):
    lines = Path(path).read_text().splitlines()
    keys = lines[0].split()
    out = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < len(keys):
            continue
        row = {}
        for k, v in zip(keys, parts):
            try:
                row[k] = int(v) if v.isdigit() or (v.startswith("-") and v[1:].isdigit()) else float(v)
            except ValueError:
                row[k] = v
        out.append(row)
    return out


def get(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def cell(r, a="wd", b="tv"):
    if r is None:
        return "—"
    return f"{r[a]:.4f} / {r[b]:.4f}"


def f4(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.4f}"


def gamma_bin(g):
    g = float(g)
    if g < 0.1:
        return "[0,0.1]"
    if g < 1.0:
        return "[0.1,1]"
    if g < 10.0:
        return "[1,10]"
    return "[10,100]"


def plot_wd(metrics, dest):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    x = np.arange(len(TAGS))
    for ax, key, title in zip(axes, ("wd", "tv"), ("WD", "TV")):
        for name in NAMES:
            ys = []
            for tag in TAGS:
                r = get(metrics, dist=name, tag=tag, kernel="poisson")
                ys.append(r[key] if r else np.nan)
            ax.plot(x, ys, marker="o", lw=1.4, ms=5, label=name)
        ax.set_xticks(x, [TAG_LAB[t] for t in TAGS], fontsize=7, rotation=20)
        ax.set_ylabel(title)
        ax.set_title(f"Poisson quadratic T=100  {title}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(dest / "wd_tv_poisson.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_grid(dest, tags, kernels, fname, title):
    nrows, ncols = len(NAMES), len(tags) if len(kernels) == 1 else len(kernels)
    # if one kernel, columns are tags; if many kernels, one tag (oracle) and columns are kernels
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.6 * nrows), squeeze=False)
    for i, name in enumerate(NAMES):
        val = ROOT / "data" / name / "val.npy"
        true_x = np.load(val) if val.is_file() else None
        for j in range(ncols):
            ax = axes[i, j]
            if len(kernels) == 1:
                tag = tags[j]
                folder = dest / "sample" / tag / name / "quad_T100"
                gens = load_gens(folder, kernels)
                lab = TAG_LAB.get(tag, tag)
            else:
                tag = tags[0]
                folder = dest / "sample" / tag / name / "quad_T100"
                gens = load_gens(folder, (kernels[j],))
                lab = kernels[j]
            if not gens:
                ax.set_axis_off()
                ax.set_title(f"{name} missing", fontsize=8)
                continue
            n_gen = max(len(v) for v in gens.values())
            if dist_kind(name) == "discrete":
                plot_discrete_ax(ax, name, gens, True, n_gen)
            else:
                plot_gamma_ax(ax, gens, True, n_gen, true_x)
            ax.set_title(f"{name}  {lab}", fontsize=8)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(dest / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def md_poisson(metrics):
    head = "| dataset | " + " | ".join(TAG_LAB[t] for t in TAGS) + " |"
    sep = "|---|" + "|".join("---:" for _ in TAGS) + "|"
    lines = [head, sep]
    for name in NAMES:
        cells = [name]
        for tag in TAGS:
            r = get(metrics, dist=name, tag=tag, kernel="poisson")
            cells.append(cell(r))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_overflow(metrics):
    head = "| dataset | " + " | ".join(TAG_LAB[t] for t in TAGS) + " |"
    sep = "|---|" + "|".join("---:" for _ in TAGS) + "|"
    lines = [head, sep]
    for name in NAMES:
        cells = [name]
        for tag in TAGS:
            r = get(metrics, dist=name, tag=tag, kernel="poisson")
            cells.append(f4(None if r is None else r.get("overflow")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_oracle(metrics):
    lines = [
        "| dataset | poisson | nb | twopois |",
        "|---|---:|---:|---:|",
    ]
    names = list(NAMES) + list(EXTRA)
    for name in names:
        cells = [name]
        for k in ("poisson", "nb", "twopois"):
            r = get(metrics, dist=name, tag="G_oracle", kernel=k)
            cells.append(cell(r))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_forward_m1(tails):
    lines = [
        "| dataset | tag | [0,0.1] | [0.1,1] | [1,10] | [10,100] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        for tag in TAGS:
            cells = [name, TAG_LAB[tag]]
            any_hit = False
            for b in BINS:
                r = get(tails, dist=name, tag=tag, k=1, bin=b, where="forward")
                if r is None:
                    cells.append("—")
                else:
                    any_hit = True
                    cells.append(f"{r['median']:.4f} / {r['p99']:.4f}")
            if any_hit:
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_onpolicy(onp):
    lines = [
        "| dataset | tag | median | p90 | p99 | max | pos frac | runaway |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        for tag in TAGS:
            r = get(onp, dist=name, tag=tag)
            if r is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        TAG_LAB[tag],
                        f4(r["median"]),
                        f4(r["p90"]),
                        f4(r["p99"]),
                        f4(r["max"]),
                        f4(r["pos_frac"]),
                        f4(r.get("runaway_frac")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def md_schedule(sched):
    """Per γ-bin: median of the per-step median, and the worst p99 in that bin."""
    lines = [
        "| dataset | tag | [0,0.1] | [0.1,1] | [1,10] | [10,100] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        for tag in TAGS:
            cells = [name, TAG_LAB[tag]]
            sub = [r for r in sched if r["dist"] == name and r["tag"] == tag]
            if not sub:
                continue
            for b in BINS:
                hit = [r for r in sub if gamma_bin(r["gamma"]) == b]
                if not hit:
                    cells.append("—")
                    continue
                med = float(np.median([r["median"] for r in hit]))
                p99 = float(np.max([r["p99"] for r in hit]))
                cells.append(f"{med:.4f} / {p99:.4f}")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_extra(metrics):
    lines = [
        "| dataset | A mean WD / TV | G poisson | G nb | G twopois |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in EXTRA:
        a = get(metrics, dist=name, tag="A_mean", kernel="poisson")
        cells = [name, cell(a)]
        for k in ("poisson", "nb", "twopois"):
            cells.append(cell(get(metrics, dist=name, tag="G_oracle", kernel=k)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(metrics, tails, onp, sched):
    md = f"""# PMF-ratio next controls  A–G

Quadratic T=100. A–F are Poisson only. G replaces the network with analytic $m_k^*$ inside the current sampler (Poisson / NB / Two-Poisson).

| tag | method | ckpt |
|---|---|---|
| A | learned posterior mean | `experiments/scale/{{name}}/best.pt` |
| B | normalized ratio $k=1$, $\\ell=e^b-Tb$ | `experiments/pmfratio/kmax1/{{name}}/best.pt` |
| C | balanced $k=1$, $\\ell=(z+1)(e^b-Tb)$ | `experiments/pmfratio/balanced_k1/{{name}}/best.pt` |
| D | raw ratio $k=1$, target $R=(\\gamma X)/(z+1)$ | `experiments/pmfratio/raw_k1/{{name}}/best.pt` |
| E | balanced $k=1,2$ | `experiments/pmfratio/balanced_k2/{{name}}/best.pt` |
| F | balanced $k=1,2,3$ | `experiments/pmfratio/balanced_k3/{{name}}/best.pt` |
| G | oracle moments, no network | — |

Cells in the WD/TV tables are **WD / TV**. Signed error is $e=\\log\\hat m_1-\\log m_1^*$. Forward cells are **median / p99**.

Raw: `metrics.txt`, `signed_tails.txt`, `onpolicy.txt`, `schedule_val.txt`.

---

## 1. Poisson quadratic  WD / TV

{md_poisson(metrics)}

Overflow (mass outside the plot window):

{md_overflow(metrics)}

![WD/TV](wd_tv_poisson.png)

![Poisson PMF](pmf_grid_poisson.png)

---

## 2. Oracle sampler  G

Same quadratic grid, moments from the known PMF, current `generate` path.

{md_oracle(metrics)}

![oracle PMF](pmf_grid_oracle.png)

---

## 3. Signed $e_1$ on forward val states

$t\\sim U[10^{{-4}},1]$, n=16384. Low-$\\gamma$ bins still have few points.

{md_forward_m1(tails)}

---

## 4. On-policy Poisson rollout

1024 trajectories, quadratic T=100. Stats pool every step. `runaway` is the fraction of trajectories that hit non-finite $z$ or $z>10^6$.

{md_onpolicy(onp)}

---

## 5. Schedule-aware forward error

At each quadratic $\\gamma_t$, 256 forward $Z\\sim\\mathrm{{Pois}}(\\gamma X)$. Cell is the **median of per-step medians / worst p99** inside that $\\gamma$ bin.

{md_schedule(sched)}

---

## 6. Extra datasets

`_common.sh` also sampled learned-mean and the oracle on pois20 / poissmix / zip / yule_simon. Those four were not trained at B–F.

{md_extra(metrics)}
"""
    (RES / "report.md").write_text(md)
    print("wrote", RES / "report.md")


def main():
    metrics = rows_of(RES / "metrics.txt")
    tails = rows_of(RES / "signed_tails.txt")
    onp = rows_of(RES / "onpolicy.txt")
    sched = json.loads((RES / "schedule_val.json").read_text())
    plot_wd(metrics, RES)
    plot_grid(
        RES,
        TAGS,
        ("poisson",),
        "pmf_grid_poisson.png",
        "Poisson quadratic T=100  log PMF / PDF",
    )
    plot_grid(
        RES,
        ("G_oracle",),
        ("poisson", "nb", "twopois"),
        "pmf_grid_oracle.png",
        "Oracle moments  quadratic T=100  log PMF / PDF",
    )
    write_report(metrics, tails, onp, sched)


if __name__ == "__main__":
    main()
