"""Data-only PDF: exponent sweep and oracle kernels."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from write_pdf import find, page_title, parse_rows, style_table  # noqa: E402

RES = HERE / "results"
OUT = RES / "20_report.pdf"
NAMES = (
    "gamma_ltj",
    "nb",
    "poismix_mod",
    "pois20",
    "poissmix",
    "poissmix3",
    "zip",
    "yule_simon",
)
KERNELS = ("poisson", "nb", "twopois")
POWERS = (1.0, 1.5, 2.0, 3.0, 4.0)
KER_LAB = {"poisson": "Poisson", "nb": "NB", "twopois": "TwoPois"}
COLORS = {"poisson": "#ff7f0e", "nb": "#2ca02c", "twopois": "#d62728"}


def finite(v):
    if v is None:
        return False
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return np.isfinite(x)


def pair(rec):
    if rec is None or not finite(rec.get("wd")) or not finite(rec.get("tv")):
        return "—"
    return f"{float(rec['wd']):.4f}/{float(rec['tv']):.4f}"


def index_exp(rows):
    return {(r["dist"], r["kernel"], float(r["power"])): r for r in rows}


def index_oracle(rows):
    return {(r["sched"], r["which"], r["dist"], r["kernel"]): r for r in rows}


def cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_title(
        fig,
        "Exponent and oracle kernels",
        "T=100, lambda=100, n=50000. Cell = WD/TV. Frozen experiments/scale.",
    )
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
    ax.axis("off")
    text = (
        "Schedule: gamma_t = 100 * (t/100)^p, t = 0..100.\n"
        "  p=1 is uniform gamma. p=2 is quadratic.\n\n"
        "Exponent pages use test_exp/results/02_exponent.\n"
        "  learned mean m(z), then Poisson / NB / TwoPois.\n"
        "  NB variance uses m(z) and m(z+1).\n"
        "  TwoPois uses m(z), m(z)m(z+1), m(z)m(z+1)m(z+2).\n\n"
        "Oracle pages pair learned and oracle from the same run:\n"
        "  Poisson:  04_oracle_mean.  m = E[X|z]\n"
        "  NB:       12_oracle_nb.     m1 = E[X|z], m2 = E[X^2|z], v = m2-m1^2\n"
        "  TwoPois:  10_oracle_twopois. true E[X^k|z], k=1,2,3\n"
        "Those learned rows are not copied from 02_exponent.\n\n"
        "WD = mean |sorted a-b|. TV = half L1 on the K_0.999 window.\n"
        "Tables: 02_exponent, 04_oracle_mean, 10_oracle_twopois, 12_oracle_nb."
    )
    ax.text(0.0, 1.0, text, va="top", ha="left", fontsize=11, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def exponent_page(pdf, exp, kernel):
    headers = ["dataset"] + [f"p={p:g}" for p in POWERS]
    cells = []
    for name in NAMES:
        row = [name]
        for p in POWERS:
            row.append(pair(exp.get((name, kernel, p))))
        cells.append(row)
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, f"{KER_LAB[kernel]} WD/TV vs exponent", "gamma_t = 100 (t/100)^p. Learned mean.")
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.78])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
    style_table(tbl, n_header_cols=1, fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def tv_plot(pdf, exp):
    fig, axes = plt.subplots(2, 4, figsize=(11, 8.5))
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.06, right=0.98, hspace=0.45, wspace=0.35)
    page_title(fig, "TV vs exponent", "learned mean. gamma_t = 100 (t/100)^p.")
    for ax, name in zip(axes.ravel(), NAMES):
        for kernel in KERNELS:
            xs, ys = [], []
            for p in POWERS:
                rec = exp.get((name, kernel, p))
                if rec is None or not finite(rec.get("tv")):
                    continue
                xs.append(p)
                ys.append(float(rec["tv"]))
            ax.plot(xs, ys, marker="o", color=COLORS[kernel], label=KER_LAB[kernel])
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("p")
        ax.set_ylabel("TV")
        ax.grid(True, alpha=0.3)
    axes.ravel()[0].legend(fontsize=7, frameon=False)
    pdf.savefig(fig)
    plt.close(fig)


def oracle_nb_page(pdf, nb):
    headers = ["dataset", "unif learned", "unif oracle", "quad learned", "quad oracle"]
    cells = []
    for name in NAMES:
        cells.append(
            [
                name,
                pair(nb.get(("uniform", "learned", name, "nb"))),
                pair(nb.get(("uniform", "oracle", name, "nb"))),
                pair(nb.get(("quad", "learned", name, "nb"))),
                pair(nb.get(("quad", "oracle", name, "nb"))),
            ]
        )
    fig = plt.figure(figsize=(11, 8.5))
    page_title(
        fig,
        "NB learned vs oracle",
        "uniform p=1 and quadratic p=2. Oracle uses E[X|z] and E[X^2|z].",
    )
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.78])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
    style_table(tbl, n_header_cols=1, fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def oracle_sched_page(pdf, sched, pois, nb, two):
    headers = [
        "dataset",
        "Pois learned",
        "Pois oracle",
        "NB learned",
        "NB oracle",
        "Two learned",
        "Two oracle",
    ]
    src = {"poisson": pois, "nb": nb, "twopois": two}
    cells = []
    for name in NAMES:
        row = [name]
        for kernel in KERNELS:
            row.append(pair(src[kernel].get((sched, "learned", name, kernel))))
            row.append(pair(src[kernel].get((sched, "oracle", name, kernel))))
        cells.append(row)
    title = "Uniform gamma, learned vs oracle" if sched == "uniform" else "Quadratic, learned vs oracle"
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, title, "T=100. Poisson, NB, and TwoPois. Cell = WD/TV.")
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.78])
    ax.axis("off")
    tbl = ax.table(
        cellText=cells,
        colLabels=headers,
        loc="upper center",
        cellLoc="center",
        colWidths=[0.14, 0.143, 0.143, 0.143, 0.143, 0.144, 0.144],
    )
    style_table(tbl, n_header_cols=1, fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    exp = index_exp(parse_rows(RES / "02_exponent" / "metrics.txt"))
    pois = index_oracle(parse_rows(RES / "04_oracle_mean" / "metrics.txt"))
    two = index_oracle(parse_rows(RES / "10_oracle_twopois" / "metrics.txt"))
    nb = index_oracle(parse_rows(RES / "12_oracle_nb" / "metrics.txt"))
    missing = []
    for name in NAMES:
        for kernel in KERNELS:
            for p in POWERS:
                if (name, kernel, p) not in exp:
                    missing.append(f"exp {name} {kernel} p={p:g}")
        for sched in ("uniform", "quad"):
            for which in ("learned", "oracle"):
                if (sched, which, name, "poisson") not in pois:
                    missing.append(f"pois {sched} {which} {name}")
                if (sched, which, name, "nb") not in nb:
                    missing.append(f"nb {sched} {which} {name}")
                if (sched, which, name, "twopois") not in two:
                    missing.append(f"two {sched} {which} {name}")
    if missing:
        raise SystemExit("missing rows:\n" + "\n".join(missing[:20]))
    with PdfPages(OUT) as pdf:
        cover(pdf)
        for kernel in KERNELS:
            exponent_page(pdf, exp, kernel)
        tv_plot(pdf, exp)
        oracle_nb_page(pdf, nb)
        oracle_sched_page(pdf, "uniform", pois, nb, two)
        oracle_sched_page(pdf, "quad", pois, nb, two)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
