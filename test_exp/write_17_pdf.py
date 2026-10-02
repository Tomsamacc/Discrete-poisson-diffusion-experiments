"""Data-only PDF for raw multi-output sampling, with TwoPois TV vs the mean-formula baseline."""
from __future__ import annotations

import json
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

RES = HERE / "results" / "17_raw_multi_sampling"
BASE = HERE / "results" / "10_oracle_twopois" / "metrics.txt"
OUT = HERE / "results" / "17_report.pdf"
NAMES = (
    "gamma_ltj",
    "nb",
    "poismix_mod",
    "poissmix3",
    "pois20",
    "poissmix",
    "zip",
    "yule_simon",
)
WEIGHTS = ("equal", "dyn", "dyn_norm")
KERNELS = ("poisson", "nb", "twopois")


def load_new():
    rows = json.loads((RES / "sampling_metrics.json").read_text())
    return {(r["dist"], r["weight"], r["kernel"]): r for r in rows}


def load_mean_formula():
    rows = parse_rows(BASE)
    out = {}
    for name in NAMES:
        rec = find(rows, sched="quad", which="learned", dist=name, kernel="twopois")
        out[name] = rec
    return out


def pair(rec):
    if rec is None:
        return "—"
    wd, tv = rec.get("wd"), rec.get("tv")
    if wd is None or tv is None:
        return "—"
    return f"{float(wd):.4f}/{float(tv):.4f}"


def num(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{float(v):.4f}"


def cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_title(
        fig,
        "Raw multi-output sampling",
        "Quadratic T=100, lambda=100. Cell = WD/TV.",
    )
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
    ax.axis("off")
    text = (
        "8 datasets x equal / dyn / dyn_norm x Poisson / NB / TwoPois.\n"
        "S_k = exp(a_k),  m_k = D_k(z) * S_k / gamma^k.\n"
        "gamma=0 uses the prior moments. n=50000.\n\n"
        "TwoPois comparison row is the earlier learned-mean sampler:\n"
        "  one posterior-mean network\n"
        "  m1 = m(z),  m2 = m(z) m(z+1),  m3 = m(z) m(z+1) m(z+2)\n"
        "  quadratic T=100, from test_exp/results/10_oracle_twopois\n"
        "  that row is the learned formula, not the oracle m1,m2,m3\n\n"
        "overflow is the fraction outside the PMF window.\n"
        "fallback is the TwoPois fraction replaced by Pois(h m1).\n"
        "runaway is the fraction of trajectories with non-finite z or z>1e6.\n\n"
        "Tables: test_exp/results/17_raw_multi_sampling/"
    )
    ax.text(0.0, 1.0, text, va="top", ha="left", fontsize=11, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def compare_page(pdf, new, base):
    headers = ["dataset", "mean formula", "equal", "dyn", "dyn_norm"]
    cells = []
    for name in NAMES:
        row = [name, pair(base.get(name))]
        for weight in WEIGHTS:
            row.append(pair(new.get((name, weight, "twopois"))))
        cells.append(row)
    fig = plt.figure(figsize=(11, 8.5))
    page_title(
        fig,
        "TwoPois WD/TV vs mean-formula baseline",
        "mean formula = m(z) m(z+1) m(z+2), quadratic T=100. Not oracle.",
    )
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.80])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
    style_table(tbl, n_header_cols=1, fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def tv_plot(pdf, new, base):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=0.88, bottom=0.28, left=0.08, right=0.98)
    page_title(fig, "TwoPois TV", "quadratic T=100")
    x = np.arange(len(NAMES))
    width = 0.18
    series = [("mean formula", None)] + [(w, w) for w in WEIGHTS]
    for i, (label, weight) in enumerate(series):
        ys = []
        for name in NAMES:
            if weight is None:
                rec = base.get(name)
            else:
                rec = new.get((name, weight, "twopois"))
            ys.append(float(rec["tv"]) if rec and rec.get("tv") is not None else np.nan)
        ax.bar(x + (i - 1.5) * width, ys, width, label=label)
    ax.set_xticks(x, NAMES, rotation=30, ha="right")
    ax.set_ylabel("TV")
    ax.grid(True, axis="y", alpha=0.3)
    fig.legend(fontsize=8, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.04))
    pdf.savefig(fig)
    plt.close(fig)


def dataset_pages(pdf, new, base):
    headers = ["weight", "Poisson", "NB", "TwoPois", "overflow p/nb/two", "runaway p/nb/two", "TwoPois fallback"]
    for name in NAMES:
        cells = []
        base_rec = base.get(name)
        cells.append(["mean formula", "—", "—", pair(base_rec), "—", "—", "—"])
        for weight in WEIGHTS:
            bits = [weight]
            ovs, rws = [], []
            fb = "—"
            for kernel in KERNELS:
                rec = new.get((name, weight, kernel))
                bits.append(pair(rec))
                ovs.append(num(None if rec is None else rec.get("overflow")))
                rws.append(num(None if rec is None else rec.get("runaway")))
                if kernel == "twopois":
                    fb = num(None if rec is None else rec.get("fallback"))
            bits.append(" / ".join(ovs))
            bits.append(" / ".join(rws))
            bits.append(fb)
            cells.append(bits)
        fig = plt.figure(figsize=(11, 8.5))
        page_title(fig, name, "cell = WD/TV. mean formula is TwoPois only.")
        ax = fig.add_axes([0.03, 0.18, 0.94, 0.70])
        ax.axis("off")
        tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
        style_table(tbl, n_header_cols=1, fontsize=7.5)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    new = load_new()
    base = load_mean_formula()
    missing = [n for n in NAMES if base.get(n) is None]
    if missing:
        raise SystemExit(f"missing mean-formula rows: {missing}")
    with PdfPages(OUT) as pdf:
        cover(pdf)
        compare_page(pdf, new, base)
        tv_plot(pdf, new, base)
        dataset_pages(pdf, new, base)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
