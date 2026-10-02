"""Data-only PDF for the ratio-consistency stage."""
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

RES = HERE / "results" / "18_ratio_consistency"
BASE = HERE / "results" / "10_oracle_twopois" / "metrics.txt"
OUT = HERE / "results" / "18_report.pdf"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
TAGS = (
    "equal",
    "equal_proj",
    "soft",
    "anchored",
    "hard",
    "dyn_norm",
    "dyn_norm_proj",
)
KERNELS = ("poisson", "nb", "twopois")
TAG_LAB = {
    "equal": "equal",
    "equal_proj": "equal proj",
    "soft": "soft",
    "anchored": "anchored",
    "hard": "hard",
    "dyn_norm": "dyn_norm",
    "dyn_norm_proj": "dyn_norm proj",
}


def load_json(name):
    return json.loads((RES / name).read_text())


def finite(v):
    if v is None:
        return False
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return np.isfinite(x)


def num(v, nd=4):
    if not finite(v):
        return "—"
    return f"{float(v):.{nd}f}"


def compact(v):
    if not finite(v):
        return "—"
    x = float(v)
    ax = abs(x)
    if ax < 5e-5:
        return "0"
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 10:
        return f"{x:.2f}"
    if ax >= 0.01:
        return f"{x:.4f}"
    return f"{x:.1e}"


def pair(rec):
    if rec is None or not finite(rec.get("wd")) or not finite(rec.get("tv")):
        return "—"
    return f"{float(rec['wd']):.4f}/{float(rec['tv']):.4f}"


def load_mean_formula():
    rows = parse_rows(BASE)
    out = {}
    for name in NAMES:
        out[name] = find(rows, sched="quad", which="learned", dist=name, kernel="twopois")
    return out


def index_sample(rows):
    return {(r["dist"], r["tag"], r["kernel"]): r for r in rows}


def index_pair(rows):
    return {(r["dist"], r["tag"]): r for r in rows}


def cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_title(
        fig,
        "Ratio consistency",
        "Quadratic T=100, lambda=100, lambda_cons=1. Cell = WD/TV.",
    )
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
    ax.axis("off")
    text = (
        "Datasets: gamma_ltj, nb, poismix_mod, poissmix3. n=50000.\n"
        "New training keeps L_data = L1 + L2 + L3, equal weights.\n"
        "  soft:      L1+L2+L3 + (delta_2^2 + delta_3^2)\n"
        "  anchored:  L1+L2+L3 + stopgrad residual on a2, a3\n"
        "  hard:      L1 only. S2 and S3 are products of S1 at z, z+1, z+2\n\n"
        "equal and dyn_norm WD/TV are the experiment 17 samples.\n"
        "equal proj and dyn_norm proj are Euclidean projections, no extra training.\n\n"
        "TwoPois comparison row is the earlier learned-mean sampler:\n"
        "  m1 = m(z),  m2 = m(z) m(z+1),  m3 = m(z) m(z+1) m(z+2)\n"
        "  quadratic T=100, from test_exp/results/10_oracle_twopois\n"
        "  that row is the learned formula, not the oracle m1,m2,m3\n\n"
        "|delta| is the residual of the log-ratios used for moments.\n"
        "delta_in is that residual before projection.\n"
        "val fallback is the TwoPois mask on the val draw.\n"
        "sample fallback is the reverse-chain TwoPois fallback.\n"
        "overflow is the fraction outside the PMF window.\n"
        "runaway is the fraction of trajectories with non-finite z or z>1e6.\n\n"
        "Tables: test_exp/results/18_ratio_consistency/"
    )
    ax.text(0.0, 1.0, text, va="top", ha="left", fontsize=11, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def twopois_page(pdf, sample, base):
    headers = ["dataset", "mean formula"] + [TAG_LAB[t] for t in TAGS]
    cells = []
    for name in NAMES:
        row = [name, pair(base.get(name))]
        for tag in TAGS:
            row.append(pair(sample.get((name, tag, "twopois"))))
        cells.append(row)
    fig = plt.figure(figsize=(11, 8.5))
    page_title(
        fig,
        "TwoPois WD/TV",
        "mean formula = m(z) m(z+1) m(z+2), quadratic T=100. Not oracle.",
    )
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.74])
    ax.axis("off")
    tbl = ax.table(
        cellText=cells,
        colLabels=headers,
        loc="upper center",
        cellLoc="center",
        colWidths=[0.12, 0.13] + [0.107] * len(TAGS),
    )
    style_table(tbl, n_header_cols=1, fontsize=6.5)
    pdf.savefig(fig)
    plt.close(fig)


def tv_plot(pdf, sample, base):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=0.86, bottom=0.24, left=0.07, right=0.98)
    page_title(fig, "TwoPois TV", "quadratic T=100")
    x = np.arange(len(NAMES))
    series = [("mean formula", None)] + [(TAG_LAB[t], t) for t in TAGS]
    width = 0.10
    for i, (label, tag) in enumerate(series):
        ys = []
        for name in NAMES:
            if tag is None:
                rec = base.get(name)
            else:
                rec = sample.get((name, tag, "twopois"))
            ys.append(float(rec["tv"]) if rec and finite(rec.get("tv")) else np.nan)
        ax.bar(
            x + (i - 3.5) * width,
            ys,
            width,
            label=label,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x, NAMES)
    ax.set_ylabel("TV")
    ax.grid(True, axis="y", alpha=0.3)
    fig.legend(fontsize=8, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.03))
    pdf.savefig(fig)
    plt.close(fig)


def delta_page(pdf, cons):
    headers = ["dataset", "tag", "|d2| med", "p99", "max", "|d3| med", "p99", "max"]
    cells = []
    for name in NAMES:
        for tag in TAGS:
            rec = cons.get((name, tag), {})
            cells.append(
                [
                    name,
                    TAG_LAB[tag],
                    compact(rec.get("d2_median")),
                    compact(rec.get("d2_p99")),
                    compact(rec.get("d2_max")),
                    compact(rec.get("d3_median")),
                    compact(rec.get("d3_p99")),
                    compact(rec.get("d3_max")),
                ]
            )
    fig = plt.figure(figsize=(11, 8.5))
    page_title(
        fig,
        "|delta| of the log-ratios used for moments",
        "hard and projection residuals are numerical zeros. Val draw, t ~ Unif[t_eps, 1], seed 0.",
    )
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.84])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
    style_table(tbl, n_header_cols=2, fontsize=6.5)
    pdf.savefig(fig)
    plt.close(fig)


def coherence_page(pdf, coh, sample):
    headers = ["dataset", "tag", "v<=0", "invalid v", "invalid atom", "val fallback", "sample fallback"]
    cells = []
    for name in NAMES:
        for tag in TAGS:
            rec = coh.get((name, tag), {})
            samp = sample.get((name, tag, "twopois"))
            cells.append(
                [
                    name,
                    TAG_LAB[tag],
                    num(rec.get("frac_v_le_0")),
                    num(rec.get("frac_invalid_v")),
                    num(rec.get("frac_invalid_atom")),
                    num(rec.get("frac_fallback")),
                    num(None if samp is None else samp.get("fallback")),
                ]
            )
    fig = plt.figure(figsize=(11, 8.5))
    page_title(
        fig,
        "Moment coherence and TwoPois fallback",
        "val columns are the shared val draw. sample fallback is the reverse chain.",
    )
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.84])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
    style_table(tbl, n_header_cols=2, fontsize=6.5)
    pdf.savefig(fig)
    plt.close(fig)


def moment_page(pdf, moments):
    headers = ["dataset", "tag", "|log m1| med", "p99", "|log m2| med", "p99", "|log m3| med", "p99"]
    cells = []
    for name in NAMES:
        for tag in TAGS:
            row = [name, TAG_LAB[tag]]
            for k in (1, 2, 3):
                rec = moments.get((name, tag, k, "all"), {})
                row.append(compact(rec.get("abs_median")))
                row.append(compact(rec.get("abs_p99")))
            cells.append(row)
    fig = plt.figure(figsize=(11, 8.5))
    page_title(
        fig,
        "|log(m_hat / m*)|",
        "all gamma, shared val draw. Signed log(m_hat/m*) median is in moment_errors.txt.",
    )
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.84])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
    style_table(tbl, n_header_cols=2, fontsize=6.5)
    pdf.savefig(fig)
    plt.close(fig)


def dataset_pages(pdf, sample, base):
    headers = ["tag", "Poisson", "NB", "TwoPois", "overflow p/nb/two", "runaway p/nb/two", "TwoPois fallback"]
    for name in NAMES:
        cells = [["mean formula", "—", "—", pair(base.get(name)), "—", "—", "—"]]
        for tag in TAGS:
            bits = [TAG_LAB[tag]]
            ovs, rws = [], []
            fb = "—"
            for kernel in KERNELS:
                rec = sample.get((name, tag, kernel))
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
        ax = fig.add_axes([0.03, 0.16, 0.94, 0.72])
        ax.axis("off")
        tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
        style_table(tbl, n_header_cols=1, fontsize=7.5)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    sample = index_sample(load_json("sampling_metrics.json"))
    cons = index_pair(load_json("consistency.json"))
    coh = index_pair(load_json("coherence.json"))
    moments = {
        (r["dist"], r["tag"], int(r["k"]), r["bin"]): r
        for r in load_json("moment_errors.json")
    }
    base = load_mean_formula()
    missing = [n for n in NAMES if base.get(n) is None]
    if missing:
        raise SystemExit(f"missing mean-formula rows: {missing}")
    with PdfPages(OUT) as pdf:
        cover(pdf)
        twopois_page(pdf, sample, base)
        tv_plot(pdf, sample, base)
        delta_page(pdf, cons)
        coherence_page(pdf, coh, sample)
        moment_page(pdf, moments)
        dataset_pages(pdf, sample, base)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
