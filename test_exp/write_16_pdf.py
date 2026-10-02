"""Data-only PDF for raw multi-output weights. No sampling."""
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
from write_direct_pdf import cell3, num  # noqa: E402
from write_pdf import page_title, style_table  # noqa: E402

RES = HERE / "results" / "16_raw_multi"
OUT = HERE / "results" / "16_report.pdf"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
WEIGHTS = ("equal", "dyn", "dyn_norm")
BINS = ("[0,0.1]", "[0.1,1]", "[1,10]", "[10,100]")
PAIRS = ("12", "13", "23")


def load(name):
    rows = json.loads((RES / name).read_text())
    for row in rows:
        if "k" in row:
            row["k"] = int(row["k"])
        if "head" in row:
            row["head"] = int(row["head"])
    return rows


def get(rows, **kw):
    for row in rows:
        if all(row.get(k) == v for k, v in kw.items()):
            return row
    return None


def cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, "Raw multi-output, three loss weights", "No sampling. No TV or WD.")
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
    ax.axis("off")
    text = (
        "One shared network, outputs (a1, a2, a3), S_k = exp(a_k).\n"
        "L_k = D(R_k, S_k), full Bregman. k=1,2,3.\n"
        "lambda=100, z_rescale, 200 epoch, Adam 1e-3, batch 256, t~U[1e-4,1].\n\n"
        "Weights, per sample. h is the quadratic T=100 reverse step at gamma=t*lambda.\n"
        "  equal      1, 1, 1\n"
        "  dyn        h, h^2/2, h^3/6\n"
        "  dyn_norm   those divided by W = h + h^2/2 + h^3/6\n\n"
        "Trunk gradients are the weighted head terms that enter the total loss.\n"
        "The last linear layer is not in the trunk.\n"
        "Cosine is cos(g_i, g_j) of those trunk gradients.\n\n"
        "Error cells are median / p99 / max.\n"
        "  eS = log(S_hat / S*)\n"
        "  em = log(m_hat / m*)\n"
        "  |m| = |m_hat - m*|\n"
        "Gamma bins [0,0.1], [0.1,1], [1,10], [10,100].\n\n"
        "Ckpts: experiments/raw_multi/{equal,dyn,dyn_norm}/{name}/best.pt\n"
        "Tables: test_exp/results/16_raw_multi/"
    )
    ax.text(0.0, 1.0, text, va="top", ha="left", fontsize=11, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def grad_page(pdf, grads):
    headers = ["weight", "dataset", "k=1", "k=2", "k=3"]
    cells = []
    for weight in WEIGHTS:
        for name in NAMES:
            row = [weight, name]
            for head in (1, 2, 3):
                rec = get(grads, weight=weight, dist=name, head=head)
                row.append(cell3(rec, "median", "p99", "max"))
            cells.append(row)
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, "Trunk gradient norm", "cell = p50 / p99 / max. p50 is the median over training steps.")
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.84])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
    style_table(tbl, n_header_cols=2, fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def grad_plot(pdf, grads):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.98, hspace=0.28, wspace=0.18)
    page_title(fig, "Trunk gradient p50 by head", "weighted head term")
    x = np.arange(3)
    for ax, name in zip(axes.ravel(), NAMES):
        for weight in WEIGHTS:
            ys = []
            for head in (1, 2, 3):
                rec = get(grads, weight=weight, dist=name, head=head)
                ys.append(rec["median"] if rec else np.nan)
            ax.plot(x, ys, marker="o", label=weight)
        ax.set_xticks(x, ["k=1", "k=2", "k=3"])
        ax.set_yscale("log")
        ax.set_title(name, loc="left", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=8, frameon=False)
    pdf.savefig(fig)
    plt.close(fig)


def cos_page(pdf, cos):
    headers = ["weight", "dataset", "cos 12", "cos 13", "cos 23"]
    cells = []
    for weight in WEIGHTS:
        for name in NAMES:
            row = [weight, name]
            for pair in PAIRS:
                rec = get(cos, weight=weight, dist=name, pair=pair)
                row.append(cell3(rec, "median", "p10", "p90"))
            cells.append(row)
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, "Trunk gradient cosine", "cell = median / p10 / p90")
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.84])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
    style_table(tbl, n_header_cols=2, fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def error_pages(pdf, ratio, moment):
    blocks = (
        ("log(S_hat / S*)", ratio, "median", "p99", "max"),
        ("log(m_hat / m*)", moment, "log_median", "log_p99", "log_max"),
        ("|m_hat - m*|", moment, "abs_median", "abs_p99", "abs_max"),
    )
    headers = ["weight", "k", "[0,0.1]", "[0.1,1]", "[1,10]", "[10,100]"]
    for name in NAMES:
        counts = []
        for b in BINS:
            rec = get(ratio, weight="equal", dist=name, k=1, bin=b)
            counts.append(str(int(rec["n"])) if rec else "—")
        fig = plt.figure(figsize=(11, 8.5))
        page_title(fig, name, "cell = median / p99 / max.  equal k=1 n: " + ", ".join(counts))
        for i, (title, rows, a, b, c) in enumerate(blocks):
            cells = []
            for weight in WEIGHTS:
                for k in (1, 2, 3):
                    cells.append(
                        [weight, str(k)]
                        + [
                            cell3(get(rows, weight=weight, dist=name, k=k, bin=lab), a, b, c)
                            for lab in BINS
                        ]
                    )
            ax = fig.add_axes([0.025, 0.655 - i * 0.305, 0.95, 0.28])
            ax.axis("off")
            ax.set_title(title, loc="left", fontsize=10, pad=1)
            tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
            style_table(tbl, n_header_cols=2, fontsize=6.5)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    grads = load("grads.json")
    cos = load("grad_cosine.json")
    ratio = load("ratio_errors.json")
    moment = load("moment_errors.json")
    with PdfPages(OUT) as pdf:
        cover(pdf)
        grad_page(pdf, grads)
        grad_plot(pdf, grads)
        cos_page(pdf, cos)
        error_pages(pdf, ratio, moment)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
