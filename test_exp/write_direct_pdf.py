"""Data-only PDF for single-head moment PRL (experiment 14)."""
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
from write_pdf import page_title, style_table  # noqa: E402

RES = HERE / "results" / "14_direct"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
BINS = ("[0,0.1]", "[0.1,1]", "[1,10]", "[10,100]")
PARAMS = ("direct", "ratio", "raw")
SETUP = (
    "Single-head  ·  loss = m_hat - X^k log m_hat  ·  lambda=100  ·  z_rescale  ·  "
    "t~U[1e-4,1]  ·  val n=50000  ·  no sampling"
)


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
                row[k] = float(v) if "." in v or "e" in v.lower() else int(v)
            except ValueError:
                row[k] = v
        out.append(row)
    return out


def get(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def num(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    a = abs(float(v))
    if a >= 1000 or (a > 0 and a < 0.001):
        return f"{float(v):.3g}"
    return f"{float(v):.4f}"


def cell3(r, a, b, c):
    if r is None:
        return "—"
    return f"{num(r[a])} / {num(r[b])} / {num(r[c])}"


def table_page(pdf, title, sub, headers, cells, fontsize=7, n_header_cols=2):
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, title, sub)
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.86])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
    style_table(tbl, n_header_cols=n_header_cols, fontsize=fontsize)
    pdf.savefig(fig)
    plt.close(fig)


def cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, "Single-head moment PRL", SETUP)
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
    ax.axis("off")
    text = (
        "Parameterization\n"
        "  direct    m_hat = softplus(f)\n"
        "  ratio     m_hat = (z+1)^up k * exp(b)\n"
        "  raw       m_hat = (z+1) * exp(f) / gamma     (k=1 only)\n\n"
        "Loss, all runs\n"
        "  loss = m_hat - X^k * log(m_hat)\n\n"
        "Logged on the validation forward states\n"
        "  e = log m_hat - log m*\n"
        "  |m_hat - m*|\n"
        "  gamma bins [0,0.1]  [0.1,1]  [1,10]  [10,100]\n"
        "  each cell below is median / p99 / max\n\n"
        "Gradient\n"
        "  ||grad loss|| over training minibatches\n"
        "  p50 / p90 / p99 / max\n\n"
        "Ckpts: experiments/direct_prl/{direct,ratio,raw}/k{k}/{name}/best.pt\n"
        "Tables: test_exp/results/14_direct/moments.txt , grads.txt"
    )
    ax.text(0.0, 1.0, text, va="top", ha="left", fontsize=11, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def grad_page(pdf, grads):
    headers = ["dataset", "param", "k", "p50", "p90", "p99", "max"]
    cells = []
    for r in grads:
        cells.append(
            [
                r["dist"],
                r["param"],
                str(int(r["k"])),
                num(r["median"]),
                num(r["p90"]),
                num(r["p99"]),
                num(r["max"]),
            ]
        )
    table_page(
        pdf,
        "Minibatch gradient norm  ||grad loss||",
        "Quantiles over all training steps. p50 is the median column in grads.txt.",
        headers,
        cells,
        fontsize=8,
        n_header_cols=3,
    )


def moment_block(moments, name, keys):
    headers = ["param", "k", "[0,0.1]", "[0.1,1]", "[1,10]", "[10,100]"]
    cells = []
    for k in (1, 2, 3):
        params = ("direct", "ratio", "raw") if k == 1 else ("direct", "ratio")
        for param in params:
            row = [param, str(k)]
            for b in BINS:
                r = get(moments, dist=name, param=param, k=k, bin=b)
                row.append(cell3(r, *keys))
            cells.append(row)
    return headers, cells


def moment_pages(pdf, moments):
    for name in NAMES:
        ns = []
        for b in BINS:
            r = get(moments, dist=name, param="direct", k=1, bin=b)
            ns.append(str(int(r["n"])) if r else "—")
        fig = plt.figure(figsize=(11, 8.5))
        page_title(
            fig,
            name,
            "cell = median / p99 / max.  k=1 counts, left to right: " + ", ".join(ns),
        )
        blocks = (
            (0.50, "e = log m_hat - log m*", ("log_median", "log_p99", "log_max")),
            (0.06, "|m_hat - m*|", ("abs_median", "abs_p99", "abs_max")),
        )
        for y, title, keys in blocks:
            headers, cells = moment_block(moments, name, keys)
            ax = fig.add_axes([0.03, y, 0.94, 0.38])
            ax.axis("off")
            ax.set_title(title, loc="left", fontsize=10, pad=2)
            tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
            style_table(tbl, n_header_cols=2, fontsize=7)
        pdf.savefig(fig)
        plt.close(fig)


def grad_plot(pdf, grads):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.08, right=0.97)
    page_title(fig, "Gradient p50 by k", "direct vs ratio. raw is k=1 only and is omitted from the lines.")
    x = np.arange(3)
    for i, name in enumerate(NAMES):
        for param, ls in (("direct", "-"), ("ratio", "--")):
            ys = []
            for k in (1, 2, 3):
                r = get(grads, dist=name, param=param, k=k)
                ys.append(r["median"] if r else np.nan)
            ax.plot(x, ys, ls=ls, marker="o", label=f"{name} {param}")
    ax.set_xticks(x, ["k=1", "k=2", "k=3"])
    ax.set_yscale("log")
    ax.set_ylabel("p50  ||grad loss||")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    moments = rows_of(RES / "moments.txt")
    grads = json.loads((RES / "grads.json").read_text())
    for g in grads:
        g["k"] = int(g["k"])
    out = RES / "report.pdf"
    with PdfPages(out) as pdf:
        cover(pdf)
        grad_page(pdf, grads)
        grad_plot(pdf, grads)
        moment_pages(pdf, moments)
    print("wrote", out)


if __name__ == "__main__":
    main()
