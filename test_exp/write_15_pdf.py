"""Data-only PDF for raw-ratio training (I) and the oracle dynamics diagnostic (J)."""
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

RAW = HERE / "results" / "15_raw"
DIAG = HERE / "results" / "15_diag"
OUT = HERE / "results" / "15_report.pdf"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
BINS = ("[0,0.1]", "[0.1,1]", "[1,10]", "[10,100]")
STEPS = (0, 1, 2, 5, 10, 50, 99)
Q_KEYS = (("q1", "q1"), ("q2", "q2"), ("q3", "q3"), ("tail_gt3", "P(K>3)"))


def load(path):
    return json.loads(Path(path).read_text())


def get(rows, **kw):
    for row in rows:
        if all(row.get(k) == v for k, v in kw.items()):
            return row
    return None


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
    page_title(fig, "Raw ratio and oracle jump diagnostic", "I and J. Data only. No sampling.")
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.84])
    ax.axis("off")
    text = (
        "I. Single-head raw PMF ratio\n"
        "  S_hat = exp(f)\n"
        "  R_k = (gamma * X)^k / (z+1)^up k\n"
        "  loss = D(R_k, S_hat) = R log(R/S) - R + S\n"
        "  lambda=100, z_rescale, 200 epoch, Adam 1e-3, batch 256, t~U[1e-4,1]\n"
        "  k=1,2,3 trained separately\n"
        "  m_hat = (z+1)^up k * S_hat / gamma^k\n"
        "  S* = m* * gamma^k / (z+1)^up k,   m* = E[X^k | Z=z, gamma]\n"
        "  cells below are p50 / p99 / max\n\n"
        "J. Oracle diagnostic, no training. n=8192 trajectories\n"
        "  quad: gamma = 100 * (step/100)^2,   unif: h=1, gamma = 0,1,...,100\n"
        "  forward: X from val, Z ~ Pois(gamma X)\n"
        "  onpolicy: exact reverse, draw X from the posterior, then K ~ Pois(h X)\n"
        "  eta_k = h^k * m_k* / k!\n"
        "  q(k) = E[ Pois(k; h X) | z ],   tail = P(K>3 | z)\n"
        "  plotted q and tail are means over the 8192 states\n\n"
        "Ckpts: experiments/raw_ratio/k{k}/{name}/best.pt\n"
        "Tables: test_exp/results/15_raw/  and  test_exp/results/15_diag/"
    )
    ax.text(0.0, 1.0, text, va="top", ha="left", fontsize=11, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def grad_page(pdf, grads):
    headers = ["dataset", "k", "p50", "p90", "p99", "max"]
    cells = []
    for name in NAMES:
        for k in (1, 2, 3):
            row = get(grads, dist=name, k=k)
            cells.append(
                [
                    name,
                    str(k),
                    num(row["median"]),
                    num(row["p90"]),
                    num(row["p99"]),
                    num(row["max"]),
                ]
            )
    table_page(
        pdf,
        "Minibatch gradient norm  ||grad loss||",
        "Quantiles over all training steps. p50 is the median.",
        headers,
        cells,
        fontsize=8,
        n_header_cols=2,
    )


def grad_plot(pdf, grads):
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.98, wspace=0.25)
    page_title(fig, "Gradient norm by k", "raw ratio, single head")
    x = np.arange(3)
    for ax, key, label in ((axes[0], "median", "p50"), (axes[1], "p99", "p99")):
        for name in NAMES:
            ys = [get(grads, dist=name, k=k)[key] for k in (1, 2, 3)]
            ax.plot(x, ys, marker="o", label=name)
        ax.set_xticks(x, ["k=1", "k=2", "k=3"])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    pdf.savefig(fig)
    plt.close(fig)


def target_page(pdf, target):
    headers = ["dataset", "k", "[0,0.1]", "[0.1,1]", "[1,10]", "[10,100]"]
    cells = []
    ns = []
    for name in NAMES:
        counts = []
        for b in BINS:
            row = get(target, dist=name, k=1, bin=b)
            counts.append(str(int(row["n"])) if row else "—")
        ns.append(f"{name} {', '.join(counts)}")
        for k in (1, 2, 3):
            cells.append(
                [name, str(k)]
                + [cell3(get(target, dist=name, k=k, bin=b), "p50", "p99", "max") for b in BINS]
            )
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, "Target R_k scale", "cell = p50 / p99 / max. n is shared across k.")
    fig.text(0.055, 0.905, "n:  " + "    ".join(ns[:2]), fontsize=7, color="0.35", va="top")
    fig.text(0.055, 0.885, "n:  " + "    ".join(ns[2:]), fontsize=7, color="0.35", va="top")
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.76])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
    style_table(tbl, n_header_cols=2, fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def error_pages(pdf, errors):
    blocks = (
        ("log(S_hat / S*)", "ratio_p50", "ratio_p99", "ratio_max"),
        ("log(m_hat / m*)", "mom_p50", "mom_p99", "mom_max"),
        ("|m_hat - m*|", "abs_p50", "abs_p99", "abs_max"),
    )
    headers = ["k", "[0,0.1]", "[0.1,1]", "[1,10]", "[10,100]"]
    for name in NAMES:
        counts = []
        for b in BINS:
            row = get(errors, dist=name, k=1, bin=b)
            counts.append(str(int(row["ratio_n"])) if row else "—")
        fig = plt.figure(figsize=(11, 8.5))
        page_title(fig, name, "cell = p50 / p99 / max.  k=1 n: " + ", ".join(counts))
        for i, (title, a, b, c) in enumerate(blocks):
            cells = []
            for k in (1, 2, 3):
                cells.append(
                    [str(k)]
                    + [cell3(get(errors, dist=name, k=k, bin=lab), a, b, c) for lab in BINS]
                )
            ax = fig.add_axes([0.03, 0.64 - i * 0.29, 0.94, 0.26])
            ax.axis("off")
            ax.set_title(title, loc="left", fontsize=10, pad=2)
            tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
            style_table(tbl, n_header_cols=1, fontsize=7.5)
        pdf.savefig(fig)
        plt.close(fig)


def series(rows, name, sched, source):
    sub = [
        r
        for r in rows
        if r["dist"] == name and r["sched"] == sched and r["source"] == source
    ]
    return sorted(sub, key=lambda r: r["step"])


def mass_plot(pdf, rows, sched, source):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.98, hspace=0.28, wspace=0.16)
    page_title(fig, f"Jump mass  {sched}  {source}", "mean over n=8192. tail = P(K>3 | z)")
    for ax, name in zip(axes.ravel(), NAMES):
        sub = series(rows, name, sched, source)
        x = [r["step"] for r in sub]
        for key, label in Q_KEYS:
            ax.plot(x, [r[key] for r in sub], label=label, lw=1.3)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("step")
    axes[0, 0].legend(fontsize=8, frameon=False, ncol=4)
    pdf.savefig(fig)
    plt.close(fig)


def eta_plot(pdf, rows, sched):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.08, right=0.98, hspace=0.32, wspace=0.18)
    page_title(fig, f"eta ratios  {sched}", "p50 of eta_2/eta_1 and eta_3/eta_1. solid = forward, dashed = onpolicy")
    for ax, name in zip(axes.ravel(), NAMES):
        for source, ls in (("forward", "-"), ("onpolicy", "--")):
            sub = series(rows, name, sched, source)
            x = [r["step"] for r in sub]
            ax.plot(x, [r["r21_p50"] for r in sub], ls=ls, color="C0", label=f"eta2/eta1 {source}")
            ax.plot(x, [r["r31_p50"] for r in sub], ls=ls, color="C1", label=f"eta3/eta1 {source}")
        ax.set_yscale("log")
        ax.set_title(name, loc="left", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def g3(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{float(v):.3g}"


def step_page(pdf, rows, sched, source):
    headers = ["dataset", "step", "h", "q0", "q1", "q2", "q3", "P(K>3)", "P(K>3) p90", "eta2/eta1", "eta3/eta1"]
    cells = []
    for name in NAMES:
        sub = {r["step"]: r for r in series(rows, name, sched, source)}
        for step in STEPS:
            r = sub.get(step)
            if r is None:
                continue
            cells.append(
                [
                    name,
                    str(step),
                    g3(r["h"]),
                    g3(r["q0"]),
                    g3(r["q1"]),
                    g3(r["q2"]),
                    g3(r["q3"]),
                    g3(r["tail_gt3"]),
                    g3(r["tail_gt3_p90"]),
                    g3(r["r21_p50"]),
                    g3(r["r31_p50"]),
                ]
            )
    table_page(
        pdf,
        f"Selected steps  {sched}  {source}",
        "q(k) and P(K>3) are means. eta ratios are p50. Full 100-step table is dynamics.txt.",
        headers,
        cells,
        fontsize=7,
        n_header_cols=2,
    )


def main():
    grads = load(RAW / "grads.json")
    errors = load(RAW / "errors.json")
    target = load(DIAG / "target_scale.json")
    dyn = load(DIAG / "dynamics.json")
    for row in grads + errors:
        row["k"] = int(row["k"])
    for row in target:
        row["k"] = int(row["k"])
    with PdfPages(OUT) as pdf:
        cover(pdf)
        grad_page(pdf, grads)
        grad_plot(pdf, grads)
        target_page(pdf, target)
        error_pages(pdf, errors)
        for sched in ("quad", "unif"):
            for source in ("forward", "onpolicy"):
                mass_plot(pdf, dyn, sched, source)
        for sched in ("quad", "unif"):
            eta_plot(pdf, dyn, sched)
        for sched in ("quad", "unif"):
            for source in ("forward", "onpolicy"):
                step_page(pdf, dyn, sched, source)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
