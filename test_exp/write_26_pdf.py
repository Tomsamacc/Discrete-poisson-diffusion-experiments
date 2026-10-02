"""Higher-head parameterization. WD/TV tables."""
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
from write_pdf import page_title, parse_rows, style_table  # noqa: E402

RES = HERE / "results"
OUT = RES / "26_report.pdf"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
DIST = {
    "gamma_ltj": "Exp(10)",
    "nb": "NB mean 20",
    "poismix_mod": "1/2 Pois(5)+1/2 Pois(50)",
    "poissmix3": "1/3 Pois(1)+1/3 Pois(50)+1/3 Pois(100)",
}
PARAMS = ("raw_log", "direct_ratio", "log_moment", "root_moment", "direct_moment")
PLAB = {
    "raw_log": "Raw log-ratio",
    "direct_ratio": "Direct ratio",
    "log_moment": "Log-moment",
    "root_moment": "Root-moment",
    "direct_moment": "Direct moment",
}
WEIGHTS = ("equal", "dyn")
WLAB = {"equal": "equal", "dyn": "dyn"}


def pair(rec):
    if rec is None:
        return "—"
    wd, tv = rec.get("wd"), rec.get("tv")
    if wd is None or tv is None:
        return "—"
    if not (np.isfinite(float(wd)) and np.isfinite(float(tv))):
        return "—"
    return f"{float(wd):.2f}/{float(tv):.3f}"


def num(v, nd=3):
    if v is None:
        return "—"
    x = float(v)
    if not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"


def table(pdf, title, sub, headers, cells, fontsize=7, n_header=2, widths=None):
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, title, sub)
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.82])
    ax.axis("off")
    tbl = ax.table(
        cellText=cells,
        colLabels=headers,
        loc="upper center",
        cellLoc="center",
        colWidths=widths,
    )
    style_table(tbl, n_header_cols=n_header, fontsize=fontsize)
    pdf.savefig(fig)
    plt.close(fig)


def text_page(pdf, title, sub, body):
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, title, sub)
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.84])
    ax.axis("off")
    ax.text(0, 1, body, va="top", ha="left", fontsize=8.5, family="DejaVu Sans", linespacing=1.35)
    pdf.savefig(fig)
    plt.close(fig)


def load():
    rows = parse_rows(RES / "26_hm" / "sampling_metrics.txt")
    idx = {(r["weight"], r["param"], r["dist"], r["kernel"]): r for r in rows}
    direct = {}
    for r in parse_rows(RES / "23_param_poisson" / "sampling_metrics.txt"):
        if r.get("tag") == "direct":
            direct[r["dist"]] = r
    return idx, direct


def kernel_page(pdf, idx, direct, kernel, title):
    headers = ["loss", "dataset", *[PLAB[p] for p in PARAMS]]
    if kernel == "poisson":
        headers.append("Single-head mean")
    cells = []
    for weight in WEIGHTS:
        for name in NAMES:
            row = [WLAB[weight], DIST[name]]
            for param in PARAMS:
                row.append(pair(idx.get((weight, param, name, kernel))))
            if kernel == "poisson":
                row.append(pair(direct.get(name)) if weight == "equal" else "—")
            cells.append(row)
    n = len(headers)
    widths = [0.08, 0.24] + [0.68 / (n - 2)] * (n - 2)
    sub = "Cell is WD/TV. Quadratic T=100, n=50000, seed 0."
    if kernel == "poisson":
        sub += " Single-head mean is direct_prl/direct/k1, Poisson only, shown on the equal rows."
    if kernel == "nb":
        sub += " NB reverse kernel uses m1 and m2."
    if kernel == "twopois":
        sub += " TwoPois uses m1, m2, m3."
    table(pdf, title, sub, headers, cells, fontsize=6.5, widths=widths)


def main():
    idx, direct = load()
    with PdfPages(OUT) as pdf:
        text_page(
            pdf,
            "Higher-head parameterization",
            "m1 is softplus on every run. k=2,3 change only the link. Loss and target stay the same.",
            "\n".join(
                [
                    "m1 = softplus(f1),   L1 = D(X, m1).",
                    "For k=2,3 the loss is L_k = D(R_k, S_k).",
                    "R_k = (gamma X)^k / (z+1)...(z+k).",
                    "c_k = (z+1)...(z+k) / gamma^k.",
                    "",
                    "Raw log-ratio:   S = exp(a),          m = c S.",
                    "Direct ratio:    S = softplus(u),     m = c S.",
                    "Log-moment:      m = exp(b),          S = m / c.",
                    "Root-moment:     m = softplus(r)^k,   S = m / c.",
                    "Direct moment:   m = softplus(v),     S = m / c.",
                    "",
                    "equal:  L1 + L2 + L3.",
                    "dyn:    L1 + (h/2) L2 + (h^2/6) L3.",
                    "h is the quadratic T=100 reverse step.",
                    "",
                    "Four datasets. config_scale, lambda=100, z_rescale, 200 epochs,",
                    "Adam 1e-3, batch 256, seed 0.",
                    "Sampling is quadratic T=100, n=50000, seed 0.",
                    "Kernels: Poisson (m1), NB (m1, m2), TwoPois (m1, m2, m3).",
                    "WD = mean |sorted a-b|.  TV = half the L1 distance on the K_0.999 window.",
                    "Runaway is 0 on every run.",
                    "",
                    "Datasets",
                    "Exp(10):                         gamma(shape 1, scale 10).",
                    "NB mean 20:                      nbinom(n=5, p=0.2).",
                    "1/2 Pois(5)+1/2 Pois(50).",
                    "1/3 Pois(1)+1/3 Pois(50)+1/3 Pois(100).",
                ]
            ),
        )
        kernel_page(pdf, idx, direct, "poisson", "Poisson WD/TV")
        kernel_page(pdf, idx, direct, "nb", "NB kernel WD/TV")
        kernel_page(pdf, idx, direct, "twopois", "TwoPois WD/TV")

        headers = ["loss", "dataset", *[PLAB[p] for p in PARAMS]]
        cells = []
        for weight in WEIGHTS:
            for name in NAMES:
                row = [WLAB[weight], DIST[name]]
                for param in PARAMS:
                    rec = idx.get((weight, param, name, "twopois"))
                    row.append(num(None if rec is None else rec.get("fallback")))
                cells.append(row)
        table(
            pdf,
            "TwoPois fallback fraction",
            "Share of reverse steps that used the fallback mixture. Runaway is 0.",
            headers,
            cells,
            fontsize=7,
            widths=[0.08, 0.24, 0.136, 0.136, 0.136, 0.136, 0.136],
        )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
