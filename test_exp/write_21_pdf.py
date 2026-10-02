"""Consistency lambda sweep and hybrid m1. Tables plus the reading of the numbers."""
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

RES = HERE / "results"
OUT = RES / "21_report.pdf"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
KERNELS = ("poisson", "nb", "twopois")
KER = {"poisson": "Poisson", "nb": "NB", "twopois": "TwoPois"}
BINS = ("[0,0.1]", "[0.1,1]", "[1,10]", "[10,100]")


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


def qpair(stat):
    if not stat or stat.get("median") is None:
        return "—"
    return f"{stat['median']:.3g}/{stat['p99']:.3g}"


def table(pdf, title, sub, headers, cells, fontsize=7, n_header=1, widths=None):
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, title, sub)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.84])
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


def load_hybrid():
    rows = parse_rows(RES / "21_hybrid" / "sampling_metrics.txt")
    return {(r["tag"], r["dist"], r["kernel"]): r for r in rows}


def load_cons():
    rows = parse_rows(RES / "22_cons_scale" / "sampling_metrics.txt")
    idx = {(r["tag"], r["dist"], r["kernel"]): r for r in rows}
    # experiment 17 stores the column as weight, not tag, so the copier missed lambda 0
    for r in json.loads((RES / "17_raw_multi_sampling" / "sampling_metrics.json").read_text()):
        if r.get("weight") != "dyn_norm":
            continue
        if r.get("dist") not in NAMES or r.get("kernel") not in KERNELS:
            continue
        idx[("dyn_norm", r["dist"], r["kernel"])] = r
    return idx


def main():
    hy = load_hybrid()
    cons = load_cons()
    moments = json.loads((RES / "21_hybrid" / "moment_errors.json").read_text())
    grads = json.loads((RES / "21_hybrid" / "grad_summary.json").read_text())
    mom = {(r["tag"], r["dist"]): {b["bin"]: b for b in r["bins"]} for r in moments}
    grad = {(r["tag"], r["dist"]): r for r in grads}

    with PdfPages(OUT) as pdf:
        text_page(
            pdf,
            "Consistency scale and hybrid m1",
            "Quadratic T=100, lambda=100, p=2, n=50000. Four datasets. Cell later is WD/TV.",
            "\n".join(
                [
                    "Both trains finished. Consistency is the lambda sweep on dyn_norm.",
                    "Hybrid replaces raw-ratio L1 with D(X, m1) and keeps raw-ratio L2, L3.",
                    "",
                    "Hybrid Poisson against the direct learned mean",
                    "gamma TV:   eq 0.063,  dyn 0.073,  dyn_norm 0.092,  mean 0.144.",
                    "nb TV:      dyn 0.083,  dyn_norm 0.079,  mean 0.100,  eq 0.122.",
                    "            WD still prefers the mean (1.30 vs eq 2.06).",
                    "poismix_mod TV:  mean 0.056,  eq 0.174,  dyn_norm 0.365,  dyn 0.544.",
                    "poissmix3 TV:    mean 0.193,  dyn_norm 0.358,  dyn 0.361,  eq 0.408.",
                    "poissmix3 WD:    mean 10.7,  eq 13.8,  dyn 14.1,  dyn_norm 51.3.",
                    "",
                    "Reading",
                    "On gamma, the direct m1 loss is enough. Hybrid Poisson beats the learned-mean baseline.",
                    "On nb, TV is close to that baseline and WD is still worse.",
                    "On both mixtures, Poisson TV stays above the learned mean. Case B: changing the",
                    "k=1 objective removes most of the dyn_norm WD blow-up on poissmix3 (51 to 14),",
                    "and it does not by itself recover the mean-model Poisson sampler.",
                    "Trunk cos(g1, g2) and cos(g1, g3) are positive at the median, so g2 and g3",
                    "usually point with g1. The lower tail is near zero, so opposition is uncommon.",
                    "hybrid_dyn cuts ||g2|| and ||g3|| and still loses to hybrid_eq on poismix_mod",
                    "(TV 0.544 vs 0.174). Case D: h/2 and h^2/6 under-train the higher heads and",
                    "do not protect m1. On-policy median log(m1hat/m1*) is about 0 on every set.",
                    "The poissmix3 failure is a heavy tail: log1 p99 about 14, local KL p99 about 20.",
                    "",
                    "On hybrid_eq, NB and TwoPois bring poismix_mod TV from 0.174 to about 0.09,",
                    "still above mean Poisson 0.056. They do not help poissmix3 TV. TwoPois",
                    "fallback on eq is 0.38 (gamma), 0.51 (poismix_mod), 0.60 (poissmix3).",
                    "",
                    "Consistency",
                    "Unweighted lambda=1 and weighted lambda=1 both blow up poismix_mod",
                    "(Poisson WD about 48-50, overflow about 0.4). lambda=0.1 does too (WD 35).",
                    "lambda=1/3 is the stable unweighted point: poismix_mod TV 0.273 (lambda 0 was",
                    "0.365) and poissmix3 WD 14.6 (lambda 0 was 51). Weighted lambda=0.3 has the",
                    "best poismix_mod TV in the sweep, 0.182, but poissmix3 WD 29 is worse than 1/3.",
                    "No consistency setting reaches learned-mean Poisson TV on the mixtures.",
                    "",
                    "Next on this line: train the direct mean to convergence, freeze the trunk and",
                    "the m1 head, and train only m2 and m3. Another consistency penalty is not the",
                    "next step, and the schedule exponent stays at p=2 until mixture Poisson matches",
                    "the learned mean.",
                ]
            ),
        )

        headers = ["dataset", "mean", "dyn_norm", "hybrid_eq", "hybrid_dyn"]
        cells = []
        for name in NAMES:
            cells.append(
                [
                    name,
                    pair(hy.get(("mean", name, "poisson"))),
                    pair(hy.get(("dyn_norm", name, "poisson"))),
                    pair(hy.get(("hybrid_eq", name, "poisson"))),
                    pair(hy.get(("hybrid_dyn", name, "poisson"))),
                ]
            )
        table(
            pdf,
            "Primary: Poisson WD/TV",
            "Direct learned mean is 02_exponent p=2. dyn_norm is raw multi-head from experiment 17.",
            headers,
            cells,
            fontsize=9,
            widths=[0.18, 0.18, 0.18, 0.18, 0.18],
        )

        headers = ["dataset", "kernel", "mean", "dyn_norm", "hybrid_eq", "hybrid_dyn"]
        cells = []
        for name in NAMES:
            for kernel in KERNELS:
                cells.append(
                    [
                        name,
                        KER[kernel],
                        pair(hy.get(("mean", name, kernel))),
                        pair(hy.get(("dyn_norm", name, kernel))),
                        pair(hy.get(("hybrid_eq", name, kernel))),
                        pair(hy.get(("hybrid_dyn", name, kernel))),
                    ]
                )
        table(
            pdf,
            "Hybrid, all three kernels",
            "WD/TV. NB uses m1, m2. TwoPois uses m1, m2, m3. Mean NB/TwoPois are the shifted-mean formulas.",
            headers,
            cells,
            fontsize=7.5,
            n_header=2,
            widths=[0.16, 0.12, 0.16, 0.16, 0.16, 0.16],
        )

        headers = ["model", "dataset", "overflow p/nb/two", "TwoPois fallback", "invalid-v", "invalid-atom"]
        cells = []
        for tag in ("hybrid_eq", "hybrid_dyn", "dyn_norm"):
            for name in NAMES:
                ov, fb, iv, ia = [], [], [], []
                for kernel in KERNELS:
                    rec = hy.get((tag, name, kernel))
                    ov.append(num(None if rec is None else rec.get("overflow")))
                    if kernel == "twopois" and rec is not None:
                        fb.append(num(rec.get("fallback")))
                        iv.append(num(rec.get("invalid_v")))
                        ia.append(num(rec.get("invalid_atom")))
                cells.append([tag, name, " / ".join(ov), fb[0] if fb else "—", iv[0] if iv else "—", ia[0] if ia else "—"])
        table(
            pdf,
            "Overflow and TwoPois fallback",
            "Runaway was 0 on every hybrid and dyn_norm row. Mean overflow was not stored in 02_exponent.",
            headers,
            cells,
            fontsize=7.5,
            n_header=2,
            widths=[0.16, 0.16, 0.22, 0.16, 0.14, 0.14],
        )

        headers = ["dataset", "bin", "mean", "dyn_norm", "hybrid_eq", "hybrid_dyn"]
        cells = []
        for name in NAMES:
            for b in BINS:
                row = [name, b]
                for tag in ("mean", "dyn_norm", "hybrid_eq", "hybrid_dyn"):
                    row.append(qpair(mom[(tag, name)][b]["log1"]))
                cells.append(row)
        table(
            pdf,
            "Offline log(m1hat / m1*): median / p99",
            "Same validation draw for all four models. Bins are gamma. p99 is the upper tail of the signed log error.",
            headers,
            cells,
            fontsize=7,
            n_header=2,
            widths=[0.16, 0.12, 0.16, 0.16, 0.16, 0.16],
        )

        headers = ["model", "dataset", "||g1||", "||g2||", "||g3||", "cos12", "cos13", "cos23"]
        cells = []
        for tag in ("hybrid_eq", "hybrid_dyn"):
            for name in NAMES:
                g = grad[(tag, name)]
                heads = {h["head"]: h for h in g["heads"]}
                cos = {c["pair"]: c for c in g["cosine"]}
                cells.append(
                    [
                        tag,
                        name,
                        num(heads[1]["median"], 4),
                        num(heads[2]["median"], 4),
                        num(heads[3]["median"], 4),
                        num(cos["12"]["median"], 2),
                        num(cos["13"]["median"], 2),
                        num(cos["23"]["median"], 2),
                    ]
                )
        table(
            pdf,
            "Trunk gradient norms and cosine, training median",
            "g2 and g3 are the gradients of the terms that enter the loss, so hybrid_dyn includes h/2 and h^2/6. Positive cosine means the heads point together.",
            headers,
            cells,
            fontsize=8,
            n_header=2,
            widths=[0.16, 0.16, 0.11, 0.11, 0.11, 0.10, 0.10, 0.10],
        )

        headers = ["model", "dataset", "log1 med/p99", "h|m1 err| med/p99", "local KL med/p99"]
        cells = []
        for tag in ("hybrid_eq", "hybrid_dyn"):
            for name in NAMES:
                path = RES / "21_hybrid" / "sample" / tag / name / "on_policy_poisson.json"
                d = json.loads(path.read_text())
                allb = next(b for b in d["bins"] if b["bin"] == "all")
                cells.append(
                    [
                        tag,
                        name,
                        qpair(allb["log1"]),
                        qpair(allb["habs1"]),
                        qpair(allb["kl"]),
                    ]
                )
        table(
            pdf,
            "On-policy Poisson states",
            "First 8192 trajectories, every reverse step, quadratic T=100. local KL = h [m* log(m*/mhat) + mhat - m*].",
            headers,
            cells,
            fontsize=8,
            n_header=2,
            widths=[0.16, 0.18, 0.20, 0.22, 0.20],
        )

        tags = [
            ("dyn_norm", "l0"),
            ("l0.1", "l0.1"),
            ("l1o3", "l1/3"),
            ("l1_unw", "l1 unw"),
            ("w0.3", "w 0.3"),
            ("w1", "w 1"),
        ]
        for kernel, title in (
            ("poisson", "Consistency Poisson WD/TV"),
            ("nb", "Consistency NB WD/TV"),
            ("twopois", "Consistency TwoPois WD/TV"),
        ):
            headers = ["dataset"] + [lab for _, lab in tags]
            cells = []
            for name in NAMES:
                cells.append([name] + [pair(cons.get((tag, name, kernel))) for tag, _ in tags])
            table(
                pdf,
                title,
                "l0 is raw dyn_norm with no consistency. l0.1 and l1/3 add lambda (delta2^2+delta3^2). w uses lambda (w2 delta2^2+w3 delta3^2). l1 unw is experiment 19.",
                headers,
                cells,
                fontsize=7.5,
                widths=[0.16, 0.13, 0.13, 0.13, 0.14, 0.13, 0.13],
            )

        headers = ["tag", "dataset", "overflow p/nb/two", "TwoPois fallback"]
        cells = []
        for tag, lab in tags:
            for name in NAMES:
                ov, fb = [], "—"
                for kernel in KERNELS:
                    rec = cons.get((tag, name, kernel))
                    ov.append(num(None if rec is None else rec.get("overflow")))
                    if kernel == "twopois" and rec is not None:
                        fb = num(rec.get("fallback"))
                cells.append([lab, name, " / ".join(ov), fb])
        table(
            pdf,
            "Consistency overflow and TwoPois fallback",
            "Runaway was 0 on the newly sampled rows. Overflow is the fraction of samples outside the metric window.",
            headers,
            cells,
            fontsize=7,
            n_header=2,
            widths=[0.14, 0.18, 0.36, 0.22],
        )

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
