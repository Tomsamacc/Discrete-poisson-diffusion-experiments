"""Compile all test_exp numbers and plots into one PDF. Data only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "pois20", "poissmix", "poissmix3", "zip", "yule_simon")
KERNELS = ("poisson", "nb", "twopois")
KERNELS4 = ("poisson", "nb", "twopois", "repoisson")
MIX3 = ("poismix_mod", "poissmix", "poissmix3")
SWEEP_TAGS = ("T100_h1", "T200_h05", "T500_h02", "T1000_h01")
SWEEP_LAB = {"T100_h1": "T=100 h=1", "T200_h05": "T=200 h=0.5", "T500_h02": "T=500 h=0.2", "T1000_h01": "T=1000 h=0.1"}
ORACLE_TAGS = ("A_baseline", "B_true_mean", "C_exact_mix")
ORACLE_LAB = {"A_baseline": "A", "B_true_mean": "B m=E[X]", "C_exact_mix": "C X~p_X"}
EXP_P = (1.0, 1.5, 2.0, 3.0, 4.0)
CUTS = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
TRAJ_G = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
TRAJ_SHOW = (0.01, 0.1, 1.0, 10.0, 100.0)
SETUP = "Frozen experiments/scale/*/best.pt  ·  λ=100  ·  z_rescale  ·  n=50000 (diag n=8192)"
METRIC_NOTE = "WD = mean |sorted a−b|   TV = ½∑|p−q| on the K_0.999 window"


def parse_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split()
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != len(header):
            continue
        d = dict(zip(header, parts))
        for k, v in list(d.items()):
            try:
                d[k] = float(v)
            except ValueError:
                pass
        rows.append(d)
    return rows


def find(rows, **kw):
    for r in rows:
        if all(r.get(k) == v or (isinstance(r.get(k), float) and isinstance(v, float) and abs(r.get(k) - v) < 1e-9) for k, v in kw.items()):
            return r
    return None


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{float(v):.{nd}f}"


def page_title(fig, title, sub=None):
    fig.text(0.055, 0.965, title, fontsize=12, fontweight="medium", va="top")
    if sub:
        fig.text(0.055, 0.932, sub, fontsize=7.5, color="0.35", va="top")


def style_table(tbl, n_header_cols=2, fontsize=7):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, 1.28)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.85")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("0.93")
            cell.set_text_props(fontweight="medium")
        elif r % 2 == 0:
            cell.set_facecolor("0.97")
        if c >= n_header_cols:
            cell.get_text().set_ha("right")
            cell.get_text().set_x(0.92)


def table_page(pdf, title, sub, headers, cells, fontsize=7, n_header_cols=2):
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, title, sub)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.86])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="upper center", cellLoc="center")
    style_table(tbl, n_header_cols=n_header_cols, fontsize=fontsize)
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf, title, path, sub=SETUP):
    path = Path(path)
    if not path.is_file():
        return
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, title, sub)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.86])
    ax.imshow(plt.imread(path))
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def image_grid(pdf, title, items, nrows, ncols, sub=SETUP):
    items = [(lab, Path(p)) for lab, p in items if Path(p).is_file()]
    if not items:
        return
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, title, sub)
    for i, (lab, p) in enumerate(items):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(plt.imread(p))
        ax.set_title(lab, fontsize=7.5, pad=2)
        ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.02, wspace=0.08, hspace=0.18)
    pdf.savefig(fig)
    plt.close(fig)


def cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    page_title(fig, "test_exp  ·  all sampling results", SETUP)
    ax = fig.add_axes([0.07, 0.08, 0.86, 0.80])
    ax.axis("off")
    ax.text(
        0,
        1,
        "\n".join(
            [
                METRIC_NOTE,
                "Decode  x̂ = z / γ_T.  RePoisson only in the earlier 3-mix block.",
                "",
                "Earlier (3 mixes: poismix_mod, poissmix, poissmix3)",
                "  A1  uniform step-size sweep     T=100/200/500/1000,  h=1/0.5/0.2/0.1",
                "  A2  quadratic vs uniform T=100  γ_t = 100 (t/100)^2",
                "  A3  oracle first step           A current; B K~Pois(h E[X]); C X~p_X, K~Pois(hX)",
                "  A4  quadratic γ: 0.1→100        z0 ~ Pois(0.1 X)     (8 datasets)",
                "  A5  uniform   γ: 0.1→100        z0 ~ Pois(0.1 X)     (8 datasets)",
                "",
                "This round (8 datasets, same ckpt, no retrain)",
                "  01  trajectory marginals        z_γ vs true Pois(γ X), quadratic T=100",
                "  02  schedule exponent           γ_t = 100 (t/100)^p,  p=1,1.5,2,3,4",
                "  03  missing variance            R_t = h v / m along quadratic trajectories",
                "  04  oracle posterior mean       learned vs m*(z,γ), Poisson kernel, unif & quad",
                "  05  early exact cutoff          exact mixed-Poisson for γ<γ_cut, then mean-Poisson",
                "  06  first-step                  E[X], m_θ(z=0), exact gap PMF at h=0.01 and h=1",
                "  07  Two-Poisson stability       v,c3,g,w,x1,x2,hx  quantiles and fallback fractions",
                "  08  step-size sweep             all 8 datasets, T=100/200/500/1000",
                "",
                "Round 2 (same ckpt, quadratic unless noted)",
                "  09  oracle-mean cutoff          m* for γ<γ_cut, then m_θ; still K~Pois(h m)",
                "  10  oracle Two-Poisson          true E[X^k|z] vs +k network moments",
                "  11  θ-trapezoidal mean-Poisson  Heun θ=1/2; euler T=100, trap T=50/T=100",
            ]
        ),
        va="top",
        family="monospace",
        fontsize=9,
        linespacing=1.45,
    )
    pdf.savefig(fig)
    plt.close(fig)


def old_block(pdf):
    old = parse_rows(HERE / "metrics.txt")
    # A1 sweep
    for metric, lab in (("tv", "TV"), ("wd", "WD")):
        headers = ["dataset", "kernel"] + [SWEEP_LAB[t] for t in SWEEP_TAGS]
        cells = []
        for dist in MIX3:
            for k in KERNELS4:
                cells.append(
                    [dist, k]
                    + [
                        fmt(None if find(old, test="sweep", tag=t, dist=dist, kernel=k) is None else find(old, test="sweep", tag=t, dist=dist, kernel=k)[metric])
                        for t in SWEEP_TAGS
                    ]
                )
        table_page(pdf, f"A1  uniform step-size sweep  {lab}", f"{SETUP}   ·   {METRIC_NOTE}", headers, cells, fontsize=7)
    image_page(pdf, "A1  uniform step-size sweep  log PMF", HERE / "sweep_log.png")

    for metric, lab in (("tv", "TV"), ("wd", "WD")):
        headers = ["dataset", "kernel", "uniform T=100 h=1", "quad T=100 h0=0.01"]
        cells = []
        for dist in MIX3:
            for k in KERNELS4:
                cells.append(
                    [dist, k]
                    + [
                        fmt((find(old, test="quad", tag=tag, dist=dist, kernel=k) or {}).get(metric))
                        for tag in ("T100_h1", "T100_quad")
                    ]
                )
        table_page(pdf, f"A2  quadratic vs uniform T=100  {lab}", f"{SETUP}   ·   {METRIC_NOTE}", headers, cells)
    image_page(pdf, r"A2  quadratic T=100  $\gamma_t=100(t/100)^2$", HERE / "quad_vs_uniform.png")

    for metric, lab in (("tv", "TV"), ("wd", "WD")):
        headers = ["dataset", "kernel"] + [ORACLE_LAB[t] for t in ORACLE_TAGS]
        cells = []
        for dist in MIX3:
            for k in KERNELS4:
                cells.append(
                    [dist, k]
                    + [
                        fmt((find(old, test="oracle", tag=t, dist=dist, kernel=k) or {}).get(metric))
                        for t in ORACLE_TAGS
                    ]
                )
        table_page(pdf, f"A3  oracle first step  {lab}", f"{SETUP}   ·   rest of trajectory unchanged", headers, cells)
    image_page(pdf, "A3  oracle first step  log PMF", HERE / "oracle.png")

    q01 = parse_rows(HERE / "metrics_quad01.txt")
    u01 = parse_rows(HERE / "metrics_unif01.txt")
    for metric, lab in (("tv", "TV"), ("wd", "WD")):
        headers = ["dataset", "kernel", "unif γ0=0.1", "quad γ0=0.1"]
        cells = []
        for dist in NAMES:
            for k in KERNELS:
                ru = find(u01, dist=dist, kernel=k)
                rq = find(q01, dist=dist, kernel=k)
                cells.append([dist, k, fmt((ru or {}).get(metric)), fmt((rq or {}).get(metric))])
        table_page(
            pdf,
            f"A4/A5  start at γ=0.1, z0~Pois(0.1 X)  {lab}",
            f"{SETUP}   ·   T=100, γ: 0.1→100",
            headers,
            cells,
        )
    image_page(pdf, r"A4  quadratic  $\gamma$: 0.1→100,  $z_0\sim\mathrm{Pois}(0.1X)$", HERE / "quad_g01.png")
    image_page(pdf, r"A5  uniform  $\gamma$: 0.1→100,  $z_0\sim\mathrm{Pois}(0.1X)$", HERE / "unif_g01.png")


def traj_block(pdf):
    rows = parse_rows(RES / "01_traj_marginal" / "metrics.txt")
    hits = json.loads((RES / "01_traj_marginal" / "actual_gamma.json").read_text()) if (RES / "01_traj_marginal" / "actual_gamma.json").is_file() else []
    map_line = "  ".join(f"{h['gamma']:g}→{h['gamma_act']:.4g}" for h in hits)
    for metric, lab in (("tv", "TV"), ("wd", "WD")):
        headers = ["dataset", "kernel"] + [f"γ={g:g}" for g in TRAJ_SHOW]
        cells = []
        for dist in NAMES:
            for k in KERNELS:
                cells.append(
                    [dist, k]
                    + [
                        fmt((find(rows, name=dist, kernel=k, gamma=float(g)) or {}).get(metric))
                        for g in TRAJ_SHOW
                    ]
                )
        table_page(
            pdf,
            f"01  trajectory-marginal {lab}(z_γ vs true Pois(γ_act X))  selected γ",
            f"{SETUP}   ·   quadratic T=100   ·   nearest schedule γ: {map_line}",
            headers,
            cells,
            fontsize=6.5,
        )
    # full TV per 4 datasets
    for chunk in (NAMES[:4], NAMES[4:]):
        headers = ["dataset", "kernel"] + [f"{g:g}" for g in TRAJ_G]
        cells = []
        for dist in chunk:
            for k in KERNELS:
                cells.append(
                    [dist, k]
                    + [fmt((find(rows, name=dist, kernel=k, gamma=float(g)) or {}).get("tv")) for g in TRAJ_G]
                )
        table_page(
            pdf,
            "01  trajectory-marginal TV  full γ grid",
            f"{SETUP}   ·   quadratic T=100   ·   columns = requested γ",
            headers,
            cells,
            fontsize=6,
        )
    image_page(pdf, r"01  TV$(z_\gamma)$ vs $\gamma$  (quadratic T=100)", RES / "01_traj_marginal" / "tv_vs_gamma.png")
    image_page(pdf, r"01  $z_\gamma$ PMF vs true Pois($\gamma X$)", RES / "01_traj_marginal" / "z_pmf_selected.png")


def exponent_block(pdf):
    rows = parse_rows(RES / "02_exponent" / "metrics.txt")
    for metric, lab in (("tv", "TV"), ("wd", "WD")):
        headers = ["dataset", "kernel"] + [f"p={p:g}" for p in EXP_P]
        cells = []
        for dist in NAMES:
            for k in KERNELS:
                cells.append(
                    [dist, k]
                    + [fmt((find(rows, dist=dist, kernel=k, power=float(p)) or {}).get(metric)) for p in EXP_P]
                )
        table_page(
            pdf,
            rf"02  schedule exponent  {lab}   $\gamma_t=100(t/100)^p$, T=100",
            f"{SETUP}   ·   all schedules end at γ=100",
            headers,
            cells,
            fontsize=6.5,
        )
    image_page(pdf, r"02  TV vs $p$", RES / "02_exponent" / "tv_vs_p.png")
    image_page(pdf, r"02  WD vs $p$", RES / "02_exponent" / "wd_vs_p.png")
    for p, tag in ((1.0, "p1"), (2.0, "p2"), (4.0, "p4")):
        items = [(n, RES / "02_exponent" / tag / n / "pmf.png") for n in NAMES]
        image_grid(pdf, rf"02  PMF  $p={p:g}$  ($\gamma_t=100(t/100)^{{{p:g}}}$)", items, 2, 4)


def r_block(pdf):
    image_page(pdf, r"03  $R_t=h\,v/m$  along quadratic T=100 trajectories", RES / "03_R" / "R_vs_gamma.png")
    headers = ["dataset", "kernel"] + [f"med@{g:g}" for g in TRAJ_SHOW] + [f"p90@{g:g}" for g in (0.01, 1.0, 100.0)]
    cells = []
    for dist in NAMES:
        for k in KERNELS:
            npz = RES / "03_R" / dist / k / "R_vs_gamma.npz"
            if not npz.is_file():
                continue
            d = np.load(npz)
            g = d["gamma"]

            def at(arr, gq):
                i = int(np.argmin(np.abs(g - gq)))
                return float(arr[i])

            cells.append(
                [dist, k]
                + [fmt(at(d["median"], gq), 4) for gq in TRAJ_SHOW]
                + [fmt(at(d["p90"], gq), 4) for gq in (0.01, 1.0, 100.0)]
            )
    if cells:
        table_page(
            pdf,
            r"03  $R=h v/m$  at selected $\gamma$  (median / p90)",
            f"{SETUP}   ·   quadratic T=100",
            headers,
            cells,
            fontsize=6,
        )


def oracle_mean_block(pdf):
    rows = parse_rows(RES / "04_oracle_mean" / "metrics.txt")
    headers = ["dataset", "unif learned", "unif oracle", "quad learned", "quad oracle"]
    for metric, lab in (("tv", "TV"), ("wd", "WD")):
        cells = []
        for dist in NAMES:
            cells.append(
                [dist]
                + [
                    fmt((find(rows, dist=dist, sched=sched, which=which) or {}).get(metric))
                    for sched, which in (
                        ("uniform", "learned"),
                        ("uniform", "oracle"),
                        ("quad", "learned"),
                        ("quad", "oracle"),
                    )
                ]
            )
        table_page(
            pdf,
            f"04  oracle posterior mean + Poisson kernel  {lab}",
            f"{SETUP}   ·   T=100  ·  m* = E[X | Z_γ=z] from the known data PMF",
            headers,
            cells,
            n_header_cols=1,
        )
    for sched, which, title in (
        ("uniform_learned", "learned", "04  uniform T=100  learned mean + Poisson"),
        ("uniform_oracle", "oracle", "04  uniform T=100  oracle mean + Poisson"),
        ("quad_learned", "learned", "04  quadratic T=100  learned mean + Poisson"),
        ("quad_oracle", "oracle", "04  quadratic T=100  oracle mean + Poisson"),
    ):
        items = [(n, RES / "04_oracle_mean" / sched / n / "pmf.png") for n in NAMES]
        image_grid(pdf, title, items, 2, 4)


def cutoff_block(pdf):
    rows = parse_rows(RES / "05_cutoff" / "metrics.txt")
    for metric, lab in (("tv", "TV"), ("wd", "WD")):
        headers = ["dataset"] + [f"cut={c:g}" for c in CUTS]
        cells = []
        for dist in NAMES:
            cells.append(
                [dist]
                + [fmt((find(rows, dist=dist, cut=float(c)) or {}).get(metric)) for c in CUTS]
            )
        table_page(
            pdf,
            f"05  early exact mixed-Poisson cutoff  {lab}  (then mean-Poisson)",
            f"{SETUP}   ·   quadratic T=100  ·  Poisson kernel after γ_cut",
            headers,
            cells,
            n_header_cols=1,
        )
    image_page(pdf, r"05  TV vs $\gamma_{\mathrm{cut}}$", RES / "05_cutoff" / "tv_vs_cut.png")
    for c, tag in ((0.01, "cut_0p01"), (0.1, "cut_0p1"), (1.0, "cut_1")):
        items = [(n, RES / "05_cutoff" / tag / n / "pmf.png") for n in NAMES]
        image_grid(pdf, rf"05  PMF  $\gamma_{{\mathrm{{cut}}}}={c:g}$", items, 2, 4)


def first_step_block(pdf):
    path = RES / "06_first_step" / "summary.txt"
    if path.is_file():
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        headers = lines[0].split()
        cells = [ln.split() for ln in lines[1:]]
        table_page(
            pdf,
            "06  first-step  E[X], m_θ(z=0), exact K=Pois(hX)",
            r"t_eps=1e-4, λ=100 ⇒ z_rescale at γ=0 uses rate λ t_eps=0.01, so m_t_eps = m_g0.01.  TV = exact gap vs Pois(h m_θ).",
            headers,
            cells,
            fontsize=6.5,
            n_header_cols=1,
        )
    items = [(n, RES / "06_first_step" / n / "gap_pmf.png") for n in NAMES]
    image_grid(pdf, "06  exact first-step gap PMF  h=0.01 (quad) and h=1 (uniform)", items, 4, 2, sub=SETUP)


def twopois_block(pdf):
    path = RES / "07_twopois" / "summary.txt"
    if path.is_file():
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        headers = lines[0].split()
        cells = [ln.split() for ln in lines[1:]]
        table_page(
            pdf,
            "07  Two-Poisson  fallback fractions and hx2 tails",
            f"{SETUP}   ·   quadratic T=100  ·  pooled over steps and samples",
            headers,
            cells,
            fontsize=6.5,
            n_header_cols=1,
        )
    keys = ("v", "c3", "g", "w", "x1", "x2", "hx1", "hx2")
    qlabs = ("median", "p90", "p99", "p999", "max")
    for key in keys:
        headers = ["dataset"] + list(qlabs)
        cells = []
        for dist in NAMES:
            js = RES / "07_twopois" / dist / "stats.json"
            if not js.is_file():
                continue
            blob = json.loads(js.read_text())[key]
            cells.append([dist] + [fmt(blob[q], 4 if key in ("w",) else 4) for q in qlabs])
        table_page(
            pdf,
            f"07  Two-Poisson  {key}  quantiles (pooled)",
            f"{SETUP}   ·   quadratic T=100",
            headers,
            cells,
            n_header_cols=1,
        )
    image_page(pdf, "07  Two-Poisson  fallback / x1<0 / extreme w  vs γ", RES / "07_twopois" / "fallback_vs_gamma.png")


def sweep_block(pdf):
    rows = parse_rows(RES / "08_sweep" / "metrics.txt")
    for metric, lab in (("tv", "TV"), ("wd", "WD")):
        headers = ["dataset", "kernel"] + [SWEEP_LAB[t] for t in SWEEP_TAGS]
        cells = []
        for dist in NAMES:
            for k in KERNELS:
                cells.append(
                    [dist, k]
                    + [fmt((find(rows, dist=dist, kernel=k, tag=t) or {}).get(metric)) for t in SWEEP_TAGS]
                )
        table_page(
            pdf,
            f"08  uniform step-size sweep  {lab}  (all 8 datasets)",
            f"{SETUP}   ·   γ_max=100",
            headers,
            cells,
            fontsize=6.5,
        )
    for tag, title in (
        ("T100_h1", "08  PMF  T=100  h=1"),
        ("T200_h05", "08  PMF  T=200  h=0.5"),
        ("T500_h02", "08  PMF  T=500  h=0.2"),
        ("T1000_h01", "08  PMF  T=1000  h=0.1"),
    ):
        items = [(n, RES / "08_sweep" / tag / n / "pmf.png") for n in NAMES]
        image_grid(pdf, title, items, 2, 4)


def mean_cut_block(pdf):
    rows = parse_rows(RES / "09_mean_cut" / "metrics.txt")
    tags = ("cut_0", "cut_0p1", "cut_1", "cut_inf")
    labs = ("learned all γ", "oracle γ<0.1", "oracle γ<1", "oracle all γ")
    for metric, lab in (("tv", "TV"), ("wd", "WD")):
        headers = ["dataset"] + list(labs)
        cells = []
        for dist in NAMES:
            cells.append(
                [dist]
                + [fmt((find(rows, dist=dist, tag=t) or {}).get(metric)) for t in tags]
            )
        table_page(
            pdf,
            f"09  oracle mean for γ<γ_cut, then learned mean + Poisson  {lab}",
            f"{SETUP}   ·   quadratic T=100  ·  still K~Pois(h m), only the mean is swapped",
            headers,
            cells,
            n_header_cols=1,
        )
    image_page(pdf, r"09  TV vs $\gamma_{\mathrm{cut}}$  (oracle mean below the cut)", RES / "09_mean_cut" / "tv_vs_cut.png")
    for tag, title in (
        ("cut_0", r"09  PMF  learned mean all $\gamma$"),
        ("cut_0p1", r"09  PMF  oracle mean $\gamma<0.1$, then learned"),
        ("cut_1", r"09  PMF  oracle mean $\gamma<1$, then learned"),
        ("cut_inf", r"09  PMF  oracle mean all $\gamma$"),
    ):
        items = [(n, RES / "09_mean_cut" / tag / n / "pmf.png") for n in NAMES]
        image_grid(pdf, title, items, 2, 4)


def oracle_twopois_block(pdf):
    rows = parse_rows(RES / "10_oracle_twopois" / "metrics.txt")
    headers = ["dataset", "unif learned", "unif oracle m123", "quad learned", "quad oracle m123"]
    for metric, lab in (("tv", "TV"), ("wd", "WD")):
        cells = []
        for dist in NAMES:
            cells.append(
                [dist]
                + [
                    fmt((find(rows, dist=dist, sched=sched, which=which) or {}).get(metric))
                    for sched, which in (
                        ("uniform", "learned"),
                        ("uniform", "oracle"),
                        ("quad", "learned"),
                        ("quad", "oracle"),
                    )
                ]
            )
        table_page(
            pdf,
            f"10  Two-Poisson  learned +k moments vs oracle E[X^k|z]  {lab}",
            f"{SETUP}   ·   T=100  ·  oracle uses true m1,m2,m3; learned uses m(z)m(z+1)m(z+2)",
            headers,
            cells,
            n_header_cols=1,
        )
    for sched, title in (
        ("uniform_learned", "10  uniform T=100  Two-Poisson  learned +k moments"),
        ("uniform_oracle", "10  uniform T=100  Two-Poisson  oracle (m1,m2,m3)"),
        ("quad_learned", "10  quadratic T=100  Two-Poisson  learned +k moments"),
        ("quad_oracle", "10  quadratic T=100  Two-Poisson  oracle (m1,m2,m3)"),
    ):
        items = [(n, RES / "10_oracle_twopois" / sched / n / "pmf.png") for n in NAMES]
        image_grid(pdf, title, items, 2, 4)


def trap_block(pdf):
    rows = parse_rows(RES / "11_trap" / "metrics.txt")
    tags = ("euler_T100", "trap_T50", "trap_T100")
    labs = ("euler T=100  100 NFE", "trap T=50  100 NFE", "trap T=100  200 NFE")
    for metric, lab in (("tv", "TV"), ("wd", "WD")):
        headers = ["dataset"] + list(labs)
        cells = []
        for dist in NAMES:
            cells.append(
                [dist]
                + [fmt((find(rows, dist=dist, tag=t) or {}).get(metric)) for t in tags]
            )
        table_page(
            pdf,
            f"11  θ-trapezoidal mean-Poisson (Heun θ=1/2)  {lab}",
            r"quadratic $\gamma_t=\gamma_{\max}(t/T)^2$  ·  $\hat z=z+h m(z,\gamma)$, $K\sim\mathrm{Pois}(h(m_0+m_1)/2)$",
            headers,
            cells,
            n_header_cols=1,
        )
    image_page(pdf, "11  TV by integrator", RES / "11_trap" / "tv_vs_method.png")
    for tag, title in (
        ("euler_T100", "11  PMF  Euler mean-Poisson  T=100  (100 NFE)"),
        ("trap_T50", "11  PMF  θ-trapezoidal  T=50  (100 NFE)"),
        ("trap_T100", "11  PMF  θ-trapezoidal  T=100  (200 NFE)"),
    ):
        items = [(n, RES / "11_trap" / tag / n / "pmf.png") for n in NAMES]
        image_grid(pdf, title, items, 2, 4)


def main():
    out_a = HERE / "results.pdf"
    out_b = RES / "results.pdf"
    RES.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_a) as pdf:
        cover(pdf)
        old_block(pdf)
        traj_block(pdf)
        exponent_block(pdf)
        r_block(pdf)
        oracle_mean_block(pdf)
        cutoff_block(pdf)
        first_step_block(pdf)
        twopois_block(pdf)
        sweep_block(pdf)
        mean_cut_block(pdf)
        oracle_twopois_block(pdf)
        trap_block(pdf)
    out_b.write_bytes(out_a.read_bytes())
    n = 0
    try:
        import pypdf

        n = len(pypdf.PdfReader(str(out_a)).pages)
    except Exception:
        pass
    print(f"saved {out_a}" + (f"  ({n} pages)" if n else "") + f"  and {out_b}", flush=True)


if __name__ == "__main__":
    main()
