"""Shared tag list and the comparison tables for the consistency stage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "gamma_ltj",
    "nb",
    "poismix_mod",
    "poissmix3",
)


def tag_specs(skip_dyn_norm=False):
    rows = [
        ("equal", "equal", "none", False),
        ("equal_proj", "equal", "project", True),
        ("soft", "soft", "none", True),
        ("anchored", "anchored", "none", True),
        ("hard", "hard", "hard", True),
        ("dyn_norm", "dyn_norm", "none", False),
        ("dyn_norm_proj", "dyn_norm", "project", True),
    ]
    if skip_dyn_norm:
        rows = [r for r in rows if r[1] != "dyn_norm"]
    return [{"tag": t, "kind": k, "mode": m, "sample": s} for t, k, m, s in rows]


def ckpt_of(kind, name, family=None):
    if family == "raw_cons_dyn_norm":
        return ROOT / "experiments" / "raw_cons_dyn_norm" / kind / name / "best.pt"
    if family == "raw_cons":
        return ROOT / "experiments" / "raw_cons" / kind / name / "best.pt"
    if kind in ("equal", "dyn_norm") or family == "raw_multi":
        return ROOT / "experiments" / "raw_multi" / kind / name / "best.pt"
    return ROOT / "experiments" / "raw_cons" / kind / name / "best.pt"


def dyn_norm_tag_specs():
    """dyn_norm data loss. soft and anchored are the new runs.

    dyn_norm, dyn_norm_proj, hard, and the equal-weight consistency runs
    are copied from earlier results and are not resampled.
    """
    rows = [
        ("dyn_norm", "dyn_norm", "none", False, "raw_multi"),
        ("dyn_norm_proj", "dyn_norm", "project", False, "raw_multi"),
        ("soft", "soft", "none", True, "raw_cons_dyn_norm"),
        ("anchored", "anchored", "none", True, "raw_cons_dyn_norm"),
        ("hard", "hard", "hard", False, "raw_cons"),
        ("eq_soft", "soft", "none", False, "raw_cons"),
        ("eq_anchored", "anchored", "none", False, "raw_cons"),
    ]
    return [
        {"tag": t, "kind": k, "mode": m, "sample": s, "family": f}
        for t, k, m, s, f in rows
    ]


def missing_ckpts(names, skip_dyn_norm=False):
    kinds = ["equal", "soft", "anchored", "hard"]
    if not skip_dyn_norm:
        kinds.append("dyn_norm")
    missing = []
    for kind in kinds:
        for name in names:
            path = ckpt_of(kind, name)
            if not path.is_file():
                missing.append(path)
    return missing


def load_json(path):
    path = Path(path)
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def num(v, spec=".6g"):
    if v is None:
        return "nan"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return format(x, spec)


def write_comparison(out_dir, tag_order=None, preamble=None):
    """Join offline residuals with sampling metrics. Missing pieces stay blank."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cons = {(r["tag"], r["dist"]): r for r in load_json(out / "consistency.json")}
    coh = {(r["tag"], r["dist"]): r for r in load_json(out / "coherence.json")}
    ratio = load_json(out / "ratio_errors.json")
    moment = load_json(out / "moment_errors.json")
    samp = load_json(out / "sampling_metrics.json")
    names = [n for n in NAMES if any(k[1] == n for k in cons) or any(r.get("dist") == n for r in samp)]
    if not names:
        names = list(NAMES)
    present = {k[0] for k in cons} | {r.get("tag") for r in samp}
    if tag_order is None:
        tags = []
        for spec in tag_specs(skip_dyn_norm=False):
            tag = spec["tag"]
            if tag in present:
                tags.append(tag)
        if not tags:
            tags = [s["tag"] for s in tag_specs()]
    else:
        tags = [tag for tag in tag_order if tag in present]

    ratio_ix = {
        (r["tag"], r["dist"], int(r["k"]), r["bin"]): r
        for r in ratio
    }
    mom_ix = {
        (r["tag"], r["dist"], int(r["k"]), r["bin"]): r
        for r in moment
    }
    samp_ix = {(r.get("tag"), r.get("dist"), r.get("kernel")): r for r in samp}

    if preamble is None:
        lines = [
            "# Ratio consistency",
            "",
            "Data loss for the new runs is L1+L2+L3. lambda_cons = 1.",
            "equal and dyn_norm WD/TV are the existing experiment 17 samples.",
            "equal_proj and dyn_norm_proj are Euclidean projections of those checkpoints.",
            "|delta| is the residual of the log-ratios used for moments.",
            "delta_in is that residual before projection.",
            "logS and logm are signed log(hat/star) on the shared val draw. abs is |log(hat/star)|.",
            "val fallback is the TwoPois mask on those val states. sample fallback is the reverse chain.",
            "",
        ]
    else:
        lines = list(preamble)
        if lines and lines[-1] != "":
            lines.append("")
    flat = []
    err_flat = []
    for name in names:
        lines.append(f"## {name}")
        lines.append("")
        lines.append(
            "| tag | d2 med | d2 p90 | d2 p99 | d2 max | d2_in med | d3 med | d3 p90 | d3 p99 | d3 max | d3_in med | v<=0 | invalid v | invalid atom | val fallback |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for tag in tags:
            c = cons.get((tag, name), {})
            h = coh.get((tag, name), {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        tag,
                        num(c.get("d2_median")),
                        num(c.get("d2_p90")),
                        num(c.get("d2_p99")),
                        num(c.get("d2_max")),
                        num(c.get("d2_in_median")),
                        num(c.get("d3_median")),
                        num(c.get("d3_p90")),
                        num(c.get("d3_p99")),
                        num(c.get("d3_max")),
                        num(c.get("d3_in_median")),
                        num(h.get("frac_v_le_0"), ".4f"),
                        num(h.get("frac_invalid_v"), ".4f"),
                        num(h.get("frac_invalid_atom"), ".4f"),
                        num(h.get("frac_fallback"), ".4f"),
                    ]
                )
                + " |"
            )
            two = samp_ix.get((tag, name, "twopois"), {})
            flat.append(
                {
                    "tag": tag,
                    "dist": name,
                    "d2_median": c.get("d2_median"),
                    "d2_p90": c.get("d2_p90"),
                    "d2_p99": c.get("d2_p99"),
                    "d2_max": c.get("d2_max"),
                    "d3_median": c.get("d3_median"),
                    "d3_p90": c.get("d3_p90"),
                    "d3_p99": c.get("d3_p99"),
                    "d3_max": c.get("d3_max"),
                    "frac_v_le_0": h.get("frac_v_le_0"),
                    "frac_invalid_v": h.get("frac_invalid_v"),
                    "frac_invalid_atom": h.get("frac_invalid_atom"),
                    "frac_val_fallback": h.get("frac_fallback"),
                    "twopois_wd": two.get("wd"),
                    "twopois_tv": two.get("tv"),
                    "twopois_overflow": two.get("overflow"),
                    "twopois_runaway": two.get("runaway"),
                    "twopois_fallback": two.get("fallback"),
                }
            )
        lines.append("")
        lines.append(
            "| tag | Poisson WD/TV | NB WD/TV | TwoPois WD/TV | overflow pois/nb/two | runaway pois/nb/two | TwoPois fallback |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for tag in tags:
            cells = []
            overs = []
            runs = []
            for kernel in ("poisson", "nb", "twopois"):
                row = samp_ix.get((tag, name, kernel))
                if row is None:
                    cells.append("nan")
                    overs.append("nan")
                    runs.append("nan")
                    continue
                cells.append(f"{num(row.get('wd'), '.4f')}/{num(row.get('tv'), '.4f')}")
                overs.append(num(row.get("overflow"), ".4f"))
                runs.append(num(row.get("runaway"), ".4f"))
            two = samp_ix.get((tag, name, "twopois"), {})
            lines.append(
                f"| {tag} | {cells[0]} | {cells[1]} | {cells[2]} | {' / '.join(overs)} | {' / '.join(runs)} | {num(two.get('fallback'), '.4f')} |"
            )
        lines.append("")
        lines.append(
            "| tag | k | logS med | abs logS med | abs logS p99 | logm med | abs logm med | abs logm p99 |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for tag in tags:
            for k in (1, 2, 3):
                r = ratio_ix.get((tag, name, k, "all"), {})
                m = mom_ix.get((tag, name, k, "all"), {})
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            tag,
                            str(k),
                            num(r.get("median")),
                            num(r.get("abs_median")),
                            num(r.get("abs_p99")),
                            num(m.get("median")),
                            num(m.get("abs_median")),
                            num(m.get("abs_p99")),
                        ]
                    )
                    + " |"
                )
                err_flat.append(
                    {
                        "tag": tag,
                        "dist": name,
                        "k": k,
                        "logS_median": r.get("median"),
                        "logS_abs_median": r.get("abs_median"),
                        "logS_abs_p90": r.get("abs_p90"),
                        "logS_abs_p99": r.get("abs_p99"),
                        "logS_abs_max": r.get("abs_max"),
                        "logm_median": m.get("median"),
                        "logm_abs_median": m.get("abs_median"),
                        "logm_abs_p90": m.get("abs_p90"),
                        "logm_abs_p99": m.get("abs_p99"),
                        "logm_abs_max": m.get("abs_max"),
                    }
                )
        lines.append("")

    (out / "comparison.md").write_text("\n".join(lines))
    _write_table(out / "comparison.txt", flat)
    _write_table(out / "comparison_errors.txt", err_flat)
    return out / "comparison.md"


def _write_table(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    lines = [" ".join(keys)]
    for row in rows:
        bits = []
        for key in keys:
            val = row.get(key)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                bits.append("nan")
            elif isinstance(val, float):
                bits.append(f"{val:.6g}")
            else:
                bits.append(str(val))
        lines.append(" ".join(bits))
    path.write_text("\n".join(lines) + "\n")
