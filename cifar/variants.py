PARAMS = ("raw_log", "direct_ratio", "log_moment", "root_moment", "direct_moment")


def _row(name, family, param, weight, heads, cons, lam, link):
    return {
        "name": name,
        "family": family,
        "param": param,
        "weight": weight,
        "heads": int(heads),
        "cons": cons,
        "lam": float(lam),
        "link": link,
    }


def _lam_tag(lam):
    if abs(float(lam) - 0.1) < 1e-8:
        return "p0p1"
    if abs(float(lam) - (1.0 / 3.0)) < 1e-6:
        return "p1o3"
    if abs(float(lam) - 1.0) < 1e-8:
        return "p1"
    raise ValueError(lam)


def build():
    rows = []
    for link in ("unbounded", "bounded"):
        rows.append(_row(f"a0_{link}", "mean", "direct", "equal", 1, "none", 0.0, link))
    rows.append(_row("b_equal", "ratio", "raw_log", "equal", 3, "none", 0.0, "unbounded"))
    rows.append(_row("b_stepnorm", "ratio", "raw_log", "dyn_norm", 3, "none", 0.0, "unbounded"))
    for link in ("unbounded", "bounded"):
        for weight in ("equal", "step"):
            for param in PARAMS:
                rows.append(
                    _row(
                        f"c_{weight}_{param}_{link}",
                        "hybrid_ratio",
                        param,
                        weight,
                        3,
                        "none",
                        0.0,
                        link,
                    )
                )
    for link in ("unbounded", "bounded"):
        for weight in ("equal", "step"):
            for param in PARAMS:
                rows.append(
                    _row(
                        f"d_{weight}_{param}_{link}",
                        "hybrid_moment",
                        param,
                        weight,
                        3,
                        "none",
                        0.0,
                        link,
                    )
                )
    anchors = (
        ("ce_raw", "hybrid_ratio", "raw_log", "equal"),
        ("cs_raw", "hybrid_ratio", "raw_log", "step"),
        ("ce_root", "hybrid_ratio", "root_moment", "equal"),
        ("cs_root", "hybrid_ratio", "root_moment", "step"),
    )
    modes = (
        ("sym", 0.1),
        ("sym", 1.0 / 3.0),
        ("sym", 1.0),
        ("step", 1.0 / 3.0),
        ("anchor", 1.0 / 3.0),
    )
    for tag, family, param, weight in anchors:
        for mode, lam in modes:
            rows.append(
                _row(
                    f"cons_{mode}_{_lam_tag(lam)}_{tag}",
                    family,
                    param,
                    weight,
                    3,
                    mode,
                    lam,
                    "unbounded",
                )
            )
    return rows


A0 = "/home/tomsama/scratch/Discrete-poisson-diffusion-experiments/experiments/cifar/s0/a0_bounded/best_ema.pt"
FOCUS_ROOT = "/home/tomsama/scratch/Discrete-poisson-diffusion-experiments/experiments/cifar/focus/s0"


def _focus(name, family, param, heads, **kw):
    row = _row(name, family, param, kw.pop("weight", "step"), heads, "none", 0.0, "bounded")
    row["init"] = kw.pop("init", "")
    row["freeze"] = list(kw.pop("freeze", []))
    row["higher_only"] = bool(kw.pop("higher_only", False))
    row["higher_scale"] = float(kw.pop("higher_scale", 1.0))
    row["l2_scale"] = float(kw.pop("l2_scale", 1.0))
    row["l3_scale"] = float(kw.pop("l3_scale", 1.0))
    row["kernel"] = str(kw.pop("kernel", "none"))
    row["kernel_lam"] = float(kw.pop("kernel_lam", 0.0))
    row["kernel_only"] = bool(kw.pop("kernel_only", False))
    row["distill_alpha"] = float(kw.pop("distill_alpha", 0.0))
    row["trunk_lr"] = float(kw.pop("trunk_lr", 1.0))
    row["epochs"] = int(kw.pop("epochs", 200))
    if kw:
        raise ValueError(kw)
    return row


def focus_rows():
    warm = f"{FOCUS_ROOT}/c8_warm/ema_final.pt"
    m2 = f"{FOCUS_ROOT}/c9_m2/ema_final.pt"
    m3 = f"{FOCUS_ROOT}/c9_m3/ema_final.pt"
    fr = ["trunk", "h0"]
    return [
        _focus("c1_frozen_m2_rkl", "hybrid_ratio", "root_moment", 2, higher_only=True, freeze=fr, init=A0),
        _focus("c2_frozen_m23_rkl", "hybrid_ratio", "root_moment", 3, higher_only=True, freeze=fr, init=A0),
        _focus("c3_frozen_m2_mkl", "hybrid_moment", "root_moment", 2, higher_only=True, freeze=fr, init=A0),
        _focus("c4_frozen_m23_mkl", "hybrid_moment", "root_moment", 3, higher_only=True, freeze=fr, init=A0),
        _focus("c7_scaled_rkl", "hybrid_ratio", "root_moment", 3, higher_scale=0.01),
        _focus("c8_warm", "hybrid_ratio", "root_moment", 3, higher_only=True, freeze=fr, init=A0, epochs=100),
        _focus("c8_joint", "hybrid_ratio", "root_moment", 3, init=warm, trunk_lr=0.1, epochs=200),
        _focus("c9_m2", "hybrid_ratio", "root_moment", 2, higher_only=True, freeze=fr, init=A0),
        _focus("c9_m3", "hybrid_ratio", "root_moment", 3, higher_only=True, l2_scale=0.0, freeze=fr + ["h1"], init=m2),
        _focus("c9_ft", "hybrid_ratio", "root_moment", 3, higher_only=True, l3_scale=0.1, freeze=fr, init=m3, epochs=40),
        _focus("c10_valid_rkl", "hybrid_ratio", "valid_shape", 3),
        _focus("c11_valid_mkl", "hybrid_moment", "valid_shape", 3),
        _focus("c12_nb_ce", "hybrid_ratio", "valid_shape", 2, kernel="nb", kernel_only=True, freeze=fr, init=A0),
        _focus("c13_twopois_ce", "hybrid_ratio", "valid_shape", 3, kernel="twopois", kernel_only=True, freeze=fr, init=A0),
        _focus("c14_mkl_ce_p0p1", "hybrid_moment", "root_moment", 3, higher_only=True, kernel="twopois", kernel_lam=0.1, freeze=fr, init=A0),
        _focus("c14_mkl_ce_p1", "hybrid_moment", "root_moment", 3, higher_only=True, kernel="twopois", kernel_lam=1.0, freeze=fr, init=A0),
        _focus("c15_mixed", "hybrid_mixed", "root_moment", 3),
        _focus("c16_distill", "hybrid_ratio", "root_moment", 3, higher_only=True, freeze=fr, init=A0, distill_alpha=0.1),
    ]


def focus_sequences():
    rows = focus_rows()
    name_to_i = {row["name"]: i for i, row in enumerate(rows)}
    singles = [
        "c1_frozen_m2_rkl",
        "c2_frozen_m23_rkl",
        "c3_frozen_m2_mkl",
        "c4_frozen_m23_mkl",
        "c7_scaled_rkl",
        "c10_valid_rkl",
        "c11_valid_mkl",
        "c12_nb_ce",
        "c13_twopois_ce",
        "c14_mkl_ce_p0p1",
        "c14_mkl_ce_p1",
        "c15_mixed",
        "c16_distill",
    ]
    seqs = [[name_to_i[name]] for name in singles]
    seqs.append([name_to_i["c8_warm"], name_to_i["c8_joint"]])
    seqs.append([name_to_i["c9_m2"], name_to_i["c9_m3"], name_to_i["c9_ft"]])
    return seqs


def rows_for(group):
    if group == "focus":
        return focus_rows()
    rows = build()
    if group == "ab":
        return [r for r in rows if r["family"] in ("mean", "ratio") and r["cons"] == "none"]
    if group == "c_ub":
        return [r for r in rows if r["family"] == "hybrid_ratio" and r["link"] == "unbounded" and r["cons"] == "none"]
    if group == "c_bd":
        return [r for r in rows if r["family"] == "hybrid_ratio" and r["link"] == "bounded" and r["cons"] == "none"]
    if group == "d_ub":
        return [r for r in rows if r["family"] == "hybrid_moment" and r["link"] == "unbounded" and r["cons"] == "none"]
    if group == "d_bd":
        return [r for r in rows if r["family"] == "hybrid_moment" and r["link"] == "bounded" and r["cons"] == "none"]
    if group == "cons":
        return [r for r in rows if r["name"].startswith("cons_")]
    raise ValueError(group)
