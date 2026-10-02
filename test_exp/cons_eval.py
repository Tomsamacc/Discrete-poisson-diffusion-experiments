"""Offline consistency, ratio, moment, and TwoPois-mask metrics.

One shared val draw per dataset, seed 0. No sampling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from cons_tables import (  # noqa: E402
    NAMES,
    ckpt_of,
    missing_ckpts,
    tag_specs,
    write_comparison,
)
from loss.loss import moments_from_ratio_pred, rising_factorial  # noqa: E402
from loss.oracle import BIN_LABELS, gamma_bin_index, oracle_moments, support_logp  # noqa: E402
from sample import log_s_eval, two_pois_moment_flags  # noqa: E402
from sample_tests import load_model  # noqa: E402
from train import _qstats, load_counts, q_sample  # noqa: E402

RESULTS = HERE / "results" / "18_ratio_consistency"


def parse_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--skip-dyn-norm", action="store_true")
    return p.parse_args()


def split_csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def prepare(kind, mode, args, name):
    args.ratio = True
    args.moment_param = "raw"
    args.ratio_loss = "raw"
    args.consistency = mode
    prior = getattr(args, "prior_moments", None)
    if prior is None or len(list(prior)) < 3:
        x = load_counts(ROOT / "data" / name / "train.npy").reshape(-1).double()
        args.prior_moments = [float(x.pow(k).mean()) for k in range(1, 4)]
    if kind == "hard":
        args.moment_loss = "raw_ratio"
    else:
        args.moment_loss = "raw_multi"
    return args


def draw_val(name, device, seed, lbd, t_eps):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    x = load_counts(ROOT / "data" / name / "val.npy")
    loader = DataLoader(TensorDataset(x), batch_size=4096, shuffle=False)
    zs, gs = [], []
    for (xb,) in loader:
        xb = xb.to(device)
        t = torch.rand(xb.shape[0], device=device) * (1.0 - float(t_eps)) + float(t_eps)
        z = q_sample(xb, t, float(lbd))
        zs.append(z.detach().cpu())
        gs.append((t * float(lbd)).detach().cpu())
    return torch.cat(zs, 0), torch.cat(gs, 0)


@torch.no_grad()
def oracle_all(z, gamma, xs, logp, device, batch):
    parts = [[], [], []]
    n = z.shape[0]
    for i in range(0, n, batch):
        zz = z[i : i + batch].to(device)
        gg = gamma[i : i + batch].to(device)
        moms = oracle_moments(zz, gg, xs, logp, k_max=3)
        for k in range(3):
            parts[k].append(moms[k].reshape(-1).detach().cpu())
    return [torch.cat(p, 0) for p in parts]


def q_abs(arr):
    signed = _qstats(arr)
    absolute = _qstats(np.abs(np.asarray(arr, dtype=np.float64)))
    return signed, absolute


def row_from_q(tag, name, k, bin_name, signed, absolute):
    return {
        "tag": tag,
        "dist": name,
        "k": int(k),
        "bin": bin_name,
        "n": signed["n"],
        "median": signed["median"],
        "p90": signed["p90"],
        "p99": signed["p99"],
        "max": signed["max"],
        "abs_median": absolute["median"],
        "abs_p90": absolute["p90"],
        "abs_p99": absolute["p99"],
        "abs_max": absolute["max"],
    }


@torch.no_grad()
def eval_one(model, args, z, gamma, stars, device, batch):
    n = int(z.shape[0])
    d2 = np.empty(n)
    d3 = np.empty(n)
    d2_in = np.empty(n)
    d3_in = np.empty(n)
    ratio = np.empty((3, n))
    mom = np.empty((3, n))
    counts = {"v_le_0": 0, "thin": 0, "bad_atom": 0, "extreme": 0, "fallback": 0}
    for i in range(0, n, batch):
        sl = slice(i, i + batch)
        zz = z[sl].to(device)
        gg = gamma[sl].to(device)
        log_s, r2, r3, i2, i3 = log_s_eval(model, zz, gg, args)
        moms = moments_from_ratio_pred(log_s, zz, gg, args)
        flags = two_pois_moment_flags(moms[0], moms[1], moms[2])
        for key, attr in (
            ("v_le_0", "v_le_0"),
            ("thin", "thin"),
            ("bad_atom", "bad_atom"),
            ("extreme", "extreme"),
            ("fallback", "fallback"),
        ):
            counts[key] += int(flags[attr].sum().item())
        log_d = log_s.double()
        D = rising_factorial(zz, 3)
        g = gg.reshape(-1).double().clamp_min(1e-12)
        d2[sl] = r2.detach().abs().double().cpu().numpy()
        d3[sl] = r3.detach().abs().double().cpu().numpy()
        d2_in[sl] = i2.detach().abs().double().cpu().numpy()
        d3_in[sl] = i3.detach().abs().double().cpu().numpy()
        for k in range(3):
            s_hat = log_d[:, k].exp()
            m_star = stars[k][sl].to(device).double()
            s_star = m_star * g.pow(k + 1) / D[:, k].clamp_min(1e-12)
            ratio[k, sl] = (
                s_hat.clamp_min(1e-12).log() - s_star.clamp_min(1e-12).log()
            ).cpu().numpy()
            m_hat = moms[k].reshape(-1).double().clamp_min(1e-12)
            mom[k, sl] = (m_hat.log() - m_star.clamp_min(1e-12).log()).cpu().numpy()
    return {
        "n": n,
        "d2": d2,
        "d3": d3,
        "d2_in": d2_in,
        "d3_in": d3_in,
        "ratio": ratio,
        "mom": mom,
        "counts": counts,
    }


def frac(num, den):
    if not den:
        return float("nan")
    return float(num) / float(den)


def write_table(path, rows):
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


def main():
    cli = parse_args()
    names = split_csv(cli.names)
    device = torch.device(cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu")
    out = Path(cli.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    specs = tag_specs(skip_dyn_norm=cli.skip_dyn_norm)
    missing = missing_ckpts(names, skip_dyn_norm=cli.skip_dyn_norm)
    if missing:
        for path in missing:
            print(f"missing {path}", flush=True)
        raise SystemExit("train the missing consistency runs, then rerun this eval")

    cons_rows, ratio_rows, mom_rows, coh_rows = [], [], [], []
    for name in names:
        base_ckpt = ckpt_of("equal", name)
        _, base_args = load_model(base_ckpt, device)
        print(f"draw {name} n_val seed={cli.seed}", flush=True)
        z, gamma = draw_val(name, device, cli.seed, base_args.lbd, base_args.t_eps)
        xs, logp = support_logp(name, device)
        stars = oracle_all(z, gamma, xs, logp, device, cli.batch)
        idx = gamma_bin_index(gamma.reshape(-1)).cpu().numpy()
        del base_args
        for spec in specs:
            ckpt = ckpt_of(spec["kind"], name, spec.get("family"))
            model, args = load_model(ckpt, device)
            prepare(spec["kind"], spec["mode"], args, name)
            print(f"eval {spec['tag']} {name} mode={spec['mode']}", flush=True)
            got = eval_one(model, args, z, gamma, stars, device, cli.batch)
            d2s = _qstats(got["d2"])
            d3s = _qstats(got["d3"])
            d2i = _qstats(got["d2_in"])
            d3i = _qstats(got["d3_in"])
            cons_rows.append(
                {
                    "tag": spec["tag"],
                    "dist": name,
                    "n": got["n"],
                    "d2_median": d2s["median"],
                    "d2_p90": d2s["p90"],
                    "d2_p99": d2s["p99"],
                    "d2_max": d2s["max"],
                    "d3_median": d3s["median"],
                    "d3_p90": d3s["p90"],
                    "d3_p99": d3s["p99"],
                    "d3_max": d3s["max"],
                    "d2_in_median": d2i["median"],
                    "d2_in_p90": d2i["p90"],
                    "d2_in_p99": d2i["p99"],
                    "d2_in_max": d2i["max"],
                    "d3_in_median": d3i["median"],
                    "d3_in_p90": d3i["p90"],
                    "d3_in_p99": d3i["p99"],
                    "d3_in_max": d3i["max"],
                }
            )
            counts = got["counts"]
            coh_rows.append(
                {
                    "tag": spec["tag"],
                    "dist": name,
                    "n": got["n"],
                    "frac_v_le_0": frac(counts["v_le_0"], got["n"]),
                    "frac_invalid_v": frac(counts["thin"], got["n"]),
                    "frac_invalid_atom": frac(counts["bad_atom"], got["n"]),
                    "frac_fallback": frac(counts["fallback"], got["n"]),
                    "frac_extreme_w": frac(counts["extreme"], got["n"]),
                }
            )
            for k in range(3):
                signed, absolute = q_abs(got["ratio"][k])
                ratio_rows.append(row_from_q(spec["tag"], name, k + 1, "all", signed, absolute))
                signed, absolute = q_abs(got["mom"][k])
                mom_rows.append(row_from_q(spec["tag"], name, k + 1, "all", signed, absolute))
                for b, lab in enumerate(BIN_LABELS):
                    mask = idx == b
                    signed, absolute = q_abs(got["ratio"][k, mask])
                    ratio_rows.append(row_from_q(spec["tag"], name, k + 1, lab, signed, absolute))
                    signed, absolute = q_abs(got["mom"][k, mask])
                    mom_rows.append(row_from_q(spec["tag"], name, k + 1, lab, signed, absolute))
            del model

    (out / "consistency.json").write_text(json.dumps(cons_rows, indent=2) + "\n")
    (out / "ratio_errors.json").write_text(json.dumps(ratio_rows, indent=2) + "\n")
    (out / "moment_errors.json").write_text(json.dumps(mom_rows, indent=2) + "\n")
    (out / "coherence.json").write_text(json.dumps(coh_rows, indent=2) + "\n")
    write_table(out / "consistency.txt", cons_rows)
    write_table(out / "ratio_errors.txt", ratio_rows)
    write_table(out / "moment_errors.txt", mom_rows)
    write_table(out / "coherence.txt", coh_rows)
    (out / "protocol.txt").write_text(
        "\n".join(
            [
                "ratio consistency, first stage",
                "new training: soft lambda_cons=1, anchored lambda_cons=1, hard one-head",
                "data loss of those runs: L1+L2+L3, full Bregman, equal weights",
                "no consistency+dyn_norm training",
                "no lambda sweep",
                "equal and dyn_norm checkpoints are reused",
                "equal_proj and dyn_norm_proj: Euclidean projection, no training",
                "val draw: t~Unif[t_eps,1], seed 0, shared across tags",
                "delta: absolute residual of the log-ratios used for moments",
                "delta_in: absolute residual before projection",
                "logS and logm rows: signed log(hat/star); abs_* is |log(hat/star)|",
                "coherence uses the moments the sampler would use on that val draw",
                f"names: {','.join(names)}",
                f"device: {device}",
                "",
            ]
        )
    )
    write_comparison(out)
    print(f"eval done -> {out}", flush=True)


if __name__ == "__main__":
    main()
