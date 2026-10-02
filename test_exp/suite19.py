"""Eval and sample dyn_norm consistency. Soft and anchored are new.

dyn_norm, dyn_norm_proj, hard, and the equal-weight soft/anchored runs
are copied from experiments 17 and 18. They are not resampled.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from cons_eval import (  # noqa: E402
    draw_val,
    eval_one,
    frac,
    oracle_all,
    prepare,
    q_abs,
    row_from_q,
    write_table,
)
from cons_tables import (  # noqa: E402
    NAMES,
    ckpt_of,
    dyn_norm_tag_specs,
    write_comparison,
)
from loss.oracle import BIN_LABELS, gamma_bin_index, support_logp  # noqa: E402
from sample_raw_cons import collect_metrics, git_rev, run_sample  # noqa: E402
from sample_tests import load_model  # noqa: E402
from train import _qstats  # noqa: E402

RESULTS = HERE / "results" / "19_ratio_cons_dyn_norm"
PREV = HERE / "results" / "18_ratio_consistency"
COPY = {
    "dyn_norm": "dyn_norm",
    "dyn_norm_proj": "dyn_norm_proj",
    "hard": "hard",
    "soft": "eq_soft",
    "anchored": "eq_anchored",
}
PREAMBLE = [
    "# Ratio consistency, dyn_norm weights",
    "",
    "New runs: soft and anchored. Data loss is dyn_norm, lambda_cons = 1.",
    "w = (h, h^2/2, h^3/6) / (h + h^2/2 + h^3/6), quadratic T=100.",
    "dyn_norm WD/TV is the experiment 17 sample.",
    "dyn_norm_proj, hard, eq_soft, and eq_anchored are the experiment 18 samples.",
    "eq_soft and eq_anchored used equal weights L1+L2+L3.",
    "hard is L1 only. It was not retrained.",
    "|delta| is the residual of the log-ratios used for moments.",
    "delta_in is that residual before projection.",
    "logS and logm are signed log(hat/star) on the shared val draw. abs is |log(hat/star)|.",
    "val fallback is the TwoPois mask on those val states. sample fallback is the reverse chain.",
]


def parse_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--kernels", default="poisson,nb,twopois")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--n-sample", type=int, default=50000)
    p.add_argument("--sample-batch", type=int, default=4096)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", default="all", choices=("all", "eval", "sample", "tables"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def split_csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def copied_rows(filename, names):
    path = PREV / filename
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    rows = []
    for row in json.loads(path.read_text()):
        src = row.get("tag", row.get("weight"))
        if src not in COPY or row.get("dist") not in names:
            continue
        item = dict(row)
        item["tag"] = COPY[src]
        item["source"] = item.get("source") or "18_ratio_consistency"
        rows.append(item)
    return rows


def eval_new(names, specs, device, out, seed, batch):
    cons_rows, ratio_rows, mom_rows, coh_rows = [], [], [], []
    fresh = [s for s in specs if s["sample"]]
    for name in names:
        base = ckpt_of("soft", name, "raw_cons_dyn_norm")
        _, base_args = load_model(base, device)
        print(f"draw {name} n_val seed={seed}", flush=True)
        z, gamma = draw_val(name, device, seed, base_args.lbd, base_args.t_eps)
        xs, logp = support_logp(name, device)
        stars = oracle_all(z, gamma, xs, logp, device, batch)
        idx = gamma_bin_index(gamma.reshape(-1)).cpu().numpy()
        del base_args
        for spec in fresh:
            ckpt = ckpt_of(spec["kind"], name, spec["family"])
            model, args = load_model(ckpt, device)
            prepare(spec["kind"], spec["mode"], args, name)
            print(f"eval {spec['tag']} {name} mode={spec['mode']}", flush=True)
            got = eval_one(model, args, z, gamma, stars, device, batch)
            d2s, d3s = _qstats(got["d2"]), _qstats(got["d3"])
            d2i, d3i = _qstats(got["d2_in"]), _qstats(got["d3_in"])
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
    cons_rows.extend(copied_rows("consistency.json", names))
    coh_rows.extend(copied_rows("coherence.json", names))
    ratio_rows.extend(copied_rows("ratio_errors.json", names))
    mom_rows.extend(copied_rows("moment_errors.json", names))
    (out / "consistency.json").write_text(json.dumps(cons_rows, indent=2) + "\n")
    (out / "ratio_errors.json").write_text(json.dumps(ratio_rows, indent=2) + "\n")
    (out / "moment_errors.json").write_text(json.dumps(mom_rows, indent=2) + "\n")
    (out / "coherence.json").write_text(json.dumps(coh_rows, indent=2) + "\n")
    write_table(out / "consistency.txt", cons_rows)
    write_table(out / "ratio_errors.txt", ratio_rows)
    write_table(out / "moment_errors.txt", mom_rows)
    write_table(out / "coherence.txt", coh_rows)


def main():
    cli = parse_args()
    names = split_csv(cli.names)
    kernels = split_csv(cli.kernels)
    device = torch.device(cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu")
    out = Path(cli.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    specs = dyn_norm_tag_specs()
    fresh = [s for s in specs if s["sample"]]
    if cli.stage in ("all", "eval", "sample"):
        missing = []
        for spec in specs:
            if spec["tag"] in ("eq_soft", "eq_anchored"):
                continue
            for name in names:
                path = ckpt_of(spec["kind"], name, spec["family"])
                if not path.is_file():
                    missing.append(path)
        if missing:
            for path in missing:
                print(f"missing {path}", flush=True)
            raise SystemExit("missing checkpoints")
    if cli.stage in ("all", "eval"):
        eval_new(names, specs, device, out, cli.seed, cli.batch)
        (out / "protocol.txt").write_text(
            "\n".join(
                [
                    "ratio consistency with dyn_norm weights",
                    "new training: soft and anchored, lambda_cons=1, raw_weight=dyn_norm",
                    "hard was not retrained",
                    "dyn_norm, dyn_norm_proj, hard, eq_soft, eq_anchored rows are copied",
                    "val draw for the new tags: t~Unif[t_eps,1], seed 0",
                    f"names: {','.join(names)}",
                    f"device: {device}",
                    "",
                ]
            )
        )
    if cli.stage in ("all", "sample"):
        run_sample(names, kernels, fresh, cli, device, out)
    if cli.stage in ("all", "tables", "sample"):
        rows = collect_metrics(
            names, kernels, fresh, out, cli.force, source="19_ratio_cons_dyn_norm"
        )
        rows.extend(copied_rows("sampling_metrics.json", names))
        rows = [r for r in rows if r.get("kernel") in kernels]
        (out / "sampling_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        write_table(out / "sampling_metrics.txt", rows)
        (out / "config.txt").write_text(
            "\n".join(
                [
                    "dyn_norm ratio consistency sampling",
                    "schedule: quadratic gamma(u)=lambda*u^2, T=100, lambda=100, gamma from 0 to 100",
                    "new models: soft, anchored under experiments/raw_cons_dyn_norm",
                    "copied: dyn_norm from experiment 17; dyn_norm_proj, hard, eq_soft, eq_anchored from experiment 18",
                    f"names: {','.join(names)}",
                    f"kernels: {','.join(kernels)}",
                    f"n_sample: {cli.n_sample}",
                    f"seed: {cli.seed}",
                    "",
                ]
            )
        )
        (out / "git_commit.txt").write_text(git_rev() + "\n")
    if cli.stage in ("all", "tables", "eval", "sample"):
        path = write_comparison(
            out,
            tag_order=[s["tag"] for s in specs],
            preamble=PREAMBLE,
        )
        print(f"tables -> {path}", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
