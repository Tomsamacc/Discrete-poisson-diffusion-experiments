"""Summarize P(K>3) from the existing oracle dynamics table. No training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "results" / "15_diag" / "dynamics.json"
OUT = HERE / "results" / "15_diag" / "tail_summary.txt"


def qstats(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return {
        "n_steps": int(a.size),
        "tail_mean": float(a.mean()),
        "tail_median": float(np.median(a)),
        "tail_p90": float(np.quantile(a, 0.90)),
        "tail_max": float(a.max()),
        "frac_steps_gt_0.1": float(np.mean(a > 0.1)),
        "frac_steps_gt_0.5": float(np.mean(a > 0.5)),
    }


def phase_mean(rows, lo, hi):
    vals = [r["tail_gt3"] for r in rows if lo <= int(r["step"]) < hi]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def write_table(path, rows):
    keys = list(rows[0].keys())
    lines = [" ".join(keys)]
    for row in rows:
        bits = []
        for key in keys:
            val = row[key]
            if isinstance(val, float):
                bits.append(f"{val:.6g}" if np.isfinite(val) else "nan")
            else:
                bits.append(str(val))
        lines.append(" ".join(bits))
    path.write_text("\n".join(lines) + "\n")


def main():
    rows = json.loads(SRC.read_text())
    groups = {}
    for row in rows:
        key = (row["dist"], row["sched"], row["source"])
        groups.setdefault(key, []).append(row)
    out = []
    for (dist, sched, source), items in groups.items():
        items = sorted(items, key=lambda r: int(r["step"]))
        tail = [r["tail_gt3"] for r in items]
        st = qstats(tail)
        state_frac = np.mean([r["frac_tail_gt3_gt_0.1"] for r in items])
        out.append(
            {
                "dist": dist,
                "sched": sched,
                "source": source,
                **st,
                "mean_state_frac_tail_gt_0.1": float(state_frac),
                "tail_steps_0_9": phase_mean(items, 0, 10),
                "tail_steps_10_49": phase_mean(items, 10, 50),
                "tail_steps_50_99": phase_mean(items, 50, 100),
                "q0_mean": float(np.mean([r["q0"] for r in items])),
                "q1_mean": float(np.mean([r["q1"] for r in items])),
                "q2_mean": float(np.mean([r["q2"] for r in items])),
                "q3_mean": float(np.mean([r["q3"] for r in items])),
            }
        )
    write_table(OUT, out)
    print(f"wrote {OUT} rows={len(out)}")


if __name__ == "__main__":
    main()
