import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from focus1d_run import PAIR, row, run_job

DATASETS = (
    ("gamma_ltj", "data/gamma_ltj"),
    ("nb", "data/nb"),
    ("poismix_mod", "data/poismix_mod"),
    ("poissmix3", "data/poissmix3"),
    ("zip", "data/zip"),
    ("yule_simon", "data/yule_simon"),
)
OUT = ROOT / "experiments" / "focus1d"


def build_jobs():
    rows = []
    for data_name, data in DATASETS:
        for seed in (1, 2):
            init = str(OUT / data_name / f"a0_s{seed}" / "latest.pt")
            rows.append(
                row(
                    data_name,
                    data,
                    f"p0_root_rkl_m2_a0s{seed}",
                    "freeze",
                    PAIR,
                    "h",
                    hm_param="root_moment",
                    higher_target="ratio_kl",
                    higher_scale=1.0,
                    k_max=2,
                    seed=0,
                    init=init,
                )
            )
    if len(rows) != 12:
        raise SystemExit(f"expected 12 runs, got {len(rows)}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not __import__("torch").cuda.is_available():
        raise SystemExit("cuda is not available")
    jobs = build_jobs()
    for job in jobs:
        if not Path(job["init"]).is_file():
            raise SystemExit(f"missing {job['init']}")
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    import concurrent.futures

    failed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futs = [pool.submit(run_job, job) for job in jobs]
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result()
            except Exception:
                failed += 1
    if failed:
        raise SystemExit(f"{failed} runs failed")


if __name__ == "__main__":
    main()
