import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sweep1d_run import sample_one

FOCUS = ROOT / "experiments" / "focus1d"
SWEEP = ROOT / "experiments" / "sweep1d"
OLD = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
TRIPLE = ("poisson", "nb", "twopois")
PAIR = ("poisson", "nb")


def kernels_of(tag):
    if tag.startswith("a0"):
        return ("poisson",)
    if tag.endswith("_m2") or "_m2_" in tag:
        return PAIR
    return TRIPLE


def add_run(jobs, run, data_name):
    if not (run / "latest.pt").is_file():
        return
    pairs = [("final", run / "latest.pt")]
    if (run / "best_h.pt").is_file():
        pairs.append(("best_h", run / "best_h.pt"))
    elif (run / "best_l1.pt").is_file():
        pairs.append(("best_l1", run / "best_l1.pt"))
    kernels = kernels_of(run.name)
    for name, ckpt in pairs:
        dest = run / "sample_lin" / name
        jobs.append((str(ckpt), str(dest), data_name, kernels))


def build_jobs():
    jobs = []
    if FOCUS.is_dir():
        for data_dir in sorted(p for p in FOCUS.iterdir() if p.is_dir()):
            for run in sorted(p for p in data_dir.iterdir() if p.is_dir()):
                add_run(jobs, run, data_dir.name)
    for name in OLD:
        add_run(jobs, SWEEP / name / "a0_kl", name)
    return jobs


def one(item):
    ckpt, dest, data_name, kernels = item
    sample_one(ckpt, dest, data_name, kernels, schedule="linear")
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if not __import__("torch").cuda.is_available():
        raise SystemExit("cuda is not available")
    jobs = build_jobs()
    print(f"linear samples {len(jobs)}", flush=True)
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    import concurrent.futures

    failed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futs = [pool.submit(one, job) for job in jobs]
        for fut in concurrent.futures.as_completed(futs):
            try:
                print("done", fut.result(), flush=True)
            except Exception as exc:
                failed += 1
                print("fail", exc, flush=True)
    if failed:
        raise SystemExit(f"{failed} samples failed")


if __name__ == "__main__":
    main()
