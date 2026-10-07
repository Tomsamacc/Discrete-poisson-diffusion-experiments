import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

OUT = ROOT / "experiments" / "sweep1d"
DATASETS = (
    ("gamma_ltj", "data/gamma_ltj"),
    ("nb", "data/nb"),
    ("poismix_mod", "data/poismix_mod"),
    ("poissmix3", "data/poissmix3"),
)
PARAMS = ("raw_log", "direct_ratio", "log_moment", "root_moment", "direct_moment")
WEIGHTS = (("equal", "equal"), ("dyn", "step"))


def base_rows():
    rows = [
        {"tag": "a0_kl", "kind": "single", "moment_loss": "moment"},
        {"tag": "a1_mse", "kind": "single", "moment_loss": "mse"},
    ]
    for param in PARAMS:
        for weight, wname in WEIGHTS:
            rows.append(
                {
                    "tag": f"b_{param}_{wname}",
                    "kind": "hm",
                    "hm_param": param,
                    "hybrid_weight": weight,
                    "higher_target": "ratio_kl",
                    "higher_scale": 1.0,
                }
            )
    for param in ("direct_ratio", "root_moment"):
        for weight, wname in WEIGHTS:
            rows.append(
                {
                    "tag": f"c_{param}_mse_{wname}",
                    "kind": "hm",
                    "hm_param": param,
                    "hybrid_weight": weight,
                    "higher_target": "ratio_mse",
                    "higher_scale": 1.0,
                }
            )
    for target, tname in (("moment_kl", "kl"), ("moment_mse", "mse")):
        for weight, wname in WEIGHTS:
            rows.append(
                {
                    "tag": f"d_root_moment_{tname}_{wname}",
                    "kind": "hm",
                    "hm_param": "root_moment",
                    "hybrid_weight": weight,
                    "higher_target": target,
                    "higher_scale": 1.0,
                }
            )
    for param, tag, mode in (
        ("direct_ratio", "e_direct_ratio_freeze", "freeze"),
        ("root_moment", "e_root_moment_freeze", "freeze"),
        ("direct_ratio", "e_direct_ratio_detach", "detach"),
        ("root_moment", "e_root_moment_detach", "detach"),
    ):
        rows.append(
            {
                "tag": tag,
                "kind": "hm",
                "hm_param": param,
                "hybrid_weight": "dyn",
                "higher_target": "ratio_kl",
                "higher_scale": 1.0,
                "needs_a0": True,
                "freeze": mode == "freeze",
                "detach": mode == "detach",
            }
        )
    for param, tag in (
        ("direct_ratio", "f_direct_ratio_staged"),
        ("root_moment", "f_root_moment_staged"),
    ):
        rows.append(
            {
                "tag": tag,
                "kind": "staged",
                "hm_param": param,
                "hybrid_weight": "dyn",
                "higher_target": "ratio_kl",
                "higher_scale": 1.0,
                "needs_a0": True,
            }
        )
    for param, tag, scale in (
        ("direct_ratio", "g_direct_ratio_s0p1", 0.1),
        ("root_moment", "g_root_moment_s0p1", 0.1),
        ("root_moment", "g_root_moment_s0p01", 0.01),
    ):
        rows.append(
            {
                "tag": tag,
                "kind": "hm",
                "hm_param": param,
                "hybrid_weight": "dyn",
                "higher_target": "ratio_kl",
                "higher_scale": scale,
            }
        )
    return rows


def build_jobs():
    rows = base_rows()
    if len(rows) != 29:
        raise SystemExit(f"expected 29 configs, got {len(rows)}")
    jobs = []
    for data_name, data in DATASETS:
        for row in rows:
            job = dict(row)
            job["data_name"] = data_name
            job["data"] = data
            job["out"] = str(OUT / data_name / row["tag"])
            job["a0"] = str(OUT / data_name / "a0_kl")
            jobs.append(job)
    if len(jobs) != 116:
        raise SystemExit(f"expected 116 runs, got {len(jobs)}")
    return jobs


def argv(job, out, epochs, init, freeze, detach, trunk_mult):
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "train.py"),
        "-c",
        str(ROOT / "config_scale.yml"),
        "--data",
        job["data"],
        "--out_dir",
        str(out),
        "--device",
        "cuda",
        "--epochs",
        str(epochs),
        "--batch_size",
        "256",
        "--lr",
        "0.001",
        "--seed",
        "0",
        "--lbd",
        "100",
        "--scale",
        "true",
        "--save_best_l1",
        "true",
        "--weight_steps",
        "100",
        "--val_every",
        "10",
    ]
    if job["kind"] == "single":
        cmd.extend(
            [
                "--moment_k",
                "1",
                "--moment_param",
                "direct",
                "--moment_loss",
                job["moment_loss"],
                "--ratio",
                "false",
            ]
        )
        return cmd
    cmd.extend(
        [
            "--moment_loss",
            "hm",
            "--hm_param",
            job["hm_param"],
            "--hybrid_weight",
            job["hybrid_weight"],
            "--higher_target",
            job["higher_target"],
            "--higher_scale",
            str(job["higher_scale"]),
            "--k_max",
            "3",
        ]
    )
    if freeze:
        cmd.extend(["--freeze_m1", "true"])
    if detach:
        cmd.extend(["--detach_higher", "true"])
    if float(trunk_mult) != 1.0:
        cmd.extend(["--trunk_lr_mult", str(trunk_mult)])
    if init:
        cmd.extend(["--init_ckpt", str(init)])
    return cmd


def call(cmd):
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def wait_file(path, timeout):
    path = Path(path)
    start = time.time()
    while not path.is_file():
        if time.time() - start > float(timeout):
            raise SystemExit(f"missing {path}")
        time.sleep(20)


def a0_ckpt(job):
    root = Path(job["a0"])
    best = root / "best.pt"
    if best.is_file():
        return best
    latest = root / "latest.pt"
    if latest.is_file():
        return latest
    raise SystemExit(f"no A0 checkpoint in {root}")


def kernels_of(job):
    if job["kind"] == "single":
        return ("poisson",)
    return ("poisson", "nb", "twopois")


def sample_one(ckpt, dest, data_name, kernels, schedule="quad"):
    import numpy as np
    import torch

    from sample import generate, make_gammas
    from sample_tests import load_model
    from test import plot_experiment

    dest = Path(dest)
    metrics_path = dest / "metrics.json"
    if metrics_path.is_file() and all((dest / f"samples_{k}.npy").is_file() for k in kernels):
        return
    device = torch.device("cuda")
    model, args = load_model(ckpt, device)
    args.sample_batch = 4096
    args.normalize = None
    if schedule == "linear":
        gammas = make_gammas("uniform", 100, 0.0, float(args.lbd), device, power=1.0)
    elif schedule == "quad":
        gammas = make_gammas("quad", 100, 0.0, float(args.lbd), device, power=2.0)
    else:
        raise ValueError(schedule)
    dest.mkdir(parents=True, exist_ok=True)
    for kernel in kernels:
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        print(f"sample {kernel} {ckpt}", flush=True)
        x = generate(model, 50000, kernel, args, device, gammas=gammas)
        np.save(dest / f"samples_{kernel}.npy", x)
    table = plot_experiment(dest, name=data_name, kernels=kernels)
    metrics_path.write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(table), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def sample_run(job, out):
    kernels = kernels_of(job)
    final = Path(out) / "latest.pt"
    best = Path(out) / "best_l1.pt"
    if not final.is_file():
        raise SystemExit(f"missing {final}")
    sample_one(final, Path(out) / "sample" / "final", job["data_name"], kernels)
    if best.is_file():
        sample_one(best, Path(out) / "sample" / "best_l1", job["data_name"], kernels)


def samples_ready(job, out):
    kernels = kernels_of(job)
    final = Path(out) / "sample" / "final" / "metrics.json"
    if not final.is_file():
        return False
    best_pt = Path(out) / "best_l1.pt"
    best = Path(out) / "sample" / "best_l1" / "metrics.json"
    if best_pt.is_file() and not best.is_file():
        return False
    return all((Path(out) / "sample" / "final" / f"samples_{k}.npy").is_file() for k in kernels)


def run_job(job):
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    label = f"{job['data_name']}/{job['tag']}"
    try:
        if (out / "finished.json").is_file() and samples_ready(job, out):
            print(f"skip {label}", flush=True)
            return label
        if job.get("needs_a0"):
            wait_file(Path(job["a0"]) / "finished.json", 3 * 3600)
            init = a0_ckpt(job)
        else:
            init = ""
        if job["kind"] == "staged":
            warm = out / "warmup"
            if not (warm / "finished.json").is_file():
                call(argv(job, warm, 40, init, True, False, 1.0))
            if not (out / "finished.json").is_file():
                call(argv(job, out, 200, warm / "latest.pt", False, False, 0.1))
        elif not (out / "finished.json").is_file():
            call(argv(job, out, 200, init, bool(job.get("freeze")), bool(job.get("detach")), 1.0))
        sample_run(job, out)
        print(f"done {label}", flush=True)
        return label
    except Exception as exc:
        (out / "error.txt").write_text(repr(exc) + "\n")
        print(f"fail {label} {exc}", flush=True)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--ntasks", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    jobs = build_jobs()
    if args.list:
        for i, job in enumerate(jobs):
            print(f"{i:03d} {job['data_name']} {job['tag']}")
        return
    if not __import__("torch").cuda.is_available():
        raise SystemExit("cuda is not available")
    mine = [job for i, job in enumerate(jobs) if i % int(args.ntasks) == int(args.task)]
    mine.sort(key=lambda job: (2 if job.get("needs_a0") else 0 if job["tag"] == "a0_kl" else 1, job["tag"]))
    print(f"task {args.task} runs {len(mine)} / {len(jobs)} workers {args.workers}", flush=True)
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    import concurrent.futures

    workers = max(1, int(args.workers))
    failed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_job, job) for job in mine]
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result()
            except Exception:
                failed += 1
    if failed:
        raise SystemExit(f"{failed} runs failed")


if __name__ == "__main__":
    main()
