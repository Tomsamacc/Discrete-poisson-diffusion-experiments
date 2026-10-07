import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sweep1d_run import call, sample_one, wait_file

OUT = ROOT / "experiments" / "focus1d"
A0 = ROOT / "experiments" / "sweep1d"
DATASETS = (
    ("gamma_ltj", "data/gamma_ltj"),
    ("nb", "data/nb"),
    ("poismix_mod", "data/poismix_mod"),
    ("poissmix3", "data/poissmix3"),
    ("zip", "data/zip"),
    ("yule_simon", "data/yule_simon"),
)
READY_A0 = {"gamma_ltj", "nb", "poismix_mod", "poissmix3"}
EPOCHS = "50,100,150,200"
TRIPLE = ("poisson", "nb", "twopois")
PAIR = ("poisson", "nb")


def row(data_name, data, tag, kind, kernels, select, **extra):
    job = {
        "data_name": data_name,
        "data": data,
        "tag": tag,
        "kind": kind,
        "kernels": kernels,
        "select": select,
        "seed": extra.pop("seed", 0),
        "init": extra.pop("init", ""),
        "out": str(OUT / data_name / tag),
    }
    job.update(extra)
    return job


def build_jobs():
    rows = []
    specs = (
        ("p0_root_rkl_m2", "root_moment", "ratio_kl", 1.0, 2, PAIR),
        ("p0_root_rkl_m23", "root_moment", "ratio_kl", 1.0, 3, TRIPLE),
        ("p0_root_mkl_m2", "root_moment", "moment_kl", 1.0, 2, PAIR),
        ("p0_root_mkl_m23", "root_moment", "moment_kl", 1.0, 3, TRIPLE),
        ("p0_dratio_rkl_m23", "direct_ratio", "ratio_kl", 1.0, 3, TRIPLE),
    )
    for data_name, data in DATASETS:
        if data_name in READY_A0:
            continue
        rows.append(
            row(
                data_name,
                data,
                "a0_s0",
                "a0",
                ("poisson",),
                "l1",
                seed=0,
            )
        )
    for data_name, data in DATASETS:
        if data_name in READY_A0:
            init = str(A0 / data_name / "a0_kl" / "latest.pt")
            wait_a0 = False
        else:
            init = str(OUT / data_name / "a0_s0" / "latest.pt")
            wait_a0 = True
        for tag, param, target, scale, kmax, kernels in specs:
            rows.append(
                row(
                    data_name,
                    data,
                    tag,
                    "freeze",
                    kernels,
                    "h",
                    hm_param=param,
                    higher_target=target,
                    higher_scale=scale,
                    k_max=kmax,
                    init=init,
                    wait_a0=wait_a0,
                )
            )
        rows.append(
            row(
                data_name,
                data,
                "f2_root_rkl",
                "staged",
                TRIPLE,
                "h",
                hm_param="root_moment",
                higher_target="ratio_kl",
                higher_scale=1.0,
                k_max=3,
                init=init,
                wait_a0=wait_a0,
            )
        )
        for seed in (1, 2):
            rows.append(
                row(
                    data_name,
                    data,
                    f"d2_s{seed}",
                    "scratch",
                    TRIPLE,
                    "l1",
                    hm_param="root_moment",
                    higher_target="moment_kl",
                    higher_scale=1.0,
                    k_max=3,
                    seed=seed,
                )
            )
            rows.append(
                row(
                    data_name,
                    data,
                    f"g3_s{seed}",
                    "scratch",
                    TRIPLE,
                    "l1",
                    hm_param="root_moment",
                    higher_target="ratio_kl",
                    higher_scale=0.01,
                    k_max=3,
                    seed=seed,
                )
            )
            rows.append(
                row(
                    data_name,
                    data,
                    f"a0_s{seed}",
                    "a0",
                    ("poisson",),
                    "l1",
                    seed=seed,
                )
            )
    if len(rows) != 74:
        raise SystemExit(f"expected 74 runs, got {len(rows)}")
    return rows


def argv(job, out, epochs, init, freeze, higher_only, trunk_mult, save_h, save_l1):
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
        str(job["seed"]),
        "--lbd",
        "100",
        "--scale",
        "true",
        "--weight_steps",
        "100",
        "--val_every",
        "10",
        "--save_epochs",
        EPOCHS,
    ]
    if job["kind"] == "a0":
        cmd.extend(
            [
                "--moment_k",
                "1",
                "--moment_param",
                "direct",
                "--moment_loss",
                "moment",
                "--ratio",
                "false",
                "--save_best_l1",
                "true",
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
            "dyn",
            "--higher_target",
            job["higher_target"],
            "--higher_scale",
            str(job["higher_scale"]),
            "--k_max",
            str(job["k_max"]),
        ]
    )
    if save_l1:
        cmd.extend(["--save_best_l1", "true"])
    if save_h:
        cmd.extend(["--save_best_h", "true"])
    if higher_only:
        cmd.extend(["--higher_only", "true"])
    if freeze:
        cmd.extend(["--freeze_m1", "true"])
    if float(trunk_mult) != 1.0:
        cmd.extend(["--trunk_lr_mult", str(trunk_mult)])
    if init:
        cmd.extend(["--init_ckpt", str(init)])
    return cmd


def sample_run(job, out):
    out = Path(out)
    pairs = [("final", out / "latest.pt")]
    if job["select"] == "h":
        pairs.append(("best_h", out / "best_h.pt"))
    else:
        pairs.append(("best_l1", out / "best_l1.pt"))
    for name, ckpt in pairs:
        if not ckpt.is_file():
            raise SystemExit(f"missing {ckpt}")
        sample_one(ckpt, out / "sample" / name, job["data_name"], job["kernels"])


def ready(job, out):
    out = Path(out)
    if not (out / "finished.json").is_file():
        return False
    names = ["final", "best_h" if job["select"] == "h" else "best_l1"]
    for name in names:
        dest = out / "sample" / name
        if not (dest / "metrics.json").is_file():
            return False
        for kernel in job["kernels"]:
            if not (dest / f"samples_{kernel}.npy").is_file():
                return False
    return True


def run_job(job):
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    label = f"{job['data_name']}/{job['tag']}"
    try:
        if ready(job, out):
            print(f"skip {label}", flush=True)
            return label
        init = job.get("init") or ""
        if job.get("wait_a0"):
            wait_file(Path(init).parent / "finished.json", 4 * 3600)
        if init and not Path(init).is_file():
            raise SystemExit(f"missing A0 final {init}")
        if job["kind"] == "staged":
            warm = out / "warmup"
            if not (warm / "finished.json").is_file():
                call(argv(job, warm, 40, init, True, True, 1.0, True, False))
            if not (out / "finished.json").is_file():
                call(argv(job, out, 200, warm / "latest.pt", False, False, 0.1, True, False))
        elif not (out / "finished.json").is_file():
            if job["kind"] == "freeze":
                call(argv(job, out, 200, init, True, True, 1.0, True, False))
            elif job["kind"] in ("scratch", "a0"):
                call(argv(job, out, 200, "", False, False, 1.0, False, True))
            else:
                raise SystemExit(job["kind"])
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
    mine.sort(key=lambda job: (0 if job["tag"] == "a0_s0" else 2 if job.get("wait_a0") else 1, job["tag"]))
    print(f"task {args.task} runs {len(mine)} / {len(jobs)} workers {args.workers}", flush=True)
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    import concurrent.futures

    failed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
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
