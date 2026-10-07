import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sweep1d_run import call, sample_one

OUT = ROOT / "experiments" / "prescreen1d"
A0 = ROOT / "experiments" / "sweep1d"
DATASETS = (
    ("gamma_ltj", "data/gamma_ltj"),
    ("nb", "data/nb"),
    ("poismix_mod", "data/poismix_mod"),
    ("poissmix3", "data/poissmix3"),
)
PAIR = ("poisson", "nb")
TRIPLE = ("poisson", "nb", "twopois")


def row(data_name, data, tag, kind, kernels, **extra):
    job = {
        "data_name": data_name,
        "data": data,
        "tag": tag,
        "kind": kind,
        "kernels": kernels,
        "k_max": extra.pop("k_max", 3),
        "hm_param": extra.pop("hm_param", "root_moment"),
        "higher_target": extra.pop("higher_target", "moment_kl"),
        "kernel_nll": extra.pop("kernel_nll", "none"),
        "kernel_lam": extra.pop("kernel_lam", 0.0),
        "moment_beta": extra.pop("moment_beta", 1.0),
        "l2_scale": extra.pop("l2_scale", 1.0),
        "l3_scale": extra.pop("l3_scale", 1.0),
        "freeze_heads": extra.pop("freeze_heads", ""),
        "init": extra.pop("init", ""),
        "out": str(OUT / data_name / tag),
    }
    if extra:
        raise ValueError(extra)
    return job


def build_jobs():
    rows = []
    for data_name, data in DATASETS:
        init = str(A0 / data_name / "a0_kl" / "latest.pt")
        rows.append(row(data_name, data, "p0_beta05_m23", "freeze", TRIPLE, higher_target="moment_beta", moment_beta=0.5, init=init))
        rows.append(row(data_name, data, "p0_beta15_m23", "freeze", TRIPLE, higher_target="moment_beta", moment_beta=1.5, init=init))
        rows.append(row(data_name, data, "p0_nb_nll", "freeze", PAIR, k_max=2, kernel_nll="nb", init=init))
        rows.append(row(data_name, data, "p0_twopois_nll", "freeze", TRIPLE, kernel_nll="twopois", init=init))
        rows.append(row(data_name, data, "p0_mixed_m23", "freeze", TRIPLE, higher_target="mixed", init=init))
        rows.append(row(data_name, data, "p0_seq_m23", "seq", TRIPLE, init=init))
        rows.append(row(data_name, data, "p0_vargap_mkl", "freeze", TRIPLE, hm_param="var_gap", higher_target="moment_kl", init=init))
        rows.append(row(data_name, data, "p0_nb_nll_m01", "freeze", PAIR, k_max=2, kernel_nll="nb", kernel_lam=0.1, init=init))
        rows.append(row(data_name, data, "p0_nb_nll_m1", "freeze", PAIR, k_max=2, kernel_nll="nb", kernel_lam=1.0, init=init))
        rows.append(row(data_name, data, "p0_twopois_nll_m01", "freeze", TRIPLE, kernel_nll="twopois", kernel_lam=0.1, init=init))
        rows.append(row(data_name, data, "p0_twopois_nll_m1", "freeze", TRIPLE, kernel_nll="twopois", kernel_lam=1.0, init=init))
    return rows


def argv(job, out, init, k_max, freeze_heads, l2_scale):
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
        "200",
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
        "--weight_steps",
        "100",
        "--val_every",
        "10",
        "--moment_loss",
        "hm",
        "--hm_param",
        job["hm_param"],
        "--hybrid_weight",
        "dyn",
        "--higher_target",
        job["higher_target"],
        "--higher_scale",
        "1",
        "--higher_only",
        "true",
        "--k_max",
        str(k_max),
        "--save_best_h",
        "true",
        "--freeze_m1",
        "true",
        "--moment_beta",
        str(job["moment_beta"]),
        "--kernel_nll",
        job["kernel_nll"],
        "--kernel_lam",
        str(job["kernel_lam"]),
        "--l2_scale",
        str(l2_scale),
        "--l3_scale",
        str(job["l3_scale"]),
        "--init_ckpt",
        str(init),
    ]
    if freeze_heads:
        cmd.extend(["--freeze_heads", freeze_heads])
    return cmd


def ready(job, out):
    out = Path(out)
    if not (out / "finished.json").is_file():
        return False
    for name in ("final", "best_h"):
        dest = out / "sample" / name
        if not (dest / "metrics.json").is_file():
            return False
        for kernel in job["kernels"]:
            if not (dest / f"samples_{kernel}.npy").is_file():
                return False
    return True


def sample_run(job, out):
    out = Path(out)
    for name, ckpt in (("final", out / "latest.pt"), ("best_h", out / "best_h.pt")):
        if not ckpt.is_file():
            raise SystemExit(f"missing {ckpt}")
        sample_one(ckpt, out / "sample" / name, job["data_name"], job["kernels"])


def run_job(job):
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    label = f"{job['data_name']}/{job['tag']}"
    try:
        if ready(job, out):
            print(f"skip {label}", flush=True)
            return label
        init = job["init"]
        if not Path(init).is_file():
            raise SystemExit(f"missing A0 {init}")
        if job["kind"] == "seq":
            warm = out / "m2"
            if not (warm / "finished.json").is_file():
                call(argv(job, warm, init, 2, "", 1.0))
            if not (out / "finished.json").is_file():
                call(argv(job, out, warm / "latest.pt", 3, "0,1", 0.0))
        elif not (out / "finished.json").is_file():
            call(argv(job, out, init, job["k_max"], job["freeze_heads"], job["l2_scale"]))
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
    print(f"task {args.task} runs {len(mine)} / {len(jobs)} workers {args.workers}", flush=True)
    import concurrent.futures
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
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
