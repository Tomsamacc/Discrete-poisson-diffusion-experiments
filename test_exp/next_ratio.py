"""PDF next-experiment matrix.

A learned mean          experiments/scale/{name}/best.pt
B normalized k=1        experiments/pmfratio/kmax1/{name}/best.pt
C balanced k=1          experiments/pmfratio/balanced_k1/{name}/best.pt
D raw-ratio k=1         experiments/pmfratio/raw_k1/{name}/best.pt
E balanced k=1,2        experiments/pmfratio/balanced_k2/{name}/best.pt
F balanced k=1,2,3      experiments/pmfratio/balanced_k3/{name}/best.pt
G oracle moments        current sample.generate path (poisson, nb, twopois)

A–F: Poisson kernel, quadratic T=100.
G: Poisson / NB / Two-Poisson, quadratic T=100.

Also: signed-tail e=log mhat-log m* (forward + on-policy) and schedule-aware
errors on the quadratic γ_t grid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from loss.loss import moments_from_ratio_pred  # noqa: E402
from loss.oracle import oracle_moments, signed_log_summary, support_logp  # noqa: E402
from sample import generate, make_gammas, predict_log_B, predict_x, reverse_step  # noqa: E402
from sample_tests import load_model  # noqa: E402
from test import plot_experiment  # noqa: E402
from train import load_counts, q_sample  # noqa: E402

RESULTS = HERE / "results" / "13_next"
NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
LEARNED = ("A_mean", "B_norm_k1", "C_bal_k1", "D_raw_k1", "E_bal_k2", "F_bal_k3")
BIN_LABELS = ("[0,0.1]", "[0.1,1]", "[1,10]", "[10,100]")
GAMMA_EDGES = (0.1, 1.0, 10.0, 100.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--n_sample", type=int, default=50000)
    p.add_argument("--n_mom", type=int, default=16384)
    p.add_argument("--n_onpolicy", type=int, default=1024)
    p.add_argument("--n_grid", type=int, default=256)
    p.add_argument("--sample_batch", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--stage",
        default="all",
        choices=("all", "sample", "oracle", "tails", "schedule"),
    )
    return p.parse_args()


def split_csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def ckpt_of(tag, name):
    if tag == "A_mean":
        return ROOT / "experiments" / "scale" / name / "best.pt"
    if tag == "B_norm_k1":
        return ROOT / "experiments" / "pmfratio" / "kmax1" / name / "best.pt"
    if tag == "C_bal_k1":
        return ROOT / "experiments" / "pmfratio" / "balanced_k1" / name / "best.pt"
    if tag == "D_raw_k1":
        return ROOT / "experiments" / "pmfratio" / "raw_k1" / name / "best.pt"
    if tag == "E_bal_k2":
        return ROOT / "experiments" / "pmfratio" / "balanced_k2" / name / "best.pt"
    if tag == "F_bal_k3":
        return ROOT / "experiments" / "pmfratio" / "balanced_k3" / name / "best.pt"
    raise ValueError(tag)


def dummy_args(device="cpu"):
    return SimpleNamespace(
        sample_batch=4096,
        normalize=None,
        lbd=100.0,
        t_eps=1e-4,
        snr_min=0.0,
        snr_max=100.0,
        sample_steps=100,
        ratio=False,
        k_max=3,
        z_rescale=True,
        device=str(device),
    )


def hats_of(model, z, gamma, args):
    if bool(getattr(args, "ratio", False)):
        pred = predict_log_B(model, z, gamma, args)
        return moments_from_ratio_pred(pred, z, gamma, args)
    return [predict_x(model, z, gamma, args)]


def write_table(path, rows, keys):
    path = Path(path)
    lines = [" ".join(f"{k:16}" for k in keys)]
    for r in rows:
        bits = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                bits.append(f"{v:16.4f}")
            else:
                bits.append(f"{str(v):16}")
        lines.append(" ".join(bits))
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text, end="", flush=True)


def gamma_bin(g):
    g = float(g)
    for i, e in enumerate(GAMMA_EDGES):
        if g < e:
            return i
    return len(GAMMA_EDGES) - 1


@torch.no_grad()
def forward_tails(model, args, x, device, xs, logp, n_mom, seed, tag, name):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    n = min(int(n_mom), int(x.shape[0]))
    idx = torch.randperm(x.shape[0], generator=g)[:n]
    xb = x[idx].to(device)
    t = torch.rand(n, device=device) * (1.0 - args.t_eps) + args.t_eps
    z = q_sample(xb, t, args.lbd)
    gamma = t * args.lbd
    hats = hats_of(model, z, gamma, args)
    stars = oracle_moments(z, gamma, xs, logp, k_max=len(hats))
    bins = np.array([gamma_bin(float(v)) for v in gamma.cpu()])
    rows = []
    for k, (h, s) in enumerate(zip(hats, stars), start=1):
        hh = h.reshape(-1).cpu().numpy()
        ss = s.reshape(-1).cpu().numpy()
        for b, lab in enumerate(BIN_LABELS):
            mask = bins == b
            if not mask.any():
                continue
            sm = signed_log_summary(hh[mask], ss[mask])
            rows.append({"dist": name, "tag": tag, "k": k, "bin": lab, "where": "forward", **sm})
    return rows


@torch.no_grad()
def onpolicy_tails(model, args, device, xs, logp, gammas, n, tag, name):
    hs = gammas[1:] - gammas[:-1]
    z = torch.zeros(int(n), 1, device=device)
    e1_all = []
    ratio_all = []
    rate_all = []
    bad = torch.zeros(int(n), dtype=torch.bool, device=device)
    for i in range(hs.numel()):
        g = float(gammas[i].item())
        h = float(hs[i].item())
        hats = hats_of(model, z, g, args)
        stars = oracle_moments(z, g, xs, logp, k_max=1)
        h1 = hats[0].reshape(-1).double().clamp_min(1e-12)
        s1 = stars[0].reshape(-1).double().clamp_min(1e-12)
        e1_all.append((h1.log() - s1.log()).cpu().numpy())
        ratio_all.append((h1 / s1).cpu().numpy())
        rate_all.append((h * h1).cpu().numpy())
        z = reverse_step(model, z, g, h, "poisson", args)
        z1 = z.reshape(-1)
        bad = bad | (~torch.isfinite(z1)) | (z1 > 1e6)
    e1_cat = np.concatenate(e1_all)
    sm = signed_log_summary(np.exp(e1_cat), np.ones_like(e1_cat))
    z_np = z.detach().float().cpu().numpy().reshape(-1)
    row = {
        "dist": name,
        "tag": tag,
        "k": 1,
        "bin": "onpolicy",
        "where": "rollout",
        **sm,
        "max_m1_over_mstar": float(np.max(np.concatenate(ratio_all))),
        "max_h_m1": float(np.max(np.concatenate(rate_all))),
        "runaway_frac": float(bad.float().mean().cpu()),
        "final_z_p99": float(np.quantile(z_np, 0.99)),
        "overflow_z": float((z_np > 1e6).mean()),
    }
    return [row]


@torch.no_grad()
def schedule_grid(model, args, x, device, xs, logp, gammas, n_grid, tag, name):
    rows = []
    n = min(int(n_grid), int(x.shape[0]))
    xb = x[:n].to(device)
    for g in gammas.tolist():
        g = float(g)
        z = torch.poisson((g * xb.clamp_min(0.0)))
        hats = hats_of(model, z, g, args)
        stars = oracle_moments(z, g, xs, logp, k_max=1)
        sm = signed_log_summary(hats[0].reshape(-1).cpu().numpy(), stars[0].reshape(-1).cpu().numpy())
        rows.append({"dist": name, "tag": tag, "gamma": g, **sm})
    return rows


def main():
    cli = parse_args()
    names = split_csv(cli.names)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    device = torch.device(
        cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu"
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    do_sample = cli.stage in ("all", "sample")
    do_oracle = cli.stage in ("all", "oracle")
    do_tails = cli.stage in ("all", "tails")
    do_sched = cli.stage in ("all", "schedule")

    loaded = {}
    for name in names:
        for tag in LEARNED:
            ckpt = ckpt_of(tag, name)
            if not ckpt.is_file():
                print(f"skip {tag} {name}: no {ckpt}", flush=True)
                continue
            print(f"load {tag} {name}", flush=True)
            model, args = load_model(ckpt, device)
            args.sample_batch = cli.sample_batch
            loaded[(name, tag)] = (model, args)

    metrics = []
    if do_sample:
        for name in names:
            for tag in LEARNED:
                pair = loaded.get((name, tag))
                if pair is None:
                    continue
                model, args = pair
                dest = RESULTS / "sample" / tag / name / "quad_T100"
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / "samples_poisson.npy"
                gammas = make_gammas("quad", 100, 0.0, float(args.lbd), device, power=2.0)
                if path.is_file():
                    print(f"    skip sample {tag} {name}", flush=True)
                else:
                    print(f"    sample Poisson {tag} {name} quad_T100", flush=True)
                    x = generate(model, cli.n_sample, "poisson", args, device, gammas=gammas)
                    np.save(path, x)
                table = plot_experiment(dest, name=name, kernels=("poisson",))
                for r in table:
                    metrics.append(
                        {
                            "dist": name,
                            "tag": tag,
                            "kernel": "poisson",
                            "wd": r["wd"],
                            "tv": r["tv"],
                            "overflow": r.get("overflow", 0.0),
                        }
                    )

    if do_oracle:
        for name in names:
            xs, logp = support_logp(name, device)
            args = dummy_args(device)
            args.sample_batch = cli.sample_batch
            gammas = make_gammas("quad", 100, 0.0, 100.0, device, power=2.0)
            for kernel in ("poisson", "nb", "twopois"):
                dest = RESULTS / "sample" / "G_oracle" / name / "quad_T100"
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / f"samples_{kernel}.npy"
                if path.is_file():
                    print(f"    skip oracle {name} {kernel}", flush=True)
                else:
                    print(f"    oracle {kernel} {name} quad_T100", flush=True)
                    x = generate(
                        None,
                        cli.n_sample,
                        kernel,
                        args,
                        device,
                        gammas=gammas,
                        oracle=(xs, logp),
                    )
                    np.save(path, x)
            table = plot_experiment(
                dest, name=name, kernels=("poisson", "nb", "twopois")
            )
            for r in table:
                metrics.append(
                    {
                        "dist": name,
                        "tag": "G_oracle",
                        "kernel": r["kernel"],
                        "wd": r["wd"],
                        "tv": r["tv"],
                        "overflow": r.get("overflow", 0.0),
                    }
                )

    if metrics:
        write_table(
            RESULTS / "metrics.txt",
            metrics,
            ("dist", "tag", "kernel", "wd", "tv", "overflow"),
        )
        (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    if do_tails:
        tail_rows = []
        gammas = make_gammas("quad", 100, 0.0, 100.0, device, power=2.0)
        for name in names:
            xs, logp = support_logp(name, device)
            xval = load_counts(ROOT / "data" / name / "val.npy")
            for tag in LEARNED:
                pair = loaded.get((name, tag))
                if pair is None:
                    continue
                model, args = pair
                print(f"  tails {tag} {name}", flush=True)
                tail_rows.extend(
                    forward_tails(
                        model, args, xval, device, xs, logp, cli.n_mom, cli.seed, tag, name
                    )
                )
                tail_rows.extend(
                    onpolicy_tails(model, args, device, xs, logp, gammas, cli.n_onpolicy, tag, name)
                )
        if tail_rows:
            write_table(
                RESULTS / "signed_tails.txt",
                tail_rows,
                (
                    "dist",
                    "tag",
                    "k",
                    "bin",
                    "where",
                    "n",
                    "median",
                    "p90",
                    "p99",
                    "max",
                    "pos_frac",
                    "pos_p90",
                ),
            )
            (RESULTS / "signed_tails.json").write_text(json.dumps(tail_rows, indent=2) + "\n")
            onp = [r for r in tail_rows if r.get("where") == "rollout"]
            if onp:
                write_table(
                    RESULTS / "onpolicy.txt",
                    onp,
                    (
                        "dist",
                        "tag",
                        "median",
                        "p90",
                        "p99",
                        "max",
                        "pos_frac",
                        "max_m1_over_mstar",
                        "max_h_m1",
                        "runaway_frac",
                        "final_z_p99",
                    ),
                )

    if do_sched:
        grid_rows = []
        gammas = make_gammas("quad", 100, 0.0, 100.0, device, power=2.0)
        for name in names:
            xs, logp = support_logp(name, device)
            xval = load_counts(ROOT / "data" / name / "val.npy")
            for tag in LEARNED:
                pair = loaded.get((name, tag))
                if pair is None:
                    continue
                model, args = pair
                print(f"  schedule {tag} {name}", flush=True)
                grid_rows.extend(
                    schedule_grid(model, args, xval, device, xs, logp, gammas, cli.n_grid, tag, name)
                )
        if grid_rows:
            write_table(
                RESULTS / "schedule_val.txt",
                grid_rows,
                ("dist", "tag", "gamma", "n", "median", "p90", "p99", "max"),
            )
            (RESULTS / "schedule_val.json").write_text(json.dumps(grid_rows, indent=2) + "\n")

    print("done ->", RESULTS, flush=True)


if __name__ == "__main__":
    main()
