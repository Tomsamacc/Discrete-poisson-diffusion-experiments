"""Frozen-ckpt sampling ablations. No retrain.

1. sweep: T=100,200,500,1000 (h=1, 0.5, 0.2, 0.1), uniform γ
2. quad:  T=100, γ_t = λ (t/T)^2  so first step 0→0.01
3. oracle first step only, rest 99 steps unchanged
     A = current sampler (same as sweep T=100)
     B = first step m=E[X], K~Pois(h m)
     C = first step X~p_X, K~Pois(h X)
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

from sample import build_model, reverse_step  # noqa: E402

SWEEP = (
    (100, "T100_h1"),
    (200, "T200_h05"),
    (500, "T500_h02"),
    (1000, "T1000_h01"),
)
DEFAULT_NAMES = "gamma_ltj,nb,poismix_mod,pois20,poissmix,poissmix3,zip,yule_simon"
DEFAULT_KERNELS = "poisson,nb,twopois"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_root", default=str(ROOT / "experiments" / "scale"))
    p.add_argument("--out_root", default=str(HERE))
    p.add_argument("--names", default=DEFAULT_NAMES)
    p.add_argument("--kernels", default=DEFAULT_KERNELS)
    p.add_argument(
        "--stage",
        default="all",
        choices=("all", "sweep", "quad", "quad01", "unif01", "oracle"),
    )
    p.add_argument("--n_sample", type=int, default=50000)
    p.add_argument("--sample_batch", type=int, default=4096)
    p.add_argument("--snr_min", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def split_csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def ckpt_of(ckpt_root, name):
    return Path(ckpt_root) / name / "best.pt"


def load_model(ckpt, device):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = blob.get("args") or {}
    args = SimpleNamespace(
        ckpt=str(ckpt),
        hidden=int(a.get("hidden", 128)),
        layers=int(a.get("layers", 3)),
        continuous_t=bool(a.get("continuous_t", True)),
        lbd=float(a.get("lbd", 100.0)),
        z_rescale=bool(a.get("z_rescale", a.get("scale", False))),
        clip_z=bool(a.get("clip_z", True)),
        clip_range=a.get("clip_range"),
        normalize=a.get("normalize"),
        t_eps=float(a.get("t_eps", 1e-4)),
        ratio=bool(a.get("ratio", False)),
        k_max=int(a.get("k_max", 3)),
        ratio_loss=str(a.get("ratio_loss", "normalized")),
        moment_param=str(a.get("moment_param", "direct")),
        moment_loss=str(a.get("moment_loss", "moment")),
        hm_param=str(a.get("hm_param", "") or ""),
        hybrid_weight=str(a.get("hybrid_weight", "equal") or "equal"),
        prior_moments=a.get("prior_moments"),
        consistency=str(a.get("consistency", "none") or "none"),
        lambda_cons=float(a.get("lambda_cons", 1.0) or 1.0),
    )
    if args.ratio_loss in ("balanced", "raw"):
        args.ratio = True
    model = build_model(args, device)
    return model, args


def make_gammas(schedule, steps, g0, g1, device, power=None):
    """γ_i = g0 + (g1-g0) (i/T)^p. uniform p=1, quad p=2."""
    steps = int(steps)
    if power is None:
        power = 1.0 if schedule == "uniform" else 2.0
    i = torch.arange(steps + 1, device=device, dtype=torch.float32)
    g = float(g0) + (float(g1) - float(g0)) * (i / float(steps)).pow(float(power))
    g[0] = float(g0)
    g[-1] = float(g1)
    return g


def first_step(z, h, oracle, x_pool, x_mean):
    if oracle == "B":
        m = torch.full_like(z, float(x_mean))
        return z + torch.poisson((h * m).clamp_min(0.0))
    if oracle == "C":
        n = z.shape[0]
        idx = torch.randint(0, x_pool.shape[0], (n,), device=z.device)
        x = x_pool[idx].view_as(z)
        return z + torch.poisson((h * x.clamp_min(0.0)))
    raise ValueError(oracle)


@torch.no_grad()
def generate(
    model, n, kernel, args, device, gammas, oracle=None, x_pool=None, x_mean=None, z_from_x=False
):
    out = []
    left = int(n)
    hs = gammas[1:] - gammas[:-1]
    g0 = float(gammas[0].item())
    while left > 0:
        b = min(int(args.sample_batch), left)
        if z_from_x:
            idx = torch.randint(0, x_pool.shape[0], (b,), device=device)
            x0 = x_pool[idx].view(b, 1)
            z = torch.poisson((g0 * x0.clamp_min(0.0)))
        else:
            z = torch.zeros(b, 1, device=device)
        for i in range(hs.numel()):
            g = float(gammas[i].item())
            h = float(hs[i].item())
            if i == 0 and oracle in ("B", "C"):
                z = first_step(z, h, oracle, x_pool, x_mean)
            else:
                z = reverse_step(model, z, g, h, kernel, args)
        g_last = float(gammas[-1].item())
        x = (z / max(g_last, 1e-12)).clamp_min(0.0)
        if args.normalize is not None:
            x = model.decode_x(x)
        out.append(x.cpu())
        left -= b
    return torch.cat(out, 0).numpy().reshape(-1)


def has_all(out_dir, kernels):
    return all((out_dir / f"samples_{k}.npy").is_file() for k in kernels)


def run_one(
    model, args, device, out_dir, kernels, n_sample, gammas, meta,
    oracle=None, x_pool=None, x_mean=None, z_from_x=False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h0 = float((gammas[1] - gammas[0]).item())
    hT = float((gammas[-1] - gammas[-2]).item())
    g0 = float(gammas[0].item())
    z0 = f"Pois({g0:g} X)" if z_from_x else "0"
    print(
        f"  {out_dir} n={n_sample} steps={gammas.numel()-1} "
        f"γ0={g0:g} z0={z0} h0={h0:g} hT={hT:g} oracle={oracle or 'A'} "
        f"z_rescale={args.z_rescale} ckpt={args.ckpt}",
        flush=True,
    )
    for k in kernels:
        path = out_dir / f"samples_{k}.npy"
        if path.is_file():
            print(f"    skip {k}", flush=True)
            continue
        x = generate(
            model, n_sample, k, args, device, gammas,
            oracle=oracle, x_pool=x_pool, x_mean=x_mean, z_from_x=z_from_x,
        )
        np.save(path, x)
        print(f"    saved {path.name} E[z/γ]={x.mean():.4g}", flush=True)
    meta = dict(meta)
    meta.update(
        {
            "ckpt": args.ckpt,
            "z_rescale": args.z_rescale,
            "lbd": args.lbd,
            "n_sample": n_sample,
            "steps": int(gammas.numel() - 1),
            "h0": h0,
            "hT": hT,
            "gamma0": g0,
            "gammaT": float(gammas[-1].item()),
            "z0": z0,
            "oracle": oracle or "A",
            "kernels": list(kernels),
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_x_pool(name, device, split="train"):
    path = ROOT / "data" / name / f"{split}.npy"
    x = np.load(path).astype(np.float32).reshape(-1)
    return torch.from_numpy(x).to(device), float(x.mean())


def main():
    cli = parse_args()
    names = split_csv(cli.names)
    kernels = split_csv(cli.kernels)
    device = torch.device(
        cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu"
    )
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    ckpt_root = Path(cli.ckpt_root)
    if not ckpt_root.is_absolute():
        ckpt_root = ROOT / ckpt_root
    out_root = Path(cli.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root = out_root.resolve()
    stages = ("sweep", "quad", "oracle") if cli.stage == "all" else (cli.stage,)

    for name in names:
        ckpt = ckpt_of(ckpt_root, name)
        if not ckpt.is_file():
            print(f"skip {name}: no {ckpt}", flush=True)
            continue
        print(f"======== {name} ckpt={ckpt} ========", flush=True)
        model, args = load_model(ckpt, device)
        args.sample_batch = cli.sample_batch
        g0, g1 = float(cli.snr_min), float(args.lbd)
        x_pool, x_mean = load_x_pool(name, device, split="train")
        x_val, _ = load_x_pool(name, device, split="val")

        if "sweep" in stages:
            for steps, tag in SWEEP:
                dest = out_root / "sweep" / tag / name
                if has_all(dest, kernels):
                    print(f"  skip sweep {tag}", flush=True)
                    continue
                gammas = make_gammas("uniform", steps, g0, g1, device)
                run_one(
                    model, args, device, dest, kernels, cli.n_sample, gammas,
                    {"test": "sweep", "tag": tag, "name": name, "schedule": "uniform"},
                )

        if "quad" in stages:
            dest = out_root / "quad" / "T100_quad" / name
            if has_all(dest, kernels):
                print("  skip quad T100", flush=True)
            else:
                gammas = make_gammas("quad", 100, 0.0, g1, device)
                run_one(
                    model, args, device, dest, kernels, cli.n_sample, gammas,
                    {"test": "quad", "tag": "T100_quad", "name": name, "schedule": "quad"},
                )

        if "quad01" in stages:
            dest = out_root / "quad" / "T100_quad_g01_zx" / name
            if has_all(dest, kernels):
                print("  skip quad T100 γ0=0.1 z~Pois(0.1 X)", flush=True)
            else:
                gammas = make_gammas("quad", 100, 0.1, g1, device)
                run_one(
                    model, args, device, dest, kernels, cli.n_sample, gammas,
                    {
                        "test": "quad01",
                        "tag": "T100_quad_g01_zx",
                        "name": name,
                        "schedule": "quad",
                        "note": "γ: 0.1→λ, z0 ~ Pois(0.1 X), X from val",
                    },
                    x_pool=x_val,
                    z_from_x=True,
                )

        if "unif01" in stages:
            dest = out_root / "unif" / "T100_g01_zx" / name
            if has_all(dest, kernels):
                print("  skip uniform T100 γ0=0.1 z~Pois(0.1 X)", flush=True)
            else:
                gammas = make_gammas("uniform", 100, 0.1, g1, device)
                run_one(
                    model, args, device, dest, kernels, cli.n_sample, gammas,
                    {
                        "test": "unif01",
                        "tag": "T100_g01_zx",
                        "name": name,
                        "schedule": "uniform",
                        "note": "original T=100 uniform, γ: 0.1→λ, z0 ~ Pois(0.1 X)",
                    },
                    x_pool=x_val,
                    z_from_x=True,
                )

        if "oracle" in stages:
            gammas = make_gammas("uniform", 100, g0, g1, device)
            base = out_root / "sweep" / "T100_h1" / name
            dest_a = out_root / "oracle" / "A_baseline" / name
            dest_a.mkdir(parents=True, exist_ok=True)
            if not has_all(base, kernels):
                print("  oracle A needs sweep T100 first (run --stage sweep)", flush=True)
            elif not has_all(dest_a, kernels):
                for k in kernels:
                    src = base / f"samples_{k}.npy"
                    dst = dest_a / f"samples_{k}.npy"
                    if src.is_file() and not dst.is_file():
                        dst.write_bytes(src.read_bytes())
                (dest_a / "meta.json").write_text(
                    json.dumps(
                        {
                            "test": "oracle",
                            "tag": "A_baseline",
                            "name": name,
                            "note": "alias of sweep/T100_h1",
                            "oracle": "A",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            for tag, oracle in (("B_true_mean", "B"), ("C_exact_mix", "C")):
                dest = out_root / "oracle" / tag / name
                if has_all(dest, kernels):
                    print(f"  skip oracle {tag}", flush=True)
                    continue
                run_one(
                    model, args, device, dest, kernels, cli.n_sample, gammas,
                    {"test": "oracle", "tag": tag, "name": name, "schedule": "uniform"},
                    oracle=oracle, x_pool=x_pool, x_mean=x_mean,
                )
            print(f"  oracle B uses E[X]={x_mean:.4g}", flush=True)

    print("done sampling", flush=True)


if __name__ == "__main__":
    main()
