"""Shared-draw moment errors for hybrid models and the direct-mean baseline.

Reports, by gamma bin, log(m_hat_k / m_k*), |m_hat_1 - m_1*|, and
|eta_hat_k - eta_k*| with eta_2 = h^2 m_2 / 2, eta_3 = h^3 m_3 / 6.
h is the quadratic T=100 step at that training gamma.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from loss.loss import quad_reverse_h  # noqa: E402
from loss.oracle import BIN_LABELS, gamma_bin_index, oracle_moments, support_logp  # noqa: E402
from sample import posterior_moments  # noqa: E402
from sample_tests import load_model  # noqa: E402
from train import load_counts, q_sample  # noqa: E402

NAMES = ("gamma_ltj", "nb", "poismix_mod", "poissmix3")
TAGS = ("hybrid_eq", "hybrid_dyn", "mean", "dyn_norm")
RESULTS = HERE / "results" / "21_hybrid"


def qstats(arr):
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "median": None, "p90": None, "p99": None, "max": None}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


def ckpt_of(tag, name):
    if tag == "mean":
        return ROOT / "experiments" / "scale" / name / "best.pt"
    if tag == "dyn_norm":
        return ROOT / "experiments" / "raw_multi" / "dyn_norm" / name / "best.pt"
    return ROOT / "experiments" / "hybrid" / tag / name / "best.pt"


def prepare(args, name):
    if str(getattr(args, "moment_loss", "")) == "hybrid" or str(getattr(args, "moment_param", "")) == "raw":
        args.ratio = True
        args.moment_param = "raw"
        args.ratio_loss = "raw"
        args.k_max = 3
    prior = getattr(args, "prior_moments", None)
    if args.ratio and (prior is None or len(list(prior)) < 3):
        x = load_counts(ROOT / "data" / name / "train.npy").reshape(-1).double()
        args.prior_moments = [float(x.pow(k).mean()) for k in range(1, 4)]
    return args


def draw_states(name, device, seed):
    x = load_counts(ROOT / "data" / name / "val.npy")
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    torch.manual_seed(seed)
    t_eps = 1e-4
    lbd = 100.0
    t = torch.rand(x.shape[0]) * (1.0 - t_eps) + t_eps
    z = q_sample(x, t, lbd)
    return {
        "z": z.cpu(),
        "t": t.cpu(),
        "gamma": (t * lbd).cpu(),
    }


def bin_rows(idx, packs):
    rows = []
    masks = [("all", np.ones(idx.shape[0], dtype=bool))]
    for b, lab in enumerate(BIN_LABELS):
        masks.append((lab, idx == b))
    for lab, m in masks:
        row = {"bin": lab}
        for key, arr in packs.items():
            row[key] = qstats(arr[m])
        rows.append(row)
    return rows


@torch.no_grad()
def eval_one(model, args, states, xs, logp, device):
    z = states["z"].to(device)
    t = states["t"].to(device)
    gamma = states["gamma"].to(device)
    moms = posterior_moments(model, z, gamma.reshape(-1, 1), args, order=3)
    stars = oracle_moments(z, gamma, xs, logp, k_max=3)
    h = quad_reverse_h(t, float(args.lbd), 100)
    idx = gamma_bin_index(gamma).cpu().numpy()
    packs = {}
    for k in range(3):
        mh = moms[k].reshape(-1).double()
        ms = stars[k].reshape(-1).double()
        packs[f"log{k + 1}"] = (mh.clamp_min(1e-12).log() - ms.clamp_min(1e-12).log()).cpu().numpy()
        packs[f"abs{k + 1}"] = (mh - ms).abs().cpu().numpy()
    mh1 = moms[0].reshape(-1).double()
    ms1 = stars[0].reshape(-1).double()
    packs["habs1"] = (h * (mh1 - ms1).abs()).cpu().numpy()
    for k, fact in ((1, 2.0), (2, 6.0)):
        hk = h.pow(k + 1)
        mh = moms[k].reshape(-1).double()
        ms = stars[k].reshape(-1).double()
        packs[f"eta{k + 1}"] = (hk * (mh - ms).abs() / fact).cpu().numpy()
    return bin_rows(idx, packs)


def grad_rows(names):
    rows = []
    for tag in ("hybrid_eq", "hybrid_dyn"):
        for name in names:
            d = ROOT / "experiments" / "hybrid" / tag / name
            heads = d / "grad_heads.json"
            cos = d / "grad_cosine.json"
            if not heads.is_file() or not cos.is_file():
                continue
            rows.append(
                {
                    "tag": tag,
                    "dist": name,
                    "heads": json.loads(heads.read_text()),
                    "cosine": json.loads(cos.read_text()),
                }
            )
    return rows


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    cli = p.parse_args()
    names = [x.strip() for x in cli.names.split(",") if x.strip()]
    device = torch.device(cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu")
    out = Path(cli.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    missing = [str(ckpt_of(tag, name)) for tag in ("hybrid_eq", "hybrid_dyn") for name in names if not ckpt_of(tag, name).is_file()]
    if missing:
        for path in missing:
            print(f"missing {path}", flush=True)
        raise SystemExit("train hybrid runs first: ./script/train_hybrid.sh")

    rows = []
    for name in names:
        state_path = out / "states" / f"{name}.pt"
        if state_path.is_file() and not cli.force:
            states = torch.load(state_path, map_location="cpu", weights_only=False)
        else:
            states = draw_states(name, device, cli.seed)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(states, state_path)
        xs, logp = support_logp(name, device)
        for tag in TAGS:
            ckpt = ckpt_of(tag, name)
            print(f"eval {tag} {name}", flush=True)
            model, args = load_model(ckpt, device)
            prepare(args, name)
            bins = eval_one(model, args, states, xs, logp, device)
            rows.append({"tag": tag, "dist": name, "ckpt": str(ckpt), "bins": bins})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    (out / "moment_errors.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "grad_summary.json").write_text(json.dumps(grad_rows(names), indent=2) + "\n")
    print(f"wrote {out / 'moment_errors.json'}", flush=True)


if __name__ == "__main__":
    main()
