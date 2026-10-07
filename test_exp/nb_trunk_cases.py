import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from loss.loss import full_bregman, higher_moment_maps, raw_ratio_targets
from loss.oracle import BIN_LABELS, gamma_bin_index
from model.ratio_mlp import RatioMLP
from model.small_diffusion import PoissonDiscreteDiffusionModel
from sample import generate, make_gammas
from sample_tests import load_model
from test import plot_experiment
from train import (
    _batch_loss,
    _qstats,
    _record_trunk,
    head_cos_rows,
    head_norm_rows,
    load_counts,
    make_loader,
    q_sample,
    run_epoch,
    set_seed,
)

PARAMS = ("direct_ratio", "root_moment")
FREEZE_EPOCHS = 40
TRACE_EPOCHS = 10
PROBE_BATCHES = 200
UNFREEZE_STEPS = (100, 500, 1000)
N_SAMPLE = 50000
LBD = 100.0
T_EPS = 1e-4
LR = 1e-3
REF = {
    "direct_poisson": {"wd": 0.59, "tv": 0.043},
    "direct_ratio_equal": {"wd": 1.50, "tv": 0.057},
    "root_moment_equal": {"wd": 7.42, "tv": 0.874},
}

RESULTS = HERE / "results" / "nb_trunk"
DIRECT_DIR = ROOT / "experiments" / "direct_prl" / "direct" / "k1" / "nb"
JOINT_ROOT = ROOT / "experiments" / "nb_trunk" / "joint"
FROZEN_ROOT = ROOT / "experiments" / "nb_trunk" / "frozen"
UNFREEZE_ROOT = ROOT / "experiments" / "nb_trunk" / "unfreeze"


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def pick_device(name):
    if name == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def need_cuda(device, stage):
    if device.type != "cuda":
        raise SystemExit(f"{stage} needs a GPU. cuda is not available on this node.")


def nb_support():
    hi = int(stats.nbinom.ppf(1.0 - 1e-12, n=5, p=0.2))
    hi = max(hi, 1)
    ks = np.arange(0, hi + 1)
    p = stats.nbinom.pmf(ks, n=5, p=0.2).astype(np.float64)
    missing = float(max(0.0, 1.0 - p.sum()))
    p = p / p.sum()
    logp = np.log(np.clip(p, 1e-300, None))
    return ks, logp, missing


def posterior_mean(z, gamma, ks, logp):
    z = z.reshape(-1).double().round()
    g = gamma.reshape(-1).double()
    x = torch.as_tensor(ks, device=z.device, dtype=torch.float64)
    lp = torch.as_tensor(logp, device=z.device, dtype=torch.float64)
    rate = g[:, None] * x[None, :]
    loglik = torch.empty(z.shape[0], x.shape[0], dtype=torch.float64, device=z.device)
    pos = x > 0
    zp = z[:, None]
    rp = rate[:, pos]
    loglik[:, pos] = zp * rp.clamp_min(1e-30).log() - rp - torch.lgamma(zp + 1.0)
    if bool((~pos).any()):
        zero_ll = torch.where(z == 0, torch.zeros_like(z), torch.full_like(z, -1e300))
        loglik[:, ~pos] = zero_ll[:, None]
    w = torch.softmax(loglik + lp.view(1, -1), dim=-1)
    return (w * x.view(1, -1)).sum(-1)


def hm_namespace(param, device):
    return SimpleNamespace(
        hidden=128,
        layers=3,
        continuous_t=True,
        lbd=LBD,
        z_rescale=True,
        scale=True,
        clip_z=True,
        clip_range=None,
        normalize=None,
        t_eps=T_EPS,
        ratio=True,
        k_max=3,
        ratio_loss="normalized",
        moment_k=0,
        moment_param="direct",
        moment_loss="hm",
        hm_param=param,
        hybrid_weight="equal",
        weight_steps=100,
        consistency="none",
        lambda_cons=1.0,
        prior_moments=None,
        sample_batch=4096,
        snr_min=0.0,
        snr_max=LBD,
        sample_steps=100,
        seed=0,
        device=str(device),
        lr=LR,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        batch_size=256,
        epochs=200,
        val_every=10,
    )


def build_hm(device):
    net = RatioMLP(in_dim=1, hidden=128, out_dim=3, layers=3, continuous_t=True)
    model = PoissonDiscreteDiffusionModel(
        net, lbd=LBD, z_rescale=True, clip_z=True, clip_range=None, normalize=None, ratio=True
    )
    return model.to(device)


def copy_m1_from_direct(direct, hm):
    src = direct.state_dict()
    dst = hm.state_dict()
    with torch.no_grad():
        for key, value in src.items():
            if key not in dst:
                raise KeyError(key)
            if key.endswith("out_fc.1.weight") or key.endswith("out_fc.1.bias"):
                dst[key][0].copy_(value[0])
            else:
                if tuple(dst[key].shape) != tuple(value.shape):
                    raise RuntimeError(f"shape mismatch {key}")
                dst[key].copy_(value)
        hm.load_state_dict(dst)


def m1_probe(direct, hm, device, n=64):
    direct.eval()
    hm.eval()
    z = torch.rand(n, 1, device=device) * 50
    t = torch.rand(n, device=device) * (1.0 - T_EPS) + T_EPS
    with torch.no_grad():
        left = direct(z, t, alpha=t).reshape(-1)
        right = torch.nn.functional.softplus(hm(z, t, alpha=t)[:, 0])
    return float((left - right).abs().max())


def freeze_trunk_and_m1(model):
    last = model.net.out_fc[-1]
    trunk = [p for p in model.parameters() if p is not last.weight and p is not last.bias]
    for p in trunk:
        p.requires_grad_(False)

    def zero_m1():
        if last.weight.grad is not None:
            last.weight.grad[0].zero_()
        if last.bias.grad is not None:
            last.bias.grad[0].zero_()

    def unfreeze():
        for p in trunk:
            p.requires_grad_(True)

    return trunk, last, zero_m1, unfreeze


def save_blob(path, model, args, epoch, opt=None, extra=None):
    blob = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "opt": None if opt is None else opt.state_dict(),
        "args": vars(args),
    }
    if extra:
        blob.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)


def data_loaders(device, batch_size=256):
    xtr = load_counts(ROOT / "data" / "nb" / "train.npy")
    xva = load_counts(ROOT / "data" / "nb" / "val.npy")
    train_loader = make_loader(xtr, batch_size, shuffle=True)
    val_loader = make_loader(xva, batch_size, shuffle=False)
    return xtr, xva, train_loader, val_loader


def train_subprocess(out_dir, extra):
    out_dir = Path(out_dir)
    if (out_dir / "finished.json").is_file():
        print(f"skip train {out_dir} (finished)", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "train.py"),
        "-c",
        str(ROOT / "config_scale.yml"),
        "--data",
        "data/nb",
        "--out_dir",
        str(out_dir),
        "--lbd",
        "100",
        "--scale",
        "true",
        "--device",
        "cuda",
    ]
    cmd.extend(extra)
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def poisson_row(ckpt, dest, device):
    dest = Path(dest)
    metrics_path = dest / "poisson.json"
    if metrics_path.is_file() and (dest / "samples_poisson.npy").is_file():
        return load_json(metrics_path)
    dest.mkdir(parents=True, exist_ok=True)
    model, args = load_model(ckpt, device)
    args.sample_batch = 4096
    args.normalize = None
    gammas = make_gammas("quad", 100, 0.0, float(args.lbd), device, power=2.0)
    torch.manual_seed(0)
    np.random.seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    print(f"sample poisson {ckpt}", flush=True)
    x = generate(model, N_SAMPLE, "poisson", args, device, gammas=gammas)
    np.save(dest / "samples_poisson.npy", x)
    table = plot_experiment(dest, name="nb", kernels=("poisson",))
    row = table[0]
    row["mean"] = float(np.mean(x))
    dump(metrics_path, row)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def trace_summary(trace):
    norms = np.asarray(trace["norms"], dtype=np.float64)
    cos = np.asarray(trace["cos"], dtype=np.float64)
    out = {"n": int(norms.shape[0])}
    if norms.size == 0:
        return out
    for k, row in enumerate(head_norm_rows(norms)):
        out[f"g{k + 1}"] = {key: row[key] for key in ("n", "median", "p90", "p99", "max")}
    for row in head_cos_rows(cos):
        out["cos" + row["pair"]] = {
            key: row[key] for key in ("n", "mean", "median", "p10", "p90")
        }
    g1 = np.clip(norms[:, 0], 1e-12, None)
    out["g2_over_g1_median"] = float(np.median(norms[:, 1] / g1))
    out["g3_over_g1_median"] = float(np.median(norms[:, 2] / g1))
    return out


def slice_trace(trace, n):
    return {"norms": trace["norms"][:n], "cos": trace["cos"][:n]}


def eval_scores(model, loader, args, device, ks, logp, kind, seed=12345):
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    was = model.training
    model.eval()
    per_all = []
    err_all = []
    bins_l1 = {i: [] for i in range(4)}
    bins_err = {i: [] for i in range(4)}
    gmin = float(args.t_eps) * float(args.lbd)
    for (xb,) in loader:
        xb = xb.to(device)
        t = torch.rand(xb.shape[0], device=device) * (1.0 - args.t_eps) + args.t_eps
        z = q_sample(xb, t, args.lbd)
        gamma = t * args.lbd
        alpha = t if args.z_rescale else None
        pred = model(z, t, alpha=alpha)
        if kind == "hm":
            m, _s = higher_moment_maps(pred, z, gamma, args.hm_param, gmin=gmin)
            m1 = m[:, 0]
        else:
            m1 = pred.reshape(-1).double()
        x = xb.reshape(-1).double().clamp_min(0)
        per = full_bregman(x, m1)
        star = posterior_mean(z.reshape(-1), gamma.reshape(-1), ks, logp)
        err = (m1.double().clamp_min(1e-12).log() - star.clamp_min(1e-12).log()).abs()
        per_all.append(per.detach().cpu())
        err_all.append(err.detach().cpu())
        idx = gamma_bin_index(gamma).detach().cpu().numpy()
        per_np = per.detach().cpu().numpy()
        err_np = err.detach().cpu().numpy()
        for b in range(4):
            mask = idx == b
            if mask.any():
                bins_l1[b].append(per_np[mask])
                bins_err[b].append(err_np[mask])
    model.train(was)
    per = torch.cat(per_all).numpy()
    err = torch.cat(err_all).numpy()
    bins = {}
    for b, lab in enumerate(BIN_LABELS):
        lb = np.concatenate(bins_l1[b]) if bins_l1[b] else np.array([])
        eb = np.concatenate(bins_err[b]) if bins_err[b] else np.array([])
        bins[lab] = {"l1": _qstats(lb), "abs_log_m1": _qstats(eb)}
    return {
        "l1": float(per.mean()),
        "abs_log_m1": float(err.mean()),
        "bins": bins,
    }


def case1(device):
    out = RESULTS / "case1_oracle.json"
    if out.is_file():
        print(f"skip case1 ({out})", flush=True)
        return load_json(out)
    ks, logp, missing = nb_support()
    _xtr, xva, _train_loader, val_loader = data_loaders(device, batch_size=8192)
    x_np = xva.reshape(-1).double()
    prior = full_bregman(x_np, torch.full_like(x_np, 20.0))
    emp = float(x_np.mean())
    prior_emp = full_bregman(x_np, torch.full_like(x_np, emp))
    passes = []
    pooled_l = []
    pooled_g = []
    for seed in range(4):
        set_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        losses = []
        gammas = []
        for (xb,) in val_loader:
            xb = xb.to(device)
            t = torch.rand(xb.shape[0], device=device) * (1.0 - T_EPS) + T_EPS
            z = q_sample(xb, t, LBD)
            gamma = (t * LBD).reshape(-1)
            star = posterior_mean(z.reshape(-1), gamma, ks, logp)
            per = full_bregman(xb.reshape(-1).double().clamp_min(0), star)
            losses.append(per.detach().cpu())
            gammas.append(gamma.detach().cpu())
        per = torch.cat(losses)
        g = torch.cat(gammas)
        passes.append(float(per.mean()))
        pooled_l.append(per)
        pooled_g.append(g)
        print(f"case1 pass {seed} L1*={passes[-1]:.6f}", flush=True)
    per = torch.cat(pooled_l).numpy()
    g = torch.cat(pooled_g).numpy()
    idx = gamma_bin_index(torch.as_tensor(g)).numpy()
    bins = {}
    for b, lab in enumerate(BIN_LABELS):
        mask = idx == b
        bins[lab] = {"n": int(mask.sum()), "l1_star": float(per[mask].mean()) if mask.any() else None}
    payload = {
        "support_hi": int(ks[-1]),
        "missing_mass": missing,
        "passes": passes,
        "l1_star_mean": float(np.mean(passes)),
        "l1_star_std": float(np.std(passes)),
        "l1_star_pooled": float(per.mean()),
        "prior_bregman_mean20": float(prior.mean()),
        "prior_bregman_empirical_mean": float(prior_emp.mean()),
        "empirical_mean": emp,
        "bins": bins,
        "quantiles": _qstats(per),
    }
    dump(out, payload)
    return payload


def case1_model(device):
    out = RESULTS / "case1_model.json"
    if out.is_file():
        print(f"skip case1 model ({out})", flush=True)
        return load_json(out)
    ckpt = DIRECT_DIR / "best.pt"
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}")
    ks, logp, _missing = nb_support()
    _xtr, _xva, _train_loader, val_loader = data_loaders(device, batch_size=8192)
    model, args = load_model(ckpt, device)
    passes = []
    for seed in range(4):
        set_seed(100 + seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(100 + seed)
        model.eval()
        l1 = []
        star = []
        excess = []
        for (xb,) in val_loader:
            xb = xb.to(device)
            t = torch.rand(xb.shape[0], device=device) * (1.0 - args.t_eps) + args.t_eps
            z = q_sample(xb, t, args.lbd)
            gamma = (t * args.lbd).reshape(-1)
            alpha = t if args.z_rescale else None
            pred = model(z, t, alpha=alpha).reshape(-1).double()
            x = xb.reshape(-1).double().clamp_min(0)
            m_star = posterior_mean(z.reshape(-1), gamma, ks, logp)
            d_hat = full_bregman(x, pred)
            d_star = full_bregman(x, m_star)
            l1.append(d_hat.detach().cpu())
            star.append(d_star.detach().cpu())
            excess.append((d_hat - d_star).detach().cpu())
        passes.append(
            {
                "l1": float(torch.cat(l1).mean()),
                "l1_star": float(torch.cat(star).mean()),
                "excess": float(torch.cat(excess).mean()),
            }
        )
        print(
            f"case1 model pass {seed} L1={passes[-1]['l1']:.4f} "
            f"L1*={passes[-1]['l1_star']:.4f} excess={passes[-1]['excess']:.4f}",
            flush=True,
        )
    payload = {
        "ckpt": str(ckpt),
        "passes": passes,
        "l1": float(np.mean([p["l1"] for p in passes])),
        "l1_star": float(np.mean([p["l1_star"] for p in passes])),
        "excess": float(np.mean([p["excess"] for p in passes])),
    }
    dump(out, payload)
    del model
    return payload


def run_direct(device):
    need_cuda(device, "direct")
    train_subprocess(
        DIRECT_DIR,
        ["--moment_k", "1", "--moment_param", "direct"],
    )
    return poisson_row(DIRECT_DIR / "best.pt", RESULTS / "sample" / "direct", device)


def probe_one(model, loader, device, n_batches):
    model.train()
    traces = {param: {"norms": [], "cos": []} for param in PARAMS}
    seen = 0
    for (xb,) in loader:
        xb = xb.to(device)
        t = torch.rand(xb.shape[0], device=device) * (1.0 - T_EPS) + T_EPS
        z = q_sample(xb, t, LBD)
        alpha = t
        pred = model(z, t, alpha=alpha)
        for param in PARAMS:
            l1, l2, l3 = hm_head_losses(pred, xb, z, t, param)
            _record_trunk(model, [l1, l2, l3], traces[param])
        seen += 1
        if seen >= n_batches:
            break
    g1 = np.asarray(traces["direct_ratio"]["norms"], dtype=np.float64)[:, 0]
    g1b = np.asarray(traces["root_moment"]["norms"], dtype=np.float64)[:, 0]
    return {
        param: trace_summary(traces[param]) for param in PARAMS
    } | {"g1_norm_max_abs_diff": float(np.max(np.abs(g1 - g1b)))}


def hm_head_losses(pred, xb, z, t, param):
    gmin = T_EPS * LBD
    m, s = higher_moment_maps(pred, z, t * LBD, param, gmin=gmin)
    x = xb.reshape(-1).double().clamp_min(0)
    l1 = full_bregman(x, m[:, 0]).mean()
    target = raw_ratio_targets(xb, z, t * LBD, 3)
    l2 = full_bregman(target[:, 1], s[:, 1]).mean()
    l3 = full_bregman(target[:, 2], s[:, 2]).mean()
    return l1, l2, l3


def case3_probe(device):
    out = RESULTS / "case3_probe.json"
    if out.is_file():
        print(f"skip case3 probe ({out})", flush=True)
        return load_json(out)
    need_cuda(device, "case3 probe")
    ckpt = DIRECT_DIR / "best.pt"
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}")
    _xtr, _xva, train_loader, _val_loader = data_loaders(device)
    args = hm_namespace("direct_ratio", device)
    direct, _dargs = load_model(ckpt, device)

    set_seed(0)
    random_model = build_hm(device)
    random_probe = probe_one(random_model, train_loader, device, PROBE_BATCHES)

    set_seed(1)
    copied = build_hm(device)
    copy_m1_from_direct(direct, copied)
    gap = m1_probe(direct, copied, device)
    print(f"m1 probe max abs {gap:.3e}", flush=True)
    copied_probe = probe_one(copied, train_loader, device, PROBE_BATCHES)
    payload = {
        "batches": PROBE_BATCHES,
        "m1_copy_max_abs": gap,
        "shared_random_init": random_probe,
        "shared_direct_m1_init": copied_probe,
    }
    dump(out, payload)
    del direct, random_model, copied
    return payload


def case2(device):
    out = RESULTS / "case2_frozen.json"
    if out.is_file():
        print(f"skip case2 ({out})", flush=True)
        return load_json(out)
    need_cuda(device, "case2")
    ckpt = DIRECT_DIR / "best.pt"
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}")
    ks, logp, _missing = nb_support()
    _xtr, _xva, train_loader, val_loader = data_loaders(device)
    direct, dargs = load_model(ckpt, device)
    base_scores = eval_scores(direct, val_loader, dargs, device, ks, logp, kind="direct")
    rows = {"direct_val": base_scores}
    for param in PARAMS:
        print(f"case2 freeze-train {param}", flush=True)
        args = hm_namespace(param, device)
        set_seed(1)
        model = build_hm(device)
        copy_m1_from_direct(direct, model)
        gap = m1_probe(direct, model, device)
        if gap > 1e-4:
            raise SystemExit(f"m1 copy failed for {param}: max abs {gap}")
        trunk, last, zero_m1, _unfreeze = freeze_trunk_and_m1(model)
        opt = torch.optim.Adam([last.weight, last.bias], lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
        w0 = last.weight.detach()[0].clone()
        b0 = last.bias.detach()[0].clone()
        trunk0 = [p.detach().clone() for p in trunk]
        before = eval_scores(model, val_loader, args, device, ks, logp, kind="hm")
        set_seed(0)
        save_blob(FROZEN_ROOT / param / "epoch0.pt", model, args, 0, opt)
        for epoch in range(1, FREEZE_EPOCHS + 1):
            train_loss, train_l1, _cons, _h = run_epoch(
                model, train_loader, None, args, device, opt=opt, after_backward=zero_m1
            )
            if epoch % 10 == 0 or epoch == FREEZE_EPOCHS:
                print(
                    f"frozen {param} epoch {epoch} train {train_loss:.4f} L1 {train_l1:.4f}",
                    flush=True,
                )
        dw = float((last.weight.detach()[0] - w0).abs().max())
        db = float((last.bias.detach()[0] - b0).abs().max())
        dtrunk = max(float((p.detach() - q).abs().max()) for p, q in zip(trunk, trunk0))
        if max(dw, db, dtrunk) > 1e-8:
            raise SystemExit(f"freeze leaked for {param}: weight {dw} bias {db} trunk {dtrunk}")
        after = eval_scores(model, val_loader, args, device, ks, logp, kind="hm")
        if abs(after["l1"] - before["l1"]) > 1e-4:
            raise SystemExit(
                f"frozen L1 moved for {param}: {before['l1']} -> {after['l1']}"
            )
        ckpt_path = FROZEN_ROOT / param / "epoch40.pt"
        save_blob(ckpt_path, model, args, FREEZE_EPOCHS, opt)
        row0 = poisson_row(FROZEN_ROOT / param / "epoch0.pt", RESULTS / "sample" / "frozen" / param / "epoch0", device)
        row40 = poisson_row(ckpt_path, RESULTS / "sample" / "frozen" / param / "epoch40", device)
        rows[param] = {
            "m1_copy_max_abs": gap,
            "weight_drift": {"m1_row": dw, "m1_bias": db, "trunk": dtrunk},
            "val_before": before,
            "val_after": after,
            "poisson_epoch0": row0,
            "poisson_epoch40": row40,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows["freeze_epochs"] = FREEZE_EPOCHS
    dump(out, rows)
    del direct
    return rows


def case34(device):
    need_cuda(device, "case34")
    rows = {}
    for param in PARAMS:
        out_dir = JOINT_ROOT / param
        train_subprocess(
            out_dir,
            [
                "--k_max",
                "3",
                "--moment_loss",
                "hm",
                "--hm_param",
                param,
                "--hybrid_weight",
                "equal",
                "--weight_steps",
                "100",
                "--trace_trunk_epochs",
                str(TRACE_EPOCHS),
                "--save_best_l1",
                "true",
            ],
        )
        best = poisson_row(out_dir / "best.pt", RESULTS / "sample" / "joint" / param / "best_total", device)
        best_l1 = poisson_row(
            out_dir / "best_l1.pt", RESULTS / "sample" / "joint" / param / "best_l1", device
        )
        norms = np.load(out_dir / "grad_head_norms.npy")
        cos = np.load(out_dir / "grad_cosine.npy")
        steps = int(norms.shape[0])
        per_epoch = steps // TRACE_EPOCHS
        trace = {"norms": norms, "cos": cos}
        blob_best = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        blob_l1 = torch.load(out_dir / "best_l1.pt", map_location="cpu", weights_only=False)
        rows[param] = {
            "best_total_epoch": blob_best.get("epoch"),
            "best_total_val": blob_best.get("val_loss"),
            "best_total_val_l1": blob_best.get("val_l1"),
            "best_l1_epoch": blob_l1.get("epoch"),
            "best_l1_val": blob_l1.get("val_loss"),
            "best_l1_val_l1": blob_l1.get("val_l1"),
            "same_checkpoint": blob_best.get("epoch") == blob_l1.get("epoch"),
            "poisson_best_total": best,
            "poisson_best_l1": best_l1,
            "trunk_first_200": trace_summary(slice_trace(trace, min(200, steps))),
            "trunk_first_epoch": trace_summary(slice_trace(trace, min(per_epoch, steps))),
            "trunk_first_10_epochs": trace_summary(trace),
        }
    dump(RESULTS / "case34_joint.json", rows)
    return rows


def case5(device):
    out = RESULTS / "case5_unfreeze.json"
    if out.is_file():
        print(f"skip case5 ({out})", flush=True)
        return load_json(out)
    need_cuda(device, "case5")
    ks, logp, _missing = nb_support()
    _xtr, _xva, train_loader, val_loader = data_loaders(device)
    rows = {}
    for param in PARAMS:
        ckpt = FROZEN_ROOT / param / "epoch40.pt"
        if not ckpt.is_file():
            raise SystemExit(f"missing {ckpt}")
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        args = hm_namespace(param, device)
        model = build_hm(device)
        model.load_state_dict(blob["model"])
        trunk, last, _zero, unfreeze = freeze_trunk_and_m1(model)
        opt = torch.optim.Adam([last.weight, last.bias], lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
        if blob.get("opt") is not None:
            opt.load_state_dict(blob["opt"])
        unfreeze()
        opt.add_param_group({"params": trunk, "lr": LR, "betas": (0.9, 0.999), "weight_decay": 0.0})
        nodes = {}

        def measure(step):
            scores = eval_scores(model, val_loader, args, device, ks, logp, kind="hm")
            path = UNFREEZE_ROOT / param / f"step_{step}.pt"
            save_blob(path, model, args, step, opt)
            pois = poisson_row(path, RESULTS / "sample" / "unfreeze" / param / f"step_{step}", device)
            nodes[str(step)] = {"val": scores, "poisson": pois}
            print(
                f"unfreeze {param} step {step} L1={scores['l1']:.4f} "
                f"|log m1|={scores['abs_log_m1']:.4f} "
                f"WD={pois['wd']:.4f} TV={pois['tv']:.4f}",
                flush=True,
            )

        measure(0)
        model.train()
        seen = 0
        targets = set(UNFREEZE_STEPS)
        while seen < max(UNFREEZE_STEPS):
            for (xb,) in train_loader:
                xb = xb.to(device)
                opt.zero_grad(set_to_none=True)
                loss, _pred, _heads, _info = _batch_loss(model, xb, None, args)
                loss.backward()
                opt.step()
                seen += 1
                if seen in targets:
                    measure(seen)
                if seen >= max(UNFREEZE_STEPS):
                    break
        rows[param] = nodes
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    dump(out, rows)
    return rows


def _fmt_pair(row):
    if not row:
        return "missing"
    return f"{row['wd']:.3f} / {row['tv']:.3f}"


def _med(block, key):
    item = (block or {}).get(key) or {}
    med = item.get("median")
    if med is None:
        return "na"
    return f"{med:.3g}"


def write_summary():
    case1_o = load_json(RESULTS / "case1_oracle.json")
    case1_m = load_json(RESULTS / "case1_model.json")
    case2_o = load_json(RESULTS / "case2_frozen.json")
    case3_o = load_json(RESULTS / "case3_probe.json")
    case34_o = load_json(RESULTS / "case34_joint.json")
    case5_o = load_json(RESULTS / "case5_unfreeze.json")
    direct = load_json(RESULTS / "sample" / "direct" / "poisson.json")
    lines = []
    lines.append("NB trunk cases")
    lines.append("")
    lines.append("Case 1. Bayes floor on the NB validation set, same noise as training.")
    if case1_o:
        lines.append(
            f"  L1* = {case1_o['l1_star_mean']:.4f}  (std across 4 passes {case1_o['l1_star_std']:.4f})"
        )
        lines.append(
            f"  D(X, 20) with no observation = {case1_o['prior_bregman_mean20']:.4f}"
        )
        for lab, row in case1_o["bins"].items():
            lines.append(f"  bin {lab}  L1* = {row['l1_star']:.4f}")
    if case1_m:
        lines.append(
            f"  single-head L1 = {case1_m['l1']:.4f}   L1 - L1* = {case1_m['excess']:.4f}"
        )
    lines.append("")
    lines.append("Case 2. Freeze trunk and m1, train higher heads 40 epochs. Poisson WD/TV.")
    lines.append(
        f"  previous machine, single-head direct: {REF['direct_poisson']['wd']:.2f} / {REF['direct_poisson']['tv']:.3f}"
    )
    if direct:
        lines.append(f"  this run, single-head direct: {_fmt_pair(direct)}")
    if case2_o:
        for param in PARAMS:
            row = case2_o.get(param) or {}
            lines.append(
                f"  {param} epoch 0: {_fmt_pair(row.get('poisson_epoch0'))}    "
                f"epoch 40: {_fmt_pair(row.get('poisson_epoch40'))}"
            )
            lines.append(
                f"    val L1 before/after {row.get('val_before', {}).get('l1')} / {row.get('val_after', {}).get('l1')}"
            )
    lines.append("")
    lines.append("Case 3. Trunk gradients on the same minibatches. Median ||g|| and cosine.")
    lines.append("  m1 is softplus in both maps, so g1 should agree when the weights agree.")
    if case3_o:
        for label in ("shared_random_init", "shared_direct_m1_init"):
            block = case3_o.get(label) or {}
            lines.append(f"  {label}, g1 norm diff {block.get('g1_norm_max_abs_diff')}")
            for param in PARAMS:
                st = block.get(param) or {}
                lines.append(
                    f"    {param}  ||g|| {_med(st, 'g1')} {_med(st, 'g2')} {_med(st, 'g3')}  "
                    f"cos {_med(st, 'cos12')} {_med(st, 'cos13')} {_med(st, 'cos23')}  "
                    f"g2/g1 {st.get('g2_over_g1_median')}  g3/g1 {st.get('g3_over_g1_median')}"
                )
    if case34_o:
        lines.append("  first 200 updates of the from-scratch runs:")
        for param in PARAMS:
            st = (case34_o.get(param) or {}).get("trunk_first_200") or {}
            lines.append(
                f"    {param}  ||g|| {_med(st, 'g1')} {_med(st, 'g2')} {_med(st, 'g3')}  "
                f"cos {_med(st, 'cos12')} {_med(st, 'cos13')} {_med(st, 'cos23')}"
            )
    lines.append("")
    lines.append("Case 4. Best total loss vs best L1. Poisson WD/TV.")
    lines.append(
        f"  previous direct_ratio {REF['direct_ratio_equal']['wd']:.2f}/{REF['direct_ratio_equal']['tv']:.3f}    "
        f"root_moment {REF['root_moment_equal']['wd']:.2f}/{REF['root_moment_equal']['tv']:.3f}"
    )
    if case34_o:
        for param in PARAMS:
            row = case34_o.get(param) or {}
            lines.append(
                f"  {param} best total epoch {row.get('best_total_epoch')} "
                f"val {row.get('best_total_val')} L1 {row.get('best_total_val_l1')} "
                f"Poisson {_fmt_pair(row.get('poisson_best_total'))}"
            )
            lines.append(
                f"  {param} best L1    epoch {row.get('best_l1_epoch')} "
                f"val {row.get('best_l1_val')} L1 {row.get('best_l1_val_l1')} "
                f"Poisson {_fmt_pair(row.get('poisson_best_l1'))}"
            )
    lines.append("")
    lines.append("Case 5. After the frozen higher heads, unfreeze trunk.")
    if case5_o:
        for param in PARAMS:
            lines.append(f"  {param}")
            for step, node in (case5_o.get(param) or {}).items():
                val = node.get("val") or {}
                lines.append(
                    f"    step {step:>4}  L1 {val.get('l1')}  |log m1| {val.get('abs_log_m1')}  "
                    f"Poisson {_fmt_pair(node.get('poisson'))}"
                )
    text = "\n".join(lines) + "\n"
    path = RESULTS / "summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(text, end="", flush=True)
    return text


def selftest(device):
    from train import MLP

    set_seed(0)
    direct = PoissonDiscreteDiffusionModel(
        MLP(in_dim=1, hidden=128, out_dim=1, layers=3, continuous_t=True),
        lbd=LBD,
        z_rescale=True,
        clip_z=True,
        ratio=False,
    ).to(device)
    set_seed(1)
    hm = build_hm(device)
    copy_m1_from_direct(direct, hm)
    gap = m1_probe(direct, hm, device)
    print(f"selftest m1 max abs {gap:.3e}", flush=True)
    if gap > 1e-5:
        raise SystemExit(f"selftest copy failed: {gap}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        default="all",
        choices=(
            "all",
            "selftest",
            "case1",
            "direct",
            "case1_model",
            "case3_probe",
            "case2",
            "case34",
            "case5",
            "summary",
        ),
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = pick_device(args.device)
    if args.stage not in ("case1", "selftest", "summary") and device.type != "cuda":
        if args.stage == "all":
            raise SystemExit("cuda is not available. Case 1 can run with --stage case1.")
        raise SystemExit("cuda is not available")
    print(f"stage {args.stage} device {device}", flush=True)
    if args.stage == "selftest":
        selftest(device)
        return
    if args.stage == "case1":
        case1(device)
        return
    if args.stage == "summary":
        write_summary()
        return
    if args.stage in ("all", "direct"):
        run_direct(device)
    if args.stage in ("all", "case1"):
        case1(device)
    if args.stage in ("all", "case1_model"):
        case1_model(device)
    if args.stage in ("all", "case3_probe"):
        case3_probe(device)
    if args.stage in ("all", "case2"):
        case2(device)
    if args.stage in ("all", "case34"):
        case34(device)
    if args.stage in ("all", "case5"):
        case5(device)
    if args.stage == "all":
        write_summary()


if __name__ == "__main__":
    main()
