import argparse
import importlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from loss.consistency import consistency_squares
from loss.loss import (
    BregmanLoss,
    MomentPRLLoss,
    NormalizedRatioLoss,
    RawRatioBregmanLoss,
    moment_from_param,
    beta_div,
    full_bregman,
    higher_moment_maps,
    kernel_nll,
    linked_mean,
    quad_reverse_h,
    raw_multi_bregman,
    raw_multi_weights,
    moments_from_ratio_pred,
    normalized_ratio_targets,
    raw_ratio_targets,
    rising_factorial,
)
from loss.oracle import format_mom_line, gamma_bin_index, oracle_moments, support_logp
from model.small_diffusion import PoissonDiscreteDiffusionModel

ROOT = Path(__file__).resolve().parent
MLP = importlib.import_module("model.1d_mlp").MLP


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        return {}
    text = path.read_text()
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    cfg = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        cfg[k.strip()] = _parse_scalar(v.strip())
    return cfg


def _parse_scalar(v):
    if v in ("", "null", "None", "~"):
        return None
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def str2bool(v):
    if isinstance(v, bool):
        return v
    low = str(v).lower()
    if low in ("1", "true", "yes", "y"):
        return True
    if low in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {v}")


def pair_or_none(v):
    if v is None or v == "null":
        return None
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return [float(v[0]), float(v[1])]
    raise argparse.ArgumentTypeError(f"expected two floats, got {v}")


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "-c",
        "--config",
        default=str(ROOT / "config.yml"),
        help="yaml file; CLI flags override it",
    )
    pre_args, _ = pre.parse_known_args()
    cfg = load_yaml(pre_args.config)

    p = argparse.ArgumentParser(
        parents=[pre],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("data")
    g.add_argument("--data", default=cfg.get("data", "data/nb"), help="dataset folder")
    g.add_argument("--train_npy", default=cfg.get("train_npy", "train.npy"), help="train file")
    g.add_argument("--val_npy", default=cfg.get("val_npy", "val.npy"), help="val file")

    g = p.add_argument_group("model")
    g.add_argument("--hidden", type=int, default=int(cfg.get("hidden", 128)), help="MLP width")
    g.add_argument("--layers", type=int, default=int(cfg.get("layers", 3)), help="cond layers")
    g.add_argument(
        "--continuous_t",
        type=str2bool,
        default=cfg.get("continuous_t", True),
        help="t in [0,1], embed 1000*t",
    )
    g.add_argument("--lbd", type=float, default=float(cfg.get("lbd", 10.0)), help="max SNR λ")
    g.add_argument(
        "--scale",
        "--z_rescale",
        dest="z_rescale",
        type=str2bool,
        default=bool(cfg.get("scale", cfg.get("z_rescale", False))),
        help="if true, MLP sees z/(λ α_t)=z/γ",
    )
    g.add_argument("--clip_z", type=str2bool, default=cfg.get("clip_z", True), help="clamp z>=0")
    g.add_argument(
        "--clip_range",
        nargs=2,
        type=float,
        default=pair_or_none(cfg.get("clip_range")),
        metavar=("LO", "HI"),
        help="clamp xhat to [LO, HI]",
    )
    g.add_argument(
        "--normalize",
        nargs=2,
        type=float,
        default=pair_or_none(cfg.get("normalize")),
        metavar=("MEAN", "STD"),
        help="affine on x0",
    )
    g.add_argument(
        "--ratio",
        type=str2bool,
        default=bool(cfg.get("ratio", False)),
        help="train normalized PMF ratios b_k=log B_k instead of posterior mean",
    )
    g.add_argument(
        "--k_max",
        type=int,
        default=int(cfg.get("k_max", 3)),
        help="ratio heads k=1..k_max",
    )
    g.add_argument(
        "--ratio_loss",
        default=str(cfg.get("ratio_loss", "normalized")),
        choices=("normalized", "balanced", "raw"),
        help="normalized: e^b-Tb; balanced: D_k(e^b-Tb); raw: learn S_k not B_k",
    )
    g.add_argument(
        "--moment_k",
        type=int,
        default=int(cfg.get("moment_k", 0)),
        help="if >0, single-head moment PRL for E[X^k|z]; 0 keeps the old loss",
    )
    g.add_argument(
        "--moment_param",
        default=str(cfg.get("moment_param", "direct")),
        choices=("direct", "ratio", "raw"),
        help="direct softplus(f); ratio D_k e^b; raw D_k e^a / γ^k",
    )
    g.add_argument(
        "--moment_loss",
        default=str(cfg.get("moment_loss", "moment")),
        choices=("moment", "mse", "raw_ratio", "raw_multi", "hybrid", "raw_bal", "soft_ratio", "offset", "hm"),
        help="moment: Bregman on m̂; mse: (X-m̂)^2; raw_ratio: single-head D(R,S); raw_bal: (z+1)/gamma * D(R,S); raw_multi: weighted sum of D(R_k,S_k)",
    )
    g.add_argument(
        "--raw_weight",
        default=str(cfg.get("raw_weight", "equal")),
        choices=("equal", "dyn", "dyn_norm"),
        help="equal: 1+1+1; dyn: h+h^2/2+h^3/6; dyn_norm: those over their sum",
    )
    g.add_argument(
        "--weight_steps",
        type=int,
        default=int(cfg.get("weight_steps", cfg.get("sample_steps", 100))),
        help="T for quadratic h used by dyn weights. gamma(u)=lambda*u^2",
    )
    g.add_argument(
        "--consistency",
        default=str(cfg.get("consistency", "none")),
        choices=("none", "soft", "anchored", "hard"),
        help="none; soft or anchored add lambda * residual^2; hard trains L1 only",
    )
    g.add_argument(
        "--lambda_cons",
        type=float,
        default=float(cfg.get("lambda_cons", 1.0)),
        help="weight on the consistency residual",
    )
    g.add_argument(
        "--hm_param",
        default=str(cfg.get("hm_param", "raw_log")),
        choices=("raw_log", "direct_ratio", "log_moment", "root_moment", "direct_moment", "var_gap"),
        help="k=2,3 link. m1 stays softplus. Loss stays D(R_k, S_k).",
    )
    g.add_argument(
        "--hybrid_weight",
        default=str(cfg.get("hybrid_weight", "equal")),
        choices=("equal", "dyn"),
        help="hybrid_eq: 1,1,1 on (L1mean, L2ratio, L3ratio); hybrid_dyn: 1, h/2, h^2/6",
    )
    g.add_argument(
        "--cons_weight",
        default=str(cfg.get("cons_weight", "equal")),
        choices=("equal", "dyn_norm"),
        help="equal: delta2^2+delta3^2; dyn_norm: w2*delta2^2+w3*delta3^2",
    )
    g.add_argument(
        "--trace_trunk_epochs",
        type=int,
        default=int(cfg.get("trace_trunk_epochs", 0)),
        help="hm only: record trunk grad norms and cosines for the first N epochs",
    )
    g.add_argument(
        "--save_best_l1",
        type=str2bool,
        default=bool(cfg.get("save_best_l1", False)),
        help="also write best_l1.pt from validation L1",
    )
    g.add_argument(
        "--higher_target",
        default=str(cfg.get("higher_target", "ratio_kl")),
        choices=("ratio_kl", "ratio_mse", "moment_kl", "moment_mse", "mixed", "moment_beta"),
        help="hm k=2,3 target. L1 stays D(X, m1)",
    )
    g.add_argument(
        "--higher_scale",
        type=float,
        default=float(cfg.get("higher_scale", 1.0)),
        help="multiplier on the weighted L2 and L3 terms",
    )
    g.add_argument(
        "--higher_only",
        type=str2bool,
        default=bool(cfg.get("higher_only", False)),
        help="hm loss is weighted L2+L3, without L1",
    )
    g.add_argument(
        "--save_best_h",
        type=str2bool,
        default=bool(cfg.get("save_best_h", False)),
        help="write best_h.pt from lowest validation higher-head loss",
    )
    g.add_argument(
        "--save_epochs",
        default=str(cfg.get("save_epochs", "") or ""),
        help="comma-separated epochs to keep as epoch_N.pt",
    )
    g.add_argument("--moment_beta", type=float, default=float(cfg.get("moment_beta", 1.0)))
    g.add_argument(
        "--kernel_nll",
        default=str(cfg.get("kernel_nll", "none")),
        choices=("none", "nb", "twopois"),
    )
    g.add_argument("--kernel_lam", type=float, default=float(cfg.get("kernel_lam", 0.0)))
    g.add_argument("--l2_scale", type=float, default=float(cfg.get("l2_scale", 1.0)))
    g.add_argument("--l3_scale", type=float, default=float(cfg.get("l3_scale", 1.0)))
    g.add_argument("--freeze_heads", default=str(cfg.get("freeze_heads", "") or ""))
    g.add_argument("--init_ckpt", default=str(cfg.get("init_ckpt", "") or ""), help="load weights before training")
    g.add_argument(
        "--freeze_m1",
        type=str2bool,
        default=bool(cfg.get("freeze_m1", False)),
        help="freeze trunk and the m1 row; train higher-head rows only",
    )
    g.add_argument(
        "--detach_higher",
        type=str2bool,
        default=bool(cfg.get("detach_higher", False)),
        help="L2 and L3 do not backprop into the trunk",
    )
    g.add_argument(
        "--trunk_lr_mult",
        type=float,
        default=float(cfg.get("trunk_lr_mult", 1.0)),
        help="trunk Adam lr = lr * this. Head lr stays lr",
    )

    g = p.add_argument_group("train")
    g.add_argument("--epochs", type=int, default=int(cfg.get("epochs", 200)), help="epochs")
    g.add_argument("--batch_size", type=int, default=int(cfg.get("batch_size", 256)), help="batch")
    g.add_argument("--lr", type=float, default=float(cfg.get("lr", 1e-3)), help="Adam lr")
    g.add_argument("--beta1", type=float, default=float(cfg.get("beta1", 0.9)), help="Adam β1")
    g.add_argument("--beta2", type=float, default=float(cfg.get("beta2", 0.999)), help="Adam β2")
    g.add_argument(
        "--weight_decay",
        type=float,
        default=float(cfg.get("weight_decay", 0.0)),
        help="Adam wd",
    )
    g.add_argument(
        "--t_eps",
        type=float,
        default=float(cfg.get("t_eps", 1e-4)),
        help="t ~ Unif[t_eps, 1], γ=t*λ",
    )
    g.add_argument("--seed", type=int, default=int(cfg.get("seed", 0)), help="seed")

    g = p.add_argument_group("io")
    g.add_argument("--out_dir", default=cfg.get("out_dir", "runs/nb"), help="ckpt dir")
    g.add_argument("--device", default=cfg.get("device", "cuda"), help="cuda or cpu")
    g.add_argument(
        "--val_every",
        type=int,
        default=int(cfg.get("val_every", cfg.get("log_every", 10))),
        help="val + save latest/best every N ep",
    )
    return p.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_counts(path):
    x = np.load(path)
    t = torch.from_numpy(np.asarray(x)).float()
    if t.ndim == 1:
        t = t[:, None]
    return t


def make_loader(x, batch_size, shuffle):
    return DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
    )


def q_sample(x, t, lbd):
    gamma = t * lbd
    rate = gamma.unsqueeze(-1) * x.clamp_min(0.0)
    return torch.poisson(rate)


def _cons_per_example(model, z, t, alpha, pred, mode, w, cons_weight):
    d2, d3 = consistency_squares(model, z, t, alpha, pred, mode)
    if str(cons_weight) == "dyn_norm":
        ww = w[:, 1:].to(dtype=d2.dtype)
        return ww[:, 0] * d2 + ww[:, 1] * d3
    if str(cons_weight) != "equal":
        raise ValueError(cons_weight)
    return d2 + d3


def _raw_multi_terms(model, xb, args):
    t = torch.rand(xb.shape[0], device=xb.device) * (1.0 - args.t_eps) + args.t_eps
    z = q_sample(xb, t, args.lbd)
    alpha = t if args.z_rescale else None
    pred = model(z, t, alpha=alpha)
    gamma = t * args.lbd
    target = raw_ratio_targets(xb, z, gamma, int(args.k_max))
    per = raw_multi_bregman(pred, target)
    w = raw_multi_weights(
        t,
        getattr(args, "raw_weight", "equal"),
        args.lbd,
        int(getattr(args, "weight_steps", 100)),
        per.shape[-1],
    )
    weighted = per * w
    heads = [weighted[:, k].mean() for k in range(weighted.shape[-1])]
    data_per = weighted.sum(dim=-1)
    cons_per = None
    mode = str(getattr(args, "consistency", "none") or "none")
    if mode in ("soft", "anchored"):
        cons_per = _cons_per_example(
            model, z, t, alpha, pred, mode, w, getattr(args, "cons_weight", "equal")
        )
    return data_per, cons_per, pred, heads, gamma


def _hm_batch(model, xb, args):
    t = torch.rand(xb.shape[0], device=xb.device) * (1.0 - args.t_eps) + args.t_eps
    z = q_sample(xb, t, args.lbd)
    alpha = t if args.z_rescale else None
    pred = model(z, t, alpha=alpha)
    gmin = float(args.t_eps) * float(args.lbd)
    _m, s = higher_moment_maps(pred, z, t * args.lbd, args.hm_param, gmin=gmin)
    x = xb.reshape(-1).double().clamp_min(0)
    l1 = full_bregman(x, _m[:, 0])
    n_head = int(_m.shape[1])
    target_name = str(getattr(args, "higher_target", "ratio_kl"))
    if target_name == "ratio_kl":
        target = raw_ratio_targets(xb, z, t * args.lbd, n_head)
        l2 = full_bregman(target[:, 1], s[:, 1])
        l3 = full_bregman(target[:, 2], s[:, 2]) if n_head >= 3 else None
    elif target_name == "ratio_mse":
        target = raw_ratio_targets(xb, z, t * args.lbd, n_head)
        l2 = (target[:, 1] - s[:, 1]).square()
        l3 = (target[:, 2] - s[:, 2]).square() if n_head >= 3 else None
    elif target_name == "moment_kl":
        l2 = full_bregman(x.pow(2), _m[:, 1])
        l3 = full_bregman(x.pow(3), _m[:, 2]) if n_head >= 3 else None
    elif target_name == "moment_mse":
        l2 = (x.pow(2) - _m[:, 1]).square()
        l3 = (x.pow(3) - _m[:, 2]).square() if n_head >= 3 else None
    elif target_name == "moment_beta":
        beta = float(getattr(args, "moment_beta", 1.0))
        l2 = beta_div(x.pow(2), _m[:, 1], beta)
        l3 = beta_div(x.pow(3), _m[:, 2], beta) if n_head >= 3 else None
    elif target_name == "mixed":
        target = raw_ratio_targets(xb, z, t * args.lbd, n_head)
        l2 = full_bregman(target[:, 1], s[:, 1])
        l3 = full_bregman(x.pow(3), _m[:, 2]) if n_head >= 3 else None
    else:
        raise ValueError(target_name)
    h = quad_reverse_h(t, args.lbd, int(getattr(args, "weight_steps", 100)))
    if str(getattr(args, "hybrid_weight", "equal")) == "dyn":
        l2 = l2 * (h / 2.0)
        if l3 is not None:
            l3 = l3 * (h.square() / 6.0)
    elif str(getattr(args, "hybrid_weight", "equal")) != "equal":
        raise ValueError(args.hybrid_weight)
    scale = float(getattr(args, "higher_scale", 1.0))
    l2 = l2 * scale * float(getattr(args, "l2_scale", 1.0))
    if l3 is not None:
        l3 = l3 * scale * float(getattr(args, "l3_scale", 1.0))
    high = l2 if l3 is None else l2 + l3
    kind = str(getattr(args, "kernel_nll", "none") or "none")
    kn = None
    if kind != "none":
        kk = torch.poisson((h.float() * x.float()).clamp_min(0)).double()
        if kind == "nb":
            kn = kernel_nll("nb", kk, h, _m[:, 0], _m[:, 1])
        else:
            if n_head < 3:
                raise ValueError("twopois kernel nll needs 3 heads")
            kn = kernel_nll("twopois", kk, h, _m[:, 0], _m[:, 1], _m[:, 2])
    heads = [l1.mean(), l2.mean()] if l3 is None else [l1.mean(), l2.mean(), l3.mean()]
    lam = float(getattr(args, "kernel_lam", 0.0))
    if kn is not None and lam <= 0.0:
        objective = kn
    elif kn is not None:
        objective = kn + lam * high
        if not bool(getattr(args, "higher_only", False)):
            objective = l1 + objective
    elif bool(getattr(args, "higher_only", False)):
        objective = high
    else:
        objective = l1 + high
    loss = objective.mean()
    return loss, pred, heads, {
        "data": float(l1.detach().mean()),
        "cons": None,
        "h": float(objective.detach().mean()),
    }


def _hybrid_batch(model, xb, args):
    """L1 = D(X, m_hat_1). L2, L3 stay raw-ratio Bregman on S_hat."""
    t = torch.rand(xb.shape[0], device=xb.device) * (1.0 - args.t_eps) + args.t_eps
    z = q_sample(xb, t, args.lbd)
    alpha = t if args.z_rescale else None
    pred = model(z, t, alpha=alpha)
    gamma = t * args.lbd
    z64 = z.reshape(-1).double()
    g = gamma.reshape(-1).double().clamp_min(1e-12)
    log_m1 = pred[:, 0].double() + (z64 + 1.0).clamp_min(1e-12).log() - g.log()
    m1 = log_m1.exp()
    x = xb.reshape(-1).double().clamp_min(0)
    l1 = full_bregman(x, m1)
    target = raw_ratio_targets(xb, z, gamma, 3)
    s = pred.double().exp()
    l2 = full_bregman(target[:, 1], s[:, 1])
    l3 = full_bregman(target[:, 2], s[:, 2])
    if str(getattr(args, "hybrid_weight", "equal")) == "dyn":
        h = quad_reverse_h(t, args.lbd, int(getattr(args, "weight_steps", 100)))
        l2 = l2 * (h / 2.0)
        l3 = l3 * (h.square() / 6.0)
    elif str(getattr(args, "hybrid_weight", "equal")) != "equal":
        raise ValueError(args.hybrid_weight)
    terms = (l1, l2, l3)
    heads = [term.mean() for term in terms]
    loss = (l1 + l2 + l3).mean()
    info = {"data": float(l1.detach().mean()), "cons": None}
    return loss, pred, heads, info


def _raw_bal_batch(model, xb, args):
    """Single head a = log S. L = (z+1)/gamma * D(R, exp(a)), R = gamma X / (z+1)."""
    t = torch.rand(xb.shape[0], device=xb.device) * (1.0 - args.t_eps) + args.t_eps
    z = q_sample(xb, t, args.lbd)
    alpha = t if args.z_rescale else None
    pred = model(z, t, alpha=alpha)
    gamma = t * args.lbd
    z64 = z.reshape(-1).double()
    g = gamma.reshape(-1).double().clamp_min(1e-12)
    denom = (z64 + 1.0).clamp_min(1e-12)
    x = xb.reshape(-1).double().clamp_min(0)
    r = g * x / denom
    s = pred.reshape(-1).double().exp()
    per = (denom / g) * full_bregman(r, s)
    loss = per.mean()
    return loss, pred, None, {"data": float(loss.detach()), "cons": None}


def _link_batch(model, xb, args, kind):
    """L = D(X, m). soft_ratio: m = (z+1)/gamma * softplus(a). offset: m = exp(b)."""
    t = torch.rand(xb.shape[0], device=xb.device) * (1.0 - args.t_eps) + args.t_eps
    z = q_sample(xb, t, args.lbd)
    alpha = t if args.z_rescale else None
    pred = model(z, t, alpha=alpha)
    gmin = float(args.t_eps) * float(args.lbd)
    m = linked_mean(pred, z, t * args.lbd, kind, gmin=gmin)
    x = xb.reshape(-1).double().clamp_min(0)
    loss = full_bregman(x, m).mean()
    return loss, pred, None, {"data": float(loss.detach()), "cons": None}


def _batch_loss(model, xb, loss_fn, args):
    kind = str(getattr(args, "moment_loss", "moment"))
    if kind in ("soft_ratio", "offset"):
        return _link_batch(model, xb, args, kind)
    if kind == "raw_bal":
        return _raw_bal_batch(model, xb, args)
    if kind == "hm":
        return _hm_batch(model, xb, args)
    if kind == "hybrid":
        return _hybrid_batch(model, xb, args)
    if kind == "raw_multi":
        data_per, cons_per, pred, heads, _gamma = _raw_multi_terms(model, xb, args)
        data = data_per.mean()
        cons = None if cons_per is None else cons_per.mean()
        loss = data if cons is None else data + float(args.lambda_cons) * cons
        info = {
            "data": float(data.detach()),
            "cons": None if cons is None else float(cons.detach()),
        }
        return loss, pred, heads, info
    t = torch.rand(xb.shape[0], device=xb.device) * (1.0 - args.t_eps) + args.t_eps
    z = q_sample(xb, t, args.lbd)
    alpha = t if args.z_rescale else None
    pred = model(z, t, alpha=alpha)
    k = int(getattr(args, "moment_k", 0) or 0)
    if k > 0 and kind == "mse":
        gamma = t * args.lbd
        m_hat = moment_from_param(pred, z, gamma, k, args.moment_param)
        x = xb.reshape(-1).double().clamp_min(0)
        m = m_hat.reshape(-1).double()
        loss = (x - m).square().mean()
        l1 = full_bregman(x, m.clamp_min(1e-12)).mean()
        return loss, pred, None, {"data": float(l1.detach()), "cons": None}
    if k > 0 and kind == "raw_ratio":
        gamma = t * args.lbd
        target = raw_ratio_targets(xb, z, gamma, k)[:, k - 1]
        loss = loss_fn(pred.reshape(-1), target)
        return loss, pred, None, {"data": float(loss.detach()), "cons": None}
    if k > 0:
        gamma = t * args.lbd
        m_hat = moment_from_param(pred, z, gamma, k, args.moment_param)
        loss = loss_fn(m_hat, xb, k)
        return loss, pred, None, {"data": float(loss.detach()), "cons": None}
    if args.ratio:
        ratio_kind = str(getattr(args, "ratio_loss", "normalized"))
        if ratio_kind == "raw":
            target = raw_ratio_targets(xb, z, t * args.lbd, args.k_max)
        else:
            target = normalized_ratio_targets(xb, z, args.k_max)
        loss = loss_fn(pred, target, z=z)
        return loss, pred, None, {"data": float(loss.detach()), "cons": None}
    loss = loss_fn(pred, model.encode_x(xb))
    return loss, pred, None, {"data": float(loss.detach()), "cons": None}


def trunk_parameters(model):
    """Shared trunk. The last linear maps one row per head, so it is excluded."""
    last = model.net.out_fc[-1]
    skip = {id(p) for p in last.parameters()}
    return [p for p in model.parameters() if p.requires_grad and id(p) not in skip]


def _flat_grads(params, grads):
    parts = []
    for p, g in zip(params, grads):
        if g is None:
            parts.append(torch.zeros(p.numel(), dtype=torch.float64, device=p.device))
        else:
            parts.append(g.detach().reshape(-1).double())
    return torch.cat(parts)


def _record_trunk(model, heads, trace):
    params = trunk_parameters(model)
    vecs = []
    for head in heads:
        gs = torch.autograd.grad(head, params, retain_graph=True, allow_unused=True)
        vecs.append(_flat_grads(params, gs))
    norms = [float(torch.linalg.vector_norm(v)) for v in vecs]
    pairs = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            ni = torch.linalg.vector_norm(vecs[i])
            nj = torch.linalg.vector_norm(vecs[j])
            if float(ni) < 1e-12 or float(nj) < 1e-12:
                pairs.append(float("nan"))
            else:
                pairs.append(float(torch.dot(vecs[i], vecs[j]) / (ni * nj)))
    trace["norms"].append(norms)
    trace["cos"].append(pairs)


def _trunk_norm(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.linalg.vector_norm(_flat_grads(params, gs)))


def eval_cons_grad(model, loader, args, device, min_count=8):
    """Trunk ||grad L_cons|| / ||grad L_data|| by gamma bin, one ratio per val batch."""
    from loss.oracle import BIN_LABELS

    mode = str(getattr(args, "consistency", "none") or "none")
    if mode not in ("soft", "anchored"):
        return None
    was_training = model.training
    model.train()
    params = trunk_parameters(model)
    keys = ("all",) + tuple(BIN_LABELS)
    buckets = {key: {"r": [], "g_data": [], "g_cons": [], "l_data": [], "l_cons": []} for key in keys}
    lam = float(args.lambda_cons)
    for (xb,) in loader:
        xb = xb.to(device)
        data_per, cons_per, _pred, _heads, gamma = _raw_multi_terms(model, xb, args)
        if cons_per is None:
            continue
        idx = gamma_bin_index(gamma).to(device=device)
        pieces = [("all", torch.ones(xb.shape[0], dtype=torch.bool, device=device))]
        for b, lab in enumerate(BIN_LABELS):
            pieces.append((lab, idx == b))
        for key, mask in pieces:
            if int(mask.sum()) < min_count:
                continue
            ld = data_per[mask].mean()
            lc = cons_per[mask].mean()
            gd = _trunk_norm(ld, params)
            gc = _trunk_norm(lc, params)
            buckets[key]["l_data"].append(float(ld.detach()))
            buckets[key]["l_cons"].append(float(lc.detach()))
            buckets[key]["g_data"].append(gd)
            buckets[key]["g_cons"].append(gc)
            if gd < 1e-12:
                buckets[key]["r"].append(float("nan"))
            else:
                buckets[key]["r"].append(gc / gd)
        del data_per, cons_per
    model.train(was_training)
    rows = []
    for key in keys:
        bucket = buckets[key]
        rows.append(
            {
                "bin": key,
                "n": len(bucket["r"]),
                "r": _qstats(bucket["r"]),
                "g_data": _qstats(bucket["g_data"]),
                "g_cons": _qstats(bucket["g_cons"]),
                "l_data": _qstats(bucket["l_data"]),
                "l_cons": _qstats(bucket["l_cons"]),
                "lambda_cons": lam,
            }
        )
    return rows


def _fmt_r(row):
    med = row["r"]["median"]
    p99 = row["r"]["p99"]
    if med is None:
        return f"{row['bin']}=na"
    return f"{row['bin']}={med:.3g}/{p99:.3g}"


def _grad_l2(model):
    sq = None
    for p in model.parameters():
        if p.grad is None:
            continue
        v = p.grad.detach().double().pow(2).sum()
        sq = v if sq is None else sq + v
    if sq is None:
        return float("nan")
    return float(torch.sqrt(sq))


def run_epoch(
    model, loader, loss_fn, args, device, opt=None, grad_norms=None, head_trace=None, after_backward=None
):
    train = opt is not None
    model.train(train)
    total = 0.0
    data_total = 0.0
    cons_total = 0.0
    h_total = 0.0
    n = 0
    n_cons = 0
    n_h = 0
    for (xb,) in loader:
        xb = xb.to(device)
        loss, _, heads, info = _batch_loss(model, xb, loss_fn, args)
        if train:
            opt.zero_grad(set_to_none=True)
            if heads is not None and head_trace is not None:
                _record_trunk(model, heads, head_trace)
            loss.backward()
            if after_backward is not None:
                after_backward()
            if grad_norms is not None:
                grad_norms.append(_grad_l2(model))
            opt.step()
        bs = xb.shape[0]
        total += float(loss.detach()) * bs
        data_total += float(info["data"]) * bs
        if info["cons"] is not None:
            cons_total += float(info["cons"]) * bs
            n_cons += bs
        if info.get("h") is not None:
            h_total += float(info["h"]) * bs
            n_h += bs
        n += bs
    cons_mean = cons_total / n_cons if n_cons else None
    h_mean = h_total / n_h if n_h else None
    return total / max(n, 1), data_total / max(n, 1), cons_mean, h_mean


@torch.no_grad()
def eval_moment_bins(model, loader, args, device, xs, logp):
    """Mean |log m_hat_k - log m*_k| on val, binned by γ."""
    k_max = int(args.k_max) if args.ratio else 1
    n_bins = 4
    sums = torch.zeros(k_max, n_bins, device=device, dtype=torch.float64)
    cnt = torch.zeros(n_bins, device=device, dtype=torch.float64)
    for (xb,) in loader:
        xb = xb.to(device)
        t = torch.rand(xb.shape[0], device=device) * (1.0 - args.t_eps) + args.t_eps
        z = q_sample(xb, t, args.lbd)
        gamma = t * args.lbd
        alpha = t if args.z_rescale else None
        pred = model(z, t, alpha=alpha)
        mk = int(getattr(args, "moment_k", 0) or 0)
        if str(getattr(args, "moment_loss", "")) == "hm":
            gmin = float(args.t_eps) * float(args.lbd)
            m, _s = higher_moment_maps(pred, z, gamma, str(args.hm_param), gmin=gmin)
            k_use = min(k_max, 3, m.shape[-1])
            hats = [m[:, k] for k in range(k_use)]
            stars = oracle_moments(z, gamma, xs, logp, k_max=k_use)
        elif mk > 0:
            hats = [moment_from_param(pred, z, gamma, mk, args.moment_param)]
            k_use = 1
            stars = [oracle_moments(z, gamma, xs, logp, k_max=mk)[mk - 1]]
        elif args.ratio:
            hats = moments_from_ratio_pred(pred, z, gamma, args)
            k_use = min(k_max, len(hats))
            stars = oracle_moments(z, gamma, xs, logp, k_max=k_use)
        else:
            hats = [pred]
            k_use = 1
            stars = oracle_moments(z, gamma, xs, logp, k_max=1)
        idx = gamma_bin_index(gamma).to(device=device, dtype=torch.int64)
        ones = torch.ones(idx.shape[0], device=device, dtype=torch.float64)
        cnt.scatter_add_(0, idx, ones)
        for k in range(k_use):
            h = hats[k].reshape(-1).double().clamp_min(1e-12)
            s = stars[k].reshape(-1).double().clamp_min(1e-12)
            err = (h.log() - s.log()).abs()
            sums[k].scatter_add_(0, idx, err)
    means = torch.full_like(sums, float("nan"))
    ok = cnt > 0
    means[:, ok] = sums[:, ok] / cnt[ok]
    return means.cpu(), cnt.cpu()


def _qstats(arr):
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


@torch.no_grad()
def eval_moment_quantiles(model, loader, args, device, xs, logp):
    """Signed log error and |m̂-m*| by γ bin. Single moment_k head."""
    from loss.oracle import BIN_LABELS

    mk = int(args.moment_k)
    buckets = {b: {"e": [], "abs": []} for b in range(4)}
    for (xb,) in loader:
        xb = xb.to(device)
        t = torch.rand(xb.shape[0], device=device) * (1.0 - args.t_eps) + args.t_eps
        z = q_sample(xb, t, args.lbd)
        gamma = t * args.lbd
        alpha = t if args.z_rescale else None
        pred = model(z, t, alpha=alpha)
        hat = moment_from_param(pred, z, gamma, mk, args.moment_param).reshape(-1).double()
        star = oracle_moments(z, gamma, xs, logp, k_max=mk)[mk - 1].reshape(-1).double()
        e = hat.clamp_min(1e-12).log() - star.clamp_min(1e-12).log()
        ad = (hat - star).abs()
        idx = gamma_bin_index(gamma).cpu().numpy()
        e_np = e.cpu().numpy()
        ad_np = ad.cpu().numpy()
        for b in range(4):
            m = idx == b
            if m.any():
                buckets[b]["e"].append(e_np[m])
                buckets[b]["abs"].append(ad_np[m])
    rows = []
    for b, lab in enumerate(BIN_LABELS):
        e = np.concatenate(buckets[b]["e"]) if buckets[b]["e"] else np.array([])
        ad = np.concatenate(buckets[b]["abs"]) if buckets[b]["abs"] else np.array([])
        rows.append({"bin": lab, "log": _qstats(e), "abs": _qstats(ad)})
    return rows


def grad_summary(norms):
    return _qstats(norms)


def _finite_col(col):
    a = np.asarray(col, dtype=np.float64).reshape(-1)
    return a[np.isfinite(a)]


def cosine_summary(col):
    a = _finite_col(col)
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.10)),
        "p90": float(np.quantile(a, 0.90)),
    }


def head_norm_rows(norms):
    arr = np.asarray(norms, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    rows = []
    for k in range(arr.shape[1]):
        st = _qstats(arr[:, k])
        rows.append({"head": k + 1, **st})
    return rows


def _write_table(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = list(rows[0].keys())
    lines = [" ".join(keys)]
    for row in rows:
        bits = []
        for key in keys:
            val = row[key]
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                bits.append("nan")
            elif isinstance(val, float):
                bits.append(f"{val:.6g}")
            else:
                bits.append(str(val))
        lines.append(" ".join(bits))
    Path(path).write_text("\n".join(lines) + "\n")


def head_cos_rows(cos):
    arr = np.asarray(cos, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    pairs = [(i, j) for i in range(1, 4) for j in range(i + 1, 4)]
    rows = []
    for c in range(arr.shape[1]):
        i, j = pairs[c] if c < len(pairs) else (c + 1, c + 2)
        rows.append({"pair": f"{i}{j}", **cosine_summary(arr[:, c])})
    return rows


def _fmt_med_p99(stats):
    med, p99 = stats.get("median"), stats.get("p99")
    if med is None or p99 is None:
        return "na"
    return f"{med:.3g}/{p99:.3g}"


@torch.no_grad()
def eval_raw_ratio_quantiles(model, loader, args, device, xs, logp):
    """log(Ŝ/S*), log(m̂/m*), |m̂-m*|, and target R scale, by γ bin."""
    from loss.oracle import BIN_LABELS

    mk = int(args.moment_k)
    buckets = {b: {"ratio": [], "mom": [], "abs": [], "target": []} for b in range(4)}
    for (xb,) in loader:
        xb = xb.to(device)
        t = torch.rand(xb.shape[0], device=device) * (1.0 - args.t_eps) + args.t_eps
        z = q_sample(xb, t, args.lbd)
        gamma = t * args.lbd
        alpha = t if args.z_rescale else None
        log_s = model(z, t, alpha=alpha).reshape(-1).double()
        target = raw_ratio_targets(xb, z, gamma, mk)[:, mk - 1]
        d_k = rising_factorial(z, mk)[:, mk - 1]
        g = gamma.reshape(-1).double().clamp_min(1e-12)
        s_hat = log_s.exp()
        m_hat = d_k * s_hat / g.pow(mk)
        m_star = oracle_moments(z, gamma, xs, logp, k_max=mk)[mk - 1].reshape(-1).double()
        s_star = m_star * g.pow(mk) / d_k.clamp_min(1e-12)
        e_s = s_hat.clamp_min(1e-12).log() - s_star.clamp_min(1e-12).log()
        e_m = m_hat.clamp_min(1e-12).log() - m_star.clamp_min(1e-12).log()
        ad = (m_hat - m_star).abs()
        idx = gamma_bin_index(gamma).cpu().numpy()
        packs = {
            "ratio": e_s.cpu().numpy(),
            "mom": e_m.cpu().numpy(),
            "abs": ad.cpu().numpy(),
            "target": target.cpu().numpy(),
        }
        for b in range(4):
            m = idx == b
            if not m.any():
                continue
            for key, arr in packs.items():
                buckets[b][key].append(arr[m])
    rows = []
    for b, lab in enumerate(BIN_LABELS):
        row = {"bin": lab}
        for key in ("ratio", "mom", "abs", "target"):
            parts = buckets[b][key]
            arr = np.concatenate(parts) if parts else np.array([])
            row[key] = _qstats(arr)
        rows.append(row)
    return rows


@torch.no_grad()
def eval_raw_multi_quantiles(model, loader, args, device, xs, logp):
    """Per-head log(S/S*), log(m/m*), |m-m*| by gamma bin."""
    from loss.oracle import BIN_LABELS

    k_max = int(args.k_max)
    buckets = {
        k: {b: {"ratio": [], "mom": [], "abs": []} for b in range(4)}
        for k in range(k_max)
    }
    for (xb,) in loader:
        xb = xb.to(device)
        t = torch.rand(xb.shape[0], device=device) * (1.0 - args.t_eps) + args.t_eps
        z = q_sample(xb, t, args.lbd)
        gamma = t * args.lbd
        alpha = t if args.z_rescale else None
        log_s = model(z, t, alpha=alpha).double()
        d_all = rising_factorial(z, k_max)
        g = gamma.reshape(-1).double().clamp_min(1e-12)
        stars = oracle_moments(z, gamma, xs, logp, k_max=k_max)
        idx = gamma_bin_index(gamma).cpu().numpy()
        for k in range(k_max):
            s_hat = log_s[:, k].exp()
            d_k = d_all[:, k]
            m_hat = d_k * s_hat / g.pow(k + 1)
            m_star = stars[k].reshape(-1).double()
            s_star = m_star * g.pow(k + 1) / d_k.clamp_min(1e-12)
            packs = {
                "ratio": (s_hat.clamp_min(1e-12).log() - s_star.clamp_min(1e-12).log()).cpu().numpy(),
                "mom": (m_hat.clamp_min(1e-12).log() - m_star.clamp_min(1e-12).log()).cpu().numpy(),
                "abs": (m_hat - m_star).abs().cpu().numpy(),
            }
            for b in range(4):
                m = idx == b
                if not m.any():
                    continue
                for key, arr in packs.items():
                    buckets[k][b][key].append(arr[m])
    rows = []
    for k in range(k_max):
        for b, lab in enumerate(BIN_LABELS):
            row = {"k": k + 1, "bin": lab}
            for key in ("ratio", "mom", "abs"):
                parts = buckets[k][b][key]
                arr = np.concatenate(parts) if parts else np.array([])
                row[key] = _qstats(arr)
            rows.append(row)
    return rows


def mom_payload(means, counts):
    arr = means.numpy() if torch.is_tensor(means) else np.asarray(means)
    cnt = counts.numpy() if torch.is_tensor(counts) else np.asarray(counts)
    from loss.oracle import BIN_LABELS

    out = {"bins": list(BIN_LABELS), "count": [int(x) for x in cnt.tolist()]}
    for k in range(arr.shape[0]):
        out[f"m{k + 1}"] = [float(x) for x in arr[k].tolist()]
    return out


def apply_consistency(args):
    """Soft and anchored train three raw heads. Hard trains L1 only.

    raw_weight is left as requested. equal is 1,1,1. dyn_norm is
    (h, h^2/2, h^3/6) divided by their sum.
    """
    mode = str(getattr(args, "consistency", "none") or "none")
    args.consistency = mode
    if str(getattr(args, "moment_loss", "")) == "hm" and mode != "none":
        raise SystemExit("hm keeps L2 and L3 as D(R_k, S_k) and does not add consistency")
    if str(getattr(args, "moment_loss", "")) == "hybrid" and mode != "none":
        raise SystemExit("hybrid trains L1 as D(X, m1) and does not add a consistency penalty")
    if str(getattr(args, "moment_loss", "")) == "raw_bal" and mode != "none":
        raise SystemExit("raw_bal is a single-head loss and does not add a consistency penalty")
    args.lambda_cons = float(getattr(args, "lambda_cons", 1.0))
    if mode == "none":
        return
    if mode in ("soft", "anchored"):
        args.moment_loss = "raw_multi"
        weight = str(getattr(args, "raw_weight", "equal") or "equal")
        args.raw_weight = weight
        args.k_max = 3
        args.ratio = True
        args.moment_param = "raw"
        args.moment_k = 0
        cw = str(getattr(args, "cons_weight", "equal") or "equal")
        args.cons_weight = cw
        if cw == "dyn_norm":
            cons_term = "w2*delta2^2 + w3*delta3^2"
        else:
            cons_term = "delta2^2 + delta3^2"
        print(
            f"consistency {mode}: raw_weight={weight}, cons_weight={cw}, "
            f"L = w1*L1+w2*L2+w3*L3 + {args.lambda_cons:g} * ({cons_term})",
            flush=True,
        )
        return
    if mode == "hard":
        args.moment_loss = "raw_ratio"
        args.moment_k = 1
        args.moment_param = "raw"
        args.ratio = True
        args.k_max = 1
        print("consistency hard: L = L1; S2 and S3 are products of S1 at inference", flush=True)
        return
    raise SystemExit(f"unknown consistency mode {mode}")


def load_init(model, path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    src = blob["model"]
    owned = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for key, value in src.items():
            if key not in owned:
                raise KeyError(key)
            slot = owned[key]
            if tuple(slot.shape) == tuple(value.shape):
                slot.copy_(value)
            elif key.endswith("out_fc.1.weight") and int(value.shape[0]) < int(slot.shape[0]) and tuple(value.shape[1:]) == tuple(slot.shape[1:]):
                slot[: int(value.shape[0])].copy_(value)
            elif key.endswith("out_fc.1.bias") and value.ndim == 1 and int(value.shape[0]) < int(slot.shape[0]):
                slot[: int(value.shape[0])].copy_(value)
            else:
                raise RuntimeError(f"cannot copy {key} {tuple(value.shape)} -> {tuple(slot.shape)}")
    model.load_state_dict(owned)


def freeze_trunk_heads(model, heads):
    last = model.net.out_fc[-1]
    for p in model.parameters():
        p.requires_grad_(p is last.weight or p is last.bias)
    idx = [int(i) for i in heads]

    def zero_heads():
        if last.weight.grad is not None:
            for i in idx:
                last.weight.grad[i].zero_()
        if last.bias is not None and last.bias.grad is not None:
            for i in idx:
                last.bias.grad[i].zero_()

    return zero_heads


def freeze_trunk_m1(model):
    return freeze_trunk_heads(model, [0])


def make_optimizer(model, args):
    betas = (args.beta1, args.beta2)
    wd = args.weight_decay
    last = model.net.out_fc[-1]
    if float(getattr(args, "trunk_lr_mult", 1.0)) != 1.0 and not bool(getattr(args, "freeze_m1", False)):
        trunk = [p for p in model.parameters() if p is not last.weight and p is not last.bias and p.requires_grad]
        heads = [p for p in (last.weight, last.bias) if p is not None and p.requires_grad]
        return torch.optim.Adam(
            [
                {"params": trunk, "lr": float(args.lr) * float(args.trunk_lr_mult)},
                {"params": heads, "lr": float(args.lr)},
            ],
            betas=betas,
            weight_decay=wd,
        )
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=args.lr, betas=betas, weight_decay=wd)


def main():
    args = parse_args()
    apply_consistency(args)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data_dir = resolve(args.data)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xtr = load_counts(data_dir / args.train_npy)
    train_loader = make_loader(xtr, args.batch_size, shuffle=True)
    val_path = data_dir / args.val_npy
    val_loader = None
    if val_path.is_file():
        val_loader = make_loader(load_counts(val_path), args.batch_size, shuffle=False)

    moment_k = int(getattr(args, "moment_k", 0) or 0)
    moment_loss = str(getattr(args, "moment_loss", "moment"))
    if moment_loss in ("raw_multi", "hybrid"):
        args.ratio = True
        args.k_max = 3
        args.moment_param = "raw"
        args.moment_k = 0
        moment_k = 0
        if moment_loss == "hybrid":
            hw = str(getattr(args, "hybrid_weight", "equal") or "equal")
            if hw not in ("equal", "dyn"):
                raise SystemExit(f"unknown hybrid_weight {hw}")
            args.hybrid_weight = hw
    elif moment_loss == "hm":
        hp = str(getattr(args, "hm_param", "") or "")
        if hp not in ("raw_log", "direct_ratio", "log_moment", "root_moment", "direct_moment", "var_gap"):
            raise SystemExit(f"unknown hm_param {hp}")
        args.hm_param = hp
        hw = str(getattr(args, "hybrid_weight", "equal") or "equal")
        if hw not in ("equal", "dyn"):
            raise SystemExit(f"unknown hybrid_weight {hw}")
        args.hybrid_weight = hw
        args.ratio = True
        if int(args.k_max) not in (2, 3):
            args.k_max = 3
        args.moment_param = "direct"
        args.moment_k = 0
        moment_k = 0
    elif moment_loss in ("soft_ratio", "offset"):
        args.ratio = True
        args.k_max = 1
        args.moment_param = "direct"
        args.moment_k = 0
        moment_k = 0
    elif moment_loss == "raw_bal":
        args.ratio = True
        args.k_max = 1
        args.moment_param = "raw"
        args.moment_k = 1
        moment_k = 1
    elif moment_loss == "raw_ratio":
        if moment_k <= 0:
            raise SystemExit("moment_loss=raw_ratio needs --moment_k > 0")
        args.ratio = True
        args.k_max = 1
        args.moment_param = "raw"
    elif moment_k > 0 and args.moment_param in ("ratio", "raw"):
        args.ratio = True
        args.k_max = 1
    elif str(getattr(args, "ratio_loss", "normalized")) in ("balanced", "raw"):
        args.ratio = True
    if args.ratio and moment_k == 0:
        xd = xtr.reshape(-1).double()
        args.prior_moments = [float(xd.pow(k).mean()) for k in range(1, int(args.k_max) + 1)]
    if str(getattr(args, "consistency", "none")) == "hard":
        xd = xtr.reshape(-1).double()
        args.prior_moments = [float(xd.pow(k).mean()) for k in range(1, 4)]

    if args.ratio:
        RatioMLP = importlib.import_module("model.ratio_mlp").RatioMLP
        net = RatioMLP(
            in_dim=xtr.shape[-1],
            hidden=args.hidden,
            out_dim=int(args.k_max),
            layers=args.layers,
            continuous_t=args.continuous_t,
            detach_higher=bool(getattr(args, "detach_higher", False)),
        )
    else:
        net = MLP(
            in_dim=xtr.shape[-1],
            hidden=args.hidden,
            out_dim=xtr.shape[-1],
            layers=args.layers,
            continuous_t=args.continuous_t,
        )
    model = PoissonDiscreteDiffusionModel(
        net,
        lbd=args.lbd,
        z_rescale=args.z_rescale,
        clip_z=args.clip_z,
        clip_range=args.clip_range,
        normalize=args.normalize,
        ratio=bool(args.ratio),
    ).to(device)
    init_ckpt = str(getattr(args, "init_ckpt", "") or "")
    if init_ckpt:
        load_init(model, resolve(init_ckpt))
    held = []
    if bool(getattr(args, "freeze_m1", False)):
        held.append(0)
    raw_heads = str(getattr(args, "freeze_heads", "") or "")
    if raw_heads:
        held.extend(int(part) for part in raw_heads.split(",") if part.strip() != "")
    held = sorted(set(held))
    zero_m1 = freeze_trunk_heads(model, held) if held else None
    opt = make_optimizer(model, args)
    if moment_loss in ("mse", "raw_multi", "hybrid", "raw_bal", "soft_ratio", "offset", "hm"):
        loss_fn = None
    elif moment_k > 0 and moment_loss == "raw_ratio":
        loss_fn = RawRatioBregmanLoss()
    elif moment_k > 0:
        loss_fn = MomentPRLLoss()
    elif args.ratio:
        loss_fn = NormalizedRatioLoss(
            reduction="mean",
            balanced=str(getattr(args, "ratio_loss", "normalized")) == "balanced",
        )
    else:
        loss_fn = BregmanLoss(reduction="mean")

    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        rev = ""
    args.git_rev = rev
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(
        f"train n={len(xtr)} val={'yes' if val_loader else 'no'} "
        f"lbd={args.lbd} scale={args.z_rescale} ratio={args.ratio} "
        f"ratio_loss={getattr(args, 'ratio_loss', 'normalized')} k_max={args.k_max} "
        f"moment_k={moment_k} moment_param={getattr(args, 'moment_param', 'direct')} "
        f"moment_loss={moment_loss} hm_param={getattr(args, 'hm_param', '')} "
        f"raw_weight={getattr(args, 'raw_weight', 'equal')} "
        f"hybrid_weight={getattr(args, 'hybrid_weight', 'equal')} "
        f"higher_target={getattr(args, 'higher_target', 'ratio_kl')} "
        f"higher_scale={float(getattr(args, 'higher_scale', 1.0)):g} "
        f"freeze_m1={bool(getattr(args, 'freeze_m1', False))} "
        f"detach_higher={bool(getattr(args, 'detach_higher', False))} "
        f"trunk_lr_mult={float(getattr(args, 'trunk_lr_mult', 1.0)):g} "
        f"consistency={getattr(args, 'consistency', 'none')} "
        f"lambda_cons={getattr(args, 'lambda_cons', 1.0):g} "
        f"cons_weight={getattr(args, 'cons_weight', 'equal')} "
        f"lr={args.lr} bs={args.batch_size} "
        f"epochs={args.epochs} val_every={args.val_every} "
        f"device={device} out={out_dir}",
        flush=True,
    )

    xs = logp = None
    try:
        xs, logp = support_logp(data_dir.name, device)
        print(f"oracle moments on {data_dir.name} support={tuple(xs.shape)}", flush=True)
    except (KeyError, ValueError) as exc:
        print(f"oracle moments skipped: {exc}", flush=True)

    best = float("inf")
    best_l1 = float("inf")
    best_h = float("inf")
    save_epochs = {
        int(x) for x in str(getattr(args, "save_epochs", "") or "").split(",") if str(x).strip()
    }
    last_val = None
    last_mom = None
    grad_norms = []
    trace_n = int(getattr(args, "trace_trunk_epochs", 0) or 0)
    trace_hm = moment_loss == "hm" and trace_n > 0
    head_trace = (
        {"norms": [], "cos": []}
        if moment_loss in ("raw_multi", "hybrid") or trace_hm
        else None
    )
    history = []
    cons_grad_hist = []
    for epoch in range(1, args.epochs + 1):
        n0 = len(grad_norms)
        h0 = 0 if head_trace is None else len(head_trace["norms"])
        this_trace = head_trace
        if trace_hm and epoch > trace_n:
            this_trace = None
        train_loss, train_data, train_cons, _train_h = run_epoch(
            model, train_loader, loss_fn, args, device, opt=opt,
            grad_norms=grad_norms if moment_k > 0 else None,
            head_trace=this_trace,
            after_backward=zero_m1,
        )
        tick = epoch % args.val_every == 0 or epoch == args.epochs
        if not tick:
            continue
        val_loss = None
        val_l1 = None
        val_h = None
        mom_line = ""
        grad_line = ""
        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_l1, _val_cons, val_h = run_epoch(
                    model, val_loader, loss_fn, args, device, opt=None
                )
                if xs is not None and moment_loss in ("raw_multi", "hybrid"):
                    rows = eval_raw_multi_quantiles(model, val_loader, args, device, xs, logp)
                    last_mom = {"epoch": epoch, "moment_loss": moment_loss, "bins": rows}
                    (out_dir / "mom_val.json").write_text(json.dumps(last_mom, indent=2) + "\n")
                    bits = []
                    key = "mom" if moment_loss == "hybrid" else "ratio"
                    tag = "em" if moment_loss == "hybrid" else "eS"
                    for row in rows:
                        bits.append(f"k{row['k']}{row['bin']}={_fmt_med_p99(row[key])}")
                    mom_line = f"  {tag} med/p99 " + " ".join(bits)
                elif xs is not None and moment_k > 0 and moment_loss == "raw_ratio":
                    rows = eval_raw_ratio_quantiles(model, val_loader, args, device, xs, logp)
                    last_mom = {"epoch": epoch, "moment_k": moment_k, "moment_loss": moment_loss, "bins": rows}
                    (out_dir / "mom_val.json").write_text(json.dumps(last_mom, indent=2) + "\n")
                    ratio_bits, abs_bits, tgt_bits = [], [], []
                    for row in rows:
                        ratio_bits.append(f"{row['bin']}={_fmt_med_p99(row['ratio'])}")
                        abs_bits.append(f"{row['bin']}={_fmt_med_p99(row['abs'])}")
                        tgt = row["target"]
                        p50 = tgt["median"]
                        tgt_bits.append(f"{row['bin']}={p50:.3g}" if p50 is not None else f"{row['bin']}=na")
                    mom_line = (
                        "  ratio med/p99 " + " ".join(ratio_bits)
                        + "  |m| med/p99 " + " ".join(abs_bits)
                        + "  R p50 " + " ".join(tgt_bits)
                    )
                elif xs is not None and moment_k > 0:
                    rows = eval_moment_quantiles(model, val_loader, args, device, xs, logp)
                    last_mom = {"epoch": epoch, "moment_k": moment_k, "bins": rows}
                    (out_dir / "mom_val.json").write_text(json.dumps(last_mom, indent=2) + "\n")
                    bits = []
                    for row in rows:
                        lg = row["log"]
                        med = lg["median"]
                        p99 = lg["p99"]
                        bits.append(
                            f"{row['bin']}={med:.3g}/{p99:.3g}" if med is not None else f"{row['bin']}=na"
                        )
                    mom_line = "  e med/p99 " + " ".join(bits)
                elif xs is not None:
                    means, counts = eval_moment_bins(model, val_loader, args, device, xs, logp)
                    last_mom = mom_payload(means, counts)
                    last_mom["epoch"] = epoch
                    (out_dir / "mom_val.json").write_text(json.dumps(last_mom, indent=2) + "\n")
                    mom_line = "  " + format_mom_line(means)
            last_val = val_loss
            if head_trace is not None and len(head_trace["norms"]) > h0:
                norms = np.asarray(head_trace["norms"][h0:], dtype=np.float64)
                cos = np.asarray(head_trace["cos"][h0:], dtype=np.float64)
                bits = []
                for k, st in enumerate(head_norm_rows(norms)):
                    bits.append(f"k{k + 1}={st['median']:.3g}/{st['p99']:.3g}")
                cos_bits = []
                for row in head_cos_rows(cos):
                    med = row["median"]
                    cos_bits.append(f"{row['pair']}={med:.3g}" if med is not None else f"{row['pair']}=na")
                grad_line = "  trunk p50/p99 " + " ".join(bits) + "  cos " + " ".join(cos_bits)
            elif moment_k > 0 and len(grad_norms) > n0:
                window = grad_summary(grad_norms[n0:])
                (out_dir / "grad_val.json").write_text(
                    json.dumps({"epoch": epoch, "window": window, "all": grad_summary(grad_norms)}, indent=2) + "\n"
                )
                grad_line = (
                    f"  grad p50={window['median']:.3g} p99={window['p99']:.3g} max={window['max']:.3g}"
                )
            r_line = ""
            if str(getattr(args, "consistency", "none") or "none") in ("soft", "anchored"):
                grows = eval_cons_grad(model, val_loader, args, device)
                if grows:
                    cons_grad_hist.append({"epoch": epoch, "bins": grows})
                    (out_dir / "cons_grad.json").write_text(
                        json.dumps(cons_grad_hist, indent=2) + "\n"
                    )
                    r_line = "  r med/p99 " + " ".join(
                        _fmt_r(row) for row in grows if row["bin"] != "all"
                    )
            loss_bits = f" data {train_data:.4f}"
            if train_cons is not None:
                loss_bits += f" cons {train_cons:.4f}"
            val_l1_bit = ""
            if val_l1 is not None and (moment_loss == "hm" or bool(getattr(args, "save_best_l1", False))):
                val_l1_bit = f" valL1 {val_l1:.4f}"
            if val_h is not None:
                val_l1_bit += f" valH {val_h:.4f}"
            print(
                f"epoch {epoch} train {train_loss:.4f}{loss_bits} val {val_loss:.4f}{val_l1_bit}{mom_line}{grad_line}{r_line}",
                flush=True,
            )
            if bool(getattr(args, "save_best_l1", False)) or (
                moment_loss == "hm" and trace_hm
            ):
                history.append(
                    {
                        "epoch": epoch,
                        "train": train_loss,
                        "train_l1": train_data,
                        "val": val_loss,
                        "val_l1": val_l1,
                        "val_h": val_h,
                    }
                )
                (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        else:
            print(f"epoch {epoch} train {train_loss:.4f}", flush=True)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss if val_loss is not None else last_val,
            "val_l1": val_l1,
            "best_val": best,
            "mom_val": last_mom,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "latest.pt")
        metric = val_loss if val_loss is not None else train_loss
        if metric < best:
            best = metric
            ckpt["best_val"] = best
            torch.save(ckpt, out_dir / "best.pt")
            if last_mom is not None:
                (out_dir / "mom_best.json").write_text(json.dumps(last_mom, indent=2) + "\n")
        if (
            bool(getattr(args, "save_best_l1", False))
            and val_l1 is not None
            and val_l1 < best_l1
        ):
            best_l1 = val_l1
            l1_ckpt = dict(ckpt)
            l1_ckpt["best_l1"] = best_l1
            torch.save(l1_ckpt, out_dir / "best_l1.pt")
        if (
            bool(getattr(args, "save_best_h", False))
            and val_h is not None
            and val_h < best_h
        ):
            best_h = val_h
            h_ckpt = dict(ckpt)
            h_ckpt["best_h"] = best_h
            torch.save(h_ckpt, out_dir / "best_h.pt")
        if epoch in save_epochs:
            torch.save(ckpt, out_dir / f"epoch_{epoch}.pt")
    if moment_k > 0 and grad_norms:
        np.save(out_dir / "grad_norms.npy", np.asarray(grad_norms, dtype=np.float64))
        (out_dir / "grad_summary.json").write_text(
            json.dumps(grad_summary(grad_norms), indent=2) + "\n"
        )
    if head_trace is not None and head_trace["norms"]:
        norms = np.asarray(head_trace["norms"], dtype=np.float64)
        cos = np.asarray(head_trace["cos"], dtype=np.float64)
        np.save(out_dir / "grad_head_norms.npy", norms)
        np.save(out_dir / "grad_cosine.npy", cos)
        head_rows = head_norm_rows(norms)
        cos_rows = head_cos_rows(cos)
        (out_dir / "grad_heads.json").write_text(json.dumps(head_rows, indent=2) + "\n")
        (out_dir / "grad_cosine.json").write_text(json.dumps(cos_rows, indent=2) + "\n")
        _write_table(out_dir / "grads.txt", head_rows)
        _write_table(out_dir / "grad_cosine.txt", cos_rows)
    (out_dir / "finished.json").write_text(json.dumps({"epochs": int(args.epochs)}) + "\n")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
