import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from loss.consistency import assemble_log_s
from loss.loss import full_bregman, higher_moment_maps, linked_mean, moments_from_ratio_pred
from loss.oracle import gamma_bin_index, oracle_moments
from model.small_diffusion import PoissonDiscreteDiffusionModel
from train import ROOT, load_yaml, pair_or_none, resolve, str2bool

MLP = importlib.import_module("model.1d_mlp").MLP
KERNELS = ("poisson", "nb", "repoisson", "twopois")


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", default=str(ROOT / "config.yml"))
    pre.add_argument("--ckpt", default=None, help="best.pt / latest.pt")
    pre_args, _ = pre.parse_known_args()
    cfg = load_yaml(pre_args.config)
    ckpt_args = {}
    if pre_args.ckpt:
        blob = torch.load(pre_args.ckpt, map_location="cpu", weights_only=False)
        ckpt_args = blob.get("args") or {}

    def pick(key, default):
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        return ckpt_args.get(key, default)

    p = argparse.ArgumentParser(
        parents=[pre],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", default=None, help="defaults to ckpt folder")
    p.add_argument("--device", default=pick("device", "cuda"))
    p.add_argument("--lbd", type=float, default=float(pick("lbd", 100.0)))
    p.add_argument("--snr_min", type=float, default=float(cfg.get("snr_min", 0.0)))
    p.add_argument("--snr_max", type=float, default=float(cfg.get("snr_max", pick("lbd", 100.0))))
    p.add_argument("--sample_steps", type=int, default=int(cfg.get("sample_steps", 100)))
    p.add_argument("--n_sample", type=int, default=int(cfg.get("n_sample", 50000)))
    p.add_argument("--sample_batch", type=int, default=int(cfg.get("sample_batch", 4096)))
    p.add_argument("--t_eps", type=float, default=float(pick("t_eps", 1e-4)))
    p.add_argument(
        "--kernels",
        default=cfg.get("kernels", "poisson,nb,twopois"),
        help="comma list: poisson,nb,twopois",
    )
    p.add_argument("--hidden", type=int, default=int(pick("hidden", 128)))
    p.add_argument("--layers", type=int, default=int(pick("layers", 3)))
    p.add_argument("--continuous_t", type=str2bool, default=pick("continuous_t", True))
    p.add_argument(
        "--scale",
        "--z_rescale",
        dest="z_rescale",
        type=str2bool,
        default=bool(ckpt_args.get("z_rescale", ckpt_args.get("scale", cfg.get("scale", False)))),
        help="MLP sees z/(λ α_t)=z/γ; match the ckpt",
    )
    p.add_argument("--clip_z", type=str2bool, default=pick("clip_z", True))
    p.add_argument("--clip_range", nargs=2, type=float, default=pair_or_none(pick("clip_range", None)))
    p.add_argument("--normalize", nargs=2, type=float, default=pair_or_none(pick("normalize", None)))
    p.add_argument("--ratio", type=str2bool, default=bool(ckpt_args.get("ratio", cfg.get("ratio", False))))
    p.add_argument("--k_max", type=int, default=int(ckpt_args.get("k_max", cfg.get("k_max", 3))))
    p.add_argument(
        "--ratio_loss",
        default=str(ckpt_args.get("ratio_loss", cfg.get("ratio_loss", "normalized"))),
        choices=("normalized", "balanced", "raw"),
    )
    p.add_argument("--seed", type=int, default=int(pick("seed", 0)))
    args = p.parse_args()
    if args.ckpt is None:
        p.error("--ckpt is required")
    if args.out_dir is None:
        args.out_dir = str(Path(args.ckpt).resolve().parent)
    if str(getattr(args, "ratio_loss", "normalized")) in ("balanced", "raw"):
        args.ratio = True
    args.prior_moments = ckpt_args.get("prior_moments")
    args.moment_param = str(ckpt_args.get("moment_param", "direct"))
    args.moment_loss = str(ckpt_args.get("moment_loss", "moment"))
    args.hm_param = str(ckpt_args.get("hm_param", "") or "")
    args.hybrid_weight = str(ckpt_args.get("hybrid_weight", "equal") or "equal")
    args.raw_weight = ckpt_args.get("raw_weight", "equal")
    args.consistency = str(ckpt_args.get("consistency", "none") or "none")
    args.lambda_cons = float(ckpt_args.get("lambda_cons", 1.0) or 1.0)
    return args


def build_model(args, device):
    if bool(getattr(args, "ratio", False)):
        RatioMLP = importlib.import_module("model.ratio_mlp").RatioMLP
        net = RatioMLP(
            in_dim=1,
            hidden=args.hidden,
            out_dim=int(getattr(args, "k_max", 3)),
            layers=args.layers,
            continuous_t=args.continuous_t,
        )
    else:
        net = MLP(
            in_dim=1,
            hidden=args.hidden,
            out_dim=1,
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
        ratio=bool(getattr(args, "ratio", False)),
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def time_of(gamma, lbd, t_eps, n, device):
    t = (float(gamma) / float(lbd)) if not torch.is_tensor(gamma) else (gamma / lbd)
    t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
    t = t.clamp_min(t_eps)
    if t.numel() == 1:
        t = t.expand(n)
    return t


def make_gammas(schedule, steps, g0, g1, device, power=None):
    steps = int(steps)
    if power is None:
        power = 1.0 if schedule == "uniform" else 2.0
    i = torch.arange(steps + 1, device=device, dtype=torch.float32)
    g = float(g0) + (float(g1) - float(g0)) * (i / float(steps)).pow(float(power))
    g[0] = float(g0)
    g[-1] = float(g1)
    return g


def predict_log_B(model, z, gamma, args):
    t = time_of(gamma, args.lbd, args.t_eps, z.shape[0], z.device)
    alpha = t if args.z_rescale else None
    return model(z, t, alpha=alpha)


def shifted_logs(model, z, gamma, args):
    """Network output at z, z+1, z+2. Same gamma, so same t."""
    return (
        predict_log_B(model, z, gamma, args),
        predict_log_B(model, z + 1.0, gamma, args),
        predict_log_B(model, z + 2.0, gamma, args),
    )


def log_s_eval(model, z, gamma, args):
    """Log S used for moments, residual of that S, and residual before projection."""
    mode = str(getattr(args, "consistency", "none") or "none")
    a, a1, a2 = shifted_logs(model, z, gamma, args)
    return assemble_log_s(a, a1, a2, mode)


def log_s_for_moments(model, z, gamma, args):
    mode = str(getattr(args, "consistency", "none") or "none")
    if mode not in ("hard", "project"):
        return predict_log_B(model, z, gamma, args)
    a, a1, a2 = shifted_logs(model, z, gamma, args)
    log_s, _, _, _, _ = assemble_log_s(a, a1, a2, mode)
    return log_s


def predict_x(model, z, gamma, args):
    kind = str(getattr(args, "moment_loss", ""))
    if kind == "hm":
        return posterior_moments(model, z, gamma, args, order=1)[0]
    if kind in ("soft_ratio", "offset"):
        pred = predict_log_B(model, z, gamma, args)
        gmin = float(getattr(args, "t_eps", 1e-4)) * float(getattr(args, "lbd", 100.0))
        return linked_mean(pred, z, gamma, kind, gmin=gmin).float().view(-1, 1)
    if bool(getattr(args, "ratio", False)):
        log_pred = predict_log_B(model, z, gamma, args)
        return moments_from_ratio_pred(log_pred, z, gamma, args)[0]
    t = time_of(gamma, args.lbd, args.t_eps, z.shape[0], z.device)
    alpha = t if args.z_rescale else None
    return model(z, t, alpha=alpha)


def posterior_moments(model, z, gamma, args, order=3):
    kind = str(getattr(args, "moment_loss", ""))
    if kind == "hm":
        pred = predict_log_B(model, z, gamma, args)
        gmin = float(getattr(args, "t_eps", 1e-4)) * float(getattr(args, "lbd", 100.0))
        m, _s = higher_moment_maps(
            pred,
            z,
            gamma,
            str(getattr(args, "hm_param", "raw_log")),
            gmin=gmin,
            prior=getattr(args, "prior_moments", None),
        )
        return [m[:, k].float().view(-1, 1) for k in range(int(order))]
    if kind in ("soft_ratio", "offset"):
        m1 = predict_x(model, z, gamma, args)
        return [m1 for _ in range(int(order))]
    if bool(getattr(args, "ratio", False)):
        log_pred = log_s_for_moments(model, z, gamma, args)
        moms = moments_from_ratio_pred(log_pred, z, gamma, args)
        return moms[: int(order)]
    acc = None
    moms = []
    for i in range(order):
        mi = predict_x(model, z + float(i), gamma, args).clamp_min(0.0)
        acc = mi if acc is None else acc * mi
        moms.append(acc)
    return moms


def step_from_moments(z, h, kernel, moms, stats=None):
    m1 = moms[0]
    if kernel == "poisson":
        return z + torch.poisson((h * m1).clamp_min(0.0))
    if kernel == "nb":
        v = (moms[1] - m1 * m1).clamp_min(0.0)
        return z + nb_gap(m1, v, h)
    if kernel in ("twopois", "2pois", "two_poisson"):
        return z + two_pois_gap(moms[0], moms[1], moms[2], h, stats=stats)
    raise ValueError(kernel)


def two_pois_gap(m1, m2, m3, h, stats=None):
    """3-moment two-point mix: K ~ w Pois(h x1) + (1-w) Pois(h x2).

    Fallback to Pois(h m1) when m1<=0, v is below tolerance, or x1<0.
    w is clamped to [1e-4, 1-1e-4] before the atoms are built.
    stats counts rows. extreme_w uses the unclamped weight and does not change K.
    """
    v = (m2 - m1 * m1).clamp_min(0.0)
    c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1.pow(3)
    m_safe = m1.clamp_min(1e-8)
    thin = (m1 <= 0) | (v <= 1e-8 * m_safe)
    sigma = v.sqrt()
    g = c3 / sigma.pow(3).clamp_min(1e-12)
    w_raw = 0.5 * (1.0 + g / (g.square() + 4.0).sqrt())
    w = w_raw.clamp(1e-4, 1.0 - 1e-4)
    x1 = m1 - sigma * ((1.0 - w) / w).sqrt()
    x2 = m1 + sigma * (w / (1.0 - w)).sqrt()
    if stats is not None:
        n = int(m1.numel())
        stats["n"] = int(stats.get("n", 0)) + n
        stats["invalid_v"] = int(stats.get("invalid_v", 0)) + int(thin.sum().item())
        bad_atom = (~thin) & (
            (x1 < 0) | (x2 < 0) | ~torch.isfinite(x1) | ~torch.isfinite(x2) | ~torch.isfinite(w_raw)
        )
        stats["invalid_atom"] = int(stats.get("invalid_atom", 0)) + int(bad_atom.sum().item())
        extreme = (~thin) & torch.isfinite(w_raw) & ((w_raw < 0.01) | (w_raw > 0.99))
        stats["extreme_w"] = int(stats.get("extreme_w", 0)) + int(extreme.sum().item())
        fallback = thin | (x1 < 0)
        stats["fallback"] = int(stats.get("fallback", 0)) + int(fallback.sum().item())
    k_p = torch.poisson((h * m1).clamp_min(0.0))
    pick = torch.rand_like(m1)
    k_mix = torch.where(
        pick < w,
        torch.poisson((h * x1).clamp_min(0.0)),
        torch.poisson((h * x2).clamp_min(0.0)),
    )
    k_mix = torch.where(x1 < 0, k_p, k_mix)
    return torch.where(thin, k_p, k_mix)


def two_pois_moment_flags(m1, m2, m3):
    """Same masks as two_pois_gap, without drawing K."""
    m1 = m1.reshape(-1)
    m2 = m2.reshape(-1)
    m3 = m3.reshape(-1)
    v_raw = m2 - m1 * m1
    v = v_raw.clamp_min(0.0)
    c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1.pow(3)
    m_safe = m1.clamp_min(1e-8)
    thin = (m1 <= 0) | (v <= 1e-8 * m_safe)
    sigma = v.sqrt()
    g = c3 / sigma.pow(3).clamp_min(1e-12)
    w_raw = 0.5 * (1.0 + g / (g.square() + 4.0).sqrt())
    w = w_raw.clamp(1e-4, 1.0 - 1e-4)
    x1 = m1 - sigma * ((1.0 - w) / w).sqrt()
    x2 = m1 + sigma * (w / (1.0 - w)).sqrt()
    bad_atom = (~thin) & (
        (x1 < 0) | (x2 < 0) | ~torch.isfinite(x1) | ~torch.isfinite(x2) | ~torch.isfinite(w_raw)
    )
    extreme = (~thin) & torch.isfinite(w_raw) & ((w_raw < 0.01) | (w_raw > 0.99))
    fallback = thin | (x1 < 0)
    return {
        "v_le_0": v_raw <= 0,
        "thin": thin,
        "bad_atom": bad_atom,
        "extreme": extreme,
        "fallback": fallback,
    }


def nb_gap(m, v, h):
    mean_k = (h * m).clamp_min(0.0)
    pois = torch.poisson(mean_k)
    m_safe = m.clamp_min(1e-8)
    v_safe = v.clamp_min(0.0)
    thin = (m <= 0) | (v_safe <= 1e-8 * m_safe)
    r = (m_safe * m_safe / v_safe.clamp_min(1e-12)).clamp(1e-4, 1e6)
    scale = (v_safe / m_safe).clamp_min(1e-12)
    xg = torch.distributions.Gamma(r, 1.0 / scale).sample()
    nb = torch.poisson((h * xg).clamp_min(0.0))
    return torch.where(thin, pois, nb)


@torch.no_grad()
def reverse_step(model, z, gamma, h, kernel, args, oracle=None, stats=None):
    if oracle is not None:
        xs, logp = oracle
        k_need = 1 if kernel == "poisson" else (2 if kernel == "nb" else 3)
        moms = oracle_moments(z, gamma, xs, logp, k_max=max(int(k_need), 3))
        return step_from_moments(z, h, kernel, moms, stats=stats)
    if bool(getattr(args, "ratio", False)):
        need = 1 if kernel == "poisson" else (2 if kernel == "nb" else 3)
        moms = posterior_moments(model, z, gamma, args, order=need)
        return step_from_moments(z, h, kernel, moms, stats=stats)
    m = predict_x(model, z, gamma, args)
    if kernel == "poisson":
        return z + torch.poisson((h * m).clamp_min(0.0))
    if kernel == "repoisson":
        m = m.clamp_min(0.0).round()
        return torch.poisson(((gamma + h) * m).clamp_min(0.0))
    if kernel == "nb":
        m1 = predict_x(model, z + 1.0, gamma, args)
        v = (m * m1 - m * m).clamp_min(0.0)
        return z + nb_gap(m, v, h)
    if kernel in ("twopois", "2pois", "two_poisson"):
        m1, m2, m3 = posterior_moments(model, z, gamma, args, order=3)
        return z + two_pois_gap(m1, m2, m3, h, stats=stats)
    raise ValueError(kernel)


def _on_policy_chunk(moms, z, gamma, h, xs, logp):
    """Errors at one reverse state. eta_k = h^k m_k / k!."""
    g = torch.full((z.shape[0],), float(gamma), device=z.device, dtype=torch.float32)
    stars = oracle_moments(z, g, xs, logp, k_max=3)
    mh = [m.reshape(-1).double() for m in moms]
    ms = [s.reshape(-1).double() for s in stars]
    logs = []
    for k in range(3):
        logs.append(
            (mh[k].clamp_min(1e-12).log() - ms[k].clamp_min(1e-12).log()).detach().cpu()
        )
    habs = (float(h) * (mh[0] - ms[0]).abs()).detach().cpu()
    kl = (float(h) * full_bregman(ms[0], mh[0])).detach().cpu()
    eta2 = ((float(h) ** 2) * (mh[1] - ms[1]).abs() / 2.0).detach().cpu()
    eta3 = ((float(h) ** 3) * (mh[2] - ms[2]).abs() / 6.0).detach().cpu()
    return {
        "idx": gamma_bin_index(g).detach().cpu(),
        "log": logs,
        "habs": habs,
        "kl": kl,
        "eta2": eta2,
        "eta3": eta3,
    }


@torch.no_grad()
def generate(model, n, kernel, args, device, gammas=None, oracle=None, counters=None, on_policy=None):
    if gammas is None:
        g0, g1 = float(args.snr_min), float(args.snr_max)
        steps = int(args.sample_steps)
        gammas = make_gammas("uniform", steps, g0, g1, device)
    out = []
    left = int(n)
    hs = gammas[1:] - gammas[:-1]
    g_last = float(gammas[-1].item())
    while left > 0:
        b = min(int(args.sample_batch), left)
        z = torch.zeros(b, 1, device=device)
        bad = torch.zeros(b, dtype=torch.bool, device=device) if counters is not None else None
        record = (
            on_policy is not None
            and oracle is None
            and model is not None
            and int(on_policy.get("seen", 0)) < int(on_policy["n"])
        )
        for i in range(hs.numel()):
            g = float(gammas[i].item())
            h = float(hs[i].item())
            twostats = counters if kernel in ("twopois", "2pois", "two_poisson") else None
            if record:
                g_t = torch.full((b, 1), g, device=device)
                moms = posterior_moments(model, z, g_t, args, order=3)
                z_before = z
                z = step_from_moments(z, h, kernel, moms, stats=twostats)
                on_policy["chunks"].append(
                    _on_policy_chunk(moms, z_before, g, h, on_policy["xs"], on_policy["logp"])
                )
            else:
                z = reverse_step(
                    model, z, g, h, kernel, args, oracle=oracle, stats=twostats,
                )
            if bad is not None:
                z1 = z.reshape(-1)
                bad = bad | (~torch.isfinite(z1)) | (z1 > 1e6)
        if record:
            on_policy["seen"] = int(on_policy.get("seen", 0)) + int(b)
        if counters is not None:
            counters["traj"] = int(counters.get("traj", 0)) + int(b)
            counters["runaway_traj"] = int(counters.get("runaway_traj", 0)) + int(bad.sum().item())
        x = (z / max(g_last, 1e-12)).clamp_min(0.0)
        if args.normalize is not None and model is not None:
            x = model.decode_x(x)
        out.append(x.cpu())
        left -= b
    return torch.cat(out, 0).numpy().reshape(-1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args, device)
    kernels = [k.strip() for k in str(args.kernels).split(",") if k.strip()]
    for k in kernels:
        if k not in KERNELS and k not in ("2pois", "two_poisson"):
            raise ValueError(f"unknown kernel {k}")
        key = "twopois" if k in ("twopois", "2pois", "two_poisson") else k
        print(
            f"sample {key} n={args.n_sample} snr={args.snr_min:g}->{args.snr_max:g} "
            f"steps={args.sample_steps} z_rescale={args.z_rescale} ckpt={args.ckpt}",
            flush=True,
        )
        x = generate(model, args.n_sample, key, args, device)
        path = out_dir / f"samples_{key}.npy"
        np.save(path, x)
        print(f"  saved {path} E[z/γ]={x.mean():.4g}", flush=True)
    with open(out_dir / "sample_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    from test import plot_experiment

    plot_keys = tuple(
        k for k in ("poisson", "nb", "twopois") if (out_dir / f"samples_{k}.npy").is_file()
    )
    plot_experiment(out_dir, kernels=plot_keys)


if __name__ == "__main__":
    main()
