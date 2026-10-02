"""Post-quadratic diagnostics. Frozen scale ckpts. No retrain.

Layout
------
test_exp/script/run_0*.sh          launchers (you run these)
test_exp/results/
  01_traj_marginal/                z_γ vs true Pois(γ X)
  02_exponent/p{p}/                γ_t = 100 (t/100)^p
  03_R/                            R_t = h v / m along trajectories
  04_oracle_mean/{uniform,quad}_*  learned vs analytic posterior mean
  05_cutoff/cut_*                  exact mixed-Poisson until γ_cut
  06_first_step/                   m_θ(0,γ) and exact first-gap PMF
  07_twopois/                      two-Poisson mixture stability
  08_sweep/T*/                     uniform T=100,200,500,1000
  09_mean_cut/                     oracle mean for γ<γ_cut, then learned mean + Poisson
  10_oracle_twopois/               Two-Poisson from oracle (m1,m2,m3) vs +k net moments
  11_trap/                         θ-trapezoidal mean-Poisson (Heun, θ=1/2) on quadratic
  12_oracle_nb/                    NB from oracle (m1,m2) vs learned shifted moments
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sample import nb_gap, posterior_moments, predict_x, reverse_step, two_pois_gap  # noqa: E402
from sample_tests import (  # noqa: E402
    DEFAULT_KERNELS,
    DEFAULT_NAMES,
    ckpt_of,
    has_all,
    load_model,
    load_x_pool,
    make_gammas,
    run_one,
    split_csv,
)
from test import emp_pmf_window, l1_window, plot_experiment, wd1  # noqa: E402
from test import true_pmf as discrete_pmf  # noqa: E402

RESULTS = HERE / "results"
TRAJ_GAMMAS = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
TRAJ_SHOW = (0.1, 1.0, 10.0, 100.0)
EXPONENTS = (1.0, 1.5, 2.0, 3.0, 4.0)
CUTOFFS = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
SWEEP = ((100, "T100_h1"), (200, "T200_h05"), (500, "T500_h02"), (1000, "T1000_h01"))
COLORS = {"poisson": "#ff7f0e", "nb": "#2ca02c", "twopois": "#d62728"}
LEGACY_SWEEP = {
    "T100_h1": HERE / "sweep" / "T100_h1",
    "T200_h05": HERE / "sweep" / "T200_h05",
    "T500_h02": HERE / "sweep" / "T500_h02",
    "T1000_h01": HERE / "sweep" / "T1000_h01",
}
LEGACY_QUAD = HERE / "quad" / "T100_quad"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--exp",
        required=True,
        choices=(
            "traj", "exponent", "R",             "oracle_mean", "cutoff", "first_step", "twopois", "sweep",
            "mean_cut", "oracle_twopois", "trap", "oracle_nb",
        ),
    )
    p.add_argument("--ckpt_root", default=str(ROOT / "experiments" / "scale"))
    p.add_argument("--names", default=DEFAULT_NAMES)
    p.add_argument("--kernels", default=DEFAULT_KERNELS)
    p.add_argument("--n_sample", type=int, default=50000)
    p.add_argument("--n_diag", type=int, default=8192)
    p.add_argument("--sample_batch", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def gtag(g):
    return f"{float(g):g}".replace(".", "p")


def fmt_cell(k, v):
    if v is None or v == "":
        return f"{'':>10}"
    if k in ("nfe", "steps"):
        return f"{int(v):10d}"
    if k in ("wd", "tv") or isinstance(v, (float, np.floating, int, np.integer)):
        return f"{float(v):10.4f}"
    return f"{str(v):12}"


def write_table(rows, path, keys):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = " ".join(
        f"{k:>10}" if k in ("wd", "tv", "gamma", "g_act", "power", "cut", "nfe", "steps") else f"{k:12}"
        for k in keys
    )
    lines = [header]
    for r in rows:
        lines.append(" ".join(fmt_cell(k, r.get(k, "")) for k in keys))
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text, end="", flush=True)


def tv_counts(a, b):
    a = np.round(np.asarray(a).reshape(-1)).astype(np.int64)
    b = np.round(np.asarray(b).reshape(-1)).astype(np.int64)
    hi = int(max(np.quantile(a, 0.999), np.quantile(b, 0.999), 1))
    pa, _ = emp_pmf_window(a, 0, hi)
    pb, _ = emp_pmf_window(b, 0, hi)
    return l1_window(pa, pb)


def setup_device(cli):
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    return torch.device(cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu")


def load_pair(cli, name, device):
    ckpt_root = Path(cli.ckpt_root)
    if not ckpt_root.is_absolute():
        ckpt_root = ROOT / ckpt_root
    ckpt = ckpt_of(ckpt_root, name)
    if not ckpt.is_file():
        return None, None, None, None
    model, args = load_model(ckpt, device)
    args.sample_batch = cli.sample_batch
    x_val, x_mean = load_x_pool(name, device, split="val")
    return model, args, x_val, x_mean


def reuse_samples(src, dest, kernels):
    src, dest = Path(src), Path(dest)
    if not src.is_dir():
        return False
    dest.mkdir(parents=True, exist_ok=True)
    copied = False
    for k in kernels:
        s, d = src / f"samples_{k}.npy", dest / f"samples_{k}.npy"
        if s.is_file() and not d.is_file():
            shutil.copy2(s, d)
            copied = True
    sm, dm = src / "meta.json", dest / "meta.json"
    if sm.is_file() and not dm.is_file():
        shutil.copy2(sm, dm)
    return copied


def support_logp(name, device):
    if name == "gamma_ltj":
        xs = np.linspace(1e-4, float(stats.gamma.ppf(0.9999, a=1.0, scale=10.0)), 1024)
        logp = stats.gamma.logpdf(xs, a=1.0, scale=10.0)
        logp = np.where(np.isfinite(logp), logp, -1e20)
        return (
            torch.tensor(xs, device=device, dtype=torch.float32),
            torch.tensor(logp, device=device, dtype=torch.float32),
        )
    bounds = {
        "pois20": (0, 50),
        "nb": (0, 120),
        "poismix_mod": (0, 90),
        "poissmix": (0, 160),
        "poissmix3": (0, 150),
        "zip": (0, 25),
        "yule_simon": (1, 2000),
    }
    lo, hi = bounds[name]
    ks = np.arange(lo, hi + 1)
    p = np.clip(discrete_pmf(name, ks), 0.0, None)
    p = p / max(float(p.sum()), 1e-12)
    logp = np.log(np.clip(p, 1e-300, 1.0))
    return (
        torch.tensor(ks, device=device, dtype=torch.float32),
        torch.tensor(logp, device=device, dtype=torch.float32),
    )


def posterior_weights(z, gamma, xs, logp):
    z = z.reshape(-1, 1).float()
    g = torch.as_tensor(gamma, device=z.device, dtype=torch.float32).reshape(-1)
    if g.numel() == 1:
        g = g.expand(z.shape[0])
    g = g.view(-1, 1).clamp_min(1e-12)
    rate = (g * xs.view(1, -1).clamp_min(1e-12)).clamp_min(1e-12)
    log_lik = z * rate.log() - rate - torch.lgamma(z + 1.0)
    return torch.softmax(log_lik + logp.view(1, -1), dim=-1)


def oracle_mean(z, gamma, xs, logp):
    w = posterior_weights(z, gamma, xs, logp)
    return (w * xs.view(1, -1)).sum(-1, keepdim=True)


def oracle_sample_x(z, gamma, xs, logp):
    w = posterior_weights(z, gamma, xs, logp)
    idx = torch.multinomial(w.clamp_min(1e-12), 1)
    return xs[idx.view(-1)].view(-1, 1)


def oracle_moments(z, gamma, xs, logp):
    """True raw moments E[X^k | Z_γ=z] on the data support."""
    w = posterior_weights(z, gamma, xs, logp)
    x = xs.view(1, -1)
    m1 = (w * x).sum(-1, keepdim=True)
    m2 = (w * x.square()).sum(-1, keepdim=True)
    m3 = (w * x.pow(3)).sum(-1, keepdim=True)
    return m1, m2, m3


def sample_loop(n, batch, device, gammas, args, model, step_fn):
    out = []
    left = int(n)
    hs = gammas[1:] - gammas[:-1]
    gT = float(gammas[-1].item())
    while left > 0:
        b = min(int(batch), left)
        z = torch.zeros(b, 1, device=device)
        for i in range(hs.numel()):
            z = step_fn(z, float(gammas[i].item()), float(hs[i].item()), float(gammas[i + 1].item()))
        out.append(decode_z(z, gT, args, model))
        left -= b
    return np.concatenate(out, 0)


def trap_poisson_step(model, z, g, h, g_next, args, theta=0.5):
    """Explicit θ-trapezoidal mean-Poisson (Heun). θ=1/2 is trapezoidal.

    m0 = m(z, γ)
    ẑ  = z + h m0
    m1 = m(ẑ, γ+h)
    K  ~ Pois( h ((1-θ) m0 + θ m1) )
    """
    m0 = predict_x(model, z, g, args).clamp_min(0.0)
    z_pred = z + float(h) * m0
    m1 = predict_x(model, z_pred, g_next, args).clamp_min(0.0)
    lam = float(h) * ((1.0 - float(theta)) * m0 + float(theta) * m1)
    return z + torch.poisson(lam.clamp_min(0.0))


def decode_z(z, gamma, args, model):
    x = (z / max(float(gamma), 1e-12)).clamp_min(0.0)
    if args.normalize is not None:
        x = model.decode_x(x)
    return x.detach().cpu().numpy().reshape(-1)


def nearest_hits(gammas, targets):
    g = gammas.detach().cpu().numpy().astype(np.float64)
    hits = []
    used = set()
    for t in targets:
        i = int(np.argmin(np.abs(g - float(t))))
        if i == 0:
            i = 1
        if i in used:
            for j in range(1, len(g)):
                if j not in used:
                    i = j
                    break
        used.add(i)
        hits.append((float(t), i, float(g[i])))
    return hits


def two_pois_stats(m1, m2, m3, h):
    v = (m2 - m1 * m1).clamp_min(0.0)
    c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1.pow(3)
    m_safe = m1.clamp_min(1e-8)
    thin = (m1 <= 0) | (v <= 1e-8 * m_safe)
    sigma = v.sqrt()
    g = c3 / sigma.pow(3).clamp_min(1e-12)
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    w = 0.5 * (1.0 + g / (g.square() + 4.0).sqrt())
    w = w.clamp(1e-4, 1.0 - 1e-4)
    x1 = m1 - sigma * ((1.0 - w) / w).sqrt()
    x2 = m1 + sigma * (w / (1.0 - w)).sqrt()
    x1 = torch.nan_to_num(x1, nan=0.0)
    x2 = torch.nan_to_num(x2, nan=0.0)
    hx1 = h * x1
    hx2 = h * x2
    fallback = thin | (x1 < 0)
    return {
        "v": v,
        "c3": c3,
        "g": g,
        "w": w,
        "x1": x1,
        "x2": x2,
        "hx1": hx1,
        "hx2": hx2,
        "x1_neg": x1 < 0,
        "w_extreme": (w < 0.01) | (w > 0.99),
        "fallback": fallback,
        "hx2_huge": hx2 > 1e4,
    }


def diag_batch(cli):
    return min(int(cli.sample_batch), 1024)


def release(*tensors):
    for x in tensors:
        del x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def wrap_axes(n, w=4.2, h=3.2, ncols=4):
    ncols = min(ncols, max(n, 1))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    return fig, axes


def hide_extra(axes, n):
    for ax in axes.ravel()[n:]:
        ax.axis("off")


# ---- 1 traj ----
def exp_traj(cli, device, names, kernels):
    root = RESULTS / "01_traj_marginal"
    rows = []
    hit_info = None
    for name in names:
        model, args, x_val, _ = load_pair(cli, name, device)
        if model is None:
            print(f"skip {name}: no ckpt", flush=True)
            continue
        g1 = float(args.lbd)
        gammas = make_gammas("quad", 100, 0.0, g1, device, power=2.0)
        hits = nearest_hits(gammas, TRAJ_GAMMAS)
        if hit_info is None:
            hit_info = [{"gamma": a, "index": int(i), "gamma_act": b} for a, i, b in hits]
            (root / "actual_gamma.json").parent.mkdir(parents=True, exist_ok=True)
            (root / "actual_gamma.json").write_text(json.dumps(hit_info, indent=2) + "\n")
        true_dir = root / name / "true"
        true_dir.mkdir(parents=True, exist_ok=True)
        n = min(int(cli.n_sample), int(x_val.numel()))
        x = x_val[:n].view(-1, 1)
        for g_req, _, g_act in hits:
            p = true_dir / f"z_g{gtag(g_req)}.npy"
            if p.is_file():
                continue
            z = torch.poisson((float(g_act) * x.clamp_min(0.0)))
            np.save(p, z.cpu().numpy().reshape(-1))
        idx_of = {g_req: idx for g_req, idx, _ in hits}
        for kernel in kernels:
            kdir = root / name / kernel
            kdir.mkdir(parents=True, exist_ok=True)
            need = [g for g, _, _ in hits if not (kdir / f"z_g{gtag(g)}.npy").is_file()]
            if not need:
                print(f"  skip traj {name} {kernel}", flush=True)
            else:
                print(f"  traj {name} {kernel}", flush=True)
                want = set(idx_of[g] for g in need)
                snaps = {idx: [] for idx in want}
                left = n
                while left > 0:
                    b = min(int(cli.sample_batch), left)
                    z = torch.zeros(b, 1, device=device)
                    hs = gammas[1:] - gammas[:-1]
                    for i in range(hs.numel()):
                        g = float(gammas[i].item())
                        h = float(hs[i].item())
                        z = reverse_step(model, z, g, h, kernel, args)
                        after = i + 1
                        if after in want:
                            snaps[after].append(z.detach().cpu().numpy().reshape(-1))
                    left -= b
                for g_req, idx, _ in hits:
                    if idx not in snaps:
                        continue
                    np.save(kdir / f"z_g{gtag(g_req)}.npy", np.concatenate(snaps[idx], 0)[:n])
            for g_req, _, g_act in hits:
                gp, tp = kdir / f"z_g{gtag(g_req)}.npy", true_dir / f"z_g{gtag(g_req)}.npy"
                if not (gp.is_file() and tp.is_file()):
                    continue
                gen = np.load(gp)
                tru = np.load(tp)
                m = min(gen.size, tru.size)
                rows.append(
                    {
                        "name": name,
                        "kernel": kernel,
                        "gamma": float(g_req),
                        "g_act": float(g_act),
                        "wd": wd1(gen[:m], tru[:m]),
                        "tv": tv_counts(gen[:m], tru[:m]),
                    }
                )
    write_table(rows, root / "metrics.txt", ("name", "kernel", "gamma", "g_act", "wd", "tv"))
    plot_tv_gamma(rows, names, kernels, root / "tv_vs_gamma.png")
    plot_traj_pmfs(root, names, kernels, hits if names else [], root / "z_pmf_selected.png")


def plot_tv_gamma(rows, names, kernels, path):
    fig, axes = wrap_axes(len(names))
    for i, name in enumerate(names):
        ax = axes.ravel()[i]
        for k in kernels:
            sub = [r for r in rows if r["name"] == name and r["kernel"] == k]
            if not sub:
                continue
            ax.plot(
                [r["g_act"] for r in sub],
                [r["tv"] for r in sub],
                marker="o",
                ms=3,
                color=COLORS.get(k, "gray"),
                label=k,
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"TV$(z_\gamma)$")
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(0, 1)
    if names:
        axes.ravel()[0].legend(fontsize=7, frameon=False)
    hide_extra(axes, len(names))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def plot_traj_pmfs(root, names, kernels, hits, path):
    if not hits:
        return
    gshow = []
    for t in TRAJ_SHOW:
        cand = min(hits, key=lambda h: abs(h[0] - t))
        gshow.append(cand)
    fig, axes = plt.subplots(len(names), len(gshow), figsize=(3.4 * len(gshow), 2.6 * max(len(names), 1)), squeeze=False)
    for i, name in enumerate(names):
        for j, (g_req, _, g_act) in enumerate(gshow):
            ax = axes[i, j]
            tp = root / name / "true" / f"z_g{gtag(g_req)}.npy"
            if not tp.is_file():
                ax.set_title(f"{name} γ≈{g_act:g}", fontsize=8)
                continue
            tru = np.load(tp)
            hi = int(max(np.quantile(tru, 0.995), 1))
            pt, _ = emp_pmf_window(tru, 0, hi)
            ks = np.arange(0, hi + 1)
            ax.plot(ks, pt, color="black", lw=1.8, label="true")
            for k in kernels:
                gp = root / name / k / f"z_g{gtag(g_req)}.npy"
                if not gp.is_file():
                    continue
                pg, _ = emp_pmf_window(np.load(gp), 0, hi)
                ax.plot(ks, pg, color=COLORS.get(k, "gray"), lw=1.1, label=k)
            ax.set_xlim(0, hi)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{name}  γ={g_act:g}", fontsize=8)
            ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


# ---- 2 exponent ----
def exp_exponent(cli, device, names, kernels):
    root = RESULTS / "02_exponent"
    rows = []
    for name in names:
        model, args, _, _ = load_pair(cli, name, device)
        if model is None:
            continue
        g1 = float(args.lbd)
        for p in EXPONENTS:
            tag = f"p{p:g}"
            dest = root / tag / name
            if p == 1.0:
                reuse_samples(LEGACY_SWEEP["T100_h1"] / name, dest, kernels)
            elif p == 2.0:
                reuse_samples(LEGACY_QUAD / name, dest, kernels)
            gammas = make_gammas("quad", 100, 0.0, g1, device, power=p)
            if has_all(dest, kernels):
                print(f"  skip exponent {tag} {name}", flush=True)
            else:
                run_one(
                    model, args, device, dest, kernels, cli.n_sample, gammas,
                    {"test": "exponent", "tag": tag, "name": name, "power": p},
                )
            table = plot_experiment(dest, name=name, kernels=kernels)
            for r in table:
                r["power"] = p
                r["tag"] = tag
                rows.append(r)
    write_table(rows, root / "metrics.txt", ("power", "dist", "kernel", "wd", "tv"))
    plot_metric_vs_p(rows, names, kernels, "tv", root / "tv_vs_p.png")
    plot_metric_vs_p(rows, names, kernels, "wd", root / "wd_vs_p.png")


def plot_metric_vs_p(rows, names, kernels, key, path):
    fig, axes = wrap_axes(len(names))
    for i, name in enumerate(names):
        ax = axes.ravel()[i]
        for k in kernels:
            sub = sorted(
                [r for r in rows if r.get("dist") == name and r["kernel"] == k],
                key=lambda r: float(r["power"]),
            )
            if not sub:
                continue
            ax.plot(
                [float(r["power"]) for r in sub],
                [r[key] for r in sub],
                marker="o",
                color=COLORS.get(k, "gray"),
                label=k,
            )
        ax.set_xlabel(r"$p$ in $\gamma \propto t^p$")
        ax.set_ylabel(key.upper())
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3)
    if names:
        axes.ravel()[0].legend(fontsize=7, frameon=False)
    hide_extra(axes, len(names))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


# ---- 3 R = h v / m ----
def exp_R(cli, device, names, kernels):
    root = RESULTS / "03_R"
    for name in names:
        model, args, _, _ = load_pair(cli, name, device)
        if model is None:
            continue
        g1 = float(args.lbd)
        gammas = make_gammas("quad", 100, 0.0, g1, device, power=2.0)
        hs = (gammas[1:] - gammas[:-1]).cpu().numpy()
        gs = gammas[:-1].cpu().numpy()
        n = int(cli.n_diag)
        for kernel in kernels:
            out = root / name / kernel
            out.mkdir(parents=True, exist_ok=True)
            npz = out / "R_vs_gamma.npz"
            if npz.is_file():
                print(f"  skip R {name} {kernel}", flush=True)
                continue
            print(f"  R {name} {kernel}", flush=True)
            chunks = []
            left = n
            bsz = diag_batch(cli)
            while left > 0:
                b = min(bsz, left)
                z = torch.zeros(b, 1, device=device)
                step_R = np.empty((len(hs), b), dtype=np.float32)
                for i in range(len(hs)):
                    g = float(gammas[i].item())
                    h = float(hs[i])
                    m = predict_x(model, z, g, args).clamp_min(0.0)
                    m1 = predict_x(model, z + 1.0, g, args).clamp_min(0.0)
                    v = (m * m1 - m * m).clamp_min(0.0)
                    R = (h * v / m.clamp_min(1e-8)).reshape(-1)
                    step_R[i] = R.detach().float().cpu().numpy()
                    z = reverse_step(model, z, g, h, kernel, args)
                chunks.append(step_R)
                left -= b
            Rall = np.concatenate(chunks, 1)
            np.savez(
                npz,
                gamma=gs,
                h=hs,
                median=np.median(Rall, 1),
                p90=np.quantile(Rall, 0.9, 1),
                p99=np.quantile(Rall, 0.99, 1),
            )
            release()
        release(model)
    plot_R(root, names, kernels, root / "R_vs_gamma.png")


def plot_R(root, names, kernels, path):
    fig, axes = wrap_axes(len(names))
    for i, name in enumerate(names):
        ax = axes.ravel()[i]
        for k in kernels:
            npz = root / name / k / "R_vs_gamma.npz"
            if not npz.is_file():
                continue
            d = np.load(npz)
            ax.plot(d["gamma"], np.clip(d["median"], 1e-12, None), color=COLORS.get(k, "gray"), label=f"{k} med")
            ax.plot(d["gamma"], np.clip(d["p90"], 1e-12, None), color=COLORS.get(k, "gray"), ls="--", lw=0.9, label=f"{k} p90")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$R=h v/m$")
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
    if names:
        axes.ravel()[0].legend(fontsize=6, frameon=False, ncol=2)
    hide_extra(axes, len(names))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


# ---- 4 oracle mean ----
def exp_oracle_mean(cli, device, names, kernels):
    del kernels
    root = RESULTS / "04_oracle_mean"
    rows = []
    for name in names:
        model, args, _, _ = load_pair(cli, name, device)
        if model is None:
            continue
        xs, logp = support_logp(name, device)
        g1 = float(args.lbd)
        for sched, power, legacy in (
            ("uniform", 1.0, LEGACY_SWEEP["T100_h1"] / name),
            ("quad", 2.0, LEGACY_QUAD / name),
        ):
            gammas = make_gammas("quad", 100, 0.0, g1, device, power=power)
            dest_l = root / f"{sched}_learned" / name
            dest_o = root / f"{sched}_oracle" / name
            reuse_samples(legacy, dest_l, ("poisson",))
            if has_all(dest_l, ("poisson",)):
                print(f"  skip learned {sched} {name}", flush=True)
            else:
                run_one(
                    model, args, device, dest_l, ("poisson",), cli.n_sample, gammas,
                    {"test": "oracle_mean", "which": "learned", "sched": sched, "name": name},
                )
            op = dest_o / "samples_poisson.npy"
            if op.is_file():
                print(f"  skip oracle {sched} {name}", flush=True)
            else:
                print(f"  oracle-mean {sched} {name}", flush=True)
                dest_o.mkdir(parents=True, exist_ok=True)
                xs_out = []
                left = int(cli.n_sample)
                hs = gammas[1:] - gammas[:-1]
                while left > 0:
                    b = min(int(cli.sample_batch), left)
                    z = torch.zeros(b, 1, device=device)
                    for i in range(hs.numel()):
                        g = float(gammas[i].item())
                        h = float(hs[i].item())
                        m = oracle_mean(z, g, xs, logp)
                        z = z + torch.poisson((h * m).clamp_min(0.0))
                    xs_out.append(decode_z(z, float(gammas[-1].item()), args, model))
                    left -= b
                np.save(op, np.concatenate(xs_out, 0))
                (dest_o / "meta.json").write_text(
                    json.dumps({"test": "oracle_mean", "which": "oracle", "sched": sched, "name": name}, indent=2) + "\n"
                )
            for tag, dest in (("learned", dest_l), ("oracle", dest_o)):
                table = plot_experiment(dest, name=name, kernels=("poisson",))
                for r in table:
                    r["sched"] = sched
                    r["which"] = tag
                    rows.append(r)
    write_table(rows, root / "metrics.txt", ("sched", "which", "dist", "kernel", "wd", "tv"))


# ---- 5 cutoff exact then mean-Poisson ----
def exp_cutoff(cli, device, names, kernels):
    del kernels
    root = RESULTS / "05_cutoff"
    rows = []
    for name in names:
        model, args, _, _ = load_pair(cli, name, device)
        if model is None:
            continue
        xs, logp = support_logp(name, device)
        g1 = float(args.lbd)
        gammas = make_gammas("quad", 100, 0.0, g1, device, power=2.0)
        for cut in CUTOFFS:
            dest = root / f"cut_{gtag(cut)}" / name
            path = dest / "samples_poisson.npy"
            if path.is_file():
                print(f"  skip cutoff {cut} {name}", flush=True)
            else:
                print(f"  cutoff {cut} {name}", flush=True)
                dest.mkdir(parents=True, exist_ok=True)
                xs_out = []
                left = int(cli.n_sample)
                hs = gammas[1:] - gammas[:-1]
                while left > 0:
                    b = min(int(cli.sample_batch), left)
                    z = torch.zeros(b, 1, device=device)
                    for i in range(hs.numel()):
                        g = float(gammas[i].item())
                        h = float(hs[i].item())
                        if g < float(cut):
                            x = oracle_sample_x(z, g, xs, logp)
                            z = z + torch.poisson((h * x.clamp_min(0.0)))
                        else:
                            m = predict_x(model, z, g, args)
                            z = z + torch.poisson((h * m).clamp_min(0.0))
                    xs_out.append(decode_z(z, float(gammas[-1].item()), args, model))
                    left -= b
                np.save(path, np.concatenate(xs_out, 0))
                (dest / "meta.json").write_text(
                    json.dumps({"test": "cutoff", "cut": cut, "name": name, "schedule": "quad T=100"}, indent=2) + "\n"
                )
            table = plot_experiment(dest, name=name, kernels=("poisson",))
            for r in table:
                r["cut"] = float(cut)
                rows.append(r)
    write_table(rows, root / "metrics.txt", ("cut", "dist", "kernel", "wd", "tv"))
    plot_cut(rows, names, root / "tv_vs_cut.png")


def plot_cut(rows, names, path):
    fig, axes = wrap_axes(len(names))
    for i, name in enumerate(names):
        ax = axes.ravel()[i]
        sub = sorted([r for r in rows if r.get("dist") == name], key=lambda r: float(r["cut"]))
        if sub:
            ax.plot([r["cut"] for r in sub], [r["tv"] for r in sub], marker="o", color="#ff7f0e")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\gamma_{\mathrm{cut}}$")
        ax.set_ylabel("TV")
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
    hide_extra(axes, len(names))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


# ---- 6 first step ----
def exp_first_step(cli, device, names, kernels):
    del kernels
    root = RESULTS / "06_first_step"
    lines = ["name         true_EX   m_t_eps  m_g0.01  EK_h0.01   EK_h1  TV_h0.01_mθ  TV_h1_mθ"]
    for name in names:
        model, args, x_val, x_mean = load_pair(cli, name, device)
        if model is None:
            continue
        dest = root / name
        dest.mkdir(parents=True, exist_ok=True)
        z0 = torch.zeros(1, 1, device=device)
        m_eps = float(predict_x(model, z0, 0.0, args).item())
        m_001 = float(predict_x(model, z0, 0.01, args).item())
        x = x_val.view(-1)
        k001 = torch.poisson((0.01 * x.clamp_min(0.0))).cpu().numpy().reshape(-1)
        k1 = torch.poisson((1.0 * x.clamp_min(0.0))).cpu().numpy().reshape(-1)
        def gap_tv(K, h, m):
            hi = int(max(np.quantile(K, 0.999), 1))
            emp, _ = emp_pmf_window(K, 0, hi)
            ks = np.arange(0, hi + 1)
            q = stats.poisson.pmf(ks, h * max(m, 0.0))
            q = q / max(float(q.sum()), 1e-12)
            return l1_window(q, emp)
        tv001 = gap_tv(k001, 0.01, m_001)
        tv1 = gap_tv(k1, 1.0, m_eps)
        stats_d = {
            "name": name,
            "true_EX": float(x_mean),
            "m_theta_z0_gamma0_teps": m_eps,
            "m_theta_z0_gamma0.01": m_001,
            "E_K_h0.01_exact": float(k001.mean()),
            "E_K_h1_exact": float(k1.mean()),
            "TV_exact_vs_Pois(h m_theta)_h0.01": tv001,
            "TV_exact_vs_Pois(h m_theta)_h1": tv1,
            "note": "t_eps=1e-4, λ=100 ⇒ z_rescale at γ=0 uses rate λ t_eps=0.01",
        }
        (dest / "stats.json").write_text(json.dumps(stats_d, indent=2) + "\n")
        np.save(dest / "K_h0p01.npy", k001)
        np.save(dest / "K_h1.npy", k1)
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
        for ax, K, h, mnet in (
            (axes[0], k001, 0.01, m_001),
            (axes[1], k1, 1.0, m_eps),
        ):
            hi = int(max(np.quantile(K, 0.999), 1))
            emp, _ = emp_pmf_window(K, 0, hi)
            ks = np.arange(0, hi + 1)
            ax.plot(ks, emp, color="black", lw=2, label="exact K=Pois(hX)")
            ax.plot(ks, stats.poisson.pmf(ks, h * max(mnet, 0.0)), color="#ff7f0e", label=r"Pois($h m_\theta$)")
            ax.plot(ks, stats.poisson.pmf(ks, h * x_mean), color="#2ca02c", label=r"Pois($h E[X]$)")
            ax.set_title(f"{name}  h={h:g}", fontsize=9)
            ax.set_xlim(0, hi)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25)
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(dest / "gap_pmf.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        line = (
            f"{name:12} {x_mean:8.4f} {m_eps:8.4f} {m_001:8.4f} "
            f"{k001.mean():8.4f} {k1.mean():8.4f} {tv001:11.4f} {tv1:8.4f}"
        )
        lines.append(line)
        print(line, flush=True)
    (root / "summary.txt").write_text("\n".join(lines) + "\n")


# ---- 7 twopois stability ----
def exp_twopois(cli, device, names, kernels):
    del kernels
    root = RESULTS / "07_twopois"
    summary = []
    for name in names:
        model, args, _, _ = load_pair(cli, name, device)
        if model is None:
            continue
        dest = root / name
        dest.mkdir(parents=True, exist_ok=True)
        js = dest / "stats.json"
        if js.is_file():
            print(f"  skip twopois {name}", flush=True)
            summary.append(json.loads(js.read_text()))
            del model
            release()
            continue
        print(f"  twopois diag {name}", flush=True)
        g1 = float(args.lbd)
        gammas = make_gammas("quad", 100, 0.0, g1, device, power=2.0)
        hs = gammas[1:] - gammas[:-1]
        gs = gammas[:-1].cpu().numpy()
        acc = {k: [] for k in ("v", "c3", "g", "w", "x1", "x2", "hx1", "hx2")}
        frac_keys = ("x1_neg", "w_extreme", "fallback", "hx2_huge")
        step_frac = {k: [] for k in frac_keys}
        step_hx2_p99 = []
        left = int(cli.n_diag)
        bsz = diag_batch(cli)
        while left > 0:
            b = min(bsz, left)
            z = torch.zeros(b, 1, device=device)
            batch_frac = {k: [] for k in frac_keys}
            batch_hx2_p99 = []
            for i in range(hs.numel()):
                g = float(gammas[i].item())
                h = float(hs[i].item())
                m1, m2, m3 = posterior_moments(model, z, g, args, order=3)
                st = two_pois_stats(m1, m2, m3, h)
                for k in acc:
                    acc[k].append(st[k].reshape(-1).detach().float().cpu().numpy())
                for k in frac_keys:
                    batch_frac[k].append(float(st[k].float().mean().item()))
                batch_hx2_p99.append(float(torch.quantile(st["hx2"].reshape(-1).float(), 0.99).item()))
                z = z + two_pois_gap(m1, m2, m3, h)
            for k in frac_keys:
                step_frac[k].append(np.asarray(batch_frac[k], dtype=np.float64))
            step_hx2_p99.append(np.asarray(batch_hx2_p99, dtype=np.float64))
            left -= b
            release(z)
        out = {}
        for k, chunks in acc.items():
            v = np.concatenate(chunks, 0)
            out[k] = {
                "median": float(np.median(v)),
                "p90": float(np.quantile(v, 0.9)),
                "p99": float(np.quantile(v, 0.99)),
                "p999": float(np.quantile(v, 0.999)),
                "max": float(np.max(v)),
            }
        for k in frac_keys:
            out[f"frac_{k}"] = float(np.mean(np.stack(step_frac[k], 0)))
        np.savez(
            dest / "vs_gamma.npz",
            gamma=gs,
            **{f"frac_{k}": np.mean(np.stack(step_frac[k], 0), 0) for k in frac_keys},
            hx2_p99=np.mean(np.stack(step_hx2_p99, 0), 0),
        )
        out["name"] = name
        js.write_text(json.dumps(out, indent=2) + "\n")
        summary.append(out)
        release(model)
    lines = ["name         frac_x1<0  frac_w_ext  frac_fallback  frac_hx2_huge   x2_p99   hx2_p99   hx2_max"]
    for s in summary:
        lines.append(
            f"{s['name']:12} {s['frac_x1_neg']:10.4f} {s['frac_w_extreme']:10.4f} "
            f"{s['frac_fallback']:13.4f} {s['frac_hx2_huge']:14.4f} "
            f"{s['x2']['p99']:9.4g} {s['hx2']['p99']:9.4g} {s['hx2']['max']:9.4g}"
        )
    text = "\n".join(lines) + "\n"
    (root / "summary.txt").write_text(text)
    print(text, end="", flush=True)
    plot_twopois_frac(root, [s["name"] for s in summary], root / "fallback_vs_gamma.png")


def plot_twopois_frac(root, names, path):
    fig, axes = wrap_axes(len(names))
    for i, name in enumerate(names):
        ax = axes.ravel()[i]
        npz = root / name / "vs_gamma.npz"
        if not npz.is_file():
            continue
        d = np.load(npz)
        ax.plot(d["gamma"], d["frac_fallback"], label="fallback", color="#d62728")
        ax.plot(d["gamma"], d["frac_x1_neg"], label="x1<0", color="#ff7f0e")
        ax.plot(d["gamma"], d["frac_w_extreme"], label="w ext", color="#1f77b4")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("fraction")
        ax.set_title(name, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, which="both")
    if names:
        axes.ravel()[0].legend(fontsize=7, frameon=False)
    hide_extra(axes, len(names))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


# ---- 8 sweep ----
def exp_sweep(cli, device, names, kernels):
    root = RESULTS / "08_sweep"
    rows = []
    for name in names:
        model, args, _, _ = load_pair(cli, name, device)
        if model is None:
            continue
        g1 = float(args.lbd)
        for steps, tag in SWEEP:
            dest = root / tag / name
            reuse_samples(LEGACY_SWEEP[tag] / name, dest, kernels)
            if has_all(dest, kernels):
                print(f"  skip sweep {tag} {name}", flush=True)
            else:
                gammas = make_gammas("uniform", steps, 0.0, g1, device)
                run_one(
                    model, args, device, dest, kernels, cli.n_sample, gammas,
                    {"test": "sweep", "tag": tag, "name": name},
                )
            table = plot_experiment(dest, name=name, kernels=kernels)
            for r in table:
                r["tag"] = tag
                r["steps"] = steps
                rows.append(r)
    write_table(rows, root / "metrics.txt", ("tag", "dist", "kernel", "wd", "tv"))


# ---- 09 oracle mean only at low SNR, then learned mean + Poisson ----
MEAN_CUTS = (0.1, 1.0)


def exp_mean_cut(cli, device, names, kernels):
    del kernels
    root = RESULTS / "09_mean_cut"
    rows = []
    for name in names:
        model, args, _, _ = load_pair(cli, name, device)
        if model is None:
            continue
        xs, logp = support_logp(name, device)
        g1 = float(args.lbd)
        gammas = make_gammas("quad", 100, 0.0, g1, device, power=2.0)
        refs = (
            ("cut_0", RESULTS / "02_exponent" / "p2" / name),
            ("cut_inf", RESULTS / "04_oracle_mean" / "quad_oracle" / name),
        )
        for tag, src in refs:
            dest = root / tag / name
            reuse_samples(src, dest, ("poisson",))
            if has_all(dest, ("poisson",)):
                table = plot_experiment(dest, name=name, kernels=("poisson",))
                for r in table:
                    r["cut"] = 0.0 if tag == "cut_0" else float("inf")
                    r["tag"] = tag
                    rows.append(r)
        for cut in MEAN_CUTS:
            dest = root / f"cut_{gtag(cut)}" / name
            path = dest / "samples_poisson.npy"
            if path.is_file():
                print(f"  skip mean_cut {cut} {name}", flush=True)
            else:
                print(f"  mean_cut {cut} {name}", flush=True)
                dest.mkdir(parents=True, exist_ok=True)

                def step_fn(z, g, h, _g_next, cut=cut, xs=xs, logp=logp, model=model, args=args):
                    if g < float(cut):
                        m = oracle_mean(z, g, xs, logp)
                    else:
                        m = predict_x(model, z, g, args)
                    return z + torch.poisson((h * m).clamp_min(0.0))

                np.save(path, sample_loop(cli.n_sample, cli.sample_batch, device, gammas, args, model, step_fn))
                (dest / "meta.json").write_text(
                    json.dumps(
                        {
                            "test": "mean_cut",
                            "cut": cut,
                            "name": name,
                            "schedule": "quad T=100",
                            "note": "K~Pois(h m); m=oracle E[X|z,γ] if γ<cut else m_θ",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            table = plot_experiment(dest, name=name, kernels=("poisson",))
            for r in table:
                r["cut"] = float(cut)
                r["tag"] = f"cut_{gtag(cut)}"
                rows.append(r)
        del model
        release()
    write_table(rows, root / "metrics.txt", ("cut", "tag", "dist", "kernel", "wd", "tv"))
    plot_mean_cut(rows, names, root / "tv_vs_cut.png")


def plot_mean_cut(rows, names, path):
    fig, axes = wrap_axes(len(names))
    for i, name in enumerate(names):
        ax = axes.ravel()[i]
        sub = [r for r in rows if r.get("dist") == name and np.isfinite(r.get("cut", np.nan)) and float(r["cut"]) > 0]
        sub = sorted(sub, key=lambda r: float(r["cut"]))
        if sub:
            ax.plot([r["cut"] for r in sub], [r["tv"] for r in sub], marker="o", color="#ff7f0e", label="oracle mean then learned")
        zero = [r for r in rows if r.get("dist") == name and r.get("tag") == "cut_0"]
        inf = [r for r in rows if r.get("dist") == name and r.get("tag") == "cut_inf"]
        if zero:
            ax.axhline(zero[0]["tv"], color="#ff7f0e", ls=":", lw=0.9, label="learned all γ")
        if inf:
            ax.axhline(inf[0]["tv"], color="0.4", ls="--", lw=0.9, label="oracle all γ")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\gamma_{\mathrm{cut}}$  (oracle mean below)")
        ax.set_ylabel("TV")
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
    if names:
        axes.ravel()[0].legend(fontsize=7, frameon=False)
    hide_extra(axes, len(names))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


# ---- 10 Two-Poisson from oracle (m1,m2,m3) ----
def exp_oracle_twopois(cli, device, names, kernels):
    del kernels
    root = RESULTS / "10_oracle_twopois"
    rows = []
    for name in names:
        model, args, _, _ = load_pair(cli, name, device)
        if model is None:
            continue
        xs, logp = support_logp(name, device)
        g1 = float(args.lbd)
        for sched, power, learned_src in (
            ("uniform", 1.0, RESULTS / "08_sweep" / "T100_h1" / name),
            ("quad", 2.0, RESULTS / "02_exponent" / "p2" / name),
        ):
            gammas = make_gammas("quad", 100, 0.0, g1, device, power=power)
            dest_l = root / f"{sched}_learned" / name
            dest_o = root / f"{sched}_oracle" / name
            reuse_samples(learned_src, dest_l, ("twopois",))
            if has_all(dest_l, ("twopois",)):
                print(f"  skip twopois learned {sched} {name}", flush=True)
            else:
                run_one(
                    model, args, device, dest_l, ("twopois",), cli.n_sample, gammas,
                    {"test": "oracle_twopois", "which": "learned", "sched": sched, "name": name},
                )
            op = dest_o / "samples_twopois.npy"
            if op.is_file():
                print(f"  skip twopois oracle {sched} {name}", flush=True)
            else:
                print(f"  oracle-twopois {sched} {name}", flush=True)
                dest_o.mkdir(parents=True, exist_ok=True)

                def step_fn(z, g, h, _g_next, xs=xs, logp=logp):
                    m1, m2, m3 = oracle_moments(z, g, xs, logp)
                    return z + two_pois_gap(m1, m2, m3, h)

                np.save(op, sample_loop(cli.n_sample, cli.sample_batch, device, gammas, args, model, step_fn))
                (dest_o / "meta.json").write_text(
                    json.dumps(
                        {
                            "test": "oracle_twopois",
                            "which": "oracle",
                            "sched": sched,
                            "name": name,
                            "note": "Two-Poisson from true E[X^k|z], k=1,2,3",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            for tag, dest in (("learned", dest_l), ("oracle", dest_o)):
                table = plot_experiment(dest, name=name, kernels=("twopois",))
                for r in table:
                    r["sched"] = sched
                    r["which"] = tag
                    rows.append(r)
        del model
        release()
    write_table(rows, root / "metrics.txt", ("sched", "which", "dist", "kernel", "wd", "tv"))


# ---- 12 NB from oracle (m1, m2) ----
def exp_oracle_nb(cli, device, names, kernels):
    del kernels
    root = RESULTS / "12_oracle_nb"
    rows = []
    for name in names:
        model, args, _, _ = load_pair(cli, name, device)
        if model is None:
            continue
        xs, logp = support_logp(name, device)
        g1 = float(args.lbd)
        for sched, power, learned_src in (
            ("uniform", 1.0, RESULTS / "02_exponent" / "p1" / name),
            ("quad", 2.0, RESULTS / "02_exponent" / "p2" / name),
        ):
            gammas = make_gammas("quad", 100, 0.0, g1, device, power=power)
            dest_l = root / f"{sched}_learned" / name
            dest_o = root / f"{sched}_oracle" / name
            reuse_samples(learned_src, dest_l, ("nb",))
            if has_all(dest_l, ("nb",)):
                print(f"  skip nb learned {sched} {name}", flush=True)
            else:
                run_one(
                    model, args, device, dest_l, ("nb",), cli.n_sample, gammas,
                    {"test": "oracle_nb", "which": "learned", "sched": sched, "name": name},
                )
            op = dest_o / "samples_nb.npy"
            if op.is_file():
                print(f"  skip nb oracle {sched} {name}", flush=True)
            else:
                print(f"  oracle-nb {sched} {name}", flush=True)
                dest_o.mkdir(parents=True, exist_ok=True)

                def step_fn(z, g, h, _g_next, xs=xs, logp=logp):
                    m1, m2, _m3 = oracle_moments(z, g, xs, logp)
                    v = (m2 - m1 * m1).clamp_min(0.0)
                    return z + nb_gap(m1, v, h)

                np.save(op, sample_loop(cli.n_sample, cli.sample_batch, device, gammas, args, model, step_fn))
                (dest_o / "meta.json").write_text(
                    json.dumps(
                        {
                            "test": "oracle_nb",
                            "which": "oracle",
                            "sched": sched,
                            "name": name,
                            "note": "NB from true E[X|z], E[X^2|z]; v = m2 - m1^2",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            for tag, dest in (("learned", dest_l), ("oracle", dest_o)):
                table = plot_experiment(dest, name=name, kernels=("nb",))
                for r in table:
                    r["sched"] = sched
                    r["which"] = tag
                    rows.append(r)
        del model
        release()
    write_table(rows, root / "metrics.txt", ("sched", "which", "dist", "kernel", "wd", "tv"))


# ---- 11 θ-trapezoidal mean-Poisson on quadratic ----
def exp_trap(cli, device, names, kernels):
    del kernels
    root = RESULTS / "11_trap"
    rows = []
    variants = (
        ("euler_T100", 100, 0.0, 100),
        ("trap_T100", 100, 0.5, 200),
        ("trap_T50", 50, 0.5, 100),
    )
    for name in names:
        model, args, _, _ = load_pair(cli, name, device)
        if model is None:
            continue
        g1 = float(args.lbd)
        for tag, steps, theta, nfe in variants:
            dest = root / tag / name
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / "samples_poisson.npy"
            if tag == "euler_T100":
                reuse_samples(RESULTS / "02_exponent" / "p2" / name, dest, ("poisson",))
            if path.is_file():
                print(f"  skip trap {tag} {name}", flush=True)
            else:
                print(f"  trap {tag} {name}", flush=True)
                gammas = make_gammas("quad", steps, 0.0, g1, device, power=2.0)
                if theta <= 0:
                    np.save(
                        path,
                        sample_loop(
                            cli.n_sample, cli.sample_batch, device, gammas, args, model,
                            lambda z, g, h, _n, model=model, args=args: reverse_step(model, z, g, h, "poisson", args),
                        ),
                    )
                else:
                    np.save(
                        path,
                        sample_loop(
                            cli.n_sample, cli.sample_batch, device, gammas, args, model,
                            lambda z, g, h, g_next, model=model, args=args, theta=theta: trap_poisson_step(
                                model, z, g, h, g_next, args, theta=theta
                            ),
                        ),
                    )
                (dest / "meta.json").write_text(
                    json.dumps(
                        {
                            "test": "trap",
                            "tag": tag,
                            "name": name,
                            "steps": steps,
                            "theta": theta,
                            "nfe": nfe,
                            "schedule": f"quad T={steps}",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            table = plot_experiment(dest, name=name, kernels=("poisson",))
            for r in table:
                r["tag"] = tag
                r["steps"] = steps
                r["theta"] = theta
                r["nfe"] = nfe
                rows.append(r)
        del model
        release()
    write_table(rows, root / "metrics.txt", ("tag", "steps", "nfe", "dist", "kernel", "wd", "tv"))
    plot_trap(rows, names, root / "tv_vs_method.png")


def plot_trap(rows, names, path):
    order = ("euler_T100", "trap_T50", "trap_T100")
    fig, axes = wrap_axes(len(names))
    for i, name in enumerate(names):
        ax = axes.ravel()[i]
        sub = {r["tag"]: r["tv"] for r in rows if r.get("dist") == name}
        xs = [t for t in order if t in sub]
        ax.bar(range(len(xs)), [sub[t] for t in xs], color="#ff7f0e")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontsize=7, rotation=15)
        ax.set_ylabel("TV")
        ax.set_title(name, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    hide_extra(axes, len(names))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def main():
    cli = parse_args()
    names = split_csv(cli.names)
    kernels = tuple(k for k in split_csv(cli.kernels) if k != "repoisson")
    device = setup_device(cli)
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"exp={cli.exp} device={device} names={names} kernels={kernels}", flush=True)
    with torch.inference_mode():
        {
            "traj": exp_traj,
            "exponent": exp_exponent,
            "R": exp_R,
            "oracle_mean": exp_oracle_mean,
            "cutoff": exp_cutoff,
            "first_step": exp_first_step,
            "twopois": exp_twopois,
            "sweep": exp_sweep,
            "mean_cut": exp_mean_cut,
            "oracle_twopois": exp_oracle_twopois,
            "trap": exp_trap,
            "oracle_nb": exp_oracle_nb,
        }[cli.exp](cli, device, names, kernels)
    print("done", flush=True)


if __name__ == "__main__":
    main()
