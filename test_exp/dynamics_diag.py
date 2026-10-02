"""J. Oracle dynamics diagnostic, plus I's target-scale table. No training.

eta_k = h^k m_k* / k! on forward states and on exact-reverse on-policy states.
q(k|z) = E[Pois(k; h X) | z], tail = P(K > 3 | z).
Target R_k scale is the training forward measure, binned by gamma.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from loss.loss import raw_ratio_targets  # noqa: E402
from loss.oracle import (  # noqa: E402
    BIN_LABELS,
    gamma_bin_index,
    poisson_mixture_pmf,
    posterior_weights,
    support_logp,
)
from sample import make_gammas  # noqa: E402
from train import load_counts, q_sample  # noqa: E402

FACTS = (1.0, 1.0, 2.0, 6.0)
SHOW_STEPS = (0, 1, 2, 3, 4, 5, 10, 20, 50, 99)


def qstats(arr):
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "n": int(a.size),
        "p50": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


def moments_from_w(w, xs, k_max=3):
    x = xs.double().view(1, -1)
    w64 = w.double()
    xk = torch.ones_like(w64)
    moms = []
    for _ in range(int(k_max)):
        xk = xk * x
        moms.append((w64 * xk).sum(-1))
    return moms


def summarize_step(z, gamma, h, xs, logp):
    w = posterior_weights(z, gamma, xs, logp)
    moms = moments_from_w(w, xs, 3)
    q, tail3 = poisson_mixture_pmf(w, xs, h, kmax=3)
    h = float(h)
    etas = [(h ** k) * moms[k - 1] / FACTS[k] for k in (1, 2, 3)]
    e1 = etas[0].detach().cpu().numpy()
    e2 = etas[1].detach().cpu().numpy()
    e3 = etas[2].detach().cpu().numpy()
    ok = np.isfinite(e1) & (e1 > 1e-8)
    r21 = np.full_like(e1, np.nan)
    r31 = np.full_like(e1, np.nan)
    r21[ok] = e2[ok] / e1[ok]
    r31[ok] = e3[ok] / e1[ok]
    q_np = q.detach().cpu().numpy()
    t3 = tail3.detach().cpu().numpy()
    t1 = (1.0 - q[:, :2].sum(-1)).clamp_min(0).detach().cpu().numpy()
    t2 = (1.0 - q[:, :3].sum(-1)).clamp_min(0).detach().cpu().numpy()
    s1, s2, s3 = qstats(e1), qstats(e2), qstats(e3)
    a21, a31 = qstats(r21), qstats(r31)
    st = qstats(t3)
    return {
        "n": int(z.shape[0]),
        "eta1_p50": s1["p50"],
        "eta1_p90": s1["p90"],
        "eta2_p50": s2["p50"],
        "eta2_p90": s2["p90"],
        "eta3_p50": s3["p50"],
        "eta3_p90": s3["p90"],
        "r21_p50": a21["p50"],
        "r21_p90": a21["p90"],
        "r21_p99": a21["p99"],
        "r31_p50": a31["p50"],
        "r31_p90": a31["p90"],
        "r31_p99": a31["p99"],
        "q0": float(np.mean(q_np[:, 0])),
        "q1": float(np.mean(q_np[:, 1])),
        "q2": float(np.mean(q_np[:, 2])),
        "q3": float(np.mean(q_np[:, 3])),
        "tail_gt1": float(np.mean(t1)),
        "tail_gt2": float(np.mean(t2)),
        "tail_gt3": float(np.mean(t3)),
        "tail_gt3_p50": st["p50"],
        "tail_gt3_p90": st["p90"],
        "tail_gt3_p99": st["p99"],
        "frac_tail_gt3_gt_0.01": float(np.mean(t3 > 0.01)),
        "frac_tail_gt3_gt_0.1": float(np.mean(t3 > 0.1)),
    }


def forward_z(x, gamma):
    if float(gamma) <= 0:
        return torch.zeros_like(x)
    return torch.poisson((float(gamma) * x.clamp_min(0)))


def onpolicy_step(z, gamma, h, xs, logp):
    w = posterior_weights(z, gamma, xs, logp)
    idx = torch.multinomial(w, 1).squeeze(-1)
    x_draw = xs[idx]
    k = torch.poisson((float(h) * x_draw.clamp_min(0)).view(-1, 1))
    return z + k


def run_chain(name, sched, gammas, source, x, xs, logp):
    rows = []
    z = torch.zeros_like(x) if source == "onpolicy" else None
    steps = int(gammas.numel() - 1)
    for i in range(steps):
        g = float(gammas[i].item())
        h = float((gammas[i + 1] - gammas[i]).item())
        if source == "forward":
            z = forward_z(x, g)
        stats = summarize_step(z, g, h, xs, logp)
        rows.append(
            {
                "dist": name,
                "sched": sched,
                "source": source,
                "step": i,
                "gamma": g,
                "h": h,
                **stats,
            }
        )
        if source == "onpolicy":
            z = onpolicy_step(z, g, h, xs, logp)
        if i in SHOW_STEPS or i == steps - 1:
            print(
                f"{name} {sched} {source} step={i} h={h:.4g} "
                f"q=({stats['q0']:.3g},{stats['q1']:.3g},{stats['q2']:.3g},{stats['q3']:.3g}) "
                f"tail3={stats['tail_gt3']:.3g} p90={stats['tail_gt3_p90']:.3g} "
                f"r21={stats['r21_p50']:.3g} r31={stats['r31_p50']:.3g}",
                flush=True,
            )
    return rows


def target_rows(name, device, n, seed, t_eps, lbd):
    x = load_counts(ROOT / "data" / name / "val.npy")
    if n > 0 and n < x.shape[0]:
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=g)[:n]
        x = x[idx]
    x = x.to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 17)
    t = torch.rand(x.shape[0], device=device, generator=gen) * (1.0 - t_eps) + t_eps
    z = q_sample(x, t, lbd)
    gamma = t * lbd
    idx = gamma_bin_index(gamma).cpu().numpy()
    rows = []
    for k in (1, 2, 3):
        r = raw_ratio_targets(x, z, gamma, k)[:, k - 1].detach().cpu().numpy()
        for b, lab in enumerate(BIN_LABELS):
            st = qstats(r[idx == b])
            rows.append({"dist": name, "k": k, "bin": lab, **st})
            print(
                f"target {name} k={k} {lab} n={st['n']} "
                f"p50={st['p50']:.4g} p90={st['p90']:.4g} p99={st['p99']:.4g} max={st['max']:.4g}",
                flush=True,
            )
    return rows


def write_table(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    lines = [" ".join(keys)]
    for row in rows:
        bits = []
        for key in keys:
            val = row[key]
            if isinstance(val, float):
                bits.append(f"{val:.6g}" if np.isfinite(val) else "nan")
            else:
                bits.append(str(val))
        lines.append(" ".join(bits))
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", default="gamma_ltj,nb,poismix_mod,poissmix3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--target-n", type=int, default=0, help="0 uses the full val set")
    p.add_argument("--schedules", default="quad,unif")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lbd", type=float, default=100.0)
    p.add_argument("--t-eps", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "results" / "15_diag"))
    p.add_argument("--skip-dynamics", action="store_true")
    p.add_argument("--skip-target", action="store_true")
    args = p.parse_args()
    names = [x.strip() for x in args.names.split(",") if x.strip()]
    scheds = [x.strip() for x in args.schedules.split(",") if x.strip()]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.skip_target:
        target = []
        for i, name in enumerate(names):
            target.extend(target_rows(name, device, args.target_n, args.seed + 100 * i, args.t_eps, args.lbd))
        write_table(out / "target_scale.txt", target)
        (out / "target_scale.json").write_text(json.dumps(target, indent=2) + "\n")
        print(f"wrote {out / 'target_scale.txt'}", flush=True)

    if args.skip_dynamics:
        return
    rows = []
    for name in names:
        xs, logp = support_logp(name, device)
        x_all = load_counts(ROOT / "data" / name / "val.npy")
        gen = torch.Generator()
        gen.manual_seed(args.seed)
        if x_all.shape[0] >= args.n:
            take = torch.randperm(x_all.shape[0], generator=gen)[: args.n]
        else:
            take = torch.randint(0, x_all.shape[0], (args.n,), generator=gen)
        x = x_all[take].to(device)
        for sched in scheds:
            power = 2.0 if sched == "quad" else 1.0
            gammas = make_gammas("quad" if sched == "quad" else "uniform", args.steps, 0.0, args.lbd, device, power=power)
            for source in ("forward", "onpolicy"):
                print(f"---- {name} {sched} {source} n={args.n} ----", flush=True)
                rows.extend(run_chain(name, sched, gammas, source, x, xs, logp))
    write_table(out / "dynamics.txt", rows)
    (out / "dynamics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out / 'dynamics.txt'}", flush=True)


if __name__ == "__main__":
    main()
