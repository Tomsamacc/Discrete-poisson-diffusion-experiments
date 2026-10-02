"""Sample raw multi-output checkpoints on the quadratic T=100 schedule.

(S_hat_1, S_hat_2, S_hat_3) -> (m1, m2, m3) -> Poisson / NB / TwoPois.
WD, TV, overflow, runaway, and TwoPois fallback counts.
On-policy moment errors use the same reverse step.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from loss.loss import moments_from_ratio_pred, rising_factorial  # noqa: E402
from loss.oracle import oracle_moments, support_logp  # noqa: E402
from sample import generate, make_gammas, predict_log_B, reverse_step  # noqa: E402
from sample_tests import load_model  # noqa: E402
from test import plot_experiment, plot_grid  # noqa: E402
from train import _qstats, _write_table  # noqa: E402

RESULTS = HERE / "results" / "17_raw_multi_sampling"
ORACLE_JSON = HERE / "results" / "13_next" / "metrics.json"
NAMES = (
    "gamma_ltj",
    "nb",
    "poismix_mod",
    "poissmix3",
    "pois20",
    "poissmix",
    "zip",
    "yule_simon",
)
WEIGHTS = ("equal", "dyn", "dyn_norm")
KERNELS = ("poisson", "nb", "twopois")
FACTS = (1.0, 2.0, 6.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--weights", default=",".join(WEIGHTS))
    p.add_argument("--kernels", default=",".join(KERNELS))
    p.add_argument("--n_sample", type=int, default=50000)
    p.add_argument("--n_onpolicy", type=int, default=1024)
    p.add_argument("--sample_batch", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", default="all", choices=("all", "sample", "onpolicy", "tables"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def split_csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def ckpt_of(weight, name):
    return ROOT / "experiments" / "raw_multi" / weight / name / "best.pt"


def sample_dir(weight, name):
    return RESULTS / "sample" / weight / name


def load_raw(ckpt, device):
    model, args = load_model(ckpt, device)
    args.ratio = True
    args.k_max = 3
    args.moment_param = "raw"
    args.moment_loss = "raw_multi"
    args.ratio_loss = "raw"
    args.sample_batch = 4096
    return model, args


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def write_config(names, weights, kernels, cli):
    RESULTS.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "raw multi-output sampling",
            "schedule: quadratic gamma(u)=lambda*u^2, T=100, lambda=100, gamma from 0 to 100",
            "S_hat_k = exp(a_k)",
            "m_hat_k = D_k(z) * S_hat_k / gamma^k",
            "gamma=0 uses prior moments E[X^k]",
            "kernels: poisson uses m1; nb uses m1,m2; twopois uses m1,m2,m3",
            "twopois fallback: Pois(h m1) if m1<=0 or v<=1e-8*max(m1,1e-8) or x1<0",
            "extreme weight diagnostic: unclamped w<0.01 or w>0.99, kernel unchanged",
            f"names: {','.join(names)}",
            f"weights: {','.join(weights)}",
            f"kernels: {','.join(kernels)}",
            f"n_sample: {cli.n_sample}",
            f"n_onpolicy: {cli.n_onpolicy}",
            f"sample_batch: {cli.sample_batch}",
            f"seed: {cli.seed}",
            "",
        ]
    )
    (RESULTS / "config.txt").write_text(text)
    (RESULTS / "git_commit.txt").write_text(git_rev() + "\n")


def frac(num, den):
    if not den:
        return float("nan")
    return float(num) / float(den)


def run_sample(names, weights, kernels, cli, device):
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    for weight in weights:
        for name in names:
            ckpt = ckpt_of(weight, name)
            dest = sample_dir(weight, name)
            dest.mkdir(parents=True, exist_ok=True)
            model, args = load_raw(ckpt, device)
            args.sample_batch = cli.sample_batch
            gammas = make_gammas("quad", 100, 0.0, float(args.lbd), device, power=2.0)
            print(
                f"sample {weight} {name} quad T=100 ratio_loss=raw k_max=3 ckpt={ckpt}",
                flush=True,
            )
            for kernel in kernels:
                path = dest / f"samples_{kernel}.npy"
                cpath = dest / f"counters_{kernel}.json"
                if path.is_file() and cpath.is_file() and not cli.force:
                    print(f"  skip {kernel}", flush=True)
                    continue
                counters = {}
                x = generate(
                    model, cli.n_sample, kernel, args, device, gammas=gammas, counters=counters
                )
                np.save(path, x)
                cpath.write_text(json.dumps(counters) + "\n")
                line = f"  saved {kernel} E[z/gamma]={float(np.mean(x)):.4g}"
                if counters.get("traj"):
                    line += f" runaway={frac(counters.get('runaway_traj', 0), counters['traj']):.4g}"
                if kernel == "twopois" and counters.get("n"):
                    line += f" fallback={frac(counters.get('fallback', 0), counters['n']):.4g}"
                print(line, flush=True)
            (dest / "meta.json").write_text(
                json.dumps(
                    {
                        "weight": weight,
                        "name": name,
                        "schedule": "quad",
                        "steps": 100,
                        "lbd": 100.0,
                        "moment": "D_k exp(a_k) / gamma^k",
                        "ckpt": str(ckpt),
                    },
                    indent=2,
                )
                + "\n"
            )


def _cat(parts):
    if not parts:
        return np.array([])
    return np.concatenate(parts)


@torch.no_grad()
def onpolicy_one(model, args, device, xs, logp, gammas, n, kernel):
    hs = gammas[1:] - gammas[:-1]
    z = torch.zeros(int(n), 1, device=device)
    store = {
        k: {"eS": [], "em": [], "abs_eta": []}
        for k in (1, 2, 3)
    }
    h_abs = []
    kl = []
    bad = torch.zeros(int(n), dtype=torch.bool, device=device)
    gmin = float(args.t_eps) * float(args.lbd)
    for i in range(hs.numel()):
        g = float(gammas[i].item())
        h = float(hs[i].item())
        log_s = predict_log_B(model, z, g, args).double()
        moms = [m.reshape(-1).double() for m in moments_from_ratio_pred(log_s, z, g, args)]
        stars = [m.reshape(-1).double() for m in oracle_moments(z, g, xs, logp, k_max=3)]
        D = rising_factorial(z, 3)
        gd = torch.as_tensor(g, device=device, dtype=torch.float64)
        for k in (1, 2, 3):
            m_hat = moms[k - 1]
            m_star = stars[k - 1]
            eta_h = (h ** k) * m_hat / FACTS[k - 1]
            eta_s = (h ** k) * m_star / FACTS[k - 1]
            store[k]["abs_eta"].append((eta_h - eta_s).abs().detach().cpu().numpy())
            if g >= gmin * (1.0 - 1e-3):
                s_hat = log_s[:, k - 1].exp()
                s_star = m_star * gd.pow(k) / D[:, k - 1].clamp_min(1e-12)
                store[k]["eS"].append(
                    (s_hat.clamp_min(1e-12).log() - s_star.clamp_min(1e-12).log()).detach().cpu().numpy()
                )
                store[k]["em"].append(
                    (m_hat.clamp_min(1e-12).log() - m_star.clamp_min(1e-12).log()).detach().cpu().numpy()
                )
        m_hat = moms[0]
        m_star = stars[0]
        h_abs.append((h * (m_hat - m_star).abs()).detach().cpu().numpy())
        ms = m_star.clamp_min(1e-12)
        mh = m_hat.clamp_min(1e-12)
        kl.append((h * (ms * (ms / mh).log() + mh - ms)).detach().cpu().numpy())
        z = reverse_step(model, z, g, h, kernel, args)
        z1 = z.reshape(-1)
        bad = bad | (~torch.isfinite(z1)) | (z1 > 1e6)
    rows = []
    for k in (1, 2, 3):
        eS = _qstats(_cat(store[k]["eS"]))
        em = _qstats(_cat(store[k]["em"]))
        ae = _qstats(_cat(store[k]["abs_eta"]))
        row = {
            "k": k,
            "eS_n": eS["n"],
            "eS_median": eS["median"],
            "eS_p90": eS["p90"],
            "eS_p99": eS["p99"],
            "eS_max": eS["max"],
            "em_median": em["median"],
            "em_p90": em["p90"],
            "em_p99": em["p99"],
            "em_max": em["max"],
            "abs_eta_median": ae["median"],
            "abs_eta_p90": ae["p90"],
            "abs_eta_p99": ae["p99"],
            "abs_eta_max": ae["max"],
            "runaway_frac": float(bad.float().mean().cpu()),
        }
        if k == 1:
            ha = _qstats(_cat(h_abs))
            kc = _qstats(_cat(kl))
            row["h_abs_m1_median"] = ha["median"]
            row["h_abs_m1_p99"] = ha["p99"]
            row["h_abs_m1_max"] = ha["max"]
            row["kl_median"] = kc["median"]
            row["kl_p99"] = kc["p99"]
            row["kl_max"] = kc["max"]
        rows.append(row)
    return rows


def run_onpolicy(names, weights, kernels, cli, device):
    torch.manual_seed(cli.seed)
    gammas = make_gammas("quad", 100, 0.0, 100.0, device, power=2.0)
    for weight in weights:
        for name in names:
            xs, logp = support_logp(name, device)
            model, args = load_raw(ckpt_of(weight, name), device)
            for kernel in kernels:
                path = sample_dir(weight, name) / f"onpolicy_{kernel}.json"
                if path.is_file() and not cli.force:
                    print(f"skip onpolicy {weight} {name} {kernel}", flush=True)
                    continue
                print(f"onpolicy {weight} {name} {kernel} n={cli.n_onpolicy}", flush=True)
                rows = onpolicy_one(model, args, device, xs, logp, gammas, cli.n_onpolicy, kernel)
                for row in rows:
                    row["weight"] = weight
                    row["dist"] = name
                    row["kernel"] = kernel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rows, indent=2) + "\n")


def load_oracle():
    if not ORACLE_JSON.is_file():
        return []
    rows = []
    for row in json.loads(ORACLE_JSON.read_text()):
        if row.get("tag") == "G_oracle":
            rows.append(row)
    return rows


def collect_metrics(names, weights, kernels, force_plot):
    rows = []
    for weight in weights:
        for name in names:
            dest = sample_dir(weight, name)
            gens = [k for k in kernels if (dest / f"samples_{k}.npy").is_file()]
            if not gens:
                continue
            side = dest / "metrics.json"
            table = None
            if side.is_file() and not force_plot:
                table = json.loads(side.read_text())
            if table is None:
                table = plot_experiment(dest, name=name, kernels=tuple(gens))
                side.write_text(json.dumps(table, indent=2) + "\n")
            by_k = {r["kernel"]: r for r in table}
            for kernel in gens:
                rec = by_k.get(kernel)
                if rec is None:
                    continue
                counters = {}
                cpath = dest / f"counters_{kernel}.json"
                if cpath.is_file():
                    counters = json.loads(cpath.read_text())
                n_tp = int(counters.get("n") or 0)
                n_tr = int(counters.get("traj") or 0)
                rows.append(
                    {
                        "weight": weight,
                        "dist": name,
                        "kernel": kernel,
                        "wd": rec["wd"],
                        "tv": rec["tv"],
                        "overflow": rec.get("overflow", float("nan")),
                        "runaway": frac(counters.get("runaway_traj", 0), n_tr),
                        "fallback": frac(counters.get("fallback", 0), n_tp) if kernel == "twopois" else float("nan"),
                        "invalid_v": frac(counters.get("invalid_v", 0), n_tp) if kernel == "twopois" else float("nan"),
                        "invalid_atom": frac(counters.get("invalid_atom", 0), n_tp) if kernel == "twopois" else float("nan"),
                        "extreme_w": frac(counters.get("extreme_w", 0), n_tp) if kernel == "twopois" else float("nan"),
                    }
                )
    return rows


def cell_pair(row):
    if row is None or not np.isfinite(row["wd"]) or not np.isfinite(row["tv"]):
        return "nan"
    return f"{row['wd']:.4f}/{row['tv']:.4f}"


def fmt_num(v):
    if v is None or not np.isfinite(v):
        return "nan"
    return f"{v:.4f}"


def write_markdown(names, weights, learned, oracle_rows):
    oracle = {(r["dist"], r["kernel"]): r for r in oracle_rows}
    learned_ix = {(r["weight"], r["dist"], r["kernel"]): r for r in learned}

    def metric_cells(get):
        pois, nb, two = get("poisson"), get("nb"), get("twopois")
        ov = " / ".join(fmt_num(r.get("overflow") if r else float("nan")) for r in (pois, nb, two))
        rw = " / ".join(fmt_num(r.get("runaway") if r else float("nan")) for r in (pois, nb, two))
        fb = fmt_num(two.get("fallback") if two else float("nan"))
        return f"{cell_pair(pois)} | {cell_pair(nb)} | {cell_pair(two)} | {ov} | {rw} | {fb}"

    def one_row(label, get):
        return f"| {label} | {metric_cells(get)} |"

    RESULTS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# raw multi-output sampling",
        "",
        "cell = WD/TV. quadratic T=100, lambda=100.",
        "oracle rows are G from test_exp/results/13_next/metrics.json.",
        "",
    ]
    head = "| weight | Poisson WD/TV | NB WD/TV | TwoPois WD/TV | overflow pois/nb/two | runaway pois/nb/two | TwoPois fallback |"
    rule = "| --- | --- | --- | --- | --- | --- | --- |"
    for name in names:
        lines.extend([f"## {name}", "", head, rule])
        lines.append(one_row("oracle", lambda k, name=name: oracle.get((name, k))))
        for weight in weights:
            lines.append(
                one_row(weight, lambda k, name=name, weight=weight: learned_ix.get((weight, name, k)))
            )
        lines.append("")
    lines.extend(
        [
            "## all datasets",
            "",
            "| dataset | weight | Poisson WD/TV | NB WD/TV | TwoPois WD/TV | overflow pois/nb/two | runaway pois/nb/two | TwoPois fallback |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name in names:
        for label, get in (
            ("oracle", lambda k, name=name: oracle.get((name, k))),
            *[
                (weight, lambda k, name=name, weight=weight: learned_ix.get((weight, name, k)))
                for weight in weights
            ],
        ):
            lines.append(f"| {name} | {label} | {metric_cells(get)} |")
    (RESULTS / "sampling_all_datasets.txt").write_text("\n".join(lines) + "\n")


def write_onpolicy_table(names, weights, kernels):
    rows = []
    for weight in weights:
        for name in names:
            for kernel in kernels:
                path = sample_dir(weight, name) / f"onpolicy_{kernel}.json"
                if not path.is_file():
                    continue
                rows.extend(json.loads(path.read_text()))
    if not rows:
        return
    keys = [
        "weight",
        "dist",
        "kernel",
        "k",
        "eS_n",
        "eS_median",
        "eS_p90",
        "eS_p99",
        "eS_max",
        "em_median",
        "em_p90",
        "em_p99",
        "em_max",
        "abs_eta_median",
        "abs_eta_p90",
        "abs_eta_p99",
        "abs_eta_max",
        "h_abs_m1_median",
        "h_abs_m1_p99",
        "h_abs_m1_max",
        "kl_median",
        "kl_p99",
        "kl_max",
        "runaway_frac",
    ]
    slim = []
    for row in rows:
        slim.append({key: row.get(key, float("nan")) for key in keys})
    _write_table(RESULTS / "onpolicy.txt", slim)
    (RESULTS / "onpolicy.json").write_text(json.dumps(rows, indent=2) + "\n")


def write_fallbacks(learned):
    rows = [r for r in learned if r["kernel"] == "twopois"]
    _write_table(
        RESULTS / "twopois_fallbacks.txt",
        [
            {
                "weight": r["weight"],
                "dist": r["dist"],
                "fallback": r["fallback"],
                "invalid_v": r["invalid_v"],
                "invalid_atom": r["invalid_atom"],
                "extreme_w": r["extreme_w"],
            }
            for r in rows
        ],
    )


def bar_figure(learned, oracle_rows, names, weights, key, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    fig.subplots_adjust(top=0.82, wspace=0.28)
    fig.suptitle(f"{key.upper()}  quadratic T=100", fontsize=12)
    labels = ["oracle", *weights]
    x = np.arange(len(names))
    width = 0.18
    learned_ix = {(r["weight"], r["dist"], r["kernel"]): r for r in learned}
    oracle = {(r["dist"], r["kernel"]): r for r in oracle_rows}
    for ax, kernel in zip(axes, KERNELS):
        for i, label in enumerate(labels):
            ys = []
            for name in names:
                if label == "oracle":
                    rec = oracle.get((name, kernel))
                else:
                    rec = learned_ix.get((label, name, kernel))
                val = rec.get(key) if rec else float("nan")
                ys.append(val if val is not None else float("nan"))
            ax.bar(x + (i - 1.5) * width, ys, width, label=label)
        ax.set_xticks(x, names, rotation=35, ha="right", fontsize=8)
        ax.set_title(kernel, loc="left", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.28))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_tables(names, weights, kernels, force_plot):
    RESULTS.mkdir(parents=True, exist_ok=True)
    learned = collect_metrics(names, weights, kernels, force_plot)
    oracle_rows = load_oracle()
    _write_table(RESULTS / "sampling_metrics.txt", learned)
    (RESULTS / "sampling_metrics.json").write_text(json.dumps(learned, indent=2) + "\n")
    write_markdown(names, weights, learned, oracle_rows)
    write_fallbacks(learned)
    write_onpolicy_table(names, weights, kernels)
    if learned:
        bar_figure(learned, oracle_rows, names, weights, "wd", RESULTS / "wd_all_datasets.png")
        bar_figure(learned, oracle_rows, names, weights, "tv", RESULTS / "tv_all_datasets.png")
    for weight in weights:
        root = RESULTS / "sample" / weight
        if any((root / name / "samples_poisson.npy").is_file() for name in names):
            plot_grid(root, list(names), kernels, False, RESULTS / f"pmf_grid_{weight}.png")
    print(f"wrote {RESULTS}", flush=True)


def main():
    cli = parse_args()
    names = split_csv(cli.names)
    weights = split_csv(cli.weights)
    kernels = split_csv(cli.kernels)
    missing = [str(ckpt_of(w, n)) for w in weights for n in names if not ckpt_of(w, n).is_file()]
    if missing and cli.stage != "tables":
        print("missing checkpoints:", flush=True)
        for path in missing:
            print(f"  {path}", flush=True)
        print("train first: ./script/train_raw_multi.sh", flush=True)
        raise SystemExit(1)
    write_config(names, weights, kernels, cli)
    device = torch.device(cli.device if torch.cuda.is_available() or cli.device == "cpu" else "cpu")
    if cli.stage in ("all", "sample"):
        run_sample(names, weights, kernels, cli, device)
    if cli.stage in ("all", "onpolicy"):
        run_onpolicy(names, weights, kernels, cli, device)
    write_tables(names, weights, kernels, cli.force)


if __name__ == "__main__":
    main()
