import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cifar.losses import consistency_penalty, data_terms, load_split, q_sample
from cifar.model import CifarPoissonNet, net_input
from cifar.variants import rows_for
from loss.oracle import BIN_LABELS, gamma_bin_index


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    blob = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "th": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        blob["cuda"] = torch.cuda.get_rng_state_all()
    return blob


def restore_rng(blob):
    if not blob:
        return
    random.setstate(blob["py"])
    np.random.set_state(blob["np"])
    torch.random.set_rng_state(blob["th"])
    if blob.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob["cuda"])


def softplus_inv(y):
    y = max(float(y), 1e-8)
    if y > 20.0:
        return y
    return math.log(math.expm1(y))


def logit(y):
    y = min(max(float(y), 1e-4), 1.0 - 1e-4)
    return math.log(y / (1.0 - y))


def _positive_bias(y, link):
    if link == "bounded":
        return logit(y)
    if link == "unbounded":
        return softplus_inv(y)
    raise ValueError(link)


def init_heads(model, spec, m1, m2, m3):
    link = spec["link"]
    for head in model.heads:
        head.weight.data.zero_()
        head.bias.data.zero_()
    if spec["family"] == "ratio":
        return
    model.heads[0].bias.data.fill_(_positive_bias(m1, link))
    if spec["family"] == "mean":
        return
    param = spec["param"]
    if param in ("raw_log", "direct_ratio"):
        return
    if param == "log_moment":
        if link == "bounded":
            b2, b3 = logit(m2), logit(m3)
        else:
            b2, b3 = math.log(max(m2, 1e-8)), math.log(max(m3, 1e-8))
        model.heads[1].bias.data.fill_(b2)
        model.heads[2].bias.data.fill_(b3)
        return
    if param == "valid_shape":
        return
    if param == "root_moment":
        if link == "bounded":
            b2 = logit(max(m2, 1e-8) ** 0.5)
            b3 = logit(max(m3, 1e-8) ** (1.0 / 3.0))
        else:
            b2 = b3 = softplus_inv(m1)
        model.heads[1].bias.data.fill_(b2)
        if len(model.heads) > 2:
            model.heads[2].bias.data.fill_(b3)
        return
    if param == "direct_moment":
        model.heads[1].bias.data.fill_(_positive_bias(m2, link))
        if len(model.heads) > 2:
            model.heads[2].bias.data.fill_(_positive_bias(m3, link))
        return
    raise ValueError(param)


def ema_copy(model):
    ema = copy.deepcopy(model)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for be, b in zip(ema.buffers(), model.buffers()):
        be.copy_(b)


def trunk_params(model):
    skip = set()
    for head in model.heads:
        for p in head.parameters():
            skip.add(id(p))
    return [p for p in model.parameters() if p.requires_grad and id(p) not in skip]


def flat_grad(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    parts = []
    for g in grads:
        if g is not None:
            parts.append(g.detach().reshape(-1).float())
    if not parts:
        return None
    return torch.cat(parts)


def trunk_report(model, g1, g2, g3):
    params = trunk_params(model)
    vecs = [flat_grad(g, params) for g in (g1, g2, g3)]

    def norm(v):
        if v is None:
            return 0.0
        return float(v.norm())

    def cos(a, b):
        if a is None or b is None:
            return None
        den = float(a.norm() * b.norm())
        if den == 0.0:
            return None
        return float(torch.dot(a, b) / den)

    return {
        "n1": norm(vecs[0]),
        "n2": norm(vecs[1]),
        "n3": norm(vecs[2]),
        "c12": cos(vecs[0], vecs[1]),
        "c13": cos(vecs[0], vecs[2]),
        "c23": cos(vecs[1], vecs[2]),
    }


def new_buckets():
    return [[0.0, 0] for _ in BIN_LABELS]


def add_buckets(buckets, values, gamma):
    bi = gamma_bin_index(gamma)
    for i in range(len(buckets)):
        m = bi == i
        if bool(m.any()):
            buckets[i][0] += float(values[m].detach().sum())
            buckets[i][1] += int(m.sum())


def bucket_means(buckets):
    out = {}
    for lab, (total, count) in zip(BIN_LABELS, buckets):
        out[lab] = None if count == 0 else total / count
    return out


def sample_t(n, t_eps, device):
    return torch.rand(n, device=device) * (1.0 - float(t_eps)) + float(t_eps)


@torch.no_grad()
def evaluate(model, loader, spec, lbd, steps, gmin, t_eps, device):
    was = model.training
    model.eval()
    total = 0.0
    l1 = 0.0
    n = 0
    buckets = new_buckets()
    l1_buckets = new_buckets()
    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        t = sample_t(x.shape[0], t_eps, device)
        z, gamma = q_sample(x, t, lbd)
        pred = model(z, gamma)
        terms = data_terms(pred, x, z, gamma, t, spec, lbd, steps, gmin)
        bs = int(x.shape[0])
        total += float(terms["loss"]) * bs
        l1 += float(terms["l1"]) * bs
        n += bs
        add_buckets(buckets, terms["per"], gamma)
        add_buckets(l1_buckets, terms["l1_per"], gamma)
    if was:
        model.train()
    return total / n, l1 / n, bucket_means(buckets), bucket_means(l1_buckets)


def train_epoch(model, ema, opt, loader, spec, args, device, step, show_zin, grad_path):
    model.train()
    total = 0.0
    n = 0
    streak = 0
    zin_stat = None
    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        t = sample_t(x.shape[0], args.t_eps, device)
        z, gamma = q_sample(x, t, args.lbd)
        if show_zin and zin_stat is None:
            zin, _ = net_input(z, gamma, args.t_eps * args.lbd)
            zin_stat = (
                float(zin.min()),
                float(zin.max()),
                float((zin > 1).float().mean()),
            )
        pred = model(z, gamma)
        terms = data_terms(pred, x, z, gamma, t, spec, args.lbd, args.weight_steps, args.t_eps * args.lbd)
        if spec.get("freeze") and "trunk" in spec["freeze"]:
            model.backbone.eval()
        if step % int(args.grad_every) == 0 and terms["g2"] is not None and terms["g3"] is not None and "trunk" not in (spec.get("freeze") or []):
            rec = trunk_report(model, terms["g1"], terms["g2"], terms["g3"])
            rec["step"] = int(step)
            with open(grad_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        loss = terms["loss"]
        alpha = float(spec.get("distill_alpha") or 0.0)
        teacher = getattr(args, "teacher", None)
        if alpha > 0.0 and teacher is not None and pred.shape[1] >= 3:
            from cifar.losses import distill_loss

            loss = loss + distill_loss(teacher, z, gamma, pred, spec, args.t_eps * args.lbd, alpha)
        held = []
        if spec["cons"] != "none":
            for mod in model.modules():
                if mod.__class__.__name__ == "Dropout" and mod.training:
                    held.append(mod)
                    mod.eval()
            index = torch.randint(0, int(x[0].numel()), (x.shape[0],), device=device)
            pen = consistency_penalty(
                model, z, gamma, t, index, spec, args.lbd, args.weight_steps, args.t_eps * args.lbd
            )
            loss = loss + float(spec["lam"]) * pen
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            for mod in held:
                mod.train()
            streak += 1
            step += 1
            if streak >= 20:
                return total / max(n, 1), step, zin_stat, "nonfinite"
            continue
        streak = 0
        loss.backward()
        for mod in held:
            mod.train()
        opt.step()
        ema_update(ema, model, args.ema)
        total += float(loss.detach()) * int(x.shape[0])
        n += int(x.shape[0])
        step += 1
    return total / max(n, 1), step, zin_stat, None


def save_latest(path, model, ema, opt, epoch, step, best, bad, spec, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "epoch": int(epoch),
            "step": int(step),
            "best": float(best),
            "bad": int(bad),
            "spec": spec,
            "args": vars(args),
            "rng": capture_rng(),
        },
        path,
    )


def save_ema(path, ema, epoch, val, spec):
    torch.save(
        {
            "ema": ema.state_dict(),
            "epoch": int(epoch),
            "val": float(val),
            "spec": spec,
        },
        path,
    )


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--lbd", type=float, default=100.0)
    p.add_argument("--t-eps", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weight-steps", type=int, default=100)
    p.add_argument("--grad-every", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data", default="/home/tomsama/scratch/itdpdm/datasets/cifar10/split_80_20.npz")
    p.add_argument("--out-root", default=str(_ROOT / "experiments" / "cifar"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("cuda required")
    rows = rows_for(args.group)
    if args.index < 0 or args.index >= len(rows):
        raise SystemExit(f"index {args.index} outside 0..{len(rows) - 1} for {args.group}")
    spec = dict(rows[args.index])
    out_dir = Path(args.out_root) / f"s{args.seed}" / spec["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "finished.json").is_file():
        print(f"already finished {out_dir}", flush=True)
        return
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)
    xtr, xva = load_split(args.data)
    if float(xtr.min()) < -1e-6 or float(xtr.max()) > 1.0 + 1e-5:
        raise SystemExit(f"expected x in [0,1], got {float(xtr.min())} {float(xtr.max())}")
    m1 = float(xtr.double().mean())
    m2 = float(xtr.double().pow(2).mean())
    m3 = float(xtr.double().pow(3).mean())
    spec["moments"] = [m1, m2, m3]
    gmin = float(args.t_eps) * float(args.lbd)
    print(
        f"run {spec['name']} group={args.group} index={args.index} "
        f"x=[{float(xtr.min()):.4g},{float(xtr.max()):.4g}] mean={m1:.4g} "
        f"link={spec['link']} lbd={args.lbd} gmin={gmin} epochs={args.epochs} patience={args.patience} "
        f"val_every={args.val_every} lr={args.lr} batch={args.batch_size}",
        flush=True,
    )
    train_loader = DataLoader(
        TensorDataset(xtr),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(xva),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = CifarPoissonNet(spec["heads"], gmin).to(device)
    if float(spec.get("distill_alpha") or 0.0) > 0.0:
        teacher = CifarPoissonNet(1, gmin).to(device)
        blob = torch.load(spec["init"], map_location=device, weights_only=False)
        teacher.load_state_dict(blob["ema"])
        teacher.eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
        args.teacher = teacher
    latest = out_dir / "latest.pt"
    start_epoch = 1
    step = 0
    best = float("inf")
    bad = 0
    if latest.is_file():
        blob = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        opt.load_state_dict(blob["opt"])
        ema = CifarPoissonNet(spec["heads"], gmin).to(device)
        ema.load_state_dict(blob["ema"])
        ema.eval()
        for q in ema.parameters():
            q.requires_grad_(False)
        start_epoch = int(blob["epoch"]) + 1
        step = int(blob["step"])
        best = float(blob["best"])
        bad = int(blob["bad"])
        restore_rng(blob.get("rng"))
        print(f"resume epoch={start_epoch} step={step} best={best:.6g} bad={bad}", flush=True)
    else:
        init_heads(model, spec, m1, m2, m3)
        if spec.get("init"):
            blob = torch.load(spec["init"], map_location=device, weights_only=False)
            src = blob.get("ema") or blob["model"]
            dst = model.state_dict()
            for key, value in src.items():
                if key in dst and tuple(dst[key].shape) == tuple(value.shape):
                    dst[key].copy_(value)
            model.load_state_dict(dst)
        for name in spec.get("freeze") or []:
            if name == "trunk":
                for q in model.backbone.parameters():
                    q.requires_grad_(False)
            elif name.startswith("h"):
                for q in model.heads[int(name[1:])].parameters():
                    q.requires_grad_(False)
        groups = []
        trunk = [q for q in model.backbone.parameters() if q.requires_grad]
        heads = [q for h in model.heads for q in h.parameters() if q.requires_grad]
        if trunk:
            groups.append({"params": trunk, "lr": float(args.lr) * float(spec.get("trunk_lr", 1.0))})
        if heads:
            groups.append({"params": heads, "lr": float(args.lr)})
        opt = torch.optim.Adam(groups, lr=args.lr, betas=(args.beta1, args.beta2))
        ema = ema_copy(model)
    if start_epoch > args.epochs:
        write_json(out_dir / "finished.json", {"reason": "epochs", "epoch": args.epochs, "best": best})
        print("budget already reached", flush=True)
        return
    write_json(out_dir / "run.json", {"spec": spec, "args": vars(args)})
    history = []
    hist_path = out_dir / "history.json"
    if hist_path.is_file():
        with open(hist_path) as f:
            history = json.load(f)
    grad_path = out_dir / "grads.jsonl"
    reason = "epochs"
    for epoch in range(start_epoch, args.epochs + 1):
        tr, step, zin_stat, stop = train_epoch(
            model,
            ema,
            opt,
            train_loader,
            spec,
            args,
            device,
            step,
            epoch == start_epoch,
            grad_path,
        )
        if zin_stat is not None:
            print(
                f"zin min={zin_stat[0]:.4g} max={zin_stat[1]:.4g} frac>1={zin_stat[2]:.4g}",
                flush=True,
            )
        save_latest(latest, model, ema, opt, epoch, step, best, bad, spec, args)
        if stop:
            reason = stop
            break
        if epoch % int(args.val_every) != 0 and epoch != args.epochs:
            print(f"epoch {epoch} train {tr:.6g}", flush=True)
            continue
        val, val_l1, bins, l1_bins = evaluate(
            ema, val_loader, spec, args.lbd, args.weight_steps, gmin, args.t_eps, device
        )
        if val < best:
            best = val
            bad = 0
            save_ema(out_dir / "best_ema.pt", ema, epoch, val, spec)
        else:
            bad += 1
        row = {
            "epoch": epoch,
            "train": tr,
            "val": val,
            "val_l1": val_l1,
            "bad": bad,
            "best": best,
            "bins": bins,
            "l1_bins": l1_bins,
        }
        history.append(row)
        write_json(hist_path, history)
        print(
            f"epoch {epoch} train {tr:.6g} val {val:.6g} val_l1 {val_l1:.6g} bad {bad} best {best:.6g}",
            flush=True,
        )
        save_latest(latest, model, ema, opt, epoch, step, best, bad, spec, args)
        if bad >= int(args.patience):
            reason = "patience"
            break
    save_ema(out_dir / "ema_final.pt", ema, epoch, best, spec)
    write_json(out_dir / "finished.json", {"reason": reason, "epoch": int(epoch), "best": best, "bad": bad})
    print(f"finished reason={reason} epoch={epoch} best={best:.6g}", flush=True)


if __name__ == "__main__":
    main()
