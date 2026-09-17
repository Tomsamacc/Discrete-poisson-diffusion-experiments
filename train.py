import argparse
import importlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from loss.loss import BregmanLoss
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


def run_epoch(model, loader, loss_fn, args, device, opt=None):
    train = opt is not None
    model.train(train)
    total = 0.0
    n = 0
    for (xb,) in loader:
        xb = xb.to(device)
        t = torch.rand(xb.shape[0], device=device) * (1.0 - args.t_eps) + args.t_eps
        z = q_sample(xb, t, args.lbd)
        target = model.encode_x(xb)
        alpha = t if args.z_rescale else None
        xhat = model(z, t, alpha=alpha)
        loss = loss_fn(xhat, target)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        bs = xb.shape[0]
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


def main():
    args = parse_args()
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
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    loss_fn = BregmanLoss(reduction="mean")

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(
        f"train n={len(xtr)} val={'yes' if val_loader else 'no'} "
        f"lbd={args.lbd} scale={args.z_rescale} lr={args.lr} bs={args.batch_size} "
        f"epochs={args.epochs} val_every={args.val_every} "
        f"device={device} out={out_dir}",
        flush=True,
    )

    best = float("inf")
    last_val = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, args, device, opt=opt)
        tick = epoch % args.val_every == 0 or epoch == args.epochs
        if not tick:
            continue
        val_loss = None
        if val_loader is not None:
            with torch.no_grad():
                val_loss = run_epoch(model, val_loader, loss_fn, args, device, opt=None)
            last_val = val_loss
            print(f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f}", flush=True)
        else:
            print(f"epoch {epoch} train {train_loss:.4f}", flush=True)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss if val_loss is not None else last_val,
            "best_val": best,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "latest.pt")
        metric = val_loss if val_loss is not None else train_loss
        if metric < best:
            best = metric
            ckpt["best_val"] = best
            torch.save(ckpt, out_dir / "best.pt")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
