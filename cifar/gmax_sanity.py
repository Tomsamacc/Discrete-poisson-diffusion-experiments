import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from cifar.fid_inception import FidInception
from cifar.losses import load_split, q_sample
from cifar.model import CifarPoissonNet
from cifar.sample import clip_image, generate
from cifar.screen_fid import features, frechet, ref_stats
from loss.loss import full_bregman as _breg

DATA = "/home/tomsama/scratch/itdpdm/datasets/cifar10/split_80_20.npz"
INCEPT = "/home/tomsama/scratch/itdpdm/refs/weights-inception-2015-12-05-6726825d.pth"
OLD_A0 = _ROOT / "experiments" / "cifar" / "s0" / "a0_bounded" / "best_ema.pt"
REF_CACHE = _ROOT / "experiments" / "cifar" / "focus" / "screen" / "cifar10_train_inception.npz"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_ppm(path, images):
    n = min(64, int(images.shape[0]))
    side = 8
    h = w = 32
    canvas = np.zeros((side * h, side * w, 3), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, side)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = np.transpose(images[i], (1, 2, 0))
    with open(path, "wb") as f:
        f.write(f"P6\n{canvas.shape[1]} {canvas.shape[0]}\n255\n".encode())
        f.write(np.ascontiguousarray(canvas).tobytes())


def trunk_equal(a, b):
    n = same = 0
    for key, value in a.items():
        if key.startswith("backbone.") and key in b and tuple(b[key].shape) == tuple(value.shape):
            n += 1
            same += int(torch.equal(b[key], value))
    return same, n


def terminal_loss(model, spec, x, lbd, device):
    model.eval()
    total = 0.0
    mae = 0.0
    hi = 0.0
    lo = 0.0
    seen = 0
    gmin = 1e-4 * float(lbd)
    link = spec["link"]
    for i in range(0, int(x.shape[0]), 64):
        batch = x[i : i + 64].to(device)
        t = torch.ones(batch.shape[0], device=device)
        z, gamma = q_sample(batch, t, lbd)
        pred = model(z, gamma)
        raw = pred[:, 0]
        if link == "bounded":
            m1 = torch.sigmoid(raw)
        else:
            m1 = torch.nn.functional.softplus(raw)
        breg = _breg(batch.reshape(-1).double(), m1.reshape(-1).double())
        total += float(breg.sum())
        mae += float((m1 - batch).abs().sum())
        hi += float((m1 > 1).sum())
        lo += float((m1 < 0).sum())
        seen += int(batch.numel())
    return {
        "terminal_bregman": total / max(seen, 1),
        "terminal_mae": mae / max(seen, 1),
        "terminal_gt1": hi / max(seen, 1),
        "terminal_lt0": lo / max(seen, 1),
        "gmin": gmin,
    }


def main():
    lbd = float(sys.argv[1])
    run = _ROOT / "experiments" / "cifar" / "gmax" / f"lbd{int(lbd)}" / "s0" / "a0_bounded"
    ckpt = run / "best_ema.pt"
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}")
    if not torch.cuda.is_available():
        raise SystemExit("cuda required")
    device = torch.device("cuda")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    spec = blob["spec"]
    sd = blob["ema"]
    digest = sha256(ckpt)
    old_digest = sha256(OLD_A0) if OLD_A0.is_file() else ""
    same = n = None
    if OLD_A0.is_file():
        old = torch.load(OLD_A0, map_location="cpu", weights_only=False)["ema"]
        same, n = trunk_equal(sd, old)
    head_max = float(sd["heads.0.weight"].abs().max())
    out_max = float(sd["backbone.out.2.weight"].abs().max())
    print(
        f"ckpt {ckpt} bytes {ckpt.stat().st_size} sha256 {digest} epoch {blob.get('epoch')} val {blob.get('val')}",
        flush=True,
    )
    print(f"old_a0 sha256 {old_digest} trunk_equal {same}/{n} head_max {head_max:.3e} out_max {out_max:.3e}", flush=True)
    if digest == old_digest:
        raise SystemExit("checkpoint hash equals old A0")
    _xtr, xva = load_split(DATA)
    model = CifarPoissonNet(int(spec["heads"]), 1e-4 * lbd).to(device)
    model.load_state_dict(sd)
    model.eval()
    term = terminal_loss(model, spec, xva, lbd, device)
    print("terminal", term, flush=True)
    torch.manual_seed(0)
    gmin = 1e-4 * lbd
    xhat, stats = generate(model, spec, 64, "poisson", 100, lbd, gmin, device, batch=16)
    raw = xhat.float()
    shown = clip_image(raw)
    u8 = (shown * 255.0).round().to(torch.uint8)
    grid_stats = {
        "before_min": float(raw.min()),
        "before_max": float(raw.max()),
        "before_mean": float(raw.mean()),
        "before_std": float(raw.std()),
        "after_min": float(shown.min()),
        "after_max": float(shown.max()),
        "after_mean": float(shown.mean()),
        "after_std": float(shown.std()),
        "uint8_min": int(u8.min()),
        "uint8_max": int(u8.max()),
        "uint8_mean": float(u8.float().mean()),
        "uint8_std": float(u8.float().std()),
        "clip_hi": float((raw > 1).float().mean()),
        "clip_lo": float((raw < 0).float().mean()),
        "img0_eq_img1": bool(torch.equal(u8[0], u8[1])),
        "sample_stats": stats,
    }
    print("grid", grid_stats, flush=True)
    np.save(run / "grid64_u8.npy", u8.cpu().numpy())
    write_ppm(run / "grid64.ppm", u8.cpu().numpy())
    flat = bool(grid_stats["img0_eq_img1"]) and grid_stats["uint8_std"] < 1.0
    dead = head_max == 0.0 and out_max == 0.0
    payload = {
        "lbd": lbd,
        "ckpt": str(ckpt),
        "bytes": ckpt.stat().st_size,
        "sha256": digest,
        "old_a0_sha256": old_digest,
        "trunk_equal": None if same is None else [same, n],
        "head_weight_max": head_max,
        "out_conv_max": out_max,
        "epoch": blob.get("epoch"),
        "val": blob.get("val"),
        "terminal": term,
        "grid": grid_stats,
        "flat": flat,
        "dead": dead,
    }
    if flat or dead:
        payload["fid1000"] = None
        (run / "sanity.json").write_text(json.dumps(payload, indent=2) + "\n")
        print("SANITY FLAT", flush=True)
        return
    torch.manual_seed(0)
    xhat, stats = generate(model, spec, 1000, "poisson", 100, lbd, gmin, device, batch=16)
    shown = clip_image(xhat)
    u8 = (shown * 255.0).round().to(torch.uint8)
    np.save(run / "gen1000_u8.npy", u8.cpu().numpy())
    net = FidInception(INCEPT).to(device)
    mu, sigma = ref_stats(net, device, REF_CACHE)
    feat = features(net, u8, device)
    score = frechet(feat.mean(axis=0), np.cov(feat, rowvar=False), mu, sigma)
    payload["fid1000"] = {"fid": score, "n": int(u8.shape[0]), "stats": stats}
    (run / "sanity.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"fid1000 {score:.4f}", flush=True)


if __name__ == "__main__":
    main()
