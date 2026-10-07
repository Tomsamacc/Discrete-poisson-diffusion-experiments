import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cifar.fid_inception import FidInception
from cifar.model import CifarPoissonNet
from cifar.sample import clip_image, generate

INCEPT = "/home/tomsama/scratch/itdpdm/refs/weights-inception-2015-12-05-6726825d.pth"
CIFAR_BATCHES = "/home/tomsama/scratch/itdpdm/datasets/cifar10/raw/cifar-10-batches-py"
N = 10000


def cifar_train_u8():
    parts = []
    folder = Path(CIFAR_BATCHES)
    for i in range(1, 6):
        with open(folder / f"data_batch_{i}", "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        parts.append(np.asarray(entry["data"], dtype=np.uint8))
    raw = np.concatenate(parts, 0).reshape(-1, 3, 32, 32)
    return torch.from_numpy(raw)


def kernels_for(spec):
    heads = int(spec["heads"])
    if heads <= 1:
        return ("poisson",)
    if heads == 2:
        return ("poisson", "nb")
    return ("poisson", "nb", "twopois")


def cases():
    rows = []
    roots = (
        (_ROOT / "experiments" / "cifar" / "s0", "s0"),
        (_ROOT / "experiments" / "cifar" / "focus" / "s0", "focus"),
    )
    for root, prefix in roots:
        if not root.is_dir():
            continue
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            ckpt = folder / "best_ema.pt"
            if ckpt.is_file() and (folder / "finished.json").is_file():
                rows.append((f"{prefix}_{folder.name}", ckpt))
    return rows


def features(net, images, device):
    outs = []
    net.eval()
    with torch.no_grad():
        for i in range(0, int(images.shape[0]), 64):
            batch = images[i : i + 64].to(device)
            outs.append(net(batch).float().cpu())
    return torch.cat(outs, 0).numpy()


def frechet(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    cov, _info = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(cov).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        cov, _info = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * cov))


def ref_stats(net, device, cache):
    if cache.is_file():
        blob = np.load(cache)
        return blob["mu"], blob["sigma"]
    images = cifar_train_u8()
    feat = features(net, images, device)
    mu = feat.mean(axis=0)
    sigma = np.cov(feat, rowvar=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, mu=mu, sigma=sigma)
    return mu, sigma


def main():
    if not torch.cuda.is_available():
        raise SystemExit("cuda required")
    task = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ntasks = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    device = torch.device("cuda")
    net = FidInception(INCEPT).to(device)
    out_root = _ROOT / "experiments" / "cifar" / "focus" / "screen"
    mu, sigma = ref_stats(net, device, out_root / "cifar10_train_inception.npz")
    rows = [row for i, row in enumerate(cases()) if i % ntasks == task]
    print(f"task {task} cases {len(rows)}", flush=True)
    for name, ckpt in rows:
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        spec = blob["spec"]
        gmin = 1e-4 * 100.0
        model = CifarPoissonNet(int(spec["heads"]), gmin).to(device)
        model.load_state_dict(blob["ema"])
        model.eval()
        for kernel in kernels_for(spec):
            dest = out_root / name / kernel
            dest.mkdir(parents=True, exist_ok=True)
            metrics_path = dest / "fid.json"
            if metrics_path.is_file():
                print(f"skip {name} {kernel}", flush=True)
                continue
            gen_path = dest / "gen_u8.npy"
            if gen_path.is_file():
                shown = torch.from_numpy(np.load(gen_path)).float() / 255.0
                stats = {}
            else:
                torch.manual_seed(0)
                print(f"sample {name} {kernel}", flush=True)
                xhat, stats = generate(model, spec, N, kernel, 100, 100.0, gmin, device, batch=16)
                shown = clip_image(xhat)
                np.save(gen_path, (shown * 255.0).round().to(torch.uint8).numpy())
            feat = features(net, (shown * 255.0).round().to(torch.uint8), device)
            score = frechet(feat.mean(axis=0), np.cov(feat, rowvar=False), mu, sigma)
            payload = {
                "fid": score,
                "n": int(shown.shape[0]),
                "ref": "cifar10_train_50000",
                "kernel": kernel,
                "ckpt": str(ckpt),
                "stats": stats,
            }
            metrics_path.write_text(json.dumps(payload) + "\n")
            print(f"{name} {kernel} fid {score:.4f}", flush=True)
            del shown
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
