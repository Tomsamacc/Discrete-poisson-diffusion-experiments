import json
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cifar.losses import load_split

DATA = "/home/tomsama/scratch/itdpdm/datasets/cifar10/split_80_20.npz"
GAMMAS = (25, 50, 100, 200, 400, 800)
OUT = _ROOT / "experiments" / "cifar" / "gmax" / "pretest"


def write_ppm(path, canvas):
    h, w, _ = canvas.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(np.ascontiguousarray(canvas).tobytes())


def main():
    _xtr, xva = load_split(DATA)
    x = xva.numpy().astype(np.float64)
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    show_n = 8
    sheet = [np.transpose(np.clip(x[:show_n] * 255.0, 0, 255).round().astype(np.uint8), (0, 2, 3, 1))]
    for gamma in GAMMAS:
        z = rng.poisson(float(gamma) * x)
        zin = z / float(gamma)
        err = zin - x
        mse = float(np.mean(err ** 2))
        rmse = math.sqrt(mse)
        psnr = float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)
        rows.append(
            {
                "gamma": int(gamma),
                "n": int(x.shape[0]),
                "rmse": rmse,
                "psnr": psnr,
                "zin_mean": float(zin.mean()),
                "zin_std": float(zin.std()),
                "x_mean": float(x.mean()),
            }
        )
        print(
            f"gamma {gamma:4d} rmse {rmse:.6f} psnr {psnr:.3f} zin_mean {zin.mean():.4f} zin_std {zin.std():.4f}",
            flush=True,
        )
        vis = np.clip(zin[:show_n] * 255.0, 0, 255).round().astype(np.uint8)
        sheet.append(np.transpose(vis, (0, 2, 3, 1)))
    h, w = 32, 32
    canvas = np.zeros((len(sheet) * h, show_n * w, 3), dtype=np.uint8)
    for r, row in enumerate(sheet):
        for c in range(show_n):
            canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = row[c]
    write_ppm(OUT / "zin_grid.ppm", canvas)
    (OUT / "shot_noise.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(OUT / "shot_noise.json", flush=True)


if __name__ == "__main__":
    main()
