import importlib.util
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cifar.losses import _expand, _pack, _unit, cifar_maps
from loss.loss import posterior_moments_from_logS


def _kernels():
    path = _ROOT / "sample.py"
    spec = importlib.util.spec_from_file_location("poisson1d_sample", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clip_image(x):
    return x.clamp(0, 1)


def predict_moments(model, z, gamma, spec, gmin):
    pred = model(z, gamma)
    if spec["family"] == "mean":
        return [_unit(pred[:, 0], spec["link"]).reshape(-1, 1)]
    packed = _pack(pred)
    g64 = _expand(gamma, z)
    z64 = z.reshape(-1).double()
    if spec["family"] == "ratio":
        return posterior_moments_from_logS(packed, z64, g64, gmin=float(gmin), prior=spec.get("moments"))
    m, _s = cifar_maps(
        packed,
        z64,
        g64,
        spec["param"],
        spec["link"],
        float(gmin),
        prior=spec.get("moments"),
    )
    n = min(int(spec["heads"]), int(m.shape[1]))
    return [m[:, k].float().view(-1, 1) for k in range(n)]


def final_image(model, z, gamma, spec, gmin):
    return predict_moments(model, z, gamma, spec, gmin)[0].view_as(z)


@torch.no_grad()
def generate(model, spec, n, kernel, steps, lbd, gmin, device, batch=8):
    if int(spec["heads"]) == 1 and kernel != "poisson":
        raise ValueError("m1-only samples with poisson")
    kern = _kernels()
    gammas = kern.make_gammas("quad", int(steps), 0.0, float(lbd), device, power=2.0)
    hs = gammas[1:] - gammas[:-1]
    left = int(n)
    outs = []
    neg = 0
    nonfinite = 0
    seen = 0
    max_raw = 0.0
    fb_sum = 0.0
    fb_count = 0
    fb_image = []
    vneg = 0
    vcount = 0
    zin_over = 0
    m_over = 0
    m_seen = 0
    while left > 0:
        b = min(int(batch), left)
        z = torch.zeros(b, 3, 32, 32, device=device)
        image_fb = torch.zeros(b, device=device)
        image_n = 0
        for i in range(hs.numel()):
            g = gammas[i].expand(b)
            h = float(hs[i].item())
            moms = predict_moments(model, z, g, spec, gmin)
            if kernel == "nb" and len(moms) > 1:
                v = moms[1] - moms[0] * moms[0]
                vneg += int((v < 0).sum().item())
                vcount += int(v.numel())
            if kernel == "twopois":
                flags = kern.two_pois_moment_flags(moms[0], moms[1], moms[2])
                fb = flags["fallback"].float().view(b, -1)
                image_fb = image_fb + fb.sum(1)
                image_n += int(fb.shape[1])
                fb_sum += float(fb.sum())
                fb_count += int(fb.numel())
            nxt = kern.step_from_moments(z.reshape(-1, 1), h, kernel, moms)
            z = nxt.reshape(b, 3, 32, 32)
        g_last = gammas[-1].expand(b)
        zin = z / g_last.view(-1, 1, 1, 1)
        xhat = final_image(model, z, g_last, spec, gmin)
        neg += int((zin < 0).sum().item())
        nonfinite += int((~torch.isfinite(xhat)).sum().item())
        max_raw = max(max_raw, float(torch.nan_to_num(zin, nan=0.0).max()))
        seen += int(zin.numel())
        zin_over += int((zin > 1).sum().item())
        m_over += int((xhat > 1).sum().item())
        m_seen += int(xhat.numel())
        outs.append(xhat.cpu())
        if kernel == "twopois" and image_n > 0:
            fb_image.append((image_fb / float(image_n)).cpu())
        left -= b
    stats = {
        "neg_frac": neg / max(seen, 1),
        "nonfinite_frac": nonfinite / max(seen, 1),
        "max_zin": max_raw,
        "zin_gt1_frac": zin_over / max(seen, 1),
        "m_gt1_frac": m_over / max(m_seen, 1),
        "vneg_frac": None if vcount == 0 else vneg / vcount,
        "fallback_frac": None if fb_count == 0 else fb_sum / fb_count,
    }
    if fb_image:
        cat = torch.cat(fb_image)
        stats["fallback_image_p50"] = float(torch.quantile(cat, 0.50))
        stats["fallback_image_p95"] = float(torch.quantile(cat, 0.95))
        stats["fallback_image_p99"] = float(torch.quantile(cat, 0.99))
    return torch.cat(outs, 0), stats
