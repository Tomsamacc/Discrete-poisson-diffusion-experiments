import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loss.loss import full_bregman, quad_reverse_h, raw_multi_bregman, raw_multi_weights, raw_ratio_targets


def _unit(raw, link):
    if link == "bounded":
        return torch.sigmoid(raw)
    if link == "unbounded":
        return F.softplus(raw)
    raise ValueError(link)


def cifar_maps(pred, z, gamma, param, link, gmin, prior=None):
    pred = pred.double()
    if pred.ndim == 1:
        pred = pred.view(-1, 1)
    z = z.reshape(-1).double()
    g = torch.as_tensor(gamma, device=z.device, dtype=torch.float64).reshape(-1)
    if g.numel() == 1:
        g = g.expand(z.shape[0])
    tiny = g < float(gmin) * (1.0 - 1e-3)
    g_use = torch.where(g < float(gmin), torch.full_like(g, float(gmin)), g)
    log_g = g_use.log()
    n_head = int(pred.shape[1])
    m = torch.empty(pred.shape[0], n_head, dtype=torch.float64, device=pred.device)
    s = torch.empty_like(m)
    m[:, 0] = _unit(pred[:, 0], link)
    s[:, 0] = m[:, 0]
    log_d = torch.zeros_like(z)
    for k in range(1, n_head + 1):
        log_d = log_d + (z + float(k)).clamp_min(1e-12).log()
        if k == 1:
            continue
        log_c = log_d - float(k) * log_g
        a = pred[:, k - 1]
        if param == "raw_log":
            sk = a.exp()
            mk = (a + log_c).exp()
        elif param == "direct_ratio":
            sk = F.softplus(a)
            mk = (sk.clamp_min(1e-12).log() + log_c).exp()
        elif param == "log_moment":
            mk = torch.sigmoid(a) if link == "bounded" else a.exp()
            sk = (mk.clamp_min(1e-12).log() - log_c).exp()
        elif param == "root_moment":
            q = _unit(a, link).clamp_min(1e-12)
            mk = q.pow(float(k))
            sk = (float(k) * q.log() - log_c).exp()
        elif param == "direct_moment":
            mk = _unit(a, link)
            sk = (mk.clamp_min(1e-12).log() - log_c).exp()
        elif param == "valid_shape":
            q = torch.sigmoid(a)
            m1 = m[:, 0]
            if k == 2:
                mk = m1.square() + (m1 - m1.square()).clamp_min(0) * q
            else:
                m2 = m[:, 1]
                tiny = m1 < 1e-4
                lower = m2.square() / m1.clamp_min(1e-4)
                lower = torch.where(tiny, torch.zeros_like(m2), lower.clamp(max=m2))
                mk = lower + (m2 - lower).clamp_min(0) * q
            sk = (mk.clamp_min(1e-12).log() - log_c).exp()
        else:
            raise ValueError(param)
        if prior is not None and param in ("raw_log", "direct_ratio") and bool(tiny.any()):
            mk = torch.where(tiny, torch.full_like(mk, float(prior[k - 1])), mk)
        m[:, k - 1] = mk
        s[:, k - 1] = sk
    return m, s


def load_split(path):
    blob = np.load(path)
    xtr = torch.from_numpy(np.asarray(blob["x_train"])).float()
    xva = torch.from_numpy(np.asarray(blob["x_test"])).float()
    if xtr.ndim == 4 and xtr.shape[-1] == 3:
        xtr = xtr.permute(0, 3, 1, 2).contiguous()
        xva = xva.permute(0, 3, 1, 2).contiguous()
    return xtr / 255.0, xva / 255.0


def q_sample(x, t, lbd):
    gamma = t.reshape(-1).float() * float(lbd)
    rate = gamma.view(-1, 1, 1, 1) * x.clamp_min(0)
    return torch.poisson(rate), gamma


def _expand(v, like):
    return v.reshape(-1).to(dtype=torch.float64).view(-1, 1, 1, 1).expand(
        like.shape[0], like.shape[1], like.shape[2], like.shape[3]
    ).reshape(-1)


def _pack(pred):
    return pred.permute(0, 2, 3, 4, 1).reshape(-1, pred.shape[1]).double()


def _log_pois(k, rate):
    kk = k.view(1, -1)
    rr = rate.reshape(-1, 1).clamp_min(0)
    logp = kk * rr.clamp_min(1e-12).log() - rr - torch.lgamma(kk + 1.0)
    return torch.where(rr <= 0, torch.where(kk == 0, torch.zeros_like(logp), torch.full_like(logp, -1e6)), logp)


def _kernel_nll(m1, m2, m3, x, h, kind):
    ks = torch.arange(0, 13, device=x.device, dtype=torch.float64)
    rate = (h.reshape(-1) * x.reshape(-1)).clamp_min(0)
    log_p = _log_pois(ks, rate)
    log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
    mu = (h.reshape(-1) * m1.reshape(-1)).clamp_min(0)
    v = (m2.reshape(-1) - m1.reshape(-1).square()).clamp_min(0)
    var = mu + h.reshape(-1).square() * v
    pois = var <= mu + 1e-8
    p = (mu / var.clamp_min(1e-8)).clamp(1e-4, 1.0 - 1e-4)
    n = (mu.square() / (var - mu).clamp_min(1e-8)).clamp(1e-4, 1e6)
    kk = ks.view(1, -1)
    log_nb = (
        torch.lgamma(kk + n.view(-1, 1))
        - torch.lgamma(n.view(-1, 1))
        - torch.lgamma(kk + 1.0)
        + n.view(-1, 1) * p.view(-1, 1).log()
        + kk * (1.0 - p).view(-1, 1).log()
    )
    log_po = _log_pois(ks, mu)
    if kind == "nb":
        log_q = torch.where(pois.view(-1, 1), log_po, log_nb)
    elif kind == "twopois":
        sigma = v.sqrt()
        c3 = m3.reshape(-1) - 3.0 * m1.reshape(-1) * m2.reshape(-1) + 2.0 * m1.reshape(-1).pow(3)
        g = c3 / sigma.pow(3).clamp_min(1e-12)
        w = (0.5 * (1.0 + g / (g.square() + 4.0).sqrt())).clamp(1e-4, 1.0 - 1e-4)
        x1 = m1.reshape(-1) - sigma * ((1.0 - w) / w).sqrt()
        x2 = m1.reshape(-1) + sigma * (w / (1.0 - w)).sqrt()
        bad = pois | (x1 < 0) | (x2 < 0)
        log_mix = torch.logsumexp(
            torch.stack(
                [
                    w.view(-1, 1).log() + _log_pois(ks, (h.reshape(-1) * x1.clamp_min(0))),
                    (1.0 - w).view(-1, 1).log() + _log_pois(ks, (h.reshape(-1) * x2.clamp_min(0))),
                ],
                dim=0,
            ),
            dim=0,
        )
        log_q = torch.where(bad.view(-1, 1), log_po, log_mix)
    else:
        raise ValueError(kind)
    return -(log_p.exp() * log_q).sum(1)


def data_terms(pred, x, z, gamma, t, spec, lbd, steps, gmin):
    x64 = x.reshape(-1).double().clamp_min(0)
    z64 = z.reshape(-1).double()
    g64 = _expand(gamma, x)
    t64 = _expand(t, x)
    b = x.shape[0]
    pix = int(x[0].numel())
    fam = spec["family"]
    if fam == "mean":
        m1 = _unit(pred[:, 0].double(), spec["link"]).reshape(-1)
        l1 = full_bregman(x64, m1)
        per = l1.view(b, pix).mean(1)
        loss = per.mean()
        return {"loss": loss, "l1": loss, "l2": None, "l3": None, "g1": loss, "g2": None, "g3": None, "per": per, "l1_per": per}
    packed = _pack(pred)
    if fam == "ratio":
        target = raw_ratio_targets(x64, z64, g64, 3)
        per_k = raw_multi_bregman(packed, target)
        mode = "dyn_norm" if spec["weight"] == "dyn_norm" else "equal"
        if spec["weight"] not in ("equal", "dyn_norm"):
            raise ValueError(spec["weight"])
        w = raw_multi_weights(t64, mode, lbd, steps, 3)
        weighted = per_k * w
        total = weighted.sum(-1)
        per = total.view(b, pix).mean(1)
        return {
            "loss": per.mean(),
            "l1": per_k[:, 0].mean(),
            "l2": per_k[:, 1].mean(),
            "l3": per_k[:, 2].mean(),
            "g1": weighted[:, 0].mean(),
            "g2": weighted[:, 1].mean(),
            "g3": weighted[:, 2].mean(),
            "per": per,
            "l1_per": per_k[:, 0].view(b, pix).mean(1),
        }
    m, s = cifar_maps(packed, z64, g64, spec["param"], spec["link"], float(gmin))
    n_head = int(m.shape[1])
    l1 = full_bregman(x64, m[:, 0])
    if fam in ("hybrid_ratio", "hybrid_mixed"):
        target = raw_ratio_targets(x64, z64, g64, n_head)
        l2 = full_bregman(target[:, 1], s[:, 1])
        l3 = full_bregman(x64.pow(3), m[:, 2]) if fam == "hybrid_mixed" and n_head >= 3 else (
            full_bregman(target[:, 2], s[:, 2]) if n_head >= 3 else None
        )
    elif fam == "hybrid_moment":
        l2 = full_bregman(x64.square(), m[:, 1])
        l3 = full_bregman(x64.pow(3), m[:, 2]) if n_head >= 3 else None
    else:
        raise ValueError(fam)
    l2_u = l2
    l3_u = l3
    if spec["weight"] == "step":
        h = quad_reverse_h(t64, lbd, steps)
        l2 = l2 * (h / 2.0)
        if l3 is not None:
            l3 = l3 * (h.square() / 6.0)
    elif spec["weight"] != "equal":
        raise ValueError(spec["weight"])
    scale = float(spec.get("higher_scale", 1.0))
    l2 = l2 * scale * float(spec.get("l2_scale", 1.0))
    if l3 is not None:
        l3 = l3 * scale * float(spec.get("l3_scale", 1.0))
    if bool(spec.get("higher_only")):
        total = l2 if l3 is None else l2 + l3
    else:
        total = l1 + l2 if l3 is None else l1 + l2 + l3
    kind = str(spec.get("kernel") or "none")
    if kind in ("nb", "twopois"):
        hh = quad_reverse_h(t64, lbd, steps)
        m3 = m[:, 2] if n_head >= 3 else m[:, 1]
        kn = _kernel_nll(m[:, 0], m[:, 1], m3, x64, hh, kind)
        if bool(spec.get("kernel_only")):
            total = kn
        else:
            total = total + float(spec.get("kernel_lam", 1.0)) * kn
    per = total.view(b, pix).mean(1)
    return {
        "loss": per.mean(),
        "l1": l1.mean(),
        "l2": l2_u.mean(),
        "l3": None if l3_u is None else l3_u.mean(),
        "g1": l1.mean(),
        "g2": l2.mean(),
        "g3": None if l3 is None else l3.mean(),
        "per": per,
        "l1_per": l1.view(b, pix).mean(1),
    }


def _gather_heads(pred, index):
    b, k = pred.shape[0], pred.shape[1]
    flat = pred.reshape(b, k, -1)
    idx = index.reshape(b, 1, 1).expand(b, k, 1)
    return flat.gather(2, idx).squeeze(-1)


def _gather_z(z, index):
    return z.reshape(z.shape[0], -1).gather(1, index.reshape(-1, 1)).squeeze(1)


def _log_c(z, g, k, gmin):
    g_use = torch.where(g < float(gmin), torch.full_like(g, float(gmin)), g)
    log_d = torch.zeros_like(z)
    for i in range(1, int(k) + 1):
        log_d = log_d + (z + float(i)).clamp_min(1e-12).log()
    return log_d - float(k) * g_use.log()


def log_s_at(pred, z, gamma, index, spec, gmin):
    heads = _gather_heads(pred, index).double()
    if spec["family"] == "ratio":
        return heads
    z_at = _gather_z(z, index).double()
    g = gamma.reshape(-1).double()
    m, s = cifar_maps(heads, z_at, g, spec["param"], spec["link"], gmin)
    log_s = s.clamp_min(1e-12).log()
    log_s = log_s.clone()
    log_s[:, 0] = m[:, 0].clamp_min(1e-12).log() - _log_c(z_at, g, 1, gmin)
    return log_s


def _shift(z, index, delta):
    out = z.clone()
    flat = out.reshape(out.shape[0], -1)
    src = flat.new_full((out.shape[0], 1), float(delta))
    flat.scatter_add_(1, index.reshape(-1, 1), src)
    return out


def _run(model, z, gamma, need_grad):
    if not need_grad:
        with torch.no_grad():
            return model(z, gamma)
    z2 = z.detach().requires_grad_(True)
    g2 = gamma.detach().requires_grad_(True)

    def _fwd(zz, gg):
        return model(zz, gg)

    return torch.utils.checkpoint.checkpoint(_fwd, z2, g2, use_reentrant=False)


def distill_loss(teacher, z, gamma, pred, spec, gmin, alpha):
    b = int(z.shape[0])
    pix = int(z[0].numel())
    index = torch.randint(0, pix, (b,), device=z.device)
    z1 = _shift(z, index, 1.0)
    z2 = _shift(z, index, 2.0)

    def mom(zz):
        raw = teacher(zz, gamma)[:, 0]
        flat = _unit(raw, "bounded").reshape(b, pix)
        return flat.gather(1, index.reshape(b, 1)).squeeze(1).double()

    with torch.no_grad():
        a = mom(z)
        c = mom(z1)
        d = mom(z2)
        y2 = (a * c).clamp_min(0)
        y3 = (a * c * d).clamp_min(0)
    packed = _pack(pred)
    m, _s = cifar_maps(packed, z.reshape(-1).double(), _expand(gamma, z), spec["param"], spec["link"], float(gmin))
    rows = torch.arange(b, device=z.device) * pix + index
    p2 = m[rows, 1]
    p3 = m[rows, 2]
    return full_bregman(y2, p2).mean() + float(alpha) * full_bregman(y3, p3).mean()


def consistency_penalty(model, z, gamma, t, index, spec, lbd, steps, gmin):
    mode = str(spec["cons"])
    if mode == "none":
        return z.new_zeros(())
    pred = _run(model, z, gamma, True)
    a = log_s_at(pred, z, gamma, index, spec, gmin)
    if mode == "anchor":
        z1 = _shift(z, index, 1.0)
        z2 = _shift(z, index, 2.0)
        with torch.no_grad():
            a1 = log_s_at(_run(model, z1, gamma, False), z1, gamma, index, spec, gmin)
            a2 = log_s_at(_run(model, z2, gamma, False), z2, gamma, index, spec, gmin)
        t2 = (a[:, 0] + a1[:, 0]).detach()
        t3 = (a[:, 0] + a1[:, 0] + a2[:, 0]).detach()
        d2 = (a[:, 1] - t2).square()
        d3 = (a[:, 2] - t3).square()
    else:
        z1 = _shift(z, index, 1.0)
        z2 = _shift(z, index, 2.0)
        a1 = log_s_at(_run(model, z1, gamma, True), z1, gamma, index, spec, gmin)
        a2 = log_s_at(_run(model, z2,gamma, True), z2, gamma, index, spec, gmin)
        d2 = (a[:, 1] - a[:, 0] - a1[:, 0]).square()
        d3 = (a[:, 2] - a[:, 0] - a1[:, 0] - a2[:, 0]).square()
    if mode == "step":
        h = quad_reverse_h(t, lbd, steps)
        w2 = h / 2.0
        w3 = h.square() / 6.0
        den = (w2 + w3).clamp_min(1e-12)
        pen = (w2 / den) * d2 + (w3 / den) * d3
    elif mode in ("sym", "anchor"):
        pen = d2 + d3
    else:
        raise ValueError(mode)
    return pen.mean()
