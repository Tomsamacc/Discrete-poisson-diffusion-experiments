"""CPU checks for raw-moment conversion and TwoPois fallback counts."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.loss import moments_from_ratio_pred, rising_factorial, uses_raw_log_s  # noqa: E402
from sample import posterior_moments, predict_log_B, two_pois_gap  # noqa: E402


def test_conversion():
    z = torch.tensor([[2.0], [4.0]])
    gamma = 2.5
    a = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.5, -1.0]])
    raw = SimpleNamespace(
        ratio_loss="normalized",
        moment_param="raw",
        moment_loss="raw_multi",
        t_eps=1e-4,
        lbd=100.0,
        prior_moments=[3.0, 10.0, 40.0],
    )
    assert uses_raw_log_s(raw)
    moms = moments_from_ratio_pred(a, z, gamma, raw)
    D = rising_factorial(z, 3)
    for k in range(3):
        expect = D[:, k] * a[:, k].double().exp() / (gamma ** (k + 1))
        got = moms[k].double().view(-1)
        assert torch.allclose(got, expect, rtol=1e-5, atol=1e-5), (k, got, expect)
    prior = moments_from_ratio_pred(a, z, 0.0, raw)
    assert torch.allclose(prior[0].view(-1), torch.full((2,), 3.0))
    assert torch.allclose(prior[2].view(-1), torch.full((2,), 40.0))
    near = 100.0 * (torch.tensor(1.0 / 100.0) ** 2).item()
    floor = moments_from_ratio_pred(a, z, near, raw)
    g_used = max(near, 1e-4 * 100.0)
    expect_floor = D[:, 0] * a[:, 0].double().exp() / g_used
    assert torch.allclose(floor[0].double().view(-1), expect_floor, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(floor[0].view(-1), torch.full((2,), 3.0))
    norm = SimpleNamespace(ratio_loss="normalized", t_eps=1e-4, lbd=100.0, prior_moments=None)
    assert not uses_raw_log_s(norm)
    old = moments_from_ratio_pred(a, z, gamma, norm)
    expect0 = D[:, 0] * a[:, 0].double().exp()
    assert torch.allclose(old[0].double().view(-1), expect0, rtol=1e-5, atol=1e-5)


def test_twopois_stats_do_not_change_k():
    m1 = torch.tensor([1.0, 2.0, 0.0, 5.0, 4.0])
    v = torch.tensor([0.5, 1e-12, 0.2, 1.0, 0.3])
    m2 = m1 * m1 + v
    c3 = torch.tensor([0.2, 0.0, 0.0, 80.0, -30.0])
    m3 = c3 + 3.0 * m1 * m2 - 2.0 * m1.pow(3)
    h = 0.2
    torch.manual_seed(1)
    k0 = two_pois_gap(m1, m2, m3, h)
    torch.manual_seed(1)
    stats = {}
    k1 = two_pois_gap(m1, m2, m3, h, stats=stats)
    assert torch.equal(k0, k1)
    assert stats["n"] == 5
    assert stats["invalid_v"] >= 2
    assert stats["fallback"] >= stats["invalid_v"]
    assert stats["extreme_w"] >= 1
    assert 0 <= stats["invalid_atom"] <= 5
    assert stats["fallback"] <= stats["n"]


def test_ckpt_gamma0_uses_prior():
    ckpt = ROOT / "experiments" / "raw_multi" / "equal" / "gamma_ltj" / "best.pt"
    if not ckpt.is_file():
        return
    sys.path.insert(0, str(ROOT / "test_exp"))
    from sample_raw_multi import load_raw

    model, args = load_raw(ckpt, torch.device("cpu"))
    z0 = torch.zeros(4, 1)
    moms = posterior_moments(model, z0, 0.0, args, order=3)
    assert torch.allclose(moms[0].view(-1), torch.full((4,), float(args.prior_moments[0])), atol=1e-4)
    z = torch.full((4, 1), 3.0)
    gamma = 10.0
    moms = posterior_moments(model, z, gamma, args, order=3)
    log_s = predict_log_B(model, z, gamma, args)
    D = rising_factorial(z, 3)
    for k in range(3):
        expect = D[:, k] * log_s[:, k].double().exp() / (gamma ** (k + 1))
        assert torch.allclose(moms[k].double().view(-1), expect, rtol=1e-4, atol=1e-3)
        assert torch.isfinite(moms[k]).all()


if __name__ == "__main__":
    test_conversion()
    test_twopois_stats_do_not_change_k()
    test_ckpt_gamma0_uses_prior()
    print("ok")
