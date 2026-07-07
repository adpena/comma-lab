"""Tests for the pure math/state helpers of tools/quadratic_basin_finisher_probe.py (#341).

Checkpoint/scorer-free: only the flatten/CG/LM/subset helpers are exercised. The CG test solves a
REAL small damped system and checks the solution against a direct solve — it would fail if cg_step
were broken (not a constants-only test)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "quadratic_basin_finisher_probe.py"


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("qbf_probe", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qbf_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_flatten_unflatten_roundtrip(probe):
    arrs = {"a": np.arange(6, dtype=np.float32).reshape(2, 3),
            "b": np.array([7.0], np.float32),
            "c": np.full((2, 2), 3.5, np.float32)}
    keys = ["a", "b", "c"]
    vec = probe.flatten_masked(arrs, keys)
    assert vec.shape == (11,)
    shapes = {k: arrs[k].shape for k in keys}
    back = probe.unflatten_masked(vec, shapes, keys)
    for k in keys:
        np.testing.assert_array_equal(back[k], arrs[k])


def test_unflatten_size_mismatch_raises(probe):
    with pytest.raises(ValueError):
        probe.unflatten_masked(np.zeros(5, np.float32), {"a": (2, 3)}, ["a"])


def test_cg_solves_damped_system(probe):
    rng = np.random.default_rng(0)
    A0 = rng.standard_normal((12, 12)).astype(np.float32)
    H = (A0 @ A0.T).astype(np.float32)  # PSD
    lam = 0.5
    g = rng.standard_normal(12).astype(np.float32)
    x = np.zeros(12, np.float32)
    r = -g.copy()
    p = r.copy()
    for _ in range(50):
        hp_lam = (H @ p + lam * p).astype(np.float32)
        x, r, p, pAp, neg = probe.cg_step(x, r, p, hp_lam)
        assert not neg
        if np.linalg.norm(r) < 1e-6:
            break
    x_direct = np.linalg.solve(H + lam * np.eye(12, dtype=np.float32), -g)
    np.testing.assert_allclose(x, x_direct, rtol=1e-3, atol=1e-4)


def test_cg_negcurv_detected(probe):
    # Indefinite direction: (H+lam I) p with negative p^T A p must flag negcurv, not step.
    x = np.zeros(3, np.float32)
    r = np.array([1.0, 0.0, 0.0], np.float32)
    p = r.copy()
    hp_lam = -p  # p^T hp = -1 < 0
    x2, r2, p2, pAp, neg = probe.cg_step(x, r, p, hp_lam)
    assert neg and pAp <= 0.0
    np.testing.assert_array_equal(x2, x)  # no step taken


def test_lm_accept_reject(probe):
    ok, rho = probe.lm_accept(l0=1.0, l1=0.9, pred_red=0.1)
    assert ok and rho == pytest.approx(1.0)
    ok2, _ = probe.lm_accept(l0=1.0, l1=1.1, pred_red=0.1)  # loss rose
    assert not ok2
    ok3, _ = probe.lm_accept(l0=1.0, l1=0.999, pred_red=0.5)  # rho too small
    assert not ok3


def test_choose_subset_deterministic(probe):
    a = probe.choose_subset_pairs(600, 8, 0)
    b = probe.choose_subset_pairs(600, 8, 0)
    assert a == b and len(set(a)) == 8 and all(0 <= x < 600 for x in a)
    assert a == sorted(a)
    assert probe.choose_subset_pairs(600, 8, 1) != a


def test_head_keys_are_affine_head_only(probe):
    # The Stage-0 mask must be exactly the affine output head (no FiLM/trunk/code).
    assert set(probe.HEAD_KEYS) == {
        "out_sdf.weight", "out_sdf.bias", "out_tex.weight", "out_tex.bias", "palette"}
