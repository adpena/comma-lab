"""Tests for the damped-Newton semi-discrete OT head-offset solver (deep-math Ch.1
tropical/Laguerre lens; Kitagawa-Merigot-Thibert 2019). Verifies it is a REAL Newton
solve that hits the target masses (KKT ``m(b*) == pi``), not a sweep.

Run: ``.venv/bin/python -m pytest src/tac/boundary_math/tests/test_damped_newton_ot_offsets.py``
"""
from __future__ import annotations

import numpy as np

from tac.boundary_math.laguerre_logit_offset import (
    apply_offset_to_sdf_bias,
    damped_newton_ot_offsets,
    hard_cell_masses,
    menon_logit_adjustment_offsets,
    power_diagram_argmax,
    soft_cell_masses,
)

_RNG = np.random.default_rng(0)


def _synth_logits(n: int = 4000, k: int = 5, spread: float = 2.0) -> np.ndarray:
    """Synthetic per-pixel K-class logits with a deliberately IMBALANCED argmax
    (class 0 dominant, class k-1 nearly collapsed) — the minority-collapse regime."""
    z = _RNG.normal(0.0, spread, size=(n, k))
    z[:, 0] += 1.5           # dominant class
    z[:, -1] -= 2.0          # near-collapsed minority class
    return z


def test_solver_hits_target_masses_soft():
    phi = _synth_logits()
    target = np.array([0.20, 0.20, 0.20, 0.20, 0.20])  # force EQUAL masses (heavy rebalance)
    b, info = damped_newton_ot_offsets(phi, target, tau=1.0, tol=1e-11)
    assert info["converged"] == 1.0, info
    m = soft_cell_masses(phi, b, tau=1.0)
    assert np.allclose(m, target, atol=1e-8), (m, target)
    # KKT: the solved masses match the requested frequencies.
    assert info["max_mass_err"] <= 1e-10, info


def test_zero_sum_gauge_and_byte_free_apply():
    phi = _synth_logits()
    target = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    b, _ = damped_newton_ot_offsets(phi, target, tau=1.0)
    assert abs(float(b.mean())) < 1e-9, "offset must be zero-sum (gauge)"
    # apply_offset_to_sdf_bias changes VALUE not SIZE => byte-free; argmax(phi+b) matches.
    params = {"out_sdf.bias": np.zeros(5, np.float32)}
    out = apply_offset_to_sdf_bias(params, b)
    assert out["out_sdf.bias"].shape == params["out_sdf.bias"].shape  # byte-free
    lab_via_bias = power_diagram_argmax(phi, out["out_sdf.bias"])
    lab_direct = power_diagram_argmax(phi, b)
    assert np.array_equal(lab_via_bias, lab_direct)


def test_quadratic_convergence_few_iters():
    """A real Newton solve converges in a handful of iters (NOT a sweep's hundreds)."""
    phi = _synth_logits()
    target = np.array([0.22, 0.22, 0.20, 0.18, 0.18])
    _, info = damped_newton_ot_offsets(phi, target, tau=1.0, tol=1e-11, max_iter=64)
    assert info["converged"] == 1.0
    assert info["iters"] <= 20.0, f"Newton should be fast, took {info['iters']}"


def test_identity_when_target_is_current_mass():
    """If the target IS the current argmax mass distribution, b* ~= 0 (nothing to do)."""
    phi = _synth_logits()
    cur = soft_cell_masses(phi, np.zeros(5), tau=1.0)
    b, info = damped_newton_ot_offsets(phi, cur, tau=1.0, tol=1e-12)
    assert info["converged"] == 1.0
    assert np.max(np.abs(b)) < 1e-6, b


def test_hard_masses_track_soft_at_small_tau():
    """The hard (argmax) cell masses approach the soft target as tau -> 0 — the
    Kitagawa power-diagram limit. Solve at a small tau, check hard masses are close."""
    phi = _synth_logits(n=8000)
    target = np.array([0.24, 0.22, 0.20, 0.18, 0.16])
    b, info = damped_newton_ot_offsets(phi, target, tau=0.25, tol=1e-11)
    assert info["converged"] == 1.0
    hard = hard_cell_masses(phi, b)
    assert np.max(np.abs(hard - target)) < 0.03, (hard, target)  # hard tracks soft within tol


def test_ot_differs_from_menon_heuristic():
    """The OT solve is NOT the Menon -log(pi) heuristic: for the same target it finds
    DIFFERENT (data-aware) offsets, and unlike Menon it EXACTLY hits the masses."""
    phi = _synth_logits()
    target = np.array([0.24, 0.22, 0.20, 0.18, 0.16])
    b_ot, _ = damped_newton_ot_offsets(phi, target, tau=1.0)
    b_menon = menon_logit_adjustment_offsets(target, tau=1.0)  # heuristic, ignores phi geometry
    assert not np.allclose(b_ot, b_menon, atol=1e-3), "OT must be data-aware, != heuristic"
    # OT hits the target exactly; the fixed heuristic does not.
    err_ot = np.max(np.abs(soft_cell_masses(phi, b_ot, tau=1.0) - target))
    err_menon = np.max(np.abs(soft_cell_masses(phi, b_menon, tau=1.0) - target))
    assert err_ot < 1e-8 < err_menon, (err_ot, err_menon)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
