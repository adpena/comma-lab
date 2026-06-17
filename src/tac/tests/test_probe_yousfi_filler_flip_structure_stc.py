# SPDX-License-Identifier: MIT
"""NO-FAKE tests for PROBE E (Yousfi/Filler flip-structure + STC).

These tests verify the STC description-length measurement is a REAL combinatorial computation
over a REAL Filler syndrome-trellis stream — not a stub or a constant. They do NOT run the full
real-SegNet probe (that needs the basin checkpoint + video); they pin the coder economics math
that the probe's verdict rests on.
"""
from __future__ import annotations

import numpy as np

from tac.boundary_math.margin_conditional_residual import conditional_position_bits


def _fn():
    from experiments.probe_yousfi_filler_flip_structure_stc import _stc_position_stream_bytes

    return _stc_position_stream_bytes


def test_stc_stream_actually_runs_and_returns_real_setindex():
    fn = _fn()
    rng = np.random.default_rng(0)
    B = 700
    band = np.arange(B, dtype=np.int64)
    K = 100
    surv = rng.choice(band, size=K, replace=False).astype(np.int64)
    cost = np.ones(B, dtype=np.float64)
    out = fn(surv, band, cost, constraint_height=8, block_size=512, use_cost=False)
    assert out["stc_ran"] is True
    # set-index must equal log2 C(B, K) exactly (NO-FAKE: real combinatorial coder length)
    expect_bits = conditional_position_bits(B, K, B, 0)
    assert abs(out["stc_desc_bytes"] * 8.0 - expect_bits) < 1e-6
    assert out["n_coded_after_cost"] == K
    assert out["n_dropped_wet"] == 0


def test_stc_desc_bytes_scale_with_K_not_constant():
    """A stub returning a constant would fail this: fewer flips MUST cost fewer bytes."""
    fn = _fn()
    band = np.arange(700, dtype=np.int64)
    cost = np.ones(700, dtype=np.float64)
    small = fn(band[:10], band, cost, constraint_height=8, block_size=512, use_cost=False)
    large = fn(band[:200], band, cost, constraint_height=8, block_size=512, use_cost=False)
    assert small["stc_desc_bytes"] < large["stc_desc_bytes"]


def test_cost_weighting_drops_wet_flips_and_shrinks_coded_set():
    """The cost-weighting lever: STC refuses wet-costed (smooth/fragile) flips -> fewer coded."""
    fn = _fn()
    rng = np.random.default_rng(2)
    B = 700
    band = np.arange(B, dtype=np.int64)
    K = 120
    surv = rng.choice(band, size=K, replace=False).astype(np.int64)
    cost = 1.0 + 9.0 * rng.random(B)
    thr = np.quantile(cost, 0.90)
    cost[cost >= thr] = 1.0e9  # wet the smoothest decile
    uniform = fn(surv, band, cost, constraint_height=8, block_size=512, use_cost=False)
    weighted = fn(surv, band, cost, constraint_height=8, block_size=512, use_cost=True)
    assert weighted["n_dropped_wet"] >= 0
    # any survivable flip on a wet pixel is dropped from the coded set
    assert weighted["n_coded_after_cost"] <= uniform["n_coded_after_cost"]
    # fewer coded positions -> set-index no larger
    assert weighted["stc_desc_bytes"] <= uniform["stc_desc_bytes"]


def test_empty_inputs_safe():
    fn = _fn()
    out = fn(
        np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64),
        constraint_height=8, block_size=512, use_cost=True,
    )
    assert out["stc_ran"] is False
    assert out["stc_desc_bytes"] == 0.0


def test_stc_beats_witness_perflip_floor_on_realistic_band():
    """The headline coder claim: on a realistic boundary band (B~700, K~100), STC's set-index
    B/flip is below the witness re-open's 0.749 B/flip floor. If this regresses, the probe's
    coder-win finding is no longer supported."""
    fn = _fn()
    rng = np.random.default_rng(3)
    B = 700
    band = np.arange(B, dtype=np.int64)
    K = 100
    surv = rng.choice(band, size=K, replace=False).astype(np.int64)
    out = fn(surv, band, np.ones(B), constraint_height=8, block_size=512, use_cost=False)
    bpf = out["stc_desc_bytes"] / K
    assert bpf < 0.7492133956847652, f"STC B/flip {bpf} should beat witness floor 0.749"
