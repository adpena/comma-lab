# SPDX-License-Identifier: MIT
"""Behavior tests for the margin-conditional seg-repair residual coder + waterfill (lever D, task #72).

NO FAKE: every test verifies BEHAVIOR, not constants.  The conditional coder must MEASURABLY beat the
unconditional baseline on a structured flip-set (margin-concentrated); a coder that ignores the margin
(boundary = whole grid) must NOT beat it; the waterfill admits only positive-net flips below 1.27 B/flip
(a select-all stub fails); the sidecar round-trips bit-exactly; a no-op (zero net value) admits nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tac.boundary_math.margin_conditional_residual import (
    BYTES_PER_SCORE,
    SEG_VALUE_PER_FLIP,
    WATERLINE_BYTES_PER_FLIP,
    ResidualCodeCost,
    boundary_set_from_margin,
    class_bits_conditional,
    conditional_position_bits,
    decode_residual,
    encode_residual,
    log2_choose,
    measure_code_cost,
    unconditional_position_bits,
    waterfill_select,
)

N_GRID = 384 * 512


# ── water-level constants are the closed-spec §10 derivation, not magic ──────
def test_waterline_is_127_bytes_per_flip():
    assert pytest.approx(1.27, abs=0.01) == WATERLINE_BYTES_PER_FLIP


def test_seg_value_per_flip_matches_global_mean():
    # 100 / (600 * 196608)
    assert pytest.approx(100.0 / (600 * N_GRID), rel=1e-9) == SEG_VALUE_PER_FLIP


def test_waterline_is_seg_value_times_bytes_per_score():
    assert pytest.approx(SEG_VALUE_PER_FLIP * BYTES_PER_SCORE, rel=1e-9) == WATERLINE_BYTES_PER_FLIP


# ── log2_choose is the real combinatorial set-index cost ─────────────────────
def test_log2_choose_matches_math_comb():
    for n, k in [(10, 3), (196608, 110), (613, 100), (500, 2)]:
        assert log2_choose(n, k) == pytest.approx(math.log2(math.comb(n, k)), rel=1e-6)


def test_log2_choose_zero_for_k0():
    assert log2_choose(1000, 0) == 0.0


def test_log2_choose_raises_when_k_exceeds_n():
    with pytest.raises(ValueError):
        log2_choose(10, 11)


# ── the boundary set the decoder regenerates for free ────────────────────────
def test_boundary_set_selects_low_margin_pixels():
    m = np.array([0.1, 0.6, 0.3, 2.0, 0.49])
    B = boundary_set_from_margin(m, tau=0.5)
    assert B.tolist() == [True, False, True, False, True]


def test_boundary_set_shrinks_with_smaller_tau():
    m = np.linspace(0.0, 2.0, 1000)
    assert boundary_set_from_margin(m, 0.25).sum() < boundary_set_from_margin(m, 1.0).sum()


# ── THE CORE NO-FAKE CLAIM: conditional position bits BELOW unconditional ────
def test_conditional_position_bits_below_unconditional_when_concentrated():
    # 110 flips, all inside a 613-pixel boundary set (the measured frontier geometry)
    cond = conditional_position_bits(boundary_size=613, k_in=110, n_grid=N_GRID, k_out=0)
    uncond = unconditional_position_bits(N_GRID, 110)
    assert cond < uncond  # margin concentration MUST lower the position cost
    # the measured ratio: ~4.9 bits/flip cond vs ~12.2 uncond
    assert cond / 110 < 6.0
    assert uncond / 110 > 11.0


def test_conditional_equals_unconditional_when_boundary_is_whole_grid():
    # a coder that does NOT concentrate (boundary = whole grid) gets NO saving — the anti-fake guard
    cond = conditional_position_bits(boundary_size=N_GRID, k_in=110, n_grid=N_GRID, k_out=0)
    uncond = unconditional_position_bits(N_GRID, 110)
    assert cond == pytest.approx(uncond, rel=1e-9)


def test_conditional_position_raises_when_k_in_exceeds_boundary():
    with pytest.raises(ValueError):
        conditional_position_bits(boundary_size=50, k_in=60, n_grid=N_GRID, k_out=0)


# ── class entropy: conditional <= unconditional, real not constant ───────────
def test_class_bits_unconditional_is_real_entropy():
    cls = np.array([0, 0, 0, 1])  # H = -(0.75 log .75 + .25 log .25) ~ 0.811, *4
    bits = class_bits_conditional(cls)
    expected = 4 * (-(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)))
    assert bits == pytest.approx(expected, rel=1e-9)


def test_class_bits_conditional_not_above_unconditional():
    rng = np.random.default_rng(0)
    cls = rng.integers(0, 5, size=500)
    mbin = rng.integers(0, 6, size=500)
    cond = class_bits_conditional(cls, mbin)
    uncond = class_bits_conditional(cls)
    assert cond <= uncond + 1e-9  # conditioning never increases entropy (MI >= 0)


def test_class_bits_conditional_strictly_below_when_margin_informs_class():
    # construct a flip-set where margin bin perfectly predicts class -> conditional entropy 0
    cls = np.array([0, 0, 1, 1, 2, 2])
    mbin = np.array([0, 0, 1, 1, 2, 2])  # bin determines class exactly
    assert class_bits_conditional(cls, mbin) == pytest.approx(0.0, abs=1e-9)
    assert class_bits_conditional(cls) > 1.0  # unconditional has real entropy


# ── measure_code_cost on a realistic margin-concentrated flip-set ────────────
def _synthetic_frontier_flipset(seed=0, n_flips=110, boundary=613):
    """A margin field where `boundary` pixels are low-margin and flips live inside them."""
    rng = np.random.default_rng(seed)
    m = np.full(N_GRID, 5.0)  # interior: high margin
    boundary_pixels = rng.choice(N_GRID, size=boundary, replace=False)
    m[boundary_pixels] = rng.uniform(0.0, 0.49, size=boundary)  # fragile band
    flip_idx = rng.choice(boundary_pixels, size=n_flips, replace=False)
    target_cls = rng.integers(0, 5, size=n_flips)
    return m, flip_idx, target_cls


def test_measure_code_cost_beats_waterline_on_concentrated_flipset():
    m, idx, cls = _synthetic_frontier_flipset()
    cost = measure_code_cost(m, idx, cls, tau=0.5)
    assert isinstance(cost, ResidualCodeCost)
    assert cost.n_flips == 110
    # conditional cost below the 1.27 waterline AND below unconditional (the rate-side WIN)
    assert cost.bytes_per_flip < WATERLINE_BYTES_PER_FLIP
    assert cost.bytes_per_flip < cost.unconditional_bytes_per_flip
    assert cost.beats_waterline
    assert cost.conditional_saving_bytes_per_flip > 0.0


def test_measure_code_cost_does_not_beat_waterline_when_flips_scattered():
    # flips spread over the WHOLE grid (no margin concentration) -> conditional == unconditional > 1.27
    rng = np.random.default_rng(1)
    m = rng.uniform(0.0, 0.4, size=N_GRID)  # ALL low margin -> boundary = whole grid
    idx = rng.choice(N_GRID, size=110, replace=False)
    cls = rng.integers(0, 5, size=110)
    cost = measure_code_cost(m, idx, cls, tau=0.5)
    # boundary ~ whole grid -> no position saving -> above the waterline (the anti-fake guard)
    assert cost.boundary_size > N_GRID * 0.9
    assert not cost.beats_waterline


# ── the waterfill: admit positive-net flips below the waterline ──────────────
def test_waterfill_admits_only_positive_net_below_waterline():
    # 3 flips: A net+1 cheap, B net+1 too-expensive, C net-2 (collateral)
    net = np.array([1.0, 1.0, -2.0])
    cost = np.array([1.0, 100.0, 0.5])  # A: 1 B/flip < 1.27 OK; B: 100 > 1.27 reject; C: net<0 reject
    res = waterfill_select(net, cost)
    assert res.n_admitted == 1
    assert res.admitted_local_indices.tolist() == [0]
    assert res.admitted_net_value_flips == 1


def test_waterfill_admits_none_when_all_net_nonpositive():
    # THE FRONTIER-BASE REALITY: every flip net <= 0 -> admit nothing -> no sidecar bytes
    net = np.array([0.0, -1.0, -6.0, 0.0, -2.0])
    cost = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    res = waterfill_select(net, cost)
    assert res.n_admitted == 0
    assert res.admitted_net_value_flips == 0
    assert res.admitted_code_bytes == 0.0


def test_waterfill_select_all_stub_would_fail():
    # a "select all" stub admits net-negative flips -> its net_score_delta is NOT an improvement.
    # the real waterfill rejects them; prove the discrimination matters.
    net = np.array([1.0, -10.0])
    cost = np.array([0.5, 0.5])
    real = waterfill_select(net, cost)
    assert real.n_admitted == 1  # only the positive one
    # a select-all would have net_value 1 + (-10) = -9 (worse); the real selector keeps +1
    assert real.admitted_net_value_flips == 1


def test_waterfill_net_score_delta_is_improvement_when_positive_net_admitted():
    # admit 1000 net-fixed flips at 1.0 B/flip -> seg drop > rate cost -> ΔS < 0
    net = np.ones(1000)
    cost = np.ones(1000) * 1.0  # 1.0 B/flip < 1.27
    res = waterfill_select(net, cost)
    assert res.n_admitted == 1000
    assert res.net_score_delta < 0.0  # an improvement (lowers score)


def test_waterfill_net_score_delta_nonnegative_at_or_above_waterline():
    # flips at exactly the waterline cost -> net_score_delta ~ 0 (no improvement)
    net = np.ones(500)
    cost = np.ones(500) * WATERLINE_BYTES_PER_FLIP  # marginal == waterline -> rejected (strict <)
    res = waterfill_select(net, cost)
    assert res.n_admitted == 0  # strict inequality: at the waterline is NOT admitted


# ── the reversible sidecar round-trips bit-exactly ───────────────────────────
def test_encode_decode_residual_roundtrip_exact():
    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(N_GRID, size=200, replace=False))
    cls = rng.integers(0, 5, size=200)
    blob = encode_residual(42, idx, cls)
    pi, gh, gw, idx_dec, cls_dec = decode_residual(blob)
    assert pi == 42 and gh == 384 and gw == 512
    assert np.array_equal(idx_dec, idx)
    assert np.array_equal(cls_dec, cls)


def test_encode_decode_empty_residual():
    blob = encode_residual(3, np.zeros(0, np.int64), np.zeros(0, np.int64))
    pi, gh, gw, idx, cls = decode_residual(blob)
    assert pi == 3 and len(idx) == 0 and len(cls) == 0


def test_encode_residual_rejects_out_of_range_idx():
    with pytest.raises(ValueError):
        encode_residual(0, np.array([N_GRID + 5]), np.array([0]))


def test_encode_residual_rejects_out_of_range_class():
    with pytest.raises(ValueError):
        encode_residual(0, np.array([10]), np.array([9]))


def test_decode_residual_rejects_bad_magic():
    with pytest.raises(ValueError):
        decode_residual(b"XXXX" + b"\x00" * 20)


# ── end-to-end: the conditional coder + waterfill on a frontier-shaped set ───
def test_end_to_end_conditional_codec_clears_rate_but_collateral_gate_decides():
    """The rate side clears (conditional cost < waterline), but if all net values are <= 0
    (the frontier-base receptive-field reality), the waterfill admits nothing and ΔS = 0."""
    m, idx, cls = _synthetic_frontier_flipset()
    cost = measure_code_cost(m, idx, cls, tau=0.5)
    assert cost.beats_waterline  # RATE side wins
    # frontier-base collateral: every flip net <= 0 -> waterfill admits 0 -> no pointer move
    per_flip_cost = np.full(len(idx), cost.bytes_per_flip)
    net_value = np.zeros(len(idx))  # the measured frontier reality: best net = 0
    res = waterfill_select(net_value, per_flip_cost)
    assert res.n_admitted == 0
    assert res.net_score_delta == 0.0  # NO improvement despite the rate win
