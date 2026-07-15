# SPDX-License-Identifier: MIT
"""Tests for the AutoClip percentile grad-clip state machine (#B-4 clip cure).

NO-FAKE: every test verifies BEHAVIOR (thresholds, ring semantics, resume
bit-faithfulness), not markers. Pure numpy — $0."""
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.adaptive_grad_clip import (
    AUTOCLIP_RESUME_PREFIX,
    GRAD_CLIP_MODES,
    AutoClipController,
    AutoClipPercentileState,
)


def test_modes_tuple_is_fixed_and_autoclip():
    assert GRAD_CLIP_MODES == ("fixed", "autoclip")


def test_warmup_returns_fallback_clip():
    st = AutoClipPercentileState(percentile=10.0, window=100, warmup_steps=5, fallback_clip=0.5)
    for g in (6.0, 5.9, 6.2, 6.1):  # 4 < warmup 5
        assert st.observe(g)
        assert st.threshold() == 0.5


def test_percentile_engages_after_warmup_and_matches_numpy():
    st = AutoClipPercentileState(percentile=10.0, window=100, warmup_steps=5, fallback_clip=0.5)
    norms = [6.0, 5.9, 6.2, 6.1, 17.5, 5.8, 6.05]
    for g in norms:
        st.observe(g)
    expected = float(np.percentile(np.asarray(norms, np.float64), 10.0))
    assert st.threshold() == pytest.approx(expected, rel=0, abs=0)


def test_saturation_cure_shape_threshold_tracks_norm_scale():
    """The C0 poison: norms ~6 vs fixed clip 0.5 => frac_clipped 1.0. Under the law the
    threshold sits at the p10 of the realized distribution => only spikes clip."""
    rng = np.random.default_rng(0)
    norms = rng.normal(6.0, 0.15, size=200).clip(5.0, 7.0)
    st = AutoClipPercentileState(percentile=10.0, window=1000, warmup_steps=10, fallback_clip=0.5)
    for g in norms:
        st.observe(float(g))
    t = st.threshold()
    assert 5.0 < t < 6.2  # near the p10 of the ~N(6,0.15) sample, NOT 0.5
    clipped_frac = float(np.mean(norms > t))
    assert clipped_frac < 0.95  # off the 1.0 saturation


def test_ring_wraparound_keeps_only_window_most_recent():
    st = AutoClipPercentileState(percentile=50.0, window=4, warmup_steps=1, fallback_clip=1.0)
    for g in (100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0):
        st.observe(g)
    # window holds only the four 1.0s
    assert st.threshold() == pytest.approx(1.0)
    assert st.filled == 4


def test_nonfinite_and_negative_norms_are_skipped():
    st = AutoClipPercentileState(percentile=10.0, window=10, warmup_steps=2, fallback_clip=0.5)
    assert not st.observe(float("nan"))
    assert not st.observe(float("inf"))
    assert not st.observe(-1.0)
    assert st.filled == 0
    assert st.threshold() == 0.5  # still warmup (no observations recorded)


def test_resume_roundtrip_is_bit_faithful():
    st = AutoClipPercentileState(percentile=10.0, window=8, warmup_steps=3, fallback_clip=0.5)
    for g in (6.0, 5.5, 7.0, 6.5, 17.5, 5.9):
        st.observe(g)
    cfg = st.state_arrays(AUTOCLIP_RESUME_PREFIX)
    fresh = AutoClipPercentileState(percentile=10.0, window=8, warmup_steps=3, fallback_clip=0.5)
    assert fresh.restore_from_cfg(AUTOCLIP_RESUME_PREFIX, cfg)
    assert fresh.threshold() == st.threshold()
    assert fresh.filled == st.filled
    # continued observation stays identical on both
    st.observe(6.1)
    fresh.observe(6.1)
    assert fresh.threshold() == st.threshold()


def test_fresh_state_persists_nothing_and_legacy_cfg_restores_fresh():
    st = AutoClipPercentileState()
    assert st.state_arrays(AUTOCLIP_RESUME_PREFIX) == {}
    assert not st.restore_from_cfg(AUTOCLIP_RESUME_PREFIX, {})  # legacy sidecar => fresh


def test_restore_with_changed_window_refuses():
    st = AutoClipPercentileState(window=8)
    st.observe(1.0)
    cfg = st.state_arrays("p_")
    other = AutoClipPercentileState(window=16)
    with pytest.raises(ValueError, match="window"):
        other.restore_from_cfg("p_", cfg)


@pytest.mark.parametrize("kwargs", [
    {"percentile": 0.0}, {"percentile": 100.0}, {"window": 0},
    {"warmup_steps": 0}, {"fallback_clip": 0.0}, {"fallback_clip": float("nan")},
])
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        AutoClipPercentileState(**kwargs)


def test_controller_groups_are_lazy_and_independent():
    ctl = AutoClipController(percentile=50.0, window=4, warmup_steps=1, fallback_clip=0.5)
    a = ctl.group("film")
    b = ctl.group("out_tex")
    assert a is ctl.group("film") and a is not b
    a.observe(2.0)
    b.observe(20.0)
    assert a.threshold() == pytest.approx(2.0)
    assert b.threshold() == pytest.approx(20.0)


def test_controller_resume_roundtrip_covers_global_and_groups():
    ctl = AutoClipController(percentile=50.0, window=4, warmup_steps=1, fallback_clip=0.5)
    ctl.global_state.observe(6.0)
    ctl.group("film").observe(2.0)
    ctl.group("eik").observe(9.0)
    cfg = ctl.state_arrays(AUTOCLIP_RESUME_PREFIX)
    fresh = AutoClipController(percentile=50.0, window=4, warmup_steps=1, fallback_clip=0.5)
    assert fresh.restore_from_cfg(AUTOCLIP_RESUME_PREFIX, cfg)
    assert fresh.global_state.threshold() == ctl.global_state.threshold()
    assert set(fresh.group_states) == {"film", "eik"}
    assert fresh.group("film").threshold() == pytest.approx(2.0)


def test_controller_row_reports_and_resets_epoch_stats():
    ctl = AutoClipController(percentile=50.0, window=4, warmup_steps=1, fallback_clip=0.5)
    ctl.global_state.observe(6.0)
    ctl.note_step(0.5, True)
    ctl.note_step(6.0, False)
    row = ctl.row(epoch=7)
    assert row["stage"] == "grad_clip_autoclip"
    assert row["epoch"] == 7 and row["steps"] == 2
    assert row["frac_clipped"] == pytest.approx(0.5)
    assert row["clip_t_min"] == pytest.approx(0.5)
    assert row["clip_t_max"] == pytest.approx(6.0)
    assert row["score_neutral"] is True
    # reset semantics
    row2 = ctl.row(epoch=8)
    assert row2["steps"] == 0 and row2["clip_t_mean"] is None


def test_law_module_math_matches_state_machine():
    from tac.canonical_equations.autoclip_percentile_grad_clip_20260715 import autoclip_threshold

    norms = [6.0, 5.9, 6.2, 6.1, 17.5, 5.8]
    st = AutoClipPercentileState(percentile=10.0, window=100, warmup_steps=1, fallback_clip=0.5)
    for g in norms:
        st.observe(g)
    assert st.threshold() == autoclip_threshold(np.asarray(norms), 10.0)


def test_lawref_evaluator_registered_for_autoclip_equation():
    from tac.canonical_equations.evaluators import LAWREF_BUILTIN_EVALUATORS

    assert "autoclip_percentile_threshold_v1" in LAWREF_BUILTIN_EVALUATORS
