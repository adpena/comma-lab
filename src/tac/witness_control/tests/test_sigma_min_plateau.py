"""Tests for the pose-finish CONDITIONING gate — rolling-slope σ_min plateau detector (owed-1, A-1 FIX).

Covers the SEALED PRIMARY criterion (SYNTHESIS_v3_v752 §A.4): fire on a de-noised plateau, do NOT fire on
a rising series (the P4-M1 negative control), guard-trip on a degenerate/noisy series, hysteresis,
monotone latch, the $0 canary suite (negative-must-not-fire + synthetic-positive-must-fire), the σ*
advisory-only sideband, and the Resumable round-trip.
"""
from __future__ import annotations

import numpy as np

from tac.witness_control.sigma_min_plateau import (
    DEGENERATE_GUARD_TRIPPED,
    INSUFFICIENT_DATA,
    NOT_PLATEAUED,
    PLATEAU_FIRED,
    RESUME_PREFIX,
    CanaryResult,
    SigmaMinPlateauConfig,
    SigmaMinPlateauDetector,
    canary_suite,
    ema_smooth,
    evaluate_plateau,
    run_detector_on_series,
    sigma_star_advisory,
    synthetic_plateau_series,
    synthetic_rising_series,
)


def _cfg(**kw) -> SigmaMinPlateauConfig:
    return SigmaMinPlateauConfig(**kw)


# ── config validation ──
def test_config_defaults_valid_and_derived():
    c = _cfg()
    assert c.validate() == []
    # DERIVED: all three window knobs descend from ⌈W_settle⌉ = 3.
    assert c.settle_window == 3 and c.hysteresis == 3 and c.ema_span == 3
    assert c.min_points == 5  # settle_window + hysteresis - 1
    assert c.flat_rel_band > 0.0


def test_config_rejects_nonsense():
    assert _cfg(ema_span=0).validate()
    assert _cfg(flat_rel_band=0.0).validate()
    assert _cfg(settle_window=2).validate()   # < MIN_ROWS_FOR_SLOPE (no stderr df)
    assert _cfg(hysteresis=0).validate()


# ── ema_smooth ──
def test_ema_smooth_shape_and_trend_preserved():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    sm = ema_smooth(vals, span=3)
    assert len(sm) == len(vals)
    assert sm[0] == 1.0                      # first point seeds the shadow (no look-ahead)
    assert all(b >= a for a, b in zip(sm, sm[1:]))   # a monotone-rising input stays rising when smoothed


# ── PRIMARY criterion: FIRE on a clean plateau ──
def test_fires_on_clean_plateau():
    v = run_detector_on_series(synthetic_plateau_series(), _cfg())
    assert v.classification == PLATEAU_FIRED, v.reason
    assert v.fired() and not v.should_ship_banked_r1()
    assert v.consecutive_flat >= v.hysteresis


# ── NEGATIVE control: do NOT fire while σ_min is RISING (P4-M1 stopped-run signature) ──
def test_does_not_fire_on_rising_sigma_min():
    v = run_detector_on_series(synthetic_rising_series(n=31), _cfg())
    assert not v.fired(), v.reason
    # a rising σ_min is either 'still trending' or too-noisy-to-trust — never a plateau fire.
    assert v.classification in (NOT_PLATEAUED, DEGENERATE_GUARD_TRIPPED)


# ── INSUFFICIENT data ──
def test_insufficient_data_below_min_points():
    # 4 points < min_points 5
    series = [(0, 0.05), (4, 0.05), (8, 0.05), (12, 0.05)]
    v = run_detector_on_series(series, _cfg())
    assert v.classification == INSUFFICIENT_DATA
    assert not v.fired()


# ── NOISE / FIT-QUALITY GUARD: a flat-on-average but too-noisy series does NOT fire (ships banked R1) ──
def test_guard_trips_on_flat_but_noisy_series():
    rng = np.random.default_rng(7)
    # mean-0.05 series with LARGE relative scatter that survives EMA smoothing → rel-stderr > band.
    series = [(i * 4, float(0.05 * (1.0 + 0.5 * rng.standard_normal()))) for i in range(20)]
    v = run_detector_on_series(series, _cfg())
    # either the guard trips (degenerate) or it reads not-plateaued — the invariant is NEVER a false fire.
    assert not v.fired()
    if v.classification == DEGENERATE_GUARD_TRIPPED:
        assert v.should_ship_banked_r1()


def test_perfectly_flat_series_fires_guard_passes():
    series = [(i * 4, 0.05) for i in range(8)]   # exact plateau: rel-stderr 0 → guard passes
    v = run_detector_on_series(series, _cfg())
    assert v.classification == PLATEAU_FIRED
    assert v.latest_rel_stderr_per_ep == 0.0 or v.latest_rel_stderr_per_ep is not None


# ── HYSTERESIS: fewer than `hysteresis` consecutive flat windows does NOT fire ──
def test_hysteresis_requires_consecutive_flat_windows():
    # rising then a SHORT flat tail: fewer than hysteresis flat windows at the end.
    rising = [(0, 0.01), (4, 0.02), (8, 0.03), (12, 0.04)]
    short_flat = [(16, 0.05), (20, 0.05)]        # only ~1-2 flat windows
    v = run_detector_on_series(rising + short_flat, _cfg(hysteresis=3))
    assert not v.fired()
    # with a LONGER flat tail it DOES fire (enough consecutive flat windows to out-run the EMA lag).
    long_flat = [(16 + 4 * i, 0.05) for i in range(12)]
    v2 = run_detector_on_series(rising + long_flat, _cfg(hysteresis=3))
    assert v2.fired(), v2.reason


# ── MONOTONE LATCH: once fired, stays fired even if σ_min later rises ──
def test_latch_is_monotone():
    det = SigmaMinPlateauDetector(_cfg())
    for ep, s in synthetic_plateau_series():
        det.observe(ep, s)
    assert det.latch_if_fired(1000) is True
    assert det.fired() and det.fired_epoch == 1000
    # feed a big RISE after the latch — engagement must NOT un-fire.
    for i in range(6):
        det.observe(2000 + i * 4, 0.2 + 0.05 * i)
    assert det.fired()
    assert det.verdict().classification == PLATEAU_FIRED
    assert det.latch_if_fired(3000) is False     # already latched


def test_latch_never_fires_on_rising():
    det = SigmaMinPlateauDetector(_cfg())
    for ep, s in synthetic_rising_series(n=31):
        det.observe(ep, s)
    assert det.latch_if_fired(500) is False
    assert not det.fired()


# ── observe idempotency (resume double-count guard) ──
def test_observe_skips_non_increasing_epochs_and_nonfinite():
    det = SigmaMinPlateauDetector(_cfg())
    det.observe(10, 0.05)
    det.observe(10, 0.06)     # same epoch → skipped
    det.observe(8, 0.06)      # backwards → skipped
    det.observe(20, float("nan"))   # non-finite → skipped
    det.observe(20, 0.05)
    assert det.n_points == 2


# ── σ* ADVISORY ONLY ──
def test_sigma_star_advisory_only_never_gates():
    # unreachable value computes when λ_min known; None when unknown (annulus probe off launch path).
    s = sigma_star_advisory(c_pose_grad_cap=25.0, delta_seg=0.5, lambda_min_f=0.25)
    assert s is not None and s > 14.0        # RED-TEAM measured ≥ 14.14
    assert sigma_star_advisory(25.0, 0.5, None) is None
    # it appears in the verdict as a sideband but does NOT change the firing classification.
    det = SigmaMinPlateauDetector(_cfg(), lambda_min_f=0.25)
    for ep, sm in synthetic_plateau_series():
        det.observe(ep, sm)
    v = det.verdict()
    assert v.sigma_star_advisory is not None and v.fired()   # fires despite σ* being unreachable


# ── $0 CANARY SUITE ──
def test_canary_suite_passes_default():
    res = canary_suite(_cfg())
    assert isinstance(res, CanaryResult)
    assert res.passed, res.reason
    assert res.negative_fired is False and res.positive_fired is True


# ── RESUMABLE round-trip ──
def test_resume_round_trip_series_and_latch():
    det = SigmaMinPlateauDetector(_cfg())
    for ep, s in synthetic_plateau_series():
        det.observe(ep, s)
    det.latch_if_fired(1000)
    arrays = det.state_arrays(RESUME_PREFIX)
    assert (RESUME_PREFIX + "eps") in arrays and (RESUME_PREFIX + "fired_epoch") in arrays
    # simulate the sidecar cfg (parsed arrays) and restore into a FRESH detector.
    cfg_sidecar = {k: np.asarray(v) for k, v in arrays.items()}
    det2 = SigmaMinPlateauDetector(_cfg())
    assert det2.restore_from_cfg(RESUME_PREFIX, cfg_sidecar) is True
    assert det2.n_points == det.n_points
    assert det2.fired() and det2.fired_epoch == 1000
    assert det2.verdict().classification == PLATEAU_FIRED


def test_resume_empty_detector_writes_nothing():
    det = SigmaMinPlateauDetector(_cfg())
    assert det.state_arrays(RESUME_PREFIX) == {}      # byte-identical: no keys for an empty, un-fired gate
    assert det.restore_from_cfg(RESUME_PREFIX, {}) is False   # missing keys → fresh


def test_resume_no_double_count_after_restore():
    det = SigmaMinPlateauDetector(_cfg())
    for ep, s in [(0, 0.05), (4, 0.05), (8, 0.05)]:
        det.observe(ep, s)
    arrays = {k: np.asarray(v) for k, v in det.state_arrays(RESUME_PREFIX).items()}
    det2 = SigmaMinPlateauDetector(_cfg())
    det2.restore_from_cfg(RESUME_PREFIX, arrays)
    det2.observe(8, 0.06)     # a re-run of the last epoch after resume → must be skipped
    det2.observe(12, 0.05)
    assert det2.n_points == 4      # 3 restored + 1 new (the re-run at ep8 skipped)


# ── evaluate_plateau latched short-circuit ──
def test_evaluate_plateau_latched_short_circuits():
    v = evaluate_plateau([0, 4, 8], [0.9, 0.1, 0.5], _cfg(), latched_fired_epoch=42)
    assert v.classification == PLATEAU_FIRED and v.latched_fired_epoch == 42


# ── RESUME through the ACTUAL ResumeRegistry (the trainer's persistence path) ──
def test_resume_through_registry_round_trip():
    from tac.witness_control.resume_registry import ResumeRegistry

    det = SigmaMinPlateauDetector(_cfg())
    for ep, s in synthetic_plateau_series():
        det.observe(ep, s)
    det.latch_if_fired(1000)

    reg = ResumeRegistry()
    reg.register("pose_finish_conditioning_gate", RESUME_PREFIX, det)
    sidecar = reg.state_arrays()            # what the trainer writes into the checkpoint
    assert any(k.startswith(RESUME_PREFIX) for k in sidecar)

    det2 = SigmaMinPlateauDetector(_cfg())
    reg2 = ResumeRegistry()
    reg2.register("pose_finish_conditioning_gate", RESUME_PREFIX, det2)
    report = reg2.restore({k: np.asarray(v) for k, v in sidecar.items()})
    assert report.restored.get("pose_finish_conditioning_gate") is True
    assert det2.fired() and det2.fired_epoch == 1000 and det2.n_points == det.n_points
