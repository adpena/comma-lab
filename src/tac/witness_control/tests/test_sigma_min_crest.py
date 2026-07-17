# SPDX-License-Identifier: MIT
"""Fire-on-crest pose-gate option (SPEC_v10 §13.2; arm B 2026-07-17).

Behavior tests for ``sigma_min_plateau.evaluate_crest`` / ``SigmaMinPlateauDetector(mode='crest')``
+ the crest canary suite + the DSL ``PoseFinishConditioningGate(engage_mode=...)`` amendment + the
trainer ``--pose-finish-engage-on sigma_min_crest`` choice. Incumbent plateau semantics are
regression-locked (default mode unchanged).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tac.witness_control import sigma_min_plateau as smp

_REPO = Path(__file__).resolve().parents[4]
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _rising(n=24, ep_step=4, start=0.001, slope=0.15):
    """Clean multiplicative-rise series (the live-c2 +15%/ep signature, noise-free)."""
    return [(i * ep_step, start * (1.0 + slope) ** i) for i in range(n)]


# ── event semantics ────────────────────────────────────────────────────────── #
def test_crest_fires_on_rise_then_decline():
    v = smp.run_detector_on_series(smp.synthetic_rise_then_decline_series(), mode=smp.MODE_CREST)
    assert v.classification == smp.CREST_FIRED
    assert v.fired()


def test_crest_fires_on_rise_then_flat():
    """A plateau after a genuine rise is ALSO a crest (slope stopped being positive)."""
    v = smp.run_detector_on_series(smp.synthetic_plateau_series(), mode=smp.MODE_CREST)
    assert v.classification == smp.CREST_FIRED


def test_crest_never_fires_on_monotone_rising():
    """The live c2 signature (σ_min still climbing) must NOT crest-fire."""
    v = smp.run_detector_on_series(_rising(), mode=smp.MODE_CREST)
    assert v.classification == smp.NOT_CRESTED
    assert not v.fired()


def test_crest_never_fires_on_flat_from_start():
    """Distinguishing semantics vs plateau: no resolved RISING phase => no crest, ever."""
    flat = [(i * 4, 0.05) for i in range(24)]
    assert smp.run_detector_on_series(flat, mode=smp.MODE_CREST).classification == smp.NOT_CRESTED
    # regression-lock the incumbent: the plateau mode DOES fire on flat-from-start.
    assert smp.run_detector_on_series(flat, mode=smp.MODE_PLATEAU).classification == smp.PLATEAU_FIRED


def test_crest_insufficient_data_below_min_points():
    cfg = smp.SigmaMinPlateauConfig()
    series = smp.synthetic_rise_then_decline_series()[: cfg.min_points]  # < min_points + 1
    v = smp.run_detector_on_series(series, cfg, mode=smp.MODE_CREST)
    assert v.classification == smp.INSUFFICIENT_DATA


def test_crest_hysteresis_not_met_right_after_peak():
    """One post-peak point is not enough: the trailing windows still read rising."""
    series = smp.synthetic_rise_then_decline_series(n_rise=10, n_decline=1)
    v = smp.run_detector_on_series(series, mode=smp.MODE_CREST)
    assert v.classification in (smp.NOT_CRESTED, smp.DEGENERATE_GUARD_TRIPPED)
    assert not v.fired()


def test_crest_degenerate_guard_on_unresolvable_slope():
    """A near-zero slope whose stderr cannot rule out rising => DEGENERATE => banked R1."""
    import numpy as np

    rng = np.random.default_rng(7)
    series = [(i * 4, 0.05 * (1.0 + 0.5 * float(rng.standard_normal()))) for i in range(24)]
    v = smp.run_detector_on_series(series, mode=smp.MODE_CREST)
    if v.classification == smp.DEGENERATE_GUARD_TRIPPED:
        assert v.should_ship_banked_r1()
    else:  # heavy noise may also fail the rising-phase resolution => NOT_CRESTED — never a fire
        assert not v.fired()


# ── detector state machine ─────────────────────────────────────────────────── #
def test_crest_latch_is_monotone():
    det = smp.SigmaMinPlateauDetector(smp.SigmaMinPlateauConfig(), mode=smp.MODE_CREST)
    for ep, s in smp.synthetic_rise_then_decline_series():
        det.observe(ep, s)
        det.latch_if_fired(ep)
    assert det.fired()
    fired_at = det.fired_epoch
    # later rising points cannot un-latch
    last_ep = det._eps[-1]
    for k in range(1, 6):
        det.observe(last_ep + 4 * k, 0.06 * (1.1 ** k))
    v = det.verdict()
    assert v.classification == smp.CREST_FIRED and v.latched_fired_epoch == fired_at


def test_crest_detector_resume_roundtrip():
    cfg = smp.SigmaMinPlateauConfig()
    det = smp.SigmaMinPlateauDetector(cfg, mode=smp.MODE_CREST)
    for ep, s in smp.synthetic_rise_then_decline_series():
        det.observe(ep, s)
        det.latch_if_fired(ep)
    state = det.state_arrays(smp.RESUME_PREFIX)
    det2 = smp.SigmaMinPlateauDetector(cfg, mode=smp.MODE_CREST)
    assert det2.restore_from_cfg(smp.RESUME_PREFIX, state)
    assert det2.fired() and det2.fired_epoch == det.fired_epoch
    assert det2.verdict().classification == smp.CREST_FIRED


def test_detector_mode_validated():
    with pytest.raises(ValueError, match="mode must be"):
        smp.SigmaMinPlateauDetector(smp.SigmaMinPlateauConfig(), mode="not_a_mode")


def test_detector_default_mode_is_plateau_incumbent():
    det = smp.SigmaMinPlateauDetector(smp.SigmaMinPlateauConfig())
    assert det.mode == smp.MODE_PLATEAU
    for ep, s in smp.synthetic_plateau_series():
        det.observe(ep, s)
    assert det.verdict().classification == smp.PLATEAU_FIRED


# ── canaries ───────────────────────────────────────────────────────────────── #
def test_crest_canary_suite_passes_default_cfg():
    c = smp.crest_canary_suite()
    assert c.passed
    assert not c.negative_fired and c.positive_fired


def test_plateau_canary_unchanged():
    assert smp.canary_suite().passed


# ── wiring surfaces ────────────────────────────────────────────────────────── #
def test_dsl_gate_engage_mode_crest_emits_flag():
    from tac.witness_dsl.curriculum_dsl import PoseFinishConditioningGate

    lever = PoseFinishConditioningGate(backstop_epoch=1000, engage_mode="sigma_min_crest")
    assert lever.overrides["--pose-finish-engage-on"] == "sigma_min_crest"


def test_dsl_gate_default_engage_mode_unchanged():
    from tac.witness_dsl.curriculum_dsl import PoseFinishConditioningGate

    lever = PoseFinishConditioningGate()
    assert lever.overrides["--pose-finish-engage-on"] == "sigma_min_plateau"


def test_dsl_gate_rejects_unknown_engage_mode():
    from tac.witness_dsl.curriculum_dsl import PoseFinishConditioningGate

    with pytest.raises(ValueError, match="engage_mode must be"):
        PoseFinishConditioningGate(engage_mode="muon")  # incumbent is the ABSENCE of this lever


def test_trainer_argparse_declares_crest_choice():
    src = _TRAINER.read_text(errors="ignore")
    m = re.search(r"add_argument\(\"--pose-finish-engage-on\".*?choices=\[([^\]]*)\]", src, re.S)
    assert m is not None and "sigma_min_crest" in m.group(1)
    # the engage/alarm paths dispatch on the sigma-mode tuple, not a single string equality
    assert "_pose_gate_sigma_modes" in src


# ── LIVE MEASURED ANCHOR [live c2 run 20260717T113932Z, advisory] ──────────── #
_LIVE_C2_SIGMA_MIN = [
    # smoothed σ_min gate-row series ep786→810 (READ-ONLY run.log): rise to the ~ep802 PEAK
    # (0.0097) then decisive decline (latest rel-slope −0.1246/ep, stderr 0.0187 ≈ 6.6σ).
    (786, 0.0010), (794, 0.0064), (798, 0.0064), (802, 0.0097), (806, 0.0057), (810, 0.0034),
]
_LIVE_C2_EXTRAP = [
    # labeled EXTRAPOLATION (not measured): the measured decline rate continued 3 more gate rows,
    # only to give the hysteresis machinery its confirmation window.
    (814, 0.0021), (818, 0.0013), (822, 0.0008),
]


def test_live_c2_crest_hysteresis_discipline_then_fire():
    """On the 6 MEASURED points the crest is real but hysteresis (3 windows) is not yet met —
    the detector correctly waits; with the measured decline continued (extrapolated, labeled),
    it FIRES. The plateau detector NEVER fires on either (a crest-then-decline trajectory never
    presents 'flat' — measured live: consecutive_flat 0/3 on the run)."""
    v_measured = smp.run_detector_on_series(_LIVE_C2_SIGMA_MIN, mode=smp.MODE_CREST)
    assert not v_measured.fired()          # hysteresis discipline: no premature fire
    full = _LIVE_C2_SIGMA_MIN + _LIVE_C2_EXTRAP
    v_full = smp.run_detector_on_series(full, mode=smp.MODE_CREST)
    assert v_full.classification == smp.CREST_FIRED
    # the plateau mode never fires on the crest-then-decline shape (the live F4 signature):
    assert not smp.run_detector_on_series(full, mode=smp.MODE_PLATEAU).fired()
