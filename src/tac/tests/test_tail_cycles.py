"""Tests for the TAIL_k warm-restart controller (tac.witness_control.tail_cycles).

Covers the sealed crucible laws (req L / v6 §2.2e / §row-9): τ_k halving+live-m_q, LR ∝ τ_k,
dwell floor, cycle-floor cap, per-cycle powerlaw_meat exit, PowerPlay stop, k_max fail-safe,
cycle arithmetic vs the sealed budget, and determinism.
"""
from __future__ import annotations

import math

import pytest

from tac.witness_control.tail_cycles import (
    TailController,
    TailCycleConfig,
    lr_for_tau,
    next_tau,
    tau_star_from_mq,
)

_LN5 = math.log(5.0)


# ----------------------------------------------------------------------------- laws
def test_tau_star_from_mq_is_mq_over_ln5():
    assert tau_star_from_mq(0.85) == pytest.approx(0.85 / _LN5)
    assert tau_star_from_mq(0.10) == pytest.approx(0.10 / _LN5)


def test_next_tau_halving_fallback_and_clamp():
    cfg = TailCycleConfig(k_max=5, tau_halving=0.5, tau_end=0.31)
    # halving fallback (live_mq None)
    assert next_tau(0.62, cfg, None) == pytest.approx(0.31)
    # clamp: never below tau_end even when halving would go lower
    assert next_tau(0.40, cfg, None) == pytest.approx(0.31)   # 0.20 -> clamped to 0.31
    assert next_tau(0.31, cfg, None) == pytest.approx(0.31)


def test_next_tau_live_mq_max_branch():
    cfg = TailCycleConfig(k_max=5, tau_halving=0.5, tau_end=0.05)
    # τ_k = max(τ_{k-1}·halving, m_q/ln5): live target 0.528 dominates the halved 0.31
    assert next_tau(0.62, cfg, live_mq=0.85) == pytest.approx(max(0.31, 0.85 / _LN5))
    # tiny m_q -> halving wins
    assert next_tau(0.62, cfg, live_mq=0.05) == pytest.approx(0.31)


def test_lr_for_tau_is_proportional():
    assert lr_for_tau(0.31, 0.62, 1e-3) == pytest.approx(0.5e-3)
    assert lr_for_tau(0.31, 0.62, 1e-3, coeff=2.0) == pytest.approx(1e-3)
    # degenerate tau_ref -> lr_ref*coeff (no division blow-up)
    assert lr_for_tau(0.31, 0.0, 1e-3, coeff=1.0) == pytest.approx(1e-3)


# ----------------------------------------------------------------------------- config validate
def test_config_validate_rejects_bad_values():
    assert TailCycleConfig(k_max=0).validate()                       # k_max < 1
    assert TailCycleConfig(k_max=5, dwell_min=500,
                           cycle_floor_epochs=387.09).validate()     # dwell > floor
    assert TailCycleConfig(k_max=5, tau_halving=1.5).validate()      # halving out of (0,1)
    assert TailCycleConfig(k_max=5, min_points=2).validate()         # < 4 for two tail models
    assert TailCycleConfig(k_max=5).validate() == []                 # sealed defaults are valid


@pytest.mark.parametrize(
    "invalid",
    [
        -1e-12,
        float("nan"),
        float("inf"),
        -float("inf"),
        "0.0",
        True,
        False,
        10**400,
        object(),
    ],
)
def test_stop_marginal_s_rejects_unsafe_values_at_both_boundaries(invalid):
    from tac.witness_dsl.curriculum_dsl import TailCycles

    with pytest.raises(ValueError, match=r"stop_marginal_s.*>= 0"):
        TailCycles(stop_marginal_s=invalid)
    cfg = TailCycleConfig(k_max=1, stop_marginal_s=invalid)
    assert any(
        "stop_marginal_s" in problem and ">= 0" in problem
        for problem in cfg.validate()
    )
    with pytest.raises(ValueError, match=r"stop_marginal_s.*>= 0"):
        TailController(cfg, tau_ref=0.62, lr_ref=1e-3, tau0=0.62)


def test_zero_stop_marginal_s_is_admitted_at_both_boundaries():
    from tac.witness_dsl.curriculum_dsl import TailCycles

    emitted = TailCycles(stop_marginal_s=0.0).overrides["--tail-stop-marginal-s"]
    assert emitted == 0.0 and type(emitted) is float
    cfg = TailCycleConfig(k_max=1, stop_marginal_s=0.0)
    assert cfg.validate() == []
    assert type(cfg.stop_marginal_s) is float


def test_zero_floor_continues_nonnegative_cycle_and_stops_negative_cycle():
    cfg = TailCycleConfig(
        k_max=2,
        cycle_floor_epochs=5,
        dwell_min=0,
        stop_marginal_s=0.0,
        min_points=4,
    )
    nonnegative = TailController(cfg, tau_ref=0.62, lr_ref=1e-3, tau0=0.62)
    nonnegative.step(0, [(0, 0.005)])
    zero_benefit = nonnegative.step(4, [(0, 0.005), (4, 0.005)])
    assert zero_benefit.begin_cycle is True
    assert zero_benefit.stop is False

    negative = TailController(cfg, tau_ref=0.62, lr_ref=1e-3, tau0=0.62)
    negative.step(0, [(0, 0.005)])
    harm = negative.step(4, [(0, 0.005), (4, 0.0051)])
    assert harm.marginal < 0.0
    assert harm.stop is True
    assert "powerplay stop" in harm.reason


def test_controller_raises_on_invalid_config():
    with pytest.raises(ValueError):
        TailController(TailCycleConfig(k_max=0), tau_ref=0.62, lr_ref=1e-3, tau0=0.62)


# ----------------------------------------------------------------------------- controller helper
def _run(ctrl, epochs, dseg_of, *, live_mq=None):
    """Feed the controller a verdict trace (dseg_of(ep) with cadence via None-returns) and
    return the list of TailStep at each epoch. dseg_of returns a float to record a verdict row
    at that epoch, or None (no verdict this epoch — cadence gap)."""
    rows: list[tuple[int, float]] = []
    steps = []
    for ep in epochs:
        st = ctrl.step(ep, list(rows), live_mq=live_mq)
        steps.append(st)
        d = dseg_of(ep)
        if d is not None:
            rows.append((ep, float(d)))
    return steps


# ----------------------------------------------------------------------------- cycle behavior
def test_first_cycle_begins_and_tau_halves_across_cycles():
    cfg = TailCycleConfig(k_max=5, cycle_floor_epochs=50, dwell_min=20, tau_halving=0.5,
                          tau_end=0.0, stop_marginal_s=0.0, min_points=8)  # stop on harm only
    ctrl = TailController(cfg, tau_ref=0.8, lr_ref=1e-3, tau0=0.8)
    # descending trajectory (positive marginal above the zero floor => never powerplay), every 5 ep
    def dseg(ep):
        return (0.02 - 2e-4 * (ep - 1000)) if ep % 5 == 0 else None
    steps = _run(ctrl, range(1000, 1260), dseg)
    begins = [s for s in steps if s.begin_cycle]
    # cycle floor 50 over a 260-ep span => ~5 cycles, capped at k_max 5
    assert begins[0].cycle_k == 1 and begins[0].reason == "first cycle"
    assert begins[0].stage_tag == "stageTail1"
    # τ halves each new cycle: 0.4, 0.2, 0.1, ...
    taus = [round(b.tau, 4) for b in begins[:3]]
    assert taus == [0.4, 0.2, 0.1]
    # LR ∝ τ_k relative to tau_ref 0.8, lr_ref 1e-3
    assert begins[0].lr == pytest.approx(1e-3 * 0.4 / 0.8)


def test_cycle_floor_cap_forces_exit_at_floor_length():
    cfg = TailCycleConfig(k_max=9, cycle_floor_epochs=40, dwell_min=15, tau_halving=0.5,
                          tau_end=0.0, stop_marginal_s=0.0, min_points=8)
    ctrl = TailController(cfg, tau_ref=0.8, lr_ref=1e-3, tau0=0.8)
    def dseg(ep):
        return (0.02 - 2e-4 * (ep - 1000)) if ep % 5 == 0 else None
    steps = _run(ctrl, range(1000, 1090), dseg)
    begin_eps = [1000 + i for i, s in enumerate(steps) if s.begin_cycle]
    # cycle 1 at ep1000; cycle 2 forced by the floor cap at length>=40 => ep1039 (1000+40-1)
    assert begin_eps[0] == 1000
    assert begin_eps[1] == 1039
    assert steps[begin_eps[1] - 1000].reason == "cycle_floor_cap"


def test_dwell_gate_blocks_meat_exit_before_dwell_min():
    # flat trajectory would meat-exhaust, but dwell_min must gate it: no exit before length>=dwell
    cfg = TailCycleConfig(k_max=5, cycle_floor_epochs=500, dwell_min=100, tau_halving=0.5,
                          tau_end=0.0, stop_marginal_s=1e-4, min_points=8)
    ctrl = TailController(cfg, tau_ref=0.8, lr_ref=1e-3, tau0=0.8)
    def dseg(ep):
        return 0.005 if ep % 5 == 0 else None   # flat
    steps = _run(ctrl, range(1000, 1080), dseg)   # 80 ep < dwell 100
    # only the first cycle ever begins; no exit/stop within the dwell window
    assert sum(s.begin_cycle for s in steps) == 1
    assert not any(s.stop for s in steps)


def test_meat_exit_and_powerplay_stop_on_flat_trajectory():
    # a flat trajectory past dwell => powerlaw_meat exhausted => exit; marginal ΔS/ep ≈ 0 < floor
    # => PowerPlay stop on the FIRST cycle exit.
    cfg = TailCycleConfig(k_max=5, cycle_floor_epochs=500, dwell_min=40, tau_halving=0.5,
                          tau_end=0.0, stop_marginal_s=1e-4, meat_floor=1e-4, min_points=8)
    ctrl = TailController(cfg, tau_ref=0.8, lr_ref=1e-3, tau0=0.8)
    def dseg(ep):
        return 0.005 if ep % 5 == 0 else None
    steps = _run(ctrl, range(1000, 1120), dseg)
    stops = [s for s in steps if s.stop]
    assert stops, "flat trajectory past dwell must PowerPlay-stop"
    assert "powerplay" in stops[0].reason and "meat_exit" in stops[0].reason
    # once stopped, further steps stay stopped (latched)
    assert steps[-1].stop


def test_k_max_fail_safe_stop_on_descending_trajectory():
    # descending (marginal above floor => never powerplay) so cycles exit via the floor cap;
    # after k_max cycles the req-B fail-safe stops it.
    cfg = TailCycleConfig(k_max=3, cycle_floor_epochs=30, dwell_min=10, tau_halving=0.5,
                          tau_end=0.0, stop_marginal_s=1e-6, min_points=8)
    ctrl = TailController(cfg, tau_ref=0.8, lr_ref=1e-3, tau0=0.8)
    def dseg(ep):
        return (0.02 - 5e-4 * (ep - 1000)) if ep % 5 == 0 else None
    steps = _run(ctrl, range(1000, 1200), dseg)
    stops = [s for s in steps if s.stop]
    assert stops, "must hit the k_max fail-safe"
    assert "k_max 3" in stops[0].reason
    assert stops[0].cycle_k == 3   # stopped at the 3rd cycle's exit


def test_cycle_arithmetic_matches_sealed_budget():
    # sealed: k_max net = floor((budget − FIN)/cycle_floor). budget window 2350 ep after ep650,
    # minus 250 FIN min-stage = 2100; /387.09 => 5.  (draft §2.2g(b)/v6.2 MINOR-1)
    cycle_floor = 387.09
    assert math.floor((2350 - 250) / cycle_floor) == 5
    assert math.floor(2350 / cycle_floor) == 6   # gross


def test_determinism_same_inputs_same_decisions():
    def build():
        cfg = TailCycleConfig(k_max=4, cycle_floor_epochs=40, dwell_min=15, tau_halving=0.5,
                              tau_end=0.0, stop_marginal_s=0.0, min_points=8)
        return TailController(cfg, tau_ref=0.8, lr_ref=1e-3, tau0=0.8)
    def dseg(ep):
        return (0.02 - 2e-4 * (ep - 1000)) if ep % 5 == 0 else None
    a = [(s.tau, s.lr, s.begin_cycle, s.stop) for s in _run(build(), range(1000, 1120), dseg)]
    b = [(s.tau, s.lr, s.begin_cycle, s.stop) for s in _run(build(), range(1000, 1120), dseg)]
    assert a == b


def test_controller_checkpoint_resume_preserves_cycle_and_next_decision():
    cfg = TailCycleConfig(k_max=4, cycle_floor_epochs=40, dwell_min=15,
                          tau_halving=0.5, tau_end=0.0,
                          stop_marginal_s=0.0, min_points=8)
    first = TailController(cfg, tau_ref=0.8, lr_ref=1e-3, tau0=0.8)
    rows = [(1000, 0.02), (1005, 0.019)]
    first.step(1000, rows[:1])
    state = first.state_dict()
    resumed = TailController(cfg, tau_ref=0.8, lr_ref=1e-3, tau0=0.8)
    resumed.load_state_dict(state)
    assert resumed.state_dict() == state
    resumed_step = resumed.step(1006, rows)
    first_step = first.step(1006, rows)
    assert resumed_step.__dict__ | {"marginal": None} == (
        first_step.__dict__ | {"marginal": None}
    )
    assert math.isnan(resumed_step.marginal) and math.isnan(first_step.marginal)
    drifted = TailController(
        TailCycleConfig(k_max=3, cycle_floor_epochs=40, dwell_min=15,
                        tau_halving=0.5, tau_end=0.0,
                        stop_marginal_s=0.0, min_points=8),
        tau_ref=0.8, lr_ref=1e-3, tau0=0.8,
    )
    with pytest.raises(ValueError, match="typed configuration"):
        drifted.load_state_dict(state)


# ----------------------------------------------------------------------------- S1-R1 rate-aware stop
def _run_rate(ctrl, epochs, dseg_of, bytes_of, *, seed):
    """Feed the controller a verdict trace WITH a coded-bytes trace (S1-R1). ``seed`` = (ep, dseg,
    bytes) pre-loaded so the FIRST cycle has a known start-of-cycle d_seg/bytes (decide-on-previous)."""
    _se, _sd, _sb = seed
    rows: list[tuple[int, float]] = [(_se, float(_sd))]
    brows: list[tuple[int, float]] = [(_se, float(_sb))]
    steps = []
    for ep in epochs:
        st = ctrl.step(ep, list(rows), live_mq=None, byte_rows=list(brows))
        steps.append(st)
        d = dseg_of(ep)
        if d is not None:
            rows.append((ep, float(d)))
            brows.append((ep, float(bytes_of(ep))))
    return steps


def _cfg_rate(stop=1e-3):
    # cycle_floor cap forces exit at length 30 (before meat: <8 verdicts by then); big margins.
    return TailCycleConfig(k_max=9, cycle_floor_epochs=30, dwell_min=10, tau_halving=0.5,
                           tau_end=0.0, stop_marginal_s=stop, min_points=8, meat_floor=1e-4)


def _dseg_drop(ep):
    # 0.010 -> 0.004 over [1000,1025] (verdict every 5); d_seg marginal ~0.02 >> 1e-3 floor.
    if ep % 5 != 0 or ep > 1025:
        return None
    return round(0.010 - 0.0012 * ((ep - 1000) // 5), 6)


def test_rate_stable_bytes_does_not_fire_on_improving_dseg():
    # STABLE bytes: the cycle keeps improving d_seg (~0.02 ΔS/ep) and the rate leg is ~0, so net ==
    # d_seg marginal > floor => NO PowerPlay-stop at the first cycle exit — it begins a 2nd cycle.
    ctrl = TailController(_cfg_rate(), tau_ref=0.8, lr_ref=1e-3, tau0=0.8)
    steps = _run_rate(ctrl, range(1000, 1035), _dseg_drop, lambda ep: 100_000,
                      seed=(999, 0.010, 100_000))
    assert not any(s.stop for s in steps)
    assert max(s.cycle_k for s in steps) >= 2, "improving d_seg with stable bytes should begin cycle 2"


def test_rate_aware_stop_fires_when_bytes_inflate():
    # SAME improving d_seg, but the coded blob INFLATES 100k->1.2M across the cycle. The rate leg
    # (25·Δbytes/B) overwhelms the d_seg improvement => net-ΔS/ep < floor => PowerPlay-stop at cycle 1.
    ctrl = TailController(_cfg_rate(), tau_ref=0.8, lr_ref=1e-3, tau0=0.8)
    steps = _run_rate(ctrl, range(1000, 1035), _dseg_drop,
                      lambda ep: 100_000 + 220_000 * ((ep - 1000) // 5),  # 100k -> 1.2M by ep1025
                      seed=(999, 0.010, 100_000))
    stops = [s for s in steps if s.stop]
    assert stops, "a byte-inflating cycle must trip the rate-aware PowerPlay stop"
    assert stops[0].cycle_k == 1, "should stop at the FIRST cycle exit (rate-aware, not d_seg-blind)"
    assert "rate-aware" in stops[0].reason and "net-ΔS" in stops[0].reason


def test_rate_aware_stop_stamps_the_marginal_numerator():
    # S4-R2: the stamped marginal is the MEASURED net-ΔS/ep NUMERATOR (< the floor), finite.
    ctrl = TailController(_cfg_rate(), tau_ref=0.8, lr_ref=1e-3, tau0=0.8)
    steps = _run_rate(ctrl, range(1000, 1035), _dseg_drop,
                      lambda ep: 100_000 + 220_000 * ((ep - 1000) // 5),
                      seed=(999, 0.010, 100_000))
    stop = next(s for s in steps if s.stop)
    assert math.isfinite(stop.marginal)
    assert stop.marginal < 1e-3                       # below the floor (the reason it stopped)
    assert f"{stop.marginal:.3e}" in stop.reason      # the numerator is stamped in the reason too


def test_byte_rows_none_is_byte_identical_to_pre_s1r1():
    # the OFF/rate-blind path: byte_rows=None must reproduce the exact d_seg-only decisions.
    def build():
        return TailController(_cfg_rate(stop=1e-4), tau_ref=0.8, lr_ref=1e-3, tau0=0.8)
    def dseg(ep):
        return (0.02 - 2e-4 * (ep - 1000)) if ep % 5 == 0 else None
    a = [(s.tau, s.lr, s.begin_cycle, s.stop, s.reason) for s in _run(build(), range(1000, 1120), dseg)]
    # same trajectory driven through step() with byte_rows explicitly None => identical.
    ctrl = build()
    rows: list[tuple[int, float]] = []
    b = []
    for ep in range(1000, 1120):
        st = ctrl.step(ep, list(rows), live_mq=None, byte_rows=None)
        b.append((st.tau, st.lr, st.begin_cycle, st.stop, st.reason))
        d = dseg(ep)
        if d is not None:
            rows.append((ep, float(d)))
    assert a == b
