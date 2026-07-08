"""Tests for the S6-R4 self-paced τ-advance octave ladder (--tau-advance-mode event).

Covers (per the build charter): default-OFF byte-identity · octave-ladder equivalence (event-mode
instant-fire reuses the geometric clock ladder VALUES) · sensor advance + max-dwell cap + loud row ·
resume mid-octave determinism · no-double-driver ordering assert · governance classification.

Pure ($0): the controller is MLX-free. The trainer-helper tests import the trainer module (adds
repo/experiments + repo/src to sys.path, like the sister levelset helper tests) but exercise only
pure per-epoch schedule functions on an argparse.Namespace (no MLX / model / scorer).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pytest

from tac.witness_control.tau_advance import (
    SENSOR_PER_BAND_RELAXATION,
    TAU_OCTAVE_CAP_SLACK,
    TauAdvanceController,
    derive_n_octaves,
    derive_octave_max_dwell,
    tau_advance_restore_from_cfg,
    tau_advance_state_arrays,
    tau_octave_ladder,
)


def _always_exhausted(traj, **_kw):
    return {"exhausted": True, "remaining_meat_estimate": 0.0, "alpha": 1.0}


def _never_exhausted(traj, **_kw):
    return {"exhausted": False, "remaining_meat_estimate": 1.0, "alpha": 0.5}


def _event_ctrl(ladder=None, *, cap=500, min_dwell=0, min_points=1, sensor=_always_exhausted):
    return TauAdvanceController(
        mode="event", ladder=ladder or tau_octave_ladder(1.0, 0.31, 6),
        per_octave_cap=cap, min_dwell=min_dwell, min_points=min_points, sensor=sensor)


# ── ladder + derivations ─────────────────────────────────────────────────────────────────────────
def test_tau_octave_ladder_endpoints_and_geometric():
    lad = tau_octave_ladder(1.0, 0.31, 6)
    assert len(lad) == 7
    assert lad[0] == pytest.approx(1.0)
    assert lad[-1] == pytest.approx(0.31)
    # geometric: each rung == start*(end/start)^(k/N)
    for k, t in enumerate(lad):
        assert t == pytest.approx(1.0 * (0.31 / 1.0) ** (k / 6))
    # strictly decreasing (a descent)
    assert all(lad[i] > lad[i + 1] for i in range(len(lad) - 1))


def test_tau_octave_ladder_guards():
    with pytest.raises(ValueError):
        tau_octave_ladder(0.0, 0.31, 6)
    with pytest.raises(ValueError):
        tau_octave_ladder(1.0, -0.1, 6)
    with pytest.raises(ValueError):
        tau_octave_ladder(1.0, 0.31, 0)


def test_derive_n_octaves_and_max_dwell_no_bare_literal():
    # N ties ladder resolution to the min-stage dwell floor: 1500/250 = 6.
    assert derive_n_octaves(1500, 250) == 6
    assert derive_n_octaves(1000, 250) == 4
    assert derive_n_octaves(100, 250) == 1  # floored at 1
    # cap = ceil(anneal/N) * tagged slack.
    assert derive_octave_max_dwell(1500, 6) == math.ceil(math.ceil(1500 / 6) * TAU_OCTAVE_CAP_SLACK)


# ── clock mode = byte-identical (the binding DEFAULT-OFF contract) ─────────────────────────────────
def test_clock_mode_tau_is_the_clock_fn_and_never_advances():
    c = TauAdvanceController(mode="clock", ladder=tau_octave_ladder(1.0, 0.31, 6),
                             per_octave_cap=1, min_dwell=0)
    clock = lambda e: 0.5 + 0.001 * e  # noqa: E731 — sentinel clock fn
    for ep in (1, 10, 100):
        assert c.tau_for_epoch(ep, clock) == pytest.approx(clock(ep))
    # clock mode never advances, never ingests, never emits telemetry.
    c.ingest([{"epoch": 5, "d_seg": 0.01}])
    assert c._octave_hist == []
    step = c.maybe_advance(50)
    assert step.advanced is False and step.telemetry is None
    # and it serializes to ZERO keys equivalently: a clock run instantiates NO controller in the
    # trainer, but a passed None controller round-trips to an empty sidecar.
    assert tau_advance_state_arrays(None) == {}


def test_state_arrays_none_is_empty_byte_identical():
    assert tau_advance_state_arrays(None) == {}
    assert tau_advance_restore_from_cfg(None, {"__ta_rung": 3}) is False


# ── octave-ladder equivalence: event mode reuses the geometric clock ladder VALUES ────────────────
def test_event_instant_fire_reuses_geometric_clock_ladder_values():
    lad = tau_octave_ladder(1.0, 0.31, 6)
    c = _event_ctrl(lad)
    # rung 0 (before any advance) == τ_start == the geometric clock value at prog=0.
    assert c.tau_for_epoch(1, lambda e: -1.0) == pytest.approx(lad[0])
    # instant-fire (min_dwell=0, min_points=1, always-exhausted) advances through the ladder VALUES
    # in order — never inventing a τ off the ladder. The post-advance τ sequence == ladder[1:].
    seen = []
    for ep in range(1, 12):
        c.ingest([{"epoch": ep, "d_seg": 0.01}])
        s = c.maybe_advance(ep)
        seen.append(s.tau)
    assert seen[:6] == [pytest.approx(t) for t in lad[1:]]
    assert seen[6:] == [pytest.approx(lad[-1])] * (len(seen) - 6)  # latched at the floor
    assert c.at_floor and c.rung == 6


def test_event_ladder_values_match_geometric_softmax_temp_clock():
    """The ladder rung values ARE the incumbent geometric clock (_softmax_temp_for_epoch) sampled at
    prog=k/N — the honest 'event == clock VALUES' claim (only the per-rung DWELL differs)."""
    repo = Path(__file__).resolve().parents[3]
    for pth in (repo / "experiments", repo / "src"):
        if str(pth) not in sys.path:
            sys.path.insert(0, str(pth))
    from train_levelset_witness_realized_through_R_mlx import _softmax_temp_for_epoch

    n = 6
    # choose anneal_epochs so epoch e_k = 1 + (k/n)*(anneal-1) hits prog=k/n exactly.
    anneal = n * 100 + 1  # 601 -> (anneal-1)=600 divisible by n=6
    args = argparse.Namespace(anneal_epochs=anneal, epochs=anneal, softmax_temp_start=1.0,
                              softmax_temp_end=0.31, tau_anneal_shape="geometric", tau_hold_frac=1.0)
    lad = tau_octave_ladder(1.0, 0.31, n)
    for k in range(n + 1):
        e_k = 1 + (k * (anneal - 1)) // n
        assert lad[k] == pytest.approx(_softmax_temp_for_epoch(e_k, args), rel=1e-9)


# ── sensor advance / hold / dwell / cap ────────────────────────────────────────────────────────
def test_event_advances_on_sensor_exhaustion():
    c = _event_ctrl(min_dwell=10, min_points=3)
    # dwell below min_dwell => no advance even with points.
    for ep in range(1, 8):
        c.ingest([{"epoch": ep, "d_seg": 0.01}])
        assert c.maybe_advance(ep).advanced is False
    # past min_dwell + enough points + exhausted => advance (fired_by='event').
    for ep in range(8, 16):
        c.ingest([{"epoch": ep, "d_seg": 0.01}])
        s = c.maybe_advance(ep)
        if s.advanced:
            assert s.fired_by == "event"
            assert s.telemetry["stage"] == "tau_octave_advance"
            assert s.telemetry["sensor"] == SENSOR_PER_BAND_RELAXATION
            break
    else:
        pytest.fail("event sensor never advanced past min_dwell")
    assert c.rung == 1


def test_event_holds_when_sensor_not_exhausted():
    c = _event_ctrl(min_dwell=0, min_points=2, sensor=_never_exhausted)
    for ep in range(1, 60):
        c.ingest([{"epoch": ep, "d_seg": 0.01}])
        s = c.maybe_advance(ep)
        # never exhausted + below cap => hold at rung 0.
        if ep < 500:
            assert s.advanced is False
    assert c.rung == 0


def test_thin_data_fail_safe_no_fire_below_min_points():
    c = _event_ctrl(min_dwell=0, min_points=8, sensor=_always_exhausted)
    # only 3 points -> sensor cannot run (thin-data) -> no advance even though min_dwell passed.
    for ep in range(1, 4):
        c.ingest([{"epoch": ep, "d_seg": 0.01}])
        assert c.maybe_advance(ep).advanced is False
    assert c.rung == 0


def test_max_dwell_cap_fires_loud_backstop():
    c = _event_ctrl(cap=20, min_dwell=5, min_points=2, sensor=_never_exhausted)
    fired = None
    for ep in range(1, 40):
        c.ingest([{"epoch": ep, "d_seg": 0.01}])
        s = c.maybe_advance(ep)
        if s.advanced:
            fired = s
            break
    assert fired is not None
    assert fired.fired_by == "cap"
    assert fired.telemetry["stage"] == "cap_fired_before_event"  # LOUD (S5 falsification-relevant)
    assert "FAIL-SAFE BACKSTOP FIRED" in fired.telemetry["note"]
    assert fired.telemetry["epoch"] >= 20  # fired at/after the cap dwell
    assert fired.telemetry["dwell_epochs"] >= 20


# ── couplings: β + LR on the octave fraction ──────────────────────────────────────────────────
def test_octave_fraction_and_lr_anneal_fraction():
    c = _event_ctrl(tau_octave_ladder(1.0, 0.31, 4))
    assert c.octave_fraction() == 0.0
    assert c.lr_anneal_fraction() == 0.0
    c._rung = 2
    assert c.octave_fraction() == pytest.approx(0.5)
    assert c.lr_anneal_fraction() == pytest.approx(0.5)
    c._rung = 4
    assert c.octave_fraction() == pytest.approx(1.0)


def test_hosc_beta_couples_on_octave_fraction_and_none_contract():
    c = _event_ctrl(tau_octave_ladder(1.0, 0.31, 4))
    # None contract (incumbent _hosc_beta_for_epoch parity): not hosc / end unset / end==start.
    assert c.hosc_beta_for_epoch(4.0, 10.0, activation="relu") is None
    assert c.hosc_beta_for_epoch(4.0, None, activation="hosc") is None
    assert c.hosc_beta_for_epoch(4.0, 4.0, activation="hosc") is None
    # rung 0 -> beta_start ; rung N -> beta_end ; linear interp on the fraction.
    assert c.hosc_beta_for_epoch(4.0, 10.0, activation="hosc") == pytest.approx(4.0)
    c._rung = 4
    assert c.hosc_beta_for_epoch(4.0, 10.0, activation="hosc") == pytest.approx(10.0)
    c._rung = 2
    assert c.hosc_beta_for_epoch(4.0, 10.0, activation="hosc") == pytest.approx(7.0)
    # cosine form endpoints match too.
    c._rung = 0
    assert c.hosc_beta_for_epoch(4.0, 10.0, activation="hosc", shape="cosine") == pytest.approx(4.0)


# ── ingest semantics ─────────────────────────────────────────────────────────────────────────
def test_ingest_only_new_points_idempotent_and_clears_on_advance():
    c = _event_ctrl(min_dwell=0, min_points=100)  # never fires (thin) so we can inspect history
    c.ingest([{"epoch": 1, "d_seg": 0.02}, {"epoch": 2, "d_seg": 0.015}])
    assert len(c._octave_hist) == 2
    c.ingest([{"epoch": 1, "d_seg": 0.02}, {"epoch": 2, "d_seg": 0.015}])  # replay -> no dup
    assert len(c._octave_hist) == 2
    c.ingest([{"epoch": 3, "d_seg": 0.01}])  # only the new one
    assert len(c._octave_hist) == 3
    # on advance the octave history clears (new octave measured afresh).
    c2 = _event_ctrl(min_dwell=0, min_points=1, sensor=_always_exhausted)
    c2.ingest([{"epoch": 1, "d_seg": 0.01}, {"epoch": 2, "d_seg": 0.01}])
    c2.maybe_advance(2)
    assert c2._octave_hist == []


# ── no-double-driver ordering + freeze ─────────────────────────────────────────────────────────
def test_maybe_advance_while_frozen_raises_no_double_driver():
    c = _event_ctrl()
    c.freeze(700)
    assert c.frozen is True
    with pytest.raises(AssertionError):
        c.maybe_advance(701)


def test_freeze_telemetry_floored_vs_not_and_idempotent():
    c = _event_ctrl()
    c._rung = 2  # mid-descent
    row = c.freeze(700)
    assert row["stage"] == "tau_advance_frozen_at_muon"
    assert row["at_floor"] is False and "NOT at floor" in row["note"]
    assert row["tau_frozen"] == pytest.approx(round(float(c.ladder[2]), 6))
    row2 = c.freeze(700)  # idempotent
    assert row2["already_frozen"] is True
    # a floored controller reports floored.
    c2 = _event_ctrl()
    c2._rung = c2.n_octaves
    assert c2.freeze(700)["at_floor"] is True


def test_ingest_and_advance_are_noops_when_frozen_or_clock():
    c = _event_ctrl()
    c.freeze(700)
    c.ingest([{"epoch": 701, "d_seg": 0.01}])  # frozen -> no ingest
    assert c._octave_hist == []


# ── resume determinism (launch-critical) ──────────────────────────────────────────────────────
def test_resume_mid_octave_reproduces_identical_subsequent_tau_sequence():
    lad = tau_octave_ladder(1.0, 0.31, 6)
    # a deterministic sensor: exhausted once the octave has >= 4 points (a reproducible relaxation).
    def _sensor(traj, **_kw):
        return {"exhausted": len(traj) >= 4, "remaining_meat_estimate": 0.0, "alpha": 1.0}

    def _fresh():
        return TauAdvanceController(mode="event", ladder=lad, per_octave_cap=1000,
                                    min_dwell=3, min_points=4, sensor=_sensor)

    # CONTINUOUS run: drive 40 epochs, recording the τ used each epoch (post-advance).
    cont = _fresh()
    cont_taus = []
    for ep in range(1, 41):
        cont.ingest([{"epoch": ep, "d_seg": 0.01}])
        cont.maybe_advance(ep)
        cont_taus.append(cont.tau_for_epoch(ep, lambda e: -1.0))

    # RESUME run: drive to ep=17 (mid-octave), serialize, restore into a fresh controller, continue.
    a = _fresh()
    for ep in range(1, 18):
        a.ingest([{"epoch": ep, "d_seg": 0.01}])
        a.maybe_advance(ep)
    arrays = tau_advance_state_arrays(a)
    # round-trip through the np-array sidecar exactly as _load_resume_state parses it.
    cfg = {k: (v.item() if v.size == 1 else v.tolist()) for k, v in arrays.items()}
    b = _fresh()
    assert tau_advance_restore_from_cfg(b, cfg) is True
    assert b.rung == a.rung and b._octave_hist == a._octave_hist and b._last_seen_epoch == a._last_seen_epoch
    resume_taus = list(cont_taus[:17])
    for ep in range(18, 41):
        b.ingest([{"epoch": ep, "d_seg": 0.01}])
        b.maybe_advance(ep)
        resume_taus.append(b.tau_for_epoch(ep, lambda e: -1.0))
    assert resume_taus == cont_taus  # bit-faithful: identical subsequent τ trajectory


def test_resume_ignores_already_seen_verdict_epochs():
    c = _event_ctrl(min_dwell=0, min_points=100)
    for ep in range(1, 6):
        c.ingest([{"epoch": ep, "d_seg": 0.01}])
    arrays = tau_advance_state_arrays(c)
    cfg = {k: (v.item() if v.size == 1 else v.tolist()) for k, v in arrays.items()}
    d = _event_ctrl(min_dwell=0, min_points=100)
    tau_advance_restore_from_cfg(d, cfg)
    n0 = len(d._octave_hist)
    # a resume re-eval at an already-seen epoch (<= last_seen) is NOT re-ingested.
    d.ingest([{"epoch": 5, "d_seg": 0.01}])
    assert len(d._octave_hist) == n0


# ── the real powerlaw_meat sensor fires on a genuine within-octave plateau (integration) ──────────
def test_default_powerlaw_sensor_fires_on_flat_octave_and_holds_on_descent():
    lad = tau_octave_ladder(1.0, 0.31, 6)
    flat = TauAdvanceController(mode="event", ladder=lad, per_octave_cap=10_000,
                                min_dwell=0, min_points=8)  # DEFAULT sensor (powerlaw_meat)
    # a clearly-relaxed (flat) octave: remaining meat ~ 0 -> exhausted -> advance.
    for i, ep in enumerate(range(1, 30)):
        flat.ingest([{"epoch": ep, "d_seg": 0.005 + 1e-9 * i}])
        if flat.maybe_advance(ep).advanced:
            break
    assert flat.rung == 1
    # a steeply-descending octave: meat remains -> hold at rung 0.
    desc = TauAdvanceController(mode="event", ladder=lad, per_octave_cap=10_000,
                                min_dwell=0, min_points=8)
    for ep in range(1, 25):
        desc.ingest([{"epoch": ep, "d_seg": 0.05 / ep}])  # 1/t descent (meat left on the bone)
        desc.maybe_advance(ep)
    assert desc.rung == 0


# ── trainer-helper tests (pure per-epoch schedule fns on an argparse.Namespace) ───────────────────
def _trainer():
    repo = Path(__file__).resolve().parents[3]
    for pth in (repo / "experiments", repo / "src"):
        if str(pth) not in sys.path:
            sys.path.insert(0, str(pth))
    import train_levelset_witness_realized_through_R_mlx as T
    return T


def test_trainer_lr_scheduled_event_warmup_realepoch_and_anneal_on_octave_frac():
    T = _trainer()
    args = argparse.Namespace(warmup_epochs=5, lr=1e-3, lr_end=1e-4, lr_hold_frac=1.0)
    # warmup is REAL-epoch (independent of octave fraction).
    assert T._lr_scheduled_event_for_epoch(3, args, 0.9) == pytest.approx(1e-3 * 3 / 5)
    # past warmup: octave_frac 0 -> lr ; 1 -> lr_end ; cosine shape.
    assert T._lr_scheduled_event_for_epoch(100, args, 0.0) == pytest.approx(1e-3)
    assert T._lr_scheduled_event_for_epoch(100, args, 1.0) == pytest.approx(1e-4)
    mid = T._lr_scheduled_event_for_epoch(100, args, 0.5)
    assert 1e-4 < mid < 1e-3


def test_trainer_validate_tau_advance_config():
    T = _trainer()
    # clock is unconstrained.
    T.validate_tau_advance_config(tau_advance_mode="clock", tau_anneal_shape="cosine",
                                  softmax_temp_start=1.0, softmax_temp_end=0.05)
    # event requires geometric shape.
    with pytest.raises(ValueError):
        T.validate_tau_advance_config(tau_advance_mode="event", tau_anneal_shape="cosine",
                                      softmax_temp_start=1.0, softmax_temp_end=0.31)
    # event requires positive endpoints.
    with pytest.raises(ValueError):
        T.validate_tau_advance_config(tau_advance_mode="event", tau_anneal_shape="geometric",
                                      softmax_temp_start=1.0, softmax_temp_end=0.0)
    # valid event config passes.
    T.validate_tau_advance_config(tau_advance_mode="event", tau_anneal_shape="geometric",
                                  softmax_temp_start=1.0, softmax_temp_end=0.31)
    with pytest.raises(ValueError):
        T.validate_tau_advance_config(tau_advance_mode="bogus", tau_anneal_shape="geometric",
                                      softmax_temp_start=1.0, softmax_temp_end=0.31)


def test_trainer_build_controller_none_for_clock_and_derived_for_event():
    T = _trainer()
    clock_args = argparse.Namespace(tau_advance_mode="clock")
    assert T._build_tau_advance_controller(clock_args, 1500) is None
    ev_args = argparse.Namespace(
        tau_advance_mode="event", tau_octaves=None, tau_octave_min_dwell=None,
        tau_octave_max_dwell=None, curriculum_min_stage_epochs=250,
        softmax_temp_start=1.0, softmax_temp_end=0.31)
    c = T._build_tau_advance_controller(ev_args, 1500)
    assert c is not None
    assert c.n_octaves == derive_n_octaves(1500, 250) == 6
    assert c.per_octave_cap == derive_octave_max_dwell(1500, 6)
    assert c.min_dwell == 250
    assert c.ladder[0] == pytest.approx(1.0) and c.ladder[-1] == pytest.approx(0.31)
    # explicit overrides win.
    ev2 = argparse.Namespace(
        tau_advance_mode="event", tau_octaves=8, tau_octave_min_dwell=100,
        tau_octave_max_dwell=333, curriculum_min_stage_epochs=250,
        softmax_temp_start=1.0, softmax_temp_end=0.31)
    c2 = T._build_tau_advance_controller(ev2, 1500)
    assert c2.n_octaves == 8 and c2.min_dwell == 100 and c2.per_octave_cap == 333
