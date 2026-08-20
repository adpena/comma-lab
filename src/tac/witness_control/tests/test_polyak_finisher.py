"""Tests for the R-7 Polyak tail-averaging finisher + the beta2-window rewarmup sizing rule.

Covers: the sizing-rule derivation, DEFAULT-OFF no-op/byte-identity, Polyak-vs-EMA divergence on a
synthetic ORBIT (the math property the finisher exploits), resume round-trip of the averager state,
registry registration (Resumable protocol), the DSL lever compile+validate, and completeness mapping.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.polyak_finisher import (
    POLYAK_SCALAR_PREFIX,
    PolyakTailAverager,
    polyak_finisher_window_provenance,
)


# --- sizing-rule derivation (finisher 1: beta2-window rewarmup) ---------------
def test_min_rewarmup_epochs_covers_beta2_memory_horizon():
    from tac.canonical_equations.curriculum_derivation_laws_20260705 import min_rewarmup_epochs
    # 1/(1-0.999) = 1000 steps; at 75 steps/ep => ceil(1000/75) = 14 ep.
    assert min_rewarmup_epochs(0.999, 75) == 14
    # window * steps/ep must COVER the memory horizon 1/(1-beta2) (the law's binding inequality).
    for beta2, spe in [(0.999, 75), (0.99, 20), (0.9999, 100), (0.95, 8)]:
        win = min_rewarmup_epochs(beta2, spe)
        assert win * spe >= 1.0 / (1.0 - beta2)
        # ...and it is the TIGHT ceil (win-1 would NOT cover) — the derivation is minimal, not slack.
        assert (win - 1) * spe < 1.0 / (1.0 - beta2)


def test_min_rewarmup_epochs_rejects_bad_inputs():
    from tac.canonical_equations.curriculum_derivation_laws_20260705 import min_rewarmup_epochs
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            min_rewarmup_epochs(bad, 75)
    with pytest.raises(ValueError):
        min_rewarmup_epochs(0.999, 0)


# --- Polyak window provenance (DERIVED-AT-CONFIG, no bare literal) ------------
def test_polyak_window_provenance_derives_from_stage():
    prov = polyak_finisher_window_provenance(1000, frac=0.2)
    assert prov["ladder_class"] == "derived_at_config"
    assert prov["equation_id"] == "muon_finisher_schedule_warmstart_and_lr_anneal_v1"
    assert prov["polyak_window_epochs"] == 200
    assert prov["polyak_start_epoch"] == 800  # stage - window
    # edge: rejects nonsense so a bad config can never silently size a zero-window tail.
    with pytest.raises(ValueError):
        polyak_finisher_window_provenance(0)
    with pytest.raises(ValueError):
        polyak_finisher_window_provenance(1000, frac=1.5)


# --- DEFAULT-OFF no-op / byte-identity ---------------------------------------
def test_default_off_is_total_noop():
    avg = PolyakTailAverager(start_epoch=0, arm=False)
    assert avg.active is False
    # observe never accumulates; every persistence surface is EMPTY => zero new checkpoint keys.
    assert avg.observe(0, {"w": np.ones(4)}) is False
    assert avg.observe(999, {"w": np.ones(4)}) is False
    assert avg.count == 0
    assert avg.mean_fp32() is None
    assert avg.heavy_state_arrays("polyakM__") == {}
    assert avg.state_arrays(POLYAK_SCALAR_PREFIX) == {}


def test_armed_but_before_start_is_noop():
    avg = PolyakTailAverager(start_epoch=100, arm=True)
    assert avg.observe(50, {"w": np.ones(4)}) is False
    assert avg.count == 0
    scalar = avg.state_arrays(POLYAK_SCALAR_PREFIX)
    assert {key: int(value) for key, value in scalar.items()} == {
        "__pta_count": 0,
        "__pta_start": 100,
        "__pta_arm": 1,
    }
    assert avg.heavy_state_arrays("polyakM__") == {}

    reopened = PolyakTailAverager(start_epoch=100, arm=True)
    assert reopened.restore_from_cfg(
        POLYAK_SCALAR_PREFIX,
        {key: int(value) for key, value in scalar.items()},
    )
    assert reopened.count == 0
    assert reopened.start_epoch == 100
    assert reopened.heavy_state_arrays("polyakM__") == {}


# --- uniform-mean correctness -------------------------------------------------
def test_observe_is_exact_uniform_mean():
    avg = PolyakTailAverager(start_epoch=0, arm=True)
    xs = [np.array([1.0, 10.0]), np.array([3.0, 20.0]), np.array([5.0, 30.0])]
    for ep, x in enumerate(xs):
        avg.observe(ep, {"w": x})
    assert avg.count == 3
    got = avg.mean_fp32()["w"]
    np.testing.assert_allclose(got, np.mean(xs, axis=0).astype(np.float32), rtol=0, atol=1e-6)


# --- THE MATH PROPERTY: tail mean beats a short-horizon EMA at an orbit -------
def test_polyak_tail_mean_beats_short_horizon_ema_on_orbit():
    """At a turnpike the iterates ORBIT a basin center. The uniform tail mean averages the orbit out;
    a fixed-horizon EMA still carries orbit phase. Over a full number of periods the uniform mean is
    the EXACT center; the EMA is strictly farther from it."""
    center = np.array([2.0, -1.0])
    period = 20
    n = 200  # exactly 10 periods
    avg = PolyakTailAverager(start_epoch=0, arm=True)
    # short-horizon EMA (decay 0.9 => horizon ~10 steps << period 20 => carries orbit phase).
    decay = 0.9
    ema = None
    for t in range(n):
        theta = 2.0 * np.pi * t / period
        x = center + np.array([np.cos(theta), np.sin(theta)])  # unit orbit around center
        avg.observe(t, {"w": x})
        ema = x.copy() if ema is None else decay * ema + (1.0 - decay) * x
    polyak = avg.mean_fp32()["w"].astype(np.float64)
    polyak_err = float(np.linalg.norm(polyak - center))
    ema_err = float(np.linalg.norm(ema - center))
    # uniform mean over whole periods == center to fp tolerance; EMA is materially off-center.
    assert polyak_err < 1e-9
    assert ema_err > 10.0 * polyak_err
    assert ema_err > 0.05


# --- resume round-trip (heavy + scalar) --------------------------------------
def test_resume_roundtrip_continues_uniform_mean_bit_faithfully():
    # run A: observe 5 epochs, then "crash".
    a = PolyakTailAverager(start_epoch=0, arm=True)
    seq = [np.array([float(i), float(2 * i)]) for i in range(1, 9)]
    for ep in range(5):
        a.observe(ep, {"w": seq[ep]})
    heavy = a.heavy_state_arrays("polyakM__")
    scalar = a.state_arrays(POLYAK_SCALAR_PREFIX)
    assert set(heavy) == {"polyakM__w"}
    # sidecar cfg parse: __-scalars become python ints (mirrors _load_resume_state .item()).
    cfg = {k: int(v) for k, v in scalar.items()}

    # run B: fresh averager, restore, continue epochs 5..7.
    b = PolyakTailAverager(start_epoch=0, arm=True)
    assert b.restore_from_cfg(POLYAK_SCALAR_PREFIX, cfg) is True
    assert b.count == 5
    heavy_np = {k[len("polyakM__"):]: v for k, v in heavy.items()}
    assert b.restore_heavy(heavy_np) is True
    for ep in range(5, 8):
        b.observe(ep, {"w": seq[ep]})

    # continuous reference: one averager over all 8 epochs.
    ref = PolyakTailAverager(start_epoch=0, arm=True)
    for ep in range(8):
        ref.observe(ep, {"w": seq[ep]})
    assert b.count == ref.count == 8
    np.testing.assert_allclose(b.mean_fp32()["w"], ref.mean_fp32()["w"], rtol=0, atol=1e-9)


def test_restore_from_cfg_missing_keys_returns_false():
    b = PolyakTailAverager(start_epoch=0, arm=True)
    assert b.restore_from_cfg(POLYAK_SCALAR_PREFIX, {}) is False
    assert b.restore_heavy(None) is False
    assert b.restore_heavy({}) is False


# --- registry registration (Resumable protocol) ------------------------------
def test_registers_through_resume_registry_and_roundtrips():
    from tac.witness_control.resume_registry import ResumeRegistry
    avg = PolyakTailAverager(start_epoch=0, arm=True)
    for ep in range(3):
        avg.observe(ep, {"w": np.array([float(ep)])})
    reg = ResumeRegistry()
    reg.register("polyak_finisher", POLYAK_SCALAR_PREFIX, avg)
    assert "polyak_finisher" in reg.names
    arrays = reg.state_arrays()  # scalars merged + manifest stamped (real state present)
    assert any(k.startswith(POLYAK_SCALAR_PREFIX) for k in arrays)
    # NOT an event controller => never trips the event fail-closed.
    assert getattr(avg, "event_mode", False) is False


def test_registry_emits_nothing_when_averager_inert():
    from tac.witness_control.resume_registry import ResumeRegistry
    avg = PolyakTailAverager(start_epoch=0, arm=False)
    reg = ResumeRegistry()
    reg.register("polyak_finisher", POLYAK_SCALAR_PREFIX, avg)
    # inert averager => {} => registry emits {} with NO manifest => byte-identical sidecar.
    assert reg.state_arrays() == {}


# --- DSL levers (finisher 1 + finisher 2) ------------------------------------
def test_beta2_window_rewarmup_lever_compiles_and_derives_window():
    from tac.witness_dsl.curriculum_dsl import Beta2WindowRewarmup
    lev = Beta2WindowRewarmup(beta2=0.999, steps_per_epoch=75)
    ov = lev.overrides
    assert ov["--stage-transition-rewarmup-epochs"] == 14  # DERIVED, not a bare literal
    assert ov["--stage-transition-reset-moments"] is True
    assert "rewarmup_beta2_memory_window_v1" in lev.notes


def test_polyak_finisher_lever_compiles():
    from tac.witness_dsl.curriculum_dsl import PolyakFinisher
    lev = PolyakFinisher(start_epoch=800)
    ov = lev.overrides
    assert ov["--polyak-finisher-arm"] is True
    assert ov["--polyak-finisher-start-epoch"] == 800
    assert "muon_finisher_schedule_warmstart_and_lr_anneal_v1" in lev.notes


def test_new_levers_are_dsl_composable_by_bare_name():
    from tac.witness_dsl.lever_registry import name_composable_levers
    names = name_composable_levers()
    assert "Beta2WindowRewarmup" in names
    assert "PolyakFinisher" in names


def test_completeness_new_polyak_flags_mapped_zero_unmapped():
    from tac.witness_dsl.lever_registry import completeness
    c = completeness()
    for f in ("--polyak-finisher-arm", "--polyak-finisher-start-epoch"):
        assert f in c.mapped, f"{f} should be DSL-mapped (0 unmapped for new flags)"
        assert f not in c.unmapped
        assert f not in c.stale


# --- TRAINER INTEGRATION (real _load_resume_state routing + heavy round-trip) -
def test_trainer_load_resume_state_routes_polyak_and_isolates_live(tmp_path):
    """Code-correctness check against the REAL trainer helpers (_RESUME_POLYAK_PREFIX,
    _atomic_savez, _load_resume_state): the heavy Polyak mean routes to rs['polyak'] (never the model
    live params), the scalar count routes to cfg, and an un-armed run yields ZERO polyak keys
    (byte-identical observable). Imports the trainer lazily (heavy MLX import)."""
    import importlib

    T = importlib.import_module(
        "experiments.train_levelset_witness_realized_through_R_mlx")
    from tac.witness_control.resume_registry import ResumeRegistry
    pfx = T._RESUME_POLYAK_PREFIX

    # ARMED: merge like _do_checkpoint, savez, reload.
    avg = PolyakTailAverager(start_epoch=0, arm=True)
    for ep in range(4):
        avg.observe(ep, {"film.weight": np.full((2, 3), float(ep)), "b": np.array([float(ep)])})
    reg = ResumeRegistry()
    reg.register("polyak_finisher", POLYAK_SCALAR_PREFIX, avg)
    arrays: dict = {"liveP__film.weight": np.zeros((2, 3), np.float32),
                    "__resume_epoch": np.asarray(3)}
    arrays.update(reg.state_arrays())
    arrays.update(avg.heavy_state_arrays(pfx))
    p = tmp_path / "levelset_resume_state.npz"
    T._atomic_savez(p, arrays)
    rs = T._load_resume_state(p)
    assert set(rs["polyak"]) == {"film.weight", "b"}
    np.testing.assert_allclose(rs["polyak"]["film.weight"], np.full((2, 3), 1.5))
    assert int(rs["cfg"][POLYAK_SCALAR_PREFIX + "count"]) == 4
    # routing isolation: no polyak key leaked into the model live-param restore.
    assert "film.weight" in rs["live"] and not any("polyak" in k for k in rs["live"])

    # OFF: an un-armed run writes no polyak keys => rs['polyak'] empty (byte-identical observable).
    off = {"liveP__film.weight": np.zeros((2, 3), np.float32), "__resume_epoch": np.asarray(3)}
    po = tmp_path / "off.npz"
    T._atomic_savez(po, off)
    assert T._load_resume_state(po)["polyak"] == {}
