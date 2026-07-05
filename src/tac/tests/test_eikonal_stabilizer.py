"""Tests for the EIK-STAB build (2026-07-05): eikonal stabilizer terms + rollback spike guard.

Covers, per the litsweep design source (.omx/research/litsweep_training_dynamics_control_20260705.md)
and the stepping-instability diagnostic:

  build 1a — ``_eikonal_steik_mlx``: StEik directional-divergence damping (arXiv 2305.18414,
             L = mean |grad m^T H(m) grad m|, L1, raw gradient) on the decision margin m;
  build 1b — ``_eikonal_visco_mlx``: ViscoReg vanishing-viscosity residual (arXiv 2507.00412,
             mean (|grad m| - 1 - eps*Lap m)^2, p=2) + ``_visco_eps_for_epoch`` linear decay;
  build 2  — ``SpikeGuardRollback``: the pure decision state machine behind
             ``--spike-guard-mode rollback`` (bounded-oscillation tolerance; sustained-runaway
             rollback; re-arm; bounded actuation budget), incl. the INDUCED-RUNAWAY scenario;
  plumbing — flags exist with OFF defaults; total_loss_fn gates on the closure cell; micro-batch
             fails closed; the trainer loop wires rollback + lr-scale + median hygiene.

The trainer-level BITWISE off-path identity (flags at defaults) is proven by the runtime n1 CPU
A/B at experiments/results/eik_stab_build_20260705/idref_{pre,post} (106/106 resume-state keys +
EMA npz byte-equal; per-batch loss values identical modulo the new always-present eik_steik=0.0
schema key) — this file guards the pure surfaces. Term math is checked on ANALYTIC fields where
the damping is hand-computable (linear plane -> 0; quadratic -> closed form) plus an independent
numpy re-implementation on random fields."""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import numpy as np
import pytest

pytest.importorskip("mlx", reason="level-set witness trainer requires mlx")
import mlx.core as mx  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[3]
_MODPATH = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _load(path: pathlib.Path, name: str):
    if not path.exists():
        pytest.skip(f"trainer not found at {path}", allow_module_level=True)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - env-dependent
        sys.modules.pop(name, None)
        pytest.skip(f"could not import {path.name}: {type(exc).__name__}: {exc}",
                    allow_module_level=True)
    return mod


MOD = _load(_MODPATH, "_levelset_trainer_eik_stab_test")
SRC = _MODPATH.read_text()

H, W, K = 16, 20, 3


def _phi_from_margin(m_hw: np.ndarray, k: int = K) -> mx.array:
    """Build a (H*W, K) phi whose decision margin top1-top2 == m_hw exactly: phi0 = m (>0
    required), phi1 = 0, remaining classes = -1 (never in the top two when m > ... use m>0 and
    the third class far below)."""
    hh, ww = m_hw.shape
    phi = np.full((hh * ww, k), -10.0, np.float32)
    phi[:, 0] = m_hw.reshape(-1).astype(np.float32)
    phi[:, 1] = 0.0
    return mx.array(phi)


def _margin_interior_np(m: np.ndarray):
    """Independent numpy re-implementation of the central interior stencil (the test oracle)."""
    m = m.astype(np.float64)
    gx = 0.5 * (m[1:-1, 2:] - m[1:-1, :-2])
    gy = 0.5 * (m[2:, 1:-1] - m[:-2, 1:-1])
    m_xx = m[1:-1, 2:] - 2.0 * m[1:-1, 1:-1] + m[1:-1, :-2]
    m_yy = m[2:, 1:-1] - 2.0 * m[1:-1, 1:-1] + m[:-2, 1:-1]
    m_xy = 0.25 * (m[2:, 2:] - m[2:, :-2] - m[:-2, 2:] + m[:-2, :-2])
    return gx, gy, m_xx, m_yy, m_xy


# ────────────────────────────────────────────── build 1a: StEik directional divergence
def test_steik_zero_on_linear_margin():
    # m linear (a plane): H(m) == 0 => the directional divergence is exactly 0 (a true-SDF
    # margin has D^2m·grad m = 0; a plane is the cleanest instance).
    x = np.arange(W, dtype=np.float64)[None, :].repeat(H, axis=0)
    m = 1.0 + 0.05 * x
    out = float(MOD._eikonal_steik_mlx(_phi_from_margin(m), H, W))
    assert abs(out) < 1e-6


def test_steik_quadratic_matches_hand_formula():
    # m = c + 0.5*a*x^2: gx = a*x (central diff EXACT on quadratics), m_xx = a, gy = m_yy = m_xy
    # = 0 => integrand = |(a x)^2 * a| = a^3 x^2; mean over the interior grid.
    a, c = 0.01, 5.0
    x = np.arange(W, dtype=np.float64)[None, :].repeat(H, axis=0)
    m = c + 0.5 * a * x * x
    out = float(MOD._eikonal_steik_mlx(_phi_from_margin(m), H, W))
    xi = x[1:-1, 1:-1]
    expected = float(np.mean((a * xi) ** 2 * a))
    assert out == pytest.approx(expected, rel=1e-3)


def test_steik_matches_independent_numpy_on_random_field():
    rng = np.random.default_rng(7)
    m = 2.0 + rng.standard_normal((H, W)) * 0.3
    m = np.maximum(m, 0.1)
    out = float(MOD._eikonal_steik_mlx(_phi_from_margin(m), H, W))
    gx, gy, m_xx, m_yy, m_xy = _margin_interior_np(m.astype(np.float32))
    expected = float(np.mean(np.abs(gx * gx * m_xx + 2 * gx * gy * m_xy + gy * gy * m_yy)))
    assert out == pytest.approx(expected, rel=1e-3)


def test_margin_interior_shapes():
    m = np.full((H, W), 3.0)
    gx, gy, m_xx, m_yy, m_xy = MOD._eikonal_margin_interior_mlx(_phi_from_margin(m), H, W)
    for t in (gx, gy, m_xx, m_yy, m_xy):
        assert tuple(t.shape) == (H - 2, W - 2)


# ────────────────────────────────────────────── build 1b: ViscoReg viscous residual
def test_visco_zero_residual_on_unit_slope_plane():
    # |grad m| = 1, Lap m = 0 => residual ~ 0 (up to the 1e-8 sqrt regularizer).
    x = np.arange(W, dtype=np.float64)[None, :].repeat(H, axis=0)
    m = 1.0 + x  # slope exactly 1
    out = float(MOD._eikonal_visco_mlx(_phi_from_margin(m), H, W, 0.5))
    assert abs(out) < 1e-10


def test_visco_quadratic_matches_hand_formula():
    # m = c + 0.5*a*x^2: |grad| = sqrt((a x)^2 + 1e-8), Lap = a =>
    # residual = (sqrt((a x)^2 + 1e-8) - 1 - eps*a)^2 elementwise on the interior.
    a, c, eps = 0.02, 4.0, 0.7
    x = np.arange(W, dtype=np.float64)[None, :].repeat(H, axis=0)
    m = c + 0.5 * a * x * x
    out = float(MOD._eikonal_visco_mlx(_phi_from_margin(m), H, W, eps))
    xi = x[1:-1, 1:-1]
    resid = np.sqrt((a * xi) ** 2 + 1e-8) - 1.0 - eps * a
    assert out == pytest.approx(float(np.mean(resid ** 2)), rel=1e-3)


def test_visco_matches_independent_numpy_on_random_field():
    rng = np.random.default_rng(11)
    m = 2.0 + rng.standard_normal((H, W)) * 0.25
    m = np.maximum(m, 0.1)
    eps = 0.3
    out = float(MOD._eikonal_visco_mlx(_phi_from_margin(m), H, W, eps))
    gx, gy, m_xx, m_yy, _ = _margin_interior_np(m.astype(np.float32))
    resid = np.sqrt(gx * gx + gy * gy + 1e-8) - 1.0 - eps * (m_xx + m_yy)
    assert out == pytest.approx(float(np.mean(resid ** 2)), rel=1e-3)


def test_visco_eps_schedule_constant_when_no_anneal():
    assert MOD._visco_eps_for_epoch(0, 1.0, 0) == 1.0
    assert MOD._visco_eps_for_epoch(500, 1.0, 0) == 1.0


def test_visco_eps_schedule_linear_decay_endpoints_and_midpoint():
    assert MOD._visco_eps_for_epoch(0, 1.0, 100) == pytest.approx(1.0)
    assert MOD._visco_eps_for_epoch(50, 1.0, 100) == pytest.approx(0.5)
    assert MOD._visco_eps_for_epoch(100, 1.0, 100) == 0.0
    assert MOD._visco_eps_for_epoch(101, 1.0, 100) == 0.0  # clamped at 0 past the anneal


def test_visco_eps_schedule_zero_eps0_is_always_zero():
    assert MOD._visco_eps_for_epoch(0, 0.0, 100) == 0.0
    assert MOD._visco_eps_for_epoch(10, 0.0, 0) == 0.0


# ────────────────────────────────────────────── schema + plumbing (source-level guards)
def test_loss_term_keys_include_eik_steik():
    assert "eik_steik" in MOD.LOSS_TERM_KEYS
    # row builder fills it as 0.0 when absent (stable schema)
    row = MOD._loss_terms_row({"seg": 1.0}, 1.0, 3, 0)
    assert row["terms"]["eik_steik"] == 0.0


def test_argparse_defaults_are_off():
    assert re.search(r'"--eikonal-steik-weight",\s*type=float,\s*default=0\.0', SRC)
    assert re.search(r'"--eikonal-viscosity",\s*type=float,\s*default=0\.0', SRC)
    assert re.search(r'"--eikonal-viscosity-anneal",\s*type=int,\s*default=0', SRC)
    assert re.search(r'"--spike-guard-mode",\s*type=str,\s*default="legacy"', SRC)


def test_total_loss_fn_gates_on_closure_cell():
    # visco REPLACES the residual only when eps > 0; steik is additive only when weight > 0.
    assert 'if _eik_stab["visco_eps"] > 0.0:' in SRC
    assert 'if _eik_stab["steik_w"] > 0.0:' in SRC
    assert 'terms_out["eik_steik"]' in SRC


def test_micro_batch_fails_closed_for_stabilizers():
    m = re.search(r'_eik_stab\["steik_w"\] > 0\.0 or _eik_stab\["visco_eps0"\] > 0\.0.*?'
                  r'raise NotImplementedError', SRC, re.S)
    assert m, "stabilizer flags must FAIL CLOSED under --micro-batch-pairs (NO-FAKE silent-drop)"


def test_visco_junction_relax_mutual_exclusion_guard():
    assert "--eikonal-viscosity > 0 REPLACES the eikonal residual" in SRC


# ────────────────────────────────────────────── build 2: SpikeGuardRollback state machine
def test_guard_ctor_validation():
    with pytest.raises(ValueError):
        MOD.SpikeGuardRollback(0, 0.5, 8)
    with pytest.raises(ValueError):
        MOD.SpikeGuardRollback(10, 0.0, 8)
    with pytest.raises(ValueError):
        MOD.SpikeGuardRollback(10, 1.5, 8)
    with pytest.raises(ValueError):
        MOD.SpikeGuardRollback(10, 0.5, 0)


def test_guard_no_trigger_below_frac():
    g = MOD.SpikeGuardRollback(10, 0.5, 8)
    # 40% spikes in every full window: never triggers
    pattern = [True, True, False, False, True, False, True, False, False, False]
    for _ in range(5):
        for s in pattern:
            assert g.observe(s) == "ok"
    assert g.rollbacks == 0


def test_guard_requires_full_window():
    g = MOD.SpikeGuardRollback(10, 0.5, 8)
    # 5 consecutive spikes (frac 1.0 > 0.5) but the window is not yet full -> no trigger
    for _ in range(5):
        assert g.observe(True) == "ok"
    assert g.rollbacks == 0


def test_guard_triggers_on_sustained_runaway():
    g = MOD.SpikeGuardRollback(10, 0.5, 8)
    acts = [g.observe(True) for _ in range(10)]
    assert acts[:-1] == ["ok"] * 9 and acts[-1] == "rollback"
    assert g.rollbacks == 1
    assert g.spike_frac() == 0.0  # self-rearmed


def test_guard_rearm_needs_window_refill():
    g = MOD.SpikeGuardRollback(6, 0.5, 8)
    for _ in range(6):
        g.observe(True)
    assert g.rollbacks == 1
    # continued runaway: the NEXT trigger needs another FULL window (bounded frequency)
    acts = [g.observe(True) for _ in range(6)]
    assert acts[:-1] == ["ok"] * 5 and acts[-1] == "rollback"
    assert g.rollbacks == 2


def test_guard_exhausted_after_budget():
    g = MOD.SpikeGuardRollback(4, 0.5, 2)
    for _ in range(8):  # two full windows of pure spikes -> 2 rollbacks = the budget
        g.observe(True)
    assert g.rollbacks == 2 and g.exhausted
    assert g.observe(True) == "exhausted"
    assert g.observe(False) == "exhausted"
    assert g.rollbacks == 2  # never exceeds the budget


def test_guard_healthy_stream_never_triggers():
    g = MOD.SpikeGuardRollback(10, 0.5, 8)
    for _ in range(100):
        assert g.observe(False) == "ok"
    assert g.rollbacks == 0 and g.spike_frac() == 0.0


def test_guard_induced_runaway_scenario_rollback_cut_rearm_continue():
    """The induced-runaway unit test: healthy training -> runaway -> the guard rolls back,
    re-arms, training CONTINUES (post-rollback healthy stream is accepted), and the actuation
    budget bounds repeated triggers."""
    g = MOD.SpikeGuardRollback(10, 0.5, 3)
    # phase 1: 50 healthy batches
    for _ in range(50):
        assert g.observe(False) == "ok"
    # phase 2: induced runaway — mixed 80% spikes; window fills with the mix
    fired_at = None
    seq = ([True, True, True, True, False] * 20)
    for i, s in enumerate(seq):
        a = g.observe(s)
        if a == "rollback":
            fired_at = i
            break
    assert fired_at is not None and g.rollbacks == 1
    # phase 3: post-rollback the run CONTINUES; a healthy stream never re-triggers
    for _ in range(30):
        assert g.observe(False) == "ok"
    assert g.rollbacks == 1
    # phase 4: sustained runaway again -> triggers until the budget, then exhausted forever
    acts = [g.observe(True) for _ in range(40)]
    assert acts.count("rollback") == 2  # budget 3, one already used
    assert acts[-1] == "exhausted"


def test_guard_rearm_clears_window():
    g = MOD.SpikeGuardRollback(5, 0.5, 8)
    for _ in range(3):
        g.observe(True)
    g.rearm()
    assert g.spike_frac() == 0.0
    # after rearm, needs a fresh FULL window again
    acts = [g.observe(True) for _ in range(5)]
    assert acts[-1] == "rollback" and acts[:-1] == ["ok"] * 4


def test_trainer_loop_wires_rollback_actuator():
    # the loop must: dispatch on mode, call the rollback closure, cut lr persistently, and keep
    # spiked batches out of the median in rollback mode.
    assert '_sg_act == "rollback"' in SRC
    assert "_sg_do_rollback(ep, batch_loss, gnorm)" in SRC
    assert '"stage": "spike_rollback"' in SRC
    assert 'if _sg_state["lr_scale"] != 1.0:' in SRC
    assert "_sg_guard is None or not (_spiked or _nonfinite)" in SRC
    # moments RESTORED on rollback (measured: restored moments DAMP 6.7x vs fresh 25.3x)
    assert 'opt.state = tree_unflatten(list(snap["opt"].items()))' in SRC


def test_trainer_snapshot_refresh_only_when_previous_epoch_healthy():
    assert '_prev_frac < float(args.spike_rollback_frac)' in SRC
    assert "_sg_take_snapshot(ep)" in SRC


# ────────────────────────────────────────────── build 4: lambda_pre probe plumbing
def test_lambda_pre_probe_flag_defaults_off_and_exits_before_training():
    assert re.search(r'"--lambda-pre-probe-iters",\s*type=int,\s*default=0', SRC)
    m = re.search(r'lambda_pre_probe_iters.*?raise SystemExit\(0\)', SRC, re.S)
    assert m, "probe mode must EXIT before any training step"
    assert '"stage": "lambda_pre"' in SRC
    assert "pi_eos" in SRC  # the dimensionless EoS group eta*lambda_pre/38
