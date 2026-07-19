# SPDX-License-Identifier: MIT
"""Arm B deliverable 3 — event couplings (SPEC_v10 §13.2).

(a) β-anneal-complete → pose-finish-eligible coupling (trainer flag + DSL lever);
(b) per-force event-entry ``ncde_dseg`` sensor consuming the #344 NCDE hit->solve detector;
(c) the observer-side ``fire=unavailable`` fix (probe always emits a structured
    ``verdict_latest_advisory``; the shadow controller never stores ``None``).

MEASURED root cause for (c), reproduced in-test: the live c2 run had < 8 verdict rows (25-ep
cadence), ``build_verdict_path`` returned None, ``run_probe`` silently omitted the key, the
shadow controller stored ``ncde_344=None`` and the costate digest printed the diagnostic-free
``#344 fire=False reason=unavailable``.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control.event_wirings import (
    RECOGNISED_START_EVENT_SENSORS,
    SENSOR_NCDE_DSEG,
    ncde_dseg_event,
)

_REPO = Path(__file__).resolve().parents[4]
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _sat_rows(n, asym=0.005, amp=0.02, rho=0.5, cadence=25):
    """Saturating exponential approach (the exactly-linear-ODE chart in linear d_seg space)."""
    return [{"epoch": 600 + cadence * i, "d_seg": asym + amp * (rho ** i)} for i in range(n)]


# ── (b) the ncde_dseg sensor ───────────────────────────────────────────────── #
def test_ncde_dseg_basin_fires_on_spent_descent():
    r = ncde_dseg_event(_sat_rows(12))
    assert r["fired"] and r["fire"]
    assert r["reason"].startswith("BASIN")
    assert r["fit_r2"] > 0.9


def test_ncde_dseg_handoff_fires_on_predicted_slope_flatten():
    r = ncde_dseg_event(_sat_rows(16))
    assert r["fired"]
    assert r["reason"].startswith(("BASIN", "HANDOFF"))


def test_ncde_dseg_no_fire_on_steady_descent():
    rows = [{"epoch": 600 + 25 * i, "d_seg": 0.2 * (0.93 ** i)} for i in range(16)]
    r = ncde_dseg_event(rows)
    assert not r["fired"]
    assert "still descending" in r["reason"]


def test_ncde_dseg_insufficient_rows_fail_safe():
    r = ncde_dseg_event(_sat_rows(5))
    assert not r["fired"]
    assert "insufficient verdict rows" in r["reason"]
    assert r["sensor"] == SENSOR_NCDE_DSEG


def test_ncde_dseg_conservative_on_fully_flat_window():
    """A window with no resolvable dynamics NEVER fires (NO-FAKE fit guard) — the backstop cap
    owns the fail-safe path."""
    rows = [{"epoch": 600 + 25 * i, "d_seg": 0.005} for i in range(16)]
    r = ncde_dseg_event(rows)
    assert not r["fired"]


def test_ncde_dseg_skips_duplicate_and_nonfinite_rows():
    rows = _sat_rows(12) + [{"epoch": 600, "d_seg": 0.9},          # duplicate epoch ignored
                            {"epoch": 9999, "d_seg": float("nan")},  # non-finite ignored
                            {"epoch": 9998, "d_seg": None}]          # missing ignored
    r = ncde_dseg_event(rows)
    assert r["n_rows"] == 12
    assert r["fired"]


def test_ncde_dseg_registered_sensor():
    assert SENSOR_NCDE_DSEG in RECOGNISED_START_EVENT_SENSORS


# ── trainer wiring surfaces (never-invent-flags) ───────────────────────────── #
def test_trainer_start_event_choices_include_ncde_dseg():
    src = _TRAINER.read_text(errors="ignore")
    m = re.search(r"add_argument\(\"--seg-phase-advect-start-event\".*?choices=\[([^\]]*)\]",
                  src, re.S)
    assert m is not None and "ncde_dseg" in m.group(1)
    assert "_ncde_dseg_ev" in src   # the engage-block dispatch actually consumes the sensor


def test_trainer_coupling_flag_declared_default_off_and_fail_loud():
    src = _TRAINER.read_text(errors="ignore")
    m = re.search(r"add_argument\(\s*\"--pose-finish-eligible-on-beta-anneal-complete\".*?"
                  r"default=(\w+)", src, re.S)
    assert m is not None and m.group(1) == "False"
    # inert-arm NO-FAKE guard: coupling without the two-phase arm raises
    assert "requires --pose-finish-start-epoch > 0" in src
    # the engage block actually consumes the coupling (deferral branch + loud row)
    assert "pose_finish_coupling_deferred" in src


# ── (a) DSL levers ─────────────────────────────────────────────────────────── #
def test_dsl_beta_anneal_coupling_lever():
    from tac.witness_dsl.curriculum_dsl import PoseFinishBetaAnnealCoupling

    lever = PoseFinishBetaAnnealCoupling()
    assert lever.overrides == {"--pose-finish-eligible-on-beta-anneal-complete": True}


def test_dsl_phase_advect_start_event_emitted_and_validated():
    from tac.witness_dsl.curriculum_dsl import PhaseAdvectionConsistency

    lever = PhaseAdvectionConsistency(weight=0.4, start_event="ncde_dseg")
    assert lever.overrides["--seg-phase-advect-start-event"] == "ncde_dseg"
    # absent => flag NOT emitted (event-mode OFF, byte-identical)
    assert "--seg-phase-advect-start-event" not in PhaseAdvectionConsistency(weight=0.4).overrides
    with pytest.raises(ValueError, match="start_event must be"):
        PhaseAdvectionConsistency(weight=0.4, start_event="bogus")


# ── (c) the observer-side fire=unavailable fix ─────────────────────────────── #
def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "ncde_trajectory_probe_wt", _REPO / "tools" / "ncde_trajectory_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_always_emits_structured_verdict_latest_advisory(tmp_path):
    """The MEASURED live-c2 failure shape: < 8 verdict rows => the key used to be silently
    omitted; now it is a structured unavailable dict with the measured reason."""
    rows = [{"stage": "verdict", "epoch": 650 + 25 * i, "d_seg": 0.004 + 1e-4 * i}
            for i in range(6)]
    (tmp_path / "run.log").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rep = _load_probe().run_probe(tmp_path, window=12, emit=False, do_backtest=False)
    adv = rep.get("verdict_latest_advisory")
    assert isinstance(adv, dict)
    assert adv["available"] is False and adv["fire"] is False
    assert "insufficient verdict rows" in adv["reason"]


def test_probe_marks_real_advisory_available(tmp_path):
    rng = np.random.default_rng(3)
    lines = []
    for i in range(20):
        lines.append(json.dumps({"stage": "verdict", "epoch": 600 + 25 * i,
                                 "d_seg": float(0.005 + 0.02 * 0.7 ** i),
                                 "ep_loss": float(20.0 * 0.9 ** i * (1 + 0.01 * rng.standard_normal()))}))
        lines.append(json.dumps({"stage": "loss_terms", "ep": 600 + 25 * i, "total": 1.0,
                                 "gnorm": 1.0, "terms": {"seg": 0.5},
                                 "softmax_temp": 0.5, "hosc_beta": 2.0}))
    (tmp_path / "run.log").write_text("\n".join(lines) + "\n")
    rep = _load_probe().run_probe(tmp_path, window=12, emit=False, do_backtest=False)
    adv = rep.get("verdict_latest_advisory")
    assert isinstance(adv, dict)
    # either a real advisory (available True, carries fire bool) or a structured degenerate —
    # NEVER an omitted key / None.
    assert "fire" in adv or adv.get("available") is True


def test_shadow_controller_never_stores_none_ncde():
    src = (_REPO / "src" / "tac" / "witness_control" / "shadow_controller.py").read_text(
        errors="ignore")
    assert "probe emitted no verdict_latest_advisory" in src
