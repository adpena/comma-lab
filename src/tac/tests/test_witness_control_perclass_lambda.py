"""Tests for the binding-term-stall detector + per-class within-flip costates + the
shadow-controller overlay (task #315). Pure/deterministic; no run-dir IO except the
backtest smoke, which is skipped when the canonical logs are absent."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tac.witness_control.costate_estimator import (
    BINDING_STALL_UNIDENTIFIABLE,
    BINDING_TERM_STALL,
    MEASURED,
    NO_STALL,
    UNIDENTIFIABLE,
    binding_term_stall,
    per_class_within_flip_costates,
)

_REPO = Path(__file__).resolve().parents[3]


def _rows(seg_form, series):
    """series = list of (epoch, d_seg, d_pose, blob_bytes, ep_loss)."""
    return [{"stage": "verdict", "seg_form": seg_form, "epoch": e, "d_seg": ds,
             "d_pose": dp, "blob_bytes": by, "ep_loss": lo}
            for (e, ds, dp, by, lo) in series]


# ─────────────────── binding-term-stall detector ───────────────────
def test_stall_fires_frozen_dseg_descending_S():
    # the v5 signature: d_seg FROZEN (~1e-4 flat) while implied_S DESCENDS strongly.
    # d_pose falls 100→60 (pose noise term drives S down) while d_seg is pinned.
    series = [(100, 0.004010, 100.0, 82000, 90.0),
              (125, 0.004008, 90.0, 82000, 90.0),
              (150, 0.004012, 80.0, 82000, 90.0),
              (175, 0.004009, 70.0, 82000, 90.0),
              (200, 0.004011, 60.0, 82000, 90.0)]
    v = binding_term_stall(_rows("l7_softplus", series))
    assert v.classification == BINDING_TERM_STALL
    assert v.binding_term == "d_seg"
    assert v.fired()
    assert abs(v.d_seg_rel_slope) <= 3e-4
    assert v.s_rel_slope <= -5e-4


def test_stall_fires_frozen_dseg_descending_loss():
    # surrogate↔verdict decoupling: d_seg frozen while ep_loss keeps falling.
    series = [(600, 0.004070, 100.0, 82000, 120.0),
              (625, 0.004072, 100.0, 82000, 110.0),
              (650, 0.004069, 100.0, 82000, 100.0),
              (675, 0.004071, 100.0, 82000, 92.0),
              (700, 0.004070, 100.0, 82000, 85.0)]
    v = binding_term_stall(_rows("l7_softplus", series))
    assert v.classification == BINDING_TERM_STALL
    assert v.loss_rel_slope <= -5e-4


def test_no_stall_on_healthy_converging_dseg():
    # d_seg descending materially → NOT flat → NO_STALL (the false-positive the 2e-3
    # gate produced on real CE windows; the corrected 3e-4 gate rejects it).
    series = [(50, 0.16, 0.1, 82000, 400.0),
              (75, 0.13, 0.1, 82000, 360.0),
              (100, 0.11, 0.1, 82000, 330.0),
              (125, 0.095, 0.1, 82000, 310.0),
              (150, 0.085, 0.1, 82000, 300.0)]
    v = binding_term_stall(_rows("ce", series))
    assert v.classification == NO_STALL
    assert v.d_seg_rel_slope < -3e-4        # genuinely descending


def test_no_stall_on_genuine_plateau_everything_flat():
    # d_seg flat AND S/loss flat = a real plateau (the scalar PLATEAU is correct).
    series = [(600, 0.00407, 100.0, 82000, 90.0),
              (625, 0.00407, 100.0, 82000, 90.0),
              (650, 0.00407, 100.0, 82000, 90.0),
              (675, 0.00407, 100.0, 82000, 90.0),
              (700, 0.00407, 100.0, 82000, 90.0)]
    v = binding_term_stall(_rows("l7_softplus", series))
    assert v.classification == NO_STALL


def test_no_stall_on_rising_dseg_erosion():
    # rising d_seg (erosion) is NOT flat → the canonical DIVERGING_ERASING rule owns
    # it, this detector stays out (NO_STALL from the flat gate).
    series = [(300, 0.0040, 100.0, 82000, 30.0),
              (325, 0.0045, 100.0, 82000, 29.0),
              (350, 0.0052, 100.0, 82000, 28.0),
              (375, 0.0060, 100.0, 82000, 27.0),
              (400, 0.0066, 100.0, 82000, 26.0)]
    v = binding_term_stall(_rows("tau_softplus", series))
    assert v.classification == NO_STALL
    assert v.d_seg_rel_slope > 3e-4         # rising


def test_stall_unidentifiable_too_few_rows():
    v = binding_term_stall(_rows("ce", [(100, 0.004, 100.0, 82000, 90.0)]))
    assert v.classification == BINDING_STALL_UNIDENTIFIABLE


def test_stall_unidentifiable_no_nonbinding_signal():
    # rows carry d_seg only (no implied_S recoverable, no ep_loss) → cannot tell a
    # stall from a plateau → honest UNIDENTIFIABLE, never fabricated.
    rows = [{"stage": "verdict", "seg_form": "ce", "epoch": e, "d_seg": 0.00407}
            for e in (100, 125, 150, 175, 200)]
    v = binding_term_stall(rows)
    assert v.classification == BINDING_STALL_UNIDENTIFIABLE
    assert "no non-binding signal" in v.reason


def test_stall_only_uses_same_stage_window():
    # the last stage is l7 (flat+loss-descending stall); prior ce rows are descending
    # and must NOT dilute the l7 slope.
    ce = _rows("ce", [(50, 0.16, 0.1, 82000, 400.0), (75, 0.12, 0.1, 82000, 360.0)])
    l7 = _rows("l7_softplus", [(600, 0.00407, 100.0, 82000, 120.0),
                               (625, 0.00407, 100.0, 82000, 110.0),
                               (650, 0.00407, 100.0, 82000, 100.0),
                               (675, 0.00407, 100.0, 82000, 92.0)])
    v = binding_term_stall(ce + l7)
    assert v.stage == "l7_softplus"
    assert v.classification == BINDING_TERM_STALL


def test_level_dominant_term_pose_when_dpose_huge():
    # v2_attrclean phenomenology: S dominated by pose (d_pose~100), binding term still d_seg.
    series = [(e, 0.00407, 100.0, 82000, 90.0) for e in (600, 625, 650, 675, 700)]
    series = [(s[0], s[1], s[2], s[3], s[4] - i * 5.0) for i, s in enumerate(series)]  # loss falls
    v = binding_term_stall(_rows("l7_softplus", series))
    assert v.level_dominant_term == "d_pose"     # sqrt(10*100)=31.6 >> 100*0.00407=0.407
    assert v.binding_term == "d_seg"             # structural binding term is still d_seg


# ─────────────────── per-class within-flip costates ───────────────────
def test_per_class_unidentifiable_when_absent():
    rows = _rows("ce", [(e, 0.02, 0.1, 82000, 100.0) for e in (100, 125, 150)])
    est = per_class_within_flip_costates(rows, "ce")
    assert est.status == UNIDENTIFIABLE
    assert "handoff_readiness" in " ".join(est.evidence)


def test_per_class_measured_identifies_worst_class():
    # class 1 (Lane) within_flip RISING (worst), class 0 (Road) descending.
    def mk(e, lane_wf, road_wf):
        return {"stage": "verdict", "seg_form": "ce", "epoch": e,
                "d_seg": 0.02, "d_pose": 0.1, "blob_bytes": 82000, "ep_loss": 100.0,
                "per_class": {"0": {"within_flip": road_wf, "part_frac": 0.23},
                              "1": {"within_flip": lane_wf, "part_frac": 0.006}}}
    rows = [mk(100, 0.30, 0.05), mk(125, 0.34, 0.045), mk(150, 0.39, 0.04)]
    est = per_class_within_flip_costates(rows, "ce")
    assert est.status == MEASURED
    assert "worst class = 1" in est.method     # Lane flip rising = worst


def test_per_class_reads_parallel_dict_shape():
    # tolerant of the parallel within_flip/part_frac dict shape.
    def mk(e, lane_wf):
        return {"stage": "verdict", "seg_form": "tau_softplus", "epoch": e,
                "d_seg": 0.004, "d_pose": 100.0, "blob_bytes": 82000, "ep_loss": 30.0,
                "within_flip": {"0": 0.05, "1": lane_wf}, "part_frac": {"0": 0.23, "1": 0.006}}
    rows = [mk(300, 0.20), mk(325, 0.25), mk(350, 0.30)]
    est = per_class_within_flip_costates(rows, "tau_softplus")
    assert est.status == MEASURED
    assert est.value is not None and est.value > 0.0   # worst class rising → positive dS/dep


# ─────────────────── shadow-controller overlay ───────────────────
def test_shadow_classifier_overlay_flags_deadlock():
    from tac.witness_control import shadow_controller as sc
    # frozen d_seg + descending loss over an l7 window → scalar says converging/plateau,
    # overlay must OVERRIDE to BINDING_TERM_STALL and preserve the scalar label.
    verdicts = _rows("l7_softplus", [(600, 0.004070, 100.0, 82000, 120.0),
                                     (625, 0.004072, 100.0, 82000, 110.0),
                                     (650, 0.004069, 100.0, 82000, 100.0),
                                     (675, 0.004071, 100.0, 82000, 92.0),
                                     (700, 0.004070, 100.0, 82000, 85.0)])
    inp = sc.RunInputs(run_dir=Path("."), verdicts=verdicts, stage_rows={}, flags={})
    out = sc._classify(inp)
    assert out is not None
    assert out["classification"] == BINDING_TERM_STALL
    assert out["scalar_classification"] in ("plateau", "converging")
    assert "BINDING-TERM STALL" in out["recommendation"]
    assert out["binding_stall"]["classification"] == BINDING_TERM_STALL


def test_shadow_overlay_absent_on_healthy_descent():
    from tac.witness_control import shadow_controller as sc
    verdicts = _rows("ce", [(50, 0.16, 0.1, 82000, 400.0), (75, 0.13, 0.1, 82000, 360.0),
                            (100, 0.11, 0.1, 82000, 330.0), (125, 0.095, 0.1, 82000, 310.0),
                            (150, 0.085, 0.1, 82000, 300.0)])
    inp = sc.RunInputs(run_dir=Path("."), verdicts=verdicts, stage_rows={}, flags={})
    out = sc._classify(inp)
    assert out is not None
    assert out["classification"] != BINDING_TERM_STALL
    assert out["binding_stall"]["classification"] == NO_STALL


# ─────────────────── backtest smoke (real logs; skip if absent) ───────────────────
def test_backtest_tool_runs_on_real_logs():
    tool = _REPO / "tools" / "witness_control_binding_stall_backtest.py"
    canon = _REPO / "experiments" / "results" / "levelset_n600_v2_attrclean_20260630T194549Z"
    if not (canon / "run.log").is_file():
        pytest.skip("canonical run log absent")
    spec = importlib.util.spec_from_file_location("_bt", tool)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rep = mod.backtest_run(canon)
    # the l7 flat-binding stall (ep675/700/725) is a stable committed-log fixture.
    assert rep["caught_count"] >= 1
    assert any(ep in rep["caught_epochs"] for ep in (675, 700, 725))
