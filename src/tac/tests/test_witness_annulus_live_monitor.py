# SPDX-License-Identifier: MIT
"""Unit tests for tools/witness_annulus_live_monitor.py -- the LIVE annulus monitor.

Covers the FREE, render-free surfaces: log parsing (verdict + loss_terms extraction on a
synthetic log fixture) and every WHY-narration classifier rule (boundary-jitter vs
structural-miss, per-class stuck-class naming, convergence-direction sign, stage-transition
detection, spike-guard-deadlock detection, margin trend). No rendering, no torch, no MLX.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "witness_annulus_live_monitor", _REPO / "tools" / "witness_annulus_live_monitor.py")
mon = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mon)


# ---------------------------------------------------------------------------
# synthetic log fixture (mirrors the real mod32cap stdout schema).
# ---------------------------------------------------------------------------
def _verdict(ep, seg_form, d_seg, ep_loss=100.0, accepted_frac=1.0, frozen=False, skipped=0):
    return json.dumps({
        "stage": "verdict", "epoch": ep, "seg_form": seg_form, "d_seg": d_seg,
        "d_pose": 123.4, "blob_bytes": 81000, "implied_S": 35.7, "ep_loss": ep_loss,
        "ts": "2026-07-06T19:41:36Z", "accepted_frac": accepted_frac, "weights_stepped": True,
        "accepted_batches": 75, "skipped_batches": skipped, "frozen_epoch": frozen, "async": True})


def _loss_terms(ep, gnorm, hosc_beta, softmax_temp, spike_skipped=False):
    return json.dumps({
        "stage": "loss_terms", "ep": ep, "accum_batch": 0,
        "terms": {"seg": 0.39, "pose": 0.0, "length": 8e-6}, "total": 0.39, "sum_terms": 0.39,
        "sum_minus_total": 0.0, "gnorm": gnorm, "spike_skipped": spike_skipped,
        "accepted_frac": 0.0, "weights_stepped": True, "hosc_beta": hosc_beta,
        "softmax_temp": softmax_temp})


SYNTHETIC_LOG = "\n".join([
    "some non-json banner line",
    '{"stage": "checkpoint", "kind": "best", "epoch": 175, "d_seg": 0.005121}',  # ignored stage
    _verdict(275, "ce", 0.004682),
    _loss_terms(319, 1.71, 1.955, 0.7816),
    _loss_terms(320, 1.42, 1.958, 0.7804),
    _verdict(300, "tau_softplus", 0.004571, ep_loss=29.092),
    _loss_terms(321, 1.63, 1.961, 0.7791),
    "not-a-json-line {but has a brace",
])


def test_parse_log_rows_extracts_verdicts_and_loss_terms():
    verdicts, loss_terms = mon.parse_log_rows(SYNTHETIC_LOG)
    assert len(verdicts) == 2
    assert len(loss_terms) == 3
    assert verdicts[0]["epoch"] == 275 and verdicts[0]["seg_form"] == "ce"
    assert verdicts[1]["epoch"] == 300 and verdicts[1]["seg_form"] == "tau_softplus"
    assert loss_terms[0]["gnorm"] == 1.71 and loss_terms[-1]["hosc_beta"] == 1.961


def test_parse_log_rows_ignores_junk_and_other_stages():
    verdicts, loss_terms = mon.parse_log_rows("garbage\n{}\n{\"stage\": \"other\", \"x\": 1}\n")
    assert verdicts == [] and loss_terms == []


def test_d_seg_trajectory_shape():
    verdicts, _ = mon.parse_log_rows(SYNTHETIC_LOG)
    traj = mon.d_seg_trajectory(verdicts)
    assert [r["epoch"] for r in traj] == [275, 300]
    assert traj[1]["seg_form"] == "tau_softplus"
    assert traj[0]["d_seg"] == 0.004682


# ---------------------------------------------------------------------------
# residual classifier.
# ---------------------------------------------------------------------------
def test_classify_residual_boundary_jitter():
    label, text = mon.classify_residual(annulus_flip_mass_share=0.95, interior_flip_frac=1e-5)
    assert label == "boundary_jitter"
    assert "BOUNDARY JITTER" in text and "separatrix" in text


def test_classify_residual_structural_miss():
    label, text = mon.classify_residual(annulus_flip_mass_share=0.95, interior_flip_frac=5e-3)
    assert label == "structural_miss"
    assert "STRUCTURAL" in text


def test_classify_residual_mixed():
    label, _ = mon.classify_residual(annulus_flip_mass_share=0.5, interior_flip_frac=1e-5)
    assert label == "mixed"


def test_classify_residual_structural_takes_precedence_over_high_mass():
    # interior above the floor => structural, even if mass share is high.
    label, _ = mon.classify_residual(annulus_flip_mass_share=0.99, interior_flip_frac=2e-3)
    assert label == "structural_miss"


# ---------------------------------------------------------------------------
# dominant stuck class (canonical comma10k naming).
# ---------------------------------------------------------------------------
def test_dominant_stuck_class_names_lane():
    per_class = {0: 0.001, 1: 0.02, 2: 0.003, 3: 0.0, 4: 0.0005}
    idx, name, val = mon.dominant_stuck_class(per_class)
    assert idx == 1 and name == "Lane" and val == 0.02


def test_dominant_stuck_class_handles_string_keys_and_empty():
    idx, name, _ = mon.dominant_stuck_class({"0": 0.01, "3": 0.05})  # JSON round-trip keys
    assert idx == 3 and name == "Movable"
    assert mon.dominant_stuck_class({}) == (-1, "none", 0.0)


# ---------------------------------------------------------------------------
# convergence direction (sign of dV/dEpoch).
# ---------------------------------------------------------------------------
def test_convergence_direction_signs():
    assert mon.convergence_direction(-4e-6)[0] == "tightening"
    assert mon.convergence_direction(+4e-6)[0] == "widening"
    assert mon.convergence_direction(0.0)[0] == "plateau"
    assert mon.convergence_direction(1e-9)[0] == "plateau"  # inside deadband
    assert mon.convergence_direction(None)[0] == "unknown"


# ---------------------------------------------------------------------------
# stage-transition detection.
# ---------------------------------------------------------------------------
def test_detect_stage_transition_fires_on_change():
    verdicts, _ = mon.parse_log_rows(SYNTHETIC_LOG)
    tr = mon.detect_stage_transition(verdicts)
    assert tr == {"from": "ce", "to": "tau_softplus", "epoch": 300}


def test_detect_stage_transition_none_when_same():
    same = "\n".join([_verdict(250, "ce", 0.005), _verdict(275, "ce", 0.0048)])
    v, _ = mon.parse_log_rows(same)
    assert mon.detect_stage_transition(v) is None
    assert mon.detect_stage_transition([]) is None


# ---------------------------------------------------------------------------
# training health + spike-guard-deadlock signature.
# ---------------------------------------------------------------------------
def test_training_health_healthy():
    _, loss_terms = mon.parse_log_rows(SYNTHETIC_LOG)
    latest_v = {"frozen_epoch": False, "ep_loss": 29.092, "accepted_frac": 1.0}
    health, text, is_deadlock = mon.training_health(loss_terms, latest_v)
    assert is_deadlock is False
    assert "HEALTHY" in text
    assert health["gnorm_min"] == 1.42 and health["gnorm_max"] == 1.71
    assert 0.0 <= health["hosc_beta_anneal_pct"] <= 1.0
    assert health["softmax_temp"] == 0.7791  # latest loss_terms row's temp
    assert health["hosc_beta"] == 1.961


def test_training_health_deadlock_frozen_epoch():
    lt = [json.loads(_loss_terms(400, 0.0, 3.0, 0.1))]
    latest_v = {"frozen_epoch": True, "ep_loss": 0.0, "accepted_frac": 0.0}
    health, text, is_deadlock = mon.training_health(lt, latest_v)
    assert is_deadlock is True
    assert "DEADLOCK" in text
    assert "frozen_epoch=true" in text and "ep_loss==0" in text


def test_training_health_deadlock_from_verdict_only_not_loss_terms_accepted_frac():
    # loss_terms accepted_frac==0.0 at accum_batch 0 must NOT by itself trigger deadlock.
    lt = [json.loads(_loss_terms(320, 1.5, 1.9, 0.78))]  # accepted_frac 0.0 inside
    latest_v = {"frozen_epoch": False, "ep_loss": 300.0, "accepted_frac": 1.0}
    _, _, is_deadlock = mon.training_health(lt, latest_v)
    assert is_deadlock is False


def test_training_health_spike_storm():
    lt = [json.loads(_loss_terms(e, 1.5, 1.9, 0.78, spike_skipped=True)) for e in range(10)]
    latest_v = {"frozen_epoch": False, "ep_loss": 300.0, "accepted_frac": 1.0}
    health, text, is_deadlock = mon.training_health(lt, latest_v)
    assert health["spike_skipped_rate"] == 1.0
    assert is_deadlock is True and "DEADLOCK" in text


def test_training_health_empty_inputs():
    health, text, is_deadlock = mon.training_health([], None)
    assert is_deadlock is False
    assert health["gnorm_min"] is None


# ---------------------------------------------------------------------------
# margin convergence text.
# ---------------------------------------------------------------------------
def test_margin_convergence_rising():
    txt = mon.margin_convergence_text([0.1, 0.3], [0.5, 0.9])
    assert "RISING" in txt


def test_margin_convergence_falling():
    txt = mon.margin_convergence_text([0.3, 0.1], [0.9, 0.4])
    assert "FALLING" in txt


def test_margin_convergence_insufficient():
    assert "insufficient" in mon.margin_convergence_text([], []).lower()


# ---------------------------------------------------------------------------
# build_narration integration (pure; no render).
# ---------------------------------------------------------------------------
def test_build_narration_contains_all_blocks():
    metrics = {
        "overall_d_seg": 0.004571,
        "threshold": {
            "annulus_flip_frac": 0.03, "interior_flip_frac": 1e-5,
            "annulus_flip_mass_share": 0.95, "annulus_area_frac": 0.02,
            "per_class_annulus_flip_frac": {0: 0.001, 1: 0.02, 2: 0.003, 3: 0.0, 4: 0.0},
            "annulus_margin": {"p10": 0.1, "p50": 0.5},
        },
    }
    rates = {"annulus_flip_frac_rate": -4e-6, "margin_p10_curve": [0.05, 0.1],
             "margin_p50_curve": [0.3, 0.5]}
    verdicts, _ = mon.parse_log_rows(SYNTHETIC_LOG)
    narration = mon.build_narration(
        run_label="run_x", epoch=300, seg_form="tau_softplus", latest_metrics=metrics,
        rates=rates, verdicts=verdicts,
        health_text="training HEALTHY: gnorm in [1.42, 1.71].",
        margin_text=mon.margin_convergence_text(rates["margin_p10_curve"], rates["margin_p50_curve"]),
        stage_transition={"from": "ce", "to": "tau_softplus", "epoch": 300},
        verdict_pairs=16, advisory_subset=True)
    assert "BOUNDARY JITTER" in narration
    assert "Lane(cls1)" in narration and "thin-dash" in narration
    assert "TIGHTENING" in narration
    assert "STAGE TRANSITION" in narration
    assert "training HEALTHY" in narration
    assert "0.19110 UNMOVED" in narration
    assert "NON-PROMOTABLE" in narration


def test_build_narration_handles_no_render():
    narration = mon.build_narration(
        run_label="run_x", epoch=300, seg_form="ce", latest_metrics=None, rates=None,
        verdicts=[], health_text="training HEALTHY.", margin_text="n/a",
        stage_transition=None, verdict_pairs=0, advisory_subset=False)
    assert "no annulus render available" in narration
    assert "0.19110 UNMOVED" in narration
