"""NO-FAKE behavioral tests for the vehicle-agnostic harvest-evidence self-driving loop.

Proves the loop works IDENTICALLY for V1 HiNeRV, V2 SNeRV, and V3 atom candidates —
only the per-vehicle hard_fail_checks differ; the receipt / CandidateActionEvaluation /
binding-constraint / decision-ladder are shared.
"""

from __future__ import annotations

import math

import pytest

from tac.optimization.evaluator_action_waterfill import contest_score
from tac.optimization.harvest_evidence import (
    CONTINUE_PREP_CUDA,
    CONTINUE_TRAINING,
    HARD_FAIL,
    INSPECT_BINDING_CONSTRAINT,
    PATCH_BRIDGE,
    VEHICLE_ATOM,
    VEHICLE_HI_NERV,
    VEHICLE_SNERV,
    apply_campaign_decision,
    binding_constraint,
    build_candidate_action_evaluation_row,
    build_harvest_receipt,
)

_FRONTIER = 0.1919853363  # contest-CPU frontier to beat.


def _receipt(vehicle, **over):
    base = {
        "vehicle": vehicle,
        "run_dir": "/Volumes/VertigoDataTier/pact/run",
        "checkpoint_label": "ep250",
        "checkpoint_sha256": "ck" + vehicle,
        "export_status": "passed",
        "archive_path": "/x/archive.zip",
        "archive_sha256": "arch" + vehicle,
        "archive_bytes": 120_000,
        "sidecar_exported": False,
        "pay_rent_gate_active": True,
        "inflate_status": "passed",
        "evaluate_status": "passed",
        "result_json_path": "/x/eval.json",
    }
    base.update(over)
    return build_harvest_receipt(**base)


def _cand(vehicle, d_seg, d_pose, bytes_, kind="backend_only"):
    return build_candidate_action_evaluation_row(
        vehicle=vehicle,
        candidate_kind=kind,
        candidate_archive_sha256="cand" + vehicle,
        candidate_d_seg=d_seg,
        candidate_d_pose=d_pose,
        candidate_bytes=bytes_,
        base_archive_sha256="frontier",
        base_score=_FRONTIER,
    )


# ---- agnostic primitives ----

def test_receipt_schema_and_idempotency_all_vehicles():
    for v in (VEHICLE_HI_NERV, VEHICLE_SNERV, VEHICLE_ATOM):
        r = _receipt(v)
        assert r["schema"] == "checkpoint_harvest_receipt.v1"
        assert r["vehicle"] == v
        assert r["promotion_eligible"] is False
        # idempotency key is deterministic + vehicle-distinct
        assert v in r["idempotency_key"] and "arch" + v in r["idempotency_key"]
    assert _receipt(VEHICLE_HI_NERV)["idempotency_key"] != _receipt(VEHICLE_SNERV)["idempotency_key"]


def test_invalid_vehicle_rejected():
    with pytest.raises(ValueError):
        build_harvest_receipt(
            vehicle="not_a_vehicle", run_dir="x", checkpoint_label="e", checkpoint_sha256="s",
            export_status="passed", archive_path=None, archive_sha256=None, archive_bytes=None,
            sidecar_exported=False, pay_rent_gate_active=True, inflate_status="passed",
            evaluate_status="passed", result_json_path=None,
        )


def test_candidate_score_is_exact_contest_score_not_handrolled():
    # The row's candidate_score MUST equal contest_score (fails if someone hand-approximates).
    row = _cand(VEHICLE_HI_NERV, 0.0015, 3.0e-5, 130_000)
    assert row["candidate_score"] == contest_score(0.0015, 3.0e-5, 130_000)
    assert row["delta_score_total"] == row["candidate_score"] - _FRONTIER


def test_pays_rent_iff_beats_frontier():
    # A candidate strictly below the frontier pays rent; one above does not.
    # Construct a clearly-better candidate (tiny seg/pose, small bytes).
    good = _cand(VEHICLE_SNERV, 0.0005, 1.0e-6, 40_000)
    bad = _cand(VEHICLE_SNERV, 0.02, 1.0e-2, 400_000)
    assert good["candidate_score"] < _FRONTIER and good["pays_rent"] is True
    assert bad["candidate_score"] > _FRONTIER and bad["pays_rent"] is False
    assert good["verdict"] == "beats_frontier" and bad["verdict"] == "above_frontier"


def test_binding_constraint_identifies_dominant_term():
    # High bytes -> rate binding; high d_seg -> seg binding; high d_pose -> pose binding.
    assert binding_constraint(1e-6, 1e-9, 30_000_000)["binding_constraint"] == "rate"
    assert binding_constraint(0.05, 1e-9, 1000)["binding_constraint"] == "seg"
    assert binding_constraint(1e-6, 5.0, 1000)["binding_constraint"] == "pose"
    bc = binding_constraint(0.001, 3e-5, 120_000)
    assert math.isclose(bc["total"], bc["seg_term"] + bc["pose_term"] + bc["rate_term"])
    assert math.isclose(sum(bc["shares"].values()), 1.0, rel_tol=1e-9)


# ---- the decision ladder, per vehicle ----

def test_hi_nerv_muon_before_stage_8_hard_fails():
    d = apply_campaign_decision(
        receipt=_receipt(VEHICLE_HI_NERV),
        candidate_eval=_cand(VEHICLE_HI_NERV, 0.001, 3e-5, 120_000),
        frontier_score=_FRONTIER,
        hard_fail_checks=[("muon_active_before_stage_8", True)],
    )
    assert d["decision"] == HARD_FAIL and d["reason"] == "muon_active_before_stage_8"
    assert d["auto_kill"] is False


def test_snerv_hf_restorer_diverged_hard_fails():
    d = apply_campaign_decision(
        receipt=_receipt(VEHICLE_SNERV),
        candidate_eval=_cand(VEHICLE_SNERV, 0.001, 3e-5, 120_000),
        frontier_score=_FRONTIER,
        hard_fail_checks=[("hf_restorer_diverged", True), ("source_forward_unproven", False)],
    )
    assert d["decision"] == HARD_FAIL and d["reason"] == "hf_restorer_diverged"


def test_atom_scorer_effect_not_survived_hard_fails():
    d = apply_campaign_decision(
        receipt=_receipt(VEHICLE_ATOM),
        candidate_eval=_cand(VEHICLE_ATOM, 0.001, 3e-5, 120_000, kind="pose_comp_atom"),
        frontier_score=_FRONTIER,
        hard_fail_checks=[("scorer_effect_not_survived", True)],
    )
    assert d["decision"] == HARD_FAIL and d["reason"] == "scorer_effect_not_survived"


def test_sidecar_exported_universal_hard_fail():
    d = apply_campaign_decision(
        receipt=_receipt(VEHICLE_HI_NERV, sidecar_exported=True),
        candidate_eval=_cand(VEHICLE_HI_NERV, 0.001, 3e-5, 120_000),
        frontier_score=_FRONTIER,
    )
    assert d["decision"] == HARD_FAIL and d["reason"] == "sidecar_exported_without_rent"


def test_evaluate_failure_patches_bridge_not_kill():
    d = apply_campaign_decision(
        receipt=_receipt(VEHICLE_SNERV, evaluate_status="failed"),
        candidate_eval=_cand(VEHICLE_SNERV, 0.001, 3e-5, 120_000),
        frontier_score=_FRONTIER,
    )
    assert d["decision"] == PATCH_BRIDGE and d["reason"] == "exact_eval_bridge_failed"


def test_beats_frontier_prep_cuda_all_vehicles():
    for v in (VEHICLE_HI_NERV, VEHICLE_SNERV, VEHICLE_ATOM):
        d = apply_campaign_decision(
            receipt=_receipt(v),
            candidate_eval=_cand(v, 0.0005, 1e-6, 40_000),  # clearly beats frontier
            frontier_score=_FRONTIER,
        )
        assert d["decision"] == CONTINUE_PREP_CUDA
        assert "binding_constraint" in d


def test_high_score_inspects_never_auto_kills():
    # An early/compressed checkpoint with a bad score must NOT auto-kill — it inspects.
    d = apply_campaign_decision(
        receipt=_receipt(VEHICLE_HI_NERV),
        candidate_eval=_cand(VEHICLE_HI_NERV, 0.02, 1e-2, 400_000),  # far above frontier
        frontier_score=_FRONTIER,
    )
    assert d["decision"] == INSPECT_BINDING_CONSTRAINT
    assert d["auto_kill"] is False
    assert d["binding_constraint"]["binding_constraint"] in ("seg", "pose", "rate")


def test_directional_improvement_continues_training():
    # Between frontier+directional and frontier: not beating frontier, but better than a
    # worse baseline -> continue training.
    baseline = 0.25
    cand = _cand(VEHICLE_HI_NERV, 0.0018, 1e-5, 30_000)  # ~0.21: above frontier, below baseline
    # Ensure it's above frontier but >0.001 below the baseline.
    assert cand["candidate_score"] > _FRONTIER
    assert cand["candidate_score"] <= baseline - 0.001
    d = apply_campaign_decision(
        receipt=_receipt(VEHICLE_HI_NERV),
        candidate_eval=cand,
        frontier_score=_FRONTIER,
        baseline_score=baseline,
    )
    assert d["decision"] == CONTINUE_TRAINING and d["reason"] == "directionally_promising"
