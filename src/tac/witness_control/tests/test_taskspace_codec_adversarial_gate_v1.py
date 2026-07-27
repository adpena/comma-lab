from __future__ import annotations

import copy

import pytest

from tac.witness_control.taskspace_codec_adversarial_gate_v1 import (
    DIRECT_CONTROL,
    POST_EVAL,
    PRE_ENCODE,
    PRE_PROMOTION,
    PRE_PUBLIC_CLOSURE,
    PROGRAM_RESIDUAL,
    REQUEST_SCHEMA,
    CodecAdversarialGateError,
    review_request,
    verify_receipt,
    write_once_receipt,
)

SHA = "1" * 64
SHA2 = "2" * 64


def frontier() -> dict[str, object]:
    return {
        "target_score": 0.172,
        "selection_rule": "minimum contest authority or official leaderboard",
        "pointer_sha256": SHA,
    }


def request(boundary: str, representation: str, evidence: dict[str, object]) -> dict[str, object]:
    return {
        "schema": REQUEST_SCHEMA,
        "review_id": f"test-{boundary.lower()}",
        "boundary": boundary,
        "requested_representation": representation,
        "frontier": frontier(),
        "evidence": evidence,
    }


def direct_pre_encode() -> dict[str, object]:
    return {
        "actual_representation": "DIRECT_TASK_LAYERED",
        "pair_count": 600,
        "scorer_batch_size": 16,
        "provider_kind": "FRESH_SOURCE_RESIZE_PLANES",
        "source_plane_definition": "Yk=round(Resize(gt_fk))",
        "semantic_archive_bytes": 0,
        "semantic_archive_sha256": None,
        "semantic_archive_counted": False,
        "semantic_archive_reopened": False,
        "program_packet_bytes": 0,
        "program_packet_sha256": None,
        "factor_count": 0,
        "behavior_changing_factor_count": 0,
        "target_payload_embedded": False,
        "historical_payload_reused": False,
    }


def test_g57_direct_row_is_refused_as_selected_preimage_claim() -> None:
    receipt = review_request(request(PRE_ENCODE, PROGRAM_RESIDUAL, direct_pre_encode()))
    assert receipt["verdict"] == "REFUSE"
    assert receipt["admit_next_stage"] is False
    assert "REQUESTED_PROGRAM_ACTUAL_REPRESENTATION_MISMATCH" in receipt["failures"]
    assert "PROVIDER_IS_NOT_G49_SELECTED_PREIMAGE_PROGRAM" in receipt["failures"]
    assert "FRESH_SEMANTIC_ARCHIVE_NOT_COUNTED" in receipt["failures"]


def test_g57_direct_row_is_admitted_only_as_named_control() -> None:
    receipt = review_request(request(PRE_ENCODE, DIRECT_CONTROL, direct_pre_encode()))
    assert receipt["verdict"] == "ADMIT_CONTROL_ONLY"
    assert receipt["authority_mode"] == "RETROSPECTIVE_ONLY"
    assert receipt["admit_next_stage"] is False
    assert receipt["candidate_admission"] is False
    assert receipt["computed"]["selected_preimage_claim_allowed"] is False


def test_real_program_identity_passes_pre_encode() -> None:
    evidence = direct_pre_encode()
    evidence.update(
        {
            "actual_representation": "PROGRAM_RESIDUAL_LAYERED",
            "provider_kind": "G49_SELECTED_PREIMAGE_PROGRAM",
            "source_plane_definition": "G49_DECODE_SELECTED_PREIMAGE_PAIR",
            "semantic_archive_bytes": 133941,
            "semantic_archive_sha256": SHA,
            "semantic_archive_counted": True,
            "semantic_archive_reopened": True,
            "program_packet_bytes": 4096,
            "program_packet_sha256": SHA2,
            "factor_count": 7,
            "behavior_changing_factor_count": 7,
        }
    )
    receipt = review_request(request(PRE_ENCODE, PROGRAM_RESIDUAL, evidence))
    assert receipt["verdict"] == "ADMIT"
    assert receipt["admit_next_stage"] is False
    assert receipt["candidate_admission"] is False
    assert receipt["computed"]["selected_preimage_claim_allowed"] is True


def test_rate_only_frontier_fit_cannot_pass_pre_public_closure() -> None:
    evidence = {
        "pair_count": 600,
        "archive_bytes": 182220,
        "archive_sha256": SHA,
        "raw_sha256": None,
        "exact_components_available": False,
        "exact_component_source": "RATE_ONLY",
        "realized_through_R": False,
        "d_seg": None,
        "d_pose": None,
    }
    receipt = review_request(request(PRE_PUBLIC_CLOSURE, DIRECT_CONTROL, evidence))
    assert receipt["verdict"] == "REFUSE"
    assert "RATE_ONLY_ADMISSION_FORBIDDEN" in receipt["failures"]
    assert receipt["computed"]["rate_term"] == pytest.approx(0.12133281843792207)


def test_g57_full_components_fail_coupled_frontier() -> None:
    evidence = {
        "pair_count": 600,
        "archive_bytes": 182220,
        "archive_sha256": SHA,
        "raw_sha256": SHA2,
        "exact_components_available": True,
        "exact_component_source": "FULL_N600_FROZEN_SCORER_ON_EXACT_DECODED_PLANES",
        "realized_through_R": True,
        "d_seg": 0.17946555,
        "d_pose": 45.10546494,
    }
    receipt = review_request(request(PRE_PUBLIC_CLOSURE, DIRECT_CONTROL, evidence))
    assert receipt["verdict"] == "REFUSE"
    assert receipt["computed"]["score"] == pytest.approx(39.30593503092899)
    assert "COUPLED_SCORE_NOT_STRICTLY_BELOW_DYNAMIC_FRONTIER" in receipt["failures"]


def promotion_evidence(axis: str) -> dict[str, object]:
    return {
        "pair_count": 600,
        "archive_bytes": 182220,
        "archive_sha256": SHA,
        "runtime_tree_sha256": SHA2,
        "upstream_recursive_closure": True,
        "two_distinct_clean_roots": True,
        "fresh_decode_count_a": 1,
        "fresh_decode_count_b": 1,
        "resume_count_a": 0,
        "resume_count_b": 0,
        "raw_sha256_a": SHA,
        "raw_sha256_b": SHA,
        "raw_bytes_a": 3662409600,
        "raw_bytes_b": 3662409600,
        "frame_count_a": 1200,
        "frame_count_b": 1200,
        "axis": axis,
    }


def test_pre_promotion_refuses_advisory_axis_and_accepts_contest_axis() -> None:
    refused = review_request(request(PRE_PROMOTION, DIRECT_CONTROL, promotion_evidence("[macOS-CPU advisory]")))
    assert "PROMOTION_AXIS_NOT_CONTEST_CPU_OR_CUDA" in refused["failures"]
    admitted = review_request(request(PRE_PROMOTION, DIRECT_CONTROL, promotion_evidence("[contest-CPU]")))
    assert admitted["verdict"] == "ADMIT"
    assert admitted["admit_next_stage"] is False
    assert admitted["candidate_admission"] is False


def post_eval_evidence() -> dict[str, object]:
    return {
        "pair_count": 600,
        "archive_bytes": 182220,
        "archive_sha256": SHA,
        "d_seg": 0.17946555,
        "d_pose": 45.10546494,
        "axis": "[macOS-CPU advisory]",
        "verdict_scope": "FORMULATION:DIRECT_TASK_LAYERED_X264RGB_23K_PLUS_23K_Q2",
        "not_killed": ["PROGRAM_RESIDUAL_LAYERED", "G49_SELECTED_PREIMAGE"],
        "evidence_receipt_sha256": SHA2,
        "integration_hooks": {
            "sensitivity_map": False,
            "pareto_allocator": False,
            "bit_allocator": False,
            "autopilot": False,
            "continual_posterior": False,
            "probe_ledger": False,
        },
        "integration_blocker": None,
    }


def test_post_eval_refuses_orphan_signal_without_typed_blocker() -> None:
    evidence = post_eval_evidence()
    refused = review_request(request(POST_EVAL, DIRECT_CONTROL, evidence))
    assert "RESULT_SIGNAL_ORPHANED_WITHOUT_TYPED_BLOCKER" in refused["failures"]
    evidence["integration_blocker"] = "shared ledger owner has not landed append-only rows"
    admitted = review_request(request(POST_EVAL, DIRECT_CONTROL, evidence))
    assert admitted["verdict"] == "ADMIT"
    assert admitted["admit_next_stage"] is False
    assert admitted["candidate_admission"] is False
    assert admitted["computed"]["missing_integration_hooks"]


def test_receipt_seal_and_write_once(tmp_path) -> None:
    receipt = review_request(request(PRE_ENCODE, DIRECT_CONTROL, direct_pre_encode()))
    verify_receipt(receipt)
    path = tmp_path / "receipt.json"
    assert write_once_receipt(path, receipt) == path
    assert write_once_receipt(path, receipt) == path
    tampered = copy.deepcopy(receipt)
    tampered["verdict"] = "REFUSE"
    with pytest.raises(CodecAdversarialGateError, match="body SHA"):
        verify_receipt(tampered)
