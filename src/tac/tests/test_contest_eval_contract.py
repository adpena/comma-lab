# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES, contest_rate_term
from tac.contest_eval_contract import (
    CAMERA_SIZE_WH,
    PUBLIC_TEST_PAIR_COUNT,
    PUBLIC_TEST_RAW_OUTPUT_BYTES,
    SCORER_INPUT_SIZE_WH,
    SEQ_LEN,
    UPSTREAM_EVAL_CONTRACT_SCHEMA,
    UPSTREAM_SALIENCY_VERIFICATION_CONTRACT_SCHEMA,
    UPSTREAM_SCORE_ALLOCATION_CONTRACT_SCHEMA,
    build_saliency_verification_contract,
    build_score_allocation_contract,
    build_upstream_eval_contract,
)


def test_score_allocation_contract_matches_upstream_score_geometry() -> None:
    contract = build_score_allocation_contract()

    assert contract["schema"] == UPSTREAM_SCORE_ALLOCATION_CONTRACT_SCHEMA
    assert contract["rate"]["canonical_denominator_bytes"] == CONTEST_ORIGINAL_BYTES
    assert contract["rate"]["rate_price_per_archive_byte"] == contest_rate_term(1)
    assert contract["rate"]["raw_output_shape_bytes_are_not_rate_denominator"] == (
        PUBLIC_TEST_RAW_OUTPUT_BYTES
    )
    assert contract["pair_geometry"]["seq_len"] == SEQ_LEN
    assert contract["pair_geometry"]["public_test_pair_count"] == PUBLIC_TEST_PAIR_COUNT
    assert contract["pair_geometry"]["camera_size_wh"] == list(CAMERA_SIZE_WH)
    assert contract["segnet"]["input_size_wh"] == list(SCORER_INPUT_SIZE_WH)
    assert contract["segnet"]["frame_scope"] == "last_frame_only"
    assert contract["segnet"]["scored_frame_index_within_pair"] == 1
    assert contract["segnet"]["unscored_frame_index_within_pair"] == 0
    assert contract["posenet"]["frame_scope"] == "both_frames_in_pair"
    assert contract["posenet"]["scored_frame_indices_within_pair"] == [0, 1]
    assert contract["autograd_guard"]["upstream_rgb_to_yuv6_uses_inplace_clamp"] is True
    assert contract["saliency_verification"]["schema"] == (
        UPSTREAM_SALIENCY_VERIFICATION_CONTRACT_SCHEMA
    )


def test_upstream_eval_contract_verifies_actual_source_snippets() -> None:
    contract = build_upstream_eval_contract()

    assert contract["schema"] == UPSTREAM_EVAL_CONTRACT_SCHEMA
    assert contract["score_claim"] is False
    assert contract["promotion_eligible"] is False
    assert contract["contract_valid"] is True
    assert contract["blockers"] == []
    assert contract["canonical_rate_denominator_bytes"] == CONTEST_ORIGINAL_BYTES
    if contract["live_rate_denominator_bytes"] is not None:
        assert contract["live_rate_denominator_bytes"] == CONTEST_ORIGINAL_BYTES
    assert all(record["exists"] for record in contract["source_custody"])
    assert all(record["exists"] for record in contract["model_custody"])
    assert all(len(record["sha256"]) == 64 for record in contract["source_custody"])
    assert all(check["present"] for check in contract["implementation_snippet_checks"])


def test_upstream_eval_contract_fails_closed_when_snapshot_missing(tmp_path) -> None:
    contract = build_upstream_eval_contract(repo_root=tmp_path, upstream_dir="missing")

    assert contract["contract_valid"] is False
    assert contract["blockers"]
    assert any(blocker.startswith("missing_source:evaluate.py") for blocker in contract["blockers"])


def test_score_formula_is_not_raw_output_byte_denominator() -> None:
    contract = build_score_allocation_contract()

    assert PUBLIC_TEST_RAW_OUTPUT_BYTES == 3_662_409_600
    assert contract["rate"]["canonical_denominator_bytes"] != PUBLIC_TEST_RAW_OUTPUT_BYTES
    assert contract["rate"]["rate_price_per_archive_byte"] == pytest.approx(
        25.0 / 37_545_489
    )


def test_saliency_verification_contract_is_fail_closed_and_contest_compliant() -> None:
    contract = build_saliency_verification_contract()

    assert contract["schema"] == UPSTREAM_SALIENCY_VERIFICATION_CONTRACT_SCHEMA
    receiver_forbidden = contract["contest_compliance"]["receiver_must_not"]
    assert "load PoseNet/SegNet" in receiver_forbidden
    assert "adapt at eval time" in receiver_forbidden
    proof_names = {proof["name"] for proof in contract["required_numerical_proofs"]}
    assert {
        "yuv6_forward_parity",
        "yuv6_gradient_nonzero",
        "segnet_last_frame_asymmetry",
        "segnet_argmax_flip_exactness",
        "posenet_pair_six_axis_exactness",
        "rate_byte_price_exactness",
    } <= proof_names
    assert "score_claim=false until exact CPU/CUDA eval" in contract["budget_spend_requires"]
