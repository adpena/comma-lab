# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES, contest_rate_term
from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score
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
from tac.contest_oracle.constants import CONTEST_RATE_DENOM_BYTES, CONTEST_RATE_PER_BYTE
from tac.eval.auth_eval import FALLBACK_UNCOMPRESSED_SIZE, compute_final_score


def test_score_allocation_contract_matches_upstream_score_geometry() -> None:
    contract = build_score_allocation_contract()

    assert contract["schema"] == UPSTREAM_SCORE_ALLOCATION_CONTRACT_SCHEMA
    assert contract["rate"]["canonical_denominator_bytes"] == CONTEST_ORIGINAL_BYTES
    assert contract["rate"]["rate_price_per_archive_byte"] == contest_rate_term(1)
    assert contract["rate"]["raw_output_shape_bytes_are_not_rate_denominator"] == (
        PUBLIC_TEST_RAW_OUTPUT_BYTES
    )
    assert contract["distortion_reduction"]["authority"] == (
        "upstream/evaluate.py full-video pair-sum reduction"
    )
    assert contract["distortion_reduction"]["update_before_full_reduction_allowed"] is False
    assert "exact accumulation" in contract["distortion_reduction"]["gradient_acquisition_rule"]
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
    assert all(record["sha256_matches_expected"] for record in contract["model_custody"])
    assert {
        record["relative_path"]: record["expected_sha256"]
        for record in contract["model_custody"]
    } == {
        "models/posenet.safetensors": (
            "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576"
        ),
        "models/segnet.safetensors": (
            "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
        ),
    }
    assert all(check["present"] for check in contract["implementation_snippet_checks"])
    snippet_names = {check["name"] for check in contract["implementation_snippet_checks"]}
    assert {
        "posenet_full_video_pair_sum",
        "segnet_full_video_pair_sum",
        "batch_size_weighted_reduction",
        "posenet_full_video_mean",
        "segnet_full_video_mean",
    } <= snippet_names


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


def test_contest_formula_and_denominator_mirrors_agree_across_modules() -> None:
    contract = build_score_allocation_contract()
    denom = contract["rate"]["canonical_denominator_bytes"]
    archive_bytes = 178_493
    seg = 0.00123
    pose = 3.4e-5

    assert denom == CONTEST_ORIGINAL_BYTES
    assert denom == ORIGINAL_VIDEO_BYTES
    assert denom == CONTEST_RATE_DENOM_BYTES
    assert denom == FALLBACK_UNCOMPRESSED_SIZE
    assert contract["rate"]["rate_price_per_archive_byte"] == pytest.approx(CONTEST_RATE_PER_BYTE)
    assert contest_formula_score(
        seg_dist=seg,
        pose_dist=pose,
        archive_bytes=archive_bytes,
    ) == pytest.approx(
        compute_final_score(
            segnet_dist=seg,
            posenet_dist=pose,
            rate=archive_bytes / denom,
        )
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
