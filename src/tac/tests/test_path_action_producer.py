# SPDX-License-Identifier: MIT
"""Tests for path-coded ActionEffect candidate production."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tac.analysis.path_action_producer import (
    BLOCKER_PATH_ACTION_INFLATE_MISSING,
    BLOCKER_PATH_ACTION_PARSEBACK_MISSING,
    BLOCKER_PATH_SUPPORT_NOT_BIRTH,
    BLOCKER_PATH_TRAJECTORY_NO_RECEIVER_PROOF,
    RENT_BLOCKER_BASE_ARCHIVE_SHA_MISSING,
    RENT_EVALUATION_SCHEMA,
    build_path_action_candidates_from_arrays,
    build_pose_temporal_path_candidates_from_arrays,
    build_selector_temporal_path_candidates_from_rows,
    decode_path_tube_payload,
    path_tube_support_from_mask,
    rdp_simplify,
    selector_sequence_encoding_comparison,
    support_mask_sha256,
)

_BASE_SHA = "9" * 64


def _hard_region_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = np.zeros((1, 24, 32), dtype=np.uint8)
    argmax = np.zeros_like(target)
    target[:, 4:20, 3:29] = 4
    argmax[:, 4:20, 3:29] = 4
    argmax[:, 11:14, 8:24] = 2
    margin = np.zeros((1, 24, 32), dtype=np.float32)
    margin[:, 11:14, 8:24] = -0.25
    pair_indices = np.asarray([17], dtype=np.int64)
    return target, argmax, margin, pair_indices


def test_path_tube_support_payload_round_trips_exact_support_identity() -> None:
    mask = np.zeros((24, 32), dtype=bool)
    mask[10:13, 4:26] = True
    mask[8, 20] = True
    mask[11, 12] = False

    support = path_tube_support_from_mask(
        mask,
        pair_index=17,
        frame_index=1,
        target_class=4,
        epsilon=0.75,
    )
    decoded = support.decode_mask()
    assert np.array_equal(decoded, mask)
    payload_decoded = decode_path_tube_payload(support.encode_payload())
    assert payload_decoded.pair_index == 17
    assert payload_decoded.frame_index == 1
    assert payload_decoded.target_class == 4
    assert np.array_equal(payload_decoded.decode_mask(), mask)

    row = support.as_dict()
    assert row["support_encoding"] == "path_tube"
    assert row["archive_executable"] is True
    assert row["support_cardinality"] == int(np.count_nonzero(mask))
    assert row["support_sha256"] == support_mask_sha256(mask)
    assert row["support_encoded_bytes"] > 0


def test_path_action_producer_uses_unsolved_component_not_full_class_mask() -> None:
    target, argmax, margin, pair_indices = _hard_region_arrays()
    result = build_path_action_candidates_from_arrays(
        target_labels_bhw=target,
        candidate_argmax_bhw=argmax,
        target_margin_bhw=margin,
        pair_indices=pair_indices,
        target_class=4,
        base_archive_bytes=1000,
        old_d_seg=0.2,
        old_d_pose=0.3,
    )

    support = result["path_action_candidates"][0]["support"]
    unsolved_count = int(np.count_nonzero((target[0] == 4) & (argmax[0] != 4)))
    full_class_count = int(np.count_nonzero(target[0] == 4))
    assert support["support_cardinality"] == unsolved_count
    assert support["support_cardinality"] < full_class_count
    assert support["support_sha256"] == result["action_effects"][0].support_sha256


def test_path_action_effect_is_rate_priced_non_promotable_and_launch_blocked() -> None:
    target, argmax, margin, pair_indices = _hard_region_arrays()
    result = build_path_action_candidates_from_arrays(
        target_labels_bhw=target,
        candidate_argmax_bhw=argmax,
        target_margin_bhw=margin,
        pair_indices=pair_indices,
        target_class=4,
        base_archive_bytes=1000,
        old_d_seg=0.2,
        old_d_pose=0.3,
    )

    effect = result["action_effects"][0]
    support = result["path_action_candidates"][0]["support"]
    assert effect.promotion_eligible is False
    assert effect.parseback_survived is False
    assert effect.inflate_survived is False
    assert effect.wrong_to_target == 0
    assert effect.delta_bytes == support["support_encoded_bytes"]
    assert effect.value_per_byte is not None
    assert effect.delta_score_total is not None and effect.delta_score_total > 0.0
    assert BLOCKER_PATH_SUPPORT_NOT_BIRTH in effect.blockers

    queue_row = result["candidate_queue"][0]
    assert queue_row["ready_for_exact_eval_dispatch"] is False
    assert queue_row["promotion_eligible"] is False
    assert queue_row["support_encoding"] == "path_tube"
    assert queue_row["support_sha256"] == support["support_sha256"]
    assert queue_row["byte_cost"] == support["support_encoded_bytes"]
    assert BLOCKER_PATH_ACTION_PARSEBACK_MISSING in queue_row["blockers"]
    assert BLOCKER_PATH_ACTION_INFLATE_MISSING in queue_row["blockers"]
    assert BLOCKER_PATH_SUPPORT_NOT_BIRTH in queue_row["blockers"]


def test_rdp_simplify_keeps_endpoints_and_reduces_near_line() -> None:
    points = [(4, x) for x in range(12)]
    assert rdp_simplify(points, epsilon=0.1) == [(4, 0), (4, 11)]


def test_pose_temporal_path_candidate_is_rate_priced_and_non_promotable() -> None:
    result = build_pose_temporal_path_candidates_from_arrays(
        pair_indices=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        pose_residuals=np.asarray([0.0, 0.1, 0.2, 0.1, 0.0], dtype=np.float32),
        base_archive_bytes=2000,
        old_d_seg=0.2,
        new_d_seg=0.2,
        old_d_pose=0.3,
        new_d_pose=0.3,
    )

    effect = result["action_effects"][0]
    candidate = result["path_action_candidates"][0]
    assert effect.action_kind == "frame0_pose_temporal_path"
    assert effect.inverse_source == "frame0_pose_temporal_path"
    assert effect.frame_index == 0
    assert effect.frame_incidence == "pose_only"
    assert effect.delta_bytes == candidate["temporal_payload_bytes"]
    assert effect.delta_score_total is not None and effect.delta_score_total > 0.0
    assert effect.value_per_byte is not None
    assert effect.parseback_survived is False
    assert effect.inflate_survived is False
    assert effect.promotion_eligible is False
    assert BLOCKER_PATH_TRAJECTORY_NO_RECEIVER_PROOF in effect.blockers
    assert result["candidate_queue"][0]["ready_for_exact_eval_dispatch"] is False


def test_selector_temporal_path_candidate_uses_pr110_sequence_and_blocks_launch() -> None:
    result = build_selector_temporal_path_candidates_from_rows(
        {
            "selector_sequence": [0, 0, 2, 2, 2, 1, 1, 0],
            "base_archive_bytes": 3000,
        },
        base_archive_bytes=3000,
        old_d_seg=0.2,
        new_d_seg=0.2,
        old_d_pose=0.3,
        new_d_pose=0.3,
    )

    effect = result["action_effects"][0]
    candidate = result["path_action_candidates"][0]
    comparison = candidate["temporal_path"]["selector_comparison"]
    assert effect.family == "pr110"
    assert effect.action_kind == "selector_temporal_path"
    assert effect.inverse_source == "selector_temporal_path"
    assert effect.frame_index == "both"
    assert effect.delta_bytes == candidate["temporal_payload_bytes"]
    assert comparison["selector_count"] == 8
    assert comparison["unique_modes"] == 3
    assert comparison["path_temporal_bytes"] == candidate["temporal_payload_bytes"]
    assert BLOCKER_PATH_TRAJECTORY_NO_RECEIVER_PROOF in result["candidate_queue"][0]["blockers"]
    assert result["candidate_queue"][0]["ready_for_exact_eval_dispatch"] is False


def test_selector_temporal_path_preserves_sparse_pair_rows() -> None:
    result = build_selector_temporal_path_candidates_from_rows(
        [
            {"scope": {"pair_index": 3}, "selector_id": 2},
            {"scope": {"pair_index": 9}, "selector_id": 4},
        ],
        base_archive_bytes=3000,
    )

    effect = result["action_effects"][0]
    temporal_path = result["path_action_candidates"][0]["temporal_path"]
    assert list(effect.pair_ids) == [3, 9]
    assert temporal_path["pair_min"] == 3
    assert temporal_path["pair_max"] == 9
    assert temporal_path["pair_count"] == 2


def test_selector_sequence_comparison_can_prefer_rle_over_path_payload() -> None:
    comparison = selector_sequence_encoding_comparison([0, 0, 0, 1, 1, 0], path_payload_bytes=99)
    assert comparison["rle_bytes"] < comparison["path_temporal_bytes"]
    assert comparison["best_encoding"] in {"raw_fixed_width", "rle"}


def test_generate_path_action_candidates_cli_writes_valid_action_effect_rows(tmp_path: Path) -> None:
    target, argmax, margin, pair_indices = _hard_region_arrays()
    npz = tmp_path / "hard_region.npz"
    np.savez(
        npz,
        target_labels_bhw=target,
        candidate_argmax_bhw=argmax,
        target_margin_bhw=margin,
        pair_indices=pair_indices,
    )
    out_dir = tmp_path / "out"
    pose_json = tmp_path / "pose.json"
    pose_json.write_text(
        json.dumps(
            {
                "pair_indices": [0, 1, 2, 3],
                "pose_residuals": [0.0, 0.1, 0.05, 0.0],
                "base_archive_bytes": 1000,
                "old_d_seg": 0.2,
                "new_d_seg": 0.2,
                "old_d_pose": 0.3,
                "new_d_pose": 0.3,
            }
        ),
        encoding="utf-8",
    )
    selector_json = tmp_path / "selector.json"
    selector_json.write_text(
        json.dumps({"selector_sequence": [0, 0, 1, 1, 2, 2], "base_archive_bytes": 1000}),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "generate_path_action_candidates.py"),
            "--hard-region-npz",
            str(npz),
            "--output-dir",
            str(out_dir),
            "--base-archive-bytes",
            "1000",
            "--old-d-seg",
            "0.2",
            "--old-d-pose",
            "0.3",
            "--pose-trajectory-json",
            str(pose_json),
            "--selector-rows-json",
            str(selector_json),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["action_effect_row_count"] == 3
    assert summary["candidate_queue_row_count"] == 3
    assert summary["ready_for_exact_eval_dispatch"] is False

    validate = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "validate_action_effect_rows.py"),
            str(out_dir / "action_effect_rows.jsonl"),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr
    validation_summary = json.loads(validate.stdout)
    assert validation_summary["failed_count"] == 0


# --- every produced atom must emit a base-bound rent-law row ---


def test_path_birth_candidate_emits_base_bound_rent_evaluation() -> None:
    target, argmax, margin, pair_indices = _hard_region_arrays()
    out = build_path_action_candidates_from_arrays(
        target_labels_bhw=target,
        candidate_argmax_bhw=argmax,
        target_margin_bhw=margin,
        pair_indices=pair_indices,
        base_archive_sha256=_BASE_SHA,
        old_d_seg=0.2,
        old_d_pose=0.3,
    )

    assert out["candidate_action_evaluations"]
    row = out["candidate_action_evaluations"][0]
    assert row["schema"] == "hi_nerv_candidate_action_evaluation.v1"
    assert row["base_archive_sha256"] == _BASE_SHA
    # A generated (not-yet-survived) path birth raises bytes with no measured
    # score drop => it does NOT pay rent yet.
    assert row["pays_rent"] is False
    assert row["promotion_eligible"] is False
    assert out["candidate_action_evaluations"][0] == out["path_action_candidates"][0][
        "candidate_action_evaluation"
    ]
    assert out["policy"]["every_action_must_pay_rent"] is True


def test_path_birth_candidate_rent_row_fails_closed_without_base_sha() -> None:
    target, argmax, margin, pair_indices = _hard_region_arrays()
    out = build_path_action_candidates_from_arrays(
        target_labels_bhw=target,
        candidate_argmax_bhw=argmax,
        target_margin_bhw=margin,
        pair_indices=pair_indices,
        old_d_seg=0.2,
        old_d_pose=0.3,
    )

    row = out["candidate_action_evaluations"][0]
    assert row["schema"] == RENT_EVALUATION_SCHEMA
    assert row["evaluable"] is False
    assert RENT_BLOCKER_BASE_ARCHIVE_SHA_MISSING in row["blockers"]
    assert row["pays_rent"] is False


def test_pose_temporal_candidate_emits_rent_evaluation() -> None:
    out = build_pose_temporal_path_candidates_from_arrays(
        pair_indices=[0, 1, 2, 3],
        pose_action_profile=[0.0, 0.1, 0.2, 0.0],
        base_archive_sha256=_BASE_SHA,
        old_d_seg=0.2,
        new_d_seg=0.2,
        old_d_pose=0.3,
        new_d_pose=0.29,
    )

    assert out["candidate_action_evaluations"]
    row = out["candidate_action_evaluations"][0]
    assert row["base_archive_sha256"] == _BASE_SHA
    assert row["schema"] == "hi_nerv_candidate_action_evaluation.v1"
    assert row["promotion_eligible"] is False


def test_selector_temporal_candidate_emits_rent_evaluation() -> None:
    rows = [
        {"pair_index": 0, "selector_id": 1},
        {"pair_index": 1, "selector_id": 2},
        {"pair_index": 2, "selector_id": 1},
    ]
    out = build_selector_temporal_path_candidates_from_rows(
        rows,
        base_archive_sha256=_BASE_SHA,
        old_d_seg=0.2,
        new_d_seg=0.2,
        old_d_pose=0.3,
        new_d_pose=0.3,
    )

    assert out["candidate_action_evaluations"]
    row = out["candidate_action_evaluations"][0]
    assert row["base_archive_sha256"] == _BASE_SHA
    assert row["promotion_eligible"] is False
