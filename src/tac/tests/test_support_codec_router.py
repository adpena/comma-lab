# SPDX-License-Identifier: MIT
"""Tests for SupportCodecRouter selection and authority boundaries."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tac.analysis.action_effect import ActionEffect
from tac.analysis.path_action_producer import path_tube_support_from_mask
from tac.analysis.support_codec_router import (
    BLOCKER_CODEC_DECODED_HASH_MISMATCH,
    BLOCKER_CODEC_RESEARCH_ONLY,
    BLOCKER_PATH_ACTION_INFLATE_MISSING,
    BLOCKER_PATH_ACTION_PARSEBACK_MISSING,
    route_support_codecs_for_path_candidate,
)


def _candidate_from_mask(mask: np.ndarray, *, action_id: str = "path-action") -> dict:
    support = path_tube_support_from_mask(
        mask,
        pair_index=0,
        frame_index=1,
        target_class=4,
        epsilon=0.5,
    ).as_dict()
    return {
        "schema": "tac.path_action_candidate.v1",
        "action_id": action_id,
        "family": "hinerv",
        "action_kind": "frame1_seg_margin_frontier_path",
        "support": support,
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _source_effect(candidate: dict) -> ActionEffect:
    support = candidate["support"]
    return ActionEffect.build(
        action_id=candidate["action_id"],
        family="hinerv",
        action_kind="frame1_seg_margin_frontier_path",
        inverse_source="path_tube_segnet_margin_frontier",
        frame_index=1,
        frame_incidence="seg_pose_joint",
        candidate_status="rejected",
        authority="batch_local_path_support",
        normalization_scope="batch_local",
        producer="fixture",
        consumer="inverse_evaluate_candidate_queue",
        pair_ids=[0],
        class_ids=[4],
        region_ids=["fixture"],
        old_d_seg=0.2,
        new_d_seg=0.2,
        old_d_pose=0.3,
        new_d_pose=0.3,
        old_bytes=0,
        new_bytes=int(support["support_encoded_bytes"]),
        receiver_surface={"uint8_changed_pixels": 0, "seg_argmax_changed_pixels": 0},
        exact_score_decision="reject",
        parseback_survived=False,
        inflate_survived=False,
        wrong_to_target=0,
        support_source="fixture",
        support_cardinality=int(support["support_cardinality"]),
        support_sha256=str(support["support_sha256"]),
        support_encoding="path_tube",
        support_encoded_bytes=int(support["support_encoded_bytes"]),
        support_research_only=False,
        blockers=["path_action_support_without_wrong_to_target_is_not_birth"],
    )


def _row_by_encoding(report: dict, encoding: str) -> dict:
    return next(row for row in report["support_codec_candidates"] if row["support_encoding"] == encoding)


def test_router_selects_rle_over_path_tube_for_two_stripe_support() -> None:
    mask = np.zeros((64, 96), dtype=bool)
    mask[8:10, 5:91] = True
    mask[42:44, 5:91] = True
    candidate = _candidate_from_mask(mask)

    report = route_support_codecs_for_path_candidate(candidate, source_effect=_source_effect(candidate))

    assert report["selected_support_encoding"] == "rle"
    rle = _row_by_encoding(report, "rle")
    path = _row_by_encoding(report, "path_tube")
    assert rle["encoded_bytes"] < path["encoded_bytes"]
    assert _row_by_encoding(report, "path_tube")["rejection_reason"] == "support_codec_dominated_by_selected_codec"
    assert len(report["candidate_queue"]) == 1
    assert report["candidate_queue"][0]["support_encoding"] == "rle"


def test_router_preserves_path_tube_win_for_curve_like_support() -> None:
    mask = np.zeros((128, 128), dtype=bool)
    for index in range(8, 120):
        mask[index, index] = True
    candidate = _candidate_from_mask(mask)

    report = route_support_codecs_for_path_candidate(candidate, source_effect=_source_effect(candidate))

    assert report["selected_support_encoding"] == "path_tube"
    path = _row_by_encoding(report, "path_tube")
    assert path["encoded_bytes"] == report["selected_total_cost_bytes"]
    assert path["encoded_bytes"] < _row_by_encoding(report, "coordinate_list_u16")["encoded_bytes"]
    assert path["encoded_bytes"] < _row_by_encoding(report, "bitmap_bitset")["encoded_bytes"]


def test_decoded_support_hash_mismatch_blocks_selection() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[3:5, 4:8] = True
    candidate = _candidate_from_mask(mask)
    candidate["support"]["support_sha256"] = "f" * 64

    report = route_support_codecs_for_path_candidate(candidate, source_effect=_source_effect(_candidate_from_mask(mask)))

    assert report["selected_support_encoding"] is None
    assert report["candidate_queue"] == []
    assert BLOCKER_CODEC_DECODED_HASH_MISMATCH in report["blockers"]
    assert all(BLOCKER_CODEC_DECODED_HASH_MISMATCH in row["blockers"] for row in report["support_codec_candidates"])


def test_research_only_support_cannot_clear_archive_closure() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:6, 2:6] = True
    candidate = _candidate_from_mask(mask)
    candidate["support"]["support_research_only"] = True

    report = route_support_codecs_for_path_candidate(candidate, source_effect=_source_effect(candidate))

    assert report["selected_support_encoding"] is None
    assert report["candidate_queue"] == []
    assert BLOCKER_CODEC_RESEARCH_ONLY in report["blockers"]
    assert all(BLOCKER_CODEC_RESEARCH_ONLY in row["blockers"] for row in report["support_codec_candidates"])


def test_selected_action_effect_includes_codec_bytes_and_stays_non_promotable() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:12, 4:28] = True
    candidate = _candidate_from_mask(mask)

    report = route_support_codecs_for_path_candidate(candidate, source_effect=_source_effect(candidate))
    selected = report["selected_action_effect"]
    selected_row = next(row for row in report["support_codec_candidates"] if row["selected"])

    assert selected["support_encoding"] == selected_row["support_encoding"]
    assert selected["support_encoded_bytes"] == selected_row["encoded_bytes"]
    assert selected["delta_bytes"] == selected_row["total_cost_bytes"]
    assert selected["delta_score_total"] is not None
    assert selected["value_per_byte"] is not None
    assert selected["promotion_eligible"] is False
    assert selected["parseback_survived"] is False
    assert selected["inflate_survived"] is False
    assert report["candidate_queue"][0]["ready_for_exact_eval_dispatch"] is False


def test_selected_rle_action_effect_survives_with_same_action_receipt() -> None:
    mask = np.zeros((64, 96), dtype=bool)
    mask[8:10, 5:91] = True
    mask[42:44, 5:91] = True
    candidate = _candidate_from_mask(mask)
    effect = _source_effect(candidate)
    receipt = {
        "schema": "tac.support_codec_survival_receipt.v1",
        "receipt_id": "rle-survival-fixture",
        "action_id": candidate["action_id"],
        "support_sha256": candidate["support"]["support_sha256"],
        "support_encoding": "rle",
        "encoded_program_sha256": "1" * 64,
        "decoded_support_sha256": candidate["support"]["support_sha256"],
        "decoded_action_sha256": "2" * 64,
        "archive_sha256": "3" * 64,
        "parseback_survived": True,
        "inflate_survived": True,
        "authority": "archive_parseback_inflate_same_action",
    }

    report = route_support_codecs_for_path_candidate(
        candidate,
        source_effect=effect,
        survival_receipts=[receipt],
    )
    selected = report["selected_action_effect"]
    selected_row = next(row for row in report["support_codec_candidates"] if row["selected"])

    assert report["selected_support_encoding"] == "rle"
    assert selected_row["parseback_survived"] is True
    assert selected_row["inflate_survived"] is True
    assert selected_row["survival_receipt_id"] == "rle-survival-fixture"
    assert selected["support_encoding"] == "rle"
    assert selected["parseback_survived"] is True
    assert selected["inflate_survived"] is True
    assert BLOCKER_PATH_ACTION_PARSEBACK_MISSING not in selected["blockers"]
    assert BLOCKER_PATH_ACTION_INFLATE_MISSING not in selected["blockers"]
    assert report["candidate_queue"][0]["support_encoding"] == "rle"
    assert report["candidate_queue"][0]["parseback_survived"] is True
    assert report["candidate_queue"][0]["inflate_survived"] is True


def test_selected_rle_survival_receipt_must_match_same_action_identity() -> None:
    mask = np.zeros((64, 96), dtype=bool)
    mask[8:10, 5:91] = True
    mask[42:44, 5:91] = True
    candidate = _candidate_from_mask(mask)
    receipt = {
        "schema": "tac.support_codec_survival_receipt.v1",
        "receipt_id": "wrong-support",
        "action_id": candidate["action_id"],
        "support_sha256": "f" * 64,
        "support_encoding": "rle",
        "parseback_survived": True,
        "inflate_survived": True,
    }

    report = route_support_codecs_for_path_candidate(
        candidate,
        source_effect=_source_effect(candidate),
        survival_receipts=[receipt],
    )
    selected = report["selected_action_effect"]
    selected_row = next(row for row in report["support_codec_candidates"] if row["selected"])

    assert report["selected_support_encoding"] == "rle"
    assert selected_row["parseback_survived"] is False
    assert selected_row["inflate_survived"] is False
    assert selected_row["survival_receipt_id"] is None
    assert selected["parseback_survived"] is False
    assert selected["inflate_survived"] is False


def test_route_path_action_support_codecs_cli_writes_selected_only_queue(tmp_path: Path) -> None:
    mask = np.zeros((64, 96), dtype=bool)
    mask[8:10, 5:91] = True
    mask[42:44, 5:91] = True
    candidate = _candidate_from_mask(mask)
    effect = _source_effect(candidate)
    candidates_path = tmp_path / "path_action_candidates.jsonl"
    effects_path = tmp_path / "action_effect_rows.jsonl"
    receipts_path = tmp_path / "survival_receipts.jsonl"
    candidates_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    effects_path.write_text(json.dumps(effect.as_dict()) + "\n", encoding="utf-8")
    receipts_path.write_text(
        json.dumps(
            {
                "schema": "tac.support_codec_survival_receipt.v1",
                "receipt_id": "cli-rle-survived",
                "action_id": candidate["action_id"],
                "support_sha256": candidate["support"]["support_sha256"],
                "support_encoding": "rle",
                "parseback_survived": True,
                "inflate_survived": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    repo_root = Path(__file__).resolve().parents[3]

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "route_path_action_support_codecs.py"),
            "--path-action-candidates",
            str(candidates_path),
            "--action-effect-rows",
            str(effects_path),
            "--survival-receipts",
            str(receipts_path),
            "--output-dir",
            str(out_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["candidate_queue_row_count"] == 1
    assert summary["candidate_queue_contains_selected_only"] is True
    assert summary["selected_support_encodings"] == ["rle"]
    report = json.loads((out_dir / "support_codec_router_report.json").read_text(encoding="utf-8"))
    assert report["reports"][0]["selected_support_encoding"] == "rle"
    assert report["reports"][0]["selected_action_effect"]["parseback_survived"] is True
    assert report["reports"][0]["selected_action_effect"]["inflate_survived"] is True
