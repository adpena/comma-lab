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
    build_path_action_candidates_from_arrays,
    decode_path_tube_payload,
    path_tube_support_from_mask,
    rdp_simplify,
    support_mask_sha256,
)


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
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["action_effect_row_count"] == 1
    assert summary["candidate_queue_row_count"] == 1
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
