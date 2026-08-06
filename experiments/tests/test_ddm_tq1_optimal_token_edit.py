# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tac.optimization import ddm_ix2_archive_container as IX2

import ddm_tq1_optimal_token_edit as tq1


def test_affected_pairs_are_exact_pair_ids_not_cell_presence() -> None:
    before = np.zeros((6, 4), dtype=np.uint8)
    after = before.copy()
    after[1, 2] = 3
    after[4, :] = np.array([1, 2, 3, 4], dtype=np.uint8)

    assert tq1.affected_pairs_for_cell(before, after).tolist() == [1, 4]


def test_derived_rungs_are_per_cell_and_do_not_replay_global_baseline() -> None:
    joint_guard = np.array([[0.99, 0.40, 0.02], [0.75, 0.10, 0.30]], dtype=np.float64)
    seg_guard = np.array([[0.99, 0.10, 0.01], [0.30, 0.10, 0.20]], dtype=np.float64)
    pose_guard = np.array([[0.99, 0.10, 0.01], [0.20, 0.10, 0.20]], dtype=np.float64)
    activity = np.array([[0.10, 0.30, 0.99], [0.20, 0.90, 0.40]], dtype=np.float64)

    rungs = tq1.derive_cell_rungs(
        joint_guard=joint_guard,
        seg_guard=seg_guard,
        pose_guard=pose_guard,
        activity=activity,
    )

    menus = set(rungs.values())
    assert len(menus) > 2
    assert tuple(tq1.DOMINATED_GLOBAL_BASELINE) not in menus
    assert rungs[(0, 0)] == ()
    assert min(rungs[(0, 2)]) <= 8


def test_mode_step_moves_circularly_toward_mode_without_overshoot() -> None:
    cell = np.array([[15, 0, 8, 7], [14, 2, 9, 6]], dtype=np.uint8)
    mode = np.array([0, 0, 10, 7], dtype=np.uint8)

    got = tq1.mode_step_codes(cell, mode, max_step=2)

    assert got.tolist() == [[0, 0, 10, 7], [0, 0, 10, 7]]


def test_apply_candidate_reencodes_through_ix2_losslessly() -> None:
    tokens = np.zeros((5, 2, 2, 4), dtype=np.uint8)
    tokens[:, 0, 1, :] = np.array([1, 3, 5, 7], dtype=np.uint8)
    tokens[2, 0, 1, :] = np.array([15, 15, 15, 15], dtype=np.uint8)
    mode, _ = IX2._factor_mode_delta(tokens, 16)
    candidate = tq1.Candidate(
        candidate_id="snap_r00_c01_L08",
        row=0,
        col=1,
        direction="snap_sublattice",
        rung=8,
        priority=1.0,
        joint_guard=0.1,
        seg_guard=0.1,
        pose_guard=0.1,
        activity=1.0,
        affected_pair_count=1,
        affected_pair_preview=(2,),
    )

    mutated = tq1.apply_candidate(tokens, candidate, mode)
    frame = IX2.encode_token_frame(mutated, levels=16)

    assert np.array_equal(IX2.decode_token_frame(frame), mutated)
    assert tq1.affected_pairs_for_cell(tokens[:, 0, 1, :], mutated[:, 0, 1, :]).tolist() == [0, 1, 3, 4]


def test_acceptance_verdict_uses_recomputed_joint_score_and_pose_guard() -> None:
    current = tq1.ComponentScore(d_seg=0.004, d_pose=0.0007, archive_bytes=360_000)
    better = tq1.ComponentScore(d_seg=0.0039, d_pose=0.0007, archive_bytes=350_000)
    worse_pose = tq1.ComponentScore(d_seg=0.0030, d_pose=0.0030, archive_bytes=350_000)

    assert tq1.acceptance_verdict(current, better)["accepted"] is True
    assert tq1.acceptance_verdict(current, worse_pose)["accepted"] is False


def test_score_from_components_rejects_silent_formula_drift() -> None:
    got = tq1.score_from_components(0.0, 0.0, tq1.DEN)
    assert got == pytest.approx(25.0)


def test_candidate_from_row_validates_phase_a_price_schema() -> None:
    row = {
        "schema": "ddm_tq1_phase_a_candidate_price.v1",
        "candidate": {
            "candidate_id": "snap_r00_c02_L12",
            "row": 0,
            "col": 2,
            "direction": "snap_sublattice",
            "rung": 12,
            "priority": 1.25,
            "joint_guard": 0.0,
            "seg_guard": 0.1,
            "pose_guard": 0.2,
            "activity": 0.3,
            "affected_pair_count": 2,
            "affected_pair_preview": [0, 17],
        },
    }

    candidate = tq1.candidate_from_row(row)

    assert candidate.candidate_id == "snap_r00_c02_L12"
    assert candidate.affected_pair_preview == (0, 17)


def test_phase_b_jsonl_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    tq1.write_phase_b_jsonl([{"candidate_id": "a", "accepted": False}], path)
    tq1.write_phase_b_jsonl([{"candidate_id": "a", "accepted": False}], path)

    with pytest.raises(RuntimeError, match="immutable output differs"):
        tq1.write_phase_b_jsonl([{"candidate_id": "b", "accepted": False}], path)


def test_tq1_ix2_receiver_adapter_orders_frames() -> None:
    class FakeDecoder:
        def __init__(self, archive_dir: Path) -> None:
            self.archive_dir = archive_dir

        def f1(self, i: int) -> np.ndarray:
            return np.full((874, 1164, 3), i + 1, dtype=np.uint8)

        def f0(self, i: int, f1_u8: np.ndarray | None = None) -> np.ndarray:
            assert f1_u8 is not None
            return np.full((874, 1164, 3), i, dtype=np.uint8)

    receiver = tq1.TQ1IX2Receiver(Path("/unused"), FakeDecoder, archive_sha256="0" * 64)
    got = receiver.render_camera_pairs([3, 4])

    assert got.shape == (2, 2, 874, 1164, 3)
    assert got.dtype == np.uint8
    assert int(got[0, 0, 0, 0, 0]) == 3
    assert int(got[0, 1, 0, 0, 0]) == 4
    assert receiver.custody["score_claim"] is False
