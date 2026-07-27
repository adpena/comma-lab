"""Whole-object authority tests for the G21 continuation ledger."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from tac.optimization.ddm_lattice_teacher_compaction_homotopy import (
    ActionKind,
    CompactionHomotopyError,
    ReceiverMeasurement,
    ScorerMeasurement,
    append_row,
    contest_score,
    measure_complete_object_row,
    measured_interaction,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _receiver(root: str):
    return lambda _path: ReceiverMeasurement(
        output_root_sha256=root,
        pair_count=600,
        deterministic_replay_sha256=root,
        runtime_seconds=1.25,
        peak_storage_bytes=4096,
    )


def _nondeterministic_receiver(root: str):
    return lambda _path: ReceiverMeasurement(
        output_root_sha256=root,
        pair_count=600,
        deterministic_replay_sha256="f" * 64,
        runtime_seconds=1.25,
        peak_storage_bytes=4096,
    )


def _scorer(_path: Path) -> ScorerMeasurement:
    return ScorerMeasurement(
        d_seg=0.001,
        d_pose=0.01,
        pair_count=600,
        axis="[synthetic-test-only]",
        receipt_sha256="b" * 64,
    )


def test_rows_price_actual_archive_and_identity_requires_output_equality(
    tmp_path: Path,
) -> None:
    baseline_path = (tmp_path / "baseline.zip").resolve()
    baseline_path.write_bytes(b"A" * 101)
    root = _sha(b"realized-n600")
    baseline = measure_complete_object_row(
        archive_path=baseline_path,
        action_id="baseline",
        action_kind=ActionKind.BASELINE,
        receiver_callback=_receiver(root),
        scorer_callback=_scorer,
    )
    assert baseline.archive_bytes == 101
    assert baseline.score == contest_score(d_seg=0.001, d_pose=0.01, archive_bytes=101)

    recode_path = (tmp_path / "recode.zip").resolve()
    recode_path.write_bytes(b"B" * 73)
    identity = measure_complete_object_row(
        archive_path=recode_path,
        action_id="population-v2",
        action_kind=ActionKind.RECODE_IDENTITY,
        receiver_callback=_receiver(root),
        parent=baseline,
    )
    assert identity.equality_proof is True
    assert identity.d_seg == baseline.d_seg
    assert identity.d_pose == baseline.d_pose
    assert identity.archive_bytes == 73
    assert append_row((baseline,), identity) == (baseline, identity)

    with pytest.raises(CompactionHomotopyError, match="double-replay"):
        measure_complete_object_row(
            archive_path=recode_path,
            action_id="nondeterministic-baseline",
            action_kind=ActionKind.BASELINE,
            receiver_callback=_nondeterministic_receiver(root),
            scorer_callback=_scorer,
        )

    with pytest.raises(CompactionHomotopyError, match="equality proof"):
        measure_complete_object_row(
            archive_path=recode_path,
            action_id="wrong-root",
            action_kind=ActionKind.RECODE_IDENTITY,
            receiver_callback=_receiver("c" * 64),
            parent=baseline,
        )


def test_lossy_row_requires_full_n600_scorer_and_interactions_are_never_imputed(
    tmp_path: Path,
) -> None:
    archive = (tmp_path / "candidate.zip").resolve()
    archive.write_bytes(b"C" * 19)
    root = "d" * 64
    with pytest.raises(CompactionHomotopyError, match="requires a full-n600 scorer"):
        measure_complete_object_row(
            archive_path=archive,
            action_id="lossy",
            action_kind=ActionKind.LOSSY_COMPACTION,
            receiver_callback=_receiver(root),
        )
    interaction = measured_interaction(
        baseline=None,
        action_a=None,
        action_b=None,
        joint=None,
    )
    assert interaction["score_interaction"] is None
    assert interaction["byte_interaction"] is None
    assert "missing complete-object corners" in str(interaction["reason"])
    assert math.isfinite(contest_score(d_seg=0.0, d_pose=0.0, archive_bytes=1))
