# SPDX-License-Identifier: MIT
"""Measured whole-object continuation ledger for G21 lattice compaction.

The scheduler has no proxy-rate path. A row can enter only after a physical
archive is hashed and a receiver callback proves its realized population.
Lossy rows additionally require a full-n600 scorer callback. Identity recodes
may inherit a parent's distortions only when the receiver proves the same
decoded population root. Scores always use the actual archive byte length.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

SCORE_DENOMINATOR_BYTES: Final = 37_545_489
ROW_SCHEMA: Final = "ddm_lattice_teacher_compaction_homotopy_row.v1"
LEDGER_SCHEMA: Final = "ddm_lattice_teacher_compaction_homotopy_ledger.v1"
INTERACTION_SCHEMA: Final = "ddm_lattice_teacher_compaction_interaction.v1"
EXPECTED_PAIR_COUNT: Final = 600
READ_CHUNK_BYTES: Final = 1 << 20


class CompactionHomotopyError(ValueError):
    """A whole-object, receiver, scorer, or continuation invariant failed."""


class ActionKind(StrEnum):
    BASELINE = "BASELINE"
    RECODE_IDENTITY = "RECODE_IDENTITY"
    LOSSY_COMPACTION = "LOSSY_COMPACTION"


@dataclass(frozen=True, slots=True)
class ReceiverMeasurement:
    output_root_sha256: str
    pair_count: int
    deterministic_replay_sha256: str
    runtime_seconds: float
    peak_storage_bytes: int


@dataclass(frozen=True, slots=True)
class ScorerMeasurement:
    d_seg: float
    d_pose: float
    pair_count: int
    axis: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class CompleteObjectRow:
    schema: str
    row_id: str
    parent_row_id: str | None
    action_id: str
    action_kind: str
    archive_path: str
    archive_bytes: int
    archive_sha256: str
    output_root_sha256: str
    deterministic_replay_sha256: str
    receiver_runtime_seconds: float
    receiver_peak_storage_bytes: int
    equality_proof: bool
    d_seg: float
    d_pose: float
    scorer_axis: str
    scorer_receipt_sha256: str | None
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "row_id": self.row_id,
            "parent_row_id": self.parent_row_id,
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "archive": {
                "path": self.archive_path,
                "bytes": self.archive_bytes,
                "sha256": self.archive_sha256,
            },
            "receiver": {
                "output_root_sha256": self.output_root_sha256,
                "deterministic_replay_sha256": self.deterministic_replay_sha256,
                "runtime_seconds": self.receiver_runtime_seconds,
                "peak_storage_bytes": self.receiver_peak_storage_bytes,
                "pair_count": EXPECTED_PAIR_COUNT,
            },
            "equality_proof": self.equality_proof,
            "distortion": {
                "d_seg": self.d_seg,
                "d_pose": self.d_pose,
                "scorer_axis": self.scorer_axis,
                "scorer_receipt_sha256": self.scorer_receipt_sha256,
            },
            "score": self.score,
            "rate_is_actual_whole_archive": True,
        }


ReceiverCallback = Callable[[Path], ReceiverMeasurement]
ScorerCallback = Callable[[Path], ScorerMeasurement]


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CompactionHomotopyError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0:
        raise CompactionHomotopyError(f"{label} must be finite and nonnegative")
    return float(value)


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CompactionHomotopyError(f"{label} must be an exact integer >= {minimum}")
    return value


def _ascii(value: object, label: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise CompactionHomotopyError(f"{label} must be nonempty ASCII")
    return value


def contest_score(*, d_seg: float, d_pose: float, archive_bytes: int) -> float:
    seg = _finite_nonnegative(d_seg, "d_seg")
    pose = _finite_nonnegative(d_pose, "d_pose")
    rate_bytes = _exact_int(archive_bytes, "archive_bytes", minimum=1)
    return 100.0 * seg + math.sqrt(10.0 * pose) + (25.0 * rate_bytes / SCORE_DENOMINATOR_BYTES)


def measure_complete_object_row(
    *,
    archive_path: Path,
    action_id: str,
    action_kind: ActionKind,
    receiver_callback: ReceiverCallback,
    parent: CompleteObjectRow | None = None,
    scorer_callback: ScorerCallback | None = None,
) -> CompleteObjectRow:
    """Build one immutable row from a physical archive and real callbacks."""

    archive = Path(archive_path)
    if not archive.is_absolute() or not archive.is_file() or archive.is_symlink():
        raise CompactionHomotopyError("archive_path must be one absolute regular non-symlink file")
    action = _ascii(action_id, "action_id")
    if not isinstance(action_kind, ActionKind):
        raise CompactionHomotopyError("action_kind must be a typed ActionKind")
    archive_bytes = _exact_int(archive.stat().st_size, "archive byte length", minimum=1)
    archive_sha = _sha_file(archive)
    receiver = receiver_callback(archive)
    if not isinstance(receiver, ReceiverMeasurement):
        raise CompactionHomotopyError("receiver callback returned an untyped result")
    output_root = _require_sha(receiver.output_root_sha256, "receiver output root")
    replay_root = _require_sha(
        receiver.deterministic_replay_sha256,
        "receiver deterministic replay root",
    )
    if receiver.pair_count != EXPECTED_PAIR_COUNT:
        raise CompactionHomotopyError("receiver callback did not measure all 600 pairs")
    if replay_root != output_root:
        raise CompactionHomotopyError("deterministic receiver double-replay proof failed")
    runtime = _finite_nonnegative(receiver.runtime_seconds, "receiver runtime_seconds")
    peak_storage = _exact_int(
        receiver.peak_storage_bytes,
        "receiver peak_storage_bytes",
    )

    equality = False
    scorer_receipt: str | None
    if action_kind is ActionKind.RECODE_IDENTITY:
        if parent is None:
            raise CompactionHomotopyError("identity recode requires a measured parent row")
        if output_root != parent.output_root_sha256:
            raise CompactionHomotopyError("identity recode receiver equality proof failed")
        if scorer_callback is not None:
            raise CompactionHomotopyError("identity recode must not silently replace equality with scoring")
        d_seg = parent.d_seg
        d_pose = parent.d_pose
        scorer_axis = f"inherited-by-exact-equality:{parent.scorer_axis}"
        scorer_receipt = parent.scorer_receipt_sha256
        equality = True
    else:
        if scorer_callback is None:
            raise CompactionHomotopyError("baseline/lossy row requires a full-n600 scorer callback")
        scored = scorer_callback(archive)
        if not isinstance(scored, ScorerMeasurement):
            raise CompactionHomotopyError("scorer callback returned an untyped result")
        if scored.pair_count != EXPECTED_PAIR_COUNT:
            raise CompactionHomotopyError("scorer callback did not measure all 600 pairs")
        d_seg = _finite_nonnegative(scored.d_seg, "scorer d_seg")
        d_pose = _finite_nonnegative(scored.d_pose, "scorer d_pose")
        scorer_axis = _ascii(scored.axis, "scorer axis")
        scorer_receipt = _require_sha(scored.receipt_sha256, "scorer receipt sha256")
    score = contest_score(d_seg=d_seg, d_pose=d_pose, archive_bytes=archive_bytes)
    identity = hashlib.sha256(
        (
            f"{parent.row_id if parent else '-'}\0{action}\0{action_kind.value}\0"
            f"{archive_sha}\0{output_root}\0{d_seg:.17g}\0{d_pose:.17g}"
        ).encode("ascii")
    ).hexdigest()
    return CompleteObjectRow(
        schema=ROW_SCHEMA,
        row_id=identity,
        parent_row_id=parent.row_id if parent else None,
        action_id=action,
        action_kind=action_kind.value,
        archive_path=str(archive),
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha,
        output_root_sha256=output_root,
        deterministic_replay_sha256=replay_root,
        receiver_runtime_seconds=runtime,
        receiver_peak_storage_bytes=peak_storage,
        equality_proof=equality,
        d_seg=d_seg,
        d_pose=d_pose,
        scorer_axis=scorer_axis,
        scorer_receipt_sha256=scorer_receipt,
        score=score,
    )


def append_row(
    rows: Sequence[CompleteObjectRow],
    row: CompleteObjectRow,
) -> tuple[CompleteObjectRow, ...]:
    """Return a new ledger after enforcing ancestry and immutable uniqueness."""

    if not isinstance(row, CompleteObjectRow):
        raise CompactionHomotopyError("ledger accepts only CompleteObjectRow")
    existing = {item.row_id: item for item in rows}
    if len(existing) != len(rows):
        raise CompactionHomotopyError("existing ledger contains duplicate row ids")
    if row.row_id in existing:
        raise CompactionHomotopyError("row id already exists in immutable ledger")
    if row.parent_row_id is not None and row.parent_row_id not in existing:
        raise CompactionHomotopyError("row parent is absent from ledger")
    return (*rows, row)


def pareto_frontier(rows: Sequence[CompleteObjectRow]) -> tuple[str, ...]:
    """Return rows not dominated in actual archive bytes, d_seg, and d_pose."""

    frontier: list[str] = []
    for candidate in rows:
        dominated = any(
            other.row_id != candidate.row_id
            and other.archive_bytes <= candidate.archive_bytes
            and other.d_seg <= candidate.d_seg
            and other.d_pose <= candidate.d_pose
            and (
                other.archive_bytes < candidate.archive_bytes
                or other.d_seg < candidate.d_seg
                or other.d_pose < candidate.d_pose
            )
            for other in rows
        )
        if not dominated:
            frontier.append(candidate.row_id)
    return tuple(frontier)


def measured_interaction(
    *,
    baseline: CompleteObjectRow | None,
    action_a: CompleteObjectRow | None,
    action_b: CompleteObjectRow | None,
    joint: CompleteObjectRow | None,
) -> dict[str, object]:
    """Compute score/byte interaction only when all physical corners exist."""

    corners = (baseline, action_a, action_b, joint)
    missing = [
        name
        for name, value in zip(("baseline", "action_a", "action_b", "joint"), corners, strict=True)
        if value is None
    ]
    if missing:
        return {
            "schema": INTERACTION_SCHEMA,
            "score_interaction": None,
            "byte_interaction": None,
            "reason": f"missing complete-object corners: {','.join(missing)}",
        }
    assert baseline is not None and action_a is not None and action_b is not None and joint is not None
    return {
        "schema": INTERACTION_SCHEMA,
        "corner_row_ids": {
            "baseline": baseline.row_id,
            "action_a": action_a.row_id,
            "action_b": action_b.row_id,
            "joint": joint.row_id,
        },
        "score_interaction": joint.score - action_a.score - action_b.score + baseline.score,
        "byte_interaction": (
            joint.archive_bytes - action_a.archive_bytes - action_b.archive_bytes + baseline.archive_bytes
        ),
        "reason": None,
    }


def ledger_receipt(rows: Sequence[CompleteObjectRow]) -> dict[str, object]:
    row_ids = [row.row_id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise CompactionHomotopyError("ledger row ids are not unique")
    return {
        "schema": LEDGER_SCHEMA,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rate_accounting": "actual_whole_archive_only",
        "row_count": len(rows),
        "row_ids": row_ids,
        "pareto_row_ids": list(pareto_frontier(rows)),
        "rows": [row.as_dict() for row in rows],
    }


__all__ = [
    "EXPECTED_PAIR_COUNT",
    "INTERACTION_SCHEMA",
    "LEDGER_SCHEMA",
    "ROW_SCHEMA",
    "ActionKind",
    "CompactionHomotopyError",
    "CompleteObjectRow",
    "ReceiverCallback",
    "ReceiverMeasurement",
    "ScorerCallback",
    "ScorerMeasurement",
    "append_row",
    "contest_score",
    "ledger_receipt",
    "measure_complete_object_row",
    "measured_interaction",
    "pareto_frontier",
]
