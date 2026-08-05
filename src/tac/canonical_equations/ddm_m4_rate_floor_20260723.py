# SPDX-License-Identifier: MIT
"""Canonical laws for the DDM M4 rate floor at a receiver-closed distortion row.

This module deliberately separates three quantities that are easy to conflate:

* a universal information-theoretic lower bound (currently only zero bytes);
* the smallest receiver-closed row in an explicitly audited candidate set; and
* the strict archive cap needed to cross a target score at fixed distortion.

An audited minimum is an upper bound on the unknown global MDL optimum.  It must
never be presented as a proof that a smaller legal program does not exist.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

ORIGINAL_UNCOMPRESSED_BYTES = 37_545_489
RATE_NUMERATOR = 25

SETTLED_D_SEG = Decimal("0.00015196")
SETTLED_D_POSE = Decimal("0.00010184")
DELEGATED_MAX_D_SEG = Decimal("0.00116")
DELEGATED_MAX_D_POSE = Decimal("0.00161")
SUB015_SCORE = Decimal("0.15")

UINT8_SCHEDULED_RECOVERY_S = Decimal("0.01583")
UINT8_REALIZED_FRACTION = Decimal("-0.014")


@dataclass(frozen=True, slots=True)
class ReceiverRow:
    """One complete same-artifact receiver/evaluator row."""

    row_id: str
    archive_bytes: int
    d_seg: Decimal
    d_pose: Decimal
    n_pairs: int
    receiver_closed: bool
    evidence_axis: str

    def __post_init__(self) -> None:
        if not self.row_id or not self.evidence_axis:
            raise ValueError("row_id and evidence_axis must be nonempty")
        if self.archive_bytes < 0:
            raise ValueError("archive_bytes must be nonnegative")
        if self.d_seg < 0 or self.d_pose < 0:
            raise ValueError("distortions must be nonnegative")
        if self.n_pairs <= 0:
            raise ValueError("n_pairs must be positive")


@dataclass(frozen=True, slots=True)
class LeverPool:
    """A non-additive description pool and its scoped composition rule."""

    pool_id: str
    levers: tuple[str, ...]
    relationship: str
    evidence_scope: str


LEVER_POOLS = (
    LeverPool(
        pool_id="P_REALIZE",
        levers=("multicoefficient-solve", "correction-synergy"),
        relationship="COMPETE_WITHIN_POOL",
        evidence_scope=(
            "Both change the same realized receiver field. Credit is the strict joint "
            "replay delta, never the sum of singleton deltas."
        ),
    ),
    LeverPool(
        pool_id="P_TEMPORAL_DESCRIPTION",
        levers=("context-arithmetic-code", "xi-once-for-pose", "chart-canonicalization"),
        relationship="COMPETE_WITHIN_POOL",
        evidence_scope=(
            "All harvest redundancy from the post-chart temporal symbol stream. The "
            "xi-temporal coder is downstream of the canonical chart and cannot receive "
            "independent additive credit."
        ),
    ),
    LeverPool(
        pool_id="P_FRAME_OWNERSHIP",
        levers=("frame-separation",),
        relationship="ORTHOGONAL_FOR_DSEG_ONLY",
        evidence_scope=(
            "SegNet reads the last frame only, so frame-0 is structurally seg-free. "
            "Pose still couples the pair, so archive-byte credit requires joint replay."
        ),
    ),
    LeverPool(
        pool_id="P_NULL_GAUGE",
        levers=("ker(A)-hide",),
        relationship="ORTHOGONAL_GEOMETRY_NOT_RATE",
        evidence_scope=(
            "COUNTED_PAYLOAD_RATE_CREDIT only: nullity / blind-mask area / "
            "range(A)-complement energy are scorer-invariant geometry, not a "
            "priced rate column. Archive-byte credit exists only when a "
            "parser-consumed counted payload is actually removed."
        ),
    ),
)


def score_terms(d_seg: Decimal, d_pose: Decimal, archive_bytes: int) -> dict[str, Decimal]:
    """Return the exact-decimal score decomposition."""

    if d_seg < 0 or d_pose < 0 or archive_bytes < 0:
        raise ValueError("distortions and archive bytes must be nonnegative")
    seg = Decimal(100) * d_seg
    pose = (Decimal(10) * d_pose).sqrt()
    rate = Decimal(RATE_NUMERATOR * archive_bytes) / Decimal(ORIGINAL_UNCOMPRESSED_BYTES)
    return {"seg": seg, "pose": pose, "rate": rate, "total": seg + pose + rate}


def strict_archive_cap_bytes(
    d_seg: Decimal,
    d_pose: Decimal,
    *,
    target_score: Decimal = SUB015_SCORE,
) -> int:
    """Largest integer byte count whose score is strictly below ``target_score``."""

    nonrate = score_terms(d_seg, d_pose, 0)["total"]
    if nonrate >= target_score:
        raise ValueError("nonrate score already meets or exceeds target_score")
    crossing = (target_score - nonrate) * Decimal(ORIGINAL_UNCOMPRESSED_BYTES) / Decimal(RATE_NUMERATOR)
    floor_value = int(crossing.to_integral_value(rounding=ROUND_FLOOR))
    if crossing == Decimal(floor_value):
        return floor_value - 1
    return floor_value


def minimum_admissible_receiver_row(
    rows: Iterable[ReceiverRow],
    *,
    max_d_seg: Decimal = DELEGATED_MAX_D_SEG,
    max_d_pose: Decimal = DELEGATED_MAX_D_POSE,
    required_pairs: int = 600,
) -> ReceiverRow:
    """Select the smallest complete receiver row inside the declared distortion box."""

    eligible = [
        row
        for row in rows
        if row.receiver_closed and row.n_pairs == required_pairs and row.d_seg <= max_d_seg and row.d_pose <= max_d_pose
    ]
    if not eligible:
        raise ValueError("no admissible receiver-closed row in the audited set")
    return min(eligible, key=lambda row: (row.archive_bytes, row.row_id))


def uint8_unrecovered_scheduled_debt(
    *,
    scheduled_recovery_s: Decimal = UINT8_SCHEDULED_RECOVERY_S,
    realized_fraction: Decimal = UINT8_REALIZED_FRACTION,
) -> Decimal:
    """Scheduled score recovery not realized after the integer receiver gate.

    A negative realized fraction means the receiver regressed.  The returned
    value is an unmet scheduled debt, not a measured recoverable gain.
    """

    if scheduled_recovery_s < 0:
        raise ValueError("scheduled_recovery_s must be nonnegative")
    return scheduled_recovery_s * (Decimal(1) - realized_fraction)


def lever_pool_map() -> dict[str, str]:
    """Return the unique lever-to-pool assignment and reject drift."""

    out: dict[str, str] = {}
    for pool in LEVER_POOLS:
        for lever in pool.levers:
            if lever in out:
                raise ValueError(f"lever appears in multiple pools: {lever}")
            out[lever] = pool.pool_id
    return out


__all__ = [
    "DELEGATED_MAX_D_POSE",
    "DELEGATED_MAX_D_SEG",
    "LEVER_POOLS",
    "ORIGINAL_UNCOMPRESSED_BYTES",
    "SETTLED_D_POSE",
    "SETTLED_D_SEG",
    "SUB015_SCORE",
    "UINT8_REALIZED_FRACTION",
    "UINT8_SCHEDULED_RECOVERY_S",
    "ReceiverRow",
    "lever_pool_map",
    "minimum_admissible_receiver_row",
    "score_terms",
    "strict_archive_cap_bytes",
    "uint8_unrecovered_scheduled_debt",
]
