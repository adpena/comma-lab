# SPDX-License-Identifier: MIT
"""Exact cap-free admission for receiver-realized contest-objective moves.

Collateral flips are diagnostic observables, never feasibility constraints.
The only admission authority is the strict delta of the contest-priced
objective evaluated on exact receiver output and exact archive bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

SOURCE_VIDEO_BYTES: Final = 37_545_489
SEGMENTATION_WEIGHT: Final = 100.0
POSE_SCALE: Final = 10.0
RATE_WEIGHT: Final = 25.0


class PurePricedObjectiveError(ValueError):
    """Raised when an alleged realized measurement is not admissible evidence."""


@dataclass(frozen=True, slots=True)
class RealizedObjectiveState:
    """One exact receiver/scorer/archive state."""

    d_seg: float
    d_pose: float
    archive_bytes: int

    def __post_init__(self) -> None:
        values = (float(self.d_seg), float(self.d_pose))
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise PurePricedObjectiveError("realized distortions must be finite and nonnegative")
        if isinstance(self.archive_bytes, bool) or not isinstance(self.archive_bytes, int):
            raise PurePricedObjectiveError("archive_bytes must be an integer")
        if self.archive_bytes < 0:
            raise PurePricedObjectiveError("archive_bytes must be nonnegative")

    @property
    def objective(self) -> float:
        return (
            SEGMENTATION_WEIGHT * float(self.d_seg)
            + math.sqrt(POSE_SCALE * float(self.d_pose))
            + RATE_WEIGHT * int(self.archive_bytes) / SOURCE_VIDEO_BYTES
        )


@dataclass(frozen=True, slots=True)
class PurePricedRealizedDelta:
    """Exact additive decomposition used for one cap-free decision."""

    seg_term: float
    pose_term: float
    rate_term: float
    joint_delta: float
    accepted: bool


def pure_priced_realized_delta(
    before: RealizedObjectiveState,
    after: RealizedObjectiveState,
) -> PurePricedRealizedDelta:
    """Return the exact contest-priced delta and strict admission decision.

    No collateral count, role-local regression, proxy loss, ranker score, or
    trust-region diagnostic is accepted by this interface.  They may be
    reported beside this result, but cannot alter ``accepted``.
    """

    seg = SEGMENTATION_WEIGHT * (float(after.d_seg) - float(before.d_seg))
    pose = math.sqrt(POSE_SCALE * float(after.d_pose)) - math.sqrt(
        POSE_SCALE * float(before.d_pose)
    )
    rate = RATE_WEIGHT * (int(after.archive_bytes) - int(before.archive_bytes)) / SOURCE_VIDEO_BYTES
    joint = seg + pose + rate
    if not all(math.isfinite(value) for value in (seg, pose, rate, joint)):
        raise PurePricedObjectiveError("realized objective delta is nonfinite")
    return PurePricedRealizedDelta(
        seg_term=seg,
        pose_term=pose,
        rate_term=rate,
        joint_delta=joint,
        accepted=joint < 0.0,
    )


def break_even_distortion_gain_per_byte() -> float:
    """Return the exact rate price used by reverse-waterfill stopping."""

    return RATE_WEIGHT / SOURCE_VIDEO_BYTES


__all__ = [
    "POSE_SCALE",
    "RATE_WEIGHT",
    "SEGMENTATION_WEIGHT",
    "SOURCE_VIDEO_BYTES",
    "PurePricedObjectiveError",
    "PurePricedRealizedDelta",
    "RealizedObjectiveState",
    "break_even_distortion_gain_per_byte",
    "pure_priced_realized_delta",
]
