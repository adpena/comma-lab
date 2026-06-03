# SPDX-License-Identifier: MIT
"""Shared contest-scorer geometry contract for NeRV analysis surfaces.

This module is formula and geometry authority only. It never turns advisory
planner rows into score, rank, kill, or promotion authority; byte-closed
``upstream/evaluate.py`` artifacts remain the scorer source of truth.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES

PEIRCE_P1_SCORER_GEOMETRY_SCHEMA = "peirce_p1_nerv_contest_scorer_geometry.v1"

FALSE_AUTHORITY_FLAGS: dict[str, Any] = {
    "authority": "false_authority_scorer_geometry_contract_no_score_claim",
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
}

SCORER_ONLY_OBJECTIVE_AUTHORITY: dict[str, Any] = {
    "objective": "contest_auth_eval_scorer_only",
    "allowed_selection_terms": [
        "SegNet_last_frame_distortion",
        "PoseNet_pair_distortion",
        "archive_zip_bytes_rate_term",
    ],
    "forbidden_selection_terms": [
        "human_visual_fidelity",
        "PSNR",
        "SSIM",
        "LPIPS",
        "perceptual_quality_unless_proven_scorer_causal",
    ],
    "rule": (
        "human visual fidelity is not an authority surface; optimize only the contest auth eval scorer and byte price"
    ),
    **FALSE_AUTHORITY_FLAGS,
}


class ContestScorerGeometryError(ValueError):
    """Raised when a scorer-geometry request would create false authority."""


@dataclass(frozen=True)
class ContestScorerGeometry:
    """Peirce P1 scorer geometry and formula constants.

    Shape conventions mirror the pinned upstream evaluator:
    ``camera_size`` is width x height, while PyTorch interpolate sizes are
    height x width.
    """

    schema: str = PEIRCE_P1_SCORER_GEOMETRY_SCHEMA
    original_video_bytes: int = ORIGINAL_VIDEO_BYTES
    seq_len: int = 2
    camera_width: int = 1164
    camera_height: int = 874
    scorer_input_width: int = 512
    scorer_input_height: int = 384
    segnet_score_coefficient: float = 100.0
    pose_score_scale: float = 10.0
    rate_score_coefficient: float = 25.0
    segnet_active_pair_frame_index: int = 1

    @property
    def camera_size_wh(self) -> tuple[int, int]:
        return (self.camera_width, self.camera_height)

    @property
    def scorer_input_size_wh(self) -> tuple[int, int]:
        return (self.scorer_input_width, self.scorer_input_height)

    @property
    def scorer_interpolate_size_hw(self) -> tuple[int, int]:
        return (self.scorer_input_height, self.scorer_input_width)

    @property
    def rate_score_per_byte(self) -> float:
        return self.rate_score_coefficient / float(self.original_video_bytes)

    @property
    def per_byte_marginal(self) -> float:
        return self.rate_score_per_byte

    @property
    def segnet_marginal_coefficient(self) -> float:
        return self.segnet_score_coefficient

    def formula_value(
        self,
        *,
        seg_dist: float,
        pose_dist: float,
        archive_bytes: int,
    ) -> float:
        """Return the official formula value without granting score authority."""

        seg = _finite_nonnegative_float(seg_dist, "seg_dist")
        pose = _finite_nonnegative_float(pose_dist, "pose_dist")
        if isinstance(archive_bytes, bool) or int(archive_bytes) != archive_bytes:
            raise ContestScorerGeometryError("archive_bytes must be an integer")
        if int(archive_bytes) < 0:
            raise ContestScorerGeometryError("archive_bytes must be nonnegative")
        return (
            self.segnet_score_coefficient * seg
            + math.sqrt(self.pose_score_scale * pose)
            + self.rate_score_coefficient * int(archive_bytes) / float(self.original_video_bytes)
        )

    def pose_marginal_coefficient(self, d_pose: float) -> float:
        """Return ``d sqrt(10*d_pose) / d d_pose`` and fail closed at zero."""

        pose = _finite_nonnegative_float(d_pose, "d_pose")
        if pose <= 0.0:
            raise ContestScorerGeometryError("d_pose must be positive for pose marginal coefficient")
        return 5.0 / math.sqrt(self.pose_score_scale * pose)

    def segnet_pair_frame_mask(self, *, num_pairs: int = 1) -> tuple[float, ...]:
        """Return frame-level SegNet activity: frame0 off, frame1 on per pair."""

        pairs = _positive_int(num_pairs, "num_pairs")
        one_pair = tuple(
            1.0 if frame_idx == self.segnet_active_pair_frame_index else 0.0 for frame_idx in range(self.seq_len)
        )
        return one_pair * pairs

    def posenet_pair_frame_mask(self, *, num_pairs: int = 1) -> tuple[float, ...]:
        """Return frame-level PoseNet activity: both frames active per pair."""

        pairs = _positive_int(num_pairs, "num_pairs")
        return (1.0,) * (pairs * self.seq_len)

    def as_false_authority_payload(self) -> dict[str, Any]:
        """Return a machine-readable contract payload with non-claim flags."""

        return {
            **FALSE_AUTHORITY_FLAGS,
            **asdict(self),
            "camera_size_wh": self.camera_size_wh,
            "scorer_input_size_wh": self.scorer_input_size_wh,
            "scorer_interpolate_size_hw": self.scorer_interpolate_size_hw,
            "rate_score_per_byte": self.rate_score_per_byte,
            "per_byte_marginal": self.per_byte_marginal,
            "segnet_pair_frame_mask": self.segnet_pair_frame_mask(),
            "posenet_pair_frame_mask": self.posenet_pair_frame_mask(),
            "segnet_contract": "x[:, -1, ...] last frame only",
            "posenet_contract": "seq_len=2 pair consumes both frames",
        }


def _finite_nonnegative_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ContestScorerGeometryError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ContestScorerGeometryError(f"{name} must be finite")
    if parsed < 0.0:
        raise ContestScorerGeometryError(f"{name} must be nonnegative")
    return parsed


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ContestScorerGeometryError(f"{name} must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise ContestScorerGeometryError(f"{name} must be positive")
    return parsed


PEIRCE_P1_CONTEST_SCORER_GEOMETRY = ContestScorerGeometry()

__all__ = [
    "FALSE_AUTHORITY_FLAGS",
    "PEIRCE_P1_CONTEST_SCORER_GEOMETRY",
    "PEIRCE_P1_SCORER_GEOMETRY_SCHEMA",
    "SCORER_ONLY_OBJECTIVE_AUTHORITY",
    "ContestScorerGeometry",
    "ContestScorerGeometryError",
]
