# SPDX-License-Identifier: MIT
"""Scorer-free collateral pricing for decoded-context conditioner proposals.

The proposal-time surface is deliberately small: a decoded semantic context
code and a counted gate payload.  Segmentation logits, labels, margins, and
realized B/H outcomes are accepted only by the offline fitting helpers; they
are not inputs to :class:`CollateralSuppressedProposer`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import brotli
import numpy as np

N_CLASSES: Final = 5
ORIENTED_CONTEXTS: Final = N_CLASSES**5
PIXELS: Final = 600 * 384 * 512
SOURCE_VIDEO_BYTES: Final = 37_545_489
SEG_SCORE_PER_NET_FLIP: Final = 100.0 / PIXELS
RATE_SCORE_PER_BYTE: Final = 25.0 / SOURCE_VIDEO_BYTES
NET_FLIPS_PER_BYTE: Final = RATE_SCORE_PER_BYTE / SEG_SCORE_PER_NET_FLIP
MAGIC: Final = b"EC2G\x01"


class EC2ProposerError(ValueError):
    """Raised when a collateral gate or priced proposal is invalid."""


@dataclass(frozen=True, slots=True)
class CollateralPricedDelta:
    """Expected complete-objective delta for one proposed support."""

    expected_beneficial: float
    expected_harmful: float
    delta_archive_bytes: int
    delta_pose_score: float
    segmentation_score: float
    rate_score: float
    joint_score: float

    @property
    def accepted(self) -> bool:
        """Return whether the complete expected delta is strictly beneficial."""

        return self.joint_score < 0.0


@dataclass(frozen=True, slots=True)
class ContextCounts:
    """Offline B/H counts for one decoded-context vocabulary."""

    beneficial: np.ndarray
    harmful: np.ndarray

    def __post_init__(self) -> None:
        beneficial = np.asarray(self.beneficial)
        harmful = np.asarray(self.harmful)
        if beneficial.shape != harmful.shape or beneficial.ndim != 1:
            raise EC2ProposerError("beneficial and harmful counts must be matching vectors")
        if not np.issubdtype(beneficial.dtype, np.integer) or not np.issubdtype(
            harmful.dtype, np.integer
        ):
            raise EC2ProposerError("context counts must be integer-valued")
        if np.any(beneficial < 0) or np.any(harmful < 0):
            raise EC2ProposerError("context counts must be nonnegative")

    @property
    def observations(self) -> np.ndarray:
        return np.asarray(self.beneficial, dtype=np.int64) + np.asarray(
            self.harmful, dtype=np.int64
        )


@dataclass(frozen=True, slots=True)
class CollateralSuppressedProposer:
    """Apply a counted decoded-context gate without scorer access."""

    keep_by_context: np.ndarray

    def __post_init__(self) -> None:
        keep = np.asarray(self.keep_by_context)
        if keep.shape != (ORIENTED_CONTEXTS,) or keep.dtype != np.bool_:
            raise EC2ProposerError(
                f"gate must have shape ({ORIENTED_CONTEXTS},) and dtype bool"
            )

    def propose(self, context_codes: np.ndarray) -> np.ndarray:
        """Return the support admitted by decoded context alone."""

        codes = np.asarray(context_codes)
        if not np.issubdtype(codes.dtype, np.integer):
            raise EC2ProposerError("context codes must be integer-valued")
        if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= ORIENTED_CONTEXTS):
            raise EC2ProposerError("context code is outside the oriented vocabulary")
        return np.asarray(self.keep_by_context)[codes]

    def to_payload(self, *, quality: int = 11) -> bytes:
        """Serialize the video-derived gate as a deterministic counted payload."""

        if quality < 0 or quality > 11:
            raise EC2ProposerError("Brotli quality must be in [0, 11]")
        packed = np.packbits(np.asarray(self.keep_by_context), bitorder="little").tobytes()
        raw = MAGIC + ORIENTED_CONTEXTS.to_bytes(2, "little") + packed
        return brotli.compress(raw, quality=quality)

    @classmethod
    def from_payload(cls, payload: bytes) -> CollateralSuppressedProposer:
        """Parse and validate one counted gate payload."""

        try:
            raw = brotli.decompress(payload)
        except brotli.error as exc:
            raise EC2ProposerError("invalid Brotli gate payload") from exc
        if not raw.startswith(MAGIC) or len(raw) < len(MAGIC) + 2:
            raise EC2ProposerError("invalid EC2 gate magic")
        count = int.from_bytes(raw[len(MAGIC) : len(MAGIC) + 2], "little")
        if count != ORIENTED_CONTEXTS:
            raise EC2ProposerError("gate vocabulary size differs")
        packed = raw[len(MAGIC) + 2 :]
        expected_bytes = math.ceil(ORIENTED_CONTEXTS / 8)
        if len(packed) != expected_bytes:
            raise EC2ProposerError("gate bit payload length differs")
        unpacked = np.unpackbits(
            np.frombuffer(packed, dtype=np.uint8), bitorder="little"
        )[:ORIENTED_CONTEXTS]
        return cls(unpacked.astype(np.bool_))


def collateral_priced_delta(
    *,
    expected_beneficial: float,
    expected_harmful: float,
    delta_archive_bytes: int,
    delta_pose_score: float = 0.0,
) -> CollateralPricedDelta:
    """Price expected B and H together with rate and pose.

    ``delta_pose_score`` is already in contest-score units.  A proposal cannot
    receive credit for B without paying the symmetric segmentation price for H.
    """

    values = (expected_beneficial, expected_harmful, delta_pose_score)
    if not all(math.isfinite(float(value)) for value in values):
        raise EC2ProposerError("priced inputs must be finite")
    if expected_beneficial < 0.0 or expected_harmful < 0.0:
        raise EC2ProposerError("expected B/H counts must be nonnegative")
    if isinstance(delta_archive_bytes, bool) or not isinstance(delta_archive_bytes, int):
        raise EC2ProposerError("delta_archive_bytes must be an integer")
    segmentation = SEG_SCORE_PER_NET_FLIP * (expected_harmful - expected_beneficial)
    rate = RATE_SCORE_PER_BYTE * delta_archive_bytes
    joint = segmentation + float(delta_pose_score) + rate
    return CollateralPricedDelta(
        expected_beneficial=float(expected_beneficial),
        expected_harmful=float(expected_harmful),
        delta_archive_bytes=delta_archive_bytes,
        delta_pose_score=float(delta_pose_score),
        segmentation_score=segmentation,
        rate_score=rate,
        joint_score=joint,
    )


def oriented_context_codes_at(
    tokens: np.ndarray,
    frame: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Encode center/left/right/up/down decoded classes at selected cells."""

    source = np.asarray(tokens)
    if source.ndim != 3 or not np.issubdtype(source.dtype, np.integer):
        raise EC2ProposerError("tokens must be an integer [frame, y, x] array")
    frame_i = np.asarray(frame, dtype=np.int64)
    y_i = np.asarray(y, dtype=np.int64)
    x_i = np.asarray(x, dtype=np.int64)
    if frame_i.shape != y_i.shape or frame_i.shape != x_i.shape:
        raise EC2ProposerError("coordinate vectors must have matching shape")
    if frame_i.ndim != 1:
        raise EC2ProposerError("coordinate vectors must be one-dimensional")
    if frame_i.size and (
        frame_i.min() < 0
        or frame_i.max() >= source.shape[0]
        or y_i.min() < 0
        or y_i.max() >= source.shape[1]
        or x_i.min() < 0
        or x_i.max() >= source.shape[2]
    ):
        raise EC2ProposerError("coordinate is outside token geometry")
    left_x = np.maximum(x_i - 1, 0)
    right_x = np.minimum(x_i + 1, source.shape[2] - 1)
    up_y = np.maximum(y_i - 1, 0)
    down_y = np.minimum(y_i + 1, source.shape[1] - 1)
    center = source[frame_i, y_i, x_i].astype(np.uint16)
    left = source[frame_i, y_i, left_x].astype(np.uint16)
    right = source[frame_i, y_i, right_x].astype(np.uint16)
    up = source[frame_i, up_y, x_i].astype(np.uint16)
    down = source[frame_i, down_y, x_i].astype(np.uint16)
    if any(int(value.max(initial=0)) >= N_CLASSES for value in (center, left, right, up, down)):
        raise EC2ProposerError("decoded token class is outside [0, 4]")
    return center + 5 * left + 25 * right + 125 * up + 625 * down


def fit_context_counts(
    context_codes: np.ndarray,
    outcomes: np.ndarray,
    *,
    buckets: int = ORIENTED_CONTEXTS,
) -> ContextCounts:
    """Fit offline counts where outcomes are +1 for B and -1 for H."""

    codes = np.asarray(context_codes)
    labels = np.asarray(outcomes)
    if codes.shape != labels.shape or codes.ndim != 1:
        raise EC2ProposerError("context codes and outcomes must be matching vectors")
    if not np.issubdtype(codes.dtype, np.integer):
        raise EC2ProposerError("context codes must be integer-valued")
    if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= buckets):
        raise EC2ProposerError("context code is outside bucket range")
    if not np.all(np.isin(labels, (-1, 1))):
        raise EC2ProposerError("outcomes must be +1 (B) or -1 (H)")
    beneficial = np.bincount(codes[labels == 1], minlength=buckets).astype(np.int64)
    harmful = np.bincount(codes[labels == -1], minlength=buckets).astype(np.int64)
    return ContextCounts(beneficial=beneficial, harmful=harmful)


def gate_from_context_counts(
    counts: ContextCounts,
    *,
    minimum_beneficial_fraction: float,
    minimum_observations: int = 1,
    prior_beneficial: float = 0.5,
    prior_strength: float = 1.0,
) -> CollateralSuppressedProposer:
    """Build a gate from smoothed offline B/H rates.

    The fitted gate is video-derived and therefore counted.  Runtime admission
    consumes only its bits and decoded semantic context.
    """

    scalars = (minimum_beneficial_fraction, prior_beneficial, prior_strength)
    if not all(math.isfinite(float(value)) for value in scalars):
        raise EC2ProposerError("gate parameters must be finite")
    if not 0.0 <= minimum_beneficial_fraction <= 1.0:
        raise EC2ProposerError("minimum_beneficial_fraction must be in [0, 1]")
    if not 0.0 <= prior_beneficial <= 1.0 or prior_strength < 0.0:
        raise EC2ProposerError("invalid smoothing prior")
    if minimum_observations < 0:
        raise EC2ProposerError("minimum_observations must be nonnegative")
    observations = counts.observations
    numerator = np.asarray(counts.beneficial, dtype=np.float64) + (
        prior_beneficial * prior_strength
    )
    denominator = observations.astype(np.float64) + prior_strength
    posterior = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, prior_beneficial),
        where=denominator > 0.0,
    )
    keep = (observations >= minimum_observations) & (
        posterior >= minimum_beneficial_fraction
    )
    if keep.shape != (ORIENTED_CONTEXTS,):
        raise EC2ProposerError("only the oriented 3,125-context vocabulary is shippable")
    return CollateralSuppressedProposer(keep.astype(np.bool_))


__all__ = [
    "MAGIC",
    "NET_FLIPS_PER_BYTE",
    "ORIENTED_CONTEXTS",
    "RATE_SCORE_PER_BYTE",
    "SEG_SCORE_PER_NET_FLIP",
    "CollateralPricedDelta",
    "CollateralSuppressedProposer",
    "ContextCounts",
    "EC2ProposerError",
    "collateral_priced_delta",
    "fit_context_counts",
    "gate_from_context_counts",
    "oriented_context_codes_at",
]
