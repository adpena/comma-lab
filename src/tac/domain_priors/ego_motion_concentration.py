# SPDX-License-Identifier: MIT
"""Ego-motion concentration atlas — wraps tac.ego_flow + FOE / pose-anchor prior.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 Step 4.1 line 405: this wrapper
exposes a per-frame ego-motion magnitude surface aligned with the
Atick-Redlich 1990 cooperative-receiver lineage + Rao-Ballard 1999
predictive-coding hierarchical-Bayesian structure (both anchored in
``feedback_six_meta_pattern_strict_gates_d_e_f_g_h_i_landed_20260516.md``
+ Z6/Z7/Z8 design memos per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via
adversarial grand council symposium" non-negotiable).

The signal comes from two complementary surfaces:

  1. **Pose-anchor magnitude**: from a per-pair pose vector (PoseNet output
     or a learned-affine-flow row), compute the L2 norm of the 6-DOF pose
     change. This is the "magnitude" surface.

  2. **Affine-flow concentration**: from a per-pair affine flow
     (``tac.ego_flow.LearnableAffineFlow`` archive bytes), compute a
     concentration metric — how spread out the flow is across the frame,
     vs concentrated in a localized region. Higher concentration =
     localized motion (e.g. wiper, blinker, obstacle); lower concentration
     = uniform ego-motion (vehicle's own translation/rotation).

Both surfaces ARE per-frame predictive-coding priors per CLAUDE.md
"PER-SUBSTRATE OPTIMAL FORM" Catalog #311 (ego-motion-conditioning is the
canonical predictive-coding form for dashcam video).

Cross-references:
  * CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 Step 4.1 line 405
  * ``tac.ego_flow`` — canonical helper this module wraps
  * Catalog #311 ``check_predictive_coding_substrate_design_has_ego_motion_conditioning``
  * Catalog #287 / #323 (axis tag + Provenance discipline)
  * CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council
    symposium" (Z6/Z7/Z8 substrate-class lineage)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from tac.provenance.contract import Provenance


_POSE_DOF_CANONICAL = 6
"""6 degrees-of-freedom for canonical pose: tx, ty, tz, rx, ry, rz."""


@dataclass(frozen=True)
class EgoMotionConcentrationEntry:
    """One frame's ego-motion concentration signal.

    Fields:
        frame_index: 0-indexed frame in the video.
        pose_magnitude_l2: L2 norm of the 6-DOF pose change relative to
            adjacent frame. Higher = more ego-motion.
        flow_concentration: dimensionless [0, 1] metric. 0 = flow uniformly
            distributed across frame (pure ego-motion / translation); 1 =
            flow concentrated in a localized region (occlusion / blinker /
            non-ego signal).
        ego_motion_score: dimensionless aggregated score combining
            magnitude and concentration. Operator-readable for ranking.
    """

    frame_index: int
    pose_magnitude_l2: float
    flow_concentration: float
    ego_motion_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError(
                f"frame_index={self.frame_index!r} must be non-negative int"
            )
        for fname, fval in (
            ("pose_magnitude_l2", self.pose_magnitude_l2),
            ("flow_concentration", self.flow_concentration),
            ("ego_motion_score", self.ego_motion_score),
        ):
            if not isinstance(fval, (int, float)):
                raise TypeError(f"{fname} must be numeric, got {type(fval).__name__}")
            if fval != fval:
                raise ValueError(f"{fname} must not be NaN")
            if fval < 0:
                raise ValueError(f"{fname}={fval} must be >= 0")
        if self.flow_concentration > 1.0 + 1e-9:
            raise ValueError(
                f"flow_concentration={self.flow_concentration} must be in [0, 1]"
            )


@dataclass(frozen=True)
class EgoMotionConcentrationAtlas:
    """Per-frame ego-motion concentration atlas — canonical wrapper output.

    Fields:
        total_frames: number of frames in the video.
        entries: per-frame entries sorted by frame_index ascending.
        top_k_ego_motion_intense_frame_indices: descending by ego_motion_score.
        source_anchor_kind: ``"pose_vector"`` / ``"affine_flow"`` /
            ``"raft_optical_flow"`` (cite the canonical signal source).
        source_archive_sha256: archive sha the ego-motion signal was
            derived from; cite-chain back.
        source_measurement_axis: e.g. ``"[predicted]"`` / ``"[contest-CPU]"``.
        atick_redlich_alignment_tag: declares this atlas is constructed
            per the Atick-Redlich 1990 cooperative-receiver framing
            (the canonical predictive-coding alignment for dashcam video).
        provenance: canonical Provenance per Catalog #323.
    """

    total_frames: int
    entries: tuple[EgoMotionConcentrationEntry, ...]
    top_k_ego_motion_intense_frame_indices: tuple[int, ...]
    source_anchor_kind: str
    source_archive_sha256: str
    source_measurement_axis: str
    atick_redlich_alignment_tag: str
    provenance: Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.total_frames, int) or self.total_frames <= 0:
            raise ValueError(
                f"total_frames={self.total_frames!r} must be a positive int"
            )
        if not isinstance(self.entries, tuple) or len(self.entries) != self.total_frames:
            raise ValueError(
                f"entries length {len(self.entries) if isinstance(self.entries, tuple) else 'N/A'} "
                f"must equal total_frames {self.total_frames}"
            )
        for i, entry in enumerate(self.entries):
            if not isinstance(entry, EgoMotionConcentrationEntry):
                raise TypeError(
                    f"entries[{i}] must be EgoMotionConcentrationEntry, got {type(entry).__name__}"
                )
            if entry.frame_index != i:
                raise ValueError(
                    f"entries[{i}].frame_index={entry.frame_index} must equal i={i}"
                )
        if not isinstance(self.top_k_ego_motion_intense_frame_indices, tuple):
            raise TypeError("top_k_ego_motion_intense_frame_indices must be tuple")
        VALID_SOURCE_KINDS = {"pose_vector", "affine_flow", "raft_optical_flow"}
        if self.source_anchor_kind not in VALID_SOURCE_KINDS:
            raise ValueError(
                f"source_anchor_kind={self.source_anchor_kind!r} must be one of {sorted(VALID_SOURCE_KINDS)!r}"
            )
        if (
            not isinstance(self.source_archive_sha256, str)
            or len(self.source_archive_sha256) != 64
            or not all(c in "0123456789abcdef" for c in self.source_archive_sha256)
        ):
            raise ValueError(
                f"source_archive_sha256={self.source_archive_sha256!r} must be 64-char hex sha256"
            )
        if not isinstance(self.source_measurement_axis, str) or not self.source_measurement_axis:
            raise ValueError("source_measurement_axis must be non-empty string")
        if not isinstance(self.atick_redlich_alignment_tag, str) or not self.atick_redlich_alignment_tag:
            raise ValueError(
                "atick_redlich_alignment_tag must be non-empty string per Catalog #311"
            )
        if not isinstance(self.provenance, Provenance):
            raise TypeError(
                f"provenance must be Provenance, got {type(self.provenance).__name__}"
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe serialization."""
        from tac.provenance.validator import provenance_to_dict

        return {
            "schema": "ego_motion_concentration_atlas_v1",
            "total_frames": self.total_frames,
            "source_anchor_kind": self.source_anchor_kind,
            "source_archive_sha256": self.source_archive_sha256,
            "source_measurement_axis": self.source_measurement_axis,
            "atick_redlich_alignment_tag": self.atick_redlich_alignment_tag,
            "entries": [
                {
                    "frame_index": e.frame_index,
                    "pose_magnitude_l2": e.pose_magnitude_l2,
                    "flow_concentration": e.flow_concentration,
                    "ego_motion_score": e.ego_motion_score,
                }
                for e in self.entries
            ],
            "top_k_ego_motion_intense_frame_indices": list(
                self.top_k_ego_motion_intense_frame_indices
            ),
            "provenance": provenance_to_dict(self.provenance),
        }


def _validate_pose_vector(pose_vector: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Coerce + validate pose vector to canonical 6-DOF rows."""
    if not isinstance(pose_vector, Sequence) or isinstance(
        pose_vector, (str, bytes, bytearray)
    ):
        raise TypeError(
            f"pose_vector must be Sequence; got {type(pose_vector).__name__}"
        )
    if len(pose_vector) == 0:
        raise ValueError("pose_vector must be non-empty (>=1 pair)")
    rows: list[tuple[float, ...]] = []
    for i, row in enumerate(pose_vector):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise TypeError(
                f"pose_vector[{i}] must be Sequence; got {type(row).__name__}"
            )
        if len(row) != _POSE_DOF_CANONICAL:
            raise ValueError(
                f"pose_vector[{i}] has length {len(row)}; canonical 6-DOF required"
            )
        coerced: list[float] = []
        for j, v in enumerate(row):
            if not isinstance(v, (int, float)) or v != v:
                raise ValueError(
                    f"pose_vector[{i}][{j}]={v!r} must be finite numeric"
                )
            coerced.append(float(v))
        rows.append(tuple(coerced))
    return tuple(rows)


def build_ego_motion_concentration_from_pose_anchors(
    pose_vector: Sequence[Sequence[float]],
    *,
    archive_sha256: str,
    measurement_axis: str,
    provenance: Provenance,
    pair_to_frame_construction: str = "adjacent",
    top_k: int = 50,
) -> EgoMotionConcentrationAtlas:
    """Build :class:`EgoMotionConcentrationAtlas` from per-pair pose vector.

    The pose vector is the canonical 6-DOF pose change per pair, typically
    from PoseNet output or a sister learned-affine-flow per-pair row.

    Per-frame ego-motion magnitude is the L2 norm of incident pair pose
    rows (using the same adjacent / non-overlap construction as
    per_frame_difficulty). Flow concentration is unavailable in the pose-
    only path; defaults to 0.0 (uniform) per the docstring contract.

    Args:
        pose_vector: per-pair pose vectors. Shape (N_pairs, 6).
        archive_sha256: archive sha (cite-chain).
        measurement_axis: axis tag.
        provenance: canonical Provenance per Catalog #323.
        pair_to_frame_construction: ``"adjacent"`` or ``"non_overlap"``.
        top_k: number of most-ego-motion-intense frames to surface.

    Returns:
        Frozen :class:`EgoMotionConcentrationAtlas`.
    """
    rows = _validate_pose_vector(pose_vector)
    total_pairs = len(rows)
    if pair_to_frame_construction == "adjacent":
        total_frames = total_pairs + 1
    elif pair_to_frame_construction == "non_overlap":
        total_frames = total_pairs * 2
    else:
        raise ValueError(
            f"pair_to_frame_construction={pair_to_frame_construction!r} "
            "must be 'adjacent' or 'non_overlap'"
        )
    # Compute per-pair magnitude.
    per_pair_magnitude: list[float] = []
    for row in rows:
        mag = math.sqrt(sum(v * v for v in row))
        per_pair_magnitude.append(mag)
    # Project to per-frame via mean over incident pairs.
    per_frame_magnitude: list[float] = []
    for frame_index in range(total_frames):
        if pair_to_frame_construction == "adjacent":
            incident: list[int] = []
            if frame_index > 0:
                incident.append(frame_index - 1)
            if frame_index < total_pairs:
                incident.append(frame_index)
        else:  # non_overlap
            incident = [frame_index // 2]
        vals = [per_pair_magnitude[p] for p in incident]
        per_frame_magnitude.append(sum(vals) / len(vals) if vals else 0.0)
    # Build entries; flow_concentration defaults to 0.0 in pose-only path.
    entries: list[EgoMotionConcentrationEntry] = []
    for frame_index in range(total_frames):
        mag = per_frame_magnitude[frame_index]
        score = mag  # flow_concentration = 0 contributes 0 to score
        entries.append(
            EgoMotionConcentrationEntry(
                frame_index=frame_index,
                pose_magnitude_l2=mag,
                flow_concentration=0.0,
                ego_motion_score=score,
            )
        )
    # Rank descending.
    ranked = sorted(range(total_frames), key=lambda i: (-entries[i].ego_motion_score, i))
    top_k_clamped = max(0, min(int(top_k), total_frames))
    top_k_indices = tuple(ranked[:top_k_clamped])
    return EgoMotionConcentrationAtlas(
        total_frames=total_frames,
        entries=tuple(entries),
        top_k_ego_motion_intense_frame_indices=top_k_indices,
        source_anchor_kind="pose_vector",
        source_archive_sha256=archive_sha256,
        source_measurement_axis=measurement_axis,
        atick_redlich_alignment_tag=(
            "Atick-Redlich-1990-cooperative-receiver-applied-to-dashcam-pose-vector"
        ),
        provenance=provenance,
    )


def _flow_concentration_from_affine_row(affine_row: Sequence[float]) -> float:
    """Compute concentration metric from a 6-element affine flow row.

    A pure translation (no rotation/skew) → uniform flow → concentration 0.
    A flow with large rotation/skew components → localized → concentration 1.

    The canonical metric: ratio of (rotation_magnitude + skew_magnitude)
    over (translation_magnitude + rotation_magnitude + skew_magnitude),
    bounded [0, 1].
    """
    if len(affine_row) != 6:
        raise ValueError(f"affine_row must have 6 elements; got {len(affine_row)}")
    # Canonical affine 2x3 layout: [a, b, tx, c, d, ty]
    # Rotation/skew terms: a-1, b, c, d-1 (deviation from identity)
    a, b, tx, c, d, ty = (float(v) for v in affine_row)
    rotation_skew = math.sqrt((a - 1.0) ** 2 + b ** 2 + c ** 2 + (d - 1.0) ** 2)
    translation = math.sqrt(tx ** 2 + ty ** 2)
    denom = rotation_skew + translation
    if denom <= 1e-12:
        return 0.0
    return min(1.0, rotation_skew / denom)


def build_ego_motion_concentration_from_affine_flow(
    affine_flow_per_pair: Sequence[Sequence[float]],
    *,
    archive_sha256: str,
    measurement_axis: str,
    provenance: Provenance,
    pair_to_frame_construction: str = "adjacent",
    top_k: int = 50,
) -> EgoMotionConcentrationAtlas:
    """Build :class:`EgoMotionConcentrationAtlas` from per-pair affine flow.

    Wraps ``tac.ego_flow.LearnableAffineFlow``: each per-pair 6-element
    affine row → per-pair (magnitude, concentration) → per-frame
    aggregation.

    Args:
        affine_flow_per_pair: per-pair affine rows. Shape (N_pairs, 6) in
            the canonical [a, b, tx, c, d, ty] layout.
        archive_sha256: archive sha.
        measurement_axis: axis tag.
        provenance: canonical Provenance.
        pair_to_frame_construction: ``"adjacent"`` or ``"non_overlap"``.
        top_k: number of intense frames to surface.

    Returns:
        Frozen :class:`EgoMotionConcentrationAtlas`.
    """
    if not isinstance(affine_flow_per_pair, Sequence) or isinstance(
        affine_flow_per_pair, (str, bytes, bytearray)
    ):
        raise TypeError(
            f"affine_flow_per_pair must be Sequence; got {type(affine_flow_per_pair).__name__}"
        )
    if len(affine_flow_per_pair) == 0:
        raise ValueError("affine_flow_per_pair must be non-empty")
    total_pairs = len(affine_flow_per_pair)
    if pair_to_frame_construction == "adjacent":
        total_frames = total_pairs + 1
    elif pair_to_frame_construction == "non_overlap":
        total_frames = total_pairs * 2
    else:
        raise ValueError(
            f"pair_to_frame_construction={pair_to_frame_construction!r} "
            "must be 'adjacent' or 'non_overlap'"
        )
    # Per-pair magnitude (= translation L2) and concentration.
    per_pair_magnitude: list[float] = []
    per_pair_concentration: list[float] = []
    for row in affine_flow_per_pair:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise TypeError(
                f"affine_flow_per_pair entry must be Sequence; got {type(row).__name__}"
            )
        coerced = [float(v) for v in row]
        if any(v != v for v in coerced):
            raise ValueError("affine_flow_per_pair row contains NaN")
        # Use translation magnitude only for magnitude axis.
        _, _, tx, _, _, ty = coerced
        per_pair_magnitude.append(math.sqrt(tx * tx + ty * ty))
        per_pair_concentration.append(_flow_concentration_from_affine_row(coerced))
    # Project per-pair → per-frame.
    entries: list[EgoMotionConcentrationEntry] = []
    for frame_index in range(total_frames):
        if pair_to_frame_construction == "adjacent":
            incident: list[int] = []
            if frame_index > 0:
                incident.append(frame_index - 1)
            if frame_index < total_pairs:
                incident.append(frame_index)
        else:
            incident = [frame_index // 2]
        mags = [per_pair_magnitude[p] for p in incident]
        cons = [per_pair_concentration[p] for p in incident]
        mag = sum(mags) / len(mags)
        con = sum(cons) / len(cons)
        # Score: magnitude * (1 - concentration) emphasizes ego-motion
        # (uniform flow); high concentration suggests local non-ego signal
        # which is interesting but distinct from ego-motion-intensity.
        score = mag * (1.0 - 0.5 * con)
        entries.append(
            EgoMotionConcentrationEntry(
                frame_index=frame_index,
                pose_magnitude_l2=mag,
                flow_concentration=con,
                ego_motion_score=score,
            )
        )
    ranked = sorted(range(total_frames), key=lambda i: (-entries[i].ego_motion_score, i))
    top_k_clamped = max(0, min(int(top_k), total_frames))
    top_k_indices = tuple(ranked[:top_k_clamped])
    return EgoMotionConcentrationAtlas(
        total_frames=total_frames,
        entries=tuple(entries),
        top_k_ego_motion_intense_frame_indices=top_k_indices,
        source_anchor_kind="affine_flow",
        source_archive_sha256=archive_sha256,
        source_measurement_axis=measurement_axis,
        atick_redlich_alignment_tag=(
            "Atick-Redlich-1990-cooperative-receiver-applied-to-dashcam-affine-flow"
        ),
        provenance=provenance,
    )


__all__ = [
    "EgoMotionConcentrationAtlas",
    "EgoMotionConcentrationEntry",
    "build_ego_motion_concentration_from_pose_anchors",
    "build_ego_motion_concentration_from_affine_flow",
]
