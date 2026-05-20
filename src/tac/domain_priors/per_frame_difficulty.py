# SPDX-License-Identifier: MIT
"""Per-frame difficulty atlas — wraps per-pair atlas + per-frame aggregator.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 Step 4.1: the master-gradient
per-pair difficulty atlas already exists at
``tac.master_gradient_consumers.per_pair_difficulty_atlas``. This wrapper
projects per-pair signal to per-frame difficulty so cathedral autopilot +
bit-allocator can ingest a per-frame surface (the contest video has 1200
frames composed into 600 pairs).

The per-pair → per-frame projection is non-trivial. Per the CARGO-CULT
audit in the master memo:

    "Per-frame difficulty atlas can be derived from per-pair master-gradient
    by aggregation" → CARGO-CULTED-PENDING-EMPIRICAL. Per-pair → per-frame
    aggregation requires choice of aggregation operator (mean / max /
    Volterra); per-pair has dual-frame coupling that per-frame loses.

This wrapper supports 3 canonical aggregation operators and surfaces the
choice explicitly in the output dataclass (per Catalog #305 observability
surface + Catalog #287 docstring-tag discipline). Future empirical anchors
(post first paired CUDA+CPU dispatch with per-frame breakdown) refit the
choice via :mod:`tac.canonical_equations` auto-recalibration.

Pair → frame mapping (canonical comma_video_compression_challenge):

    frame_0 → pair_0 only (warp source)
    frame_t → pair_(t-1) target + pair_t source (for 0 < t < 1199)
    frame_1199 → pair_1198 target only

Cross-references:
  * CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 Step 4.1 line 404
  * Catalog #305 observability surface (every aggregator choice surfaced)
  * Catalog #287 docstring-tag-vs-evidence discipline ([predicted] axis)
  * Catalog #323 canonical Provenance umbrella
  * Catalog #354 master-gradient exploit consumer bundle (sister consumer
    `per_pair_difficulty_atlas_consumer` for the per-PAIR surface; this
    module is the per-FRAME surface)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from tac.provenance.contract import Provenance


PER_FRAME_DIFFICULTY_AGGREGATOR_VALID: frozenset[str] = frozenset({
    "mean_over_incident_pairs",
    "max_over_incident_pairs",
    "sum_over_incident_pairs",
})
"""Canonical aggregator operators.

  * ``mean_over_incident_pairs`` — average per-pair difficulty across the
    1-or-2 pairs each frame participates in. Balanced; default.
  * ``max_over_incident_pairs`` — worst-case per-pair difficulty. Surface
    for bit-allocator "hardest-frame" prioritization.
  * ``sum_over_incident_pairs`` — additive contribution. Useful for
    Lagrangian-dual variable bookkeeping when total per-frame leverage
    matters (per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable).
"""


_FRAME_INDEX_RE_INPUT_RANGE = (0, 1200)


def _validate_aggregator(aggregator: str) -> str:
    if not isinstance(aggregator, str):
        raise TypeError(
            f"aggregator must be a string; got {type(aggregator).__name__}"
        )
    if aggregator not in PER_FRAME_DIFFICULTY_AGGREGATOR_VALID:
        raise ValueError(
            f"aggregator={aggregator!r} must be one of "
            f"{sorted(PER_FRAME_DIFFICULTY_AGGREGATOR_VALID)!r}"
        )
    return aggregator


@dataclass(frozen=True)
class PerFrameDifficultyEntry:
    """One frame's aggregated difficulty signal.

    Fields:
        frame_index: 0-indexed frame in [0, total_frames).
        difficulty: aggregated per-pair difficulty score for this frame.
        incident_pair_indices: pair indices the frame participates in
            (typically length 1 for boundary frames, length 2 for interior
            frames).
        difficulty_rank: 0-indexed rank within the video (0 = hardest).
    """

    frame_index: int
    difficulty: float
    incident_pair_indices: tuple[int, ...]
    difficulty_rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.frame_index, int):
            raise TypeError(
                f"frame_index must be int, got {type(self.frame_index).__name__}"
            )
        if self.frame_index < 0:
            raise ValueError(f"frame_index={self.frame_index} must be >= 0")
        if not isinstance(self.difficulty, (int, float)):
            raise TypeError(
                f"difficulty must be numeric, got {type(self.difficulty).__name__}"
            )
        if self.difficulty != self.difficulty:  # NaN
            raise ValueError("difficulty must not be NaN")
        if self.difficulty < 0:
            raise ValueError(
                f"difficulty={self.difficulty} must be >= 0 (gradient norms are non-negative)"
            )
        if not isinstance(self.incident_pair_indices, tuple):
            raise TypeError(
                f"incident_pair_indices must be tuple, got {type(self.incident_pair_indices).__name__}"
            )
        if not self.incident_pair_indices:
            raise ValueError(
                "incident_pair_indices must be non-empty (every frame participates in >= 1 pair)"
            )
        for i, idx in enumerate(self.incident_pair_indices):
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(
                    f"incident_pair_indices[{i}]={idx!r} must be non-negative int"
                )
        if not isinstance(self.difficulty_rank, int) or self.difficulty_rank < 0:
            raise ValueError(
                f"difficulty_rank={self.difficulty_rank!r} must be non-negative int"
            )


@dataclass(frozen=True)
class PerFrameDifficultyAtlas:
    """Per-frame difficulty atlas — canonical wrapper output.

    Per Catalog #305 observability surface: every field below is queryable
    post-hoc + cite-able to the source pair-difficulty atlas + diff-able
    across runs (per-frame ordering deterministic given identical input).

    Fields:
        total_frames: number of frames in the source video (typically 1200).
        total_pairs: number of source pairs (typically 600 for comma-vcc).
        entries: per-frame entries sorted by frame_index ascending.
        top_k_hardest_frame_indices: descending by difficulty.
        bottom_k_easiest_frame_indices: ascending by difficulty.
        aggregator: aggregation operator used (see
            :data:`PER_FRAME_DIFFICULTY_AGGREGATOR_VALID`).
        source_pair_atlas_archive_sha256: archive sha the per-pair atlas
            was computed against; cite-chain back to per-pair source.
        source_measurement_axis: e.g. ``"[predicted]"`` /
            ``"[contest-CPU]"`` / ``"[contest-CUDA]"``. Always inherited
            from the per-pair source (apples-to-apples per CLAUDE.md
            "Apples-to-apples evidence discipline").
        provenance: canonical Provenance per Catalog #323.
    """

    total_frames: int
    total_pairs: int
    entries: tuple[PerFrameDifficultyEntry, ...]
    top_k_hardest_frame_indices: tuple[int, ...]
    bottom_k_easiest_frame_indices: tuple[int, ...]
    aggregator: str
    source_pair_atlas_archive_sha256: str
    source_measurement_axis: str
    provenance: Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.total_frames, int) or self.total_frames <= 0:
            raise ValueError(
                f"total_frames={self.total_frames!r} must be a positive int"
            )
        if not isinstance(self.total_pairs, int) or self.total_pairs <= 0:
            raise ValueError(
                f"total_pairs={self.total_pairs!r} must be a positive int"
            )
        if self.total_frames != self.total_pairs + 1 and self.total_frames != self.total_pairs * 2:
            # The canonical mapping is total_frames = total_pairs + 1
            # (adjacent-pair construction). The * 2 form is the non-
            # overlapping construction the contest scorer uses. Both are
            # valid; warn-via-error only if neither matches.
            raise ValueError(
                f"total_frames={self.total_frames} must satisfy "
                f"== total_pairs + 1 ({self.total_pairs + 1}) (adjacent) "
                f"OR == total_pairs * 2 ({self.total_pairs * 2}) (non-overlap); "
                f"got total_pairs={self.total_pairs}"
            )
        if not isinstance(self.entries, tuple):
            raise TypeError(
                f"entries must be tuple, got {type(self.entries).__name__}"
            )
        if len(self.entries) != self.total_frames:
            raise ValueError(
                f"entries length {len(self.entries)} != total_frames {self.total_frames}"
            )
        for i, entry in enumerate(self.entries):
            if not isinstance(entry, PerFrameDifficultyEntry):
                raise TypeError(
                    f"entries[{i}] must be PerFrameDifficultyEntry, got {type(entry).__name__}"
                )
            if entry.frame_index != i:
                raise ValueError(
                    f"entries[{i}].frame_index={entry.frame_index} must equal i={i}"
                )
        _validate_aggregator(self.aggregator)
        if not isinstance(self.source_pair_atlas_archive_sha256, str):
            raise TypeError(
                "source_pair_atlas_archive_sha256 must be a string"
            )
        if len(self.source_pair_atlas_archive_sha256) != 64 or not all(
            c in "0123456789abcdef" for c in self.source_pair_atlas_archive_sha256
        ):
            raise ValueError(
                f"source_pair_atlas_archive_sha256={self.source_pair_atlas_archive_sha256!r} "
                "must be a 64-char hex sha256"
            )
        if not isinstance(self.source_measurement_axis, str) or not self.source_measurement_axis:
            raise ValueError("source_measurement_axis must be non-empty string")
        if not isinstance(self.top_k_hardest_frame_indices, tuple):
            raise TypeError("top_k_hardest_frame_indices must be tuple")
        if not isinstance(self.bottom_k_easiest_frame_indices, tuple):
            raise TypeError("bottom_k_easiest_frame_indices must be tuple")
        if not isinstance(self.provenance, Provenance):
            raise TypeError(
                f"provenance must be Provenance, got {type(self.provenance).__name__}"
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe serialization for canonical-equation anchors etc."""
        from tac.provenance.validator import provenance_to_dict

        return {
            "schema": "per_frame_difficulty_atlas_v1",
            "total_frames": self.total_frames,
            "total_pairs": self.total_pairs,
            "aggregator": self.aggregator,
            "source_pair_atlas_archive_sha256": self.source_pair_atlas_archive_sha256,
            "source_measurement_axis": self.source_measurement_axis,
            "entries": [
                {
                    "frame_index": e.frame_index,
                    "difficulty": e.difficulty,
                    "incident_pair_indices": list(e.incident_pair_indices),
                    "difficulty_rank": e.difficulty_rank,
                }
                for e in self.entries
            ],
            "top_k_hardest_frame_indices": list(self.top_k_hardest_frame_indices),
            "bottom_k_easiest_frame_indices": list(self.bottom_k_easiest_frame_indices),
            "provenance": provenance_to_dict(self.provenance),
        }


def _build_canonical_adjacent_pair_to_frame_map(total_pairs: int) -> dict[int, tuple[int, ...]]:
    """Map frame_index → tuple of incident pair indices.

    Canonical adjacent-pair construction: frame_t participates in
    pair_(t-1) as target and pair_t as source (interior frames).

    Args:
        total_pairs: number of pairs (typically 600 for comma-vcc).

    Returns:
        Mapping {frame_index: (pair_indices...)}; total_frames = total_pairs + 1.
    """
    total_frames = total_pairs + 1
    mapping: dict[int, tuple[int, ...]] = {}
    for frame_index in range(total_frames):
        incident: list[int] = []
        # Previous pair has this frame as target.
        if frame_index > 0:
            incident.append(frame_index - 1)
        # Next pair has this frame as source.
        if frame_index < total_pairs:
            incident.append(frame_index)
        mapping[frame_index] = tuple(incident)
    return mapping


def _build_canonical_non_overlapping_pair_to_frame_map(total_pairs: int) -> dict[int, tuple[int, ...]]:
    """Map frame_index → incident pair indices for the contest non-overlap layout.

    The contest scorer's non-overlapping batching (per CLAUDE.md
    "1199 OVERLAPPING PAIRS vs 600 NON-OVERLAPPING" anchor) treats
    frame_(2k) and frame_(2k+1) as the source+target of pair_k.

    Args:
        total_pairs: number of pairs (typically 600 for comma-vcc).

    Returns:
        Mapping {frame_index: (pair_index,)}; total_frames = total_pairs * 2.
    """
    mapping: dict[int, tuple[int, ...]] = {}
    for pair_index in range(total_pairs):
        # frame_(2k) and frame_(2k+1) are both incident on pair_k.
        mapping[2 * pair_index] = (pair_index,)
        mapping[2 * pair_index + 1] = (pair_index,)
    return mapping


def build_per_frame_difficulty_from_per_pair_atlas(
    per_pair_difficulty: Sequence[float],
    *,
    archive_sha256: str,
    measurement_axis: str,
    provenance: Provenance,
    aggregator: str = "mean_over_incident_pairs",
    pair_to_frame_construction: str = "adjacent",
    top_k: int = 50,
    bottom_k: int = 50,
) -> PerFrameDifficultyAtlas:
    """Build :class:`PerFrameDifficultyAtlas` from per-pair difficulty vector.

    Wraps the per-pair output of ``tac.master_gradient_consumers.
    per_pair_difficulty_atlas`` and projects to per-frame via one of the
    3 canonical aggregators.

    Args:
        per_pair_difficulty: per-pair difficulty score vector (length =
            total_pairs). Typically ``np.linalg.norm(per_pair_gradient, axis=(0,2))``
            from the canonical per-pair atlas.
        archive_sha256: archive sha the per-pair atlas was computed against
            (cite-chain back to source).
        measurement_axis: axis tag of the underlying per-pair source
            (``"[predicted]"`` / ``"[contest-CPU]"`` / ``"[contest-CUDA]"``).
        provenance: canonical Provenance per Catalog #323.
        aggregator: one of :data:`PER_FRAME_DIFFICULTY_AGGREGATOR_VALID`.
            Default ``"mean_over_incident_pairs"``.
        pair_to_frame_construction: ``"adjacent"`` (total_frames = pairs+1)
            or ``"non_overlap"`` (total_frames = pairs*2; matches contest
            scorer's non-overlapping batching).
        top_k: number of hardest frames to surface.
        bottom_k: number of easiest frames to surface.

    Returns:
        Frozen :class:`PerFrameDifficultyAtlas` ready for cathedral consumer
        ingestion + canonical-equation anchor seeding.

    Raises:
        TypeError / ValueError on contract violations.
    """
    if not isinstance(per_pair_difficulty, Sequence) or isinstance(
        per_pair_difficulty, (str, bytes, bytearray)
    ):
        raise TypeError(
            f"per_pair_difficulty must be a Sequence of floats; got {type(per_pair_difficulty).__name__}"
        )
    total_pairs = len(per_pair_difficulty)
    if total_pairs <= 0:
        raise ValueError(
            "per_pair_difficulty must be non-empty (>=1 pair required)"
        )
    for i, v in enumerate(per_pair_difficulty):
        if not isinstance(v, (int, float)) or v != v:
            raise ValueError(
                f"per_pair_difficulty[{i}]={v!r} must be a finite numeric"
            )
        if v < 0:
            raise ValueError(
                f"per_pair_difficulty[{i}]={v} must be non-negative (norms)"
            )
    aggregator = _validate_aggregator(aggregator)
    if pair_to_frame_construction == "adjacent":
        mapping = _build_canonical_adjacent_pair_to_frame_map(total_pairs)
    elif pair_to_frame_construction == "non_overlap":
        mapping = _build_canonical_non_overlapping_pair_to_frame_map(total_pairs)
    else:
        raise ValueError(
            f"pair_to_frame_construction={pair_to_frame_construction!r} "
            "must be 'adjacent' or 'non_overlap'"
        )
    total_frames = len(mapping)
    # Aggregate per-pair into per-frame.
    per_frame_difficulty: list[float] = []
    per_frame_incident: list[tuple[int, ...]] = []
    for frame_index in range(total_frames):
        incident = mapping[frame_index]
        pair_values = [float(per_pair_difficulty[p]) for p in incident]
        if aggregator == "mean_over_incident_pairs":
            agg = sum(pair_values) / len(pair_values)
        elif aggregator == "max_over_incident_pairs":
            agg = max(pair_values)
        elif aggregator == "sum_over_incident_pairs":
            agg = sum(pair_values)
        else:  # pragma: no cover - guarded above
            raise AssertionError(f"unreachable aggregator={aggregator!r}")
        per_frame_difficulty.append(agg)
        per_frame_incident.append(incident)
    # Rank frames by difficulty descending.
    ranked_indices = sorted(
        range(total_frames),
        key=lambda i: (-per_frame_difficulty[i], i),
    )
    rank_lookup: dict[int, int] = {idx: rank for rank, idx in enumerate(ranked_indices)}
    entries = tuple(
        PerFrameDifficultyEntry(
            frame_index=i,
            difficulty=per_frame_difficulty[i],
            incident_pair_indices=per_frame_incident[i],
            difficulty_rank=rank_lookup[i],
        )
        for i in range(total_frames)
    )
    top_k_clamped = max(0, min(int(top_k), total_frames))
    bottom_k_clamped = max(0, min(int(bottom_k), total_frames))
    top_k_indices = tuple(ranked_indices[:top_k_clamped])
    bottom_k_indices = tuple(ranked_indices[-bottom_k_clamped:][::-1]) if bottom_k_clamped > 0 else ()
    return PerFrameDifficultyAtlas(
        total_frames=total_frames,
        total_pairs=total_pairs,
        entries=entries,
        top_k_hardest_frame_indices=top_k_indices,
        bottom_k_easiest_frame_indices=bottom_k_indices,
        aggregator=aggregator,
        source_pair_atlas_archive_sha256=archive_sha256,
        source_measurement_axis=measurement_axis,
        provenance=provenance,
    )


__all__ = [
    "PER_FRAME_DIFFICULTY_AGGREGATOR_VALID",
    "PerFrameDifficultyAtlas",
    "PerFrameDifficultyEntry",
    "build_per_frame_difficulty_from_per_pair_atlas",
]
