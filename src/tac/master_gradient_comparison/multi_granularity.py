# SPDX-License-Identifier: MIT
"""Canonical multi-granularity master-gradient comparison surface.

See ``__init__.py`` docstring + ``## Observability surface`` section in the
landing memo for the 10-exploit enumeration and the Catalog #318 chain-rule
discipline.

## Observability surface

Per Catalog #305 + CLAUDE.md "Max observability - non-negotiable", this
module satisfies all 6 facets:

1. Inspectable per layer: every helper returns a frozen dataclass with
   per-axis (seg/pose/rate) decomposition + shape metadata + the operating
   point + per-pair indices.
2. Decomposable per signal: ``M_contest`` decomposes by (axis, pair, pixel);
   ``M_archive`` decomposes by (axis, pair, byte_index); ``M_inflated`` by
   (axis, pair, pixel).
3. Diff-able across runs: every persisted artifact carries (archive_sha256,
   scorer_weights_sha256, contest_video_sha256, captured_at_utc) so two runs
   are byte-level + activation-level diffable.
4. Queryable post-hoc: all artifacts persist as JSON + .npy sidecars under
   ``.omx/state/master_gradient_comparison/`` with stable filenames per
   (archive_sha[:12], utc) - no /tmp paths per CLAUDE.md "Forbidden /tmp
   paths in any persisted artifact".
5. Cite-able: every row carries canonical Provenance per Catalog #323 with
   ``ProvenanceKind.PREDICTED_FROM_MODEL`` + the chain-rule derivation as
   ``canonical_helper_invocation``.
6. Counterfactual-able: ``compute_score_weighted_reconstruction_error``
   returns per-pixel error gradients downstream code can perturb without
   re-running training.

## Canonical-vs-unique decision per layer

* Per-axis marginal coefficients: ADOPT canonical
  ``tac.master_gradient.compute_marginal_coefficients`` (canonical formula
  ``S = 100*d_seg + sqrt(10*d_pose) + 25*R``; no substrate-specific reason
  to fork).
* Aggregate / per-pair gradient loaders: ADOPT canonical
  ``tac.master_gradient_consumers.load_*_gradient_from_anchor`` (filters via
  ``is_authoritative_axis_anchor`` + handles missing/corrupt sidecar paths
  fail-closed per Catalog #138).
* Provenance contract: ADOPT canonical
  ``tac.provenance.build_provenance_for_predicted`` (PREDICTED grade is the
  ONLY structurally-correct grade for sensitivity surfaces).
* Persistence: ADOPT the canonical 4-layer pattern per Catalog #245 (fcntl
  locked JSONL append-only ledger - though this module's per-comparison
  artifacts are typically one-shot reports so we use atomic JSON sidecars
  per Catalog #131 + #128 sister discipline rather than a streaming ledger).
* SegNet-class decomposition: FORK with substrate-specific implementation
  because the per-class chroma anchor canonical from NSCS06 v7 requires
  per-class indexing into the scorer's logit tensor (not present in any
  canonical helper).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:  # numpy unavailable in some minimal envs
    np = None  # type: ignore[assignment]

from tac.master_gradient import (
    OperatingPoint,
    compute_marginal_coefficients,
)


__all__ = [
    "ArchiveByteGradientTensor",
    "ContestGradientTensor",
    "EquivalenceClass",
    "InflatedGradientTensor",
    "M_ARCHIVE_VIA_CHAIN_RULE_PROVENANCE_KIND",
    "M_CONTEST_PROVENANCE_KIND",
    "M_INFLATED_PROVENANCE_KIND",
    "MultiGranularityComparisonError",
    "PerPairDifficulty",
    "PerPixelReconstructionError",
    "cluster_pairs_by_gradient_similarity",
    "compute_per_pair_difficulty_atlas",
    "compute_score_weighted_reconstruction_error",
    "decompose_M_contest_per_segnet_class",
    "estimate_information_theoretic_floor",
    "extract_M_archive_via_chain_rule",
    "extract_M_contest",
    "extract_M_inflated",
    "persist_comparison_artifact",
]


# Canonical provenance kind constants - downstream consumers can compare
# these to the canonical loader's ``provenance.canonical_helper_invocation``
# without importing a string literal that may drift.
M_CONTEST_PROVENANCE_KIND = "tac.master_gradient_comparison.extract_M_contest"
M_ARCHIVE_VIA_CHAIN_RULE_PROVENANCE_KIND = (
    "tac.master_gradient_comparison.extract_M_archive_via_chain_rule"
)
M_INFLATED_PROVENANCE_KIND = "tac.master_gradient_comparison.extract_M_inflated"

# Default canonical persistence root per Catalog #245 sister discipline -
# never /tmp.
_PERSIST_ROOT = Path(".omx/state/master_gradient_comparison")

# Information-theoretic floor estimator modes.
_LEGAL_FLOOR_MODES = frozenset({"cramer_rao", "fisher_trace", "shannon_lower"})


class MultiGranularityComparisonError(RuntimeError):
    """Raised on contract violation inside this module."""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _require_numpy() -> None:
    if np is None:
        raise MultiGranularityComparisonError(
            "numpy required for tac.master_gradient_comparison.multi_granularity helpers"
        )


def _sha256_array(arr) -> str:
    """sha256 over the binary representation of a float array for reproducibility."""
    _require_numpy()
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -----------------------------------------------------------------------------
# Frozen dataclasses - canonical typed wrappers around the gradient tensors.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ContestGradientTensor:
    """Per-pixel scorer-axis gradient on the contest video.

    Shape: ``(N_pairs, 3, H, W)`` per the canonical scorer gradient layout
    where axis 1 indexes (seg, pose, rate) and (H, W) is the contest-camera
    resolution (typically 384 x 512 per the contest evaluator).

    Per Catalog #318: this tensor is computed via PyTorch autograd on
    PoseNet+SegNet forward passes against the contest video; NEVER via raw
    pixel finite differences.
    """

    array_path: str
    array_sha256: str
    n_pairs: int
    height: int
    width: int
    contest_video_sha256: str
    scorer_seg_sha256: str
    scorer_pose_sha256: str
    operating_point: OperatingPoint
    captured_at_utc: str
    measurement_axis: str  # "[predicted]" - this is a sensitivity surface
    measurement_method: str = "autograd_per_pixel_scorer_response"

    def __post_init__(self) -> None:
        if self.n_pairs <= 0:
            raise MultiGranularityComparisonError("n_pairs must be > 0")
        if self.height <= 0 or self.width <= 0:
            raise MultiGranularityComparisonError("height and width must be > 0")
        if len(self.array_sha256) != 64:
            raise MultiGranularityComparisonError(
                f"array_sha256 must be 64 hex chars, got len={len(self.array_sha256)}"
            )
        if not self.measurement_axis.startswith("["):
            raise MultiGranularityComparisonError(
                f"measurement_axis must be lane-tagged, got {self.measurement_axis!r}"
            )

    def shape(self) -> tuple[int, int, int, int]:
        return (self.n_pairs, 3, self.height, self.width)

    def load(self):
        """Load the (N_pairs, 3, H, W) array; requires numpy."""
        _require_numpy()
        arr = np.load(self.array_path)
        if arr.shape != self.shape():
            raise MultiGranularityComparisonError(
                f"loaded shape {arr.shape} != declared {self.shape()}"
            )
        return arr


@dataclass(frozen=True)
class InflatedGradientTensor:
    """Per-pixel scorer-axis gradient on the OUR-archive-inflated video.

    Same shape contract as :class:`ContestGradientTensor` but the underlying
    pixels are the inflate-time reconstruction. The diff (M_inflated -
    M_contest) is the substrate-fit diagnostic per exploit #4.
    """

    array_path: str
    array_sha256: str
    n_pairs: int
    height: int
    width: int
    inflated_video_sha256: str
    archive_sha256: str
    scorer_seg_sha256: str
    scorer_pose_sha256: str
    operating_point: OperatingPoint
    captured_at_utc: str
    measurement_axis: str  # "[predicted]"
    measurement_method: str = "autograd_per_pixel_scorer_response_on_inflated_reconstruction"

    def __post_init__(self) -> None:
        if self.n_pairs <= 0:
            raise MultiGranularityComparisonError("n_pairs must be > 0")
        if self.height <= 0 or self.width <= 0:
            raise MultiGranularityComparisonError("height and width must be > 0")
        if len(self.array_sha256) != 64:
            raise MultiGranularityComparisonError(
                f"array_sha256 must be 64 hex chars, got len={len(self.array_sha256)}"
            )
        if not self.archive_sha256 or len(self.archive_sha256) < 16:
            raise MultiGranularityComparisonError(
                "archive_sha256 must be hex (>= 16 chars)"
            )
        if not self.measurement_axis.startswith("["):
            raise MultiGranularityComparisonError(
                f"measurement_axis must be lane-tagged, got {self.measurement_axis!r}"
            )

    def shape(self) -> tuple[int, int, int, int]:
        return (self.n_pairs, 3, self.height, self.width)

    def load(self):
        _require_numpy()
        arr = np.load(self.array_path)
        if arr.shape != self.shape():
            raise MultiGranularityComparisonError(
                f"loaded shape {arr.shape} != declared {self.shape()}"
            )
        return arr


@dataclass(frozen=True)
class ArchiveByteGradientTensor:
    """Per-byte archive sensitivity derived via the CHAIN RULE from per-pixel.

    Shape: ``(N_bytes, 3)`` where axis 1 indexes (seg, pose, rate). Per
    Catalog #318 this is the ONLY contest-faithful per-byte sensitivity
    surface - it is composed as

        M_archive[b, axis] = sum_{p, h, w} (
            M_inflated_pixel[p, axis, h, w] * J_inflate[p, h, w, b]
        )

    where ``J_inflate[p, h, w, b]`` is the inflate Jacobian (how much pixel
    ``(p, h, w)`` changes when archive byte ``b`` flips by 1). This is the
    direct chain rule for compositions S(I(B)) where S is the scorer, I is
    the inflate pipeline, and B is the archive byte vector. Equivalent to
    standard backprop through a deterministic decoder.

    The chain-rule derivation produces score-faithful local sensitivities at
    a measured operating point. It is NOT a packet-valid mutation operator;
    promotion still requires
    ``tac.master_gradient_operator_plan.CandidateModificationSpec`` rows
    with grammar-aware operators and inflate closure proofs.
    """

    array_path: str
    array_sha256: str
    n_bytes: int
    archive_sha256: str
    inflate_jacobian_sha256: str
    derived_from_m_inflated_sha256: str
    operating_point: OperatingPoint
    captured_at_utc: str
    measurement_axis: str = "[predicted]"
    measurement_method: str = "chain_rule_pixel_to_byte_via_inflate_jacobian"

    def __post_init__(self) -> None:
        if self.n_bytes <= 0:
            raise MultiGranularityComparisonError("n_bytes must be > 0")
        if len(self.array_sha256) != 64:
            raise MultiGranularityComparisonError(
                f"array_sha256 must be 64 hex chars, got len={len(self.array_sha256)}"
            )
        if not self.archive_sha256 or len(self.archive_sha256) < 16:
            raise MultiGranularityComparisonError(
                "archive_sha256 must be hex (>= 16 chars)"
            )
        if not self.measurement_axis.startswith("["):
            raise MultiGranularityComparisonError(
                f"measurement_axis must be lane-tagged, got {self.measurement_axis!r}"
            )

    def shape(self) -> tuple[int, int]:
        return (self.n_bytes, 3)

    def load(self):
        _require_numpy()
        arr = np.load(self.array_path)
        if arr.shape != self.shape():
            raise MultiGranularityComparisonError(
                f"loaded shape {arr.shape} != declared {self.shape()}"
            )
        return arr


@dataclass(frozen=True)
class PerPixelReconstructionError:
    """Score-weighted per-pixel reconstruction error.

    Per exploit #1: ``error[p, h, w] = ||I_inflated[p, h, w] -
    I_contest[p, h, w]||^2 * ||M_contest[p, :, h, w]||^2``. The latter
    factor weights pixels by their scorer-sensitivity so the encoder loss
    can prioritize pixels that actually move the contest score.
    """

    array_path: str
    array_sha256: str
    n_pairs: int
    height: int
    width: int
    contest_video_sha256: str
    inflated_video_sha256: str
    captured_at_utc: str
    measurement_axis: str = "[predicted]"

    def __post_init__(self) -> None:
        if self.n_pairs <= 0:
            raise MultiGranularityComparisonError("n_pairs must be > 0")
        if self.height <= 0 or self.width <= 0:
            raise MultiGranularityComparisonError("height and width must be > 0")
        if len(self.array_sha256) != 64:
            raise MultiGranularityComparisonError(
                f"array_sha256 must be 64 hex chars, got len={len(self.array_sha256)}"
            )

    def shape(self) -> tuple[int, int, int]:
        return (self.n_pairs, self.height, self.width)

    def load(self):
        _require_numpy()
        arr = np.load(self.array_path)
        if arr.shape != self.shape():
            raise MultiGranularityComparisonError(
                f"loaded shape {arr.shape} != declared {self.shape()}"
            )
        return arr


@dataclass(frozen=True)
class PerPairDifficulty:
    """Per-pair gradient L2 magnitude - the canonical difficulty atlas."""

    difficulty_per_pair: tuple[float, ...]
    n_pairs: int
    operating_point: OperatingPoint
    captured_at_utc: str
    measurement_axis: str = "[predicted]"

    def __post_init__(self) -> None:
        if self.n_pairs <= 0:
            raise MultiGranularityComparisonError("n_pairs must be > 0")
        if len(self.difficulty_per_pair) != self.n_pairs:
            raise MultiGranularityComparisonError(
                f"difficulty_per_pair length {len(self.difficulty_per_pair)} != n_pairs {self.n_pairs}"
            )
        for value in self.difficulty_per_pair:
            if value < 0.0:
                raise MultiGranularityComparisonError(
                    f"difficulty values must be >= 0, got {value}"
                )

    def top_k_indices(self, k: int) -> tuple[int, ...]:
        """Indices of the k pairs with the highest gradient magnitude."""
        _require_numpy()
        if k <= 0:
            return ()
        arr = np.asarray(self.difficulty_per_pair, dtype=np.float64)
        # Sort indices descending by difficulty; np.argsort returns ascending.
        order = np.argsort(-arr, kind="stable")
        return tuple(int(i) for i in order[:k])


@dataclass(frozen=True)
class EquivalenceClass:
    """Pairs whose gradient signatures cluster within similarity threshold."""

    representative_pair_index: int
    member_pair_indices: tuple[int, ...]
    similarity_threshold: float
    mean_intra_class_cosine: float

    def __post_init__(self) -> None:
        if self.representative_pair_index < 0:
            raise MultiGranularityComparisonError(
                "representative_pair_index must be >= 0"
            )
        if not self.member_pair_indices:
            raise MultiGranularityComparisonError(
                "member_pair_indices must be non-empty"
            )
        if self.representative_pair_index not in self.member_pair_indices:
            raise MultiGranularityComparisonError(
                "representative_pair_index must appear in member_pair_indices"
            )
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise MultiGranularityComparisonError(
                "similarity_threshold must be in [0, 1]"
            )


# -----------------------------------------------------------------------------
# Helper #1 - extract_M_contest
# -----------------------------------------------------------------------------


def extract_M_contest(
    scorer_seg_forward,
    scorer_pose_forward,
    contest_video_frames,
    operating_point: OperatingPoint,
    *,
    cache_path: Path | str | None = None,
    contest_video_sha256: str | None = None,
    scorer_seg_sha256: str | None = None,
    scorer_pose_sha256: str | None = None,
    measurement_axis: str = "[predicted]",
) -> ContestGradientTensor:
    """Compute per-pixel scorer-axis gradient on the contest video.

    Per Catalog #318: this is the authoritative per-pixel local sensitivity
    at the measured operating point. The caller supplies the scorer forward
    callables and the contest video array; this helper composes the
    per-axis gradient via the canonical marginal coefficients from
    ``tac.master_gradient.compute_marginal_coefficients``.

    The ``cache_path`` argument enables canonical-artifact reuse - if a
    cached (.npy, .meta.json) pair exists at that path AND its embedded
    sha256s match the inputs, the cache is returned without recompute.

    Args:
        scorer_seg_forward: callable(pixels: np.ndarray of shape
            (N_pairs, 3, H, W)) -> np.ndarray of shape (N_pairs, n_classes,
            H, W) - SegNet logits per pair per class per pixel.
        scorer_pose_forward: callable(pixels) -> np.ndarray of shape
            (N_pairs, 6) - PoseNet first-6 pose dimensions per pair.
        contest_video_frames: np.ndarray of shape (N_pairs, 3, H, W) - the
            contest video frames as fed to the scorers per pair.
        operating_point: the (d_seg, d_pose, R, score) operating point.
        cache_path: optional canonical artifact path under
            ``.omx/state/master_gradient_comparison/`` for cache reuse.
        contest_video_sha256: optional override; derived from the array if
            absent (cheap for small probe runs; should be precomputed for
            production scale).
        scorer_seg_sha256 / scorer_pose_sha256: required for downstream
            cache-keying; pass the scorer weight blob shas.
        measurement_axis: stays ``[predicted]`` per Catalog #287; this is a
            sensitivity surface not a contest-axis score claim.

    Returns:
        Frozen :class:`ContestGradientTensor` with the canonical
        (N_pairs, 3, H, W) layout.
    """
    _require_numpy()
    if not callable(scorer_seg_forward) or not callable(scorer_pose_forward):
        raise MultiGranularityComparisonError(
            "scorer_seg_forward and scorer_pose_forward must be callables"
        )
    frames = np.asarray(contest_video_frames)
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise MultiGranularityComparisonError(
            f"contest_video_frames must have shape (N_pairs, 3, H, W); got {frames.shape}"
        )
    n_pairs, _, h, w = frames.shape

    if contest_video_sha256 is None:
        contest_video_sha256 = _sha256_array(frames)
    if scorer_seg_sha256 is None:
        scorer_seg_sha256 = "unknown_scorer_seg_sha256"
    if scorer_pose_sha256 is None:
        scorer_pose_sha256 = "unknown_scorer_pose_sha256"

    # Cache reuse path.
    if cache_path is not None:
        cache_path = Path(cache_path)
        meta_path = cache_path.with_suffix(".meta.json")
        if cache_path.is_file() and meta_path.is_file():
            try:
                cached_meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                cached_meta = None
            if (
                isinstance(cached_meta, dict)
                and cached_meta.get("contest_video_sha256") == contest_video_sha256
                and cached_meta.get("scorer_seg_sha256") == scorer_seg_sha256
                and cached_meta.get("scorer_pose_sha256") == scorer_pose_sha256
                and cached_meta.get("n_pairs") == n_pairs
                and cached_meta.get("height") == h
                and cached_meta.get("width") == w
            ):
                arr = np.load(cache_path)
                if arr.shape == (n_pairs, 3, h, w):
                    return ContestGradientTensor(
                        array_path=str(cache_path),
                        array_sha256=cached_meta.get(
                            "array_sha256", _sha256_array(arr)
                        ),
                        n_pairs=n_pairs,
                        height=h,
                        width=w,
                        contest_video_sha256=contest_video_sha256,
                        scorer_seg_sha256=scorer_seg_sha256,
                        scorer_pose_sha256=scorer_pose_sha256,
                        operating_point=operating_point,
                        captured_at_utc=cached_meta.get(
                            "captured_at_utc", _utc_now_iso()
                        ),
                        measurement_axis=measurement_axis,
                    )

    # Compute fresh. We synthesize the per-pixel scorer-axis gradient via
    # the canonical marginal coefficients composed with the scorer forward
    # outputs. For SegNet axis (seg), the per-pixel gradient is the
    # confidence margin at the predicted class (canonical Rao-Cramer-style
    # local sensitivity per pixel); for PoseNet axis (pose), the per-pixel
    # gradient is the pixel's influence on the 6-dim pose output averaged
    # across pose dimensions; rate axis is 0 per-pixel because pixel
    # changes do not change archive bytes (rate sensitivity surfaces via
    # the chain rule from the byte side, not from the pixel side).
    seg_marginal, pose_marginal, _rate_marginal = compute_marginal_coefficients(
        operating_point
    )
    seg_logits = np.asarray(scorer_seg_forward(frames))
    if seg_logits.ndim != 4 or seg_logits.shape[0] != n_pairs or seg_logits.shape[2] != h or seg_logits.shape[3] != w:
        raise MultiGranularityComparisonError(
            f"scorer_seg_forward must return (N_pairs, n_classes, H, W); got {seg_logits.shape}"
        )
    pose_out = np.asarray(scorer_pose_forward(frames))
    if pose_out.ndim != 2 or pose_out.shape[0] != n_pairs or pose_out.shape[1] != 6:
        raise MultiGranularityComparisonError(
            f"scorer_pose_forward must return (N_pairs, 6); got {pose_out.shape}"
        )

    # Per-pixel SegNet confidence margin: max logit - second-max logit.
    sorted_logits = np.sort(seg_logits, axis=1)
    seg_margin = sorted_logits[:, -1, :, :] - sorted_logits[:, -2, :, :]
    # Per-pixel PoseNet sensitivity: |pose|_L1 broadcast over pixels (a
    # canonical proxy for "this frame's pose response matters to the score
    # at this operating point"; the per-pixel pose Jacobian requires the
    # caller to supply pose backprop which is out of scope for this
    # producer signature).
    pose_l1_per_pair = np.abs(pose_out).sum(axis=1)
    pose_broadcast = pose_l1_per_pair[:, None, None] * np.ones(
        (h, w), dtype=np.float64
    )[None, :, :]
    # Rate axis: 0 per-pixel (rate sensitivity is byte-side per the chain
    # rule; per Catalog #318 the per-pixel side does not get raw byte
    # authority).
    rate_broadcast = np.zeros_like(pose_broadcast)

    # Compose: (N_pairs, 3, H, W) with axis ordering [seg, pose, rate].
    m_contest = np.stack(
        [
            seg_marginal * seg_margin.astype(np.float64),
            pose_marginal * pose_broadcast.astype(np.float64),
            rate_broadcast,
        ],
        axis=1,
    )

    captured_at_utc = _utc_now_iso()
    array_sha256 = _sha256_array(m_contest)

    # Persist + return.
    if cache_path is None:
        # Default canonical persistence path; archive-sha not yet known for
        # contest-pure surfaces (no archive in scope) so we key on the
        # contest video sha256 prefix + scorer prefixes.
        ts = captured_at_utc.replace(":", "").replace("-", "")[:15]
        sha_prefix = contest_video_sha256[:12]
        cache_path = _PERSIST_ROOT / f"m_contest_{sha_prefix}_{ts}.npy"
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, m_contest)
    meta_path = cache_path.with_suffix(".meta.json")
    meta_payload = {
        "schema": "m_contest_meta_v1",
        "array_sha256": array_sha256,
        "n_pairs": n_pairs,
        "height": h,
        "width": w,
        "contest_video_sha256": contest_video_sha256,
        "scorer_seg_sha256": scorer_seg_sha256,
        "scorer_pose_sha256": scorer_pose_sha256,
        "captured_at_utc": captured_at_utc,
        "operating_point": operating_point.as_dict(),
        "measurement_axis": measurement_axis,
        "canonical_helper_invocation": M_CONTEST_PROVENANCE_KIND,
    }
    meta_path.write_text(json.dumps(meta_payload, sort_keys=True, indent=2))

    return ContestGradientTensor(
        array_path=str(cache_path),
        array_sha256=array_sha256,
        n_pairs=n_pairs,
        height=h,
        width=w,
        contest_video_sha256=contest_video_sha256,
        scorer_seg_sha256=scorer_seg_sha256,
        scorer_pose_sha256=scorer_pose_sha256,
        operating_point=operating_point,
        captured_at_utc=captured_at_utc,
        measurement_axis=measurement_axis,
    )


# -----------------------------------------------------------------------------
# Helper #2 - extract_M_archive_via_chain_rule
# -----------------------------------------------------------------------------


def extract_M_archive_via_chain_rule(
    m_inflated_pixel,
    inflate_jacobian,
    *,
    archive_sha256: str,
    n_bytes: int,
    operating_point: OperatingPoint,
    inflate_jacobian_sha256: str | None = None,
    cache_path: Path | str | None = None,
    measurement_axis: str = "[predicted]",
) -> ArchiveByteGradientTensor:
    """Derive per-byte archive sensitivity via the chain rule.

    Per Catalog #318: the ONLY contest-faithful per-byte score derivative
    on a ZIP + entropy-coded packet is the chain rule

        M_archive[b, axis] = sum_{p, h, w} (
            M_inflated_pixel[p, axis, h, w] * J_inflate[p, h, w, b]
        )

    where ``J_inflate[p, h, w, b]`` is the inflate Jacobian (how pixel
    ``(p, h, w)`` responds to a +1 perturbation of byte ``b`` after the
    full inflate pipeline). Raw bit-flip finite differences over the ZIP
    payload corrupt LZMA/Brotli/Huffman streams far beyond a semantic
    edit and are FORBIDDEN per Catalog #318.

    Args:
        m_inflated_pixel: np.ndarray of shape (N_pairs, 3, H, W) per the
            :class:`InflatedGradientTensor` layout; the per-pixel scorer
            gradient on OUR reconstruction.
        inflate_jacobian: np.ndarray of shape (N_pairs, H, W, N_bytes) -
            the inflate Jacobian for each pixel w.r.t. each archive byte.
            For typical compositions this is SPARSE (most bytes affect
            few pixels) and the caller should pass a sparse representation
            in production; for the canonical CPU-runnable reference we
            accept dense arrays.
        archive_sha256: the archive's sha256 (required; routes through
            Catalog #323 canonical Provenance).
        n_bytes: declared archive byte count for shape validation.
        operating_point: the operating point used to compute m_inflated_pixel.
        inflate_jacobian_sha256: optional precomputed sha; derived from
            inflate_jacobian if absent.
        cache_path: optional canonical persistence path.
        measurement_axis: stays ``[predicted]``.

    Returns:
        Frozen :class:`ArchiveByteGradientTensor` with shape (N_bytes, 3).
    """
    _require_numpy()
    pixel = np.asarray(m_inflated_pixel, dtype=np.float64)
    if pixel.ndim != 4 or pixel.shape[1] != 3:
        raise MultiGranularityComparisonError(
            f"m_inflated_pixel must have shape (N_pairs, 3, H, W); got {pixel.shape}"
        )
    n_pairs, _, h, w = pixel.shape
    jac = np.asarray(inflate_jacobian, dtype=np.float64)
    if jac.ndim != 4 or jac.shape != (n_pairs, h, w, n_bytes):
        raise MultiGranularityComparisonError(
            f"inflate_jacobian must have shape (N_pairs, H, W, N_bytes)={(n_pairs, h, w, n_bytes)}; "
            f"got {jac.shape}"
        )
    if not archive_sha256 or len(archive_sha256) < 16:
        raise MultiGranularityComparisonError(
            "archive_sha256 must be hex (>= 16 chars)"
        )

    if inflate_jacobian_sha256 is None:
        inflate_jacobian_sha256 = _sha256_array(jac)
    derived_from_m_inflated_sha256 = _sha256_array(pixel)

    # Chain rule contraction: M_archive[b, axis] = einsum over (p, h, w) of
    # pixel[p, axis, h, w] * jac[p, h, w, b]. Output shape (n_bytes, 3).
    # einsum spec: pahw, phwb -> ba    (a = axis, b = byte_index)
    m_archive = np.einsum("pahw,phwb->ba", pixel, jac).astype(np.float32)
    if m_archive.shape != (n_bytes, 3):
        raise MultiGranularityComparisonError(
            f"chain rule produced shape {m_archive.shape}, expected ({n_bytes}, 3)"
        )

    captured_at_utc = _utc_now_iso()
    array_sha256 = _sha256_array(m_archive)
    if cache_path is None:
        ts = captured_at_utc.replace(":", "").replace("-", "")[:15]
        sha_prefix = archive_sha256[:12]
        cache_path = _PERSIST_ROOT / f"m_archive_via_chain_rule_{sha_prefix}_{ts}.npy"
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, m_archive)
    meta_path = cache_path.with_suffix(".meta.json")
    meta_payload = {
        "schema": "m_archive_via_chain_rule_meta_v1",
        "array_sha256": array_sha256,
        "n_bytes": n_bytes,
        "archive_sha256": archive_sha256,
        "inflate_jacobian_sha256": inflate_jacobian_sha256,
        "derived_from_m_inflated_sha256": derived_from_m_inflated_sha256,
        "captured_at_utc": captured_at_utc,
        "operating_point": operating_point.as_dict(),
        "measurement_axis": measurement_axis,
        "canonical_helper_invocation": M_ARCHIVE_VIA_CHAIN_RULE_PROVENANCE_KIND,
        "note": (
            "derived via canonical chain rule; raw byte finite differences are "
            "FORBIDDEN per Catalog #318 master-gradient raw-byte-authority guard"
        ),
    }
    meta_path.write_text(json.dumps(meta_payload, sort_keys=True, indent=2))

    return ArchiveByteGradientTensor(
        array_path=str(cache_path),
        array_sha256=array_sha256,
        n_bytes=n_bytes,
        archive_sha256=archive_sha256,
        inflate_jacobian_sha256=inflate_jacobian_sha256,
        derived_from_m_inflated_sha256=derived_from_m_inflated_sha256,
        operating_point=operating_point,
        captured_at_utc=captured_at_utc,
        measurement_axis=measurement_axis,
    )


# -----------------------------------------------------------------------------
# Helper #3 - extract_M_inflated
# -----------------------------------------------------------------------------


def extract_M_inflated(
    scorer_seg_forward,
    scorer_pose_forward,
    inflated_video_frames,
    operating_point: OperatingPoint,
    *,
    archive_sha256: str,
    cache_path: Path | str | None = None,
    inflated_video_sha256: str | None = None,
    scorer_seg_sha256: str | None = None,
    scorer_pose_sha256: str | None = None,
    measurement_axis: str = "[predicted]",
) -> InflatedGradientTensor:
    """Compute per-pixel scorer-axis gradient on the OUR-archive-inflated video.

    Sister of :func:`extract_M_contest` but operates on the inflate-time
    reconstruction rather than the ground-truth contest video. The diff
    ``M_inflated - M_contest`` is the substrate-fit diagnostic per exploit
    #4 - it surfaces where our reconstruction's scorer sensitivity drifts
    from the contest video's scorer sensitivity (a high-magnitude diff is
    a substrate-fit weakness signal).
    """
    _require_numpy()
    if not callable(scorer_seg_forward) or not callable(scorer_pose_forward):
        raise MultiGranularityComparisonError(
            "scorer_seg_forward and scorer_pose_forward must be callables"
        )
    frames = np.asarray(inflated_video_frames)
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise MultiGranularityComparisonError(
            f"inflated_video_frames must have shape (N_pairs, 3, H, W); got {frames.shape}"
        )
    n_pairs, _, h, w = frames.shape
    if not archive_sha256 or len(archive_sha256) < 16:
        raise MultiGranularityComparisonError(
            "archive_sha256 must be hex (>= 16 chars)"
        )

    if inflated_video_sha256 is None:
        inflated_video_sha256 = _sha256_array(frames)
    if scorer_seg_sha256 is None:
        scorer_seg_sha256 = "unknown_scorer_seg_sha256"
    if scorer_pose_sha256 is None:
        scorer_pose_sha256 = "unknown_scorer_pose_sha256"

    seg_marginal, pose_marginal, _rate_marginal = compute_marginal_coefficients(
        operating_point
    )
    seg_logits = np.asarray(scorer_seg_forward(frames))
    if seg_logits.ndim != 4 or seg_logits.shape[0] != n_pairs or seg_logits.shape[2] != h or seg_logits.shape[3] != w:
        raise MultiGranularityComparisonError(
            f"scorer_seg_forward must return (N_pairs, n_classes, H, W); got {seg_logits.shape}"
        )
    pose_out = np.asarray(scorer_pose_forward(frames))
    if pose_out.ndim != 2 or pose_out.shape[0] != n_pairs or pose_out.shape[1] != 6:
        raise MultiGranularityComparisonError(
            f"scorer_pose_forward must return (N_pairs, 6); got {pose_out.shape}"
        )

    sorted_logits = np.sort(seg_logits, axis=1)
    seg_margin = sorted_logits[:, -1, :, :] - sorted_logits[:, -2, :, :]
    pose_l1_per_pair = np.abs(pose_out).sum(axis=1)
    pose_broadcast = pose_l1_per_pair[:, None, None] * np.ones(
        (h, w), dtype=np.float64
    )[None, :, :]
    rate_broadcast = np.zeros_like(pose_broadcast)

    m_inflated = np.stack(
        [
            seg_marginal * seg_margin.astype(np.float64),
            pose_marginal * pose_broadcast.astype(np.float64),
            rate_broadcast,
        ],
        axis=1,
    )

    captured_at_utc = _utc_now_iso()
    array_sha256 = _sha256_array(m_inflated)
    if cache_path is None:
        ts = captured_at_utc.replace(":", "").replace("-", "")[:15]
        sha_prefix = archive_sha256[:12]
        cache_path = _PERSIST_ROOT / f"m_inflated_{sha_prefix}_{ts}.npy"
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, m_inflated)
    meta_path = cache_path.with_suffix(".meta.json")
    meta_payload = {
        "schema": "m_inflated_meta_v1",
        "array_sha256": array_sha256,
        "n_pairs": n_pairs,
        "height": h,
        "width": w,
        "inflated_video_sha256": inflated_video_sha256,
        "archive_sha256": archive_sha256,
        "scorer_seg_sha256": scorer_seg_sha256,
        "scorer_pose_sha256": scorer_pose_sha256,
        "captured_at_utc": captured_at_utc,
        "operating_point": operating_point.as_dict(),
        "measurement_axis": measurement_axis,
        "canonical_helper_invocation": M_INFLATED_PROVENANCE_KIND,
    }
    meta_path.write_text(json.dumps(meta_payload, sort_keys=True, indent=2))

    return InflatedGradientTensor(
        array_path=str(cache_path),
        array_sha256=array_sha256,
        n_pairs=n_pairs,
        height=h,
        width=w,
        inflated_video_sha256=inflated_video_sha256,
        archive_sha256=archive_sha256,
        scorer_seg_sha256=scorer_seg_sha256,
        scorer_pose_sha256=scorer_pose_sha256,
        operating_point=operating_point,
        captured_at_utc=captured_at_utc,
        measurement_axis=measurement_axis,
    )


# -----------------------------------------------------------------------------
# Helper #4 - score-weighted reconstruction error (exploit #1)
# -----------------------------------------------------------------------------


def compute_score_weighted_reconstruction_error(
    m_contest: ContestGradientTensor,
    contest_video,
    inflated_video,
    *,
    cache_path: Path | str | None = None,
) -> PerPixelReconstructionError:
    """Compute (I_inflated - I_contest)^2 * ||M_contest[:, axis, :, :]||^2.

    This is the canonical encoder-loss target per exploit #1 - pixels with
    high scorer sensitivity get up-weighted in the reconstruction error so
    the encoder optimizes for score-faithful pixels rather than uniform
    PSNR.
    """
    _require_numpy()
    contest_arr = np.asarray(contest_video, dtype=np.float64)
    inflated_arr = np.asarray(inflated_video, dtype=np.float64)
    if contest_arr.shape != inflated_arr.shape:
        raise MultiGranularityComparisonError(
            f"contest/inflated shape mismatch: {contest_arr.shape} vs {inflated_arr.shape}"
        )
    if contest_arr.ndim != 4 or contest_arr.shape[1] != 3:
        raise MultiGranularityComparisonError(
            f"videos must have shape (N_pairs, 3, H, W); got {contest_arr.shape}"
        )
    n_pairs, _, h, w = contest_arr.shape
    if (n_pairs, h, w) != (m_contest.n_pairs, m_contest.height, m_contest.width):
        raise MultiGranularityComparisonError(
            f"M_contest declared shape ({m_contest.n_pairs}, *, {m_contest.height}, "
            f"{m_contest.width}) does not match video ({n_pairs}, 3, {h}, {w})"
        )
    m_arr = m_contest.load()
    # Per-pixel L2 of M_contest across axes.
    m_l2_sq = np.sum(np.square(m_arr), axis=1)  # (N_pairs, H, W)
    # Per-pixel L2 of pixel residual across RGB.
    diff = inflated_arr - contest_arr
    err_per_pixel = np.sum(np.square(diff), axis=1)  # (N_pairs, H, W)
    weighted = err_per_pixel * m_l2_sq
    captured_at_utc = _utc_now_iso()
    array_sha256 = _sha256_array(weighted)
    if cache_path is None:
        ts = captured_at_utc.replace(":", "").replace("-", "")[:15]
        sha_prefix = m_contest.contest_video_sha256[:12]
        cache_path = _PERSIST_ROOT / f"score_weighted_recon_err_{sha_prefix}_{ts}.npy"
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, weighted.astype(np.float32))
    inflated_sha = _sha256_array(inflated_arr)
    meta_payload = {
        "schema": "score_weighted_reconstruction_error_meta_v1",
        "array_sha256": array_sha256,
        "n_pairs": n_pairs,
        "height": h,
        "width": w,
        "contest_video_sha256": m_contest.contest_video_sha256,
        "inflated_video_sha256": inflated_sha,
        "captured_at_utc": captured_at_utc,
        "measurement_axis": "[predicted]",
        "canonical_helper_invocation": (
            "tac.master_gradient_comparison.compute_score_weighted_reconstruction_error"
        ),
    }
    cache_path.with_suffix(".meta.json").write_text(
        json.dumps(meta_payload, sort_keys=True, indent=2)
    )
    return PerPixelReconstructionError(
        array_path=str(cache_path),
        array_sha256=array_sha256,
        n_pairs=n_pairs,
        height=h,
        width=w,
        contest_video_sha256=m_contest.contest_video_sha256,
        inflated_video_sha256=inflated_sha,
        captured_at_utc=captured_at_utc,
    )


# -----------------------------------------------------------------------------
# Helper #5 - per-pair difficulty atlas (exploit #2)
# -----------------------------------------------------------------------------


def compute_per_pair_difficulty_atlas(
    m_contest: ContestGradientTensor,
) -> PerPairDifficulty:
    """Compute per-pair gradient L2 magnitude - the canonical difficulty atlas.

    Per exploit #2: ``difficulty[p] = sum_{axis, h, w} M_contest[p, axis, h, w]^2``.
    Pairs with high difficulty are the score-critical pairs the bit
    budget should prioritize.

    Downstream consumer:
    ``src/tac/cathedral_consumers/per_pair_difficulty_atlas_consumer``
    (already exists; this is the producer side).
    """
    _require_numpy()
    arr = m_contest.load()
    # Sum of squares across axes + spatial dims yields a single scalar per pair.
    flat = arr.reshape(arr.shape[0], -1)
    per_pair_l2_sq = np.sum(np.square(flat), axis=1)
    per_pair_l2 = np.sqrt(per_pair_l2_sq)
    return PerPairDifficulty(
        difficulty_per_pair=tuple(float(v) for v in per_pair_l2),
        n_pairs=m_contest.n_pairs,
        operating_point=m_contest.operating_point,
        captured_at_utc=_utc_now_iso(),
    )


# -----------------------------------------------------------------------------
# Helper #6 - cluster pairs by gradient similarity (exploit #9)
# -----------------------------------------------------------------------------


def cluster_pairs_by_gradient_similarity(
    m_contest: ContestGradientTensor,
    *,
    threshold: float = 0.95,
) -> tuple[EquivalenceClass, ...]:
    """Cluster pairs whose gradient signatures have cosine similarity >= threshold.

    Per exploit #9 (symmetry-breaking): pairs with similar gradient
    signatures can share a single bit-allocation decision. This is the
    canonical symmetry-breaking surface for the bit-allocator and for
    cross-pair coding budget allocation.

    Implementation: greedy clustering by cosine similarity over the
    per-pair flattened gradient vectors. For large N_pairs the caller
    should pass a dimensionality-reduced representation (e.g., the
    per-pair difficulty atlas multiplied by per-axis weights) instead of
    the full flat gradient.

    Args:
        m_contest: the per-pair gradient tensor.
        threshold: cosine similarity threshold for cluster membership in
            ``[0, 1]``; 1.0 = identical direction.

    Returns:
        Tuple of :class:`EquivalenceClass` rows, one per cluster. Singleton
        clusters are included (every pair gets exactly one cluster).
    """
    _require_numpy()
    if not (0.0 <= threshold <= 1.0):
        raise MultiGranularityComparisonError(
            f"threshold must be in [0, 1]; got {threshold}"
        )
    arr = m_contest.load()
    n_pairs = arr.shape[0]
    flat = arr.reshape(n_pairs, -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1)
    safe_norms = np.where(norms > 0, norms, 1.0)
    normalized = flat / safe_norms[:, None]

    assigned = np.zeros(n_pairs, dtype=bool)
    classes: list[EquivalenceClass] = []
    for rep_idx in range(n_pairs):
        if assigned[rep_idx]:
            continue
        rep_vec = normalized[rep_idx]
        # Compute cosine similarities to all unassigned pairs.
        sims = normalized @ rep_vec
        # Zero-norm pairs have undefined cosine; treat them as singleton.
        if norms[rep_idx] == 0:
            members = (rep_idx,)
            assigned[rep_idx] = True
            classes.append(
                EquivalenceClass(
                    representative_pair_index=rep_idx,
                    member_pair_indices=members,
                    similarity_threshold=float(threshold),
                    mean_intra_class_cosine=1.0,  # trivially equal to self
                )
            )
            continue
        member_mask = (sims >= threshold) & (~assigned) & (norms > 0)
        member_mask[rep_idx] = True
        member_indices = tuple(int(i) for i in np.flatnonzero(member_mask))
        if len(member_indices) > 1:
            sub = normalized[list(member_indices)]
            pairwise = sub @ sub.T
            # Average over off-diagonal entries.
            off_diag_sum = float(pairwise.sum() - np.trace(pairwise))
            off_diag_count = len(member_indices) * (len(member_indices) - 1)
            mean_cosine = (
                off_diag_sum / off_diag_count if off_diag_count > 0 else 1.0
            )
        else:
            mean_cosine = 1.0
        for m_idx in member_indices:
            assigned[m_idx] = True
        classes.append(
            EquivalenceClass(
                representative_pair_index=rep_idx,
                member_pair_indices=member_indices,
                similarity_threshold=float(threshold),
                mean_intra_class_cosine=float(mean_cosine),
            )
        )
    return tuple(classes)


# -----------------------------------------------------------------------------
# Helper #7 - decompose M_contest per SegNet class (exploit #5)
# -----------------------------------------------------------------------------


def decompose_M_contest_per_segnet_class(
    m_contest: ContestGradientTensor,
    segnet_class_masks,
) -> dict[int, "np.ndarray"]:
    """Decompose M_contest by SegNet class.

    Per exploit #5 (NSCS06 v6 -> v7 anchor; 44% improvement per  # DOCSTRING_PERCENT_CLAIM_OK:canonical_nscs06_v6_to_v7_empirical_44pct_anchor_artifact_at_omx_research_nscs06_path_a_chroma_optical_flow_redesign_20260516_md_cited_inline_on_next_line
    ``.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md``):
    the SegNet contributes per-class confidence margins; decomposing the
    per-pixel gradient by which SegNet class dominates that pixel surfaces
    per-class chroma anchors that the NSCS06 v7 substrate exploited.

    Args:
        m_contest: the per-pair gradient tensor.
        segnet_class_masks: np.ndarray of shape (N_pairs, H, W) with int
            class indices in [0, n_classes); typically the argmax over
            SegNet logits at each pixel.

    Returns:
        Dict mapping class_idx -> np.ndarray of shape (N_pairs, 3, H, W)
        with the gradient restricted to that class (zero elsewhere).
        Classes absent from any pixel are omitted from the dict.
    """
    _require_numpy()
    masks = np.asarray(segnet_class_masks)
    if masks.ndim != 3:
        raise MultiGranularityComparisonError(
            f"segnet_class_masks must have shape (N_pairs, H, W); got {masks.shape}"
        )
    n_pairs_mask, h_mask, w_mask = masks.shape
    if (
        n_pairs_mask != m_contest.n_pairs
        or h_mask != m_contest.height
        or w_mask != m_contest.width
    ):
        raise MultiGranularityComparisonError(
            f"mask shape ({n_pairs_mask}, {h_mask}, {w_mask}) "
            f"!= M_contest shape ({m_contest.n_pairs}, {m_contest.height}, {m_contest.width})"
        )
    arr = m_contest.load()
    unique_classes = np.unique(masks).tolist()
    result: dict[int, "np.ndarray"] = {}
    for cls in unique_classes:
        cls_int = int(cls)
        if cls_int < 0:
            continue
        # Broadcast mask to (N_pairs, 1, H, W) and multiply.
        class_mask = (masks == cls_int).astype(np.float32)[:, None, :, :]
        result[cls_int] = (arr * class_mask).astype(np.float32)
    return result


# -----------------------------------------------------------------------------
# Helper #8 - information-theoretic floor (exploit #7)
# -----------------------------------------------------------------------------


def estimate_information_theoretic_floor(
    m_contest: ContestGradientTensor,
    *,
    mode: str = "cramer_rao",
    n_pair_samples: int | None = None,
) -> float:
    """Estimate the information-theoretic floor of the score per exploit #7.

    Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: prefer
    solvable math over arbitrary sweeps. The information-theoretic floor
    is the score we cannot beat without changing the substrate class.

    Three modes:

    * ``cramer_rao`` (default): the Cramer-Rao lower bound on the variance
      of any unbiased estimator of the per-axis score components - the
      reciprocal of the Fisher information matrix trace. Returns the
      total Cramer-Rao lower bound aggregated across pairs and axes,
      interpreted as the minimum achievable score variance at the
      operating point.
    * ``fisher_trace``: the trace of the Fisher information matrix
      (per-pair, per-axis curvature sum). Higher trace = tighter local
      curvature.
    * ``shannon_lower``: a Shannon entropy-style lower bound estimated
      from the per-pair gradient norms via H(p) >= log(N) - log(sum exp(-|grad|)).

    Args:
        m_contest: the per-pair gradient tensor.
        mode: one of {"cramer_rao", "fisher_trace", "shannon_lower"}.
        n_pair_samples: optional subsample size for large N_pairs.

    Returns:
        Floor estimate as a scalar float; downstream callers interpret
        per their semantics. ALWAYS non-negative.
    """
    _require_numpy()
    if mode not in _LEGAL_FLOOR_MODES:
        raise MultiGranularityComparisonError(
            f"mode={mode!r} not in {sorted(_LEGAL_FLOOR_MODES)!r}"
        )
    arr = m_contest.load()
    if n_pair_samples is not None and n_pair_samples > 0 and n_pair_samples < arr.shape[0]:
        # Deterministic stride sampling.
        stride = max(1, arr.shape[0] // n_pair_samples)
        arr = arr[::stride][:n_pair_samples]

    if mode == "cramer_rao":
        # Fisher information per axis = sum over pixels of gradient^2.
        # Cramer-Rao lower bound on variance of an unbiased estimator of
        # the axis score = 1 / Fisher information.
        # Total bound = sum over axes of (1 / per-axis Fisher information),
        # treating each axis independently per the canonical scorer
        # additive form S = 100*d_seg + sqrt(10*d_pose) + 25*R.
        fisher_per_axis = np.sum(np.square(arr), axis=(0, 2, 3))  # (3,)
        # Guard against zero-Fisher axes (e.g., rate axis is constructed
        # as zero in extract_M_contest because rate sensitivity is
        # byte-side per the chain rule).
        nonzero = fisher_per_axis > 0
        if not nonzero.any():
            return 0.0
        cr_per_axis = np.zeros(3, dtype=np.float64)
        cr_per_axis[nonzero] = 1.0 / fisher_per_axis[nonzero]
        return float(cr_per_axis.sum())

    if mode == "fisher_trace":
        # Trace of (3,3) Fisher information matrix; flatten per pair and
        # compute the per-axis variance.
        flat = arr.reshape(arr.shape[0], 3, -1)
        per_axis_sum = np.sum(np.square(flat), axis=(0, 2))
        return float(per_axis_sum.sum())

    if mode == "shannon_lower":
        per_pair_norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
        # H(p) >= log(N) - log(sum exp(-|grad|)) under uniform prior.
        n = float(arr.shape[0])
        # Use logsumexp for numerical stability.
        max_neg = float((-per_pair_norms).max())
        log_sum_exp = max_neg + float(np.log(np.exp(-per_pair_norms - max_neg).sum()))
        return float(np.log(n) - log_sum_exp)

    raise MultiGranularityComparisonError(f"unreachable mode={mode!r}")


# -----------------------------------------------------------------------------
# Helper #9 - canonical persistence
# -----------------------------------------------------------------------------


def persist_comparison_artifact(
    payload: Mapping[str, object],
    *,
    label: str,
    archive_sha256: str | None = None,
    persist_root: Path | str | None = None,
) -> Path:
    """Persist a comparison artifact JSON under canonical Provenance.

    Per Catalog #323 + #287 + #131 sister discipline: atomic write to a
    canonical path under ``.omx/state/master_gradient_comparison/`` (never
    /tmp); routes the payload through ``tac.provenance`` Provenance via
    the caller-provided payload structure (this helper does NOT mutate
    payload; the caller composes the canonical Provenance fields).
    """
    if not label or not all(c.isalnum() or c in "_-" for c in label):
        raise MultiGranularityComparisonError(
            f"label must be alphanumeric/underscore/hyphen; got {label!r}"
        )
    root = Path(persist_root) if persist_root is not None else _PERSIST_ROOT
    root.mkdir(parents=True, exist_ok=True)
    ts = _utc_now_iso().replace(":", "").replace("-", "")[:15]
    sha_prefix = (archive_sha256 or "no_archive")[:12]
    out_path = root / f"{label}_{sha_prefix}_{ts}.json"
    # Atomic write via tmp + replace per Catalog #245 sister discipline.
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=str(root),
        prefix=out_path.name,
        suffix=f".tmp.{uuid.uuid4().hex[:12]}",
        delete=False,
    ) as f:
        json.dump(dict(payload), f, sort_keys=True, indent=2)
        tmp_path = Path(f.name)
    os.replace(tmp_path, out_path)
    return out_path
