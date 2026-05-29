# SPDX-License-Identifier: MIT
"""Canonical cls_lowres downsample policies for v8 CH08 v3 cls_stream slot.

Per Wave 5 NSCS06 v8 cargo-cult re-audit (2026-05-29) finding #6: the
existing trainer's strided ``cls_full[:, ::ds, ::ds]`` indexing point-samples
the TOP-LEFT pixel of each ``ds * ds`` cell. This is the canonical inverse
of inflate.py's ``Image.NEAREST`` upsample so byte-parity is preserved at the
"uniform-cls" round-trip invariant tested in
``tests/test_cls_stream_wire_in.py::test_inflate_v3_with_uniform_class_matches_v2``.

**Cargo-cult #6 (Wave 5 NEW)**: in boundary cells where a ``ds * ds`` patch
has 3-of-4 pixels of class C_dom but the top-left pixel is class C_boundary,
the strided NEAREST downsample throws away the dominant class. The L0
SCAFFOLD inflate then upsamples class=C_boundary back to the full ``ds * ds``
cell so the chroma LUT lookup uses the WRONG per-class anchor for 75% of
boundary-cell pixels.

This is the SAME META-class as cargo-cult #4 (per-(level,class) MEDIAN
aggregation per ``CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS``):
both choose a representative pixel/value per spatial bin without empirical
validation against alternative aggregation policies. The strided NEAREST
choice is INHERITED from the canonical sister pattern at inflate.py's
``Image.NEAREST`` upsample but NEVER empirically validated against per-cell
MODE (most-frequent class) on the actual contest scorer response.

**Honest disclosure** per CLAUDE.md "Apples-to-apples evidence discipline":
- The NEAREST (strided) policy is currently the BYTE-DEFAULT and preserves
  archive-byte parity with all prior dispatches.
- The MODE policy is an UNWIND alternative that may produce different
  archive bytes but preserves the canonical bit-budget identity (same
  ``num_pairs * grayscale_h * grayscale_w`` byte cost).
- Operator opt-in is REQUIRED to switch from NEAREST to MODE because
  it changes archive bytes (per CLAUDE.md "Frontier scores are
  pointer-only" + Catalog #110/#113 HISTORICAL_PROVENANCE for prior
  empirical anchors keyed by NEAREST policy).

CLAUDE.md compliance:
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: NEW module; no
  mutation of sister inflate.py or archive.py
- Catalog #287 + #323 canonical Provenance: no score claim; pure helper
- Catalog #290 canonical-vs-unique decision per layer: this is a
  PER-SUBSTRATE-UNIQUE policy choice (NOT shared with sister substrates;
  the v7 strip-everything substrate uses arith-coded cls stream so the
  downsample policy is different there)
- Catalog #335 cathedral consumer auto-discovery: this module is the
  canonical helper a future ``cls_lowres_policy_consumer`` would route
  through
- Catalog #344 canonical equation cross-reference: see
  ``canonical_equations/registry.jsonl`` entry
  ``cls_lowres_downsample_policy_boundary_preservation_v1`` (registered
  Wave 5)

6-hook wire-in declaration per Catalog #125:

* hook #1 sensitivity-map = ACTIVE (per-cell MODE preserves per-pixel
  SegNet class boundary information that point-sample NEAREST destroys)
* hook #2 Pareto constraint = N/A (same canonical cls_stream byte cost
  ``num_pairs * gh * gw`` regardless of policy choice)
* hook #3 bit-allocator = N/A (same byte cost; no per-tensor bit
  reallocation)
* hook #4 cathedral autopilot dispatch = ACTIVE (policy selection IS
  the canonical disambiguator between BYTE-DEFAULT-NEAREST vs
  BOUNDARY-PRESERVING-MODE arms)
* hook #5 continual-learning posterior = ACTIVE (paired smoke would
  emit empirical anchor for canonical equation
  ``cls_lowres_downsample_policy_boundary_preservation_v1``)
* hook #6 probe-disambiguator = ACTIVE PRIMARY (this module IS the
  canonical disambiguator for cargo-cult #6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

__all__ = [
    "CANONICAL_DOWNSAMPLE_POLICY_BYTE_DEFAULT",
    "CLS_LOWRES_DOWNSAMPLE_POLICY_NON_PROMOTABLE_PROVENANCE",
    "SUPPORTED_DOWNSAMPLE_POLICIES",
    "ClsLowresDownsampleError",
    "ClsLowresDownsampleVerdict",
    "derive_cls_lowres_from_cls_full",
    "verify_cls_lowres_downsample_invariants",
]


SUPPORTED_DOWNSAMPLE_POLICIES: Final[tuple[str, ...]] = (
    "nearest_strided_top_left",
    "mode_per_cell",
)
"""Canonical supported policy set.

* ``nearest_strided_top_left``: BYTE-DEFAULT. Point-samples the top-left
  pixel of each ``ds * ds`` cell via ``cls_full[:, ::ds, ::ds]`` strided
  indexing. Canonical inverse of ``Image.NEAREST`` upsample at inflate.
  Preserves archive-byte parity with all dispatches prior to 2026-05-29.

* ``mode_per_cell``: UNWIND PATH per Wave 5 cargo-cult #6. For each
  ``ds * ds`` cell, selects the MOST-FREQUENT class label across all
  ``ds * ds`` pixels via ``np.argmax(np.bincount(cell_flat))``. Boundary-
  preserving: a cell with 3-of-4 pixels of class C_dom is labeled C_dom
  regardless of top-left pixel. Same byte cost as NEAREST but different
  archive bytes (operator opt-in REQUIRED per CLAUDE.md "Frontier scores
  are pointer-only").
"""

CANONICAL_DOWNSAMPLE_POLICY_BYTE_DEFAULT: Final[str] = "nearest_strided_top_left"
"""Canonical byte-default policy. PRESERVES archive-byte parity with all
prior empirical anchors keyed by NEAREST strided downsample at trainer.

Switching the default away from this token requires operator opt-in
because it changes archive bytes for all v3 dispatches per CLAUDE.md
"Frontier scores are pointer-only" non-negotiable + Catalog #110/#113
HISTORICAL_PROVENANCE.
"""

CLS_LOWRES_DOWNSAMPLE_POLICY_NON_PROMOTABLE_PROVENANCE: dict[str, Any] = {
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "axis_tag": "[predicted]",
    "evidence_grade": "research-signal",
    "blockers": (
        "cls_lowres_downsample_policy_is_canonical_helper_not_score_claim",
        "policy_choice_disambiguates_byte_default_vs_boundary_preserving_arms",
    ),
}
"""Canonical non-promotable provenance per Catalog #287 + #323 + #341."""


class ClsLowresDownsampleError(ValueError):
    """Raised when the cls_lowres downsample policy cannot be honored faithfully."""


@dataclass(frozen=True)
class ClsLowresDownsampleVerdict:
    """Verdict from the cls_lowres downsample helper.

    Carries:
    - the canonical policy token (nearest_strided_top_left or mode_per_cell)
    - the derived cls_lowres array shape + dtype + sha256
    - per-cell mode-vs-nearest agreement fraction (for empirical comparison)
    - non-promotable contract per Catalog #287 + #323

    The agreement_fraction is a research-signal metric: if it is close to
    1.0 (e.g. >0.99), the BYTE-DEFAULT NEAREST policy is empirically
    indistinguishable from MODE for the input cls_full; the cargo-cult #6
    unwind is unnecessary at this input distribution. If it is far from
    1.0 (e.g. <0.9), MODE materially differs from NEAREST and the unwind
    becomes empirically relevant.
    """

    policy: str
    cls_lowres_shape: tuple[int, int, int]
    cls_lowres_sha256: str
    mode_vs_nearest_agreement_fraction: float
    """Per-cell agreement fraction between MODE and NEAREST. In [0.0, 1.0].
    1.0 means the two policies produce byte-identical cls_lowres; lower
    values quantify the empirical relevance of cargo-cult #6 unwind for
    this input cls_full distribution."""

    axis_tag: str = "[predicted]"
    score_claim: bool = False
    promotion_eligible: bool = False
    evidence_grade: str = "research-signal"

    def __post_init__(self) -> None:
        if self.policy not in SUPPORTED_DOWNSAMPLE_POLICIES:
            raise ClsLowresDownsampleError(
                f"policy={self.policy!r} not in {SUPPORTED_DOWNSAMPLE_POLICIES}"
            )
        if len(self.cls_lowres_shape) != 3:
            raise ClsLowresDownsampleError(
                f"cls_lowres_shape must be 3D; got {self.cls_lowres_shape}"
            )
        if not (0.0 <= self.mode_vs_nearest_agreement_fraction <= 1.0):
            raise ClsLowresDownsampleError(
                f"mode_vs_nearest_agreement_fraction={self.mode_vs_nearest_agreement_fraction} "
                "outside [0.0, 1.0]"
            )
        if self.score_claim is not False:
            raise ClsLowresDownsampleError(
                "score_claim MUST be False for cls_lowres verdict per Catalog #287"
            )
        if self.promotion_eligible is not False:
            raise ClsLowresDownsampleError(
                "promotion_eligible MUST be False per Catalog #287 + #323"
            )

    def as_dict(self) -> dict[str, object]:
        """Serialize to JSON-safe dict per Catalog #287 + #323."""
        return {
            "policy": self.policy,
            "cls_lowres_shape": list(self.cls_lowres_shape),
            "cls_lowres_sha256": self.cls_lowres_sha256,
            "mode_vs_nearest_agreement_fraction": self.mode_vs_nearest_agreement_fraction,
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "evidence_grade": self.evidence_grade,
        }


def derive_cls_lowres_from_cls_full(
    cls_full: np.ndarray,
    *,
    grayscale_h: int,
    grayscale_w: int,
    grayscale_downsample: int,
    num_segnet_classes: int,
    policy: Literal[
        "nearest_strided_top_left", "mode_per_cell"
    ] = CANONICAL_DOWNSAMPLE_POLICY_BYTE_DEFAULT,
) -> tuple[np.ndarray, ClsLowresDownsampleVerdict]:
    """Derive the cls_lowres array from cls_full per the canonical policy.

    Per Wave 5 NSCS06 v8 cargo-cult #6 unwind 2026-05-29: this helper
    extracts the strided-NEAREST logic that previously lived inline in
    ``experiments/train_substrate_nscs06_v8_chroma_lut.py`` and adds the
    boundary-preserving MODE alternative arm.

    Args:
        cls_full: ``(n_pairs, H, W)`` uint8 array of SegNet argmax class
            labels at full output resolution.
        grayscale_h, grayscale_w: target low-res shape (must satisfy
            ``H == grayscale_h * grayscale_downsample`` and
            ``W == grayscale_w * grayscale_downsample``).
        grayscale_downsample: downsample factor (positive int). Must
            evenly divide H AND W per the canonical inflate-side
            ``Image.NEAREST`` upsample contract.
        num_segnet_classes: number of SegNet classes (default 5; used for
            input validation and MODE-policy bincount sizing).
        policy: one of :data:`SUPPORTED_DOWNSAMPLE_POLICIES`. Default
            :data:`CANONICAL_DOWNSAMPLE_POLICY_BYTE_DEFAULT` preserves
            archive-byte parity.

    Returns:
        Tuple of ``(cls_lowres, verdict)`` where ``cls_lowres`` has shape
        ``(n_pairs, grayscale_h, grayscale_w)`` uint8 and ``verdict``
        carries the canonical research-signal Provenance.

    Raises:
        ClsLowresDownsampleError: on input shape mismatch, invalid policy,
            invalid grayscale_downsample, or class-label out-of-range.
    """
    import hashlib

    if cls_full.dtype != np.uint8:
        raise ClsLowresDownsampleError(
            f"cls_full dtype must be uint8; got {cls_full.dtype}"
        )
    if cls_full.ndim != 3:
        raise ClsLowresDownsampleError(
            f"cls_full must be 3D (n_pairs, H, W); got {cls_full.shape}"
        )
    if grayscale_downsample < 1:
        raise ClsLowresDownsampleError(
            f"grayscale_downsample={grayscale_downsample} must be >= 1"
        )
    if num_segnet_classes < 1:
        raise ClsLowresDownsampleError(
            f"num_segnet_classes={num_segnet_classes} must be >= 1"
        )
    if policy not in SUPPORTED_DOWNSAMPLE_POLICIES:
        raise ClsLowresDownsampleError(
            f"policy={policy!r} not in {SUPPORTED_DOWNSAMPLE_POLICIES}"
        )

    n_pairs, h_full, w_full = cls_full.shape
    expected_h = grayscale_h * grayscale_downsample
    expected_w = grayscale_w * grayscale_downsample
    if h_full < expected_h or w_full < expected_w:
        raise ClsLowresDownsampleError(
            f"cls_full HxW={h_full}x{w_full} smaller than required "
            f"{expected_h}x{expected_w} (grayscale_h*ds x grayscale_w*ds)"
        )
    cls_full_in_range = cls_full[:, :expected_h, :expected_w]
    if int(cls_full_in_range.max(initial=0)) >= num_segnet_classes:
        raise ClsLowresDownsampleError(
            f"cls_full label {int(cls_full_in_range.max())} >= num_segnet_classes "
            f"{num_segnet_classes}"
        )

    # ALWAYS compute NEAREST as the reference for the agreement-fraction
    # research-signal metric; this is byte-cheap and lets every call surface
    # the empirical relevance of the MODE unwind.
    nearest = cls_full_in_range[:, ::grayscale_downsample, ::grayscale_downsample]
    if nearest.shape != (n_pairs, grayscale_h, grayscale_w):
        raise ClsLowresDownsampleError(
            f"nearest downsample shape {nearest.shape} != expected "
            f"({n_pairs}, {grayscale_h}, {grayscale_w}) — invariant violated"
        )

    if policy == "nearest_strided_top_left":
        cls_lowres = nearest
        mode_vs_nearest_agreement_fraction = 1.0  # tautology when policy IS nearest
    else:
        # MODE per cell: reshape (n_pairs, gh, ds, gw, ds) -> per-cell mode.
        reshaped = cls_full_in_range.reshape(
            n_pairs, grayscale_h, grayscale_downsample, grayscale_w, grayscale_downsample
        )
        # Move ds axes adjacent: (n_pairs, gh, gw, ds, ds)
        per_cell = reshaped.transpose(0, 1, 3, 2, 4).reshape(
            n_pairs, grayscale_h, grayscale_w, grayscale_downsample * grayscale_downsample
        )
        # Bincount per cell along the last axis; argmax = mode.
        # Cell-by-cell loop in numpy is O(n_pairs * gh * gw * num_segnet_classes)
        # which is acceptable at canonical contest shapes (600 * 96 * 128 * 5 ~ 37M ops).
        cls_lowres = np.zeros((n_pairs, grayscale_h, grayscale_w), dtype=np.uint8)
        for c in range(num_segnet_classes):
            # Count of class c per cell: shape (n_pairs, gh, gw)
            counts_c = (per_cell == c).sum(axis=-1)
            if c == 0:
                best_counts = counts_c.copy()
                # cls_lowres already 0
            else:
                # Update where counts_c strictly exceeds best (first-wins tie-break;
                # preserves NEAREST tie-break behavior when ds==1 trivially)
                update_mask = counts_c > best_counts
                cls_lowres[update_mask] = np.uint8(c)
                best_counts = np.where(update_mask, counts_c, best_counts)
        # Compute agreement fraction (research-signal): how often does MODE
        # agree with NEAREST at the same (n_pairs, gh, gw) cell?
        agreement_count = int((cls_lowres == nearest).sum())
        total_cells = n_pairs * grayscale_h * grayscale_w
        mode_vs_nearest_agreement_fraction = (
            agreement_count / total_cells if total_cells > 0 else 1.0
        )

    cls_lowres = np.ascontiguousarray(cls_lowres, dtype=np.uint8)
    sha = hashlib.sha256(cls_lowres.tobytes()).hexdigest()

    verdict = ClsLowresDownsampleVerdict(
        policy=policy,
        cls_lowres_shape=cls_lowres.shape,
        cls_lowres_sha256=sha,
        mode_vs_nearest_agreement_fraction=mode_vs_nearest_agreement_fraction,
    )
    return cls_lowres, verdict


def verify_cls_lowres_downsample_invariants(
    cls_lowres: np.ndarray,
    *,
    expected_shape: tuple[int, int, int],
    num_segnet_classes: int,
) -> None:
    """Verify canonical post-derivation invariants on cls_lowres.

    Per CLAUDE.md "Beauty, simplicity, and developer experience" + Catalog
    #295 strict-load discipline: every consumer of cls_lowres should call
    this to surface invariant violations at the call site rather than at
    archive pack/parse time.
    """
    if cls_lowres.dtype != np.uint8:
        raise ClsLowresDownsampleError(
            f"cls_lowres dtype must be uint8; got {cls_lowres.dtype}"
        )
    if cls_lowres.shape != expected_shape:
        raise ClsLowresDownsampleError(
            f"cls_lowres shape {cls_lowres.shape} != expected {expected_shape}"
        )
    if int(cls_lowres.max(initial=0)) >= num_segnet_classes:
        raise ClsLowresDownsampleError(
            f"cls_lowres label {int(cls_lowres.max())} >= num_segnet_classes "
            f"{num_segnet_classes}"
        )
