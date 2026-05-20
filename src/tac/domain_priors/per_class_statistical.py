# SPDX-License-Identifier: MIT
"""Per-class statistical priors — wraps SegNet 5-class output for cathedral.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 Step 4.1 line 406: this wrapper
exposes per-class (SegNet 5-class) statistical priors so the cathedral
autopilot ranker + per-class bit-allocator (sister of Catalog #354 exploit
#5 ``per_segnet_class_chroma_consumer``) can ingest priors WITHOUT manual
edits.

SegNet outputs 5 classes per upstream/modules.py (verified via PV):

  * Class 0: background / sky / road-surface (large pixel count)
  * Class 1: vehicle (medium pixel count; high pose-relevance)
  * Class 2: pedestrian (small pixel count; high pose-relevance)
  * Class 3: lane-marking (medium pixel count; high seg-relevance)
  * Class 4: other-foreground (small pixel count; varies)

The priors are 3-fold per class:

  1. ``pixel_count_fraction`` — fraction of total pixels assigned to this
     class across the video. Drives per-class bit-allocator weighting.
  2. ``chroma_variance`` — variance of the chroma channels (Cb, Cr)
     within this class's pixel mask. Drives per-class chroma allocation
     (canonical mechanism for Catalog #354 exploit #5).
  3. ``motion_magnitude`` — average pose-magnitude associated with this
     class (cite the canonical openpilot mask prior contract).

Cross-references:
  * CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 Step 4.1 line 406
  * Catalog #354 ``check_master_gradient_exploit_consumers_complete``
    (exploit #5 ``per_segnet_class_chroma_consumer`` consumes this prior)
  * ``tac.categorical_substrate`` + ``tac.categorical_label_atoms``
    (sister surfaces this module sister-extends per Decision 10)
  * Catalog #287 / #323 (axis tag + Provenance discipline)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from tac.provenance.contract import Provenance


SEGNET_CLASS_COUNT: int = 5
"""Canonical 5-class output from upstream SegNet (smp.Unet effnet-b2)."""

SEGNET_CLASS_NAMES: tuple[str, ...] = (
    "background_sky_road",
    "vehicle",
    "pedestrian",
    "lane_marking",
    "other_foreground",
)
"""Operator-readable class names per SegNet class index."""


@dataclass(frozen=True)
class PerClassPrior:
    """One SegNet-class's statistical prior.

    Fields:
        class_index: SegNet class index in [0, 5).
        class_name: canonical name from :data:`SEGNET_CLASS_NAMES`.
        pixel_count_fraction: fraction in [0, 1] of total pixels in video
            assigned to this class.
        chroma_variance: variance of Cb+Cr chroma channels within this
            class's pixel mask. Higher = more chroma-coding budget needed.
        motion_magnitude_mean: average pose-magnitude (per-pair L2 of pose
            change) associated with this class. Higher = more
            pose-relevant.
    """

    class_index: int
    class_name: str
    pixel_count_fraction: float
    chroma_variance: float
    motion_magnitude_mean: float

    def __post_init__(self) -> None:
        if not isinstance(self.class_index, int):
            raise TypeError("class_index must be int")
        if self.class_index < 0 or self.class_index >= SEGNET_CLASS_COUNT:
            raise ValueError(
                f"class_index={self.class_index} must be in [0, {SEGNET_CLASS_COUNT})"
            )
        if not isinstance(self.class_name, str):
            raise TypeError("class_name must be string")
        if self.class_name != SEGNET_CLASS_NAMES[self.class_index]:
            raise ValueError(
                f"class_name={self.class_name!r} must equal canonical "
                f"SEGNET_CLASS_NAMES[{self.class_index}]={SEGNET_CLASS_NAMES[self.class_index]!r}"
            )
        for fname, fval in (
            ("pixel_count_fraction", self.pixel_count_fraction),
            ("chroma_variance", self.chroma_variance),
            ("motion_magnitude_mean", self.motion_magnitude_mean),
        ):
            if not isinstance(fval, (int, float)):
                raise TypeError(f"{fname} must be numeric, got {type(fval).__name__}")
            if fval != fval:
                raise ValueError(f"{fname} must not be NaN")
            if fval < 0:
                raise ValueError(f"{fname}={fval} must be >= 0")
        if self.pixel_count_fraction > 1.0 + 1e-9:
            raise ValueError(
                f"pixel_count_fraction={self.pixel_count_fraction} must be in [0, 1]"
            )


@dataclass(frozen=True)
class PerClassStatisticalPriors:
    """Per-class statistical priors atlas — canonical wrapper output.

    Fields:
        class_priors: tuple of 5 :class:`PerClassPrior` rows (one per
            SegNet class index).
        source_archive_sha256: archive sha priors were computed against.
        source_measurement_axis: e.g. ``"[predicted]"`` /
            ``"[contest-CPU]"``.
        source_scorer_kind: ``"segnet_5_class_argmax"`` (canonical) /
            ``"segnet_5_class_softmax"`` / ``"openpilot_mask_prior"``.
        canonical_openpilot_mask_prior_contract_cited: bool marking that
            this atlas is sister-extending the canonical openpilot mask
            prior contract (cite-chain per CLAUDE.md "Subagent coherence-
            by-default").
        provenance: canonical Provenance per Catalog #323.
    """

    class_priors: tuple[PerClassPrior, ...]
    source_archive_sha256: str
    source_measurement_axis: str
    source_scorer_kind: str
    canonical_openpilot_mask_prior_contract_cited: bool
    provenance: Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.class_priors, tuple):
            raise TypeError("class_priors must be tuple")
        if len(self.class_priors) != SEGNET_CLASS_COUNT:
            raise ValueError(
                f"class_priors length {len(self.class_priors)} != SEGNET_CLASS_COUNT {SEGNET_CLASS_COUNT}"
            )
        for i, p in enumerate(self.class_priors):
            if not isinstance(p, PerClassPrior):
                raise TypeError(
                    f"class_priors[{i}] must be PerClassPrior, got {type(p).__name__}"
                )
            if p.class_index != i:
                raise ValueError(
                    f"class_priors[{i}].class_index={p.class_index} must equal i={i}"
                )
        # Sum of pixel_count_fraction across classes must equal 1.0 (every
        # pixel must be assigned to exactly one class).
        total = sum(p.pixel_count_fraction for p in self.class_priors)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"sum of pixel_count_fraction = {total} must equal 1.0; "
                "every pixel must be assigned to exactly one class"
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
            raise ValueError("source_measurement_axis must be non-empty")
        VALID_KINDS = {"segnet_5_class_argmax", "segnet_5_class_softmax", "openpilot_mask_prior"}
        if self.source_scorer_kind not in VALID_KINDS:
            raise ValueError(
                f"source_scorer_kind={self.source_scorer_kind!r} must be one of {sorted(VALID_KINDS)!r}"
            )
        if not isinstance(self.canonical_openpilot_mask_prior_contract_cited, bool):
            raise TypeError(
                "canonical_openpilot_mask_prior_contract_cited must be bool"
            )
        if not isinstance(self.provenance, Provenance):
            raise TypeError(
                f"provenance must be Provenance, got {type(self.provenance).__name__}"
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe serialization."""
        from tac.provenance.validator import provenance_to_dict

        return {
            "schema": "per_class_statistical_priors_v1",
            "source_archive_sha256": self.source_archive_sha256,
            "source_measurement_axis": self.source_measurement_axis,
            "source_scorer_kind": self.source_scorer_kind,
            "canonical_openpilot_mask_prior_contract_cited": self.canonical_openpilot_mask_prior_contract_cited,
            "class_priors": [
                {
                    "class_index": p.class_index,
                    "class_name": p.class_name,
                    "pixel_count_fraction": p.pixel_count_fraction,
                    "chroma_variance": p.chroma_variance,
                    "motion_magnitude_mean": p.motion_magnitude_mean,
                }
                for p in self.class_priors
            ],
            "provenance": provenance_to_dict(self.provenance),
        }


def build_per_class_statistical_priors_from_scorer_output(
    pixel_count_fractions: Sequence[float],
    chroma_variances: Sequence[float],
    motion_magnitudes: Sequence[float],
    *,
    archive_sha256: str,
    measurement_axis: str,
    provenance: Provenance,
    source_scorer_kind: str = "segnet_5_class_argmax",
    canonical_openpilot_mask_prior_contract_cited: bool = True,
) -> PerClassStatisticalPriors:
    """Build :class:`PerClassStatisticalPriors` from per-class signal vectors.

    Args:
        pixel_count_fractions: length-5 sequence summing to 1.0.
        chroma_variances: length-5 sequence of non-negative variances.
        motion_magnitudes: length-5 sequence of non-negative mean magnitudes.
        archive_sha256: archive sha (cite-chain).
        measurement_axis: axis tag.
        provenance: canonical Provenance per Catalog #323.
        source_scorer_kind: ``"segnet_5_class_argmax"`` (canonical) etc.
        canonical_openpilot_mask_prior_contract_cited: declares that the
            atlas is sister-extending the canonical openpilot mask prior
            contract.

    Returns:
        Frozen :class:`PerClassStatisticalPriors`.
    """
    for name, vec in (
        ("pixel_count_fractions", pixel_count_fractions),
        ("chroma_variances", chroma_variances),
        ("motion_magnitudes", motion_magnitudes),
    ):
        if not isinstance(vec, Sequence) or isinstance(vec, (str, bytes, bytearray)):
            raise TypeError(f"{name} must be Sequence; got {type(vec).__name__}")
        if len(vec) != SEGNET_CLASS_COUNT:
            raise ValueError(
                f"{name} length {len(vec)} must equal SEGNET_CLASS_COUNT {SEGNET_CLASS_COUNT}"
            )
    priors: list[PerClassPrior] = []
    for class_index in range(SEGNET_CLASS_COUNT):
        priors.append(
            PerClassPrior(
                class_index=class_index,
                class_name=SEGNET_CLASS_NAMES[class_index],
                pixel_count_fraction=float(pixel_count_fractions[class_index]),
                chroma_variance=float(chroma_variances[class_index]),
                motion_magnitude_mean=float(motion_magnitudes[class_index]),
            )
        )
    return PerClassStatisticalPriors(
        class_priors=tuple(priors),
        source_archive_sha256=archive_sha256,
        source_measurement_axis=measurement_axis,
        source_scorer_kind=source_scorer_kind,
        canonical_openpilot_mask_prior_contract_cited=bool(
            canonical_openpilot_mask_prior_contract_cited
        ),
        provenance=provenance,
    )


__all__ = [
    "SEGNET_CLASS_COUNT",
    "SEGNET_CLASS_NAMES",
    "PerClassPrior",
    "PerClassStatisticalPriors",
    "build_per_class_statistical_priors_from_scorer_output",
]
