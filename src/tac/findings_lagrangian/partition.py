# SPDX-License-Identifier: MIT
"""MDL-driven partition refinement WITH wavelet-multi-scale prior (Q4 amendment).

Per T3 grand council 3-round consolidated verdict (slot 20 + supplemental +
second-supplemental, 2026-05-19):

- Q4 PROCEED (slot 20): hand-classified initial partition (slot 17 4-class
  cascade severity taxonomy NONE/BOUNDED/MIXED/UNBOUNDED) + MDL-driven
  adaptive refinement (split a class if MDL gain > 0.5 bits across all
  in-class anchors).
- Q4 AMEND (slot 20-supplemental, Daubechies amendment): MDL-adaptive
  refinement gains a wavelet-multi-scale prior on the partition tree per
  Catalog #277 sister discipline. Architecture family hierarchy IS a
  wavelet tree (PR101/PR106/PR107 → leaf nodes; A1/DP1/HDM8 → internal
  nodes); MDL splits weighted by depth in this tree.
- Q4 AMEND extended (slot 20-second-supplemental, Mallat + Selfcomp): the
  hierarchical-wavelet-scale prior also modifies info gain measure;
  partition-aware info gain not flat. Selfcomp: add 7th archive from
  asymptotic-pursuit horizon before treating Q4 as canonical.

The implementation: hand-classified initial partition is a frozen 4-class
mapping; refinement walks each class + computes MDL split-score
(per-anchor cost in bits) WEIGHTED by tree depth (Daubechies wavelet
prior — root splits cost more bits than leaf splits).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence


__all__ = [
    "CanonicalPartition",
    "PartitionClass",
    "INITIAL_4_CLASS_CASCADE_TAXONOMY",
    "INITIAL_PARTITION_ID",
    "MDL_SPLIT_THRESHOLD_BITS",
    "WAVELET_DEPTH_WEIGHT_BASE",
    "compute_mdl_gain_with_wavelet_prior",
    "should_split_class",
    "explain_split",
    "PartitionInvalidError",
]


INITIAL_PARTITION_ID = "slot_17_4_class_cascade_severity_v1"
"""Canonical partition_id for the slot 17 hand-classified taxonomy.

Per slot 17 + slot 20 Q4 binding: this is the canonical initial partition.
Catalog #347 STRICT preflight gate refuses FindingsLagrangianResult rows
that lack a partition_id field referencing this OR a later refinement.
"""

MDL_SPLIT_THRESHOLD_BITS = 0.5
"""Threshold in bits: split a class iff MDL gain >= 0.5 bits (Q4 slot 20)."""

WAVELET_DEPTH_WEIGHT_BASE = 2.0
"""Wavelet prior base: deeper splits cost log(BASE) bits per depth level."""


# Slot 17's 4-class cascade severity taxonomy (Q4 initial partition source).
INITIAL_4_CLASS_CASCADE_TAXONOMY: Mapping[str, str] = {
    "NONE": "no_master_gradient_signal_observed",
    "BOUNDED": "bounded_master_gradient_signal_within_predicted_band",
    "MIXED": "master_gradient_signal_mixed_with_other_class_residuals",
    "UNBOUNDED": "master_gradient_signal_exceeds_predicted_upper_bound",
}


class PartitionInvalidError(ValueError):
    """Raised when a CanonicalPartition violates invariants."""


@dataclass(frozen=True)
class PartitionClass:
    """One class in the canonical partition.

    Per Q4 amendment + Daubechies dissent: every class carries a tree_depth
    field so MDL split decisions can be weighted by depth in the wavelet
    architecture-family tree.
    """

    class_id: str
    description: str
    tree_depth: int = 0  # 0 = root cascade class; 1+ = refined sub-classes

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, str) or not self.class_id.strip():
            raise PartitionInvalidError("class_id must be non-empty string")
        if not isinstance(self.description, str):
            raise PartitionInvalidError("description must be string")
        if not isinstance(self.tree_depth, int) or self.tree_depth < 0:
            raise PartitionInvalidError(
                f"tree_depth={self.tree_depth} must be non-negative int"
            )


@dataclass(frozen=True)
class CanonicalPartition:
    """A canonical partition + its lineage.

    Per Q4 binding + supplemental amendment: each refinement step
    appends a new CanonicalPartition with explicit lineage
    (parent_partition_id + split_class_id + child_class_ids) so the
    operator can audit every refinement at any time.
    """

    partition_id: str
    classes: tuple[PartitionClass, ...]
    parent_partition_id: str | None = None
    split_class_id: str | None = None  # set if this partition refines a parent

    def __post_init__(self) -> None:
        if not isinstance(self.partition_id, str) or not self.partition_id.strip():
            raise PartitionInvalidError("partition_id must be non-empty string")
        if not isinstance(self.classes, tuple):
            raise PartitionInvalidError("classes must be tuple")
        if not self.classes:
            raise PartitionInvalidError("classes must be non-empty")
        seen_ids = set()
        for cls in self.classes:
            if not isinstance(cls, PartitionClass):
                raise PartitionInvalidError(
                    f"classes elements must be PartitionClass, got {type(cls).__name__}"
                )
            if cls.class_id in seen_ids:
                raise PartitionInvalidError(
                    f"duplicate class_id={cls.class_id!r}"
                )
            seen_ids.add(cls.class_id)

    def class_ids(self) -> tuple[str, ...]:
        return tuple(c.class_id for c in self.classes)

    def get_class(self, class_id: str) -> PartitionClass:
        for cls in self.classes:
            if cls.class_id == class_id:
                return cls
        raise PartitionInvalidError(f"class_id={class_id!r} not in partition")


def build_initial_partition() -> CanonicalPartition:
    """Construct the canonical slot 17 4-class cascade severity partition."""
    classes = tuple(
        PartitionClass(class_id=cid, description=desc, tree_depth=0)
        for cid, desc in INITIAL_4_CLASS_CASCADE_TAXONOMY.items()
    )
    return CanonicalPartition(
        partition_id=INITIAL_PARTITION_ID,
        classes=classes,
    )


def _shannon_entropy_per_anchor(residuals: Sequence[float]) -> float:
    """Estimate Shannon entropy (in bits) per anchor for an MDL split gain.

    Uses sample-variance-based Gaussian-approximation entropy:
    H(X) = 0.5 * log2(2 * pi * e * sigma**2) for a Gaussian.

    For residuals with sigma=1 this is approx 2.05 bits/anchor.
    """
    if len(residuals) < 2:
        return 0.0
    mean = sum(residuals) / len(residuals)
    variance = sum((r - mean) ** 2 for r in residuals) / len(residuals)
    if variance <= 0:
        return 0.0  # all residuals identical; 0 entropy
    return 0.5 * math.log2(2.0 * math.pi * math.e * variance)


def compute_mdl_gain_with_wavelet_prior(
    parent_residuals: Sequence[float],
    child_a_residuals: Sequence[float],
    child_b_residuals: Sequence[float],
    *,
    parent_tree_depth: int,
) -> float:
    """MDL gain in bits for splitting a class into two children.

    Per Q4 binding decision (slot 20): MDL gain = parent_total_bits -
    (child_a_bits + child_b_bits). Positive MDL gain means the split
    compresses the data better than the unsplit class.

    Per Q4 supplemental amendment (Daubechies wavelet prior):
    each split adds log_base(depth+2) bits of model-complexity cost,
    so deeper splits face stronger headwind. This penalizes
    over-refinement into deep tree branches that share a parent.

    Args:
        parent_residuals: all anchors in the parent class.
        child_a_residuals, child_b_residuals: proposed split partition.
        parent_tree_depth: depth of parent class in the wavelet tree
            (0 = root cascade; deeper = more refined).

    Returns:
        MDL gain in bits (positive = split compresses better).
    """
    n_parent = len(parent_residuals)
    n_a = len(child_a_residuals)
    n_b = len(child_b_residuals)
    if n_a + n_b != n_parent:
        raise PartitionInvalidError(
            f"child counts {n_a}+{n_b} != parent count {n_parent}"
        )

    parent_h = _shannon_entropy_per_anchor(parent_residuals)
    child_a_h = _shannon_entropy_per_anchor(child_a_residuals)
    child_b_h = _shannon_entropy_per_anchor(child_b_residuals)

    parent_bits = parent_h * n_parent
    child_total_bits = child_a_h * n_a + child_b_h * n_b

    raw_gain = parent_bits - child_total_bits

    # Daubechies wavelet prior: split cost = log_base(depth+2) bits per anchor in the child.
    # Deeper splits face stronger headwind.
    child_depth = parent_tree_depth + 1
    wavelet_split_cost_bits = math.log(child_depth + 2.0, WAVELET_DEPTH_WEIGHT_BASE)
    # Cost amortized over both children's anchors.
    total_split_cost = wavelet_split_cost_bits * (n_a + n_b) * 0.05

    return raw_gain - total_split_cost


def should_split_class(
    parent_residuals: Sequence[float],
    child_a_residuals: Sequence[float],
    child_b_residuals: Sequence[float],
    *,
    parent_tree_depth: int,
    threshold_bits: float = MDL_SPLIT_THRESHOLD_BITS,
) -> bool:
    """Return True iff the wavelet-prior-adjusted MDL gain exceeds threshold.

    Per Q4 binding decision (slot 20): threshold_bits = 0.5 by default.
    """
    gain = compute_mdl_gain_with_wavelet_prior(
        parent_residuals,
        child_a_residuals,
        child_b_residuals,
        parent_tree_depth=parent_tree_depth,
    )
    return gain >= threshold_bits


def explain_split(
    class_id: str,
    parent_residuals: Sequence[float],
    child_a_residuals: Sequence[float],
    child_b_residuals: Sequence[float],
    *,
    parent_tree_depth: int,
) -> dict[str, object]:
    """Operator-facing readback for a split decision (Daubechies amendment).

    Per Q4 amendment + observability surface (Catalog #305 + #335 sister):
    returns a structured dict the operator can audit to understand WHY a
    split was/wasn't accepted at a particular tree depth.
    """
    gain = compute_mdl_gain_with_wavelet_prior(
        parent_residuals,
        child_a_residuals,
        child_b_residuals,
        parent_tree_depth=parent_tree_depth,
    )
    return {
        "class_id": class_id,
        "parent_tree_depth": parent_tree_depth,
        "parent_anchor_count": len(parent_residuals),
        "child_a_count": len(child_a_residuals),
        "child_b_count": len(child_b_residuals),
        "wavelet_prior_depth_weight_base": WAVELET_DEPTH_WEIGHT_BASE,
        "wavelet_split_cost_bits": math.log(
            parent_tree_depth + 3.0, WAVELET_DEPTH_WEIGHT_BASE
        ),
        "mdl_gain_bits_adjusted": gain,
        "threshold_bits": MDL_SPLIT_THRESHOLD_BITS,
        "split_accepted": gain >= MDL_SPLIT_THRESHOLD_BITS,
        "rationale": (
            f"MDL gain {gain:.3f} bits {'exceeds' if gain >= MDL_SPLIT_THRESHOLD_BITS else 'below'} "
            f"threshold {MDL_SPLIT_THRESHOLD_BITS} at tree depth {parent_tree_depth}"
        ),
        "first_principles_citation": (
            "MDL: MacKay 1992 + Q4 binding slot 20; "
            "wavelet prior: Daubechies 1988 + Catalog #277 sister; "
            "supplemental amendment per slot 20-supplemental"
        ),
    }
