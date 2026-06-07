# SPDX-License-Identifier: MIT
"""Representative hard-region miner for HiNeRV target-class birth.

The launch gate (``tac.analysis.nerv_long_run_launch_gate``) refuses an L5
verdict when the only birth evidence is the trivial *birth-at-init* case: a
single easy region solved once.  The
``hi_nerv_representative_region_coverage.v1`` schema is the contract that
proves birth generalizes across the kinds of regions the contest SegNet term
actually cares about — across distinct classes and across size scales.

This module produces that coverage evidence's *input*: a deterministic,
diversity-enforced ranking of the worst connected target-class regions on a
batch of frames.  It reuses the canonical pricing in
``tac.substrates.hi_nerv.target_region_birth`` (so the score-debt units are the
exact same contest units the actuator and gate use) and augments each region
with:

* margin difficulty (mean / min frontier margin via ``region_margin_stats``)
  — a region whose wrong pixels sit just below the decision boundary is easier
  to birth than one deep in an impostor's basin;
* a size class (``small`` < 64 px, ``medium`` < 4096 px, ``large`` otherwise)
  so coverage can prove birth at multiple spatial scales;
* an optional pose-coupling risk (mean of a supplied per-pixel pose-Jacobian
  magnitude map over the region) — high pose coupling means a seg birth there
  risks PoseNet geometry, so the planner should treat it carefully.

Diversity is enforced by **round-robin across ``class_index``**: a frame with
ten class-0 regions and one class-3 region cannot let class-0 fill the whole
``top_k`` and starve class-3 from the plan.

Everything is numpy/scipy only — no torch, no MLX — so the same ranking serves
the live MLX forward, fake-quant forward, parse-back, and inflate replay
surfaces without backend drift.  The rows carry NO score/promotion authority
keys: they are planning evidence that warm-starts the actuator, never a score
claim.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from tac.substrates.hi_nerv.target_region_birth import (
    TargetRegionDebt,
    find_target_region_debts,
    region_margin_stats,
)

HARD_REGION_PLAN_SCHEMA = "hi_nerv_hard_region_mining_plan.v1"
REPRESENTATIVE_COVERAGE_SCHEMA = "hi_nerv_representative_region_coverage.v1"

# Size-class boundaries (pixel counts).  Closed-open: small ``[1, 64)``,
# medium ``[64, 4096)``, large ``[4096, +inf)``.  4096 == 64*64; 64 == 8*8.
SIZE_CLASS_SMALL_MAX = 64
SIZE_CLASS_MEDIUM_MAX = 4096
SIZE_CLASSES: tuple[str, ...] = ("small", "medium", "large")

# Outcome vocabulary the coverage builder accepts (mirrors the gate's notion of
# an accepted vs rejected birth on a NAMED first-failing surface).
OUTCOME_BIRTH_ACCEPTED = "birth_accepted"
OUTCOME_BIRTH_REJECTED = "birth_rejected_named_surface"
_VALID_OUTCOMES = frozenset({OUTCOME_BIRTH_ACCEPTED, OUTCOME_BIRTH_REJECTED})

# A planning row must never carry these as truthy — coverage rows embed inside
# substrate_artifact_metadata where the harness custody validator refuses them.
_AUTHORITY_MARKER = "planning_control_false_authority"


class HardRegionMinerError(ValueError):
    """Raised when miner inputs are structurally invalid."""


def size_class_for_pixels(region_pixel_count: int) -> str:
    """Return the canonical size class for a region pixel count."""

    n = int(region_pixel_count)
    if n < 1:
        raise HardRegionMinerError(f"region_pixel_count must be >= 1; got {n}")
    if n < SIZE_CLASS_SMALL_MAX:
        return "small"
    if n < SIZE_CLASS_MEDIUM_MAX:
        return "medium"
    return "large"


@dataclass(frozen=True)
class HardRegion:
    """A ranked target-class region augmented with birth-difficulty signal."""

    rank: int
    debt: TargetRegionDebt
    size_class: str
    margin_mean: float
    margin_min: float
    region_hard_won_pixels: float
    pose_coupling_risk_mean: float | None

    @property
    def batch_index(self) -> int:
        return self.debt.batch_index

    @property
    def class_index(self) -> int:
        return self.debt.class_index

    @property
    def region_label(self) -> int:
        return self.debt.region_label

    @property
    def class_size_key(self) -> tuple[int, str]:
        """The (class, size_class) bucket coverage counts as one representative."""

        return (int(self.debt.class_index), self.size_class)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rank": int(self.rank),
            "batch_index": int(self.debt.batch_index),
            "class_index": int(self.debt.class_index),
            "region_label": int(self.debt.region_label),
            "region_pixel_count": int(self.debt.region_pixel_count),
            "region_unsolved_pixel_count": int(self.debt.region_unsolved_pixel_count),
            "score_debt_units_local": float(self.debt.score_debt_units),
            "normalization_authority": "batch_local_scored_pixels",
            "total_scored_pixels": int(self.debt.total_scored_pixels),
            "frame_fraction": float(self.debt.frame_fraction),
            "bbox_y0": int(self.debt.bbox_y0),
            "bbox_y1": int(self.debt.bbox_y1),
            "bbox_x0": int(self.debt.bbox_x0),
            "bbox_x1": int(self.debt.bbox_x1),
            "size_class": self.size_class,
            "margin_mean": float(self.margin_mean),
            "margin_min": float(self.margin_min),
            "region_hard_won_pixels": float(self.region_hard_won_pixels),
            "pose_coupling_risk_mean": (
                None if self.pose_coupling_risk_mean is None else float(self.pose_coupling_risk_mean)
            ),
            # Authority marker WITHOUT the canonical authority/readiness keys.
            "authority": _AUTHORITY_MARKER,
        }
        return payload


def _validate_bhw(name: str, array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise HardRegionMinerError(f"{name} must be BHW (or BHW1); got shape {np.asarray(array).shape}")
    return arr


def _region_mask_for(
    target_bhw: np.ndarray,
    candidate_bhw: np.ndarray,
    *,
    batch_index: int,
    class_index: int,
    region_label: int,
    expected_pixels: int,
    expected_unsolved_pixels: int,
) -> np.ndarray:
    """Reconstruct the BHW float32 mask for ONE priced region.

    ``find_target_region_debts`` assigns positive-debt ``region_label`` values
    over connected wrong target-class pixels, not over the whole semantic class.
    Reuse that exact debt mask here so the miner, actuator, and parse-back
    receipts share one coordinate system.  Solved diagnostic rows still fall
    back to full class components.
    """

    frame_target = target_bhw[batch_index]
    frame_candidate = candidate_bhw[batch_index]
    if int(expected_unsolved_pixels) > 0:
        label_mask = (frame_target == class_index) & (frame_candidate != class_index)
    else:
        label_mask = frame_target == class_index
    labeled, _count = ndimage.label(label_mask)
    mask = np.zeros(target_bhw.shape, dtype=np.float32)
    mask[batch_index] = (labeled == region_label).astype(np.float32)
    found = int(np.count_nonzero(mask))
    if found != expected_pixels:
        raise HardRegionMinerError(
            "region mask reconstruction drifted from the priced row; "
            f"mask={found} row={expected_pixels} "
            f"(batch={batch_index} class={class_index} region={region_label})"
        )
    return mask


def _round_robin_by_class(
    ranked: Sequence[tuple[float, int, int, int, HardRegion]],
    *,
    top_k: int,
) -> list[HardRegion]:
    """Interleave class-grouped regions so one class cannot fill ``top_k``.

    ``ranked`` is already in global hardness order.  We bucket by
    ``class_index`` (preserving within-class hardness order), then take one
    region from each class per round.  Class buckets are visited in ascending
    ``class_index`` so the result is fully deterministic.
    """

    by_class: dict[int, list[HardRegion]] = defaultdict(list)
    for _key in ranked:
        region = _key[4]
        by_class[int(region.class_index)].append(region)
    ordered_classes = sorted(by_class)
    selected: list[HardRegion] = []
    exhausted = False
    cursor = 0
    while len(selected) < top_k and not exhausted:
        exhausted = True
        for cls in ordered_classes:
            bucket = by_class[cls]
            if cursor < len(bucket):
                exhausted = False
                selected.append(bucket[cursor])
                if len(selected) >= top_k:
                    break
        cursor += 1
    return selected


def mine_hard_regions(
    target_labels: np.ndarray,
    candidate_argmax: np.ndarray,
    logits_bhwc: np.ndarray,
    *,
    top_k: int = 8,
    pose_coupling: np.ndarray | None = None,
    min_region_pixels: int = 1,
    include_solved_regions: bool = False,
) -> list[HardRegion]:
    """Return a deterministic, class-diverse ranking of worst target regions.

    Parameters
    ----------
    target_labels, candidate_argmax:
        BHW (or BHW1) integer label maps — ground-truth target classes and the
        candidate's current argmax respectively.
    logits_bhwc:
        BHWC float logits for the candidate; used for frontier-margin
        difficulty per region (PR95 ``impostor - class`` convention).
    top_k:
        Maximum number of regions to return after diversity enforcement.
    pose_coupling:
        Optional BHW float map of per-pixel pose-Jacobian magnitude (e.g. from
        a differentiable PoseNet).  When present, each region records the mean
        coupling over its pixels as a birth-risk tiebreak/annotation.
    min_region_pixels:
        Drop connected components smaller than this before ranking.
    include_solved_regions:
        When False (default) regions with zero unsolved pixels (already-solved
        or pure-background) are excluded — they carry no birth debt.  When True
        they are kept (useful for diagnostics).

    Ordering
    --------
    Regions are ranked globally by ``(-score_debt_units, -region_hard_won_pixels,
    -|pose_coupling_mean|, batch_index, class_index, region_label)`` then passed
    through round-robin-by-class so no single class can fill ``top_k``.  Within
    a class the hardness order is preserved.  The result's ``rank`` field
    reflects the FINAL diversity-enforced position (0-based).
    """

    if top_k < 1:
        raise HardRegionMinerError(f"top_k must be >= 1; got {top_k}")
    target = _validate_bhw("target_labels", target_labels)
    # ``candidate_argmax`` shape is validated against ``target`` inside
    # ``find_target_region_debts`` (matching-shape check) below; validate it
    # here too for an early, miner-scoped error message.
    candidate = _validate_bhw("candidate_argmax", candidate_argmax)
    logits = np.asarray(logits_bhwc, dtype=np.float64)
    if logits.ndim != 4:
        raise HardRegionMinerError(f"logits_bhwc must be BHWC; got shape {np.asarray(logits_bhwc).shape}")
    if logits.shape[:3] != target.shape:
        raise HardRegionMinerError(
            f"logits BHW must match labels BHW; got logits={logits.shape[:3]} labels={target.shape}"
        )
    pose_map: np.ndarray | None = None
    if pose_coupling is not None:
        pose_map = _validate_bhw("pose_coupling", pose_coupling).astype(np.float64, copy=False)
        if pose_map.shape != target.shape:
            raise HardRegionMinerError(
                f"pose_coupling BHW must match labels BHW; got pose={pose_map.shape} labels={target.shape}"
            )

    debts = find_target_region_debts(
        target_labels,
        candidate_argmax,
        min_region_pixels=min_region_pixels,
    )

    # Build augmented rows (pre-diversity); compute a sort key per region.
    keyed: list[tuple[float, int, int, int, HardRegion]] = []
    for debt in debts:
        if not include_solved_regions and debt.region_unsolved_pixel_count <= 0:
            continue
        mask = _region_mask_for(
            target,
            candidate,
            batch_index=debt.batch_index,
            class_index=debt.class_index,
            region_label=debt.region_label,
            expected_pixels=debt.region_pixel_count,
            expected_unsolved_pixels=debt.region_unsolved_pixel_count,
        )
        stats = region_margin_stats(logits, mask, debt.class_index)
        pose_risk: float | None = None
        if pose_map is not None:
            flat_mask = mask.reshape(-1) > 0.0
            pose_risk = float(np.mean(pose_map.reshape(-1)[flat_mask]))
        region = HardRegion(
            rank=-1,  # assigned after diversity ordering
            debt=debt,
            size_class=size_class_for_pixels(debt.region_pixel_count),
            margin_mean=float(stats["margin_mean"]),
            margin_min=float(stats["margin_min"]),
            region_hard_won_pixels=float(stats["region_hard_won_pixels"]),
            pose_coupling_risk_mean=pose_risk,
        )
        # Carry the priced identity flat; the full difficulty-aware sort key is
        # computed in ``_sort_key`` below so the pose/margin signal stays
        # attached to the region object rather than duplicated here.
        keyed.append(
            (
                -float(debt.score_debt_units),
                int(debt.batch_index),
                int(debt.class_index),
                int(debt.region_label),
                region,
            )
        )

    # Sort globally with full deterministic key including the difficulty signal.
    def _sort_key(item: tuple[float, int, int, int, HardRegion]) -> tuple[float, float, float, int, int, int]:
        neg_debt, b, c, r, region = item
        pose_sort = abs(region.pose_coupling_risk_mean) if region.pose_coupling_risk_mean is not None else 0.0
        return (
            neg_debt,
            -float(region.region_hard_won_pixels),
            -pose_sort,
            b,
            c,
            r,
        )

    keyed.sort(key=_sort_key)
    selected = _round_robin_by_class(keyed, top_k=top_k)
    return [
        HardRegion(
            rank=index,
            debt=region.debt,
            size_class=region.size_class,
            margin_mean=region.margin_mean,
            margin_min=region.margin_min,
            region_hard_won_pixels=region.region_hard_won_pixels,
            pose_coupling_risk_mean=region.pose_coupling_risk_mean,
        )
        for index, region in enumerate(selected)
    ]


def mine_hard_regions_from_margin_map(
    target_labels: np.ndarray,
    candidate_argmax: np.ndarray,
    target_margin_bhw: np.ndarray,
    *,
    top_k: int = 8,
    pose_coupling: np.ndarray | None = None,
    min_region_pixels: int = 1,
    include_solved_regions: bool = False,
) -> list[HardRegion]:
    """Rank hard regions from a compact per-pixel target margin map.

    ``target_margin_bhw`` stores the same PR95 frontier convention used by
    ``region_margin_stats``: ``max_other_logit - target_class_logit`` for the
    target class at each scored pixel.  It is the sufficient statistic for
    region-level margin difficulty, so long smokes can persist ``BHW`` margins
    instead of bulky ``BHWC`` logits while keeping deterministic miner output.
    """

    if top_k < 1:
        raise HardRegionMinerError(f"top_k must be >= 1; got {top_k}")
    target = _validate_bhw("target_labels", target_labels)
    candidate = _validate_bhw("candidate_argmax", candidate_argmax)
    margin_map = _validate_bhw("target_margin_bhw", target_margin_bhw).astype(
        np.float64,
        copy=False,
    )
    if margin_map.shape != target.shape:
        raise HardRegionMinerError(
            f"target_margin_bhw must match labels BHW; got margin={margin_map.shape} labels={target.shape}"
        )
    pose_map: np.ndarray | None = None
    if pose_coupling is not None:
        pose_map = _validate_bhw("pose_coupling", pose_coupling).astype(np.float64, copy=False)
        if pose_map.shape != target.shape:
            raise HardRegionMinerError(
                f"pose_coupling BHW must match labels BHW; got pose={pose_map.shape} labels={target.shape}"
            )

    debts = find_target_region_debts(
        target_labels,
        candidate_argmax,
        min_region_pixels=min_region_pixels,
    )
    keyed: list[tuple[float, int, int, int, HardRegion]] = []
    for debt in debts:
        if not include_solved_regions and debt.region_unsolved_pixel_count <= 0:
            continue
        mask = _region_mask_for(
            target,
            candidate,
            batch_index=debt.batch_index,
            class_index=debt.class_index,
            region_label=debt.region_label,
            expected_pixels=debt.region_pixel_count,
            expected_unsolved_pixels=debt.region_unsolved_pixel_count,
        )
        stats = _margin_map_stats(margin_map, mask)
        pose_risk: float | None = None
        if pose_map is not None:
            flat_mask = mask.reshape(-1) > 0.0
            pose_risk = float(np.mean(pose_map.reshape(-1)[flat_mask]))
        region = HardRegion(
            rank=-1,
            debt=debt,
            size_class=size_class_for_pixels(debt.region_pixel_count),
            margin_mean=float(stats["margin_mean"]),
            margin_min=float(stats["margin_min"]),
            region_hard_won_pixels=float(stats["region_hard_won_pixels"]),
            pose_coupling_risk_mean=pose_risk,
        )
        keyed.append(
            (
                -float(debt.score_debt_units),
                int(debt.batch_index),
                int(debt.class_index),
                int(debt.region_label),
                region,
            )
        )

    def _sort_key(item: tuple[float, int, int, int, HardRegion]) -> tuple[float, float, float, int, int, int]:
        neg_debt, b, c, r, region = item
        pose_sort = abs(region.pose_coupling_risk_mean) if region.pose_coupling_risk_mean is not None else 0.0
        return (
            neg_debt,
            -float(region.region_hard_won_pixels),
            -pose_sort,
            b,
            c,
            r,
        )

    keyed.sort(key=_sort_key)
    selected = _round_robin_by_class(keyed, top_k=top_k)
    return [
        HardRegion(
            rank=index,
            debt=region.debt,
            size_class=region.size_class,
            margin_mean=region.margin_mean,
            margin_min=region.margin_min,
            region_hard_won_pixels=region.region_hard_won_pixels,
            pose_coupling_risk_mean=region.pose_coupling_risk_mean,
        )
        for index, region in enumerate(selected)
    ]


def _margin_map_stats(target_margin_bhw: np.ndarray, region_mask_bhw: np.ndarray) -> dict[str, float]:
    mask = np.asarray(region_mask_bhw).reshape(-1) > 0.0
    if not np.any(mask):
        raise HardRegionMinerError("region mask must contain at least one pixel")
    margin = np.asarray(target_margin_bhw, dtype=np.float64).reshape(-1)[mask]
    if margin.size == 0:
        raise HardRegionMinerError("target margin map produced an empty region")
    hard_won = int(np.count_nonzero(margin < 0.0))
    return {
        "margin_min": float(np.min(margin)),
        "margin_p50": float(np.median(margin)),
        "margin_mean": float(np.mean(margin)),
        "region_hard_won_pixels": float(hard_won),
        "region_hard_ratio": float(hard_won / margin.size),
    }


def build_hard_region_mining_plan(
    regions: Sequence[HardRegion],
    *,
    source: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Wrap ranked regions in a serializable plan envelope (planning-only)."""

    region_rows = [region.as_dict() for region in regions]
    distinct_classes = sorted({int(region.class_index) for region in regions})
    seen: set[tuple[int, str]] = set()
    distinct_class_size: list[list[Any]] = []
    for region in regions:
        key = region.class_size_key
        if key not in seen:
            seen.add(key)
            distinct_class_size.append([int(key[0]), key[1]])
    plan: dict[str, Any] = {
        "schema": HARD_REGION_PLAN_SCHEMA,
        "source": str(source) if source is not None else None,
        "top_k_requested": None if top_k is None else int(top_k),
        "region_count": len(region_rows),
        "distinct_classes": distinct_classes,
        "distinct_class_size_buckets": distinct_class_size,
        "size_class_histogram": _size_class_histogram(regions),
        "regions": region_rows,
        "authority": _AUTHORITY_MARKER,
    }
    return plan


def _size_class_histogram(regions: Sequence[HardRegion]) -> dict[str, int]:
    hist: dict[str, int] = dict.fromkeys(SIZE_CLASSES, 0)
    for region in regions:
        hist[region.size_class] = hist.get(region.size_class, 0) + 1
    return hist


def build_representative_coverage_row(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    min_distinct_classes: int = 2,
    min_distinct_class_size_buckets: int = 3,
) -> dict[str, Any]:
    """Assemble the ``hi_nerv_representative_region_coverage.v1`` row.

    Each outcome is ``{"region": <HardRegion|row dict>, "outcome": <str>,
    "first_failing_surface": <str|None>}``.  ``region_classes_covered`` counts
    the number of DISTINCT ``(class_index, size_class)`` buckets that reached
    ``birth_accepted`` — this is the quantity that proves birth generalizes
    beyond the single birth-at-init case.

    The row is fixture-safe: it carries the planning authority marker and NONE
    of the forbidden authority/readiness keys, so it passes the launch gate's
    ``require_no_truthy_authority_fields`` custody check when embedded in an
    evidence file.
    """

    accepted_buckets: set[tuple[int, str]] = set()
    rejected_buckets: set[tuple[int, str]] = set()
    all_buckets: set[tuple[int, str]] = set()
    accepted_classes: set[int] = set()
    parsed_outcomes: list[dict[str, Any]] = []
    failing_surfaces: list[str] = []

    for index, item in enumerate(outcomes):
        if not isinstance(item, Mapping):
            raise HardRegionMinerError(f"outcome[{index}] must be a mapping; got {type(item).__name__}")
        outcome = str(item.get("outcome") or "")
        if outcome not in _VALID_OUTCOMES:
            raise HardRegionMinerError(
                f"outcome[{index}].outcome must be one of {sorted(_VALID_OUTCOMES)}; got {outcome!r}"
            )
        region = item.get("region")
        class_index, size_class, region_summary = _coverage_region_fields(region, index=index)
        bucket = (class_index, size_class)
        all_buckets.add(bucket)
        first_failing = item.get("first_failing_surface")
        if outcome == OUTCOME_BIRTH_ACCEPTED:
            if first_failing:
                raise HardRegionMinerError(
                    f"outcome[{index}] is birth_accepted but names first_failing_surface={first_failing!r}"
                )
            accepted_buckets.add(bucket)
            accepted_classes.add(class_index)
        else:  # rejected
            if not first_failing:
                raise HardRegionMinerError(
                    f"outcome[{index}] is birth_rejected_named_surface but first_failing_surface is empty"
                )
            rejected_buckets.add(bucket)
            failing_surfaces.append(str(first_failing))
        parsed_outcomes.append(
            {
                "class_index": class_index,
                "size_class": size_class,
                "outcome": outcome,
                "first_failing_surface": (None if not first_failing else str(first_failing)),
                "region": region_summary,
            }
        )

    region_classes_covered = len(accepted_buckets)
    distinct_classes_accepted = len(accepted_classes)
    distinct_size_classes_accepted = len({size for _cls, size in accepted_buckets})
    passed = bool(
        distinct_classes_accepted >= int(min_distinct_classes)
        and region_classes_covered >= int(min_distinct_class_size_buckets)
    )
    row: dict[str, Any] = {
        "schema": REPRESENTATIVE_COVERAGE_SCHEMA,
        "passed": passed,
        "region_classes_covered": int(region_classes_covered),
        "distinct_classes_accepted": int(distinct_classes_accepted),
        "distinct_size_classes_accepted": int(distinct_size_classes_accepted),
        "accepted_class_size_buckets": sorted([list(b) for b in accepted_buckets]),
        "rejected_class_size_buckets": sorted([list(b) for b in rejected_buckets]),
        "all_class_size_buckets": sorted([list(b) for b in all_buckets]),
        "first_failing_surfaces": sorted(set(failing_surfaces)),
        "outcome_count": len(parsed_outcomes),
        "accepted_count": sum(1 for o in parsed_outcomes if o["outcome"] == OUTCOME_BIRTH_ACCEPTED),
        "rejected_count": sum(1 for o in parsed_outcomes if o["outcome"] == OUTCOME_BIRTH_REJECTED),
        "min_distinct_classes": int(min_distinct_classes),
        "min_distinct_class_size_buckets": int(min_distinct_class_size_buckets),
        "outcomes": parsed_outcomes,
        # Authority marker WITHOUT the canonical authority/readiness keys.
        "authority": _AUTHORITY_MARKER,
        "human_visual_fidelity_objective": False,
    }
    return row


def _coverage_region_fields(region: Any, *, index: int) -> tuple[int, str, dict[str, Any]]:
    """Extract (class_index, size_class, summary) from a HardRegion or row dict."""

    if isinstance(region, HardRegion):
        summary = {
            "batch_index": int(region.batch_index),
            "class_index": int(region.class_index),
            "region_label": int(region.region_label),
            "region_pixel_count": int(region.debt.region_pixel_count),
            "size_class": region.size_class,
        }
        return int(region.class_index), region.size_class, summary
    if isinstance(region, Mapping):
        if "class_index" not in region:
            raise HardRegionMinerError(f"outcome[{index}].region dict missing class_index")
        class_index = int(region["class_index"])
        size_class = region.get("size_class")
        if not size_class:
            if "region_pixel_count" not in region:
                raise HardRegionMinerError(
                    f"outcome[{index}].region dict needs size_class or region_pixel_count"
                )
            size_class = size_class_for_pixels(int(region["region_pixel_count"]))
        size_class = str(size_class)
        if size_class not in SIZE_CLASSES:
            raise HardRegionMinerError(
                f"outcome[{index}].region size_class must be one of {SIZE_CLASSES}; got {size_class!r}"
            )
        summary = {
            "class_index": class_index,
            "size_class": size_class,
        }
        for key in ("batch_index", "region_label", "region_pixel_count"):
            if key in region:
                summary[key] = int(region[key])
        return class_index, size_class, summary
    raise HardRegionMinerError(
        f"outcome[{index}].region must be a HardRegion or mapping; got {type(region).__name__}"
    )


__all__ = [
    "HARD_REGION_PLAN_SCHEMA",
    "OUTCOME_BIRTH_ACCEPTED",
    "OUTCOME_BIRTH_REJECTED",
    "REPRESENTATIVE_COVERAGE_SCHEMA",
    "SIZE_CLASSES",
    "HardRegion",
    "HardRegionMinerError",
    "build_hard_region_mining_plan",
    "build_representative_coverage_row",
    "mine_hard_regions",
    "mine_hard_regions_from_margin_map",
    "size_class_for_pixels",
]
