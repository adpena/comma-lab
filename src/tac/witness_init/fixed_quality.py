"""Pure fixed-quality comparison for witness-training initialization arms.

This module turns emitted ``{epoch, d_seg}`` verdict histories into a typed,
hashable measurement receipt.  It deliberately does not interpolate between
verdicts: an arm reaches quality only at the first emitted epoch whose
``d_seg`` is at or below the frozen baseline-derived threshold.

Positive reduction values mean that the treatment used fewer training epochs
or scorer pair-calls.  Negative values are retained when the treatment is
worse.  If either arm misses the threshold within the fixed epoch budget, all
reduction fields are ``None``; the result reports explicit right-censoring
instead of inventing a comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from numbers import Integral, Real
from typing import Literal, TypeAlias

FIXED_QUALITY_SCHEMA = "witness_fixed_quality_comparison_v2"


@dataclass(frozen=True)
class QualityPoint:
    """One emitted realized-through-R quality verdict."""

    epoch: int
    d_seg: float
    elapsed_seconds: float | None = None


QualityHistoryRow: TypeAlias = QualityPoint | Mapping[str, object]


@dataclass(frozen=True)
class FixedQualityThreshold:
    """Frozen threshold and the baseline pretraining verdict that defines it."""

    factor: float
    d_seg: float
    source_epoch: int
    source_d_seg: float


@dataclass(frozen=True)
class ArmFixedQualityOutcome:
    """First emitted crossing, or a fixed-budget right-censoring receipt."""

    arm: str
    observed_start_epoch: int
    observed_end_epoch: int
    emitted_verdict_count: int
    crossed: bool
    crossing_epoch: int | None
    crossing_d_seg: float | None
    training_epochs_to_threshold: int | None
    training_scorer_pair_calls_to_threshold: int | None
    one_time_init_scorer_forward_calls: int
    one_time_init_scorer_pair_equivalents: int
    total_scorer_pair_equivalents_to_threshold: int | None
    training_elapsed_seconds_to_threshold: float | None
    total_wall_seconds_to_threshold: float | None
    right_censored_at_epoch: int | None


@dataclass(frozen=True)
class FixedQualityComparison:
    """Complete or explicitly right-censored fixed-quality A/B comparison."""

    schema: str
    status: Literal["complete", "right_censored"]
    fixed_epoch_budget: int
    scorer_pairs_per_epoch: int
    threshold: FixedQualityThreshold
    baseline: ArmFixedQualityOutcome
    treatment: ArmFixedQualityOutcome
    right_censored_arms: tuple[str, ...]
    epoch_reduction: int | None
    epoch_reduction_fraction: float | None
    training_scorer_pair_call_reduction: int | None
    total_scorer_pair_equivalent_reduction: int | None
    baseline_one_time_init_seconds: float
    treatment_one_time_init_seconds: float
    baseline_measured_total_wall_seconds: float
    treatment_measured_total_wall_seconds: float
    measured_total_wall_seconds_reduction: float | None
    wall_seconds_to_threshold_reduction: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready record without writing any files."""

        return asdict(self)

    def canonical_json(self) -> str:
        """Return deterministic JSON suitable for a durable receipt."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def sha256(self) -> str:
        """Return the SHA-256 of :meth:`canonical_json`."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _finite_nonnegative_seconds(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _quality_point(row: QualityHistoryRow, *, arm: str, index: int) -> QualityPoint:
    if isinstance(row, QualityPoint):
        raw_epoch: object = row.epoch
        raw_d_seg: object = row.d_seg
        raw_elapsed: object = row.elapsed_seconds
    elif isinstance(row, Mapping):
        missing = {"epoch", "d_seg"}.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{arm} history row {index} is missing required field(s): {names}")
        raw_epoch = row["epoch"]
        raw_d_seg = row["d_seg"]
        raw_elapsed = row.get("elapsed_seconds")
    else:
        raise ValueError(
            f"{arm} history row {index} must be QualityPoint or a mapping with epoch and d_seg"
        )

    epoch = _nonnegative_integer(raw_epoch, name=f"{arm} history row {index} epoch")
    if isinstance(raw_d_seg, bool) or not isinstance(raw_d_seg, Real):
        raise ValueError(f"{arm} history row {index} d_seg must be finite and in [0, 1]")
    d_seg = float(raw_d_seg)
    if not math.isfinite(d_seg) or not 0.0 <= d_seg <= 1.0:
        raise ValueError(f"{arm} history row {index} d_seg must be finite and in [0, 1]")
    elapsed = None
    if raw_elapsed is not None:
        elapsed = _finite_nonnegative_seconds(
            raw_elapsed,
            name=f"{arm} history row {index} elapsed_seconds",
        )
    return QualityPoint(epoch=epoch, d_seg=d_seg, elapsed_seconds=elapsed)


def _validated_history(
    history: Iterable[QualityHistoryRow],
    *,
    arm: str,
    fixed_epoch_budget: int,
) -> tuple[QualityPoint, ...]:
    if isinstance(history, Mapping):
        raise ValueError(f"{arm} history must be an iterable of rows, not one mapping")
    try:
        raw_rows = tuple(history)
    except TypeError as exc:
        raise ValueError(f"{arm} history must be a non-empty iterable of rows") from exc
    if not raw_rows:
        raise ValueError(f"{arm} history must not be empty")

    points = tuple(
        _quality_point(row, arm=arm, index=index) for index, row in enumerate(raw_rows)
    )
    if points[0].epoch != 0:
        raise ValueError(
            f"{arm} history must start at epoch 0; first emitted epoch is "
            f"{points[0].epoch}"
        )
    for previous, current in pairwise(points):
        if current.epoch == previous.epoch:
            raise ValueError(f"{arm} history contains duplicate epoch {current.epoch}")
        if current.epoch < previous.epoch:
            raise ValueError(f"{arm} history epochs must be strictly increasing")
        if (
            previous.elapsed_seconds is not None
            and current.elapsed_seconds is not None
            and current.elapsed_seconds < previous.elapsed_seconds
        ):
            raise ValueError(f"{arm} history elapsed_seconds must be non-decreasing")
    if points[-1].epoch > fixed_epoch_budget:
        raise ValueError(
            f"{arm} history extends past fixed_epoch_budget={fixed_epoch_budget}: "
            f"last epoch is {points[-1].epoch}"
        )
    return points


def _first_crossing(
    points: tuple[QualityPoint, ...],
    *,
    arm: str,
    threshold: float,
    fixed_epoch_budget: int,
    scorer_pairs_per_epoch: int,
    init_scorer_forward_calls: int,
    init_scorer_pair_equivalents: int,
    init_seconds: float,
) -> ArmFixedQualityOutcome:
    crossing = next((point for point in points if point.d_seg <= threshold), None)
    if crossing is None and points[-1].epoch != fixed_epoch_budget:
        raise ValueError(
            f"{arm} cannot be declared right-censored before fixed_epoch_budget="
            f"{fixed_epoch_budget}; last emitted epoch is {points[-1].epoch}"
        )

    if crossing is None:
        return ArmFixedQualityOutcome(
            arm=arm,
            observed_start_epoch=points[0].epoch,
            observed_end_epoch=points[-1].epoch,
            emitted_verdict_count=len(points),
            crossed=False,
            crossing_epoch=None,
            crossing_d_seg=None,
            training_epochs_to_threshold=None,
            training_scorer_pair_calls_to_threshold=None,
            one_time_init_scorer_forward_calls=init_scorer_forward_calls,
            one_time_init_scorer_pair_equivalents=init_scorer_pair_equivalents,
            total_scorer_pair_equivalents_to_threshold=None,
            training_elapsed_seconds_to_threshold=None,
            total_wall_seconds_to_threshold=None,
            right_censored_at_epoch=fixed_epoch_budget,
        )

    training_epochs = crossing.epoch - points[0].epoch
    training_pair_calls = scorer_pairs_per_epoch * training_epochs
    training_elapsed = crossing.elapsed_seconds
    if crossing.epoch == points[0].epoch and training_elapsed is None:
        training_elapsed = 0.0
    return ArmFixedQualityOutcome(
        arm=arm,
        observed_start_epoch=points[0].epoch,
        observed_end_epoch=points[-1].epoch,
        emitted_verdict_count=len(points),
        crossed=True,
        crossing_epoch=crossing.epoch,
        crossing_d_seg=crossing.d_seg,
        training_epochs_to_threshold=training_epochs,
        training_scorer_pair_calls_to_threshold=training_pair_calls,
        one_time_init_scorer_forward_calls=init_scorer_forward_calls,
        one_time_init_scorer_pair_equivalents=init_scorer_pair_equivalents,
        total_scorer_pair_equivalents_to_threshold=(
            init_scorer_pair_equivalents + training_pair_calls
        ),
        training_elapsed_seconds_to_threshold=training_elapsed,
        total_wall_seconds_to_threshold=(
            None if training_elapsed is None else init_seconds + training_elapsed
        ),
        right_censored_at_epoch=None,
    )


def compare_fixed_quality(
    baseline_history: Iterable[QualityHistoryRow],
    treatment_history: Iterable[QualityHistoryRow],
    *,
    fixed_epoch_budget: int,
    scorer_pairs_per_epoch: int,
    baseline_one_time_init_seconds: float,
    treatment_one_time_init_seconds: float,
    baseline_measured_total_wall_seconds: float,
    treatment_measured_total_wall_seconds: float,
    baseline_init_scorer_forward_calls: int = 0,
    treatment_init_scorer_forward_calls: int = 0,
    baseline_init_scorer_pair_equivalents: int = 0,
    treatment_init_scorer_pair_equivalents: int = 0,
    threshold_factor: float = 0.90,
    baseline_arm: str = "baseline",
    treatment_arm: str = "treatment",
) -> FixedQualityComparison:
    """Compare first emitted epochs reaching a baseline-derived fixed quality.

    The threshold is frozen as ``threshold_factor`` times the baseline's
    lowest-epoch (pretraining) ``d_seg``.  Histories must already be sorted and
    must use the same starting epoch.  An arm that does not cross must include
    an emitted verdict exactly at ``fixed_epoch_budget`` so right-censoring is
    evidence-backed.  A crossing arm may stop early, including at epoch zero.

    ``epoch_reduction`` and ``training_scorer_pair_call_reduction`` are signed:
    positive is faster, zero is tied, and negative is worse.  They are emitted
    only when both arms cross.  The proportional epoch reduction is ``None``
    when the baseline crosses at its initial epoch because division by zero has
    no honest interpretation.
    """

    budget = _nonnegative_integer(fixed_epoch_budget, name="fixed_epoch_budget")
    pairs = _positive_integer(scorer_pairs_per_epoch, name="scorer_pairs_per_epoch")
    if isinstance(threshold_factor, bool) or not isinstance(threshold_factor, Real):
        raise ValueError("threshold_factor must be finite and strictly between 0 and 1")
    factor = float(threshold_factor)
    if not math.isfinite(factor) or not 0.0 < factor < 1.0:
        raise ValueError("threshold_factor must be finite and strictly between 0 and 1")
    if not isinstance(baseline_arm, str) or not baseline_arm.strip():
        raise ValueError("baseline_arm must be a non-empty string")
    if not isinstance(treatment_arm, str) or not treatment_arm.strip():
        raise ValueError("treatment_arm must be a non-empty string")
    if baseline_arm == treatment_arm:
        raise ValueError("baseline_arm and treatment_arm must be distinct")

    baseline_points = _validated_history(
        baseline_history,
        arm=baseline_arm,
        fixed_epoch_budget=budget,
    )
    treatment_points = _validated_history(
        treatment_history,
        arm=treatment_arm,
        fixed_epoch_budget=budget,
    )
    # _validated_history independently requires the real pretraining verdict
    # at epoch zero for each arm.  Do not normalize a resumed/nonzero history
    # into an apparent cold-start convergence comparison.

    baseline_init = _finite_nonnegative_seconds(
        baseline_one_time_init_seconds,
        name="baseline_one_time_init_seconds",
    )
    treatment_init = _finite_nonnegative_seconds(
        treatment_one_time_init_seconds,
        name="treatment_one_time_init_seconds",
    )
    baseline_wall = _finite_nonnegative_seconds(
        baseline_measured_total_wall_seconds,
        name="baseline_measured_total_wall_seconds",
    )
    treatment_wall = _finite_nonnegative_seconds(
        treatment_measured_total_wall_seconds,
        name="treatment_measured_total_wall_seconds",
    )
    baseline_init_forwards = _nonnegative_integer(
        baseline_init_scorer_forward_calls,
        name="baseline_init_scorer_forward_calls",
    )
    treatment_init_forwards = _nonnegative_integer(
        treatment_init_scorer_forward_calls,
        name="treatment_init_scorer_forward_calls",
    )
    baseline_init_pairs = _nonnegative_integer(
        baseline_init_scorer_pair_equivalents,
        name="baseline_init_scorer_pair_equivalents",
    )
    treatment_init_pairs = _nonnegative_integer(
        treatment_init_scorer_pair_equivalents,
        name="treatment_init_scorer_pair_equivalents",
    )

    source = baseline_points[0]
    threshold = FixedQualityThreshold(
        factor=factor,
        d_seg=factor * source.d_seg,
        source_epoch=source.epoch,
        source_d_seg=source.d_seg,
    )
    baseline_outcome = _first_crossing(
        baseline_points,
        arm=baseline_arm,
        threshold=threshold.d_seg,
        fixed_epoch_budget=budget,
        scorer_pairs_per_epoch=pairs,
        init_scorer_forward_calls=baseline_init_forwards,
        init_scorer_pair_equivalents=baseline_init_pairs,
        init_seconds=baseline_init,
    )
    treatment_outcome = _first_crossing(
        treatment_points,
        arm=treatment_arm,
        threshold=threshold.d_seg,
        fixed_epoch_budget=budget,
        scorer_pairs_per_epoch=pairs,
        init_scorer_forward_calls=treatment_init_forwards,
        init_scorer_pair_equivalents=treatment_init_pairs,
        init_seconds=treatment_init,
    )

    censored_arms = tuple(
        outcome.arm for outcome in (baseline_outcome, treatment_outcome) if not outcome.crossed
    )
    if censored_arms:
        return FixedQualityComparison(
            schema=FIXED_QUALITY_SCHEMA,
            status="right_censored",
            fixed_epoch_budget=budget,
            scorer_pairs_per_epoch=pairs,
            threshold=threshold,
            baseline=baseline_outcome,
            treatment=treatment_outcome,
            right_censored_arms=censored_arms,
            epoch_reduction=None,
            epoch_reduction_fraction=None,
            training_scorer_pair_call_reduction=None,
            total_scorer_pair_equivalent_reduction=None,
            baseline_one_time_init_seconds=baseline_init,
            treatment_one_time_init_seconds=treatment_init,
            baseline_measured_total_wall_seconds=baseline_wall,
            treatment_measured_total_wall_seconds=treatment_wall,
            measured_total_wall_seconds_reduction=None,
            wall_seconds_to_threshold_reduction=None,
        )

    baseline_epochs = baseline_outcome.training_epochs_to_threshold
    treatment_epochs = treatment_outcome.training_epochs_to_threshold
    assert baseline_epochs is not None and treatment_epochs is not None
    epoch_reduction = baseline_epochs - treatment_epochs
    epoch_fraction = epoch_reduction / baseline_epochs if baseline_epochs > 0 else None
    baseline_total_pairs = baseline_outcome.total_scorer_pair_equivalents_to_threshold
    treatment_total_pairs = treatment_outcome.total_scorer_pair_equivalents_to_threshold
    assert baseline_total_pairs is not None and treatment_total_pairs is not None
    baseline_wall_to_quality = baseline_outcome.total_wall_seconds_to_threshold
    treatment_wall_to_quality = treatment_outcome.total_wall_seconds_to_threshold
    return FixedQualityComparison(
        schema=FIXED_QUALITY_SCHEMA,
        status="complete",
        fixed_epoch_budget=budget,
        scorer_pairs_per_epoch=pairs,
        threshold=threshold,
        baseline=baseline_outcome,
        treatment=treatment_outcome,
        right_censored_arms=(),
        epoch_reduction=epoch_reduction,
        epoch_reduction_fraction=epoch_fraction,
        training_scorer_pair_call_reduction=pairs * epoch_reduction,
        total_scorer_pair_equivalent_reduction=(
            baseline_total_pairs - treatment_total_pairs
        ),
        baseline_one_time_init_seconds=baseline_init,
        treatment_one_time_init_seconds=treatment_init,
        baseline_measured_total_wall_seconds=baseline_wall,
        treatment_measured_total_wall_seconds=treatment_wall,
        measured_total_wall_seconds_reduction=baseline_wall - treatment_wall,
        wall_seconds_to_threshold_reduction=(
            None
            if baseline_wall_to_quality is None or treatment_wall_to_quality is None
            else baseline_wall_to_quality - treatment_wall_to_quality
        ),
    )
