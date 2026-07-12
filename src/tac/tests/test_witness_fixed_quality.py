"""Tests for the fixed-quality witness-training A/B receipt."""

from __future__ import annotations

import json

import pytest

from tac.witness_init.fixed_quality import FIXED_QUALITY_SCHEMA, compare_fixed_quality


def _compare(
    baseline: list[dict[str, float | int]],
    treatment: list[dict[str, float | int]],
    **overrides: object,
):
    kwargs: dict[str, object] = {
        "fixed_epoch_budget": 3,
        "scorer_pairs_per_epoch": 8,
        "baseline_one_time_init_seconds": 0.25,
        "treatment_one_time_init_seconds": 1.75,
        "baseline_measured_total_wall_seconds": 12.0,
        "treatment_measured_total_wall_seconds": 9.0,
    }
    kwargs.update(overrides)
    return compare_fixed_quality(baseline, treatment, **kwargs)


def test_freezes_baseline_pretraining_threshold_and_uses_first_emitted_crossing() -> None:
    result = _compare(
        [
            {"epoch": 0, "d_seg": 0.10},
            {"epoch": 1, "d_seg": 0.091},
            {"epoch": 2, "d_seg": 0.089},
        ],
        [
            {"epoch": 0, "d_seg": 0.12},
            {"epoch": 1, "d_seg": 0.0895},
        ],
    )

    assert result.schema == FIXED_QUALITY_SCHEMA
    assert result.status == "complete"
    assert result.threshold.factor == 0.90
    assert result.threshold.source_epoch == 0
    assert result.threshold.source_d_seg == 0.10
    assert result.threshold.d_seg == pytest.approx(0.09)
    assert result.baseline.crossing_epoch == 2
    assert result.baseline.crossing_d_seg == 0.089
    assert result.treatment.crossing_epoch == 1
    assert result.epoch_reduction == 1
    assert result.epoch_reduction_fraction == 0.5
    assert result.training_scorer_pair_call_reduction == 8
    assert result.baseline.training_scorer_pair_calls_to_threshold == 16
    assert result.treatment.training_scorer_pair_calls_to_threshold == 8
    assert result.baseline_one_time_init_seconds == 0.25
    assert result.treatment_one_time_init_seconds == 1.75
    assert result.baseline_measured_total_wall_seconds == 12.0
    assert result.treatment_measured_total_wall_seconds == 9.0
    assert result.measured_total_wall_seconds_reduction == 3.0


def test_treatment_epoch_zero_crossing_counts_zero_training_scorer_calls() -> None:
    result = _compare(
        [
            {"epoch": 0, "d_seg": 0.10},
            {"epoch": 1, "d_seg": 0.095},
            {"epoch": 2, "d_seg": 0.09},
        ],
        [{"epoch": 0, "d_seg": 0.08}],
    )

    assert result.status == "complete"
    assert result.treatment.crossing_epoch == 0
    assert result.treatment.training_epochs_to_threshold == 0
    assert result.treatment.training_scorer_pair_calls_to_threshold == 0
    assert result.epoch_reduction == 2
    assert result.epoch_reduction_fraction == 1.0
    assert result.training_scorer_pair_call_reduction == 16


def test_signed_reductions_retain_a_slower_treatment() -> None:
    result = _compare(
        [
            {"epoch": 0, "d_seg": 0.10},
            {"epoch": 1, "d_seg": 0.089},
        ],
        [
            {"epoch": 0, "d_seg": 0.11},
            {"epoch": 1, "d_seg": 0.10},
            {"epoch": 2, "d_seg": 0.095},
            {"epoch": 3, "d_seg": 0.089},
        ],
        scorer_pairs_per_epoch=4,
    )

    assert result.status == "complete"
    assert result.epoch_reduction == -2
    assert result.epoch_reduction_fraction == -2.0
    assert result.training_scorer_pair_call_reduction == -8


def test_right_censoring_never_fabricates_any_reduction() -> None:
    result = _compare(
        [
            {"epoch": 0, "d_seg": 0.10},
            {"epoch": 2, "d_seg": 0.089},
        ],
        [
            {"epoch": 0, "d_seg": 0.11},
            {"epoch": 1, "d_seg": 0.105},
            {"epoch": 3, "d_seg": 0.10},
        ],
    )

    assert result.status == "right_censored"
    assert result.right_censored_arms == ("treatment",)
    assert result.baseline.crossed is True
    assert result.treatment.crossed is False
    assert result.treatment.right_censored_at_epoch == 3
    assert result.epoch_reduction is None
    assert result.epoch_reduction_fraction is None
    assert result.training_scorer_pair_call_reduction is None
    assert result.measured_total_wall_seconds_reduction is None
    # Raw timing evidence remains present under censoring.
    assert result.baseline_one_time_init_seconds == 0.25
    assert result.treatment_one_time_init_seconds == 1.75
    assert result.baseline_measured_total_wall_seconds == 12.0
    assert result.treatment_measured_total_wall_seconds == 9.0


@pytest.mark.parametrize(
    ("baseline", "match"),
    [
        (
            [{"epoch": 0, "d_seg": 0.1}, {"epoch": 2, "d_seg": 0.09}, {"epoch": 1, "d_seg": 0.08}],
            "strictly increasing",
        ),
        (
            [{"epoch": 0, "d_seg": 0.1}, {"epoch": 1, "d_seg": 0.09}, {"epoch": 1, "d_seg": 0.08}],
            "duplicate epoch 1",
        ),
        ([{"epoch": 0, "d_seg": float("nan")}], "finite and in"),
        ([{"epoch": 0, "d_seg": float("inf")}], "finite and in"),
    ],
)
def test_rejects_unsorted_duplicate_and_nonfinite_histories(
    baseline: list[dict[str, float | int]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _compare(baseline, [{"epoch": 0, "d_seg": 0.08}])


def test_refuses_to_call_an_incomplete_history_right_censored() -> None:
    with pytest.raises(ValueError, match="before fixed_epoch_budget"):
        _compare(
            [
                {"epoch": 0, "d_seg": 0.10},
                {"epoch": 1, "d_seg": 0.095},
            ],
            [{"epoch": 0, "d_seg": 0.08}],
        )


def test_each_history_must_start_at_epoch_zero() -> None:
    with pytest.raises(ValueError, match="must start at epoch 0"):
        _compare(
            [{"epoch": 1, "d_seg": 0.10}],
            [{"epoch": 0, "d_seg": 0.08}],
        )


@pytest.mark.parametrize("factor", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")])
def test_threshold_factor_must_be_strictly_inside_unit_interval(factor: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        _compare(
            [{"epoch": 0, "d_seg": 0.10}],
            [{"epoch": 0, "d_seg": 0.08}],
            fixed_epoch_budget=0,
            threshold_factor=factor,
        )


def test_canonical_json_and_hash_are_stable_and_json_roundtrippable() -> None:
    result = _compare(
        [
            {"epoch": 0, "d_seg": 0.10},
            {"epoch": 1, "d_seg": 0.089},
        ],
        [
            {"epoch": 0, "d_seg": 0.11},
            {"epoch": 1, "d_seg": 0.088},
        ],
    )

    payload = json.loads(result.canonical_json())
    assert payload["schema"] == FIXED_QUALITY_SCHEMA
    assert payload["threshold"]["source_epoch"] == 0
    assert result.sha256() == result.sha256()
    assert len(result.sha256()) == 64


def test_init_scorer_overhead_can_erase_a_one_epoch_training_win() -> None:
    result = _compare(
        [
            {"epoch": 0, "d_seg": 0.10},
            {"epoch": 1, "d_seg": 0.095},
            {"epoch": 2, "d_seg": 0.089},
        ],
        [
            {"epoch": 0, "d_seg": 0.11},
            {"epoch": 1, "d_seg": 0.089},
        ],
        baseline_init_scorer_forward_calls=1,
        treatment_init_scorer_forward_calls=93,
        baseline_init_scorer_pair_equivalents=1,
        treatment_init_scorer_pair_equivalents=93,
    )

    assert result.epoch_reduction == 1
    assert result.training_scorer_pair_call_reduction == 8
    assert result.baseline.total_scorer_pair_equivalents_to_threshold == 17
    assert result.treatment.total_scorer_pair_equivalents_to_threshold == 101
    assert result.total_scorer_pair_equivalent_reduction == -84
    assert result.treatment.one_time_init_scorer_forward_calls == 93


def test_wall_to_quality_uses_crossing_timestamp_not_final_run_wall() -> None:
    result = _compare(
        [
            {"epoch": 0, "d_seg": 0.10, "elapsed_seconds": 0.0},
            {"epoch": 1, "d_seg": 0.095, "elapsed_seconds": 4.0},
            {"epoch": 2, "d_seg": 0.089, "elapsed_seconds": 8.0},
        ],
        [
            {"epoch": 0, "d_seg": 0.11, "elapsed_seconds": 0.0},
            {"epoch": 1, "d_seg": 0.089, "elapsed_seconds": 5.0},
        ],
    )

    assert result.baseline.total_wall_seconds_to_threshold == pytest.approx(8.25)
    assert result.treatment.total_wall_seconds_to_threshold == pytest.approx(6.75)
    assert result.wall_seconds_to_threshold_reduction == pytest.approx(1.5)
    # These are independent raw final-run timings and are not reused as crossing time.
    assert result.measured_total_wall_seconds_reduction == pytest.approx(3.0)


def test_elapsed_seconds_must_be_monotone_when_present() -> None:
    with pytest.raises(ValueError, match="elapsed_seconds must be non-decreasing"):
        _compare(
            [
                {"epoch": 0, "d_seg": 0.10, "elapsed_seconds": 4.0},
                {"epoch": 1, "d_seg": 0.089, "elapsed_seconds": 3.0},
            ],
            [{"epoch": 0, "d_seg": 0.08, "elapsed_seconds": 0.0}],
        )
