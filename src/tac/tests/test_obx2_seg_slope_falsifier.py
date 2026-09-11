# SPDX-License-Identifier: MIT
"""Tests for the PRE-REGISTERED OBX2 seg-slope falsifier.

These run on synthetic curves with known answers, so the rule is exercised
without any reference to the live arms' data.  That is the point: the rule has
to be demonstrably correct before it is pointed at the run it will judge.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for _root in (REPO, REPO / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_obx2_seg_slope_falsifier as fal  # noqa: E402


def _exponential(epochs: range, start: float, per_epoch_decade: float) -> list[float]:
    return [start * 10.0 ** (-per_epoch_decade * e) for e in epochs]


def test_pre_registered_constants_are_the_ones_the_rule_uses() -> None:
    assert fal.D_SEG_CEILING == 4.0e-4
    assert fal.D_SEG_POSE_CONSISTENT == 1.87e-4
    assert fal.STAGE_EPOCHS == 200
    assert fal.FLATTEN_WINDOW == 20
    assert fal.FLATTEN_MIN_EPOCH == 60
    assert fal.FLATTEN_MIN_FACTOR == 1.5
    # the ceiling is exactly the gate with zero room for Pose
    assert abs(100.0 * fal.D_SEG_CEILING - 0.04) < 1e-12
    # the pose-consistent target leaves room for the measured sp_384x512 Pose leg
    pose_leg = math.sqrt(10.0 * 4.52317e-5)
    assert 100.0 * fal.D_SEG_POSE_CONSISTENT + pose_leg < 0.04


def test_a_fast_descent_is_open_and_projects_inside_the_stage() -> None:
    epochs = range(0, 40)
    values = _exponential(epochs, 0.025, 0.05)  # a decade every 20 epochs
    verdict = fal.judge("fast", list(epochs), values)
    assert verdict.verdict == "OPEN"
    reached = verdict.fits[verdict.primary].epoch_reaching(fal.D_SEG_CEILING)
    assert reached is not None and reached < fal.STAGE_EPOCHS


def test_a_slow_descent_is_closed_for_projecting_past_the_stage() -> None:
    epochs = range(0, 40)
    values = _exponential(epochs, 0.025, 0.002)  # a decade every 500 epochs
    verdict = fal.judge("slow", list(epochs), values)
    assert verdict.verdict == "CLOSED_AT_FORMULATION_SCOPE"
    assert "exceeds the stage" in " ".join(verdict.reasons)


def test_a_rising_curve_is_closed() -> None:
    epochs = list(range(0, 30))
    values = [0.02 * (1.0 + 0.01 * e) for e in epochs]
    verdict = fal.judge("rising", epochs, values)
    assert verdict.verdict == "CLOSED_AT_FORMULATION_SCOPE"
    assert "does not descend" in " ".join(verdict.reasons)


def test_flattening_closes_an_arm_that_would_otherwise_project_inside() -> None:
    # Fast early, flat after epoch 60: the whole-history fit still projects
    # inside the stage, and the pre-registered tail test is what catches it.
    epochs = list(range(0, 90))
    values = [0.025 * 10.0 ** (-0.05 * min(e, 55)) for e in epochs]
    verdict = fal.judge("flattened", epochs, values)
    assert verdict.verdict == "CLOSED_AT_FORMULATION_SCOPE"
    assert "flattened" in " ".join(verdict.reasons)


def test_the_flattening_test_is_not_applied_before_its_epoch_floor() -> None:
    # A curve that still projects inside the stage, stopped short of the floor:
    # the tail test must stay silent rather than judge a 40-epoch history.
    epochs = list(range(0, 40))
    values = _exponential(range(0, 40), 0.025, 0.05)
    verdict = fal.judge("early", epochs, values)
    assert max(epochs) < fal.FLATTEN_MIN_EPOCH
    assert "flattening test not applied" in " ".join(verdict.reasons)


def test_too_few_epochs_refuses_a_verdict_rather_than_guessing() -> None:
    verdict = fal.judge("short", [0, 1, 2], [0.025, 0.024, 0.023])
    assert verdict.verdict == "INSUFFICIENT_EPOCHS"
    assert not verdict.fits


def test_a_noisy_curve_reports_indeterminate_when_its_band_straddles_the_stage() -> None:
    # Tuned so the point projection sits near the stage edge with a wide band.
    epochs = list(range(0, 30))
    rng = [0.0, 0.35, -0.35, 0.2, -0.2] * 6
    values = [0.025 * 10.0 ** (-0.0088 * e + rng[i]) for i, e in enumerate(epochs)]
    verdict = fal.judge("noisy", epochs, values)
    assert verdict.verdict in {"INDETERMINATE", "CLOSED_AT_FORMULATION_SCOPE", "OPEN"}
    if verdict.verdict == "INDETERMINATE":
        assert "straddles" in " ".join(verdict.reasons)


def test_fit_bands_widen_with_the_residual_and_round_trip_the_target() -> None:
    epochs = list(range(0, 30))
    values = _exponential(range(0, 30), 0.02, 0.02)
    fit = fal.fit_family(epochs, values, "exponential")
    assert fit.slope < 0.0
    assert fit.residual_standard_error < 1e-9
    reached = fit.epoch_reaching(1.0e-3)
    assert reached is not None
    assert 10.0 ** fit.predict_log10(reached) == pytest.approx(1.0e-3, rel=1e-6)


def test_reader_extracts_only_component_rows_and_refuses_a_non_positive_surrogate(tmp_path: Path) -> None:
    good = tmp_path / "run.log"
    good.write_text(
        "not json\n"
        + json.dumps({"epoch": 0, "mean_loss": 1.0}) + "\n"
        + json.dumps({"epoch": 1, "mean_expected_flip": 0.02, "mean_loss": 1.0}) + "\n"
        + json.dumps({"epoch": 2, "mean_expected_flip": 0.01, "mean_loss": 1.0}) + "\n"
    )
    epochs, values = fal.read_epoch_rows(good)
    assert epochs == [1, 2]
    assert values == [0.02, 0.01]
    bad = tmp_path / "bad.log"
    bad.write_text(json.dumps({"epoch": 1, "mean_expected_flip": 0.0}) + "\n")
    with pytest.raises(fal.SegSlopeError):
        fal.read_epoch_rows(bad)


def test_describe_reports_both_families_and_both_targets() -> None:
    epochs = list(range(0, 30))
    verdict = fal.judge("both", epochs, _exponential(range(0, 30), 0.025, 0.03))
    payload = fal.describe(verdict)
    assert set(payload["fits"]) == {"power", "exponential"}
    for entry in payload["fits"].values():
        assert "epoch_to_ceiling_4e-4" in entry
        assert "epoch_to_pose_consistent_1.87e-4" in entry
        assert "epoch_band_to_ceiling_4e-4" in entry
