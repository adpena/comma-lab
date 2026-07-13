import pytest

from tac.canonical_equations.amdahl_measured_wall_split_20260713 import (
    AmdahlWallSplit,
    MeasuredLever,
    MeasuredSeconds,
)


def ms(value, source="test-artifact.json"):
    return MeasuredSeconds(value, source)


def test_composes_measured_lever_and_separate_async_cpu_share():
    result = AmdahlWallSplit(
        baseline_seconds=ms(100),
        levers=(MeasuredLever("halo", "scorer_forward", ms(50), ms(25)),),
        async_cpu_verdict_service_seconds=ms(20),
        async_cpu_verdict_critical_path_seconds=ms(0),
    ).compose()
    assert result["composed_seconds"] == pytest.approx(75.0)
    assert result["async_cpu_verdict_service_to_training_wall_ratio"] == pytest.approx(0.2)
    assert result["async_cpu_verdict_critical_path_share"] == pytest.approx(0.0)
    assert result["inputs_status"] == "MEASURED_ONLY_DISJOINT_COMPONENTS"


def test_rejects_unmeasured_or_invalid_inputs():
    with pytest.raises(ValueError, match="status=MEASURED"):
        MeasuredSeconds(1, "x", status="DERIVED")
    with pytest.raises(ValueError, match="source_artifact"):
        MeasuredSeconds(1, "")
    with pytest.raises(ValueError, match="exceed"):
        MeasuredLever("bad", "component", ms(1), ms(2))


def test_rejects_overlapping_levers_instead_of_multiplying_guesses():
    with pytest.raises(ValueError, match="overlapping levers jointly"):
        AmdahlWallSplit(
            baseline_seconds=ms(100),
            levers=(
                MeasuredLever("tile", "scorer_forward", ms(40), ms(20)),
                MeasuredLever("fp16", "scorer_forward", ms(40), ms(25)),
            ),
        )


def test_disjoint_component_savings_are_additive():
    result = AmdahlWallSplit(
        baseline_seconds=ms(100),
        levers=(
            MeasuredLever("forward", "scorer_forward", ms(40), ms(20)),
            MeasuredLever("render", "render", ms(10), ms(5)),
        ),
    ).compose()
    assert result["composed_seconds"] == pytest.approx(75.0)
    assert result["composed_speedup"] == pytest.approx(4.0 / 3.0)
