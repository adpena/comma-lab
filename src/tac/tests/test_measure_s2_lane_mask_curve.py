# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "tools" / "measure_s2_lane_mask_curve.py"
SPEC = importlib.util.spec_from_file_location("measure_s2_lane_mask_curve", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
measure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure)


def test_confusion_and_metrics_are_exact() -> None:
    truth = np.array([[True, True], [False, False]])
    predicted = np.array([[True, False], [True, False]])
    counts = measure._confusion(predicted, truth)
    assert counts == (1, 1, 1, 1)
    metrics = measure._metrics(counts)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_boundary_marks_both_sides_of_transition() -> None:
    mask = np.array([[False, False, True], [False, False, True]])
    expected = np.array([[False, True, True], [False, True, True]])
    assert np.array_equal(measure._boundary(mask), expected)


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "atom_prefixes": [2, 4],
        "phase_bins": 8,
        "qstep": 1.0 / 64.0,
        "correction_threshold": 0.5,
        "max_per_residual_sign": 32,
        "max_zero_samples": 32,
        "chunk_pairs": 25,
        "minimum_stage_free_bytes": 512 << 20,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("atom_prefixes", [1], "atom prefixes"),
        ("phase_bins", 0, "phase bins"),
        ("qstep", float("nan"), "qstep"),
        ("correction_threshold", 0.0, "correction threshold"),
        ("max_zero_samples", -1, "sample caps"),
        ("chunk_pairs", 0, "chunk pairs"),
        ("minimum_stage_free_bytes", -1, "free bytes"),
    ],
)
def test_measurement_arguments_fail_closed(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        measure._validate_args(_args(**{field: value}))
