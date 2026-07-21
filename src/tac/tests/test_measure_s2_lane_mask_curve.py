# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

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
