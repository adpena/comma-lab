# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "probe_regmax_family", ROOT / "tools/probe_regmax_family.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sparsemax_projects_to_simplex_and_obeys_unit_margin_one_hot_law() -> None:
    logits = np.array(
        [
            [2.0, 1.0, -3.0, -4.0, -5.0],
            [1.9, 1.0, -3.0, -4.0, -5.0],
        ]
    )
    projected = MODULE.sparsemax(logits)
    np.testing.assert_allclose(projected.sum(axis=-1), 1.0, atol=1e-12)
    np.testing.assert_array_equal(np.count_nonzero(projected > 0.0, axis=-1), [1, 2])
    np.testing.assert_allclose(projected[0], [1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_prediction_surface_reports_class_and_margin_strata() -> None:
    logits = np.array(
        [
            [2.0, 1.0, -3.0, -4.0, -5.0],
            [1.9, 1.0, -3.0, -4.0, -5.0],
            [-2.0, 3.0, 0.0, -4.0, -5.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 1], dtype=np.uint8)
    result = MODULE.measure_prediction_surface(logits, labels)
    assert result["overall"]["pixels"] == 3
    assert result["per_stratum"]["high_margin_interior_ge_1"]["pixels"] == 2
    assert result["per_stratum"]["boundary_annulus_or_tie_lt_1"]["pixels"] == 1
    assert result["overall"]["sparsemax_exact_one_hot_fraction"] == pytest.approx(2 / 3)
    assert result["fp16_argmax_vs_hard_label_mismatches"] == 1
    assert set(result["per_class"]) == {"0", "1", "2", "3", "4"}


def test_invalid_sparsemax_inputs_fail_closed() -> None:
    with pytest.raises(MODULE.ProbeError):
        MODULE.sparsemax(np.array([[np.nan, 0.0]]))
    with pytest.raises(MODULE.ProbeError):
        MODULE.sparsemax(np.array([[1.0, 0.0]]), scale=0.0)


def test_na_receipt_preserves_preregistered_falsifier_without_fake_counts() -> None:
    receipt = MODULE._n_a_receipt(
        probe_id=MODULE.HOPFIELD_PROBE,
        falsifier=MODULE.HOPFIELD_FALSIFIER,
        blockers=["missing fixture"],
        common={"schema": "test"},
    )
    assert receipt["verdict"] == "N-A"
    assert receipt["falsifier_evaluated"] is False
    assert receipt["hard_cpu_torch_oracle"]["invoked"] is False
    assert receipt["hard_accepts"] is None
    assert receipt["exact_oracle_calls"] is None
    assert receipt["candidate_bytes_same_coder"] is None
    assert "does not improve" in receipt["falsifier"]
