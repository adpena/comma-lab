# SPDX-License-Identifier: MIT
"""Pure-NumPy contract tests for the Pythagorean exact-arithmetic MLX probe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "pythagorean_exact_probe",
    ROOT / "tools/probe_pythagorean_exact_arithmetic_bitident.py",
)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_bicubic_fixture_is_actual_render_r_axis() -> None:
    indices, weights = PROBE.bicubic_indices_weights_numpy()
    assert indices.shape == (874, 4)
    assert weights.shape == (874, 4)
    assert indices.dtype == np.int32
    assert weights.dtype == np.float32
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=2.0e-6)


def test_q15_integer_accumulator_is_overflow_safe_and_deterministic() -> None:
    first = PROBE.build_resize_adjoint_fixture()
    second = PROBE.build_resize_adjoint_fixture()
    assert int(first["max_abs_integer_accumulator"]) < np.iinfo(np.int32).max
    assert int(first["max_contributions_per_destination"]) > 1
    np.testing.assert_array_equal(first["destination_u32"], second["destination_u32"])
    np.testing.assert_array_equal(first["int_reference"], second["int_reference"])
    np.testing.assert_array_equal(first["float_reference"], second["float_reference"])


def test_q15_dequantization_obeys_derived_numpy_fp32_tolerance() -> None:
    fixture = PROBE.build_resize_adjoint_fixture()
    dequantized = np.asarray(fixture["int_reference"], dtype=np.int32).astype(np.float64) / float(PROBE.Q_SCALE)
    reference = np.asarray(fixture["float_reference"], dtype=np.float32).astype(np.float64)
    max_abs_error = float(np.max(np.abs(dequantized - reference)))
    assert max_abs_error <= float(fixture["authority_tolerance"])


def test_partial_receipt_never_emits_a_scientific_verdict() -> None:
    receipt = PROBE._base_receipt(PROBE.N_PROCESSES)
    summary = PROBE._summarize(receipt)
    assert summary["complete"] is False
    assert summary["decisive_positive"] is False
    assert summary["overall_verdict"] == "INCOMPLETE"


def test_resume_refuses_changed_probe_bytes(tmp_path: Path) -> None:
    receipt = PROBE._base_receipt(PROBE.N_PROCESSES)
    receipt["source_custody"]["probe"]["sha256"] = "0" * 64
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="probe bytes differ"):
        PROBE.run_parent(output=path, n=PROBE.N_PROCESSES, resume=True)


@pytest.mark.parametrize(
    ("float_hashes", "integer_hashes", "expected"),
    [
        ([f"f{i}" for i in range(10)], ["i"] * 10, "REAL-L70-LEVER"),
        ([f"f{i}" for i in range(10)], ["i0"] * 5 + ["i1"] * 5, "L70-DEEPER-THAN-FP-REORDER"),
        (["f"] * 10, ["i"] * 10, "INERT-CURIO"),
    ],
)
def test_complete_receipt_taxonomy(float_hashes: list[str], integer_hashes: list[str], expected: str) -> None:
    receipt = PROBE._base_receipt(PROBE.N_PROCESSES)
    receipt["trials"]["float_atomic"] = [
        {
            "output_sha256": value,
            "parity": {"within_derived_tolerance": True},
        }
        for value in float_hashes
    ]
    receipt["trials"]["fixed_q15_int32_atomic"] = [
        {
            "output_sha256": value,
            "parity": {
                "integer_bit_identical": True,
                "within_derived_tolerance": True,
            },
        }
        for value in integer_hashes
    ]
    summary = PROBE._summarize(receipt)
    assert summary["complete"] is True
    assert summary["overall_verdict"] == expected
