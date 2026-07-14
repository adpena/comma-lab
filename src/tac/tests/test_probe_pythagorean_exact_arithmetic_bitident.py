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


def test_full_r_chain_order_geometry_and_int32_proof() -> None:
    plans = PROBE.build_full_r_plans()
    assert [plan["name"] for plan in plans] == [
        "down_w_transpose_512_to_1164",
        "down_h_transpose_384_to_874",
        "up_w_transpose_1164_to_512",
        "up_h_transpose_874_to_384",
    ]
    assert [(plan["out_size"], plan["in_size"]) for plan in plans] == [
        (512, 1164),
        (384, 874),
        (1164, 512),
        (874, 384),
    ]
    proof = PROBE.full_r_static_overflow_proof(plans)
    assert all(row["safe"] for row in proof)
    assert [row["max_fan_in"] for row in proof] == [1, 1, 10, 10]
    assert max(row["max_abs_accumulator_bound"] for row in proof) < np.iinfo(np.int32).max


def test_signed_requantization_covers_both_halfway_signs() -> None:
    values = np.asarray([-7, -6, -5, -4, 4, 5, 6, 7], dtype=np.int64)
    np.testing.assert_array_equal(
        PROBE.signed_round_divide(values, 4),
        np.asarray([-2, -2, -1, -1, 1, 1, 2, 2], dtype=np.int32),
    )


def test_integer_stage_overflow_preflight_fails_closed() -> None:
    plan = PROBE._transpose_plan(
        name="forced_overflow", in_size=3, out_size=11, mode="bicubic"
    )
    with pytest.raises(OverflowError, match="int32 proof failed"):
        PROBE.integer_stage_preflight(
            plan,
            max_abs_input=np.iinfo(np.int32).max,
            in_state_bits=7,
            out_state_bits=7,
        )


def test_small_non_square_integer_transpose_is_order_independent() -> None:
    plan = PROBE._transpose_plan(
        name="small_non_square", in_size=5, out_size=9, mode="bicubic"
    )
    source = np.arange(2 * 9 * 3, dtype=np.int32).reshape(2, 9, 3) - 20
    actual, _ = PROBE._transpose_axis_numpy_integer(
        source,
        plan,
        in_state_bits=7,
        out_state_bits=7,
    )
    source_indices, destinations, weights = PROBE._scatter_vectors_numpy(
        plan, left=2, right=3, integer=True
    )
    contributions = source.reshape(-1)[source_indices] * weights
    rng = np.random.default_rng(494)
    accumulator = np.zeros(actual.size, dtype=np.int64)
    for index in rng.permutation(contributions.size):
        accumulator[int(destinations[index])] += int(contributions[index])
    expected = PROBE.signed_round_divide(accumulator.reshape(actual.shape), 1 << PROBE.Q_BITS)
    np.testing.assert_array_equal(actual, expected)


def _fake_full_receipt(*, float_hashes: list[str], integer_hashes: list[str]) -> dict:
    authority_hash = "integer-authority"
    return {
        "contract": {"n_processes_per_variant": 10, "frames": 1200},
        "numpy_authority": {
            "summary": {
                "status": "MEASURED",
                "coverage_exact": True,
                "integer_corpus_sha256": authority_hash,
                "within_derived_bound": True,
            }
        },
        "trials": {
            "float_atomic": [
                {"status": "MEASURED", "frames": 1200, "corpus_sha256": value}
                for value in float_hashes
            ],
            "fixed_q15_int32_atomic": [
                {"status": "MEASURED", "frames": 1200, "corpus_sha256": value}
                for value in integer_hashes
            ],
        },
    }


def test_full_r_summary_positive_and_scoped_formulation_negative() -> None:
    positive = PROBE._summarize_full(
        _fake_full_receipt(
            float_hashes=[f"float-{index}" for index in range(10)],
            integer_hashes=["integer-authority"] * 10,
        )
    )
    assert positive["overall_verdict"] == "REAL-L70-LEVER-FULL-R-N600"
    assert positive["decisive_positive"] is True

    negative = PROBE._summarize_full(
        _fake_full_receipt(
            float_hashes=[f"float-{index}" for index in range(10)],
            integer_hashes=["wrong"] * 10,
        )
    )
    assert negative["overall_verdict"] == "FULL-R-INTEGER-FORMULATION-NO-GO"
    assert negative["verdict_scope"].startswith("FORMULATION:")


def test_full_r_summary_reports_missing_metal_as_blocked() -> None:
    receipt = _fake_full_receipt(float_hashes=[], integer_hashes=[])
    receipt["trials"]["float_atomic"] = [
        {"status": "BLOCKED_NOT_MEASURED", "blocker": "no Metal"}
    ]
    summary = PROBE._summarize_full(receipt)
    assert summary["overall_verdict"] == "BLOCKED_NOT_MEASURED"
    assert summary["complete"] is False


def test_real_cache_stream_covers_both_members_once() -> None:
    cache = PROBE.DEFAULT_GT_CACHE
    if not cache.is_file():
        pytest.skip("canonical real n600 cache not present")
    rows = list(PROBE._ordered_real_frames(gt_cache=cache, pair_start=11, pair_count=2))
    assert [(pair, member) for pair, member, _ in rows] == [
        (11, "gt_f0.npy"),
        (11, "gt_f1.npy"),
        (12, "gt_f0.npy"),
        (12, "gt_f1.npy"),
    ]
    assert all(frame.shape == (874, 1164, 3) and frame.dtype == np.uint8 for _, _, frame in rows)
