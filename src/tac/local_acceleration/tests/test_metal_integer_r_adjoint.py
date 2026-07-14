# SPDX-License-Identifier: MIT
"""CPU contract tests for the order-independent integer render-R adjoint."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from tac.local_acceleration import metal_integer_r_adjoint as integer_r
from tac.local_acceleration.metal_fused_r_operator import resize_indices_weights_numpy


@pytest.mark.parametrize(
    ("in_size", "out_size", "mode"),
    [(1164, 512, "bilinear"), (384, 874, "bicubic")],
)
def test_resize_taps_match_render_r_authority(in_size: int, out_size: int, mode: str) -> None:
    actual_indices, actual_weights = integer_r.resize_taps(
        in_size=in_size, out_size=out_size, mode=mode
    )
    expected_indices, expected_weights = resize_indices_weights_numpy(
        in_size=in_size, out_size=out_size, mode=mode
    )
    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_array_equal(actual_weights.view(np.uint32), expected_weights.view(np.uint32))


def test_full_chain_order_and_static_overflow_proof() -> None:
    plans = integer_r.full_r_integer_plans()
    assert [plan.name for plan in plans] == [
        "down_w_transpose_512_to_1164",
        "down_h_transpose_384_to_874",
        "up_w_transpose_1164_to_512",
        "up_h_transpose_874_to_384",
    ]
    assert [(plan.out_size, plan.in_size) for plan in plans] == [
        (512, 1164),
        (384, 874),
        (1164, 512),
        (874, 384),
    ]
    proof = integer_r.full_r_int32_proof()
    assert len(proof) == 4
    assert all(row["safe"] for row in proof)
    assert max(int(row["max_abs_accumulator_bound"]) for row in proof) < integer_r.INT32_LIMIT
    assert proof[2]["in_state_bits"] == 7
    assert proof[2]["out_state_bits"] == 5


def test_signed_rounding_is_symmetric_and_has_explicit_half_rule() -> None:
    values = np.asarray([-7, -6, -5, -4, 4, 5, 6, 7], dtype=np.int64)
    actual = integer_r.signed_round_divide(values, 4)
    np.testing.assert_array_equal(actual, np.asarray([-2, -2, -1, -1, 1, 1, 2, 2]))


def test_overflow_proof_fails_closed() -> None:
    plan = integer_r.build_integer_transpose_plan(
        name="forced_overflow", in_size=3, out_size=11, mode="bicubic"
    )
    with pytest.raises(OverflowError, match="exceeds int32"):
        integer_r.prove_stage_int32(
            plan,
            max_abs_input_integer=integer_r.INT32_LIMIT,
            in_state_bits=7,
            out_state_bits=7,
        )


def test_integer_transpose_matches_permuted_scatter_order() -> None:
    plan = integer_r.build_integer_transpose_plan(
        name="small_non_square", in_size=5, out_size=9, mode="bicubic"
    )
    source = np.arange(2 * 9 * 3, dtype=np.int32).reshape(2, 9, 3) - 20
    actual = integer_r.integer_transpose_numpy(
        source, plan, in_state_bits=7, out_state_bits=7
    )

    contributions: list[tuple[int, int, int]] = []
    for destination in range(plan.in_size):
        start = int(plan.starts[destination])
        for offset in range(int(plan.counts[destination])):
            slot = start + offset
            contributions.append(
                (destination, int(plan.source_indices[slot]), int(plan.q15_weights[slot]))
            )
    rng = np.random.default_rng(494)
    accumulator = np.zeros((2, plan.in_size, 3), dtype=np.int64)
    for index in rng.permutation(len(contributions)):
        destination, source_index, weight = contributions[int(index)]
        accumulator[:, destination, :] += source[:, source_index, :].astype(np.int64) * weight
    expected = integer_r.signed_round_divide(accumulator, 1 << integer_r.Q_WEIGHT_BITS)
    np.testing.assert_array_equal(actual, expected)


def test_kernel_is_gather_only_and_signature_preserves_authority_boundary() -> None:
    source = inspect.getsource(integer_r._kernel)
    assert "atomic_fetch" not in source
    signature = integer_r.integer_r_signature()
    assert signature["atomic"] is False
    assert signature["default_enabled"] is False
    assert signature["score_claim"] is False
    assert "CPU/CUDA" in signature["terminal_authority"]


def test_config_rejects_unregistered_precision() -> None:
    integer_r.IntegerRAdjointConfig(cotangent_unit=1.0).validate()
    with pytest.raises(ValueError, match="state schedule"):
        integer_r.IntegerRAdjointConfig(
            cotangent_unit=1.0,
            state_bits_by_boundary=(7, 7, 7, 7, 7),
        ).validate()


def test_receipt_gate_requires_full_custody_and_speed(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    payload = {
        "schema": "pythagorean_exact_arithmetic_full_r_n600.v2",
        "summary": {
            "overall_verdict": "REAL-L70-LEVER-FULL-R-N600",
            "complete": True,
            "decisive_positive": True,
        },
        "source_custody": {"probe": {"sha256": "p"}},
        "training_integration": {
            "policy_sha256": "d",
            "kernel_benchmark": {"measured": True, "speedup_x": 1.01},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    admitted = integer_r.receipt_admits_integer_r(
        path, expected_probe_sha256="p", expected_policy_sha256="d"
    )
    assert admitted["summary"]["decisive_positive"] is True
    payload["training_integration"]["kernel_benchmark"]["speedup_x"] = 0.99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="positive_speed"):
        integer_r.receipt_admits_integer_r(
            path, expected_probe_sha256="p", expected_policy_sha256="d"
        )
