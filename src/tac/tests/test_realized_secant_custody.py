# SPDX-License-Identifier: MIT
"""Focused synthetic tests for G2e realized-secant custody primitives."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from tac.optimization.realized_secant_custody import (
    RECEIPT_SCHEMA,
    PairSolveStatus,
    QPStatus,
    RealizedSecantCustodyError,
    SecantObservation,
    WriteSecantObservation,
    build_pair_trust_region_custody,
    build_trust_regions,
    canonical_sha256,
    decode_coefficient_packet,
    encode_coefficient_packet,
    solve_minimal_norm_inequalities,
    validate_receipt,
)


def _write(
    *,
    target_class: int,
    bucket: str,
    predicted: float,
    realized: float,
    amplitude: float,
) -> WriteSecantObservation:
    return WriteSecantObservation(
        ordinal=0,
        target_class=target_class,
        current_class=0,
        pre_margin=-0.5,
        margin_bucket=bucket,
        expected_sign=1 if predicted > 0 else -1,
        feature_displacement=tuple([0.0] * 143 + [realized]),
        predicted_margin_delta=predicted,
        realized_margin_delta=realized,
        secant_ratio=realized / amplitude,
    )


def _observation(
    *,
    pair: int = 0,
    column: int = 0,
    amplitude: float = 2.0,
    target_class: int = 1,
    bucket: str = "negative_1_to_0",
    predicted: float = 1.0,
    realized: float = 0.9,
) -> SecantObservation:
    return SecantObservation(
        pair_index=pair,
        column_index=column,
        signed_amplitude=amplitude,
        applied_rgb_l2=2.0,
        applied_rgb_linf=2.0,
        uint8_saturation_count=0,
        writes=(
            _write(
                target_class=target_class,
                bucket=bucket,
                predicted=predicted,
                realized=realized,
                amplitude=amplitude,
            ),
        ),
    )


def test_signed_secant_ratios_and_class_bucket_isolation() -> None:
    positive = _observation()
    negative = _observation(
        pair=1,
        amplitude=-2.0,
        target_class=2,
        bucket="negative_gt_1",
        predicted=-1.0,
        realized=-0.8,
    )
    assert positive.writes[0].secant_ratio == pytest.approx(0.45)
    assert negative.writes[0].secant_ratio == pytest.approx(0.4)
    regions = build_trust_regions((positive, negative), relative_residual_tolerance=0.25)
    assert [(row.target_class, row.margin_bucket) for row in regions] == [
        (1, "negative_1_to_0"),
        (2, "negative_gt_1"),
    ]
    assert all(row.usable for row in regions)


def test_failed_row_refuses_only_its_class_bucket_region() -> None:
    good = _observation(target_class=1, bucket="negative_1_to_0")
    bad = _observation(
        pair=1,
        target_class=2,
        bucket="negative_gt_1",
        predicted=1.0,
        realized=-0.1,
    )
    regions = build_trust_regions((good, bad), relative_residual_tolerance=0.25)
    assert regions[0].usable is True
    assert regions[1].usable is False
    assert set(regions[1].refusal_reasons) == {
        "REALIZED_SIGN_OR_ZERO",
        "RELATIVE_SECANT_RESIDUAL",
    }


def test_minimal_norm_solver_handles_active_inequality_and_uint8_bound() -> None:
    result = solve_minimal_norm_inequalities(
        np.asarray([[1.0]]),
        np.asarray([1.0]),
        np.asarray([[1.0]]),
        np.asarray([254.0]),
    )
    assert result.status is QPStatus.SOLVED
    assert result.coefficients == pytest.approx((1.0,))
    assert result.max_primal_violation <= 1e-12
    assert result.min_active_multiplier is not None
    assert result.min_active_multiplier >= -1e-12
    assert result.stationarity_residual <= 1e-12


def test_minimal_norm_solver_refuses_saturation_limited_infeasible_problem() -> None:
    result = solve_minimal_norm_inequalities(
        np.asarray([[1.0]]),
        np.asarray([2.0]),
        np.asarray([[1.0]]),
        np.asarray([254.0]),
    )
    assert result.status is QPStatus.INFEASIBLE
    assert result.objective is None
    assert result.max_primal_violation is None
    assert result.min_active_multiplier is None
    assert result.stationarity_residual is None
    assert json.dumps(result.as_dict(), allow_nan=False)


def test_double_solve_and_coefficient_decode_are_deterministic() -> None:
    kwargs = {
        "margin_jacobian": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "required_margin_delta": np.asarray([1.0, 2.0]),
        "rgb_direction_matrix": np.eye(2),
        "baseline_rgb": np.asarray([100.0, 100.0]),
    }
    first = solve_minimal_norm_inequalities(**kwargs)
    second = solve_minimal_norm_inequalities(**kwargs)
    assert first == second
    assert first.coefficients == pytest.approx((1.0, 2.0))
    payload = encode_coefficient_packet(first.coefficients)
    assert decode_coefficient_packet(payload) == decode_coefficient_packet(payload)
    assert encode_coefficient_packet(decode_coefficient_packet(payload)) == payload


def _rehash_receipt(receipt: dict) -> None:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = canonical_sha256(unsigned)


def _valid_receipt(observations: list[SecantObservation] | None = None) -> dict:
    typed_rows = observations or [_observation(pair=pair, column=column) for pair in range(2) for column in range(2)]
    trust_rows = list(
        build_pair_trust_region_custody(
            typed_rows,
            pair_count=2,
            relative_residual_tolerance=0.25,
        )
    )
    unusable_pairs = {row["pair_index"] for row in trust_rows if row["usable"] is False}
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "completed_prefix": 2,
        "config": {"relative_secant_residual_tolerance": 0.25},
        "column_indices": [0, 1],
        "secant_observations": [row.as_dict() for row in typed_rows],
        "pair_trust_regions": trust_rows,
        "pair_solves": [
            {
                "pair_index": pair,
                "status": (
                    PairSolveStatus.TRUST_REGION_REFUSED.value
                    if pair in unusable_pairs
                    else PairSolveStatus.QP_INFEASIBLE.value
                ),
                "admitted": False,
            }
            for pair in range(2)
        ],
    }
    _rehash_receipt(receipt)
    return receipt


def test_malformed_observation_and_receipt_are_rejected() -> None:
    with pytest.raises(RealizedSecantCustodyError, match="secant_ratio"):
        WriteSecantObservation(
            ordinal=0,
            target_class=1,
            current_class=0,
            pre_margin=-0.5,
            margin_bucket="negative_1_to_0",
            expected_sign=1,
            feature_displacement=tuple([0.0] * 144),
            predicted_margin_delta=1.0,
            realized_margin_delta=1.0,
            secant_ratio=float("nan"),
        )

    receipt = _valid_receipt()
    assert validate_receipt(receipt, expected_pair_count=2) == receipt["receipt_sha256"]

    missing = copy.deepcopy(receipt)
    missing["secant_observations"].pop()
    _rehash_receipt(missing)
    with pytest.raises(RealizedSecantCustodyError, match="exactly one observation"):
        validate_receipt(missing, expected_pair_count=2)

    corrupt = bytearray(encode_coefficient_packet((1.0, 2.0)))
    corrupt[-1] ^= 1
    with pytest.raises(RealizedSecantCustodyError, match="checksum"):
        decode_coefficient_packet(bytes(corrupt))


def test_receipt_strictly_rederives_hashed_per_pair_trust_regions() -> None:
    receipt = _valid_receipt()
    assert all(len(row["row_sha256"]) == 64 for row in receipt["pair_trust_regions"])
    assert validate_receipt(receipt, expected_pair_count=2) == receipt["receipt_sha256"]

    corrupted = copy.deepcopy(receipt)
    corrupted["pair_trust_regions"][0]["usable"] = False
    _rehash_receipt(corrupted)
    with pytest.raises(RealizedSecantCustodyError, match="trust-region custody mismatch"):
        validate_receipt(corrupted, expected_pair_count=2)


@pytest.mark.parametrize("status", ["", "REFUSED", "QP_SOLVED_PENDING_HARD_ORACLE"])
def test_receipt_refuses_nonterminal_or_unrecognized_pair_status(status: str) -> None:
    receipt = _valid_receipt()
    receipt["pair_solves"][0]["status"] = status
    _rehash_receipt(receipt)
    with pytest.raises(RealizedSecantCustodyError, match="recognized nonempty terminal status"):
        validate_receipt(receipt, expected_pair_count=2)


def test_receipt_refuses_prefix_pair_index_and_admission_corruption() -> None:
    prefix = _valid_receipt()
    prefix["completed_prefix"] = 1
    _rehash_receipt(prefix)
    with pytest.raises(RealizedSecantCustodyError, match="completed_prefix"):
        validate_receipt(prefix, expected_pair_count=2)

    noncontiguous = _valid_receipt()
    noncontiguous["pair_solves"][1]["pair_index"] = 0
    _rehash_receipt(noncontiguous)
    with pytest.raises(RealizedSecantCustodyError, match="not contiguous"):
        validate_receipt(noncontiguous, expected_pair_count=2)

    non_bool = _valid_receipt()
    non_bool["pair_solves"][0]["admitted"] = 0
    _rehash_receipt(non_bool)
    with pytest.raises(RealizedSecantCustodyError, match="exact bool"):
        validate_receipt(non_bool, expected_pair_count=2)

    inconsistent = _valid_receipt()
    inconsistent["pair_solves"][0].update(
        status=PairSolveStatus.ADMITTED_RECEIVER_CLOSED.value,
        admitted=False,
    )
    _rehash_receipt(inconsistent)
    with pytest.raises(RealizedSecantCustodyError, match="status/admitted consistency"):
        validate_receipt(inconsistent, expected_pair_count=2)


def test_receipt_unusable_rederived_trust_region_requires_refusal() -> None:
    observations = [
        _observation(
            pair=pair,
            column=column,
            realized=-0.1 if pair == 0 else 0.9,
        )
        for pair in range(2)
        for column in range(2)
    ]
    receipt = _valid_receipt(observations)
    assert receipt["pair_solves"][0] == {
        "pair_index": 0,
        "status": PairSolveStatus.TRUST_REGION_REFUSED.value,
        "admitted": False,
    }
    assert validate_receipt(receipt, expected_pair_count=2) == receipt["receipt_sha256"]

    corrupted = copy.deepcopy(receipt)
    corrupted["pair_solves"][0]["status"] = PairSolveStatus.QP_INFEASIBLE.value
    _rehash_receipt(corrupted)
    with pytest.raises(RealizedSecantCustodyError, match="unusable trust region"):
        validate_receipt(corrupted, expected_pair_count=2)
