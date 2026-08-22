from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ec2_collateral_suppressed_proposer import (
    NET_FLIPS_PER_BYTE,
    ORIENTED_CONTEXTS,
    CollateralSuppressedProposer,
    EC2ProposerError,
    collateral_priced_delta,
    fit_context_counts,
    gate_from_context_counts,
    oriented_context_codes_at,
)


def test_context_code_matches_center_left_right_up_down_base5_order() -> None:
    tokens = np.array([[[0, 1, 2], [3, 4, 0], [1, 2, 3]]], dtype=np.uint8)
    code = oriented_context_codes_at(
        tokens,
        np.array([0]),
        np.array([1]),
        np.array([1]),
    )
    expected = 4 + 5 * 3 + 25 * 0 + 125 * 1 + 625 * 2
    assert code.tolist() == [expected]


def test_context_gate_roundtrips_and_uses_only_codes() -> None:
    keep = np.zeros(ORIENTED_CONTEXTS, dtype=np.bool_)
    keep[[4, 25, 624, 3124]] = True
    proposer = CollateralSuppressedProposer(keep)
    payload = proposer.to_payload()
    repeat = proposer.to_payload()
    parsed = CollateralSuppressedProposer.from_payload(payload)
    assert payload == repeat
    assert np.array_equal(parsed.keep_by_context, keep)
    assert parsed.propose(np.array([4, 5, 3124], dtype=np.uint16)).tolist() == [True, False, True]


def test_collateral_is_symmetric_and_rate_is_charged() -> None:
    break_even = collateral_priced_delta(
        expected_beneficial=NET_FLIPS_PER_BYTE,
        expected_harmful=0.0,
        delta_archive_bytes=1,
    )
    harmful = collateral_priced_delta(
        expected_beneficial=10.0,
        expected_harmful=11.0,
        delta_archive_bytes=0,
    )
    assert break_even.joint_score == pytest.approx(0.0, abs=1e-18)
    assert not break_even.accepted
    assert harmful.segmentation_score > 0.0
    assert not harmful.accepted


def test_count_fit_and_smoothed_gate_fail_closed_on_sparse_contexts() -> None:
    codes = np.array([3, 3, 3, 9, 9, 12], dtype=np.uint16)
    outcomes = np.array([1, 1, -1, -1, -1, 1], dtype=np.int8)
    counts = fit_context_counts(codes, outcomes)
    gate = gate_from_context_counts(
        counts,
        minimum_beneficial_fraction=0.55,
        minimum_observations=2,
    )
    assert gate.keep_by_context[3]
    assert not gate.keep_by_context[9]
    assert not gate.keep_by_context[12]


def test_invalid_runtime_inputs_are_rejected() -> None:
    keep = np.zeros(ORIENTED_CONTEXTS, dtype=np.bool_)
    proposer = CollateralSuppressedProposer(keep)
    with pytest.raises(EC2ProposerError):
        proposer.propose(np.array([ORIENTED_CONTEXTS], dtype=np.int64))
    with pytest.raises(EC2ProposerError):
        CollateralSuppressedProposer.from_payload(b"not-a-payload")
