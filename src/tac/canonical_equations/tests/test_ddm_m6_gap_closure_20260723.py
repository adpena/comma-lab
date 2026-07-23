# SPDX-License-Identifier: MIT
"""Tests for the DDM M6 pool-aware gap-closure law."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.ddm_m6_gap_closure_20260723 import (
    DECISIVE_GAP_BYTES,
    PoolCredit,
    compose_gap_closure,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _credit(pool: str, reduction: int, *, closed: bool = True) -> PoolCredit:
    return PoolCredit(
        pool_id=pool,
        lever_ids=(f"{pool}-lever",),
        joint_reduction_bytes=reduction,
        receiver_closed=closed,
        evidence_scope="test",
    )


def test_m6_exact_composition_is_y13_with_22632_residual() -> None:
    result = compose_gap_closure(
        (
            _credit("P_REALIZE", 0),
            _credit("P_TEMPORAL_DESCRIPTION", 13),
            _credit("P_NULL_GAUGE", 0),
        ),
        final_archive_bytes=177_156,
        final_same_artifact_receiver_closed=True,
    )
    assert DECISIVE_GAP_BYTES == 22_645
    assert result.admitted_reduction_bytes == 13
    assert result.residual_gap_bytes == 22_632
    assert result.sub015_reached is False


def test_unclosed_final_artifact_receives_zero_credit() -> None:
    result = compose_gap_closure(
        (_credit("P_TEMPORAL_DESCRIPTION", 13),),
        final_archive_bytes=177_156,
        final_same_artifact_receiver_closed=False,
    )
    assert result.admitted_reduction_bytes == 0
    assert result.residual_gap_bytes == 22_645


def test_duplicate_pool_refuses_singleton_addition() -> None:
    with pytest.raises(ValueError, match="exactly one joint credit"):
        compose_gap_closure(
            (
                _credit("P_REALIZE", 10),
                _credit("P_REALIZE", 20),
            ),
            final_archive_bytes=177_139,
            final_same_artifact_receiver_closed=True,
        )


def test_final_delta_cannot_exceed_pool_bound() -> None:
    with pytest.raises(ValueError, match="exceeds admitted pool credit"):
        compose_gap_closure(
            (_credit("P_TEMPORAL_DESCRIPTION", 12),),
            final_archive_bytes=177_156,
            final_same_artifact_receiver_closed=True,
        )


def test_durable_receipt_preserves_authority_boundary() -> None:
    receipt_path = (
        REPO_ROOT
        / ".omx/research/ddm_m6_close_22645_byte_gap_20260723_receipt.json"
    )
    if not receipt_path.is_file():
        pytest.skip("receipt is added after deterministic derivation")
    receipt = json.loads(receipt_path.read_text())
    result = receipt["pool_aware_composition"]
    assert result["Y_bytes"] == 13
    assert result["residual_gap_bytes"] == 22_632
    assert result["sub015_reached"] is False
    assert receipt["candidate_and_dispatch"]["byte_close_candidate_flagged"] is False
    assert receipt["candidate_and_dispatch"]["r6_exact_eval_flagged"] is False
    assert receipt["score_claim"] is False
    assert receipt["main_landing_review"]["required"] is True
