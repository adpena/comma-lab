# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

import pytest

from tac.optimization.pairset_component_marginal import (
    CONTEST_RATE_DENOMINATOR_BYTES,
    build_component_score_delta_payload,
    canonical_signal_refs,
    component_marginal_status,
    component_score_delta,
    rate_delta_for_archive_byte_delta,
)


def test_rate_delta_for_one_byte_drop_matches_contest_rate_term() -> None:
    expected = -25.0 / CONTEST_RATE_DENOMINATOR_BYTES
    assert rate_delta_for_archive_byte_delta(-1) == pytest.approx(expected)


def test_component_score_delta_supports_archive_byte_delta() -> None:
    value = component_score_delta(
        segnet_delta=0.0,
        posenet_delta=0.0,
        archive_byte_delta=-1,
    )
    assert value == pytest.approx(-25.0 / CONTEST_RATE_DENOMINATOR_BYTES)


def test_component_marginal_status_classifies_safe_and_protected_pairs() -> None:
    assert component_marginal_status(rate_delta=-1e-6) == (
        "rate_credit_exceeds_scorer_penalty"
    )
    assert component_marginal_status(segnet_delta=2e-6, rate_delta=-1e-6) == (
        "scorer_penalty_exceeds_rate_credit"
    )
    assert component_marginal_status(segnet_delta=1e-6, rate_delta=-1e-6) == (
        "rate_credit_ties_scorer_penalty"
    )


def test_component_score_delta_rejects_double_rate_inputs() -> None:
    with pytest.raises(ValueError, match="rate_delta OR archive_byte_delta"):
        component_score_delta(rate_delta=-1e-6, archive_byte_delta=-1)


def test_component_marginal_status_rejects_double_rate_inputs() -> None:
    with pytest.raises(ValueError, match="rate_delta OR archive_byte_delta"):
        component_marginal_status(rate_delta=-1e-6, archive_byte_delta=-1)


def test_component_payload_keeps_false_authority() -> None:
    payload = build_component_score_delta_payload(
        candidate_id="pairset_drop_one_rank021_pair0371",
        axis="contest_cpu",
        pair_index=371,
        dropped_pair_rank=21,
        archive_byte_delta=-1,
    )
    assert payload["schema"] == "pairset_component_marginal_score_delta.v1"
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert math.isfinite(payload["net_component_delta"])


def test_canonical_signal_refs_wire_master_gradient_xray_and_equation() -> None:
    refs = canonical_signal_refs()
    assert "pairset_component_marginal" in refs["xray_primitives"]
    assert "pairset_component_marginal_score_decomposition_v1" in refs[
        "canonical_equations"
    ]
    assert "tac.master_gradient_consumers.per_pair_difficulty_atlas" in refs[
        "master_gradient_consumers"
    ]
    assert refs["score_claim"] is False
