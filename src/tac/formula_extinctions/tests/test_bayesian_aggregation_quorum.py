# SPDX-License-Identifier: MIT
"""Tests for Row #5 — Bayesian-aggregation quorum (Surowiecki + Kemeny-Snell)."""
from __future__ import annotations

import pytest

from tac.formula_extinctions.bayesian_aggregation_quorum import (
    QuorumInput,
    canonical_bayesian_aggregation_quorum,
)


def test_t1_low_calibration_simple_majority():
    """T1 working group with low calibration -> simple majority K_majority."""
    r = canonical_bayesian_aggregation_quorum(QuorumInput(
        member_count=6, per_member_calibration=0.55, tier="T1",
    ))
    assert r.intermediate_values["quorum_regime"] == "simple_majority"
    assert r.solved_value == r.intermediate_values["K_majority"]


def test_t4_high_calibration_consensus():
    """T4 symposium with high calibration -> consensus required."""
    r = canonical_bayesian_aggregation_quorum(QuorumInput(
        member_count=20, per_member_calibration=0.95, tier="T4",
    ))
    assert r.intermediate_values["quorum_regime"] == "consensus_required"
    assert r.solved_value == 20


def test_t3_medium_calibration_super_majority():
    """T3 grand council medium calibration -> super_majority regime."""
    r = canonical_bayesian_aggregation_quorum(QuorumInput(
        member_count=20, per_member_calibration=0.70, tier="T3",
    ))
    assert r.intermediate_values["quorum_regime"] == "super_majority"
    assert r.intermediate_values["K_majority"] <= r.solved_value <= 20


def test_invalid_inputs_raise():
    """Bad inputs raise ValueError."""
    with pytest.raises(ValueError, match="member_count"):
        QuorumInput(member_count=1, per_member_calibration=0.6)
    with pytest.raises(ValueError, match="calibration"):
        QuorumInput(member_count=6, per_member_calibration=0.5)
    with pytest.raises(ValueError, match="calibration"):
        QuorumInput(member_count=6, per_member_calibration=1.1)
    with pytest.raises(ValueError, match="tier"):
        QuorumInput(member_count=6, per_member_calibration=0.7, tier="T5")  # type: ignore[arg-type]


def test_k_majority_consensus_bounds():
    """K_star is always in [K_majority, K_consensus]."""
    for n in (4, 6, 10, 20):
        for c in (0.55, 0.70, 0.85, 0.95):
            for t in ("T1", "T2", "T3", "T4"):
                r = canonical_bayesian_aggregation_quorum(QuorumInput(
                    member_count=n, per_member_calibration=c, tier=t,  # type: ignore[arg-type]
                ))
                k_maj = r.intermediate_values["K_majority"]
                k_cons = r.intermediate_values["K_consensus"]
                assert k_maj <= r.solved_value <= k_cons, (n, c, t, r.solved_value)


def test_citation_surowiecki_kemeny_snell():
    """Citations present."""
    r = canonical_bayesian_aggregation_quorum(QuorumInput(
        member_count=6, per_member_calibration=0.7,
    ))
    assert "Surowiecki" in r.literature_citation
    assert "Kemeny-Snell" in r.literature_citation
