# SPDX-License-Identifier: MIT
"""Tests for pairset component marginal canonical equation."""

from __future__ import annotations

from tac.canonical_equations.pairset_component_marginal import (
    build_pairset_component_marginal_score_decomposition_v1,
    pairset_component_marginal_payload,
    pairset_component_marginal_score_delta,
)


def test_pairset_component_marginal_score_delta_matches_rate_formula():
    one_byte_rate_credit = -25.0 / 37_545_489.0
    assert pairset_component_marginal_score_delta(
        segnet_delta=0.0,
        posenet_delta=0.0,
        archive_byte_delta=-1,
    ) == one_byte_rate_credit
    assert pairset_component_marginal_score_delta(
        segnet_delta=0.000001,
        posenet_delta=0.0,
        archive_byte_delta=-1,
    ) > 0.0


def test_pairset_component_marginal_equation_has_false_authority_payload():
    payload = pairset_component_marginal_payload(
        segnet_delta=0.000001,
        archive_byte_delta=-1,
        axis="contest_cpu",
        candidate_id="pairset_drop_one_rank026_pair0320",
        pair_index=320,
        dropped_pair_rank=26,
    )
    assert payload["schema"] == "pairset_component_marginal_score_delta.v1"
    assert payload["component_marginal_status"] == "scorer_penalty_exceeds_rate_credit"
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_build_pairset_component_marginal_canonical_equation():
    equation = build_pairset_component_marginal_score_decomposition_v1()
    assert equation.equation_id == "pairset_component_marginal_score_decomposition_v1"
    assert len(equation.empirical_anchors) == 8
    assert {
        anchor.anchor_id for anchor in equation.empirical_anchors
    } == {
        "dqs1_pair0371_drop_one_component_cpu_20260522",
        "dqs1_pair0371_drop_one_component_cuda_t4_20260522",
        "dqs1_pair0320_drop_one_component_cpu_penalty_20260522",
        "dqs1_pair0378_drop_one_component_cpu_penalty_20260522",
        "dqs1_pair0296_drop_one_component_cpu_penalty_20260522",
        "dqs1_pair0430_drop_one_component_cpu_penalty_20260522",
        "dqs1_pair0167_drop_one_component_cpu_penalty_20260522",
        "dqs1_pair0151_drop_one_component_cpu_penalty_20260522",
    }
    source_by_anchor = {
        anchor.anchor_id: anchor.source_artifact for anchor in equation.empirical_anchors
    }
    assert source_by_anchor[
        "dqs1_pair0151_drop_one_component_cpu_penalty_20260522"
    ].endswith(
        "codex_findings_dqs1_rank019_queue_worker_drift_hardening_"
        "20260522T201951Z_codex.md"
    )
    assert source_by_anchor[
        "dqs1_pair0167_drop_one_component_cpu_penalty_20260522"
    ].endswith(
        "codex_findings_dqs1_pairset_observation_feedback_20260522T164706Z_codex.md"
    )
    assert "tac.optimization.cross_family_candidate_portfolio" in equation.canonical_consumers
