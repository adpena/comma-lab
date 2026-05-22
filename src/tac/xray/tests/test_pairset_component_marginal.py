# SPDX-License-Identifier: MIT
"""Tests for the pairset component-marginal xray primitive."""

from __future__ import annotations

from tac.xray.pairset_component_marginal import PairsetComponentMarginalXRay


def test_pairset_component_marginal_extracts_safe_and_protected_pairs():
    primitive = PairsetComponentMarginalXRay()
    result = primitive.compute(
        {
            "schema": "pairset_component_marginal_model.v1",
            "active": True,
            "training_row_count": 3,
            "axes": ["contest_cpu", "contest_cuda"],
            "axis_models": {
                "contest_cpu": {
                    "safe_drop_pair_indices": [371],
                    "protected_drop_pair_indices": [327, 376],
                },
                "contest_cuda": {
                    "safe_drop_pair_indices": [],
                    "protected_drop_pair_indices": [371],
                },
            },
            "cross_axis_transfer_diagnostics": [
                {"transfer_status": "cpu_improves_cuda_regresses"}
            ],
            "allowed_use": "component_marginal_planning_signal_only_no_score_or_dispatch_authority",
            "identity_policy": "candidate_id_and_selected_pair_indices_required_and_matched",
        }
    )

    report = result.primitive_value
    assert result.primitive_name == "pairset_component_marginal"
    assert result.evidence_grade == "empirical-anchor"
    assert report.safe_drop_pair_indices_by_axis["contest_cpu"] == (371,)
    assert report.protected_drop_pair_indices_by_axis["contest_cpu"] == (327, 376)
    assert report.protected_drop_pair_indices_by_axis["contest_cuda"] == (371,)
    assert report.cross_axis_transfer_statuses == ("cpu_improves_cuda_regresses",)
    assert result.metadata["score_claim"] is False
