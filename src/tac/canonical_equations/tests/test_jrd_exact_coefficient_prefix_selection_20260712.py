# SPDX-License-Identifier: MIT
"""Triality-leg tests for exact JRD coefficient-prefix selection."""
from __future__ import annotations

import json
from pathlib import Path

from tac.canonical_equations.jrd_exact_coefficient_prefix_selection_20260712 import (
    EQUATION_ID,
    build_jrd_exact_coefficient_prefix_selection_v1,
    populate_jrd_exact_coefficient_prefix_selection_v1,
)


def test_equation_preserves_exact_gate_and_authority_boundary() -> None:
    equation = build_jrd_exact_coefficient_prefix_selection_v1()
    anchor = equation.empirical_anchors[0]

    assert equation.equation_id == EQUATION_ID
    assert "B(q)<B_{current}" in equation.latex_form
    assert equation.domain_of_validity["noise_floor"] == {"d_seg": 0.0, "d_pose": 0.0}
    assert equation.domain_of_validity["promotion_eligible"] is False
    assert equation.domain_of_validity["score_claim"] is False
    assert equation.domain_of_validity["review_status"].startswith(
        "recovery-written-UNREVIEWED"
    )
    assert "when present" in equation.domain_of_validity["review_status"]
    assert "adversarial_review_receipt.json" in equation.domain_of_validity[
        "review_status"
    ]
    assert "N/A-with-reason" in equation.domain_of_validity["dsl_leg"]
    assert anchor.empirical_output["response_rows"] == 288
    assert anchor.empirical_output["sealed_sections"] == 18
    assert anchor.empirical_output["sealed_coefficients"] == 71_223
    assert anchor.empirical_output["baseline_archive_bytes"] == 83_905
    assert anchor.empirical_output["selected_archive_bytes"] == 81_154
    assert anchor.empirical_output["archive_bytes_saved"] == 2751
    assert anchor.empirical_output["raw_precision_bits_removed"] == 40_416
    assert anchor.empirical_output["baseline_d_seg"] == 0.023157755533854168
    assert anchor.empirical_output["selected_d_seg"] == 0.0218505859375
    assert anchor.empirical_output["delta_d_seg"] == -0.0013071695963541678
    assert anchor.empirical_output["baseline_d_pose"] == 116.59830629690003
    assert anchor.empirical_output["selected_d_pose"] == 92.42743674059255
    assert anchor.empirical_output["delta_d_pose"] == -24.17086955630748
    assert anchor.empirical_output["accepted_combined_steps"] == 5
    assert anchor.empirical_output["rejected_combined_steps"] == 2
    assert anchor.empirical_output["fixture_verdict"] == "GO"
    assert anchor.empirical_output["task_verdict"] == "NEEDS-MORE"
    assert "V9/v8" in anchor.empirical_output["verdict_scope"]
    assert equation.provenance.score_claim_valid is False


def test_population_uses_append_only_registry_writer(tmp_path: Path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    lock = tmp_path / "canonical_equations.jsonl.lock"
    populated = populate_jrd_exact_coefficient_prefix_selection_v1(
        path=registry,
        lock_path=lock,
        agent="codex",
        subagent_id="test_jrd_prefix",
    )

    rows = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    assert populated.equation_id == EQUATION_ID
    assert len(rows) == 1
    assert rows[0]["event_type"] == "registered"
    assert rows[0]["equation_id"] == EQUATION_ID
