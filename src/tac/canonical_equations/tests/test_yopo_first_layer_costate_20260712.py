# SPDX-License-Identifier: MIT
"""Triality checks for the isolated YOPO first-layer provider law."""

from __future__ import annotations

import json

from tac.canonical_equations import (
    build_yopo_first_layer_costate_v1 as package_build_yopo_first_layer_costate_v1,
)
from tac.canonical_equations.yopo_first_layer_costate_20260712 import (
    YOPO_FIRST_LAYER_COSTATE_EQUATION_ID,
    build_yopo_first_layer_costate_v1,
    populate_yopo_first_layer_costate_v1,
)


def test_yopo_equation_pins_cut_citation_and_fail_closed_boundary() -> None:
    equation = build_yopo_first_layer_costate_v1()
    assert equation.equation_id == YOPO_FIRST_LAYER_COSTATE_EQUATION_ID
    assert "Zhang, Zhang, Lu, Zhu, Dong (2019)" in equation.domain_of_validity["citation"]
    assert "arXiv:1905.00877" in equation.domain_of_validity["citation"]
    assert "blocks[0]" in equation.domain_of_validity["cut"]
    assert "universal cosine threshold" in equation.domain_of_validity["excluded"][1]
    assert "full_teacher" in equation.domain_of_validity["fallback"]
    assert equation.units_in["p1"] == "teacher_loss_per_split_activation_unit"
    assert equation.units_out["lambda_hat"] == "teacher_loss_per_rendered_frame_unit"
    assert equation.provenance.score_claim_valid is False
    assert package_build_yopo_first_layer_costate_v1().equation_id == equation.equation_id


def test_yopo_equation_population_uses_append_only_registry(tmp_path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    populated = populate_yopo_first_layer_costate_v1(
        path=registry,
        lock_path=tmp_path / "canonical_equations.jsonl.lock",
        agent="codex",
        subagent_id="test_yopo_first_layer",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    assert populated.equation_id == YOPO_FIRST_LAYER_COSTATE_EQUATION_ID
    assert len(rows) == 1
    assert rows[0]["event_type"] == "registered"
    assert rows[0]["equation_id"] == YOPO_FIRST_LAYER_COSTATE_EQUATION_ID
