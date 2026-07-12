# SPDX-License-Identifier: MIT
"""Triality-leg tests for the frozen-SegNet input-costate injection law."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tac.boundary_math.segnet_gradient_replacement import costate_injection_loss_numpy
from tac.canonical_equations.segnet_costate_injection_20260712 import (
    SEGNET_COSTATE_INJECTION_EQUATION_ID,
    build_segnet_input_costate_injection_v1,
    populate_segnet_input_costate_injection_v1,
)


def test_equation_builds_with_exactness_boundary_and_non_orphan_triality() -> None:
    equation = build_segnet_input_costate_injection_v1()
    assert equation.equation_id == SEGNET_COSTATE_INJECTION_EQUATION_ID
    assert "lambda_hat equals" in equation.domain_of_validity["exact_identity_condition"]
    assert "forward/logit agreement" in equation.domain_of_validity["excluded"][0]
    assert "mask-only agreement" in equation.domain_of_validity["excluded"][1]
    assert "scorer, preprocess, receiver R" in equation.domain_of_validity[
        "objective_context_binding"
    ]
    assert "verified at compile" in equation.domain_of_validity["custody_binding"]
    assert "anchor and current frame hashes must be equal" in equation.domain_of_validity[
        "approximate_provider_policy"
    ]
    assert "rehashed on every decision" in equation.domain_of_validity[
        "objective_context_binding"
    ]
    assert "score_claim=false" in equation.domain_of_validity["authority"]
    assert "tac.witness_dsl.scorer_gradient_policy" in equation.canonical_consumers
    assert "tools.probe_segnet_costate_injection" in equation.canonical_producers
    assert equation.provenance.score_claim_valid is False
    assert equation.empirical_anchors == ()


def test_canonical_callable_computes_real_injection_functional() -> None:
    frame = np.array([[1.0, 2.0], [3.0, 4.0]])
    costate = np.array([[0.5, -1.0], [2.0, 0.25]])
    assert costate_injection_loss_numpy(frame, costate) == np.sum(frame * costate)


def test_population_uses_append_only_registry_writer(tmp_path: Path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    lock = tmp_path / "canonical_equations.jsonl.lock"
    populated = populate_segnet_input_costate_injection_v1(
        path=registry,
        lock_path=lock,
        agent="codex",
        subagent_id="test_segnet_costate_injection",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    assert populated.equation_id == SEGNET_COSTATE_INJECTION_EQUATION_ID
    assert len(rows) == 1
    assert rows[0]["event_type"] == "registered"
    assert rows[0]["equation_id"] == SEGNET_COSTATE_INJECTION_EQUATION_ID
