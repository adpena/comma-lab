# SPDX-License-Identifier: MIT
"""Triality equation tests for the DC1 score-quotient contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tac.canonical_equations.ddm_score_quotient_functional_20260724 import (
    EQUATION_ID,
    build_ddm_score_quotient_functional_v1,
    populate_ddm_score_quotient_functional_v1,
)
from tac.optimization.ddm_score_quotient_functional_contract import (
    FunctionalParametersV1,
    TemporalLatentV1,
    compile_score_quotient_packet,
    score_quotient_functional_objective,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _receipt():
    parameters = FunctionalParametersV1(
        base_rgb_u8=np.zeros((2, 3), dtype=np.uint8),
        row_basis_i8=np.zeros((2, 3, 384), dtype=np.int8),
        col_basis_i8=np.zeros((2, 3, 512), dtype=np.int8),
    )
    latent = TemporalLatentV1(
        pair_index=0,
        coefficients_q8=(0, 0, 0, 0, 0, 0),
        xi_q12=(0, 0, 0, 0, 0, 0),
    )
    return compile_score_quotient_packet(
        named_base="fixture",
        named_base_bytes=b"base",
        parameters=parameters,
        temporal_latents=(latent,),
    ).receipt


def test_equation_is_design_only_fail_closed_and_uses_typed_receipt() -> None:
    equation = build_ddm_score_quotient_functional_v1(
        provenance_root=REPO_ROOT
    )
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["score_claim"] is False
    assert equation.domain_of_validity["current_verdict"] == "INCOMPLETE"
    assert (
        equation.domain_of_validity["missing_stream"]
        == "FIT_RESULT_RECEIVER_CLOSED_V14_OR_BETTER"
    )
    assert equation.empirical_anchors == ()
    objective = score_quotient_functional_objective(0.01, 0.001, _receipt())
    assert objective.exact_real_coder_bytes is True


def test_populate_uses_append_only_registry_callable(tmp_path: Path) -> None:
    registry = tmp_path / "equations.jsonl"
    lock = tmp_path / "equations.jsonl.lock"
    populated = populate_ddm_score_quotient_functional_v1(
        path=registry,
        lock_path=lock,
        provenance_root=REPO_ROOT,
        agent="codex",
        subagent_id="dc1-test",
    )
    assert populated.equation_id == EQUATION_ID
    rows = [json.loads(line) for line in registry.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event_type"] == "registered"
    assert rows[0]["equation_id"] == EQUATION_ID
    assert rows[0]["equation_payload"]["domain_of_validity"]["score_claim"] is False
