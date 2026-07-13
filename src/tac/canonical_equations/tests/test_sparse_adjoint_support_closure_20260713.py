from __future__ import annotations

import json
import math

import pytest

from tac.canonical_equations.sparse_adjoint_support_closure_20260713 import (
    EQUATION_ID,
    ORACLE_MASK_GLOBAL_RELATIVE_L2_ERROR,
    ORACLE_MASK_IDEAL_SPATIAL_SPEEDUP,
    RAW_RANK_FOR_95PCT_ENERGY,
    RECEIPT_SHA256,
    SOURCE_MARGIN_MASK_GLOBAL_RELATIVE_L2_ERROR,
    build_sparse_adjoint_mask_error_and_se_support_closure_v1,
    eckart_young_relative_error,
    ideal_backward_speedup,
    masked_adjoint_error_bound,
    populate_sparse_adjoint_mask_error_and_se_support_closure_v1,
)


def test_scalar_laws_refuse_invalid_domains_and_match_identities() -> None:
    assert masked_adjoint_error_bound(
        jacobian_operator_norm=3.0, omitted_output_gradient_norm=2.0
    ) == 6.0
    assert ideal_backward_speedup(dense_flops=10.0, active_flops=4.0) == 2.5
    assert eckart_young_relative_error(singular_values=[4.0, 3.0], rank=1) == 0.6
    assert eckart_young_relative_error(singular_values=[4.0, 3.0], rank=2) == 0.0
    with pytest.raises(ValueError):
        masked_adjoint_error_bound(
            jacobian_operator_norm=math.inf, omitted_output_gradient_norm=1.0
        )
    with pytest.raises(ValueError):
        ideal_backward_speedup(dense_flops=4.0, active_flops=5.0)
    with pytest.raises(ValueError):
        eckart_young_relative_error(singular_values=[1.0], rank=2)


def test_equation_pins_measured_authority_and_scoped_no_go() -> None:
    equation = build_sparse_adjoint_mask_error_and_se_support_closure_v1()
    assert equation.equation_id == EQUATION_ID
    assert "J_F(x)^T(I-M)g=0" in equation.domain_of_validity["exactness"]
    assert "custom sparse kernels" in equation.domain_of_validity["speedup_boundary"]
    assert "score or pointer claims" in equation.domain_of_validity["excluded"]
    assert equation.provenance.score_claim_valid is False
    anchor = equation.empirical_anchors[0]
    assert anchor.inputs["receipt_sha256"] == RECEIPT_SHA256
    assert anchor.empirical_output["task455_hash_matches"] == 600
    assert anchor.empirical_output["raw_rank_for_95pct_energy"] == (
        RAW_RANK_FOR_95PCT_ENERGY
    )
    assert anchor.empirical_output["oracle_mask_global_relative_l2_error"] == (
        ORACLE_MASK_GLOBAL_RELATIVE_L2_ERROR
    )
    assert anchor.empirical_output["source_margin_mask_global_relative_l2_error"] == (
        SOURCE_MARGIN_MASK_GLOBAL_RELATIVE_L2_ERROR
    )
    assert anchor.empirical_output["ideal_spatial_backward_speedup_upper_bound"] == (
        ORACLE_MASK_IDEAL_SPATIAL_SPEEDUP
    )
    assert anchor.empirical_output["dense_kernel_realized_speedup"] == 1.0
    assert anchor.empirical_output["verdict"] == "NO_GO_DENSE_FULLRANK"


def test_equation_populates_only_explicit_temporary_registry(tmp_path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    populated = populate_sparse_adjoint_mask_error_and_se_support_closure_v1(
        path=registry,
        lock_path=tmp_path / "canonical_equations.jsonl.lock",
        agent="codex",
        subagent_id="test_sparse_adjoint",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    assert populated.equation_id == EQUATION_ID
    assert len(rows) == 1
    assert rows[0]["event_type"] == "registered"
    assert rows[0]["equation_id"] == EQUATION_ID
