# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

import pytest

from tac.canonical_equations.ddm_pf2_dimension_conditioned_five_type_20260724 import (
    EQUATION_ID,
    RECEIPT_SHA256,
    build_ddm_pf2_metric_eligible_five_type_formulation_adjudication_v1,
    formulation_is_admissible,
    populate_ddm_pf2_metric_eligible_five_type_formulation_adjudication_v1,
)
from tac.canonical_equations.registry import load_registry_events_lenient
from tac.optimization.ddm_dimension_conditioned_two_type import (
    IDENTICAL_CONTENT_CODER_CONTROL,
    IDENTITY_EUCLIDEAN_CONTROL,
)


def test_callable_excludes_identity_and_requires_exact_content() -> None:
    assert not formulation_is_admissible(
        metric_status=IDENTITY_EUCLIDEAN_CONTROL,
        delta=-1.0,
        identical_content_proven=False,
    )
    assert formulation_is_admissible(
        metric_status=IDENTICAL_CONTENT_CODER_CONTROL,
        delta=-1.0,
        identical_content_proven=True,
    )
    assert not formulation_is_admissible(
        metric_status=IDENTICAL_CONTENT_CODER_CONTROL,
        delta=1.0,
        identical_content_proven=True,
    )
    with pytest.raises(ValueError, match="identical-content"):
        formulation_is_admissible(
            metric_status=IDENTICAL_CONTENT_CODER_CONTROL,
            delta=-1.0,
            identical_content_proven=False,
        )


def test_equation_rederives_scoped_incomplete_family_anchor() -> None:
    equation = (
        build_ddm_pf2_metric_eligible_five_type_formulation_adjudication_v1()
    )
    assert equation.equation_id == EQUATION_ID
    anchor = equation.empirical_anchors[0]
    assert anchor.inputs["receipt_sha256"] == RECEIPT_SHA256
    assert anchor.empirical_output["eligible_formulation_count"] == 2
    assert anchor.empirical_output["ineligible_formulation_count"] == 1
    assert anchor.empirical_output["accepted_formulation_count"] == 1
    assert anchor.empirical_output["f2_verdict_eligible"] is False
    assert anchor.empirical_output["f3_delta_bytes"] == 33_359
    assert anchor.empirical_output["all_routes_held"] is True
    assert any(
        "complete family verdict" in exclusion
        for exclusion in equation.domain_of_validity["excluded"]
    )


def test_populate_uses_locked_temporary_registry(tmp_path) -> None:
    registry = tmp_path / "registry.jsonl"
    equation = (
        populate_ddm_pf2_metric_eligible_five_type_formulation_adjudication_v1(
            path=registry,
            lock_path=tmp_path / "registry.lock",
            agent="codex",
            subagent_id="test_ddm_pf2",
        )
    )
    rows = load_registry_events_lenient(registry)
    assert equation.equation_id == EQUATION_ID
    assert len(rows) == 1
    assert rows[0]["event_type"] == "registered"
    assert json.loads(json.dumps(rows[0]))["equation_id"] == EQUATION_ID
