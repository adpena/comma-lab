from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_lp1_layer_pricing_20260725 import (
    RECEIPT,
    build_ddm_lp1_deepest_home_context_waterfill_v1,
    deepest_surviving_layer,
    populate_ddm_lp1_deepest_home_context_waterfill_v1,
    price_context_and_allocate,
)
from tac.canonical_equations.registry import load_registry_events_lenient


def test_context_and_marginal_admission_are_fail_closed() -> None:
    result = price_context_and_allocate(
        [
            {
                "explicit_bytes": 100,
                "contextual_bytes": 60,
                "context_parameter_bytes": 10,
                "same_object": True,
            },
            {
                "explicit_bytes": 100,
                "contextual_bytes": 60,
                "context_parameter_bytes": 10,
                "same_object": False,
            },
        ],
        measured_receiver_closed_marginals=[
            {"stream_id": "keep", "bytes": 5, "joint_score_delta": -0.1},
            {"stream_id": "drop", "bytes": 2, "joint_score_delta": 0.0},
        ],
    )
    assert result["context_rows"][0]["keep_context"] is True
    assert result["context_rows"][0]["selected_bytes"] == 70
    assert result["context_rows"][1]["keep_context"] is False
    assert result["context_rows"][1]["selected_bytes"] == 100
    assert result["allocated_bytes"] == 5
    assert result["unmeasured_reserves_allocate_zero"] is True


def test_context_and_marginal_types_fail_closed() -> None:
    with pytest.raises(ValueError, match="same_object must be boolean"):
        price_context_and_allocate(
            [
                {
                    "explicit_bytes": 100,
                    "contextual_bytes": 60,
                    "context_parameter_bytes": 10,
                    "same_object": 1,
                }
            ],
            measured_receiver_closed_marginals=[],
        )
    with pytest.raises(ValueError, match="must be finite numeric"):
        price_context_and_allocate(
            [],
            measured_receiver_closed_marginals=[
                {"stream_id": "nan", "bytes": 1, "joint_score_delta": float("nan")}
            ],
        )


def test_deepest_surviving_layer_rejects_nonmonotone_claims() -> None:
    assert (
        deepest_surviving_layer(
            {
                "L1_program": True,
                "L2_chart_grammar": True,
                "L3_RGB_realization": True,
                "L4_scorer_feature": False,
            }
        )
        == "L3_RGB_realization"
    )
    with pytest.raises(ValueError, match="cannot resume"):
        deepest_surviving_layer(
            {
                "L1_program": True,
                "L2_chart_grammar": False,
                "L3_RGB_realization": True,
                "L4_scorer_feature": False,
            }
        )


def test_equation_build_and_locked_population(tmp_path) -> None:
    equation = build_ddm_lp1_deepest_home_context_waterfill_v1()
    assert equation.equation_id == "ddm_lp1_deepest_home_context_waterfill_v1"
    assert equation.domain_of_validity["score_claim"] is False

    registry = tmp_path / "canonical_equations_registry.jsonl"
    populated = populate_ddm_lp1_deepest_home_context_waterfill_v1(
        source_receipt=RECEIPT,
        path=registry,
        lock_path=tmp_path / "registry.lock",
        agent="codex",
        subagent_id="test_ddm_lp1",
    )
    assert populated.equation_id == equation.equation_id
    events = load_registry_events_lenient(registry)
    assert len(events) == 1
    assert events[0]["equation_id"] == equation.equation_id
