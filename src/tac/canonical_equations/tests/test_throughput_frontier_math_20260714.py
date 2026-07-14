from __future__ import annotations

import json

import numpy as np
import pytest

from tac.canonical_equations.equation import VERIFIED_VIA_SOURCE_INSPECTION
from tac.canonical_equations.throughput_frontier_math_20260714 import (
    EQUATION_IDS,
    SUPPORT_RECEIPT_SHA256,
    build_throughput_frontier_math_equations,
    populate_throughput_frontier_math_equations,
)
from tac.local_acceleration.throughput_frontier_math import (
    PrecisionLayer,
    PrecisionOption,
    SupportCost,
    certify_argmax_intervals,
    fixed_width_reduction_certificate,
    solve_discrete_precision_waterfill,
    support_closure_flop_accounting,
)


def test_builds_exactly_six_honestly_scoped_equations() -> None:
    equations = build_throughput_frontier_math_equations()
    assert tuple(equation.equation_id for equation in equations) == EQUATION_IDS
    assert len(equations) == 6
    assert len(set(EQUATION_IDS)) == 6

    required_domain_fields = {
        "research_only",
        "domain",
        "verdict_scope",
        "authority",
        "req_R",
        "distinct_from",
    }
    for equation in equations:
        domain = equation.domain_of_validity
        assert required_domain_fields <= domain.keys()
        assert domain["research_only"] is True
        assert domain["score_claim"] is False
        assert domain["promotion_eligible"] is False
        assert domain["pointer_moved"] is False
        assert "n600" in str(domain["req_R"])
        assert domain["distinct_from"]
        assert equation.provenance.score_claim_valid is False
        assert equation.provenance.promotion_eligible is False

    assert all(equation.empirical_anchors == () for equation in equations[:5])
    assert equations[3].equation_id.endswith("ordinal_margin_minimality_v1")
    assert equations[4].equation_id.endswith("sigma_metric_closure_gamma_admissibility_v1")
    assert len(equations[5].empirical_anchors) == 1
    support_anchor = equations[5].empirical_anchors[0]
    assert support_anchor.empirical_verification_status == VERIFIED_VIA_SOURCE_INSPECTION
    assert support_anchor.inputs["receipt_sha256"] == SUPPORT_RECEIPT_SHA256
    assert support_anchor.provenance.source_sha256 == SUPPORT_RECEIPT_SHA256
    assert support_anchor.empirical_output["ideal_exact_speedup_upper_bound"] == 1.0
    assert support_anchor.empirical_output["execution_status"] == (
        "STRUCTURALLY_REFUSED_BEFORE_EXECUTION"
    )


def test_backend_free_exact_reduction_and_strict_argmax_laws() -> None:
    # The settled Q15 absolute-sum bound needs 25 signed bits. The theorem is
    # about the declared bound, not a claim that all future tensors share it.
    reduction = fixed_width_reduction_certificate(
        max_abs_term=11_159_918,
        fan_in=1,
        accumulator_bits=32,
    )
    assert reduction["sum_abs_bound"] == 11_159_918
    assert reduction["minimum_signed_bits"] == 25
    assert reduction["no_overflow"] is True

    logits = np.asarray([[2.0, 1.0]], dtype=np.float32)
    strict = certify_argmax_intervals(logits, 0.49)
    tied = certify_argmax_intervals(logits, 0.5)
    assert strict.certified.tolist() == [True]
    assert strict.robust_margin[0] == pytest.approx(0.02)
    assert tied.robust_margin[0] == pytest.approx(0.0)
    assert tied.certified.tolist() == [False]


def test_backend_free_discrete_waterfill_and_dependency_closure() -> None:
    layers = (
        PrecisionLayer(
            "early",
            (
                PrecisionOption(bits=8, error_bound=0.01, measured_cost=8.0),
                PrecisionOption(bits=4, error_bound=0.08, measured_cost=3.0),
            ),
        ),
        PrecisionLayer(
            "head",
            (
                PrecisionOption(bits=8, error_bound=0.01, measured_cost=8.0),
                PrecisionOption(bits=4, error_bound=0.08, measured_cost=3.0),
            ),
        ),
    )
    allocation = solve_discrete_precision_waterfill(layers, error_budget=0.09)
    assert [choice.bits for choice in allocation.choices] == [4, 8]
    assert allocation.total_error_bound == pytest.approx(0.09)
    assert allocation.total_measured_cost == pytest.approx(11.0)

    closure = support_closure_flop_accounting(
        (
            SupportCost(
                name="frozen_segnet",
                dense_flops=100.0,
                requested_active_fraction=0.04736597696940104,
                closed_active_fraction=1.0,
                global_dependency=True,
            ),
        )
    )
    assert closure["naive_mask_speedup_upper_bound"] > 20.0
    assert closure["dependency_closed_speedup_upper_bound"] == 1.0
    assert closure["dependency_closed_flops"] == closure["dense_flops"]


def test_aggregate_population_writes_exactly_six_temporary_events(tmp_path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    lock = tmp_path / "canonical_equations.jsonl.lock"
    populated = populate_throughput_frontier_math_equations(
        path=registry,
        lock_path=lock,
        agent="codex",
        subagent_id="canonical_math_leg",
    )

    rows = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    assert tuple(equation.equation_id for equation in populated) == EQUATION_IDS
    assert len(rows) == 6
    assert [row["event_type"] for row in rows] == ["registered"] * 6
    assert tuple(row["equation_id"] for row in rows) == EQUATION_IDS
    assert all(row["subagent_id"] == "canonical_math_leg" for row in rows)
