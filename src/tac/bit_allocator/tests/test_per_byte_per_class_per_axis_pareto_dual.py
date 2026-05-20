# SPDX-License-Identifier: MIT
"""Tests for the WAVE-2-PREREQ-BIT-ALLOCATOR namespace extension.

Covers per_byte / per_class / per_axis / pareto_dual allocators landed
2026-05-20 per CATHEDRAL-SMARTER-DESIGN-MEMO Decision 10.

The sister per_pair_difficulty_weighted allocator is covered by
``src/tac/bit_allocator/tests/test_per_pair_difficulty_weighted.py``;
this file focuses on the 4 new sister allocators + namespace-level
cross-allocator invariants.

Lane: ``lane_wave_2_prereq_bit_allocator_namespace_20260520``.
"""
from __future__ import annotations

import json
import math

import pytest

from tac.bit_allocator import (
    # per_byte
    PER_BYTE_CANONICAL_EQUATION_ID,
    PerByteAllocationError,
    PerByteAllocationMethod,
    PerByteAllocationPlan,
    allocate_per_byte,
    # per_class
    CANONICAL_SEGNET_CLASS_NAMES,
    PerClassAllocationError,
    PerClassAllocationStrategy,
    PerClassBitAllocationPlan,
    SEGNET_CLASS_COUNT,
    allocate_per_class,
    # per_axis
    CANONICAL_SCORER_AXES,
    CANONICAL_SCORER_COEFFICIENTS_FORMULA_WEIGHTED,
    PerAxisAllocationError,
    PerAxisAllocationStrategy,
    PerAxisBitAllocationPlan,
    allocate_per_axis,
    # pareto_dual
    DEFAULT_BISECTION_ITERS,
    ParetoDualBitAllocationPlan,
    ParetoDualError,
    ParetoDualMethod,
    allocate_via_lagrangian_dual,
)
from tac.provenance import ProvenanceEvidenceGrade, ProvenanceKind


# =====================================================================
# per_byte allocator
# =====================================================================


class TestPerByteTopK:
    def test_top_k_greedy_assigns_cap_to_top_bytes(self) -> None:
        plan = allocate_per_byte(
            total_budget_bits=20,
            sensitivity_per_byte={0: 5.0, 1: 0.1, 2: 9.0, 3: 0.5},
            method=PerByteAllocationMethod.TOP_K_BY_SENSITIVITY,
            top_k=2,
            per_byte_bit_cap=8,
        )
        assert plan.method is PerByteAllocationMethod.TOP_K_BY_SENSITIVITY
        # Top 2 by sensitivity (9.0 then 5.0) → bytes 2, 0 get 8 bits each.
        assert plan.bits_per_byte[2] == 8
        assert plan.bits_per_byte[0] == 8
        # Remaining 4 bits go to next-ranked (0.5 > 0.1) → byte 3 gets 4.
        assert plan.bits_per_byte[3] == 4
        assert plan.residual_bits == 0

    def test_top_k_respects_per_byte_bit_cap(self) -> None:
        plan = allocate_per_byte(
            total_budget_bits=1000,
            sensitivity_per_byte={0: 10.0, 1: 5.0},
            method=PerByteAllocationMethod.TOP_K_BY_SENSITIVITY,
            top_k=2,
            per_byte_bit_cap=8,
        )
        # Both bytes capped at 8; residual = 1000 - 16 = 984.
        assert plan.bits_per_byte[0] == 8
        assert plan.bits_per_byte[1] == 8
        assert plan.residual_bits == 984

    def test_top_k_tie_break_by_lower_byte_offset(self) -> None:
        # Equal sensitivities; tie-break should prefer lower offset.
        plan = allocate_per_byte(
            total_budget_bits=8,
            sensitivity_per_byte={5: 1.0, 0: 1.0, 3: 1.0},
            method=PerByteAllocationMethod.TOP_K_BY_SENSITIVITY,
            top_k=1,
            per_byte_bit_cap=8,
        )
        # Tie at 1.0 sensitivity; byte 0 wins by tie-break.
        assert plan.bits_per_byte[0] == 8

    def test_top_k_budget_partially_assigned_when_top_k_fits_with_residual(self) -> None:
        plan = allocate_per_byte(
            total_budget_bits=10,
            sensitivity_per_byte={0: 9.0, 1: 5.0, 2: 0.1},
            method=PerByteAllocationMethod.TOP_K_BY_SENSITIVITY,
            top_k=1,
            per_byte_bit_cap=8,
        )
        # top-1 (byte 0) gets 8; remaining 2 bits go to byte 1 (next-ranked).
        assert plan.bits_per_byte[0] == 8
        assert plan.bits_per_byte[1] == 2
        assert plan.residual_bits == 0


class TestPerByteUniform:
    def test_uniform_distributes_equally(self) -> None:
        plan = allocate_per_byte(
            total_budget_bits=20,
            sensitivity_per_byte={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
            method=PerByteAllocationMethod.UNIFORM_BASELINE,
            per_byte_bit_cap=8,
        )
        assert plan.method is PerByteAllocationMethod.UNIFORM_BASELINE
        # 20 / 4 = 5 each (under cap; no residual).
        assert plan.bits_per_byte == {0: 5, 1: 5, 2: 5, 3: 5}
        assert plan.residual_bits == 0

    def test_uniform_residual_when_budget_indivisible(self) -> None:
        plan = allocate_per_byte(
            total_budget_bits=10,
            sensitivity_per_byte={0: 1.0, 1: 1.0, 2: 1.0},
            method=PerByteAllocationMethod.UNIFORM_BASELINE,
            per_byte_bit_cap=8,
        )
        # 10 / 3 = 3 base + 1 remainder distributed to lowest-index byte.
        assert plan.bits_per_byte == {0: 4, 1: 3, 2: 3}

    def test_uniform_caps_at_per_byte_bit_cap(self) -> None:
        plan = allocate_per_byte(
            total_budget_bits=100,
            sensitivity_per_byte={0: 1.0, 1: 1.0},
            method=PerByteAllocationMethod.UNIFORM_BASELINE,
            per_byte_bit_cap=8,
        )
        # 50/byte exceeds cap of 8; residual is 100 - 16 = 84.
        assert plan.bits_per_byte == {0: 8, 1: 8}
        assert plan.residual_bits == 84


class TestPerByteInvariants:
    def test_canonical_equation_id_cited(self) -> None:
        plan = allocate_per_byte(
            total_budget_bits=8,
            sensitivity_per_byte={0: 1.0},
        )
        assert plan.canonical_equation_id == PER_BYTE_CANONICAL_EQUATION_ID
        assert (
            PER_BYTE_CANONICAL_EQUATION_ID == "per_byte_leverage_uniformly_distributed_v1"
        )

    def test_canonical_provenance_predicted_grade(self) -> None:
        plan = allocate_per_byte(
            total_budget_bits=8,
            sensitivity_per_byte={0: 1.0, 1: 2.0},
        )
        assert plan.provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED
        assert plan.provenance.artifact_kind == ProvenanceKind.PREDICTED_FROM_MODEL
        assert plan.provenance.promotion_eligible is False
        assert plan.provenance.score_claim_valid is False

    def test_score_claim_invariant_refused(self) -> None:
        plan = allocate_per_byte(
            total_budget_bits=8,
            sensitivity_per_byte={0: 1.0, 1: 2.0},
        )
        with pytest.raises(PerByteAllocationError, match="score_claim must be False"):
            PerByteAllocationPlan(
                bits_per_byte=dict(plan.bits_per_byte),
                method=plan.method,
                total_budget_bits=plan.total_budget_bits,
                residual_bits=plan.residual_bits,
                n_bytes_allocated=plan.n_bytes_allocated,
                n_bytes_in_scope=plan.n_bytes_in_scope,
                per_byte_bit_cap=plan.per_byte_bit_cap,
                canonical_equation_id=plan.canonical_equation_id,
                provenance=plan.provenance,
                score_claim=True,
            )

    def test_axis_tag_must_be_predicted(self) -> None:
        plan = allocate_per_byte(
            total_budget_bits=8,
            sensitivity_per_byte={0: 1.0, 1: 2.0},
        )
        with pytest.raises(PerByteAllocationError, match="axis_tag must be"):
            PerByteAllocationPlan(
                bits_per_byte=dict(plan.bits_per_byte),
                method=plan.method,
                total_budget_bits=plan.total_budget_bits,
                residual_bits=plan.residual_bits,
                n_bytes_allocated=plan.n_bytes_allocated,
                n_bytes_in_scope=plan.n_bytes_in_scope,
                per_byte_bit_cap=plan.per_byte_bit_cap,
                canonical_equation_id=plan.canonical_equation_id,
                provenance=plan.provenance,
                axis_tag="[contest-CUDA]",
            )

    def test_method_string_coerces_to_enum(self) -> None:
        plan = allocate_per_byte(
            total_budget_bits=8,
            sensitivity_per_byte={0: 1.0},
            method="top_k_by_sensitivity",
        )
        assert plan.method is PerByteAllocationMethod.TOP_K_BY_SENSITIVITY

    def test_empty_sensitivity_raises(self) -> None:
        with pytest.raises(PerByteAllocationError, match="at least one byte"):
            allocate_per_byte(total_budget_bits=8, sensitivity_per_byte={})

    def test_negative_sensitivity_raises(self) -> None:
        with pytest.raises(PerByteAllocationError, match="non-negative"):
            allocate_per_byte(
                total_budget_bits=8, sensitivity_per_byte={0: -1.0}
            )

    def test_non_finite_sensitivity_raises(self) -> None:
        with pytest.raises(PerByteAllocationError, match="finite"):
            allocate_per_byte(
                total_budget_bits=8, sensitivity_per_byte={0: math.nan}
            )

    def test_as_dict_json_serializable(self) -> None:
        plan = allocate_per_byte(
            total_budget_bits=16,
            sensitivity_per_byte={0: 1.0, 1: 2.0, 2: 4.0},
            method=PerByteAllocationMethod.TOP_K_BY_SENSITIVITY,
            top_k=2,
        )
        payload = plan.as_dict()
        blob = json.dumps(payload, sort_keys=True)
        parsed = json.loads(blob)
        assert parsed["score_claim"] is False
        assert parsed["axis_tag"] == "[predicted]"
        assert parsed["canonical_equation_id"] == PER_BYTE_CANONICAL_EQUATION_ID


# =====================================================================
# per_class allocator
# =====================================================================


class TestPerClassAllocator:
    def test_uniform_strategy_distributes_equally(self) -> None:
        plan = allocate_per_class(
            total_budget_bits=500,
            prior_per_class={0: 1.0, 1: 99.0, 2: 50.0, 3: 1.0, 4: 1.0},
            strategy=PerClassAllocationStrategy.UNIFORM,
        )
        # 500/5 = 100 each.
        assert plan.bits_per_class == {0: 100, 1: 100, 2: 100, 3: 100, 4: 100}

    def test_proportional_strategy(self) -> None:
        plan = allocate_per_class(
            total_budget_bits=1000,
            prior_per_class={0: 1.0, 1: 4.0, 2: 5.0, 3: 0.0, 4: 0.0},
            strategy=PerClassAllocationStrategy.PROPORTIONAL,
        )
        assert plan.bits_per_class == {0: 100, 1: 400, 2: 500, 3: 0, 4: 0}

    def test_sqrt_strategy(self) -> None:
        plan = allocate_per_class(
            total_budget_bits=500,
            prior_per_class={0: 1.0, 1: 4.0, 2: 9.0, 3: 16.0, 4: 25.0},
            strategy=PerClassAllocationStrategy.SQRT,
        )
        # sqrt = (1, 2, 3, 4, 5), sum = 15; budgets = (33.33, 66.66, 100, 133.33, 166.66)
        # Hamilton: floors (33, 66, 100, 133, 166), residual 2 to largest fracs (1, 2 indices first).
        assert sum(plan.bits_per_class.values()) == 500
        # 0,1 fractions are 0.33, 0.66; 2 is exact; 3,4 are 0.33, 0.66.
        # Top two fractions (largest first, tie-break lower index): 4 then 1.
        assert plan.bits_per_class[4] == 167
        assert plan.bits_per_class[1] == 67

    def test_default_5_class_segnet(self) -> None:
        plan = allocate_per_class(
            total_budget_bits=500,
            prior_per_class={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        )
        assert plan.n_classes == SEGNET_CLASS_COUNT
        assert plan.notes["class_names"] == list(CANONICAL_SEGNET_CLASS_NAMES)

    def test_missing_class_index_gets_zero_prior(self) -> None:
        plan = allocate_per_class(
            total_budget_bits=400,
            prior_per_class={1: 1.0, 3: 1.0},  # only 2 of 5 classes
            strategy=PerClassAllocationStrategy.PROPORTIONAL,
        )
        # Classes 0, 2, 4 have prior 0; 1, 3 share 400.
        assert plan.bits_per_class[0] == 0
        assert plan.bits_per_class[2] == 0
        assert plan.bits_per_class[4] == 0
        assert plan.bits_per_class[1] == 200
        assert plan.bits_per_class[3] == 200

    def test_canonical_provenance_predicted(self) -> None:
        plan = allocate_per_class(
            total_budget_bits=100,
            prior_per_class={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        )
        assert plan.provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED
        assert plan.provenance.artifact_kind == ProvenanceKind.PREDICTED_FROM_MODEL

    def test_score_claim_invariant_refused(self) -> None:
        plan = allocate_per_class(
            total_budget_bits=100,
            prior_per_class={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        )
        with pytest.raises(PerClassAllocationError, match="score_claim must be False"):
            PerClassBitAllocationPlan(
                bits_per_class=dict(plan.bits_per_class),
                strategy=plan.strategy,
                total_budget_bits=plan.total_budget_bits,
                n_classes=plan.n_classes,
                provenance=plan.provenance,
                score_claim=True,
            )

    def test_class_index_out_of_range_raises(self) -> None:
        with pytest.raises(PerClassAllocationError, match="out of range"):
            allocate_per_class(
                total_budget_bits=100,
                prior_per_class={99: 1.0},
                n_classes=5,
            )

    def test_empty_prior_raises(self) -> None:
        with pytest.raises(PerClassAllocationError, match="at least one class"):
            allocate_per_class(total_budget_bits=100, prior_per_class={})


# =====================================================================
# per_axis allocator
# =====================================================================


class TestPerAxisAllocator:
    def test_canonical_axes_pinned(self) -> None:
        assert CANONICAL_SCORER_AXES == ("seg", "pose", "rate")

    def test_canonical_scorer_coefficients_pinned(self) -> None:
        assert CANONICAL_SCORER_COEFFICIENTS_FORMULA_WEIGHTED["seg"] == 100.0
        assert (
            CANONICAL_SCORER_COEFFICIENTS_FORMULA_WEIGHTED["pose"]
            == math.sqrt(10.0)
        )
        assert CANONICAL_SCORER_COEFFICIENTS_FORMULA_WEIGHTED["rate"] == 25.0

    def test_uniform_strategy_distributes_equally(self) -> None:
        plan = allocate_per_axis(
            total_budget_bits=300,
            strategy=PerAxisAllocationStrategy.UNIFORM,
        )
        assert plan.bits_per_axis == {"seg": 100, "pose": 100, "rate": 100}

    def test_scorer_formula_weighted_strategy(self) -> None:
        plan = allocate_per_axis(
            total_budget_bits=300,
            strategy=PerAxisAllocationStrategy.SCORER_FORMULA_WEIGHTED,
        )
        # Weights = (100, sqrt(10)=3.162, 25); sum = 128.16
        # Bits should be heavily seg-weighted.
        assert plan.bits_per_axis["seg"] > plan.bits_per_axis["rate"]
        assert plan.bits_per_axis["rate"] > plan.bits_per_axis["pose"]
        assert sum(plan.bits_per_axis.values()) == 300

    def test_sensitivity_weighted_requires_sensitivity(self) -> None:
        with pytest.raises(PerAxisAllocationError, match="requires sensitivity"):
            allocate_per_axis(
                total_budget_bits=300,
                strategy=PerAxisAllocationStrategy.SENSITIVITY_WEIGHTED,
            )

    def test_sensitivity_weighted_uses_provided_priors(self) -> None:
        # Operating-point-aware allocation per CLAUDE.md
        # "SegNet vs PoseNet importance — operating-point dependent":
        # at PR106 frontier, pose marginal is 2.71x SegNet's.
        plan = allocate_per_axis(
            total_budget_bits=396,  # sum of (100 + 271 + 25) = 396
            strategy=PerAxisAllocationStrategy.SENSITIVITY_WEIGHTED,
            sensitivity_per_axis={"seg": 100.0, "pose": 271.0, "rate": 25.0},
        )
        assert plan.bits_per_axis["seg"] == 100
        assert plan.bits_per_axis["pose"] == 271
        assert plan.bits_per_axis["rate"] == 25

    def test_operating_point_aware_flag_in_notes(self) -> None:
        plan = allocate_per_axis(
            total_budget_bits=300,
            strategy=PerAxisAllocationStrategy.SENSITIVITY_WEIGHTED,
            sensitivity_per_axis={"seg": 1.0, "pose": 1.0, "rate": 1.0},
        )
        assert plan.notes["operating_point_aware"] is True

    def test_canonical_provenance_predicted(self) -> None:
        plan = allocate_per_axis(total_budget_bits=300)
        assert plan.provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED

    def test_score_claim_invariant_refused(self) -> None:
        plan = allocate_per_axis(total_budget_bits=300)
        with pytest.raises(PerAxisAllocationError, match="score_claim must be False"):
            PerAxisBitAllocationPlan(
                bits_per_axis=dict(plan.bits_per_axis),
                strategy=plan.strategy,
                total_budget_bits=plan.total_budget_bits,
                axes=plan.axes,
                provenance=plan.provenance,
                score_claim=True,
            )

    def test_non_canonical_axis_name_raises(self) -> None:
        with pytest.raises(PerAxisAllocationError, match="not in CANONICAL_SCORER_AXES"):
            allocate_per_axis(
                total_budget_bits=100,
                strategy=PerAxisAllocationStrategy.SENSITIVITY_WEIGHTED,
                sensitivity_per_axis={"fake_axis": 1.0},
            )

    def test_axes_subset_works(self) -> None:
        # Operator may want seg+pose only (no rate axis).
        plan = allocate_per_axis(
            total_budget_bits=200,
            strategy=PerAxisAllocationStrategy.UNIFORM,
            axes=("seg", "pose"),
        )
        assert plan.bits_per_axis == {"seg": 100, "pose": 100}
        assert "rate" not in plan.bits_per_axis


# =====================================================================
# pareto_dual allocator
# =====================================================================


class TestParetoDualLagrangian:
    def test_lagrangian_solves_kkt_under_budget(self) -> None:
        plan = allocate_via_lagrangian_dual(
            total_budget_bits=20,
            sensitivity_per_element={0: 5.0, 1: 0.1, 2: 9.0, 3: 0.5},
            min_bits=0,
            max_bits=8,
        )
        assert plan.method is ParetoDualMethod.LAGRANGIAN_DUAL
        assert plan.is_pareto_feasible is True
        assert plan.lagrangian_lambda > 0.0
        # High-sensitivity bytes get more bits.
        assert plan.bits_per_element[2] >= plan.bits_per_element[0]
        assert plan.bits_per_element[0] >= plan.bits_per_element[3]
        assert plan.bits_per_element[3] >= plan.bits_per_element[1]
        assert sum(plan.bits_per_element.values()) <= 20

    def test_lagrangian_kkt_residual_small_for_interior_optima(self) -> None:
        plan = allocate_via_lagrangian_dual(
            total_budget_bits=20,
            sensitivity_per_element={i: float(i + 1) for i in range(8)},
            min_bits=0,
            max_bits=8,
        )
        # KKT residual measures variance of u'(b) across interior elements.
        # For concave utility + bisected lambda, residual should be small.
        assert plan.kkt_residual < 1.0  # generous bound; tighter for larger N

    def test_lagrangian_ceiling_fits_returns_full_max(self) -> None:
        plan = allocate_via_lagrangian_dual(
            total_budget_bits=100,
            sensitivity_per_element={0: 1.0, 1: 2.0, 2: 3.0},
            min_bits=0,
            max_bits=8,
        )
        # 3 elements * 8 max = 24 <= 100; everyone gets 8.
        assert all(b == 8 for b in plan.bits_per_element.values())
        assert plan.residual_bits == 100 - 24

    def test_lagrangian_floor_total_too_high_raises(self) -> None:
        with pytest.raises(ParetoDualError, match="infeasible"):
            allocate_via_lagrangian_dual(
                total_budget_bits=2,
                sensitivity_per_element={0: 1.0, 1: 1.0, 2: 1.0},
                min_bits=2,
                max_bits=8,
            )

    def test_lagrangian_all_zero_sensitivity_uniform_floor(self) -> None:
        plan = allocate_via_lagrangian_dual(
            total_budget_bits=10,
            sensitivity_per_element={0: 0.0, 1: 0.0, 2: 0.0},
            min_bits=1,
            max_bits=8,
        )
        # All zero sensitivity → uniform floor allocation.
        assert all(b == 1 for b in plan.bits_per_element.values())
        assert plan.residual_bits == 10 - 3


class TestParetoDualDykstra:
    def test_dykstra_projection_returns_feasible(self) -> None:
        plan = allocate_via_lagrangian_dual(
            total_budget_bits=20,
            sensitivity_per_element={0: 5.0, 1: 0.1, 2: 9.0, 3: 0.5},
            method=ParetoDualMethod.DYKSTRA_PROJECTION,
            min_bits=0,
            max_bits=8,
        )
        assert plan.method is ParetoDualMethod.DYKSTRA_PROJECTION
        assert plan.is_pareto_feasible is True
        assert sum(plan.bits_per_element.values()) <= 20
        # All bits in box.
        for b in plan.bits_per_element.values():
            assert 0 <= b <= 8

    def test_dykstra_respects_box(self) -> None:
        plan = allocate_via_lagrangian_dual(
            total_budget_bits=100,
            sensitivity_per_element={0: 1.0, 1: 1.0, 2: 1.0},
            method=ParetoDualMethod.DYKSTRA_PROJECTION,
            min_bits=2,
            max_bits=6,
        )
        for b in plan.bits_per_element.values():
            assert 2 <= b <= 6


class TestParetoDualInvariants:
    def test_default_bisection_iters_pinned(self) -> None:
        assert DEFAULT_BISECTION_ITERS == 64

    def test_canonical_provenance_predicted(self) -> None:
        plan = allocate_via_lagrangian_dual(
            total_budget_bits=10,
            sensitivity_per_element={0: 1.0, 1: 2.0},
        )
        assert plan.provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED
        assert plan.provenance.artifact_kind == ProvenanceKind.PREDICTED_FROM_MODEL

    def test_meta_lagrangian_reference_cited_in_notes(self) -> None:
        plan = allocate_via_lagrangian_dual(
            total_budget_bits=10,
            sensitivity_per_element={0: 1.0, 1: 2.0},
        )
        # Cites the META Lagrangian per CLAUDE.md non-negotiable.
        assert (
            plan.notes["meta_lagrangian_reference"]
            == "tac.findings_lagrangian.compute_findings_lagrangian"
        )

    def test_score_claim_invariant_refused(self) -> None:
        plan = allocate_via_lagrangian_dual(
            total_budget_bits=10,
            sensitivity_per_element={0: 1.0, 1: 2.0},
        )
        with pytest.raises(ParetoDualError, match="score_claim must be False"):
            ParetoDualBitAllocationPlan(
                bits_per_element=dict(plan.bits_per_element),
                method=plan.method,
                total_budget_bits=plan.total_budget_bits,
                residual_bits=plan.residual_bits,
                min_bits=plan.min_bits,
                max_bits=plan.max_bits,
                lagrangian_lambda=plan.lagrangian_lambda,
                kkt_residual=plan.kkt_residual,
                n_elements=plan.n_elements,
                is_pareto_feasible=plan.is_pareto_feasible,
                provenance=plan.provenance,
                score_claim=True,
            )

    def test_method_string_coerces_to_enum(self) -> None:
        plan = allocate_via_lagrangian_dual(
            total_budget_bits=10,
            sensitivity_per_element={0: 1.0, 1: 2.0},
            method="dykstra_projection",
        )
        assert plan.method is ParetoDualMethod.DYKSTRA_PROJECTION

    def test_unknown_method_string_raises(self) -> None:
        with pytest.raises(ParetoDualError, match="unknown method"):
            allocate_via_lagrangian_dual(
                total_budget_bits=10,
                sensitivity_per_element={0: 1.0},
                method="newton_raphson",
            )

    def test_invalid_bit_range_raises(self) -> None:
        with pytest.raises(ParetoDualError, match="invalid bit range"):
            allocate_via_lagrangian_dual(
                total_budget_bits=10,
                sensitivity_per_element={0: 1.0},
                min_bits=8,
                max_bits=0,
            )

    def test_empty_sensitivity_raises(self) -> None:
        with pytest.raises(ParetoDualError, match="at least one element"):
            allocate_via_lagrangian_dual(
                total_budget_bits=10,
                sensitivity_per_element={},
            )


# =====================================================================
# Cross-allocator namespace invariants
# =====================================================================


class TestNamespaceInvariants:
    def test_all_allocators_return_non_promotable_axis_tag(self) -> None:
        """Catalog #323 cross-allocator non-promotable contract.

        Every allocator MUST return axis_tag == "[predicted]" so downstream
        autopilot consumers cannot silently promote a bit-allocation plan
        to a contest score claim.
        """
        plans = [
            allocate_per_byte(total_budget_bits=8, sensitivity_per_byte={0: 1.0}),
            allocate_per_class(
                total_budget_bits=100,
                prior_per_class={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            ),
            allocate_per_axis(total_budget_bits=100),
            allocate_via_lagrangian_dual(
                total_budget_bits=10, sensitivity_per_element={0: 1.0, 1: 2.0}
            ),
        ]
        for plan in plans:
            assert plan.axis_tag == "[predicted]"
            assert plan.score_claim is False
            assert plan.promotion_eligible is False

    def test_all_allocators_carry_canonical_provenance(self) -> None:
        """Catalog #323 cross-allocator Provenance contract."""
        plans = [
            allocate_per_byte(total_budget_bits=8, sensitivity_per_byte={0: 1.0}),
            allocate_per_class(
                total_budget_bits=100,
                prior_per_class={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            ),
            allocate_per_axis(total_budget_bits=100),
            allocate_via_lagrangian_dual(
                total_budget_bits=10, sensitivity_per_element={0: 1.0, 1: 2.0}
            ),
        ]
        for plan in plans:
            assert plan.provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED
            assert (
                plan.provenance.artifact_kind == ProvenanceKind.PREDICTED_FROM_MODEL
            )
            assert plan.provenance.promotion_eligible is False
            assert plan.provenance.score_claim_valid is False

    def test_all_allocators_archive_sha_threads_into_provenance(self) -> None:
        """Archive sha attribution flows through into Provenance source_sha256."""
        a = allocate_per_byte(
            total_budget_bits=8,
            sensitivity_per_byte={0: 1.0, 1: 2.0},
            archive_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
            captured_at_utc="2026-05-20T00:00:00+00:00",
        )
        b = allocate_per_byte(
            total_budget_bits=8,
            sensitivity_per_byte={0: 1.0, 1: 2.0},
            archive_sha256=None,
            captured_at_utc="2026-05-20T00:00:00+00:00",
        )
        # Different archive sha → different inputs hash → different source_sha256.
        assert a.provenance.source_sha256 != b.provenance.source_sha256

    def test_all_allocators_as_dict_json_serializable(self) -> None:
        plans = [
            allocate_per_byte(total_budget_bits=8, sensitivity_per_byte={0: 1.0}),
            allocate_per_class(
                total_budget_bits=100,
                prior_per_class={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            ),
            allocate_per_axis(total_budget_bits=100),
            allocate_via_lagrangian_dual(
                total_budget_bits=10, sensitivity_per_element={0: 1.0, 1: 2.0}
            ),
        ]
        for plan in plans:
            payload = plan.as_dict()
            blob = json.dumps(payload, sort_keys=True)
            parsed = json.loads(blob)
            assert parsed["score_claim"] is False
            assert parsed["axis_tag"] == "[predicted]"

    def test_all_allocators_deterministic_across_calls(self) -> None:
        """Same inputs → identical outputs (determinism contract)."""
        a1 = allocate_per_byte(
            total_budget_bits=20,
            sensitivity_per_byte={0: 5.0, 1: 0.1, 2: 9.0},
            captured_at_utc="2026-05-20T00:00:00+00:00",
        )
        a2 = allocate_per_byte(
            total_budget_bits=20,
            sensitivity_per_byte={0: 5.0, 1: 0.1, 2: 9.0},
            captured_at_utc="2026-05-20T00:00:00+00:00",
        )
        assert a1.bits_per_byte == a2.bits_per_byte
        assert a1.provenance.source_sha256 == a2.provenance.source_sha256

        c1 = allocate_via_lagrangian_dual(
            total_budget_bits=20,
            sensitivity_per_element={0: 5.0, 1: 0.1, 2: 9.0},
            captured_at_utc="2026-05-20T00:00:00+00:00",
        )
        c2 = allocate_via_lagrangian_dual(
            total_budget_bits=20,
            sensitivity_per_element={0: 5.0, 1: 0.1, 2: 9.0},
            captured_at_utc="2026-05-20T00:00:00+00:00",
        )
        assert c1.bits_per_element == c2.bits_per_element
        assert c1.lagrangian_lambda == c2.lagrangian_lambda

    def test_per_byte_cites_canonical_equation_registry(self) -> None:
        """Per CLAUDE.md "Canonical equations + models registry — non-negotiable":
        the per_byte allocator cites the canonical equation
        ``per_byte_leverage_uniformly_distributed_v1`` so downstream
        equation-registry consumers can recalibrate the allocator on new
        empirical anchors.
        """
        plan = allocate_per_byte(
            total_budget_bits=8, sensitivity_per_byte={0: 1.0}
        )
        assert (
            plan.canonical_equation_id
            == "per_byte_leverage_uniformly_distributed_v1"
        )
        # The equation IS importable from the canonical registry.
        from tac.canonical_equations import get_equation_by_id

        equation = get_equation_by_id(plan.canonical_equation_id)
        assert equation is not None
        assert equation.equation_id == plan.canonical_equation_id


class TestNamespacePublicAPI:
    def test_per_byte_allocator_exported(self) -> None:
        from tac.bit_allocator import (
            PerByteAllocationMethod,
            PerByteAllocationPlan,
            allocate_per_byte,
        )

        assert callable(allocate_per_byte)
        assert PerByteAllocationMethod.TOP_K_BY_SENSITIVITY.value == "top_k_by_sensitivity"
        assert PerByteAllocationPlan is not None

    def test_per_class_allocator_exported(self) -> None:
        from tac.bit_allocator import (
            PerClassAllocationStrategy,
            PerClassBitAllocationPlan,
            allocate_per_class,
        )

        assert callable(allocate_per_class)
        assert PerClassAllocationStrategy.SQRT.value == "sqrt"
        assert PerClassBitAllocationPlan is not None

    def test_per_axis_allocator_exported(self) -> None:
        from tac.bit_allocator import (
            PerAxisAllocationStrategy,
            PerAxisBitAllocationPlan,
            allocate_per_axis,
        )

        assert callable(allocate_per_axis)
        assert (
            PerAxisAllocationStrategy.SENSITIVITY_WEIGHTED.value
            == "sensitivity_weighted"
        )
        assert PerAxisBitAllocationPlan is not None

    def test_pareto_dual_allocator_exported(self) -> None:
        from tac.bit_allocator import (
            ParetoDualBitAllocationPlan,
            ParetoDualMethod,
            allocate_via_lagrangian_dual,
        )

        assert callable(allocate_via_lagrangian_dual)
        assert ParetoDualMethod.LAGRANGIAN_DUAL.value == "lagrangian_dual"
        assert ParetoDualBitAllocationPlan is not None

    def test_legacy_per_pair_still_exported(self) -> None:
        """Sister-extension does not break the existing per_pair surface."""
        from tac.bit_allocator import (
            AllocationStrategy,
            BitAllocationResult,
            allocate_bits_per_pair,
        )

        plan = allocate_bits_per_pair(
            total_bits=100,
            difficulty_per_pair={0: 1.0, 1: 1.0},
            strategy=AllocationStrategy.UNIFORM,
        )
        assert isinstance(plan, BitAllocationResult)
        assert sum(plan.bits_per_pair.values()) == 100
