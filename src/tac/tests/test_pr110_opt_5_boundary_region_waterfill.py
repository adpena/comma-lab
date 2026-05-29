# SPDX-License-Identifier: MIT
"""Tests for PR110-OPT-5 Boundary/region waterfill SegNet class-region-aware perturbation budget L0 SCAFFOLD.

Per Slot TT cap=3 parallel-cascade canonical Fridrich-Yousfi 3-axis cascade
Axis 3 + design memo::

    .omx/research/pr110_opt_5_boundary_region_waterfill_segnet_class_region_aware_\
perturbation_budget_canonical_fridrich_yousfi_3_axis_cascade_design_20260529.md

Test coverage:

- Canonical anchor constants preserved per Catalog #110/#113 HISTORICAL_PROVENANCE.
- Enum (4 BoundaryRegionWaterfillStrategy values per Catalog #308).
- Canonical Config dataclass invariants (frozen + ValueError on each bad path).
- ``compute_uniward_weighted_per_class_budget`` correctness across all 4 strategies
  + canonical analytical primitive regression.
- ``apply_boundary_region_waterfill_to_pr110_archive`` Tier A canonical-routing
  markers contract per Catalog #341 + #357.
- Canonical :class:`AxisDecomposition` emission per Catalog #356 + canonical
  :class:`Provenance` per Catalog #323.
- Per-substrate empirical verification stub per Slot QQ canonical META-LESSON.
- Canonical Fridrich-Yousfi 3-axis cascade Axis 3 anchor + sister cross-references.
- Slot CC dissent binding revision anchor preservation.
- Parametrized strategy dispatch (4-strategy paired-comparison; HARDCODED
  smoke fixture per Slot QQ canonical EMPIRICAL VERIFICATION discipline).
"""

from __future__ import annotations

import hashlib
import math
import random

import pytest

from tac.cathedral.consumer_contract import AxisDecomposition
from tac.composition.pr110_opt_5_boundary_region_waterfill import (
    BoundaryRegionWaterfillConfig,
    BoundaryRegionWaterfillStrategy,
    CANONICAL_BOUNDARY_K_DEFAULT,
    CANONICAL_HEADER_OVERHEAD_BYTES,
    CANONICAL_N_PAIRS,
    CANONICAL_PER_CLASS_UNIFORM_DELTA_BYTES_VS_FEC6,
    CANONICAL_PER_CLASS_UNIFORM_PROPORTIONAL_SAVINGS,
    CANONICAL_PER_CLASS_UNIFORM_WIRE_BYTES,
    CANONICAL_PER_CLASS_WEIGHTED_BY_AREA_WIRE_BYTES,
    CANONICAL_PER_REGION_AT_BOUNDARY_WIRE_BYTES,
    CANONICAL_PER_REGION_INTERIOR_WIRE_BYTES,
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
    FEC6_BASELINE_WIRE_BYTES,
    SEGNET_N_CLASSES,
    apply_boundary_region_waterfill_to_pr110_archive,
    compute_uniward_weighted_per_class_budget,
)
from tac.composition.pr110_opt_5_boundary_region_waterfill import (
    _compute_class_region_signature,
    _compute_per_substrate_empirical_verification,
    _compute_uniward_cost_per_class,
    _select_per_class_budgets,
)
from tac.provenance.contract import Provenance


def _make_canonical_inputs(seed: int = 42) -> list[float]:
    """Synthetic SegNet 5-class response (positive floats)."""
    rng = random.Random(seed)
    return [abs(rng.gauss(0.5, 0.3)) for _ in range(SEGNET_N_CLASSES)]


# -----------------------------------------------------------------------------
# Section 1: Canonical anchor constants preserved per Catalog #110/#113
# -----------------------------------------------------------------------------


class TestCanonicalAnchorConstantsPreserved:
    def test_fec6_baseline_249_bytes(self) -> None:
        assert FEC6_BASELINE_WIRE_BYTES == 249

    def test_segnet_n_classes_5(self) -> None:
        assert SEGNET_N_CLASSES == 5

    def test_canonical_n_pairs_600(self) -> None:
        assert CANONICAL_N_PAIRS == 600

    def test_canonical_header_overhead_6(self) -> None:
        assert CANONICAL_HEADER_OVERHEAD_BYTES == 6

    def test_canonical_per_class_uniform_wire_11(self) -> None:
        assert CANONICAL_PER_CLASS_UNIFORM_WIRE_BYTES == 11
        # Verifies analytical: 6-byte header + 5-byte per-class budget
        assert (
            CANONICAL_PER_CLASS_UNIFORM_WIRE_BYTES
            == CANONICAL_HEADER_OVERHEAD_BYTES + SEGNET_N_CLASSES
        )

    def test_canonical_per_class_weighted_by_area_wire_16(self) -> None:
        assert CANONICAL_PER_CLASS_WEIGHTED_BY_AREA_WIRE_BYTES == 16

    def test_canonical_per_region_at_boundary_wire_91(self) -> None:
        assert CANONICAL_PER_REGION_AT_BOUNDARY_WIRE_BYTES == 91

    def test_canonical_per_region_interior_wire_131(self) -> None:
        assert CANONICAL_PER_REGION_INTERIOR_WIRE_BYTES == 131

    def test_canonical_per_class_uniform_delta_neg_238(self) -> None:
        assert CANONICAL_PER_CLASS_UNIFORM_DELTA_BYTES_VS_FEC6 == -238
        assert (
            CANONICAL_PER_CLASS_UNIFORM_DELTA_BYTES_VS_FEC6
            == CANONICAL_PER_CLASS_UNIFORM_WIRE_BYTES - FEC6_BASELINE_WIRE_BYTES
        )

    def test_canonical_per_class_uniform_proportional_savings_canonical(self) -> None:
        expected = 25.0 * -238 / 37_545_489
        assert CANONICAL_PER_CLASS_UNIFORM_PROPORTIONAL_SAVINGS == pytest.approx(
            expected, rel=1e-12
        )

    def test_canonical_boundary_k_default_20(self) -> None:
        assert CANONICAL_BOUNDARY_K_DEFAULT == 20

    def test_canonical_rate_multiplier_25(self) -> None:
        assert CANONICAL_RATE_MULTIPLIER == 25.0

    def test_canonical_rate_denom_37_545_489(self) -> None:
        assert CANONICAL_RATE_DENOM_BYTES == 37_545_489


# -----------------------------------------------------------------------------
# Section 2: BoundaryRegionWaterfillStrategy enum (Catalog #308)
# -----------------------------------------------------------------------------


class TestBoundaryRegionWaterfillStrategyEnum:
    def test_per_class_uniform_member(self) -> None:
        assert (
            BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM.value
            == "per_class_uniform"
        )

    def test_per_class_weighted_by_area_member(self) -> None:
        assert (
            BoundaryRegionWaterfillStrategy.PER_CLASS_WEIGHTED_BY_AREA.value
            == "per_class_weighted_by_area"
        )

    def test_per_region_at_boundary_member(self) -> None:
        assert (
            BoundaryRegionWaterfillStrategy.PER_REGION_AT_BOUNDARY.value
            == "per_region_at_boundary"
        )

    def test_per_region_interior_member(self) -> None:
        assert (
            BoundaryRegionWaterfillStrategy.PER_REGION_INTERIOR.value
            == "per_region_interior"
        )

    def test_enum_has_4_members(self) -> None:
        assert len(list(BoundaryRegionWaterfillStrategy)) == 4

    def test_str_enum_isinstance(self) -> None:
        assert isinstance(BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM, str)


# -----------------------------------------------------------------------------
# Section 3: BoundaryRegionWaterfillConfig dataclass invariants
# -----------------------------------------------------------------------------


class TestBoundaryRegionWaterfillConfigInvariants:
    def test_default_construction(self) -> None:
        cfg = BoundaryRegionWaterfillConfig()
        assert cfg.strategy == BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM
        assert cfg.n_classes == 5
        assert cfg.n_pairs == 600
        assert cfg.boundary_k == 20
        assert cfg.uniward_epsilon == 1e-6
        assert cfg.header_overhead_bytes == 6
        assert cfg.emit_axis_decomposition is True

    def test_frozen(self) -> None:
        cfg = BoundaryRegionWaterfillConfig()
        with pytest.raises(Exception):
            cfg.strategy = BoundaryRegionWaterfillStrategy.PER_REGION_INTERIOR

    def test_invalid_strategy_type(self) -> None:
        with pytest.raises(ValueError, match="strategy must be"):
            BoundaryRegionWaterfillConfig(strategy="per_class_uniform")

    def test_invalid_n_classes_type(self) -> None:
        with pytest.raises(ValueError, match="n_classes must be int"):
            BoundaryRegionWaterfillConfig(n_classes="5")

    def test_invalid_n_classes_bool(self) -> None:
        with pytest.raises(ValueError, match="n_classes must be int"):
            BoundaryRegionWaterfillConfig(n_classes=True)

    def test_invalid_n_classes_zero(self) -> None:
        with pytest.raises(ValueError, match="n_classes must be > 0"):
            BoundaryRegionWaterfillConfig(n_classes=0)

    def test_invalid_n_classes_mismatch_segnet(self) -> None:
        with pytest.raises(ValueError, match="n_classes must equal SEGNET_N_CLASSES"):
            BoundaryRegionWaterfillConfig(n_classes=10)

    def test_invalid_n_pairs_zero(self) -> None:
        with pytest.raises(ValueError, match="n_pairs must be > 0"):
            BoundaryRegionWaterfillConfig(n_pairs=0)

    def test_invalid_n_pairs_bool(self) -> None:
        with pytest.raises(ValueError, match="n_pairs must be int"):
            BoundaryRegionWaterfillConfig(n_pairs=True)

    def test_invalid_boundary_k_zero(self) -> None:
        with pytest.raises(ValueError, match="boundary_k must be > 0"):
            BoundaryRegionWaterfillConfig(boundary_k=0)

    def test_invalid_boundary_k_negative(self) -> None:
        with pytest.raises(ValueError, match="boundary_k must be > 0"):
            BoundaryRegionWaterfillConfig(boundary_k=-1)

    def test_invalid_uniward_epsilon_zero(self) -> None:
        with pytest.raises(ValueError, match="uniward_epsilon must be > 0"):
            BoundaryRegionWaterfillConfig(uniward_epsilon=0.0)

    def test_invalid_uniward_epsilon_bool(self) -> None:
        with pytest.raises(ValueError, match="uniward_epsilon must be number not bool"):
            BoundaryRegionWaterfillConfig(uniward_epsilon=True)

    def test_invalid_header_overhead_negative(self) -> None:
        with pytest.raises(ValueError, match="header_overhead_bytes must be >= 0"):
            BoundaryRegionWaterfillConfig(header_overhead_bytes=-1)


# -----------------------------------------------------------------------------
# Section 4: _compute_uniward_cost_per_class canonical Fridrich primitive
# -----------------------------------------------------------------------------


class TestComputeUniwardCostPerClass:
    def test_inverse_response_canonical(self) -> None:
        # cost(c) = 1 / (epsilon + response(c))
        response = [0.1, 0.2, 0.3, 0.4, 0.5]
        epsilon = 1e-6
        costs = _compute_uniward_cost_per_class(response, epsilon)
        for i, r in enumerate(response):
            assert costs[i] == pytest.approx(1.0 / (epsilon + r), rel=1e-12)

    def test_higher_response_lower_cost(self) -> None:
        # Higher per-class response => higher detectability => LOWER cost weight
        response = [0.1, 0.5, 1.0]
        costs = _compute_uniward_cost_per_class(response, 1e-6)
        assert costs[0] > costs[1] > costs[2]

    def test_zero_response_clamped_to_zero(self) -> None:
        # max(0.0, response) clamping
        response = [-0.1, 0.0, 0.1]
        epsilon = 1e-6
        costs = _compute_uniward_cost_per_class(response, epsilon)
        # All -0.1 and 0.0 should yield same cost (clamped)
        assert costs[0] == costs[1]
        assert costs[0] == pytest.approx(1.0 / epsilon, rel=1e-12)

    def test_canonical_5_class_canonical_response(self) -> None:
        canonical = _make_canonical_inputs(seed=42)
        assert len(canonical) == 5
        costs = _compute_uniward_cost_per_class(canonical, 1e-6)
        assert len(costs) == 5
        assert all(c > 0 for c in costs)


# -----------------------------------------------------------------------------
# Section 5: _select_per_class_budgets canonical per-strategy
# -----------------------------------------------------------------------------


class TestSelectPerClassBudgets:
    def test_per_class_uniform_returns_uniform_1byte(self) -> None:
        costs = [1.0, 2.0, 3.0, 4.0, 5.0]
        budgets = _select_per_class_budgets(
            costs, BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM
        )
        assert budgets == [1, 1, 1, 1, 1]
        assert sum(budgets) == 5  # 1 byte per class

    def test_per_class_weighted_by_area_returns_weighted_budgets(self) -> None:
        # Equal costs => equal budgets
        costs = [1.0, 1.0, 1.0, 1.0, 1.0]
        budgets = _select_per_class_budgets(
            costs, BoundaryRegionWaterfillStrategy.PER_CLASS_WEIGHTED_BY_AREA
        )
        assert all(b == 1 for b in budgets)

    def test_per_class_weighted_by_area_skewed_distribution(self) -> None:
        # Skewed: high-cost class gets higher budget
        costs = [10.0, 1.0, 1.0, 1.0, 1.0]
        budgets = _select_per_class_budgets(
            costs, BoundaryRegionWaterfillStrategy.PER_CLASS_WEIGHTED_BY_AREA
        )
        # Class 0 should have highest budget
        assert budgets[0] >= budgets[1]
        assert budgets[0] >= budgets[2]

    def test_per_class_weighted_by_area_zero_total_defaults_uniform(self) -> None:
        costs = [0.0, 0.0, 0.0, 0.0, 0.0]
        budgets = _select_per_class_budgets(
            costs, BoundaryRegionWaterfillStrategy.PER_CLASS_WEIGHTED_BY_AREA
        )
        assert budgets == [1, 1, 1, 1, 1]

    def test_per_region_at_boundary_uniform_budgets(self) -> None:
        costs = [1.0, 2.0, 3.0, 4.0, 5.0]
        budgets = _select_per_class_budgets(
            costs, BoundaryRegionWaterfillStrategy.PER_REGION_AT_BOUNDARY
        )
        assert budgets == [1, 1, 1, 1, 1]

    def test_per_region_interior_uniform_budgets(self) -> None:
        costs = [1.0, 2.0, 3.0, 4.0, 5.0]
        budgets = _select_per_class_budgets(
            costs, BoundaryRegionWaterfillStrategy.PER_REGION_INTERIOR
        )
        assert budgets == [1, 1, 1, 1, 1]


# -----------------------------------------------------------------------------
# Section 6: compute_uniward_weighted_per_class_budget canonical primitive
# -----------------------------------------------------------------------------


class TestComputeUniwardWeightedPerClassBudget:
    def test_per_class_uniform_canonical(self) -> None:
        response = _make_canonical_inputs(seed=42)
        cfg = BoundaryRegionWaterfillConfig(
            strategy=BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM
        )
        result = compute_uniward_weighted_per_class_budget(response, cfg)
        assert result["wire_bytes_estimate"] == 11
        assert result["fec6_baseline_wire_bytes"] == 249
        assert result["delta_vs_fec6_bytes"] == -238
        assert result["strategy"] == "per_class_uniform"
        assert result["n_classes"] == 5
        assert result["per_class_budgets"] == [1, 1, 1, 1, 1]

    def test_per_class_weighted_by_area_canonical(self) -> None:
        response = _make_canonical_inputs(seed=42)
        cfg = BoundaryRegionWaterfillConfig(
            strategy=BoundaryRegionWaterfillStrategy.PER_CLASS_WEIGHTED_BY_AREA
        )
        result = compute_uniward_weighted_per_class_budget(response, cfg)
        assert result["wire_bytes_estimate"] == 16
        assert result["delta_vs_fec6_bytes"] == -233
        assert result["strategy"] == "per_class_weighted_by_area"

    def test_per_region_at_boundary_canonical(self) -> None:
        response = _make_canonical_inputs(seed=42)
        cfg = BoundaryRegionWaterfillConfig(
            strategy=BoundaryRegionWaterfillStrategy.PER_REGION_AT_BOUNDARY
        )
        result = compute_uniward_weighted_per_class_budget(response, cfg)
        # 6-byte header + 5-byte budget + 20 × 4 = 91 bytes
        assert result["wire_bytes_estimate"] == 91
        assert result["delta_vs_fec6_bytes"] == -158
        assert result["strategy"] == "per_region_at_boundary"

    def test_per_region_interior_canonical(self) -> None:
        response = _make_canonical_inputs(seed=42)
        cfg = BoundaryRegionWaterfillConfig(
            strategy=BoundaryRegionWaterfillStrategy.PER_REGION_INTERIOR
        )
        result = compute_uniward_weighted_per_class_budget(response, cfg)
        assert result["wire_bytes_estimate"] == 131
        assert result["delta_vs_fec6_bytes"] == -118
        assert result["strategy"] == "per_region_interior"

    def test_mismatched_class_count_rejected(self) -> None:
        cfg = BoundaryRegionWaterfillConfig()
        with pytest.raises(ValueError, match="length must match"):
            compute_uniward_weighted_per_class_budget(
                [0.1, 0.2, 0.3],  # only 3 classes; expected 5
                cfg,
            )

    def test_proportional_savings_canonical_per_class_uniform(self) -> None:
        response = _make_canonical_inputs(seed=42)
        cfg = BoundaryRegionWaterfillConfig(
            strategy=BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM
        )
        result = compute_uniward_weighted_per_class_budget(response, cfg)
        # Verify analytical: proportional_savings = 25 * -238 / 37_545_489
        expected = 25.0 * -238 / 37_545_489
        assert result["proportional_savings"] == pytest.approx(
            expected, rel=1e-12
        )

    def test_proportional_savings_canonical_all_strategies_negative(self) -> None:
        # All 4 strategies should produce negative proportional_savings
        # (i.e. score savings) because all are wire-bytes < FEC6 baseline
        response = _make_canonical_inputs(seed=42)
        for strategy in BoundaryRegionWaterfillStrategy:
            cfg = BoundaryRegionWaterfillConfig(strategy=strategy)
            result = compute_uniward_weighted_per_class_budget(response, cfg)
            assert result["proportional_savings"] < 0, (
                f"strategy={strategy.value}: proportional_savings should be negative"
            )

    def test_canonical_boundary_k_alternative(self) -> None:
        # PER_REGION_AT_BOUNDARY wire bytes scale with boundary_k
        response = _make_canonical_inputs(seed=42)
        cfg = BoundaryRegionWaterfillConfig(
            strategy=BoundaryRegionWaterfillStrategy.PER_REGION_AT_BOUNDARY,
            boundary_k=40,
        )
        result = compute_uniward_weighted_per_class_budget(response, cfg)
        # 6 + 5 + 40*4 = 171 bytes
        assert result["wire_bytes_estimate"] == 171


# -----------------------------------------------------------------------------
# Section 7: apply_boundary_region_waterfill_to_pr110_archive Tier A contract
# -----------------------------------------------------------------------------


class TestApplyBoundaryRegionWaterfillTierAMarkers:
    def test_predicted_delta_adjustment_zero(self) -> None:
        # Catalog #341: Tier A canonical-routing markers
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        assert result["predicted_delta_adjustment"] == 0.0

    def test_promotable_false(self) -> None:
        # Catalog #341 + #1 device-fork trap protection
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        assert result["promotable"] is False

    def test_axis_tag_predicted(self) -> None:
        # Catalog #287 evidence-tag discipline
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        assert result["axis_tag"] == "[predicted]"

    def test_verdict_deferred_pending_paired_cuda(self) -> None:
        # Catalog #325 verdict
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        assert result["verdict"] == "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"


# -----------------------------------------------------------------------------
# Section 8: AxisDecomposition per Catalog #356 emission
# -----------------------------------------------------------------------------


class TestAxisDecompositionEmission:
    def test_axis_decomposition_payload_present_when_enabled(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(emit_axis_decomposition=True),
        )
        decomp = result["predicted_axis_decomposition"]
        assert decomp is not None
        assert "predicted_d_seg_delta" in decomp
        assert "predicted_d_pose_delta" in decomp
        assert "predicted_archive_bytes_delta" in decomp
        assert "axis_tag" in decomp
        assert "canonical_provenance" in decomp

    def test_axis_decomposition_payload_none_when_disabled(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(emit_axis_decomposition=False),
        )
        assert result["predicted_axis_decomposition"] is None

    def test_axis_decomposition_seg_delta_zero_at_l0(self) -> None:
        # L0 SCAFFOLD: no actual perturbation applied
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        assert result["predicted_axis_decomposition"]["predicted_d_seg_delta"] == 0.0

    def test_axis_decomposition_pose_delta_zero_at_l0(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        assert result["predicted_axis_decomposition"]["predicted_d_pose_delta"] == 0.0

    def test_axis_decomposition_archive_bytes_delta_per_strategy(self) -> None:
        # Strategy => predicted_archive_bytes_delta
        expected = {
            BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM: -238,
            BoundaryRegionWaterfillStrategy.PER_CLASS_WEIGHTED_BY_AREA: -233,
            BoundaryRegionWaterfillStrategy.PER_REGION_AT_BOUNDARY: -158,
            BoundaryRegionWaterfillStrategy.PER_REGION_INTERIOR: -118,
        }
        for strategy, exp_delta in expected.items():
            cfg = BoundaryRegionWaterfillConfig(strategy=strategy)
            result = apply_boundary_region_waterfill_to_pr110_archive(
                _make_canonical_inputs(),
                cfg,
            )
            assert (
                result["predicted_axis_decomposition"]["predicted_archive_bytes_delta"]
                == exp_delta
            ), f"strategy={strategy.value}"

    def test_axis_decomposition_axis_tag_predicted(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        assert result["predicted_axis_decomposition"]["axis_tag"] == "[predicted]"


# -----------------------------------------------------------------------------
# Section 9: Per-substrate empirical verification per Slot QQ META-LESSON
# -----------------------------------------------------------------------------


class TestPerSubstrateEmpiricalVerificationSlotQQ:
    def test_verification_status_pending_at_l0(self) -> None:
        # Slot QQ META-LESSON: L0 SCAFFOLD always returns PENDING
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        verif = result["per_substrate_empirical_verification"]
        assert verif["verification_status"] == "PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"

    def test_per_substrate_empirically_verified_false_at_l0(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        verif = result["per_substrate_empirical_verification"]
        assert verif["per_substrate_empirically_verified"] is False

    def test_slot_qq_meta_lesson_commit_sha_anchored(self) -> None:
        # Per Slot QQ commit 40476d935 IMPLEMENTATION-LEVEL FALSIFICATION
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        verif = result["per_substrate_empirical_verification"]
        assert verif["slot_qq_meta_lesson_citation_commit_sha"] == "40476d935"

    def test_substrate_id_propagated(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
            substrate_id="pr106_format0d",
        )
        verif = result["per_substrate_empirical_verification"]
        assert verif["substrate_id"] == "pr106_format0d"

    def test_substrate_id_default_pr110_fec6_baseline(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        verif = result["per_substrate_empirical_verification"]
        assert verif["substrate_id"] == "pr110_fec6_baseline"

    def test_reactivation_criterion_mentions_paired_cuda(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        verif = result["per_substrate_empirical_verification"]
        assert "paired-CUDA" in verif["verification_reactivation_criterion"]
        assert "Catalog #246" in verif["verification_reactivation_criterion"]

    def test_segnet_response_signature_deterministic(self) -> None:
        # Same response => same signature (Catalog #305 diff-able-across-runs)
        response = [0.1, 0.2, 0.3, 0.4, 0.5]
        v1 = _compute_per_substrate_empirical_verification("sub_a", response)
        v2 = _compute_per_substrate_empirical_verification("sub_a", response)
        assert v1["segnet_response_signature_sha256"] == v2["segnet_response_signature_sha256"]

    def test_segnet_response_signature_changes_with_response(self) -> None:
        v1 = _compute_per_substrate_empirical_verification("sub_a", [0.1, 0.2, 0.3, 0.4, 0.5])
        v2 = _compute_per_substrate_empirical_verification("sub_a", [0.2, 0.3, 0.4, 0.5, 0.6])
        assert v1["segnet_response_signature_sha256"] != v2["segnet_response_signature_sha256"]


# -----------------------------------------------------------------------------
# Section 10: Canonical Fridrich-Yousfi 3-axis cascade anchor + sister refs
# -----------------------------------------------------------------------------


class TestCanonicalFridrichYousfi3AxisCascadeAnchor:
    def test_canonical_anchor_present(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        anchor = result["canonical_anchor"]
        assert anchor["fridrich_yousfi_3_axis_cascade_position"] == "axis_3_segnet_class_region_surface"

    def test_axis_1_sister_module_cited(self) -> None:
        # Slot FF OPT-7 LANDED commit 0adecdc5b
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        anchor = result["canonical_anchor"]
        assert "pr110_opt_7" in anchor["axis_1_sister_module_path"]
        assert anchor["axis_1_sister_commit_sha"] == "0adecdc5b"

    def test_axis_2_sister_module_cited(self) -> None:
        # Slot RR OPT-6 LANDED
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        anchor = result["canonical_anchor"]
        assert "pr110_opt_6" in anchor["axis_2_sister_module_path"]
        assert "pose_axis_null_projection" in anchor["axis_2_sister_module_path"]

    def test_axis_4_sister_module_cited(self) -> None:
        # Slot X OPT-4 LANDED commit 0eb7cb615 (sister pattern)
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        anchor = result["canonical_anchor"]
        assert "pr110_opt_4" in anchor["axis_4_sister_module_path"]
        assert anchor["axis_4_sister_commit_sha"] == "0eb7cb615"

    def test_canonical_constants_in_anchor(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        anchor = result["canonical_anchor"]
        assert anchor["canonical_per_class_uniform_wire_bytes"] == 11
        assert anchor["canonical_per_class_uniform_delta_bytes_vs_fec6"] == -238
        assert anchor["fec6_baseline_wire_bytes"] == 249
        assert anchor["segnet_n_classes"] == 5
        assert anchor["canonical_n_pairs"] == 600

    def test_catalog_308_alternative_reducer_enumeration_4_canonicals(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        anchor = result["canonical_anchor"]
        alternatives = anchor["catalog_308_alternative_reducer_enumeration"]
        assert len(alternatives) == 4
        assert any("per_class_uniform" in a for a in alternatives)
        assert any("per_class_weighted_by_area" in a for a in alternatives)
        assert any("per_region_at_boundary" in a for a in alternatives)
        assert any("per_region_interior" in a for a in alternatives)

    def test_canonical_citation_fridrich_holub(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        anchor = result["canonical_anchor"]
        assert "Holub-Fridrich-Denemark" in anchor["canonical_citation"]
        assert "Sallee 2003" in anchor["canonical_citation"]


# -----------------------------------------------------------------------------
# Section 11: Slot CC dissent + design memo + horizon class
# -----------------------------------------------------------------------------


class TestSlotCCDissentAndMisc:
    def test_slot_cc_dissent_anchor_commit_sha(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        assert result["slot_cc_dissent_anchor"]["commit_sha"] == "18c6cd571"
        assert result["slot_cc_dissent_anchor"]["council_tier"] == "T3"

    def test_slot_qq_meta_lesson_anchor_present(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        anchor = result["slot_qq_meta_lesson_anchor"]
        assert anchor["commit_sha"] == "40476d935"
        assert "per-substrate" in anchor["meta_lesson"].lower()
        assert anchor["extinction_pattern"] == "IMPLEMENTATION-LEVEL FALSIFICATION per Catalog #307"

    def test_design_memo_path_set(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        assert "pr110_opt_5" in result["design_memo_path"]
        assert "boundary_region_waterfill" in result["design_memo_path"]
        assert "fridrich_yousfi_3_axis_cascade" in result["design_memo_path"]

    def test_horizon_class_plateau_adjacent(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        assert result["horizon_class"] == "plateau_adjacent"

    def test_wire_analysis_present(self) -> None:
        result = apply_boundary_region_waterfill_to_pr110_archive(
            _make_canonical_inputs(),
            BoundaryRegionWaterfillConfig(),
        )
        wire = result["wire_analysis"]
        assert "wire_bytes_estimate" in wire
        assert "delta_vs_fec6_bytes" in wire
        assert "per_class_budgets" in wire
        assert "uniward_costs_per_class" in wire


# -----------------------------------------------------------------------------
# Section 12: Parametrized strategy dispatch (4-strategy paired comparison)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy,expected_wire_bytes,expected_delta_vs_fec6",
    [
        (BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM, 11, -238),
        (BoundaryRegionWaterfillStrategy.PER_CLASS_WEIGHTED_BY_AREA, 16, -233),
        (BoundaryRegionWaterfillStrategy.PER_REGION_AT_BOUNDARY, 91, -158),
        (BoundaryRegionWaterfillStrategy.PER_REGION_INTERIOR, 131, -118),
    ],
)
def test_strategy_dispatch_wire_bytes_canonical(
    strategy: BoundaryRegionWaterfillStrategy,
    expected_wire_bytes: int,
    expected_delta_vs_fec6: int,
) -> None:
    """Canonical 4-strategy paired-comparison per Catalog #308 alternative-reducer enumeration."""
    cfg = BoundaryRegionWaterfillConfig(strategy=strategy)
    result = apply_boundary_region_waterfill_to_pr110_archive(
        _make_canonical_inputs(seed=42),
        cfg,
    )
    assert result["wire_analysis"]["wire_bytes_estimate"] == expected_wire_bytes
    assert result["wire_analysis"]["delta_vs_fec6_bytes"] == expected_delta_vs_fec6


# -----------------------------------------------------------------------------
# Section 13: Signature determinism (Catalog #305 diff-able-across-runs)
# -----------------------------------------------------------------------------


class TestClassRegionSignatureDeterminism:
    def test_signature_deterministic_same_inputs(self) -> None:
        response = [0.1, 0.2, 0.3, 0.4, 0.5]
        s1 = _compute_class_region_signature(
            response, BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM, 20
        )
        s2 = _compute_class_region_signature(
            response, BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM, 20
        )
        assert s1 == s2

    def test_signature_changes_with_strategy(self) -> None:
        response = [0.1, 0.2, 0.3, 0.4, 0.5]
        s1 = _compute_class_region_signature(
            response, BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM, 20
        )
        s2 = _compute_class_region_signature(
            response, BoundaryRegionWaterfillStrategy.PER_REGION_AT_BOUNDARY, 20
        )
        assert s1 != s2

    def test_signature_changes_with_boundary_k(self) -> None:
        response = [0.1, 0.2, 0.3, 0.4, 0.5]
        s1 = _compute_class_region_signature(
            response, BoundaryRegionWaterfillStrategy.PER_REGION_AT_BOUNDARY, 20
        )
        s2 = _compute_class_region_signature(
            response, BoundaryRegionWaterfillStrategy.PER_REGION_AT_BOUNDARY, 40
        )
        assert s1 != s2

    def test_signature_sha256_hex_length(self) -> None:
        response = [0.1, 0.2, 0.3, 0.4, 0.5]
        sig = _compute_class_region_signature(
            response, BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM, 20
        )
        assert len(sig) == 64  # sha256 hex
        int(sig, 16)  # valid hex
