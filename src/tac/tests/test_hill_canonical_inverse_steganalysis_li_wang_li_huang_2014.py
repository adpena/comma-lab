# SPDX-License-Identifier: MIT
"""Canonical tests for HILL canonical inverse-steganalysis L0 SCAFFOLD.

Per Slot YY canonical landing 2026-05-29 + canonical Slot UU canonical
TOP-1 9/9 ranking + canonical Fridrich-Yousfi inverse-steganalysis
cascade Axis 5 extension + canonical Li-Wang-Li-Huang 2014 reference.

Coverage sections:

1. Canonical enum membership + Config invariants per Catalog #287
2. Canonical Li-Wang reference implementation correctness
3. Tier A canonical-routing markers per Catalog #341
4. AxisDecomposition per Catalog #356
5. Provenance per Catalog #323
6. Sister citations (Li-Wang + Slot UU + OVERNIGHT-EEE + Slot FF)
7. Parametrized strategy dispatch
"""

from __future__ import annotations

import random

import pytest

from tac.composition.hill_canonical_inverse_steganalysis_li_wang_li_huang_2014 import (
    CANONICAL_FEC6_BASELINE_WIRE_BYTES,
    CANONICAL_HILL_EPSILON,
    CANONICAL_KB_KERNEL_3X3,
    CANONICAL_KB_KERNEL_SIZE,
    CANONICAL_L1_KERNEL_SIZE,
    CANONICAL_L2_KERNEL_SIZE,
    CANONICAL_LI_WANG_2014_CITATION_URL,
    CANONICAL_N_MODES,
    CANONICAL_N_PAIRS,
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
    CANONICAL_SPARSE_K_DEFAULT,
    CANONICAL_WIDENED_K_DEFAULT,
    HILLConfig,
    HILLCostMatrixStrategy,
    apply_hill_canonical_cost_matrix_to_pr110_archive,
    compute_hill_canonical_cost_matrix_for_pr110_catalog,
)


# Internal helpers we exercise via private API access (test-only).
from tac.composition.hill_canonical_inverse_steganalysis_li_wang_li_huang_2014 import (
    _aggregate_cost_matrix_to_per_pair_priority,
    _build_averaging_kernel,
    _compute_canonical_li_wang_hill_cost_matrix,
    _compute_hill_canonical_signature,
    _convolve_2d_canonical,
    _select_sparse_k_pairs_by_priority,
)


# ---------------------------------------------------------------------------
# Section 1: Canonical enum + Config invariants
# ---------------------------------------------------------------------------


class TestHILLCostMatrixStrategyEnum:
    """Canonical enum membership + Catalog #308 alternative-reducer enumeration."""

    def test_enum_has_four_canonical_values(self):
        members = list(HILLCostMatrixStrategy)
        assert len(members) == 4

    def test_canonical_baseline_member_present(self):
        assert HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS.value == (
            "canonical_3x3_kb_high_pass"
        )

    def test_extended_5x5_member_present(self):
        assert HILLCostMatrixStrategy.EXTENDED_5X5_HIGH_PASS.value == (
            "extended_5x5_high_pass"
        )

    def test_per_region_member_present(self):
        assert HILLCostMatrixStrategy.PER_REGION_VARIABLE_KERNEL.value == (
            "per_region_variable_kernel"
        )

    def test_per_pixel_member_present(self):
        assert HILLCostMatrixStrategy.PER_PIXEL_ADAPTIVE_KERNEL.value == (
            "per_pixel_adaptive_kernel"
        )

    def test_enum_is_string_enum(self):
        # Catalog #305 cite-able + diff-able-across-runs facet
        assert isinstance(
            HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS.value, str
        )


class TestHILLConfigInvariants:
    """Catalog #287 placeholder rejection + __post_init__ invariants."""

    def test_default_config_valid(self):
        cfg = HILLConfig()
        assert cfg.strategy == HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS
        assert cfg.kb_kernel_size == CANONICAL_KB_KERNEL_SIZE
        assert cfg.l1_kernel_size == CANONICAL_L1_KERNEL_SIZE
        assert cfg.l2_kernel_size == CANONICAL_L2_KERNEL_SIZE
        assert cfg.epsilon == CANONICAL_HILL_EPSILON
        assert cfg.sparse_k == CANONICAL_SPARSE_K_DEFAULT
        assert cfg.n_pairs == CANONICAL_N_PAIRS
        assert cfg.n_modes == CANONICAL_N_MODES

    def test_strategy_must_be_enum(self):
        with pytest.raises(ValueError, match="strategy must be HILLCostMatrixStrategy"):
            HILLConfig(strategy="canonical_3x3_kb_high_pass")  # type: ignore[arg-type]

    def test_kb_kernel_size_must_be_int(self):
        with pytest.raises(ValueError, match="kb_kernel_size must be int"):
            HILLConfig(kb_kernel_size=3.0)  # type: ignore[arg-type]

    def test_kb_kernel_size_rejects_bool(self):
        with pytest.raises(ValueError, match="kb_kernel_size must be int"):
            HILLConfig(kb_kernel_size=True)  # type: ignore[arg-type]

    def test_kb_kernel_size_must_be_3_or_5(self):
        with pytest.raises(ValueError, match=r"kb_kernel_size must be in \(3, 5\)"):
            HILLConfig(kb_kernel_size=7)

    def test_l1_kernel_size_must_be_3_5_or_7(self):
        with pytest.raises(ValueError, match=r"l1_kernel_size must be in \(3, 5, 7\)"):
            HILLConfig(l1_kernel_size=9)

    def test_l2_kernel_size_must_be_9_15_or_21(self):
        with pytest.raises(ValueError, match=r"l2_kernel_size must be in \(9, 15, 21\)"):
            HILLConfig(l2_kernel_size=7)

    def test_epsilon_must_be_positive(self):
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            HILLConfig(epsilon=0.0)

    def test_epsilon_rejects_nan(self):
        with pytest.raises(ValueError, match="epsilon must be finite"):
            HILLConfig(epsilon=float("nan"))

    def test_epsilon_rejects_inf(self):
        with pytest.raises(ValueError, match="epsilon must be finite"):
            HILLConfig(epsilon=float("inf"))

    def test_epsilon_rejects_bool(self):
        with pytest.raises(ValueError, match="epsilon must be number"):
            HILLConfig(epsilon=True)  # type: ignore[arg-type]

    def test_sparse_k_must_be_positive(self):
        with pytest.raises(ValueError, match="sparse_k must be > 0"):
            HILLConfig(sparse_k=0)

    def test_sparse_k_must_be_le_n_pairs(self):
        with pytest.raises(ValueError, match="sparse_k .* must be <= n_pairs"):
            HILLConfig(sparse_k=601)

    def test_n_pairs_must_be_positive(self):
        with pytest.raises(ValueError, match="n_pairs must be > 0"):
            HILLConfig(n_pairs=0)

    def test_n_modes_must_be_positive(self):
        with pytest.raises(ValueError, match="n_modes must be > 0"):
            HILLConfig(n_modes=0)

    def test_header_overhead_bytes_must_be_non_negative(self):
        with pytest.raises(ValueError, match="header_overhead_bytes must be >= 0"):
            HILLConfig(header_overhead_bytes=-1)

    def test_config_frozen(self):
        cfg = HILLConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.epsilon = 1e-3  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Section 2: Canonical Li-Wang reference implementation correctness
# ---------------------------------------------------------------------------


class TestCanonicalKBKernel:
    """Ker-Bohme 2008 canonical 3x3 kernel."""

    def test_kb_kernel_shape(self):
        assert len(CANONICAL_KB_KERNEL_3X3) == 3
        for row in CANONICAL_KB_KERNEL_3X3:
            assert len(row) == 3

    def test_kb_kernel_center_negative_one(self):
        assert CANONICAL_KB_KERNEL_3X3[1][1] == -1.00

    def test_kb_kernel_corners_negative_quarter(self):
        assert CANONICAL_KB_KERNEL_3X3[0][0] == -0.25
        assert CANONICAL_KB_KERNEL_3X3[0][2] == -0.25
        assert CANONICAL_KB_KERNEL_3X3[2][0] == -0.25
        assert CANONICAL_KB_KERNEL_3X3[2][2] == -0.25

    def test_kb_kernel_edges_positive_half(self):
        assert CANONICAL_KB_KERNEL_3X3[0][1] == 0.50
        assert CANONICAL_KB_KERNEL_3X3[1][0] == 0.50
        assert CANONICAL_KB_KERNEL_3X3[1][2] == 0.50
        assert CANONICAL_KB_KERNEL_3X3[2][1] == 0.50


class TestBuildAveragingKernel:
    def test_3x3_averaging(self):
        kernel = _build_averaging_kernel(3)
        assert len(kernel) == 3
        assert all(len(row) == 3 for row in kernel)
        weight = 1.0 / 9.0
        for row in kernel:
            for cell in row:
                assert cell == pytest.approx(weight)

    def test_7x7_averaging_canonical(self):
        kernel = _build_averaging_kernel(7)
        weight = 1.0 / 49.0
        for row in kernel:
            for cell in row:
                assert cell == pytest.approx(weight)

    def test_15x15_averaging_canonical(self):
        kernel = _build_averaging_kernel(15)
        weight = 1.0 / 225.0
        for row in kernel:
            for cell in row:
                assert cell == pytest.approx(weight)

    def test_even_size_rejected(self):
        with pytest.raises(ValueError, match="odd positive"):
            _build_averaging_kernel(4)

    def test_zero_size_rejected(self):
        with pytest.raises(ValueError, match="odd positive"):
            _build_averaging_kernel(0)


class TestConvolve2dCanonical:
    def test_identity_kernel_returns_image(self):
        img = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        identity = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
        result = _convolve_2d_canonical(img, identity)
        for i in range(3):
            for j in range(3):
                assert result[i][j] == pytest.approx(img[i][j])

    def test_kb_kernel_on_constant_image_yields_zero(self):
        # Constant image has zero high-frequency content
        img = [[5.0 for _ in range(7)] for _ in range(7)]
        result = _convolve_2d_canonical(img, CANONICAL_KB_KERNEL_3X3)
        # Interior pixels should be exactly zero
        for i in range(1, 6):
            for j in range(1, 6):
                assert result[i][j] == pytest.approx(0.0, abs=1e-9)

    def test_kb_kernel_detects_isolated_impulse(self):
        # KB kernel is a 2D high-pass; pure axis-aligned steps don't excite
        # it (its rows sum to zero against a horizontal-invariant step).
        # An isolated impulse on a flat background DOES excite it (canonical
        # residual extraction); verify center response is the dominant one.
        img = [[0.0 for _ in range(7)] for _ in range(7)]
        img[3][3] = 10.0
        result = _convolve_2d_canonical(img, CANONICAL_KB_KERNEL_3X3)
        # Center pixel should have largest absolute response (impulse hits
        # KB kernel center weight = -1.00; result[3][3] = -10.0).
        center_response = abs(result[3][3])
        far_response = abs(result[0][0])
        assert center_response > far_response
        assert center_response == pytest.approx(10.0)

    def test_empty_image_rejected(self):
        with pytest.raises(ValueError, match="non-empty 2D"):
            _convolve_2d_canonical([], ((1.0,),))

    def test_non_uniform_width_rejected(self):
        with pytest.raises(ValueError, match="uniform width"):
            _convolve_2d_canonical([[1.0, 2.0], [3.0]], ((1.0,),))

    def test_non_square_kernel_rejected(self):
        with pytest.raises(ValueError, match="must be square"):
            _convolve_2d_canonical([[1.0]], [(1.0, 2.0), (3.0,)])

    def test_even_kernel_rejected(self):
        with pytest.raises(ValueError, match="must be odd"):
            _convolve_2d_canonical([[1.0]], ((1.0, 2.0), (3.0, 4.0)))


class TestComputeCanonicalLiWangHillCostMatrix:
    def test_cost_matrix_shape_matches_image(self):
        img = [[float(i + j) for j in range(20)] for i in range(20)]
        cfg = HILLConfig(n_pairs=10, sparse_k=5)
        cost = _compute_canonical_li_wang_hill_cost_matrix(img, cfg)
        assert len(cost) == 20
        assert all(len(row) == 20 for row in cost)

    def test_cost_matrix_all_positive(self):
        # Reciprocal with epsilon means all costs are positive
        random.seed(42)
        img = [[random.random() for _ in range(20)] for _ in range(20)]
        cfg = HILLConfig(n_pairs=10, sparse_k=5)
        cost = _compute_canonical_li_wang_hill_cost_matrix(img, cfg)
        for row in cost:
            for v in row:
                assert v > 0.0

    def test_cost_matrix_constant_image_canonical(self):
        # Constant image: residual ~0, intermediate ~0, reciprocal ~1/eps,
        # smoothed reciprocal ~1/eps
        img = [[5.0 for _ in range(20)] for _ in range(20)]
        cfg = HILLConfig(n_pairs=10, sparse_k=5, epsilon=1e-3)
        cost = _compute_canonical_li_wang_hill_cost_matrix(img, cfg)
        # Interior cost should be ~1/eps = 1000
        expected = 1.0 / 1e-3
        for i in range(10, 12):
            for j in range(10, 12):
                # Allow boundary effects to relax tolerance
                assert cost[i][j] == pytest.approx(expected, rel=0.5)

    def test_cost_matrix_with_extended_kb_kernel(self):
        img = [[float(i * j) for j in range(20)] for i in range(20)]
        cfg = HILLConfig(
            strategy=HILLCostMatrixStrategy.EXTENDED_5X5_HIGH_PASS,
            kb_kernel_size=5,
            n_pairs=10,
            sparse_k=5,
        )
        cost = _compute_canonical_li_wang_hill_cost_matrix(img, cfg)
        assert len(cost) == 20
        # All costs positive
        for row in cost:
            for v in row:
                assert v > 0.0


class TestAggregateCostMatrixToPerPairPriority:
    def test_aggregation_yields_n_pairs(self):
        cost_matrix = [[float(i) for _ in range(10)] for i in range(20)]
        priorities = _aggregate_cost_matrix_to_per_pair_priority(cost_matrix, 10)
        assert len(priorities) == 10

    def test_aggregation_monotonic_for_monotonic_input(self):
        # Row-monotonic cost matrix should give monotonic per-pair priorities
        cost_matrix = [[float(i) for _ in range(10)] for i in range(20)]
        priorities = _aggregate_cost_matrix_to_per_pair_priority(cost_matrix, 10)
        for i in range(len(priorities) - 1):
            assert priorities[i] < priorities[i + 1]

    def test_aggregation_empty_rejected(self):
        with pytest.raises(ValueError, match="non-empty 2D"):
            _aggregate_cost_matrix_to_per_pair_priority([], 5)

    def test_aggregation_zero_pairs_rejected(self):
        with pytest.raises(ValueError, match="n_pairs must be > 0"):
            _aggregate_cost_matrix_to_per_pair_priority([[1.0]], 0)


class TestSelectSparseKPairsByPriority:
    def test_select_top_k(self):
        priorities = [0.1, 0.9, 0.5, 0.3, 0.7]
        selected = _select_sparse_k_pairs_by_priority(priorities, 2)
        # Top 2 are indices 1 (0.9) and 4 (0.7); sorted ascending = [1, 4]
        assert selected == [1, 4]

    def test_select_all_when_k_ge_n(self):
        priorities = [0.1, 0.2, 0.3]
        selected = _select_sparse_k_pairs_by_priority(priorities, 5)
        assert selected == [0, 1, 2]

    def test_selected_indices_sorted_ascending(self):
        # Catalog #305 diff-able-across-runs facet
        priorities = [0.5, 0.1, 0.9, 0.3, 0.7]
        selected = _select_sparse_k_pairs_by_priority(priorities, 3)
        assert selected == sorted(selected)


# ---------------------------------------------------------------------------
# Section 3: Tier A canonical-routing markers per Catalog #341
# ---------------------------------------------------------------------------


class TestTierARoutingMarkers:
    """Catalog #341 + #357 dual-tier canonical-routing markers."""

    def setup_method(self):
        random.seed(42)
        self.img = [[random.random() for _ in range(32)] for _ in range(32)]
        self.cfg = HILLConfig(n_pairs=16, sparse_k=8)

    def test_predicted_delta_adjustment_is_zero(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        assert result["predicted_delta_adjustment"] == 0.0

    def test_promotable_is_false(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        assert result["promotable"] is False

    def test_axis_tag_is_predicted(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        assert result["axis_tag"] == "[predicted]"

    def test_verdict_is_deferred_pending_paired_cuda(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        assert result["verdict"] == "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"


# ---------------------------------------------------------------------------
# Section 4: AxisDecomposition per Catalog #356
# ---------------------------------------------------------------------------


class TestAxisDecomposition:
    """Catalog #356 per-axis decomposition contract."""

    def setup_method(self):
        random.seed(42)
        self.img = [[random.random() for _ in range(32)] for _ in range(32)]
        self.cfg = HILLConfig(n_pairs=16, sparse_k=8)

    def test_axis_decomposition_present_by_default(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        assert result["predicted_axis_decomposition"] is not None

    def test_axis_decomposition_absent_when_disabled(self):
        cfg = HILLConfig(n_pairs=16, sparse_k=8, emit_axis_decomposition=False)
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, cfg)
        assert result["predicted_axis_decomposition"] is None

    def test_axis_decomposition_has_canonical_keys(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        decomp = result["predicted_axis_decomposition"]
        assert "predicted_d_seg_delta" in decomp
        assert "predicted_d_pose_delta" in decomp
        assert "predicted_archive_bytes_delta" in decomp
        assert "axis_tag" in decomp
        assert "canonical_provenance" in decomp

    def test_axis_decomposition_seg_pose_zero_at_l0_scaffold(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        decomp = result["predicted_axis_decomposition"]
        assert decomp["predicted_d_seg_delta"] == 0.0
        assert decomp["predicted_d_pose_delta"] == 0.0

    def test_axis_decomposition_archive_bytes_int(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        decomp = result["predicted_axis_decomposition"]
        assert isinstance(decomp["predicted_archive_bytes_delta"], int)

    def test_axis_decomposition_axis_tag_predicted(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        decomp = result["predicted_axis_decomposition"]
        assert decomp["axis_tag"] == "[predicted]"


# ---------------------------------------------------------------------------
# Section 5: Provenance per Catalog #323
# ---------------------------------------------------------------------------


class TestProvenance:
    """Catalog #323 canonical Provenance umbrella."""

    def setup_method(self):
        random.seed(42)
        self.img = [[random.random() for _ in range(32)] for _ in range(32)]
        self.cfg = HILLConfig(n_pairs=16, sparse_k=8)

    def test_provenance_dict_form(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        decomp = result["predicted_axis_decomposition"]
        prov = decomp["canonical_provenance"]
        assert isinstance(prov, dict)
        # artifact_kind field present per canonical Provenance umbrella
        assert "artifact_kind" in prov
        assert prov["artifact_kind"] == "predicted_from_model"

    def test_signature_deterministic(self):
        sig1 = _compute_hill_canonical_signature(
            16, 8, HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS,
            3, 7, 15, 1e-6,
        )
        sig2 = _compute_hill_canonical_signature(
            16, 8, HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS,
            3, 7, 15, 1e-6,
        )
        assert sig1 == sig2

    def test_signature_changes_per_strategy(self):
        sig1 = _compute_hill_canonical_signature(
            16, 8, HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS,
            3, 7, 15, 1e-6,
        )
        sig2 = _compute_hill_canonical_signature(
            16, 8, HILLCostMatrixStrategy.EXTENDED_5X5_HIGH_PASS,
            3, 7, 15, 1e-6,
        )
        assert sig1 != sig2

    def test_signature_changes_per_epsilon(self):
        sig1 = _compute_hill_canonical_signature(
            16, 8, HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS,
            3, 7, 15, 1e-6,
        )
        sig2 = _compute_hill_canonical_signature(
            16, 8, HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS,
            3, 7, 15, 1e-3,
        )
        assert sig1 != sig2

    def test_signature_64_char_hex(self):
        sig = _compute_hill_canonical_signature(
            16, 8, HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS,
            3, 7, 15, 1e-6,
        )
        assert len(sig) == 64
        # Hex characters only
        assert all(c in "0123456789abcdef" for c in sig)


# ---------------------------------------------------------------------------
# Section 6: Sister citations
# ---------------------------------------------------------------------------


class TestSisterCitations:
    """Catalog #305 cite-able facet + canonical citation chain."""

    def setup_method(self):
        random.seed(42)
        self.img = [[random.random() for _ in range(32)] for _ in range(32)]
        self.cfg = HILLConfig(n_pairs=16, sparse_k=8)

    def test_li_wang_2014_anchor_citation_url(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        anchor = result["canonical_li_wang_2014_anchor"]
        assert anchor["citation_url"] == CANONICAL_LI_WANG_2014_CITATION_URL
        assert "Li-Wang-Li-Huang 2014" in anchor["canonical_citation"]

    def test_slot_uu_roadmap_anchor(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        anchor = result["slot_uu_roadmap_anchor"]
        assert anchor["commit_sha"] == "2b573f105"
        assert anchor["rank"] == 1
        assert anchor["score"] == "9/9"
        assert anchor["operator_binding_directive"] == "#10"

    def test_overnight_eee_implementation_boundary_anchor(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        anchor = result["overnight_eee_implementation_boundary_anchor"]
        assert "overnight_eee_hill_filter" in anchor["landing_memo_path"]
        assert "NULL_SIGNAL_DEFER" in anchor["verdict"]
        assert (
            anchor["catalog_307_classification"]
            == "IMPLEMENTATION_LEVEL_BOUNDARY_PARADIGM_INTACT"
        )

    def test_slot_ff_sister_cascade_anchor(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        anchor = result["slot_ff_sister_cascade_anchor"]
        assert anchor["commit_sha"] == "0adecdc5b"
        assert "pr110_opt_7_uniward" in anchor["sister_pattern_path"]
        assert "Axis 5 (HILL)" in anchor["axis_position_in_cascade"]

    def test_design_memo_path_canonical(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        path = result["design_memo_path"]
        assert "hill_canonical_inverse_steganalysis_li_wang_li_huang_2014" in path
        assert "design_20260529.md" in path

    def test_horizon_class_plateau_adjacent(self):
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        assert result["horizon_class"] == "plateau_adjacent"

    def test_per_substrate_empirical_verification_stub_present(self):
        # Per Slot QQ canonical META-LESSON
        result = apply_hill_canonical_cost_matrix_to_pr110_archive(self.img, self.cfg)
        stub = result["per_substrate_empirical_verification_stub"]
        assert stub["status"] == "pending_per_substrate_empirical_verification"
        assert "catalog_246" in stub["next_action"]
        assert "Slot QQ canonical META-LESSON" in stub["canonical_lesson_reference"]


# ---------------------------------------------------------------------------
# Section 7: Parametrized strategy dispatch
# ---------------------------------------------------------------------------


class TestStrategyDispatch:
    """Catalog #308 alternative-reducer enumeration + canonical strategy dispatch."""

    def setup_method(self):
        random.seed(42)
        self.img = [[random.random() for _ in range(32)] for _ in range(32)]

    @pytest.mark.parametrize(
        "strategy",
        [
            HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS,
            HILLCostMatrixStrategy.EXTENDED_5X5_HIGH_PASS,
            HILLCostMatrixStrategy.PER_REGION_VARIABLE_KERNEL,
            HILLCostMatrixStrategy.PER_PIXEL_ADAPTIVE_KERNEL,
        ],
    )
    def test_all_strategies_return_canonical_shape(self, strategy):
        # Use kb_kernel_size=5 for EXTENDED_5X5; 3 for others
        kb_size = 5 if strategy == HILLCostMatrixStrategy.EXTENDED_5X5_HIGH_PASS else 3
        cfg = HILLConfig(
            strategy=strategy,
            kb_kernel_size=kb_size,
            n_pairs=16,
            sparse_k=8,
        )
        result = compute_hill_canonical_cost_matrix_for_pr110_catalog(self.img, cfg)
        # Canonical shape contract
        assert "cost_matrix_summary" in result
        assert "per_pair_priorities" in result
        assert "selected_pair_indices" in result
        assert "wire_bytes_estimate" in result
        assert "fec6_baseline_wire_bytes" in result
        assert "delta_vs_fec6_bytes" in result
        assert "strategy" in result
        assert "n_selected_pairs" in result
        assert "hill_concentration_factor" in result
        assert result["strategy"] == strategy.value

    @pytest.mark.parametrize(
        "strategy",
        [
            HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS,
            HILLCostMatrixStrategy.PER_REGION_VARIABLE_KERNEL,
            HILLCostMatrixStrategy.PER_PIXEL_ADAPTIVE_KERNEL,
        ],
    )
    def test_strategies_select_k_pairs(self, strategy):
        cfg = HILLConfig(strategy=strategy, n_pairs=16, sparse_k=8)
        result = compute_hill_canonical_cost_matrix_for_pr110_catalog(self.img, cfg)
        assert len(result["selected_pair_indices"]) == 8
        assert result["n_selected_pairs"] == 8

    def test_wire_bytes_canonical_anchored(self):
        cfg = HILLConfig(n_pairs=16, sparse_k=8)
        result = compute_hill_canonical_cost_matrix_for_pr110_catalog(self.img, cfg)
        # Sister of Slot FF PR110-OPT-7 sparse-K wire estimate
        assert result["fec6_baseline_wire_bytes"] == 249

    def test_concentration_factor_in_unit_interval(self):
        cfg = HILLConfig(n_pairs=16, sparse_k=8)
        result = compute_hill_canonical_cost_matrix_for_pr110_catalog(self.img, cfg)
        assert 0.0 <= result["hill_concentration_factor"] <= 1.0

    def test_cost_matrix_summary_all_keys(self):
        cfg = HILLConfig(n_pairs=16, sparse_k=8)
        result = compute_hill_canonical_cost_matrix_for_pr110_catalog(self.img, cfg)
        summary = result["cost_matrix_summary"]
        assert "mean" in summary
        assert "std" in summary
        assert "min" in summary
        assert "max" in summary
        assert "n_pixels" in summary
        assert summary["mean"] > 0.0
        assert summary["std"] >= 0.0


class TestCanonicalAnchorConstants:
    """Pin canonical Slot FF + Slot UU + Li-Wang anchor constants."""

    def test_canonical_l1_kernel_size_canonical_seven_per_operator_prompt(self):
        # Operator binding prompt specifies 7x7 L1; sister Li-Wang allows {3, 5, 7}
        assert CANONICAL_L1_KERNEL_SIZE == 7

    def test_canonical_l2_kernel_size_canonical_fifteen_per_li_wang(self):
        assert CANONICAL_L2_KERNEL_SIZE == 15

    def test_canonical_kb_kernel_size_canonical_three_per_ker_bohme(self):
        assert CANONICAL_KB_KERNEL_SIZE == 3

    def test_canonical_epsilon_canonical_per_slot_ff_sister(self):
        assert CANONICAL_HILL_EPSILON == 1e-6

    def test_canonical_sparse_k_default_canonical_per_slot_ff_sister(self):
        assert CANONICAL_SPARSE_K_DEFAULT == 100

    def test_canonical_widened_k_canonical_per_catalog_308(self):
        assert CANONICAL_WIDENED_K_DEFAULT == 200

    def test_canonical_fec6_baseline_canonical_per_slot_ff_sister(self):
        assert CANONICAL_FEC6_BASELINE_WIRE_BYTES == 249

    def test_canonical_n_pairs_canonical_per_pr110(self):
        assert CANONICAL_N_PAIRS == 600

    def test_canonical_n_modes_canonical_per_pr110(self):
        assert CANONICAL_N_MODES == 21

    def test_canonical_rate_multiplier_canonical_contest_formula(self):
        assert CANONICAL_RATE_MULTIPLIER == 25.0

    def test_canonical_rate_denom_canonical_contest_video_bytes(self):
        assert CANONICAL_RATE_DENOM_BYTES == 37_545_489

    def test_canonical_citation_url_semanticscholar(self):
        assert "semanticscholar.org" in CANONICAL_LI_WANG_2014_CITATION_URL
        assert "Li-Wang" in CANONICAL_LI_WANG_2014_CITATION_URL


# -----------------------------------------------------------------------------
# Slot EEE remediation: per-pixel MLX real-video bind helper tests
# -----------------------------------------------------------------------------

import pytest

from pathlib import Path
from tac.composition.hill_canonical_inverse_steganalysis_li_wang_li_huang_2014 import (
    apply_hill_canonical_per_pixel_mlx_to_real_video_frames,
)

_REAL_VIDEO_PATH = Path("upstream/videos/0.mkv")
_REAL_VIDEO_AVAILABLE = _REAL_VIDEO_PATH.exists()


@pytest.mark.skipif(not _REAL_VIDEO_AVAILABLE, reason="upstream/videos/0.mkv not available")
class TestApplyHillPerPixelMlxRealVideoFrames:
    """Slot EEE remediation: REAL per-pixel MLX HILL on REAL upstream video frames."""

    def test_returns_tier_a_canonical_routing_markers(self):
        result = apply_hill_canonical_per_pixel_mlx_to_real_video_frames(
            num_frames=2, target_resolution=(96, 72)
        )
        assert result["predicted_delta_adjustment"] == 0.0
        assert result["promotable"] is False
        assert result["score_claim"] is False
        assert result["axis_tag"] == "[predicted]"

    def test_verdict_is_deferred_pending_paired_cuda(self):
        result = apply_hill_canonical_per_pixel_mlx_to_real_video_frames(
            num_frames=1, target_resolution=(64, 48)
        )
        assert "DEFERRED_PENDING_PAIRED_CUDA" in result["verdict"]
        assert "PER_PIXEL_MLX_REAL_VIDEO_SMOKE_GREEN" in result["verdict"]

    def test_smoke_result_includes_canonical_provenance(self):
        result = apply_hill_canonical_per_pixel_mlx_to_real_video_frames(
            num_frames=1, target_resolution=(64, 48)
        )
        sr = result["smoke_result"]
        assert sr["canonical_provenance"]["score_claim_valid"] is False
        assert "macOS-CPU advisory" in sr["canonical_provenance"]["axis_tag"]
        assert sr["canonical_routing_markers"]["promotable"] is False
        assert sr["canonical_routing_markers"]["axis_tag"] == "[predicted]"

    def test_per_pixel_cost_matrix_shape_matches_frame_resolution(self):
        result = apply_hill_canonical_per_pixel_mlx_to_real_video_frames(
            num_frames=1, target_resolution=(128, 96)
        )
        sr = result["smoke_result"]
        # (H, W) == (96, 128)
        assert tuple(sr["cost_matrix_shape"]) == (96, 128)

    def test_real_video_cost_discrimination_non_trivial(self):
        """REAL per-pixel HILL on REAL video shows non-trivial cost-discrimination.

        Sister of Slot EEE Axis C smoke-realism FAIL anti-test: synthetic random
        noise produces uniform cost; real video frames produce textured-vs-flat
        cost discrimination >20 dB dynamic range.
        """
        result = apply_hill_canonical_per_pixel_mlx_to_real_video_frames(
            num_frames=2, target_resolution=(128, 96)
        )
        dr = result["smoke_result"]["cost_matrix_dynamic_range_db"]
        # Real video frames typically show 20-35 dB dynamic range on HILL cost
        assert dr > 20.0

    def test_slot_eee_remediation_anchor_present(self):
        result = apply_hill_canonical_per_pixel_mlx_to_real_video_frames(
            num_frames=1, target_resolution=(64, 48)
        )
        anchor = result["per_pixel_real_video_remediation_anchor"]
        assert anchor["slot_eee_audit_axis_a_verdict"] == "PARTIAL_remediated"
        assert anchor["slot_eee_audit_axis_c_verdict"] == "FAIL_remediated"
        assert anchor["canonical_helper_module"] == "tac.inverse_steganalysis_real_video_mlx"
        assert anchor["canonical_helper_function"] == "compute_hill_per_pixel_cost_mlx"

    def test_uses_mlx_by_default(self):
        result = apply_hill_canonical_per_pixel_mlx_to_real_video_frames(
            num_frames=1, target_resolution=(64, 48)
        )
        assert result["smoke_result"]["used_mlx"] is True

    def test_numpy_only_path_works(self):
        result = apply_hill_canonical_per_pixel_mlx_to_real_video_frames(
            num_frames=1, target_resolution=(64, 48), use_mlx=False
        )
        assert result["smoke_result"]["used_mlx"] is False
        # Cost map still shows non-trivial discrimination
        assert result["smoke_result"]["cost_matrix_dynamic_range_db"] > 0


class TestApplyHillPerPixelMlxExportedFromPackage:
    """The new canonical bind helper is exported via __all__."""

    def test_exported(self):
        from tac.composition import hill_canonical_inverse_steganalysis_li_wang_li_huang_2014 as mod
        assert "apply_hill_canonical_per_pixel_mlx_to_real_video_frames" in mod.__all__
        assert hasattr(mod, "apply_hill_canonical_per_pixel_mlx_to_real_video_frames")
