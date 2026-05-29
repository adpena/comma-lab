# SPDX-License-Identifier: MIT
"""Canonical tests for MiPOD canonical inverse-steganalysis L0 SCAFFOLD.

Per Slot AAA canonical landing 2026-05-29 + canonical Slot UU canonical
TOP-2 8/9 ranking + canonical Fridrich-Yousfi inverse-steganalysis
cascade Axis 6 extension + canonical Sedighi-Cogranne-Fridrich 2016
reference.

Coverage sections:

1. Canonical enum membership + Config invariants per Catalog #287
2. Canonical Sedighi-Cogranne reference implementation correctness
3. Tier A canonical-routing markers per Catalog #341
4. AxisDecomposition per Catalog #356
5. Provenance per Catalog #323
6. Sister citations (Sedighi-Cogranne + Slot UU + Slot FF + Slot YY)
7. Parametrized strategy dispatch
"""

from __future__ import annotations

import math
import random

import pytest

from tac.composition.mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016 import (
    CANONICAL_FEC6_BASELINE_WIRE_BYTES,
    CANONICAL_MIPOD_EPSILON,
    CANONICAL_N_MODES,
    CANONICAL_N_PAIRS,
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
    CANONICAL_SEDIGHI_COGRANNE_FRIDRICH_2016_CITATION,
    CANONICAL_SEDIGHI_COGRANNE_FRIDRICH_2016_CITATION_URL,
    CANONICAL_SEDIGHI_COGRANNE_REFERENCE_EPSILON,
    CANONICAL_SLOT_UU_RANK,
    CANONICAL_SLOT_UU_SCORE_STRING,
    CANONICAL_SPARSE_K_DEFAULT,
    CANONICAL_VARIANCE_ESTIMATION_WINDOW,
    CANONICAL_WIDENED_K_DEFAULT,
    CANONICAL_WIENER_KERNEL_SIZE,
    MiPODConfig,
    MiPODGaussianCoverStrategy,
    apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive,
    compute_mipod_canonical_fisher_information_cost_matrix_for_pr110_catalog,
)

# Internal helpers exercised via private-API access (test-only).
from tac.composition.mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016 import (
    _aggregate_cost_matrix_to_per_pair_priority,
    _compute_canonical_sedighi_cogranne_fridrich_mipod_fisher_information_cost_matrix,
    _compute_mipod_canonical_signature,
    _local_mean_2d,
    _select_sparse_k_pairs_by_priority,
    _wiener_filter_canonical,
)


# ---------------------------------------------------------------------------
# Section 1: Canonical enum + Config invariants
# ---------------------------------------------------------------------------


class TestMiPODGaussianCoverStrategyEnum:
    """Canonical enum membership + Catalog #308 alternative-reducer enumeration."""

    def test_enum_has_four_canonical_values(self):
        members = list(MiPODGaussianCoverStrategy)
        assert len(members) == 4

    def test_canonical_baseline_member_present(self):
        assert MiPODGaussianCoverStrategy.CANONICAL_WIENER_FILTER_VARIANCE.value == (
            "canonical_wiener_filter_variance"
        )

    def test_canonical_local_3x3_member_present(self):
        assert MiPODGaussianCoverStrategy.CANONICAL_LOCAL_VARIANCE_3X3.value == (
            "canonical_local_variance_3x3"
        )

    def test_per_region_member_present(self):
        assert MiPODGaussianCoverStrategy.PER_REGION_ADAPTIVE_VARIANCE.value == (
            "per_region_adaptive_variance"
        )

    def test_per_pixel_member_present(self):
        assert MiPODGaussianCoverStrategy.PER_PIXEL_GLOBAL_VARIANCE_NORMALIZATION.value == (
            "per_pixel_global_variance_normalization"
        )

    def test_enum_is_string_enum(self):
        # Catalog #305 cite-able + diff-able-across-runs facet
        assert isinstance(
            MiPODGaussianCoverStrategy.CANONICAL_WIENER_FILTER_VARIANCE.value, str
        )


class TestMiPODConfigInvariants:
    """Catalog #287 placeholder rejection + __post_init__ invariants."""

    def test_default_config_valid(self):
        cfg = MiPODConfig()
        assert cfg.strategy == MiPODGaussianCoverStrategy.CANONICAL_WIENER_FILTER_VARIANCE
        assert cfg.wiener_kernel_size == CANONICAL_WIENER_KERNEL_SIZE
        assert cfg.variance_estimation_window == CANONICAL_VARIANCE_ESTIMATION_WINDOW
        assert cfg.epsilon == CANONICAL_MIPOD_EPSILON
        assert cfg.sparse_k == CANONICAL_SPARSE_K_DEFAULT
        assert cfg.n_pairs == CANONICAL_N_PAIRS
        assert cfg.n_modes == CANONICAL_N_MODES
        assert cfg.header_overhead_bytes == 4
        assert cfg.emit_axis_decomposition is True

    def test_config_is_frozen(self):
        cfg = MiPODConfig()
        with pytest.raises(Exception):
            cfg.strategy = MiPODGaussianCoverStrategy.PER_REGION_ADAPTIVE_VARIANCE

    def test_strategy_must_be_enum(self):
        with pytest.raises(ValueError, match="strategy must be"):
            MiPODConfig(strategy="not_an_enum")

    def test_wiener_kernel_size_must_be_int(self):
        with pytest.raises(ValueError, match="wiener_kernel_size must be int"):
            MiPODConfig(wiener_kernel_size=5.0)

    def test_wiener_kernel_size_must_be_valid_size(self):
        with pytest.raises(ValueError, match="wiener_kernel_size must be in"):
            MiPODConfig(wiener_kernel_size=4)

    def test_wiener_kernel_size_3_accepted(self):
        cfg = MiPODConfig(wiener_kernel_size=3)
        assert cfg.wiener_kernel_size == 3

    def test_wiener_kernel_size_7_accepted(self):
        cfg = MiPODConfig(wiener_kernel_size=7)
        assert cfg.wiener_kernel_size == 7

    def test_variance_window_must_be_int(self):
        with pytest.raises(ValueError, match="variance_estimation_window must be int"):
            MiPODConfig(variance_estimation_window=3.0)

    def test_variance_window_must_be_valid(self):
        with pytest.raises(ValueError, match="variance_estimation_window must be in"):
            MiPODConfig(variance_estimation_window=4)

    def test_variance_window_5_accepted(self):
        cfg = MiPODConfig(variance_estimation_window=5)
        assert cfg.variance_estimation_window == 5

    def test_epsilon_nan_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be finite"):
            MiPODConfig(epsilon=float("nan"))

    def test_epsilon_inf_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be finite"):
            MiPODConfig(epsilon=float("inf"))

    def test_epsilon_zero_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            MiPODConfig(epsilon=0.0)

    def test_epsilon_negative_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            MiPODConfig(epsilon=-1e-6)

    def test_sparse_k_must_be_int(self):
        with pytest.raises(ValueError, match="sparse_k must be int"):
            MiPODConfig(sparse_k=100.0)

    def test_sparse_k_must_be_positive(self):
        with pytest.raises(ValueError, match="sparse_k must be > 0"):
            MiPODConfig(sparse_k=0)

    def test_sparse_k_cant_exceed_n_pairs(self):
        with pytest.raises(ValueError, match="must be <= n_pairs"):
            MiPODConfig(sparse_k=700, n_pairs=600)

    def test_n_pairs_must_be_int(self):
        with pytest.raises(ValueError, match="n_pairs must be int"):
            MiPODConfig(n_pairs=600.0)

    def test_n_modes_must_be_positive(self):
        with pytest.raises(ValueError, match="n_modes must be > 0"):
            MiPODConfig(n_modes=0)

    def test_header_overhead_must_be_non_negative(self):
        with pytest.raises(ValueError, match="header_overhead_bytes must be >= 0"):
            MiPODConfig(header_overhead_bytes=-1)


# ---------------------------------------------------------------------------
# Section 2: Canonical Sedighi-Cogranne reference implementation correctness
# ---------------------------------------------------------------------------


def _make_test_image(height: int = 30, width: int = 40, seed: int = 42):
    """Build canonical reproducible test image (luma 0-255)."""
    rng = random.Random(seed)
    return [[rng.uniform(0.0, 255.0) for _ in range(width)] for _ in range(height)]


class TestLocalMean2D:
    """Canonical local-mean filter unit tests."""

    def test_local_mean_uniform_image_returns_same(self):
        image = [[5.0 for _ in range(10)] for _ in range(10)]
        result = _local_mean_2d(image, 3)
        for row in result:
            for v in row:
                assert math.isclose(v, 5.0, abs_tol=1e-9)

    def test_local_mean_rejects_empty(self):
        with pytest.raises(ValueError, match="image must be non-empty"):
            _local_mean_2d([], 3)

    def test_local_mean_rejects_even_window(self):
        with pytest.raises(ValueError, match="window_size must be odd positive"):
            _local_mean_2d([[1.0]], 4)


class TestWienerFilterCanonical:
    """Canonical Wiener-filter unit tests per Sedighi-Cogranne 2016 Algorithm 1."""

    def test_wiener_filter_uniform_returns_same(self):
        image = [[10.0 for _ in range(10)] for _ in range(10)]
        result = _wiener_filter_canonical(image, 5)
        for row in result:
            for v in row:
                assert math.isclose(v, 10.0, abs_tol=1e-9)

    def test_wiener_filter_output_shape_matches_input(self):
        image = _make_test_image(height=15, width=20)
        result = _wiener_filter_canonical(image, 5)
        assert len(result) == 15
        assert all(len(row) == 20 for row in result)


class TestCanonicalCostMatrix:
    """Canonical Sedighi-Cogranne Fisher-information cost-matrix correctness."""

    def test_cost_matrix_uniform_image_high_cost(self):
        """Uniform image has zero variance → infinite cost → clipped to 1/epsilon."""
        image = [[100.0 for _ in range(20)] for _ in range(30)]
        cfg = MiPODConfig()
        cost = _compute_canonical_sedighi_cogranne_fridrich_mipod_fisher_information_cost_matrix(
            image, cfg
        )
        max_cost = 1.0 / cfg.epsilon
        # For uniform image, sigma^2 = 0 so cost = 1/epsilon (clipped)
        for row in cost:
            for v in row:
                assert math.isclose(v, max_cost, rel_tol=1e-6)

    def test_cost_matrix_output_shape(self):
        image = _make_test_image(height=15, width=20)
        cfg = MiPODConfig()
        cost = _compute_canonical_sedighi_cogranne_fridrich_mipod_fisher_information_cost_matrix(
            image, cfg
        )
        assert len(cost) == 15
        assert all(len(row) == 20 for row in cost)

    def test_cost_matrix_clipped_to_max_cost(self):
        image = _make_test_image()
        cfg = MiPODConfig(epsilon=1e-3)
        cost = _compute_canonical_sedighi_cogranne_fridrich_mipod_fisher_information_cost_matrix(
            image, cfg
        )
        max_cost = 1.0 / cfg.epsilon
        for row in cost:
            for v in row:
                assert cfg.epsilon <= v <= max_cost + 1e-9

    def test_cost_matrix_textured_image_lower_cost_than_uniform(self):
        """Random textured image has HIGHER variance → LOWER cost than uniform."""
        textured = _make_test_image(height=30, width=40, seed=7)
        cfg = MiPODConfig()
        cost = _compute_canonical_sedighi_cogranne_fridrich_mipod_fisher_information_cost_matrix(
            textured, cfg
        )
        mean_cost = sum(v for row in cost for v in row) / (30 * 40)
        max_cost = 1.0 / cfg.epsilon
        # textured-image cost should be LOWER than uniform (clipped max)
        assert mean_cost < max_cost


class TestAggregateToPerPairPriority:
    """Per-pair priority aggregation unit tests."""

    def test_aggregate_returns_n_pairs_priorities(self):
        cost = [[1.0 for _ in range(10)] for _ in range(60)]
        result = _aggregate_cost_matrix_to_per_pair_priority(cost, n_pairs=6)
        assert len(result) == 6

    def test_aggregate_rejects_empty_cost(self):
        with pytest.raises(ValueError, match="cost_matrix must be non-empty"):
            _aggregate_cost_matrix_to_per_pair_priority([], 5)

    def test_aggregate_rejects_zero_n_pairs(self):
        with pytest.raises(ValueError, match="n_pairs must be > 0"):
            _aggregate_cost_matrix_to_per_pair_priority([[1.0]], 0)


class TestSparseKSelection:
    """Sparse-K selection unit tests."""

    def test_sparse_k_returns_sorted_indices(self):
        priorities = [3.0, 1.0, 5.0, 2.0, 4.0]
        selected = _select_sparse_k_pairs_by_priority(priorities, sparse_k=3)
        # Top-3 by priority: idx 2 (5.0), idx 4 (4.0), idx 0 (3.0); sorted ascending
        assert selected == [0, 2, 4]

    def test_sparse_k_returns_all_when_k_exceeds_n(self):
        priorities = [3.0, 1.0, 5.0]
        selected = _select_sparse_k_pairs_by_priority(priorities, sparse_k=10)
        assert selected == [0, 1, 2]


# ---------------------------------------------------------------------------
# Section 3: Tier A canonical-routing markers per Catalog #341
# ---------------------------------------------------------------------------


class TestTierACanonicalRoutingMarkers:
    """Canonical Tier A canonical-routing-markers per Catalog #341."""

    def test_apply_returns_predicted_delta_adjustment_zero(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        assert result["predicted_delta_adjustment"] == 0.0

    def test_apply_returns_promotable_false(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        assert result["promotable"] is False

    def test_apply_returns_axis_tag_predicted(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        assert result["axis_tag"] == "[predicted]"

    def test_apply_returns_deferred_verdict(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        assert result["verdict"] == "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"


# ---------------------------------------------------------------------------
# Section 4: AxisDecomposition per Catalog #356
# ---------------------------------------------------------------------------


class TestAxisDecomposition:
    """Canonical AxisDecomposition per Catalog #356."""

    def test_axis_decomposition_present_by_default(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        assert result["predicted_axis_decomposition"] is not None

    def test_axis_decomposition_absent_when_disabled(self):
        image = _make_test_image()
        cfg = MiPODConfig(emit_axis_decomposition=False)
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        assert result["predicted_axis_decomposition"] is None

    def test_axis_decomposition_seg_pose_deltas_zero_at_l0_scaffold(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        decomp = result["predicted_axis_decomposition"]
        assert decomp["predicted_d_seg_delta"] == 0.0
        assert decomp["predicted_d_pose_delta"] == 0.0

    def test_axis_decomposition_archive_bytes_matches_wire_analysis(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        decomp = result["predicted_axis_decomposition"]
        wire = result["wire_analysis"]
        assert decomp["predicted_archive_bytes_delta"] == wire["delta_vs_fec6_bytes"]

    def test_axis_decomposition_has_canonical_provenance(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        decomp = result["predicted_axis_decomposition"]
        assert "canonical_provenance" in decomp
        assert isinstance(decomp["canonical_provenance"], dict)


# ---------------------------------------------------------------------------
# Section 5: Provenance per Catalog #323
# ---------------------------------------------------------------------------


class TestProvenance:
    """Canonical Provenance per Catalog #323."""

    def test_provenance_axis_predicted(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        prov = result["predicted_axis_decomposition"]["canonical_provenance"]
        # canonical Provenance dict shape per Catalog #323 has measurement_axis
        assert prov.get("measurement_axis") == "[predicted]"

    def test_signature_changes_with_strategy(self):
        sig_default = _compute_mipod_canonical_signature(
            n_pairs=600,
            sparse_k=100,
            strategy=MiPODGaussianCoverStrategy.CANONICAL_WIENER_FILTER_VARIANCE,
            wiener_kernel_size=5,
            variance_estimation_window=3,
            epsilon=1e-6,
        )
        sig_local = _compute_mipod_canonical_signature(
            n_pairs=600,
            sparse_k=100,
            strategy=MiPODGaussianCoverStrategy.CANONICAL_LOCAL_VARIANCE_3X3,
            wiener_kernel_size=5,
            variance_estimation_window=3,
            epsilon=1e-6,
        )
        assert sig_default != sig_local

    def test_signature_deterministic(self):
        sig1 = _compute_mipod_canonical_signature(
            n_pairs=600,
            sparse_k=100,
            strategy=MiPODGaussianCoverStrategy.CANONICAL_WIENER_FILTER_VARIANCE,
            wiener_kernel_size=5,
            variance_estimation_window=3,
            epsilon=1e-6,
        )
        sig2 = _compute_mipod_canonical_signature(
            n_pairs=600,
            sparse_k=100,
            strategy=MiPODGaussianCoverStrategy.CANONICAL_WIENER_FILTER_VARIANCE,
            wiener_kernel_size=5,
            variance_estimation_window=3,
            epsilon=1e-6,
        )
        assert sig1 == sig2

    def test_signature_is_sha256_hex(self):
        sig = _compute_mipod_canonical_signature(
            n_pairs=600,
            sparse_k=100,
            strategy=MiPODGaussianCoverStrategy.CANONICAL_WIENER_FILTER_VARIANCE,
            wiener_kernel_size=5,
            variance_estimation_window=3,
            epsilon=1e-6,
        )
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)


# ---------------------------------------------------------------------------
# Section 6: Sister citations
# ---------------------------------------------------------------------------


class TestSisterCitations:
    """Canonical sister-cascade citations + Slot UU TOP-2 ranking."""

    def test_canonical_sedighi_cogranne_citation_anchor_present(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        anchor = result["canonical_sedighi_cogranne_fridrich_2016_anchor"]
        assert anchor["citation_url"] == CANONICAL_SEDIGHI_COGRANNE_FRIDRICH_2016_CITATION_URL
        assert "Sedighi" in anchor["canonical_citation"]
        assert "Cogranne" in anchor["canonical_citation"]
        assert "Fridrich" in anchor["canonical_citation"]
        assert "2016" in anchor["canonical_citation"]

    def test_slot_uu_roadmap_anchor_present(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        anchor = result["slot_uu_roadmap_anchor"]
        assert anchor["commit_sha"] == "2b573f105"
        assert anchor["rank"] == CANONICAL_SLOT_UU_RANK
        assert anchor["score"] == CANONICAL_SLOT_UU_SCORE_STRING
        assert anchor["operator_binding_directive"] == "#10"

    def test_slot_ff_sister_cascade_anchor_present(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        anchor = result["slot_ff_sister_cascade_anchor"]
        assert anchor["commit_sha"] == "0adecdc5b"
        assert "pr110_opt_7" in anchor["sister_pattern_path"]
        assert "Axis 6" in anchor["axis_position_in_cascade"]

    def test_slot_yy_sister_cascade_anchor_present(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        anchor = result["slot_yy_sister_cascade_anchor"]
        assert "hill_canonical" in anchor["sister_pattern_path"]
        assert "Axis 5" in anchor["axis_position_in_cascade"]

    def test_per_substrate_empirical_verification_stub_present(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        stub = result["per_substrate_empirical_verification_stub"]
        assert stub["status"] == "pending_per_substrate_empirical_verification"
        assert "Slot QQ" in stub["canonical_lesson_reference"]

    def test_design_memo_path_canonical(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        memo = result["design_memo_path"]
        assert "mipod_canonical_inverse_steganalysis" in memo
        assert "sedighi_cogranne_fridrich_2016" in memo
        assert "design_20260529.md" in memo

    def test_horizon_class_plateau_adjacent(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        assert result["horizon_class"] == "plateau_adjacent"


# ---------------------------------------------------------------------------
# Section 7: Parametrized strategy dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [
        MiPODGaussianCoverStrategy.CANONICAL_WIENER_FILTER_VARIANCE,
        MiPODGaussianCoverStrategy.CANONICAL_LOCAL_VARIANCE_3X3,
        MiPODGaussianCoverStrategy.PER_REGION_ADAPTIVE_VARIANCE,
        MiPODGaussianCoverStrategy.PER_PIXEL_GLOBAL_VARIANCE_NORMALIZATION,
    ],
)
class TestParametrizedStrategyDispatch:
    """All 4 canonical strategies produce valid wire analysis."""

    def test_compute_with_strategy_returns_wire_analysis(self, strategy):
        image = _make_test_image()
        cfg = MiPODConfig(strategy=strategy)
        wire = compute_mipod_canonical_fisher_information_cost_matrix_for_pr110_catalog(
            image, cfg
        )
        assert wire["strategy"] == strategy.value
        assert wire["wire_bytes_estimate"] > 0
        assert wire["n_selected_pairs"] == cfg.sparse_k
        assert wire["fec6_baseline_wire_bytes"] == CANONICAL_FEC6_BASELINE_WIRE_BYTES

    def test_apply_with_strategy_returns_tier_a_markers(self, strategy):
        image = _make_test_image()
        cfg = MiPODConfig(strategy=strategy)
        result = apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
            image, cfg
        )
        assert result["predicted_delta_adjustment"] == 0.0
        assert result["promotable"] is False
        assert result["axis_tag"] == "[predicted]"


# ---------------------------------------------------------------------------
# Section 8: Canonical anchor constants
# ---------------------------------------------------------------------------


class TestCanonicalConstants:
    """Canonical anchor constants pinned per Slot UU + Slot FF/YY sister pattern."""

    def test_canonical_wiener_kernel_size_5(self):
        assert CANONICAL_WIENER_KERNEL_SIZE == 5

    def test_canonical_variance_window_3(self):
        assert CANONICAL_VARIANCE_ESTIMATION_WINDOW == 3

    def test_canonical_mipod_epsilon_1e_6(self):
        assert CANONICAL_MIPOD_EPSILON == 1e-6

    def test_canonical_sedighi_cogranne_reference_epsilon_2_neg_6(self):
        assert CANONICAL_SEDIGHI_COGRANNE_REFERENCE_EPSILON == 0.015625

    def test_canonical_sparse_k_default_100(self):
        assert CANONICAL_SPARSE_K_DEFAULT == 100

    def test_canonical_widened_k_default_200(self):
        assert CANONICAL_WIDENED_K_DEFAULT == 200

    def test_canonical_fec6_baseline_249(self):
        assert CANONICAL_FEC6_BASELINE_WIRE_BYTES == 249

    def test_canonical_n_pairs_600(self):
        assert CANONICAL_N_PAIRS == 600

    def test_canonical_n_modes_21(self):
        assert CANONICAL_N_MODES == 21

    def test_canonical_rate_multiplier_25(self):
        assert CANONICAL_RATE_MULTIPLIER == 25.0

    def test_canonical_rate_denom_bytes_37545489(self):
        assert CANONICAL_RATE_DENOM_BYTES == 37_545_489

    def test_canonical_slot_uu_rank_2(self):
        assert CANONICAL_SLOT_UU_RANK == 2

    def test_canonical_slot_uu_score_string(self):
        assert CANONICAL_SLOT_UU_SCORE_STRING == "8/9"

    def test_canonical_sedighi_cogranne_citation_url(self):
        assert CANONICAL_SEDIGHI_COGRANNE_FRIDRICH_2016_CITATION_URL == (
            "https://hal.science/hal-01906608/document"
        )


# ---------------------------------------------------------------------------
# Section 9: Wire analysis arithmetic
# ---------------------------------------------------------------------------


class TestWireAnalysisArithmetic:
    """Wire-bytes arithmetic correctness per canonical sparse-K selector format."""

    def test_default_wire_bytes_estimate(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        wire = compute_mipod_canonical_fisher_information_cost_matrix_for_pr110_catalog(
            image, cfg
        )
        # Default cfg: header=4, sparse_k=100, n_pairs=600
        # index_byte_width = ceil(log2(600)/8) = ceil(9.23/8) = 2
        # wire = 4 + 100*2 + 100 = 304
        assert wire["wire_bytes_estimate"] == 304

    def test_default_delta_vs_fec6(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        wire = compute_mipod_canonical_fisher_information_cost_matrix_for_pr110_catalog(
            image, cfg
        )
        # 304 - 249 = +55 bytes (canonical baseline rate-axis NEUTRAL-TO-POSITIVE)
        assert wire["delta_vs_fec6_bytes"] == 55

    def test_cost_matrix_summary_keys(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        wire = compute_mipod_canonical_fisher_information_cost_matrix_for_pr110_catalog(
            image, cfg
        )
        summary = wire["cost_matrix_summary"]
        assert "mean" in summary
        assert "std" in summary
        assert "min" in summary
        assert "max" in summary
        assert "n_pixels" in summary

    def test_concentration_factor_in_range(self):
        image = _make_test_image()
        cfg = MiPODConfig()
        wire = compute_mipod_canonical_fisher_information_cost_matrix_for_pr110_catalog(
            image, cfg
        )
        # Concentration = top-K priorities / all priorities; bounded [0, 1]
        assert 0.0 <= wire["mipod_concentration_factor"] <= 1.0
