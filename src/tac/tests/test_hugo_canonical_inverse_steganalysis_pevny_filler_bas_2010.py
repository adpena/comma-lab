# SPDX-License-Identifier: MIT
"""Canonical Slot CCC HUGO canonical inverse-steganalysis L0 SCAFFOLD tests.

Per canonical Slot UU canonical landing 2026-05-29 ~14:05 CST commit
``2b573f105`` + canonical Slot YY HILL canonical reference pattern landing
2026-05-29 + canonical Slot FF PR110-OPT-7 sister-cascade pattern (commit
``0adecdc5b``) + canonical Fridrich-Yousfi inverse-steganalysis cascade
Axis 7 extension + operator binding directive #10 explicit follow-through.

Test coverage:
  - Canonical Pevný-Filler-Bas 2010 anchor constants
  - HUGOSPAMFeatureStrategy enum membership
  - HUGOConfig invariants per Catalog #287
  - Canonical residual + truncation + Markov-chain co-occurrence primitives
  - Canonical per-pixel SPAM-feature delta cost
  - Canonical per-pair aggregation + sparse-K selection
  - Tier A canonical-routing markers per Catalog #341
  - AxisDecomposition per Catalog #356
  - Canonical Provenance per Catalog #323
  - Canonical sister citations (Slot UU + Slot YY + Slot FF + Pevný-Filler-Bas 2010)
  - Canonical parametrized strategy dispatch (4 strategies)
"""

from __future__ import annotations

import hashlib
import math

import pytest

from tac.composition.hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010 import (
    CANONICAL_4_DIRECTION_OFFSETS,
    CANONICAL_8_DIRECTION_OFFSETS,
    CANONICAL_FEC6_BASELINE_WIRE_BYTES,
    CANONICAL_HUGO_EPSILON,
    CANONICAL_N_MODES,
    CANONICAL_N_PAIRS,
    CANONICAL_PEVNY_BAS_FRIDRICH_2010_SPAM_CITATION_TEXT,
    CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_TEXT,
    CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_URL,
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
    CANONICAL_SPAM_COOCCURRENCE_ORDER,
    CANONICAL_SPAM_TRUNCATION_T,
    CANONICAL_SPARSE_K_DEFAULT,
    CANONICAL_WIDENED_K_DEFAULT,
    HUGOConfig,
    HUGOSPAMFeatureStrategy,
    apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive,
    compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog,
)
from tac.composition.hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010 import (
    _aggregate_cost_matrix_to_per_pair_priority,
    _canonical_truncate_residual,
    _compute_canonical_markov_chain_cooccurrence_matrix,
    _compute_canonical_residual_per_direction,
    _compute_hugo_canonical_signature,
    _compute_spam_feature_delta_per_pixel,
    _select_sparse_k_pairs_by_priority,
)


# ============================================================================
# Section 1: Canonical Pevný-Filler-Bas 2010 anchor constants
# ============================================================================


class TestCanonicalAnchorConstants:
    """Canonical anchor constants pinned per Pevný-Filler-Bas 2010 reference."""

    def test_pevny_filler_bas_2010_citation_url_is_springer(self):
        assert (
            CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_URL
            == "https://link.springer.com/chapter/10.1007/978-3-642-16435-4_13"
        )

    def test_pevny_filler_bas_2010_citation_text_canonical(self):
        assert "Pevn" in CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_TEXT
        assert "Filler" in CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_TEXT
        assert "Bas" in CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_TEXT
        assert "2010" in CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_TEXT
        assert "Information Hiding" in CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_TEXT

    def test_pevny_bas_fridrich_2010_spam_citation_text_canonical(self):
        assert "SPAM" not in CANONICAL_PEVNY_BAS_FRIDRICH_2010_SPAM_CITATION_TEXT
        # canonical citation uses "Subtractive Pixel Adjacency Matrix" expansion
        assert (
            "Subtractive Pixel Adjacency Matrix"
            in CANONICAL_PEVNY_BAS_FRIDRICH_2010_SPAM_CITATION_TEXT
        )
        assert "IEEE TIFS" in CANONICAL_PEVNY_BAS_FRIDRICH_2010_SPAM_CITATION_TEXT

    def test_4_direction_offsets_canonical_baseline(self):
        assert len(CANONICAL_4_DIRECTION_OFFSETS) == 4
        # canonical: horizontal + vertical + 2 diagonals
        expected = {(0, 1), (1, 0), (1, 1), (1, -1)}
        assert set(CANONICAL_4_DIRECTION_OFFSETS) == expected

    def test_8_direction_offsets_canonical_extension(self):
        assert len(CANONICAL_8_DIRECTION_OFFSETS) == 8
        # canonical: 8 cardinal + ordinal
        expected = {
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (-1, -1), (1, -1), (-1, 1),
        }
        assert set(CANONICAL_8_DIRECTION_OFFSETS) == expected

    def test_spam_truncation_T_canonical_4(self):
        assert CANONICAL_SPAM_TRUNCATION_T == 4

    def test_spam_cooccurrence_order_canonical_1(self):
        assert CANONICAL_SPAM_COOCCURRENCE_ORDER == 1

    def test_hugo_epsilon_canonical_1e_minus_6(self):
        assert CANONICAL_HUGO_EPSILON == 1e-6

    def test_sparse_k_default_canonical_100_sister_slot_ff(self):
        assert CANONICAL_SPARSE_K_DEFAULT == 100

    def test_widened_k_default_canonical_200(self):
        assert CANONICAL_WIDENED_K_DEFAULT == 200

    def test_fec6_baseline_wire_bytes_canonical_249(self):
        assert CANONICAL_FEC6_BASELINE_WIRE_BYTES == 249

    def test_n_pairs_canonical_600(self):
        assert CANONICAL_N_PAIRS == 600

    def test_n_modes_canonical_21(self):
        assert CANONICAL_N_MODES == 21

    def test_canonical_rate_multiplier_25(self):
        assert CANONICAL_RATE_MULTIPLIER == 25.0

    def test_canonical_rate_denom_bytes_37545489(self):
        assert CANONICAL_RATE_DENOM_BYTES == 37_545_489


# ============================================================================
# Section 2: HUGOSPAMFeatureStrategy enum
# ============================================================================


class TestHUGOSPAMFeatureStrategyEnum:
    """Canonical enum membership per Catalog #308 alternative-reducer enumeration."""

    def test_enum_has_4_canonical_strategies(self):
        members = list(HUGOSPAMFeatureStrategy)
        assert len(members) == 4

    def test_canonical_4_direction_spam_baseline(self):
        assert (
            HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM.value
            == "canonical_4_direction_spam"
        )

    def test_canonical_8_direction_spam_extension(self):
        assert (
            HUGOSPAMFeatureStrategy.CANONICAL_8_DIRECTION_SPAM.value
            == "canonical_8_direction_spam"
        )

    def test_per_region_variable_direction_extension(self):
        assert (
            HUGOSPAMFeatureStrategy.PER_REGION_VARIABLE_DIRECTION.value
            == "per_region_variable_direction"
        )

    def test_per_pixel_global_spam_normalization_extension(self):
        assert (
            HUGOSPAMFeatureStrategy.PER_PIXEL_GLOBAL_SPAM_NORMALIZATION.value
            == "per_pixel_global_spam_normalization"
        )

    def test_enum_inherits_from_str(self):
        assert issubclass(HUGOSPAMFeatureStrategy, str)


# ============================================================================
# Section 3: HUGOConfig invariants
# ============================================================================


class TestHUGOConfigInvariants:
    """Canonical HUGOConfig __post_init__ invariants per Catalog #287."""

    def test_default_config_constructs(self):
        config = HUGOConfig()
        assert config.strategy == HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM
        assert config.truncation_T == 4
        assert config.cooccurrence_order == 1
        assert config.epsilon == 1e-6
        assert config.sparse_k == 100
        assert config.n_pairs == 600
        assert config.n_modes == 21
        assert config.header_overhead_bytes == 4
        assert config.emit_axis_decomposition is True

    def test_frozen_dataclass(self):
        config = HUGOConfig()
        with pytest.raises((AttributeError, Exception)):
            config.sparse_k = 50  # type: ignore[misc]

    def test_invalid_strategy_type_rejected(self):
        with pytest.raises(ValueError, match="strategy must be HUGOSPAMFeatureStrategy"):
            HUGOConfig(strategy="invalid")  # type: ignore[arg-type]

    def test_truncation_T_must_be_int(self):
        with pytest.raises(ValueError, match="truncation_T must be int"):
            HUGOConfig(truncation_T=4.0)  # type: ignore[arg-type]

    def test_truncation_T_must_be_in_2_3_4(self):
        with pytest.raises(ValueError, match=r"truncation_T must be in \(2, 3, 4\)"):
            HUGOConfig(truncation_T=5)
        with pytest.raises(ValueError):
            HUGOConfig(truncation_T=1)
        HUGOConfig(truncation_T=2)
        HUGOConfig(truncation_T=3)
        HUGOConfig(truncation_T=4)

    def test_truncation_T_bool_rejected(self):
        with pytest.raises(ValueError, match="truncation_T must be int"):
            HUGOConfig(truncation_T=True)  # type: ignore[arg-type]

    def test_cooccurrence_order_must_be_1_or_2(self):
        HUGOConfig(cooccurrence_order=1)
        HUGOConfig(cooccurrence_order=2)
        with pytest.raises(ValueError, match=r"cooccurrence_order must be in \(1, 2\)"):
            HUGOConfig(cooccurrence_order=3)

    def test_epsilon_must_be_positive(self):
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            HUGOConfig(epsilon=0.0)
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            HUGOConfig(epsilon=-1e-6)

    def test_epsilon_nan_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be finite"):
            HUGOConfig(epsilon=float("nan"))

    def test_epsilon_inf_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be finite"):
            HUGOConfig(epsilon=float("inf"))

    def test_epsilon_bool_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be number"):
            HUGOConfig(epsilon=True)  # type: ignore[arg-type]

    def test_sparse_k_must_be_positive(self):
        with pytest.raises(ValueError, match="sparse_k must be > 0"):
            HUGOConfig(sparse_k=0)
        with pytest.raises(ValueError, match="sparse_k must be > 0"):
            HUGOConfig(sparse_k=-1)

    def test_sparse_k_must_be_int(self):
        with pytest.raises(ValueError, match="sparse_k must be int"):
            HUGOConfig(sparse_k=100.0)  # type: ignore[arg-type]

    def test_sparse_k_bool_rejected(self):
        with pytest.raises(ValueError, match="sparse_k must be int"):
            HUGOConfig(sparse_k=True)  # type: ignore[arg-type]

    def test_n_pairs_must_be_positive(self):
        with pytest.raises(ValueError, match="n_pairs must be > 0"):
            HUGOConfig(n_pairs=0)

    def test_sparse_k_le_n_pairs(self):
        # Construct config that would violate sparse_k <= n_pairs
        with pytest.raises(ValueError, match=r"sparse_k \(700\) must be <= n_pairs \(600\)"):
            HUGOConfig(sparse_k=700, n_pairs=600)

    def test_n_modes_must_be_positive(self):
        with pytest.raises(ValueError, match="n_modes must be > 0"):
            HUGOConfig(n_modes=0)

    def test_header_overhead_must_be_nonneg(self):
        with pytest.raises(ValueError, match="header_overhead_bytes must be >= 0"):
            HUGOConfig(header_overhead_bytes=-1)

    def test_header_overhead_must_be_int(self):
        with pytest.raises(ValueError, match="header_overhead_bytes must be int"):
            HUGOConfig(header_overhead_bytes=4.0)  # type: ignore[arg-type]


# ============================================================================
# Section 4: Canonical residual + truncation + Markov-chain co-occurrence primitives
# ============================================================================


class TestCanonicalResidualPrimitive:
    """Canonical per-direction residual extraction per Pevný-Bas-Fridrich 2010."""

    def test_horizontal_residual_canonical(self):
        # canonical horizontal direction (0, 1): r[i, j] = I[i, j] - I[i, j+1]
        image = [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ]
        residual = _compute_canonical_residual_per_direction(image, (0, 1))
        # r[0, 0] = 10 - 20 = -10; r[0, 1] = 20 - 30 = -10; r[0, 2] = 0 (boundary)
        assert residual[0][0] == -10.0
        assert residual[0][1] == -10.0
        assert residual[0][2] == 0.0
        assert residual[1][0] == -10.0
        assert residual[1][1] == -10.0
        assert residual[1][2] == 0.0

    def test_vertical_residual_canonical(self):
        # canonical vertical direction (1, 0): r[i, j] = I[i, j] - I[i+1, j]
        image = [
            [10.0, 20.0],
            [40.0, 50.0],
            [70.0, 80.0],
        ]
        residual = _compute_canonical_residual_per_direction(image, (1, 0))
        assert residual[0][0] == -30.0  # 10 - 40
        assert residual[0][1] == -30.0  # 20 - 50
        assert residual[1][0] == -30.0  # 40 - 70
        assert residual[1][1] == -30.0  # 50 - 80
        assert residual[2][0] == 0.0  # boundary
        assert residual[2][1] == 0.0  # boundary

    def test_diagonal_major_residual_canonical(self):
        # canonical diagonal-major (1, 1): r[i, j] = I[i, j] - I[i+1, j+1]
        image = [
            [10.0, 20.0],
            [30.0, 40.0],
        ]
        residual = _compute_canonical_residual_per_direction(image, (1, 1))
        assert residual[0][0] == -30.0  # 10 - 40
        assert residual[0][1] == 0.0  # boundary
        assert residual[1][0] == 0.0  # boundary
        assert residual[1][1] == 0.0  # boundary

    def test_empty_image_rejected(self):
        with pytest.raises(ValueError, match="image must be non-empty"):
            _compute_canonical_residual_per_direction([], (0, 1))
        with pytest.raises(ValueError, match="image must be non-empty"):
            _compute_canonical_residual_per_direction([[]], (0, 1))

    def test_non_uniform_width_rejected(self):
        with pytest.raises(ValueError, match="image rows must have uniform width"):
            _compute_canonical_residual_per_direction(
                [[1.0, 2.0], [3.0]], (0, 1)
            )


class TestCanonicalTruncatePrimitive:
    """Canonical SPAM truncation per Pevný-Bas-Fridrich 2010."""

    def test_truncation_T_4_canonical(self):
        residual = [
            [-10.0, -4.0, -1.0, 0.0],
            [1.0, 4.0, 7.0, 100.0],
        ]
        truncated = _canonical_truncate_residual(residual, 4)
        assert truncated[0] == [-4, -4, -1, 0]
        assert truncated[1] == [1, 4, 4, 4]

    def test_truncation_T_2(self):
        residual = [[-5.0, -2.0, 0.0, 2.0, 5.0]]
        truncated = _canonical_truncate_residual(residual, 2)
        assert truncated[0] == [-2, -2, 0, 2, 2]

    def test_T_must_be_positive(self):
        with pytest.raises(ValueError, match="T must be > 0"):
            _canonical_truncate_residual([[1.0]], 0)
        with pytest.raises(ValueError, match="T must be > 0"):
            _canonical_truncate_residual([[1.0]], -1)

    def test_truncation_rounds_floats(self):
        residual = [[1.7, -1.7, 3.4, -3.4]]
        truncated = _canonical_truncate_residual(residual, 4)
        assert truncated[0] == [2, -2, 3, -3]


class TestCanonicalMarkovChainCooccurrence:
    """Canonical Markov-chain co-occurrence matrix per Pevný-Bas-Fridrich 2010."""

    def test_simple_horizontal_cooccurrence_canonical(self):
        # 2 rows of truncated residuals, direction (0, 1), T=2
        # Matrix size = 2*2+1 = 5x5
        truncated = [
            [0, 1, 2, -1],
            [0, 0, 1, 2],
        ]
        M = _compute_canonical_markov_chain_cooccurrence_matrix(
            truncated, (0, 1), 2, 1
        )
        # Matrix indexed by (a+T, b+T)
        assert len(M) == 5
        assert len(M[0]) == 5
        # Pairs (truncated[i][j], truncated[i][j+1]):
        # row 0: (0, 1), (1, 2), (2, -1)
        # row 1: (0, 0), (0, 1), (1, 2)
        # Idx (0+2, 1+2) = (2, 3): count 2
        # Idx (1+2, 2+2) = (3, 4): count 2
        # Idx (2+2, -1+2) = (4, 1): count 1
        # Idx (0+2, 0+2) = (2, 2): count 1
        assert M[2][3] == 2
        assert M[3][4] == 2
        assert M[4][1] == 1
        assert M[2][2] == 1

    def test_T_must_be_positive(self):
        with pytest.raises(ValueError, match="T must be > 0"):
            _compute_canonical_markov_chain_cooccurrence_matrix(
                [[0]], (0, 1), 0, 1
            )

    def test_invalid_order_rejected(self):
        with pytest.raises(ValueError, match=r"cooccurrence_order must be in \(1, 2\)"):
            _compute_canonical_markov_chain_cooccurrence_matrix(
                [[0]], (0, 1), 2, 3
            )

    def test_matrix_size_is_2T_plus_1(self):
        for T in [2, 3, 4]:
            M = _compute_canonical_markov_chain_cooccurrence_matrix(
                [[0, 0], [0, 0]], (0, 1), T, 1
            )
            assert len(M) == 2 * T + 1
            assert len(M[0]) == 2 * T + 1

    def test_empty_residual_rejected(self):
        with pytest.raises(ValueError, match="truncated_residual must be non-empty"):
            _compute_canonical_markov_chain_cooccurrence_matrix(
                [], (0, 1), 4, 1
            )


# ============================================================================
# Section 5: Canonical per-pixel SPAM-feature delta cost
# ============================================================================


class TestCanonicalSPAMFeatureDeltaPerPixel:
    """Canonical per-pixel SPAM-feature delta cost per Pevný-Filler-Bas 2010."""

    def test_returns_cost_matrix_same_shape_as_input(self):
        image = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]
        cost = _compute_spam_feature_delta_per_pixel(
            image, CANONICAL_4_DIRECTION_OFFSETS, 4, 1
        )
        assert len(cost) == 3
        assert all(len(row) == 3 for row in cost)

    def test_all_costs_nonneg(self):
        image = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
        cost = _compute_spam_feature_delta_per_pixel(
            image, CANONICAL_4_DIRECTION_OFFSETS, 4, 1
        )
        flat = [v for row in cost for v in row]
        assert all(v >= 0.0 for v in flat)

    def test_interior_pixels_have_higher_cost_than_corners(self):
        # Canonical larger image so interior pixels have more neighbors
        image = [[float(i + j) for j in range(5)] for i in range(5)]
        cost = _compute_spam_feature_delta_per_pixel(
            image, CANONICAL_4_DIRECTION_OFFSETS, 4, 1
        )
        # Interior pixel (2, 2): all 8 incoming + outgoing edges valid
        # Corner pixel (0, 0): fewer valid edges
        assert cost[2][2] >= cost[0][0]

    def test_8_direction_higher_cost_than_4_direction_for_textured_image(self):
        # canonical larger checkered image to allow 8 direction structure
        image = [[(i + j) % 2 * 10.0 for j in range(5)] for i in range(5)]
        cost_4 = _compute_spam_feature_delta_per_pixel(
            image, CANONICAL_4_DIRECTION_OFFSETS, 4, 1
        )
        cost_8 = _compute_spam_feature_delta_per_pixel(
            image, CANONICAL_8_DIRECTION_OFFSETS, 4, 1
        )
        # 8 directions should accumulate more delta units per pixel
        # interior pixel:
        center_4 = cost_4[2][2]
        center_8 = cost_8[2][2]
        assert center_8 >= center_4

    def test_empty_image_rejected(self):
        with pytest.raises(ValueError, match="cover_image must be non-empty"):
            _compute_spam_feature_delta_per_pixel(
                [], CANONICAL_4_DIRECTION_OFFSETS, 4, 1
            )


# ============================================================================
# Section 6: Canonical per-pair aggregation + sparse-K selection
# ============================================================================


class TestPerPairAggregation:
    """Canonical aggregation to per-pair priority."""

    def test_returns_n_pairs_priorities(self):
        cost_matrix = [[1.0, 2.0, 3.0] for _ in range(10)]
        priorities = _aggregate_cost_matrix_to_per_pair_priority(
            cost_matrix, n_pairs=5
        )
        assert len(priorities) == 5

    def test_aggregation_returns_mean_per_band(self):
        # canonical 4-row image; n_pairs=2; band_size=2
        cost_matrix = [[10.0, 10.0], [10.0, 10.0], [20.0, 20.0], [20.0, 20.0]]
        priorities = _aggregate_cost_matrix_to_per_pair_priority(
            cost_matrix, n_pairs=2
        )
        assert priorities[0] == 10.0
        assert priorities[1] == 20.0

    def test_empty_matrix_rejected(self):
        with pytest.raises(ValueError, match="cost_matrix must be non-empty"):
            _aggregate_cost_matrix_to_per_pair_priority([], n_pairs=5)
        with pytest.raises(ValueError, match="cost_matrix must be non-empty"):
            _aggregate_cost_matrix_to_per_pair_priority([[]], n_pairs=5)

    def test_n_pairs_must_be_positive(self):
        cost_matrix = [[1.0]]
        with pytest.raises(ValueError, match="n_pairs must be > 0"):
            _aggregate_cost_matrix_to_per_pair_priority(cost_matrix, n_pairs=0)


class TestSparseKSelection:
    """Canonical sparse-K selection by priority (descending; sorted ascending for diff-able)."""

    def test_returns_sorted_ascending_indices(self):
        priorities = [3.0, 1.0, 2.0, 5.0, 4.0]
        selected = _select_sparse_k_pairs_by_priority(priorities, sparse_k=3)
        # Top-3 priorities are at indices 3 (5.0), 4 (4.0), 0 (3.0)
        # Returns sorted ascending: [0, 3, 4]
        assert selected == [0, 3, 4]

    def test_k_ge_n_returns_all_pair_indices(self):
        priorities = [3.0, 1.0, 2.0]
        selected = _select_sparse_k_pairs_by_priority(priorities, sparse_k=5)
        assert selected == [0, 1, 2]

    def test_k_equal_n_returns_all(self):
        priorities = [3.0, 1.0, 2.0]
        selected = _select_sparse_k_pairs_by_priority(priorities, sparse_k=3)
        assert selected == [0, 1, 2]


# ============================================================================
# Section 7: Canonical signature
# ============================================================================


class TestCanonicalSignature:
    """Canonical sha256 signature for canonical Provenance + diff-able-across-runs."""

    def test_signature_is_deterministic(self):
        sig1 = _compute_hugo_canonical_signature(
            n_pairs=600, sparse_k=100,
            strategy=HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM,
            truncation_T=4, cooccurrence_order=1, epsilon=1e-6,
        )
        sig2 = _compute_hugo_canonical_signature(
            n_pairs=600, sparse_k=100,
            strategy=HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM,
            truncation_T=4, cooccurrence_order=1, epsilon=1e-6,
        )
        assert sig1 == sig2

    def test_signature_differs_per_strategy(self):
        sig_4dir = _compute_hugo_canonical_signature(
            n_pairs=600, sparse_k=100,
            strategy=HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM,
            truncation_T=4, cooccurrence_order=1, epsilon=1e-6,
        )
        sig_8dir = _compute_hugo_canonical_signature(
            n_pairs=600, sparse_k=100,
            strategy=HUGOSPAMFeatureStrategy.CANONICAL_8_DIRECTION_SPAM,
            truncation_T=4, cooccurrence_order=1, epsilon=1e-6,
        )
        assert sig_4dir != sig_8dir

    def test_signature_differs_per_truncation_T(self):
        sig_T2 = _compute_hugo_canonical_signature(
            n_pairs=600, sparse_k=100,
            strategy=HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM,
            truncation_T=2, cooccurrence_order=1, epsilon=1e-6,
        )
        sig_T4 = _compute_hugo_canonical_signature(
            n_pairs=600, sparse_k=100,
            strategy=HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM,
            truncation_T=4, cooccurrence_order=1, epsilon=1e-6,
        )
        assert sig_T2 != sig_T4

    def test_signature_is_sha256_hex(self):
        sig = _compute_hugo_canonical_signature(
            n_pairs=600, sparse_k=100,
            strategy=HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM,
            truncation_T=4, cooccurrence_order=1, epsilon=1e-6,
        )
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)


# ============================================================================
# Section 8: Canonical analytical primitive output (compute_*_for_pr110_catalog)
# ============================================================================


def _make_test_image(height: int = 10, width: int = 10) -> list[list[float]]:
    """Canonical small test image for primitive testing."""
    return [
        [float((i * width + j) % 256) for j in range(width)]
        for i in range(height)
    ]


class TestCanonicalAnalyticalPrimitive:
    """Canonical compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog tests."""

    def test_returns_canonical_dict_keys(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        for key in (
            "cost_matrix_summary",
            "per_pair_priorities",
            "selected_pair_indices",
            "wire_bytes_estimate",
            "fec6_baseline_wire_bytes",
            "delta_vs_fec6_bytes",
            "strategy",
            "n_selected_pairs",
            "hugo_concentration_factor",
            "n_directions_used",
        ):
            assert key in result, f"missing key: {key}"

    def test_cost_matrix_summary_has_canonical_5_subkeys(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        summary = result["cost_matrix_summary"]
        for key in ("mean", "std", "min", "max", "n_pixels"):
            assert key in summary

    def test_4_direction_strategy_uses_4_directions(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(
            strategy=HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM,
            sparse_k=10, n_pairs=20,
        )
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        assert result["n_directions_used"] == 4

    def test_8_direction_strategy_uses_8_directions(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(
            strategy=HUGOSPAMFeatureStrategy.CANONICAL_8_DIRECTION_SPAM,
            sparse_k=10, n_pairs=20,
        )
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        assert result["n_directions_used"] == 8

    def test_per_region_degenerates_to_4_directions_at_L0(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(
            strategy=HUGOSPAMFeatureStrategy.PER_REGION_VARIABLE_DIRECTION,
            sparse_k=10, n_pairs=20,
        )
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        # Per design memo: PER_REGION degenerates to 4-direction at L0 SCAFFOLD level
        assert result["n_directions_used"] == 4

    def test_per_pixel_global_degenerates_to_4_directions_at_L0(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(
            strategy=HUGOSPAMFeatureStrategy.PER_PIXEL_GLOBAL_SPAM_NORMALIZATION,
            sparse_k=10, n_pairs=20,
        )
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        assert result["n_directions_used"] == 4

    def test_wire_bytes_estimate_canonical_formula(self):
        image = _make_test_image(20, 20)
        # canonical formula: header + K * index_byte_width + K * 1 (perturbation byte)
        # index_byte_width = ceil(log2(600)/8) = ceil(9.23/8) = 2
        # for K=100: 4 + 100*2 + 100 = 304
        config = HUGOConfig(sparse_k=100, n_pairs=600, header_overhead_bytes=4)
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        expected_wire = 4 + 100 * 2 + 100
        assert result["wire_bytes_estimate"] == expected_wire

    def test_delta_vs_fec6_bytes_canonical(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=100, n_pairs=600, header_overhead_bytes=4)
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        # wire 304 - FEC6 249 = 55
        assert result["delta_vs_fec6_bytes"] == 304 - 249

    def test_fec6_baseline_canonical_249(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        assert result["fec6_baseline_wire_bytes"] == 249

    def test_selected_pair_indices_sorted_ascending(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=5, n_pairs=20)
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        selected = result["selected_pair_indices"]
        assert selected == sorted(selected)

    def test_n_selected_pairs_matches_sparse_k(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=8, n_pairs=20)
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        assert result["n_selected_pairs"] == 8

    def test_concentration_in_unit_interval(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        c = result["hugo_concentration_factor"]
        assert 0.0 <= c <= 1.0

    def test_strategy_field_matches_input(self):
        image = _make_test_image(20, 20)
        for strategy in HUGOSPAMFeatureStrategy:
            config = HUGOConfig(strategy=strategy, sparse_k=5, n_pairs=20)
            result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
                image, config
            )
            assert result["strategy"] == strategy.value


# ============================================================================
# Section 9: Canonical apply entry point (Tier A markers + AxisDecomposition + Provenance)
# ============================================================================


class TestCanonicalApplyEntryPoint:
    """Canonical apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive tests."""

    def test_tier_a_canonical_routing_markers_present(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        # Per Catalog #341 Tier A:
        assert result["predicted_delta_adjustment"] == 0.0
        assert result["promotable"] is False
        assert result["axis_tag"] == "[predicted]"

    def test_axis_decomposition_emitted_when_configured(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20, emit_axis_decomposition=True)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        decomp = result["predicted_axis_decomposition"]
        assert decomp is not None
        assert "predicted_d_seg_delta" in decomp
        assert "predicted_d_pose_delta" in decomp
        assert "predicted_archive_bytes_delta" in decomp
        assert "axis_tag" in decomp
        assert "canonical_provenance" in decomp

    def test_axis_decomposition_omitted_when_disabled(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20, emit_axis_decomposition=False)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        assert result["predicted_axis_decomposition"] is None

    def test_axis_decomposition_carries_archive_bytes_delta(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        decomp = result["predicted_axis_decomposition"]
        wire = result["wire_analysis"]
        assert decomp["predicted_archive_bytes_delta"] == wire["delta_vs_fec6_bytes"]

    def test_l0_scaffold_pose_and_seg_deltas_are_zero(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        decomp = result["predicted_axis_decomposition"]
        # L0 SCAFFOLD: no actual perturbation applied
        assert decomp["predicted_d_seg_delta"] == 0.0
        assert decomp["predicted_d_pose_delta"] == 0.0

    def test_axis_decomposition_axis_tag_predicted(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        decomp = result["predicted_axis_decomposition"]
        assert decomp["axis_tag"] == "[predicted]"

    def test_canonical_provenance_dict_form(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        prov = result["predicted_axis_decomposition"]["canonical_provenance"]
        assert isinstance(prov, dict)
        # Canonical Provenance fields per Catalog #323
        assert "kind" in prov or "provenance_kind" in prov or len(prov) > 0

    def test_verdict_is_deferred_pending_paired_cuda(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        assert result["verdict"] == "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"

    def test_wire_analysis_passed_through(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        wire = result["wire_analysis"]
        assert "cost_matrix_summary" in wire
        assert "per_pair_priorities" in wire
        assert "selected_pair_indices" in wire

    def test_horizon_class_plateau_adjacent(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        assert result["horizon_class"] == "plateau_adjacent"


# ============================================================================
# Section 10: Canonical sister citations
# ============================================================================


class TestCanonicalSisterCitations:
    """Canonical sister anchor citations per CLAUDE.md "Results must become system intelligence"."""

    def test_pevny_filler_bas_2010_anchor_complete(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        anchor = result["canonical_pevny_filler_bas_2010_anchor"]
        assert anchor["citation_url"] == CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_URL
        assert anchor["citation_text"] == CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_TEXT
        assert (
            anchor["spam_feature_citation_text"]
            == CANONICAL_PEVNY_BAS_FRIDRICH_2010_SPAM_CITATION_TEXT
        )
        assert anchor["truncation_T_canonical"] == 4
        assert anchor["cooccurrence_order_canonical"] == 1
        assert anchor["epsilon_canonical"] == 1e-6
        assert anchor["n_directions_4_canonical"] == 4
        assert anchor["n_directions_8_canonical"] == 8

    def test_slot_uu_roadmap_anchor_canonical(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        anchor = result["slot_uu_roadmap_anchor"]
        assert anchor["commit_sha"] == "2b573f105"
        assert anchor["rank"] == 4
        assert anchor["score"] == "6/9"
        assert anchor["operator_binding_directive"] == "#10"
        assert "design_memo_path" in anchor

    def test_slot_yy_hill_reference_pattern_anchor_canonical(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        anchor = result["slot_yy_hill_reference_pattern_anchor"]
        assert "hill_canonical_inverse_steganalysis_li_wang_li_huang_2014" in anchor[
            "sister_pattern_path"
        ]
        assert "Axis 5" in anchor["axis_position_in_cascade"]
        assert "Axis 7" in anchor["axis_position_in_cascade"]
        assert anchor["reference_pattern_loc"] == 963

    def test_slot_ff_sister_cascade_anchor_canonical(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        anchor = result["slot_ff_sister_cascade_anchor"]
        assert anchor["commit_sha"] == "0adecdc5b"
        assert "pr110_opt_7_uniward_inverse_scorer_basis_expansion" in anchor[
            "sister_pattern_path"
        ]
        assert "Axis 1" in anchor["axis_position_in_cascade"]
        assert "Axis 7" in anchor["axis_position_in_cascade"]
        # canonical Cauchy-Schwarz META-LIFT-1+2 acknowledged (field name carries
        # the canonical Cauchy-Schwarz reference; field value cites canonical
        # orthogonality verification + Slot UU canonical anti-pattern #3
        # phantom-compounding warning per Pevný-Filler-Bas 2010 -> WOW ->
        # UNIWARD wavelet-residual lineage descent).
        partial_overlap_field = anchor[
            "canonical_cauchy_schwarz_meta_lift_acknowledged_partial_overlap"
        ]
        assert "orthogonality" in partial_overlap_field
        assert "UNIWARD" in partial_overlap_field
        assert "phantom-compounding" in partial_overlap_field

    def test_per_substrate_empirical_verification_stub_present(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        stub = result["per_substrate_empirical_verification_stub"]
        assert stub["status"] == "pending_per_substrate_empirical_verification"
        assert "paired_CUDA_RATIFICATION" in stub["next_action"]
        assert "Slot QQ" in stub["canonical_lesson_reference"]
        # canonical orthogonality verification requirement
        assert (
            "orthogonality"
            in stub["canonical_orthogonality_verification_required"]
        )
        assert "UNIWARD" in stub["canonical_orthogonality_verification_required"]

    def test_design_memo_path_canonical(self):
        image = _make_test_image(20, 20)
        config = HUGOConfig(sparse_k=10, n_pairs=20)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        path = result["design_memo_path"]
        assert "hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010" in path
        assert "20260529" in path


# ============================================================================
# Section 11: Parametrized strategy dispatch (4 strategies × paired-comparison)
# ============================================================================


@pytest.mark.parametrize(
    "strategy",
    [
        HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM,
        HUGOSPAMFeatureStrategy.CANONICAL_8_DIRECTION_SPAM,
        HUGOSPAMFeatureStrategy.PER_REGION_VARIABLE_DIRECTION,
        HUGOSPAMFeatureStrategy.PER_PIXEL_GLOBAL_SPAM_NORMALIZATION,
    ],
)
class TestParametrizedStrategyDispatch:
    """Canonical parametrized tests per Catalog #308 alternative-reducer enumeration."""

    def test_all_strategies_produce_valid_output(self, strategy):
        image = _make_test_image(15, 15)
        config = HUGOConfig(strategy=strategy, sparse_k=5, n_pairs=15)
        result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
            image, config
        )
        assert result["strategy"] == strategy.value
        assert result["n_selected_pairs"] == 5

    def test_all_strategies_emit_tier_a_routing_markers(self, strategy):
        image = _make_test_image(15, 15)
        config = HUGOConfig(strategy=strategy, sparse_k=5, n_pairs=15)
        result = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
            image, config
        )
        assert result["predicted_delta_adjustment"] == 0.0
        assert result["promotable"] is False
        assert result["axis_tag"] == "[predicted]"


# ============================================================================
# Section 12: Canonical truncation T variants
# ============================================================================


@pytest.mark.parametrize("T", [2, 3, 4])
def test_canonical_truncation_T_variants_produce_valid_output(T):
    """Canonical truncation T in {2, 3, 4} per Pevný-Bas-Fridrich literature range."""
    image = _make_test_image(15, 15)
    config = HUGOConfig(truncation_T=T, sparse_k=5, n_pairs=15)
    result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
        image, config
    )
    assert result["cost_matrix_summary"]["n_pixels"] == 15 * 15


# ============================================================================
# Section 13: Canonical cooccurrence_order variants
# ============================================================================


@pytest.mark.parametrize("order", [1, 2])
def test_canonical_cooccurrence_order_variants_produce_valid_output(order):
    """Canonical cooccurrence_order in {1, 2}."""
    image = _make_test_image(15, 15)
    config = HUGOConfig(cooccurrence_order=order, sparse_k=5, n_pairs=15)
    result = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
        image, config
    )
    assert result["strategy"] == HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM.value
