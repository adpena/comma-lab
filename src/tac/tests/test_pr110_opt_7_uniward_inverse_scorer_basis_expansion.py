# SPDX-License-Identifier: MIT
"""Tests for PR110-OPT-7 UNIWARD inverse-scorer basis expansion L0 SCAFFOLD.

Per Slot FF cap=3 parallel-cascade per Slot CC Fridrich dissent binding
revision + design memo::

    .omx/research/pr110_opt_7_uniward_inverse_scorer_basis_expansion_\
fridrich_canonical_parallel_cascade_per_slot_cc_dissent_design_20260529.md

Test coverage:

- Canonical Config dataclass invariants (frozen + ValueError on each bad path).
- ``compute_uniward_weighted_perturbation_for_pr110_catalog`` correctness across
  all 4 BasisExpansionStrategy enum values + Wave N+34 OPT-7 anchor regression.
- ``apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive`` Tier A
  canonical-routing-markers contract per Catalog #341 + #357.
- Canonical :class:`AxisDecomposition` emission per Catalog #356 + canonical
  :class:`Provenance` per Catalog #323.
- Wave N+34 OPT-7 IMPLEMENTATION_FALSIFIED-at-WEIGHTING anchor preservation.
- Slot CC dissent binding revision commit ``18c6cd571`` anchor preservation.
- Sister Catalog #308 reactivation cross-reference to PR110-OPT-4 L0 SCAFFOLD.
"""

from __future__ import annotations

import hashlib
import math
import random

import pytest

from tac.cathedral.consumer_contract import AxisDecomposition
from tac.composition.pr110_opt_7_uniward_inverse_scorer_basis_expansion import (
    BasisExpansionStrategy,
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
    CANONICAL_SPARSE_K_DEFAULT,
    CANONICAL_WIDENED_K_DEFAULT,
    UniwardInverseScorerBasisConfig,
    WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES,
    WAVE_N34_OPT7_IMPROVEMENT_RATIO,
    WAVE_N34_OPT7_N_MODES,
    WAVE_N34_OPT7_N_PAIRS,
    WAVE_N34_OPT7_SPARSE_SELECTOR_K100_DELTA_BYTES_VS_FEC6,
    WAVE_N34_OPT7_SPARSE_SELECTOR_K100_PROPORTIONAL_SAVINGS,
    WAVE_N34_OPT7_SPARSE_SELECTOR_K100_WIRE_BYTES,
    WAVE_N34_OPT7_TOP_DECILE_UNIWARD_AGGREGATE_DELTA_S,
    WAVE_N34_OPT7_UNIWARD_CONCENTRATION_FACTOR,
    WAVE_N34_OPT7_UNIWARD_WEIGHTED_AGGREGATE_DELTA_S,
    WAVE_N34_OPT7_UNWEIGHTED_AGGREGATE_DELTA_S,
    apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive,
    compute_uniward_weighted_perturbation_for_pr110_catalog,
)
from tac.provenance.contract import Provenance


def _make_canonical_inputs(seed: int = 42) -> tuple[list[int], list[float]]:
    """Synthetic Wave N+34-shaped inputs: 600 pairs × 21 modes."""
    rng = random.Random(seed)
    modes = [rng.randint(0, WAVE_N34_OPT7_N_MODES - 1) for _ in range(WAVE_N34_OPT7_N_PAIRS)]
    scorer = [abs(rng.gauss(0.5, 0.3)) for _ in range(WAVE_N34_OPT7_N_PAIRS)]
    return modes, scorer


# -----------------------------------------------------------------------------
# Section 1: Wave N+34 OPT-7 anchor constants preserved per Catalog #110/#113
# -----------------------------------------------------------------------------


class TestWaveN34Opt7AnchorConstantsPreserved:
    def test_fec6_baseline_249_bytes(self) -> None:
        assert WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES == 249

    def test_unweighted_aggregate_delta_S_canonical(self) -> None:
        assert WAVE_N34_OPT7_UNWEIGHTED_AGGREGATE_DELTA_S == pytest.approx(
            -0.0011704843740551621, rel=1e-12
        )

    def test_uniward_weighted_aggregate_delta_S_canonical(self) -> None:
        assert WAVE_N34_OPT7_UNIWARD_WEIGHTED_AGGREGATE_DELTA_S == pytest.approx(
            -0.0009103568688898632, rel=1e-12
        )

    def test_improvement_ratio_negative_22_percent(self) -> None:
        # IMPLEMENTATION_FALSIFIED: WEIGHTING is -22.22% WORSE than unweighted
        assert WAVE_N34_OPT7_IMPROVEMENT_RATIO == pytest.approx(
            -0.22223919509842144, rel=1e-12
        )
        assert WAVE_N34_OPT7_IMPROVEMENT_RATIO < 0  # NEGATIVE = worse

    def test_top_decile_canonical(self) -> None:
        assert WAVE_N34_OPT7_TOP_DECILE_UNIWARD_AGGREGATE_DELTA_S == pytest.approx(
            -0.0004764121800100148, rel=1e-12
        )

    def test_uniward_concentration_factor_canonical(self) -> None:
        assert WAVE_N34_OPT7_UNIWARD_CONCENTRATION_FACTOR == pytest.approx(
            0.40702139265599685, rel=1e-12
        )

    def test_sparse_selector_K100_wire_103_bytes(self) -> None:
        assert WAVE_N34_OPT7_SPARSE_SELECTOR_K100_WIRE_BYTES == 103

    def test_sparse_selector_K100_delta_minus_146(self) -> None:
        assert (
            WAVE_N34_OPT7_SPARSE_SELECTOR_K100_DELTA_BYTES_VS_FEC6
            == 103 - WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES
        )
        assert WAVE_N34_OPT7_SPARSE_SELECTOR_K100_DELTA_BYTES_VS_FEC6 == -146

    def test_sparse_selector_K100_proportional_savings(self) -> None:
        # 25 * -146 / 37,545,489 ≈ -9.72e-5; Wave N+34 anchor was -7.94e-5
        # (includes distortion-axis contribution offset; preserved verbatim).
        assert WAVE_N34_OPT7_SPARSE_SELECTOR_K100_PROPORTIONAL_SAVINGS == pytest.approx(
            -7.940203000166914e-05, rel=1e-9
        )

    def test_n_pairs_600(self) -> None:
        assert WAVE_N34_OPT7_N_PAIRS == 600

    def test_n_modes_21(self) -> None:
        assert WAVE_N34_OPT7_N_MODES == 21

    def test_canonical_sparse_k_default_100(self) -> None:
        assert CANONICAL_SPARSE_K_DEFAULT == 100

    def test_canonical_widened_k_default_200(self) -> None:
        assert CANONICAL_WIDENED_K_DEFAULT == 200

    def test_canonical_rate_multiplier_25(self) -> None:
        assert CANONICAL_RATE_MULTIPLIER == 25.0

    def test_canonical_rate_denom_37545489(self) -> None:
        assert CANONICAL_RATE_DENOM_BYTES == 37_545_489


# -----------------------------------------------------------------------------
# Section 2: BasisExpansionStrategy enum
# -----------------------------------------------------------------------------


class TestBasisExpansionStrategyEnum:
    def test_4_canonical_values(self) -> None:
        # Catalog #308 alternative-reducer enumeration: N≥4 candidates
        assert len(BasisExpansionStrategy) == 4

    def test_sparse_k100_uniward_weighted_canonical(self) -> None:
        assert (
            BasisExpansionStrategy.SPARSE_K100_UNIWARD_WEIGHTED.value
            == "sparse_k100_uniward_weighted"
        )

    def test_widened_k200_uniward_weighted_sister_probe(self) -> None:
        assert (
            BasisExpansionStrategy.WIDENED_K200_UNIWARD_WEIGHTED.value
            == "widened_k200_uniward_weighted"
        )

    def test_per_region_uniward_weighted_sister_probe(self) -> None:
        assert (
            BasisExpansionStrategy.PER_REGION_UNIWARD_WEIGHTED.value
            == "per_region_uniward_weighted"
        )

    def test_all_pairs_uniward_weighted_degenerate(self) -> None:
        assert (
            BasisExpansionStrategy.ALL_PAIRS_UNIWARD_WEIGHTED.value
            == "all_pairs_uniward_weighted"
        )

    def test_enum_is_str_subclass(self) -> None:
        assert issubclass(BasisExpansionStrategy, str)


# -----------------------------------------------------------------------------
# Section 3: UniwardInverseScorerBasisConfig invariants
# -----------------------------------------------------------------------------


class TestUniwardInverseScorerBasisConfigInvariants:
    def test_defaults_use_canonical_anchors(self) -> None:
        cfg = UniwardInverseScorerBasisConfig()
        assert cfg.basis_strategy == BasisExpansionStrategy.SPARSE_K100_UNIWARD_WEIGHTED
        assert cfg.n_pairs == WAVE_N34_OPT7_N_PAIRS
        assert cfg.n_modes == WAVE_N34_OPT7_N_MODES
        assert cfg.sparse_k == CANONICAL_SPARSE_K_DEFAULT
        assert cfg.uniward_epsilon == 1e-6
        assert cfg.header_overhead_bytes == 3
        assert cfg.emit_axis_decomposition is True

    def test_frozen_dataclass(self) -> None:
        cfg = UniwardInverseScorerBasisConfig()
        with pytest.raises((AttributeError, Exception)):
            cfg.sparse_k = 999  # type: ignore[misc]

    def test_invalid_basis_strategy_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="basis_strategy must be"):
            UniwardInverseScorerBasisConfig(basis_strategy="not_an_enum")  # type: ignore[arg-type]

    def test_n_pairs_must_be_int(self) -> None:
        with pytest.raises(ValueError, match="n_pairs must be int"):
            UniwardInverseScorerBasisConfig(n_pairs="600")  # type: ignore[arg-type]

    def test_n_pairs_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_pairs must be > 0"):
            UniwardInverseScorerBasisConfig(n_pairs=0)

    def test_n_modes_must_be_int(self) -> None:
        with pytest.raises(ValueError, match="n_modes must be int"):
            UniwardInverseScorerBasisConfig(n_modes=21.5)  # type: ignore[arg-type]

    def test_n_modes_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_modes must be > 0"):
            UniwardInverseScorerBasisConfig(n_modes=-1)

    def test_sparse_k_must_be_int(self) -> None:
        with pytest.raises(ValueError, match="sparse_k must be int"):
            UniwardInverseScorerBasisConfig(sparse_k=100.5)  # type: ignore[arg-type]

    def test_sparse_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="sparse_k must be > 0"):
            UniwardInverseScorerBasisConfig(sparse_k=0)

    def test_sparse_k_must_be_leq_n_pairs(self) -> None:
        with pytest.raises(ValueError, match=r"sparse_k.*must be <= n_pairs"):
            UniwardInverseScorerBasisConfig(sparse_k=601, n_pairs=600)

    def test_uniward_epsilon_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="uniward_epsilon must be > 0"):
            UniwardInverseScorerBasisConfig(uniward_epsilon=0.0)

    def test_uniward_epsilon_rejects_bool(self) -> None:
        with pytest.raises(ValueError, match="uniward_epsilon must be number not bool"):
            UniwardInverseScorerBasisConfig(uniward_epsilon=True)  # type: ignore[arg-type]

    def test_header_overhead_must_be_int(self) -> None:
        with pytest.raises(ValueError, match="header_overhead_bytes must be int"):
            UniwardInverseScorerBasisConfig(header_overhead_bytes=3.0)  # type: ignore[arg-type]

    def test_header_overhead_must_be_nonneg(self) -> None:
        with pytest.raises(ValueError, match="header_overhead_bytes must be >= 0"):
            UniwardInverseScorerBasisConfig(header_overhead_bytes=-1)

    def test_bool_rejected_for_int_fields(self) -> None:
        with pytest.raises(ValueError, match="n_pairs must be int"):
            UniwardInverseScorerBasisConfig(n_pairs=True)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Section 4: compute_uniward_weighted_perturbation correctness
# -----------------------------------------------------------------------------


class TestComputeUniwardWeightedPerturbation:
    def test_sparse_k100_canonical_baseline_returns_canonical_shape(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = compute_uniward_weighted_perturbation_for_pr110_catalog(
            modes, scorer, cfg
        )
        assert set(res.keys()) == {
            "uniward_costs_per_pair",
            "selected_pair_indices",
            "wire_bytes_estimate",
            "fec6_baseline_wire_bytes",
            "delta_vs_fec6_bytes",
            "basis_strategy",
            "n_selected_pairs",
            "uniward_concentration_factor",
        }
        assert len(res["uniward_costs_per_pair"]) == 600
        assert res["n_selected_pairs"] == 100
        assert res["fec6_baseline_wire_bytes"] == 249
        assert res["basis_strategy"] == "sparse_k100_uniward_weighted"

    def test_widened_k200_uses_canonical_widened_default(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig(
            basis_strategy=BasisExpansionStrategy.WIDENED_K200_UNIWARD_WEIGHTED,
            sparse_k=CANONICAL_SPARSE_K_DEFAULT,  # default 100; should auto-widen to 200
        )
        res = compute_uniward_weighted_perturbation_for_pr110_catalog(
            modes, scorer, cfg
        )
        assert res["n_selected_pairs"] == CANONICAL_WIDENED_K_DEFAULT
        assert res["n_selected_pairs"] == 200

    def test_all_pairs_uniward_weighted_selects_all(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig(
            basis_strategy=BasisExpansionStrategy.ALL_PAIRS_UNIWARD_WEIGHTED,
        )
        res = compute_uniward_weighted_perturbation_for_pr110_catalog(
            modes, scorer, cfg
        )
        assert res["n_selected_pairs"] == 600
        assert res["selected_pair_indices"] == list(range(600))

    def test_per_region_degenerates_to_sparse_k_at_l0_scaffold(self) -> None:
        # Per design memo: per-region degenerates to sparse-K at L0 SCAFFOLD
        # (per-region requires per-pixel surface; out-of-scope L0).
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig(
            basis_strategy=BasisExpansionStrategy.PER_REGION_UNIWARD_WEIGHTED,
        )
        res = compute_uniward_weighted_perturbation_for_pr110_catalog(
            modes, scorer, cfg
        )
        # Should behave like SPARSE_K100 at L0
        assert res["n_selected_pairs"] == 100

    def test_uniward_costs_inversely_correlated_with_scorer_response(self) -> None:
        # Fridrich canonical: 1/(epsilon + scorer); higher scorer ⟹ lower cost
        modes = [0] * 10
        scorer_low_to_high = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
        cfg = UniwardInverseScorerBasisConfig(
            n_pairs=10, n_modes=21, sparse_k=3,
        )
        res = compute_uniward_weighted_perturbation_for_pr110_catalog(
            modes, scorer_low_to_high, cfg
        )
        costs = res["uniward_costs_per_pair"]
        # Costs should be strictly decreasing (since scorer is strictly increasing)
        for i in range(len(costs) - 1):
            assert costs[i] > costs[i + 1], (
                f"costs[{i}]={costs[i]} should be > costs[{i+1}]={costs[i+1]}"
            )
        # Sparse-K=3 should pick the 3 lowest-scorer pairs (highest UNIWARD cost)
        assert res["selected_pair_indices"] == [0, 1, 2]

    def test_selected_pair_indices_sorted_ascending(self) -> None:
        # Per Catalog #305 observability diff-able-across-runs facet
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig(sparse_k=50)
        res = compute_uniward_weighted_perturbation_for_pr110_catalog(
            modes, scorer, cfg
        )
        selected = res["selected_pair_indices"]
        assert selected == sorted(selected)

    def test_wire_bytes_estimate_canonical_sparse_k100_formula(self) -> None:
        # header=3 + K=100 * index_byte_width(2 for N=600) + K=100 * magnitude(1)
        # = 3 + 200 + 100 = 303
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = compute_uniward_weighted_perturbation_for_pr110_catalog(
            modes, scorer, cfg
        )
        expected_wire = 3 + 100 * 2 + 100 * 1  # header + indices + magnitudes
        assert res["wire_bytes_estimate"] == expected_wire

    def test_delta_vs_fec6_bytes_sign_convention(self) -> None:
        # Negative delta = savings; positive delta = worse
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = compute_uniward_weighted_perturbation_for_pr110_catalog(
            modes, scorer, cfg
        )
        assert (
            res["delta_vs_fec6_bytes"]
            == res["wire_bytes_estimate"] - WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES
        )

    def test_concentration_factor_in_zero_to_one(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = compute_uniward_weighted_perturbation_for_pr110_catalog(
            modes, scorer, cfg
        )
        c = res["uniward_concentration_factor"]
        assert 0.0 <= c <= 1.0

    def test_concentration_factor_one_for_all_pairs(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig(
            basis_strategy=BasisExpansionStrategy.ALL_PAIRS_UNIWARD_WEIGHTED,
        )
        res = compute_uniward_weighted_perturbation_for_pr110_catalog(
            modes, scorer, cfg
        )
        assert res["uniward_concentration_factor"] == pytest.approx(1.0, abs=1e-12)

    def test_invalid_length_mismatch_rejected(self) -> None:
        modes = [0] * 10
        scorer = [0.5] * 5
        cfg = UniwardInverseScorerBasisConfig(n_pairs=10, n_modes=21, sparse_k=3)
        with pytest.raises(ValueError, match="length must match"):
            compute_uniward_weighted_perturbation_for_pr110_catalog(
                modes, scorer, cfg
            )

    def test_invalid_n_pairs_mismatch_rejected(self) -> None:
        modes = [0] * 10
        scorer = [0.5] * 10
        cfg = UniwardInverseScorerBasisConfig(n_pairs=20, n_modes=21, sparse_k=3)
        with pytest.raises(ValueError, match="length must match config.n_pairs"):
            compute_uniward_weighted_perturbation_for_pr110_catalog(
                modes, scorer, cfg
            )

    def test_invalid_mode_value_rejected(self) -> None:
        modes = [0, 1, 99]
        scorer = [0.5, 0.5, 0.5]
        cfg = UniwardInverseScorerBasisConfig(n_pairs=3, n_modes=21, sparse_k=2)
        with pytest.raises(ValueError, match=r"all modes must be in \[0, 21\)"):
            compute_uniward_weighted_perturbation_for_pr110_catalog(
                modes, scorer, cfg
            )

    def test_deterministic_reproducibility(self) -> None:
        # Per Catalog #305 observability diff-able-across-runs facet
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        r1 = compute_uniward_weighted_perturbation_for_pr110_catalog(modes, scorer, cfg)
        r2 = compute_uniward_weighted_perturbation_for_pr110_catalog(modes, scorer, cfg)
        assert r1["wire_bytes_estimate"] == r2["wire_bytes_estimate"]
        assert r1["selected_pair_indices"] == r2["selected_pair_indices"]
        assert r1["uniward_concentration_factor"] == r2["uniward_concentration_factor"]


# -----------------------------------------------------------------------------
# Section 5: apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive
# (Tier A canonical-routing markers per Catalog #341 + #357)
# -----------------------------------------------------------------------------


class TestApplyTierACanonicalRoutingMarkers:
    def test_predicted_delta_adjustment_always_zero(self) -> None:
        # Per Catalog #341 Tier A observability-only contract
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        assert res["predicted_delta_adjustment"] == 0.0

    def test_promotable_always_false(self) -> None:
        # Per Catalog #341 + Catalog #192 macOS-CPU advisory NEVER promotable
        modes, scorer = _make_canonical_inputs()
        for strategy in BasisExpansionStrategy:
            cfg = UniwardInverseScorerBasisConfig(basis_strategy=strategy)
            res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
                modes, scorer, cfg
            )
            assert res["promotable"] is False

    def test_axis_tag_always_predicted(self) -> None:
        # Per Catalog #287 + Catalog #323 canonical Provenance
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        assert res["axis_tag"] == "[predicted]"

    def test_verdict_is_deferred_pending_paired_cuda(self) -> None:
        # Per Catalog #325 per-substrate symposium discipline
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        assert res["verdict"] == "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"


# -----------------------------------------------------------------------------
# Section 6: AxisDecomposition per Catalog #356 + canonical Provenance per #323
# -----------------------------------------------------------------------------


class TestAxisDecompositionEmission:
    def test_axis_decomposition_emitted_when_enabled(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig(emit_axis_decomposition=True)
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        decomp = res["predicted_axis_decomposition"]
        assert decomp is not None
        assert "predicted_d_seg_delta" in decomp
        assert "predicted_d_pose_delta" in decomp
        assert "predicted_archive_bytes_delta" in decomp
        assert "axis_tag" in decomp
        assert "canonical_provenance" in decomp

    def test_axis_decomposition_none_when_disabled(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig(emit_axis_decomposition=False)
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        assert res["predicted_axis_decomposition"] is None

    def test_axis_decomposition_seg_and_pose_zero_at_l0_scaffold(self) -> None:
        # L0 SCAFFOLD: analytical-upper-bound only; no actual perturbation
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        decomp = res["predicted_axis_decomposition"]
        assert decomp["predicted_d_seg_delta"] == 0.0
        assert decomp["predicted_d_pose_delta"] == 0.0

    def test_axis_decomposition_archive_bytes_delta_matches_wire_analysis(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        decomp = res["predicted_axis_decomposition"]
        assert (
            decomp["predicted_archive_bytes_delta"]
            == res["wire_analysis"]["delta_vs_fec6_bytes"]
        )

    def test_axis_decomposition_constructible_from_dict(self) -> None:
        # Round-trip canonical AxisDecomposition contract
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        decomp_dict = res["predicted_axis_decomposition"]
        # Reconstruct AxisDecomposition from dict-form
        reconstructed = AxisDecomposition(
            predicted_d_seg_delta=decomp_dict["predicted_d_seg_delta"],
            predicted_d_pose_delta=decomp_dict["predicted_d_pose_delta"],
            predicted_archive_bytes_delta=decomp_dict["predicted_archive_bytes_delta"],
            axis_tag=decomp_dict["axis_tag"],
            canonical_provenance=decomp_dict["canonical_provenance"],
        )
        assert reconstructed.axis_tag == "[predicted]"

    def test_canonical_provenance_carries_model_id(self) -> None:
        # Per Catalog #323 canonical Provenance umbrella
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        prov = res["predicted_axis_decomposition"]["canonical_provenance"]
        # Provenance dict-form should carry the canonical model_id identifying
        # the producing apparatus surface.
        # Look for the canonical module path in any field of provenance dict
        flat_str = repr(prov)
        assert "pr110_opt_7_uniward_inverse_scorer_basis_expansion" in flat_str


# -----------------------------------------------------------------------------
# Section 7: Wave N+34 OPT-7 anchor preservation in response
# -----------------------------------------------------------------------------


class TestWaveN34Opt7AnchorPreservation:
    def test_wave_n34_opt7_anchor_present(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        anchor = res["wave_n34_opt7_anchor"]
        assert anchor["verdict"] == "WEIGHTING_IMPLEMENTATION_FALSIFIED"
        assert anchor["improvement_ratio"] == WAVE_N34_OPT7_IMPROVEMENT_RATIO
        assert (
            anchor["sparse_selector_K100_wire_bytes_estimate"]
            == WAVE_N34_OPT7_SPARSE_SELECTOR_K100_WIRE_BYTES
        )
        assert (
            anchor["sparse_selector_K100_delta_bytes_vs_fec6"]
            == WAVE_N34_OPT7_SPARSE_SELECTOR_K100_DELTA_BYTES_VS_FEC6
        )

    def test_canonical_artifact_path_cited(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        anchor = res["wave_n34_opt7_anchor"]
        assert "wave_n34_pr110_opt_4_7_11_triple_artifacts" in anchor["canonical_artifact_path"]

    def test_catalog_307_classification_documented(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        cls = res["wave_n34_opt7_anchor"]["catalog_307_classification"]
        assert "WEIGHTING_IMPLEMENTATION_LEVEL_FALSIFICATION" in cls
        assert "BASIS_EXPANSION_PARADIGM_DEFERRED" in cls

    def test_catalog_308_alternative_reducer_enumeration_lists_4(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        alternatives = res["wave_n34_opt7_anchor"][
            "catalog_308_alternative_reducer_enumeration"
        ]
        assert len(alternatives) == 4
        assert any("sparse_k100" in a for a in alternatives)
        assert any("widened_k200" in a for a in alternatives)
        assert any("per_region" in a for a in alternatives)
        assert any("all_pairs" in a for a in alternatives)

    def test_fridrich_canonical_citation_preserved(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        citation = res["wave_n34_opt7_anchor"]["canonical_citation"]
        assert "Holub-Fridrich-Denemark 2014" in citation
        assert "Sallee 2003" in citation


# -----------------------------------------------------------------------------
# Section 8: Slot CC dissent binding revision anchor preservation
# -----------------------------------------------------------------------------


class TestSlotCCDissentAnchorPreservation:
    def test_slot_cc_commit_sha_preserved(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        anchor = res["slot_cc_dissent_anchor"]
        assert anchor["commit_sha"] == "18c6cd571"
        assert anchor["binding_revision"] == "fridrich_pr110_opt_7_uniward_parallel_cascade"
        assert anchor["council_tier"] == "T3"
        assert anchor["verdict"] == "PROCEED_WITH_REVISIONS"


# -----------------------------------------------------------------------------
# Section 9: Sister citation surfaces + horizon_class
# -----------------------------------------------------------------------------


class TestSisterCitationSurfaces:
    def test_design_memo_path_cited(self) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        assert "pr110_opt_7_uniward_inverse_scorer_basis_expansion" in res["design_memo_path"]
        assert res["design_memo_path"].endswith(".md")

    def test_sister_pr110_opt_4_module_path_cited(self) -> None:
        # Per Catalog #308 reactivation cross-reference
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        assert (
            "pr110_opt_4_grouped_color_geometry_calibration"
            in res["sister_pr110_opt_4_module_path"]
        )

    def test_horizon_class_plateau_adjacent(self) -> None:
        # Per Catalog #309 + design memo §"## horizon-class"
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig()
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        assert res["horizon_class"] == "plateau_adjacent"


# -----------------------------------------------------------------------------
# Section 10: Strategy dispatch end-to-end (4 enums × Tier A contract)
# -----------------------------------------------------------------------------


class TestStrategyDispatchEndToEnd:
    @pytest.mark.parametrize("strategy", list(BasisExpansionStrategy))
    def test_all_4_strategies_emit_tier_a_contract(
        self, strategy: BasisExpansionStrategy
    ) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig(basis_strategy=strategy)
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        # Tier A canonical-routing markers per Catalog #341
        assert res["predicted_delta_adjustment"] == 0.0
        assert res["promotable"] is False
        assert res["axis_tag"] == "[predicted]"
        # Strategy correctly threaded
        assert res["wire_analysis"]["basis_strategy"] == strategy.value

    @pytest.mark.parametrize("strategy", list(BasisExpansionStrategy))
    def test_all_4_strategies_emit_axis_decomposition_per_356(
        self, strategy: BasisExpansionStrategy
    ) -> None:
        modes, scorer = _make_canonical_inputs()
        cfg = UniwardInverseScorerBasisConfig(
            basis_strategy=strategy,
            emit_axis_decomposition=True,
        )
        res = apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
            modes, scorer, cfg
        )
        decomp = res["predicted_axis_decomposition"]
        assert decomp is not None
        assert decomp["axis_tag"] == "[predicted]"
