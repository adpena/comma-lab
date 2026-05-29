# SPDX-License-Identifier: MIT
"""Tests for PR110-OPT-4 Grouped Color/Geometry Calibration L0 SCAFFOLD.

Per Slot X cap≥4 maintenance + design memo
``.omx/research/pr110_opt_4_grouped_color_geometry_calibration_cross_pair_perturbation_reuse_design_20260529.md``.

Test coverage:

- Canonical Config dataclass invariants (frozen + ValueError on each bad path).
- :func:`compute_grouped_wire_bytes` correctness across all 4 GroupingStrategy
  enum values + Wave N+34 anchor regression.
- :func:`apply_grouped_color_geometry_calibration_to_pr110_archive` Tier A
  canonical-routing-markers contract per Catalog #341 + #357.
- Canonical :class:`AxisDecomposition` emission per Catalog #356 + canonical
  :class:`Provenance` per Catalog #323.
- Wave N+34 IMPLEMENTATION_FALSIFIED anchor preserved verbatim per
  Catalog #110/#113 HISTORICAL_PROVENANCE.
- Framework-agnostic routing (no torch/mlx import).
- Catalog #287 evidence tag discipline (axis_tag="[predicted]").
- Catalog #308 alternative-reducer enumeration.
"""

from __future__ import annotations

import hashlib
import math

import pytest

from tac.cathedral.consumer_contract import AxisDecomposition
from tac.composition.pr110_opt_4_grouped_color_geometry_calibration import (
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
    GroupedColorGeometryCalibrationConfig,
    GroupingStrategy,
    WAVE_N34_FEC6_BASELINE_WIRE_BYTES,
    WAVE_N34_GROUPED_WIRE_BYTES_FIXED,
    WAVE_N34_GROUPED_WIRE_BYTES_SHANNON,
    WAVE_N34_MEAN_PER_PAIR_COMPONENT_DELTA_S,
    WAVE_N34_N_GROUPS,
    WAVE_N34_N_MODES,
    WAVE_N34_N_PAIRS,
    WAVE_N34_SHANNON_ENTROPY_BITS,
    apply_grouped_color_geometry_calibration_to_pr110_archive,
    compute_grouped_wire_bytes,
)
from tac.composition.pr110_opt_4_grouped_color_geometry_calibration import (
    __init__ as pr110_opt_4_module,
)
from tac.provenance.contract import Provenance


# -----------------------------------------------------------------------------
# Canonical Wave N+34 anchor constants (regression guards)
# -----------------------------------------------------------------------------


class TestWaveN34AnchorConstantsPreserved:
    """Wave N+34 IMPLEMENTATION_FALSIFIED anchor preserved verbatim.

    Per Catalog #110/#113 HISTORICAL_PROVENANCE: the canonical constants
    MUST match Wave N+34 artifact JSON verbatim; any drift is a regression.
    """

    def test_fec6_baseline_249_bytes(self) -> None:
        assert WAVE_N34_FEC6_BASELINE_WIRE_BYTES == 249

    def test_grouped_shannon_coded_258_bytes(self) -> None:
        assert WAVE_N34_GROUPED_WIRE_BYTES_SHANNON == 258

    def test_grouped_fixed_width_383_bytes(self) -> None:
        assert WAVE_N34_GROUPED_WIRE_BYTES_FIXED == 383

    def test_n_modes_21(self) -> None:
        assert WAVE_N34_N_MODES == 21

    def test_n_groups_17(self) -> None:
        assert WAVE_N34_N_GROUPS == 17

    def test_n_pairs_600(self) -> None:
        assert WAVE_N34_N_PAIRS == 600

    def test_shannon_entropy_bits_matches_anchor(self) -> None:
        assert math.isclose(
            WAVE_N34_SHANNON_ENTROPY_BITS,
            3.3313203895039005,
            rel_tol=1e-12,
        )

    def test_mean_per_pair_component_delta_s_matches_anchor(self) -> None:
        assert math.isclose(
            WAVE_N34_MEAN_PER_PAIR_COMPONENT_DELTA_S,
            -0.0011704843740551626,
            rel_tol=1e-12,
        )

    def test_canonical_rate_multiplier(self) -> None:
        assert CANONICAL_RATE_MULTIPLIER == 25.0

    def test_canonical_rate_denom_bytes(self) -> None:
        assert CANONICAL_RATE_DENOM_BYTES == 37_545_489


# -----------------------------------------------------------------------------
# Canonical Config dataclass invariants
# -----------------------------------------------------------------------------


class TestGroupedColorGeometryCalibrationConfig:
    """Canonical config frozen-dataclass invariants per CLAUDE.md.

    Per Catalog #287 placeholder rejection sister discipline + Catalog #229
    premise-verification-before-edit: every invariant raises ValueError on
    bad input with explicit error message naming the field.
    """

    def test_defaults_match_wave_n34(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig()
        assert cfg.grouping_strategy == GroupingStrategy.SHANNON_CODED
        assert cfg.n_pairs == WAVE_N34_N_PAIRS
        assert cfg.n_modes == WAVE_N34_N_MODES
        assert cfg.n_groups_hint is None
        assert cfg.header_overhead_bytes_per_group == 1
        assert cfg.emit_axis_decomposition is True

    def test_frozen_dataclass_rejects_mutation(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig()
        with pytest.raises(Exception):
            cfg.n_pairs = 999  # type: ignore[misc]

    def test_grouping_strategy_must_be_enum(self) -> None:
        with pytest.raises(ValueError, match="grouping_strategy must be GroupingStrategy"):
            GroupedColorGeometryCalibrationConfig(grouping_strategy="shannon_coded")  # type: ignore[arg-type]

    def test_n_pairs_must_be_int(self) -> None:
        with pytest.raises(ValueError, match="n_pairs must be int"):
            GroupedColorGeometryCalibrationConfig(n_pairs=600.0)  # type: ignore[arg-type]

    def test_n_pairs_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_pairs must be > 0"):
            GroupedColorGeometryCalibrationConfig(n_pairs=0)

    def test_n_modes_must_be_int(self) -> None:
        with pytest.raises(ValueError, match="n_modes must be int"):
            GroupedColorGeometryCalibrationConfig(n_modes=21.0)  # type: ignore[arg-type]

    def test_n_modes_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_modes must be > 0"):
            GroupedColorGeometryCalibrationConfig(n_modes=-1)

    def test_n_groups_hint_must_be_int_or_none(self) -> None:
        with pytest.raises(ValueError, match="n_groups_hint must be int or None"):
            GroupedColorGeometryCalibrationConfig(n_groups_hint="17")  # type: ignore[arg-type]

    def test_n_groups_hint_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_groups_hint must be > 0"):
            GroupedColorGeometryCalibrationConfig(n_groups_hint=0)

    def test_n_groups_hint_must_be_le_n_modes(self) -> None:
        with pytest.raises(ValueError, match="n_groups_hint .* must be <="):
            GroupedColorGeometryCalibrationConfig(n_modes=10, n_groups_hint=20)

    def test_header_overhead_must_be_int(self) -> None:
        with pytest.raises(ValueError, match="header_overhead_bytes_per_group must be int"):
            GroupedColorGeometryCalibrationConfig(header_overhead_bytes_per_group=1.5)  # type: ignore[arg-type]

    def test_header_overhead_must_be_nonnegative(self) -> None:
        with pytest.raises(ValueError, match="header_overhead_bytes_per_group must be >= 0"):
            GroupedColorGeometryCalibrationConfig(header_overhead_bytes_per_group=-1)


# -----------------------------------------------------------------------------
# compute_grouped_wire_bytes
# -----------------------------------------------------------------------------


class TestComputeGroupedWireBytes:
    """Canonical analytical primitive correctness."""

    def test_input_length_must_match_n_pairs(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10)
        with pytest.raises(ValueError, match="length .* != config.n_pairs"):
            compute_grouped_wire_bytes([1, 2, 3], cfg)

    def test_input_must_be_sequence(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10)
        with pytest.raises(ValueError, match="must be Sequence"):
            compute_grouped_wire_bytes(42, cfg)  # type: ignore[arg-type]

    def test_shannon_coded_strategy_returns_expected_shape(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(
            n_pairs=10, grouping_strategy=GroupingStrategy.SHANNON_CODED
        )
        result = compute_grouped_wire_bytes([0] * 10, cfg)
        assert "wire_bytes_estimate" in result
        assert "header_bytes" in result
        assert "bitstream_bytes" in result
        assert "n_groups" in result
        assert "shannon_entropy_bits" in result
        assert "delta_vs_fec6_bytes" in result
        assert result["grouping_strategy"] == "shannon_coded"

    def test_fixed_width_strategy_returns_expected_shape(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(
            n_pairs=10, grouping_strategy=GroupingStrategy.FIXED_WIDTH
        )
        result = compute_grouped_wire_bytes([i % 4 for i in range(10)], cfg)
        assert result["grouping_strategy"] == "fixed_width"
        # 10 pairs * ceil(log2(4)) = 10 * 2 = 20 bits = 3 bytes (ceil)
        assert result["bitstream_bytes"] == math.ceil(20 / 8.0)

    def test_per_region_strategy_returns_expected_shape(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(
            n_pairs=10, grouping_strategy=GroupingStrategy.PER_REGION
        )
        result = compute_grouped_wire_bytes([0] * 10, cfg)
        assert result["grouping_strategy"] == "per_region"

    def test_per_temporal_window_strategy_returns_expected_shape(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(
            n_pairs=10, grouping_strategy=GroupingStrategy.PER_TEMPORAL_WINDOW
        )
        result = compute_grouped_wire_bytes([0] * 10, cfg)
        assert result["grouping_strategy"] == "per_temporal_window"

    def test_single_mode_zero_entropy(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10)
        result = compute_grouped_wire_bytes([0] * 10, cfg)
        assert result["shannon_entropy_bits"] == 0.0
        assert result["n_groups"] == 1

    def test_uniform_modes_log2_n_modes_entropy(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=16, n_modes=4)
        result = compute_grouped_wire_bytes([i % 4 for i in range(16)], cfg)
        assert math.isclose(result["shannon_entropy_bits"], 2.0, rel_tol=1e-9)
        assert result["n_groups"] == 4

    def test_fixed_width_single_group_zero_bitstream(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(
            n_pairs=10, grouping_strategy=GroupingStrategy.FIXED_WIDTH
        )
        result = compute_grouped_wire_bytes([0] * 10, cfg)
        assert result["bitstream_bytes"] == 0  # log2(1) = 0 bits

    def test_delta_vs_fec6_signed_positive_means_worse(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=600, n_modes=21)
        # Synthetic source matching Wave N+34 scope (uniform 21-mode distribution)
        modes = [i % 21 for i in range(600)]
        result = compute_grouped_wire_bytes(modes, cfg)
        # Synthetic uniform should exceed FEC6 baseline (Wave N+34 verdict)
        assert result["delta_vs_fec6_bytes"] > 0, (
            "Synthetic uniform 21-mode distribution should be WORSE than "
            "FEC6 baseline per Wave N+34 IMPLEMENTATION_FALSIFIED verdict"
        )

    def test_n_groups_hint_overrides_unique_count(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(
            n_pairs=10, n_modes=21, n_groups_hint=5
        )
        # Even though source has 10 unique modes, hint forces 5 groups
        result = compute_grouped_wire_bytes(list(range(10)), cfg)
        assert result["n_groups"] == 5


# -----------------------------------------------------------------------------
# apply_grouped_color_geometry_calibration_to_pr110_archive
# -----------------------------------------------------------------------------


class TestApplyGroupedColorGeometryCalibrationToPR110Archive:
    """Canonical L0 SCAFFOLD entry point Tier A contract."""

    def test_tier_a_canonical_routing_markers_per_catalog_341(self) -> None:
        """Catalog #341: Tier A consumer MUST return predicted_delta_adjustment=0.0 + promotable=False + axis_tag='[predicted]'."""
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10)
        result = apply_grouped_color_geometry_calibration_to_pr110_archive(
            [0] * 10, cfg
        )
        assert result["predicted_delta_adjustment"] == 0.0
        assert result["promotable"] is False
        assert result["axis_tag"] == "[predicted]"

    def test_verdict_is_deferred_per_catalog_308(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10)
        result = apply_grouped_color_geometry_calibration_to_pr110_archive(
            [0] * 10, cfg
        )
        assert result["verdict"] == "DEFERRED_PENDING_ALTERNATIVE_REDUCER"

    def test_wave_n34_anchor_preserved_in_result(self) -> None:
        """Catalog #110/#113 HISTORICAL_PROVENANCE: Wave N+34 anchor verbatim."""
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10)
        result = apply_grouped_color_geometry_calibration_to_pr110_archive(
            [0] * 10, cfg
        )
        anchor = result["wave_n34_anchor"]
        assert anchor["verdict"] == "IMPLEMENTATION_FALSIFIED"
        assert anchor["shannon_coded_wire_bytes"] == 258
        assert anchor["fixed_width_wire_bytes"] == 383
        assert anchor["fec6_baseline_wire_bytes"] == 249
        assert anchor["n_modes_in_source"] == 21
        assert anchor["n_groups_in_source"] == 17
        assert anchor["n_pairs_in_source"] == 600
        assert "IMPLEMENTATION_LEVEL_FALSIFICATION_PARADIGM_INTACT" in anchor[
            "catalog_307_classification"
        ]

    def test_catalog_308_alternative_reducer_enumeration_present(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10)
        result = apply_grouped_color_geometry_calibration_to_pr110_archive(
            [0] * 10, cfg
        )
        alts = result["wave_n34_anchor"]["catalog_308_alternative_reducer_enumeration"]
        assert len(alts) >= 4, "Catalog #308 requires N>=3 alternative reducers; landed 4"
        assert "widened_mode_catalog_>=40_modes" in alts
        assert "per_region_grouping" in alts
        assert "per_temporal_window_grouping" in alts
        assert "composition_with_opt_7_uniward_sparse_selector" in alts

    def test_horizon_class_plateau_adjacent(self) -> None:
        """Catalog #309 horizon_class declaration."""
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10)
        result = apply_grouped_color_geometry_calibration_to_pr110_archive(
            [0] * 10, cfg
        )
        assert result["horizon_class"] == "plateau_adjacent"

    def test_design_memo_path_cited(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10)
        result = apply_grouped_color_geometry_calibration_to_pr110_archive(
            [0] * 10, cfg
        )
        assert (
            "pr110_opt_4_grouped_color_geometry_calibration_cross_pair_perturbation_reuse_design_20260529.md"
            in result["design_memo_path"]
        )

    def test_axis_decomposition_emitted_per_catalog_356(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10, emit_axis_decomposition=True)
        result = apply_grouped_color_geometry_calibration_to_pr110_archive(
            [0] * 10, cfg
        )
        decomp = result["predicted_axis_decomposition"]
        assert decomp is not None
        assert decomp["axis_tag"] == "[predicted]"
        assert decomp["predicted_d_seg_delta"] == 0.0
        assert decomp["predicted_d_pose_delta"] == 0.0
        assert isinstance(decomp["predicted_archive_bytes_delta"], int)
        # Canonical Provenance dict-form per Catalog #323
        prov = decomp["canonical_provenance"]
        assert prov.get("artifact_kind") == "predicted_from_model"
        assert prov.get("evidence_grade") == "predicted"
        assert prov.get("promotion_eligible") is False
        assert prov.get("score_claim_valid") is False
        assert (
            "build_provenance_for_predicted"
            in prov.get("canonical_helper_invocation", "")
        )

    def test_axis_decomposition_omitted_when_disabled(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10, emit_axis_decomposition=False)
        result = apply_grouped_color_geometry_calibration_to_pr110_archive(
            [0] * 10, cfg
        )
        assert result["predicted_axis_decomposition"] is None

    def test_axis_decomposition_roundtrip_via_from_dict(self) -> None:
        """Catalog #356: AxisDecomposition.as_dict / from_dict roundtrip."""
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10)
        result = apply_grouped_color_geometry_calibration_to_pr110_archive(
            [0] * 10, cfg
        )
        decomp_dict = result["predicted_axis_decomposition"]
        restored = AxisDecomposition.from_dict(decomp_dict)
        assert restored.axis_tag == "[predicted]"
        assert restored.predicted_d_seg_delta == 0.0
        assert restored.predicted_d_pose_delta == 0.0

    def test_wire_analysis_includes_delta_vs_fec6(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=10)
        result = apply_grouped_color_geometry_calibration_to_pr110_archive(
            [0] * 10, cfg
        )
        wire = result["wire_analysis"]
        assert "wire_bytes_estimate" in wire
        assert "delta_vs_fec6_bytes" in wire
        assert wire["grouping_strategy"] == "shannon_coded"


# -----------------------------------------------------------------------------
# Framework-agnostic routing
# -----------------------------------------------------------------------------


class TestFrameworkAgnostic:
    """L0 SCAFFOLD encoder/decoder operates on list[int] / bytes / numpy only.

    Per Catalog #290 canonical-vs-unique decision: framework-agnostic core
    is consumable by both MLX-LOCAL smoke + Vast.ai/Modal CUDA dispatch via
    the same entry point.
    """

    def test_no_torch_import_at_module_level(self) -> None:
        # The module MUST NOT pull in torch (would break framework-agnostic
        # invariant + the macOS-CPU $0 advisory smoke runner)
        import sys
        for name in list(sys.modules.keys()):
            if name.startswith(
                "tac.composition.pr110_opt_4_grouped_color_geometry_calibration"
            ):
                del sys.modules[name]
        # Re-import and check torch wasn't loaded as a side effect
        torch_loaded_before = "torch" in sys.modules
        import tac.composition.pr110_opt_4_grouped_color_geometry_calibration  # noqa: F401
        torch_loaded_after = "torch" in sys.modules
        # We can't reject torch being loaded by another path, but importing
        # THIS module should not add it if it wasn't already loaded.
        if not torch_loaded_before:
            assert not torch_loaded_after, (
                "PR110-OPT-4 L0 SCAFFOLD imported torch as side effect; "
                "violates framework-agnostic invariant"
            )

    def test_no_mlx_import_at_module_level(self) -> None:
        # MLX is even more restrictive (only available on macOS)
        import sys
        mlx_loaded_before = "mlx" in sys.modules
        import tac.composition.pr110_opt_4_grouped_color_geometry_calibration  # noqa: F401
        mlx_loaded_after = "mlx" in sys.modules
        if not mlx_loaded_before:
            assert not mlx_loaded_after, (
                "PR110-OPT-4 L0 SCAFFOLD imported mlx as side effect; "
                "violates framework-agnostic invariant"
            )


# -----------------------------------------------------------------------------
# Determinism + observability
# -----------------------------------------------------------------------------


class TestDeterministicReproducibility:
    """Catalog #305 6-facet observability: diff-able across runs."""

    def test_same_input_produces_same_output(self) -> None:
        cfg = GroupedColorGeometryCalibrationConfig(n_pairs=600, n_modes=21)
        modes = [i % 21 for i in range(600)]
        a = compute_grouped_wire_bytes(modes, cfg)
        b = compute_grouped_wire_bytes(modes, cfg)
        # Provenance captured_at_utc differs across runs; wire analysis must match
        assert a == b

    def test_canonical_provenance_inputs_sha_deterministic(self) -> None:
        """Catalog #305 cite-able facet: inputs_sha256 deterministic."""
        from tac.composition.pr110_opt_4_grouped_color_geometry_calibration import (
            _compute_grouping_signature,
        )
        a = _compute_grouping_signature([0, 1, 2, 3, 4])
        b = _compute_grouping_signature([0, 1, 2, 3, 4])
        assert a == b
        # Different inputs produce different signatures
        c = _compute_grouping_signature([5, 4, 3, 2, 1])
        assert a != c
