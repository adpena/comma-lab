# SPDX-License-Identifier: MIT
"""Tests for Slot RR FAKE rename + REAL perturbation via canonical helper.

Per Slot EEE 6-axis honesty audit 2026-05-29 HIGH-priority operator-routable +
operator binding 5-invariant standing directive 2026-05-29 + sister Slot YY
HILL bind helper pattern (commit ``32a70c051``).

Covers:

* Renamed function ``build_pose_axis_null_projection_menu_for_pr110_archive``
  returns expected menu-size constants (regression of existing 64 behavior).
* Old name ``apply_pose_axis_null_projection_to_pr110_archive`` still callable
  via deprecation alias (backward compat).
* NEW bind helper
  ``apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive``
  produces non-zero perturbation on real upstream/videos/0.mkv frames.
* NEW bind helper Tier A markers present per Catalog #341.
* NEW bind helper canonical Provenance present per Catalog #323.
* NEW bind helper AxisDecomposition emission per Catalog #356.
* NEW bind helper macOS-CPU advisory tagging per Catalog #192 NEVER promotable.
* Strategy enum non-degeneracy (per-pixel real-video produces different output
  than baseline 4-direction families across strategies).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet import (
    CANONICAL_FRAME1_MENU_TOTAL,
    CANONICAL_PIXEL_ROLL_FRAME1_COUNT,
    CANONICAL_DCT_CHROMA_FRAME1_COUNT,
    CANONICAL_HADAMARD_TILE_FRAME1_COUNT,
    CANONICAL_GAUSSIAN_NOISE_FRAME1_COUNT,
    MotionPairRepairPoseAxisNullProjectionConfig,
    PoseAxisNullProjectionStrategy,
    STRATEGY_PER_PIXEL_REAL_VIDEO_MLX,
    apply_pose_axis_null_projection_to_pr110_archive,
    apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive,
    build_canonical_frame1_pose_axis_null_projection_menu,
    build_pose_axis_null_projection_menu_for_pr110_archive,
)

CANONICAL_UPSTREAM_VIDEO = Path("upstream/videos/0.mkv")

# Skip the REAL-frame tests if canonical upstream video is absent; the rename
# + alias + strategy-constant tests do NOT need video.
REAL_VIDEO_PRESENT = CANONICAL_UPSTREAM_VIDEO.exists()
real_video_required = pytest.mark.skipif(
    not REAL_VIDEO_PRESENT,
    reason="canonical upstream/videos/0.mkv not present; per Catalog #213",
)


# --- Part 1: Rename + backward-compat alias ---------------------------------


def test_renamed_function_returns_canonical_menu_size_constants_per_pixel_roll():
    """Renamed build_*_menu function returns canonical PER_PIXEL_ROLL menu size."""
    cfg = MotionPairRepairPoseAxisNullProjectionConfig(
        substrate_id="fec6",
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
    )
    result = build_pose_axis_null_projection_menu_for_pr110_archive(cfg)
    assert result["canonical_menu_size"] == CANONICAL_PIXEL_ROLL_FRAME1_COUNT
    assert result["canonical_menu_total_all_strategies"] == CANONICAL_FRAME1_MENU_TOTAL


def test_renamed_function_returns_canonical_menu_size_constants_dct_chroma_basis():
    """Renamed build_*_menu function returns canonical DCT_CHROMA_BASIS menu size."""
    cfg = MotionPairRepairPoseAxisNullProjectionConfig(
        substrate_id="fec6",
        strategy=PoseAxisNullProjectionStrategy.DCT_CHROMA_BASIS,
    )
    result = build_pose_axis_null_projection_menu_for_pr110_archive(cfg)
    assert result["canonical_menu_size"] == CANONICAL_DCT_CHROMA_FRAME1_COUNT


def test_renamed_function_returns_canonical_menu_size_constants_hadamard_tile():
    """Renamed build_*_menu function returns canonical HADAMARD_TILE menu size."""
    cfg = MotionPairRepairPoseAxisNullProjectionConfig(
        substrate_id="fec6",
        strategy=PoseAxisNullProjectionStrategy.HADAMARD_TILE,
    )
    result = build_pose_axis_null_projection_menu_for_pr110_archive(cfg)
    assert result["canonical_menu_size"] == CANONICAL_HADAMARD_TILE_FRAME1_COUNT


def test_renamed_function_returns_canonical_menu_size_constants_gaussian_noise():
    """Renamed build_*_menu function returns canonical GAUSSIAN_NOISE menu size."""
    cfg = MotionPairRepairPoseAxisNullProjectionConfig(
        substrate_id="fec6",
        strategy=PoseAxisNullProjectionStrategy.GAUSSIAN_NOISE,
    )
    result = build_pose_axis_null_projection_menu_for_pr110_archive(cfg)
    assert result["canonical_menu_size"] == CANONICAL_GAUSSIAN_NOISE_FRAME1_COUNT


def test_old_name_callable_via_deprecation_alias_backward_compat():
    """Legacy apply_* name still callable as alias for renamed build_*_menu_for_*."""
    cfg = MotionPairRepairPoseAxisNullProjectionConfig(
        substrate_id="fec6",
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
    )
    legacy_result = apply_pose_axis_null_projection_to_pr110_archive(cfg)
    renamed_result = build_pose_axis_null_projection_menu_for_pr110_archive(cfg)
    # Legacy alias returns canonical-identical result to renamed function
    # because the alias IS the same function object per the rename.
    assert legacy_result["canonical_menu_size"] == renamed_result["canonical_menu_size"]
    assert legacy_result["verdict"] == renamed_result["verdict"]
    assert legacy_result["strategy"] == renamed_result["strategy"]


def test_old_name_is_identity_alias_for_renamed_function():
    """Legacy alias is identity object — apply_*_to_pr110_archive IS build_*_menu_for_*."""
    assert apply_pose_axis_null_projection_to_pr110_archive is (
        build_pose_axis_null_projection_menu_for_pr110_archive
    )


# --- Part 2: NEW REAL perturbation via canonical helper ---------------------


@real_video_required
def test_new_bind_helper_produces_nonzero_perturbation_per_pixel_roll():
    """NEW canonical bind helper produces REAL non-zero perturbation on real video."""
    result = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_frames=2,
        frame_resolution_hw=(48, 64),  # smaller for cheap test
        use_mlx=False,  # numpy fallback for reproducible test
    )
    # Canonical REAL-vs-FAKE disambiguator: aggregate_mean_abs_delta > 0
    # proves the perturbation actually modified frame bytes (vs the FAKE
    # legacy apply_* which returned ZERO perturbation per Slot EEE audit).
    assert result["aggregate_mean_abs_delta_across_modes"] > 0.0
    assert result["aggregate_max_abs_delta_across_modes"] > 0.0


@real_video_required
def test_new_bind_helper_produces_nonzero_perturbation_dct_chroma_basis():
    """NEW canonical bind helper REAL perturbation on DCT_CHROMA_BASIS strategy."""
    result = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.DCT_CHROMA_BASIS,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    assert result["aggregate_mean_abs_delta_across_modes"] > 0.0


@real_video_required
def test_new_bind_helper_produces_nonzero_perturbation_hadamard_tile():
    """NEW canonical bind helper REAL perturbation on HADAMARD_TILE strategy."""
    result = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.HADAMARD_TILE,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    assert result["aggregate_mean_abs_delta_across_modes"] > 0.0


@real_video_required
def test_new_bind_helper_produces_nonzero_perturbation_gaussian_noise():
    """NEW canonical bind helper REAL perturbation on GAUSSIAN_NOISE strategy."""
    result = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.GAUSSIAN_NOISE,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    assert result["aggregate_mean_abs_delta_across_modes"] > 0.0


@real_video_required
def test_new_bind_helper_tier_a_routing_markers_present_per_catalog_341():
    """NEW canonical bind helper carries canonical Tier A markers per Catalog #341."""
    result = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["score_claim"] is False
    # Inner canonical_routing_markers also carries them
    markers = result["canonical_routing_markers"]
    assert markers["predicted_delta_adjustment"] == 0.0
    assert markers["promotable"] is False
    assert markers["score_claim"] is False
    assert markers["evidence_grade"] == "predicted"


@real_video_required
def test_new_bind_helper_canonical_provenance_present_per_catalog_323():
    """NEW canonical bind helper canonical Provenance per Catalog #323."""
    result = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    prov = result["canonical_provenance"]
    assert isinstance(prov, dict)
    # Canonical Provenance per build_provenance_for_predicted produces:
    # artifact_kind == "predicted_from_model" + evidence_grade == "predicted"
    assert prov.get("artifact_kind") == "predicted_from_model"
    assert prov.get("evidence_grade") == "predicted"
    # Canonical Provenance carries source identification + sha256 per builder
    assert "source_path" in prov
    assert "source_sha256" in prov
    # Per Catalog #323: score_claim_valid + promotion_eligible both False
    assert prov.get("score_claim_valid") is False
    assert prov.get("promotion_eligible") is False


@real_video_required
def test_new_bind_helper_axis_decomposition_emission_per_catalog_356():
    """NEW canonical bind helper AxisDecomposition emission per Catalog #356."""
    result = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    decomp = result["predicted_axis_decomposition"]
    assert isinstance(decomp, dict)
    assert decomp["predicted_d_seg_delta"] == 0.0
    assert decomp["predicted_d_pose_delta"] == 0.0
    assert decomp["predicted_archive_bytes_delta"] == 0
    assert decomp["axis_tag"] == "[predicted]"
    assert "canonical_provenance" in decomp


@real_video_required
def test_new_bind_helper_macos_cpu_advisory_tagging_per_catalog_192():
    """NEW canonical bind helper tagged macOS-CPU advisory NEVER promotable per Catalog #192."""
    result = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    assert result["axis_tag"] == "[macOS-CPU advisory]"
    # Per Catalog #192: NEVER promotable
    assert result["promotable"] is False
    assert result["score_claim"] is False
    # Canonical Provenance carries the canonical advisory tag in
    # measurement_axis per Catalog #323 schema
    prov = result["canonical_provenance"]
    assert prov.get("measurement_axis") == "[macOS-CPU advisory]"
    assert prov.get("hardware_substrate") == "macos_arm64_mlx"


@real_video_required
def test_new_bind_helper_strategy_enum_non_degeneracy():
    """Different strategies produce different per-mode statistics (non-degenerate)."""
    result_pixel = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    result_dct = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.DCT_CHROMA_BASIS,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    result_hadamard = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.HADAMARD_TILE,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    result_gaussian = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.GAUSSIAN_NOISE,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )

    # Each strategy has its canonical menu count (per Catalog #308 enumeration).
    assert result_pixel["canonical_menu_size"] == CANONICAL_PIXEL_ROLL_FRAME1_COUNT
    assert result_dct["canonical_menu_size"] == CANONICAL_DCT_CHROMA_FRAME1_COUNT
    assert result_hadamard["canonical_menu_size"] == CANONICAL_HADAMARD_TILE_FRAME1_COUNT
    assert result_gaussian["canonical_menu_size"] == CANONICAL_GAUSSIAN_NOISE_FRAME1_COUNT

    # Canonical non-degeneracy: the aggregate delta differs across strategies.
    # (PER_PIXEL_ROLL produces translation-style deltas; DCT/Hadamard/Gaussian
    # produce additive perturbations of distinct spatial structure.)
    deltas = {
        "pixel": result_pixel["aggregate_mean_abs_delta_across_modes"],
        "dct": result_dct["aggregate_mean_abs_delta_across_modes"],
        "hadamard": result_hadamard["aggregate_mean_abs_delta_across_modes"],
        "gaussian": result_gaussian["aggregate_mean_abs_delta_across_modes"],
    }
    # At least 3 of 4 must be distinct (allowing rare coincidental ties).
    unique = len(set(round(d, 6) for d in deltas.values()))
    assert unique >= 3, f"strategies should produce distinct deltas; got {deltas}"


@real_video_required
def test_new_bind_helper_per_mode_statistics_shape():
    """Per-mode statistics structure per Catalog #305 observability."""
    result = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    stats = result["per_mode_perturbation_stats"]
    assert len(stats) == CANONICAL_PIXEL_ROLL_FRAME1_COUNT
    for stat in stats:
        assert "mode_id" in stat
        assert "family" in stat
        assert "mean_abs_delta" in stat
        assert "max_abs_delta" in stat
        assert "frame_count" in stat
        assert stat["frame_count"] == 2
        assert stat["family"] == "frame1_pixel_roll"


@real_video_required
def test_new_bind_helper_slot_rr_remediation_anchor_present():
    """Slot EEE Axis A + C remediation anchor present per Catalog #348."""
    result = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    anchor = result["slot_rr_remediation_anchor"]
    assert "slot_eee_audit_finding" in anchor
    assert "slot_eee_audit_axis_a" in anchor
    assert "slot_eee_audit_axis_c" in anchor
    assert "remediation_canonical_helper" in anchor
    assert "remediation_canonical_sister_pattern_landed_in_commit" in anchor
    assert "remediation_disambiguator_canonical" in anchor
    assert "canonical_paradigm_per_catalog_307" in anchor
    # The disambiguator must mention the canonical REAL vs FAKE test
    assert "aggregate_mean_abs_delta" in anchor["remediation_disambiguator_canonical"]


@real_video_required
def test_new_bind_helper_verdict_canonical_deferred_pending_paired_cuda():
    """Verdict is canonical DEFERRED_PENDING_PAIRED_CUDA per Catalog #325."""
    result = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    verdict = result["verdict"]
    assert "DEFERRED_PENDING_PAIRED_CUDA" in verdict
    assert "PER_PIXEL_REAL_VIDEO_MLX" in verdict
    assert "SMOKE_GREEN" in verdict


# --- Part 3: Strategy constant + observability ------------------------------


def test_strategy_per_pixel_real_video_mlx_constant_exists():
    """Canonical strategy identifier constant for the Slot RR REAL path."""
    assert STRATEGY_PER_PIXEL_REAL_VIDEO_MLX == "per_pixel_real_video_mlx"


@real_video_required
def test_new_bind_helper_used_mlx_flag_threaded():
    """The used_mlx flag in the return value matches caller intent."""
    result_numpy = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    assert result_numpy["used_mlx"] is False


@real_video_required
def test_new_bind_helper_elapsed_seconds_observability_per_catalog_305():
    """Wall-clock observability per Catalog #305."""
    result = apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive(
        strategy=PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL,
        num_frames=2,
        frame_resolution_hw=(48, 64),
        use_mlx=False,
    )
    assert result["elapsed_seconds"] > 0.0
    # Cheap test should complete in under 30 seconds.
    assert result["elapsed_seconds"] < 30.0


# --- Part 4: Internal perturbation helper ------------------------------------


def test_internal_perturbation_helper_pixel_roll_changes_array():
    """Internal _apply_perturbation_for_mode_canonical produces non-trivial output."""
    from tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet import (
        _apply_perturbation_for_mode_canonical,
    )

    luma = np.random.RandomState(42).rand(16, 16).astype(np.float32)
    mode_pixel = {
        "mode_id": "frame1_pixel_roll_dx+1_dy+0",
        "family": "frame1_pixel_roll",
        "params": {"dx": 1, "dy": 0},
    }
    result = _apply_perturbation_for_mode_canonical(luma, mode_pixel)
    assert result.shape == luma.shape
    assert result.dtype == np.float32
    # Pixel roll changes most pixels (shift) — non-trivial delta.
    delta = np.abs(result - luma)
    assert np.mean(delta) > 0.0


def test_internal_perturbation_helper_dct_chroma_basis_changes_array():
    """Internal helper produces non-trivial DCT chroma delta."""
    from tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet import (
        _apply_perturbation_for_mode_canonical,
    )

    luma = np.full((16, 16), 0.5, dtype=np.float32)  # constant baseline
    mode_dct = {
        "mode_id": "frame1_dct_chroma_u1_v2_amp_1",
        "family": "frame1_dct_chroma",
        "params": {"u": 1, "v": 2, "amp": 1},
    }
    result = _apply_perturbation_for_mode_canonical(luma, mode_dct)
    assert result.shape == luma.shape
    # DCT basis adds sinusoidal pattern; mean abs delta > 0.
    delta = np.abs(result - luma)
    assert np.mean(delta) > 0.0


def test_internal_perturbation_helper_hadamard_tile_changes_array():
    """Internal helper produces non-trivial Hadamard tile delta."""
    from tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet import (
        _apply_perturbation_for_mode_canonical,
    )

    luma = np.full((16, 16), 0.5, dtype=np.float32)
    mode_hadamard = {
        "mode_id": "frame1_hadamard_tile_amp_1",
        "family": "frame1_hadamard_tile",
        "params": {"amp": 1, "tile_size": 8},
    }
    result = _apply_perturbation_for_mode_canonical(luma, mode_hadamard)
    assert result.shape == luma.shape
    delta = np.abs(result - luma)
    assert np.mean(delta) > 0.0


def test_internal_perturbation_helper_gaussian_noise_deterministic_per_seed():
    """Internal helper Gaussian noise is deterministic per seed."""
    from tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet import (
        _apply_perturbation_for_mode_canonical,
    )

    luma = np.full((16, 16), 0.5, dtype=np.float32)
    mode_gauss = {
        "mode_id": "frame1_gaussian_noise_sigma1.0_seed42",
        "family": "frame1_gaussian_noise",
        "params": {"sigma": 1.0, "seed": 42},
    }
    result1 = _apply_perturbation_for_mode_canonical(luma, mode_gauss)
    result2 = _apply_perturbation_for_mode_canonical(luma, mode_gauss)
    np.testing.assert_array_equal(result1, result2)
    # Different seed produces different output.
    mode_gauss_2 = {
        "mode_id": "frame1_gaussian_noise_sigma1.0_seed1",
        "family": "frame1_gaussian_noise",
        "params": {"sigma": 1.0, "seed": 1},
    }
    result3 = _apply_perturbation_for_mode_canonical(luma, mode_gauss_2)
    assert not np.array_equal(result1, result3)


def test_internal_perturbation_helper_unknown_family_raises():
    """Internal helper rejects unknown family per defensive contract."""
    from tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet import (
        _apply_perturbation_for_mode_canonical,
    )

    luma = np.full((16, 16), 0.5, dtype=np.float32)
    bad_mode = {
        "mode_id": "frame1_unknown_family_foo",
        "family": "frame1_unknown_family",
        "params": {},
    }
    with pytest.raises(ValueError, match="Unknown canonical mode family"):
        _apply_perturbation_for_mode_canonical(luma, bad_mode)


def test_internal_perturbation_helper_rejects_non_2d():
    """Internal helper rejects 1D + 3D inputs per shape invariant."""
    from tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet import (
        _apply_perturbation_for_mode_canonical,
    )

    mode_pixel = {
        "mode_id": "frame1_pixel_roll_dx+1_dy+0",
        "family": "frame1_pixel_roll",
        "params": {"dx": 1, "dy": 0},
    }
    with pytest.raises(ValueError, match="must be 2D"):
        _apply_perturbation_for_mode_canonical(
            np.zeros(10, dtype=np.float32), mode_pixel
        )
    with pytest.raises(ValueError, match="must be 2D"):
        _apply_perturbation_for_mode_canonical(
            np.zeros((4, 4, 3), dtype=np.float32), mode_pixel
        )
