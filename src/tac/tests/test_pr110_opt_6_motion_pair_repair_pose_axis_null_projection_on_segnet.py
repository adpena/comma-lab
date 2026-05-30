# SPDX-License-Identifier: MIT
"""Canonical tests for PR110-OPT-6 motion-pair repair pose-axis null-projection on SegNet L0 SCAFFOLD.

Per task #1318 + Slot RR cap≥3 parallel-cascade directive 2026-05-29 + canonical
sister of Slot X PR110-OPT-4 + Slot FF PR110-OPT-7 L0 SCAFFOLD test templates.
"""

from __future__ import annotations

import importlib.util as _u
import sys
from pathlib import Path

import pytest


def _load_pr110_opt_6_module():
    """Load the canonical PR110-OPT-6 module without triggering
    ``src/tac/composition/__init__.py`` side-effects (Catalog #229 PV-discipline
    pattern; mirrors Slot X + Slot FF test template).
    """
    repo_root = Path(__file__).resolve().parents[3]
    mod_path = (
        repo_root
        / "src/tac/composition/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet/__init__.py"
    )
    spec = _u.spec_from_file_location("pr110_opt_6_module_under_test", str(mod_path))
    mod = _u.module_from_spec(spec)
    sys.modules["pr110_opt_6_module_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_pr110_opt_6_module()


# ----------------------------------------------------------------------
# Canonical constants section
# ----------------------------------------------------------------------


def test_canonical_opt12_anchor_constants(mod):
    """Per design memo § 3.1: OPT-12 PoseNet-null bottom-decile canonical anchor."""
    assert mod.OPT12_POSENET_NULL_TYPICAL_ABS_POSE_DELTA == 1.25e-7
    assert mod.OPT12_POSENET_NULL_DOMINANT_FAMILY_FRACTION == 0.875


def test_canonical_predicted_score_delta_band(mod):
    """Per design memo § 3.6 Dykstra-feasibility check."""
    assert mod.PREDICTED_SCORE_DELTA_BAND_LOWER == -0.0010
    assert mod.PREDICTED_SCORE_DELTA_BAND_UPPER == -0.0001
    assert mod.PREDICTED_SCORE_DELTA_BAND_LOWER < mod.PREDICTED_SCORE_DELTA_BAND_UPPER


def test_canonical_fec6_baseline_constants(mod):
    """Per design memo § 3.1 + Catalog #343 canonical frontier pointer."""
    assert mod.FEC6_BASELINE_WIRE_BYTES == 249
    assert mod.FEC6_BASELINE_ARCHIVE_SHA_PREFIX == "b7106c9bdbb8"
    assert mod.PR110_NUM_PAIRS == 600
    assert mod.PR110_K_SYMBOLS == 16


def test_canonical_menu_family_counts(mod):
    """Per design memo § 3.1 canonical 4-family menu enumeration."""
    assert mod.CANONICAL_PIXEL_ROLL_FRAME1_COUNT == 8
    assert mod.CANONICAL_DCT_CHROMA_FRAME1_COUNT == 16
    assert mod.CANONICAL_HADAMARD_TILE_FRAME1_COUNT == 3
    assert mod.CANONICAL_GAUSSIAN_NOISE_FRAME1_COUNT == 16
    assert mod.CANONICAL_FRAME1_MENU_TOTAL == 43
    assert mod.CANONICAL_FRAME1_MENU_TOTAL == (
        mod.CANONICAL_PIXEL_ROLL_FRAME1_COUNT
        + mod.CANONICAL_DCT_CHROMA_FRAME1_COUNT
        + mod.CANONICAL_HADAMARD_TILE_FRAME1_COUNT
        + mod.CANONICAL_GAUSSIAN_NOISE_FRAME1_COUNT
    )


def test_canonical_slot_qq_implementation_falsification_constants(mod):
    """Per design memo § 2 PHASE 0 PV: Slot QQ empirical falsification discipline."""
    assert mod.SLOT_MM_QUANTITATIVE_PREDICTION_DEPRECATED is True
    assert mod.SLOT_QQ_EMPIRICAL_FALSIFICATION_CHECKPOINT_UTC == "2026-05-29T13:33:40Z"
    assert mod.CANONICAL_FRIDRICH_YOUSFI_INVERSE_STEGANALYSIS_PARADIGM_INTACT is True


def test_canonical_equation_candidate_id(mod):
    """Per Catalog #344 + design memo § 4.3."""
    assert mod.CANONICAL_EQUATION_CANDIDATE_ID == (
        "pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_savings_v1"
    )
    assert mod.CANONICAL_ANTI_PATTERN_CANDIDATE_ID == (
        "pr110_opt_6_motion_pair_repair_segnet_null_axis_implementation_falsified_v1"
    )


# ----------------------------------------------------------------------
# Canonical PoseAxisNullProjectionStrategy enum
# ----------------------------------------------------------------------


def test_pose_axis_null_projection_strategy_enum_values(mod):
    """Per Catalog #308 alternative-reducer enumeration: 4 canonical strategies."""
    expected_values = {"per_pixel_roll", "dct_chroma_basis", "hadamard_tile", "gaussian_noise"}
    actual_values = {s.value for s in mod.PoseAxisNullProjectionStrategy}
    assert actual_values == expected_values


def test_pose_axis_null_projection_strategy_str_enum_inheritance(mod):
    """Per Catalog #265 canonical contract: enum inherits from str for JSON safety."""
    assert issubclass(mod.PoseAxisNullProjectionStrategy, str)
    assert mod.PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL == "per_pixel_roll"


# ----------------------------------------------------------------------
# Canonical Config dataclass __post_init__ invariants per Catalog #287
# ----------------------------------------------------------------------


def test_config_canonical_construction(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="fec6")
    assert cfg.substrate_id == "fec6"
    assert cfg.strategy == mod.PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL
    assert cfg.d_seg_epsilon == 1e-9
    assert cfg.target_d_pose_lower == 1e-7
    assert cfg.target_d_pose_upper == 1e-5
    assert cfg.emit_axis_decomposition is True


def test_config_rejects_empty_substrate_id(mod):
    with pytest.raises(ValueError, match="non-empty"):
        mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="")


def test_config_rejects_whitespace_substrate_id(mod):
    with pytest.raises(ValueError, match="non-empty"):
        mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="   ")


def test_config_rejects_placeholder_substrate_id(mod):
    """Per Catalog #287 sister discipline."""
    with pytest.raises(ValueError, match="placeholder"):
        mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="<substrate_id>")


def test_config_rejects_non_string_substrate_id(mod):
    with pytest.raises(ValueError, match="non-empty string"):
        mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id=42)  # type: ignore


def test_config_rejects_non_enum_strategy(mod):
    with pytest.raises(ValueError, match="PoseAxisNullProjectionStrategy"):
        mod.MotionPairRepairPoseAxisNullProjectionConfig(
            substrate_id="fec6", strategy="per_pixel_roll"  # type: ignore
        )


def test_config_rejects_negative_d_seg_epsilon(mod):
    with pytest.raises(ValueError, match="d_seg_epsilon"):
        mod.MotionPairRepairPoseAxisNullProjectionConfig(
            substrate_id="fec6", d_seg_epsilon=-1e-9
        )


def test_config_rejects_negative_d_pose_lower(mod):
    with pytest.raises(ValueError, match="target_d_pose_lower"):
        mod.MotionPairRepairPoseAxisNullProjectionConfig(
            substrate_id="fec6", target_d_pose_lower=-1e-7
        )


def test_config_rejects_inverted_d_pose_band(mod):
    with pytest.raises(ValueError, match="target_d_pose_upper"):
        mod.MotionPairRepairPoseAxisNullProjectionConfig(
            substrate_id="fec6", target_d_pose_lower=1e-5, target_d_pose_upper=1e-7
        )


def test_config_is_frozen(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="fec6")
    with pytest.raises((AttributeError, TypeError)):
        cfg.substrate_id = "new_id"  # type: ignore


# ----------------------------------------------------------------------
# Canonical menu construction helper
# ----------------------------------------------------------------------


def test_per_pixel_roll_menu_size(mod):
    menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(
        mod.PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL
    )
    assert len(menu) == 8


def test_per_pixel_roll_excludes_identity(mod):
    menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(
        mod.PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL
    )
    for m in menu:
        assert not (m["params"]["dx"] == 0 and m["params"]["dy"] == 0)


def test_per_pixel_roll_all_unique_dx_dy(mod):
    menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(
        mod.PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL
    )
    seen = {(m["params"]["dx"], m["params"]["dy"]) for m in menu}
    assert len(seen) == 8


def test_dct_chroma_basis_menu_size(mod):
    menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(
        mod.PoseAxisNullProjectionStrategy.DCT_CHROMA_BASIS
    )
    assert len(menu) == 16


def test_dct_chroma_basis_uses_8_freq_bins(mod):
    menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(
        mod.PoseAxisNullProjectionStrategy.DCT_CHROMA_BASIS
    )
    bins = {(m["params"]["u"], m["params"]["v"]) for m in menu}
    assert len(bins) == 8


def test_dct_chroma_basis_uses_2_amps(mod):
    menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(
        mod.PoseAxisNullProjectionStrategy.DCT_CHROMA_BASIS
    )
    amps = {m["params"]["amp"] for m in menu}
    assert amps == {1, 2}


def test_hadamard_tile_menu_size(mod):
    menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(
        mod.PoseAxisNullProjectionStrategy.HADAMARD_TILE
    )
    assert len(menu) == 3
    amps = {m["params"]["amp"] for m in menu}
    assert amps == {1, 2, 3}


def test_hadamard_tile_size_is_canonical_8(mod):
    menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(
        mod.PoseAxisNullProjectionStrategy.HADAMARD_TILE
    )
    for m in menu:
        assert m["params"]["tile_size"] == 8


def test_gaussian_noise_menu_size(mod):
    menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(
        mod.PoseAxisNullProjectionStrategy.GAUSSIAN_NOISE
    )
    assert len(menu) == 16


def test_gaussian_noise_4_sigmas_4_seeds(mod):
    menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(
        mod.PoseAxisNullProjectionStrategy.GAUSSIAN_NOISE
    )
    sigmas = {m["params"]["sigma"] for m in menu}
    seeds = {m["params"]["seed"] for m in menu}
    assert sigmas == {0.5, 1.0, 1.5, 2.0}
    assert seeds == {1, 2, 3, 4}


def test_total_menu_size_matches_constant(mod):
    """Canonical total across all 4 families == CANONICAL_FRAME1_MENU_TOTAL."""
    total = sum(
        len(mod.build_canonical_frame1_pose_axis_null_projection_menu(s))
        for s in mod.PoseAxisNullProjectionStrategy
    )
    assert total == mod.CANONICAL_FRAME1_MENU_TOTAL


def test_menu_mode_ids_all_unique_within_family(mod):
    for strategy in mod.PoseAxisNullProjectionStrategy:
        menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(strategy)
        ids = [m["mode_id"] for m in menu]
        assert len(ids) == len(set(ids)), f"Duplicate mode_id in {strategy.value}"


def test_menu_all_modes_have_frame1_family_prefix(mod):
    """Per design memo § 3.1: all canonical modes are frame-1 perturbations."""
    for strategy in mod.PoseAxisNullProjectionStrategy:
        menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(strategy)
        for m in menu:
            assert m["family"].startswith("frame1_"), (
                f"Mode {m['mode_id']} family {m['family']} should start with frame1_"
            )


def test_menu_descriptors_have_required_keys(mod):
    required_keys = {"mode_id", "family", "params", "description"}
    for strategy in mod.PoseAxisNullProjectionStrategy:
        menu = mod.build_canonical_frame1_pose_axis_null_projection_menu(strategy)
        for m in menu:
            assert required_keys.issubset(m.keys())


def test_menu_construction_deterministic(mod):
    """Per design memo § 3.5 observability: diff-able across runs."""
    for strategy in mod.PoseAxisNullProjectionStrategy:
        menu_1 = mod.build_canonical_frame1_pose_axis_null_projection_menu(strategy)
        menu_2 = mod.build_canonical_frame1_pose_axis_null_projection_menu(strategy)
        assert menu_1 == menu_2


# ----------------------------------------------------------------------
# Canonical AxisDecomposition per Catalog #356
# ----------------------------------------------------------------------


def test_build_axis_decomposition_canonical_zero_byte_invariant(mod):
    """Canonical zero-byte bolt-on per per-pair selector reuse."""
    decomp = mod.build_axis_decomposition_for_pr110_opt_6(
        substrate_id="fec6",
        current_archive_bytes=178517,
        current_d_pose=4.94e-5,
    )
    assert decomp["predicted_archive_bytes_delta"] == 0


def test_build_axis_decomposition_canonical_seg_null_default(mod):
    """Canonical d_seg = 0.0 (canonical SegNet-null filter)."""
    decomp = mod.build_axis_decomposition_for_pr110_opt_6(
        substrate_id="fec6", current_archive_bytes=178517, current_d_pose=4.94e-5
    )
    assert decomp["predicted_d_seg_delta"] == 0.0


def test_build_axis_decomposition_canonical_pose_delta_midpoint(mod):
    """Canonical midpoint of [-0.0010, -0.0001] band per § 3.6."""
    decomp = mod.build_axis_decomposition_for_pr110_opt_6(
        substrate_id="fec6", current_archive_bytes=178517, current_d_pose=4.94e-5
    )
    assert decomp["predicted_d_pose_delta"] == -0.0005


def test_build_axis_decomposition_tier_a_markers(mod):
    """Per Catalog #341 + #357 Tier A canonical-routing markers."""
    decomp = mod.build_axis_decomposition_for_pr110_opt_6(
        substrate_id="fec6", current_archive_bytes=178517, current_d_pose=4.94e-5
    )
    assert decomp["promotable"] is False
    assert decomp["predicted_delta_adjustment"] == 0.0
    assert decomp["axis_tag"] == "[predicted]"


def test_build_axis_decomposition_canonical_provenance_present(mod):
    """Per Catalog #323 canonical Provenance umbrella."""
    decomp = mod.build_axis_decomposition_for_pr110_opt_6(
        substrate_id="fec6", current_archive_bytes=178517, current_d_pose=4.94e-5
    )
    assert "canonical_provenance" in decomp
    assert isinstance(decomp["canonical_provenance"], dict)


def test_build_axis_decomposition_substrate_id_threaded(mod):
    decomp = mod.build_axis_decomposition_for_pr110_opt_6(
        substrate_id="pr106_format0d", current_archive_bytes=186876, current_d_pose=1e-5
    )
    assert decomp["substrate_id"] == "pr106_format0d"


def test_build_axis_decomposition_canonical_equation_candidate_ref(mod):
    decomp = mod.build_axis_decomposition_for_pr110_opt_6(
        substrate_id="fec6", current_archive_bytes=178517, current_d_pose=4.94e-5
    )
    assert decomp["canonical_equation_candidate_id"] == mod.CANONICAL_EQUATION_CANDIDATE_ID


def test_build_axis_decomposition_predicted_band_bounds_threaded(mod):
    decomp = mod.build_axis_decomposition_for_pr110_opt_6(
        substrate_id="fec6", current_archive_bytes=178517, current_d_pose=4.94e-5
    )
    assert decomp["predicted_score_delta_band_lower"] == mod.PREDICTED_SCORE_DELTA_BAND_LOWER
    assert decomp["predicted_score_delta_band_upper"] == mod.PREDICTED_SCORE_DELTA_BAND_UPPER


def test_build_axis_decomposition_deterministic_for_same_input(mod):
    """Per design memo § 3.5 observability + Catalog #105 byte-stable."""
    d1 = mod.build_axis_decomposition_for_pr110_opt_6(
        substrate_id="fec6", current_archive_bytes=178517, current_d_pose=4.94e-5
    )
    d2 = mod.build_axis_decomposition_for_pr110_opt_6(
        substrate_id="fec6", current_archive_bytes=178517, current_d_pose=4.94e-5
    )
    # Canonical provenance carries captured_at_utc; compare structural fields.
    assert d1["predicted_d_seg_delta"] == d2["predicted_d_seg_delta"]
    assert d1["predicted_d_pose_delta"] == d2["predicted_d_pose_delta"]
    assert d1["predicted_archive_bytes_delta"] == d2["predicted_archive_bytes_delta"]
    assert d1["substrate_id"] == d2["substrate_id"]


# ----------------------------------------------------------------------
# Canonical apply helper Tier A contract per Catalog #341
# ----------------------------------------------------------------------


def test_apply_returns_canonical_tier_a_markers(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="fec6")
    res = mod.apply_pose_axis_null_projection_to_pr110_archive(cfg)
    assert res["predicted_delta_adjustment"] == 0.0
    assert res["promotable"] is False
    assert res["axis_tag"] == "[predicted]"


def test_apply_verdict_is_canonical_deferred(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="fec6")
    res = mod.apply_pose_axis_null_projection_to_pr110_archive(cfg)
    assert res["verdict"] == "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"


def test_apply_canonical_menu_size_matches_strategy(mod):
    """Per design memo § 3.1 menu family counts."""
    for strategy, expected_size in [
        (mod.PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL, 8),
        (mod.PoseAxisNullProjectionStrategy.DCT_CHROMA_BASIS, 16),
        (mod.PoseAxisNullProjectionStrategy.HADAMARD_TILE, 3),
        (mod.PoseAxisNullProjectionStrategy.GAUSSIAN_NOISE, 16),
    ]:
        cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(
            substrate_id="fec6", strategy=strategy
        )
        res = mod.apply_pose_axis_null_projection_to_pr110_archive(cfg)
        assert res["canonical_menu_size"] == expected_size


def test_apply_canonical_menu_total_is_43(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="fec6")
    res = mod.apply_pose_axis_null_projection_to_pr110_archive(cfg)
    assert res["canonical_menu_total_all_strategies"] == 43


def test_apply_threads_substrate_id(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="pr106_format0d")
    res = mod.apply_pose_axis_null_projection_to_pr110_archive(cfg)
    assert res["substrate_id"] == "pr106_format0d"


def test_apply_threads_strategy_value(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(
        substrate_id="fec6",
        strategy=mod.PoseAxisNullProjectionStrategy.DCT_CHROMA_BASIS,
    )
    res = mod.apply_pose_axis_null_projection_to_pr110_archive(cfg)
    assert res["strategy"] == "dct_chroma_basis"


def test_apply_canonical_band_threaded(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="fec6")
    res = mod.apply_pose_axis_null_projection_to_pr110_archive(cfg)
    assert res["predicted_score_delta_band"] == (
        mod.PREDICTED_SCORE_DELTA_BAND_LOWER,
        mod.PREDICTED_SCORE_DELTA_BAND_UPPER,
    )


def test_apply_canonical_equation_candidate_id_threaded(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="fec6")
    res = mod.apply_pose_axis_null_projection_to_pr110_archive(cfg)
    assert res["canonical_equation_candidate_id"] == mod.CANONICAL_EQUATION_CANDIDATE_ID
    assert (
        res["canonical_anti_pattern_candidate_id"]
        == mod.CANONICAL_ANTI_PATTERN_CANDIDATE_ID
    )


def test_apply_canonical_slot_qq_falsification_disclaimer(mod):
    """Per design memo § 2 + Slot QQ empirical falsification."""
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="fec6")
    res = mod.apply_pose_axis_null_projection_to_pr110_archive(cfg)
    assert res["slot_mm_quantitative_prediction_deprecated"] is True
    assert res["canonical_fridrich_yousfi_paradigm_intact"] is True
    assert res["slot_qq_empirical_falsification_checkpoint_utc"] == "2026-05-29T13:33:40Z"


def test_apply_emits_axis_decomposition_when_enabled(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(
        substrate_id="fec6", emit_axis_decomposition=True
    )
    res = mod.apply_pose_axis_null_projection_to_pr110_archive(cfg)
    assert "axis_decomposition" in res
    assert res["axis_decomposition"]["predicted_d_seg_delta"] == 0.0


def test_apply_omits_axis_decomposition_when_disabled(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(
        substrate_id="fec6", emit_axis_decomposition=False
    )
    res = mod.apply_pose_axis_null_projection_to_pr110_archive(cfg)
    assert "axis_decomposition" not in res


def test_apply_with_custom_archive_bytes_and_pose(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="pr106_format0d")
    res = mod.apply_pose_axis_null_projection_to_pr110_archive(
        cfg, current_archive_bytes=186876, current_d_pose=1e-5
    )
    assert res["axis_decomposition"]["current_archive_bytes"] == 186876
    assert res["axis_decomposition"]["current_d_pose"] == 1e-5


def test_apply_fec6_baseline_constants_threaded(mod):
    cfg = mod.MotionPairRepairPoseAxisNullProjectionConfig(substrate_id="fec6")
    res = mod.apply_pose_axis_null_projection_to_pr110_archive(cfg)
    assert res["fec6_baseline_wire_bytes"] == 249
    assert res["fec6_baseline_archive_sha_prefix"] == "b7106c9bdbb8"


# ----------------------------------------------------------------------
# Canonical operator-routable paired-CUDA RATIFICATION targets
# ----------------------------------------------------------------------


def test_list_canonical_paired_cuda_targets_count(mod):
    """Per design memo § 4.4 + Slot LL sister-pattern template."""
    targets = mod.list_canonical_paired_cuda_ratification_targets()
    assert len(targets) == 4


def test_list_canonical_paired_cuda_targets_substrate_ids(mod):
    targets = mod.list_canonical_paired_cuda_ratification_targets()
    substrate_ids = {t["substrate_id"] for t in targets}
    assert substrate_ids == {"v14_v2_dqs1", "fec6", "pr106_format0d", "nscs06_v8_stacked"}


def test_list_canonical_paired_cuda_targets_envelope(mod):
    """Per design memo § 4.4 canonical $0.30 per target."""
    targets = mod.list_canonical_paired_cuda_ratification_targets()
    total_envelope = sum(t["paired_cuda_envelope_usd"] for t in targets)
    assert total_envelope == 1.20  # canonical 4 targets × $0.30
    for t in targets:
        assert t["paired_cuda_envelope_usd"] == 0.30


def test_list_canonical_paired_cuda_targets_predicted_band(mod):
    targets = mod.list_canonical_paired_cuda_ratification_targets()
    for t in targets:
        assert t["predicted_delta_s_band"] == (
            mod.PREDICTED_SCORE_DELTA_BAND_LOWER,
            mod.PREDICTED_SCORE_DELTA_BAND_UPPER,
        )


def test_list_canonical_paired_cuda_targets_required_keys(mod):
    required_keys = {
        "substrate_id",
        "canonical_sha_prefix",
        "frontier_role",
        "predicted_delta_s_band",
        "paired_cuda_envelope_usd",
    }
    targets = mod.list_canonical_paired_cuda_ratification_targets()
    for t in targets:
        assert required_keys.issubset(t.keys())


def test_list_canonical_paired_cuda_targets_canonical_frontier_pointer_match(mod):
    """Per Catalog #343 canonical frontier pointer SoT."""
    targets = mod.list_canonical_paired_cuda_ratification_targets()
    fec6 = next(t for t in targets if t["substrate_id"] == "fec6")
    assert fec6["canonical_sha_prefix"] == "b7106c9bdbb8"
    pr106 = next(t for t in targets if t["substrate_id"] == "pr106_format0d")
    assert pr106["canonical_sha_prefix"] == "9cb989cef519"


# ----------------------------------------------------------------------
# Canonical __all__ exports
# ----------------------------------------------------------------------


def test_canonical_all_exports_complete(mod):
    """Canonical __all__ exposes every public API per Catalog #265 contract.

    Updated 2026-05-29 per Slot RR FAKE rename + REAL perturbation migration
    via canonical shared helper (Slot EEE 6-axis honesty audit remediation):
    adds 3 NEW exports — the canonical menu-builder rename
    ``build_pose_axis_null_projection_menu_for_pr110_archive``, the canonical
    REAL perturbation apply
    ``apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive``,
    and the canonical strategy identifier ``STRATEGY_PER_PIXEL_REAL_VIDEO_MLX``.
    The legacy ``apply_pose_axis_null_projection_to_pr110_archive`` is
    preserved as a backward-compat alias for the menu-builder per CLAUDE.md
    "Forbidden premature KILL" non-negotiable.

    Re-updated 2026-05-29 per Slot GGG Part 3 real-scorer-axis verification
    (Slot EEE Audit Axis F closure): adds 6 NEW exports — the canonical
    scorer-axis verification helper
    ``apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive``,
    invariant constants ``SEGNET_ARGMAX_NULL_TOLERANCE``,
    ``POSENET_NULL_CARRIER_BAND_LOWER``, ``POSENET_NULL_CARRIER_BAND_UPPER``,
    and per-mode verdict strings
    ``VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE`` /
    ``VERDICT_NULL_PROJECTION_FALSIFIED_PER_MODE``. The new helper closes
    the cite-vs-impl gap by empirically verifying the function name's two
    claims (SegNet argmax invariance + PoseNet carrier band) on real
    upstream/videos/0.mkv frame pairs via the canonical
    ``tac.substrates.score_aware_common.score_pair_components`` helper.
    """
    required_exports = {
        "OPT12_POSENET_NULL_TYPICAL_ABS_POSE_DELTA",
        "OPT12_POSENET_NULL_DOMINANT_FAMILY_FRACTION",
        "PREDICTED_SCORE_DELTA_BAND_LOWER",
        "PREDICTED_SCORE_DELTA_BAND_UPPER",
        "FEC6_BASELINE_WIRE_BYTES",
        "FEC6_BASELINE_ARCHIVE_SHA_PREFIX",
        "PR110_NUM_PAIRS",
        "PR110_K_SYMBOLS",
        "CANONICAL_PIXEL_ROLL_FRAME1_COUNT",
        "CANONICAL_DCT_CHROMA_FRAME1_COUNT",
        "CANONICAL_HADAMARD_TILE_FRAME1_COUNT",
        "CANONICAL_GAUSSIAN_NOISE_FRAME1_COUNT",
        "CANONICAL_FRAME1_MENU_TOTAL",
        "SLOT_MM_QUANTITATIVE_PREDICTION_DEPRECATED",
        "SLOT_QQ_EMPIRICAL_FALSIFICATION_CHECKPOINT_UTC",
        "CANONICAL_FRIDRICH_YOUSFI_INVERSE_STEGANALYSIS_PARADIGM_INTACT",
        "CANONICAL_EQUATION_CANDIDATE_ID",
        "CANONICAL_ANTI_PATTERN_CANDIDATE_ID",
        "STRATEGY_PER_PIXEL_REAL_VIDEO_MLX",
        "PoseAxisNullProjectionStrategy",
        "MotionPairRepairPoseAxisNullProjectionConfig",
        "build_canonical_frame1_pose_axis_null_projection_menu",
        "build_axis_decomposition_for_pr110_opt_6",
        "build_pose_axis_null_projection_menu_for_pr110_archive",
        "apply_pose_axis_null_projection_to_pr110_archive",
        "apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive",
        # Slot GGG Part 3 real-scorer verification surfaces
        "SEGNET_ARGMAX_NULL_TOLERANCE",
        "POSENET_NULL_CARRIER_BAND_LOWER",
        "POSENET_NULL_CARRIER_BAND_UPPER",
        "VERDICT_NULL_PROJECTION_CONFIRMED_PER_MODE",
        "VERDICT_NULL_PROJECTION_FALSIFIED_PER_MODE",
        "apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive",
        "list_canonical_paired_cuda_ratification_targets",
        # Slot GGG SCALE-UP MATRIX 2026-05-30 (N modes x M pairs x contest resolution)
        "SCALE_UP_TIER_A_DEFAULT_N_MODES",
        "SCALE_UP_TIER_A_DEFAULT_NUM_PAIRS",
        "SCALE_UP_TIER_A_DEFAULT_RESOLUTION_HW",
        "VERDICT_SCALE_UP_ALL_MODES_CONFIRMED",
        "VERDICT_SCALE_UP_ALL_MODES_FALSIFIED",
        "VERDICT_SCALE_UP_PARTIAL_CONFIRMED",
        "build_unified_canonical_scale_up_menu",
        "rank_confirmed_modes_by_capacity_per_cost",
        "apply_pose_axis_null_projection_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution",
    }
    assert set(mod.__all__) == required_exports
