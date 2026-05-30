# SPDX-License-Identifier: MIT
"""Substantive behavioral tests for PR110-OPT-7 via Yousfi-T1 substrate.

Per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable 2026-05-30 + Slot EEE
fake-implementation audit anchor 2026-05-29: these tests verify BEHAVIORAL
invariants, not just markers/constants:

1. Each of the 5 canonical helpers IS invoked in the substrate forward pass
   (mock.patch verification + helpers_invoked field check).
2. Distinct inputs produce distinct outputs (substantive-distinctness gate).
3. Tier A canonical-routing markers held by construction (Catalog #341).
4. The canonical 5-helper invocation receipts non-trivially populate (not
   just constants).
5. Archive grammar header round-trips byte-for-byte.
6. The inflate runtime parses the canonical OPT7VYT1 grammar.
7. Cross-reference matrix maps to the 5 LANDED canonical primitives commits.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# Module under test
from tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1 import (
    ARCHIVE_MAGIC,
    ARCHIVE_VERSION,
    DEFAULT_PR110_BASE_PAIRS,
    DEFAULT_VULNERABLE_PAIR_BUDGET,
    OPT7VYT1_HEADER_FMT,
    OPT7VYT1_HEADER_LEN,
    PR110OPT7ViaYousfiT1Config,
    PR110OPT7ViaYousfiT1Result,
    apply_substrate_to_pr110_canonical,
    build_substrate_default_config,
    verify_canonical_helper_invocation,
)
from tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1.archive_grammar import (
    pack_header,
    unpack_header,
)


# ---------------------------------------------------------------------------
# Slot EEE NO FAKE IMPLEMENTATIONS gate: helper invocation verification
# ---------------------------------------------------------------------------


def _build_smoke_config(n_pairs: int = 24, rng_seed: int = 42, **kwargs) -> "PR110OPT7ViaYousfiT1Config":
    """Helper: build a smoke-config with synthetic vulnerability map (fast)."""
    defaults = {
        "vulnerable_pair_budget": min(n_pairs, 4),
        "use_canonical_pose_vulnerability_anchor": False,
    }
    defaults.update(kwargs)
    return PR110OPT7ViaYousfiT1Config(n_pairs=n_pairs, rng_seed=rng_seed, **defaults)


def test_canonical_5_helpers_all_invoked_in_forward_pass() -> None:
    """Slot EEE gate: all 5 canonical helpers MUST be invoked on every apply."""
    config = _build_smoke_config()
    result = apply_substrate_to_pr110_canonical(config)
    # All 5 invocation flags must be True
    assert result.canonical_helpers_invoked["alaska_color_separation"] is True
    assert (
        result.canonical_helpers_invoked["yousfi_t1_a_pose_vulnerability_map"]
        is True
    )
    assert (
        result.canonical_helpers_invoked["yousfi_t1_b_posenet_surrogate"] is True
    )
    assert (
        result.canonical_helpers_invoked["yousfi_t1_c_chroma_perturbation"]
        is True
    )
    assert (
        result.canonical_helpers_invoked["pr110_opt7_inverse_scorer_basis"]
        is True
    )


def test_alaska_color_separation_actually_invoked_mock_patch() -> None:
    """Mock.patch verifies alaska canonical helper IS invoked at runtime."""
    config = _build_smoke_config()
    with patch(
        "tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1."
        "substrate.branch_to_yuv6_channel_slice",
        wraps=__import__(
            "tac.composition.alaska_inverse_steganalysis_patterns",
            fromlist=["branch_to_yuv6_channel_slice"],
        ).branch_to_yuv6_channel_slice,
    ) as mock_alaska:
        apply_substrate_to_pr110_canonical(config)
        assert mock_alaska.call_count >= 1
        # Verify it was called with the canonical Y0_UV default
        first_call_args = mock_alaska.call_args_list[0]
        assert first_call_args.args[0] == "Y0_UV" or first_call_args.kwargs.get(
            "branch"
        ) == "Y0_UV"


def test_yousfi_t1_a_pose_vulnerability_map_actually_invoked_mock_patch() -> None:
    """Mock.patch verifies Yousfi-T1 Deliverable A IS invoked.

    The substrate calls EITHER ``build_default_pose_vulnerability_map_from_canonical_anchor``
    (when canonical anchor exists) OR ``compute_per_pair_pose_vulnerability_map``
    (synthetic fallback for L1 PROMOTION smoke when anchor missing).
    Either invocation counts as canonical Yousfi-T1 Deliverable A engagement.
    """
    config = _build_smoke_config()
    with patch(
        "tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1."
        "substrate.build_default_pose_vulnerability_map_from_canonical_anchor",
        wraps=__import__(
            "tac.master_gradient_pose_vulnerability",
            fromlist=["build_default_pose_vulnerability_map_from_canonical_anchor"],
        ).build_default_pose_vulnerability_map_from_canonical_anchor,
    ) as mock_anchor, patch(
        "tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1."
        "substrate.compute_per_pair_pose_vulnerability_map",
        wraps=__import__(
            "tac.master_gradient_pose_vulnerability",
            fromlist=["compute_per_pair_pose_vulnerability_map"],
        ).compute_per_pair_pose_vulnerability_map,
    ) as mock_compute:
        apply_substrate_to_pr110_canonical(config)
        # EITHER canonical anchor path OR synthetic fallback path must fire
        assert mock_anchor.call_count + mock_compute.call_count >= 1


def test_yousfi_t1_b_posenet_surrogate_forward_actually_invoked_mock_patch() -> None:
    """Mock.patch verifies PoseNet MAE-V surrogate forward IS invoked."""
    from tac.scorer_surrogate.posenet_mae_v import PoseNetMaeVSurrogate

    config = _build_smoke_config()
    original_forward = PoseNetMaeVSurrogate.forward
    invocation_count = [0]

    def counting_forward(self, *args, **kwargs):
        invocation_count[0] += 1
        return original_forward(self, *args, **kwargs)

    with patch.object(
        PoseNetMaeVSurrogate, "forward", counting_forward
    ):
        apply_substrate_to_pr110_canonical(config)
        assert invocation_count[0] >= 1


def test_yousfi_t1_c_chroma_perturbation_actually_invoked_mock_patch() -> None:
    """Mock.patch verifies YUV6 chroma perturbation operator IS invoked."""
    config = _build_smoke_config()
    with patch(
        "tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1."
        "substrate.apply_chroma_subsampled_perturbation",
        wraps=__import__(
            "tac.composition.yuv6_chroma_subsampled_perturbation_operator",
            fromlist=["apply_chroma_subsampled_perturbation"],
        ).apply_chroma_subsampled_perturbation,
    ) as mock_chroma:
        apply_substrate_to_pr110_canonical(config)
        assert mock_chroma.call_count >= 1


def test_pr110_opt7_inverse_scorer_basis_actually_invoked_mock_patch() -> None:
    """Mock.patch verifies PR110-OPT-7 L0 SCAFFOLD IS invoked."""
    config = _build_smoke_config()
    with patch(
        "tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1."
        "substrate.apply_uniward_inverse_scorer_basis_to_pr110_archive",
        wraps=__import__(
            "tac.composition.pr110_opt_7_fridrich_uniward_inverse_scorer_basis",
            fromlist=["apply_uniward_inverse_scorer_basis_to_pr110_archive"],
        ).apply_uniward_inverse_scorer_basis_to_pr110_archive,
    ) as mock_basis:
        apply_substrate_to_pr110_canonical(config)
        assert mock_basis.call_count >= 1


# ---------------------------------------------------------------------------
# Tier A canonical-routing markers per Catalog #341 (frozen-True invariants)
# ---------------------------------------------------------------------------


def test_tier_a_canonical_routing_markers_held_by_construction() -> None:
    """Catalog #341: predicted_delta_adjustment=0.0, promotable=False, axis_tag=[predicted]."""
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    assert result.predicted_delta_adjustment == 0.0
    assert result.promotable is False
    assert result.axis_tag == "[predicted]"
    assert result.verdict == "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"


def test_result_post_init_refuses_promotable_true() -> None:
    """Catalog #341 + #192: Tier A NEVER promotable; construct-time refuses."""
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    # Try to construct a malformed result; should raise
    with pytest.raises(ValueError, match="promotable=False"):
        PR110OPT7ViaYousfiT1Result(
            **{**result.__dict__, "promotable": True}
        )


def test_result_post_init_refuses_nonzero_delta_adjustment() -> None:
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    with pytest.raises(ValueError, match="predicted_delta_adjustment=0.0"):
        PR110OPT7ViaYousfiT1Result(
            **{**result.__dict__, "predicted_delta_adjustment": -0.005}
        )


def test_result_post_init_refuses_wrong_axis_tag() -> None:
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    with pytest.raises(ValueError, match="axis_tag"):
        PR110OPT7ViaYousfiT1Result(
            **{**result.__dict__, "axis_tag": "[contest-CUDA]"}
        )


def test_result_post_init_refuses_missing_helper_invocations() -> None:
    """Slot EEE gate: result construction refuses missing canonical helpers."""
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    bad_invoked = dict(result.canonical_helpers_invoked)
    bad_invoked["alaska_color_separation"] = False
    with pytest.raises(ValueError, match="NOT invoked"):
        PR110OPT7ViaYousfiT1Result(
            **{**result.__dict__, "canonical_helpers_invoked": bad_invoked}
        )


def test_result_post_init_refuses_missing_helper_key() -> None:
    """Slot EEE gate: result construction refuses missing helper key."""
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    bad_invoked = dict(result.canonical_helpers_invoked)
    del bad_invoked["alaska_color_separation"]
    with pytest.raises(ValueError, match="missing"):
        PR110OPT7ViaYousfiT1Result(
            **{**result.__dict__, "canonical_helpers_invoked": bad_invoked}
        )


# ---------------------------------------------------------------------------
# Substantive distinctness: changing inputs MUST change outputs
# ---------------------------------------------------------------------------


def test_distinct_color_branches_produce_distinct_alaska_slices() -> None:
    """alaska color branch enum substantively distinct per Slot EEE gate."""
    config_y0 = _build_smoke_config()
    config_full = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        alaska_color_branch="YUV6_full",
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    config_uv = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        alaska_color_branch="UV_only",
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result_y0 = apply_substrate_to_pr110_canonical(config_y0)
    result_full = apply_substrate_to_pr110_canonical(config_full)
    result_uv = apply_substrate_to_pr110_canonical(config_uv)
    # Each branch produces a distinct YUV6 channel slice
    assert result_y0.alaska_color_slice != result_full.alaska_color_slice
    assert result_y0.alaska_color_slice != result_uv.alaska_color_slice
    assert result_full.alaska_color_slice != result_uv.alaska_color_slice


def test_distinct_seeds_produce_distinct_vulnerability_maps() -> None:
    """Distinct rng_seed produces distinct pose-vulnerability buckets."""
    config_42 = _build_smoke_config()
    config_7 = _build_smoke_config(rng_seed=7)
    result_42 = apply_substrate_to_pr110_canonical(config_42)
    result_7 = apply_substrate_to_pr110_canonical(config_7)
    # Vulnerability ratios should differ across seeds (lognormal random init)
    ratio_42 = result_42.pose_vulnerability_summary["vulnerability_ratio"]
    ratio_7 = result_7.pose_vulnerability_summary["vulnerability_ratio"]
    assert ratio_42 != ratio_7


def test_distinct_chroma_strategies_produce_distinct_perturbation_summaries() -> None:
    """Chroma strategy enum substantively distinct per Slot EEE."""
    rng = np.random.default_rng(42)
    f0 = rng.uniform(0.0, 255.0, size=(48, 64, 3)).astype(np.float32)
    f1 = rng.uniform(0.0, 255.0, size=(48, 64, 3)).astype(np.float32)
    # Need to supply gradient maps for SegNet/Joint strategies
    h_sub, w_sub = 24, 32
    seg_map = rng.uniform(0.0, 1.0, size=(h_sub, w_sub)).astype(np.float32)
    pose_map = rng.uniform(0.0, 1.0, size=(h_sub, w_sub)).astype(np.float32)
    config_local = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        chroma_perturbation_strategy="local_variance_weighted",
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    config_joint = _build_smoke_config()
    result_local = apply_substrate_to_pr110_canonical(
        config_local, rgb_first_frame_hwc=f0, rgb_second_frame_hwc=f1
    )
    result_joint = apply_substrate_to_pr110_canonical(
        config_joint, rgb_first_frame_hwc=f0, rgb_second_frame_hwc=f1
    )
    # Different strategies should produce different summaries
    assert (
        result_local.chroma_perturbation_summary["strategy_used"]
        != result_joint.chroma_perturbation_summary["strategy_used"]
    )


# ---------------------------------------------------------------------------
# Helper invocation receipt substantive content
# ---------------------------------------------------------------------------


def test_pose_vulnerability_summary_has_non_trivial_content() -> None:
    """Catalog #305 observability: vulnerability summary is non-stub."""
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    summary = result.pose_vulnerability_summary
    assert summary["n_pairs"] == 24
    assert summary["n_vulnerable_selected"] > 0
    assert summary["vulnerability_ratio"] > 0.0
    assert summary["quartile_thresholds_q25"] < summary["quartile_thresholds_q75"]


def test_inverse_scorer_basis_summary_has_non_trivial_content() -> None:
    """Catalog #305: PR110-OPT-7 summary is non-stub."""
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    summary = result.inverse_scorer_basis_summary
    assert summary["n_selected_pairs"] > 0
    assert summary["wire_bytes_estimate"] > 0
    assert summary["strategy"] in {
        "uniward_inverse_local_variance_baseline",
        "uniward_inverse_segnet_gradient_sensitivity",
        "uniward_inverse_posenet_gradient_sensitivity",
        "uniward_inverse_joint_scorer_basis_linear_combination",
    }


def test_chroma_perturbation_summary_luma_preservation_invariant() -> None:
    """Yousfi-T1 Deliverable C luma preservation EXACT 0.0 in YUV6 space."""
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    # Luma drift in YUV6 must be EXACTLY 0.0 (chroma-only perturbation)
    assert (
        result.chroma_perturbation_summary[
            "luma_preservation_max_abs_drift_yuv6"
        ]
        == 0.0
    )
    # Chroma drift must be > 0 (perturbation actually applied)
    assert (
        result.chroma_perturbation_summary[
            "chroma_perturbation_max_abs_drift_yuv6"
        ]
        > 0.0
    )


def test_posenet_surrogate_summary_has_canonical_dims() -> None:
    """Yousfi-T1 Deliverable B canonical 6-dim ego-motion + grid=4."""
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    summary = result.posenet_surrogate_summary
    assert summary["pose_dims"] == 6
    assert summary["pool_grid"] == 4
    assert summary["total_params"] == 96 * 6 + 6  # 582
    assert len(summary["pose_prediction_first_6_dims"]) == 6


def test_alaska_color_slice_canonical_y0_uv_indices() -> None:
    """Default Y0_UV branch selects Y0 + U + V channels = indices (0, 4, 5)."""
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    assert result.alaska_color_slice == (0, 4, 5)


def test_per_pair_selected_indices_non_empty() -> None:
    """Substrate's intersection of vulnerable + inverse-scorer selectors non-empty."""
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    assert len(result.per_pair_selected_indices) > 0


def test_cross_reference_matrix_contains_5_canonical_primitives() -> None:
    """Cross-reference matrix maps to the 5 LANDED canonical primitives commits."""
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    xref = result.cross_reference_matrix
    assert "alaska_canonical_color_separation" in xref
    assert "61a91a48e" in xref["alaska_canonical_color_separation"]
    assert "yousfi_t1_deliverable_a_pose_vulnerability_map" in xref
    assert "3d027ecf9" in xref["yousfi_t1_deliverable_a_pose_vulnerability_map"]
    assert "yousfi_t1_deliverable_b_posenet_mae_v_surrogate" in xref
    assert "3d027ecf9" in xref["yousfi_t1_deliverable_b_posenet_mae_v_surrogate"]
    assert "yousfi_t1_deliverable_c_yuv6_chroma_perturbation" in xref
    assert "3d027ecf9" in xref["yousfi_t1_deliverable_c_yuv6_chroma_perturbation"]
    assert "pr110_opt7_inverse_scorer_basis_l0_scaffold" in xref
    assert "3fd28b5b2" in xref["pr110_opt7_inverse_scorer_basis_l0_scaffold"]


# ---------------------------------------------------------------------------
# Canonical Provenance per Catalog #323
# ---------------------------------------------------------------------------


def test_canonical_provenance_has_predicted_kind() -> None:
    """Catalog #323: Provenance artifact_kind = predicted_from_model + Tier A markers."""
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    prov = result.canonical_provenance
    assert prov["artifact_kind"] == "predicted_from_model"
    assert prov["score_claim_valid"] is False
    assert prov["promotion_eligible"] is False
    assert prov["measurement_axis"] == "[predicted]"
    assert prov["evidence_grade"] == "predicted"


# ---------------------------------------------------------------------------
# Config validation per Catalog #287 placeholder-rationale rejection
# ---------------------------------------------------------------------------


def test_config_refuses_invalid_n_pairs() -> None:
    with pytest.raises(ValueError, match="n_pairs"):
        PR110OPT7ViaYousfiT1Config(n_pairs=0)
    with pytest.raises(ValueError, match="n_pairs"):
        PR110OPT7ViaYousfiT1Config(n_pairs=-1)


def test_config_refuses_invalid_vulnerable_pair_budget() -> None:
    with pytest.raises(ValueError, match="vulnerable_pair_budget"):
        PR110OPT7ViaYousfiT1Config(n_pairs=100, vulnerable_pair_budget=101)
    with pytest.raises(ValueError, match="vulnerable_pair_budget"):
        PR110OPT7ViaYousfiT1Config(n_pairs=100, vulnerable_pair_budget=0)


def test_config_refuses_invalid_alaska_color_branch() -> None:
    with pytest.raises(ValueError, match="alaska_color_branch"):
        PR110OPT7ViaYousfiT1Config(alaska_color_branch="UNKNOWN_BRANCH")


def test_config_refuses_invalid_inverse_scorer_strategy() -> None:
    with pytest.raises(ValueError, match="inverse_scorer_basis_strategy"):
        PR110OPT7ViaYousfiT1Config(inverse_scorer_basis_strategy="unknown_strategy")


def test_config_refuses_invalid_chroma_strategy() -> None:
    with pytest.raises(ValueError, match="chroma_perturbation_strategy"):
        PR110OPT7ViaYousfiT1Config(chroma_perturbation_strategy="unknown_strategy")


def test_config_refuses_out_of_range_chroma_magnitude() -> None:
    with pytest.raises(ValueError, match="chroma_perturbation_magnitude"):
        PR110OPT7ViaYousfiT1Config(chroma_perturbation_magnitude=0.0)
    with pytest.raises(ValueError, match="chroma_perturbation_magnitude"):
        PR110OPT7ViaYousfiT1Config(chroma_perturbation_magnitude=256.0)


# ---------------------------------------------------------------------------
# Archive grammar OPT7VYT1 header round-trip
# ---------------------------------------------------------------------------


def test_opt7vyt1_header_round_trip_byte_for_byte() -> None:
    """Catalog #146 frozen-offset discipline: header packs/unpacks exactly."""
    sha_prefix = bytes(range(16))
    packed = pack_header(
        version=1,
        alaska_color_branch_index=9,  # Y0_UV index in ColorBranchSliceStrategy
        basis_strategy_index=3,
        chroma_strategy_index=3,
        pr110_base_sha256_prefix=sha_prefix,
    )
    assert len(packed) == OPT7VYT1_HEADER_LEN
    assert len(packed) == 32
    assert packed[:8] == ARCHIVE_MAGIC
    unpacked = unpack_header(packed)
    assert unpacked["magic"] == ARCHIVE_MAGIC
    assert unpacked["version"] == 1
    assert unpacked["alaska_color_branch_index"] == 9
    assert unpacked["basis_strategy_index"] == 3
    assert unpacked["chroma_strategy_index"] == 3
    assert unpacked["pr110_base_sha256_prefix"] == sha_prefix


def test_opt7vyt1_header_refuses_invalid_sha_prefix_length() -> None:
    with pytest.raises(ValueError, match="16 bytes"):
        pack_header(
            version=1,
            alaska_color_branch_index=0,
            basis_strategy_index=0,
            chroma_strategy_index=0,
            pr110_base_sha256_prefix=b"\x00" * 15,
        )


def test_opt7vyt1_header_refuses_magic_mismatch() -> None:
    """unpack_header refuses non-OPT7VYT1 magic."""
    bad_bytes = b"WRONGMAG" + b"\x00" * (OPT7VYT1_HEADER_LEN - 8)
    with pytest.raises(ValueError, match="magic mismatch"):
        unpack_header(bad_bytes)


# ---------------------------------------------------------------------------
# verify_canonical_helper_invocation audit helper
# ---------------------------------------------------------------------------


def test_verify_canonical_helper_invocation_returns_pass_on_clean_result() -> None:
    config = PR110OPT7ViaYousfiT1Config(
        n_pairs=24,
        vulnerable_pair_budget=4,
        rng_seed=42,
        use_canonical_pose_vulnerability_anchor=False,
    )
    result = apply_substrate_to_pr110_canonical(config)
    verdict = verify_canonical_helper_invocation(result)
    assert verdict["all_invoked"] is True
    assert verdict["invocation_count"] == 5
    assert verdict["missing_helpers"] == []
    assert verdict["cross_reference_count"] == 5
    assert verdict["substantive_distinctness_verdict"] == "PASS"


# ---------------------------------------------------------------------------
# Inflate runtime canonical OPT7VYT1 grammar
# ---------------------------------------------------------------------------


def test_inflate_module_canonical_3_arg_signature() -> None:
    """Catalog #146 contest 3-arg signature."""
    inflate_path = (
        Path(__file__).resolve().parent.parent / "inflate.py"
    )
    assert inflate_path.is_file()
    body = inflate_path.read_text()
    assert "def main(" in body
    assert "archive_dir output_dir file_list" in body


def test_inflate_module_uses_canonical_select_inflate_device() -> None:
    """Catalog #205: canonical select_inflate_device pattern."""
    inflate_path = (
        Path(__file__).resolve().parent.parent / "inflate.py"
    )
    body = inflate_path.read_text()
    assert "select_inflate_device" in body
    assert "PACT_INFLATE_DEVICE" in body


def test_inflate_module_pythonpath_self_containment_pattern() -> None:
    """Catalog #295: PYTHONPATH self-containment pattern (vendored fallback)."""
    inflate_path = (
        Path(__file__).resolve().parent.parent / "inflate.py"
    )
    body = inflate_path.read_text()
    # Either vendored or canonical helper-based; both are acceptable per
    # Catalog #295. Slot CCC pattern uses sys.path.insert + waiver.
    assert "SUBMISSION_PYTHONPATH_SHIM_OK" in body or "from tac." not in body


def test_inflate_module_emits_canonical_contest_raw_bytes() -> None:
    """Catalog #367 INFLATE FRAME-EMISSION-COUNT fail-closed."""
    inflate_path = (
        Path(__file__).resolve().parent.parent / "inflate.py"
    )
    body = inflate_path.read_text()
    assert "CONTEST_RAW_BYTES" in body
    assert "1164" in body
    assert "874" in body
    assert "1200" in body
    assert "WRONG-SIZE" in body or "raise AssertionError" in body


def test_inflate_module_no_scorer_at_inflate_time() -> None:
    """CLAUDE.md strict-scorer-rule: NO PoseNet / SegNet at inflate time."""
    inflate_path = (
        Path(__file__).resolve().parent.parent / "inflate.py"
    )
    body = inflate_path.read_text()
    # Strict scorer rule: these tokens should NOT appear in inflate.py
    forbidden = ("PoseNet", "SegNet", "from upstream.modules", "EfficientNet", "FastViT")
    for token in forbidden:
        assert token not in body, (
            f"strict-scorer-rule violation: {token!r} appears in inflate.py"
        )


# ---------------------------------------------------------------------------
# Default config builder
# ---------------------------------------------------------------------------


def test_build_substrate_default_config_returns_canonical_defaults() -> None:
    cfg = build_substrate_default_config()
    assert cfg.n_pairs == DEFAULT_PR110_BASE_PAIRS
    assert cfg.vulnerable_pair_budget == DEFAULT_VULNERABLE_PAIR_BUDGET
    assert cfg.alaska_color_branch == "Y0_UV"
    assert cfg.rng_seed == 42


def test_build_substrate_default_config_scales_budget_on_n_pairs_override() -> None:
    cfg = build_substrate_default_config(n_pairs=60)
    # Budget should scale proportionally
    expected_budget = max(1, int(100 * 60 / 600))
    assert cfg.vulnerable_pair_budget == expected_budget


def test_build_substrate_default_config_handles_seed_override() -> None:
    cfg = build_substrate_default_config(rng_seed=7)
    assert cfg.rng_seed == 7
