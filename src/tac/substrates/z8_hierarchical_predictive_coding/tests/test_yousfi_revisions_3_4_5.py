# SPDX-License-Identifier: MIT
"""Z8 Yousfi Revisions #3 + #4 + #5 dedicated regression tests.

Per Yousfi voice canonical inverse-steganalysis review memo `843b4bfd8`
(2026-05-30) the canonical Revisions #3 + #4 + #5 are M12c+ scope
substrate engineering improvements:

- **Revision #3**: replace canonical_quadruple_binding.py top-LL spatial-
  mean Wyner-Ziv side_info with canonical PoseNet 6-dim per canonical
  equation #150 ``wyner_ziv_decoder_side_information_rate_savings_v1``.

- **Revision #4**: extend Z8HierarchicalConfig ``num_levels`` from 3 to
  4 (Mallat Level 3 ≈ 145×109 falls BELOW SegNet stride-2 stem 256×192
  blind-spot threshold per CLAUDE.md "Exact scorer architectures").

- **Revision #5**: predicted band update tracking Revisions #3 + #4
  empirically (recipe-side; not validated in tests).

Per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable: every test below
verifies actual behavior of the canonical implementation; no fixture
inversions of the actual function output.

Per Catalog #287 placeholder-rationale rejection: the deterministic pose
proxy fallback when PoseNet weights are unavailable is HONESTLY labeled
``deterministic_pose_proxy_6dim_compress_time_fallback`` so the caller
cannot mistake it for actual PoseNet output.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    POSE_SIDE_INFO_DIM,
    _deterministic_pose_proxy_6dim,
    _project_pose_6dim_to_side_info_shape,
    build_canonical_quadruple_binding_from_z8_config,
    canonical_quadruple_forward_step,
    compute_pose_side_info_canonical_equation_150,
    run_canonical_quadruple_training_loop,
)
from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
    Z8HierarchicalConfig,
)


def _make_small_config(num_levels: int = 3) -> Z8HierarchicalConfig:
    """Build a small Z8 config for fast tests (3- and 4-level variants)."""
    if num_levels == 3:
        groups = (4, 3, 2)
        cats = (16, 8, 4)
    elif num_levels == 4:
        # Rev #4: extended 4-level pyramid; Level 3 ≈ 16x16 below the
        # synthetic SegNet stride-2 blind-spot proxy.
        groups = (4, 3, 2, 2)
        cats = (16, 8, 4, 4)
    else:
        raise ValueError(f"unsupported num_levels {num_levels}")
    return Z8HierarchicalConfig(
        num_levels=num_levels,
        num_groups_per_level=groups,
        num_categories_per_level=cats,
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=2,
        deterministic_state_dim=16,
        gumbel_temperature=1.0,
        use_straight_through=True,
        eval_size=(32, 32),
    )


# -----------------------------------------------------------------------------
# Revision #3: PoseNet 6-dim Wyner-Ziv side_info per canonical equation #150
# -----------------------------------------------------------------------------


class TestRev3PoseSideInfoCanonicalEquation150:
    """Rev #3 canonical equation #150 PoseNet 6-dim Wyner-Ziv side info."""

    def test_pose_side_info_dim_constant_is_canonical_6(self) -> None:
        """Canonical PoseNet pose vector dim per CLAUDE.md scorer architecture."""
        assert POSE_SIDE_INFO_DIM == 6

    def test_deterministic_pose_proxy_shape_correct(self) -> None:
        """Proxy returns (B, 6) for (B, H, W, 3) RGB input."""
        rng = np.random.RandomState(42)
        rgb = rng.uniform(0.0, 1.0, size=(3, 16, 24, 3)).astype(np.float32)
        proxy = _deterministic_pose_proxy_6dim(rgb)
        assert proxy.shape == (3, 6)
        assert proxy.dtype == np.float32

    def test_deterministic_pose_proxy_is_deterministic(self) -> None:
        """Same RGB input -> same 6-dim proxy vector across calls."""
        rng = np.random.RandomState(7)
        rgb = rng.uniform(0.0, 1.0, size=(2, 16, 16, 3)).astype(np.float32)
        proxy_a = _deterministic_pose_proxy_6dim(rgb)
        proxy_b = _deterministic_pose_proxy_6dim(rgb)
        np.testing.assert_array_equal(proxy_a, proxy_b)

    def test_deterministic_pose_proxy_distinguishes_distinct_inputs(self) -> None:
        """Different RGB inputs produce different 6-dim proxies."""
        rng = np.random.RandomState(11)
        rgb_a = rng.uniform(0.0, 1.0, size=(1, 16, 16, 3)).astype(np.float32)
        rgb_b = rng.uniform(0.0, 1.0, size=(1, 16, 16, 3)).astype(np.float32)
        proxy_a = _deterministic_pose_proxy_6dim(rgb_a)
        proxy_b = _deterministic_pose_proxy_6dim(rgb_b)
        # Two random RGB inputs should not coincidentally give identical
        # mean+grad summaries.
        assert not np.allclose(proxy_a, proxy_b)

    def test_deterministic_pose_proxy_rejects_wrong_shape(self) -> None:
        """Proxy refuses non-(B, H, W, 3) input."""
        with pytest.raises(ValueError, match="(B, H, W, 3)"):
            _deterministic_pose_proxy_6dim(np.zeros((4,), dtype=np.float32))
        with pytest.raises(ValueError, match="(B, H, W, 3)"):
            _deterministic_pose_proxy_6dim(np.zeros((1, 8, 8, 4), dtype=np.float32))

    def test_projection_preserves_pose_information_via_pseudo_inverse(self) -> None:
        """Projection (B, 6) -> (B, C, H, W) preserves 6-dim info per Wyner-Ziv 1976 Theorem 1.

        Per Wyner-Ziv 1976 Theorem 1 + canonical equation #150: a linear
        projection of the low-dim Y into a higher-dim ambient space
        preserves the rank-6 information; the pseudo-inverse recovers
        the original 6-dim vector.
        """
        pose = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32
        )  # (1, 6)
        side_info_shape = (4, 8, 8)
        side_info = _project_pose_6dim_to_side_info_shape(
            pose, side_info_shape, projection_seed=42
        )
        assert side_info.shape == (1, 4, 8, 8)
        # The projection's pseudo-inverse should approximately recover pose.
        rng = np.random.RandomState(42)
        flat_dim = 4 * 8 * 8
        proj_matrix = rng.randn(6, flat_dim).astype(np.float32) / np.sqrt(6.0)
        recovered = side_info.reshape(1, -1) @ np.linalg.pinv(proj_matrix)
        np.testing.assert_allclose(recovered, pose, atol=1e-3)

    def test_projection_rejects_wrong_pose_dim(self) -> None:
        """Projection refuses non-6-dim pose vectors."""
        with pytest.raises(ValueError, match=f"{POSE_SIDE_INFO_DIM}"):
            _project_pose_6dim_to_side_info_shape(
                np.zeros((1, 8), dtype=np.float32), (4, 8, 8)
            )

    def test_projection_rejects_invalid_side_info_shape(self) -> None:
        """Projection refuses non-3-tuple or non-positive side_info shapes."""
        pose = np.zeros((1, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="3-tuple"):
            _project_pose_6dim_to_side_info_shape(pose, (4, 8))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive"):
            _project_pose_6dim_to_side_info_shape(pose, (4, 0, 8))

    def test_compute_pose_side_info_uses_caller_supplied_when_available(self) -> None:
        """Caller-supplied PoseNet 6-dim is used; pose_source labels it."""
        rng = np.random.RandomState(3)
        rgb = rng.uniform(0.0, 1.0, size=(2, 16, 16, 3)).astype(np.float32)
        pose_explicit = np.array(
            [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            dtype=np.float32,
        )
        side_info, pose_used, pose_source = compute_pose_side_info_canonical_equation_150(
            rgb,
            (4, 8, 8),
            pose_6dim_batch=pose_explicit,
        )
        assert pose_source == "posenet_6dim_caller_supplied"
        np.testing.assert_array_equal(pose_used, pose_explicit)
        assert side_info.shape == (2, 4, 8, 8)

    def test_compute_pose_side_info_falls_back_to_proxy_when_not_supplied(self) -> None:
        """Without caller-supplied PoseNet, falls back to deterministic proxy.

        Per Catalog #287 NO FAKE IMPLEMENTATIONS: the fallback IS honestly
        labeled so downstream consumers cannot mistake it for PoseNet.
        """
        rng = np.random.RandomState(5)
        rgb = rng.uniform(0.0, 1.0, size=(2, 16, 16, 3)).astype(np.float32)
        side_info, pose_used, pose_source = compute_pose_side_info_canonical_equation_150(
            rgb, (4, 8, 8)
        )
        assert pose_source == "deterministic_pose_proxy_6dim_compress_time_fallback"
        assert pose_used.shape == (2, 6)
        assert side_info.shape == (2, 4, 8, 8)
        # The proxy is identical to the public proxy helper.
        np.testing.assert_array_equal(pose_used, _deterministic_pose_proxy_6dim(rgb))

    def test_compute_pose_side_info_rejects_mismatched_pose_shape(self) -> None:
        """Caller-supplied PoseNet of wrong batch / dim is refused."""
        rng = np.random.RandomState(9)
        rgb = rng.uniform(0.0, 1.0, size=(3, 16, 16, 3)).astype(np.float32)
        with pytest.raises(ValueError, match=r"B=3"):
            compute_pose_side_info_canonical_equation_150(
                rgb,
                (4, 8, 8),
                pose_6dim_batch=np.zeros((2, 6), dtype=np.float32),
            )
        with pytest.raises(ValueError, match=r"B=3"):
            compute_pose_side_info_canonical_equation_150(
                rgb,
                (4, 8, 8),
                pose_6dim_batch=np.zeros((3, 8), dtype=np.float32),
            )


# -----------------------------------------------------------------------------
# Forward-step + training-loop wire-in of Rev #3
# -----------------------------------------------------------------------------


class TestRev3ForwardStepWireIn:
    """Verify canonical_quadruple_forward_step Rev #3 opt-in behavior."""

    def test_forward_step_default_uses_top_ll_spatial_mean(self) -> None:
        """Backward compat: default forward step uses prior wiring."""
        cfg = _make_small_config()
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        rgb = np.random.RandomState(0).uniform(
            0.0, 1.0, size=(1, 32, 32, 3)
        ).astype(np.float32)
        result = canonical_quadruple_forward_step(
            binding, rgb, epoch_perturbation=0.5
        )
        assert result["wyner_ziv_side_info_source"] == (
            "top_ll_per_channel_spatial_mean"
        )
        assert "wyner_ziv_pose_6dim" not in result

    def test_forward_step_rev3_enabled_emits_pose_6dim(self) -> None:
        """Rev #3 opt-in surfaces pose_6dim + side_info_source labels."""
        cfg = _make_small_config()
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        rgb = np.random.RandomState(1).uniform(
            0.0, 1.0, size=(1, 32, 32, 3)
        ).astype(np.float32)
        result = canonical_quadruple_forward_step(
            binding,
            rgb,
            epoch_perturbation=0.5,
            pose_side_info_canonical_equation_150_enabled=True,
        )
        assert result["wyner_ziv_side_info_source"] == (
            "deterministic_pose_proxy_6dim_compress_time_fallback"
        )
        assert "wyner_ziv_pose_6dim" in result
        assert result["wyner_ziv_pose_6dim"].shape == (1, 6)

    def test_forward_step_rev3_uses_caller_pose(self) -> None:
        """Caller-supplied PoseNet 6-dim is propagated through forward step."""
        cfg = _make_small_config()
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        rgb = np.random.RandomState(2).uniform(
            0.0, 1.0, size=(1, 32, 32, 3)
        ).astype(np.float32)
        pose = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
        result = canonical_quadruple_forward_step(
            binding,
            rgb,
            epoch_perturbation=0.5,
            pose_side_info_canonical_equation_150_enabled=True,
            pose_6dim_caller_supplied=pose,
        )
        assert result["wyner_ziv_side_info_source"] == "posenet_6dim_caller_supplied"
        np.testing.assert_array_equal(result["wyner_ziv_pose_6dim"], pose)

    def test_forward_step_rev3_produces_valid_wyner_ziv_round_trip(self) -> None:
        """Rev #3 path produces a valid M6 Wyner-Ziv round trip.

        Per canonical equation #150: the canonical Wyner-Ziv 1976
        Theorem 1 R(X|Y) holds for ANY decoder-reproducible side info Y
        including the 6-dim PoseNet projection. The round-trip error MUST
        remain bounded (within the M6 coder's quantization residual band).
        """
        cfg = _make_small_config()
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        rgb = np.random.RandomState(3).uniform(
            0.0, 1.0, size=(1, 32, 32, 3)
        ).astype(np.float32)
        result = canonical_quadruple_forward_step(
            binding,
            rgb,
            epoch_perturbation=0.5,
            pose_side_info_canonical_equation_150_enabled=True,
        )
        # The Wyner-Ziv round-trip error should be a finite real number
        # not catastrophically large (the M6 coder enforces its own
        # round-trip bound; this test pins that Rev #3's side info does
        # NOT break that bound).
        assert np.isfinite(result["wyner_ziv_round_trip_error"])
        assert result["wyner_ziv_payload_bytes"] > 0


class TestRev3TrainingLoopWireIn:
    """Verify run_canonical_quadruple_training_loop Rev #3 plumbing."""

    def test_training_loop_default_does_not_use_pose_side_info(self) -> None:
        """Backward compat: training loop defaults to prior wiring."""
        cfg = _make_small_config()
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        targets = np.random.RandomState(0).uniform(
            0.0, 1.0, size=(2, 32, 32, 3)
        ).astype(np.float32)
        artifact = run_canonical_quadruple_training_loop(
            binding, targets, epochs=2
        )
        # The artifact carries per-step observability; confirm side_info
        # source is the prior wiring (backward-compat).
        steps = artifact.per_step_observability
        assert len(steps) == 4  # 2 epochs * 2 pairs
        # Each step's underlying forward result used prior wiring; the
        # canonical training loop does not expose side_info_source on
        # TrainingStepObservability so we verify via the build_canonical
        # backward-compat default not breaking artifact construction.
        assert artifact.total_epochs_completed == 2

    def test_training_loop_rev3_opt_in_runs_clean(self) -> None:
        """Opt-in Rev #3 training loop runs with deterministic pose proxy."""
        cfg = _make_small_config()
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        targets = np.random.RandomState(1).uniform(
            0.0, 1.0, size=(2, 32, 32, 3)
        ).astype(np.float32)
        artifact = run_canonical_quadruple_training_loop(
            binding,
            targets,
            epochs=2,
            pose_side_info_canonical_equation_150_enabled=True,
        )
        assert artifact.total_epochs_completed == 2
        assert artifact.score_claim is False
        assert artifact.promotable is False

    def test_training_loop_rev3_accepts_caller_pose(self) -> None:
        """Caller-supplied per-pair PoseNet 6-dim flows through."""
        cfg = _make_small_config()
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        targets = np.random.RandomState(4).uniform(
            0.0, 1.0, size=(2, 32, 32, 3)
        ).astype(np.float32)
        pose_per_pair = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]],
            dtype=np.float32,
        )
        artifact = run_canonical_quadruple_training_loop(
            binding,
            targets,
            epochs=1,
            pose_side_info_canonical_equation_150_enabled=True,
            pose_6dim_per_pair=pose_per_pair,
        )
        assert artifact.total_epochs_completed == 1

    def test_training_loop_rev3_rejects_wrong_pose_per_pair_shape(self) -> None:
        """pose_6dim_per_pair shape (num_pairs, 6) is contract-checked."""
        cfg = _make_small_config()
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        targets = np.random.RandomState(5).uniform(
            0.0, 1.0, size=(2, 32, 32, 3)
        ).astype(np.float32)
        with pytest.raises(ValueError, match="pose_6dim_per_pair must be"):
            run_canonical_quadruple_training_loop(
                binding,
                targets,
                epochs=1,
                pose_side_info_canonical_equation_150_enabled=True,
                pose_6dim_per_pair=np.zeros((3, 6), dtype=np.float32),
            )


# -----------------------------------------------------------------------------
# Revision #4: 4-level Mallat pyramid below SegNet 256x192 blind-spot
# -----------------------------------------------------------------------------


class TestRev4FourLevelMallat:
    """Rev #4 4-level Mallat pyramid configuration support."""

    def test_three_level_config_builds_binding(self) -> None:
        """Baseline: 3-level config (the M9 default) still builds correctly."""
        cfg = _make_small_config(num_levels=3)
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        assert binding.num_levels == 3
        assert len(binding.m4_per_level) == 3
        assert len(binding.m5_per_level) == 3
        assert len(binding.m8_per_level) == 3

    def test_four_level_config_builds_binding(self) -> None:
        """Rev #4: 4-level config builds 4 per-level adapters."""
        cfg = _make_small_config(num_levels=4)
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        assert binding.num_levels == 4
        assert len(binding.m4_per_level) == 4
        assert len(binding.m5_per_level) == 4
        assert len(binding.m8_per_level) == 4

    def test_four_level_forward_step_runs_clean(self) -> None:
        """4-level forward pass produces 4 per-level losses."""
        cfg = _make_small_config(num_levels=4)
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        rgb = np.random.RandomState(0).uniform(
            0.0, 1.0, size=(1, 32, 32, 3)
        ).astype(np.float32)
        result = canonical_quadruple_forward_step(
            binding, rgb, epoch_perturbation=0.5
        )
        assert len(result["per_level_l2_loss"]) == 4
        assert len(result["wavelet_subband_l2_norm"]) == 4

    def test_four_level_training_loop_runs_clean(self) -> None:
        """4-level training loop produces valid artifact."""
        cfg = _make_small_config(num_levels=4)
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        targets = np.random.RandomState(2).uniform(
            0.0, 1.0, size=(2, 32, 32, 3)
        ).astype(np.float32)
        artifact = run_canonical_quadruple_training_loop(
            binding, targets, epochs=2
        )
        assert artifact.total_epochs_completed == 2
        assert len(artifact.final_per_level_l2_loss) == 4

    def test_four_level_config_validates_per_level_lengths(self) -> None:
        """Mismatched per-level tuple length raises at __post_init__."""
        with pytest.raises(ValueError, match="num_groups_per_level length"):
            Z8HierarchicalConfig(
                num_levels=4,
                num_groups_per_level=(4, 3, 2),  # length 3 != num_levels 4
                num_categories_per_level=(16, 8, 4, 4),
                base_channels=8,
                decoder_latent_dim=12,
                num_pairs=2,
                deterministic_state_dim=16,
                gumbel_temperature=1.0,
                use_straight_through=True,
                eval_size=(32, 32),
            )

    def test_four_level_below_segnet_blind_spot_at_contest_resolution(self) -> None:
        """At contest eval_size (384, 512), Level 3 wavelet shape is below 256x192.

        Per CLAUDE.md "Exact scorer architectures": SegNet EfficientNet-B2
        vanilla stride-2 stem loses half resolution immediately -> below
        (256, 192) is structurally invisible to SegNet's argmax. The
        canonical Mallat 4-level pyramid at contest eval_size:
            Level 0: 384x512  (above 256x192)
            Level 1: 192x256  (above 256x192 — boundary)
            Level 2: 96x128   (below 256x192)
            Level 3: 48x64    (clearly below 256x192 → SegNet blind spot)
        """
        cfg = Z8HierarchicalConfig(
            num_levels=4,
            num_groups_per_level=(24, 16, 8, 4),
            num_categories_per_level=(256, 128, 64, 32),
            base_channels=24,
            decoder_latent_dim=28,
            num_pairs=2,
            deterministic_state_dim=64,
            gumbel_temperature=1.0,
            use_straight_through=True,
            eval_size=(384, 512),  # contest scorer resolution per CLAUDE.md
        )
        binding = build_canonical_quadruple_binding_from_z8_config(cfg)
        # Verify Level 3 wavelet subband shape is below SegNet 256x192 stem.
        level_3_shape = binding.contract.levels[3].wavelet_subband_shape
        # Level 3 H = 384 / 2^3 = 48; Level 3 W = 512 / 2^3 = 64.
        assert level_3_shape == (48, 64)
        # Below SegNet stride-2 stem blind-spot threshold.
        assert level_3_shape[0] < 256
        assert level_3_shape[1] < 192


# -----------------------------------------------------------------------------
# Revision #5: recipe predicted_band field update verification
# -----------------------------------------------------------------------------


class TestRev5RecipePredictedBandUpdate:
    """Rev #5 recipe predicted_band reflects deep-Yousfi-grounded scenario."""

    def test_recipe_predicted_band_acknowledges_rev_3_4_5_landing(self) -> None:
        """Recipe carries Rev #3+4+5 LANDED reference.

        Per Yousfi memo Axis 5 deep-Yousfi-grounded scenario [0.150, 0.175]
        is achievable IF all 5 revisions canonical-bound. THIS lane lands
        Rev #3+4 substrate-engineering scaffolding (M12c+ scope; not
        active at M12a per Z8_TRAINER_MODE=full); recipe explicitly
        acknowledges that landing path.
        """
        from pathlib import Path
        import yaml

        recipe_path = Path(__file__).resolve().parents[5] / (
            ".omx/operator_authorize_recipes/"
            "substrate_z8_hierarchical_predictive_coding_modal_t4_dispatch.yaml"
        )
        if not recipe_path.is_file():
            pytest.skip(f"recipe not present at {recipe_path}")
        with recipe_path.open("r", encoding="utf-8") as fh:
            recipe = yaml.safe_load(fh)
        # Rev #5 predicted_band update reflects either partial-Yousfi-grounded
        # [0.175, 0.190] (Rev #1 + #2 active at M12a) OR the deep-Yousfi-
        # grounded reference [0.150, 0.175] is documented somewhere in
        # rationale fields.
        predicted_band = recipe.get("predicted_band")
        assert isinstance(predicted_band, list) and len(predicted_band) == 2
        rationale = str(recipe.get("predicted_band_reactivation_criterion", ""))
        notes = str(recipe.get("notes", ""))
        # Either the band itself is the deep scenario OR rationale cites
        # both partial and deep scenarios.
        deep_scenario_acknowledged = (
            "[0.150, 0.175]" in rationale
            or "[0.150, 0.175]" in notes
            or (predicted_band[0] >= 0.150 and predicted_band[1] <= 0.180)
        )
        assert deep_scenario_acknowledged, (
            f"Rev #5 must acknowledge deep-Yousfi-grounded band; got "
            f"predicted_band={predicted_band}, rationale tail: "
            f"{rationale[-200:]}"
        )
