# SPDX-License-Identifier: MIT
"""Tests for C1 -> Z5 routing + autopilot Decision 6 halve (2026-05-14).

Per the 2026-05-14 grand-council reconvening (commit 3a63e3394) BINDING
verdicts:

- Decision 1 (β UNANIMOUS 11/11): route C1 compute INTO Z5
  (~50 LOC plumbing). Z5 already L1 scaffolded with canonical Hafner
  DreamerV3 pattern; Z5's HierarchicalPredictor self-arbitrates the
  ``predictive_world_model`` vs ``identity_predictor`` regime.
- Decision 6 (HALF-MEASURE): halve autopilot v2 ranker C1-row class-shift
  literature-anchor reward 0.01 -> 0.005 for C1-class tokens (Hafner /
  DreamerV3 / Ha-Schmidhuber). Other class-shift tokens (Wyner-Ziv /
  cooperative-receiver / predictive-coding / foveation / MDL-IBPS /
  time-traveler / etc.) keep the FULL 0.01 reward.

Critical CLAUDE.md guarantees verified by these tests:

- Decision δ (DROP WorldModelModule) was REJECTED 11/11; the test class
  ``TestWorldModelModulePreserved`` enforces that the class definition +
  composition path remain intact. DO NOT REMOVE.
- Foveation finding (``ego_motion_radial`` 57% margin) is INDEPENDENT of
  this review and STANDS — ``TestFoveationBehaviorUnchanged`` enforces.
- Z5's HierarchicalPredictor mode-arbitration is the dispositive test
  primitive — ``TestZ5RoutingMockArbitration`` exercises both modes and
  verifies the routing kernel matches Z5's canonical behavior.

Cross-references:
- Council ledger ``.omx/research/grand_council_c1_post_probe_v2_reconvene_20260514.md``
- Memory file ``feedback_c1_council_reconvene_post_probe_v2_landed_20260514.md``
- Sister substrate Z5: ``src/tac/substrates/z5_predictive_coding_world_model/architecture.py``
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from tac.substrates.c1_world_model_foveation import (
    FoveationStrategy,
    WorldModelConfig,
    WorldModelFoveationConfig,
    WorldModelFoveationSubstrate,
    WorldModelModule,
    WorldModelRecurrenceMode,
    Z5RoutedWorldModel,
)
from tac.substrates.c1_world_model_foveation.architecture import (
    FoveatedDecoderModule,
    FoveationMapModule,
)
from tac.substrates.z5_predictive_coding_world_model.architecture import (
    HierarchicalPredictor,
)


REPO_ROOT = Path(__file__).resolve().parents[5]


def _load_autopilot_module():
    """Load the cathedral_autopilot_autonomous_loop module from `tools/`.

    Mirrors the loader used by sister test files so we can introspect the
    module-level constants and helper functions for the Decision 6 reward
    arithmetic.
    """
    autopilot_path = REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py"
    spec = importlib.util.spec_from_file_location(
        "cathedral_autopilot_autonomous_loop_for_c1_z5_test", str(autopilot_path)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_tiny_native_cfg() -> WorldModelFoveationConfig:
    """Tiny C1 config with the NATIVE WorldModelModule kernel."""
    wm_cfg = WorldModelConfig(
        recurrence_mode=WorldModelRecurrenceMode.GRU,
        latent_dim=8,
        hidden_dim=8,
        route_compute_to_z5=False,
    )
    return WorldModelFoveationConfig(
        world_model_cfg=wm_cfg,
        foveation_strategy=FoveationStrategy.EGO_MOTION_RADIAL,
        output_height=24,
        output_width=32,
        num_pairs=4,
        decoder_channels=(8, 4, 4),
    )


def _make_tiny_z5_routed_cfg(
    *, identity_predictor_mode: bool = False
) -> WorldModelFoveationConfig:
    """Tiny C1 config with the Z5-ROUTED kernel (Decision 1 β)."""
    wm_cfg = WorldModelConfig(
        recurrence_mode=WorldModelRecurrenceMode.GRU,
        latent_dim=8,
        hidden_dim=8,
        route_compute_to_z5=True,
        z5_predictor_hidden_dim=16,
        z5_predictor_num_layers=2,
        z5_predictor_ego_motion_dim=8,
        z5_identity_predictor_mode=identity_predictor_mode,
    )
    return WorldModelFoveationConfig(
        world_model_cfg=wm_cfg,
        foveation_strategy=FoveationStrategy.EGO_MOTION_RADIAL,
        output_height=24,
        output_width=32,
        num_pairs=4,
        decoder_channels=(8, 4, 4),
    )


# =====================================================================
# 1. WorldModelModule preservation — Decision δ REJECTED 11/11 (DO NOT REMOVE)
# =====================================================================


class TestWorldModelModulePreserved:
    """Decision δ (DROP WorldModelModule) was REJECTED 11/11 by the council
    on 2026-05-14. The class definition AND composition path MUST stay in
    place per CLAUDE.md "Forbidden premature KILL without research
    exhaustion" + the council's explicit PAUSE verdict pending dispositive
    contest-scale evidence."""

    def test_world_model_module_class_still_defined(self) -> None:
        # Importable symbol; canonical reference for the council ledger.
        assert WorldModelModule is not None
        assert isinstance(WorldModelModule, type)

    def test_world_model_module_unroll_still_works(self) -> None:
        # The native kernel still produces (n_steps, latent_dim) output.
        wm_cfg = WorldModelConfig(latent_dim=8, hidden_dim=8)
        wm = WorldModelModule(wm_cfg)
        z_init = torch.zeros(8)
        out = wm.unroll(z_init, n_steps=5)
        assert out.shape == (5, 8)

    def test_substrate_default_uses_native_world_model(self) -> None:
        # Default (route_compute_to_z5=False) MUST stay native.
        cfg = _make_tiny_native_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        assert isinstance(substrate.world_model, WorldModelModule)
        assert not isinstance(substrate.world_model, Z5RoutedWorldModel)
        assert substrate.routes_compute_to_z5 is False

    def test_native_substrate_still_renders_pairs(self) -> None:
        cfg = _make_tiny_native_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        rgb_0, rgb_1 = substrate.render_pair(pair_idx=0)
        assert rgb_0.shape == (1, 3, 24, 32)
        assert rgb_1.shape == (1, 3, 24, 32)

    def test_native_substrate_still_renders_all_frames(self) -> None:
        cfg = _make_tiny_native_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        rgb, fov = substrate.render_all_frames()
        # T = 2 * num_pairs = 8
        assert rgb.shape == (8, 3, 24, 32)
        assert fov.shape == (8, 1, 24, 32)


# =====================================================================
# 2. Foveation behavior unchanged — independent finding STANDS
# =====================================================================


class TestFoveationBehaviorUnchanged:
    """Per the 2026-05-14 council §13 finding: ``ego_motion_radial`` 57%
    margin on real video is INDEPENDENT of the world-model review and
    STANDS. The foveation map module + strategy enum + per-pixel sparsity
    behavior MUST be untouched by the routing scaffold."""

    def test_foveation_map_module_still_imports(self) -> None:
        assert FoveationMapModule is not None

    def test_foveation_strategy_uniform_returns_ones(self) -> None:
        cfg = WorldModelFoveationConfig(
            world_model_cfg=WorldModelConfig(latent_dim=8, hidden_dim=8),
            foveation_strategy=FoveationStrategy.UNIFORM,
            output_height=24,
            output_width=32,
            num_pairs=4,
        )
        fov = FoveationMapModule(cfg)
        z_t = torch.randn(2, 8)
        m = fov.map(z_t)
        assert m.shape == (2, 1, 24, 32)
        assert torch.allclose(m, torch.ones_like(m))

    def test_foveation_strategy_ego_motion_radial_concentrates_center(self) -> None:
        cfg = WorldModelFoveationConfig(
            world_model_cfg=WorldModelConfig(latent_dim=8, hidden_dim=8),
            foveation_strategy=FoveationStrategy.EGO_MOTION_RADIAL,
            output_height=24,
            output_width=32,
            num_pairs=4,
        )
        fov = FoveationMapModule(cfg)
        z_t = torch.randn(1, 8)
        m = fov.map(z_t)
        # Center should be 1.0 (peak of Gaussian).
        center_val = float(m[0, 0, 12, 16].item())
        corner_val = float(m[0, 0, 0, 0].item())
        assert center_val > 0.9
        assert corner_val < 0.5  # heavily attenuated

    def test_foveation_strategy_routing_independent_of_z5(self) -> None:
        """Routing path uses Z5 kernel for world-model only; foveation
        map module is unaffected (independent finding STANDS)."""
        cfg = _make_tiny_z5_routed_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        # FoveationMapModule is the same class regardless of routing.
        assert isinstance(substrate.foveation, FoveationMapModule)
        # And the map output is the same shape regardless of routing.
        rgb, fov = substrate.render_all_frames()
        assert fov.shape == (8, 1, 24, 32)


# =====================================================================
# 3. Z5 routing primitive — Decision 1 β UNANIMOUS 11/11
# =====================================================================


class TestZ5RoutingPrimitive:
    """Decision 1 β (UNANIMOUS 11/11): route C1 compute INTO Z5 wholesale.
    The :class:`Z5RoutedWorldModel` wrapper must be a drop-in replacement
    for :class:`WorldModelModule.unroll` at the interface contract."""

    def test_z5_routed_world_model_constructs(self) -> None:
        wm_cfg = WorldModelConfig(latent_dim=8, hidden_dim=8, route_compute_to_z5=True)
        routed = Z5RoutedWorldModel(wm_cfg)
        assert isinstance(routed.predictor, HierarchicalPredictor)

    def test_z5_routed_unroll_signature_matches_native(self) -> None:
        wm_cfg = WorldModelConfig(latent_dim=8, hidden_dim=8, route_compute_to_z5=True)
        routed = Z5RoutedWorldModel(wm_cfg)
        z_init = torch.zeros(8)
        out = routed.unroll(z_init, n_steps=5)
        assert out.shape == (5, 8)

    def test_z5_routed_unroll_accepts_2d_z_init(self) -> None:
        wm_cfg = WorldModelConfig(latent_dim=8, hidden_dim=8, route_compute_to_z5=True)
        routed = Z5RoutedWorldModel(wm_cfg)
        z_init = torch.zeros(1, 8)
        out = routed.unroll(z_init, n_steps=3)
        assert out.shape == (3, 8)

    def test_z5_routed_unroll_rejects_wrong_latent_dim(self) -> None:
        wm_cfg = WorldModelConfig(latent_dim=8, hidden_dim=8, route_compute_to_z5=True)
        routed = Z5RoutedWorldModel(wm_cfg)
        bad_z = torch.zeros(99)
        with pytest.raises(ValueError, match="latent_dim"):
            routed.unroll(bad_z, n_steps=2)

    def test_substrate_with_routing_uses_z5_kernel(self) -> None:
        cfg = _make_tiny_z5_routed_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        assert isinstance(substrate.world_model, Z5RoutedWorldModel)
        assert not isinstance(substrate.world_model, WorldModelModule)
        assert substrate.routes_compute_to_z5 is True

    def test_routed_substrate_renders_pairs_with_correct_shape(self) -> None:
        cfg = _make_tiny_z5_routed_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        rgb_0, rgb_1 = substrate.render_pair(pair_idx=0)
        assert rgb_0.shape == (1, 3, 24, 32)
        assert rgb_1.shape == (1, 3, 24, 32)

    def test_routed_substrate_renders_all_frames_with_correct_shape(self) -> None:
        cfg = _make_tiny_z5_routed_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        rgb, fov = substrate.render_all_frames()
        assert rgb.shape == (8, 3, 24, 32)
        assert fov.shape == (8, 1, 24, 32)

    def test_routed_substrate_gradient_flows_into_z_init(self) -> None:
        cfg = _make_tiny_z5_routed_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        rgb, _ = substrate.render_all_frames()
        loss = rgb.pow(2).mean()
        loss.backward()
        assert substrate.z_init.grad is not None

    def test_decoder_module_unaffected_by_routing(self) -> None:
        # The decoder module is identical in both routing modes.
        native = WorldModelFoveationSubstrate(_make_tiny_native_cfg())
        routed = WorldModelFoveationSubstrate(_make_tiny_z5_routed_cfg())
        assert isinstance(native.decoder, FoveatedDecoderModule)
        assert isinstance(routed.decoder, FoveatedDecoderModule)


# =====================================================================
# 4. Z5 mode-arbitration consumed by routed C1
# =====================================================================


class TestZ5ModeArbitrationConsumed:
    """Per Decision 1 β: Z5's ``identity_predictor`` mode-arbitration IS
    the dispositive test primitive at trainer scale. The routed C1
    substrate must propagate the mode flag faithfully so the operator
    can dispatch BOTH ``predictive_world_model`` (False) AND
    ``identity_predictor`` (True) variants for the side-effect ablation
    per Time-Traveler staircase Step 3."""

    def test_predictive_world_model_mode_default(self) -> None:
        wm_cfg = WorldModelConfig(latent_dim=8, hidden_dim=8, route_compute_to_z5=True)
        routed = Z5RoutedWorldModel(wm_cfg)
        assert routed.identity_predictor_mode is False
        assert routed.predictor.identity_predictor is False

    def test_identity_predictor_mode_opt_in(self) -> None:
        wm_cfg = WorldModelConfig(
            latent_dim=8,
            hidden_dim=8,
            route_compute_to_z5=True,
            z5_identity_predictor_mode=True,
        )
        routed = Z5RoutedWorldModel(wm_cfg)
        assert routed.identity_predictor_mode is True
        assert routed.predictor.identity_predictor is True

    def test_identity_predictor_unroll_returns_constant_latent(self) -> None:
        """In identity_predictor mode, every step is z_t = z_{t-1} so the
        unroll returns the SAME z_init replicated n_steps times. This is
        the dispositive ablation primitive."""
        wm_cfg = WorldModelConfig(
            latent_dim=8,
            hidden_dim=8,
            route_compute_to_z5=True,
            z5_identity_predictor_mode=True,
        )
        routed = Z5RoutedWorldModel(wm_cfg)
        z_init = torch.randn(8)
        out = routed.unroll(z_init, n_steps=4)
        # Every row should equal z_init (identity_predictor branch).
        for t in range(out.shape[0]):
            assert torch.allclose(out[t], z_init, atol=1e-6)

    def test_predictive_world_model_unroll_diverges_from_identity(self) -> None:
        """The predictive-world-model regime should produce DIFFERENT
        latents per step (the network applies a real prediction); compare
        against the identity_predictor=True regime which is constant."""
        z_init = torch.randn(8)
        torch.manual_seed(0)
        wm_cfg_predictive = WorldModelConfig(
            latent_dim=8,
            hidden_dim=8,
            route_compute_to_z5=True,
            z5_identity_predictor_mode=False,
        )
        routed_predictive = Z5RoutedWorldModel(wm_cfg_predictive)
        out_predictive = routed_predictive.unroll(z_init, n_steps=4)
        # At least one row should differ from z_init under the predictive
        # mode (predictor net applies a non-trivial transform).
        any_diff = any(
            not torch.allclose(out_predictive[t], z_init, atol=1e-6)
            for t in range(out_predictive.shape[0])
        )
        assert any_diff, (
            "predictive_world_model regime produced identity output "
            "(unexpected; the predictor net should apply a non-trivial transform)"
        )

    def test_routed_substrate_mode_propagates_to_inner_predictor(self) -> None:
        cfg = _make_tiny_z5_routed_cfg(identity_predictor_mode=True)
        substrate = WorldModelFoveationSubstrate(cfg)
        assert isinstance(substrate.world_model, Z5RoutedWorldModel)
        assert substrate.world_model.identity_predictor_mode is True
        assert substrate.world_model.predictor.identity_predictor is True


# =====================================================================
# 5. Autopilot reward arithmetic — Decision 6 HALF-MEASURE
# =====================================================================


class TestAutopilotDecision6Halve:
    """Decision 6 (HALF-MEASURE 4/11 explicit; conditional revision):
    halve the literature-anchor reward 0.01 -> 0.005 for C1-class tokens
    (Hafner / DreamerV3 / Ha-Schmidhuber). Other class-shift tokens keep
    the FULL 0.01 reward. The lane_class reward (0.02) is unaffected.

    Combined effect:
      C1-class candidate: lane_class 0.02 + literature 0.005 = 0.025
                          (was 0.03 stacked; halved by 0.005)
      Sister substrate (e.g. cooperative-receiver): lane_class 0.02 +
                          literature 0.01 = 0.03 (unchanged)

    Per the council ledger the halve is conditional and reversible:
    update to full reward IF Z5 (Decision 1 β) returns dispositive
    positive; revert to zero IF Z5 + Decision 1 α both negative.
    """

    def test_c1_halved_literature_tokens_constant_exists(self) -> None:
        mod = _load_autopilot_module()
        assert hasattr(mod, "_C1_HALVED_LITERATURE_TOKENS")
        assert "Hafner" in mod._C1_HALVED_LITERATURE_TOKENS
        assert "DreamerV3" in mod._C1_HALVED_LITERATURE_TOKENS
        assert "Ha-Schmidhuber" in mod._C1_HALVED_LITERATURE_TOKENS

    def test_c1_halve_factor_is_50_percent(self) -> None:
        mod = _load_autopilot_module()
        assert hasattr(mod, "_C1_LITERATURE_ANCHOR_HALVE_FACTOR")
        assert mod._C1_LITERATURE_ANCHOR_HALVE_FACTOR == 0.5

    def test_c1_tokens_still_in_class_shift_literature_tokens_RETAIN(self) -> None:
        """Per Decision δ REJECTION + Decision 6 RETAIN clause: the C1
        tokens MUST still be in the broader class-shift token set so the
        autopilot ranker continues to prioritize the lane (no closure)."""
        mod = _load_autopilot_module()
        for tok in ("Hafner", "DreamerV3", "Ha-Schmidhuber"):
            assert tok in mod._CLASS_SHIFT_LITERATURE_TOKENS

    def test_hafner_literature_anchor_receives_halved_reward(self) -> None:
        mod = _load_autopilot_module()
        adjusted = mod.adjust_predicted_delta_for_class_shift(
            base_delta=-0.03,
            literature_anchor="Hafner DreamerV3 2023",
        )
        # Halved: 0.005 subtracted (not 0.01).
        assert adjusted == pytest.approx(-0.035)

    def test_dreamerv3_literature_anchor_receives_halved_reward(self) -> None:
        mod = _load_autopilot_module()
        adjusted = mod.adjust_predicted_delta_for_class_shift(
            base_delta=-0.03,
            literature_anchor="DreamerV3 mastering diverse domains",
        )
        assert adjusted == pytest.approx(-0.035)

    def test_ha_schmidhuber_literature_anchor_receives_halved_reward(self) -> None:
        mod = _load_autopilot_module()
        adjusted = mod.adjust_predicted_delta_for_class_shift(
            base_delta=-0.03,
            literature_anchor="Ha-Schmidhuber 2018 World Models",
        )
        assert adjusted == pytest.approx(-0.035)

    def test_cooperative_receiver_anchor_receives_FULL_reward_unchanged(
        self,
    ) -> None:
        """Sister substrate (Atick-Redlich cooperative-receiver) must NOT
        be halved. Decision 6 only applies to C1-class tokens."""
        mod = _load_autopilot_module()
        adjusted = mod.adjust_predicted_delta_for_class_shift(
            base_delta=-0.03,
            literature_anchor="Atick-Redlich cooperative-receiver",
        )
        # Full 0.01 reward subtracted.
        assert adjusted == pytest.approx(-0.04)

    def test_predictive_coding_anchor_receives_FULL_reward_unchanged(
        self,
    ) -> None:
        mod = _load_autopilot_module()
        adjusted = mod.adjust_predicted_delta_for_class_shift(
            base_delta=-0.03,
            literature_anchor="Rao-Ballard predictive-coding",
        )
        assert adjusted == pytest.approx(-0.04)

    def test_mdl_ibps_anchor_receives_FULL_reward_unchanged(self) -> None:
        """C6 IBPS1 (across-class predictive-coding sister) keeps full reward."""
        mod = _load_autopilot_module()
        adjusted = mod.adjust_predicted_delta_for_class_shift(
            base_delta=-0.03,
            literature_anchor="MDL-IBPS Tishby-Zaslavsky",
        )
        assert adjusted == pytest.approx(-0.04)

    def test_lane_class_reward_unaffected_by_halve(self) -> None:
        """Decision 6 only halves the literature-anchor 0.01; the
        lane_class 0.02 reward stays at 0.02 for all class-shift lanes."""
        mod = _load_autopilot_module()
        adjusted = mod.adjust_predicted_delta_for_class_shift(
            base_delta=-0.03,
            lane_class="substrate_class_shift",
            literature_anchor="",  # no literature anchor
        )
        # Only lane_class 0.02 subtracted.
        assert adjusted == pytest.approx(-0.05)

    def test_c1_combined_lane_class_PLUS_halved_literature(self) -> None:
        """C1 candidate with BOTH ``substrate_class_shift`` lane_class AND
        ``Hafner`` literature anchor: expected reward is 0.02 (lane_class
        full) + 0.005 (literature halved) = 0.025 stacked. Council ledger
        Decision 6 specifies this is the canonical post-halve C1 reward."""
        mod = _load_autopilot_module()
        adjusted = mod.adjust_predicted_delta_for_class_shift(
            base_delta=-0.03,
            lane_class="substrate_class_shift",
            literature_anchor="Hafner DreamerV3 2023",
        )
        # 0.02 + 0.005 = 0.025 reward stacked.
        assert adjusted == pytest.approx(-0.055)

    def test_sister_substrate_combined_full_reward(self) -> None:
        """Sister substrate (e.g. cooperative-receiver class-shift lane)
        keeps the full 0.03 stacked reward (unchanged from pre-Decision-6)."""
        mod = _load_autopilot_module()
        adjusted = mod.adjust_predicted_delta_for_class_shift(
            base_delta=-0.03,
            lane_class="cooperative_receiver",
            literature_anchor="Atick-Redlich",
        )
        # 0.02 + 0.01 = 0.03 reward stacked.
        assert adjusted == pytest.approx(-0.06)

    def test_no_anchor_no_lane_class_no_change(self) -> None:
        """Candidate with neither lane_class nor literature anchor gets
        zero bonus; the halve has no effect."""
        mod = _load_autopilot_module()
        adjusted = mod.adjust_predicted_delta_for_class_shift(
            base_delta=-0.03,
            lane_class=None,
            literature_anchor="",
        )
        assert adjusted == pytest.approx(-0.03)


# =====================================================================
# 6. Apples-to-apples: routed substrate composition contract
# =====================================================================


class TestRoutedSubstrateCompositionContract:
    """Per CLAUDE.md "Apples-to-apples evidence discipline" the routed
    substrate must produce the SAME interface contract (output shape +
    dtype + range) as the native substrate so downstream consumers
    (score-aware loss, archive packer, inflate runtime) are unchanged."""

    def test_native_vs_routed_render_pair_same_shape(self) -> None:
        torch.manual_seed(42)
        native = WorldModelFoveationSubstrate(_make_tiny_native_cfg())
        torch.manual_seed(42)
        routed = WorldModelFoveationSubstrate(_make_tiny_z5_routed_cfg())
        n0, n1 = native.render_pair(pair_idx=0)
        r0, r1 = routed.render_pair(pair_idx=0)
        assert n0.shape == r0.shape
        assert n1.shape == r1.shape
        assert n0.dtype == r0.dtype
        # Both produce sigmoid-bounded RGB.
        assert 0.0 <= float(n0.min().item()) and float(n0.max().item()) <= 1.0
        assert 0.0 <= float(r0.min().item()) and float(r0.max().item()) <= 1.0

    def test_native_vs_routed_render_all_frames_same_shape(self) -> None:
        native = WorldModelFoveationSubstrate(_make_tiny_native_cfg())
        routed = WorldModelFoveationSubstrate(_make_tiny_z5_routed_cfg())
        n_rgb, n_fov = native.render_all_frames()
        r_rgb, r_fov = routed.render_all_frames()
        assert n_rgb.shape == r_rgb.shape
        assert n_fov.shape == r_fov.shape

    def test_routed_substrate_param_count_increases_with_z5_predictor(
        self,
    ) -> None:
        """Sanity: the routed substrate has MORE params than the native
        substrate at default z5_predictor_hidden_dim because Z5's
        HierarchicalPredictor adds the (z_to_hidden + ego_to_hidden +
        fused + hidden_to_z) ladder. Empirically Z5's 2-layer 16-hidden
        predictor at latent_dim=8 ego_dim=8 has ~640 more params than
        the GRUCell at hidden_dim=8."""
        native = WorldModelFoveationSubstrate(_make_tiny_native_cfg())
        routed = WorldModelFoveationSubstrate(_make_tiny_z5_routed_cfg())
        n_params = sum(p.numel() for p in native.parameters())
        r_params = sum(p.numel() for p in routed.parameters())
        # Routed has Z5's HierarchicalPredictor; should have at least one
        # more parameter than the native substrate.
        assert r_params > n_params


__all__ = [
    "TestAutopilotDecision6Halve",
    "TestFoveationBehaviorUnchanged",
    "TestRoutedSubstrateCompositionContract",
    "TestWorldModelModulePreserved",
    "TestZ5ModeArbitrationConsumed",
    "TestZ5RoutingPrimitive",
]
