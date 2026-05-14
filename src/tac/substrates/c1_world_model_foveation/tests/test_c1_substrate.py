"""C1 world-model + foveation substrate scaffold tests.

Verifies structure-parity, world-model unroll determinism, foveation map
differentiability, archive roundtrip, byte-identity, and NotImplementedError
on the full main pending Phase 3 council approval.

Test categories:

1. Architecture (module construction, forward shapes, gradient flow)
2. World-model unroll (GRU/LSTM/Transformer modes; determinism)
3. Foveation map (UNIFORM/EGO_MOTION_RADIAL/LEARNED_PER_PIXEL strategies)
4. Archive roundtrip (pack/parse byte-identity; deterministic compression)
5. Score-aware loss (rate-term + scorer-term + residual + foveation; eval_roundtrip enforcement)
6. Phase 3 gate (_full_main raises NotImplementedError before council approval)
"""

from __future__ import annotations

import json

import pytest
import torch

from tac.substrates.c1_world_model_foveation import (
    C1WMFV1_HEADER_SIZE,
    C1WMFV1_MAGIC,
    C1WMFV1_SCHEMA_VERSION,
    FoveationStrategy,
    WorldModelConfig,
    WorldModelFoveationConfig,
    WorldModelFoveationLossWeights,
    WorldModelFoveationSubstrate,
    WorldModelRecurrenceMode,
    pack_archive,
    parse_archive,
)
from tac.substrates.c1_world_model_foveation.architecture import (
    FoveatedDecoderModule,
    FoveationMapModule,
    WorldModelModule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_cfg(
    recurrence: WorldModelRecurrenceMode = WorldModelRecurrenceMode.GRU,
    fov: FoveationStrategy = FoveationStrategy.EGO_MOTION_RADIAL,
) -> WorldModelFoveationConfig:
    """Tiny config for fast tests: 4 pairs, 24x32, latent_dim=8."""
    wm_cfg = WorldModelConfig(
        recurrence_mode=recurrence,
        latent_dim=8,
        hidden_dim=8,
    )
    return WorldModelFoveationConfig(
        world_model_cfg=wm_cfg,
        foveation_strategy=fov,
        output_height=24,
        output_width=32,
        num_pairs=4,
        decoder_channels=(8, 4, 4),
    )


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestArchitecture:
    def test_substrate_constructs_with_default_cfg(self):
        cfg = _make_tiny_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        assert substrate.cfg.num_pairs == 4
        assert substrate.z_init.shape == (8,)

    def test_substrate_render_all_frames_shapes(self):
        cfg = _make_tiny_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        rgb, fov = substrate.render_all_frames()
        # T = 2 * num_pairs = 8, 3 channels, H=24, W=32
        assert rgb.shape == (8, 3, 24, 32)
        assert fov.shape == (8, 1, 24, 32)
        # RGB in [0, 1] after sigmoid
        assert rgb.min().item() >= 0.0
        assert rgb.max().item() <= 1.0
        # Foveation in [0, 1]
        assert fov.min().item() >= 0.0
        assert fov.max().item() <= 1.0

    def test_substrate_render_pair_shapes(self):
        cfg = _make_tiny_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        rgb_0, rgb_1 = substrate.render_pair(pair_idx=0)
        assert rgb_0.shape == (1, 3, 24, 32)
        assert rgb_1.shape == (1, 3, 24, 32)

    def test_substrate_render_pair_out_of_range_raises(self):
        cfg = _make_tiny_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        with pytest.raises(ValueError, match="pair_idx"):
            substrate.render_pair(pair_idx=99)
        with pytest.raises(ValueError, match="pair_idx"):
            substrate.render_pair(pair_idx=-1)

    def test_substrate_gradient_flows_into_z_init(self):
        cfg = _make_tiny_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        rgb, _ = substrate.render_all_frames()
        loss = rgb.pow(2).mean()
        loss.backward()
        assert substrate.z_init.grad is not None
        assert substrate.z_init.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# World-model recurrence mode tests
# ---------------------------------------------------------------------------

class TestWorldModelRecurrence:
    def test_gru_mode_unroll_deterministic(self):
        cfg = _make_tiny_cfg(recurrence=WorldModelRecurrenceMode.GRU)
        wm = WorldModelModule(cfg.world_model_cfg)
        z_init = torch.zeros(cfg.world_model_cfg.latent_dim)
        out_a = wm.unroll(z_init, n_steps=8)
        out_b = wm.unroll(z_init, n_steps=8)
        assert torch.allclose(out_a, out_b)
        assert out_a.shape == (8, 8)

    def test_lstm_mode_unroll_shapes(self):
        cfg = _make_tiny_cfg(recurrence=WorldModelRecurrenceMode.LSTM)
        wm = WorldModelModule(cfg.world_model_cfg)
        z_init = torch.randn(cfg.world_model_cfg.latent_dim)
        out = wm.unroll(z_init, n_steps=8)
        assert out.shape == (8, 8)

    def test_transformer_mode_unroll_shapes(self):
        cfg = _make_tiny_cfg(recurrence=WorldModelRecurrenceMode.TRANSFORMER)
        wm = WorldModelModule(cfg.world_model_cfg)
        z_init = torch.randn(cfg.world_model_cfg.latent_dim)
        out = wm.unroll(z_init, n_steps=8)
        assert out.shape == (8, 8)

    def test_unroll_rejects_wrong_latent_dim(self):
        cfg = _make_tiny_cfg()
        wm = WorldModelModule(cfg.world_model_cfg)
        bad_z = torch.zeros(99)
        with pytest.raises(ValueError, match="latent_dim"):
            wm.unroll(bad_z, n_steps=4)


# ---------------------------------------------------------------------------
# Foveation strategy tests
# ---------------------------------------------------------------------------

class TestFoveationStrategy:
    def test_uniform_returns_all_ones(self):
        cfg = _make_tiny_cfg(fov=FoveationStrategy.UNIFORM)
        fov = FoveationMapModule(cfg)
        z_t = torch.zeros(2, cfg.world_model_cfg.latent_dim)
        m = fov.map(z_t)
        assert m.shape == (2, 1, 24, 32)
        assert torch.allclose(m, torch.ones_like(m))

    def test_ego_motion_radial_peaks_at_center(self):
        cfg = _make_tiny_cfg(fov=FoveationStrategy.EGO_MOTION_RADIAL)
        fov = FoveationMapModule(cfg)
        z_t = torch.zeros(1, cfg.world_model_cfg.latent_dim)
        m = fov.map(z_t)
        # Center pixel (12, 16) should have higher value than corner (0, 0).
        assert m[0, 0, 12, 16].item() > m[0, 0, 0, 0].item()
        # Map is in [0, 1].
        assert m.min().item() >= 0.0
        assert m.max().item() <= 1.0 + 1e-5

    def test_learned_per_pixel_has_params(self):
        cfg = _make_tiny_cfg(fov=FoveationStrategy.LEARNED_PER_PIXEL)
        fov = FoveationMapModule(cfg)
        assert fov.learned_head is not None
        z_t = torch.randn(1, cfg.world_model_cfg.latent_dim)
        m = fov.map(z_t)
        assert m.shape == (1, 1, 24, 32)


# ---------------------------------------------------------------------------
# Archive pack/parse roundtrip tests
# ---------------------------------------------------------------------------

class TestArchiveRoundtrip:
    def _build_minimal_archive_inputs(self):
        cfg = _make_tiny_cfg()
        substrate = WorldModelFoveationSubstrate(cfg)
        wm_sd = substrate.world_model.state_dict()
        dec_sd = substrate.decoder.state_dict()
        return cfg, substrate, wm_sd, dec_sd

    def test_pack_then_parse_preserves_header(self):
        cfg, substrate, wm_sd, dec_sd = self._build_minimal_archive_inputs()
        archive_bytes = pack_archive(
            num_pairs=cfg.num_pairs,
            recurrence_mode=0,
            foveation_strategy=1,
            latent_dim=cfg.world_model_cfg.latent_dim,
            output_h=cfg.output_height,
            output_w=cfg.output_width,
            world_model_state_dict=wm_sd,
            decoder_state_dict=dec_sd,
            z_init=substrate.z_init.detach(),
            foveation_meta={"sigma": 6.0, "center_y": 12.0, "center_x": 16.0},
            residual_blob=b"",  # smoke -- no residual
            meta={"smoke": True, "lane_id": "lane_c1_test"},  # FAKE_LANE_OK:test-fixture-meta-not-real-lane
        )
        arc = parse_archive(archive_bytes)
        assert arc.version == C1WMFV1_SCHEMA_VERSION
        assert arc.num_pairs == 4
        assert arc.recurrence_mode == 0
        assert arc.foveation_strategy == 1
        assert arc.latent_dim == 8
        assert arc.output_h == 24
        assert arc.output_w == 32
        assert arc.meta.get("smoke") is True
        assert arc.meta.get("lane_id") == "lane_c1_test"  # FAKE_LANE_OK:test-fixture-meta-not-real-lane

    def test_pack_is_deterministic_byte_identical(self):
        cfg, substrate, wm_sd, dec_sd = self._build_minimal_archive_inputs()
        meta = {"smoke": True, "lane_id": "lane_c1_test"}  # FAKE_LANE_OK:test-fixture-meta-not-real-lane
        a1 = pack_archive(
            num_pairs=cfg.num_pairs, recurrence_mode=0, foveation_strategy=1,
            latent_dim=8, output_h=24, output_w=32,
            world_model_state_dict=wm_sd, decoder_state_dict=dec_sd,
            z_init=substrate.z_init.detach(),
            foveation_meta={"sigma": 6.0},
            residual_blob=b"", meta=meta,
        )
        a2 = pack_archive(
            num_pairs=cfg.num_pairs, recurrence_mode=0, foveation_strategy=1,
            latent_dim=8, output_h=24, output_w=32,
            world_model_state_dict=wm_sd, decoder_state_dict=dec_sd,
            z_init=substrate.z_init.detach(),
            foveation_meta={"sigma": 6.0},
            residual_blob=b"", meta=meta,
        )
        assert a1 == a2, "pack_archive must be deterministic byte-identical"

    def test_parse_rejects_bad_magic(self):
        bad = b"BAD!" + b"\x00" * (C1WMFV1_HEADER_SIZE - 4)
        with pytest.raises(ValueError, match="magic mismatch"):
            parse_archive(bad)

    def test_parse_rejects_truncated_header(self):
        with pytest.raises(ValueError, match="too short"):
            parse_archive(b"\x00" * 4)

    def test_pack_rejects_invalid_recurrence_mode(self):
        cfg, substrate, wm_sd, dec_sd = self._build_minimal_archive_inputs()
        with pytest.raises(ValueError, match="recurrence_mode"):
            pack_archive(
                num_pairs=4, recurrence_mode=99,  # invalid
                foveation_strategy=0,
                latent_dim=8, output_h=24, output_w=32,
                world_model_state_dict=wm_sd, decoder_state_dict=dec_sd,
                z_init=substrate.z_init.detach(),
                foveation_meta={},
                residual_blob=b"",
                meta={},
            )

    def test_pack_rejects_invalid_foveation_strategy(self):
        cfg, substrate, wm_sd, dec_sd = self._build_minimal_archive_inputs()
        with pytest.raises(ValueError, match="foveation_strategy"):
            pack_archive(
                num_pairs=4, recurrence_mode=0,
                foveation_strategy=99,  # invalid
                latent_dim=8, output_h=24, output_w=32,
                world_model_state_dict=wm_sd, decoder_state_dict=dec_sd,
                z_init=substrate.z_init.detach(),
                foveation_meta={},
                residual_blob=b"",
                meta={},
            )

    def test_pack_byte_count_within_targets(self):
        """Packed archive bytes should be in a reasonable range for the tiny cfg."""
        cfg, substrate, wm_sd, dec_sd = self._build_minimal_archive_inputs()
        archive_bytes = pack_archive(
            num_pairs=cfg.num_pairs, recurrence_mode=0, foveation_strategy=1,
            latent_dim=8, output_h=24, output_w=32,
            world_model_state_dict=wm_sd, decoder_state_dict=dec_sd,
            z_init=substrate.z_init.detach(),
            foveation_meta={"sigma": 6.0},
            residual_blob=b"",
            meta={"smoke": True},
        )
        # Tiny config: 8-dim latent_dim + linear-to-8x8 decoder. Expect
        # in the 10-25 KB range (the linear head dominates byte count at
        # this scale; production cfg with FP4 + smaller linear head will
        # be much smaller).
        assert len(archive_bytes) > C1WMFV1_HEADER_SIZE
        assert len(archive_bytes) < 30_000

    def test_meta_sorted_keys_deterministic(self):
        """Meta JSON should be sorted-keys (CLAUDE.md L8 determinism)."""
        cfg, substrate, wm_sd, dec_sd = self._build_minimal_archive_inputs()
        # Two metas with same keys in different insertion order.
        meta_a = {"z": 1, "a": 2, "m": 3}
        meta_b = {"a": 2, "m": 3, "z": 1}
        a1 = pack_archive(
            num_pairs=4, recurrence_mode=0, foveation_strategy=1,
            latent_dim=8, output_h=24, output_w=32,
            world_model_state_dict=wm_sd, decoder_state_dict=dec_sd,
            z_init=substrate.z_init.detach(),
            foveation_meta={"sigma": 6.0},
            residual_blob=b"",
            meta=meta_a,
        )
        a2 = pack_archive(
            num_pairs=4, recurrence_mode=0, foveation_strategy=1,
            latent_dim=8, output_h=24, output_w=32,
            world_model_state_dict=wm_sd, decoder_state_dict=dec_sd,
            z_init=substrate.z_init.detach(),
            foveation_meta={"sigma": 6.0},
            residual_blob=b"",
            meta=meta_b,
        )
        assert a1 == a2


# ---------------------------------------------------------------------------
# Score-aware loss tests
# ---------------------------------------------------------------------------

class TestScoreAwareLoss:
    def test_loss_weights_dataclass_defaults(self):
        w = WorldModelFoveationLossWeights()
        assert w.alpha_rate == 25.0
        assert w.beta_seg == 100.0
        assert w.contest_normalizer == 37_545_489.0
        assert w.lambda_residual == 0.1
        assert w.lambda_foveation == 0.01

    def test_loss_forward_rejects_eval_roundtrip_false(self):
        """eval_roundtrip=False MUST raise per CLAUDE.md non-negotiable."""
        from tac.substrates.c1_world_model_foveation.score_aware_loss import (
            WorldModelFoveationScoreAwareLoss,
        )

        # Build a dummy scorer with required preprocess_input method.
        class _DummyScorer(torch.nn.Module):
            def preprocess_input(self, x):
                return x
            def forward(self, x):
                return torch.zeros(1, 5, 384, 512)

        loss_fn = WorldModelFoveationScoreAwareLoss(
            seg_scorer=_DummyScorer(),
            pose_scorer=_DummyScorer(),
            weights=WorldModelFoveationLossWeights(),
        )
        rgb_0 = torch.full((1, 3, 24, 32), 128.0)
        rgb_1 = torch.full((1, 3, 24, 32), 128.0)
        # noinspection PyTypeChecker
        with pytest.raises(ValueError, match="apply_eval_roundtrip"):
            loss_fn.forward(
                reconstructed_rgb_0=rgb_0,
                reconstructed_rgb_1=rgb_1,
                gt_rgb_0=rgb_0,
                gt_rgb_1=rgb_1,
                archive_bytes_proxy=torch.tensor(1000.0),
                residual=torch.zeros(1, 3, 6, 8),
                foveation_map=torch.ones(1, 1, 24, 32),
                apply_eval_roundtrip=False,
            )


# ---------------------------------------------------------------------------
# Phase 3 council-approval gate
# ---------------------------------------------------------------------------

class TestPhase3CouncilGate:
    def test_full_main_raises_not_implemented(self):
        """The _full_main entry point MUST refuse to run before Phase 3 council.

        Per CLAUDE.md "Design decisions -- non-negotiable" and "Long-burn
        score-lowering campaign default" the multi-stage training schedule
        ($30-50 over 3-4 weeks) requires council approval before any
        dispatch. Smoke is the only operator-authorized path at L1.
        """
        import experiments.train_substrate_c1_world_model_foveation as trainer_mod
        import argparse

        parser = trainer_mod._build_parser()
        args = parser.parse_args([
            "--output-dir", "/tmp/c1_test",  # noqa: S108 -- test fixture only
        ])
        args.smoke = False
        with pytest.raises(
            NotImplementedError, match="Phase 3 council approval"
        ):
            trainer_mod._full_main(args)
