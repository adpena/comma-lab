# SPDX-License-Identifier: MIT
"""Tests for Z7-Mamba-2 FULL LANDING substrate package.

Per the Z7-Mamba-2 full landing 2026-05-18 + lane
`lane_z7_as_mamba_2_full_landing_20260518`. Sister to the scaffold tests
at `test_z7_mamba2_scaffold.py` (which validate the Mamba2Predictor
canonical helper) — these tests validate the SUBSTRATE PACKAGE built on
top: architecture (Z7Mamba2PredictiveCodingSubstrate), archive
(Z7MCM2 grammar), inflate runtime, score-aware loss.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture
def tiny_config():
    """Tiny config for fast unit tests."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingConfig,
    )
    return Z7Mamba2PredictiveCodingConfig(
        num_pairs=4,
        latent_dim=8,
        ego_motion_dim=4,
        d_model=16,
        d_state=8,
        expand=2,
        d_conv=4,
        decoder_embed_dim=8,
        decoder_initial_grid_h=4,
        decoder_initial_grid_w=4,
        decoder_channels=(8, 8, 4, 4),
        decoder_num_upsample_blocks=2,
        output_height=16,
        output_width=16,
    )


def test_substrate_package_imports_canonical_public_api():
    """All canonical public API names must be importable from package root."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingConfig,
        Z7Mamba2PredictiveCodingSubstrate,
        Z7Mamba2PredictiveCodingArchive,
        Z7Mamba2PredictiveCodingLossWeights,
        Z7Mamba2PredictiveCodingScoreAwareLoss,
        pack_archive,
        parse_archive,
        parse_z7mcm2_archive_bytes,
        replay_latent_sequence,
        replay_latent_sequence_with_context,
        Z7MCM2_MAGIC,
        Z7MCM2_SCHEMA_VERSION,
        Z7MCM2_SECTION_ROLES,
        MAMBA_SSM_AVAILABLE,
    )
    assert Z7MCM2_MAGIC == b"Z7M2"
    assert Z7MCM2_SCHEMA_VERSION == 1


def test_substrate_instantiates_and_reports_param_breakdown(tiny_config):
    """Substrate construction must yield non-empty param breakdown."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingSubstrate,
    )
    sub = Z7Mamba2PredictiveCodingSubstrate(tiny_config)
    breakdown = sub.num_parameters_breakdown()
    assert breakdown["total"] > 0
    assert breakdown["decoder"] > 0
    assert breakdown["predictor"] > 0
    assert breakdown["latent_init"] == tiny_config.latent_dim
    assert breakdown["residuals"] == tiny_config.num_pairs * tiny_config.latent_dim
    assert breakdown["context_conditioner"] == 0  # no context conditioning


def test_substrate_reconstruct_all_pairs_produces_correct_shapes(tiny_config):
    """reconstruct_all_pairs must yield (rgb_0, rgb_1, latents) with canonical shapes."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingSubstrate,
    )
    sub = Z7Mamba2PredictiveCodingSubstrate(tiny_config)
    sub.eval()
    rgb_0, rgb_1, latents = sub.reconstruct_all_pairs()
    assert rgb_0.shape == (
        tiny_config.num_pairs,
        3,
        tiny_config.output_height,
        tiny_config.output_width,
    )
    assert rgb_1.shape == rgb_0.shape
    assert latents.shape == (tiny_config.num_pairs, tiny_config.latent_dim)


def test_substrate_reconstruct_pair_supports_mini_batch_per_catalog_218(tiny_config):
    """Per Catalog #218 mini-batch reconstruct defense against D4 T4 OOM bug class."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingSubstrate,
    )
    sub = Z7Mamba2PredictiveCodingSubstrate(tiny_config)
    sub.eval()
    pair_indices = torch.tensor([0, 2], dtype=torch.long)
    rgb_0, rgb_1, latents = sub.reconstruct_pair(pair_indices)
    assert rgb_0.shape[0] == 2
    assert latents.shape == (2, tiny_config.latent_dim)


def test_substrate_decoder_metadata_carries_canonical_fields(tiny_config):
    """decoder_metadata must carry all fields required by Z7MCM2 inflate runtime."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingSubstrate,
    )
    sub = Z7Mamba2PredictiveCodingSubstrate(tiny_config)
    meta = sub.decoder_metadata()
    required_fields = {
        "decoder_embed_dim",
        "decoder_initial_grid_h",
        "decoder_initial_grid_w",
        "decoder_channels",
        "decoder_num_upsample_blocks",
        "output_height",
        "output_width",
        "mamba2_d_model",
        "mamba2_d_state",
        "mamba2_expand",
        "mamba2_d_conv",
        "mamba2_backend_active",
    }
    missing = required_fields - set(meta.keys())
    assert not missing, f"Missing canonical decoder_metadata fields: {missing}"


def test_z7mcm2_pack_parse_roundtrip_preserves_config_and_tensors(tiny_config):
    """Z7MCM2 pack -> parse roundtrip must preserve config + tensor shapes."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingSubstrate,
        pack_archive,
        parse_archive,
    )
    sub = Z7Mamba2PredictiveCodingSubstrate(tiny_config)
    sub.eval()
    meta = {**sub.decoder_metadata(), "loss_mode": "proxy"}
    blob = pack_archive(
        {},
        sub.decoder.state_dict(),
        sub.predictor.state_dict(),
        sub.latent_init.detach().cpu(),
        sub.residuals.detach().cpu(),
        sub.ego_motion_buffer.detach().cpu(),
        meta,
        config=tiny_config,
    )
    archive = parse_archive(blob)
    assert archive.config.latent_dim == tiny_config.latent_dim
    assert archive.config.num_pairs == tiny_config.num_pairs
    assert archive.config.d_model == tiny_config.d_model
    assert archive.config.d_state == tiny_config.d_state
    # Inflate-time backend MUST be reference_torch per HNeRV parity L4 + L9
    assert archive.config.backend == "reference_torch"
    assert archive.latent_init.shape == (tiny_config.latent_dim,)
    assert archive.residuals.shape == (tiny_config.num_pairs, tiny_config.latent_dim)


def test_z7mcm2_archive_carries_non_promotable_authority_flags(tiny_config):
    """Z7MCM2 meta MUST stamp non-promotable authority blockers per CLAUDE.md."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingSubstrate,
        pack_archive,
        parse_archive,
    )
    sub = Z7Mamba2PredictiveCodingSubstrate(tiny_config)
    sub.eval()
    meta = {**sub.decoder_metadata(), "loss_mode": "proxy"}
    blob = pack_archive(
        {},
        sub.decoder.state_dict(),
        sub.predictor.state_dict(),
        sub.latent_init.detach().cpu(),
        sub.residuals.detach().cpu(),
        sub.ego_motion_buffer.detach().cpu(),
        meta,
        config=tiny_config,
    )
    archive = parse_archive(blob)
    authority_meta = archive.meta["z7_mamba2_recurrent_predictive_coding_meta"]
    assert authority_meta["score_claim"] is False
    assert authority_meta["promotion_eligible"] is False
    assert authority_meta["ready_for_exact_eval_dispatch"] is False
    assert authority_meta["ready_for_paid_dispatch"] is False
    blockers = authority_meta["blockers"]
    assert any("post_training_tier_c" in b for b in blockers), (
        "Catalog #324 post-training Tier-C blocker required"
    )
    assert any("wave_n_plus_1_council" in b for b in blockers), (
        "Wave-N+1 council blocker required per Z7 parent symposium"
    )


def test_z7mcm2_replay_reconstructs_latent_sequence(tiny_config):
    """replay_latent_sequence must reconstruct the autoregressive latent stream."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingSubstrate,
        pack_archive,
        parse_archive,
        replay_latent_sequence,
    )
    sub = Z7Mamba2PredictiveCodingSubstrate(tiny_config)
    sub.eval()
    meta = {**sub.decoder_metadata(), "loss_mode": "proxy"}
    blob = pack_archive(
        {},
        sub.decoder.state_dict(),
        sub.predictor.state_dict(),
        sub.latent_init.detach().cpu(),
        sub.residuals.detach().cpu(),
        sub.ego_motion_buffer.detach().cpu(),
        meta,
        config=tiny_config,
    )
    archive = parse_archive(blob)
    latents = replay_latent_sequence(archive)
    assert latents.shape == (tiny_config.num_pairs, tiny_config.latent_dim)


def test_inflate_runtime_closes_loop_byte_faithful_per_hnerv_parity_l9(tiny_config):
    """Per HNeRV parity L9 runtime closure: inflate.py must produce contest-shaped raw."""
    from tac.substrates._shared.inflate_runtime import CAMERA_HW
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingSubstrate,
        pack_archive,
    )
    from tac.substrates.time_traveler_l5_z7_mamba2.inflate import inflate_one_video

    sub = Z7Mamba2PredictiveCodingSubstrate(tiny_config)
    sub.eval()
    meta = {**sub.decoder_metadata(), "loss_mode": "proxy"}
    blob = pack_archive(
        {},
        sub.decoder.state_dict(),
        sub.predictor.state_dict(),
        sub.latent_init.detach().cpu(),
        sub.residuals.detach().cpu(),
        sub.ego_motion_buffer.detach().cpu(),
        meta,
        config=tiny_config,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "0.raw"
        frames = inflate_one_video(blob, out_path)
        # 2 frames per pair × num_pairs
        assert frames == tiny_config.num_pairs * 2
        # Output is contest-camera-resolution (writer upsamples per
        # canonical inflate_runtime.write_rgb_pair_to_raw)
        expected = (
            tiny_config.num_pairs * 2 * 3 * CAMERA_HW[0] * CAMERA_HW[1]
        )
        assert out_path.stat().st_size == expected


def test_substrate_distinguishing_feature_consumed_at_inflate(tiny_config):
    """Per Catalog #220 + #272 distinguishing-feature contract: byte mutations
    of the Mamba-2 predictor MUST change inflate output."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingConfig,
        Z7Mamba2PredictiveCodingSubstrate,
        pack_archive,
    )
    from tac.substrates.time_traveler_l5_z7_mamba2.inflate import inflate_one_video

    # Build the recurrent substrate
    sub = Z7Mamba2PredictiveCodingSubstrate(tiny_config)
    sub.eval()
    # Mutate the residuals slightly to ensure non-zero output difference
    with torch.no_grad():
        sub.residuals.add_(torch.randn_like(sub.residuals) * 0.1)
    meta = {**sub.decoder_metadata(), "loss_mode": "proxy"}
    blob = pack_archive(
        {},
        sub.decoder.state_dict(),
        sub.predictor.state_dict(),
        sub.latent_init.detach().cpu(),
        sub.residuals.detach().cpu(),
        sub.ego_motion_buffer.detach().cpu(),
        meta,
        config=tiny_config,
    )
    # Build identity-predictor control with same decoder + zeroed residuals
    control_cfg = Z7Mamba2PredictiveCodingConfig(
        **{**tiny_config.__dict__, "identity_predictor": True, "stateful": False},
    )
    control_sub = Z7Mamba2PredictiveCodingSubstrate(control_cfg)
    control_sub.load_state_dict(
        {k: v for k, v in sub.state_dict().items() if "predictor" not in k},
        strict=False,
    )
    control_sub.eval()
    control_meta = {**control_sub.decoder_metadata(), "loss_mode": "proxy"}
    control_blob = pack_archive(
        {},
        control_sub.decoder.state_dict(),
        control_sub.predictor.state_dict(),
        control_sub.latent_init.detach().cpu(),
        control_sub.residuals.detach().cpu(),
        control_sub.ego_motion_buffer.detach().cpu(),
        control_meta,
        config=control_cfg,
    )
    with tempfile.TemporaryDirectory() as tmp:
        rec_path = Path(tmp) / "rec.raw"
        ctrl_path = Path(tmp) / "ctrl.raw"
        inflate_one_video(blob, rec_path)
        inflate_one_video(control_blob, ctrl_path)
        rec_bytes = rec_path.read_bytes()
        ctrl_bytes = ctrl_path.read_bytes()
        # Recurrent vs identity_predictor MUST produce different bytes
        # (proves Mamba-2 distinguishing feature is consumed at inflate)
        assert rec_bytes != ctrl_bytes, (
            "Z7-Mamba-2 distinguishing-feature (Mamba-2 selective state-space "
            "predictor) must produce inflate output different from identity-predictor "
            "control per Catalog #220 + #272 byte-mutation contract"
        )


def test_score_aware_loss_construction_canonical_helper_routing():
    """Z7-Mamba-2 score-aware loss MUST route through canonical
    score_pair_components_dispatch per Catalog #164."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingLossWeights,
        Z7Mamba2PredictiveCodingScoreAwareLoss,
    )
    import inspect
    source = inspect.getsource(Z7Mamba2PredictiveCodingScoreAwareLoss.forward)
    assert "score_pair_components_dispatch" in source, (
        "Score-aware loss MUST route through canonical helper per Catalog #164"
    )
    assert "apply_eval_roundtrip" in source, (
        "Score-aware loss MUST enforce eval_roundtrip per CLAUDE.md non-negotiable"
    )


def test_score_aware_loss_train_keeps_frozen_scorers_eval():
    """Calling train() on the wrapper must not put contest scorers in train mode."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingLossWeights,
        Z7Mamba2PredictiveCodingScoreAwareLoss,
    )

    class MockScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=0.5)

        def forward(self, x):
            return self.dropout(x).mean()

    seg_scorer = MockScorer()
    pose_scorer = MockScorer()
    loss = Z7Mamba2PredictiveCodingScoreAwareLoss(
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        weights=Z7Mamba2PredictiveCodingLossWeights(),
    )
    loss.train()

    assert loss.training is True
    assert seg_scorer.training is False
    assert pose_scorer.training is False
    assert seg_scorer.dropout.training is False
    assert pose_scorer.dropout.training is False


def test_score_aware_loss_refuses_eval_roundtrip_false():
    """Per CLAUDE.md eval_roundtrip non-negotiable + Catalog #5:
    score-aware loss MUST refuse apply_eval_roundtrip=False."""
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingLossWeights,
        Z7Mamba2PredictiveCodingScoreAwareLoss,
    )

    # Build mock scorers
    class MockScorer(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3)) if x.dim() == 4 else x.mean()

    loss = Z7Mamba2PredictiveCodingScoreAwareLoss(
        seg_scorer=MockScorer(),
        pose_scorer=MockScorer(),
        weights=Z7Mamba2PredictiveCodingLossWeights(),
    )
    fake_rgb = torch.full((2, 3, 16, 16), 128.0)
    fake_archive = torch.tensor(1000.0)
    fake_residuals = torch.zeros(2, 8)
    fake_latents = torch.zeros(2, 8)
    with pytest.raises(ValueError, match="apply_eval_roundtrip=False is forbidden"):
        loss(
            reconstructed_rgb_0=fake_rgb,
            reconstructed_rgb_1=fake_rgb,
            gt_rgb_0=fake_rgb,
            gt_rgb_1=fake_rgb,
            archive_bytes_proxy=fake_archive,
            residuals=fake_residuals,
            latents=fake_latents,
            apply_eval_roundtrip=False,
        )


def test_identity_predictor_substrate_yields_predictor_zero_params(tiny_config):
    """identity_predictor=True must yield 0 predictor params per probe-disambiguator contract."""
    from dataclasses import replace
    from tac.substrates.time_traveler_l5_z7_mamba2 import (
        Z7Mamba2PredictiveCodingSubstrate,
    )
    identity_cfg = replace(tiny_config, identity_predictor=True)
    sub = Z7Mamba2PredictiveCodingSubstrate(identity_cfg)
    breakdown = sub.num_parameters_breakdown()
    assert breakdown["predictor"] == 0, (
        "identity_predictor substrate must have 0 trainable predictor params "
        "per Catalog #125 hook #6 probe-disambiguator contract"
    )


def test_lane_class_substrate_engineering_in_lane_registry():
    """Per HNeRV parity L7: substrate engineering tag must be declared in registry."""
    import json
    registry_path = Path(__file__).resolve().parents[3] / ".omx" / "state" / "lane_registry.json"
    if not registry_path.exists():
        pytest.skip(f"lane_registry.json not found at {registry_path}")
    registry = json.loads(registry_path.read_text())
    lanes = registry.get("lanes", [])
    matching = [
        lane for lane in lanes
        if lane.get("id") == "lane_z7_as_mamba_2_full_landing_20260518"
    ]
    if not matching:
        pytest.skip("lane_z7_as_mamba_2_full_landing_20260518 not yet registered")
    lane = matching[0]
    # At L1+ should declare substrate_engineering OR research_only
    notes = str(lane.get("notes", ""))
    assert (
        lane.get("lane_class") == "substrate_engineering"
        or lane.get("research_only") is True
        or "research_only" in notes.lower()
        or "substrate_engineering" in notes.lower()
    ), (
        f"Lane must declare lane_class=substrate_engineering or research_only=true; "
        f"got lane={lane}"
    )
