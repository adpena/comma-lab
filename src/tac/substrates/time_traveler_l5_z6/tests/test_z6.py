# SPDX-License-Identifier: MIT
"""Z6 Time-Traveler L5 F-asymptote-node substrate tests.

Test groups (≥10 per op-routable #2 spec):
- (A) FilmConditionedNextFramePredictor: 6 tests — shape, kernel sizes, identity mode,
       gradient flow, invalid kernel, invalid input dim
- (B) Encoder/Decoder: 3 tests — output shape/range, gradient flow, invalid input
- (C) Architecture: 4 tests — reconstruct_pair, num_parameters_breakdown,
       pair-index validation, autoregressive consistency
- (D) Archive: 7 tests — magic, version, roundtrip, determinism, header size,
       parse errors, predictive-coding tag presence
- (E) Inflate: 3 tests — contest 3-arg contract, no scorer load, end-to-end roundtrip
- (F) Score-aware loss: 3 tests — eval_roundtrip mandatory, RGB-255 validation,
       residual-norm scaling
- (G) SubstrateContract / SCAFFOLD discipline: 4 tests — META layer
       decoration succeeds; substrate IS research_only; full_main raises
       NotImplementedError; Catalog #240 opt-out tokens present

Per CLAUDE.md "Subagent commits MUST use serializer" and Catalog #117/#157/#174.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest
import torch

import tac.substrates.time_traveler_l5_z6 as z6_module
from tac.substrates.time_traveler_l5_z6 import (
    Z6PCWM1_HEADER_FMT,
    Z6PCWM1_HEADER_SIZE,
    Z6PCWM1_MAGIC,
    Z6PCWM1_SCHEMA_VERSION,
    FilmConditionedNextFramePredictor,
    Z6PredictiveCodingConfig,
    Z6PredictiveCodingLossWeights,
    Z6PredictiveCodingScoreAwareLoss,
    Z6PredictiveCodingSubstrate,
    pack_archive,
    parse_archive,
)
from tac.substrates.time_traveler_l5_z6.architecture import (
    _Z6Decoder,
    _Z6Encoder,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DRIVER_PATH = REPO_ROOT / "scripts" / "remote_lane_substrate_time_traveler_l5_z6.sh"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ===========================================================================
# (G) SubstrateContract / SCAFFOLD discipline
# ===========================================================================


def test_z6_module_declares_research_only_until_full_main_lands() -> None:
    """The substrate-level status must match the operator recipe gate."""
    doc = z6_module.__doc__ or ""
    assert "research_only: true" in doc
    assert "lane_class: ``substrate_engineering``" in doc
    assert "NotImplementedError" in doc


def test_z6_module_declares_archive_grammar_8_fields() -> None:
    """Catalog #124: 8 archive-grammar fields must appear in module docstring."""
    doc = z6_module.__doc__ or ""
    for field in (
        "archive_grammar",
        "parser_section_manifest",
        "inflate_runtime_loc_budget",
        "runtime_dep_closure",
        "export_format",
        "score_aware_loss",
        "bolt_on_loc_budget",
        "no_op_detector_planned",
    ):
        assert field in doc, f"missing Catalog #124 field {field!r}"


def test_substrate_contract_decoration_succeeds() -> None:
    """Catalog #241/#242: importing the trainer decorates the contract.

    The decoration happens at module import time via @register_substrate;
    if any contract field is invalid, the import raises
    SubstrateContractError. This test confirms the contract is well-formed.
    """
    # Importing the trainer module forces the decorator to execute
    import experiments.train_substrate_time_traveler_l5_z6 as z6_trainer

    contract = z6_trainer.TIME_TRAVELER_L5_Z6_SUBSTRATE_CONTRACT
    assert contract.id == "time_traveler_l5_z6"
    assert contract.score_improvement_mechanism_status == "RESEARCH_ONLY"
    assert contract.recipe_research_only is True
    assert contract.recipe_smoke_only is True
    assert contract.hook_autopilot_ranker_class_shift_token == "Rao-Ballard"


def test_smoke_effective_epochs_caps_default_full_epoch_count() -> None:
    """Smoke remains a <=3 epoch artifact even though full default is 300."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6_trainer

    assert z6_trainer._smoke_effective_epochs(300) == 3
    assert z6_trainer._smoke_effective_epochs(3) == 3
    assert z6_trainer._smoke_effective_epochs(1) == 1
    assert z6_trainer._smoke_effective_epochs(0) == 1


def test_smoke_populates_nonzero_ego_motion_by_default_control() -> None:
    """Ramp smoke mode prevents zero-ego FiLM cargo-cult probes."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6_trainer

    sub = Z6PredictiveCodingSubstrate(_tiny_config())
    z6_trainer._populate_smoke_ego_motion(sub, mode="ramp", seed=0)
    assert float((sub.ego_motion_buffer.abs() > 0).float().mean().item()) == 1.0
    assert float(sub.ego_motion_buffer.pow(2).sum().sqrt().item()) > 0.0

    z6_trainer._populate_smoke_ego_motion(sub, mode="zero", seed=0)
    assert float((sub.ego_motion_buffer.abs() > 0).float().mean().item()) == 0.0
    assert float(sub.ego_motion_buffer.pow(2).sum().sqrt().item()) == 0.0


def test_remote_driver_verifies_active_claim_and_preserves_full_epoch_default() -> None:
    """Remote driver must verify the active claim and keep full default at 300."""
    src = DRIVER_PATH.read_text(encoding="utf-8")
    assert 'Z6_EPOCHS="${Z6_EPOCHS:-}"' in src
    assert 'Z6_EPOCHS="3"' in src
    assert 'Z6_EPOCHS="300"' in src
    assert "verify_active_dispatch_claim()" in src
    assert "claim_lane_dispatch.py\" summary" in src
    assert "--live-only" in src
    assert "CLAIM_VERIFIED=1" in src
    assert "trap cleanup EXIT" in src


def test_full_main_raises_not_implemented_phase_2_gate() -> None:
    """Catalog #240: _full_main must fail-loud pending Phase 2 council approval."""
    import argparse

    import experiments.train_substrate_time_traveler_l5_z6 as z6_trainer

    args = argparse.Namespace(
        full_cpu=False, advisory_cpu_explicitly_waived=False, smoke=False,
    )
    with pytest.raises(NotImplementedError, match="Phase 2 council approval"):
        z6_trainer._full_main(args)


# ===========================================================================
# (A) FilmConditionedNextFramePredictor — 6 tests
# ===========================================================================


def test_predictor_output_shape_kernel_3() -> None:
    p = FilmConditionedNextFramePredictor(
        latent_dim=8, hidden_dim=16, film_mlp_hidden_dim=8,
        ego_motion_dim=4, kernel_size=3,
    )
    z_prev = torch.rand(3, 8)
    ego = torch.rand(3, 4)
    z_pred = p(z_prev, ego)
    assert z_pred.shape == (3, 8)


def test_predictor_output_shape_kernel_1_and_5() -> None:
    """Test both supported odd kernel sizes."""
    for k in (1, 5):
        p = FilmConditionedNextFramePredictor(
            latent_dim=8, hidden_dim=16, film_mlp_hidden_dim=8,
            ego_motion_dim=4, kernel_size=k,
        )
        z_pred = p(torch.rand(2, 8), torch.rand(2, 4))
        assert z_pred.shape == (2, 8)


def test_predictor_invalid_kernel_size_refused() -> None:
    """kernel_size must be 1, 3, or 5 AND must be odd."""
    with pytest.raises(ValueError, match="must be 1, 3, or 5"):
        FilmConditionedNextFramePredictor(
            latent_dim=8, hidden_dim=16, film_mlp_hidden_dim=8,
            ego_motion_dim=4, kernel_size=7,
        )
    # Even kernel size (2) — 2 is also not in [1, 3, 5], so the first check fires
    with pytest.raises(ValueError, match="must be 1, 3, or 5"):
        FilmConditionedNextFramePredictor(
            latent_dim=8, hidden_dim=16, film_mlp_hidden_dim=8,
            ego_motion_dim=4, kernel_size=2,
        )


def test_predictor_identity_mode_returns_input() -> None:
    """identity_predictor=True returns z_prev verbatim (probe-disambiguator)."""
    p = FilmConditionedNextFramePredictor(
        latent_dim=8, hidden_dim=16, film_mlp_hidden_dim=8,
        ego_motion_dim=4, kernel_size=3, identity_predictor=True,
    )
    z_prev = torch.rand(3, 8)
    ego = torch.rand(3, 4)
    z_pred = p(z_prev, ego)
    assert torch.equal(z_pred, z_prev)
    # Identity predictor has zero trainable params
    assert p.num_parameters() == 0


def test_predictor_gradient_flow() -> None:
    p = FilmConditionedNextFramePredictor(
        latent_dim=8, hidden_dim=16, film_mlp_hidden_dim=8,
        ego_motion_dim=4, kernel_size=3,
    )
    z_prev = torch.rand(2, 8, requires_grad=True)
    ego = torch.rand(2, 4, requires_grad=True)
    z_pred = p(z_prev, ego)
    z_pred.sum().backward()
    assert z_prev.grad is not None
    assert ego.grad is not None
    assert torch.isfinite(z_prev.grad).all()
    assert torch.isfinite(ego.grad).all()


def test_predictor_invalid_input_dim() -> None:
    p = FilmConditionedNextFramePredictor(
        latent_dim=8, hidden_dim=16, film_mlp_hidden_dim=8,
        ego_motion_dim=4, kernel_size=3,
    )
    with pytest.raises(ValueError, match="z_prev last dim"):
        p(torch.rand(2, 16), torch.rand(2, 4))
    with pytest.raises(ValueError, match="ego_motion last dim"):
        p(torch.rand(2, 8), torch.rand(2, 16))


# ===========================================================================
# (B) Encoder/Decoder — 3 tests
# ===========================================================================


def test_encoder_output_shape() -> None:
    enc = _Z6Encoder(input_channels=3, hidden_dim=32, latent_dim=16)
    mu, logvar = enc(torch.rand(2, 3, 48, 64))
    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)


def test_decoder_output_shape_and_range() -> None:
    dec = _Z6Decoder(
        latent_dim=8, embed_dim=16, initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(12, 10, 8, 6), num_upsample_blocks=4,
        output_height=48, output_width=64,
    )
    rgb_0, rgb_1 = dec(torch.rand(2, 8))
    assert rgb_0.shape == (2, 3, 48, 64)
    assert rgb_1.shape == (2, 3, 48, 64)
    # sigmoid → [0, 1]
    assert (rgb_0 >= 0).all() and (rgb_0 <= 1).all()
    assert (rgb_1 >= 0).all() and (rgb_1 <= 1).all()


def test_encoder_invalid_input() -> None:
    enc = _Z6Encoder(input_channels=3, hidden_dim=32, latent_dim=16)
    with pytest.raises(ValueError, match="expects \\(B, C, H, W\\)"):
        enc(torch.rand(2, 3, 48))


# ===========================================================================
# (C) Architecture — 4 tests
# ===========================================================================


def _tiny_config(num_pairs: int = 5) -> Z6PredictiveCodingConfig:
    return Z6PredictiveCodingConfig(
        latent_dim=8, decoder_embed_dim=16,
        decoder_channels=(12, 10, 8, 6), decoder_num_upsample_blocks=4,
        num_pairs=num_pairs, output_height=48, output_width=64,
        predictor_hidden_dim=16, predictor_film_mlp_hidden_dim=8,
        predictor_ego_motion_dim=4,
    )


def test_reconstruct_pair_shape_and_consistency() -> None:
    cfg = _tiny_config(num_pairs=5)
    sub = Z6PredictiveCodingSubstrate(cfg)
    idx = torch.tensor([0, 2, 4], dtype=torch.long)
    rgb_0, rgb_1, z_t = sub.reconstruct_pair(idx)
    assert rgb_0.shape == (3, 3, 48, 64)
    assert rgb_1.shape == (3, 3, 48, 64)
    assert z_t.shape == (3, 8)


def test_num_parameters_breakdown() -> None:
    cfg = _tiny_config()
    sub = Z6PredictiveCodingSubstrate(cfg)
    breakdown = sub.num_parameters_breakdown()
    for key in ("encoder", "decoder", "predictor", "latent_init", "residuals", "total"):
        assert key in breakdown
        assert breakdown[key] >= 0
    # Sum of non-total ≈ total (excluding latent_init + residuals which are also params)
    expected_total = (
        breakdown["encoder"] + breakdown["decoder"] + breakdown["predictor"]
        + breakdown["latent_init"] + breakdown["residuals"]
    )
    assert breakdown["total"] == expected_total


def test_reconstruct_pair_invalid_indices() -> None:
    cfg = _tiny_config(num_pairs=5)
    sub = Z6PredictiveCodingSubstrate(cfg)
    with pytest.raises(ValueError, match=r"pair_indices must be torch\.long"):
        sub.reconstruct_pair(torch.tensor([0, 1], dtype=torch.float32))
    with pytest.raises(ValueError, match="out of range"):
        sub.reconstruct_pair(torch.tensor([0, 99], dtype=torch.long))


def test_autoregressive_recurrence_consistency() -> None:
    """The predictor must roll forward consistently: z[t] = predictor(z[t-1]) + r[t]."""
    cfg = _tiny_config(num_pairs=5)
    cfg = Z6PredictiveCodingConfig(
        latent_dim=cfg.latent_dim, decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_channels=cfg.decoder_channels,
        decoder_num_upsample_blocks=cfg.decoder_num_upsample_blocks,
        num_pairs=cfg.num_pairs, output_height=cfg.output_height,
        output_width=cfg.output_width,
        predictor_hidden_dim=cfg.predictor_hidden_dim,
        predictor_film_mlp_hidden_dim=cfg.predictor_film_mlp_hidden_dim,
        predictor_ego_motion_dim=cfg.predictor_ego_motion_dim,
        identity_predictor=True,  # identity for easier verification
    )
    sub = Z6PredictiveCodingSubstrate(cfg)
    # With identity predictor: z[t] = z[t-1] + residuals[t]
    # So z[t] = latent_init + sum(residuals[1..t])
    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    _, _, z_t = sub.reconstruct_pair(idx)
    # z[0] = latent_init
    assert torch.allclose(z_t[0], sub.latent_init)
    # z[1] = z[0] + residuals[1]
    expected_z1 = sub.latent_init + sub.residuals[1]
    assert torch.allclose(z_t[1], expected_z1)


# ===========================================================================
# (D) Archive — 7 tests
# ===========================================================================


def _build_smoke_archive() -> bytes:
    cfg = _tiny_config(num_pairs=3)
    sub = Z6PredictiveCodingSubstrate(cfg)
    meta = {
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "predictor_hidden_dim": cfg.predictor_hidden_dim,
        "predictor_film_mlp_hidden_dim": cfg.predictor_film_mlp_hidden_dim,
        "latent_init_std": cfg.latent_init_std,
        "smoke": True,
    }
    return pack_archive(
        sub.encoder.state_dict(),
        sub.decoder.state_dict(),
        sub.predictor.state_dict(),
        sub.latent_init.detach().cpu(),
        sub.residuals.detach().cpu(),
        sub.ego_motion_buffer.detach().cpu(),
        meta,
        lambda_residual_entropy=1.0,
        predictor_kernel_size=3,
        identity_predictor=False,
    )


def test_archive_magic_and_version() -> None:
    archive = _build_smoke_archive()
    assert archive[:4] == Z6PCWM1_MAGIC
    # version byte at offset 4
    assert archive[4] == Z6PCWM1_SCHEMA_VERSION


def test_archive_header_size_invariant() -> None:
    assert Z6PCWM1_HEADER_SIZE == 39
    assert struct.calcsize(Z6PCWM1_HEADER_FMT) == Z6PCWM1_HEADER_SIZE


def test_archive_roundtrip() -> None:
    archive = _build_smoke_archive()
    arc = parse_archive(archive)
    assert arc.schema_version == Z6PCWM1_SCHEMA_VERSION
    assert arc.latent_init.shape == (8,)
    assert arc.residuals.shape == (3, 8)
    assert arc.ego_motion.shape == (3, 4)
    assert "predictive_coding_world_model_meta" in arc.meta


def test_archive_determinism() -> None:
    """Pack twice with identical inputs -> identical bytes."""
    torch.manual_seed(0)
    a1 = _build_smoke_archive()
    torch.manual_seed(0)
    a2 = _build_smoke_archive()
    assert a1 == a2


def test_archive_bad_magic_refused() -> None:
    archive = _build_smoke_archive()
    bad = b"ZXXX" + archive[4:]
    with pytest.raises(ValueError, match="bad magic"):
        parse_archive(bad)


def test_archive_truncated_refused() -> None:
    archive = _build_smoke_archive()
    with pytest.raises(ValueError, match=r"archive size .* expected"):
        parse_archive(archive[:-10])


def test_archive_predictive_coding_tag_present() -> None:
    """The predictive_coding_world_model_meta provenance tag MUST be present."""
    archive = _build_smoke_archive()
    arc = parse_archive(archive)
    pcwm = arc.meta["predictive_coding_world_model_meta"]
    assert pcwm["literature_anchor"].startswith("Rao-Ballard")
    assert pcwm["f_asymptote_node"] == "z6"
    assert pcwm["z_variant"] == "z6_film_conditioned_next_frame_predictor"
    assert pcwm["prediction_band_verdict"]["planning_band"] == [0.13, 0.16]
    assert pcwm["prediction_band_verdict"]["score_claim"] is False


# ===========================================================================
# (E) Inflate — 3 tests
# ===========================================================================


def test_inflate_module_exposes_3_arg_cli_per_catalog_146() -> None:
    """Catalog #146: inflate.py must honor 3-positional-arg contract."""
    from tac.substrates.time_traveler_l5_z6 import inflate as z6_inflate

    # The module has main_cli() entry point that takes 3 positional args
    assert hasattr(z6_inflate, "main_cli")
    assert hasattr(z6_inflate, "inflate_one_video")
    assert hasattr(z6_inflate, "_read_single_member_archive_bytes")


def test_inflate_no_scorer_load_per_catalog_6() -> None:
    """Catalog #6: inflate must not import scorer modules."""
    import inspect

    from tac.substrates.time_traveler_l5_z6 import inflate as z6_inflate

    src = inspect.getsource(z6_inflate)
    # Forbidden tokens per CLAUDE.md "Strict scorer rule"
    for forbidden in ("from upstream.modules", "import upstream.modules",
                      "rgb_to_yuv6", "EfficientNet", "FastViT",
                      "PoseNet", "SegNet"):
        assert forbidden not in src, f"inflate.py must not reference {forbidden!r}"


def test_inflate_end_to_end_roundtrip(tmp_path) -> None:
    """End-to-end: pack archive → inflate → produce raw frames."""
    from tac.substrates.time_traveler_l5_z6 import inflate as z6_inflate

    archive = _build_smoke_archive()
    raw_path = tmp_path / "0.raw"
    frames = z6_inflate.inflate_one_video(archive, raw_path, device="cpu")
    # 3 pairs x 2 frames = 6 frames written
    assert frames == 6
    assert raw_path.exists()
    assert raw_path.stat().st_size > 0


# ===========================================================================
# (F) Score-aware loss — 3 tests
# ===========================================================================


def test_score_aware_loss_eval_roundtrip_mandatory() -> None:
    """Catalog #5: apply_eval_roundtrip=False is forbidden."""
    weights = Z6PredictiveCodingLossWeights()
    seg = torch.nn.Identity()
    pose = torch.nn.Identity()
    loss_fn = Z6PredictiveCodingScoreAwareLoss(seg, pose, weights)
    rgb = torch.rand(2, 3, 48, 64) * 255.0
    residuals = torch.rand(5, 8)
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        loss_fn(
            reconstructed_rgb_0=rgb, reconstructed_rgb_1=rgb,
            gt_rgb_0=rgb, gt_rgb_1=rgb,
            archive_bytes_proxy=torch.tensor(100_000.0),
            residuals=residuals,
            apply_eval_roundtrip=False,
        )


def test_score_aware_loss_rgb_255_domain_validation() -> None:
    """Per RGB-255 domain validator: unit-domain RGB refused."""
    weights = Z6PredictiveCodingLossWeights()
    seg = torch.nn.Identity()
    pose = torch.nn.Identity()
    loss_fn = Z6PredictiveCodingScoreAwareLoss(seg, pose, weights)
    rgb_unit = torch.rand(2, 3, 48, 64)  # in [0, 1] — INVALID
    residuals = torch.rand(5, 8)
    with pytest.raises(ValueError, match="appears to be unit-domain RGB"):
        loss_fn(
            reconstructed_rgb_0=rgb_unit, reconstructed_rgb_1=rgb_unit,
            gt_rgb_0=rgb_unit, gt_rgb_1=rgb_unit,
            archive_bytes_proxy=torch.tensor(100_000.0),
            residuals=residuals,
        )


def test_score_aware_loss_negative_lambda_refused() -> None:
    """lambda_residual_entropy must be >= 0."""
    weights = Z6PredictiveCodingLossWeights(lambda_residual_entropy=-1.0)
    seg = torch.nn.Identity()
    pose = torch.nn.Identity()
    loss_fn = Z6PredictiveCodingScoreAwareLoss(seg, pose, weights)
    rgb = torch.rand(2, 3, 48, 64) * 255.0
    residuals = torch.rand(5, 8)
    with pytest.raises(ValueError, match="lambda_residual_entropy must be >= 0"):
        loss_fn(
            reconstructed_rgb_0=rgb, reconstructed_rgb_1=rgb,
            gt_rgb_0=rgb, gt_rgb_1=rgb,
            archive_bytes_proxy=torch.tensor(100_000.0),
            residuals=residuals,
        )
