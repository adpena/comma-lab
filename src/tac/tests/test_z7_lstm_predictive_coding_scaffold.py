# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import inspect
import json
import math
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from tac.substrates._shared.inflate_runtime import CAMERA_HW
from tac.substrates.time_traveler_l5_z6.architecture import _Z6Decoder
from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding import (
    GruRecurrentPredictor,
    LatentAffineContextConditioner,
    Z7PCWM1_MAGIC,
    Z7PCWM1_SECTION_ROLES,
    Z7GruPredictiveCodingConfig,
    Z7GruPredictiveCodingSubstrate,
    pack_archive,
    parse_archive,
    parse_z7pcwm1_archive_bytes,
    replay_latent_sequence,
    replay_latent_sequence_with_context,
)
from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding import (
    inflate as z7_inflate,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_PATH = (
    REPO_ROOT / "experiments" / "train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py"
)


def _load_trainer():
    spec = importlib.util.spec_from_file_location("z7_gru_trainer", TRAINER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakePoseScorer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(12, 12)

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        b, t, _c, _h, _w = pair_btchw.shape
        pooled = pair_btchw.mean(dim=(3, 4))
        pooled = torch.cat([pooled, pooled, pooled, pooled], dim=-1)
        return pooled.reshape(b, t, 12)

    def forward(self, x_btc: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t, c = x_btc.shape
        return {"pose": self.linear(x_btc.reshape(b * t, c)).reshape(b, t, 12)}


class _FakeSegScorer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, kernel_size=1)

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        return pair_btchw[:, -1]

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        return self.conv(x_bchw)


def _fake_score_aware_loss():
    from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.score_aware_loss import (
        Z7GruPredictiveCodingScoreAwareLoss,
        Z7PredictiveCodingLossWeights,
    )

    pose = _FakePoseScorer()
    seg = _FakeSegScorer()
    for param in list(pose.parameters()) + list(seg.parameters()):
        param.requires_grad_(False)
    pose.eval()
    seg.eval()
    return Z7GruPredictiveCodingScoreAwareLoss(
        seg_scorer=seg,
        pose_scorer=pose,
        weights=Z7PredictiveCodingLossWeights(gamma_pose=math.sqrt(10.0)),
    )


def test_z7_gru_predictor_matches_z6_forward_signature() -> None:
    cfg = Z7GruPredictiveCodingConfig(
        latent_dim=24,
        ego_motion_dim=8,
        gru_hidden_dim=32,
        gru_num_layers=2,
        stateful=True,
    )
    predictor = GruRecurrentPredictor(cfg)
    z_prev = torch.randn(3, cfg.latent_dim)
    ego = torch.randn(3, cfg.ego_motion_dim)

    out = predictor(z_prev, ego)

    assert out.shape == z_prev.shape
    assert predictor.num_parameters() > 0
    assert "forward(z_prev" in predictor.to_z6_compatible_signature()


def test_z7_gru_identity_predictor_is_zero_param_control() -> None:
    cfg = Z7GruPredictiveCodingConfig(identity_predictor=True)
    predictor = GruRecurrentPredictor(cfg)
    z_prev = torch.randn(2, cfg.latent_dim)
    ego = torch.randn(2, cfg.ego_motion_dim)

    out = predictor(z_prev, ego)

    assert torch.allclose(out, z_prev)
    assert predictor.num_parameters() == 0


def test_z7_gru_predictor_gradients_reach_latent_and_ego_inputs() -> None:
    cfg = Z7GruPredictiveCodingConfig(gru_hidden_dim=32, stateful=False)
    predictor = GruRecurrentPredictor(cfg)
    z_prev = torch.randn(4, cfg.latent_dim, requires_grad=True)
    ego = torch.randn(4, cfg.ego_motion_dim, requires_grad=True)

    predictor(z_prev, ego).sum().backward()

    assert z_prev.grad is not None
    assert z_prev.grad.abs().sum() > 0
    assert ego.grad is not None
    assert ego.grad.abs().sum() > 0


def test_z7_gru_stateful_recurrence_changes_repeated_input_output() -> None:
    torch.manual_seed(7)
    cfg = Z7GruPredictiveCodingConfig(gru_hidden_dim=32, stateful=True)
    predictor = GruRecurrentPredictor(cfg)
    predictor.eval()
    z_prev = torch.randn(1, cfg.latent_dim)
    ego = torch.randn(1, cfg.ego_motion_dim)

    with torch.no_grad():
        out_1 = predictor(z_prev, ego)
        out_2 = predictor(z_prev, ego)

    assert not torch.allclose(out_1, out_2)


def test_z7_gru_substrate_reconstructs_rgb_pairs_and_exports_decoder_metadata() -> None:
    cfg = Z7GruPredictiveCodingConfig(
        latent_dim=6,
        ego_motion_dim=3,
        gru_hidden_dim=8,
        num_pairs=2,
        decoder_embed_dim=4,
        decoder_initial_grid_h=2,
        decoder_initial_grid_w=2,
        decoder_channels=(4, 4),
        decoder_num_upsample_blocks=2,
        output_height=8,
        output_width=8,
    )
    model = Z7GruPredictiveCodingSubstrate(cfg)
    idx = torch.tensor([0, 1], dtype=torch.long)

    rgb_0, rgb_1, latents = model.reconstruct_pair(idx)

    assert rgb_0.shape == (2, 3, 8, 8)
    assert rgb_1.shape == (2, 3, 8, 8)
    assert latents.shape == (2, cfg.latent_dim)
    assert model.decoder_metadata()["decoder_channels"] == [4, 4]
    assert model.num_parameters_breakdown()["total"] == model.num_parameters()


def test_z7_latent_affine_context_conditioning_is_opt_in_and_trainable() -> None:
    cfg = Z7GruPredictiveCodingConfig(
        latent_dim=6,
        ego_motion_dim=3,
        gru_hidden_dim=8,
        num_pairs=2,
        decoder_embed_dim=4,
        decoder_initial_grid_h=2,
        decoder_initial_grid_w=2,
        decoder_channels=(4, 4),
        decoder_num_upsample_blocks=2,
        output_height=8,
        output_width=8,
        context_conditioning_mode="latent_affine",
        context_affine_strength=0.25,
    )
    model = Z7GruPredictiveCodingSubstrate(cfg)

    latents, contexts = model.replay_latents_and_contexts()
    conditioned = model.condition_latents(latents, contexts)
    rgb_0, rgb_1, replayed = model.reconstruct_all_pairs()
    (rgb_0.mean() + rgb_1.mean()).backward()

    assert model.context_conditioner is not None
    assert conditioned.shape == latents.shape
    assert not torch.allclose(conditioned, latents)
    assert replayed.shape == latents.shape
    assert model.decoder_metadata()["context_conditioning_mode"] == "latent_affine"
    assert model.decoder_metadata()["context_conditioner_state_dict_in_encoder_blob"]
    assert model.num_parameters_breakdown()["context_conditioner"] > 0
    grads = [
        param.grad
        for param in model.context_conditioner.parameters()
        if param.grad is not None
    ]
    assert grads
    assert sum(float(grad.abs().sum()) for grad in grads) > 0.0


def test_z7_gru_smoke_trainer_writes_false_authority_stats(tmp_path: Path) -> None:
    trainer = _load_trainer()
    args = Namespace(
        output_dir=str(tmp_path),
        video_path=str(REPO_ROOT / "upstream" / "videos" / "0.mkv"),
        epochs="1",
        batch_size="2",
        lr="5e-4",
        gru_hidden_dim="32",
        gru_num_layers="1",
        ego_source="posenet_projection",
        ego_motion_dim="8",
        identity_predictor=False,
        stateful=True,
        beta_ib="1.0",
        smoke=True,
        device="cpu",
    )

    assert trainer._smoke_main(args) == 0

    stats_path = tmp_path / "z7_gru_scaffold_smoke_stats.json"
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["substrate_id"] == "time_traveler_l5_z7_lstm_predictive_coding"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert "full_main_export_smoke_available_but_not_score_authority" in payload[
        "result_review_blockers"
    ]


def test_z7_gru_full_main_writes_byte_closed_prebuild_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _load_trainer()

    def fake_decode_real_pairs(*_args, **kwargs):
        n_pairs = int(kwargs["max_pairs"])
        return torch.rand(n_pairs, 2, 3, 16, 16) * 255.0

    monkeypatch.setattr(trainer, "_decode_real_pairs", fake_decode_real_pairs)
    args = Namespace(
        output_dir=str(tmp_path),
        video_path=str(REPO_ROOT / "upstream" / "videos" / "0.mkv"),
        epochs="1",
        batch_size="2",
        lr="1e-3",
        gru_hidden_dim="8",
        gru_num_layers="1",
        ego_source="posenet_projection",
        ego_motion_dim="3",
        identity_predictor=False,
        stateful=True,
        beta_ib="1.0",
        latent_dim="6",
        max_pairs="2",
        decoder_embed_dim="4",
        decoder_channels="4,4",
        decoder_num_upsample_blocks="2",
        decoder_initial_grid_h="2",
        decoder_initial_grid_w="2",
        output_height="8",
        output_width="8",
        inflate_verify=False,
        emit_static_control=True,
        loss_mode="proxy",
        upstream_dir=str(REPO_ROOT / "upstream"),
        alpha_rate="25.0",
        beta_seg="100.0",
        gamma_pose=str(math.sqrt(10.0)),
        noise_std="0.0",
        smoke=False,
        device="cpu",
    )

    assert trainer._full_main(args) == 0

    stats_path = tmp_path / "z7_gru_prebuild_full_main_export_stats.json"
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert payload["archive_bin_bytes"] > 0
    assert payload["timing_smoke"]["axis"] == "[local-trainer-timing advisory]"
    assert payload["timing_smoke"]["loss_mode"] == "proxy"
    assert payload["timing_smoke"]["seconds_per_epoch"] >= 0.0
    assert payload["timing_smoke"]["seconds_per_pair_epoch"] >= 0.0
    assert (
        payload["timing_smoke"]["stage_wall_seconds"]["export_packaging_seconds"]
        >= 0.0
    )
    control = payload["static_capacity_control"]
    assert control["same_archive_zip_bytes_as_recurrent"] is True
    assert control["archive_zip_bytes"] == payload["archive_zip_bytes"]
    assert control["archive_bin_sha256"] != payload["archive_bin_sha256"]
    assert (tmp_path / "0.bin").is_file()
    assert (tmp_path / "archive.zip").is_file()
    assert (tmp_path / "static_capacity_control" / "0.bin").is_file()
    assert (tmp_path / "static_capacity_control" / "archive.zip").is_file()
    assert (tmp_path / "submission_runtime" / "inflate.sh").is_file()
    assert "not_score_aware_scorer_loss" in payload["result_review_blockers"]


def test_z7_score_aware_loss_routes_scorer_contract_and_backprops() -> None:
    loss_fn = _fake_score_aware_loss()
    rgb_0 = (torch.rand(1, 3, 8, 8) * 255.0).requires_grad_()
    rgb_1 = (torch.rand(1, 3, 8, 8) * 255.0).requires_grad_()
    gt_0 = torch.rand(1, 3, 8, 8) * 255.0
    gt_1 = torch.rand(1, 3, 8, 8) * 255.0
    residuals = torch.randn(2, 6, requires_grad=True)
    latents = torch.randn(2, 6, requires_grad=True)

    loss, parts = loss_fn(
        reconstructed_rgb_0=rgb_0,
        reconstructed_rgb_1=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
        archive_bytes_proxy=torch.tensor(5_000.0),
        residuals=residuals,
        latents=latents,
        apply_eval_roundtrip=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert parts["seg_term"].item() >= 0.0
    assert parts["pose_term"].item() >= 0.0
    assert rgb_0.grad is not None and rgb_0.grad.abs().sum() > 0
    assert residuals.grad is not None and residuals.grad.abs().sum() > 0


def test_z7_score_aware_loss_refuses_eval_roundtrip_false() -> None:
    loss_fn = _fake_score_aware_loss()
    with pytest.raises(ValueError, match="apply_eval_roundtrip"):
        loss_fn(
            reconstructed_rgb_0=torch.rand(1, 3, 8, 8) * 255.0,
            reconstructed_rgb_1=torch.rand(1, 3, 8, 8) * 255.0,
            gt_rgb_0=torch.rand(1, 3, 8, 8) * 255.0,
            gt_rgb_1=torch.rand(1, 3, 8, 8) * 255.0,
            archive_bytes_proxy=torch.tensor(5_000.0),
            residuals=torch.randn(2, 6),
            latents=torch.randn(2, 6),
            apply_eval_roundtrip=False,
        )


def test_z7_gru_full_main_score_aware_mode_keeps_false_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _load_trainer()

    def fake_decode_real_pairs(*_args, **kwargs):
        n_pairs = int(kwargs["max_pairs"])
        return torch.rand(n_pairs, 2, 3, 8, 8) * 255.0

    monkeypatch.setattr(trainer, "_decode_real_pairs", fake_decode_real_pairs)
    monkeypatch.setattr(
        trainer,
        "_build_score_aware_loss",
        lambda **_kwargs: _fake_score_aware_loss(),
    )
    args = Namespace(
        output_dir=str(tmp_path),
        video_path=str(REPO_ROOT / "upstream" / "videos" / "0.mkv"),
        epochs="1",
        batch_size="1",
        lr="1e-3",
        gru_hidden_dim="8",
        gru_num_layers="1",
        ego_source="posenet_projection",
        ego_motion_dim="3",
        identity_predictor=False,
        stateful=True,
        beta_ib="1.0",
        latent_dim="6",
        max_pairs="1",
        decoder_embed_dim="4",
        decoder_channels="4,4",
        decoder_num_upsample_blocks="2",
        decoder_initial_grid_h="2",
        decoder_initial_grid_w="2",
        output_height="8",
        output_width="8",
        inflate_verify=False,
        emit_static_control=False,
        loss_mode="score_aware",
        upstream_dir=str(REPO_ROOT / "upstream"),
        alpha_rate="25.0",
        beta_seg="100.0",
        gamma_pose=str(math.sqrt(10.0)),
        noise_std="0.0",
        smoke=False,
        device="cpu",
    )

    assert trainer._full_main(args) == 0

    payload = json.loads(
        (tmp_path / "z7_gru_prebuild_full_main_export_stats.json").read_text(
            encoding="utf-8"
        )
    )
    parsed = parse_archive((tmp_path / "0.bin").read_bytes())
    meta = parsed.meta["z7_recurrent_predictive_coding_meta"]
    assert payload["loss_mode"] == "score_aware"
    assert payload["score_aware_scorer_loss_used"] is True
    assert payload["final_loss_proxy"] is None
    assert payload["final_loss_score_aware"] == payload["final_loss"]
    assert payload["timing_smoke"]["loss_mode"] == "score_aware"
    assert payload["timing_smoke"]["num_pairs"] == 1
    assert payload["timing_smoke"]["score_claim"] is False
    assert (
        payload["timing_smoke"]["stage_wall_seconds"][
            "score_aware_scorer_load_seconds"
        ]
        >= 0.0
    )
    assert payload["score_claim"] is False
    assert "not_score_aware_scorer_loss" not in payload["result_review_blockers"]
    assert "score_aware_trained_packet_not_auth_eval_validated" in payload[
        "result_review_blockers"
    ]
    assert meta["score_aware_scorer_loss_used"] is True
    assert "score_aware_training_absent_prebuild" not in meta["blockers"]
    assert "score_aware_trained_packet_not_auth_eval_validated" in meta["blockers"]


def _archive_config() -> Z7GruPredictiveCodingConfig:
    return Z7GruPredictiveCodingConfig(
        latent_dim=6,
        ego_motion_dim=3,
        gru_hidden_dim=8,
        gru_num_layers=1,
        num_pairs=5,
    )


def _archive_inputs(config: Z7GruPredictiveCodingConfig) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(17)
    predictor = GruRecurrentPredictor(config)
    if config.context_conditioning_mode == "latent_affine":
        encoder = LatentAffineContextConditioner(config).state_dict()
    else:
        encoder = {"encoder.weight": torch.randn(2, 3)}
    decoder = {"decoder.bias": torch.randn(4)}
    latent_init = torch.linspace(-0.5, 0.5, config.latent_dim)
    residuals = torch.randn(config.num_pairs, config.latent_dim) * 0.05
    ego_motion = torch.randn(config.num_pairs, config.ego_motion_dim) * 0.1
    return (
        encoder,
        decoder,
        predictor.state_dict(),
        latent_init,
        residuals,
        ego_motion,
    )


def _inflatable_archive_config() -> Z7GruPredictiveCodingConfig:
    return Z7GruPredictiveCodingConfig(
        latent_dim=6,
        ego_motion_dim=3,
        gru_hidden_dim=8,
        gru_num_layers=1,
        num_pairs=2,
    )


def _tiny_z6_decoder(config: Z7GruPredictiveCodingConfig) -> _Z6Decoder:
    torch.manual_seed(23)
    return _Z6Decoder(
        latent_dim=config.latent_dim,
        embed_dim=4,
        initial_grid_h=2,
        initial_grid_w=2,
        decoder_channels=(4, 4),
        num_upsample_blocks=2,
        output_height=8,
        output_width=8,
    )


def _inflatable_archive_blob(
    *,
    decoder_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[bytes, Z7GruPredictiveCodingConfig]:
    config = _inflatable_archive_config()
    encoder, _decoder, predictor_sd, latent_init, residuals, ego_motion = (
        _archive_inputs(config)
    )
    if decoder_state_dict is None:
        decoder_state_dict = _tiny_z6_decoder(config).state_dict()
    meta = {
        "test_vector": "z7_inflate_runtime_scaffold",
        "decoder_embed_dim": 4,
        "decoder_initial_grid_h": 2,
        "decoder_initial_grid_w": 2,
        "decoder_channels": [4, 4],
        "decoder_num_upsample_blocks": 2,
        "output_height": 8,
        "output_width": 8,
    }
    return (
        pack_archive(
            encoder,
            decoder_state_dict,
            predictor_sd,
            latent_init,
            residuals,
            ego_motion,
            meta,
            config=config,
        ),
        config,
    )


def test_z7pcwm1_archive_roundtrip_is_deterministic_and_false_authority() -> None:
    config = _archive_config()
    args = _archive_inputs(config)

    blob_a = pack_archive(*args, {"test_vector": "z7pcwm1"}, config=config)
    blob_b = pack_archive(*args, {"test_vector": "z7pcwm1"}, config=config)
    parsed = parse_archive(blob_a)

    assert blob_a == blob_b
    assert blob_a[:4] == Z7PCWM1_MAGIC
    assert parsed.config == config
    assert parsed.latent_init.shape == (config.latent_dim,)
    assert parsed.residuals.shape == (config.num_pairs, config.latent_dim)
    assert parsed.ego_motion.shape == (config.num_pairs, config.ego_motion_dim)
    meta = parsed.meta["z7_recurrent_predictive_coding_meta"]
    assert meta["score_claim"] is False
    assert meta["promotion_eligible"] is False
    assert meta["ready_for_paid_dispatch"] is False
    assert "proxy_trained_packet_not_score_aware_or_auth_eval_validated" in meta[
        "blockers"
    ]


def test_z7pcwm1_section_parser_roles_and_size_guards() -> None:
    config = _archive_config()
    blob = pack_archive(*_archive_inputs(config), {"test_vector": "sections"}, config=config)
    sections = parse_z7pcwm1_archive_bytes(blob)

    assert list(sections) == [
        "z7pcwm1_header",
        "encoder_blob",
        "decoder_blob",
        "predictor_blob",
        "latent_init_blob",
        "residuals_blob",
        "ego_motion_blob",
        "meta_blob",
    ]
    assert Z7PCWM1_SECTION_ROLES["predictor_blob"] == "decoder_weight_stream"
    assert Z7PCWM1_SECTION_ROLES["ego_motion_blob"] == "sidecar_or_correction_stream"
    with pytest.raises(ValueError, match="bad magic"):
        parse_z7pcwm1_archive_bytes(b"BAD!" + blob[4:])
    with pytest.raises(ValueError, match="archive size"):
        parse_z7pcwm1_archive_bytes(blob + b"\x00")
    with pytest.raises(ValueError, match="too short"):
        parse_z7pcwm1_archive_bytes(blob[:10])


def test_z7pcwm1_replay_consumes_predictor_bytes() -> None:
    config = _archive_config()
    encoder, decoder, predictor_sd, latent_init, residuals, ego_motion = _archive_inputs(config)
    base_blob = pack_archive(
        encoder,
        decoder,
        predictor_sd,
        latent_init,
        residuals,
        ego_motion,
        {"test_vector": "predictor_consumed"},
        config=config,
    )
    mutated_sd = {k: v.clone() for k, v in predictor_sd.items()}
    first_key = sorted(mutated_sd)[0]
    mutated_sd[first_key] = mutated_sd[first_key] + 0.25
    mutated_blob = pack_archive(
        encoder,
        decoder,
        mutated_sd,
        latent_init,
        residuals,
        ego_motion,
        {"test_vector": "predictor_consumed"},
        config=config,
    )

    base_replay = replay_latent_sequence(parse_archive(base_blob))
    mutated_replay = replay_latent_sequence(parse_archive(mutated_blob))

    assert base_replay.shape == (config.num_pairs, config.latent_dim)
    assert not torch.allclose(base_replay, mutated_replay)


def test_z7pcwm1_context_conditioner_is_byte_closed_and_parse_consumed() -> None:
    config = Z7GruPredictiveCodingConfig(
        latent_dim=6,
        ego_motion_dim=3,
        gru_hidden_dim=8,
        gru_num_layers=1,
        num_pairs=3,
        context_conditioning_mode="latent_affine",
        context_affine_strength=0.25,
    )
    encoder, decoder, predictor_sd, latent_init, residuals, ego_motion = (
        _archive_inputs(config)
    )
    blob = pack_archive(
        encoder,
        decoder,
        predictor_sd,
        latent_init,
        residuals,
        ego_motion,
        {"test_vector": "context_conditioner_consumed"},
        config=config,
    )

    parsed = parse_archive(blob)
    latents, contexts = replay_latent_sequence_with_context(parsed)
    conditioner = LatentAffineContextConditioner(parsed.config)
    conditioner.load_state_dict(parsed.encoder_state_dict, strict=True)
    conditioned = conditioner(latents, contexts)
    meta = parsed.meta["z7_recurrent_predictive_coding_meta"]

    assert parsed.config.context_conditioning_mode == "latent_affine"
    assert parsed.encoder_state_dict
    assert meta["decoder_context_conditioning"] == "latent_affine"
    assert "context_conditioned_decoder_requires_paired_exact_eval" in meta["blockers"]
    assert not torch.allclose(conditioned, latents)


def test_z7pcwm1_context_conditioned_archive_requires_conditioner_state() -> None:
    config = Z7GruPredictiveCodingConfig(
        latent_dim=6,
        ego_motion_dim=3,
        gru_hidden_dim=8,
        gru_num_layers=1,
        num_pairs=3,
        context_conditioning_mode="latent_affine",
    )
    _encoder, decoder, predictor_sd, latent_init, residuals, ego_motion = (
        _archive_inputs(config)
    )

    with pytest.raises(ValueError, match="context conditioner state_dict"):
        pack_archive(
            {},
            decoder,
            predictor_sd,
            latent_init,
            residuals,
            ego_motion,
            {"test_vector": "missing_context_conditioner"},
            config=config,
        )


def test_z7_inflate_runtime_scaffold_is_scorer_free_and_three_arg_cli() -> None:
    source = inspect.getsource(z7_inflate)

    assert "usage: inflate.py <archive_dir> <output_dir> <file_list>" in source
    for forbidden in (
        "tac.scorer",
        "contest_auth_eval",
        "PoseNet",
        "SegNet",
        "upstream.modules",
    ):
        assert forbidden not in source


def test_z7_inflate_one_video_consumes_archive_and_writes_raw(tmp_path: Path) -> None:
    blob, config = _inflatable_archive_blob()
    raw_path = tmp_path / "out.raw"

    frames = z7_inflate.inflate_one_video(blob, raw_path, device="cpu")

    assert frames == 2 * config.num_pairs
    assert raw_path.stat().st_size == frames * CAMERA_HW[0] * CAMERA_HW[1] * 3


def test_z7_inflate_runtime_consumes_decoder_weight_stream(tmp_path: Path) -> None:
    config = _inflatable_archive_config()
    base_decoder_sd = _tiny_z6_decoder(config).state_dict()
    mutated_decoder_sd = {k: v.clone() for k, v in base_decoder_sd.items()}
    final_bias_key = next(
        key
        for key in sorted(mutated_decoder_sd)
        if key.endswith("bias") and mutated_decoder_sd[key].numel() == 6
    )
    mutated_decoder_sd[final_bias_key] = mutated_decoder_sd[final_bias_key] + 4.0
    base_blob, _config = _inflatable_archive_blob(decoder_state_dict=base_decoder_sd)
    mutated_blob, _config = _inflatable_archive_blob(
        decoder_state_dict=mutated_decoder_sd,
    )
    base_raw = tmp_path / "base.raw"
    mutated_raw = tmp_path / "mutated.raw"

    z7_inflate.inflate_one_video(base_blob, base_raw, device="cpu")
    z7_inflate.inflate_one_video(mutated_blob, mutated_raw, device="cpu")

    assert base_raw.read_bytes() != mutated_raw.read_bytes()


def test_z7_inflate_runtime_consumes_context_conditioner_stream(tmp_path: Path) -> None:
    config = Z7GruPredictiveCodingConfig(
        latent_dim=6,
        ego_motion_dim=3,
        gru_hidden_dim=8,
        gru_num_layers=1,
        num_pairs=2,
        context_conditioning_mode="latent_affine",
        context_affine_strength=0.25,
    )
    encoder, _decoder, predictor_sd, latent_init, residuals, ego_motion = (
        _archive_inputs(config)
    )
    mutated_encoder = {k: v.clone() for k, v in encoder.items()}
    mutated_encoder["proj.bias"] = mutated_encoder["proj.bias"] + 4.0
    decoder_sd = _tiny_z6_decoder(config).state_dict()
    meta = {
        "test_vector": "z7_context_conditioner_runtime_scaffold",
        "decoder_embed_dim": 4,
        "decoder_initial_grid_h": 2,
        "decoder_initial_grid_w": 2,
        "decoder_channels": [4, 4],
        "decoder_num_upsample_blocks": 2,
        "output_height": 8,
        "output_width": 8,
    }
    base_blob = pack_archive(
        encoder,
        decoder_sd,
        predictor_sd,
        latent_init,
        residuals,
        ego_motion,
        meta,
        config=config,
    )
    mutated_blob = pack_archive(
        mutated_encoder,
        decoder_sd,
        predictor_sd,
        latent_init,
        residuals,
        ego_motion,
        meta,
        config=config,
    )
    base_raw = tmp_path / "context_base.raw"
    mutated_raw = tmp_path / "context_mutated.raw"

    z7_inflate.inflate_one_video(base_blob, base_raw, device="cpu")
    z7_inflate.inflate_one_video(mutated_blob, mutated_raw, device="cpu")

    assert base_raw.read_bytes() != mutated_raw.read_bytes()
