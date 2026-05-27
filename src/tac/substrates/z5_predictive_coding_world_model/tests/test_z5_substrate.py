# SPDX-License-Identifier: MIT
"""Z5 predictive-coding world-model substrate tests — 25+ tests across 6 module groups.

Test groups:
- (A) HierarchicalPredictor: 5 tests — shape, gradient flow, layer count, identity mode, num_parameters
- (B) Encoder/Decoder: 3 tests — output shape/range, gradient flow
- (C) Architecture: 5 tests — reconstruct_pair, num_parameters_breakdown,
       pair-index validation, autoregressive consistency
- (D) Archive: 7 tests — magic, version, roundtrip, determinism, header sizes,
       parse errors, predictive-coding tag presence
- (E) Inflate: 4 tests — contest 3-arg contract, no scorer load,
       end-to-end roundtrip, ambiguous archive refusal
- (F) Score-aware loss: 5 tests — eval_roundtrip mandatory, RGB-255 validation,
       residual-norm scaling, parts dict shape, identity-predictor regime

Per CLAUDE.md "Subagent commits MUST use serializer" and Catalog #117/#157/#174.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest
import torch

import tac.substrates.z5_predictive_coding_world_model as z5_module
from tac.substrates.z5_predictive_coding_world_model import (
    Z5PCWM1_MAGIC,
    Z5PCWM1_SCHEMA_VERSION,
    HierarchicalPredictor,
    PredictiveCodingConfig,
    PredictiveCodingLossWeights,
    PredictiveCodingScoreAwareLoss,
    PredictiveCodingSubstrate,
    pack_archive,
    parse_archive,
)
from tac.substrates.z5_predictive_coding_world_model.architecture import (
    _Z5Decoder,
    _Z5Encoder,
)
from tac.substrates.z5_predictive_coding_world_model.archive import (
    Z5PCWM1_HEADER_FMT,
    Z5PCWM1_HEADER_SIZE,
)

# ===========================================================================
# (A) HierarchicalPredictor — 5 tests
# ===========================================================================


def test_z5_module_declares_research_only_until_paid_dispatch_clears() -> None:
    """The substrate-level status must match the operator recipe gate.

    CLASS-SHIFT-FULL-MAIN-CLUSTER 2026-05-27: the trainer ``_full_main`` is now
    IMPLEMENTED (code complete) but PAID DISPATCH stays gated by ``research_only:
    true`` on the recipe per Catalog #325 until the per-substrate symposium + Z4
    canary anchor clear it. The package docstring records that recipe gate.
    """
    doc = z5_module.__doc__ or ""
    assert "research_only: true" in doc
    assert "research_only: false" not in doc


def test_z5_trainer_full_main_implemented_and_cuda_gated(tmp_path) -> None:
    """CLASS-SHIFT-FULL-MAIN-CLUSTER 2026-05-27: _full_main IMPLEMENTED + CUDA-gated.

    NotImplementedError extinguished; ``_full_main`` routes the canonical
    score-aware loop through ``run_pact_nerv_score_aware_training``. Full
    (non-smoke) path is CUDA-required (Catalog #1/#325): ``--device cpu``
    without ``--full-cpu`` refuses via SystemExit. PAID DISPATCH stays gated
    by the recipe per Catalog #325.
    """
    import importlib
    import inspect

    import pytest

    trainer = importlib.import_module(
        "experiments.train_substrate_z5_predictive_coding_world_model"
    )
    src = inspect.getsource(trainer._full_main)
    assert "raise NotImplementedError" not in src, (
        "_full_main NotImplementedError must be extinguished"
    )
    assert "run_pact_nerv_score_aware_training" in src
    args = trainer._build_parser().parse_args(
        ["--output-dir", str(tmp_path / "out"), "--device", "cpu", "--epochs", "1"]
    )
    with pytest.raises(SystemExit):
        trainer._full_main(args)

def test_predictor_output_shape_2_layer() -> None:
    p = HierarchicalPredictor(
        latent_dim=8, hidden_dim=16, num_layers=2, ego_motion_dim=4
    )
    z_prev = torch.rand(3, 8)
    ego = torch.rand(3, 4)
    z_pred = p(z_prev, ego)
    assert z_pred.shape == (3, 8)


def test_predictor_output_shape_3_layer() -> None:
    p = HierarchicalPredictor(
        latent_dim=8, hidden_dim=16, num_layers=3, ego_motion_dim=4
    )
    z_prev = torch.rand(2, 8)
    ego = torch.rand(2, 4)
    z_pred = p(z_prev, ego)
    assert z_pred.shape == (2, 8)


def test_predictor_invalid_num_layers_refused() -> None:
    """num_layers must be 2 or 3 (HierarchicalPredictor depth constraint)."""
    with pytest.raises(ValueError, match="predictor_num_layers must be 2 or 3"):
        HierarchicalPredictor(latent_dim=8, hidden_dim=16, num_layers=1, ego_motion_dim=4)
    with pytest.raises(ValueError, match="predictor_num_layers must be 2 or 3"):
        HierarchicalPredictor(latent_dim=8, hidden_dim=16, num_layers=4, ego_motion_dim=4)


def test_predictor_identity_mode_returns_input() -> None:
    """identity_predictor=True returns z_prev verbatim."""
    p = HierarchicalPredictor(
        latent_dim=8, hidden_dim=16, num_layers=2, ego_motion_dim=4,
        identity_predictor=True,
    )
    z_prev = torch.rand(3, 8)
    ego = torch.rand(3, 4)
    z_pred = p(z_prev, ego)
    assert torch.equal(z_pred, z_prev)
    # Identity predictor has zero trainable params
    assert p.num_parameters() == 0


def test_predictor_gradient_flow() -> None:
    p = HierarchicalPredictor(
        latent_dim=8, hidden_dim=16, num_layers=2, ego_motion_dim=4
    )
    z_prev = torch.rand(2, 8, requires_grad=True)
    ego = torch.rand(2, 4)
    z_pred = p(z_prev, ego)
    z_pred.sum().backward()
    assert z_prev.grad is not None
    assert torch.isfinite(z_prev.grad).all()


def test_predictor_invalid_input_dim() -> None:
    p = HierarchicalPredictor(latent_dim=8, hidden_dim=16, num_layers=2, ego_motion_dim=4)
    with pytest.raises(ValueError, match="z_prev last dim"):
        p(torch.rand(2, 16), torch.rand(2, 4))
    with pytest.raises(ValueError, match="ego_motion last dim"):
        p(torch.rand(2, 8), torch.rand(2, 16))


# ===========================================================================
# (B) Encoder/Decoder — 3 tests
# ===========================================================================

def test_encoder_output_shape() -> None:
    enc = _Z5Encoder(input_channels=3, hidden_dim=32, latent_dim=16)
    mu, logvar = enc(torch.rand(2, 3, 48, 64))
    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)


def test_decoder_output_shape_and_range() -> None:
    dec = _Z5Decoder(
        latent_dim=8,
        embed_dim=16,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(12, 10, 8, 6),
        num_upsample_blocks=4,
        output_height=48,
        output_width=64,
    )
    z = torch.rand(2, 8)
    rgb_0, rgb_1 = dec(z)
    assert rgb_0.shape == (2, 3, 48, 64)
    assert float(rgb_0.min()) >= 0.0 and float(rgb_0.max()) <= 1.0


def test_decoder_gradient_flow() -> None:
    dec = _Z5Decoder(
        latent_dim=8,
        embed_dim=16,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(12, 10, 8, 6),
        num_upsample_blocks=4,
        output_height=48,
        output_width=64,
    )
    z = torch.rand(2, 8, requires_grad=True)
    rgb_0, rgb_1 = dec(z)
    (rgb_0.sum() + rgb_1.sum()).backward()
    assert z.grad is not None


# ===========================================================================
# (C) Architecture — 5 tests
# ===========================================================================

def _make_tiny_config(num_pairs: int = 5) -> PredictiveCodingConfig:
    return PredictiveCodingConfig(
        latent_dim=8,
        decoder_embed_dim=16,
        decoder_channels=(12, 10, 8, 6),
        decoder_num_upsample_blocks=4,
        num_pairs=num_pairs,
        output_height=48,
        output_width=64,
        predictor_hidden_dim=16,
        predictor_num_layers=2,
        predictor_ego_motion_dim=4,
    )


def test_substrate_reconstruct_pair_shape() -> None:
    s = PredictiveCodingSubstrate(_make_tiny_config(num_pairs=5))
    idx = torch.tensor([0, 2, 4], dtype=torch.long)
    rgb_0, rgb_1, z_t = s.reconstruct_pair(idx)
    assert rgb_0.shape == (3, 3, 48, 64)
    assert rgb_1.shape == (3, 3, 48, 64)
    assert z_t.shape == (3, 8)


def test_substrate_num_parameters_breakdown_sums_to_total() -> None:
    s = PredictiveCodingSubstrate(_make_tiny_config())
    bd = s.num_parameters_breakdown()
    assert (
        bd["encoder"] + bd["decoder"] + bd["predictor"]
        + bd["latent_init"] + bd["residuals"] == bd["total"]
    )


def test_substrate_pair_index_out_of_range() -> None:
    s = PredictiveCodingSubstrate(_make_tiny_config(num_pairs=5))
    idx = torch.tensor([0, 1, 6], dtype=torch.long)
    with pytest.raises(ValueError, match="out of range"):
        s.reconstruct_pair(idx)


def test_substrate_pair_index_wrong_dtype() -> None:
    s = PredictiveCodingSubstrate(_make_tiny_config())
    idx = torch.arange(3, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"must be torch\.long"):
        s.reconstruct_pair(idx)


def test_substrate_index_0_returns_latent_init() -> None:
    """For pair index 0, z_t MUST equal latent_init (autoregressive base case)."""
    s = PredictiveCodingSubstrate(_make_tiny_config(num_pairs=5))
    idx = torch.tensor([0], dtype=torch.long)
    _, _, z_t = s.reconstruct_pair(idx)
    expected = s.latent_init.unsqueeze(0)
    assert torch.allclose(z_t, expected)


def test_substrate_autoregression_recurrence() -> None:
    """Reconstruct pair t > 0 honors recurrence z_t = predictor(z_{t-1}) + residual_t."""
    s = PredictiveCodingSubstrate(_make_tiny_config(num_pairs=5))
    s.eval()
    with torch.no_grad():
        idx_0 = torch.tensor([0], dtype=torch.long)
        _, _, z_0 = s.reconstruct_pair(idx_0)
        idx_1 = torch.tensor([1], dtype=torch.long)
        _, _, z_1 = s.reconstruct_pair(idx_1)
        # Manually compute: z_1_pred = predictor(z_0, ego_motion[1]); z_1 = z_1_pred + residuals[1]
        ego_1 = s.ego_motion_buffer[1].unsqueeze(0)
        z_1_pred = s.predictor(z_0, ego_1)
        z_1_expected = z_1_pred + s.residuals[1].unsqueeze(0)
    assert torch.allclose(z_1, z_1_expected, atol=1e-6)


# ===========================================================================
# (D) Archive — 7 tests
# ===========================================================================

def _make_archive_inputs(num_pairs: int = 5):
    cfg = _make_tiny_config(num_pairs=num_pairs)
    s = PredictiveCodingSubstrate(cfg)
    return (
        s.encoder.state_dict(),
        s.decoder.state_dict(),
        s.predictor.state_dict(),
        s.latent_init.detach().cpu(),
        s.residuals.detach().cpu(),
        s.ego_motion_buffer.detach().cpu(),
        {
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
            "latent_init_std": cfg.latent_init_std,
        },
        cfg,
    )


def test_archive_magic_and_version() -> None:
    enc, dec, pred, li, r, e, meta, _ = _make_archive_inputs()
    blob = pack_archive(enc, dec, pred, li, r, e, meta)
    assert blob[:4] == Z5PCWM1_MAGIC
    assert blob[4] == Z5PCWM1_SCHEMA_VERSION


def test_archive_header_size_invariant() -> None:
    assert Z5PCWM1_HEADER_SIZE == 39, f"got {Z5PCWM1_HEADER_SIZE}"
    assert struct.calcsize(Z5PCWM1_HEADER_FMT) == 39


def test_archive_roundtrip_preserves_residuals_and_ego_motion() -> None:
    enc, dec, pred, li, r, e, meta, _ = _make_archive_inputs(num_pairs=4)
    blob = pack_archive(enc, dec, pred, li, r, e, meta)
    parsed = parse_archive(blob)
    assert parsed.residuals.shape == r.shape
    assert parsed.ego_motion.shape == e.shape
    # Both quantized via int8 with min/max range; tolerate quantization step
    assert float((parsed.residuals - r).abs().max()) < 0.05
    assert float((parsed.ego_motion - e).abs().max()) < 0.05


def test_archive_deterministic_same_inputs() -> None:
    enc, dec, pred, li, r, e, meta, _ = _make_archive_inputs()
    a = pack_archive(enc, dec, pred, li, r, e, meta)
    b = pack_archive(enc, dec, pred, li, r, e, meta)
    assert a == b


def test_archive_parse_bad_magic_raises() -> None:
    bad = b"XXXX" + b"\x01" + b"\x00" * 34
    with pytest.raises(ValueError, match="bad magic"):
        parse_archive(bad)


def test_archive_parse_truncated_raises() -> None:
    enc, dec, pred, li, r, e, meta, _ = _make_archive_inputs()
    blob = pack_archive(enc, dec, pred, li, r, e, meta)
    with pytest.raises(ValueError, match=r"too short|archive size"):
        parse_archive(blob[:20])


def test_archive_predictive_coding_meta_tag_present() -> None:
    """Z5 archives MUST carry the predictive_coding_world_model_meta tag."""
    enc, dec, pred, li, r, e, meta, _ = _make_archive_inputs()
    blob = pack_archive(
        enc, dec, pred, li, r, e, meta,
        lambda_residual_entropy=2.0,
        predictor_num_layers=3,
        identity_predictor=False,
    )
    parsed = parse_archive(blob)
    assert "predictive_coding_world_model_meta" in parsed.meta
    pc = parsed.meta["predictive_coding_world_model_meta"]
    assert pc["lambda_residual_entropy"] == 2.0
    assert pc["predictor_num_layers"] == 3
    assert pc["identity_predictor"] is False
    assert pc["literature_anchor"] == "Rao-Ballard1999"
    assert pc["staircase_step"] == 3
    assert "predicted_band_lo" not in pc
    assert "predicted_band_hi" not in pc
    verdict = pc["prediction_band_verdict"]
    assert verdict["planning_band"] == [0.155, 0.180]
    assert verdict["valid_for_rank_reward"] is False
    assert verdict["score_claim"] is False
    assert verdict["promotion_eligible"] is False
    assert verdict["ready_for_exact_eval_dispatch"] is False


# ===========================================================================
# (E) Inflate — 4 tests
# ===========================================================================

def test_inflate_main_cli_3_arg_contract() -> None:
    """Per Catalog #146 inflate.py MUST accept 3 positional args."""
    from tac.substrates.z5_predictive_coding_world_model import inflate as inflate_module
    old_argv = sys.argv
    try:
        sys.argv = ["inflate.py"]
        rc = inflate_module.main_cli()
        assert rc == 2
    finally:
        sys.argv = old_argv


def test_inflate_no_scorer_imports() -> None:
    """Per CLAUDE.md 'Strict scorer rule' + Catalog #6: inflate.py must NOT import scorer code."""
    import inspect

    from tac.substrates.z5_predictive_coding_world_model import inflate
    src = inspect.getsource(inflate)
    forbidden = ("posenet", "segnet", "from upstream.modules", "import upstream.modules",
                 "rgb_to_yuv6", "efficientnet", "fastvit", "load_differentiable_scorers")
    src_lower = src.lower()
    for tok in forbidden:
        assert tok not in src_lower, f"forbidden token in inflate.py: {tok!r}"


def test_inflate_one_video_writes_raw(tmp_path: Path) -> None:
    """End-to-end inflate roundtrip on a synthetic Z5PCWM1 archive."""
    from tac.substrates.z5_predictive_coding_world_model.inflate import inflate_one_video
    enc, dec, pred, li, r, e, meta, _ = _make_archive_inputs(num_pairs=2)
    blob = pack_archive(enc, dec, pred, li, r, e, meta)
    out_raw = tmp_path / "out.raw"
    frames = inflate_one_video(blob, out_raw, device="cpu")
    assert frames == 2 * 2, f"expected 4 frames written; got {frames}"
    expected_size = 4 * 874 * 1164 * 3
    assert out_raw.stat().st_size == expected_size


def test_inflate_ambiguous_archive_member_raises(tmp_path: Path) -> None:
    """Both 0.bin and x present → ValueError."""
    from tac.substrates.z5_predictive_coding_world_model.inflate import _read_single_member_archive_bytes
    (tmp_path / "0.bin").write_bytes(b"hello")
    (tmp_path / "x").write_bytes(b"world")
    with pytest.raises(ValueError, match="ambiguous archive"):
        _read_single_member_archive_bytes(tmp_path)


# ===========================================================================
# (F) Score-aware loss — 5 tests
# ===========================================================================

class _StubScorer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        last = pair_btchw[:, -1, ...]
        return torch.nn.functional.interpolate(
            last, size=(384, 512), mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_score_aware_loss_eval_roundtrip_mandatory() -> None:
    seg = _StubScorer()
    pose = _StubScorer()
    loss_fn = PredictiveCodingScoreAwareLoss(seg, pose, PredictiveCodingLossWeights())
    rgb_0 = torch.full((1, 3, 384, 512), 100.0)
    rgb_1 = torch.full((1, 3, 384, 512), 110.0)
    gt_0 = torch.full((1, 3, 384, 512), 95.0)
    gt_1 = torch.full((1, 3, 384, 512), 105.0)
    with pytest.raises(ValueError, match="apply_eval_roundtrip=False is forbidden"):
        loss_fn(
            reconstructed_rgb_0=rgb_0,
            reconstructed_rgb_1=rgb_1,
            gt_rgb_0=gt_0,
            gt_rgb_1=gt_1,
            archive_bytes_proxy=torch.tensor(100_000.0),
            residuals=torch.zeros(5, 8),
            apply_eval_roundtrip=False,
            noise_std=0.0,
        )


def test_score_aware_loss_unit_domain_rgb_refused() -> None:
    seg = _StubScorer()
    pose = _StubScorer()
    loss_fn = PredictiveCodingScoreAwareLoss(seg, pose, PredictiveCodingLossWeights())
    rgb_0 = torch.full((1, 3, 384, 512), 0.5)
    rgb_1 = torch.full((1, 3, 384, 512), 0.6)
    gt_0 = torch.full((1, 3, 384, 512), 95.0)
    gt_1 = torch.full((1, 3, 384, 512), 105.0)
    with pytest.raises(ValueError, match="unit-domain"):
        loss_fn(
            reconstructed_rgb_0=rgb_0,
            reconstructed_rgb_1=rgb_1,
            gt_rgb_0=gt_0,
            gt_rgb_1=gt_1,
            archive_bytes_proxy=torch.tensor(100_000.0),
            residuals=torch.zeros(5, 8),
            noise_std=0.0,
        )


def test_score_aware_loss_negative_lambda_residual_refused() -> None:
    seg = _StubScorer()
    pose = _StubScorer()
    weights = PredictiveCodingLossWeights(lambda_residual_entropy=-1.0)
    loss_fn = PredictiveCodingScoreAwareLoss(seg, pose, weights)
    rgb_0 = torch.full((1, 3, 384, 512), 100.0)
    rgb_1 = torch.full((1, 3, 384, 512), 110.0)
    gt_0 = torch.full((1, 3, 384, 512), 95.0)
    gt_1 = torch.full((1, 3, 384, 512), 105.0)
    with pytest.raises(ValueError, match="lambda_residual_entropy must be >= 0"):
        loss_fn(
            reconstructed_rgb_0=rgb_0,
            reconstructed_rgb_1=rgb_1,
            gt_rgb_0=gt_0,
            gt_rgb_1=gt_1,
            archive_bytes_proxy=torch.tensor(100_000.0),
            residuals=torch.zeros(5, 8),
            noise_std=0.0,
        )


def test_score_aware_loss_residuals_wrong_rank_refused() -> None:
    seg = _StubScorer()
    pose = _StubScorer()
    loss_fn = PredictiveCodingScoreAwareLoss(seg, pose, PredictiveCodingLossWeights())
    rgb_0 = torch.full((1, 3, 384, 512), 100.0)
    rgb_1 = torch.full((1, 3, 384, 512), 110.0)
    gt_0 = torch.full((1, 3, 384, 512), 95.0)
    gt_1 = torch.full((1, 3, 384, 512), 105.0)
    with pytest.raises(ValueError, match="residuals must be 2-D"):
        loss_fn(
            reconstructed_rgb_0=rgb_0,
            reconstructed_rgb_1=rgb_1,
            gt_rgb_0=gt_0,
            gt_rgb_1=gt_1,
            archive_bytes_proxy=torch.tensor(100_000.0),
            residuals=torch.zeros(5),  # 1-D not 2-D
            noise_std=0.0,
        )


def test_score_aware_loss_weights_propagate() -> None:
    """LossWeights dataclass fields propagate into the loss module."""
    seg = _StubScorer()
    pose = _StubScorer()
    weights = PredictiveCodingLossWeights(
        lambda_residual_entropy=3.5,
        beta_seg=42.0,
        alpha_rate=99.0,
    )
    loss_fn = PredictiveCodingScoreAwareLoss(seg, pose, weights)
    assert loss_fn.weights.lambda_residual_entropy == 3.5
    assert loss_fn.weights.beta_seg == 42.0
    assert loss_fn.weights.alpha_rate == 99.0
