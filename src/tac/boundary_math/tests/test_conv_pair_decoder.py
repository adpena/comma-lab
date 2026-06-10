# SPDX-License-Identifier: MIT
"""Behavior tests for the lever-C conv per-pair decoder + joint-objective mechanism (task #62).

NO-FAKE (class 2 + class 8): these assert the conv decoder ACTUALLY does the work its name claims:
 - the numpy conv/pixelshuffle/bilinear primitives match torch to float precision (portability);
 - the full numpy forward depends on (pair latent, x, y) — a constant-frame stub FAILS;
 - the byte cost is the brotli of the ACTUAL quantized weights+latents (varies with capacity);
 - quant/dequant round-trips within quant error;
 - the per-pair output varies; a SHARED decoder + distinct latents produces distinct frames;
 - the null-space (margin-free-budget) recon weight ACTUALLY redistributes (mean-1, boundary-heavy);
 - the Jacobian saliency weight ACTUALLY changes the gradient vs uniform (load-bearing).

The exact-scorer d_seg/d_pose reduction (the score-effect) is the trainer's job (its result JSON +
the viability-smoke verdict); those need the scorer + GT video and are not unit-tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.conv_pair_decoder import (
    CAMERA_H,
    CAMERA_W,
    ConvDecoderByteAccount,
    ConvDecoderConfig,
    _bilinear_resize,
    _conv3x3,
    _pixel_shuffle,
    decoder_frame,
    decoder_param_count,
    dequantize_params,
    measure_decoder_bytes,
    numpy_reference_forward,
    quantize_params,
)

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")


def _random_weights(cfg: ConvDecoderConfig, seed: int = 0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Small random (but bounded) decoder weights matching the cfg shapes + per-pair latents."""

    rng = np.random.default_rng(seed)
    w: dict[str, np.ndarray] = {}
    seed_out = cfg.seed_ch * cfg.seed_h * cfg.seed_w
    w["seed.weight"] = (rng.standard_normal((seed_out, cfg.latent_dim)) * 0.1).astype(np.float32)
    w["seed.bias"] = (rng.standard_normal(seed_out) * 0.1).astype(np.float32)
    in_ch = cfg.seed_ch
    for i, out_ch in enumerate(cfg.stage_channels):
        w[f"stage{i}.weight"] = (rng.standard_normal((out_ch * 4, in_ch, 3, 3)) * 0.1).astype(np.float32)
        w[f"stage{i}.bias"] = (rng.standard_normal(out_ch * 4) * 0.1).astype(np.float32)
        w[f"stage{i}.skip"] = (rng.standard_normal((out_ch, in_ch, 1, 1)) * 0.1).astype(np.float32)
        w[f"stage{i}.skip_bias"] = (rng.standard_normal(out_ch) * 0.1).astype(np.float32)
        in_ch = out_ch
    w["out.weight"] = (rng.standard_normal((cfg.n_channels, in_ch, 3, 3)) * 0.1).astype(np.float32)
    w["out.bias"] = (rng.standard_normal(cfg.n_channels) * 0.1).astype(np.float32)
    latents = (rng.standard_normal((cfg.num_pairs, cfg.latent_dim)) * 0.3).astype(np.float32)
    return w, latents


# --------------------------------------------------------------------------- #
# numpy primitive parity vs torch (the portability contract)                   #
# --------------------------------------------------------------------------- #
def test_conv3x3_matches_torch():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 6, 8))
    w = rng.standard_normal((12, 4, 3, 3))
    b = rng.standard_normal(12)
    np_out = _conv3x3(x, w, b)
    t_out = F.conv2d(torch.tensor(x)[None], torch.tensor(w), torch.tensor(b), padding=1)[0].numpy()
    assert np.abs(np_out - t_out).max() < 1e-9


def test_pixel_shuffle_matches_torch():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((12, 6, 8))
    np_ps = _pixel_shuffle(x, 2)
    t_ps = F.pixel_shuffle(torch.tensor(x)[None], 2)[0].numpy()
    assert np.abs(np_ps - t_ps).max() == 0.0
    assert np_ps.shape == (3, 12, 16)


def test_bilinear_resize_matches_torch():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 12, 16))
    np_b = _bilinear_resize(x, 24, 32)
    t_b = F.interpolate(torch.tensor(x)[None], size=(24, 32), mode="bilinear", align_corners=False)[0].numpy()
    assert np.abs(np_b - t_b).max() < 1e-9


def test_bilinear_resize_identity_when_same_size():
    x = np.random.default_rng(4).standard_normal((3, 10, 10))
    out = _bilinear_resize(x, 10, 10)
    assert out is x  # identity short-circuit, no copy


# --------------------------------------------------------------------------- #
# full forward — shape, variation, load-bearing                                 #
# --------------------------------------------------------------------------- #
def test_decoder_frame_shape_and_dtype():
    cfg = ConvDecoderConfig(num_pairs=3, latent_dim=8, seed_ch=8, stage_channels=(8, 8, 8, 8))
    w, lat = _random_weights(cfg)
    f = decoder_frame(w, cfg, lat, 0, out_h=128, out_w=160)
    assert f.shape == (128, 160, 3)
    assert f.dtype == np.uint8


def test_decoder_frame_varies_across_pixels():
    """A real conv decode is NOT a flat constant frame (NO-FAKE: constant stub would be flat)."""

    cfg = ConvDecoderConfig(num_pairs=2, latent_dim=8, seed_ch=12, stage_channels=(12, 8, 8, 8))
    w, lat = _random_weights(cfg, seed=5)
    f = decoder_frame(w, cfg, lat, 0, out_h=96, out_w=128)
    assert float(f.std()) > 0.5  # the frame has spatial structure, not a constant


def test_decoder_frame_varies_across_pairs():
    """Distinct per-pair latents → distinct frames (the per-pair latent is load-bearing)."""

    cfg = ConvDecoderConfig(num_pairs=3, latent_dim=12, seed_ch=16, stage_channels=(16, 12, 8, 8))
    w, lat = _random_weights(cfg, seed=6)
    f0 = decoder_frame(w, cfg, lat, 0, out_h=96, out_w=128)
    f1 = decoder_frame(w, cfg, lat, 1, out_h=96, out_w=128)
    assert float(np.mean(f0 != f1)) > 0.05  # pairs genuinely differ


def test_zero_latents_collapse_pair_distinction():
    """NO-FAKE control: if all latents are identical, all pairs decode to the SAME frame.

    Proves the per-pair frame difference COMES FROM the latent (not a hidden per-pair table).
    """

    cfg = ConvDecoderConfig(num_pairs=3, latent_dim=8, seed_ch=12, stage_channels=(12, 8, 8, 8))
    w, lat = _random_weights(cfg, seed=7)
    lat[:] = lat[0]  # force all latents equal
    f0 = decoder_frame(w, cfg, lat, 0, out_h=96, out_w=128)
    f1 = decoder_frame(w, cfg, lat, 2, out_h=96, out_w=128)
    assert np.array_equal(f0, f1)  # identical latents → identical frames


# --------------------------------------------------------------------------- #
# byte accounting — tracks capacity, honest brotli                              #
# --------------------------------------------------------------------------- #
def test_byte_account_total_is_sum():
    cfg = ConvDecoderConfig(num_pairs=8, latent_dim=16, seed_ch=16, stage_channels=(16, 12, 8, 8))
    w, lat = _random_weights(cfg)
    ba = measure_decoder_bytes(w, lat, cfg)
    assert isinstance(ba, ConvDecoderByteAccount)
    assert ba.total_bytes == ba.weight_bytes + ba.latent_bytes + ba.scale_bytes
    assert ba.weight_bytes > 0 and ba.latent_bytes > 0


def test_byte_account_tracks_capacity():
    """A bigger decoder costs more bytes (capacity is load-bearing on the rate)."""

    small = ConvDecoderConfig(num_pairs=8, latent_dim=8, seed_ch=12, stage_channels=(12, 8, 8, 8))
    big = ConvDecoderConfig(num_pairs=8, latent_dim=24, seed_ch=48, stage_channels=(48, 32, 24, 16))
    ws, ls = _random_weights(small)
    wb, lb = _random_weights(big)
    bs = measure_decoder_bytes(ws, ls, small)
    bb = measure_decoder_bytes(wb, lb, big)
    assert bb.total_bytes > bs.total_bytes


def test_latent_bytes_track_num_pairs():
    """More pairs → more per-pair latent bytes (the per-pair tail scales with data)."""

    cfg8 = ConvDecoderConfig(num_pairs=8, latent_dim=16, seed_ch=16, stage_channels=(16, 12, 8, 8))
    cfg600 = ConvDecoderConfig(num_pairs=600, latent_dim=16, seed_ch=16, stage_channels=(16, 12, 8, 8))
    w8, l8 = _random_weights(cfg8)
    w600, l600 = _random_weights(cfg600)
    b8 = measure_decoder_bytes(w8, l8, cfg8)
    b600 = measure_decoder_bytes(w600, l600, cfg600)
    assert b600.latent_bytes > b8.latent_bytes
    # the shared decoder-weight cost is (near) constant regardless of pair count.
    assert abs(b600.weight_bytes - b8.weight_bytes) <= max(b8.weight_bytes, 1)


def test_param_count_matches_weight_shapes():
    cfg = ConvDecoderConfig(num_pairs=5, latent_dim=10, seed_ch=14, stage_channels=(14, 10, 8, 6))
    w, lat = _random_weights(cfg)
    actual = sum(int(np.asarray(v).size) for v in w.values()) + int(lat.size)
    assert decoder_param_count(cfg) == actual


# --------------------------------------------------------------------------- #
# quant round-trip                                                              #
# --------------------------------------------------------------------------- #
def test_quant_dequant_round_trip_within_error():
    cfg = ConvDecoderConfig(num_pairs=3, latent_dim=8, seed_ch=8, stage_channels=(8, 8))
    w, _ = _random_weights(cfg)
    codes, scales = quantize_params(w, bits=8)
    deq = dequantize_params(codes, scales)
    for k in w:
        amax = float(np.max(np.abs(w[k]))) + 1e-12
        step = amax / 127.0
        assert np.max(np.abs(deq[k] - w[k])) <= step * 1.001  # within one quant step


def test_quant_codes_within_int8_range():
    cfg = ConvDecoderConfig(num_pairs=2, latent_dim=6, seed_ch=8, stage_channels=(8, 8))
    w, _ = _random_weights(cfg)
    codes, _ = quantize_params(w, bits=8)
    for c in codes.values():
        assert int(np.max(np.abs(c))) <= 127


# --------------------------------------------------------------------------- #
# config / final resolution invariants                                          #
# --------------------------------------------------------------------------- #
def test_final_hw_lifts_seed_by_two_per_stage():
    cfg = ConvDecoderConfig(num_pairs=1, seed_h=6, seed_w=8, stage_channels=(8, 8, 8, 8, 8))
    fh, fw = cfg.final_hw()
    assert (fh, fw) == (6 * 32, 8 * 32)  # 5 PixelShuffle(2) stages = 32x


def test_decoder_frame_defaults_to_camera_resolution():
    cfg = ConvDecoderConfig(num_pairs=2, latent_dim=8, seed_ch=8, stage_channels=(8, 8, 8, 8, 8))
    w, lat = _random_weights(cfg)
    f = decoder_frame(w, cfg, lat, 0)
    assert f.shape == (CAMERA_H, CAMERA_W, 3)


def test_numpy_reference_forward_returns_block_resolution():
    cfg = ConvDecoderConfig(num_pairs=2, latent_dim=8, seed_ch=8, stage_channels=(8, 8, 8))
    w, lat = _random_weights(cfg)
    out = numpy_reference_forward(w, cfg, lat[0])
    fh, fw = cfg.final_hw()
    assert out.shape == (3, fh, fw)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 255.0  # sigmoid*255


# --------------------------------------------------------------------------- #
# torch <-> numpy parity (the inflate-time portability contract)                #
# --------------------------------------------------------------------------- #
def test_torch_numpy_forward_parity_within_1lsb():
    """The numpy inflate-time decode reproduces the torch training forward within 1 LSB."""

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
    from lever_c_train_conv_pair_decoder import TorchConvPairDecoder, _to_camera

    torch.manual_seed(0)
    cfg = ConvDecoderConfig(num_pairs=3, latent_dim=12, seed_ch=20, stage_channels=(20, 16, 12, 8))
    m = TorchConvPairDecoder(cfg).eval()
    w, lat = m.numpy_params()
    for j in range(3):
        with torch.inference_mode():
            tr = _to_camera(m(j)).permute(1, 2, 0)
            tr = torch.clamp(torch.round(tr), 0, 255).numpy().astype(np.uint8)
        npf = decoder_frame(w, cfg, lat, j, CAMERA_H, CAMERA_W)
        within = float(np.mean(np.abs(tr.astype(np.int32) - npf.astype(np.int32)) <= 1))
        assert within >= 0.99, f"pair {j} parity {within:.4f} < 0.99"


# --------------------------------------------------------------------------- #
# the joint-objective mechanism (null-space + Jacobian weighting load-bearing)  #
# --------------------------------------------------------------------------- #
def test_margin_free_budget_weight_redistributes_to_boundary():
    """The null-space recon weight (#52 margin polytope) is boundary-heavy + mean-1 (redistributes)."""

    from tac.boundary_math.margin_polytope import free_budget_from_margin_jacobian

    # a margin field with a clear boundary band (small margin) and interior (large margin).
    margin = np.full((32, 32), 5.0)
    margin[14:18, :] = 0.05  # a horizontal boundary band
    fb = free_budget_from_margin_jacobian(margin, free_quantile=0.5)
    b = fb.budget
    seg_floor = 0.05
    w = 1.0 / (seg_floor + b / (b.max() + 1e-8))
    w = w / w.mean()
    assert abs(float(w.mean()) - 1.0) < 1e-5  # mean-1 (redistribute, not rescale)
    # the boundary band must weigh MORE than the interior (boundary protection).
    assert float(w[14:18, :].mean()) > float(w[:10, :].mean())


def test_jacobian_weight_changes_gradient_vs_uniform():
    """The Jacobian saliency weight ACTUALLY redistributes the recon gradient (load-bearing).

    With a non-uniform saliency, the weighted recon-MSE gradient differs from the uniform-weighted
    one. Replacing the saliency with a uniform field recovers the dense (uniform) gradient.
    """

    from tac.boundary_math.posenet_jacobian_saliency import (
        PixelSaliencyField,
        identity_weight_map,
        saliency_to_weight_map,
    )

    rng = np.random.default_rng(12)
    sal = rng.random((16, 16)).astype(np.float32)
    sal[4:8, 4:8] = 10.0  # a high-saliency patch
    field = PixelSaliencyField(saliency=sal, h=16, w=16, frame_slot=1,
                               compute_path="cpu_torch", nonzero_fraction=1.0, max_value=float(sal.max()))
    wmap = saliency_to_weight_map(field, floor=0.05, gamma=1.0, normalize=True)
    uniform = identity_weight_map(16, 16)

    err = rng.standard_normal((16, 16)).astype(np.float32) ** 2  # per-pixel recon error
    g_weighted = (err * wmap).mean()
    g_uniform = (err * uniform).mean()
    # the weighted loss differs from uniform (the field is load-bearing).
    assert abs(float(g_weighted) - float(g_uniform)) > 1e-4
    # and the high-saliency patch carries more weight than the mean.
    assert float(wmap[4:8, 4:8].mean()) > float(wmap.mean())
