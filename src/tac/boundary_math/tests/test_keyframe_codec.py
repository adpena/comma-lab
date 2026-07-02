# SPDX-License-Identifier: MIT
"""Unit tests for tac.boundary_math.keyframe_codec (the #202 keyframe rate primitives).

Covers: degradation ops (shape/dtype/monotone info loss), order-0 entropy, exact-invertible
residual coding, ego-warp residual < prev-copy residual on a translated frame, still + temporal
codec byte measurement, class-mean texture-free render, rate accounting. NO torch / NO mlx.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math import keyframe_codec as kc


@pytest.fixture()
def frame():
    rng = np.random.default_rng(7)
    H, W = 120, 160
    xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    base = (np.stack([np.sin(6 * xx), np.cos(5 * yy), 0.5 * (xx + yy)], -1) * 120 + 128)
    return np.clip(base + rng.integers(-8, 8, base.shape), 0, 255).astype(np.uint8)


# ----------------------------- degradation ops ----------------------------- #
@pytest.mark.parametrize("fn", [
    lambda f: kc.resize_roundtrip(f, 80, 60),
    lambda f: kc.downsample_only(f, 80, 60),
    lambda f: kc.gaussian_blur(f, 2.0),
    lambda f: kc.dct_truncate(f, 24, 24),
    lambda f: kc.bitdepth_quantize(f, 4),
])
def test_degradation_shape_dtype(frame, fn):
    out = fn(frame)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3


def test_downsample_actual_size(frame):
    assert kc.downsample_only(frame, 80, 60).shape == (60, 80, 3)


def test_bitdepth_reduces_distinct_levels(frame):
    q = kc.bitdepth_quantize(frame, 3)
    assert len(np.unique(q)) <= 8 * 3  # <= 8 levels/channel


def test_gaussian_blur_identity_at_zero(frame):
    assert np.array_equal(kc.gaussian_blur(frame, 0.0), frame)


def test_dct_truncate_more_keep_is_closer(frame):
    near = kc.dct_truncate(frame, 100, 100).astype(int)
    far = kc.dct_truncate(frame, 8, 8).astype(int)
    assert np.abs(near - frame).mean() < np.abs(far - frame).mean()


# ----------------------------- entropy ----------------------------- #
def test_order0_entropy_bounds(frame):
    h = kc.order0_entropy_bits_per_symbol(frame)
    assert 0.0 <= h <= 8.0


def test_order0_entropy_constant_is_zero():
    assert kc.order0_entropy_bits_per_symbol(np.zeros((10, 10, 3), np.uint8)) == 0.0


def test_order0_entropy_bytes_matches_formula():
    a = np.array([0, 0, 1, 1], np.uint8)  # entropy = 1 bit/sym, 4 syms -> 0.5 bytes
    assert kc.order0_entropy_bytes(a) == pytest.approx(0.5)


# ----------------------------- residual coding ----------------------------- #
def test_residual_wraparound_exact_invertible(frame):
    pred = np.roll(frame, 5, axis=0)
    res = kc.residual_wraparound_u8(frame, pred)
    recon = kc.reconstruct_from_residual(pred, res)
    assert np.array_equal(recon, frame)


def test_ego_warp_residual_beats_prev_copy_on_translation(frame):
    shifted = np.roll(frame, 6, axis=1)  # pure translation -> ego-warp should recover it
    fit = kc.fit_ego_homography(frame, shifted, mode="ecc", ecc_width=128)
    res_ego = kc.ego_warp_residual_u8(frame, shifted, fit)
    res_prev = kc.residual_wraparound_u8(shifted, frame)
    # ego-warp residual entropy strictly lower (motion compensated away)
    assert kc.order0_entropy_bits_per_symbol(res_ego) < kc.order0_entropy_bits_per_symbol(res_prev)


def test_ego_warp_residual_exact_reconstruction(frame):
    shifted = np.roll(frame, 4, axis=1)
    fit = kc.fit_ego_homography(frame, shifted, mode="ecc", ecc_width=128)
    pred = kc.warp_by_homography(frame, fit.H)
    res = kc.ego_warp_residual_u8(frame, shifted, fit)
    assert np.array_equal(kc.reconstruct_from_residual(pred, res), shifted)


# ----------------------------- still codecs ----------------------------- #
def test_still_codec_bytes_positive(frame):
    assert kc.png_bytes(frame) > 0
    assert kc.webp_bytes(frame, 50) > 0
    assert kc.webp_bytes(frame, 100, lossless=True) > kc.webp_bytes(frame, 30)
    assert kc.jpeg2000_bytes(frame) > 0


def test_zlib_brotli_positive(frame):
    assert kc.zlib_bytes(frame) > 0
    assert kc.brotli_bytes(frame) > 0


# ----------------------------- temporal codecs ----------------------------- #
def test_temporal_codecs_encode_decode():
    frames = [np.roll(np.full((64, 96, 3), 120, np.uint8) + (np.arange(96)[None, :, None] % 40).astype(np.uint8),
                       i, axis=1) for i in range(6)]
    for codec in ("x265", "svtav1", "vp9"):
        nb, dec = kc.encode_decode_video_stream(frames, codec=codec, crf=40, gop=9999, bframes=0)
        assert nb > 0
        assert len(dec) == len(frames)
        assert dec[0].shape == frames[0].shape


def test_temporal_odd_dims_padded():
    # 65x97 is odd both ways -> must be handled (even-padding), decode cropped back
    frames = [np.random.default_rng(i).integers(0, 255, (65, 97, 3), np.uint8) for i in range(4)]
    nb, dec = kc.encode_decode_video_stream(frames, codec="svtav1", crf=40)
    assert nb > 0 and dec[0].shape == (65, 97, 3)


# ----------------------------- class-mean render ----------------------------- #
def test_class_mean_render_removes_texture(frame):
    am = (np.floor(np.linspace(0, 1, frame.shape[1])[None, :] * 4).astype(np.int64) % 5) \
        * np.ones((frame.shape[0], 1), np.int64)
    cm = kc.class_mean_render(frame, am, n_classes=5)
    assert cm.shape == frame.shape and cm.dtype == np.uint8
    # within a class column-band, class-mean render is piecewise-constant -> lower entropy than source
    assert kc.order0_entropy_bits_per_symbol(cm) < kc.order0_entropy_bits_per_symbol(frame)


def test_class_mean_lowfreq_residual_closer(frame):
    am = (np.floor(np.linspace(0, 1, frame.shape[1])[None, :] * 4).astype(np.int64) % 5) \
        * np.ones((frame.shape[0], 1), np.int64)
    flat = kc.class_mean_render(frame, am, residual_lowfreq=0).astype(int)
    withlf = kc.class_mean_render(frame, am, residual_lowfreq=32).astype(int)
    assert np.abs(withlf - frame).mean() < np.abs(flat - frame).mean()


def test_upsample_argmax_label_safe():
    am = np.array([[0, 1], [2, 3]], np.int64)
    up = kc.upsample_argmax_nearest(am, 4, 4)
    assert up.shape == (4, 4)
    assert set(np.unique(up).tolist()) <= {0, 1, 2, 3}


# ----------------------------- rate accounting ----------------------------- #
def test_rate_from_bytes_matches_contest_formula():
    assert kc.rate_from_bytes(37_545_489) == pytest.approx(25.0)
    assert kc.rate_from_bytes(0) == 0.0
