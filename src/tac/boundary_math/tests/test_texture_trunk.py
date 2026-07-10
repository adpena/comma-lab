"""Tests for the band-designed per-class stationary texture trunk (T of W=(G, ξ, T), #395).

Covers: band-limit enforcement (construction refuses out-of-band periods; the spectral report proves
EVERY feature is in the measured stem pass-band), deterministic bank (no RNG, reproducible),
placement-by-softmax-mask (class-k texture appears only where soft puts mass on k), annulus
attenuation (monotone, off at power 0), byte accounting (counted = coeffs only), numpy<->MLX parity,
and the zero-mean texture property. Every number is design-authority / NON-PROMOTABLE.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.texture_trunk import (
    STEM_NYQUIST_PERIOD_PX,
    TextureBandSpec,
    TextureTrunkError,
    band_limit_report,
    build_gabor_bank_numpy,
    counted_bytes,
    default_band_spec,
    make_texture_trunk_mlx,
    texture_trunk_numpy_forward,
)

N_CLASSES = 5


# --------------------------------------------------------------------------- #
# Band spec — clause-B: the bank support IS the measured stem pass-band.       #
# --------------------------------------------------------------------------- #
def test_default_spec_feature_count():
    spec = default_band_spec()
    # periods {4,6,8} × orient {0,45,90,135} × phase {cos,sin} = 3*4*2
    assert spec.n_features == 24


def test_spec_refuses_period_below_nyquist():
    # period-2 aliases away below the stride-2 stem Nyquist — must be REFUSED at construction.
    with pytest.raises(TextureTrunkError):
        TextureBandSpec(periods=(2.0, 4.0))


def test_spec_refuses_period_above_band_hi():
    # period-16 reads as flat (palette base's DC job) — refused.
    with pytest.raises(TextureTrunkError):
        TextureBandSpec(periods=(4.0, 16.0), band_hi=8.0)


def test_spec_accepts_custom_in_band():
    spec = TextureBandSpec(periods=(4.0, 5.0, 8.0), orientations_deg=(0.0, 90.0), n_phase=1, band_hi=8.0)
    assert spec.n_features == 3 * 2 * 1


def test_spec_refuses_bad_n_phase():
    with pytest.raises(TextureTrunkError):
        TextureBandSpec(n_phase=3)


def test_spec_refuses_empty_periods():
    with pytest.raises(TextureTrunkError):
        TextureBandSpec(periods=())


def test_stem_nyquist_constant():
    # The band floor is pinned to the MEASURED stem Nyquist (2*stride = 4).
    assert STEM_NYQUIST_PERIOD_PX == 4.0


# --------------------------------------------------------------------------- #
# Band-limit — spectral proof every feature is in-band.                        #
# --------------------------------------------------------------------------- #
def test_band_limit_report_all_in_band_real_gridish():
    # A grid whose dims are multiples of the periods so the FFT bins land cleanly.
    rep = band_limit_report(96, 96, default_band_spec())
    assert rep["all_in_band"] is True
    assert rep["peak_period_min_px"] >= STEM_NYQUIST_PERIOD_PX - 1.0
    assert rep["peak_period_max_px"] <= default_band_spec().band_hi + 1.0


def test_band_limit_nothing_at_period_2():
    # No feature's spectral peak lands at the aliased period-2 (must be >= ~3).
    rep = band_limit_report(96, 96, default_band_spec())
    assert min(rep["per_feature_peak_period_px"]) > 2.5


def test_band_limit_report_feature_count_matches():
    spec = TextureBandSpec(periods=(4.0, 8.0), orientations_deg=(0.0, 90.0), n_phase=2)
    rep = band_limit_report(64, 64, spec)
    assert rep["n_features"] == spec.n_features == 8
    assert len(rep["per_feature_peak_period_px"]) == 8


# --------------------------------------------------------------------------- #
# Deterministic bank.                                                          #
# --------------------------------------------------------------------------- #
def test_bank_shape_and_dtype():
    bank = build_gabor_bank_numpy(16, 32, default_band_spec())
    assert bank.shape == (16 * 32, 24)
    assert bank.dtype == np.float32


def test_bank_deterministic():
    a = build_gabor_bank_numpy(24, 24)
    b = build_gabor_bank_numpy(24, 24)
    assert np.array_equal(a, b)  # no RNG -> bit-identical (rule-118 free table)


def test_bank_zero_mean_large_grid():
    # Texture is a zero-mean perturbation ON TOP of the palette DC; on the real render grid every
    # feature's spatial mean is ~0 (pure sinusoid over many cycles).
    bank = build_gabor_bank_numpy(384, 512)
    assert float(np.abs(bank.mean(axis=0)).max()) < 0.01


def test_bank_refuses_tiny_grid():
    with pytest.raises(TextureTrunkError):
        build_gabor_bank_numpy(1, 1)


def test_bank_values_bounded():
    bank = build_gabor_bank_numpy(20, 20)
    assert float(bank.min()) >= -1.0000001 and float(bank.max()) <= 1.0000001


# --------------------------------------------------------------------------- #
# numpy forward — placement + attenuation.                                     #
# --------------------------------------------------------------------------- #
def _grid_forward(h, w, wtex_setup, soft_setup, **kw):
    spec = default_band_spec()
    bank = build_gabor_bank_numpy(h, w, spec)
    F = spec.n_features
    wtex = np.zeros((F, N_CLASSES, 3), np.float32)
    wtex_setup(wtex)
    soft = np.zeros((h * w, N_CLASSES), np.float32)
    soft_setup(soft)
    return texture_trunk_numpy_forward(bank, wtex, soft, n_classes=N_CLASSES, **kw)


def test_forward_placement_class_region_uses_own_coeffs():
    # class-0-only coefficients in a class-0 region -> nonzero texture.
    def setw(w):
        w[0, 0, :] = 1.0

    def sets(s):
        s[:, 0] = 1.0

    tex = _grid_forward(8, 8, setw, sets)
    assert tex.shape == (64, 3)
    assert float(np.abs(tex).sum()) > 0.0


def test_forward_placement_guard_other_region_zero():
    # class-0-only coefficients but a class-1 region -> the mask kills the class-0 texture.
    def setw(w):
        w[0, 0, :] = 1.0

    def sets(s):
        s[:, 1] = 1.0  # all mass on class 1

    tex = _grid_forward(8, 8, setw, sets)
    assert float(np.abs(tex).sum()) == 0.0


def test_forward_soft_blend_partial_mask():
    # half mass on class 0 -> texture is half the full-mask amplitude.
    def setw(w):
        w[1, 0, :] = 0.7

    def sets_full(s):
        s[:, 0] = 1.0

    def sets_half(s):
        s[:, 0] = 0.5
        s[:, 2] = 0.5

    full = _grid_forward(8, 8, setw, sets_full)
    half = _grid_forward(8, 8, setw, sets_half)
    assert np.allclose(half, 0.5 * full, atol=1e-5)


def test_forward_annulus_attenuation_off_at_power_zero():
    def setw(w):
        w[0, 0, :] = 1.0

    def sets(s):
        s[:, 0] = 0.8
        s[:, 1] = 0.2

    a = _grid_forward(8, 8, setw, sets, annulus_power=0.0)
    # peak 0.8 -> gate (0.8-0.2)/(0.8) = 0.75 < 1, so power>0 attenuates vs power 0.
    b = _grid_forward(8, 8, setw, sets, annulus_power=1.0)
    assert float(np.abs(b).sum()) < float(np.abs(a).sum())


def test_forward_annulus_monotone_in_power():
    def setw(w):
        w[0, 0, :] = 1.0

    def sets(s):
        s[:, 0] = 0.6
        s[:, 1] = 0.4  # peak 0.6, gate<1 -> higher power attenuates more

    p1 = _grid_forward(8, 8, setw, sets, annulus_power=1.0)
    p2 = _grid_forward(8, 8, setw, sets, annulus_power=2.0)
    assert float(np.abs(p2).sum()) <= float(np.abs(p1).sum())


def test_forward_bias_adds_offset():
    spec = default_band_spec()
    bank = build_gabor_bank_numpy(8, 8, spec)
    F = spec.n_features
    wtex = np.zeros((F, N_CLASSES, 3), np.float32)
    soft = np.zeros((64, N_CLASSES), np.float32)
    soft[:, 0] = 1.0
    bias = np.zeros((N_CLASSES, 3), np.float32)
    bias[0, :] = 0.3
    tex = texture_trunk_numpy_forward(bank, wtex, soft, bias=bias, n_classes=N_CLASSES)
    assert np.allclose(tex, 0.3, atol=1e-5)  # zero coeffs -> texture == bias in class-0 region


def test_forward_batched_shape():
    spec = default_band_spec()
    bank = build_gabor_bank_numpy(8, 8, spec)
    F = spec.n_features
    wtex = np.zeros((F, N_CLASSES, 3), np.float32)
    wtex[0, 0, :] = 1.0
    soft = np.zeros((3, 64, N_CLASSES), np.float32)
    soft[:, :, 0] = 1.0
    tex = texture_trunk_numpy_forward(bank, wtex, soft, n_classes=N_CLASSES)
    assert tex.shape == (3, 64, 3)


def test_forward_rejects_bad_wtex_shape():
    bank = build_gabor_bank_numpy(8, 8)
    with pytest.raises(TextureTrunkError):
        texture_trunk_numpy_forward(bank, np.zeros((5, N_CLASSES, 3)), np.zeros((64, N_CLASSES)))


# --------------------------------------------------------------------------- #
# Byte accounting — counted = coefficients only (bank is free).               #
# --------------------------------------------------------------------------- #
def test_counted_bytes_default():
    cb = counted_bytes(default_band_spec(), quant_bits=8)
    # 24*5*3 weights + 5*3 bias = 360 + 15 = 375 coeffs -> 375 bytes at 8-bit.
    assert cb["n_coeff"] == 375
    assert cb["raw_bytes_uncoded"] == 375.0
    assert cb["rate_term_uncoded_S"] == pytest.approx(25.0 * 375.0 / 37_545_489.0)


def test_counted_bytes_scales_with_features():
    small = TextureBandSpec(periods=(4.0,), orientations_deg=(0.0,), n_phase=1)
    cb = counted_bytes(small, quant_bits=8)
    assert cb["n_coeff"] == 1 * N_CLASSES * 3 + N_CLASSES * 3


# --------------------------------------------------------------------------- #
# MLX submodule + numpy parity.                                                #
# --------------------------------------------------------------------------- #
def test_mlx_trunk_builds_and_shapes():
    mx = pytest.importorskip("mlx.core")
    tt = make_texture_trunk_mlx(8, 8, default_band_spec(), coeff_scale=0.05, seed=0)
    soft = mx.array(np.eye(N_CLASSES, dtype=np.float32)[np.random.default_rng(1).integers(0, N_CLASSES, size=64)])
    out = np.asarray(tt(soft))
    assert out.shape == (64, 3)


def test_mlx_bank_is_frozen_not_counted():
    pytest.importorskip("mlx.core")
    tt = make_texture_trunk_mlx(8, 8, default_band_spec(), seed=0)
    # only w_tex + bias are trainable params; the bank buffer is frozen (rule-118 free, not counted).
    pnames = set(dict(tt.trainable_parameters()).keys())
    assert "w_tex" in pnames and "bias" in pnames
    assert "bank_B" not in pnames


def test_mlx_bank_key_has_rule118_suffix():
    # The bank buffer's key MUST end in "_B" so the trainer's byte-close (measure_witness_blob_bytes /
    # _quantize_blob_from_flat / _load_decoder_params) excludes it from the COUNTED archive (rule-118
    # free; regenerated at decode). A rename that drops the suffix would silently COUNT 4.7M floats.
    pytest.importorskip("mlx.core")
    from mlx.utils import tree_flatten
    tt = make_texture_trunk_mlx(8, 8, default_band_spec(), seed=0)
    keys = [k for k, _ in tree_flatten(tt.parameters())]
    bank_keys = [k for k in keys if "bank" in k]
    assert bank_keys and all(k.endswith("_B") for k in bank_keys), bank_keys


def test_mlx_numpy_parity_fp32():
    mx = pytest.importorskip("mlx.core")
    tt = make_texture_trunk_mlx(8, 8, default_band_spec(), coeff_scale=0.05, seed=3)
    soft_np = np.eye(N_CLASSES, dtype=np.float32)[np.random.default_rng(7).integers(0, N_CLASSES, size=64)]
    out_mlx = np.asarray(tt(mx.array(soft_np)))
    out_np = texture_trunk_numpy_forward(
        np.asarray(tt.bank_B), np.asarray(tt.w_tex), soft_np, bias=np.asarray(tt.bias), n_classes=N_CLASSES)
    assert float(np.abs(out_mlx - out_np).max()) < 1e-3  # fp32 (MLX) vs fp64-cast (numpy) agreement


def test_mlx_annulus_matches_numpy():
    mx = pytest.importorskip("mlx.core")
    tt = make_texture_trunk_mlx(8, 8, default_band_spec(), coeff_scale=0.05, annulus_power=1.5, seed=2)
    rng = np.random.default_rng(5)
    soft_np = rng.random((64, N_CLASSES)).astype(np.float32)
    soft_np = soft_np / soft_np.sum(axis=-1, keepdims=True)
    out_mlx = np.asarray(tt(mx.array(soft_np)))
    out_np = texture_trunk_numpy_forward(
        np.asarray(tt.bank_B), np.asarray(tt.w_tex), soft_np, bias=np.asarray(tt.bias),
        n_classes=N_CLASSES, annulus_power=1.5)
    assert float(np.abs(out_mlx - out_np).max()) < 1e-3


def test_mlx_deterministic_init():
    pytest.importorskip("mlx.core")
    a = make_texture_trunk_mlx(8, 8, default_band_spec(), seed=11)
    b = make_texture_trunk_mlx(8, 8, default_band_spec(), seed=11)
    assert np.array_equal(np.asarray(a.w_tex), np.asarray(b.w_tex))
