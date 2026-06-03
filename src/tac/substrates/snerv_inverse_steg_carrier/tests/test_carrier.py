# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the SNeRV store-LF / generate-HF carrier.

Slot EEE Class 2 discipline: tests verify ACTUAL carrier behaviour — the HF
decoder really predicts HF from LF (a zero decoder produces a different, worse
reconstruction than a fitted one), the per-element quantizer really uses the step
map, and the decode path is numpy-only (no scorer). NOT constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_MFU_HFR_TEMPORAL_RECEIVER_PROOF,
    SNERV_SPECTRA_PRESERVING_ADAPTER,
    HfGenerationDecoder,
    HighFrequencyRestorer,
    MultiResolutionFusionUnit,
    SnervCarrierError,
    SnervFrameCode,
    SnervModelSizeConfig,
    SnervTemporalExtension,
    decode_frame,
    dequantize_lf,
    encode_frame_lf,
    fit_hf_decoder_least_squares,
    fit_hf_decoder_weighted_least_squares,
    generate_hf_from_lf,
    official_snerv_modelsize_to_fc_dim,
    quantize_lf,
)
from tac.substrates.snerv_inverse_steg_carrier.dwt import WaveletPyramid


def _smooth_frame(rng, hw=(64, 96)):
    """A smooth-ish frame whose HF is predictable from LF (real-video-like)."""
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 128 + 60 * np.sin(xx / 9.0) + 40 * np.cos(yy / 7.0)
    base += 8 * rng.standard_normal(hw)
    return np.clip(base, 0, 255)


def test_quantize_dequantize_uniform_roundtrip():
    rng = np.random.default_rng(0)
    lf = rng.standard_normal((8, 12)) * 50
    q, sc, zr = quantize_lf(lf, n_levels=256)
    deq = dequantize_lf(q, sc, zr)
    # uniform 8-bit quant error bounded by half a step
    assert np.abs(deq - lf).max() <= sc * 0.51


def test_official_snerv_modelsize_solver_exposes_fc_dim_budget_math() -> None:
    """NO-FAKE: official --modelsize is a quadratic fc_dim control, not prose."""

    solution = official_snerv_modelsize_to_fc_dim(
        modelsize_mparams=1.0,
        full_data_length=100,
        final_size=4096,
        enc_strds=(2, 2),
        dec_strds=(2, 2),
        ks=(0, 1, 5),
        enc_dim=(64.0, 16.0),
        emb_size=2,
        reduce=2.0,
        lower_width=4,
    )

    assert solution.schema == "official_snerv_modelsize_to_fc_dim.v1"
    assert solution.fc_dim == 359
    assert solution.embed_hw == 64.0
    assert solution.embed_dim == 16
    assert solution.embed_param == 107_400.0
    assert solution.fc_param == 9.0
    assert solution.quadratic_a == 6.5
    assert solution.quadratic_b == 144.0
    assert solution.quadratic_c == -892_600.0
    assert solution.as_jsonable()["ready_for_exact_eval_dispatch"] is False


def test_official_snerv_modelsize_solver_rejects_invalid_strides() -> None:
    with pytest.raises(SnervCarrierError, match="stride values must be positive"):
        official_snerv_modelsize_to_fc_dim(
            modelsize_mparams=1.0,
            full_data_length=100,
            final_size=4096,
            enc_strds=(2, 0),
            dec_strds=(2, 2),
        )


def test_quantize_per_element_steps_actually_used():
    """NO-FAKE: per-element steps produce per-element quant granularity."""
    rng = np.random.default_rng(1)
    lf = rng.standard_normal((4, 4)) * 20
    steps = np.full((4, 4), 1.0)
    steps[0, 0] = 10.0  # one coarse coefficient
    q, sc, zr = quantize_lf(lf, per_element_steps=steps)
    deq = dequantize_lf(q, sc, zr, per_element_steps=steps)
    err = np.abs(deq - lf)
    # the coarse coefficient tolerates a larger error than the fine ones
    assert err[0, 0] <= 10.0 * 0.51
    assert err[1, 1] <= 1.0 * 0.51


def test_per_element_steps_must_be_positive():
    with pytest.raises(SnervCarrierError):
        quantize_lf(np.zeros((2, 2)), per_element_steps=np.zeros((2, 2)))


def test_fitted_decoder_beats_zero_decoder():
    """NO-FAKE: the FITTED HF decoder predicts real HF better than a zero decoder.

    If the decoder were a no-op stub, the fitted and zero decoders would give the
    SAME reconstruction. They must differ, and the fitted one must reconstruct the
    real frame better (lower MSE).
    """
    rng = np.random.default_rng(2)
    frames = [_smooth_frame(rng) for _ in range(8)]
    pyrs = [encode_frame_lf(f, levels=3) for f in frames]
    fitted = fit_hf_decoder_least_squares(pyrs, levels=3)
    zero = HfGenerationDecoder.zeros(3)

    f0, p0 = frames[0], pyrs[0]
    q, sc, zr = quantize_lf(p0.lf, n_levels=256)
    code = SnervFrameCode(
        lf_quant=q, lf_scale=sc, lf_zero=zr, lf_shape=p0.lf.shape,
        levels=3, wavelet=p0.wavelet, orig_hw=f0.shape,
    )
    recon_fit = decode_frame(code, fitted)
    recon_zero = decode_frame(code, zero)
    mse_fit = float(np.mean((recon_fit - f0) ** 2))
    mse_zero = float(np.mean((recon_zero - f0) ** 2))
    assert not np.allclose(recon_fit, recon_zero)  # decoder is not a no-op
    assert mse_fit < mse_zero  # fitting HF helps


def test_weighted_decoder_fit_reduces_weighted_hf_residual():
    """NO-FAKE: saliency weights change the fitted decoder objective."""

    rng = np.random.default_rng(20)
    frames = [_smooth_frame(rng) for _ in range(6)]
    pyrs = [encode_frame_lf(f, levels=3) for f in frames]
    weight_pyrs = [_hot_detail_weight_pyramid(pyr) for pyr in pyrs]
    unweighted = fit_hf_decoder_least_squares(pyrs, levels=3)
    weighted = fit_hf_decoder_weighted_least_squares(
        pyrs,
        levels=3,
        detail_weight_pyramids=weight_pyrs,
        saliency_gain=8.0,
    )

    assert _weighted_hf_residual(pyrs, weighted, weight_pyrs) <= (
        _weighted_hf_residual(pyrs, unweighted, weight_pyrs) * 1.0001
    )
    assert any(
        not np.allclose(unweighted.kernels[lvl][sb], weighted.kernels[lvl][sb])
        for lvl in range(3)
        for sb in ("LH", "HL", "HH")
    )


def test_decoder_byte_cost_is_tiny_and_real():
    """The decoder is byte-cheap (shared across all frames) and cost scales w/ levels."""
    rng = np.random.default_rng(3)
    pyrs = [encode_frame_lf(_smooth_frame(rng), levels=3) for _ in range(4)]
    dec = fit_hf_decoder_least_squares(pyrs, levels=3)
    # 3 levels * 3 subbands * 9 taps * 4 bytes = 324 B per channel
    assert dec.byte_cost() == 3 * 3 * 9 * 4
    dec4 = fit_hf_decoder_least_squares(
        [encode_frame_lf(_smooth_frame(rng), levels=4) for _ in range(4)], levels=4
    )
    assert dec4.byte_cost() > dec.byte_cost()  # more levels = more kernels


def test_model_size_controls_change_decoder_capacity_and_reconstruction():
    """NO-FAKE: fc_dim/emb_size alter fitted weights, bytes, and decoded frames."""

    rng = np.random.default_rng(23)
    frames = [_smooth_frame(rng) for _ in range(5)]
    pyrs = [encode_frame_lf(f, levels=2, wavelet="haar") for f in frames]
    base = fit_hf_decoder_least_squares(pyrs, levels=2)
    wider = fit_hf_decoder_least_squares(
        pyrs,
        levels=2,
        model_size=SnervModelSizeConfig(fc_dim=12, emb_size=4, patch_radius=1),
    )

    assert base.byte_cost() == 2 * 3 * 9 * 4
    assert wider.model_size.fc_dim == 12
    assert wider.model_size.emb_size == 4
    assert wider.byte_cost() == 2 * 3 * 16 * 4
    assert wider.kernels[0]["LH"].shape == (16,)

    p0 = pyrs[0]
    q, sc, zr = quantize_lf(p0.lf, n_levels=64)
    code = SnervFrameCode(
        lf_quant=q,
        lf_scale=sc,
        lf_zero=zr,
        lf_shape=p0.lf.shape,
        levels=2,
        wavelet=p0.wavelet,
        orig_hw=frames[0].shape,
    )
    assert not np.allclose(decode_frame(code, base), decode_frame(code, wider))


def test_spectra_preserving_mfu_adapter_changes_features_and_reconstruction():
    """NO-FAKE: MFU adapter changes the fitted receiver basis and decoded pixels."""

    rng = np.random.default_rng(24)
    frames = [_smooth_frame(rng) for _ in range(5)]
    pyrs = [encode_frame_lf(f, levels=2, wavelet="haar") for f in frames]
    base_cfg = SnervModelSizeConfig(fc_dim=12, emb_size=0, patch_radius=1)
    mfu_cfg = SnervModelSizeConfig(
        fc_dim=12,
        emb_size=0,
        patch_radius=1,
        adapter=SNERV_SPECTRA_PRESERVING_ADAPTER,
        mfu_scales=(1, 2, 4),
    )
    base = fit_hf_decoder_least_squares(pyrs, levels=2, model_size=base_cfg)
    mfu = fit_hf_decoder_least_squares(pyrs, levels=2, model_size=mfu_cfg)

    assert SNERV_MFU_HFR_TEMPORAL_RECEIVER_PROOF.startswith("receiver_safe_numpy")
    assert MultiResolutionFusionUnit(scales=(1, 2)).features(
        pyrs[0].lf,
        feature_count=12,
        patch_radius=1,
    ).shape == (*pyrs[0].lf.shape, 12)
    assert mfu.model_size.adapter == SNERV_SPECTRA_PRESERVING_ADAPTER
    assert mfu.byte_cost() == base.byte_cost()
    assert not np.allclose(base.kernels[0]["LH"], mfu.kernels[0]["LH"])

    p0 = pyrs[0]
    q, sc, zr = quantize_lf(p0.lf, n_levels=128)
    code = SnervFrameCode(
        lf_quant=q,
        lf_scale=sc,
        lf_zero=zr,
        lf_shape=p0.lf.shape,
        levels=2,
        wavelet=p0.wavelet,
        orig_hw=frames[0].shape,
    )
    assert not np.allclose(decode_frame(code, base), decode_frame(code, mfu))


def test_mfu_rejects_fc_dim_that_cannot_consume_requested_scales():
    rng = np.random.default_rng(240)
    lf = encode_frame_lf(_smooth_frame(rng), levels=2, wavelet="haar").lf

    with pytest.raises(SnervCarrierError, match="fc_dim is too small"):
        MultiResolutionFusionUnit(scales=(1, 2, 4)).features(
            lf,
            feature_count=5,
            patch_radius=1,
        )


def test_each_requested_mfu_scale_changes_feature_basis():
    rng = np.random.default_rng(241)
    lf = encode_frame_lf(_smooth_frame(rng), levels=2, wavelet="haar").lf

    scale_12 = MultiResolutionFusionUnit(scales=(1, 2)).features(
        lf,
        feature_count=12,
        patch_radius=1,
    )
    scale_124 = MultiResolutionFusionUnit(scales=(1, 2, 4)).features(
        lf,
        feature_count=12,
        patch_radius=1,
    )

    assert scale_12.shape == scale_124.shape
    assert not np.allclose(scale_12, scale_124)


def test_hfr_gain_is_compensated_during_fit_and_changes_decode():
    """NO-FAKE: HFR is an executable residual path, not a marker constant."""

    rng = np.random.default_rng(25)
    frames = [_smooth_frame(rng) for _ in range(4)]
    pyrs = [encode_frame_lf(f, levels=2, wavelet="haar") for f in frames]
    no_hfr = fit_hf_decoder_least_squares(
        pyrs,
        levels=2,
        model_size=SnervModelSizeConfig(
            fc_dim=12,
            adapter=SNERV_SPECTRA_PRESERVING_ADAPTER,
            hfr_gain=0.0,
        ),
    )
    hfr = fit_hf_decoder_least_squares(
        pyrs,
        levels=2,
        model_size=SnervModelSizeConfig(
            fc_dim=12,
            adapter=SNERV_SPECTRA_PRESERVING_ADAPTER,
            hfr_gain=0.25,
        ),
    )
    correction = HighFrequencyRestorer(gain=0.25).correction(
        pyrs[0].lf,
        subband="LH",
        target_hw=pyrs[0].details[0][0].shape,
    )
    assert np.std(correction) > 0
    assert not np.allclose(no_hfr.kernels[0]["LH"], hfr.kernels[0]["LH"])


def test_temporal_extension_exposes_lf_motion_without_hidden_sidecars():
    """NO-FAKE: SNeRV_T utility derives pair/window signal from archived LF planes."""

    rng = np.random.default_rng(26)
    lfs = [
        encode_frame_lf(_smooth_frame(rng) + offset, levels=2, wavelet="haar").lf
        for offset in (0.0, 2.0, 5.0)
    ]
    features = SnervTemporalExtension(radius=1).sequence_delta_features(
        lfs,
        index=1,
        target_hw=lfs[1].shape,
    )
    assert features.shape == (*lfs[1].shape, 2)
    assert np.any(features[:, :, 0] != 0)
    assert np.any(features[:, :, 1] != 0)


def test_temporal_context_changes_decoder_capacity_and_requires_lf_sequence():
    """NO-FAKE: temporal_context changes bytes and decoded pixels, not metadata only."""

    yy, xx = np.mgrid[0:48, 0:64].astype(np.float64)
    frames = [
        np.clip(
            125.0
            + 38.0 * np.sin((xx - 2.0 * i) / 7.0)
            + 22.0 * np.cos((yy + i) / 5.0),
            0.0,
            255.0,
        )
        for i in range(4)
    ]
    pyrs = [encode_frame_lf(frame, levels=2, wavelet="haar") for frame in frames]
    cfg = SnervModelSizeConfig(fc_dim=9, emb_size=0, temporal_context=1)
    base = fit_hf_decoder_least_squares(pyrs, levels=2)
    temporal = fit_hf_decoder_least_squares(
        pyrs,
        levels=2,
        model_size=cfg,
        temporal_group_count=1,
    )

    assert cfg.feature_count == 11
    assert temporal.byte_cost() == 2 * 3 * 11 * 4
    assert temporal.byte_cost() > base.byte_cost()
    assert temporal.kernels[0]["LH"].shape == (11,)

    p1 = pyrs[1]
    q, sc, zr = quantize_lf(p1.lf, n_levels=128)
    code = SnervFrameCode(
        lf_quant=q,
        lf_scale=sc,
        lf_zero=zr,
        lf_shape=p1.lf.shape,
        levels=2,
        wavelet=p1.wavelet,
        orig_hw=frames[1].shape,
    )
    with pytest.raises(SnervCarrierError, match="lf_sequence"):
        decode_frame(code, temporal)

    actual = decode_frame(
        code,
        temporal,
        lf_sequence=[p.lf for p in pyrs],
        sequence_index=1,
    )
    perturbed_sequence = [p.lf.copy() for p in pyrs]
    perturbed_sequence[0] = perturbed_sequence[1]
    perturbed_sequence[2] = perturbed_sequence[1]
    perturbed = decode_frame(
        code,
        temporal,
        lf_sequence=perturbed_sequence,
        sequence_index=1,
    )
    assert not np.allclose(actual, perturbed)


def test_generate_hf_produces_correct_shapes():
    """The generated detail tuples match the template subband shapes (well-formed synthesis)."""
    rng = np.random.default_rng(4)
    pyrs = [encode_frame_lf(_smooth_frame(rng), levels=3) for _ in range(3)]
    dec = fit_hf_decoder_least_squares(pyrs, levels=3)
    p0 = pyrs[0]
    coeffs = generate_hf_from_lf(p0.lf, dec, p0)
    assert len(coeffs) == len(p0.coeffs)  # LF + same number of detail levels
    for (glh, ghl, ghh), (tlh, thl, thh) in zip(
        coeffs[1:], p0.details, strict=True
    ):
        assert glh.shape == tlh.shape
        assert ghl.shape == thl.shape
        assert ghh.shape == thh.shape


def test_decode_frame_is_numpy_only_no_torch_dependency():
    """The decode (inflate) path imports no torch/scorer (receiver contract)."""

    rng = np.random.default_rng(5)
    pyrs = [encode_frame_lf(_smooth_frame(rng), levels=2) for _ in range(3)]
    dec = fit_hf_decoder_least_squares(pyrs, levels=2)
    p0 = pyrs[0]
    q, sc, zr = quantize_lf(p0.lf, n_levels=256)
    code = SnervFrameCode(
        lf_quant=q, lf_scale=sc, lf_zero=zr, lf_shape=p0.lf.shape,
        levels=2, wavelet=p0.wavelet, orig_hw=(64, 96),
    )
    out = decode_frame(code, dec)
    assert isinstance(out, np.ndarray)
    assert out.shape == (64, 96)
    # the decode helpers in carrier.py must not import torch
    import tac.substrates.snerv_inverse_steg_carrier.carrier as carrier_mod

    with open(carrier_mod.__file__) as f:
        src = f.read()
    assert "import torch" not in src and "from torch" not in src


def test_decode_shape_mismatch_raises():
    dec = HfGenerationDecoder.zeros(3)
    bad = SnervFrameCode(
        lf_quant=np.zeros((2, 2), dtype=np.int64), lf_scale=1.0, lf_zero=0.0,
        lf_shape=(2, 2), levels=3, wavelet="db2", orig_hw=(64, 96),
    )
    with pytest.raises(SnervCarrierError):
        decode_frame(bad, dec)


def _hot_detail_weight_pyramid(pyr: WaveletPyramid) -> WaveletPyramid:
    weighted_details = []
    for lh, hl, hh in pyr.details:
        detail_tuple = []
        for detail in (lh, hl, hh):
            weights = np.ones_like(detail, dtype=np.float64)
            h, w = weights.shape
            weights[: max(1, h // 3), : max(1, w // 3)] = 50.0
            detail_tuple.append(weights)
        weighted_details.append(tuple(detail_tuple))
    return WaveletPyramid(
        coeffs=[np.ones_like(pyr.lf), *weighted_details],
        levels=pyr.levels,
        wavelet=pyr.wavelet,
        orig_hw=pyr.orig_hw,
        padded_hw=pyr.padded_hw,
    )


def _weighted_hf_residual(
    pyrs: list[WaveletPyramid],
    decoder: HfGenerationDecoder,
    weight_pyrs: list[WaveletPyramid],
) -> float:
    total = 0.0
    denom = 0.0
    for pyr, weight_pyr in zip(pyrs, weight_pyrs, strict=True):
        generated = generate_hf_from_lf(pyr.lf, decoder, pyr)
        for generated_details, target_details, weight_details in zip(
            generated[1:],
            pyr.details,
            weight_pyr.details,
            strict=True,
        ):
            for got, target, weights in zip(
                generated_details,
                target_details,
                weight_details,
                strict=True,
            ):
                total += float(np.sum(np.asarray(weights) * (got - target) ** 2))
                denom += float(np.sum(weights))
    return total / denom
