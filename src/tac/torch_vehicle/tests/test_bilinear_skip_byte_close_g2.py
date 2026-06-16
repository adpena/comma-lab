# SPDX-License-Identifier: MIT
"""G2 — NO-FAKE proofs that the small-basis (taper) HNeRV decoder is BYTE-CLOSEABLE.

Refutes the symposium's claimed C8 blocker ("bilinear-skip archive export is a
``NotImplementedError`` / bilinear-skip CANNOT be byte-closed", frontier-ledger #81).
The bilinear-skip is a ``forward()`` op, NOT an archive section — the vendored codec
serializes weights by name/shape/scale and is schedule-agnostic, so the bilinear-skip
round-trips trivially as the torch decoder forward on both build and parse-back sides.

These tests assert REAL behavior (actual archive bytes, actual parse-back, actual
decoder forwards incl. the bilinear-skip op) — never constants. Every test would FAIL
if the byte-close path silently dropped weights, was non-deterministic, or if bilinear
were not reproducible. Authority: ``[advisory]`` local CPU; no score is claimed.

Verdict memo: ``.omx/research/g2_c8_bilinear_skip_byte_close_verdict_20260616T160046Z.md``.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from tac.torch_vehicle.configurable_taper_decoder import (
    ConfigurableTaperHNeRVDecoder,
    decoder_param_count,
    dseg_aware_taper,
)

_LAT, _BASE, _N = 28, 20, 24


def _bundle():
    from tac.torch_vehicle.driver import import_vendored_bundle

    return import_vendored_bundle()


def _small_basis_decoder(seed: int = 0):
    """The actual small-basis vehicle: base_ch=20 d_seg-aware taper (~83K params)."""
    torch.manual_seed(seed)
    taper = dseg_aware_taper(_BASE, latent_dim=_LAT)
    dec = ConfigurableTaperHNeRVDecoder(
        latent_dim=_LAT, base_channels=_BASE, eval_size=(384, 512), channels=taper
    ).eval()
    return dec, taper


def _to_uint8_camera(frames: torch.Tensor) -> np.ndarray:
    """Mirror the vendored inflate: decoder float frames -> bicubic 874x1164 -> uint8."""
    flat = frames.reshape(-1, 3, 384, 512)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    return up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).numpy()


def test_no_notimplementederror_on_byte_close_path():
    """The claimed blocker is a NotImplementedError. Prove the path executes cleanly.

    A real build_archive + parse_archive on the small-basis decoder must NOT raise.
    """
    v = _bundle()
    dec, _ = _small_basis_decoder()
    latents = torch.randn(_N, _LAT)
    meta = {"latent_dim": _LAT, "base_channels": _BASE, "eval_size": [384, 512], "n_pairs": _N}
    archive = v.build_archive(dec.state_dict(), latents, meta_dict=meta)  # must not raise
    decoder_sd, latents_back, meta_back = v.parse_archive(archive)  # must not raise
    assert len(archive) > 0
    assert set(decoder_sd.keys()) == set(dec.state_dict().keys())
    assert latents_back.shape == latents.shape
    assert meta_back["base_channels"] == _BASE


def test_byte_close_roundtrip_reproduces_frames_subquant():
    """PROOF A — parse-back decoder reproduces the frames to within the int8 quant step.

    The ONLY divergence is int8 weight-quant rounding (<= 1 uint8 LSB). The bilinear-skip
    runs identically on both sides; a bilinear failure would blow this far past 1.
    """
    v = _bundle()
    dec, taper = _small_basis_decoder()
    latents = torch.randn(_N, _LAT)
    meta = {"latent_dim": _LAT, "base_channels": _BASE, "eval_size": [384, 512], "n_pairs": _N}

    with torch.inference_mode():
        ref = dec(latents[:8])  # incl. bilinear-skip

    archive = v.build_archive(dec.state_dict(), latents, meta_dict=meta)
    dec_sd, lat_back, meta_back = v.parse_archive(archive)
    dec2 = ConfigurableTaperHNeRVDecoder(
        latent_dim=meta_back["latent_dim"],
        base_channels=meta_back["base_channels"],
        eval_size=tuple(meta_back["eval_size"]),
        channels=taper,
    ).eval()
    missing, unexpected = dec2.load_state_dict(dec_sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    with torch.inference_mode():
        out = dec2(lat_back[:8])

    diff = np.abs(_to_uint8_camera(ref).astype(np.int32) - _to_uint8_camera(out).astype(np.int32))
    # Sub-quant: <= 1 LSB everywhere (int8 weight quant is the only source of divergence).
    assert int(diff.max()) <= 1, f"max uint8 diff {diff.max()} exceeds the 1-LSB quant step"


def test_codec_is_fixed_point_parseback_idempotent():
    """PROOF B — parse-back of a parse-back is BIT-EXACT (weights + latents).

    Once on the int8 grid the archive is a fixed point; the codec round-trips exactly.
    """
    v = _bundle()
    dec, _ = _small_basis_decoder(seed=1)
    latents = torch.randn(_N, _LAT)
    meta = {"latent_dim": _LAT, "base_channels": _BASE, "eval_size": [384, 512], "n_pairs": _N}
    arc1 = v.build_archive(dec.state_dict(), latents, meta_dict=meta)
    sd1, lat1, _ = v.parse_archive(arc1)
    arc2 = v.build_archive(sd1, lat1, meta_dict=meta)
    sd2, lat2, _ = v.parse_archive(arc2)
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1), "parse-back weights not fixed-point"
    assert torch.equal(lat1, lat2), "parse-back latents not fixed-point"


def test_build_archive_is_byte_deterministic():
    """PROOF C — build_archive on identical inputs is byte-identical."""
    v = _bundle()
    dec, _ = _small_basis_decoder(seed=2)
    latents = torch.randn(_N, _LAT)
    meta = {"latent_dim": _LAT, "base_channels": _BASE, "eval_size": [384, 512], "n_pairs": _N}
    a = v.build_archive(dec.state_dict(), latents, meta_dict=meta)
    b = v.build_archive(dec.state_dict(), latents, meta_dict=meta)
    assert a == b, "build_archive is non-deterministic"
    assert hashlib.sha256(a).digest() == hashlib.sha256(b).digest()


def test_parseback_frames_are_deterministic_incl_bilinear_skip():
    """PROOF D — two parse-back decoders produce BIT-IDENTICAL float frames.

    This exercises the bilinear-skip op (the alleged blocker) on byte-closed weights and
    proves it is deterministic — i.e. fully byte-closeable.
    """
    v = _bundle()
    dec, taper = _small_basis_decoder(seed=3)
    latents = torch.randn(_N, _LAT)
    meta = {"latent_dim": _LAT, "base_channels": _BASE, "eval_size": [384, 512], "n_pairs": _N}
    sd, lat, _ = v.parse_archive(v.build_archive(dec.state_dict(), latents, meta_dict=meta))
    d_a = ConfigurableTaperHNeRVDecoder(
        latent_dim=_LAT, base_channels=_BASE, eval_size=(384, 512), channels=taper
    ).eval()
    d_b = ConfigurableTaperHNeRVDecoder(
        latent_dim=_LAT, base_channels=_BASE, eval_size=(384, 512), channels=taper
    ).eval()
    d_a.load_state_dict(sd)
    d_b.load_state_dict(sd)
    with torch.inference_mode():
        fa, fb = d_a(lat[:6]), d_b(lat[:6])
    assert torch.equal(fa, fb), "parse-back forward (incl. bilinear-skip) is non-deterministic"


def test_bilinear_align_corners_false_is_numpy_portable():
    """PROOF E — pure-numpy bilinear (align_corners=False) matches torch to float32 ULP.

    So even a numpy-ONLY inflate (no torch dep) reproduces the bilinear-skip scorer-
    identically. Max abs drift must be << the uint8 step (1.0); empirically ~3.6e-7.
    """

    def numpy_bilinear_2x_acf(x_nchw: np.ndarray) -> np.ndarray:
        B, C, H, W = x_nchw.shape

        def axis(out_len, in_len):
            dst = np.arange(out_len)
            src = np.clip((dst + 0.5) * (in_len / out_len) - 0.5, 0, in_len - 1)
            lo = np.floor(src).astype(np.int64)
            hi = np.minimum(lo + 1, in_len - 1)
            return lo, hi, (src - lo).astype(np.float64)

        yl, yh, wy = axis(2 * H, H)
        xl, xh, wx = axis(2 * W, W)
        x = x_nchw.astype(np.float64)
        rowi = x[:, :, yl, :] * (1 - wy)[None, None, :, None] + x[:, :, yh, :] * wy[None, None, :, None]
        return rowi[:, :, :, xl] * (1 - wx)[None, None, None, :] + rowi[:, :, :, xh] * wx[None, None, None, :]

    torch.manual_seed(4)
    for (h, w) in [(6, 8), (12, 16), (24, 32), (48, 64), (96, 128), (192, 256)]:
        x = torch.randn(1, 10, h, w)
        t = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False).numpy().astype(np.float64)
        n = numpy_bilinear_2x_acf(x.numpy())
        max_abs = float(np.abs(t - n).max())
        assert max_abs < 1e-5, f"numpy bilinear drift {max_abs} at {h}x{w} (>= 1e-5)"


def test_small_basis_param_count_and_archive_size_sane():
    """Sanity: the small-basis vehicle is ~83K params and byte-closes to ~85KB."""
    v = _bundle()
    dec, taper = _small_basis_decoder(seed=5)
    n_params = decoder_param_count(taper, latent_dim=_LAT)
    assert 70_000 < n_params < 100_000, n_params
    latents = torch.randn(_N, _LAT)
    meta = {"latent_dim": _LAT, "base_channels": _BASE, "eval_size": [384, 512], "n_pairs": _N}
    archive = v.build_archive(dec.state_dict(), latents, meta_dict=meta)
    # int8(~83K params) + quantized latents + brotli -> well under the 177KB frontier.
    assert 50_000 < len(archive) < 150_000, len(archive)
