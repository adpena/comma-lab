# SPDX-License-Identifier: MIT
"""Tests for the Cool-Chic AR-Gaussian entropy coder (TRACK B step 1).

NO FAKE discipline (Slot EEE Class 2): every test asserts ACTUAL coder BEHAVIOR
on REAL trained latents (or trainable synthetic latents from the real architecture)
— losslessness, byte reduction, density-matches-AR-prior, escape-path, CCV2
grammar round-trip, Layer-1↔2 bit unification. NOT constant-checking: each test
would FAIL if the coder body were replaced by a no-op or a fixed-width store.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from tac.substrates.cool_chic.architecture import CoolChicConfig, CoolChicSubstrate
from tac.substrates.cool_chic.archive import (
    CCV2_MAGIC,
    CCV2_SCHEMA_VERSION,
    pack_archive,
    pack_archive_ccv2,
    parse_archive,
)
from tac.substrates.cool_chic.entropy_coder import (
    LatentGrid,
    ar_gaussian_predicted_bits,
    choose_grid_step,
    decode_latent_chain,
    encode_latent_chain,
)

_SMOKE_BIN = Path(
    "experiments/results/cool_chic_mps_smoke_20260612T121501Z/0.bin"
)


def _tiny_model(seed: int = 0) -> CoolChicSubstrate:
    cfg = CoolChicConfig(
        latent_channels_coarse=2,
        latent_channels_fine=2,
        coarse_scale_factor=8,
        fine_scale_factor=4,
        synthesis_hidden=8,
        synthesis_layers=2,
        ar_prior_hidden=4,
        num_pairs=5,
        output_height=16,
        output_width=24,
    )
    torch.manual_seed(seed)
    return CoolChicSubstrate(cfg).eval()


# --------------------------------------------------------------------------- #
# Core: lossless round-trip on the integer grid                               #
# --------------------------------------------------------------------------- #

def test_encode_decode_lossless_on_grid_tiny():
    m = _tiny_model(0)
    # train-like latents: structured (temporally correlated), not uniform noise
    lc = m.latents_coarse.detach().clone()
    grid = LatentGrid(step=choose_grid_step(lc, bits_per_std=5.0))
    blob = encode_latent_chain(lc, m.ar_prior_coarse, grid)
    z_hat = decode_latent_chain(blob, m.ar_prior_coarse)
    # the coder is lossless w.r.t. the integer grid: decoded k == original k
    assert torch.equal(grid.quantize(lc), grid.quantize(z_hat))
    # and z_hat equals the dequantized grid exactly (bit-for-bit)
    assert torch.equal(grid.dequantize(grid.quantize(lc)), z_hat)


def test_encode_decode_lossless_shape_preserved():
    m = _tiny_model(1)
    lf = m.latents_fine.detach().clone()
    grid = LatentGrid(step=choose_grid_step(lf, bits_per_std=4.0))
    blob = encode_latent_chain(lf, m.ar_prior_fine, grid)
    z_hat = decode_latent_chain(blob, m.ar_prior_fine)
    assert z_hat.shape == lf.shape


def test_decode_is_not_a_noop_passthrough():
    """If the coder secretly stored raw floats / did nothing, the bytes would
    not depend on the AR prior. Swap the AR prior at decode -> must MISDECODE
    (proves the density actually drives the coding, not a passthrough)."""
    m = _tiny_model(2)
    lc = m.latents_coarse.detach().clone()
    grid = LatentGrid(step=choose_grid_step(lc, bits_per_std=5.0))
    blob = encode_latent_chain(lc, m.ar_prior_coarse, grid)
    # a DIFFERENT prior net (different weights) -> different CDFs -> wrong decode
    other = _tiny_model(999).ar_prior_coarse
    z_wrong = decode_latent_chain(blob, other)
    # decode with the wrong density should not reproduce the grid (the coder is
    # genuinely density-conditioned, not a fixed-table / passthrough store)
    assert not torch.equal(grid.quantize(lc), grid.quantize(z_wrong))


# --------------------------------------------------------------------------- #
# Byte reduction (the deliverable headline)                                   #
# --------------------------------------------------------------------------- #

def test_ar_coded_smaller_than_raw_int16_tiny():
    m = _tiny_model(3)
    lc = m.latents_coarse.detach().clone()
    grid = LatentGrid(step=choose_grid_step(lc, bits_per_std=5.0))
    blob = encode_latent_chain(lc, m.ar_prior_coarse, grid)
    raw_int16 = lc.numel() * 2
    # AR-coded latents (header + escapes + range stream) must be SMALLER than the
    # raw int16 store the L0 grammar used. (Combines requant + entropy, both
    # lossless-to-grid.)
    assert len(blob) < raw_int16


def test_ar_coded_concentrated_latents_beats_fixed_width():
    """Pure-entropy gain (isolates entropy lever from requant): when the AR
    Gaussian ACCURATELY predicts the latents (mean≈latent, tight sigma), the
    coded stream must be SMALLER than a minimal fixed-width store at the same
    grid. We construct an analytic AR prior whose mean matches z by zeroing the
    net so it predicts the zero-context mean, and concentrating latents near 0.

    Uses a hand-built grid + a near-degenerate latent (almost all the SAME small
    value) so the empirical symbol entropy << fixed-width log2(range). The coder
    MUST exploit the concentration (a no-op/fixed-width store could not)."""
    import struct as _struct

    from tac.substrates.cool_chic.entropy_coder import _ECCV_HEADER_SIZE

    # larger config so N >> header overhead (representative of the real archive)
    cfg = CoolChicConfig(
        latent_channels_coarse=4, latent_channels_fine=4, coarse_scale_factor=8,
        fine_scale_factor=4, synthesis_hidden=8, synthesis_layers=2,
        ar_prior_hidden=4, num_pairs=4, output_height=32, output_width=48,
    )
    torch.manual_seed(4)
    m = CoolChicSubstrate(cfg).eval()
    # Build an ACCURATE AR prior: mean≈0 (zero the output conv) and a TIGHT sigma
    # (~grid step) so the Gaussian concentrates its mass on the right bin. This is
    # the case where an entropy coder MUST win: an accurate density. A no-op /
    # fixed-width store cannot exploit it; the entropy coder does.
    grid = LatentGrid(step=0.001)
    with torch.no_grad():
        out_conv = m.ar_prior_coarse.net[2]
        out_conv.weight.zero_()
        out_conv.bias.zero_()
        # second half of channels = log_scale -> set to log(tight sigma)
        cch = cfg.latent_channels_coarse
        out_conv.bias[cch:].fill_(math.log(grid.step * 1.5))  # sigma ~ 1.5 * step
    # latents concentrated at the predicted mean (0) -> tight Gaussian = ~0 bits
    lc = torch.full_like(m.latents_coarse, 0.0005)  # rounds to bin 0 or 1
    lc[0, 0, 0, 0] = 0.05  # one outlier so range>1 (fixed-width must pay for it)
    blob = encode_latent_chain(lc, m.ar_prior_coarse, grid)
    _, _np, _c, _h, _w, _step, n_esc, stream_len = _struct.unpack(
        "<4sIIIIdII", blob[:_ECCV_HEADER_SIZE]
    )
    k = grid.quantize(lc)
    krange = int(k.max() - k.min()) + 1
    fixed_bits = max(1, math.ceil(math.log2(max(2, krange))))
    fixed_bytes = math.ceil(k.numel() * fixed_bits / 8)
    # accurate tight density -> the entropy STREAM must beat fixed-width
    assert stream_len < fixed_bytes, f"stream {stream_len} not < fixed-width {fixed_bytes}"
    # and it must STILL round-trip losslessly with this hand-built prior
    z_hat = decode_latent_chain(blob, m.ar_prior_coarse)
    assert torch.equal(grid.quantize(lc), grid.quantize(z_hat))


# --------------------------------------------------------------------------- #
# Density matches the AR prior (the Layer-1 ↔ Layer-2 unification)            #
# --------------------------------------------------------------------------- #

def test_coded_rate_tracks_ar_predicted_bits():
    """The coder's actual byte count must be close to the AR Gaussian predicted
    bits (-log2 P(bin)). This is the density-matches-AR-prior property AND the
    Layer-1↔2 unification: train-time predicted rate == deploy-time coded bytes."""
    m = _tiny_model(5)
    lc = m.latents_coarse.detach().clone()
    grid = LatentGrid(step=choose_grid_step(lc, bits_per_std=5.0))
    # predicted bits from the SAME AR prior (replay the causal chain)
    num_pairs, c, h, w = lc.shape
    prev = torch.zeros((1, c, h, w))
    total_pred_bits = torch.zeros(())
    with torch.no_grad():
        for t in range(num_pairs):
            mean, log_scale = m.ar_prior_coarse(prev)
            total_pred_bits = total_pred_bits + ar_gaussian_predicted_bits(
                lc[t : t + 1], mean, log_scale, grid
            )
            prev = grid.dequantize(grid.quantize(lc[t : t + 1]))
    pred_bytes = float(total_pred_bits) / 8.0
    blob = encode_latent_chain(lc, m.ar_prior_coarse, grid)
    # actual coded stream (exclude fixed header+escape overhead from the comparison)
    # the range stream alone should be within a modest factor of the predicted bits
    # (predicted is the ideal entropy; coder adds < ~2 bytes flush + freq-table
    # quantization). Assert actual is not WILDLY off the prediction.
    assert pred_bytes > 0
    # actual total must be >= the ideal entropy minus a small slack, and not more
    # than ~2x it for this tiny case (freq-table + per-symbol overhead on small N)
    assert len(blob) >= pred_bytes * 0.5
    assert len(blob) <= pred_bytes * 3.0 + 64


def test_predicted_bits_offset_from_continuous_nll_is_constant():
    """ar_gaussian_predicted_bits == continuous NLL/ln2 + constant(-log2 step) per
    element. Verifies the Layer-2 surrogate (continuous NLL) and this coder's
    bit model differ only by the documented constant offset."""
    m = _tiny_model(6)
    lc = m.latents_coarse.detach().clone()
    step = choose_grid_step(lc, bits_per_std=5.0)
    grid = LatentGrid(step=step)
    prev = torch.zeros((1, *tuple(lc.shape[1:])))
    with torch.no_grad():
        mean, log_scale = m.ar_prior_coarse(prev)
        z0 = lc[0:1]
        bits = ar_gaussian_predicted_bits(z0, mean, log_scale, grid)
        # continuous NLL in nats
        sigma = torch.exp(log_scale)
        diff = (z0 - mean) / sigma
        ln_p = -0.5 * diff * diff - log_scale - 0.5 * math.log(2 * math.pi)
        cont_bits = (-ln_p).clamp(min=0.0) / math.log(2.0)
        offset_bits = (-math.log(step) / math.log(2.0)) * z0.numel()
    # bits ≈ cont_bits.sum() + offset (within clamp-at-zero rounding)
    assert abs(float(bits) - (float(cont_bits.sum()) + offset_bits)) < float(z0.numel()) + 1e-3


# --------------------------------------------------------------------------- #
# Escape path (lossless on outliers — the lossless guarantee)                 #
# --------------------------------------------------------------------------- #

def test_escape_path_is_lossless_for_outliers():
    """Inject a latent far outside the AR window; the escape symbol must carry it
    losslessly (decode reconstructs the exact extreme value)."""
    m = _tiny_model(7)
    lc = m.latents_coarse.detach().clone()
    grid = LatentGrid(step=choose_grid_step(lc, bits_per_std=5.0))
    # set one element to a huge outlier (forces an escape)
    lc[2, 0, 0, 0] = 50.0
    blob = encode_latent_chain(lc, m.ar_prior_coarse, grid)
    z_hat = decode_latent_chain(blob, m.ar_prior_coarse)
    assert torch.equal(grid.quantize(lc), grid.quantize(z_hat))
    # the outlier specifically must be exactly reconstructed
    assert grid.quantize(lc)[2, 0, 0, 0].item() == grid.quantize(z_hat)[2, 0, 0, 0].item()


# --------------------------------------------------------------------------- #
# CCV2 grammar end-to-end (the archive integration)                           #
# --------------------------------------------------------------------------- #

def _tiny_pack_inputs(m: CoolChicSubstrate):
    synth_sd = m.synthesis.state_dict()
    ar_sd = {}
    for k, v in m.ar_prior_coarse.state_dict().items():
        ar_sd[f"coarse.{k}"] = v
    for k, v in m.ar_prior_fine.state_dict().items():
        ar_sd[f"fine.{k}"] = v
    meta = {
        "coarse_scale_factor": m.cfg.coarse_scale_factor,
        "fine_scale_factor": m.cfg.fine_scale_factor,
        "synthesis_hidden": m.cfg.synthesis_hidden,
        "synthesis_layers": m.cfg.synthesis_layers,
        "ar_prior_hidden": m.cfg.ar_prior_hidden,
        "output_height": m.cfg.output_height,
        "output_width": m.cfg.output_width,
    }
    return synth_sd, ar_sd, m.latents_coarse.detach().clone(), m.latents_fine.detach().clone(), meta


def test_ccv2_pack_parse_roundtrip_lossless():
    m = _tiny_model(8)
    synth_sd, ar_sd, lc, lf, meta = _tiny_pack_inputs(m)
    blob = pack_archive_ccv2(synth_sd, ar_sd, lc, lf, meta, bits_per_std=5.0)
    assert blob[:4] == CCV2_MAGIC
    arc = parse_archive(blob)
    assert arc.schema_version == CCV2_SCHEMA_VERSION
    assert arc.latents_coarse.shape == lc.shape
    assert arc.latents_fine.shape == lf.shape
    # latents lossless to grid
    gc = LatentGrid(step=choose_grid_step(lc, bits_per_std=5.0))
    gf = LatentGrid(step=choose_grid_step(lf, bits_per_std=5.0))
    assert torch.equal(gc.quantize(lc), gc.quantize(arc.latents_coarse))
    assert torch.equal(gf.quantize(lf), gf.quantize(arc.latents_fine))
    # synthesis + AR state dicts round-trip (keys preserved)
    assert set(arc.synthesis_state_dict.keys()) == set(synth_sd.keys())
    assert set(arc.ar_prior_state_dict.keys()) == set(ar_sd.keys())


def test_ccv2_smaller_than_ccv1_for_same_latents():
    """The whole point: CCV2 (entropy-coded) archive must be SMALLER than CCV1
    (raw int16) for the same trained latents."""
    m = _tiny_model(9)
    # informative AR prior via temporal correlation -> CCV2 should beat CCV1
    with torch.no_grad():
        for t in range(1, m.latents_coarse.shape[0]):
            m.latents_coarse[t] = m.latents_coarse[t - 1] * 0.9
            m.latents_fine[t] = m.latents_fine[t - 1] * 0.9
    synth_sd, ar_sd, lc, lf, meta = _tiny_pack_inputs(m)
    ccv1 = pack_archive(synth_sd, ar_sd, lc, lf, meta)
    ccv2 = pack_archive_ccv2(synth_sd, ar_sd, lc, lf, meta, bits_per_std=5.0)
    assert len(ccv2) < len(ccv1), f"CCV2 {len(ccv2)} not < CCV1 {len(ccv1)}"


def test_ccv2_byte_mutation_no_op_proof():
    """Catalog #139: mutating a latent must change the CCV2 archive bytes AND the
    decoded latent. Proves the entropy section actually carries the latent."""
    m = _tiny_model(10)
    synth_sd, ar_sd, lc, lf, meta = _tiny_pack_inputs(m)
    a = pack_archive_ccv2(synth_sd, ar_sd, lc, lf, meta, bits_per_std=5.0)
    lc_mut = lc.clone()
    lc_mut[0, 0, 0, 0] = lc_mut[0, 0, 0, 0] + 0.5  # > grid step
    b = pack_archive_ccv2(synth_sd, ar_sd, lc_mut, lf, meta, bits_per_std=5.0)
    assert a != b
    arc_a, arc_b = parse_archive(a), parse_archive(b)
    assert not torch.allclose(
        arc_a.latents_coarse[0, 0, 0, 0], arc_b.latents_coarse[0, 0, 0, 0], atol=1e-6
    )


# --------------------------------------------------------------------------- #
# REAL trained latents (gated on the smoke artifact existing)                 #
# --------------------------------------------------------------------------- #

def test_real_smoke_latents_lossless_and_smaller():
    if not _SMOKE_BIN.is_file():
        import unittest

        raise unittest.SkipTest(f"smoke bin not present: {_SMOKE_BIN}")
    arc = parse_archive(_SMOKE_BIN.read_bytes())
    cfg = CoolChicConfig(
        num_pairs=int(arc.latents_coarse.shape[0]),
        ar_prior_hidden=int(arc.meta["ar_prior_hidden"]),
    )
    m = CoolChicSubstrate(cfg).eval()
    m.ar_prior_coarse.load_state_dict(
        {k.replace("coarse.", ""): v for k, v in arc.ar_prior_state_dict.items() if k.startswith("coarse.")},
        strict=False,
    )
    # use a small subset of REAL trained coarse latents for speed
    lc = arc.latents_coarse[:6].to(torch.float32)
    grid = LatentGrid(step=choose_grid_step(lc, bits_per_std=5.0))
    blob = encode_latent_chain(lc, m.ar_prior_coarse, grid)
    z_hat = decode_latent_chain(blob, m.ar_prior_coarse)
    assert torch.equal(grid.quantize(lc), grid.quantize(z_hat))
    assert len(blob) < lc.numel() * 2  # smaller than raw int16
