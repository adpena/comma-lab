# SPDX-License-Identifier: MIT
"""Byte-close + numpy inflate for the Cool-Chic non-conv basis carrier.

Lane ``lane_cool_chic_score_aware_basis_20260611``. This converts B1's ARM
ENTROPY ESTIMATE into a REAL self-contained archive blob + a numpy-portable
inflate, so the ``charged_bytes`` claim becomes a MEASURED byte count and the
d_seg / d_pose become MEASURED on the byte-closed-reloaded render (not the
training logits).

Honest byte-close (per CLAUDE.md "Bit-level deconstruction and entropy
discipline" + "Forbidden closed-form-CDF-allocator-without-empirical-bit-spend
-proof"):

  - The ARM ``-log2 p`` is a CONTINUOUS Laplacian estimate. A real archive must
    store INTEGERS. We quantize each latent grid (and per-pair frame1 deltas)
    with a per-tensor step, then RANGE-CODE the integer symbols with a static
    frequency table built from the EMPIRICAL symbol histogram (the table is
    written into the header, so the cost is real, not free). Reuses the proven
    32-bit range coder ``tac.lossless.range_coder``.
  - The synthesis + ARM-free weights are INT8-packed with a per-tensor fp16
    scale (PR101 centered-int8 pattern: ``dequant = int8 * scale``).
  - The inflate (``inflate_numpy``) decodes the blob and renders via the numpy
    reference ``synthesize_rgb_numpy`` — bit-exact to the torch render is the
    portability contract (verified on REAL non-zero weights, not zero-init).

The blob is a single length-prefixed byte string (the PR95 monolithic-0.bin
discipline): a JSON header (shapes, quant steps, per-grid frequency tables,
int8 weight scales) + the range-coded latent stream + the int8 weight bytes.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

from tac.lossless.range_coder import (
    cumulative_frequencies,
    decode_static_symbols,
    encode_static_symbols,
)
from tac.residual_basis.cool_chic_synthesis_numpy import (
    SynthesisWeightsNumpy,
    bilinear_upsample_numpy,
    synthesize_rgb_numpy,
)

MAGIC = b"CCBC"  # Cool-Chic Byte-Close
VERSION = 1


# --------------------------------------------------------------------------
# quantization helpers (honest: integer symbols + per-tensor scale)
# --------------------------------------------------------------------------


def quantize_to_symbols(x: np.ndarray, step: float) -> tuple[np.ndarray, int]:
    """Round ``x/step`` to ints, shift to non-negative symbols.

    Returns ``(symbols, offset)`` where ``symbols >= 0`` and the dequantized
    value is ``(symbols - offset) * step``. ``offset`` is the min int so the
    range coder sees small non-negative symbols.
    """
    q = np.rint(x.astype(np.float64) / step).astype(np.int64)
    offset = int(-q.min()) if q.size else 0
    return (q + offset).astype(np.int64), offset


def dequantize_symbols(symbols: np.ndarray, offset: int, step: float) -> np.ndarray:
    return (symbols.astype(np.float64) - offset) * step


def _histogram_freqs(symbols: np.ndarray, n_symbols: int) -> list[int]:
    """Empirical frequency table (>=1 each so the range coder is well-formed)."""
    counts = np.bincount(symbols.ravel(), minlength=n_symbols).astype(np.int64)
    counts = np.maximum(counts, 1)
    return counts.tolist()


def _int8_pack(w: np.ndarray) -> tuple[bytes, float]:
    """Per-tensor symmetric int8 pack. ``dequant = int8 * scale``."""
    amax = float(np.abs(w).max()) if w.size else 0.0
    scale = (amax / 127.0) if amax > 0 else 1.0
    q = np.clip(np.rint(w.astype(np.float64) / scale), -127, 127).astype(np.int8)
    return q.tobytes(), scale


def _int8_unpack(b: bytes, shape, scale: float) -> np.ndarray:
    q = np.frombuffer(b, dtype=np.int8).astype(np.float64).reshape(shape)
    return q * scale


# --------------------------------------------------------------------------
# the carrier-state extraction (torch -> numpy)
# --------------------------------------------------------------------------


@dataclass
class CarrierState:
    """The plain-numpy state a Cool-Chic carrier needs to render a pair.

    ``grids`` are the shared frame0 latent grids; ``frame1_delta`` is the
    per-pair coarse delta; ``w1/b1/w2/b2`` the synthesis weights; ``delta_proj``
    projects the coarse delta into the c_in feature space; ``out_hw`` the render
    size; ``spec_c_in`` / ``coarse_shape`` the layout.
    """

    grids: list[np.ndarray]
    frame1_delta: np.ndarray  # (n_pairs, c0, h0, w0)
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    delta_proj: np.ndarray  # (c_in, c0)
    out_hw: tuple[int, int]
    coarse_shape: tuple[int, int, int]


def carrier_to_state(carrier) -> CarrierState:
    """Snapshot a (torch) ``CoolChicPairCarrier`` into plain numpy."""
    import torch  # local import; numpy inflate path has no torch dep

    with torch.no_grad():
        grids = [g.detach().cpu().numpy().astype(np.float64) for g in carrier.latent_grids]
        return CarrierState(
            grids=grids,
            frame1_delta=carrier.frame1_delta.detach().cpu().numpy().astype(np.float64),
            w1=carrier.w1.detach().cpu().numpy().astype(np.float64),
            b1=carrier.b1.detach().cpu().numpy().astype(np.float64),
            w2=carrier.w2.detach().cpu().numpy().astype(np.float64),
            b2=carrier.b2.detach().cpu().numpy().astype(np.float64),
            delta_proj=carrier.delta_proj.detach().cpu().numpy().astype(np.float64),
            out_hw=(int(carrier.out_hw[0]), int(carrier.out_hw[1])),
            coarse_shape=tuple(int(v) for v in carrier._coarse_shape),
        )


# --------------------------------------------------------------------------
# numpy render (matches CoolChicPairCarrier.reconstruct_pair, no torch)
# --------------------------------------------------------------------------


def render_pair_numpy(state: CarrierState, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Render ``(rgb0, rgb1)`` each ``(3, H, W)`` in [0,255] for pair ``idx``.

    Mirrors ``CoolChicPairCarrier.reconstruct_pair`` exactly: frame0 from the
    shared grids; frame1 adds the projected+upsampled coarse delta in feature
    space before synthesis.
    """
    h, w = state.out_hw
    weights = SynthesisWeightsNumpy(w1=state.w1, b1=state.b1, w2=state.w2, b2=state.b2)
    # feat0 = concat(upsample(grids))  -> (c_in, H, W)
    ups = [bilinear_upsample_numpy(g, h, w) for g in state.grids]
    feat0 = np.concatenate(ups, axis=0)
    rgb0 = synthesize_rgb_numpy(state.grids, weights, h, w) * 255.0

    # frame1: project coarse delta -> c_in, upsample, add to feat0, synth.
    c0, h0, w0 = state.coarse_shape
    delta = state.frame1_delta[idx]  # (c0, h0, w0)
    dflat = delta.reshape(c0, h0 * w0)
    proj = state.delta_proj @ dflat  # (c_in, h0*w0)
    c_in = state.delta_proj.shape[0]
    proj = proj.reshape(c_in, h0, w0)
    proj_up = bilinear_upsample_numpy(proj, h, w)  # (c_in, H, W)
    feat1 = feat0 + proj_up
    # synth from feat1 directly (synthesize_rgb_numpy re-upsamples grids, so we
    # inline the 1x1-conv math on feat1 to match feat0+delta).
    flat = feat1.reshape(c_in, h * w)
    from tac.residual_basis.cool_chic_synthesis_numpy import _gelu

    hidden = _gelu(weights.w1 @ flat + weights.b1[:, None])
    out = weights.w2 @ hidden + weights.b2[:, None]
    rgb1 = (1.0 / (1.0 + np.exp(-out))).reshape(3, h, w) * 255.0
    return rgb0, rgb1


# --------------------------------------------------------------------------
# byte-close (encode) + inflate (decode)
# --------------------------------------------------------------------------


def byte_close(
    state: CarrierState,
    *,
    grid_step: float = 0.05,
    delta_step: float = 0.05,
) -> bytes:
    """Encode the carrier state into a single self-contained archive blob.

    Latent grids + per-pair frame1 deltas are quantized to integer symbols and
    range-coded with per-tensor empirical frequency tables (written in the
    header). Synthesis weights are int8-packed. Returns the blob bytes.
    """
    header: dict = {
        "version": VERSION,
        "out_hw": list(state.out_hw),
        "coarse_shape": list(state.coarse_shape),
        "grid_shapes": [list(g.shape) for g in state.grids],
        "delta_shape": list(state.frame1_delta.shape),
        "grid_step": grid_step,
        "delta_step": delta_step,
        "grids": [],
        "delta": None,
        "weights": {},
    }
    streams: list[bytes] = []

    # --- latent grids ---
    for g in state.grids:
        syms, offset = quantize_to_symbols(g, grid_step)
        n_sym = int(syms.max()) + 1
        freqs = _histogram_freqs(syms, n_sym)
        coded = encode_static_symbols(syms.ravel().tolist(), frequencies=freqs)
        header["grids"].append(
            {"offset": offset, "n_sym": n_sym, "freqs": freqs, "count": int(syms.size)}
        )
        streams.append(coded)

    # --- per-pair frame1 deltas (one stream over all pairs, shared table) ---
    syms, offset = quantize_to_symbols(state.frame1_delta, delta_step)
    n_sym = int(syms.max()) + 1
    freqs = _histogram_freqs(syms, n_sym)
    coded = encode_static_symbols(syms.ravel().tolist(), frequencies=freqs)
    header["delta"] = {"offset": offset, "n_sym": n_sym, "freqs": freqs, "count": int(syms.size)}
    streams.append(coded)

    # --- synthesis weights (int8) ---
    weight_blobs: list[bytes] = []
    for name in ("w1", "b1", "w2", "b2", "delta_proj"):
        w = getattr(state, name)
        b, scale = _int8_pack(w)
        header["weights"][name] = {"shape": list(w.shape), "scale": scale, "nbytes": len(b)}
        weight_blobs.append(b)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", len(header_bytes))
    out += header_bytes
    for s in streams:
        out += struct.pack("<I", len(s))
        out += s
    for b in weight_blobs:
        out += b  # length is in the header (nbytes)
    return bytes(out)


def inflate_numpy(blob: bytes) -> CarrierState:
    """Decode an archive blob back into a ``CarrierState`` (pure numpy).

    The portability-contract inflate: no torch. Decodes the range-coded integer
    latents, dequantizes, int8-unpacks the weights, and returns the renderable
    state. ``render_pair_numpy(state, idx)`` then produces the RGB pair.
    """
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    pos = 4
    (hlen,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    header = json.loads(blob[pos : pos + hlen].decode("utf-8"))
    pos += hlen

    grids: list[np.ndarray] = []
    for gi, gmeta in enumerate(header["grids"]):
        (slen,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        coded = blob[pos : pos + slen]
        pos += slen
        cum, total = cumulative_frequencies(gmeta["freqs"])  # noqa: F841 (validated inside)
        syms = decode_static_symbols(coded, count=gmeta["count"], frequencies=gmeta["freqs"])
        arr = np.asarray(syms, dtype=np.int64).reshape(header["grid_shapes"][gi])
        grids.append(dequantize_symbols(arr, gmeta["offset"], header["grid_step"]))

    # delta stream
    (slen,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    coded = blob[pos : pos + slen]
    pos += slen
    dmeta = header["delta"]
    syms = decode_static_symbols(coded, count=dmeta["count"], frequencies=dmeta["freqs"])
    delta = dequantize_symbols(
        np.asarray(syms, dtype=np.int64).reshape(header["delta_shape"]),
        dmeta["offset"],
        header["delta_step"],
    )

    # weights
    wvals: dict[str, np.ndarray] = {}
    for name in ("w1", "b1", "w2", "b2", "delta_proj"):
        wm = header["weights"][name]
        nb = wm["nbytes"]
        wvals[name] = _int8_unpack(blob[pos : pos + nb], tuple(wm["shape"]), wm["scale"])
        pos += nb

    return CarrierState(
        grids=grids,
        frame1_delta=delta,
        w1=wvals["w1"],
        b1=wvals["b1"],
        w2=wvals["w2"],
        b2=wvals["b2"],
        delta_proj=wvals["delta_proj"],
        out_hw=tuple(header["out_hw"]),
        coarse_shape=tuple(header["coarse_shape"]),
    )
