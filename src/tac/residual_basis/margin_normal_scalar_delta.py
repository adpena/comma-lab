# SPDX-License-Identifier: MIT
"""Cheap margin-normal SCALAR flip-delta (lever-D reactivation of the v2 repair atom).

The validated surgical repair atom (``tac.optimization.frame1_seg_repair_atoms`` +
``run_residual_flip_delta_surgical_v2``) NET-reduces ``d_seg`` by pushing recoverable
flip pixels back toward the GT class.  But it codes the full per-pixel ``GT - base``
appearance gap as up to THREE int8 RGB channels => ~187 B/flip, 147x the 1.27 B/flip
contest water level => BLOCKED-BY-RATE.

This module codes the SAME boundary-crossing as a MINIMAL margin-normal SCALAR per
support pixel: ONE quantized magnitude along ONE decoder-reconstructable per-pixel
direction (the single dominant-|gap| channel; the dominant-channel index is 2 bits).
No GT is stored — the decoder only needs (flat index, dominant-channel id, signed
magnitude).  Per pixel: ~1 byte magnitude + ~2 bits channel + entropy-coded index gap,
target < 2 B/flip (lever-D territory).

THE OPEN RISK (Yousfi's named risk): does a cheap one-channel scalar nudge still flip
the SegNet argmax through uint8/resize, the way the full 3-channel RGB gap did?  The
runner re-segments the cheap-nudged witness under the EXACT frozen SegNet and measures
the NET d_seg effect + new off-target flips.  If the cheap nudge cannot move the
boundary (the full RGB gap was load-bearing), that is a real MEASURED finding.

Authority: ``[macOS-CPU advisory]`` — exact frozen scorer but not the contest
600-sample Linux-x86_64 harness, non-promotable per the GOAL authority ladder.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

from tac.lossless.range_coder import (
    decode_static_symbols,
    encode_static_symbols,
)

MAGIC = b"MNS1"  # Margin-Normal Scalar delta v1


@dataclass
class MarginNormalScalarPlan:
    """Per-pair cheap margin-normal scalar flip-delta at the witness resolution.

    For each support pixel we store: a flat (y, x) index into the (H_w, W_w) witness,
    a dominant-channel id in {0,1,2} (the RGB channel with the largest |GT-base| gap),
    and a signed magnitude in [0,255] units (the gap projected onto that channel).
    The decoder adds ``magnitude`` to ONLY that channel and clips to [0,255].
    """

    pair_idx: int
    witness_hw: tuple[int, int]
    flat_px: np.ndarray  # int64 (k,) flat y*W + x pixel index, sorted
    chan: np.ndarray  # uint8 (k,) dominant channel id 0..2
    mag: np.ndarray  # float64 (k,) signed [0,255]-unit magnitude
    n_support_pixels: int


def build_margin_normal_scalar_plan(
    correction_hwc: np.ndarray,
    *,
    pair_idx: int,
    witness_hw: tuple[int, int],
    l_inf_budget: float,
) -> MarginNormalScalarPlan:
    """Reduce a canonical (H,W,3) repair-atom correction to ONE scalar per support px.

    The repair atom's ``correction`` is the full per-channel ``correction_fraction *
    (GT - base)`` gap on the support pixels.  We collapse it to the single dominant
    channel (largest |gap|): the minimal margin-normal step is the one-channel nudge
    along the direction that most increases the GT-class logit.  The correction is
    resized to the witness grid first (the frame we perturb + re-segment).
    """

    corr = np.asarray(correction_hwc, dtype=np.float64)
    if corr.ndim != 3 or corr.shape[-1] != 3:
        raise ValueError(f"correction must be (H,W,3); got {corr.shape}")
    h, w = witness_hw
    if corr.shape[:2] != (h, w):
        chans = []
        ph, pw = corr.shape[:2]
        ys = (np.arange(h) * ph / h).astype(np.int64).clip(0, ph - 1)
        xs = (np.arange(w) * pw / w).astype(np.int64).clip(0, pw - 1)
        for c in range(3):
            chans.append(corr[:, :, c][ys][:, xs])
        corr = np.stack(chans, axis=2)

    corr = np.clip(corr, -l_inf_budget, l_inf_budget)
    # dominant channel per pixel = argmax|gap|; its signed value is the scalar.
    absc = np.abs(corr)
    chan = np.argmax(absc, axis=2).astype(np.uint8)  # (h,w)
    ys, xs = np.indices((h, w))
    mag = corr[ys, xs, chan]  # (h,w) the dominant-channel signed gap

    # support = pixels where the dominant magnitude is non-trivial (> 0.5/255 unit).
    support = np.abs(mag) > 0.5
    sy, sx = np.nonzero(support)
    flat = (sy * w + sx).astype(np.int64)
    order = np.argsort(flat)
    flat = flat[order]
    return MarginNormalScalarPlan(
        pair_idx=pair_idx,
        witness_hw=witness_hw,
        flat_px=flat,
        chan=chan[sy, sx][order].astype(np.uint8),
        mag=mag[sy, sx][order].astype(np.float64),
        n_support_pixels=int(flat.size),
    )


def apply_margin_normal_scalar(
    witness_hwc: np.ndarray, plan: MarginNormalScalarPlan
) -> np.ndarray:
    """Apply the one-channel scalar nudge to the witness, clip to [0,255]. Copy."""
    out = np.asarray(witness_hwc, dtype=np.float64).copy()
    h, w, _ = out.shape
    if (h, w) != plan.witness_hw:
        raise ValueError(f"witness {h}x{w} != plan {plan.witness_hw}")
    if plan.flat_px.size:
        ys = plan.flat_px // w
        xs = plan.flat_px % w
        cur = out[ys, xs, plan.chan]
        out[ys, xs, plan.chan] = np.clip(cur + plan.mag, 0.0, 255.0)
    return out


# --------------------------------------------------------------------------
# byte-close: range-coded (index gaps + 2-bit channels + int8 magnitudes)
# --------------------------------------------------------------------------


def _histogram_freqs(symbols: np.ndarray, n_symbols: int) -> list[int]:
    counts = np.bincount(symbols.ravel(), minlength=n_symbols).astype(np.int64)
    return np.maximum(counts, 1).tolist()


# Flat pixel index < H*W <= 384*512 = 196608 < 2^24, so 3 little-endian bytes hold
# any index gap.  Fixed-width keeps the gap-symbol alphabet bounded at 256 (no dense
# freq-table blowup) AND avoids a per-gap length sidecar; small gaps make the high
# byte-planes ~all-zero so the entropy coder spends ~0 bits on them.
_GAP_BYTES = 3


def _gaps_to_byteplanes(gaps: np.ndarray) -> list[int]:
    """Decompose each gap into ``_GAP_BYTES`` little-endian bytes (flat, gap-major)."""
    if gaps.size == 0:
        return []
    g = gaps.astype(np.int64)
    if int(g.max()) >= (1 << (8 * _GAP_BYTES)):
        raise ValueError("index gap exceeds 3-byte range")
    planes = np.stack([(g >> (8 * b)) & 0xFF for b in range(_GAP_BYTES)], axis=1)
    return planes.reshape(-1).astype(np.int64).tolist()


def _byteplanes_to_gaps(byte_syms: list[int], k: int) -> np.ndarray:
    """Reassemble ``k`` gaps from ``k*_GAP_BYTES`` little-endian byte symbols."""
    if k == 0:
        return np.zeros((0,), dtype=np.int64)
    arr = np.asarray(byte_syms[: k * _GAP_BYTES], dtype=np.int64).reshape(k, _GAP_BYTES)
    val = np.zeros((k,), dtype=np.int64)
    for b in range(_GAP_BYTES):
        val |= arr[:, b] << (8 * b)
    return val


def byte_close_margin_normal_scalar(
    plans: list[MarginNormalScalarPlan], *, l_inf_budget: float
) -> bytes:
    """Range-code all per-pair cheap scalar deltas into a self-contained blob.

    Three entropy-coded streams share empirical tables in the header:
      - pixel-index gaps (delta-encoded sorted flat pixel indices), decomposed into
        little-endian base-256 BYTES so the gap alphabet is bounded at 256 (a single
        large gap must NOT blow the dense JSON freq table to ~10^5 entries),
      - dominant-channel ids (3 symbols),
      - int8 signed magnitudes (scale = l_inf/127, shifted non-negative).
    The cost is REAL (every byte charged), not estimated.
    """

    header: dict = {"version": 1, "l_inf_budget": float(l_inf_budget), "pairs": []}
    all_gap_bytes: list[int] = []
    all_chan: list[int] = []
    all_mag: list[int] = []
    mag_scale = float(l_inf_budget) / 127.0 if l_inf_budget > 0 else 1.0

    for p in plans:
        if p.flat_px.size:
            gaps = np.diff(np.concatenate([[0], p.flat_px])).astype(np.int64)
        else:
            gaps = np.zeros((0,), dtype=np.int64)
        qm = np.clip(np.rint(p.mag / mag_scale), -127, 127).astype(np.int64)
        all_gap_bytes.extend(_gaps_to_byteplanes(gaps))
        header["pairs"].append(
            {
                "pair_idx": p.pair_idx,
                "witness_hw": list(p.witness_hw),
                "k": int(p.flat_px.size),
            }
        )
        all_chan.extend(p.chan.astype(np.int64).tolist())
        all_mag.extend(qm.tolist())

    gapb_arr = np.asarray(all_gap_bytes, dtype=np.int64)
    chan_arr = np.asarray(all_chan, dtype=np.int64)
    mag_arr = np.asarray(all_mag, dtype=np.int64)
    mag_off = int(-mag_arr.min()) if mag_arr.size else 0
    mag_syms = (mag_arr + mag_off).astype(np.int64)

    mag_nsym = int(mag_syms.max()) + 1 if mag_syms.size else 1
    gapb_freqs = _histogram_freqs(gapb_arr, 256) if gapb_arr.size else [1] * 256
    chan_freqs = _histogram_freqs(chan_arr, 3) if chan_arr.size else [1, 1, 1]
    mag_freqs = _histogram_freqs(mag_syms, mag_nsym) if mag_syms.size else [1]

    header.update(
        {
            "idxgapbyte_freqs": gapb_freqs,
            "idxgapbyte_count": int(gapb_arr.size),
            "chan_freqs": chan_freqs,
            "chan_count": int(chan_arr.size),
            "mag_offset": mag_off,
            "mag_freqs": mag_freqs,
            "mag_count": int(mag_syms.size),
            "mag_scale": mag_scale,
        }
    )

    gap_stream = (
        encode_static_symbols(gapb_arr.tolist(), frequencies=gapb_freqs)
        if gapb_arr.size
        else b""
    )
    chan_stream = (
        encode_static_symbols(chan_arr.tolist(), frequencies=chan_freqs)
        if chan_arr.size
        else b""
    )
    mag_stream = (
        encode_static_symbols(mag_syms.tolist(), frequencies=mag_freqs)
        if mag_syms.size
        else b""
    )

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", len(header_bytes))
    out += header_bytes
    for s in (gap_stream, chan_stream, mag_stream):
        out += struct.pack("<I", len(s))
        out += s
    return bytes(out)


def inflate_margin_normal_scalar(blob: bytes) -> list[MarginNormalScalarPlan]:
    """Decode the cheap scalar delta blob into per-pair plans (pure numpy)."""
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    pos = 4
    (hlen,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    header = json.loads(blob[pos : pos + hlen].decode("utf-8"))
    pos += hlen

    streams = []
    for _ in range(3):
        (slen,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        streams.append(blob[pos : pos + slen])
        pos += slen
    gap_stream, chan_stream, mag_stream = streams

    gap_bytes = (
        decode_static_symbols(
            gap_stream, count=header["idxgapbyte_count"], frequencies=header["idxgapbyte_freqs"]
        )
        if header["idxgapbyte_count"]
        else []
    )
    chans = (
        np.asarray(
            decode_static_symbols(
                chan_stream, count=header["chan_count"], frequencies=header["chan_freqs"]
            ),
            dtype=np.int64,
        )
        if header["chan_count"]
        else np.zeros((0,), dtype=np.int64)
    )
    mag_syms = (
        np.asarray(
            decode_static_symbols(
                mag_stream, count=header["mag_count"], frequencies=header["mag_freqs"]
            ),
            dtype=np.int64,
        )
        if header["mag_count"]
        else np.zeros((0,), dtype=np.int64)
    )
    mags = (mag_syms - header["mag_offset"]).astype(np.float64) * header["mag_scale"]

    plans: list[MarginNormalScalarPlan] = []
    cur = 0  # cursor over magnitudes/channels
    gb = 0  # cursor over gap bytes
    for pm in header["pairs"]:
        k = pm["k"]
        cseg = chans[cur : cur + k]
        mseg = mags[cur : cur + k]
        cur += k
        # reassemble fixed-width base-256 byteplanes -> gaps -> prefix-sum indices.
        gseg = _byteplanes_to_gaps(gap_bytes[gb : gb + k * _GAP_BYTES], k)
        gb += k * _GAP_BYTES
        flat = np.cumsum(gseg).astype(np.int64) if k else np.zeros((0,), dtype=np.int64)
        plans.append(
            MarginNormalScalarPlan(
                pair_idx=pm["pair_idx"],
                witness_hw=tuple(pm["witness_hw"]),
                flat_px=flat,
                chan=cseg.astype(np.uint8),
                mag=mseg,
                n_support_pixels=int(k),
            )
        )
    return plans
