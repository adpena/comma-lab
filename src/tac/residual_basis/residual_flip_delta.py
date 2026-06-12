# SPDX-License-Identifier: MIT
"""Residual argmax-FLIP delta on the Cool-Chic byte-closed base (the Yousfi eureka).

Lane ``lane_cool_chic_residual_flip_delta_20260611``.  The Cool-Chic n48 base
reaches ``d_seg ~= 0.0276`` at ~9458 real bytes — 49x the frontier basin (5.6e-4)
— and training it the rest of the way down hits the below-floor surrogate wall
(``smooth_disagreement`` is noise below d_seg 5e-3).  The Yousfi move bypasses
the convergence wall: STOP the base at d_seg ~= 0.01-0.03 and code ONLY the
sparse set of pixels where the base's RE-SEGMENTED argmax disagrees with ``L*``
(the GT SegNet argmax), via an RGB perturbation that pushes those pixels across
their SegNet decision boundary.

This is a CHANGE TO THE RGB WITNESS, not a label edit.  The delta is added to
the byte-closed render at the witness resolution; the perturbed witness is
re-segmented under the EXACT frozen SegNet through the SAME uint8/resize eval
roundtrip the scorer uses.  d_seg is measured on the RE-SEGMENTED perturbed RGB
— NOT on a relabel.

THE BIGGEST RISK (Yousfi's named risk, validated FIRST in the runner): the
sparse RGB delta must NET-reduce d_seg under the exact re-segmentation without
inducing new OFF-TARGET flips through the resize.  A local pixel nudge at a flip
pixel may not move the SegNet boundary (receptive field), and may push a
neighbouring pixel across its own boundary.  The runner measures the net effect
on the real argmax witness; if new flips cancel the gain, that is a real MEASURED
finding (ANTI-SIGNAL-LOSS: report it honestly).

Byte-close discipline (CLAUDE.md "Bit-level deconstruction and entropy
discipline"): the surviving sparse delta is range-coded (flat (frame,y,x,c)
index gaps + int8 quantized values, per-tensor empirical frequency tables in the
header) into a self-contained blob with a pure-numpy inflate.  The delta bytes
are charged to the contest rate; the cost is REAL, not estimated.

Authority: ``[macOS-CPU advisory]`` — exact frozen scorer but not the contest
600-sample Linux-x86_64 harness, so non-promotable per the GOAL authority ladder.
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

MAGIC = b"RFD1"  # Residual Flip Delta v1


# --------------------------------------------------------------------------
# flip identification + RGB delta construction (on the witness RGB)
# --------------------------------------------------------------------------


@dataclass
class FlipDeltaPlan:
    """Per-pair sparse RGB flip-delta + the measured re-segmentation effect.

    All deltas live at the WITNESS resolution ``(H_w, W_w)`` — the RGB frame that
    is re-segmented (post-interpolate, pre-roundtrip).  ``delta`` is in [0,255]
    units (added to the witness then clipped to [0,255]).
    """

    pair_idx: int
    witness_hw: tuple[int, int]
    # flat (y*W*3 + x*3 + c) indices into the (H_w, W_w, 3) witness, sorted.
    flat_idx: np.ndarray  # int64, shape (k,)
    values: np.ndarray  # float64 [0,255]-unit delta, shape (k,)
    n_flips_before: int  # base argmax != L* pixel count (full 384x512)
    target_flip_pixels: int  # flip pixels the delta touches (at witness res)


def build_appearance_flip_delta(
    base_witness_hwc: np.ndarray,
    gt_witness_hwc: np.ndarray,
    base_argmax_hw: np.ndarray,
    lstar_argmax_hw: np.ndarray,
    *,
    pair_idx: int,
    l_inf_budget: float = 64.0,
    dilate: int = 1,
    strength: float = 1.0,
) -> FlipDeltaPlan:
    """Construct the sparse RGB flip-delta from the GT-base appearance gap.

    The delta is the appearance gap ``GT - base`` (the direction that makes the
    witness look like GT, which segments to L*), restricted to the flip pixels
    (where ``base_argmax != lstar``) optionally DILATED by ``dilate`` pixels (the
    SegNet receptive field means a boundary pixel's class is set by its
    neighbourhood, so we nudge a small tube), scaled by ``strength`` and clipped
    to the per-pixel L-inf budget.

    Args:
        base_witness_hwc / gt_witness_hwc: (H_w, W_w, 3) RGB in [0,255] at the
            witness resolution (the re-segmented frame, post-interpolate).
        base_argmax_hw / lstar_argmax_hw: (384, 512) SegNet argmax maps (the
            scorer always outputs 384x512); flips computed there then resized to
            witness res for delta placement.
        l_inf_budget: hard cap on |delta| per channel in [0,255] units.
        dilate: morphological dilation radius of the flip mask (witness res).
        strength: scalar on the appearance gap (1.0 = full GT-base gap).
    """

    base = np.asarray(base_witness_hwc, dtype=np.float64)
    gt = np.asarray(gt_witness_hwc, dtype=np.float64)
    if base.shape != gt.shape or base.ndim != 3 or base.shape[-1] != 3:
        raise ValueError(f"witness frames must match (H,W,3); got {base.shape} vs {gt.shape}")
    hw = base.shape[0], base.shape[1]

    flip_full = base_argmax_hw != lstar_argmax_hw  # (384,512) bool
    n_flips_before = int(flip_full.sum())

    # Resize the flip mask to witness resolution (nearest, so it stays a mask).
    flip_w = _resize_mask_nearest(flip_full, hw[0], hw[1])
    if dilate > 0:
        flip_w = _dilate_bool(flip_w, dilate)

    gap = (gt - base) * float(strength)  # (H,W,3) the correction direction
    gap = np.clip(gap, -l_inf_budget, l_inf_budget)
    # mask gap to flip tube only
    mask3 = np.repeat(flip_w[:, :, None], 3, axis=2)
    gap = np.where(mask3, gap, 0.0)

    # flatten to sparse (y*W*3 + x*3 + c) indices of non-zero entries.
    nz = np.abs(gap) > 0.5  # below 0.5/255 unit is sub-int8 noise after quant
    ys, xs, cs = np.nonzero(nz)
    flat = (ys * (hw[1] * 3) + xs * 3 + cs).astype(np.int64)
    order = np.argsort(flat)
    flat = flat[order]
    vals = gap[ys, xs, cs][order]

    return FlipDeltaPlan(
        pair_idx=pair_idx,
        witness_hw=hw,
        flat_idx=flat,
        values=vals.astype(np.float64),
        n_flips_before=n_flips_before,
        target_flip_pixels=int(flip_w.sum()),
    )


def apply_flip_delta(witness_hwc: np.ndarray, plan: FlipDeltaPlan) -> np.ndarray:
    """Apply the sparse RGB delta to the witness, clip to [0,255]. Returns a copy."""
    out = np.asarray(witness_hwc, dtype=np.float64).copy()
    h, w, _ = out.shape
    if (h, w) != plan.witness_hw:
        raise ValueError(f"witness {h}x{w} != plan {plan.witness_hw}")
    flat = out.reshape(-1)
    flat[plan.flat_idx] = np.clip(flat[plan.flat_idx] + plan.values, 0.0, 255.0)
    return out


# --------------------------------------------------------------------------
# byte-close (range-coded sparse delta) + numpy inflate
# --------------------------------------------------------------------------


def _quantize_values(values: np.ndarray, l_inf_budget: float) -> tuple[np.ndarray, float]:
    """Int8 quantize the delta values with scale = l_inf_budget/127. Symbols >=0."""
    scale = float(l_inf_budget) / 127.0 if l_inf_budget > 0 else 1.0
    q = np.clip(np.rint(values / scale), -127, 127).astype(np.int64)
    return q, scale


def _histogram_freqs(symbols: np.ndarray, n_symbols: int) -> list[int]:
    counts = np.bincount(symbols.ravel(), minlength=n_symbols).astype(np.int64)
    return np.maximum(counts, 1).tolist()


def byte_close_flip_delta(
    plans: list[FlipDeltaPlan], *, l_inf_budget: float = 64.0
) -> bytes:
    """Range-code all per-pair sparse flip-deltas into a self-contained blob.

    Index gaps (delta-encoded sorted flat indices) and int8 value symbols each
    get a per-stream empirical frequency table in the header (the cost is real).
    """
    header: dict = {
        "version": 1,
        "l_inf_budget": float(l_inf_budget),
        "pairs": [],
        "idxgap_freqs": None,
        "val_offset": None,
        "val_freqs": None,
    }
    all_gaps: list[int] = []
    all_vals: list[int] = []
    val_scale = float(l_inf_budget) / 127.0 if l_inf_budget > 0 else 1.0

    for p in plans:
        if p.flat_idx.size:
            gaps = np.diff(np.concatenate([[0], p.flat_idx])).astype(np.int64)
        else:
            gaps = np.zeros((0,), dtype=np.int64)
        qv, _ = _quantize_values(p.values, l_inf_budget)
        header["pairs"].append(
            {
                "pair_idx": p.pair_idx,
                "witness_hw": list(p.witness_hw),
                "k": int(p.flat_idx.size),
                "n_flips_before": p.n_flips_before,
                "target_flip_pixels": p.target_flip_pixels,
            }
        )
        all_gaps.extend(gaps.tolist())
        all_vals.extend(qv.tolist())

    gaps_arr = np.asarray(all_gaps, dtype=np.int64)
    vals_arr = np.asarray(all_vals, dtype=np.int64)
    # value symbols shifted to non-negative
    val_offset = int(-vals_arr.min()) if vals_arr.size else 0
    val_syms = (vals_arr + val_offset).astype(np.int64)

    gap_nsym = int(gaps_arr.max()) + 1 if gaps_arr.size else 1
    val_nsym = int(val_syms.max()) + 1 if val_syms.size else 1
    gap_freqs = _histogram_freqs(gaps_arr, gap_nsym) if gaps_arr.size else [1]
    val_freqs = _histogram_freqs(val_syms, val_nsym) if val_syms.size else [1]

    header["idxgap_freqs"] = gap_freqs
    header["idxgap_count"] = int(gaps_arr.size)
    header["val_offset"] = val_offset
    header["val_freqs"] = val_freqs
    header["val_count"] = int(val_syms.size)
    header["val_scale"] = val_scale

    gap_stream = (
        encode_static_symbols(gaps_arr.tolist(), frequencies=gap_freqs)
        if gaps_arr.size
        else b""
    )
    val_stream = (
        encode_static_symbols(val_syms.tolist(), frequencies=val_freqs)
        if val_syms.size
        else b""
    )

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", len(header_bytes))
    out += header_bytes
    out += struct.pack("<I", len(gap_stream))
    out += gap_stream
    out += struct.pack("<I", len(val_stream))
    out += val_stream
    return bytes(out)


def inflate_flip_delta(blob: bytes) -> list[FlipDeltaPlan]:
    """Decode the range-coded flip-delta blob into per-pair plans (pure numpy)."""
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    pos = 4
    (hlen,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    header = json.loads(blob[pos : pos + hlen].decode("utf-8"))
    pos += hlen
    (glen,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    gap_stream = blob[pos : pos + glen]
    pos += glen
    (vlen,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    val_stream = blob[pos : pos + vlen]
    pos += vlen

    gaps = (
        np.asarray(
            decode_static_symbols(
                gap_stream, count=header["idxgap_count"], frequencies=header["idxgap_freqs"]
            ),
            dtype=np.int64,
        )
        if header["idxgap_count"]
        else np.zeros((0,), dtype=np.int64)
    )
    val_syms = (
        np.asarray(
            decode_static_symbols(
                val_stream, count=header["val_count"], frequencies=header["val_freqs"]
            ),
            dtype=np.int64,
        )
        if header["val_count"]
        else np.zeros((0,), dtype=np.int64)
    )
    vals = (val_syms - header["val_offset"]).astype(np.float64) * header["val_scale"]

    plans: list[FlipDeltaPlan] = []
    gpos = 0
    for pm in header["pairs"]:
        k = pm["k"]
        gseg = gaps[gpos : gpos + k]
        vseg = vals[gpos : gpos + k]
        gpos += k
        flat = np.cumsum(gseg).astype(np.int64) if k else np.zeros((0,), dtype=np.int64)
        plans.append(
            FlipDeltaPlan(
                pair_idx=pm["pair_idx"],
                witness_hw=tuple(pm["witness_hw"]),
                flat_idx=flat,
                values=vseg,
                n_flips_before=pm["n_flips_before"],
                target_flip_pixels=pm["target_flip_pixels"],
            )
        )
    return plans


# --------------------------------------------------------------------------
# small numpy morphology helpers (no scipy dep)
# --------------------------------------------------------------------------


def _resize_mask_nearest(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize a bool mask to (h, w)."""
    mh, mw = mask.shape
    if (mh, mw) == (h, w):
        return mask.astype(bool)
    ys = (np.arange(h) * mh / h).astype(np.int64).clip(0, mh - 1)
    xs = (np.arange(w) * mw / w).astype(np.int64).clip(0, mw - 1)
    return mask[ys][:, xs].astype(bool)


def _dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-structuring-element binary dilation by ``radius`` (pure numpy)."""
    out = mask.copy()
    for _ in range(int(radius)):
        d = out.copy()
        d[1:, :] |= out[:-1, :]
        d[:-1, :] |= out[1:, :]
        d[:, 1:] |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        out = d
    return out
