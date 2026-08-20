# SPDX-License-Identifier: MIT
"""Context-adaptive arithmetic codec for the SegNet argmax partition ``L*``.

This is the TOP-AIML (SOTA) replacement for the prototype's LZMA-over-raw-labels
partition store
(``tac.boundary_math.dense_raster_lzma_baseline.partition_description_bytes``).
The prototype measured 669-873 B/frame -> rate term 0.27-0.35 alone, DEAD on rate.

The partition is a 5-class label map of large smooth regions (road / sky / car /
lane).  A real codec for such a source is a **context-model arithmetic coder**
(JBIG / LOCO-I / CABAC family), not a generic byte compressor.  We code each pixel
``L*[r,c]`` under a causal raster context and range-code it with a real entropy
coder (``constriction`` RangeEncoder over a per-pixel Categorical), so the
achievable length is ``sum_ctx N_ctx * H(p_ctx)`` (the measured floor in
``tac.optimization.partition_contour_entropy``) plus a small transmitted model.

Two context templates (the two structural levers the operator named):

  * SPATIAL  : ctx = (left, up)           -> 5^2 = 25 contexts.  The JBIG/LOCO-I
               causal template; subsumes RLE + most contour coders.  (lever A:
               boundary/contour coding of a smooth partition.)
  * TEMPORAL : ctx = (left, up, prev_pixel) -> 5^3 = 125 contexts.  Adds the
               co-located pixel of the PREVIOUS frame's partition -> exploits the
               heavy dashcam temporal redundancy.  (lever B: temporal / motion-
               free delta of partitions.)

Model handling (the side-info honesty — NO FAKE):

  We build ONE per-context PMF table over the WHOLE partition (all frames), so the
  model is transmitted ONCE and amortized across all frames.  The model is stored
  as per-context integer counts (uint16, clamped), brotli-compressed; those bytes
  are COUNTED in the codec's total (no free model).  The decoder rebuilds the
  identical Categorical rows from the transmitted counts, so the static-context
  coder is exactly reversible without a per-symbol adaptive loop (which is what
  lets the constriction rank-2 vectorized encode/decode run fast on CPU).

NO FAKE contract (asserted by tests):

  * ``encode_partition_stack`` -> ``decode_partition_stack`` reconstructs the input
    label stack BIT-EXACT (lossless mode).
  * ``PartitionStackCode.total_bytes`` is the REAL length of the emitted byte
    payload (range-coder words + brotli'd model + tiny header), never an estimate.
  * the coder never assigns zero probability to an in-alphabet symbol (constriction
    floors it), so any label stack in ``[0, n_classes)`` is encodable.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import brotli
import constriction
import numpy as np

N_SEG_CLASSES = 5
_MODEL_COUNT_DTYPE = np.uint16
_MODEL_COUNT_MAX = 65535
_MAGIC = b"CPC1"  # context-partition-codec v1


# ---------------------------------------------------------------------------
# Context construction (causal raster templates)
# ---------------------------------------------------------------------------

def _spatial_context_ids(frame: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-pixel context id ``left*n + up`` for one frame (out-of-frame = class 0).

    Causal: only already-decoded neighbours (left, up) are used, so the decoder can
    reproduce the same context as it reconstructs the raster.
    """
    a = frame.astype(np.int64)
    left = np.zeros_like(a)
    left[:, 1:] = a[:, :-1]
    up = np.zeros_like(a)
    up[1:, :] = a[:-1, :]
    return left * n_classes + up


def _temporal_context_ids(
    frame: np.ndarray, prev_frame: np.ndarray | None, n_classes: int
) -> np.ndarray:
    """Per-pixel context id ``(left*n + up)*n + prev`` (prev out-of-stream = 0)."""
    a = frame.astype(np.int64)
    left = np.zeros_like(a)
    left[:, 1:] = a[:, :-1]
    up = np.zeros_like(a)
    up[1:, :] = a[:-1, :]
    prev = np.zeros_like(a) if prev_frame is None else prev_frame.astype(np.int64)
    return (left * n_classes + up) * n_classes + prev


def _n_contexts(template: str, n_classes: int) -> int:
    if template == "spatial":
        return n_classes * n_classes
    if template == "temporal":
        return n_classes * n_classes * n_classes
    raise ValueError(f"unknown template {template!r} (want 'spatial'|'temporal')")


def _build_context_ids(
    frames: list[np.ndarray], template: str, n_classes: int
) -> list[np.ndarray]:
    """Build the per-pixel context-id array for every frame (causal; decoder-replayable)."""
    out: list[np.ndarray] = []
    for fi, f in enumerate(frames):
        if template == "spatial":
            out.append(_spatial_context_ids(f, n_classes))
        else:
            prev = None if fi == 0 else frames[fi - 1]
            out.append(_temporal_context_ids(f, prev, n_classes))
    return out


# ---------------------------------------------------------------------------
# Model (per-context PMF) — built once over the whole stack, transmitted once
# ---------------------------------------------------------------------------

def _build_model_counts(
    frames: list[np.ndarray],
    ctx_ids: list[np.ndarray],
    template: str,
    n_classes: int,
) -> np.ndarray:
    """Joint histogram ``counts[ctx, symbol]`` over ALL pixels of ALL frames.

    Returns a ``(n_contexts, n_classes)`` int64 count table.  This is the shared
    model the coder transmits once and uses for every frame.
    """
    n_ctx = _n_contexts(template, n_classes)
    counts = np.zeros(n_ctx * n_classes, dtype=np.int64)
    for f, ctx in zip(frames, ctx_ids, strict=True):
        combined = ctx.reshape(-1) * n_classes + f.reshape(-1).astype(np.int64)
        binc = np.bincount(combined, minlength=n_ctx * n_classes)
        counts += binc[: n_ctx * n_classes]
    return counts.reshape(n_ctx, n_classes)


def _quantize_model_counts(counts: np.ndarray) -> np.ndarray:
    """Clamp per-context counts to uint16 so the transmitted model is compact.

    Scales each row so its max fits in uint16 while keeping all observed symbols at
    >= 1 (so the decoder's Categorical never zeroes an in-alphabet symbol).  Rows
    with no observations stay all-zero and the decoder uses a uniform fallback
    (matching the encoder), so empty contexts cost no model bytes after brotli.
    """
    out = np.zeros_like(counts, dtype=_MODEL_COUNT_DTYPE)
    for i in range(counts.shape[0]):
        row = counts[i].astype(np.float64)
        rmax = row.max()
        if rmax <= 0:
            continue  # empty context — uniform fallback on both sides
        if rmax <= _MODEL_COUNT_MAX:
            q = np.maximum(row, (row > 0)).astype(np.float64)
        else:
            scale = _MODEL_COUNT_MAX / rmax
            q = np.round(row * scale)
            q = np.where(row > 0, np.maximum(q, 1.0), 0.0)
        out[i] = np.clip(q, 0, _MODEL_COUNT_MAX).astype(_MODEL_COUNT_DTYPE)
    return out


def _row_probabilities(model_counts_u16: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert quantized counts -> normalized per-context probability rows (float64).

    Empty contexts (all-zero row) become uniform.  Both encoder and decoder call
    this identically, so the probability tables match bit-for-bit.
    """
    p = model_counts_u16.astype(np.float64)
    row_sums = p.sum(axis=1, keepdims=True)
    empty = row_sums[:, 0] <= 0
    p[empty] = 1.0
    row_sums = p.sum(axis=1, keepdims=True)
    p = p / row_sums
    # constriction floors zero probs; keep a tiny epsilon for numerical safety.
    return np.maximum(p, 1e-12)


# ---------------------------------------------------------------------------
# Encoded artifact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartitionStackCode:
    """A reversible, byte-exact encoding of a whole partition stack.

    ``payload`` is the full self-describing byte stream (header + brotli'd model +
    range-coder words).  ``total_bytes`` is its real length; ``bytes_per_frame`` is
    the amortized cost.  ``model_bytes`` / ``stream_bytes`` break that down.
    """

    payload: bytes
    n_frames: int
    shape: tuple[int, int]
    n_classes: int
    template: str
    model_bytes: int
    stream_bytes: int

    @property
    def total_bytes(self) -> int:
        return len(self.payload)

    @property
    def bytes_per_frame(self) -> float:
        return self.total_bytes / max(1, self.n_frames)


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def encode_partition_stack(
    frames: list[np.ndarray],
    *,
    n_classes: int = N_SEG_CLASSES,
    template: str = "temporal",
) -> PartitionStackCode:
    """Encode a stack of partitions with a shared context-model range coder.

    All frames must share one ``(H, W)`` shape and have labels in ``[0, n_classes)``.
    The model is built over the whole stack and transmitted once (brotli'd counts);
    every frame is then vectorized range-coded under its causal context.
    """
    if not frames:
        raise ValueError("frames is empty")
    shape = frames[0].shape
    for f in frames:
        a = np.asarray(f)
        if a.shape != shape:
            raise ValueError(f"all frames must share shape {shape}; got {a.shape}")
        if a.ndim != 2:
            raise ValueError(f"each frame must be (H, W); got {a.shape}")
        if a.min() < 0 or a.max() >= n_classes:
            raise ValueError(f"labels must be in [0,{n_classes})")
    frames = [np.asarray(f).astype(np.int64) for f in frames]

    ctx_ids = _build_context_ids(frames, template, n_classes)
    counts = _build_model_counts(frames, ctx_ids, template, n_classes)
    model_u16 = _quantize_model_counts(counts)
    probs_by_ctx = _row_probabilities(model_u16, n_classes)

    # Single range encoder over the whole stack (one tail, like PR103 trick #6).
    enc = constriction.stream.queue.RangeEncoder()
    cat = constriction.stream.model.Categorical(perfect=False)
    for f, ctx in zip(frames, ctx_ids, strict=True):
        syms = f.reshape(-1).astype(np.int32)
        probs = probs_by_ctx[ctx.reshape(-1)]  # (n_pixels, n_classes) float64
        enc.encode(syms, cat, probs)
    stream_bytes = enc.get_compressed().tobytes()

    # Transmit the model: brotli over the quantized uint16 count table.
    model_raw = model_u16.tobytes()
    model_blob = brotli.compress(model_raw, quality=11)

    # Self-describing header.
    template_code = 0 if template == "spatial" else 1
    header = _MAGIC + struct.pack(
        "<BBHHIII",
        n_classes,
        template_code,
        shape[0],
        shape[1],
        len(frames),
        len(model_blob),
        len(stream_bytes),
    )
    payload = header + model_blob + stream_bytes
    return PartitionStackCode(
        payload=payload,
        n_frames=len(frames),
        shape=(int(shape[0]), int(shape[1])),
        n_classes=n_classes,
        template=template,
        model_bytes=len(model_blob),
        stream_bytes=len(stream_bytes),
    )


_HEADER_FMT = "<BBHHIII"
_HEADER_LEN = len(_MAGIC) + struct.calcsize(_HEADER_FMT)


def decode_partition_stack(payload: bytes) -> list[np.ndarray]:
    """Decode a :func:`encode_partition_stack` payload to the bit-exact label stack.

    The decoder rebuilds the SAME shared model from the transmitted counts, then
    reconstructs the raster causally so each pixel's context matches the encoder.
    Because the context uses only already-decoded (left, up, prev-frame) pixels,
    the decode is exact.
    """
    if payload[: len(_MAGIC)] != _MAGIC:
        raise ValueError("bad magic; not a context-partition-codec payload")
    off = len(_MAGIC)
    n_classes, template_code, h, w, n_frames, model_len, stream_len = struct.unpack(
        _HEADER_FMT, payload[off : off + struct.calcsize(_HEADER_FMT)]
    )
    off += struct.calcsize(_HEADER_FMT)
    template = "spatial" if template_code == 0 else "temporal"
    model_blob = payload[off : off + model_len]
    off += model_len
    stream_bytes = payload[off : off + stream_len]
    off += stream_len
    if off != len(payload):
        raise ValueError(f"trailing bytes: parsed {off} of {len(payload)}")

    n_ctx = _n_contexts(template, n_classes)
    model_raw = brotli.decompress(model_blob)
    model_u16 = np.frombuffer(model_raw, dtype=_MODEL_COUNT_DTYPE).reshape(n_ctx, n_classes)
    probs_by_ctx = _row_probabilities(model_u16, n_classes)

    buf = np.frombuffer(stream_bytes, dtype=np.uint32)
    dec = constriction.stream.queue.RangeDecoder(buf)
    cat = constriction.stream.model.Categorical(perfect=False)

    frames: list[np.ndarray] = []
    n_pixels = h * w
    for fi in range(n_frames):
        # Reconstruct this frame row-by-row so context (left,up,prev) is causal.
        # We can decode a full frame in one vectorized call ONLY if we know the
        # context of every pixel up-front — but left/up depend on THIS frame's
        # not-yet-decoded pixels.  We therefore decode row-by-row: each row's
        # context depends only on the row above (already decoded) and the pixel
        # to the left (decoded earlier in the same row).  Within a row, the
        # left-neighbour dependency forces a per-pixel decode for correctness.
        frame = np.zeros((h, w), dtype=np.int64)
        prev = frames[fi - 1] if (template == "temporal" and fi > 0) else None
        for r in range(h):
            up_row = frame[r - 1] if r > 0 else np.zeros(w, dtype=np.int64)
            prev_row = prev[r] if prev is not None else np.zeros(w, dtype=np.int64)
            left = 0
            for c in range(w):
                if template == "spatial":
                    ctx = left * n_classes + int(up_row[c])
                else:
                    ctx = (left * n_classes + int(up_row[c])) * n_classes + int(prev_row[c])
                sym = int(np.asarray(dec.decode(cat, probs_by_ctx[ctx : ctx + 1])).reshape(-1)[0])
                frame[r, c] = sym
                left = sym
        frames.append(frame)
    if sum(f.size for f in frames) != n_frames * n_pixels:
        raise ValueError("decoded pixel count mismatch")
    return frames


def partition_stack_bytes_per_frame(
    frames: list[np.ndarray],
    *,
    n_classes: int = N_SEG_CLASSES,
    template: str = "temporal",
) -> float:
    """Convenience: real amortized B/frame of the context-coded partition stack."""
    return encode_partition_stack(
        frames, n_classes=n_classes, template=template
    ).bytes_per_frame


__all__ = [
    "N_SEG_CLASSES",
    "PartitionStackCode",
    "decode_partition_stack",
    "encode_partition_stack",
    "partition_stack_bytes_per_frame",
]
