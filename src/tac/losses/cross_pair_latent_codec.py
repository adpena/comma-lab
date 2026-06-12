# SPDX-License-Identifier: MIT
"""Cross-pair latent codec (ITEM D1) — measure-and-select cross-pair-redundancy coding.

The base_ch=20 vehicle stores latents as ``(n_pairs, latent_dim)`` — one quantized
per-frame-PAIR code per row (HNeRV-parity L19). The vendored latent path
(``codec.encode_latents``) does per-dim asymmetric uint8 + 1st-order temporal delta +
zigzag + lo/hi byte split + brotli (HNeRV-parity L25/L24-flavor). This module is the
**carrier-agnostic** cross-pair-redundancy lever: it tries SEVERAL alternative encodings
that exploit cross-pair structure the vendored 1st-order delta leaves on the table, then
**selects whichever round-trips bit-exact at the smallest byte count** — with the vendored
format ALWAYS in the candidate set as the byte-identical baseline.

THE MEASURED REALITY (honest, on the three real base_ch20 latent tensors — see the landing
memo): on THESE latents the vendored 1st-order-delta + lo/hi + brotli is within ~1.3% of the
symbol-entropy floor and BEATS every alternative we measured:
  * cross-pair exact dedup: 0 exact dups (600 -> 600 unique) — dead on these latents;
  * shared-codebook / VQ (K=64..256): +1.5 KB (codebook overhead > savings);
  * 2nd-order / motion-predicted delta: +1.7 KB (latents are not temporally smooth);
  * static global range coder on delta symbols: +~100 B (explicit freq-table side cost
    exceeds brotli's online-adaptive model);
  * per-dim range coder: +4 KB (28× freq-table overhead).
So on these latents the SELECTOR picks the vendored format and the deployed delta is **0 B**
(byte-identical archive). That is an HONEST measured negative for the cross-pair-redundancy
thesis on the current latents — NOT a failure: the apparatus is real, the selection is
data-driven, and the negative is LATENT-SPECIFIC (it can flip on Track B or on future latents
with genuine temporal structure, where the dedup/VQ candidates would win and the selector
would pick them automatically). The default-preserving guard makes wiring it cost nothing.

CARRIER-AGNOSTIC CONTRACT: the CORE (``encode_latents_best`` / ``decode_latents_best`` and the
per-candidate encoders) imports NO base_ch20 constant. It operates on a generic
``(n_pairs, latent_dim)`` float tensor + an optional config. The thin base_ch20 ADAPTER is
``build_latent_blob_dedup_or_vendored`` (returns the EXACT vendored ``encode_latents`` payload
bytes — wrapped in brotli by the caller — when no candidate beats vendored, so the archive is
byte-identical; only emits the new framed format when a candidate ACTUALLY saves bytes).

NO-FAKE discipline (verified in ``src/tac/tests/test_cross_pair_latent_codec.py``): this module
ACTUALLY quantizes, ACTUALLY dedups/codebooks/delta-codes, and round-trips the QUANTIZED codes
BIT-EXACT for every format. A test that passes if a codec returns its input unchanged is
worthless; the tests prove decode(encode(x)) reproduces the quantized codes exactly AND that a
no-op codec would FAIL the byte-savings + round-trip assertions.

Score-claim discipline: this changes archive bytes directly, so a measured byte delta is a real
deployed byte delta — but a SCORE claim still requires the dual CPU/CUDA 600-pair exact eval
(ITEM E). ``SCORE_CLAIM = False``.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np
import torch

# Format flags (the 1-byte framing header on the cross-pair path; the vendored path is
# UNFRAMED and byte-identical to ``codec.encode_latents``).
FORMAT_VENDORED = 0  # sentinel only — the vendored path never writes a flag (unframed).
FORMAT_FRAMED_DELTA = 1  # framed: explicit per-dim min/scale + 1st-order delta (self-coded).
FORMAT_DEDUP = 2  # cross-pair exact dedup: unique rows + per-pair index into them.
FORMAT_CODEBOOK = 3  # shared codebook (VQ) + per-pair index + temporal-delta residual.

# The vendored asymmetric-uint8 quant grid (rows map to [0, _QUANT_MAX]).
_QUANT_MAX = 254
_SCALE_FLOOR = 1e-10


@dataclass(frozen=True)
class CrossPairLatentConfig:
    """Configuration for the cross-pair latent codec.

    ``quant_max`` is the vendored uint8 grid ceiling (254). ``codebook_sizes`` are the VQ
    candidate sizes tried by the selector (the best is kept only if it beats vendored).
    ``kmeans_iters`` / ``kmeans_seed`` make the VQ candidate DETERMINISTIC (reproducible
    byte counts — required for the default-preserving guarantee + deterministic tests).
    """

    quant_max: int = _QUANT_MAX
    codebook_sizes: tuple[int, ...] = (64, 128, 256)
    kmeans_iters: int = 8
    kmeans_seed: int = 0


# ============================================================================
# Quantization (mirrors the vendored per-dim asymmetric uint8 grid exactly).
# ============================================================================

def quantize_pairs(
    latents: torch.Tensor, config: CrossPairLatentConfig | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-dim asymmetric uint8 quant — the vendored lossy step, returned as (q, mins, scales).

    ``q`` is ``(n, d)`` uint8 in ``[0, quant_max]``; ``mins`` / ``scales`` are ``(d,)`` float16
    arrays (the side info the dequant needs). This is byte-for-byte the vendored
    ``encode_latents`` quant (same min/max/254 formula) so the QUANTIZED codes — and therefore
    the reconstructed latents — are IDENTICAL across every format this module emits.
    """
    cfg = config or CrossPairLatentConfig()
    t = latents.detach().cpu().float()
    mins = t.min(dim=0).values
    maxs = t.max(dim=0).values
    scales = ((maxs - mins) / float(cfg.quant_max)).clamp(min=_SCALE_FLOOR)
    q = (
        ((t - mins.unsqueeze(0)) / scales.unsqueeze(0))
        .round()
        .clamp(0, cfg.quant_max)
        .to(torch.uint8)
        .numpy()
    )
    mins_f16 = mins.to(torch.float16).numpy()
    scales_f16 = scales.to(torch.float16).numpy()
    return q, mins_f16, scales_f16


def dequantize_pairs(
    q: np.ndarray, mins_f16: np.ndarray, scales_f16: np.ndarray
) -> torch.Tensor:
    """Inverse of :func:`quantize_pairs` — returns ``(n, d)`` float32 latents.

    ``float(q) * scale + min`` with the side info promoted to float32 — the SAME dequant math
    the vendored ``decode_latents`` uses, so a round-trip through THIS module reproduces the
    vendored-reconstructed latents bit-exactly.
    """
    mins = torch.from_numpy(mins_f16.astype(np.float16)).float()
    scales = torch.from_numpy(scales_f16.astype(np.float16)).float()
    return torch.from_numpy(q.astype(np.float32)) * scales.unsqueeze(0) + mins.unsqueeze(0)


# ============================================================================
# Side info + zigzag helpers (shared by the framed candidates).
# ============================================================================

def _pack_side(n: int, d: int, mins_f16: np.ndarray, scales_f16: np.ndarray) -> bytes:
    return (
        struct.pack("<II", n, d)
        + mins_f16.astype(np.float16).tobytes()
        + scales_f16.astype(np.float16).tobytes()
    )


def _unpack_side(buf: io.BytesIO) -> tuple[int, int, np.ndarray, np.ndarray]:
    n, d = struct.unpack("<II", buf.read(8))
    mins = np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()
    scales = np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()
    return n, d, mins, scales


def _zigzag_i16_to_u16(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.int32)
    return np.where(a >= 0, 2 * a, -2 * a - 1).astype(np.uint16)


def _zigzag_u16_to_i16(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.int32)
    return np.where(a % 2 == 0, a // 2, -(a // 2) - 1).astype(np.int16)


def _delta_lohi_bytes(delta: np.ndarray) -> bytes:
    """1st-order delta array (n, d) int16 -> zigzag uint16 -> lo+hi byte split (vendored trick)."""
    dz = _zigzag_i16_to_u16(delta)
    lo = (dz & 0xFF).astype(np.uint8).tobytes()
    hi = (dz >> 8).astype(np.uint8).tobytes()
    return lo + hi


def _undo_delta_lohi(raw: bytes, n: int, d: int) -> np.ndarray:
    """Inverse of :func:`_delta_lohi_bytes` -> 1st-order delta array (n, d) int16."""
    total = n * d
    lo = np.frombuffer(raw[:total], dtype=np.uint8).astype(np.uint16)
    hi = np.frombuffer(raw[total : 2 * total], dtype=np.uint8).astype(np.uint16)
    dz = ((hi << 8) | lo).reshape(n, d)
    return _zigzag_u16_to_i16(dz)


def _temporal_delta(q: np.ndarray) -> np.ndarray:
    """1st-order temporal delta of a (n, d) integer array -> int16 (row 0 is the raw row)."""
    delta = np.empty_like(q, dtype=np.int16)
    delta[0] = q[0].astype(np.int16)
    delta[1:] = q[1:].astype(np.int16) - q[:-1].astype(np.int16)
    return delta


def _undo_temporal_delta(delta: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Prefix-sum inverse of :func:`_temporal_delta`."""
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, delta.shape[0]):
        q[i] = q[i - 1] + delta[i]
    return q.astype(dtype)


# ============================================================================
# Candidate encoders (each returns the PRE-brotli payload bytes; caller brotli-wraps).
# Every candidate round-trips the QUANTIZED codes ``q`` bit-exact.
# ============================================================================

def _encode_framed_delta(q: np.ndarray, mins_f16: np.ndarray, scales_f16: np.ndarray) -> bytes:
    """FORMAT_FRAMED_DELTA: 1-byte flag + side + 1st-order delta lo/hi (self-coded vendored).

    This is the vendored encoding re-expressed in the framed grammar (with the 1-byte flag).
    It is +1 byte vs the unframed vendored payload (the flag) — it exists so the unified
    decoder has a self-consistent framed path; the byte-identical-to-vendored guarantee lives
    in the builder, which emits the UNFRAMED vendored bytes on the no-win path.
    """
    n, d = q.shape
    delta = _temporal_delta(q)
    out = io.BytesIO()
    out.write(struct.pack("<B", FORMAT_FRAMED_DELTA))
    out.write(_pack_side(n, d, mins_f16, scales_f16))
    out.write(_delta_lohi_bytes(delta))
    return out.getvalue()


def _encode_dedup(q: np.ndarray, mins_f16: np.ndarray, scales_f16: np.ndarray) -> bytes:
    """FORMAT_DEDUP: collapse identical pair-codes to a unique table + per-pair index.

    Wins when adjacent (or repeated) pairs share an EXACT quantized code. Stored as:
      flag + side + (n_unique:u32) + unique rows (1st-order-delta lo/hi over the unique table)
      + per-pair index (uint16/uint32) — the index stream is itself temporal-delta-zigzag'd so
      a run of repeats compresses to near-zero under brotli.
    """
    n, d = q.shape
    uniq, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.astype(np.int64)
    n_uniq = uniq.shape[0]
    # Unique table coded with the same 1st-order-delta-lohi trick (over the unique rows).
    uniq_delta = _temporal_delta(uniq)
    uniq_blob = _delta_lohi_bytes(uniq_delta)
    # Index stream: temporal delta (zigzag) so repeat-runs -> 0; width chosen by n_uniq.
    idx_dtype = np.uint16 if n_uniq <= 0xFFFF else np.uint32
    idx_width = 2 if idx_dtype is np.uint16 else 4
    idx_delta = np.empty(n, dtype=np.int64)
    idx_delta[0] = inv[0]
    idx_delta[1:] = inv[1:] - inv[:-1]
    idx_zz = np.where(idx_delta >= 0, 2 * idx_delta, -2 * idx_delta - 1).astype(idx_dtype)
    out = io.BytesIO()
    out.write(struct.pack("<B", FORMAT_DEDUP))
    out.write(_pack_side(n, d, mins_f16, scales_f16))
    out.write(struct.pack("<IB", n_uniq, idx_width))
    out.write(uniq_blob)
    out.write(idx_zz.tobytes())
    return out.getvalue()


def _kmeans_uint8(
    q: np.ndarray, k: int, iters: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic Lloyd k-means on uint8 pair-codes. Returns (codebook_uint8 (k,d), assign)."""
    n, d = q.shape
    k = min(k, n)
    rng = np.random.default_rng(seed)
    init = rng.choice(n, size=k, replace=False)
    cb = q[init].astype(np.float64)
    assign = np.zeros(n, dtype=np.int64)
    qf = q.astype(np.float64)
    for _ in range(iters):
        # (n, k) squared distances; argmin assignment.
        dist = ((qf[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
        assign = dist.argmin(1)
        for c in range(k):
            m = assign == c
            if m.any():
                cb[c] = qf[m].mean(0)
    cbq = np.round(cb).clip(0, _QUANT_MAX).astype(np.uint8)
    # Recompute assignment against the QUANTIZED codebook (the deployed one).
    dist = ((qf[:, None, :] - cbq.astype(np.float64)[None, :, :]) ** 2).sum(-1)
    assign = dist.argmin(1).astype(np.int64)
    return cbq, assign


def _encode_codebook(
    q: np.ndarray, mins_f16: np.ndarray, scales_f16: np.ndarray, k: int, cfg: CrossPairLatentConfig
) -> bytes:
    """FORMAT_CODEBOOK: VQ codebook + per-pair index + temporal-delta residual (exact).

    The residual ``q - codebook[assign]`` is stored EXACTLY (temporal-delta lo/hi), so this
    round-trips ``q`` bit-exact regardless of codebook quality — the codebook is a PREDICTOR,
    the residual carries the exact remainder. Wins only when the residual entropy + codebook +
    index < the plain delta blob (rare; measured to LOSE on the current latents).
    """
    n, d = q.shape
    cbq, assign = _kmeans_uint8(q, k, cfg.kmeans_iters, cfg.kmeans_seed)
    k_eff = cbq.shape[0]
    recon = cbq[assign]
    resid = q.astype(np.int16) - recon.astype(np.int16)  # exact residual
    resid_delta = _temporal_delta(resid)  # 1st-order over residual rows
    resid_blob = _delta_lohi_bytes(resid_delta)
    idx_dtype = np.uint16 if k_eff <= 0xFFFF else np.uint32
    idx_width = 2 if idx_dtype is np.uint16 else 4
    out = io.BytesIO()
    out.write(struct.pack("<B", FORMAT_CODEBOOK))
    out.write(_pack_side(n, d, mins_f16, scales_f16))
    out.write(struct.pack("<IB", k_eff, idx_width))
    out.write(cbq.tobytes())  # (k_eff, d) uint8
    out.write(assign.astype(idx_dtype).tobytes())
    out.write(resid_blob)
    return out.getvalue()


# ============================================================================
# Candidate decoders (inverse of each framed encoder) -> reconstructed QUANT codes q.
# ============================================================================

def _decode_framed_delta(buf: io.BytesIO) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d, mins, scales = _unpack_side(buf)
    delta = _undo_delta_lohi(buf.read(2 * n * d), n, d)
    q = _undo_temporal_delta(delta, dtype=np.uint8)
    return q, mins, scales


def _decode_dedup(buf: io.BytesIO) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d, mins, scales = _unpack_side(buf)
    n_uniq, idx_width = struct.unpack("<IB", buf.read(5))
    uniq_delta = _undo_delta_lohi(buf.read(2 * n_uniq * d), n_uniq, d)
    uniq = _undo_temporal_delta(uniq_delta, dtype=np.uint8)
    idx_dtype = np.uint16 if idx_width == 2 else np.uint32
    idx_zz = np.frombuffer(buf.read(n * idx_width), dtype=idx_dtype).astype(np.int64)
    idx_delta = np.where(idx_zz % 2 == 0, idx_zz // 2, -(idx_zz // 2) - 1)
    inv = np.cumsum(idx_delta).astype(np.int64)
    q = uniq[inv]
    return q, mins, scales


def _decode_codebook(buf: io.BytesIO) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d, mins, scales = _unpack_side(buf)
    k_eff, idx_width = struct.unpack("<IB", buf.read(5))
    cbq = np.frombuffer(buf.read(k_eff * d), dtype=np.uint8).reshape(k_eff, d)
    idx_dtype = np.uint16 if idx_width == 2 else np.uint32
    assign = np.frombuffer(buf.read(n * idx_width), dtype=idx_dtype).astype(np.int64)
    resid_delta = _undo_delta_lohi(buf.read(2 * n * d), n, d)
    resid = _undo_temporal_delta(resid_delta, dtype=np.int16)
    q = (cbq[assign].astype(np.int16) + resid).astype(np.uint8)
    return q, mins, scales


# ============================================================================
# CORE public API: measure-and-select encode + unified decode.
# ============================================================================

def _candidate_payloads(
    q: np.ndarray,
    mins_f16: np.ndarray,
    scales_f16: np.ndarray,
    cfg: CrossPairLatentConfig,
) -> dict[int, bytes]:
    """Build every framed candidate PRE-brotli payload, keyed by format flag."""
    cands: dict[int, bytes] = {
        FORMAT_FRAMED_DELTA: _encode_framed_delta(q, mins_f16, scales_f16),
        FORMAT_DEDUP: _encode_dedup(q, mins_f16, scales_f16),
    }
    for k in cfg.codebook_sizes:
        payload = _encode_codebook(q, mins_f16, scales_f16, k, cfg)
        # Keep the single best codebook candidate (smallest pre-brotli payload as a cheap
        # pre-filter; final selection re-measures post-brotli in :func:`encode_latents_best`).
        prev = cands.get(FORMAT_CODEBOOK)
        if prev is None or len(payload) < len(prev):
            cands[FORMAT_CODEBOOK] = payload
    return cands


# The STRUCTURAL formats — those that exploit genuine cross-pair redundancy (dedup, codebook).
# FORMAT_FRAMED_DELTA is information-EQUIVALENT to the vendored encoding (same per-dim quant +
# 1st-order delta), so any byte difference between it and the vendored blob is pure brotli
# block-ALIGNMENT noise from the 1-byte flag — NOT a real reduction. The adapter must never let
# that alignment artifact masquerade as a cross-pair-redundancy win (R8-style instrument-bug
# discipline). Only a STRUCTURAL candidate strictly smaller than vendored is a real win.
_STRUCTURAL_FORMATS = (FORMAT_DEDUP, FORMAT_CODEBOOK)


def encode_latents_best(
    latents: torch.Tensor,
    config: CrossPairLatentConfig | None = None,
    *,
    structural_only: bool = False,
) -> tuple[bytes, int]:
    """CORE measure-and-select: encode ``(n, d)`` latents at the smallest POST-brotli framed size.

    Returns ``(framed_payload_bytes, format_flag)`` — the PRE-brotli framed payload of the
    WINNING candidate (the caller brotli-wraps it, exactly like the vendored path) and the flag.
    Every candidate round-trips the quantized codes bit-exact; the selector compares their
    POST-brotli sizes (the deployed cost) and returns the smallest.

    ``structural_only=True`` restricts the selectable winners to the STRUCTURAL formats
    (``_STRUCTURAL_FORMATS`` — dedup / codebook), falling back to FRAMED_DELTA only as the
    decoder-consistent baseline when no structural candidate is smaller. The ADAPTER uses this
    so a brotli block-ALIGNMENT artifact (FRAMED_DELTA vs vendored, which carry identical
    information) can never be reported as a cross-pair-redundancy win. The default
    (``structural_only=False``) is the pure smallest-framed selection for agnostic callers.

    Carrier-agnostic: imports no base_ch20 constant. Reusable on any ``(n_pairs, latent_dim)``.
    """
    import brotli

    cfg = config or CrossPairLatentConfig()
    q, mins_f16, scales_f16 = quantize_pairs(latents, cfg)
    cands = _candidate_payloads(q, mins_f16, scales_f16, cfg)
    # The FRAMED_DELTA baseline is always the decoder-consistent fallback.
    best_flag = FORMAT_FRAMED_DELTA
    best_payload = cands[FORMAT_FRAMED_DELTA]
    best_size = len(brotli.compress(best_payload, quality=11))
    selectable = _STRUCTURAL_FORMATS if structural_only else tuple(cands)
    for flag in selectable:
        if flag not in cands:
            continue
        size = len(brotli.compress(cands[flag], quality=11))
        if size < best_size:
            best_size, best_flag, best_payload = size, flag, cands[flag]
    return best_payload, best_flag


def decode_latents_best(framed_payload: bytes) -> torch.Tensor:
    """Unified decoder for every FRAMED format this module emits -> ``(n, d)`` float32 latents.

    Reads the 1-byte format flag and dispatches. Does NOT decode the UNFRAMED vendored payload
    (that is read by ``codec.decode_latents``); the adapter records which format it emitted.
    """
    buf = io.BytesIO(framed_payload)
    flag = struct.unpack("<B", buf.read(1))[0]
    if flag == FORMAT_FRAMED_DELTA:
        q, mins, scales = _decode_framed_delta(buf)
    elif flag == FORMAT_DEDUP:
        q, mins, scales = _decode_dedup(buf)
    elif flag == FORMAT_CODEBOOK:
        q, mins, scales = _decode_codebook(buf)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown cross-pair latent format flag {flag}")
    return dequantize_pairs(q, mins, scales)


# ============================================================================
# Thin base_ch20 ADAPTER: default-preserving builder over the vendored grammar.
# ============================================================================

def build_latent_blob_dedup_or_vendored(
    latents: torch.Tensor, config: CrossPairLatentConfig | None = None
) -> tuple[bytes, bool]:
    """ADAPTER: build the latents blob, returning ``(latents_brotli, is_framed_format)``.

    DEFAULT-PRESERVING: brotli-wraps BOTH the unframed vendored payload (from
    ``codec.encode_latents``) AND the best framed candidate, then returns the SMALLER. On a tie
    or no-win — which is the MEASURED reality on the current base_ch20 latents — it returns the
    EXACT vendored brotli bytes (``is_framed_format=False``), so the archive is byte-identical
    to a vendored build. Only when a framed candidate ACTUALLY saves bytes does it return the
    framed blob (``is_framed_format=True``). The caller persists the 1-bit format flag (e.g. in
    the archive meta) so the inflate side picks ``codec.decode_latents`` vs
    ``decode_latents_best``.

    The returned bytes are the FINAL latents-section bytes (already brotli-compressed), matching
    what ``codec.build_archive`` writes after ``brotli.compress(encode_latents(...))``.
    """
    import brotli

    from tac.torch_vehicle.vendored_imports import import_vendored

    cfg = config or CrossPairLatentConfig()
    codec = import_vendored("codec")
    vendored_blob = brotli.compress(codec.encode_latents(latents), quality=11)

    # structural_only=True: only a genuine cross-pair-redundancy format (dedup / codebook) may
    # win. FRAMED_DELTA is information-identical to vendored, so the selector never returns it as
    # a "win" here — its byte difference would be brotli block-ALIGNMENT noise, not a real saving.
    framed_payload, flag = encode_latents_best(latents, cfg, structural_only=True)
    is_structural = flag in _STRUCTURAL_FORMATS
    framed_blob = brotli.compress(framed_payload, quality=11)

    # STRICT default-preserving: emit the framed format ONLY when a STRUCTURAL candidate is
    # strictly smaller than the vendored blob (a tie, a no-win, or a framed-delta fallback all
    # keep the EXACT vendored bytes — so the no-win path is byte-identical to a vendored build).
    if is_structural and len(framed_blob) < len(vendored_blob):
        return framed_blob, True
    return vendored_blob, False


def decode_latent_blob(latents_brotli: bytes, is_framed_format: bool) -> torch.Tensor:
    """Inflate-side dispatch: decode the latents blob given the persisted format bit.

    ``is_framed_format=False`` -> the vendored ``codec.decode_latents`` (byte-identical path);
    ``True`` -> this module's unified ``decode_latents_best``. One ``if`` at the inflate seam.
    """
    import brotli

    if is_framed_format:
        return decode_latents_best(brotli.decompress(latents_brotli))
    from tac.torch_vehicle.vendored_imports import import_vendored

    codec = import_vendored("codec")
    return codec.decode_latents(brotli.decompress(latents_brotli))


# Score-claim discipline metadata.
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
READY_FOR_EXACT_EVAL_DISPATCH = False

__all__ = [
    "FORMAT_CODEBOOK",
    "FORMAT_DEDUP",
    "FORMAT_FRAMED_DELTA",
    "FORMAT_VENDORED",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
    "SCORE_CLAIM",
    "CrossPairLatentConfig",
    "build_latent_blob_dedup_or_vendored",
    "decode_latent_blob",
    "decode_latents_best",
    "dequantize_pairs",
    "encode_latents_best",
    "quantize_pairs",
]
