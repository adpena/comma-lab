# SPDX-License-Identifier: MIT
"""Structural Q* compression of the learned HNeRV decoder (#71 — the convergent
singular pointer-mover).

THE CONVERGENCE (frontier_pointer_move_ledger_20260610.md): #64 (lossless
exhausted), #72 (residual codes cheap but receptive-field collateral kills
application), and #73 (generic SVD/sparse basis needs >=625 KB to hold the pose
tube) ALL prove the SAME thing from three directions — the 177 KB *learned*
HNeRV nonlinear basis IS the cheap-feasible representation for holding pose+seg
simultaneously. The only sub-frontier path that still holds both terms is a
SMALLER LEARNED basis. So #71 must compress the LEARNED basis ITSELF (factor /
prune / share / distill the memorized renderer), NOT substitute a generic one
(which #73 already proved cannot win).

THE MATH (the prize): frontier S = rate 0.118 (162,127 B decoder blob, 177,169 B
archive) + distortion 0.073. The decoder blob is 91.5% of the archive. A
score-equivalent renderer at a SMALLER decoder blob moves the pointer by RATE
ALONE at constant distortion:
    S(bytes) = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes / 37_545_489.
At held distortion, dS = -25 * (decoder_blob_saved) / 37_545_489 per byte.

THE GRAMMAR (codec.py, the EXACT byte cost surface this module measures against):
the decoder is stored per-tensor as int8 codes (zigzag/byte-map) + one fp16
scale, in a specific tensor STORAGE ORDER, partitioned into 7 independent brotli
streams (``DECODER_STREAM_ENDS``). The 162,127 bytes is the brotli-compressed
size of those int8 code streams. So a structural transform only moves the
pointer if it reduces the BROTLI-COMPRESSED size of the int8 code streams while
the dequantized weights still hold d_seg/d_pose inside the Q* cell. Param count
is a proxy; the authority is the brotli byte count, which this module measures
EXACTLY by replaying the codec encode path.

NO FAKE (CLAUDE.md class 1 + class 8): the transforms here REALLY factor / prune
the weights (a no-op transform that returns the input is caught by tests); the
byte cost is the REAL brotli size of the re-encoded grammar; Q* membership is
measured on the EXACT frozen CPU scorer (never MPS). This module produces the
transform + the byte measurement; the EXACT scorer measurement + the
``scorer_quotient_candidate_row.v1`` emission happen in the driver
(``experiments/`` thin CLI) that has the heavy render/score deps.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import brotli
import numpy as np

# ---------------------------------------------------------------------------
# The frontier codec grammar constants (codec.py, byte-identical).
# These describe the EXACT byte cost surface; the byte measurement replays them.
# ---------------------------------------------------------------------------
DECODER_BLOB_LEN_FRONTIER = 162_127  # the measured frontier decoder blob length

# Tensor storage order (index into probe.state_dict().items()).
DECODER_STORAGE_ORDER = (
    14, 22, 7, 6, 19, 10, 25, 4, 20, 9, 12, 15, 5, 11,
    18, 1, 21, 3, 27, 13, 2, 26, 24, 17, 16, 23, 8, 0,
)
# Stream boundaries (cumulative tensor counts in storage order; 7 streams).
DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)

CONV4_STORAGE_PERMS = {
    2: (3, 0, 2, 1), 4: (3, 0, 2, 1), 6: (0, 1, 2, 3), 8: (3, 0, 1, 2),
    10: (3, 0, 2, 1), 12: (3, 0, 1, 2), 14: (1, 0, 2, 3), 16: (3, 0, 2, 1),
    18: (1, 0, 2, 3), 20: (0, 3, 2, 1), 22: (0, 3, 2, 1), 24: (0, 2, 3, 1),
    26: (0, 1, 3, 2),
}
DECODER_BYTE_MAPS = {9: "negzig", 14: "negzig", 20: "twos", 27: "off"}

BROTLI_QUALITY = 11  # the codec uses brotli quality=11 (max) for the source recode


# ---------------------------------------------------------------------------
# byte maps (encode side — inverse of codec.decode_mapped_u8)
# ---------------------------------------------------------------------------
def zigzag_encode_i8(arr_i8: np.ndarray) -> np.ndarray:
    """Signed int8 residual -> unsigned zigzag symbol (inverse of decode)."""
    a = arr_i8.astype(np.int32)
    return np.where(a >= 0, 2 * a, -2 * a - 1).astype(np.uint8)


def encode_mapped_i8(arr_i8: np.ndarray, byte_map: str) -> np.ndarray:
    """Encode one int8 tensor stream into stored uint8 per its byte map.

    Exact inverse of ``codec.decode_mapped_u8`` for each declared map.
    """
    a = arr_i8.astype(np.int16)
    if byte_map == "zig":
        return zigzag_encode_i8(arr_i8)
    if byte_map == "negzig":
        # decode: (-zigzag_decode(u8)).int8  =>  encode: zigzag_encode(-q)
        return zigzag_encode_i8((-a).astype(np.int8))
    if byte_map == "off":
        # decode: (u8 - 128).int8  =>  encode: (q + 128) & 255
        return ((a + 128) & 0xFF).astype(np.uint8)
    if byte_map == "twos":
        # decode: u8.view(int8)  =>  encode: q.view(uint8)
        return arr_i8.astype(np.int8).view(np.uint8)
    raise ValueError(f"unknown decoder byte map: {byte_map}")


# ---------------------------------------------------------------------------
# int8 code recovery (the dequantized weights ARE int8 * fp16_scale; recover the
# exact codes so a structural transform operates on the SAME representation the
# codec stores).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TensorCode:
    """One tensor's exact stored representation: int8 codes + fp16 scale."""

    name: str
    idx: int  # index into state_dict().items() (storage-order key)
    shape: tuple[int, ...]
    codes_i8: np.ndarray  # int8, the dequantized weights = codes_i8 * scale
    scale: float  # the fp16-rounded scale


def recover_tensor_code(name: str, idx: int, weight: np.ndarray) -> TensorCode:
    """Recover (int8 codes, fp16 scale) from a dequantized weight tensor.

    The codec stores W = q_int8 * fp16(scale). The scale is the quantization
    step; recover it as the smallest positive gap between distinct dequantized
    magnitudes (robust to the smallest-nonzero != step case seen on stem/rgb
    where min|nz| == 2*step). Falls back to min|nonzero| when no finer gap
    exists. The recovered codes round-trip W exactly when W truly is int8*scale.
    """
    w = weight.astype(np.float64)
    nz = np.abs(w[w != 0.0])
    if nz.size == 0:
        return TensorCode(name, idx, tuple(weight.shape),
                          np.zeros(weight.shape, dtype=np.int8), 0.0)
    # candidate step = smallest nonzero magnitude; test whether all dequantized
    # values are (near) integer multiples of step and also of step/2 .. step/8.
    base = float(nz.min())
    best_step = base
    for div in (1, 2, 3, 4):
        step = base / div
        q = w / step
        if np.max(np.abs(q - np.round(q))) < 1e-3 and np.abs(np.round(q)).max() <= 127:
            best_step = step
            break
    scale = float(np.float16(best_step))
    codes = np.round(w / scale).astype(np.int64)
    codes = np.clip(codes, -127, 127).astype(np.int8)
    return TensorCode(name, idx, tuple(weight.shape), codes, scale)


def recover_all_codes(state_dict_items: list[tuple[str, np.ndarray]]) -> dict[int, TensorCode]:
    """Recover int8 codes for EVERY tensor (weights AND biases), in the canonical
    probe state_dict order (codec.decode_decoder_compact iterates
    ``probe.state_dict().items()`` for the idx->name->shape mapping, and stores
    every tensor including 1D biases). Returns idx -> TensorCode for ALL tensors
    so ``encode_decoder_blob_bytes`` can replay the full grammar EXACTLY (the
    weight streams dominate the blob, but biases are part of the stream bytes)."""
    out: dict[int, TensorCode] = {}
    for idx, (name, w) in enumerate(state_dict_items):
        out[idx] = recover_tensor_code(name, idx, w)
    return out


# ---------------------------------------------------------------------------
# EXACT byte cost: replay the codec encode of the int8 code streams.
# ---------------------------------------------------------------------------
def _stored_u8_for_tensor(tc: TensorCode) -> bytes:
    """Produce the stored bytes for one tensor: mapped-u8 codes (in storage perm
    order for 4D) followed by the fp16 scale, matching codec.decode_decoder_compact
    consumption order (codes then scale)."""
    codes = tc.codes_i8
    if codes.ndim == 4:
        perm = CONV4_STORAGE_PERMS[tc.idx]
        codes = np.transpose(codes, perm).copy()
    flat = codes.reshape(-1)
    u8 = encode_mapped_i8(flat, DECODER_BYTE_MAPS.get(tc.idx, "zig"))
    scale_bytes = np.float16(tc.scale).tobytes()
    return u8.tobytes() + scale_bytes


def encode_decoder_blob_bytes(codes: dict[int, TensorCode]) -> int:
    """Return the EXACT brotli-compressed decoder blob length for a given set of
    int8 tensor codes, replaying the codec's 7-stream grammar.

    This is the authority byte cost for any structural transform: a transform
    reduces the frontier 162,127-byte blob iff this returns a smaller number.
    """
    # Build per-tensor stored bytes in storage order, partition into 7 streams.
    per_tensor = []
    for pos, idx in enumerate(DECODER_STORAGE_ORDER):
        if idx not in codes:
            raise ValueError(f"missing code for storage idx {idx}")
        per_tensor.append(_stored_u8_for_tensor(codes[idx]))

    stream_starts = (0,) + DECODER_STREAM_ENDS[:-1]
    total = 0
    for s_start, s_end in zip(stream_starts, DECODER_STREAM_ENDS):
        chunk = b"".join(per_tensor[s_start:s_end])
        total += len(brotli.compress(chunk, quality=BROTLI_QUALITY))
    return total


# ---------------------------------------------------------------------------
# Structural transforms (each REALLY transforms the codes; no-op is detectable).
# ---------------------------------------------------------------------------
def low_rank_truncate_weight(weight: np.ndarray, rank: int) -> np.ndarray:
    """Rank-``rank`` SVD reconstruction of a weight tensor (4D conv reshaped to
    (O, I*kh*kw)). Returns a float weight of the SAME shape. A faithful
    factorization: reduces effective rank, reconstructs within SVD tolerance."""
    a = weight.astype(np.float64)
    sh = a.shape
    if a.ndim == 4:
        o, i, kh, kw = sh
        m = a.reshape(o, i * kh * kw)
    elif a.ndim == 2:
        m = a
    else:
        raise ValueError(f"unsupported ndim {a.ndim}")
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    r = max(1, min(int(rank), len(s)))
    # explicit float64 contiguous slices; the math is the exact rank-r
    # reconstruction. A spurious BLAS warning can fire on degenerate singular
    # values (denormal/zero) on some platforms; it does not affect the result.
    ur = np.ascontiguousarray(u[:, :r])
    vr = np.ascontiguousarray(vt[:r])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mr = (ur * s[:r][np.newaxis, :]) @ vr
    return mr.reshape(sh).astype(np.float32)


def factored_param_count(shape: tuple[int, ...], rank: int) -> tuple[int, int]:
    """(dense_params, factored_params) for storing W as U(O,r)+V(r,IKK).

    Factored storage is cheaper iff factored < dense, i.e. rank < O*IKK/(O+IKK).
    """
    if len(shape) == 4:
        o, i, kh, kw = shape
        ikk = i * kh * kw
    elif len(shape) == 2:
        o, ikk = shape
    else:
        raise ValueError(f"unsupported ndim {len(shape)}")
    r = max(1, min(int(rank), min(o, ikk)))
    return o * ikk, r * (o + ikk)


def score_aware_prune_codes(
    codes_i8: np.ndarray,
    sensitivity: np.ndarray,
    keep_fraction: float,
) -> np.ndarray:
    """Zero the lowest-sensitivity weights (structured pruning by MEASURED
    scorer-sensitivity), keeping ``keep_fraction`` of the magnitude*sensitivity
    mass. Returns int8 codes with the pruned entries set to 0 (brotli-friendly).

    ``sensitivity`` is a per-weight non-negative importance (e.g. |d score /
    d weight| from the exact scorer, or |weight| * per-tensor scorer gradient).
    Pruning by sensitivity must beat random pruning (a test asserts this).
    """
    if codes_i8.shape != sensitivity.shape:
        raise ValueError("sensitivity must match codes shape")
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("keep_fraction in (0,1]")
    flat_c = codes_i8.reshape(-1)
    flat_s = sensitivity.reshape(-1).astype(np.float64)
    n = flat_c.size
    n_keep = max(1, int(round(keep_fraction * n)))
    if n_keep >= n:
        return codes_i8.copy()
    # keep the n_keep entries with the highest sensitivity; zero the rest.
    keep_idx = np.argpartition(flat_s, n - n_keep)[n - n_keep:]
    out = np.zeros_like(flat_c)
    out[keep_idx] = flat_c[keep_idx]
    return out.reshape(codes_i8.shape)


def magnitude_sensitivity(codes_i8: np.ndarray) -> np.ndarray:
    """Fallback sensitivity = |code| (magnitude pruning). A real, non-constant
    importance map; score-aware pruning should beat this on the exact scorer."""
    return np.abs(codes_i8.astype(np.float64))


# ---------------------------------------------------------------------------
# rate-only delta-score (held distortion) helper
# ---------------------------------------------------------------------------
RATE_DENOM = 37_545_489


def rate_only_delta_score(decoder_blob_bytes_after: int, decoder_blob_bytes_before: int) -> float:
    """ΔS contributed by the rate term alone when distortion is held constant.

    Other archive sections (latents 15,387 B + sidecar/selector/dqs1) are
    unchanged, so the archive-byte delta equals the decoder-blob delta.
    """
    delta_bytes = decoder_blob_bytes_after - decoder_blob_bytes_before
    return 25.0 * delta_bytes / RATE_DENOM
