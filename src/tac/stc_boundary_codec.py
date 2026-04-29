"""Lane STC — boundary-focused mask-class codec (Stage 2 implementation).

Replaces the AV1 monochrome ``masks.mkv`` payload with a deterministic,
lossless boundary-coding format. Predicted savings vs Lane A's ~200KB
masks.mkv: 60-80 KB (rate -0.04 to -0.05).

See ``docs/paper/lane_stc_boundary_coding_design_20260429.md`` for the
full design + risks discussion. This module is the v1 production codec
described in the design's "Stage 2" section.

KEY DECISIONS (deviating from a naive bitmap-AQ approach):

  1. The boundary BITMAP is sparse (~5% density). Storing it as a dense
     arithmetic-coded 0/1 stream costs ~7 KB/frame at the Shannon bound.
     Across 1200 frames that's ~8.4 MB — completely unacceptable.
     Instead we encode boundaries as raster-order GAPS between
     consecutive boundary positions. With 5% density mean gap ~20, so
     the total bit cost for 1200 frames * 9.8K boundaries is on the
     order of 50-80 KB after arithmetic coding the gap distribution.
  2. We use ONE arithmetic-coding pass per stream type across ALL
     frames, not per-frame. AQv1 has a ~30B fixed header; per-frame
     coding would burn 144 KB on headers alone.
  3. Per-frame side info is only:
        majority_class    : 1 byte
        n_boundary        : 4 bytes
        n_exceptions      : 4 bytes
     for ~11 KB of metadata across 1200 frames.
  4. Non-boundary majority is the per-frame mode. The "exception"
     stream carries (a) sparse positions of non-boundary pixels whose
     class != majority and (b) those pixels' class IDs. Like the
     boundary bitmap, exception positions are gap-coded.
  5. The arithmetic coder underneath is the AQv1 implementation in
     ``tac.arithmetic_qint_codec`` (Witten/Neal/Cleary 1987 form,
     bit-deterministic across CPython platforms). Reusing it inherits
     bitexact-roundtrip guarantees.

Container layout (STCB v1, little-endian):

    magic                  : 4 bytes  = b"STCB"
    version                : 2 bytes  uint16 = 1
    num_classes            : 1 byte   uint8 = NUM_CLASSES (5)
    n_frames               : 4 bytes  uint32
    height                 : 2 bytes  uint16
    width                  : 2 bytes  uint16
    boundary_fraction_pct  : 1 byte   uint8 (encoder-time fraction *100)
    flags                  : 1 byte   uint8 (reserved, currently 0)

    --- per-frame side info (3 * n_frames * (1+4+4) = 9 bytes/frame) ---
    For each frame f in [0, n_frames):
        majority_class : 1 byte  uint8
        n_boundary     : 4 bytes uint32
        n_exceptions   : 4 bytes uint32

    --- 4 monolithic AQv1 streams (one each, concatenated across frames) ---

    boundary_gaps_blob_len    : 8 bytes uint64
    boundary_gaps_blob        : bytes (AQv1, alphabet up to MAX_GAP)
    boundary_classes_blob_len : 8 bytes uint64
    boundary_classes_blob     : bytes (AQv1, alphabet num_classes)
    exception_gaps_blob_len   : 8 bytes uint64
    exception_gaps_blob       : bytes (AQv1, alphabet up to MAX_GAP)
    exception_classes_blob_len: 8 bytes uint64
    exception_classes_blob    : bytes (AQv1, alphabet num_classes)

For the gap streams we use a fixed alphabet size MAX_GAP_ALPHABET. If a
gap exceeds the cap we emit (MAX_GAP_ALPHABET - 1) repeatedly until the
remainder fits. This is the same trick used in JPEG run-length coding.
The decoder mirrors this logic.

CLAUDE.md compliance notes
--------------------------
* Strict scorer rule: no scorer/SegNet/PoseNet imports anywhere here.
* eval_roundtrip irrelevant (encoder-only mask packaging; bytes are exact).
* Bitexact deterministic across CPython platforms (inherited from AQv1).
* Encoder verifies decode-roundtrip before returning bytes by default.
"""
from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tac.arithmetic_qint_codec import (
    decode_qints_arithmetic,
    encode_qints_arithmetic,
)
from tac.camera import NUM_CLASSES

_STCB_MAGIC: bytes = b"STCB"
_STCB_VERSION: int = 1

# Maximum encodable single gap. Larger gaps emit (_MAX_GAP - 1) overflow
# tokens followed by the remainder. With 5% boundary density mean gap
# is ~20, so 1024 is generous and keeps the alphabet small enough that
# the AQv1 frequency table (4 bytes per symbol) stays at 4 KB.
_MAX_GAP_ALPHABET: int = 1024


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------


def detect_boundary_pixels(
    masks: torch.Tensor,
    boundary_fraction: float = 0.05,
    *,
    per_frame: bool = True,
) -> torch.Tensor:
    """Return a boolean tensor marking class-boundary pixels.

    Boundary detection runs a deterministic 3x3 Sobel magnitude on the
    integer argmax class-id map (no logits, no scorers). The threshold
    is set so that approximately ``boundary_fraction`` of pixels per
    frame are classified as boundary.

    Args:
        masks: (N, H, W) integer tensor of class IDs in [0, NUM_CLASSES).
        boundary_fraction: target fraction of pixels per frame marked as
            boundary. Default 0.05 (5%).
        per_frame: if True, compute one threshold per frame; otherwise
            one threshold for the whole video.

    Returns:
        (N, H, W) bool tensor, True at boundary positions.
    """
    if masks.dim() != 3:
        raise ValueError(f"masks must be (N, H, W); got {tuple(masks.shape)}")
    if not (0.0 < boundary_fraction < 1.0):
        raise ValueError(
            f"boundary_fraction must be in (0, 1); got {boundary_fraction}"
        )

    n, h, w = masks.shape
    m = masks.to(torch.float32).unsqueeze(1)  # (N, 1, H, W)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(m, sobel_x, padding=1)
    gy = F.conv2d(m, sobel_y, padding=1)
    rho = torch.sqrt(gx * gx + gy * gy).squeeze(1)  # (N, H, W)

    if per_frame:
        flat = rho.reshape(n, -1)
        k = max(1, int(round(h * w * boundary_fraction)))
        k_smallest = max(1, h * w - k + 1)
        thresh = flat.kthvalue(k_smallest, dim=1).values  # (N,)
        boundary = rho >= thresh.view(n, 1, 1)
    else:
        flat = rho.reshape(-1)
        k = max(1, int(round(flat.numel() * boundary_fraction)))
        k_smallest = max(1, flat.numel() - k + 1)
        thresh = flat.kthvalue(k_smallest).values
        boundary = rho >= thresh

    return boundary


# ---------------------------------------------------------------------------
# Sparse-position encoding (gap-based)
# ---------------------------------------------------------------------------


def _positions_to_gaps(positions: np.ndarray) -> np.ndarray:
    """Convert sorted raster positions to a gap stream with overflow tokens.

    For an empty position list, returns an empty array. Otherwise, the
    output stream has the following property: walking it left-to-right,
    accumulating values into ``cursor``, and emitting ``cursor`` as a
    new position whenever a non-overflow token is seen, recovers the
    original positions. Overflow tokens are ``_MAX_GAP_ALPHABET - 1``
    and represent that many positions of "skip" without emitting a
    position.

    Args:
        positions: sorted 1-D int64 array of positions (raster order).

    Returns:
        1-D int64 array of token IDs in [0, _MAX_GAP_ALPHABET).
    """
    if positions.size == 0:
        return np.zeros(0, dtype=np.int64)
    pos = np.ascontiguousarray(positions, dtype=np.int64)
    if not np.all(np.diff(pos) > 0):
        raise ValueError("positions must be strictly increasing in raster order")

    # Gap[i] = pos[i] - pos[i-1] - 1 for i>=1 (number of NON-boundary
    # positions between consecutive boundaries). Gap[0] = pos[0] (number
    # of NON-boundary positions before the first boundary).
    gaps = np.empty(pos.size, dtype=np.int64)
    gaps[0] = pos[0]
    if pos.size > 1:
        gaps[1:] = pos[1:] - pos[:-1] - 1

    # Encode each gap with overflow tokens.
    # A gap of value g is emitted as: floor(g / OVF) overflow tokens,
    # followed by 1 token of value (g % OVF).
    # OVF = _MAX_GAP_ALPHABET - 1 (the overflow token value).
    OVF = _MAX_GAP_ALPHABET - 1
    out = []
    for g in gaps.tolist():
        n_ovf = int(g) // OVF
        rem = int(g) % OVF
        if n_ovf > 0:
            out.extend([OVF] * n_ovf)
        out.append(rem)
    return np.asarray(out, dtype=np.int64)


def _gaps_to_positions(tokens: np.ndarray, n_positions: int) -> np.ndarray:
    """Inverse of _positions_to_gaps.

    Args:
        tokens: 1-D int64 array of token IDs (output of _positions_to_gaps).
        n_positions: number of positions to recover.

    Returns:
        1-D int64 array of n_positions sorted positions.
    """
    if n_positions == 0:
        return np.zeros(0, dtype=np.int64)
    OVF = _MAX_GAP_ALPHABET - 1
    out = np.empty(n_positions, dtype=np.int64)
    cursor = -1
    accum = 0
    pos_idx = 0
    for t in tokens.tolist():
        if t == OVF:
            accum += OVF
            continue
        gap = accum + t
        accum = 0
        if pos_idx == 0:
            cursor = gap
        else:
            cursor = cursor + gap + 1
        out[pos_idx] = cursor
        pos_idx += 1
        if pos_idx >= n_positions:
            break
    if pos_idx != n_positions:
        raise ValueError(
            f"gap-decode produced {pos_idx} positions; expected {n_positions}"
        )
    return out


# ---------------------------------------------------------------------------
# Top-level encode / decode
# ---------------------------------------------------------------------------


def encode_mask_video_stc(
    masks: torch.Tensor,
    output_path: str | Path,
    *,
    boundary_fraction: float = 0.05,
    per_frame_threshold: bool = True,
    verify_roundtrip: bool = True,
) -> int:
    """Encode an (N, H, W) class-id mask tensor as an STC-boundary file.

    Args:
        masks: (N, H, W) long/integer tensor of class IDs.
        output_path: where to write the .stcb file.
        boundary_fraction: target fraction of pixels per frame marked
            as boundary. Default 0.05.
        per_frame_threshold: if True, threshold per-frame (recommended).
        verify_roundtrip: if True (default), decode-and-compare the
            written file before returning. Refuses to ship corrupted
            archives.

    Returns:
        Number of bytes written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if masks.dim() != 3:
        raise ValueError(f"masks must be (N, H, W); got {tuple(masks.shape)}")
    n, h, w = masks.shape
    if h > 65535 or w > 65535:
        raise ValueError(f"frame dims must fit in uint16: H={h}, W={w}")
    if n > 0xFFFFFFFF:
        raise ValueError(f"n_frames must fit in uint32: N={n}")

    masks_int = masks.to(torch.int64).contiguous()
    if int(masks_int.min().item()) < 0 or int(masks_int.max().item()) >= NUM_CLASSES:
        raise ValueError(f"class IDs must be in [0, {NUM_CLASSES})")

    boundary = detect_boundary_pixels(
        masks_int,
        boundary_fraction=boundary_fraction,
        per_frame=per_frame_threshold,
    )

    masks_np = masks_int.numpy()
    boundary_np = boundary.numpy().astype(np.uint8)

    bf_pct = max(0, min(100, int(round(boundary_fraction * 100))))

    # Per-frame side info + accumulated streams.
    per_frame_records = []  # list of (majority, n_boundary, n_exception)
    boundary_gap_tokens_all: list[np.ndarray] = []
    boundary_class_tokens_all: list[np.ndarray] = []
    exception_gap_tokens_all: list[np.ndarray] = []
    exception_class_tokens_all: list[np.ndarray] = []

    hw = h * w
    for f in range(n):
        frame_classes = masks_np[f].reshape(-1)
        frame_boundary = boundary_np[f].reshape(-1)

        non_boundary_mask = frame_boundary == 0
        non_boundary_classes = frame_classes[non_boundary_mask]
        if non_boundary_classes.size == 0:
            counts = np.bincount(frame_classes, minlength=NUM_CLASSES)
        else:
            counts = np.bincount(non_boundary_classes, minlength=NUM_CLASSES)
        majority = int(np.argmax(counts))

        boundary_positions = np.flatnonzero(frame_boundary == 1)
        boundary_classes = frame_classes[boundary_positions]

        # Exception positions: non-boundary pixels whose class != majority.
        non_boundary_positions = np.flatnonzero(non_boundary_mask)
        exc_mask = non_boundary_classes != majority
        exception_positions_in_frame = non_boundary_positions[exc_mask]
        exception_classes = non_boundary_classes[exc_mask]

        boundary_gap_tokens_all.append(_positions_to_gaps(boundary_positions))
        boundary_class_tokens_all.append(boundary_classes.astype(np.int64))
        exception_gap_tokens_all.append(
            _positions_to_gaps(exception_positions_in_frame)
        )
        exception_class_tokens_all.append(exception_classes.astype(np.int64))

        per_frame_records.append((majority, len(boundary_positions), len(exception_classes)))

    # Concatenate.
    bg_tokens = (
        np.concatenate(boundary_gap_tokens_all)
        if boundary_gap_tokens_all
        else np.zeros(0, dtype=np.int64)
    )
    bc_tokens = (
        np.concatenate(boundary_class_tokens_all)
        if boundary_class_tokens_all
        else np.zeros(0, dtype=np.int64)
    )
    eg_tokens = (
        np.concatenate(exception_gap_tokens_all)
        if exception_gap_tokens_all
        else np.zeros(0, dtype=np.int64)
    )
    ec_tokens = (
        np.concatenate(exception_class_tokens_all)
        if exception_class_tokens_all
        else np.zeros(0, dtype=np.int64)
    )

    bg_blob = _arith_encode(bg_tokens.astype(np.int64), _MAX_GAP_ALPHABET)
    bc_blob = _arith_encode(bc_tokens.astype(np.int64), NUM_CLASSES)
    eg_blob = _arith_encode(eg_tokens.astype(np.int64), _MAX_GAP_ALPHABET)
    ec_blob = _arith_encode(ec_tokens.astype(np.int64), NUM_CLASSES)

    buf = io.BytesIO()
    buf.write(_STCB_MAGIC)
    buf.write(struct.pack("<H", _STCB_VERSION))
    buf.write(struct.pack("<B", NUM_CLASSES))
    buf.write(struct.pack("<I", n))
    buf.write(struct.pack("<H", h))
    buf.write(struct.pack("<H", w))
    buf.write(struct.pack("<B", bf_pct))
    buf.write(struct.pack("<B", 0))  # reserved flags

    for (majority, nb, nx) in per_frame_records:
        buf.write(struct.pack("<B", majority))
        buf.write(struct.pack("<I", nb))
        buf.write(struct.pack("<I", nx))

    buf.write(struct.pack("<Q", len(bg_blob)))
    buf.write(bg_blob)
    buf.write(struct.pack("<Q", len(bc_blob)))
    buf.write(bc_blob)
    buf.write(struct.pack("<Q", len(eg_blob)))
    buf.write(eg_blob)
    buf.write(struct.pack("<Q", len(ec_blob)))
    buf.write(ec_blob)

    payload = buf.getvalue()
    output_path.write_bytes(payload)

    if verify_roundtrip:
        decoded = decode_mask_video_stc(output_path)
        if decoded.shape != masks_int.shape:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"STC roundtrip shape mismatch: encoded "
                f"{tuple(masks_int.shape)} vs decoded {tuple(decoded.shape)}"
            )
        if not torch.equal(decoded, masks_int):
            n_diff = int((decoded != masks_int).sum().item())
            output_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"STC roundtrip class-ID mismatch: {n_diff} pixels differ"
            )

    return len(payload)


def decode_mask_video_stc(input_path: str | Path) -> torch.Tensor:
    """Decode an STC-boundary file back to (N, H, W) int64 class IDs.

    Pure integer / byte parsing. No SegNet, no logits, no scorer load.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"STC file not found: {path}")
    blob = path.read_bytes()
    if len(blob) < 17 or blob[:4] != _STCB_MAGIC:
        raise ValueError(
            f"bad STCB magic in {path}: first bytes = {blob[:4]!r}"
        )

    buf = io.BytesIO(blob[4:])
    (version,) = struct.unpack("<H", buf.read(2))
    if version != _STCB_VERSION:
        raise ValueError(f"unsupported STCB version: {version}")
    (num_classes,) = struct.unpack("<B", buf.read(1))
    if num_classes != NUM_CLASSES:
        raise ValueError(
            f"STCB num_classes mismatch: file={num_classes} build={NUM_CLASSES}"
        )
    (n,) = struct.unpack("<I", buf.read(4))
    (h,) = struct.unpack("<H", buf.read(2))
    (w,) = struct.unpack("<H", buf.read(2))
    (_bf_pct,) = struct.unpack("<B", buf.read(1))
    (_flags,) = struct.unpack("<B", buf.read(1))

    per_frame_records = []
    for _ in range(n):
        rec = buf.read(9)
        if len(rec) != 9:
            raise ValueError("truncated per-frame side-info")
        majority = rec[0]
        nb = struct.unpack("<I", rec[1:5])[0]
        nx = struct.unpack("<I", rec[5:9])[0]
        per_frame_records.append((majority, nb, nx))

    def _read_blob(stream: io.BytesIO) -> bytes:
        len_bytes = stream.read(8)
        if len(len_bytes) != 8:
            raise ValueError("truncated stream-length header")
        (length,) = struct.unpack("<Q", len_bytes)
        body = stream.read(length)
        if len(body) != length:
            raise ValueError(
                f"truncated stream body: declared {length}B, got {len(body)}B"
            )
        return body

    bg_blob = _read_blob(buf)
    bc_blob = _read_blob(buf)
    eg_blob = _read_blob(buf)
    ec_blob = _read_blob(buf)

    bg_tokens = _arith_decode(bg_blob)
    bc_tokens = _arith_decode(bc_blob)
    eg_tokens = _arith_decode(eg_blob)
    ec_tokens = _arith_decode(ec_blob)

    out = np.empty((n, h, w), dtype=np.int64)
    hw = h * w

    bg_cursor = 0
    bc_cursor = 0
    eg_cursor = 0
    ec_cursor = 0

    OVF = _MAX_GAP_ALPHABET - 1
    for f, (majority, n_boundary, n_exception) in enumerate(per_frame_records):
        # Boundary positions: walk bg_tokens until we have n_boundary positions.
        boundary_positions, bg_cursor = _consume_gap_stream(
            bg_tokens, bg_cursor, n_boundary
        )
        boundary_classes = bc_tokens[bc_cursor : bc_cursor + n_boundary]
        bc_cursor += n_boundary
        if boundary_classes.size != n_boundary:
            raise ValueError(
                f"frame {f}: boundary class stream short ({boundary_classes.size} vs {n_boundary})"
            )

        # Exception positions: gap stream walks over RASTER pixels (same
        # space as boundary positions), but only the ones NOT in the
        # boundary set are exceptions. Encoder used the absolute raster
        # position when emitting the gap stream, so the decoder gets
        # absolute raster positions back.
        exception_positions, eg_cursor = _consume_gap_stream(
            eg_tokens, eg_cursor, n_exception
        )
        exception_classes = ec_tokens[ec_cursor : ec_cursor + n_exception]
        ec_cursor += n_exception
        if exception_classes.size != n_exception:
            raise ValueError(
                f"frame {f}: exception class stream short ({exception_classes.size} vs {n_exception})"
            )

        frame = np.full(hw, majority, dtype=np.int64)
        if exception_positions.size > 0:
            frame[exception_positions] = exception_classes
        if boundary_positions.size > 0:
            frame[boundary_positions] = boundary_classes
        out[f] = frame.reshape(h, w)

    if bc_cursor != bc_tokens.size:
        raise ValueError(
            f"trailing boundary class tokens: {bc_tokens.size - bc_cursor} unconsumed"
        )
    if ec_cursor != ec_tokens.size:
        raise ValueError(
            f"trailing exception class tokens: {ec_tokens.size - ec_cursor} unconsumed"
        )

    return torch.from_numpy(out)


def _consume_gap_stream(
    tokens: np.ndarray, cursor: int, n_positions: int
) -> tuple[np.ndarray, int]:
    """Pull n_positions positions out of a gap stream, advancing cursor.

    Returns (positions, new_cursor).
    """
    if n_positions == 0:
        return np.zeros(0, dtype=np.int64), cursor
    OVF = _MAX_GAP_ALPHABET - 1
    out = np.empty(n_positions, dtype=np.int64)
    pos_idx = 0
    accum = 0
    cur_position = -1
    while pos_idx < n_positions:
        if cursor >= tokens.size:
            raise ValueError(
                f"gap stream exhausted at pos_idx={pos_idx}/{n_positions}"
            )
        t = int(tokens[cursor])
        cursor += 1
        if t == OVF:
            accum += OVF
            continue
        gap = accum + t
        accum = 0
        if pos_idx == 0:
            cur_position = gap
        else:
            cur_position = cur_position + gap + 1
        out[pos_idx] = cur_position
        pos_idx += 1
    return out, cursor


# ---------------------------------------------------------------------------
# AQv1 thin wrappers (avoid the empty-array assertion in the underlying API)
# ---------------------------------------------------------------------------


def _arith_encode(symbols: np.ndarray, num_symbols: int) -> bytes:
    """Encode an int64 stream with AQv1. Handles empty streams by emitting
    a single sentinel byte 0x00 (the decoder reads this and returns []).

    The empty-stream path bypasses ``encode_qints_arithmetic`` because
    that function rejects zero-size arrays at its min/max validator.
    """
    if symbols.size == 0:
        return b"\x00"
    if int(symbols.min()) < 0 or int(symbols.max()) >= num_symbols:
        raise ValueError(
            f"symbols out of range [0, {num_symbols}): "
            f"min={int(symbols.min())} max={int(symbols.max())}"
        )
    payload = encode_qints_arithmetic(
        symbols.astype(np.int64), num_symbols=num_symbols, offset=0
    )
    # Prefix a 0x01 byte so the decoder can detect the empty-stream sentinel.
    return b"\x01" + payload


def _arith_decode(blob: bytes) -> np.ndarray:
    """Inverse of _arith_encode."""
    if len(blob) == 0:
        raise ValueError("empty arith blob")
    sentinel = blob[0]
    if sentinel == 0:
        return np.zeros(0, dtype=np.int64)
    if sentinel != 1:
        raise ValueError(f"bad arith sentinel: {sentinel}")
    arr = decode_qints_arithmetic(blob[1:], expected_dtype=np.int64)
    return arr.astype(np.int64, copy=False)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def estimate_symbol_entropy_bits(symbols: np.ndarray, num_symbols: int) -> float:
    """Return the empirical 0-th order entropy in bits/symbol."""
    if symbols.size == 0:
        return 0.0
    counts = np.bincount(symbols.ravel().astype(np.int64), minlength=num_symbols)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def measure_stc_overhead(masks: torch.Tensor, *, boundary_fraction: float = 0.05) -> dict:
    """Measure actual byte cost vs Shannon-bound bytes for an input.

    The Shannon bound here is the per-frame 0-th order entropy of the
    class-ID stream — a generous lower bound (a real entropy coder can't
    exploit spatial structure cheaply).
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".stcb", delete=False) as tf:
        path = Path(tf.name)
    try:
        actual = encode_mask_video_stc(
            masks, path, boundary_fraction=boundary_fraction
        )
    finally:
        if path.exists():
            path.unlink()

    masks_np = masks.to(torch.int64).numpy()
    n_frames = masks_np.shape[0]
    total_bits = 0.0
    for f in range(n_frames):
        h_bits = estimate_symbol_entropy_bits(masks_np[f].ravel(), NUM_CLASSES)
        total_bits += h_bits * masks_np[f].size
    bound_bytes = total_bits / 8.0
    overhead_pct = 100.0 * (actual - bound_bytes) / max(1.0, bound_bytes)
    return {
        "actual_bytes": int(actual),
        "shannon_bound_bytes": float(bound_bytes),
        "overhead_pct": float(overhead_pct),
    }


__all__ = [
    "_STCB_MAGIC",
    "_STCB_VERSION",
    "detect_boundary_pixels",
    "encode_mask_video_stc",
    "decode_mask_video_stc",
    "estimate_symbol_entropy_bits",
    "measure_stc_overhead",
]
