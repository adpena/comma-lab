# SPDX-License-Identifier: MIT
"""ddm_ix2 — cross-member archive container + the two member re-layouts it ships.

WHAT THIS IS
------------
`ddm_ix1` measured that on our own vehicle the token stream's **layout** is worth
5,184 B while its **coder** is worth −5 B (brotli is *worse* than stored: the r7
stream is already at its coder's entropy).  It applied that toolbox to exactly one
member.  This module applies it to the **archive**, which is where the firmware
inversion actually bites: an embedded unpacker knows the layout at compile time and
ships **one contiguous image with computed offsets**, not N independently framed
sections.  Ours can too — `inflate.py` is untimed, unsized generic code
(`upstream/evaluate.py:63` charges `archive.zip` bytes only).

Three primitives, all LOSSLESS:

``IX2TOK01``
    Token frame.  ``(P,R,C,K)`` int4 lattice → per-cell mode ``base`` + modulo
    residual, residual transposed to **cell-major** ``(R,C,K,P)`` and nibble-packed,
    each block coded by the measured-best generic coder.  Decode is proved
    bit-identical **from the frame alone** — ``base`` is SHIPPED, never recomputed
    from the codes the decoder does not have (this is the exact fake ``ddm_ix1``
    caught in its own first headline).

``IX2REN01``
    Renderer frame.  Field-splits the lottery mask (max-entropy, kept stored) from
    the f16 gain/bias table (byte-plane transposed, then coded), instead of running
    one coder over the concatenation of the two.

``IX2CNT01``
    The container.  Length-prefixed sections in a fixed order; the **order and the
    names live in `inflate.py`**, so no per-section name, magic, or directory entry
    is counted.  One ZIP member instead of six.

MEASURED on the live ``dc1_fold`` vehicle (360,309 B), n600, byte-closed:
container alone −738 B · +IX2TOK −5,921 B · +binary config section −6,543 B ·
+IX2REN −6,627 B ⇒ ΔS −0.0044126, 0.608% of the gap to the PR130 floor.

NON-GOALS / what this module is NOT
-----------------------------------
* **Not a ranker.**  Every rung here is lossless, so ``d_seg`` and ``d_pose`` are
  invariant *by construction* and BYTES are the whole currency.  No sensitivity
  model, no metric choice, no Fisher/Bregman question arises — those bite the
  moment a rung becomes lossy, and none of these are.
* **Not a migration authority.**  ``inflate.py`` may carry GENERIC algorithm and
  GENERIC tables only (rule-118).  A value fitted to the clip is COUNTED however it
  is spelled.  ``rs_beta_mags`` is the live worked example: a 13-entry magnitude
  codebook derived from this clip's solve, so it stays counted (in
  ``pack_config_section``) — migrating it would be hide-data-in-code.
  ``st_grid`` is the sharper example: it is byte-equal to the vendored ``ST_GRID``
  on ``dc1_fold`` (generic *there*) and FITTED on ``ms8`` / ``pj2`` (video-derived
  *there*).  **The verdict is a (field, ARCHIVE) pair property, not a field
  property** — so ``classify_against_vendored`` computes it per archive and
  ``pack_config_section`` refuses to decide without the reference table.
  **Ambiguous ⇒ count it.**
* **Not an exemption.**  This line is safe to re-represent only because it is
  EXACTLY lossless.  ``ddm_cp1`` measured that a re-representation costs a tighter
  solve relatively more (``ms8`` −0.009%, ``pj2`` +0.099%, 11× worse) because the
  f16 storage lattice is not invariant even when the algebra is.  The moment a rung
  here goes lossy or re-quantizes an already-solved value, that law applies and it
  lands hardest on our best-solved members.

Every parser here closes exactly (``offset == len(payload)``) and raises on the
remainder, so a counted-but-inert section (#417) cannot survive a parse.
"""

from __future__ import annotations

import io
import lzma
import struct
import zipfile
import zlib
from collections.abc import Sequence
from typing import Final

import brotli
import numpy as np

__all__ = [
    "CODER_NAMES",
    "CONTAINER_MAGIC",
    "RENDERER_FRAME_MAGIC",
    "TABLE_F16",
    "TABLE_F32",
    "TABLE_F64",
    "TABLE_SCALED_INT",
    "TOKEN_FRAME_MAGIC",
    "IX2ContainerError",
    "build_payload",
    "decode_exact_table",
    "encode_exact_table",
    "build_single_member_zip",
    "classify_against_vendored",
    "code_block",
    "decode_block",
    "decode_renderer_frame",
    "decode_token_frame",
    "encode_renderer_frame",
    "encode_token_frame",
    "pack_config_section",
    "pack_container",
    "parse_payload",
    "unpack_config_section",
    "unpack_container",
    "zip_framing_overhead",
]

TOKEN_FRAME_MAGIC: Final = b"IX2TOK01"
RENDERER_FRAME_MAGIC: Final = b"IX2REN01"
CONTAINER_MAGIC: Final = b"IX2CNT01"

# Coder id is written into the frame, so the decoder never guesses.
CODER_NAMES: Final = ("stored", "deflate", "brotli", "lzma")
_LZMA_FILTERS: Final = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 1 << 24, "lc": 3, "lp": 0, "pb": 0}
]

_TOKEN_HEADER: Final = struct.Struct("<BBBBHHHHII")
_RENDERER_HEADER: Final = struct.Struct("<IIBI")
_CONFIG_HEADER: Final = struct.Struct("<fBBBB")
_PAYLOAD_HEADER: Final = struct.Struct("<IB")

# Counted-table encodings, tried in ascending size and admitted only when the
# round-trip is EXACT in float64.  ``ddm_cx1``: on ``pj2`` the FITTED ``st_grid``
# must be COUNTED (it is not the vendored ladder) and it is NOT representable in
# f16 — 0.06 / 0.065 / 0.07 are decimal literals.  Rounding them would move
# ``s_t``, hence the homography, hence the rendered frames, which would make this
# line LOSSY and forfeit the whole "d_seg and d_pose invariant by construction"
# exemption.  So the section carries a per-table FORMAT and refuses anything that
# does not reproduce the shipped float64 bit for bit.
TABLE_F16: Final = 0
TABLE_F32: Final = 1
TABLE_F64: Final = 2
TABLE_SCALED_INT: Final = 3
_MAX_DECIMAL_SCALE: Final = 9
# Indexed by the width byte written into the lane; tried in ascending itemsize so
# a non-negative table (``st_grid``) gets the unsigned lane rather than paying a
# sign bit it never uses.
_SCALED_INT_DTYPES: Final = ("<i1", "<u1", "<i2", "<u2", "<i4", "<u4")


class IX2ContainerError(ValueError):
    """Raised when a frame or container does not close exactly."""


# --------------------------------------------------------------------------- #
# generic block coding                                                         #
# --------------------------------------------------------------------------- #


def code_block(payload: bytes) -> tuple[int, bytes]:
    """Return ``(coder_id, coded)`` for the measured-smallest generic coder.

    ``stored`` is always a candidate so "already at entropy" is representable —
    on our token stream brotli is +5 B WORSE than stored, and a racer without a
    stored rung cannot say so.
    """

    candidates: list[bytes] = [
        bytes(payload),
        zlib.compress(payload, 9),
        brotli.compress(payload, quality=11, lgwin=24),
        lzma.compress(payload, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS),
    ]
    best = min(range(len(candidates)), key=lambda i: len(candidates[i]))
    return best, candidates[best]


def decode_block(coder_id: int, coded: bytes) -> bytes:
    if coder_id == 0:
        return bytes(coded)
    if coder_id == 1:
        return zlib.decompress(coded)
    if coder_id == 2:
        return brotli.decompress(coded)
    if coder_id == 3:
        return lzma.decompress(coded, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)
    raise IX2ContainerError(f"unknown coder id {coder_id}")


def _pack_nibbles(values: np.ndarray) -> bytes:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if flat.size and int(flat.max()) >= 16:
        raise IX2ContainerError("nibble lane values must be below 16")
    out = np.zeros((flat.size + 1) // 2, dtype=np.uint8)
    out[:] = flat[0::2] << 4
    if flat.size // 2:
        out[: flat.size // 2] |= flat[1::2]
    return out.tobytes()


def _unpack_nibbles(payload: bytes, count: int) -> np.ndarray:
    raw = np.frombuffer(payload, dtype=np.uint8)
    if raw.size < (count + 1) // 2:
        raise IX2ContainerError("nibble lane is truncated")
    out = np.empty(raw.size * 2, dtype=np.uint8)
    out[0::2] = raw >> 4
    out[1::2] = raw & 0x0F
    return np.ascontiguousarray(out[:count])


# --------------------------------------------------------------------------- #
# IX2TOK01 — token frame                                                       #
# --------------------------------------------------------------------------- #


def _factor_mode_delta(codes: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell temporal mode + modulo residual (the r7 factorisation).

    Reimplemented here so the container has no import edge into the experiments
    tree; ``test_mode_factorisation_matches_r7`` pins it to the r7 original.
    """

    value = np.ascontiguousarray(np.asarray(codes, dtype=np.uint8))
    flat = value.reshape(value.shape[0], -1)
    base = np.empty(flat.shape[1], dtype=np.uint8)
    for index in range(flat.shape[1]):
        base[index] = np.bincount(flat[:, index], minlength=levels).argmax()
    base = base.reshape(value.shape[1:])
    delta = (value.astype(np.int16) - base[None].astype(np.int16)) % levels
    return np.ascontiguousarray(base), np.ascontiguousarray(delta.astype(np.uint8))


def encode_token_frame(codes: np.ndarray, *, levels: int = 16) -> bytes:
    """Encode a ``(P,R,C,K)`` token lattice as a self-describing IX2TOK01 frame.

    The residual is stored **cell-major** ``(R,C,K,P)``: every pair's value for one
    cell is contiguous, which is what lets a stock coder LZ-copy the long
    same-as-before runs.  MEASURED on the live lattice: cell-major nibble 339,970 B
    vs native AoS nibble 395,026 B.
    """

    array = np.ascontiguousarray(np.asarray(codes, dtype=np.uint8))
    if array.ndim != 4:
        raise IX2ContainerError("token lattice must be 4-D (P,R,C,K)")
    if not 2 <= levels <= 16:
        raise IX2ContainerError("levels must be in [2, 16]")
    if array.size and int(array.max()) >= levels:
        raise IX2ContainerError("token code exceeds levels")
    p, r, c, k = array.shape
    if max(p, r, c, k) > 0xFFFF:
        raise IX2ContainerError("token lattice axis exceeds uint16")
    base, delta = _factor_mode_delta(array, levels)
    residual = np.ascontiguousarray(np.transpose(delta, (1, 2, 3, 0)))
    coder_r, block_r = code_block(_pack_nibbles(residual.reshape(-1)))
    coder_b, block_b = code_block(_pack_nibbles(base.reshape(-1)))
    header = _TOKEN_HEADER.pack(
        levels, coder_r, coder_b, 0, p, r, c, k, len(block_r), len(block_b)
    )
    return TOKEN_FRAME_MAGIC + header + block_r + block_b


def decode_token_frame(frame: bytes) -> np.ndarray:
    """Reconstruct the ``(P,R,C,K)`` lattice **from the frame alone**."""

    if frame[: len(TOKEN_FRAME_MAGIC)] != TOKEN_FRAME_MAGIC:
        raise IX2ContainerError("token frame magic differs")
    off = len(TOKEN_FRAME_MAGIC)
    if len(frame) < off + _TOKEN_HEADER.size:
        raise IX2ContainerError("token frame header is truncated")
    (levels, coder_r, coder_b, reserved, p, r, c, k, len_r, len_b) = (
        _TOKEN_HEADER.unpack_from(frame, off)
    )
    if reserved != 0:
        raise IX2ContainerError("token frame reserved byte must be zero")
    off += _TOKEN_HEADER.size
    block_r = frame[off : off + len_r]
    off += len_r
    block_b = frame[off : off + len_b]
    off += len_b
    if off != len(frame):
        raise IX2ContainerError(
            f"token frame has {len(frame) - off} unconsumed bytes"
        )
    if len(block_r) != len_r or len(block_b) != len_b:
        raise IX2ContainerError("token frame block is truncated")
    residual = _unpack_nibbles(decode_block(coder_r, block_r), r * c * k * p)
    base = _unpack_nibbles(decode_block(coder_b, block_b), r * c * k)
    delta = np.ascontiguousarray(
        np.transpose(residual.reshape(r, c, k, p), (3, 0, 1, 2))
    )
    value = (base.reshape(r, c, k)[None].astype(np.int16) + delta.astype(np.int16))
    return np.ascontiguousarray((value % levels).astype(np.uint8))


# --------------------------------------------------------------------------- #
# IX2REN01 — renderer frame                                                    #
# --------------------------------------------------------------------------- #


def encode_renderer_frame(mask_bits: np.ndarray, floats_f16_be: bytes) -> bytes:
    """Field-split renderer frame: binary lottery mask + f16 gain/bias table.

    MEASURED on the live member: the mask is 22,248 bits at density 0.4966 whose
    order-0 entropy is 2,780.9 B and whose combinatorial floor is 2,780.0 B — we
    ship 2,781 B, i.e. **one byte above the floor**, so the mask is incompressible
    and is kept stored.  The 246-value f16 table *is* compressible once it stops
    sharing a coder with the mask: byte-plane + raw-LZMA1 takes 492 → 464 B.
    """

    bits = np.asarray(mask_bits, dtype=np.uint8).reshape(-1)
    if bits.size and int(bits.max()) > 1:
        raise IX2ContainerError("renderer mask is not binary")
    if len(floats_f16_be) % 2:
        raise IX2ContainerError("f16 table must be an even number of bytes")
    packed = np.packbits(bits, bitorder="big").tobytes()
    words = np.frombuffer(floats_f16_be, dtype=">u2").astype(np.uint16)
    planes = (words >> 8).astype(np.uint8).tobytes() + (
        words & 0xFF
    ).astype(np.uint8).tobytes()
    coder, block = code_block(planes)
    header = _RENDERER_HEADER.pack(bits.size, words.size, coder, len(block))
    return RENDERER_FRAME_MAGIC + header + packed + block


def decode_renderer_frame(frame: bytes) -> tuple[np.ndarray, bytes]:
    """Return ``(mask_bits, floats_f16_be)`` from an IX2REN01 frame alone."""

    if frame[: len(RENDERER_FRAME_MAGIC)] != RENDERER_FRAME_MAGIC:
        raise IX2ContainerError("renderer frame magic differs")
    off = len(RENDERER_FRAME_MAGIC)
    if len(frame) < off + _RENDERER_HEADER.size:
        raise IX2ContainerError("renderer frame header is truncated")
    mask_count, float_count, coder, block_len = _RENDERER_HEADER.unpack_from(frame, off)
    off += _RENDERER_HEADER.size
    mask_bytes = (mask_count + 7) // 8
    packed = frame[off : off + mask_bytes]
    off += mask_bytes
    block = frame[off : off + block_len]
    off += block_len
    if off != len(frame):
        raise IX2ContainerError(
            f"renderer frame has {len(frame) - off} unconsumed bytes"
        )
    if len(packed) != mask_bytes or len(block) != block_len:
        raise IX2ContainerError("renderer frame is truncated")
    all_bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")
    if np.any(all_bits[mask_count:] != 0):
        raise IX2ContainerError("renderer mask padding carries counted inert bits")
    planes = decode_block(coder, block)
    if len(planes) != 2 * float_count:
        raise IX2ContainerError("renderer f16 planes do not close exactly")
    high = np.frombuffer(planes[:float_count], dtype=np.uint8).astype(np.uint16)
    low = np.frombuffer(planes[float_count:], dtype=np.uint8).astype(np.uint16)
    words = ((high << 8) | low).astype(">u2")
    return np.ascontiguousarray(all_bits[:mask_count]), words.tobytes()


# --------------------------------------------------------------------------- #
# counted configuration section                                                #
# --------------------------------------------------------------------------- #


def classify_against_vendored(
    value: Sequence[float] | None, vendored: Sequence[float]
) -> str:
    """``GENERIC`` iff the shipped table is byte-equal to the vendored one.

    **The generic-vs-video-derived test is a property of a (field, archive) PAIR,
    not of a field name.**  MEASURED by ``ddm_cp1``: ``st_grid`` equals the vendored
    ``ST_GRID`` on ``dc1_fold`` (generic there, migratable) and is a FITTED table on
    ``ms8`` / ``pj2`` (video-derived there, counted).  Same key, same slot, opposite
    rule-118 verdict.  So the classification is *computed per archive*, never
    asserted from the field name — which is what this function is for.
    """

    if value is None:
        return "ABSENT"
    shipped = np.asarray(value, dtype=np.float64)
    reference = np.asarray(vendored, dtype=np.float64)
    if shipped.shape == reference.shape and np.array_equal(shipped, reference):
        return "GENERIC"
    return "VIDEO_DERIVED"


def _check_table(values: np.ndarray, label: str) -> np.ndarray:
    if values.ndim != 1 or not 1 <= values.size <= 255:
        raise IX2ContainerError(f"{label} must be 1-D with 1..255 entries")
    if not np.all(np.isfinite(values)):
        raise IX2ContainerError(f"{label} carries a non-finite entry")
    return values


def _scaled_int_candidate(values: np.ndarray) -> bytes | None:
    """Smallest exact ``codes / 10**k`` encoding, or ``None`` when none exists.

    Fitted ladders are authored as DECIMAL literals in the manifest, so the
    natural exact lane is an integer over a power of ten — and it is also the
    narrowest: ``pj2``'s 11-entry ``st_grid`` is 11 signed bytes at ``k=3``
    against 22 B at f16 (which cannot represent it at all) and 88 B at f64.
    Exactness is VERIFIED per entry (``code / 10**k == v`` in float64), never
    assumed from integrality of ``v * 10**k`` — ``0.07 * 1000`` is
    ``70.00000000000001``, so the integrality test alone would reject a lane that
    is in fact exact, and a looser test would admit one that is not.
    """

    for k in range(_MAX_DECIMAL_SCALE + 1):
        scale = float(10**k)
        codes = np.rint(values * scale)
        if not np.array_equal(codes / scale, values):
            continue
        order = sorted(
            range(len(_SCALED_INT_DTYPES)),
            key=lambda i: (np.dtype(_SCALED_INT_DTYPES[i]).itemsize, i),
        )
        for index in order:
            dtype = _SCALED_INT_DTYPES[index]
            info = np.iinfo(dtype)
            if codes.min() < info.min or codes.max() > info.max:
                continue
            return bytes([k, index]) + codes.astype(dtype).tobytes()
        return None
    return None


def encode_exact_table(values: Sequence[float]) -> tuple[int, bytes]:
    """Return ``(format_id, payload)`` for the smallest EXACT encoding.

    "Exact" means the decoded float64 tuple equals the input float64 tuple
    element for element.  ``TABLE_F64`` always qualifies, so the ladder never
    fails closed on a finite table — it only ever picks something narrower when
    that narrower lane is provably lossless.
    """

    array = _check_table(np.asarray(values, dtype=np.float64), "table")
    candidates: list[tuple[int, int, bytes]] = []
    for fmt, dtype in ((TABLE_F16, "<f2"), (TABLE_F32, "<f4"), (TABLE_F64, "<f8")):
        blob = array.astype(dtype).tobytes()
        if np.array_equal(np.frombuffer(blob, dtype=dtype).astype(np.float64), array):
            candidates.append((len(blob), fmt, blob))
    scaled = _scaled_int_candidate(array)
    if scaled is not None:
        candidates.append((len(scaled), TABLE_SCALED_INT, scaled))
    if not candidates:  # pragma: no cover - f64 is always exact for finite input
        raise IX2ContainerError("no exact encoding for a finite table")
    size, fmt, blob = min(candidates)
    del size
    return fmt, blob


def decode_exact_table(
    fmt: int, payload: bytes, count: int, offset: int = 0
) -> tuple[tuple[float, ...], int]:
    """Return ``(values, bytes_consumed)``; every width is computable from ``fmt``."""

    if fmt == TABLE_SCALED_INT:
        if len(payload) < offset + 2:
            raise IX2ContainerError("scaled-int table header is truncated")
        k = payload[offset]
        index = payload[offset + 1]
        if k > _MAX_DECIMAL_SCALE:
            raise IX2ContainerError("scaled-int decimal scale out of range")
        if index >= len(_SCALED_INT_DTYPES):
            raise IX2ContainerError(f"scaled-int width id {index} is not supported")
        dtype = _SCALED_INT_DTYPES[index]
        need = 2 + np.dtype(dtype).itemsize * count
        if len(payload) < offset + need:
            raise IX2ContainerError("scaled-int table is truncated")
        codes = np.frombuffer(
            payload, dtype=dtype, count=count, offset=offset + 2
        ).astype(np.float64)
        return tuple(float(v) for v in codes / float(10**k)), need
    dtype = {TABLE_F16: "<f2", TABLE_F32: "<f4", TABLE_F64: "<f8"}.get(fmt)
    if dtype is None:
        raise IX2ContainerError(f"unknown table format {fmt}")
    width = np.dtype(dtype).itemsize
    need = width * count
    if len(payload) < offset + need:
        raise IX2ContainerError("table is truncated")
    values = np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
    return tuple(float(v) for v in values.astype(np.float64)), need


def pack_config_section(
    pose_dim0_offset: float,
    beta_mags: Sequence[float],
    st_grid: Sequence[float] | None = None,
    *,
    vendored_st_grid: Sequence[float] | None = None,
) -> bytes:
    """Pack the manifest values that are genuinely VIDEO-DERIVED, and only those.

    ``pose_dim0_offset`` is fitted to this clip.  ``beta_mags`` is a **fitted
    magnitude codebook** derived from this clip's solve by the builder on every v4d
    archive — counted, never migrated.  ``st_grid`` is the base-dependent case: pass
    the archive's table together with the vendored ladder and this function COUNTS it
    when they differ and OMITS it (one flag byte) when they match.  Everything else
    the live manifest carried (six ``sha256`` fields, ``tr1_metadata``, ``schema``,
    ``frame0_policy``, ``beta_idx_counts``, ``selector_num_two``) is generic,
    redundant, or unread and is not represented in this section at all.

    Passing ``st_grid`` without ``vendored_st_grid`` is refused: without the
    reference there is no way to decide, and "ambiguous ⇒ count it" cannot be
    applied by guessing.
    """

    mags = _check_table(np.asarray(beta_mags, dtype=np.float64), "beta_mags")
    packed = np.float32(pose_dim0_offset)
    if float(packed) != float(pose_dim0_offset):
        raise IX2ContainerError("pose_dim0_offset is not exact in f32")
    mags_fmt, mags_bytes = encode_exact_table(mags)
    grid_bytes = b""
    grid_count = 0
    grid_fmt = TABLE_F16
    if st_grid is not None:
        if vendored_st_grid is None:
            raise IX2ContainerError(
                "st_grid supplied without vendored_st_grid: the generic-vs-video-"
                "derived verdict is a (field, archive) pair property and cannot be "
                "decided without the reference table"
            )
        if classify_against_vendored(st_grid, vendored_st_grid) == "VIDEO_DERIVED":
            grid = _check_table(np.asarray(st_grid, dtype=np.float64), "st_grid")
            grid_fmt, grid_bytes = encode_exact_table(grid)
            grid_count = int(grid.size)
    return (
        _CONFIG_HEADER.pack(
            float(packed), int(mags.size), int(mags_fmt), grid_count, int(grid_fmt)
        )
        + mags_bytes
        + grid_bytes
    )


def unpack_config_section(
    payload: bytes,
) -> tuple[float, tuple[float, ...], tuple[float, ...] | None]:
    """Return ``(pose_dim0_offset, beta_mags, st_grid_or_None)``.

    ``None`` means "the vendored ladder" — the receiver substitutes its own generic
    constant, which is legal precisely because the encoder PROVED equality first.
    """

    if len(payload) < _CONFIG_HEADER.size:
        raise IX2ContainerError("config section is truncated")
    offset, count, mags_fmt, grid_count, grid_fmt = _CONFIG_HEADER.unpack_from(
        payload, 0
    )
    body = payload[_CONFIG_HEADER.size :]
    mags, used = decode_exact_table(mags_fmt, body, count)
    grid: tuple[float, ...] | None = None
    if grid_count:
        grid, grid_used = decode_exact_table(grid_fmt, body, grid_count, used)
        used += grid_used
    if used != len(body):
        raise IX2ContainerError("config section does not close exactly")
    return float(offset), mags, grid


# --------------------------------------------------------------------------- #
# IX2CNT01 — the single-member container                                       #
# --------------------------------------------------------------------------- #


def pack_container(sections: Sequence[bytes]) -> bytes:
    """One contiguous image: ``u32`` length per section except the last (implied).

    Section ORDER and section NAMES are the receiver's compile-time knowledge and
    live in ``inflate.py`` — generic, so they cost zero counted bytes.  The final
    length is omitted because it is computable, which is the same "the unpacker
    already knows" argument one level down.
    """

    if not sections:
        raise IX2ContainerError("container needs at least one section")
    if len(sections) > 0xFF:
        raise IX2ContainerError("container supports at most 255 sections")
    prefix = b"".join(struct.pack("<I", len(s)) for s in sections[:-1])
    return prefix + b"".join(bytes(s) for s in sections)


def unpack_container(payload: bytes, count: int) -> tuple[bytes, ...]:
    """Split a container into exactly ``count`` sections, closing on the last byte."""

    if count < 1:
        raise IX2ContainerError("count must be at least 1")
    head = 4 * (count - 1)
    if len(payload) < head:
        raise IX2ContainerError("container length table is truncated")
    lengths = [
        struct.unpack_from("<I", payload, 4 * i)[0] for i in range(count - 1)
    ]
    total = sum(lengths)
    if head + total > len(payload):
        raise IX2ContainerError("container sections overrun the payload")
    out: list[bytes] = []
    off = head
    for ln in lengths:
        out.append(payload[off : off + ln])
        off += ln
    out.append(payload[off:])
    consumed = head + sum(len(s) for s in out)
    if consumed != len(payload):
        raise IX2ContainerError(
            f"container has {len(payload) - consumed} unconsumed bytes"
        )
    return tuple(out)


def build_payload(bulk: bytes, joint: Sequence[bytes]) -> bytes:
    """Two-tier archive payload: one stored bulk section + one JOINTLY coded group.

    Six independently framed members each pay their own coder warm-up; a single
    stream over the small members shares one model.  MEASURED on the live archive:
    the five small members cost 13,157 B when each is coded independently (their
    shipped best) and **13,001 B under one coder** — −156 B purely from sharing the
    state, which is CLAUDE.md L23's split-stream observation read from the other
    direction (split when the distributions differ; share when they are small).

    ``bulk`` is the token frame: 96% of the archive and already at entropy, so it is
    stored and kept OUT of the joint stream, where it would only dilute the model.
    """

    if not joint:
        raise IX2ContainerError("joint group needs at least one section")
    coder_id, block = code_block(pack_container(list(joint)))
    return (
        _PAYLOAD_HEADER.pack(len(bulk), coder_id)
        + bytes(bulk)
        + bytes([len(joint)])
        + block
    )


def parse_payload(payload: bytes) -> tuple[bytes, tuple[bytes, ...]]:
    """Return ``(bulk, joint_sections)``, closing exactly on the final byte."""

    if len(payload) < _PAYLOAD_HEADER.size + 1:
        raise IX2ContainerError("payload header is truncated")
    bulk_len, coder_id = _PAYLOAD_HEADER.unpack_from(payload, 0)
    off = _PAYLOAD_HEADER.size
    bulk = payload[off : off + bulk_len]
    off += bulk_len
    if len(bulk) != bulk_len or off >= len(payload):
        raise IX2ContainerError("payload bulk section is truncated")
    count = payload[off]
    off += 1
    sections = unpack_container(decode_block(coder_id, payload[off:]), count)
    return bulk, sections


def build_single_member_zip(payload: bytes, *, name: str = "0.bin") -> bytes:
    """Deterministic single-member STORED zip (fixed timestamp, fixed attrs)."""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        info = zipfile.ZipInfo(name)
        info.compress_type = zipfile.ZIP_STORED
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.external_attr = 0o600 << 16
        archive.writestr(info, payload)
    return buf.getvalue()


def zip_framing_overhead(names: Sequence[str], payload_sizes: Sequence[int]) -> int:
    """Counted ZIP bytes that carry no payload, for a given member set.

    A local header (30 B) + central-directory entry (46 B) + the filename **twice**
    per member, plus one 22 B end-of-central-directory.  MEASURED on the live
    6-member archive: 674 B, i.e. ~112 B per member, of which up to 40 B is the
    filename alone (``state/pose_warp.stp`` is 20 characters).  Consolidating to one
    member named ``0.bin`` leaves 108 B.
    """

    if len(names) != len(payload_sizes):
        raise IX2ContainerError("names and payload_sizes must be the same length")
    per_member = sum(30 + 46 + 2 * len(n.encode()) for n in names)
    return per_member + 22
