"""PR92 ``qzs3_range_joint_r258`` RMC1 / RSA1 / RSB1 joint-stream primitives.

PR92's ``submissions/qzs3_range_joint_r258/inflate.py`` upgraded PR81 with
JOINT range coding of correlated mask + action streams (RMC1 magic) and
side actions (RSA1 / RSB1 magics with Brotli fallback). The novel insight
is the META-CODEC pattern: when two byte streams are correlated, encoding
their JOIN as one composite payload is strictly smaller than encoding each
independently. This module surfaces the magic-prefix grammar so future
archives that have a similar correlated-pair (e.g. pose-residual +
flow-residual; sidecar latent + per-pair delta) can opt into the same
framing.

Three magic-prefix grammars land here:

1. **RMC1 — Range Mask Composite** (:func:`pack_rmc1_composite` /
   :func:`unpack_rmc1_composite`)

   Frame: ``b"RMC1" || <uint32 seg_len LE> || <uint32 side_len LE> ||
   seg_bytes || side_bytes``. Either component may be empty.

2. **RSA1 — Side Action (range-coded)** (:func:`pack_rsa1_side` /
   :func:`unpack_rsa1_side`)

   Frame: ``b"RSA1" || <uint16 count LE> || <uint8 action_bits> ||
   <uint8 table_id> || packed_bits``. The packed-bits layout matches PR81's
   ROUTER_ACTION packing (LSB-first), exposed as
   :func:`tac.packet_compiler.encode_router_actions` for the body bytes.

3. **RSB1 — Side Action (brotli-fallback)** (:func:`pack_rsb1_side` /
   :func:`unpack_rsb1_side`)

   Frame: ``b"RSB1" || <uint16 count LE> || <uint8 table_id> || <uint8 0> ||
   brotli(body_bytes)`` — body is exactly ``count`` raw uint8 actions
   compressed under Brotli. PR92 uses this when the action distribution is
   too peaked for the small fixed-width RSA1 path.

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr92_intake_20260505_auto/source/submissions/qzs3_range_joint_r258/inflate.py``
(constants ``COMPOSITE_ACTION_MAGIC``, ``SIDE_ACTION_MAGIC``,
``SIDE_ACTION_BROTLI_MAGIC``; parser ``parse_local_actions``,
``unpack_router_actions``).

CLAUDE.md compliance
====================

* No scorer load — pure numpy + brotli + struct + stdlib.
* No MPS / torch import.
* No ``/tmp`` paths.
* Frozen dataclasses; ``encode → decode`` is bit-exact on the
  ``pr92_rmc_joint_stream_v1`` golden vector.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import brotli
import numpy as np

#: 4-byte magic identifying a RMC1 composite mask+action payload.
MAGIC_RMC1: bytes = b"RMC1"
#: 4-byte magic identifying a RSA1 range-coded side-action payload.
MAGIC_RSA1: bytes = b"RSA1"
#: 4-byte magic identifying a RSB1 brotli-compressed side-action payload.
MAGIC_RSB1: bytes = b"RSB1"


# ── Public dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class RMC1Composite:
    """Parsed RMC1 composite mask + side-action payload."""

    payload: bytes
    seg_bytes: bytes
    side_bytes: bytes


@dataclass(frozen=True)
class RSA1Side:
    """Parsed RSA1 side-action payload (range-coded)."""

    payload: bytes
    count: int
    action_bits: int
    table_id: int
    body: bytes


@dataclass(frozen=True)
class RSB1Side:
    """Parsed RSB1 side-action payload (brotli-compressed)."""

    payload: bytes
    count: int
    table_id: int
    body_bytes: bytes


# ── RMC1 composite ──────────────────────────────────────────────────────────


def pack_rmc1_composite(seg_bytes: bytes, side_bytes: bytes) -> RMC1Composite:
    """Pack two correlated byte streams under the RMC1 magic frame.

    Both inputs may be empty. Each length is stored as a little-endian
    uint32 immediately after the magic so the parser can split the
    concatenation back without external metadata.
    """
    seg_bytes = bytes(seg_bytes)
    side_bytes = bytes(side_bytes)
    if len(seg_bytes) > 0xFFFFFFFF or len(side_bytes) > 0xFFFFFFFF:
        raise ValueError(
            "RMC1 component must be < 2^32 bytes; got "
            f"seg={len(seg_bytes)} side={len(side_bytes)}"
        )
    header = struct.pack("<II", len(seg_bytes), len(side_bytes))
    payload = MAGIC_RMC1 + header + seg_bytes + side_bytes
    return RMC1Composite(payload=payload, seg_bytes=seg_bytes, side_bytes=side_bytes)


def unpack_rmc1_composite(payload: bytes) -> RMC1Composite:
    """Inverse of :func:`pack_rmc1_composite`."""
    payload = bytes(payload)
    if payload[:4] != MAGIC_RMC1:
        raise ValueError(f"missing RMC1 magic; got prefix {payload[:4]!r}")
    if len(payload) < 12:
        raise ValueError("truncated RMC1 header")
    seg_len, side_len = struct.unpack_from("<II", payload, 4)
    off = 12
    end = off + seg_len + side_len
    if end != len(payload):
        raise ValueError(
            f"RMC1 length mismatch: header={end} actual={len(payload)}"
        )
    seg_bytes = payload[off : off + seg_len]
    side_bytes = payload[off + seg_len : end]
    return RMC1Composite(payload=payload, seg_bytes=seg_bytes, side_bytes=side_bytes)


# ── RSA1 side-action ────────────────────────────────────────────────────────


def pack_rsa1_side(
    *, count: int, action_bits: int, table_id: int, body: bytes
) -> RSA1Side:
    """Frame an RSA1 side-action payload.

    Parameters
    ----------
    count:
        Number of actions encoded in ``body``. Stored as ``uint16``.
    action_bits:
        Bit-width per action (must satisfy ``1 <= action_bits <= 8`` per
        PR92).
    table_id:
        Caller-defined table identifier (PR92 uses small integers to choose
        among multiple action codebooks). Stored as ``uint8``.
    body:
        Bit-packed action stream (LSB-first; same layout as
        :func:`tac.packet_compiler.encode_router_actions`). Must contain at
        least ``ceil(count * action_bits / 8)`` bytes.

    Returns
    -------
    RSA1Side
        Container with the framed payload + the inputs (for diagnostics).
    """
    if not (0 <= count < (1 << 16)):
        raise ValueError(f"count must satisfy 0 <= count < 65536; got {count}")
    if not (1 <= action_bits <= 8):
        raise ValueError(
            f"action_bits must satisfy 1 <= action_bits <= 8; got {action_bits}"
        )
    if not (0 <= table_id < 256):
        raise ValueError(f"table_id must satisfy 0 <= table_id < 256; got {table_id}")
    body = bytes(body)
    required = (count * action_bits + 7) // 8
    if len(body) < required:
        raise ValueError(
            f"body too small for count={count} action_bits={action_bits}; "
            f"need >= {required} bytes, got {len(body)}"
        )
    header = struct.pack("<HBB", count, action_bits, table_id)
    payload = MAGIC_RSA1 + header + body
    return RSA1Side(
        payload=payload,
        count=int(count),
        action_bits=int(action_bits),
        table_id=int(table_id),
        body=body,
    )


def unpack_rsa1_side(payload: bytes) -> RSA1Side:
    """Inverse of :func:`pack_rsa1_side`."""
    payload = bytes(payload)
    if payload[:4] != MAGIC_RSA1:
        raise ValueError(f"missing RSA1 magic; got prefix {payload[:4]!r}")
    if len(payload) < 8:
        raise ValueError("truncated RSA1 side-action payload")
    count, action_bits, table_id = struct.unpack_from("<HBB", payload, 4)
    if not (1 <= action_bits <= 8):
        raise ValueError(
            f"unsupported RSA1 action_bits {action_bits}; expected 1..8"
        )
    body = payload[8:]
    return RSA1Side(
        payload=payload,
        count=int(count),
        action_bits=int(action_bits),
        table_id=int(table_id),
        body=body,
    )


# ── RSB1 side-action (brotli) ───────────────────────────────────────────────


def pack_rsb1_side(
    *,
    actions: np.ndarray,
    table_id: int = 0,
    brotli_quality: int = 11,
    brotli_lgwin: int = 22,
) -> RSB1Side:
    """Frame an RSB1 brotli-compressed side-action payload.

    Parameters
    ----------
    actions:
        1D ``uint8``-castable array of action ids. Each must fit in a byte.
        Stored brotli-compressed in the payload body.
    table_id:
        Caller-defined table identifier (PR92 uses small integers).
        Default 0.
    brotli_quality, brotli_lgwin:
        Brotli parameters (defaults match PR92).
    """
    arr = np.asarray(actions, dtype=np.int64).reshape(-1)
    count = arr.size
    if not (0 <= count < (1 << 16)):
        raise ValueError(f"count must satisfy 0 <= count < 65536; got {count}")
    if count and (int(arr.min()) < 0 or int(arr.max()) > 0xFF):
        raise ValueError(
            f"actions must fit in uint8; got min={int(arr.min())} max={int(arr.max())}"
        )
    if not (0 <= table_id < 256):
        raise ValueError(f"table_id must satisfy 0 <= table_id < 256; got {table_id}")
    body_bytes = arr.astype(np.uint8).tobytes()
    # PR92 header: count u16, table_id u8, reserved u8.
    header = struct.pack("<HBB", count, table_id, 0)
    compressed = brotli.compress(body_bytes, quality=brotli_quality, lgwin=brotli_lgwin)
    payload = MAGIC_RSB1 + header + compressed
    return RSB1Side(
        payload=payload,
        count=int(count),
        table_id=int(table_id),
        body_bytes=body_bytes,
    )


def unpack_rsb1_side(payload: bytes) -> RSB1Side:
    """Inverse of :func:`pack_rsb1_side`."""
    payload = bytes(payload)
    if payload[:4] != MAGIC_RSB1:
        raise ValueError(f"missing RSB1 magic; got prefix {payload[:4]!r}")
    if len(payload) < 8:
        raise ValueError("truncated RSB1 side-action payload")
    count, table_id, _reserved = struct.unpack_from("<HBB", payload, 4)
    compressed = payload[8:]
    body_bytes = brotli.decompress(compressed)
    if len(body_bytes) != count:
        raise ValueError(
            f"decoded {len(body_bytes)} RSB1 actions, expected {count}"
        )
    return RSB1Side(
        payload=payload,
        count=int(count),
        table_id=int(table_id),
        body_bytes=body_bytes,
    )


__all__ = [
    "MAGIC_RMC1",
    "MAGIC_RSA1",
    "MAGIC_RSB1",
    "RMC1Composite",
    "RSA1Side",
    "RSB1Side",
    "pack_rmc1_composite",
    "pack_rsa1_side",
    "pack_rsb1_side",
    "unpack_rmc1_composite",
    "unpack_rsa1_side",
    "unpack_rsb1_side",
]
