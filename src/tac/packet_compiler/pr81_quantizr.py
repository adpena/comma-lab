"""PR81 ``qzs3`` Quantizr primitives — FP4 codebook + ROUTER_ACTION packing.

Two primitives land here, extracted from
``submissions/qzs3_range_mask/inflate.py`` in PR81:

1. **PR81 FP4 codebook** (:class:`FP4Codebook`)

   PR81's Quantizr family quantises weights with an *asymmetric* 8-level
   positive codebook ``[0, 0.5, 1, 1.5, 2, 3, 4, 6]`` plus an explicit sign
   bit. Each value is encoded as a 4-bit nibble laid out as ``sssM`` where
   ``s`` is the sign bit and ``MMM`` is the 3-bit magnitude index. Weights
   are packed two nibbles per byte (``hi << 4 | lo``) and decoded against a
   per-block fp16 scale. This is the canonical block-FP weight codec the
   Quantizr lineage (PR81, PR91, PR92, PR93, PR97) shares.

   The :class:`FP4Codebook` dataclass surfaces ``quantize_to_nibbles`` +
   ``dequantize_from_nibbles`` + ``pack_nibbles`` + ``unpack_nibbles``
   so any caller can opt into PR81's codebook over `tac.quantization`'s
   symmetric FP4 variants.

2. **PR81 ROUTER_ACTION 3-bit packing** (:func:`encode_router_actions` /
   :func:`decode_router_actions`)

   PR81 packs 600 frame-level router decisions into 600 × 3 = 1800 bits =
   225 bytes (LSB-first stream). The action is a small integer in
   ``[0, 7]`` and the result is the smallest information-theoretically
   admissible bit-packed stream for an 8-symbol uniform alphabet.

   We generalise the count + bit-width through arguments so a caller can
   pack any small-integer per-frame stream (e.g. 1200 frames × 4 bits =
   600 bytes).

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr81_intake_20260505_auto/source/submissions/qzs3_range_mask/inflate.py``
(see ``FP4Codebook`` + ``unpack_router_actions``).

CLAUDE.md compliance
====================

* No scorer load — pure numpy + stdlib.
* No MPS / torch import.
* No ``/tmp`` paths.
* Frozen dataclass; ``encode → decode`` is bit-exact on the
  ``pr81_fp4_codebook_v1`` and ``pr81_router_action_v1`` golden vectors.
* OSS-friendly: public surface is the 5 names re-exported from
  ``tac.packet_compiler``; everything else is ``_``-prefixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Canonical PR81/Quantizr asymmetric positive-level table. Index 0 → 0.0
#: through index 7 → 6.0; the sign bit doubles the alphabet to 16 codes.
PR81_POS_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ── Public dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class FP4Codebook:
    """PR81-compatible asymmetric 8-level FP4 codebook with sign bit.

    Attributes
    ----------
    pos_levels:
        The 8-entry non-negative magnitude table (default: PR81's
        ``(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)``).
    """

    pos_levels: tuple[float, ...] = PR81_POS_LEVELS

    def __post_init__(self) -> None:
        if len(self.pos_levels) != 8:
            raise ValueError(
                f"pos_levels must have exactly 8 entries; got {len(self.pos_levels)}"
            )
        if any(level < 0 for level in self.pos_levels):
            raise ValueError(
                f"pos_levels must be non-negative; got {self.pos_levels}"
            )
        prev = -1.0
        for level in self.pos_levels:
            if not (level >= prev):
                raise ValueError(
                    "pos_levels must be non-decreasing; got "
                    f"{self.pos_levels}"
                )
            prev = level

    def levels_array(self) -> np.ndarray:
        """Return ``pos_levels`` as a numpy array."""
        return np.asarray(self.pos_levels, dtype=np.float32)

    def quantize(
        self,
        values: np.ndarray,
        *,
        scales: np.ndarray,
        block_size: int,
    ) -> np.ndarray:
        """Quantise float values to PR81 4-bit nibbles.

        Parameters
        ----------
        values:
            1D float array (length must be a multiple of ``block_size`` OR
            a multiple of ``block_size`` after padding with zeros; the
            caller is responsible for ensuring the padding semantic).
        scales:
            Per-block fp16/fp32 scale. ``len(scales) == ceil(values.size /
            block_size)``.
        block_size:
            Number of values per scale (PR81 uses 32).

        Returns
        -------
        np.ndarray
            ``uint8`` nibble array of length ``values.size``; each nibble is
            ``s<<3 | mag_idx`` with ``s ∈ {0, 1}`` and ``mag_idx ∈ [0, 7]``.
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0; got {block_size}")
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        scales = np.asarray(scales, dtype=np.float32).reshape(-1)
        n_blocks_needed = (flat.size + block_size - 1) // block_size
        if scales.size != n_blocks_needed:
            raise ValueError(
                f"scales has {scales.size} entries; need "
                f"{n_blocks_needed} for {flat.size} values @ block_size={block_size}"
            )
        if not np.all(scales > 0):
            raise ValueError("scales must be strictly positive")
        # Pad to a multiple of block_size for clean reshape.
        pad = n_blocks_needed * block_size - flat.size
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        blocks = flat.reshape(n_blocks_needed, block_size)
        scaled = blocks / scales[:, None]
        signs = (scaled < 0).astype(np.uint8)
        mags = np.abs(scaled)
        levels = self.levels_array()
        # mag_idx = argmin over level distances
        dist = np.abs(mags[..., None] - levels[None, None, :])
        mag_idx = np.argmin(dist, axis=-1).astype(np.uint8)
        nibbles = ((signs << 3) | (mag_idx & 0x7)).astype(np.uint8)
        return nibbles.reshape(-1)[: n_blocks_needed * block_size]

    def dequantize_from_nibbles(
        self,
        nibbles: np.ndarray,
        *,
        scales: np.ndarray,
        block_size: int,
        n_values: int | None = None,
    ) -> np.ndarray:
        """Inverse of :meth:`quantize`. Returns a 1D float32 array.

        Parameters
        ----------
        nibbles:
            ``uint8`` array of nibble codes (``s<<3 | mag_idx``).
        scales:
            Per-block fp16/fp32 scale parallel to ``nibbles`` blocks.
        block_size:
            Number of nibbles per scale.
        n_values:
            Optional length of the original (pre-padding) value stream.
            When supplied, the trailing padding is trimmed off so a caller
            that quantised an array with non-divisible length recovers
            exactly the original element count.

        Returns
        -------
        np.ndarray
            ``float32`` array of length ``min(n_values, nibbles.size)``.
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0; got {block_size}")
        raw = np.asarray(nibbles).reshape(-1)
        if raw.size and (int(raw.min()) < 0 or int(raw.max()) > 0xF):
            raise ValueError("nibble values must fit in 4 bits ([0, 15])")
        nib = raw.astype(np.uint8, copy=False)
        scales = np.asarray(scales, dtype=np.float32).reshape(-1)
        n_blocks_needed = (nib.size + block_size - 1) // block_size
        if scales.size != n_blocks_needed:
            raise ValueError(
                f"scales has {scales.size} entries; need "
                f"{n_blocks_needed} for {nib.size} nibbles @ block_size={block_size}"
            )
        # Pad nibbles up to block_size multiple, then reshape.
        pad = n_blocks_needed * block_size - nib.size
        if pad:
            nib = np.concatenate([nib, np.zeros(pad, dtype=np.uint8)])
        blocks = nib.reshape(n_blocks_needed, block_size)
        signs = (blocks >> 3) & 0x1
        mag_idx = (blocks & 0x7).astype(np.int64)
        levels = self.levels_array()
        q = levels[mag_idx]
        q = np.where(signs.astype(bool), -q, q)
        dq = (q * scales[:, None]).astype(np.float32)
        flat = dq.reshape(-1)
        if n_values is not None:
            if n_values < 0:
                raise ValueError(f"n_values must be >= 0; got {n_values}")
            flat = flat[:n_values]
        return flat


# ── PR81 nibble packing helpers ─────────────────────────────────────────────


def pack_nibbles(nibbles: np.ndarray) -> bytes:
    """Pack a ``uint8`` nibble array (values in ``[0, 15]``) to bytes.

    Two nibbles per byte, ``hi << 4 | lo``. PR81's exact layout.
    Length must be even — if odd, the caller should pad with 0 first.
    """
    raw = np.asarray(nibbles).reshape(-1)
    if raw.size and (int(raw.min()) < 0 or int(raw.max()) > 0xF):
        raise ValueError("nibble values must fit in 4 bits ([0, 15])")
    flat = raw.astype(np.uint8, copy=False)
    if flat.size & 1:
        raise ValueError(
            f"nibble count must be even for clean packing; got {flat.size}"
        )
    hi = flat[0::2].astype(np.uint8) & 0xF
    lo = flat[1::2].astype(np.uint8) & 0xF
    packed = ((hi << 4) | lo).astype(np.uint8)
    return packed.tobytes()


def unpack_nibbles(packed: bytes, count: int) -> np.ndarray:
    """Inverse of :func:`pack_nibbles`.

    Parameters
    ----------
    packed:
        Bytes carrying two nibbles per byte (hi-first).
    count:
        Number of nibbles to return (must satisfy ``count <= 2 * len(packed)``).

    Returns
    -------
    np.ndarray
        ``uint8`` array of length ``count``.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0; got {count}")
    if count > 2 * len(packed):
        raise ValueError(
            f"count {count} exceeds available nibbles {2 * len(packed)}"
        )
    flat = np.frombuffer(packed, dtype=np.uint8)
    hi = (flat >> 4) & 0xF
    lo = flat & 0xF
    out = np.empty(flat.size * 2, dtype=np.uint8)
    out[0::2] = hi
    out[1::2] = lo
    return out[:count]


# ── PR81 ROUTER_ACTION 3-bit packing ────────────────────────────────────────


def encode_router_actions(
    actions: np.ndarray,
    *,
    bits: int = 3,
) -> bytes:
    """Pack small-integer per-frame actions as an LSB-first bit stream.

    PR81 uses ``bits=3`` (8-class router) and ``count=600`` → 225 bytes.
    The bit stream is byte-aligned via padding to a whole byte at the end.

    Parameters
    ----------
    actions:
        ``uint8``-castable 1D array of action ids; each must fit in
        ``bits`` bits (``< 1 << bits``).
    bits:
        Per-action bit-width (default 3 per PR81). Must satisfy
        ``1 <= bits <= 8``.

    Returns
    -------
    bytes
        ``ceil(len(actions) * bits / 8)`` bytes of packed stream.
    """
    if not (1 <= bits <= 8):
        raise ValueError(f"bits must be in [1, 8]; got {bits}")
    arr = np.asarray(actions, dtype=np.int64).reshape(-1)
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= (1 << bits)):
        raise ValueError(
            f"action values out of range [0, {1 << bits}); "
            f"min={int(arr.min())} max={int(arr.max())}"
        )
    total_bits = arr.size * bits
    out_len = (total_bits + 7) // 8
    out = bytearray(out_len)
    acc = 0
    accbits = 0
    j = 0
    for value in arr:
        acc |= (int(value) & ((1 << bits) - 1)) << accbits
        accbits += bits
        while accbits >= 8:
            out[j] = acc & 0xFF
            acc >>= 8
            accbits -= 8
            j += 1
    if accbits > 0:
        out[j] = acc & 0xFF
        j += 1
    if j != out_len:
        raise AssertionError(
            f"packed length mismatch: wrote {j} bytes, expected {out_len}"
        )
    return bytes(out)


def decode_router_actions(
    payload: bytes,
    *,
    count: int,
    bits: int = 3,
) -> np.ndarray:
    """Inverse of :func:`encode_router_actions`. LSB-first.

    Parameters
    ----------
    payload:
        Packed byte stream produced by :func:`encode_router_actions`.
    count:
        Number of actions to recover. Must satisfy
        ``count * bits <= 8 * len(payload)``.
    bits:
        Per-action bit-width (default 3 per PR81).

    Returns
    -------
    np.ndarray
        ``uint8`` 1D array of length ``count``.
    """
    if not (1 <= bits <= 8):
        raise ValueError(f"bits must be in [1, 8]; got {bits}")
    if count < 0:
        raise ValueError(f"count must be >= 0; got {count}")
    total_bits = count * bits
    if total_bits > 8 * len(payload):
        raise ValueError(
            f"count={count} bits={bits} requires {total_bits} bits but "
            f"payload has only {8 * len(payload)} bits"
        )
    out = np.empty(count, dtype=np.uint8)
    acc = 0
    accbits = 0
    mask = (1 << bits) - 1
    j = 0
    for byte in payload:
        acc |= int(byte) << accbits
        accbits += 8
        while accbits >= bits and j < count:
            out[j] = acc & mask
            acc >>= bits
            accbits -= bits
            j += 1
        if j >= count:
            break
    if j != count:
        raise AssertionError(
            f"decoded {j} actions, expected {count}"
        )
    return out


__all__ = [
    "FP4Codebook",
    "PR81_POS_LEVELS",
    "decode_router_actions",
    "encode_router_actions",
    "pack_nibbles",
    "unpack_nibbles",
]
