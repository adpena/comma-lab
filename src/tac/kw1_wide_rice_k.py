"""KW1: a wide Rice-``k`` field for the packed CAP1 carrier metadata.

The shipped packed CAP1 metadata block is 40 bytes and stores the twelve Rice
parameters as ``k_base`` (one u8) plus **one bit per dimension**
(``residual_archive._restore_packed_cap1_metadata``, bytes ``[139]`` and
``[140:142]``).  One bit means the twelve ``k`` values must span at most two
adjacent integers, which is a real constraint on what the carrier can carry:

* the live pointer body sits at ``k = 5`` on all twelve dimensions;
* a dimension whose coefficients are all zero wants ``k = 0`` -- at ``k = 5``
  every one of its 600 symbols still pays five raw bypass bits under DX2, i.e.
  ``600 * 5 / 8 = 375`` bytes of pure waste per dropped dimension;
* ``k = 0`` is five away from ``k = 5``, so the one-bit field cannot express it.

KW1 stores the twelve ``k`` values as **4-bit absolute** values instead
(``12 * 4 = 48`` bits = 6 bytes) in place of ``k_base`` + the 1-bit field
(1 + 2 = 3 bytes), so the block grows 40 -> 43 bytes: **+3 bytes**, and any
``k`` in ``[0, 15]`` is representable per dimension with no base coupling.  The
receiver's existing domain check (``ks < 12``) still binds.

Absolute 4-bit is one byte more than the cheapest representable form (a 4-bit
``k_base`` nibble plus twelve 3-bit deltas would be 5 bytes, +2 B).  The extra
byte buys the removal of the base/span coupling entirely: a re-solve can never
produce a ``k`` vector this field cannot hold, so there is no fail-closed path
to get wrong at seal time.  One byte is 6.66e-07 score units.

Layout, relative to the start of the packed metadata block (unchanged fields
first, so the Huffman-length span the RR5 rider reads keeps its offsets):

===============  =========  =============  ===================================
field            shipped    KW1            note
===============  =========  =============  ===================================
``factor_base``  ``[0:1]``  ``[0:1]``      u8
``factors``      ``[1:12]``  ``[1:12]``    12 x 7 bits
``biases``       ``[12:21]`` ``[12:21]``   12 x 6 bits
``lengths``      ``[21:37]`` ``[21:37]``   32 x 4 bits (RR5 reads this span)
``k_base``       ``[37:38]`` --            dropped by KW1
``ks``           ``[38:40]`` ``[37:43]``   1 bit/dim -> 4 bits/dim, absolute
===============  =========  =============  ===================================

Both forms restore to the SAME canonical 80-byte metadata, so every stage of
the receiver downstream of the restore is bit-identical between them.

This file is both the encoder reference and the receiver implementation; the
builder copies these exact bytes into the candidate runtime tree and records the
sha256, following the RR5/DX2 precedent.
"""

from __future__ import annotations

import numpy as np

#: RX1 ``reserved`` bit selecting the wide-k packed metadata block.  ``0x80`` is
#: the only bit the shipped receiver leaves unclaimed (it pins
#: ``SZ1_RESERVED_KNOWN_BITS = 0x7F``); a receiver that understands KW1 widens
#: that mask to ``0xFF``.
KW1_RESERVED_WIDE_RICE_K = 0x80

CARRIER_DIM = 12
#: Bytes shared by both forms: factor_base + factors + biases + lengths.
KW1_PREFIX_BYTES = 37
#: Shipped packed metadata block size.
SHIPPED_PACKED_METADATA_BYTES = 40
#: KW1 packed metadata block size.
KW1_PACKED_METADATA_BYTES = KW1_PREFIX_BYTES + (CARRIER_DIM * 4 + 7) // 8  # 43
#: Canonical (restored) metadata: int16 factors + int8 biases + 32 lengths + ks.
CANONICAL_METADATA_BYTES = 2 * CARRIER_DIM + CARRIER_DIM + 32 + CARRIER_DIM  # 80
#: The receiver's Rice domain: ``_rice_encode`` searches ``range(12)``.
K_DOMAIN = 12


class Kw1Error(ValueError):
    """The packed metadata is malformed for the format it claims."""


def packed_metadata_bytes(reserved: int) -> int:
    """Packed CAP1 metadata block size implied by an RX1 ``reserved`` byte."""
    return (
        KW1_PACKED_METADATA_BYTES
        if int(reserved) & KW1_RESERVED_WIDE_RICE_K
        else SHIPPED_PACKED_METADATA_BYTES
    )


def _pack_nibbles(values: np.ndarray) -> bytes:
    """Pack ``CARRIER_DIM`` 4-bit values, low nibble first, matching the
    receiver's little-endian ``_unpack_unsigned`` bit order."""
    items = np.asarray(values, dtype=np.int64).reshape(-1)
    if items.shape != (CARRIER_DIM,):
        raise Kw1Error(f"expected {CARRIER_DIM} Rice parameters, got {items.shape}")
    if np.any(items < 0) or np.any(items >= 16):
        raise Kw1Error("Rice parameter escapes the 4-bit packed field")
    out = bytearray(CARRIER_DIM // 2)
    for index, value in enumerate(items.tolist()):
        byte, shift = divmod(index * 4, 8)
        out[byte] |= int(value) << shift
    return bytes(out)


def _unpack_nibbles(raw: bytes) -> np.ndarray:
    if len(raw) != CARRIER_DIM // 2:
        raise Kw1Error("wide Rice-k field has the wrong length")
    out = np.empty(CARRIER_DIM, dtype=np.int64)
    for index in range(CARRIER_DIM):
        byte, shift = divmod(index * 4, 8)
        out[index] = (raw[byte] >> shift) & 0x0F
    return out


def unpack_ks(metadata: bytes) -> np.ndarray:
    """Read the twelve Rice ``k`` values from EITHER packed form.

    The form is identified by the block length, so a caller that already knows
    the block boundary does not also have to carry the reserved bit.
    """
    block = bytes(metadata)
    if len(block) == SHIPPED_PACKED_METADATA_BYTES:
        base = block[37]
        deltas = _unpack_shipped_bits(block[38:40])
        ks = base + deltas
    elif len(block) == KW1_PACKED_METADATA_BYTES:
        ks = _unpack_nibbles(block[KW1_PREFIX_BYTES:])
    else:
        raise Kw1Error(
            f"packed CAP1 metadata must be {SHIPPED_PACKED_METADATA_BYTES} or "
            f"{KW1_PACKED_METADATA_BYTES} bytes, got {len(block)}"
        )
    if np.any(ks < 0) or np.any(ks >= K_DOMAIN):
        raise Kw1Error("Rice parameters are outside the receiver's fixed domain")
    return ks.astype(np.uint8)


def _unpack_shipped_bits(raw: bytes) -> np.ndarray:
    """The shipped 12 x 1-bit field, with its zero-padding check."""
    if len(raw) != 2:
        raise Kw1Error("shipped Rice-k field has the wrong length")
    if raw[1] >> 4:
        raise Kw1Error("shipped Rice-k field has nonzero padding")
    bits = int.from_bytes(raw, "little")
    return np.asarray([(bits >> i) & 1 for i in range(CARRIER_DIM)], dtype=np.int64)


def widen_metadata(metadata: bytes, ks: np.ndarray | None = None) -> bytes:
    """Rewrite a 40-byte shipped block as a 43-byte KW1 block.

    With ``ks`` omitted the shipped parameters are carried over unchanged, which
    makes the widening a pure re-framing: the restored canonical 80 bytes are
    then byte-identical to the shipped body's.
    """
    block = bytes(metadata)
    if len(block) != SHIPPED_PACKED_METADATA_BYTES:
        raise Kw1Error("widen_metadata expects the 40-byte shipped block")
    parameters = unpack_ks(block) if ks is None else np.asarray(ks, dtype=np.int64)
    if parameters.reshape(-1).shape != (CARRIER_DIM,):
        raise Kw1Error(f"expected {CARRIER_DIM} Rice parameters")
    if np.any(parameters < 0) or np.any(parameters >= K_DOMAIN):
        raise Kw1Error("Rice parameters are outside the receiver's fixed domain")
    return block[:KW1_PREFIX_BYTES] + _pack_nibbles(parameters)


def with_ks(metadata: bytes, ks: np.ndarray) -> bytes:
    """Return a KW1 block carrying ``ks``, from either input form."""
    block = bytes(metadata)
    if len(block) == SHIPPED_PACKED_METADATA_BYTES:
        return widen_metadata(block, ks)
    if len(block) != KW1_PACKED_METADATA_BYTES:
        raise Kw1Error("with_ks expects a 40-byte or 43-byte packed block")
    parameters = np.asarray(ks, dtype=np.int64).reshape(-1)
    if parameters.shape != (CARRIER_DIM,):
        raise Kw1Error(f"expected {CARRIER_DIM} Rice parameters")
    if np.any(parameters < 0) or np.any(parameters >= K_DOMAIN):
        raise Kw1Error("Rice parameters are outside the receiver's fixed domain")
    return block[:KW1_PREFIX_BYTES] + _pack_nibbles(parameters)


def restore_canonical_metadata(packed_block: bytes) -> bytes:
    """Expand either packed form into the canonical 80-byte metadata.

    Mirrors ``residual_archive._restore_packed_cap1_metadata`` field for field,
    so a KW1 body and a shipped body that carry the same parameters restore to
    byte-identical canonical metadata.
    """
    block = bytes(packed_block)
    factor_base = block[0]
    factors = factor_base + _unpack_unsigned(block[1:12], CARRIER_DIM, 7)
    bias_codes = _unpack_unsigned(block[12:21], CARRIER_DIM, 6)
    biases = np.where(bias_codes >= 32, bias_codes - 64, bias_codes).astype(np.int8)
    lengths = _unpack_unsigned(block[21:37], 32, 4).astype(np.uint8)
    ks = unpack_ks(block)
    if np.any(factors > 512) or np.any(biases < -16) or np.any(biases > 16):
        raise Kw1Error("packed CAP1 metadata exceeds canonical domains")
    canonical = (
        factors.astype("<i2").tobytes()
        + biases.tobytes()
        + lengths.tobytes()
        + ks.tobytes()
    )
    if len(canonical) != CANONICAL_METADATA_BYTES:
        raise Kw1Error("canonical metadata has the wrong length")
    return canonical


def _unpack_unsigned(raw: bytes, count: int, bits: int) -> np.ndarray:
    """Verbatim port of ``residual_archive._unpack_unsigned``."""
    if len(raw) != (count * bits + 7) // 8:
        raise Kw1Error("packed CAP1 field has the wrong length")
    if count * bits % 8 and raw[-1] >> (count * bits % 8):
        raise Kw1Error("packed CAP1 field has nonzero padding")
    output = np.empty(count, dtype=np.int16)
    for index in range(count):
        offset = index * bits
        byte, shift = divmod(offset, 8)
        word = raw[byte]
        if byte + 1 < len(raw):
            word |= raw[byte + 1] << 8
        output[index] = (word >> shift) & ((1 << bits) - 1)
    return output
