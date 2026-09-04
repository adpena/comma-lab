"""Byte-close a re-solved F26 pose carrier by splicing the archive the receiver's way.

ddm_up3 / task #1136.  ``ddm_up2`` solved the frame-0 pose coefficients against the
shipping (DALI-GT) objective and converged, but could not write the candidate
``archive.zip``.  Its two recorded blockers were both packaging, and both are resolved
here by reading the RECEIVER as the layout specification instead of guessing offsets.

The receiver decodes the shipped body every day, so its parse code IS the ground truth::

    archive.zip                                    (ZIP_STORED, single member ``p``)
      -> outer                                     runtime/residual_archive.py:449-452
      -> RX1 header + 3 model streams              residual_archive.py:161-178
      -> brotli(carrier_stream)                    residual_archive.py:185
      -> CK2 2-plane un-interleave (reserved&0x4)  residual_archive.py:188-189, :674
      -> packed CAP1 section                       residual_archive.py:195-202
         _restore_packed_cap1_metadata()           residual_archive.py:124-149
      -> canonical CAP1 blob (_restore_cap1)       residual_archive.py:361-381
      -> CPR1 canonical carrier (decode_cap1)      entropy/coefficient_ar1_codec.py:86
      -> Rice -> unzigzag -> restore_ar1_bias      -> the int12 coefficient codes

Every one of those steps is an exact, total inverse, so the whole chain runs backwards.

Two corrections to ``ddm_up2``'s blocker report, both measured here:

* ``k_base = packed[139]`` is the RECEIVER'S OWN offset and it is CORRECT for the
  packed carrier section (``residual_archive.py:139``).  On this body it reads **8**,
  and ``8 + ks_1bit`` yields exactly the shipped ``[9,9,9,8,8,9,9,9,9,9,9,9]``.  The
  ``177`` ``ddm_up2`` observed came from applying that offset to a DIFFERENT buffer.
  The defect was the buffer, not the literal.

* The 9,759-byte Rice payload is not "unlocatable" in the stored section.  It is the
  CAP1 residual payload and it sits at a DERIVED offset -- ``6 + 96 + 40 + basis_bytes``
  -- where ``basis_bytes`` comes from the section's own u24 bit count.  It is only
  invisible to a fixed-offset search because both the framing and the CK2 plane
  interleave move it.

This module therefore hardcodes no body-specific positional literal.  Every offset is
computed from the format's own length fields, and the forward packer is asserted
against the shipped bytes before it is trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RUNTIME = Path(
    "/Volumes/APDataStore/pact/ddm_to1/generations/to1_tail_override_r1"
)
DEFAULT_SOLVED = Path("/Volumes/APDataStore/pact/ddm_up2/retained/solved_codes.npy")

# The pointer body, re-read from ddm_to1's receipts (never inferred).
POINTER_ARCHIVE_SHA256 = (
    "50e561454b23026d3870f056747e848a49bd5f2b1e23930155d1281aeee91927"
)
POINTER_ARCHIVE_BYTES = 176_420

N_PAIRS = 600
CARRIER_DIM = 12

# Brotli parameters, DISCOVERED by identity search over the shipped carrier stream and
# then asserted by the identity control on every splice.  They are not assumed.
BROTLI_QUALITY = 11
BROTLI_LGWIN = 24

# The receiver calls a bare ``brotli.decompress`` (residual_archive.py:91) and reads the
# CK2 carrier interleave from ``reserved`` bit 2 (:188), so BOTH are free encoder
# choices: any option below decodes to the same carrier body on the shipped receiver.
# The list is ORDERED and the search keeps the FIRST minimum, so option 0 -- the shipped
# container shape -- wins every tie and the byte-identity control stays exact.
# (ck2_carrier_plane2, quality, lgwin)
CONTAINER_OPTIONS: tuple[tuple[bool, int, int], ...] = (
    (True, BROTLI_QUALITY, BROTLI_LGWIN),  # 0: exactly what the shipped body used
    (True, 10, 16),
    (True, 9, 16),
    (False, BROTLI_QUALITY, BROTLI_LGWIN),
    (False, 11, 10),
    (False, 10, 16),
    (False, 10, 10),
    (False, 9, 16),
)

# Packed CAP1 metadata field widths, in bits per element, in stored order.  These come
# from the receiver's inverse at residual_archive.py:126-141; the byte offsets below are
# DERIVED from these widths rather than copied.
_PACKED_METADATA_FIELDS: tuple[tuple[str, int, int], ...] = (
    ("factor_base", 1, 8),  # (name, count, bits)
    ("factors", CARRIER_DIM, 7),
    ("biases", CARRIER_DIM, 6),
    ("lengths", 32, 4),
    ("k_base", 1, 8),
    ("ks", CARRIER_DIM, 1),
)


class Up3Error(RuntimeError):
    """The candidate cannot be byte-closed against the shipped container."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _field_bytes(count: int, bits: int) -> int:
    return (count * bits + 7) // 8


def packed_metadata_layout() -> dict[str, tuple[int, int]]:
    """Byte spans of the packed CAP1 metadata block, derived from the field widths."""
    layout: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, count, bits in _PACKED_METADATA_FIELDS:
        width = _field_bytes(count, bits)
        layout[name] = (offset, offset + width)
        offset += width
    return layout


PACKED_METADATA_BYTES = sum(
    _field_bytes(count, bits) for _name, count, bits in _PACKED_METADATA_FIELDS
)
# Canonical CAP1 carries these unpacked: int16 factors + int8 biases + 32 lengths + ks.
CANONICAL_METADATA_BYTES = 2 * CARRIER_DIM + CARRIER_DIM + 32 + CARRIER_DIM

BIT_COUNT_BYTES = 6  # two little-endian u24 counts: basis_bits, residual_bits
SCALES_BYTES = 8 * CARRIER_DIM
PACKED_METADATA_OFFSET = BIT_COUNT_BYTES + SCALES_BYTES


def _import_runtime(runtime_dir: Path):
    """Import the SHIPPED receiver modules; they are the layout specification."""
    runtime_dir = runtime_dir.resolve()
    sys.path.insert(0, str(runtime_dir))
    try:
        from runtime import carrier_repack as carrier_repack
        from runtime import residual_archive as residual_archive
        from runtime.entropy import (
            coefficient_ar1_codec as ar1_codec,
        )
        from runtime.entropy import (
            coefficient_predictor as predictor,
        )
    finally:
        sys.path.pop(0)
    return residual_archive, carrier_repack, ar1_codec, predictor


def _import_entropy_riders(
    runtime_dir: Path, *, require_rr5: bool, require_dx2: bool
):
    """Import the shipped RR5/DX2 forward and inverse byte transforms.

    These riders are optional receiver layers selected by RX1 reserved bits.  Keeping
    the imports runtime-bound makes the splice use the exact algorithms carried by the
    archive it is rebuilding instead of a recalled copy from another generation.
    """
    runtime_dir = runtime_dir.resolve()
    sys.path.insert(0, str(runtime_dir))
    try:
        if require_rr5:
            from runtime import rr5_arith_basis as rr5
        else:
            rr5 = None
        if require_dx2:
            from runtime import dx2_cabac_coefficients as dx2
        else:
            dx2 = None
    finally:
        sys.path.pop(0)
    return rr5, dx2


def _restore_entropy_riders(
    body: bytes, *, reserved: int, runtime_dir: Path, residual_archive: Any
) -> bytes:
    """Follow the receiver's RR5-then-DX2 inverse order when those bits are set."""
    rr5_bit = int(getattr(residual_archive, "RR5_RESERVED_ARITH_BASIS", 0))
    dx2_bit = int(
        getattr(residual_archive, "DX2_RESERVED_CABAC_COEFFICIENTS", 0)
    )
    if not (reserved & (rr5_bit | dx2_bit)):
        return body
    rr5, dx2 = _import_entropy_riders(
        runtime_dir,
        require_rr5=bool(reserved & rr5_bit),
        require_dx2=bool(reserved & dx2_bit),
    )
    restored = body
    if reserved & rr5_bit:
        if rr5 is None:
            raise Up3Error("RR5 rider bit is set but its runtime module is absent")
        restored = rr5.restore_carrier_body(restored)
    if reserved & dx2_bit:
        if dx2 is None:
            raise Up3Error("DX2 rider bit is set but its runtime module is absent")
        restored = dx2.restore_carrier_body(restored)
    return restored


def _apply_entropy_riders(
    body: bytes, *, reserved: int, runtime_dir: Path, residual_archive: Any
) -> bytes:
    """Apply DX2 then RR5, the exact inverse of the receiver's restore order."""
    rr5_bit = int(getattr(residual_archive, "RR5_RESERVED_ARITH_BASIS", 0))
    dx2_bit = int(
        getattr(residual_archive, "DX2_RESERVED_CABAC_COEFFICIENTS", 0)
    )
    if not (reserved & (rr5_bit | dx2_bit)):
        return body
    rr5, dx2 = _import_entropy_riders(
        runtime_dir,
        require_rr5=bool(reserved & rr5_bit),
        require_dx2=bool(reserved & dx2_bit),
    )
    encoded = body
    if reserved & dx2_bit:
        if dx2 is None:
            raise Up3Error("DX2 rider bit is set but its runtime module is absent")
        encoded = bytes(dx2.apply_cabac_to_carrier_body(encoded)["body"])
    if reserved & rr5_bit:
        if rr5 is None:
            raise Up3Error("RR5 rider bit is set but its runtime module is absent")
        encoded = bytes(rr5.apply_rider_to_carrier_body(encoded)["body"])
    if _restore_entropy_riders(
        encoded,
        reserved=reserved,
        runtime_dir=runtime_dir,
        residual_archive=residual_archive,
    ) != body:
        raise Up3Error("RR5/DX2 carrier forward-inverse identity failed")
    return encoded


def _pack_unsigned(values: Sequence[int], count: int, bits: int) -> bytes:
    """Forward of ``residual_archive._unpack_unsigned`` (little-endian bit packing)."""
    values = [int(v) for v in values]
    if len(values) != count:
        raise Up3Error(f"expected {count} values to pack, got {len(values)}")
    limit = 1 << bits
    if any(v < 0 or v >= limit for v in values):
        raise Up3Error(f"value escapes the {bits}-bit packed field")
    out = bytearray(_field_bytes(count, bits))
    for index, value in enumerate(values):
        offset = index * bits
        byte, shift = divmod(offset, 8)
        word = value << shift
        out[byte] |= word & 0xFF
        if byte + 1 < len(out):
            out[byte + 1] |= (word >> 8) & 0xFF
    return bytes(out)


def pack_cap1_metadata(
    *,
    factors: np.ndarray,
    biases: np.ndarray,
    lengths: np.ndarray,
    ks: np.ndarray,
) -> bytes:
    """Build the 40-byte packed metadata block (inverse of the receiver's restore).

    ``factor_base`` and ``k_base`` are the minima, which is the only choice that keeps
    every residual inside its packed field width.
    """
    factors = np.asarray(factors, dtype=np.int64)
    biases = np.asarray(biases, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    ks = np.asarray(ks, dtype=np.int64).reshape(-1)

    factor_base = int(factors.min())
    if not 0 <= factor_base <= 255:
        raise Up3Error(f"factor_base {factor_base} does not fit u8")
    k_base = int(ks.min())
    if not 0 <= k_base <= 255:
        raise Up3Error(f"k_base {k_base} does not fit u8")
    bias_codes = np.where(biases < 0, biases + 64, biases)
    return b"".join(
        (
            bytes((factor_base,)),
            _pack_unsigned((factors - factor_base).tolist(), CARRIER_DIM, 7),
            _pack_unsigned(bias_codes.tolist(), CARRIER_DIM, 6),
            _pack_unsigned(lengths.tolist(), 32, 4),
            bytes((k_base,)),
            _pack_unsigned((ks - k_base).tolist(), CARRIER_DIM, 1),
        )
    )


def _ck2_interleave_planes(body: bytes) -> bytes:
    """Forward of ``residual_archive._ck2_uninterleave_planes`` (:674)."""
    span = len(body) & ~1
    planes = np.frombuffer(body[:span], dtype=np.uint8)
    return planes[0::2].tobytes() + planes[1::2].tobytes() + body[span:]


@dataclass(frozen=True)
class ShippedBody:
    """Every field of the shipped archive that a splice must preserve or replace."""

    archive_bytes: bytes
    archive_sha256: str
    zip_info: dict[str, Any]
    outer: bytes
    rx1_header: tuple
    hpac_stream: bytes
    semantic_stream: bytes
    carrier_stream: bytes
    section_tail: bytes
    carrier_raw: bytes  # brotli-decompressed, pre-CK2
    carrier_body: bytes  # post-CK2: the packed CAP1 section + selector tail
    basis_bits: int
    residual_bits: int
    basis_blob: bytes
    rice_payload: bytes
    scales: bytes
    factors: np.ndarray
    biases: np.ndarray
    lengths: np.ndarray
    ks: np.ndarray
    packed_metadata: bytes
    body_tail: bytes  # selector remainder after the packed CAP1 portion
    codes: np.ndarray

    @property
    def ck2_carrier(self) -> bool:
        return bool(self.rx1_header[4] & 0x04)


@dataclass(frozen=True)
class CarrierSection:
    """The packed CAP1 carrier section, split the way the RECEIVER splits it.

    The canonical surface for any tool that wants the carrier bytes out of an archive.
    It exists because three sites in the ``ddm_t1h`` pricers did this by hand as
    ``brotli.decompress(stream)[:PACKED_CAP1_SECTION_BYTES]``, which is correct only for
    a body whose ``reserved`` bit 2 is clear AND whose packed portion happens to equal
    that pinned constant.  On the to1/ck2 lineage both premises fail, silently: the body
    is still 2-plane interleaved, so byte 139 reads 177 instead of the true ``k_base`` 8.
    """

    reserved: int
    ck2_carrier: bool
    stream: bytes  # the brotli stream stored in the archive
    raw: bytes  # brotli-decompressed, still interleaved if ck2_carrier
    body: bytes  # post-CK2: what the receiver's offset arithmetic reads
    packed_portion: int  # DERIVED from the body's own u24 bit counts
    packed: bytes  # packed CAP1 section incl. the selector tail
    overlay: bytes  # trailing compensation overlay, or b""


def carrier_section_from_archive(
    archive_path: Path, runtime_dir: Path = DEFAULT_RUNTIME
) -> CarrierSection:
    """Extract the carrier section from ``archive_path`` exactly as the receiver does."""
    ra, _cr, _ar1, _cp = _import_runtime(runtime_dir)
    member = zipfile.ZipFile(archive_path).read("p")
    fields = ra.RX1_MODEL_HEADER.unpack_from(member)
    reserved, hpac_bytes, semantic_bytes, carrier_bytes = fields[4:]
    start = ra.RX1_MODEL_HEADER.size + hpac_bytes + semantic_bytes
    stream = member[start : start + carrier_bytes]
    raw = ra._decompress_brotli(stream)
    ck2 = bool(reserved & ra.CK2_RESERVED_CARRIER_PLANE2)
    body = ra._ck2_uninterleave_planes(raw) if ck2 else raw
    basis_bits = int.from_bytes(body[0:3], "little")
    residual_bits = int.from_bytes(body[3:6], "little")
    portion = (
        PACKED_METADATA_OFFSET
        + PACKED_METADATA_BYTES
        + (basis_bits + 7) // 8
        + (residual_bits + 7) // 8
    )
    if not 0 < portion <= len(body):
        raise Up3Error(
            f"derived packed CAP1 portion {portion} is outside the {len(body)}-byte body"
        )
    # The selector tail follows the packed portion; a compensation overlay, when present,
    # follows the selector and is magic-tagged (residual_archive.py:231-234).
    tail = body[portion:]
    selector_bytes = len(tail)
    if tail[9:].startswith(ra.COMPENSATION_MAGIC):
        selector_bytes = 9
    packed_len = portion + selector_bytes
    section = CarrierSection(
        reserved=reserved,
        ck2_carrier=ck2,
        stream=stream,
        raw=raw,
        body=body,
        packed_portion=portion,
        packed=body[:packed_len],
        overlay=body[packed_len:],
    )
    layout = packed_metadata_layout()
    k_base = section.packed[PACKED_METADATA_OFFSET + layout["k_base"][0]]
    if k_base >= 12:
        raise Up3Error(
            f"parsed Rice k_base {k_base} is outside the receiver's domain (<12); the "
            "carrier section was read from the wrong buffer -- apply the CK2 "
            "un-interleave and derive the packed portion from the body's u24 counts"
        )
    return section


def parse_shipped_body(
    runtime_dir: Path = DEFAULT_RUNTIME, *, verify_sha: bool = True
) -> ShippedBody:
    """Walk the receiver's decode chain and capture every field it reads."""
    ra, _cr, _ar1, cp = _import_runtime(runtime_dir)
    archive_path = runtime_dir / "archive.zip"
    archive_bytes = archive_path.read_bytes()
    archive_sha = _sha256(archive_bytes)
    if verify_sha and archive_sha != POINTER_ARCHIVE_SHA256:
        raise Up3Error(
            f"archive.zip is not the pointer body: {archive_sha} "
            f"!= {POINTER_ARCHIVE_SHA256}"
        )

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if archive.namelist() != ["p"]:
            raise Up3Error("archive must contain exactly member p")
        info = archive.getinfo("p")
        zip_info = {
            "date_time": tuple(info.date_time),
            "compress_type": int(info.compress_type),
            "external_attr": int(info.external_attr),
            "create_system": int(info.create_system),
        }
        outer = archive.read("p")

    header = ra.RX1_MODEL_HEADER.unpack_from(outer)
    magic, version, codec, _table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = header
    if magic != ra.RX1_MAGIC or version != 1 or codec != ra.RX1_CODEC_BROTLI:
        raise Up3Error(f"unsupported RX1 model: magic={magic!r} ver={version} codec={codec}")
    offset = ra.RX1_MODEL_HEADER.size
    hpac_stream = outer[offset : offset + hpac_bytes]
    offset += hpac_bytes
    semantic_stream = outer[offset : offset + semantic_bytes]
    offset += semantic_bytes
    carrier_stream = outer[offset : offset + carrier_bytes]
    offset += carrier_bytes
    section_tail = outer[offset:]

    carrier_raw = ra._decompress_brotli(carrier_stream)
    carrier_body = (
        ra._ck2_uninterleave_planes(carrier_raw)
        if reserved & ra.CK2_RESERVED_CARRIER_PLANE2
        else carrier_raw
    )
    carrier_body = _restore_entropy_riders(
        carrier_body,
        reserved=reserved,
        runtime_dir=runtime_dir,
        residual_archive=ra,
    )

    basis_bits = int.from_bytes(carrier_body[0:3], "little")
    residual_bits = int.from_bytes(carrier_body[3:6], "little")
    basis_len = (basis_bits + 7) // 8
    rice_len = (residual_bits + 7) // 8
    packed_portion = PACKED_METADATA_OFFSET + PACKED_METADATA_BYTES + basis_len + rice_len
    if len(carrier_body) < packed_portion:
        raise Up3Error("packed CAP1 section is truncated")

    scales = carrier_body[BIT_COUNT_BYTES:PACKED_METADATA_OFFSET]
    meta_start = PACKED_METADATA_OFFSET
    packed_metadata = carrier_body[meta_start : meta_start + PACKED_METADATA_BYTES]
    layout = packed_metadata_layout()
    factor_base = packed_metadata[layout["factor_base"][0]]
    factors = factor_base + ra._unpack_unsigned(
        packed_metadata[slice(*layout["factors"])], CARRIER_DIM, 7
    )
    bias_codes = ra._unpack_unsigned(
        packed_metadata[slice(*layout["biases"])], CARRIER_DIM, 6
    )
    biases = np.where(bias_codes >= 32, bias_codes - 64, bias_codes).astype(np.int8)
    lengths = ra._unpack_unsigned(
        packed_metadata[slice(*layout["lengths"])], 32, 4
    ).astype(np.uint8)
    k_base = packed_metadata[layout["k_base"][0]]
    ks = (k_base + ra._unpack_unsigned(
        packed_metadata[slice(*layout["ks"])], CARRIER_DIM, 1
    )).astype(np.uint8)

    basis_start = meta_start + PACKED_METADATA_BYTES
    basis_blob = carrier_body[basis_start : basis_start + basis_len]
    rice_payload = carrier_body[basis_start + basis_len : packed_portion]
    body_tail = carrier_body[packed_portion:]

    codes = decode_codes(
        rice_payload,
        residual_bits=residual_bits,
        ks=ks,
        factors=factors,
        biases=biases,
        runtime_dir=runtime_dir,
    )
    return ShippedBody(
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha,
        zip_info=zip_info,
        outer=outer,
        rx1_header=header,
        hpac_stream=hpac_stream,
        semantic_stream=semantic_stream,
        carrier_stream=carrier_stream,
        section_tail=section_tail,
        carrier_raw=carrier_raw,
        carrier_body=carrier_body,
        basis_bits=basis_bits,
        residual_bits=residual_bits,
        basis_blob=basis_blob,
        rice_payload=rice_payload,
        scales=scales,
        factors=np.asarray(factors, dtype=np.int16),
        biases=np.asarray(biases, dtype=np.int8),
        lengths=np.asarray(lengths, dtype=np.uint8),
        ks=np.asarray(ks, dtype=np.uint8),
        packed_metadata=packed_metadata,
        body_tail=body_tail,
        codes=np.asarray(codes, dtype=np.int32),
    )


def decode_codes(
    rice_payload: bytes,
    *,
    residual_bits: int,
    ks: np.ndarray,
    factors: np.ndarray,
    biases: np.ndarray,
    runtime_dir: Path = DEFAULT_RUNTIME,
) -> np.ndarray:
    """Rice -> unzigzag -> AR(1)+bias restore, exactly as ``decode_cap1`` does."""
    _ra, cr, _ar1, cp = _import_runtime(runtime_dir)
    residuals = cr._unzigzag(
        cr._rice_decode(
            np.asarray(ks, dtype=np.int64).reshape(CARRIER_DIM, 1),
            rice_payload,
            residual_bits,
            N_PAIRS,
            CARRIER_DIM,
        )
    )
    model = cp.Ar1BiasModel(
        np.asarray(factors, dtype=np.int16), np.asarray(biases, dtype=np.int16)
    )
    return np.asarray(cp.restore_ar1_bias(residuals, model), dtype=np.int32)


def forward_ar1_bias(codes: np.ndarray, factors, biases, runtime_dir=DEFAULT_RUNTIME):
    """Inverse of ``coefficient_predictor.restore_ar1_bias`` (:95-107).

    ``restore`` runs ``out[i] = signed_mod(prediction(out[i-1]) + residual[i])``; the
    encoder therefore runs the same prediction off the ALREADY-KNOWN codes and
    subtracts.  The prediction depends only on ``codes[i-1]``, so no fixed point is
    involved and the inverse is exact.
    """
    _ra, _cr, _ar1, cp = _import_runtime(runtime_dir)
    values = np.asarray(codes, dtype=np.int32)
    factors_i16 = np.asarray(factors, dtype=np.int16)
    biases_i16 = np.asarray(biases, dtype=np.int16)
    out = np.empty_like(values)
    out[0] = values[0]
    for frame in range(1, values.shape[0]):
        prediction = cp.signed_mod(cp.round_q8(values[frame - 1], factors_i16) + biases_i16)
        out[frame] = cp.signed_mod(values[frame] - prediction)
    return out


def encode_codes(
    codes: np.ndarray, *, factors, biases, runtime_dir: Path = DEFAULT_RUNTIME
) -> tuple[np.ndarray, bytes, int]:
    """Codes -> AR(1) residuals -> zigzag -> Rice.  Returns (ks, payload, bit_count)."""
    _ra, cr, _ar1, _cp = _import_runtime(runtime_dir)
    residuals = forward_ar1_bias(codes, factors, biases, runtime_dir=runtime_dir)
    ks, payload, bit_count = cr._rice_encode(cr._zigzag(residuals), 1)
    return np.asarray(ks, dtype=np.uint8).reshape(-1), payload, int(bit_count)


def build_archive(
    body: ShippedBody,
    codes: np.ndarray,
    *,
    runtime_dir: Path = DEFAULT_RUNTIME,
    container_search: bool = False,
    container_options: Sequence[tuple[bool, int, int]] | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    """Rebuild ``archive.zip`` carrying ``codes``, running every layer backwards.

    With ``container_search`` the encoder-only choices (brotli parameters and the CK2
    carrier interleave) are searched over ``CONTAINER_OPTIONS`` and the smallest stream
    wins, ties going to the shipped shape.  Every candidate is verified to decompress
    back to the exact carrier bytes before it can be selected.

    ``container_options`` overrides BOTH branches with an explicit ordered list.
    It exists because ``BROTLI_QUALITY``/``BROTLI_LGWIN`` above are this module's
    OWN generation's shipped shape, and a later body may ship a different one:
    ``ddm_fs1`` measured the F26 semantic-joint body at q=9/lgwin=16, where
    q=11 costs 2 bytes MORE.  Defaulting a successor body to q=11 would silently
    lose those 2 bytes and, worse, break the byte-identity control the whole
    splice is anchored on
    ([[binding-instruction-numbers-expire-and-nobody-rederives-them]]).

    With ``verify`` (the default) the finished bytes are parsed back through the
    receiver and refused unless they decode to exactly ``codes``.  The builder must not
    be able to hand out bytes it has not proven, so the caller cannot forget the check.
    """
    import brotli

    ra, _cr, _ar1, _cp = _import_runtime(runtime_dir)
    codes = np.asarray(codes, dtype=np.int32)
    if codes.shape != (N_PAIRS, CARRIER_DIM):
        raise Up3Error(f"codes must be ({N_PAIRS}, {CARRIER_DIM}), got {codes.shape}")
    if codes.min() < -2048 or codes.max() > 2047:
        raise Up3Error("codes escape the CPR1 signed-int12 domain")

    ks, rice_payload, residual_bits = encode_codes(
        codes, factors=body.factors, biases=body.biases, runtime_dir=runtime_dir
    )
    # The packed k field is 1 bit per dimension over a u8 base, so the container can
    # only carry ks spanning at most 2 adjacent values.  This is a REAL constraint (it
    # is what ddm_t1h meant to check) and it fails closed.
    if int(ks.max()) - int(ks.min()) > 1:
        raise Up3Error(
            f"candidate Rice ks {ks.tolist()} span more than the packed 1-bit field"
        )
    if not 0 < residual_bits < (1 << 24):
        raise Up3Error(f"residual_bits {residual_bits} does not fit the CAP1 u24 field")
    packed_metadata = pack_cap1_metadata(
        factors=body.factors, biases=body.biases, lengths=body.lengths, ks=ks
    )
    carrier_body = b"".join(
        (
            body.basis_bits.to_bytes(3, "little"),
            residual_bits.to_bytes(3, "little"),
            body.scales,
            packed_metadata,
            body.basis_blob,
            rice_payload,
            body.body_tail,
        )
    )
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, _old = (
        body.rx1_header
    )
    encoded_carrier_body = _apply_entropy_riders(
        carrier_body,
        reserved=reserved,
        runtime_dir=runtime_dir,
        residual_archive=ra,
    )
    if container_options is not None:
        options = tuple(
            (bool(a), int(b), int(c)) for a, b, c in container_options
        )
        if not options:
            raise Up3Error("container_options was given but is empty")
    else:
        options = (
            CONTAINER_OPTIONS
            if container_search
            else ((body.ck2_carrier, BROTLI_QUALITY, BROTLI_LGWIN),)
        )
    chosen: tuple[bool, int, int] | None = None
    carrier_stream = b""
    for use_ck2, quality, lgwin in options:
        raw = (
            _ck2_interleave_planes(encoded_carrier_body)
            if use_ck2
            else encoded_carrier_body
        )
        stream = brotli.compress(raw, quality=quality, lgwin=lgwin)
        if brotli.decompress(stream) != raw:
            raise Up3Error(f"brotli round-trip failed for q={quality} lgwin={lgwin}")
        if chosen is None or len(stream) < len(carrier_stream):
            chosen, carrier_stream = (use_ck2, quality, lgwin), stream
    if chosen is None:
        raise Up3Error("no container option produced a stream")
    reserved = (
        reserved | ra.CK2_RESERVED_CARRIER_PLANE2
        if chosen[0]
        else reserved & ~ra.CK2_RESERVED_CARRIER_PLANE2
    )
    outer = b"".join(
        (
            ra.RX1_MODEL_HEADER.pack(
                magic,
                version,
                codec,
                table_mode,
                reserved,
                hpac_bytes,
                semantic_bytes,
                len(carrier_stream),
            ),
            body.hpac_stream,
            body.semantic_stream,
            carrier_stream,
            body.section_tail,
        )
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        entry = zipfile.ZipInfo("p", date_time=tuple(body.zip_info["date_time"]))
        entry.compress_type = body.zip_info["compress_type"]
        entry.external_attr = body.zip_info["external_attr"]
        entry.create_system = body.zip_info["create_system"]
        archive.writestr(entry, outer)
    archive_bytes = buffer.getvalue()
    if verify:
        try:
            recovered, _info = parse_back_codes(archive_bytes, runtime_dir=runtime_dir)
        except Exception as error:  # the receiver refused its own bytes
            raise Up3Error(
                f"written archive does not parse back through the receiver: {error}"
            ) from error
        if not np.array_equal(codes.astype(np.int64), recovered.astype(np.int64)):
            worst = int(
                np.abs(codes.astype(np.int64) - recovered.astype(np.int64)).max()
            )
            raise Up3Error(
                f"written archive does not parse back to the requested codes "
                f"(max |delta| {worst}); refusing to return unverified bytes"
            )
    return {
        "archive_bytes": archive_bytes,
        "archive_sha256": _sha256(archive_bytes),
        "archive_size": len(archive_bytes),
        "rice_ks": ks.tolist(),
        "rice_bits": residual_bits,
        "rice_payload_bytes": len(rice_payload),
        "carrier_body_bytes": len(carrier_body),
        "encoded_carrier_body_bytes": len(encoded_carrier_body),
        "carrier_stream_bytes": len(carrier_stream),
        "packed_metadata_identical": packed_metadata == body.packed_metadata,
        "rice_payload_identical": rice_payload == body.rice_payload,
        "delta_archive_bytes": len(archive_bytes) - POINTER_ARCHIVE_BYTES,
        "container": {
            "ck2_carrier_plane2": bool(chosen[0]),
            "brotli_quality": chosen[1],
            "brotli_lgwin": chosen[2],
            "rx1_reserved": f"{reserved:#x}",
            "searched": bool(container_search) or container_options is not None,
            "options_considered": [list(o) for o in options],
            "identical_to_shipped_shape": bool(chosen == options[0]),
            "shipped_shape": list(options[0]),
        },
    }


def parse_back_codes(archive_bytes: bytes, *, runtime_dir: Path = DEFAULT_RUNTIME):
    """Decode codes out of candidate archive bytes through the receiver's own path."""
    ra, cr, ar1, cp = _import_runtime(runtime_dir)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        outer = archive.read("p")
    header = ra.RX1_MODEL_HEADER.unpack_from(outer)
    reserved, hpac_bytes, semantic_bytes, carrier_bytes = header[4:]
    offset = ra.RX1_MODEL_HEADER.size + hpac_bytes + semantic_bytes
    carrier_stream = outer[offset : offset + carrier_bytes]
    body = ra._decompress_brotli(carrier_stream)
    if reserved & ra.CK2_RESERVED_CARRIER_PLANE2:
        body = ra._ck2_uninterleave_planes(body)
    body = _restore_entropy_riders(
        body,
        reserved=reserved,
        runtime_dir=runtime_dir,
        residual_archive=ra,
    )
    basis_bits = int.from_bytes(body[0:3], "little")
    residual_bits = int.from_bytes(body[3:6], "little")
    packed_portion = (
        PACKED_METADATA_OFFSET
        + PACKED_METADATA_BYTES
        + (basis_bits + 7) // 8
        + (residual_bits + 7) // 8
    )
    restored = ra._restore_packed_cap1_metadata(body[:packed_portion]) + body[packed_portion:]
    cap1 = ra._restore_cap1(restored[: ra._cap1_body_bytes(restored)])
    info = ar1.inspect_cap1(cap1, frames=N_PAIRS, dimensions=CARRIER_DIM)
    canonical = ar1.decode_cap1(cap1, frames=N_PAIRS, dimensions=CARRIER_DIM)
    cpr1_basis_bits, cpr1_rice_bits = struct.unpack_from("<II", canonical, 4)
    head = 4 + 8 + SCALES_BYTES + 32
    ks = np.frombuffer(canonical[head : head + CARRIER_DIM], dtype=np.uint8)
    rice_start = head + CARRIER_DIM + (cpr1_basis_bits + 7) // 8
    residuals = cr._unzigzag(
        cr._rice_decode(
            ks.astype(np.int64).reshape(CARRIER_DIM, 1),
            canonical[rice_start:],
            cpr1_rice_bits,
            N_PAIRS,
            CARRIER_DIM,
        )
    )
    codes = cp.signed_mod(np.cumsum(residuals.astype(np.int64), axis=0)).astype(np.int32)
    return codes, info


# --------------------------------------------------------------------------
# Controls -- each one is an executable falsifier of the splice, not a comment.
# --------------------------------------------------------------------------


def control_identity(runtime_dir: Path = DEFAULT_RUNTIME) -> dict[str, Any]:
    """Splice the SHIPPED codes back in; the archive must be byte-identical."""
    body = parse_shipped_body(runtime_dir)
    built = build_archive(body, body.codes, runtime_dir=runtime_dir)
    return {
        "control": "identity_shipped_codes_reproduce_pointer_archive",
        "expected_sha256": POINTER_ARCHIVE_SHA256,
        "observed_sha256": built["archive_sha256"],
        "byte_identical": built["archive_sha256"] == POINTER_ARCHIVE_SHA256,
        "expected_bytes": POINTER_ARCHIVE_BYTES,
        "observed_bytes": built["archive_size"],
        "packed_metadata_identical": built["packed_metadata_identical"],
        "rice_payload_identical": built["rice_payload_identical"],
        "rice_bits": built["rice_bits"],
        "rice_ks": built["rice_ks"],
    }


def control_roundtrip(
    codes: np.ndarray,
    runtime_dir: Path = DEFAULT_RUNTIME,
    *,
    container_search: bool = False,
) -> dict[str, Any]:
    """Splice candidate codes and parse them back out of the written bytes."""
    body = parse_shipped_body(runtime_dir)
    built = build_archive(
        body, codes, runtime_dir=runtime_dir, container_search=container_search
    )
    recovered, info = parse_back_codes(built["archive_bytes"], runtime_dir=runtime_dir)
    return {
        "control": "roundtrip_candidate_codes_survive_parse_back",
        "codes_exact": bool(np.array_equal(np.asarray(codes, np.int64), recovered.astype(np.int64))),
        "max_abs_code_delta": int(np.abs(np.asarray(codes, np.int64) - recovered.astype(np.int64)).max()),
        "archive_sha256": built["archive_sha256"],
        "archive_size": built["archive_size"],
        "delta_archive_bytes": built["delta_archive_bytes"],
        "rice_bits": built["rice_bits"],
        "rice_ks": built["rice_ks"],
        "parsed_back_rice_ks": info["rice_ks"],
        "parsed_back_rice_payload_bytes": info["rice_payload_bytes"],
        "container": built["container"],
    }


def control_determinism(
    codes: np.ndarray,
    runtime_dir: Path = DEFAULT_RUNTIME,
    *,
    container_search: bool = False,
) -> dict[str, Any]:
    """Compile twice; the bytes must be identical (deterministic reproducibility)."""
    body = parse_shipped_body(runtime_dir)
    first = build_archive(
        body, codes, runtime_dir=runtime_dir, container_search=container_search
    )
    second = build_archive(
        parse_shipped_body(runtime_dir),
        codes,
        runtime_dir=runtime_dir,
        container_search=container_search,
    )
    return {
        "control": "double_compile_determinism",
        "first_sha256": first["archive_sha256"],
        "second_sha256": second["archive_sha256"],
        "identical": first["archive_bytes"] == second["archive_bytes"],
    }


def _load_codes(path: Path) -> np.ndarray:
    codes = np.load(path)
    if codes.shape != (N_PAIRS, CARRIER_DIM):
        raise Up3Error(f"{path} has shape {codes.shape}, expected ({N_PAIRS}, {CARRIER_DIM})")
    return np.asarray(codes, dtype=np.int32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="report every parsed carrier field")
    inspect.add_argument("--json-out", type=Path)

    controls = sub.add_parser("controls", help="run identity + roundtrip + determinism")
    controls.add_argument("--codes", type=Path, default=DEFAULT_SOLVED)
    controls.add_argument("--container-search", action="store_true")
    controls.add_argument("--json-out", type=Path)

    splice = sub.add_parser("splice", help="write the candidate archive")
    splice.add_argument("--codes", type=Path, default=DEFAULT_SOLVED)
    splice.add_argument("--out", type=Path, required=True)
    splice.add_argument("--container-search", action="store_true")
    splice.add_argument("--json-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        body = parse_shipped_body(args.runtime_dir)
        report = {
            "archive_sha256": body.archive_sha256,
            "archive_bytes": len(body.archive_bytes),
            "rx1_reserved": f"{body.rx1_header[4]:#x}",
            "ck2_carrier_plane2": body.ck2_carrier,
            "basis_bits": body.basis_bits,
            "residual_bits": body.residual_bits,
            "rice_payload_bytes": len(body.rice_payload),
            "basis_bytes": len(body.basis_blob),
            "rice_ks": body.ks.tolist(),
            "factors_q8": body.factors.tolist(),
            "biases": body.biases.tolist(),
            "carrier_stream_bytes": len(body.carrier_stream),
            "carrier_body_bytes": len(body.carrier_body),
            "body_tail_bytes": len(body.body_tail),
            "packed_metadata_layout": {
                k: list(v) for k, v in packed_metadata_layout().items()
            },
        }
    elif args.command == "controls":
        codes = _load_codes(args.codes)
        report = {
            "codes_path": str(args.codes),
            "container_search": bool(args.container_search),
            "identity": control_identity(args.runtime_dir),
            "roundtrip": control_roundtrip(
                codes, args.runtime_dir, container_search=args.container_search
            ),
            "determinism": control_determinism(
                codes, args.runtime_dir, container_search=args.container_search
            ),
        }
    else:
        codes = _load_codes(args.codes)
        body = parse_shipped_body(args.runtime_dir)
        identity = control_identity(args.runtime_dir)
        if not identity["byte_identical"]:
            raise Up3Error("identity control failed; refusing to write a candidate")
        built = build_archive(
            body,
            codes,
            runtime_dir=args.runtime_dir,
            container_search=args.container_search,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(built["archive_bytes"])
        recovered, _info = parse_back_codes(
            args.out.read_bytes(), runtime_dir=args.runtime_dir
        )
        report = {
            "out": str(args.out),
            "archive_sha256": built["archive_sha256"],
            "archive_size": built["archive_size"],
            "delta_archive_bytes": built["delta_archive_bytes"],
            "rice_bits": built["rice_bits"],
            "rice_ks": built["rice_ks"],
            "container": built["container"],
            "identity_control": identity,
            "parse_back_codes_exact": bool(
                np.array_equal(codes.astype(np.int64), recovered.astype(np.int64))
            ),
            "coefficients_changed_vs_shipped": int((codes != body.codes).sum()),
        }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if getattr(args, "json_out", None):
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
