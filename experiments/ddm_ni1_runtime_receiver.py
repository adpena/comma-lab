"""Shipping-runtime adapter for the retained NR1 K32 task-cell quotient.

This file is copied byte-for-byte into a fresh DX2 runtime.  It accepts only
the single-member NI1A container, verifies the copied semantic, carrier, and
residual sections plus the counted NR1 packet, restores the shipping DX2
semantic/carrier objects through that runtime's own parser, and exposes the
NR1-decoded token field to the unchanged full-RGB renderer.

The old HPAC probability model is deliberately absent: QCTX and QPAIR are the
counted context and temporal consumers for this representation.  Keeping HPAC
as an output-inert paid section would violate the exact-consumer contract.
"""

from __future__ import annotations

import hashlib
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from . import nr1_taskcell_quotient as nr1
from .residual_archive import (
    FIXED_MAGIC,
    RX1_CODEC_BROTLI,
    RX1_MAGIC,
    RX1_MODEL_HEADER,
    ResidualArchiveError,
    ResidualArchiveParts,
    _decode_fixed_table,
    _decode_rx1_models,
)

NI1_MAGIC = b"NI1A"
NI1_VERSION = 1
NI1_FLAGS = 1  # bit 0: HPAC is replaced by counted NR1 context surfaces.
NI1_HEADER = struct.Struct("<4sBBHIIII32s32s32s32s32s")

DX2_ARCHIVE_SHA256 = (
    "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
)
DX2_RESERVED = 0x1A
DX2_TABLE_MODE = 0
EXPECTED_SHAPE = (600, 384, 512)
EXPECTED_NR1_PACKET_BYTES = 69_004
EXPECTED_NR1_PACKET_SHA256 = (
    "a68765dc683fa8302b560ef3db0d4a1507eeeccc695322fb8b69f684ed6dab28"
)
EXPECTED_DECODED_TOKEN_BYTES = 117_964_800
EXPECTED_DECODED_TOKEN_SHA256 = (
    "d416895a250ce79be7f485188d4f7dfd1690a269a250063c2f6bc5f48cf8b8d8"
)
# Deterministic Brotli-q11 encoding of the generic four-byte IHS1 format marker.
# It is algorithmic, video-independent runtime code, not learned/countable state.
GENERIC_HPAC_STREAM = bytes.fromhex("8b01804948533103")


class NI1ReceiverError(ValueError):
    """An NI1 container, paid-section, or receiver invariant failed closed."""


def _sha256_bytes(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_single_member(archive_path: Path) -> bytes:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            if archive.namelist() != ["p"]:
                raise NI1ReceiverError("NI1 archive must contain exactly member p")
            return archive.read("p")
    except zipfile.BadZipFile as error:
        raise NI1ReceiverError("NI1 archive is not a valid ZIP container") from error


def _split_outer(outer: bytes) -> tuple[bytes, bytes, bytes, bytes, str]:
    if len(outer) < NI1_HEADER.size:
        raise NI1ReceiverError("NI1 member is truncated")
    (
        magic,
        version,
        flags,
        reserved,
        semantic_bytes,
        carrier_bytes,
        residual_bytes,
        packet_bytes,
        source_sha,
        semantic_sha,
        carrier_sha,
        residual_sha,
        packet_sha,
    ) = NI1_HEADER.unpack_from(outer)
    if (magic, version, flags, reserved) != (
        NI1_MAGIC,
        NI1_VERSION,
        NI1_FLAGS,
        0,
    ):
        raise NI1ReceiverError("NI1 header metadata differs")
    lengths = (semantic_bytes, carrier_bytes, residual_bytes, packet_bytes)
    if min(lengths) <= 0 or len(outer) != NI1_HEADER.size + sum(lengths):
        raise NI1ReceiverError("NI1 paid-section lengths differ")
    if source_sha.hex() != DX2_ARCHIVE_SHA256:
        raise NI1ReceiverError("NI1 source-DX2 custody digest differs")

    cursor = NI1_HEADER.size
    sections = []
    for length, expected, name in zip(
        lengths,
        (semantic_sha, carrier_sha, residual_sha, packet_sha),
        ("semantic", "carrier", "residual", "NR1 packet"),
        strict=True,
    ):
        end = cursor + length
        payload = outer[cursor:end]
        cursor = end
        if hashlib.sha256(payload).digest() != expected:
            raise NI1ReceiverError(f"NI1 {name} digest differs")
        sections.append(payload)
    semantic, carrier, residual, packet = sections
    if (
        len(packet) != EXPECTED_NR1_PACKET_BYTES
        or _sha256_bytes(packet) != EXPECTED_NR1_PACKET_SHA256
    ):
        raise NI1ReceiverError("NI1 counted packet custody pin differs")
    return semantic, carrier, residual, packet, source_sha.hex()


def read_ni1_archive(
    archive_path: Path,
) -> tuple[ResidualArchiveParts, nr1.DecodeResult, str, dict[str, Any]]:
    """Parse NI1A and restore the exact shipping DX2 non-token receiver state."""
    archive_path = Path(archive_path).resolve()
    outer = _read_single_member(archive_path)
    semantic_stream, carrier_stream, compact_fixed, packet, source_sha = _split_outer(
        outer
    )

    try:
        parsed = nr1.parse_packet(packet)
        attribution = nr1.physical_attribution(packet)
        decoded = nr1.decode_packet(packet)
    except nr1.NR1FormatError as error:
        raise NI1ReceiverError(f"NI1 strict NR1 parser refused the packet: {error}") from error
    if (parsed.pair_count, parsed.height, parsed.width) != EXPECTED_SHAPE:
        raise NI1ReceiverError("NI1 decoded-token geometry differs")
    decoded.trace.require_exact_once()
    decoded_digest = _sha256_bytes(memoryview(decoded.tokens))
    if (
        decoded.tokens.nbytes != EXPECTED_DECODED_TOKEN_BYTES
        or decoded_digest != EXPECTED_DECODED_TOKEN_SHA256
    ):
        raise NI1ReceiverError("NI1 decoded-token payload differs")

    # Exercise the actual shipping RX1 inverse for every retained non-token
    # section.  IHS1 is a generic format marker generated at decode time; it
    # contains no learned or video-derived probability state.
    synthetic_rx1 = (
        RX1_MODEL_HEADER.pack(
            RX1_MAGIC,
            1,
            RX1_CODEC_BROTLI,
            DX2_TABLE_MODE,
            DX2_RESERVED,
            len(GENERIC_HPAC_STREAM),
            len(semantic_stream),
            len(carrier_stream),
        )
        + GENERIC_HPAC_STREAM
        + semantic_stream
        + carrier_stream
        + compact_fixed
        + packet
    )
    try:
        restored = _decode_rx1_models(synthetic_rx1)
    except ResidualArchiveError as error:
        raise NI1ReceiverError(
            f"shipping DX2 semantic/carrier inverse refused NI1: {error}"
        ) from error
    if restored is None:
        raise NI1ReceiverError("shipping parser did not recognize the NI1 bridge")
    semantic, carrier, hpac, compensation, section, compressed = restored
    if section != compact_fixed + packet:
        raise NI1ReceiverError("shipping parser changed the NI1 residual/token boundary")
    if hpac != b"IHS1":
        raise NI1ReceiverError("generated generic HPAC marker did not round-trip")

    residual = FIXED_MAGIC + compact_fixed
    try:
        table = _decode_fixed_table(residual)
    except ResidualArchiveError as error:
        raise NI1ReceiverError("NI1 copied residual table is not canonical") from error
    parts = ResidualArchiveParts(
        semantic_blob=semantic,
        carrier_blob=carrier,
        hpac_blob=hpac,
        token_stream=packet,
        table=table,
        schema="fixed_boundary_int6",
        residual_payload=residual,
        compressed_models=compressed,
        compensation_blob=compensation,
        token_codec="nr1q-k32",
    )
    trace = {
        "QPARAM": decoded.trace.qparam,
        "QCTX": decoded.trace.qctx,
        "QPAIR": decoded.trace.qpair,
        "QEVENT": decoded.trace.qevent,
    }
    report = {
        "schema": "ddm_ni1_nr1_k32_shipping_bridge.v1",
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "member_bytes": len(outer),
        "member_sha256": _sha256_bytes(outer),
        "source_dx2_archive_sha256": source_sha,
        "nr1_packet_bytes": len(packet),
        "nr1_packet_sha256": _sha256_bytes(packet),
        "decoded_token_sha256": decoded_digest,
        "decoded_token_shape": list(decoded.tokens.shape),
        "exact_once_consumption": trace,
        "physical_attribution": {
            section.value: {"start": start, "end": end, "bytes": end - start}
            for section, (start, end) in attribution.items()
        },
        "shipping_receiver_surface": {
            "parser": "runtime.residual_archive._decode_rx1_models",
            "semantic_bytes_restored": len(semantic),
            "carrier_bytes_restored": len(carrier),
            "residual_bytes_restored": len(residual),
            "compensation_bytes_restored": (
                None if compensation is None else len(compensation)
            ),
            "hpac_video_derived_bytes": 0,
            "hpac_replacement": "QCTX+QPAIR",
        },
    }
    return parts, decoded, decoded_digest, report


def decode_ni1_tokens(
    decoded: nr1.DecodeResult,
    expected_digest: str,
) -> tuple[Any, dict[str, Any]]:
    """Return the exact uint8 NR1 field consumed by the shipping renderer."""
    decoded.trace.require_exact_once()
    tokens = np.asarray(decoded.tokens)
    actual_digest = _sha256_bytes(memoryview(tokens))
    if (
        tokens.shape != EXPECTED_SHAPE
        or tokens.dtype != np.uint8
        or tokens.nbytes != EXPECTED_DECODED_TOKEN_BYTES
        or actual_digest != expected_digest
        or actual_digest != EXPECTED_DECODED_TOKEN_SHA256
    ):
        raise NI1ReceiverError("NI1 materialized token field differs")
    tokens.setflags(write=True)
    import torch

    return torch.from_numpy(tokens), {
        "schema": "ddm_ni1_nr1_k32_token_decode.v1",
        "decoder": "task-cell-quotient-independent-receiver",
        "token_codec": "nr1q-k32",
        "token_shape": list(tokens.shape),
        "token_bytes": tokens.nbytes,
        "token_sha256": actual_digest,
        "exact_once_consumption": {
            "QPARAM": decoded.trace.qparam,
            "QCTX": decoded.trace.qctx,
            "QPAIR": decoded.trace.qpair,
            "QEVENT": decoded.trace.qevent,
        },
    }
