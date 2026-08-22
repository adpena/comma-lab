"""Shipping-runtime adapter for the retained RC1 terminal-program payload.

This file is copied byte-for-byte into a fresh DX2 runtime.  It accepts only
the single-member RC1A container, verifies every copied physical section and
the source-DX2 custody digest, restores the shipping DX2 semantic/carrier
objects through that runtime's own parser, and exposes RC1-decoded tokens to
the unchanged full-RGB renderer.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import brotli
import numpy as np

from . import rc1_terminal_program_vq as rc1
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

DX2_RESERVED = 0x1A
DX2_TABLE_MODE = 0
EXPECTED_ASSIGNMENT_SHAPE = (384, 512)
EXPECTED_CODEBOOK_SHAPE = (2_048, 600)
EXPECTED_RC1_PAYLOAD_BYTES = 59_884
EXPECTED_RC1_PAYLOAD_SHA256 = (
    "eab66bad9d113ed79475a810f4002ec821deb335c3e87fc1b1e90ef2b8e61164"
)
EXPECTED_DECODED_TOKEN_SHA256 = (
    "2c85d29698782b2b12f75a897665f80c59a40a9549f0697e18db16feaca93168"
)


class RI1ReceiverError(ValueError):
    """An RC1 container or receiver invariant failed closed."""


def _sha256_bytes(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_single_member(archive_path: Path) -> bytes:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            if archive.namelist() != ["p"]:
                raise RI1ReceiverError("RI1 archive must contain exactly member p")
            return archive.read("p")
    except zipfile.BadZipFile as error:
        raise RI1ReceiverError("RI1 archive is not a valid ZIP container") from error


def _rc1_payload_from_outer(outer: bytes) -> bytes:
    if len(outer) < rc1.SHADOW_HEADER.size:
        raise RI1ReceiverError("RI1 shadow member is truncated")
    fields = rc1.SHADOW_HEADER.unpack_from(outer)
    semantic_bytes, carrier_bytes, residual_bytes, rc1_bytes = fields[3:7]
    start = rc1.SHADOW_HEADER.size + semantic_bytes + carrier_bytes + residual_bytes
    payload = outer[start:]
    if len(payload) != rc1_bytes:
        raise RI1ReceiverError("RI1 payload length differs from its shadow header")
    return payload


def read_ri1_archive(
    archive_path: Path,
) -> tuple[ResidualArchiveParts, rc1.TokenVQModel, str, dict[str, Any]]:
    """Parse RC1A and restore the exact shipping DX2 non-token receiver state."""
    archive_path = Path(archive_path).resolve()
    outer = _read_single_member(archive_path)
    try:
        sections, model, decoded_digest = rc1.parse_shadow_outer(outer)
    except rc1.RC1FormatError as error:
        raise RI1ReceiverError(f"RI1 strict parser refused the member: {error}") from error
    if sections.source_archive_sha256 != rc1.DX2_ARCHIVE_SHA256:
        raise RI1ReceiverError("RI1 source-DX2 custody digest differs")
    if model.assignments.shape != EXPECTED_ASSIGNMENT_SHAPE:
        raise RI1ReceiverError("RI1 assignment lattice geometry differs")
    if model.codebook.shape != EXPECTED_CODEBOOK_SHAPE:
        raise RI1ReceiverError("RI1 terminal codebook geometry differs")
    if decoded_digest != EXPECTED_DECODED_TOKEN_SHA256:
        raise RI1ReceiverError("RI1 decoded-token digest differs from the selected row")

    rc1_payload = _rc1_payload_from_outer(outer)
    if (
        len(rc1_payload) != EXPECTED_RC1_PAYLOAD_BYTES
        or _sha256_bytes(rc1_payload) != EXPECTED_RC1_PAYLOAD_SHA256
    ):
        raise RI1ReceiverError("RI1 counted payload custody pin differs")

    # RC1 legitimately removes the learned HPAC stream because its terminal
    # program replaces that token model.  A four-byte generic IHS1 marker is
    # generated at decode time solely to exercise the actual shipping RX1
    # semantic/carrier inverse.  No video-derived bytes are hidden here.
    generic_hpac_stream = brotli.compress(b"IHS1", quality=11)
    synthetic_rx1 = (
        RX1_MODEL_HEADER.pack(
            RX1_MAGIC,
            1,
            RX1_CODEC_BROTLI,
            DX2_TABLE_MODE,
            DX2_RESERVED,
            len(generic_hpac_stream),
            len(sections.semantic),
            len(sections.carrier),
        )
        + generic_hpac_stream
        + sections.semantic
        + sections.carrier
        + sections.residual
        + rc1_payload
    )
    try:
        restored = _decode_rx1_models(synthetic_rx1)
    except ResidualArchiveError as error:
        raise RI1ReceiverError(
            f"shipping DX2 semantic/carrier inverse refused RI1: {error}"
        ) from error
    if restored is None:
        raise RI1ReceiverError("shipping DX2 RX1 parser did not recognize the RI1 bridge")
    semantic, carrier, hpac, compensation, section, compressed = restored
    if section != sections.residual + rc1_payload:
        raise RI1ReceiverError("shipping parser changed the RI1 residual/token boundary")
    if hpac != b"IHS1":
        raise RI1ReceiverError("generated generic HPAC marker did not round-trip")

    compact_fixed = sections.residual
    residual = FIXED_MAGIC + compact_fixed
    try:
        table = _decode_fixed_table(residual)
    except ResidualArchiveError as error:
        raise RI1ReceiverError("RI1 copied residual table is not shipping-canonical") from error
    parts = ResidualArchiveParts(
        semantic_blob=semantic,
        carrier_blob=carrier,
        hpac_blob=hpac,
        token_stream=rc1_payload,
        table=table,
        schema="fixed_boundary_int6",
        residual_payload=residual,
        compressed_models=compressed,
        compensation_blob=compensation,
        token_codec="rc1v",
    )
    report = {
        "schema": "ddm_ri1_rc1_shipping_bridge.v1",
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "member_bytes": len(outer),
        "member_sha256": _sha256_bytes(outer),
        "source_dx2_archive_sha256": sections.source_archive_sha256,
        "rc1_payload_bytes": len(rc1_payload),
        "rc1_payload_sha256": _sha256_bytes(rc1_payload),
        "decoded_token_sha256": decoded_digest,
        "assignment_shape": list(model.assignments.shape),
        "codebook_shape": list(model.codebook.shape),
        "shipping_receiver_surface": {
            "parser": "runtime.residual_archive._decode_rx1_models",
            "semantic_bytes_restored": len(semantic),
            "carrier_bytes_restored": len(carrier),
            "residual_bytes_restored": len(residual),
            "compensation_bytes_restored": (
                None if compensation is None else len(compensation)
            ),
        },
    }
    return parts, model, decoded_digest, report


def decode_ri1_tokens(
    model: rc1.TokenVQModel,
    expected_digest: str,
) -> tuple[Any, dict[str, Any]]:
    """Expand RC1 to the exact uint8 token tensor consumed by the renderer."""
    if expected_digest != EXPECTED_DECODED_TOKEN_SHA256:
        raise RI1ReceiverError("RI1 caller supplied a different decoded-token digest")
    tokens = np.empty(
        (EXPECTED_CODEBOOK_SHAPE[1], *EXPECTED_ASSIGNMENT_SHAPE),
        dtype=np.uint8,
    )
    digest = hashlib.sha256()
    frame_count = 0
    for frame_count, frame in enumerate(rc1.iter_decoded_frames(model), start=1):
        contiguous = np.ascontiguousarray(frame, dtype=np.uint8)
        tokens[frame_count - 1] = contiguous
        digest.update(memoryview(contiguous))
    if frame_count != EXPECTED_CODEBOOK_SHAPE[1]:
        raise RI1ReceiverError("RI1 receiver emitted the wrong frame count")
    actual_digest = digest.hexdigest()
    if actual_digest != expected_digest:
        raise RI1ReceiverError("RI1 materialized token digest differs")
    import torch

    return torch.from_numpy(tokens), {
        "schema": "ddm_ri1_rc1_token_decode.v1",
        "decoder": "terminal-program-vq-independent-receiver",
        "token_codec": "rc1v",
        "token_shape": list(tokens.shape),
        "token_bytes": tokens.nbytes,
        "token_sha256": actual_digest,
        "assignment_method": rc1.ASSIGNMENT_METHOD_NAMES[1],
        "codebook_method": rc1.CODEBOOK_METHOD_NAMES[102],
    }
