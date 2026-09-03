# SPDX-License-Identifier: MIT
"""Deterministic replacement of the consumed semantic section in an RX1 archive."""

from __future__ import annotations

import hashlib
import os
import struct
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import brotli

from experiments.ddm_sd1_semantic_rd_curve import pack_semantic_state

from .contracts import PipelineBlocked, file_fact

if TYPE_CHECKING:
    import torch

RX1_HEADER = struct.Struct("<4sBBBBHHH")
RX1_MAGIC = b"RX1M"
RX1_BROTLI = 2
SEMANTIC_PLANE2 = 0x02


def _interleave_planes(body: bytes) -> bytes:
    span = len(body) & ~1
    return body[:span:2] + body[1:span:2] + body[span:]


def _deterministic_zip(payload: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(temporary, "w", allowZip64=False) as archive:
        archive.writestr(info, payload)
    if (
        destination.is_file()
        and file_fact(temporary)["sha256"] != file_fact(destination)["sha256"]
    ):
        temporary.unlink()
        raise PipelineBlocked(f"refusing to overwrite a different retained archive: {destination}")
    os.replace(temporary, destination)


def replace_semantic_state(
    source_archive: Path,
    destination: Path,
    state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Serialize ``state`` and replace the RX1 semantic stream byte-for-byte."""

    with zipfile.ZipFile(source_archive) as archive:
        if archive.namelist() != ["p"]:
            raise PipelineBlocked("semantic source archive must contain only member p")
        outer = archive.read("p")
    if len(outer) < RX1_HEADER.size:
        raise PipelineBlocked("semantic source archive is truncated")
    magic, version, codec, table_mode, reserved, hpac_n, semantic_n, carrier_n = RX1_HEADER.unpack_from(outer)
    if magic != RX1_MAGIC or version != 1 or codec != RX1_BROTLI:
        raise PipelineBlocked("semantic replacement requires the RX1 Brotli representation")
    model_end = RX1_HEADER.size + hpac_n + semantic_n + carrier_n
    if model_end >= len(outer):
        raise PipelineBlocked("RX1 model sections do not leave a token tail")
    offset = RX1_HEADER.size
    hpac_stream = outer[offset : offset + hpac_n]
    offset += hpac_n
    old_semantic_stream = outer[offset : offset + semantic_n]
    offset += semantic_n
    carrier_stream = outer[offset : offset + carrier_n]
    tail = outer[model_end:]
    allocation = {name: 4 for name, value in state.items() if value.ndim >= 2}
    semantic_body, quantized_state = pack_semantic_state(state, allocation, legacy_int4=False)
    stored_body = _interleave_planes(semantic_body) if reserved & SEMANTIC_PLANE2 else semantic_body
    semantic_stream = brotli.compress(stored_body, quality=11, mode=brotli.MODE_GENERIC)
    if len(semantic_stream) > 0xFFFF:
        raise PipelineBlocked("trained semantic stream exceeds the RX1 uint16 field")
    header = RX1_HEADER.pack(
        magic,
        version,
        codec,
        table_mode,
        reserved,
        hpac_n,
        len(semantic_stream),
        carrier_n,
    )
    payload = header + hpac_stream + semantic_stream + carrier_stream + tail
    _deterministic_zip(payload, destination)
    repeat = destination.with_name(f"{destination.stem}.repeat{destination.suffix}")
    _deterministic_zip(payload, repeat)
    first = file_fact(destination)
    second = file_fact(repeat)
    if first["sha256"] != second["sha256"]:
        raise PipelineBlocked("fresh semantic archive is not deterministic")
    return {
        "archive": first,
        "archive_repeat": second,
        "repeat_byte_identical": True,
        "source_archive": file_fact(source_archive),
        "semantic_body_bytes": len(semantic_body),
        "semantic_body_sha256": hashlib.sha256(semantic_body).hexdigest(),
        "semantic_stream_bytes": len(semantic_stream),
        "semantic_stream_sha256": hashlib.sha256(semantic_stream).hexdigest(),
        "replaced_semantic_stream_bytes": len(old_semantic_stream),
        "preserved_hpac_stream_sha256": hashlib.sha256(hpac_stream).hexdigest(),
        "preserved_carrier_stream_sha256": hashlib.sha256(carrier_stream).hexdigest(),
        "preserved_tail_sha256": hashlib.sha256(tail).hexdigest(),
        "quantized_state": quantized_state,
    }
