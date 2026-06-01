# SPDX-License-Identifier: MIT
"""Section-neutralization helpers for PVQ PACT-NeRV-VQ archives."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

import torch

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.pact_nerv_vq.archive import (
    PVQ_HEADER_FMT,
    PVQ_HEADER_SIZE,
    pack_archive,
    parse_archive,
)

PVQ_SECTION_VALUE_PROFILE_SCHEMA = "hprc_mlx_component_neutralization_profile.v1"
PVQ_SECTION_VALUE_SOURCE_SCHEMA = "pact_nerv_vq_section_value_profile.v1"
PVQ_SUPPORTED_SECTION_NAMES = (
    "decoder_qw",
    "codebooks_q",
    "selectors_rc",
    "receiver_state",
)


@dataclass(frozen=True)
class PvqSectionLayoutRow:
    """Byte range for one PVQ logical section inside ``0.bin``."""

    name: str
    offset: int
    length: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "offset": self.offset,
            "length": self.length,
            "sha256": self.sha256,
        }


def pvq_section_layout(blob: bytes) -> list[PvqSectionLayoutRow]:
    """Return byte ranges for PVQ payload sections."""

    if len(blob) < PVQ_HEADER_SIZE:
        raise ValueError(f"PVQ blob too short: {len(blob)}")
    (
        _magic,
        _version,
        _latent_dim,
        _num_pairs,
        _codebook_size,
        decoder_len,
        codebook_len,
        indices_len,
        meta_len,
    ) = struct.unpack(PVQ_HEADER_FMT, blob[:PVQ_HEADER_SIZE])
    ranges = [
        ("decoder_qw", PVQ_HEADER_SIZE, int(decoder_len)),
        ("codebooks_q", PVQ_HEADER_SIZE + int(decoder_len), int(codebook_len)),
        (
            "selectors_rc",
            PVQ_HEADER_SIZE + int(decoder_len) + int(codebook_len),
            int(indices_len),
        ),
        (
            "receiver_state",
            PVQ_HEADER_SIZE + int(decoder_len) + int(codebook_len) + int(indices_len),
            int(meta_len),
        ),
    ]
    rows: list[PvqSectionLayoutRow] = []
    for name, offset, length in ranges:
        payload = blob[offset : offset + length]
        rows.append(
            PvqSectionLayoutRow(
                name=name,
                offset=offset,
                length=length,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    expected = PVQ_HEADER_SIZE + sum(row.length for row in rows)
    if expected != len(blob):
        raise ValueError(f"PVQ layout size {expected} != blob size {len(blob)}")
    return rows


def neutralize_pvq_section(blob: bytes, section_name: str) -> bytes:
    """Return valid PVQ bytes with one logical section neutralized.

    Neutralization is semantic and parse-preserving:
    - decoder weights are zeroed in the decoder state dict;
    - codebook atoms are zeroed while preserving shape and quantization metadata;
    - selector/index bytes are collapsed to index 0 for every pair;
    - receiver_state is refused because it owns decode shape and quant constants.
    """

    section = str(section_name).strip().lower()
    if section not in PVQ_SUPPORTED_SECTION_NAMES:
        raise ValueError(
            f"unsupported PVQ section {section_name!r}; "
            f"expected one of {PVQ_SUPPORTED_SECTION_NAMES}"
        )
    if section == "receiver_state":
        raise ValueError("receiver_state neutralization would invalidate decoder shape")
    arc = parse_archive(blob)
    decoder_state = arc.decoder_state_dict
    codebook = arc.codebook
    indices = arc.indices
    if section == "decoder_qw":
        decoder_state = {
            name: torch.zeros_like(tensor) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in decoder_state.items()
        }
    elif section == "codebooks_q":
        codebook = torch.zeros_like(codebook)
    elif section == "selectors_rc":
        indices = torch.zeros_like(indices)
    return pack_archive(
        decoder_state,
        codebook,
        indices,
        dict(arc.meta),
        schema_version=arc.schema_version,
    )


def pvq_layout_report(*, blob: bytes) -> dict[str, Any]:
    """Return a JSON-ready PVQ layout report."""

    arc = parse_archive(blob)
    return {
        "schema": "pact_nerv_vq_section_layout.v1",
        "sections": [row.as_dict() for row in pvq_section_layout(blob)],
        "num_pairs": int(arc.indices.shape[0]),
        "latent_dim": int(arc.codebook.shape[1]),
        "codebook_size": int(arc.codebook.shape[0]),
        "meta": dict(arc.meta),
        **FALSE_AUTHORITY,
    }


__all__ = [
    "PVQ_SECTION_VALUE_PROFILE_SCHEMA",
    "PVQ_SECTION_VALUE_SOURCE_SCHEMA",
    "PVQ_SUPPORTED_SECTION_NAMES",
    "PvqSectionLayoutRow",
    "neutralize_pvq_section",
    "pvq_layout_report",
    "pvq_section_layout",
]
