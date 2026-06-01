# SPDX-License-Identifier: MIT
"""Section-neutralization helpers for PSV4 PACT-NeRV selector archives."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

import torch

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.pact_nerv_selector_v4.archive import (
    PSV4_HEADER_FMT,
    PSV4_HEADER_SIZE,
    pack_archive,
    parse_archive,
)

PSV4_SECTION_VALUE_PROFILE_SCHEMA = "hprc_mlx_component_neutralization_profile.v1"
PSV4_SECTION_VALUE_SOURCE_SCHEMA = "pact_nerv_selector_v4_section_value_profile.v1"
PSV4_SUPPORTED_SECTION_NAMES = (
    "decoder_qw",
    "latents_rc",
    "selectors_rc",
    "receiver_state",
)


@dataclass(frozen=True)
class Psv4SectionLayoutRow:
    """Byte range for one PSV4 logical section inside ``0.bin``."""

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


def psv4_section_layout(blob: bytes) -> list[Psv4SectionLayoutRow]:
    """Return byte ranges for PSV4 payload sections."""

    if len(blob) < PSV4_HEADER_SIZE:
        raise ValueError(f"PSV4 blob too short: {len(blob)}")
    (
        _magic,
        _version,
        _latent_dim,
        _num_pairs,
        _palette_size,
        dec_len,
        lat_len,
        sel_len,
        meta_len,
    ) = struct.unpack(PSV4_HEADER_FMT, blob[:PSV4_HEADER_SIZE])
    ranges = [
        ("decoder_qw", PSV4_HEADER_SIZE, int(dec_len)),
        ("latents_rc", PSV4_HEADER_SIZE + int(dec_len), int(lat_len)),
        (
            "selectors_rc",
            PSV4_HEADER_SIZE + int(dec_len) + int(lat_len),
            int(sel_len),
        ),
        (
            "receiver_state",
            PSV4_HEADER_SIZE + int(dec_len) + int(lat_len) + int(sel_len),
            int(meta_len),
        ),
    ]
    rows = []
    for name, offset, length in ranges:
        payload = blob[offset : offset + length]
        rows.append(
            Psv4SectionLayoutRow(
                name=name,
                offset=offset,
                length=length,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    expected = PSV4_HEADER_SIZE + sum(row.length for row in rows)
    if expected != len(blob):
        raise ValueError(f"PSV4 layout size {expected} != blob size {len(blob)}")
    return rows


def neutralize_psv4_section(blob: bytes, section_name: str) -> bytes:
    """Return valid PSV4 bytes with one logical section neutralized.

    Neutralization is semantic and parse-preserving: decoder weights and
    latents are zeroed through the archive grammar, selectors are removed as a
    charged selector stream, and receiver-state neutralization is refused
    because meta fields define the decoder/runtime shape.
    """

    section = str(section_name).strip().lower()
    if section not in PSV4_SUPPORTED_SECTION_NAMES:
        raise ValueError(
            f"unsupported PSV4 section {section_name!r}; "
            f"expected one of {PSV4_SUPPORTED_SECTION_NAMES}"
        )
    if section == "receiver_state":
        raise ValueError("receiver_state neutralization would invalidate decoder shape")
    arc = parse_archive(blob)
    decoder_state = arc.decoder_state_dict
    latents = arc.latents
    selector_bytes = arc.selector_bytes
    if section == "decoder_qw":
        decoder_state = {
            name: torch.zeros_like(tensor) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in decoder_state.items()
        }
    elif section == "latents_rc":
        latents = torch.zeros_like(latents)
    elif section == "selectors_rc":
        selector_bytes = b""
    return pack_archive(
        decoder_state,
        latents,
        selector_bytes,
        dict(arc.meta),
        palette_size=arc.palette_size,
        schema_version=arc.schema_version,
        decoder_codec=_decoder_codec_from_meta(arc.meta),
    )


def psv4_layout_report(*, blob: bytes) -> dict[str, Any]:
    """Return a JSON-ready PSV4 layout report."""

    arc = parse_archive(blob)
    return {
        "schema": "pact_nerv_selector_v4_section_layout.v1",
        "sections": [row.as_dict() for row in psv4_section_layout(blob)],
        "num_pairs": int(arc.latents.shape[0]),
        "latent_dim": int(arc.latents.shape[1]),
        "palette_size": int(arc.palette_size),
        "meta": dict(arc.meta),
        **FALSE_AUTHORITY,
    }


def _decoder_codec_from_meta(meta: dict[str, Any]) -> str:
    codec = meta.get("_decoder_state_codec")
    if isinstance(codec, dict):
        return str(codec.get("codec", "fp16_brotli_legacy"))
    return "fp16_brotli_legacy"


__all__ = [
    "PSV4_SECTION_VALUE_PROFILE_SCHEMA",
    "PSV4_SECTION_VALUE_SOURCE_SCHEMA",
    "PSV4_SUPPORTED_SECTION_NAMES",
    "Psv4SectionLayoutRow",
    "neutralize_psv4_section",
    "psv4_layout_report",
    "psv4_section_layout",
]
