# SPDX-License-Identifier: MIT
"""Section-neutralization helpers for PSV3 PACT-NeRV selector archives."""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.pact_nerv_selector_v3.archive import (
    DECODER_QUANT_FP16_BROTLI_Q9,
    PSV3_HEADER_FMT,
    PSV3_HEADER_SIZE,
    pack_archive,
    parse_archive,
)

PSV3_SECTION_VALUE_PROFILE_SCHEMA = "hprc_mlx_component_neutralization_profile.v1"
PSV3_SECTION_VALUE_SOURCE_SCHEMA = "pact_nerv_selector_v3_section_value_profile.v1"
PSV3_SUPPORTED_SECTION_NAMES = (
    "decoder_qw",
    "latents_rc",
    "selectors_rc",
    "receiver_state",
)
_FIXED_ZIP_DATE = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class Psv3SectionLayoutRow:
    """Byte range for one PSV3 logical section inside ``0.bin``."""

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


def psv3_section_layout(blob: bytes) -> list[Psv3SectionLayoutRow]:
    """Return byte ranges for PSV3 payload sections."""

    if len(blob) < PSV3_HEADER_SIZE:
        raise ValueError(f"PSV3 blob too short: {len(blob)}")
    _magic, _version, _latent_dim, _num_pairs, _palette_size, dec_len, lat_len, sel_len, meta_len = (
        struct.unpack(PSV3_HEADER_FMT, blob[:PSV3_HEADER_SIZE])
    )
    ranges = [
        ("decoder_qw", PSV3_HEADER_SIZE, int(dec_len)),
        ("latents_rc", PSV3_HEADER_SIZE + int(dec_len), int(lat_len)),
        (
            "selectors_rc",
            PSV3_HEADER_SIZE + int(dec_len) + int(lat_len),
            int(sel_len),
        ),
        (
            "receiver_state",
            PSV3_HEADER_SIZE + int(dec_len) + int(lat_len) + int(sel_len),
            int(meta_len),
        ),
    ]
    rows = []
    for name, offset, length in ranges:
        payload = blob[offset : offset + length]
        rows.append(
            Psv3SectionLayoutRow(
                name=name,
                offset=offset,
                length=length,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    expected = PSV3_HEADER_SIZE + sum(row.length for row in rows)
    if expected != len(blob):
        raise ValueError(f"PSV3 layout size {expected} != blob size {len(blob)}")
    return rows


def neutralize_psv3_section(blob: bytes, section_name: str) -> bytes:
    """Return valid PSV3 bytes with one logical section neutralized.

    Neutralization is intentionally semantic rather than byte flipping:
    decoder weights and latents are zeroed through the archive grammar,
    selectors are removed because this V3 runtime does not consume them in the
    pixel path, and receiver-state neutralization is refused because the config
    defines the decoder shape.
    """

    section = str(section_name).strip().lower()
    if section not in PSV3_SUPPORTED_SECTION_NAMES:
        raise ValueError(
            f"unsupported PSV3 section {section_name!r}; "
            f"expected one of {PSV3_SUPPORTED_SECTION_NAMES}"
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
    decoder_quantization = str(
        arc.meta.get("decoder_quantization") or DECODER_QUANT_FP16_BROTLI_Q9
    )
    return pack_archive(
        decoder_state,
        latents,
        selector_bytes,
        dict(arc.meta),
        palette_size=arc.palette_size,
        schema_version=arc.schema_version,
        decoder_quantization=decoder_quantization,
    )


def write_zip_replacing_member(
    *,
    source_archive: str | Path,
    output_archive: str | Path,
    member_name: str,
    replacement_bytes: bytes,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Write a deterministic ZIP copy with one member replaced."""

    source = Path(source_archive).expanduser().resolve(strict=False)
    output = Path(output_archive).expanduser().resolve(strict=False)
    if output.exists() and not allow_overwrite:
        raise FileExistsError(f"output archive exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized_member = _normalize_member_name(member_name)
    copied: list[dict[str, Any]] = []
    replaced: dict[str, Any] | None = None
    with zipfile.ZipFile(source, "r") as zin:
        names = {info.filename for info in zin.infolist() if not info.is_dir()}
        if normalized_member not in names:
            raise ValueError(f"ZIP member missing: {normalized_member}")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
            for info in sorted(zin.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                old = zin.read(info.filename)
                if info.filename == normalized_member:
                    _writestr(zout, info.filename, replacement_bytes)
                    replaced = {
                        "member": info.filename,
                        "old_bytes": len(old),
                        "old_sha256": hashlib.sha256(old).hexdigest(),
                        "bytes": len(replacement_bytes),
                        "sha256": hashlib.sha256(replacement_bytes).hexdigest(),
                    }
                else:
                    _writestr(zout, info.filename, old)
                    copied.append(
                        {
                            "member": info.filename,
                            "bytes": len(old),
                            "sha256": hashlib.sha256(old).hexdigest(),
                        }
                    )
    if replaced is None:
        raise ValueError(f"ZIP member was not replaced: {normalized_member}")
    return {
        "schema": "pact_nerv_selector_v3_zip_member_replacement.v1",
        "source_archive": _file_row(source),
        "output_archive": _file_row(output),
        "replaced_member": replaced,
        "copied_members": copied,
        **FALSE_AUTHORITY,
    }


def psv3_layout_report(*, blob: bytes) -> dict[str, Any]:
    """Return a JSON-ready PSV3 layout report."""

    arc = parse_archive(blob)
    return {
        "schema": "pact_nerv_selector_v3_section_layout.v1",
        "sections": [row.as_dict() for row in psv3_section_layout(blob)],
        "num_pairs": int(arc.latents.shape[0]),
        "latent_dim": int(arc.latents.shape[1]),
        "palette_size": int(arc.palette_size),
        "meta": json.loads(json.dumps(arc.meta, sort_keys=True, default=str)),
        **FALSE_AUTHORITY,
    }


def _file_row(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256_file(path) if path.is_file() else None,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _writestr(zout: zipfile.ZipFile, member: str, data: bytes) -> None:
    info = zipfile.ZipInfo(_normalize_member_name(member), date_time=_FIXED_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zout.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _normalize_member_name(value: str) -> str:
    member = str(value).strip().replace("\\", "/")
    if not member or member.startswith("/") or "/../" in f"/{member}/":
        raise ValueError(f"unsafe ZIP member name: {value!r}")
    return member


__all__ = [
    "PSV3_SECTION_VALUE_PROFILE_SCHEMA",
    "PSV3_SECTION_VALUE_SOURCE_SCHEMA",
    "PSV3_SUPPORTED_SECTION_NAMES",
    "neutralize_psv3_section",
    "psv3_layout_report",
    "psv3_section_layout",
    "write_zip_replacing_member",
]
