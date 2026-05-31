# SPDX-License-Identifier: MIT
"""PR95/HNeRV control adapter for HPRC.

The adapter does not claim HPRC improves PR95. It imports PR95's byte anatomy
as the control receiver so every HPRC candidate can be judged against a proven
frontier-scale compact decoder+latent packet instead of against the much easier
Z8 explicit-wavelet baseline.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tac.substrates.hprc.archive import (
    HprcPacketConfig,
    HprcSectionKind,
    pack_hprc_packet,
    parse_hprc_packet,
)

PR95_HNERV_DECODER_FAMILY_ID = 95
PR95_RGB_COLOR_TRANSFORM_ID = 0


class Pr95AdapterError(ValueError):
    """Raised when a PR95 archive cannot be adapted safely."""


@dataclass(frozen=True)
class Pr95HprcControlPacket:
    """Adapted PR95 control packet plus byte-accounting metadata."""

    hprc_bin: bytes
    manifest: dict[str, object]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.hprc_bin).hexdigest()


def _read_single_member_zip(path: Path) -> tuple[str, bytes, int, str]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if names != ["0.bin"]:
            raise Pr95AdapterError(f"expected single 0.bin member, got {names!r}")
        payload = zf.read("0.bin")
    return "0.bin", payload, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def _read_u32_prefixed_section(blob: bytes, pos: int, name: str) -> tuple[bytes, int]:
    if pos + 4 > len(blob):
        raise Pr95AdapterError(f"truncated before {name} length")
    (length,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    end = pos + int(length)
    if end > len(blob):
        raise Pr95AdapterError(f"{name} section extends past PR95 payload")
    return blob[pos:end], end


def parse_pr95_hnerv_payload(blob: bytes) -> dict[str, object]:
    """Parse the public PR95/HNeRV ``0.bin`` section envelope.

    Grammar observed in the public PR95 intake:
    ``[meta_len:u32][meta_brotli][decoder_len:u32][decoder_brotli][latents_len:u32][latents_brotli]``.
    The compressed sections are preserved byte-for-byte; no torch import or
    semantic tensor decode is needed for HPRC byte-control accounting.
    """

    pos = 0
    meta_blob, pos = _read_u32_prefixed_section(blob, pos, "meta")
    decoder_blob, pos = _read_u32_prefixed_section(blob, pos, "decoder")
    latents_blob, pos = _read_u32_prefixed_section(blob, pos, "latents")
    if pos != len(blob):
        raise Pr95AdapterError(f"PR95 payload has trailing bytes: pos={pos}, len={len(blob)}")
    return {
        "meta_blob": meta_blob,
        "decoder_blob": decoder_blob,
        "latents_blob": latents_blob,
        "meta_sha256": hashlib.sha256(meta_blob).hexdigest(),
        "decoder_sha256": hashlib.sha256(decoder_blob).hexdigest(),
        "latents_sha256": hashlib.sha256(latents_blob).hexdigest(),
        "payload_bytes": len(blob),
    }


def build_pr95_hprc_control_packet(archive_zip: Path) -> Pr95HprcControlPacket:
    """Wrap PR95's compressed decoder+latent control into ``hprc.bin``."""

    member_name, payload, zip_bytes, zip_sha = _read_single_member_zip(archive_zip)
    sections = parse_pr95_hnerv_payload(payload)
    manifest_payload = {
        "schema": "hprc_pr95_control_manifest.v1",
        "source_archive_path": str(archive_zip),
        "source_archive_bytes": zip_bytes,
        "source_archive_sha256": zip_sha,
        "source_member_name": member_name,
        "source_member_bytes": len(payload),
        "source_member_sha256": hashlib.sha256(payload).hexdigest(),
        "meta_sha256": sections["meta_sha256"],
        "decoder_sha256": sections["decoder_sha256"],
        "latents_sha256": sections["latents_sha256"],
        "score_claim": False,
        "promotion_eligible": False,
        "role": "PR95/HNeRV byte-scale control arm for HPRC",
    }
    hprc_bin = pack_hprc_packet(
        {
            HprcSectionKind.DECODER_QW: sections["decoder_blob"],
            HprcSectionKind.LATENTS_RC: sections["latents_blob"],
            HprcSectionKind.MANIFEST_JSON: json.dumps(
                manifest_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        },
        config=HprcPacketConfig(
            frames=1200,
            pairs=600,
            height=384,
            width=512,
            decoder_family_id=PR95_HNERV_DECODER_FAMILY_ID,
            color_transform_id=PR95_RGB_COLOR_TRANSFORM_ID,
            gop_size=2,
        ),
    )
    parsed = parse_hprc_packet(hprc_bin)
    manifest = parsed.manifest()
    manifest["pr95_control"] = manifest_payload
    manifest["byte_delta_vs_source_member"] = int(len(hprc_bin) - len(payload))
    manifest["byte_delta_vs_source_archive_zip"] = int(len(hprc_bin) - zip_bytes)
    return Pr95HprcControlPacket(hprc_bin=hprc_bin, manifest=manifest)
