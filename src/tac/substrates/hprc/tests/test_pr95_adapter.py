# SPDX-License-Identifier: MIT
"""Tests for the PR95 -> HPRC control adapter."""

from __future__ import annotations

import json
import struct
import zipfile

from tac.substrates.hprc.archive import HprcSectionKind, parse_hprc_packet
from tac.substrates.hprc.pr95_adapter import (
    PR95_HNERV_DECODER_FAMILY_ID,
    build_pr95_hprc_control_packet,
    parse_pr95_hnerv_payload,
)


def _pr95_payload() -> bytes:
    parts = []
    for payload in (b'{"n_pairs":600}', b"decoder", b"latents"):
        parts.append(struct.pack("<I", len(payload)))
        parts.append(payload)
    return b"".join(parts)


def test_parse_pr95_hnerv_payload_sections() -> None:
    parsed = parse_pr95_hnerv_payload(_pr95_payload())

    assert parsed["payload_bytes"] == len(_pr95_payload())
    assert parsed["decoder_blob"] == b"decoder"
    assert parsed["latents_blob"] == b"latents"
    assert len(parsed["decoder_sha256"]) == 64


def test_pr95_control_packet_wraps_compressed_sections(tmp_path) -> None:
    archive_zip = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", _pr95_payload())

    control = build_pr95_hprc_control_packet(archive_zip)
    packet = parse_hprc_packet(control.hprc_bin)
    section_map = packet.section_map()

    assert packet.config.decoder_family_id == PR95_HNERV_DECODER_FAMILY_ID
    assert section_map[HprcSectionKind.DECODER_QW] == b"decoder"
    assert section_map[HprcSectionKind.LATENTS_RC] == b"latents"
    embedded = json.loads(section_map[HprcSectionKind.MANIFEST_JSON])
    assert embedded["score_claim"] is False
    assert control.manifest["score_claim"] is False
    assert control.manifest["promotion_eligible"] is False
    assert control.manifest["byte_delta_vs_source_member"] > 0
