# SPDX-License-Identifier: MIT
"""Tests for the HPRC V0 packet grammar."""

from __future__ import annotations

import json

import pytest

from tac.substrates.hprc.archive import (
    HPRC_MAGIC,
    HprcArchiveError,
    HprcPacketConfig,
    HprcSectionKind,
    pack_hprc_packet,
    parse_hprc_packet,
    write_hprc_manifest,
)


def test_hprc_packet_round_trip_is_byte_deterministic() -> None:
    config = HprcPacketConfig(
        frames=1200,
        pairs=600,
        height=384,
        width=512,
        decoder_family_id=95,
        color_transform_id=1,
        gop_size=2,
    )
    sections = {
        HprcSectionKind.LATENTS_RC: b"latents",
        "decoder_qw": b"decoder",
        6: b"rdo",
        HprcSectionKind.RESIDUAL_RC: b"",
    }

    packet_1 = pack_hprc_packet(sections, config=config)
    parsed = parse_hprc_packet(packet_1)
    packet_2 = pack_hprc_packet(parsed.section_map(), config=parsed.config)

    assert packet_1 == packet_2
    assert packet_1.startswith(HPRC_MAGIC)
    assert parsed.config == config
    assert [section.kind for section in parsed.sections] == [
        HprcSectionKind.DECODER_QW,
        HprcSectionKind.LATENTS_RC,
        HprcSectionKind.RESIDUAL_RC,
        HprcSectionKind.RDO_PLAN,
    ]
    assert parsed.section_map()[HprcSectionKind.RESIDUAL_RC] == b""


def test_hprc_manifest_is_json_stable_and_fail_closed() -> None:
    packet = pack_hprc_packet(
        {
            HprcSectionKind.DECODER_QW: b"abc",
            HprcSectionKind.LATENTS_RC: b"defg",
            HprcSectionKind.MANIFEST_JSON: b'{"source":"unit"}',
        }
    )

    manifest = write_hprc_manifest(packet)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    decoded = json.loads(encoded)

    assert decoded["score_claim"] is False
    assert decoded["promotion_eligible"] is False
    assert decoded["byte_accounting"]["runtime_bytes_included"] is False
    assert decoded["byte_accounting"]["contest_rate_bytes_authority"] is False
    assert "archive.zip" in decoded["byte_accounting"]["authority_note"]
    assert decoded["receiver_proof"]["status"] == "raw_flip_only"
    assert "valid semantic section mutation" in decoded["receiver_proof"]["required_for_promotion"]
    assert decoded["config"]["pairs"] == 600
    assert [row["name"] for row in decoded["sections"]] == [
        "decoder_qw",
        "latents_rc",
        "manifest_json",
    ]
    assert all(len(row["sha256"]) == 64 for row in decoded["sections"])


def test_hprc_packet_rejects_payload_mutation() -> None:
    packet = bytearray(
        pack_hprc_packet(
            {
                HprcSectionKind.DECODER_QW: b"decoder",
                HprcSectionKind.LATENTS_RC: b"latents",
            }
        )
    )
    packet[-1] ^= 0xFF

    with pytest.raises(HprcArchiveError, match=r"crc mismatch|sha256 mismatch"):
        parse_hprc_packet(packet)


def test_hprc_packet_rejects_bad_magic_and_empty_sections() -> None:
    with pytest.raises(HprcArchiveError, match="at least one section"):
        pack_hprc_packet({})

    packet = bytearray(pack_hprc_packet({HprcSectionKind.DECODER_QW: b"decoder"}))
    packet[:4] = b"NOPE"
    with pytest.raises(HprcArchiveError, match="magic mismatch"):
        parse_hprc_packet(packet)


def test_hprc_packet_rejects_out_of_range_config() -> None:
    with pytest.raises(HprcArchiveError, match="outside u16 range"):
        HprcPacketConfig(frames=70000)
