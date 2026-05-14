# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tac.frontier_archive_layout import inspect_frontier_archive_layout
from tac.hnerv_generated_schema_packet import (
    PACKET_PREAMBLE,
    HNeRVGeneratedSchemaPacketError,
    build_hnerv_generated_schema_packet,
    inspect_hnerv_generated_schema_packet,
    parse_hnerv_generated_schema_packet,
)


def test_hngp_packet_build_parse_is_deterministic_and_non_score() -> None:
    hngs_decoder = b"HNGS" + b"\x00generated-schema-decoder"
    latent_blob = b"latent-payload"
    sidecar_blob = b"sidecar-payload"

    first = build_hnerv_generated_schema_packet(
        hngs_decoder=hngs_decoder,
        latent_blob=latent_blob,
        sidecar_blob=sidecar_blob,
        metadata={"candidate_id": "unit-hngp-v1"},
    )
    second = build_hnerv_generated_schema_packet(
        hngs_decoder=hngs_decoder,
        latent_blob=latent_blob,
        sidecar_blob=sidecar_blob,
        metadata={"candidate_id": "unit-hngp-v1"},
    )
    parsed = parse_hnerv_generated_schema_packet(first.packet)

    assert first.packet == second.packet
    assert parsed.hngs_decoder == hngs_decoder
    assert parsed.latent_blob == latent_blob
    assert parsed.sidecar_blob == sidecar_blob
    assert parsed.manifest == inspect_hnerv_generated_schema_packet(first.packet)
    assert parsed.manifest["score_claim"] is False
    assert parsed.manifest["promotion_eligible"] is False
    assert parsed.manifest["ready_for_exact_eval_dispatch"] is False
    assert parsed.manifest["packet_sha256"] == hashlib.sha256(first.packet).hexdigest()
    assert parsed.header["metadata"] == {"candidate_id": "unit-hngp-v1"}

    sections = {section["name"]: section for section in parsed.manifest["sections"]}
    assert list(sections) == ["header", "hngs_decoder", "latent_blob", "sidecar_blob"]
    assert sections["header"]["offset"] == 0
    assert sections["hngs_decoder"]["offset"] == sections["header"]["len"]
    assert sections["latent_blob"]["offset"] == (
        sections["hngs_decoder"]["offset"] + len(hngs_decoder)
    )
    assert sections["sidecar_blob"]["offset"] == (
        sections["latent_blob"]["offset"] + len(latent_blob)
    )
    assert sections["hngs_decoder"]["sha256"] == hashlib.sha256(hngs_decoder).hexdigest()
    assert sections["latent_blob"]["sha256"] == hashlib.sha256(latent_blob).hexdigest()
    assert sections["sidecar_blob"]["sha256"] == hashlib.sha256(sidecar_blob).hexdigest()


def test_hngp_packet_rejects_truncated_sections() -> None:
    packet = build_hnerv_generated_schema_packet(
        hngs_decoder=b"HNGS-decoder",
        latent_blob=b"latent",
        sidecar_blob=b"sidecar",
    ).packet

    with pytest.raises(HNeRVGeneratedSchemaPacketError, match="truncated HNGP section"):
        parse_hnerv_generated_schema_packet(packet[:-1])


def test_hngp_packet_rejects_section_sha_mismatch() -> None:
    built = build_hnerv_generated_schema_packet(
        hngs_decoder=b"HNGS-decoder",
        latent_blob=b"latent",
        sidecar_blob=b"sidecar",
    )
    latent = next(section for section in built.sections if section.name == "latent_blob")
    tampered = bytearray(built.packet)
    tampered[latent.offset] ^= 0x01

    with pytest.raises(HNeRVGeneratedSchemaPacketError, match="sha256 mismatch"):
        parse_hnerv_generated_schema_packet(bytes(tampered))


def test_hngp_packet_rejects_duplicate_sections() -> None:
    built = build_hnerv_generated_schema_packet(
        hngs_decoder=b"HNGS-decoder",
        latent_blob=b"latent",
        sidecar_blob=b"sidecar",
    )

    duplicate = _mutate_header(
        built.packet,
        lambda header: header["sections"].__setitem__(
            2,
            {
                "name": "latent_blob",
                "role": "generated_schema_latent_payload",
                "len": len(b"sidecar"),
                "sha256": hashlib.sha256(b"sidecar").hexdigest(),
            },
        ),
    )

    with pytest.raises(HNeRVGeneratedSchemaPacketError, match="duplicate HNGP section"):
        parse_hnerv_generated_schema_packet(duplicate)


def test_hngp_packet_rejects_malformed_inputs_fail_closed() -> None:
    with pytest.raises(HNeRVGeneratedSchemaPacketError, match="HNGS magic"):
        build_hnerv_generated_schema_packet(
            hngs_decoder=b"BAD-decoder",
            latent_blob=b"latent",
            sidecar_blob=b"sidecar",
        )


def test_frontier_layout_recognizes_hngp_packet(tmp_path: Path) -> None:
    packet = build_hnerv_generated_schema_packet(
        hngs_decoder=b"HNGS-decoder",
        latent_blob=b"latent",
        sidecar_blob=b"sidecar",
    ).packet
    archive = tmp_path / "hngp.zip"
    info = zipfile.ZipInfo("payload.hngp")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(info, packet)

    manifest = inspect_frontier_archive_layout(archive)

    logical = manifest["logical_layout"]
    assert logical["grammar"] == "hngp_v1"
    assert logical["parser_proof_strength"] == "canonical_hngp_parse"
    assert [section["name"] for section in logical["sections"]] == [
        "header",
        "hngs_decoder",
        "latent_blob",
        "sidecar_blob",
    ]

    valid = build_hnerv_generated_schema_packet(
        hngs_decoder=b"HNGS-decoder",
        latent_blob=b"latent",
        sidecar_blob=b"sidecar",
    ).packet

    bad_score_header = _mutate_header(
        valid,
        lambda header: header.__setitem__("score_claim", True),
    )
    with pytest.raises(HNeRVGeneratedSchemaPacketError, match="score_claim=false"):
        parse_hnerv_generated_schema_packet(bad_score_header)

    with pytest.raises(HNeRVGeneratedSchemaPacketError, match="metadata contains reserved"):
        build_hnerv_generated_schema_packet(
            hngs_decoder=b"HNGS-decoder",
            latent_blob=b"latent",
            sidecar_blob=b"sidecar",
            metadata={"score_claim": False},
        )


def _mutate_header(packet: bytes, mutator) -> bytes:
    magic, version, header_len = PACKET_PREAMBLE.unpack_from(packet, 0)
    header_start = PACKET_PREAMBLE.size
    header_end = header_start + int(header_len)
    header = json.loads(packet[header_start:header_end].decode("utf-8"))
    mutator(header)
    header_bytes = json.dumps(
        header,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        PACKET_PREAMBLE.pack(magic, version, len(header_bytes))
        + header_bytes
        + packet[header_end:]
    )
