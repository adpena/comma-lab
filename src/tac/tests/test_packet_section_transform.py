from __future__ import annotations

from pathlib import Path

import brotli

from tac.hnerv_lowlevel_packer import parse_ff_packed_brotli_hnerv, write_stored_single_member_zip
from tac.packet_section_transform import (
    BrotliRecodeSectionTransform,
    build_hnerv_packet_ir,
    compile_hnerv_pr106_section_transform_candidate,
)


def test_build_hnerv_packet_ir_records_parser_sections(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    payload = _packed_payload(
        brotli.compress(b"decoder" * 200, quality=1),
        brotli.compress(b"latents" * 100, quality=1),
    )
    write_stored_single_member_zip(archive, member_name="0.bin", payload=payload)

    packet_ir = build_hnerv_packet_ir(archive, label="fixture")

    assert packet_ir.archive_bytes == archive.stat().st_size
    assert packet_ir.member_name == "0.bin"
    assert packet_ir.parser_name == "pr106_ff_packed_hnerv"
    assert [section.name for section in packet_ir.sections] == [
        "packed_header_ff_len24",
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ]
    assert packet_ir.parser_section_gate["ready"] is True


def test_compile_pr106_brotli_recode_candidate_updates_header_and_proves_raw_equivalence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    candidate = tmp_path / "candidate.zip"
    payload = _packed_payload(
        brotli.compress(b"decoder-record-" * 3000, quality=1),
        brotli.compress(b"latent-record-" * 1000, quality=1),
    )
    write_stored_single_member_zip(source, member_name="0.bin", payload=payload)

    result = compile_hnerv_pr106_section_transform_candidate(
        source_archive=source,
        label="PR106-fixture",
        transform=BrotliRecodeSectionTransform(
            target_section="decoder_packed_brotli",
            qualities=(11,),
            lgwins=(None,),
            jobs=1,
        ),
        output_archive=candidate,
    )

    assert result["score_claim"] is False
    assert result["dispatch_attempted"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["ready_for_archive_preflight"] is True
    assert result["blockers"] == []
    assert result["candidate_archive_path"] == str(candidate)
    assert result["archive_byte_delta"] < 0
    assert candidate.exists()

    changed = {row["section_name"]: row for row in result["changed_sections"]}
    assert set(changed) == {
        "packed_header_ff_len24",
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    }
    assert changed["decoder_packed_brotli"]["byte_delta"] < 0
    # The header is score-affecting bytes too: PR106 stores decoder length in len24.
    assert changed["packed_header_ff_len24"]["byte_delta"] == 0
    assert changed["packed_header_ff_len24"]["content_changed"] is True
    # The latent payload bytes are unchanged, but its offset shifts after decoder recoding.
    assert changed["latents_and_sidecar_brotli"]["content_changed"] is False
    assert changed["latents_and_sidecar_brotli"]["offset_changed"] is True

    output = result["transform"]["outputs"][0]
    assert output["section_name"] == "decoder_packed_brotli"
    assert output["metadata"]["raw_equivalence"]["raw_equal"] is True
    assert output["metadata"]["choice"]["byte_delta"] < 0

    source_payload = parse_ff_packed_brotli_hnerv(payload)
    candidate_ir = result["candidate_packet_ir"]
    assert candidate_ir["parser_section_gate"]["ready"] is True
    assert candidate_ir["member_name"] == "0.bin"
    assert candidate_ir["sections"][1]["length"] < len(source_payload.decoder_packed_brotli)


def test_compile_pr106_transform_blocks_nonmatching_section(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    candidate = tmp_path / "candidate.zip"
    payload = _packed_payload(
        brotli.compress(b"decoder" * 200, quality=1),
        brotli.compress(b"latents" * 100, quality=1),
    )
    write_stored_single_member_zip(source, member_name="0.bin", payload=payload)

    result = compile_hnerv_pr106_section_transform_candidate(
        source_archive=source,
        label="PR106-fixture",
        transform=BrotliRecodeSectionTransform(target_section="not_a_real_section"),
        output_archive=candidate,
    )

    assert result["ready_for_archive_preflight"] is False
    assert result["candidate_packet_ir"] is None
    assert not candidate.exists()
    assert "transform_matched_no_sections" in result["blockers"]
    assert "transform_produced_no_changed_sections" in result["blockers"]


def _packed_payload(decoder_brotli: bytes, latents_brotli: bytes) -> bytes:
    return bytes([0xFF]) + len(decoder_brotli).to_bytes(3, "little") + decoder_brotli + latents_brotli
