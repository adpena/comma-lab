# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli

from tac.hnerv_lowlevel_packer import parse_ff_packed_brotli_hnerv, write_stored_single_member_zip
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106SidecarPacket,
    PR106_SIDECAR_FORMAT_BROTLI,
    emit_pr106_sidecar_packet,
    parse_pr106_sidecar_packet,
)
from tac.packet_section_transform import (
    BrotliRecodeSectionTransform,
    build_hnerv_packet_ir,
    certify_hnerv_grammar_preserving_candidate_pair,
    compile_hnerv_pr106_section_transform_candidate,
    scan_hnerv_brotli_recode_opportunities,
)

REPO = Path(__file__).resolve().parents[3]


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


def test_build_hnerv_packet_ir_accepts_pr106_sidecar_wrapper(tmp_path: Path) -> None:
    archive = tmp_path / "source_sidecar.zip"
    inner = _packed_payload(
        brotli.compress(b"decoder" * 200, quality=1),
        brotli.compress(b"latents" * 100, quality=1),
    )
    wrapper = _sidecar_payload(inner)
    write_stored_single_member_zip(archive, member_name="0.bin", payload=wrapper)

    packet_ir = build_hnerv_packet_ir(archive, label="sidecar-fixture")

    assert packet_ir.archive_bytes == archive.stat().st_size
    assert packet_ir.member_bytes == len(wrapper)
    assert packet_ir.parser_name == "pr106_ff_packed_hnerv"
    assert packet_ir.parser_input["kind"] == "pr106_sidecar_inner_payload"
    assert packet_ir.pr106_sidecar_wrapper is not None
    assert packet_ir.pr106_sidecar_wrapper["format_id"] == "0x01"
    assert [section.name for section in packet_ir.sections] == [
        "packed_header_ff_len24",
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ]
    parsed_inner = parse_ff_packed_brotli_hnerv(
        parse_pr106_sidecar_packet(wrapper).pr106_bytes
    )
    assert packet_ir.sections[1].length == len(parsed_inner.decoder_packed_brotli)
    assert packet_ir.sections[2].offset == 4 + len(parsed_inner.decoder_packed_brotli)
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


def test_compile_pr106_brotli_recode_candidate_preserves_sidecar_wrapper(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_sidecar.zip"
    candidate = tmp_path / "candidate_sidecar.zip"
    inner = _packed_payload(
        brotli.compress(b"decoder-record-" * 3000, quality=1),
        brotli.compress(b"latent-record-" * 1000, quality=1),
    )
    source_wrapper = _sidecar_payload(inner)
    write_stored_single_member_zip(source, member_name="0.bin", payload=source_wrapper)

    result = compile_hnerv_pr106_section_transform_candidate(
        source_archive=source,
        label="PR106-sidecar-fixture",
        transform=BrotliRecodeSectionTransform(
            target_section="decoder_packed_brotli",
            qualities=(11,),
            lgwins=(None,),
            jobs=1,
        ),
        output_archive=candidate,
    )

    assert result["ready_for_archive_preflight"] is True
    assert result["blockers"] == []
    assert result["archive_byte_delta"] < 0
    assert result["source_packet_ir"]["member_bytes"] == len(source_wrapper)
    assert result["source_packet_ir"]["parser_input"]["kind"] == "pr106_sidecar_inner_payload"
    assert result["source_packet_ir"]["pr106_sidecar_wrapper"]["sidecar_payload_bytes"] == len(b"opaque-sidecar")
    assert result["candidate_packet_ir"]["member_name"] == "0.bin"
    assert result["candidate_packet_ir"]["parser_input"]["kind"] == "pr106_sidecar_inner_payload"
    candidate_wrapper = parse_pr106_sidecar_packet(
        _single_payload(candidate)
    )
    source_packet = parse_pr106_sidecar_packet(source_wrapper)
    assert candidate_wrapper.format_id == source_packet.format_id
    assert candidate_wrapper.sidecar_payload == source_packet.sidecar_payload
    assert candidate_wrapper.framing_meta == source_packet.framing_meta
    assert candidate_wrapper.pr106_bytes != source_packet.pr106_bytes

    cert = certify_hnerv_grammar_preserving_candidate_pair(
        source_archive=source,
        candidate_archive=candidate,
        label="PR106-sidecar-fixture",
    )
    assert cert["ready_for_archive_preflight"] is True
    assert cert["readiness_blockers"] == []
    assert cert["source_packet_ir"]["member_bytes"] == len(source_wrapper)
    assert cert["source_packet_ir"]["pr106_sidecar_wrapper"]["format_id"] == "0x01"
    assert cert["candidate_packet_ir"]["member_bytes"] == len(_single_payload(candidate))
    assert cert["candidate_packet_ir"]["pr106_sidecar_wrapper"]["sidecar_payload_bytes"] == len(b"opaque-sidecar")
    equivalence = {row["section_name"]: row for row in cert["section_equivalence"]}
    assert equivalence["decoder_packed_brotli"]["raw_equal"] is True


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


def test_packet_section_transform_cli_builds_candidate_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    candidate = tmp_path / "candidate.zip"
    manifest = tmp_path / "candidate.json"
    payload = _packed_payload(
        brotli.compress(b"decoder-record-" * 3000, quality=1),
        brotli.compress(b"latent-record-" * 1000, quality=1),
    )
    write_stored_single_member_zip(source, member_name="0.bin", payload=payload)

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_packet_section_transform_candidate.py"),
            "--source-archive",
            str(source),
            "--output-archive",
            str(candidate),
            "--label",
            "PR106-fixture",
            "--target-section",
            "decoder_packed_brotli",
            "--quality",
            "11",
            "--lgwin",
            "default",
            "--jobs",
            "1",
            "--json-out",
            str(manifest),
            "--fail-if-blocked",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert proc.stdout == ""
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["ready_for_archive_preflight"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["archive_byte_delta"] < 0
    assert candidate.exists()


def test_scan_brotli_recode_opportunities_marks_non_pr106_runtime_adapter(
    tmp_path: Path,
) -> None:
    pr106 = tmp_path / "pr106.zip"
    a2k1 = tmp_path / "a2k1.zip"
    raw = b"section-record-" * 4000
    source_brotli = brotli.compress(raw, quality=1)
    write_stored_single_member_zip(
        pr106,
        member_name="0.bin",
        payload=_packed_payload(source_brotli, brotli.compress(b"latents" * 100, quality=1)),
    )
    write_stored_single_member_zip(
        a2k1,
        member_name="x",
        payload=(
            b"A2K1"
            + len(source_brotli).to_bytes(4, "little")
            + source_brotli
            + b"l" * 15_387
            + b"sidecar"
        ),
    )

    result = scan_hnerv_brotli_recode_opportunities(
        [
            ("PR106-fixture", pr106, "pr106_ff_packed_hnerv"),
            ("A2K1-fixture", a2k1, "a2k1_variable_decoder_pr101"),
        ],
        qualities=(11,),
        lgwins=(None,),
        lgblocks=(None,),
        jobs=1,
    )

    assert result["score_claim"] is False
    assert result["dispatch_attempted"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["summary"]["rate_positive_count"] >= 2
    assert result["summary"]["candidate_compilable_by_existing_bridge_count"] >= 1
    assert result["summary"]["runtime_adapter_required_count"] >= 1
    pr106_row = next(
        row
        for row in result["sections"]
        if row["label"] == "PR106-fixture" and row["section_name"] == "decoder_packed_brotli"
    )
    assert pr106_row["ready_for_archive_preflight"] is True
    assert pr106_row["runtime_adapter_required"] is False
    a2k1_row = next(
        row
        for row in result["sections"]
        if row["label"] == "A2K1-fixture" and row["section_name"] == "decoder_blob"
    )
    assert a2k1_row["rate_positive"] is True
    assert a2k1_row["ready_for_archive_preflight"] is False
    assert a2k1_row["runtime_adapter_required"] is True
    assert any(
        blocker.startswith("runtime_adapter_required:")
        for blocker in a2k1_row["blockers"]
    )


def test_scan_brotli_recode_opportunities_cli_writes_json(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out = tmp_path / "scan.json"
    raw = b"decoder-record-" * 3000
    write_stored_single_member_zip(
        source,
        member_name="0.bin",
        payload=_packed_payload(
            brotli.compress(raw, quality=1),
            brotli.compress(b"latent-record-" * 500, quality=1),
        ),
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "scan_hnerv_brotli_section_recode_opportunities.py"),
            "--archive",
            f"fixture={source}",
            "--parser",
            "pr106_ff_packed_hnerv",
            "--quality",
            "11",
            "--lgwin",
            "default",
            "--jobs",
            "1",
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "hnerv_brotli_section_recode_opportunities.v1"
    assert payload["summary"]["candidate_compilable_by_existing_bridge_count"] >= 1


def test_certify_hnerv_grammar_preserving_pr106_pair_accepts_raw_equivalent_recode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    candidate = tmp_path / "candidate.zip"
    raw_decoder = b"decoder-record-" * 3000
    raw_latents = b"latent-record-" * 500
    source_payload = _packed_payload(
        brotli.compress(raw_decoder, quality=1),
        brotli.compress(raw_latents, quality=1),
    )
    candidate_decoder = brotli.compress(raw_decoder, quality=11)
    candidate_payload = _packed_payload(
        candidate_decoder,
        brotli.compress(raw_latents, quality=1),
    )
    write_stored_single_member_zip(source, member_name="0.bin", payload=source_payload)
    write_stored_single_member_zip(candidate, member_name="0.bin", payload=candidate_payload)

    result = certify_hnerv_grammar_preserving_candidate_pair(
        source_archive=source,
        candidate_archive=candidate,
        label="fixture",
    )

    assert result["score_claim"] is False
    assert result["dispatch_attempted"] is False
    assert result["ready_for_archive_preflight"] is True
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["grammar_preserving"] is True
    assert result["rate_positive"] is True
    assert result["readiness_blockers"] == []
    assert result["archive_byte_delta"] < 0
    equivalence = {row["section_name"]: row for row in result["section_equivalence"]}
    assert equivalence["packed_header_ff_len24"]["equivalence_kind"] == "pr106_len24_control"
    assert equivalence["decoder_packed_brotli"]["raw_equal"] is True
    assert equivalence["latents_and_sidecar_brotli"]["raw_equal"] is True


def test_certify_hnerv_grammar_preserving_pr106_pair_blocks_raw_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    candidate = tmp_path / "candidate.zip"
    source_payload = _packed_payload(
        brotli.compress(b"decoder-record-" * 3000, quality=1),
        brotli.compress(b"latent-record-" * 500, quality=1),
    )
    candidate_payload = _packed_payload(
        brotli.compress(b"different-decoder-record-" * 500, quality=11),
        brotli.compress(b"latent-record-" * 500, quality=1),
    )
    write_stored_single_member_zip(source, member_name="0.bin", payload=source_payload)
    write_stored_single_member_zip(candidate, member_name="0.bin", payload=candidate_payload)

    result = certify_hnerv_grammar_preserving_candidate_pair(
        source_archive=source,
        candidate_archive=candidate,
        label="fixture",
    )

    assert result["ready_for_archive_preflight"] is False
    assert result["grammar_preserving"] is False
    assert "brotli_raw_mismatch:decoder_packed_brotli" in result["readiness_blockers"]
    assert "brotli_raw_mismatch:decoder_packed_brotli" in result["dispatch_blockers"]


def test_certify_hnerv_packet_transform_cli_writes_json(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    candidate = tmp_path / "candidate.zip"
    out = tmp_path / "cert.json"
    raw_decoder = b"decoder-record-" * 3000
    source_payload = _packed_payload(
        brotli.compress(raw_decoder, quality=1),
        brotli.compress(b"latent-record-" * 500, quality=1),
    )
    candidate_payload = _packed_payload(
        brotli.compress(raw_decoder, quality=11),
        brotli.compress(b"latent-record-" * 500, quality=1),
    )
    write_stored_single_member_zip(source, member_name="0.bin", payload=source_payload)
    write_stored_single_member_zip(candidate, member_name="0.bin", payload=candidate_payload)

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "certify_hnerv_packet_transform_candidate.py"),
            "--source-archive",
            str(source),
            "--candidate-archive",
            str(candidate),
            "--label",
            "fixture",
            "--json-out",
            str(out),
            "--fail-if-blocked",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "hnerv_packet_transform_candidate_cert.v1"
    assert payload["ready_for_archive_preflight"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False


def _packed_payload(decoder_brotli: bytes, latents_brotli: bytes) -> bytes:
    return bytes([0xFF]) + len(decoder_brotli).to_bytes(3, "little") + decoder_brotli + latents_brotli


def _sidecar_payload(inner_payload: bytes) -> bytes:
    return emit_pr106_sidecar_packet(
        PR106SidecarPacket(
            format_id=PR106_SIDECAR_FORMAT_BROTLI,
            pr106_bytes=inner_payload,
            sidecar_payload=b"opaque-sidecar",
            framing_meta=None,
        )
    )


def _single_payload(archive: Path) -> bytes:
    import zipfile

    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
        assert len(infos) == 1
        return zf.read(infos[0])
