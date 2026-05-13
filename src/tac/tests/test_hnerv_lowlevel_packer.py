from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import brotli
import pytest

from tac.hnerv_lowlevel_packer import (
    HnervLowlevelPackError,
    brotli_recode_search,
    build_lowlevel_brotli_repack_candidate,
    parse_a1_headered_split_brotli_hnerv,
    parse_ff_packed_brotli_hnerv,
    read_packed_archive_view,
    read_strict_single_member_zip,
    sha256_bytes,
    write_stored_single_member_zip,
)
from tac.hnerv_section_repack import audit_candidate_section_diff, build_section_repack_plan
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_SIDECAR_FORMAT_BROTLI,
    emit_pr106_sidecar_packet,
    parse_pr106_sidecar_packet,
)

REPO = Path(__file__).resolve().parents[3]
PR106_R2_PR101_ARCHIVE = REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip"


def test_parse_ff_packed_payload_round_trips_sections() -> None:
    payload = _packed_payload(
        brotli.compress(b"decoder" * 100, quality=1),
        brotli.compress(b"latent" * 100, quality=1),
    )

    parsed = parse_ff_packed_brotli_hnerv(payload)

    assert parsed.header == payload[:4]
    assert parsed.to_bytes() == payload
    assert brotli.decompress(parsed.decoder_packed_brotli).startswith(b"decoder")
    assert brotli.decompress(parsed.latents_and_sidecar_brotli).startswith(b"latent")


def test_parse_a1_headered_split_brotli_payload_round_trips_sections() -> None:
    payload = _a1_payload(
        [
            brotli.compress((f"decoder-stream-{idx}-".encode() * 500), quality=1)
            for idx in range(7)
        ],
        b"latent-and-sidecar",
    )

    parsed = parse_a1_headered_split_brotli_hnerv(payload)

    assert parsed.header_section_name == "packed_header_a1_u32_section_total"
    assert parsed.to_bytes() == payload
    assert parsed.latents_and_sidecar_brotli == b"latent-and-sidecar"


def test_strict_single_member_zip_rejects_zip_slip(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    write_stored_single_member_zip(tmp_path / "good.zip", member_name="x", payload=b"ok")
    with pytest.raises(HnervLowlevelPackError, match="parent"):
        write_stored_single_member_zip(archive, member_name="../x", payload=b"bad")
    with pytest.raises(HnervLowlevelPackError, match="hidden"):
        write_stored_single_member_zip(archive, member_name=".x", payload=b"bad")
    with pytest.raises(HnervLowlevelPackError, match="backslash"):
        write_stored_single_member_zip(archive, member_name="bad\\x", payload=b"bad")

    good = read_strict_single_member_zip(tmp_path / "good.zip")
    assert good.member_name == "x"
    assert good.payload == b"ok"

    dir_archive = tmp_path / "dir.zip"
    with zipfile.ZipFile(dir_archive, "w") as zf:
        zf.writestr("nested/", b"")
        zf.writestr("x", b"ok")
    with pytest.raises(HnervLowlevelPackError, match="exactly one ZIP entry"):
        read_strict_single_member_zip(dir_archive)


def test_build_lowlevel_brotli_repack_candidate_proves_changed_section(tmp_path: Path) -> None:
    decoder = brotli.compress((b"decoder-record-" * 3000), quality=1)
    latents = brotli.compress((b"latent-row-" * 2000), quality=1)
    source_archive = tmp_path / "source.zip"
    payload = _packed_payload(decoder, latents)
    write_stored_single_member_zip(source_archive, member_name="x", payload=payload)
    source = read_strict_single_member_zip(source_archive)
    scorecard = _scorecard(source, "PR106x")

    result = build_lowlevel_brotli_repack_candidate(
        source_archive=source_archive,
        scorecard=scorecard,
        source_label="PR106x",
        output_dir=tmp_path / "out",
        target_sections=["decoder_packed_brotli"],
        qualities=[11],
        lgwins=[None],
    )

    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["ready_for_archive_preflight"] is True
    assert Path(result["candidate_archive_path"]).exists()
    assert result["candidate_archive_sha256"] != source.archive_sha256
    assert all(row["raw_equal"] is True for row in result["brotli_raw_equivalence"])
    assert all(row["raw_equal"] is True for row in result["candidate_diff"]["brotli_raw_equivalence"])
    audit = result["candidate_diff_audit"]
    assert audit["changed_section_count"] >= 1
    assert audit["total_byte_delta"] < 0
    assert audit["ready_for_archive_preflight"] is True

    plan = build_section_repack_plan(scorecard, labels=["PR106x"])
    reaudit = audit_candidate_section_diff(plan, result["candidate_diff"])
    assert reaudit["ready_for_archive_preflight"] is True


def test_read_packed_archive_view_unwraps_real_pr106_sidecar_wrapper() -> None:
    view = read_packed_archive_view(PR106_R2_PR101_ARCHIVE)

    assert view.payload_kind == "pr106_sidecar_wrapper"
    assert view.sidecar_packet is not None
    assert view.sidecar_packet.format_id == 0x02
    assert view.packed.to_bytes() == view.sidecar_packet.pr106_bytes


def test_build_lowlevel_brotli_repack_candidate_reemits_pr106_sidecar_wrapper(
    tmp_path: Path,
) -> None:
    decoder = brotli.compress((b"wrapped-decoder-record-" * 3000), quality=1)
    latents = brotli.compress((b"wrapped-latent-row-" * 2000), quality=1)
    inner_payload = _packed_payload(decoder, latents)
    source_packet_payload = emit_pr106_sidecar_packet(
        _sidecar_packet(inner_payload, sidecar_payload=b"runtime-visible-sidecar")
    )
    source_archive = tmp_path / "source.zip"
    write_stored_single_member_zip(
        source_archive,
        member_name="0.bin",
        payload=source_packet_payload,
    )

    result = build_lowlevel_brotli_repack_candidate(
        source_archive=source_archive,
        scorecard={"schema_version": 1, "payload_section_manifests": []},
        source_label="PR106_R2_PR101_GRAMMAR",
        output_dir=tmp_path / "out",
        target_sections=["decoder_packed_brotli"],
        qualities=[11],
        lgwins=[None],
    )

    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["ready_for_archive_preflight"] is True
    assert result["source_payload_kind"] == "pr106_sidecar_wrapper"
    assert result["scorecard_anchor"]["derived_from_source_archive"] is True
    assert result["packet_ir_consumed_byte_proof"]["all_payload_bytes_accounted"] is True
    assert result["sidecar_packet"]["sidecar_payload_preserved"] is True
    assert result["sidecar_packet"]["framing_meta_preserved"] is True

    candidate = read_strict_single_member_zip(result["candidate_archive_path"])
    candidate_packet = parse_pr106_sidecar_packet(candidate.payload)
    source_packet = parse_pr106_sidecar_packet(source_packet_payload)
    assert candidate_packet.sidecar_payload == source_packet.sidecar_payload
    assert candidate_packet.framing_meta == source_packet.framing_meta
    assert candidate_packet.pr106_bytes != source_packet.pr106_bytes
    candidate_inner = parse_ff_packed_brotli_hnerv(candidate_packet.pr106_bytes)
    source_inner = parse_ff_packed_brotli_hnerv(source_packet.pr106_bytes)
    assert brotli.decompress(candidate_inner.decoder_packed_brotli) == brotli.decompress(
        source_inner.decoder_packed_brotli
    )


def test_build_lowlevel_brotli_repack_candidate_reemits_a1_split_brotli(
    tmp_path: Path,
) -> None:
    source_streams = [
        brotli.compress((f"a1-decoder-stream-{idx}-".encode() * 4000), quality=1)
        for idx in range(7)
    ]
    source_archive = tmp_path / "a1.zip"
    write_stored_single_member_zip(
        source_archive,
        member_name="x",
        payload=_a1_payload(source_streams, b"latent-lzma-and-sidecar"),
    )

    result = build_lowlevel_brotli_repack_candidate(
        source_archive=source_archive,
        scorecard={"schema_version": 1, "payload_section_manifests": []},
        source_label="A1",
        output_dir=tmp_path / "out",
        qualities=[11],
        lgwins=[None],
    )

    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["ready_for_archive_preflight"] is True
    assert result["source_payload_kind"] == "a1_headered_split_brotli_hnerv"
    assert result["scorecard_anchor"]["derived_from_source_archive"] is True
    assert {row["section_name"] for row in result["attempts"]} == {"decoder_packed_brotli"}
    assert all(row["raw_equal"] is True for row in result["brotli_raw_equivalence"])

    candidate = read_strict_single_member_zip(result["candidate_archive_path"])
    candidate_parsed = parse_a1_headered_split_brotli_hnerv(candidate.payload)
    source_parsed = parse_a1_headered_split_brotli_hnerv(
        read_strict_single_member_zip(source_archive).payload
    )
    assert candidate_parsed.latents_and_sidecar_brotli == source_parsed.latents_and_sidecar_brotli
    assert candidate_parsed.decoder_packed_brotli != source_parsed.decoder_packed_brotli


def test_brotli_recode_search_parallel_matches_serial() -> None:
    source = brotli.compress((b"parallel-hnerv-search-" * 4000), quality=1)

    serial_choice, serial_payload = brotli_recode_search(
        "decoder_packed_brotli",
        source,
        qualities=[9, 10, 11],
        lgwins=[None, 18, 20, 22],
        lgblocks=[None, 16],
        jobs=1,
    )
    parallel_choice, parallel_payload = brotli_recode_search(
        "decoder_packed_brotli",
        source,
        qualities=[9, 10, 11],
        lgwins=[None, 18, 20, 22],
        lgblocks=[None, 16],
        jobs=4,
    )

    assert parallel_choice == serial_choice
    assert parallel_payload == serial_payload
    assert parallel_choice.lgblock in {None, 16}


def test_build_lowlevel_brotli_repack_candidate_blocks_noop(tmp_path: Path) -> None:
    decoder = brotli.compress((b"already-best-" * 3000), quality=11)
    latents = brotli.compress((b"latent-best-" * 2000), quality=11)
    source_archive = tmp_path / "source.zip"
    payload = _packed_payload(decoder, latents)
    write_stored_single_member_zip(source_archive, member_name="x", payload=payload)
    source = read_strict_single_member_zip(source_archive)

    result = build_lowlevel_brotli_repack_candidate(
        source_archive=source_archive,
        scorecard=_scorecard(source, "PR106x"),
        source_label="PR106x",
        output_dir=tmp_path / "out",
        target_sections=["decoder_packed_brotli"],
        qualities=[11],
        lgwins=[None],
    )

    assert result["ready_for_archive_preflight"] is False
    assert "no_rate_positive_section_recode" in result["blockers"]
    assert "candidate_archive_path" not in result


def test_build_lowlevel_brotli_repack_candidate_fails_closed_on_stale_scorecard(tmp_path: Path) -> None:
    decoder = brotli.compress((b"decoder-stale-" * 3000), quality=1)
    latents = brotli.compress((b"latent-stale-" * 2000), quality=1)
    source_archive = tmp_path / "source.zip"
    write_stored_single_member_zip(source_archive, member_name="x", payload=_packed_payload(decoder, latents))
    source = read_strict_single_member_zip(source_archive)
    scorecard = _scorecard(source, "PR106x")
    del scorecard["payload_section_manifests"][0]["payload_sha256"]

    result = build_lowlevel_brotli_repack_candidate(
        source_archive=source_archive,
        scorecard=scorecard,
        source_label="PR106x",
        output_dir=tmp_path / "out",
        target_sections=["decoder_packed_brotli"],
        qualities=[11],
        lgwins=[None],
    )

    assert result["ready_for_archive_preflight"] is False
    assert "source_payload_sha256_missing_or_invalid" in result["candidate_diff_audit"]["blockers"]


def test_build_hnerv_lowlevel_repack_candidate_cli(tmp_path: Path) -> None:
    decoder = brotli.compress((b"decoder-cli-" * 3000), quality=1)
    latents = brotli.compress((b"latent-cli-" * 2000), quality=1)
    source_archive = tmp_path / "source.zip"
    write_stored_single_member_zip(source_archive, member_name="x", payload=_packed_payload(decoder, latents))
    source = read_strict_single_member_zip(source_archive)
    scorecard = tmp_path / "scorecard.json"
    json_out = tmp_path / "result.json"
    scorecard.write_text(json.dumps(_scorecard(source, "PR106x")), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_lowlevel_repack_candidate.py"),
            "--source-archive",
            str(source_archive),
            "--scorecard",
            str(scorecard),
            "--source-label",
            "PR106x",
            "--output-dir",
            str(tmp_path / "out"),
            "--target-section",
            "decoder_packed_brotli",
            "--quality",
            "11",
            "--lgblock",
            "default",
            "--json-out",
            str(json_out),
            "--fail-if-blocked",
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text())
    assert payload["ready_for_archive_preflight"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["candidate_diff_audit"]["changed_section_count"] >= 1


def _packed_payload(decoder_brotli: bytes, latents_brotli: bytes) -> bytes:
    if len(decoder_brotli) > 0xFFFFFF:
        raise AssertionError("test decoder too large")
    return bytes([0xFF]) + len(decoder_brotli).to_bytes(3, "little") + decoder_brotli + latents_brotli


def _a1_payload(decoder_streams: list[bytes], tail: bytes) -> bytes:
    decoder_blob = b"".join(decoder_streams)
    section_total = 4 + len(decoder_blob)
    return section_total.to_bytes(4, "little") + decoder_blob + tail


def _sidecar_packet(inner_payload: bytes, *, sidecar_payload: bytes):
    from tac.packet_compiler.pr106_sidecar_packet import PR106SidecarPacket

    return PR106SidecarPacket(
        format_id=PR106_SIDECAR_FORMAT_BROTLI,
        pr106_bytes=inner_payload,
        sidecar_payload=sidecar_payload,
        framing_meta=None,
    )


def _scorecard(source, label: str) -> dict:
    parsed = parse_ff_packed_brotli_hnerv(source.payload)
    sections = []
    start = 0
    for index, (name, data, role) in enumerate(
        [
            ("packed_header_ff_len24", parsed.header, "control_or_metadata"),
            ("decoder_packed_brotli", parsed.decoder_packed_brotli, "decoder_weight_stream"),
            ("latents_and_sidecar_brotli", parsed.latents_and_sidecar_brotli, "latent_stream"),
        ]
    ):
        end = start + len(data)
        sections.append(
            {
                "index": index,
                "name": name,
                "start": start,
                "end": end,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "entropy_bits_per_byte": 7.0,
                "optimization_role": role,
            }
        )
        start = end
    return {
        "schema_version": 1,
        "tool": "build_hnerv_frontier_scorecard",
        "score_truth": "byte_fixture",
        "payload_section_manifests": [
            {
                "label": label,
                "archive_sha256": source.archive_sha256,
                "archive_bytes": source.archive_bytes,
                "zip_member": source.member_name,
                "payload_sha256": sha256_bytes(source.payload),
                "member_bytes": source.member_bytes,
                "profile_match_key": "member_sha256",
                "score_claim": False,
                "dispatch_attempted": False,
                "sections": sections,
            }
        ],
    }
