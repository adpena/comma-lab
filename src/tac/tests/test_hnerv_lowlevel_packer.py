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
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
    write_stored_single_member_zip,
)
from tac.hnerv_section_repack import audit_candidate_section_diff, build_section_repack_plan

REPO = Path(__file__).resolve().parents[3]


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


def test_brotli_recode_search_parallel_matches_serial() -> None:
    source = brotli.compress((b"parallel-hnerv-search-" * 4000), quality=1)

    serial_choice, serial_payload = brotli_recode_search(
        "decoder_packed_brotli",
        source,
        qualities=[9, 10, 11],
        lgwins=[None, 18, 20, 22],
        jobs=1,
    )
    parallel_choice, parallel_payload = brotli_recode_search(
        "decoder_packed_brotli",
        source,
        qualities=[9, 10, 11],
        lgwins=[None, 18, 20, 22],
        jobs=4,
    )

    assert parallel_choice == serial_choice
    assert parallel_payload == serial_payload


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
