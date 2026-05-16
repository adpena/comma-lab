from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.packet_compiler.pr106_context_recode import (
    build_pr106_context_recode_report,
    build_stored_zip_for_tests,
    decode_context_recode_section,
    emit_pr106_context_source_payload,
    encode_context_recode_section,
    load_pr106_context_source_from_archive,
    parse_pr106_context_source,
    prove_pr106_context_archive_identity,
    prove_pr106_context_source_identity,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools/plan_or_build_pr106_context_recode.py"
PR106_R2_ARCHIVE = REPO / "submissions/pr106_latent_sidecar_r2/archive.zip"
PR106_R2_SHA = "7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f"
PR106_R2_PR101_ARCHIVE = (
    REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip"
)
PR106_R2_PR101_SHA = "c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383"


def test_context_recode_section_changes_bytes_and_roundtrips() -> None:
    source = (b"ABCD" * 96) + (b"ABCE" * 96) + bytes(range(32))

    recode = encode_context_recode_section(
        "decoder_packed_brotli",
        source,
        context_order=2,
    )

    assert recode.encoded_bytes != source
    assert decode_context_recode_section(recode.encoded_bytes) == source
    manifest = recode.manifest()
    assert manifest["target_section_bytes_changed"] is True
    assert manifest["lossless_roundtrip_proven"] is True
    assert manifest["no_op_detector_passed"] is True
    assert "prototype_runtime_decoder_not_integrated" in manifest["blockers"]


def test_pr106_context_report_builds_fail_closed_changed_target() -> None:
    source = parse_pr106_context_source(_synthetic_pr106_payload())

    result = build_pr106_context_recode_report(
        source,
        target_section="decoder_packed_brotli",
        context_order=2,
        context_orders=(0, 1, 2),
        build_prototype=True,
    )

    report = result.report
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["zero_order_arithmetic_baseline_verdict"].startswith("falsified")
    assert result.prototype_section_bytes is not None
    candidate = report["prototype_candidate"]
    assert candidate["section_name"] == "decoder_packed_brotli"
    assert candidate["target_section_bytes_changed"] is True
    assert candidate["lossless_roundtrip_proven"] is True
    assert "prototype_runtime_decoder_not_integrated" in report["dispatch_blockers"]


def test_pr106_context_report_declares_low_order_blocked() -> None:
    source = parse_pr106_context_source(_synthetic_pr106_payload())

    result = build_pr106_context_recode_report(
        source,
        target_section="decoder_packed_brotli",
        context_order=0,
        context_orders=(0, 1, 2),
        build_prototype=True,
    )

    candidate = result.report["prototype_candidate"]
    assert candidate["target_section_bytes_changed"] is True
    assert "context_order_not_high_order" in candidate["blockers"]
    assert "zero_order_arithmetic_control_falsified_not_candidate" in candidate["blockers"]
    assert result.report["ready_for_exact_eval_dispatch"] is False


def test_pr106_context_source_identity_proof_is_nonpromotable() -> None:
    payload = _synthetic_pr106_payload()
    source = parse_pr106_context_source(payload)
    proof = prove_pr106_context_source_identity(source)

    assert emit_pr106_context_source_payload(source) == payload
    assert proof["schema"] == "pr106_context_source_identity_proof_v1"
    assert proof["context_packet_ir_identity_passed"] is True
    assert proof["blockers"] == []
    assert proof["emitted_inner_payload"]["byte_identical_to_source_inner"] is True
    assert proof["emitted_payload"]["byte_identical_to_source_payload"] is True
    assert proof["runtime_consumption_claim"] is False
    assert proof["full_frame_inflate_output_parity_claim"] is False
    assert proof["contest_axis_claim"] is False
    assert proof["score_claim"] is False
    assert proof["promotion_eligible"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_pr106_context_archive_identity_proof_on_release_archives() -> None:
    for archive, expected_sha, expected_format in (
        (PR106_R2_ARCHIVE, PR106_R2_SHA, "0x01"),
        (PR106_R2_PR101_ARCHIVE, PR106_R2_PR101_SHA, "0x02"),
    ):
        proof = prove_pr106_context_archive_identity(
            archive_path=archive,
            expected_archive_sha256=expected_sha,
        )

        assert proof["schema"] == "pr106_context_source_identity_proof_v1"
        assert proof["context_packet_ir_identity_passed"] is True
        assert proof["blockers"] == []
        assert proof["archive"]["expected_sha256_matches"] is True
        assert proof["member"]["name"] == "0.bin"
        assert proof["source"]["wrapper"]["format_id"] == expected_format
        assert proof["emitted_inner_payload"]["byte_identical_to_source_inner"] is True
        assert proof["emitted_payload"]["byte_identical_to_source_payload"] is True
        assert proof["emitted_archive"]["byte_identical_to_source_archive"] is True
        assert proof["score_claim"] is False
        assert proof["ready_for_exact_eval_dispatch"] is False


def test_pr106_context_archive_identity_proof_fails_closed_on_sha_mismatch() -> None:
    proof = prove_pr106_context_archive_identity(
        archive_path=PR106_R2_ARCHIVE,
        expected_archive_sha256="0" * 64,
    )

    assert proof["context_packet_ir_identity_passed"] is False
    assert proof["archive"]["expected_sha256_matches"] is False
    assert proof["blockers"] == ["expected_archive_sha256_mismatch"]
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_cli_writes_profile_markdown_and_prototype(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(build_stored_zip_for_tests("0.bin", _synthetic_pr106_payload()))
    json_out = tmp_path / "profile.json"
    md_out = tmp_path / "profile.md"
    prototype_out = tmp_path / "candidate.pcr1"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--archive",
            str(archive),
            "--target-section",
            "decoder_packed_brotli",
            "--context-order",
            "2",
            "--orders",
            "0,1,2",
            "--build-prototype",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
            "--prototype-out",
            str(prototype_out),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["prototype_candidate"]["target_section_bytes_changed"] is True
    assert report["ready_for_exact_eval_dispatch"] is False
    assert prototype_out.exists()
    assert "PR106 Context Recode Profile" in md_out.read_text(encoding="utf-8")
    assert load_pr106_context_source_from_archive(archive).source["member_name"] == "0.bin"


def test_cli_payload_input_is_not_overridden_by_default_archive(tmp_path: Path) -> None:
    payload = tmp_path / "member.bin"
    payload.write_bytes(_synthetic_pr106_payload())
    json_out = tmp_path / "profile.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--payload",
            str(payload),
            "--target-section",
            "latents_and_sidecar_brotli",
            "--context-order",
            "2",
            "--orders",
            "0,1,2",
            "--json-out",
            str(json_out),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["source"]["mode"] == "payload"
    assert report["source"]["path"] == str(payload)
    assert "archive_sha256" not in report["source"]


def _synthetic_pr106_payload() -> bytes:
    decoder = (b"abcabcabd" * 24) + bytes([5, 5, 6, 5, 5, 7]) * 8
    latents = (b"\x00\x01\x00\x01\x00\x02" * 64) + b"tail"
    return b"\xff" + len(decoder).to_bytes(3, "little") + decoder + latents
