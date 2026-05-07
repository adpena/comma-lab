from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli
import pytest

from tac.hnerv_decoder_recode import PACKED_STATE_SCHEMA, parse_packed_decoder_brotli
from tac.hnerv_lowlevel_packer import (
    PackedHnervPayload,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    write_stored_single_member_zip,
)
from tac.hnerv_pr101_schema_packer import (
    DECODER_STREAM_ENDS,
    FP16_SCALE_PROBE_VARIANT_NAME,
    VARIANT_NAME,
    build_pr101_schema_archive_candidate,
    decode_pr101_schema_split_fixture,
    decode_pr101_schema_split_fp16_scale_probe,
    encode_pr101_schema_split_fixture,
    encode_pr101_schema_split_fp16_scale_probe,
    restore_pr101_schema_payload_to_legacy_brotli,
)

REPO = Path(__file__).resolve().parents[3]


def test_pr101_schema_split_fixture_roundtrips_pr106_raw() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))

    payload, stats = encode_pr101_schema_split_fixture(parsed)
    restored = decode_pr101_schema_split_fixture(payload)

    assert restored.to_raw() == parsed.to_raw()
    assert stats["variant"] == VARIANT_NAME
    assert stats["stream_count"] == len(DECODER_STREAM_ENDS)
    assert stats["scale_bytes_per_record"] == 4
    assert stats["raw_bytes"] == len(parsed.to_raw())
    assert len(stats["stream_rows"]) == len(DECODER_STREAM_ENDS)
    assert payload == encode_pr101_schema_split_fixture(parsed)[0]


def test_pr101_schema_fp16_scale_probe_preserves_q_but_not_raw_scales() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))

    payload, stats = encode_pr101_schema_split_fp16_scale_probe(parsed)
    restored = decode_pr101_schema_split_fp16_scale_probe(payload)

    assert restored.q_stream == parsed.q_stream
    assert restored.scale_stream != parsed.scale_stream
    assert restored.to_raw() != parsed.to_raw()
    assert stats["variant"] == FP16_SCALE_PROBE_VARIANT_NAME
    assert stats["scale_bytes_per_record"] == 2
    assert stats["q_roundtrip_equal"] is True
    assert stats["scale_raw_equal"] is False
    assert stats["raw_equivalence_claim"] is False
    assert stats["runtime_parity_required"] is True
    assert stats["hidden_gem_classification"]["classification"] == "substitute_candidate_probe"
    assert stats["hidden_gem_classification"]["score_claim"] is False
    assert stats["payload_bytes"] <= encode_pr101_schema_split_fixture(parsed)[1]["payload_bytes"]


def test_pr101_schema_runtime_adapter_restores_legacy_brotli_payload() -> None:
    parsed = parse_packed_decoder_brotli(brotli.compress(_synthetic_context_decoder_raw(), quality=5))
    schema_stream, _stats = encode_pr101_schema_split_fixture(parsed)
    legacy_decoder = brotli.compress(parsed.to_raw(), quality=5)
    candidate_payload = PackedHnervPayload(
        header=b"\xff" + len(schema_stream).to_bytes(3, "little"),
        decoder_packed_brotli=schema_stream,
        latents_and_sidecar_brotli=brotli.compress(b"latents", quality=5),
    ).to_bytes()

    with pytest.raises(brotli.error):
        brotli.decompress(schema_stream)

    restored_payload, proof = restore_pr101_schema_payload_to_legacy_brotli(
        candidate_payload,
        require_pr101_schema=True,
    )

    restored = parse_ff_packed_brotli_hnerv(restored_payload)
    assert proof["mode"] == "pr101_schema_restored_to_legacy_brotli"
    assert proof["score_claim"] is False
    assert proof["dispatch_attempted"] is False
    assert proof["legacy_decode_without_adapter"] is False
    assert proof["ready_for_public_runtime_inflate"] is True
    assert proof["ready_for_exact_eval_dispatch"] is False
    assert proof["payload_changed"] is True
    assert brotli.decompress(restored.decoder_packed_brotli) == parsed.to_raw()
    assert restored.latents_and_sidecar_brotli == parse_ff_packed_brotli_hnerv(
        candidate_payload
    ).latents_and_sidecar_brotli

    legacy_payload = PackedHnervPayload(
        header=b"\xff" + len(legacy_decoder).to_bytes(3, "little"),
        decoder_packed_brotli=legacy_decoder,
        latents_and_sidecar_brotli=restored.latents_and_sidecar_brotli,
    ).to_bytes()
    passthrough_payload, passthrough = restore_pr101_schema_payload_to_legacy_brotli(
        legacy_payload
    )
    assert passthrough_payload == legacy_payload
    assert passthrough["mode"] == "legacy_brotli_passthrough"
    assert passthrough["legacy_decode_without_adapter"] is True


def test_build_pr101_schema_archive_candidate_allows_forensic_archive_with_override(
    tmp_path: Path,
) -> None:
    source_archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(source_archive)
    source_packed = parse_ff_packed_brotli_hnerv(source.payload)
    source_raw = brotli.decompress(source_packed.decoder_packed_brotli)

    manifest = build_pr101_schema_archive_candidate(
        source_archive=source_archive,
        output_dir=tmp_path / "out",
        source_label="PR106x lowlevel",
        allow_rate_regression=True,
        repo_root=REPO,
    )

    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_archive_preflight"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["candidate_variant"] == VARIANT_NAME
    assert manifest["candidate_archive_sha256"] != source.archive_sha256
    assert manifest["decoder_raw_equivalence"]["raw_equal"] is True
    assert manifest["fp16_scale_probe"]["variant"] == FP16_SCALE_PROBE_VARIANT_NAME
    assert manifest["fp16_scale_probe"]["score_claim"] is False
    assert manifest["fp16_scale_probe"]["hidden_gem_classification"]["substitute_candidate"] is True
    assert manifest["fp16_scale_probe"]["q_roundtrip_equal"] is True
    assert manifest["fp16_scale_probe"]["scale_raw_equal"] is False
    assert manifest["fp16_scale_probe"]["ready_for_exact_eval_dispatch"] is False
    assert manifest["runtime_adapter_proof"]["schema_payload_adapter_parity_closed"] is True
    assert manifest["runtime_adapter_proof"]["candidate_legacy_decode_without_adapter"] is False
    assert manifest["runtime_adapter_proof"]["ready_for_public_runtime_inflate"] is True
    expected_classification = (
        "stack_candidate"
        if manifest["candidate_decoder_section_byte_delta"] < 0
        else "blocked_schema_probe"
    )
    assert manifest["hidden_gem_classification"]["classification"] == expected_classification
    assert manifest["hidden_gem_classification"]["runtime_adapter_payload_parity_closed"] is True
    assert "pr101_schema_submission_runtime_adapter_not_integrated" in manifest["dispatch_blockers"]
    assert "exact_cuda_auth_eval_missing" in manifest["dispatch_blockers"]

    candidate_archive = REPO / manifest["candidate_archive_path"]
    if not candidate_archive.exists():
        candidate_archive = Path(manifest["candidate_archive_path"])
    candidate = read_strict_single_member_zip(candidate_archive)
    candidate_packed = parse_ff_packed_brotli_hnerv(candidate.payload)
    restored = decode_pr101_schema_split_fixture(candidate_packed.decoder_packed_brotli)
    assert restored.to_raw() == source_raw
    assert candidate_packed.latents_and_sidecar_brotli == source_packed.latents_and_sidecar_brotli


def test_pr101_schema_archive_candidate_blocks_no_rate_win(tmp_path: Path) -> None:
    source_archive = _source_archive(tmp_path, source_quality=11)

    manifest = build_pr101_schema_archive_candidate(
        source_archive=source_archive,
        output_dir=tmp_path / "out",
        source_label="no win",
        repo_root=REPO,
    )

    assert manifest["ready_for_archive_preflight"] is False
    assert "pr101_schema_decoder_section_not_rate_positive" in manifest["archive_build_blockers"]
    assert manifest["candidate_archive_path"] == ""
    assert manifest["decoder_raw_equivalence"]["raw_equal"] is True


def test_build_hnerv_pr101_schema_candidate_cli_writes_manifest(tmp_path: Path) -> None:
    source_archive = _source_archive(tmp_path)
    json_out = tmp_path / "result.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_pr101_schema_candidate.py"),
            "--source-archive",
            str(source_archive),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-label",
            "PR106x",
            "--allow-rate-regression",
            "--json-out",
            str(json_out),
            "--fail-if-blocked",
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["ready_for_archive_preflight"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["tool_run_manifest"]["tool"] == "tools/build_hnerv_pr101_schema_candidate.py"
    assert Path(payload["candidate_archive_path"]).exists()


def test_build_hnerv_pr101_schema_candidate_cli_can_emit_fp16_probe_archive(tmp_path: Path) -> None:
    source_archive = _source_archive(tmp_path)
    json_out = tmp_path / "result.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_pr101_schema_candidate.py"),
            "--source-archive",
            str(source_archive),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-label",
            "PR106x",
            "--allow-rate-regression",
            "--emit-fp16-probe-archive",
            "--json-out",
            str(json_out),
            "--fail-if-blocked",
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    probe = payload["fp16_scale_probe"]
    assert probe["score_claim"] is False
    assert probe["ready_for_exact_eval_dispatch"] is False
    assert probe["probe_archive_sha256"]
    assert Path(probe["probe_archive_path"]).exists()
    assert "pr101_fp16_scale_exact_cuda_auth_eval_missing" in probe["dispatch_blockers"]


def _source_archive(tmp_path: Path, *, source_quality: int = 0) -> Path:
    raw = _synthetic_context_decoder_raw()
    decoder_brotli = brotli.compress(raw, quality=source_quality)
    latents = brotli.compress(b"latents" * 100, quality=5)
    source_archive = tmp_path / "source.zip"
    write_stored_single_member_zip(
        source_archive,
        member_name="x",
        payload=_packed_payload(decoder_brotli, latents),
    )
    return source_archive


def _packed_payload(decoder_brotli: bytes, latents_brotli: bytes) -> bytes:
    return bytes([0xFF]) + len(decoder_brotli).to_bytes(3, "little") + decoder_brotli + latents_brotli


def _synthetic_context_decoder_raw() -> bytes:
    q_parts = []
    scale_parts = []
    for index, (_name, shape) in enumerate(PACKED_STATE_SCHEMA):
        count = 1
        for dim in shape:
            count *= dim
        pattern = bytes(((index + i // 5) % 17) for i in range(64))
        repeats, remainder = divmod(count, len(pattern))
        q_parts.append(pattern * repeats + pattern[:remainder])
        scale_parts.append((index + 1).to_bytes(4, "little"))
    return b"".join(q_parts) + b"".join(scale_parts)
