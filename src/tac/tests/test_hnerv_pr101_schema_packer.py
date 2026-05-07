from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli

from tac.hnerv_decoder_recode import PACKED_STATE_SCHEMA, parse_packed_decoder_brotli
from tac.hnerv_lowlevel_packer import (
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    write_stored_single_member_zip,
)
from tac.hnerv_pr101_schema_packer import (
    DECODER_STREAM_ENDS,
    VARIANT_NAME,
    build_pr101_schema_archive_candidate,
    decode_pr101_schema_split_fixture,
    encode_pr101_schema_split_fixture,
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
