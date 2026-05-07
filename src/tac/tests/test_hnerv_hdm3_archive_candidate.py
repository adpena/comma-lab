from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import brotli

from tac.hnerv_decoder_recode import (
    PACKED_STATE_SCHEMA,
    decode_hdm3_q_brotli_split_fixture,
)
from tac.hnerv_hdm3_archive_candidate import build_hdm3_archive_candidate
from tac.hnerv_lowlevel_packer import (
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    write_stored_single_member_zip,
)

REPO = Path(__file__).resolve().parents[3]


def test_build_hdm3_archive_candidate_is_byte_closed_but_not_dispatch_ready(
    tmp_path: Path,
) -> None:
    source_archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(source_archive)
    source_packed = parse_ff_packed_brotli_hnerv(source.payload)
    source_raw = brotli.decompress(source_packed.decoder_packed_brotli)

    manifest = build_hdm3_archive_candidate(
        source_archive=source_archive,
        output_dir=tmp_path / "out",
        source_label="PR106x frontier",
        repo_root=REPO,
    )

    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_archive_preflight"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["archive_build_gate"] is True
    assert manifest["candidate_variant"] == "hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales"
    assert manifest["candidate_rate_positive"] is True
    assert manifest["candidate_decoder_section_byte_delta"] < 0
    assert manifest["candidate_archive_sha256"] != source.archive_sha256
    assert manifest["decoder_raw_equivalence"]["raw_equal"] is True
    assert manifest["decoder_raw_equivalence"]["q_roundtrip_equal"] is True
    assert manifest["decoder_raw_equivalence"]["scale_roundtrip_equal"] is True
    assert manifest["runtime_adapter_proof"]["submission_runtime_integrated"] is True
    assert manifest["runtime_adapter_proof"]["runtime_adapter_module"] == "tac.hnerv_hdm3_runtime_adapter"
    assert "hdm3_runtime_adapter_archive_parity_proof_missing" in manifest["dispatch_blockers"]
    assert "exact_cuda_auth_eval_missing" in manifest["dispatch_blockers"]

    candidate_archive = REPO / manifest["candidate_archive_path"]
    if not candidate_archive.exists():
        candidate_archive = Path(manifest["candidate_archive_path"])
    assert candidate_archive.exists()
    candidate = read_strict_single_member_zip(candidate_archive)
    assert candidate.member_name == source.member_name
    candidate_packed = parse_ff_packed_brotli_hnerv(candidate.payload)
    assert candidate_packed.decoder_packed_brotli.startswith(b"HDM3")
    assert candidate_packed.latents_and_sidecar_brotli == source_packed.latents_and_sidecar_brotli
    restored = decode_hdm3_q_brotli_split_fixture(candidate_packed.decoder_packed_brotli)
    assert restored.to_raw() == source_raw

    with zipfile.ZipFile(candidate_archive) as zf:
        infos = zf.infolist()
    assert [info.filename for info in infos] == ["x"]
    assert infos[0].date_time == (1980, 1, 1, 0, 0, 0)


def test_build_hdm3_archive_candidate_fails_closed_without_rate_win(tmp_path: Path) -> None:
    raw = _synthetic_context_decoder_raw()
    decoder_brotli = brotli.compress(raw, quality=11)
    latents = brotli.compress(b"latents", quality=11)
    source_archive = tmp_path / "source.zip"
    write_stored_single_member_zip(
        source_archive,
        member_name="x",
        payload=_packed_payload(decoder_brotli, latents),
    )

    manifest = build_hdm3_archive_candidate(
        source_archive=source_archive,
        output_dir=tmp_path / "out",
        source_label="no win",
        repo_root=REPO,
    )

    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "hdm3_decoder_section_not_rate_positive" in manifest["archive_build_blockers"]
    assert manifest["candidate_archive_path"] == ""
    assert manifest["decoder_raw_equivalence"]["raw_equal"] is True


def test_build_hnerv_hdm3_archive_candidate_cli_writes_manifest(tmp_path: Path) -> None:
    source_archive = _source_archive(tmp_path)
    json_out = tmp_path / "result.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_hdm3_archive_candidate.py"),
            "--source-archive",
            str(source_archive),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-label",
            "PR106x",
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
    assert payload["tool_run_manifest"]["tool"] == "tools/build_hnerv_hdm3_archive_candidate.py"
    assert Path(payload["candidate_archive_path"]).exists()


def _source_archive(tmp_path: Path) -> Path:
    raw = _synthetic_context_decoder_raw()
    decoder_brotli = brotli.compress(raw, quality=0)
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
