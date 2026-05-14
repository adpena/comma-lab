# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO_ROOT / "tools" / "build_hnerv_generated_schema_candidate.py"
PROOF_PATH = REPO_ROOT / "tools" / "prove_hnerv_generated_schema_runtime_packet.py"


def _load_tool(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    hngs_decoder = tmp_path / "decoder.hngs"
    latent_blob = tmp_path / "latents.bin"
    sidecar_blob = tmp_path / "sidecar.bin"
    hngs_decoder.write_bytes(b"HNGS" + b"\x00runtime-proof-decoder")
    latent_blob.write_bytes(b"latent-runtime-proof")
    sidecar_blob.write_bytes(b"sidecar-runtime-proof")
    return hngs_decoder, latent_blob, sidecar_blob


def _build_candidate(tmp_path: Path, candidate_id: str = "unit_runtime_proof") -> Path:
    builder = _load_tool(BUILDER_PATH, "build_hnerv_generated_schema_candidate_for_proof")
    hngs_decoder, latent_blob, sidecar_blob = _write_inputs(tmp_path)
    archive = tmp_path / f"{candidate_id}.zip"
    builder.build_hnerv_generated_schema_candidate_archive(
        hngs_decoder=hngs_decoder,
        latent_blob=latent_blob,
        sidecar_blob=sidecar_blob,
        output_archive=archive,
        manifest_output=tmp_path / f"{candidate_id}.manifest.json",
        candidate_id=candidate_id,
    )
    return archive


def _write_deterministic_zip(path: Path, member_name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(member_name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.flag_bits = 0
    info.internal_attr = 0
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def test_proves_good_hnerv_generated_schema_runtime_packet(tmp_path: Path) -> None:
    proof_tool = _load_tool(PROOF_PATH, "prove_hnerv_generated_schema_runtime_packet_good")
    archive = _build_candidate(tmp_path)

    payload = proof_tool.build_hnerv_generated_schema_runtime_packet_proof(
        candidate_archive=archive,
        command_text="prove runtime packet in unit test",
    )

    archive_bytes = archive.read_bytes()
    assert payload["schema"] == "tac_hnerv_generated_schema_runtime_packet_proof_v1"
    assert payload["proof_family"] == "tac_runtime_consumption_proof_v1"
    assert payload["ready_for_exact_eval_runtime"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["omx_state_touched"] is False
    assert payload["blockers"] == []
    assert payload["candidate_archive_bytes"] == len(archive_bytes)
    assert payload["candidate_archive_sha256"] == hashlib.sha256(archive_bytes).hexdigest()
    assert payload["member_sha256"] == payload["new_member_sha256"]
    assert payload["oracle_packet_sha256"] == payload["member_sha256"]
    assert payload["standalone_packet_sha256"] == payload["member_sha256"]
    assert payload["section_names_match"] is True
    assert payload["section_offsets_match"] is True
    assert payload["section_lengths_match"] is True
    assert payload["section_sha256s_match"] is True
    assert [row["section_name"] for row in payload["consumed_sections"]] == [
        "header",
        "hngs_decoder",
        "latent_blob",
        "sidecar_blob",
    ]
    assert [row["section_name"] for row in payload["changed_sections"]] == [
        "hngs_decoder",
        "latent_blob",
        "sidecar_blob",
    ]
    assert all(row["runtime_consumed"] is True for row in payload["consumed_sections"])
    assert len(payload["command_sha256"]) == 64
    assert len(payload["proof_transcript_sha256"]) == 64


def test_tampered_archive_fails_closed_on_zip_metadata(tmp_path: Path) -> None:
    proof_tool = _load_tool(PROOF_PATH, "prove_hnerv_generated_schema_runtime_packet_tampered")
    archive = _build_candidate(tmp_path, candidate_id="unit_tampered_runtime_proof")
    with zipfile.ZipFile(archive, "a") as zf:
        zf.comment = b"tampered"

    payload = proof_tool.build_hnerv_generated_schema_runtime_packet_proof(
        candidate_archive=archive,
        command_text="prove tampered archive",
    )

    assert payload["ready_for_exact_eval_runtime"] is False
    assert payload["score_claim"] is False
    assert "zip_comment_present" in payload["blockers"]
    assert payload["archive"]["zip_comment_bytes"] == len(b"tampered")
    assert payload["member_sha256"] == payload["oracle_packet_sha256"]
    assert payload["standalone_packet_sha256"] == payload["member_sha256"]


def test_malformed_hngp_member_fails_closed(tmp_path: Path) -> None:
    proof_tool = _load_tool(PROOF_PATH, "prove_hnerv_generated_schema_runtime_packet_bad_hngp")
    archive = tmp_path / "bad-hngp.zip"
    _write_deterministic_zip(archive, "bad_hngp.hngp", b"BAD-HNGP")

    payload = proof_tool.build_hnerv_generated_schema_runtime_packet_proof(
        candidate_archive=archive,
        command_text="prove malformed hngp",
    )

    assert payload["ready_for_exact_eval_runtime"] is False
    assert payload["score_claim"] is False
    assert any(blocker.startswith("oracle_parse_failed:") for blocker in payload["blockers"])
    assert any(
        blocker.startswith("standalone_runtime_parse_failed:")
        for blocker in payload["blockers"]
    )
    assert "oracle_sections_missing" in payload["blockers"]
    assert payload["consumed_sections"] == []


def test_cli_writes_runtime_packet_proof_json(tmp_path: Path) -> None:
    proof_tool = _load_tool(PROOF_PATH, "prove_hnerv_generated_schema_runtime_packet_cli")
    archive = _build_candidate(tmp_path, candidate_id="unit_runtime_proof_cli")
    out = tmp_path / "proof.json"

    rc = proof_tool.main(
        [
            "--candidate-archive",
            str(archive),
            "--command-text",
            "prove cli runtime packet",
            "--json-out",
            str(out),
            "--fail-if-not-ready",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ready_for_exact_eval_runtime"] is True
    assert payload["blockers"] == []
