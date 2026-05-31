# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli
import pytest

from tac.master_gradient_pr101_operator_candidate import (
    build_pr101_pose_axis_decoder_recompression_candidate,
)
from tac.monolithic_packet_closure_gate import build_monolithic_packet_closure_gate
from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROOF_TOOL = REPO_ROOT / "tools" / "prove_monolithic_runtime_consumption.py"
CANONICAL_PROOF_BUILDER = REPO_ROOT / "tools" / "build_monolithic_runtime_consumption_proof.py"
PR101_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
OP7_MANIFEST = (
    REPO_ROOT / ".omx/research/pose_axis_operator_pr101_manifest_20260519T074500Z.json"
)


def _load_tool(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_single_member_zip(path: Path, member_name: str, payload: bytes) -> None:
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


def _candidate_manifest(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    decoder_section = brotli.compress(b"decoder-runtime-consumption-proof" * 8)
    tail_section = brotli.compress(b"latents-and-sidecar-runtime-proof" * 4)
    header = bytes([0xFF]) + len(decoder_section).to_bytes(3, "little")
    member = header + decoder_section + tail_section
    archive = tmp_path / "candidate.zip"
    _write_single_member_zip(archive, "x", member)
    archive_bytes = archive.read_bytes()
    manifest = {
        "schema": "tac_monolithic_packet_candidate_v1",
        "candidate_id": "unit-pr106-runtime-consumption",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_archive": {
            "path": str(archive),
            "bytes": len(archive_bytes),
            "sha256": _sha256(archive_bytes),
        },
        "monolithic_layout": {
            "grammar": "pr106_ff_packed_hnerv",
            "member_name": "x",
            "new_member_sha256": _sha256(member),
            "sections": [
                {
                    "name": "ff_header",
                    "role": "internal_length_header",
                    "old_offset": 0,
                    "new_offset": 0,
                    "old_len": 4,
                    "new_len": 4,
                    "old_sha256": _sha256(header),
                    "new_sha256": _sha256(header),
                    "changed": False,
                },
                {
                    "name": "decoder_packed_brotli",
                    "role": "renderer_decoder_weights",
                    "old_offset": 4,
                    "new_offset": 4,
                    "old_len": len(decoder_section) + 1,
                    "new_len": len(decoder_section),
                    "old_sha256": "1" * 64,
                    "new_sha256": _sha256(decoder_section),
                    "changed": True,
                },
                {
                    "name": "latents_and_sidecar_brotli",
                    "role": "latent_sidecar_not_separate_pose_or_mask_member",
                    "old_offset": 4 + len(decoder_section) + 1,
                    "new_offset": 4 + len(decoder_section),
                    "old_len": len(tail_section),
                    "new_len": len(tail_section),
                    "old_sha256": _sha256(tail_section),
                    "new_sha256": _sha256(tail_section),
                    "changed": False,
                },
            ],
        },
        "replacements": [
            {
                "section_name": "decoder_packed_brotli",
                "old_sha256": "1" * 64,
                "new_sha256": _sha256(decoder_section),
                "old_bytes": len(decoder_section) + 1,
                "new_bytes": len(decoder_section),
            }
        ],
        "dispatch_blockers": [
            "runtime_consumption_proof_missing",
            "active_lane_claim_missing",
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, manifest


def test_pr106_runtime_probe_emits_gate_compatible_log(tmp_path: Path) -> None:
    proof_tool = _load_tool(PROOF_TOOL, "prove_monolithic_runtime_consumption_unit")
    canonical_builder = _load_tool(
        CANONICAL_PROOF_BUILDER,
        "build_monolithic_runtime_consumption_proof_from_probe",
    )
    manifest_path, manifest = _candidate_manifest(tmp_path)
    log_path = tmp_path / "runtime.log"

    probe = proof_tool.build_monolithic_runtime_consumption_proof(
        candidate_manifest_path=manifest_path,
        runtime_log_out=log_path,
        command_text="unit runtime probe",
    )
    canonical = canonical_builder.build_runtime_consumption_proof(
        candidate_manifest_path=manifest_path,
        command_text="unit runtime probe",
        runtime_log=log_path,
    )
    gate = build_monolithic_packet_closure_gate(
        manifest,
        runtime_proof=canonical,
        dry_run=True,
        active_rate_only_floor_archive_bytes=None,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert probe["schema"] == "tac_runtime_consumption_proof_v1"
    assert probe["proof_kind"] == "tac_monolithic_runtime_consumption_probe_v1"
    assert probe["ready_for_exact_eval_runtime"] is True
    assert probe["blockers"] == []
    assert manifest["candidate_archive"]["sha256"] in log_text
    assert manifest["monolithic_layout"]["new_member_sha256"] in log_text
    assert manifest["replacements"][0]["new_sha256"] in log_text
    assert canonical["ready_for_exact_eval_runtime"] is True
    assert canonical["blockers"] == []
    assert canonical["archive_bound_candidate_contract_schema"] == (
        ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    )
    contract = canonical["archive_bound_candidate_contract"]
    assert contract["schema"] == ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    assert contract["candidate_archive"]["sha256"] == manifest["candidate_archive"]["sha256"]
    assert contract["archive_bound_candidate_ready"] is True
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["score_claim"] is False
    assert gate["runtime_blockers"] == []
    assert gate["closure_gate_passed"] is True
    assert gate["ready_for_exact_eval_dispatch"] is False


def test_pr106_runtime_probe_fails_closed_on_archive_sha_mismatch(tmp_path: Path) -> None:
    proof_tool = _load_tool(PROOF_TOOL, "prove_monolithic_runtime_consumption_bad_sha")
    manifest_path, manifest = _candidate_manifest(tmp_path)
    manifest["candidate_archive"]["sha256"] = "a" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    payload = proof_tool.build_monolithic_runtime_consumption_proof(
        candidate_manifest_path=manifest_path,
        runtime_log_out=tmp_path / "runtime.log",
        command_text="unit bad sha",
    )

    assert payload["ready_for_exact_eval_runtime"] is False
    assert payload["score_claim"] is False
    assert "candidate_archive_sha256_mismatch" in payload["blockers"]


def test_pr106_runtime_probe_cli_writes_json_and_log(tmp_path: Path) -> None:
    proof_tool = _load_tool(PROOF_TOOL, "prove_monolithic_runtime_consumption_cli")
    manifest_path, _manifest = _candidate_manifest(tmp_path)
    out = tmp_path / "proof.json"
    log = tmp_path / "runtime.log"

    rc = proof_tool.main(
        [
            "--candidate-manifest",
            str(manifest_path),
            "--runtime-log-out",
            str(log),
            "--command-text",
            "unit cli runtime probe",
            "--json-out",
            str(out),
            "--fail-if-not-ready",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["ready_for_exact_eval_runtime"] is True
    assert payload["log_sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()


def test_pr101_runtime_probe_consumes_split_brotli_decoder_candidate(tmp_path: Path) -> None:
    if not PR101_ARCHIVE.exists():
        pytest.skip("public PR101 intake archive is an ignored local custody artifact")
    if not OP7_MANIFEST.exists():
        pytest.skip("OP-7 pose-axis operator manifest is unavailable")
    proof_tool = _load_tool(PROOF_TOOL, "prove_monolithic_runtime_consumption_pr101")
    operator_manifest = json.loads(OP7_MANIFEST.read_text(encoding="utf-8"))
    candidate = build_pr101_pose_axis_decoder_recompression_candidate(
        source_archive=PR101_ARCHIVE,
        operator_manifest=operator_manifest,
        output_dir=tmp_path / "candidate",
        candidate_id="unit-pr101-runtime-consumption",
        candidate_rank=1,
        operator_manifest_path=OP7_MANIFEST,
    )
    candidate_manifest_path = Path(candidate["candidate_manifest_path"])
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    log_path = tmp_path / "runtime-pr101.log"

    proof = proof_tool.build_monolithic_runtime_consumption_proof(
        candidate_manifest_path=candidate_manifest_path,
        runtime_log_out=log_path,
        command_text="unit pr101 split-brotli runtime probe",
    )
    gate = build_monolithic_packet_closure_gate(
        candidate_manifest,
        runtime_proof=proof,
        dry_run=True,
        active_rate_only_floor_archive_bytes=None,
    )

    log_text = log_path.read_text(encoding="utf-8")
    changed = proof["changed_sections"][0]
    consumed = {
        section["section_name"]: section
        for section in proof["consumed_sections"]
    }
    assert proof["ready_for_exact_eval_runtime"] is True
    assert proof["blockers"] == []
    assert proof["runtime_grammar"] == "pr101_fixed_offset_hnerv_microcodec"
    assert changed["section_name"] == "decoder_blob"
    assert changed["runtime_consumed"] is True
    assert changed["new_sha256"] == candidate["replacement_decoder_section"]["sha256"]
    assert consumed["decoder_blob"]["split_brotli_stream_count"] == 7
    assert candidate_manifest["candidate_archive"]["sha256"] in log_text
    assert candidate_manifest["monolithic_layout"]["new_member_sha256"] in log_text
    assert candidate_manifest["replacements"][0]["new_sha256"] in log_text
    assert gate["runtime_blockers"] == []
    assert gate["closure_gate_passed"] is True
    assert gate["ready_for_exact_eval_dispatch"] is False
