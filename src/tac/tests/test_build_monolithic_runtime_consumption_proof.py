# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_HELPER_PATH = Path(__file__).resolve().parents[3] / "tools" / "build_monolithic_runtime_consumption_proof.py"
spec = importlib.util.spec_from_file_location("build_monolithic_runtime_consumption_proof", _HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
sys.modules["build_monolithic_runtime_consumption_proof"] = helper
spec.loader.exec_module(helper)


def _candidate_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    payload = {
        "schema": "tac_monolithic_packet_candidate_v1",
        "candidate_id": "unit-candidate",
        "score_claim": False,
        "candidate_archive": {
            "path": "candidate.zip",
            "bytes": 123,
            "sha256": "a" * 64,
        },
        "monolithic_layout": {
            "new_member_sha256": "b" * 64,
        },
        "replacements": [
            {
                "section_name": "decoder_packed_brotli",
                "new_sha256": "c" * 64,
            },
            {
                "section_name": "latents_and_sidecar_brotli",
                "new_sha256": "d" * 64,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_monolithic_runtime_consumption_proof_ready(tmp_path: Path) -> None:
    manifest = _candidate_manifest(tmp_path)
    log = tmp_path / "runtime.log"
    log.write_text(
        "runtime consumed archive "
        + "a" * 64
        + " member "
        + "b" * 64
        + " decoder "
        + "c" * 64
        + " tail "
        + "d" * 64,
        encoding="utf-8",
    )

    payload = helper.build_runtime_consumption_proof(
        candidate_manifest_path=manifest,
        command_text="inflate.sh candidate.zip",
        runtime_log=log,
    )

    assert payload["schema"] == "tac_runtime_consumption_proof_v1"
    assert payload["ready_for_exact_eval_runtime"] is True
    assert payload["blockers"] == []
    assert payload["candidate_archive_sha256"] == "a" * 64
    assert payload["rebuilt_member_sha256"] == "b" * 64
    assert payload["new_member_sha256"] == "b" * 64
    assert len(payload["command_sha256"]) == 64
    assert len(payload["log_sha256"]) == 64
    assert [row["section_name"] for row in payload["changed_sections"]] == [
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ]


def test_build_monolithic_runtime_consumption_proof_fails_closed_on_missing_tokens(
    tmp_path: Path,
) -> None:
    manifest = _candidate_manifest(tmp_path)
    log = tmp_path / "runtime.log"
    log.write_text("runtime consumed only " + "a" * 64, encoding="utf-8")

    payload = helper.build_runtime_consumption_proof(
        candidate_manifest_path=manifest,
        command_text="inflate.sh candidate.zip",
        runtime_log=log,
    )

    assert payload["ready_for_exact_eval_runtime"] is False
    assert "runtime_log_missing_new_member_sha256" in payload["blockers"]
    assert "runtime_log_missing_changed_section_sha:decoder_packed_brotli" in payload["blockers"]
    assert "runtime_log_missing_changed_section_sha:latents_and_sidecar_brotli" in payload["blockers"]


def test_build_monolithic_runtime_consumption_proof_cli_writes_json(tmp_path: Path) -> None:
    manifest = _candidate_manifest(tmp_path)
    log = tmp_path / "runtime.log"
    out = tmp_path / "proof.json"
    log.write_text(" ".join(["a" * 64, "b" * 64, "c" * 64, "d" * 64]), encoding="utf-8")

    rc = helper.main(
        [
            "--candidate-manifest",
            str(manifest),
            "--command-text",
            "inflate.sh candidate.zip",
            "--runtime-log",
            str(log),
            "--json-out",
            str(out),
            "--fail-if-not-ready",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ready_for_exact_eval_runtime"] is True
