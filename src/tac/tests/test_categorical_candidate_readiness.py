from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tac.categorical_candidate_readiness import (
    REQUIRED_CONTROL_NAMES,
    audit_categorical_candidate_manifest,
)
from tac.repo_io import sha256_bytes, sha256_file, write_json
from tac.semantic_label_contract import CONTEST_SEGNET_CLASS_NAME_TUPLE, SELFCOMP_CLASS_TO_GRAY

REPO = Path(__file__).resolve().parents[3]


def _base_candidate(tmp_path: Path) -> dict:
    archive = tmp_path / "candidate.zip"
    payload = b"\x00\x01\x01\x02categorical"
    decoder = b"#!/usr/bin/env python3\n"
    inflate = b"#!/usr/bin/env bash\nset -euo pipefail\n"
    codebook = b"{\"class_order\":\"contest\"}\n"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("inflate.sh", inflate)
        zf.writestr("categorical_payload.bin", payload)
        zf.writestr("runtime_decoder.py", decoder)
        zf.writestr("class_codebook.json", codebook)

    return {
        "source_archive_sha256": "a" * 64,
        "archive_member_manifest_sha256": "b" * 64,
        "candidate_archive_contract": "contest_archive_zip",
        "candidate_archive": {
            "path": archive.as_posix(),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        },
        "semantic_class_order": list(CONTEST_SEGNET_CLASS_NAME_TUPLE),
        "selfcomp_gray_codebook": [
            SELFCOMP_CLASS_TO_GRAY[index] for index in range(len(SELFCOMP_CLASS_TO_GRAY))
        ],
        "runtime_consumer": {
            "path": "src/tac/qma9_range_mask_contract.py",
            "consumes_charged_members": True,
        },
        "charged_members": [
            {
                "name": "categorical_payload.bin",
                "role": "categorical_payload",
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            },
            {
                "name": "inflate.sh",
                "role": "decoder_or_runtime_consumer",
                "bytes": len(inflate),
                "sha256": sha256_bytes(inflate),
            },
            {
                "name": "runtime_decoder.py",
                "role": "decoder_table",
                "bytes": len(decoder),
                "sha256": sha256_bytes(decoder),
            },
            {
                "name": "class_codebook.json",
                "role": "decoder_table",
                "bytes": len(codebook),
                "sha256": sha256_bytes(codebook),
            },
        ],
        "no_op_controls": {name: {"passed": True} for name in REQUIRED_CONTROL_NAMES},
    }


def test_audit_categorical_candidate_manifest_accepts_byte_closed_fixture(tmp_path: Path) -> None:
    candidate = _base_candidate(tmp_path)

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)

    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is True
    assert manifest["promotion_eligible"] is False
    assert manifest["evidence_grade"] == "archive_readiness_audit"
    assert manifest["dispatch_blockers"] == []
    assert manifest["candidate_archive"]["contract"] == "contest_archive_zip"
    assert manifest["candidate_archive"]["contains_inflate_sh"] is True
    assert manifest["semantic_contract"]["matches_candidate"] is True
    assert manifest["runtime_consumer"]["consumes_charged_members"] is True
    assert manifest["charged_member_summary"]["roles"]["categorical_payload"] == 1
    assert manifest["charged_member_summary"]["roles"]["decoder_or_runtime_consumer"] == 1


def test_audit_categorical_candidate_manifest_fails_closed_on_label_and_control_drift(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    candidate["semantic_class_order"] = list(reversed(candidate["semantic_class_order"]))
    candidate["no_op_controls"]["label_permutation_fail_closed_control"] = {"passed": False}
    candidate["runtime_consumer"]["consumes_charged_members"] = False

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)
    blockers = set(manifest["dispatch_blockers"])

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["semantic_contract"]["matches_candidate"] is False
    assert "semantic_class_order_mismatch" in blockers
    assert "no_op_control_not_passed:label_permutation_fail_closed_control" in blockers
    assert "runtime_consumer_does_not_declare_charged_member_use" in blockers


def test_audit_categorical_candidate_manifest_checks_archive_member_fidelity(tmp_path: Path) -> None:
    candidate = _base_candidate(tmp_path)
    candidate["charged_members"][0]["sha256"] = "f" * 64

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert (
        "charged_member_archive_sha256_mismatch:categorical_payload.bin"
        in manifest["dispatch_blockers"]
    )


def test_audit_categorical_candidate_manifest_rejects_local_sidecar_runtime(tmp_path: Path) -> None:
    candidate = _base_candidate(tmp_path)
    outside_runtime = tmp_path / "runtime.py"
    outside_runtime.write_text("# sidecar runtime\n", encoding="utf-8")
    candidate["runtime_consumer"]["path"] = outside_runtime.as_posix()

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "runtime_consumer_path_outside_repo" in manifest["dispatch_blockers"]


def test_audit_categorical_candidate_manifest_requires_contest_archive_shape(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    candidate.pop("candidate_archive_contract")
    with zipfile.ZipFile(candidate["candidate_archive"]["path"], "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("categorical_payload.bin", b"\x00\x01")
    archive = Path(candidate["candidate_archive"]["path"])
    candidate["candidate_archive"]["bytes"] = archive.stat().st_size
    candidate["candidate_archive"]["sha256"] = sha256_file(archive)
    candidate["charged_members"] = [
        {
            "name": "categorical_payload.bin",
            "role": "categorical_payload",
            "bytes": 2,
            "sha256": sha256_bytes(b"\x00\x01"),
        }
    ]

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)
    blockers = set(manifest["dispatch_blockers"])

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "candidate_archive_contract_not_contest_archive_zip" in blockers
    assert "candidate_archive_missing_inflate_sh" in blockers


def test_audit_categorical_candidate_readiness_cli_records_tool_manifest(tmp_path: Path) -> None:
    candidate = _base_candidate(tmp_path)
    candidate_path = tmp_path / "candidate.json"
    out = tmp_path / "readiness.json"
    write_json(candidate_path, candidate)

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_categorical_candidate_readiness.py"),
            "--candidate-json",
            str(candidate_path),
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    manifest = json.loads(out.read_text(encoding="utf-8"))
    tool_run = manifest["tool_run_manifest"]
    assert manifest["kind"] == "categorical_candidate_readiness"
    assert manifest["ready_for_exact_eval_dispatch"] is True
    assert tool_run["tool"] == "tools/audit_categorical_candidate_readiness.py"
    assert tool_run["score_claim"] is False
    assert len(tool_run["input_files"]) == 2
