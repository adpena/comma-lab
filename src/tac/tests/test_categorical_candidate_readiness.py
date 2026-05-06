from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tac.categorical_candidate_readiness import (
    DETERMINISTIC_ZIP_CREATE_SYSTEM,
    DETERMINISTIC_ZIP_DATE_TIME,
    DETERMINISTIC_ZIP_FILE_MODE,
    DETERMINISTIC_ZIP_INFLATE_MODE,
    REQUIRED_CONTROL_NAMES,
    audit_categorical_candidate_manifest,
)
from tac.repo_io import sha256_bytes, sha256_file, write_json
from tac.semantic_label_contract import CONTEST_SEGNET_CLASS_NAME_TUPLE, SELFCOMP_CLASS_TO_GRAY

REPO = Path(__file__).resolve().parents[3]


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=DETERMINISTIC_ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_STORED
    mode = DETERMINISTIC_ZIP_INFLATE_MODE if name == "inflate.sh" else DETERMINISTIC_ZIP_FILE_MODE
    info.external_attr = mode << 16
    info.create_system = DETERMINISTIC_ZIP_CREATE_SYSTEM
    return info


def _base_candidate(tmp_path: Path) -> dict:
    archive = tmp_path / "candidate.zip"
    member_payloads = [
        ("categorical_payload.bin", b"\x00\x01\x01\x02categorical", "categorical_payload"),
        ("class_codebook.json", b"{\"class_order\":\"contest\"}\n", "decoder_table"),
        ("inflate.sh", b"#!/usr/bin/env bash\nset -euo pipefail\n", "decoder_or_runtime_consumer"),
        ("runtime_decoder.py", b"#!/usr/bin/env python3\n", "decoder_table"),
    ]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, raw, _role in member_payloads:
            zf.writestr(_zip_info(name), raw, compress_type=zipfile.ZIP_STORED)
    charged_members = [
        {
            "name": name,
            "role": role,
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        }
        for name, raw, role in member_payloads
    ]
    archive_member_manifest = {
        "schema_version": 1,
        "kind": "categorical_test_archive_member_manifest",
        "members": charged_members,
    }
    archive_member_manifest_path = tmp_path / "archive_member_manifest.json"
    write_json(archive_member_manifest_path, archive_member_manifest)

    return {
        "schema_version": 1,
        "kind": "categorical_candidate_manifest",
        "source_archive_sha256": "a" * 64,
        "archive_member_manifest_sha256": sha256_file(archive_member_manifest_path),
        "archive_member_manifest": {
            "path": archive_member_manifest_path.as_posix(),
            "bytes": archive_member_manifest_path.stat().st_size,
            "sha256": sha256_file(archive_member_manifest_path),
        },
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
        "conditioning_priors": [
            {
                "family": "openpilot_priors",
                "name": "ego_lane_atom_ranker",
                "usage": "compression_time_atom_ranking_only",
                "runtime_consumed": False,
            },
            {
                "family": "clade_spade",
                "name": "fixture_class_conditioning",
                "usage": "inflate_runtime_conditioning",
                "runtime_consumed": True,
                "charged_member": "class_codebook.json",
            },
        ],
        "charged_members": charged_members,
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
    assert manifest["candidate_manifest"]["schema_valid"] is True
    assert manifest["candidate_archive"]["contract"] == "contest_archive_zip"
    assert manifest["candidate_archive"]["contains_inflate_sh"] is True
    assert manifest["candidate_archive"]["member_order_matches_manifest"] is True
    assert manifest["candidate_archive"]["zip_determinism_contract"]["passed"] is True
    assert manifest["archive_member_manifest"]["schema_valid"] is True
    assert manifest["semantic_contract"]["matches_candidate"] is True
    assert manifest["runtime_consumer"]["consumes_charged_members"] is True
    assert manifest["conditioning_prior_contract"]["passed"] is True
    assert manifest["conditioning_prior_contract"]["runtime_consumed_count"] == 1
    assert manifest["conditioning_prior_contract"]["compression_time_only_count"] == 1
    assert manifest["charged_member_summary"]["roles"]["categorical_payload"] == 1
    assert manifest["charged_member_summary"]["roles"]["decoder_or_runtime_consumer"] == 1


def test_audit_categorical_candidate_manifest_rejects_uncharged_runtime_openpilot_prior(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    candidate["conditioning_priors"] = [
        {
            "family": "openpilot_priors",
            "name": "supercombo_lane_features",
            "usage": "inflate_runtime_conditioning",
            "runtime_consumed": True,
        }
    ]

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)
    blockers = set(manifest["dispatch_blockers"])

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["conditioning_prior_contract"]["passed"] is False
    assert (
        "conditioning_prior_charged_member_missing_or_unsafe:"
        "openpilot_priors:supercombo_lane_features"
    ) in blockers


def test_audit_categorical_candidate_manifest_rejects_undeclared_clade_spade_member(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    candidate["conditioning_priors"] = [
        {
            "family": "clade_spade",
            "name": "class_affine_table",
            "usage": "inflate_runtime_conditioning",
            "runtime_consumed": True,
            "charged_member": "missing_clade_table.bin",
        }
    ]

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert (
        "conditioning_prior_charged_member_not_declared:"
        "clade_spade:class_affine_table:missing_clade_table.bin"
    ) in manifest["dispatch_blockers"]


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


def test_audit_categorical_candidate_manifest_rejects_central_local_name_mismatch(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    archive = Path(candidate["candidate_archive"]["path"])
    raw = bytearray(archive.read_bytes())
    with zipfile.ZipFile(archive) as zf:
        offset = zf.getinfo("categorical_payload.bin").header_offset
    # Mutate only the local header name. The central directory still points to
    # categorical_payload.bin, so strict archive custody must fail closed.
    name_start = offset + 30
    raw[name_start] = ord("x")
    archive.write_bytes(raw)
    candidate["candidate_archive"]["bytes"] = archive.stat().st_size
    candidate["candidate_archive"]["sha256"] = sha256_file(archive)

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)
    wire = manifest["candidate_archive"]["zip_wire_contract"]

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert wire["passed"] is False
    assert wire["central_local_name_mismatches"]
    assert "candidate_archive_not_readable_zip" in manifest["dispatch_blockers"]
    assert "candidate_archive_zip_wire_contract_failed" in manifest["dispatch_blockers"]


def test_audit_categorical_candidate_manifest_rejects_duplicate_archive_members(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    archive = Path(candidate["candidate_archive"]["path"])
    with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("categorical_payload.bin", b"duplicate")
    candidate["candidate_archive"]["bytes"] = archive.stat().st_size
    candidate["candidate_archive"]["sha256"] = sha256_file(archive)

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "candidate_archive_duplicate_member_names" in manifest["dispatch_blockers"]
    assert "candidate_archive_zip_wire_contract_failed" in manifest["dispatch_blockers"]


def test_audit_categorical_candidate_manifest_rejects_untracked_archive_members(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    archive = Path(candidate["candidate_archive"]["path"])
    with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("debug_sidecar.json", b"{}")
    candidate["candidate_archive"]["bytes"] = archive.stat().st_size
    candidate["candidate_archive"]["sha256"] = sha256_file(archive)

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "candidate_archive_untracked_members" in manifest["dispatch_blockers"]
    assert manifest["candidate_archive"]["untracked_members"] == ["debug_sidecar.json"]


def test_audit_categorical_candidate_manifest_checks_member_manifest_content(tmp_path: Path) -> None:
    candidate = _base_candidate(tmp_path)
    bad_manifest_path = tmp_path / "bad_archive_member_manifest.json"
    write_json(
        bad_manifest_path,
        {
            "schema_version": 1,
            "kind": "categorical_test_archive_member_manifest",
            "members": candidate["charged_members"][1:],
        },
    )
    candidate["archive_member_manifest"] = {
        "path": bad_manifest_path.as_posix(),
        "bytes": bad_manifest_path.stat().st_size,
        "sha256": sha256_file(bad_manifest_path),
    }
    candidate["archive_member_manifest_sha256"] = sha256_file(bad_manifest_path)

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "archive_member_manifest_members_mismatch" in manifest["dispatch_blockers"]


def test_audit_categorical_candidate_manifest_requires_schema_and_manifest_kind(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    candidate.pop("schema_version")
    candidate["kind"] = "categorical_but_not_a_candidate_manifest"
    bad_manifest_path = tmp_path / "bad_archive_member_manifest.json"
    write_json(
        bad_manifest_path,
        {
            "kind": "not_categorical_archive_manifest",
            "members": candidate["charged_members"],
        },
    )
    candidate["archive_member_manifest"] = {
        "path": bad_manifest_path.as_posix(),
        "bytes": bad_manifest_path.stat().st_size,
        "sha256": sha256_file(bad_manifest_path),
    }
    candidate["archive_member_manifest_sha256"] = sha256_file(bad_manifest_path)

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)
    blockers = set(manifest["dispatch_blockers"])

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["candidate_manifest"]["schema_valid"] is False
    assert manifest["archive_member_manifest"]["schema_valid"] is False
    assert "candidate_manifest_schema_version_missing_or_invalid" in blockers
    assert "candidate_manifest_kind_missing_or_invalid" in blockers
    assert "archive_member_manifest_schema_version_missing_or_invalid" in blockers
    assert "archive_member_manifest_kind_missing_or_invalid" in blockers


def test_audit_categorical_candidate_manifest_rejects_nondeterministic_zip_metadata(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    archive = Path(candidate["candidate_archive"]["path"])
    # Preserve charged-member bytes/hashes, but keep default ZIP metadata and a
    # reversed member order from writestr(name, ...).
    payloads = {
        "categorical_payload.bin": b"\x00\x01\x01\x02categorical",
        "class_codebook.json": b"{\"class_order\":\"contest\"}\n",
        "inflate.sh": b"#!/usr/bin/env bash\nset -euo pipefail\n",
        "runtime_decoder.py": b"#!/usr/bin/env python3\n",
    }
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in reversed([record["name"] for record in candidate["charged_members"]]):
            zf.writestr(name, payloads[name])
    candidate["candidate_archive"]["bytes"] = archive.stat().st_size
    candidate["candidate_archive"]["sha256"] = sha256_file(archive)

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)
    blockers = set(manifest["dispatch_blockers"])
    deterministic = manifest["candidate_archive"]["zip_determinism_contract"]

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["candidate_archive"]["member_order_matches_manifest"] is False
    assert deterministic["passed"] is False
    assert deterministic["bad_timestamps"]
    assert deterministic["bad_external_attr_modes"]
    assert "candidate_archive_zip_determinism_contract_failed" in blockers
    assert "candidate_archive_member_order_mismatch" in blockers


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
    assert len(tool_run["input_files"]) == 3
