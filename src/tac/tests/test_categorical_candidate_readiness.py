from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tac.categorical_candidate_plan import (
    CATEGORICAL_CONSTRUCTION_PLAN_CONTRACT,
    build_categorical_charged_label_plan,
)
from tac.categorical_candidate_readiness import (
    ARCHIVE_MEMBER_MANIFEST_CONTRACT,
    CANDIDATE_MANIFEST_CONTRACT,
    DECODE_REENCODE_PARITY_CONTRACT,
    DETERMINISTIC_ZIP_CREATE_SYSTEM,
    DETERMINISTIC_ZIP_DATE_TIME,
    DETERMINISTIC_ZIP_FILE_MODE,
    DETERMINISTIC_ZIP_INFLATE_MODE,
    HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
    REQUIRED_CONTROL_NAMES,
    RUNTIME_LOADER_PARITY_CONTRACT,
    audit_categorical_candidate_manifest,
)
from tac.repo_io import sha256_bytes, sha256_file, write_json
from tac.semantic_label_contract import CONTEST_SEGNET_CLASS_NAME_TUPLE, SELFCOMP_CLASS_TO_GRAY

REPO = Path(__file__).resolve().parents[3]
RUNTIME_CONSUMER_REPO_PATH = "src/tac/qma9_range_mask_contract.py"


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=DETERMINISTIC_ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_STORED
    mode = DETERMINISTIC_ZIP_INFLATE_MODE if name == "inflate.sh" else DETERMINISTIC_ZIP_FILE_MODE
    info.external_attr = mode << 16
    info.create_system = DETERMINISTIC_ZIP_CREATE_SYSTEM
    return info


def _base_candidate(tmp_path: Path) -> dict:
    archive = tmp_path / "candidate.zip"
    runtime_source = REPO / RUNTIME_CONSUMER_REPO_PATH
    runtime_source_raw = runtime_source.read_bytes()
    runtime_source_sha = sha256_bytes(runtime_source_raw)
    member_payloads = [
        ("categorical_payload.bin", b"\x00\x01\x01\x02categorical", "categorical_payload"),
        ("class_codebook.json", b"{\"class_order\":\"contest\"}\n", "decoder_table"),
        ("inflate.sh", b"#!/usr/bin/env bash\nset -euo pipefail\n", "inflate_entrypoint"),
        ("runtime_decoder.py", runtime_source_raw, "decoder_or_runtime_consumer"),
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
    payload_sha = charged_members[0]["sha256"]
    archive_member_manifest = {
        "schema_version": 1,
        "kind": "categorical_test_archive_member_manifest",
        "archive_member_manifest_contract": ARCHIVE_MEMBER_MANIFEST_CONTRACT,
        "member_count": len(charged_members),
        "member_order": [record["name"] for record in charged_members],
        "members": charged_members,
    }
    archive_member_manifest_path = tmp_path / "archive_member_manifest.json"
    write_json(archive_member_manifest_path, archive_member_manifest)

    return {
        "schema_version": 1,
        "kind": "categorical_candidate_manifest",
        "candidate_manifest_contract": CANDIDATE_MANIFEST_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
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
            "path": RUNTIME_CONSUMER_REPO_PATH,
            "consumes_charged_members": True,
        },
        "runtime_loader_parity": {
            "schema_version": 1,
            "runtime_loader_parity_contract": RUNTIME_LOADER_PARITY_CONTRACT,
            "passed": True,
            "score_claim": False,
            "dispatch_attempted": False,
            "runtime_consumer_path": RUNTIME_CONSUMER_REPO_PATH,
            "runtime_consumer_sha256": runtime_source_sha,
            "loader_member": "runtime_decoder.py",
            "loader_member_sha256": runtime_source_sha,
            "byte_identical_to_runtime_consumer": True,
            "sidecar_free": True,
            "fallback_used": False,
            "loaded_charged_members": [
                "categorical_payload.bin",
                "class_codebook.json",
            ],
        },
        "decode_reencode_parity": {
            "schema_version": 1,
            "decode_reencode_parity_contract": DECODE_REENCODE_PARITY_CONTRACT,
            "passed": True,
            "score_claim": False,
            "dispatch_attempted": False,
            "payload_member": "categorical_payload.bin",
            "payload_member_sha256": payload_sha,
            "full_decode": {
                "passed": True,
                "frame_count": 1,
                "decoded_masks_sha256": "b" * 64,
            },
            "byte_exact_reencode": {
                "passed": True,
                "byte_exact": True,
                "reencoded_payload_sha256": payload_sha,
            },
            "sidecar_free": True,
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


def _structural_hpm1_inventory(payload_sha: str) -> dict:
    return {
        "schema_version": 1,
        "kind": "hpm1_payload_structural_decode_inventory",
        "hpm1_structural_decode_inventory_contract": HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "segment": {"sha256": payload_sha},
        "structural_reencode": {
            "matches_source_segment": True,
            "reencoded_segment_sha256": payload_sha,
            "not_semantic_decode_reencode_parity": True,
        },
        "full_decode": {"passed": False},
        "byte_exact_semantic_reencode": {"passed": False},
        "unsupported_wire_constructs": [
            {"name": "hpac_autoregressive_probability_rows"}
        ],
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
    assert manifest["candidate_manifest"]["contract"] == CANDIDATE_MANIFEST_CONTRACT
    assert manifest["candidate_archive"]["contract"] == "contest_archive_zip"
    assert manifest["candidate_archive"]["contains_inflate_sh"] is True
    assert manifest["candidate_archive"]["member_order_matches_manifest"] is True
    assert manifest["candidate_archive"]["zip_determinism_contract"]["passed"] is True
    assert manifest["archive_member_manifest"]["schema_valid"] is True
    assert (
        manifest["archive_member_manifest"]["member_order_matches_charged_members"] is True
    )
    assert (
        manifest["archive_member_manifest"]["member_count_matches_charged_members"] is True
    )
    assert manifest["semantic_contract"]["matches_candidate"] is True
    assert manifest["runtime_consumer"]["consumes_charged_members"] is True
    assert manifest["runtime_loader_parity"]["accepted"] is True
    assert manifest["runtime_loader_parity"]["loader_member"] == "runtime_decoder.py"
    assert (
        manifest["runtime_loader_parity"]["contract"]
        == RUNTIME_LOADER_PARITY_CONTRACT
    )
    assert manifest["conditioning_prior_contract"]["passed"] is True
    assert manifest["decode_reencode_parity"]["accepted"] is True
    assert manifest["decode_reencode_parity"]["payload_member"] == "categorical_payload.bin"
    assert manifest["conditioning_prior_contract"]["runtime_consumed_count"] == 1
    assert manifest["conditioning_prior_contract"]["compression_time_only_count"] == 1
    assert manifest["charged_member_summary"]["roles"]["categorical_payload"] == 1
    assert manifest["charged_member_summary"]["roles"]["decoder_or_runtime_consumer"] == 1
    assert manifest["candidate_construction_plan"]["declared"] is False
    assert manifest["hpm1_structural_decode_inventory"]["declared"] is False


def test_audit_categorical_candidate_manifest_accepts_structural_hpm1_inventory(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    payload_sha = candidate["decode_reencode_parity"]["payload_member_sha256"]
    inventory_path = tmp_path / "hpm1_structural_inventory.json"
    write_json(inventory_path, _structural_hpm1_inventory(payload_sha))
    candidate["hpm1_structural_decode_inventory"] = {
        "path": inventory_path.name,
        "bytes": inventory_path.stat().st_size,
        "sha256": sha256_file(inventory_path),
        "contract": HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
        "payload_member": "categorical_payload.bin",
        "payload_member_sha256": payload_sha,
        "full_decode_proven": False,
        "byte_exact_semantic_reencode_proven": False,
    }

    manifest = audit_categorical_candidate_manifest(
        candidate,
        repo_root=REPO,
        manifest_dir=tmp_path,
    )

    assert manifest["ready_for_exact_eval_dispatch"] is True
    assert manifest["hpm1_structural_decode_inventory"]["accepted"] is True
    assert manifest["hpm1_structural_decode_inventory"][
        "structural_reencode_matches_source"
    ] is True
    assert manifest["hpm1_structural_decode_inventory"][
        "unsupported_wire_constructs"
    ] == ["hpac_autoregressive_probability_rows"]


def test_audit_categorical_candidate_manifest_rejects_unsafe_structural_hpm1_inventory_path(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    payload_sha = candidate["decode_reencode_parity"]["payload_member_sha256"]
    inventory_path = tmp_path / "hpm1_structural_inventory.json"
    write_json(inventory_path, _structural_hpm1_inventory(payload_sha))
    candidate["hpm1_structural_decode_inventory"] = {
        "path": inventory_path.as_posix(),
        "bytes": inventory_path.stat().st_size,
        "sha256": sha256_file(inventory_path),
        "contract": HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
        "payload_member": "categorical_payload.bin",
        "payload_member_sha256": payload_sha,
        "full_decode_proven": False,
        "byte_exact_semantic_reencode_proven": False,
    }

    manifest = audit_categorical_candidate_manifest(
        candidate,
        repo_root=REPO,
        manifest_dir=tmp_path,
    )

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert (
        "hpm1_structural_inventory_path_unsafe"
        in manifest["dispatch_blockers"]
    )


def test_audit_categorical_candidate_manifest_rejects_traversal_structural_hpm1_inventory_path(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    payload_sha = candidate["decode_reencode_parity"]["payload_member_sha256"]
    escaped_inventory_path = tmp_path.parent / f"{tmp_path.name}_escape_hpm1.json"
    write_json(escaped_inventory_path, _structural_hpm1_inventory(payload_sha))
    candidate["hpm1_structural_decode_inventory"] = {
        "path": f"../{escaped_inventory_path.name}",
        "bytes": escaped_inventory_path.stat().st_size,
        "sha256": sha256_file(escaped_inventory_path),
        "contract": HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
        "payload_member": "categorical_payload.bin",
        "payload_member_sha256": payload_sha,
        "full_decode_proven": False,
        "byte_exact_semantic_reencode_proven": False,
    }

    manifest = audit_categorical_candidate_manifest(
        candidate,
        repo_root=REPO,
        manifest_dir=tmp_path,
    )

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert (
        "hpm1_structural_inventory_path_unsafe"
        in manifest["dispatch_blockers"]
    )


def test_audit_categorical_candidate_manifest_rejects_bad_structural_hpm1_inventory(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    candidate["hpm1_structural_decode_inventory"] = {
        "path": "missing_hpm1_structural_inventory.json",
        "bytes": 123,
        "sha256": "0" * 64,
        "contract": HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
        "payload_member": "categorical_payload.bin",
        "payload_member_sha256": candidate["decode_reencode_parity"][
            "payload_member_sha256"
        ],
        "full_decode_proven": False,
        "byte_exact_semantic_reencode_proven": False,
    }

    manifest = audit_categorical_candidate_manifest(
        candidate,
        repo_root=REPO,
        manifest_dir=tmp_path,
    )

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert (
        "hpm1_structural_inventory_path_missing"
        in manifest["dispatch_blockers"]
    )


def test_categorical_charged_label_plan_grounds_classes_and_stays_non_dispatchable(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    plan = build_categorical_charged_label_plan(
        source_archive_sha256=candidate["source_archive_sha256"],
        charged_members=candidate["charged_members"],
        conditioning_priors=candidate["conditioning_priors"],
        candidate_archive_sha256=candidate["candidate_archive"]["sha256"],
        archive_member_manifest_sha256=candidate["archive_member_manifest_sha256"],
    )
    candidate["candidate_construction_plan"] = plan

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)
    construction = manifest["candidate_construction_plan"]

    assert manifest["ready_for_exact_eval_dispatch"] is True
    assert construction["accepted"] is True
    assert construction["contract"] == CATEGORICAL_CONSTRUCTION_PLAN_CONTRACT
    assert construction["ready_for_exact_eval_dispatch"] is False
    assert construction["candidate_construction_ready"] is True
    assert "real_byte_closed_archive_parity_missing" in construction["planning_dispatch_blockers"]
    assert construction["byte_closed_archive_parity"]["proven"] is False
    rows = construction["class_rows"]
    assert [row["name"] for row in rows] == list(CONTEST_SEGNET_CLASS_NAME_TUPLE)
    assert rows[0]["charged_label_member"] == "class_codebook.json"
    assert rows[0]["categorical_payload_member"] == "categorical_payload.bin"
    assert rows[0]["default_quant_bits"] == 8
    assert rows[1]["openpilot_prior_hint"] == "lane_marking_track_prior"
    assert construction["conditioning_prior_contract"]["passed"] is True


def test_categorical_construction_plan_cannot_claim_dispatch_readiness(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    plan = build_categorical_charged_label_plan(
        source_archive_sha256=candidate["source_archive_sha256"],
        charged_members=candidate["charged_members"],
        conditioning_priors=candidate["conditioning_priors"],
    )
    plan["ready_for_exact_eval_dispatch"] = True
    candidate["candidate_construction_plan"] = plan

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["candidate_construction_plan"]["accepted"] is False
    assert (
        "candidate_construction_plan_ready_for_exact_eval_dispatch_must_be_false"
        in manifest["dispatch_blockers"]
    )


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
            "archive_member_manifest_contract": ARCHIVE_MEMBER_MANIFEST_CONTRACT,
            "member_count": len(candidate["charged_members"]) - 1,
            "member_order": [record["name"] for record in candidate["charged_members"][1:]],
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


def test_audit_categorical_candidate_manifest_requires_member_order_and_count(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    bad_manifest_path = tmp_path / "bad_archive_member_manifest.json"
    write_json(
        bad_manifest_path,
        {
            "schema_version": 1,
            "kind": "categorical_test_archive_member_manifest",
            "archive_member_manifest_contract": ARCHIVE_MEMBER_MANIFEST_CONTRACT,
            "member_count": len(candidate["charged_members"]) + 1,
            "member_order": list(reversed([record["name"] for record in candidate["charged_members"]])),
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
    assert "archive_member_manifest_member_order_mismatch" in blockers
    assert "archive_member_manifest_member_count_mismatch" in blockers


def test_audit_categorical_candidate_manifest_requires_schema_and_manifest_kind(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    candidate.pop("schema_version")
    candidate.pop("candidate_manifest_contract")
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
    assert "candidate_manifest_contract_missing_or_invalid" in blockers
    assert "archive_member_manifest_schema_version_missing_or_invalid" in blockers
    assert "archive_member_manifest_kind_missing_or_invalid" in blockers
    assert "archive_member_manifest_contract_missing_or_invalid" in blockers


def test_audit_categorical_candidate_manifest_rejects_score_claim_proxy_and_sidecar_rows(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    candidate["score_claim"] = True
    candidate["dispatch_attempted"] = True
    candidate["candidate_rows"] = [
        {
            "row_id": "proxy_probe",
            "score_claim": False,
            "dispatch_attempted": False,
            "evidence_grade": "proxy",
            "sidecar": True,
        }
    ]
    candidate["evidence_rows"] = [
        {
            "row_id": "charged_sidecar_debug",
            "role": "debug_sidecar",
            "score_claim": False,
        }
    ]

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)
    blockers = set(manifest["dispatch_blockers"])

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["candidate_manifest"]["schema_valid"] is False
    assert "candidate_manifest_score_claim_must_be_false" in blockers
    assert "candidate_manifest_dispatch_attempted_must_be_false" in blockers
    assert "candidate_rows_proxy_marker:0" in blockers
    assert "candidate_rows_sidecar_marker:0" in blockers
    assert "evidence_rows_sidecar_marker:0" in blockers


def test_audit_categorical_candidate_manifest_rejects_sidecar_archive_member_record(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    candidate["charged_members"][1]["role"] = "debug_sidecar"
    candidate["charged_members"][1]["sidecar"] = True

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)
    blockers = set(manifest["dispatch_blockers"])

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "charged_member_1_sidecar_role_forbidden" in blockers
    assert "charged_member_1_sidecar_flag_forbidden" in blockers


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


def test_audit_categorical_candidate_manifest_requires_runtime_loader_parity(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    candidate.pop("runtime_loader_parity")

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["runtime_loader_parity"]["declared"] is False
    assert "runtime_loader_parity_missing" in manifest["dispatch_blockers"]


def test_audit_categorical_candidate_manifest_requires_decode_reencode_parity(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    candidate.pop("decode_reencode_parity")

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["decode_reencode_parity"]["declared"] is False
    assert "decode_reencode_parity_missing" in manifest["dispatch_blockers"]


def test_audit_categorical_candidate_manifest_rejects_loader_source_mismatch(
    tmp_path: Path,
) -> None:
    candidate = _base_candidate(tmp_path)
    candidate["runtime_loader_parity"]["loader_member_sha256"] = "e" * 64
    candidate["runtime_loader_parity"]["byte_identical_to_runtime_consumer"] = False
    candidate["runtime_loader_parity"]["sidecar_free"] = False

    manifest = audit_categorical_candidate_manifest(candidate, repo_root=REPO)
    blockers = set(manifest["dispatch_blockers"])

    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["runtime_loader_parity"]["accepted"] is False
    assert "runtime_loader_parity_loader_member_sha256_mismatch" in blockers
    assert "runtime_loader_parity_not_byte_identical" in blockers
    assert "runtime_loader_parity_sidecar_free_not_proven" in blockers


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
