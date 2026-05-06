"""Deterministic builder for local categorical payload candidates.

The builder materializes a byte-closed local archive from an existing
categorical payload plus the canonical class codebook and a charged runtime
consumer skeleton. It deliberately emits fail-closed proof skeletons instead
of claiming decode/re-encode or exact-eval readiness.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from tac.categorical_candidate_plan import (
    build_categorical_charged_label_plan,
    build_categorical_class_codebook,
)
from tac.categorical_candidate_readiness import (
    ARCHIVE_MEMBER_MANIFEST_CONTRACT,
    CANDIDATE_MANIFEST_CONTRACT,
    DECODE_REENCODE_PARITY_CONTRACT,
    DETERMINISTIC_ZIP_CREATE_SYSTEM,
    DETERMINISTIC_ZIP_DATE_TIME,
    DETERMINISTIC_ZIP_FILE_MODE,
    DETERMINISTIC_ZIP_INFLATE_MODE,
    RUNTIME_LOADER_PARITY_CONTRACT,
    audit_categorical_candidate_manifest,
)
from tac.pr85_bundle import Pr85BundleError, parse_pr85_bundle
from tac.pr91_hpm1_codec import (
    DEFAULT_PR91_ARCHIVE,
    EXPECTED_PR91_HPM1_MASK_BYTES,
    EXPECTED_PR91_HPM1_MASK_SHA256,
    EXPECTED_PR91_MEMBER_X_SHA256,
)
from tac.repo_io import json_text, repo_relative, sha256_bytes, sha256_file, write_json
from tac.semantic_label_contract import CONTEST_SEGNET_CLASS_NAME_TUPLE, SELFCOMP_CLASS_TO_GRAY

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
BUILD_KIND = "categorical_byte_closed_local_candidate_build"
CANDIDATE_KIND = "categorical_qma9_clade_spade_openpilot_candidate_manifest"
ARCHIVE_MEMBER_MANIFEST_KIND = "categorical_local_candidate_archive_member_manifest"
RUNTIME_PROOF_SKELETON_KIND = "categorical_runtime_consumer_proof_skeleton"
RUNTIME_PROOF_SKELETON_CONTRACT = "categorical_runtime_consumer_proof_skeleton_v1"
RUNTIME_CONSUMER_REPO_PATH = "src/tac/categorical_candidate_runtime_skeleton.py"
MEMBER_ROLES = {
    "categorical_payload.bin": "categorical_payload",
    "class_codebook.json": "decoder_table",
    "inflate.sh": "inflate_entrypoint",
    "runtime_consumer.py": "decoder_or_runtime_consumer",
    "runtime_consumer_proof_skeleton.json": "runtime_consumer_proof",
}
MEMBER_ORDER = tuple(sorted(MEMBER_ROLES))


class CategoricalPayloadCandidateError(RuntimeError):
    """Raised when a local categorical payload candidate cannot be built."""


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=DETERMINISTIC_ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_STORED
    mode = DETERMINISTIC_ZIP_INFLATE_MODE if name == "inflate.sh" else DETERMINISTIC_ZIP_FILE_MODE
    info.external_attr = mode << 16
    info.create_system = DETERMINISTIC_ZIP_CREATE_SYSTEM
    return info


def _inflate_script() -> bytes:
    return (
        b"#!/usr/bin/env bash\n"
        b"set -euo pipefail\n"
        b"cd \"$(dirname \"$0\")\"\n"
        b"python3 runtime_consumer.py --archive-root . >/dev/stderr || true\n"
        b"echo 'categorical payload candidate is fail-closed: decode/re-encode and runtime parity missing' >&2\n"
        b"exit 2\n"
    )


def extract_pr91_hpm1_categorical_payload(
    source_archive: str | Path = DEFAULT_PR91_ARCHIVE,
) -> tuple[bytes, dict[str, Any]]:
    """Extract the PR91 HPM1 mask segment as a categorical payload source."""

    archive_path = Path(source_archive)
    if not archive_path.is_file():
        raise CategoricalPayloadCandidateError(f"source archive missing: {archive_path}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if names != ["x"]:
                raise CategoricalPayloadCandidateError(f"expected single member x, got {names!r}")
            member_x = archive.read("x")
    except zipfile.BadZipFile as exc:
        raise CategoricalPayloadCandidateError(f"source archive is not a readable zip: {exc}") from exc
    if sha256_bytes(member_x) != EXPECTED_PR91_MEMBER_X_SHA256:
        raise CategoricalPayloadCandidateError("source archive member x does not match expected PR91 custody")
    try:
        bundle = parse_pr85_bundle(member_x)
        payload = bytes(bundle.segments["mask"])
    except (KeyError, Pr85BundleError, ValueError) as exc:
        raise CategoricalPayloadCandidateError(f"could not extract PR91 HPM1 mask segment: {exc}") from exc
    payload_sha = sha256_bytes(payload)
    if len(payload) != EXPECTED_PR91_HPM1_MASK_BYTES or payload_sha != EXPECTED_PR91_HPM1_MASK_SHA256:
        raise CategoricalPayloadCandidateError("PR91 HPM1 mask payload does not match expected custody")
    return payload, {
        "kind": "pr91_hpm1_mask_segment",
        "source_archive_path": repo_relative(archive_path, REPO_ROOT),
        "source_archive_bytes": archive_path.stat().st_size,
        "source_archive_sha256": sha256_file(archive_path),
        "source_member": "x",
        "source_member_sha256": sha256_bytes(member_x),
        "segment_name": "mask",
        "payload_bytes": len(payload),
        "payload_sha256": payload_sha,
        "payload_magic": payload[:4].hex(),
    }


def _runtime_proof_skeleton(
    *,
    payload_source: dict[str, Any],
    payload_sha256: str,
    class_codebook_sha256: str,
    runtime_consumer_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RUNTIME_PROOF_SKELETON_KIND,
        "runtime_consumer_proof_skeleton_contract": RUNTIME_PROOF_SKELETON_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "payload_source": payload_source,
        "charged_member_names": list(MEMBER_ORDER),
        "charged_member_sha256": {
            "categorical_payload.bin": payload_sha256,
            "class_codebook.json": class_codebook_sha256,
            "runtime_consumer.py": runtime_consumer_sha256,
        },
        "proof_status": {
            "archive_contains_payload_codebook_and_runtime": True,
            "full_decode_reencode_parity": False,
            "runtime_output_parity": False,
            "exact_cuda_auth_eval": False,
        },
        "dispatch_blockers": [
            "decode_reencode_parity_not_proven",
            "runtime_loader_parity_not_proven",
            "exact_cuda_auth_eval_after_lane_claim_missing",
        ],
    }


def _write_archive(path: Path, member_payloads: dict[str, bytes]) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in MEMBER_ORDER:
            raw = member_payloads[name]
            archive.writestr(_zip_info(name), raw, compress_type=zipfile.ZIP_STORED)
            records.append(
                {
                    "name": name,
                    "role": MEMBER_ROLES[name],
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                }
            )
    return records


def build_categorical_payload_candidate(
    *,
    out_dir: str | Path,
    categorical_payload: bytes,
    payload_source: dict[str, Any],
    repo_root: str | Path,
    source_archive_sha256: str,
) -> dict[str, Any]:
    """Build a local byte-closed candidate archive and readiness manifest."""

    root = Path(repo_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime_source_path = root / RUNTIME_CONSUMER_REPO_PATH
    if not runtime_source_path.is_file():
        raise CategoricalPayloadCandidateError(f"runtime skeleton missing: {runtime_source_path}")

    class_codebook = json_text(build_categorical_class_codebook()).encode("utf-8")
    runtime_consumer = runtime_source_path.read_bytes()
    payload_sha = sha256_bytes(categorical_payload)
    class_codebook_sha = sha256_bytes(class_codebook)
    runtime_consumer_sha = sha256_bytes(runtime_consumer)
    proof_skeleton = json_text(
        _runtime_proof_skeleton(
            payload_source=payload_source,
            payload_sha256=payload_sha,
            class_codebook_sha256=class_codebook_sha,
            runtime_consumer_sha256=runtime_consumer_sha,
        )
    ).encode("utf-8")
    member_payloads = {
        "categorical_payload.bin": categorical_payload,
        "class_codebook.json": class_codebook,
        "inflate.sh": _inflate_script(),
        "runtime_consumer.py": runtime_consumer,
        "runtime_consumer_proof_skeleton.json": proof_skeleton,
    }
    archive_path = out / "archive.zip"
    member_records = _write_archive(archive_path, member_payloads)
    archive_sha = sha256_file(archive_path)

    archive_member_manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": ARCHIVE_MEMBER_MANIFEST_KIND,
        "archive_member_manifest_contract": ARCHIVE_MEMBER_MANIFEST_CONTRACT,
        "fixture_only": False,
        "member_count": len(member_records),
        "member_order": [record["name"] for record in member_records],
        "members": member_records,
    }
    archive_member_manifest_path = out / "archive_member_manifest.json"
    write_json(archive_member_manifest_path, archive_member_manifest)
    archive_member_manifest_sha = sha256_file(archive_member_manifest_path)

    construction_plan = build_categorical_charged_label_plan(
        source_archive_sha256=source_archive_sha256,
        charged_members=member_records,
        conditioning_priors=[
            {
                "family": "qma9",
                "name": "local_categorical_payload",
                "usage": "inflate_runtime_conditioning",
                "runtime_consumed": True,
                "charged_member": "categorical_payload.bin",
            },
            {
                "family": "openpilot_priors",
                "name": "ego_lane_atom_ranker",
                "usage": "compression_time_atom_ranking_only",
                "runtime_consumed": False,
            },
            {
                "family": "clade_spade",
                "name": "canonical_class_codebook_conditioning",
                "usage": "inflate_runtime_conditioning",
                "runtime_consumed": True,
                "charged_member": "class_codebook.json",
            },
        ],
        candidate_archive_sha256=archive_sha,
        archive_member_manifest_sha256=archive_member_manifest_sha,
    )
    candidate = {
        "schema_version": SCHEMA_VERSION,
        "kind": CANDIDATE_KIND,
        "candidate_manifest_contract": CANDIDATE_MANIFEST_CONTRACT,
        "fixture_only": False,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "source_archive_sha256": source_archive_sha256,
        "archive_member_manifest_sha256": archive_member_manifest_sha,
        "archive_member_manifest": {
            "path": "archive_member_manifest.json",
            "bytes": archive_member_manifest_path.stat().st_size,
            "sha256": archive_member_manifest_sha,
        },
        "candidate_archive_contract": "contest_archive_zip",
        "candidate_archive": {
            "path": "archive.zip",
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha,
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
            "schema_version": SCHEMA_VERSION,
            "runtime_loader_parity_contract": RUNTIME_LOADER_PARITY_CONTRACT,
            "passed": False,
            "score_claim": False,
            "dispatch_attempted": False,
            "runtime_consumer_path": RUNTIME_CONSUMER_REPO_PATH,
            "runtime_consumer_sha256": runtime_consumer_sha,
            "loader_member": "runtime_consumer.py",
            "loader_member_sha256": runtime_consumer_sha,
            "byte_identical_to_runtime_consumer": True,
            "sidecar_free": True,
            "fallback_used": False,
            "loaded_charged_members": [
                "categorical_payload.bin",
                "class_codebook.json",
                "runtime_consumer_proof_skeleton.json",
            ],
            "blocker": "runtime_output_parity_not_proven",
        },
        "decode_reencode_parity": {
            "schema_version": SCHEMA_VERSION,
            "decode_reencode_parity_contract": DECODE_REENCODE_PARITY_CONTRACT,
            "passed": False,
            "score_claim": False,
            "dispatch_attempted": False,
            "payload_member": "categorical_payload.bin",
            "payload_member_sha256": payload_sha,
            "full_decode": {
                "passed": False,
                "frame_count": None,
                "decoded_masks_sha256": "",
                "blocker": "full_categorical_payload_decode_not_proven",
            },
            "byte_exact_reencode": {
                "passed": False,
                "byte_exact": False,
                "reencoded_payload_sha256": "",
                "blocker": "byte_exact_reencode_not_proven",
            },
            "sidecar_free": True,
        },
        "candidate_construction_plan": construction_plan,
        "conditioning_priors": construction_plan["conditioning_priors"],
        "charged_members": member_records,
        "no_op_controls": {
            "decode_reencode_identity_control": {
                "passed": False,
                "scope": "real_payload_requires_decode_reencode_parity",
            },
            "label_permutation_fail_closed_control": {
                "passed": False,
                "scope": "real_payload_requires_label_permutation_control",
            },
            "charged_member_presence_control": {
                "passed": True,
                "scope": "archive_member_manifest_and_zip_member_sha256",
            },
            "runtime_consumes_conditioning_control": {
                "passed": False,
                "scope": "runtime_skeleton_verifies_members_but_does_not_reconstruct",
            },
        },
        "payload_source": payload_source,
        "runtime_consumer_proof_skeleton_member": {
            "name": "runtime_consumer_proof_skeleton.json",
            "bytes": len(proof_skeleton),
            "sha256": sha256_bytes(proof_skeleton),
            "contract": RUNTIME_PROOF_SKELETON_CONTRACT,
        },
        "candidate_rows": [
            {
                "row_id": "local_categorical_payload_custody",
                "score_claim": False,
                "dispatch_attempted": False,
                "ready_for_exact_eval_dispatch": False,
                "evidence_grade": "local_payload_custody",
                "payload_member": "categorical_payload.bin",
                "payload_sha256": payload_sha,
                "status": "byte_closed_local_candidate_blocked_on_parity",
            }
        ],
    }
    candidate_path = out / "candidate.json"
    construction_plan_path = out / "construction_plan.json"
    readiness_path = out / "readiness.json"
    write_json(construction_plan_path, construction_plan)
    write_json(candidate_path, candidate)
    readiness = audit_categorical_candidate_manifest(
        candidate,
        repo_root=root,
        manifest_dir=out,
    )
    write_json(readiness_path, readiness)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": BUILD_KIND,
        "fixture_only": False,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "payload_source": payload_source,
        "paths": {
            "archive": repo_relative(archive_path, root),
            "archive_member_manifest": repo_relative(archive_member_manifest_path, root),
            "construction_plan": repo_relative(construction_plan_path, root),
            "candidate": repo_relative(candidate_path, root),
            "readiness": repo_relative(readiness_path, root),
        },
        "archive_sha256": archive_sha,
        "archive_bytes": archive_path.stat().st_size,
        "charged_members": member_records,
        "readiness_blockers": readiness["dispatch_blockers"],
    }
    write_json(out / "summary.json", summary)
    return {
        "archive_member_manifest": archive_member_manifest,
        "candidate_manifest": candidate,
        "readiness": readiness,
        "summary": summary,
    }


__all__ = [
    "ARCHIVE_MEMBER_MANIFEST_KIND",
    "BUILD_KIND",
    "CANDIDATE_KIND",
    "MEMBER_ORDER",
    "RUNTIME_CONSUMER_REPO_PATH",
    "RUNTIME_PROOF_SKELETON_CONTRACT",
    "CategoricalPayloadCandidateError",
    "build_categorical_payload_candidate",
    "extract_pr91_hpm1_categorical_payload",
]
