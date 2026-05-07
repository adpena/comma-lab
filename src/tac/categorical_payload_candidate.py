"""Deterministic builder for local categorical payload candidates.

The builder materializes a byte-closed local archive from an existing
categorical payload plus the canonical class codebook and a charged runtime
consumer. It executes the archived inflate entrypoint locally to prove the
runtime consumes charged members, while decode/re-encode and exact-eval
readiness remain explicit blockers.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
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
    DECODE_REENCODE_INDEPENDENT_PROOF_KIND,
    DECODE_REENCODE_PARITY_CONTRACT,
    DETERMINISTIC_ZIP_CREATE_SYSTEM,
    DETERMINISTIC_ZIP_DATE_TIME,
    DETERMINISTIC_ZIP_FILE_MODE,
    DETERMINISTIC_ZIP_INFLATE_MODE,
    RUNTIME_EXECUTION_PROOF_KIND,
    RUNTIME_LOADER_PARITY_CONTRACT,
    audit_categorical_candidate_manifest,
)
from tac.categorical_label_prior_payload_manifest import (
    LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT,
    LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER,
    LABEL_PRIOR_PAYLOAD_MANIFEST_ROLE,
    build_categorical_label_prior_payload_manifest,
)
from tac.categorical_openpilot_mask_prior_contract import RUNTIME_LABEL_CONTRACT
from tac.hpm1_payload_structure import (
    HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
    build_hpm1_structural_decode_inventory,
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
RUNTIME_EXECUTION_PROOF_FILENAME = "runtime_execution_proof.json"
LABEL_PERMUTATION_CONTROL_FILENAME = "label_permutation_control.json"
HPM1_STRUCTURAL_INVENTORY_FILENAME = "hpm1_structural_inventory.json"
DECODE_REENCODE_BLOCKED_PROOF_FILENAME = "decode_reencode_blocked_proof.json"
RUNTIME_CONSUMER_REPO_PATH = "src/tac/categorical_candidate_runtime_skeleton.py"
MEMBER_ROLES = {
    "categorical_payload.bin": "categorical_payload",
    "class_codebook.json": "decoder_table",
    "inflate.sh": "inflate_entrypoint",
    LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER: LABEL_PRIOR_PAYLOAD_MANIFEST_ROLE,
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
        b"\"${PYTHON:-python3}\" runtime_consumer.py --archive-root . --json-out runtime_consumer_report.json >/dev/stderr\n"
        b"echo 'categorical payload candidate is fail-closed: decode/re-encode and exact eval missing' >&2\n"
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
    label_prior_payload_manifest_sha256: str,
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
            LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER: label_prior_payload_manifest_sha256,
            "runtime_consumer.py": runtime_consumer_sha256,
        },
        "proof_status": {
            "archive_contains_payload_codebook_and_runtime": True,
            "charged_label_prior_payload_manifest": True,
            "runtime_consumes_charged_members": True,
            "full_decode_reencode_parity": False,
            "exact_cuda_auth_eval": False,
        },
        "dispatch_blockers": [
            "decode_reencode_parity_not_proven",
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


def _extract_candidate_archive_for_runtime(archive_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if names != list(MEMBER_ORDER):
            raise CategoricalPayloadCandidateError(
                f"candidate archive member order mismatch for runtime proof: {names!r}"
            )
        for name in names:
            if name not in MEMBER_ROLES or "/" in name or "\\" in name or name.startswith("."):
                raise CategoricalPayloadCandidateError(f"unsafe runtime proof member: {name!r}")
            target = target_dir / name
            target.write_bytes(archive.read(name))
            if name == "inflate.sh":
                target.chmod(0o755)


def _load_runtime_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = path.read_text(encoding="utf-8")
        loaded = json.loads(payload)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_runtime_execution_proof(
    *,
    out_dir: Path,
    archive_path: Path,
    candidate_archive_sha256: str,
    runtime_consumer_sha256: str,
    loaded_charged_members: list[str],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="categorical-runtime-proof-") as temp_name:
        temp_dir = Path(temp_name)
        _extract_candidate_archive_for_runtime(archive_path, temp_dir)
        env = dict(os.environ)
        env["PYTHON"] = sys.executable
        env["PYTHONPATH"] = ""
        completed = subprocess.run(
            ["/bin/bash", "inflate.sh"],
            cwd=temp_dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        runtime_report = _load_runtime_report(temp_dir / "runtime_consumer_report.json")

    consumed_members = runtime_report.get("consumed_charged_members")
    if not isinstance(consumed_members, list):
        consumed_members = []
    consumed_member_names = [name for name in consumed_members if isinstance(name, str)]
    runtime_output_sha = runtime_report.get("runtime_output_sha256")
    if not isinstance(runtime_output_sha, str):
        runtime_output_sha = sha256_bytes(
            (completed.stdout + "\n" + completed.stderr).encode("utf-8")
        )
    proof = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUNTIME_EXECUTION_PROOF_KIND,
        "independent_proof": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "producer_tool": "tac.categorical_payload_candidate.build_categorical_payload_candidate",
        "proof_scope": "archive_inflate_runtime_execution",
        "candidate_archive_sha256": candidate_archive_sha256,
        "runtime_consumer_sha256": runtime_consumer_sha256,
        "loader_member": "runtime_consumer.py",
        "loader_member_sha256": runtime_consumer_sha256,
        "runtime_executed": runtime_report.get("runtime_executed") is True,
        "executed_archive_inflate": completed.returncode == 2,
        "inflate_returncode": completed.returncode,
        "expected_fail_closed_exit": completed.returncode == 2,
        "sidecar_free": runtime_report.get("sidecar_free") is True,
        "fallback_used": runtime_report.get("fallback_used") is True,
        "consumed_charged_members": consumed_member_names,
        "loaded_charged_members": loaded_charged_members,
        "runtime_output_sha256": runtime_output_sha,
        "runtime_report_summary": {
            "kind": runtime_report.get("kind", ""),
            "payload_codec": (
                runtime_report.get("categorical_payload", {}).get("codec", "")
                if isinstance(runtime_report.get("categorical_payload"), dict)
                else ""
            ),
            "hpm1_structural_reencode_passed": (
                runtime_report.get("categorical_payload", {})
                .get("structural_reencode", {})
                .get("passed")
                is True
                if isinstance(runtime_report.get("categorical_payload"), dict)
                else False
            ),
            "hpm1_hpac_model_load_passed": (
                runtime_report.get("categorical_payload", {})
                .get("hpac_model_load", {})
                .get("passed")
                is True
                if isinstance(runtime_report.get("categorical_payload"), dict)
                else False
            ),
            "full_decode_reencode_parity_proven": runtime_report.get(
                "full_decode_reencode_parity_proven"
            )
            is True,
        },
    }
    proof_path = out_dir / RUNTIME_EXECUTION_PROOF_FILENAME
    write_json(proof_path, proof)
    return {
        "path": RUNTIME_EXECUTION_PROOF_FILENAME,
        "bytes": proof_path.stat().st_size,
        "sha256": sha256_file(proof_path),
    }


def _write_label_permutation_control(
    *,
    out_dir: Path,
    archive_path: Path,
    candidate_archive_sha256: str,
    runtime_consumer_sha256: str,
    class_codebook_sha256: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="categorical-label-permutation-") as temp_name:
        temp_dir = Path(temp_name)
        _extract_candidate_archive_for_runtime(archive_path, temp_dir)
        codebook_path = temp_dir / "class_codebook.json"
        codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
        classes = codebook.get("classes")
        if not isinstance(classes, list) or len(classes) < 2:
            raise CategoricalPayloadCandidateError("class codebook has no permutable classes")
        original_order = [row.get("name", "") for row in classes if isinstance(row, dict)]
        permuted_classes = list(reversed(classes))
        codebook["classes"] = permuted_classes
        codebook_path.write_text(json_text(codebook), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = ""
        completed = subprocess.run(
            [
                sys.executable,
                "runtime_consumer.py",
                "--archive-root",
                ".",
                "--json-out",
                "label_permutation_runtime_report.json",
            ],
            cwd=temp_dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )

    stderr = completed.stderr or ""
    stdout = completed.stdout or ""
    expected_failure = "class codebook class order mismatch" in stderr
    passed = completed.returncode == 2 and expected_failure
    proof = {
        "schema_version": SCHEMA_VERSION,
        "kind": "categorical_label_permutation_fail_closed_control",
        "independent_proof": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "producer_tool": "tac.categorical_payload_candidate.build_categorical_payload_candidate",
        "proof_scope": "label_permutation_fail_closed_control",
        "candidate_archive_sha256": candidate_archive_sha256,
        "runtime_consumer_sha256": runtime_consumer_sha256,
        "class_codebook_sha256": class_codebook_sha256,
        "control_name": "label_permutation_fail_closed_control",
        "passed": passed,
        "mutation": {
            "member": "class_codebook.json",
            "operation": "reverse_classes_order",
            "original_class_order": original_order,
            "permuted_class_order": list(reversed(original_order)),
        },
        "runtime_invocation": {
            "returncode": completed.returncode,
            "expected_returncode": 2,
            "expected_failure_observed": expected_failure,
            "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
            "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
        },
        "failure_contract": {
            "expected_runtime_error": "class codebook class order mismatch",
            "fail_closed": passed,
            "sidecar_free": True,
            "fallback_used": False,
        },
    }
    proof_path = out_dir / LABEL_PERMUTATION_CONTROL_FILENAME
    write_json(proof_path, proof)
    return {
        "path": LABEL_PERMUTATION_CONTROL_FILENAME,
        "bytes": proof_path.stat().st_size,
        "sha256": sha256_file(proof_path),
        "passed": passed,
        "control_name": "label_permutation_fail_closed_control",
        "mutation": proof["mutation"],
        "expected_failure_observed": expected_failure,
    }


def _write_hpm1_structural_inventory(
    *,
    out_dir: Path,
    categorical_payload: bytes,
    payload_source: dict[str, Any],
    candidate_archive_path: Path,
) -> dict[str, Any] | None:
    source_archive_raw = payload_source.get("source_archive_path")
    source_archive_path: Path | None = None
    if isinstance(source_archive_raw, str) and source_archive_raw:
        source_archive_path = Path(source_archive_raw)
        if not source_archive_path.is_absolute():
            source_archive_path = REPO_ROOT / source_archive_path
    try:
        inventory = build_hpm1_structural_decode_inventory(
            categorical_payload,
            payload_member="categorical_payload.bin",
            source_archive=source_archive_path,
            source_member=f"{payload_source.get('source_member', 'x')}:"
            f"{payload_source.get('segment_name', 'mask')}",
            candidate_archive=candidate_archive_path,
        )
    except (Pr85BundleError, ValueError, struct.error):
        return None
    inventory_path = out_dir / HPM1_STRUCTURAL_INVENTORY_FILENAME
    write_json(inventory_path, inventory)
    return {
        "path": HPM1_STRUCTURAL_INVENTORY_FILENAME,
        "bytes": inventory_path.stat().st_size,
        "sha256": sha256_file(inventory_path),
        "contract": HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
        "payload_member": "categorical_payload.bin",
        "payload_member_sha256": sha256_bytes(categorical_payload),
        "structural_reencode_matches_source": inventory["structural_reencode"][
            "matches_source_segment"
        ],
        "full_decode_proven": inventory["full_decode"]["passed"],
        "byte_exact_semantic_reencode_proven": inventory[
            "byte_exact_semantic_reencode"
        ]["passed"],
        "unsupported_wire_constructs": [
            row["name"] for row in inventory["unsupported_wire_constructs"]
        ],
    }


def _write_decode_reencode_blocked_proof(
    *,
    out_dir: Path,
    candidate_archive_sha256: str,
    payload_member_sha256: str,
    hpm1_structural_inventory: dict[str, Any] | None,
) -> dict[str, Any]:
    """Write an independent negative proof for the current semantic parity blocker."""

    unsupported = (
        hpm1_structural_inventory.get("unsupported_wire_constructs", [])
        if hpm1_structural_inventory is not None
        else []
    )
    proof = {
        "schema_version": SCHEMA_VERSION,
        "kind": DECODE_REENCODE_INDEPENDENT_PROOF_KIND,
        "independent_proof": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "producer_tool": "tac.categorical_payload_candidate.build_categorical_payload_candidate",
        "proof_scope": "full_decode_reencode",
        "candidate_archive_sha256": candidate_archive_sha256,
        "payload_member": "categorical_payload.bin",
        "payload_member_sha256": payload_member_sha256,
        "full_decode": {
            "passed": False,
            "frame_count": None,
            "decoded_masks_sha256": "",
            "blocker": "hpm1_semantic_decode_not_proven",
        },
        "byte_exact_reencode": {
            "passed": False,
            "byte_exact": False,
            "reencoded_payload_sha256": "",
            "blocker": "hpm1_byte_exact_semantic_reencode_not_proven",
        },
        "sidecar_free": True,
        "negative_proof": {
            "status": "blocked_on_hpm1_semantic_decode_reencode",
            "structural_inventory_attached": hpm1_structural_inventory is not None,
            "structural_reencode_matches_source": bool(
                hpm1_structural_inventory
                and hpm1_structural_inventory.get("structural_reencode_matches_source") is True
            ),
            "unsupported_wire_constructs": list(unsupported),
        },
    }
    proof_path = out_dir / DECODE_REENCODE_BLOCKED_PROOF_FILENAME
    write_json(proof_path, proof)
    return {
        "path": DECODE_REENCODE_BLOCKED_PROOF_FILENAME,
        "bytes": proof_path.stat().st_size,
        "sha256": sha256_file(proof_path),
        "kind": DECODE_REENCODE_INDEPENDENT_PROOF_KIND,
        "accepted_expected": False,
    }


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
    conditioning_priors = [
        {
            "family": "qma9",
            "name": "local_categorical_payload",
            "usage": "inflate_runtime_conditioning",
            "runtime_consumed": True,
            "charged_member": "categorical_payload.bin",
            "charged_member_sha256": payload_sha,
            "label_contract": RUNTIME_LABEL_CONTRACT,
            "source_provenance": {
                "kind": "charged_archive_member",
                "charged_member": "categorical_payload.bin",
                "sha256": payload_sha,
                "source_kind": payload_source.get("kind", ""),
            },
        },
        {
            "family": "openpilot_priors",
            "name": "ego_lane_atom_ranker",
            "usage": "compression_time_atom_ranking_only",
            "runtime_consumed": False,
            "label_contract": "openpilot_prior_hints_non_runtime",
            "source_provenance": {
                "kind": "compression_time_only_derivation",
                "source": "categorical_candidate_plan.openpilot_prior_hints",
                "runtime_consumed": False,
            },
        },
        {
            "family": "clade_spade",
            "name": "canonical_class_codebook_conditioning",
            "usage": "inflate_runtime_conditioning",
            "runtime_consumed": True,
            "charged_member": "class_codebook.json",
            "charged_member_sha256": class_codebook_sha,
            "label_contract": RUNTIME_LABEL_CONTRACT,
            "source_provenance": {
                "kind": "charged_archive_member",
                "charged_member": "class_codebook.json",
                "sha256": class_codebook_sha,
                "source": "build_categorical_class_codebook",
            },
        },
    ]
    label_prior_payload_manifest = json_text(
        build_categorical_label_prior_payload_manifest(
            source_archive_sha256=source_archive_sha256,
            payload_member="categorical_payload.bin",
            payload_member_sha256=payload_sha,
            class_codebook_member="class_codebook.json",
            class_codebook_sha256=class_codebook_sha,
            conditioning_priors=conditioning_priors,
        )
    ).encode("utf-8")
    label_prior_payload_manifest_sha = sha256_bytes(label_prior_payload_manifest)
    proof_skeleton = json_text(
        _runtime_proof_skeleton(
            payload_source=payload_source,
            payload_sha256=payload_sha,
            class_codebook_sha256=class_codebook_sha,
            label_prior_payload_manifest_sha256=label_prior_payload_manifest_sha,
            runtime_consumer_sha256=runtime_consumer_sha,
        )
    ).encode("utf-8")
    member_payloads = {
        "categorical_payload.bin": categorical_payload,
        "class_codebook.json": class_codebook,
        "inflate.sh": _inflate_script(),
        LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER: label_prior_payload_manifest,
        "runtime_consumer.py": runtime_consumer,
        "runtime_consumer_proof_skeleton.json": proof_skeleton,
    }
    archive_path = out / "archive.zip"
    member_records = _write_archive(archive_path, member_payloads)
    archive_sha = sha256_file(archive_path)
    loaded_charged_members = [
        "categorical_payload.bin",
        "class_codebook.json",
        LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER,
        "runtime_consumer_proof_skeleton.json",
    ]
    runtime_execution_proof = _write_runtime_execution_proof(
        out_dir=out,
        archive_path=archive_path,
        candidate_archive_sha256=archive_sha,
        runtime_consumer_sha256=runtime_consumer_sha,
        loaded_charged_members=loaded_charged_members,
    )
    label_permutation_control = _write_label_permutation_control(
        out_dir=out,
        archive_path=archive_path,
        candidate_archive_sha256=archive_sha,
        runtime_consumer_sha256=runtime_consumer_sha,
        class_codebook_sha256=class_codebook_sha,
    )
    hpm1_structural_inventory = _write_hpm1_structural_inventory(
        out_dir=out,
        categorical_payload=categorical_payload,
        payload_source=payload_source,
        candidate_archive_path=archive_path,
    )
    decode_reencode_blocked_proof = _write_decode_reencode_blocked_proof(
        out_dir=out,
        candidate_archive_sha256=archive_sha,
        payload_member_sha256=payload_sha,
        hpm1_structural_inventory=hpm1_structural_inventory,
    )

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
        conditioning_priors=conditioning_priors,
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
        "label_prior_payload_manifest": {
            "member": LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER,
            "bytes": len(label_prior_payload_manifest),
            "sha256": label_prior_payload_manifest_sha,
            "contract": LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT,
        },
        **(
            {"hpm1_structural_decode_inventory": hpm1_structural_inventory}
            if hpm1_structural_inventory is not None
            else {}
        ),
        "runtime_loader_parity": {
            "schema_version": SCHEMA_VERSION,
            "runtime_loader_parity_contract": RUNTIME_LOADER_PARITY_CONTRACT,
            "passed": True,
            "score_claim": False,
            "dispatch_attempted": False,
            "runtime_consumer_path": RUNTIME_CONSUMER_REPO_PATH,
            "runtime_consumer_sha256": runtime_consumer_sha,
            "loader_member": "runtime_consumer.py",
            "loader_member_sha256": runtime_consumer_sha,
            "byte_identical_to_runtime_consumer": True,
            "sidecar_free": True,
            "fallback_used": False,
            "loaded_charged_members": loaded_charged_members,
            "runtime_execution_proof": runtime_execution_proof,
            "semantic_runtime_output_parity_proven": False,
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
            "independent_proof_artifact": decode_reencode_blocked_proof,
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
                "passed": label_permutation_control["passed"],
                "scope": "runtime_rejects_permuted_charged_class_codebook",
                "proof_artifact": label_permutation_control,
            },
            "charged_member_presence_control": {
                "passed": True,
                "scope": "archive_member_manifest_and_zip_member_sha256",
            },
            "runtime_consumes_conditioning_control": {
                "passed": True,
                "scope": "runtime_consumer_executes_from_archive_and_consumes_charged_members",
            },
        },
        "payload_source": payload_source,
        "runtime_consumer_proof_skeleton_member": {
            "name": "runtime_consumer_proof_skeleton.json",
            "bytes": len(proof_skeleton),
            "sha256": sha256_bytes(proof_skeleton),
            "contract": RUNTIME_PROOF_SKELETON_CONTRACT,
        },
        "label_prior_payload_manifest_member": {
            "name": LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER,
            "bytes": len(label_prior_payload_manifest),
            "sha256": label_prior_payload_manifest_sha,
            "contract": LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT,
        },
        "label_permutation_control": label_permutation_control,
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
            **(
                {
                    "hpm1_structural_inventory": repo_relative(
                        out / HPM1_STRUCTURAL_INVENTORY_FILENAME,
                        root,
                    )
                }
                if hpm1_structural_inventory is not None
                else {}
            ),
            "runtime_execution_proof": repo_relative(
                out / RUNTIME_EXECUTION_PROOF_FILENAME,
                root,
            ),
            "label_permutation_control": repo_relative(
                out / LABEL_PERMUTATION_CONTROL_FILENAME,
                root,
            ),
            "decode_reencode_blocked_proof": repo_relative(
                out / DECODE_REENCODE_BLOCKED_PROOF_FILENAME,
                root,
            ),
        },
        "archive_sha256": archive_sha,
        "archive_bytes": archive_path.stat().st_size,
        "charged_members": member_records,
        **(
            {"hpm1_structural_decode_inventory": hpm1_structural_inventory}
            if hpm1_structural_inventory is not None
            else {}
        ),
        "readiness_blockers": readiness["dispatch_blockers"],
        "runtime_execution_proof": runtime_execution_proof,
        "label_permutation_control": label_permutation_control,
        "decode_reencode_blocked_proof": decode_reencode_blocked_proof,
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
    "HPM1_STRUCTURAL_INVENTORY_FILENAME",
    "LABEL_PERMUTATION_CONTROL_FILENAME",
    "LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT",
    "LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER",
    "MEMBER_ORDER",
    "RUNTIME_CONSUMER_REPO_PATH",
    "RUNTIME_EXECUTION_PROOF_FILENAME",
    "RUNTIME_PROOF_SKELETON_CONTRACT",
    "CategoricalPayloadCandidateError",
    "build_categorical_payload_candidate",
    "extract_pr91_hpm1_categorical_payload",
]
