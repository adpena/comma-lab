# SPDX-License-Identifier: MIT
"""Family-agnostic byte-shaving materializers.

These materializers are intentionally conservative: they can emit byte-closed
candidate archives for HNeRV, HNeRV bolt-ons, broader NeRV-family packets, and
non-NeRV ZIP/tensor archives, but they never claim score or exact-eval
readiness without an explicit runtime-consumption proof.
"""

from __future__ import annotations

import io
import math
import struct
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli

from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.repo_io import read_json, sha256_bytes, sha256_file, write_bytes_artifact, write_json_artifact

ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA = "archive_section_entropy_recode_candidate.v1"
PACKET_MEMBER_RECOMPRESS_SCHEMA = "packet_member_recompress_candidate.v1"
PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA = "packet_member_zip_header_elide_candidate.v1"
TENSOR_FACTORIZE_SCHEMA = "tensor_factorize_candidate.v1"
ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND = "archive_section_entropy_recode_v1"
PACKET_MEMBER_RECOMPRESS_TARGET_KIND = "packet_member_recompress_v1"
PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND = "packet_member_zip_header_elide_v1"
TENSOR_FACTORIZE_TARGET_KIND = "tensor_factorize_v1"
ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER_ID = "archive_section_entropy_recode_adapter"
PACKET_MEMBER_RECOMPRESS_MATERIALIZER_ID = "packet_member_recompress_adapter"
PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER_ID = "packet_member_zip_header_elide_adapter"
TENSOR_FACTORIZE_MATERIALIZER_ID = "tensor_factorize_adapter"
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}


class FamilyAgnosticMaterializerError(ValueError):
    """Raised when a family-agnostic materializer cannot build a candidate."""


def materialize_packet_member_recompress_candidate(
    *,
    archive_path: str | Path,
    output_archive: str | Path,
    packet_member_manifest: str | Path | Mapping[str, Any] | None = None,
    member_name: str | None = None,
    compression_methods: Sequence[str] = ("stored", "deflated"),
    compresslevels: Sequence[int] = (9,),
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None = None,
    runtime_consumption_proof_out: str | Path | None = None,
    repo_root: str | Path | None = None,
    allow_size_regression: bool = False,
    allow_overwrite: bool = False,
    expected_existing_output_sha256: str | None = None,
    expected_existing_runtime_consumption_proof_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Recompress one ZIP member without changing its member payload bytes."""

    repo = _repo(repo_root)
    if runtime_consumption_proof is not None and runtime_consumption_proof_out is not None:
        raise FamilyAgnosticMaterializerError(
            "runtime_consumption_proof and runtime_consumption_proof_out are mutually exclusive"
        )
    archive = _resolve_path(archive_path, repo=repo)
    output = _resolve_path(output_archive, repo=repo)
    proof_out = (
        _resolve_path(runtime_consumption_proof_out, repo=repo)
        if runtime_consumption_proof_out is not None
        else None
    )
    manifest = _load_mapping(packet_member_manifest, repo=repo)
    target_member = _select_member_name(
        archive,
        explicit=member_name,
        manifest=manifest,
    )
    source_member_bytes = _zip_member_bytes(archive, target_member)
    source_record = _archive_record(archive)
    member_record = _member_record(
        archive,
        target_member,
        payload=source_member_bytes,
    )

    candidates: list[dict[str, Any]] = []
    for method in _normalized_compression_methods(compression_methods):
        levels = (None,) if method == zipfile.ZIP_STORED else _ordered_unique_ints(
            int(level) for level in compresslevels
        )
        for level in levels:
            payload = _zip_archive_bytes_with_replacement(
                archive,
                member_name=target_member,
                replacement=source_member_bytes,
                compression=method,
                compresslevel=level,
            )
            candidates.append(
                {
                    "compression_method": _compression_method_name(method),
                    "compresslevel": level,
                    "archive_bytes": len(payload),
                    "archive_sha256": sha256_bytes(payload),
                    "payload": payload,
                }
            )
    if not candidates:
        raise FamilyAgnosticMaterializerError("no compression candidates were produced")
    best = min(
        candidates,
        key=lambda item: (int(item["archive_bytes"]), str(item["compression_method"])),
    )
    saved_bytes = int(source_record["bytes"]) - int(best["archive_bytes"])
    blockers = []
    if saved_bytes <= 0 and not allow_size_regression:
        blockers.append("candidate_not_rate_positive")
    write_result = write_bytes_artifact(
        output,
        bytes(best["payload"]),
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_existing_output_sha256,
        min_free_bytes=min_free_bytes,
    )
    candidate_record = _archive_record(output)
    candidate_member_record = _member_record(output, target_member)
    proof_write_result = None
    runtime_proof_ref: str | Path | Mapping[str, Any] | None = runtime_consumption_proof
    if proof_out is not None:
        runtime_proof_payload = _packet_member_recompress_runtime_consumption_proof(
            source_archive=source_record,
            source_member=member_record,
            candidate_archive=candidate_record,
            candidate_member=candidate_member_record,
            selected_member_name=target_member,
            selected_compression={
                "compression_method": best["compression_method"],
                "compresslevel": best["compresslevel"],
                "source_archive_bytes": source_record["bytes"],
                "candidate_archive_bytes": candidate_record["bytes"],
                "saved_bytes": saved_bytes,
            },
        )
        proof_write_result = write_json_artifact(
            proof_out,
            runtime_proof_payload,
            allow_overwrite=allow_overwrite,
            expected_existing_sha256=expected_existing_runtime_consumption_proof_sha256,
            min_free_bytes=min_free_bytes,
        )
        runtime_proof_ref = proof_out
    receiver_verification = verify_runtime_consumption_proof(
        runtime_consumption_proof=runtime_proof_ref,
        required_candidate_archive_sha256=candidate_record["sha256"],
        required_candidate_member_sha256=member_record["sha256"],
        repo_root=repo,
    )
    readiness_blockers = _readiness_blockers(
        blockers,
        receiver_verification,
        receiver_blocker="packet_member_recompress_receiver_contract_not_satisfied",
    )
    return {
        "schema": PACKET_MEMBER_RECOMPRESS_SCHEMA,
        "materializer_id": PACKET_MEMBER_RECOMPRESS_MATERIALIZER_ID,
        "target_kind": PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        "receiver_contract_id": f"{PACKET_MEMBER_RECOMPRESS_TARGET_KIND}.receiver.v1",
        "receiver_contract_kind": "family_agnostic_packet_member_recompress",
        "byte_closed_candidate_emitted": True,
        "source_archive": source_record,
        "source_member": member_record,
        "candidate_archive": candidate_record,
        "candidate_member": candidate_member_record,
        "selected_member_name": target_member,
        "selected_compression": {
            "compression_method": best["compression_method"],
            "compresslevel": best["compresslevel"],
            "source_archive_bytes": source_record["bytes"],
            "candidate_archive_bytes": candidate_record["bytes"],
            "saved_bytes": saved_bytes,
        },
        "candidate_trials": [
            {
                key: value
                for key, value in trial.items()
                if key != "payload"
            }
            for trial in candidates
        ],
        "receiver_verification": receiver_verification,
        "runtime_consumption_proof_path": (
            proof_out.as_posix() if proof_out is not None else receiver_verification.get("proof_path")
        ),
        "runtime_consumption_proof_write": (
            proof_write_result.__dict__ if proof_write_result is not None else None
        ),
        "receiver_contract_satisfied": (
            receiver_verification["receiver_contract_satisfied"] is True
        ),
        "readiness_blockers": readiness_blockers,
        "artifact_write": write_result.__dict__,
        **FALSE_AUTHORITY,
    }


def _packet_member_recompress_runtime_consumption_proof(
    *,
    source_archive: Mapping[str, Any],
    source_member: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
    candidate_member: Mapping[str, Any],
    selected_member_name: str,
    selected_compression: Mapping[str, Any],
) -> dict[str, Any]:
    source_member_sha = _clean_str(source_member.get("sha256"))
    candidate_member_sha = _clean_str(candidate_member.get("sha256"))
    member_payload_identical = (
        source_member_sha is not None and candidate_member_sha == source_member_sha
    )
    return {
        "schema": "family_agnostic_runtime_consumption_proof_v1",
        "proof_kind": "packet_member_recompress_payload_identity_receiver_proof.v1",
        "proof_scope": "zip_member_payload_identity_after_archive_recompression",
        "target_kind": PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        "materializer_id": PACKET_MEMBER_RECOMPRESS_MATERIALIZER_ID,
        "receiver_contract_kind": "family_agnostic_packet_member_recompress",
        "receiver_contract_id": f"{PACKET_MEMBER_RECOMPRESS_TARGET_KIND}.receiver.v1",
        "selected_member_name": selected_member_name,
        "source_archive": dict(source_archive),
        "source_member": dict(source_member),
        "candidate_archive": dict(candidate_archive),
        "candidate_member": dict(candidate_member),
        "candidate_archive_sha256": candidate_archive.get("sha256"),
        "candidate_member_sha256": candidate_member_sha,
        "member_sha256": candidate_member_sha,
        "source_member_sha256": source_member_sha,
        "candidate_member_payload_identical_to_source": member_payload_identical,
        "runtime_consumption_probe": {
            "schema": "packet_member_payload_identity_probe.v1",
            "passed": member_payload_identical,
            "source_member_sha256": source_member_sha,
            "candidate_member_sha256": candidate_member_sha,
            "candidate_member_bytes": candidate_member.get("bytes"),
            "source_member_bytes": source_member.get("bytes"),
        },
        "selected_compression": dict(selected_compression),
        "receiver_contract_satisfied": member_payload_identical,
        "runtime_consumption_proof_passed": member_payload_identical,
        "passed": member_payload_identical,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }


def materialize_packet_member_zip_header_elide_candidate(
    *,
    archive_path: str | Path,
    output_archive: str | Path,
    packet_member_manifest: str | Path | Mapping[str, Any] | None = None,
    member_name: str | None = None,
    member_names: Sequence[str] = (),
    all_members: bool = False,
    header_elision_contract: str | Path | Mapping[str, Any] | None = None,
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None = None,
    runtime_consumption_proof_out: str | Path | None = None,
    repo_root: str | Path | None = None,
    allow_size_regression: bool = False,
    allow_overwrite: bool = False,
    expected_existing_output_sha256: str | None = None,
    expected_existing_runtime_consumption_proof_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Strip deterministic ZIP/member header metadata while preserving payload bytes."""

    repo = _repo(repo_root)
    if runtime_consumption_proof is not None and runtime_consumption_proof_out is not None:
        raise FamilyAgnosticMaterializerError(
            "runtime_consumption_proof and runtime_consumption_proof_out are mutually exclusive"
        )
    archive = _resolve_path(archive_path, repo=repo)
    output = _resolve_path(output_archive, repo=repo)
    proof_out = (
        _resolve_path(runtime_consumption_proof_out, repo=repo)
        if runtime_consumption_proof_out is not None
        else None
    )
    manifest = _load_mapping(packet_member_manifest, repo=repo)
    contract = _load_mapping(header_elision_contract, repo=repo)
    target_members = _select_member_names(
        archive,
        explicit=member_name,
        explicit_many=member_names,
        manifest=manifest,
        contract=contract,
        all_members=all_members,
    )
    primary_member = target_members[0]
    source_member_bytes = _zip_member_bytes(archive, primary_member)
    source_record = _archive_record(archive)
    source_member_records = [
        _member_record(archive, target_member)
        for target_member in target_members
    ]
    source_member_record = _member_record(
        archive,
        primary_member,
        payload=source_member_bytes,
    )
    source_headers = [_zip_header_record(archive, target_member) for target_member in target_members]
    source_header = _zip_header_record(archive, primary_member)
    source_header_summary = _zip_header_summary(archive, target_members)
    elision_options = _zip_header_elision_options(contract)
    payload = _zip_archive_bytes_with_header_elision(
        archive,
        member_names=target_members,
        strip_member_extra=elision_options["strip_member_extra"],
        strip_member_comment=elision_options["strip_member_comment"],
        strip_archive_comment=elision_options["strip_archive_comment"],
    )
    saved_bytes = int(source_record["bytes"]) - len(payload)
    blockers: list[str] = []
    if saved_bytes <= 0 and not allow_size_regression:
        blockers.append("candidate_not_rate_positive")
    write_result = write_bytes_artifact(
        output,
        payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_existing_output_sha256,
        min_free_bytes=min_free_bytes,
    )
    candidate_record = _archive_record(output)
    candidate_member_records = [
        _member_record(output, target_member)
        for target_member in target_members
    ]
    candidate_member_record = _member_record(output, primary_member)
    candidate_headers = [_zip_header_record(output, target_member) for target_member in target_members]
    candidate_header = _zip_header_record(output, primary_member)
    candidate_header_summary = _zip_header_summary(output, target_members)
    proof_write_result = None
    runtime_proof_ref: str | Path | Mapping[str, Any] | None = runtime_consumption_proof
    if proof_out is not None:
        runtime_proof_payload = _packet_member_zip_header_elide_runtime_consumption_proof(
            source_archive=source_record,
            source_member=source_member_record,
            source_members=source_member_records,
            source_header=source_header,
            source_headers=source_headers,
            source_header_summary=source_header_summary,
            candidate_archive=candidate_record,
            candidate_member=candidate_member_record,
            candidate_members=candidate_member_records,
            candidate_header=candidate_header,
            candidate_headers=candidate_headers,
            candidate_header_summary=candidate_header_summary,
            selected_member_name=primary_member,
            selected_member_names=target_members,
            selected_elision={
                **elision_options,
                "source_archive_bytes": source_record["bytes"],
                "candidate_archive_bytes": candidate_record["bytes"],
                "saved_bytes": saved_bytes,
            },
            contract=contract,
        )
        proof_write_result = write_json_artifact(
            proof_out,
            runtime_proof_payload,
            allow_overwrite=allow_overwrite,
            expected_existing_sha256=expected_existing_runtime_consumption_proof_sha256,
            min_free_bytes=min_free_bytes,
        )
        runtime_proof_ref = proof_out
    receiver_verification = verify_runtime_consumption_proof(
        runtime_consumption_proof=runtime_proof_ref,
        required_candidate_archive_sha256=candidate_record["sha256"],
        required_candidate_member_sha256=(
            source_member_record["sha256"] if len(target_members) == 1 else None
        ),
        required_candidate_member_sha256s={
            str(row["name"]): str(row["sha256"])
            for row in source_member_records
        },
        repo_root=repo,
    )
    readiness_blockers = _readiness_blockers(
        blockers,
        receiver_verification,
        receiver_blocker="packet_member_zip_header_elide_receiver_contract_not_satisfied",
    )
    return {
        "schema": PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA,
        "materializer_id": PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER_ID,
        "target_kind": PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        "receiver_contract_id": f"{PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND}.receiver.v1",
        "receiver_contract_kind": "family_agnostic_packet_member_zip_header_elide",
        "byte_closed_candidate_emitted": True,
        "source_archive": source_record,
        "source_member": source_member_record,
        "source_members": source_member_records,
        "source_zip_header": source_header,
        "source_zip_headers": source_headers,
        "source_zip_header_summary": source_header_summary,
        "candidate_archive": candidate_record,
        "candidate_member": candidate_member_record,
        "candidate_members": candidate_member_records,
        "candidate_zip_header": candidate_header,
        "candidate_zip_headers": candidate_headers,
        "candidate_zip_header_summary": candidate_header_summary,
        "selected_member_name": primary_member,
        "selected_member_names": target_members,
        "selection_scope": "all_members" if all_members else (
            "multi_member" if len(target_members) > 1 else "single_member"
        ),
        "selected_elision": {
            **elision_options,
            "selected_member_names": target_members,
            "source_archive_bytes": source_record["bytes"],
            "candidate_archive_bytes": candidate_record["bytes"],
            "saved_bytes": saved_bytes,
            "elided_header_bytes": (
                int(source_header_summary["total_elidable_header_bytes"])
                - int(candidate_header_summary["total_elidable_header_bytes"])
            ),
        },
        "receiver_verification": receiver_verification,
        "runtime_consumption_proof_path": (
            proof_out.as_posix() if proof_out is not None else receiver_verification.get("proof_path")
        ),
        "runtime_consumption_proof_write": (
            proof_write_result.__dict__ if proof_write_result is not None else None
        ),
        "receiver_contract_satisfied": (
            receiver_verification["receiver_contract_satisfied"] is True
        ),
        "readiness_blockers": readiness_blockers,
        "artifact_write": write_result.__dict__,
        **FALSE_AUTHORITY,
    }


def _packet_member_zip_header_elide_runtime_consumption_proof(
    *,
    source_archive: Mapping[str, Any],
    source_member: Mapping[str, Any],
    source_members: Sequence[Mapping[str, Any]],
    source_header: Mapping[str, Any],
    source_headers: Sequence[Mapping[str, Any]],
    source_header_summary: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
    candidate_member: Mapping[str, Any],
    candidate_members: Sequence[Mapping[str, Any]],
    candidate_header: Mapping[str, Any],
    candidate_headers: Sequence[Mapping[str, Any]],
    candidate_header_summary: Mapping[str, Any],
    selected_member_name: str,
    selected_member_names: Sequence[str],
    selected_elision: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    source_member_sha = _clean_str(source_member.get("sha256"))
    candidate_member_sha = _clean_str(candidate_member.get("sha256"))
    member_payload_identical = (
        source_member_sha is not None and candidate_member_sha == source_member_sha
    )
    source_by_name = {str(row.get("name")): row for row in source_members}
    candidate_by_name = {str(row.get("name")): row for row in candidate_members}
    member_proofs: list[dict[str, Any]] = []
    all_member_payloads_identical = bool(selected_member_names)
    all_member_compressed_streams_identical = bool(selected_member_names)
    for name in selected_member_names:
        source_row = source_by_name.get(str(name), {})
        candidate_row = candidate_by_name.get(str(name), {})
        source_sha = _clean_str(source_row.get("sha256"))
        candidate_sha = _clean_str(candidate_row.get("sha256"))
        source_compressed_bytes = source_row.get("zip_compressed_bytes")
        candidate_compressed_bytes = candidate_row.get("zip_compressed_bytes")
        source_compressed_sha = _clean_str(source_row.get("zip_compressed_sha256"))
        candidate_compressed_sha = _clean_str(candidate_row.get("zip_compressed_sha256"))
        payload_identical = source_sha is not None and candidate_sha == source_sha
        compressed_stream_identical = (
            source_compressed_sha is not None
            and candidate_compressed_sha == source_compressed_sha
        )
        all_member_payloads_identical = all_member_payloads_identical and payload_identical
        all_member_compressed_streams_identical = (
            all_member_compressed_streams_identical and compressed_stream_identical
        )
        member_proofs.append(
            {
                "member_name": str(name),
                "source_member_sha256": source_sha,
                "candidate_member_sha256": candidate_sha,
                "member_sha256": candidate_sha,
                "candidate_member_payload_identical_to_source": payload_identical,
                "source_zip_compressed_bytes": source_compressed_bytes,
                "candidate_zip_compressed_bytes": candidate_compressed_bytes,
                "candidate_zip_compressed_sha256": candidate_compressed_sha,
                "source_zip_compressed_sha256": source_compressed_sha,
                "candidate_compressed_stream_identical_to_source": (
                    compressed_stream_identical
                ),
                "candidate_compressed_stream_length_identical_to_source": (
                    source_compressed_bytes is not None
                    and candidate_compressed_bytes == source_compressed_bytes
                ),
                "passed": payload_identical and compressed_stream_identical,
            }
        )
    elided_header_bytes = (
        int(source_header_summary.get("total_elidable_header_bytes", 0))
        - int(candidate_header_summary.get("total_elidable_header_bytes", 0))
    )
    passed = (
        all_member_payloads_identical
        and all_member_compressed_streams_identical
        and elided_header_bytes >= 0
    )
    candidate_member_sha256s = {
        str(row.get("name")): row.get("sha256")
        for row in candidate_members
        if _clean_str(row.get("name")) and _clean_str(row.get("sha256"))
    }
    return {
        "schema": "family_agnostic_runtime_consumption_proof_v1",
        "proof_kind": "packet_member_zip_header_elide_payload_identity_receiver_proof.v1",
        "proof_scope": "zip_header_metadata_elision_with_member_payload_identity",
        "target_kind": PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        "materializer_id": PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER_ID,
        "receiver_contract_kind": "family_agnostic_packet_member_zip_header_elide",
        "receiver_contract_id": f"{PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND}.receiver.v1",
        "selected_member_name": selected_member_name,
        "selected_member_names": list(selected_member_names),
        "source_archive": dict(source_archive),
        "source_member": dict(source_member),
        "source_members": [dict(row) for row in source_members],
        "source_zip_header": dict(source_header),
        "source_zip_headers": [dict(row) for row in source_headers],
        "source_zip_header_summary": dict(source_header_summary),
        "candidate_archive": dict(candidate_archive),
        "candidate_member": dict(candidate_member),
        "candidate_members": [dict(row) for row in candidate_members],
        "candidate_zip_header": dict(candidate_header),
        "candidate_zip_headers": [dict(row) for row in candidate_headers],
        "candidate_zip_header_summary": dict(candidate_header_summary),
        "candidate_archive_sha256": candidate_archive.get("sha256"),
        "candidate_member_sha256": candidate_member_sha,
        "candidate_member_sha256s": candidate_member_sha256s,
        "member_sha256": candidate_member_sha,
        "source_member_sha256": source_member_sha,
        "candidate_member_payload_identical_to_source": member_payload_identical,
        "all_member_payloads_identical_to_source": all_member_payloads_identical,
        "all_member_compressed_streams_identical_to_source": (
            all_member_compressed_streams_identical
        ),
        "all_member_compressed_stream_lengths_identical_to_source": (
            all_member_compressed_streams_identical
        ),
        "selected_elision": dict(selected_elision),
        "header_elision_contract": dict(contract),
        "runtime_consumption_probe": {
            "schema": "packet_member_zip_header_elide_payload_identity_probe.v1",
            "passed": passed,
            "source_member_sha256": source_member_sha,
            "candidate_member_sha256": candidate_member_sha,
            "candidate_member_bytes": candidate_member.get("bytes"),
            "source_member_bytes": source_member.get("bytes"),
            "source_elidable_header_bytes": source_header.get("total_elidable_header_bytes"),
            "candidate_elidable_header_bytes": candidate_header.get("total_elidable_header_bytes"),
            "source_total_elidable_header_bytes": (
                source_header_summary.get("total_elidable_header_bytes")
            ),
            "candidate_total_elidable_header_bytes": (
                candidate_header_summary.get("total_elidable_header_bytes")
            ),
            "elided_header_bytes": elided_header_bytes,
            "member_proofs": member_proofs,
        },
        "receiver_contract_satisfied": passed,
        "runtime_consumption_proof_passed": passed,
        "passed": passed,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": int(selected_elision.get("saved_bytes", 0)) != 0,
    }


def materialize_archive_section_entropy_recode_candidate(
    *,
    archive_path: str | Path,
    section_manifest: str | Path | Mapping[str, Any],
    output_archive: str | Path,
    section_names: Sequence[str] = (),
    brotli_qualities: Sequence[int] = (9, 10, 11),
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None = None,
    runtime_consumption_proof_out: str | Path | None = None,
    repo_root: str | Path | None = None,
    allow_size_regression: bool = False,
    allow_overwrite: bool = False,
    expected_existing_output_sha256: str | None = None,
    expected_existing_runtime_consumption_proof_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Recode Brotli-decodable sections from a parser-section manifest."""

    repo = _repo(repo_root)
    if runtime_consumption_proof is not None and runtime_consumption_proof_out is not None:
        raise FamilyAgnosticMaterializerError(
            "runtime_consumption_proof and runtime_consumption_proof_out are mutually exclusive"
        )
    archive = _resolve_path(archive_path, repo=repo)
    output = _resolve_path(output_archive, repo=repo)
    proof_out = (
        _resolve_path(runtime_consumption_proof_out, repo=repo)
        if runtime_consumption_proof_out is not None
        else None
    )
    manifest = _require_mapping(section_manifest, repo=repo)
    target_member = _select_member_name(archive, explicit=None, manifest=manifest)
    member_payload = _zip_member_bytes(archive, target_member)
    sections = _section_records(manifest)
    if not sections:
        raise FamilyAgnosticMaterializerError("section_manifest contains no usable sections")
    selected = set(section_names)
    section_outputs: list[dict[str, Any]] = []
    replacements: dict[tuple[int, int], bytes] = {}
    for section in sections:
        name = str(section["name"])
        if selected and name not in selected:
            section_outputs.append({**section, "selected": False, "changed": False})
            continue
        payload = member_payload[int(section["offset"]): int(section["offset"]) + int(section["length"])]
        if sha256_bytes(payload) != str(section["sha256"]):
            section_outputs.append(
                {
                    **section,
                    "selected": True,
                    "changed": False,
                    "blockers": ["section_sha256_mismatch"],
                }
            )
            continue
        recode = _best_brotli_recode(payload, qualities=brotli_qualities)
        if recode is None:
            section_outputs.append(
                {
                    **section,
                    "selected": True,
                    "changed": False,
                    "blockers": ["section_not_brotli_decompressible"],
                }
            )
            continue
        replacements[(int(section["offset"]), int(section["length"]))] = recode["payload"]
        section_outputs.append(
            {
                **section,
                "selected": True,
                "changed": recode["sha256"] != str(section["sha256"]),
                "raw_payload_sha256": recode["raw_sha256"],
                "candidate_sha256": recode["sha256"],
                "candidate_length": recode["bytes"],
                "candidate_quality": recode["quality"],
                "saved_bytes": int(section["length"]) - int(recode["bytes"]),
            }
        )
    if selected:
        missing = sorted(selected.difference(str(section["name"]) for section in sections))
        if missing:
            raise FamilyAgnosticMaterializerError(
                "requested section_name not present in section_manifest: " + ", ".join(missing)
            )
    if not replacements:
        raise FamilyAgnosticMaterializerError("no selected Brotli section could be recoded")

    candidate_member_payload = _replace_ranges(member_payload, replacements)
    archive_payload = _zip_archive_bytes_with_replacement(
        archive,
        member_name=target_member,
        replacement=candidate_member_payload,
        compression=_source_zip_compression(archive, target_member),
        compresslevel=None,
    )
    source_record = _archive_record(archive)
    saved_bytes = int(source_record["bytes"]) - len(archive_payload)
    blockers = []
    if saved_bytes <= 0 and not allow_size_regression:
        blockers.append("candidate_not_rate_positive")
    changed_lengths = [
        row["name"]
        for row in section_outputs
        if row.get("changed") is True and row.get("candidate_length") != row.get("length")
    ]
    write_result = write_bytes_artifact(
        output,
        archive_payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_existing_output_sha256,
        min_free_bytes=min_free_bytes,
    )
    candidate_record = _archive_record(output)
    candidate_member = _member_record(output, target_member)
    source_member = _member_record(archive, target_member)
    proof_write_result = None
    runtime_proof_ref: str | Path | Mapping[str, Any] | None = runtime_consumption_proof
    if proof_out is not None:
        runtime_proof_payload = _archive_section_entropy_recode_runtime_consumption_proof(
            source_archive=source_record,
            source_member=source_member,
            candidate_archive=candidate_record,
            candidate_member=candidate_member,
            selected_member_name=target_member,
            source_member_payload=member_payload,
            candidate_member_payload=candidate_member_payload,
            section_outputs=section_outputs,
        )
        proof_write_result = write_json_artifact(
            proof_out,
            runtime_proof_payload,
            allow_overwrite=allow_overwrite,
            expected_existing_sha256=expected_existing_runtime_consumption_proof_sha256,
            min_free_bytes=min_free_bytes,
        )
        runtime_proof_ref = proof_out
    receiver_verification = verify_runtime_consumption_proof(
        runtime_consumption_proof=runtime_proof_ref,
        required_candidate_archive_sha256=candidate_record["sha256"],
        required_candidate_member_sha256=candidate_member["sha256"],
        repo_root=repo,
    )
    if (
        changed_lengths
        and receiver_verification.get("receiver_contract_satisfied") is not True
    ):
        blockers.append("section_length_changed_requires_runtime_consumption_proof")
    readiness_blockers = _readiness_blockers(
        blockers,
        receiver_verification,
        receiver_blocker="archive_section_entropy_recode_receiver_contract_not_satisfied",
    )
    return {
        "schema": ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA,
        "materializer_id": ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER_ID,
        "target_kind": ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        "receiver_contract_id": f"{ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND}.receiver.v1",
        "receiver_contract_kind": "family_agnostic_archive_section_entropy_recode",
        "byte_closed_candidate_emitted": True,
        "source_archive": source_record,
        "source_member": source_member,
        "candidate_archive": candidate_record,
        "candidate_member": candidate_member,
        "selected_member_name": target_member,
        "section_manifest_schema": manifest.get("schema"),
        "section_recode": {
            "selected_section_names": list(section_names),
            "changed_section_count": sum(1 for row in section_outputs if row.get("changed") is True),
            "changed_length_section_names": changed_lengths,
            "source_archive_bytes": source_record["bytes"],
            "candidate_archive_bytes": candidate_record["bytes"],
            "saved_bytes": saved_bytes,
        },
        "sections": section_outputs,
        "receiver_verification": receiver_verification,
        "receiver_contract_satisfied": (
            receiver_verification["receiver_contract_satisfied"] is True
        ),
        "runtime_consumption_proof_path": (
            proof_out.as_posix() if proof_out is not None else receiver_verification.get("proof_path")
        ),
        "runtime_consumption_proof_write": (
            proof_write_result.__dict__ if proof_write_result is not None else None
        ),
        "readiness_blockers": readiness_blockers,
        "artifact_write": write_result.__dict__,
        **FALSE_AUTHORITY,
    }


def _archive_section_entropy_recode_runtime_consumption_proof(
    *,
    source_archive: Mapping[str, Any],
    source_member: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
    candidate_member: Mapping[str, Any],
    selected_member_name: str,
    source_member_payload: bytes,
    candidate_member_payload: bytes,
    section_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    section_proofs = []
    cumulative_delta = 0
    passed = True
    for section in sorted(section_outputs, key=lambda row: int(row.get("offset", 0))):
        if section.get("selected") is not True:
            continue
        source_offset = int(section["offset"])
        source_length = int(section["length"])
        candidate_length = int(section.get("candidate_length", source_length))
        candidate_offset = source_offset + cumulative_delta
        source_payload = source_member_payload[source_offset:source_offset + source_length]
        candidate_payload = candidate_member_payload[
            candidate_offset: candidate_offset + candidate_length
        ]
        source_sha = sha256_bytes(source_payload)
        candidate_sha = sha256_bytes(candidate_payload)
        source_raw_sha = _brotli_raw_sha256(source_payload)
        candidate_raw_sha = _brotli_raw_sha256(candidate_payload)
        expected_source_sha = _clean_str(section.get("sha256"))
        expected_candidate_sha = _clean_str(section.get("candidate_sha256"))
        expected_raw_sha = _clean_str(section.get("raw_payload_sha256"))
        length_preserved = candidate_length == source_length
        section_passed = (
            source_raw_sha is not None
            and candidate_raw_sha is not None
            and source_raw_sha == candidate_raw_sha
            and length_preserved
            and (expected_source_sha is None or source_sha == expected_source_sha)
            and (expected_candidate_sha is None or candidate_sha == expected_candidate_sha)
            and (expected_raw_sha is None or candidate_raw_sha == expected_raw_sha)
        )
        passed = passed and section_passed
        section_proofs.append(
            {
                "name": section.get("name"),
                "index": section.get("index"),
                "source_offset": source_offset,
                "source_length": source_length,
                "candidate_offset": candidate_offset,
                "candidate_length": candidate_length,
                "source_section_sha256": source_sha,
                "candidate_section_sha256": candidate_sha,
                "source_raw_payload_sha256": source_raw_sha,
                "candidate_raw_payload_sha256": candidate_raw_sha,
                "raw_payload_identical": source_raw_sha == candidate_raw_sha,
                "section_length_preserved": length_preserved,
                "passed": section_passed,
            }
        )
        cumulative_delta += candidate_length - source_length
    if not section_proofs:
        passed = False
    return {
        "schema": "family_agnostic_runtime_consumption_proof_v1",
        "proof_kind": "archive_section_entropy_recode_raw_payload_identity_receiver_proof.v1",
        "proof_scope": "brotli_section_raw_payload_identity_after_entropy_recode",
        "target_kind": ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        "materializer_id": ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER_ID,
        "receiver_contract_kind": "family_agnostic_archive_section_entropy_recode",
        "receiver_contract_id": f"{ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND}.receiver.v1",
        "selected_member_name": selected_member_name,
        "source_archive": dict(source_archive),
        "source_member": dict(source_member),
        "candidate_archive": dict(candidate_archive),
        "candidate_member": dict(candidate_member),
        "candidate_archive_sha256": candidate_archive.get("sha256"),
        "candidate_member_sha256": candidate_member.get("sha256"),
        "member_sha256": candidate_member.get("sha256"),
        "source_member_sha256": source_member.get("sha256"),
        "section_proofs": section_proofs,
        "section_count": len(section_proofs),
        "all_selected_sections_raw_payload_identical": all(
            proof["raw_payload_identical"] is True for proof in section_proofs
        ),
        "all_selected_section_lengths_preserved": all(
            proof["section_length_preserved"] is True for proof in section_proofs
        ),
        "receiver_contract_satisfied": passed,
        "runtime_consumption_proof_passed": passed,
        "passed": passed,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
    }


def materialize_tensor_factorize_candidate(
    *,
    archive_path: str | Path,
    tensor_manifest: str | Path | Mapping[str, Any],
    factorization_contract: str | Path | Mapping[str, Any],
    output_archive: str | Path,
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None = None,
    runtime_consumption_proof_out: str | Path | None = None,
    repo_root: str | Path | None = None,
    allow_size_regression: bool = False,
    allow_overwrite: bool = False,
    expected_existing_output_sha256: str | None = None,
    expected_existing_runtime_consumption_proof_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Replace a NumPy tensor member with a deterministic low-rank NPZ packet."""

    import numpy as np

    repo = _repo(repo_root)
    if runtime_consumption_proof is not None and runtime_consumption_proof_out is not None:
        raise FamilyAgnosticMaterializerError(
            "runtime_consumption_proof and runtime_consumption_proof_out are mutually exclusive"
        )
    archive = _resolve_path(archive_path, repo=repo)
    output = _resolve_path(output_archive, repo=repo)
    proof_out = (
        _resolve_path(runtime_consumption_proof_out, repo=repo)
        if runtime_consumption_proof_out is not None
        else None
    )
    manifest = _require_mapping(tensor_manifest, repo=repo)
    contract = _require_mapping(factorization_contract, repo=repo)
    member = _select_member_name(archive, explicit=_clean_str(manifest.get("member_name")), manifest=manifest)
    rank = _positive_int(contract.get("rank"), field="factorization_contract.rank")
    tensor_payload = _zip_member_bytes(archive, member)
    with io.BytesIO(tensor_payload) as handle:
        tensor = np.load(handle, allow_pickle=False)
    if not hasattr(tensor, "ndim") or tensor.ndim != 2:
        raise FamilyAgnosticMaterializerError("tensor_factorize_v1 currently requires a 2D .npy member")
    if rank > min(tensor.shape):
        raise FamilyAgnosticMaterializerError("factorization rank exceeds tensor shape")
    matrix = np.asarray(tensor, dtype=np.float32)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    u_r = u[:, :rank].astype(np.float32)
    s_r = s[:rank].astype(np.float32)
    vt_r = vt[:rank, :].astype(np.float32)
    factor_payload = _npz_bytes(
        {
            "schema": TENSOR_FACTORIZE_SCHEMA,
            "source_shape": list(tensor.shape),
            "source_dtype": str(tensor.dtype),
            "rank": rank,
        },
        u=u_r,
        s=s_r,
        vt=vt_r,
    )
    archive_payload = _zip_archive_bytes_with_replacement(
        archive,
        member_name=member,
        replacement=factor_payload,
        compression=zipfile.ZIP_STORED,
        compresslevel=None,
    )
    source_record = _archive_record(archive)
    saved_bytes = int(source_record["bytes"]) - len(archive_payload)
    blockers: list[str] = []
    if saved_bytes <= 0 and not allow_size_regression:
        blockers.append("candidate_not_rate_positive")
    write_result = write_bytes_artifact(
        output,
        archive_payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_existing_output_sha256,
        min_free_bytes=min_free_bytes,
    )
    candidate_record = _archive_record(output)
    candidate_member = _member_record(output, member)
    source_member = _member_record(archive, member)
    proof_write_result = None
    runtime_proof_ref: str | Path | Mapping[str, Any] | None = runtime_consumption_proof
    if proof_out is not None:
        runtime_proof_payload = _tensor_factorize_runtime_consumption_proof(
            source_archive=source_record,
            source_member=source_member,
            candidate_archive=candidate_record,
            candidate_member=candidate_member,
            selected_member_name=member,
            source_tensor=matrix,
            source_dtype=str(tensor.dtype),
            rank=rank,
            u=u_r,
            s=s_r,
            vt=vt_r,
            factor_payload_bytes=len(factor_payload),
            contract=contract,
        )
        proof_write_result = write_json_artifact(
            proof_out,
            runtime_proof_payload,
            allow_overwrite=allow_overwrite,
            expected_existing_sha256=expected_existing_runtime_consumption_proof_sha256,
            min_free_bytes=min_free_bytes,
        )
        runtime_proof_ref = proof_out
    receiver_verification = verify_runtime_consumption_proof(
        runtime_consumption_proof=runtime_proof_ref,
        required_candidate_archive_sha256=candidate_record["sha256"],
        required_candidate_member_sha256=candidate_member["sha256"],
        repo_root=repo,
    )
    if receiver_verification.get("receiver_contract_satisfied") is not True:
        blockers.append("tensor_factorized_payload_requires_cooperative_receiver")
    readiness_blockers = _readiness_blockers(
        blockers,
        receiver_verification,
        receiver_blocker="tensor_factorize_receiver_contract_not_satisfied",
    )
    return {
        "schema": TENSOR_FACTORIZE_SCHEMA,
        "materializer_id": TENSOR_FACTORIZE_MATERIALIZER_ID,
        "target_kind": TENSOR_FACTORIZE_TARGET_KIND,
        "receiver_contract_id": f"{TENSOR_FACTORIZE_TARGET_KIND}.receiver.v1",
        "receiver_contract_kind": "family_agnostic_tensor_factorize",
        "byte_closed_candidate_emitted": True,
        "source_archive": source_record,
        "source_member": source_member,
        "candidate_archive": candidate_record,
        "candidate_member": candidate_member,
        "selected_member_name": member,
        "factorization": {
            "rank": rank,
            "source_shape": list(tensor.shape),
            "source_dtype": str(tensor.dtype),
            "factor_payload_bytes": len(factor_payload),
            "source_archive_bytes": source_record["bytes"],
            "candidate_archive_bytes": candidate_record["bytes"],
            "saved_bytes": saved_bytes,
        },
        "receiver_verification": receiver_verification,
        "receiver_contract_satisfied": (
            receiver_verification["receiver_contract_satisfied"] is True
        ),
        "runtime_consumption_proof_path": (
            proof_out.as_posix() if proof_out is not None else receiver_verification.get("proof_path")
        ),
        "runtime_consumption_proof_write": (
            proof_write_result.__dict__ if proof_write_result is not None else None
        ),
        "readiness_blockers": readiness_blockers,
        "artifact_write": write_result.__dict__,
        **FALSE_AUTHORITY,
    }


def _tensor_factorize_runtime_consumption_proof(
    *,
    source_archive: Mapping[str, Any],
    source_member: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
    candidate_member: Mapping[str, Any],
    selected_member_name: str,
    source_tensor: Any,
    source_dtype: str,
    rank: int,
    u: Any,
    s: Any,
    vt: Any,
    factor_payload_bytes: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    reconstruction = (np.asarray(u, dtype=np.float32) * np.asarray(s, dtype=np.float32)) @ np.asarray(
        vt,
        dtype=np.float32,
    )
    source = np.asarray(source_tensor, dtype=np.float32)
    delta = reconstruction - source
    abs_delta = np.abs(delta)
    max_abs_error = float(abs_delta.max(initial=0.0))
    rmse = float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))
    denom = np.maximum(np.abs(source), np.finfo(np.float32).eps)
    max_relative_error = float(np.max(abs_delta / denom, initial=0.0))
    max_abs_tolerance = _optional_nonnegative_float(
        contract.get("max_abs_error_tolerance"),
        field="factorization_contract.max_abs_error_tolerance",
    )
    max_relative_tolerance = _optional_nonnegative_float(
        contract.get("max_relative_error_tolerance"),
        field="factorization_contract.max_relative_error_tolerance",
    )
    tolerance_declared = max_abs_tolerance is not None or max_relative_tolerance is not None
    cooperative_receiver_id = _mapping_string_any(
        contract,
        ("cooperative_receiver_id", "receiver_id", "runtime_adapter_id"),
    )
    receiver_adapter_kind = _mapping_string_any(
        contract,
        ("receiver_adapter_kind", "cooperative_receiver_adapter_kind", "runtime_adapter_kind"),
    )
    if max_abs_tolerance is None:
        abs_passed = True if tolerance_declared else max_abs_error == 0.0
    else:
        abs_passed = max_abs_error <= max_abs_tolerance
    if max_relative_tolerance is None:
        relative_passed = True if tolerance_declared else max_relative_error == 0.0
    else:
        relative_passed = max_relative_error <= max_relative_tolerance
    finite = (
        math.isfinite(max_abs_error)
        and math.isfinite(rmse)
        and math.isfinite(max_relative_error)
    )
    receiver_declared = cooperative_receiver_id is not None and receiver_adapter_kind is not None
    passed = finite and abs_passed and relative_passed and receiver_declared
    return {
        "schema": "family_agnostic_runtime_consumption_proof_v1",
        "proof_kind": "tensor_factorize_cooperative_receiver_reconstruction_proof.v1",
        "proof_scope": "low_rank_npz_packet_reconstructs_source_tensor_for_cooperative_receiver",
        "target_kind": TENSOR_FACTORIZE_TARGET_KIND,
        "materializer_id": TENSOR_FACTORIZE_MATERIALIZER_ID,
        "receiver_contract_kind": "family_agnostic_tensor_factorize",
        "receiver_contract_id": f"{TENSOR_FACTORIZE_TARGET_KIND}.receiver.v1",
        "selected_member_name": selected_member_name,
        "source_archive": dict(source_archive),
        "source_member": dict(source_member),
        "candidate_archive": dict(candidate_archive),
        "candidate_member": dict(candidate_member),
        "candidate_archive_sha256": candidate_archive.get("sha256"),
        "candidate_member_sha256": candidate_member.get("sha256"),
        "member_sha256": candidate_member.get("sha256"),
        "source_member_sha256": source_member.get("sha256"),
        "factorization": {
            "rank": rank,
            "source_shape": list(source.shape),
            "source_dtype": source_dtype,
            "factor_payload_bytes": factor_payload_bytes,
        },
        "cooperative_receiver": {
            "cooperative_receiver_id": cooperative_receiver_id,
            "receiver_adapter_kind": receiver_adapter_kind,
            "declared": receiver_declared,
            "reconstruction_formula": "float32((u * s) @ vt)",
        },
        "runtime_consumption_probe": {
            "schema": "tensor_factorize_reconstruction_probe.v1",
            "passed": passed,
            "finite": finite,
            "max_abs_error": max_abs_error,
            "max_abs_error_tolerance": max_abs_tolerance,
            "max_abs_error_passed": abs_passed,
            "rmse": rmse,
            "max_relative_error": max_relative_error,
            "max_relative_error_tolerance": max_relative_tolerance,
            "max_relative_error_passed": relative_passed,
            "cooperative_receiver_declared": receiver_declared,
        },
        "receiver_contract_satisfied": passed,
        "runtime_consumption_proof_passed": passed,
        "passed": passed,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
    }


def verify_runtime_consumption_proof(
    *,
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None,
    required_candidate_archive_sha256: str | None = None,
    required_candidate_member_sha256: str | None = None,
    required_candidate_member_sha256s: Mapping[str, str] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the generic receiver/runtime proof used by these materializers."""

    repo = _repo(repo_root)
    if runtime_consumption_proof is None:
        return {
            "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
            "receiver_contract_satisfied": False,
            "proof_present": False,
            "proof_path": None,
            "blockers": ["runtime_consumption_proof_missing"],
        }
    proof_path = (
        _resolve_path(runtime_consumption_proof, repo=repo).as_posix()
        if isinstance(runtime_consumption_proof, (str, Path))
        else None
    )
    proof = _require_mapping(runtime_consumption_proof, repo=repo)
    blockers: list[str] = []
    try:
        require_no_truthy_authority_fields(
            proof,
            context="family_agnostic_runtime_consumption_proof",
        )
    except ValueError as exc:
        blockers.append(str(exc))
    if not _proof_passed(proof):
        blockers.append("runtime_consumption_proof_not_passed")
    archive_sha = _nested_clean_str(proof, ("candidate_archive", "sha256"))
    archive_sha = archive_sha or _clean_str(proof.get("candidate_archive_sha256"))
    if required_candidate_archive_sha256:
        if not archive_sha:
            blockers.append("runtime_consumption_proof_candidate_archive_sha_missing")
        elif archive_sha != required_candidate_archive_sha256:
            blockers.append("runtime_consumption_proof_candidate_archive_sha_mismatch")
    member_sha = _nested_clean_str(proof, ("candidate_member", "sha256"))
    member_sha = member_sha or _clean_str(proof.get("candidate_member_sha256"))
    member_sha = member_sha or _clean_str(proof.get("member_sha256"))
    if required_candidate_member_sha256:
        if not member_sha:
            blockers.append("runtime_consumption_proof_candidate_member_sha_missing")
        elif member_sha != required_candidate_member_sha256:
            blockers.append("runtime_consumption_proof_candidate_member_sha_mismatch")
    member_sha_map = _proof_candidate_member_sha256s(proof)
    if required_candidate_member_sha256s:
        for name, required_sha in required_candidate_member_sha256s.items():
            candidate_sha = member_sha_map.get(str(name))
            if not candidate_sha:
                blockers.append(
                    f"runtime_consumption_proof_candidate_member_sha_missing:{name}"
                )
            elif candidate_sha != str(required_sha):
                blockers.append(
                    f"runtime_consumption_proof_candidate_member_sha_mismatch:{name}"
                )
    return {
        "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
        "receiver_contract_satisfied": not blockers,
        "proof_present": True,
        "proof_path": proof_path,
        "proof_schema": proof.get("schema"),
        "candidate_archive_sha256": archive_sha,
        "candidate_member_sha256": member_sha,
        "candidate_member_sha256s": member_sha_map,
        "blockers": ordered_unique(blockers),
    }


def _proof_candidate_member_sha256s(proof: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    raw_map = proof.get("candidate_member_sha256s")
    if isinstance(raw_map, Mapping):
        for name, value in raw_map.items():
            sha = _clean_str(value)
            if sha:
                out[str(name)] = sha
    rows = proof.get("candidate_members")
    if isinstance(rows, Sequence) and not isinstance(rows, (bytes, bytearray, str)):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            name = _clean_str(row.get("name"))
            sha = _clean_str(row.get("sha256"))
            if name and sha:
                out[name] = sha
    probe = proof.get("runtime_consumption_probe")
    member_proofs = probe.get("member_proofs") if isinstance(probe, Mapping) else None
    if isinstance(member_proofs, Sequence) and not isinstance(member_proofs, (bytes, bytearray, str)):
        for row in member_proofs:
            if not isinstance(row, Mapping):
                continue
            name = _clean_str(row.get("member_name"))
            sha = _clean_str(row.get("candidate_member_sha256"))
            if name and sha:
                out[name] = sha
    return out


def _mapping_string_any(mapping: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = _clean_str(mapping.get(key))
        if value is not None:
            return value
    return None


def _repo(repo_root: str | Path | None) -> Path:
    return Path(repo_root).resolve(strict=False) if repo_root is not None else Path.cwd()


def _resolve_path(path: str | Path, *, repo: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    return candidate.resolve(strict=False)


def _load_mapping(value: str | Path | Mapping[str, Any] | None, *, repo: Path) -> dict[str, Any]:
    if value is None:
        return {}
    return _require_mapping(value, repo=repo)


def _require_mapping(value: str | Path | Mapping[str, Any], *, repo: Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return dict(read_json(_resolve_path(value, repo=repo)))


def _clean_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _nested_clean_str(mapping: Mapping[str, Any], path: Sequence[str]) -> str | None:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return _clean_str(value)


def _proof_passed(proof: Mapping[str, Any]) -> bool:
    if proof.get("receiver_contract_satisfied") is True:
        return True
    if proof.get("runtime_consumption_proof_passed") is True:
        return True
    if proof.get("passed") is True:
        return True
    probe = proof.get("runtime_consumption_probe")
    return isinstance(probe, Mapping) and probe.get("passed") is True


def _brotli_raw_sha256(payload: bytes) -> str | None:
    try:
        return sha256_bytes(brotli.decompress(payload))
    except brotli.error:
        return None


def _readiness_blockers(
    blockers: Sequence[str],
    receiver_verification: Mapping[str, Any],
    *,
    receiver_blocker: str,
) -> list[str]:
    out = [str(blocker) for blocker in blockers]
    out.extend(str(blocker) for blocker in receiver_verification.get("blockers") or [])
    if receiver_verification.get("receiver_contract_satisfied") is not True:
        out.append(receiver_blocker)
    return ordered_unique(out)


def _select_member_name(
    archive: Path,
    *,
    explicit: str | None,
    manifest: Mapping[str, Any],
) -> str:
    if _clean_str(explicit):
        return str(explicit)
    for key in ("member_name", "archive_member_name"):
        value = _clean_str(manifest.get(key))
        if value:
            return value
    member = manifest.get("member")
    if isinstance(member, Mapping):
        value = _clean_str(member.get("name"))
        if value:
            return value
    members = _zip_member_names(archive)
    if len(members) != 1:
        raise FamilyAgnosticMaterializerError(
            "member_name is required when archive does not have exactly one member"
        )
    return members[0]


def _select_member_names(
    archive: Path,
    *,
    explicit: str | None,
    explicit_many: Sequence[str],
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    all_members: bool,
) -> list[str]:
    members = _zip_member_names(archive)
    if not members:
        raise FamilyAgnosticMaterializerError("archive contains no non-directory members")
    if all_members or _contract_selects_all_members(contract) or _manifest_selects_all_members(manifest):
        return _require_unique_existing_member_names(archive, members)
    explicit_names = _string_items(explicit_many)
    if _clean_str(explicit):
        explicit_names.insert(0, str(explicit).strip())
    if explicit_names:
        return _require_unique_existing_member_names(archive, explicit_names)
    manifest_names = _manifest_member_names(manifest)
    if manifest_names:
        return _require_unique_existing_member_names(archive, manifest_names)
    return [_select_member_name(archive, explicit=None, manifest=manifest)]


def _contract_selects_all_members(contract: Mapping[str, Any]) -> bool:
    for key in ("all_members", "select_all_members", "zip_header_elide_all_members"):
        if contract.get(key) is True:
            return True
    selection = _clean_str(contract.get("member_selection"))
    return selection in {"all", "*", "all_members"}


def _manifest_selects_all_members(manifest: Mapping[str, Any]) -> bool:
    for key in ("all_members", "select_all_members", "zip_header_elide_all_members"):
        if manifest.get(key) is True:
            return True
    selection = _clean_str(manifest.get("member_selection"))
    return selection in {"all", "*", "all_members"}


def _manifest_member_names(manifest: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("member_names", "archive_member_names", "packet_member_names"):
        names.extend(_string_items(manifest.get(key)))
    members = manifest.get("members")
    if isinstance(members, Sequence) and not isinstance(members, (bytes, bytearray, str)):
        for row in members:
            if isinstance(row, Mapping):
                name = _clean_str(row.get("name"))
                if name:
                    names.append(name)
            else:
                names.extend(_string_items(row))
    return ordered_unique(names)


def _string_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return ordered_unique(str(item).strip() for item in value if str(item).strip())
    return []


def _require_unique_existing_member_names(archive: Path, names: Sequence[str]) -> list[str]:
    requested = ordered_unique(str(name).strip() for name in names if str(name).strip())
    if not requested:
        raise FamilyAgnosticMaterializerError("at least one ZIP member must be selected")
    available = _zip_member_names(archive)
    counts: dict[str, int] = {}
    for name in available:
        counts[name] = counts.get(name, 0) + 1
    missing = [name for name in requested if name not in counts]
    if missing:
        raise FamilyAgnosticMaterializerError(
            "selected ZIP member not found: " + ", ".join(missing)
        )
    duplicate_selected = [name for name in requested if counts.get(name, 0) != 1]
    if duplicate_selected:
        raise FamilyAgnosticMaterializerError(
            "zip header elision requires selected member names to be unique: "
            + ", ".join(duplicate_selected)
        )
    return list(requested)


def _zip_member_names(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive, "r") as zf:
        return [info.filename for info in zf.infolist() if not info.is_dir()]


def _zip_member_bytes(archive: Path, member_name: str) -> bytes:
    with zipfile.ZipFile(archive, "r") as zf:
        try:
            return zf.read(member_name)
        except KeyError as exc:
            raise FamilyAgnosticMaterializerError(
                f"archive member not found: {member_name}"
            ) from exc


def _source_zip_compression(archive: Path, member_name: str) -> int:
    with zipfile.ZipFile(archive, "r") as zf:
        return zf.getinfo(member_name).compress_type


def _zip_archive_bytes_with_replacement(
    archive: Path,
    *,
    member_name: str,
    replacement: bytes,
    compression: int,
    compresslevel: int | None,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(output, "w") as target:
        for info in source.infolist():
            if info.is_dir():
                target.mkdir(_copy_zip_info(info))
                continue
            payload = replacement if info.filename == member_name else source.read(info.filename)
            out_info = _copy_zip_info(info)
            if info.filename == member_name:
                out_info.compress_type = compression
            kwargs = {}
            if out_info.compress_type == zipfile.ZIP_DEFLATED and compresslevel is not None:
                kwargs["compresslevel"] = compresslevel
            target.writestr(out_info, payload, **kwargs)
    return output.getvalue()


def _zip_archive_bytes_with_header_elision(
    archive: Path,
    *,
    member_names: Sequence[str],
    strip_member_extra: bool,
    strip_member_comment: bool,
    strip_archive_comment: bool,
) -> bytes:
    raw = archive.read_bytes()
    with zipfile.ZipFile(archive, "r") as source:
        infos = source.infolist()
        selected_names = set(_require_unique_existing_member_names(archive, member_names))
        archive_comment = b"" if strip_archive_comment else (source.comment or b"")

    if len(infos) > 0xFFFF:
        raise FamilyAgnosticMaterializerError("zip header elision does not support ZIP64 entry counts")

    local_chunks: list[bytes] = []
    central_chunks: list[bytes] = []
    cursor = 0
    for info in infos:
        selected = info.filename in selected_names
        parts = _raw_zip_member_parts(raw, info)
        local_extra = (
            b""
            if selected and strip_member_extra
            else parts["local_extra"]
        )
        member_offset = cursor
        local_header = struct.pack(
            "<IHHHHHIIIHH",
            0x0403_4B50,
            int(parts["version_needed"]),
            int(parts["flag_bits"]),
            int(parts["compress_type"]),
            int(parts["mod_time"]),
            int(parts["mod_date"]),
            int(parts["crc32"]),
            int(parts["compressed_bytes"]),
            int(parts["uncompressed_bytes"]),
            len(parts["name_bytes"]),
            len(local_extra),
        )
        local_chunk = local_header + parts["name_bytes"] + local_extra + parts["compressed_payload"]
        local_chunks.append(local_chunk)
        cursor += len(local_chunk)

        central_extra = b"" if selected and strip_member_extra else (info.extra or b"")
        central_comment = (
            b""
            if selected and strip_member_comment
            else (getattr(info, "comment", b"") or b"")
        )
        central_chunks.append(
            _zip_central_directory_header(
                info,
                name_bytes=parts["name_bytes"],
                extra=central_extra,
                comment=central_comment,
                local_header_offset=member_offset,
            )
        )

    central_directory_offset = cursor
    central_directory = b"".join(central_chunks)
    end_record = struct.pack(
        "<IHHHHIIH",
        0x0605_4B50,
        0,
        0,
        len(infos),
        len(infos),
        len(central_directory),
        central_directory_offset,
        len(archive_comment),
    )
    return b"".join(local_chunks) + central_directory + end_record + archive_comment


def _raw_zip_member_parts(raw: bytes, info: zipfile.ZipInfo) -> dict[str, Any]:
    if info.flag_bits & 0x08:
        raise FamilyAgnosticMaterializerError(
            f"zip header elision does not support data descriptors: {info.filename}"
        )
    if (
        info.file_size >= 0xFFFF_FFFF
        or info.compress_size >= 0xFFFF_FFFF
        or info.header_offset >= 0xFFFF_FFFF
    ):
        raise FamilyAgnosticMaterializerError(
            f"zip header elision does not support ZIP64 members: {info.filename}"
        )
    offset = int(info.header_offset)
    if offset < 0 or offset + 30 > len(raw):
        raise FamilyAgnosticMaterializerError(
            f"local header offset out of range for {info.filename}: {offset}"
        )
    (
        signature,
        version_needed,
        flag_bits,
        compress_type,
        mod_time,
        mod_date,
        crc32,
        compressed_bytes,
        uncompressed_bytes,
        name_len,
        extra_len,
    ) = struct.unpack_from("<IHHHHHIIIHH", raw, offset)
    if signature != 0x0403_4B50:
        raise FamilyAgnosticMaterializerError(
            f"bad local ZIP header signature for {info.filename}: offset={offset}"
        )
    if int(flag_bits) & 0x08:
        raise FamilyAgnosticMaterializerError(
            f"zip header elision does not support data descriptors: {info.filename}"
        )
    if int(compressed_bytes) != int(info.compress_size):
        raise FamilyAgnosticMaterializerError(
            f"local/central compressed size mismatch for {info.filename}"
        )
    if int(uncompressed_bytes) != int(info.file_size):
        raise FamilyAgnosticMaterializerError(
            f"local/central uncompressed size mismatch for {info.filename}"
        )
    name_start = offset + 30
    name_end = name_start + int(name_len)
    extra_end = name_end + int(extra_len)
    payload_end = extra_end + int(compressed_bytes)
    if name_end > len(raw) or extra_end > len(raw) or payload_end > len(raw):
        raise FamilyAgnosticMaterializerError(
            f"local ZIP member bounds out of range for {info.filename}"
        )
    return {
        "version_needed": int(version_needed),
        "flag_bits": int(flag_bits),
        "compress_type": int(compress_type),
        "mod_time": int(mod_time),
        "mod_date": int(mod_date),
        "crc32": int(crc32),
        "compressed_bytes": int(compressed_bytes),
        "uncompressed_bytes": int(uncompressed_bytes),
        "name_bytes": raw[name_start:name_end],
        "local_extra": raw[name_end:extra_end],
        "compressed_payload": raw[extra_end:payload_end],
    }


def _zip_central_directory_header(
    info: zipfile.ZipInfo,
    *,
    name_bytes: bytes,
    extra: bytes,
    comment: bytes,
    local_header_offset: int,
) -> bytes:
    if local_header_offset >= 0xFFFF_FFFF:
        raise FamilyAgnosticMaterializerError(
            f"zip header elision does not support ZIP64 offsets: {info.filename}"
        )
    version_made_by = (int(info.create_system) << 8) | int(info.create_version)
    central = struct.pack(
        "<IHHHHHHIIIHHHHHII",
        0x0201_4B50,
        version_made_by,
        int(info.extract_version),
        int(info.flag_bits),
        int(info.compress_type),
        _dos_time(info),
        _dos_date(info),
        int(info.CRC),
        int(info.compress_size),
        int(info.file_size),
        len(name_bytes),
        len(extra),
        len(comment),
        0,
        int(info.internal_attr),
        int(info.external_attr),
        int(local_header_offset),
    )
    return central + name_bytes + extra + comment


def _dos_time(info: zipfile.ZipInfo) -> int:
    _year, _month, _day, hour, minute, second = info.date_time
    return (int(hour) << 11) | (int(minute) << 5) | (int(second) // 2)


def _dos_date(info: zipfile.ZipInfo) -> int:
    year, month, day, _hour, _minute, _second = info.date_time
    return ((int(year) - 1980) << 9) | (int(month) << 5) | int(day)


def _copy_zip_info(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    copied = zipfile.ZipInfo(info.filename, info.date_time)
    copied.comment = info.comment
    copied.extra = info.extra
    copied.internal_attr = info.internal_attr
    copied.external_attr = info.external_attr
    copied.create_system = info.create_system
    copied.extract_version = info.extract_version
    copied.create_version = info.create_version
    copied.flag_bits = info.flag_bits
    copied.compress_type = info.compress_type
    return copied


def _archive_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _member_record(archive: Path, member_name: str, *, payload: bytes | None = None) -> dict[str, Any]:
    member_payload = _zip_member_bytes(archive, member_name) if payload is None else payload
    with zipfile.ZipFile(archive, "r") as zf:
        info = zf.getinfo(member_name)
        compress_type = info.compress_type
        compress_size = info.compress_size
    compressed_payload_sha256 = _zip_member_compressed_payload_sha256(archive, member_name)
    return {
        "name": member_name,
        "bytes": len(member_payload),
        "sha256": sha256_bytes(member_payload),
        "zip_compression_method": _compression_method_name(compress_type),
        "zip_compressed_bytes": compress_size,
        "zip_compressed_sha256": compressed_payload_sha256,
        "zip_compressed_payload_sha256": compressed_payload_sha256,
    }


def _zip_member_compressed_payload_sha256(archive: Path, member_name: str) -> str | None:
    try:
        raw = archive.read_bytes()
        with zipfile.ZipFile(archive, "r") as zf:
            info = zf.getinfo(member_name)
        return sha256_bytes(_raw_zip_member_parts(raw, info)["compressed_payload"])
    except (OSError, KeyError, FamilyAgnosticMaterializerError, struct.error):
        return None


def _zip_header_record(archive: Path, member_name: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive, "r") as zf:
        info = zf.getinfo(member_name)
        member_extra = info.extra or b""
        member_comment = getattr(info, "comment", b"") or b""
        archive_comment = zf.comment or b""
        return {
            "member_name": member_name,
            "member_filename_bytes": len(member_name.encode("utf-8")),
            "member_extra_bytes": len(member_extra),
            "member_comment_bytes": len(member_comment),
            "archive_comment_bytes": len(archive_comment),
            "total_elidable_header_bytes": (
                len(member_extra) + len(member_comment) + len(archive_comment)
            ),
            "header_offset": info.header_offset,
            "flag_bits": info.flag_bits,
            "create_system": info.create_system,
            "create_version": info.create_version,
            "extract_version": info.extract_version,
            "zip_compression_method": _compression_method_name(info.compress_type),
            "zip_compressed_bytes": info.compress_size,
        }


def _zip_header_summary(archive: Path, member_names: Sequence[str]) -> dict[str, Any]:
    selected = _require_unique_existing_member_names(archive, member_names)
    headers = [_zip_header_record(archive, name) for name in selected]
    with zipfile.ZipFile(archive, "r") as zf:
        archive_comment_bytes = len(zf.comment or b"")
    member_extra_bytes = sum(int(row["member_extra_bytes"]) for row in headers)
    member_comment_bytes = sum(int(row["member_comment_bytes"]) for row in headers)
    return {
        "member_count": len(selected),
        "member_names": selected,
        "member_extra_bytes": member_extra_bytes,
        "member_comment_bytes": member_comment_bytes,
        "archive_comment_bytes": archive_comment_bytes,
        "total_elidable_header_bytes": (
            member_extra_bytes + member_comment_bytes + archive_comment_bytes
        ),
    }


def _zip_header_elision_options(contract: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "strip_member_extra": _bool_contract_any(
            contract,
            ("strip_member_extra", "elide_member_extra", "strip_extra"),
            default=True,
        ),
        "strip_member_comment": _bool_contract_any(
            contract,
            ("strip_member_comment", "elide_member_comment", "strip_comment"),
            default=True,
        ),
        "strip_archive_comment": _bool_contract_any(
            contract,
            ("strip_archive_comment", "elide_archive_comment"),
            default=True,
        ),
    }


def _bool_contract_any(
    contract: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: bool,
) -> bool:
    for key in keys:
        value = contract.get(key)
        if isinstance(value, bool):
            return value
    return default


def _compression_method_name(method: int) -> str:
    if method == zipfile.ZIP_STORED:
        return "stored"
    if method == zipfile.ZIP_DEFLATED:
        return "deflated"
    return str(method)


def _normalized_compression_methods(values: Sequence[str]) -> tuple[int, ...]:
    methods: list[int] = []
    for value in values:
        normalized = str(value).strip().lower()
        if normalized in {"store", "stored", "zip_stored"}:
            methods.append(zipfile.ZIP_STORED)
        elif normalized in {"deflate", "deflated", "zip_deflated"}:
            methods.append(zipfile.ZIP_DEFLATED)
        else:
            raise FamilyAgnosticMaterializerError(f"unsupported zip compression method: {value}")
    return _ordered_unique_ints(methods)


def _ordered_unique_ints(values: Iterable[int]) -> tuple[int, ...]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        parsed = int(value)
        if parsed not in seen:
            out.append(parsed)
            seen.add(parsed)
    return tuple(out)


def _section_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    sections = manifest.get("sections")
    if not isinstance(sections, list):
        return []
    out: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        if not isinstance(section, Mapping):
            continue
        try:
            offset = int(section["offset"])
            length = int(section["length"])
        except (KeyError, TypeError, ValueError):
            continue
        if offset < 0 or length < 1:
            continue
        sha = _clean_str(section.get("sha256"))
        if sha is None:
            continue
        out.append(
            {
                "name": str(section.get("name") or f"section_{index:04d}"),
                "index": int(section.get("index", index)),
                "offset": offset,
                "length": length,
                "sha256": sha,
                "optimization_role": section.get("optimization_role"),
            }
        )
    return out


def _best_brotli_recode(payload: bytes, *, qualities: Sequence[int]) -> dict[str, Any] | None:
    try:
        raw = brotli.decompress(payload)
    except brotli.error:
        return None
    trials = []
    for quality in _ordered_unique_ints(int(item) for item in qualities):
        candidate = brotli.compress(raw, quality=quality)
        trials.append((len(candidate), quality, candidate))
    if not trials:
        return None
    _size, quality, candidate = min(trials, key=lambda item: (item[0], item[1]))
    return {
        "payload": candidate,
        "bytes": len(candidate),
        "sha256": sha256_bytes(candidate),
        "quality": quality,
        "raw_sha256": sha256_bytes(raw),
    }


def _replace_ranges(payload: bytes, replacements: Mapping[tuple[int, int], bytes]) -> bytes:
    pieces: list[bytes] = []
    cursor = 0
    for offset, length in sorted(replacements):
        if offset < cursor:
            raise FamilyAgnosticMaterializerError("section ranges overlap")
        pieces.append(payload[cursor:offset])
        pieces.append(replacements[(offset, length)])
        cursor = offset + length
    pieces.append(payload[cursor:])
    return b"".join(pieces)


def _positive_int(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise FamilyAgnosticMaterializerError(f"{field} must be an integer") from exc
    if parsed < 1:
        raise FamilyAgnosticMaterializerError(f"{field} must be >= 1")
    return parsed


def _optional_nonnegative_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FamilyAgnosticMaterializerError(f"{field} must be a finite number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise FamilyAgnosticMaterializerError(f"{field} must be a finite nonnegative number")
    return parsed


def _npz_bytes(metadata: Mapping[str, Any], **arrays: Any) -> bytes:
    import numpy as np

    output = io.BytesIO()
    np.savez_compressed(output, metadata=str(dict(metadata)), **arrays)
    return output.getvalue()
