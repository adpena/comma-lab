# SPDX-License-Identifier: MIT
"""Family-agnostic byte-shaving materializers.

These materializers are intentionally conservative: they can emit byte-closed
candidate archives for HNeRV, HNeRV bolt-ons, broader NeRV-family packets, and
non-NeRV ZIP/tensor archives, but they never claim score or exact-eval
readiness without an explicit runtime-consumption proof.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import struct
import zipfile
import zlib
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
PACKET_MEMBER_MERGE_SCHEMA = "packet_member_merge_candidate.v1"
RENDERER_PAYLOAD_DFL1_SCHEMA = "renderer_payload_dfl1_candidate.v1"
PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA = "packet_member_zip_header_elide_candidate.v1"
TENSOR_FACTORIZE_SCHEMA = "tensor_factorize_candidate.v1"
ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND = "archive_section_entropy_recode_v1"
PACKET_MEMBER_RECOMPRESS_TARGET_KIND = "packet_member_recompress_v1"
PACKET_MEMBER_MERGE_TARGET_KIND = "packet_member_merge_v1"
RENDERER_PAYLOAD_DFL1_TARGET_KIND = "renderer_payload_dfl1_v1"
SHELL_INFLATE_PARITY_PROOF_SCHEMA = "shell_inflate_parity_proof_v2"
PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND = "packet_member_zip_header_elide_v1"
TENSOR_FACTORIZE_TARGET_KIND = "tensor_factorize_v1"
ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER_ID = "archive_section_entropy_recode_adapter"
PACKET_MEMBER_RECOMPRESS_MATERIALIZER_ID = "packet_member_recompress_adapter"
PACKET_MEMBER_MERGE_MATERIALIZER_ID = "packet_member_merge_adapter"
RENDERER_PAYLOAD_DFL1_MATERIALIZER_ID = "renderer_payload_dfl1_adapter"
PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER_ID = "packet_member_zip_header_elide_adapter"
TENSOR_FACTORIZE_MATERIALIZER_ID = "tensor_factorize_adapter"
PACKET_MEMBER_MERGE_RUNTIME_ADAPTER_PROOF_KIND = (
    "packet_member_merge_runtime_adapter_consumption_proof.v1"
)
PACKET_MEMBER_MERGE_RECEIVER_CONTRACT_KIND = "family_agnostic_packet_member_merge"
PACKET_MEMBER_MERGE_PAYLOAD_MAGIC = b"TAC_PACKET_MEMBER_MERGE_V1\0"
PACKET_MEMBER_MERGE_BINARY_PAYLOAD_MAGIC = b"TAC_PACKET_MEMBER_MERGE_BIN1\0"
PACKET_MEMBER_MERGE_DEFLATE_SEQUENCE_PAYLOAD_MAGIC = b"TAC_PACKET_MEMBER_MERGE_DFL1\0"
RENDERER_PAYLOAD_DFL1_MAGIC = b"DFL1"
RENDERER_PAYLOAD_DFL1_MEMBER_NAMES = ("renderer.bin", "masks.mkv", "optimized_poses.pt")
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
PORTABILITY_CONTRACT_SCHEMA = "family_agnostic_materializer_portability_contract.v1"


class FamilyAgnosticMaterializerError(ValueError):
    """Raised when a family-agnostic materializer cannot build a candidate."""


def _materializer_portability_contract(
    *,
    materializer_id: str,
    target_kind: str,
    required_python_modules: Sequence[str],
    deterministic_surface: str,
    unsupported_features: Sequence[str] = (),
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": PORTABILITY_CONTRACT_SCHEMA,
        "materializer_id": materializer_id,
        "target_kind": target_kind,
        "portability_class": "portable_python_reference",
        "implementation_language": "python",
        "requires_cuda": False,
        "requires_mlx": False,
        "requires_metal": False,
        "requires_mps": False,
        "requires_gpu": False,
        "required_python_modules": list(required_python_modules),
        "deterministic_surface": deterministic_surface,
        "unsupported_features": list(unsupported_features),
        "score_authority": False,
        "promotion_authority": False,
        "rank_or_kill_authority": False,
        "notes": notes,
    }


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
        "portability_contract": _materializer_portability_contract(
            materializer_id=PACKET_MEMBER_RECOMPRESS_MATERIALIZER_ID,
            target_kind=PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
            required_python_modules=("zipfile",),
            deterministic_surface="python_stdlib_zipfile_member_rewrite",
        ),
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


def materialize_packet_member_merge_candidate(
    *,
    archive_path: str | Path,
    output_archive: str | Path,
    packet_member_manifest: str | Path | Mapping[str, Any] | None = None,
    member_name: str | None = None,
    member_names: Sequence[str] = (),
    all_members: bool = False,
    merge_contract: str | Path | Mapping[str, Any] | None = None,
    merged_member_name: str | None = None,
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None = None,
    runtime_consumption_proof_out: str | Path | None = None,
    repo_root: str | Path | None = None,
    allow_size_regression: bool = False,
    allow_overwrite: bool = False,
    expected_existing_output_sha256: str | None = None,
    expected_existing_runtime_consumption_proof_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Merge selected ZIP members into a deterministic cooperative-receiver packet."""

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
    contract = _load_mapping(merge_contract, repo=repo)
    target_members = _select_member_names(
        archive,
        explicit=member_name,
        explicit_many=member_names,
        manifest=manifest,
        contract=contract,
        all_members=all_members,
    )
    if len(target_members) < 2:
        raise FamilyAgnosticMaterializerError(
            "packet_member_merge_v1 requires at least two selected members"
        )
    target_merged_member_name = _merged_member_name(
        archive,
        explicit=merged_member_name,
        contract=contract,
        selected_member_names=target_members,
    )
    source_record = _archive_record(archive)
    selected_entries = [
        _merge_member_entry(archive, target_member)
        for target_member in target_members
    ]
    selected_payloads = [
        (str(entry["name"]), bytes(entry["payload"]))
        for entry in selected_entries
    ]
    source_member_records = [
        _member_record(archive, target_member, payload=payload)
        for target_member, payload in selected_payloads
    ]
    source_member_sha256s = {
        str(row["name"]): str(row["sha256"])
        for row in source_member_records
    }
    non_selected_member_records = [
        _member_record(archive, target_member)
        for target_member in _zip_member_names(archive)
        if target_member not in set(target_members)
    ]
    merge_payload_variants = _packet_member_merge_payload_variants(selected_entries)
    candidate_trials: list[dict[str, Any]] = []
    for payload_variant in merge_payload_variants:
        for compression, compresslevel in _merge_contract_compression_trials(contract):
            archive_payload = _zip_archive_bytes_with_member_merge(
                archive,
                selected_member_names=target_members,
                merged_member_name=target_merged_member_name,
                merged_payload=bytes(payload_variant["payload"]),
                compression=compression,
                compresslevel=compresslevel,
            )
            candidate_trials.append(
                {
                    "merge_payload_codec": payload_variant["payload_codec"],
                    "merge_payload_bytes": len(bytes(payload_variant["payload"])),
                    "merge_table_bytes": int(
                        payload_variant["table"].get(
                            "binary_table_bytes",
                            len(_canonical_json_bytes(payload_variant["table"])),
                        )
                    ),
                    "zip_compression_method": _compression_method_name(compression),
                    "zip_compresslevel": compresslevel,
                    "archive_bytes": len(archive_payload),
                    "archive_sha256": sha256_bytes(archive_payload),
                    "payload": archive_payload,
                    "merge_payload": payload_variant["payload"],
                    "merge_table": payload_variant["table"],
                }
            )
    if not candidate_trials:
        raise FamilyAgnosticMaterializerError("no packet_member_merge candidates were produced")
    best = min(
        candidate_trials,
        key=lambda item: (
            int(item["archive_bytes"]),
            str(item["zip_compression_method"]),
            -1 if item["zip_compresslevel"] is None else int(item["zip_compresslevel"]),
        ),
    )
    archive_payload = bytes(best["payload"])
    merge_payload = bytes(best["merge_payload"])
    merge_table = best["merge_table"]
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
    candidate_merged_member = _member_record(output, target_merged_member_name)
    candidate_non_selected_records = [
        _member_record(output, target_member)
        for target_member in _zip_member_names(output)
        if target_member != target_merged_member_name
    ]
    proof_write_result = None
    runtime_proof_ref: str | Path | Mapping[str, Any] | None = runtime_consumption_proof
    local_reconstruction_proof_satisfied: bool | None = None
    if proof_out is not None:
        runtime_proof_payload = _packet_member_merge_runtime_consumption_proof(
            source_archive=source_record,
            candidate_archive=candidate_record,
            source_members=source_member_records,
            source_member_sha256s=source_member_sha256s,
            source_non_selected_members=non_selected_member_records,
            candidate_merged_member=candidate_merged_member,
            candidate_non_selected_members=candidate_non_selected_records,
            merged_member_name=target_merged_member_name,
            selected_member_names=target_members,
            merge_payload=merge_payload,
            merge_table=merge_table,
            compression_method=str(best["zip_compression_method"]),
            compresslevel=best["zip_compresslevel"],
            contract=contract,
        )
        runtime_probe = runtime_proof_payload.get("runtime_consumption_probe")
        if isinstance(runtime_probe, Mapping):
            local_reconstruction_proof_satisfied = (
                runtime_probe.get("internal_reconstruction_passed") is True
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
        required_candidate_member_sha256=candidate_merged_member["sha256"],
        required_proof_kind=PACKET_MEMBER_MERGE_RUNTIME_ADAPTER_PROOF_KIND,
        required_receiver_contract_kind=PACKET_MEMBER_MERGE_RECEIVER_CONTRACT_KIND,
        required_target_kind=PACKET_MEMBER_MERGE_TARGET_KIND,
        required_materializer_id=PACKET_MEMBER_MERGE_MATERIALIZER_ID,
        repo_root=repo,
    )
    if receiver_verification.get("runtime_adapter_ready") is not True:
        blockers.append(
            "packet_member_merge_exact_readiness_refused_until_byte_closed_runtime_adapter_lands"
        )
    readiness_blockers = _readiness_blockers(
        blockers,
        receiver_verification,
        receiver_blocker="packet_member_merge_receiver_contract_not_satisfied",
    )
    return {
        "schema": PACKET_MEMBER_MERGE_SCHEMA,
        "materializer_id": PACKET_MEMBER_MERGE_MATERIALIZER_ID,
        "target_kind": PACKET_MEMBER_MERGE_TARGET_KIND,
        "portability_contract": _materializer_portability_contract(
            materializer_id=PACKET_MEMBER_MERGE_MATERIALIZER_ID,
            target_kind=PACKET_MEMBER_MERGE_TARGET_KIND,
            required_python_modules=("json", "struct", "zipfile"),
            deterministic_surface="python_stdlib_zipfile_deterministic_member_merge_packet",
            notes=(
                "Portable reference merge packet; contest runtime must use a "
                "cooperative receiver adapter to reconstruct original member names."
            ),
        ),
        "receiver_contract_id": f"{PACKET_MEMBER_MERGE_TARGET_KIND}.receiver.v1",
        "receiver_contract_kind": PACKET_MEMBER_MERGE_RECEIVER_CONTRACT_KIND,
        "byte_closed_candidate_emitted": True,
        "source_archive": source_record,
        "source_members": source_member_records,
        "source_member_sha256s": source_member_sha256s,
        "source_non_selected_members": non_selected_member_records,
        "candidate_archive": candidate_record,
        "candidate_merged_member": candidate_merged_member,
        "candidate_member": candidate_merged_member,
        "candidate_non_selected_members": candidate_non_selected_records,
        "selected_member_names": target_members,
        "merged_member_name": target_merged_member_name,
        "merge_payload_format": "tac_packet_member_merge_payload.v1",
        "merge_table": merge_table,
        "selected_merge": {
            "source_archive_bytes": source_record["bytes"],
            "candidate_archive_bytes": candidate_record["bytes"],
            "saved_bytes": saved_bytes,
            "selected_member_count": len(target_members),
            "merged_member_name": target_merged_member_name,
            "merge_payload_codec": best["merge_payload_codec"],
            "merge_payload_bytes": best["merge_payload_bytes"],
            "merge_table_bytes": best["merge_table_bytes"],
            "zip_compression_method": best["zip_compression_method"],
            "zip_compresslevel": best["zip_compresslevel"],
        },
        "candidate_trials": [
            {
                key: value
                for key, value in trial.items()
                if key not in {"payload", "merge_payload", "merge_table"}
            }
            for trial in candidate_trials
        ],
        "receiver_verification": receiver_verification,
        "runtime_consumption_proof_path": (
            proof_out.as_posix() if proof_out is not None else receiver_verification.get("proof_path")
        ),
        "runtime_consumption_proof_write": (
            proof_write_result.__dict__ if proof_write_result is not None else None
        ),
        "reconstruction_proof_satisfied": (
            local_reconstruction_proof_satisfied
            if local_reconstruction_proof_satisfied is not None
            else receiver_verification["receiver_contract_satisfied"] is True
        ),
        "receiver_contract_satisfied": (
            receiver_verification["receiver_contract_satisfied"] is True
            and receiver_verification.get("runtime_adapter_ready") is True
        ),
        "runtime_adapter_ready": receiver_verification.get("runtime_adapter_ready") is True,
        "readiness_blockers": readiness_blockers,
        "artifact_write": write_result.__dict__,
        **FALSE_AUTHORITY,
    }


def materialize_renderer_payload_dfl1_candidate(
    *,
    archive_path: str | Path,
    output_archive: str | Path,
    packet_member_manifest: str | Path | Mapping[str, Any] | None = None,
    member_names: Sequence[str] = RENDERER_PAYLOAD_DFL1_MEMBER_NAMES,
    payload_member_name: str = "p",
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None = None,
    runtime_consumption_proof_out: str | Path | None = None,
    full_frame_inflate_parity_proof: str | Path | Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    allow_size_regression: bool = False,
    allow_overwrite: bool = False,
    expected_existing_output_sha256: str | None = None,
    expected_existing_runtime_consumption_proof_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Pack renderer/mask/pose members into source-runtime-native DFL1 payload."""

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
    target_members = _renderer_payload_dfl1_member_names(
        archive,
        explicit_many=member_names,
        manifest=manifest,
    )
    output_member = _renderer_payload_dfl1_payload_member_name(
        archive,
        payload_member_name,
    )
    selected_entries = [_merge_member_entry(archive, name) for name in target_members]
    unsupported = [
        str(entry["name"])
        for entry in selected_entries
        if int(entry["zip_compress_type"]) != zipfile.ZIP_DEFLATED
    ]
    if unsupported:
        raise FamilyAgnosticMaterializerError(
            "renderer_payload_dfl1_v1 requires source ZIP_DEFLATED members: "
            + ", ".join(unsupported)
        )
    source_record = _archive_record(archive)
    source_member_records = [
        _member_record(archive, str(entry["name"]), payload=bytes(entry["payload"]))
        for entry in selected_entries
    ]
    payload = _renderer_payload_dfl1_payload(selected_entries)
    candidate_trials: list[dict[str, Any]] = []
    for compression, compresslevel in (
        (zipfile.ZIP_STORED, None),
        (zipfile.ZIP_DEFLATED, 9),
    ):
        archive_payload = _zip_archive_bytes_single_member(
            member_name=output_member,
            payload=payload,
            compression=compression,
            compresslevel=compresslevel,
        )
        candidate_trials.append(
            {
                "payload_member_name": output_member,
                "payload_codec": "native_fixed_order_raw_deflate_sequence_v1",
                "payload_bytes": len(payload),
                "zip_compression_method": _compression_method_name(compression),
                "zip_compresslevel": compresslevel,
                "archive_bytes": len(archive_payload),
                "archive_sha256": sha256_bytes(archive_payload),
                "payload": archive_payload,
            }
        )
    best = min(
        candidate_trials,
        key=lambda item: (
            int(item["archive_bytes"]),
            str(item["zip_compression_method"]),
        ),
    )
    archive_payload = bytes(best["payload"])
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
    candidate_member_record = _member_record(output, output_member)
    full_frame_parity_verification = (
        verify_renderer_payload_dfl1_full_frame_inflate_parity_proof(
            full_frame_inflate_parity_proof=full_frame_inflate_parity_proof,
            required_source_archive_sha256=source_record["sha256"],
            required_candidate_archive_sha256=candidate_record["sha256"],
            repo_root=repo,
        )
    )
    if (
        full_frame_parity_verification.get("full_frame_inflate_parity_satisfied")
        is not True
    ):
        blockers.append("renderer_payload_dfl1_full_frame_inflate_parity_missing")
        blockers.extend(
            str(blocker)
            for blocker in full_frame_parity_verification.get("blockers") or []
        )
    proof_write_result = None
    runtime_proof_ref: str | Path | Mapping[str, Any] | None = runtime_consumption_proof
    if proof_out is not None:
        runtime_proof_payload = _renderer_payload_dfl1_runtime_consumption_proof(
            repo_root=repo,
            source_archive=source_record,
            candidate_archive=candidate_record,
            source_members=source_member_records,
            candidate_member=candidate_member_record,
            payload_member_name=output_member,
            payload=payload,
            selected_member_names=target_members,
            full_frame_parity_verification=full_frame_parity_verification,
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
        required_candidate_member_sha256=candidate_member_record["sha256"],
        required_proof_kind="renderer_payload_dfl1_native_unpacker_reconstruction_smoke.v1",
        required_receiver_contract_kind="source_runtime_native_renderer_payload_dfl1",
        required_target_kind=RENDERER_PAYLOAD_DFL1_TARGET_KIND,
        required_materializer_id=RENDERER_PAYLOAD_DFL1_MATERIALIZER_ID,
        repo_root=repo,
    )
    readiness_blockers = _readiness_blockers(
        blockers,
        receiver_verification,
        receiver_blocker="renderer_payload_dfl1_receiver_contract_not_satisfied",
    )
    return {
        "schema": RENDERER_PAYLOAD_DFL1_SCHEMA,
        "materializer_id": RENDERER_PAYLOAD_DFL1_MATERIALIZER_ID,
        "target_kind": RENDERER_PAYLOAD_DFL1_TARGET_KIND,
        "portability_contract": _materializer_portability_contract(
            materializer_id=RENDERER_PAYLOAD_DFL1_MATERIALIZER_ID,
            target_kind=RENDERER_PAYLOAD_DFL1_TARGET_KIND,
            required_python_modules=("zipfile", "zlib"),
            deterministic_surface="source_runtime_native_renderer_payload_fixed_deflate_sequence",
            notes=(
                "Uses the robust renderer payload unpacker directly; no generated "
                "receiver wrapper is needed, but full-frame parity and auth eval "
                "remain separate gates."
            ),
        ),
        "receiver_contract_id": f"{RENDERER_PAYLOAD_DFL1_TARGET_KIND}.receiver.v1",
        "receiver_contract_kind": "source_runtime_native_renderer_payload_dfl1",
        "byte_closed_candidate_emitted": True,
        "source_archive": source_record,
        "source_members": source_member_records,
        "candidate_archive": candidate_record,
        "candidate_member": candidate_member_record,
        "selected_member_names": target_members,
        "payload_member_name": output_member,
        "selected_payload": {
            "source_archive_bytes": source_record["bytes"],
            "candidate_archive_bytes": candidate_record["bytes"],
            "saved_bytes": saved_bytes,
            "payload_member_name": output_member,
            "payload_codec": "native_fixed_order_raw_deflate_sequence_v1",
            "payload_bytes": len(payload),
            "zip_compression_method": best["zip_compression_method"],
            "zip_compresslevel": best["zip_compresslevel"],
        },
        "candidate_trials": [
            {key: value for key, value in trial.items() if key != "payload"}
            for trial in candidate_trials
        ],
        "receiver_verification": receiver_verification,
        "full_frame_inflate_parity_verification": full_frame_parity_verification,
        "full_frame_inflate_parity_proven": (
            full_frame_parity_verification.get("full_frame_inflate_parity_satisfied")
            is True
        ),
        "renderer_payload_dfl1_inflate_parity_proof_path": (
            full_frame_parity_verification.get("proof_path")
        ),
        "renderer_payload_dfl1_inflate_parity_proof_sha256": (
            full_frame_parity_verification.get("proof_sha256")
        ),
        "renderer_payload_dfl1_inflate_parity_satisfied": (
            full_frame_parity_verification.get("full_frame_inflate_parity_satisfied")
            is True
        ),
        "runtime_consumption_proof_path": (
            proof_out.as_posix() if proof_out is not None else receiver_verification.get("proof_path")
        ),
        "runtime_consumption_proof_write": (
            proof_write_result.__dict__ if proof_write_result is not None else None
        ),
        "reconstruction_proof_satisfied": (
            receiver_verification["receiver_contract_satisfied"] is True
        ),
        "receiver_contract_satisfied": (
            receiver_verification["receiver_contract_satisfied"] is True
            and receiver_verification.get("runtime_adapter_ready") is True
        ),
        "runtime_adapter_ready": receiver_verification.get("runtime_adapter_ready") is True,
        "readiness_blockers": readiness_blockers,
        "artifact_write": write_result.__dict__,
        **FALSE_AUTHORITY,
    }


def _packet_member_merge_runtime_consumption_proof(
    *,
    source_archive: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
    source_members: Sequence[Mapping[str, Any]],
    source_member_sha256s: Mapping[str, str],
    source_non_selected_members: Sequence[Mapping[str, Any]],
    candidate_merged_member: Mapping[str, Any],
    candidate_non_selected_members: Sequence[Mapping[str, Any]],
    merged_member_name: str,
    selected_member_names: Sequence[str],
    merge_payload: bytes,
    merge_table: Mapping[str, Any],
    compression_method: str,
    compresslevel: int | None,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    reconstruction = parse_packet_member_merge_payload(merge_payload)
    reconstructed_member_sha256s = {
        name: sha256_bytes(payload)
        for name, payload in reconstruction["members"].items()
    }
    selected_member_proofs = []
    all_selected_reconstructable = True
    for name in selected_member_names:
        source_sha = str(source_member_sha256s.get(str(name)) or "")
        reconstructed_sha = str(reconstructed_member_sha256s.get(str(name)) or "")
        passed = bool(source_sha) and reconstructed_sha == source_sha
        all_selected_reconstructable = all_selected_reconstructable and passed
        selected_member_proofs.append(
            {
                "member_name": str(name),
                "source_member_sha256": source_sha or None,
                "reconstructed_member_sha256": reconstructed_sha or None,
                "reconstructed_member_matches_source": passed,
                "passed": passed,
            }
        )
    source_non_selected = {str(row.get("name")): row for row in source_non_selected_members}
    candidate_non_selected = {
        str(row.get("name")): row for row in candidate_non_selected_members
    }
    non_selected_member_proofs = []
    all_non_selected_payloads_identical = True
    all_non_selected_compressed_payloads_identical = True
    for name, source_row in sorted(source_non_selected.items()):
        candidate_row = candidate_non_selected.get(name, {})
        source_sha = _clean_str(source_row.get("sha256"))
        candidate_sha = _clean_str(candidate_row.get("sha256"))
        payload_passed = source_sha is not None and candidate_sha == source_sha
        source_compressed_sha = _clean_str(
            source_row.get("zip_compressed_payload_sha256")
            or source_row.get("zip_compressed_sha256")
        )
        candidate_compressed_sha = _clean_str(
            candidate_row.get("zip_compressed_payload_sha256")
            or candidate_row.get("zip_compressed_sha256")
        )
        compressed_passed = (
            source_compressed_sha is not None
            and candidate_compressed_sha == source_compressed_sha
        )
        passed = payload_passed and compressed_passed
        all_non_selected_payloads_identical = (
            all_non_selected_payloads_identical and payload_passed
        )
        all_non_selected_compressed_payloads_identical = (
            all_non_selected_compressed_payloads_identical and compressed_passed
        )
        non_selected_member_proofs.append(
            {
                "member_name": name,
                "source_member_sha256": source_sha,
                "candidate_member_sha256": candidate_sha,
                "member_payload_identical_to_source": payload_passed,
                "source_zip_compressed_payload_sha256": source_compressed_sha,
                "candidate_zip_compressed_payload_sha256": candidate_compressed_sha,
                "zip_compressed_payload_identical_to_source": compressed_passed,
                "passed": passed,
            }
        )
    cooperative_receiver_id = _mapping_string_any(
        contract,
        ("cooperative_receiver_id", "receiver_id", "runtime_adapter_id"),
    )
    receiver_adapter_kind = _mapping_string_any(
        contract,
        ("receiver_adapter_kind", "cooperative_receiver_adapter_kind", "runtime_adapter_kind"),
    )
    receiver_declared = cooperative_receiver_id is not None and receiver_adapter_kind is not None
    internal_reconstruction_passed = (
        all_selected_reconstructable
        and all_non_selected_payloads_identical
        and all_non_selected_compressed_payloads_identical
        and receiver_declared
        and reconstruction["table"].get("schema") == "packet_member_merge_table.v1"
    )
    return {
        "schema": "family_agnostic_runtime_consumption_proof_v1",
        "proof_kind": "packet_member_merge_original_member_reconstruction_receiver_proof.v1",
        "proof_scope": "merged_zip_member_reconstructs_original_selected_member_payloads",
        "target_kind": PACKET_MEMBER_MERGE_TARGET_KIND,
        "materializer_id": PACKET_MEMBER_MERGE_MATERIALIZER_ID,
        "portability_contract": _materializer_portability_contract(
            materializer_id=PACKET_MEMBER_MERGE_MATERIALIZER_ID,
            target_kind=PACKET_MEMBER_MERGE_TARGET_KIND,
            required_python_modules=("json", "struct", "zipfile"),
            deterministic_surface="python_stdlib_zipfile_deterministic_member_merge_packet",
            notes=(
                "Portable reference merge packet; contest runtime must use a "
                "cooperative receiver adapter to reconstruct original member names."
            ),
        ),
        "receiver_contract_kind": PACKET_MEMBER_MERGE_RECEIVER_CONTRACT_KIND,
        "receiver_contract_id": f"{PACKET_MEMBER_MERGE_TARGET_KIND}.receiver.v1",
        "source_archive": dict(source_archive),
        "source_members": [dict(row) for row in source_members],
        "candidate_archive": dict(candidate_archive),
        "candidate_merged_member": dict(candidate_merged_member),
        "candidate_member": dict(candidate_merged_member),
        "candidate_archive_sha256": candidate_archive.get("sha256"),
        "candidate_member_sha256": candidate_merged_member.get("sha256"),
        "member_sha256": candidate_merged_member.get("sha256"),
        "selected_member_names": list(selected_member_names),
        "merged_member_name": merged_member_name,
        "merge_payload_format": "tac_packet_member_merge_payload.v1",
        "merge_table": dict(merge_table),
        "compression": {
            "zip_compression_method": compression_method,
            "zip_compresslevel": compresslevel,
        },
        "reconstructed_member_sha256s": reconstructed_member_sha256s,
        "selected_member_proofs": selected_member_proofs,
        "non_selected_member_proofs": non_selected_member_proofs,
        "cooperative_receiver": {
            "cooperative_receiver_id": cooperative_receiver_id,
            "receiver_adapter_kind": receiver_adapter_kind,
            "declared": receiver_declared,
            "runtime_adapter_ready": False,
            "reconstruction_formula": (
                "parse merge table, slice concatenated payload by offset/length, "
                "write original member names"
            ),
        },
        "runtime_consumption_probe": {
            "schema": "packet_member_merge_reconstruction_probe.v1",
            "passed": False,
            "internal_reconstruction_passed": internal_reconstruction_passed,
            "selected_member_proofs": selected_member_proofs,
            "non_selected_member_proofs": non_selected_member_proofs,
            "all_selected_members_reconstructable": all_selected_reconstructable,
            "all_non_selected_member_payloads_identical": (
                all_non_selected_payloads_identical
            ),
            "all_non_selected_member_zip_streams_identical": (
                all_non_selected_compressed_payloads_identical
            ),
            "cooperative_receiver_declared": receiver_declared,
            "runtime_adapter_ready": False,
        },
        "receiver_contract_satisfied": False,
        "runtime_adapter_ready": False,
        "runtime_consumption_proof_passed": False,
        "passed": False,
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
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
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
        "portability_contract": _materializer_portability_contract(
            materializer_id=PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER_ID,
            target_kind=PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
            required_python_modules=("struct", "zipfile"),
            deterministic_surface="python_stdlib_raw_zip32_wire_rewrite",
            unsupported_features=("zip64", "data_descriptors", "duplicate_member_names"),
        ),
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
        "portability_contract": _materializer_portability_contract(
            materializer_id=PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER_ID,
            target_kind=PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
            required_python_modules=("struct", "zipfile"),
            deterministic_surface="python_stdlib_raw_zip32_wire_rewrite",
            unsupported_features=("zip64", "data_descriptors", "duplicate_member_names"),
        ),
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
        "portability_contract": _materializer_portability_contract(
            materializer_id=ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER_ID,
            target_kind=ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
            required_python_modules=("brotli", "zipfile"),
            deterministic_surface="python_zipfile_brotli_section_recode",
        ),
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
        "portability_contract": _materializer_portability_contract(
            materializer_id=TENSOR_FACTORIZE_MATERIALIZER_ID,
            target_kind=TENSOR_FACTORIZE_TARGET_KIND,
            required_python_modules=("numpy", "zipfile"),
            deterministic_surface="numpy_svd_npz_tensor_factorization",
            notes=(
                "Portable CPU reference path; numerical output may vary by BLAS/SVD "
                "backend and remains false-authority until receiver tolerance proof passes."
            ),
        ),
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
    required_proof_kind: str | None = None,
    required_receiver_contract_kind: str | None = None,
    required_target_kind: str | None = None,
    required_materializer_id: str | None = None,
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
    if required_proof_kind is not None and proof.get("proof_kind") != required_proof_kind:
        blockers.append("runtime_consumption_proof_kind_mismatch")
    if (
        required_receiver_contract_kind is not None
        and proof.get("receiver_contract_kind") != required_receiver_contract_kind
    ):
        blockers.append("runtime_consumption_proof_receiver_contract_kind_mismatch")
    if required_target_kind is not None and proof.get("target_kind") != required_target_kind:
        blockers.append("runtime_consumption_proof_target_kind_mismatch")
    if (
        required_materializer_id is not None
        and proof.get("materializer_id") != required_materializer_id
    ):
        blockers.append("runtime_consumption_proof_materializer_id_mismatch")
    runtime_adapter_manifest = proof.get("runtime_adapter_manifest")
    runtime_adapter_ready = proof.get("runtime_adapter_ready") is True or (
        isinstance(runtime_adapter_manifest, Mapping)
        and runtime_adapter_manifest.get("runtime_adapter_ready") is True
    )
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
        "proof_kind": proof.get("proof_kind"),
        "receiver_contract_kind": proof.get("receiver_contract_kind"),
        "target_kind": proof.get("target_kind"),
        "materializer_id": proof.get("materializer_id"),
        "candidate_archive_sha256": archive_sha,
        "candidate_member_sha256": member_sha,
        "candidate_member_sha256s": member_sha_map,
        "runtime_adapter_ready": runtime_adapter_ready,
        "blockers": ordered_unique(blockers),
    }


def verify_renderer_payload_dfl1_full_frame_inflate_parity_proof(
    *,
    full_frame_inflate_parity_proof: str | Path | Mapping[str, Any] | None,
    required_source_archive_sha256: str,
    required_candidate_archive_sha256: str,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a shell-level full-frame parity proof for DFL1 candidates."""

    repo = _repo(repo_root)
    if full_frame_inflate_parity_proof is None:
        return {
            "schema": "renderer_payload_dfl1_full_frame_parity_verification.v1",
            "proof_present": False,
            "proof_path": None,
            "full_frame_inflate_parity_satisfied": False,
            "blockers": ["renderer_payload_dfl1_full_frame_inflate_parity_proof_missing"],
            **FALSE_AUTHORITY,
        }
    proof_path = (
        _resolve_path(full_frame_inflate_parity_proof, repo=repo).as_posix()
        if isinstance(full_frame_inflate_parity_proof, (str, Path))
        else None
    )
    proof = _require_mapping(full_frame_inflate_parity_proof, repo=repo)
    blockers: list[str] = []
    try:
        require_no_truthy_authority_fields(
            proof,
            context="renderer_payload_dfl1_full_frame_inflate_parity_proof",
        )
    except ValueError as exc:
        blockers.append(str(exc))
    if proof.get("schema") != SHELL_INFLATE_PARITY_PROOF_SCHEMA:
        blockers.append("shell_inflate_parity_proof_schema_mismatch")
    if proof.get("full_frame_file_list_claim") is not True:
        blockers.append("shell_inflate_parity_full_frame_file_list_claim_missing")
    if proof.get("full_frame_inflate_output_parity_claim") is not True:
        blockers.append("shell_inflate_full_frame_output_parity_claim_missing")
    full_frame_source = _clean_str(proof.get("full_frame_file_list_source"))
    expected_file_list_sha = _clean_str(
        proof.get("expected_full_frame_file_list_sha256")
    )
    if not full_frame_source:
        blockers.append("shell_inflate_parity_full_frame_file_list_source_missing")
    if expected_file_list_sha is None or len(expected_file_list_sha) != 64 or any(
        char not in "0123456789abcdef" for char in expected_file_list_sha
    ):
        blockers.append("shell_inflate_parity_expected_file_list_sha256_missing")
    if proof.get("full_frame_file_list_sha256_match") is not True:
        blockers.append("shell_inflate_parity_file_list_sha256_match_not_true")
    expected_entry_count = proof.get("expected_full_frame_entry_count")
    if not isinstance(expected_entry_count, int) or expected_entry_count < 1:
        blockers.append("shell_inflate_parity_expected_entry_count_invalid")
    if proof.get("full_frame_entry_count_match") is not True:
        blockers.append("shell_inflate_parity_entry_count_match_not_true")
    for key in (
        "output_bytes_match",
        "output_sha256_match",
        "output_manifest_sha256_match",
        "cmp_equal",
    ):
        if proof.get(key) is not True:
            blockers.append(f"shell_inflate_parity_{key}_not_true")
    proof_blockers = proof.get("blockers")
    if isinstance(proof_blockers, Sequence) and not isinstance(
        proof_blockers, (bytes, bytearray, str)
    ):
        blockers.extend(str(blocker) for blocker in proof_blockers if str(blocker))
    elif proof_blockers:
        blockers.append("shell_inflate_parity_blockers_not_list")

    left = proof.get("left") if isinstance(proof.get("left"), Mapping) else {}
    right = proof.get("right") if isinstance(proof.get("right"), Mapping) else {}
    source_side, candidate_side = _match_parity_archive_sides(
        left=left,
        right=right,
        required_source_archive_sha256=required_source_archive_sha256,
        required_candidate_archive_sha256=required_candidate_archive_sha256,
    )
    if source_side is None or candidate_side is None:
        blockers.append("shell_inflate_parity_archive_sha_pair_mismatch")
    left_tree = _clean_str(left.get("submission_tree_sha256"))
    right_tree = _clean_str(right.get("submission_tree_sha256"))
    if not left_tree or not right_tree:
        blockers.append("shell_inflate_parity_submission_tree_sha_missing")
    elif left_tree != right_tree:
        blockers.append("shell_inflate_parity_submission_tree_sha_mismatch")
    file_list_entry_count = proof.get("file_list_entry_count")
    if not isinstance(file_list_entry_count, int) or file_list_entry_count < 1:
        blockers.append("shell_inflate_parity_file_list_entry_count_invalid")
    actual_file_list_sha = _clean_str(proof.get("file_list_sha256"))
    if expected_file_list_sha is not None:
        if not actual_file_list_sha:
            blockers.append("shell_inflate_parity_file_list_sha256_missing")
        elif actual_file_list_sha != expected_file_list_sha:
            blockers.append("shell_inflate_parity_file_list_sha256_mismatch")
    if (
        isinstance(expected_entry_count, int)
        and isinstance(file_list_entry_count, int)
        and file_list_entry_count != expected_entry_count
    ):
        blockers.append("shell_inflate_parity_entry_count_mismatch")
    output_count = proof.get("output_count")
    if not isinstance(output_count, int) or output_count != file_list_entry_count:
        blockers.append("shell_inflate_parity_output_count_invalid")
    return {
        "schema": "renderer_payload_dfl1_full_frame_parity_verification.v1",
        "proof_present": True,
        "proof_path": proof_path,
        "proof_sha256": sha256_file(_resolve_path(full_frame_inflate_parity_proof, repo=repo))
        if isinstance(full_frame_inflate_parity_proof, (str, Path))
        else None,
        "proof_schema": proof.get("schema"),
        "full_frame_inflate_parity_satisfied": not blockers,
        "file_list_entry_count": file_list_entry_count,
        "file_list_sha256": actual_file_list_sha,
        "full_frame_file_list_source": full_frame_source,
        "expected_full_frame_file_list_sha256": expected_file_list_sha,
        "expected_full_frame_entry_count": expected_entry_count,
        "source_archive_sha256": required_source_archive_sha256,
        "candidate_archive_sha256": required_candidate_archive_sha256,
        "source_side_label": source_side.get("label") if source_side else None,
        "candidate_side_label": candidate_side.get("label") if candidate_side else None,
        "submission_tree_sha256": left_tree if left_tree == right_tree else None,
        "output_manifest_sha256": _clean_str(left.get("output_manifest_sha256"))
        if _clean_str(left.get("output_manifest_sha256"))
        == _clean_str(right.get("output_manifest_sha256"))
        else None,
        "blockers": ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _match_parity_archive_sides(
    *,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    required_source_archive_sha256: str,
    required_candidate_archive_sha256: str,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    left_sha = _clean_str(left.get("archive_sha256"))
    right_sha = _clean_str(right.get("archive_sha256"))
    if (
        left_sha == required_source_archive_sha256
        and right_sha == required_candidate_archive_sha256
    ):
        return left, right
    if (
        right_sha == required_source_archive_sha256
        and left_sha == required_candidate_archive_sha256
    ):
        return right, left
    return None, None


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


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


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


def _zip_archive_bytes_single_member(
    *,
    member_name: str,
    payload: bytes,
    compression: int,
    compresslevel: int | None,
) -> bytes:
    output = io.BytesIO()
    info = zipfile.ZipInfo(member_name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o644 << 16
    kwargs = {}
    if compression == zipfile.ZIP_DEFLATED and compresslevel is not None:
        kwargs["compresslevel"] = compresslevel
    with zipfile.ZipFile(output, "w") as zf:
        zf.writestr(info, payload, **kwargs)
    return output.getvalue()


def _merged_member_name(
    archive: Path,
    *,
    explicit: str | None,
    contract: Mapping[str, Any],
    selected_member_names: Sequence[str],
) -> str:
    candidate = (
        _clean_str(explicit)
        or _mapping_string_any(
            contract,
            ("merged_member_name", "candidate_member_name", "output_member_name"),
        )
        or "__packet_member_merge_v1.bin"
    )
    existing = set(_zip_member_names(archive))
    if candidate in existing:
        raise FamilyAgnosticMaterializerError(
            "merged_member_name must not collide with an existing archive member: "
            f"{candidate}"
        )
    if candidate in set(selected_member_names):
        raise FamilyAgnosticMaterializerError(
            "merged_member_name must be distinct from selected member names"
        )
    return candidate


def _merge_contract_compression(contract: Mapping[str, Any]) -> int:
    raw = _mapping_string_any(
        contract,
        ("zip_compression_method", "compression_method", "merged_member_compression_method"),
    )
    if raw is None:
        return zipfile.ZIP_STORED
    methods = _normalized_compression_methods((raw,))
    if len(methods) != 1:
        raise FamilyAgnosticMaterializerError(
            f"unsupported packet_member_merge compression method: {raw}"
        )
    return methods[0]


def _merge_contract_compresslevel(
    contract: Mapping[str, Any],
    *,
    compression: int,
) -> int | None:
    if compression == zipfile.ZIP_STORED:
        return None
    for key in ("zip_compresslevel", "compresslevel", "merged_member_compresslevel"):
        value = contract.get(key)
        if value is None:
            continue
        parsed = _positive_int(value, field=f"merge_contract.{key}")
        return parsed
    return 9


def _merge_contract_compression_trials(contract: Mapping[str, Any]) -> tuple[tuple[int, int | None], ...]:
    raw_methods = _string_items(
        contract.get("zip_compression_methods")
        or contract.get("compression_methods")
        or contract.get("merged_member_compression_methods")
    )
    if not raw_methods:
        raw_method = _mapping_string_any(
            contract,
            (
                "zip_compression_method",
                "compression_method",
                "merged_member_compression_method",
            ),
        )
        raw_methods = [raw_method] if raw_method is not None else ["stored", "deflated"]
    methods = _normalized_compression_methods(raw_methods)
    raw_levels = _string_items(
        contract.get("zip_compresslevels")
        or contract.get("compresslevels")
        or contract.get("merged_member_compresslevels")
    )
    single_level = _mapping_string_any(
        contract,
        ("zip_compresslevel", "compresslevel", "merged_member_compresslevel"),
    )
    if single_level is not None and not raw_levels:
        raw_levels = [single_level]
    trials: list[tuple[int, int | None]] = []
    for method in methods:
        if method == zipfile.ZIP_STORED:
            trials.append((method, None))
            continue
        levels = (
            _ordered_unique_ints(int(level) for level in raw_levels)
            if raw_levels
            else (_merge_contract_compresslevel(contract, compression=method) or 9,)
        )
        for level in levels:
            trials.append((method, int(level)))
    return tuple(trials)


def _merge_member_entry(archive: Path, member_name: str) -> dict[str, Any]:
    payload = _zip_member_bytes(archive, member_name)
    with zipfile.ZipFile(archive, "r") as zf:
        info = zf.getinfo(member_name)
        compression_method = _compression_method_name(info.compress_type)
        compress_type = int(info.compress_type)
        compressed_payload = _zip_member_compressed_payload(archive, member_name)
    return {
        "name": member_name,
        "payload": payload,
        "payload_sha256": sha256_bytes(payload),
        "payload_bytes": len(payload),
        "zip_compress_type": compress_type,
        "zip_compression_method": compression_method,
        "zip_compressed_payload": compressed_payload,
        "zip_compressed_payload_sha256": sha256_bytes(compressed_payload),
        "zip_compressed_bytes": len(compressed_payload),
    }


def _renderer_payload_dfl1_member_names(
    archive: Path,
    *,
    explicit_many: Sequence[str],
    manifest: Mapping[str, Any],
) -> list[str]:
    manifest_names = _manifest_member_names(manifest)
    selected = manifest_names or _string_items(explicit_many)
    if not selected:
        selected = list(RENDERER_PAYLOAD_DFL1_MEMBER_NAMES)
    selected = _require_unique_existing_member_names(archive, selected)
    if tuple(selected) != RENDERER_PAYLOAD_DFL1_MEMBER_NAMES:
        raise FamilyAgnosticMaterializerError(
            "renderer_payload_dfl1_v1 requires fixed-order members: "
            + ", ".join(RENDERER_PAYLOAD_DFL1_MEMBER_NAMES)
        )
    return selected


def _renderer_payload_dfl1_payload_member_name(
    archive: Path,
    payload_member_name: str,
) -> str:
    name = str(payload_member_name or "").strip() or "p"
    if "/" in name or name.startswith(".") or name in {"", ".."}:
        raise FamilyAgnosticMaterializerError(
            f"unsafe renderer_payload_dfl1 payload member name: {payload_member_name!r}"
        )
    if name in set(_zip_member_names(archive)):
        raise FamilyAgnosticMaterializerError(
            "renderer_payload_dfl1 payload member must not collide with source member: "
            f"{name}"
        )
    return name


def _renderer_payload_dfl1_payload(
    selected_entries: Sequence[Mapping[str, Any]],
) -> bytes:
    if len(selected_entries) != len(RENDERER_PAYLOAD_DFL1_MEMBER_NAMES):
        raise FamilyAgnosticMaterializerError(
            "renderer_payload_dfl1_v1 requires exactly three selected entries"
        )
    if any(int(entry["zip_compress_type"]) != zipfile.ZIP_DEFLATED for entry in selected_entries):
        raise FamilyAgnosticMaterializerError(
            "renderer_payload_dfl1_v1 can only carry ZIP raw-deflate member streams"
        )
    return RENDERER_PAYLOAD_DFL1_MAGIC + b"".join(
        bytes(entry["zip_compressed_payload"]) for entry in selected_entries
    )


def parse_renderer_payload_dfl1_payload(
    payload: bytes,
    *,
    member_names: Sequence[str] = RENDERER_PAYLOAD_DFL1_MEMBER_NAMES,
) -> dict[str, Any]:
    """Parse a source-runtime-native renderer DFL1 payload."""

    if not payload.startswith(RENDERER_PAYLOAD_DFL1_MAGIC):
        raise FamilyAgnosticMaterializerError("renderer payload DFL1 has bad magic")
    names = list(member_names)
    if len(names) != len(RENDERER_PAYLOAD_DFL1_MEMBER_NAMES):
        raise FamilyAgnosticMaterializerError(
            "renderer payload DFL1 parser requires exactly three member names"
        )
    remaining = payload[len(RENDERER_PAYLOAD_DFL1_MAGIC):]
    out: dict[str, bytes] = {}
    members = []
    offset = 0
    for name in names:
        decoded, consumed = _decompress_next_zip_deflate_stream(remaining, member_name=name)
        encoded = remaining[:consumed]
        out[name] = decoded
        members.append(
            {
                "name": name,
                "offset": offset,
                "length": len(encoded),
                "compressed_sha256": sha256_bytes(encoded),
                "sha256": sha256_bytes(decoded),
                "uncompressed_length": len(decoded),
                "zip_compress_type": zipfile.ZIP_DEFLATED,
            }
        )
        offset += consumed
        remaining = remaining[consumed:]
    if remaining:
        raise FamilyAgnosticMaterializerError(
            "renderer payload DFL1 has trailing bytes"
        )
    table = {
        "schema": "renderer_payload_dfl1_table.v1",
        "payload_codec": "native_fixed_order_raw_deflate_sequence_v1",
        "table_format": "fixed_order_no_charged_table_v1",
        "member_count": len(members),
        "members": members,
        "binary_table_bytes": len(RENDERER_PAYLOAD_DFL1_MAGIC),
        "binary_table_sha256": sha256_bytes(RENDERER_PAYLOAD_DFL1_MAGIC),
        "concatenated_payload_bytes": len(payload) - len(RENDERER_PAYLOAD_DFL1_MAGIC),
        "concatenated_payload_sha256": sha256_bytes(
            payload[len(RENDERER_PAYLOAD_DFL1_MAGIC):]
        ),
        "merged_payload_sha256": sha256_bytes(payload),
    }
    return {"table": table, "members": out}


def _renderer_payload_dfl1_runtime_consumption_proof(
    *,
    repo_root: Path,
    source_archive: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
    source_members: Sequence[Mapping[str, Any]],
    candidate_member: Mapping[str, Any],
    payload_member_name: str,
    payload: bytes,
    selected_member_names: Sequence[str],
    full_frame_parity_verification: Mapping[str, Any],
) -> dict[str, Any]:
    reconstruction = parse_renderer_payload_dfl1_payload(
        payload,
        member_names=selected_member_names,
    )
    native_probe = _renderer_payload_dfl1_native_unpacker_probe(
        repo_root=repo_root,
        payload=payload,
        selected_member_names=selected_member_names,
    )
    expected = {str(row["name"]): str(row["sha256"]) for row in source_members}
    actual = {
        name: sha256_bytes(member_payload)
        for name, member_payload in reconstruction["members"].items()
    }
    member_proofs = [
        {
            "member_name": name,
            "source_member_sha256": expected.get(name),
            "candidate_reconstructed_sha256": actual.get(name),
            "passed": expected.get(name) == actual.get(name),
        }
        for name in selected_member_names
    ]
    native_hashes = native_probe.get("member_sha256s")
    if not isinstance(native_hashes, Mapping):
        native_hashes = {}
    native_member_proofs = [
        {
            "member_name": name,
            "source_member_sha256": expected.get(name),
            "native_unpacker_sha256": native_hashes.get(name),
            "passed": expected.get(name) == native_hashes.get(name),
        }
        for name in selected_member_names
    ]
    reconstruction_passed = (
        all(row["passed"] is True for row in member_proofs)
        and all(row["passed"] is True for row in native_member_proofs)
        and native_probe["passed"] is True
    )
    full_frame_parity_satisfied = (
        full_frame_parity_verification.get("full_frame_inflate_parity_satisfied")
        is True
    )
    proof_passed = reconstruction_passed and full_frame_parity_satisfied
    runtime_adapter_manifest = dict(native_probe["runtime_adapter_manifest"])
    runtime_adapter_manifest["runtime_adapter_ready"] = proof_passed
    runtime_adapter_manifest["native_unpacker_parse_ready"] = native_probe["passed"] is True
    runtime_adapter_manifest["requires_full_frame_shell_parity"] = True
    return {
        "schema": "family_agnostic_runtime_consumption_proof_v1",
        "proof_kind": "renderer_payload_dfl1_native_unpacker_reconstruction_smoke.v1",
        "target_kind": RENDERER_PAYLOAD_DFL1_TARGET_KIND,
        "materializer_id": RENDERER_PAYLOAD_DFL1_MATERIALIZER_ID,
        "receiver_contract_kind": "source_runtime_native_renderer_payload_dfl1",
        "runtime_adapter_ready": proof_passed,
        "source_runtime_native_unpacker": True,
        "source_runtime_unpacker_parse_satisfied": reconstruction_passed,
        "full_frame_inflate_parity_verification": dict(full_frame_parity_verification),
        "runtime_adapter_manifest": runtime_adapter_manifest,
        "source_archive": dict(source_archive),
        "candidate_archive": dict(candidate_archive),
        "candidate_archive_sha256": candidate_archive.get("sha256"),
        "candidate_member": dict(candidate_member),
        "candidate_member_sha256": candidate_member.get("sha256"),
        "candidate_member_sha256s": {payload_member_name: candidate_member.get("sha256")},
        "payload_member_name": payload_member_name,
        "selected_member_names": list(selected_member_names),
        "payload_table": reconstruction["table"],
        "reconstructed_member_sha256s": actual,
        "runtime_consumption_probe": {
            "schema": "renderer_payload_dfl1_reconstruction_probe.v1",
            "passed": proof_passed,
            "parser_reconstruction_passed": reconstruction_passed,
            "full_frame_inflate_parity_satisfied": full_frame_parity_satisfied,
            "member_proofs": member_proofs,
            "native_member_proofs": native_member_proofs,
            "native_unpacker_probe": native_probe,
        },
        "receiver_contract_satisfied": proof_passed,
        "runtime_consumption_proof_passed": proof_passed,
        "passed": proof_passed,
        **FALSE_AUTHORITY,
    }


def _renderer_payload_dfl1_native_unpacker_probe(
    *,
    repo_root: Path,
    payload: bytes,
    selected_member_names: Sequence[str],
) -> dict[str, Any]:
    unpacker_path = repo_root / "submissions" / "robust_current" / "unpack_renderer_payload.py"
    manifest: dict[str, Any] = {
        "schema": "source_runtime_native_unpacker_manifest.v1",
        "runtime_adapter_ready": False,
        "path": unpacker_path.as_posix(),
        "sha256": None,
    }
    blockers: list[str] = []
    if not unpacker_path.is_file():
        blockers.append("native_unpacker_missing")
    else:
        manifest["sha256"] = sha256_file(unpacker_path)
    if blockers:
        return {
            "schema": "renderer_payload_dfl1_native_unpacker_probe.v1",
            "passed": False,
            "runtime_adapter_manifest": manifest,
            "blockers": blockers,
            "member_sha256s": {},
        }
    spec = importlib.util.spec_from_file_location(
        "_tac_renderer_payload_dfl1_native_unpacker_probe",
        unpacker_path,
    )
    if spec is None or spec.loader is None:
        blockers.append("native_unpacker_import_spec_failed")
        return {
            "schema": "renderer_payload_dfl1_native_unpacker_probe.v1",
            "passed": False,
            "runtime_adapter_manifest": manifest,
            "blockers": blockers,
            "member_sha256s": {},
        }
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        _header, native_members = module._parse_payload(payload)
    except Exception as exc:  # pragma: no cover - defensive import/probe diagnostics
        blockers.append(f"native_unpacker_parse_failed:{exc}")
        native_members = {}
    member_sha256s = {
        str(name): sha256_bytes(bytes(data))
        for name, data in native_members.items()
    }
    missing = [name for name in selected_member_names if name not in member_sha256s]
    if missing:
        blockers.append("native_unpacker_missing_members:" + ",".join(missing))
    extra = [name for name in member_sha256s if name not in set(selected_member_names)]
    if extra:
        blockers.append("native_unpacker_extra_members:" + ",".join(extra))
    manifest["runtime_adapter_ready"] = not blockers
    return {
        "schema": "renderer_payload_dfl1_native_unpacker_probe.v1",
        "passed": not blockers,
        "runtime_adapter_manifest": manifest,
        "blockers": blockers,
        "member_sha256s": member_sha256s,
    }


def _packet_member_merge_payload_variants(
    selected_entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    variants = [
        _packet_member_merge_payload(
            selected_entries,
            payload_codec="raw_member_payload_v1",
        ),
        _packet_member_merge_payload(
            selected_entries,
            payload_codec="source_zip_compressed_stream_v1",
        ),
        _packet_member_merge_binary_payload(
            selected_entries,
            payload_codec="source_zip_compressed_stream_binary_table_v1",
        ),
    ]
    deflate_sequence = _packet_member_merge_deflate_sequence_payload(
        selected_entries,
        payload_codec="fixed_order_raw_deflate_sequence_v1",
    )
    if deflate_sequence is not None:
        variants.append(deflate_sequence)
    return tuple(variants)


def _packet_member_merge_deflate_sequence_payload(
    selected_entries: Sequence[Mapping[str, Any]],
    *,
    payload_codec: str,
) -> dict[str, Any] | None:
    if payload_codec != "fixed_order_raw_deflate_sequence_v1":
        raise FamilyAgnosticMaterializerError(
            f"unsupported packet member merge deflate sequence codec: {payload_codec}"
        )
    if any(int(entry["zip_compress_type"]) != zipfile.ZIP_DEFLATED for entry in selected_entries):
        return None
    table_chunks = [PACKET_MEMBER_MERGE_DEFLATE_SEQUENCE_PAYLOAD_MAGIC]
    table_chunks.append(_encode_uvarint(len(selected_entries)))
    payload_chunks = []
    members = []
    offset = 0
    for entry in selected_entries:
        name = str(entry["name"])
        name_bytes = name.encode("utf-8")
        payload = bytes(entry["zip_compressed_payload"])
        table_chunks.append(_encode_uvarint(len(name_bytes)))
        table_chunks.append(name_bytes)
        payload_chunks.append(payload)
        members.append(
            {
                "name": name,
                "offset": offset,
                "length": len(payload),
                "compressed_sha256": sha256_bytes(payload),
                "sha256": str(entry["payload_sha256"]),
                "uncompressed_length": int(entry["payload_bytes"]),
                "zip_compress_type": zipfile.ZIP_DEFLATED,
            }
        )
        offset += len(payload)
    binary_table = b"".join(table_chunks)
    concatenated_payload = b"".join(payload_chunks)
    merged_payload = binary_table + concatenated_payload
    table = {
        "schema": "packet_member_merge_table.v1",
        "payload_codec": payload_codec,
        "table_format": "uleb_name_raw_deflate_sequence_v1",
        "member_count": len(members),
        "members": members,
        "binary_table_bytes": len(binary_table),
        "binary_table_sha256": sha256_bytes(binary_table),
        "concatenated_payload_bytes": len(concatenated_payload),
        "concatenated_payload_sha256": sha256_bytes(concatenated_payload),
        "merged_payload_sha256": sha256_bytes(merged_payload),
    }
    return {
        "payload_codec": payload_codec,
        "payload": merged_payload,
        "table": table,
    }


def _packet_member_merge_binary_payload(
    selected_entries: Sequence[Mapping[str, Any]],
    *,
    payload_codec: str,
) -> dict[str, Any]:
    if payload_codec != "source_zip_compressed_stream_binary_table_v1":
        raise FamilyAgnosticMaterializerError(
            f"unsupported packet member merge binary codec: {payload_codec}"
        )
    table_chunks = [PACKET_MEMBER_MERGE_BINARY_PAYLOAD_MAGIC]
    table_chunks.append(_encode_uvarint(len(selected_entries)))
    payload_chunks = []
    members = []
    offset = 0
    for entry in selected_entries:
        name = str(entry["name"])
        name_bytes = name.encode("utf-8")
        payload = bytes(entry["zip_compressed_payload"])
        compress_type = int(entry["zip_compress_type"])
        uncompressed_length = int(entry["payload_bytes"])
        table_chunks.append(_encode_uvarint(len(name_bytes)))
        table_chunks.append(name_bytes)
        table_chunks.append(_encode_uvarint(compress_type))
        table_chunks.append(_encode_uvarint(len(payload)))
        table_chunks.append(_encode_uvarint(uncompressed_length))
        payload_chunks.append(payload)
        members.append(
            {
                "name": name,
                "offset": offset,
                "length": len(payload),
                "compressed_sha256": sha256_bytes(payload),
                "sha256": str(entry["payload_sha256"]),
                "uncompressed_length": int(entry["payload_bytes"]),
                "zip_compress_type": int(entry["zip_compress_type"]),
            }
        )
        offset += len(payload)
    binary_table = b"".join(table_chunks)
    concatenated_payload = b"".join(payload_chunks)
    merged_payload = binary_table + concatenated_payload
    table: dict[str, Any] = {
        "schema": "packet_member_merge_table.v1",
        "payload_codec": payload_codec,
        "table_format": "uleb_name_compressed_stream_table_v1",
        "member_count": len(members),
        "members": members,
        "binary_table_bytes": len(binary_table),
        "binary_table_sha256": sha256_bytes(binary_table),
        "concatenated_payload_bytes": len(concatenated_payload),
        "concatenated_payload_sha256": sha256_bytes(concatenated_payload),
        "merged_payload_sha256": sha256_bytes(merged_payload),
    }
    return {
        "payload_codec": payload_codec,
        "payload": merged_payload,
        "table": table,
    }


def _packet_member_merge_payload(
    selected_entries: Sequence[Mapping[str, Any]],
    *,
    payload_codec: str,
) -> dict[str, Any]:
    offset = 0
    members = []
    chunks = []
    for entry in selected_entries:
        name = str(entry["name"])
        if payload_codec == "raw_member_payload_v1":
            payload = bytes(entry["payload"])
            member_row = {
                "name": name,
                "offset": offset,
                "length": len(payload),
                "sha256": sha256_bytes(payload),
            }
        elif payload_codec in {
            "source_zip_compressed_stream_v1",
            "source_zip_compressed_stream_binary_table_v1",
        }:
            payload = bytes(entry["zip_compressed_payload"])
            member_row = {
                "name": name,
                "offset": offset,
                "length": len(payload),
                "compressed_sha256": sha256_bytes(payload),
                "sha256": str(entry["payload_sha256"]),
                "uncompressed_length": int(entry["payload_bytes"]),
                "zip_compress_type": int(entry["zip_compress_type"]),
                "zip_compression_method": str(entry["zip_compression_method"]),
            }
        else:
            raise FamilyAgnosticMaterializerError(
                f"unsupported packet member merge payload codec: {payload_codec}"
            )
        chunks.append(payload)
        members.append(member_row)
        offset += len(payload)
    concatenated_payload = b"".join(chunks)
    table: dict[str, Any] = {
        "schema": "packet_member_merge_table.v1",
        "payload_codec": payload_codec,
        "member_count": len(members),
        "members": members,
        "concatenated_payload_bytes": len(concatenated_payload),
        "concatenated_payload_sha256": sha256_bytes(concatenated_payload),
    }
    table_payload = _canonical_json_bytes(table)
    payload = (
        PACKET_MEMBER_MERGE_PAYLOAD_MAGIC
        + struct.pack("<Q", len(table_payload))
        + table_payload
        + concatenated_payload
    )
    return {
        "payload_codec": payload_codec,
        "payload": payload,
        "table": table | {"merged_payload_sha256": sha256_bytes(payload)},
    }


def parse_packet_member_merge_payload(payload: bytes) -> dict[str, Any]:
    """Parse a packet-member merge payload and reconstruct original members."""

    if payload.startswith(PACKET_MEMBER_MERGE_DEFLATE_SEQUENCE_PAYLOAD_MAGIC):
        return _parse_packet_member_merge_deflate_sequence_payload(payload)
    if payload.startswith(PACKET_MEMBER_MERGE_BINARY_PAYLOAD_MAGIC):
        return _parse_packet_member_merge_binary_payload(payload)
    return _parse_packet_member_merge_json_payload(payload)


def _parse_packet_member_merge_json_payload(payload: bytes) -> dict[str, Any]:
    prefix_len = len(PACKET_MEMBER_MERGE_PAYLOAD_MAGIC)
    if not payload.startswith(PACKET_MEMBER_MERGE_PAYLOAD_MAGIC):
        raise FamilyAgnosticMaterializerError("merged member payload has bad magic")
    if len(payload) < prefix_len + 8:
        raise FamilyAgnosticMaterializerError("merged member payload is truncated")
    table_len = struct.unpack_from("<Q", payload, prefix_len)[0]
    table_start = prefix_len + 8
    table_end = table_start + int(table_len)
    if table_end > len(payload):
        raise FamilyAgnosticMaterializerError("merged member table extends past payload")
    table = json.loads(payload[table_start:table_end].decode("utf-8"))
    if not isinstance(table, Mapping):
        raise FamilyAgnosticMaterializerError("merged member table must be a JSON object")
    concatenated_payload = payload[table_end:]
    if sha256_bytes(concatenated_payload) != _clean_str(table.get("concatenated_payload_sha256")):
        raise FamilyAgnosticMaterializerError("merged member concatenated payload SHA mismatch")
    out: dict[str, bytes] = {}
    rows = table.get("members")
    if not isinstance(rows, Sequence) or isinstance(rows, (bytes, bytearray, str)):
        raise FamilyAgnosticMaterializerError("merged member table has no members array")
    payload_codec = _clean_str(table.get("payload_codec")) or "raw_member_payload_v1"
    for row in rows:
        if not isinstance(row, Mapping):
            raise FamilyAgnosticMaterializerError("merged member table row must be an object")
        name = _clean_str(row.get("name"))
        if name is None:
            raise FamilyAgnosticMaterializerError("merged member table row missing name")
        offset = int(row.get("offset"))
        length = int(row.get("length"))
        if offset < 0 or length < 0 or offset + length > len(concatenated_payload):
            raise FamilyAgnosticMaterializerError(
                f"merged member table row bounds out of range: {name}"
            )
        encoded_member_payload = concatenated_payload[offset:offset + length]
        if payload_codec == "raw_member_payload_v1":
            member_payload = encoded_member_payload
        elif payload_codec in {
            "source_zip_compressed_stream_v1",
            "source_zip_compressed_stream_binary_table_v1",
        }:
            expected_compressed_sha = _clean_str(row.get("compressed_sha256"))
            if (
                expected_compressed_sha is not None
                and sha256_bytes(encoded_member_payload) != expected_compressed_sha
            ):
                raise FamilyAgnosticMaterializerError(
                    f"merged member compressed stream SHA mismatch: {name}"
                )
            member_payload = _decompress_zip_member_payload(
                encoded_member_payload,
                compress_type=int(row.get("zip_compress_type")),
                member_name=name,
            )
        else:
            raise FamilyAgnosticMaterializerError(
                f"unsupported merged member payload codec: {payload_codec}"
            )
        if sha256_bytes(member_payload) != _clean_str(row.get("sha256")):
            raise FamilyAgnosticMaterializerError(
                f"merged member reconstructed payload SHA mismatch: {name}"
            )
        out[name] = member_payload
    return {"table": dict(table), "members": out}


def _parse_packet_member_merge_binary_payload(payload: bytes) -> dict[str, Any]:
    cursor = len(PACKET_MEMBER_MERGE_BINARY_PAYLOAD_MAGIC)
    member_count, cursor = _decode_uvarint(
        payload,
        cursor,
        label="packet_member_merge_member_count",
    )
    members = []
    for _index in range(member_count):
        name_length, cursor = _decode_uvarint(
            payload,
            cursor,
            label="packet_member_merge_name_length",
        )
        name_end = cursor + name_length
        if name_end > len(payload):
            raise FamilyAgnosticMaterializerError(
                "packet member merge binary table name extends past payload"
            )
        name = payload[cursor:name_end].decode("utf-8")
        cursor = name_end
        compress_type, cursor = _decode_uvarint(
            payload,
            cursor,
            label=f"packet_member_merge_compress_type:{name}",
        )
        compressed_length, cursor = _decode_uvarint(
            payload,
            cursor,
            label=f"packet_member_merge_compressed_length:{name}",
        )
        uncompressed_length, cursor = _decode_uvarint(
            payload,
            cursor,
            label=f"packet_member_merge_uncompressed_length:{name}",
        )
        members.append(
            {
                "name": name,
                "zip_compress_type": int(compress_type),
                "length": int(compressed_length),
                "uncompressed_length": int(uncompressed_length),
            }
        )
    binary_table_bytes = cursor
    concatenated_payload = payload[cursor:]
    offset = 0
    out: dict[str, bytes] = {}
    normalized_members = []
    for row in members:
        name = str(row["name"])
        compressed_length = int(row["length"])
        if compressed_length < 0 or offset + compressed_length > len(concatenated_payload):
            raise FamilyAgnosticMaterializerError(
                f"packet member merge binary row bounds out of range: {name}"
            )
        encoded_member_payload = concatenated_payload[offset:offset + compressed_length]
        member_payload = _decompress_zip_member_payload(
            encoded_member_payload,
            compress_type=int(row["zip_compress_type"]),
            member_name=name,
        )
        if len(member_payload) != int(row["uncompressed_length"]):
            raise FamilyAgnosticMaterializerError(
                f"packet member merge binary row length mismatch: {name}"
            )
        normalized_row = {
            "name": name,
            "offset": offset,
            "length": compressed_length,
            "compressed_sha256": sha256_bytes(encoded_member_payload),
            "sha256": sha256_bytes(member_payload),
            "uncompressed_length": len(member_payload),
            "zip_compress_type": int(row["zip_compress_type"]),
        }
        normalized_members.append(normalized_row)
        out[name] = member_payload
        offset += compressed_length
    if offset != len(concatenated_payload):
        raise FamilyAgnosticMaterializerError(
            "packet member merge binary payload has trailing bytes"
        )
    binary_table = payload[:binary_table_bytes]
    table = {
        "schema": "packet_member_merge_table.v1",
        "payload_codec": "source_zip_compressed_stream_binary_table_v1",
        "table_format": "uleb_name_compressed_stream_table_v1",
        "member_count": len(normalized_members),
        "members": normalized_members,
        "binary_table_bytes": len(binary_table),
        "binary_table_sha256": sha256_bytes(binary_table),
        "concatenated_payload_bytes": len(concatenated_payload),
        "concatenated_payload_sha256": sha256_bytes(concatenated_payload),
        "merged_payload_sha256": sha256_bytes(payload),
    }
    return {"table": table, "members": out}


def _parse_packet_member_merge_deflate_sequence_payload(payload: bytes) -> dict[str, Any]:
    cursor = len(PACKET_MEMBER_MERGE_DEFLATE_SEQUENCE_PAYLOAD_MAGIC)
    member_count, cursor = _decode_uvarint(
        payload,
        cursor,
        label="packet_member_merge_deflate_sequence_member_count",
    )
    names = []
    for _index in range(member_count):
        name_length, cursor = _decode_uvarint(
            payload,
            cursor,
            label="packet_member_merge_deflate_sequence_name_length",
        )
        name_end = cursor + name_length
        if name_end > len(payload):
            raise FamilyAgnosticMaterializerError(
                "packet member merge deflate sequence name extends past payload"
            )
        names.append(payload[cursor:name_end].decode("utf-8"))
        cursor = name_end
    stream = payload[cursor:]
    out: dict[str, bytes] = {}
    members = []
    offset = 0
    remaining = stream
    for index, name in enumerate(names):
        decoded, consumed = _decompress_next_zip_deflate_stream(remaining, member_name=name)
        encoded = remaining[:consumed]
        out[name] = decoded
        members.append(
            {
                "name": name,
                "offset": offset,
                "length": len(encoded),
                "compressed_sha256": sha256_bytes(encoded),
                "sha256": sha256_bytes(decoded),
                "uncompressed_length": len(decoded),
                "zip_compress_type": zipfile.ZIP_DEFLATED,
            }
        )
        offset += consumed
        remaining = remaining[consumed:]
        if index == len(names) - 1 and remaining:
            raise FamilyAgnosticMaterializerError(
                "packet member merge deflate sequence has trailing bytes"
            )
    table = {
        "schema": "packet_member_merge_table.v1",
        "payload_codec": "fixed_order_raw_deflate_sequence_v1",
        "table_format": "uleb_name_raw_deflate_sequence_v1",
        "member_count": len(members),
        "members": members,
        "binary_table_bytes": cursor,
        "binary_table_sha256": sha256_bytes(payload[:cursor]),
        "concatenated_payload_bytes": len(stream),
        "concatenated_payload_sha256": sha256_bytes(stream),
        "merged_payload_sha256": sha256_bytes(payload),
    }
    return {"table": table, "members": out}


def _decompress_next_zip_deflate_stream(
    payload: bytes,
    *,
    member_name: str,
) -> tuple[bytes, int]:
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    try:
        decoded = decompressor.decompress(payload)
        decoded += decompressor.flush()
    except zlib.error as exc:
        raise FamilyAgnosticMaterializerError(
            f"merged member deflate stream could not be decompressed: {member_name}"
        ) from exc
    if not decompressor.eof:
        raise FamilyAgnosticMaterializerError(
            f"merged member deflate stream did not terminate: {member_name}"
        )
    consumed = len(payload) - len(decompressor.unused_data)
    if consumed <= 0:
        raise FamilyAgnosticMaterializerError(
            f"merged member deflate stream consumed no bytes: {member_name}"
        )
    return decoded, consumed


def _encode_uvarint(value: int) -> bytes:
    if value < 0:
        raise FamilyAgnosticMaterializerError("uvarint cannot encode negative values")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _decode_uvarint(data: bytes, offset: int, *, label: str) -> tuple[int, int]:
    value = 0
    shift = 0
    cursor = offset
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            raise FamilyAgnosticMaterializerError(f"{label} uvarint too wide")
    raise FamilyAgnosticMaterializerError(f"{label} uvarint truncated")


def _parse_packet_member_merge_payload(payload: bytes) -> dict[str, Any]:
    return parse_packet_member_merge_payload(payload)


def _zip_archive_bytes_with_member_merge(
    archive: Path,
    *,
    selected_member_names: Sequence[str],
    merged_member_name: str,
    merged_payload: bytes,
    compression: int,
    compresslevel: int | None,
) -> bytes:
    selected = set(_require_unique_existing_member_names(archive, selected_member_names))
    output = io.BytesIO()
    wrote_merged_member = False
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(output, "w") as target:
        for info in source.infolist():
            if info.is_dir():
                target.mkdir(_copy_zip_info(info))
                continue
            if info.filename in selected:
                if not wrote_merged_member:
                    out_info = _copy_zip_info(info)
                    out_info.filename = merged_member_name
                    out_info.extra = b""
                    out_info.comment = b""
                    out_info.compress_type = compression
                    kwargs = {}
                    if compression == zipfile.ZIP_DEFLATED and compresslevel is not None:
                        kwargs["compresslevel"] = compresslevel
                    target.writestr(out_info, merged_payload, **kwargs)
                    wrote_merged_member = True
                continue
            target.writestr(_copy_zip_info(info), source.read(info.filename))
    if not wrote_merged_member:
        raise FamilyAgnosticMaterializerError("packet_member_merge wrote no merged member")
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
        return sha256_bytes(_zip_member_compressed_payload(archive, member_name))
    except (OSError, KeyError, FamilyAgnosticMaterializerError, struct.error):
        return None


def _zip_member_compressed_payload(archive: Path, member_name: str) -> bytes:
    raw = archive.read_bytes()
    with zipfile.ZipFile(archive, "r") as zf:
        info = zf.getinfo(member_name)
    return bytes(_raw_zip_member_parts(raw, info)["compressed_payload"])


def _decompress_zip_member_payload(
    payload: bytes,
    *,
    compress_type: int,
    member_name: str,
) -> bytes:
    if compress_type == zipfile.ZIP_STORED:
        return payload
    if compress_type == zipfile.ZIP_DEFLATED:
        try:
            return zlib.decompress(payload, -zlib.MAX_WBITS)
        except zlib.error as exc:
            raise FamilyAgnosticMaterializerError(
                f"merged member deflate stream could not be decompressed: {member_name}"
            ) from exc
    raise FamilyAgnosticMaterializerError(
        "packet_member_merge_v1 does not support reconstructing ZIP compression "
        f"method {compress_type}: {member_name}"
    )


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
