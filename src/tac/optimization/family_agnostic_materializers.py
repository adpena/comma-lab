# SPDX-License-Identifier: MIT
"""Family-agnostic byte-shaving materializers.

These materializers are intentionally conservative: they can emit byte-closed
candidate archives for HNeRV, HNeRV bolt-ons, broader NeRV-family packets, and
non-NeRV ZIP/tensor archives, but they never claim score or exact-eval
readiness without an explicit runtime-consumption proof.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli

from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.repo_io import read_json, sha256_bytes, sha256_file, write_bytes_artifact

ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA = "archive_section_entropy_recode_candidate.v1"
PACKET_MEMBER_RECOMPRESS_SCHEMA = "packet_member_recompress_candidate.v1"
TENSOR_FACTORIZE_SCHEMA = "tensor_factorize_candidate.v1"
ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND = "archive_section_entropy_recode_v1"
PACKET_MEMBER_RECOMPRESS_TARGET_KIND = "packet_member_recompress_v1"
TENSOR_FACTORIZE_TARGET_KIND = "tensor_factorize_v1"
ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER_ID = "archive_section_entropy_recode_adapter"
PACKET_MEMBER_RECOMPRESS_MATERIALIZER_ID = "packet_member_recompress_adapter"
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
    repo_root: str | Path | None = None,
    allow_size_regression: bool = False,
    allow_overwrite: bool = False,
    expected_existing_output_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Recompress one ZIP member without changing its member payload bytes."""

    repo = _repo(repo_root)
    archive = _resolve_path(archive_path, repo=repo)
    output = _resolve_path(output_archive, repo=repo)
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
        levels = (None,) if method == zipfile.ZIP_STORED else tuple(
            ordered_unique([int(level) for level in compresslevels])
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
    best = min(candidates, key=lambda item: (int(item["archive_bytes"]), str(item["compression_method"])))
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
    receiver_verification = verify_runtime_consumption_proof(
        runtime_consumption_proof=runtime_consumption_proof,
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
        "candidate_member": _member_record(output, target_member),
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
        "receiver_contract_satisfied": (
            receiver_verification["receiver_contract_satisfied"] is True
        ),
        "readiness_blockers": readiness_blockers,
        "artifact_write": write_result.__dict__,
        **FALSE_AUTHORITY,
    }


def materialize_archive_section_entropy_recode_candidate(
    *,
    archive_path: str | Path,
    section_manifest: str | Path | Mapping[str, Any],
    output_archive: str | Path,
    section_names: Sequence[str] = (),
    brotli_qualities: Sequence[int] = (9, 10, 11),
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    allow_size_regression: bool = False,
    allow_overwrite: bool = False,
    expected_existing_output_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Recode Brotli-decodable sections from a parser-section manifest."""

    repo = _repo(repo_root)
    archive = _resolve_path(archive_path, repo=repo)
    output = _resolve_path(output_archive, repo=repo)
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
    if changed_lengths:
        blockers.append("section_length_changed_requires_runtime_consumption_proof")
    write_result = write_bytes_artifact(
        output,
        archive_payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_existing_output_sha256,
        min_free_bytes=min_free_bytes,
    )
    candidate_record = _archive_record(output)
    candidate_member = _member_record(output, target_member)
    receiver_verification = verify_runtime_consumption_proof(
        runtime_consumption_proof=runtime_consumption_proof,
        required_candidate_archive_sha256=candidate_record["sha256"],
        required_candidate_member_sha256=candidate_member["sha256"],
        repo_root=repo,
    )
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
        "source_member": _member_record(archive, target_member),
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
        "readiness_blockers": readiness_blockers,
        "artifact_write": write_result.__dict__,
        **FALSE_AUTHORITY,
    }


def materialize_tensor_factorize_candidate(
    *,
    archive_path: str | Path,
    tensor_manifest: str | Path | Mapping[str, Any],
    factorization_contract: str | Path | Mapping[str, Any],
    output_archive: str | Path,
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    allow_size_regression: bool = False,
    allow_overwrite: bool = False,
    expected_existing_output_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Replace a NumPy tensor member with a deterministic low-rank NPZ packet."""

    import numpy as np

    repo = _repo(repo_root)
    archive = _resolve_path(archive_path, repo=repo)
    output = _resolve_path(output_archive, repo=repo)
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
    factor_payload = _npz_bytes(
        {
            "schema": TENSOR_FACTORIZE_SCHEMA,
            "source_shape": list(tensor.shape),
            "source_dtype": str(tensor.dtype),
            "rank": rank,
        },
        u=u[:, :rank].astype(np.float32),
        s=s[:rank].astype(np.float32),
        vt=vt[:rank, :].astype(np.float32),
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
    blockers = ["tensor_factorized_payload_requires_cooperative_receiver"]
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
    receiver_verification = verify_runtime_consumption_proof(
        runtime_consumption_proof=runtime_consumption_proof,
        required_candidate_archive_sha256=candidate_record["sha256"],
        required_candidate_member_sha256=candidate_member["sha256"],
        repo_root=repo,
    )
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
        "source_member": _member_record(archive, member),
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
        "readiness_blockers": readiness_blockers,
        "artifact_write": write_result.__dict__,
        **FALSE_AUTHORITY,
    }


def verify_runtime_consumption_proof(
    *,
    runtime_consumption_proof: str | Path | Mapping[str, Any] | None,
    required_candidate_archive_sha256: str | None = None,
    required_candidate_member_sha256: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the generic receiver/runtime proof used by these materializers."""

    repo = _repo(repo_root)
    if runtime_consumption_proof is None:
        return {
            "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
            "receiver_contract_satisfied": False,
            "proof_present": False,
            "blockers": ["runtime_consumption_proof_missing"],
        }
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
    return {
        "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
        "receiver_contract_satisfied": not blockers,
        "proof_present": True,
        "proof_schema": proof.get("schema"),
        "candidate_archive_sha256": archive_sha,
        "candidate_member_sha256": member_sha,
        "blockers": ordered_unique(blockers),
    }


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
    return {
        "name": member_name,
        "bytes": len(member_payload),
        "sha256": sha256_bytes(member_payload),
        "zip_compression_method": _compression_method_name(compress_type),
        "zip_compressed_bytes": compress_size,
    }


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
    return tuple(ordered_unique(methods))


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
    for quality in ordered_unique([int(item) for item in qualities]):
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


def _npz_bytes(metadata: Mapping[str, Any], **arrays: Any) -> bytes:
    import numpy as np

    output = io.BytesIO()
    np.savez_compressed(output, metadata=str(dict(metadata)), **arrays)
    return output.getvalue()
