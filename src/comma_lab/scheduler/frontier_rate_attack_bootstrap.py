# SPDX-License-Identifier: MIT
"""Bootstrap queue-owned final-rate attacks from frontier archive evidence.

This module is intentionally a thin compiler around existing scheduler and
family-agnostic materializer surfaces. It does not introduce a new executor or
score authority; it turns canonical frontier/archive evidence into an
``experiment_queue.v1`` that existing local workers can execute.
"""

from __future__ import annotations

import json
import re
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.repo_io import sha256_file

from .byte_shaving_campaign_queue import (
    MATERIALIZER_BACKLOG_SCHEMA,
    MATERIALIZER_CONTEXTS_SCHEMA,
    build_materializer_execution_queue,
    build_materializer_work_queue,
    materializer_contexts_from_payload,
)
from .byte_shaving_materializer_registry import (
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    TENSOR_FACTORIZE_TARGET_KIND,
    registry_manifest,
)
from .experiment_queue import ExperimentQueueError, normalize_queue_definition

BOOTSTRAP_SCHEMA = "frontier_final_rate_attack_bootstrap.v1"
FRONTIER_ARCHIVE_RESOLUTION_SCHEMA = "frontier_archive_resolution.v1"
FRONTIER_ARCHIVE_RECORD_SCHEMA = "frontier_rate_attack_archive_record.v1"
DEFAULT_FRONTIER_POINTER = ".omx/state/canonical_frontier_pointer.json"
DEFAULT_EXECUTABLE_TARGET_KINDS = (
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
)
DEFAULT_OPTIONAL_TARGET_KINDS = (
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    TENSOR_FACTORIZE_TARGET_KIND,
)
_AXIS_POINTER_KEYS = {
    "contest_cpu": "our_local_frontier_contest_cpu",
    "contest-cpu": "our_local_frontier_contest_cpu",
    "cpu": "our_local_frontier_contest_cpu",
    "contest_cuda": "our_local_frontier_contest_cuda",
    "contest-cuda": "our_local_frontier_contest_cuda",
    "cuda": "our_local_frontier_contest_cuda",
}


class FrontierRateAttackBootstrapError(ExperimentQueueError):
    """Raised when frontier-rate bootstrap input would be ambiguous or unsafe."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _clean_id(value: str, *, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return cleaned or fallback


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FrontierRateAttackBootstrapError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrontierRateAttackBootstrapError(f"{path}: expected JSON object")
    return payload


def _zip_member_records(path: Path) -> list[dict[str, Any]]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            records = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                records.append(
                    {
                        "name": info.filename,
                        "compress_type": info.compress_type,
                        "compress_size": info.compress_size,
                        "file_size": info.file_size,
                        "crc": f"{info.CRC:08x}",
                        "extra_bytes": len(info.extra or b""),
                        "comment_bytes": len(info.comment or b""),
                    }
                )
    except zipfile.BadZipFile as exc:
        raise FrontierRateAttackBootstrapError(f"{path}: not a readable ZIP archive") from exc
    if not records:
        raise FrontierRateAttackBootstrapError(f"{path}: ZIP archive has no file members")
    return records


def archive_record(
    *,
    label: str,
    archive_path: str | Path,
    repo_root: str | Path,
    source_kind: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    """Return a checked archive record with ZIP member metadata."""

    repo = Path(repo_root)
    archive = _resolve_path(archive_path, repo_root=repo)
    if not archive.is_file():
        raise FrontierRateAttackBootstrapError(f"archive not found: {archive}")
    size = archive.stat().st_size
    if expected_bytes is not None and size != int(expected_bytes):
        raise FrontierRateAttackBootstrapError(
            f"{archive}: byte size mismatch expected={expected_bytes} actual={size}"
        )
    digest = sha256_file(archive)
    if expected_sha256 is not None and digest != expected_sha256:
        raise FrontierRateAttackBootstrapError(
            f"{archive}: sha256 mismatch expected={expected_sha256} actual={digest}"
        )
    members = _zip_member_records(archive)
    return {
        "schema": FRONTIER_ARCHIVE_RECORD_SCHEMA,
        "label": _clean_id(label, fallback="archive"),
        "path": _repo_rel(archive, repo),
        "absolute_path": archive.as_posix(),
        "source_kind": source_kind,
        "bytes": size,
        "sha256": digest,
        "zip_member_count": len(members),
        "zip_members": members,
        **FALSE_AUTHORITY,
    }


def parse_archive_spec(spec: str, *, repo_root: str | Path) -> dict[str, Any]:
    """Parse ``label=path`` archive specs into checked archive records."""

    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise FrontierRateAttackBootstrapError(
                "archive specs must be label=path when '=' is present"
            )
    else:
        raw_path = spec.strip()
        if not raw_path:
            raise FrontierRateAttackBootstrapError("archive spec must not be empty")
        label = Path(raw_path).stem
    return archive_record(
        label=label,
        archive_path=raw_path,
        repo_root=repo_root,
        source_kind="explicit_archive_spec",
    )


def _frontier_pointer_entry(pointer: Mapping[str, Any], axis: str) -> Mapping[str, Any]:
    key = _AXIS_POINTER_KEYS.get(axis.strip().lower())
    if key is None:
        raise FrontierRateAttackBootstrapError(
            f"unsupported frontier axis {axis!r}; expected one of {sorted(_AXIS_POINTER_KEYS)}"
        )
    entry = pointer.get(key)
    if not isinstance(entry, Mapping):
        raise FrontierRateAttackBootstrapError(f"frontier pointer missing {key}")
    return entry


def _frontier_expected_bytes(entry: Mapping[str, Any]) -> int | None:
    extra = entry.get("extra")
    if isinstance(extra, Mapping):
        value = extra.get("archive_bytes")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _candidate_archive_path_from_request(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Path | None:
    for key in ("archive_path", "canonical_path"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidate = _resolve_path(value, repo_root=repo_root)
            if candidate.name == "archive.zip":
                return candidate
            if candidate.is_file():
                return candidate
    submission_dir = payload.get("submission_dir")
    if isinstance(submission_dir, str) and submission_dir.strip():
        candidate = _resolve_path(submission_dir, repo_root=repo_root) / "archive.zip"
        return candidate
    return None


def _request_file_matches_frontier(
    payload: Mapping[str, Any],
    *,
    expected_sha256: str,
    expected_bytes: int | None,
) -> bool:
    sha_values = [
        payload.get("archive_sha256"),
        payload.get("expected_archive_sha256"),
        payload.get("submission_dir_zip_sha256"),
    ]
    if expected_sha256 not in {value for value in sha_values if isinstance(value, str)}:
        return False
    byte_values = [
        payload.get("archive_bytes"),
        payload.get("archive_size_bytes"),
        payload.get("bytes"),
    ]
    if expected_bytes is None:
        return True
    return any(value == expected_bytes for value in byte_values if isinstance(value, int))


def _request_roots(repo_root: Path, roots: Sequence[str | Path]) -> list[Path]:
    if roots:
        return [_resolve_path(root, repo_root=repo_root) for root in roots]
    return [
        repo_root / "experiments" / "results" / "modal_auth_eval_cpu",
        repo_root / "experiments" / "results" / "modal_auth_eval",
        repo_root / "experiments" / "results" / "modal_auth_eval_cuda",
    ]


def resolve_current_frontier_archive(
    *,
    repo_root: str | Path,
    frontier_axis: str = "contest_cpu",
    pointer_path: str | Path = DEFAULT_FRONTIER_POINTER,
    request_search_roots: Sequence[str | Path] = (),
    archive_search_roots: Sequence[str | Path] = (),
    max_archive_candidates: int = 512,
) -> dict[str, Any]:
    """Resolve the canonical frontier pointer to exactly one archive path."""

    repo = Path(repo_root)
    pointer_file = _resolve_path(pointer_path, repo_root=repo)
    pointer = _load_json(pointer_file)
    entry = _frontier_pointer_entry(pointer, frontier_axis)
    expected_sha256 = entry.get("archive_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise FrontierRateAttackBootstrapError(
            f"{pointer_file}: frontier entry for {frontier_axis} has no archive_sha256"
        )
    expected_bytes = _frontier_expected_bytes(entry)
    matches: list[dict[str, Any]] = []
    inspected_request_files = 0
    for root in _request_roots(repo, request_search_roots):
        if not root.exists():
            continue
        for request_file in root.rglob("*.json"):
            if "request" not in request_file.name:
                continue
            inspected_request_files += 1
            try:
                payload = _load_json(request_file)
            except FrontierRateAttackBootstrapError:
                continue
            if not _request_file_matches_frontier(
                payload,
                expected_sha256=expected_sha256,
                expected_bytes=expected_bytes,
            ):
                continue
            archive_path = _candidate_archive_path_from_request(payload, repo_root=repo)
            if archive_path is None or not archive_path.is_file():
                continue
            if expected_bytes is not None and archive_path.stat().st_size != expected_bytes:
                continue
            if sha256_file(archive_path) != expected_sha256:
                continue
            matches.append(
                {
                    "path": _repo_rel(archive_path, repo),
                    "absolute_path": archive_path.as_posix(),
                    "source": "auth_eval_request",
                    "request_path": _repo_rel(request_file, repo),
                }
            )

    if not matches:
        roots = [_resolve_path(root, repo_root=repo) for root in archive_search_roots]
        inspected_archives = 0
        for root in roots:
            if not root.exists():
                continue
            for archive_path in root.rglob("archive.zip"):
                inspected_archives += 1
                if inspected_archives > max_archive_candidates:
                    raise FrontierRateAttackBootstrapError(
                        "frontier archive fallback search exceeded "
                        f"max_archive_candidates={max_archive_candidates}"
                    )
                if expected_bytes is not None and archive_path.stat().st_size != expected_bytes:
                    continue
                if sha256_file(archive_path) != expected_sha256:
                    continue
                matches.append(
                    {
                        "path": _repo_rel(archive_path, repo),
                        "absolute_path": archive_path.as_posix(),
                        "source": "bounded_archive_search",
                        "request_path": None,
                    }
                )

    unique = {item["absolute_path"]: item for item in matches}
    if not unique:
        raise FrontierRateAttackBootstrapError(
            f"could not resolve current {frontier_axis} frontier archive "
            f"sha256={expected_sha256} bytes={expected_bytes}"
        )
    if len(unique) != 1:
        paths = sorted(unique)
        raise FrontierRateAttackBootstrapError(
            "frontier archive resolution is ambiguous: " + ", ".join(paths)
        )
    match = next(iter(unique.values()))
    record = archive_record(
        label=f"current_{frontier_axis}_frontier",
        archive_path=match["absolute_path"],
        repo_root=repo,
        source_kind="canonical_frontier_pointer",
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    return {
        "schema": FRONTIER_ARCHIVE_RESOLUTION_SCHEMA,
        "frontier_axis": frontier_axis,
        "pointer_path": _repo_rel(pointer_file, repo),
        "archive_sha256": expected_sha256,
        "archive_bytes": expected_bytes,
        "score": entry.get("score"),
        "evidence_grade": entry.get("evidence_grade"),
        "hardware_substrate": entry.get("hardware_substrate"),
        "measured_at_utc": entry.get("measured_at_utc"),
        "inspected_request_files": inspected_request_files,
        "match": match,
        "archive_record": record,
        **FALSE_AUTHORITY,
    }


def _adapter_by_target_kind(target_kind: str) -> dict[str, Any]:
    for row in registry_manifest()["adapters"]:
        if row.get("target_kind") == target_kind:
            return dict(row)
    raise FrontierRateAttackBootstrapError(f"materializer target kind is not registered: {target_kind}")


def _archive_specs(records: Sequence[Mapping[str, Any]]) -> list[str]:
    specs = []
    for record in records:
        label = str(record.get("label") or "").strip()
        path = str(record.get("path") or "").strip()
        if not label or not path:
            raise FrontierRateAttackBootstrapError("archive records require label and path")
        specs.append(f"{label}={path}")
    return ordered_unique(specs)


def _shared_single_member_name(records: Sequence[Mapping[str, Any]]) -> str | None:
    names: list[str] = []
    for record in records:
        members = record.get("zip_members")
        if not isinstance(members, list) or len(members) != 1:
            return None
        member = members[0]
        if not isinstance(member, Mapping):
            return None
        name = member.get("name")
        if not isinstance(name, str) or not name:
            return None
        names.append(name)
    unique = ordered_unique(names)
    return unique[0] if len(unique) == 1 else None


def _target_context(
    *,
    target_kind: str,
    archive_records: Sequence[Mapping[str, Any]],
    output_root: Path,
    member_name: str | None,
    section_manifest: str | None,
    section_names: Sequence[str],
    tensor_manifest: str | None,
    factorization_contract: str | None,
    tensor_factorize_rank: int | None,
    zip_compression_methods: Sequence[str],
    zip_compresslevels: Sequence[int],
    min_free_bytes: int,
    allow_overwrite: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    context: dict[str, Any] = {
        "sweep_archive_specs": _archive_specs(archive_records),
        "sweep_output_dir": (output_root / target_kind).as_posix(),
        "sweep_output_json": (output_root / target_kind / "sweep.json").as_posix(),
        "sweep_observation_jsonl": (
            output_root / target_kind / "observations.jsonl"
        ).as_posix(),
        "min_free_bytes": min_free_bytes,
        "allow_overwrite": allow_overwrite,
        **FALSE_AUTHORITY,
    }
    blockers: list[str] = []
    if target_kind == PACKET_MEMBER_RECOMPRESS_TARGET_KIND:
        if member_name is None:
            blockers.append("packet_member_recompress_requires_single_shared_member_or_member_name")
        else:
            context["member_name"] = member_name
        context["zip_compression_methods"] = list(zip_compression_methods)
        context["zip_compresslevels"] = [str(level) for level in zip_compresslevels]
    elif target_kind == PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND:
        if member_name is not None:
            context["member_name"] = member_name
        else:
            context["all_members"] = True
            context["member_selection"] = "all_members"
    elif target_kind == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND:
        if section_manifest is None:
            blockers.append("archive_section_entropy_recode_requires_section_manifest")
        else:
            context["section_manifest"] = section_manifest
            context["section_names"] = list(section_names)
    elif target_kind == TENSOR_FACTORIZE_TARGET_KIND:
        if tensor_manifest is None:
            blockers.append("tensor_factorize_requires_tensor_manifest")
        else:
            context["tensor_manifest"] = tensor_manifest
        if factorization_contract is None and tensor_factorize_rank is None:
            blockers.append("tensor_factorize_requires_factorization_contract_or_rank")
        elif factorization_contract is not None:
            context["factorization_contract"] = factorization_contract
        else:
            context["rank"] = tensor_factorize_rank
    else:
        blockers.append(f"unsupported_frontier_rate_attack_target:{target_kind}")
    return (None, blockers) if blockers else (context, [])


def build_frontier_rate_attack_payloads(
    *,
    repo_root: str | Path,
    queue_id: str,
    archive_records: Sequence[Mapping[str, Any]],
    results_root: str | Path,
    target_kinds: Sequence[str] = DEFAULT_EXECUTABLE_TARGET_KINDS,
    include_optional_target_blockers: bool = True,
    member_name: str | None = None,
    section_manifest: str | None = None,
    section_names: Sequence[str] = (),
    tensor_manifest: str | None = None,
    factorization_contract: str | None = None,
    tensor_factorize_rank: int | None = None,
    zip_compression_methods: Sequence[str] = ("stored", "deflated"),
    zip_compresslevels: Sequence[int] = (1, 6, 9),
    min_free_bytes: int = 0,
    allow_overwrite: bool = False,
    local_cpu_concurrency: int = 1,
    lane_id: str | None = None,
    source_work_queue_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compile archives and materializer targets into durable queue payloads."""

    repo = Path(repo_root)
    if not queue_id.strip():
        raise FrontierRateAttackBootstrapError("queue_id must be non-empty")
    if not archive_records:
        raise FrontierRateAttackBootstrapError("at least one archive record is required")
    if local_cpu_concurrency < 1:
        raise FrontierRateAttackBootstrapError("local_cpu_concurrency must be >= 1")
    checked_records = [dict(record) for record in archive_records]
    for index, record in enumerate(checked_records):
        require_no_truthy_authority_fields(record, context=f"archive_records[{index}]")
    output_root = _resolve_path(results_root, repo_root=repo) / queue_id
    shared_member = member_name or _shared_single_member_name(checked_records)
    requested_targets = ordered_unique(
        [
            *target_kinds,
            *(DEFAULT_OPTIONAL_TARGET_KINDS if include_optional_target_blockers else ()),
        ]
    )
    contexts_rows: list[dict[str, Any]] = []
    backlog_rows: list[dict[str, Any]] = []
    target_omissions: list[dict[str, Any]] = []
    for rank, target_kind in enumerate(requested_targets, start=1):
        adapter = _adapter_by_target_kind(target_kind)
        backlog_key = f"frontier_rate_attack:{target_kind}"
        context, blockers = _target_context(
            target_kind=target_kind,
            archive_records=checked_records,
            output_root=output_root,
            member_name=shared_member,
            section_manifest=section_manifest,
            section_names=section_names,
            tensor_manifest=tensor_manifest,
            factorization_contract=factorization_contract,
            tensor_factorize_rank=tensor_factorize_rank,
            zip_compression_methods=zip_compression_methods,
            zip_compresslevels=zip_compresslevels,
            min_free_bytes=min_free_bytes,
            allow_overwrite=allow_overwrite,
        )
        if blockers:
            target_omissions.append(
                apply_proxy_evidence_boundary(
                    {
                        "schema": "frontier_rate_attack_target_omission.v1",
                        "target_kind": target_kind,
                        "materializer_id": adapter.get("materializer_id"),
                        "blockers": ordered_unique(blockers),
                        **FALSE_AUTHORITY,
                    },
                    dispatch_blockers=blockers,
                )
            )
            continue
        assert context is not None
        contexts_rows.append(
            {
                "backlog_key": backlog_key,
                "target_kind": target_kind,
                "materializer_id": adapter.get("materializer_id"),
                "context": context,
            }
        )
        backlog_rows.append(
            apply_proxy_evidence_boundary(
                {
                    "schema": "byte_shaving_materializer_backlog_row.v1",
                    "backlog_key": backlog_key,
                    "backlog_rank": rank,
                    "gap_class": "frontier_final_rate_attack_materializer_sweep",
                    "unit_kind": adapter.get("unit_kind"),
                    "operation_family": adapter.get("operation_family"),
                    "target_kind": target_kind,
                    "materializer_id": adapter.get("materializer_id"),
                    "receiver_contract_id": adapter.get("receiver_contract_id"),
                    "receiver_contract_kind": adapter.get("receiver_contract_kind"),
                    "receiver_contract_status": "local_receiver_proof_required",
                    "cooperative_receiver_required": adapter.get("cooperative_receiver_required"),
                    "materialization_resource_kind": adapter.get("materialization_resource_kind")
                    or "local_cpu",
                    "suggested_materializer_count": 1,
                    "suggested_materializers": [adapter],
                    "blocked_row_count": 1,
                    "blocked_resolution_count": 1,
                    "selected_operation_count": len(checked_records),
                    "affected_unit_count": len(checked_records),
                    "candidate_saved_bytes_sum": 0,
                    "expected_score_gain_sum": 0.0,
                    "source_unit_ids": [record["label"] for record in checked_records],
                    "source_selection_ids": [record["label"] for record in checked_records],
                    "source_selection_samples": [
                        {
                            "selection_id": record["label"],
                            "selection_kind": "frontier_archive",
                            "archive_sha256": record["sha256"],
                            "archive_bytes": record["bytes"],
                        }
                        for record in checked_records[:8]
                    ],
                    **FALSE_AUTHORITY,
                },
                dispatch_blockers=("frontier_rate_attack_local_materializer_sweep_only",),
            )
        )
    if not backlog_rows:
        raise FrontierRateAttackBootstrapError(
            "no executable frontier-rate materializer targets; blockers: "
            + json.dumps(target_omissions, sort_keys=True)
        )
    contexts_payload = apply_proxy_evidence_boundary(
        {
            "schema": MATERIALIZER_CONTEXTS_SCHEMA,
            "generated_at_utc": _utc_now(),
            "queue_id": queue_id,
            "rows": contexts_rows,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=("frontier_rate_attack_contexts_are_local_only",),
    )
    contexts = materializer_contexts_from_payload(contexts_payload)
    backlog = apply_proxy_evidence_boundary(
        {
            "schema": MATERIALIZER_BACKLOG_SCHEMA,
            "tool": "comma_lab.scheduler.frontier_rate_attack_bootstrap",
            "generated_at_utc": _utc_now(),
            "backlog_row_count": len(backlog_rows),
            "rows": backlog_rows,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=("frontier_rate_attack_backlog_is_planning_only",),
    )
    work_queue = build_materializer_work_queue(
        backlog,
        repo_root=repo,
        contexts=contexts,
        source_plan_path=None,
    )
    queue = build_materializer_execution_queue(
        work_queue,
        queue_id=queue_id,
        repo_root=repo,
        lane_id=lane_id,
        source_work_queue_path=source_work_queue_path,
        local_cpu_concurrency=local_cpu_concurrency,
        resource_concurrency={"local_cpu": local_cpu_concurrency},
        include_exact_readiness_followup=False,
    )
    queue_metadata = apply_proxy_evidence_boundary(
        {
            "schema": BOOTSTRAP_SCHEMA,
            "archive_count": len(checked_records),
            "archive_labels": [record["label"] for record in checked_records],
            "target_kinds": [row["target_kind"] for row in backlog_rows],
            "target_omissions": target_omissions,
            "allowed_use": "local_final_rate_attack_materializer_sweep_only",
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(
            "frontier_rate_attack_is_local_materializer_signal_only",
            "exact_auth_eval_required_before_score_claim",
        ),
    )
    queue = normalize_queue_definition(queue)
    for experiment in queue["experiments"]:
        metadata = dict(experiment.get("metadata") or {})
        metadata["frontier_rate_attack_bootstrap"] = queue_metadata
        experiment["metadata"] = metadata
    queue = normalize_queue_definition(queue)
    bootstrap = apply_proxy_evidence_boundary(
        {
            "schema": BOOTSTRAP_SCHEMA,
            "generated_at_utc": _utc_now(),
            "queue_id": queue_id,
            "archive_count": len(checked_records),
            "archives": checked_records,
            "executable_target_count": len(backlog_rows),
            "executable_target_kinds": [row["target_kind"] for row in backlog_rows],
            "target_omissions": target_omissions,
            "results_root": _repo_rel(output_root, repo),
            "shared_member_name": shared_member,
            "queue_schema": queue.get("schema"),
            "experiment_count": len(queue.get("experiments", [])),
            "step_count": sum(len(exp.get("steps", [])) for exp in queue.get("experiments", [])),
            "controls": queue.get("controls"),
            "allowed_use": "local_final_rate_attack_materializer_sweep_only",
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(
            "frontier_rate_attack_is_local_materializer_signal_only",
            "exact_auth_eval_required_before_score_claim",
        ),
    )
    return {
        "bootstrap": bootstrap,
        "contexts": contexts_payload,
        "backlog": backlog,
        "work_queue": work_queue,
        "queue": queue,
    }


__all__ = [
    "BOOTSTRAP_SCHEMA",
    "DEFAULT_EXECUTABLE_TARGET_KINDS",
    "DEFAULT_FRONTIER_POINTER",
    "DEFAULT_OPTIONAL_TARGET_KINDS",
    "FRONTIER_ARCHIVE_RECORD_SCHEMA",
    "FRONTIER_ARCHIVE_RESOLUTION_SCHEMA",
    "FrontierRateAttackBootstrapError",
    "archive_record",
    "build_frontier_rate_attack_payloads",
    "parse_archive_spec",
    "resolve_current_frontier_archive",
]
