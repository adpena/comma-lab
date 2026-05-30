#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run a grouped family-agnostic archive-state materializer request."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.byte_shaving_campaign_queue import (  # noqa: E402
    GROUPED_ARCHIVE_STATE_MATERIALIZER_CHAIN_SCHEMA,
    GROUPED_ARCHIVE_STATE_MATERIALIZER_REQUEST_SCHEMA,
    GROUPED_ARCHIVE_STATE_SUPPORTED_TARGET_KINDS,
    MATERIALIZER_WORK_QUEUE_SCHEMA,
)
from tac.optimization.archive_bound_candidate_contract import (  # noqa: E402
    archive_bound_candidate_contract_fields_for_row,
)
from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY  # noqa: E402
from tac.optimization.proxy_candidate_contract import (  # noqa: E402
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _resolve(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _flag_value(command: Sequence[str], flag: str) -> str | None:
    for index, item in enumerate(command[:-1]):
        if item == flag:
            return str(command[index + 1])
    return None


def _set_flag(command: list[str], flag: str, value: str) -> list[str]:
    out = list(command)
    for index, item in enumerate(out[:-1]):
        if item == flag:
            out[index + 1] = value
            return out
    out.extend([flag, value])
    return out


def _drop_flags_with_values(command: list[str], flags: set[str]) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        if item in flags:
            index += 2
            continue
        out.append(item)
        index += 1
    return out


def _request_by_id(work_queue: Mapping[str, Any], request_id: str) -> Mapping[str, Any]:
    for request in _as_list(work_queue.get("grouped_archive_state_materializer_requests")):
        if isinstance(request, Mapping) and request.get("request_id") == request_id:
            return request
    raise ValueError(f"grouped request not found: {request_id}")


def _rows_by_work_id(work_queue: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in _as_list(work_queue.get("rows")):
        if isinstance(row, Mapping) and isinstance(row.get("work_id"), str):
            rows[str(row["work_id"])] = row
    return rows


def _validate_request(request: Mapping[str, Any]) -> None:
    if request.get("schema") != GROUPED_ARCHIVE_STATE_MATERIALIZER_REQUEST_SCHEMA:
        raise ValueError(f"expected schema {GROUPED_ARCHIVE_STATE_MATERIALIZER_REQUEST_SCHEMA}")
    require_no_truthy_authority_fields(request, context="grouped_archive_state_request")
    if request.get("executable") is not True or request.get("grouped_execution_ready") is not True:
        blockers = ", ".join(str(item) for item in _as_list(request.get("grouped_execution_blockers")))
        raise ValueError(f"grouped request is not executable: {blockers or '<no blockers recorded>'}")


def _rewrite_step_command(
    command: Sequence[Any],
    *,
    archive_path: Path,
    output_archive: Path,
    output_manifest: Path,
    allow_overwrite: bool,
) -> list[str]:
    out = [str(item) for item in command]
    out = _drop_flags_with_values(
        out,
        {
            "--expected-existing-output-sha256",
            "--expected-existing-manifest-sha256",
            "--expected-existing-runtime-consumption-proof-sha256",
        },
    )
    out = _set_flag(out, "--archive-path", archive_path.as_posix())
    out = _set_flag(out, "--output-archive", output_archive.as_posix())
    out = _set_flag(out, "--output-manifest", output_manifest.as_posix())
    runtime_proof = output_manifest.with_name(
        f"{output_manifest.stem}.runtime_consumption_proof.json"
    )
    out = _drop_flags_with_values(
        out,
        {"--runtime-consumption-proof", "--runtime-consumption-proof-out"},
    )
    out.extend(["--runtime-consumption-proof-out", runtime_proof.as_posix()])
    if allow_overwrite and "--allow-overwrite" not in out:
        out.append("--allow-overwrite")
    return out


def _archive_record(path: Path, *, repo_root: Path) -> dict[str, Any]:
    return {
        "path": _repo_rel(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _archive_bound_contract_fields_for_chain(
    *,
    request_id: str,
    source_record: Mapping[str, Any],
    final_record: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
    blockers: Sequence[str],
    repo_root: Path,
) -> dict[str, Any]:
    final_runtime_proof_path = None
    if operations:
        final_runtime_proof_path = operations[-1].get("runtime_consumption_proof_path")
    ready = not blockers
    return archive_bound_candidate_contract_fields_for_row(
        {
            "archive_native_transform_kind": GROUPED_ARCHIVE_STATE_MATERIALIZER_CHAIN_SCHEMA,
            "target_kind": GROUPED_ARCHIVE_STATE_MATERIALIZER_CHAIN_SCHEMA,
            "candidate_id": request_id,
            "candidate_chain_id": request_id,
            "candidate_archive": dict(final_record),
            "source_archive": dict(source_record),
            "byte_closed_candidate_emitted": ready,
            "byte_closed_candidate_materialized": ready,
            "candidate_archive_materialized": ready,
            "runtime_consumption_proof_ready": ready,
            "runtime_consumption_proof_path": final_runtime_proof_path,
            "receiver_contract_kind": "grouped_archive_state_runtime_consumption_chain",
            "receiver_contract_satisfied": ready,
            "runtime_adapter_ready": ready,
            "readiness_blockers": list(blockers),
            "saved_bytes": int(source_record["bytes"]) - int(final_record["bytes"]),
            **FALSE_AUTHORITY,
        },
        repo_root=repo_root,
        selected_transform_kind=GROUPED_ARCHIVE_STATE_MATERIALIZER_CHAIN_SCHEMA,
        family_id="grouped_family_agnostic_materializer",
        candidate_chain_id=request_id,
    )


def run_grouped_request(
    *,
    work_queue_path: Path,
    request_id: str,
    output_dir: Path,
    output_manifest: Path,
    repo_root: Path,
    allow_overwrite: bool,
) -> dict[str, Any]:
    work_queue = _read_json(work_queue_path)
    if work_queue.get("schema") != MATERIALIZER_WORK_QUEUE_SCHEMA:
        raise ValueError(f"expected schema {MATERIALIZER_WORK_QUEUE_SCHEMA}")
    require_no_truthy_authority_fields(work_queue, context="materializer_work_queue")
    request = _request_by_id(work_queue, request_id)
    _validate_request(request)
    rows_by_id = _rows_by_work_id(work_queue)
    ordered_work_ids = [str(item) for item in _as_list(request.get("ordered_work_ids"))]
    if len(ordered_work_ids) < 2:
        raise ValueError("grouped request must contain at least two work rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    current_archive: Path | None = None
    source_archive: Path | None = None
    operations: list[dict[str, Any]] = []
    blockers: list[str] = []
    for index, work_id in enumerate(ordered_work_ids, start=1):
        row = rows_by_id.get(work_id)
        if row is None:
            raise ValueError(f"grouped request references missing work row: {work_id}")
        target_kind = str(row.get("target_kind") or "")
        if target_kind not in GROUPED_ARCHIVE_STATE_SUPPORTED_TARGET_KINDS:
            raise ValueError(f"grouped target kind is not supported: {target_kind}")
        if row.get("executable") is not True:
            raise ValueError(f"grouped work row is not executable: {work_id}")
        command = row.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError(f"grouped work row has no command: {work_id}")
        initial_archive = _flag_value([str(item) for item in command], "--archive-path")
        if initial_archive is None:
            raise ValueError(f"grouped work row command missing --archive-path: {work_id}")
        if current_archive is None:
            current_archive = _resolve(initial_archive, repo_root=repo_root)
            source_archive = current_archive
        step_archive = output_dir / f"step_{index:02d}_{work_id}.archive.zip"
        step_manifest = output_dir / f"step_{index:02d}_{work_id}.json"
        step_command = _rewrite_step_command(
            command,
            archive_path=current_archive,
            output_archive=step_archive,
            output_manifest=step_manifest,
            allow_overwrite=allow_overwrite,
        )
        completed = subprocess.run(
            step_command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(
                f"grouped materializer step failed for {work_id}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        step_payload = _read_json(step_manifest)
        require_no_truthy_authority_fields(
            step_payload,
            context=f"grouped_materializer_step.{work_id}",
        )
        step_blockers = [str(item) for item in _as_list(step_payload.get("readiness_blockers"))]
        if step_payload.get("byte_closed_candidate_emitted") is not True:
            blockers.append(f"grouped_step_byte_closed_candidate_missing:{work_id}")
        if step_payload.get("receiver_contract_satisfied") is not True:
            blockers.append(f"grouped_step_receiver_contract_unsatisfied:{work_id}")
        blockers.extend(f"grouped_step_blocker:{work_id}:{item}" for item in step_blockers)
        operations.append(
            {
                "index": index,
                "work_id": work_id,
                "target_kind": target_kind,
                "command": step_command,
                "manifest_path": _repo_rel(step_manifest, repo_root),
                "candidate_archive": _archive_record(step_archive, repo_root=repo_root),
                "runtime_consumption_proof_path": step_payload.get(
                    "runtime_consumption_proof_path"
                ),
                "receiver_contract_satisfied": step_payload.get(
                    "receiver_contract_satisfied"
                )
                is True,
                "byte_closed_candidate_emitted": step_payload.get(
                    "byte_closed_candidate_emitted"
                )
                is True,
                "readiness_blockers": step_blockers,
                **FALSE_AUTHORITY,
            }
        )
        current_archive = step_archive

    assert current_archive is not None
    assert source_archive is not None
    blockers = ordered_unique(blockers)
    source_record = _archive_record(source_archive, repo_root=repo_root)
    final_record = _archive_record(current_archive, repo_root=repo_root)
    contract_fields = _archive_bound_contract_fields_for_chain(
        request_id=request_id,
        source_record=source_record,
        final_record=final_record,
        operations=operations,
        blockers=blockers,
        repo_root=repo_root,
    )
    manifest = {
        "schema": GROUPED_ARCHIVE_STATE_MATERIALIZER_CHAIN_SCHEMA,
        "request_id": request_id,
        "source_work_queue_path": _repo_rel(work_queue_path, repo_root),
        "source_packet_ir_operation_set_id": request.get(
            "source_packet_ir_operation_set_id"
        ),
        "ordered_work_ids": ordered_work_ids,
        "operation_count": len(operations),
        "source_archive": source_record,
        "final_candidate_archive": final_record,
        "candidate_archive": final_record,
        "candidate_archive_bytes": final_record["bytes"],
        "candidate_archive_sha256": final_record["sha256"],
        "source_archive_bytes": source_record["bytes"],
        "serialized_archive_delta": {
            "schema": "serialized_archive_delta_contract.v1",
            "source_archive_bytes": source_record["bytes"],
            "candidate_archive_bytes": final_record["bytes"],
            "saved_bytes": int(source_record["bytes"]) - int(final_record["bytes"]),
        },
        "operations": operations,
        "byte_closed_candidate_emitted": not blockers,
        "runtime_adapter_ready": not blockers,
        "receiver_proof_ready": not blockers,
        "receiver_contract_satisfied": not blockers,
        "candidate_runtime_adapter_blocker_cleared": not blockers,
        "readiness_blockers": blockers,
        **contract_fields,
        **FALSE_AUTHORITY,
    }
    write_json_artifact(output_manifest, manifest, allow_overwrite=allow_overwrite)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-queue", required=True, type=Path)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run_grouped_request(
            work_queue_path=args.work_queue,
            request_id=args.request_id,
            output_dir=args.output_dir,
            output_manifest=args.output_manifest,
            repo_root=args.repo_root,
            allow_overwrite=args.allow_overwrite,
        )
    except (OSError, ValueError, ArtifactWriteError) as exc:
        print(f"FATAL: grouped materializer failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True), end="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
