#!/usr/bin/env python3
"""Automatically close a detached Modal endpoint and retain every payload.

The closer is armed at dispatch time through ``tools/launch_detached_process.py``.
It composes the canonical Modal poller, lane-claim helper, call-id ledger,
NP1 ``NEXT_IF_RESUMED`` extractor, and commit serializer.  It does not contain
a second implementation of any of those mechanisms.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.deploy.claims import (  # noqa: E402
    DispatchClaimSpec,
    is_terminal_status,
    terminal_dispatch_claim,
)
from tac.deploy.modal.call_id_ledger import (  # noqa: E402
    TERMINAL_STATUSES,
    query_by_call_id,
    register_dispatched_call_id_fail_closed,
    update_call_id_outcome,
)
from tools.codex_arm_queue import (  # noqa: E402
    extract_next_if_resumed,
    next_if_resumed_blocks,
)
from tools.modal_harvest_poller import (  # noqa: E402
    POLL_DEADLINE,
    POLL_REMOTE_FAILURE,
    POLL_RESULT,
    poll_modal_call,
)

CLOSURE_MANIFEST_SCHEMA = "modal_endpoint_closure_manifest.v1"
MEMO_HANDOFF_SCHEMA = "modal_endpoint_memo_handoff.v1"
RECEIPT_SCHEMA = "modal_endpoint_closure_receipt.v1"
DONE_SCHEMA = "modal_endpoint_closure_done.v1"
RECEIPT_NAME = "ENDPOINT_CLOSURE.receipt.json"
DONE_NAME = "ENDPOINT_CLOSURE.done.json"
PAYLOAD_BLOCK_KEYS = frozenset({"retained_payloads", "payloads"})
SUCCESS_STATUS_WORDS = frozenset({"complete", "completed", "harvested", "ok", "success", "succeeded"})
FAILURE_STATUS_WORDS = frozenset({"cancelled", "error", "failed", "failure", "timeout", "timed_out"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VOLUME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_COMPLETE_RECEIPT_STATUSES = frozenset({"CLOSED", "CLOSED_REMOTE_FAILED"})


class EndpointClosureError(RuntimeError):
    """Fail-closed endpoint closure error."""


class PayloadCustodyError(EndpointClosureError):
    """A payload was not retained byte-identically."""

    def __init__(self, message: str, rows: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.rows = rows


@dataclasses.dataclass(frozen=True)
class TerminalDecision:
    terminal: bool
    remote_success: bool
    ledger_status: str | None
    claim_status: str | None
    rc: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f"{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    with staging.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json_bytes(value))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _coerce_rc(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value)
    return default


def derive_terminal_status(
    result: Any,
    *,
    poll_kind: str = POLL_RESULT,
) -> TerminalDecision:
    """Derive canonical ledger and claim terminality from the returned object."""

    if poll_kind == POLL_DEADLINE:
        return TerminalDecision(
            terminal=False,
            remote_success=False,
            ledger_status=None,
            claim_status=None,
            rc=124,
            reason="local_poll_deadline_is_not_provider_terminality",
        )
    if poll_kind == POLL_REMOTE_FAILURE:
        return TerminalDecision(
            terminal=True,
            remote_success=False,
            ledger_status="failed",
            claim_status="failed_endpoint_remote_exception",
            rc=1,
            reason="provider_reported_terminal_exception",
        )
    if not isinstance(result, dict):
        return TerminalDecision(
            terminal=True,
            remote_success=False,
            ledger_status="failed",
            claim_status="failed_endpoint_non_mapping_result",
            rc=1,
            reason="terminal_result_is_not_a_mapping",
        )

    raw_rc = result.get("returncode", result.get("rc"))
    rc = _coerce_rc(raw_rc, 0)
    status_word = str(result.get("status") or result.get("state") or "").strip().lower()
    error_value = result.get("error") or result.get("error_class") or result.get("exception")
    training_complete = result.get("training_complete")

    if raw_rc is not None and rc != 0:
        return TerminalDecision(True, False, "failed", "failed_endpoint_nonzero_rc", rc, "nonzero_returncode")
    if error_value:
        return TerminalDecision(True, False, "failed", "failed_endpoint_error_result", 1, "error_field_present")
    if training_complete is False:
        return TerminalDecision(
            True, False, "failed", "failed_endpoint_training_incomplete", 1, "training_complete_false"
        )
    if status_word in FAILURE_STATUS_WORDS:
        return TerminalDecision(True, False, "failed", f"failed_endpoint_{status_word}", 1, "failure_status_word")
    if training_complete is True:
        return TerminalDecision(True, True, "harvested", "completed_endpoint_harvested", 0, "training_complete_true")
    if raw_rc is not None and rc == 0:
        return TerminalDecision(True, True, "harvested", "completed_endpoint_harvested", 0, "zero_returncode")
    if status_word in SUCCESS_STATUS_WORDS:
        return TerminalDecision(True, True, "harvested", "completed_endpoint_harvested", 0, "success_status_word")
    if (
        result.get("schema") == "ddm_ec2_adapter_trainer_result.v1"
        and isinstance(result.get("endpoint"), dict)
        and isinstance(result.get("selected"), dict)
    ):
        return TerminalDecision(True, True, "harvested", "completed_endpoint_harvested", 0, "legacy_ec2_final_fixture")
    return TerminalDecision(
        True,
        False,
        "failed",
        "failed_endpoint_ambiguous_result",
        1,
        "terminal_result_lacks_success_evidence",
    )


def _safe_relative_remote_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise EndpointClosureError(f"unsafe Modal volume path: {value!r}")
    return path.as_posix()


def remote_path_from_record(path_value: str, *, run_id: str | None) -> str:
    """Convert a Modal mount/internal path into a volume-relative path."""

    path = PurePosixPath(path_value)
    if not path.is_absolute():
        return _safe_relative_remote_path(path.as_posix())
    parts = list(path.parts)
    if run_id and run_id in parts:
        return _safe_relative_remote_path(PurePosixPath(*parts[parts.index(run_id) :]).as_posix())
    raise EndpointClosureError(f"absolute payload path does not contain declared run_id={run_id!r}: {path_value}")


def payload_entries_from_result(
    value: Any,
    *,
    run_id: str | None,
    prefix: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Collect exact records from every ``retained_payloads``/``payloads`` block."""

    entries: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "closure_manifest":
                continue
            if key in PAYLOAD_BLOCK_KEYS and isinstance(child, dict):
                for name, record in child.items():
                    if not isinstance(record, dict):
                        continue
                    if not {"path", "sha256", "bytes"}.issubset(record):
                        continue
                    remote_path = remote_path_from_record(str(record["path"]), run_id=run_id)
                    entries.append(
                        {
                            "name": ".".join((*prefix, key, str(name))),
                            "remote_path": remote_path,
                            "bytes": int(record["bytes"]),
                            "sha256": str(record["sha256"]).lower(),
                            "source": ".".join((*prefix, key)),
                        }
                    )
                continue
            entries.extend(payload_entries_from_result(child, run_id=run_id, prefix=(*prefix, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            entries.extend(payload_entries_from_result(child, run_id=run_id, prefix=(*prefix, str(index))))
    return entries


def _validate_payload_entry(entry: dict[str, Any]) -> dict[str, Any]:
    required = {"name", "remote_path", "bytes", "sha256"}
    if not required.issubset(entry):
        raise EndpointClosureError(f"closure payload entry missing {sorted(required - set(entry))}: {entry!r}")
    digest = str(entry["sha256"]).lower()
    if not _SHA256_RE.fullmatch(digest):
        raise EndpointClosureError(f"invalid payload sha256 for {entry.get('name')}: {digest!r}")
    size = int(entry["bytes"])
    if size < 0:
        raise EndpointClosureError(f"negative payload size for {entry.get('name')}: {size}")
    return {
        **entry,
        "name": str(entry["name"]),
        "remote_path": _safe_relative_remote_path(str(entry["remote_path"])),
        "bytes": size,
        "sha256": digest,
    }


def validate_closure_manifest(
    manifest: Any,
    *,
    lane_id: str,
    instance_job_id: str,
) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("schema") != CLOSURE_MANIFEST_SCHEMA:
        raise EndpointClosureError(f"missing or invalid {CLOSURE_MANIFEST_SCHEMA} block")
    volume_name = str(manifest.get("volume_name") or "")
    if not _VOLUME_RE.fullmatch(volume_name):
        raise EndpointClosureError(f"invalid Modal volume name: {volume_name!r}")
    if manifest.get("lane_id") != lane_id:
        raise EndpointClosureError(f"closure manifest lane mismatch: {manifest.get('lane_id')!r} != {lane_id!r}")
    if manifest.get("instance_job_id") != instance_job_id:
        raise EndpointClosureError(
            f"closure manifest instance-job mismatch: {manifest.get('instance_job_id')!r} != {instance_job_id!r}"
        )
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise EndpointClosureError("closure manifest payloads must be a list")
    normalized = [_validate_payload_entry(dict(entry)) for entry in payloads if isinstance(entry, dict)]
    if len(normalized) != len(payloads):
        raise EndpointClosureError("closure manifest payload table contains a non-object row")
    remote_paths = [entry["remote_path"] for entry in normalized]
    if len(remote_paths) != len(set(remote_paths)):
        raise EndpointClosureError("closure manifest contains duplicate remote payload paths")
    return {**manifest, "volume_name": volume_name, "payloads": normalized}


def _payload_identity(entry: dict[str, Any]) -> tuple[str, int, str]:
    return (str(entry["sha256"]), int(entry["bytes"]), str(entry["remote_path"]))


def resolve_closure_manifest(
    result: dict[str, Any],
    *,
    lane_id: str,
    instance_job_id: str,
    supplemental_results: Iterable[dict[str, Any]] = (),
    allow_legacy: bool = False,
) -> dict[str, Any]:
    manifest = result.get("closure_manifest")
    run_id = str(result.get("run_id") or "") or None
    observed: list[dict[str, Any]] = []
    observed.extend(payload_entries_from_result(result, run_id=run_id))
    for supplemental in supplemental_results:
        observed.extend(payload_entries_from_result(supplemental, run_id=run_id))

    if manifest is None:
        if not allow_legacy:
            raise EndpointClosureError("terminal success result has no closure_manifest")
        volume_name = str(result.get("volume_name") or "")
        manifest = {
            "schema": CLOSURE_MANIFEST_SCHEMA,
            "volume_name": volume_name,
            "lane_id": lane_id,
            "instance_job_id": instance_job_id,
            "payloads": observed,
            "legacy_derived": True,
        }
    normalized = validate_closure_manifest(
        manifest,
        lane_id=lane_id,
        instance_job_id=instance_job_id,
    )
    declared = {_payload_identity(entry) for entry in normalized["payloads"]}
    missing = [entry for entry in observed if _payload_identity(entry) not in declared]
    if missing:
        raise EndpointClosureError(
            "closure manifest omits result payload entries: " + ", ".join(entry["name"] for entry in missing[:10])
        )
    return normalized


def _safe_artifact_name(value: str) -> str:
    if Path(value).name != value:
        raise EndpointClosureError(f"unsafe returned artifact name: {value!r}")
    safe = _SAFE_NAME_RE.sub("_", value).strip("._")
    if not safe:
        raise EndpointClosureError(f"empty returned artifact name after normalization: {value!r}")
    return safe


def artifact_payload_bytes(payload: Any) -> tuple[bytes, str]:
    """Coerce ANY returned artifact to bytes. Never returns without content.

    ALWAYS KEEP THE PAYLOAD applies to receipts. Before 2026-08-18 this module
    persisted only ``bytes`` artifacts and recorded every other type as the bare
    label ``{"embedded_value_type": "str"}`` — a measured TYPE with the content
    discarded, which is the canonical measure-and-discard signature. The rr4 CUDA
    row returned ``contest_auth_eval.json`` and ``report.txt`` as ``str`` and both
    were dropped (rv2 finding FO-2), forcing a recovery from the Modal result
    cache. Encoding is recorded so the persisted bytes stay interpretable.
    """

    if isinstance(payload, bytes):
        return payload, "bytes"
    if isinstance(payload, (bytearray, memoryview)):
        return bytes(payload), "bytes"
    if isinstance(payload, str):
        return payload.encode("utf-8"), "utf-8"
    try:
        return canonical_json_bytes(payload), "json"
    except (TypeError, ValueError):
        # Last resort: keep SOMETHING and say so. Raising here would discard the
        # whole harvest of a paid call over one unserializable entry.
        return repr(payload).encode("utf-8"), "repr"


def persist_remote_result(result: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize embedded artifacts, then persist a JSON-safe remote result."""

    safe_result = dict(result)
    artifact_records: dict[str, Any] = {}
    artifacts = safe_result.pop("artifacts", {})
    if artifacts:
        if not isinstance(artifacts, dict):
            raise EndpointClosureError("result artifacts must be a mapping")
        for raw_name, payload in artifacts.items():
            name = _safe_artifact_name(str(raw_name))
            # Distinct raw names can normalize to the SAME safe name ("a-b" and "a_b"),
            # and the later one would silently overwrite the earlier file — a payload
            # dropped by a different door than the one this landing closed. Disambiguate
            # rather than refuse: keeping both bytes beats failing a paid harvest.
            if name in artifact_records:
                stem, dot, suffix = name.partition(".")
                collision = 2
                while f"{stem}__{collision}{dot}{suffix}" in artifact_records:
                    collision += 1
                name = f"{stem}__{collision}{dot}{suffix}"
            blob, encoding = artifact_payload_bytes(payload)
            destination = output_dir / "returned_artifacts" / name
            atomic_bytes(destination, blob)
            record = file_record(destination)
            record["source_name"] = str(raw_name)
            record["source_value_type"] = type(payload).__name__
            record["persisted_encoding"] = encoding
            if encoding == "repr":
                record["lossy_repr"] = True
            artifact_records[name] = record
    safe_result["materialized_artifacts"] = artifact_records
    result_path = output_dir / "MODAL_REMOTE_RESULT.json"
    atomic_json(result_path, safe_result)
    return safe_result, file_record(result_path)


def _claim_summary(
    *,
    repo_root: Path,
    claims_path: Path,
    python_executable: Path,
) -> dict[str, Any]:
    command = [
        str(python_executable),
        str(repo_root / "tools/claim_lane_dispatch.py"),
        "summary",
        "--claims-path",
        str(claims_path),
        "--live-only",
        "--format",
        "json",
    ]
    process = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)
    if process.returncode:
        raise EndpointClosureError(f"canonical claim summary failed rc={process.returncode}: {process.stderr[-500:]}")
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise EndpointClosureError("canonical claim summary did not emit JSON") from exc


def _latest_claim(
    summary: dict[str, Any],
    *,
    lane_id: str,
    instance_job_id: str,
) -> dict[str, Any] | None:
    for section in ("active", "stale_nonterminal", "terminal_latest"):
        for row in summary.get(section, []):
            if row.get("lane_id") == lane_id and row.get("instance_job_id") == instance_job_id:
                return row
    return None


def _validated_spawn_registration(
    metadata: dict[str, Any],
    *,
    call_id: str,
    lane_id: str,
    instance_job_id: str,
    agent: str,
) -> dict[str, Any]:
    """Authenticate legacy recovery from the exact dispatcher spawn receipt."""

    if metadata.get("schema_version") != "modal_auth_eval_spawn_v1":
        raise EndpointClosureError("spawn metadata recovery schema differs")
    expected = {
        "call_id": call_id,
        "lane_id": lane_id,
        "instance_job_id": instance_job_id,
        "claim_agent": agent,
        "claim_platform": "modal",
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise EndpointClosureError(
            "spawn metadata does not authenticate ledger recovery: "
            + json.dumps(mismatches, sort_keys=True)
        )
    tool = str(metadata.get("tool") or "")
    if tool not in {
        "experiments/modal_auth_eval.py",
        "experiments/modal_auth_eval_cpu.py",
    }:
        raise EndpointClosureError(f"unsupported spawn metadata recovery tool: {tool!r}")
    axis = str(metadata.get("axis") or "")
    if not axis:
        raise EndpointClosureError("spawn metadata recovery has no axis")
    expected_app = (
        "comma-auth-eval-cpu"
        if tool == "experiments/modal_auth_eval_cpu.py"
        else "comma-auth-eval"
    )
    if metadata.get("app") != expected_app:
        raise EndpointClosureError(
            f"spawn metadata recovery app/tool mismatch: {metadata.get('app')!r} != {expected_app!r}"
        )
    if tool.endswith("_cpu.py") and axis != "contest_cpu":
        raise EndpointClosureError(
            f"CPU spawn metadata recovery axis differs: {axis!r}"
        )
    local_request = metadata.get("local_request")
    if not isinstance(local_request, dict):
        raise EndpointClosureError("spawn metadata recovery has no local request")
    archive_sha256 = str(local_request.get("archive_sha256") or "")
    if archive_sha256 and not _SHA256_RE.fullmatch(archive_sha256):
        raise EndpointClosureError("spawn metadata recovery has invalid archive SHA-256")
    return {
        "call_id": call_id,
        "lane_id": lane_id,
        "label": "modal_auth_eval_cpu" if tool.endswith("_cpu.py") else "modal_auth_eval",
        "platform": "modal",
        "gpu": "CPU" if axis == "contest_cpu" else str(metadata.get("gpu") or "T4"),
        "expected_axis": axis,
        "recipe": tool,
        "dispatched_at_utc": metadata.get("dispatched_at_utc"),
        "mounted_code_git_head": local_request.get("source_repo_commit"),
        "agent": agent,
        "base_archive_sha256": archive_sha256 or None,
        "composed_archive_sha256": archive_sha256 or None,
        "archive_count": 1,
        "pair_group_id": local_request.get("pair_group_id"),
        "legacy_registration_recovery": RECEIPT_SCHEMA,
    }


def ensure_dual_ledgers_terminal(
    *,
    call_id: str,
    lane_id: str,
    instance_job_id: str,
    agent: str,
    decision: TerminalDecision,
    result_record: dict[str, Any],
    repo_root: Path = REPO_ROOT,
    claims_path: Path | None = None,
    call_ledger_path: Path | None = None,
    spawn_metadata: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Close claim first, then call-id ledger, using both canonical writers."""

    if not decision.terminal or not decision.ledger_status or not decision.claim_status:
        raise EndpointClosureError("refusing terminal ledger writes for a nonterminal poll outcome")
    claims_path = claims_path or repo_root / ".omx/state/active_lane_dispatch_claims.md"
    call_ledger_path = call_ledger_path or repo_root / ".omx/state/modal_call_id_ledger.jsonl"
    python_executable = repo_root / ".venv/bin/python"

    try:
        summary_before = _claim_summary(
            repo_root=repo_root,
            claims_path=claims_path,
            python_executable=python_executable,
        )
        claim_before = _latest_claim(summary_before, lane_id=lane_id, instance_job_id=instance_job_id)
        call_rows_before = query_by_call_id(call_id, path=call_ledger_path)
    except (Exception, SystemExit) as exc:
        raise EndpointClosureError(f"dual-ledger read failed: {exc}") from exc
    if claim_before is None:
        raise EndpointClosureError(f"no canonical claim row for lane={lane_id} instance-job={instance_job_id}")
    registration_action = "not_needed"
    if not call_rows_before:
        if spawn_metadata is None:
            raise EndpointClosureError(f"no canonical call-id ledger row for {call_id}")
        registration = _validated_spawn_registration(
            spawn_metadata,
            call_id=call_id,
            lane_id=lane_id,
            instance_job_id=instance_job_id,
            agent=agent,
        )
        if dry_run:
            call_rows_before = [{**registration, "status": "dispatched"}]
            registration_action = "would_recover_from_spawn_metadata"
        else:
            try:
                register_dispatched_call_id_fail_closed(
                    **registration,
                    path=call_ledger_path,
                    lock_path=call_ledger_path.with_suffix(call_ledger_path.suffix + ".lock"),
                )
                call_rows_before = query_by_call_id(call_id, path=call_ledger_path)
            except (Exception, SystemExit) as exc:
                raise EndpointClosureError(
                    f"canonical legacy call-id registration recovery failed: {exc}"
                ) from exc
            if not call_rows_before:
                raise EndpointClosureError(
                    f"legacy call-id registration recovery wrote no canonical row for {call_id}"
                )
            registration_action = "recovered_from_spawn_metadata"
    claim_was_terminal = is_terminal_status(str(claim_before.get("status") or ""))
    call_was_terminal = str(call_rows_before[-1].get("status")) in TERMINAL_STATUSES

    if not dry_run and not claim_was_terminal:
        try:
            terminal_dispatch_claim(
                repo_root=repo_root,
                spec=DispatchClaimSpec(
                    lane_id=lane_id,
                    instance_job_id=instance_job_id,
                    agent=agent,
                    platform="modal",
                    force=True,
                ),
                status=decision.claim_status,
                notes=f"automatic endpoint closure; call_id={call_id}; reason={decision.reason}",
                python_executable=str(python_executable),
                claim_tool=repo_root / "tools/claim_lane_dispatch.py",
                claims_path=claims_path,
            )
        except (Exception, SystemExit) as exc:
            raise EndpointClosureError(f"canonical claim closure failed: {exc}") from exc

    if not dry_run and not call_was_terminal:
        try:
            update_call_id_outcome(
                call_id=call_id,
                status=decision.ledger_status,
                rc=decision.rc,
                agent=agent,
                lane_id=lane_id,
                harvest_result={
                    "endpoint_closer": RECEIPT_SCHEMA,
                    "terminal_reason": decision.reason,
                    "result": result_record,
                },
                path=call_ledger_path,
                lock_path=call_ledger_path.with_suffix(call_ledger_path.suffix + ".lock"),
            )
        except (Exception, SystemExit) as exc:
            raise EndpointClosureError(f"canonical call-id closure failed: {exc}") from exc

    if dry_run:
        return {
            "claim": {"before": claim_before, "action": "no-op" if claim_was_terminal else "would_close"},
            "call_id": {
                "before": call_rows_before[-1],
                "registration_action": registration_action,
                "action": "no-op" if call_was_terminal else "would_close",
            },
            "both_terminal": claim_was_terminal and call_was_terminal,
            "dry_run": True,
        }

    try:
        summary_after = _claim_summary(
            repo_root=repo_root,
            claims_path=claims_path,
            python_executable=python_executable,
        )
        claim_after = _latest_claim(summary_after, lane_id=lane_id, instance_job_id=instance_job_id)
        call_rows_after = query_by_call_id(call_id, path=call_ledger_path)
    except (Exception, SystemExit) as exc:
        raise EndpointClosureError(f"dual-ledger verification failed: {exc}") from exc
    both_terminal = bool(
        claim_after
        and is_terminal_status(str(claim_after.get("status") or ""))
        and call_rows_after
        and str(call_rows_after[-1].get("status")) in TERMINAL_STATUSES
    )
    if not both_terminal:
        raise EndpointClosureError("dual-ledger terminal invariant failed after canonical writes")
    return {
        "claim": {
            "before": claim_before,
            "after": claim_after,
            "action": "no-op" if claim_was_terminal else "closed",
        },
        "call_id": {
            "before": call_rows_before[-1],
            "after": call_rows_after[-1],
            "registration_action": registration_action,
            "action": "no-op" if call_was_terminal else "closed",
        },
        "both_terminal": True,
        "dry_run": False,
    }


def _destination_for(local_store: Path, remote_path: str) -> Path:
    root = local_store.resolve(strict=False)
    destination = (root / PurePosixPath(remote_path)).resolve(strict=False)
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise EndpointClosureError(f"payload destination escapes local store: {remote_path}") from exc
    return destination


def _verify_exact(path: Path, entry: dict[str, Any]) -> tuple[bool, int, str]:
    if not path.is_file():
        return False, -1, ""
    size = path.stat().st_size
    digest = sha256_file(path)
    return size == entry["bytes"] and digest == entry["sha256"], size, digest


def _find_fixture(entry: dict[str, Any], roots: Iterable[Path]) -> Path | None:
    basename = PurePosixPath(entry["remote_path"]).name
    for root in roots:
        for candidate in (root / entry["remote_path"], root / basename):
            ok, _, _ = _verify_exact(candidate, entry)
            if ok:
                return candidate
        if root.is_dir():
            for candidate in root.rglob(basename):
                ok, _, _ = _verify_exact(candidate, entry)
                if ok:
                    return candidate
    return None


def _default_volume_get(
    *,
    modal_executable: Path,
    volume_name: str,
    remote_path: str,
    destination: Path,
) -> None:
    command = [
        str(modal_executable),
        "volume",
        "get",
        "--force",
        volume_name,
        remote_path,
        str(destination),
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode:
        raise EndpointClosureError(
            f"modal volume get failed rc={process.returncode} for {remote_path}: {process.stderr[-500:]}"
        )


def harvest_payloads(
    *,
    manifest: dict[str, Any],
    local_store: Path,
    modal_executable: Path,
    dry_run: bool = False,
    fixture_roots: Iterable[Path] = (),
    volume_get: Callable[..., None] | None = None,
    reserve_bytes: int = 256 * 1024 * 1024,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pull and verify every declared payload without deleting mismatches."""

    rows: list[dict[str, Any]] = []
    entries = list(manifest["payloads"])
    missing_bytes = sum(
        entry["bytes"]
        for entry in entries
        if not _verify_exact(_destination_for(local_store, entry["remote_path"]), entry)[0]
    )
    probe_root = local_store
    while not probe_root.exists() and probe_root != probe_root.parent:
        probe_root = probe_root.parent
    usage = shutil.disk_usage(probe_root)
    storage = {
        "path": str(local_store.resolve(strict=False)),
        "free_bytes": usage.free,
        "missing_payload_bytes": missing_bytes,
        "reserve_bytes": reserve_bytes,
        "required_bytes": missing_bytes + reserve_bytes,
        "passed": dry_run or usage.free >= missing_bytes + reserve_bytes,
        "dry_run": dry_run,
    }
    if not storage["passed"]:
        raise PayloadCustodyError(
            f"storage preflight failed: free={usage.free} required={missing_bytes + reserve_bytes}",
            rows,
        )

    if dry_run:
        fixture_roots = tuple(fixture_roots)
        for entry in entries:
            fixture = _find_fixture(entry, fixture_roots)
            rows.append(
                {
                    **entry,
                    "local_path": str(fixture.resolve()) if fixture else None,
                    "status": "fixture_verified" if fixture else "would_pull",
                    "verified": bool(fixture),
                }
            )
        return rows, storage

    local_store.mkdir(parents=True, exist_ok=True)
    getter = volume_get or _default_volume_get
    for entry in entries:
        destination = _destination_for(local_store, entry["remote_path"])
        ok, actual_bytes, actual_sha = _verify_exact(destination, entry)
        if ok:
            rows.append(
                {**entry, "local_path": str(destination), "status": "already_present_verified", "verified": True}
            )
            continue
        if destination.exists():
            rows.append(
                {
                    **entry,
                    "local_path": str(destination),
                    "status": "existing_destination_mismatch_preserved",
                    "verified": False,
                    "actual_bytes": actual_bytes,
                    "actual_sha256": actual_sha,
                }
            )
            raise PayloadCustodyError(
                f"existing destination differs for {entry['remote_path']}; preserved in place",
                rows,
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f"{destination.name}.download.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            getter(
                modal_executable=modal_executable,
                volume_name=manifest["volume_name"],
                remote_path=entry["remote_path"],
                destination=staging,
            )
        except Exception as exc:
            rows.append(
                {
                    **entry,
                    "local_path": str(staging),
                    "status": "download_failed_staging_preserved",
                    "verified": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise PayloadCustodyError(
                f"payload download failed for {entry['remote_path']}; remote retained",
                rows,
            ) from exc
        ok, actual_bytes, actual_sha = _verify_exact(staging, entry)
        if not ok:
            rows.append(
                {
                    **entry,
                    "local_path": str(staging),
                    "status": "download_mismatch_preserved",
                    "verified": False,
                    "actual_bytes": actual_bytes,
                    "actual_sha256": actual_sha,
                }
            )
            raise PayloadCustodyError(
                f"downloaded payload differs for {entry['remote_path']}; remote and local mismatch preserved",
                rows,
            )
        os.replace(staging, destination)
        rows.append({**entry, "local_path": str(destination), "status": "downloaded_verified", "verified": True})
    return rows, storage


def memo_handoff_from_final_messages(paths: Iterable[Path]) -> dict[str, Any] | None:
    handoffs: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _JSON_FENCE_RE.finditer(text):
            try:
                value = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("schema") == MEMO_HANDOFF_SCHEMA:
                handoffs.append(value)
    if not handoffs:
        return None
    canonical = json.dumps(handoffs[0], sort_keys=True)
    if any(json.dumps(value, sort_keys=True) != canonical for value in handoffs[1:]):
        raise EndpointClosureError("conflicting memo handoffs declared by arm final messages")
    return handoffs[0]


def commit_memo_handoff(
    handoff: dict[str, Any] | None,
    *,
    repo_root: Path,
    dry_run: bool,
    serializer_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if handoff is None:
        return {"status": "none"}
    if handoff.get("schema") != MEMO_HANDOFF_SCHEMA:
        raise EndpointClosureError(f"invalid memo handoff schema: {handoff.get('schema')!r}")
    raw_path = Path(str(handoff.get("path") or ""))
    path = raw_path if raw_path.is_absolute() else repo_root / raw_path
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise EndpointClosureError(f"memo handoff escapes repository: {path}") from exc
    digest = str(handoff.get("sha256") or "").lower()
    message = str(handoff.get("message") or "")
    if not resolved.is_file() or not _SHA256_RE.fullmatch(digest) or sha256_file(resolved) != digest:
        raise EndpointClosureError(f"memo handoff sha gate failed: {resolved}")
    if "[no-triality]" not in message or "[p0-ledger-ok]" not in message:
        raise EndpointClosureError("memo handoff commit message lacks required serializer tags")

    head = subprocess.run(
        ["git", "show", f"HEAD:{relative.as_posix()}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if head.returncode == 0 and sha256_bytes(head.stdout) == digest:
        return {"status": "already_committed", "path": relative.as_posix(), "sha256": digest}
    if dry_run:
        return {"status": "verified_would_commit", "path": relative.as_posix(), "sha256": digest, "message": message}

    command = [
        str(repo_root / ".venv/bin/python"),
        str(repo_root / "tools/subagent_commit_serializer.py"),
        "--message",
        message,
        "--files",
        relative.as_posix(),
        "--expected-content-sha256",
        f"{relative.as_posix()}={digest}",
        "--no-co-author",
        "--label",
        "modal_endpoint_close",
    ]
    tracked = (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
            cwd=repo_root,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    base_sha = handoff.get("base_sha256")
    if not tracked:
        command.extend(["--base-content-sha256", f"{relative.as_posix()}=new"])
    elif base_sha:
        command.extend(["--base-content-sha256", f"{relative.as_posix()}={base_sha}"])
    env = os.environ.copy()
    if resolved.suffix in {".md", ".sh"}:
        env["REVIEW_GATE_OVERRIDE"] = "1"
    process = serializer_runner(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise EndpointClosureError(f"memo serializer failed rc={process.returncode}: {(process.stderr or '')[-800:]}")
    return {
        "status": "committed",
        "path": relative.as_posix(),
        "sha256": digest,
        "message": message,
        "serializer_stdout": (process.stdout or "")[-800:],
    }


def extract_next_surface(
    *,
    final_message_paths: Iterable[Path],
    inline_final_message: str | None,
    dry_run: bool,
    next_store: Path,
    name: str,
) -> dict[str, Any]:
    paths = [path for path in final_message_paths if path.is_file()]
    blocks: list[dict[str, Any]] = []
    for path in paths:
        blocks.extend(
            {**block, "source_path": str(path)} for block in next_if_resumed_blocks(path.read_text(errors="replace"))
        )
    if inline_final_message:
        blocks.extend(
            {**block, "source_path": "inline_final_message"} for block in next_if_resumed_blocks(inline_final_message)
        )
    summary = None
    if paths and not dry_run:
        summary = extract_next_if_resumed(
            paths,
            provenance="endpoint-closure",
            name=name,
            out_path=next_store,
        )
    return {"blocks": blocks, "extract_summary": summary, "dry_run": dry_run}


def validate_receipt(receipt: Any) -> dict[str, Any]:
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise EndpointClosureError("invalid endpoint closure receipt schema")
    required = {
        "call_id",
        "status",
        "process_rc",
        "terminal_decision",
        "ledgers",
        "payloads",
        "memo_commit",
        "next_if_resumed",
    }
    if not required.issubset(receipt):
        raise EndpointClosureError(f"endpoint closure receipt missing {sorted(required - set(receipt))}")
    if not isinstance(receipt["payloads"], list):
        raise EndpointClosureError("endpoint closure receipt payloads must be a list")
    if not isinstance(receipt["process_rc"], int):
        raise EndpointClosureError("endpoint closure receipt process_rc must be int")
    return receipt


def _write_receipt_and_done(output_dir: Path, receipt: dict[str, Any]) -> None:
    validate_receipt(receipt)
    receipt_path = output_dir / RECEIPT_NAME
    atomic_json(receipt_path, receipt)
    receipt_record = file_record(receipt_path)
    atomic_json(
        output_dir / DONE_NAME,
        {
            "schema": DONE_SCHEMA,
            "call_id": receipt["call_id"],
            "status": receipt["status"],
            "rc": receipt["process_rc"],
            "receipt": receipt_record,
            "written_at_utc": utc_now(),
        },
    )


def execute_endpoint_closure(
    *,
    call_id: str,
    result: dict[str, Any],
    poll_kind: str,
    output_dir: Path,
    local_store: Path,
    lane_id: str,
    instance_job_id: str,
    agent: str,
    supplemental_results: Iterable[dict[str, Any]] = (),
    final_message_paths: Iterable[Path] = (),
    allow_legacy_manifest: bool = False,
    dry_run: bool = False,
    fixture_roots: Iterable[Path] = (),
    repo_root: Path = REPO_ROOT,
    claims_path: Path | None = None,
    call_ledger_path: Path | None = None,
    spawn_metadata: dict[str, Any] | None = None,
    next_store: Path | None = None,
    modal_executable: Path | None = None,
    volume_get: Callable[..., None] | None = None,
    serializer_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / RECEIPT_NAME
    if receipt_path.is_file():
        prior = validate_receipt(json.loads(receipt_path.read_text()))
        if prior.get("call_id") == call_id and prior.get("status") in _COMPLETE_RECEIPT_STATUSES:
            return prior

    safe_result, result_record = persist_remote_result(result, output_dir)
    decision = derive_terminal_status(result, poll_kind=poll_kind)
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "written_at_utc": utc_now(),
        "call_id": call_id,
        "lane_id": lane_id,
        "instance_job_id": instance_job_id,
        "dry_run": dry_run,
        "status": "REFUSED_NONTERMINAL",
        "process_rc": decision.rc,
        "terminal_decision": decision.as_dict(),
        "result": result_record,
        "closure_manifest": None,
        "ledgers": {"both_terminal": False, "action": "not_attempted"},
        "payloads": [],
        "storage_preflight": None,
        "memo_commit": {"status": "not_attempted"},
        "next_if_resumed": {"blocks": [], "extract_summary": None, "dry_run": dry_run},
        "errors": [],
    }
    if not decision.terminal:
        receipt["errors"].append(decision.reason)
        _write_receipt_and_done(output_dir, receipt)
        return receipt

    try:
        receipt["ledgers"] = ensure_dual_ledgers_terminal(
            call_id=call_id,
            lane_id=lane_id,
            instance_job_id=instance_job_id,
            agent=agent,
            decision=decision,
            result_record=result_record,
            repo_root=repo_root,
            claims_path=claims_path,
            call_ledger_path=call_ledger_path,
            spawn_metadata=spawn_metadata,
            dry_run=dry_run,
        )
    except EndpointClosureError as exc:
        receipt["status"] = "REFUSED_DUAL_LEDGER"
        receipt["process_rc"] = 1
        receipt["errors"].append(str(exc))
        _write_receipt_and_done(output_dir, receipt)
        return receipt

    final_message_paths = tuple(final_message_paths)
    if decision.remote_success:
        try:
            manifest = resolve_closure_manifest(
                safe_result,
                lane_id=lane_id,
                instance_job_id=instance_job_id,
                supplemental_results=supplemental_results,
                allow_legacy=allow_legacy_manifest,
            )
            receipt["closure_manifest"] = manifest
            payload_rows, storage = harvest_payloads(
                manifest=manifest,
                local_store=local_store,
                modal_executable=modal_executable or repo_root / ".venv/bin/modal",
                dry_run=dry_run,
                fixture_roots=fixture_roots,
                volume_get=volume_get,
            )
            receipt["payloads"] = payload_rows
            receipt["storage_preflight"] = storage
        except PayloadCustodyError as exc:
            receipt["payloads"] = exc.rows
            receipt["status"] = "REFUSED_PAYLOAD_CUSTODY"
            receipt["process_rc"] = 1
            receipt["errors"].append(str(exc))
            _write_receipt_and_done(output_dir, receipt)
            return receipt
        except EndpointClosureError as exc:
            receipt["status"] = "REFUSED_MANIFEST"
            receipt["process_rc"] = 1
            receipt["errors"].append(str(exc))
            _write_receipt_and_done(output_dir, receipt)
            return receipt

    try:
        manifest_handoff = (receipt.get("closure_manifest") or {}).get("memo_handoff")
        result_handoff = safe_result.get("git_blocked_memo")
        final_handoff = memo_handoff_from_final_messages(final_message_paths)
        handoffs = [value for value in (manifest_handoff, result_handoff, final_handoff) if value]
        if handoffs and any(
            json.dumps(value, sort_keys=True) != json.dumps(handoffs[0], sort_keys=True) for value in handoffs[1:]
        ):
            raise EndpointClosureError("conflicting result/final-message memo handoffs")
        receipt["memo_commit"] = commit_memo_handoff(
            handoffs[0] if handoffs else None,
            repo_root=repo_root,
            dry_run=dry_run,
            serializer_runner=serializer_runner,
        )
        receipt["next_if_resumed"] = extract_next_surface(
            final_message_paths=final_message_paths,
            inline_final_message=safe_result.get("final_message")
            if isinstance(safe_result.get("final_message"), str)
            else None,
            dry_run=dry_run,
            next_store=next_store or repo_root / ".omx/state/codex_arm_queue.next_if_resumed.jsonl",
            name=lane_id,
        )
    except EndpointClosureError as exc:
        receipt["status"] = "REFUSED_MEMO_OR_NEXT_SURFACE"
        receipt["process_rc"] = 1
        receipt["errors"].append(str(exc))
        _write_receipt_and_done(output_dir, receipt)
        return receipt

    if dry_run:
        receipt["status"] = "DRY_RUN_VALIDATED"
        receipt["process_rc"] = 0
    elif decision.remote_success:
        receipt["status"] = "CLOSED"
        receipt["process_rc"] = 0
    else:
        receipt["status"] = "CLOSED_REMOTE_FAILED"
        receipt["process_rc"] = max(decision.rc, 1)
    _write_receipt_and_done(output_dir, receipt)
    return receipt


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise EndpointClosureError(f"JSON root must be an object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--local-store", required=True, type=Path)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--instance-job-id", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--deadline-s", type=float, default=3 * 3600)
    parser.add_argument("--poll-s", type=float, default=60)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--closure-manifest-file", type=Path)
    parser.add_argument("--spawn-metadata", type=Path)
    parser.add_argument("--payload-manifest-source", action="append", default=[], type=Path)
    parser.add_argument("--arm-final-message", action="append", default=[], type=Path)
    parser.add_argument("--allow-legacy-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixture-root", action="append", default=[], type=Path)
    parser.add_argument("--claims-path", type=Path, default=REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md")
    parser.add_argument("--call-ledger-path", type=Path, default=REPO_ROOT / ".omx/state/modal_call_id_ledger.jsonl")
    parser.add_argument(
        "--next-store", type=Path, default=REPO_ROOT / ".omx/state/codex_arm_queue.next_if_resumed.jsonl"
    )
    parser.add_argument("--modal-executable", type=Path, default=REPO_ROOT / ".venv/bin/modal")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.result_file:
        result = _load_json(args.result_file)
        poll_kind = POLL_RESULT
    else:
        poll = poll_modal_call(
            call_id=args.call_id,
            deadline_s=args.deadline_s,
            poll_s=args.poll_s,
        )
        poll_kind = str(poll["kind"])
        result = poll.get("result")
        if not isinstance(result, dict):
            result = {
                "schema": "modal_endpoint_poll_failure.v1",
                "error_class": poll.get("error_class"),
                "error": poll.get("error"),
            }
    if args.closure_manifest_file:
        supplied_manifest = _load_json(args.closure_manifest_file)
        if result.get("closure_manifest") not in (None, supplied_manifest):
            raise EndpointClosureError(
                "result closure manifest conflicts with --closure-manifest-file"
            )
        result = {**result, "closure_manifest": supplied_manifest}
    supplemental = [_load_json(path) for path in args.payload_manifest_source]
    spawn_metadata = _load_json(args.spawn_metadata) if args.spawn_metadata else None
    receipt = execute_endpoint_closure(
        call_id=args.call_id,
        result=result,
        poll_kind=poll_kind,
        output_dir=args.output_dir,
        local_store=args.local_store,
        lane_id=args.lane_id,
        instance_job_id=args.instance_job_id,
        agent=args.agent,
        supplemental_results=supplemental,
        final_message_paths=args.arm_final_message,
        allow_legacy_manifest=args.allow_legacy_manifest,
        dry_run=args.dry_run,
        fixture_roots=args.fixture_root,
        claims_path=args.claims_path,
        call_ledger_path=args.call_ledger_path,
        spawn_metadata=spawn_metadata,
        next_store=args.next_store,
        modal_executable=args.modal_executable,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return int(receipt["process_rc"])


if __name__ == "__main__":
    raise SystemExit(main())
