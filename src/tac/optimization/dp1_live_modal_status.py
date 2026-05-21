# SPDX-License-Identifier: MIT
"""Parse DP1 live Modal training logs into a queryable status packet."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

SCHEMA = "dp1_live_modal_status_v1"
MODAL_CALL_STATUS_SCHEMA = "dp1_live_modal_call_status_v1"
TOOL_PATH = "tools/inspect_dp1_live_modal_status.py"

_STAGE_RE = re.compile(
    r"^\[archive-only-eval\]\s+(?P<ts>\S+)\s+Stage\s+(?P<stage>\d+[^:]*):\s+(?P<text>.*)$"
)
_CLAIM_RE = re.compile(
    r"^\[lane-dpp\]\s+(?P<ts>\S+)\s+stage_0b_dispatch_claim_verified\s+lane=(?P<lane>\S+)\s+job=(?P<job>\S+)"
)
_ROUNDTRIP_RE = re.compile(
    r"archive pack/parse roundtrip:\s+(?P<bytes>\d+)\s+bytes;\s+pairs=(?P<pairs>\d+);\s+header=(?P<header>\d+)"
)
_PROCEDURAL_RE = re.compile(
    r"procedural codebook replacement:\s+(?P<before>\d+)\s+B\s+->\s+(?P<after>\d+)\s+B\s+"
    r"\(saved\s+(?P<saved>\d+)\s+B;\s+predicted\s+[^=]+=?(?P<delta>[-+0-9.eE]+)\)"
)
_WROTE_RE = re.compile(r"wrote\s+(?P<kind>[A-Za-z0-9_.-]+):\s+(?P<path>\S+)")
_FINISHED_RE = re.compile(r"finished in\s+(?P<seconds>[0-9.]+)s\s+rc=(?P<rc>-?\d+)")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _parse_utc_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_call_id_budget(
    *,
    call_id: str,
    ledger_path: Path,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Return dispatch age/max_seconds budget from the Modal call-id ledger."""

    if not call_id or not ledger_path.is_file():
        return {
            "schema": "dp1_modal_call_budget_v1",
            "call_id": call_id,
            "ledger_path": str(ledger_path),
            "found": False,
        }
    rows: list[Mapping[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and payload.get("call_id") == call_id:
            rows.append(payload)
    dispatch_rows = [row for row in rows if row.get("event_type") == "dispatched"]
    if not dispatch_rows:
        return {
            "schema": "dp1_modal_call_budget_v1",
            "call_id": call_id,
            "ledger_path": str(ledger_path),
            "found": False,
        }
    row = dispatch_rows[-1]
    registered_at = _parse_utc_datetime(row.get("written_at_utc")) or _parse_utc_datetime(
        row.get("dispatched_at_utc")
    )
    max_seconds = row.get("max_seconds")
    now = (now_utc or datetime.now(UTC)).astimezone(UTC)
    elapsed: float | None = None
    seconds_until_max: float | None = None
    max_seconds_exceeded = False
    if registered_at is not None:
        elapsed = max(0.0, (now - registered_at).total_seconds())
        if isinstance(max_seconds, int | float) and not isinstance(max_seconds, bool):
            seconds_until_max = float(max_seconds) - elapsed
            max_seconds_exceeded = seconds_until_max < 0
    return {
        "schema": "dp1_modal_call_budget_v1",
        "call_id": call_id,
        "ledger_path": str(ledger_path),
        "found": True,
        "status": row.get("status"),
        "event_type": row.get("event_type"),
        "registered_at_utc": registered_at.isoformat() if registered_at else None,
        "checked_at_utc": now.isoformat(),
        "max_seconds": max_seconds,
        "elapsed_since_registration_seconds": elapsed,
        "seconds_until_max": seconds_until_max,
        "max_seconds_exceeded": max_seconds_exceeded,
    }


def parse_dp1_live_modal_log(
    path: str | Path,
    *,
    variant: str,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Return structured DP1 progress from one synced Modal lane log."""

    root = Path(repo_root).resolve()
    log_path = Path(path).resolve()
    text = log_path.read_text(encoding="utf-8", errors="replace")
    stage_events: list[dict[str, Any]] = []
    written_artifacts: dict[str, str] = {}
    procedural_replacement: dict[str, Any] | None = None
    smoke_roundtrip: dict[str, Any] | None = None
    dispatch_claim: dict[str, Any] | None = None
    finished: dict[str, Any] | None = None
    deterministic_warnings = 0

    for line in text.splitlines():
        if "does not have a deterministic implementation" in line:
            deterministic_warnings += 1
        if match := _CLAIM_RE.search(line):
            dispatch_claim = {
                "timestamp_utc": match.group("ts"),
                "lane_id": match.group("lane"),
                "job_id": match.group("job"),
            }
            continue
        if match := _STAGE_RE.search(line):
            stage_events.append(
                {
                    "timestamp_utc": match.group("ts"),
                    "stage": match.group("stage"),
                    "text": match.group("text"),
                }
            )
            continue
        if match := _ROUNDTRIP_RE.search(line):
            smoke_roundtrip = {
                "archive_bytes": int(match.group("bytes")),
                "pairs": int(match.group("pairs")),
                "header_bytes": int(match.group("header")),
            }
            continue
        if match := _PROCEDURAL_RE.search(line):
            procedural_replacement = {
                "before_bytes": int(match.group("before")),
                "after_bytes": int(match.group("after")),
                "saved_bytes": int(match.group("saved")),
                "predicted_delta_s": float(match.group("delta")),
            }
            continue
        if match := _WROTE_RE.search(line):
            written_artifacts[match.group("kind")] = match.group("path")
            continue
        if match := _FINISHED_RE.search(line):
            finished = {
                "elapsed_seconds": float(match.group("seconds")),
                "returncode": int(match.group("rc")),
            }

    last_stage = stage_events[-1] if stage_events else None
    return {
        "schema": "dp1_live_modal_variant_status_v1",
        "variant": variant,
        "log_path": _repo_rel(log_path, root),
        "log_bytes": log_path.stat().st_size,
        "log_sha256": _sha256_file(log_path),
        "dispatch_claim": dispatch_claim,
        "stage_events": stage_events,
        "last_stage": last_stage,
        "stage4_full_training_started": any(
            str(event.get("stage", "")).startswith("4") for event in stage_events
        ),
        "finished": finished is not None,
        "finish": finished,
        "smoke_roundtrip": smoke_roundtrip,
        "procedural_codebook_replacement": procedural_replacement,
        "written_artifacts": written_artifacts,
        "deterministic_warning_count": deterministic_warnings,
    }


def build_dp1_live_modal_status(
    *,
    baseline_log: str | Path,
    procedural_log: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Return paired baseline/procedural live status from synced logs."""

    baseline = parse_dp1_live_modal_log(
        baseline_log,
        variant="baseline",
        repo_root=repo_root,
    )
    procedural = parse_dp1_live_modal_log(
        procedural_log,
        variant="procedural",
        repo_root=repo_root,
    )
    blockers: list[str] = []
    if not baseline["stage4_full_training_started"]:
        blockers.append("baseline_stage4_full_training_not_started")
    if not procedural["stage4_full_training_started"]:
        blockers.append("procedural_stage4_full_training_not_started")
    if baseline["finished"] and baseline.get("finish", {}).get("returncode") != 0:
        blockers.append("baseline_finished_nonzero")
    if procedural["finished"] and procedural.get("finish", {}).get("returncode") != 0:
        blockers.append("procedural_finished_nonzero")

    baseline_bytes = None
    procedural_bytes = None
    if isinstance(baseline.get("smoke_roundtrip"), Mapping):
        baseline_bytes = baseline["smoke_roundtrip"].get("archive_bytes")
    if isinstance(procedural.get("smoke_roundtrip"), Mapping):
        procedural_bytes = procedural["smoke_roundtrip"].get("archive_bytes")
    smoke_delta_bytes = (
        int(procedural_bytes) - int(baseline_bytes)
        if isinstance(baseline_bytes, int) and isinstance(procedural_bytes, int)
        else None
    )
    both_finished = bool(baseline["finished"] and procedural["finished"])
    aggregate_status = "finished" if both_finished and not blockers else "running"

    return {
        "schema": SCHEMA,
        "producer": TOOL_PATH,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "status": aggregate_status if not blockers else "needs_attention",
        "blockers": blockers,
        "baseline": baseline,
        "procedural": procedural,
        "smoke_delta_bytes_procedural_minus_baseline": smoke_delta_bytes,
        "smoke_delta_interpretation": (
            "Stage-3 smoke parser/packer signal only; Stage-4 training and paired CPU/CUDA auth eval remain required."
        ),
    }


def parse_dp1_modal_metadata(
    path: str | Path,
    *,
    variant: str,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Return the DP1 dispatch metadata fields needed for live Modal polling."""

    root = Path(repo_root).resolve()
    metadata_path = Path(path).resolve()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{metadata_path} did not contain a JSON object")
    call_id = str(payload.get("call_id") or "").strip()
    sentinels = payload.get("sentinel_files_local_sha256")
    return {
        "schema": "dp1_modal_metadata_summary_v1",
        "variant": variant,
        "metadata_path": _repo_rel(metadata_path, root),
        "metadata_sha256": _sha256_file(metadata_path),
        "label": payload.get("label"),
        "lane_id": payload.get("lane_id"),
        "call_id": call_id,
        "live_volume": payload.get("live_volume"),
        "live_volume_prefix": payload.get("live_volume_prefix"),
        "dispatched_at": payload.get("dispatched_at"),
        "max_seconds": payload.get("max_seconds"),
        "mounted_code_git_head": payload.get("mounted_code_git_head"),
        "mounted_code_git_branch": payload.get("mounted_code_git_branch"),
        "working_tree_dirty": payload.get("working_tree_dirty"),
        "working_tree_dirty_paths_count": payload.get("working_tree_dirty_paths_count"),
        "sentinel_file_count": len(sentinels) if isinstance(sentinels, Mapping) else 0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
    }


def poll_modal_call_status(
    call_id: str,
    *,
    timeout_seconds: float = 2.0,
    function_call_from_id: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Poll one Modal FunctionCall without harvesting artifacts.

    A timeout means Modal has not returned a result within the short poll
    window. This function intentionally does not write artifacts, update
    ledgers, or grant score authority; the canonical harvester remains the
    terminal-result owner.
    """

    checked_at = datetime.now(UTC).isoformat()
    if not call_id:
        return {
            "schema": "dp1_modal_call_poll_v1",
            "checked_at_utc": checked_at,
            "call_id": call_id,
            "status": "missing_call_id",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
        }
    if function_call_from_id is None:
        from modal.functions import FunctionCall  # type: ignore[import-untyped]

        function_call_from_id = FunctionCall.from_id
    try:
        result = function_call_from_id(call_id).get(timeout=timeout_seconds)
    except TimeoutError as exc:
        return {
            "schema": "dp1_modal_call_poll_v1",
            "checked_at_utc": checked_at,
            "call_id": call_id,
            "status": "running_or_pending",
            "exception_type": type(exc).__name__,
            "detail": str(exc)[:500],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
        }
    except Exception as exc:  # noqa: BLE001 - external Modal boundary
        return {
            "schema": "dp1_modal_call_poll_v1",
            "checked_at_utc": checked_at,
            "call_id": call_id,
            "status": "poll_error",
            "exception_type": type(exc).__name__,
            "detail": str(exc)[:500],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
        }
    if not isinstance(result, Mapping):
        return {
            "schema": "dp1_modal_call_poll_v1",
            "checked_at_utc": checked_at,
            "call_id": call_id,
            "status": "finished_non_dict",
            "result_type": type(result).__name__,
            "result_repr": repr(result)[:500],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
        }
    artifacts = result.get("artifacts")
    artifact_keys = sorted(str(key) for key in artifacts) if isinstance(artifacts, Mapping) else []
    stdout_tail = str(result.get("stdout_tail") or "")
    return {
        "schema": "dp1_modal_call_poll_v1",
        "checked_at_utc": checked_at,
        "call_id": call_id,
        "status": "finished",
        "returncode": result.get("returncode"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "timed_out": result.get("timed_out"),
        "artifact_count": len(artifact_keys),
        "artifact_keys": artifact_keys,
        "stdout_tail": stdout_tail[-1000:],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
    }


def build_dp1_modal_call_status(
    *,
    baseline_metadata: str | Path,
    procedural_metadata: str | Path,
    repo_root: str | Path = ".",
    timeout_seconds: float = 2.0,
    call_id_ledger_path: str | Path | None = None,
    now_utc: datetime | None = None,
    function_call_from_id: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Return paired DP1 Modal call status from local metadata files."""

    root = Path(repo_root).resolve()
    ledger_path = (
        Path(call_id_ledger_path)
        if call_id_ledger_path is not None
        else root / ".omx" / "state" / "modal_call_id_ledger.jsonl"
    )
    baseline = parse_dp1_modal_metadata(
        baseline_metadata,
        variant="baseline",
        repo_root=root,
    )
    procedural = parse_dp1_modal_metadata(
        procedural_metadata,
        variant="procedural",
        repo_root=root,
    )
    for row in (baseline, procedural):
        row["poll"] = poll_modal_call_status(
            str(row.get("call_id") or ""),
            timeout_seconds=timeout_seconds,
            function_call_from_id=function_call_from_id,
        )
        row["budget"] = _load_call_id_budget(
            call_id=str(row.get("call_id") or ""),
            ledger_path=ledger_path,
            now_utc=now_utc,
        )

    blockers: list[str] = []
    if baseline["poll"]["status"] == "missing_call_id":
        blockers.append("baseline_missing_call_id")
    if procedural["poll"]["status"] == "missing_call_id":
        blockers.append("procedural_missing_call_id")
    if baseline["poll"]["status"] == "poll_error":
        blockers.append("baseline_poll_error")
    if procedural["poll"]["status"] == "poll_error":
        blockers.append("procedural_poll_error")
    if (
        baseline["poll"]["status"] == "running_or_pending"
        and baseline.get("budget", {}).get("max_seconds_exceeded") is True
    ):
        blockers.append("baseline_modal_call_exceeded_max_seconds")
    if (
        procedural["poll"]["status"] == "running_or_pending"
        and procedural.get("budget", {}).get("max_seconds_exceeded") is True
    ):
        blockers.append("procedural_modal_call_exceeded_max_seconds")
    if (
        baseline["poll"]["status"] == "finished"
        and baseline["poll"].get("returncode") not in {0, None}
    ):
        blockers.append("baseline_finished_nonzero")
    if (
        procedural["poll"]["status"] == "finished"
        and procedural["poll"].get("returncode") not in {0, None}
    ):
        blockers.append("procedural_finished_nonzero")

    baseline_finished_ok = (
        baseline["poll"]["status"] == "finished"
        and baseline["poll"].get("returncode") == 0
    )
    procedural_finished_ok = (
        procedural["poll"]["status"] == "finished"
        and procedural["poll"].get("returncode") == 0
    )
    ready_for_training_harvest = bool(
        baseline_finished_ok and procedural_finished_ok and not blockers
    )
    if ready_for_training_harvest:
        status = "ready_for_training_harvest"
    elif blockers:
        status = "needs_attention"
    else:
        status = "running"

    return {
        "schema": MODAL_CALL_STATUS_SCHEMA,
        "producer": TOOL_PATH,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "blockers": blockers,
        "ready_for_training_harvest": ready_for_training_harvest,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "baseline": baseline,
        "procedural": procedural,
        "next_action": (
            "run tools/harvest_modal_calls.py --execute for the two call_ids, then "
            "feed harvested archive/runtime dirs into tools/plan_dp1_procedural_paired_harvest.py"
            if ready_for_training_harvest
            else "keep polling; do not dispatch paired auth eval until both training arms harvest cleanly"
        ),
    }


def render_markdown(status: Mapping[str, Any]) -> str:
    baseline = status.get("baseline") if isinstance(status.get("baseline"), Mapping) else {}
    procedural = (
        status.get("procedural") if isinstance(status.get("procedural"), Mapping) else {}
    )
    if status.get("schema") == MODAL_CALL_STATUS_SCHEMA:
        rows = [
            "# DP1 Modal Call Status",
            "",
            f"- Schema: `{status.get('schema')}`",
            f"- Status: `{status.get('status')}`",
            f"- Blockers: `{', '.join(status.get('blockers') or []) or 'none'}`",
            f"- Ready for training harvest: `{status.get('ready_for_training_harvest')}`",
            "",
            "| variant | call id | poll status | rc | artifacts | elapsed s | remaining s |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in (baseline, procedural):
            poll = row.get("poll") if isinstance(row.get("poll"), Mapping) else {}
            budget = row.get("budget") if isinstance(row.get("budget"), Mapping) else {}
            elapsed = budget.get("elapsed_since_registration_seconds")
            remaining = budget.get("seconds_until_max")
            rows.append(
                "| {variant} | `{call_id}` | `{status}` | `{rc}` | `{artifacts}` | `{elapsed}` | `{remaining}` |".format(
                    variant=row.get("variant"),
                    call_id=row.get("call_id"),
                    status=poll.get("status"),
                    rc=poll.get("returncode", ""),
                    artifacts=poll.get("artifact_count", ""),
                    elapsed=(
                        f"{float(elapsed):.1f}" if isinstance(elapsed, int | float) else ""
                    ),
                    remaining=(
                        f"{float(remaining):.1f}"
                        if isinstance(remaining, int | float)
                        else ""
                    ),
                )
            )
        rows.extend(["", f"- Next action: {status.get('next_action')}"])
        return "\n".join(rows) + "\n"

    rows = [
        "# DP1 Live Modal Status",
        "",
        f"- Schema: `{status.get('schema')}`",
        f"- Status: `{status.get('status')}`",
        f"- Blockers: `{', '.join(status.get('blockers') or []) or 'none'}`",
        f"- Smoke byte delta procedural-baseline: `{status.get('smoke_delta_bytes_procedural_minus_baseline')}`",
        "",
        "| variant | stage 4 started | finished | smoke bytes | deterministic warnings |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in (baseline, procedural):
        smoke = row.get("smoke_roundtrip") if isinstance(row.get("smoke_roundtrip"), Mapping) else {}
        rows.append(
            "| {variant} | {stage4} | {finished} | {bytes} | {warnings} |".format(
                variant=row.get("variant"),
                stage4=row.get("stage4_full_training_started"),
                finished=row.get("finished"),
                bytes=smoke.get("archive_bytes", ""),
                warnings=row.get("deterministic_warning_count"),
            )
        )
    replacement = procedural.get("procedural_codebook_replacement")
    if isinstance(replacement, Mapping):
        rows.extend(
            [
                "",
                "## Procedural Replacement",
                "",
                f"- Before bytes: `{replacement.get('before_bytes')}`",
                f"- After bytes: `{replacement.get('after_bytes')}`",
                f"- Saved bytes: `{replacement.get('saved_bytes')}`",
                f"- Predicted rate-only delta S: `{replacement.get('predicted_delta_s')}`",
            ]
        )
    return "\n".join(rows) + "\n"


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "SCHEMA",
    "MODAL_CALL_STATUS_SCHEMA",
    "TOOL_PATH",
    "build_dp1_live_modal_status",
    "build_dp1_modal_call_status",
    "parse_dp1_live_modal_log",
    "parse_dp1_modal_metadata",
    "poll_modal_call_status",
    "render_markdown",
    "write_json",
]
