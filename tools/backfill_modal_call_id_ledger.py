#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""One-time backfill of the canonical Modal call_id ledger.

Crawls the historical sources of Modal call_id provenance and emits a
single canonical JSONL ledger at ``.omx/state/modal_call_id_ledger.jsonl``.
Idempotent: deduplicates by call_id; will not re-emit rows already present
in the live ledger.

Sources (in order of authority):
  1. ``experiments/results/lane_*_modal/modal_metadata.json`` — primary
     post-launch JSON metadata (call_id, lane_id, label, gpu, dispatched_at,
     mounted_code_git_head, etc.)
  2. ``experiments/results/lane_*_modal/modal_call_id.txt`` — fallback
     if metadata.json is missing (old-format directories)
  3. ``experiments/results/lane_*_modal/_harvest_summary.json`` and sister
     ``harvest_summary.json`` — when present, used to attach a follow-up
     ``harvested`` event (rc, elapsed_seconds, score if extractable).

Usage (one-shot, after canonical helper lands):

    .venv/bin/python tools/backfill_modal_call_id_ledger.py --execute

Default is DRY-RUN (prints the planned events without writing). Pass
``--execute`` to actually append rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.deploy.modal.call_id_ledger import (  # noqa: E402
    EVENT_FAILED,
    EVENT_HARVESTED,
    MODAL_CALL_ID_LEDGER_PATH,
    STATUS_FAILED,
    STATUS_HARVESTED,
    load_call_ids,
    register_dispatched_call_id,
    update_call_id_outcome,
)


def _load_metadata(metadata_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _harvest_summary_for(lane_dir: Path) -> dict[str, Any] | None:
    for candidate_name in (
        "harvest_summary.json",
        "harvested_artifacts/_harvest_summary.json",
    ):
        path = lane_dir / candidate_name
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return None


def _sanitize_harvest_summary_for_ledger(value: Any, *, repo_root: Path) -> Any:
    """Return JSON-compatible ``value`` with local absolute repo paths removed.

    The committed Modal call_id ledger is historical provenance, not a raw
    machine transcript. Preserve file identity while avoiding host-specific
    ``/Users/.../pact`` paths in the public repo.
    """
    if isinstance(value, dict):
        return {str(k): _sanitize_harvest_summary_for_ledger(v, repo_root=repo_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_harvest_summary_for_ledger(item, repo_root=repo_root) for item in value]
    if isinstance(value, str):
        root = str(repo_root.resolve())
        root_slash = root + "/"
        if value.startswith(root_slash):
            return value[len(root_slash) :]
        if value == root:
            return "."
        return value.replace(root_slash, "").replace(root, ".")
    return value


def discover_historical_dispatches(repo_root: Path) -> list[dict[str, Any]]:
    """Return list of {metadata, lane_dir} dicts for every historical dispatch."""
    results_dir = repo_root / "experiments" / "results"
    if not results_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for lane_dir in sorted(results_dir.glob("lane_*_modal")):
        metadata_path = lane_dir / "modal_metadata.json"
        call_id_txt = lane_dir / "modal_call_id.txt"

        metadata = None
        if metadata_path.is_file():
            metadata = _load_metadata(metadata_path)

        if metadata is None and call_id_txt.is_file():
            # Fallback: minimal record reconstructed from sentinel
            try:
                cid = call_id_txt.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not cid:
                continue
            # Infer lane_id + label from directory name: lane_<label>_modal
            dir_name = lane_dir.name
            label = dir_name[len("lane_") : -len("_modal")] if dir_name.endswith("_modal") else dir_name
            metadata = {
                "call_id": cid,
                "lane_id": label,
                "label": label,
                "gpu": "unknown",
                "max_seconds": None,
                "dispatched_at": None,
                "mounted_code_git_head": None,
            }

        if not metadata or not metadata.get("call_id"):
            continue

        out.append(
            {
                "metadata": metadata,
                "lane_dir": lane_dir,
                "harvest_summary": _harvest_summary_for(lane_dir),
            }
        )
    return out


def plan_backfill(
    repo_root: Path,
    *,
    ledger_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Return (dispatched_to_emit, outcome_to_emit, already_present)."""
    historical = discover_historical_dispatches(repo_root)
    existing_rows = load_call_ids(ledger_path)
    existing_call_ids = {r["call_id"] for r in existing_rows if isinstance(r.get("call_id"), str)}
    existing_terminal_call_ids = {
        r["call_id"]
        for r in existing_rows
        if isinstance(r.get("call_id"), str)
        and r.get("status") in {"harvested", "failed", "stale", "manually_terminated"}
    }
    dispatched_to_emit: list[dict[str, Any]] = []
    outcome_to_emit: list[dict[str, Any]] = []

    for entry in historical:
        meta = entry["metadata"]
        cid = meta["call_id"]
        if cid not in existing_call_ids:
            dispatched_to_emit.append(entry)
        # Add outcome event if harvest summary present and not already terminal
        hs = entry["harvest_summary"]
        if hs and cid not in existing_terminal_call_ids:
            outcome_to_emit.append(entry)
    return dispatched_to_emit, outcome_to_emit, existing_call_ids


def emit_dispatched(entry: dict[str, Any]) -> None:
    meta = entry["metadata"]
    lane_dir = entry["lane_dir"]
    # Derive a fallback label/lane_id from the directory name when metadata
    # was written before lane_id was a stable field (older dispatches).
    dir_name = lane_dir.name
    fallback_label = (
        dir_name[len("lane_") : -len("_modal")]
        if dir_name.startswith("lane_") and dir_name.endswith("_modal")
        else dir_name
    )
    lane_id = meta.get("lane_id") or fallback_label
    label = meta.get("label") or fallback_label
    register_dispatched_call_id(
        call_id=str(meta["call_id"]),
        lane_id=str(lane_id),
        label=str(label),
        dispatched_at_utc=str(meta["dispatched_at"]) if meta.get("dispatched_at") else None,
        platform="modal",
        gpu=str(meta.get("gpu") or "unknown"),
        max_seconds=meta.get("max_seconds"),
        mounted_code_git_head=meta.get("mounted_code_git_head"),
        agent="backfill",
        recipe=str(meta.get("recipe") or meta.get("lane_script") or "unknown"),
    )


def emit_outcome(entry: dict[str, Any]) -> None:
    meta = entry["metadata"]
    hs = entry["harvest_summary"] or {}
    rc = hs.get("rc")
    if rc is None:
        rc = hs.get("returncode")
    status_token = hs.get("status") or ""
    if isinstance(rc, int) and rc == 0:
        status = STATUS_HARVESTED
        event_type = EVENT_HARVESTED
    elif status_token in {"expired", "function_timeout"}:
        status = "stale"
        event_type = "stale"
    else:
        status = STATUS_FAILED
        event_type = EVENT_FAILED
    update_call_id_outcome(
        call_id=str(meta["call_id"]),
        status=status,
        event_type=event_type,
        rc=rc if isinstance(rc, int) else None,
        elapsed_seconds=hs.get("elapsed_seconds"),
        agent="backfill",
        harvest_result=_sanitize_harvest_summary_for_ledger(hs, repo_root=REPO_ROOT),
        lane_id=str(meta.get("lane_id") or meta.get("label") or "unknown_lane"),
        label=str(meta.get("label") or "unknown_label"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write to the ledger. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repo root (default: auto-detect).",
    )
    args = parser.parse_args(argv)

    dispatched_to_emit, outcome_to_emit, existing_call_ids = plan_backfill(
        args.repo_root,
        ledger_path=MODAL_CALL_ID_LEDGER_PATH,
    )
    print(
        f"[backfill] {len(dispatched_to_emit)} new dispatched events to emit; "
        f"{len(outcome_to_emit)} new outcome events to emit; "
        f"{len(existing_call_ids)} call_ids already in ledger."
    )

    if not args.execute:
        for entry in dispatched_to_emit[:5]:
            meta = entry["metadata"]
            print(
                f"  DRY-RUN dispatched: call_id={meta['call_id']} "
                f"lane={meta.get('lane_id', '?')} gpu={meta.get('gpu', '?')}"
            )
        for entry in outcome_to_emit[:5]:
            meta = entry["metadata"]
            print(f"  DRY-RUN outcome:    call_id={meta['call_id']}")
        if dispatched_to_emit or outcome_to_emit:
            print("[backfill] re-run with --execute to actually append rows.")
        return 0

    for entry in dispatched_to_emit:
        emit_dispatched(entry)
    for entry in outcome_to_emit:
        emit_outcome(entry)
    print(
        f"[backfill] emitted {len(dispatched_to_emit)} dispatched + "
        f"{len(outcome_to_emit)} outcome rows to {MODAL_CALL_ID_LEDGER_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
