# SPDX-License-Identifier: MIT
"""Registry-first join over memo follow-ons, handoffs, and task rows.

This is the coarse query surface for the follow-on backlog: memo-derived
follow-ons and handoff edges are reconciled against the repo-visible canonical
task-status ledger, while preserving the existing conservative execution
predicates from :mod:`tac.followon_ledger`.

It is scorer-free apparatus. It reports what its bounded stores can prove, and
turns undecidable rows into owned fire orders rather than dropping them.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tac.canonical_task_status import latest_statuses
from tac.followon_ledger import (
    ADVANCED,
    EXECUTED,
    HANDOFF_VERDICTS,
    LIVE,
    ORPHANED,
    STAGED,
    UNKNOWN,
    VALID_VERDICTS,
    ExecutionCorpus,
    ExecutionVerdict,
    FollowOn,
    Handoff,
    SuccessorIndex,
    classify_execution,
    classify_handoff,
    classify_task_execution,
    extract_followons,
    extract_handoffs,
    task_join_canary,
)

SCHEMA = "tac.followon_backlog_join.v1"
AXIS = "[macOS-CPU advisory; scorer-free backlog join; no scorer forwards]"
DEFAULT_OWNER = "codex-qj1-followon-drain"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASK_REF_RX = re.compile(r"#(\d{1,4})\b")
_CLOSED_STATUSES = frozenset({"completed", "cancelled"})


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z")


def _scope_dict(scope: Any) -> dict[str, Any]:
    if hasattr(scope, "as_dict"):
        return dict(scope.as_dict())
    return {}


def task_refs(text: str) -> tuple[str, ...]:
    """Task ids explicitly named in text, without the leading ``#``."""
    seen: list[str] = []
    for match in _TASK_REF_RX.finditer(text):
        value = match.group(1)
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def load_repo_task_rows(repo_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Latest canonical task-status rows as plain dictionaries."""
    root = _REPO_ROOT if repo_root is None else Path(repo_root)
    return [
        row.to_json_obj()
        for row in sorted(latest_statuses(root).values(), key=lambda r: str(r.task_id))
    ]


def task_index(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id")): dict(row) for row in rows if row.get("task_id")}


def _task_matches(refs: Iterable[str], tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ref in refs:
        row = tasks.get(str(ref))
        if row is not None:
            out.append(row)
    return out


def _concrete_owner(value: object) -> str:
    """Return a non-deferral owner string for queued rows.

    Historical ledger rows often carry placeholders such as ``MAIN`` or
    ``unassigned``. The qj1 contract explicitly rejects those as dispositions,
    so this join preserves them only inside the source row and assigns the row
    to the qj1 drain owner in the emitted disposition table.
    """
    owner = str(value or "").strip()
    folded = owner.casefold()
    if not owner:
        return DEFAULT_OWNER
    placeholder = (
        folded in {"main", "unassigned", "unknown", "tbd", "todo", "?"}
        or folded.startswith("main ")
        or folded.startswith("main/")
        or folded.startswith("main:")
        or folded.startswith("main-")
        or " main" in folded
        or "/main" in folded
        or "unassigned" in folded
    )
    return DEFAULT_OWNER if placeholder else owner


def _owner_from_matches(matches: Sequence[dict[str, Any]]) -> str:
    for row in matches:
        owner = _concrete_owner(row.get("owner"))
        if str(row.get("status")) not in _CLOSED_STATUSES and owner != DEFAULT_OWNER:
            return owner
    if matches:
        owner = _concrete_owner(matches[0].get("owner"))
        if owner != DEFAULT_OWNER:
            return owner
    return DEFAULT_OWNER


def _handoff_owner(row: Handoff, matches: Sequence[dict[str, Any]]) -> str:
    owner = _owner_from_matches(matches)
    if owner != DEFAULT_OWNER:
        return owner
    for target in row.targets:
        if target.kind == "ARM":
            return f"ddm_{target.key}"
        if target.kind == "PATH":
            return f"path-owner:{target.key}"
    return DEFAULT_OWNER


def _short(text: str, limit: int = 320) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _disposition_for_memo(verdict: ExecutionVerdict) -> tuple[str, str]:
    if verdict.verdict == EXECUTED:
        return (
            "FOLDED",
            "Output artifact is present. Hand-verify if the row is safety-critical, then close or cite the artifact.",
        )
    if verdict.verdict == ORPHANED:
        return (
            "QUEUED-WITH-FIRE-ORDER",
            "Named output or runner is missing in the scanned artifact scope. Produce the missing receipt or fold with a written reason.",
        )
    if verdict.verdict == STAGED:
        return (
            "QUEUED-WITH-FIRE-ORDER",
            "Runner exists but no output receipt is named. Run it only if still applicable, then persist a CLOSING-ARTIFACT.",
        )
    return (
        "QUEUED-WITH-FIRE-ORDER",
        "No artifact-shaped join token. Register a canonical task row or append a CLOSING-ARTIFACT before treating this as closed.",
    )


def _disposition_for_task(row: dict[str, Any], verdict: ExecutionVerdict) -> tuple[str, str]:
    status = str(row.get("status") or "")
    if verdict.verdict == EXECUTED or status in _CLOSED_STATUSES:
        return (
            "FOLDED",
            "Task is closed by status or by a present run artifact; keep the artifact citation with the closure.",
        )
    return (
        "QUEUED-WITH-FIRE-ORDER",
        "Task remains open or artifact-undecidable. Owner must append a CLOSING-ARTIFACT, typed blocker, or explicit fold.",
    )


def _disposition_for_handoff(verdict: ExecutionVerdict) -> tuple[str, str]:
    if verdict.verdict == ADVANCED:
        return (
            "FOLDED",
            "Successor activity was found. This is not proof of completion; hand-verify before closing the named work.",
        )
    if verdict.verdict == LIVE:
        return (
            "QUEUED-WITH-FIRE-ORDER",
            "Inside the drain window. Do not relaunch; re-query after the window or when the successor emits a closing artifact.",
        )
    if verdict.verdict == ORPHANED:
        return (
            "QUEUED-WITH-FIRE-ORDER",
            "Past the drain window with no successor activity in the scanned channel. Successor owner must fire or fold.",
        )
    return (
        "QUEUED-WITH-FIRE-ORDER",
        "Successor channel is unverifiable. Supply a task ledger, tracked path, or explicit closing artifact.",
    )


def _memo_record(
    row: FollowOn,
    verdict: ExecutionVerdict,
    *,
    tasks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    refs = task_refs(row.text)
    matches = _task_matches(refs, tasks)
    disposition, fire_order = _disposition_for_memo(verdict)
    return {
        "source": "memo_followon",
        "source_id": row.row_id,
        "memo": row.memo,
        "line_no": row.line_no,
        "text": _short(row.text),
        "verdict": verdict.verdict,
        "reason": verdict.reason,
        "evidence": list(verdict.evidence),
        "task_refs": list(refs),
        "task_matches": [
            {
                "task_id": str(match.get("task_id")),
                "status": str(match.get("status")),
                "owner": str(match.get("owner")),
                "title": _short(str(match.get("title") or ""), 160),
            }
            for match in matches
        ],
        "disposition": disposition,
        "owner": _owner_from_matches(matches),
        "fire_order": fire_order,
    }


def _task_record(row: dict[str, Any], verdict: ExecutionVerdict) -> dict[str, Any]:
    disposition, fire_order = _disposition_for_task(row, verdict)
    return {
        "source": "canonical_task",
        "source_id": f"task#{row.get('task_id')}",
        "task_id": str(row.get("task_id")),
        "title": _short(str(row.get("title") or "")),
        "status": str(row.get("status") or ""),
        "verdict": verdict.verdict,
        "reason": verdict.reason,
        "evidence": list(verdict.evidence),
        "disposition": disposition,
        "owner": _concrete_owner(row.get("owner")),
        "fire_order": fire_order,
    }


def _handoff_record(
    row: Handoff,
    verdict: ExecutionVerdict,
    *,
    tasks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    refs = [target.key.lstrip("#") for target in row.targets if target.kind == "TASK"]
    matches = _task_matches(refs, tasks)
    disposition, fire_order = _disposition_for_handoff(verdict)
    return {
        "source": "handoff",
        "source_id": row.row_id,
        "memo": row.memo,
        "line_no": row.line_no,
        "stratum": row.stratum,
        "heading": row.heading,
        "text": _short(row.text),
        "targets": [target.as_dict() for target in row.targets],
        "verdict": verdict.verdict,
        "reason": verdict.reason,
        "evidence": list(verdict.evidence),
        "task_matches": [
            {
                "task_id": str(match.get("task_id")),
                "status": str(match.get("status")),
                "owner": str(match.get("owner")),
                "title": _short(str(match.get("title") or ""), 160),
            }
            for match in matches
        ],
        "disposition": disposition,
        "owner": _handoff_owner(row, matches),
        "fire_order": fire_order,
    }


def _counts(values: Iterable[str], declared: Sequence[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: int(counts.get(key, 0)) for key in declared}


def build_followon_backlog_join(
    *,
    repo_root: str | Path | None = None,
    memo_root: str | Path | None = None,
    since: _dt.date | None = None,
    today: _dt.date | None = None,
    task_rows: Sequence[dict[str, Any]] | None = None,
    corpus: ExecutionCorpus | None = None,
    successor_index: SuccessorIndex | None = None,
    cache_ttl_s: float | None = 6 * 3600.0,
    max_dispositions: int | None = None,
) -> dict[str, Any]:
    """Build the bounded follow-on backlog join report.

    ``max_dispositions=None`` emits every non-folded row plus folded candidates.
    Passing a number is only for presentation/testing and is marked in the report.
    """
    root = _REPO_ROOT if repo_root is None else Path(repo_root)
    memos = root / ".omx" / "research" if memo_root is None else Path(memo_root)
    today = _dt.date.today() if today is None else today
    if since is None:
        since = today - _dt.timedelta(days=17)

    rows = list(task_rows) if task_rows is not None else load_repo_task_rows(root)
    tasks = task_index(rows)
    closed_task_ids = frozenset(
        str(row.get("task_id"))
        for row in rows
        if str(row.get("status")) in _CLOSED_STATUSES and row.get("task_id")
    )

    corpus = (
        ExecutionCorpus.build(memos, cache_ttl_s=cache_ttl_s)
        if corpus is None
        else corpus
    )

    followons, followon_scope = extract_followons(memos, since=since)
    memo_pairs = [(row, classify_execution(row, corpus)) for row in followons]
    memo_pairs.sort(key=lambda pair: (VALID_VERDICTS.index(pair[1].verdict), pair[0].memo, pair[0].line_no))

    canary_ok, task_canary = task_join_canary(corpus)
    task_pairs = (
        [(row, classify_task_execution(row, corpus)) for row in rows]
        if canary_ok
        else []
    )

    handoffs, handoff_extract_scope = extract_handoffs(memos, since=since)
    if successor_index is None:
        successor_index = SuccessorIndex.build(since=since, repo_root=root)
    handoff_pairs = [
        (
            row,
            classify_handoff(
                row,
                successor_index,
                today=today,
                closed_task_ids=closed_task_ids,
            ),
        )
        for row in handoffs
    ]
    handoff_pairs.sort(key=lambda pair: (HANDOFF_VERDICTS.index(pair[1].verdict), pair[0].memo, pair[0].line_no))

    dispositions: list[dict[str, Any]] = []
    dispositions.extend(_memo_record(row, verdict, tasks=tasks) for row, verdict in memo_pairs)
    dispositions.extend(_task_record(row, verdict) for row, verdict in task_pairs)
    dispositions.extend(_handoff_record(row, verdict, tasks=tasks) for row, verdict in handoff_pairs)

    priority = {
        "QUEUED-WITH-FIRE-ORDER": 0,
        "HONESTLY-DROPPED-WITH-REASON": 1,
        "FIRED": 2,
        "FOLDED": 3,
    }
    dispositions.sort(
        key=lambda row: (
            priority.get(str(row.get("disposition")), 9),
            str(row.get("source")),
            str(row.get("source_id")),
        )
    )
    truncated = False
    if max_dispositions is not None and len(dispositions) > max_dispositions:
        dispositions = dispositions[:max_dispositions]
        truncated = True

    task_status_counts = Counter(str(row.get("status") or "UNKNOWN") for row in rows)
    task_join_counts = _counts((verdict.verdict for _, verdict in task_pairs), (EXECUTED, UNKNOWN))
    memo_counts = _counts((verdict.verdict for _, verdict in memo_pairs), VALID_VERDICTS)
    handoff_counts = _counts((verdict.verdict for _, verdict in handoff_pairs), HANDOFF_VERDICTS)
    with_refs = sum(1 for row, _ in memo_pairs if task_refs(row.text))
    refs_present = sum(
        1
        for row, _ in memo_pairs
        if any(ref in tasks for ref in task_refs(row.text))
    )

    unowned = [
        row["source_id"]
        for row in dispositions
        if row.get("disposition") == "QUEUED-WITH-FIRE-ORDER" and not row.get("owner")
    ]

    return {
        "schema": SCHEMA,
        "generated_at_utc": _utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "repo_root": str(root),
        "since": since.isoformat(),
        "today": today.isoformat(),
        "registry": {
            "task_rows": len(rows),
            "distinct_task_ids": len(tasks),
            "status_counts": dict(sorted(task_status_counts.items())),
            "closed_task_ids": len(closed_task_ids),
        },
        "scopes": {
            "memo_followons": _scope_dict(followon_scope),
            "canonical_tasks": {
                "surface": "canonical-task-status-latest",
                "examined": len(rows),
                "declared": len(rows),
                "population": len(rows),
                "note": "repo-visible canonical task-status latest rows; not the live harness TaskList",
            },
            "task_execution_join": {
                "surface": "task-execution-join",
                "examined": len(task_pairs) if canary_ok else 0,
                "declared": len(rows),
                "population": len(rows),
                "note": task_canary,
            },
            "handoff_extraction": _scope_dict(handoff_extract_scope),
            "handoff_join": {
                "surface": "handoff-join",
                "examined": len(handoff_pairs),
                "declared": len(handoffs),
                "population": handoff_extract_scope.population,
                "note": successor_index.reason,
            },
        },
        "summaries": {
            "memo_followon_verdicts": memo_counts,
            "task_execution_verdicts": task_join_counts,
            "handoff_verdicts": handoff_counts,
            "memo_rows_with_task_refs": with_refs,
            "memo_rows_with_repo_task_match": refs_present,
            "memo_task_ref_coverage": (
                None if with_refs == 0 else refs_present / with_refs
            ),
            "dispositions": dict(Counter(str(row.get("disposition")) for row in dispositions)),
            "queued_with_owner": sum(
                1
                for row in dispositions
                if row.get("disposition") == "QUEUED-WITH-FIRE-ORDER" and row.get("owner")
            ),
            "unowned_queued_rows": len(unowned),
            "truncated": truncated,
        },
        "canaries": {
            "task_join": {"ok": canary_ok, "note": task_canary},
        },
        "dispositions": dispositions,
        "unowned_queued_row_ids": unowned,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact operator-facing disposition table."""
    summaries = report["summaries"]
    lines = [
        "# ddm_qj1 follow-on backlog join",
        "",
        f"Generated: {report['generated_at_utc']}",
        f"Axis: {report['axis']}",
        "",
        "## Answer first",
        "",
        (
            f"Memo follow-ons: {summaries['memo_followon_verdicts']}; "
            f"task rows: {summaries['task_execution_verdicts']}; "
            f"handoffs: {summaries['handoff_verdicts']}."
        ),
        (
            f"Queued rows with owner: {summaries['queued_with_owner']}; "
            f"unowned queued rows: {summaries['unowned_queued_rows']}."
        ),
        "",
        "## Disposition Head",
        "",
        "| source | id | verdict | disposition | owner | fire order |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["dispositions"][:40]:
        lines.append(
            "| {source} | {source_id} | {verdict} | {disposition} | {owner} | {fire_order} |".format(
                source=str(row.get("source", "")).replace("|", "\\|"),
                source_id=str(row.get("source_id", "")).replace("|", "\\|"),
                verdict=str(row.get("verdict", "")).replace("|", "\\|"),
                disposition=str(row.get("disposition", "")).replace("|", "\\|"),
                owner=str(row.get("owner", "")).replace("|", "\\|"),
                fire_order=_short(str(row.get("fire_order", "")), 180).replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- This is a repo-visible join, not the live harness TaskList.",
            "- `EXECUTED`/`ADVANCED` are candidate closure signals, not proof without hand verification.",
            "- Score pointer is not touched; `score_claim=false`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
