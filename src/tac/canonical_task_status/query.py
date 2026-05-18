# SPDX-License-Identifier: MIT
"""Query helpers for canonical task status."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .loader import latest_statuses, load_canonical_task_status_strict

if TYPE_CHECKING:
    from .contract import CanonicalTaskStatusRow


def query_tasks_by_status(
    status: str,
    *,
    owner: str | None = None,
    source_design_memo: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> list[CanonicalTaskStatusRow]:
    """Return latest rows matching a status and optional owner/directive."""

    source = None if source_design_memo is None else str(source_design_memo)
    rows = [
        row
        for row in latest_statuses(repo_root).values()
        if row.status == status
        and (owner is None or row.owner == owner)
        and (source is None or row.source_design_memo == source)
    ]
    return sorted(
        rows,
        key=lambda row: (
            row.predicted_delta_s_band[0] if row.predicted_delta_s_band else 0.0,
            row.event_timestamp_utc,
            row.task_id,
        ),
    )


def query_tasks_by_directive(
    source_design_memo: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> list[CanonicalTaskStatusRow]:
    source = str(source_design_memo)
    return sorted(
        (row for row in latest_statuses(repo_root).values() if row.source_design_memo == source),
        key=lambda row: row.task_id,
    )


def query_task_history(
    task_id: str,
    *,
    repo_root: str | Path | None = None,
) -> list[CanonicalTaskStatusRow]:
    return [
        row
        for row in load_canonical_task_status_strict(repo_root)
        if row.task_id == task_id
    ]
