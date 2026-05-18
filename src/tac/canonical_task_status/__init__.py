# SPDX-License-Identifier: MIT
"""Canonical task-status ledger helpers.

JSONL is the source of truth. DuckDB, CLIs, and dashboards are consumers.
"""

from .checks import (
    canonical_task_status_violations,
    check_canonical_task_status_no_dangling_transitions,
)
from .contract import (
    SCHEMA_VERSION,
    CanonicalTaskStatusCorruptError,
    CanonicalTaskStatusError,
    CanonicalTaskStatusInvalidTransitionError,
    CanonicalTaskStatusRow,
    task_id_for_memo_item,
)
from .loader import (
    latest_status_by_task_id,
    latest_statuses,
    load_canonical_task_status_strict,
)
from .query import (
    query_task_history,
    query_tasks_by_directive,
    query_tasks_by_status,
)
from .writer import (
    append_note,
    register_task,
    update_status,
    validate_ledger,
)

__all__ = [
    "SCHEMA_VERSION",
    "CanonicalTaskStatusCorruptError",
    "CanonicalTaskStatusError",
    "CanonicalTaskStatusInvalidTransitionError",
    "CanonicalTaskStatusRow",
    "append_note",
    "canonical_task_status_violations",
    "check_canonical_task_status_no_dangling_transitions",
    "latest_status_by_task_id",
    "latest_statuses",
    "load_canonical_task_status_strict",
    "query_task_history",
    "query_tasks_by_directive",
    "query_tasks_by_status",
    "register_task",
    "task_id_for_memo_item",
    "update_status",
    "validate_ledger",
]
