# SPDX-License-Identifier: MIT
"""In-flight-agent recovery manifest — a thin canonical layer OVER subagent_progress.jsonl.

Background — the recovery gap this closes
──────────────────────────────────────────
2026-07-09 (task #388): a session limit killed 3 agents mid-flight; recovery required
hand-reconstructing each one's context. The sister open failure-ledger class is
``sigurg_144_harness_kills_bg_bash_process_group`` (harness kills background processes).
The canonical crash-resume store already exists — ``.omx/state/subagent_progress.jsonl``
via ``tools/subagent_checkpoint.py`` (Catalog #206). Per P1 (one-fact-one-store) this module
is a THIN LAYER over that store, NOT a parallel one: it writes through the tool's
``append_checkpoint`` (extended additively with ``respawn_context`` + ``expected_outputs``)
and reads through the tool's ``read_checkpoints``. The new value it adds is
:func:`recover_report`: for every agent whose latest checkpoint is in-progress with no recent
heartbeat, it renders a ready-to-paste respawn block so a successor can be spawned with the
predecessor's context intact instead of from scratch.

The checkpoint tool is loaded by file path (``importlib``) so this module does not depend on
``tools/`` being an importable package.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

__all__ = [
    "DEFAULT_STALE_AFTER_SECONDS",
    "RecoveryEntry",
    "complete",
    "heartbeat",
    "recover_report",
    "register_inflight",
    "render_report",
]

# An agent whose latest in-progress checkpoint is older than this (no heartbeat) is a
# recovery candidate. 300s (5 min) mirrors the empirical spawned-long-runner death window
# (MEMORY: "SPAWNED long-runners DIE ~5min").
DEFAULT_STALE_AFTER_SECONDS = 300.0

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOL_PATH = _REPO_ROOT / "tools" / "subagent_checkpoint.py"

_tool_mod: ModuleType | None = None


def _checkpoint_tool() -> ModuleType:
    """Load (and cache) ``tools/subagent_checkpoint.py`` as a module by file path.

    Reused rather than reimplemented so both surfaces write/read the SAME store with the
    SAME lock (P1 one-fact-one-store).
    """
    global _tool_mod
    if _tool_mod is None:
        spec = importlib.util.spec_from_file_location(
            "_tac_subagent_checkpoint_tool", _TOOL_PATH
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise RuntimeError(f"could not load checkpoint tool from {_TOOL_PATH}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _tool_mod = mod
    return _tool_mod


def register_inflight(
    subagent_id: str,
    respawn_context: str,
    *,
    expected_outputs: list[str] | None = None,
    files_touched: list[str] | None = None,
    next_action: str = "",
    parent_id_or_session: str | None = None,
    lane_id: str | None = None,
    notes: str = "",
    step: int = 1,
) -> dict:
    """Register a new in-flight agent with a pointer-rich respawn context.

    ``respawn_context`` should be <=2KB and carry everything a successor needs (task#, spec
    paths, expected outputs, protocol). Writes an ``in_progress`` checkpoint through the
    canonical tool.
    """
    if not isinstance(respawn_context, str) or not respawn_context.strip():
        raise ValueError("respawn_context must be a non-empty string")
    return _checkpoint_tool().append_checkpoint(
        subagent_id=subagent_id,
        step=step,
        status="in_progress",
        files_touched=list(files_touched or []),
        next_action=next_action,
        notes=notes,
        parent_id_or_session=parent_id_or_session,
        lane_id=lane_id,
        respawn_context=respawn_context,
        expected_outputs=expected_outputs,
    )


def heartbeat(
    subagent_id: str,
    step: int,
    *,
    files_touched: list[str] | None = None,
    next_action: str = "",
    notes: str = "",
    respawn_context: str | None = None,
    expected_outputs: list[str] | None = None,
) -> dict:
    """Emit a liveness heartbeat (an ``in_progress`` checkpoint) for an agent.

    A recent heartbeat is what keeps :func:`recover_report` from flagging the agent as
    crashed. ``respawn_context`` may be refreshed here as the agent's state advances.
    """
    return _checkpoint_tool().append_checkpoint(
        subagent_id=subagent_id,
        step=step,
        status="in_progress",
        files_touched=list(files_touched or []),
        next_action=next_action,
        notes=notes,
        respawn_context=respawn_context,
        expected_outputs=expected_outputs,
    )


def complete(
    subagent_id: str,
    *,
    files_touched: list[str] | None = None,
    notes: str = "",
) -> dict:
    """Mark an agent complete so it drops out of the recovery report."""
    return _checkpoint_tool().append_checkpoint(
        subagent_id=subagent_id,
        step="complete",
        status="complete",
        files_touched=list(files_touched or []),
        next_action="",
        notes=notes,
    )


def _parse_iso(ts: str | None) -> datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    s = ts.strip()
    # Tool writes datetime.isoformat() (``+00:00``); tolerate a trailing ``Z`` too.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


@dataclass(frozen=True)
class RecoveryEntry:
    """One in-flight agent that appears crashed (no recent heartbeat)."""

    subagent_id: str
    last_step: object
    last_status: str
    last_heartbeat_utc: str | None
    age_seconds: float | None
    respawn_context: str | None
    expected_outputs: tuple[str, ...]
    files_touched: tuple[str, ...]
    next_action: str
    parent_id_or_session: str | None = None
    lane_id: str | None = None
    notes: str = ""
    _extra: dict = field(default_factory=dict, repr=False)

    def render(self) -> str:
        """A ready-to-paste respawn block for a successor prompt."""
        age = (
            f"{self.age_seconds:.0f}s ago"
            if self.age_seconds is not None
            else "unknown age"
        )
        lines = [
            f"=== RESPAWN: {self.subagent_id} ===",
            f"Last checkpoint: step {self.last_step} status {self.last_status} "
            f"at {self.last_heartbeat_utc} ({age} — no heartbeat since)",
        ]
        if self.lane_id:
            lines.append(f"Lane: {self.lane_id}")
        if self.parent_id_or_session:
            lines.append(f"Parent/session: {self.parent_id_or_session}")
        if self.files_touched:
            lines.append(f"Files touched: {', '.join(self.files_touched)}")
        if self.next_action:
            lines.append(f"Next action: {self.next_action}")
        if self.expected_outputs:
            lines.append(f"Expected outputs: {', '.join(self.expected_outputs)}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        lines.append("--- respawn context (paste into successor prompt) ---")
        lines.append(
            self.respawn_context
            if self.respawn_context
            else "(no respawn_context was recorded — reconstruct from files/next-action above)"
        )
        return "\n".join(lines)


def _records_by_agent(rows: list[dict]) -> dict[str, list[dict]]:
    """All records grouped per subagent_id, preserving append order (first-seen ordering)."""
    grouped: dict[str, list[dict]] = {}
    for rec in rows:
        sid = rec.get("subagent_id")
        if isinstance(sid, str) and sid:
            grouped.setdefault(sid, []).append(rec)
    return grouped


def _last_non_null(records: list[dict], key: str):
    """Most-recent non-null value of ``key`` across an agent's record history.

    Terse heartbeats often omit the fat ``respawn_context`` / ``expected_outputs`` that the
    initial ``register_inflight`` carried; recovery must surface the BEST available context,
    not just whatever the latest (possibly terse) row held.
    """
    for rec in reversed(records):
        val = rec.get(key)
        if val:
            return val
    return None


def recover_report(
    *,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    now: datetime | None = None,
) -> list[RecoveryEntry]:
    """Return recovery entries for in-flight agents with no recent heartbeat.

    An agent is a candidate iff its LATEST checkpoint status is not ``complete`` AND its
    latest heartbeat is older than ``stale_after_seconds`` (a record with an unparseable
    timestamp is treated as stale so it is never silently dropped). Sorted most-stale-first.
    ``respawn_context`` / ``expected_outputs`` / ``lane_id`` / ``parent`` carry forward the
    most-recent non-null value across the agent's history (terse heartbeats do not erase the
    fat context the register carried).

    ``now`` is injectable for deterministic testing.
    """
    now = now or datetime.now(UTC)
    rows = _checkpoint_tool().read_checkpoints()
    entries: list[RecoveryEntry] = []
    for sid, records in _records_by_agent(rows).items():
        rec = records[-1]  # latest record: authoritative for status/step/heartbeat time
        status = rec.get("status")
        if status == "complete":
            continue
        ts_raw = rec.get("written_at_utc")
        dt = _parse_iso(ts_raw)
        age = (now - dt).total_seconds() if dt is not None else None
        # Unparseable timestamp → treat as stale (fail toward surfacing, not hiding).
        if age is not None and age < stale_after_seconds:
            continue
        expected = _last_non_null(records, "expected_outputs") or []
        files = _last_non_null(records, "files_touched") or []
        entries.append(
            RecoveryEntry(
                subagent_id=sid,
                last_step=rec.get("step"),
                last_status=str(status),
                last_heartbeat_utc=ts_raw if isinstance(ts_raw, str) else None,
                age_seconds=age,
                respawn_context=_last_non_null(records, "respawn_context"),
                expected_outputs=tuple(
                    e for e in expected if isinstance(e, str)
                ),
                files_touched=tuple(f for f in files if isinstance(f, str)),
                next_action=str(rec.get("next_action", "")),
                parent_id_or_session=_last_non_null(records, "parent_id_or_session"),
                lane_id=_last_non_null(records, "lane_id"),
                notes=str(rec.get("notes", "")),
            )
        )
    # Most-stale first; None age (unparseable) sorts to the top.
    entries.sort(
        key=lambda e: (e.age_seconds is not None, e.age_seconds or 0.0),
        reverse=True,
    )
    return entries


def render_report(entries: list[RecoveryEntry]) -> str:
    """Render a full recovery report (all entries) as one paste-able string."""
    if not entries:
        return "[session-recover] no in-flight agents without a recent heartbeat."
    header = (
        f"[session-recover] {len(entries)} in-flight agent(s) appear crashed "
        f"(no recent heartbeat):\n"
    )
    return header + "\n\n".join(e.render() for e in entries)
