# SPDX-License-Identifier: MIT
"""Session bus — cross-agent coordination surfaces (task #388).

Two thin, fcntl-locked coordination surfaces so long-running and parallel agents stay
coherent without polling or hand-reconstruction:

* :mod:`tac.session_bus.bulletin` — session-events broadcast (``post_event`` /
  ``events_since`` / ``staleness_check``). Answers "did anything about my subjects land
  since I started?" (the mid-round staleness gap).
* :mod:`tac.session_bus.recovery_manifest` — thin layer over the canonical
  ``.omx/state/subagent_progress.jsonl`` store: ``register_inflight`` / ``heartbeat`` /
  ``complete`` / ``recover_report`` for respawning crashed agents with context intact.

Neither surface holds durable authority — both are LIVE_STATE (gitignored). See each
module's docstring for the boundary vs. sibling stores (P1 one-fact-one-store).
"""

from __future__ import annotations

from tac.session_bus.bulletin import (
    EVENT_AGENT_COMPLETED,
    EVENT_AGENT_DIED,
    EVENT_AGENT_SPAWNED,
    EVENT_GATE_RULED,
    EVENT_KINDS,
    EVENT_MEMO_LANDED,
    EVENT_SPEC_EDITED,
    EVENT_VERDICT_LANDED,
    SessionBusError,
    StalenessReport,
    event_count,
    events_since,
    post_event,
    read_events,
    staleness_check,
)
from tac.session_bus.recovery_manifest import (
    RecoveryEntry,
    complete,
    heartbeat,
    recover_report,
    register_inflight,
    render_report,
)

__all__ = [  # noqa: RUF022 - grouped by module (bulletin / recovery) for readability
    # bulletin
    "post_event",
    "read_events",
    "events_since",
    "event_count",
    "staleness_check",
    "StalenessReport",
    "SessionBusError",
    "EVENT_KINDS",
    "EVENT_GATE_RULED",
    "EVENT_VERDICT_LANDED",
    "EVENT_SPEC_EDITED",
    "EVENT_MEMO_LANDED",
    "EVENT_AGENT_SPAWNED",
    "EVENT_AGENT_COMPLETED",
    "EVENT_AGENT_DIED",
    # recovery manifest
    "register_inflight",
    "heartbeat",
    "complete",
    "recover_report",
    "render_report",
    "RecoveryEntry",
]
