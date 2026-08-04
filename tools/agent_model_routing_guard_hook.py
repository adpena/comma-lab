#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Agent-model routing guard — PreToolUse hook on the ``Agent`` tool.

THE INCIDENT (2026-08-04, operator): weeks of Fable-5 rate limit burned in a
single day. Cause: subagents INHERIT the parent session model unless the spawn
passes ``model`` explicitly. The session client default is ``fable``, so every
arm spawned without an override silently ran on Fable-5 — six concurrent arms
at 300-600k tokens each. The standing routing law (operator 2026-07-31, memory
``m27``: "Opus for MAIN and all arms") existed the whole time and was applied
NOWHERE, because it lived in a memo instead of at the SPAWN SITE.

This hook moves the law to the spawn site: an ``Agent`` call that does not name
an allowed model is REFUSED, so the expensive default can no longer be reached
by omission.

Scope notes:
  * ``subagent_type: "fork"`` ALWAYS inherits the parent model by design (the
    tool contract ignores ``model`` for forks) — forks are exempt, since there
    is no override that could be passed.
  * ``TAC_AGENT_MODEL_GUARD_OK=1`` is the deliberate escape hatch for a
    genuinely-intended non-default routing (record the reason in the prompt).

Design invariants (mirroring ``tools/launch_guard_hook.py``):
  * FAIL-OPEN — any exception ⇒ allow (exit 0, no output). A PreToolUse hook
    must NEVER brick the session. Errors append (best-effort) to
    ``.omx/state/agent_model_routing_guard_errors.log``.
  * PURE decision surface — ``decide(tool_input, env)`` has no I/O; unit-tested
    in ``src/tac/tests/test_agent_model_routing_guard_hook.py``.
  * Block emits both the current ``hookSpecificOutput.permissionDecision``
    shape and the legacy ``decision/reason`` shape for compatibility.

Wired via ``.claude/settings.json`` ``hooks.PreToolUse`` (matcher: Agent).
Not placed in ``tac`` — Claude-workflow apparatus, not contest/codec logic.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_ERROR_LOG = _REPO / ".omx" / "state" / "agent_model_routing_guard_errors.log"

# Models an arm may be routed to WITHOUT tripping the guard. The routing law
# names Opus; haiku is allowed for genuinely trivial mechanical arms because
# naming it is still an explicit, deliberate choice (the bug was OMISSION).
ALLOWED_MODELS = frozenset({"opus", "sonnet", "haiku"})
# Forks inherit by contract — `model` is ignored for them, so there is nothing
# to enforce and refusing would be unactionable.
_INHERITING_SUBAGENT_TYPES = frozenset({"fork"})
_ESCAPE_ENV = "TAC_AGENT_MODEL_GUARD_OK"

BLOCK_MESSAGE = (
    "BLOCKED by tools/agent_model_routing_guard_hook.py: this Agent spawn names no "
    "`model`, so the subagent INHERITS the parent session model (client default: "
    "fable). That omission burned weeks of Fable-5 rate limit in one day on "
    "2026-08-04 across six concurrent arms. The standing routing law (operator "
    "2026-07-31) is Opus for MAIN and all arms. Pass model=\"opus\" explicitly "
    "(or another allowed model: {allowed}). Forks are exempt (they inherit by "
    "contract). Deliberate exception: set {escape}=1."
)


def decide(tool_input: dict, env: dict) -> tuple[bool, str]:
    """Return ``(blocked, reason)`` for one Agent tool_input. Pure."""
    if env.get(_ESCAPE_ENV) == "1":
        return False, ""
    subagent_type = tool_input.get("subagent_type")
    if isinstance(subagent_type, str) and subagent_type.strip().lower() in _INHERITING_SUBAGENT_TYPES:
        return False, ""
    model = tool_input.get("model")
    if isinstance(model, str) and model.strip().lower() in ALLOWED_MODELS:
        return False, ""
    return True, BLOCK_MESSAGE.format(
        allowed=", ".join(sorted(ALLOWED_MODELS)), escape=_ESCAPE_ENV
    )


def _log_error(exc: BaseException) -> None:
    try:  # best-effort; fail-open must stay silent to the harness
        _ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _ERROR_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.time():.3f}\t{type(exc).__name__}\t{exc}\n")
    except Exception:
        pass


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0  # nothing to judge — fail open
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return 0
        if payload.get("tool_name") not in (None, "Agent"):
            return 0
        tool_input = payload.get("tool_input")
        # A real Agent spawn always carries a prompt; an absent/empty/non-dict
        # tool_input means we are not looking at one, so judging it would deny
        # on nothing. Fail open — the guard only speaks about actual spawns.
        if not isinstance(tool_input, dict) or not tool_input:
            return 0
        import os

        blocked, reason = decide(tool_input, dict(os.environ))
        if not blocked:
            return 0
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    },
                    "decision": "block",
                    "reason": reason,
                }
            )
        )
        return 0
    except Exception as exc:  # FAIL-OPEN
        _log_error(exc)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
