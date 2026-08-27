#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
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

THE LIVE RULE (operator 2026-08-04 base + named carve-outs): codex arms are the
DEFAULT route, via the canonical keeper (``tools/codex_arm_queue.py``), which
runs through Bash and draws no Claude quota. A spawn that OMITS ``model``
always refuses (the expensive-default-by-omission class). Two explicit grants
pass: ``model: "fable"`` (operator 2026-08-04, convocation-class) and
``model: "opus"`` (operator 2026-08-15, close-supervision contract, granted
during the codex credit outage). The model-omission history is retained because
it explains WHY the guard exists and what widening the allow-set re-exposes.

Scope notes:
  * Forks are NOT exempt. A fork inherits the parent model by tool contract, but
    it is still a Claude subagent on the same quota, so under the codex-only
    directive it is refused like any other spawn.
  * ``TAC_AGENT_MODEL_GUARD_OK=1`` is the deliberate escape hatch (record the
    reason in the prompt).

SECOND DUTY — CHARTER LINT (task #1082, ddm_cl3, 2026-08-17). Registry row
``charter_lint_is_spawn_path_conditional_20260817`` measured the gap: all three
charter-lint legs (``_lint_stale_numbers`` :1491 · ``_lint_falsified_premises``
:1532 · ``lint_charter_optimal_form`` :1702) live ONLY in
``tools/codex_arm_queue.py`` — the CODEX keeper path. Codex has been WALLED
since 2026-08-15 (#1079), so every arm spawns through the Agent tool instead,
and this hook is the ONLY PreToolUse matcher on ``Agent``. Consequence: every
charter written during the wall was UNLINTED. The genus is a correct cure wired
to a population that stopped being live (sisters: gl1's allow-listed suffixes
going blind when the live artifacts changed extension; a discovery sweep
returning zero because it searched the wrong disk). Two clauses fall out and are
implemented here:
  1. ALLOW-LISTS FAIL OPEN — so an Agent spawn this hook cannot lint is
     announced LOUDLY on stderr, never passed in silence. An unrecognised spawn
     shape is exactly how this defect was born.
  2. A SEARCH THAT FINDS NOTHING MUST PROVE IT LOOKED — ``charter_lint`` returns
     an explicit ``status`` so "ran, 0 findings" is distinguishable from "never
     ran". Vacuity is never reported as a pass.
The legs are IMPORTED from their one home, never copied: a second copy of a lint
is a second thing to drift.

Design invariants (mirroring ``tools/launch_guard_hook.py``):
  * FAIL-OPEN — any exception ⇒ allow (exit 0, no output). A PreToolUse hook
    must NEVER brick the session. Errors append (best-effort) to
    ``.omx/state/agent_model_routing_guard_errors.log``. This binds the charter
    lint doubly: it gates ALL Agent spawns, so a lint that fails closed on its
    own bug is strictly worse than the coverage gap it cures.
  * PURE decision surface — ``decide(tool_input, env)`` has no I/O; unit-tested
    in ``src/tac/tests/test_agent_model_routing_guard_hook.py``. The charter
    lint reads files, so it lives in its own ``charter_lint`` surface rather
    than widening ``decide``.
  * Block emits both the current ``hookSpecificOutput.permissionDecision``
    shape and the legacy ``decision/reason`` shape for compatibility.

Wired via ``.claude/settings.json`` ``hooks.PreToolUse`` (matcher: Agent).
Not placed in ``tac`` — Claude-workflow apparatus, not contest/codec logic.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_ERROR_LOG = _REPO / ".omx" / "state" / "agent_model_routing_guard_errors.log"
_QUEUE = _REPO / "tools" / "codex_arm_queue.py"

# OPERATOR DIRECTIVE 2026-08-04 (superseding the 07-31 "Opus for all arms" law):
# CODEX ARMS ONLY by default. No Claude subagent may be spawned — not Opus, not
# Sonnet, not Haiku, and not a fork (a fork is still a Claude subagent drawing
# the same quota). CARVE-OUT (operator 2026-08-04, later the same day, verbatim
# "You can use fable for that" — convocation-class deep work): an explicit
# `model: "fable"` spawn is permitted. The explicitness is the point — the
# 08-04 quota burn came from spawns that OMITTED `model` and silently inherited
# fable; a spawn that NAMES fable is a deliberate operator-authorized act, and
# forks (which always run the parent model) must still declare it to pass.
# CARVE-OUT 2 (operator 2026-08-15, during the codex credit outage, verbatim
# "You can use opus subagents but they will need close supervision"): an
# explicit `model: "opus"` spawn is permitted under the close-supervision
# contract (bounded charter, checkpoint discipline, MAIN reviews every landing
# — memory: opus_subagents_close_supervision_20260815). Codex remains the
# default arm route whenever credits allow. Omission still refuses.
ALLOWED_MODELS: frozenset[str] = frozenset({"fable", "opus"})
_ESCAPE_ENV = "TAC_AGENT_MODEL_GUARD_OK"

BLOCK_MESSAGE = (
    "BLOCKED by tools/agent_model_routing_guard_hook.py: Claude subagents require an "
    "EXPLICIT allowed model. Operator routing law 2026-08-04: codex arms are the "
    "default (tools/codex_arm_queue.py keeper); a spawn that OMITS `model` inherits "
    "the expensive fable session default (the 08-04 quota burn) and is refused. "
    "Named exceptions: model:'opus' (operator grant 2026-08-15, close-supervision "
    "contract — bounded charter, checkpoints, MAIN reviews every landing) and "
    "model:'fable' (operator grant 2026-08-04, convocation-class only). "
    "Last-resort escape: set {escape}=1."
)


def decide(tool_input: dict, env: dict) -> tuple[bool, str]:
    """Return ``(blocked, reason)`` for one Agent tool_input. Pure.

    With ``ALLOWED_MODELS`` empty (the live 2026-08-04 directive) every spawn is
    refused regardless of ``model`` or ``subagent_type`` — forks included, since
    a fork is still a Claude subagent on the same quota.
    """
    if env.get(_ESCAPE_ENV) == "1":
        return False, ""
    model = tool_input.get("model")
    if ALLOWED_MODELS and isinstance(model, str) and model.strip().lower() in ALLOWED_MODELS:
        return False, ""
    return True, BLOCK_MESSAGE.format(escape=_ESCAPE_ENV)


# --- charter lint (task #1082) --------------------------------------------------

#: Only ``lint_charter_optimal_form`` may gate a spawn. Every other entry point
#: is advisory FOREVER, exactly as ``codex_arm_queue.cmd_add`` treats them — that
#: file's own note reads "so TAC_CHARTER_LINT_STRICT never upgrades an FM result
#: into a refusal". Mirroring that split is the point of this binding.
_LINT_STRICT_ENV = "TAC_CHARTER_LINT_STRICT"

#: BIND THE UNION ENTRY POINTS, NEVER THE SUB-LEGS (MAIN adjudication, #1082).
#: The first version of this file enumerated three sub-legs. That is an
#: ALLOW-LIST, and allow-lists fail open: a sixth recall leg lands inside
#: ``lint_charter_recall_advisories`` and this path silently goes blind to it —
#: the exact defect this hook was built to cure, reproduced one level down inside
#: the cure. Binding the union means new sub-legs are picked up for free.
_LINT_GATING_ENTRY = "lint_charter_optimal_form"
_LINT_ADVISORY_ENTRIES = (
    "lint_charter_recall_advisories",  # union of 5 recall legs (ownership,
    # stale numbers, falsified premises, frontier literals, bare task ids) and
    # owner of the ``recall_lint_na:`` opt-out.
    "lint_charter_capability_advisories",  # cheap, deterministic, registry-read.
)
#: Entry points DELIBERATELY not bound, with the reason. The parity suite
#: enumerates every ``lint_charter_*`` in ``codex_arm_queue`` and FAILS on any
#: name that is neither bound above nor waived here — a deny-list that is loud
#: about an unrecognised member instead of passing it in silence.
_LINT_ENTRY_POINTS_WAIVED = {
    "lint_charter_fm_advisories": (
        "model-backed (fm.charter_class, timeout=15s per call). This hook is "
        "SYNCHRONOUS in the spawn path with a measured 0.02s budget; codex's "
        "cmd_add is a queueing CLI where a 15s stall is acceptable and this one "
        "is not. Genuinely path-specific, not an oversight."
    ),
}


def _load_lint_legs():
    """Import the three lint legs from their single home in ``codex_arm_queue``.

    Deliberately an import, never a copy. ``codex_arm_queue`` is a ``tools/``
    script rather than a package, so it is loaded by path; it guards its own
    ``main()`` and executes in ~7 ms with no side effects (measured 2026-08-17).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_codex_arm_queue_for_lint", _QUEUE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load lint legs from {_QUEUE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def charter_lint(tool_input: dict, env: dict) -> tuple[bool, list[str], str]:
    """Lint one Agent spawn's charter. Returns ``(blocked, messages, status)``.

    ``status`` is the clause-2 receipt — it proves the search looked:
      * ``"ran"``            — the legs executed against a real prompt.
      * ``"no-prompt"``      — an Agent spawn carrying no lintable prompt. NOT a
                               pass; the caller announces it loudly.
      * ``"unavailable:..."`` — the legs could not be loaded or run. Fail-open,
                               but never silent.

    Enforcement mirrors ``codex_arm_queue``'s env contract: only optimal-form
    problems are gateable, and ``TAC_CHARTER_LINT_STRICT=1`` turns them into a
    block. See the memo for the measured default-divergence between the two
    paths (codex ``cmd_add`` is strict-by-default since 2026-08-13; this path is
    warn-by-default per the #1082 charter, because the very charter that
    commissioned this work trips the optimal-form leg).
    """
    prompt = tool_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return False, [], "no-prompt"
    try:
        legs = _load_lint_legs()
    except Exception as exc:
        return False, [], f"unavailable:{type(exc).__name__}: {exc}"

    problems: list[str] = []
    advisories: list[str] = []
    handle = None
    try:
        # Every bound entry point takes a charter PATH, so the in-memory prompt
        # is staged to ephemeral scratch. Never an evidence path; deleted below.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", encoding="utf-8", delete=False
        ) as fh:
            handle = Path(fh.name)
            fh.write(prompt)
        problems.extend(getattr(legs, _LINT_GATING_ENTRY)(str(handle)))
        for name in _LINT_ADVISORY_ENTRIES:
            try:
                advisories.extend(getattr(legs, name)(str(handle)))
            except Exception as exc:  # one broken entry point must not mute the rest
                advisories.append(f"{name} unavailable ({type(exc).__name__}: {exc})")
    except Exception as exc:
        return False, [], f"unavailable:{type(exc).__name__}: {exc}"
    finally:
        if handle is not None:
            try:
                handle.unlink()
            except OSError:
                pass

    strict = env.get(_LINT_STRICT_ENV) == "1"
    tag = "REFUSED" if (strict and problems) else "WARN"
    messages = [f"charter-lint {tag}: {p}" for p in problems]
    messages += [f"charter-lint WARN: {a}" for a in advisories]
    return bool(strict and problems), messages, "ran"


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

        env = dict(os.environ)
        blocked, reason = decide(tool_input, env)
        if not blocked:
            # Model routing passed; now lint the charter. Wrapped so a lint bug
            # can never brick a spawn this hook had already cleared.
            try:
                lint_blocked, messages, status = charter_lint(tool_input, env)
            except Exception as exc:  # FAIL-OPEN, but LOUD
                _log_error(exc)
                print(
                    f"[agent_model_routing_guard_hook] charter-lint DID NOT RUN "
                    f"({type(exc).__name__}: {exc}) — allowing the spawn UNLINTED.",
                    file=sys.stderr,
                )
                return 0
            if status != "ran":
                # Clause 1 + 2: an Agent spawn we could not examine is announced,
                # never silently passed as clean. "no findings" and "never ran"
                # must not look the same.
                print(
                    f"[agent_model_routing_guard_hook] charter-lint DID NOT RUN "
                    f"(status={status}) — this spawn was NOT examined; allowing "
                    "UNLINTED (fail-open).",
                    file=sys.stderr,
                )
                return 0
            for line in messages:
                print(f"[agent_model_routing_guard_hook] {line}", file=sys.stderr)
            if not lint_blocked:
                return 0
            reason = (
                "BLOCKED by tools/agent_model_routing_guard_hook.py charter lint "
                f"({_LINT_STRICT_ENV}=1): " + " | ".join(messages)
            )
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
