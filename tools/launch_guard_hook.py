#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Launch-guard PreToolUse hook — the P0 machine-crash admission gate (#254)
lifted to the Claude Code harness surface (#338).

Refuses a RAW heavy witness-trainer launch that bypasses the governed path:
a Bash command that EXECUTES ``train_levelset_witness*.py`` under a python
interpreter, without ``safe_run`` / ``launch_witness_run`` /
``--skip-admission-gate`` and without ``TAC_LAUNCH_GUARD_OK`` set. Everything
else passes — including grep/cat/tail/vim of the trainer file, ``python -m
pytest`` over trainer tests, and tools that merely MENTION the trainer path
(execution is detected positionally: the first non-option argv token after a
python interpreter must BE the trainer script).

Design invariants:
  * FAIL-OPEN — any exception ⇒ allow (exit 0, no output). A PreToolUse hook
    must NEVER brick the session. Errors are appended (best-effort) to
    ``.omx/state/launch_guard_hook_errors.log`` so fail-open is not silent.
  * PURE decision surface — ``decide(command, env)`` has no I/O; unit-tested
    in ``src/tac/tests/test_launch_guard_hook.py``.
  * Block emits both the current ``hookSpecificOutput.permissionDecision``
    shape and the legacy ``decision/reason`` shape for compatibility.

Wired via ``.claude/settings.json`` ``hooks.PreToolUse`` (matcher: Bash).
Not placed in ``tac`` — Claude-workflow apparatus, not contest/codec logic.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_ERROR_LOG = _REPO / ".omx" / "state" / "launch_guard_hook_errors.log"
# Round-2 review F6: loud-escalation thresholds for the fail-open error log —
# >= _ESCALATION_THRESHOLD errors within _ESCALATION_WINDOW_S triggers a one-line
# stderr warning (visible in hook output) while STILL failing open (allow).
_ESCALATION_THRESHOLD = 3
_ESCALATION_WINDOW_S = 24 * 3600.0

# Trainer scripts covered by the gate (the levelset entry point + the base).
_TRAINER_RE = re.compile(r"train_levelset_witness\S*\.py$")
# A python interpreter token: python / python3 / python3.12, bare or any path
# (e.g. .venv/bin/python, /usr/bin/python3).
_PY_RE = re.compile(r"(?:^|/)python(?:3(?:\.\d+)?)?$")
# Presence of ANY of these anywhere in the command ⇒ governed/authorized path.
_SAFE_TOKENS = ("safe_run", "launch_witness_run", "--skip-admission-gate", "TAC_LAUNCH_GUARD_OK")
# Benign command prefixes we skip to find the real command word.
_WRAPPERS = {"nohup", "exec", "env", "time", "nice", "caffeinate", "stdbuf", "command"}
_SHELLS = {"bash", "sh", "zsh", "dash"}
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

BLOCK_MESSAGE = (
    "BLOCKED by tools/launch_guard_hook.py: raw witness-trainer launches bypass the "
    "memory admission gate (P0 machine-crash class, #254 — a concurrent >128GB launch "
    "has crashed this box). Use the governed path: "
    ".venv/bin/python tools/launch_witness_run.py (derives+validates the config, runs "
    "the memory preflight, emits launch.sh). For a deliberate, operator-approved raw "
    "run set TAC_LAUNCH_GUARD_OK=1 in the command."
)


def _split_segments(command: str) -> list[str]:
    """Split a shell command on ; && || | and newlines (coarse, conservative)."""
    parts = re.split(r"(?:;|&&|\|\||\||\n)", command)
    return [p.strip() for p in parts if p.strip()]


def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _segment_executes_trainer(segment: str, depth: int = 0) -> bool:
    """True iff this segment EXECUTES the trainer script under python.

    Positional detection: skip env assignments / wrapper commands / interpreter
    options; the executed unit must be a python interpreter whose first
    non-option argument matches the trainer script. ``-m``/``-c`` mean the
    executed unit is a module/inline code, not a script path ⇒ not a raw
    trainer launch. ``bash -c "<string>"`` recurses into the string (bounded).
    """
    if depth > 3:
        return False
    toks = _tokens(segment)
    i = 0
    # Skip leading VAR=val assignments and benign wrappers (+ their dash-options).
    while i < len(toks):
        tok = toks[i]
        if _ENV_ASSIGN_RE.match(tok):
            i += 1
            continue
        base = os.path.basename(tok)
        if base in _WRAPPERS:
            i += 1
            continue
        if base in _SHELLS:
            # bash -c "<string>" — recurse into the string payload.
            j = i + 1
            while j < len(toks):
                if toks[j] == "-c" and j + 1 < len(toks):
                    return any(
                        _segment_executes_trainer(seg, depth + 1)
                        for seg in _split_segments(toks[j + 1])
                    )
                j += 1
            return False
        if tok.startswith("-"):
            i += 1  # wrapper option (e.g. `env -i`, `nice -n`); conservative skip
            continue
        break
    else:
        return False
    if not _PY_RE.search(toks[i]):
        return False  # not a python execution (grep/cat/tail/vim/etc. pass)
    i += 1
    while i < len(toks):
        tok = toks[i]
        if tok in ("-m", "-c"):
            return False  # module/inline execution (e.g. python -m pytest)
        if tok.startswith("-"):
            i += 1
            continue
        return bool(_TRAINER_RE.search(tok))  # first non-option token = the script
    return False


def decide(command: str, env: dict | None = None) -> tuple[bool, str]:
    """(allow, reason) — PURE. allow=True means the Bash call proceeds."""
    environ = os.environ if env is None else env
    if not command or not command.strip():
        return True, ""
    if environ.get("TAC_LAUNCH_GUARD_OK"):
        return True, ""
    if any(tok in command for tok in _SAFE_TOKENS):
        return True, ""
    # Whole-command pass FIRST (additive: can only add blocks, never new
    # allows): catches `bash -c "python trainer.py | tee log"`, where the
    # coarse pipe-split below would break the quoted payload before shlex
    # could see it intact.
    if _segment_executes_trainer(command):
        return False, BLOCK_MESSAGE
    for segment in _split_segments(command):
        if _segment_executes_trainer(segment):
            return False, BLOCK_MESSAGE
    return True, ""


def _recent_error_count(now: float) -> int:
    """Count timestamped error-log lines within the escalation window.

    Lines are ``<epoch>\\t<ExcName>: <msg>``; legacy lines without a parseable
    leading epoch are ignored (never counted). Best-effort: any read failure
    returns 0 (the counter must never break fail-open)."""
    try:
        n = 0
        for line in _ERROR_LOG.read_text().splitlines():
            head = line.split("\t", 1)[0]
            try:
                ts = float(head)
            except ValueError:
                continue
            # 1s forward slack: the appended line's %.3f timestamp can round UP past `now`.
            if -1.0 <= now - ts <= _ESCALATION_WINDOW_S:
                n += 1
        return n
    except Exception:
        return 0


def _log_error(exc: BaseException) -> None:
    """Best-effort fail-open visibility (a silent guard degrades forever).

    Round-2 review F6 loud-escalation: when the log accrues >= 3 errors within
    24h, additionally print ONE stderr warning line (visible in hook output)
    so repeated silent degradation surfaces — while STILL failing open."""
    try:
        _ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        with _ERROR_LOG.open("a") as fh:
            fh.write(f"{now:.3f}\t{type(exc).__name__}: {exc}\n")
        if _recent_error_count(now) >= _ESCALATION_THRESHOLD:
            print(
                f"[launch_guard_hook] WARNING: fail-open error log has "
                f">={_ESCALATION_THRESHOLD} errors within 24h ({_ERROR_LOG}) — "
                "the guard is degrading; investigate (still allowing, fail-open).",
                file=sys.stderr,
            )
    except Exception:
        pass


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if payload.get("tool_name") not in (None, "Bash"):
            return 0
        tool_input = payload.get("tool_input") or {}
        command = tool_input.get("command") or ""
        allow, reason = decide(command)
        if not allow:
            print(
                json.dumps(
                    {
                        "decision": "block",
                        "reason": reason,
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": reason,
                        },
                    }
                )
            )
    except Exception as exc:  # FAIL-OPEN: never brick the session
        _log_error(exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
