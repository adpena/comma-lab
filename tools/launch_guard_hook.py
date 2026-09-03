#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Launch-guard PreToolUse hook — the P0 machine-crash admission gate (#254)
lifted to the Claude Code harness surface (#338).

# OBSERVER_ROLE_OK:this hook classifies a proposed Bash command and never enumerates live processes

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

SIGURG_BLOCK_MESSAGE = (
    "BLOCKED by tools/launch_guard_hook.py: SIGURG/rc=144 kill class (operator "
    "permanent-fix 2026-08-04; recurring since 2026-04-28). Hand-rolled "
    "nohup/&-disown detach and long-running background Bash BOTH die to the "
    "session reaper — the canonical launcher already exists: "
    ".venv/bin/python tools/launch_detached_process.py --output-dir <run_dir> "
    "--done-receipt <name> -- <cmd...>  (true start_new_session detach + "
    "manifest + the .done receipt lands in .omx/tmp/codex_runs/ so the fleet "
    "watcher NOTIFIES MAIN on completion). Codex arms: codex_arm_queue.py "
    "saturate --spawn. Deliberate override: TAC_LAUNCH_GUARD_OK=1."
)

# Canonical detach surfaces — their presence means the launch is already immune.
_DETACH_SAFE_MARKERS = (
    "launch_detached_process.py",
    "spawn_durable_daemon.py",
    "codex_arm_queue.py",
    "_keeper.py",
    "claude_cli_delegate",
)
# Long-runner command shapes that die at the harness reaper when run through
# Bash run_in_background (measured: comma10k git clone rc=144, 2026-08-04).
_LONG_RUNNER_RE = re.compile(
    r"(?:^|[;&|]\s*|\s)(git\s+clone|rsync\s|wget\s|curl\s+[^|;]*(?:-O|-L)|"
    r"huggingface-cli\s+download)"
)


def _is_hand_rolled_detach(command: str) -> bool:
    """True for nohup/&-disown incantations outside the canonical launchers.

    codex_arm_queue.py:184 has documented since July that ``nohup ... &
    disown`` is NOT sufficient (disown clears the job table; the child stays
    in the killable group unless setsid'd) — yet the pattern kept being
    hand-typed. This makes the documented insufficiency ENFORCED."""
    if any(marker in command for marker in _DETACH_SAFE_MARKERS):
        return False
    stripped = command.strip()
    if "nohup" not in stripped:
        return False
    return "disown" in stripped or stripped.endswith("&") or "& disown" in stripped


def _is_backgrounded_long_runner(command: str, run_in_background: bool) -> bool:
    """True when a known long-runner rides Bash run_in_background (reaped ~144)."""
    if not run_in_background:
        return False
    if any(marker in command for marker in _DETACH_SAFE_MARKERS):
        return False
    return bool(_LONG_RUNNER_RE.search(command))


# --- #1121 waiter discipline — the STRUCTURAL half of the two-landing cure ------
#
# Landing 1 (`5d4b1818f5`, 2026-08-18) put WAITER_DISCIPLINE into every composed
# subagent prompt (`src/tac/subagent_contract.py:145`). That half is VOLITIONAL:
# it asks an arm not to type the shape. This half REFUSES it at the typing moment.
#
# MEASURED 2026-08-18 (ddm_iv1): (a) the NOISE half — four consecutive zero-signal
# MAIN re-invocations as backgrounded sleep-waiters expired one by one, each
# costing a full orchestrator turn; (b) the DANGEROUS half — an
# `until ! pgrep <predecessor>; do sleep; done; <launch successor>` fired ~30 min
# after its step had already been run and adjudicated, launching a duplicate on
# course to overwrite an adjudicated receipt mid-read. It was caught only because
# one notification said "completed" instead of "killed".
#
# The distinguishing test is the PREDICATE, in the contract's own words: "Wait on
# an artifact's existence, never on a clock." An artifact-bound wait is the
# CORRECT canonical pattern and stays ALLOWED — this must never refuse
# `until [ -f "$DONE" ]; do sleep 30; done`, which is exactly what arms should
# write. Only clock-bound and process-table-bound waits are refused, because only
# those can fire when nothing happened.
#
# Deliberately NOT covered: `scripts/remote_*.sh` heartbeat subshells
# (`( while true; do ...; sleep 60; done ) &`). Those run on rented GPU hosts,
# are paired with a `trap ... EXIT` by `tools/canonical_lane_template.py:69-75`,
# and have no channel into the session. They are a different population.
# The `sleep` must sit INSIDE the loop body (between `do` and `done`). Without the
# trailing `\bdone\b` this matched `git log | while read c; do echo $c; done && sleep 5`
# — a perfectly ordinary command with a trailing sleep — as a poll loop. Review
# pass 2 caught it; the false-positive cost on a PreToolUse Bash hook is every arm.
_POLL_LOOP_RE = re.compile(r"\b(?:while|until)\b.*?\bdo\b.*?\bsleep\b.*?\bdone\b", re.S)
_BARE_SLEEP_RE = re.compile(r"^\s*sleep\s+[\d.]+\s*$")
_SLEEP_THEN_ACT_RE = re.compile(r"(?:^|[;&]\s*)sleep\s+[\d.]+\s*;")
# Bracket/test forms only — a loose `-f` would match `rm -f`, `grep -f`, etc.
_ARTIFACT_PREDICATE_RE = re.compile(
    r"\[\s*!?\s*-[fesd]\s|\btest\s+!?\s*-[fesd]\s|\.done\b|\breceipt\b|--done-receipt\b"
)
_PROCESS_PREDICATE_RE = re.compile(r"\bpgrep\b|\bkill\s+-0\b|\bps\s+-p\b|\bpidof\b")
_ACTUATION_RE = re.compile(
    r"(?:\.venv/bin/)?python[0-9.]*\s|\bbash\s|\bmodal\s+run\b|\bnohup\b|launch_"
)

WAITER_BLOCK_MESSAGE = (
    "BLOCKED by tools/launch_guard_hook.py: orphan-prone waiter (#1121, measured "
    "2026-08-18). A wait bound to a CLOCK or to the PROCESS TABLE outlives its "
    "subject, expires on its own, and each death re-invokes MAIN with NO "
    "information (four consecutive zero-signal notifications in one day). Bind "
    "the wait to the completion ARTIFACT instead: launch through "
    ".venv/bin/python tools/launch_detached_process.py --output-dir <run_dir> "
    "--done-receipt <name> -- <cmd...>  (the .done receipt lands in "
    ".omx/tmp/codex_runs/ and the fleet watcher notifies MAIN exactly once), or "
    "poll a file the work itself writes: until [ -f \"$DONE\" ]; do sleep 30; "
    "done. Wait on an artifact's existence, never on a clock. Deliberate "
    "override: TAC_LAUNCH_GUARD_OK=1."
)

WAITER_ACTUATOR_BLOCK_MESSAGE = (
    "BLOCKED by tools/launch_guard_hook.py: latent actuator, not a waiter (#1121 "
    "rule 1, measured 2026-08-18). `until ! pgrep <predecessor>; do sleep; done; "
    "<launch successor>` is not a wait — a wait CONDITION and a launch DECISION "
    "have different lifetimes, and the condition can come true long after the "
    "decision stopped being correct. One such waiter fired ~30 min after its step "
    "had already been run and adjudicated, launching a duplicate on course to "
    "overwrite an adjudicated receipt mid-read. Re-decide at fire time with fresh "
    "state (derive from run dirs/receipts, with a live-pid + receipt spawn-guard "
    "— see tools/supervise_ddm_r1c_rung1.py:419), or don't fire. Deliberate "
    "override: TAC_LAUNCH_GUARD_OK=1."
)


def _head_is_readonly(command: str) -> bool:
    """True when the command merely INSPECTS text (grep/cat/echo about a waiter)."""
    stripped = command.strip()
    if not stripped:
        return True
    head = stripped.split()[0].rsplit("/", 1)[-1]
    return head in _READONLY_HEADS


def _is_orphan_prone_waiter(command: str, run_in_background: bool) -> bool:
    """#1121 rule 2 — a BACKGROUNDED wait bound to a clock rather than an artifact."""
    if not run_in_background:
        return False
    if any(marker in command for marker in _DETACH_SAFE_MARKERS):
        return False
    if _head_is_readonly(command):
        return False
    has_wait_shape = bool(
        _POLL_LOOP_RE.search(command)
        or _BARE_SLEEP_RE.search(command)
        or _SLEEP_THEN_ACT_RE.search(command)
    )
    if not has_wait_shape:
        return False
    # The artifact predicate is the whole point: it makes the waiter fire on a
    # real event. Its presence means this is the CORRECT pattern.
    return not _ARTIFACT_PREDICATE_RE.search(command)


def _is_latent_actuator_waiter(command: str) -> bool:
    """#1121 rule 1 — a process-table poll loop that LAUNCHES when it drains.

    Not gated on run_in_background: this shape is dangerous in the foreground
    too, because the damage is the duplicate launch, not the notification.
    """
    if any(marker in command for marker in _DETACH_SAFE_MARKERS):
        return False
    if _head_is_readonly(command):
        return False
    if not _POLL_LOOP_RE.search(command):
        return False
    if not _PROCESS_PREDICATE_RE.search(command):
        return False
    # Word-boundary split: a bare `rsplit("done")` also splits "abandoned",
    # "undone", "done_receipt" — and would then read the wrong text as the tail.
    tail = re.split(r"\bdone\b", command)[-1]
    return bool(_ACTUATION_RE.search(tail))


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


_CODEX_EXEC_RE = re.compile(r"\bcodex\s+exec\b")
# Commands that only READ the process table / files may name `codex exec` freely.
_READONLY_HEADS = frozenset(
    {"ps", "pgrep", "grep", "rg", "awk", "sed", "echo", "cat", "tail", "head", "wc"}
)
# Either mechanism proves session detachment: the canonical spawners, or an
# inline fork+setsid (what those spawners do).
_DETACH_MARKERS = (
    "os.setsid",
    "codex_arm_queue.py",
    "codex_companion_spawn.sh",
)
CODEX_SPAWN_BLOCK_MESSAGE = (
    "BLOCKED by tools/launch_guard_hook.py: hand-rolled `codex exec` spawn. ROOT "
    "CAUSE (measured 2026-08-04 via exit receipts): com.vertigo.claude-code-reaper — "
    "a launchd agent firing every 60s (~/Projects/fleet/scripts/claude-code-reaper.sh) — "
    "SIGTERMs ANY process matching \\b(claude|codex)\\b with no TTY and (PPID==1 or "
    "stdin=/dev/null|pipe) older than 300s. Every hand-rolled spawn shape (nohup+disown, "
    "even fork+setsid) matches those orphan criteria and dies at ~5-6 min: receipts "
    "signal=TERM at elapsed 335/337/337s; killed wk1/wk2 on 08-03 the same way.\n"
    "USE THE CANONICAL PATH: `.venv/bin/python tools/codex_arm_queue.py saturate --spawn` "
    "— it spawns via a KEEPER (ps line has no codex word; codex runs as its child with "
    "regular-file stdin, so the reaper's own live-session checks skip it), carries "
    "--add-dir for the SSD tier, and enforces the fleet cap + one-scorer rule. "
    "tools/codex_companion_spawn.sh for companion tasks.\n"
    "Deliberate exception: set TAC_LAUNCH_GUARD_OK=1."
)


def _is_hand_rolled_codex_spawn(command: str) -> bool:
    """True when the command starts a codex arm that a group signal could reap."""
    if not _CODEX_EXEC_RE.search(command):
        return False
    if any(marker in command for marker in _DETACH_MARKERS):
        return False  # canonical spawner, or an inline fork+setsid
    head = command.strip().split()[0].rsplit("/", 1)[-1] if command.strip() else ""
    if head in _READONLY_HEADS:
        return False  # inspecting the process table, not spawning
    return True


def decide(
    command: str, env: dict | None = None, run_in_background: bool = False
) -> tuple[bool, str]:
    """(allow, reason) — PURE. allow=True means the Bash call proceeds."""
    environ = os.environ if env is None else env
    if not command or not command.strip():
        return True, ""
    if environ.get("TAC_LAUNCH_GUARD_OK"):
        return True, ""
    # ONLY the explicit hatch token overrides the codex-spawn block. Round-2
    # ordering made the advertised inline hatch a no-op (dogfooding: a python
    # heredoc containing the words was blocked twice); round-3's fix of that
    # let ANY trainer safe-token bypass the codex block — executed control
    # confirmed `nohup codex exec "review tools/launch_witness_run.py" &
    # disown` (the exact killed-arm shape) was allowed. Trainer tokens scope
    # the trainer gate only; they move back below the codex check.
    if "TAC_LAUNCH_GUARD_OK" in command:
        return True, ""
    if _is_hand_rolled_codex_spawn(command):
        return False, CODEX_SPAWN_BLOCK_MESSAGE
    # SIGURG kill class — like the codex block, only the explicit hatch
    # overrides (trainer safe-tokens must not bypass a detach violation).
    if _is_hand_rolled_detach(command) or _is_backgrounded_long_runner(
        command, run_in_background
    ):
        return False, SIGURG_BLOCK_MESSAGE
    # #1121 waiter discipline. Placed with the other kill/orphan-class blocks and
    # ABOVE the trainer safe-tokens, for the same reason those were moved: a
    # trainer token must not buy a bypass of an orphan violation. The actuator
    # check runs first — it is the harm-bearing half (a duplicate launch over an
    # adjudicated receipt), the noise half only costs turns.
    if _is_latent_actuator_waiter(command):
        return False, WAITER_ACTUATOR_BLOCK_MESSAGE
    if _is_orphan_prone_waiter(command, run_in_background):
        return False, WAITER_BLOCK_MESSAGE
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
        allow, reason = decide(
            command, run_in_background=bool(tool_input.get("run_in_background"))
        )
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
