"""Canonical filtered-monitor emitter for witness ``run.log`` watches.

ONE source of truth for the question every live-run watch re-answers by hand:
*which* ``run.log`` lines are genuine state changes worth a notification, vs the
~95% benign per-epoch telemetry (``loss_terms`` / ``grad_clip_activation`` /
``witness_component_wallclock`` / ``jacobian_basin`` / pose-gate ``fired:false``)
that should stay silent — AND the failure / confound-alarm signatures that must
NEVER be filtered out (per the Monitor discipline "silence is not success:
widen for failures, narrow for progress").

The hand-built filter this replaces had two real bugs (2026-07-17): it matched
``pose.finish`` (every gate evaluation, ``fired:false`` included → a notification
every ~4 epochs) and it replayed the whole file (``tail -n +1``). Canonicalizing
here means the vocabulary is defined + tested once and emitted, not retyped and
mis-tuned per run.

Consumers:
  * the Monitor tool — ``build_monitor_command(run_dir)`` returns the
    ``tail | grep | grep -v`` pipeline string;
  * programmatic classification — ``classify_line(line)`` returns the event
    category (or ``None`` for benign) using the SAME vocabulary as the grep, so
    a Python watcher and the shell monitor agree by construction.

Sisters (do not duplicate their vocabularies — this is the runtime-log-watch
surface only):
  * ``tac.confound_gates`` — the STATIC-preflight side of the confound immune
    system (CLAUDE.md "Confound self-protection" L1 runtime alarms:
    ``spike_deadlock`` / ``frozen_epoch`` / ``term_domination`` /
    ``gnorm_hijack`` / ``adaptive_eps_INERT`` / ``confound_alarm``);
  * ``tac.witness_run_artifacts`` — the #420 run-artifact/log-path contract.

Read-only by construction: it only ever ``tail``s an existing log — safe against
a SACRED live run dir. Process-death is intentionally NOT covered here (a hard
SIGKILL emits no Traceback and ``tail`` just goes quiet); the separate
chain-watchdog owns liveness. This module owns in-log events + crash tracebacks.
"""

from __future__ import annotations

import re
from pathlib import Path

# --- genuine progress events (narrow: only the transitions worth surfacing) ---
# JSON ``"stage": "<name>"`` values that represent a real state change.
GENUINE_EVENT_STAGES: tuple[str, ...] = (
    "verdict",                 # the d_seg / d_pose measurement itself
    "verdict_async_done",      # async verdict completion (carries the result)
    "checkpoint",              # a (stage / periodic) checkpoint save
    "lever_engage",            # a lever actually engaged (not "would_fire")
    "reorient",                # self-orient converged-cache re-anchor
    "warm_start_weights_only", # a warm-start boundary
    "tie_locus",               # tie-locus displacement event
)
# stage-transition + pose-gate-FIRED markers (matched by substring, not exact JSON)
_TRANSITION_MARKERS: tuple[str, ...] = ("stage_transition", "stage_ckpt", "TieLocus")
_FIRED_TRUE = r'"fired": ?true'  # pose-finish / lever engagement actually fired

# --- failure + confound-alarm signatures (WIDE: never silently drop a crash) ---
# Crash / OOM / kill signatures. ``Error`` is case-sensitive so it does NOT match
# the benign ``"errors": []`` field emitted every epoch in the wallclock row.
FAILURE_PATTERNS: tuple[str, ...] = (
    "Traceback", "RuntimeError", "Error", "FAILED", "OOM", "Killed",
    "jetsam", "status=oom", "CRITICAL", "FATAL", "MemoryError",
)
# Runtime confound alarms (L1 of the CLAUDE.md 3-layer immune system; the
# runtime-emitter side of tac.confound_gates' static vocabulary).
CONFOUND_ALARM_PATTERNS: tuple[str, ...] = (
    "confound_alarm", "spike_deadlock", "frozen_epoch", "term_domination",
    "gnorm_hijack", "adaptive_eps_INERT",
)

# --- benign exclusions (belt-and-suspenders; applied AFTER the include grep) ---
# These strip lines that an include pattern would otherwise catch but which are
# NOT actionable. None of them can match a real Traceback / alarm line.
EXCLUDE_PATTERNS: tuple[str, ...] = (
    "ErrorCode=0",       # benign "Error" substring
    "no error",          # benign "Error" substring
    r'"fired": ?false',  # pose-gate evaluated but did NOT fire (the noise source)
    r'"errors": ?\[\]',  # empty errors field on the per-epoch wallclock row
)


def _stage_alt() -> str:
    return "|".join(re.escape(s) for s in GENUINE_EVENT_STAGES)


def build_include_regex() -> str:
    """The extended-regex the Monitor's first ``grep -E`` uses (lines to KEEP)."""

    parts = [
        rf'"stage": ?"({_stage_alt()})"',  # genuine progress stages
        _FIRED_TRUE,                        # pose-gate / lever FIRED
        *(re.escape(m) for m in _TRANSITION_MARKERS),
        *(re.escape(p) for p in FAILURE_PATTERNS),
        *(re.escape(p) for p in CONFOUND_ALARM_PATTERNS),
    ]
    return "|".join(parts)


def build_exclude_regex() -> str:
    """The extended-regex the Monitor's second ``grep -vE`` uses (lines to DROP)."""

    return "|".join(EXCLUDE_PATTERNS)


def build_monitor_command(
    run_dir: str | Path, *, log_name: str = "run.log", from_start: bool = False
) -> str:
    """Emit the ``tail -F | grep -E | grep -vE`` pipeline for a witness run.

    ``from_start=False`` (default) starts at the tail (``-n0``) so re-arming the
    monitor never replays the whole file. Set ``from_start=True`` only for a
    one-shot backfill scan.
    """

    log = (Path(run_dir) / log_name).as_posix()
    tail_flag = "-n +1" if from_start else "-n0"
    inc = build_include_regex()
    exc = build_exclude_regex()
    return (
        f'tail {tail_flag} -F "{log}" 2>/dev/null '
        f"| /usr/bin/grep -E --line-buffered '{inc}' "
        f"| /usr/bin/grep -vE --line-buffered '{exc}'"
    )


# category → compiled include pattern (first match wins, failure/alarm before
# progress so a crash line is classified as a failure even if it also mentions a
# stage). ``re.escape`` on literals; the two regex markers used verbatim.
_ORDERED_CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("failure", re.compile("|".join(re.escape(p) for p in FAILURE_PATTERNS))),
    ("confound_alarm", re.compile("|".join(
        # frozen_epoch is a benign per-verdict field ("frozen_epoch": false); the ALARM
        # emits it as a value/kind, never as ": false" — so exclude that literal form.
        r'frozen_epoch(?!"?\s*:\s*false)' if p == "frozen_epoch" else re.escape(p)
        for p in CONFOUND_ALARM_PATTERNS))),
    ("pose_gate_fired", re.compile(_FIRED_TRUE)),
    ("verdict", re.compile(r'"stage": ?"verdict(_async_done)?"')),
    ("checkpoint", re.compile(r'"stage": ?"checkpoint"|stage_ckpt')),
    ("lever_engage", re.compile(r'"stage": ?"lever_engage"')),
    ("transition", re.compile(
        r'"stage": ?"(reorient|warm_start_weights_only|tie_locus)"'
        + "|" + "|".join(re.escape(m) for m in _TRANSITION_MARKERS))),
)
_EXCLUDE_RE = re.compile(build_exclude_regex())


def classify_line(line: str) -> str | None:
    """Return the event category for a ``run.log`` line, or ``None`` if benign.

    Mirrors the shell pipeline exactly: a line surfaces iff an include category
    matches AND no benign-exclusion matches. The exclude applies uniformly (as
    the shell's second ``grep -vE`` does) — e.g. ``ErrorCode=0`` / ``no error``
    exist precisely to drop benign ``Error``-substring hits; a real ``Traceback``
    matches none of the exclusions and survives.
    """

    for category, pat in _ORDERED_CATEGORIES:
        if pat.search(line):
            return None if _EXCLUDE_RE.search(line) else category
    return None


if __name__ == "__main__":  # emit the command for a run dir (don't hand-type it)
    import sys

    if len(sys.argv) < 2:
        sys.exit("usage: python -m tac.witness_run_monitor <run_dir> [--from-start]")
    print(build_monitor_command(sys.argv[1], from_start="--from-start" in sys.argv[2:]))
