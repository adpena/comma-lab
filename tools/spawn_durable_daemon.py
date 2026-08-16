#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Spawn / stop / inspect a TRULY detached daemon that survives the parent
process (e.g. a Claude Code process exit / context-compaction / harness restart).

WHY (start): nohup+disown is NOT enough on macOS — the child stays in the
launcher's process GROUP and dies when that group is torn down (which is what
killed the basin + dashboard + codex together when the Claude process exited).
This uses ``subprocess.Popen(start_new_session=True)`` so the child calls
``setsid()`` and gets its OWN session + process group, fully detached from the
launcher. The worker IS its own session+process-group leader (worker pid ==
pgid), with NO bash wrapper in the middle.

WARNING — SANDBOXED-LAUNCH NON-DURABILITY (the inverse of the orphan class;
EMPIRICAL ANCHOR 2026-07-07): invoking THIS spawner from a SANDBOXED agent
Bash tool call does NOT produce a durable daemon. setsid/start_new_session
detaches from the launcher's process group, but the macOS sandbox lifetime is
a SEPARATE kill scope: every process created inside the sandbox — including a
setsid'd session leader with ppid 1 — is terminated when the sandbox session
is torn down (~the agent's turn boundary). Signature: N unrelated daemons die
SIMULTANEOUSLY, silently (no traceback / no safe_run kill line / no governor
event; registry still says "running"), each mid-healthy-work; the #307/#336
measurement daemons died this way TWICE (~6 min lifetime each) before the
cause was isolated. THE FIX: launch the spawner from an UNSANDBOXED shell
(operator terminal, or an agent Bash call with the sandbox explicitly
disabled). PROOF OF DETACHMENT after launch: ``ps -o pid,pgid,ppid -p <pid>``
must show pid==pgid AND ppid==1. Related registry gap: exits are reconciled
only by an explicit ``--reconcile`` run (no automatic exit reconciliation), so
a killed daemon reads "running" until someone reconciles — do not trust the
registry status without the live-check column.

WHY (stop): the ORPHAN bug class. A daemon launched as
``nohup bash -c '... | tee LOG > /dev/null' & disown`` has a process tree
``bash-wrapper -> {python worker, tee}``. The wrapper PID != the worker PID;
killing the obvious/wrapper PID ORPHANS the python worker (it is in the pipe,
gets reparented to init, and keeps running). Result: zombie daemons that
race/conflict. EMPIRICAL ANCHOR 2026-06-23: killing the dashboard bash-wrapper
left its python child alive; two renderer pythons then both wrote the same
dashboard index.html every 20s and the operator saw a STALE stopped run.

THE FIX: ``--stop <label>`` resolves the daemon's process GROUP (pgid) from the
fcntl-locked registry and sends ``os.killpg(pgid, SIGTERM)`` — killing the
WHOLE group so the worker AND any children (tee, subprocesses) die together =
NO ORPHAN. Because the canonical launcher already starts the worker as a
session/process-group leader, the worker pid IS the pgid, so the group kill
reaches every descendant without any bash wrapper to mis-target.

Usage:
    # start (default mode):
    .venv/bin/python tools/spawn_durable_daemon.py \
        --log <path> --label <name> -- <cmd> [args...]

    # stop (process-GROUP kill — no orphan):
    .venv/bin/python tools/spawn_durable_daemon.py --stop <name>

    # status (registry + live-check):
    .venv/bin/python tools/spawn_durable_daemon.py --status

Registry: ``.omx/state/durable_daemons.json`` (fcntl-locked atomic write per the
Catalog #128 / #131 / #245 pattern — the same lock-load-mutate-save cycle as
``tac.deploy.azure.active_vms_state``). Each row:
``{label, pid, pgid, cmd, log, started_utc, cwd, status}``.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import errno
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

# --------------------------------------------------------------------------
# Canonical registry location (durable, under .omx/state/ — NEVER /tmp per
# CLAUDE.md "Forbidden /tmp paths in any persisted artifact" non-negotiable).
# --------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# Ensure the tools/ dir is importable so the sibling governor / black-box modules resolve regardless
# of the caller's sys.path (the admission gate + black-box auto-start must not silently no-op).
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from tac import process_liveness  # noqa: E402  (needs the _SRC_DIR bootstrap above)

_REGISTRY_PATH = _REPO_ROOT / ".omx" / "state" / "durable_daemons.json"
_REGISTRY_LOCK = _REPO_ROOT / ".omx" / "state" / ".durable_daemons.lock"

# In-process re-entrancy depth for the fcntl lock (mirrors the canonical
# active_vms_state depth-counter so a nested mutate cannot deadlock + so the
# atomic-write helper can refuse a write made outside the lock).
_registry_lock_depth = 0


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _registry_lock_held() -> bool:
    return _registry_lock_depth > 0


@contextlib.contextmanager
def _registry_lock(lock_path: Path | None = None):
    """Acquire fcntl exclusive lock on the registry lock file.

    Process-advisory (``fcntl.flock`` ``LOCK_EX``); concurrent launchers /
    stoppers serialize on the lock file. Re-entrant within a single process
    via the depth counter (the canonical active_vms_state pattern).
    """
    global _registry_lock_depth
    p = lock_path or _REGISTRY_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    if _registry_lock_depth > 0:
        _registry_lock_depth += 1
        try:
            yield None
        finally:
            _registry_lock_depth -= 1
        return
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _registry_lock_depth += 1
        try:
            yield fd
        finally:
            _registry_lock_depth -= 1
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _load_registry(path: Path | None = None) -> list[dict]:
    """Lenient read: corrupt/non-list returns []. Safe without the lock
    (writers commit via os.replace so readers see a stable snapshot)."""
    p = path or _REGISTRY_PATH
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _save_registry_atomic(rows: list[dict], path: Path | None = None) -> None:
    """Atomic write — unique tmp + fsync + os.replace. MUST be called inside
    ``_registry_lock`` (runtime-asserted; comment-only contracts are FORBIDDEN
    per CLAUDE.md)."""
    if not _registry_lock_held():
        raise RuntimeError(
            "_save_registry_atomic called WITHOUT holding _registry_lock. "
            "Use _update_registry_locked which owns the lock-load-mutate-save "
            "cycle (concurrency-bug guard, mirrors Catalog #140)."
        )
    p = path or _REGISTRY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    tmp = p.with_suffix(p.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    try:
        tmp.write_text(payload, encoding="utf-8")
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


def _update_registry_locked(mutate_fn, *, path=None, lock_path=None) -> list[dict]:
    """Locked transactional update: lock -> reload -> mutate -> atomic save."""
    p_path = path or _REGISTRY_PATH
    l_path = lock_path or _REGISTRY_LOCK
    with _registry_lock(l_path):
        rows = _load_registry(p_path)
        new_rows = mutate_fn(rows)
        _save_registry_atomic(new_rows, p_path)
        return new_rows


def _register_daemon(record: dict, *, path=None, lock_path=None) -> list[dict]:
    """Upsert ``record`` by label (same label re-running replaces its row)."""
    label = record["label"]

    def _upsert(rows: list[dict]) -> list[dict]:
        kept = [r for r in rows if r.get("label") != label]
        kept.append(record)
        return kept

    return _update_registry_locked(_upsert, path=path, lock_path=lock_path)


def _mark_stopped(label: str, *, path=None, lock_path=None, reason: str | None = None) -> list[dict]:
    def _mark(rows: list[dict]) -> list[dict]:
        for r in rows:
            if r.get("label") == label:
                r["status"] = "stopped"
                r["stopped_utc"] = _utc_now_iso()
                if reason:
                    r["stopped_reason"] = reason
        return rows

    return _update_registry_locked(_mark, path=path, lock_path=lock_path)


# --------------------------------------------------------------------------
# PENDING admission reservations (review-fix CRITICAL — the launch TOCTOU).
# The old flow read the registry in _system_admission_gate with no lock and
# reserved (via _register_daemon) only AFTER Popen — so two near-simultaneous
# governed launches could BOTH pass admission before either was visible to the
# other's accounting (each individually under the ceiling, their SUM over it =
# the exact 2026-07-02 crash shape). _do_start now holds _registry_lock across
# {stale-pending sweep -> admission decision -> PENDING reservation write}, so
# a second launcher blocks on the lock and then SEES the first's reservation
# (the governor's list_tracked_jobs counts fresh "admitting" rows' projected
# peaks as growth headroom). The reservation is promoted to the real running
# row right after Popen (upsert by label) or removed on spawn failure; a stale
# pending row (crashed launcher, > _PENDING_MAX_AGE_S, no pid) is swept at the
# next launch so a phantom reservation can never permanently block admission.
# Sister read-side: system_memory_governor.pending_reservation_rows.
# --------------------------------------------------------------------------
_PENDING_STATUS = "admitting"
_PENDING_MAX_AGE_S = 120.0


def _sweep_stale_pending_rows(
    rows: list[dict], *, now_ts: float | None = None, max_age_s: float = _PENDING_MAX_AGE_S
) -> list[dict]:
    """Drop STALE pending reservations: status=="admitting", no pid, and either older than
    ``max_age_s`` or carrying a missing/unparsable ``reserved_ts`` (a crashed/killed launcher must
    not leak a phantom reservation that blocks future admissions). Pure list-in/list-out — used as
    the mutate_fn of ``_update_registry_locked``."""
    now = time.time() if now_ts is None else float(now_ts)
    kept: list[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("status") == _PENDING_STATUS and not r.get("pid"):
            try:
                fresh = (now - float(r["reserved_ts"])) <= float(max_age_s)
            except (KeyError, TypeError, ValueError):
                fresh = False  # malformed timestamp -> stale by definition
            if not fresh:
                continue
        kept.append(r)
    return kept


def _write_pending_reservation(
    label: str, projected_peak_gib: float | None, *, path=None, lock_path=None
) -> dict:
    """Upsert a PENDING admission reservation row for ``label`` (called INSIDE the admission
    critical section, right after an ADMIT decision, before Popen). The row shape matches the
    daemon record so the same registry/read paths handle it; ``pid=None`` + status "admitting"
    mark it as a reservation, ``reserved_ts`` ages it for the stale sweep."""
    record = {
        "label": label,
        "pid": None,
        "pgid": None,
        "cmd": [],
        "log": "",
        "started_utc": _utc_now_iso(),
        "reserved_ts": time.time(),
        "cwd": os.getcwd(),
        "status": _PENDING_STATUS,
    }
    if projected_peak_gib is not None:
        record["projected_peak_gib"] = float(projected_peak_gib)

    def _upsert(rows: list[dict]) -> list[dict]:
        kept = [r for r in rows if r.get("label") != label]
        kept.append(record)
        return kept

    _update_registry_locked(_upsert, path=path, lock_path=lock_path)
    return record


def _clear_pending_reservation(label: str, *, path=None, lock_path=None) -> None:
    """Remove ``label``'s PENDING reservation (spawn failed / final label differs). Only rows still
    in status "admitting" are dropped — a promoted running row is never touched."""

    def _drop(rows: list[dict]) -> list[dict]:
        return [r for r in rows
                if not (r.get("label") == label and r.get("status") == _PENDING_STATUS)]

    _update_registry_locked(_drop, path=path, lock_path=lock_path)


# --------------------------------------------------------------------------
# Liveness helpers
# --------------------------------------------------------------------------
def _try_reap(pid: int) -> bool:
    """Best-effort non-blocking reap of ``pid`` IF it is our own child.

    In the canonical production flow the launcher exits right after Popen, so
    the worker is reparented to init and we are NOT its parent at --stop time —
    this is a no-op (ECHILD). It only matters in-process (tests / a launcher
    that also stops in the same run) where a SIGTERM/SIGKILL'd child would
    otherwise linger as a ZOMBIE (``os.kill(pid, 0)`` succeeds on zombies,
    falsely reporting "alive"). Returns True iff the child was reaped (now
    truly dead).
    """
    try:
        reaped, _status = os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        return False
    return reaped == pid


def _pid_alive(pid: int) -> bool:
    """True iff the PID is alive AND not a reaped zombie.

    Delegates to the canonical tri-state read (``tac.process_liveness``), with
    ``reap_own_child=True`` to PRESERVE this site's deliberate extra: a
    non-blocking ``waitpid`` first, so a zombified OWN child is reported dead
    rather than falsely alive (the rationale in ``_try_reap`` above).  The
    probe order is unchanged: reap, then signal-0, then the zombie check.

    This site was already correct on both contested legs (ESRCH => dead,
    EPERM => alive) and on ``pid <= 0``, so the migration is behaviour-
    preserving except that a zombie we CANNOT reap -- one that is not our own
    child -- now reads dead too, which is what the local docstring already
    claimed this function did.
    """
    return (
        process_liveness.pid_state(pid, reap_own_child=True)
        == process_liveness.ALIVE
    )


def _pgid_alive(pgid: int) -> bool:
    """True iff the process GROUP is alive (killpg signal 0 probe)."""
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return False
    return True


# --------------------------------------------------------------------------
# Post-spawn liveness verification (NO silent failures) — the bug class this
# closes: Popen succeeds (the FORK is fine) but the child DIES at exec (e.g.
# safe_run's `failed to start <cmd[0]>` when an unquoted shell expansion folded
# the whole command into a single argv[0]). The old flow printed a healthy
# "pid=... (detached session)" line and returned 0 regardless — masking the
# immediate failure. We now bound-wait + scan the log so a dead launch EXITS
# NONZERO with detailed debug instead of reporting a phantom-healthy daemon.
# --------------------------------------------------------------------------
_FAILURE_MARKERS = (
    "failed to start",
    "command not found",
    "Traceback (most recent call last)",
    "ModuleNotFoundError",
    "No such file or directory",
    "no command given",
    "safe_run.py:",
)


def _log_tail(log_path, n: int = 20) -> str:
    """Last ``n`` lines of the daemon log (best-effort; '' on any error)."""
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def _scan_log_for_failures(log_path) -> list[str]:
    """Return stripped log lines carrying a known spawn/exec failure marker."""
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [ln.strip() for ln in lines if any(m in ln for m in _FAILURE_MARKERS)]


def _decode_exit(code) -> str:
    """Readable status from a Popen.poll() returncode (negative => killed by signal)."""
    if code is None:
        return "still running"
    if code < 0:
        try:
            return f"killed by signal {signal.Signals(-code).name} ({-code})"
        except Exception:
            return f"killed by signal {-code}"
    return f"exited rc={code}"


def _verify_child_survived(proc, pgid: int, log_path, verify_s: float) -> tuple[bool, dict]:
    """Bounded liveness verification of a just-spawned child.

    Polls up to ``verify_s`` seconds. Returns ``(ok, info)``. ``ok`` is False ONLY
    when the child exited NONZERO during the window, OR exited rc=0 BUT the log
    carries a failure marker (a contradiction that means the real work never ran).
    A still-running child (the healthy daemon case) and a clean rc=0 fast-exit are
    both ``ok``. ``verify_s <= 0`` skips the wait (back-compat / instant return).
    """
    code = None
    if verify_s and verify_s > 0:
        deadline = time.time() + float(verify_s)
        while time.time() < deadline:
            code = proc.poll()
            if code is not None:
                break
            time.sleep(0.1)
    failures = _scan_log_for_failures(log_path)
    alive = (code is None) and (_pid_alive(proc.pid) or _pgid_alive(pgid))
    info = {"code": code, "alive": alive, "failures": failures, "exit_str": _decode_exit(code)}
    if code is not None and code != 0:
        return False, info
    if code is not None and code == 0 and failures:
        return False, info
    return True, info


def _pty_wrap(cmd: list[str], platform: str | None = None) -> list[str]:
    """Wrap ``cmd`` in script(1) so the detached child gets a CONTROLLING PTY.

    WHY (task #525, MEASURED 2026-07-17): the operator's fleet launchd agent
    ``com.vertigo.claude-code-reaper`` (60s cadence, GRACE=300s) SIGTERMs any
    process matching ``\\b(claude|codex)\\b`` that has NO controlling terminal
    (ps tty == "??") and a dead stdin (/dev/null|PIPE|FIFO) — the exact
    signature of a headless detached arm. All four of the reaper's phases
    require ``tty == ??``, so a pty-attached chain is classified as a LIVE
    terminal session (the reaper's own documented exclusion: "sessions
    attached to a terminal") and is exempt. This is honest classification,
    not evasion: a delegated arm IS a live supervised session (events-log
    START/DONE custody + codex_status liveness + STALLED detection).

    macOS/BSD form: ``script -q /dev/null cmd args...``.
    util-linux form: ``script -q -e -c "<cmd>" /dev/null``.
    """
    plat = platform if platform is not None else sys.platform
    if plat == "darwin":
        return ["/usr/bin/script", "-q", "/dev/null", *cmd]
    import shlex as _shlex
    return ["script", "-q", "-e", "-c", _shlex.join(cmd), "/dev/null"]


def spawn_detached_verified(
    cmd: list[str],
    log_path,
    *,
    with_pty: bool = False,
    verify_s: float = 3.0,
    env: dict | None = None,
    cwd=None,
):
    """Shared detached-session spawn core (single implementation — task #525).

    REFACTORED out of ``_do_start``'s Popen body and ADOPTED by
    ``tools/codex_delegate.py`` so there is no copy-paste second spawn path.
    Spawns ``cmd`` fully detached (``start_new_session=True`` → own session +
    pgid), stdin from /dev/null, stdout+stderr appended to ``log_path``.
    ``with_pty=True`` additionally allocates a controlling pty via
    :func:`_pty_wrap` (reaper immunity for claude/codex-named children).

    Returns ``(proc, pgid, ok, vinfo)`` where ``ok``/``vinfo`` come from
    :func:`_verify_child_survived` (``verify_s <= 0`` skips the bounded wait
    so callers like ``_do_start`` can run their own verification later).
    Raises ``OSError``/``FileNotFoundError`` when exec itself fails — callers
    keep their own loud failure paths.
    """
    run_cmd = [str(c) for c in cmd]
    if with_pty:
        run_cmd = _pty_wrap(run_cmd)
    log = open(log_path, "ab", buffering=0)  # noqa: SIM115 — handed to the detached child
    devnull = open(os.devnull, "rb")  # noqa: SIM115
    try:
        proc = subprocess.Popen(
            run_cmd,
            stdin=devnull,
            stdout=log,
            stderr=log,
            start_new_session=True,  # setsid() -> own session/pgroup -> survives parent
            close_fds=True,
            cwd=str(cwd) if cwd else os.getcwd(),
            env=env,
        )
    finally:
        with contextlib.suppress(Exception):
            log.close()
            devnull.close()
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = proc.pid  # raced exit; best-effort
    ok, vinfo = _verify_child_survived(proc, pgid, log_path, verify_s)
    return proc, pgid, ok, vinfo


def _synth_label(cmd: list[str], pid: int) -> str:
    """Backward-compat: synthesize a label from cmd + pid when --label omitted."""
    base = Path(cmd[0]).name if cmd else "daemon"
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", base)[:40] or "daemon"
    return f"{base}_pid{pid}"


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def _mem_preflight(a: argparse.Namespace) -> int | None:
    """OOM launch-preflight (CLAUDE.md fail-closed-if-no-space, RAM analogue).

    Refuse to START a job if launching it would breach the 30 GB free-memory
    floor: ``available - projected < min_free`` → REFUSE. Returns an exit code
    to abort with, or None to proceed. Skipped with --skip-mem-preflight (for
    the watchdog / control-plane infra that MUST launch even under pressure).

    Imports the canonical guard lazily from the sibling tools/memory_guard.py so
    this launcher has no hard import dependency if the guard is absent.
    """
    if getattr(a, "skip_mem_preflight", False):
        return None
    guard_path = Path(__file__).resolve().parent / "memory_guard.py"
    if not guard_path.exists():
        print("[durable-daemon] WARNING: memory_guard.py missing; skipping OOM preflight", file=sys.stderr)
        return None
    try:
        import importlib.util as _ilu

        spec = _ilu.spec_from_file_location("_durable_daemon_memory_guard", guard_path)
        mg = _ilu.module_from_spec(spec)
        sys.modules.setdefault("_durable_daemon_memory_guard", mg)
        spec.loader.exec_module(mg)  # type: ignore[union-attr]
        decision = mg.check_launch_ok(
            min_free_gb=float(a.min_free_gb), projected_gb=float(a.projected_gb)
        )
    except Exception as exc:  # never let a guard hiccup BLOCK a launch silently
        print(f"[durable-daemon] WARNING: OOM preflight errored ({exc!r}); proceeding", file=sys.stderr)
        return None
    if not decision.ok:
        print(
            "[durable-daemon] REFUSED: launching this job would breach the "
            f"{decision.min_free_gb:.0f}GB free-memory floor "
            f"(available={decision.available_gb:.1f}GB, projected={decision.projected_gb:.1f}GB, "
            f"headroom_after={decision.headroom_after_gb:.1f}GB). "
            "Free memory / reduce concurrency / lower --projected-gb, or pass "
            "--skip-mem-preflight ONLY for control-plane/watchdog infra.",
            file=sys.stderr,
        )
        return 3
    print(
        f"[durable-daemon] OOM preflight OK: available={decision.available_gb:.1f}GB "
        f"projected={decision.projected_gb:.1f}GB headroom_after={decision.headroom_after_gb:.1f}GB "
        f"floor={decision.min_free_gb:.0f}GB"
    )
    return None


def _is_protection_infra_cmd(cmd: list[str]) -> bool:
    """True iff cmd is control-plane / protection infra OR a read-only observability SENSE organ that
    MUST launch even under memory pressure and is never SUM-over-RAM admission-gated (else it could not
    do its job — protect us, or OBSERVE a run that by definition already fills the box).

    Two exempt classes (both still honor the per-arm safe_run RSS cap + the free-floor OOM preflight —
    only the aggregate SUM-over-RAM admission gate is bypassed, so a genuine OOM is still impossible):
      1. control-plane / protection infra (black-box / memory guard / system governor); and
      2. the score-neutral #247 costate observability SENSE organs (costate_observer_loop /
         costate_shadow_report). Per CLAUDE.md "'Off' is a tracked queue" ("observability DEFAULTS ON …
         gate ONLY on genuine compute cost"): these are tiny (0.1 GiB), strictly READ-ONLY w.r.t. the
         trainer (they only append the canonical advisory costate sidecar), and their whole
         purpose is to observe a large run — so the SUM gate (which refuses everything once the trainer
         fills the box) must NOT orphan them. The per-arm RSS cap + free-floor preflight keep them safe.
    """
    joined = " ".join(str(t) for t in cmd)
    return any(tok in joined for tok in
               ("memory_blackbox.py", "memory_guard.py", "system_memory_governor.py",
                "costate_observer_loop.py", "costate_shadow_report.py"))


def _rationale_is_real(text: str | None) -> bool:
    """Reject empty / placeholder override rationales (per Catalog #287 placeholder discipline)."""
    if not text or not text.strip():
        return False
    low = text.strip().lower()
    return low not in {"<rationale>", "<reason>", "placeholder", "tbd", "todo", "n/a"} and len(low) >= 8


def _witness_dsl_compile_hash_gate(a: argparse.Namespace, cmd: list[str]) -> int | None:
    """Governor-side Catalog #406 admission over exact launch artifacts.

    Every command that reaches a witness trainer is in scope.  A raw trainer argv
    is refused because there is no artifact from which the governor can recompute
    its DSL origin; ``bash launch.sh`` is admitted only after the provenance,
    run-manifest, hash header, exact shell argv, #332 bijection, and LawRefs agree.
    There is intentionally no override and no advisory mode.
    """

    # Observer/monitor flags carry the trainer NAME as a flag VALUE (e.g. the
    # dashboard supervisor's ``--training-sig train_levelset_witness``); such a
    # process only WATCHES a trainer and must not be classified as launching one
    # (2026-07-17 false-refuse: rc=8 blocked the always-on tunnel supervisor).
    # Scope stays fail-closed: only the known observer flag's value is excluded.
    scan_tokens: list[str] = []
    skip_next = False
    for token in cmd:
        text = str(token)
        if skip_next:
            skip_next = False
            continue
        if text == "--training-sig":
            skip_next = True
            continue
        if text.startswith("--training-sig="):
            continue
        scan_tokens.append(text)
    joined = " ".join(scan_tokens)
    witness_token = "train_levelset_witness" in joined or "train_witness" in joined
    launch_sh: Path | None = None
    shell_script: Path | None = None
    for index, token in enumerate(cmd):
        candidate: Path | None = None
        if index == 0 and str(token).endswith(".sh"):
            candidate = Path(token)
        elif token in ("bash", "sh", "/bin/bash") and index + 1 < len(cmd):
            following = Path(cmd[index + 1])
            if not str(following).startswith("-"):
                candidate = following
        if candidate is not None:
            shell_script = candidate
            if candidate.name == "launch.sh":
                launch_sh = candidate
            break
    if shell_script is not None:
        if not shell_script.is_file():
            if launch_sh is None and not witness_token:
                return None
            print(
                f"[durable-daemon] REFUSED rc=8 (Catalog #406): launch artifact missing: {shell_script}",
                file=sys.stderr,
            )
            return 8
        try:
            launch_text = shell_script.read_text(encoding="utf-8")
        except OSError as exc:
            print(
                f"[durable-daemon] REFUSED rc=8 (Catalog #406): cannot read {shell_script}: {exc}",
                file=sys.stderr,
            )
            return 8
        witness_token = witness_token or "train_levelset_witness" in launch_text or "train_witness" in launch_text
    if not witness_token:
        return None
    if shell_script is not None and shell_script.name != "launch.sh":
        print(
            "[durable-daemon] REFUSED rc=8 (Catalog #406): witness shell entrypoint "
            f"{shell_script} is not the canonical launch.sh artifact; no alternate script name "
            "can bypass DSL provenance recomputation.",
            file=sys.stderr,
        )
        return 8
    if launch_sh is None:
        print(
            "[durable-daemon] REFUSED rc=8 (Catalog #406): hand-authored witness argv "
            "has no launch.sh/dsl_provenance.json from which the governor can recompute "
            "dsl_compile_hash. Route through tools/launch_witness_run.py.",
            file=sys.stderr,
        )
        return 8
    try:
        from tac.v9_provenance_gates import verify_dsl_provenance_artifacts

        ok, detail = verify_dsl_provenance_artifacts(launch_sh)
    except Exception as exc:
        ok, detail = False, f"verifier failure: {type(exc).__name__}: {exc}"
    if not ok:
        print(
            "[durable-daemon] REFUSED rc=8 (Catalog #406 governor DSL admission): "
            f"{detail}",
            file=sys.stderr,
        )
        return 8
    match = re.search(
        r"^#\s*dsl_compile_hash:\s*([0-9a-f]{64})\s*$",
        launch_sh.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:  # verifier already requires this; keep the carried value explicit.
        print("[durable-daemon] REFUSED rc=8: missing carried DSL hash", file=sys.stderr)
        return 8
    a.dsl_compile_hash = match.group(1)
    a.dsl_launch_sh = str(launch_sh.resolve())
    print(f"[durable-daemon] GOVERNOR DSL ADMISSION OK: {detail}")
    return None


_READER_CLASS_MAX_PROJECTED_GIB = 4.0
_READER_CLASS_MAX_RSS_CAP_MB = 4096


def _reader_class_refusal(a: argparse.Namespace) -> tuple[int, str] | None:
    """Fail-closed validation of the DECLARED reader/control-plane job class (task #525).

    WHY: the SUM-over-RAM training admission model refuses a ~2GB codex/reader
    control-plane job while a 60GiB trainer runs — but such small readers are
    exactly the class the #370 control-plane exemption pattern exists for. A
    ``--job-class reader`` declaration buys a preflight-only admission lane, but
    ONLY with an explicit small envelope: a declared projected peak ≤ 4 GiB AND
    a runtime ``--rss-cap-mb`` ≤ 4096 (safe_run enforces the envelope at
    runtime, so a mislabeled trainer cannot ride the reader lane past 4GB).
    Returns ``(exit_code, message)`` to refuse, or ``None`` when valid /
    not a reader job.
    """
    if getattr(a, "job_class", "training") != "reader":
        return None
    proj = getattr(a, "projected_peak_gib", None)
    if proj is None:
        return (7, "REFUSED (--job-class reader): reader jobs must DECLARE their envelope — "
                   f"pass --projected-peak-gib <= {_READER_CLASS_MAX_PROJECTED_GIB} "
                   "(the declared projection is recorded in the daemon registry).")
    if float(proj) > _READER_CLASS_MAX_PROJECTED_GIB:
        return (7, f"REFUSED (--job-class reader): projected peak {float(proj):.1f}GiB exceeds the "
                   f"reader-class ceiling {_READER_CLASS_MAX_PROJECTED_GIB}GiB. A job this large is "
                   "not a reader/control-plane job — use the default training class (full "
                   "SUM-over-RAM admission) instead.")
    rss_cap = getattr(a, "rss_cap_mb", None)
    if not rss_cap:
        return (7, "REFUSED (--job-class reader): reader jobs must enforce their envelope at "
                   f"runtime — pass --rss-cap-mb <= {_READER_CLASS_MAX_RSS_CAP_MB} (safe_run "
                   "wraps the job so it cannot outgrow the declared reader envelope).")
    if int(rss_cap) > _READER_CLASS_MAX_RSS_CAP_MB:
        return (7, f"REFUSED (--job-class reader): --rss-cap-mb {int(rss_cap)} exceeds the "
                   f"reader-class ceiling {_READER_CLASS_MAX_RSS_CAP_MB}MiB.")
    return None


def _system_admission_gate(a: argparse.Namespace, cmd: list[str]) -> int | None:
    """SYSTEM-aware admission HARD gate (the P0 crash-prevention). Returns an exit code to ABORT with,
    or None to proceed.

    REFUSE (rc=5) when launching this job would push projected SYSTEM-wide used RAM over the adaptive
    ceiling (current used + active jobs' remaining growth-to-peak + this job's projected peak). The
    ``used`` reading is the REAL vm_stat total (counts EVERY consumer: OS + control-plane + training +
    byte-close + inflate + bsdtar + probes) — job-type-blind, the exact SUM-over-128GB gap that crashed
    the machine. Bypass ONLY via --skip-admission-gate (infra) or a real operator-quoted
    --admission-override-rationale. Protection infra (black-box/guard/governor) is auto-exempt.
    """
    if getattr(a, "skip_admission_gate", False) or _is_protection_infra_cmd(cmd):
        return None
    if getattr(a, "job_class", "training") == "reader":
        # DECLARED reader/control-plane lane (task #525; mirrors the #370 control-plane
        # exemption): the SUM-over-RAM TRAINING admission model is skipped — the OOM
        # free-floor preflight (_mem_preflight) has already run, the ≤4GiB envelope was
        # validated fail-closed by _reader_class_refusal, and safe_run enforces
        # --rss-cap-mb at runtime. The declaration + reason are recorded in the registry.
        print(
            "[durable-daemon] ADMISSION: job-class=reader — preflight-only lane "
            f"(projected_peak={float(getattr(a, 'projected_peak_gib', 0.0)):.1f}GiB "
            f"rss_cap={int(getattr(a, 'rss_cap_mb', 0))}MiB; SUM-over-RAM training "
            "admission skipped per the #370 control-plane exemption pattern)."
        )
        return None
    try:
        import system_memory_governor as gov  # tools/ on sys.path
    except Exception as exc:  # never let a governor import hiccup BLOCK a launch silently
        print(f"[durable-daemon] WARNING: system admission gate unavailable ({exc!r}); proceeding "
              f"(per-arm safe_run cap + free-floor preflight remain).", file=sys.stderr)
        return None
    projected_new = getattr(a, "projected_peak_gib", None)
    if projected_new is None:
        projected_new = float(getattr(a, "projected_gb", 25.0))
    try:
        ctx = gov.live_admission_decision(projected_new_gib=float(projected_new))
    except Exception as exc:
        print(f"[durable-daemon] WARNING: admission read failed ({exc!r}); proceeding.", file=sys.stderr)
        return None
    d = ctx.decision
    if d.admit:
        print(f"[durable-daemon] ADMISSION OK: {d.reason}")
        return None
    if _rationale_is_real(getattr(a, "admission_override_rationale", None)):
        print(f"[durable-daemon] ADMISSION OVERRIDE (operator rationale): "
              f"{a.admission_override_rationale!r} — proceeding despite: {d.reason}", file=sys.stderr)
        return None
    enforcing = gov.admission_enforcing()
    mode = "ENFORCE" if enforcing else "ADVISORY"
    print(
        f"[durable-daemon] {'REFUSED' if enforcing else 'WOULD-REFUSE (ADVISORY)'} "
        f"(system admission gate — SUM-over-RAM crash guard) [{mode}]:\n"
        f"  {d.reason}\n"
        f"  total={ctx.snapshot.total_gib:.0f}GiB "
        f"used[{'committed' if ctx.snapshot.reclaimable_ok else 'legacy'}]="
        f"{(ctx.snapshot.used_committed_gib if ctx.snapshot.reclaimable_ok else ctx.snapshot.used_gib):.1f}GiB "
        f"reclaimable_avail={ctx.snapshot.available_reclaimable_gib:.1f}GiB "
        f"legacy_used={ctx.snapshot.used_gib:.1f}GiB "
        f"ceiling={ctx.ceiling.adaptive_ceiling_gib:.1f}GiB "
        f"budget={ctx.ceiling.training_budget_gib:.1f}GiB active_jobs={len(ctx.active_jobs)} "
        f"fail_safe={ctx.snapshot.fail_safe}\n"
        f"  Another concurrent job would push the SYSTEM over the ceiling. Wait for an active job to\n"
        f"  finish, reduce this job's --projected-peak-gib, free RAM, or pass\n"
        f"  --admission-override-rationale \"<operator verbatim>\" to override.",
        file=sys.stderr,
    )
    if not enforcing:
        print("[durable-daemon] NOTE: admission gate is in ADVISORY mode (ships advisory pending "
              "independent adversarial review); PROCEEDING. Flip to enforce with "
              f"{gov.ADMISSION_ENFORCE_ENV}=1 after review.", file=sys.stderr)
        return None
    return 5


def _launch_readiness_gate(a: argparse.Namespace, cmd: list[str]) -> int | None:
    """CONFIG-FRESHNESS gate (operator 2026-07-10: "a gate/hook to protect
    against naively running long-running stuff the wrong way"). Refuses a
    WITNESS long-run whose ``launch.sh`` config skips the fire-now rungs the
    default-off decision table (#405) ranks at the top of the remaining descent.

    The memory/governor admission gate above answers "can the machine hold this
    run?"; it does NOT answer "is this the RIGHT config?". This gate does — it
    binds the #405 decision table to the launch, so firing the stale validated
    config (the exact 2026-07-10 incident: HorizonWeightedMargin 43.8% +
    StepNativeActivation 31.6% un-included) is refused with the missing rungs
    named. Returns an exit code to ABORT with, or None to proceed.

    Scope: only ``bash <…>/launch.sh`` witness-trainer commands. Escape hatch:
    ``--skip-readiness-gate`` (infra) OR an operator-quoted
    ``--readiness-override-rationale`` OR a per-rung
    ``# LAUNCH_READINESS_DEFER:<rung>=<reason>`` in the launch.sh itself.
    FAIL-OPEN on gate-own errors (import/parse) — never brick a launch on the
    gate breaking; it only blocks on POSITIVE evidence of a naive launch."""
    if getattr(a, "skip_readiness_gate", False) or _is_protection_infra_cmd(cmd):
        return None
    # Resolve the launch.sh path from a `bash <path>` command; anything else is
    # out of scope (this gate is for the governed witness launch.sh flow).
    launch_sh: Path | None = None
    for i, tok in enumerate(cmd):
        if tok.endswith("launch.sh") and Path(tok).is_file():
            launch_sh = Path(tok)
            break
        if tok in ("bash", "sh", "/bin/bash") and i + 1 < len(cmd):
            cand = Path(cmd[i + 1])
            if cand.name.endswith(".sh") and cand.is_file():
                launch_sh = cand
                break
    if launch_sh is None:
        return None
    try:
        txt = launch_sh.read_text(encoding="utf-8")
    except OSError:
        return None
    if "train_levelset_witness" not in txt and "train_witness" not in txt:
        return None  # not a witness training launch — out of scope
    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "_wlrg", Path(__file__).resolve().parent / "witness_launch_readiness_gate.py")
        _wlrg = importlib.util.module_from_spec(_spec)
        sys.modules["_wlrg"] = _wlrg
        _spec.loader.exec_module(_wlrg)
        verdict = _wlrg.assess_launch_readiness(launch_sh)
    except Exception as exc:  # gate-own breakage ⇒ fail-open
        print(f"[durable-daemon] WARNING: launch-readiness gate errored ({exc!r}); "
              "proceeding (fail-open).", file=sys.stderr)
        return None
    if verdict.proceed:
        if verdict.advisory_missing:
            print("[durable-daemon] launch-readiness: PROCEED (advisory rungs not "
                  f"included: {', '.join(verdict.advisory_missing[:6])})", file=sys.stderr)
        return None
    if _rationale_is_real(getattr(a, "readiness_override_rationale", None)):
        print("[durable-daemon] launch-readiness OVERRIDE by operator rationale "
              f"{a.readiness_override_rationale!r} — proceeding despite un-reconciled "
              "fire-now rungs.", file=sys.stderr)
        return None
    print(verdict.render(), file=sys.stderr)
    print("[durable-daemon] REFUSED (launch-readiness — config skips top decision-table "
          "rungs). Reconcile the config (add the rung's flag), record a "
          "`# LAUNCH_READINESS_DEFER:<rung>=<reason>` in the launch.sh, or pass "
          "--readiness-override-rationale \"<operator verbatim>\".", file=sys.stderr)
    return 6


def _maybe_autostart_blackbox(a: argparse.Namespace, cmd: list[str]) -> None:
    """Auto-start the always-on memory black-box recorder before a training launch (idempotent
    singleton). Skipped for the black box itself (recursion guard) + infra + --skip-blackbox-autostart."""
    if getattr(a, "skip_blackbox_autostart", False):
        return
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return  # never spawn a real daemon during tests (hermetic)
    if _is_protection_infra_cmd(cmd) or (getattr(a, "label", None) == "memory_blackbox"):
        return
    try:
        import memory_blackbox as mbb  # tools/ on sys.path
        mbb.ensure_blackbox_running(verbose=True)
    except Exception as exc:  # never let the black-box hiccup block a launch
        print(f"[durable-daemon] WARNING: black-box auto-start skipped ({exc!r})", file=sys.stderr)


_SAFE_RUN = Path(__file__).resolve().parent / "safe_run.py"
# When wrapping a training arm in safe_run for a per-arm RSS cap, default the
# walltime cap to ~14 days so safe_run's 30s default never kills a long run; the
# operator can pass an explicit --walltime-cap-s to bound it tighter.
_SAFE_RUN_DEFAULT_WALLTIME_S = 14 * 24 * 3600


def _maybe_wrap_safe_run(cmd: list[str], a: argparse.Namespace) -> list[str]:
    """Wrap ``cmd`` in tools/safe_run.py when a per-arm RSS/walltime cap is set.

    This is OOM-protection LAYER 3 (defense-in-depth, D3 review fix): every
    TRAINING-ARM spawn should bound its OWN process-group RSS so it SIGKILLs
    itself before it can balloon — bounding each arm, not just the whole machine
    (the launch-preflight = layer 1, the watchdog = layer 2). safe_run is
    control-plane-safe by construction (it only ever kills the group IT spawns).
    The recorded daemon cmd keeps the trainer script token, so the watchdog's
    custody IDENTITY gate still matches.
    """
    rss_cap = getattr(a, "rss_cap_mb", None)
    wall_cap = getattr(a, "walltime_cap_s", None)
    if not rss_cap and not wall_cap:
        return cmd  # no cap requested → unwrapped (e.g. control-plane infra)
    if not _SAFE_RUN.exists():
        print(f"[durable-daemon] WARNING: safe_run.py missing at {_SAFE_RUN}; launching UNCAPPED", file=sys.stderr)
        return cmd
    timeout_s = wall_cap if wall_cap else _SAFE_RUN_DEFAULT_WALLTIME_S
    rss_mb = rss_cap if rss_cap else 0
    wrapped = [sys.executable, str(_SAFE_RUN)]
    if rss_mb:
        wrapped += ["--rss-mb", str(int(rss_mb))]
    # (review-fix CRITICAL) The durable-daemon runs the AUTHORITATIVE system admission gate
    # (_system_admission_gate, with the ACCURATE a.projected_peak_gib) right after this wrap and
    # before Popen. Without this, safe_run's OWN gate would re-litigate the SUM-over-RAM decision a
    # SECOND time from --rss-mb/1024 — the conservative RSS *cap* (deliberately set high, e.g.
    # 90000MB -> 87.9GiB) rather than the real projection (~68-72GiB) the daemon just validated. On
    # an armed-enforce box that inflated number can FALSE-REFUSE an already-admitted governed
    # #205/witness launch (the wrapper killing the thing it protects). The daemon is the single
    # gate point, so tell the inner safe_run to SKIP its gate; forward the accurate projection too
    # as defense-in-depth (correct number if this ordering ever changes).
    wrapped += ["--skip-admission-gate"]
    _proj = getattr(a, "projected_peak_gib", None)
    if isinstance(_proj, (int, float)) and _proj > 0:
        wrapped += ["--projected-gib", str(float(_proj))]
    wrapped += ["--timeout", str(float(timeout_s)), "--label", a.label or "training_arm", "--", *cmd]
    print(
        f"[durable-daemon] per-arm cap ACTIVE (safe_run): rss-mb={rss_mb or 'none'} "
        f"timeout-s={timeout_s}",
        file=sys.stderr,
    )
    return wrapped


def _do_start(a: argparse.Namespace) -> int:
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        print("error: no command (use: --log L --label NAME -- cmd args)", file=sys.stderr)
        return 2

    # Catalog #406 is the first governor admission rung and is permanently
    # fail-closed.  Run it on the caller's exact command before safe_run wrapping,
    # memory admission, reservation, or Popen can obscure the witness argv.
    refusal = _witness_dsl_compile_hash_gate(a, cmd)
    if refusal is not None:
        return refusal

    # Reader-class declaration validation (task #525): fail-closed BEFORE any
    # preflight so a mislabeled/undeclared reader job starts nothing.
    reader_refusal = _reader_class_refusal(a)
    if reader_refusal is not None:
        rc, message = reader_refusal
        print(f"[durable-daemon] {message}", file=sys.stderr)
        return rc
    if getattr(a, "job_class", "training") == "reader":
        # The validated reader envelope IS the projection: the free-floor preflight
        # must not charge the 25GB training default against a declared ≤4GiB reader
        # (that phantom projection would re-create the exact refusal this class fixes).
        a.projected_gb = float(a.projected_peak_gib)

    # OOM launch-preflight: refuse to breach the 30 GB free floor (never OOM the
    # machine again). Runs BEFORE any Popen so a refused launch starts nothing.
    # ALWAYS on — the reader class skips only the SUM-over-RAM training model.
    refusal = _mem_preflight(a)
    if refusal is not None:
        return refusal

    # Layer 3: wrap a training arm in safe_run for a per-arm RSS/walltime cap.
    cmd = _maybe_wrap_safe_run(cmd, a)

    # Auto-start the always-on memory black-box recorder BEFORE any heavy job, so the trajectory into a
    # future crash is captured (idempotent singleton; skipped for infra + the black box itself).
    # MUST stay OUTSIDE the admission critical section below: it re-enters this module's main() via
    # a SEPARATE module instance (memory_blackbox imports spawn_durable_daemon), whose registration
    # flock on the same lock file would DEADLOCK against a lock we already hold in this process.
    _maybe_autostart_blackbox(a, cmd)

    # SYSTEM admission HARD gate (P0 crash prevention): REFUSE if launching this job would push the
    # SUM over the adaptive ceiling. Runs BEFORE any Popen so a refused launch starts nothing.
    #
    # (review-fix CRITICAL — TOCTOU) The {decision -> reservation} pair is one fcntl-locked
    # critical section: sweep stale pending rows, decide, and (on ADMIT) write a PENDING
    # reservation carrying this job's projected peak — all under _registry_lock, so a second
    # near-simultaneous governed launch blocks on the lock and then SEES this reservation in its
    # own admission arithmetic (list_tracked_jobs counts fresh "admitting" rows). The reservation
    # is promoted to the real running row right after Popen, or removed on spawn failure.
    # (2026-07-10) CONFIG-FRESHNESS gate BEFORE the admission lock: refuse a
    # naive witness long-run whose config skips the fire-now decision-table
    # rungs. Runs before Popen so a refused launch starts nothing.
    _readiness_refusal = _launch_readiness_gate(a, cmd)
    if _readiness_refusal is not None:
        return _readiness_refusal

    pending_label = a.label or f"__admitting__{uuid.uuid4().hex[:10]}"
    pending_written = False
    with _registry_lock():
        # GROUND-TRUTH reconcile BEFORE the admission projection (2026-07-11 phantom-reservation fix):
        # converge the registry the gate counts to kernel truth — mark DEAD recorded=running daemons
        # stopped (out-of-band death can never write recorded=stopped) AND drop STALE pending
        # reservations (crashed launcher: no pid, old reserved_ts). The gate previously projected off
        # phantom running/pending rows on a nearly-idle machine; the admission path never actually
        # auto-reconciled despite the docstring. Runs under the held lock; reconcile's own
        # _update_registry_locked is re-entrant. Fail-open: a hiccup here must never block an admit.
        with contextlib.suppress(Exception):
            reconcile_dead_daemons(verbose=False)
        refusal = _system_admission_gate(a, cmd)
        if refusal is not None:
            return refusal
        if not (getattr(a, "skip_admission_gate", False) or _is_protection_infra_cmd(cmd)):
            _reserve_proj = getattr(a, "projected_peak_gib", None)
            if _reserve_proj is None:
                _reserve_proj = float(getattr(a, "projected_gb", 25.0))
            try:
                _write_pending_reservation(pending_label, _reserve_proj)
                pending_written = True
            except Exception as exc:  # a reservation hiccup must never block an admitted launch
                print(f"[durable-daemon] WARNING: pending-reservation write failed ({exc!r}); "
                      "proceeding (admission decision stands).", file=sys.stderr)

    with open(a.log, "ab", buffering=0) as _launch_line_log:
        _launch_line_log.write(f"[durable-daemon] launching: {' '.join(cmd)}\n".encode())
    # (#254) the P0 admission gate above JUST passed => this daemon is GOVERNED. Stamp the child
    # env (propagates through `bash launch.sh` -> trainer) so the heavy entrypoint's admission
    # guard passes. Set by stable name (== tac.admission_guard.GOVERNED_MARKER_ENV). A raw launch
    # that never reached this gate lacks the marker and is refused when enforce is armed.
    _child_env = dict(os.environ)
    _child_env["TAC_GOVERNED_ADMISSION"] = "1"
    # B2 flush-safe daemon logging (p0_launcher_chain_durability_20260717): the child's
    # stdout is THIS block-buffered log file; a hard kill (SIGKILL / sandbox teardown) EATS
    # the unflushed tail — the 20260716T211713Z chain's log truncated at 5.4K mid-history,
    # destroying the death narrative. Unbuffered python in the whole child tree makes every
    # completed write durable immediately (telemetry volume is modest; cost negligible).
    _child_env["PYTHONUNBUFFERED"] = "1"
    # B3 durable-spawn marker: descendants can PROVE they were launched via this spawner
    # (launch_witness_run's dry-start warns when a chain runs as a bare harness child).
    # NOTE this proves the spawner was used, NOT that the invoking shell was unsandboxed —
    # the sandboxed-launch non-durability class (docstring above) is invisible from inside;
    # tools/witness_chain_watchdog.py detects the resulting chain-dead-without-receipt.
    _child_env["TAC_DURABLE_SPAWN"] = "1"
    if getattr(a, "dsl_compile_hash", None):
        _child_env["TAC_DSL_COMPILE_HASH"] = str(a.dsl_compile_hash)
        _child_env["TAC_DSL_LAUNCH_SH_PATH"] = str(a.dsl_launch_sh)
        _child_env["TAC_DSL_PROVENANCE_PATH"] = str(
            Path(a.dsl_launch_sh).with_name("dsl_provenance.json")
        )
    try:
        # Shared detached-spawn core (task #525): verify_s=0 here because _do_start runs its
        # own bounded verification AFTER registry write below (behavior-preserving refactor).
        proc, pgid, _ok_unused, _vinfo_unused = spawn_detached_verified(
            cmd,
            a.log,
            with_pty=bool(getattr(a, "with_pty", False)),
            verify_s=0.0,
            env=_child_env,
            cwd=os.getcwd(),
        )
    except (FileNotFoundError, OSError) as exc:
        # NO silent failures: exec itself failed (e.g. cmd[0] is a collapsed
        # single-token command from an unquoted shell expansion, or a missing
        # interpreter). Emit detailed debug + a failure line into the daemon log,
        # then EXIT NONZERO — never report a phantom-healthy daemon.
        msg = (
            f"[durable-daemon] ERROR: failed to start child (exec failed before "
            f"any process): {exc!r}\n"
            f"  label      : {a.label}\n"
            f"  cmd[0]     : {(cmd[0] if cmd else '<none>')!r}\n"
            f"  cmd (full) : {cmd!r} (len={len(cmd)})\n"
            f"  cwd        : {os.getcwd()!r}\n"
            f"  log        : {a.log}\n"
            f"  HINT: a len(cmd)==1 cmd[0] containing spaces means an unquoted shell "
            f"expansion collapsed the command into a single argv[0]; pass separate "
            f"argv tokens or a script file (bash launch.sh)."
        )
        with contextlib.suppress(Exception), open(a.log, "ab", buffering=0) as _fail_log:
            _fail_log.write((msg + "\n").encode())
        print(msg, file=sys.stderr)
        # spawn failed -> release the pending admission reservation (nothing will grow to its peak).
        if pending_written:
            with contextlib.suppress(Exception):
                _clear_pending_reservation(pending_label)
        return 4

    label = a.label
    if not label:
        label = _synth_label(cmd, proc.pid)
        print(
            f"[durable-daemon] WARNING: no --label given; synthesized '{label}' "
            "(pass --label NAME so --stop can address it cleanly)",
            file=sys.stderr,
        )

    # The worker is its own session leader, so its pgid == its pid; the shared
    # spawn core already read the kernel-truth pgid back for the registry.

    record = {
        "label": label,
        "pid": proc.pid,
        "pgid": pgid,
        "cmd": cmd,
        "log": str(Path(a.log)),
        "started_utc": _utc_now_iso(),
        "cwd": os.getcwd(),
        "status": "running",
    }
    # System-governor metadata: the projected peak (so the NEXT launch's admission gate can sum it)
    # and an explicit throttle priority (so the governor can rank this job when shedding). getattr so a
    # partial namespace (tests / non-CLI callers) never crashes the launch.
    _proj_peak = getattr(a, "projected_peak_gib", None)
    if _proj_peak is not None:
        record["projected_peak_gib"] = float(_proj_peak)
    _prio = getattr(a, "priority", None)
    if _prio is not None:
        record["priority"] = int(_prio)
    # Reader-class custody (task #525): record the declared class + the reason the
    # SUM-over-RAM training admission was skipped, so the governor/status readers
    # can see WHY this job bypassed the training model (queryable, never silent).
    if getattr(a, "job_class", "training") == "reader":
        record["job_class"] = "reader"
        record["admission_class_reason"] = (
            "declared reader/control-plane job: preflight-only admission "
            f"(envelope projected<={_READER_CLASS_MAX_PROJECTED_GIB}GiB, "
            f"rss_cap<={_READER_CLASS_MAX_RSS_CAP_MB}MiB enforced by safe_run; "
            "#370 control-plane exemption pattern)"
        )
    try:
        _register_daemon(record)
        # Promote/replace semantics: _register_daemon upserts by label, so when the final label
        # equals the reservation label the pending row is REPLACED by the running row. A
        # synthesized final label (no --label) leaves the provisional reservation behind — clear it.
        if pending_written and label != pending_label:
            with contextlib.suppress(Exception):
                _clear_pending_reservation(pending_label)
    except Exception as exc:  # never let a registry hiccup orphan a live daemon
        print(
            f"[durable-daemon] WARNING: registry write failed ({exc}); daemon "
            f"pid={proc.pid} pgid={pgid} is RUNNING but unregistered — stop it "
            f"manually with: kill -TERM -{pgid}",
            file=sys.stderr,
        )
        # Best-effort: don't leave a pending reservation behind for a job we could not register
        # (the stale sweep would drop it in <=120s anyway; this just tightens the window).
        if pending_written:
            with contextlib.suppress(Exception):
                _clear_pending_reservation(pending_label)

    # NO silent failures: VERIFY the child actually survived exec before we report
    # a healthy detached daemon. A dead launch EXITS NONZERO with detailed debug.
    verify_s = getattr(a, "verify_s", 3.0)
    ok, vinfo = _verify_child_survived(proc, pgid, a.log, verify_s)
    if not ok:
        with contextlib.suppress(Exception):
            _mark_stopped(label, reason=f"died_at_exec_or_early_exit:{vinfo['exit_str']}")
        tail = _log_tail(a.log, 20)
        fail_block = (
            ("  failure marker(s) in log:\n    " + "\n    ".join(vinfo["failures"]) + "\n")
            if vinfo["failures"] else ""
        )
        tail_block = ("    " + tail.replace("\n", "\n    ")) if tail else "    <empty log>"
        print(
            f"[durable-daemon] ERROR: launched child DID NOT survive the "
            f"{float(verify_s):.1f}s liveness window — NO healthy daemon (the "
            f"optimistic 'detached session' report is SUPPRESSED).\n"
            f"  label      : {label}\n"
            f"  status     : {vinfo['exit_str']}\n"
            f"  cmd[0]     : {cmd[0]!r}\n"
            f"  cmd (full) : {cmd!r} (len={len(cmd)})\n"
            f"  cwd        : {os.getcwd()!r}\n"
            f"  log        : {a.log}\n"
            f"{fail_block}"
            f"  --- log tail (last 20 lines) ---\n"
            f"{tail_block}\n"
            f"  --------------------------------\n"
            f"  HINT: if cmd[0] is a long string containing spaces (len(cmd) small), an "
            f"unquoted shell expansion collapsed the command into argv[0]; pass the "
            f"command as separate argv tokens or via a script file (bash launch.sh).",
            file=sys.stderr,
        )
        return 4

    print(
        f"[durable-daemon] pid={proc.pid} pgid={pgid} label={label} "
        f"(detached session, VERIFIED alive {float(verify_s):.1f}s) log={a.log}"
    )
    return 0


def _do_stop(label: str, *, term_grace_s: float = 3.0) -> int:
    rows = _load_registry()
    matches = [r for r in rows if r.get("label") == label and r.get("status") == "running"]
    if not matches:
        # Maybe it exists but already marked stopped, or unknown label.
        any_label = [r for r in rows if r.get("label") == label]
        if any_label:
            print(f"[durable-daemon] label '{label}' already marked stopped; nothing to do.")
            return 0
        print(f"[durable-daemon] no running daemon registered under label '{label}'", file=sys.stderr)
        return 1

    rc = 0
    for r in matches:
        pid = int(r.get("pid", 0))
        pgid = int(r.get("pgid", 0)) or pid
        if not _pgid_alive(pgid) and not _pid_alive(pid):
            print(f"[durable-daemon] label '{label}' (pid={pid} pgid={pgid}) already dead.")
            continue
        # WHOLE-GROUP kill — no orphan. SIGTERM first.
        # EPERM can transiently surface when the group is mid-teardown (a member
        # became a zombie); suppress it alongside ProcessLookupError so a
        # SIGKILL escalation is never aborted by a benign race.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
        deadline = time.time() + term_grace_s
        while time.time() < deadline:
            _try_reap(pid)  # reap own-child zombie so liveness is truthful
            if not _pgid_alive(pgid) and not _pid_alive(pid):
                break
            time.sleep(0.1)
        # Escalate to SIGKILL if still alive.
        if _pgid_alive(pgid) or _pid_alive(pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
            kdeadline = time.time() + 2.0
            while time.time() < kdeadline:
                _try_reap(pid)
                if not _pgid_alive(pgid) and not _pid_alive(pid):
                    break
                time.sleep(0.05)
        if _pgid_alive(pgid) or _pid_alive(pid):
            print(
                f"[durable-daemon] ERROR: label '{label}' (pid={pid} pgid={pgid}) "
                "still alive after SIGTERM+SIGKILL",
                file=sys.stderr,
            )
            rc = 1
        else:
            print(f"[durable-daemon] stopped label '{label}' (pid={pid} pgid={pgid}) — group killed, no orphan.")

    with contextlib.suppress(Exception):
        _mark_stopped(label)
    return rc


def _stale_pending_labels(rows: list[dict], *, now_ts: float | None = None) -> set:
    """Labels of STALE pending reservations (status "admitting", no pid, missing/old ``reserved_ts``).

    The complement of ``_sweep_stale_pending_rows``'s KEEP set restricted to pending rows: a pending
    row that the sweep would DROP is stale. Kept in sync by reusing the sweep as the single source of
    truth for the freshness predicate (no duplicated timestamp logic)."""
    kept = _sweep_stale_pending_rows(rows, now_ts=now_ts)
    kept_ids = {id(r) for r in kept}
    return {
        r.get("label")
        for r in rows
        if isinstance(r, dict)
        and r.get("status") == _PENDING_STATUS
        and not r.get("pid")
        and id(r) not in kept_ids
    }


def reconcile_dead_daemons(*, verbose: bool = True, now_ts: float | None = None) -> int:
    """Converge the registry to GROUND TRUTH — the SINGLE store the admission gate counts — and return
    the number of rows reconciled. TWO reconciliations, in ONE fcntl-locked transaction (idempotent):

      1. mark every recorded=``running`` daemon whose process is DEAD as ``stopped``; and
      2. DROP every STALE pending ``admitting`` reservation (crashed launcher: no pid, missing/old
         ``reserved_ts``) — the launchers count FRESH pending rows as growth headroom, so a stale one
         is a phantom reservation that would otherwise inflate the projection until swept at the next
         launch. Reconcile now converges it directly (2026-07-11 fix: the gate observed reconcile mark
         dead running rows yet ``active_jobs`` stayed high because reconcile ignored the pending store).

    Recovery primitive for the OOM incident (2026-06-25) + the phantom-reservation false-refuse
    (2026-07-11): a machine crash / blind fleet-launch / refused-retry storm leaves the registry
    asserting daemons ``running`` when their processes are gone, or leaves stale ``admitting`` rows.
    The OOM memory guard's custody gate + the admission projection read this registry; stale rows
    mislead both. This reconciles to kernel truth WITHOUT signalling anything (dead processes stay
    dead; stale reservations reserved nothing). Importable + quiet-able so the admission path
    auto-reconciles before EVERY governed launch decision (out-of-band death — SIGKILL/OOM/jetsam/
    machine-sleep/SIGURG-144 — can never write ``recorded=stopped``; the apparatus holds that memory,
    not the operator — the "off is a tracked queue, never a forgotten default" discipline). Fail-open
    + IDEMPOTENT by construction: a no-op that returns 0 when the registry is already accurate, and
    running it twice reconciles nothing the second time.
    """
    rows = _load_registry()
    stale_running = [
        r for r in rows
        if r.get("status") == "running"
        and not (_pid_alive(int(r.get("pid", 0))) and _pgid_alive(int(r.get("pgid", 0)) or int(r.get("pid", 0))))
    ]
    stale_pending_labels = _stale_pending_labels(rows, now_ts=now_ts)
    if not stale_running and not stale_pending_labels:
        if verbose:
            print("[durable-daemon] reconcile: no stale running/pending rows; registry already accurate.")
        return 0

    stale_running_labels = {r.get("label") for r in stale_running}

    def _reconcile(rows_in: list[dict]) -> list[dict]:
        out: list[dict] = []
        for r in rows_in:
            # 2. drop STALE pending reservations (phantom growth headroom otherwise).
            if (isinstance(r, dict) and r.get("status") == _PENDING_STATUS and not r.get("pid")
                    and r.get("label") in stale_pending_labels):
                continue
            # 1. mark DEAD running daemons stopped (re-check liveness under the lock).
            if r.get("label") in stale_running_labels and r.get("status") == "running":
                pid = int(r.get("pid", 0))
                pgid = int(r.get("pgid", 0)) or pid
                if not (_pid_alive(pid) and _pgid_alive(pgid)):
                    r["status"] = "stopped"
                    r["stopped_utc"] = _utc_now_iso()
                    r["stopped_reason"] = "reconcile_dead_process"
            out.append(r)
        return out

    _update_registry_locked(_reconcile)
    total = len(stale_running) + len(stale_pending_labels)
    if verbose:
        print(f"[durable-daemon] reconcile: converged {total} stale row(s) to ground truth "
              f"({len(stale_running)} dead running -> stopped, {len(stale_pending_labels)} stale "
              f"pending reservation(s) dropped):")
        for r in stale_running:
            print(f"  - {r.get('label')} pid={r.get('pid')} (was running, process DEAD)")
        for lbl in stale_pending_labels:
            print(f"  - {lbl} (stale pending reservation, no pid -> dropped)")
    return total


def _do_reconcile() -> int:
    """CLI wrapper: reconcile + print, return rc 0."""
    reconcile_dead_daemons(verbose=True)
    return 0


def _do_status() -> int:
    rows = _load_registry()
    if not rows:
        print("[durable-daemon] registry empty (no daemons registered).")
        return 0
    print(f"[durable-daemon] {len(rows)} registered daemon(s):")
    for r in rows:
        label = r.get("label", "?")
        pid = int(r.get("pid", 0))
        pgid = int(r.get("pgid", 0)) or pid
        recorded = r.get("status", "?")
        live = _pid_alive(pid) and _pgid_alive(pgid)
        live_s = "LIVE" if live else "DEAD"
        cmd = " ".join(r.get("cmd", []))[:80]
        print(
            f"  - {label:<32} pid={pid:<7} pgid={pgid:<7} "
            f"recorded={recorded:<8} actual={live_s}  cmd={cmd}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", help="append-only log for the daemon's stdout+stderr (start mode)")
    ap.add_argument("--label", help="addressable label for the daemon (start mode; required-recommended)")
    ap.add_argument("--stop", metavar="LABEL", help="stop the daemon with this label (process-GROUP kill, no orphan)")
    ap.add_argument("--status", action="store_true", help="list registered daemons + live-check")
    ap.add_argument("--reconcile", action="store_true",
                    help="mark recorded=running daemons whose process is DEAD as stopped (OOM recovery)")
    # OOM launch-preflight (start mode). Defaults conservative: an MLX through-R
    # training arm is projected at 25 GB; the 30 GB free floor is the operator's
    # binding minimum. --skip-mem-preflight is the escape hatch for control-plane
    # / watchdog infra that MUST launch even under memory pressure.
    ap.add_argument("--projected-gb", type=float, default=25.0,
                    help="GB the launched job is projected to need (default 25; MLX through-R arm)")
    ap.add_argument("--min-free-gb", type=float, default=30.0,
                    help="minimum free GB to protect after launch (default 30; operator floor)")
    ap.add_argument("--skip-mem-preflight", action="store_true",
                    help="skip the OOM launch-preflight (ONLY for control-plane/watchdog infra)")
    # Layer-3 per-arm cap: wrap the spawned training arm in tools/safe_run.py so
    # it bounds its OWN process-group RSS (and optionally walltime). Training-arm
    # launches (e.g. the n600 realized-axis witness) MUST set --rss-cap-mb.
    ap.add_argument("--rss-cap-mb", type=int, default=None,
                    help="per-arm process-group RSS cap in MiB (wraps cmd in safe_run; layer 3)")
    ap.add_argument("--walltime-cap-s", type=float, default=None,
                    help="per-arm walltime cap in seconds (with --rss-cap-mb; defaults ~14d if only RSS set)")
    # SYSTEM-aware admission (the P0 crash-prevention HARD gate). Before Popen, refuse to START a job
    # whose projected peak, ADDED to the current system-wide used RAM + the remaining growth of all
    # active jobs, would exceed the adaptive ceiling (total - baseline - margin). This is the SUM-over-
    # 128GB crash fix. It is a PREVENT gate (nonzero exit, starts nothing); the ONLY bypass is an
    # explicit operator-quoted --admission-override-rationale (or --skip-admission-gate for infra).
    ap.add_argument("--projected-peak-gib", type=float, default=None,
                    help="projected PEAK RSS (GiB) of THIS job for the system admission gate + registry "
                         "record (falls back to --projected-gb when unset)")
    ap.add_argument("--priority", type=int, default=None,
                    help="explicit throttle priority for THIS job (higher = shed/pause LAST); recorded "
                         "in the registry so the governor's throttle can rank it")
    ap.add_argument("--skip-admission-gate", action="store_true",
                    help="skip the SYSTEM admission gate (ONLY for control-plane/watchdog/black-box infra)")
    ap.add_argument("--skip-readiness-gate", action="store_true",
                    help="skip the CONFIG-FRESHNESS launch-readiness gate (ONLY for infra / "
                         "non-witness launches / deliberate stale-config replays)")
    ap.add_argument("--readiness-override-rationale", default=None,
                    help="operator-quoted rationale to OVERRIDE a launch-readiness REFUSAL "
                         "(a witness config that skips top decision-table fire-now rungs)")
    ap.add_argument("--admission-override-rationale", default=None,
                    help="operator-quoted rationale to OVERRIDE a system admission REFUSAL (the only "
                         "non-infra bypass; placeholder/empty rejected)")
    ap.add_argument("--skip-blackbox-autostart", action="store_true",
                    help="do NOT auto-start the memory black-box recorder before launching (infra only)")
    # NO-silent-failure liveness verification: after Popen, bound-wait this many
    # seconds and confirm the child survived exec (did not die at exec / exit
    # nonzero) before reporting a healthy detached daemon. 0 disables (instant
    # return, back-compat). Detailed debug + nonzero exit on a dead launch.
    ap.add_argument("--verify-s", type=float, default=3.0,
                    help="seconds to verify the child survived exec before reporting success "
                         "(default 3.0; 0 disables — dead launch -> nonzero + detailed debug)")
    # Reaper immunity (task #525): allocate a controlling PTY via script(1) so a
    # claude/codex-named child is classified as a live terminal session by the
    # fleet claude-code-reaper (all its phases require tty == "??"). Default OFF
    # (byte-identical behavior when unset); codex arms get it via codex_delegate.
    ap.add_argument("--with-pty", action="store_true",
                    help="allocate a controlling pty (script -q) for the detached child — required "
                         "for claude/codex-named long-runners (fleet reaper kills no-TTY ones at 5min)")
    # DECLARED job class (task #525): 'reader' = small control-plane/reader job
    # (codex reader arms, log watchers) admitted on the OOM free-floor preflight
    # ONLY (the SUM-over-RAM training model would refuse a 2GB reader while a
    # 60GiB trainer runs). Fail-closed envelope: requires an explicit
    # --projected-peak-gib <= 4 AND --rss-cap-mb <= 4096 (runtime-enforced).
    ap.add_argument("--job-class", choices=["training", "reader"], default="training",
                    help="admission class: 'training' (default; full SUM-over-RAM admission) or "
                         "'reader' (control-plane/reader lane: preflight-only, requires declared "
                         "--projected-peak-gib <= 4 and --rss-cap-mb <= 4096)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <command> [args...] (start mode)")
    a = ap.parse_args(argv)

    # Exactly one mode.
    if a.status:
        return _do_status()
    if a.reconcile:
        return _do_reconcile()
    if a.stop:
        return _do_stop(a.stop)
    # Default = start.
    if not a.log:
        print("error: start mode requires --log (or use --stop/--status)", file=sys.stderr)
        return 2
    return _do_start(a)


if __name__ == "__main__":
    raise SystemExit(main())
