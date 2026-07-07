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
# Ensure the tools/ dir is importable so the sibling governor / black-box modules resolve regardless
# of the caller's sys.path (the admission gate + black-box auto-start must not silently no-op).
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
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

    signal-0 probe: ESRCH => dead; EPERM => alive (owned by another user). We
    first attempt a non-blocking reap so a zombified own-child is reported as
    dead rather than falsely alive.
    """
    if pid <= 0:
        return False
    if _try_reap(pid):
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True  # exists but owned by another user
        return False
    return True


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
    """True iff cmd is control-plane / protection infra (black-box / memory guard / governor) that MUST
    launch even under memory pressure and is never admission-gated (else it could not protect us)."""
    joined = " ".join(str(t) for t in cmd)
    return any(tok in joined for tok in
               ("memory_blackbox.py", "memory_guard.py", "system_memory_governor.py"))


def _rationale_is_real(text: str | None) -> bool:
    """Reject empty / placeholder override rationales (per Catalog #287 placeholder discipline)."""
    if not text or not text.strip():
        return False
    low = text.strip().lower()
    return low not in {"<rationale>", "<reason>", "placeholder", "tbd", "todo", "n/a"} and len(low) >= 8


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
        f"  total={ctx.snapshot.total_gib:.0f}GiB used={ctx.snapshot.used_gib:.1f}GiB "
        f"available={ctx.snapshot.available_gib:.1f}GiB ceiling={ctx.ceiling.adaptive_ceiling_gib:.1f}GiB "
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

    # OOM launch-preflight: refuse to breach the 30 GB free floor (never OOM the
    # machine again). Runs BEFORE any Popen so a refused launch starts nothing.
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
    pending_label = a.label or f"__admitting__{uuid.uuid4().hex[:10]}"
    pending_written = False
    with _registry_lock():
        # A crashed launcher's stale pending row (> _PENDING_MAX_AGE_S, no pid) must not leak a
        # phantom reservation into this (or any future) admission decision.
        _update_registry_locked(_sweep_stale_pending_rows)
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

    log = open(a.log, "ab", buffering=0)  # noqa: SIM115 — kept open for the child
    devnull = open(os.devnull, "rb")  # noqa: SIM115
    log.write(f"[durable-daemon] launching: {' '.join(cmd)}\n".encode())
    # (#254) the P0 admission gate above JUST passed => this daemon is GOVERNED. Stamp the child
    # env (propagates through `bash launch.sh` -> trainer) so the heavy entrypoint's admission
    # guard passes. Set by stable name (== tac.admission_guard.GOVERNED_MARKER_ENV). A raw launch
    # that never reached this gate lacks the marker and is refused when enforce is armed.
    _child_env = dict(os.environ)
    _child_env["TAC_GOVERNED_ADMISSION"] = "1"
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=devnull,
            stdout=log,
            stderr=log,
            start_new_session=True,  # setsid() -> new session/pgroup -> survives parent
            close_fds=True,
            cwd=os.getcwd(),
            env=_child_env,
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
        with contextlib.suppress(Exception):
            log.write((msg + "\n").encode())
        print(msg, file=sys.stderr)
        with contextlib.suppress(Exception):
            log.close()
            devnull.close()
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

    # The worker is its own session leader, so its pgid == its pid. Read it
    # back rather than assuming, so the registry records the kernel truth.
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = proc.pid  # raced exit; best-effort

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


def _do_reconcile() -> int:
    """Mark every recorded=running daemon whose process is DEAD as stopped.

    Recovery primitive for the OOM incident (2026-06-25): a machine crash /
    blind fleet-launch leaves the registry asserting daemons are ``running``
    when their processes are gone (recorded=running / actual=DEAD). The OOM
    memory guard's custody gate reads this registry; stale running rows would
    waste custody checks and mislead status. This reconciles the registry to
    kernel truth WITHOUT signalling anything (the processes are already dead).
    """
    rows = _load_registry()
    stale = [
        r for r in rows
        if r.get("status") == "running"
        and not (_pid_alive(int(r.get("pid", 0))) and _pgid_alive(int(r.get("pgid", 0)) or int(r.get("pid", 0))))
    ]
    if not stale:
        print("[durable-daemon] reconcile: no stale running rows; registry already accurate.")
        return 0

    stale_labels = {r.get("label") for r in stale}

    def _mark_dead(rows_in: list[dict]) -> list[dict]:
        for r in rows_in:
            if r.get("label") in stale_labels and r.get("status") == "running":
                pid = int(r.get("pid", 0))
                pgid = int(r.get("pgid", 0)) or pid
                if not (_pid_alive(pid) and _pgid_alive(pgid)):
                    r["status"] = "stopped"
                    r["stopped_utc"] = _utc_now_iso()
                    r["stopped_reason"] = "reconcile_dead_process"
        return rows_in

    _update_registry_locked(_mark_dead)
    print(f"[durable-daemon] reconcile: marked {len(stale)} dead daemon(s) as stopped:")
    for r in stale:
        print(f"  - {r.get('label')} pid={r.get('pid')} (was running, process DEAD)")
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
