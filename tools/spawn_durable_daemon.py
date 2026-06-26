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


def _mark_stopped(label: str, *, path=None, lock_path=None) -> list[dict]:
    def _mark(rows: list[dict]) -> list[dict]:
        for r in rows:
            if r.get("label") == label:
                r["status"] = "stopped"
                r["stopped_utc"] = _utc_now_iso()
        return rows

    return _update_registry_locked(_mark, path=path, lock_path=lock_path)


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

    log = open(a.log, "ab", buffering=0)  # noqa: SIM115 — kept open for the child
    devnull = open(os.devnull, "rb")  # noqa: SIM115
    log.write(f"[durable-daemon] launching: {' '.join(cmd)}\n".encode())
    proc = subprocess.Popen(
        cmd,
        stdin=devnull,
        stdout=log,
        stderr=log,
        start_new_session=True,  # setsid() -> new session/pgroup -> survives parent
        close_fds=True,
        cwd=os.getcwd(),
    )

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
    try:
        _register_daemon(record)
    except Exception as exc:  # never let a registry hiccup orphan a live daemon
        print(
            f"[durable-daemon] WARNING: registry write failed ({exc}); daemon "
            f"pid={proc.pid} pgid={pgid} is RUNNING but unregistered — stop it "
            f"manually with: kill -TERM -{pgid}",
            file=sys.stderr,
        )

    print(
        f"[durable-daemon] pid={proc.pid} pgid={pgid} label={label} "
        f"(detached session) log={a.log}"
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
