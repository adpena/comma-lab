#!/usr/bin/env python3
"""Idempotent ensure-up + one-line status for the level-set witness LIVE dashboard.

WHY (the incident this closes, 2026-07-10): a manual dashboard restart launched from a
harness Bash tool DIED (SIGURG / process-group + sandbox teardown of the detached child)
and left the dashboard DOWN with nothing watching it. The robust relaunch that SURVIVED
was the canonical durable path::

    tools/spawn_durable_daemon.py ... -- .venv/bin/python tools/dashboard_server.py --port 8790 --reuse-port

This tool makes that path IDEMPOTENT + AUTOMATABLE so the dashboard can never-go-dead
unattended and auto-picks-up code edits:

  ``ensure-up`` (idempotent, self-healing)
     * healthy + code fresh -> NO-OP (never disrupts a working server).
     * healthy + code STALE  -> zero-downtime durable RELOAD (tools/dashboard_reload.py,
       SO_REUSEPORT overlap) so a code edit auto-applies WITHOUT the fragile manual kill.
     * down / duplicate / unhealthy -> durable RESTART via spawn_durable_daemon.py
       (killpg-safe, start_new_session, VERIFIED-alive) — the canonical path above.
     * ALWAYS verifies :port answers HTTP 200 after acting; a failed restart is LOUD
       (rc != 0), never a silent down.
     * defers to a running tools/dashboard_supervisor.py (its monitor loop self-heals);
       only escalates to a bare restart if the supervisor is ALSO gone.

  ``status``  one line: UP/DOWN · which run it watches · last-update age · code fresh/stale.

The auto-reload staleness check is REAL: the server stamps the mtime of the source THIS
process is running into ``/healthz`` (``code_mtime``); we compare it to the CURRENT
on-disk mtime. NEVER touches the training run (kill signatures match ONLY dashboard
server procs on the port, never ``train_levelset_witness``).

Usage:
  .venv/bin/python tools/dashboard_ctl.py status
  .venv/bin/python tools/dashboard_ctl.py ensure-up
  .venv/bin/python tools/dashboard_ctl.py ensure-up --quiet   # for the SessionStart hook
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# NB: after the sys.path bootstrap above -- import position is deliberate.
from tac import process_liveness  # noqa: E402

_TOOLS = _REPO_ROOT / "tools"
_DEFAULT_PORT = 8790
_TRAINING_SIG = "train_levelset_witness"  # NEVER kill this
_SERVER_SIG = "dashboard_server.py"
_SUPERVISOR_SIG = "dashboard_supervisor.py"
# Stable access-key file the supervisor writes; preserve app-layer gating across a restart
# (disclosure hygiene — a naive relaunch would drop the key and serve the method publicly).
_ACCESS_KEY_FILE = _REPO_ROOT / ".omx" / "tmp" / "dash_levelset_deploy" / ".access_key"
# Source files whose content is baked into the running server (mirror of
# dashboard_server._CODE_SOURCE_FILES — kept in lockstep; a change to any needs a reload).
_CODE_SOURCE_FILES = ("dashboard_server.py", "render_levelset_dashboard.py",
                      "dashboard_flow_client.js", "dashboard_whyhow_client.js")


# ───────────────────────── probes (patchable for tests) ─────────────────────────
def probe_health(port: int, timeout: float = 4.0) -> dict | None:
    """GET /healthz and return the parsed JSON dict, or None if the port is down /
    unreachable / non-200. The dict carries ``code_mtime`` + ``watched`` + liveness."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=timeout) as r:
            if r.getcode() != 200:
                return None
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def _ps_rows() -> list[tuple[int, int, str]]:
    """[(pid, pgid, cmd)] for every process (best-effort; [] on error)."""
    try:
        out = subprocess.run(["ps", "-axww", "-o", "pid=,pgid=,command="],
                             capture_output=True, text=True).stdout
    except Exception:
        return []
    rows: list[tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows


def server_procs(port: int) -> list[tuple[int, int, str]]:
    """Dashboard SERVER procs bound to this port (the uvicorn ``dashboard_server.py --port
    <port>`` children). Excludes the training daemon + the safe_run/reload wrappers so the
    caller counts real server instances (duplicate-detection) and can killpg only servers."""
    needle = f"--port {port}"
    out = []
    for pid, pgid, cmd in _ps_rows():
        if _TRAINING_SIG in cmd:
            continue
        if _SERVER_SIG in cmd and needle in cmd and "safe_run.py" not in cmd and "dashboard_reload.py" not in cmd:
            out.append((pid, pgid, cmd))
    return out


def supervisor_alive() -> bool:
    """True iff a dashboard_supervisor.py monitor daemon is running (its self-heal loop
    owns the server+tunnel, so ensure-up defers to it rather than fighting it).

    BUG FIXED 2026-07-11: the old ``_TRAINING_SIG not in cmd`` exclusion (meant to
    skip the trainer) excluded the SUPERVISOR ITSELF, because the supervisor's own
    argv carries ``--training-sig train_levelset_witness`` as a parameter value —
    so status permanently reported "supervisor gone" while it ran (false-negative).
    A row containing the supervisor script + ``--run`` IS the supervisor; the
    trainer's cmdline never contains dashboard_supervisor.py, so no exclusion is
    needed for aliveness."""
    return any(_SUPERVISOR_SIG in cmd and "--run" in cmd
               for _, _, cmd in _ps_rows())


def disk_code_mtime() -> float:
    """Max mtime over the server's baked-in source files, on disk NOW (0.0 if none)."""
    newest = 0.0
    for name in _CODE_SOURCE_FILES:
        try:
            newest = max(newest, (_TOOLS / name).stat().st_mtime)
        except OSError:
            continue
    return newest


# ───────────────────────── pure decision logic (unit-tested) ─────────────────────────
def is_code_stale(health: dict | None, disk_mtime: float, grace_s: float = 2.0) -> bool:
    """True iff the on-disk source is NEWER than the code the running server started with.

    ``grace_s`` absorbs mtime jitter / a same-second save so a no-op edit never flaps a
    reload. A server exposing NO ``code_mtime`` predates this feature and is treated as
    stale (one-time adoption reload). An invalid / non-positive ``code_mtime`` is False
    (cannot justify a reload)."""
    if not health:
        return False
    if "code_mtime" not in health or health.get("code_mtime") is None:
        # An OLD server that predates the code_mtime field is, by definition, running
        # pre-this-feature code -> a one-time adoption reload picks up the new server.
        return True
    try:
        server_mtime = float(health["code_mtime"])
    except (TypeError, ValueError):
        return False
    if server_mtime <= 0.0:
        return False
    return disk_mtime > (server_mtime + float(grace_s))


def decide_action(health: dict | None, code_stale: bool, n_server_procs: int) -> str:
    """PURE ensure-up decision. Returns one of:

      ``restart`` — down / unreachable, OR duplicate server instances (dedupe).
      ``reload``  — healthy singleton but the code on disk changed since it started.
      ``noop``    — healthy singleton, code fresh (the common, do-nothing case).
    """
    if not health:
        return "restart"
    if n_server_procs > 1:  # multiple servers on the port -> collapse to one fresh instance
        return "restart"
    if code_stale:
        return "reload"
    return "noop"


# ───────────────────────── actions (shell out to the canonical durable tools) ─────────────────────────
def _venv_python() -> str:
    return sys.executable or str(_REPO_ROOT / ".venv" / "bin" / "python")


def _utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _killpg_server_procs(port: int) -> int:
    """SIGTERM (then SIGKILL) the process GROUP of each dashboard SERVER on the port — no
    orphan (the servers are session leaders via spawn_durable_daemon). Never the trainer."""
    procs = server_procs(port)
    for _, pgid, _ in procs:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pgid, signal.SIGTERM)
    if procs:
        time.sleep(1.5)
        for pid, pgid, _ in procs:
            # Canonical liveness (tac.process_liveness) instead of a bare
            # kill(pid, 0).  CHANGED: a survivor we cannot SIGNAL (EPERM) now
            # reads ALIVE and gets the SIGKILL escalation it was previously
            # skipped for; a ZOMBIE now reads DEAD and is correctly NOT
            # escalated (it already exited -- SIGKILLing its group was noise).
            if process_liveness.pid_state(pid) == process_liveness.ALIVE:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(pgid, signal.SIGKILL)
    return len(procs)


def do_restart(port: int, *, verify_s: float = 12.0, quiet: bool = False) -> int:
    """Durable bare restart — the incident's canonical survive-the-harness path. Kills any
    stale/duplicate server procs, launches a fresh durable daemon (killpg-safe,
    start_new_session, control-plane-exempt admission), then VERIFIES 200. LOUD on failure."""
    killed = _killpg_server_procs(port)
    log = _REPO_ROOT / ".omx" / "tmp" / f"dashboard_ctl_restart_{_utc()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # Preserve app-layer access gating across the restart (disclosure hygiene).
    env = dict(os.environ)
    env["DASH_PORT"] = str(port)
    with contextlib.suppress(OSError):
        k = _ACCESS_KEY_FILE.read_text().strip()
        if k:
            env["DASH_ACCESS_KEY"] = k
    cmd = [
        _venv_python(), str(_TOOLS / "spawn_durable_daemon.py"),
        "--label", f"levelset_dash_server_{_utc()}",
        "--log", str(log),
        # The dashboard SERVER is fixed-size observability CONTROL-PLANE (~2.4 GiB,
        # non-growing) — the "guard NEVER kills the control-plane" non-negotiable requires
        # it be always-admissible (mirror of dashboard_reload.py's rationale). The per-arm
        # --rss-cap-mb (real safety) REMAINS; only the growth-reservation gates are skipped.
        "--skip-admission-gate", "--skip-readiness-gate",
        "--skip-blackbox-autostart", "--skip-mem-preflight",
        "--rss-cap-mb", "2500", "--walltime-cap-s", "1209600",
        "--verify-s", "6",
        "--", _venv_python(), str(_TOOLS / "dashboard_server.py"),
        "--port", str(port), "--reuse-port",
    ]
    r = _run(cmd, env=env)
    if not quiet:
        print(f"[dashboard_ctl] restart: killed {killed} stale server proc(s); "
              f"spawn_durable_daemon rc={r.returncode}; log={log}")
    ok = _wait_healthy(port, verify_s)
    if not ok:
        print(f"[dashboard_ctl] ERROR: dashboard did NOT answer 200 on :{port} within "
              f"{verify_s:.0f}s after restart — daemon log tail:\n{_tail(log)}", file=sys.stderr)
        return 1
    if not quiet:
        h = probe_health(port) or {}
        print(f"[dashboard_ctl] restart OK — :{port} healthy (pid={h.get('pid')}, "
              f"watching {h.get('watched')}).")
    return 0


def do_reload(port: int, *, verify_s: float = 30.0, quiet: bool = False) -> int:
    """Zero-downtime durable reload (SO_REUSEPORT overlap) to pick up new server code
    without dropping the tunnel origin. Falls back to a bare restart if the overlap swap
    cannot co-bind (an OLD instance predating --reuse-port)."""
    cmd = [_venv_python(), str(_TOOLS / "dashboard_reload.py"), "--port", str(port)]
    r = _run(cmd)
    if not quiet:
        print(f"[dashboard_ctl] reload (zero-downtime): dashboard_reload rc={r.returncode}")
    if r.returncode != 0:
        # reload could not co-bind (rc 3/4) or failed — fall back to the durable bare restart
        # so a stale-code server is never left running just because the overlap swap balked.
        if not quiet:
            print("[dashboard_ctl] reload did not complete cleanly — escalating to durable restart.")
        return do_restart(port, verify_s=verify_s, quiet=quiet)
    if not _wait_healthy(port, verify_s):
        print(f"[dashboard_ctl] ERROR: :{port} not 200 after reload — escalating to restart.",
              file=sys.stderr)
        return do_restart(port, verify_s=verify_s, quiet=quiet)
    return 0


# ───────────────────────── orchestration ─────────────────────────
def _run(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT), env=env)


def _wait_healthy(port: int, timeout_s: float) -> bool:
    deadline = time.time() + max(0.0, float(timeout_s))
    while True:
        if probe_health(port) is not None:
            return True
        if time.time() >= deadline:
            return False
        time.sleep(1.0)


def _tail(path: Path, n: int = 20) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(log unavailable)"


def ensure_up(port: int, *, code_grace_s: float = 2.0, verify_s: float = 12.0,
              supervisor_grace_s: float = 8.0, quiet: bool = False) -> int:
    """Idempotently ensure a healthy, current dashboard on ``port`` via the durable path.

    Health-first: never disrupts a working, fresh server. A stale-code server is
    zero-downtime reloaded; a down/duplicate one is durably restarted; both verify 200.
    Defers to a running supervisor (its monitor self-heals) unless it is ALSO gone."""
    health = probe_health(port)
    # Down but a supervisor is alive: give ITS monitor a grace window to self-heal before
    # we intervene (avoid two managers racing to restart). Re-probe after the grace.
    if health is None and supervisor_alive():
        if not quiet:
            print(f"[dashboard_ctl] :{port} down but supervisor alive — waiting "
                  f"{supervisor_grace_s:.0f}s for its self-heal…")
        if _wait_healthy(port, supervisor_grace_s):
            health = probe_health(port)

    disk_mtime = disk_code_mtime()
    stale = is_code_stale(health, disk_mtime, code_grace_s)
    action = decide_action(health, stale, len(server_procs(port)))

    if action == "noop":
        if not quiet:
            age = (health or {}).get("last_update_age_s")
            print(f"[dashboard_ctl] :{port} healthy + code fresh — no-op "
                  f"(watching {(health or {}).get('watched')}, last update "
                  f"{_fmt_age(age)} ago).")
        return 0
    if action == "reload":
        if not quiet:
            print(f"[dashboard_ctl] :{port} healthy but code changed on disk since start "
                  f"— zero-downtime reload.")
        return do_reload(port, quiet=quiet)
    # restart
    return do_restart(port, verify_s=verify_s, quiet=quiet)


# ───────────────────────── status ─────────────────────────
def _fmt_age(s) -> str:
    if s is None:
        return "?"
    try:
        s = max(0, int(float(s)))
    except (TypeError, ValueError):
        return "?"
    if s < 90:
        return f"{s}s"
    m = s / 60.0
    return f"{m:.1f}m" if m < 90 else f"{m / 60.0:.1f}h"


def cmd_status(port: int) -> int:
    """One-line health + which run + last-update age + code freshness + proc count."""
    health = probe_health(port)
    procs = server_procs(port)
    sup = supervisor_alive()
    tunnel = any("cloudflared" in c and "tunnel" in c for _, _, c in _ps_rows())
    if health is None:
        print(f"dashboard :{port} DOWN · {len(procs)} server proc(s) · "
              f"supervisor {'alive' if sup else 'gone'} · tunnel {'up' if tunnel else 'DOWN'} "
              f"· run `dashboard_ctl.py ensure-up`")
        return 0
    stale = is_code_stale(health, disk_code_mtime())
    watched = health.get("watched") or "?"
    wdir = health.get("watched_dir")
    run = f"{watched}" + (f" @ {Path(wdir).name}" if wdir else "")
    age = _fmt_age(health.get("last_update_age_s"))
    ep = health.get("last_epoch")
    nxt = health.get("next_epoch")
    print(
        f"dashboard :{port} UP · watching {run}"
        f"{f' · ep{ep}' if ep is not None else ''}"
        f" · last update {age} ago"
        f"{f' · next verdict @ ep{nxt}' if nxt is not None else ''}"
        f" · {health.get('n_points', 0)} pts"
        f" · code {'STALE (edit pending reload)' if stale else 'fresh'}"
        f" · {len(procs)} proc · supervisor {'alive' if sup else 'gone'}"
        f" · tunnel {'up' if tunnel else 'DOWN'}"
        f" · up since {health.get('started_utc', '?')}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="status",
                    choices=["status", "ensure-up", "ensure"],
                    help="status (default) | ensure-up (idempotent durable ensure)")
    ap.add_argument("--ensure", action="store_true",
                    help="alias for `ensure-up` (idempotent durable ensure)")
    ap.add_argument("--port", type=int, default=_DEFAULT_PORT)
    ap.add_argument("--code-grace-s", type=float, default=2.0,
                    help="mtime jitter absorbed before a code edit triggers a reload")
    ap.add_argument("--verify-s", type=float, default=12.0,
                    help="seconds to wait for :port to answer 200 after a restart")
    ap.add_argument("--quiet", action="store_true", help="minimal output (for hooks/cron)")
    a = ap.parse_args(argv)

    if a.ensure or a.mode in ("ensure-up", "ensure"):
        return ensure_up(a.port, code_grace_s=a.code_grace_s, verify_s=a.verify_s, quiet=a.quiet)
    return cmd_status(a.port)


if __name__ == "__main__":
    raise SystemExit(main())
