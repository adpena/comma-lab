#!/usr/bin/env python3
"""Singleton launcher for the level-set witness LIVE dashboard — ONLY ONE at a time.

Extincts the stale-dashboard bug class (operator 2026-06-30): a prior dashboard
daemon left running on an OLD run log/glob silently misleads monitoring. This
enforces EXACTLY ONE dashboard set for a given (run-dir, port):

    {render_levelset_dashboard --watch}  +  {http.server :port}  +  {cloudflared tunnel :port}

Default = keep-if-healthy (idempotent): if exactly one healthy set is already
serving this run-dir+port, print its URL and exit WITHOUT disruption. Otherwise
(zero / multiple / stale / unhealthy) it KILLS every dashboard proc and starts
exactly one. ``--force`` always replaces. ``--status`` lists, never changes.

NEVER kills the training daemon — the kill signatures match ONLY the three
dashboard process kinds (render_levelset_dashboard / http.server:port /
cloudflared tunnel:port), never ``train_levelset_witness_*``.

Each daemon is launched via the canonical ``tools/spawn_durable_daemon.py``
(killpg-based, no orphan — CLAUDE.md daemon discipline).

Usage:
  .venv/bin/python tools/dashboard_up.py --run-dir experiments/results/levelset_openpilot_seeded_n200_DEPLOY
  .venv/bin/python tools/dashboard_up.py --run-dir ... --force   # always replace
  .venv/bin/python tools/dashboard_up.py --status                # list, no change
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

RENDER_SIG = "render_levelset_dashboard"
TRAINING_SIG = "train_levelset_witness"  # NEVER kill this
URL_RE = re.compile(r"https://[a-z0-9.-]+\.trycloudflare\.com")


def _ps_rows() -> list[tuple[int, str]]:
    out = subprocess.run(
        ["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True
    ).stdout
    rows: list[tuple[int, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str, _, cmd = line.partition(" ")
        try:
            rows.append((int(pid_str), cmd))
        except ValueError:
            continue
    return rows


def _dash_procs(port: int) -> list[tuple[int, str, str]]:
    """Return (pid, cmd, kind) for every dashboard proc on this port. Excludes training."""
    port_s = str(port)
    found: list[tuple[int, str, str]] = []
    for pid, cmd in _ps_rows():
        if TRAINING_SIG in cmd:
            continue  # never the training daemon
        if RENDER_SIG in cmd:
            kind = "render"
        elif "http.server" in cmd and port_s in cmd:
            kind = "httpd"
        elif "cloudflared" in cmd and "tunnel" in cmd and port_s in cmd:
            kind = "tunnel"
        else:
            continue
        found.append((pid, cmd, kind))
    return found


def _url(cf_log: str) -> str | None:
    try:
        m = URL_RE.search(open(cf_log, encoding="utf-8", errors="replace").read())
        return m.group(0) if m else None
    except OSError:
        return None


def _httpd_ok(port: int) -> bool:
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=4).getcode() == 200
    except Exception:
        return False


def _kill(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    time.sleep(2)
    # SIGKILL stragglers
    for pid in pids:
        try:
            os.kill(pid, 0)  # alive?
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass


def _spawn(label: str, log: str, proj_gb: float, rss_mb: int, cmd: list[str]) -> int:
    argv = [
        sys.executable, "tools/spawn_durable_daemon.py",
        "--label", label, "--log", log,
        "--projected-gb", str(proj_gb), "--min-free-gb", "10",
        "--rss-cap-mb", str(rss_mb), "--walltime-cap-s", "288000", "--",
    ] + cmd
    return subprocess.run(argv, capture_output=True, text=True).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Singleton level-set dashboard launcher (one at a time).")
    ap.add_argument("--run-dir", help="DEPLOY run dir; the dashboard follows <run-dir>/*.log")
    ap.add_argument("--dash-dir", default=".omx/tmp/dash_levelset_deploy")
    ap.add_argument("--port", type=int, default=8789)
    ap.add_argument("--goal-dseg", type=float, default=0.00092)
    ap.add_argument("--tau", type=int, default=300)
    ap.add_argument("--l7", type=int, default=600)
    ap.add_argument("--refresh-seconds", type=int, default=30)
    ap.add_argument("--force", action="store_true", help="always replace, even if healthy")
    ap.add_argument("--status", action="store_true", help="list dashboard procs + URL, no change")
    a = ap.parse_args()

    cf_log = os.path.join(a.dash_dir, "cloudflared.log")
    existing = _dash_procs(a.port)
    render_n = sum(1 for _, _, k in existing if k == "render")

    if a.status:
        print(f"dashboard procs on :{a.port} -> {len(existing)} (render={render_n})")
        for pid, cmd, kind in existing:
            print(f"  {kind:7s} pid={pid}: {cmd[:110]}")
        print("httpd_ok:", _httpd_ok(a.port), "| url:", _url(cf_log))
        return 0

    # keep-if-healthy: exactly one render + httpd 200 + a captured URL
    if not a.force and render_n == 1 and _httpd_ok(a.port) and _url(cf_log):
        print("DASHBOARD already singleton + healthy — keeping it (no disruption).")
        print("CLOUDFLARE_URL:", _url(cf_log))
        return 0

    if existing:
        print(f"enforcing singleton: killing {len(existing)} existing dashboard proc(s) "
              f"(render={render_n}; stale/duplicate/unhealthy)...")
        _kill([pid for pid, _, _ in existing])

    if not a.run_dir:
        print("ERROR: --run-dir required to (re)start the dashboard.", file=sys.stderr)
        return 2

    os.makedirs(a.dash_dir, exist_ok=True)
    idx = os.path.join(a.dash_dir, "index.html")
    rc1 = _spawn(
        "deploy_dash", os.path.join(a.dash_dir, "dashboard_watch.log"), 2.0, 3000,
        [sys.executable, "tools/render_levelset_dashboard.py",
         "--log-glob", os.path.join(a.run_dir, "*.log"), "--out", idx,
         "--watch", "--refresh-seconds", str(a.refresh_seconds),
         "--tau", str(a.tau), "--l7", str(a.l7), "--goal-dseg", str(a.goal_dseg)],
    )
    rc2 = _spawn(
        "deploy_dash_http", os.path.join(a.dash_dir, "httpd.log"), 1.0, 1000,
        [sys.executable, "-m", "http.server", str(a.port), "--bind", "127.0.0.1",
         "--directory", a.dash_dir],
    )
    rc3 = _spawn(
        "deploy_dash_tunnel", cf_log, 1.0, 1000,
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{a.port}"],
    )
    print(f"spawned: render rc={rc1} httpd rc={rc2} tunnel rc={rc3}")

    # capture URL (cloudflared takes a few seconds)
    url = None
    for _ in range(15):
        time.sleep(2)
        url = _url(cf_log)
        if url:
            break
    print("index.html:", os.path.getsize(idx) if os.path.exists(idx) else "(pending first verdict)")
    print("httpd_ok:", _httpd_ok(a.port))
    print("CLOUDFLARE_URL:", url or "(not captured yet — re-run --status in ~15s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
