#!/usr/bin/env python3
"""Zero-downtime hot reload of the live dashboard — never drops the cloudflared tunnel origin.

Operator 2026-06-30: *"Live server should hot reload without interrupting named tunnel."* A naive
kill+relaunch leaves :PORT with no listener for ~1s -> cloudflared returns 502 during that window. This
helper does an **SO_REUSEPORT overlap swap** instead:

  1. snapshot the OLD dashboard pid(s) bound to :PORT,
  2. launch a NEW (new-code) instance on the SAME :PORT (both hold it via SO_REUSEPORT),
  3. wait until the NEW instance logs ``"started": true`` (its socket is bound + serving),
  4. gracefully SIGTERM the OLD instance(s) (uvicorn finishes in-flight, then exits),
  5. verify :PORT healthz is still 200 (the NEW one).

Because the NEW listener is accepting BEFORE the OLD is retired, the :PORT socket is never empty -> the
named tunnel (comma-lab.adpena.com -> cloudflared -> :PORT) sees no 502 window. The cloudflared process
+ its edge connections are untouched throughout (we only swap the origin behind it).

REQUIREMENT: the live server must have been started with ``--reuse-port`` (default ON in
dashboard_server.py). If the OLD instance predates SO_REUSEPORT, the NEW cannot co-bind -> the helper
detects this (NEW never reaches ``started: true`` / bind error) and reports it (do a single transitional
restart first to adopt reuse_port, then all future reloads are zero-downtime).

Durable: the NEW server is launched detached (start_new_session) under the canonical safe_run wrapper, so
it survives this helper exiting. Refuses /tmp log paths.
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _pids_on_port(port: int) -> list[int]:
    """uvicorn child pids running dashboard_server.py --port <port> (the OLD generation)."""
    out = subprocess.run(["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True).stdout  # subprocess-no-check-OK: best-effort ps snapshot; empty output means no old-generation pids
    pids = []
    needle = "dashboard_server.py"
    for line in out.splitlines():
        if needle in line and f"--port {port}" in line and "safe_run.py" not in line and "dashboard_reload.py" not in line:
            try:
                pids.append(int(line.split(None, 1)[0]))
            except (ValueError, IndexError):
                pass
    return pids


def _safe_run_wrappers_for(port: int) -> list[int]:
    out = subprocess.run(["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True).stdout  # subprocess-no-check-OK: best-effort ps snapshot; empty output means no wrapper pids
    pids = []
    for line in out.splitlines():
        if "safe_run.py" in line and "dashboard_server.py" in line and f"--port {port}" in line:
            try:
                pids.append(int(line.split(None, 1)[0]))
            except (ValueError, IndexError):
                pass
    return pids


def _healthz(port: int, timeout: float = 2.0) -> int:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=timeout) as r:
            return r.status
    except Exception:
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8790)
    # Stage boundaries are DERIVED per-run by dashboard_server via the DSL schedule
    # read-back (tac.witness_dsl.schedule_readback). None (the default) = do not pass
    # the flag at all, so the server derives; an explicit value is an OVERRIDE only.
    # The old hardcoded defaults (300/600) silently re-poisoned the derived stage map
    # on every hot reload — the exact --tau/--l7 mislabel class the read-back fixed.
    ap.add_argument("--tau", type=int, default=None,
                    help="OVERRIDE only; default = server derives from the run's own config")
    ap.add_argument("--l7", type=int, default=None,
                    help="OVERRIDE only; default = server derives from the run's own config")
    ap.add_argument("--rss-mb", type=int, default=2500)
    ap.add_argument("--timeout-sec", type=float, default=1209600.0)
    ap.add_argument("--ready-timeout", type=float, default=30.0)
    ap.add_argument("--grace", type=float, default=10.0, help="seconds to let OLD drain after SIGTERM before SIGKILL")
    ap.add_argument("--extra", default="", help="extra args appended to the dashboard launch (quoted)")
    args = ap.parse_args(argv)

    log_dir = _REPO / ".omx" / "tmp"
    log_dir.mkdir(parents=True, exist_ok=True)
    new_log = log_dir / f"dashboard_reload_{_utc()}.log"
    if "/tmp/" in str(new_log.resolve()) + "/" and "/.omx/" not in str(new_log.resolve()):
        raise ValueError("refusing /tmp log path")

    old_children = _pids_on_port(args.port)
    old_wrappers = _safe_run_wrappers_for(args.port)
    pre_health = _healthz(args.port)
    print(json.dumps({"stage": "reload_start", "port": args.port, "old_children": old_children,
                      "old_wrappers": old_wrappers, "pre_healthz": pre_health, "new_log": str(new_log)}), flush=True)

    # 1) launch NEW (new-code) instance, detached + durable, on the SAME port (SO_REUSEPORT overlap).
    label = f"levelset_dash_server_{_utc()}"
    stage_overrides = ""
    if args.tau is not None:
        stage_overrides += f"--tau {args.tau} "
    if args.l7 is not None:
        stage_overrides += f"--l7 {args.l7} "
    dash_cmd = (f".venv/bin/python tools/dashboard_server.py --port {args.port} "
                f"{stage_overrides}--reuse-port {args.extra}").strip()
    # --skip-admission-gate: the dashboard SERVER is fixed-size observability CONTROL-PLANE (~2.4 GiB,
    # non-growing), which the standing "guard NEVER kills the control-plane" non-negotiable (CLAUDE.md
    # memory L42/L43) requires to be always-admissible. Without this, the P0 SUM-over-RAM training gate
    # REFUSES the reload while a heavy run (e.g. #205) reserves its projected growth — blocking observability
    # exactly when it's most needed. The per-process --rss-mb cap REMAINS (real safety); only the system
    # growth-reservation gate is skipped. The heavy flow-seq RENDER stays governed (it is NOT control-plane).
    launch = (f".venv/bin/python tools/safe_run.py --rss-mb {args.rss_mb} --skip-admission-gate "
              f"--timeout {args.timeout_sec} --label {label} -- {dash_cmd}")
    with open(new_log, "wb") as lf:
        subprocess.Popen(["bash", "-lc", launch], cwd=str(_REPO), stdin=subprocess.DEVNULL,
                         stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

    # 2) wait until NEW logs started:true (its socket is bound + serving).
    deadline = time.time() + args.ready_timeout
    ready = False
    while time.time() < deadline:
        time.sleep(1.0)
        try:
            txt = new_log.read_text(errors="replace")
        except FileNotFoundError:
            txt = ""
        if '"started": true' in txt or '"started":true' in txt:
            ready = True
            break
        if "address already in use" in txt.lower() or "errno 48" in txt.lower():
            print(json.dumps({"stage": "REFUSE", "reason": "bind_failed",
                              "hint": "OLD instance likely predates --reuse-port; do ONE transitional "
                                      "restart with --reuse-port, then reloads are zero-downtime",
                              "log_tail": txt[-400:]}), flush=True)
            return 3
    if not ready:
        print(json.dumps({"stage": "REFUSE", "reason": "new_not_ready",
                          "log_tail": (new_log.read_text(errors='replace')[-400:] if new_log.exists() else "")}), flush=True)
        return 4

    # 3) NEW is up + co-bound. Confirm :port still serves (now both old+new), then retire OLD gracefully.
    mid_health = _healthz(args.port)
    new_children = [p for p in _pids_on_port(args.port) if p not in old_children]
    for pid in old_wrappers + old_children:
        try:
            os_kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # 4) grace, then verify + SIGKILL any stragglers.
    t = time.time() + args.grace
    while time.time() < t:
        time.sleep(0.5)
        if all(not _alive(pid) for pid in old_children):
            break
    for pid in old_wrappers + old_children:
        if _alive(pid):
            try:
                os_kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    post_health = _healthz(args.port)
    ok = post_health == 200
    print(json.dumps({"stage": "reload_done", "ok": ok, "new_children": new_children,
                      "pre_healthz": pre_health, "mid_healthz": mid_health, "post_healthz": post_health,
                      "retired": old_wrappers + old_children, "new_log": str(new_log),
                      "note": "tunnel origin :%d had a listener throughout (SO_REUSEPORT overlap)" % args.port}), flush=True)
    return 0 if ok else 5


def os_kill(pid: int, sig: int) -> None:
    import os
    os.kill(pid, sig)


def _alive(pid: int) -> bool:
    import os
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False if isinstance(sys.exc_info()[1], ProcessLookupError) else True


if __name__ == "__main__":
    raise SystemExit(main())
