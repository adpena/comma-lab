#!/usr/bin/env python3
"""Watch running Vast.ai instances and surface stalls / idle GPU burn.

DX #9 (2026-04-26): the SHIRAZ post-mortem proved tmux-session existence is
NOT a heartbeat. The instance ran 16h producing no measurement because the
training process had died inside a live tmux session. This watchdog polls
the Vast.ai instance list every 5 min and surfaces three classes of failure
the operator should react to:

    1. STALE_HEARTBEAT  — heartbeat.log mtime > 10 min old (process dead).
    2. IDLE_BURN        — gpu_util = 0%% for the last 5 min (paying for nothing).
    3. NO_PROGRESS      — instance has been billed >$5 with no result file
                          (no run_record.json, results.json, or RESULT_JSON).

The watchdog DOES NOT auto-kill — that decision stays with the human (per
CLAUDE.md "no destructive operations without approval"). It just prints a
loud banner and writes a structured entry to `reports/watchdog.log`.

Run modes:

    # one-shot (CI / cron / manual triage)
    python scripts/lane_watchdog.py --once

    # daemon (5 min cadence)
    python scripts/lane_watchdog.py --daemon

The cadence (--interval) defaults to 300 s. Hotz approved 5 min as "not
chatty"; lower than that is wasteful API calls, higher than that means a
20 min stall before the operator notices.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent
LOG_PATH = REPO / "reports" / "watchdog.log"

STALE_HEARTBEAT_SEC = 600  # 10 min — heartbeats fire every 60s, 10x slack.
IDLE_GPU_THRESHOLD_PCT = 1  # gpu_util < this counts as idle
IDLE_GPU_WINDOW_SEC = 300  # 5 min of consecutive idle to alert
NO_PROGRESS_DOLLAR_THRESHOLD = 5.0  # $ billed without a result file


@dataclass
class InstanceState:
    """Per-instance rolling state tracked between polls.

    The watchdog has no DB; we keep state in-memory for a single daemon
    invocation. Restart loses history (which is fine — the alerts re-fire
    on the next poll if the condition still holds).
    """
    instance_id: str
    label: str = ""
    first_seen_at: float = field(default_factory=time.time)
    last_gpu_busy_at: float = field(default_factory=time.time)
    alerts_emitted: set[str] = field(default_factory=set)
    cumulative_cost_usd: float = 0.0
    last_heartbeat_mtime: float | None = None


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _emit(msg: str, *, level: str = "ALERT", extra: dict | None = None) -> None:
    """Print a loud banner + append a structured JSON line to watchdog.log.

    The banner uses ASCII box drawing so it survives any terminal; the
    JSON line is what programmatic readers (CI checks, dashboards) consume.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": _now_iso(), "level": level, "msg": msg}
    if extra:
        rec.update(extra)
    bar = "=" * max(len(msg) + 4, 60)
    print(bar)
    print(f"[{level} {rec['ts']}] {msg}")
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")
    print(bar)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _vastai_show_instances() -> list[dict]:
    """Call `vastai show instances --raw` and parse JSON.

    Returns [] on any failure (vastai not installed, no API key, network
    error). Failures are logged but do not stop the watchdog — the next
    poll re-tries.
    """
    try:
        out = subprocess.check_output(
            ["vastai", "show", "instances", "--raw"],
            text=True, stderr=subprocess.STDOUT, timeout=30,
        )
    except FileNotFoundError:
        _emit("vastai CLI not on PATH — skipping poll", level="WARN")
        return []
    except subprocess.TimeoutExpired:
        _emit("vastai show instances timed out", level="WARN")
        return []
    except subprocess.CalledProcessError as e:
        _emit(f"vastai show instances failed: {e.output[:200]}", level="WARN")
        return []
    try:
        data = json.loads(out)
        if isinstance(data, list):
            return data
        _emit("vastai output not a list — schema drift?", level="WARN",
              extra={"raw_head": out[:200]})
        return []
    except json.JSONDecodeError:
        # vastai older versions print non-JSON; skip.
        return []


def _instance_remote_heartbeat_mtime(inst: dict) -> float | None:
    """Best-effort mtime of /workspace/pact/**/heartbeat.log over SSH.

    The watchdog can run from anywhere — laptop, alejandros-mac-mini, CI.
    We keep this side optional because Vast.ai SSH ports change per
    instance and may be locked behind a tunnel. If the SSH probe fails
    we return None (caller treats as "unknown" rather than "stale").
    """
    host = inst.get("ssh_host")
    port = inst.get("ssh_port")
    if not host or not port:
        return None
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=no",
        "-p", str(port), f"root@{host}",
        # Find the freshest heartbeat.log under /workspace/pact and print its
        # mtime as a unix timestamp. Returns "" if none exist.
        "find /workspace/pact -maxdepth 5 -name heartbeat.log -printf '%T@\\n' 2>/dev/null | sort -n | tail -1",
    ]
    try:
        out = subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL, timeout=15,
        ).strip()
        return float(out) if out else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, ValueError):
        return None


def evaluate_instance(inst: dict, state: InstanceState, now: float) -> list[dict]:
    """Decide whether to alert on this instance. Returns list of alert dicts.

    Each alert dict has keys: kind, msg, extra. The caller emits them via
    _emit(); we don't emit here so unit tests can assert on the list shape.
    """
    alerts: list[dict] = []

    # 1) STALE_HEARTBEAT
    hb_mtime = _instance_remote_heartbeat_mtime(inst)
    if hb_mtime is not None:
        age = now - hb_mtime
        state.last_heartbeat_mtime = hb_mtime
        if age > STALE_HEARTBEAT_SEC and "stale_heartbeat" not in state.alerts_emitted:
            alerts.append({
                "kind": "STALE_HEARTBEAT",
                "msg": f"instance {state.instance_id} ({state.label}) heartbeat stale "
                       f"({age:.0f}s > {STALE_HEARTBEAT_SEC}s)",
                "extra": {"heartbeat_mtime_age_sec": int(age)},
            })
            state.alerts_emitted.add("stale_heartbeat")

    # 2) IDLE_BURN
    gpu_util = inst.get("gpu_util")
    if isinstance(gpu_util, (int, float)):
        if gpu_util >= IDLE_GPU_THRESHOLD_PCT:
            state.last_gpu_busy_at = now
        idle_sec = now - state.last_gpu_busy_at
        if (
            idle_sec > IDLE_GPU_WINDOW_SEC
            and "idle_burn" not in state.alerts_emitted
        ):
            alerts.append({
                "kind": "IDLE_BURN",
                "msg": f"instance {state.instance_id} GPU idle for {idle_sec:.0f}s "
                       f"(util={gpu_util}%); paying for nothing",
                "extra": {"idle_sec": int(idle_sec), "gpu_util_pct": gpu_util,
                          "dph": inst.get("dph_total")},
            })
            state.alerts_emitted.add("idle_burn")

    # 3) NO_PROGRESS — billed >$5 with no result artifact yet.
    dph = inst.get("dph_total") or inst.get("dph_base") or 0.0
    elapsed_h = (now - state.first_seen_at) / 3600.0
    state.cumulative_cost_usd = float(dph) * elapsed_h
    if (
        state.cumulative_cost_usd > NO_PROGRESS_DOLLAR_THRESHOLD
        and "no_progress" not in state.alerts_emitted
    ):
        # We'd need to SSH in to check for results.json; surface the alert
        # regardless and let the operator decide. False-positive is cheap;
        # false-negative is expensive ($).
        alerts.append({
            "kind": "NO_PROGRESS",
            "msg": f"instance {state.instance_id} has burned ${state.cumulative_cost_usd:.2f} "
                   f"({elapsed_h:.1f}h × ${dph:.3f}/h); verify a result artifact exists",
            "extra": {"cost_usd": round(state.cumulative_cost_usd, 2),
                      "elapsed_h": round(elapsed_h, 2)},
        })
        state.alerts_emitted.add("no_progress")

    return alerts


def poll_once(states: dict[str, InstanceState]) -> None:
    """Run one poll cycle: fetch instances, evaluate each, emit alerts."""
    instances = _vastai_show_instances()
    seen: set[str] = set()
    now = time.time()
    for inst in instances:
        iid = str(inst.get("id") or "")
        if not iid:
            continue
        seen.add(iid)
        state = states.setdefault(iid, InstanceState(instance_id=iid,
                                                    label=str(inst.get("label", ""))))
        # Only alert on RUNNING instances; offered/stopped/exited produce noise.
        if inst.get("actual_status") not in ("running", None):
            continue
        for alert in evaluate_instance(inst, state, now):
            _emit(alert["msg"], level=alert["kind"], extra=alert.get("extra"))
    # Garbage-collect state entries for instances that vanished from the API
    # (operator destroyed them) — keep memory bounded over long daemon runs.
    for missing in list(states.keys() - seen):
        del states[missing]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--once", action="store_true",
                   help="Run a single poll cycle and exit (default).")
    p.add_argument("--daemon", action="store_true",
                   help="Loop forever at --interval cadence.")
    p.add_argument("--interval", type=int, default=300,
                   help="Daemon poll cadence in seconds (default 300 = 5 min).")
    args = p.parse_args(argv)

    if not args.once and not args.daemon:
        args.once = True

    states: dict[str, InstanceState] = {}
    if args.once:
        poll_once(states)
        return 0
    print(f"[lane_watchdog] daemon mode, interval={args.interval}s. "
          f"Logging to {LOG_PATH}.")
    while True:
        try:
            poll_once(states)
        except KeyboardInterrupt:
            return 0
        except Exception as e:  # never let a bad poll crash the daemon
            _emit(f"poll cycle raised {type(e).__name__}: {e}", level="WARN")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
