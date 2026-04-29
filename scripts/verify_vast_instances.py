#!/usr/bin/env python3
"""Per-instance health verify for Vast.ai active fleet.

For each registered instance in `.omx/state/vastai_active_instances.json`:
  1. Confirm Vast.ai still reports it alive
  2. SSH in (with timeout) and check heartbeat.log freshness + GPU activity
  3. Classify: HEALTHY / IDLE / CRASHED / UNREACHABLE / SETUP / GONE

Usage:
    python scripts/verify_vast_instances.py
    python scripts/verify_vast_instances.py --auto-destroy-stale \\
        --stale-minutes 30 --setup-stale-minutes 90

Emits a JSON report to stdout + (optional) destroys CRASHED/IDLE instances
older than the threshold AND SETUP instances stuck > --setup-stale-minutes
(R31 cross-cutting fix: SETUP that's TRULY hung — setup_full.sh deadlocked,
never writes heartbeat — would otherwise accrue cost forever, since the
existing --stale-minutes only fires on IDLE).

State tracking for SETUP-first-seen lives in
``.omx/state/instance_setup_first_seen.json`` so we know when each instance
was first observed in SETUP and can age it past --setup-stale-minutes.

Reference memories:
  - feedback_vastai_launch_returns_success_before_lane_starts
  - feedback_vastai_cost_paranoia
  - feedback_canonical_remote_bootstraps
  - feedback_setup_stuck_cost_leak_FIXED_20260428
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKER_PATH = REPO_ROOT / ".omx/state/vastai_active_instances.json"
SETUP_FIRST_SEEN_PATH = REPO_ROOT / ".omx/state/instance_setup_first_seen.json"
VASTAI_BIN = REPO_ROOT / ".venv/bin/vastai"
SSH_BASE = [
    "ssh",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=10",
    "-o", "LogLevel=ERROR",
]


def _load_setup_first_seen() -> dict[str, float]:
    """Read the SETUP-first-seen tracker.

    Map of instance_id (str) → unix timestamp (float) of first SETUP
    observation. Used to age SETUP instances past --setup-stale-minutes.
    """
    if not SETUP_FIRST_SEEN_PATH.exists():
        return {}
    try:
        data = json.loads(SETUP_FIRST_SEEN_PATH.read_text())
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return {}


def _save_setup_first_seen(data: dict[str, float]) -> None:
    SETUP_FIRST_SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETUP_FIRST_SEEN_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


@dataclass
class InstanceHealth:
    instance_id: str
    label: str
    classification: str  # HEALTHY / IDLE / CRASHED / UNREACHABLE / SETUP / GONE
    last_heartbeat_age_minutes: Optional[float]
    gpu_util_pct: Optional[float]
    ssh_host: Optional[str]
    ssh_port: Optional[int]
    crash_signal: Optional[str]
    notes: str
    # R31 cross-cutting: SETUP-stuck cost-leak detection. age_in_setup
    # is None when the instance is NOT classified SETUP. When SETUP, it
    # is the minutes since this instance was first observed in SETUP
    # (read/written to .omx/state/instance_setup_first_seen.json).
    setup_age_minutes: Optional[float] = None


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -1, "", f"EXCEPTION: {e}"


def _vast_show_instance(instance_id: str) -> Optional[dict]:
    rc, out, err = _run([str(VASTAI_BIN), "show", "instance", instance_id, "--raw"])
    if rc != 0:
        return None
    try:
        d = json.loads(out)
        return d[0] if isinstance(d, list) else d
    except (json.JSONDecodeError, IndexError):
        return None


def _ssh_check_heartbeat(host: str, port: int) -> tuple[Optional[float], Optional[str]]:
    """Returns (heartbeat_age_minutes, crash_signal_or_None).

    Crash signal comes from grep'ing the run.log + train.log for common
    Python tracebacks / FATAL markers.
    """
    cmd = SSH_BASE + ["-p", str(port), f"root@{host}", (
        "find /workspace/pact -name 'heartbeat.log' -printf '%T@\\n' 2>/dev/null | sort -n | tail -1; "
        "echo '---CRASH_SIG---'; "
        "for f in $(find /workspace/pact -name 'run.log' -o -name 'train.log' -o -name 'auth_eval.log' 2>/dev/null); do "
        "  grep -E 'Traceback|FATAL|CUDA_ERROR|RuntimeError|out of memory' \"$f\" | tail -3; "
        "done; echo '---END---'"
    )]
    rc, out, err = _run(cmd, timeout=20)
    if rc != 0:
        return None, f"SSH_FAILED: {err.strip()[:80]}"
    parts = out.split("---CRASH_SIG---")
    if len(parts) < 2:
        return None, None
    hb_section, crash_section = parts[0].strip(), parts[1].split("---END---")[0].strip()
    age_minutes = None
    if hb_section:
        try:
            mtime = float(hb_section.splitlines()[-1].strip())
            age_minutes = (datetime.now(timezone.utc).timestamp() - mtime) / 60.0
        except (ValueError, IndexError):
            pass
    crash_signal = None
    if crash_section:
        crash_signal = crash_section.splitlines()[0][:120]
    return age_minutes, crash_signal


def classify(
    age_min: Optional[float], gpu_util: Optional[float],
    crash_signal: Optional[str], stale_minutes: float,
    ssh_succeeded: bool = True,
) -> str:
    """Classify instance health. Distinguishes:
    - SSH_FAILED (genuinely unreachable) → UNREACHABLE
    - SSH succeeded but heartbeat absent (lane still in setup_full.sh) → SETUP
    - Heartbeat fresh + GPU active → HEALTHY
    - Heartbeat stale > stale_minutes → IDLE
    - GPU 0% > 20min after first heartbeat → IDLE (genuinely stuck)
    - Crash signal in logs → CRASHED
    """
    if crash_signal and crash_signal.startswith("SSH_FAILED"):
        return "UNREACHABLE"
    if crash_signal and crash_signal not in (None, ""):
        return "CRASHED"
    if age_min is None:
        if ssh_succeeded:
            # SSH OK but no heartbeat yet — lane in early setup_full.sh
            return "SETUP"
        return "UNREACHABLE"
    if age_min > stale_minutes:
        return "IDLE"
    if gpu_util is not None and gpu_util < 5.0 and age_min > 20.0:
        # GPU util 0% with heartbeat > 20 min indicates genuinely stuck
        # (DALI install + setup takes ~10 min; 20 min gives safety margin).
        # Lower threshold (5 min) caused false-positive destroys during setup.
        return "IDLE"
    return "HEALTHY"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--auto-destroy-stale", action="store_true",
                   help=(
                       "Destroy IDLE/CRASHED instances older than "
                       "--stale-minutes AND SETUP instances stuck longer "
                       "than --setup-stale-minutes (R31 SETUP-stuck "
                       "cost-leak fix)."
                   ))
    p.add_argument("--stale-minutes", type=float, default=30.0,
                   help="Heartbeat older than this is IDLE (default: 30)")
    p.add_argument("--setup-stale-minutes", type=float, default=90.0,
                   help=(
                       "SETUP classification older than this counts as "
                       "stuck-in-setup (default: 90). DALI install + "
                       "Stage 0..N typically finishes within 15 min on a "
                       "good host; 90 min is well past max install time, "
                       "so anything beyond is a TRULY hung setup_full.sh "
                       "deadlock that would otherwise accrue cost forever."
                   ))
    p.add_argument("--json", action="store_true", help="Emit JSON only")
    args = p.parse_args()

    if not TRACKER_PATH.exists():
        print(f"FATAL: tracker not found at {TRACKER_PATH}", file=sys.stderr)
        return 2
    data = json.loads(TRACKER_PATH.read_text())
    if not data:
        if not args.json:
            print("Tracker is empty — no instances to verify.")
        return 0

    # R31 SETUP-stuck tracking — load prior first-seen times, prune
    # entries for instances no longer in the tracker so the file doesn't
    # grow unbounded.
    setup_first_seen = _load_setup_first_seen()
    tracked_ids = {str(entry["instance_id"]) for entry in data}
    setup_first_seen = {k: v for k, v in setup_first_seen.items() if k in tracked_ids}
    now_ts = datetime.now(timezone.utc).timestamp()

    healths: list[InstanceHealth] = []
    for entry in data:
        iid = str(entry["instance_id"])
        label = entry.get("label", "?")

        # Step 1: vast.ai metadata
        meta = _vast_show_instance(iid)
        if meta is None:
            healths.append(InstanceHealth(
                instance_id=iid, label=label, classification="GONE",
                last_heartbeat_age_minutes=None, gpu_util_pct=None,
                ssh_host=None, ssh_port=None, crash_signal=None,
                notes="vastai show returned no data — instance destroyed or API error",
            ))
            continue

        host = meta.get("ssh_host")
        port = meta.get("ssh_port")
        gpu_util = meta.get("gpu_util")

        # Step 2: heartbeat + crash signals via SSH
        age_min, crash_sig = (None, None)
        if host and port:
            age_min, crash_sig = _ssh_check_heartbeat(host, int(port))

        # ssh_succeeded if crash_sig is None or non-SSH-error (heartbeat absent
        # is NOT an SSH failure; only "SSH_FAILED:" prefix is)
        ssh_ok = not (crash_sig and crash_sig.startswith("SSH_FAILED"))
        cls = classify(age_min, gpu_util, crash_sig, args.stale_minutes, ssh_ok)

        # R31 SETUP-stuck tracking: record first-seen-as-SETUP timestamp
        # so we can age it past --setup-stale-minutes and auto-destroy.
        # Drop the entry once the instance leaves SETUP (it's no longer
        # the in-flight state we're trying to time-box).
        setup_age_min: Optional[float] = None
        if cls == "SETUP":
            if iid not in setup_first_seen:
                setup_first_seen[iid] = now_ts
            setup_age_min = (now_ts - setup_first_seen[iid]) / 60.0
        else:
            setup_first_seen.pop(iid, None)

        healths.append(InstanceHealth(
            instance_id=iid, label=label, classification=cls,
            last_heartbeat_age_minutes=age_min, gpu_util_pct=gpu_util,
            ssh_host=host, ssh_port=port, crash_signal=crash_sig,
            notes=(meta.get("status_msg") or "")[:80],
            setup_age_minutes=setup_age_min,
        ))

    # Persist SETUP-first-seen tracker for the next verify pass.
    _save_setup_first_seen(setup_first_seen)

    # Output
    if args.json:
        print(json.dumps([asdict(h) for h in healths], indent=2))
    else:
        print(f"\n=== Vast.ai instance verify ({len(healths)} tracked) ===")
        for h in healths:
            age_s = f"{h.last_heartbeat_age_minutes:.1f}min" if h.last_heartbeat_age_minutes is not None else "?"
            util_s = f"{h.gpu_util_pct:.0f}%" if isinstance(h.gpu_util_pct, (int, float)) else "?"
            tag = {"HEALTHY": "✓", "IDLE": "⚠", "CRASHED": "✗", "UNREACHABLE": "?", "GONE": "·", "SETUP": "○"}.get(h.classification, "·")
            print(f"  {tag} {h.instance_id:>10} {h.label:<32} {h.classification:<12} hb={age_s:<8} util={util_s:<5}")
            if h.crash_signal and not h.crash_signal.startswith("SSH_FAILED"):
                print(f"      crash: {h.crash_signal[:100]}")

    # Auto-destroy
    # R31 cross-cutting fix: dual-threshold auto-destroy. IDLE/CRASHED
    # are time-boxed by --stale-minutes (heartbeat freshness); SETUP is
    # time-boxed independently by --setup-stale-minutes (first-seen-in-
    # SETUP age). Without the SETUP timer, a TRULY hung setup_full.sh
    # (deadlocked, never writes heartbeat) accrues cost forever — the
    # IDLE timer never fires because there's no heartbeat to be stale.
    if args.auto_destroy_stale:
        to_destroy = [
            h for h in healths
            if h.classification in ("IDLE", "CRASHED")
        ]
        stuck_setup = [
            h for h in healths
            if h.classification == "SETUP"
            and h.setup_age_minutes is not None
            and h.setup_age_minutes > args.setup_stale_minutes
        ]
        to_destroy.extend(stuck_setup)
        if to_destroy:
            print(f"\n=== Auto-destroying {len(to_destroy)} stale instance(s) ===")
            for h in to_destroy:
                reason = h.classification
                if h in stuck_setup:
                    reason = (
                        f"SETUP-STUCK (>{args.setup_stale_minutes:.0f}min "
                        f"in setup, age={h.setup_age_minutes:.1f}min)"
                    )
                print(f"  destroying {h.instance_id} ({h.label}, {reason})...")
                rc, out, err = _run([
                    "bash", "-c",
                    f"echo y | {VASTAI_BIN} destroy instance {h.instance_id}",
                ], timeout=30)
                print(f"    rc={rc} {out.strip()[:80]}")

    # Return code: 0 if all HEALTHY, 1 if any IDLE/CRASHED/UNREACHABLE
    bad = sum(1 for h in healths if h.classification not in ("HEALTHY", "GONE"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
