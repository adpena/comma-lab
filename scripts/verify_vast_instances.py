#!/usr/bin/env python3
"""Per-instance health verify for Vast.ai active fleet.

For each registered instance in `.omx/state/vastai_active_instances.json`:
  1. Confirm Vast.ai still reports it alive
  2. SSH in (with timeout) and check heartbeat.log freshness + GPU activity
  3. Classify: HEALTHY / IDLE / CRASHED / UNREACHABLE / GONE

Usage:
    python scripts/verify_vast_instances.py
    python scripts/verify_vast_instances.py --auto-destroy-stale --stale-minutes 30

Emits a JSON report to stdout + (optional) destroys CRASHED/IDLE instances
older than the threshold.

Reference memories:
  - feedback_vastai_launch_returns_success_before_lane_starts
  - feedback_vastai_cost_paranoia
  - feedback_canonical_remote_bootstraps
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
VASTAI_BIN = REPO_ROOT / ".venv/bin/vastai"
SSH_BASE = [
    "ssh",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=10",
    "-o", "LogLevel=ERROR",
]


@dataclass
class InstanceHealth:
    instance_id: str
    label: str
    classification: str  # HEALTHY / IDLE / CRASHED / UNREACHABLE / GONE
    last_heartbeat_age_minutes: Optional[float]
    gpu_util_pct: Optional[float]
    ssh_host: Optional[str]
    ssh_port: Optional[int]
    crash_signal: Optional[str]
    notes: str


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
                   help="Destroy IDLE/CRASHED instances older than --stale-minutes")
    p.add_argument("--stale-minutes", type=float, default=30.0,
                   help="Heartbeat older than this is IDLE (default: 30)")
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
        healths.append(InstanceHealth(
            instance_id=iid, label=label, classification=cls,
            last_heartbeat_age_minutes=age_min, gpu_util_pct=gpu_util,
            ssh_host=host, ssh_port=port, crash_signal=crash_sig,
            notes=(meta.get("status_msg") or "")[:80],
        ))

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
    if args.auto_destroy_stale:
        to_destroy = [h for h in healths if h.classification in ("IDLE", "CRASHED")]
        if to_destroy:
            print(f"\n=== Auto-destroying {len(to_destroy)} stale instance(s) ===")
            for h in to_destroy:
                print(f"  destroying {h.instance_id} ({h.label}, {h.classification})...")
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
