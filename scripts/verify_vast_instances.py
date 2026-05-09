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
import fcntl
import json
import os
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


def _setup_first_seen_lock_path() -> Path:
    """Sibling lockfile path for fcntl coordination."""
    return SETUP_FIRST_SEEN_PATH.with_suffix(SETUP_FIRST_SEEN_PATH.suffix + ".lock")


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
    """Transactional REPLACE write of the SETUP-first-seen tracker.

    Per CLAUDE.md non-negotiable + catalog #131: every shared-state write
    must serialize on a sibling fcntl lockfile and use a unique tmp path
    to prevent concurrent verify_vast_instances.py invocations from
    silently dropping each other's updates.

    HIGH 1 FIX (codex round 3, catalog #132): the previous version did
    ``existing = _load_setup_first_seen(); existing.update(data)`` INSIDE
    the lock, which silently re-introduced any stale ``first_seen`` rows
    that the caller (``main()``) had pruned because the instance is no
    longer in the tracker. Stale rows then made ``--auto-destroy-stale``
    target fresh instances that inherited an old age.

    The contract is TRANSACTIONAL REPLACE: the caller is the sole source
    of truth for the post-prune map and we write ``data`` directly. The
    caller MUST have already merged any concurrent updates and pruned
    deletions; this helper only performs the atomic commit.

    NOTE — codex round 4 MEDIUM 2 (catalog #135) prefers
    ``update_setup_first_seen_locked`` for callers that need the
    "load + merge + prune + save" full transaction inside ONE lock window.
    This helper remains for cases where the caller already holds canonical
    state (e.g., the future state-update API after refactor). New code
    should prefer the transactional update helper.

    Sister of:

    - ``tac.continual_learning.posterior_update_locked`` (catalog #128)
    - ``tac.deploy.lightning.active_jobs_state.update_active_jobs_locked``
      (catalog #131)
    - ``tac.deploy.azure.active_vms_state.update_active_vms_locked`` (#133)
    - ``tac.vastai_tracker._write_records``

    Memory: feedback_codex_round3_findings_fix_landed_20260509.md +
    feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md.
    """
    SETUP_FIRST_SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _setup_first_seen_lock_path()
    with open(lock_path, "w") as lockfd:
        fcntl.flock(lockfd.fileno(), fcntl.LOCK_EX)
        try:
            # TRANSACTIONAL REPLACE — write the caller's post-prune map
            # directly. DO NOT reload + .update() here; that re-introduces
            # stale rows the caller deliberately removed.
            normalized = {str(k): float(v) for k, v in data.items()}
            payload = json.dumps(normalized, indent=2, sort_keys=True)
            tmp = SETUP_FIRST_SEEN_PATH.with_suffix(
                SETUP_FIRST_SEEN_PATH.suffix + f".tmp.{os.getpid()}"
            )
            try:
                tmp.write_text(payload)
                with open(tmp, "rb") as f:
                    os.fsync(f.fileno())
                os.replace(tmp, SETUP_FIRST_SEEN_PATH)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
        finally:
            fcntl.flock(lockfd.fileno(), fcntl.LOCK_UN)


def update_setup_first_seen_locked(
    *,
    observed_setup_ids: set[str],
    tracked_ids: set[str],
    now_ts: float,
) -> dict[str, float]:
    """Locked transactional update of the SETUP-first-seen tracker.

    Codex round 4 MEDIUM 2 fix (2026-05-09, catalog #135): replaces the
    main()-flow lost-update race. Previously the main loop did:

        setup_first_seen = _load_setup_first_seen()        # OUTSIDE lock
        # ... long per-instance verify (~minutes) ...
        # ... mutate setup_first_seen ...
        _save_setup_first_seen(setup_first_seen)           # only the WRITE locked

    Two overlapping verifier runs (or a new instance registered mid-run)
    both loaded the same stale snapshot, did per-instance work for minutes,
    then the slower run replaced the file with its now-stale view —
    deleting first-seen timestamps that the faster run had created.
    SETUP age would silently reset for affected instances and
    ``--auto-destroy-stale`` would miss stuck paid instances.

    The new contract performs load + merge + prune + save INSIDE one
    fcntl-locked window:

      1. Acquire fcntl exclusive lock on the SETUP tracker lock file
      2. Reload the on-disk state INSIDE the lock (defends against any
         concurrent sister run that committed during our verify loop)
      3. For each id in ``observed_setup_ids``:
           - if the id is not yet in the on-disk map, set it to ``now_ts``
           - if already present, KEEP the OLDER timestamp (``min``) so a
             SETUP that has been stuck longer keeps its earliest first-seen
      4. For each id in the on-disk map: prune if not in ``tracked_ids``
         (drop entries for instances no longer in the vastai tracker)
      5. Atomically commit via ``_save_setup_first_seen`` (still inside lock)

    The merge-on-min semantics is the correct semantics for "first
    seen" — a SETUP that has been stuck longer should keep its earliest
    timestamp regardless of which verifier run committed last.

    Returns the post-merge / post-prune map (the canonical on-disk view
    after this transaction).

    Args:
        observed_setup_ids: set of instance ids that THIS verifier run
            observed in SETUP state.
        tracked_ids: set of instance ids currently in the vastai tracker;
            used to prune entries for destroyed instances.
        now_ts: unix timestamp to use for newly-observed SETUP ids.

    Memory: feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md.
    """
    SETUP_FIRST_SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _setup_first_seen_lock_path()
    with open(lock_path, "w") as lockfd:
        fcntl.flock(lockfd.fileno(), fcntl.LOCK_EX)
        try:
            # ── Step 1: STRICT-reload INSIDE the lock ────────────────
            current: dict[str, float] = {}
            if SETUP_FIRST_SEEN_PATH.exists():
                try:
                    raw = json.loads(SETUP_FIRST_SEEN_PATH.read_text())
                    if isinstance(raw, dict):
                        current = {str(k): float(v) for k, v in raw.items()}
                except (json.JSONDecodeError, ValueError, TypeError):
                    # Corrupt state — quarantine and start fresh
                    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    quarantine = SETUP_FIRST_SEEN_PATH.with_suffix(
                        SETUP_FIRST_SEEN_PATH.suffix + f".corrupt.{ts}"
                    )
                    counter = 0
                    while quarantine.exists():
                        counter += 1
                        quarantine = SETUP_FIRST_SEEN_PATH.with_suffix(
                            SETUP_FIRST_SEEN_PATH.suffix + f".corrupt.{ts}.{counter}"
                        )
                    try:
                        os.rename(SETUP_FIRST_SEEN_PATH, quarantine)
                    except OSError:
                        pass
                    current = {}

            # ── Step 2: merge observed SETUPs (KEEP older timestamp) ─
            for iid in observed_setup_ids:
                iid_s = str(iid)
                if iid_s in current:
                    # KEEP the older first-seen — never reset SETUP age
                    current[iid_s] = min(current[iid_s], float(now_ts))
                else:
                    current[iid_s] = float(now_ts)

            # ── Step 3: prune entries no longer tracked ──────────────
            # Drop ids not in the vastai tracker AND not observed in this
            # run (the latter could be a fresh insert by a sister verifier
            # we haven't seen yet — but we use ``tracked_ids`` as truth).
            tracked_set = {str(t) for t in tracked_ids}
            current = {k: v for k, v in current.items() if k in tracked_set}

            # Also drop ids that ARE tracked but NOT observed in SETUP
            # by this run AND not in the on-disk state from a prior run.
            # The merge in Step 2 already added our observations; the
            # prune in Step 3 removes destroyed instances. Ids that left
            # SETUP (now HEALTHY/IDLE/CRASHED) are dropped by the caller's
            # main flow but ONLY for ids THIS run observed leaving SETUP.
            # See the caller's main() flow for the per-id decision.

            # ── Step 4: atomic commit (still inside the lock) ───────
            payload = json.dumps(current, indent=2, sort_keys=True)
            tmp = SETUP_FIRST_SEEN_PATH.with_suffix(
                SETUP_FIRST_SEEN_PATH.suffix + f".tmp.{os.getpid()}"
            )
            try:
                tmp.write_text(payload)
                with open(tmp, "rb") as f:
                    os.fsync(f.fileno())
                os.replace(tmp, SETUP_FIRST_SEEN_PATH)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

            return current
        finally:
            fcntl.flock(lockfd.fileno(), fcntl.LOCK_UN)


def remove_setup_first_seen_locked(ids_to_remove: set[str]) -> dict[str, float]:
    """Locked transactional removal of SETUP-first-seen entries.

    Used by the main() flow to drop ids that are no longer in SETUP
    state (now HEALTHY / IDLE / CRASHED / GONE) within the same locked
    window as the sister update helper. Ensures concurrent verifier runs
    don't see a half-applied transaction.

    Returns the post-removal map.
    """
    SETUP_FIRST_SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _setup_first_seen_lock_path()
    with open(lock_path, "w") as lockfd:
        fcntl.flock(lockfd.fileno(), fcntl.LOCK_EX)
        try:
            current: dict[str, float] = {}
            if SETUP_FIRST_SEEN_PATH.exists():
                try:
                    raw = json.loads(SETUP_FIRST_SEEN_PATH.read_text())
                    if isinstance(raw, dict):
                        current = {str(k): float(v) for k, v in raw.items()}
                except (json.JSONDecodeError, ValueError, TypeError):
                    current = {}
            for iid in ids_to_remove:
                current.pop(str(iid), None)
            payload = json.dumps(current, indent=2, sort_keys=True)
            tmp = SETUP_FIRST_SEEN_PATH.with_suffix(
                SETUP_FIRST_SEEN_PATH.suffix + f".tmp.{os.getpid()}.rm"
            )
            try:
                tmp.write_text(payload)
                with open(tmp, "rb") as f:
                    os.fsync(f.fileno())
                os.replace(tmp, SETUP_FIRST_SEEN_PATH)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
            return current
        finally:
            fcntl.flock(lockfd.fileno(), fcntl.LOCK_UN)


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

    # R31 SETUP-stuck tracking — codex round 4 MEDIUM 2 fix (catalog #135):
    # the previous flow loaded + pruned the SETUP-first-seen map OUTSIDE the
    # lock, ran per-instance verify (~minutes), then saved at the end. Two
    # overlapping verifier runs (or a new instance registered mid-run) both
    # loaded the same stale snapshot, did per-instance work, then the slower
    # run replaced the file with its now-stale view — deleting first-seen
    # timestamps the faster run had created. SETUP age would silently reset.
    #
    # The fix: do the per-instance work using a local "observation" snapshot,
    # then compose ONE transactional update at the end. The transactional
    # helper does load+merge+prune+save inside a single lock window with
    # KEEP-OLDER-TIMESTAMP merge semantics; concurrent runs all converge.
    tracked_ids = {str(entry["instance_id"]) for entry in data}
    now_ts = datetime.now(timezone.utc).timestamp()

    # Snapshot of what's currently on-disk, used ONLY to compute setup_age_min
    # for the per-instance health row (informational; the canonical
    # transactional update at end-of-run uses an INSIDE-lock reload).
    observed_snapshot = _load_setup_first_seen()
    observed_snapshot = {k: v for k, v in observed_snapshot.items() if k in tracked_ids}

    observed_setup_ids: set[str] = set()
    left_setup_ids: set[str] = set()
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
            # GONE instances should not have first-seen entries either
            left_setup_ids.add(iid)
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
        setup_age_min: Optional[float] = None
        if cls == "SETUP":
            observed_setup_ids.add(iid)
            # For health-row reporting, prefer the older first-seen; the
            # transactional update at end-of-run will recompute against
            # an INSIDE-lock reload that may differ from this snapshot.
            first_seen_ts = observed_snapshot.get(iid, now_ts)
            setup_age_min = (now_ts - first_seen_ts) / 60.0
        else:
            # Instance left SETUP (or never was) — drop its first-seen entry
            # in the transactional update at end-of-run.
            left_setup_ids.add(iid)

        healths.append(InstanceHealth(
            instance_id=iid, label=label, classification=cls,
            last_heartbeat_age_minutes=age_min, gpu_util_pct=gpu_util,
            ssh_host=host, ssh_port=port, crash_signal=crash_sig,
            notes=(meta.get("status_msg") or "")[:80],
            setup_age_minutes=setup_age_min,
        ))

    # Codex round 4 MEDIUM 2 (catalog #135): single transactional update
    # at end-of-run. Reloads INSIDE the lock so concurrent verifier
    # commits during our verify loop do NOT get clobbered. Merges with
    # KEEP-OLDER semantics (a SETUP that's been stuck longer keeps its
    # earliest first-seen across all sister runs).
    #
    # First the merge-update for currently-observed SETUPs, then the
    # explicit removal for instances that left SETUP this run. Both are
    # locked transactions; ordering is: merge before remove so a flicker
    # between SETUP and HEALTHY in the same run is handled correctly
    # (the SETUP observation gets the timestamp, then the leave-SETUP
    # removal drops it cleanly).
    update_setup_first_seen_locked(
        observed_setup_ids=observed_setup_ids,
        tracked_ids=tracked_ids,
        now_ts=now_ts,
    )
    if left_setup_ids:
        remove_setup_first_seen_locked(left_setup_ids)

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
