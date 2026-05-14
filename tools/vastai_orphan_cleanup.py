#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Detect and (optionally) destroy orphaned Vast.ai instances.

Reads the canonical `.omx/state/vastai_active_instances.json` tracker
written by `tac.vastai_tracker.register_instance(...)` and cross-references
it with `vastai show instances --raw`. An orphan is defined as a tracker
record where one of the following is true:

  * the instance is no longer present in `vastai show instances` output
    (the tracker entry should be removed)
  * the instance is present but has been registered for longer than
    `--idle-min` minutes AND its `gpu_util` has been 0 for the same
    duration (idle, presumed forgotten)
  * the instance is present but its `actual_status` is `stopped` /
    `exited` / `error` (paid contract, never running, never destroyed)

Usage:

    # Dry-run: list orphans only.
    python tools/vastai_orphan_cleanup.py

    # Destroy orphans (requires confirmation).
    python tools/vastai_orphan_cleanup.py --destroy

    # Override idle threshold.
    python tools/vastai_orphan_cleanup.py --idle-min 60
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from tac.vastai_tracker import (  # noqa: E402
    list_instances, remove_instance, tracker_path,
)


def _vastai_show_instances() -> list[dict]:
    """Run `vastai show instances --raw` and return parsed list."""
    try:
        result = subprocess.run(
            ["vastai", "show", "instances", "--raw"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"FATAL: could not query vastai: {e!r}", file=sys.stderr)
        sys.exit(2)
    if result.returncode != 0:
        print(
            f"FATAL: vastai show instances exited {result.returncode}: "
            f"{result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"FATAL: vastai output not JSON: {e!r}", file=sys.stderr)
        sys.exit(2)
    return data if isinstance(data, list) else []


def _age_minutes(iso_ts: str) -> float | None:
    try:
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    now = datetime.now(timezone.utc)
    return (now - ts).total_seconds() / 60.0


def _classify(record: dict, live_by_id: dict[str, dict], idle_min: float) -> str | None:
    iid = str(record.get("instance_id", ""))
    if not iid:
        return None
    live = live_by_id.get(iid)
    if live is None:
        return "missing"  # gone from vastai, tracker stale
    status = str(live.get("actual_status", ""))
    if status in ("stopped", "exited", "error"):
        return f"never-running ({status})"
    age = _age_minutes(record.get("registered_at_utc", ""))
    if age is None:
        return None
    if age < idle_min:
        return None
    gpu_util = live.get("gpu_util")
    try:
        gpu_util_f = float(gpu_util) if gpu_util is not None else 0.0
    except (TypeError, ValueError):
        gpu_util_f = 0.0
    if gpu_util_f < 1.0:
        return f"idle ({age:.0f} min, gpu_util={gpu_util_f:.0f}%)"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--destroy", action="store_true", default=False,
                   help="Actually destroy orphans (requires --yes too).")
    p.add_argument("--yes", action="store_true", default=False,
                   help="Skip interactive confirmation when --destroy is set.")
    p.add_argument("--idle-min", type=float, default=30.0,
                   help="Minutes of zero gpu_util before a live instance is "
                        "flagged orphan (default 30).")
    p.add_argument("--prune-missing", action="store_true", default=False,
                   help="Remove tracker records for instances that are no "
                        "longer in `vastai show instances` (no destroy needed).")
    args = p.parse_args()

    records = list_instances()
    print(f"vastai_active_instances tracker: {tracker_path()}")
    print(f"  records: {len(records)}")
    if not records:
        print("  (no registered instances)")
        return 0

    live = _vastai_show_instances()
    live_by_id = {str(li.get("id", "")): li for li in live}
    print(f"  live instances: {len(live)}")

    orphans: list[tuple[dict, str]] = []
    for r in records:
        reason = _classify(r, live_by_id, args.idle_min)
        if reason:
            orphans.append((r, reason))

    if not orphans:
        print("OK — no orphans detected.")
        return 0

    print(f"\nORPHANS ({len(orphans)}):")
    for r, reason in orphans:
        print(
            f"  • id={r.get('instance_id')} label={r.get('label')!r} "
            f"reason={reason}"
        )

    if not args.destroy and not args.prune_missing:
        print("\n(dry-run; pass --destroy to act, --prune-missing to clean tracker)")
        return 0

    # Confirmation
    if args.destroy and not args.yes:
        try:
            resp = input("\nDestroy these instances? [yes/NO]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            resp = ""
        if resp != "yes":
            print("Aborted.")
            return 1

    rc = 0
    for r, reason in orphans:
        iid = str(r.get("instance_id"))
        if reason == "missing":
            if args.prune_missing or args.destroy:
                if remove_instance(iid):
                    print(f"  pruned tracker entry id={iid}")
            continue
        if not args.destroy:
            continue
        # Destroy live instance
        try:
            result = subprocess.run(
                ["vastai", "destroy", "instance", iid],
                capture_output=True, text=True, timeout=60, check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  destroy id={iid} failed: {e!r}")
            rc = 1
            continue
        if result.returncode == 0:
            print(f"  destroyed id={iid}")
            remove_instance(iid)
        else:
            print(f"  destroy id={iid} returned {result.returncode}: "
                  f"{result.stderr.strip()}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
