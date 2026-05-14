#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reconcile Vast.ai live state against local dispatch trackers.

This is intentionally non-destructive. It compares:

1. `vastai show instances --raw` live contracts
2. `.omx/state/vastai_active_instances.json` launcher tracker
3. `.omx/state/active_dispatches.md` operator-facing active table

The goal is to surface drift before it becomes cost leak or false evidence:
stale tracker rows, active-dispatch rows whose instance is gone, and live
instances missing from local source-of-truth files.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
VASTAI = REPO_ROOT / ".venv/bin/vastai"
TRACKER_PATH = REPO_ROOT / ".omx/state/vastai_active_instances.json"
ACTIVE_DISPATCHES_PATH = REPO_ROOT / ".omx/state/active_dispatches.md"


@dataclass(frozen=True)
class ActiveDispatchRow:
    lane_label: str
    instance_id: str | None
    raw_instance_cell: str
    timestamp_utc: str


def normalize_attempt_label(label: str) -> str:
    """Strip launcher attempt suffixes for prefix matching."""
    return re.sub(r"_a\d+$", "", str(label or "").strip())


def _load_json_path(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _query_vastai_live() -> list[dict[str, Any]]:
    proc = subprocess.run(
        [str(VASTAI), "show", "instances", "--raw"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"vastai show instances failed rc={proc.returncode}: "
            f"{proc.stderr.strip()[:300]}"
        )
    data = json.loads(proc.stdout)
    if not isinstance(data, list):
        raise RuntimeError("vastai show instances returned non-list JSON")
    return [x for x in data if isinstance(x, dict)]


def load_live_instances(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return _query_vastai_live()
    data = _load_json_path(path)
    if not isinstance(data, list):
        raise RuntimeError(f"{path} did not contain a JSON list")
    return [x for x in data if isinstance(x, dict)]


def parse_active_dispatches(text: str) -> list[ActiveDispatchRow]:
    """Parse only the `## Active` markdown table."""
    rows: list[ActiveDispatchRow] = []
    in_active = False
    for line in text.splitlines():
        if line.strip() == "## Active":
            in_active = True
            continue
        if in_active and line.startswith("## "):
            break
        if not in_active:
            continue
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if "---" in stripped or "timestamp_utc" in stripped:
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        timestamp, lane_label, instance_cell = cells[0], cells[1], cells[2]
        m = re.search(r"\b(\d{5,})\b", instance_cell)
        rows.append(ActiveDispatchRow(
            lane_label=lane_label,
            instance_id=m.group(1) if m else None,
            raw_instance_cell=instance_cell,
            timestamp_utc=timestamp,
        ))
    return rows


def load_active_dispatches(path: Path) -> list[ActiveDispatchRow]:
    if not path.exists():
        return []
    return parse_active_dispatches(path.read_text())


def reconcile(
    live: list[dict[str, Any]],
    tracker: list[dict[str, Any]],
    active_rows: list[ActiveDispatchRow],
) -> dict[str, Any]:
    live_by_id = {str(x.get("id")): x for x in live if x.get("id") is not None}
    tracker_by_id = {
        str(x.get("instance_id")): x
        for x in tracker
        if isinstance(x, dict) and x.get("instance_id") is not None
    }
    active_ids = {r.instance_id for r in active_rows if r.instance_id}
    active_prefixes = {
        normalize_attempt_label(r.lane_label)
        for r in active_rows
        if r.lane_label
    }

    live_labels = {
        str(x.get("id")): str(x.get("label") or "")
        for x in live
        if x.get("id") is not None
    }
    live_prefixes = {normalize_attempt_label(label) for label in live_labels.values()}

    tracker_missing_live = [
        {
            "instance_id": iid,
            "label": rec.get("label"),
            "registered_at_utc": rec.get("registered_at_utc"),
        }
        for iid, rec in sorted(tracker_by_id.items())
        if iid not in live_by_id
    ]
    live_missing_tracker = [
        {
            "instance_id": iid,
            "label": live_labels.get(iid),
            "actual_status": live_by_id[iid].get("actual_status"),
            "ssh_host": live_by_id[iid].get("ssh_host"),
            "ssh_port": live_by_id[iid].get("ssh_port"),
        }
        for iid in sorted(live_by_id)
        if iid not in tracker_by_id
    ]
    active_missing_live = [
        asdict(row)
        for row in active_rows
        if row.instance_id is not None and row.instance_id not in live_by_id
    ]
    active_label_without_live_prefix = [
        asdict(row)
        for row in active_rows
        if row.instance_id is None
        and normalize_attempt_label(row.lane_label) not in live_prefixes
    ]
    live_missing_active = [
        {
            "instance_id": iid,
            "label": label,
            "normalized_label": normalize_attempt_label(label),
        }
        for iid, label in sorted(live_labels.items())
        if iid not in active_ids and normalize_attempt_label(label) not in active_prefixes
    ]

    return {
        "live_count": len(live),
        "tracker_count": len(tracker_by_id),
        "active_dispatch_count": len(active_rows),
        "tracker_missing_live": tracker_missing_live,
        "live_missing_tracker": live_missing_tracker,
        "active_missing_live": active_missing_live,
        "active_label_without_live_prefix": active_label_without_live_prefix,
        "live_missing_active": live_missing_active,
    }


def _print_human(report: dict[str, Any], max_items: int) -> None:
    print("=== Vast Dispatch State Reconciliation ===")
    print(
        f"live={report['live_count']} tracker={report['tracker_count']} "
        f"active_dispatches={report['active_dispatch_count']}"
    )
    for key in (
        "tracker_missing_live",
        "live_missing_tracker",
        "active_missing_live",
        "active_label_without_live_prefix",
        "live_missing_active",
    ):
        items = report[key]
        print(f"\n{key}: {len(items)}")
        for item in items[:max_items]:
            print(f"  - {json.dumps(item, sort_keys=True)}")
        if len(items) > max_items:
            print(f"  ... {len(items) - max_items} more")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live-json", type=Path, help="Use saved vastai raw JSON")
    p.add_argument("--tracker-json", type=Path, default=TRACKER_PATH)
    p.add_argument("--active-dispatches", type=Path, default=ACTIVE_DISPATCHES_PATH)
    p.add_argument("--json", action="store_true", help="Emit JSON only")
    p.add_argument("--strict", action="store_true", help="Exit 1 if any drift exists")
    p.add_argument("--max-items", type=int, default=20)
    args = p.parse_args(argv)

    live = load_live_instances(args.live_json)
    tracker = _load_json_path(args.tracker_json)
    if not isinstance(tracker, list):
        raise RuntimeError(f"{args.tracker_json} did not contain a JSON list")
    active_rows = load_active_dispatches(args.active_dispatches)
    report = reconcile(live, tracker, active_rows)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report, args.max_items)

    drift = sum(
        len(report[k])
        for k in (
            "tracker_missing_live",
            "live_missing_tracker",
            "active_missing_live",
            "active_label_without_live_prefix",
            "live_missing_active",
        )
    )
    return 1 if args.strict and drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
