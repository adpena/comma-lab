#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Archive over-size JSONL state files into per-month buckets.

Per CLAUDE.md "Operator gates must be wired and used" + premortem #10
(`.omx/research/12_month_frustration_premortem_and_recommendations_
20260516.md` Category L + Section 3 #10):

`.omx/state/*.jsonl` files exceeding `--threshold-mb` (default 10 MB)
MUST archive older rows to `.omx/state/archive/<filename>_<YYYY-MM>.jsonl`
and keep only the most-recent `--retain-days` (default 90) window in the
live file. The append-only ledgers (commit-serializer.log,
modal_call_id_ledger.jsonl, subagent_progress.jsonl, cost_band_posterior.
jsonl) honor the archival policy.

Premortem L manifestation: `.omx/state/` passes 5 GB at the 12-month
horizon; Catalog #117 / #157 / #174 / #206 / #289 gates that scan the
serializer log get slower (linear scan over 50K rows). The serializer
log + modal_call_id_ledger.jsonl are the immediate hot paths.

This tool is the canonical archival helper. It:
1. Scans `.omx/state/` for `*.jsonl` files exceeding `--threshold-mb`.
2. For each, partitions rows by the row's UTC timestamp field
   (heuristic: tries `written_at_utc`, `logged_at_utc`,
   `dispatched_at_utc`, `timestamp` in that order).
3. Rows older than `--retain-days` move to
   `.omx/state/archive/<filename>_<YYYY-MM>.jsonl` (append-only).
4. The live file is rewritten with only the kept rows + an index
   manifest at `.omx/state/archive/_index.json` is updated.
5. `--dry-run` (default) only previews; `--apply` actually mutates.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
this tool follows the canonical-helper fcntl-locked write pattern (mirrors
Catalog #128 / #131 / #245 sister discipline) so concurrent subagent reads
of the live JSONL never see torn writes.

Usage:
    .venv/bin/python tools/archive_jsonl_state.py
    .venv/bin/python tools/archive_jsonl_state.py --apply
    .venv/bin/python tools/archive_jsonl_state.py --threshold-mb 5 --retain-days 60
    .venv/bin/python tools/archive_jsonl_state.py --target .omx/state/commit-serializer.log
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / ".omx" / "state"
ARCHIVE_DIR = STATE_DIR / "archive"
INDEX_PATH = ARCHIVE_DIR / "_index.json"

DEFAULT_THRESHOLD_MB = 10.0
DEFAULT_RETAIN_DAYS = 90

# Timestamp field names tried in order. Per CLAUDE.md "Apples-to-apples
# evidence discipline": every JSONL row in `.omx/state/` should declare
# its own UTC timestamp. The fallback `timestamp` matches the older
# lane_maturity_audit.log convention.
_TS_FIELD_CANDIDATES: tuple[str, ...] = (
    "written_at_utc",
    "logged_at_utc",
    "dispatched_at_utc",
    "timestamp",
    "ts",
    "iso_utc",
)


def _row_timestamp_iso(row: dict) -> str | None:
    for field in _TS_FIELD_CANDIDATES:
        v = row.get(field)
        if isinstance(v, str) and v:
            return v
    return None


def _parse_iso(ts: str) -> datetime | None:
    try:
        cleaned = ts.rstrip("Z").split("+")[0]
        return datetime.fromisoformat(cleaned).replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def _ym_bucket(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def plan_archive_jsonl(
    target: Path,
    *,
    threshold_mb: float = DEFAULT_THRESHOLD_MB,
    retain_days: int = DEFAULT_RETAIN_DAYS,
    now: datetime | None = None,
) -> dict:
    """Return a planning dict: which rows go to which bucket, which stay.

    No file mutation occurs.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now.timestamp() - retain_days * 86400
    plan = {
        "target": str(target),
        "exists": target.is_file(),
        "size_mb": 0.0,
        "above_threshold": False,
        "threshold_mb": threshold_mb,
        "retain_days": retain_days,
        "total_rows": 0,
        "rows_to_archive_by_bucket": {},
        "rows_to_keep": 0,
        "rows_undated": 0,
        "rows_malformed": 0,
    }
    if not target.is_file():
        return plan
    size_bytes = target.stat().st_size
    plan["size_mb"] = round(size_bytes / 1024 / 1024, 3)
    plan["above_threshold"] = size_bytes >= threshold_mb * 1024 * 1024

    bucket_counts: dict[str, int] = {}
    keep_count = 0
    undated = 0
    malformed = 0

    text = target.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if not line.strip():
            continue
        plan["total_rows"] += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            keep_count += 1  # Preserve malformed rows in-place.
            continue
        if not isinstance(row, dict):
            keep_count += 1
            continue
        ts_iso = _row_timestamp_iso(row)
        if not ts_iso:
            undated += 1
            keep_count += 1  # Preserve undated rows in-place (no way to bucket).
            continue
        dt = _parse_iso(ts_iso)
        if dt is None:
            undated += 1
            keep_count += 1
            continue
        if dt.timestamp() >= cutoff:
            keep_count += 1
        else:
            bucket = _ym_bucket(dt)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    plan["rows_to_archive_by_bucket"] = bucket_counts
    plan["rows_to_keep"] = keep_count
    plan["rows_undated"] = undated
    plan["rows_malformed"] = malformed
    plan["total_rows_to_archive"] = sum(bucket_counts.values())
    return plan


def apply_archive_jsonl(
    target: Path,
    plan: dict,
    *,
    archive_dir: Path = ARCHIVE_DIR,
    retain_days: int = DEFAULT_RETAIN_DAYS,
    now: datetime | None = None,
) -> dict:
    """Execute the archival plan: split target into archived + kept rows.

    Uses fcntl LOCK_EX + atomic rename (write-tmp + os.replace) per
    Catalog #128 / #131 / #245 sister discipline.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now.timestamp() - retain_days * 86400
    archive_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "target": str(target),
        "archived": {},
        "kept_rows": 0,
        "skipped_malformed": 0,
    }
    if not plan.get("above_threshold"):
        result["status"] = "skipped_below_threshold"
        return result

    # Acquire LOCK_EX on a sibling lock file so concurrent reads via
    # canonical helpers see a consistent state.
    lock_path = target.with_suffix(target.suffix + ".archive.lock")
    keep_rows: list[str] = []
    bucket_rows: dict[str, list[str]] = {}

    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    keep_rows.append(line)
                    result["skipped_malformed"] += 1
                    continue
                if not isinstance(row, dict):
                    keep_rows.append(line)
                    continue
                ts_iso = _row_timestamp_iso(row)
                dt = _parse_iso(ts_iso) if ts_iso else None
                if dt is None or dt.timestamp() >= cutoff:
                    keep_rows.append(line)
                else:
                    bucket = _ym_bucket(dt)
                    bucket_rows.setdefault(bucket, []).append(line)

            # Write archive buckets first (append-only).
            for bucket, rows in bucket_rows.items():
                archive_path = (
                    archive_dir / f"{target.name}_{bucket}.jsonl"
                )
                with open(archive_path, "a", encoding="utf-8") as af:
                    for row_line in rows:
                        af.write(row_line + "\n")
                result["archived"][bucket] = {
                    "path": str(archive_path),
                    "rows_added": len(rows),
                }

            # Atomically rewrite live file with only kept rows.
            tmp_path = target.with_suffix(
                target.suffix + f".tmp.{uuid.uuid4().hex[:12]}"
            )
            with open(tmp_path, "w", encoding="utf-8") as tf:
                for row_line in keep_rows:
                    tf.write(row_line + "\n")
            os.replace(tmp_path, target)
            result["kept_rows"] = len(keep_rows)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    result["status"] = "applied"
    return result


def _update_index(
    apply_result: dict,
    index_path: Path = INDEX_PATH,
) -> None:
    """Update the archive manifest at `.omx/state/archive/_index.json`."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = (
            json.loads(index_path.read_text(encoding="utf-8"))
            if index_path.is_file() else {"archived_runs": []}
        )
    except (OSError, json.JSONDecodeError):
        existing = {"archived_runs": []}
    if "archived_runs" not in existing or not isinstance(
        existing["archived_runs"], list
    ):
        existing["archived_runs"] = []
    entry = {
        "target": apply_result.get("target"),
        "status": apply_result.get("status"),
        "kept_rows": apply_result.get("kept_rows"),
        "archived": apply_result.get("archived", {}),
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    existing["archived_runs"].append(entry)
    index_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _discover_targets(
    state_dir: Path,
    threshold_mb: float,
) -> list[Path]:
    if not state_dir.is_dir():
        return []
    candidates = sorted(state_dir.glob("*.jsonl")) + sorted(
        state_dir.glob("*.log")
    )
    over = []
    for p in candidates:
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz >= threshold_mb * 1024 * 1024:
            over.append(p)
    return over


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir", type=Path, default=STATE_DIR,
        help="Directory to scan for over-size JSONL files",
    )
    parser.add_argument(
        "--target", type=Path, default=None,
        help="Archive a specific file (skip auto-discovery)",
    )
    parser.add_argument(
        "--threshold-mb", type=float, default=DEFAULT_THRESHOLD_MB,
    )
    parser.add_argument(
        "--retain-days", type=int, default=DEFAULT_RETAIN_DAYS,
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually archive (default is dry-run preview)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON output instead of human-readable",
    )
    args = parser.parse_args()

    if args.target is not None:
        targets = [args.target]
    else:
        targets = _discover_targets(args.state_dir, args.threshold_mb)

    if not targets:
        msg = (
            f"No JSONL/log files >= {args.threshold_mb} MB in "
            f"{args.state_dir}"
        )
        if args.json:
            print(json.dumps({"status": "no_targets", "message": msg}))
        else:
            print(msg)
        return 0

    overall = {"dry_run": not args.apply, "files": []}
    for target in targets:
        plan = plan_archive_jsonl(
            target,
            threshold_mb=args.threshold_mb,
            retain_days=args.retain_days,
        )
        entry = {"plan": plan}
        if args.apply and plan.get("above_threshold"):
            apply_result = apply_archive_jsonl(
                target, plan, retain_days=args.retain_days
            )
            _update_index(apply_result)
            entry["apply_result"] = apply_result
        overall["files"].append(entry)

    if args.json:
        print(json.dumps(overall, indent=2))
        return 0

    for entry in overall["files"]:
        plan = entry["plan"]
        print(f"\n=== {plan['target']} ===")
        print(f"Size: {plan['size_mb']} MB "
              f"(threshold: {plan['threshold_mb']} MB)")
        print(f"Total rows: {plan['total_rows']}")
        print(f"Rows to keep (within {plan['retain_days']}d window): "
              f"{plan['rows_to_keep']}")
        print(f"Rows to archive: {plan.get('total_rows_to_archive', 0)}")
        for bucket, count in sorted(
            plan["rows_to_archive_by_bucket"].items()
        ):
            print(f"  -> archive/{Path(plan['target']).name}_{bucket}.jsonl"
                  f"  +{count} rows")
        print(f"Rows undated (kept in-place): {plan['rows_undated']}")
        print(f"Rows malformed (kept in-place): {plan['rows_malformed']}")
        if "apply_result" in entry:
            print(f"\nApplied: {entry['apply_result']['status']}")
            print(f"  kept_rows: {entry['apply_result']['kept_rows']}")
            for bucket, info in entry["apply_result"].get(
                "archived", {}
            ).items():
                print(f"  archived {bucket}: {info['rows_added']} rows "
                      f"-> {info['path']}")

    if not args.apply:
        print(f"\nDry-run only. Pass `--apply` to actually archive.")
    else:
        print(f"\nArchive index updated at {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
