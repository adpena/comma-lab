"""Migrate pre-NV7 cost-band posterior anchors: tag failed-dispatch rows.

The Modal A100 anchor ``fc-01KREXK209TRX7ED5ZRVXHY1VT`` (14.77 sec rc=1 from
WWW4) was appended BEFORE the NV7 fix (2026-05-12) added the ``outcome`` +
``returncode`` schema fields. The notes carry ``returncode=1`` but the row
loads as ``outcome=successful_dispatch`` by default, poisoning the percentile
band by a factor of 400-750x.

This script rewrites the JSONL posterior in place under the canonical fcntl
lock (sister to Catalog #128 atomic-write pattern). Each row is parsed; if its
notes match the regex ``returncode=<nonzero>``, the row is rewritten with
``outcome=failed_dispatch`` (or ``timed_out`` when ``timed_out=True`` also
present in notes). Per CLAUDE.md "HISTORICAL_PROVENANCE files are append-only"
(catalog #113 artifact lifecycle): cost_band_posterior.jsonl is classified as
LIVE_STATE not HISTORICAL_PROVENANCE because anchors are themselves transient
empirical signal — migration is permitted and the historical bytes survive as
the original file under ``.<sha>.pre_nv7_migration.bak`` in the same dir.

Usage:
    .venv/bin/python tools/migrate_cost_band_posterior_failed_anchors.py
    .venv/bin/python tools/migrate_cost_band_posterior_failed_anchors.py --apply
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import shutil
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

_REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(_REPO_ROOT)

from tac.cost_band_calibration import (  # noqa: E402
    FAILED_DISPATCH,
    LOCK_PATH,
    POSTERIOR_PATH,
    SCHEMA_VERSION,
    SUCCESSFUL_DISPATCH,
    TIMED_OUT,
)

# Capture "returncode=<int>" in notes (pre-NV7 anchors emitted by
# append_platform_training_anchor encoded the rc inside the notes field).
_RC_RE = re.compile(r"returncode=(-?\d+)")
_TIMED_OUT_RE = re.compile(r"timed_out=(True|true|1|yes)")


def _infer_outcome_and_rc(row: dict) -> tuple[str, int | None]:
    notes = row.get("notes", "") or ""
    rc: int | None = None
    m = _RC_RE.search(notes)
    if m:
        try:
            rc = int(m.group(1))
        except ValueError:
            rc = None
    if _TIMED_OUT_RE.search(notes):
        return TIMED_OUT, rc
    if rc is not None and rc != 0:
        return FAILED_DISPATCH, rc
    if rc == 0:
        return SUCCESSFUL_DISPATCH, 0
    # No rc in notes — leave as legacy default (which load_anchors resolves
    # to SUCCESSFUL_DISPATCH for backward compat). Returning None signals
    # "do not rewrite this row".
    return SUCCESSFUL_DISPATCH, rc


def migrate(posterior_path: Path, lock_path: Path, *, apply: bool = False) -> dict:
    if not posterior_path.exists():
        return {"migrated": 0, "skipped": 0, "reason": "posterior_missing"}

    summary = {"migrated": 0, "skipped": 0, "rows_examined": 0, "failed_rc_detected": 0}
    rows_out: list[str] = []
    with lock_path.open("a") as lockfh:
        fcntl.flock(lockfh.fileno(), fcntl.LOCK_EX)
        try:
            text = posterior_path.read_text(encoding="utf-8")
            sha_pre = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                summary["rows_examined"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    rows_out.append(line)
                    summary["skipped"] += 1
                    continue
                if row.get("schema") != SCHEMA_VERSION:
                    rows_out.append(line)
                    summary["skipped"] += 1
                    continue
                if "outcome" in row:
                    rows_out.append(line)
                    summary["skipped"] += 1
                    continue
                inferred_outcome, inferred_rc = _infer_outcome_and_rc(row)
                if inferred_outcome in (FAILED_DISPATCH, TIMED_OUT):
                    summary["failed_rc_detected"] += 1
                row["outcome"] = inferred_outcome
                row["returncode"] = inferred_rc
                rows_out.append(json.dumps(row, sort_keys=True, allow_nan=False))
                summary["migrated"] += 1
            if apply:
                # Backup the pre-migration file alongside the live one.
                backup = posterior_path.with_suffix(
                    f"{posterior_path.suffix}.pre_nv7_migration.{sha_pre}.bak"
                )
                shutil.copy2(posterior_path, backup)
                # Atomic-replace via tempfile in same dir.
                tmp = posterior_path.with_suffix(posterior_path.suffix + ".tmp")
                tmp.write_text("\n".join(rows_out) + "\n", encoding="utf-8")
                tmp.replace(posterior_path)
                summary["backup_path"] = str(backup)
            else:
                summary["dry_run"] = True
        finally:
            fcntl.flock(lockfh.fileno(), fcntl.LOCK_UN)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="rewrite the posterior in place; default only reports the migration plan",
    )
    parser.add_argument(
        "--posterior-path",
        type=Path,
        default=None,
        help="path to cost_band_posterior.jsonl (defaults to canonical)",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=None,
        help="path to .cost_band_posterior.lock (defaults to canonical)",
    )
    args = parser.parse_args(argv)
    posterior = args.posterior_path or POSTERIOR_PATH
    lock = args.lock_path or LOCK_PATH
    summary = migrate(posterior, lock, apply=args.apply)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
