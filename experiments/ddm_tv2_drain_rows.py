#!/usr/bin/env python3
"""ddm_tv2 -- drain the pending tolerance-curve rows through the canonical firer.

WHY THIS EXISTS
---------------
The evaluator-tolerance ladder needs ten local CPU advisory n600 rows against one
matched base.  Each row costs ~500 s of inflate plus ~460 s of evaluate, so the
ladder is wall-clock bound, and the failure mode that already cost this campaign
a wave was firing every row at once: four concurrent inflates tripped a kill and
returned rc=241 (SIGTERM) with nothing retained.

The cure is NOT to fire serially and idle the machine -- it is to keep exactly
one launch in flight at a time up to a small concurrency ceiling, and to let
``tools/safe_run.py``'s admission gate remain the final authority on capacity.
A refusal from that gate is information, not an obstacle: this drainer records
it and retries later rather than routing around it.

Every launch goes through ``tools/fire_local_advisory.py`` -- the ONE canonical
local advisory path, which carries the pyshim PATH prefix and
PYTHONDONTWRITEBYTECODE=1 that two prior arms lost by hand-assembling argv.
This drainer never assembles launcher argv itself.

Rows are fired against ALREADY-BUILT counterfactual token caches.  The receiver
validates each cache on load against a canonical manifest plus a full-payload
SHA-256, so reuse of a retained cache is proven at consumption rather than
assumed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIRER = REPO_ROOT / "tools" / "fire_local_advisory.py"
DONE_DIR = REPO_ROOT / ".omx" / "tmp" / "codex_runs"

# A row in flight shows up as THREE processes -- the detached supervisor, the
# safe_run wrapper, and the real invocation -- all carrying the same argv, so
# counting process matches over-reports capacity by 3x.  The row's work
# directory appears exactly once per row in those command lines, so counting
# DISTINCT work dirs is the artifact-immune measure of how many rows are live.
#
# The optional ``attempt_NNNN`` segment is REQUIRED, not cosmetic: rows fired by
# this drainer live at ``rows/<label>/attempt_0001/work`` (the canonical firer
# refuses a non-empty attempt dir, so every retry mints a fresh one), while rows
# fired directly live at ``rows/<label>/work``.  A pattern matching only the
# second form silently under-counts this drainer's OWN rows -- reporting zero
# live while three are rendering -- which reads as free capacity.  Measured and
# fixed 2026-08-24; the ``max_own_inflight`` ceiling and the host governor both
# held while it was wrong, so no row was over-committed.
INFLIGHT_WORKDIR_RE = re.compile(
    r"/rows/([A-Za-z0-9_]+)/(?:attempt_[0-9]+/)?work\b")


class DrainError(RuntimeError):
    """Fail-closed error for the drainer."""


def inflight_rows() -> set[str]:
    """Labels of rows live on this host, read from running command lines.

    Host-wide rather than arm-scoped on purpose: a sister arm's row consumes the
    same RAM and the same cores, so capacity has to be read from the machine.
    """
    proc = subprocess.run(["ps", "-axo", "command"], capture_output=True,
                          text=True, check=False)
    if proc.returncode != 0:
        raise DrainError(f"ps failed rc={proc.returncode}: {proc.stderr.strip()}")
    return set(INFLIGHT_WORKDIR_RE.findall(proc.stdout))


def inflight_count() -> int:
    """Number of distinct rows currently live on this host."""
    return len(inflight_rows())


def attempt_dirs(row_dir: Path) -> list[Path]:
    """Existing attempt dirs for a row, oldest first."""
    if not row_dir.is_dir():
        return []
    return sorted(p for p in row_dir.glob("attempt_*") if p.is_dir())


def mint_attempt_dir(row_dir: Path) -> Path:
    """Mint a FRESH attempt dir.

    The canonical firer refuses a non-empty attempt dir -- a deliberate guard
    against silently overwriting a prior attempt's receipts, which is how a
    failed row gets mistaken for an unrun one.  Honouring that guard means every
    retry gets its own numbered dir, so the refusal history stays on disk.
    """
    row_dir.mkdir(parents=True, exist_ok=True)
    existing = attempt_dirs(row_dir)
    nxt = 1 + max((int(p.name.split("_")[-1]) for p in existing), default=0)
    fresh = row_dir / f"attempt_{nxt:04d}"
    fresh.mkdir()
    return fresh


def row_is_complete(row_dir: Path) -> bool:
    """A row is complete only when SOME attempt's eval receipt exists and parses."""
    for attempt in attempt_dirs(row_dir):
        receipt = attempt / "contest_auth_eval.json"
        if not receipt.is_file():
            continue
        try:
            json.loads(receipt.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        return True
    return False


def row_status(row_dir: Path) -> str:
    """Status of the LATEST attempt; 'absent' when the row never launched."""
    attempts = attempt_dirs(row_dir)
    if not attempts:
        return "absent"
    receipt = attempts[-1] / "safe_run_status.json"
    if not receipt.is_file():
        return "absent"
    try:
        return str(json.loads(receipt.read_text()).get("status", "unknown"))
    except (json.JSONDecodeError, OSError):
        return "unparsable"


def fire_row(label: str, attempt_dir: Path, runtime_dir: Path, cache_root: Path,
             rss_mb: int, projected_gib: int, inflate_timeout: int) -> int:
    """Fire one row through the canonical firer. Returns the firer's returncode."""
    if not cache_root.is_dir():
        raise DrainError(f"token cache root missing for {label}: {cache_root}")
    cmd = [
        sys.executable,
        str(FIRER),
        "--runtime-dir", str(runtime_dir),
        "--archive", str(runtime_dir / "archive.zip"),
        "--attempt-dir", str(attempt_dir),
        "--label", label,
        "--env", f"F26_ADVISORY_DECODE_CACHE_ROOT={cache_root}",
        "--rss-mb", str(rss_mb),
        "--projected-gib", str(projected_gib),
        "--inflate-timeout", str(inflate_timeout),
        "--receipt-supersede",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
                          check=False)
    sys.stdout.write(f"[fire] {label} rc={proc.returncode}\n")
    if proc.returncode != 0:
        sys.stdout.write(f"[fire] stderr: {proc.stderr.strip()[:800]}\n")
    sys.stdout.flush()
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-root", required=True, type=Path,
                        help="arm directory holding rows/ (retained custody)")
    parser.add_argument("--runtime-dir", required=True, type=Path,
                        help="candidate runtime tree (the SHIPPED receiver)")
    parser.add_argument("--cache-root-parent", required=True, type=Path,
                        help="directory holding one published token cache per label")
    parser.add_argument("--labels", required=True, nargs="+",
                        help="row labels to drain, in fire order")
    parser.add_argument("--label-prefix", default="ddm_tv2_",
                        help="prefix for the launch label / done receipt name")
    parser.add_argument("--max-inflight", type=int, default=2,
                        help="ceiling on concurrent rows host-wide")
    parser.add_argument("--max-own-inflight", type=int, default=1,
                        help="ceiling on concurrent rows fired by THIS drainer")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--refusal-backoff-seconds", type=int, default=240,
                        help="wait this long after a governor refusal before retrying")
    parser.add_argument("--max-wait-seconds", type=int, default=36000)
    parser.add_argument("--rss-mb", type=int, default=24576)
    parser.add_argument("--projected-gib", type=int, default=8)
    parser.add_argument("--inflate-timeout", type=int, default=9000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    rows_dir = args.arm_root / "rows"
    pending = list(args.labels)
    fired: list[str] = []
    started = time.time()
    # A governor refusal is information, not an obstacle. Back off after one so
    # the drainer waits for real capacity instead of hammering the gate.
    refused_until = 0.0

    sys.stdout.write(f"[drain] {len(pending)} row(s) pending: {', '.join(pending)}\n")
    sys.stdout.write(f"[drain] max_inflight={args.max_inflight} "
                     f"runtime={args.runtime_dir}\n")
    sys.stdout.flush()

    if args.dry_run:
        for label in pending:
            cache = args.cache_root_parent / label
            sys.stdout.write(f"[dry-run] {label} cache={cache} "
                             f"exists={cache.is_dir()}\n")
        return 0

    while pending or fired:
        if time.time() - started > args.max_wait_seconds:
            sys.stdout.write("[drain] max wait exceeded; stopping\n")
            return 3

        # Retire finished rows.
        for label in list(fired):
            row_dir = rows_dir / label
            if row_is_complete(row_dir):
                sys.stdout.write(f"[done] {label} eval receipt landed\n")
                fired.remove(label)
                continue
            status = row_status(row_dir)
            if status in ("admission_refused", "failed", "oom", "timeout", "absent"):
                sys.stdout.write(f"[retry] {label} status={status}; requeued\n")
                fired.remove(label)
                pending.append(label)
                refused_until = time.time() + args.refusal_backoff_seconds

        live = inflight_count()
        now = time.time()
        # Two independent ceilings. ``max_inflight`` reads HOST capacity, which a
        # sister arm's rows also consume. ``max_own_inflight`` keeps THIS arm's
        # ladder serial, so a rung's cost is measured against a comparable
        # machine load rather than against however many rows happened to overlap.
        if (pending and live < args.max_inflight
                and len(fired) < args.max_own_inflight
                and now >= refused_until):
            label = pending.pop(0)
            row_dir = rows_dir / label
            if row_is_complete(row_dir):
                sys.stdout.write(f"[skip] {label} already complete\n")
                continue
            cache_root = args.cache_root_parent / label
            attempt_dir = mint_attempt_dir(row_dir)
            rc = fire_row(f"{args.label_prefix}{label}", attempt_dir,
                          args.runtime_dir, cache_root, args.rss_mb,
                          args.projected_gib, args.inflate_timeout)
            if rc != 0:
                # Launch never took; requeue and wait for real capacity.
                pending.append(label)
                refused_until = time.time() + args.refusal_backoff_seconds
                sys.stdout.write(
                    f"[backoff] {label} launch rc={rc}; retry in "
                    f"{args.refusal_backoff_seconds}s\n")
                sys.stdout.flush()
                continue
            fired.append(label)
            # Let the launch settle before re-reading capacity.
            time.sleep(30)
            continue

        if not pending and not fired:
            break
        sys.stdout.write(f"[wait] live={live} pending={len(pending)} "
                         f"inflight={len(fired)} t={int(time.time() - started)}s\n")
        sys.stdout.flush()
        time.sleep(args.poll_seconds)

    sys.stdout.write("[drain] all rows drained\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
