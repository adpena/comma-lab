#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PRE-WRITE sister-activity check: refuse Writes when sister already landed.

Operator-runnable helper for subagent PV (per Catalog #229 premise-verification)
that closes the empirical gap surfaced by NERV-FAMILY-L0-BUILD stand-down
2026-05-20 (~30 min wasted duplicating sister commit ``18b0beed6``'s
ego_nerv + e_nerv + nervdc trainers because PV did not run
``git log -- <target>`` before first Write).

Sister of ``tools/check_sister_checkpoint_before_git_add.py`` (Catalog #340
helper sister) at the PRE-WRITE / git-log surface. Together they close the
multi-subagent edit/commit collision class bidirectionally:

- ``check_sister_checkpoint_before_git_add.py`` (Catalog #340) — fires at
  staging/commit time against in-flight ``subagent_progress.jsonl`` sister
  checkpoints (STILL-RUNNING sisters).
- ``check_sister_files_recently_landed.py`` (this script) — fires at PV /
  pre-Write time against landed git-log sister commits (sisters that
  ALREADY SHIPPED equivalent work in the lookback window).

Usage
─────

As a manual PV check before first Write::

    python tools/check_sister_files_recently_landed.py \\
        --files src/tac/foo.py src/tac/bar.py \\
        --lookback-hours 6 \\
        --own-subagent-id wave-3-my-subagent-id-20260520

As part of a subagent prompt template (canonical pattern; see
``docs/canonical_subagent_pre_flight_checklist.md``)::

    Before first Write, run:
    .venv/bin/python tools/check_sister_files_recently_landed.py \\
        --files <targets> --lookback-hours 6 --own-subagent-id <self>

Exit codes
──────────
0   PROCEED — no sister activity in lookback window; safe to write
8   STAND_DOWN_DUPLICATE — sister landed equivalent work; stand down + clean up
9   WAIT_AND_REASSESS — sister activity present but ambiguous; re-read landings
2   CLI error (bad arguments, etc.)

(Exit codes 8/9 mirror the sister Catalog #340 helper for operator habit
consistency: rc=8 = stand down, rc=9 = wait + reassess.)

Memory: ``feedback_wave_3_pre_write_sister_activity_check_helper_landed_20260520.md``.
Lane: ``lane_wave_3_pre_write_sister_activity_check_helper_20260520``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.commit_safety import (  # noqa: E402
    DEFAULT_LOOKBACK_HOURS,
    check_sister_files_recently_landed,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PRE-WRITE sister-activity check: refuse Writes when sister already "
            "landed equivalent work on the target files within the lookback "
            "window. Sister of tools/check_sister_checkpoint_before_git_add.py "
            "(Catalog #340) at the PRE-WRITE / git-log surface."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--files", nargs="*", default=None,
        help="Files the caller intends to Write (repo-relative paths).",
    )
    parser.add_argument(
        "--files-from-stdin", action="store_true",
        help="Read newline-separated file paths from stdin (in addition to "
             "any --files).",
    )
    parser.add_argument(
        "--lookback-hours", type=int, default=DEFAULT_LOOKBACK_HOURS,
        help=f"Lookback window in hours. Default {DEFAULT_LOOKBACK_HOURS} "
             "(per NERV-FAMILY-L0-BUILD empirical anchor; sister landed "
             "~4.5h before duplicate dispatch).",
    )
    parser.add_argument(
        "--own-subagent-id", default=os.environ.get("SUBAGENT_LABEL", None),
        help="Caller's subagent_id (so caller-authored commits are filtered "
             "out via Co-Authored-By body match). Default: $SUBAGENT_LABEL.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress PROCEED diagnostic on stdout (errors still go to stderr).",
    )
    parser.add_argument(
        "--repo-root", default=None,
        help="Override repo root (default: auto-detect from helper module).",
    )
    args = parser.parse_args(argv)

    files = list(args.files or [])
    if args.files_from_stdin:
        for line in sys.stdin:
            line = line.strip()
            if line:
                files.append(line)

    if not files:
        parser.error("must pass --files or --files-from-stdin")

    try:
        verdict = check_sister_files_recently_landed(
            files,
            lookback_hours=args.lookback_hours,
            own_subagent_id=args.own_subagent_id,
            repo_root=args.repo_root,
        )
    except (TypeError, ValueError) as exc:
        print(
            f"[check_sister_files_recently_landed] ERROR: {exc!s}",
            file=sys.stderr,
        )
        return 2

    if verdict.recommendation == "PROCEED":
        if not args.quiet:
            print(
                f"[check_sister_files_recently_landed] OK: {verdict.rationale}"
            )
        return 0

    if verdict.recommendation == "STAND_DOWN_DUPLICATE":
        print(
            f"[check_sister_files_recently_landed] REFUSED: STAND_DOWN_DUPLICATE "
            f"per Catalog #229 PV discipline + CLAUDE.md \"Subagent coherence-"
            f"by-default\" non-negotiable.\n{verdict.rationale}",
            file=sys.stderr,
        )
        return 8

    if verdict.recommendation == "WAIT_AND_REASSESS":
        print(
            f"[check_sister_files_recently_landed] WAIT_AND_REASSESS: "
            f"sister activity ambiguous; re-read sister landing memos first.\n"
            f"{verdict.rationale}",
            file=sys.stderr,
        )
        return 9

    # Defensive: unknown recommendation
    print(
        f"[check_sister_files_recently_landed] REFUSED: unknown recommendation "
        f"{verdict.recommendation!r}; fail-closed.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
