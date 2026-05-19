#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pre-commit hook: refuse bare ``git add`` if a sister subagent is in-flight.

Operator-runnable helper for the ``/commit`` slash command (which does bare
``git add`` + ``git commit`` outside the canonical
``tools/subagent_commit_serializer.py`` per the commit-commands plugin at
``~/.claude/plugins/marketplaces/claude-plugins-official/plugins/commit-commands/commands/commit.md``).

This script is layer 3 of the CATALOG-314-PREVENTION-ENHANCEMENT 4-surface
extinction (canonical helper / serializer wire-in / pre-commit hook /
STRICT preflight gate). The ``/commit`` slash command is operator-owned and
cannot be edited from this repo, so this helper documents the integration
pattern: invoke this tool BEFORE the bare ``git add``, refuse the commit if
it returns non-zero.

Usage
─────

As a manual pre-commit check::

    python tools/check_sister_checkpoint_before_git_add.py \\
        --files src/tac/preflight.py CLAUDE.md \\
        --label MY-SUBAGENT-ID

As a git pre-commit hook (operator installs at ``.git/hooks/pre-commit``)::

    #!/bin/bash
    set -euo pipefail
    files=$(git diff --cached --name-only)
    python tools/check_sister_checkpoint_before_git_add.py \\
        --files-from-stdin \\
        --label "${SUBAGENT_LABEL:-anonymous}" \\
        <<<"$files"

Exit codes
──────────
0   PROCEED — no conflict; commit may proceed
8   ABORT — sister subagent in-flight on overlapping file(s); coordinate
9   WAIT_AND_RETRY — overlap but sister near completion; retry shortly
10  Bare paired-env bypass attempt without rationale (Catalog #199 discipline)
11  Corrupt subagent_progress.jsonl (Catalog #138 fail-closed)
2   CLI error (bad arguments, etc.)

Paired-env bypass (per Catalog #199)::

    export SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE=1
    export SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE_RATIONALE="<text ≥4 chars>"
    git commit -m "..."

The same paired-env discipline used by ``subagent_commit_serializer.py``
applies here so the bypass surfaces consistently regardless of which
commit path the operator uses.

Memory: ``feedback_catalog_314_prevention_enhancement_landed_20260519.md``.
Lane: ``lane_catalog_314_prevention_enhancement_20260519``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.commit_safety import (  # noqa: E402
    bare_override_attempted,
    check_files_against_sister_checkpoints,
    parse_override_env,
)
from tac.commit_safety.sister_checkpoint_guard import (  # noqa: E402
    CorruptCheckpointError,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-commit hook: refuse bare git add when a sister subagent has "
            "declared the same files in its in-flight checkpoint. Mirrors "
            "tools/subagent_commit_serializer.py STAGING-surface check at "
            "the /commit slash-command surface."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--files", nargs="*", default=None,
        help="Files about to be staged (repo-relative paths).",
    )
    parser.add_argument(
        "--files-from-stdin", action="store_true",
        help="Read newline-separated file paths from stdin (in addition to "
             "any --files).",
    )
    parser.add_argument(
        "--label", default=os.environ.get("SUBAGENT_LABEL", "anonymous"),
        help="Caller's subagent_id (so the caller doesn't flag itself). "
             "Default: $SUBAGENT_LABEL or 'anonymous'.",
    )
    parser.add_argument(
        "--lookback-minutes", type=int, default=60,
        help="In-flight checkpoint lookback window in minutes. Default 60.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress PROCEED diagnostic on stdout (errors still go to stderr).",
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

    # Bare paired-env override attempt → rc=10 (same discipline as serializer).
    if bare_override_attempted(dict(os.environ)):
        print(
            "[check_sister_checkpoint_before_git_add] REFUSED: bare paired-env "
            "bypass. SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE=1 REQUIRES "
            "paired SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE_RATIONALE=<text> "
            "(≥4 chars, NOT a placeholder). Per Catalog #199 paired-env + "
            "Catalog #340 STAGING-surface PREVENT.",
            file=sys.stderr,
        )
        return 10

    # Paired-env bypass active → log + proceed.
    bypass_active, bypass_rationale = parse_override_env(dict(os.environ))
    if bypass_active:
        if not args.quiet:
            print(
                f"[check_sister_checkpoint_before_git_add] BYPASS ACTIVE: "
                f"paired-env override accepted (rationale: "
                f"{bypass_rationale!r}). Commit proceeds; audit-trail entry "
                f"recommended via .omx/research/ memo per Catalog #230.",
                file=sys.stderr,
            )
        return 0

    try:
        verdict = check_files_against_sister_checkpoints(
            files,
            current_subagent_id=args.label,
            lookback_minutes=args.lookback_minutes,
        )
    except CorruptCheckpointError as exc:
        print(
            f"[check_sister_checkpoint_before_git_add] REFUSED: corrupt "
            f"subagent_progress.jsonl. Per Catalog #138 fail-closed + "
            f"Catalog #340 STAGING-surface PREVENT.\n  {exc!s}",
            file=sys.stderr,
        )
        return 11

    if verdict.recommendation == "PROCEED":
        if not args.quiet:
            print(
                f"[check_sister_checkpoint_before_git_add] OK: "
                f"{verdict.diagnostic}"
            )
        return 0

    if verdict.recommendation == "ABORT":
        print(
            f"[check_sister_checkpoint_before_git_add] REFUSED: ABORT per "
            f"Catalog #340 STAGING-surface PREVENT (sister of Catalog #314 "
            f"POST-COMMIT detect). Coordinate via Catalog #230 ownership "
            f"map OR set paired-env bypass.\n{verdict.diagnostic}",
            file=sys.stderr,
        )
        return 8

    if verdict.recommendation == "WAIT_AND_RETRY":
        print(
            f"[check_sister_checkpoint_before_git_add] REFUSED: "
            f"WAIT_AND_RETRY per Catalog #340. Retry with exponential "
            f"backoff (e.g. 30s/60s/120s); escalate to Catalog #230 if "
            f"still ABORT after reasonable delay.\n{verdict.diagnostic}",
            file=sys.stderr,
        )
        return 9

    # Defensive: unknown recommendation
    print(
        f"[check_sister_checkpoint_before_git_add] REFUSED: unknown "
        f"recommendation {verdict.recommendation!r}; fail-closed.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
