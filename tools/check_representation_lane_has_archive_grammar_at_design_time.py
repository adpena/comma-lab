#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Standalone wrapper for preflight catalog #124.

The canonical implementation lives in ``tac.preflight`` so the normal
operator preflight and this CLI cannot drift. This wrapper exists only for
focused local audits and CI/runbook discoverability.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.preflight import (  # noqa: E402
    PreflightError,
    check_representation_lane_has_archive_grammar_at_design_time,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    try:
        violations = check_representation_lane_has_archive_grammar_at_design_time(
            repo_root=args.repo_root,
            strict=args.strict,
            verbose=True,
        )
    except PreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 1 if args.strict and violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
