#!/usr/bin/env python3
"""Compose four non-authorizing duty-ticket packages; never starts a trainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.witness_dsl.duty_queue_fire_tickets_20260719 import (  # noqa: E402
    DEFAULT_MAIN_REPO,
    DEFAULT_MAIN_SOURCE_REPO,
    DEFAULT_OUT_DIR,
    materialize_duty_queue_fire_tickets,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--main-repo", default=str(DEFAULT_MAIN_REPO))
    parser.add_argument(
        "--main-source-repo",
        default=str(DEFAULT_MAIN_SOURCE_REPO),
        help="Read-only MAIN checkout used to prove source ancestry and authority-file byte identity.",
    )
    parser.add_argument(
        "--created-utc",
        default=None,
        help="Optional caller-supplied timestamp; omission keeps artifacts deterministic.",
    )
    args = parser.parse_args(argv)
    summary = materialize_duty_queue_fire_tickets(
        args.out_dir,
        main_repo=args.main_repo,
        main_source_repo=args.main_source_repo,
        repo_root=REPO_ROOT,
        created_utc=args.created_utc,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
