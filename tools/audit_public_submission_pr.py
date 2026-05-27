#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit a public contest submission PR before maintainer review.

Examples:
  .venv/bin/python tools/audit_public_submission_pr.py --pr 110 --format text
  .venv/bin/python tools/audit_public_submission_pr.py --pr 110 --inflate-smoke \
    --expected-output-sha256 <raw-sha> --python-bin .venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.submission_packet.public_pr_audit import (  # noqa: E402
    DEFAULT_TARGET_REPO,
    audit_public_submission_pr,
    config_from_args,
    format_text_report,
    self_test_result,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_TARGET_REPO, help="Target upstream repo, owner/name.")
    parser.add_argument("--pr", type=int, default=0, help="Pull request number to audit.")
    parser.add_argument("--submission-name", help="Override submission directory name.")
    parser.add_argument("--work-dir", help="Reusable working directory for clone/download artifacts.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep the temporary clone/download directory.")
    parser.add_argument("--expected-archive-sha256", help="Expected release archive SHA-256.")
    parser.add_argument("--expected-archive-bytes", type=int, help="Expected release archive byte count.")
    parser.add_argument("--inflate-smoke", action="store_true", help="Run inflate.sh on the downloaded archive.")
    parser.add_argument("--expected-output-sha256", help="Expected 0.raw SHA-256 for inflate smoke.")
    parser.add_argument("--video-name", default="0.mkv", help="Video name to place in file_list for inflate smoke.")
    parser.add_argument("--python-bin", help="Python interpreter exported as PACT_PYTHON_BIN for inflate smoke.")
    parser.add_argument("--encoder-rebuild", action="store_true", help="Run an optional encoder rebuild command.")
    parser.add_argument(
        "--encoder-rebuild-command-json",
        help="Path to JSON list of argv strings for the optional rebuild command.",
    )
    parser.add_argument("--command-timeout-s", type=float, default=120.0)
    parser.add_argument("--network-timeout-s", type=float, default=30.0)
    parser.add_argument("--output-json", help="Write audit JSON to this path.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="Run a no-network parser/self-test.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        result = self_test_result()
    else:
        if args.pr <= 0:
            raise SystemExit("--pr is required unless --self-test is used")
        result = audit_public_submission_pr(config_from_args(args))

    payload = result.as_dict()
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_text_report(result))
    return 0 if result.overall_clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
