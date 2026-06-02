#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build the no-authority NeRV modelsize-to-byte-cap planning curve."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_modelsize_archive_curve import (  # noqa: E402
    build_modelsize_archive_curve,
    parse_byte_caps,
)
from tac.repo_io import write_json_artifact  # noqa: E402


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/nerv_modelsize_archive_curve_{stamp}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--byte-cap",
        action="append",
        help="Repeated or comma-separated byte caps. Defaults to 36k..PR95 cap.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing --out only with --expected-existing-sha256.",
    )
    parser.add_argument("--expected-existing-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    byte_caps = parse_byte_caps(args.byte_cap)
    payload = build_modelsize_archive_curve(byte_caps=byte_caps)
    out_path = args.out or _default_out()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    result = write_json_artifact(
        out_path,
        payload,
        allow_overwrite=args.allow_overwrite,
        expected_existing_sha256=args.expected_existing_sha256,
    )
    print("[NeRV modelsize archive curve] false-authority")
    print(f"  verdict: {payload['verdict']}")
    print(f"  rows: {len(payload['curve_rows'])}")
    print(f"  byte_caps: {payload['byte_caps']}")
    print(f"  blockers: {len(payload['blockers'])}")
    print(f"  wrote {result.path} ({result.bytes_written} bytes sha256={result.sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
