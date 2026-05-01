#!/usr/bin/env python3
"""Scan for the 2026-04-30 Lightning PyPI compromise indicators.

This wrapper is intentionally narrow: it reuses the fail-closed TAC preflight
guard, avoids importing the PyPI ``lightning`` package, and emits a portable
JSON record suitable for local, Lightning Studio, Modal, or Vast harvest logs.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import socket
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.preflight import check_no_compromised_lightning_supply_chain  # noqa: E402


PACKAGE_NAMES = ("lightning", "pytorch-lightning", "lightning-sdk", "lightning_sdk")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _site_packages(items: list[str]) -> list[Path] | None:
    if not items:
        return None
    return [Path(item).expanduser().resolve() for item in items]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--site-packages",
        action="append",
        default=[],
        help="Explicit site-packages root to scan; repeatable. Defaults to active env roots.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if violations are found.")
    parser.add_argument("--quiet", action="store_true", help="Suppress human-readable preflight output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    violations = check_no_compromised_lightning_supply_chain(
        repo_root=repo_root,
        site_packages_roots=_site_packages(args.site_packages),
        strict=False,
        verbose=not args.quiet,
    )
    payload = {
        "schema_version": 1,
        "tool": "scripts/scan_lightning_supply_chain.py",
        "recorded_at_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "repo_root": str(repo_root),
        "python": sys.executable,
        "package_versions": _package_versions(),
        "violation_count": len(violations),
        "violations": violations,
        "status": "FAIL" if violations else "OK",
        "strict": bool(args.strict),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    print(text, end="")
    if violations and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
