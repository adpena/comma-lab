#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit the strict read-only V15/MS1 coordinate compatibility receipt."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.witness_dsl.v15_ms1_coordinate_compatibility import (  # noqa: E402
    AuditPaths,
    audit_coordinate_compatibility,
    canonical_json,
)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--receipt-only",
        action="store_true",
        help="skip the V15 parse/render contract check; artifact and receipt checks remain strict",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = audit_coordinate_compatibility(
        AuditPaths.canonical(args.repo_root),
        strict_v15_parse=not args.receipt_only,
    )
    payload = canonical_json(receipt) + b"\n"
    if args.output is not None:
        _atomic_write(args.output.resolve(), payload)
    sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
