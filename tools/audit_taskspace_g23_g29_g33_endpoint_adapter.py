#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the strict G36 endpoint-adapter readiness receipt."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.witness_control.taskspace_g23_g29_g33_endpoint_adapter import (  # noqa: E402
    G36G29PreflightEvidenceBytesV1,
    audit_g36_endpoint_adapter_readiness,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-receipt", type=Path, required=True)
    parser.add_argument("--placement-receipt", type=Path, required=True)
    parser.add_argument("--decoder-abi-receipt", type=Path, required=True)
    parser.add_argument("--source-audit-receipt", type=Path, action="append", required=True)
    parser.add_argument("--dependency-discovery-receipt", type=Path, required=True)
    parser.add_argument("--execution-readiness-receipt", type=Path, required=True)
    parser.add_argument("--public-evaluator-execution-receipt", type=Path)
    parser.add_argument(
        "--dynamic-pointer",
        type=Path,
        default=REPO_ROOT / ".omx/state/canonical_frontier_pointer.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read(path: Path) -> bytes:
    return path.resolve(strict=True).read_bytes()


def _atomic_write(path: Path, payload: bytes) -> None:
    resolved = path.resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = _parser().parse_args()
    evidence = G36G29PreflightEvidenceBytesV1(
        compile_receipt_bytes=_read(args.compile_receipt),
        placement_receipt_bytes=_read(args.placement_receipt),
        decoder_abi_receipt_bytes=_read(args.decoder_abi_receipt),
        generic_source_audit_receipt_bytes=tuple(_read(path) for path in args.source_audit_receipt),
        dependency_discovery_receipt_bytes=_read(args.dependency_discovery_receipt),
        execution_readiness_receipt_bytes=_read(args.execution_readiness_receipt),
        public_evaluator_execution_receipt_bytes=(
            None if args.public_evaluator_execution_receipt is None else _read(args.public_evaluator_execution_receipt)
        ),
    )
    receipt = audit_g36_endpoint_adapter_readiness(
        evidence,
        dynamic_pointer_bytes=_read(args.dynamic_pointer),
    )
    _atomic_write(args.output, receipt.to_receipt_bytes())
    print(receipt.identity_sha256)
    print(args.output.resolve(strict=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
