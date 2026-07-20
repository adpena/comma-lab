#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit the fail-closed M1 positive-band prerequisite custody audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.prereq_surfaces import (  # noqa: E402
    audit_m1_positive_band_prerequisites,
)

DEFAULT_MANIFESTS = (
    Path(
        "/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/"
        "chunk_000_010_024_composed/manifest.json"
    ),
    Path(
        "/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/"
        "chunk_012_017_019_023_025_composed/manifest.json"
    ),
)
DEFAULT_PROTOTYPE = Path(
    ".omx/research/prereq_surfaces_flush_20260720/"
    "surface_2_rank4_prototype_bank.json"
)
DEFAULT_CANDIDATE = Path(
    "/Volumes/VertigoDataTier/pact/evidence/"
    "r2b_sparse_target_selection_20260720T1621Z/receipt.json"
)
DEFAULT_OUTPUT = Path(
    ".omx/research/prereq_surfaces_flush_20260720/"
    "surface_4_m1_positive_band_blocker.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vjp-manifest", action="append", type=Path)
    parser.add_argument("--prototype-receipt", type=Path, default=DEFAULT_PROTOTYPE)
    parser.add_argument("--candidate-receipt", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--trust-manifest-sidecar-hashes",
        action="store_true",
        help="do not rehash sidecar bytes; receipt records the weaker audit mode",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifests = tuple(args.vjp_manifest or DEFAULT_MANIFESTS)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite preserved readiness receipt: {output}")
    if output in (Path("/tmp"), Path("/private/tmp"), Path("/var/tmp")) or any(
        root in output.parents for root in (Path("/tmp"), Path("/private/tmp"), Path("/var/tmp"))
    ):
        raise SystemExit("output must be durable and may not be a temporary directory")
    receipt = audit_m1_positive_band_prerequisites(
        manifests,
        args.prototype_receipt,
        args.candidate_receipt,
        verify_sidecar_bytes=not args.trust_manifest_sidecar_hashes,
    )
    _atomic_json(output, receipt)
    print(f"{output}\t{_sha256_file(output)}")
    bundle_path = output.parent / "manifest.json"
    if bundle_path.is_file():
        bundle = json.loads(bundle_path.read_text(encoding="ascii"))
        if bundle.get("schema") != "prereq_surfaces_flush_receipt_manifest.v1":
            raise SystemExit("refusing to update mismatched prerequisite receipt manifest")
        receipts = bundle.get("receipts")
        if not isinstance(receipts, dict) or output.name in receipts:
            raise SystemExit("refusing duplicate surface-4 manifest record")
        receipts[output.name] = {
            "sha256": _sha256_file(output),
            "bytes": output.stat().st_size,
        }
        bundle["surface_4_auditor_sha256"] = _sha256_file(Path(__file__).resolve())
        _atomic_json(bundle_path, bundle)
        print(f"{bundle_path}\t{_sha256_file(bundle_path)}")
    return 0 if receipt["ready_to_assemble"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
