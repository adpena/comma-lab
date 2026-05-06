"""Fail-closed runtime skeleton for categorical payload candidates.

This module is intentionally not a contest decoder. It is packaged as a
charged archive member by the categorical payload builder so local manifests can
prove that inflate-time state is archive-contained while decode/re-encode and
runtime-output parity remain explicit blockers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_MEMBERS = (
    "categorical_payload.bin",
    "class_codebook.json",
    "runtime_consumer_proof_skeleton.json",
)


class RuntimeSkeletonError(RuntimeError):
    """Raised when the charged-member skeleton contract is not satisfied."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_charged_members(archive_root: str | Path) -> dict[str, Any]:
    """Verify required charged members without loading any uncharged sidecars."""

    root = Path(archive_root)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in REQUIRED_MEMBERS:
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        records.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if missing:
        raise RuntimeSkeletonError("missing charged runtime member(s): " + ", ".join(missing))
    return {
        "schema_version": 1,
        "kind": "categorical_runtime_skeleton_member_check",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "charged_members_verified": records,
        "runtime_output_parity_proven": False,
        "dispatch_blockers": [
            "categorical_runtime_skeleton_not_a_decoder",
            "decode_reencode_parity_missing",
            "runtime_output_parity_missing",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        payload = verify_charged_members(args.archive_root)
    except RuntimeSkeletonError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True), end="\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
