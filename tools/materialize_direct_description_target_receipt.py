#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the Task #603 target receipt from existing C1 custody."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402
from tac.optimization.direct_description_real_target_rung0 import (  # noqa: E402
    build_target_plane_receipt,
    write_or_verify_target_receipt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-receipt", type=Path, required=True)
    parser.add_argument("--contest-cpu-receipt", type=Path, required=True)
    parser.add_argument("--contest-cpu-provenance", type=Path, required=True)
    parser.add_argument("--upstream-repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = build_target_plane_receipt(
            prepare_receipt_path=args.prepare_receipt,
            contest_cpu_receipt_path=args.contest_cpu_receipt,
            contest_cpu_provenance_path=args.contest_cpu_provenance,
            repo_root=REPO_ROOT,
            upstream_repo_root=args.upstream_repo_root,
        )
        path = write_or_verify_target_receipt(args.output, receipt)
    except (DirectDescriptionError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"path": str(path), "receipt": receipt.model_dump(mode="json", by_alias=True)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
