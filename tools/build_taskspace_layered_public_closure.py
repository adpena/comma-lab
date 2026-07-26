#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build or promote the typed G55 public selected-plane archive closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _path in (REPO_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.witness_dsl.taskspace_layered_public_closure_v1 import (  # noqa: E402
    ClosureError,
    build_preview,
    promote,
    read_json,
    sha256_file,
    stage_exact_eval,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--auth-receipt", type=Path)
    parser.add_argument("--auth-receipt-sha256")
    parser.add_argument(
        "--stage-exact-eval",
        action="store_true",
        help=("materialize archive.zip for upstream/evaluate.sh without making a promotion or score claim"),
    )
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        receipt = build_preview(config)
        if args.stage_exact_eval:
            staged_archive, staging_receipt = stage_exact_eval(
                Path(receipt["archive_preview"]["path"]),
                receipt,
            )
            receipt["exact_eval_staged_archive_path"] = str(staged_archive)
            receipt["exact_eval_staging_receipt"] = staging_receipt
        if args.auth_receipt is not None:
            if not args.auth_receipt_sha256:
                raise ClosureError("--auth-receipt-sha256 is required for promotion")
            build_receipt_path = Path(receipt["receipt_path"])
            build_receipt = read_json(
                build_receipt_path,
                sha256_file(build_receipt_path),
                "G55 build receipt",
            )
            archive_path = promote(
                Path(receipt["archive_preview"]["path"]),
                build_receipt,
                args.auth_receipt,
                args.auth_receipt_sha256,
                repo_root=Path(config.get("repo_root", ".")),
            )
            receipt["promoted_archive_path"] = str(archive_path)
            receipt["promoted_archive_sha256"] = sha256_file(archive_path)
    except (ClosureError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
