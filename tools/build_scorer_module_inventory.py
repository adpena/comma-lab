#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build or verify the research-only frozen scorer module inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.optimization.scorer_module_inventory import (
    build_inventory,
    read_and_validate_receipt,
    wrap_receipt,
    write_receipt_once,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--created-at-utc")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        receipt = read_and_validate_receipt(args.output)
    else:
        if args.upstream_root is None or not args.created_at_utc:
            parser.error("--upstream-root and --created-at-utc are required to build")
        receipt = wrap_receipt(
            build_inventory(
                upstream_root=args.upstream_root,
                created_at_utc=args.created_at_utc,
            )
        )
        write_receipt_once(args.output, receipt)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "schema": receipt["body"]["schema"],
                "body_sha256": receipt["body_sha256"],
                "binding_status": receipt["body"]["analytic_binding"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
