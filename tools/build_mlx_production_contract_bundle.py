#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a strict non-authoritative MLX production-contract bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.local_acceleration.mlx_production_contract import (
    build_mlx_scorer_production_contract_bundle_manifest,
    load_json_object,
    write_production_contract_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        action="append",
        default=[],
        type=Path,
        help="Strict MLX production contract JSON path. May repeat.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    contracts = [load_json_object(path) for path in args.contract]
    manifest = build_mlx_scorer_production_contract_bundle_manifest(
        contracts,
        run_id=args.run_id,
        producer="tools.build_mlx_production_contract_bundle",
    )
    write_production_contract_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": manifest["passed"],
                "verdict": manifest["verdict"],
                "contract_count": manifest["summary"]["contract_count"],
                "strict_contract_count": manifest["summary"]["strict_contract_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
