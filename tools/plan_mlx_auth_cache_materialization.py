#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan the next auth-cache materialization step for MLX local acceleration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.local_acceleration.mlx_auth_cache_materialization import (
    build_mlx_auth_cache_materialization_plan,
    load_json_object,
    write_materialization_plan,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-auth-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--production-contract", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    plan = build_mlx_auth_cache_materialization_plan(
        load_json_object(args.cache_auth_audit),
        production_contract=(
            load_json_object(args.production_contract)
            if args.production_contract is not None
            else None
        ),
        cache_audit_path=str(args.cache_auth_audit),
        production_contract_path=(
            str(args.production_contract) if args.production_contract is not None else None
        ),
        run_id=args.run_id,
    )
    write_materialization_plan(plan, args.output)
    print(
        json.dumps(
            {
                "passed": plan["passed"],
                "verdict": plan["verdict"],
                "next_materialization_action": plan["next_materialization_action"],
            },
            sort_keys=True,
        )
    )
    return 0 if plan["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
