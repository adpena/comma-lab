#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check production contract for local MLX scorer-response artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.local_acceleration.mlx_production_contract import (
    build_mlx_scorer_production_contract_manifest,
    load_json_object,
    write_production_contract_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-stability", type=Path, default=None)
    parser.add_argument("--batch-invariance", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--no-require-cache-identity",
        action="store_true",
        help="Do not fail if cache identity custody fields are absent.",
    )
    parser.add_argument(
        "--no-require-profile-stability",
        action="store_true",
        help="Advisory/dev mode only: do not fail if profile-stability gate is absent.",
    )
    parser.add_argument(
        "--no-require-batch-invariance",
        action="store_true",
        help="Advisory/dev mode only: do not fail if batch-invariance gate is absent.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = build_mlx_scorer_production_contract_manifest(
        load_json_object(args.response),
        profile_stability=(
            load_json_object(args.profile_stability)
            if args.profile_stability is not None
            else None
        ),
        batch_invariance=(
            load_json_object(args.batch_invariance)
            if args.batch_invariance is not None
            else None
        ),
        run_id=args.run_id,
        require_cache_identity=not args.no_require_cache_identity,
        require_profile_stability=not args.no_require_profile_stability,
        require_batch_invariance=not args.no_require_batch_invariance,
    )
    write_production_contract_manifest(manifest, args.output)
    print(json.dumps({"passed": manifest["passed"], "verdict": manifest["verdict"]}, sort_keys=True))
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
