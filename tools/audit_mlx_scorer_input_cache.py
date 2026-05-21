#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit an MLX scorer-input cache against auth-eval custody."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.local_acceleration.mlx_cache_audit import (
    audit_mlx_scorer_input_cache_against_auth_eval,
    write_cache_audit,
)
from tac.local_acceleration.mlx_scorer_fidelity import load_json_object


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", required=True, type=Path)
    parser.add_argument("--auth-eval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-pair-count", type=int, default=None)
    parser.add_argument(
        "--reference-cache-manifest",
        type=Path,
        default=None,
        help=(
            "Independent scorer-input hash manifest from the auth-side raw surface. "
            "Required when the auth-eval JSON predates scorer_input_cache_hash_manifest provenance."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    audit = audit_mlx_scorer_input_cache_against_auth_eval(
        load_json_object(args.cache_manifest),
        load_json_object(args.auth_eval),
        expected_pair_count=args.expected_pair_count,
        reference_cache_manifest=(
            load_json_object(args.reference_cache_manifest)
            if args.reference_cache_manifest is not None
            else None
        ),
    )
    write_cache_audit(audit, args.output)
    print(json.dumps({"passed": audit["passed"], "verdict": audit["verdict"]}, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
