#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit an MLX scorer-input cache against auth-eval custody."""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument(
        "--stamp-cache-manifest-on-pass",
        action="store_true",
        help=(
            "If the audit passes, stamp the audited cache manifest with the "
            "audit path/hash so downstream MLX response tooling can require "
            "PASS_CACHE_AUTH_EVAL_IDENTITY provenance."
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
    if args.stamp_cache_manifest_on_pass:
        if audit["passed"] is not True:
            raise SystemExit("refusing to stamp cache manifest because audit did not pass")
        _stamp_cache_manifest(args.cache_manifest, args.output, args.auth_eval, audit)
    print(json.dumps({"passed": audit["passed"], "verdict": audit["verdict"]}, sort_keys=True))
    return 0 if audit["passed"] else 2


def _stamp_cache_manifest(
    cache_manifest: Path,
    audit_output: Path,
    auth_eval: Path,
    audit: dict,
) -> None:
    manifest = load_json_object(cache_manifest)
    manifest["eligible_for_local_mlx_transfer_calibration"] = True
    manifest["auth_eval_identity_audit"] = {
        "schema_version": audit.get("schema_version"),
        "path": str(audit_output.resolve()),
        "sha256": _file_sha256(audit_output),
        "verdict": audit.get("verdict"),
        "passed": True,
        "identity_residual": audit.get("identity_residual"),
        "auth_eval_path": str(auth_eval.resolve()),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    cache_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
