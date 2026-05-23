#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit an MLX scorer-input cache against a local CPU advisory raw surface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tac.local_acceleration.mlx_cache_audit import (
    audit_mlx_scorer_input_cache_against_local_cpu_advisory,
    write_cache_audit,
)
from tac.local_acceleration.mlx_scorer_fidelity import load_json_object


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", required=True, type=Path)
    parser.add_argument("--local-cpu-advisory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-pair-count", type=int, default=600)
    parser.add_argument(
        "--stamp-cache-manifest-on-pass",
        action="store_true",
        help=(
            "If the audit passes, stamp the cache manifest for local CPU-advisory "
            "debug runs. This does not make the cache auth-axis transfer eligible."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    audit = audit_mlx_scorer_input_cache_against_local_cpu_advisory(
        load_json_object(args.cache_manifest),
        load_json_object(args.local_cpu_advisory),
        expected_pair_count=args.expected_pair_count,
    )
    write_cache_audit(audit, args.output)
    if args.stamp_cache_manifest_on_pass:
        if audit["passed"] is not True:
            raise SystemExit("refusing to stamp cache manifest because audit did not pass")
        _stamp_cache_manifest(args.cache_manifest, args.output, args.local_cpu_advisory, audit)
    print(json.dumps({"passed": audit["passed"], "verdict": audit["verdict"]}, sort_keys=True))
    return 0 if audit["passed"] else 2


def _stamp_cache_manifest(
    cache_manifest: Path,
    audit_output: Path,
    local_cpu_advisory: Path,
    audit: dict,
) -> None:
    manifest = load_json_object(cache_manifest)
    manifest["eligible_for_local_mlx_local_advisory_debug"] = True
    manifest["eligible_for_local_mlx_transfer_calibration"] = False
    manifest["local_cpu_advisory_cache_identity_audit"] = {
        "schema_version": audit.get("schema_version"),
        "path": str(audit_output.resolve()),
        "sha256": _file_sha256(audit_output),
        "verdict": audit.get("verdict"),
        "passed": True,
        "local_cpu_advisory_path": str(local_cpu_advisory.resolve()),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
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
