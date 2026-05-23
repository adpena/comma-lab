#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build and stamp an MLX scorer-input cache from a local CPU advisory run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.mlx_cache_audit import (  # noqa: E402
    audit_mlx_scorer_input_cache_against_local_cpu_advisory,
    write_cache_audit,
)
from tac.local_acceleration.mlx_preprocess import (  # noqa: E402
    load_raw_video_memmap,
    non_overlapping_pair_indices,
    write_scorer_input_cache_from_raw_file,
)

_AUTHORITY_FALSE_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "promotable",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)
_CACHE_FILES = ("manifest.json", "segnet_last_rgb.npy", "posenet_yuv6_pair.npy", "pair_indices.npy")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-cpu-advisory", required=True, type=Path)
    parser.add_argument("--output-cache-dir", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--expected-pair-count", type=int, default=600)
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument(
        "--large-cache-pair-threshold",
        type=int,
        default=64,
        help="Refuse full tensor cache writes above this pair count unless acknowledged.",
    )
    parser.add_argument("--allow-large-tensor-cache", action="store_true")
    parser.add_argument(
        "--stamp-cache-manifest-on-pass",
        action="store_true",
        help=(
            "Stamp the cache manifest with PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY "
            "when the local advisory audit passes."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.expected_pair_count < 1:
        raise SystemExit("--expected-pair-count must be >= 1")
    if args.batch_pairs < 1:
        raise SystemExit("--batch-pairs must be >= 1")
    if args.large_cache_pair_threshold < 1:
        raise SystemExit("--large-cache-pair-threshold must be >= 1")
    _refuse_existing_cache_outputs(args.output_cache_dir)
    if args.audit_output.exists():
        raise SystemExit(f"refusing to overwrite audit artifact: {args.audit_output}")

    advisory = _load_json_object(args.local_cpu_advisory)
    _require_local_cpu_advisory(advisory, args.local_cpu_advisory)
    raw_surface = _raw_surface_from_local_advisory(advisory, args.local_cpu_advisory)
    pair_count = _raw_pair_count(raw_surface["raw_path"])
    if pair_count > args.large_cache_pair_threshold and not args.allow_large_tensor_cache:
        raise SystemExit(
            "refusing full MLX scorer-input tensor cache for "
            f"{pair_count} pairs (> threshold {args.large_cache_pair_threshold}); "
            "pass --allow-large-tensor-cache after confirming disk budget"
        )

    manifest = write_scorer_input_cache_from_raw_file(
        raw_surface["raw_path"],
        args.output_cache_dir,
        archive_sha256=raw_surface["archive_sha256"],
        inflated_outputs_aggregate_sha256=raw_surface["inflated_outputs_aggregate_sha256"],
        batch_pairs=args.batch_pairs,
    )
    if manifest.get("raw_sha256") != raw_surface["raw_sha256"]:
        raise SystemExit(
            "raw SHA mismatch after cache materialization: "
            f"cache={manifest.get('raw_sha256')} advisory={raw_surface['raw_sha256']}"
        )
    audit = audit_mlx_scorer_input_cache_against_local_cpu_advisory(
        manifest,
        advisory,
        expected_pair_count=args.expected_pair_count,
    )
    write_cache_audit(audit, args.audit_output)
    if args.stamp_cache_manifest_on_pass:
        if audit.get("passed") is not True:
            raise SystemExit(
                "refusing to stamp cache manifest because local advisory identity audit failed"
            )
        _stamp_cache_manifest(args.output_cache_dir / "manifest.json", args.audit_output, audit)
    print(
        json.dumps(
            {
                "cache_dir": str(args.output_cache_dir),
                "manifest": str(args.output_cache_dir / "manifest.json"),
                "audit_output": str(args.audit_output),
                "audit_passed": audit["passed"],
                "audit_verdict": audit["verdict"],
                "pair_count": manifest["pair_count"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0 if audit["passed"] else 2


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return payload


def _require_local_cpu_advisory(payload: dict[str, Any], path: Path) -> None:
    if payload.get("score_axis") != "cpu_advisory":
        raise SystemExit(f"{path}: score_axis must be cpu_advisory")
    if payload.get("evidence_semantics") != "non_contest_cpu_auth_eval_advisory":
        raise SystemExit(f"{path}: evidence_semantics must be non_contest_cpu_auth_eval_advisory")
    for field in _AUTHORITY_FALSE_FIELDS:
        if payload.get(field) is not False:
            raise SystemExit(f"{path}: {field} must be exactly false")
    if not isinstance(payload.get("archive_size_bytes"), int) or payload["archive_size_bytes"] <= 0:
        raise SystemExit(f"{path}: archive_size_bytes must be a positive integer")


def _raw_surface_from_local_advisory(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit(f"{path}: provenance must be an object")
    archive_sha = _required_sha(provenance.get("archive_sha256"), f"{path}: provenance.archive_sha256")
    inflated_record = provenance.get("inflated_output_manifest")
    if not isinstance(inflated_record, dict):
        raise SystemExit(f"{path}: provenance.inflated_output_manifest must be an object")
    inflated = inflated_record.get("payload")
    if not isinstance(inflated, dict):
        raise SystemExit(f"{path}: inflated_output_manifest.payload must be an object")
    inflated_dir_value = inflated.get("inflated_dir")
    if not isinstance(inflated_dir_value, str) or not inflated_dir_value.strip():
        raise SystemExit(f"{path}: inflated_output_manifest.payload.inflated_dir missing")
    aggregate_sha = _required_sha(
        inflated.get("aggregate_sha256"),
        f"{path}: inflated_output_manifest.payload.aggregate_sha256",
    )
    files = inflated.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise SystemExit(f"{path}: expected exactly one inflated raw file")
    raw_row = files[0]
    if raw_row.get("exists") is not True:
        raise SystemExit(f"{path}: inflated raw file is not marked exists=true")
    raw_sha = _required_sha(raw_row.get("sha256"), f"{path}: inflated raw sha256")
    relative = raw_row.get("relative_path")
    if not isinstance(relative, str) or not relative.strip():
        raise SystemExit(f"{path}: inflated raw relative_path missing")
    raw_path = Path(inflated_dir_value) / relative
    if not raw_path.is_file():
        raise SystemExit(f"{path}: inflated raw file not found: {raw_path}")
    return {
        "archive_sha256": archive_sha,
        "inflated_outputs_aggregate_sha256": aggregate_sha,
        "raw_sha256": raw_sha,
        "raw_path": raw_path,
    }


def _required_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise SystemExit(f"{label} must be a lowercase sha256 hex digest")
    return value


def _raw_pair_count(raw_path: Path) -> int:
    raw = load_raw_video_memmap(raw_path)
    return len(non_overlapping_pair_indices(raw.shape[0]))


def _refuse_existing_cache_outputs(cache_dir: Path) -> None:
    existing = [name for name in _CACHE_FILES if (cache_dir / name).exists()]
    if existing:
        raise SystemExit(
            f"refusing to overwrite existing MLX cache outputs in {cache_dir}: {existing}"
        )


def _stamp_cache_manifest(manifest_path: Path, audit_output: Path, audit: dict[str, Any]) -> None:
    manifest = _load_json_object(manifest_path)
    manifest["eligible_for_local_mlx_local_advisory_debug"] = True
    manifest["eligible_for_local_mlx_transfer_calibration"] = False
    manifest["local_cpu_advisory_cache_identity_audit"] = {
        "schema_version": audit.get("schema_version"),
        "path": str(audit_output.resolve()),
        "sha256": _file_sha256(audit_output),
        "verdict": audit.get("verdict"),
        "passed": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
