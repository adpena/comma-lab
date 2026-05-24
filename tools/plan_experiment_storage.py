#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit a false-authority storage-tier plan for experiment queues."""
from __future__ import annotations

import argparse
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

from comma_lab.operator_storage_waterfall import (  # noqa: E402
    POLICY_ID,
    POLICY_SCHEMA,
    operator_storage_policy_payload,
    storage_preflight_artifact_catalog_metadata,
)
from comma_lab.storage_tiers import (  # noqa: E402
    DEFAULT_RESERVE_FREE_GB,
    DEFAULT_WORKLOAD_SUBDIR,
    StorageTierError,
    parse_storage_tier_specs,
    plan_experiment_storage,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def _resolve_expected(path: str | None) -> str | None:
    if path is None:
        return None
    expected = Path(path).expanduser()
    if not expected.is_absolute():
        expected = REPO_ROOT / expected
    return str(expected.resolve(strict=False))


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    tiers = parse_storage_tier_specs(
        args.storage_tier,
        repo_root=REPO_ROOT,
        reserve_free_gb=args.reserve_free_gb,
        allow_local_disk=args.allow_local_storage_tier,
    )
    plan = plan_experiment_storage(
        tiers,
        workload_subdir=args.workload_subdir,
        requested_bytes=args.requested_bytes,
        min_free_bytes=args.min_free_bytes,
        create=args.create,
    )
    payload = plan.to_dict()
    storage_plan_path = args.storage_plan_path or args.output
    payload["operator_storage_policy"] = operator_storage_policy_payload(
        storage_tier_overrides=tuple(args.storage_tier),
        allow_local_disk=bool(args.allow_local_storage_tier),
        policy_id=args.policy_id,
        policy_schema=args.policy_schema,
    )
    payload["artifact_catalog_metadata"] = storage_preflight_artifact_catalog_metadata(
        policy_id=args.policy_id,
        policy_schema=args.policy_schema,
        storage_plan_path=storage_plan_path,
        cleanup_plan_path=args.cleanup_plan_path,
        journal_path=args.journal_path,
        lifecycle_kind=args.lifecycle_kind,
    )
    expected = _resolve_expected(args.expected_workload_root)
    selected = payload.get("selected_workload_root")
    if isinstance(selected, str):
        selected = str(Path(selected).resolve(strict=False))
        payload["selected_workload_root"] = selected
    matches = expected is None or selected == expected
    payload.update(
        {
            "expected_workload_root": expected,
            "selected_workload_root_matches_expected": matches,
            "storage_required": True,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    )
    blockers = list(payload.get("blockers") or [])
    if payload.get("selected_workload_root") is None:
        blockers.append("selected_workload_root_missing")
    if not matches:
        blockers.append("selected_workload_root_mismatch")
    payload["blockers"] = blockers
    return payload, not blockers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--storage-tier", action="append", default=[], help="name=/path tier override")
    parser.add_argument("--workload-subdir", default=DEFAULT_WORKLOAD_SUBDIR)
    parser.add_argument("--reserve-free-gb", type=float, default=DEFAULT_RESERVE_FREE_GB)
    parser.add_argument("--requested-bytes", type=int, default=0)
    parser.add_argument("--min-free-bytes", type=int, default=0)
    parser.add_argument("--expected-workload-root", default=None)
    parser.add_argument("--policy-id", default=POLICY_ID)
    parser.add_argument("--policy-schema", default=POLICY_SCHEMA)
    parser.add_argument("--storage-plan-path", default=None)
    parser.add_argument("--cleanup-plan-path", default=None)
    parser.add_argument("--journal-path", default=None)
    parser.add_argument("--lifecycle-kind", default="HISTORICAL_PROVENANCE")
    parser.add_argument(
        "--expected-output-sha256",
        default=None,
        help="required SHA-256 of existing --output when replacing a prior plan",
    )
    parser.add_argument("--create", action="store_true", help="create selected workload directories if possible")
    parser.add_argument(
        "--allow-local-storage-tier",
        action="store_true",
        help="allow repo/local-disk fallback; default is fail-closed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload, ok = build_payload(args)
    except StorageTierError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    if args.output:
        try:
            write_json_artifact(
                args.output,
                payload,
                allow_overwrite=True,
                expected_existing_sha256=args.expected_output_sha256,
            )
        except ArtifactWriteError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
