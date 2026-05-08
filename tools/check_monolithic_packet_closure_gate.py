#!/usr/bin/env python3
"""Fail-closed exact-dispatch gate for monolithic packet candidates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_entropy_frontier_selector import ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES  # noqa: E402
from tac.monolithic_packet_closure_gate import build_monolithic_packet_closure_gate  # noqa: E402


class MonolithicPacketClosureGateCliError(ValueError):
    """Raised when the closure-gate CLI cannot load requested inputs."""


def build_gate_from_paths(
    *,
    candidate_manifest_path: Path,
    runtime_proof_json: Path | None = None,
    lane_claim_json: Path | None = None,
    dry_run: bool = False,
    active_rate_only_floor_archive_bytes: int | None = ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
    expected_lane_id: str | None = None,
    expected_instance_job_id: str | None = None,
) -> dict[str, Any]:
    """Load JSON files and return the reusable monolithic closure gate.

    ``expected_lane_id`` / ``expected_instance_job_id`` bind the supplied
    lane claim to the candidate's intended dispatch (Level-2 dispatch
    custody, codex HIGH finding #2 2026-05-08). When omitted, the
    binding falls back to ``candidate_manifest["lane_claim"]`` (if
    present); otherwise the gate refuses readiness.
    """

    manifest = _load_json_object(candidate_manifest_path, label="candidate manifest")
    runtime_proof = (
        _load_json_object(runtime_proof_json, label="runtime proof")
        if runtime_proof_json is not None
        else None
    )
    lane_claim = (
        _load_json_object(lane_claim_json, label="lane claim")
        if lane_claim_json is not None
        else None
    )
    gate = build_monolithic_packet_closure_gate(
        manifest,
        runtime_proof=runtime_proof,
        lane_claim=lane_claim,
        dry_run=dry_run,
        active_rate_only_floor_archive_bytes=active_rate_only_floor_archive_bytes,
        expected_lane_id=expected_lane_id,
        expected_instance_job_id=expected_instance_job_id,
    )
    return {
        **gate,
        "input_paths": {
            "candidate_manifest": str(candidate_manifest_path),
            "runtime_proof_json": str(runtime_proof_json) if runtime_proof_json is not None else None,
            "lane_claim_json": str(lane_claim_json) if lane_claim_json is not None else None,
        },
        "cli_expected_lane_id": expected_lane_id,
        "cli_expected_instance_job_id": expected_instance_job_id,
    }


def _load_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise MonolithicPacketClosureGateCliError(f"{label} at {path} must contain a JSON object")
    return payload


def dumps_json(payload: Mapping[str, Any]) -> str:
    """Return stable pretty JSON."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--runtime-proof-json", type=Path)
    parser.add_argument("--lane-claim-json", type=Path)
    parser.add_argument(
        "--active-rate-only-floor-archive-bytes",
        type=int,
        default=ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Allow missing lane-claim proof for closure review; never authorizes dispatch.",
    )
    parser.add_argument(
        "--expected-lane-id",
        type=str,
        default=None,
        help=(
            "Bind the supplied lane claim to this lane id; refuses readiness "
            "if the claim's lane_id does not match (Level-2 dispatch custody)."
        ),
    )
    parser.add_argument(
        "--expected-instance-job-id",
        type=str,
        default=None,
        help=(
            "Bind the supplied lane claim to this instance/job id; refuses "
            "readiness if the claim's instance_job_id does not match."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload = build_gate_from_paths(
            candidate_manifest_path=args.candidate_manifest,
            runtime_proof_json=args.runtime_proof_json,
            lane_claim_json=args.lane_claim_json,
            dry_run=args.dry_run,
            active_rate_only_floor_archive_bytes=args.active_rate_only_floor_archive_bytes,
            expected_lane_id=args.expected_lane_id,
            expected_instance_job_id=args.expected_instance_job_id,
        )
    except (MonolithicPacketClosureGateCliError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"monolithic packet closure gate failed: {exc}") from None

    text = dumps_json(payload)
    if args.json_out is None:
        print(text, end="")
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    if args.fail_if_not_ready and payload["ready_for_exact_eval_dispatch"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
