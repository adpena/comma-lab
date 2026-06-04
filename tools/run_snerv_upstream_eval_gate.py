#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the upstream evaluator on a materialized SNeRV submission bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.evaluate import evaluate_external_submission_dir  # noqa: E402
from tac.analysis.snerv_upstream_eval_feedback import (  # noqa: E402
    write_snerv_upstream_eval_candidate_feedback,
)
from tac.repo_io import read_json, write_json  # noqa: E402

SCHEMA = "snerv_upstream_eval_gate.v1"
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_snerv_upstream_eval_gate(
        bundle_json=args.bundle_json,
        submission_dir=args.submission_dir,
        upstream_root=args.upstream_root,
        artifact_dir=args.artifact_dir,
        output_json=args.output_json,
        candidate_feedback_json=args.candidate_feedback_json,
        device=args.device,
        keep_inflated=bool(args.keep_inflated),
        min_free_bytes=int(args.min_free_bytes),
        require_upstream_venv=not args.no_require_upstream_venv,
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output_json": args.output_json.as_posix(),
                "archive_bytes": report["evaluation"]["archive_zip"]["bytes"],
                "returncode": report["evaluation"]["returncode"],
                "candidate_feedback_json": report["candidate_feedback_row_path"],
                "blockers": report["blockers"],
                **FALSE_AUTHORITY,
            },
            sort_keys=True,
        )
    )
    return 0 if report["evaluation"]["returncode"] == 0 else 1


def run_snerv_upstream_eval_gate(
    *,
    bundle_json: str | Path,
    submission_dir: str | Path | None,
    upstream_root: str | Path | None,
    artifact_dir: str | Path,
    output_json: str | Path,
    candidate_feedback_json: str | Path | None = None,
    device: str = "cpu",
    keep_inflated: bool = False,
    min_free_bytes: int = 5 * 1024 * 1024 * 1024,
    require_upstream_venv: bool = True,
) -> dict[str, Any]:
    bundle_path = Path(bundle_json).expanduser().resolve(strict=False)
    bundle = _load_bundle(bundle_path)
    resolved_submission = _resolve_submission_dir(bundle, explicit=submission_dir)
    artifact_root = Path(artifact_dir).expanduser().resolve(strict=False)
    evaluation = evaluate_external_submission_dir(
        submission_dir=resolved_submission,
        device=device,
        upstream_root=None if upstream_root is None else Path(upstream_root),
        artifact_dir=artifact_root,
        keep_inflated=keep_inflated,
        min_free_bytes=min_free_bytes,
        require_upstream_venv=require_upstream_venv,
    ).to_dict()
    feedback_path = (
        Path(candidate_feedback_json).expanduser().resolve(strict=False)
        if candidate_feedback_json is not None
        else artifact_root / "snerv_upstream_eval_candidate_feedback_row.json"
    )
    blockers = _blockers(bundle=bundle, evaluation=evaluation)
    report = {
        "schema": SCHEMA,
        "operation": "snerv_upstream_data_only_submission_eval_gate",
        "source_bundle_json": {
            "path": bundle_path.as_posix(),
            "schema": bundle.get("schema"),
            "archive_zip": bundle.get("archive_zip"),
            "receiver_proof": bundle.get("receiver_proof"),
        },
        "submission_dir": resolved_submission.as_posix(),
        "artifact_dir": artifact_root.as_posix(),
        "candidate_feedback_row_path": feedback_path.as_posix(),
        "evaluation": evaluation,
        "launchability": {
            "candidate_package_launchable": False,
            "blocked_long_training_rows_must_not_launch": True,
            "reason": "paired contest CPU/CUDA auth eval and compliance gate remain missing",
        },
        "blockers": blockers,
        "next_actions": [
            "feed_upstream_eval_gate_json_into_nerv_campaign_queue",
            "run_paired_contest_cpu_cuda_auth_eval_if_component_score_is_frontier_relevant",
            "materialize_minified_external_runtime_only_if_runtime_source_bytes_are_charged",
        ],
        **FALSE_AUTHORITY,
    }
    write_json(output_json, report)
    write_snerv_upstream_eval_candidate_feedback(
        gate_report=report,
        gate_report_path=output_json,
        output_json=feedback_path,
    )
    return report


def _load_bundle(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"bundle JSON must be an object: {path}")
    if payload.get("schema") != "snerv_upstream_submission_bundle_materialization.v1":
        raise ValueError(f"not a SNeRV upstream bundle materialization: {path}")
    return payload


def _resolve_submission_dir(bundle: dict[str, Any], *, explicit: str | Path | None) -> Path:
    raw = explicit if explicit is not None else bundle.get("output_submission_dir")
    if not isinstance(raw, (str, Path)):
        raise ValueError("submission dir missing from bundle and --submission-dir not supplied")
    submission = Path(raw).expanduser().resolve(strict=False)
    if not submission.is_dir():
        raise FileNotFoundError(f"submission dir not found: {submission}")
    return submission


def _blockers(*, bundle: dict[str, Any], evaluation: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    archive = bundle.get("archive_zip")
    if not isinstance(archive, dict) or archive.get("data_only") is not True:
        blockers.append("snerv_bundle_archive_not_data_only")
    contract = bundle.get("upstream_contest_contract")
    if not isinstance(contract, dict) or contract.get("runtime_source_outside_archive_zip") is not True:
        blockers.append("snerv_upstream_runtime_externalization_contract_missing")
    receiver_proof = bundle.get("receiver_proof")
    if (
        not isinstance(receiver_proof, dict)
        or receiver_proof.get("runtime_consumption_proof_passed") is not True
    ):
        blockers.append("snerv_receiver_proof_missing_or_failed")
    blockers.extend(str(item) for item in evaluation.get("blockers") or [])
    if evaluation.get("returncode") != 0:
        blockers.append("upstream_evaluate_gate_failed")
    blockers.extend(
        [
            "paired_contest_cpu_cuda_auth_eval_missing",
            "pre_submission_compliance_gate_missing",
        ]
    )
    return sorted(dict.fromkeys(blockers))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-json", required=True, type=Path)
    parser.add_argument("--submission-dir", type=Path, default=None)
    parser.add_argument("--upstream-root", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--candidate-feedback-json", type=Path, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--min-free-bytes", type=int, default=5 * 1024 * 1024 * 1024)
    parser.add_argument("--keep-inflated", action="store_true")
    parser.add_argument(
        "--no-require-upstream-venv",
        action="store_true",
        help="use the current Python environment; recorded in the output JSON",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
