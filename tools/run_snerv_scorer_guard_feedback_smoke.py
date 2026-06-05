#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run a tiny SNeRV scorer guard smoke and emit queue-consumable feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_candidate_feedback import (  # noqa: E402
    write_nerv_candidate_feedback_files,
)
from tac.repo_io import write_json_artifact  # noqa: E402
from tools.run_snerv_scorer_tether_smoke import (  # noqa: E402
    FALSE_AUTHORITY,
    run_snerv_scorer_tether_smoke,
)

SCHEMA = "snerv_scorer_guard_feedback_smoke.v1"
RUNNER_REPORT_SCHEMA = "snerv_scorer_guard_feedback_smoke_runner_report.v1"
DEFAULT_OUTPUT_ROOT = Path("/Volumes/VertigoDataTier/pact")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT
        / f"snerv_scorer_guard_feedback_smoke_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir = output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke = run_snerv_scorer_tether_smoke(steps=int(args.steps))
    smoke_result = write_json_artifact(
        output_dir / "snerv_scorer_tether_guard_smoke.json",
        smoke,
    )
    runner_report = _runner_report_from_smoke(
        smoke,
        smoke_path=smoke_result.path,
        smoke_sha256=smoke_result.sha256,
        candidate_id=str(args.candidate_id),
    )
    runner_result = write_json_artifact(
        output_dir / "snerv_scorer_guard_feedback_runner_report.json",
        runner_report,
    )
    feedback = write_nerv_candidate_feedback_files(
        runner_report=runner_report,
        output_dir=output_dir,
        source_report_path=runner_result.path,
    )
    row_path = Path(str(feedback["row_path"]))
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "output_dir": output_dir.as_posix(),
        "smoke_report_path": smoke_result.path,
        "smoke_report_sha256": smoke_result.sha256,
        "runner_report_path": runner_result.path,
        "runner_report_sha256": runner_result.sha256,
        "candidate_feedback_row_path": row_path.as_posix(),
        "candidate_feedback_row_sha256": _sha256_file(row_path),
        "candidate_feedback_ledger_path": feedback["ledger_path"],
        "candidate_id": str(args.candidate_id),
        "candidate_feedback_guard_proof": feedback["row"].get(
            "snerv_scorer_input_distribution_guard_proof"
        ),
        "candidate_feedback_scorer_domain_tether_health": feedback["row"].get(
            "snerv_scorer_domain_tether_health"
        ),
        "passed": bool(
            smoke.get("passed") is True
            and feedback["row"].get(
                "snerv_scorer_input_distribution_guard_proof_passed"
            )
            is True
            and feedback["row"].get("snerv_scorer_domain_tether_passed") is True
        ),
        "smoke_passed": smoke.get("passed") is True,
        "guard_proof_passed": feedback["row"].get(
            "snerv_scorer_input_distribution_guard_proof_passed"
        )
        is True,
        "scorer_domain_tether_passed": feedback["row"].get(
            "snerv_scorer_domain_tether_passed"
        )
        is True,
        "blockers": _blockers(smoke, feedback["row"]),
        **FALSE_AUTHORITY,
    }
    manifest_result = write_json_artifact(
        output_dir / "snerv_scorer_guard_feedback_smoke_manifest.json",
        manifest,
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "manifest_path": manifest_result.path,
                "manifest_sha256": manifest_result.sha256,
                "candidate_feedback_row_path": manifest["candidate_feedback_row_path"],
                "candidate_feedback_row_sha256": manifest[
                    "candidate_feedback_row_sha256"
                ],
                "passed": manifest["passed"],
                "blockers": manifest["blockers"],
                **FALSE_AUTHORITY,
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--candidate-id",
        default="snerv_scorer_guard_feedback_smoke",
    )
    return parser


def _runner_report_from_smoke(
    smoke: dict[str, Any],
    *,
    smoke_path: str,
    smoke_sha256: str,
    candidate_id: str,
) -> dict[str, Any]:
    telemetry_contract = dict(
        smoke.get("score_aware_long_training_telemetry_contract") or {}
    )
    required_control_contract = dict(
        smoke.get("score_aware_long_training_required_control_contract") or {}
    )
    gate = {
        "schema": "snerv_score_aware_long_training_scorer_tether_gate.v1",
        "required": True,
        "executed": True,
        "passed": smoke.get("passed") is True,
        "steps": smoke.get("steps"),
        "smoke_schema": smoke.get("schema"),
        "smoke_report_path": smoke_path,
        "smoke_report_sha256": smoke_sha256,
        "smoke_report": smoke,
        "blockers": [
            str(blocker) for blocker in smoke.get("blockers") or [] if str(blocker)
        ],
        **FALSE_AUTHORITY,
    }
    return {
        "schema": RUNNER_REPORT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "execute_family": "snerv",
        "family": "snerv",
        "candidate_id": candidate_id,
        "num_pairs": 8,
        "score_aware_training": {
            "schema": "compact_snerv_native_mlx_guard_smoke.v1",
            "candidate_id": candidate_id,
            "scorer_tether_smoke_gate": gate,
            "training_telemetry_contract": telemetry_contract,
            "required_control_contract": required_control_contract,
            "score_aware_long_training_scorer_input_distribution_guard_bound": (
                True
            ),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "score_aware_long_training_telemetry_contract": telemetry_contract,
        "score_aware_long_training_required_control_contract": (
            required_control_contract
        ),
        "snerv_scorer_tether_smoke_gate": gate,
        "source_smoke_report_path": smoke_path,
        "source_smoke_report_sha256": smoke_sha256,
        "blockers": [],
        **FALSE_AUTHORITY,
    }


def _blockers(smoke: dict[str, Any], row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if smoke.get("passed") is not True:
        blockers.append("snerv_scorer_guard_feedback_smoke_failed")
        blockers.extend(str(blocker) for blocker in smoke.get("blockers") or [])
    if row.get("snerv_scorer_domain_tether_passed") is not True:
        blockers.extend(
            str(blocker)
            for blocker in row.get("snerv_scorer_domain_tether_blockers") or []
        )
    if row.get("snerv_scorer_input_distribution_guard_proof_passed") is not True:
        blockers.extend(
            str(blocker)
            for blocker in row.get("snerv_scorer_input_distribution_guard_blockers")
            or []
        )
    return _ordered_unique(blockers)


def _ordered_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
