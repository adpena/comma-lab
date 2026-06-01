#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate expensive local replay when archive rate alone cannot clear target."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.archive_byte_profile import contest_rate_term  # noqa: E402
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY  # noqa: E402

ARCHIVE_RATE_LOCAL_REPLAY_GATE_SCHEMA = "archive_rate_local_replay_gate.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", required=True, type=Path)
    parser.add_argument("--auth-frontier-score", type=float)
    parser.add_argument("--local-baseline-score", type=float)
    parser.add_argument("--min-local-improvement", type=float, default=0.0)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--success-on-blocked",
        action="store_true",
        help="Return 0 after writing a blocked/negative report.",
    )
    return parser


def build_archive_rate_local_replay_gate(
    *,
    training_result: dict[str, Any],
    training_result_path: str | Path,
    auth_frontier_score: float | None,
    local_baseline_score: float | None,
    min_local_improvement: float = 0.0,
) -> dict[str, Any]:
    artifact = training_result.get("artifact")
    artifact = artifact if isinstance(artifact, dict) else {}
    archive_bytes = _positive_int(artifact.get("archive_bytes"))
    rate_term = None if archive_bytes is None else contest_rate_term(archive_bytes)
    margin = max(float(min_local_improvement), 0.0)
    targets = [
        float(value) - margin
        for value in (auth_frontier_score, local_baseline_score)
        if value is not None and math.isfinite(float(value))
    ]
    threshold = min(targets) if targets else None
    blockers: list[str] = []
    if archive_bytes is None:
        blockers.append("archive_bytes_missing_or_nonpositive")
    if rate_term is None:
        blockers.append("archive_rate_term_missing")
    if threshold is None:
        blockers.append("rate_gate_target_missing")
    if rate_term is not None and threshold is not None and not rate_term < threshold:
        blockers.append("archive_rate_term_not_below_target_before_distortion")
    return {
        "schema": ARCHIVE_RATE_LOCAL_REPLAY_GATE_SCHEMA,
        "training_result_path": str(training_result_path),
        "archive_zip_path": artifact.get("archive_path"),
        "archive_zip_sha256": artifact.get("archive_sha256"),
        "archive_zip_bytes": archive_bytes,
        "archive_rate_term": rate_term,
        "auth_frontier_score": auth_frontier_score,
        "local_baseline_score": local_baseline_score,
        "min_local_improvement": float(min_local_improvement),
        "rate_gate_threshold": threshold,
        "local_replay_recommended": not blockers,
        "blockers": blockers,
        "next_required_action": (
            "run_local_cpu_replay"
            if not blockers
            else "skip_local_replay_until_archive_rate_improves"
        ),
        **FALSE_AUTHORITY,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    training_result = _load_json_object(args.training_result)
    report = build_archive_rate_local_replay_gate(
        training_result=training_result,
        training_result_path=args.training_result,
        auth_frontier_score=args.auth_frontier_score,
        local_baseline_score=args.local_baseline_score,
        min_local_improvement=float(args.min_local_improvement),
    )
    _write_json(args.out_json, report, allow_overwrite=bool(args.allow_overwrite))
    print(json.dumps({**report, "report_path": args.out_json.as_posix()}, sort_keys=True))
    return 0 if report["local_replay_recommended"] or args.success_on_blocked else 2


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _write_json(path: Path, payload: dict[str, Any], *, allow_overwrite: bool) -> None:
    expected = None
    if allow_overwrite and path.is_file():
        from tac.repo_io import sha256_file

        expected = sha256_file(path)
    write_json_artifact(
        path,
        payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected,
    )


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (ArtifactWriteError, ValueError, FileNotFoundError) as exc:
        print(f"gate_archive_rate_for_local_replay failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
