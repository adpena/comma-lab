#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate expensive CPU replay from an HPRC MLX scorer-response prefilter."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY  # noqa: E402

HPRC_MLX_PREFILTER_LOCAL_REPLAY_GATE_SCHEMA = "hprc_mlx_prefilter_local_replay_gate.v1"
HPRC_PROFILE_SCHEMA = "hprc_mlx_component_neutralization_profile.v1"
MLX_AXIS_TAG = "[macOS-MLX research-signal]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", required=True, type=Path)
    parser.add_argument("--auth-frontier-score", type=float)
    parser.add_argument("--local-baseline-score", type=float)
    parser.add_argument("--min-local-improvement", type=float, default=0.0)
    parser.add_argument(
        "--max-mlx-score-for-local-replay",
        type=float,
        default=0.5,
        help=(
            "Hard advisory demotion threshold. HPRC candidates with full-video "
            "MLX score above this are not competitive enough to spend local CPU replay."
        ),
    )
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--success-on-blocked",
        action="store_true",
        help="Return 0 after writing a blocked/negative report.",
    )
    return parser


def build_hprc_mlx_prefilter_local_replay_gate(
    *,
    profile: dict[str, Any],
    profile_path: str | Path,
    auth_frontier_score: float | None,
    local_baseline_score: float | None,
    min_local_improvement: float = 0.0,
    max_mlx_score_for_local_replay: float | None = 0.5,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings = ["mlx_axis_is_advisory_cpu_replay_still_required_for_survivors"]
    if profile.get("schema") != HPRC_PROFILE_SCHEMA:
        blockers.append(f"profile_schema_unsupported:{profile.get('schema')!r}")
    scope = profile.get("scope_status")
    if not isinstance(scope, dict) or scope.get("full_video") != "executed":
        blockers.append("mlx_prefilter_not_full_video")
    if int(profile.get("scorer_batch_pairs") or 1) != 1:
        blockers.append("mlx_prefilter_batch_shape_research_signal_not_singleton")
    baseline_response_path = _baseline_response_path(profile)
    baseline_payload: dict[str, Any] | None = None
    if baseline_response_path is None:
        blockers.append("baseline_mlx_response_path_missing")
    else:
        try:
            baseline_payload = _load_json_object(baseline_response_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"baseline_mlx_response_unreadable:{exc}")
    mlx_score = _finite_float(
        None if baseline_payload is None else baseline_payload.get("canonical_score")
    )
    if mlx_score is None and baseline_payload is not None:
        mlx_score = _finite_float(baseline_payload.get("score_recomputed_from_components"))
    if mlx_score is None:
        blockers.append("baseline_mlx_score_missing_or_nonfinite")
    axis = str(
        (baseline_payload or {}).get("axis_tag")
        or (baseline_payload or {}).get("local_axis")
        or MLX_AXIS_TAG
    )
    if axis != MLX_AXIS_TAG:
        blockers.append(f"mlx_axis_mismatch:{axis or 'missing'}")

    margin = max(float(min_local_improvement), 0.0)
    targets = [
        float(value) - margin
        for value in (auth_frontier_score, local_baseline_score)
        if value is not None and math.isfinite(float(value))
    ]
    threshold = min(targets) if targets else None
    if threshold is None:
        blockers.append("mlx_prefilter_target_missing")
    if mlx_score is not None and threshold is not None and not mlx_score < threshold:
        blockers.append("mlx_score_not_below_target")
    hard_demote = _finite_float(max_mlx_score_for_local_replay)
    if mlx_score is not None and hard_demote is not None and mlx_score > hard_demote:
        blockers.append("mlx_score_above_hard_demote_threshold")

    recommended = not blockers
    return {
        "schema": HPRC_MLX_PREFILTER_LOCAL_REPLAY_GATE_SCHEMA,
        "profile_path": str(profile_path),
        "baseline_mlx_response_path": None
        if baseline_response_path is None
        else str(baseline_response_path),
        "axis_tag": axis,
        "mlx_score_estimate": mlx_score,
        "auth_frontier_score": auth_frontier_score,
        "local_baseline_score": local_baseline_score,
        "min_local_improvement": float(min_local_improvement),
        "max_mlx_score_for_local_replay": hard_demote,
        "mlx_gate_threshold": threshold,
        "local_replay_recommended": recommended,
        "next_required_action": (
            "run_local_cpu_replay"
            if recommended
            else "skip_cpu_replay_until_mlx_prefilter_improves"
        ),
        "blockers": blockers,
        "warnings": warnings,
        **FALSE_AUTHORITY,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = _load_json_object(args.profile_json)
    report = build_hprc_mlx_prefilter_local_replay_gate(
        profile=profile,
        profile_path=args.profile_json,
        auth_frontier_score=args.auth_frontier_score,
        local_baseline_score=args.local_baseline_score,
        min_local_improvement=float(args.min_local_improvement),
        max_mlx_score_for_local_replay=args.max_mlx_score_for_local_replay,
    )
    _write_json(args.out_json, report, allow_overwrite=bool(args.allow_overwrite))
    print(json.dumps({**report, "report_path": args.out_json.as_posix()}, sort_keys=True))
    return 0 if report["local_replay_recommended"] or args.success_on_blocked else 2


def _baseline_response_path(profile: dict[str, Any]) -> Path | None:
    rows = profile.get("variant_rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("variant_id") == "baseline":
            value = row.get("mlx_response")
            if isinstance(value, str) and value:
                return Path(value)
    return None


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


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
        print(f"gate_hprc_mlx_prefilter_for_local_replay failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
