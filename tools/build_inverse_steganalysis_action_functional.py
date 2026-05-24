#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a planning-only inverse-steganalysis action functional."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.inverse_steganalysis_acquisition import (  # noqa: E402
    CONTEST_RATE_SCORE_PER_BYTE,
    InverseSteganalysisAcquisitionError,
    action_atoms_from_byte_shaving_campaign_plan,
    action_atoms_from_byte_shaving_signal_surface,
    action_atoms_from_inverse_scorer_surface,
    build_discrete_scorer_action_functional,
    inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection,
    observations_from_queue_performance_summary,
    paired_exact_auth_calibration_observations_from_review_packets,
)
from tac.optimization.scorer_inverse_decision_surface import (  # noqa: E402
    build_inverse_scorer_decision_surface,
)
from tac.optimization.scorer_response_dataset import (  # noqa: E402
    ScorerResponseDatasetError,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected object for {label}")
    return payload


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _as_rows(payload: Any, *, path: Path, key: str) -> list[dict[str, Any]]:
    rows = payload.get(key) if isinstance(payload, dict) else payload
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SystemExit(f"{path}: expected object/list rows for {key}")
    return [dict(row) for row in rows]


def _render_markdown(payload: dict[str, Any]) -> str:
    bucket = payload.get("water_bucket") or {}
    totals = payload.get("integral_totals") or {}
    lines = [
        "# Inverse-Steganalysis Action Functional",
        "",
        f"- schema: `{payload['schema']}`",
        f"- cells: `{totals.get('cell_count')}`",
        f"- blocked_cells: `{totals.get('blocked_cell_count')}`",
        f"- selected_cells: `{bucket.get('selected_count')}`",
        f"- selected_water_fill_cost_bytes: `{bucket.get('selected_water_fill_cost_bytes')}`",
        f"- selected_expected_score_gain: `{bucket.get('selected_expected_score_gain')}`",
        "",
        "## Math Model",
        f"- representation: `{payload['math_model']['representation']}`",
        f"- stationarity_rule: `{payload['math_model']['stationarity_rule']}`",
        f"- lambda_rate: `{payload['math_model']['lambda_rate']}`",
        "",
        "## Selected Water Buckets",
        "| rank | atom | candidate | scope | component | bytes | expected gain | residual |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(bucket.get("selected_cells") or [], start=1):
        lines.append(
            "| {rank} | `{atom}` | `{candidate}` | `{scope}` | `{component}` | "
            "{bytes_} | `{gain}` | `{residual}` |".format(
                rank=rank,
                atom=row.get("atom_id"),
                candidate=row.get("candidate_id"),
                scope=row.get("scope_axis"),
                component=row.get("component"),
                bytes_=row.get("water_fill_cost_bytes"),
                gain=row.get("expected_score_gain"),
                residual=row.get("euler_lagrange_residual"),
            )
        )
    lines.extend(
        [
            "",
            "## Authority Boundary",
            "- score_claim: `false`",
            "- score_claim_valid: `false`",
            "- promotion_eligible: `false`",
            "- rank_or_kill_eligible: `false`",
            "- ready_for_exact_eval_dispatch: `false`",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--scorer-response", action="append", default=[])
    parser.add_argument("--inverse-scorer-surface", action="append", default=[])
    parser.add_argument("--byte-shaving-signal-surface", action="append", default=[])
    parser.add_argument("--byte-shaving-campaign-plan", action="append", default=[])
    parser.add_argument("--mlx-effective-spend-triage-selection", action="append", default=[])
    parser.add_argument("--atom", action="append", default=[])
    parser.add_argument("--observation", action="append", default=[])
    parser.add_argument("--exact-auth-calibration-packet", action="append", default=[])
    parser.add_argument("--exact-auth-calibration-candidate-id", default=None)
    parser.add_argument("--queue-performance-summary", action="append", default=[])
    parser.add_argument("--queue-performance-runtime-identity", type=Path, default=None)
    parser.add_argument("--queue-performance-cache-identity", type=Path, default=None)
    parser.add_argument("--queue-performance-candidate-map", type=Path, default=None)
    parser.add_argument(
        "--queue-performance-axis",
        default="[local-queue-performance advisory]",
    )
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--resource-kind", default="local_mlx")
    parser.add_argument("--elapsed-seconds", type=float, default=None)
    parser.add_argument("--artifact-bytes", type=int, default=None)
    parser.add_argument("--total-byte-budget", type=int, default=None)
    parser.add_argument("--lambda-rate", type=float, default=CONTEST_RATE_SCORE_PER_BYTE)
    parser.add_argument("--inverse-scorer-max-units", type=int, default=32)
    parser.add_argument("--inverse-scorer-null-delta-epsilon", type=float, default=1e-6)
    parser.add_argument("--inverse-scorer-fragile-delta-threshold", type=float, default=0.0)
    parser.add_argument(
        "--inverse-scorer-allow-native-mlx-window-objective",
        action="store_true",
    )
    args = parser.parse_args(argv)

    atoms: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    try:
        for raw_path in args.scorer_response:
            path = Path(raw_path)
            payload = _load_json(path)
            surface = build_inverse_scorer_decision_surface(
                payload,
                source_label=_repo_rel(path, args.repo_root),
                max_units=args.inverse_scorer_max_units,
                null_scorer_delta_epsilon=args.inverse_scorer_null_delta_epsilon,
                fragile_scorer_delta_threshold=args.inverse_scorer_fragile_delta_threshold,
                allow_native_mlx_window_objective=(
                    args.inverse_scorer_allow_native_mlx_window_objective
                ),
            )
            atoms.extend(
                action_atoms_from_inverse_scorer_surface(
                    surface,
                    candidate_id=args.candidate_id,
                    elapsed_seconds=args.elapsed_seconds,
                    artifact_bytes=args.artifact_bytes,
                    resource_kind=args.resource_kind,
                )
            )
        for raw_path in args.inverse_scorer_surface:
            path = Path(raw_path)
            atoms.extend(
                action_atoms_from_inverse_scorer_surface(
                    _load_json(path),
                    candidate_id=args.candidate_id,
                    elapsed_seconds=args.elapsed_seconds,
                    artifact_bytes=args.artifact_bytes,
                    resource_kind=args.resource_kind,
                )
            )
        for raw_path in args.byte_shaving_signal_surface:
            path = Path(raw_path)
            atoms.extend(
                action_atoms_from_byte_shaving_signal_surface(
                    _load_mapping(path, label="byte-shaving signal surface"),
                    source_path=_repo_rel(path, args.repo_root),
                    candidate_id=args.candidate_id,
                    elapsed_seconds=args.elapsed_seconds,
                    artifact_bytes=args.artifact_bytes,
                    resource_kind=args.resource_kind,
                )
            )
        for raw_path in args.byte_shaving_campaign_plan:
            path = Path(raw_path)
            atoms.extend(
                action_atoms_from_byte_shaving_campaign_plan(
                    _load_mapping(path, label="byte-shaving campaign plan"),
                    source_path=_repo_rel(path, args.repo_root),
                    candidate_id=args.candidate_id,
                    elapsed_seconds=args.elapsed_seconds,
                    artifact_bytes=args.artifact_bytes,
                    resource_kind=args.resource_kind,
                )
            )
        for raw_path in args.mlx_effective_spend_triage_selection:
            path = Path(raw_path)
            atoms.extend(
                inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection(
                    _load_json(path),
                    source_path=_repo_rel(path, args.repo_root),
                    elapsed_seconds=args.elapsed_seconds,
                    artifact_bytes=args.artifact_bytes,
                    resource_kind=args.resource_kind,
                )
            )
        for raw_path in args.atom:
            path = Path(raw_path)
            atoms.extend(_as_rows(_load_json(path), path=path, key="atoms"))
        for raw_path in args.observation:
            path = Path(raw_path)
            observations.extend(_as_rows(_load_json(path), path=path, key="observations"))
        if args.exact_auth_calibration_packet:
            calibration_candidate_id = (
                args.exact_auth_calibration_candidate_id or args.candidate_id
            )
            if calibration_candidate_id is None:
                raise SystemExit(
                    "--exact-auth-calibration-candidate-id or --candidate-id "
                    "is required with --exact-auth-calibration-packet"
                )
            packet_paths = [Path(raw_path) for raw_path in args.exact_auth_calibration_packet]
            observations.extend(
                paired_exact_auth_calibration_observations_from_review_packets(
                    [_load_mapping(path, label="exact auth calibration packet") for path in packet_paths],
                    candidate_id=calibration_candidate_id,
                    packet_paths=[_repo_rel(path, args.repo_root) for path in packet_paths],
                    source_path=",".join(_repo_rel(path, args.repo_root) for path in packet_paths),
                )
            )
        if args.queue_performance_summary:
            if args.queue_performance_runtime_identity is None:
                raise SystemExit(
                    "--queue-performance-runtime-identity is required with "
                    "--queue-performance-summary"
                )
            if args.queue_performance_cache_identity is None:
                raise SystemExit(
                    "--queue-performance-cache-identity is required with "
                    "--queue-performance-summary"
                )
            runtime_identity = _load_mapping(
                args.queue_performance_runtime_identity,
                label="queue performance runtime identity",
            )
            cache_identity = _load_mapping(
                args.queue_performance_cache_identity,
                label="queue performance cache identity",
            )
            candidate_map = None
            if args.queue_performance_candidate_map is not None:
                candidate_map = _load_mapping(
                    args.queue_performance_candidate_map,
                    label="queue performance candidate map",
                )
            for raw_path in args.queue_performance_summary:
                path = Path(raw_path)
                observations.extend(
                    observations_from_queue_performance_summary(
                        _load_mapping(path, label="queue performance summary"),
                        runtime_identity=runtime_identity,
                        cache_identity=cache_identity,
                        axis=args.queue_performance_axis,
                        source_path=_repo_rel(path, args.repo_root),
                        candidate_id_by_experiment=candidate_map,
                    )
                )
        if not atoms:
            raise SystemExit(
                "provide at least one --scorer-response, --inverse-scorer-surface, "
                "--byte-shaving-signal-surface, --byte-shaving-campaign-plan, "
                "--mlx-effective-spend-triage-selection, or --atom"
            )
        action = build_discrete_scorer_action_functional(
            atoms,
            observations=observations,
            total_byte_budget=args.total_byte_budget,
            lambda_rate=args.lambda_rate,
        )
    except (InverseSteganalysisAcquisitionError, ScorerResponseDatasetError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(action, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_render_markdown(action), encoding="utf-8")
    print(
        f"wrote {args.output} "
        f"(cells={action['integral_totals']['cell_count']}, "
        f"selected={action['water_bucket']['selected_count']})"
    )
    print(
        "score_claim=false promotion_eligible=false "
        "rank_or_kill_eligible=false ready_for_exact_eval_dispatch=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
