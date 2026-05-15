#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize PR106 CUDA latent-correction probe plans into dry-run tasks.

This is the fail-closed bridge after
``tools/build_pr106_cuda_latent_correction_probe.py``. It consumes that tool's
false-authority plan and writes deterministic per-pair/per-latent/per-delta
task artifacts plus dry-run score-table commands. The emitted commands use
``experiments/build_pr106_latent_score_table.py --dry-run-plan`` only: no CUDA
scoring, no remote dispatch, no archive mutation, and no score authority.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import (  # noqa: E402
    json_line,
    json_text,
    read_json,
    repo_relative,
    sha256_bytes,
    sha256_file,
)

PLAN_SCHEMA = "pr106_cuda_latent_correction_probe_plan_v1"
SCHEMA = "pr106_cuda_latent_correction_materializer_v1"
TASK_SCHEMA = "pr106_cuda_latent_correction_probe_task_v1"
COMMAND_SCHEMA = "pr106_cuda_latent_correction_scoretable_command_v1"
TOOL = "tools/pr106_cuda_latent_correction_materializer.py"
SCORE_TABLE_TOOL = "experiments/build_pr106_latent_score_table.py"
FALSE_AUTHORITY_FLAGS = {
    "research_only": True,
    "score_claim": False,
    "score_claim_valid": False,
    "contest_axis_claim": False,
    "dispatch_attempted": False,
    "remote_jobs_dispatched": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_broad_waterfill_dispatch": False,
    "frontier_language_allowed": False,
}
FORBIDDEN_TRUE_KEYS = tuple(
    key for key, expected in FALSE_AUTHORITY_FLAGS.items() if expected is False
)
DEFAULT_DISPATCH_BLOCKERS = [
    "dry_run_materializer_only",
    "scoretable_commands_use_dry_run_plan",
    "real_cuda_scoring_requires_active_lane_claim_and_explicit_separate_dispatch",
    "byte_closed_archive_not_emitted",
    "paired_exact_contest_cuda_and_contest_cpu_eval_missing",
    "not_score_or_promotion_authority",
]


def _json_object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": repo_relative(path, REPO_ROOT),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _authority_view(payload: dict[str, Any]) -> dict[str, Any]:
    authority = payload.get("authority")
    if isinstance(authority, dict):
        return {**payload, **authority}
    return payload


def validate_false_authority(payload: dict[str, Any], *, label: str) -> None:
    view = _authority_view(payload)
    for key in FORBIDDEN_TRUE_KEYS:
        if view.get(key) is True:
            raise ValueError(f"{label} has {key}=true; refusing score-authority input")
    if view.get("research_only") is not True:
        raise ValueError(f"{label} must carry research_only=true")


def validate_plan(plan: dict[str, Any], *, label: str = "plan") -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"{label} schema must be {PLAN_SCHEMA!r}, got {plan.get('schema')!r}")
    validate_false_authority(plan, label=label)
    candidates = plan.get("candidate_pairs")
    if not isinstance(candidates, list):
        raise ValueError(f"{label} is missing candidate_pairs list")
    inputs = plan.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("source_archive"), dict):
        raise ValueError(f"{label} is missing inputs.source_archive")
    materialization = plan.get("materialization")
    if isinstance(materialization, dict) and materialization.get("supported") is True:
        raise ValueError(f"{label} already advertises materialization.supported=true")


def validate_source_archive(plan: dict[str, Any], *, allow_missing: bool) -> dict[str, Any]:
    source = plan["inputs"]["source_archive"]
    path_text = source.get("path")
    expected_sha = source.get("sha256")
    expected_bytes = source.get("bytes")
    if not isinstance(path_text, str) or not path_text:
        raise ValueError("plan inputs.source_archive.path must be a non-empty string")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("plan inputs.source_archive.sha256 must be a SHA-256 hex string")
    source_path = _resolve_path(path_text)
    if not source_path.is_file():
        if allow_missing:
            return {
                "path": path_text,
                "resolved_path": source_path.as_posix(),
                "exists": False,
                "sha256_match": False,
                "bytes_match": False,
                "verification_blocker": "source_archive_missing",
            }
        raise ValueError(f"source archive is missing: {source_path}")
    actual_sha = sha256_file(source_path)
    actual_bytes = int(source_path.stat().st_size)
    if actual_sha != expected_sha:
        raise ValueError(
            f"source archive SHA mismatch for {source_path}: "
            f"got {actual_sha}, expected {expected_sha}"
        )
    if isinstance(expected_bytes, int) and actual_bytes != expected_bytes:
        raise ValueError(
            f"source archive byte mismatch for {source_path}: "
            f"got {actual_bytes}, expected {expected_bytes}"
        )
    return {
        "path": path_text,
        "resolved_path": source_path.as_posix(),
        "exists": True,
        "bytes": actual_bytes,
        "sha256": actual_sha,
        "sha256_match": True,
        "bytes_match": expected_bytes == actual_bytes,
        "verification_blocker": None,
    }


def _int_value(value: Any, *, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def _delta_values(row: dict[str, Any], plan: dict[str, Any]) -> list[int]:
    values = row.get("delta_q_values")
    if values is None:
        policy = plan.get("selection_policy")
        values = policy.get("delta_q_values") if isinstance(policy, dict) else None
    if not isinstance(values, list) or not values:
        raise ValueError("candidate pair is missing non-empty delta_q_values")
    parsed: list[int] = []
    for idx, value in enumerate(values):
        delta = _int_value(value, name=f"delta_q_values[{idx}]", minimum=-127, maximum=127)
        if delta == 0:
            raise ValueError("delta_q_values must not include 0")
        parsed.append(delta)
    return parsed


def _candidate_grid_index(*, dim_idx: int, delta_q: int, delta_radius: int) -> int:
    if abs(delta_q) > delta_radius:
        raise ValueError(f"delta_q={delta_q} exceeds delta_radius={delta_radius}")
    deltas = [delta for delta in range(-delta_radius, delta_radius + 1) if delta != 0]
    return 1 + dim_idx * len(deltas) + deltas.index(delta_q)


def _task_id(*, pair_idx: int, dim_idx: int, delta_q: int) -> str:
    sign = "p" if delta_q > 0 else "m"
    return f"pr106_pair_{pair_idx:04d}_dim_{dim_idx:02d}_dq_{sign}{abs(delta_q)}"


def _command_args(
    *,
    python_executable: str,
    source_archive: str,
    pair_idx: int,
    pair_output_dir: Path,
    delta_radius: int,
    latent_dim: int,
    n_pairs: int,
    lane_id: str,
) -> list[str]:
    return [
        python_executable,
        SCORE_TABLE_TOOL,
        "--pr106-archive",
        source_archive,
        "--out-dir",
        pair_output_dir.as_posix(),
        "--delta-radius",
        str(delta_radius),
        "--latent-dim",
        str(latent_dim),
        "--n-pairs",
        str(n_pairs),
        "--max-pairs",
        str(pair_idx + 1),
        "--lane-id",
        lane_id,
        "--dry-run-plan",
    ]


def build_materialization(
    *,
    plan_path: Path,
    output_dir: Path,
    python_executable: str = ".venv/bin/python",
    n_pairs: int = 600,
    lane_id: str = "lane_pr106_latent_score_table",
    allow_missing_source_archive: bool = False,
) -> dict[str, Any]:
    plan = _json_object(plan_path)
    validate_plan(plan)
    source_verification = validate_source_archive(
        plan,
        allow_missing=allow_missing_source_archive,
    )
    if not source_verification["exists"]:
        raise ValueError("source archive must exist for fail-closed materialization")

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_dir = output_dir / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)

    source_archive = str(plan["inputs"]["source_archive"]["path"])
    plan_sha = sha256_file(plan_path)
    plan_bytes = int(plan_path.stat().st_size)
    commands: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    pair_artifacts: list[Path] = []
    command_by_pair: dict[int, dict[str, Any]] = {}

    for row_idx, row in enumerate(plan["candidate_pairs"]):
        if not isinstance(row, dict):
            raise ValueError(f"candidate_pairs[{row_idx}] must be an object")
        pair_idx = _int_value(row.get("pair_idx"), name="pair_idx", minimum=0)
        if pair_idx >= n_pairs:
            raise ValueError(f"pair_idx={pair_idx} is outside n_pairs={n_pairs}")
        latent_dim = _int_value(row.get("latent_dim_count"), name="latent_dim_count", minimum=1, maximum=254)
        deltas = _delta_values(row, plan)
        delta_radius = max(abs(delta) for delta in deltas)
        priority_rank = _int_value(
            row.get("priority_rank", row_idx + 1),
            name="priority_rank",
            minimum=1,
        )
        pair_output_dir = output_dir / "scoretable_dryrun" / f"pair_{pair_idx:04d}"
        args = _command_args(
            python_executable=python_executable,
            source_archive=source_archive,
            pair_idx=pair_idx,
            pair_output_dir=pair_output_dir,
            delta_radius=delta_radius,
            latent_dim=latent_dim,
            n_pairs=n_pairs,
            lane_id=lane_id,
        )
        command_id = f"pair_{pair_idx:04d}_scoretable_dryrun"
        command = {
            "schema": COMMAND_SCHEMA,
            "command_id": command_id,
            "pair_idx": pair_idx,
            "priority_rank": priority_rank,
            "dry_run_plan_only": True,
            "executes_gpu": False,
            "dispatches_remote": False,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "command_args": args,
            "command": shlex.join(args),
            "expected_outputs": {
                "score_table_manifest": (pair_output_dir / "score_table_manifest.json").as_posix(),
                "candidate_grid": (pair_output_dir / "candidate_grid.npy").as_posix(),
            },
            "blockers_to_real_cuda_scoring": [
                "requires_explicit_active_lane_claim",
                "requires_real_cuda_execution_outside_this_materializer",
                "requires_completed_score_table_manifest_before_archive_builder",
            ],
        }
        commands.append(command)
        command_by_pair[pair_idx] = command

        pair_task_ids: list[str] = []
        for dim_idx in range(latent_dim):
            for delta_q in deltas:
                task_id = _task_id(pair_idx=pair_idx, dim_idx=dim_idx, delta_q=delta_q)
                pair_task_ids.append(task_id)
                tasks.append(
                    {
                        "schema": TASK_SCHEMA,
                        "task_id": task_id,
                        "plan_from_state_hash": plan.get("from_state_hash"),
                        "source_plan_sha256": plan_sha,
                        "priority_rank": priority_rank,
                        "pair_idx": pair_idx,
                        "latent_dim_idx": dim_idx,
                        "delta_q": delta_q,
                        "candidate_grid_index": _candidate_grid_index(
                            dim_idx=dim_idx,
                            delta_q=delta_q,
                            delta_radius=delta_radius,
                        ),
                        "planned_mode": row.get("planned_mode"),
                        "dominant_component": row.get("dominant_component"),
                        "axis_dominant_component": row.get("axis_dominant_component"),
                        "scoretable_command_id": command_id,
                        "scoretable_command": command["command"],
                        "authority": {
                            **FALSE_AUTHORITY_FLAGS,
                            "dry_run_task_only": True,
                            "archive_mutation_performed": False,
                        },
                        "dispatch_blockers": list(DEFAULT_DISPATCH_BLOCKERS),
                    }
                )
        pair_payload = {
            "schema": "pr106_cuda_latent_correction_pair_probe_artifact_v1",
            "pair_idx": pair_idx,
            "priority_rank": priority_rank,
            "latent_dim_count": latent_dim,
            "delta_q_values": deltas,
            "task_ids": pair_task_ids,
            "task_count": len(pair_task_ids),
            "scoretable_command_id": command_id,
            "scoretable_command": command["command"],
            "authority": FALSE_AUTHORITY_FLAGS,
        }
        pair_path = pair_dir / f"pair_{pair_idx:04d}.json"
        pair_path.write_text(json_text(pair_payload), encoding="utf-8")
        pair_artifacts.append(pair_path)

    tasks_path = output_dir / "pr106_cuda_latent_correction_probe_tasks.jsonl"
    tasks_path.write_text("".join(json_line(row) for row in tasks), encoding="utf-8")
    commands_jsonl_path = output_dir / "pr106_cuda_latent_correction_scoretable_commands.jsonl"
    commands_jsonl_path.write_text("".join(json_line(row) for row in commands), encoding="utf-8")
    commands_sh_path = output_dir / "pr106_cuda_latent_correction_scoretable_commands.sh"
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Generated dry-run score-table commands only.",
        "# These commands do not score CUDA, dispatch remote jobs, or claim authority.",
    ]
    script_lines.extend(command["command"] for command in commands)
    commands_sh_path.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
    os.chmod(commands_sh_path, 0o755)

    manifest_path = output_dir / "pr106_cuda_latent_correction_materializer_manifest.json"
    rebuild_args = [
        python_executable,
        TOOL,
        "--plan",
        plan_path.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--n-pairs",
        str(n_pairs),
        "--lane-id",
        lane_id,
    ]
    if python_executable != ".venv/bin/python":
        rebuild_args.extend(["--python-executable", python_executable])
    state_hash = sha256_bytes(
        json_text(
            {
                "plan_sha256": plan_sha,
                "task_count": len(tasks),
                "command_count": len(commands),
                "n_pairs": n_pairs,
                "lane_id": lane_id,
                "source_archive_sha256": source_verification.get("sha256"),
            }
        ).encode("utf-8")
    )[:16]
    manifest = {
        "schema": SCHEMA,
        "producer": TOOL,
        "state_hash": state_hash,
        "source_plan": {
            "path": repo_relative(plan_path, REPO_ROOT),
            "bytes": plan_bytes,
            "sha256": plan_sha,
            "schema": plan.get("schema"),
            "from_state_hash": plan.get("from_state_hash"),
        },
        "source_archive": source_verification,
        "authority": {
            **FALSE_AUTHORITY_FLAGS,
            "dry_run_materializer_only": True,
            "archive_mutation_performed": False,
            "scoretable_commands_execute_gpu": False,
        },
        "counts": {
            "candidate_pair_count": len(plan["candidate_pairs"]),
            "scoretable_command_count": len(commands),
            "probe_task_count": len(tasks),
        },
        "scoretable_contract": {
            "tool": SCORE_TABLE_TOOL,
            "command_mode": "--dry-run-plan",
            "lane_id": lane_id,
            "n_pairs": n_pairs,
            "unique_pair_commands": [row["command_id"] for row in commands],
            "real_cuda_commands_emitted": False,
        },
        "outputs": {
            "tasks_jsonl": _artifact(tasks_path),
            "commands_jsonl": _artifact(commands_jsonl_path),
            "commands_sh": _artifact(commands_sh_path),
            "pair_artifacts": [_artifact(path) for path in pair_artifacts],
        },
        "dispatch_blockers": list(DEFAULT_DISPATCH_BLOCKERS),
        "required_next_proofs": [
            "run real CUDA score-table generation under an active lane claim outside this dry-run materializer",
            "reduce completed score table into charged correction bytes",
            "prove runtime consumes correction bytes and changes intended frames",
            "run paired exact [contest-CUDA] and [contest-CPU] auth eval before any frontier language",
        ],
        "rebuild_command": shlex.join(rebuild_args),
    }
    manifest_path.write_text(json_text(manifest), encoding="utf-8")
    manifest["outputs"]["manifest"] = _artifact(manifest_path)
    manifest_path.write_text(json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="PR106 CUDA latent-correction probe plan JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-executable", default=".venv/bin/python")
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--lane-id", default="lane_pr106_latent_score_table")
    parser.add_argument(
        "--allow-missing-source-archive",
        action="store_true",
        help="Reserved for forensic replay manifests; still refuses materialization output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.allow_missing_source_archive:
        print(
            "ERROR: --allow-missing-source-archive is reserved; this materializer "
            "requires source archive custody before writing tasks.",
            file=sys.stderr,
        )
        return 2
    try:
        manifest = build_materialization(
            plan_path=args.plan,
            output_dir=args.output_dir,
            python_executable=args.python_executable,
            n_pairs=args.n_pairs,
            lane_id=args.lane_id,
            allow_missing_source_archive=args.allow_missing_source_archive,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "[pr106-cuda-latent-correction-materializer] wrote "
        f"{manifest['outputs']['manifest']['path']}"
    )
    print(
        "[pr106-cuda-latent-correction-materializer] wrote "
        f"{manifest['outputs']['tasks_jsonl']['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
