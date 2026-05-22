# SPDX-License-Identifier: MIT
"""Execution-plan bridge for local MLX scorer-response profile selections.

The output of this module is intentionally not a score artifact. It turns a
profile-stability row selection into runner arguments while preserving the
false-authority contract that MLX output is local candidate-generation signal
until paired CPU/CUDA auth eval validates it.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_profile_stability import SCHEMA_VERSION as STABILITY_SCHEMA
from tac.local_acceleration.mlx_scorer_response import BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER

SCHEMA_VERSION = "mlx_scorer_response_execution_plan.v1"
PRODUCER = "tac.local_acceleration.mlx_execution_plan"

_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "ready_for_exact_eval_dispatch",
    "rank_or_kill_eligible",
)


class MLXExecutionPlanError(ValueError):
    """Raised when a profile selection cannot be converted safely."""


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MLXExecutionPlanError(f"{path}: expected JSON object")
    return payload


def write_execution_plan(plan: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mlx_scorer_response_execution_plan(
    stability_manifest: dict[str, Any],
    *,
    archive_size_bytes: int | None = None,
    repo_root: str | Path = ".",
    response_output: str | Path | None = None,
    components_dir: str | Path | None = None,
    progress_every: int = 0,
    allow_gpu_research_signal: bool = False,
    allow_batch_shape_research_signal: bool = False,
) -> dict[str, Any]:
    """Build safe runner args from a profile-stability manifest selection."""

    if not isinstance(stability_manifest, dict):
        raise MLXExecutionPlanError("stability manifest must be a JSON object")
    if stability_manifest.get("schema_version") != STABILITY_SCHEMA:
        raise MLXExecutionPlanError("stability manifest schema_version mismatch")
    _require_false_authority(stability_manifest, "stability manifest")
    if stability_manifest.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise MLXExecutionPlanError(
            f"stability manifest evidence_grade must be {EVIDENCE_GRADE_MLX}"
        )

    selection = stability_manifest.get("selection")
    if not isinstance(selection, dict):
        raise MLXExecutionPlanError("stability manifest selection must be an object")
    row = selection.get("recommended_row")
    if not isinstance(row, dict):
        raise MLXExecutionPlanError("stability manifest has no recommended_row")
    eligible = selection.get("eligible_row_indices")
    if not isinstance(eligible, list) or int(row.get("index", -1)) not in {
        int(item) for item in eligible
    }:
        raise MLXExecutionPlanError("recommended_row is not listed as eligible")

    profile_summary = stability_manifest.get("profile_summary")
    if not isinstance(profile_summary, dict):
        raise MLXExecutionPlanError("stability manifest profile_summary must be an object")
    reference_cache_dir = _required_str(profile_summary, "reference_cache_dir")
    candidate_cache_dir = _required_str(profile_summary, "candidate_cache_dir")
    resolved_archive_size = _resolve_archive_size(
        archive_size_bytes,
        profile_summary.get("archive_size_bytes"),
    )

    pair_window = _pair_window(row.get("pair_window"))
    start_pair = _optional_int(row.get("start_pair"))
    if start_pair is None:
        start_pair = pair_window[0]
    n_samples = _positive_int(row.get("n_samples"), "recommended_row.n_samples")
    max_pairs = pair_window[1] - pair_window[0]
    if max_pairs <= 0:
        max_pairs = n_samples
    batch_pairs = _positive_int(row.get("batch_pairs"), "recommended_row.batch_pairs")
    device = str(row.get("device"))
    if device not in {"cpu", "gpu"}:
        raise MLXExecutionPlanError("recommended_row.device must be cpu or gpu")
    if device == "gpu" and not allow_gpu_research_signal:
        raise MLXExecutionPlanError(
            "recommended_row.device=gpu requires allow_gpu_research_signal; "
            "GPU MLX scorer responses are local research signal only"
        )
    if batch_pairs != 1 and not allow_batch_shape_research_signal:
        raise MLXExecutionPlanError(
            f"{BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER}: recommended_row.batch_pairs="
            f"{batch_pairs}; pass allow_batch_shape_research_signal=True only for "
            "explicit batch-shape research probes"
        )
    progress = _nonnegative_int(progress_every, "progress_every")

    command_args = [
        "tools/run_mlx_scorer_response_cache.py",
        "--reference-cache-dir",
        reference_cache_dir,
        "--candidate-cache-dir",
        candidate_cache_dir,
        "--archive-size-bytes",
        str(resolved_archive_size),
        "--repo-root",
        str(repo_root),
        "--batch-pairs",
        str(batch_pairs),
        "--start-pair",
        str(start_pair),
        "--max-pairs",
        str(max_pairs),
        "--device",
        device,
    ]
    if response_output is not None:
        command_args.extend(["--output", str(response_output)])
    else:
        command_args.extend(["--output", "<required-response-output.json>"])
    if progress > 0:
        command_args.extend(["--progress-every", str(progress)])
    if components_dir is not None:
        command_args.extend(["--components-dir", str(components_dir)])
    if device == "gpu":
        command_args.append("--allow-gpu-research-signal")
    if batch_pairs != 1:
        command_args.append("--allow-batch-shape-research-signal")

    source_blockers = list(stability_manifest.get("blockers") or [])
    source_warnings = list(stability_manifest.get("warnings") or [])
    plan_warnings = list(source_warnings)
    if stability_manifest.get("passed") is not True:
        plan_warnings.append(
            "source_profile_failed_but_recommended_row_is_eligible; "
            "using row-level selection only; not exact-eval spend authority"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "source_schema_version": stability_manifest.get("schema_version"),
        "source_run_id": stability_manifest.get("run_id"),
        "source_profile_verdict": stability_manifest.get("verdict"),
        "source_profile_passed": stability_manifest.get("passed"),
        "source_profile_blockers": source_blockers,
        "selection_policy": selection.get("policy"),
        "selection_recommended_reason": selection.get("recommended_reason"),
        "profile_full_pass_required": False,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "allowed_uses": [
            "local_mlx_training_gradient_shaping",
            "local_scorer_response_dataset_harvest",
            "local_sweep_reranking_after_transfer_calibration",
        ],
        "forbidden_uses": [
            "leaderboard_or_pr_score_claim",
            "promotion_or_rank_or_kill_decision_without_exact_eval",
            "cuda_cpu_axis_conversion",
        ],
        "recommended_execution": {
            "tool": "tools/run_mlx_scorer_response_cache.py",
            "device": device,
            "batch_pairs": batch_pairs,
            "start_pair": start_pair,
            "max_pairs": max_pairs,
            "pair_window": pair_window,
            "n_samples": n_samples,
            "reference_cache_dir": reference_cache_dir,
            "candidate_cache_dir": candidate_cache_dir,
            "archive_size_bytes": resolved_archive_size,
            "repo_root": str(repo_root),
            "response_output": (
                None if response_output is None else str(response_output)
            ),
            "components_dir": None if components_dir is None else str(components_dir),
            "progress_every": progress,
            "allow_gpu_research_signal_required": device == "gpu",
            "allow_batch_shape_research_signal_required": batch_pairs != 1,
            "command_args": command_args,
            "python_command_args": [".venv/bin/python", *command_args],
        },
        "recommended_row": row,
        "warnings": plan_warnings,
        "authority_status": (
            "Execution plan is local MLX candidate-generation guidance only; paired "
            "contest CPU/CUDA auth eval remains required for score claims or promotion."
        ),
        "batch_shape_research_signal_blocker": BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER,
    }


def render_execution_plan_markdown(plan: dict[str, Any]) -> str:
    execution = plan["recommended_execution"]
    lines = [
        "# MLX Scorer-Response Execution Plan",
        "",
        f"- Score claim: {plan['score_claim']}",
        f"- Evidence tag: {plan['evidence_tag']}",
        f"- Source verdict: {plan.get('source_profile_verdict')}",
        f"- Recommended device: `{execution['device']}`",
        f"- Recommended batch pairs: `{execution['batch_pairs']}`",
        f"- Pair window: `{execution['pair_window']}`",
        f"- Pairs/sec: `{plan.get('recommended_row', {}).get('pairs_per_second')}`",
        "",
        "## Command",
        "",
        "```bash",
        shlex.join(str(part) for part in execution["python_command_args"]),
        "```",
        "",
        "## Authority",
        "",
        "- This plan is candidate-generation guidance only.",
        "- Exact contest CPU/CUDA auth eval is required before score claims or promotion.",
        "",
    ]
    warnings = plan.get("warnings") or []
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- `{warning}`")
        lines.append("")
    return "\n".join(lines)


def _require_false_authority(payload: dict[str, Any], label: str) -> None:
    for key in _FALSE_AUTHORITY_FIELDS:
        if payload.get(key) is not False:
            raise MLXExecutionPlanError(f"{label} {key} must be explicit false")
    if "promotable" in payload and payload.get("promotable") is not False:
        raise MLXExecutionPlanError(f"{label} promotable must be false")


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or str(value) == "":
        raise MLXExecutionPlanError(f"profile_summary.{key} is required")
    return str(value)


def _resolve_archive_size(cli_value: int | None, manifest_value: Any) -> int:
    if cli_value is not None:
        return _positive_int(cli_value, "archive_size_bytes")
    if manifest_value is None:
        raise MLXExecutionPlanError(
            "archive_size_bytes missing; pass --archive-size-bytes or regenerate "
            "the stability manifest from a profile containing archive_size_bytes"
        )
    return _positive_int(manifest_value, "profile_summary.archive_size_bytes")


def _pair_window(value: Any) -> list[int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise MLXExecutionPlanError("recommended_row.pair_window must have two entries")
    start = _nonnegative_int(value[0], "recommended_row.pair_window[0]")
    end = _positive_int(value[1], "recommended_row.pair_window[1]")
    if end <= start:
        raise MLXExecutionPlanError("recommended_row.pair_window end must exceed start")
    return [start, end]


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise MLXExecutionPlanError(f"{label} must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise MLXExecutionPlanError(f"{label} must be an integer") from exc
    if out <= 0:
        raise MLXExecutionPlanError(f"{label} must be positive")
    return out


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise MLXExecutionPlanError(f"{label} must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise MLXExecutionPlanError(f"{label} must be an integer") from exc
    if out < 0:
        raise MLXExecutionPlanError(f"{label} must be non-negative")
    return out


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, "optional_int")
