# SPDX-License-Identifier: MIT
"""Harvest MLX drift-scope search summaries into reusable recommendations."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

from .experiment_queue import ExperimentQueueError

MLX_DRIFT_SCOPE_RECOMMENDATION_SCHEMA = "mlx_drift_scope_recommendation.v1"
MLX_DRIFT_SCOPE_RECOMMENDATION_BATCH_SCHEMA = "mlx_drift_scope_recommendation_batch.v1"
SUPPORTED_SUMMARY_SCHEMAS = frozenset(
    {
        "pr95_hnerv_mlx_conv2d_drift_scope_search.v1",
    }
)
SELECTION_POLICIES = frozenset(
    {
        "minimal_no_cliff_then_best_delta",
        "minimal_no_cliff",
        "minimal_passed",
        "best_delta",
    }
)
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_packet_ready": False,
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentQueueError(f"{label} must be an object")
    return value


def _candidate_for_policy(
    summary: Mapping[str, Any],
    *,
    selection_policy: str,
) -> tuple[str, Mapping[str, Any]]:
    if selection_policy not in SELECTION_POLICIES:
        raise ExperimentQueueError(f"unsupported selection_policy {selection_policy!r}")
    ordered_keys = {
        "minimal_no_cliff_then_best_delta": (
            "minimal_no_cliff_candidate",
            "best_by_delta_candidate",
        ),
        "minimal_no_cliff": ("minimal_no_cliff_candidate",),
        "minimal_passed": ("minimal_passed_candidate",),
        "best_delta": ("best_by_delta_candidate",),
    }[selection_policy]
    for key in ordered_keys:
        candidate = summary.get(key)
        if isinstance(candidate, Mapping):
            return key, candidate
    raise ExperimentQueueError(
        f"summary has no candidate for selection_policy {selection_policy!r}"
    )


def _preset_from_candidate(candidate: Mapping[str, Any]) -> str | None:
    if candidate.get("kind") != "preset":
        return None
    candidate_id = str(candidate.get("candidate_id") or "")
    return candidate_id.removeprefix("preset_") or None


def build_mlx_drift_scope_recommendation(
    summary: Mapping[str, Any],
    *,
    summary_path: str | Path,
    repo_root: str | Path,
    selection_policy: str = "minimal_no_cliff_then_best_delta",
) -> dict[str, Any]:
    """Return a fail-closed recommendation from one drift-scope summary."""

    schema = str(summary.get("schema") or "")
    if schema not in SUPPORTED_SUMMARY_SCHEMAS:
        raise ExperimentQueueError(f"unsupported drift summary schema {schema!r}")
    try:
        require_no_truthy_authority_fields(summary, context="drift_scope_summary")
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    exact = _require_mapping(
        summary.get("exact_readiness_refusal"),
        "summary.exact_readiness_refusal",
    )
    if exact.get("ready") is not False:
        raise ExperimentQueueError("drift summary must refuse exact readiness")

    source_path = Path(summary_path)
    repo = Path(repo_root)
    selected_key, candidate = _candidate_for_policy(
        summary,
        selection_policy=selection_policy,
    )
    overrides = _require_mapping(
        candidate.get("conv2d_accumulation_overrides"),
        "candidate.conv2d_accumulation_overrides",
    )
    target = summary.get("optimization_target")
    if not isinstance(target, Mapping):
        target = {}
    no_cliff = candidate.get("drift_cliff_name") is None
    preset = _preset_from_candidate(candidate)
    recommendation = {
        "schema": MLX_DRIFT_SCOPE_RECOMMENDATION_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_summary_path": _repo_rel(source_path, repo),
        "source_summary_sha256": _sha256_file(source_path)
        if source_path.is_file()
        else None,
        "source_summary_schema": schema,
        "lane_id": summary.get("lane_id"),
        "archive_family": summary.get("archive_family"),
        "candidate_family": summary.get("candidate_family"),
        "optimization_target": dict(target),
        "selection_policy": selection_policy,
        "selected_summary_key": selected_key,
        "selected_candidate_id": candidate.get("candidate_id"),
        "selected_candidate_kind": candidate.get("kind"),
        "recommended_conv2d_override_preset": preset,
        "recommended_conv2d_accumulation_overrides": dict(overrides),
        "recommended_override_count": int(candidate.get("override_count") or 0),
        "no_cliff": no_cliff,
        "max_abs": float(candidate.get("max_abs") or 0.0),
        "mean_abs": float(candidate.get("mean_abs") or 0.0),
        "p99_abs": float(candidate.get("p99_abs") or 0.0),
        "p999_abs": float(candidate.get("p999_abs") or 0.0),
        "candidate_count": int(summary.get("candidate_count") or 0),
        "evidence_grade": summary.get("evidence_grade"),
        "adoption": {
            "schema": "mlx_drift_scope_recommendation_adoption.v1",
            "tool": "tools/run_pr95_mlx_timing_smoke.py",
            "flags": (
                ["--mlx-gpu-drift-conv2d-override-preset", preset]
                if preset
                else [
                    item
                    for name, mode in sorted(overrides.items())
                    for item in ("--mlx-gpu-drift-conv2d-override", f"{name}={mode}")
                ]
            ),
            "applies_to": [
                "pr95_mlx_gpu_forward_drift_attestation",
                "pr95_mlx_decoder_trace",
                "pr95_mlx_pytorch_export_parity_when_device_gpu",
            ],
        },
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "drift_scope_recommendation_is_local_mlx_research_signal",
                "requires_full_frame_inflate_parity_before_score_claim",
                "requires_contest_cpu_or_cuda_auth_eval_before_promotion",
            ],
        },
        **FALSE_AUTHORITY,
    }
    return recommendation


def build_mlx_drift_scope_recommendation_batch(
    summaries: Sequence[Mapping[str, Any]],
    *,
    summary_paths: Sequence[str | Path],
    repo_root: str | Path,
    selection_policy: str = "minimal_no_cliff_then_best_delta",
) -> dict[str, Any]:
    if len(summaries) != len(summary_paths):
        raise ExperimentQueueError("summaries and summary_paths length mismatch")
    if not summaries:
        raise ExperimentQueueError("at least one drift summary is required")
    recommendations = [
        build_mlx_drift_scope_recommendation(
            summary,
            summary_path=path,
            repo_root=repo_root,
            selection_policy=selection_policy,
        )
        for summary, path in zip(summaries, summary_paths, strict=True)
    ]
    primary = sorted(
        recommendations,
        key=lambda row: (
            not bool(row["no_cliff"]),
            int(row["recommended_override_count"]),
            float(row["max_abs"]),
            float(row["mean_abs"]),
        ),
    )[0]
    return {
        "schema": MLX_DRIFT_SCOPE_RECOMMENDATION_BATCH_SCHEMA,
        "generated_at_utc": _utc_now(),
        "selection_policy": selection_policy,
        "recommendation_count": len(recommendations),
        "primary_recommendation": primary,
        "recommendations": recommendations,
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "drift_scope_recommendation_batch_is_not_score_authority",
                "requires_exact_auth_eval_before_promotion",
            ],
        },
        **FALSE_AUTHORITY,
    }


__all__ = [
    "MLX_DRIFT_SCOPE_RECOMMENDATION_BATCH_SCHEMA",
    "MLX_DRIFT_SCOPE_RECOMMENDATION_SCHEMA",
    "build_mlx_drift_scope_recommendation",
    "build_mlx_drift_scope_recommendation_batch",
]
