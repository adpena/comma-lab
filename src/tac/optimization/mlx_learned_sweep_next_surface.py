# SPDX-License-Identifier: MIT
"""Next-surface routing for MLX learned-sweep plans.

The learned-sweep planner can rank rows for several execution substrates. This
helper turns the remaining ready rows into a typed routing report instead of
letting exhausted local queues degrade into opaque CLI failures.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from tac.optimization.mlx_dynamic_learned_sweep import FALSE_AUTHORITY
from tac.optimization.mlx_dynamic_learned_sweep_local_actuator import (
    SUPPORTED_SWEEP_CONFIG_IDS,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

SCHEMA = "mlx_dynamic_learned_sweep_next_surface_report.v1"
TOOL = "tac.optimization.mlx_learned_sweep_next_surface"
PLAN_SCHEMA = "mlx_dynamic_learned_sweep_plan.v1"
ROW_SCHEMA = "mlx_dynamic_learned_sweep_row.v1"

LOCAL_MLX_CONFIG_ID = "mlx_local_response"
MACOS_CPU_ADVISORY_CONFIG_ID = "macos_cpu_advisory"
CONTEST_EXACT_CONFIG_IDS = frozenset(
    {"contest_cpu_exact_candidate", "contest_cuda_diagnostic"}
)


class MLXLearnedSweepNextSurfaceError(ValueError):
    """Raised when a learned-sweep next-surface report cannot be built."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key, expected in FALSE_AUTHORITY.items():
        if payload.get(key) is not expected:
            raise MLXLearnedSweepNextSurfaceError(
                f"{label}: {key} must be explicit {expected!r}"
            )
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise MLXLearnedSweepNextSurfaceError(str(exc)) from exc


def _plan_rows(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = plan.get("ranked_sweep_rows")
    if not isinstance(rows, list):
        raise MLXLearnedSweepNextSurfaceError("plan ranked_sweep_rows must be a list")
    out: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise MLXLearnedSweepNextSurfaceError(
                f"plan ranked_sweep_rows[{index}] must be an object"
            )
        if row.get("schema") == ROW_SCHEMA:
            _require_false_authority(row, label=f"ranked_sweep_rows[{index}]")
            out.append(row)
    return out


def _ordered_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _summarize_by_config(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        config_id = str(row.get("sweep_config_id") or "unknown")
        bucket = buckets.setdefault(
            config_id,
            {
                "sweep_config_id": config_id,
                "total_row_count": 0,
                "ready_row_count": 0,
                "exact_eval_candidate_row_count": 0,
                "execution_layers": [],
                "substrates": [],
                "optimization_pass_ids": [],
            },
        )
        bucket["total_row_count"] += 1
        if row.get("ready_for_local_sweep") is True:
            bucket["ready_row_count"] += 1
        if row.get("exact_eval_candidate") is True:
            bucket["exact_eval_candidate_row_count"] += 1
        for key, field in (
            ("execution_layer", "execution_layers"),
            ("substrate", "substrates"),
            ("optimization_pass_id", "optimization_pass_ids"),
        ):
            value = row.get(key)
            if isinstance(value, str) and value and value not in bucket[field]:
                bucket[field].append(value)
    return [buckets[key] for key in sorted(buckets)]


def _ready_rows_for_config(
    rows: Sequence[Mapping[str, Any]],
    *,
    config_id: str,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("sweep_config_id") == config_id
        and row.get("ready_for_local_sweep") is True
    ]


def _top_ready_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("rank") or 10**12),
            str(row.get("queue_candidate_id") or ""),
        ),
    )
    for row in sorted_rows[:max_rows]:
        out.append(
            {
                "queue_candidate_id": row.get("queue_candidate_id"),
                "candidate_id": row.get("candidate_id"),
                "sweep_config_id": row.get("sweep_config_id"),
                "optimization_pass_id": row.get("optimization_pass_id"),
                "execution_layer": row.get("execution_layer"),
                "substrate": row.get("substrate"),
                "rank": row.get("rank"),
                "acquisition_value": row.get("acquisition_value"),
                "expected_improvement": row.get("expected_improvement"),
                "selected_pair_indices": row.get("selected_pair_indices"),
                "allowed_use": row.get("allowed_use"),
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "promotable": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
            }
        )
    return out


def _next_surface(
    *,
    local_mlx_ready_count: int,
    macos_cpu_ready_count: int,
    exact_candidate_total_count: int,
    exact_ready_count: int,
) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    routing_notes: list[str] = []
    if local_mlx_ready_count > 0:
        return (
            {
                "id": "build_local_mlx_autopilot_queue",
                "status": "ready",
                "target_sweep_config_id": LOCAL_MLX_CONFIG_ID,
                "queue_owner": "comma_lab.scheduler.mlx_learned_sweep_autopilot_queue",
                "actuator": "tools/run_mlx_dynamic_learned_sweep_autopilot.py",
                "rationale": "ready cache-backed local MLX response rows remain",
            },
            blockers,
            routing_notes,
        )

    routing_notes.append("no_ready_mlx_local_response_rows")
    if macos_cpu_ready_count > 0:
        blockers.append("macos_cpu_advisory_is_not_exact_score_authority")
        if exact_candidate_total_count > 0 and exact_ready_count == 0:
            blockers.append("contest_exact_rows_not_ready_without_auth_axis_payload")
        if MACOS_CPU_ADVISORY_CONFIG_ID in SUPPORTED_SWEEP_CONFIG_IDS:
            routing_notes.append("macos_cpu_advisory_actuator_supported")
            return (
                {
                    "id": "build_macos_cpu_advisory_autopilot_queue",
                    "status": "ready_pending_selection_artifact_validation",
                    "target_sweep_config_id": MACOS_CPU_ADVISORY_CONFIG_ID,
                    "queue_owner": "comma_lab.scheduler.mlx_learned_sweep_autopilot_queue",
                    "actuator": (
                        "tools/run_mlx_dynamic_learned_sweep_autopilot.py "
                        "--sweep-config-id macos_cpu_advisory --device cpu"
                    ),
                    "required_helper": (
                        "tac.optimization.mlx_learned_sweep_advisory_handoff"
                    ),
                    "required_inputs": [
                        "selection rows with local_cpu_advisory_source_path",
                        "selection rows with window_baseline_local_cpu_advisory_source_path",
                        "candidate payloads preserved as planning-only proxy inputs",
                    ],
                    "required_contract": (
                        "append advisory observations only; never emit score, "
                        "promotion, rank/kill, or exact-dispatch authority"
                    ),
                    "rationale": (
                        "macOS-CPU advisory rows are ranked and ready, and the "
                        "learned-sweep local actuator supports advisory artifact "
                        "harvest on local CPU"
                    ),
                },
                _ordered_unique(blockers),
                _ordered_unique(routing_notes),
            )
        blockers.extend(
            [
                "macos_cpu_advisory_executor_missing_for_learned_sweep",
            ]
        )
        return (
            {
                "id": "wire_macos_cpu_advisory_actuator",
                "status": "blocked_until_executor_lands",
                "target_sweep_config_id": MACOS_CPU_ADVISORY_CONFIG_ID,
                "queue_owner": "comma_lab.scheduler",
                "required_helper": (
                    "tac.optimization.mlx_learned_sweep_advisory_handoff"
                ),
                "required_contract": (
                    "append advisory observations only; never emit score, "
                    "promotion, rank/kill, or exact-dispatch authority"
                ),
                "rationale": (
                    "macOS-CPU advisory rows are ranked and ready, but no learned-"
                    "sweep actuator consumes them yet"
                ),
            },
            _ordered_unique(blockers),
            _ordered_unique(routing_notes),
        )

    if exact_candidate_total_count > 0:
        blockers.append("contest_exact_rows_not_ready_without_auth_axis_payload")
        return (
            {
                "id": "exact_calibration_or_candidate_regeneration_required",
                "status": "blocked_until_exact_axis_payload_or_new_candidates",
                "target_sweep_config_id": None,
                "queue_owner": "tac.optimization.mlx_dynamic_learned_sweep",
                "rationale": (
                    "only exact-gated candidates remain, and they are not ready for "
                    "dispatch under the false-authority contract"
                ),
            },
            _ordered_unique(blockers),
            _ordered_unique(routing_notes),
        )

    blockers.append("no_ranked_sweep_rows_ready_for_any_known_surface")
    return (
        {
            "id": "regenerate_candidate_payload",
            "status": "blocked_until_new_candidate_payload",
            "target_sweep_config_id": None,
            "queue_owner": "tac.optimization.mlx_effective_spend_triage",
            "rationale": "the plan contains no ready local, advisory, or exact rows",
        },
        _ordered_unique(blockers),
        _ordered_unique(routing_notes),
    )


def build_mlx_learned_sweep_next_surface_report(
    plan: Mapping[str, Any],
    *,
    source_plan: Mapping[str, Any] | None = None,
    max_top_rows: int = 8,
) -> dict[str, Any]:
    """Return a fail-closed next-surface report for a learned-sweep plan."""

    if plan.get("schema") != PLAN_SCHEMA:
        raise MLXLearnedSweepNextSurfaceError(f"plan schema must be {PLAN_SCHEMA}")
    _require_false_authority(plan, label="plan")
    if max_top_rows < 0:
        raise MLXLearnedSweepNextSurfaceError("max_top_rows must be non-negative")

    rows = _plan_rows(plan)
    local_mlx_ready_rows = _ready_rows_for_config(rows, config_id=LOCAL_MLX_CONFIG_ID)
    macos_cpu_ready_rows = _ready_rows_for_config(
        rows,
        config_id=MACOS_CPU_ADVISORY_CONFIG_ID,
    )
    exact_rows = [
        row for row in rows if row.get("sweep_config_id") in CONTEST_EXACT_CONFIG_IDS
    ]
    exact_ready_rows = [
        row for row in exact_rows if row.get("ready_for_local_sweep") is True
    ]
    recommended, blockers, routing_notes = _next_surface(
        local_mlx_ready_count=len(local_mlx_ready_rows),
        macos_cpu_ready_count=len(macos_cpu_ready_rows),
        exact_candidate_total_count=len(exact_rows),
        exact_ready_count=len(exact_ready_rows),
    )
    top_source_rows = local_mlx_ready_rows or macos_cpu_ready_rows or exact_ready_rows

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": _utc_now(),
        **FALSE_AUTHORITY,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "source_plan": dict(source_plan or {"schema": plan.get("schema")}),
        "summary": {
            "ranked_row_count": len(rows),
            "ready_row_count": sum(
                1 for row in rows if row.get("ready_for_local_sweep") is True
            ),
            "ready_mlx_local_response_row_count": len(local_mlx_ready_rows),
            "ready_macos_cpu_advisory_row_count": len(macos_cpu_ready_rows),
            "contest_exact_candidate_row_count": len(exact_rows),
            "contest_exact_ready_row_count": len(exact_ready_rows),
            "plan_summary": dict(plan.get("summary") or {}),
        },
        "sweep_config_summaries": _summarize_by_config(rows),
        "recommended_next_surface": recommended,
        "blockers": blockers,
        "routing_notes": routing_notes,
        "top_ready_rows": _top_ready_rows(top_source_rows, max_rows=max_top_rows),
        "authority_boundary": {
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
            "allowed_use": "queue_routing_and_actuator_gap_analysis_only",
            "forbidden_uses": [
                "score_claim",
                "promotion",
                "rank_or_kill",
                "exact_eval_dispatch_authorization",
            ],
        },
    }
    _require_false_authority(report, label="next_surface_report")
    return report


def render_mlx_learned_sweep_next_surface_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing next-surface report."""

    if report.get("schema") != SCHEMA:
        raise MLXLearnedSweepNextSurfaceError(f"report schema must be {SCHEMA}")
    _require_false_authority(report, label="next_surface_report")
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    next_surface = (
        report.get("recommended_next_surface")
        if isinstance(report.get("recommended_next_surface"), Mapping)
        else {}
    )
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    routing_notes = (
        report.get("routing_notes")
        if isinstance(report.get("routing_notes"), list)
        else []
    )
    rows = report.get("top_ready_rows") if isinstance(report.get("top_ready_rows"), list) else []

    lines = [
        "# MLX Learned Sweep Next Surface",
        "",
        f"- schema: `{SCHEMA}`",
        f"- generated_at_utc: `{report.get('generated_at_utc')}`",
        f"- recommended_next_surface: `{next_surface.get('id')}`",
        f"- status: `{next_surface.get('status')}`",
        f"- ranked rows: `{summary.get('ranked_row_count')}`",
        f"- ready `mlx_local_response` rows: `{summary.get('ready_mlx_local_response_row_count')}`",
        f"- ready `macos_cpu_advisory` rows: `{summary.get('ready_macos_cpu_advisory_row_count')}`",
        f"- contest exact candidate rows: `{summary.get('contest_exact_candidate_row_count')}`",
        "",
        "## Authority Boundary",
        "",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- rank_or_kill_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "- dispatch_attempted: `false`",
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Routing Notes", ""])
    if routing_notes:
        lines.extend(f"- `{note}`" for note in routing_notes)
    else:
        lines.append("- none")
    lines.extend(["", "## Top Ready Rows", ""])
    if rows:
        lines.append("| rank | sweep_config | optimization_pass | queue_candidate_id |")
        lines.append("| ---: | --- | --- | --- |")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| "
                f"{row.get('rank')} | "
                f"`{row.get('sweep_config_id')}` | "
                f"`{row.get('optimization_pass_id')}` | "
                f"`{row.get('queue_candidate_id')}` |"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


__all__ = [
    "LOCAL_MLX_CONFIG_ID",
    "MACOS_CPU_ADVISORY_CONFIG_ID",
    "PLAN_SCHEMA",
    "ROW_SCHEMA",
    "SCHEMA",
    "TOOL",
    "MLXLearnedSweepNextSurfaceError",
    "build_mlx_learned_sweep_next_surface_report",
    "render_mlx_learned_sweep_next_surface_markdown",
]
