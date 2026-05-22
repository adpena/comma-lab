# SPDX-License-Identifier: MIT
"""Fail-closed selection from strict effective MLX spend-triage evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.optimization.scorer_response_dataset import (
    render_authority_markdown_block,
)

SCHEMA = "mlx_effective_spend_triage_candidate_selection.v1"
ROW_SCHEMA = "mlx_effective_spend_triage_candidate_row.v1"
TOOL = "tac.optimization.mlx_effective_spend_triage_selection"
DEFAULT_PREDICTION_FIELD = "ll_predicted_delta_vs_baseline_score"

_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "promotion_eligible",
    "ready_for_exact_eval_dispatch",
    "rank_or_kill_eligible",
    "promotable",
)


class MLXEffectiveSpendTriageSelectionError(ValueError):
    """Raised when MLX spend-triage selection would lose authority context."""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a source artifact consumed by the selector."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _require_false_authority(payload: dict[str, Any], *, label: str) -> None:
    for key in _FALSE_AUTHORITY_FIELDS:
        if payload.get(key) is not False:
            raise MLXEffectiveSpendTriageSelectionError(
                f"{label} {key} must be explicit false"
            )


def _require_gate(
    plan: dict[str, Any],
    key: str,
    *,
    status: str = "strict_pass",
) -> dict[str, Any]:
    gate = plan.get(key)
    if not isinstance(gate, dict):
        raise MLXEffectiveSpendTriageSelectionError(f"{key} gate missing")
    if gate.get("status") != status:
        raise MLXEffectiveSpendTriageSelectionError(
            f"{key} gate must be {status}, got {gate.get('status')!r}"
        )
    _require_false_authority(gate, label=key)
    return gate


def _calibrated_min_gap(
    plan: dict[str, Any],
    explicit_min_observed_gain: float | None,
) -> float:
    if explicit_min_observed_gain is not None:
        if not math.isfinite(explicit_min_observed_gain) or explicit_min_observed_gain < 0:
            raise MLXEffectiveSpendTriageSelectionError(
                "min_observed_gain must be finite and non-negative"
            )
        return float(explicit_min_observed_gain)
    calibration_gate = _require_gate(plan, "mlx_score_calibration_gate")
    summary = calibration_gate.get("summary")
    if not isinstance(summary, dict):
        raise MLXEffectiveSpendTriageSelectionError(
            "mlx_score_calibration_gate summary missing"
        )
    min_gap = _as_float(summary.get("recommended_min_mlx_gap_for_spend_triage"))
    if min_gap is None or min_gap < 0:
        raise MLXEffectiveSpendTriageSelectionError(
            "recommended_min_mlx_gap_for_spend_triage missing"
        )
    return min_gap


def _validate_plan_for_selection(plan: dict[str, Any]) -> dict[str, Any]:
    _require_false_authority(plan, label="plan")
    effective_gate = _require_gate(plan, "effective_mlx_spend_triage_gate")
    if effective_gate.get("schema") != "ll_effective_mlx_spend_triage_gate.v1":
        raise MLXEffectiveSpendTriageSelectionError(
            "effective_mlx_spend_triage_gate schema mismatch"
        )
    if effective_gate.get("mlx_exact_eval_spend_triage_allowed") is not True:
        raise MLXEffectiveSpendTriageSelectionError(
            "effective MLX spend-triage gate did not allow spend triage"
        )
    if effective_gate.get("candidate_generation_only") is not True:
        raise MLXEffectiveSpendTriageSelectionError(
            "effective gate must stay candidate_generation_only"
        )
    response_gate = plan.get("response_validation_gate")
    if not isinstance(response_gate, dict) or response_gate.get("passed") is not True:
        raise MLXEffectiveSpendTriageSelectionError(
            "response_validation_gate must be a pass"
        )
    _require_false_authority(response_gate, label="response_validation_gate")
    parity_gate = _require_gate(plan, "mlx_torch_parity_sweep_gate")
    if parity_gate.get("mlx_rows_allowed_for_planner") is not True:
        raise MLXEffectiveSpendTriageSelectionError(
            "mlx_torch_parity_sweep_gate must allow planner rows"
        )
    calibration_gate = _require_gate(plan, "mlx_score_calibration_gate")
    if calibration_gate.get("mlx_spend_triage_allowed") is not True:
        raise MLXEffectiveSpendTriageSelectionError(
            "mlx_score_calibration_gate must allow spend triage"
        )
    production_gate = _require_gate(plan, "mlx_production_contract_gate")
    if production_gate.get("mlx_spend_triage_allowed") is not True:
        raise MLXEffectiveSpendTriageSelectionError(
            "mlx_production_contract_gate must allow spend triage"
        )
    return effective_gate


def _validate_dataset(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    if dataset.get("schema") != "scorer_response_dataset.v1":
        raise MLXEffectiveSpendTriageSelectionError("dataset schema mismatch")
    _require_false_authority(dataset, label="dataset")
    rows = dataset.get("rows")
    if not isinstance(rows, list):
        raise MLXEffectiveSpendTriageSelectionError("dataset rows[] missing")
    return [row for row in rows if isinstance(row, dict)]


def _is_candidate_row(
    row: dict[str, Any],
    *,
    families: set[str] | None,
    min_observed_gain: float,
    prediction_field: str,
    require_prediction_negative: bool,
    require_singleton_windows: bool,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    for key in _FALSE_AUTHORITY_FIELDS:
        if row.get(key) is not False:
            blockers.append(f"{key}_not_false")
    if row.get("axis") != EVIDENCE_TAG_MLX:
        blockers.append("axis_not_mlx_research_signal")
    if row.get("source_evidence_grade") != EVIDENCE_GRADE_MLX:
        blockers.append("source_evidence_grade_not_mlx")
    if row.get("source_evidence_tag") != EVIDENCE_TAG_MLX:
        blockers.append("source_evidence_tag_not_mlx")
    if families is not None and row.get("family") not in families:
        blockers.append("family_not_selected")
    if row.get("source_schema") != "mlx_scorer_response.v1":
        blockers.append("source_schema_not_mlx_scorer_response")

    total_delta = _as_float(row.get("delta_vs_baseline_score"))
    scorer_delta = _as_float(row.get("scorer_delta_vs_baseline"))
    observed_gain = _as_float(row.get("observed_scorer_gain_vs_baseline"))
    margin = _as_float(row.get("byte_budget_margin_vs_break_even"))
    if total_delta is None or total_delta >= -min_observed_gain:
        blockers.append("total_delta_not_calibrated_improvement")
    if scorer_delta is None or scorer_delta >= -min_observed_gain:
        blockers.append("scorer_delta_not_calibrated_improvement")
    if observed_gain is None or observed_gain < min_observed_gain:
        blockers.append("observed_gain_below_calibrated_gap")
    if margin is None or margin < 0.0:
        blockers.append("byte_budget_margin_negative_or_missing")

    if require_singleton_windows:
        if row.get("source_batch_pairs") != 1:
            blockers.append("source_batch_pairs_not_singleton")
        if row.get("source_n_samples") != 1:
            blockers.append("source_n_samples_not_singleton")
        pair_window = row.get("source_pair_window")
        if (
            not isinstance(pair_window, list)
            or len(pair_window) != 2
            or pair_window[1] - pair_window[0] != 1
        ):
            blockers.append("source_pair_window_not_singleton")

    predicted_delta = _as_float(row.get(prediction_field))
    if require_prediction_negative and (
        predicted_delta is None or predicted_delta >= 0.0
    ):
        blockers.append("prediction_not_negative")
    return not blockers, blockers


def _selection_row(
    row: dict[str, Any],
    *,
    rank: int,
    min_observed_gain: float,
    prediction_field: str,
) -> dict[str, Any]:
    predicted_delta = _as_float(row.get(prediction_field))
    return {
        "schema": ROW_SCHEMA,
        "rank": rank,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "candidate_generation_only": True,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "selection_basis": "observed_strict_gated_mlx_singleton_response_gain",
        "row_id": row.get("row_id"),
        "family": row.get("family"),
        "candidate_id": row.get("candidate_id"),
        "pair_indices": row.get("pair_indices"),
        "source_pair_window": row.get("source_pair_window"),
        "source_path": row.get("source_path"),
        "window_baseline_source_path": row.get("window_baseline_source_path"),
        "archive_sha256": row.get("archive_sha256"),
        "raw_sha256": row.get("raw_sha256"),
        "source_inflated_outputs_aggregate_sha256": row.get(
            "source_inflated_outputs_aggregate_sha256"
        ),
        "source_candidate_cache_array_sha256": row.get(
            "source_candidate_cache_array_sha256"
        ),
        "source_reference_cache_array_sha256": row.get(
            "source_reference_cache_array_sha256"
        ),
        "source_posenet_sha256": row.get("source_posenet_sha256"),
        "source_segnet_sha256": row.get("source_segnet_sha256"),
        "window_baseline_candidate_cache_array_sha256": row.get(
            "window_baseline_candidate_cache_array_sha256"
        ),
        "window_baseline_reference_cache_array_sha256": row.get(
            "window_baseline_reference_cache_array_sha256"
        ),
        "observed_delta_vs_baseline_score": row.get("delta_vs_baseline_score"),
        "observed_scorer_delta_vs_baseline": row.get("scorer_delta_vs_baseline"),
        "observed_scorer_gain_vs_baseline": row.get(
            "observed_scorer_gain_vs_baseline"
        ),
        "byte_budget_margin_vs_break_even": row.get(
            "byte_budget_margin_vs_break_even"
        ),
        "break_even_added_bytes_from_scorer_gain": row.get(
            "break_even_added_bytes_from_scorer_gain"
        ),
        "added_archive_bytes": row.get("added_archive_bytes"),
        "calibrated_min_mlx_gap_for_spend_triage": min_observed_gain,
        "prediction_field": prediction_field,
        "predicted_delta_vs_baseline_score": predicted_delta,
        "prediction_agrees_with_observed_gain": (
            None if predicted_delta is None else predicted_delta < 0.0
        ),
        "next_required_step": (
            "materialize_byte_closed_archive_before_claim_or_exact_eval_dispatch"
        ),
    }


def build_mlx_effective_spend_triage_selection(
    dataset: dict[str, Any],
    plan: dict[str, Any],
    *,
    top_k: int = 32,
    families: list[str] | None = None,
    min_observed_gain: float | None = None,
    prediction_field: str = DEFAULT_PREDICTION_FIELD,
    require_prediction_negative: bool = False,
    require_singleton_windows: bool = True,
    source_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select strict-gated MLX rows for candidate generation, never scoring."""

    if top_k <= 0:
        raise MLXEffectiveSpendTriageSelectionError("top_k must be positive")
    effective_gate = _validate_plan_for_selection(plan)
    rows = _validate_dataset(dataset)
    selected_families = None if families is None else set(families)
    if selected_families is not None and not selected_families:
        raise MLXEffectiveSpendTriageSelectionError("families cannot be empty")
    calibrated_gap = _calibrated_min_gap(plan, min_observed_gain)

    eligible: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    for row in rows:
        ok, blockers = _is_candidate_row(
            row,
            families=selected_families,
            min_observed_gain=calibrated_gap,
            prediction_field=prediction_field,
            require_prediction_negative=require_prediction_negative,
            require_singleton_windows=require_singleton_windows,
        )
        if ok:
            eligible.append(row)
        for blocker in blockers:
            rejection_counts[blocker] = rejection_counts.get(blocker, 0) + 1

    eligible.sort(
        key=lambda row: (
            -float(row["observed_scorer_gain_vs_baseline"]),
            -float(row["byte_budget_margin_vs_break_even"]),
            str(row.get("row_id")),
        )
    )
    selected = [
        _selection_row(
            row,
            rank=index + 1,
            min_observed_gain=calibrated_gap,
            prediction_field=prediction_field,
        )
        for index, row in enumerate(eligible[:top_k])
    ]
    if not selected:
        raise MLXEffectiveSpendTriageSelectionError(
            "no rows survived strict MLX spend-triage selection"
        )

    gains = [
        float(row["observed_scorer_gain_vs_baseline"])
        for row in eligible
        if _as_float(row.get("observed_scorer_gain_vs_baseline")) is not None
    ]
    prediction_agree_count = sum(
        1 for row in selected if row["prediction_agrees_with_observed_gain"] is True
    )
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "candidate_generation_only": True,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "allowed_use": (
            "candidate_generation_filter_after_strict_effective_mlx_spend_triage_gate"
        ),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "source_artifacts": source_artifacts or {},
        "gates": {
            "effective_mlx_spend_triage_gate": {
                "schema": effective_gate.get("schema"),
                "status": effective_gate.get("status"),
                "mlx_exact_eval_spend_triage_allowed": effective_gate.get(
                    "mlx_exact_eval_spend_triage_allowed"
                ),
                "allowed_use": effective_gate.get("allowed_use"),
            },
            "response_validation_status": plan["response_validation_gate"].get(
                "status"
            ),
            "torch_parity_status": plan["mlx_torch_parity_sweep_gate"].get(
                "status"
            ),
            "score_calibration_status": plan["mlx_score_calibration_gate"].get(
                "status"
            ),
            "production_contract_status": plan["mlx_production_contract_gate"].get(
                "status"
            ),
        },
        "selection_policy": {
            "top_k": top_k,
            "families": None if selected_families is None else sorted(selected_families),
            "min_observed_gain": calibrated_gap,
            "prediction_field": prediction_field,
            "require_prediction_negative": require_prediction_negative,
            "require_singleton_windows": require_singleton_windows,
            "sort": [
                "observed_scorer_gain_vs_baseline_desc",
                "byte_budget_margin_vs_break_even_desc",
                "row_id_asc",
            ],
        },
        "summary": {
            "dataset_row_count": len(rows),
            "eligible_row_count": len(eligible),
            "selected_count": len(selected),
            "rejected_row_count": len(rows) - len(eligible),
            "rejection_counts": rejection_counts,
            "observed_gain_min": None if not gains else min(gains),
            "observed_gain_max": None if not gains else max(gains),
            "prediction_agree_selected_count": prediction_agree_count,
            "prediction_disagree_selected_count": (
                len(selected) - prediction_agree_count
            ),
        },
        "selected_rows": selected,
    }


def render_mlx_effective_spend_triage_selection_markdown(
    selection: dict[str, Any],
) -> str:
    """Render a concise Markdown report for the MLX selection manifest."""

    summary = selection.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    policy = selection.get("selection_policy")
    if not isinstance(policy, dict):
        policy = {}
    gates = selection.get("gates")
    if not isinstance(gates, dict):
        gates = {}
    lines = [
        "# MLX Effective Spend-Triage Candidate Selection",
        "",
        f"- Selected rows: `{summary.get('selected_count')}`",
        f"- Eligible rows: `{summary.get('eligible_row_count')}`",
        f"- Dataset rows: `{summary.get('dataset_row_count')}`",
        f"- Calibrated min MLX gap: `{policy.get('min_observed_gain')}`",
        f"- Prediction agree selected rows: `{summary.get('prediction_agree_selected_count')}`",
        f"- Prediction disagree selected rows: `{summary.get('prediction_disagree_selected_count')}`",
        "",
    ]
    lines.extend(render_authority_markdown_block(selection))
    lines.extend(
        [
            "## Gates",
            "",
            f"- Effective gate: `{gates.get('effective_mlx_spend_triage_gate')}`",
            f"- Response validation: `{gates.get('response_validation_status')}`",
            f"- Torch parity: `{gates.get('torch_parity_status')}`",
            f"- Score calibration: `{gates.get('score_calibration_status')}`",
            f"- Production contract: `{gates.get('production_contract_status')}`",
            "",
            "## Policy",
            "",
            f"- Top K: `{policy.get('top_k')}`",
            f"- Families: `{policy.get('families')}`",
            f"- Prediction field: `{policy.get('prediction_field')}`",
            f"- Require prediction negative: `{policy.get('require_prediction_negative')}`",
            f"- Require singleton windows: `{policy.get('require_singleton_windows')}`",
            "",
            "## Selected Rows",
            "",
        ]
    )
    for row in selection.get("selected_rows", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "- rank={rank} family=`{family}` pair=`{pair}` "
            "gain=`{gain}` margin=`{margin}` predicted_delta=`{pred}` "
            "source=`{source}`".format(
                rank=row.get("rank"),
                family=row.get("family"),
                pair=row.get("source_pair_window"),
                gain=row.get("observed_scorer_gain_vs_baseline"),
                margin=row.get("byte_budget_margin_vs_break_even"),
                pred=row.get("predicted_delta_vs_baseline_score"),
                source=row.get("source_path"),
            )
        )
    lines.extend(
        [
            "",
            "## Required Next Step",
            "",
            (
                "Selected rows are strict-gated `[macOS-MLX research-signal]` "
                "candidate-generation inputs only. They require byte-closed archive "
                "materialization and claimed contest CPU/CUDA auth eval before any "
                "score, rank, promotion, kill, or submission decision."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def source_artifact_metadata(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Build deterministic source path/hash metadata for CLI manifests."""

    metadata: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        metadata[label] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    return metadata


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object with a selector-specific error message."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MLXEffectiveSpendTriageSelectionError(f"{path}: expected JSON object")
    return payload


def dumps_json(payload: dict[str, Any]) -> str:
    """Stable JSON writer used by the CLI and tests."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


__all__ = [
    "DEFAULT_PREDICTION_FIELD",
    "ROW_SCHEMA",
    "SCHEMA",
    "MLXEffectiveSpendTriageSelectionError",
    "build_mlx_effective_spend_triage_selection",
    "dumps_json",
    "load_json_object",
    "render_mlx_effective_spend_triage_selection_markdown",
    "source_artifact_metadata",
]
