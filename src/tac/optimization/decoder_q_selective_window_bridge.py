# SPDX-License-Identifier: MIT
"""Bridge strict MLX decoder-q windows into byte-closed work orders.

The bridge deliberately does not claim that selected MLX windows are score
authority. It turns them into auditable runtime work units and records the
missing selective decoder-q grammar that must exist before dispatch.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.optimization.decoder_q_selective_runtime_packet import FEC6_PAIR_COUNT
from tac.optimization.normalized_objective import (
    RATE_SCORE_PER_BYTE,
    NormalizedObjectiveError,
    compute_normalized_full_video_gain,
    require_normalized_full_video_objective,
)
from tac.optimization.scorer_response_dataset import render_authority_markdown_block

SCHEMA = "decoder_q_selective_window_bridge_plan.v1"
TOOL = "tac.optimization.decoder_q_selective_window_bridge"
SELECTION_SCHEMA = "mlx_effective_spend_triage_candidate_selection.v1"
SELECTION_ROW_SCHEMA = "mlx_effective_spend_triage_candidate_row.v1"
CANDIDATE_MANIFEST_SCHEMA = "fec6_decoder_q_materialized_candidate_v1"
CANONICAL_SELECTION_BASIS = "normalized_full_video_mlx_singleton_response_gain"
LEGACY_OBSERVED_SELECTION_BASIS = "observed_strict_gated_mlx_singleton_response_gain"
_NORMALIZED_OBJECTIVE_FIELDS = (
    "source_n_samples",
    "full_video_denominator",
    "normalized_full_video_scorer_gain_vs_baseline",
    "projected_full_video_delta_vs_baseline_score",
    "break_even_added_bytes_from_normalized_full_video_gain",
    "normalized_full_video_byte_budget_margin_vs_break_even",
)

_STRICT_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "ready_for_exact_eval_dispatch",
    "rank_or_kill_eligible",
    "promotable",
)
_MATERIALIZED_CANDIDATE_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "promotion_eligible",
    "ready_for_exact_eval_dispatch",
)


class DecoderQSelectiveWindowBridgeError(ValueError):
    """Raised when a decoder-q selective bridge plan would lose custody."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DecoderQSelectiveWindowBridgeError(f"{path}: expected JSON object")
    return payload


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _as_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DecoderQSelectiveWindowBridgeError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise DecoderQSelectiveWindowBridgeError(f"{label} must be finite")
    return result


def _as_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise DecoderQSelectiveWindowBridgeError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DecoderQSelectiveWindowBridgeError(f"{label} must be an integer") from exc
    if result != value and not (isinstance(value, str) and str(result) == value):
        raise DecoderQSelectiveWindowBridgeError(f"{label} must be integral")
    return result


def _require_false_authority(
    payload: dict[str, Any],
    *,
    label: str,
    fields: tuple[str, ...] = _STRICT_FALSE_AUTHORITY_FIELDS,
) -> None:
    for key in fields:
        if payload.get(key) is not False:
            raise DecoderQSelectiveWindowBridgeError(
                f"{label} {key} must be explicit false"
            )


def _resolve_artifact_path(path_value: Any, *, repo_root: Path) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise DecoderQSelectiveWindowBridgeError("artifact path must be a non-empty string")
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _validate_pair_window(row: dict[str, Any], *, label: str) -> tuple[int, int]:
    pair_window = row.get("source_pair_window")
    if not isinstance(pair_window, list) or len(pair_window) != 2:
        raise DecoderQSelectiveWindowBridgeError(f"{label} source_pair_window invalid")
    start = _as_int(pair_window[0], label=f"{label} source_pair_window[0]")
    end = _as_int(pair_window[1], label=f"{label} source_pair_window[1]")
    if end <= start:
        raise DecoderQSelectiveWindowBridgeError(
            f"{label} source_pair_window must be increasing"
        )
    if row.get("pair_indices") != pair_window:
        raise DecoderQSelectiveWindowBridgeError(
            f"{label} pair_indices must match source_pair_window"
        )
    return start, end


def _canonicalize_selection_row_objective(
    row: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    start, end = _validate_pair_window(row, label=label)
    basis = row.get("selection_basis")
    if basis not in {
        CANONICAL_SELECTION_BASIS,
        LEGACY_OBSERVED_SELECTION_BASIS,
    }:
        raise DecoderQSelectiveWindowBridgeError(f"{label} selection_basis mismatch")

    canonical = dict(row)
    if basis != CANONICAL_SELECTION_BASIS:
        canonical["legacy_selection_basis"] = basis
        canonical["selection_basis"] = CANONICAL_SELECTION_BASIS
        canonical["selection_planning_value_scope"] = "normalized_full_video"

    missing_normalized_fields = any(canonical.get(key) is None for key in _NORMALIZED_OBJECTIVE_FIELDS)
    if missing_normalized_fields:
        source_n_samples = end - start
        canonical.setdefault("source_n_samples", source_n_samples)
        canonical.setdefault("full_video_denominator", FEC6_PAIR_COUNT)
        source_n_samples = _as_int(
            canonical.get("source_n_samples"),
            label=f"{label} source_n_samples",
        )
        denominator = _as_int(
            canonical.get("full_video_denominator"),
            label=f"{label} full_video_denominator",
        )
        observed_gain = _as_float(
            canonical.get("observed_scorer_gain_vs_baseline"),
            label=f"{label} observed_scorer_gain_vs_baseline",
        )
        added_archive_bytes = _as_float(
            canonical.get("added_archive_bytes"),
            label=f"{label} added_archive_bytes",
        )
        normalized_gain = compute_normalized_full_video_gain(
            observed_gain,
            source_n_samples,
            full_video_denominator=denominator,
        )
        break_even_bytes = normalized_gain / RATE_SCORE_PER_BYTE
        canonical["normalized_full_video_scorer_gain_vs_baseline"] = normalized_gain
        canonical["projected_full_video_delta_vs_baseline_score"] = (
            RATE_SCORE_PER_BYTE * added_archive_bytes - normalized_gain
        )
        canonical["break_even_added_bytes_from_normalized_full_video_gain"] = (
            break_even_bytes
        )
        canonical["normalized_full_video_byte_budget_margin_vs_break_even"] = (
            break_even_bytes - added_archive_bytes
        )
        canonical["normalized_objective_backfilled"] = True
        canonical["normalized_objective_backfill_source"] = (
            "decoder_q_selective_window_bridge.v1"
        )

    return canonical


def _validate_selection(selection: dict[str, Any]) -> list[dict[str, Any]]:
    if selection.get("schema") != SELECTION_SCHEMA:
        raise DecoderQSelectiveWindowBridgeError("selection schema mismatch")
    _require_false_authority(selection, label="selection")
    if selection.get("candidate_generation_only") is not True:
        raise DecoderQSelectiveWindowBridgeError(
            "selection candidate_generation_only must be true"
        )
    if selection.get("archive_materialization_required") is not True:
        raise DecoderQSelectiveWindowBridgeError(
            "selection archive_materialization_required must be true"
        )
    if selection.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise DecoderQSelectiveWindowBridgeError("selection evidence_grade must be MLX")
    if selection.get("evidence_tag") != EVIDENCE_TAG_MLX:
        raise DecoderQSelectiveWindowBridgeError("selection evidence_tag must be MLX")
    rows = selection.get("selected_rows")
    if not isinstance(rows, list) or not rows:
        raise DecoderQSelectiveWindowBridgeError("selection selected_rows[] missing")

    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DecoderQSelectiveWindowBridgeError(
                f"selection row {index} must be an object"
            )
        label = f"selection row {index}"
        if row.get("schema") != SELECTION_ROW_SCHEMA:
            raise DecoderQSelectiveWindowBridgeError(f"{label} schema mismatch")
        _require_false_authority(row, label=label)
        if row.get("candidate_generation_only") is not True:
            raise DecoderQSelectiveWindowBridgeError(
                f"{label} candidate_generation_only must be true"
            )
        if row.get("archive_materialization_required") is not True:
            raise DecoderQSelectiveWindowBridgeError(
                f"{label} archive_materialization_required must be true"
            )
        if row.get("requires_exact_auth_eval_before_score_claim") is not True:
            raise DecoderQSelectiveWindowBridgeError(
                f"{label} must require exact auth eval before score claim"
            )
        if row.get("family") != "mlx_decoder_q":
            raise DecoderQSelectiveWindowBridgeError(f"{label} family must be mlx_decoder_q")
        row = _canonicalize_selection_row_objective(row, label=label)
        _validate_pair_window(row, label=label)
        if _as_float(row.get("observed_scorer_gain_vs_baseline"), label=f"{label} gain") <= 0:
            raise DecoderQSelectiveWindowBridgeError(f"{label} gain must be positive")
        if (
            _as_float(
                row.get("normalized_full_video_scorer_gain_vs_baseline"),
                label=f"{label} normalized full-video gain",
            )
            <= 0
        ):
            raise DecoderQSelectiveWindowBridgeError(
                f"{label} normalized full-video gain must be positive"
            )
        if (
            _as_float(
                row.get("projected_full_video_delta_vs_baseline_score"),
                label=f"{label} projected full-video delta",
            )
            >= 0
        ):
            raise DecoderQSelectiveWindowBridgeError(
                f"{label} projected full-video delta must be negative"
            )
        if (
            _as_float(
                row.get("normalized_full_video_byte_budget_margin_vs_break_even"),
                label=f"{label} normalized margin",
            )
            < 0
        ):
            raise DecoderQSelectiveWindowBridgeError(f"{label} margin must be non-negative")
        try:
            require_normalized_full_video_objective(row, label=f"{label} normalized objective")
        except NormalizedObjectiveError as exc:
            raise DecoderQSelectiveWindowBridgeError(str(exc)) from exc
        validated.append(row)
    return validated


def _validate_candidate_manifest(
    candidate_manifest: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    if candidate_manifest.get("schema") != CANDIDATE_MANIFEST_SCHEMA:
        raise DecoderQSelectiveWindowBridgeError("candidate manifest schema mismatch")
    _require_false_authority(
        candidate_manifest,
        label="candidate manifest",
        fields=_MATERIALIZED_CANDIDATE_FALSE_AUTHORITY_FIELDS,
    )
    candidate_id = candidate_manifest.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise DecoderQSelectiveWindowBridgeError("candidate manifest candidate_id missing")
    archive_sha = candidate_manifest.get("archive_zip_sha256")
    if not isinstance(archive_sha, str) or len(archive_sha) != 64:
        raise DecoderQSelectiveWindowBridgeError(
            "candidate manifest archive_zip_sha256 missing"
        )
    archive_path = _resolve_artifact_path(
        candidate_manifest.get("archive_zip_path"), repo_root=repo_root
    )
    if not archive_path.is_file():
        raise DecoderQSelectiveWindowBridgeError(
            f"candidate archive file missing: {archive_path}"
        )
    actual_sha = file_sha256(archive_path)
    if actual_sha != archive_sha:
        raise DecoderQSelectiveWindowBridgeError(
            "candidate archive file SHA mismatch: "
            f"manifest={archive_sha} actual={actual_sha}"
        )
    mutation_row = candidate_manifest.get("mutation_row")
    if not isinstance(mutation_row, dict):
        raise DecoderQSelectiveWindowBridgeError("candidate mutation_row missing")
    mutation = mutation_row.get("mutation")
    if not isinstance(mutation, dict):
        raise DecoderQSelectiveWindowBridgeError("candidate mutation missing")
    tensor_name = mutation.get("tensor_name")
    if not isinstance(tensor_name, str) or not tensor_name:
        raise DecoderQSelectiveWindowBridgeError("candidate mutation tensor_name missing")
    _as_int(mutation.get("q_offset"), label="candidate mutation q_offset")
    _as_int(mutation.get("delta"), label="candidate mutation delta")
    if mutation_row.get("fixed_length_runtime_compatible") is not True:
        raise DecoderQSelectiveWindowBridgeError(
            "candidate must be fixed_length_runtime_compatible"
        )
    if _as_int(mutation_row.get("length_delta"), label="candidate length_delta") != 0:
        raise DecoderQSelectiveWindowBridgeError("candidate length_delta must be zero")
    return {
        "candidate_id": candidate_id,
        "archive_zip_path": str(archive_path),
        "archive_zip_bytes": candidate_manifest.get("archive_zip_bytes"),
        "archive_zip_sha256": archive_sha,
        "archive_bin_sha256": candidate_manifest.get("archive_bin_sha256"),
        "mutation": {
            "tensor_name": tensor_name,
            "q_offset": int(mutation["q_offset"]),
            "delta": int(mutation["delta"]),
            "q_before": mutation_row.get("q_before"),
            "q_after": mutation_row.get("q_after"),
            "source_decoder_sha256": mutation_row.get("source_decoder_sha256"),
            "mutated_decoder_sha256": mutation_row.get("mutated_decoder_sha256"),
            "raw_q_range": (
                mutation_row.get("tensor", {}).get("raw_q_range")
                if isinstance(mutation_row.get("tensor"), dict)
                else None
            ),
            "approx_compressed_range": (
                mutation_row.get("op3v3_target_evidence", {}).get(
                    "approx_compressed_range"
                )
                if isinstance(mutation_row.get("op3v3_target_evidence"), dict)
                else None
            ),
        },
    }


def _row_source_metadata(row: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    source_path = _resolve_artifact_path(row.get("source_path"), repo_root=repo_root)
    baseline_path = _resolve_artifact_path(
        row.get("window_baseline_source_path"), repo_root=repo_root
    )
    for label, path in (
        ("source_path", source_path),
        ("window_baseline_source_path", baseline_path),
    ):
        if not path.is_file():
            raise DecoderQSelectiveWindowBridgeError(f"{label} missing: {path}")
    return {
        "source_path": str(source_path),
        "source_path_sha256": file_sha256(source_path),
        "window_baseline_source_path": str(baseline_path),
        "window_baseline_source_path_sha256": file_sha256(baseline_path),
    }


def _build_window_unit(
    row: dict[str, Any],
    *,
    candidate: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    start, end = _validate_pair_window(row, label=f"row {row.get('rank')}")
    archive_sha = row.get("archive_sha256")
    if archive_sha != candidate["archive_zip_sha256"]:
        raise DecoderQSelectiveWindowBridgeError(
            "selected row archive_sha256 does not match materialized candidate: "
            f"row={archive_sha} candidate={candidate['archive_zip_sha256']}"
        )
    source_meta = _row_source_metadata(row, repo_root=repo_root)
    observed_gain = _as_float(
        row.get("observed_scorer_gain_vs_baseline"),
        label=f"row {row.get('rank')} observed gain",
    )
    source_n_samples = _as_int(
        row.get("source_n_samples"),
        label=f"row {row.get('rank')} source_n_samples",
    )
    window_n_samples = end - start
    if source_n_samples != window_n_samples:
        raise DecoderQSelectiveWindowBridgeError(
            "selected row source_n_samples must match pair_window width: "
            f"row={row.get('rank')} source_n_samples={source_n_samples} "
            f"pair_window={[start, end]}"
        )
    denominator = _as_int(
        row.get("full_video_denominator"),
        label=f"row {row.get('rank')} full_video_denominator",
    )
    if denominator != FEC6_PAIR_COUNT:
        raise DecoderQSelectiveWindowBridgeError(
            f"row {row.get('rank')} full_video_denominator must be {FEC6_PAIR_COUNT}"
        )
    normalized_gain = observed_gain * float(source_n_samples) / float(denominator)
    upstream_normalized_gain = row.get(
        "normalized_full_video_scorer_gain_vs_baseline"
    )
    upstream_value = _as_float(
        upstream_normalized_gain,
        label=f"row {row.get('rank')} normalized gain",
    )
    if not math.isclose(upstream_value, normalized_gain, rel_tol=0.0, abs_tol=1e-15):
        raise DecoderQSelectiveWindowBridgeError(
            "selected row normalized_full_video_scorer_gain_vs_baseline "
            f"mismatch: row={row.get('rank')} upstream={upstream_value} "
            f"computed={normalized_gain}"
        )
    margin = _as_float(
        row.get("normalized_full_video_byte_budget_margin_vs_break_even"),
        label=f"row {row.get('rank')} normalized byte margin",
    )
    rank = _as_int(row.get("rank"), label="row rank")
    return {
        "schema": "decoder_q_selective_window_work_unit.v1",
        "unit_id": f"decoder_q_window_{start:04d}_{end:04d}",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "candidate_generation_only": True,
        "rank": rank,
        "source_row_id": row.get("row_id"),
        "source_candidate_id": row.get("candidate_id"),
        "source_selection_basis": row.get("selection_basis"),
        "legacy_selection_basis": row.get("legacy_selection_basis"),
        "normalized_objective_backfilled": row.get(
            "normalized_objective_backfilled",
            False,
        ),
        "normalized_objective_backfill_source": row.get(
            "normalized_objective_backfill_source"
        ),
        "pair_window": [start, end],
        "materialized_decoder_q_candidate_id": candidate["candidate_id"],
        "archive_sha256": archive_sha,
        "observed_mlx_window_gain": observed_gain,
        "source_n_samples": source_n_samples,
        "full_video_denominator": denominator,
        "normalized_full_video_gain": normalized_gain,
        "normalized_full_video_scorer_gain_vs_baseline": normalized_gain,
        "projected_full_video_delta_vs_baseline_score": row.get(
            "projected_full_video_delta_vs_baseline_score"
        ),
        "break_even_added_bytes_from_normalized_full_video_gain": row.get(
            "break_even_added_bytes_from_normalized_full_video_gain"
        ),
        "normalized_full_video_byte_budget_margin_vs_break_even": margin,
        "added_archive_bytes": row.get("added_archive_bytes"),
        "observed_scorer_gain_vs_baseline": observed_gain,
        "predicted_delta_vs_baseline_score": row.get(
            "predicted_delta_vs_baseline_score"
        ),
        "prediction_agrees_with_observed_gain": row.get(
            "prediction_agrees_with_observed_gain"
        ),
        "raw_sha256": row.get("raw_sha256"),
        "source_inflated_outputs_aggregate_sha256": row.get(
            "source_inflated_outputs_aggregate_sha256"
        ),
        "source_posenet_sha256": row.get("source_posenet_sha256"),
        "source_segnet_sha256": row.get("source_segnet_sha256"),
        "source_candidate_cache_array_sha256": row.get(
            "source_candidate_cache_array_sha256"
        ),
        "source_reference_cache_array_sha256": row.get(
            "source_reference_cache_array_sha256"
        ),
        "window_baseline_candidate_cache_array_sha256": row.get(
            "window_baseline_candidate_cache_array_sha256"
        ),
        "window_baseline_reference_cache_array_sha256": row.get(
            "window_baseline_reference_cache_array_sha256"
        ),
        **source_meta,
        "runtime_work_required": (
            "encode decoder-q tensor mutation as a selective per-window runtime arm"
        ),
    }


def _coalesce_window_runs(
    units: list[dict[str, Any]],
    *,
    coalesce_gap: int,
) -> list[dict[str, Any]]:
    if coalesce_gap < 0:
        raise DecoderQSelectiveWindowBridgeError("coalesce_gap must be non-negative")
    sorted_units = sorted(
        units,
        key=lambda unit: (
            int(unit["pair_window"][0]),
            int(unit["pair_window"][1]),
            int(unit["rank"]),
        ),
    )
    runs: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    active_start: int | None = None
    active_end: int | None = None

    def flush() -> None:
        nonlocal active, active_start, active_end
        if not active:
            return
        assert active_start is not None and active_end is not None
        ranks = sorted(int(unit["rank"]) for unit in active)
        raw_gain_sum = sum(float(unit["observed_mlx_window_gain"]) for unit in active)
        normalized_gain_sum = sum(
            float(unit["normalized_full_video_gain"]) for unit in active
        )
        runs.append(
            {
                "schema": "decoder_q_selective_window_run.v1",
                "run_id": f"decoder_q_run_{active_start:04d}_{active_end:04d}",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "candidate_generation_only": True,
                "pair_window": [active_start, active_end],
                "window_count": len(active),
                "source_ranks": ranks,
                "unit_ids": [str(unit["unit_id"]) for unit in active],
                "local_mlx_window_gain_sum_non_authoritative": raw_gain_sum,
                "normalized_full_video_gain_sum_non_authoritative": (
                    normalized_gain_sum
                ),
                "full_video_denominator": FEC6_PAIR_COUNT,
                "gain_additivity_assumption": (
                    "not_assumed_batch_behavior_requires_runtime_probe"
                ),
            }
        )
        active = []
        active_start = None
        active_end = None

    for unit in sorted_units:
        start, end = int(unit["pair_window"][0]), int(unit["pair_window"][1])
        if not active:
            active = [unit]
            active_start, active_end = start, end
            continue
        assert active_end is not None
        if start <= active_end + coalesce_gap:
            active.append(unit)
            active_end = max(active_end, end)
            continue
        flush()
        active = [unit]
        active_start, active_end = start, end
    flush()
    return runs


def build_decoder_q_selective_window_bridge_plan(
    selection: dict[str, Any],
    candidate_manifest: dict[str, Any],
    *,
    repo_root: Path,
    lane_id: str,
    max_windows: int | None = None,
    coalesce_gap: int = 0,
    source_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic work order from strict MLX decoder-q windows."""

    if not lane_id:
        raise DecoderQSelectiveWindowBridgeError("lane_id is required")
    if max_windows is not None and max_windows <= 0:
        raise DecoderQSelectiveWindowBridgeError("max_windows must be positive")
    rows = _validate_selection(selection)
    rows = sorted(rows, key=lambda row: _as_int(row.get("rank"), label="row rank"))
    if max_windows is not None:
        rows = rows[:max_windows]
    candidate = _validate_candidate_manifest(candidate_manifest, repo_root=repo_root)
    units = [
        _build_window_unit(row, candidate=candidate, repo_root=repo_root)
        for row in rows
    ]
    ranks = [int(unit["rank"]) for unit in units]
    if len(set(ranks)) != len(ranks):
        raise DecoderQSelectiveWindowBridgeError("selected rows contain duplicate ranks")
    unit_ids = [str(unit["unit_id"]) for unit in units]
    if len(set(unit_ids)) != len(unit_ids):
        raise DecoderQSelectiveWindowBridgeError(
            "selected rows contain duplicate pair windows"
        )
    runs = _coalesce_window_runs(units, coalesce_gap=coalesce_gap)
    top_unit = units[0]
    selected_archive_shas = sorted({str(unit["archive_sha256"]) for unit in units})
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        "lane_id": lane_id,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "candidate_generation_only": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "bridge_status": "ready_for_dqs1_tail_trailer_materialization",
        "allowed_use": (
            "input_for_byte_closed_decoder_q_selective_runtime_materialization"
        ),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "source_artifacts": source_artifacts or {},
        "source_selection": {
            "schema": selection.get("schema"),
            "producer": selection.get("producer"),
            "selection_policy": selection.get("selection_policy"),
            "gates": selection.get("gates"),
            "summary": selection.get("summary"),
        },
        "materialized_decoder_q_candidate": candidate,
        "bridge_policy": {
            "max_windows": max_windows,
            "coalesce_gap": coalesce_gap,
            "sort": "selection_rank_ascending",
            "runtime_strategy": "dqs1_tail_trailer_selective_runtime",
        },
        "summary": {
            "selected_window_count": len(units),
            "coalesced_run_count": len(runs),
            "selected_archive_sha256_count": len(selected_archive_shas),
            "selected_archive_sha256": selected_archive_shas,
            "top_pair_window": top_unit["pair_window"],
            "top_observed_mlx_window_gain": top_unit["observed_mlx_window_gain"],
            "top_normalized_full_video_gain": top_unit["normalized_full_video_gain"],
            "top_normalized_full_video_byte_budget_margin_vs_break_even": top_unit[
                "normalized_full_video_byte_budget_margin_vs_break_even"
            ],
            "all_units_prediction_agree_count": sum(
                1 for unit in units if unit["prediction_agrees_with_observed_gain"] is True
            ),
            "all_units_prediction_disagree_count": sum(
                1
                for unit in units
                if unit["prediction_agrees_with_observed_gain"] is not True
            ),
        },
        "work_units": units,
        "coalesced_runs": runs,
        "dispatch_blockers": [
            "DQS1 packet materialization not run for this bridge plan",
            "official inflate.sh raw-output locality controls not run for selective packet",
            "claimed contest CPU/CUDA auth eval not run for selective packet",
            "MLX window gains are candidate-generation signal only",
        ],
        "required_next_steps": [
            "materialize DQS1 tail-trailer archive.zip for singleton and coalesced-run work units",
            "run official inflate.sh raw-output locality controls against FEC6 parent and selective packets",
            "run local advisory scorer only as a smoke gate with score_claim=false",
            "claim dispatch lane before any exact contest CPU/CUDA auth eval",
        ],
    }


def source_artifact_metadata(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        metadata[label] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    return metadata


def render_decoder_q_selective_window_bridge_markdown(plan: dict[str, Any]) -> str:
    summary = plan.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    candidate = plan.get("materialized_decoder_q_candidate")
    if not isinstance(candidate, dict):
        candidate = {}
    mutation = candidate.get("mutation")
    if not isinstance(mutation, dict):
        mutation = {}
    lines = [
        "# Decoder-Q Selective Window Bridge Plan",
        "",
        f"- Bridge status: `{plan.get('bridge_status')}`",
        f"- Selected windows: `{summary.get('selected_window_count')}`",
        f"- Coalesced runs: `{summary.get('coalesced_run_count')}`",
        f"- Materialized candidate: `{candidate.get('candidate_id')}`",
        f"- Archive SHA-256: `{candidate.get('archive_zip_sha256')}`",
        f"- Mutation: `{mutation.get('tensor_name')}` q_offset=`{mutation.get('q_offset')}` delta=`{mutation.get('delta')}`",
        f"- Top window: `{summary.get('top_pair_window')}`",
        f"- Top observed window gain: `{summary.get('top_observed_mlx_window_gain')}` `[macOS-MLX research-signal]`",
        f"- Top normalized full-video gain: `{summary.get('top_normalized_full_video_gain')}` `[macOS-MLX research-signal]`",
        "",
    ]
    lines.extend(render_authority_markdown_block(plan))
    lines.extend(
        [
            "## Dispatch Blockers",
            "",
        ]
    )
    for blocker in plan.get("dispatch_blockers", []):
        lines.append(f"- {blocker}")
    lines.extend(["", "## Work Units", ""])
    for unit in plan.get("work_units", []):
        if not isinstance(unit, dict):
            continue
        lines.append(
            "- rank={rank} unit=`{unit}` pair=`{pair}` gain=`{gain}` "
            "normalized_full_video_gain=`{normalized}` margin=`{margin}` "
            "prediction_agrees=`{agree}`".format(
                rank=unit.get("rank"),
                unit=unit.get("unit_id"),
                pair=unit.get("pair_window"),
                gain=unit.get("observed_mlx_window_gain"),
                normalized=unit.get("normalized_full_video_gain"),
                margin=unit.get(
                    "normalized_full_video_byte_budget_margin_vs_break_even"
                ),
                agree=unit.get("prediction_agrees_with_observed_gain"),
            )
        )
    lines.extend(["", "## Coalesced Runs", ""])
    for run in plan.get("coalesced_runs", []):
        if not isinstance(run, dict):
            continue
        lines.append(
            "- run=`{run}` pair=`{pair}` windows=`{count}` ranks=`{ranks}` "
            "window_gain_sum_non_authoritative=`{gain}` "
            "normalized_gain_sum_non_authoritative=`{normalized}`".format(
                run=run.get("run_id"),
                pair=run.get("pair_window"),
                count=run.get("window_count"),
                ranks=run.get("source_ranks"),
                gain=run.get("local_mlx_window_gain_sum_non_authoritative"),
                normalized=run.get(
                    "normalized_full_video_gain_sum_non_authoritative"
                ),
            )
        )
    lines.extend(["", "## Required Next Steps", ""])
    for step in plan.get("required_next_steps", []):
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "SCHEMA",
    "DecoderQSelectiveWindowBridgeError",
    "build_decoder_q_selective_window_bridge_plan",
    "dumps_json",
    "load_json_object",
    "render_decoder_q_selective_window_bridge_markdown",
    "source_artifact_metadata",
]
