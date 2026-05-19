# SPDX-License-Identifier: MIT
"""Block-aware trust-region plans for master-gradient operator candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.repo_io import read_json, repo_relative, sha256_file

SCHEMA = "master_gradient_trust_region_candidate_manifest_v1"
TRUST_REGION_MODES: tuple[str, ...] = (
    "uniform",
    "gradient_weighted",
    "stc_style",
    "segnet_boundary_preserving",
)
PROMOTION_BLOCKERS: tuple[str, ...] = (
    "candidate_archive_not_materialized",
    "packet_proofs_missing",
    "inflate_success_proof_missing",
    "runtime_byte_consumption_noop_detector_missing",
    "score_response_probe_missing",
    "exact_eval_missing",
)


def build_master_gradient_trust_region_candidates(
    *,
    operator_manifest: Mapping[str, Any],
    repo_root: Path,
    independence_report: Mapping[str, Any] | None = None,
    score_response_outcome: Mapping[str, Any] | None = None,
    modes: Sequence[str] = TRUST_REGION_MODES,
    max_rows_per_candidate: int = 8,
    base_mutation_intensity: float = 0.25,
    pair_block_size: int | None = None,
) -> dict[str, Any]:
    """Build score-claim-false trust-region candidate manifests.

    The output is a compiler plan, not a packet. It consumes the OP-7 exact
    negative as a trust-region boundary and the 600-pair independence diagnostic
    as a block-aware assumption guard.
    """

    if max_rows_per_candidate <= 0:
        raise ValueError("max_rows_per_candidate must be > 0")
    if not (0.0 < base_mutation_intensity <= 1.0):
        raise ValueError("base_mutation_intensity must be in (0, 1]")
    unknown_modes = [mode for mode in modes if mode not in TRUST_REGION_MODES]
    if unknown_modes:
        raise ValueError(f"unknown trust-region modes: {unknown_modes!r}")

    entries = _operator_entries(operator_manifest, repo_root=repo_root)
    block_model = _block_aware_pair_model(
        independence_report,
        operator_manifest=operator_manifest,
        pair_block_size=pair_block_size,
    )
    regression_guard = _score_response_regression_guard(score_response_outcome)
    intensity = min(
        float(base_mutation_intensity),
        float(block_model["recommended_mutation_intensity_cap"]),
        float(regression_guard["recommended_mutation_intensity_cap"]),
    )

    candidates = [
        _candidate_for_mode(
            mode,
            entries=entries,
            max_rows=max_rows_per_candidate,
            mutation_intensity=intensity,
            block_model=block_model,
            regression_guard=regression_guard,
        )
        for mode in modes
    ]
    return {
        "schema": SCHEMA,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "source_operator_manifest": _source_operator_summary(operator_manifest, repo_root),
        "source_negative_control": regression_guard,
        "block_aware_pair_model": block_model,
        "candidate_count": len(candidates),
        "trust_region_candidates": candidates,
        "blockers": list(PROMOTION_BLOCKERS),
        "next_step": (
            "Materialize the smallest candidate through a grammar-aware packet builder, "
            "then run score-response probes before any exact-eval dispatch."
        ),
    }


def _operator_entries(
    operator_manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    selector_entries = _selector_entries(operator_manifest, repo_root=repo_root)
    selector_by_rank = {
        int(entry.get("rank")): entry
        for entry in selector_entries
        if _is_int_like(entry.get("rank"))
    }
    resolution = operator_manifest.get("grammar_aware_operator_candidate_resolution")
    if not isinstance(resolution, Mapping):
        resolution = {}
    resolved_rows = resolution.get("resolved_pose_axis_candidates")
    if not isinstance(resolved_rows, list):
        resolved_rows = []
    specs = operator_manifest.get("candidate_modification_specs")
    if not isinstance(specs, list):
        specs = []
    spec_by_id = {
        spec.get("spec_id"): spec
        for spec in specs
        if isinstance(spec, Mapping) and isinstance(spec.get("spec_id"), str)
    }

    entries: list[dict[str, Any]] = []
    for row in resolved_rows:
        if not isinstance(row, Mapping):
            continue
        rank = int(row.get("rank", len(entries) + 1))
        selector = selector_by_rank.get(rank, {})
        spec = spec_by_id.get(row.get("spec_id"), {})
        seg = _float(selector.get("seg_axis_abs_score_contribution"), default=0.0)
        pose = _float(selector.get("pose_axis_abs_score_contribution"), default=0.0)
        rate = _float(selector.get("rate_axis_abs_score_contribution"), default=0.0)
        entries.append(
            {
                "rank": rank,
                "spec_id": row.get("spec_id") or spec.get("spec_id"),
                "operator_id": spec.get("operator_id"),
                "diagnostic_gradient_subject_byte_index": row.get(
                    "diagnostic_gradient_subject_byte_index"
                ),
                "section_name": row.get("section_name"),
                "section_role": row.get("section_role"),
                "section_relative_offset": row.get("section_relative_offset"),
                "mutation_operator": row.get("mutation_operator"),
                "pose_axis_share": _float(selector.get("pose_axis_share"), default=0.0),
                "pose_axis_abs_score_contribution": pose,
                "seg_axis_abs_score_contribution": seg,
                "rate_axis_abs_score_contribution": rate,
                "seg_to_pose_score_ratio": seg / pose if pose > 0 else math.inf,
                "source_spec_score_claim": spec.get("score_claim", False),
                "source_spec_ready_for_provider_dispatch": spec.get(
                    "ready_for_provider_dispatch", False
                ),
            }
        )
    if not entries:
        raise ValueError("operator manifest has no resolved pose-axis candidates")
    return sorted(entries, key=lambda e: (int(e["rank"]), str(e.get("spec_id") or "")))


def _selector_entries(
    operator_manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[Mapping[str, Any]]:
    embedded = operator_manifest.get("selector_sidecar_payload")
    if isinstance(embedded, Mapping):
        dominance = embedded.get("score_axis_dominance")
        if isinstance(dominance, Mapping) and isinstance(dominance.get("selected"), list):
            return [e for e in dominance["selected"] if isinstance(e, Mapping)]

    path_value = operator_manifest.get("selector_sidecar_path")
    if not isinstance(path_value, str) or not path_value:
        return []
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        return []
    payload = read_json(path)
    dominance = payload.get("score_axis_dominance") if isinstance(payload, Mapping) else None
    if isinstance(dominance, Mapping) and isinstance(dominance.get("selected"), list):
        return [e for e in dominance["selected"] if isinstance(e, Mapping)]
    return []


def _block_aware_pair_model(
    independence_report: Mapping[str, Any] | None,
    *,
    operator_manifest: Mapping[str, Any],
    pair_block_size: int | None,
) -> dict[str, Any]:
    n_pairs_total = _source_anchor_int(operator_manifest, "n_pairs_total", default=600)
    report_series = independence_report.get("series_reports") if independence_report else None
    series = [s for s in report_series if isinstance(s, Mapping)] if isinstance(report_series, list) else []
    min_ess = min(
        (_float(s.get("serial_effective_sample_size"), default=float(n_pairs_total)) for s in series),
        default=float(n_pairs_total),
    )
    max_abs_autocorr = max(
        (_float(s.get("max_abs_autocorrelation"), default=0.0) for s in series),
        default=0.0,
    )
    max_lag = 0
    params = independence_report.get("parameters") if independence_report else None
    if isinstance(params, Mapping):
        max_lag = int(params.get("max_lag") or 0)
    if max_lag <= 0 and series:
        first_lags = series[0].get("lag_autocorrelation")
        if isinstance(first_lags, Mapping):
            max_lag = max((int(k) for k in first_lags if str(k).isdigit()), default=0)
    block_size = int(pair_block_size or max(1, max_lag + 1, math.ceil(n_pairs_total / max(1.0, min_ess))))
    return {
        "iid_assumption_allowed": False,
        "source_schema": independence_report.get("schema") if independence_report else None,
        "aggregate_verdict": independence_report.get("aggregate_verdict") if independence_report else None,
        "cross_series_verdict": (
            independence_report.get("cross_series_dependence", {}).get("verdict")
            if isinstance(independence_report, Mapping)
            and isinstance(independence_report.get("cross_series_dependence"), Mapping)
            else None
        ),
        "n_pairs_total": int(n_pairs_total),
        "serial_effective_sample_size_floor": float(min_ess),
        "max_abs_autocorrelation": float(max_abs_autocorr),
        "temporal_block_size_pairs": block_size,
        "temporal_block_count": math.ceil(n_pairs_total / block_size),
        "recommended_mutation_intensity_cap": max(
            0.05,
            min(0.25, math.sqrt(max(1.0, min_ess) / max(1.0, n_pairs_total))),
        ),
    }


def _score_response_regression_guard(
    outcome: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cpu_delta = _float(outcome.get("contest_cpu_total_delta"), default=0.0) if outcome else 0.0
    cuda_delta = _float(outcome.get("contest_cuda_total_delta"), default=0.0) if outcome else 0.0
    exact_regression = cpu_delta > 0.0 or cuda_delta > 0.0
    return {
        "probe_id": outcome.get("probe_id") if outcome else None,
        "verdict": outcome.get("verdict") if outcome else None,
        "source_archive_sha256": outcome.get("source_archive_sha256") if outcome else None,
        "retired_candidate_archive_sha256": outcome.get("candidate_archive_sha256") if outcome else None,
        "contest_cpu_total_delta": cpu_delta,
        "contest_cuda_total_delta": cuda_delta,
        "same_length_raw_delta_regressed": exact_regression,
        "rerun_same_archive_forbidden": exact_regression,
        "recommended_mutation_intensity_cap": 0.25 if exact_regression else 1.0,
    }


def _candidate_for_mode(
    mode: str,
    *,
    entries: list[dict[str, Any]],
    max_rows: int,
    mutation_intensity: float,
    block_model: Mapping[str, Any],
    regression_guard: Mapping[str, Any],
) -> dict[str, Any]:
    ranked = _rank_entries_for_mode(mode, entries)
    selected = ranked[:max_rows]
    weights = _mode_weights(mode, selected)
    candidate_rows = []
    for entry, weight in zip(selected, weights, strict=True):
        candidate_rows.append(
            {
                **entry,
                "trust_region_weight": weight,
                "mutation_intensity": mutation_intensity * weight,
                "byte_delta_semantics": (
                    "fractional_intensity_for_future_stochastic_or_symbolic_packet_builder"
                ),
            }
        )
    return {
        "candidate_id": f"master_gradient_trust_region::{mode}",
        "mode": mode,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "selected_row_count": len(candidate_rows),
        "base_mutation_intensity": mutation_intensity,
        "temporal_block_size_pairs": block_model["temporal_block_size_pairs"],
        "serial_effective_sample_size_floor": block_model["serial_effective_sample_size_floor"],
        "same_length_raw_delta_regression_guard": regression_guard[
            "same_length_raw_delta_regressed"
        ],
        "rows": candidate_rows,
        "blockers": list(PROMOTION_BLOCKERS),
        "rationale": _mode_rationale(mode),
    }


def _rank_entries_for_mode(mode: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if mode == "uniform":
        return sorted(entries, key=lambda e: int(e["rank"]))
    if mode == "gradient_weighted":
        return sorted(
            entries,
            key=lambda e: (
                -float(e["pose_axis_abs_score_contribution"]),
                float(e["seg_axis_abs_score_contribution"]),
                int(e["rank"]),
            ),
        )
    if mode == "stc_style":
        return sorted(
            entries,
            key=lambda e: (
                float(e["seg_to_pose_score_ratio"]),
                -float(e["pose_axis_share"]),
                int(e["rank"]),
            ),
        )
    if mode == "segnet_boundary_preserving":
        return sorted(
            entries,
            key=lambda e: (
                float(e["seg_axis_abs_score_contribution"]) > 0.0,
                float(e["seg_axis_abs_score_contribution"]),
                -float(e["pose_axis_abs_score_contribution"]),
                int(e["rank"]),
            ),
        )
    raise AssertionError(mode)


def _mode_weights(mode: str, entries: list[dict[str, Any]]) -> list[float]:
    if not entries:
        return []
    if mode == "gradient_weighted":
        raw = [max(0.0, float(e["pose_axis_abs_score_contribution"])) for e in entries]
    elif mode == "segnet_boundary_preserving":
        raw = [
            max(0.0, float(e["pose_axis_abs_score_contribution"]))
            / (1.0 + max(0.0, float(e["seg_axis_abs_score_contribution"])) * 1000.0)
            for e in entries
        ]
    else:
        raw = [1.0 for _ in entries]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(entries) for _ in entries]
    return [float(v / total) for v in raw]


def _mode_rationale(mode: str) -> str:
    return {
        "uniform": "small uniform intensity across the resolved grammar-aware OP-7 rows",
        "gradient_weighted": "allocate more trust-region mass to rows with larger pose-axis contribution",
        "stc_style": "treat selected rows as a sparse syndrome carrier; packet builder must enforce code constraints",
        "segnet_boundary_preserving": "prefer rows with zero or minimal SegNet-axis contribution before pose gain",
    }[mode]


def _source_operator_summary(operator_manifest: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    path_value = operator_manifest.get("manifest_path")
    summary = {
        "schema": operator_manifest.get("schema"),
        "archive_sha256": operator_manifest.get("archive_sha256"),
        "blockers": operator_manifest.get("blockers"),
        "manifest_path": path_value,
    }
    if isinstance(path_value, str) and path_value:
        path = Path(path_value)
        if not path.is_absolute():
            path = repo_root / path
        if path.is_file():
            summary["manifest_sha256"] = sha256_file(path)
            summary["manifest_path"] = repo_relative(path, repo_root)
    return summary


def _source_anchor_int(
    operator_manifest: Mapping[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    source_anchor = operator_manifest.get("source_anchor")
    if isinstance(source_anchor, Mapping) and _is_int_like(source_anchor.get(key)):
        return int(source_anchor[key])
    return default


def _float(value: object, *, default: float) -> float:
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_int_like(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


__all__ = [
    "PROMOTION_BLOCKERS",
    "SCHEMA",
    "TRUST_REGION_MODES",
    "build_master_gradient_trust_region_candidates",
]
