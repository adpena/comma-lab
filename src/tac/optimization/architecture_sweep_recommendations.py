"""Normalize CPU/MPS architecture sweep manifests into planning recommendations.

This module is intentionally score-claim hostile. It accepts cheap local
architecture, sparsity, and self-compression planning artifacts and emits a
single manifest that can guide local follow-up work without promoting any row
to exact-eval dispatch. Contest score use still requires a produced archive,
strict packet gates, a lane claim, and exact CUDA auth eval.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "architecture_sweep_recommendations.v1"
TOOL_NAME = "tac.optimization.architecture_sweep_recommendations"

RESEARCH_SIGNAL_BLOCKERS = (
    "cpu_mps_research_signal_not_score_evidence",
    "no_score_claim_from_normalized_manifest",
    "not_exact_cuda_auth_eval",
    "no_adjudicated_contest_archive",
    "not_promotion_eligible",
)

EXACT_CUDA_PROMOTION_GATES = (
    "produce_score_affecting_archive_with_changed_payload_bytes",
    "record_archive_bytes_sha256_runtime_tree_and_manifest",
    "run_strict_pre_submission_compliance_check",
    "claim_lane_dispatch_before_any_gpu_eval",
    "run_full_sample_contest_cuda_auth_eval",
    "adjudicate_components_rate_term_and_payload_closure",
)

TERMINAL_CLAIM_PREFIXES = (
    "completed",
    "failed",
    "stopped",
    "stale",
    "refused",
)

CANONICAL_RESEARCH_EVIDENCE_GRADE = "empirical"


class ArchitectureSweepRecommendationError(ValueError):
    """Raised when a sweep source cannot be normalized safely."""


@dataclass(frozen=True)
class ManifestSource:
    """A loaded planning artifact and its stable source label."""

    source_path: str
    payload: Mapping[str, Any]


def load_manifest_source(path: Path) -> ManifestSource:
    """Load one JSON manifest source."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ArchitectureSweepRecommendationError(f"{path}: manifest root must be an object")
    return ManifestSource(source_path=path.as_posix(), payload=payload)


def load_manifest_sources(paths: Sequence[Path]) -> list[ManifestSource]:
    """Load multiple JSON manifest sources."""

    return [load_manifest_source(path) for path in paths]


def build_architecture_sweep_recommendations(
    sources: Iterable[ManifestSource],
    *,
    run_id: str,
    lightning_active_jobs: Sequence[Mapping[str, Any]] | None = None,
    dispatch_claims_markdown: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed architecture sweep recommendation manifest.

    Args:
        sources: existing local planning manifests. Supported inputs include
            MPS research-signal manifests, PR101 architecture-shrink plans,
            PR101 sparsity byte sweeps, and small self-compression summaries.
        run_id: stable run id for this normalizer pass.
        lightning_active_jobs: optional parsed `.omx/state/lightning_active_jobs.json`.
        dispatch_claims_markdown: optional active dispatch claim ledger text.
    """

    normalized_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for source in sources:
        rows = _rows_from_source(source)
        normalized_rows.extend(rows)
        source_records.append(
            {
                "source_path": source.source_path,
                "schema": _source_schema(source.payload),
                "row_count": len(rows),
            }
        )

    curves = _curve_summaries(normalized_rows)
    active_jobs = _arch_shrink_active_jobs(lightning_active_jobs or [])
    active_claims = _arch_shrink_claims(dispatch_claims_markdown or "")
    recommendations = _recommendations(
        rows=normalized_rows,
        curves=curves,
        active_jobs=active_jobs,
        active_claims=active_claims,
    )

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "run_id": run_id,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "evidence_grade": CANONICAL_RESEARCH_EVIDENCE_GRADE,
        "evidence_semantics": "cpu_mps_curve_shape_and_byte_planning_only",
        "allowed_uses": [
            "local_architecture_sweep_planning",
            "curve_shape_triage",
            "candidate_generation_prior",
            "dispatch_blocker_review",
        ],
        "forbidden_uses": [
            "score_claim",
            "promotion",
            "rank_frontier_candidate",
            "method_retirement",
            "exact_eval_dispatch_authorization",
        ],
        "source_manifests": source_records,
        "row_count": len(normalized_rows),
        "rows": normalized_rows,
        "curve_count": len(curves),
        "curves": curves,
        "lightning_arch_shrink_state": {
            "active_jobs": active_jobs,
            "active_claims": active_claims,
            "do_not_duplicate_dispatch": bool(active_jobs or active_claims),
        },
        "dispatch_recommendations": recommendations,
        "dispatch_blockers": list(RESEARCH_SIGNAL_BLOCKERS),
        "exact_cuda_promotion_gates": list(EXACT_CUDA_PROMOTION_GATES),
    }


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact operator-facing Markdown summary."""

    lines = [
        f"# Architecture Sweep Recommendations - {manifest.get('run_id', '<unknown>')}",
        "",
        "## Evidence Boundary",
        "",
        "- `score_claim=false`; this is CPU/MPS research-signal planning only.",
        "- `ready_for_exact_eval_dispatch=false`; rows cannot authorize GPU dispatch.",
        "- Exact CUDA promotion still requires the gates listed in the JSON manifest.",
        "",
        "## Curves",
        "",
        "| family | curve | device | points | metric | best research row | dispatch |",
        "|---|---|---|---:|---|---|---|",
    ]
    for curve in manifest.get("curves", []):
        if not isinstance(curve, Mapping):
            continue
        best = curve.get("best_research_signal_row") or {}
        if not isinstance(best, Mapping):
            best = {}
        lines.append(
            "| {family} | {curve_id} | {device} | {points} | {metric} | {best_variant} | {dispatch} |".format(
                family=curve.get("family", ""),
                curve_id=curve.get("curve_id", ""),
                device=curve.get("device_family", ""),
                points=curve.get("point_count", 0),
                metric=curve.get("metric_name", ""),
                best_variant=best.get("variant_id", ""),
                dispatch=curve.get("dispatch_recommendation", ""),
            )
        )

    jobs = (manifest.get("lightning_arch_shrink_state") or {}).get("active_jobs") or []
    claims = (manifest.get("lightning_arch_shrink_state") or {}).get("active_claims") or []
    if jobs or claims:
        lines.extend(["", "## Lightning State", ""])
        for job in jobs:
            if isinstance(job, Mapping):
                lines.append(
                    "- Active arch-shrink job `{job_name}` submitted `{submitted}` terminal_status `{terminal}`.".format(
                        job_name=job.get("job_name", ""),
                        submitted=job.get("submitted_at_utc", ""),
                        terminal=job.get("terminal_status"),
                    )
                )
        for claim in claims:
            if isinstance(claim, Mapping):
                lines.append(
                    "- Active claim `{lane_id}` job `{instance_job_id}` status `{status}` expires `{expires}`.".format(
                        lane_id=claim.get("lane_id", ""),
                        instance_job_id=claim.get("instance_job_id", ""),
                        status=claim.get("status", ""),
                        expires=claim.get("expires_at", ""),
                    )
                )

    lines.extend(["", "## Recommendations", ""])
    for item in manifest.get("dispatch_recommendations", []):
        if isinstance(item, Mapping):
            lines.append(f"- `{item.get('recommendation')}`: {item.get('reason')}")
    return "\n".join(lines) + "\n"


def json_text(payload: Any) -> str:
    """Deterministic JSON text for manifest outputs."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def load_lightning_active_jobs(path: Path | None) -> list[dict[str, Any]]:
    """Load local Lightning active-job state if present."""

    if path is None or not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ArchitectureSweepRecommendationError(f"{path}: expected JSON list")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def load_dispatch_claims_text(path: Path | None) -> str:
    """Load the dispatch claim ledger text if present."""

    if path is None or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _source_schema(payload: Mapping[str, Any]) -> str:
    schema = payload.get("schema", payload.get("schema_version"))
    if schema is not None:
        return str(schema)
    if _looks_like_self_compress_summary(payload):
        return "renderer_self_compress_summary.inferred"
    if "headroom_test_general_compressors_on_renderer_bin" in payload:
        return "lane_e_self_compression_results.inferred"
    return "unknown"


def _rows_from_source(source: ManifestSource) -> list[dict[str, Any]]:
    payload = source.payload
    schema = _source_schema(payload)
    if schema == "mps_research_signal_manifest.v1":
        return _rows_from_mps_manifest(source)
    if schema == "pr101_arch_shrink_retraining_plan.v1":
        return _rows_from_arch_shrink_plan(source)
    if schema == "pr101_sparsity_block_sweep.v1":
        return _rows_from_sparsity_sweep(source)
    if schema == "renderer_self_compress_summary.inferred":
        return _rows_from_self_compress_summary(source)
    if schema == "lane_e_self_compression_results.inferred":
        return _rows_from_lane_e_self_compress(source)
    raise ArchitectureSweepRecommendationError(
        f"{source.source_path}: unsupported architecture sweep manifest schema {schema!r}"
    )


def _base_row(
    *,
    source: ManifestSource,
    family: str,
    curve_id: str,
    variant_id: str,
    device_family: str,
    evidence_grade: str,
    evidence_semantics: str,
    metric_name: str,
    metric_value: float | int | None,
    archive_bytes: int | None,
    payload_bytes: int | None = None,
    byte_delta_vs_anchor: int | None = None,
    params: Mapping[str, Any] | None = None,
    source_artifact: str | None = None,
    dispatch_blockers: Sequence[str] | None = None,
) -> dict[str, Any]:
    blockers = list(dict.fromkeys([*(dispatch_blockers or ()), *RESEARCH_SIGNAL_BLOCKERS]))
    return {
        "source_path": source.source_path,
        "source_schema": _source_schema(source.payload),
        "source_artifact": source_artifact or "",
        "family": family,
        "curve_id": curve_id,
        "variant_id": variant_id,
        "device_family": device_family,
        "evidence_grade": evidence_grade,
        "evidence_semantics": evidence_semantics,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_direction": "lower_is_better",
        "archive_bytes": archive_bytes,
        "payload_bytes": payload_bytes,
        "byte_delta_vs_anchor": byte_delta_vs_anchor,
        "params": dict(params or {}),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatchable": False,
        "research_signal_only": True,
        "dispatch_recommendation": "do_not_dispatch_from_cpu_mps_research_signal",
        "dispatch_blockers": blockers,
    }


def _rows_from_mps_manifest(source: ManifestSource) -> list[dict[str, Any]]:
    payload = source.payload
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ArchitectureSweepRecommendationError(f"{source.source_path}: MPS manifest rows[] missing")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        metric_name, metric_value = _pick_metric(
            row,
            ("proxy_loss", "proxy_formula_value", "d_seg_proxy", "d_pose_proxy"),
        )
        out.append(
            _base_row(
                source=source,
                family=str(row.get("family") or "arch_sweep"),
                curve_id=str(row.get("curve_id") or row.get("family") or "mps_curve"),
                variant_id=str(row.get("variant_id") or f"row_{len(out)}"),
                device_family="mps",
                evidence_grade=CANONICAL_RESEARCH_EVIDENCE_GRADE,
                evidence_semantics=str(
                    row.get("evidence_semantics")
                    or payload.get("evidence_semantics")
                    or "mps_proxy_curve_shape_only"
                ),
                metric_name=metric_name,
                metric_value=metric_value,
                archive_bytes=_optional_int(row.get("archive_bytes")),
                byte_delta_vs_anchor=_optional_int(row.get("byte_delta_vs_anchor")),
                params=_mapping_or_empty(row.get("params")),
                source_artifact=str(row.get("source_artifact") or payload.get("source") or ""),
                dispatch_blockers=_string_list(row.get("dispatch_blockers")),
            )
        )
    return out


def _rows_from_arch_shrink_plan(source: ManifestSource) -> list[dict[str, Any]]:
    rows = source.payload.get("scenarios")
    if not isinstance(rows, list):
        raise ArchitectureSweepRecommendationError(f"{source.source_path}: arch-shrink scenarios[] missing")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        estimate = _mapping_or_empty(row.get("byte_estimate"))
        targets = _mapping_or_empty(row.get("targets"))
        archive_bytes = _optional_int(estimate.get("expected_archive_bytes"))
        metric_value: float | int | None = archive_bytes
        out.append(
            _base_row(
                source=source,
                family=str(row.get("driver_family") or "hnerv_arch_shrink"),
                curve_id=str(row.get("driver_family") or "hnerv_arch_shrink"),
                variant_id=str(row.get("name") or f"scenario_{len(out)}"),
                device_family="cpu",
                evidence_grade="prediction",
                evidence_semantics="cpu_rate_side_architecture_prediction_no_score",
                metric_name="expected_archive_bytes",
                metric_value=metric_value,
                archive_bytes=archive_bytes,
                byte_delta_vs_anchor=_optional_int(estimate.get("delta_archive_bytes_vs_reference")),
                params=targets,
                dispatch_blockers=_string_list(row.get("dispatch_blockers")),
            )
        )
    return out


def _rows_from_sparsity_sweep(source: ManifestSource) -> list[dict[str, Any]]:
    rows = source.payload.get("rows")
    if not isinstance(rows, list):
        raise ArchitectureSweepRecommendationError(f"{source.source_path}: sparsity rows[] missing")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        alpha = row.get("alpha")
        out.append(
            _base_row(
                source=source,
                family="pr101_sparsity",
                curve_id="post_hoc_sparsity_alpha",
                variant_id=f"alpha_{_variant_number(alpha)}",
                device_family="cpu",
                evidence_grade=CANONICAL_RESEARCH_EVIDENCE_GRADE,
                evidence_semantics="post_hoc_byte_anchor_no_retraining_no_score",
                metric_name="archive_bytes",
                metric_value=_optional_int(row.get("archive_bytes")),
                archive_bytes=_optional_int(row.get("archive_bytes")),
                params={"alpha": alpha, "fraction_zeroed": row.get("fraction_zeroed")},
                dispatch_blockers=_string_list(source.payload.get("dispatch_blockers")),
            )
        )
    return out


def _rows_from_self_compress_summary(source: ManifestSource) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model_name, summary in sorted(source.payload.items()):
        if not isinstance(summary, Mapping):
            continue
        for field in ("uniform_i4lz_bytes", "mixed_latents8_bytes"):
            payload_bytes = _optional_int(summary.get(field))
            if payload_bytes is None:
                continue
            out.append(
                _base_row(
                    source=source,
                    family="renderer_self_compress",
                    curve_id=str(model_name),
                    variant_id=f"{model_name}:{field}",
                    device_family="cpu",
                    evidence_grade=CANONICAL_RESEARCH_EVIDENCE_GRADE,
                    evidence_semantics="local_renderer_self_compression_payload_bytes_no_score",
                    metric_name="payload_bytes",
                    metric_value=payload_bytes,
                    archive_bytes=None,
                    payload_bytes=payload_bytes,
                    params={
                        "param_count": summary.get("param_count"),
                        "state_path": summary.get("state_path"),
                        "codec": field,
                    },
                    source_artifact=str(summary.get("state_path") or ""),
                )
            )
    return out


def _rows_from_lane_e_self_compress(source: ManifestSource) -> list[dict[str, Any]]:
    test = _mapping_or_empty(source.payload.get("headroom_test_general_compressors_on_renderer_bin"))
    original = _optional_int(test.get("original_bytes"))
    brotli = _optional_int(test.get("brotli_q11_bytes"))
    rows: list[dict[str, Any]] = []
    if original is not None:
        rows.append(
            _base_row(
                source=source,
                family="renderer_bin_recompression",
                curve_id="general_compressor_headroom",
                variant_id="original_renderer_bin",
                device_family="cpu",
                evidence_grade=CANONICAL_RESEARCH_EVIDENCE_GRADE,
                evidence_semantics="generic_renderer_bin_recompression_probe_no_score",
                metric_name="payload_bytes",
                metric_value=original,
                archive_bytes=None,
                payload_bytes=original,
            )
        )
    if brotli is not None:
        rows.append(
            _base_row(
                source=source,
                family="renderer_bin_recompression",
                curve_id="general_compressor_headroom",
                variant_id="brotli_q11_renderer_bin",
                device_family="cpu",
                evidence_grade=CANONICAL_RESEARCH_EVIDENCE_GRADE,
                evidence_semantics="generic_renderer_bin_recompression_probe_no_score",
                metric_name="payload_bytes",
                metric_value=brotli,
                archive_bytes=None,
                payload_bytes=brotli,
                params={"rate_savings_brotli": test.get("rate_savings_brotli")},
            )
        )
    return rows


def _curve_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("family") or ""),
                str(row.get("curve_id") or ""),
                str(row.get("device_family") or ""),
            )
        ].append(row)

    curves: list[dict[str, Any]] = []
    for (family, curve_id, device), curve_rows in sorted(grouped.items()):
        ordered = sorted(
            curve_rows,
            key=lambda row: (
                _sort_number(row.get("archive_bytes"), row.get("payload_bytes")),
                str(row.get("variant_id") or ""),
            ),
        )
        metric_name = _common_metric_name(ordered)
        best = _best_row(ordered, metric_name)
        curves.append(
            {
                "family": family,
                "curve_id": curve_id,
                "device_family": device,
                "point_count": len(ordered),
                "metric_name": metric_name,
                "metric_direction": "lower_is_better",
                "byte_range": _byte_range(ordered),
                "finite_difference_slopes": _slopes(ordered, metric_name),
                "best_research_signal_row": best,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatchable": False,
                "dispatch_blockers": list(RESEARCH_SIGNAL_BLOCKERS),
                "dispatch_recommendation": "do_not_dispatch_from_cpu_mps_research_signal",
                "interpretation": "candidate_generation_prior_only",
            }
        )
    return curves


def _arch_shrink_active_jobs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        lane_id = str(row.get("lane_id") or "")
        job_name = str(row.get("job_name") or "")
        if "arch_shrink" not in lane_id and "arch-shrink" not in job_name and "arch_shrink" not in job_name:
            continue
        terminal = row.get("terminal_status")
        if terminal is not None:
            continue
        out.append(
            {
                "lane_id": lane_id,
                "job_name": job_name,
                "submitted_at_utc": row.get("submitted_at_utc"),
                "machine": row.get("machine"),
                "profile": row.get("profile"),
                "target_elements": row.get("target_elements"),
                "predicted_band": row.get("predicted_band"),
                "expected_artifact_dir": row.get("expected_artifact_dir"),
                "expected_auth_eval_json": row.get("expected_auth_eval_json"),
                "terminal_status": terminal,
            }
        )
    out.sort(key=lambda item: str(item.get("submitted_at_utc") or ""), reverse=True)
    return out


def _arch_shrink_claims(text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    closed_keys: set[tuple[str, str]] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) < 8 or parts[0] == "timestamp_utc":
            continue
        lane_id = parts[2]
        instance_job_id = parts[4]
        if not _is_arch_shrink_identifier(lane_id, instance_job_id):
            continue
        key = (lane_id, instance_job_id)
        status = parts[6]
        if _is_terminal_status(status):
            closed_keys.add(key)
            continue
        if key in closed_keys:
            continue
        claims.append(
            {
                "timestamp_utc": parts[0],
                "agent": parts[1],
                "lane_id": lane_id,
                "platform": parts[3],
                "instance_job_id": instance_job_id,
                "expires_at": parts[5],
                "status": status,
                "notes": parts[7],
            }
        )
    return claims


def _recommendations(
    *,
    rows: Sequence[Mapping[str, Any]],
    curves: Sequence[Mapping[str, Any]],
    active_jobs: Sequence[Mapping[str, Any]],
    active_claims: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    if active_jobs or active_claims:
        recommendations.append(
            {
                "recommendation": "do_not_duplicate_active_arch_shrink_lightning_dispatch",
                "reason": "Local Lightning state or claim ledger already has an active arch-shrink lane.",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatchable": False,
                "dispatch_blockers": list(RESEARCH_SIGNAL_BLOCKERS),
            }
        )
    if curves:
        best_curves = sorted(
            curves,
            key=lambda curve: (
                _sort_number(
                    (curve.get("best_research_signal_row") or {}).get("archive_bytes")
                    if isinstance(curve.get("best_research_signal_row"), Mapping)
                    else None,
                    (curve.get("best_research_signal_row") or {}).get("payload_bytes")
                    if isinstance(curve.get("best_research_signal_row"), Mapping)
                    else None,
                ),
                str(curve.get("family") or ""),
            ),
        )[:3]
        recommendations.append(
            {
                "recommendation": "use_top_research_curves_for_local_build_order_only",
                "reason": "Cheapest normalized curves can choose CPU/MPS follow-up order, not score rank or GPU dispatch.",
                "candidate_generation_curve_ids": [
                    {
                        "family": curve.get("family"),
                        "curve_id": curve.get("curve_id"),
                        "best_variant_id": (
                            (curve.get("best_research_signal_row") or {}).get("variant_id")
                            if isinstance(curve.get("best_research_signal_row"), Mapping)
                            else None
                        ),
                        "candidate_generation_only": True,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                        "dispatchable": False,
                        "dispatch_blockers": list(RESEARCH_SIGNAL_BLOCKERS),
                    }
                    for curve in best_curves
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatchable": False,
                "dispatch_blockers": list(RESEARCH_SIGNAL_BLOCKERS),
            }
        )
    if rows:
        recommendations.append(
            {
                "recommendation": "preserve_exact_cuda_promotion_gates",
                "reason": "Every normalized row is CPU/MPS or prediction evidence and must pass archive, compliance, lane-claim, and full CUDA auth-eval gates before score use.",
                "gates": list(EXACT_CUDA_PROMOTION_GATES),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatchable": False,
                "dispatch_blockers": list(RESEARCH_SIGNAL_BLOCKERS),
            }
        )
    return recommendations


def _pick_metric(row: Mapping[str, Any], candidates: Sequence[str]) -> tuple[str, float | int | None]:
    for name in candidates:
        value = row.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return name, value
    return candidates[0], None


def _common_metric_name(rows: Sequence[Mapping[str, Any]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("metric_name") or "")] += 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _best_row(rows: Sequence[Mapping[str, Any]], metric_name: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("metric_name") == metric_name and isinstance(row.get("metric_value"), (int, float))
    ]
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda row: (
            float(row["metric_value"]),
            _sort_number(row.get("archive_bytes"), row.get("payload_bytes")),
        ),
    )
    return {
        "variant_id": best.get("variant_id"),
        "metric_name": best.get("metric_name"),
        "metric_value": best.get("metric_value"),
        "archive_bytes": best.get("archive_bytes"),
        "payload_bytes": best.get("payload_bytes"),
        "candidate_generation_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatchable": False,
        "dispatch_blockers": list(RESEARCH_SIGNAL_BLOCKERS),
    }


def _byte_range(rows: Sequence[Mapping[str, Any]]) -> list[int] | None:
    values = [
        _sort_number(row.get("archive_bytes"), row.get("payload_bytes"))
        for row in rows
        if _sort_number(row.get("archive_bytes"), row.get("payload_bytes")) < math.inf
    ]
    if not values:
        return None
    return [int(min(values)), int(max(values))]


def _slopes(rows: Sequence[Mapping[str, Any]], metric_name: str) -> list[dict[str, Any]]:
    sortable = [
        row
        for row in rows
        if _sort_number(row.get("archive_bytes"), row.get("payload_bytes")) < math.inf
        and row.get("metric_name") == metric_name
        and isinstance(row.get("metric_value"), (int, float))
    ]
    ordered = sorted(
        sortable,
        key=lambda row: (_sort_number(row.get("archive_bytes"), row.get("payload_bytes")), str(row.get("variant_id") or "")),
    )
    out: list[dict[str, Any]] = []
    for previous, current in pairwise(ordered):
        prev_bytes = _sort_number(previous.get("archive_bytes"), previous.get("payload_bytes"))
        cur_bytes = _sort_number(current.get("archive_bytes"), current.get("payload_bytes"))
        byte_step = int(cur_bytes - prev_bytes)
        if byte_step == 0:
            continue
        metric_delta = float(current["metric_value"]) - float(previous["metric_value"])
        out.append(
            {
                "from_variant_id": previous.get("variant_id"),
                "to_variant_id": current.get("variant_id"),
                "byte_step": byte_step,
                "metric_delta": metric_delta,
                "metric_delta_per_byte": metric_delta / float(byte_step),
            }
        )
    return out


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _sort_number(*values: Any) -> float:
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None:
            return float(parsed)
    return math.inf


def _variant_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "_")
    return str(value)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _looks_like_self_compress_summary(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False
    return all(
        isinstance(value, Mapping)
        and "param_count" in value
        and ("uniform_i4lz_bytes" in value or "mixed_latents8_bytes" in value)
        for value in payload.values()
    )


def _is_terminal_status(status: str) -> bool:
    lowered = status.lower()
    return any(lowered.startswith(prefix) for prefix in TERMINAL_CLAIM_PREFIXES)


def _is_arch_shrink_identifier(*values: str) -> bool:
    return any(
        "arch_shrink" in value or "arch-shrink" in value
        for value in values
    )


__all__ = [
    "CANONICAL_RESEARCH_EVIDENCE_GRADE",
    "EXACT_CUDA_PROMOTION_GATES",
    "SCHEMA_VERSION",
    "ArchitectureSweepRecommendationError",
    "ManifestSource",
    "build_architecture_sweep_recommendations",
    "json_text",
    "load_dispatch_claims_text",
    "load_lightning_active_jobs",
    "load_manifest_source",
    "load_manifest_sources",
    "render_markdown",
]
