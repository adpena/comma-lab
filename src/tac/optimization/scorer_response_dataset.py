# SPDX-License-Identifier: MIT
"""Dataset builder for scorer-response surrogate probes.

This module turns existing advisory candidate artifacts into a small,
fail-closed response table. It does not run evals, train models, dispatch
jobs, or claim scores. The point is to give LL-style backprop/saliency lanes a
held-out response surface instead of trusting a single local gradient.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.optimization.candidate_evidence_contract import CONTEST_UNCOMPRESSED_BYTES

SCHEMA = "scorer_response_dataset.v1"
ROW_SCHEMA = "scorer_response_row.v1"
NULL_BYTE_PRIORITY_SCHEMA = "ll_null_byte_priority_weights.v1"


_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "promotion_eligible",
    "ready_for_exact_eval_dispatch",
    "rank_or_kill_eligible",
    "promotable",
)

_CORE_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "promotion_eligible",
    "ready_for_exact_eval_dispatch",
)

_LEGACY_EXTENDED_FALSE_AUTHORITY_FIELDS = (
    "rank_or_kill_eligible",
    "promotable",
)

_SOURCE_OPTIONAL_FALSE_AUTHORITY_FIELDS = (
    "score_claim_valid",
    "promotion_eligible",
    "ready_for_exact_eval_dispatch",
    "rank_or_kill_eligible",
    "promotable",
)


def _require_explicit_false_authority(
    payload: dict[str, Any],
    *,
    label: str,
    fields: tuple[str, ...] = _FALSE_AUTHORITY_FIELDS,
    allow_legacy_missing_authority: bool = False,
) -> list[str]:
    legacy_missing: list[str] = []
    for key in fields:
        if key not in payload or payload.get(key) is None:
            if allow_legacy_missing_authority:
                legacy_missing.append(key)
                continue
            raise ScorerResponseDatasetError(f"{label} {key} must be explicit false")
        if payload.get(key) is not False:
            raise ScorerResponseDatasetError(f"{label} {key} must be false")
    return legacy_missing


def _require_source_advisory_authority_false(
    *,
    parent_authority: dict[str, Any],
    candidate_authority: dict[str, Any],
    label: str,
) -> None:
    """Fail closed when advisory source evidence carries score authority.

    Historical scorer-response inputs did not consistently carry every
    promotion flag, so optional flags may be absent. But a source score claim
    must be explicit false, and any present authority flag must be false. This
    prevents LL training-data rows from laundering score-bearing evidence into
    a non-promotional normalized row.
    """

    score_claim_values = [
        value
        for value in (
            parent_authority.get("score_claim"),
            candidate_authority.get("score_claim"),
        )
        if value is not None
    ]
    if not score_claim_values:
        raise ScorerResponseDatasetError(
            f"{label} source score_claim must be explicit false"
        )
    if any(value is not False for value in score_claim_values):
        raise ScorerResponseDatasetError(f"{label} source score_claim must be false")

    for key in _SOURCE_OPTIONAL_FALSE_AUTHORITY_FIELDS:
        for source_label, authority in (
            ("parent", parent_authority),
            ("candidate", candidate_authority),
        ):
            if key in authority and authority.get(key) is not False:
                raise ScorerResponseDatasetError(
                    f"{label} {source_label} authority {key} must be false"
                )


TOOL = "tac.optimization.scorer_response_dataset"
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_UNCOMPRESSED_BYTES


class ScorerResponseDatasetError(ValueError):
    """Raised when scorer-response artifacts cannot be normalized."""


def _validate_core_false_authority(payload: dict[str, Any], *, label: str) -> None:
    for key in _CORE_FALSE_AUTHORITY_FIELDS:
        if key not in payload or payload.get(key) is None:
            raise ScorerResponseDatasetError(f"{label} {key} must be explicit false")
        if payload.get(key) is not False:
            raise ScorerResponseDatasetError(f"{label} {key} must be false")


def _backfill_extended_false_authority(
    payload: dict[str, Any],
    *,
    label: str,
    repairs: list[dict[str, str]],
) -> None:
    for key in _LEGACY_EXTENDED_FALSE_AUTHORITY_FIELDS:
        if key not in payload or payload.get(key) is None:
            payload[key] = False
            repairs.append({"label": label, "field": key})
            continue
        if payload.get(key) is not False:
            raise ScorerResponseDatasetError(f"{label} {key} must be false")


def normalize_legacy_response_dataset_authority(
    dataset: dict[str, Any],
    *,
    source_label: str | None = None,
) -> dict[str, Any]:
    """Backfill old scorer-response datasets to the current false-authority shape.

    Only the historical extended fields ``rank_or_kill_eligible`` and
    ``promotable`` may be inserted as explicit ``False``. Core authority fields
    and row-level source-score authority must already be present and false.
    """

    if not isinstance(dataset, dict):
        raise ScorerResponseDatasetError("scorer-response dataset must be a JSON object")
    if dataset.get("schema") != SCHEMA:
        raise ScorerResponseDatasetError("scorer-response dataset schema mismatch")
    normalized = copy.deepcopy(dataset)
    repairs: list[dict[str, str]] = []

    _validate_core_false_authority(normalized, label="scorer-response dataset")
    _backfill_extended_false_authority(
        normalized,
        label="scorer-response dataset",
        repairs=repairs,
    )
    authority = normalized.get("authority")
    if not isinstance(authority, dict):
        raise ScorerResponseDatasetError("scorer-response dataset authority must be a JSON object")
    _validate_core_false_authority(authority, label="scorer-response dataset authority")
    _backfill_extended_false_authority(
        authority,
        label="scorer-response dataset authority",
        repairs=repairs,
    )

    rows = normalized.get("rows")
    if not isinstance(rows, list):
        raise ScorerResponseDatasetError("scorer-response dataset rows must be a list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ScorerResponseDatasetError(f"scorer-response row {index} must be a JSON object")
        if row.get("schema") != ROW_SCHEMA:
            raise ScorerResponseDatasetError(f"scorer-response row {index} schema mismatch")
        label = f"scorer-response row {index}"
        _validate_core_false_authority(row, label=label)
        _backfill_extended_false_authority(row, label=label, repairs=repairs)
        if "authority_source_score_claim" not in row or row.get("authority_source_score_claim") is None:
            raise ScorerResponseDatasetError(
                f"{label} authority_source_score_claim must be explicit false"
            )
        if row.get("authority_source_score_claim") is not False:
            raise ScorerResponseDatasetError(
                f"{label} authority_source_score_claim must be false"
            )

    normalized["authority_normalization"] = {
        "schema": "scorer_response_dataset_authority_normalization.v1",
        "producer": TOOL,
        "source_label": source_label,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "backfilled_missing_false_fields": repairs,
        "backfilled_missing_false_field_count": len(repairs),
    }
    return normalized


@dataclass(frozen=True)
class ResponseBaseline:
    score: float
    archive_bytes: int
    avg_posenet_dist: float | None = None
    avg_segnet_dist: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or self.score <= 0:
            raise ScorerResponseDatasetError("baseline score must be positive and finite")
        if self.archive_bytes <= 0:
            raise ScorerResponseDatasetError("baseline archive_bytes must be positive")
        if self.avg_posenet_dist is not None and self.avg_posenet_dist < 0:
            raise ScorerResponseDatasetError("baseline avg_posenet_dist must be non-negative")
        if self.avg_segnet_dist is not None and self.avg_segnet_dist < 0:
            raise ScorerResponseDatasetError("baseline avg_segnet_dist must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "archive_bytes": self.archive_bytes,
            "avg_posenet_dist": self.avg_posenet_dist,
            "avg_segnet_dist": self.avg_segnet_dist,
        }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScorerResponseDatasetError(f"{path}: expected JSON object")
    return payload


def _sha_fold(value: str, folds: int = 5) -> int:
    if folds <= 1:
        raise ScorerResponseDatasetError("fold count must be > 1")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def _get_path(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _score_terms(*, archive_bytes: int | None, pose: float | None, seg: float | None) -> dict[str, float | None]:
    rate = 25.0 * float(archive_bytes) / CONTEST_UNCOMPRESSED_BYTES if archive_bytes is not None else None
    pose_term = math.sqrt(10.0 * pose) if pose is not None and pose >= 0 else None
    seg_term = 100.0 * seg if seg is not None and seg >= 0 else None
    scorer = pose_term + seg_term if pose_term is not None and seg_term is not None else None
    total = scorer + rate if scorer is not None and rate is not None else None
    return {
        "rate_term": rate,
        "pose_term": pose_term,
        "seg_term": seg_term,
        "scorer_term": scorer,
        "recomputed_score_from_report_fields": total,
    }


def _candidate_items(path: Path, payload: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    if isinstance(payload.get("candidates"), list):
        out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for index, candidate in enumerate(payload["candidates"]):
            if isinstance(candidate, dict):
                cid = str(candidate.get("candidate_id") or candidate.get("spec_id") or index)
                out.append((cid, candidate, payload))
        return out
    if isinstance(payload.get("candidate"), dict):
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        cid = str(
            summary.get("candidate_id")
            or summary.get("spec_id")
            or payload["candidate"].get("candidate_id")
            or path.stem
        )
        return [(cid, payload["candidate"], payload)]
    raise ScorerResponseDatasetError(f"{path}: no candidate or candidates[] payload found")


def _family_for(path: Path, parent: dict[str, Any], candidate: dict[str, Any]) -> str:
    producer = str(parent.get("producer") or candidate.get("producer") or "")
    schema = str(parent.get("schema") or candidate.get("schema") or "")
    smoke_kind = str(
        parent.get("smoke_kind")
        or candidate.get("smoke_kind")
        or parent.get("probe_kind")
        or candidate.get("probe_kind")
        or ""
    )
    source = f"{producer} {schema} {smoke_kind} {path}".lower()
    if "distilled_vs_direct_scorer_paired_smoke" in source:
        return "distilled_vs_direct_scorer_paired_smoke"
    if "scorer_gradient" in source:
        return "scorer_gradient_sparse_residual"
    if "sparse_residual" in source:
        return "sparse_residual_oracle"
    if "postprocess" in source:
        return "inflate_postprocess"
    if "decoder_q" in source:
        return "decoder_q"
    return "unknown"


def _local_pair_summary(candidate: dict[str, Any]) -> dict[str, float | int | None]:
    evals = candidate.get("local_pair_evals")
    if not isinstance(evals, list) or not evals:
        return {
            "local_pair_count": 0,
            "local_pose_delta_sum": None,
            "local_seg_delta_sum": None,
            "local_worse_or_null_count": 0,
        }
    pose_sum = 0.0
    seg_sum = 0.0
    count = 0
    worse = 0
    for item in evals:
        if not isinstance(item, dict):
            continue
        delta = item.get("delta")
        if not isinstance(delta, dict):
            continue
        pose = _as_float(delta.get("pose_dist_delta"))
        seg = _as_float(delta.get("seg_dist_delta"))
        if pose is not None:
            pose_sum += pose
        if seg is not None:
            seg_sum += seg
        count += 1
        if item.get("worse_or_null") is True:
            worse += 1
    return {
        "local_pair_count": count,
        "local_pose_delta_sum": pose_sum if count else None,
        "local_seg_delta_sum": seg_sum if count else None,
        "local_worse_or_null_count": worse,
    }


def normalize_response_row(
    *,
    path: Path,
    candidate_id: str,
    candidate: dict[str, Any],
    parent: dict[str, Any],
    baseline: ResponseBaseline | None,
) -> dict[str, Any] | None:
    advisory = candidate.get("advisory_eval")
    if not isinstance(advisory, dict) or advisory.get("skipped") is True:
        return None
    score = _as_float(advisory.get("canonical_score"))
    archive_bytes = _as_int(advisory.get("archive_size_bytes") or _get_path(advisory, ("archive", "bytes")))
    pose = _as_float(advisory.get("avg_posenet_dist"))
    seg = _as_float(advisory.get("avg_segnet_dist"))
    if score is None and (archive_bytes is None or pose is None or seg is None):
        return None

    summary = parent.get("summary") if isinstance(parent.get("summary"), dict) else {}
    cand_summary = candidate.get("summary") if isinstance(candidate.get("summary"), dict) else {}
    plan = candidate.get("plan") if isinstance(candidate.get("plan"), dict) else {}
    apply_result = candidate.get("candidate") if isinstance(candidate.get("candidate"), dict) else {}
    archive = advisory.get("archive") if isinstance(advisory.get("archive"), dict) else {}
    authority = parent.get("authority") if isinstance(parent.get("authority"), dict) else {}
    candidate_authority = candidate.get("authority") if isinstance(candidate.get("authority"), dict) else {}
    row_id = f"{path}:{candidate_id}"
    _require_source_advisory_authority_false(
        parent_authority=authority,
        candidate_authority=candidate_authority,
        label=row_id,
    )
    terms = _score_terms(archive_bytes=archive_bytes, pose=pose, seg=seg)
    score_value = score if score is not None else terms["recomputed_score_from_report_fields"]
    if score_value is None:
        return None

    reported_delta = _as_float(candidate.get("delta_vs_baseline_score") or summary.get("delta_vs_baseline_score"))
    rate_delta = None
    scorer_delta = None
    added_archive_bytes = None
    break_even_added_bytes = None
    byte_budget_margin = None
    observed_scorer_gain = None
    required_scorer_gain_for_added_bytes = None
    scorer_gain_shortfall_to_break_even = None
    total_delta = reported_delta
    if baseline is not None:
        total_delta = score_value - baseline.score
        if archive_bytes is not None:
            added_archive_bytes = archive_bytes - baseline.archive_bytes
            rate_delta = RATE_SCORE_PER_BYTE * float(added_archive_bytes)
        if rate_delta is not None:
            scorer_delta = total_delta - rate_delta
        if added_archive_bytes is not None:
            required_scorer_gain_for_added_bytes = max(0.0, RATE_SCORE_PER_BYTE * float(added_archive_bytes))
        if scorer_delta is not None and scorer_delta < 0.0:
            observed_scorer_gain = -scorer_delta
            break_even_added_bytes = -scorer_delta / RATE_SCORE_PER_BYTE
            if added_archive_bytes is not None:
                byte_budget_margin = break_even_added_bytes - float(added_archive_bytes)
                scorer_gain_shortfall_to_break_even = max(
                    0.0,
                    required_scorer_gain_for_added_bytes - observed_scorer_gain,
                )
        elif required_scorer_gain_for_added_bytes is not None:
            observed_scorer_gain = 0.0
            scorer_gain_shortfall_to_break_even = required_scorer_gain_for_added_bytes
    local = _local_pair_summary(candidate)
    return {
        "schema": ROW_SCHEMA,
        "row_id": row_id,
        "holdout_fold": _sha_fold(row_id),
        "source_path": str(path),
        "family": _family_for(path, parent, candidate),
        "candidate_id": candidate_id,
        "axis": advisory.get("axis"),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "authority_source_score_claim": False,
        "authority_blockers": list(authority.get("promotion_blockers") or candidate_authority.get("promotion_blockers") or advisory.get("blockers") or []),
        "advisory_score_report_derived": score_value,
        "delta_vs_baseline_score": total_delta,
        "reported_delta_vs_baseline_score": reported_delta,
        "added_archive_bytes": added_archive_bytes,
        "rate_delta_vs_baseline": rate_delta,
        "scorer_delta_vs_baseline": scorer_delta,
        "observed_scorer_gain_vs_baseline": observed_scorer_gain,
        "required_scorer_gain_for_added_bytes": required_scorer_gain_for_added_bytes,
        "scorer_gain_shortfall_to_break_even": scorer_gain_shortfall_to_break_even,
        "break_even_added_bytes_from_scorer_gain": break_even_added_bytes,
        "byte_budget_margin_vs_break_even": byte_budget_margin,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        **terms,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive.get("sha256"),
        "raw_sha256": _get_path(advisory, ("raw", "sha256")) or _get_path(advisory, ("cache_key", "raw_sha256")),
        "changed_pixel_count": _as_int(summary.get("changed_pixel_count") or cand_summary.get("changed_pixel_count") or apply_result.get("changed_pixel_count")),
        "changed_byte_count": _as_int(summary.get("changed_byte_count") or cand_summary.get("changed_byte_count") or apply_result.get("changed_byte_count")),
        "changed_frame_count": _as_int(summary.get("changed_frame_count") or cand_summary.get("changed_frame_count") or apply_result.get("changed_frame_count")),
        "packed_bytes": _as_int(summary.get("packed_bytes") or cand_summary.get("packed_bytes") or plan.get("packed_bytes")),
        "selected_gain_sum": _as_float(plan.get("selected_gain_sum")),
        "n_kept": _as_int(summary.get("n_kept") or cand_summary.get("n_kept") or plan.get("n_kept")),
        "component": summary.get("component") or cand_summary.get("component"),
        "pair_indices": summary.get("pair_indices") or cand_summary.get("pair_indices"),
        "target_raw_sha256": _get_path(candidate, ("inputs", "target_raw_sha256")),
        **local,
    }


def build_response_dataset(
    paths: list[Path],
    *,
    baseline: ResponseBaseline | None = None,
    include_distilled_vs_direct_rows: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in paths:
        try:
            payload = _load_json(path)
            items = _candidate_items(path, payload)
        except (OSError, json.JSONDecodeError, ScorerResponseDatasetError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            continue
        for candidate_id, candidate, parent in items:
            family = _family_for(path, parent, candidate)
            if (
                family == "distilled_vs_direct_scorer_paired_smoke"
                and not include_distilled_vs_direct_rows
            ):
                skipped.append(
                    {
                        "path": str(path),
                        "reason": (
                            f"{candidate_id}: distilled_vs_direct_scorer_paired_smoke "
                            "requires include_distilled_vs_direct_rows"
                        ),
                    }
                )
                continue
            try:
                row = normalize_response_row(
                    path=path,
                    candidate_id=candidate_id,
                    candidate=candidate,
                    parent=parent,
                    baseline=baseline,
                )
            except ScorerResponseDatasetError as exc:
                skipped.append({"path": str(path), "reason": f"{candidate_id}: {exc}"})
                continue
            if row is None:
                skipped.append({"path": str(path), "reason": f"{candidate_id}: no usable advisory row"})
            else:
                rows.append(row)

    rows.sort(key=lambda row: (row["family"], row["delta_vs_baseline_score"] if row["delta_vs_baseline_score"] is not None else 1e9, row["row_id"]))
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "promotable": False,
            "evidence_grade": "macOS-CPU advisory response dataset",
            "notes": "Rows are report-rounded advisory observations for surrogate fitting and ranking only.",
        },
        "baseline": None if baseline is None else baseline.as_dict(),
        "summary": summarize_rows(rows),
        "feature_correlations": feature_correlations(rows),
        "rows": rows,
        "skipped": skipped,
    }


def build_null_byte_priority_weights(
    null_byte_matrix: dict[str, Any],
    *,
    max_candidates: int = 8,
    seed_budget_k: int = 16,
    allow_legacy_missing_authority: bool = False,
) -> dict[str, Any]:
    """Turn a null-byte matrix into fail-closed LL sampling priorities.

    The matrix is a routing prior for scorer-response harvest, not score
    evidence. All emitted rows remain non-promotional by construction.
    """

    if not isinstance(null_byte_matrix, dict):
        raise ScorerResponseDatasetError("null-byte matrix must be a JSON object")
    if max_candidates <= 0:
        raise ScorerResponseDatasetError("max_candidates must be positive")
    if seed_budget_k <= 0:
        raise ScorerResponseDatasetError("seed_budget_k must be positive")
    if null_byte_matrix.get("schema") != "null_byte_master_gradient_probe_matrix_v1":
        raise ScorerResponseDatasetError("null-byte matrix schema mismatch")
    legacy_missing_authority_fields = _require_explicit_false_authority(
        null_byte_matrix,
        label="null-byte matrix",
        allow_legacy_missing_authority=allow_legacy_missing_authority,
    )
    if null_byte_matrix.get("axis_tag") != "[predicted]":
        raise ScorerResponseDatasetError("null-byte matrix axis_tag must be [predicted]")

    candidates = null_byte_matrix.get("top5_replacement_candidates")
    if not isinstance(candidates, list) or not candidates:
        candidates = [
            row
            for row in null_byte_matrix.get("per_anchor", [])
            if isinstance(row, dict) and row.get("status") == "OK"
        ]
    if not isinstance(candidates, list):
        raise ScorerResponseDatasetError("null-byte matrix has no candidate rows")

    priority_rows: list[dict[str, Any]] = []
    seed_key = f"K={seed_budget_k}"
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        n_null = _as_int(candidate.get("n_null_bytes"))
        if n_null is None or n_null <= 0:
            continue
        if n_null <= seed_budget_k:
            continue
        predicted = candidate.get("predicted_delta_s_per_seed_budget")
        if not isinstance(predicted, dict) or seed_key not in predicted:
            raise ScorerResponseDatasetError(
                f"null-byte matrix candidate missing predicted_delta_s_per_seed_budget.{seed_key}"
            )
        predicted_delta_s = _as_float(predicted.get(seed_key))
        if predicted_delta_s is None:
            raise ScorerResponseDatasetError(
                f"null-byte matrix candidate has invalid {seed_key} predicted delta"
            )
        priority_weight = max(0.0, -predicted_delta_s)
        if priority_weight <= 0.0:
            continue
        priority_rows.append(
            {
                "substrate_label": str(candidate.get("substrate_label") or "unknown_substrate"),
                "scored_archive_sha256": candidate.get("scored_archive_sha256"),
                "codec_family": candidate.get("codec_family"),
                "axis": candidate.get("axis"),
                "anchor_index": candidate.get("anchor_index"),
                "n_null_bytes": n_null,
                "null_fraction": _as_float(candidate.get("null_fraction")),
                "seed_budget_k": seed_budget_k,
                "predicted_delta_s": predicted_delta_s,
                "priority_weight": priority_weight,
                "priority_weight_units": "absolute_predicted_score_delta",
                "priority_weight_source": (
                    f"null_byte_matrix.predicted_delta_s_per_seed_budget.{seed_key}"
                ),
                "predicted_delta_s_per_seed_budget": predicted,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
            }
        )

    priority_rows.sort(
        key=lambda row: (
            -float(row["priority_weight"]),
            -int(row["n_null_bytes"]),
            str(row["substrate_label"]),
            str(row.get("scored_archive_sha256")),
        )
    )
    priority_rows = priority_rows[:max_candidates]
    total_weight = sum(float(row["priority_weight"]) for row in priority_rows)
    for rank, row in enumerate(priority_rows, start=1):
        row["rank"] = rank
        row["ll_sampling_weight"] = (
            float(row["priority_weight"]) / total_weight if total_weight > 0.0 else 0.0
        )

    return {
        "schema": NULL_BYTE_PRIORITY_SCHEMA,
        "source_schema": null_byte_matrix.get("schema"),
        "producer": TOOL,
        "weighting_method": "absolute_predicted_score_delta_for_selected_seed_budget",
        "seed_budget_k": seed_budget_k,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "legacy_missing_authority_fields_accepted": legacy_missing_authority_fields,
        "summary": {
            "candidate_count": len(priority_rows),
            "source_n_anchors_probed_ok": null_byte_matrix.get("n_anchors_probed_ok"),
            "max_candidates": max_candidates,
            "legacy_missing_authority_field_count": len(
                legacy_missing_authority_fields
            ),
        },
        "priority_rows": priority_rows,
    }


def build_magic_codec_seed_boundary(
    boundary_smoke: dict[str, Any],
) -> dict[str, Any]:
    """Normalize pair #4 procedural-seed boundary evidence for LL planning."""

    if not isinstance(boundary_smoke, dict):
        raise ScorerResponseDatasetError(
            "magic-codec seed boundary smoke must be a JSON object"
        )
    if boundary_smoke.get("smoke_pair_id") != (
        "pair_4_magic_codec_x_procedural_codebook_seed_bytes"
    ):
        raise ScorerResponseDatasetError(
            "magic-codec seed boundary smoke has wrong smoke_pair_id"
        )
    _require_explicit_false_authority(
        boundary_smoke,
        label="magic-codec seed boundary smoke",
        fields=(
            "score_claim",
            "score_claim_valid",
            "promotion_eligible",
            "ready_for_exact_eval_dispatch",
            "rank_or_kill_eligible",
            "promotable",
        ),
    )
    verdict = str(boundary_smoke.get("cascade_verdict") or "")
    canonical_rows = _as_int(boundary_smoke.get("n_canonical_reversible_ordering_rows"))
    raw_wins = _as_int(
        boundary_smoke.get("n_canonical_reversible_ordering_rows_raw_seed_dominates")
    )
    min_nonraw_delta = _as_int(
        boundary_smoke.get("min_canonical_reversible_best_nonraw_delta_vs_raw_bytes")
    )
    if canonical_rows is None or canonical_rows <= 0:
        raise ScorerResponseDatasetError(
            "magic-codec seed boundary smoke has no canonical ordering rows"
        )
    if raw_wins is None:
        raise ScorerResponseDatasetError(
            "magic-codec seed boundary smoke missing raw-dominates count"
        )
    boundary_validated = (
        verdict == "PAIR_4_BOUNDARY_VALIDATED_RAW_SEED_DOMINATES"
        and raw_wins == canonical_rows
    )
    return {
        "schema": "ll_magic_codec_seed_boundary.v1",
        "producer": TOOL,
        "source_smoke_label": boundary_smoke.get("smoke_label"),
        "source_smoke_pair_id": boundary_smoke.get("smoke_pair_id"),
        "source_cascade_verdict": verdict,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "boundary_validated_raw_seed_dominates": boundary_validated,
        "n_canonical_reversible_ordering_rows": canonical_rows,
        "n_canonical_reversible_ordering_rows_raw_seed_dominates": raw_wins,
        "min_canonical_reversible_best_nonraw_delta_vs_raw_bytes": min_nonraw_delta,
        "ordering_dimension": boundary_smoke.get("ordering_dimension"),
        "codec_dimensions": boundary_smoke.get("codec_dimensions"),
    }


def build_next_probe_plan(
    dataset: dict[str, Any],
    *,
    null_byte_matrix: dict[str, Any] | None = None,
    null_byte_seed_budget_k: int = 16,
    allow_legacy_null_byte_matrix_missing_authority: bool = False,
    magic_codec_seed_boundary_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic LL next-probe plan from response economics."""

    rows = dataset.get("rows")
    if not isinstance(rows, list):
        raise ScorerResponseDatasetError("dataset rows[] missing")
    best_total = None
    best_scorer = None
    best_margin = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        total_delta = _as_float(row.get("delta_vs_baseline_score"))
        scorer_delta = _as_float(row.get("scorer_delta_vs_baseline"))
        margin = _as_float(row.get("byte_budget_margin_vs_break_even"))
        if total_delta is not None and (best_total is None or total_delta < best_total["delta_vs_baseline_score"]):
            best_total = {
                "row_id": row.get("row_id"),
                "family": row.get("family"),
                "delta_vs_baseline_score": total_delta,
                "added_archive_bytes": row.get("added_archive_bytes"),
            }
        if scorer_delta is not None and (best_scorer is None or scorer_delta < best_scorer["scorer_delta_vs_baseline"]):
            best_scorer = {
                "row_id": row.get("row_id"),
                "family": row.get("family"),
                "scorer_delta_vs_baseline": scorer_delta,
                "observed_scorer_gain_vs_baseline": row.get("observed_scorer_gain_vs_baseline"),
                "break_even_added_bytes_from_scorer_gain": row.get("break_even_added_bytes_from_scorer_gain"),
                "added_archive_bytes": row.get("added_archive_bytes"),
            }
        if margin is not None and (best_margin is None or margin > best_margin["byte_budget_margin_vs_break_even"]):
            best_margin = {
                "row_id": row.get("row_id"),
                "family": row.get("family"),
                "byte_budget_margin_vs_break_even": margin,
                "break_even_added_bytes_from_scorer_gain": row.get("break_even_added_bytes_from_scorer_gain"),
                "added_archive_bytes": row.get("added_archive_bytes"),
            }

    prohibitions: list[dict[str, Any]] = []
    if best_margin is None or float(best_margin["byte_budget_margin_vs_break_even"]) < 0.0:
        prohibitions.append(
            {
                "rule": "do_not_widen_coordinate_sparse_residual_sidecar",
                "reason": "observed scorer gains cannot pay for current residual payload bytes",
                "best_byte_budget_margin": best_margin,
            }
        )

    magic_codec_seed_boundary = None
    if magic_codec_seed_boundary_smoke is not None:
        magic_codec_seed_boundary = build_magic_codec_seed_boundary(
            magic_codec_seed_boundary_smoke
        )
        if magic_codec_seed_boundary["boundary_validated_raw_seed_dominates"]:
            prohibitions.append(
                {
                    "rule": "do_not_wrap_procedural_seed_bytes_with_magic_codec",
                    "reason": (
                        "pair #4 boundary smoke found raw seed dominates all "
                        "canonical reversible seed/order rows; route magic_codec "
                        "to residual streams, not the procedural seed itself"
                    ),
                    "boundary": magic_codec_seed_boundary,
                }
            )

    probes = [
        {
            "probe_id": "ll_byte_neutral_decoder_q_response_model",
            "priority": 1,
            "class": "byte_neutral_representation_mutation",
            "rationale": (
                "The best total row is byte-neutral but still positive-delta; "
                "learn response around decoder-q observables before adding payload bytes."
            ),
            "input_rows": [
                row.get("row_id")
                for row in rows
                if isinstance(row, dict) and row.get("family") == "decoder_q"
            ][:8],
            "acceptance_gate": "full advisory delta_vs_baseline_score < 0 with added_archive_bytes <= 0",
        },
        {
            "probe_id": "ll_amortized_residual_grammar_gate",
            "priority": 2,
            "class": "payload_amortization_or_runtime_transform",
            "rationale": (
                "Sparse residual showed nominal scorer gain but break-even byte allowance "
                "is sub-byte; any residual lane must first reduce payload overhead or "
                "increase scorer gain by orders of magnitude."
            ),
            "input_rows": [] if best_scorer is None else [best_scorer["row_id"]],
            "acceptance_gate": "byte_budget_margin_vs_break_even >= 0 before widening",
        },
        {
            "probe_id": "ll_response_dataset_expansion",
            "priority": 3,
            "class": "surrogate_training_data",
            "rationale": "Current response table is too small for a learned surrogate; add diverse byte-neutral and amortized probes.",
            "input_rows": [],
            "acceptance_gate": ">=50 rows with at least two families containing held-out folds 0..4",
        },
    ]
    null_byte_priority_weights = None
    if null_byte_matrix is not None:
        null_byte_priority_weights = build_null_byte_priority_weights(
            null_byte_matrix,
            seed_budget_k=null_byte_seed_budget_k,
            allow_legacy_missing_authority=allow_legacy_null_byte_matrix_missing_authority,
        )
        for probe in probes:
            probe["priority"] = int(probe["priority"]) + 1
        probes.insert(
            0,
            {
                "probe_id": "ll_null_byte_procedural_codebook_candidates",
                "priority": 1,
                "class": "surrogate_training_data",
                "rationale": (
                    "Use the null-byte master-gradient matrix as the next LL "
                    "training-data harvest prior; highest null-byte budgets get "
                    "sampled first while remaining fail-closed routing signals. "
                    "If pair #4 seed-boundary evidence is present, keep seeds raw "
                    "and learn only where residual/runtime streams could exist."
                ),
                "input_rows": [],
                "null_byte_priority_rows": null_byte_priority_weights["priority_rows"],
                "acceptance_gate": (
                    "typed CandidateModificationSpec + byte-consumption/no-op proof "
                    "+ exact contest eval before any score claim"
                ),
            },
        )

    return {
        "schema": "ll_scorer_response_next_probe_plan.v1",
        "producer": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "dataset_summary": dataset.get("summary"),
        "best_total_row": best_total,
        "best_scorer_row": best_scorer,
        "best_byte_budget_margin_row": best_margin,
        "null_byte_priority_weights": null_byte_priority_weights,
        "magic_codec_seed_boundary": magic_codec_seed_boundary,
        "prohibitions": prohibitions,
        "probes": probes,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, int] = {}
    improved = 0
    scorer_improved = 0
    best = None
    worst = None
    best_scorer = None
    best_margin = None
    for row in rows:
        by_family[row["family"]] = by_family.get(row["family"], 0) + 1
        delta = row.get("delta_vs_baseline_score")
        scorer_delta = row.get("scorer_delta_vs_baseline")
        if isinstance(delta, (int, float)) and delta < 0:
            improved += 1
        if isinstance(scorer_delta, (int, float)) and scorer_delta < 0:
            scorer_improved += 1
        if isinstance(delta, (int, float)):
            if best is None or delta < best["delta_vs_baseline_score"]:
                best = {"row_id": row["row_id"], "family": row["family"], "delta_vs_baseline_score": delta}
            if worst is None or delta > worst["delta_vs_baseline_score"]:
                worst = {"row_id": row["row_id"], "family": row["family"], "delta_vs_baseline_score": delta}
        if isinstance(scorer_delta, (int, float)):
            if best_scorer is None or scorer_delta < best_scorer["scorer_delta_vs_baseline"]:
                best_scorer = {
                    "row_id": row["row_id"],
                    "family": row["family"],
                    "scorer_delta_vs_baseline": scorer_delta,
                    "break_even_added_bytes_from_scorer_gain": row.get("break_even_added_bytes_from_scorer_gain"),
                }
        margin = row.get("byte_budget_margin_vs_break_even")
        if isinstance(margin, (int, float)):
            if best_margin is None or margin > best_margin["byte_budget_margin_vs_break_even"]:
                best_margin = {
                    "row_id": row["row_id"],
                    "family": row["family"],
                    "byte_budget_margin_vs_break_even": margin,
                }
    return {
        "row_count": len(rows),
        "family_counts": by_family,
        "improved_total_score_count": improved,
        "improved_scorer_term_count": scorer_improved,
        "best_delta": best,
        "worst_delta": worst,
        "best_scorer_delta": best_scorer,
        "best_byte_budget_margin": best_margin,
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / math.sqrt(vx * vy)


def feature_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = (
        "archive_bytes",
        "added_archive_bytes",
        "changed_pixel_count",
        "changed_byte_count",
        "changed_frame_count",
        "packed_bytes",
        "selected_gain_sum",
        "n_kept",
        "local_pose_delta_sum",
        "local_seg_delta_sum",
    )
    out: list[dict[str, Any]] = []
    for target in ("delta_vs_baseline_score", "scorer_delta_vs_baseline"):
        for feature in features:
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                x = _as_float(row.get(feature))
                y = _as_float(row.get(target))
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            corr = _pearson(xs, ys)
            if corr is not None:
                out.append(
                    {
                        "target": target,
                        "feature": feature,
                        "n": len(xs),
                        "pearson_r": corr,
                    }
                )
    out.sort(key=lambda row: abs(float(row["pearson_r"])), reverse=True)
    return out


def render_markdown(dataset: dict[str, Any]) -> str:
    summary = dataset["summary"]
    lines = [
        "# Scorer Response Dataset",
        "",
        f"- Rows: {summary['row_count']}",
        f"- Total-score improvements: {summary['improved_total_score_count']}",
        f"- Scorer-term improvements: {summary['improved_scorer_term_count']}",
        f"- Score claim: {dataset['score_claim']}",
        "",
        "## Families",
        "",
    ]
    for family, count in sorted(summary["family_counts"].items()):
        lines.append(f"- `{family}`: {count}")
    lines.extend(["", "## Best/Worst", ""])
    for label in ("best_delta", "worst_delta"):
        value = summary.get(label)
        lines.append(f"- `{label}`: `{value}`")
    lines.extend(["", "## Correlations", ""])
    for row in dataset["feature_correlations"][:12]:
        lines.append(
            f"- `{row['target']}` vs `{row['feature']}`: r={row['pearson_r']:.6g}, n={row['n']}"
        )
    lines.extend(["", "## Rows", ""])
    for row in dataset["rows"]:
        lines.append(
            "- "
            f"`{row['family']}` `{row['candidate_id']}` "
            f"fold={row['holdout_fold']} "
            f"delta={row['delta_vs_baseline_score']} "
            f"scorer_delta={row['scorer_delta_vs_baseline']} "
            f"byte_margin={row['byte_budget_margin_vs_break_even']} "
            f"archive_bytes={row['archive_bytes']}"
        )
    lines.append("")
    return "\n".join(lines)


def render_next_probe_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# LL Scorer Response Next-Probe Plan",
        "",
        f"- Score claim: {plan['score_claim']}",
        f"- Best total row: `{plan.get('best_total_row')}`",
        f"- Best scorer row: `{plan.get('best_scorer_row')}`",
        f"- Best byte-budget margin row: `{plan.get('best_byte_budget_margin_row')}`",
        "",
    ]
    null_priority = plan.get("null_byte_priority_weights")
    if isinstance(null_priority, dict):
        lines.extend(["## Null-Byte Matrix Priority", ""])
        legacy_missing = null_priority.get("legacy_missing_authority_fields_accepted")
        if legacy_missing:
            lines.append(
                "- Legacy missing authority fields accepted: "
                f"`{legacy_missing}`"
            )
        for row in null_priority.get("priority_rows", [])[:8]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- "
                f"P{row.get('rank')} `{row.get('substrate_label')}` "
                f"null_bytes={row.get('n_null_bytes')} "
                f"weight={float(row.get('ll_sampling_weight', 0.0)):.6g}"
            )
        lines.append("")
    lines.extend(["## Prohibitions", ""])
    for item in plan.get("prohibitions", []):
        lines.append(f"- `{item['rule']}`: {item['reason']}")
    lines.extend(["", "## Probes", ""])
    for probe in plan.get("probes", []):
        lines.append(
            f"- P{probe['priority']} `{probe['probe_id']}` "
            f"({probe['class']}): {probe['acceptance_gate']}"
        )
    lines.append("")
    return "\n".join(lines)
