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
import importlib
import importlib.util
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.optimization.candidate_evidence_contract import CONTEST_UNCOMPRESSED_BYTES

SCHEMA = "scorer_response_dataset.v1"
ROW_SCHEMA = "scorer_response_row.v1"
NULL_BYTE_PRIORITY_SCHEMA = "ll_null_byte_priority_weights.v1"
CONSUMER_ROUTING_SCHEMA = "scorer_response_dataset_consumer_routing.v1"
VALIDATION_GATE_SCHEMA = "scorer_response_dataset_validation_gate.v1"
MLX_SCORER_RESPONSE_SCHEMA = "mlx_scorer_response.v1"
MLX_TORCH_PARITY_SWEEP_SCHEMA = "mlx_scorer_torch_parity_sweep.v1"

DEFAULT_RESPONSE_PREDICTION_FIELDS = (
    "predicted_delta_vs_baseline_score",
    "predicted_scorer_delta_vs_baseline",
    "ll_predicted_delta_vs_baseline_score",
    "ll_predicted_scorer_delta_vs_baseline",
    "expected_delta_vs_baseline_score",
    "expected_scorer_delta_vs_baseline",
)


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
    if payload.get("schema_version") == MLX_SCORER_RESPONSE_SCHEMA:
        return [_mlx_scorer_response_candidate_item(path, payload)]
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


def _false_authority_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "score_claim": payload.get("score_claim"),
        "score_claim_valid": payload.get("score_claim_valid"),
        "promotion_eligible": payload.get("promotion_eligible"),
        "ready_for_exact_eval_dispatch": payload.get("ready_for_exact_eval_dispatch"),
        "rank_or_kill_eligible": payload.get("rank_or_kill_eligible"),
        "promotable": payload.get("promotable", False),
        "evidence_grade": payload.get("evidence_grade"),
    }


def _require_mlx_response_false_authority(
    payload: dict[str, Any],
    *,
    label: str,
    allow_candidate_array_identity: bool = False,
) -> None:
    for field in (
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
    ):
        if payload.get(field) is not False:
            raise ScorerResponseDatasetError(f"{label} {field} must be explicit false")
    if payload.get("promotable", False) is not False:
        raise ScorerResponseDatasetError(f"{label} promotable must be false")
    if payload.get("candidate_generation_only") is not True:
        raise ScorerResponseDatasetError(f"{label} candidate_generation_only must be true")
    if payload.get("requires_exact_eval_before_promotion") is not True:
        raise ScorerResponseDatasetError(
            f"{label} requires_exact_eval_before_promotion must be true"
        )
    if payload.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise ScorerResponseDatasetError(
            f"{label} evidence_grade must be {EVIDENCE_GRADE_MLX}"
        )
    if payload.get("evidence_tag") != EVIDENCE_TAG_MLX:
        raise ScorerResponseDatasetError(f"{label} evidence_tag must be {EVIDENCE_TAG_MLX}")
    if payload.get("score_axis") != EVIDENCE_TAG_MLX:
        raise ScorerResponseDatasetError(f"{label} score_axis must be {EVIDENCE_TAG_MLX}")
    if payload.get("canonical_score_source") != "score_recomputed_from_components":
        raise ScorerResponseDatasetError(
            f"{label} canonical_score_source must be score_recomputed_from_components"
        )
    _require_mlx_finite(payload, "canonical_score", label=label)
    _require_mlx_finite(payload, "score_recomputed_from_components", label=label)
    _require_mlx_finite(payload, "avg_posenet_dist", label=label)
    _require_mlx_finite(payload, "avg_segnet_dist", label=label)
    if not math.isclose(
        float(payload["canonical_score"]),
        float(payload["score_recomputed_from_components"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ScorerResponseDatasetError(
            f"{label} canonical_score must match score_recomputed_from_components"
        )
    device_contract = payload.get("device_contract")
    if not isinstance(device_contract, dict):
        raise ScorerResponseDatasetError(f"{label} device_contract must be an object")
    forbidden = device_contract.get("forbidden_uses")
    if not isinstance(forbidden, list) or "score_claim" not in forbidden:
        raise ScorerResponseDatasetError(
            f"{label} device_contract.forbidden_uses must include score_claim"
        )
    cache_identity = payload.get("cache_identity")
    if not isinstance(cache_identity, dict):
        raise ScorerResponseDatasetError(f"{label} cache_identity must be an object")
    if cache_identity.get("pair_indices_equal") is not True:
        raise ScorerResponseDatasetError(f"{label} cache_identity.pair_indices_equal must be true")
    for side in ("reference", "candidate"):
        item = cache_identity.get(side)
        if not isinstance(item, dict):
            raise ScorerResponseDatasetError(f"{label} cache_identity.{side} must be an object")
        archive_hashes_valid = all(
            _is_sha256(str(item.get(key, "")))
            for key in ("archive_sha256", "inflated_outputs_aggregate_sha256", "raw_sha256")
        )
        array_hashes = item.get("array_sha256")
        array_hashes_valid = isinstance(array_hashes, dict) and all(
            _is_sha256(str(array_hashes.get(key, "")))
            for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb")
        )
        if side == "candidate" and not archive_hashes_valid and not (
            allow_candidate_array_identity and array_hashes_valid
        ):
            raise ScorerResponseDatasetError(
                f"{label} cache_identity.{side} archive/raw hashes must be sha256"
            )
        if side == "reference" and not archive_hashes_valid and not array_hashes_valid:
            raise ScorerResponseDatasetError(
                f"{label} cache_identity.{side} must include archive/raw hashes or array_sha256"
            )
        for key in ("archive_sha256", "inflated_outputs_aggregate_sha256", "raw_sha256"):
            if item.get(key) is None:
                continue
            if not _is_sha256(str(item.get(key, ""))):
                raise ScorerResponseDatasetError(
                    f"{label} cache_identity.{side}.{key} must be sha256"
                )
        if isinstance(array_hashes, dict):
            for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"):
                if not _is_sha256(str(array_hashes.get(key, ""))):
                    raise ScorerResponseDatasetError(
                        f"{label} cache_identity.{side}.array_sha256.{key} must be sha256"
                    )


def _require_mlx_finite(payload: dict[str, Any], key: str, *, label: str) -> None:
    if _as_float(payload.get(key)) is None:
        raise ScorerResponseDatasetError(f"{label} {key} must be finite")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def _normalize_family_label(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ScorerResponseDatasetError(f"{label} must be non-empty when provided")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-.")
    if any(ch not in allowed for ch in text):
        raise ScorerResponseDatasetError(
            f"{label} may contain only lowercase letters, digits, underscore, dash, or dot"
        )
    return text


def _explicit_response_family(parent: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    for source_label, source in (("candidate", candidate), ("parent", parent)):
        family = _normalize_family_label(
            source.get("response_family"),
            label=f"{source_label}.response_family",
        )
        if family is not None:
            return family
    return None


def _mlx_scorer_response_candidate_item(
    path: Path,
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    label = f"{path}:mlx_scorer_response"
    _require_mlx_response_false_authority(payload, label=label)
    response_family = _normalize_family_label(
        payload.get("response_family"),
        label=f"{label}.response_family",
    )
    candidate_id = str(
        payload.get("run_id")
        or f"mlx_pairs_{payload.get('start_pair')}_{payload.get('max_pairs')}_{payload.get('batch_pairs')}"
        or path.stem
    )
    authority = _false_authority_from_payload(payload)
    archive = {
        "sha256": payload.get("archive_sha256"),
        "bytes": payload.get("archive_size_bytes"),
    }
    raw_sha = payload.get("raw_sha256")
    advisory = {
        "canonical_score": payload.get("canonical_score"),
        "archive_size_bytes": payload.get("archive_size_bytes"),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "axis": payload.get("score_axis") or payload.get("evidence_tag"),
        "archive": archive,
        "raw": {"sha256": raw_sha},
        "cache_key": {"raw_sha256": raw_sha},
        "blockers": [],
    }
    candidate = {
        "candidate_id": candidate_id,
        "advisory_eval": advisory,
        "authority": authority,
        "response_family": response_family,
        "summary": {
            "component": "mlx_scorer_response",
            "pair_indices": payload.get("pair_window"),
        },
    }
    parent = {
        "schema": MLX_SCORER_RESPONSE_SCHEMA,
        "schema_version": payload.get("schema_version"),
        "producer": "tac.local_acceleration.mlx_scorer_response",
        "authority": authority,
        "evidence_grade": payload.get("evidence_grade"),
        "evidence_tag": payload.get("evidence_tag"),
        "hardware_substrate": payload.get("hardware_substrate"),
        "batch_pairs": payload.get("batch_pairs"),
        "n_samples": payload.get("n_samples"),
        "pair_window": payload.get("pair_window"),
        "start_pair": payload.get("start_pair"),
        "max_pairs": payload.get("max_pairs"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "components": payload.get("components"),
        "cache_identity": payload.get("cache_identity"),
        "device_contract": payload.get("device_contract"),
        "response_family": response_family,
    }
    return candidate_id, candidate, parent


def _mlx_response_window_key(payload: dict[str, Any], *, label: str) -> str:
    if payload.get("schema_version") != MLX_SCORER_RESPONSE_SCHEMA:
        raise ScorerResponseDatasetError(
            f"{label} schema_version must be {MLX_SCORER_RESPONSE_SCHEMA}"
        )
    pair_window = payload.get("pair_window")
    if not isinstance(pair_window, list) or len(pair_window) != 2:
        raise ScorerResponseDatasetError(f"{label} pair_window must be length-2 list")
    start = _as_int(payload.get("start_pair"))
    max_pairs = _as_int(payload.get("max_pairs"))
    batch_pairs = _as_int(payload.get("batch_pairs"))
    if start is None or max_pairs is None or batch_pairs is None:
        raise ScorerResponseDatasetError(
            f"{label} start_pair/max_pairs/batch_pairs must be integers"
        )
    return f"start={start}:max={max_pairs}:window={pair_window[0]}-{pair_window[1]}"


def _family_for(path: Path, parent: dict[str, Any], candidate: dict[str, Any]) -> str:
    explicit_family = _explicit_response_family(parent, candidate)
    if explicit_family is not None:
        return explicit_family
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
    if "mlx_scorer_response" in source:
        return "mlx_scorer_response"
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
        "source_schema": parent.get("schema") or parent.get("schema_version"),
        "source_evidence_grade": parent.get("evidence_grade") or authority.get("evidence_grade"),
        "source_evidence_tag": parent.get("evidence_tag"),
        "source_hardware_substrate": parent.get("hardware_substrate"),
        "source_batch_pairs": _as_int(parent.get("batch_pairs") or candidate.get("batch_pairs")),
        "source_n_samples": _as_int(parent.get("n_samples") or candidate.get("n_samples")),
        "source_pair_window": parent.get("pair_window") or candidate.get("pair_window"),
        "source_start_pair": _as_int(parent.get("start_pair") or candidate.get("start_pair")),
        "source_max_pairs": _as_int(parent.get("max_pairs") or candidate.get("max_pairs")),
        "source_elapsed_seconds": _as_float(parent.get("elapsed_seconds") or candidate.get("elapsed_seconds")),
        "source_posenet_sha256": _get_path(parent, ("components", "posenet_sha256")),
        "source_segnet_sha256": _get_path(parent, ("components", "segnet_sha256")),
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


def build_windowed_mlx_response_dataset(
    *,
    candidate_paths: list[Path],
    baseline_paths: list[Path],
) -> dict[str, Any]:
    """Build response rows with one MLX baseline per scorer-pair window.

    Direct MLX scorer-response rows are usually harvested on slices of the
    video. A single global baseline would make those deltas meaningless, so this
    helper matches each candidate response to a baseline response with the same
    scorer-pair window and then emits the normal non-promotional dataset shape.
    """

    baseline_by_window: dict[str, ResponseBaseline] = {}
    baseline_sources: dict[str, str] = {}
    duplicate_baseline_windows: set[str] = set()
    skipped: list[dict[str, str]] = []
    for path in baseline_paths:
        try:
            payload = _load_json(path)
            label = f"baseline {path}"
            _require_mlx_response_false_authority(
                payload,
                label=label,
                allow_candidate_array_identity=True,
            )
            key = _mlx_response_window_key(payload, label=label)
            if key in baseline_by_window or key in duplicate_baseline_windows:
                duplicate_baseline_windows.add(key)
                baseline_by_window.pop(key, None)
                baseline_sources.pop(key, None)
                skipped.append(
                    {
                        "path": str(path),
                        "reason": f"baseline: duplicate baseline window {key}",
                    }
                )
                continue
            baseline_by_window[key] = ResponseBaseline(
                score=float(payload["canonical_score"]),
                archive_bytes=int(payload["archive_size_bytes"]),
                avg_posenet_dist=_as_float(payload.get("avg_posenet_dist")),
                avg_segnet_dist=_as_float(payload.get("avg_segnet_dist")),
            )
            baseline_sources[key] = str(path)
        except (OSError, json.JSONDecodeError, ScorerResponseDatasetError, KeyError, ValueError) as exc:
            skipped.append({"path": str(path), "reason": f"baseline: {exc}"})

    rows: list[dict[str, Any]] = []
    for path in candidate_paths:
        try:
            payload = _load_json(path)
            key = _mlx_response_window_key(payload, label=f"candidate {path}")
            if key in duplicate_baseline_windows:
                skipped.append(
                    {
                        "path": str(path),
                        "reason": f"candidate: ambiguous duplicate baseline window {key}",
                    }
                )
                continue
            baseline = baseline_by_window.get(key)
            if baseline is None:
                skipped.append(
                    {
                        "path": str(path),
                        "reason": f"candidate: no matching baseline window {key}",
                    }
                )
                continue
            partial = build_response_dataset([path], baseline=baseline)
        except (OSError, json.JSONDecodeError, ScorerResponseDatasetError) as exc:
            skipped.append({"path": str(path), "reason": f"candidate: {exc}"})
            continue
        for item in partial.get("skipped", []):
            if isinstance(item, dict):
                skipped.append({"path": str(item.get("path", path)), "reason": str(item.get("reason"))})
        for row in partial.get("rows", []):
            if not isinstance(row, dict):
                continue
            baseline_terms = _score_terms(
                archive_bytes=baseline.archive_bytes,
                pose=baseline.avg_posenet_dist,
                seg=baseline.avg_segnet_dist,
            )
            row["window_baseline_source_path"] = baseline_sources[key]
            row["window_baseline_key"] = key
            row["window_baseline_score"] = baseline.score
            row["window_baseline_archive_bytes"] = baseline.archive_bytes
            row["window_baseline_avg_posenet_dist"] = baseline.avg_posenet_dist
            row["window_baseline_avg_segnet_dist"] = baseline.avg_segnet_dist
            row["window_baseline_pose_term"] = baseline_terms["pose_term"]
            row["window_baseline_seg_term"] = baseline_terms["seg_term"]
            row["window_baseline_scorer_term"] = baseline_terms["scorer_term"]
            row["window_baseline_rate_term"] = baseline_terms["rate_term"]
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["family"],
            row["delta_vs_baseline_score"]
            if row["delta_vs_baseline_score"] is not None
            else 1e9,
            row["row_id"],
        )
    )
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
            "evidence_grade": EVIDENCE_GRADE_MLX,
            "notes": (
                "Rows are local MLX scorer-response observations paired with "
                "same-window MLX baselines for surrogate fitting and ranking only."
            ),
        },
        "baseline": {
            "mode": "per_window_mlx_response",
            "window_count": len(baseline_by_window),
            "window_sources": baseline_sources,
        },
        "summary": summarize_rows(rows),
        "feature_correlations": feature_correlations(rows),
        "rows": rows,
        "skipped": skipped,
    }


def merge_scorer_response_datasets(
    datasets: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Merge normalized scorer-response datasets without creating authority."""

    if not datasets:
        raise ScorerResponseDatasetError("at least one dataset is required")
    rows: list[dict[str, Any]] = []
    seen_row_ids: set[str] = set()
    sources: list[dict[str, Any]] = []
    for label, dataset in datasets:
        normalized = normalize_legacy_response_dataset_authority(
            dataset,
            source_label=label,
        )
        source_rows = normalized["rows"]
        sources.append(
            {
                "label": label,
                "row_count": len(source_rows),
                "summary": normalized.get("summary"),
                "authority_normalization": normalized.get("authority_normalization"),
            }
        )
        for row in source_rows:
            row_id = str(row.get("row_id"))
            if row_id in seen_row_ids:
                raise ScorerResponseDatasetError(f"duplicate row_id across datasets: {row_id}")
            seen_row_ids.add(row_id)
            merged = copy.deepcopy(row)
            merged["source_dataset"] = label
            rows.append(merged)

    rows.sort(
        key=lambda row: (
            str(row.get("family")),
            row.get("delta_vs_baseline_score")
            if isinstance(row.get("delta_vs_baseline_score"), (int, float))
            else 1e9,
            str(row.get("row_id")),
        )
    )
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
            "evidence_grade": "merged non-authoritative scorer-response dataset",
            "notes": (
                "Merged rows preserve source false-authority contracts; this "
                "artifact remains local planning signal only."
            ),
        },
        "merge_sources": sources,
        "summary": summarize_rows(rows),
        "feature_correlations": feature_correlations(rows),
        "rows": rows,
        "skipped": [],
    }


def build_scorer_response_consumer_routing(
    dataset: dict[str, Any],
    *,
    consumer_modules: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Route scorer-response rows through opt-in cathedral consumers.

    This is an observability-only bridge. It calls only consumers declaring
    ``CONSUMES_SCORER_RESPONSE_DATASET = True`` and records their
    non-promotional verdicts; it does not mutate the dataset, dispatch work, or
    create score authority.
    """

    rows = dataset.get("rows")
    if not isinstance(rows, list):
        raise ScorerResponseDatasetError("dataset rows[] missing")
    modules = (
        list(consumer_modules)
        if consumer_modules is not None
        else _discover_scorer_response_consumer_modules()
    )
    verdicts: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("row_id") or row.get("candidate_id") or row_index)
        for mod in modules:
            if not getattr(mod, "CONSUMES_SCORER_RESPONSE_DATASET", False):
                continue
            verdicts.append(_invoke_scorer_response_consumer(mod, row, row_id=row_id))

    return {
        "schema": CONSUMER_ROUTING_SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "dataset_schema": dataset.get("schema"),
        "dataset_summary": dataset.get("summary"),
        "row_count": len([row for row in rows if isinstance(row, dict)]),
        "consumer_count": len(
            [mod for mod in modules if getattr(mod, "CONSUMES_SCORER_RESPONSE_DATASET", False)]
        ),
        "verdict_count": len(verdicts),
        "verdicts": verdicts,
    }


def _discover_scorer_response_consumer_modules() -> list[Any]:
    try:
        spec = importlib.util.find_spec("tools.cathedral_autopilot_autonomous_loop")
        if spec is None:
            return []
        loop_module = importlib.import_module("tools.cathedral_autopilot_autonomous_loop")
        discover = getattr(loop_module, "discover_compliant_consumer_modules", None)
        if not callable(discover):
            return []
        modules = discover()
    except Exception:
        return []
    return [
        mod
        for mod in modules
        if getattr(mod, "CONSUMES_SCORER_RESPONSE_DATASET", False)
    ]


def _invoke_scorer_response_consumer(
    module: Any,
    row: dict[str, Any],
    *,
    row_id: str,
) -> dict[str, Any]:
    module_name = getattr(module, "__name__", "<unknown>")
    consumer_name = str(getattr(module, "CONSUMER_NAME", module_name))
    try:
        hook = module.consume_candidate
        verdict = hook(dict(row))
    except Exception as exc:
        return {
            "consumer_module": module_name,
            "consumer_name": consumer_name,
            "row_id": row_id,
            "error": f"{type(exc).__name__}: {exc}",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "promotable": False,
        }
    if not isinstance(verdict, dict):
        return {
            "consumer_module": module_name,
            "consumer_name": consumer_name,
            "row_id": row_id,
            "error": f"consume_candidate returned {type(verdict).__name__}, expected dict",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "promotable": False,
        }
    return {
        "consumer_module": module_name,
        "consumer_name": consumer_name,
        "consumer_version": str(getattr(module, "CONSUMER_VERSION", "unknown")),
        "row_id": row_id,
        "consumer_signal_kind": verdict.get("consumer_signal_kind"),
        "predicted_delta_adjustment": _as_float(verdict.get("predicted_delta_adjustment")) or 0.0,
        "axis_tag": str(verdict.get("axis_tag", "[predicted]")),
        "rationale": str(verdict.get("rationale", ""))[:512],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "raw_verdict": verdict,
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


def _summary_distribution_max(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, dict):
        return _as_float(value.get("max"))
    return _as_float(value)


def _parity_sweep_failed_rows(rows: list[Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("passed") is True:
            continue
        failed.append(
            {
                "index": row.get("index"),
                "pair_window": row.get("pair_window"),
                "verdict": row.get("verdict"),
                "blockers": list(row.get("blockers") or []),
                "deltas": row.get("deltas") if isinstance(row.get("deltas"), dict) else {},
            }
        )
    return failed[:8]


def build_mlx_torch_parity_sweep_gate(
    parity_sweep: dict[str, Any],
    *,
    allow_mlx_parity_research_signal_override: bool = False,
) -> dict[str, Any]:
    """Normalize PyTorch-vs-MLX parity evidence for LL planner gating.

    The gate can allow MLX rows into local surrogate planning, but it never gives
    those rows score, rank, or promotion authority.
    """

    if not isinstance(parity_sweep, dict):
        raise ScorerResponseDatasetError("MLX parity sweep must be a JSON object")
    source_schema = parity_sweep.get("schema_version") or parity_sweep.get("schema")
    if source_schema != MLX_TORCH_PARITY_SWEEP_SCHEMA:
        raise ScorerResponseDatasetError("MLX parity sweep schema mismatch")
    _require_explicit_false_authority(
        parity_sweep,
        label="MLX parity sweep",
        fields=(
            "score_claim",
            "score_claim_valid",
            "promotion_eligible",
            "ready_for_exact_eval_dispatch",
            "rank_or_kill_eligible",
        ),
    )
    if parity_sweep.get("candidate_generation_only") is not True:
        raise ScorerResponseDatasetError(
            "MLX parity sweep candidate_generation_only must be true"
        )
    if parity_sweep.get("requires_exact_eval_before_promotion") is not True:
        raise ScorerResponseDatasetError(
            "MLX parity sweep requires_exact_eval_before_promotion must be true"
        )
    if parity_sweep.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise ScorerResponseDatasetError("MLX parity sweep evidence_grade mismatch")
    if parity_sweep.get("evidence_tag") != EVIDENCE_TAG_MLX:
        raise ScorerResponseDatasetError("MLX parity sweep evidence_tag mismatch")
    if parity_sweep.get("score_axis") != EVIDENCE_TAG_MLX:
        raise ScorerResponseDatasetError("MLX parity sweep score_axis mismatch")
    rows = parity_sweep.get("rows")
    if not isinstance(rows, list):
        raise ScorerResponseDatasetError("MLX parity sweep rows must be a list")

    summary = parity_sweep.get("summary") if isinstance(parity_sweep.get("summary"), dict) else {}
    failed_rows = _parity_sweep_failed_rows(rows)
    source_passed = parity_sweep.get("passed") is True
    source_blockers = list(parity_sweep.get("blockers") or [])
    summary_failed_windows = _as_int(summary.get("failed_windows"))
    strict_pass = (
        source_passed
        and not source_blockers
        and not failed_rows
        and summary_failed_windows in (None, 0)
    )
    if strict_pass:
        status = "strict_pass"
        mlx_rows_allowed = True
    elif allow_mlx_parity_research_signal_override:
        status = "research_signal_override"
        mlx_rows_allowed = True
    else:
        status = "blocked"
        mlx_rows_allowed = False

    blockers = list(source_blockers)
    if failed_rows:
        blockers.append("mlx_torch_parity_sweep_has_failed_rows")
    if summary_failed_windows not in (None, 0):
        blockers.append("mlx_torch_parity_sweep_summary_failed_windows_nonzero")
    if not strict_pass:
        blockers.insert(0, "mlx_torch_parity_sweep_not_strict_pass")
    return {
        "schema": "ll_mlx_torch_parity_sweep_gate.v1",
        "producer": TOOL,
        "source_schema": source_schema,
        "source_verdict": parity_sweep.get("verdict"),
        "source_passed": source_passed,
        "status": status,
        "mlx_rows_allowed_for_planner": mlx_rows_allowed,
        "research_signal_override": bool(allow_mlx_parity_research_signal_override),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "device_type": parity_sweep.get("device_type"),
        "window_count": _as_int(parity_sweep.get("window_count")) or len(rows),
        "covered_pair_window": parity_sweep.get("covered_pair_window"),
        "summary": {
            "passed_windows": _as_int(summary.get("passed_windows")),
            "failed_windows": _as_int(summary.get("failed_windows")),
            "segnet_argmax_diff_pixels_max": _summary_distribution_max(
                summary,
                "segnet_argmax_diff_pixels",
            ),
            "segnet_argmax_diff_fraction_max": _summary_distribution_max(
                summary,
                "segnet_argmax_diff_fraction",
            ),
            "segnet_logit_abs_max": _summary_distribution_max(
                summary,
                "segnet_logit_abs_max",
            ),
            "posenet_output_abs_max": _summary_distribution_max(
                summary,
                "posenet_output_abs_max",
            ),
            "posenet_component_abs_max": _summary_distribution_max(
                summary,
                "posenet_component_abs_max",
            ),
            "segnet_argmax_mismatch_pixels_total": _as_int(
                summary.get("segnet_argmax_mismatch_pixels_total")
            ),
        },
        "blockers": blockers,
        "failed_rows": failed_rows,
        "allowed_use": (
            "local_ll_surrogate_planner_input"
            if mlx_rows_allowed
            else "blocked_until_strict_torch_parity_or_explicit_research_override"
        ),
    }


def build_scorer_response_validation_gate(
    dataset: dict[str, Any],
    *,
    min_rows: int = 50,
    min_families: int = 2,
    required_folds: Iterable[int] = range(5),
    target: str = "delta_vs_baseline_score",
    prediction_fields: Iterable[str] = DEFAULT_RESPONSE_PREDICTION_FIELDS,
    min_prediction_pairs_per_fold: int = 3,
    min_pearson_r: float = 0.2,
    require_single_axis: bool = True,
) -> dict[str, Any]:
    """Validate whether a response dataset is ready for LL held-out use.

    This is a fail-closed research-signal gate. A pass means the dataset has
    enough rows, family diversity, fold coverage, and at least one explicit
    prediction field with held-out correlation against the requested target.
    It never creates score authority or exact-eval dispatch eligibility.
    """

    if min_rows <= 0:
        raise ScorerResponseDatasetError("min_rows must be positive")
    if min_families <= 0:
        raise ScorerResponseDatasetError("min_families must be positive")
    if min_prediction_pairs_per_fold <= 1:
        raise ScorerResponseDatasetError("min_prediction_pairs_per_fold must be > 1")
    if not math.isfinite(min_pearson_r):
        raise ScorerResponseDatasetError("min_pearson_r must be finite")
    folds_required = sorted({int(fold) for fold in required_folds})
    if not folds_required:
        raise ScorerResponseDatasetError("required_folds must be non-empty")

    normalized = normalize_legacy_response_dataset_authority(dataset)
    rows = normalized["rows"]
    family_counts: dict[str, int] = {}
    family_folds: dict[str, set[int]] = {}
    fold_counts: dict[int, int] = {}
    axis_counts: dict[str, int] = {}
    for row in rows:
        family = str(row.get("family") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        axis = str(row.get("axis") or row.get("source_evidence_tag") or "missing")
        axis_counts[axis] = axis_counts.get(axis, 0) + 1
        fold = _as_int(row.get("holdout_fold"))
        if fold is None:
            continue
        fold_counts[fold] = fold_counts.get(fold, 0) + 1
        family_folds.setdefault(family, set()).add(fold)

    required_set = set(folds_required)
    global_folds_present = {fold for fold in fold_counts if fold in required_set}
    missing_global_folds = sorted(required_set - global_folds_present)
    families_with_required_folds = sorted(
        family
        for family, folds in family_folds.items()
        if required_set.issubset(folds)
    )

    prediction_evaluations = [
        _evaluate_prediction_field(
            rows=rows,
            field=str(field),
            target=target,
            required_folds=folds_required,
            min_prediction_pairs_per_fold=min_prediction_pairs_per_fold,
            min_pearson_r=min_pearson_r,
        )
        for field in prediction_fields
    ]
    prediction_evaluations = [
        item for item in prediction_evaluations if item["present_pair_count"] > 0
    ]
    passing_predictions = [item for item in prediction_evaluations if item["passed"] is True]

    blockers: list[str] = []
    if len(rows) < min_rows:
        blockers.append(f"row_count_below_min:{len(rows)}<{min_rows}")
    if len(family_counts) < min_families:
        blockers.append(f"family_count_below_min:{len(family_counts)}<{min_families}")
    if len(families_with_required_folds) < min_families:
        blockers.append(
            "families_with_required_folds_below_min:"
            f"{len(families_with_required_folds)}<{min_families}"
        )
    if missing_global_folds:
        blockers.append(f"missing_global_holdout_folds:{missing_global_folds}")
    if require_single_axis:
        if "missing" in axis_counts:
            blockers.append("missing_axis_target")
        if len(axis_counts) != 1:
            blockers.append(f"mixed_axis_targets:{sorted(axis_counts)}")
    if not prediction_evaluations:
        blockers.append("no_prediction_fields_present")
    elif not passing_predictions:
        blockers.append("no_prediction_field_passed_heldout_correlation")

    status = "passed" if not blockers else "blocked"
    return {
        "schema": VALIDATION_GATE_SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "status": status,
        "passed": status == "passed",
        "blockers": blockers,
        "allowed_use": (
            "local_ll_surrogate_training_gate"
            if status == "passed"
            else "blocked_until_family_diversity_and_heldout_correlation"
        ),
        "not_allowed_uses": [
            "score_claim",
            "promotion",
            "rank_or_kill",
            "exact_eval_dispatch_selection_without_cuda_confirmation",
        ],
        "thresholds": {
            "min_rows": int(min_rows),
            "min_families": int(min_families),
            "required_folds": folds_required,
            "target": target,
            "prediction_fields": [str(field) for field in prediction_fields],
            "min_prediction_pairs_per_fold": int(min_prediction_pairs_per_fold),
            "min_pearson_r": float(min_pearson_r),
            "require_single_axis": bool(require_single_axis),
        },
        "coverage": {
            "row_count": len(rows),
            "family_counts": family_counts,
            "axis_counts": axis_counts,
            "fold_counts": {str(k): v for k, v in sorted(fold_counts.items())},
            "missing_global_folds": missing_global_folds,
            "families_with_required_folds": families_with_required_folds,
            "family_fold_counts": {
                family: {str(fold): fold_count_for_family(rows, family, fold) for fold in folds_required}
                for family in sorted(family_counts)
            },
        },
        "prediction_evaluations": prediction_evaluations,
        "passing_prediction_fields": [item["field"] for item in passing_predictions],
    }


def fold_count_for_family(rows: list[dict[str, Any]], family: str, fold: int) -> int:
    return sum(
        1
        for row in rows
        if str(row.get("family") or "unknown") == family and _as_int(row.get("holdout_fold")) == fold
    )


def _evaluate_prediction_field(
    *,
    rows: list[dict[str, Any]],
    field: str,
    target: str,
    required_folds: list[int],
    min_prediction_pairs_per_fold: int,
    min_pearson_r: float,
) -> dict[str, Any]:
    xs_all: list[float] = []
    ys_all: list[float] = []
    per_fold: list[dict[str, Any]] = []
    for fold in required_folds:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            if _as_int(row.get("holdout_fold")) != fold:
                continue
            pred = _as_float(row.get(field))
            observed = _as_float(row.get(target))
            if pred is None or observed is None:
                continue
            xs.append(pred)
            ys.append(observed)
        corr = _pearson(xs, ys)
        passed = (
            corr is not None
            and len(xs) >= min_prediction_pairs_per_fold
            and corr >= min_pearson_r
        )
        per_fold.append(
            {
                "fold": fold,
                "n": len(xs),
                "pearson_r": corr,
                "passed": passed,
            }
        )
        xs_all.extend(xs)
        ys_all.extend(ys)
    overall = _pearson(xs_all, ys_all)
    return {
        "field": field,
        "target": target,
        "present_pair_count": len(xs_all),
        "overall_pearson_r": overall,
        "folds": per_fold,
        "passed": bool(per_fold) and all(item["passed"] for item in per_fold),
    }


def _blocked_response_validation_gate(reason: str, *, row_count: int) -> dict[str, Any]:
    return {
        "schema": VALIDATION_GATE_SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "status": "blocked",
        "passed": False,
        "blockers": [f"dataset_validation_error:{reason}"],
        "allowed_use": "blocked_until_dataset_authority_contract_valid",
        "not_allowed_uses": [
            "score_claim",
            "promotion",
            "rank_or_kill",
            "exact_eval_dispatch_selection_without_cuda_confirmation",
        ],
        "thresholds": {},
        "coverage": {"row_count": row_count},
        "prediction_evaluations": [],
        "passing_prediction_fields": [],
    }


def build_next_probe_plan(
    dataset: dict[str, Any],
    *,
    null_byte_matrix: dict[str, Any] | None = None,
    null_byte_seed_budget_k: int = 16,
    allow_legacy_null_byte_matrix_missing_authority: bool = False,
    magic_codec_seed_boundary_smoke: dict[str, Any] | None = None,
    mlx_torch_parity_sweep: dict[str, Any] | None = None,
    allow_mlx_parity_research_signal_override: bool = False,
    decoder_q_response_surface: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic LL next-probe plan from response economics."""

    rows = dataset.get("rows")
    if not isinstance(rows, list):
        raise ScorerResponseDatasetError("dataset rows[] missing")
    try:
        response_validation_gate = build_scorer_response_validation_gate(dataset)
    except ScorerResponseDatasetError as exc:
        response_validation_gate = _blocked_response_validation_gate(
            str(exc),
            row_count=len(rows),
        )
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
    if not response_validation_gate["passed"]:
        prohibitions.append(
            {
                "rule": "do_not_use_response_dataset_for_exact_eval_selection",
                "reason": (
                    "response dataset has not passed family-diverse held-out "
                    "correlation validation"
                ),
                "gate_status": response_validation_gate["status"],
                "blockers": response_validation_gate["blockers"],
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

    mlx_torch_parity_sweep_gate = None
    if mlx_torch_parity_sweep is not None:
        mlx_torch_parity_sweep_gate = build_mlx_torch_parity_sweep_gate(
            mlx_torch_parity_sweep,
            allow_mlx_parity_research_signal_override=(
                allow_mlx_parity_research_signal_override
            ),
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
    mlx_rows = [
        row.get("row_id")
        for row in rows
        if isinstance(row, dict) and row.get("family") == "mlx_scorer_response"
    ][:8]
    if mlx_rows:
        if mlx_torch_parity_sweep_gate is None:
            prohibitions.append(
                {
                    "rule": "do_not_use_mlx_rows_without_torch_parity_sweep",
                    "reason": (
                        "MLX scorer-response rows require a PyTorch-vs-MLX "
                        "parity sweep gate before they can train or rank the LL "
                        "surrogate planner"
                    ),
                    "input_rows": mlx_rows,
                }
            )
            for probe in probes:
                probe["priority"] = int(probe["priority"]) + 1
            probes.insert(
                0,
                {
                    "probe_id": "ll_mlx_torch_parity_sweep_required",
                    "priority": 1,
                    "class": "parity_gate",
                    "rationale": (
                        "MLX rows exist, but planner use is blocked until an "
                        "MLX-vs-upstream-PyTorch parity sweep is attached."
                    ),
                    "input_rows": mlx_rows,
                    "acceptance_gate": (
                        "PASS_MLX_TORCH_SCORER_PARITY_SWEEP on the intended "
                        "cache/window before MLX rows train or rank LL probes"
                    ),
                },
            )
        elif not mlx_torch_parity_sweep_gate["mlx_rows_allowed_for_planner"]:
            prohibitions.append(
                {
                    "rule": "do_not_use_mlx_rows_after_failed_strict_parity_sweep",
                    "reason": (
                        "the attached MLX-vs-PyTorch sweep is not a strict pass; "
                        "repair parity or pass the explicit research-only override"
                    ),
                    "gate": mlx_torch_parity_sweep_gate,
                    "input_rows": mlx_rows,
                }
            )
            for probe in probes:
                probe["priority"] = int(probe["priority"]) + 1
            probes.insert(
                0,
                {
                    "probe_id": "ll_mlx_torch_parity_repair_or_override",
                    "priority": 1,
                    "class": "parity_gate",
                    "rationale": (
                        "The MLX parity sweep failed strict conformance, so MLX "
                        "rows stay out of LL planning unless parity is repaired "
                        "or explicitly demoted to research-only signal."
                    ),
                    "input_rows": mlx_rows,
                    "mlx_torch_parity_gate": mlx_torch_parity_sweep_gate,
                    "acceptance_gate": (
                        "strict parity pass, or explicit research-only override "
                        "with score_claim=false and exact-eval-before-promotion"
                    ),
                },
            )
        else:
            for probe in probes:
                probe["priority"] = int(probe["priority"]) + 1
            probes.insert(
                0,
                {
                    "probe_id": "ll_mlx_cpu_stable_response_harvest",
                    "priority": 1,
                    "class": "surrogate_training_data",
                    "rationale": (
                        "Direct MLX scorer-response rows are normalized as "
                        "non-authoritative local signal and have an attached "
                        "PyTorch-vs-MLX parity gate; expand stable CPU windows "
                        "before using MLX rows for surrogate fitting or spend filters."
                    ),
                    "input_rows": mlx_rows,
                    "mlx_torch_parity_gate": mlx_torch_parity_sweep_gate,
                    "acceptance_gate": (
                        ">=50 MLX rows across stable CPU windows/families, all "
                        "score_claim=false, with parity-gated held-out "
                        "correlation before any exact-eval dispatch selection"
                    ),
                },
            )
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

    decoder_q_response_surface_summary = None
    if decoder_q_response_surface is not None:
        decoder_q_response_surface_summary = _validate_decoder_q_response_surface(
            decoder_q_response_surface
        )
        for probe in probes:
            probe["priority"] = int(probe["priority"]) + 1
        probes.insert(
            0,
            {
                "probe_id": "ll_decoder_q_window_signed_response_surface",
                "priority": 1,
                "class": "byte_neutral_representation_mutation",
                "rationale": (
                    "Matched MLX family deltas show decoder-q response is "
                    "window-signed; preserve improving windows and suppress "
                    "or invert regressing windows before exact-eval spend."
                ),
                "input_rows": [],
                "response_surface_summary": decoder_q_response_surface_summary,
                "top_preserve_windows": decoder_q_response_surface.get("top_preserve_windows", [])[:8],
                "top_suppress_windows": decoder_q_response_surface.get("top_suppress_windows", [])[:8],
                "acceptance_gate": (
                    "new decoder-q candidate improves matched local response "
                    "surface and remains byte-neutral/fixed-length before "
                    "official inflate and exact CUDA eval"
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
        "response_validation_gate": response_validation_gate,
        "null_byte_priority_weights": null_byte_priority_weights,
        "magic_codec_seed_boundary": magic_codec_seed_boundary,
        "mlx_torch_parity_sweep_gate": mlx_torch_parity_sweep_gate,
        "decoder_q_response_surface_summary": decoder_q_response_surface_summary,
        "prohibitions": prohibitions,
        "probes": probes,
    }


def _validate_decoder_q_response_surface(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ScorerResponseDatasetError("decoder-q response surface must be a JSON object")
    if payload.get("schema") != "decoder_q_response_surface_plan.v1":
        raise ScorerResponseDatasetError("decoder-q response surface schema mismatch")
    _require_explicit_false_authority(payload, label="decoder-q response surface")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ScorerResponseDatasetError("decoder-q response surface summary missing")
    preserve = _as_int(summary.get("preserve_candidate_effect_count"))
    suppress = _as_int(summary.get("suppress_or_invert_candidate_effect_count"))
    matched = _as_int(summary.get("matched_count"))
    if preserve is None or suppress is None or matched is None:
        raise ScorerResponseDatasetError(
            "decoder-q response surface summary counts must be finite integers"
        )
    return {
        "schema": payload.get("schema"),
        "matched_count": matched,
        "preserve_candidate_effect_count": preserve,
        "suppress_or_invert_candidate_effect_count": suppress,
        "neutral_or_uncertain_count": _as_int(summary.get("neutral_or_uncertain_count")),
        "preserve_gain_sum": _as_float(summary.get("preserve_gain_sum")),
        "suppress_harm_sum": _as_float(summary.get("suppress_harm_sum")),
        "axis_dominance_counts": summary.get("axis_dominance_counts"),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
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
        if isinstance(scorer_delta, (int, float)) and (
            best_scorer is None
            or scorer_delta < best_scorer["scorer_delta_vs_baseline"]
        ):
            best_scorer = {
                "row_id": row["row_id"],
                "family": row["family"],
                "scorer_delta_vs_baseline": scorer_delta,
                "break_even_added_bytes_from_scorer_gain": row.get("break_even_added_bytes_from_scorer_gain"),
            }
        margin = row.get("byte_budget_margin_vs_break_even")
        if isinstance(margin, (int, float)) and (
            best_margin is None
            or margin > best_margin["byte_budget_margin_vs_break_even"]
        ):
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
        "window_baseline_score",
        "window_baseline_avg_posenet_dist",
        "window_baseline_avg_segnet_dist",
        "window_baseline_pose_term",
        "window_baseline_seg_term",
        "window_baseline_scorer_term",
        "diagnostic_seg_last_l1",
        "diagnostic_pose_pair_l1",
        "diagnostic_total_pair_l1",
        "decoder_q_score_impact_abs_sum",
        "decoder_q_axis_share_seg",
        "decoder_q_top_byte_count",
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


def render_validation_gate_markdown(gate: dict[str, Any]) -> str:
    coverage = gate.get("coverage") if isinstance(gate.get("coverage"), dict) else {}
    thresholds = gate.get("thresholds") if isinstance(gate.get("thresholds"), dict) else {}
    lines = [
        "# Scorer Response Validation Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Score claim: `{gate.get('score_claim')}`",
        f"- Allowed use: `{gate.get('allowed_use')}`",
        f"- Blockers: `{gate.get('blockers')}`",
        "",
        "## Thresholds",
        "",
        f"- Min rows: `{thresholds.get('min_rows')}`",
        f"- Min families: `{thresholds.get('min_families')}`",
        f"- Required folds: `{thresholds.get('required_folds')}`",
        f"- Target: `{thresholds.get('target')}`",
        f"- Min Pearson r: `{thresholds.get('min_pearson_r')}`",
        f"- Require single axis: `{thresholds.get('require_single_axis')}`",
        "",
        "## Coverage",
        "",
        f"- Rows: `{coverage.get('row_count')}`",
        f"- Family counts: `{coverage.get('family_counts')}`",
        f"- Axis counts: `{coverage.get('axis_counts')}`",
        f"- Fold counts: `{coverage.get('fold_counts')}`",
        f"- Families with required folds: `{coverage.get('families_with_required_folds')}`",
        "",
        "## Prediction Fields",
        "",
    ]
    for item in gate.get("prediction_evaluations", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            "- "
            f"`{item.get('field')}` n={item.get('present_pair_count')} "
            f"overall_r={item.get('overall_pearson_r')} "
            f"passed={item.get('passed')}"
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
    validation_gate = plan.get("response_validation_gate")
    if isinstance(validation_gate, dict):
        coverage = (
            validation_gate.get("coverage")
            if isinstance(validation_gate.get("coverage"), dict)
            else {}
        )
        lines.extend(["## Response Validation Gate", ""])
        lines.append(f"- Status: `{validation_gate.get('status')}`")
        lines.append(f"- Rows: `{coverage.get('row_count')}`")
        lines.append(f"- Families: `{coverage.get('family_counts')}`")
        lines.append(f"- Blockers: `{validation_gate.get('blockers')}`")
        lines.append("")
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
    mlx_plan = plan.get("mlx_scorer_response_execution_plan")
    if isinstance(mlx_plan, dict):
        execution = mlx_plan.get("recommended_execution")
        if isinstance(execution, dict):
            lines.extend(["## MLX Execution Recommendation", ""])
            lines.append(f"- Score claim: `{mlx_plan.get('score_claim')}`")
            lines.append(f"- Device: `{execution.get('device')}`")
            lines.append(f"- Batch pairs: `{execution.get('batch_pairs')}`")
            lines.append(f"- Pair window: `{execution.get('pair_window')}`")
            lines.append(f"- Response output: `{execution.get('response_output')}`")
            lines.append("")
    mlx_gate = plan.get("mlx_torch_parity_sweep_gate")
    if isinstance(mlx_gate, dict):
        gate_summary = mlx_gate.get("summary") if isinstance(mlx_gate.get("summary"), dict) else {}
        lines.extend(["## MLX Torch Parity Gate", ""])
        lines.append(f"- Status: `{mlx_gate.get('status')}`")
        lines.append(f"- Source verdict: `{mlx_gate.get('source_verdict')}`")
        lines.append(f"- Windows: `{mlx_gate.get('window_count')}`")
        lines.append(f"- Covered pair window: `{mlx_gate.get('covered_pair_window')}`")
        lines.append(
            "- Failed windows: "
            f"`{gate_summary.get('failed_windows')}`"
        )
        lines.append(
            "- Max SegNet argmax diff fraction: "
            f"`{gate_summary.get('segnet_argmax_diff_fraction_max')}`"
        )
        lines.append(f"- Blockers: `{mlx_gate.get('blockers')}`")
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
