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

from tac.exact_eval_custody import CONTEST_EXACT_SAMPLE_COUNT
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_production_contract import (
    BUNDLE_PASS_VERDICT as MLX_PRODUCTION_CONTRACT_BUNDLE_PASS_VERDICT,
)
from tac.local_acceleration.mlx_production_contract import (
    BUNDLE_SCHEMA_VERSION as MLX_PRODUCTION_CONTRACT_BUNDLE_SCHEMA,
)
from tac.local_acceleration.mlx_production_contract import (
    GATE_SET_VERSION as MLX_PRODUCTION_CONTRACT_GATE_SET_VERSION,
)
from tac.local_acceleration.mlx_production_contract import (
    PASS_VERDICT as MLX_PRODUCTION_CONTRACT_PASS_VERDICT,
)
from tac.local_acceleration.mlx_production_contract import (
    SCHEMA_VERSION as MLX_PRODUCTION_CONTRACT_SCHEMA,
)
from tac.local_acceleration.mlx_production_contract import (
    build_mlx_scorer_production_contract_manifest,
)
from tac.local_acceleration.mlx_score_calibration import (
    STRICT_AUTH_AXIS_SPEND_TRIAGE_ALLOWED_USE,
)
from tac.optimization.candidate_evidence_contract import CONTEST_UNCOMPRESSED_BYTES

SCHEMA = "scorer_response_dataset.v1"
ROW_SCHEMA = "scorer_response_row.v1"
NULL_BYTE_PRIORITY_SCHEMA = "ll_null_byte_priority_weights.v1"
CONSUMER_ROUTING_SCHEMA = "scorer_response_dataset_consumer_routing.v1"
VALIDATION_GATE_SCHEMA = "scorer_response_dataset_validation_gate.v1"
MLX_SCORER_RESPONSE_SCHEMA = "mlx_scorer_response.v1"
MLX_TORCH_PARITY_SWEEP_SCHEMA = "mlx_scorer_torch_parity_sweep.v1"
MLX_SCORE_CALIBRATION_SCHEMA = "mlx_score_calibration.v1"
LL_MLX_PRODUCTION_CONTRACT_GATE_SCHEMA = "ll_mlx_production_contract_gate.v1"
DECODER_Q_SURFACE_SIGN_CALIBRATION_SCHEMA = "decoder_q_surface_sign_calibration_labels.v1"
SOURCE_IDENTITY_REFRESH_SCHEMA = "scorer_response_source_identity_refresh.v1"
MLX_PARENT_PRODUCTION_CONTRACT_PLAN_SCHEMA = (
    "mlx_parent_production_contract_plan.v1"
)

DEFAULT_RESPONSE_PREDICTION_FIELDS = (
    "predicted_delta_vs_baseline_score",
    "predicted_scorer_delta_vs_baseline",
    "ll_predicted_delta_vs_baseline_score",
    "ll_predicted_scorer_delta_vs_baseline",
    "expected_delta_vs_baseline_score",
    "expected_scorer_delta_vs_baseline",
)
PREDICTION_CANDIDATE_FAMILY_TOP_K = (8, 16, 32)
PREDICTION_CANDIDATE_FAMILY_MIN_PEARSON_R = 0.2
PREDICTION_CANDIDATE_FAMILY_MIN_TOP_K_OVERLAP = 1
PREDICTION_CANDIDATE_FAMILY_MIN_NEGATIVE_PREDICTIONS = 1


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


def render_authority_markdown_block(
    payload: dict[str, Any],
    *,
    title: str = "Authority",
) -> list[str]:
    """Render the standard non-authority block for generated Markdown reports."""

    authority = payload.get("authority")
    if not isinstance(authority, dict):
        authority = {}

    def value(key: str, default: Any = False) -> Any:
        if key in payload:
            return payload.get(key)
        if key in authority:
            return authority.get(key)
        return default

    evidence_grade = value("evidence_grade", None)
    evidence_tag = value("evidence_tag", None)
    if evidence_tag is None:
        evidence_tag = _single_payload_evidence_tag(payload)
    if evidence_grade == EVIDENCE_GRADE_MLX and evidence_tag is None:
        evidence_tag = EVIDENCE_TAG_MLX
    score_axis = value("score_axis", evidence_tag)
    if evidence_grade == EVIDENCE_GRADE_MLX and score_axis is None:
        score_axis = EVIDENCE_TAG_MLX
    return [
        f"## {title}",
        "",
        f"- Evidence grade: `{evidence_grade}`",
        f"- Evidence tag: `{evidence_tag}`",
        f"- Score axis: `{score_axis}`",
        f"- Score claim: `{value('score_claim')}`",
        f"- Score claim valid: `{value('score_claim_valid')}`",
        f"- Promotion eligible: `{value('promotion_eligible')}`",
        f"- Rank/kill eligible: `{value('rank_or_kill_eligible')}`",
        f"- Ready for exact-eval dispatch: `{value('ready_for_exact_eval_dispatch')}`",
        f"- Promotable: `{value('promotable')}`",
        "",
    ]


def _single_payload_evidence_tag(payload: dict[str, Any]) -> str | None:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    tags = {
        str(row.get("axis") or row.get("source_evidence_tag"))
        for row in rows
        if isinstance(row, dict)
        and (row.get("axis") is not None or row.get("source_evidence_tag") is not None)
    }
    if len(tags) == 1:
        return next(iter(tags))
    return None


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


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


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
        "run_id": payload.get("run_id"),
        "authority": authority,
        "evidence_grade": payload.get("evidence_grade"),
        "evidence_tag": payload.get("evidence_tag"),
        "archive_sha256": payload.get("archive_sha256"),
        "inflated_outputs_aggregate_sha256": payload.get(
            "inflated_outputs_aggregate_sha256"
        ),
        "raw_sha256": payload.get("raw_sha256"),
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


def _require_auth_audited_mlx_window(payload: dict[str, Any], *, label: str) -> None:
    cache_identity = payload.get("cache_identity")
    if not isinstance(cache_identity, dict):
        raise ScorerResponseDatasetError(f"{label} cache_identity must be an object")
    candidate = cache_identity.get("candidate")
    if not isinstance(candidate, dict):
        raise ScorerResponseDatasetError(
            f"{label} cache_identity.candidate must be an object"
        )
    if candidate.get("eligible_for_local_mlx_transfer_calibration") is not True:
        raise ScorerResponseDatasetError(
            f"{label} candidate cache is not eligible for local MLX transfer calibration"
        )
    audit = candidate.get("auth_eval_identity_audit")
    if not isinstance(audit, dict):
        raise ScorerResponseDatasetError(
            f"{label} candidate cache auth_eval_identity_audit missing"
        )
    if audit.get("passed") is not True:
        raise ScorerResponseDatasetError(
            f"{label} candidate cache auth_eval_identity_audit did not pass"
        )
    if audit.get("verdict") != "PASS_CACHE_AUTH_EVAL_IDENTITY":
        raise ScorerResponseDatasetError(
            f"{label} candidate cache auth_eval_identity_audit verdict mismatch"
        )
    if audit.get("identity_residual") != 0:
        raise ScorerResponseDatasetError(
            f"{label} candidate cache auth_eval_identity_audit residual must be 0"
        )
    audit_payload = _load_auth_eval_identity_audit_payload(audit, label=label)
    candidate_arrays = _candidate_array_identity(payload, label=label)
    cache_arrays = _array_hashes_from_mapping(
        audit_payload.get("cache"),
        label=f"{label} auth_eval_identity_audit.cache",
    )
    auth_arrays = _array_hashes_from_mapping(
        audit_payload.get("auth_eval"),
        key="scorer_input_array_sha256",
        label=f"{label} auth_eval_identity_audit.auth_eval",
    )
    if candidate_arrays != cache_arrays:
        raise ScorerResponseDatasetError(
            f"{label} candidate cache arrays do not match dereferenced auth audit cache arrays"
        )
    if candidate_arrays != auth_arrays:
        raise ScorerResponseDatasetError(
            f"{label} candidate cache arrays do not match dereferenced auth audit auth arrays"
        )


def _load_auth_eval_identity_audit_payload(
    audit_stamp: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    path_value = audit_stamp.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ScorerResponseDatasetError(
            f"{label} candidate cache auth_eval_identity_audit.path missing"
        )
    audit_path = Path(path_value)
    try:
        payload = _load_json(audit_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ScorerResponseDatasetError(
            f"{label} candidate cache auth_eval_identity_audit.path unreadable: {exc}"
        ) from exc
    expected_sha = audit_stamp.get("sha256")
    if not _is_sha256(str(expected_sha or "")):
        raise ScorerResponseDatasetError(
            f"{label} candidate cache auth_eval_identity_audit.sha256 missing"
        )
    actual_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ScorerResponseDatasetError(
            f"{label} candidate cache auth_eval_identity_audit.sha256 mismatch"
        )
    if payload.get("schema_version") != "mlx_scorer_input_cache_auth_eval_audit.v1":
        raise ScorerResponseDatasetError(
            f"{label} dereferenced auth audit schema_version mismatch"
        )
    if payload.get("passed") is not True:
        raise ScorerResponseDatasetError(f"{label} dereferenced auth audit did not pass")
    if payload.get("verdict") != "PASS_CACHE_AUTH_EVAL_IDENTITY":
        raise ScorerResponseDatasetError(
            f"{label} dereferenced auth audit verdict mismatch"
        )
    if payload.get("identity_residual") != 0:
        raise ScorerResponseDatasetError(
            f"{label} dereferenced auth audit identity_residual must be 0"
        )
    for field in (
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "promotable",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if payload.get(field) is not False:
            raise ScorerResponseDatasetError(
                f"{label} dereferenced auth audit {field} must be false"
            )
    return payload


def _array_hashes_from_mapping(
    value: Any,
    *,
    label: str,
    key: str = "array_sha256",
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ScorerResponseDatasetError(f"{label} must be an object")
    arrays = value.get(key)
    if not isinstance(arrays, dict):
        raise ScorerResponseDatasetError(f"{label}.{key} must be an object")
    result: dict[str, str] = {}
    for item in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"):
        hash_value = str(arrays.get(item, ""))
        if not _is_sha256(hash_value):
            raise ScorerResponseDatasetError(f"{label}.{key}.{item} must be sha256")
        result[item] = hash_value
    return result


def _reference_array_identity(payload: dict[str, Any], *, label: str) -> dict[str, str]:
    cache_identity = payload.get("cache_identity")
    if not isinstance(cache_identity, dict):
        raise ScorerResponseDatasetError(f"{label} cache_identity must be an object")
    reference = cache_identity.get("reference")
    if not isinstance(reference, dict):
        raise ScorerResponseDatasetError(
            f"{label} cache_identity.reference must be an object"
        )
    arrays = reference.get("array_sha256")
    if not isinstance(arrays, dict):
        raise ScorerResponseDatasetError(
            f"{label} cache_identity.reference.array_sha256 must be an object"
        )
    result: dict[str, str] = {}
    for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"):
        value = str(arrays.get(key, ""))
        if not _is_sha256(value):
            raise ScorerResponseDatasetError(
                f"{label} cache_identity.reference.array_sha256.{key} must be sha256"
            )
        result[key] = value
    return result


def _candidate_array_identity(payload: dict[str, Any], *, label: str) -> dict[str, str]:
    cache_identity = payload.get("cache_identity")
    if not isinstance(cache_identity, dict):
        raise ScorerResponseDatasetError(f"{label} cache_identity must be an object")
    candidate = cache_identity.get("candidate")
    if not isinstance(candidate, dict):
        raise ScorerResponseDatasetError(
            f"{label} cache_identity.candidate must be an object"
        )
    arrays = candidate.get("array_sha256")
    if not isinstance(arrays, dict):
        raise ScorerResponseDatasetError(
            f"{label} cache_identity.candidate.array_sha256 must be an object"
        )
    result: dict[str, str] = {}
    for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"):
        value = str(arrays.get(key, ""))
        if not _is_sha256(value):
            raise ScorerResponseDatasetError(
                f"{label} cache_identity.candidate.array_sha256.{key} must be sha256"
            )
        result[key] = value
    return result


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
    source_n_samples = _as_int(_first_present(parent.get("n_samples"), candidate.get("n_samples")))
    normalized_full_video_scorer_gain = None
    projected_full_video_delta = None
    normalized_break_even_added_bytes = None
    normalized_byte_budget_margin = None
    if observed_scorer_gain is not None and source_n_samples is not None:
        normalized_full_video_scorer_gain = (
            observed_scorer_gain
            * float(source_n_samples)
            / float(CONTEST_EXACT_SAMPLE_COUNT)
        )
        if rate_delta is not None:
            projected_full_video_delta = rate_delta - normalized_full_video_scorer_gain
        normalized_break_even_added_bytes = (
            normalized_full_video_scorer_gain / RATE_SCORE_PER_BYTE
        )
        if added_archive_bytes is not None:
            normalized_byte_budget_margin = (
                normalized_break_even_added_bytes - float(added_archive_bytes)
            )
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
        "full_video_denominator": CONTEST_EXACT_SAMPLE_COUNT,
        "normalized_full_video_scorer_gain_vs_baseline": (
            normalized_full_video_scorer_gain
        ),
        "projected_full_video_delta_vs_baseline_score": projected_full_video_delta,
        "break_even_added_bytes_from_normalized_full_video_gain": (
            normalized_break_even_added_bytes
        ),
        "normalized_full_video_byte_budget_margin_vs_break_even": (
            normalized_byte_budget_margin
        ),
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
        "source_run_id": parent.get("run_id") or candidate.get("run_id"),
        "source_evidence_grade": parent.get("evidence_grade") or authority.get("evidence_grade"),
        "source_evidence_tag": parent.get("evidence_tag"),
        "source_hardware_substrate": parent.get("hardware_substrate"),
        "source_batch_pairs": _as_int(_first_present(parent.get("batch_pairs"), candidate.get("batch_pairs"))),
        "source_n_samples": source_n_samples,
        "source_pair_window": _first_present(parent.get("pair_window"), candidate.get("pair_window")),
        "source_start_pair": _as_int(_first_present(parent.get("start_pair"), candidate.get("start_pair"))),
        "source_max_pairs": _as_int(_first_present(parent.get("max_pairs"), candidate.get("max_pairs"))),
        "source_elapsed_seconds": _as_float(_first_present(parent.get("elapsed_seconds"), candidate.get("elapsed_seconds"))),
        "source_inflated_outputs_aggregate_sha256": (
            parent.get("inflated_outputs_aggregate_sha256")
            or _get_path(
                parent,
                ("cache_identity", "candidate", "inflated_outputs_aggregate_sha256"),
            )
        ),
        "source_candidate_cache_array_sha256": _get_path(
            parent,
            ("cache_identity", "candidate", "array_sha256"),
        ),
        "source_reference_cache_array_sha256": _get_path(
            parent,
            ("cache_identity", "reference", "array_sha256"),
        ),
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
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "authority": {
            "score_claim": False,
            "score_claim_valid": False,
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
    require_auth_audited_windows: bool = True,
) -> dict[str, Any]:
    """Build response rows with one MLX baseline per scorer-pair window.

    Direct MLX scorer-response rows are usually harvested on slices of the
    video. A single global baseline would make those deltas meaningless, so this
    helper matches each candidate response to a baseline response with the same
    scorer-pair window and then emits the normal non-promotional dataset shape.
    """

    baseline_by_window: dict[str, ResponseBaseline] = {}
    baseline_sources: dict[str, str] = {}
    baseline_reference_arrays: dict[str, dict[str, str]] = {}
    baseline_candidate_arrays: dict[str, dict[str, str]] = {}
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
            if require_auth_audited_windows:
                _require_auth_audited_mlx_window(payload, label=label)
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
            baseline_reference_arrays[key] = _reference_array_identity(
                payload,
                label=label,
            )
            baseline_candidate_arrays[key] = _candidate_array_identity(
                payload,
                label=label,
            )
        except (OSError, json.JSONDecodeError, ScorerResponseDatasetError, KeyError, ValueError) as exc:
            skipped.append({"path": str(path), "reason": f"baseline: {exc}"})

    rows: list[dict[str, Any]] = []
    for path in candidate_paths:
        try:
            payload = _load_json(path)
            if require_auth_audited_windows:
                _require_auth_audited_mlx_window(
                    payload,
                    label=f"candidate {path}",
                )
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
            if require_auth_audited_windows:
                candidate_reference_arrays = _reference_array_identity(
                    payload,
                    label=f"candidate {path}",
                )
                if candidate_reference_arrays != baseline_reference_arrays.get(key):
                    skipped.append(
                        {
                            "path": str(path),
                            "reason": (
                                "candidate: reference cache array identity mismatch "
                                f"for baseline window {key}"
                            ),
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
            row["window_baseline_candidate_cache_array_sha256"] = (
                baseline_candidate_arrays.get(key)
            )
            row["window_baseline_reference_cache_array_sha256"] = (
                baseline_reference_arrays.get(key)
            )
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
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "authority": {
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "promotable": False,
            "evidence_grade": EVIDENCE_GRADE_MLX,
            "evidence_tag": EVIDENCE_TAG_MLX,
            "score_axis": EVIDENCE_TAG_MLX,
            "notes": (
                "Rows are local MLX scorer-response observations paired with "
                "same-window MLX baselines for surrogate fitting and ranking only."
            ),
            "require_auth_audited_windows": require_auth_audited_windows,
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


def refresh_mlx_scorer_response_source_identity(
    dataset: dict[str, Any],
) -> dict[str, Any]:
    """Refresh MLX scorer-response row identity from each row's source payload.

    Historical derived datasets can retain prediction/features while losing
    production-contract identity fields. This helper re-reads each MLX
    scorer-response row's source payload, verifies non-authority, and fills only
    matching or missing custody fields. Any mismatch is fail-closed.
    """

    rows = dataset.get("rows")
    if not isinstance(rows, list):
        raise ScorerResponseDatasetError("dataset rows[] missing")
    out = copy.deepcopy(dataset)
    out_rows = out["rows"]
    blockers: list[str] = []
    refreshed_row_ids: list[str] = []
    updated_row_ids: list[str] = []
    changed_field_count = 0
    mlx_row_count = 0
    required_fields = (
        "archive_sha256",
        "source_inflated_outputs_aggregate_sha256",
        "source_batch_pairs",
        "source_n_samples",
        "source_pair_window",
    )
    for row in out_rows:
        if not isinstance(row, dict) or not _is_mlx_scorer_response_row(row):
            continue
        mlx_row_count += 1
        row_id = str(row.get("row_id") or "<unknown>")
        source_path_value = row.get("source_path")
        if not isinstance(source_path_value, str) or not source_path_value:
            blockers.append(f"source_identity_source_path_missing:{row_id}")
            continue
        source_path = Path(source_path_value)
        try:
            payload = _load_json(source_path)
            _require_mlx_response_false_authority(
                payload,
                label=f"{source_path}:source_identity_refresh",
                allow_candidate_array_identity=True,
            )
        except (OSError, json.JSONDecodeError, ScorerResponseDatasetError) as exc:
            blockers.append(f"source_identity_source_payload_invalid:{row_id}:{exc}")
            continue

        identity = _mlx_source_identity_fields(payload)
        row_blockers_start = len(blockers)
        row_changed = False
        for field, value in identity.items():
            if value is None:
                continue
            existing = row.get(field)
            if existing not in (None, "") and existing != value:
                blockers.append(f"source_identity_field_mismatch:{row_id}:{field}")
                continue
            if existing != value:
                row[field] = copy.deepcopy(value)
                row_changed = True
                changed_field_count += 1
        for field in required_fields:
            value = row.get(field)
            if value is None or value == "":
                blockers.append(f"source_identity_required_field_missing:{row_id}:{field}")
        if len(blockers) == row_blockers_start:
            refreshed_row_ids.append(row_id)
            if row_changed:
                updated_row_ids.append(row_id)

    passed = not blockers
    out["source_identity_refresh"] = {
        "schema": SOURCE_IDENTITY_REFRESH_SCHEMA,
        "producer": TOOL,
        "passed": passed,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "mlx_row_count": mlx_row_count,
        "refreshed_row_count": len(refreshed_row_ids),
        "updated_row_count": len(updated_row_ids),
        "changed_field_count": changed_field_count,
        "refreshed_row_ids_sample": refreshed_row_ids[:8],
        "updated_row_ids_sample": updated_row_ids[:8],
        "blockers": blockers,
    }
    summary = out.get("summary")
    if isinstance(summary, dict):
        summary["mlx_source_identity_refresh_passed"] = passed
        summary["mlx_source_identity_refreshed_row_count"] = len(refreshed_row_ids)
        summary["mlx_source_identity_updated_row_count"] = len(updated_row_ids)
        summary["mlx_source_identity_changed_field_count"] = changed_field_count
    return out


def _mlx_source_identity_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "archive_sha256": payload.get("archive_sha256"),
        "raw_sha256": payload.get("raw_sha256"),
        "source_batch_pairs": _as_int(payload.get("batch_pairs")),
        "source_n_samples": _as_int(payload.get("n_samples")),
        "source_pair_window": payload.get("pair_window"),
        "source_start_pair": _as_int(payload.get("start_pair")),
        "source_max_pairs": _as_int(payload.get("max_pairs")),
        "source_elapsed_seconds": _as_float(payload.get("elapsed_seconds")),
        "source_inflated_outputs_aggregate_sha256": (
            payload.get("inflated_outputs_aggregate_sha256")
            or _get_path(
                payload,
                ("cache_identity", "candidate", "inflated_outputs_aggregate_sha256"),
            )
        ),
        "source_candidate_cache_array_sha256": _get_path(
            payload,
            ("cache_identity", "candidate", "array_sha256"),
        ),
        "source_reference_cache_array_sha256": _get_path(
            payload,
            ("cache_identity", "reference", "array_sha256"),
        ),
        "source_posenet_sha256": _get_path(payload, ("components", "posenet_sha256")),
        "source_segnet_sha256": _get_path(payload, ("components", "segnet_sha256")),
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
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "authority": {
            "score_claim": False,
            "score_claim_valid": False,
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


def build_mlx_score_calibration_gate(calibration: dict[str, Any]) -> dict[str, Any]:
    """Normalize MLX score-calibration evidence for spend-triage gating.

    The gate may allow local MLX pairwise ordering to filter exact-eval spend,
    but it never creates score, rank, or promotion authority.
    """

    if not isinstance(calibration, dict):
        raise ScorerResponseDatasetError("MLX score calibration must be a JSON object")
    source_schema = calibration.get("schema_version") or calibration.get("schema")
    if source_schema != MLX_SCORE_CALIBRATION_SCHEMA:
        raise ScorerResponseDatasetError("MLX score calibration schema mismatch")
    _require_explicit_false_authority(
        calibration,
        label="MLX score calibration",
        fields=(
            "score_claim",
            "score_claim_valid",
            "promotion_eligible",
            "ready_for_exact_eval_dispatch",
            "rank_or_kill_eligible",
        ),
    )
    if calibration.get("candidate_generation_only") is not True:
        raise ScorerResponseDatasetError(
            "MLX score calibration candidate_generation_only must be true"
        )
    if calibration.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise ScorerResponseDatasetError("MLX score calibration evidence_grade mismatch")
    if calibration.get("evidence_tag") != EVIDENCE_TAG_MLX:
        raise ScorerResponseDatasetError("MLX score calibration evidence_tag mismatch")

    decision_policy = calibration.get("decision_policy")
    if not isinstance(decision_policy, dict):
        raise ScorerResponseDatasetError("MLX score calibration decision_policy missing")
    _require_explicit_false_authority(
        decision_policy,
        label="MLX score calibration decision_policy",
        fields=(
            "score_claim",
            "promotion_eligible",
            "ready_for_exact_eval_dispatch",
            "rank_or_kill_eligible",
        ),
    )
    if decision_policy.get("allowed_use") != STRICT_AUTH_AXIS_SPEND_TRIAGE_ALLOWED_USE:
        raise ScorerResponseDatasetError(
            "MLX score calibration decision_policy allowed_use mismatch"
        )
    if decision_policy.get("forbidden_use") != "score_claim_or_rank_or_kill_or_promotion":
        raise ScorerResponseDatasetError(
            "MLX score calibration decision_policy forbidden_use mismatch"
        )

    summary = calibration.get("summary")
    if not isinstance(summary, dict):
        raise ScorerResponseDatasetError("MLX score calibration summary missing")
    uncertain_count = _as_int(summary.get("mlx_spend_triage_pairwise_uncertain_count"))
    certified_count = _as_int(summary.get("mlx_spend_triage_pairwise_certified_count"))
    total_count = _as_int(summary.get("mlx_spend_triage_pairwise_total_count"))
    min_gap = _as_float(summary.get("recommended_min_mlx_gap_for_spend_triage"))
    uncertainty = _as_float(summary.get("calibration_uncertainty_score"))
    if min_gap is None:
        min_gap = _as_float(decision_policy.get("recommended_min_mlx_gap_for_spend_triage"))
    if uncertainty is None:
        uncertainty = _as_float(decision_policy.get("calibration_uncertainty_score"))

    blockers: list[str] = []
    if uncertain_count is None:
        blockers.append("mlx_score_calibration_uncertain_count_missing")
    elif uncertain_count > 0:
        blockers.append("mlx_score_calibration_has_uncertain_pairwise_decisions")
    if certified_count is None or certified_count <= 0:
        blockers.append("mlx_score_calibration_has_no_certified_pairwise_decisions")
    if total_count is None or total_count <= 0:
        blockers.append("mlx_score_calibration_pairwise_total_missing")
    if min_gap is None or min_gap <= 0.0:
        blockers.append("mlx_score_calibration_min_gap_missing_or_nonpositive")
    if uncertainty is None or uncertainty < 0.0:
        blockers.append("mlx_score_calibration_uncertainty_missing_or_negative")

    status = "strict_pass" if not blockers else "blocked"
    return {
        "schema": "ll_mlx_score_calibration_gate.v1",
        "producer": TOOL,
        "source_schema": source_schema,
        "source_run_id": calibration.get("run_id"),
        "status": status,
        "mlx_spend_triage_allowed": status == "strict_pass",
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
        "summary": {
            "certified_pairwise_count": certified_count,
            "uncertain_pairwise_count": uncertain_count,
            "total_pairwise_count": total_count,
            "recommended_min_mlx_gap_for_spend_triage": min_gap,
            "calibration_uncertainty_score": uncertainty,
            "decision_safety_factor": _as_float(
                decision_policy.get("decision_safety_factor")
            ),
            "calibration_uncertainty_basis": decision_policy.get(
                "calibration_uncertainty_basis"
            ),
        },
        "blockers": blockers,
        "allowed_use": (
            "local_exact_eval_spend_triage_filter"
            if not blockers
            else "blocked_until_calibration_pairwise_decisions_are_certified"
        ),
    }


def _mlx_contract_row_identity_blockers(
    response_summary: dict[str, Any],
    rows: Iterable[dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    expected_archive = response_summary.get("archive_sha256")
    expected_inflated = response_summary.get("inflated_outputs_aggregate_sha256")
    expected_batch_pairs = _as_int(response_summary.get("batch_pairs"))
    expected_n_samples = _as_int(response_summary.get("n_samples"))
    expected_pair_window = response_summary.get("pair_window")

    for row in rows:
        row_id = str(row.get("row_id") or "<unknown>")
        checks = (
            ("archive_sha256", row.get("archive_sha256"), expected_archive),
            (
                "inflated_outputs_aggregate_sha256",
                row.get("source_inflated_outputs_aggregate_sha256"),
                expected_inflated,
            ),
            ("batch_pairs", _as_int(row.get("source_batch_pairs")), expected_batch_pairs),
            ("n_samples", _as_int(row.get("source_n_samples")), expected_n_samples),
            ("pair_window", row.get("source_pair_window"), expected_pair_window),
        )
        for field, actual, expected in checks:
            if actual is None or expected is None:
                blockers.append(
                    f"mlx_production_contract_row_{field}_missing:{row_id}"
                )
                continue
            if actual != expected:
                blockers.append(
                    f"mlx_production_contract_row_{field}_mismatch:{row_id}"
                )
        rich_checks = (
            ("response_run_id", row.get("source_run_id"), response_summary.get("response_run_id")),
            (
                "candidate_cache_array_sha256",
                row.get("source_candidate_cache_array_sha256"),
                response_summary.get("candidate_cache_array_sha256"),
            ),
            (
                "reference_cache_array_sha256",
                row.get("source_reference_cache_array_sha256"),
                response_summary.get("reference_cache_array_sha256"),
            ),
            ("posenet_sha256", row.get("source_posenet_sha256"), response_summary.get("posenet_sha256")),
            ("segnet_sha256", row.get("source_segnet_sha256"), response_summary.get("segnet_sha256")),
        )
        for field, actual, expected in rich_checks:
            if actual is None:
                continue
            if expected is None:
                blockers.append(
                    f"mlx_production_contract_row_{field}_missing:{row_id}"
                )
                continue
            if actual != expected:
                blockers.append(
                    f"mlx_production_contract_row_{field}_mismatch:{row_id}"
                )
    return blockers


def _pair_window_contains(parent: Any, child: Any) -> bool:
    if not (
        isinstance(parent, list)
        and isinstance(child, list)
        and len(parent) == 2
        and len(child) == 2
    ):
        return False
    parent_start = _as_int(parent[0])
    parent_stop = _as_int(parent[1])
    child_start = _as_int(child[0])
    child_stop = _as_int(child[1])
    if None in (parent_start, parent_stop, child_start, child_stop):
        return False
    return parent_start <= child_start and child_stop <= parent_stop


def _mlx_contract_summary_covers_row_window(
    summary: dict[str, Any],
    row: dict[str, Any],
) -> bool:
    """Return true when a strict parent-window contract covers a child row.

    This is intentionally narrower than exact identity matching. It requires
    archive/inflated identity, batch shape, full scorer-input cache array
    hashes, and pair-window containment. It does not compare response component
    hashes because a parent response's full distortion-vector hash cannot equal
    a child singleton-window response's slice hash.
    """

    if not _pair_window_contains(
        summary.get("pair_window"),
        row.get("source_pair_window"),
    ):
        return False
    checks = (
        (summary.get("archive_sha256"), row.get("archive_sha256")),
        (
            summary.get("inflated_outputs_aggregate_sha256"),
            row.get("source_inflated_outputs_aggregate_sha256"),
        ),
        (_as_int(summary.get("batch_pairs")), _as_int(row.get("source_batch_pairs"))),
        (
            summary.get("candidate_cache_array_sha256"),
            row.get("source_candidate_cache_array_sha256"),
        ),
        (
            summary.get("reference_cache_array_sha256"),
            row.get("source_reference_cache_array_sha256"),
        ),
    )
    return all(expected is not None and expected == actual for expected, actual in checks)


def _is_mlx_scorer_response_row(row: Any) -> bool:
    return isinstance(row, dict) and (
        row.get("family") == "mlx_scorer_response"
        or row.get("source_schema") == MLX_SCORER_RESPONSE_SCHEMA
    )


def _mlx_contract_summary_identity_key(
    summary: dict[str, Any],
) -> tuple[Any, ...] | None:
    batch_pairs = _as_int(summary.get("batch_pairs"))
    n_samples = _as_int(summary.get("n_samples"))
    pair_window = summary.get("pair_window")
    if not isinstance(pair_window, list):
        return None
    values = (
        summary.get("archive_sha256"),
        summary.get("inflated_outputs_aggregate_sha256"),
        batch_pairs,
        n_samples,
        tuple(pair_window),
    )
    if any(value is None for value in values):
        return None
    return values


def _mlx_row_identity_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    batch_pairs = _as_int(row.get("source_batch_pairs"))
    n_samples = _as_int(row.get("source_n_samples"))
    pair_window = row.get("source_pair_window")
    if not isinstance(pair_window, list):
        return None
    values = (
        row.get("archive_sha256"),
        row.get("source_inflated_outputs_aggregate_sha256"),
        batch_pairs,
        n_samples,
        tuple(pair_window),
    )
    if any(value is None for value in values):
        return None
    return values


def _build_mlx_production_contract_bundle_gate(
    bundle: dict[str, Any],
    *,
    rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    row_list = None if rows is None else list(rows)
    _require_explicit_false_authority(
        bundle,
        label="MLX production contract bundle",
        fields=(
            "score_authority",
            "contest_authority",
            "score_claim",
            "score_claim_valid",
            "promotion_eligible",
            "ready_for_exact_eval_dispatch",
            "rank_or_kill_eligible",
            "promotable",
        ),
    )
    if bundle.get("candidate_generation_only") is not True:
        raise ScorerResponseDatasetError(
            "MLX production contract bundle candidate_generation_only must be true"
        )
    if bundle.get("requires_exact_eval_before_promotion") is not True:
        raise ScorerResponseDatasetError(
            "MLX production contract bundle requires_exact_eval_before_promotion must be true"
        )
    if bundle.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise ScorerResponseDatasetError(
            "MLX production contract bundle evidence_grade mismatch"
        )
    if bundle.get("evidence_tag") != EVIDENCE_TAG_MLX:
        raise ScorerResponseDatasetError(
            "MLX production contract bundle evidence_tag mismatch"
        )
    if bundle.get("score_axis") != EVIDENCE_TAG_MLX:
        raise ScorerResponseDatasetError(
            "MLX production contract bundle score_axis mismatch"
        )
    contracts = bundle.get("contracts")
    if not isinstance(contracts, list):
        raise ScorerResponseDatasetError(
            "MLX production contract bundle contracts must be a list"
        )

    blockers: list[str] = []
    if bundle.get("passed") is not True:
        blockers.append("mlx_production_contract_bundle_not_passed")
    if bundle.get("verdict") != MLX_PRODUCTION_CONTRACT_BUNDLE_PASS_VERDICT:
        blockers.append("mlx_production_contract_bundle_verdict_not_pass")
    child_summaries: list[dict[str, Any]] = []
    contracts_by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    strict_contracts: list[dict[str, Any]] = []
    for index, contract in enumerate(contracts):
        if not isinstance(contract, dict):
            blockers.append(f"mlx_production_contract_bundle_child_not_object:{index}")
            continue
        gate = build_mlx_production_contract_gate(contract)
        summary = gate.get("summary") if isinstance(gate.get("summary"), dict) else {}
        child_summary = {
            "index": index,
            "status": gate.get("status"),
            "source_run_id": gate.get("source_run_id"),
            "source_verdict": gate.get("source_verdict"),
            "blockers": list(gate.get("blockers") or []),
            "summary": summary,
        }
        child_summaries.append(child_summary)
        if gate.get("status") != "strict_pass":
            blockers.append(f"mlx_production_contract_bundle_child_blocked:{index}")
            continue
        strict_contracts.append(
            {
                "index": index,
                "gate": gate,
                "summary": summary,
            }
        )
        identity_key = _mlx_contract_summary_identity_key(summary)
        if identity_key is None:
            blockers.append(f"mlx_production_contract_bundle_child_identity_missing:{index}")
            continue
        contracts_by_identity.setdefault(identity_key, []).append(
            {
                "index": index,
                "gate": gate,
                "summary": summary,
            }
        )

    matched_row_ids: list[str] = []
    parent_window_matched_row_ids: list[str] = []
    unmatched_row_ids: list[str] = []
    if row_list is not None:
        for row in row_list:
            row_id = str(row.get("row_id") or "<unknown>")
            identity_key = _mlx_row_identity_key(row)
            if identity_key is None:
                unmatched_row_ids.append(row_id)
                blockers.append(f"mlx_production_contract_bundle_row_identity_missing:{row_id}")
                continue
            matching_contracts = contracts_by_identity.get(identity_key, [])
            if not matching_contracts:
                parent_matches = [
                    item
                    for item in strict_contracts
                    if _mlx_contract_summary_covers_row_window(item["summary"], row)
                ]
                if not parent_matches:
                    unmatched_row_ids.append(row_id)
                    blockers.append(f"mlx_production_contract_bundle_row_unmatched:{row_id}")
                    continue
                matched_row_ids.append(row_id)
                parent_window_matched_row_ids.append(row_id)
                continue
            detail_blocker_sets = [
                _mlx_contract_row_identity_blockers(item["summary"], [row])
                for item in matching_contracts
            ]
            if not any(not detail_blockers for detail_blockers in detail_blocker_sets):
                unmatched_row_ids.append(row_id)
                blockers.append(
                    f"mlx_production_contract_bundle_row_detail_mismatch:{row_id}"
                )
                for detail_blockers in detail_blocker_sets:
                    blockers.extend(detail_blockers)
                continue
            matched_row_ids.append(row_id)

    status = "strict_pass" if not blockers else "blocked"
    return {
        "schema": LL_MLX_PRODUCTION_CONTRACT_GATE_SCHEMA,
        "producer": TOOL,
        "source_schema": MLX_PRODUCTION_CONTRACT_BUNDLE_SCHEMA,
        "source_run_id": bundle.get("run_id"),
        "source_verdict": bundle.get("verdict"),
        "source_passed": bundle.get("passed") is True,
        "status": status,
        "mlx_production_signal_allowed": status == "strict_pass",
        "mlx_spend_triage_allowed": status == "strict_pass",
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
        "summary": {
            "contract_count": len(contracts),
            "strict_contract_count": sum(
                1 for item in child_summaries if item.get("status") == "strict_pass"
            ),
            "row_count": 0 if row_list is None else len(row_list),
            "matched_row_count": len(matched_row_ids),
            "parent_window_matched_row_count": len(parent_window_matched_row_ids),
            "parent_window_matched_row_ids": parent_window_matched_row_ids[:8],
            "unmatched_row_count": len(unmatched_row_ids),
            "unmatched_row_ids": unmatched_row_ids[:8],
            "child_contracts": child_summaries[:8],
        },
        "blockers": blockers,
        "allowed_use": (
            "local_exact_eval_spend_triage_filter_after_strict_production_contract_bundle"
            if not blockers
            else "blocked_until_every_mlx_row_has_matching_strict_production_contract"
        ),
    }


def build_mlx_production_contract_gate(
    contract: dict[str, Any],
    *,
    rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize the strict MLX production contract for LL planner spend gating.

    A passing contract allows local MLX rows to filter exact-eval spend only as
    a non-authoritative signal. It does not create score, rank, promotion, or
    dispatch authority.
    """

    if not isinstance(contract, dict):
        raise ScorerResponseDatasetError("MLX production contract must be a JSON object")
    source_schema = contract.get("schema_version") or contract.get("schema")
    if source_schema == MLX_PRODUCTION_CONTRACT_BUNDLE_SCHEMA:
        return _build_mlx_production_contract_bundle_gate(contract, rows=rows)
    if source_schema != MLX_PRODUCTION_CONTRACT_SCHEMA:
        raise ScorerResponseDatasetError("MLX production contract schema mismatch")
    _require_explicit_false_authority(
        contract,
        label="MLX production contract",
        fields=(
            "score_authority",
            "contest_authority",
            "score_claim",
            "score_claim_valid",
            "promotion_eligible",
            "ready_for_exact_eval_dispatch",
            "rank_or_kill_eligible",
            "promotable",
        ),
    )
    if contract.get("gate_set_version") != MLX_PRODUCTION_CONTRACT_GATE_SET_VERSION:
        raise ScorerResponseDatasetError("MLX production contract gate_set_version mismatch")
    if contract.get("candidate_generation_only") is not True:
        raise ScorerResponseDatasetError(
            "MLX production contract candidate_generation_only must be true"
        )
    if contract.get("requires_exact_eval_before_promotion") is not True:
        raise ScorerResponseDatasetError(
            "MLX production contract requires_exact_eval_before_promotion must be true"
        )
    if contract.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise ScorerResponseDatasetError(
            "MLX production contract evidence_grade mismatch"
        )
    if contract.get("evidence_tag") != EVIDENCE_TAG_MLX:
        raise ScorerResponseDatasetError("MLX production contract evidence_tag mismatch")
    if contract.get("score_axis") != EVIDENCE_TAG_MLX:
        raise ScorerResponseDatasetError("MLX production contract score_axis mismatch")

    required_gates = (
        contract.get("required_gates")
        if isinstance(contract.get("required_gates"), dict)
        else {}
    )
    response_summary = (
        contract.get("response_summary")
        if isinstance(contract.get("response_summary"), dict)
        else {}
    )
    source_blockers = list(contract.get("blockers") or [])
    blockers = list(source_blockers)
    if contract.get("passed") is not True:
        blockers.append("mlx_production_contract_not_passed")
    if contract.get("verdict") != MLX_PRODUCTION_CONTRACT_PASS_VERDICT:
        blockers.append("mlx_production_contract_verdict_not_pass")
    if required_gates.get("strict_gate_policy") is not True:
        blockers.append("mlx_production_contract_strict_gate_policy_not_true")
    for gate_name in (
        "cache_identity",
        "cache_auth_audit",
        "torch_parity",
        "reference_torch_parity",
        "profile_stability",
        "score_calibration",
    ):
        if required_gates.get(gate_name) is not True:
            blockers.append(
                f"mlx_production_contract_required_gate_{gate_name}_not_true"
            )
    if not response_summary:
        blockers.append("mlx_production_contract_response_summary_missing")
    if rows is not None:
        blockers.extend(_mlx_contract_row_identity_blockers(response_summary, rows))

    status = "strict_pass" if not blockers else "blocked"
    return {
        "schema": LL_MLX_PRODUCTION_CONTRACT_GATE_SCHEMA,
        "producer": TOOL,
        "source_schema": source_schema,
        "source_run_id": contract.get("run_id"),
        "source_verdict": contract.get("verdict"),
        "source_passed": contract.get("passed") is True,
        "status": status,
        "mlx_production_signal_allowed": status == "strict_pass",
        "mlx_spend_triage_allowed": status == "strict_pass",
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
        "summary": {
            "archive_sha256": response_summary.get("archive_sha256"),
            "inflated_outputs_aggregate_sha256": response_summary.get(
                "inflated_outputs_aggregate_sha256"
            ),
            "response_schema_version": response_summary.get("schema_version"),
            "response_run_id": response_summary.get("response_run_id"),
            "hardware_substrate": response_summary.get("hardware_substrate"),
            "batch_pairs": _as_int(response_summary.get("batch_pairs")),
            "n_samples": _as_int(response_summary.get("n_samples")),
            "pair_window": response_summary.get("pair_window"),
            "candidate_cache_array_sha256": response_summary.get(
                "candidate_cache_array_sha256"
            ),
            "reference_cache_array_sha256": response_summary.get(
                "reference_cache_array_sha256"
            ),
            "posenet_sha256": response_summary.get("posenet_sha256"),
            "segnet_sha256": response_summary.get("segnet_sha256"),
            "required_gates": dict(required_gates),
            "warnings": list(contract.get("warnings") or []),
        },
        "blockers": blockers,
        "allowed_use": (
            "local_exact_eval_spend_triage_filter_after_strict_production_contract"
            if not blockers
            else "blocked_until_strict_mlx_production_contract_passes"
        ),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mlx_parent_contract_group_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    batch_pairs = _as_int(row.get("source_batch_pairs"))
    candidate_cache = row.get("source_candidate_cache_array_sha256")
    reference_cache = row.get("source_reference_cache_array_sha256")
    values = (
        row.get("archive_sha256"),
        row.get("source_inflated_outputs_aggregate_sha256"),
        batch_pairs,
        _canonical_json(candidate_cache) if isinstance(candidate_cache, dict) else None,
        _canonical_json(reference_cache) if isinstance(reference_cache, dict) else None,
    )
    if any(value is None for value in values):
        return None
    return values


def _valid_pair_window(value: Any) -> list[int] | None:
    if not (isinstance(value, list) and len(value) == 2):
        return None
    start = _as_int(value[0])
    stop = _as_int(value[1])
    if start is None or stop is None or start >= stop:
        return None
    return [start, stop]


def _contract_summary_covers_parent_group(
    summary: dict[str, Any],
    group: dict[str, Any],
) -> bool:
    required_window = group.get("required_parent_pair_window")
    if not _pair_window_contains(summary.get("pair_window"), required_window):
        return False
    checks = (
        (summary.get("archive_sha256"), group.get("archive_sha256")),
        (
            summary.get("inflated_outputs_aggregate_sha256"),
            group.get("inflated_outputs_aggregate_sha256"),
        ),
        (_as_int(summary.get("batch_pairs")), _as_int(group.get("source_batch_pairs"))),
        (
            summary.get("candidate_cache_array_sha256"),
            group.get("candidate_cache_array_sha256"),
        ),
        (
            summary.get("reference_cache_array_sha256"),
            group.get("reference_cache_array_sha256"),
        ),
    )
    return all(expected is not None and expected == actual for expected, actual in checks)


def _cache_auth_audit_summaries(
    cache_auth_audits: Iterable[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if cache_auth_audits is None:
        return [], []
    summaries: list[dict[str, Any]] = []
    blockers: list[str] = []
    for index, audit in enumerate(cache_auth_audits):
        if not isinstance(audit, dict):
            blockers.append(f"cache_auth_audit_not_object:{index}")
            continue
        source_path = audit.get("_source_path")
        prefix = f"cache_auth_audit_invalid:{index}"
        if audit.get("schema_version") != "mlx_scorer_input_cache_auth_eval_audit.v1":
            blockers.append(f"{prefix}:schema_version")
            continue
        for field in _FALSE_AUTHORITY_FIELDS:
            value = audit.get(field)
            if value is None and field in _LEGACY_EXTENDED_FALSE_AUTHORITY_FIELDS:
                continue
            if value is not False:
                blockers.append(f"{prefix}:{field}_not_false")
        cache = audit.get("cache")
        if not isinstance(cache, dict):
            blockers.append(f"{prefix}:cache_missing")
            continue
        auth_eval = audit.get("auth_eval")
        if not isinstance(auth_eval, dict):
            blockers.append(f"{prefix}:auth_eval_missing")
            continue
        cache_archive_sha256 = cache.get("archive_sha256")
        cache_inflated_sha256 = cache.get("inflated_outputs_aggregate_sha256")
        cache_raw_sha256 = cache.get("raw_sha256")
        cache_array_sha256 = cache.get("array_sha256")
        auth_archive_sha256 = auth_eval.get("archive_sha256")
        auth_inflated_sha256 = auth_eval.get("inflated_outputs_aggregate_sha256")
        auth_raw_sha256 = auth_eval.get("raw_file_sha256")
        auth_array_sha256 = auth_eval.get("scorer_input_array_sha256")
        if not isinstance(auth_archive_sha256, str) or not auth_archive_sha256:
            blockers.append(f"{prefix}:auth_archive_sha256_missing")
            continue
        if not isinstance(auth_inflated_sha256, str) or not auth_inflated_sha256:
            blockers.append(f"{prefix}:auth_inflated_outputs_aggregate_sha256_missing")
            continue
        if not isinstance(auth_raw_sha256, str) or not auth_raw_sha256:
            blockers.append(f"{prefix}:auth_raw_sha256_missing")
            continue
        if not auth_eval.get("scorer_input_hash_domain"):
            blockers.append(f"{prefix}:auth_hash_domain_missing")
            continue
        if _as_int(auth_eval.get("n_samples")) is None:
            blockers.append(f"{prefix}:auth_n_samples_missing")
            continue
        if not isinstance(auth_array_sha256, dict):
            blockers.append(f"{prefix}:auth_scorer_input_array_sha256_missing")
            continue
        missing_hashes = [
            key
            for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb")
            if not isinstance(auth_array_sha256.get(key), str)
            or not auth_array_sha256.get(key)
        ]
        if missing_hashes:
            blockers.append(
                f"{prefix}:auth_scorer_input_array_sha256_missing:"
                + ",".join(missing_hashes)
            )
            continue
        summaries.append(
            {
                "index": index,
                "source_path": source_path if isinstance(source_path, str) else None,
                "audit_passed": audit.get("passed") is True,
                "audit_verdict": audit.get("verdict"),
                "identity_residual": audit.get("identity_residual"),
                "cache_identity": {
                    "archive_sha256": (
                        cache_archive_sha256
                        if isinstance(cache_archive_sha256, str)
                        else None
                    ),
                    "inflated_outputs_aggregate_sha256": (
                        cache_inflated_sha256
                        if isinstance(cache_inflated_sha256, str)
                        else None
                    ),
                    "raw_sha256": (
                        cache_raw_sha256 if isinstance(cache_raw_sha256, str) else None
                    ),
                    "pair_count": _as_int(cache.get("pair_count")),
                    "candidate_cache_array_sha256": (
                        dict(cache_array_sha256)
                        if isinstance(cache_array_sha256, dict)
                        else None
                    ),
                },
                "auth_identity": {
                    "archive_sha256": auth_archive_sha256,
                    "inflated_outputs_aggregate_sha256": auth_inflated_sha256,
                    "raw_sha256": auth_raw_sha256,
                    "pair_count": _as_int(auth_eval.get("n_samples")),
                    "candidate_cache_array_sha256": dict(auth_array_sha256),
                    "hash_domain": auth_eval.get("scorer_input_hash_domain"),
                },
            }
        )
    return summaries, blockers


def _cache_auth_audit_coverage_for_group(
    group: dict[str, Any],
    audit_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    same_archive = [
        summary
        for summary in audit_summaries
        if summary.get("auth_identity", {}).get("archive_sha256")
        == group.get("archive_sha256")
        or summary.get("cache_identity", {}).get("archive_sha256")
        == group.get("archive_sha256")
    ]
    if not same_archive:
        return {
            "status": "not_supplied_for_archive",
            "matched": False,
            "source_path": None,
            "blockers": [],
            "mismatches": [],
        }

    best_mismatches: list[str] | None = None
    best_summary: dict[str, Any] | None = None
    for summary in same_archive:
        auth_identity = summary.get("auth_identity")
        if not isinstance(auth_identity, dict):
            continue
        mismatches: list[str] = []
        if (
            group.get("inflated_outputs_aggregate_sha256")
            and auth_identity.get("inflated_outputs_aggregate_sha256")
            and group.get("inflated_outputs_aggregate_sha256")
            != auth_identity.get("inflated_outputs_aggregate_sha256")
        ):
            mismatches.append("inflated_outputs_aggregate_sha256")
        if (
            group.get("raw_sha256")
            and auth_identity.get("raw_sha256")
            and group.get("raw_sha256") != auth_identity.get("raw_sha256")
        ):
            mismatches.append("raw_sha256")
        group_arrays = group.get("candidate_cache_array_sha256")
        audit_arrays = auth_identity.get("candidate_cache_array_sha256")
        if isinstance(group_arrays, dict) and isinstance(audit_arrays, dict):
            for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"):
                if (
                    group_arrays.get(key)
                    and audit_arrays.get(key)
                    and group_arrays.get(key) != audit_arrays.get(key)
                ):
                    mismatches.append(f"candidate_cache_array_sha256.{key}")
        if not mismatches:
            return {
                "status": "matched_same_archive_identity",
                "matched": True,
                "source_path": summary.get("source_path"),
                "audit_index": summary.get("index"),
                "audit_passed": summary.get("audit_passed"),
                "audit_verdict": summary.get("audit_verdict"),
                "blockers": [],
                "mismatches": [],
                "audit_identity": {
                    "archive_sha256": auth_identity.get("archive_sha256"),
                    "inflated_outputs_aggregate_sha256": auth_identity.get(
                        "inflated_outputs_aggregate_sha256"
                    ),
                    "raw_sha256": auth_identity.get("raw_sha256"),
                    "candidate_cache_array_sha256": auth_identity.get(
                        "candidate_cache_array_sha256"
                    ),
                    "pair_count": auth_identity.get("pair_count"),
                },
            }
        if best_mismatches is None or len(mismatches) < len(best_mismatches):
            best_mismatches = mismatches
            best_summary = summary

    mismatches = best_mismatches or []
    blockers = []
    for field in mismatches:
        if field.startswith("candidate_cache_array_sha256."):
            blockers.append(
                "mlx_parent_contract_auth_cache_candidate_array_sha256_mismatch:"
                f"{group['group_id']}:{field.rsplit('.', 1)[-1]}"
            )
        else:
            blockers.append(
                "mlx_parent_contract_auth_cache_auth_identity_mismatch:"
                f"{group['group_id']}:{field}"
            )
    best_auth_identity = (
        best_summary.get("auth_identity")
        if isinstance(best_summary, dict)
        and isinstance(best_summary.get("auth_identity"), dict)
        else {}
    )
    return {
        "status": "mismatched_same_archive_identity",
        "matched": False,
        "source_path": None if best_summary is None else best_summary.get("source_path"),
        "audit_index": None if best_summary is None else best_summary.get("index"),
        "audit_passed": None if best_summary is None else best_summary.get("audit_passed"),
        "audit_verdict": None if best_summary is None else best_summary.get("audit_verdict"),
        "blockers": blockers,
        "mismatches": mismatches,
        "audit_identity": {
            "archive_sha256": best_auth_identity.get("archive_sha256"),
            "inflated_outputs_aggregate_sha256": best_auth_identity.get(
                "inflated_outputs_aggregate_sha256"
            ),
            "raw_sha256": best_auth_identity.get("raw_sha256"),
            "candidate_cache_array_sha256": best_auth_identity.get(
                "candidate_cache_array_sha256"
            ),
            "pair_count": best_auth_identity.get("pair_count"),
        },
    }


def _source_parent_response_candidates_for_group(
    group: dict[str, Any],
    *,
    repo_root: Path | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    if repo_root is None:
        return [], []
    source_paths = group.get("source_paths_sample")
    if not isinstance(source_paths, list):
        return [], []
    candidates: list[str] = []
    probes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in source_paths:
        if not isinstance(source, str) or not source:
            continue
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        search_root = source_path.parent.parent if source_path.parent.name == "candidate_windows" else source_path.parent
        if not search_root.exists():
            continue
        for path in sorted(search_root.glob("*parent*.json")):
            try:
                payload = _load_json(path)
            except (OSError, json.JSONDecodeError, ScorerResponseDatasetError):
                continue
            if payload.get("schema_version") != MLX_SCORER_RESPONSE_SCHEMA:
                continue
            summary = {
                "archive_sha256": payload.get("archive_sha256"),
                "inflated_outputs_aggregate_sha256": (
                    payload.get("inflated_outputs_aggregate_sha256")
                    or _get_path(
                        payload,
                        (
                            "cache_identity",
                            "candidate",
                            "inflated_outputs_aggregate_sha256",
                        ),
                    )
                ),
                "batch_pairs": _as_int(payload.get("batch_pairs")),
                "pair_window": payload.get("pair_window"),
                "candidate_cache_array_sha256": _get_path(
                    payload,
                    ("cache_identity", "candidate", "array_sha256"),
                ),
                "reference_cache_array_sha256": _get_path(
                    payload,
                    ("cache_identity", "reference", "array_sha256"),
                ),
            }
            if not _contract_summary_covers_parent_group(summary, group):
                continue
            display = path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
            text = str(display)
            if text not in seen:
                seen.add(text)
                candidates.append(text)
                if len(probes) < 8:
                    probe = build_mlx_scorer_production_contract_manifest(
                        payload,
                        run_id=f"parent_contract_probe:{text}",
                        require_cache_identity=True,
                        require_cache_auth_audit=True,
                        require_torch_parity=True,
                        require_profile_stability=True,
                        require_batch_invariance=True,
                        require_score_calibration=True,
                    )
                    probes.append(
                        {
                            "path": text,
                            "passed": probe.get("passed") is True,
                            "verdict": probe.get("verdict"),
                            "blockers": list(probe.get("blockers") or []),
                            "warnings": list(probe.get("warnings") or []),
                        }
                    )
    return candidates[:8], probes[:8]


def _strict_contract_summaries_from_payload(
    contract: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[str]]:
    if contract is None:
        return [], None, []
    schema = contract.get("schema_version") or contract.get("schema")
    strict_summaries: list[dict[str, Any]] = []
    blockers: list[str] = []
    source_gate: dict[str, Any] | None = None
    if schema == MLX_PRODUCTION_CONTRACT_BUNDLE_SCHEMA:
        try:
            source_gate = build_mlx_production_contract_gate(contract)
        except ScorerResponseDatasetError as exc:
            return [], None, [f"production_contract_bundle_invalid:{exc}"]
        contracts = contract.get("contracts")
        if not isinstance(contracts, list):
            return [], source_gate, ["production_contract_bundle_contracts_missing"]
        for index, child in enumerate(contracts):
            if not isinstance(child, dict):
                blockers.append(f"production_contract_bundle_child_not_object:{index}")
                continue
            try:
                gate = build_mlx_production_contract_gate(child)
            except ScorerResponseDatasetError as exc:
                blockers.append(f"production_contract_bundle_child_invalid:{index}:{exc}")
                continue
            if gate.get("status") != "strict_pass":
                continue
            summary = gate.get("summary")
            if isinstance(summary, dict):
                strict_summaries.append(
                    {
                        "contract_index": index,
                        "source_run_id": gate.get("source_run_id"),
                        "source_verdict": gate.get("source_verdict"),
                        "summary": summary,
                    }
                )
        return strict_summaries, source_gate, blockers
    if schema == MLX_PRODUCTION_CONTRACT_SCHEMA:
        try:
            source_gate = build_mlx_production_contract_gate(contract)
        except ScorerResponseDatasetError as exc:
            return [], None, [f"production_contract_invalid:{exc}"]
        if source_gate.get("status") == "strict_pass":
            summary = source_gate.get("summary")
            if isinstance(summary, dict):
                strict_summaries.append(
                    {
                        "contract_index": 0,
                        "source_run_id": source_gate.get("source_run_id"),
                        "source_verdict": source_gate.get("source_verdict"),
                        "summary": summary,
                    }
                )
        return strict_summaries, source_gate, blockers
    return [], None, ["production_contract_schema_mismatch"]


def build_mlx_parent_production_contract_plan(
    dataset: dict[str, Any],
    *,
    production_contract: dict[str, Any] | None = None,
    cache_auth_audits: Iterable[dict[str, Any]] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Plan the strict parent-window MLX contracts needed by a dataset.

    The plan is non-authoritative. It only inventories which parent response
    contracts are required before MLX rows may be used for exact-eval spend
    triage through the effective gate.
    """

    normalized = normalize_legacy_response_dataset_authority(dataset)
    rows = normalized.get("rows")
    if not isinstance(rows, list):
        raise ScorerResponseDatasetError("dataset rows[] missing")
    mlx_rows = [row for row in rows if _is_mlx_scorer_response_row(row)]
    strict_contract_summaries, source_gate, source_blockers = (
        _strict_contract_summaries_from_payload(production_contract)
    )
    cache_auth_audit_summaries, cache_auth_audit_blockers = (
        _cache_auth_audit_summaries(cache_auth_audits)
    )

    groups_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    blockers: list[str] = [*source_blockers, *cache_auth_audit_blockers]
    for row in mlx_rows:
        row_id = str(row.get("row_id") or "<unknown>")
        key = _mlx_parent_contract_group_key(row)
        window = _valid_pair_window(row.get("source_pair_window"))
        if key is None:
            blockers.append(f"mlx_parent_contract_row_identity_missing:{row_id}")
            continue
        if window is None:
            blockers.append(f"mlx_parent_contract_row_pair_window_invalid:{row_id}")
            continue
        group = groups_by_key.get(key)
        if group is None:
            group_id = hashlib.sha256(_canonical_json(key).encode("utf-8")).hexdigest()[:16]
            group = {
                "group_id": f"mlx_parent_contract_{group_id}",
                "status": "blocked_missing_strict_parent_contract",
                "archive_sha256": row.get("archive_sha256"),
                "inflated_outputs_aggregate_sha256": row.get(
                    "source_inflated_outputs_aggregate_sha256"
                ),
                "raw_sha256": row.get("raw_sha256"),
                "source_batch_pairs": _as_int(row.get("source_batch_pairs")),
                "candidate_cache_array_sha256": row.get(
                    "source_candidate_cache_array_sha256"
                ),
                "reference_cache_array_sha256": row.get(
                    "source_reference_cache_array_sha256"
                ),
                "required_parent_pair_window": [window[0], window[1]],
                "row_count": 0,
                "families": {},
                "row_ids_sample": [],
                "source_paths_sample": [],
                "blockers": [],
                "coverage": {
                    "covered_by_supplied_contract": False,
                    "contract_index": None,
                    "source_run_id": None,
                },
            }
            groups_by_key[key] = group
        row_raw_sha = row.get("raw_sha256")
        if isinstance(row_raw_sha, str) and row_raw_sha:
            if not group.get("raw_sha256"):
                group["raw_sha256"] = row_raw_sha
            elif group.get("raw_sha256") != row_raw_sha:
                blocker = f"mlx_parent_contract_row_raw_sha256_mismatch:{row_id}"
                group["blockers"].append(blocker)
                blockers.append(blocker)
        required_window = group["required_parent_pair_window"]
        required_window[0] = min(required_window[0], window[0])
        required_window[1] = max(required_window[1], window[1])
        group["row_count"] += 1
        family = str(row.get("family") or row.get("source_schema") or "unknown")
        group["families"][family] = group["families"].get(family, 0) + 1
        if len(group["row_ids_sample"]) < 8:
            group["row_ids_sample"].append(row_id)
        source_path = row.get("source_path")
        if (
            isinstance(source_path, str)
            and source_path
            and len(group["source_paths_sample"]) < 8
            and source_path not in group["source_paths_sample"]
        ):
            group["source_paths_sample"].append(source_path)

    groups = sorted(groups_by_key.values(), key=lambda item: item["group_id"])
    for group in groups:
        cache_auth_coverage = _cache_auth_audit_coverage_for_group(
            group,
            cache_auth_audit_summaries,
        )
        group["cache_auth_audit_coverage"] = cache_auth_coverage
        for blocker in cache_auth_coverage.get("blockers") or []:
            group["blockers"].append(blocker)
            blockers.append(blocker)
        parent_candidates, parent_contract_probes = (
            _source_parent_response_candidates_for_group(group, repo_root=repo_root)
        )
        group["source_parent_response_candidates"] = parent_candidates
        group["source_parent_response_contract_probes"] = parent_contract_probes
        matching_contract = next(
            (
                item
                for item in strict_contract_summaries
                if _contract_summary_covers_parent_group(item["summary"], group)
            ),
            None,
        )
        if matching_contract is None:
            blocker = f"mlx_parent_contract_group_uncovered:{group['group_id']}"
            group["blockers"].append(blocker)
            blockers.append(blocker)
            continue
        group["status"] = "covered_by_supplied_contract"
        group["coverage"] = {
            "covered_by_supplied_contract": True,
            "contract_index": matching_contract["contract_index"],
            "source_run_id": matching_contract["source_run_id"],
        }

    covered_group_count = sum(
        1 for group in groups if group["status"] == "covered_by_supplied_contract"
    )
    auth_audit_matched_group_count = sum(
        1
        for group in groups
        if group.get("cache_auth_audit_coverage", {}).get("matched") is True
    )
    auth_audit_mismatched_group_count = sum(
        1
        for group in groups
        if group.get("cache_auth_audit_coverage", {}).get("status")
        == "mismatched_same_archive_identity"
    )
    auth_audit_missing_group_count = sum(
        1
        for group in groups
        if group.get("cache_auth_audit_coverage", {}).get("status")
        == "not_supplied_for_archive"
    )
    status = "strict_pass" if groups and not blockers else "blocked"
    if not mlx_rows:
        blockers.append("no_mlx_scorer_response_rows")
        status = "blocked"
    source_gate_summary = (
        source_gate.get("summary")
        if isinstance(source_gate, dict) and isinstance(source_gate.get("summary"), dict)
        else {}
    )
    return {
        "schema": MLX_PARENT_PRODUCTION_CONTRACT_PLAN_SCHEMA,
        "producer": TOOL,
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
        "status": status,
        "mlx_exact_eval_spend_triage_allowed": False,
        "authority_status": (
            "Planner output is an inventory only; MLX rows remain blocked until "
            "the effective MLX spend-triage gate receives strict production "
            "contract coverage and exact eval remains required before promotion."
        ),
        "summary": {
            "mlx_row_count": len(mlx_rows),
            "required_parent_contract_count": len(groups),
            "covered_parent_contract_group_count": covered_group_count,
            "missing_parent_contract_group_count": len(groups) - covered_group_count,
            "strict_supplied_contract_count": len(strict_contract_summaries),
            "production_contract_gate_status": (
                None if source_gate is None else source_gate.get("status")
            ),
            "production_contract_gate_source_schema": (
                None if source_gate is None else source_gate.get("source_schema")
            ),
            "production_contract_gate_row_count": source_gate_summary.get("row_count"),
            "production_contract_gate_matched_row_count": source_gate_summary.get(
                "matched_row_count"
            ),
            "production_contract_gate_unmatched_row_count": source_gate_summary.get(
                "unmatched_row_count"
            ),
            "cache_auth_audit_count": len(cache_auth_audit_summaries),
            "cache_auth_audit_matched_group_count": auth_audit_matched_group_count,
            "cache_auth_audit_mismatched_group_count": auth_audit_mismatched_group_count,
            "cache_auth_audit_missing_group_count": auth_audit_missing_group_count,
        },
        "required_parent_contracts": groups,
        "blockers": blockers,
        "allowed_use": (
            "contract_build_queue_planning_only"
            if status == "blocked"
            else "contract_coverage_inventory_only_use_effective_gate_for_spend_triage"
        ),
    }


def render_mlx_parent_production_contract_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# MLX Parent Production Contract Plan",
        "",
        "## Authority",
        "",
        "- Score claim: `False`",
        "- Promotion eligible: `False`",
        "- Ready for exact-eval dispatch: `False`",
        "- Rank/kill eligible: `False`",
        "- Spend triage authority: `False`",
        "",
        "## Summary",
        "",
    ]
    summary = plan.get("summary") if isinstance(plan.get("summary"), dict) else {}
    lines.extend(
        [
            f"- Status: `{plan.get('status')}`",
            f"- MLX rows: `{summary.get('mlx_row_count')}`",
            f"- Required parent contracts: `{summary.get('required_parent_contract_count')}`",
            f"- Covered parent groups: `{summary.get('covered_parent_contract_group_count')}`",
            f"- Missing parent groups: `{summary.get('missing_parent_contract_group_count')}`",
            f"- Strict supplied contracts: `{summary.get('strict_supplied_contract_count')}`",
            f"- Production contract gate: `{summary.get('production_contract_gate_status')}`",
            f"- Cache/auth audits supplied: `{summary.get('cache_auth_audit_count')}`",
            "- Cache/auth matched groups: "
            f"`{summary.get('cache_auth_audit_matched_group_count')}`",
            "- Cache/auth mismatched groups: "
            f"`{summary.get('cache_auth_audit_mismatched_group_count')}`",
            "- Cache/auth missing groups: "
            f"`{summary.get('cache_auth_audit_missing_group_count')}`",
            f"- Blockers: `{plan.get('blockers')}`",
            "",
            "## Required Parent Contracts",
            "",
        ]
    )
    for group in plan.get("required_parent_contracts", []):
        lines.extend(
            [
                f"### {group.get('group_id')}",
                "",
                f"- Status: `{group.get('status')}`",
                f"- Rows: `{group.get('row_count')}`",
                f"- Families: `{group.get('families')}`",
                f"- Pair window required: `{group.get('required_parent_pair_window')}`",
                f"- Archive SHA-256: `{group.get('archive_sha256')}`",
                "- Inflated aggregate SHA-256: "
                f"`{group.get('inflated_outputs_aggregate_sha256')}`",
                f"- Raw SHA-256: `{group.get('raw_sha256')}`",
                "- Candidate cache arrays: "
                f"`{group.get('candidate_cache_array_sha256')}`",
                "- Reference cache arrays: "
                f"`{group.get('reference_cache_array_sha256')}`",
                "- Cache/auth audit coverage: "
                f"`{group.get('cache_auth_audit_coverage')}`",
                f"- Parent response candidates: `{group.get('source_parent_response_candidates')}`",
                "- Parent response contract probes: "
                f"`{group.get('source_parent_response_contract_probes')}`",
                f"- Row sample: `{group.get('row_ids_sample')}`",
                f"- Blockers: `{group.get('blockers')}`",
                "",
            ]
        )
    return "\n".join(lines)


def build_effective_mlx_spend_triage_gate(
    *,
    mlx_rows: list[dict[str, Any]],
    response_validation_gate: dict[str, Any],
    mlx_torch_parity_sweep_gate: dict[str, Any] | None,
    mlx_score_calibration_gate: dict[str, Any] | None,
    mlx_production_contract_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compose MLX constituent gates into one spend-triage decision.

    Constituent MLX gates can be useful individually, but downstream automation
    should consume this effective gate when deciding whether MLX rows may filter
    exact-eval spend. A pass still creates no score, rank, promotion, or dispatch
    authority.
    """

    def _summary_from(gate: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(gate, dict):
            return {}
        summary = gate.get("summary")
        return summary if isinstance(summary, dict) else {}

    def _sample_list(gate: dict[str, Any] | None, key: str, limit: int = 8) -> list[Any]:
        if not isinstance(gate, dict):
            return []
        value = gate.get(key)
        if not isinstance(value, list):
            return []
        return list(value[:limit])

    def _summary_list(summary: dict[str, Any], key: str, limit: int = 8) -> list[Any]:
        value = summary.get(key)
        if not isinstance(value, list):
            return []
        return list(value[:limit])

    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    blockers: list[str] = []
    if not mlx_rows:
        blockers.append("no_mlx_rows")
    if response_validation_gate.get("passed") is not True:
        blockers.append("response_validation_gate_not_passed")
    if response_validation_gate.get("prediction_spend_triage_usable") is not True:
        blockers.append("response_validation_prediction_spend_triage_not_usable")
    required_spend_triage_families = _string_list(
        response_validation_gate.get("required_spend_triage_families")
    )
    spend_triage_allowed_families = _string_list(
        response_validation_gate.get("spend_triage_allowed_families")
    )
    spend_triage_blocked_families = _string_list(
        response_validation_gate.get("spend_triage_blocked_families")
    )
    if (
        required_spend_triage_families
        and response_validation_gate.get("required_family_spend_triage_passed") is not True
    ):
        blockers.append("response_validation_required_family_spend_triage_not_passed")
    if not spend_triage_allowed_families:
        blockers.append("response_validation_no_family_spend_triage_allowed")
    mlx_families = sorted(
        {
            str(row.get("family"))
            for row in mlx_rows
            if isinstance(row, dict) and row.get("family") is not None
        }
    )
    mlx_families_without_gate = sorted(
        family
        for family in mlx_families
        if family not in set(spend_triage_allowed_families)
    )
    if mlx_families_without_gate:
        blockers.append("mlx_rows_include_family_without_spend_triage_gate")
    if (
        mlx_torch_parity_sweep_gate is None
        or mlx_torch_parity_sweep_gate.get("status") != "strict_pass"
        or mlx_torch_parity_sweep_gate.get("mlx_rows_allowed_for_planner") is not True
    ):
        blockers.append("mlx_torch_parity_sweep_gate_not_strict_pass")
    if (
        mlx_score_calibration_gate is None
        or mlx_score_calibration_gate.get("status") != "strict_pass"
        or mlx_score_calibration_gate.get("mlx_spend_triage_allowed") is not True
    ):
        blockers.append("mlx_score_calibration_gate_not_strict_pass")
    if (
        mlx_production_contract_gate is None
        or mlx_production_contract_gate.get("status") != "strict_pass"
        or mlx_production_contract_gate.get("mlx_spend_triage_allowed") is not True
    ):
        blockers.append("mlx_production_contract_gate_not_strict_pass")

    production_summary = _summary_from(mlx_production_contract_gate)
    production_unmatched_row_ids = _summary_list(
        production_summary, "unmatched_row_ids"
    )
    production_row_count = production_summary.get("row_count")
    production_matched_row_count = production_summary.get("matched_row_count")
    production_unmatched_row_count = production_summary.get("unmatched_row_count")
    if production_unmatched_row_count is None:
        production_unmatched_row_count = len(production_unmatched_row_ids)
    if (
        production_row_count is None
        and isinstance(mlx_production_contract_gate, dict)
        and mlx_production_contract_gate.get("status") == "strict_pass"
    ):
        production_row_count = len(mlx_rows)
        production_matched_row_count = len(mlx_rows)
    status = "strict_pass" if not blockers else "blocked"
    return {
        "schema": "ll_effective_mlx_spend_triage_gate.v1",
        "producer": TOOL,
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
        "family_spend_triage_gate_enforced": True,
        "required_spend_triage_families": required_spend_triage_families,
        "spend_triage_allowed_families": spend_triage_allowed_families,
        "spend_triage_blocked_families": spend_triage_blocked_families,
        "mlx_families": mlx_families,
        "mlx_families_without_spend_triage_gate": mlx_families_without_gate,
        "status": status,
        "mlx_exact_eval_spend_triage_allowed": status == "strict_pass",
        "blockers": blockers,
        "input_rows": [row.get("row_id") for row in mlx_rows][:8],
        "summary": {
            "mlx_row_count": len(mlx_rows),
            "response_validation_status": response_validation_gate.get("status"),
            "response_prediction_spend_triage_usable": response_validation_gate.get(
                "prediction_spend_triage_usable"
            ),
            "required_spend_triage_families": required_spend_triage_families,
            "required_family_spend_triage_passed": response_validation_gate.get(
                "required_family_spend_triage_passed"
            ),
            "required_family_spend_triage_blockers": _string_list(
                response_validation_gate.get("required_family_spend_triage_blockers")
            ),
            "spend_triage_allowed_families": spend_triage_allowed_families,
            "spend_triage_blocked_families": spend_triage_blocked_families,
            "mlx_families": mlx_families,
            "mlx_families_without_spend_triage_gate": mlx_families_without_gate,
            "torch_parity_status": (
                None
                if mlx_torch_parity_sweep_gate is None
                else mlx_torch_parity_sweep_gate.get("status")
            ),
            "score_calibration_status": (
                None
                if mlx_score_calibration_gate is None
                else mlx_score_calibration_gate.get("status")
            ),
            "production_contract_status": (
                None
                if mlx_production_contract_gate is None
                else mlx_production_contract_gate.get("status")
            ),
            "response_validation_blockers_sample": _sample_list(
                response_validation_gate, "blockers"
            ),
            "torch_parity_blockers_sample": _sample_list(
                mlx_torch_parity_sweep_gate, "blockers"
            ),
            "score_calibration_blockers_sample": _sample_list(
                mlx_score_calibration_gate, "blockers"
            ),
            "production_contract_blockers_sample": _sample_list(
                mlx_production_contract_gate, "blockers"
            ),
            "production_contract_row_count": production_row_count,
            "production_contract_matched_row_count": production_matched_row_count,
            "production_contract_parent_window_matched_row_count": (
                production_summary.get("parent_window_matched_row_count")
            ),
            "production_contract_unmatched_row_count": production_unmatched_row_count,
            "production_contract_unmatched_row_ids_sample": (
                production_unmatched_row_ids
            ),
            "production_contract_strict_contract_count": production_summary.get(
                "strict_contract_count"
            ),
        },
        "allowed_use": (
            "local_exact_eval_spend_triage_filter_after_all_mlx_and_dataset_gates"
            if status == "strict_pass"
            else "blocked_until_response_dataset_and_all_mlx_gates_strict_pass"
        ),
    }


def build_scorer_response_validation_gate(
    dataset: dict[str, Any],
    *,
    min_rows: int = 50,
    min_families: int = 2,
    required_folds: Iterable[int] = range(5),
    required_spend_triage_families: Iterable[str] = (),
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
    required_families = sorted(
        {
            family.strip()
            for family in (str(value) for value in required_spend_triage_families)
            if family.strip()
        }
    )

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
    spend_triage_usable_predictions = [
        item
        for item in passing_predictions
        if item.get("candidate_family_spend_triage_usable") is True
    ]
    family_spend_triage_gates = _family_spend_triage_gates(
        family_counts=family_counts,
        prediction_evaluations=prediction_evaluations,
        passing_predictions=passing_predictions,
    )
    spend_triage_usable_families = [
        family
        for family, gate in family_spend_triage_gates.items()
        if gate.get("spend_triage_usable") is True
    ]
    spend_triage_blocked_families = [
        family
        for family, gate in family_spend_triage_gates.items()
        if gate.get("spend_triage_usable") is not True
    ]
    required_family_spend_triage_blockers: list[str] = []
    for family in required_families:
        family_gate = family_spend_triage_gates.get(family)
        if family_gate is None:
            required_family_spend_triage_blockers.append(
                f"required_family_oof_gate_missing:{family}"
            )
        elif family_gate.get("spend_triage_usable") is not True:
            required_family_spend_triage_blockers.append(
                f"required_family_oof_gate_blocked:{family}"
            )
    required_family_spend_triage_passed = not required_family_spend_triage_blockers

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
    authority = normalized.get("authority") if isinstance(normalized.get("authority"), dict) else {}
    return {
        "schema": VALIDATION_GATE_SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "evidence_grade": authority.get("evidence_grade"),
        "evidence_tag": authority.get("evidence_tag"),
        "score_axis": authority.get("score_axis") or authority.get("evidence_tag"),
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "status": status,
        "passed": status == "passed",
        "prediction_spend_triage_usable": (
            status == "passed" and bool(spend_triage_usable_predictions)
            and required_family_spend_triage_passed
        ),
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
            "required_spend_triage_families": required_families,
            "target": target,
            "prediction_fields": [str(field) for field in prediction_fields],
            "min_prediction_pairs_per_fold": int(min_prediction_pairs_per_fold),
            "min_pearson_r": float(min_pearson_r),
            "require_single_axis": bool(require_single_axis),
            "candidate_family_min_pearson_r": (
                PREDICTION_CANDIDATE_FAMILY_MIN_PEARSON_R
            ),
            "candidate_family_min_top_k_overlap": (
                PREDICTION_CANDIDATE_FAMILY_MIN_TOP_K_OVERLAP
            ),
            "candidate_family_min_negative_predictions": (
                PREDICTION_CANDIDATE_FAMILY_MIN_NEGATIVE_PREDICTIONS
            ),
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
            "spend_triage_usable_families": spend_triage_usable_families,
            "spend_triage_blocked_families": spend_triage_blocked_families,
        },
        "prediction_evaluations": prediction_evaluations,
        "family_spend_triage_gates": family_spend_triage_gates,
        "required_spend_triage_families": required_families,
        "required_family_spend_triage_passed": required_family_spend_triage_passed,
        "required_family_spend_triage_blockers": (
            required_family_spend_triage_blockers
        ),
        "spend_triage_allowed_families": (
            required_families
            if required_families and required_family_spend_triage_passed
            else spend_triage_usable_families
        ),
        "spend_triage_usable_families": spend_triage_usable_families,
        "spend_triage_blocked_families": spend_triage_blocked_families,
        "passing_prediction_fields": [item["field"] for item in passing_predictions],
        "spend_triage_usable_prediction_fields": [
            item["field"] for item in spend_triage_usable_predictions
        ],
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
    family_metrics = _candidate_family_prediction_metrics(
        rows=rows,
        field=field,
        target=target,
    )
    candidate_family_spend_triage_usable = any(
        metric.get("spend_triage_usable") is True for metric in family_metrics.values()
    )
    return {
        "field": field,
        "target": target,
        "present_pair_count": len(xs_all),
        "overall_pearson_r": overall,
        "folds": per_fold,
        "candidate_family_metrics": family_metrics,
        "candidate_family_spend_triage_usable": candidate_family_spend_triage_usable,
        "passed": bool(per_fold) and all(item["passed"] for item in per_fold),
    }


def _family_spend_triage_gates(
    *,
    family_counts: dict[str, int],
    prediction_evaluations: list[dict[str, Any]],
    passing_predictions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    passing_fields = {str(item.get("field")) for item in passing_predictions}
    out: dict[str, dict[str, Any]] = {}
    for family in sorted(family_counts):
        usable_fields: list[str] = []
        blocked_fields: list[str] = []
        field_metrics: dict[str, dict[str, Any]] = {}
        for evaluation in prediction_evaluations:
            field = str(evaluation.get("field"))
            metrics_by_family = evaluation.get("candidate_family_metrics")
            metrics = (
                metrics_by_family.get(family)
                if isinstance(metrics_by_family, dict)
                else None
            )
            if not isinstance(metrics, dict):
                continue
            field_metrics[field] = metrics
            if metrics.get("spend_triage_usable") is True and field in passing_fields:
                usable_fields.append(field)
            else:
                blocked_fields.append(field)
        blockers: list[str] = []
        if not field_metrics:
            blockers.append("family_has_no_prediction_metrics")
        if not usable_fields:
            blockers.append("family_has_no_spend_triage_usable_prediction_field")
        out[family] = {
            "schema": "scorer_response_family_spend_triage_gate.v1",
            "family": family,
            "status": "strict_pass" if not blockers else "blocked",
            "spend_triage_usable": not blockers,
            "usable_prediction_fields": usable_fields,
            "blocked_prediction_fields": blocked_fields,
            "blockers": blockers,
            "field_metrics": field_metrics,
            "allowed_use": (
                "family_local_spend_triage_candidate_generation"
                if not blockers
                else "blocked_until_family_level_oof_metrics_pass"
            ),
            "not_allowed_uses": [
                "score_claim",
                "promotion",
                "rank_or_kill",
                "exact_eval_dispatch_selection_without_family_gate",
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "promotable": False,
        }
    return out


def _candidate_family_prediction_metrics(
    *,
    rows: list[dict[str, Any]],
    field: str,
    target: str,
) -> dict[str, dict[str, Any]]:
    families = sorted({str(row.get("family") or "unknown") for row in rows})
    out: dict[str, dict[str, Any]] = {}
    for family in families:
        family_rows = [
            row for row in rows if str(row.get("family") or "unknown") == family
        ]
        xs_all: list[float] = []
        ys_all: list[float] = []
        fold_metrics: list[dict[str, Any]] = []
        folds = sorted(
            {
                fold
                for row in family_rows
                if (fold := _as_int(row.get("holdout_fold"))) is not None
            }
        )
        for fold in folds:
            xs: list[float] = []
            ys: list[float] = []
            for row in family_rows:
                if _as_int(row.get("holdout_fold")) != fold:
                    continue
                pred = _as_float(row.get(field))
                observed = _as_float(row.get(target))
                if pred is None or observed is None:
                    continue
                xs.append(pred)
                ys.append(observed)
            fold_metrics.append(
                {
                    "fold": fold,
                    "n": len(xs),
                    "pearson_r": _pearson(xs, ys),
                }
            )
            xs_all.extend(xs)
            ys_all.extend(ys)
        overall = _pearson(xs_all, ys_all)
        negative_prediction_count = sum(1 for value in xs_all if value < 0.0)
        observed_improvement_count = sum(1 for value in ys_all if value < 0.0)
        top_k = _candidate_family_top_k_metrics(
            family_rows,
            field=field,
            target=target,
        )
        top8 = top_k.get("8", {})
        top8_mean = _as_float(top8.get("mean_observed_delta_in_predicted_top_k"))
        top8_overlap = int(top8.get("overlap_count") or 0)
        spend_triage_usable = (
            len(xs_all) >= max(PREDICTION_CANDIDATE_FAMILY_TOP_K[0], 3)
            and overall is not None
            and overall >= PREDICTION_CANDIDATE_FAMILY_MIN_PEARSON_R
            and negative_prediction_count
            >= PREDICTION_CANDIDATE_FAMILY_MIN_NEGATIVE_PREDICTIONS
            and observed_improvement_count > 0
            and top8_overlap >= PREDICTION_CANDIDATE_FAMILY_MIN_TOP_K_OVERLAP
            and top8_mean is not None
            and top8_mean < 0.0
        )
        out[family] = {
            "family": family,
            "n": len(xs_all),
            "overall_pearson_r": overall,
            "folds": fold_metrics,
            "negative_prediction_count": negative_prediction_count,
            "observed_improvement_count": observed_improvement_count,
            "top_k": top_k,
            "spend_triage_usable": spend_triage_usable,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "promotable": False,
        }
    return out


def _candidate_family_top_k_metrics(
    rows: list[dict[str, Any]],
    *,
    field: str,
    target: str,
) -> dict[str, dict[str, Any]]:
    pairs: list[tuple[str, float, float]] = []
    for row in rows:
        pred = _as_float(row.get(field))
        observed = _as_float(row.get(target))
        if pred is None or observed is None:
            continue
        pairs.append((str(row.get("row_id")), pred, observed))
    by_prediction = sorted(pairs, key=lambda item: (item[1], item[0]))
    by_observed = sorted(pairs, key=lambda item: (item[2], item[0]))
    out: dict[str, dict[str, Any]] = {}
    for k in PREDICTION_CANDIDATE_FAMILY_TOP_K:
        limit = min(k, len(pairs))
        predicted_ids = {item[0] for item in by_prediction[:limit]}
        observed_ids = {item[0] for item in by_observed[:limit]}
        observed_in_predicted = [item[2] for item in by_prediction[:limit]]
        out[str(k)] = {
            "k": k,
            "effective_k": limit,
            "overlap_count": len(predicted_ids & observed_ids),
            "overlap_fraction": (
                None if limit == 0 else len(predicted_ids & observed_ids) / limit
            ),
            "mean_observed_delta_in_predicted_top_k": _mean(
                observed_in_predicted
            ),
            "best_observed_delta_in_predicted_top_k": (
                None if not observed_in_predicted else min(observed_in_predicted)
            ),
        }
    return out


def _blocked_response_validation_gate(reason: str, *, row_count: int) -> dict[str, Any]:
    return {
        "schema": VALIDATION_GATE_SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "score_claim_valid": False,
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
    mlx_score_calibration: dict[str, Any] | None = None,
    mlx_production_contract: dict[str, Any] | None = None,
    required_spend_triage_families: Iterable[str] = (),
    decoder_q_response_surface: dict[str, Any] | None = None,
    decoder_q_surface_advisory_batch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic LL next-probe plan from response economics."""

    rows = dataset.get("rows")
    if not isinstance(rows, list):
        raise ScorerResponseDatasetError("dataset rows[] missing")
    dataset_authority = dataset.get("authority") if isinstance(dataset.get("authority"), dict) else {}
    try:
        response_validation_gate = build_scorer_response_validation_gate(
            dataset,
            required_spend_triage_families=required_spend_triage_families,
        )
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
        total_delta = _planning_delta_vs_baseline(row)
        scorer_delta = _planning_scorer_delta_vs_baseline(row)
        margin = _planning_byte_budget_margin(row)
        planning_scope = _planning_scope(row)
        if total_delta is not None and (best_total is None or total_delta < best_total["delta_vs_baseline_score"]):
            best_total = {
                "row_id": row.get("row_id"),
                "family": row.get("family"),
                "delta_vs_baseline_score": total_delta,
                "planning_value_scope": planning_scope,
                "added_archive_bytes": row.get("added_archive_bytes"),
            }
        if scorer_delta is not None and (best_scorer is None or scorer_delta < best_scorer["scorer_delta_vs_baseline"]):
            best_scorer = {
                "row_id": row.get("row_id"),
                "family": row.get("family"),
                "scorer_delta_vs_baseline": scorer_delta,
                "observed_scorer_gain_vs_baseline": _planning_scorer_gain(row),
                "break_even_added_bytes_from_scorer_gain": _planning_break_even_bytes(row),
                "planning_value_scope": planning_scope,
                "added_archive_bytes": row.get("added_archive_bytes"),
            }
        if margin is not None and (best_margin is None or margin > best_margin["byte_budget_margin_vs_break_even"]):
            best_margin = {
                "row_id": row.get("row_id"),
                "family": row.get("family"),
                "byte_budget_margin_vs_break_even": margin,
                "break_even_added_bytes_from_scorer_gain": _planning_break_even_bytes(row),
                "planning_value_scope": planning_scope,
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
    elif response_validation_gate.get("prediction_spend_triage_usable") is not True:
        prohibitions.append(
            {
                "rule": "do_not_use_oof_predictions_for_spend_triage_selection",
                "reason": (
                    "overall held-out response correlation passed, but no "
                    "candidate family had usable spend-triage prediction "
                    "metrics; route selection through observed strict-gated "
                    "MLX rows or add stronger pre-response features"
                ),
                "passing_prediction_fields": response_validation_gate.get(
                    "passing_prediction_fields"
                ),
                "spend_triage_usable_prediction_fields": (
                    response_validation_gate.get(
                        "spend_triage_usable_prediction_fields"
                    )
                ),
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

    mlx_row_items = [
        row
        for row in rows
        if _is_mlx_scorer_response_row(row)
    ]
    mlx_torch_parity_sweep_gate = None
    if mlx_torch_parity_sweep is not None:
        mlx_torch_parity_sweep_gate = build_mlx_torch_parity_sweep_gate(
            mlx_torch_parity_sweep,
            allow_mlx_parity_research_signal_override=(
                allow_mlx_parity_research_signal_override
            ),
        )
    mlx_score_calibration_gate = None
    if mlx_score_calibration is not None:
        mlx_score_calibration_gate = build_mlx_score_calibration_gate(
            mlx_score_calibration
        )
    mlx_production_contract_gate = None
    if mlx_production_contract is not None:
        mlx_production_contract_gate = build_mlx_production_contract_gate(
            mlx_production_contract,
            rows=mlx_row_items,
        )
    effective_mlx_spend_triage_gate = None
    if mlx_row_items:
        effective_mlx_spend_triage_gate = build_effective_mlx_spend_triage_gate(
            mlx_rows=mlx_row_items,
            response_validation_gate=response_validation_gate,
            mlx_torch_parity_sweep_gate=mlx_torch_parity_sweep_gate,
            mlx_score_calibration_gate=mlx_score_calibration_gate,
            mlx_production_contract_gate=mlx_production_contract_gate,
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
    mlx_rows = [row.get("row_id") for row in mlx_row_items][:8]
    if mlx_rows:
        if mlx_production_contract_gate is None:
            prohibitions.append(
                {
                    "rule": "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract",
                    "reason": (
                        "MLX parity and score calibration are constituent checks; "
                        "exact-eval spend filtering also requires a strict MLX "
                        "production contract tying cache identity, reference and "
                        "candidate parity, profile stability, and calibration to "
                        "the same response identity."
                    ),
                    "input_rows": mlx_rows,
                }
            )
        elif not mlx_production_contract_gate["mlx_spend_triage_allowed"]:
            prohibitions.append(
                {
                    "rule": "do_not_use_mlx_rows_for_exact_eval_spend_triage_after_failed_production_contract",
                    "reason": (
                        "the attached MLX production contract is not a strict pass; "
                        "local MLX rows remain non-authoritative research signal "
                        "until the full production gate passes"
                    ),
                    "gate": mlx_production_contract_gate,
                    "input_rows": mlx_rows,
                }
            )
        if mlx_score_calibration_gate is None:
            prohibitions.append(
                {
                    "rule": "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_score_calibration",
                    "reason": (
                        "MLX rows may train local response models, but exact-eval "
                        "spend filtering requires an attached MLX score-calibration "
                        "decision-band manifest"
                    ),
                    "input_rows": mlx_rows,
                }
            )
        elif not mlx_score_calibration_gate["mlx_spend_triage_allowed"]:
            prohibitions.append(
                {
                    "rule": "do_not_use_mlx_rows_for_exact_eval_spend_triage_after_uncertain_calibration",
                    "reason": (
                        "the attached MLX score-calibration manifest has uncertain "
                        "or invalid pairwise decision bands"
                    ),
                    "gate": mlx_score_calibration_gate,
                    "input_rows": mlx_rows,
                }
            )
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
                    "mlx_score_calibration_gate": mlx_score_calibration_gate,
                    "mlx_production_contract_gate": mlx_production_contract_gate,
                    "effective_mlx_spend_triage_gate": effective_mlx_spend_triage_gate,
                    "acceptance_gate": (
                        ">=50 MLX rows across stable CPU windows/families, all "
                        "score_claim=false, with parity-gated held-out "
                        "correlation, calibrated score-gap decision bands, and a "
                        "strict MLX production contract before any exact-eval "
                        "dispatch selection"
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
    decoder_q_surface_advisory_gate = None
    if decoder_q_surface_advisory_batch is not None:
        decoder_q_surface_advisory_gate = build_decoder_q_surface_advisory_gate(
            decoder_q_surface_advisory_batch
        )
    if decoder_q_response_surface is not None:
        decoder_q_response_surface_summary = _validate_decoder_q_response_surface(
            decoder_q_response_surface
        )
        if decoder_q_surface_advisory_gate is None:
            prohibitions.append(
                {
                    "rule": "do_not_dispatch_decoder_q_response_surface_without_advisory_sign_calibration",
                    "reason": (
                        "decoder-q response surfaces are planning priors only; "
                        "exact-eval spend requires an advisory batch proving at "
                        "least one surface-guided fixed-length candidate improves"
                    ),
                    "response_surface_summary": decoder_q_response_surface_summary,
                }
            )
        elif not decoder_q_surface_advisory_gate["decoder_q_surface_exact_eval_allowed"]:
            prohibitions.append(
                {
                    "rule": "do_not_dispatch_decoder_q_response_surface_after_advisory_regression",
                    "reason": (
                        "the attached decoder-q surface advisory batch did not "
                        "produce an improving surface-guided candidate"
                    ),
                    "gate": decoder_q_surface_advisory_gate,
                }
            )
        for probe in probes:
            probe["priority"] = int(probe["priority"]) + 1
        if (
            decoder_q_surface_advisory_gate is not None
            and not decoder_q_surface_advisory_gate["decoder_q_surface_exact_eval_allowed"]
        ):
            probes.insert(
                0,
                {
                    "probe_id": "ll_decoder_q_surface_sign_calibration_repair",
                    "priority": 1,
                    "class": "sign_calibration",
                    "rationale": (
                        "The response surface found high-leverage decoder-q atoms, "
                        "but advisory scoring showed the current suppress/invert "
                        "direction worsens score; learn a signed calibration before "
                        "materializing more exact-eval candidates."
                    ),
                    "input_rows": [],
                    "response_surface_summary": decoder_q_response_surface_summary,
                    "decoder_q_surface_advisory_gate": decoder_q_surface_advisory_gate,
                    "acceptance_gate": (
                        "surface-guided candidates must include at least one "
                        "advisory-improving fixed-length archive before exact CUDA "
                        "dispatch selection"
                    ),
                },
            )
        else:
            probes.insert(
                0,
                {
                    "probe_id": "ll_decoder_q_window_signed_response_surface",
                    "priority": 1,
                    "class": "byte_neutral_representation_mutation",
                    "rationale": (
                        "Matched MLX family deltas show decoder-q response is "
                        "window-signed; preserve improving windows and suppress "
                        "or invert regressing ones, with advisory sign calibration "
                        "required before exact-eval spend."
                    ),
                    "input_rows": [],
                    "response_surface_summary": decoder_q_response_surface_summary,
                    "decoder_q_surface_advisory_gate": decoder_q_surface_advisory_gate,
                    "top_preserve_windows": decoder_q_response_surface.get("top_preserve_windows", [])[:8],
                    "top_suppress_windows": decoder_q_response_surface.get("top_suppress_windows", [])[:8],
                    "acceptance_gate": (
                        "new decoder-q candidate improves matched local response "
                        "surface, remains byte-neutral/fixed-length, passes official "
                        "inflate, and has an advisory-improving sign-calibration "
                        "batch before exact CUDA eval"
                    ),
                },
            )

    return {
        "schema": "ll_scorer_response_next_probe_plan.v1",
        "producer": TOOL,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "evidence_grade": dataset_authority.get("evidence_grade"),
        "evidence_tag": dataset_authority.get("evidence_tag"),
        "score_axis": dataset_authority.get("score_axis") or dataset_authority.get("evidence_tag"),
        "dataset_summary": dataset.get("summary"),
        "best_total_row": best_total,
        "best_scorer_row": best_scorer,
        "best_byte_budget_margin_row": best_margin,
        "response_validation_gate": response_validation_gate,
        "null_byte_priority_weights": null_byte_priority_weights,
        "magic_codec_seed_boundary": magic_codec_seed_boundary,
        "mlx_torch_parity_sweep_gate": mlx_torch_parity_sweep_gate,
        "mlx_score_calibration_gate": mlx_score_calibration_gate,
        "mlx_production_contract_gate": mlx_production_contract_gate,
        "effective_mlx_spend_triage_gate": effective_mlx_spend_triage_gate,
        "decoder_q_response_surface_summary": decoder_q_response_surface_summary,
        "decoder_q_surface_advisory_gate": decoder_q_surface_advisory_gate,
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


def build_decoder_q_surface_advisory_gate(advisory_batch: dict[str, Any]) -> dict[str, Any]:
    """Gate decoder-q response-surface spend against measured advisory signs."""

    if not isinstance(advisory_batch, dict):
        raise ScorerResponseDatasetError("decoder-q surface advisory batch must be a JSON object")
    if advisory_batch.get("schema") != "fec6_decoder_q_candidate_advisory_batch_v1":
        raise ScorerResponseDatasetError("decoder-q surface advisory batch schema mismatch")
    if advisory_batch.get("producer") != "tools/run_decoder_q_candidate_advisory_batch.py":
        raise ScorerResponseDatasetError("decoder-q surface advisory batch producer mismatch")
    authority = advisory_batch.get("authority")
    if not isinstance(authority, dict):
        raise ScorerResponseDatasetError("decoder-q surface advisory batch authority missing")
    _validate_core_false_authority(
        authority,
        label="decoder-q surface advisory batch authority",
    )

    candidates = advisory_batch.get("candidates")
    if not isinstance(candidates, list):
        raise ScorerResponseDatasetError("decoder-q surface advisory batch candidates[] missing")

    records: list[dict[str, Any]] = []
    for index, row in enumerate(candidates):
        if not isinstance(row, dict):
            raise ScorerResponseDatasetError(
                f"decoder-q surface advisory candidate {index} must be an object"
            )
        _validate_core_false_authority(
            row,
            label=f"decoder-q surface advisory candidate {index}",
        )
        manifest = row.get("mutation_manifest")
        if not isinstance(manifest, dict):
            manifest = {}
        _require_present_false_authority(
            manifest,
            label=f"decoder-q surface advisory candidate {index} manifest",
        )
        advisory = row.get("advisory_eval")
        if not isinstance(advisory, dict):
            advisory = {}
        _require_present_false_authority(
            advisory,
            label=f"decoder-q surface advisory candidate {index} advisory_eval",
        )
        surface_objective = manifest.get("response_surface_objective")
        if isinstance(surface_objective, dict):
            _require_present_false_authority(
                surface_objective,
                label=(
                    f"decoder-q surface advisory candidate {index} "
                    "response_surface_objective"
                ),
            )
        manifest_atoms = _decoder_q_manifest_atom_keys(manifest)
        inputs = advisory_batch.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        baseline = _as_float(inputs.get("baseline_score"))
        score = _as_float(advisory.get("canonical_score"))
        reported_delta = _as_float(row.get("delta_vs_baseline_score"))
        recomputed_delta = score - baseline if baseline is not None and score is not None else None
        delta_consistent = (
            reported_delta is None
            or recomputed_delta is None
            or abs(reported_delta - recomputed_delta) <= 1.0e-12
        )
        delta = recomputed_delta if recomputed_delta is not None else reported_delta
        if delta is None:
            summary = advisory_batch.get("summary") if isinstance(advisory_batch.get("summary"), dict) else {}
            if row.get("candidate_id") == summary.get("best_candidate_id"):
                delta = _as_float(summary.get("best_delta_vs_baseline_score"))
        length_delta = _as_int(manifest.get("length_delta"))
        records.append(
            {
                "candidate_id": row.get("candidate_id"),
                "bucket": manifest.get("bucket"),
                "edit_budget": manifest.get("edit_budget"),
                "fixed_length_runtime_compatible": (
                    manifest.get("fixed_length_runtime_compatible") is True
                    and length_delta == 0
                ),
                "advisory_success": advisory.get("returncode") == 0,
                "delta_vs_baseline_score": delta,
                "reported_delta_vs_baseline_score": reported_delta,
                "recomputed_delta_vs_baseline_score": recomputed_delta,
                "delta_consistent": delta_consistent,
                "canonical_score": _as_float(advisory.get("canonical_score")),
                "avg_segnet_dist": _as_float(advisory.get("avg_segnet_dist")),
                "avg_posenet_dist": _as_float(advisory.get("avg_posenet_dist")),
                "archive_size_bytes": _as_int(advisory.get("archive_size_bytes")),
                "changed_frame_count": _as_int(_get_path(row, ("raw_comparison", "changed_frame_count"))),
                "changed_byte_count": _as_int(_get_path(row, ("raw_comparison", "byte_delta_summary", "changed_byte_count"))),
                "atom_mutation_keys": manifest_atoms,
                "atom_mutation_key_count": len(manifest_atoms),
                "surface_objective_strategy": _get_path(
                    manifest,
                    ("response_surface_objective", "strategy"),
                ),
                "surface_objective_preferred_direction": _get_path(
                    manifest,
                    ("response_surface_objective", "preferred_direction"),
                ),
                "surface_objective_dominant_axis": _get_path(
                    manifest,
                    ("response_surface_objective", "dominant_axis"),
                ),
                "surface_proxy_priority": _as_float(
                    _get_path(manifest, ("response_surface_objective", "proxy_priority_sum"))
                ),
            }
        )

    surface_records = [
        record
        for record in records
        if record.get("bucket") == "response_surface_guided"
    ]
    fixed_surface_records = [
        record
        for record in surface_records
        if record.get("fixed_length_runtime_compatible") is True
    ]
    successful_surface_records = [
        record
        for record in fixed_surface_records
        if record.get("advisory_success") is True
    ]
    improving_surface_records = [
        record
        for record in successful_surface_records
        if (record.get("delta_vs_baseline_score") is not None and float(record["delta_vs_baseline_score"]) < 0.0)
    ]
    regressing_surface_records = [
        record
        for record in successful_surface_records
        if (record.get("delta_vs_baseline_score") is not None and float(record["delta_vs_baseline_score"]) >= 0.0)
    ]
    best_record = min(
        (
            record
            for record in successful_surface_records
            if record.get("delta_vs_baseline_score") is not None
        ),
        key=lambda record: float(record["delta_vs_baseline_score"]),
        default=None,
    )

    blockers: list[str] = []
    if not records:
        blockers.append("decoder_q_surface_advisory_batch_empty")
    if not surface_records:
        blockers.append("decoder_q_surface_advisory_has_no_surface_guided_candidates")
    if surface_records and not fixed_surface_records:
        blockers.append("decoder_q_surface_advisory_has_no_fixed_length_surface_guided_candidates")
    if not successful_surface_records:
        blockers.append("decoder_q_surface_advisory_has_no_successful_surface_guided_candidates")
    if any(record.get("delta_consistent") is False for record in surface_records):
        blockers.append("decoder_q_surface_advisory_delta_mismatch")
    if successful_surface_records and not improving_surface_records:
        blockers.append("decoder_q_surface_advisory_surface_guided_all_non_improving")
    if best_record is None:
        blockers.append("decoder_q_surface_advisory_best_delta_missing")

    status = "strict_pass" if not blockers else "blocked"
    signed_calibration_labels = _decoder_q_surface_sign_calibration_labels(
        surface_records
    )
    return {
        "schema": "ll_decoder_q_surface_advisory_gate.v1",
        "producer": TOOL,
        "source_schema": advisory_batch.get("schema"),
        "status": status,
        "decoder_q_surface_exact_eval_allowed": status == "strict_pass",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "summary": {
            "candidate_count": len(records),
            "surface_guided_candidate_count": len(surface_records),
            "fixed_length_surface_guided_candidate_count": len(fixed_surface_records),
            "successful_surface_guided_candidate_count": len(successful_surface_records),
            "improving_surface_guided_candidate_count": len(improving_surface_records),
            "regressing_surface_guided_candidate_count": len(regressing_surface_records),
            "best_surface_guided_candidate_id": None if best_record is None else best_record.get("candidate_id"),
            "best_surface_guided_delta_vs_baseline_score": None if best_record is None else best_record.get("delta_vs_baseline_score"),
            "signed_calibration_label_count": signed_calibration_labels["summary"][
                "label_count"
            ],
            "signed_calibration_sign_mismatch_count": signed_calibration_labels[
                "summary"
            ]["sign_mismatch_count"],
        },
        "surface_guided_records": surface_records,
        "signed_calibration_labels": signed_calibration_labels,
        "blockers": blockers,
        "allowed_use": (
            "decoder_q_response_surface_exact_eval_spend_filter"
            if not blockers
            else "blocked_until_surface_guided_advisory_candidate_improves"
        ),
    }


def _decoder_q_manifest_atom_keys(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    atoms = manifest.get("atoms")
    if not isinstance(atoms, list):
        return []
    out: list[dict[str, Any]] = []
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        mutation = atom.get("mutation")
        if not isinstance(mutation, dict):
            continue
        tensor_name = mutation.get("tensor_name")
        q_offset = _as_int(mutation.get("q_offset"))
        delta = _as_int(mutation.get("delta"))
        if tensor_name is None or q_offset is None or delta is None:
            continue
        out.append(
            {
                "tensor_name": str(tensor_name),
                "q_offset": q_offset,
                "delta": delta,
            }
        )
    return out


def _score_delta_sign(value: float | None, *, eps: float = 1.0e-12) -> int | None:
    if value is None:
        return None
    if value < -eps:
        return -1
    if value > eps:
        return 1
    return 0


def _decoder_q_surface_sign_calibration_labels(
    surface_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preserve advisory sign outcomes for future decoder-q planning.

    A response-surface-guided candidate is only selected because the local
    surface predicts that it should be spend-worthy. Therefore its expected
    advisory sign is improvement (negative score delta). Regressions are not
    score authority; they are signed calibration labels for future local
    planning and exact-eval spend filters.
    """

    labels: list[dict[str, Any]] = []
    for record in surface_records:
        if record.get("fixed_length_runtime_compatible") is not True:
            continue
        if record.get("advisory_success") is not True:
            continue
        observed_delta = _as_float(record.get("delta_vs_baseline_score"))
        observed_sign = _score_delta_sign(observed_delta)
        if observed_sign is None:
            continue
        expected_sign = -1
        sign_matches = observed_sign == expected_sign
        label_kind = (
            "surface_guided_improved"
            if observed_sign < 0
            else (
                "surface_guided_regressed"
                if observed_sign > 0
                else "surface_guided_neutral"
            )
        )
        labels.append(
            {
                "schema": "decoder_q_surface_sign_calibration_label.v1",
                "candidate_id": record.get("candidate_id"),
                "bucket": record.get("bucket"),
                "edit_budget": record.get("edit_budget"),
                "expected_score_delta_sign": expected_sign,
                "observed_score_delta_sign": observed_sign,
                "sign_matches_expectation": sign_matches,
                "label_kind": label_kind,
                "observed_delta_vs_baseline_score": observed_delta,
                "canonical_score": record.get("canonical_score"),
                "avg_segnet_dist": record.get("avg_segnet_dist"),
                "avg_posenet_dist": record.get("avg_posenet_dist"),
                "surface_objective_strategy": record.get("surface_objective_strategy"),
                "surface_objective_preferred_direction": record.get(
                    "surface_objective_preferred_direction"
                ),
                "surface_objective_dominant_axis": record.get(
                    "surface_objective_dominant_axis"
                ),
                "surface_proxy_priority": record.get("surface_proxy_priority"),
                "atom_mutation_keys": record.get("atom_mutation_keys") or [],
                "recommended_atom_action": (
                    "suppress_same_sign_try_inverse"
                    if observed_sign > 0
                    else (
                        "preserve_or_expand_same_sign"
                        if observed_sign < 0
                        else "treat_as_low_priority_neutral"
                    )
                ),
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
            }
        )
    mismatch_count = len(
        [label for label in labels if label["sign_matches_expectation"] is False]
    )
    regressed_count = len(
        [label for label in labels if label["label_kind"] == "surface_guided_regressed"]
    )
    improved_count = len(
        [label for label in labels if label["label_kind"] == "surface_guided_improved"]
    )
    neutral_count = len(
        [label for label in labels if label["label_kind"] == "surface_guided_neutral"]
    )
    return {
        "schema": DECODER_Q_SURFACE_SIGN_CALIBRATION_SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "allowed_use": "local_decoder_q_sign_calibration_only",
        "forbidden_use": "score_claim_or_rank_or_kill_or_promotion",
        "summary": {
            "label_count": len(labels),
            "sign_mismatch_count": mismatch_count,
            "regressed_label_count": regressed_count,
            "improved_label_count": improved_count,
            "neutral_label_count": neutral_count,
            "all_labels_regressed": bool(labels) and regressed_count == len(labels),
        },
        "labels": labels,
    }


def _require_present_false_authority(payload: dict[str, Any], *, label: str) -> None:
    for key in ("score_claim", *_SOURCE_OPTIONAL_FALSE_AUTHORITY_FIELDS):
        if key in payload and payload.get(key) is not False:
            raise ScorerResponseDatasetError(f"{label} {key} must be false")


def _planning_scope(row: dict[str, Any]) -> str:
    return "normalized_full_video" if _is_mlx_scorer_response_row(row) else "native_row"


def _planning_delta_vs_baseline(row: dict[str, Any]) -> float | None:
    if _is_mlx_scorer_response_row(row):
        value = _as_float(row.get("projected_full_video_delta_vs_baseline_score"))
        if value is not None:
            return value
    return _as_float(row.get("delta_vs_baseline_score"))


def _planning_scorer_gain(row: dict[str, Any]) -> float | None:
    if _is_mlx_scorer_response_row(row):
        value = _as_float(row.get("normalized_full_video_scorer_gain_vs_baseline"))
        if value is not None:
            return value
    return _as_float(row.get("observed_scorer_gain_vs_baseline"))


def _planning_scorer_delta_vs_baseline(row: dict[str, Any]) -> float | None:
    gain = _planning_scorer_gain(row)
    if gain is not None:
        return -gain
    return _as_float(row.get("scorer_delta_vs_baseline"))


def _planning_break_even_bytes(row: dict[str, Any]) -> float | None:
    if _is_mlx_scorer_response_row(row):
        value = _as_float(row.get("break_even_added_bytes_from_normalized_full_video_gain"))
        if value is not None:
            return value
    return _as_float(row.get("break_even_added_bytes_from_scorer_gain"))


def _planning_byte_budget_margin(row: dict[str, Any]) -> float | None:
    if _is_mlx_scorer_response_row(row):
        value = _as_float(row.get("normalized_full_video_byte_budget_margin_vs_break_even"))
        if value is not None:
            return value
    return _as_float(row.get("byte_budget_margin_vs_break_even"))


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
        delta = _planning_delta_vs_baseline(row)
        scorer_delta = _planning_scorer_delta_vs_baseline(row)
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
                "break_even_added_bytes_from_scorer_gain": _planning_break_even_bytes(row),
                "planning_value_scope": _planning_scope(row),
            }
        margin = _planning_byte_budget_margin(row)
        if isinstance(margin, (int, float)) and (
            best_margin is None
            or margin > best_margin["byte_budget_margin_vs_break_even"]
        ):
            best_margin = {
                "row_id": row["row_id"],
                "family": row["family"],
                "byte_budget_margin_vs_break_even": margin,
                "planning_value_scope": _planning_scope(row),
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
        "",
    ]
    lines.extend(render_authority_markdown_block(dataset))
    lines.extend([
        "## Families",
        "",
    ])
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
        f"- Allowed use: `{gate.get('allowed_use')}`",
        f"- Prediction spend-triage usable: `{gate.get('prediction_spend_triage_usable')}`",
        f"- Required spend-triage families: `{gate.get('required_spend_triage_families')}`",
        f"- Required family blockers: `{gate.get('required_family_spend_triage_blockers')}`",
        f"- Spend-triage usable families: `{gate.get('spend_triage_usable_families')}`",
        f"- Spend-triage blocked families: `{gate.get('spend_triage_blocked_families')}`",
        f"- Blockers: `{gate.get('blockers')}`",
        "",
    ]
    lines.extend(render_authority_markdown_block(gate))
    lines.extend([
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
    ])
    for item in gate.get("prediction_evaluations", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            "- "
            f"`{item.get('field')}` n={item.get('present_pair_count')} "
            f"overall_r={item.get('overall_pearson_r')} "
            f"candidate_spend_triage={item.get('candidate_family_spend_triage_usable')} "
            f"passed={item.get('passed')}"
        )
    lines.append("")
    return "\n".join(lines)


def render_next_probe_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# LL Scorer Response Next-Probe Plan",
        "",
        f"- Best total row: `{plan.get('best_total_row')}`",
        f"- Best scorer row: `{plan.get('best_scorer_row')}`",
        f"- Best byte-budget margin row: `{plan.get('best_byte_budget_margin_row')}`",
        "",
    ]
    lines.extend(render_authority_markdown_block(plan))
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
    mlx_calibration_gate = plan.get("mlx_score_calibration_gate")
    if isinstance(mlx_calibration_gate, dict):
        gate_summary = (
            mlx_calibration_gate.get("summary")
            if isinstance(mlx_calibration_gate.get("summary"), dict)
            else {}
        )
        lines.extend(["## MLX Score Calibration Gate", ""])
        lines.append(f"- Status: `{mlx_calibration_gate.get('status')}`")
        lines.append(
            "- Certified pairwise decisions: "
            f"`{gate_summary.get('certified_pairwise_count')}`"
        )
        lines.append(
            "- Uncertain pairwise decisions: "
            f"`{gate_summary.get('uncertain_pairwise_count')}`"
        )
        lines.append(
            "- Min MLX gap for spend triage: "
            f"`{gate_summary.get('recommended_min_mlx_gap_for_spend_triage')}`"
        )
        lines.append(f"- Blockers: `{mlx_calibration_gate.get('blockers')}`")
        lines.append("")
    mlx_production_gate = plan.get("mlx_production_contract_gate")
    if isinstance(mlx_production_gate, dict):
        gate_summary = (
            mlx_production_gate.get("summary")
            if isinstance(mlx_production_gate.get("summary"), dict)
            else {}
        )
        lines.extend(["## MLX Production Contract Gate", ""])
        lines.append(f"- Status: `{mlx_production_gate.get('status')}`")
        lines.append(f"- Source verdict: `{mlx_production_gate.get('source_verdict')}`")
        if mlx_production_gate.get("source_schema") == MLX_PRODUCTION_CONTRACT_BUNDLE_SCHEMA:
            lines.append(f"- Contracts: `{gate_summary.get('contract_count')}`")
            lines.append(f"- Strict contracts: `{gate_summary.get('strict_contract_count')}`")
            lines.append(f"- Dataset rows: `{gate_summary.get('row_count')}`")
            lines.append(f"- Rows covered: `{gate_summary.get('matched_row_count')}`")
            lines.append(
                f"- Rows uncovered: `{gate_summary.get('unmatched_row_count')}`"
            )
            lines.append(f"- Unmatched rows: `{gate_summary.get('unmatched_row_ids')}`")
        else:
            lines.append(f"- Batch pairs: `{gate_summary.get('batch_pairs')}`")
            lines.append(f"- Pair window: `{gate_summary.get('pair_window')}`")
            lines.append(
                "- Archive SHA-256: "
                f"`{gate_summary.get('archive_sha256')}`"
            )
            lines.append(
                "- Inflated aggregate SHA-256: "
                f"`{gate_summary.get('inflated_outputs_aggregate_sha256')}`"
            )
        lines.append(f"- Blockers: `{mlx_production_gate.get('blockers')}`")
        lines.append("")
    effective_mlx_gate = plan.get("effective_mlx_spend_triage_gate")
    if isinstance(effective_mlx_gate, dict):
        gate_summary = (
            effective_mlx_gate.get("summary")
            if isinstance(effective_mlx_gate.get("summary"), dict)
            else {}
        )
        lines.extend(["## Effective MLX Spend Triage Gate", ""])
        lines.append(f"- Status: `{effective_mlx_gate.get('status')}`")
        lines.append(
            "- Spend triage allowed: "
            f"`{effective_mlx_gate.get('mlx_exact_eval_spend_triage_allowed')}`"
        )
        lines.append(f"- MLX rows: `{gate_summary.get('mlx_row_count')}`")
        lines.append(
            "- Response validation: "
            f"`{gate_summary.get('response_validation_status')}`"
        )
        lines.append(
            "- Torch parity: "
            f"`{gate_summary.get('torch_parity_status')}`"
        )
        lines.append(
            "- Score calibration: "
            f"`{gate_summary.get('score_calibration_status')}`"
        )
        lines.append(
            "- Production contract: "
            f"`{gate_summary.get('production_contract_status')}`"
        )
        lines.append(
            "- Production contract rows: "
            f"`{gate_summary.get('production_contract_row_count')}`"
        )
        lines.append(
            "- Production contract matched rows: "
            f"`{gate_summary.get('production_contract_matched_row_count')}`"
        )
        lines.append(
            "- Production contract parent-window matched rows: "
            f"`{gate_summary.get('production_contract_parent_window_matched_row_count')}`"
        )
        lines.append(
            "- Production contract unmatched rows: "
            f"`{gate_summary.get('production_contract_unmatched_row_count')}`"
        )
        lines.append(
            "- Production contract unmatched row sample: "
            f"`{gate_summary.get('production_contract_unmatched_row_ids_sample')}`"
        )
        lines.append(
            "- Production contract blocker sample: "
            f"`{gate_summary.get('production_contract_blockers_sample')}`"
        )
        lines.append(f"- Blockers: `{effective_mlx_gate.get('blockers')}`")
        lines.append("")
    decoder_q_surface_gate = plan.get("decoder_q_surface_advisory_gate")
    if isinstance(decoder_q_surface_gate, dict):
        gate_summary = (
            decoder_q_surface_gate.get("summary")
            if isinstance(decoder_q_surface_gate.get("summary"), dict)
            else {}
        )
        lines.extend(["## Decoder-Q Surface Advisory Gate", ""])
        lines.append(f"- Status: `{decoder_q_surface_gate.get('status')}`")
        lines.append(
            "- Surface-guided candidates: "
            f"`{gate_summary.get('surface_guided_candidate_count')}`"
        )
        lines.append(
            "- Improving surface-guided candidates: "
            f"`{gate_summary.get('improving_surface_guided_candidate_count')}`"
        )
        lines.append(
            "- Best surface-guided delta: "
            f"`{gate_summary.get('best_surface_guided_delta_vs_baseline_score')}`"
        )
        lines.append(f"- Blockers: `{decoder_q_surface_gate.get('blockers')}`")
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
