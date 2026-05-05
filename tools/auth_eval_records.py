"""Shared canonical parser for contest_auth_eval-style JSON artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuthEvalRecord:
    score: float | None
    archive_bytes: int | None
    archive_sha256: str | None
    avg_segnet_dist: float | None
    avg_posenet_dist: float | None
    rate_unscaled: float | None
    samples: int | None
    device: str
    gpu_t4_match: bool
    promotion_eligible: bool
    score_claim_valid: bool
    evidence_grade: str


def _get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_auth_eval_payload(payload: dict[str, Any]) -> AuthEvalRecord | None:
    """Parse a score JSON using canonical auth-eval fields first.

    ``final_score`` is a display-rounded upstream report field in many
    artifacts. Ranking tools must prefer ``canonical_score`` and
    ``score_recomputed_from_components`` when present.
    """
    if not isinstance(payload, dict):
        return None
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}

    score = (
        _float(payload.get("canonical_score"))
        or _float(payload.get("score_recomputed_from_components"))
        or _float(payload.get("score"))
        or _float(payload.get("total_score"))
        or _float(payload.get("final_score"))
        or _float(_get(payload, "result", "final_score"))
    )
    if score is None:
        return None

    archive_bytes = (
        _int(payload.get("archive_size_bytes"))
        or _int(payload.get("archive_bytes"))
        or _int(payload.get("archive_size"))
        or _int(payload.get("bytes"))
    )
    device = str(provenance.get("device") or payload.get("device") or "?")
    samples = _int(payload.get("n_samples")) or _int(payload.get("samples"))
    gpu_t4_match = bool(provenance.get("gpu_t4_match") or payload.get("gpu_t4_match"))
    promotion_eligible = bool(payload.get("promotion_eligible", False))
    score_claim_valid = bool(payload.get("score_claim_valid", False))
    if device == "cuda" and samples == 600 and gpu_t4_match:
        promotion_eligible = True if "promotion_eligible" not in payload else promotion_eligible
        score_claim_valid = True if "score_claim_valid" not in payload else score_claim_valid
    evidence_grade = str(payload.get("evidence_grade") or "")
    if not evidence_grade:
        if device == "cuda" and samples == 600 and gpu_t4_match:
            evidence_grade = "A++"
        elif device == "cuda":
            evidence_grade = "A"
        else:
            evidence_grade = "invalid"

    return AuthEvalRecord(
        score=score,
        archive_bytes=archive_bytes,
        archive_sha256=str(provenance.get("archive_sha256") or payload.get("archive_sha256") or payload.get("sha256") or "") or None,
        avg_segnet_dist=(
            _float(payload.get("avg_segnet_dist"))
            or _float(payload.get("seg_dist_avg"))
            or _float(_get(payload, "components", "segnet_avg"))
            or _float(payload.get("segnet_distortion"))
            or _float(payload.get("segnet"))
        ),
        avg_posenet_dist=(
            _float(payload.get("avg_posenet_dist"))
            or _float(payload.get("pose_dist_avg"))
            or _float(_get(payload, "components", "posenet_avg"))
            or _float(payload.get("posenet_distortion"))
            or _float(payload.get("posenet"))
            or _float(payload.get("pose_distortion"))
        ),
        rate_unscaled=_float(payload.get("rate_unscaled")) or _float(payload.get("rate")) or _float(_get(payload, "components", "rate")),
        samples=samples,
        device=device,
        gpu_t4_match=gpu_t4_match,
        promotion_eligible=promotion_eligible,
        score_claim_valid=score_claim_valid,
        evidence_grade=evidence_grade,
    )
