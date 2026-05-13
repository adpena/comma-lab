"""Fail-closed helpers for contest auth-eval result JSON.

The auth-eval result is the boundary between proxy work and score claims. A
missing score, ``NaN``, or non-finite value must never become a posterior anchor
or provenance ``score_claim=true``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


CONTEST_UNCOMPRESSED_BYTES = 37_545_489.0


@dataclass(frozen=True)
class AuthEvalScore:
    """Parsed finite score plus optional component recomputation evidence."""

    score: float
    source_key: str
    recomputed_score: float | None = None
    recomputed_matches: bool | None = None


@dataclass(frozen=True)
class AuthEvalScoreClaim:
    """Parsed score plus the auth-eval evidence contract that permits claims."""

    score: float
    source_key: str
    score_axis: str
    lane_tag: str
    evidence_grade: str
    exact_cuda_eval_complete: bool
    score_claim: bool
    score_claim_valid: bool
    recomputed_score: float | None = None


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_finite(payload: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[float, str] | None:
    for key in keys:
        if key not in payload:
            continue
        value = _finite_float(payload.get(key))
        if value is not None:
            return value, key
    return None


def recompute_contest_score_from_payload(payload: Mapping[str, Any]) -> float | None:
    """Recompute contest score when component fields are present.

    The repo has accumulated a few JSON field spellings across eval wrappers,
    so this accepts the known aliases but still requires all three components.
    """

    seg = _first_finite(
        payload,
        (
            "avg_segnet_dist",
            "seg_dist",
            "seg_distortion",
            "segmentation_distortion",
            "mask_dist",
            "mask_distortion",
        ),
    )
    pose = _first_finite(
        payload,
        ("avg_posenet_dist", "pose_dist", "pose_distortion"),
    )
    archive_bytes = _first_finite(
        payload,
        ("archive_bytes", "archive_size_bytes", "compressed_size", "bytes"),
    )
    rate_unscaled = _first_finite(payload, ("rate_unscaled", "rate"))
    if seg is None or pose is None or (archive_bytes is None and rate_unscaled is None):
        return None
    seg_value = seg[0]
    pose_value = pose[0]
    if seg_value < 0.0 or pose_value < 0.0:
        return None
    if archive_bytes is not None:
        byte_value = archive_bytes[0]
        if byte_value <= 0.0:
            return None
        rate_term = 25.0 * byte_value / CONTEST_UNCOMPRESSED_BYTES
    else:
        rate_value = rate_unscaled[0]
        if rate_value < 0.0:
            return None
        rate_term = 25.0 * rate_value
    return 100.0 * seg_value + math.sqrt(10.0 * pose_value) + rate_term


def parse_finite_auth_eval_score(
    payload: Mapping[str, Any],
    *,
    score_keys: tuple[str, ...] = (
        "canonical_score",
        "score_recomputed_from_components",
        "recomputed_score",
        "score",
        "final_score",
    ),
    require_component_recompute: bool = False,
    atol: float = 1e-6,
) -> AuthEvalScore | None:
    """Return a score only when it is finite and, when available, coherent.

    ``None`` means "no score claim". Callers should not update posteriors or
    set promotion/score provenance from such a result.
    """

    parsed = _first_finite(payload, score_keys)
    if parsed is None:
        return None
    score, source_key = parsed
    recomputed = recompute_contest_score_from_payload(payload)
    if recomputed is None:
        if require_component_recompute:
            return None
        return AuthEvalScore(score=score, source_key=source_key)
    matches = math.isclose(score, recomputed, rel_tol=0.0, abs_tol=atol)
    if not matches:
        return None
    return AuthEvalScore(
        score=score,
        source_key=source_key,
        recomputed_score=recomputed,
        recomputed_matches=True,
    )


def parse_auth_eval_score_claim(
    payload: Mapping[str, Any],
    *,
    required_score_axis: str | None = None,
    require_component_recompute: bool = True,
) -> AuthEvalScoreClaim | None:
    """Return a score only when the auth-eval contract authorizes a claim.

    This is stricter than :func:`parse_finite_auth_eval_score`: a finite score
    may still be diagnostic-only, such as CUDA on a non-T4 GPU or macOS CPU.
    Callers that update posteriors, greenlight paid full dispatch, or print
    ``[contest-CUDA]`` must use this helper at the score-claim boundary.
    """

    parsed = parse_finite_auth_eval_score(
        payload,
        require_component_recompute=require_component_recompute,
    )
    if parsed is None:
        return None
    score_axis = str(payload.get("score_axis") or "")
    if required_score_axis is not None and score_axis != required_score_axis:
        return None
    score_claim = payload.get("score_claim") is True
    score_claim_valid = payload.get("score_claim_valid") is True
    if not (score_claim and score_claim_valid):
        return None
    exact_cuda_eval_complete = payload.get("exact_cuda_eval_complete") is True
    if required_score_axis == "contest_cuda" and not exact_cuda_eval_complete:
        return None
    return AuthEvalScoreClaim(
        score=parsed.score,
        source_key=parsed.source_key,
        score_axis=score_axis,
        lane_tag=str(payload.get("lane_tag") or ""),
        evidence_grade=str(payload.get("evidence_grade") or ""),
        exact_cuda_eval_complete=exact_cuda_eval_complete,
        score_claim=score_claim,
        score_claim_valid=score_claim_valid,
        recomputed_score=parsed.recomputed_score,
    )


__all__ = [
    "AuthEvalScoreClaim",
    "AuthEvalScore",
    "CONTEST_UNCOMPRESSED_BYTES",
    "parse_auth_eval_score_claim",
    "parse_finite_auth_eval_score",
    "recompute_contest_score_from_payload",
]
