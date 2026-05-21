# SPDX-License-Identifier: MIT
"""Fail-closed transfer-fidelity gate for MLX scorer/training signals.

MLX is useful here because the local machine can run large batches cheaply,
but a local MLX number is not an auth-eval score.  This module compares an
MLX-produced scorer-response or surrogate payload against a byte-closed
contest auth-eval payload and emits a manifest that is intentionally
non-promotable even when the transfer check passes.

The portable spine is NumPy/JSON: MLX and PyTorch tensors should be exported to
NumPy for custody, hashing, and cross-device comparison, then PyTorch/CUDA (or
the upstream CPU evaluator) remains the authority path for promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from tac.auth_eval_schema import (
    FULL_CONTEST_SAMPLE_COUNT,
    ORIGINAL_VIDEO_BYTES,
    contest_formula_score,
    eval_metric_summary,
)
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

__all__ = [
    "MLXScorerFidelityThresholds",
    "build_mlx_scorer_training_signal_fidelity_manifest",
    "load_json_object",
    "write_fidelity_manifest",
]

SCHEMA_VERSION = "mlx_scorer_training_signal_fidelity.v1"
PASS_VERDICT = "PASS_TRAINING_SIGNAL_FIDELITY"
FAIL_VERDICT = "FAIL_TRAINING_SIGNAL_FIDELITY"


@dataclass(frozen=True)
class MLXScorerFidelityThresholds:
    """Strict transfer thresholds for local MLX scorer signals.

    The defaults are deliberately tight enough to catch the MPS-style noisy
    authority problem before a local surrogate can route expensive exact evals.
    For small calibration subsets, callers may pass ``expected_n_samples=None``
    while keeping the same per-axis deltas.
    """

    max_score_abs_delta: float = 1.0e-3
    max_seg_contribution_abs_delta: float = 5.0e-4
    max_pose_contribution_abs_delta: float = 5.0e-4
    max_rate_contribution_abs_delta: float = 0.0
    expected_n_samples: int | None = FULL_CONTEST_SAMPLE_COUNT
    require_archive_identity: bool = True
    require_inflated_output_identity: bool = True


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON object, failing loudly on non-object payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def write_fidelity_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Write a fidelity manifest with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mlx_scorer_training_signal_fidelity_manifest(
    mlx_payload: dict[str, Any] | None,
    auth_eval_payload: dict[str, Any] | None,
    *,
    thresholds: MLXScorerFidelityThresholds | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Compare MLX scorer signal against byte-closed auth-eval ground truth.

    The result can be used to decide whether a local MLX surrogate is calibrated
    enough to guide candidate generation.  It never becomes score evidence.
    """

    limits = thresholds or MLXScorerFidelityThresholds()
    blockers: list[str] = []

    mlx = mlx_payload if isinstance(mlx_payload, dict) else None
    auth = auth_eval_payload if isinstance(auth_eval_payload, dict) else None
    if mlx is None:
        blockers.append("mlx_payload_missing_or_not_object")
        mlx = {}
    if auth is None:
        blockers.append("auth_eval_payload_missing_or_not_object")
        auth = {}

    mlx_metrics = _normalise_metrics(mlx)
    auth_metrics = _normalise_metrics(auth)

    blockers.extend(_metric_blockers("mlx", mlx_metrics))
    blockers.extend(_metric_blockers("auth_eval", auth_metrics))

    _append_identity_blockers(
        blockers,
        mlx=mlx,
        auth=auth,
        require_archive_identity=limits.require_archive_identity,
        require_inflated_output_identity=limits.require_inflated_output_identity,
    )
    _append_axis_and_authority_blockers(blockers, mlx)
    _append_sample_blockers(blockers, mlx_metrics, auth_metrics, limits.expected_n_samples)

    deltas = _component_deltas(mlx_metrics, auth_metrics)
    _append_delta_blockers(blockers, deltas, limits)

    passed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "passed": passed,
        "blockers": blockers,
        "thresholds": asdict(limits),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "device_contract": {
            "allowed_uses": [
                "local_mlx_training_gradient_shaping",
                "local_sweep_reranking_after_passing_transfer_calibration",
                "candidate_generation_prior",
                "signal_exposure",
                "prepaid_dispatch_spend_filter",
            ],
            "forbidden_uses": [
                "auth_eval",
                "score_claim",
                "promotion",
                "rank_or_kill",
                "leaderboard_claim",
                "replacement_for_cuda_t4_or_linux_x86_64_eval",
            ],
        },
        "byte_closure": {
            "archive_sha256": _identity_pair(mlx, auth, _archive_sha256),
            "archive_size_bytes": {
                "mlx": mlx_metrics.get("archive_size_bytes"),
                "auth_eval": auth_metrics.get("archive_size_bytes"),
                "match": mlx_metrics.get("archive_size_bytes")
                is not None
                and mlx_metrics.get("archive_size_bytes") == auth_metrics.get("archive_size_bytes"),
            },
            "inflated_outputs_aggregate_sha256": _identity_pair(
                mlx, auth, _inflated_outputs_aggregate_sha256
            ),
        },
        "axis_metrics": {
            "mlx": mlx_metrics,
            "auth_eval": auth_metrics,
            "delta": deltas,
        },
        "signal_exposure": {
            "score_abs_delta": abs_or_none(deltas.get("score")),
            "seg_contribution_abs_delta": abs_or_none(deltas.get("seg_contribution")),
            "pose_contribution_abs_delta": abs_or_none(deltas.get("pose_contribution")),
            "rate_contribution_abs_delta": abs_or_none(deltas.get("rate_contribution")),
            "tightest_failed_axis": _tightest_failed_axis(deltas, limits),
            "transfer_margin": {
                "score": _margin(limits.max_score_abs_delta, deltas.get("score")),
                "seg_contribution": _margin(
                    limits.max_seg_contribution_abs_delta, deltas.get("seg_contribution")
                ),
                "pose_contribution": _margin(
                    limits.max_pose_contribution_abs_delta, deltas.get("pose_contribution")
                ),
                "rate_contribution": _margin(
                    limits.max_rate_contribution_abs_delta, deltas.get("rate_contribution")
                ),
            },
        },
        "next_authority_step": (
            "If this passes, use the MLX signal only to prioritize byte-closed "
            "candidate archives, then run paired contest auth eval on CUDA/CPU."
        ),
    }


def _normalise_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(eval_metric_summary(payload))
    archive_bytes = metrics.get("archive_size_bytes")
    pose_avg = metrics.get("pose_avg")
    seg_avg = metrics.get("seg_avg")

    if metrics.get("rate_unscaled") is None and archive_bytes is not None:
        metrics["rate_unscaled"] = int(archive_bytes) / ORIGINAL_VIDEO_BYTES
    if metrics.get("rate") is None and archive_bytes is not None:
        metrics["rate"] = 25.0 * int(archive_bytes) / ORIGINAL_VIDEO_BYTES
    if (
        metrics.get("score") is None
        and pose_avg is not None
        and seg_avg is not None
        and archive_bytes is not None
    ):
        metrics["score"] = contest_formula_score(
            seg_dist=float(seg_avg),
            pose_dist=float(pose_avg),
            archive_bytes=int(archive_bytes),
        )
        metrics["canonical_score_source"] = "score_recomputed_from_components"
    metrics["seg_contribution"] = None if seg_avg is None else 100.0 * float(seg_avg)
    metrics["pose_contribution"] = None if pose_avg is None else math.sqrt(10.0 * float(pose_avg))
    metrics["rate_contribution"] = metrics.get("rate")
    return metrics


def _metric_blockers(prefix: str, metrics: dict[str, Any]) -> list[str]:
    blockers = []
    for key in ("score", "pose_avg", "seg_avg", "archive_size_bytes"):
        if metrics.get(key) is None:
            blockers.append(f"{prefix}_{key}_missing")
    return blockers


def _append_identity_blockers(
    blockers: list[str],
    *,
    mlx: dict[str, Any],
    auth: dict[str, Any],
    require_archive_identity: bool,
    require_inflated_output_identity: bool,
) -> None:
    archive = _identity_pair(mlx, auth, _archive_sha256)
    if require_archive_identity and not archive["match"]:
        blockers.append("archive_sha256_identity_mismatch_or_missing")
    inflated = _identity_pair(mlx, auth, _inflated_outputs_aggregate_sha256)
    if require_inflated_output_identity and not inflated["match"]:
        blockers.append("inflated_outputs_aggregate_sha256_identity_mismatch_or_missing")


def _append_axis_and_authority_blockers(blockers: list[str], mlx: dict[str, Any]) -> None:
    if mlx.get("score_claim") is True or mlx.get("score_claim_valid") is True:
        blockers.append("mlx_payload_attempts_score_claim")
    if mlx.get("promotion_eligible") is True:
        blockers.append("mlx_payload_attempts_promotion_eligibility")
    if mlx.get("rank_or_kill_eligible") is True:
        blockers.append("mlx_payload_attempts_rank_or_kill_eligibility")
    evidence_grade = mlx.get("evidence_grade")
    if evidence_grade is not None and evidence_grade != EVIDENCE_GRADE_MLX:
        blockers.append(f"mlx_evidence_grade_not_{EVIDENCE_GRADE_MLX}")


def _append_sample_blockers(
    blockers: list[str],
    mlx_metrics: dict[str, Any],
    auth_metrics: dict[str, Any],
    expected_n_samples: int | None,
) -> None:
    if mlx_metrics.get("n_samples") != auth_metrics.get("n_samples"):
        blockers.append(
            "n_samples_mismatch:"
            f"mlx={mlx_metrics.get('n_samples')}:auth_eval={auth_metrics.get('n_samples')}"
        )
    if expected_n_samples is not None and mlx_metrics.get("n_samples") != expected_n_samples:
        blockers.append(
            f"mlx_n_samples_not_expected:mlx={mlx_metrics.get('n_samples')}:"
            f"expected={expected_n_samples}"
        )
    if expected_n_samples is not None and auth_metrics.get("n_samples") != expected_n_samples:
        blockers.append(
            f"auth_eval_n_samples_not_expected:auth_eval={auth_metrics.get('n_samples')}:"
            f"expected={expected_n_samples}"
        )


def _append_delta_blockers(
    blockers: list[str],
    deltas: dict[str, float | None],
    limits: MLXScorerFidelityThresholds,
) -> None:
    checks = (
        ("score", limits.max_score_abs_delta),
        ("seg_contribution", limits.max_seg_contribution_abs_delta),
        ("pose_contribution", limits.max_pose_contribution_abs_delta),
        ("rate_contribution", limits.max_rate_contribution_abs_delta),
    )
    for key, limit in checks:
        value = deltas.get(key)
        if value is None:
            blockers.append(f"{key}_delta_missing")
        elif abs(value) > limit:
            blockers.append(f"{key}_delta_exceeds_threshold:{abs(value):.12g}>{limit:.12g}")


def _component_deltas(
    mlx_metrics: dict[str, Any],
    auth_metrics: dict[str, Any],
) -> dict[str, float | None]:
    keys = (
        "score",
        "pose_avg",
        "seg_avg",
        "rate",
        "rate_unscaled",
        "seg_contribution",
        "pose_contribution",
        "rate_contribution",
    )
    out: dict[str, float | None] = {}
    for key in keys:
        lhs = mlx_metrics.get(key)
        rhs = auth_metrics.get(key)
        out[key] = None if lhs is None or rhs is None else float(lhs) - float(rhs)
    archive_lhs = mlx_metrics.get("archive_size_bytes")
    archive_rhs = auth_metrics.get("archive_size_bytes")
    out["archive_size_bytes"] = (
        None if archive_lhs is None or archive_rhs is None else float(archive_lhs) - float(archive_rhs)
    )
    return out


def _identity_pair(
    mlx: dict[str, Any],
    auth: dict[str, Any],
    getter: Any,
) -> dict[str, Any]:
    mlx_value = getter(mlx)
    auth_value = getter(auth)
    return {
        "mlx": mlx_value,
        "auth_eval": auth_value,
        "match": mlx_value is not None and mlx_value == auth_value,
    }


def _archive_sha256(payload: dict[str, Any]) -> str | None:
    return _first_nested_string(
        payload,
        (
            "archive_sha256",
            "archive_sha",
            "archive_zip_sha256",
            "archive_zip_sha",
        ),
    )


def _inflated_outputs_aggregate_sha256(payload: dict[str, Any]) -> str | None:
    return _first_nested_string(
        payload,
        (
            "inflated_outputs_aggregate_sha256",
            "inflated_output_aggregate_sha256",
            "raw_inflated_aggregate_sha256",
            "aggregate_sha256",
        ),
    )


def _first_nested_string(payload: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            found = _first_nested_string(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _first_nested_string(value, keys)
            if found is not None:
                return found
    return None


def abs_or_none(value: float | None) -> float | None:
    return None if value is None else abs(float(value))


def _margin(limit: float, value: float | None) -> float | None:
    return None if value is None else float(limit) - abs(float(value))


def _tightest_failed_axis(
    deltas: dict[str, float | None],
    limits: MLXScorerFidelityThresholds,
) -> str | None:
    ratios: list[tuple[float, str]] = []
    for key, limit in (
        ("score", limits.max_score_abs_delta),
        ("seg_contribution", limits.max_seg_contribution_abs_delta),
        ("pose_contribution", limits.max_pose_contribution_abs_delta),
        ("rate_contribution", limits.max_rate_contribution_abs_delta),
    ):
        value = deltas.get(key)
        if value is None:
            continue
        if limit == 0.0:
            if abs(value) > 0.0:
                ratios.append((float("inf"), key))
        else:
            ratios.append((abs(float(value)) / limit, key))
    if not ratios:
        return None
    ratio, key = max(ratios, key=lambda item: item[0])
    return key if ratio > 1.0 else None
