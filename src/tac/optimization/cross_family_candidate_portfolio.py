# SPDX-License-Identifier: MIT
"""Cross-family exact-eval portfolio planner.

This module is planning-only. It composes local MLX triage, DQS1 pairset
acquisition, byte-closed outside-class manifests, and manual candidate rows into
one Bayesian design queue while preserving false-authority boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.bayesian_experimental_design import (
    BayesianExperimentalDesignError,
    rank_exact_eval_candidates,
)
from tac.optimization.mlx_dynamic_sweep_observations import (
    normalize_observation_row,
    summarize_observations,
)
from tac.repo_io import json_text, repo_relative, sha256_file

SCHEMA = "cross_family_candidate_portfolio.v1"
ROW_SCHEMA = "cross_family_candidate_portfolio_row.v1"
TOOL = "tac.optimization.cross_family_candidate_portfolio"

DEFAULT_EXPECTED_IMPROVEMENT_WEIGHT = 1.0
DEFAULT_INFORMATION_GAIN_WEIGHT = 0.01
DEFAULT_MLX_VARIANCE_FLOOR = 1e-8
DEFAULT_PAIRSET_VARIANCE = 2.5e-8
DEFAULT_PAIRSET_OBSERVATION_MODEL_VARIANCE_FLOOR = 2.5e-9
DEFAULT_OUTSIDE_CLASS_VARIANCE = 1e-3
_EXACT_AXIS_ALIASES = {
    "contest_cpu": "contest_cpu",
    "contest-CPU": "contest_cpu",
    "[contest-CPU]": "contest_cpu",
    "contest_cuda": "contest_cuda",
    "contest-CUDA": "contest_cuda",
    "[contest-CUDA]": "contest_cuda",
}

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}

DEFAULT_FAMILY_BELIEFS: dict[str, dict[str, float]] = {
    "decoder_q_selective_dqs1": {
        "prior_score_variance": 2.5e-8,
        "observation_noise_variance": 1e-8,
    },
    "fec6_decoder_q": {
        "prior_score_variance": 1e-7,
        "observation_noise_variance": 2.5e-8,
    },
    "hfv1_foveation": {
        "prior_score_variance": 1e-4,
        "observation_noise_variance": 2.5e-5,
    },
    "hfv2_sparse_sidecar": {
        "prior_score_variance": DEFAULT_OUTSIDE_CLASS_VARIANCE,
        "observation_noise_variance": 1e-4,
    },
    "hnerv_wave": {
        "prior_score_variance": 5e-4,
        "observation_noise_variance": 1e-4,
    },
    "mlx_decoder_q": {
        "prior_score_variance": 6e-6,
        "observation_noise_variance": 4e-6,
    },
    "pr106_format0d": {
        "prior_score_variance": 1e-4,
        "observation_noise_variance": 2.5e-5,
    },
    "score_surface_stack": {
        "prior_score_variance": 2.5e-4,
        "observation_noise_variance": 5e-5,
    },
}


class CrossFamilyCandidatePortfolioError(ValueError):
    """Raised when portfolio inputs are invalid or would blur authority."""


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise CrossFamilyCandidatePortfolioError(f"{label} must be numeric")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise CrossFamilyCandidatePortfolioError(f"{label} must be numeric") from exc
    if not math.isfinite(out):
        raise CrossFamilyCandidatePortfolioError(f"{label} must be finite")
    return out


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if key in payload and payload.get(key) is not False:
            raise CrossFamilyCandidatePortfolioError(f"{label} {key} must be false")


def _source_artifact(path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": repo_relative(path, repo_root or Path.cwd()),
        "exists": exists,
        "sha256": sha256_file(path) if exists else "",
        "size_bytes": path.stat().st_size if exists else None,
    }


def _source_key(path: Path) -> str:
    stem = path.stem.replace("-", "_")
    return stem or "source"


def source_artifacts_from_paths(
    paths: Mapping[str, Path | Sequence[Path] | None],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Return deterministic source metadata for planner inputs."""

    out: dict[str, Any] = {}
    for key, value in paths.items():
        if value is None:
            continue
        if isinstance(value, Path):
            out[key] = _source_artifact(value, repo_root=repo_root)
            continue
        out[key] = [_source_artifact(path, repo_root=repo_root) for path in value]
    return out


def _merge_family_beliefs(
    candidates: Sequence[Mapping[str, Any]],
    family_beliefs: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {
        family: dict(values) for family, values in DEFAULT_FAMILY_BELIEFS.items()
    }
    if isinstance(family_beliefs, Mapping):
        for family, raw in family_beliefs.items():
            values = raw if isinstance(raw, Mapping) else {"prior_score_variance": raw}
            merged[str(family)] = {
                "prior_score_variance": _finite_float(
                    values.get("prior_score_variance", values.get("prior_variance", 1.0)),
                    label=f"{family}.prior_score_variance",
                ),
                "observation_noise_variance": _finite_float(
                    values.get(
                        "observation_noise_variance",
                        values.get("noise_variance", 1e-4),
                    ),
                    label=f"{family}.observation_noise_variance",
                ),
            }
    elif family_beliefs is not None:
        for row in family_beliefs:
            family = str(row.get("family_id") or row.get("family") or "")
            if not family:
                raise CrossFamilyCandidatePortfolioError("family belief missing family_id")
            merged[family] = {
                "prior_score_variance": _finite_float(
                    row.get("prior_score_variance", row.get("prior_variance", 1.0)),
                    label=f"{family}.prior_score_variance",
                ),
                "observation_noise_variance": _finite_float(
                    row.get("observation_noise_variance", row.get("noise_variance", 1e-4)),
                    label=f"{family}.observation_noise_variance",
                ),
            }
    for candidate in candidates:
        family = str(candidate.get("family") or candidate.get("family_id") or "unknown")
        if family not in merged:
            variance = _finite_float(
                candidate.get("predicted_score_variance", DEFAULT_OUTSIDE_CLASS_VARIANCE),
                label=f"{family}.predicted_score_variance",
            )
            merged[family] = {
                "prior_score_variance": max(variance, DEFAULT_MLX_VARIANCE_FLOOR),
                "observation_noise_variance": max(variance * 0.25, 1e-8),
            }
    return dict(sorted(merged.items()))


def _candidate(
    *,
    candidate_id: str,
    family: str,
    predicted_score_mean: float,
    predicted_score_variance: float,
    source_kind: str,
    source_rank: int | None = None,
    source_artifact_path: str | None = None,
    exact_archive_custody: Mapping[str, Any] | None = None,
    family_couplings: Mapping[str, float] | None = None,
    source_dispatch_blockers: Sequence[Any] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not candidate_id:
        raise CrossFamilyCandidatePortfolioError("candidate_id must be non-empty")
    if not family:
        raise CrossFamilyCandidatePortfolioError(f"{candidate_id}: family must be non-empty")
    variance = _finite_float(
        predicted_score_variance,
        label=f"{candidate_id}.predicted_score_variance",
    )
    if variance < 0.0:
        raise CrossFamilyCandidatePortfolioError(
            f"{candidate_id}.predicted_score_variance must be non-negative"
        )
    row = {
        "candidate_id": candidate_id,
        "family": family,
        "predicted_score_mean": _finite_float(
            predicted_score_mean,
            label=f"{candidate_id}.predicted_score_mean",
        ),
        "predicted_score_variance": variance,
        "source_kind": source_kind,
        "source_rank": source_rank,
        "source_artifact_path": source_artifact_path,
        "candidate_generation_only": True,
        **FALSE_AUTHORITY,
    }
    if exact_archive_custody is not None:
        row["exact_archive_custody"] = dict(exact_archive_custody)
    if family_couplings:
        row["family_couplings"] = {
            str(key): _finite_float(value, label=f"{candidate_id}.family_couplings.{key}")
            for key, value in family_couplings.items()
        }
    if source_dispatch_blockers:
        row["source_dispatch_blockers"] = sorted(
            {str(blocker) for blocker in source_dispatch_blockers if str(blocker)}
        )
    if source_metadata:
        row["source_metadata"] = dict(source_metadata)
    return row


def _mlx_candidate_rows(
    selection: Mapping[str, Any],
    *,
    incumbent_score: float,
    source_artifact_path: str | None,
) -> list[dict[str, Any]]:
    if selection.get("schema") != "mlx_effective_spend_triage_candidate_selection.v1":
        raise CrossFamilyCandidatePortfolioError("MLX selection schema mismatch")
    _require_false_authority(selection, label="MLX selection")
    selected_rows = selection.get("selected_rows")
    if not isinstance(selected_rows, list):
        raise CrossFamilyCandidatePortfolioError("MLX selection selected_rows[] missing")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        if not isinstance(row, Mapping):
            raise CrossFamilyCandidatePortfolioError(f"MLX selected row {index} must be object")
        _require_false_authority(row, label=f"MLX selected row {index}")
        candidate_id = str(row.get("candidate_id") or row.get("row_id") or f"mlx_row_{index:04d}")
        family = str(row.get("family") or "mlx_decoder_q")
        observed_delta = _finite_float(
            row.get("observed_delta_vs_baseline_score"),
            label=f"{candidate_id}.observed_delta_vs_baseline_score",
        )
        predicted_delta = row.get("predicted_delta_vs_baseline_score")
        disagreement = 0.0
        if predicted_delta is not None:
            disagreement = abs(
                observed_delta
                - _finite_float(
                    predicted_delta,
                    label=f"{candidate_id}.predicted_delta_vs_baseline_score",
                )
            )
        calibrated_gap = abs(
            _finite_float(
                row.get("calibrated_min_mlx_gap_for_spend_triage", 0.0),
                label=f"{candidate_id}.calibrated_min_mlx_gap_for_spend_triage",
            )
        )
        variance = max(
            DEFAULT_MLX_VARIANCE_FLOOR,
            disagreement * disagreement,
            calibrated_gap * calibrated_gap,
        )
        out.append(
            _candidate(
                candidate_id=candidate_id,
                family=family,
                predicted_score_mean=incumbent_score + observed_delta,
                predicted_score_variance=variance,
                source_kind="mlx_effective_spend_triage_selection",
                source_rank=int(row.get("rank", index + 1)),
                source_artifact_path=source_artifact_path,
                family_couplings={
                    "decoder_q_selective_dqs1": 0.7,
                    "fec6_decoder_q": 0.55,
                    "score_surface_stack": 0.35,
                },
                source_metadata={
                    "selection_basis": row.get("selection_basis"),
                    "pair_indices": row.get("pair_indices"),
                    "observed_delta_vs_baseline_score": observed_delta,
                    "predicted_delta_vs_baseline_score": predicted_delta,
                    "byte_budget_margin_vs_break_even": row.get(
                        "byte_budget_margin_vs_break_even"
                    ),
                    "requires_exact_auth_eval_before_score_claim": row.get(
                        "requires_exact_auth_eval_before_score_claim"
                    ),
                },
            )
        )
    return out


def _pairset_candidate_rows(
    acquisition: Mapping[str, Any],
    *,
    source_artifact_path: str | None,
) -> list[dict[str, Any]]:
    if acquisition.get("schema") != "decoder_q_pairset_acquisition.v1":
        raise CrossFamilyCandidatePortfolioError("pairset acquisition schema mismatch")
    _require_false_authority(acquisition, label="pairset acquisition")
    candidates = acquisition.get("candidates")
    if not isinstance(candidates, list):
        raise CrossFamilyCandidatePortfolioError("pairset acquisition candidates[] missing")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise CrossFamilyCandidatePortfolioError(f"pairset candidate {index} must be object")
        _require_false_authority(row, label=f"pairset candidate {index}")
        candidate_id = str(row.get("acquisition_id") or row.get("selector_id") or f"pairset_{index:04d}")
        predicted = row.get("predicted_score_mean")
        estimate = row.get("exact_cpu_calibrated_estimate")
        if predicted is None and isinstance(estimate, Mapping):
            predicted = estimate.get("predicted_score")
        if predicted is None:
            raise CrossFamilyCandidatePortfolioError(
                f"{candidate_id}: pairset predicted_score_mean missing"
            )
        out.append(
            _candidate(
                candidate_id=candidate_id,
                family="decoder_q_selective_dqs1",
                predicted_score_mean=_finite_float(
                    predicted,
                    label=f"{candidate_id}.predicted_score_mean",
                ),
                predicted_score_variance=_finite_float(
                    row.get("predicted_score_variance", DEFAULT_PAIRSET_VARIANCE),
                    label=f"{candidate_id}.predicted_score_variance",
                ),
                source_kind="decoder_q_pairset_acquisition",
                source_rank=int(row.get("acquisition_rank", index + 1)),
                source_artifact_path=source_artifact_path,
                family_couplings={
                    "fec6_decoder_q": 0.9,
                    "mlx_decoder_q": 0.7,
                    "score_surface_stack": 0.3,
                },
                source_metadata={
                    "selector_kind": row.get("selector_kind"),
                    "selected_pair_count": row.get("selected_pair_count"),
                    "selected_pair_indices": row.get("selected_pair_indices"),
                    "payload_bytes": row.get("payload_bytes"),
                    "rate_delta": row.get("rate_delta"),
                    "predicted_score_source": row.get("predicted_score_source"),
                },
            )
        )
    return out


def _hfv2_candidate_rows(
    manifest: Mapping[str, Any],
    *,
    incumbent_score: float,
    source_artifact_path: str | None,
) -> list[dict[str, Any]]:
    if manifest.get("schema") != "hfv1_to_hfv2_sparse_sidecar_candidate_v1":
        raise CrossFamilyCandidatePortfolioError("HFV2 manifest schema mismatch")
    _require_false_authority(manifest, label="HFV2 manifest")
    archive_path = str(
        manifest.get("output_submission_archive")
        or manifest.get("output_archive")
        or ""
    )
    archive_sha = str(manifest.get("output_archive_sha256") or "")
    archive_size = manifest.get("output_archive_bytes")
    if not archive_path or not archive_sha or archive_size is None:
        raise CrossFamilyCandidatePortfolioError("HFV2 manifest missing output archive custody")
    predicted = manifest.get("predicted_score_mean")
    if predicted is None:
        predicted = incumbent_score + max(
            0.0,
            _finite_float(
                manifest.get("rate_delta_vs_baseline_archive", 0.0),
                label="HFV2 rate_delta_vs_baseline_archive",
            ),
        )
    return [
        _candidate(
            candidate_id=str(manifest.get("candidate_id") or "hfv2_sparse_sidecar_magic_bin"),
            family="hfv2_sparse_sidecar",
            predicted_score_mean=_finite_float(
                predicted,
                label="HFV2 predicted_score_mean",
            ),
            predicted_score_variance=_finite_float(
                manifest.get("predicted_score_variance", DEFAULT_OUTSIDE_CLASS_VARIANCE),
                label="HFV2 predicted_score_variance",
            ),
            source_kind="hfv2_sparse_sidecar_manifest",
            source_rank=1,
            source_artifact_path=source_artifact_path,
            exact_archive_custody={
                "archive_path": archive_path,
                "archive_sha256": archive_sha,
                "archive_size_bytes": int(archive_size),
            },
            family_couplings={
                "hfv1_foveation": 0.85,
                "pr106_format0d": 0.4,
                "hnerv_wave": 0.25,
                "score_surface_stack": 0.2,
            },
            source_dispatch_blockers=(
                manifest.get("dispatch_blockers")
                if isinstance(manifest.get("dispatch_blockers"), Sequence)
                and not isinstance(manifest.get("dispatch_blockers"), str)
                else None
            ),
            source_metadata={
                "sparse_pair_count": manifest.get("sparse_pair_count"),
                "bytes_delta_vs_baseline_archive": manifest.get(
                    "bytes_delta_vs_baseline_archive"
                ),
                "row_parity_exact": manifest.get("row_parity_exact"),
                "dispatch_blockers": manifest.get("dispatch_blockers"),
                "target_modes": manifest.get("target_modes"),
            },
        )
    ]


def _manual_candidate_rows(
    manual_candidates: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(manual_candidates or ()):
        _require_false_authority(row, label=f"manual candidate {index}")
        candidate_id = str(row.get("candidate_id") or row.get("id") or "")
        family = str(row.get("family") or row.get("family_id") or "")
        if not candidate_id or not family:
            raise CrossFamilyCandidatePortfolioError(
                f"manual candidate {index} missing candidate_id or family"
            )
        out.append(
            _candidate(
                candidate_id=candidate_id,
                family=family,
                predicted_score_mean=_finite_float(
                    row.get("predicted_score_mean", row.get("score_mean")),
                    label=f"{candidate_id}.predicted_score_mean",
                ),
                predicted_score_variance=_finite_float(
                    row.get("predicted_score_variance", row.get("score_variance", 1e-6)),
                    label=f"{candidate_id}.predicted_score_variance",
                ),
                source_kind=str(row.get("source_kind") or "manual_candidate"),
                source_rank=(
                    int(row["source_rank"])
                    if row.get("source_rank") is not None
                    else None
                ),
                source_artifact_path=(
                    str(row["source_artifact_path"])
                    if row.get("source_artifact_path")
                    else None
                ),
                exact_archive_custody=(
                    row.get("exact_archive_custody")
                    if isinstance(row.get("exact_archive_custody"), Mapping)
                    else None
                ),
                family_couplings=(
                    row.get("family_couplings")
                    if isinstance(row.get("family_couplings"), Mapping)
                    else None
                ),
                source_metadata=(
                    row.get("source_metadata")
                    if isinstance(row.get("source_metadata"), Mapping)
                    else None
                ),
            )
        )
    return out


def _normalize_observations(
    observations: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    return [normalize_observation_row(row) for row in observations or ()]


def _exact_observation_axis(row: Mapping[str, Any]) -> str | None:
    axes = {
        _EXACT_AXIS_ALIASES.get(str(row.get("observed_axis") or "")),
        _EXACT_AXIS_ALIASES.get(str(row.get("evidence_grade") or "")),
        _EXACT_AXIS_ALIASES.get(str(row.get("evidence_tag") or "")),
    }
    axes.discard(None)
    if not axes:
        return None
    if len(axes) != 1:
        raise CrossFamilyCandidatePortfolioError(
            "observation exact-axis labels disagree across observed_axis/evidence_grade/evidence_tag"
        )
    return next(iter(axes))


def _observation_axis_is_exact(row: Mapping[str, Any]) -> bool:
    return _exact_observation_axis(row) is not None


def _normalize_incumbent_scores_by_axis(
    incumbent_score: float,
    incumbent_scores_by_axis: Mapping[str, Any] | None,
) -> dict[str, float]:
    scores = {"contest_cuda": _finite_float(incumbent_score, label="incumbent_score")}
    for key, value in (incumbent_scores_by_axis or {}).items():
        axis = _EXACT_AXIS_ALIASES.get(str(key))
        if axis is None:
            raise CrossFamilyCandidatePortfolioError(
                f"unsupported incumbent score axis: {key!r}"
            )
        score = _finite_float(value, label=f"incumbent_scores_by_axis.{axis}")
        if score < 0.0:
            raise CrossFamilyCandidatePortfolioError(
                f"incumbent_scores_by_axis.{axis} must be non-negative"
            )
        scores[axis] = score
    return scores


def _candidate_selected_pair_count(row: Mapping[str, Any]) -> int | None:
    metadata = row.get("source_metadata")
    raw = row.get("selected_pair_count")
    if raw is None and isinstance(metadata, Mapping):
        raw = metadata.get("selected_pair_count")
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise CrossFamilyCandidatePortfolioError(
            f"{row.get('candidate_id')}.selected_pair_count must be an integer"
        )
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise CrossFamilyCandidatePortfolioError(
            f"{row.get('candidate_id')}.selected_pair_count must be an integer"
        ) from exc
    if count <= 0:
        raise CrossFamilyCandidatePortfolioError(
            f"{row.get('candidate_id')}.selected_pair_count must be positive"
        )
    return count


def _fit_line(xs: Sequence[float], ys: Sequence[float]) -> dict[str, float] | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    variance_x = sum((x - x_mean) ** 2 for x in xs)
    if variance_x <= 0.0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / variance_x
    intercept = y_mean - slope * x_mean
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True)]
    residual_mse = sum(value * value for value in residuals) / max(1, len(residuals) - 2)
    return {
        "intercept": intercept,
        "slope_per_pair": slope,
        "residual_mse": residual_mse,
    }


def _pairset_observation_training_rows(
    candidates: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> tuple[str | None, list[dict[str, Any]], dict[str, int]]:
    candidates_by_id = {
        str(row.get("candidate_id") or ""): row
        for row in candidates
        if str(row.get("source_kind") or "") == "decoder_q_pairset_acquisition"
    }
    rows_by_axis: dict[str, list[dict[str, Any]]] = {}
    axis_counts: dict[str, int] = {}
    for observation in observations:
        axis = _exact_observation_axis(observation)
        if axis is None:
            continue
        candidate_id = str(observation.get("candidate_id") or "")
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            continue
        count = _candidate_selected_pair_count(candidate)
        if count is None:
            continue
        score = _finite_float(
            observation.get("observed_score_or_delta"),
            label=f"{candidate_id}.observed_score_or_delta",
        )
        rows_by_axis.setdefault(axis, []).append(
            {
                "candidate_id": candidate_id,
                "selected_pair_count": count,
                "observed_score": score,
                "observed_at_utc": observation.get("observed_at_utc"),
            }
        )
        axis_counts[axis] = axis_counts.get(axis, 0) + 1
    if not rows_by_axis:
        return None, [], axis_counts
    selected_axis = sorted(
        rows_by_axis,
        key=lambda axis: (-len(rows_by_axis[axis]), axis),
    )[0]
    return selected_axis, rows_by_axis[selected_axis], axis_counts


def _apply_pairset_observation_response_model(
    candidates: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use exact pairset observations to recalibrate unobserved pairset priors.

    This is a planning prior only: it updates predicted means/variances for
    ordering future local controls and exact-eval spend, while leaving all
    authority fields false.
    """

    axis, training_rows, axis_counts = _pairset_observation_training_rows(
        candidates,
        observations,
    )
    base_summary: dict[str, Any] = {
        "schema": "pairset_observation_response_model.v1",
        "active": False,
        "axis": axis,
        "axis_observation_counts": dict(sorted(axis_counts.items())),
        "training_row_count": len(training_rows),
        **FALSE_AUTHORITY,
    }
    xs = [float(row["selected_pair_count"]) for row in training_rows]
    ys = [float(row["observed_score"]) for row in training_rows]
    fit = _fit_line(xs, ys)
    if fit is None:
        return [dict(row) for row in candidates], {
            **base_summary,
            "inactive_reason": "need_two_distinct_selected_pair_counts",
        }

    observed_counts = sorted({int(row["selected_pair_count"]) for row in training_rows})
    observed_ids = sorted({str(row["candidate_id"]) for row in training_rows})
    blend_weight = min(0.85, len(training_rows) / float(len(training_rows) + 3))
    residual_mse = max(0.0, float(fit["residual_mse"]))
    updated: list[dict[str, Any]] = []
    updated_count = 0
    for candidate in candidates:
        row = dict(candidate)
        if str(row.get("source_kind") or "") != "decoder_q_pairset_acquisition":
            updated.append(row)
            continue
        count = _candidate_selected_pair_count(row)
        if count is None:
            updated.append(row)
            continue
        prior_mean = _finite_float(
            row.get("predicted_score_mean"),
            label=f"{row.get('candidate_id')}.predicted_score_mean",
        )
        model_mean = fit["intercept"] + fit["slope_per_pair"] * float(count)
        posterior_mean = (1.0 - blend_weight) * prior_mean + blend_weight * model_mean
        nearest_count = min(observed_counts, key=lambda value: abs(value - count))
        distance = abs(float(count - nearest_count))
        distance_variance = (abs(float(fit["slope_per_pair"])) * distance * 0.25) ** 2
        prior_variance = _finite_float(
            row.get("predicted_score_variance", DEFAULT_PAIRSET_VARIANCE),
            label=f"{row.get('candidate_id')}.predicted_score_variance",
        )
        posterior_variance = max(
            DEFAULT_PAIRSET_OBSERVATION_MODEL_VARIANCE_FLOOR,
            residual_mse,
            distance_variance,
            prior_variance * 0.1,
        )
        metadata = dict(row.get("source_metadata") or {})
        metadata["observation_response_model"] = {
            **base_summary,
            "active": True,
            "model_kind": "linear_selected_pair_count_exact_axis",
            "observed_candidate_ids": observed_ids,
            "observed_selected_pair_counts": observed_counts,
            "intercept": round(float(fit["intercept"]), 15),
            "slope_per_pair": round(float(fit["slope_per_pair"]), 15),
            "residual_mse": residual_mse,
            "prior_predicted_score_mean": prior_mean,
            "model_predicted_score_mean": model_mean,
            "posterior_blend_weight": blend_weight,
            "nearest_observed_selected_pair_count": nearest_count,
            "selected_pair_count_distance": distance,
            "posterior_score_variance": posterior_variance,
            "allowed_use": "planning_prior_only_no_score_or_dispatch_authority",
        }
        row["source_metadata"] = metadata
        row["predicted_score_mean"] = posterior_mean
        row["predicted_score_variance"] = posterior_variance
        row["prediction_source"] = "exact_pairset_observation_response_model_planning_prior"
        updated_count += 1
        updated.append(row)

    return updated, {
        **base_summary,
        "active": True,
        "model_kind": "linear_selected_pair_count_exact_axis",
        "updated_candidate_count": updated_count,
        "observed_candidate_ids": observed_ids,
        "observed_selected_pair_counts": observed_counts,
        "intercept": round(float(fit["intercept"]), 15),
        "slope_per_pair": round(float(fit["slope_per_pair"]), 15),
        "residual_mse": residual_mse,
        "posterior_blend_weight": blend_weight,
        "variance_floor": DEFAULT_PAIRSET_OBSERVATION_MODEL_VARIANCE_FLOOR,
        "allowed_use": "planning_prior_only_no_score_or_dispatch_authority",
    }


def _apply_observation_feedback(
    candidates: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    incumbent_scores_by_axis: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Attach exact-axis observation feedback without granting authority."""

    by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for row in observations:
        by_candidate.setdefault(str(row.get("candidate_id") or ""), []).append(row)

    out: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        matching = [
            obs
            for obs in by_candidate.get(str(row.get("candidate_id") or ""), [])
            if _observation_axis_is_exact(obs)
        ]
        if not matching:
            out.append(row)
            continue
        matching.sort(key=lambda obs: str(obs.get("observed_at_utc") or ""))
        latest = matching[-1]
        observed_axis = _exact_observation_axis(latest)
        if observed_axis is None:
            out.append(row)
            continue
        observed_score = _finite_float(
            latest.get("observed_score_or_delta"),
            label=f"{row.get('candidate_id')}.observed_score_or_delta",
        )
        axis_baseline = incumbent_scores_by_axis.get(observed_axis)
        observation_feedback: dict[str, Any] = {
            "candidate_id": latest.get("candidate_id"),
            "observed_axis": observed_axis,
            "evidence_grade": latest.get("evidence_grade"),
            "latest_observed_at_utc": latest.get("observed_at_utc"),
            "observed_score": observed_score,
            "source_artifact_path": latest.get("source_artifact_path"),
            "source_artifact_sha256": latest.get("source_artifact_sha256"),
            **FALSE_AUTHORITY,
        }
        if axis_baseline is None:
            status = "observed_exact_axis_without_axis_baseline"
            observation_feedback["axis_baseline_available"] = False
        else:
            observed_delta = observed_score - axis_baseline
            status = (
                "observed_exact_axis_regressed_vs_axis_baseline"
                if observed_delta >= 0.0
                else "observed_exact_axis_improved_vs_axis_baseline"
            )
            observation_feedback.update(
                {
                    "axis_baseline_available": True,
                    "axis_baseline_score": axis_baseline,
                    "observed_delta_vs_axis_baseline": observed_delta,
                }
            )
        observation_feedback["status"] = status
        source_metadata = dict(row.get("source_metadata") or {})
        source_metadata["observation_feedback"] = observation_feedback
        row["source_metadata"] = source_metadata
        blockers = set(row.get("source_dispatch_blockers") or [])
        blockers.add(f"candidate_already_observed_{observed_axis}_do_not_repeat_same_axis")
        if status == "observed_exact_axis_regressed_vs_axis_baseline":
            blockers.add(f"candidate_observed_{observed_axis}_regressed_vs_axis_baseline")
        row["source_dispatch_blockers"] = sorted(blockers)
        out.append(row)
    return out


def _enforce_portfolio_false_authority(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    required_blockers = {
        "portfolio_planning_only_requires_separate_lane_claim",
        "auth_axis_gate_required_before_dispatch",
        "score_claim_requires_contest_auth_eval",
    }
    for row in rows:
        portfolio_row = dict(row)
        source_ready = bool(portfolio_row.get("ready_for_exact_eval_dispatch"))
        source_blockers = set(portfolio_row.get("dispatch_blockers") or [])
        source_blockers.update(portfolio_row.get("source_dispatch_blockers") or [])
        portfolio_row["schema"] = ROW_SCHEMA
        portfolio_row["bayesian_ready_for_exact_eval_dispatch"] = source_ready
        portfolio_row["exact_archive_custody_ready"] = bool(
            isinstance(portfolio_row.get("exact_archive_custody"), Mapping)
            and portfolio_row["exact_archive_custody"].get("verified") is True
        )
        portfolio_row.update(FALSE_AUTHORITY)
        portfolio_row["ready_for_exact_eval_dispatch"] = False
        portfolio_row["dispatch_blockers"] = sorted(source_blockers | required_blockers)
        portfolio_row["operator_note"] = "portfolio_rank_only_no_dispatch"
        out.append(portfolio_row)
    return out


def _operator_action_for_row(row: Mapping[str, Any]) -> str:
    source_kind = str(row.get("source_kind") or "")
    blockers = set(row.get("source_dispatch_blockers") or [])
    if source_kind == "decoder_q_pairset_acquisition":
        return "materialize_pairset_archive_and_run_local_controls"
    if source_kind == "mlx_effective_spend_triage_selection":
        return "materialize_mlx_selected_window_archive_and_run_controls"
    if source_kind == "hfv2_sparse_sidecar_manifest" and blockers:
        return "hold_or_refresh_current_runtime_anchor_only_if_deliberately_needed"
    if source_kind == "hfv2_sparse_sidecar_manifest":
        return "run_local_controls_before_any_auth_axis_spend"
    return "manual_review_before_materialization_or_dispatch"


def _operator_action_priority(row: Mapping[str, Any]) -> tuple[int, int, float, str]:
    source_kind = str(row.get("source_kind") or "")
    source_blocked = bool(row.get("source_dispatch_blockers"))
    observed_feedback = (
        row.get("source_metadata", {}).get("observation_feedback")
        if isinstance(row.get("source_metadata"), Mapping)
        else None
    )
    if source_kind == "decoder_q_pairset_acquisition":
        base = 10
    elif source_kind == "mlx_effective_spend_triage_selection":
        base = 20
    elif source_kind == "hfv2_sparse_sidecar_manifest":
        base = 80 if source_blocked else 30
    else:
        base = 40
    if isinstance(observed_feedback, Mapping):
        status = str(observed_feedback.get("status") or "")
        if status == "observed_exact_axis_improved_vs_axis_baseline":
            base -= 5
        else:
            base += 100
    source_rank = row.get("source_rank")
    rank_value = int(source_rank) if source_rank is not None else 999_999
    acquisition_value = -float(row.get("acquisition_value", 0.0))
    if source_kind == "decoder_q_pairset_acquisition" and str(row.get("prediction_source") or "") == (
        "exact_pairset_observation_response_model_planning_prior"
    ):
        return (
            base,
            acquisition_value,
            rank_value,
            str(row.get("candidate_id") or ""),
        )
    return (
        base,
        rank_value,
        acquisition_value,
        str(row.get("candidate_id") or ""),
    )


def _build_operator_action_rows(
    ranked_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in ranked_rows:
        action_row = dict(row)
        action_row["operator_next_action"] = _operator_action_for_row(action_row)
        action_row["operator_action_priority"] = list(_operator_action_priority(action_row))
        rows.append(action_row)
    rows.sort(key=_operator_action_priority)
    for index, row in enumerate(rows, start=1):
        row["operator_action_rank"] = index
    return rows


def build_cross_family_candidate_portfolio(
    *,
    incumbent_score: float,
    mlx_selections: Sequence[Mapping[str, Any]] | None = None,
    pairset_acquisitions: Sequence[Mapping[str, Any]] | None = None,
    hfv2_manifests: Sequence[Mapping[str, Any]] | None = None,
    manual_candidates: Sequence[Mapping[str, Any]] | None = None,
    observations: Sequence[Mapping[str, Any]] | None = None,
    incumbent_scores_by_axis: Mapping[str, Any] | None = None,
    family_beliefs: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    source_artifacts: Mapping[str, Any] | None = None,
    source_artifact_paths: Mapping[str, Sequence[str | None]] | None = None,
    top_k: int | None = 32,
    expected_improvement_weight: float = DEFAULT_EXPECTED_IMPROVEMENT_WEIGHT,
    information_gain_weight: float = DEFAULT_INFORMATION_GAIN_WEIGHT,
) -> dict[str, Any]:
    """Build a false-authority ranked cross-family candidate portfolio."""

    incumbent = _finite_float(incumbent_score, label="incumbent_score")
    if incumbent < 0.0:
        raise CrossFamilyCandidatePortfolioError("incumbent_score must be non-negative")
    if top_k is not None and int(top_k) <= 0:
        raise CrossFamilyCandidatePortfolioError("top_k must be positive")

    source_paths = source_artifact_paths or {}
    candidates: list[dict[str, Any]] = []
    for index, selection in enumerate(mlx_selections or ()):
        path_list = source_paths.get("mlx_selections", ())
        path = path_list[index] if index < len(path_list) else None
        candidates.extend(
            _mlx_candidate_rows(
                selection,
                incumbent_score=incumbent,
                source_artifact_path=path,
            )
        )
    for index, acquisition in enumerate(pairset_acquisitions or ()):
        path_list = source_paths.get("pairset_acquisitions", ())
        path = path_list[index] if index < len(path_list) else None
        candidates.extend(
            _pairset_candidate_rows(acquisition, source_artifact_path=path)
        )
    for index, manifest in enumerate(hfv2_manifests or ()):
        path_list = source_paths.get("hfv2_manifests", ())
        path = path_list[index] if index < len(path_list) else None
        candidates.extend(
            _hfv2_candidate_rows(
                manifest,
                incumbent_score=incumbent,
                source_artifact_path=path,
            )
        )
    candidates.extend(_manual_candidate_rows(manual_candidates))

    if not candidates:
        raise CrossFamilyCandidatePortfolioError("no candidates supplied")
    normalized_observations = _normalize_observations(observations)
    axis_incumbents = _normalize_incumbent_scores_by_axis(
        incumbent,
        incumbent_scores_by_axis,
    )
    candidates, pairset_observation_response_model = _apply_pairset_observation_response_model(
        candidates,
        normalized_observations,
    )
    candidates = _apply_observation_feedback(
        candidates,
        normalized_observations,
        incumbent_scores_by_axis=axis_incumbents,
    )

    beliefs = _merge_family_beliefs(candidates, family_beliefs)
    try:
        ranked = rank_exact_eval_candidates(
            candidates,
            incumbent_score=incumbent,
            family_beliefs=beliefs,
            source=TOOL,
            expected_improvement_weight=expected_improvement_weight,
            information_gain_weight=information_gain_weight,
            top_k=top_k,
        )
    except BayesianExperimentalDesignError as exc:
        raise CrossFamilyCandidatePortfolioError(str(exc)) from exc

    candidates_by_id = {row["candidate_id"]: row for row in candidates}
    ranked_rows = _enforce_portfolio_false_authority(ranked["rows"])
    for row in ranked_rows:
        source_row = candidates_by_id.get(str(row["candidate_id"]), {})
        for key in (
            "source_kind",
            "source_rank",
            "source_artifact_path",
            "source_dispatch_blockers",
            "source_metadata",
            "prediction_source",
        ):
            if key in source_row:
                row[key] = source_row[key]
        if row.get("source_dispatch_blockers"):
            row["dispatch_blockers"] = sorted(
                set(row.get("dispatch_blockers") or [])
                | set(row.get("source_dispatch_blockers") or [])
            )
        row["operator_next_action"] = _operator_action_for_row(row)
        row["operator_action_priority"] = list(_operator_action_priority(row))
    operator_action_rows = _build_operator_action_rows(ranked_rows)

    source_counts: dict[str, int] = {}
    for candidate in candidates:
        kind = str(candidate["source_kind"])
        source_counts[kind] = source_counts.get(kind, 0) + 1

    custody_ready_count = sum(
        1 for row in ranked_rows if row.get("exact_archive_custody_ready") is True
    )
    aggregate_blockers = sorted(
        {
            blocker
            for row in ranked_rows
            for blocker in row.get("dispatch_blockers", [])
        }
        | {
            "portfolio_planning_only_requires_separate_lane_claim",
            "exact_eval_dispatch_requires_claimed_operator_action",
        }
    )
    materialization_required_count = sum(
        1
        for row in ranked_rows
        if isinstance(row.get("exact_archive_custody"), Mapping)
        and row["exact_archive_custody"].get("verified") is not True
    )
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "allowed_use": (
            "cross_family_exact_eval_spend_triage_planning_only_no_score_or_dispatch_authority"
        ),
        "incumbent_exact_cuda_score": round(incumbent, 12),
        "cross_family_policy": {
            "cross_family": True,
            "cross_paradigm": True,
            "outside_class_allowed": True,
            "exact_eval_candidate_queue_requires_separate_claim": True,
            "ranker_ready_flag_is_advisory_until_portfolio_gate": True,
            **FALSE_AUTHORITY,
        },
        "acquisition_weights": {
            "expected_improvement": round(float(expected_improvement_weight), 12),
            "expected_information_gain": round(float(information_gain_weight), 12),
        },
        "family_prior_policy": {
            "default_family_beliefs_are_weak_planning_priors": True,
            "caller_family_beliefs_override_defaults": family_beliefs is not None,
        },
        "source_artifacts": dict(source_artifacts or {}),
        "incumbent_scores_by_axis": {
            axis: round(score, 12) for axis, score in sorted(axis_incumbents.items())
        },
        "observation_feedback": {
            **summarize_observations(normalized_observations),
            "candidate_suppression_policy": (
                "exact-axis same-candidate observations demote repeat operator actions"
            ),
            "pairset_observation_response_model": pairset_observation_response_model,
        },
        "portfolio_summary": {
            "candidate_count_before_top_k": len(candidates),
            "ranked_candidate_count": len(ranked_rows),
            "operator_action_candidate_count": len(operator_action_rows),
            "recommended_next_candidate_id": (
                operator_action_rows[0]["candidate_id"] if operator_action_rows else None
            ),
            "recommended_next_action": (
                operator_action_rows[0]["operator_next_action"]
                if operator_action_rows
                else None
            ),
            "candidate_archive_custody_ready_count": custody_ready_count,
            "materialization_required_count": materialization_required_count,
            "source_counts": dict(sorted(source_counts.items())),
            "observation_row_count": len(normalized_observations),
            "observed_candidate_count": len(
                {
                    str(row.get("candidate_id") or "")
                    for row in normalized_observations
                    if _observation_axis_is_exact(row)
                }
            ),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "dispatch_blockers": aggregate_blockers,
        "bayesian_ranker_payload": {
            key: ranked[key]
            for key in (
                "schema_version",
                "tool",
                "source",
                "acquisition_formula",
                "expected_improvement_formula",
                "information_gain_formula",
                "family_beliefs",
                "dispatch_blockers",
            )
        },
        "ranked_rows": ranked_rows,
        "operator_action_rows": operator_action_rows,
    }


def render_cross_family_candidate_portfolio_markdown(
    portfolio: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing portfolio report."""

    summary = portfolio.get("portfolio_summary") if isinstance(portfolio, Mapping) else {}
    if not isinstance(summary, Mapping):
        summary = {}
    lines = [
        "# Cross-Family Candidate Portfolio",
        "",
        f"- Schema: `{portfolio.get('schema')}`",
        f"- Incumbent exact CUDA score: `{portfolio.get('incumbent_exact_cuda_score')}`",
        f"- Score claim: `{portfolio.get('score_claim')}`",
        f"- Ready for exact eval dispatch: `{portfolio.get('ready_for_exact_eval_dispatch')}`",
        f"- Ranked candidates: `{summary.get('ranked_candidate_count')}`",
        f"- Recommended next candidate: `{summary.get('recommended_next_candidate_id')}`",
        f"- Recommended next action: `{summary.get('recommended_next_action')}`",
        f"- Custody-ready archives: `{summary.get('candidate_archive_custody_ready_count')}`",
        "",
        "## Operator Action Queue",
        "",
        "| action rank | bayes rank | candidate | family | source | action | acquisition | archive ready |",
        "|---:|---:|---|---|---|---|---:|---|",
    ]
    for row in portfolio.get("operator_action_rows", [])[:32]:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {action_rank} | {rank} | `{candidate}` | `{family}` | `{source}` | "
            "`{action}` | {acq:.12g} | `{ready}` |".format(
                action_rank=row.get("operator_action_rank"),
                rank=row.get("rank"),
                candidate=row.get("candidate_id"),
                family=row.get("family_id"),
                source=row.get("source_kind"),
                action=row.get("operator_next_action"),
                acq=float(row.get("acquisition_value", 0.0)),
                ready=row.get("exact_archive_custody_ready"),
            )
        )
    lines.extend(
        [
            "",
            "## Bayesian Rank",
            "",
        "| rank | candidate | family | source | acquisition | mean | variance | archive ready |",
        "|---:|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in portfolio.get("ranked_rows", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {rank} | `{candidate}` | `{family}` | `{source}` | {acq:.12g} | "
            "{mean:.12g} | {variance:.12g} | `{ready}` |".format(
                rank=row.get("rank"),
                candidate=row.get("candidate_id"),
                family=row.get("family_id"),
                source=row.get("source_kind"),
                acq=float(row.get("acquisition_value", 0.0)),
                mean=float(row.get("predicted_score_mean", 0.0)),
                variance=float(row.get("predicted_score_variance", 0.0)),
                ready=row.get("exact_archive_custody_ready"),
            )
        )
    lines.extend(
        [
            "",
            "## Dispatch Blockers",
            "",
        ]
    )
    for blocker in portfolio.get("dispatch_blockers", []):
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(payload), encoding="utf-8")


__all__ = [
    "CrossFamilyCandidatePortfolioError",
    "build_cross_family_candidate_portfolio",
    "render_cross_family_candidate_portfolio_markdown",
    "source_artifacts_from_paths",
    "write_json",
]
