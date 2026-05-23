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
from tac.optimization.pairset_component_marginal import (
    PAIRSET_COMPONENT_MARGINAL_MODEL_SCHEMA,
    build_component_score_delta_payload,
    canonical_signal_refs,
    component_marginal_status,
    component_score_delta,
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
        if row.get("full_video_denominator") != 600:
            raise CrossFamilyCandidatePortfolioError(
                f"{candidate_id}.full_video_denominator must be 600"
            )
        observed_delta = _finite_float(
            row.get("projected_full_video_delta_vs_baseline_score"),
            label=f"{candidate_id}.projected_full_video_delta_vs_baseline_score",
        )
        normalized_gain = _finite_float(
            row.get("normalized_full_video_scorer_gain_vs_baseline"),
            label=f"{candidate_id}.normalized_full_video_scorer_gain_vs_baseline",
        )
        normalized_margin = _finite_float(
            row.get("normalized_full_video_byte_budget_margin_vs_break_even"),
            label=f"{candidate_id}.normalized_full_video_byte_budget_margin_vs_break_even",
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
                    "projected_full_video_delta_vs_baseline_score": observed_delta,
                    "normalized_full_video_scorer_gain_vs_baseline": normalized_gain,
                    "predicted_delta_vs_baseline_score": predicted_delta,
                    "normalized_full_video_byte_budget_margin_vs_break_even": normalized_margin,
                    "planning_value_scope": "normalized_full_video",
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
                    "acquisition_operation": row.get("acquisition_operation"),
                    "acquisition_score": row.get("acquisition_score"),
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


def _candidate_selector_kind(row: Mapping[str, Any]) -> str:
    metadata = row.get("source_metadata")
    raw = row.get("selector_kind")
    if raw is None and isinstance(metadata, Mapping):
        raw = metadata.get("selector_kind")
    text = str(raw or "").strip()
    return text or "unknown_selector"


def _candidate_selected_pair_indices(row: Mapping[str, Any]) -> list[int] | None:
    metadata = row.get("source_metadata")
    raw = row.get("selected_pair_indices")
    if raw is None and isinstance(metadata, Mapping):
        raw = metadata.get("selected_pair_indices")
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise CrossFamilyCandidatePortfolioError(
            f"{row.get('candidate_id')}.selected_pair_indices must be a sequence"
        )
    out: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            raise CrossFamilyCandidatePortfolioError(
                f"{row.get('candidate_id')}.selected_pair_indices values must be integers"
            )
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise CrossFamilyCandidatePortfolioError(
                f"{row.get('candidate_id')}.selected_pair_indices values must be integers"
            ) from exc
        out.append(parsed)
    return out


def _candidate_acquisition_operation(row: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = row.get("source_metadata")
    raw = row.get("acquisition_operation")
    if raw is None and isinstance(metadata, Mapping):
        raw = metadata.get("acquisition_operation")
    if not isinstance(raw, Mapping):
        return None
    return dict(raw)


def _observation_component_deltas(row: Mapping[str, Any]) -> dict[str, float] | None:
    raw = row.get("component_deltas", row.get("component_axis_deltas"))
    source: Mapping[str, Any] = raw if isinstance(raw, Mapping) else row
    out: dict[str, float] = {}
    for key in ("segnet_delta", "posenet_delta", "rate_delta"):
        value = source.get(key, row.get(key))
        if value is None:
            return None
        out[key] = _finite_float(value, label=f"{row.get('candidate_id')}.{key}")
    return out


def _component_net_delta(component_deltas: Mapping[str, float]) -> float:
    return component_score_delta(
        segnet_delta=float(component_deltas.get("segnet_delta", 0.0)),
        posenet_delta=float(component_deltas.get("posenet_delta", 0.0)),
        rate_delta=float(component_deltas.get("rate_delta", 0.0)),
    )


def _score_delta_status(delta: float | None) -> str:
    if delta is None:
        return "axis_baseline_missing"
    if delta < 0.0:
        return "improves_vs_axis_baseline"
    if delta > 0.0:
        return "regresses_vs_axis_baseline"
    return "ties_axis_baseline"


def _component_marginal_status(component_deltas: Mapping[str, float]) -> str:
    return component_marginal_status(
        segnet_delta=float(component_deltas.get("segnet_delta", 0.0)),
        posenet_delta=float(component_deltas.get("posenet_delta", 0.0)),
        rate_delta=float(component_deltas.get("rate_delta", 0.0)),
    )


def _transfer_status(axis_statuses: Mapping[str, str]) -> str:
    cpu = axis_statuses.get("contest_cpu")
    cuda = axis_statuses.get("contest_cuda")
    if cpu == "improves_vs_axis_baseline" and cuda == "regresses_vs_axis_baseline":
        return "cpu_improves_cuda_regresses"
    if cpu == "regresses_vs_axis_baseline" and cuda == "improves_vs_axis_baseline":
        return "cpu_regresses_cuda_improves"
    if cpu == cuda and cpu is not None:
        return f"same_status_{cpu}"
    if cpu is None or cuda is None:
        return "single_axis_only"
    return "mixed_axis_status"


def _build_pairset_component_marginal_model(
    candidates: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    incumbent_scores_by_axis: Mapping[str, float],
) -> dict[str, Any]:
    """Canonicalize pair/frame component deltas for future acquisition logic."""

    candidates_by_id = {
        str(row.get("candidate_id") or ""): row
        for row in candidates
        if str(row.get("source_kind") or "") == "decoder_q_pairset_acquisition"
    }
    identity_counts = {
        "candidate_id_match_count": 0,
        "selected_pair_indices_missing_candidate_count": 0,
        "selected_pair_indices_verified_count": 0,
        "selected_pair_indices_missing_observation_count": 0,
        "selected_pair_indices_mismatch_count": 0,
    }
    training_rows: list[dict[str, Any]] = []
    for observation in observations:
        axis = _exact_observation_axis(observation)
        if axis is None:
            continue
        candidate_id = str(observation.get("candidate_id") or "")
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            continue
        identity_counts["candidate_id_match_count"] += 1
        observed_indices = _candidate_selected_pair_indices(observation)
        candidate_indices = _candidate_selected_pair_indices(candidate)
        if candidate_indices is None:
            identity_counts["selected_pair_indices_missing_candidate_count"] += 1
            continue
        if observed_indices is None:
            identity_counts["selected_pair_indices_missing_observation_count"] += 1
            continue
        if observed_indices != candidate_indices:
            identity_counts["selected_pair_indices_mismatch_count"] += 1
            continue
        operation = _candidate_acquisition_operation(candidate)
        component_deltas = _observation_component_deltas(observation)
        if operation is None or component_deltas is None:
            continue
        identity_counts["selected_pair_indices_verified_count"] += 1
        observed_score = _finite_float(
            observation.get("observed_score_or_delta"),
            label=f"{candidate_id}.observed_score_or_delta",
        )
        baseline = incumbent_scores_by_axis.get(axis)
        observed_delta = None if baseline is None else observed_score - baseline
        training_rows.append(
            {
                "candidate_id": candidate_id,
                "axis": axis,
                "selector_kind": _candidate_selector_kind(candidate),
                "selected_pair_count": _candidate_selected_pair_count(candidate),
                "selected_pair_indices": observed_indices,
                "operation": operation,
                "component_deltas": component_deltas,
                "net_component_delta": _component_net_delta(component_deltas),
                "component_marginal_status": _component_marginal_status(
                    component_deltas
                ),
                "observed_score": observed_score,
                "axis_baseline_score": baseline,
                "observed_delta_vs_axis_baseline": observed_delta,
                "score_delta_status": _score_delta_status(observed_delta),
                "observed_at_utc": observation.get("observed_at_utc"),
            }
        )

    base = {
        "schema": PAIRSET_COMPONENT_MARGINAL_MODEL_SCHEMA,
        "active": bool(training_rows),
        "training_row_count": len(training_rows),
        "identity_policy": "candidate_id_and_selected_pair_indices_required_and_matched",
        "identity_counts": dict(sorted(identity_counts.items())),
        "canonical_signal_refs": canonical_signal_refs(),
        **FALSE_AUTHORITY,
        "allowed_use": "component_marginal_planning_signal_only_no_score_or_dispatch_authority",
    }
    if not training_rows:
        return {
            **base,
            "active": False,
            "inactive_reason": "no_exact_pairset_observations_with_component_deltas",
        }

    axis_models: dict[str, dict[str, Any]] = {}
    rows_by_axis: dict[str, list[dict[str, Any]]] = {}
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in training_rows:
        rows_by_axis.setdefault(str(row["axis"]), []).append(row)
        rows_by_candidate.setdefault(str(row["candidate_id"]), []).append(row)

    for axis, axis_rows in sorted(rows_by_axis.items()):
        drop_one_rows = [
            row
            for row in axis_rows
            if str(row["operation"].get("op") or "") == "drop_one"
        ]
        drop_one_marginals: list[dict[str, Any]] = []
        for row in drop_one_rows:
            operation = row["operation"]
            pair = int(operation["dropped_pair_index"])
            rank = int(operation["dropped_pair_rank"])
            component_deltas = dict(row["component_deltas"])
            scorer_penalty = component_deltas["segnet_delta"] + component_deltas["posenet_delta"]
            rate_credit = -component_deltas["rate_delta"]
            component_payload = build_component_score_delta_payload(
                candidate_id=str(row["candidate_id"]),
                axis=axis,
                pair_index=pair,
                dropped_pair_rank=rank,
                segnet_delta=component_deltas["segnet_delta"],
                posenet_delta=component_deltas["posenet_delta"],
                rate_delta=component_deltas["rate_delta"],
            )
            drop_one_marginals.append(
                {
                    "pair_index": pair,
                    "dropped_pair_rank": rank,
                    "candidate_id": row["candidate_id"],
                    "observed_score": row["observed_score"],
                    "axis_baseline_score": row["axis_baseline_score"],
                    "observed_delta_vs_axis_baseline": row[
                        "observed_delta_vs_axis_baseline"
                    ],
                    "component_deltas": component_deltas,
                    "scorer_penalty": scorer_penalty,
                    "rate_credit": rate_credit,
                    "net_component_delta": row["net_component_delta"],
                    "score_delta_status": row["score_delta_status"],
                    "component_marginal_status": row["component_marginal_status"],
                    "component_score_delta_payload": component_payload,
                    **FALSE_AUTHORITY,
                }
            )
        drop_two_rows = [
            row
            for row in axis_rows
            if str(row["operation"].get("op") or "") == "drop_two"
        ]
        axis_models[axis] = {
            "axis": axis,
            "training_row_count": len(axis_rows),
            "drop_one_training_row_count": len(drop_one_rows),
            "drop_two_training_row_count": len(drop_two_rows),
            "drop_one_pair_marginals": sorted(
                drop_one_marginals,
                key=lambda row: (
                    float(row["net_component_delta"]),
                    int(row["dropped_pair_rank"]),
                    int(row["pair_index"]),
                ),
            ),
            "drop_two_interactions": [
                {
                    "candidate_id": row["candidate_id"],
                    "dropped_pair_indices": row["operation"].get(
                        "dropped_pair_indices"
                    ),
                    "dropped_pair_ranks": row["operation"].get("dropped_pair_ranks"),
                    "observed_score": row["observed_score"],
                    "axis_baseline_score": row["axis_baseline_score"],
                    "observed_delta_vs_axis_baseline": row[
                        "observed_delta_vs_axis_baseline"
                    ],
                    "component_deltas": row["component_deltas"],
                    "net_component_delta": row["net_component_delta"],
                    "score_delta_status": row["score_delta_status"],
                    "component_marginal_status": row["component_marginal_status"],
                    **FALSE_AUTHORITY,
                }
                for row in drop_two_rows
            ],
            "safe_drop_pair_indices": [
                int(row["pair_index"])
                for row in drop_one_marginals
                if row["component_marginal_status"]
                == "rate_credit_exceeds_scorer_penalty"
            ],
            "protected_drop_pair_indices": [
                int(row["pair_index"])
                for row in drop_one_marginals
                if row["component_marginal_status"]
                == "scorer_penalty_exceeds_rate_credit"
            ],
            **FALSE_AUTHORITY,
        }

    cross_axis: list[dict[str, Any]] = []
    for candidate_id, candidate_rows in sorted(rows_by_candidate.items()):
        axes = sorted({str(row["axis"]) for row in candidate_rows})
        if len(axes) < 2:
            continue
        by_axis = {str(row["axis"]): row for row in candidate_rows}
        axis_statuses = {
            axis: str(by_axis[axis]["score_delta_status"])
            for axis in axes
        }
        cross_axis.append(
            {
                "candidate_id": candidate_id,
                "axes": axes,
                "axis_statuses": axis_statuses,
                "transfer_status": _transfer_status(axis_statuses),
                "component_deltas_by_axis": {
                    axis: by_axis[axis]["component_deltas"]
                    for axis in axes
                },
                "observed_delta_vs_axis_baseline_by_axis": {
                    axis: by_axis[axis]["observed_delta_vs_axis_baseline"]
                    for axis in axes
                },
                **FALSE_AUTHORITY,
            }
        )

    return {
        **base,
        "active": True,
        "model_kind": "exact_axis_component_marginal_ledger",
        "axes": sorted(axis_models),
        "axis_models": axis_models,
        "cross_axis_transfer_diagnostics": cross_axis,
    }


def _nearest_drop_one_marginal(
    axis_model: Mapping[str, Any],
    *,
    dropped_pair_index: int,
    dropped_pair_rank: int,
) -> dict[str, Any] | None:
    rows = axis_model.get("drop_one_pair_marginals")
    if not isinstance(rows, list) or not rows:
        return None
    candidates = [row for row in rows if isinstance(row, Mapping)]
    if not candidates:
        return None
    nearest = min(
        candidates,
        key=lambda row: (
            abs(int(row.get("dropped_pair_rank", 0)) - dropped_pair_rank),
            abs(int(row.get("pair_index", 0)) - dropped_pair_index),
            str(row.get("candidate_id") or ""),
        ),
    )
    return {
        "axis": axis_model.get("axis"),
        "source_candidate_id": nearest.get("candidate_id"),
        "source_pair_index": nearest.get("pair_index"),
        "source_dropped_pair_rank": nearest.get("dropped_pair_rank"),
        "rank_distance": abs(
            int(nearest.get("dropped_pair_rank", 0)) - dropped_pair_rank
        ),
        "pair_index_distance": abs(
            int(nearest.get("pair_index", 0)) - dropped_pair_index
        ),
        "source_net_component_delta": nearest.get("net_component_delta"),
        "source_component_marginal_status": nearest.get("component_marginal_status"),
        "source_score_delta_status": nearest.get("score_delta_status"),
        **FALSE_AUTHORITY,
    }


def _component_delta_status_from_net_delta(delta: float) -> str:
    if delta < 0.0:
        return "rate_credit_expected_to_exceed_scorer_penalty"
    if delta > 0.0:
        return "scorer_penalty_expected_to_exceed_rate_credit"
    return "rate_credit_expected_to_tie_scorer_penalty"


def _axis_component_marginal_action_prior(
    axis_model: Mapping[str, Any],
    *,
    dropped_pair_index: int,
    dropped_pair_rank: int,
) -> dict[str, Any] | None:
    rows = axis_model.get("drop_one_pair_marginals")
    if not isinstance(rows, list) or not rows:
        return None
    weighted_sum = 0.0
    weight_total = 0.0
    evidence_rows: list[dict[str, Any]] = []
    exact_status = None
    same_pair_observed_statuses: list[str] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        rank = int(raw.get("dropped_pair_rank", 0))
        pair = int(raw.get("pair_index", 0))
        rank_distance = abs(rank - dropped_pair_rank)
        pair_index_distance = abs(pair - dropped_pair_index)
        net_delta = _finite_float(
            raw.get("net_component_delta"),
            label=f"{raw.get('candidate_id')}.net_component_delta",
        )
        weight = 1.0 / (1.0 + float(rank_distance))
        weighted_sum += weight * net_delta
        weight_total += weight
        status = str(raw.get("component_marginal_status") or "")
        if rank_distance == 0 and pair_index_distance == 0:
            exact_status = status
        if pair_index_distance == 0 and status:
            same_pair_observed_statuses.append(status)
        evidence_rows.append(
            {
                "source_candidate_id": raw.get("candidate_id"),
                "source_pair_index": pair,
                "source_dropped_pair_rank": rank,
                "rank_distance": rank_distance,
                "pair_index_distance": pair_index_distance,
                "source_net_component_delta": net_delta,
                "source_component_marginal_status": status,
                **FALSE_AUTHORITY,
            }
        )
    if weight_total <= 0.0:
        return None
    expected_delta = weighted_sum / weight_total
    return {
        "axis": axis_model.get("axis"),
        "evidence_count": len(evidence_rows),
        "weighting": "inverse_1_plus_rank_distance",
        "expected_net_component_delta": expected_delta,
        "expected_component_marginal_status": _component_delta_status_from_net_delta(
            expected_delta
        ),
        "exact_observed_component_marginal_status": exact_status,
        "same_pair_observed_component_marginal_statuses": sorted(
            set(same_pair_observed_statuses)
        ),
        "evidence_rows": sorted(
            evidence_rows,
            key=lambda row: (
                int(row["rank_distance"]),
                int(row["pair_index_distance"]),
                str(row.get("source_candidate_id") or ""),
            ),
        ),
        **FALSE_AUTHORITY,
    }


def _component_marginal_action_prior(
    axis_models: Mapping[str, Any],
    *,
    dropped_pair_index: int,
    dropped_pair_rank: int,
) -> dict[str, Any] | None:
    by_axis = {
        str(axis): _axis_component_marginal_action_prior(
            axis_model,
            dropped_pair_index=dropped_pair_index,
            dropped_pair_rank=dropped_pair_rank,
        )
        for axis, axis_model in axis_models.items()
        if isinstance(axis_model, Mapping)
    }
    by_axis = {axis: value for axis, value in by_axis.items() if value is not None}
    if not by_axis:
        return None
    primary_axis = "contest_cpu" if "contest_cpu" in by_axis else sorted(by_axis)[0]
    primary = by_axis[primary_axis]
    blockers: list[str] = []
    primary_statuses = set(
        primary.get("same_pair_observed_component_marginal_statuses") or []
    )
    if (
        primary.get("exact_observed_component_marginal_status")
        == "scorer_penalty_exceeds_rate_credit"
        or "scorer_penalty_exceeds_rate_credit" in primary_statuses
    ):
        blockers.append("component_marginal_exact_axis_protected_pair")
    return {
        "schema": "pairset_component_marginal_action_prior.v1",
        "primary_axis": primary_axis,
        "primary_axis_expected_net_component_delta": primary[
            "expected_net_component_delta"
        ],
        "primary_axis_expected_component_marginal_status": primary[
            "expected_component_marginal_status"
        ],
        "axis_priors": by_axis,
        "planning_blockers": blockers,
        "allowed_use": (
            "operator_action_sorting_only_requires_local_controls_and_exact_auth_axis_anchor"
        ),
        **FALSE_AUTHORITY,
    }


def _apply_pairset_component_marginal_feedback(
    candidates: Sequence[Mapping[str, Any]],
    component_model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if component_model.get("active") is not True:
        return [dict(row) for row in candidates]
    axis_models = component_model.get("axis_models")
    if not isinstance(axis_models, Mapping):
        return [dict(row) for row in candidates]
    updated: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        if str(row.get("source_kind") or "") != "decoder_q_pairset_acquisition":
            updated.append(row)
            continue
        operation = _candidate_acquisition_operation(row)
        if not operation or str(operation.get("op") or "") != "drop_one":
            updated.append(row)
            continue
        dropped_pair = int(operation["dropped_pair_index"])
        dropped_rank = int(operation["dropped_pair_rank"])
        nearest = {
            axis: _nearest_drop_one_marginal(
                axis_model,
                dropped_pair_index=dropped_pair,
                dropped_pair_rank=dropped_rank,
            )
            for axis, axis_model in axis_models.items()
            if isinstance(axis_model, Mapping)
        }
        nearest = {axis: value for axis, value in nearest.items() if value is not None}
        action_prior = _component_marginal_action_prior(
            axis_models,
            dropped_pair_index=dropped_pair,
            dropped_pair_rank=dropped_rank,
        )
        metadata = dict(row.get("source_metadata") or {})
        metadata["component_marginal_model"] = {
            "schema": "pairset_component_marginal_candidate_feedback.v1",
            "active": True,
            "model_kind": component_model.get("model_kind"),
            "dropped_pair_index": dropped_pair,
            "dropped_pair_rank": dropped_rank,
            "nearest_drop_one_evidence_by_axis": nearest,
            "component_marginal_action_prior": action_prior,
            "canonical_signal_refs": component_model.get("canonical_signal_refs"),
            "allowed_use": (
                "candidate_planning_signal_only_requires_exact_axis_materialization"
            ),
            **FALSE_AUTHORITY,
        }
        if action_prior is not None:
            blockers = set(row.get("source_dispatch_blockers") or [])
            blockers.update(action_prior.get("planning_blockers") or [])
            if blockers:
                row["source_dispatch_blockers"] = sorted(blockers)
        row["source_metadata"] = metadata
        updated.append(row)
    return updated


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
) -> tuple[str | None, list[dict[str, Any]], dict[str, int], dict[str, int]]:
    candidates_by_id = {
        str(row.get("candidate_id") or ""): row
        for row in candidates
        if str(row.get("source_kind") or "") == "decoder_q_pairset_acquisition"
    }
    rows_by_axis: dict[str, list[dict[str, Any]]] = {}
    axis_counts: dict[str, int] = {}
    identity_counts = {
        "candidate_id_match_count": 0,
        "selected_pair_indices_missing_candidate_count": 0,
        "selected_pair_indices_verified_count": 0,
        "selected_pair_indices_missing_observation_count": 0,
        "selected_pair_indices_mismatch_count": 0,
    }
    for observation in observations:
        axis = _exact_observation_axis(observation)
        if axis is None:
            continue
        candidate_id = str(observation.get("candidate_id") or "")
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            continue
        identity_counts["candidate_id_match_count"] += 1
        observed_indices = _candidate_selected_pair_indices(observation)
        candidate_indices = _candidate_selected_pair_indices(candidate)
        if candidate_indices is None:
            identity_counts["selected_pair_indices_missing_candidate_count"] += 1
            continue
        if observed_indices is None:
            identity_counts["selected_pair_indices_missing_observation_count"] += 1
            continue
        if observed_indices != candidate_indices:
            identity_counts["selected_pair_indices_mismatch_count"] += 1
            continue
        identity_counts["selected_pair_indices_verified_count"] += 1
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
                "selector_kind": _candidate_selector_kind(candidate),
                "selected_pair_count": count,
                "selected_pair_indices_verified": observed_indices is not None,
                "observed_score": score,
                "observed_at_utc": observation.get("observed_at_utc"),
            }
        )
        axis_counts[axis] = axis_counts.get(axis, 0) + 1
    if not rows_by_axis:
        return None, [], axis_counts, identity_counts
    selected_axis = sorted(
        rows_by_axis,
        key=lambda axis: (-len(rows_by_axis[axis]), axis),
    )[0]
    return selected_axis, rows_by_axis[selected_axis], axis_counts, identity_counts


def _apply_pairset_observation_response_model(
    candidates: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    incumbent_scores_by_axis: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use exact pairset observations to recalibrate unobserved pairset priors.

    This is a planning prior only: it updates predicted means/variances for
    ordering future local controls and exact-eval spend, while leaving all
    authority fields false.
    """

    axis, training_rows, axis_counts, identity_counts = _pairset_observation_training_rows(
        candidates,
        observations,
    )
    base_summary: dict[str, Any] = {
        "schema": "pairset_observation_response_model.v1",
        "active": False,
        "axis": axis,
        "axis_observation_counts": dict(sorted(axis_counts.items())),
        "training_row_count": len(training_rows),
        "identity_policy": "candidate_id_and_selected_pair_indices_required_and_matched",
        "identity_counts": dict(sorted(identity_counts.items())),
        **FALSE_AUTHORITY,
    }
    rows_by_selector: dict[str, list[dict[str, Any]]] = {}
    for row in training_rows:
        rows_by_selector.setdefault(str(row["selector_kind"]), []).append(row)
    selector_models: dict[str, dict[str, Any]] = {}
    inactive_selector_reasons: dict[str, str] = {}
    axis_incumbent = incumbent_scores_by_axis.get(axis or "")
    for selector_kind, selector_rows in sorted(rows_by_selector.items()):
        xs = [float(row["selected_pair_count"]) for row in selector_rows]
        ys = [float(row["observed_score"]) for row in selector_rows]
        fit = _fit_line(xs, ys)
        if fit is None:
            inactive_selector_reasons[selector_kind] = "need_two_distinct_selected_pair_counts"
            continue
        best_observed = min(ys)
        regression_only = axis_incumbent is not None and all(score >= axis_incumbent for score in ys)
        selector_models[selector_kind] = {
            "selector_kind": selector_kind,
            "training_row_count": len(selector_rows),
            "observed_candidate_ids": sorted({str(row["candidate_id"]) for row in selector_rows}),
            "observed_selected_pair_counts": sorted(
                {int(row["selected_pair_count"]) for row in selector_rows}
            ),
            "intercept": float(fit["intercept"]),
            "slope_per_pair": float(fit["slope_per_pair"]),
            "residual_mse": max(0.0, float(fit["residual_mse"])),
            "best_observed_score": best_observed,
            "axis_incumbent_score": axis_incumbent,
            "regression_only_cap_active": regression_only,
            "selected_pair_indices_verified_count": sum(
                1 for row in selector_rows if row.get("selected_pair_indices_verified") is True
            ),
        }
    if not selector_models:
        return [dict(row) for row in candidates], {
            **base_summary,
            "inactive_reason": "need_two_distinct_selected_pair_counts_per_selector_kind",
            "inactive_selector_reasons": inactive_selector_reasons,
        }

    blend_weight = min(0.85, len(training_rows) / float(len(training_rows) + 3))
    updated: list[dict[str, Any]] = []
    updated_count = 0
    for candidate in candidates:
        row = dict(candidate)
        if str(row.get("source_kind") or "") != "decoder_q_pairset_acquisition":
            updated.append(row)
            continue
        selector_kind = _candidate_selector_kind(row)
        selector_model = selector_models.get(selector_kind)
        if selector_model is None:
            metadata = dict(row.get("source_metadata") or {})
            metadata["observation_response_model"] = {
                **base_summary,
                "active": False,
                "selector_kind": selector_kind,
                "inactive_reason": "no_exact_observations_for_selector_kind",
                "available_selector_kinds": sorted(selector_models),
                "allowed_use": "planning_prior_only_no_score_or_dispatch_authority",
            }
            row["source_metadata"] = metadata
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
        model_mean_raw = selector_model["intercept"] + selector_model["slope_per_pair"] * float(count)
        model_mean = model_mean_raw
        if selector_model["regression_only_cap_active"]:
            model_mean = max(model_mean, float(selector_model["best_observed_score"]))
        posterior_mean = (1.0 - blend_weight) * prior_mean + blend_weight * model_mean
        if selector_model["regression_only_cap_active"]:
            posterior_mean = max(posterior_mean, float(selector_model["best_observed_score"]))
        observed_counts = list(selector_model["observed_selected_pair_counts"])
        nearest_count = min(observed_counts, key=lambda value: abs(value - count))
        distance = abs(float(count - nearest_count))
        distance_variance = (abs(float(selector_model["slope_per_pair"])) * distance * 0.25) ** 2
        prior_variance = _finite_float(
            row.get("predicted_score_variance", DEFAULT_PAIRSET_VARIANCE),
            label=f"{row.get('candidate_id')}.predicted_score_variance",
        )
        posterior_variance = max(
            DEFAULT_PAIRSET_OBSERVATION_MODEL_VARIANCE_FLOOR,
            float(selector_model["residual_mse"]),
            distance_variance,
            prior_variance * 0.1,
        )
        metadata = dict(row.get("source_metadata") or {})
        metadata["observation_response_model"] = {
            **base_summary,
            "active": True,
            "model_kind": "linear_selected_pair_count_exact_axis",
            "selector_kind": selector_kind,
            "observed_candidate_ids": selector_model["observed_candidate_ids"],
            "observed_selected_pair_counts": observed_counts,
            "intercept": round(float(selector_model["intercept"]), 15),
            "slope_per_pair": round(float(selector_model["slope_per_pair"]), 15),
            "residual_mse": selector_model["residual_mse"],
            "prior_predicted_score_mean": prior_mean,
            "model_predicted_score_mean_raw": model_mean_raw,
            "model_predicted_score_mean": model_mean,
            "best_observed_score": selector_model["best_observed_score"],
            "axis_incumbent_score": selector_model["axis_incumbent_score"],
            "regression_only_cap_active": selector_model["regression_only_cap_active"],
            "posterior_blend_weight": blend_weight,
            "nearest_observed_selected_pair_count": nearest_count,
            "selected_pair_count_distance": distance,
            "posterior_score_variance": posterior_variance,
            "prediction_axis": axis,
            "allowed_use": "planning_prior_only_no_score_or_dispatch_authority",
        }
        row["source_metadata"] = metadata
        row["predicted_score_mean"] = posterior_mean
        row["predicted_score_variance"] = posterior_variance
        row["prediction_source"] = "exact_pairset_observation_response_model_planning_prior"
        row["prediction_axis"] = axis
        updated_count += 1
        updated.append(row)

    primary_selector = sorted(
        selector_models,
        key=lambda key: (-int(selector_models[key]["training_row_count"]), key),
    )[0]
    primary_model = selector_models[primary_selector]
    return updated, {
        **base_summary,
        "active": True,
        "model_kind": "linear_selected_pair_count_exact_axis",
        "selector_family_scoped": True,
        "updated_candidate_count": updated_count,
        "active_selector_kinds": sorted(selector_models),
        "inactive_selector_reasons": inactive_selector_reasons,
        "primary_selector_kind": primary_selector,
        "observed_candidate_ids": primary_model["observed_candidate_ids"],
        "observed_selected_pair_counts": primary_model["observed_selected_pair_counts"],
        "intercept": round(float(primary_model["intercept"]), 15),
        "slope_per_pair": round(float(primary_model["slope_per_pair"]), 15),
        "residual_mse": primary_model["residual_mse"],
        "best_observed_score": primary_model["best_observed_score"],
        "axis_incumbent_score": primary_model["axis_incumbent_score"],
        "regression_only_cap_active": primary_model["regression_only_cap_active"],
        "selector_models": {
            key: {
                **value,
                "intercept": round(float(value["intercept"]), 15),
                "slope_per_pair": round(float(value["slope_per_pair"]), 15),
            }
            for key, value in sorted(selector_models.items())
        },
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


def _component_marginal_priority_value(row: Mapping[str, Any]) -> float | None:
    metadata = row.get("source_metadata")
    if not isinstance(metadata, Mapping):
        return None
    feedback = metadata.get("component_marginal_model")
    if not isinstance(feedback, Mapping):
        return None
    action_prior = feedback.get("component_marginal_action_prior")
    if not isinstance(action_prior, Mapping):
        return None
    value = action_prior.get("primary_axis_expected_net_component_delta")
    if value is None:
        return None
    return _finite_float(
        value,
        label=f"{row.get('candidate_id')}.component_marginal_priority",
    )


def _operator_action_priority(row: Mapping[str, Any]) -> tuple[Any, ...]:
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
    if source_kind == "decoder_q_pairset_acquisition" and source_blocked:
        base += 50
    source_rank = row.get("source_rank")
    rank_value = int(source_rank) if source_rank is not None else 999_999
    acquisition_value = -float(row.get("acquisition_value", 0.0))
    component_priority = _component_marginal_priority_value(row)
    if source_kind == "decoder_q_pairset_acquisition" and component_priority is not None:
        return (
            base,
            0,
            component_priority,
            rank_value,
            acquisition_value,
            str(row.get("candidate_id") or ""),
        )
    if source_kind == "decoder_q_pairset_acquisition" and str(row.get("prediction_source") or "") == (
        "exact_pairset_observation_response_model_planning_prior"
    ):
        return (
            base,
            1,
            acquisition_value,
            rank_value,
            str(row.get("candidate_id") or ""),
        )
    return (
        base,
        1,
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
        incumbent_scores_by_axis=axis_incumbents,
    )
    pairset_component_marginal_model = _build_pairset_component_marginal_model(
        candidates,
        normalized_observations,
        incumbent_scores_by_axis=axis_incumbents,
    )
    candidates = _apply_pairset_component_marginal_feedback(
        candidates,
        pairset_component_marginal_model,
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
            "pairset_component_marginal_model": pairset_component_marginal_model,
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
