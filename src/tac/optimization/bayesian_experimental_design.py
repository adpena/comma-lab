# SPDX-License-Identifier: MIT
"""Bayesian experimental design for exact CUDA candidate selection.

This module is planning-only. It ranks candidate archives/configs by closed-form
expected improvement and Gaussian expected information gain, then reports exact
eval readiness blockers. It never launches GPU work and never emits score
claims.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.repo_io import sha256_file

CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25 / CONTEST_ORIGINAL_BYTES
SCHEMA_VERSION = 1
DEFAULT_PRIOR_SCORE_VARIANCE = 1.0
DEFAULT_OBSERVATION_NOISE_VARIANCE = 1e-4
MIN_VARIANCE = 1e-18
_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


class BayesianExperimentalDesignError(ValueError):
    """Raised when Bayesian candidate-ranking inputs are invalid."""


def contest_score(seg_dist: float, pose_dist: float, archive_bytes: int | float) -> float:
    """Return the official contest score from deterministic components."""

    seg = float(seg_dist)
    pose = float(pose_dist)
    bytes_value = float(archive_bytes)
    if seg < 0.0:
        raise BayesianExperimentalDesignError("seg_dist must be non-negative")
    if pose < 0.0:
        raise BayesianExperimentalDesignError("pose_dist must be non-negative")
    if bytes_value < 0.0:
        raise BayesianExperimentalDesignError("archive_bytes must be non-negative")
    return 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * bytes_value / CONTEST_ORIGINAL_BYTES


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / _SQRT_2PI


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / _SQRT_2))


def expected_improvement_minimize(
    predicted_mean: float,
    predicted_variance: float,
    incumbent_score: float,
) -> float:
    """Closed-form expected improvement for minimizing a Gaussian score.

    ``incumbent_score`` must be exact CUDA evidence supplied by the caller or
    by an upstream custody artifact. This function only evaluates the formula.
    """

    mean = float(predicted_mean)
    variance = float(predicted_variance)
    incumbent = float(incumbent_score)
    if variance < 0.0:
        raise BayesianExperimentalDesignError("predicted_variance must be non-negative")
    if not math.isfinite(mean) or not math.isfinite(variance) or not math.isfinite(incumbent):
        raise BayesianExperimentalDesignError("expected improvement inputs must be finite")
    if variance <= 0.0:
        return max(incumbent - mean, 0.0)
    sigma = math.sqrt(variance)
    z = (incumbent - mean) / sigma
    return (incumbent - mean) * _normal_cdf(z) + sigma * _normal_pdf(z)


def gaussian_posterior_variance_after_family_observation(
    *,
    target_prior_variance: float,
    source_prior_variance: float,
    observation_noise_variance: float,
    coupling: float,
) -> float:
    """Return target-family posterior variance after one noisy source eval.

    The deterministic model is a joint Gaussian family prior. ``coupling`` is a
    correlation coefficient in ``[0, 1]`` between the target family and the
    family observed by the candidate's exact eval. A coupling of 1.0 on the
    source family gives the usual scalar Normal update.
    """

    target_var = float(target_prior_variance)
    source_var = float(source_prior_variance)
    noise_var = float(observation_noise_variance)
    coupling_value = float(coupling)
    if target_var < 0.0 or source_var < 0.0:
        raise BayesianExperimentalDesignError("family variances must be non-negative")
    if noise_var < 0.0:
        raise BayesianExperimentalDesignError("observation_noise_variance must be non-negative")
    if not 0.0 <= coupling_value <= 1.0:
        raise BayesianExperimentalDesignError("family coupling must be in [0, 1]")
    if target_var <= 0.0 or source_var <= 0.0 or coupling_value == 0.0:
        return target_var
    denominator = source_var + max(noise_var, MIN_VARIANCE)
    posterior = target_var * (1.0 - coupling_value * coupling_value * source_var / denominator)
    return max(0.0, posterior)


def gaussian_information_gain_nats(prior_variance: float, posterior_variance: float) -> float:
    """Return Gaussian entropy reduction in nats from variance contraction."""

    prior = float(prior_variance)
    posterior = float(posterior_variance)
    if prior < 0.0 or posterior < 0.0:
        raise BayesianExperimentalDesignError("variances must be non-negative")
    if prior <= 0.0:
        return 0.0
    if posterior >= prior:
        return 0.0
    return 0.5 * math.log(prior / max(posterior, MIN_VARIANCE))


def family_uncertainty_reduction(
    *,
    source_family_id: str,
    family_beliefs: Mapping[str, Mapping[str, Any]],
    observation_noise_variance: float,
    family_couplings: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compute expected uncertainty reduction if one family is exact-evaluated."""

    source_family = str(source_family_id)
    if source_family not in family_beliefs:
        raise BayesianExperimentalDesignError(f"missing family belief: {source_family}")
    source_prior_variance = float(family_beliefs[source_family]["prior_score_variance"])
    noise_var = float(observation_noise_variance)
    couplings: dict[str, float] = {source_family: 1.0}
    if family_couplings:
        couplings.update({str(key): float(value) for key, value in family_couplings.items()})
        couplings[source_family] = 1.0

    affected: list[dict[str, Any]] = []
    total_prior_variance = 0.0
    total_posterior_variance = 0.0
    total_information_gain = 0.0
    for target_family in sorted(couplings):
        if target_family not in family_beliefs:
            continue
        coupling = couplings[target_family]
        target_prior = float(family_beliefs[target_family]["prior_score_variance"])
        posterior = gaussian_posterior_variance_after_family_observation(
            target_prior_variance=target_prior,
            source_prior_variance=source_prior_variance,
            observation_noise_variance=noise_var,
            coupling=coupling,
        )
        information_gain = gaussian_information_gain_nats(target_prior, posterior)
        total_prior_variance += target_prior
        total_posterior_variance += posterior
        total_information_gain += information_gain
        affected.append(
            {
                "family_id": target_family,
                "coupling": round(coupling, 12),
                "prior_score_variance": round(target_prior, 12),
                "posterior_score_variance": round(posterior, 12),
                "variance_reduction": round(target_prior - posterior, 12),
                "information_gain_nats": round(information_gain, 12),
            }
        )
    return {
        "source_family_id": source_family,
        "observation_noise_variance": round(noise_var, 12),
        "affected_families": affected,
        "total_prior_score_variance": round(total_prior_variance, 12),
        "total_posterior_score_variance": round(total_posterior_variance, 12),
        "total_variance_reduction": round(total_prior_variance - total_posterior_variance, 12),
        "total_information_gain_nats": round(total_information_gain, 12),
    }


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    for key in ("candidate_id", "card_id", "atom_id", "lane_id", "id"):
        value = candidate.get(key)
        if value:
            return str(value)
    raise BayesianExperimentalDesignError("candidate missing candidate_id/card_id/atom_id")


def _family_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("family_id") or candidate.get("family") or candidate.get("family_group") or "unknown")


def _score_mean(candidate: Mapping[str, Any]) -> float:
    if "predicted_score_mean" in candidate:
        return float(candidate["predicted_score_mean"])
    if "score_mean" in candidate:
        return float(candidate["score_mean"])
    component_keys = {
        "predicted_seg_dist_mean",
        "predicted_pose_dist_mean",
    }
    if component_keys.issubset(candidate.keys()) and (
        "predicted_archive_bytes_mean" in candidate or "archive_size_bytes" in candidate
    ):
        archive_bytes = candidate.get("predicted_archive_bytes_mean", candidate.get("archive_size_bytes"))
        return contest_score(
            float(candidate["predicted_seg_dist_mean"]),
            float(candidate["predicted_pose_dist_mean"]),
            float(archive_bytes),
        )
    raise BayesianExperimentalDesignError(f"{_candidate_id(candidate)}: missing predicted score mean")


def _score_variance(candidate: Mapping[str, Any], family_prior_variance: float) -> float:
    for key in ("predicted_score_variance", "score_variance"):
        if key in candidate:
            variance = float(candidate[key])
            if variance < 0.0:
                raise BayesianExperimentalDesignError(f"{_candidate_id(candidate)}: {key} must be non-negative")
            return variance
    return max(0.0, float(family_prior_variance))


def _candidate_couplings(candidate: Mapping[str, Any]) -> dict[str, float]:
    raw = candidate.get("family_couplings") or candidate.get("related_family_couplings") or {}
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise BayesianExperimentalDesignError(f"{_candidate_id(candidate)}: family_couplings must be a mapping")
    couplings: dict[str, float] = {}
    for key, value in raw.items():
        coupling = float(value)
        if not 0.0 <= coupling <= 1.0:
            raise BayesianExperimentalDesignError(f"{_candidate_id(candidate)}: family coupling must be in [0, 1]")
        couplings[str(key)] = coupling
    return couplings


def _normalize_family_beliefs(
    candidates: Iterable[Mapping[str, Any]],
    family_beliefs: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    if isinstance(family_beliefs, Mapping):
        for family, payload in family_beliefs.items():
            belief = payload if isinstance(payload, Mapping) else {"prior_score_variance": payload}
            normalized[str(family)] = _normalize_family_belief(str(family), belief)
    elif family_beliefs is not None:
        for belief in family_beliefs:
            family = str(belief.get("family_id") or belief.get("family") or "")
            if not family:
                raise BayesianExperimentalDesignError("family belief missing family_id")
            normalized[family] = _normalize_family_belief(family, belief)

    for candidate in candidates:
        family = _family_id(candidate)
        if family in normalized:
            continue
        prior_variance = float(
            candidate.get(
                "family_prior_score_variance",
                candidate.get(
                    "family_prior_variance",
                    candidate.get("predicted_score_variance", DEFAULT_PRIOR_SCORE_VARIANCE),
                ),
            )
        )
        observation_noise = float(candidate.get("observation_noise_variance", DEFAULT_OBSERVATION_NOISE_VARIANCE))
        normalized[family] = _normalize_family_belief(
            family,
            {
                "prior_score_variance": prior_variance,
                "observation_noise_variance": observation_noise,
            },
        )
    return dict(sorted(normalized.items()))


def _normalize_family_belief(family: str, belief: Mapping[str, Any]) -> dict[str, Any]:
    prior_variance = float(
        belief.get("prior_score_variance", belief.get("prior_variance", DEFAULT_PRIOR_SCORE_VARIANCE))
    )
    observation_noise = float(
        belief.get("observation_noise_variance", belief.get("noise_variance", DEFAULT_OBSERVATION_NOISE_VARIANCE))
    )
    if prior_variance < 0.0:
        raise BayesianExperimentalDesignError(f"{family}: prior_score_variance must be non-negative")
    if observation_noise < 0.0:
        raise BayesianExperimentalDesignError(f"{family}: observation_noise_variance must be non-negative")
    return {
        "family_id": family,
        "prior_score_variance": prior_variance,
        "observation_noise_variance": observation_noise,
    }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _exact_archive_custody(candidate: Mapping[str, Any]) -> dict[str, Any]:
    raw_custody = candidate.get("exact_archive_custody")
    custody = raw_custody if isinstance(raw_custody, Mapping) else {}
    path_value = str(custody.get("archive_path") or candidate.get("exact_archive_path") or "")
    sha_value = str(custody.get("archive_sha256") or candidate.get("exact_archive_sha256") or "")
    size_value = custody.get("archive_size_bytes", candidate.get("exact_archive_size_bytes"))
    blockers: list[str] = []
    if not path_value and not sha_value and size_value is None:
        return {
            "provided": False,
            "verified": False,
            "archive_path": "",
            "archive_sha256": "",
            "archive_size_bytes": None,
            "archive_size_matches": False,
            "sha256_actual": "",
            "blockers": ["missing_exact_archive_custody"],
        }
    if not path_value:
        blockers.append("exact_archive_path_missing")
    if not _is_sha256(sha_value):
        blockers.append("exact_archive_sha256_missing_or_invalid")
    expected_size: int | None = None
    if size_value is None:
        blockers.append("exact_archive_size_bytes_missing")
    else:
        expected_size = int(size_value)
        if expected_size < 0:
            blockers.append("exact_archive_size_bytes_negative")

    path = Path(path_value) if path_value else Path()
    exists = bool(path_value and path.exists())
    is_file = bool(exists and path.is_file())
    actual_sha = ""
    actual_size: int | None = None
    size_matches = False
    if path_value and not exists:
        blockers.append("exact_archive_path_missing")
    elif exists and not is_file:
        blockers.append("exact_archive_path_not_file")
    elif is_file:
        actual_size = path.stat().st_size
        if expected_size is not None:
            size_matches = actual_size == expected_size
            if not size_matches:
                blockers.append("exact_archive_size_bytes_mismatch")
        if _is_sha256(sha_value):
            actual_sha = sha256_file(path)
            if actual_sha != sha_value:
                blockers.append("exact_archive_sha256_mismatch")
    verified = bool(is_file and expected_size is not None and size_matches and actual_sha == sha_value)
    return {
        "provided": True,
        "verified": verified,
        "archive_path": path_value,
        "archive_sha256": sha_value,
        "archive_size_bytes": expected_size,
        "archive_size_actual": actual_size,
        "archive_size_matches": size_matches,
        "sha256_actual": actual_sha,
        "blockers": blockers,
    }


def expected_design_row(
    candidate: Mapping[str, Any],
    *,
    incumbent_score: float,
    family_beliefs: Mapping[str, Mapping[str, Any]],
    expected_improvement_weight: float = 1.0,
    information_gain_weight: float = 1.0,
) -> dict[str, Any]:
    """Return one planning-only Bayesian design row for an exact eval candidate."""

    candidate_id = _candidate_id(candidate)
    family = _family_id(candidate)
    if family not in family_beliefs:
        raise BayesianExperimentalDesignError(f"{candidate_id}: missing family belief for {family}")
    family_belief = family_beliefs[family]
    predicted_mean = _score_mean(candidate)
    predicted_variance = _score_variance(candidate, float(family_belief["prior_score_variance"]))
    if predicted_mean < 0.0:
        raise BayesianExperimentalDesignError(f"{candidate_id}: predicted score mean must be non-negative")
    incumbent = float(incumbent_score)
    if incumbent < 0.0:
        raise BayesianExperimentalDesignError("incumbent_score must be non-negative")
    ei = expected_improvement_minimize(predicted_mean, predicted_variance, incumbent)
    noise_variance = float(candidate.get("observation_noise_variance", family_belief["observation_noise_variance"]))
    reduction = family_uncertainty_reduction(
        source_family_id=family,
        family_beliefs=family_beliefs,
        observation_noise_variance=noise_variance,
        family_couplings=_candidate_couplings(candidate),
    )
    eig = float(reduction["total_information_gain_nats"])
    acquisition_value = expected_improvement_weight * ei + information_gain_weight * eig
    custody = _exact_archive_custody(candidate)
    dispatch_blockers = list(custody["blockers"])
    if candidate.get("score_claim") is True:
        dispatch_blockers.append("source_candidate_score_claim_true")
    if candidate.get("launch_gpu") is True or candidate.get("dispatch_now") is True:
        dispatch_blockers.append("source_requested_gpu_launch_refused")
    ready = bool(custody["verified"] and not dispatch_blockers)
    return {
        "candidate_id": candidate_id,
        "family_id": family,
        "score_claim": False,
        "promotion_eligible": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "ready_for_exact_eval_dispatch": ready,
        "dispatch_blockers": sorted(dict.fromkeys(dispatch_blockers)),
        "predicted_score_mean": round(predicted_mean, 12),
        "predicted_score_variance": round(predicted_variance, 12),
        "incumbent_exact_cuda_score": round(incumbent, 12),
        "expected_improvement": round(ei, 12),
        "expected_information_gain_nats": round(eig, 12),
        "acquisition_value": round(acquisition_value, 12),
        "acquisition_terms": {
            "expected_improvement_weight": round(float(expected_improvement_weight), 12),
            "information_gain_weight": round(float(information_gain_weight), 12),
            "expected_improvement_contribution": round(float(expected_improvement_weight) * ei, 12),
            "information_gain_contribution": round(float(information_gain_weight) * eig, 12),
        },
        "family_uncertainty_reduction": reduction,
        "exact_archive_custody": custody,
        "operator_note": "rank_only_no_gpu_launch",
    }


def rank_exact_eval_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    incumbent_score: float,
    family_beliefs: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    source: str = "manual",
    expected_improvement_weight: float = 1.0,
    information_gain_weight: float = 1.0,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Rank exact CUDA candidates by EI + EIG without dispatching anything."""

    candidate_list = list(candidates)
    beliefs = _normalize_family_beliefs(candidate_list, family_beliefs)
    rows = [
        expected_design_row(
            candidate,
            incumbent_score=incumbent_score,
            family_beliefs=beliefs,
            expected_improvement_weight=expected_improvement_weight,
            information_gain_weight=information_gain_weight,
        )
        for candidate in candidate_list
    ]
    rows.sort(
        key=lambda row: (
            -float(row["acquisition_value"]),
            -float(row["expected_improvement"]),
            -float(row["expected_information_gain_nats"]),
            float(row["predicted_score_mean"]),
            str(row["candidate_id"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    if top_k is not None:
        rows = rows[: int(top_k)]
    ready_count = sum(1 for row in rows if row["ready_for_exact_eval_dispatch"])
    aggregate_blockers = sorted(
        {
            blocker
            for row in rows
            for blocker in row["dispatch_blockers"]
        }
    )
    if ready_count == 0 and "no_candidate_with_verified_exact_archive_custody" not in aggregate_blockers:
        aggregate_blockers.append("no_candidate_with_verified_exact_archive_custody")
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.optimization.bayesian_experimental_design.rank_exact_eval_candidates",
        "source": source,
        "score_claim": False,
        "promotion_eligible": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "ready_for_exact_eval_dispatch": bool(ready_count),
        "ready_candidate_count": ready_count,
        "candidate_count": len(rows),
        "incumbent_exact_cuda_score": round(float(incumbent_score), 12),
        "acquisition_formula": "expected_improvement_weight * EI_minimize + information_gain_weight * EIG_family_nats",
        "expected_improvement_formula": "EI=(best-mu)*Phi((best-mu)/sigma)+sigma*phi((best-mu)/sigma)",
        "information_gain_formula": "0.5*log(prior_variance/posterior_variance)",
        "family_beliefs": {
            family: {
                "prior_score_variance": round(float(belief["prior_score_variance"]), 12),
                "observation_noise_variance": round(float(belief["observation_noise_variance"]), 12),
            }
            for family, belief in beliefs.items()
        },
        "dispatch_blockers": aggregate_blockers,
        "rows": rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOW gap closure widened wave 2026-05-17 — BUCKET B per-pair EIG extension
# ─────────────────────────────────────────────────────────────────────────────
# Per `.omx/research/comprehensive_wire_in_coverage_matrix_20260517.md` LOW
# gap candidate #1: extend BED with per-pair Expected Information Gain (EIG)
# estimation consuming the per-pair gradient anchor. The per-pair gradient
# carries per-pair signal variance which directly maps to the per-pair
# variance reduction term in the canonical EIG formula:
#
#   EIG_pair = 0.5 * log( 1 + per_pair_grad_var / observation_noise_var )
#
# This is the canonical Gaussian per-observation EIG formula applied to each
# pair separately, exposing per-pair information gain rather than the
# aggregate-only family-level EIG that existed pre-extension.
#
# Per CLAUDE.md "Apples-to-apples evidence discipline" the outcome is
# `[predicted; bayesian-experimental-design per-pair v1]` — NO score claim.

PER_PAIR_EIG_SCHEMA = "tac_bayesian_experimental_design_per_pair_eig_v1"


def compute_expected_information_gain_per_pair(
    archive_sha256: str,
    *,
    per_pair_gradient: Any | None = None,  # np.ndarray (N_bytes, N_pairs, 3)
    observation_noise_variance: float = DEFAULT_OBSERVATION_NOISE_VARIANCE,
    auto_load: bool = True,
) -> dict[int, float]:
    """Per-pair Expected Information Gain (EIG) estimation per Catalog #125 hook #4.

    Computes per-pair EIG directly from per-pair gradient variance using the
    canonical Gaussian per-observation formula::

        EIG_pair = 0.5 * log( 1 + per_pair_grad_var / observation_noise_var )

    where ``per_pair_grad_var`` is the L2 magnitude of the per-pair gradient
    column (a scalar per pair summarizing how much "signal energy" the pair
    contributes across all bytes + score axes).

    Per CLAUDE.md "Apples-to-apples evidence discipline" the outcome is
    `[predicted; bayesian-experimental-design per-pair v1]` — NO score claim.

    Parameters
    ----------
    archive_sha256
        64-char hex sha of the target archive bytes. Used to auto-load the
        canonical per-pair gradient anchor via
        ``tac.master_gradient_consumers.load_per_pair_gradient_from_anchor``.
    per_pair_gradient
        Optional (N_bytes, N_pairs, 3) tensor. When None and ``auto_load=True``,
        the helper auto-loads from the canonical anchor.
    observation_noise_variance
        Gaussian observation-noise variance σ² used in the EIG formula. Default
        ``DEFAULT_OBSERVATION_NOISE_VARIANCE`` (1e-4).
    auto_load
        When True (default), auto-loads missing inputs from canonical anchors.

    Returns
    -------
    dict[int, float]
        Per-pair EIG mapping ``pair_index -> EIG_nats``. EIG is non-negative
        by construction. Empty dict if no per-pair gradient is available and
        ``auto_load=False``.

    Raises
    ------
    BayesianExperimentalDesignError
        On malformed inputs.
    """
    if (
        not isinstance(archive_sha256, str)
        or len(archive_sha256) < 12
        or any(c not in "0123456789abcdefABCDEF" for c in archive_sha256)
    ):
        raise BayesianExperimentalDesignError(
            f"archive_sha256 must be a 12+ char hex string; got {archive_sha256!r}"
        )
    noise_var = float(observation_noise_variance)
    if noise_var <= 0 or not math.isfinite(noise_var):
        raise BayesianExperimentalDesignError(
            f"observation_noise_variance must be positive finite; got {noise_var!r}"
        )

    if per_pair_gradient is None and auto_load:
        try:
            from tac.master_gradient_consumers import (
                load_per_pair_gradient_from_anchor,
            )

            per_pair_gradient, _ = load_per_pair_gradient_from_anchor(
                archive_sha256=archive_sha256
            )
        except (ImportError, ValueError, FileNotFoundError, OSError):
            per_pair_gradient = None

    if per_pair_gradient is None:
        return {}

    # Shape: (N_bytes, N_pairs, 3) per PER_PAIR_GRADIENT_TENSOR_KIND
    if per_pair_gradient.ndim != 3 or per_pair_gradient.shape[-1] != 3:
        raise BayesianExperimentalDesignError(
            f"per_pair_gradient must have shape (N_bytes, N_pairs, 3); got "
            f"{per_pair_gradient.shape}"
        )

    n_pairs = int(per_pair_gradient.shape[1])
    # Per-pair signal-energy variance estimate: sum over (bytes × axes) of
    # squared gradient → L2-squared per pair column. Each pair is treated as
    # an independent Gaussian observation surface.
    per_pair_sq_sum = (per_pair_gradient ** 2).sum(axis=(0, 2))  # (N_pairs,)
    # Convert to per-pair variance estimate (divide by sample count = N_bytes × 3
    # to get average per-element variance contribution per pair).
    sample_count = float(per_pair_gradient.shape[0] * 3)
    per_pair_var = per_pair_sq_sum / max(sample_count, 1.0)

    eig_map: dict[int, float] = {}
    for pair_idx in range(n_pairs):
        var_p = float(per_pair_var[pair_idx])
        # Canonical Gaussian EIG (in nats): 0.5 * log(1 + signal_var / noise_var)
        eig = 0.5 * math.log(1.0 + var_p / noise_var)
        eig_map[int(pair_idx)] = max(0.0, eig)

    return eig_map


__all__ = [
    "CONTEST_ORIGINAL_BYTES",
    "RATE_SCORE_PER_BYTE",
    "PER_PAIR_EIG_SCHEMA",
    "SCHEMA_VERSION",
    "BayesianExperimentalDesignError",
    "compute_expected_information_gain_per_pair",
    "contest_score",
    "expected_design_row",
    "expected_improvement_minimize",
    "family_uncertainty_reduction",
    "gaussian_information_gain_nats",
    "gaussian_posterior_variance_after_family_observation",
    "rank_exact_eval_candidates",
]
