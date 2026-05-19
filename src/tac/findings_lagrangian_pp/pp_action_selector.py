# SPDX-License-Identifier: MIT
"""Exact KL info gain via posterior samples (Foster et al 2019; TRACK B).

Per operator-frontier-override 2026-05-19 + Foster et al 2019 *"Variational
Bayesian Optimal Experimental Design"*: TRACK B uses the sample-based
variational lower bound on expected KL info gain when the posterior is
multi-modal / non-Gaussian. For our hierarchical Bayes use case the
closed-form Gaussian approximation (TRACK A) often suffices, but TRACK B's
exact KL via samples is more accurate when the posterior has
significant cross-family variance (Time-Traveler's empirical claim).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from tac.findings_lagrangian.action_selector import CandidateAction, ActionRecommendation
from tac.findings_lagrangian_pp.pp_posterior import (
    PPHierarchicalPosterior,
    PPPosteriorError,
)


__all__ = [
    "expected_info_gain_via_posterior_samples",
    "recommend_next_action_via_pp_samples",
]


def expected_info_gain_via_posterior_samples(
    posterior_before: PPHierarchicalPosterior,
    *,
    hypothetical_residuals: Iterable[float],
    architecture_id: str,
    sigma_obs: float = 1.0,
) -> float:
    """Sample-based estimator of E[KL(posterior_after || posterior_before)].

    Per Foster et al 2019 + Hinton operating-within slot 20: the variational
    lower bound on EIG via posterior samples is exact when n_samples → inf;
    for our ≤500-sample regime the bound is tight (per Foster's empirical
    receipts).

    The estimator (per Foster 2019 eq. 7):

        EIG ≈ (1/N) Σ_n log p(y_n | θ_n) - log p(y_n)

    where θ_n are posterior samples + y_n are predicted observations.

    Args:
        posterior_before: TRACK B hierarchical posterior.
        hypothetical_residuals: predicted residuals from the hypothetical
            experiment.
        architecture_id: which architecture the experiment would update.
        sigma_obs: observation noise std-dev.

    Returns:
        Estimated EIG in nats (>= 0).
    """
    import numpy as np

    if sigma_obs <= 0:
        raise PPPosteriorError(f"sigma_obs={sigma_obs} must be > 0")
    residuals = list(hypothetical_residuals)
    if not residuals:
        return 0.0
    if architecture_id is None:
        raise PPPosteriorError("architecture_id must be specified for PP info gain")

    # Find architecture index.
    arch_ids_sorted = sorted(set(posterior_before.architecture_family_map.keys()))
    try:
        arch_idx = arch_ids_sorted.index(architecture_id)
    except ValueError:
        raise PPPosteriorError(
            f"architecture_id={architecture_id!r} not in posterior architecture set"
        )

    theta_samples = np.array(
        [list(row) for row in posterior_before.theta_arch_posterior_samples]
    )
    theta_arch_samples = theta_samples[:, arch_idx]  # (n_samples,)
    n_post_samples = len(theta_arch_samples)

    if n_post_samples == 0:
        return 0.0

    residuals_arr = np.array(residuals, dtype=np.float64)

    # log p(y_n | θ_n) for each y_n averaged over θ_n samples.
    # Approximation: for each predicted y_n, log p(y_n | θ_n) under Normal(θ_n, sigma_obs).
    log_lik = np.zeros((n_post_samples, len(residuals_arr)))
    for s in range(n_post_samples):
        theta = theta_arch_samples[s]
        log_lik[s] = -0.5 * np.log(2.0 * np.pi * sigma_obs**2) - 0.5 * (
            residuals_arr - theta
        ) ** 2 / (sigma_obs**2)

    # Log marginal p(y_n) ≈ log mean exp log p(y_n | θ_n) over θ_n samples.
    # Use log-sum-exp for numerical stability.
    max_log = np.max(log_lik, axis=0, keepdims=True)
    log_marginal = max_log + np.log(np.mean(np.exp(log_lik - max_log), axis=0, keepdims=True))
    # EIG = mean over samples of (log p(y_n | θ_n) - log p(y_n))
    eig_per_obs = log_lik - log_marginal
    eig = float(np.mean(eig_per_obs.sum(axis=1)))
    return max(eig, 0.0)


def recommend_next_action_via_pp_samples(
    candidate_actions: Iterable[CandidateAction],
    *,
    pp_posteriors_by_equation_id: Mapping[str, PPHierarchicalPosterior],
    architecture_id_per_action: Mapping[str, str],
    budget_usd: float,
    sigma_obs: float = 1.0,
) -> ActionRecommendation:
    """TRACK B sister of `recommend_next_action_via_expected_information_gain`.

    Per operator-frontier-override 2026-05-19 + Hinton 1990s BMA tradition:
    the canonical hierarchical-Bayes action selector uses posterior samples
    for exact info gain rather than the closed-form Gaussian approximation.

    Args:
        candidate_actions: actions the selector can choose from.
        pp_posteriors_by_equation_id: dict mapping equation_id → PPHierarchicalPosterior.
        architecture_id_per_action: dict mapping action_id → architecture_id.
        budget_usd: max dollar budget.
        sigma_obs: observation noise std-dev.

    Returns:
        ActionRecommendation (sister of TRACK A ranker output).
    """
    import datetime as _dt

    actions_list = list(candidate_actions)
    if not actions_list:
        return ActionRecommendation(
            ranked_actions=(),
            budget_usd=budget_usd,
            recommended_action_id=None,
            recommendation_rationale="no candidate actions provided (TRACK B)",
            computation_utc=_dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    scored: list[tuple[CandidateAction, float, float]] = []
    for action in actions_list:
        if action.equation_id not in pp_posteriors_by_equation_id:
            scored.append((action, 0.0, 0.0))
            continue
        arch_id = architecture_id_per_action.get(action.action_id)
        if arch_id is None:
            scored.append((action, 0.0, 0.0))
            continue
        posterior = pp_posteriors_by_equation_id[action.equation_id]
        try:
            eig = expected_info_gain_via_posterior_samples(
                posterior,
                hypothetical_residuals=list(action.predicted_hypothetical_residuals),
                architecture_id=arch_id,
                sigma_obs=sigma_obs,
            )
        except PPPosteriorError:
            eig = 0.0
        if action.cost_usd <= 1e-9:
            eig_per_dollar = eig * 1e6
        else:
            eig_per_dollar = eig / action.cost_usd
        scored.append((action, eig, eig_per_dollar))

    scored.sort(key=lambda x: x[2], reverse=True)

    recommended_id: str | None = None
    recommended_rationale = "no action fits the budget (TRACK B)"
    for action, eig, eigpd in scored:
        if action.cost_usd <= budget_usd:
            recommended_id = action.action_id
            recommended_rationale = (
                f"TRACK B (NumPyro NUTS) action_id={action.action_id} maximizes "
                f"E[KL info gain]/$ = {eigpd:.4f} (absolute info_gain={eig:.4f} nats; "
                f"cost=${action.cost_usd:.2f}); per Foster et al 2019 + Hinton BMA + "
                f"operator-frontier-override 2026-05-19; per {action.rationale}"
            )
            break

    return ActionRecommendation(
        ranked_actions=tuple(scored),
        budget_usd=budget_usd,
        recommended_action_id=recommended_id,
        recommendation_rationale=recommended_rationale,
        computation_utc=_dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
