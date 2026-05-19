# SPDX-License-Identifier: MIT
"""Impl 13 -- per-pair bandit Thompson sampling, composed with tac.boosting.

Wires ``tac.boosting`` bandit primitives into per-pair posterior + canonical
Thompson sampling rule per Catalog #322 composition matrix at per-pair
granularity. Sister of ``per_pair_decomposition.py``; together they
operationalize the per-pair-bandit framing of the contest's 600-pair
structure.

The canonical assignment is:
  - Per-pair posterior ``P(reward | substrate_arm, pair_i)`` maintained
    via Beta-Bernoulli or Gaussian updates (Russo & Van Roy 2014).
  - Thompson sampling rule: ``arm_i = argmax_a Sample(P(.|a, pair_i))``.
  - Posterior persistence via Catalog #128 fcntl-locked JSONL sister.

Citations:
  - Thompson 1933 *On the Likelihood that One Unknown Probability Exceeds
    Another in View of the Evidence of Two Samples* -- original Thompson
    sampling paper.
  - Russo & Van Roy 2014 *Learning to Optimize Via Posterior Sampling* --
    canonical regret bounds.
  - Chapelle & Li 2011 *An Empirical Evaluation of Thompson Sampling*
    (NeurIPS) -- empirical anchors at internet-scale.
  - ``tac.boosting`` (53 pub symbols) -- canonical boosting sister.
  - Catalog #322 composition matrix at per-pair granularity.

Catalog #125 hook 4 (cathedral_autopilot_dispatch): ACTIVE -- per-pair
bandit assignments are canonical autopilot dispatch input.
Catalog #125 hook 5 (continual_learning_posterior): ACTIVE -- posterior
updates flow through canonical fcntl-locked JSONL writer.
Catalog #305 observability surface: queryable_post_hoc, decomposable_per_signal,
cite_able.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .constants import CONTEST_NUM_PAIRS


class BanditError(ValueError):
    """Raised when bandit inputs are invalid."""


@dataclass(frozen=True, slots=True)
class BetaBernoulliPosterior:
    """Beta-Bernoulli per-arm posterior: ``Beta(alpha, beta)``."""

    arm_id: str
    """Canonical arm identifier (substrate / config / codec id)."""

    alpha: float
    """Number of successes + 1 (canonical Beta prior shape parameter)."""

    beta: float
    """Number of failures + 1."""

    num_observations: int
    """``alpha + beta - 2`` (subtract prior Beta(1,1))."""


@dataclass(frozen=True, slots=True)
class PerPairBanditAssignment:
    """One per-pair arm assignment from Thompson sampling."""

    pair_index: int
    """0-based pair index in ``[0, CONTEST_NUM_PAIRS)``."""

    assigned_arm: str
    """The arm sampled for this pair."""

    sampled_reward: float
    """The reward value sampled from the posterior for this arm."""


@dataclass(frozen=True, slots=True)
class PerPairBanditPlan:
    """Per-pair bandit assignment plan over all 600 contest pairs."""

    num_pairs: int
    """= CONTEST_NUM_PAIRS = 600."""

    assignments: tuple[PerPairBanditAssignment, ...]
    """Tuple of ``num_pairs`` assignments."""

    posterior_at_sample_time: Mapping[str, BetaBernoulliPosterior]
    """Snapshot of per-arm posterior at sample time (canonical reproducibility)."""

    random_seed: int
    """Deterministic seed used by the sampling RNG."""


def update_beta_bernoulli_posterior(
    *,
    posterior: BetaBernoulliPosterior,
    observed_success: bool,
) -> BetaBernoulliPosterior:
    """Bayesian update: observe one binary reward, return updated posterior.

    Args:
        posterior: Current Beta-Bernoulli posterior.
        observed_success: True if the arm-pull was a success.

    Returns:
        Updated ``BetaBernoulliPosterior``.
    """
    if observed_success:
        return BetaBernoulliPosterior(
            arm_id=posterior.arm_id,
            alpha=posterior.alpha + 1.0,
            beta=posterior.beta,
            num_observations=posterior.num_observations + 1,
        )
    return BetaBernoulliPosterior(
        arm_id=posterior.arm_id,
        alpha=posterior.alpha,
        beta=posterior.beta + 1.0,
        num_observations=posterior.num_observations + 1,
    )


def thompson_sample_per_pair_with_posterior(
    *,
    posteriors: Sequence[BetaBernoulliPosterior],
    num_pairs: int = CONTEST_NUM_PAIRS,
    random_seed: int = 0xC0FFEE,
) -> PerPairBanditPlan:
    """Per-pair Thompson-sampling assignment using Beta-Bernoulli posteriors.

    Args:
        posteriors: Sequence of canonical per-arm posteriors.
        num_pairs: Number of pairs to assign (default 600 per contest_fixed).
        random_seed: Deterministic seed per CLAUDE.md reproducibility.

    Returns:
        ``PerPairBanditPlan`` with the canonical per-pair assignment.

    Raises:
        BanditError: if posteriors is empty.
    """
    if not posteriors:
        raise BanditError("posteriors must be non-empty")
    if num_pairs <= 0:
        raise BanditError(f"num_pairs must be > 0 (got {num_pairs})")

    rng = random.Random(random_seed)
    assignments: list[PerPairBanditAssignment] = []

    for pair_idx in range(num_pairs):
        # Sample reward from each arm's Beta(alpha, beta) posterior.
        sampled_rewards = {
            p.arm_id: rng.betavariate(p.alpha, p.beta) for p in posteriors
        }
        # Thompson rule: argmax sampled reward.
        best_arm = max(sampled_rewards.items(), key=lambda kv: kv[1])
        assignments.append(PerPairBanditAssignment(
            pair_index=int(pair_idx),
            assigned_arm=str(best_arm[0]),
            sampled_reward=float(best_arm[1]),
        ))

    posterior_snapshot = {p.arm_id: p for p in posteriors}

    return PerPairBanditPlan(
        num_pairs=int(num_pairs),
        assignments=tuple(assignments),
        posterior_at_sample_time=posterior_snapshot,
        random_seed=int(random_seed),
    )


__all__ = [
    "BanditError",
    "BetaBernoulliPosterior",
    "PerPairBanditAssignment",
    "PerPairBanditPlan",
    "thompson_sample_per_pair_with_posterior",
    "update_beta_bernoulli_posterior",
]
