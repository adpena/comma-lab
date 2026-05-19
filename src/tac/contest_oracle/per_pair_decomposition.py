# SPDX-License-Identifier: MIT
"""Impl 4 + Impl 13 -- 600-pair additive decomposition (canonical alias + bandit ext).

Canonical alias for ``tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual``
plus bandit-style extension per Impl 13 (Thompson sampling over per-pair
substrate / config / codec assignments).

The per-pair structure of the contest scorer (600 non-overlapping pairs;
``upstream/evaluate.py``) is fully exposed; cross-pair correlation IS the
residual signal. Per-pair Thompson sampling lets us assign DIFFERENT
substrates per pair (polymorphic dispatch), which is the bandit-optimal
extension.

Citations:
  - ``tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual``
    -- canonical sister this module re-exports + extends.
  - Auer 2002 *Finite-time Analysis of the Multiarmed Bandit Problem* --
    Thompson sampling theory.
  - Russo & Van Roy 2014 *Learning to Optimize Via Posterior Sampling* --
    posterior-sampling regret bounds.
  - CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable -- per-pair
    decomposition is canonical primal-dual surface.
  - Task #800-#802 per-pair wire-ins.

Catalog #125 hook 1 (sensitivity_map): ACTIVE -- per-pair sensitivities are
the canonical decomposition surface.
Catalog #125 hook 4 (cathedral_autopilot_dispatch): ACTIVE -- per-pair
bandit assignments feed the autopilot ranker as polymorphic dispatch.
Catalog #305 observability surface: decomposable_per_signal,
queryable_post_hoc, cite_able.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .constants import CONTEST_NUM_PAIRS

# Re-export canonical helper from the sister consumers module so callers
# can import the per-pair optimal-treatment plan canonical from contest_oracle
# (operator standing directive: composable canonical surface, not duplicate).
try:
    from tac.master_gradient_consumers import (
        per_pair_optimal_treatment_plan_via_lagrangian_dual as _canonical_per_pair_plan,
    )

    CANONICAL_PER_PAIR_PLAN_AVAILABLE: Final[bool] = True
except ImportError:  # pragma: no cover -- shim for partial-checkout safety
    _canonical_per_pair_plan = None
    CANONICAL_PER_PAIR_PLAN_AVAILABLE = False


def per_pair_optimal_treatment_plan(*args, **kwargs):
    """Canonical alias for ``tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual``.

    Routes to the canonical helper without duplication; this module is the
    contest_oracle's namespaced surface so callers can:

        from tac.contest_oracle.per_pair_decomposition import per_pair_optimal_treatment_plan

    instead of importing from the sister consumers module directly. The
    alias preserves the canonical signature; see the sister docstring for
    full API details.
    """
    if not CANONICAL_PER_PAIR_PLAN_AVAILABLE:
        raise ImportError(
            "tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual "
            "not available; install master_gradient consumers package"
        )
    return _canonical_per_pair_plan(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class PerPairThompsonSamplingPlan:
    """Bandit-style per-pair substrate / config / codec assignment plan (Impl 13).

    Each of the 600 contest pairs gets a (substrate_arm, config_arm, codec_arm)
    triple sampled from the posterior over per-pair-arm reward distributions.
    Polymorphic dispatch: different pairs may use different substrates.
    """

    num_pairs: int
    """Number of pairs (= 600 per contest_fixed CONTEST_NUM_PAIRS)."""

    per_pair_assignments: tuple[tuple[str, str, str], ...]
    """Tuple of ``num_pairs`` triples ``(substrate_arm, config_arm, codec_arm)``."""

    posterior_evidence_grade: str = "predicted_thompson_sampling"
    """Per Catalog #287/#323: this is a SAMPLED prediction not a contest
    score claim. The posterior is updated from empirical anchors per
    Catalog #128 ``tac.continual_learning.posterior_update_locked``."""


def thompson_sample_per_pair_assignment(
    *,
    substrate_arms: tuple[str, ...],
    config_arms: tuple[str, ...],
    codec_arms: tuple[str, ...],
    num_pairs: int = CONTEST_NUM_PAIRS,
    random_seed: int = 0xC0FFEE,
) -> PerPairThompsonSamplingPlan:
    """Sample per-pair substrate / config / codec assignment via Thompson rule.

    The Thompson rule samples ONE arm per pair from a uniform posterior
    (placeholder for a richer posterior loaded via continual-learning).
    Replaces uniform random by a posterior-weighted sample once the
    canonical posterior is wired in via Catalog #128 sister.

    Args:
        substrate_arms: Tuple of canonical substrate identifiers.
        config_arms: Tuple of canonical configuration identifiers.
        codec_arms: Tuple of canonical codec identifiers.
        num_pairs: Number of pairs to assign (default = 600 per contest_fixed).
        random_seed: Deterministic seed for reproducibility per CLAUDE.md
            "Canonical pipeline standard" non-negotiable.

    Returns:
        ``PerPairThompsonSamplingPlan`` with one (substrate, config, codec)
        triple per pair.

    Raises:
        ValueError: if any arm tuple is empty.
    """
    if not substrate_arms:
        raise ValueError("substrate_arms must be non-empty")
    if not config_arms:
        raise ValueError("config_arms must be non-empty")
    if not codec_arms:
        raise ValueError("codec_arms must be non-empty")
    if num_pairs <= 0:
        raise ValueError(f"num_pairs must be > 0 (got {num_pairs})")

    # Deterministic-seeded RNG per CLAUDE.md reproducibility non-negotiable.
    import random

    rng = random.Random(random_seed)

    assignments = tuple(
        (
            rng.choice(substrate_arms),
            rng.choice(config_arms),
            rng.choice(codec_arms),
        )
        for _ in range(num_pairs)
    )
    return PerPairThompsonSamplingPlan(
        num_pairs=int(num_pairs),
        per_pair_assignments=assignments,
    )


__all__ = [
    "CANONICAL_PER_PAIR_PLAN_AVAILABLE",
    "PerPairThompsonSamplingPlan",
    "per_pair_optimal_treatment_plan",
    "thompson_sample_per_pair_assignment",
]
