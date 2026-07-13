# SPDX-License-Identifier: MIT
"""Paper-equation SFESS learned-logit arm for cached, counted objectives.

This is an independent implementation derived from the equations in Wijk,
Vinuesa, and Azizpour (2025), *SFESS: Score Function Estimators for k-Subset
Sampling*, OpenReview:q87GUkdQBm (no DOI or arXiv ID found); its precursor is
Wijk et al. (2024), *Revisiting Score Function Estimators for k-Subset
Sampling*, arXiv:2407.16058.  No source code from the authors' repository is
copied here because the repository exposed no license file when checked on
2026-07-13.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tac.sfess_cached_replay import (
    CountedCachedOracle,
    SFESSError,
    sfess_leave_one_out_gradient,
)


@dataclass(frozen=True)
class LearnedLogitResult:
    """Counted result and diagnostics for one fixed-k SFESS arm."""

    best_mask: tuple[int, ...]
    best_value: float
    calls: int
    gradient_steps: int
    accepted_optimizer_updates: int
    rejected_optimizer_updates: int
    strict_gate_calls: int
    zero_variance_skips: int
    padding_calls: int
    sampled_value_spreads: tuple[float, ...]
    final_logits: tuple[float, ...]


def _top_k_mask(logits: np.ndarray, k: int) -> np.ndarray:
    """Deterministic MAP mask, with stable index-order tie breaking."""

    order = np.argsort(-logits, kind="stable")
    mask = np.zeros(logits.size, dtype=np.uint8)
    mask[order[:k]] = 1
    return mask


def run_learned_logit_sfess(
    oracle: CountedCachedOracle,
    *,
    k: int,
    samples_per_gradient: int,
    seed: int,
    learning_rate: float,
    comparison_noise_floor_s: float,
) -> LearnedLogitResult:
    """Minimize a counted fixed-k objective with SFESS and Adam.

    Control laws are fixed and paper-anchored: Adam uses the ICLR paper's
    reported 1e-4 learning rate and standard beta/epsilon constants
    from Kingma and Ba (2015), *Adam: A Method for Stochastic Optimization*,
    arXiv:1412.6980.  A
    group whose observed value spread is at or below the registered comparison
    floor skips the update and gate.  Every non-skipped update is admitted only
    through a separately counted exact-value MAP gate.
    """

    n_bits = oracle.table.n_bits
    if not 0 < k < n_bits:
        raise SFESSError("learned-logit SFESS requires 0 < k < n_bits")
    if samples_per_gradient < 2:
        raise SFESSError("learned-logit SFESS requires at least two samples")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise SFESSError("learning_rate must be finite and positive")
    if not math.isfinite(comparison_noise_floor_s) or comparison_noise_floor_s < 0.0:
        raise SFESSError("comparison_noise_floor_s must be finite and nonnegative")

    probability = k / n_bits
    logits = np.full(n_bits, math.log(probability / (1.0 - probability)))
    current_mask = _top_k_mask(logits, k)
    current_value = oracle(current_mask, purpose="initial")
    best_mask = current_mask.copy()
    best_value = current_value
    rng = np.random.default_rng(seed)
    first_moment = np.zeros(n_bits, dtype=np.float64)
    second_moment = np.zeros(n_bits, dtype=np.float64)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1.0e-8
    accepted_adam_steps = 0
    gradient_steps = 0
    accepted_optimizer_updates = 0
    rejected_optimizer_updates = 0
    strict_gate_calls = 0
    zero_variance_skips = 0
    padding_calls = 0
    spreads: list[float] = []

    while oracle.calls + samples_per_gradient + 1 <= oracle.budget:
        sample = sfess_leave_one_out_gradient(
            lambda mask: oracle(mask, purpose="sfess_sample"),
            logits,
            k,
            samples_per_gradient,
            rng,
        )
        spread = float(max(sample.values) - min(sample.values))
        spreads.append(spread)
        if spread <= comparison_noise_floor_s:
            zero_variance_skips += 1
            continue

        gradient_steps += 1
        gradient = sample.gradient
        staged_first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        staged_second_moment = beta2 * second_moment + (1.0 - beta2) * np.square(gradient)
        staged_adam_step = accepted_adam_steps + 1
        first_unbiased = staged_first_moment / (1.0 - beta1**staged_adam_step)
        second_unbiased = staged_second_moment / (1.0 - beta2**staged_adam_step)
        staged_logits = logits - learning_rate * first_unbiased / (
            np.sqrt(second_unbiased) + epsilon
        )

        proposal = _top_k_mask(staged_logits, k)
        proposal_value = oracle(proposal, purpose="strict_exact_gate")
        strict_gate_calls += 1
        if proposal_value < current_value - comparison_noise_floor_s:
            logits = staged_logits
            first_moment = staged_first_moment
            second_moment = staged_second_moment
            accepted_adam_steps = staged_adam_step
            accepted_optimizer_updates += 1
            current_mask = proposal
            current_value = proposal_value
            if proposal_value < best_value - comparison_noise_floor_s:
                best_mask = proposal.copy()
                best_value = proposal_value
        else:
            rejected_optimizer_updates += 1

    while oracle.calls < oracle.budget:
        oracle(current_mask, purpose="budget_padding")
        padding_calls += 1

    return LearnedLogitResult(
        best_mask=tuple(int(bit) for bit in best_mask),
        best_value=best_value,
        calls=oracle.calls,
        gradient_steps=gradient_steps,
        accepted_optimizer_updates=accepted_optimizer_updates,
        rejected_optimizer_updates=rejected_optimizer_updates,
        strict_gate_calls=strict_gate_calls,
        zero_variance_skips=zero_variance_skips,
        padding_calls=padding_calls,
        sampled_value_spreads=tuple(spreads),
        final_logits=tuple(float(value) for value in logits),
    )
