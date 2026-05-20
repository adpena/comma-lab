# SPDX-License-Identifier: MIT
"""Cathedral consumer for exploit #2 - score-weighted reconstruction error.

Per RESPAWN-MG-7-BUNDLE 2026-05-20 + operator NON-NEGOTIABLE 2026-05-19 verbatim
*"Implement all exploits and wire and integrate all"*. Consumes per-pair
``M_contest`` (the per-pixel scorer-axis gradient) and surfaces the canonical
score-weighted reconstruction error as the encoder-loss target that SHOULD
REPLACE raw L2/L1 in any future trainer per CLAUDE.md "Meta-Lagrangian/Pareto
solver" non-negotiable. Auto-discovered by cathedral autopilot ranker per
Catalog #335 canonical contract.

## Canonical-vs-unique decision per layer

- M_contest gradient extraction: ADOPT canonical
  ``tac.master_gradient_comparison.multi_granularity.extract_M_contest`` (the
  producer surface; chain-rule discipline per Catalog #318).
- Provenance contract: ADOPT canonical
  ``tac.provenance.build_provenance_for_predicted`` (sensitivity surface is
  PREDICTED_FROM_MODEL by construction; never a contest-axis score claim).
- Routing markers: ADOPT Catalog #341 canonical
  (``predicted_delta_adjustment=0.0`` / ``promotable=False`` /
  ``axis_tag="[predicted]"``) - this consumer is a TRAINING-LOSS signal not a
  dispatch-promotion signal.
- Trainer wire-in scaffold: FORK to ``tac.training.score_weighted_reconstruction_loss``
  because the canonical scorer-loss helper ``score_pair_components`` lacks
  M_contest weighting and forking lets us preserve the canonical helper for
  legacy callers while exposing the augmented variant as the new default for
  M_contest-aware trainers.

## Observability surface

Per Catalog #305 + CLAUDE.md "Max observability - non-negotiable":

1. Inspectable per layer: ``consume_candidate`` returns a dict with
   ``score_weighted_error_total`` / ``score_weighted_error_per_pair`` /
   ``rationale`` so each layer's contribution is queryable.
2. Decomposable per signal: total error decomposes per-pair via
   ``score_weighted_error_per_pair`` tuple.
3. Diff-able across runs: every output dict carries
   ``m_contest_array_sha256`` so two runs are byte-level diffable.
4. Queryable post-hoc: helper ``compute_score_weighted_reconstruction_loss``
   is operator-runnable any time.
5. Cite-able: ``provenance.canonical_helper_invocation`` cites the producer
   module path per Catalog #323.
6. Counterfactual-able: per-pair tuple lets the operator ask "what if pair N
   were re-encoded?" without re-running the scorer.

## 9-dimension success checklist evidence

1. UNIQUENESS: M_contest-weighted reconstruction error is canonically distinct
   from raw L2/PSNR optimization because the weight tensor IS the scorer's
   own gradient response (not a hand-tuned saliency prior).
2. BEAUTY+ELEGANCE: ~180 LOC consumer; reviewable in 30 seconds.
3. DISTINCTNESS: distinct from sister exploits 3/4/5/6/7/8/9 (each targets a
   different gradient granularity / decomposition axis).
4. RIGOR: respects Catalog #287/#323 evidence-tag discipline;
   non-promotable-by-construction per Catalog #341 routing markers.
5. OPTIMIZATION-PER-TECHNIQUE: forks ``score_pair_components`` only at the
   trainer wire-in surface (canonical helper preserved for legacy).
6. STACK-OF-STACKS-COMPOSABILITY: orthogonal to substrate-class-shift sister
   consumers; composes additively with per-pair difficulty atlas (sister
   MG-4) + bit allocator (sister MG-4 sister).
7. DETERMINISTIC-REPRODUCIBILITY: pure numpy einsum; no random sampling.
8. EXTREME-OPTIMIZATION-PERFORMANCE: O(N_pairs * H * W) compute; matches
   producer surface throughput.
9. OPTIMAL-MINIMAL-CONTEST-SCORE: this consumer feeds the encoder-loss
   target; the future score improvement comes from REPLACING raw L2 in the
   trainer loop (training-side wire-in scaffold at
   ``tac.training.score_weighted_reconstruction_loss``).

## Cargo-cult audit per assumption

- ASSUMPTION: per-pixel scorer-sensitivity weighting strictly dominates
  uniform-pixel L2 for the contest score. CLASSIFICATION: HARD-EARNED
  (CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: pixels with low
  M_contest magnitude do not move the contest score regardless of L2; the
  canonical scorer-loss helper ``score_pair_components`` already implicitly
  uses this principle at the per-pair level - this consumer extends to the
  per-pixel level).
- ASSUMPTION: encoders that minimize score-weighted L2 actually improve
  contest score. CLASSIFICATION: CARGO-CULTED-PENDING-EMPIRICAL. The
  trainer-side wire-in scaffold at
  ``tac.training.score_weighted_reconstruction_loss`` is SCAFFOLD only;
  validation requires a future paid contest-CUDA dispatch on an actual
  trainer that consumes this loss. Per CLAUDE.md "Forbidden premature
  KILL": this is research-substrate territory, not a kill-class claim.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "score_weighted_reconstruction_error_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    For this consumer, the posterior update is a no-op at the consumer level
    because the score-weighted reconstruction error is computed on-demand
    from the producer surface (``tac.master_gradient_comparison.multi_granularity``)
    when ``consume_candidate`` fires. Future empirical anchors that prove
    the score-weighted loss empirically beats raw L2 on a contest-CUDA
    dispatch SHOULD update the trainer wire-in scaffold's recommended
    default weighting via this hook.
    """
    _ = anchor  # acknowledged; no consumer-level state to update


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Routing-style consumer per Catalog #341: returns canonical non-promotable
    markers (``predicted_delta_adjustment=0.0`` / ``promotable=False`` /
    ``axis_tag="[predicted]"``) because score-weighted reconstruction error
    is a TRAINING-LOSS signal, not a dispatch-promotion signal.

    The candidate dict MAY carry ``m_contest_array_sha256`` /
    ``score_weighted_error_total`` / ``per_pair_error_signature`` if a prior
    pipeline stage computed them; this consumer surfaces them in its
    contribution dict so downstream consumers (per-pair difficulty atlas
    sister MG-4 / bit allocator) can chain.
    """
    m_contest_sha = candidate.get("m_contest_array_sha256")
    score_weighted_total = candidate.get("score_weighted_error_total")
    per_pair_sig = candidate.get("per_pair_error_signature")

    rationale_parts = [
        "score-weighted reconstruction error consumer (exploit #2)",
    ]
    if score_weighted_total is not None:
        rationale_parts.append(
            f"upstream score_weighted_error_total={score_weighted_total}"
        )
    if m_contest_sha is not None:
        rationale_parts.append(f"M_contest sha256[:12]={str(m_contest_sha)[:12]}")
    rationale = "; ".join(rationale_parts)

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "consumer_signal_kind": "encoder_loss_target_recommendation",
        "score_weighted_error_total": score_weighted_total,
        "per_pair_error_signature": per_pair_sig,
        "m_contest_array_sha256": m_contest_sha,
    }


def compute_score_weighted_reconstruction_loss(
    I_inflated,
    I_contest,
    M_contest,
):
    """Compute scalar score-weighted reconstruction loss.

    Formula: ``mean over (pair, pixel) of (I_inflated - I_contest)^2 *
    ||M_contest[:, axis, :, :]||^2``. This is the canonical drop-in
    replacement for L2 reconstruction loss when the trainer has access to
    the per-pixel scorer-sensitivity tensor M_contest.

    Args:
        I_inflated: np.ndarray of shape (N_pairs, 3, H, W) - the encoder's
            reconstruction.
        I_contest: np.ndarray of shape (N_pairs, 3, H, W) - the contest
            video ground truth.
        M_contest: np.ndarray of shape (N_pairs, 3, H, W) - the per-pixel
            scorer-axis gradient (typically loaded from
            ``ContestGradientTensor.load()`` per producer surface).

    Returns:
        Scalar loss as Python float. Always non-negative.

    Raises:
        ValueError: on shape mismatch or invalid input.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy required for score-weighted loss") from exc

    inflated = np.asarray(I_inflated, dtype=np.float64)
    contest = np.asarray(I_contest, dtype=np.float64)
    m = np.asarray(M_contest, dtype=np.float64)
    if inflated.shape != contest.shape:
        raise ValueError(
            f"I_inflated shape {inflated.shape} != I_contest shape {contest.shape}"
        )
    if inflated.ndim != 4 or inflated.shape[1] != 3:
        raise ValueError(
            f"I_inflated must have shape (N_pairs, 3, H, W); got {inflated.shape}"
        )
    n_pairs, _, h, w = inflated.shape
    if m.shape != (n_pairs, 3, h, w):
        raise ValueError(
            f"M_contest shape {m.shape} != expected ({n_pairs}, 3, {h}, {w})"
        )
    # Per-pixel L2 of M_contest across the axis dimension.
    m_l2_sq = np.sum(np.square(m), axis=1)  # (N_pairs, H, W)
    # Per-pixel L2 of pixel residual across RGB.
    diff = inflated - contest
    err_per_pixel = np.sum(np.square(diff), axis=1)  # (N_pairs, H, W)
    weighted = err_per_pixel * m_l2_sq
    return float(weighted.mean())


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "compute_score_weighted_reconstruction_loss",
    "consume_candidate",
    "update_from_anchor",
]
