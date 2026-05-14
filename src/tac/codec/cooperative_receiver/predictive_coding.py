"""Rao-Ballard predictive-coding-hierarchy primitive (1999).

The Rao-Ballard predictive-coding theorem describes a top-down hierarchy in
which each level predicts the activity of the level below; only the
PREDICTION ERROR (residual) is propagated upward. Operationally for a learned
codec this means: a good world model predicts most of the per-pair signal;
only the residual the world model failed to predict needs to be encoded as
side information. Penalizing the L2 norm of that residual rewards the world
model for predicting more, indirectly minimizing the per-pair side-info bit
budget.

This primitive computes the per-pair residual L2 term
``H_pred = ||residual||_2^2 / N`` (mean of squared residuals) and exposes it
as a scalar tensor with full gradient flow into the residual input. The
training-loop Lagrangian composes it with a ``delta_predict`` weight:
``L = L_cooperative + alpha_rate * B/N + delta_predict * H_pred``.

This primitive is **distinct** from the Atick-Redlich cooperative-receiver
primitive in :mod:`tac.codec.cooperative_receiver.atick_redlich`:

- Atick-Redlich (efficient coding 1990/1992): the encoder maximizes
  ``MI(B; S(B))`` against a FIXED known scorer ``S``. Operationally:
  scorer-distortion loss with eval-roundtrip and differentiable
  preprocess_input.
- Rao-Ballard (predictive coding 1999): a TOP-DOWN HIERARCHY transmits only
  prediction errors. Operationally: penalize the magnitude of the per-pair
  residual the world model failed to predict.

Both primitives can compose orthogonally — the cooperative-receiver loss
operates on the SCORED OUTPUT (RGB → scorer pair), while the predictive-
coding term operates on the per-pair RESIDUAL (the side-info the world model
could not generate from its own state). The time-traveler substrate stacks
both.

References
----------
- Rao & Ballard, "Predictive coding in the visual cortex: a functional
  interpretation of some extra-classical receptive-field effects", Nature
  Neuroscience 2(1):79-87, 1999.
- Friston, "The free-energy principle: a unified brain theory?", Nature
  Reviews Neuroscience 11(2):127-138, 2010 (free-energy formulation that
  generalizes Rao-Ballard).

Cross-references
----------------
- :mod:`tac.codec.cooperative_receiver.atick_redlich` — sister cooperative-
  receiver primitive that the time-traveler substrate composes with this one.
- :mod:`tac.substrates.time_traveler_l5_autonomy.score_aware_loss` — the
  in-tree consumer of both primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PredictiveCodingWeights:
    """Lagrangian weight + numerical guard for the predictive-coding term.

    Args:
        delta_predict: Weight applied to the per-pair residual L2 term.
            Default 0.1 matches the time-traveler default; 0 disables the
            primitive while preserving the call site (the returned scaled
            term is exactly zero with no gradient flow).
        residual_floor: Lower bound on the residual magnitude before
            squaring. Default 0.0 (no floor) keeps the term mathematically
            equal to ``mean(residual ** 2)``. Set to a small positive value
            only if you need to avoid pathologically zero gradients.
    """

    delta_predict: float = 0.1
    residual_floor: float = 0.0

    def __post_init__(self) -> None:
        if self.delta_predict < 0.0:
            raise ValueError(
                f"delta_predict must be >= 0; got {self.delta_predict}"
            )
        if self.residual_floor < 0.0:
            raise ValueError(
                f"residual_floor must be >= 0; got {self.residual_floor}"
            )


@dataclass(frozen=True)
class PredictiveCodingOutput:
    """Result of one Rao-Ballard primitive invocation.

    Attributes:
        scaled_term: Scalar tensor ``delta_predict * H_pred`` with gradient
            flow into the residual input. When ``delta_predict == 0`` the
            tensor is exactly zero (a constant), which is fine — it is safe
            to add to a loss; backprop simply contributes nothing through
            this term.
        unscaled_residual_l2: Scalar ``mean(residual ** 2)`` for diagnostic
            logging (gradient-reachable).
    """

    scaled_term: torch.Tensor
    unscaled_residual_l2: torch.Tensor


def predictive_coding_residual_term(
    residual: torch.Tensor,
    *,
    weights: PredictiveCodingWeights | None = None,
) -> PredictiveCodingOutput:
    """Compute the Rao-Ballard predictive-coding residual L2 term.

    The world model is rewarded for predicting more (smaller residual =
    better world model) by penalizing the mean of the squared residual.
    The returned :class:`PredictiveCodingOutput` carries both the scaled
    loss term (already multiplied by ``delta_predict``, ready to add to
    the training-loop Lagrangian) and the unscaled L2 for diagnostic
    logging.

    Args:
        residual: Tensor of any shape carrying gradient. Typically the
            per-pair side info that the world model failed to predict
            (e.g., ``(B, N_pairs, residual_dim)``). The L2 is computed as
            ``mean(residual ** 2)`` over ALL dimensions, so the scale is
            invariant to batch / pair count.
        weights: Lagrangian weight + numerical guard; defaults to the
            time-traveler defaults (``delta_predict=0.1``, no floor).

    Returns:
        :class:`PredictiveCodingOutput` carrying the scaled loss term and
        the unscaled L2 for logging.
    """

    if weights is None:
        weights = PredictiveCodingWeights()

    if weights.residual_floor > 0.0:
        # Soft floor: clamp absolute residual magnitude to the floor before
        # squaring. This is gradient-reachable through clamp.
        bounded = torch.clamp(residual.abs(), min=weights.residual_floor)
        # Re-attach sign so tests that depend on residual.sign() see it,
        # though the sign cancels in the square. We avoid it for purity.
        unscaled = bounded.pow(2).mean()
    else:
        unscaled = residual.pow(2).mean()
    scaled = weights.delta_predict * unscaled
    return PredictiveCodingOutput(scaled_term=scaled, unscaled_residual_l2=unscaled)


__all__ = [
    "PredictiveCodingOutput",
    "PredictiveCodingWeights",
    "predictive_coding_residual_term",
]
