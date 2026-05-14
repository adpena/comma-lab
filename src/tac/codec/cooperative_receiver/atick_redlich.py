"""Atick-Redlich cooperative-receiver primitive (efficient-coding 1990/1992).

The Atick-Redlich efficient-coding theorem states that under a fixed downstream
receiver / scorer ``S``, the optimal encoder maximizes the mutual information
``MI(B; S(B))`` between its encoded representation ``B`` and the receiver's
output ``S(B)``, NOT generic reconstruction fidelity ``MI(B; X)`` where ``X`` is
the source. When the scorer is known and fixed (as in the comma video
compression contest, where ``S = SegNet + PoseNet`` is published in
``upstream/modules.py``), training the encoder against ``d(S(B), S(X))``
(scorer-output distortion) is mathematically equivalent to maximizing
``MI(B; S(B))`` subject to the rate constraint.

Operationally this primitive applies the contest eval-roundtrip to the
predicted RGB pair, then runs both the prediction and the ground truth through
the canonical ``score_pair_components`` helper to obtain the SegNet and
PoseNet distortion terms. The returned ``(seg_term, pose_term)`` pair is
gradient-reachable end-to-end into the predicted RGB inputs (both eval-
roundtrip and the canonical preprocess_input pathway preserve gradient flow
per CLAUDE.md "eval_roundtrip — non-negotiable" + Catalog #164).

This is the **canonical** Atick-Redlich primitive for any substrate that ships
a full RGB renderer and trains against the contest scorer pair. It is distinct
from:

- :func:`tac.codec.cooperative_receiver.predictive_coding.predictive_coding_residual_term`
  (Rao-Ballard 1999): predictive coding penalizes the magnitude of the
  residual the world model failed to predict; that is a top-down prediction-
  error penalty, NOT mutual-information maximization.
- :mod:`tac.packet_compiler.wyner_ziv` (sister-subagent scope, Slepian-Wolf-
  Wyner-Ziv 1973-1976): Wyner-Ziv is a SOURCE-CODING bound for cooperative
  receivers with side information; Atick-Redlich is the efficient-coding
  RECEIVER-MATCHING principle that motivates training the encoder against the
  scorer rather than reconstruction. Both are "cooperative-receiver" ideas in
  the broad sense, but they are mathematically distinct and target different
  pipeline surfaces.

References
----------
- Atick & Redlich, "Towards a theory of early visual processing", Neural
  Computation 2(3):308-320, 1990.
- Atick & Redlich, "What does the retina know about natural scenes?", Neural
  Computation 4(2):196-210, 1992.

Cross-references
----------------
- :func:`tac.substrates.score_aware_common.score_pair_components` — the
  canonical scorer-loss helper this primitive delegates to.
- :mod:`tac.differentiable_eval_roundtrip` — eval-roundtrip pipeline that
  preserves gradient flow into the predicted RGB inputs.
- CLAUDE.md "HNeRV parity discipline" lesson L6 (score-domain Lagrangian) +
  lesson L8 (eval-roundtrip + differentiable scorer-preprocess training).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components,
)


@dataclass(frozen=True)
class AtickRedlichWeights:
    """Lagrangian weights for the cooperative-receiver loss terms.

    Defaults match the contest formula
    ``L = beta_seg * d_seg + gamma_pose * sqrt(d_pose)`` so callers that just
    want "the scorer-loss as defined by upstream" get it without thinking
    about weights.

    Args:
        beta_seg: Weight on the SegNet distortion term. Contest default 100.0.
        gamma_pose: Weight on the ``sqrt(d_pose)`` term. Contest default
            ``sqrt(10)`` per the upstream evaluator formula.
        pose_weight_scale: Multiplicative tilt on the pose term. 1.0 = pure
            contest formula. Operating-point experiments (e.g., PR106 r2 has
            2.71x marginal pose value) can tilt without redefining gamma.
    """

    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.beta_seg < 0.0:
            raise ValueError(f"beta_seg must be >= 0; got {self.beta_seg}")
        if self.gamma_pose < 0.0:
            raise ValueError(f"gamma_pose must be >= 0; got {self.gamma_pose}")
        if self.pose_weight_scale < 0.0:
            raise ValueError(
                f"pose_weight_scale must be >= 0; got {self.pose_weight_scale}"
            )


class _ScorerLike(Protocol):
    """Structural contract for a scorer compatible with the canonical pathway.

    The canonical helper :func:`score_pair_components` requires a
    ``preprocess_input`` callable (see Catalog #164). This Protocol is for
    documentation; runtime contract enforcement happens inside
    :func:`score_pair_components` via :class:`ScoreAwareScorerContractError`.
    """

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, x: torch.Tensor): ...  # pragma: no cover - structural


@dataclass(frozen=True)
class CooperativeReceiverOutput:
    """Result of one Atick-Redlich primitive invocation.

    Attributes:
        cooperative_loss: Scalar tensor combining
            ``beta_seg * seg + gamma_pose * pose_weight_scale * sqrt(pose)``
            with full gradient flow into the predicted RGB inputs.
        seg_term: Scalar SegNet distortion term (gradient-reachable).
        pose_term: Scalar PoseNet distortion term (gradient-reachable).
        pose_sqrt: Scalar ``sqrt(max(pose_term, eps))`` (gradient-reachable).
    """

    cooperative_loss: torch.Tensor
    seg_term: torch.Tensor
    pose_term: torch.Tensor
    pose_sqrt: torch.Tensor


def _coerce_eval_roundtrip(
    apply_eval_roundtrip: bool,
    eval_roundtrip_fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the eval-roundtrip callable per CLAUDE.md non-negotiable.

    ``apply_eval_roundtrip=False`` is forbidden per CLAUDE.md
    "eval_roundtrip — non-negotiable"; this helper raises ``ValueError`` on
    that case.  When ``eval_roundtrip_fn`` is None we lazily import the
    canonical ``apply_eval_roundtrip_during_training`` so callers do not pay
    the import cost at module-load time.
    """

    if not apply_eval_roundtrip:
        raise ValueError(
            "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
            "'eval_roundtrip — non-negotiable'"
        )
    if eval_roundtrip_fn is not None:
        return eval_roundtrip_fn
    from tac.differentiable_eval_roundtrip import (
        apply_eval_roundtrip_during_training,
    )

    return apply_eval_roundtrip_during_training


def cooperative_receiver_loss(
    rgb_0: torch.Tensor,
    rgb_1: torch.Tensor,
    gt_rgb_0: torch.Tensor,
    gt_rgb_1: torch.Tensor,
    *,
    seg_scorer: torch.nn.Module,
    pose_scorer: torch.nn.Module,
    weights: AtickRedlichWeights | None = None,
    apply_eval_roundtrip: bool = True,
    eval_roundtrip_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    pose_sqrt_floor: float = 1e-12,
) -> CooperativeReceiverOutput:
    """Compute the Atick-Redlich cooperative-receiver loss for one pair batch.

    The encoder is cooperatively trained against the FIXED known scorer
    ``(seg_scorer, pose_scorer)``. The loss combines the SegNet and PoseNet
    distortion terms via the contest Lagrangian formula
    ``beta_seg * d_seg + gamma_pose * pose_weight_scale * sqrt(d_pose)``.

    The RATE term is intentionally NOT included here — rate is a non-
    differentiable archive-byte count that lives in the training-loop
    Lagrangian outside this primitive. Substrates compose the cooperative
    loss with their own ``alpha_rate * archive_bytes / N`` term.

    Args:
        rgb_0, rgb_1: Predicted RGB pair tensors ``(B, 3, H, W)`` in
            ``[0, 255]``. Gradient flows into these tensors.
        gt_rgb_0, gt_rgb_1: Ground-truth RGB pair tensors ``(B, 3, H, W)``
            in ``[0, 255]``. Gradient does NOT flow into the targets.
        seg_scorer: Contest SegNet module (``smp.Unet`` or stand-in)
            exposing ``preprocess_input`` per the Catalog #164 contract.
        pose_scorer: Contest PoseNet module (FastViT-T12 or stand-in)
            exposing ``preprocess_input`` per the same contract.
        weights: Lagrangian weights; defaults to the contest formula.
        apply_eval_roundtrip: Must be True per CLAUDE.md non-negotiable;
            ValueError on False.
        eval_roundtrip_fn: Optional override for the eval-roundtrip
            callable (e.g., for tests that want to inject a no-op). When
            None the canonical
            :func:`tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training`
            is used.
        pose_sqrt_floor: Minimum value passed to ``sqrt`` to avoid NaN at
            ``pose_term=0``. Default 1e-12 matches the time-traveler.

    Returns:
        :class:`CooperativeReceiverOutput` carrying the scalar
        cooperative loss and the unweighted seg / pose / pose_sqrt
        components for diagnostic logging.
    """

    if weights is None:
        weights = AtickRedlichWeights()
    eval_roundtrip = _coerce_eval_roundtrip(apply_eval_roundtrip, eval_roundtrip_fn)

    rgb_0_rt = eval_roundtrip(rgb_0)
    rgb_1_rt = eval_roundtrip(rgb_1)

    seg_term, pose_term = score_pair_components(
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        rgb_0_rt=rgb_0_rt,
        rgb_1_rt=rgb_1_rt,
        gt_rgb_0=gt_rgb_0,
        gt_rgb_1=gt_rgb_1,
    )
    pose_sqrt = torch.sqrt(pose_term.clamp(min=pose_sqrt_floor))
    cooperative_loss = (
        weights.beta_seg * seg_term
        + weights.gamma_pose * weights.pose_weight_scale * pose_sqrt
    )
    return CooperativeReceiverOutput(
        cooperative_loss=cooperative_loss,
        seg_term=seg_term,
        pose_term=pose_term,
        pose_sqrt=pose_sqrt,
    )


__all__ = [
    "AtickRedlichWeights",
    "CooperativeReceiverOutput",
    "cooperative_receiver_loss",
]
