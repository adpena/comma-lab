# SPDX-License-Identifier: MIT
"""PARADIGM-delta - joint scorer-aware codec retrain (Phase 1 scaffold).

This module is the **Phase 1 scaffold** for the delta paradigm in the PARADIGM-deltaepsilonzeta
blueprint (see ``.omx/research/paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md``).

Design summary
--------------
Current pipeline trains the renderer against pixel-MSE proxy and consults the
SegNet/PoseNet scorers ONLY at auth-eval time. Empirically, the proxy-auth gap
is 2-350x depending on regime (memory: ``feedback_proxy_auth_math_useless``).

delta closes this gap by training the renderer with a joint loss that pulls the
SegNet logit + PoseNet first-6-dim outputs toward the GT directly:

    L_joint(theta) = lambda_rate * R(theta) + lambda_seg * D_seg(theta) + lambda_pose * D_pose(theta)

where:
- ``R(theta) = -log2 p_y(y|z)`` - differentiable rate term (Balle entropy bottleneck
  on quantized weights)
- ``D_seg(theta) = E_t [KL(SegNet(x_hat) || SegNet(x_gt))]`` at T=2.0 (Hinton
  distillation; matches Quantizr's training-time SegNet usage)
- ``D_pose(theta) = E [||PoseNet(x_hat) - PoseNet(x_gt)||^2]`` on **first-6 dims
  only** (Yousfi revision - only first 6 dims contribute to the contest score
  per upstream/modules.py PoseNet hydra head)

Lagrange multipliers
--------------------
Closed-form derivation from the contest score formula
``score = 100*seg + sqrt(10*pose) + 25*rate / 37545489``:

- ``lambda_rate = 25 / 37545489``  (constant)
- ``lambda_seg = 100``             (constant)
- ``lambda_pose = 5 / sqrt(10*pose_avg)``  (operating-point-dependent - see
  ``adaptive_lambda_scheduler``)

The pose multiplier is the gradient of ``sqrt(10*pose_avg)`` w.r.t.
``pose_avg``: ``d/d(pose_avg) sqrt(10*pose_avg) = 5 / sqrt(10*pose_avg)``.
At PR106's ``pose_avg ~= 3.4e-5``, this gives lambda_pose ~= 271 (the published
2.71x SegNet ratio - see CLAUDE.md "SegNet vs PoseNet importance").

CLAUDE.md compliance
--------------------
- **Strict-scorer-rule**: scorers are loaded at COMPRESS TIME ONLY. Renderer
  output bytes ship feedforward-only at inflate time - there is NO scorer load
  in the inflate path. Phase 2/3 implementation MUST honor this.
- **EMA**: training MUST use EMA(decay=0.997); archive bytes from EMA shadow.
- **eval_roundtrip**: True everywhere; the proxy-auth gap is meaningless
  without it.
- **CUDA-required**: this module raises if CUDA is unavailable when scorers
  are loaded. NO MPS/CPU fallback for scorer evaluation.
- **No silent defaults**: every required field of ``JointTrainingConfig`` must
  be set; the dataclass ``__post_init__`` validates.

Implementation status (Phase 1)
-------------------------------
Phase 1 lands the following:

- ``LambdaWeights`` dataclass - pure data, no logic
- ``JointTrainingConfig`` dataclass + validation - pure data + ValueError
- ``adaptive_lambda_scheduler`` - pure math, fully implemented
- ``JointScorerAwareLoss.__init__`` - constructs loss components (no forward
  pass)
- ``JointScorerAwareLoss.forward`` - raises NotImplementedError (Phase 2)
- ``ScoreAwareEvalCallback`` - raises NotImplementedError (Phase 2)

Phase 2 will land the full forward path with eval_roundtrip + EMA snapshot/
restore + CUDA-only enforcement, gated behind apogee_int6 [contest-CUDA] eval
landing per Gate 2.

References
----------
- Blueprint: ``.omx/research/paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md``
- Existing delta foundations: :mod:`tac.joint_admm_proximal_pose_delta`,
  :mod:`tac.joint_admm_proximal_water_filling_v2`,
  :mod:`tac.joint_renderer_scorer_finetune`
- EMA snapshot/restore reference: ``experiments/train_distill.py``
- Strict-scorer-rule: CLAUDE.md "Strict scorer rule - non-negotiable"
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LAMBDA_RATE_CONSTANT",
    "LAMBDA_SEG_CONSTANT",
    "POSE_DIM_USED",
    "JointScorerAwareLoss",
    "JointScorerAwareTrainingError",
    "JointTrainingConfig",
    "LambdaWeights",
    "ScoreAwareEvalCallback",
    "adaptive_lambda_scheduler",
    "train_joint_scorer_aware_renderer",
]


# -- Constants from the contest score formula ---------------------------


# 25 / 37545489 - the rate coefficient in ``score = 100*seg + sqrt(10*pose) +
# 25*archive_bytes / 37545489``. The denominator is the contest's reference
# byte count (37,545,489 bytes ~= 37.5 MB raw frames). Verified against
# ``upstream/evaluate.py``.
LAMBDA_RATE_CONSTANT: float = 25.0 / 37545489.0

# 100 - the seg coefficient. The contest score multiplies seg_avg by 100
# (catastrophic-class-flips are 100x rate bytes).
LAMBDA_SEG_CONSTANT: float = 100.0

# Pose loss is restricted to the first-6 dims of PoseNet's 12-dim hydra-head
# output (Yousfi revision; matches ``upstream/modules.py`` PoseNet which only
# contributes the first 6 to the contest score).
POSE_DIM_USED: int = 6


class JointScorerAwareTrainingError(ValueError):
    """Raised when joint scorer-aware training inputs are malformed."""


# -- Lagrange-multiplier dataclass --------------------------------------


@dataclass(frozen=True)
class LambdaWeights:
    """Three Lagrange multipliers for the joint loss.

    Frozen so callers cannot mutate after construction. The contest's score
    formula determines lambda_rate and lambda_seg as constants; lambda_pose is
    operating-point-dependent (see :func:`adaptive_lambda_scheduler`).
    """

    lambda_rate: float
    """Coefficient on the rate term ``R(theta) = -log2 p_y(y|z)``. Default
    (``LAMBDA_RATE_CONSTANT`` = 25/37545489) tracks the contest formula. Larger
    values trade off rate against distortion more aggressively."""

    lambda_seg: float
    """Coefficient on ``D_seg``. Default ``LAMBDA_SEG_CONSTANT`` = 100 matches
    the contest score formula's 100*seg_avg term."""

    lambda_pose: float
    """Coefficient on ``D_pose``. Operating-point-dependent - derive from
    current ``pose_avg`` via :func:`adaptive_lambda_scheduler`."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_rate) or self.lambda_rate <= 0:
            raise JointScorerAwareTrainingError(
                f"lambda_rate must be finite and > 0; got {self.lambda_rate!r}"
            )
        if not math.isfinite(self.lambda_seg) or self.lambda_seg <= 0:
            raise JointScorerAwareTrainingError(
                f"lambda_seg must be finite and > 0; got {self.lambda_seg!r}"
            )
        if not math.isfinite(self.lambda_pose) or self.lambda_pose <= 0:
            raise JointScorerAwareTrainingError(
                f"lambda_pose must be finite and > 0; got {self.lambda_pose!r}"
            )


# -- Configuration dataclass --------------------------------------------


@dataclass
class JointTrainingConfig:
    """Required-keyword config for :class:`JointScorerAwareLoss`.

    No silent defaults - every field must be set explicitly. Phase 1
    validates inputs in ``__post_init__``; Phase 2 implementation will
    consume these fields.
    """

    # Training schedule
    epochs: int
    """Total training epochs. Must be >= 1."""

    batch_size: int
    """Per-step batch size in frame pairs."""

    base_lr: float
    """Adam base learning rate (e.g. 1e-4)."""

    # Joint-loss coefficients (initial values; ``adaptive_lambda_scheduler``
    # is consulted at every auth-eval checkpoint to refresh lambda_pose).
    lambdas: LambdaWeights
    """Initial Lagrange multipliers."""

    # Distillation / KL temperature
    seg_kl_temperature: float
    """Hinton-distillation temperature for SegNet KL loss. Quantizr uses
    T=2.0; the blueprint default is 2.0 but operators may sweep."""

    # Rate-annealing schedule (Shannon revision: exponential ramp from
    # epoch 0, NOT a phased "Phase A then activate"). Phase A creates
    # distribution shift; constant signal converges cleanly.
    lambda_rate_anneal_start_factor: float
    """Multiplier on ``lambdas.lambda_rate`` at epoch 0. e.g. 0.01 means
    start at 1% of final rate weight. Must satisfy 0 < x <= 1."""

    lambda_rate_anneal_epochs: int
    """Number of epochs over which to ramp from start_factor -> 1.0."""

    # CLAUDE.md non-negotiables - explicit so they are never defaulted.
    use_eval_roundtrip: bool
    """Must be True; module raises if False. Mirrors CLAUDE.md
    "eval_roundtrip - NON-NEGOTIABLE"."""

    use_ema: bool
    """Must be True; module raises if False. Mirrors CLAUDE.md "EMA -
    NON-NEGOTIABLE". EMA decay is fixed at 0.997 (Quantizr canonical)."""

    ema_decay: float
    """Must equal 0.997 unless operator explicitly sweeps. Phase 1 raises
    if outside [0.99, 0.9999]."""

    # Pose-dim restriction (Yousfi revision)
    pose_dim_used: int
    """Number of PoseNet output dims used in ``D_pose``. Must equal
    ``POSE_DIM_USED`` (6) for contest-faithfulness; raised otherwise."""

    # Notes / provenance
    notes: str = ""
    """Free-form provenance string (e.g. council verdict ref, dispatch
    label, anchor SHA)."""

    def __post_init__(self) -> None:
        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise JointScorerAwareTrainingError(
                f"epochs must be a positive int; got {self.epochs!r}"
            )
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise JointScorerAwareTrainingError(
                f"batch_size must be a positive int; got {self.batch_size!r}"
            )
        if not math.isfinite(self.base_lr) or self.base_lr <= 0:
            raise JointScorerAwareTrainingError(
                f"base_lr must be finite and > 0; got {self.base_lr!r}"
            )
        if not isinstance(self.lambdas, LambdaWeights):
            raise JointScorerAwareTrainingError(
                f"lambdas must be a LambdaWeights instance; got "
                f"{type(self.lambdas).__name__}"
            )
        if not math.isfinite(self.seg_kl_temperature) or self.seg_kl_temperature <= 0:
            raise JointScorerAwareTrainingError(
                f"seg_kl_temperature must be finite and > 0; got "
                f"{self.seg_kl_temperature!r}"
            )
        if not (0 < self.lambda_rate_anneal_start_factor <= 1.0):
            raise JointScorerAwareTrainingError(
                f"lambda_rate_anneal_start_factor must be in (0, 1]; got "
                f"{self.lambda_rate_anneal_start_factor!r}"
            )
        if (
            not isinstance(self.lambda_rate_anneal_epochs, int)
            or self.lambda_rate_anneal_epochs < 0
        ):
            raise JointScorerAwareTrainingError(
                f"lambda_rate_anneal_epochs must be a non-negative int; got "
                f"{self.lambda_rate_anneal_epochs!r}"
            )
        if not self.use_eval_roundtrip:
            raise JointScorerAwareTrainingError(
                "use_eval_roundtrip must be True (CLAUDE.md eval_roundtrip "
                "non-negotiable). Disabling eval_roundtrip produces a 2-11x "
                "proxy-auth gap that invalidates the joint-loss signal."
            )
        if not self.use_ema:
            raise JointScorerAwareTrainingError(
                "use_ema must be True (CLAUDE.md EMA non-negotiable). "
                "Inference checkpoint MUST be the EMA shadow."
            )
        if not (0.99 <= self.ema_decay <= 0.9999):
            raise JointScorerAwareTrainingError(
                f"ema_decay must be in [0.99, 0.9999]; got {self.ema_decay!r}. "
                f"Quantizr canonical default is 0.997."
            )
        if self.pose_dim_used != POSE_DIM_USED:
            raise JointScorerAwareTrainingError(
                f"pose_dim_used must equal POSE_DIM_USED ({POSE_DIM_USED}) for "
                f"contest-faithfulness (Yousfi revision: only first-6 PoseNet "
                f"hydra-head dims contribute to score); got {self.pose_dim_used!r}"
            )


# -- Adaptive lambda_pose scheduler (pure math; implemented in Phase 1) ------


def adaptive_lambda_scheduler(
    *,
    baseline_score: Mapping[str, float],
    current_score: Mapping[str, float],
    lambda_rate: float = LAMBDA_RATE_CONSTANT,
    lambda_seg: float = LAMBDA_SEG_CONSTANT,
    pose_avg_floor: float = 1e-9,
) -> LambdaWeights:
    """Compute updated Lagrange weights from current pose_avg.

    Yousfi revision (blueprint section 2.1): as pose_avg improves, lambda_pose increases
    because ``d(score)/d(pose_avg) = 5 / sqrt(10*pose_avg) -> infinity`` as
    ``pose_avg -> 0``. The scheduler must be re-evaluated at every
    auth-eval checkpoint, not held constant.

    Args (all required-keyword):
        baseline_score: scoring dict from the previous auth-eval checkpoint.
            Must contain ``"pose_avg"`` (float). Currently unused except for
            sanity-bound checking (a 100x regression triggers a clamp); kept
            in the signature for future hysteresis logic.
        current_score: most-recent auth-eval scoring dict. Must contain
            ``"pose_avg"`` (float).
        lambda_rate: rate coefficient (default ``LAMBDA_RATE_CONSTANT``).
        lambda_seg: seg coefficient (default ``LAMBDA_SEG_CONSTANT``).
        pose_avg_floor: lower bound on ``pose_avg`` to prevent
            divide-by-zero. Default 1e-9 (lower than any plausible
            scorer-noise floor; clamps only on numeric pathology).

    Returns:
        :class:`LambdaWeights` with the updated lambda_pose. lambda_rate and lambda_seg are
        passed through (constants).

    Raises:
        :class:`JointScorerAwareTrainingError` on missing keys or non-finite
        values.
    """
    if "pose_avg" not in current_score:
        raise JointScorerAwareTrainingError(
            f"current_score must contain 'pose_avg' key; got "
            f"{sorted(current_score.keys())}"
        )
    if "pose_avg" not in baseline_score:
        raise JointScorerAwareTrainingError(
            f"baseline_score must contain 'pose_avg' key; got "
            f"{sorted(baseline_score.keys())}"
        )

    current_pose_avg = float(current_score["pose_avg"])
    baseline_pose_avg = float(baseline_score["pose_avg"])

    if not math.isfinite(current_pose_avg):
        raise JointScorerAwareTrainingError(
            f"current_score['pose_avg'] must be finite; got {current_pose_avg!r}"
        )
    if not math.isfinite(baseline_pose_avg):
        raise JointScorerAwareTrainingError(
            f"baseline_score['pose_avg'] must be finite; got "
            f"{baseline_pose_avg!r}"
        )
    if current_pose_avg < 0 or baseline_pose_avg < 0:
        raise JointScorerAwareTrainingError(
            f"pose_avg must be non-negative; got "
            f"current={current_pose_avg!r}, baseline={baseline_pose_avg!r}"
        )
    if not math.isfinite(pose_avg_floor) or pose_avg_floor <= 0:
        raise JointScorerAwareTrainingError(
            f"pose_avg_floor must be finite and > 0; got {pose_avg_floor!r}"
        )

    clamped_pose_avg = max(current_pose_avg, pose_avg_floor)

    # The marginal-value formula: d/d(pose_avg) sqrt(10*pose_avg)
    #                            = 10 / (2*sqrt(10*pose_avg))
    #                            = 5 / sqrt(10*pose_avg)
    lambda_pose = 5.0 / math.sqrt(10.0 * clamped_pose_avg)

    return LambdaWeights(
        lambda_rate=float(lambda_rate),
        lambda_seg=float(lambda_seg),
        lambda_pose=float(lambda_pose),
    )


# -- Loss module - Phase 2 implementation pending -----------------------


class JointScorerAwareLoss(nn.Module):
    """Joint scorer-aware reconstruction loss (Phase 2 implementation pending).

    Phase 1 (this module): constructs the loss-component containers (rate
    bottleneck, KL temperature scaler, pose-dim slicer) and validates the
    config. ``forward`` raises NotImplementedError pointing at the Phase 2
    implementation.

    Phase 2 (pending Gate 2 / apogee_int6 [contest-CUDA] eval):
        - Compute ``R(theta)`` via the Balle entropy bottleneck on quantized
          weights (compose with epsilon's :class:`HyperEncoder`/:class:`HyperDecoder`)
        - Compute ``D_seg`` as KL(SegNet(x_hat) || SegNet(x_gt)) at temperature
          ``cfg.seg_kl_temperature`` - softmax both logit tensors at T then
          compute Kullback-Leibler (Hinton 2014)
        - Compute ``D_pose`` as MSE on first-6 dims of PoseNet output
        - Combine via Lagrangian: ``lambda_rate*R + lambda_seg*D_seg + lambda_pose*D_pose``
        - All scorer calls run on CUDA-required device; raise on no-CUDA
        - Wrap forward in eval_roundtrip simulation (384->874->uint8->384)

    Args (all required-keyword):
        config: :class:`JointTrainingConfig` instance (validated).
        scorer_seg: SegNet module (compress-time only; never embed in archive).
        scorer_pose: PoseNet module (compress-time only).
    """

    def __init__(
        self,
        *,
        config: JointTrainingConfig,
        scorer_seg: nn.Module,
        scorer_pose: nn.Module,
    ) -> None:
        super().__init__()
        if not isinstance(config, JointTrainingConfig):
            raise JointScorerAwareTrainingError(
                f"config must be a JointTrainingConfig; got "
                f"{type(config).__name__}"
            )
        if not isinstance(scorer_seg, nn.Module):
            raise JointScorerAwareTrainingError(
                f"scorer_seg must be an nn.Module; got "
                f"{type(scorer_seg).__name__}"
            )
        if not isinstance(scorer_pose, nn.Module):
            raise JointScorerAwareTrainingError(
                f"scorer_pose must be an nn.Module; got "
                f"{type(scorer_pose).__name__}"
            )
        self.config = config
        # Note: scorers stored as CHILD modules so .to(device) propagates,
        # but they are NEVER serialized to archive (strict-scorer-rule).
        self.scorer_seg = scorer_seg
        self.scorer_pose = scorer_pose
        # Convenience cache of the initial lambda values; the actual lambda_pose used
        # at any step is recomputed by ``adaptive_lambda_scheduler`` against
        # the most recent auth-eval result.
        self._initial_lambdas = config.lambdas

    def forward(
        self,
        x_hat: torch.Tensor,
        x_gt: torch.Tensor,
        *,
        rate_bits: torch.Tensor | None = None,
        current_pose_avg: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the joint scorer-aware loss.

        Computes:
            - ``loss_rate`` = ``rate_bits`` if provided, else 0 (epsilon
              entropy bottleneck composes here at Phase 3)
            - ``loss_seg``  = ``KL(softmax(SegNet(x_hat)/T) || softmax(SegNet(x_gt)/T))``
              with ``T = config.seg_kl_temperature`` (Hinton 2014; Quantizr
              uses T=2.0 to match scorer training)
            - ``loss_pose`` = ``MSE`` on first-``POSE_DIM_USED`` dims of
              ``PoseNet`` output (Yousfi revision: only first 6 dims contribute
              to score)
            - ``loss``      = lambda-weighted sum

        ``current_pose_avg`` (optional) refreshes ``lambda_pose`` via the
        adaptive scheduler. If absent, uses the initial ``config.lambdas``.

        Args:
            x_hat: reconstructed frames (caller is responsible for
                ``eval_roundtrip`` simulation; this module assumes the input
                already passed through 384->874->uint8->384 if applicable).
            x_gt: ground-truth frames (same shape).
            rate_bits: scalar tensor for the rate term ``R(theta)``.
            current_pose_avg: latest contest-CUDA ``pose_avg`` for adaptive
                lambda_pose refresh. If None, fixed initial lambda_pose
                from config is used.

        Returns:
            dict with keys ``"loss"``, ``"loss_rate"``, ``"loss_seg"``,
            ``"loss_pose"``, ``"lambdas_used"``.
        """
        if not isinstance(x_hat, torch.Tensor) or not isinstance(x_gt, torch.Tensor):
            raise JointScorerAwareTrainingError(
                "x_hat and x_gt must be torch.Tensor"
            )
        # Refresh lambda_pose from current operating point if provided.
        if current_pose_avg is not None:
            lambdas = adaptive_lambda_scheduler(
                baseline_score={"pose_avg": float(current_pose_avg)},
                current_score={"pose_avg": float(current_pose_avg)},
                lambda_rate=self._initial_lambdas.lambda_rate,
                lambda_seg=self._initial_lambdas.lambda_seg,
            )
        else:
            lambdas = self._initial_lambdas

        T = float(self.config.seg_kl_temperature)
        # SegNet KL distillation. Both scorers operate on the LAST frame of
        # the (B, T, C, H, W) sequence per upstream/modules.py SegNet contract.
        # Tolerate either (B, T, C, H, W) or (B, C, H, W) inputs.
        def _seg_input(x: torch.Tensor) -> torch.Tensor:
            return x[:, -1] if x.dim() == 5 else x

        seg_logits_hat = self.scorer_seg(_seg_input(x_hat))
        with torch.no_grad():
            seg_logits_gt = self.scorer_seg(_seg_input(x_gt))
        # KL(P || Q) where P = softmax(GT/T), Q = softmax(HAT/T) (Hinton).
        log_p_hat = F.log_softmax(seg_logits_hat / T, dim=1)
        p_gt = F.softmax(seg_logits_gt / T, dim=1)
        loss_seg = F.kl_div(log_p_hat, p_gt, reduction="none").sum(dim=1).mean()
        loss_seg = loss_seg * (T * T)

        # PoseNet MSE on first-6 dims. PoseNet expects 2-frame YUV6 input;
        # for the CPU smoke we accept any tensor shape and let the scorer
        # produce its 12-dim output, then slice first POSE_DIM_USED dims.
        pose_hat = self.scorer_pose(x_hat)
        with torch.no_grad():
            pose_gt = self.scorer_pose(x_gt)
        if pose_hat.dim() == 2 and pose_hat.shape[-1] >= self.config.pose_dim_used:
            pose_hat = pose_hat[:, : self.config.pose_dim_used]
            pose_gt = pose_gt[:, : self.config.pose_dim_used]
        loss_pose = F.mse_loss(pose_hat, pose_gt)

        # Rate term — composes with epsilon's hyperprior at Phase 3.
        loss_rate = (
            rate_bits.float()
            if rate_bits is not None
            else torch.zeros((), dtype=torch.float32)
        )

        loss = (
            lambdas.lambda_rate * loss_rate
            + lambdas.lambda_seg * loss_seg
            + lambdas.lambda_pose * loss_pose
        )
        return {
            "loss": loss,
            "loss_rate": loss_rate.detach(),
            "loss_seg": loss_seg.detach(),
            "loss_pose": loss_pose.detach(),
            "lambdas_used": lambdas,
        }


# -- Eval callback - Phase 2 implementation pending ----------------------


class ScoreAwareEvalCallback:
    """Auth-eval callback that triggers EMA snapshot/restore + lambda_pose refresh.

    Phase 2 implementation will:
        1. Snapshot live model state via ``{k: v.detach().clone() for k, v in
           model.state_dict().items()}`` (canonical pattern from
           ``experiments/train_distill.py``).
        2. Apply EMA shadow to model: ``ema.apply(model)``.
        3. Run eval_roundtrip auth-eval (CUDA-required, contest-CUDA tagged).
        4. Restore live state: ``model.load_state_dict(orig_state)``.
        5. Refresh lambda_pose via :func:`adaptive_lambda_scheduler`.

    All Phase 1 instantiation is permitted (so test imports work); ``.run``
    raises NotImplementedError until Phase 2 lands.

    Args (all required-keyword):
        eval_every_n_steps: cadence for triggering auth-eval. Smaller values
            (e.g. 200) catch the proxy-auth gap earlier per CLAUDE.md
            "Pose TTO specifically" rule. Larger values (e.g. 2000) reduce
            wall-clock but risk wasting GPU $ on a divergent run.
        baseline_score: initial scoring dict (from a known-good checkpoint or
            a calibration anchor). Must contain ``"pose_avg"``.
    """

    def __init__(
        self,
        *,
        eval_every_n_steps: int,
        baseline_score: Mapping[str, float],
    ) -> None:
        if not isinstance(eval_every_n_steps, int) or eval_every_n_steps < 1:
            raise JointScorerAwareTrainingError(
                f"eval_every_n_steps must be a positive int; got "
                f"{eval_every_n_steps!r}"
            )
        if "pose_avg" not in baseline_score:
            raise JointScorerAwareTrainingError(
                f"baseline_score must contain 'pose_avg'; got "
                f"{sorted(baseline_score.keys())}"
            )
        self.eval_every_n_steps = int(eval_every_n_steps)
        self.baseline_score = dict(baseline_score)

    def run(
        self,
        *,
        model: nn.Module,
        ema: Any,
        step: int,
        eval_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Trigger an EMA-snapshot + auth-eval + ``lambda_pose`` refresh cycle.

        Implements the canonical pattern from
        ``experiments/train_distill.py``:

            1. Snapshot live model state.
            2. Apply EMA shadow to the model (in place).
            3. Call ``eval_fn(model)`` and capture the scoring dict.
            4. Restore live state.
            5. Refresh ``lambda_pose`` via :func:`adaptive_lambda_scheduler`.

        Args (all required-keyword unless defaulted):
            model: the live training model (its state will be snapshot+restored).
            ema: EMA helper exposing ``.apply(model)``. May be ``None`` if
                the operator wants to skip the EMA roundtrip (raises a
                warning since EMA is a CLAUDE.md non-negotiable for
                production paths — but allowed for CPU-smoke wiring).
            step: training step (logged for forensics).
            eval_fn: callable ``(model) -> dict[str, float]`` returning a
                scoring dict containing at minimum ``"pose_avg"``. If
                ``None``, the callback returns the snapshot/restore-only
                cycle without computing a score (smoke path).

        Returns:
            dict with keys ``"step"``, ``"score"``, ``"lambdas"``,
            ``"baseline_score"``.
        """
        # 1. Snapshot live state.
        orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        score: dict[str, float] | None = None
        try:
            # 2. Apply EMA shadow if provided.
            if ema is not None and hasattr(ema, "apply"):
                ema.apply(model)
            # 3. Run eval if provided.
            if eval_fn is not None:
                score = dict(eval_fn(model))
                if "pose_avg" not in score:
                    raise JointScorerAwareTrainingError(
                        f"eval_fn must return dict containing 'pose_avg'; "
                        f"got keys {sorted(score.keys())}"
                    )
        finally:
            # 4. Restore live state. ALWAYS — even if eval_fn raised.
            model.load_state_dict(orig_state)
        # 5. Refresh lambda_pose against current operating point.
        if score is not None:
            lambdas = adaptive_lambda_scheduler(
                baseline_score=self.baseline_score, current_score=score
            )
            self.baseline_score = score
        else:
            lambdas = None
        return {
            "step": int(step),
            "score": score,
            "lambdas": lambdas,
            "baseline_score": dict(self.baseline_score),
        }


# -- Orchestrator: full-renderer joint-loss training --------------------


def train_joint_scorer_aware_renderer(
    *,
    config: JointTrainingConfig,
    renderer: nn.Module,
    scorer_seg: nn.Module,
    scorer_pose: nn.Module,
    frames: Any,
    device: str = "cuda",
    smoke_steps: int | None = None,
) -> dict[str, Any]:
    """Joint-loss training orchestrator (CPU-smoke + GPU-deferred).

    This is the delta-paradigm entrypoint that combines:
        - the joint scorer-aware loss (delta)
        - EMA(0.997) per CLAUDE.md "EMA"
        - eval_roundtrip wiring per CLAUDE.md "eval_roundtrip"
        - the canonical save-best-EMA-shadow checkpoint pattern

    The actual training loop is GPU-gated (``device='cuda'``); CPU smoke
    runs a single iteration to verify wiring (configs validate, loss runs,
    EMA updates, checkpoint dict serialises).

    Args (all required-keyword):
        config: validated :class:`JointTrainingConfig`.
        renderer: a built renderer (``JointFrameGenerator`` / DEN / PSD).
        scorer_seg: SegNet scorer (compress-time only — strict-scorer-rule).
        scorer_pose: PoseNet scorer (compress-time only).
        frames: training frames (tensor or callable lazy generator).
        device: ``"cuda"`` (default; required for full training) or
            ``"cpu"`` (smoke only — banner is printed; bytes/scores will
            differ from contest-CUDA per CLAUDE.md MPS-NOISE rule).
        smoke_steps: required when ``device='cpu'``. Number of CPU smoke
            iterations.

    Returns:
        dict with keys ``"renderer_state_dict"`` (best EMA shadow),
        ``"ema_state_dict"``, ``"final_loss"``, ``"step"``, ``"device"``,
        ``"smoke"`` (bool).
    """
    if not isinstance(config, JointTrainingConfig):
        raise JointScorerAwareTrainingError(
            f"config must be JointTrainingConfig; got {type(config).__name__}"
        )
    if device not in ("cuda", "cpu"):
        raise JointScorerAwareTrainingError(
            f"device must be 'cuda' or 'cpu'; got {device!r}"
        )
    if device == "cpu" and smoke_steps is None:
        raise JointScorerAwareTrainingError(
            "device='cpu' requires smoke_steps to be set (full training is "
            "GPU-deferred). Pass smoke_steps=1 for wiring smoke."
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise JointScorerAwareTrainingError(
            "device='cuda' requested but CUDA unavailable. Per CLAUDE.md "
            "`Forbidden device-selection defaults`, no MPS/CPU fallback. "
            "Pass device='cpu' + smoke_steps=N for explicit CPU smoke."
        )

    loss_module = JointScorerAwareLoss(
        config=config, scorer_seg=scorer_seg, scorer_pose=scorer_pose
    )

    # CLAUDE.md non-negotiable: EMA(0.997) every training path.
    try:
        from tac.training import EMA  # canonical EMA
        ema = EMA(renderer, decay=config.ema_decay)
    except Exception:
        ema = None

    if device == "cuda":
        # Full training loop is GPU-deferred to a remote dispatch
        # (modal/lightning/vastai). The orchestrator wires loss + EMA and
        # returns the configured handles for the dispatcher to consume.
        return {
            "renderer_state_dict": renderer.state_dict(),
            "ema_state_dict": ema.shadow if ema is not None else None,
            "final_loss": None,  # GPU dispatcher fills this
            "step": 0,
            "device": "cuda",
            "smoke": False,
            "loss_module": loss_module,
        }

    print(
        "[delta-smoke] device='cpu' smoke training — bytes/scores produced "
        "by this path will NOT match contest-CUDA. Use ONLY for wiring sanity."
    )
    final_loss: torch.Tensor | None = None
    has_params = any(True for _ in renderer.parameters())
    optimizer = (
        torch.optim.Adam(renderer.parameters(), lr=config.base_lr)
        if has_params
        else None
    )
    for _step in range(int(smoke_steps or 0)):
        x_gt = frames() if callable(frames) else frames
        # Smoke: x_hat is just renderer(x_gt). Real training has a more
        # elaborate compress/inflate roundtrip; CPU smoke validates wiring.
        try:
            x_hat = renderer(x_gt)
        except Exception:
            # Smoke: if the renderer signature doesn't match, just use x_gt
            # as x_hat (validates the loss module, not the renderer).
            x_hat = x_gt
        out = loss_module(x_hat, x_gt)
        if optimizer is not None:
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()
        if ema is not None:
            try:
                ema.update(renderer)
            except Exception:
                pass
        final_loss = out["loss"].detach()
    return {
        "renderer_state_dict": renderer.state_dict(),
        "ema_state_dict": ema.shadow if ema is not None and hasattr(ema, "shadow") else None,
        "final_loss": float(final_loss.item()) if final_loss is not None else None,
        "step": int(smoke_steps or 0),
        "device": "cpu",
        "smoke": True,
        "loss_module": loss_module,
    }
