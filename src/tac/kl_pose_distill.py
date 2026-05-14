# SPDX-License-Identifier: MIT
"""KL pose-axis distillation loss — T20 lateral leap (coherence council 2026-05-09).

Per the portfolio coherence audit
(``feedback_grand_council_portfolio_coherence_journal_grade_20260509.md``
§3.A redundancy disambiguation), T7 + T8 + T11 all attack the SegNet axis.
T20 closes the pose axis with a Hinton-Vinyals-Dean 2014-style soft-target
KL distillation at ``T = 2.0``.

Mathematical form (Hinton et al. 2014 §2):

    L_T20 = T^2 * KL( softmax(student_logits / T) || softmax(teacher_logits / T) )

The ``T^2`` term is the canonical Hinton normalization that preserves the
gradient magnitude scale across temperature settings (the partial derivative
of cross-entropy w.r.t. logits scales as ``1/T``, so squaring ``T`` keeps
the effective learning rate temperature-invariant — see Hinton §2.1
"Note on the gradients").

The PoseNet head outputs a 12-dim vector per pair (``Hydra`` in
``upstream/modules.py:26`` → ``Head('pose', hidden=32, out=12)``); the
contest scorer's ``compute_distortion`` consumes only the first 6 dims as
continuous regression targets (``upstream/modules.py:84``):

    distortion = (out1[:, :6] - out2[:, :6]).pow(2).mean(dim=...)

T20 reframes the 12-dim head output AS LOGITS over a soft simplex via
``softmax(/T)`` and applies KL distillation between teacher and student.
This is a **knowledge-transfer signal** — it asks the student to match the
teacher's *full distribution* over the 12-dim head, not just the first-6
regression mean. Hinton's empirical finding (2014 §3) is that the residual
("dark-knowledge") information in the non-target logits carries useful
generalization signal that a pure regression / hard-target loss discards.

Two operating modes are exposed in the config (per Quantizr's verified 0.33
archive recipe + the regression alternative for empirical comparison):

* ``"distill_softmax_full"``  — softmax over all 12 dims, KL on full distrib.
  This is the canonical Hinton form and the operator-specified default.
* ``"distill_softmax_first6"`` — softmax restricted to first-6 dims (the
  scored region). Tighter to the scorer; loses dark-knowledge from the
  remaining 6 dims.
* ``"regression_mse"``         — the contest-scorer-faithful pure MSE on
  first-6 dims. Provided as a baseline for ablation; no T-scaling.

The teacher pose is stop-gradient by construction (``.detach()`` inside
the loss). T = 2.0 default matches Quantizr's verified Hinton-T2
configuration that fielded the 0.33 archive.

Cross-references
----------------

* Coherence council memo:
  ``feedback_grand_council_portfolio_coherence_journal_grade_20260509.md``
  §"§3.A pose-axis gap → T20 closes".
* Exemplar pattern:
  ``feedback_t11_t13_t19_free_lateral_leaps_landed_20260509.md``.
* Hinton, Vinyals, Dean 2014. *Distilling the Knowledge in a Neural
  Network*. NIPS Deep Learning Workshop.
* Quantizr verified configuration: kl_on_logits(T=2.0) for SegNet during
  staged training; 0.33 archive (CLAUDE.md "Quantizr intelligence" §).
* Pose head contract: ``upstream/modules.py:26`` (Hydra Head 'pose', out=12),
  ``upstream/modules.py:84`` (compute_distortion uses out[..., :6]),
  ``experiments/auth_eval_renderer.py:204`` (renderer pose_dim wiring).

Score-impact prediction
-----------------------

Per the operator brief, predicted Δ score on Phase 1 trainer if substituted
for raw pose-MSE: ``[-0.003, -0.012]``. CPU-axis pose marginal is 5.04× the
CUDA-axis (R_pose calibration; HNeRV cluster anchor); PR106 frontier
operating point puts pose marginal 2.71× over SegNet (CLAUDE.md SegNet vs
PoseNet section), so T20's pose-targeted action is high-EV at this
operating point. **Tagged ``[predicted; T20 KL pose distill at T=2.0]``**
per CLAUDE.md Forbidden Score Claims.

CLAUDE.md compliance
--------------------

* Pure-PyTorch, differentiable, fail-loud on shape/value/eps validation.
* Training-time loss only — NO scorer load (PoseNet weights are referenced
  ONLY for shape contract; the loss takes pre-computed logits).
* Strict-scorer-rule: this is a TRAINING SIGNAL, not an authoritative
  score. Contest score still requires exact CUDA auth eval on archive bytes.
* Archive grammar / score_aware_loss / bolt_on_loc_budget: N/A — T20 is a
  training-time loss, not an archive component (per HNeRV parity discipline
  declaration in the landing memo).
* No MPS-falsification hazard (this is a loss, not a score).
* No silent defaults: every config field documented; explicit eps/T validation.
* No premature kill: default verdict on negative empirical is
  DEFERRED-pending-research (per ``forbidden_premature_kill_*``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

DEFAULT_POSE_HEAD_OUT_DIM = 12
DEFAULT_POSE_SCORED_DIM = 6
DEFAULT_TEMPERATURE = 2.0
DEFAULT_EPS = 1e-9
VALID_MODES = ("distill_softmax_full", "distill_softmax_first6", "regression_mse")


def _validate_temperature(temperature: float) -> float:
    """Validate the distillation temperature."""
    if isinstance(temperature, bool):
        raise ValueError("temperature must be a finite positive number")
    try:
        value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"temperature must be a finite positive number; got {temperature!r}"
        )
    return value


def _validate_eps(eps: float) -> float:
    """Validate the KL-stability epsilon."""
    if isinstance(eps, bool):
        raise ValueError("eps must be a finite number in (0, 1e-3)")
    try:
        value = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError("eps must be a finite number in (0, 1e-3)") from exc
    if not math.isfinite(value) or value <= 0.0 or value >= 1e-3:
        raise ValueError(f"eps must be a finite number in (0, 1e-3); got {eps!r}")
    return value


def _validate_mode(mode: str) -> str:
    """Validate the loss mode token."""
    if not isinstance(mode, str):
        raise ValueError(
            f"mode must be a string in {VALID_MODES}; got {type(mode).__name__}"
        )
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}; got {mode!r}")
    return mode


def _validate_first_n_dims(first_n_dims: int, total_dims: int) -> int:
    """Validate the first-N-dim truncation index."""
    if isinstance(first_n_dims, bool) or not isinstance(first_n_dims, int):
        raise ValueError(
            f"first_n_dims must be a positive integer; got {first_n_dims!r}"
        )
    if first_n_dims <= 0:
        raise ValueError(
            f"first_n_dims must be > 0; got {first_n_dims}"
        )
    if first_n_dims > total_dims:
        raise ValueError(
            f"first_n_dims={first_n_dims} exceeds total_dims={total_dims}"
        )
    return first_n_dims


@dataclass(frozen=True)
class KLPoseDistillConfig:
    """Configuration for the T20 KL pose distillation loss.

    Attributes:
        temperature: Hinton softmax temperature ``T``. Default ``2.0`` matches
            Quantizr's verified 0.33 archive recipe (CLAUDE.md "Quantizr
            intelligence" §). Higher T → softer distribution → more
            dark-knowledge transfer; lower T → sharper, closer to one-hot.
        weight_pose: Scalar multiplier applied to the loss when added to a
            composite trainer objective. Default ``1.0`` (callers responsible
            for setting this in their loss-weighting scheme).
        weight_first_n_dims: How many leading dims of the 12-dim pose head
            to consume. Default ``6`` matches the contest scorer's
            ``compute_distortion`` slice. Used by ``distill_softmax_first6``
            and ``regression_mse``; ignored by ``distill_softmax_full``.
        mode: Loss form — see module docstring for the three valid tokens.
            Default ``"distill_softmax_full"`` is the canonical Hinton form
            and the operator-specified primary path.
        eps: Numerical-stability floor inside the softmax/log_softmax
            computation. Default ``1e-9``.

    Validation: every field is range-checked at construction; invalid values
    raise ``ValueError`` per CLAUDE.md "fail-loud, not silent" rule.
    """

    temperature: float = DEFAULT_TEMPERATURE
    weight_pose: float = 1.0
    weight_first_n_dims: int = DEFAULT_POSE_SCORED_DIM
    mode: str = "distill_softmax_full"
    eps: float = DEFAULT_EPS

    def __post_init__(self) -> None:
        _validate_temperature(self.temperature)
        if isinstance(self.weight_pose, bool):
            raise ValueError(
                f"weight_pose must be a finite non-negative number; got {self.weight_pose!r}"
            )
        try:
            wp = float(self.weight_pose)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"weight_pose must be a finite non-negative number; got {self.weight_pose!r}"
            ) from exc
        if not math.isfinite(wp) or wp < 0.0:
            raise ValueError(
                f"weight_pose must be finite and >= 0; got {self.weight_pose!r}"
            )
        _validate_mode(self.mode)
        _validate_eps(self.eps)
        # weight_first_n_dims validated lazily against logit shape at call site
        # because total_dims depends on the runtime tensor.
        if (
            isinstance(self.weight_first_n_dims, bool)
            or not isinstance(self.weight_first_n_dims, int)
            or self.weight_first_n_dims <= 0
        ):
            raise ValueError(
                f"weight_first_n_dims must be a positive integer; got {self.weight_first_n_dims!r}"
            )


def kl_pose_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    eps: float = DEFAULT_EPS,
    mode: str = "distill_softmax_full",
    first_n_dims: int = DEFAULT_POSE_SCORED_DIM,
) -> torch.Tensor:
    """Compute the T20 KL pose-axis distillation loss.

    Per Hinton-Vinyals-Dean 2014 §2, the soft-target KL is computed with a
    temperature-rescaled softmax on both student and teacher, then the result
    is multiplied by ``T^2`` to preserve gradient magnitude across T values.

    The teacher is automatically stop-gradient (``.detach()``); the student
    receives gradient through the softmax / log-softmax.

    Args:
        student_logits: ``(B, ..., D)`` student pose-head logits. ``B`` is
            the leading batch dim; ``D`` is the head output dim
            (``DEFAULT_POSE_HEAD_OUT_DIM = 12`` for the contest scorer).
        teacher_logits: ``(B, ..., D)`` teacher pose-head logits, same shape
            as ``student_logits``. Will be ``.detach()``ed.
        temperature: Hinton temperature ``T``. Default ``2.0``.
        eps: Stability floor. Default ``1e-9``. Applied as
            ``teacher_softmax.clamp_min(eps)`` so ``teacher.log()`` never
            sees a literal zero (which would yield ``-inf`` and propagate
            ``NaN`` through the KL sum). The teacher softmax is then
            renormalized so the per-sample probabilities still sum to one
            (drift bound: ``D * eps`` ≤ ``1.2e-8`` at canonical D=12).
            Student-side ``log_softmax`` is internally numerically stable
            via PyTorch's log-sum-exp trick and does NOT consume eps.
        mode: One of ``VALID_MODES``. Default ``"distill_softmax_full"``.
        first_n_dims: When mode is ``"distill_softmax_first6"`` or
            ``"regression_mse"``, only the leading ``first_n_dims`` of the
            head output are consumed. Default ``6`` (contest scorer slice).

    Returns:
        Scalar ``torch.Tensor`` of the per-sample mean loss. Result includes
        the canonical ``T^2`` scaling for KL modes; ``regression_mse`` mode
        returns plain MSE without temperature scaling (T is ignored).

    Raises:
        ValueError: shape mismatch, unknown mode, invalid temperature/eps,
            or empty input.
    """
    temperature_value = _validate_temperature(temperature)
    eps_value = _validate_eps(eps)
    mode_value = _validate_mode(mode)
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student_logits shape {tuple(student_logits.shape)} != "
            f"teacher_logits shape {tuple(teacher_logits.shape)}"
        )
    if student_logits.numel() == 0:
        raise ValueError("student_logits / teacher_logits must be non-empty")
    if student_logits.ndim < 2:
        raise ValueError(
            "logits must have at least 2 dims (batch + head); "
            f"got shape {tuple(student_logits.shape)}"
        )
    total_dims = int(student_logits.shape[-1])
    first_n_value = _validate_first_n_dims(first_n_dims, total_dims)

    # Stop-gradient on teacher per Hinton 2014 §2 — knowledge distillation
    # treats the teacher as a fixed target. detach() before any reshape so
    # autograd never tracks teacher.
    teacher_detached = teacher_logits.detach()

    if mode_value == "regression_mse":
        # Pure MSE on first-N dims; T is ignored. Provided as ablation baseline.
        student_slice = student_logits[..., :first_n_value]
        teacher_slice = teacher_detached[..., :first_n_value]
        return F.mse_loss(student_slice, teacher_slice, reduction="mean")

    if mode_value == "distill_softmax_first6":
        student_logits_used = student_logits[..., :first_n_value]
        teacher_logits_used = teacher_detached[..., :first_n_value]
    else:  # "distill_softmax_full"
        student_logits_used = student_logits
        teacher_logits_used = teacher_detached

    # Hinton 2014 KL distillation:
    #   p_teacher = softmax(z_teacher / T)
    #   q_student = softmax(z_student / T)
    #   L = T^2 * KL(p_teacher || q_student)
    #     = T^2 * sum( p_teacher * (log p_teacher - log q_student) )
    #
    # PyTorch's F.kl_div(input=log_q, target=p) computes
    #   sum( p * (log p - log q) ) when log_target=False
    # so we feed log_softmax for the student (input) and softmax for the
    # teacher (target). The eps_value path ensures the explicit numerical
    # safety: log_softmax is internally stable, but we additionally clamp
    # any teacher-softmax value below eps to eps (avoids log(0) in the
    # entropy term and matches the canonical KLDivLoss reduction behavior).
    log_student = F.log_softmax(student_logits_used / temperature_value, dim=-1)
    teacher_soft = F.softmax(teacher_logits_used / temperature_value, dim=-1)
    teacher_soft_safe = teacher_soft.clamp_min(eps_value)
    # Renormalize after the clamp so probabilities still sum to 1 (eps drift
    # is bounded by D * eps which is < 1e-7 for D=12; renormalization keeps
    # the loss interpretable as a true KL divergence).
    teacher_soft_safe = teacher_soft_safe / teacher_soft_safe.sum(dim=-1, keepdim=True)
    # Per-element cross-entropy minus per-element teacher-entropy = pointwise KL.
    # Reduce over the head dim with sum (KL is a sum over the support), then
    # mean over batch + spatial dims so the loss is a scalar comparable across
    # batch sizes.
    kl_per_sample = (teacher_soft_safe * (teacher_soft_safe.log() - log_student)).sum(
        dim=-1
    )
    # T^2 normalization (Hinton 2014 §2.1 gradient note).
    return (temperature_value ** 2) * kl_per_sample.mean()


def apply_kl_pose_distill(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    config: KLPoseDistillConfig,
) -> torch.Tensor:
    """Config-driven entry point for the T20 KL pose distillation loss.

    Convenience wrapper around :func:`kl_pose_distill_loss` that pulls every
    parameter from a :class:`KLPoseDistillConfig` and applies the ``weight_pose``
    multiplier. Trainers should prefer this entry point so that all knobs
    are routed through one validated config object.

    Args:
        student_logits: ``(B, ..., D)`` student pose-head logits.
        teacher_logits: ``(B, ..., D)`` teacher pose-head logits.
        config: Validated :class:`KLPoseDistillConfig`.

    Returns:
        Scalar tensor: ``config.weight_pose * kl_pose_distill_loss(...)``.

    Raises:
        TypeError: ``config`` is not a :class:`KLPoseDistillConfig`.
        ValueError: forwarded from :func:`kl_pose_distill_loss`.
    """
    if not isinstance(config, KLPoseDistillConfig):
        raise TypeError(
            f"config must be a KLPoseDistillConfig; got {type(config).__name__}"
        )
    raw_loss = kl_pose_distill_loss(
        student_logits,
        teacher_logits,
        temperature=config.temperature,
        eps=config.eps,
        mode=config.mode,
        first_n_dims=config.weight_first_n_dims,
    )
    return config.weight_pose * raw_loss


__all__ = [
    "DEFAULT_POSE_HEAD_OUT_DIM",
    "DEFAULT_POSE_SCORED_DIM",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_EPS",
    "VALID_MODES",
    "KLPoseDistillConfig",
    "kl_pose_distill_loss",
    "apply_kl_pose_distill",
]
