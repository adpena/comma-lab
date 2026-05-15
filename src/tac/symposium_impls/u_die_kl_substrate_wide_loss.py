# SPDX-License-Identifier: MIT
"""U-DIE-KL substrate-wide loss term — UNIWARD × DIE × KL distillation.

Per the Grand Reunion symposium 2026-05-15 Phase F Composite #5
(Yousfi + Fridrich + Hinton). Composes UNIWARD-style cost weighting + DIE
blind-region map + KL-distillation (T=2 per Hinton 2014) into a substrate-
agnostic loss term that drops into ANY trainer.

Math contract
=============

Given a substrate's per-pixel reconstruction loss ``L_pixel(p) =
||Î(p) - I(p)||²`` and the contest scorer's per-pixel attention map
``A(p)``, the U-DIE-KL composite loss is

    L_UDIE_KL = sum_p w(p) · L_pixel(p) + λ_KL · KL(student || teacher; T=2)

with the per-pixel weight

    w(p) = α · UNIWARD_cost(p) + β · A(p)
    w(p) := w(p) · (1 - DIE_blind(p))   # zero loss in scorer-blind regions

and

    KL(s || t; T) = T² · sum_c (softmax(s/T))_c · (log softmax(s/T) - log softmax(t/T))_c

(Hinton, Vinyals, Dean 2014 *Distilling the Knowledge in a Neural Network*
§2.1: the temperature ``T`` controls softness of the teacher's class
distribution; ``T² · KL`` rescales gradients to match hard-target
magnitudes.) The default ``T=2`` is the canonical Hinton-derived value;
Quantizr's PR #56 uses the same.

Composition rationale per the symposium:

* UNIWARD weighting concentrates loss in textured, scorer-relevant
  regions (Fridrich)
* DIE map zeros out scorer-blind regions (Yousfi)
* KL distillation transfers soft predictions from the contest scorer
  (Hinton)

The expected ``ΔS`` per substrate is ``[-0.005, -0.020]`` per the
symposium predicted band.

[verified-against: Hinton, Vinyals, Dean 2014 §2.1 (KL distillation +
T² rescaling); Holub, Fridrich, Denemark 2014 (UNIWARD cost); Yousfi
2022 (DIE blind regions); Quantizr PR #56 (canonical KL-T=2 distill).]

Lane: ``lane_symposium_impl_u_die_kl_loss_20260515``.
Catalog #263.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Final

import numpy as np

__all__ = (
    "DEFAULT_KL_TEMPERATURE",
    "DEFAULT_UDIE_LAMBDA_KL",
    "UDIEKLConfig",
    "UDIEKLLossResult",
    "compose_per_pixel_weights",
    "kl_distillation_loss_with_temperature",
    "u_die_kl_substrate_loss",
    "update_from_anchor",
    "weighted_pixel_reconstruction_loss",
)

DEFAULT_KL_TEMPERATURE: Final[float] = 2.0  # Hinton 2014 + Quantizr PR #56 canonical.
DEFAULT_UDIE_LAMBDA_KL: Final[float] = 1.0


@dataclasses.dataclass(frozen=True)
class UDIEKLConfig:
    """Hyperparameters for the U-DIE-KL substrate-wide loss."""

    alpha_uniward_weight: float = 0.5
    beta_attention_weight: float = 0.5
    lambda_kl_distillation: float = DEFAULT_UDIE_LAMBDA_KL
    kl_temperature: float = DEFAULT_KL_TEMPERATURE

    def __post_init__(self) -> None:
        if self.alpha_uniward_weight < 0 or self.beta_attention_weight < 0:
            raise ValueError("alpha and beta must be >= 0")
        if self.lambda_kl_distillation < 0:
            raise ValueError("lambda_kl_distillation must be >= 0")
        if self.kl_temperature <= 0:
            raise ValueError("kl_temperature must be > 0")


@dataclasses.dataclass(frozen=True)
class UDIEKLLossResult:
    """Decomposed U-DIE-KL loss terms + total."""

    weighted_pixel_term: float
    kl_distillation_term: float
    total: float
    notes: str


def compose_per_pixel_weights(
    *,
    uniward: np.ndarray,
    attention: np.ndarray,
    die_blind: np.ndarray,
    config: UDIEKLConfig,
) -> np.ndarray:
    """Combine UNIWARD + attention + DIE into per-pixel loss weights.

    Per the math contract:

        w(p) = α · uniward(p) + β · attention(p)
        w(p) := w(p) · (1 - die_blind(p))
    """
    if uniward.shape != attention.shape or uniward.shape != die_blind.shape:
        raise ValueError("uniward, attention, die_blind must share shape")

    def _normalize(a: np.ndarray) -> np.ndarray:
        a_min, a_max = float(a.min()), float(a.max())
        if a_max - a_min <= 0:
            return np.zeros_like(a, dtype=np.float64)
        return (a.astype(np.float64) - a_min) / (a_max - a_min)

    u_n = _normalize(uniward)
    a_n = _normalize(attention)
    d_clamped = np.clip(die_blind, 0.0, 1.0)
    w = config.alpha_uniward_weight * u_n + config.beta_attention_weight * a_n
    return w * (1.0 - d_clamped)


def weighted_pixel_reconstruction_loss(
    *,
    reconstruction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    """L_pixel = sum_p w(p) · ||Î(p) - I(p)||² / sum_p w(p)."""
    if reconstruction.shape != target.shape:
        raise ValueError("reconstruction and target must share shape")
    if weights.shape != reconstruction.shape[: weights.ndim]:
        raise ValueError("weights must broadcast over leading dimensions of reconstruction")
    if reconstruction.size == 0:
        return 0.0
    diff = reconstruction.astype(np.float64) - target.astype(np.float64)
    sq = (diff**2).reshape(*weights.shape, -1).mean(axis=-1)
    weighted_sum = float((weights * sq).sum())
    weight_total = float(weights.sum())
    if weight_total <= 0:
        return 0.0
    return weighted_sum / weight_total


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=axis, keepdims=True)


def kl_distillation_loss_with_temperature(
    *,
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    temperature: float = DEFAULT_KL_TEMPERATURE,
) -> float:
    """Hinton-style KL distillation: T² · KL(softmax(s/T) || softmax(t/T)).

    Per Hinton, Vinyals, Dean 2014 §2.1 (eq. 2): the gradient magnitudes
    of the soft-target loss scale as 1/T²; multiplying the loss by T²
    keeps gradients comparable to hard-target magnitudes.

    NOTE: the canonical DISTILLATION objective is ``KL(student || teacher)``
    where the STUDENT predicts the teacher's soft distribution. Some
    references use the reverse direction; we follow Hinton 2014 strictly:
    student is the predicting distribution, teacher provides the soft
    target.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher logits must share shape")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if student_logits.size == 0:
        return 0.0
    p_student = _softmax(student_logits / temperature, axis=-1)
    p_teacher = _softmax(teacher_logits / temperature, axis=-1)
    # KL(student || teacher) = sum p_s * log(p_s / p_t)
    eps = 1e-12
    log_ratio = np.log(p_student + eps) - np.log(p_teacher + eps)
    kl_per_sample = (p_student * log_ratio).sum(axis=-1)
    kl_mean = float(kl_per_sample.mean())
    return temperature**2 * kl_mean


def u_die_kl_substrate_loss(
    *,
    config: UDIEKLConfig,
    reconstruction: np.ndarray,
    target: np.ndarray,
    uniward: np.ndarray,
    attention: np.ndarray,
    die_blind: np.ndarray,
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
) -> UDIEKLLossResult:
    """Compose the full U-DIE-KL substrate loss."""
    weights = compose_per_pixel_weights(
        uniward=uniward, attention=attention, die_blind=die_blind, config=config
    )
    pixel_term = weighted_pixel_reconstruction_loss(
        reconstruction=reconstruction, target=target, weights=weights
    )
    kl_term = kl_distillation_loss_with_temperature(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        temperature=config.kl_temperature,
    )
    total = pixel_term + config.lambda_kl_distillation * kl_term
    notes = (
        f"[prediction; first-principles] U-DIE-KL substrate-wide loss. "
        f"α={config.alpha_uniward_weight}, β={config.beta_attention_weight}, "
        f"λ_KL={config.lambda_kl_distillation}, T={config.kl_temperature}. "
        "Catalog #263."
    )
    return UDIEKLLossResult(
        weighted_pixel_term=pixel_term,
        kl_distillation_term=kl_term,
        total=total,
        notes=notes,
    )


def update_from_anchor(
    anchor: Mapping[str, object],
    *,
    config: UDIEKLConfig | None = None,
) -> UDIEKLLossResult | None:
    """Re-emit the U-DIE-KL loss from an anchor's tensors.

    The anchor must carry: ``reconstruction``, ``target``, ``uniward``,
    ``attention``, ``die_blind``, ``student_logits``, ``teacher_logits``
    as ``np.ndarray``.
    """
    needed = (
        "reconstruction",
        "target",
        "uniward",
        "attention",
        "die_blind",
        "student_logits",
        "teacher_logits",
    )
    for key in needed:
        if not isinstance(anchor.get(key), np.ndarray):
            return None
    if config is None:
        config = UDIEKLConfig()
    return u_die_kl_substrate_loss(
        config=config,
        reconstruction=anchor["reconstruction"],  # type: ignore[arg-type]
        target=anchor["target"],  # type: ignore[arg-type]
        uniward=anchor["uniward"],  # type: ignore[arg-type]
        attention=anchor["attention"],  # type: ignore[arg-type]
        die_blind=anchor["die_blind"],  # type: ignore[arg-type]
        student_logits=anchor["student_logits"],  # type: ignore[arg-type]
        teacher_logits=anchor["teacher_logits"],  # type: ignore[arg-type]
    )
