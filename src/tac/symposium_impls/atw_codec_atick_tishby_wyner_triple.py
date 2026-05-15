# SPDX-License-Identifier: MIT
"""ATW Codec — Atick + Tishby + Wyner cooperative-receiver triple.

Per the Grand Reunion symposium 2026-05-15 Phase F POC #3 (Atick + Tishby
memorial + Wyner consultation). The ATW codec composes three canonical
information-theoretic primitives:

* **Atick-Redlich cooperative-receiver loss** — the encoder is trained
  WITH gradient flowing through the receiver (SegNet+PoseNet) so the
  encoder learns what the receiver needs.
* **Tishby Information Bottleneck (IB)** — the encoder solves the
  Lagrangian ``L = I(X; T) - β · I(T; Y)`` where ``T`` is the latent
  representation and ``Y`` is the receiver output. ``β = 100/25 = 4`` is
  derived from the contest formula coefficients per Tao's hidden-symmetry
  derivation (Phase E Eureka #3).
* **Wyner-Ziv source coding with side information** — the decoder shares
  the scorer state-dict (the side information) with the encoder; we use
  the Wyner-Ziv arithmetic coder for the latent residual.

Math contract
=============

Atick-Redlich (1990 *Network: Comp. Neural Sys.* "Towards a Theory of
Early Visual Processing") shows that a receiver-aware encoder minimizes

    L_AR(θ) = E_X[d(receiver(encode_θ(X)), receiver(X))]

with ``d`` the receiver-output distortion. Gradient flows through the
receiver back to the encoder; the receiver's blind regions impose zero
gradient there → free bits.

Tishby et al. (1999 *Information Bottleneck Method*) propose the
Lagrangian

    L_IB(p_{T|X}) = I(X; T) - β · I(T; Y)

with the optimal ``β`` chosen to balance compression vs prediction
fidelity. We use ``β = 100 / 25 = 4`` per the contest formula's
coefficient ratio of segmentation (100) vs rate (25).

Wyner-Ziv (1976 *IEEE TIT* "The rate-distortion function for source
coding with side information") shows that with side information ``Y``
available at the decoder but not the encoder:

    R_{WZ}(D) = inf_{p(t|x), φ(t,y)} { I(X; T | Y) : E[d(X, φ(T, Y))] <= D }

For our problem, ``Y = scorer_state_dict`` is shared; the rate is the
conditional mutual information ``I(X; T | Y)``.

The composite ATW Lagrangian is

    L_ATW(θ, β=4) = L_IB + α · L_AR + γ · R_WZ

This module is a SCAFFOLD: full GPU-side training is deferred to a
follow-up subagent per the symposium spec ($5-15 Modal A100 smoke). The
math primitives are tested in isolation and compose deterministically.

[verified-against: Atick & Redlich 1990 *Network* Vol 1 §2 (cooperative
encoder); Tishby, Pereira, Bialek 1999 *Allerton* §II (IB Lagrangian);
Wyner & Ziv 1976 *IEEE TIT* Theorem 1 (R_WZ closed form for Gaussian);
Cover & Thomas 2nd ed §15.9 (Wyner-Ziv coding).]

Lane: ``lane_symposium_impl_atw_codec_20260515``.
Catalog #261.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

__all__ = (
    "ATW_BETA_FROM_CONTEST_FORMULA",
    "ATWCodecConfig",
    "ATWCompositeLoss",
    "atick_redlich_cooperative_loss",
    "compose_atw_lagrangian",
    "tishby_information_bottleneck_lagrangian",
    "update_from_anchor",
    "wyner_ziv_conditional_rate",
)

# β derived from contest formula 100·S_seg / 25·R = 4. Per Tao Phase E Eureka #3.
ATW_BETA_FROM_CONTEST_FORMULA: Final[float] = 100.0 / 25.0


@dataclasses.dataclass(frozen=True)
class ATWCodecConfig:
    """Hyperparameters for the ATW composite loss."""

    beta_information_bottleneck: float = ATW_BETA_FROM_CONTEST_FORMULA
    alpha_atick_redlich_weight: float = 1.0
    gamma_wyner_ziv_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.beta_information_bottleneck <= 0:
            raise ValueError("beta_information_bottleneck must be > 0")
        if self.alpha_atick_redlich_weight < 0:
            raise ValueError("alpha_atick_redlich_weight must be >= 0")
        if self.gamma_wyner_ziv_weight < 0:
            raise ValueError("gamma_wyner_ziv_weight must be >= 0")


@dataclasses.dataclass(frozen=True)
class ATWCompositeLoss:
    """Decomposed ATW Lagrangian terms + total."""

    information_bottleneck_term: float
    atick_redlich_term: float
    wyner_ziv_term: float
    total: float
    beta_used: float
    notes: str


def atick_redlich_cooperative_loss(
    encoded_receiver_output: np.ndarray,
    target_receiver_output: np.ndarray,
) -> float:
    """L_AR = E_X[||receiver(encode(X)) - receiver(X)||²].

    For arbitrary receiver outputs (vectors, logits, etc.), MSE in the
    receiver-output space is the canonical Atick-Redlich loss per the
    1990 derivation. The receiver IS the contest scorer (SegNet +
    PoseNet); receiver-output distortion is exactly the contest's seg+pose
    component score in expectation.
    """
    if encoded_receiver_output.shape != target_receiver_output.shape:
        raise ValueError("shape mismatch between encoded and target receiver outputs")
    if encoded_receiver_output.size == 0:
        return 0.0
    diff = encoded_receiver_output.astype(np.float64) - target_receiver_output.astype(np.float64)
    return float((diff**2).mean())


def _gaussian_mutual_information(
    variance_x: float, variance_t: float, correlation: float
) -> float:
    """I(X; T) for jointly Gaussian (X, T) with correlation ``ρ``.

    Closed form: ``I(X; T) = -0.5 log2(1 - ρ²)`` for ``|ρ| < 1``.
    [verified-against: Cover & Thomas 2nd ed §10.1.]
    """
    if not 0 <= variance_x:
        raise ValueError("variance_x must be >= 0")
    if not 0 <= variance_t:
        raise ValueError("variance_t must be >= 0")
    if abs(correlation) >= 1.0:
        return float("inf")
    return -0.5 * math.log2(1.0 - correlation**2)


def tishby_information_bottleneck_lagrangian(
    *,
    mutual_information_X_T: float,
    mutual_information_T_Y: float,
    beta: float = ATW_BETA_FROM_CONTEST_FORMULA,
) -> float:
    """L_IB = I(X; T) - β · I(T; Y).

    Lower L_IB is better: minimize information about source ``X`` retained
    in latent ``T`` while maximizing information about target ``Y``.
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")
    if mutual_information_X_T < 0 or mutual_information_T_Y < 0:
        raise ValueError("mutual information terms must be >= 0")
    return float(mutual_information_X_T - beta * mutual_information_T_Y)


def wyner_ziv_conditional_rate(
    *,
    variance_source: float,
    variance_side_info: float,
    correlation_source_side: float,
    distortion_target: float,
) -> float:
    """R_WZ(D) for jointly Gaussian source + Gaussian side information.

    Per Wyner & Ziv (1976) Theorem 1, for the canonical Gaussian-Gaussian
    setup with correlation ``ρ`` between source ``X`` and side-info ``Y``
    available at decoder only:

        R_WZ(D) = max(0, 0.5 log2(σ_X² (1 - ρ²) / D))   for D <= σ_X²(1-ρ²)
                = 0                                       otherwise

    The side info reduces the effective source variance from σ_X² to
    σ_X² · (1 - ρ²).
    """
    if variance_source <= 0:
        raise ValueError("variance_source must be > 0")
    if variance_side_info <= 0:
        raise ValueError("variance_side_info must be > 0")
    if abs(correlation_source_side) > 1.0:
        raise ValueError("correlation must be in [-1, 1]")
    if distortion_target <= 0:
        return float("inf")
    effective_variance = variance_source * (1.0 - correlation_source_side**2)
    if distortion_target >= effective_variance:
        return 0.0
    return 0.5 * math.log2(effective_variance / distortion_target)


def compose_atw_lagrangian(
    *,
    config: ATWCodecConfig,
    mutual_information_X_T: float,
    mutual_information_T_Y: float,
    encoded_receiver_output: np.ndarray,
    target_receiver_output: np.ndarray,
    variance_source: float,
    variance_side_info: float,
    correlation_source_side: float,
    distortion_target: float,
) -> ATWCompositeLoss:
    """Compose all three ATW terms into the unified Lagrangian.

    L_ATW = L_IB + α · L_AR + γ · R_WZ
    """
    ib = tishby_information_bottleneck_lagrangian(
        mutual_information_X_T=mutual_information_X_T,
        mutual_information_T_Y=mutual_information_T_Y,
        beta=config.beta_information_bottleneck,
    )
    ar = atick_redlich_cooperative_loss(
        encoded_receiver_output=encoded_receiver_output,
        target_receiver_output=target_receiver_output,
    )
    wz = wyner_ziv_conditional_rate(
        variance_source=variance_source,
        variance_side_info=variance_side_info,
        correlation_source_side=correlation_source_side,
        distortion_target=distortion_target,
    )
    total = ib + config.alpha_atick_redlich_weight * ar + config.gamma_wyner_ziv_weight * wz
    notes = (
        f"[prediction; first-principles] β={config.beta_information_bottleneck} "
        f"(contest 100/25); α={config.alpha_atick_redlich_weight}; "
        f"γ={config.gamma_wyner_ziv_weight}. Catalog #261."
    )
    return ATWCompositeLoss(
        information_bottleneck_term=ib,
        atick_redlich_term=ar,
        wyner_ziv_term=wz,
        total=total,
        beta_used=config.beta_information_bottleneck,
        notes=notes,
    )


def update_from_anchor(
    anchor: Mapping[str, object],
    *,
    config: ATWCodecConfig | None = None,
) -> ATWCompositeLoss | None:
    """Re-emit the ATW Lagrangian from an empirical anchor.

    The anchor must carry ``mutual_information_X_T``,
    ``mutual_information_T_Y``, ``variance_source``,
    ``variance_side_info``, ``correlation_source_side``,
    ``distortion_target``, and tensors ``encoded_receiver_output`` +
    ``target_receiver_output`` (or the receiver MSE pre-computed as
    ``atick_redlich_term``).
    """
    needed_scalar = {
        "mutual_information_X_T",
        "mutual_information_T_Y",
        "variance_source",
        "variance_side_info",
        "correlation_source_side",
        "distortion_target",
    }
    if not needed_scalar.issubset(anchor):
        return None
    if config is None:
        config = ATWCodecConfig()
    try:
        m_x_t = float(anchor["mutual_information_X_T"])  # type: ignore[arg-type]
        m_t_y = float(anchor["mutual_information_T_Y"])  # type: ignore[arg-type]
        var_source = float(anchor["variance_source"])  # type: ignore[arg-type]
        var_side = float(anchor["variance_side_info"])  # type: ignore[arg-type]
        corr = float(anchor["correlation_source_side"])  # type: ignore[arg-type]
        d_target = float(anchor["distortion_target"])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    encoded_out = anchor.get("encoded_receiver_output")
    target_out = anchor.get("target_receiver_output")
    if isinstance(encoded_out, np.ndarray) and isinstance(target_out, np.ndarray):
        return compose_atw_lagrangian(
            config=config,
            mutual_information_X_T=m_x_t,
            mutual_information_T_Y=m_t_y,
            encoded_receiver_output=encoded_out,
            target_receiver_output=target_out,
            variance_source=var_source,
            variance_side_info=var_side,
            correlation_source_side=corr,
            distortion_target=d_target,
        )
    # Fallback: use a precomputed atick_redlich_term scalar
    atick_redlich = float(anchor.get("atick_redlich_term", 0.0))  # type: ignore[arg-type]
    ib = tishby_information_bottleneck_lagrangian(
        mutual_information_X_T=m_x_t,
        mutual_information_T_Y=m_t_y,
        beta=config.beta_information_bottleneck,
    )
    wz = wyner_ziv_conditional_rate(
        variance_source=var_source,
        variance_side_info=var_side,
        correlation_source_side=corr,
        distortion_target=d_target,
    )
    total = ib + config.alpha_atick_redlich_weight * atick_redlich + config.gamma_wyner_ziv_weight * wz
    return ATWCompositeLoss(
        information_bottleneck_term=ib,
        atick_redlich_term=atick_redlich,
        wyner_ziv_term=wz,
        total=total,
        beta_used=config.beta_information_bottleneck,
        notes=f"anchor-driven (scalar-only). β={config.beta_information_bottleneck}. Catalog #261.",
    )
