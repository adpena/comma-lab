# SPDX-License-Identifier: MIT
"""ATW codec V1 score-aware Lagrangian — Atick-Redlich + Tishby IB + Wyner-Ziv composition.

Per the 2026-05-15 grand reunion symposium Composite #1 (lines 727-770) and
the design memo at ``.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md``:

The ATW Lagrangian:

::

    L_ATW = α · B(θ)/N                          (rate from archive bytes)
          + β_seg · d_seg(θ)                    (Atick-Redlich SegNet term)
          + γ_pose · sqrt(d_pose(θ))            (Atick-Redlich PoseNet term)
          + κ_IB · I(T; Y_predicted)            (Tishby IB info-preservation)
          + λ_WZ · R_WZ_residual(t | t̂(s))      (Wyner-Ziv side-info residual)
          + λ_pixel · MSE(decoded, GT)          (Z3 pixel-MSE residual)

Defaults (κ_IB=0, λ_WZ=1, λ_pixel=0) recover the CLEAN ATW canonical form.
Knob-zero ablations recover the four corner regimes:

* (0, 0, 0)   → Atick-Redlich pure (= Z4 verbatim)
* (0, 1, 0)   → ATW canonical
* (0.1, 0, 0) → Tishby IB pure
* (0, 0, 1)   → Z3 baseline (pixel-MSE-only)

The IB term ``I(T; Y_predicted)`` is a SECOND-ORDER approximation:
``-0.5 * E[(z_residual)^T · z_residual]`` where ``z_residual`` is the
Wyner-Ziv residual. This is a tractable proxy that captures the bit cost
of representing the latent above the side-info prediction (the Tishby IB
``I(T;Y)`` is reduced by the bits the side-info head already encoded
implicitly via ``z_predicted``).

The WZ term ``R_WZ_residual`` is the L2 norm of ``z_residual``: it
PENALIZES residuals that the WZ head failed to predict. The encoder is
incentivized to drive ``z`` close to ``z_predicted`` (i.e. close to the
manifold the side-info head can recover), which is exactly the
Wyner-Ziv "minimize what you transmit beyond what the decoder predicts"
intuition.

Per CLAUDE.md "HNeRV parity discipline" lesson L6: score-domain Lagrangian.
Per CLAUDE.md "eval_roundtrip — non-negotiable": ``apply_eval_roundtrip=True``
is the only acceptable training mode.
Per CLAUDE.md "EMA — non-negotiable": the trainer applies EMA externally.
Per Catalog #164: the loss MUST route through ``score_pair_components``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components_dispatch,
)

_RGB_255_DOMAIN_EPS = 1e-3


def _validate_rgb_255_domain(name: str, tensor: torch.Tensor) -> None:
    """Fail closed when callers pass nonzero unit-domain RGB by mistake."""
    detached = tensor.detach()
    if not torch.isfinite(detached).all():
        raise ValueError(f"{name} must contain finite RGB values")
    min_value = float(detached.min().item())
    max_value = float(detached.max().item())
    if min_value < -_RGB_255_DOMAIN_EPS or max_value > 255.0 + _RGB_255_DOMAIN_EPS:
        raise ValueError(
            f"{name} must be in [0, 255]; got min={min_value} max={max_value}"
        )
    if max_value > 0.0 and max_value <= 1.0:
        raise ValueError(
            f"{name} appears to be unit-domain RGB; expected [0, 255] scorer input"
        )


@dataclass(frozen=True)
class ATWLossWeights:
    """ATW codec score-domain Lagrangian weights.

    Defaults match the contest formula for (α, β_seg, γ_pose) and the ATW
    canonical mode for (κ_IB=0, λ_WZ=1, λ_pixel=0). Setting all three
    knobs to specific corner values recovers the four ablation regimes.
    """

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    """Operating-point tilt — 1.0 = contest formula."""
    kappa_ib: float = 0.0
    """Tishby IB regularizer weight; 0 = no IB; 0.05-0.1 = IB regime."""
    lambda_wz: float = 1.0
    """Wyner-Ziv residual term weight; 0 = no WZ; 1 = ATW canonical."""
    lambda_pixel: float = 0.0
    """Pixel-MSE residual weight; 0 = pure ATW; 1 = Z3 baseline."""
    contest_normalizer: float = 37_545_489.0


class ATWScoreAwareLoss(torch.nn.Module):
    """ATW codec score-aware Lagrangian as a torch Module.

    Routes scorer calls through ``score_pair_components_dispatch`` per
    Catalog #164 (sister-substrate canonical pattern).

    The encoder + decoder + per-pair latent + WZ side-info head ALL carry
    gradients flowing through:

    1. The canonical scorer distortion terms (Atick-Redlich form).
    2. The IB regularizer on ``z_residual`` (when κ_IB > 0).
    3. The WZ residual term penalizing ``||z_residual||^2`` (when λ_WZ > 0).
    4. The pixel-MSE residual (when λ_pixel > 0; Z3 baseline mode).
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: ATWLossWeights,
    ) -> None:
        super().__init__()
        self.seg_scorer = seg_scorer
        self.pose_scorer = pose_scorer
        self.weights = weights

    def forward(
        self,
        *,
        reconstructed_rgb_0: torch.Tensor,
        reconstructed_rgb_1: torch.Tensor,
        gt_rgb_0: torch.Tensor,
        gt_rgb_1: torch.Tensor,
        archive_bytes_proxy: torch.Tensor,
        z_residual: torch.Tensor | None = None,
        z_predicted: torch.Tensor | None = None,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
        gt_pose_batch: torch.Tensor | None = None,
        gt_seg_batch: torch.Tensor | None = None,
        gt_seg_already_probs: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the ATW score-domain Lagrangian.

        Args:
            reconstructed_rgb_0: predicted frame_0 (B, 3, H, W) in [0, 255].
            reconstructed_rgb_1: predicted frame_1 (B, 3, H, W) in [0, 255].
            gt_rgb_0, gt_rgb_1: ground-truth RGB pair tensors in [0, 255].
            archive_bytes_proxy: scalar tensor with archive byte count.
            z_residual: ``(B, latent_dim)`` Wyner-Ziv residual (z - z_predicted).
                Required when ``λ_WZ > 0`` OR ``κ_IB > 0``; ignored when
                both are 0 (Atick-Redlich pure or Z3 baseline modes).
            z_predicted: ``(B, latent_dim)`` Wyner-Ziv prediction. Used only
                for diagnostic logging; not for loss assembly (the loss
                composes ``z_residual`` not ``z_predicted`` directly).
            apply_eval_roundtrip: must be True; ValueError on False.
            noise_std: per-pixel additive noise during training.

        Returns:
            ``(loss, parts_dict)`` with parts including rate_term, seg_term,
            pose_term, pose_sqrt, ib_term, wz_term, pixel_term, loss_total.
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0; got {noise_std}")
        if self.weights.lambda_pixel < 0.0:
            raise ValueError(
                f"lambda_pixel must be >= 0; got {self.weights.lambda_pixel}"
            )
        if self.weights.lambda_wz < 0.0:
            raise ValueError(
                f"lambda_wz must be >= 0; got {self.weights.lambda_wz}"
            )
        if self.weights.kappa_ib < 0.0:
            raise ValueError(
                f"kappa_ib must be >= 0; got {self.weights.kappa_ib}"
            )

        wz_or_ib_active = (
            self.weights.lambda_wz > 0.0 or self.weights.kappa_ib > 0.0
        )
        if wz_or_ib_active and z_residual is None:
            raise ValueError(
                "z_residual is required when lambda_wz > 0 or kappa_ib > 0; "
                "got z_residual=None"
            )

        for name, tensor in (
            ("reconstructed_rgb_0", reconstructed_rgb_0),
            ("reconstructed_rgb_1", reconstructed_rgb_1),
            ("gt_rgb_0", gt_rgb_0),
            ("gt_rgb_1", gt_rgb_1),
        ):
            _validate_rgb_255_domain(name, tensor)

        # Lazy import per canonical pattern (matches Z4 + sister substrates).
        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        rgb_0 = reconstructed_rgb_0
        rgb_1 = reconstructed_rgb_1
        if self.training and noise_std > 0.0:
            rgb_0 = rgb_0 + (torch.rand_like(rgb_0) - 0.5) * (2.0 * noise_std)
            rgb_1 = rgb_1 + (torch.rand_like(rgb_1) - 0.5) * (2.0 * noise_std)

        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)

        # Canonical scorer-preprocess routing per Catalog #164.
        seg_term, pose_term = score_pair_components_dispatch(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            rgb_0_rt=rgb_0_rt,
            rgb_1_rt=rgb_1_rt,
            gt_rgb_0=gt_rgb_0,
            gt_rgb_1=gt_rgb_1,
            gt_pose_batch=gt_pose_batch,
            gt_seg_batch=gt_seg_batch,
            gt_seg_already_probs=gt_seg_already_probs,
        )

        rate_term = (
            self.weights.alpha_rate
            * archive_bytes_proxy
            / self.weights.contest_normalizer
        )
        pose_sqrt = torch.sqrt(pose_term.clamp(min=1e-12))

        device_for_zero = rgb_0_rt.device
        dtype_for_zero = rgb_0_rt.dtype

        # Tishby IB regularizer (second-order proxy on z_residual energy).
        # I(T; Y) is approximated by ``E[||z_residual||^2 / 2]`` — a tractable
        # bound that penalizes the residual carrying side-info-redundant info.
        if self.weights.kappa_ib > 0.0 and z_residual is not None:
            ib_proxy = 0.5 * z_residual.pow(2).mean()
            ib_term = self.weights.kappa_ib * ib_proxy
        else:
            ib_term = torch.zeros((), device=device_for_zero, dtype=dtype_for_zero)

        # Wyner-Ziv residual term: penalize ||z_residual||^2.
        # When λ_WZ > 0, the encoder is incentivized to drive z close to
        # z_predicted (the side-info-head's prediction), which IS the WZ
        # "minimize what you transmit beyond what the decoder predicts" intuition.
        if self.weights.lambda_wz > 0.0 and z_residual is not None:
            wz_residual_energy = z_residual.pow(2).mean()
            wz_term = self.weights.lambda_wz * wz_residual_energy
        else:
            wz_term = torch.zeros((), device=device_for_zero, dtype=dtype_for_zero)

        # Optional pixel-MSE residual (Z3 baseline mode).
        if self.weights.lambda_pixel > 0.0:
            pixel_mse_0 = (rgb_0_rt - gt_rgb_0).pow(2).mean()
            pixel_mse_1 = (rgb_1_rt - gt_rgb_1).pow(2).mean()
            pixel_mse = 0.5 * (pixel_mse_0 + pixel_mse_1)
            pixel_term = self.weights.lambda_pixel * pixel_mse
        else:
            pixel_term = torch.zeros((), device=device_for_zero, dtype=dtype_for_zero)

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose * self.weights.pose_weight_scale * pose_sqrt
            + ib_term
            + wz_term
            + pixel_term
        )

        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "pose_sqrt": pose_sqrt.detach(),
            "ib_term": ib_term.detach(),
            "wz_term": wz_term.detach(),
            "pixel_term": pixel_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = ["ATWLossWeights", "ATWScoreAwareLoss"]
