"""Score-aware Lagrangian for the SAR coherent pose-pair substrate.

L = alpha * B / N + beta * d_seg + gamma * sqrt(d_pose) + delta * H_temporal

where:

* ``B`` is the archive byte count (rate term)
* ``N`` is the contest normalizer (37,545,489 frames-equivalent)
* ``d_seg``, ``d_pose`` are the canonical scorer distortions through the
  ``score_pair_components`` pipeline (Catalog #164)
* ``H_temporal`` is the SAR-coherent auxiliary term: an L2 penalty on the
  HIGH-frequency tail of the pose-delta rFFT spectrum. This nudges the
  pose-trajectory toward temporally-smooth (low-frequency-dominant) form,
  which is the precondition for the L2 ledger §2.4 falsification — sparse
  rFFT codecs only outperform raw int8 when the spectrum concentrates in
  the low-frequency bins.

The Atick-Redlich cooperative-receiver hook: ``score_pair_components``
already runs the eval-roundtrip-corrected RGB through the FIXED contest
scorer (SegNet + PoseNet). This is mathematically equivalent to maximizing
``MI(B; S(B))`` subject to the rate constraint.

Per CLAUDE.md "HNeRV parity discipline" lesson L6: score-domain Lagrangian
(NOT weight-domain proxies like rel_err²).
Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE": ``apply_eval_roundtrip=True``
is the only acceptable training mode.
Per CLAUDE.md "EMA — non-negotiable": the trainer applies EMA externally.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components,
)


def _validate_rgb_255_domain(name: str, tensor: torch.Tensor) -> None:
    """Fail closed when callers pass nonzero unit-domain RGB by mistake."""

    detached = tensor.detach()
    if not torch.isfinite(detached).all():
        raise ValueError(f"{name} must contain finite RGB values")
    min_value = float(detached.min().item())
    max_value = float(detached.max().item())
    if min_value < 0.0 or max_value > 255.0:
        raise ValueError(
            f"{name} must be in [0, 255]; got min={min_value} max={max_value}"
        )
    if max_value > 0.0 and max_value <= 1.0:
        raise ValueError(
            f"{name} appears to be unit-domain RGB; expected [0, 255] scorer input"
        )


@dataclass(frozen=True)
class SARCoherentLossWeights:
    """Score-domain Lagrangian weights.

    Defaults match the contest formula. ``delta_temporal`` controls the
    SAR-coherent auxiliary term; default 0.05 keeps it small relative to seg
    + pose so the substrate still primarily learns to satisfy the scorer.
    """

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    """Operating-point tilt — 1.0 = contest formula. PR106 r2 has 2.71x
    pose marginal, but that is an experiment knob not a default."""
    delta_temporal: float = 0.05
    """Weight on the SAR-coherent temporal-smoothness auxiliary term."""
    contest_normalizer: float = 37_545_489.0
    sar_low_freq_fraction: float = 0.10
    """Fraction of low-frequency rFFT bins considered "in-budget" (matches
    ``cfg.sar_topk_keep_fraction``); the high-freq tail outside this is
    L2-penalized."""


class SARCoherentScoreAwareLoss(torch.nn.Module):
    """Score-aware Lagrangian as a torch Module.

    The renderer + per-pair pose deltas + per-pair RGB residual all carry
    gradient. Quantization (int16 sparse rFFT, int8 RGB residual) is a
    STE-friendly archive-build-time operation; gradient flows through the
    float pre-quant tensors.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: SARCoherentLossWeights,
    ) -> None:
        super().__init__()
        self.seg_scorer = seg_scorer
        self.pose_scorer = pose_scorer
        self.weights = weights

    def forward(
        self,
        rgb_0: torch.Tensor,
        rgb_1: torch.Tensor,
        gt_rgb_0: torch.Tensor,
        gt_rgb_1: torch.Tensor,
        archive_bytes_proxy: torch.Tensor,
        *,
        pose_deltas_dense: torch.Tensor | None = None,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Lagrangian.

        Args:
            rgb_0, rgb_1: predicted RGB pair tensors (B, 3, H, W) in [0, 255].
            gt_rgb_0, gt_rgb_1: ground-truth RGB pair tensors (B, 3, H, W) in [0, 255].
            archive_bytes_proxy: scalar tensor with the archive byte count
                (carries no gradient — rate term is non-differentiable).
            pose_deltas_dense: optional ``(num_pairs, pose_dim)`` tensor with
                the dense pose-delta trajectory. When provided, an L2 penalty
                is applied to the high-frequency rFFT tail (encouraging
                temporal smoothness — the precondition for SAR-coherent
                compression per L2 ledger §2.4).
            apply_eval_roundtrip: must be True per CLAUDE.md eval_roundtrip
                non-negotiable; ValueError on False.
            noise_std: per-pixel additive noise during training.

        Returns:
            ``(loss, parts_dict)`` — loss is scalar tensor with gradient;
            parts_dict has detached tensor values for logging.
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0; got {noise_std}")
        for name, tensor in (
            ("rgb_0", rgb_0),
            ("rgb_1", rgb_1),
            ("gt_rgb_0", gt_rgb_0),
            ("gt_rgb_1", gt_rgb_1),
        ):
            _validate_rgb_255_domain(name, tensor)

        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        if self.training and noise_std > 0.0:
            rgb_0 = rgb_0 + (torch.rand_like(rgb_0) - 0.5) * (2.0 * noise_std)
            rgb_1 = rgb_1 + (torch.rand_like(rgb_1) - 0.5) * (2.0 * noise_std)

        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)

        seg_term, pose_term = score_pair_components(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            rgb_0_rt=rgb_0_rt,
            rgb_1_rt=rgb_1_rt,
            gt_rgb_0=gt_rgb_0,
            gt_rgb_1=gt_rgb_1,
        )

        rate_term = (
            self.weights.alpha_rate
            * archive_bytes_proxy
            / self.weights.contest_normalizer
        )
        pose_sqrt = torch.sqrt(pose_term.clamp(min=1e-12))

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose * self.weights.pose_weight_scale * pose_sqrt
        )

        temporal_term_value: torch.Tensor | float = 0.0
        if pose_deltas_dense is not None and self.weights.delta_temporal > 0.0:
            # SAR-coherent temporal-smoothness penalty: L2 on the
            # high-frequency rFFT tail. Gradient nudges pose trajectory
            # toward low-frequency-dominant (temporally-smooth) form.
            n_pairs = pose_deltas_dense.shape[0]
            rfft_full = torch.fft.rfft(pose_deltas_dense, dim=0)
            n_bins = rfft_full.shape[0]
            n_keep = max(1, int(round(self.weights.sar_low_freq_fraction * n_bins)))
            high_freq_tail = rfft_full[n_keep:]
            temporal_term = high_freq_tail.abs().pow(2).mean()
            loss = loss + self.weights.delta_temporal * temporal_term
            temporal_term_value = temporal_term.detach()

        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "pose_sqrt": pose_sqrt.detach(),
            "temporal_term": (
                temporal_term_value
                if isinstance(temporal_term_value, torch.Tensor)
                else torch.tensor(float(temporal_term_value))
            ),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = [
    "SARCoherentLossWeights",
    "SARCoherentScoreAwareLoss",
]
