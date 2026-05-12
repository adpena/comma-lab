"""wavelet score-aware Lagrangian — L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose).

Per CLAUDE.md HNeRV parity discipline L6 + L8: the score-domain Lagrangian must
backprop through SegNet/PoseNet via the differentiable eval-roundtrip + the
patched yuv6 (PR #95/#106 monkey-patch contract).

For wavelet, the rate proxy can use a per-subband bit-count proxy:
    bits[k] = entropy_estimate(subband_k_coefficients_quantized_int16)
where ``entropy_estimate`` is the Shannon entropy of the empirical
distribution (closed-form bound). The substrate-specific knob is
``subband_rate_weights`` letting the trainer weight LL/LH/HL/HH differently
when allocating bits per Mallat-hierarchy logic.

CLAUDE.md compliance:
- eval_roundtrip=True (NON-NEGOTIABLE)
- noise_std reserved for forward-compat
- No silent CUDA fallback (caller supplies device + scorers)
- No /tmp paths
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ScoreAwareLossWeights:
    """The (alpha, beta, gamma) of the score-domain Lagrangian for wavelet."""

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = 1.0
    pose_weight_scale: float = 1.0
    contest_normalizer: float = 37_545_489.0

    subband_rate_weights: tuple[float, float, float, float] = field(
        default_factory=lambda: (1.0, 1.0, 1.0, 1.0)
    )
    """(w_LL, w_LH, w_HL, w_HH) — Mallat-hierarchy rate weights.

    Default uniform; trainer can downweight LH/HL/HH to encourage coarser
    quantization of detail subbands when score-aware loss reveals that
    high-frequency detail doesn't matter for PoseNet/SegNet.
    """


class WaveletScoreAwareLoss(torch.nn.Module):
    """Lagrangian as a torch Module for wavelet substrate.

    Mirror of sane_hnerv's SaneHnervScoreAwareLoss with a per-subband rate
    proxy split off so the trainer can apply Mallat-hierarchy-aware weights.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: ScoreAwareLossWeights,
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
        subband_bit_proxies: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Lagrangian.

        Args:
            rgb_0, rgb_1: rendered frame pair, ``(B, 3, H, W)``, in [0, 1].
            gt_rgb_0, gt_rgb_1: ground-truth frame pair.
            archive_bytes_proxy: scalar tensor — closed-form upper bound on
                archive byte count.
            subband_bit_proxies: ``(LL, LH, HL, HH)`` scalar tensors of estimated
                bit count for each subband (e.g., empirical entropy * num_elements).

        Returns:
            ``(loss_total, parts)`` with named components.
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )

        from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training

        del noise_std  # reserved for forward-compat
        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)

        # SegNet on rgb_1 per upstream contract
        seg_out = self.seg_scorer(rgb_1_rt.unsqueeze(1))
        seg_gt = self.seg_scorer(gt_rgb_1.unsqueeze(1))
        seg_term = _seg_distortion_proxy(seg_out, seg_gt)

        # PoseNet on (rgb_0, rgb_1) pair
        pose_in = torch.cat([rgb_0_rt, rgb_1_rt], dim=1)
        pose_gt = torch.cat([gt_rgb_0, gt_rgb_1], dim=1)
        pose_out = self.pose_scorer(pose_in)
        pose_target = self.pose_scorer(pose_gt)
        pose_term = ((pose_out[:, :6] - pose_target[:, :6]) ** 2).mean()

        # Base archive-bytes proxy
        rate_archive = (
            self.weights.alpha_rate * archive_bytes_proxy / self.weights.contest_normalizer
        )

        # Per-subband rate proxy (Mallat-hierarchy aware)
        w_ll, w_lh, w_hl, w_hh = self.weights.subband_rate_weights
        ll_p, lh_p, hl_p, hh_p = subband_bit_proxies
        rate_subband = (
            (w_ll * ll_p + w_lh * lh_p + w_hl * hl_p + w_hh * hh_p)
            / self.weights.contest_normalizer
        )
        rate_term = rate_archive + rate_subband

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
        )

        parts = {
            "rate_term": rate_term.detach(),
            "rate_archive": rate_archive.detach(),
            "rate_subband": rate_subband.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


def _seg_distortion_proxy(
    seg_logits_pred: torch.Tensor, seg_logits_gt: torch.Tensor
) -> torch.Tensor:
    """Soft cross-entropy between predicted and gt seg logits."""
    log_p = torch.log_softmax(seg_logits_pred, dim=1)
    q = torch.softmax(seg_logits_gt, dim=1)
    return -(q * log_p).sum(dim=1).mean()
