# SPDX-License-Identifier: MIT
"""IBMDLLoss — Information Bottleneck variational upper bound (Tishby-Zaslavsky).

Per the C6 across-class shift hypothesis: the substrate minimizes a joint
MDL × IB objective.

Information-theoretic contract:

    Mutual information I(z; frames) is upper-bounded by the variational
    KL(q(z|frames) || p(z)) (Tishby-Zaslavsky 2017; Alemi et al. 2017 VIB).

For p(z) = N(0, I) and q(z|frames) = N(μ, σ²):

    KL(N(μ, σ²) || N(0, 1)) = 0.5 * sum(μ² + σ² - log(σ²) - 1)
                            = 0.5 * sum(μ² + exp(logvar) - logvar - 1)

The MDL-IBPS loss is the IB regularizer:

    L_IB = β · KL(q(z|frames) || N(0, I))

The trainer composes this with the score-aware loss (in score_aware_loss.py):

    L_total = L_score + L_IB
            = L_rate + L_seg + L_pose + β · KL

where β is the IB Lagrangian (operator-tunable; typical range 0.001 - 1.0).

Per CLAUDE.md FORBIDDEN_PATTERNS:
- NO scorer load in this module.
- NO /tmp paths.
- NO MPS-falsified strategic decisions (this is a TRAINING SIGNAL; auth eval
  is mandatory before any score claim).

References:
- Tishby & Zaslavsky 2015 "Deep Learning and the Information Bottleneck Principle"
- Alemi et al. 2017 "Deep Variational Information Bottleneck" (ICLR)
- Kingma & Welling 2013 "Auto-Encoding Variational Bayes" (ICLR; ELBO derivation)
"""

from __future__ import annotations

import torch


def kl_gaussian_to_standard_normal(
    mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Compute KL(N(μ, σ²) || N(0, I)) per-sample.

    Formula: 0.5 * sum_dims(μ² + exp(logvar) - logvar - 1)

    Args:
        mu: posterior mean ``(B, d_z)``.
        logvar: posterior log-variance ``(B, d_z)``.

    Returns:
        kl per sample ``(B,)`` in nats.
    """
    if mu.shape != logvar.shape:
        raise ValueError(
            f"mu and logvar must have matching shape; got "
            f"{tuple(mu.shape)} vs {tuple(logvar.shape)}"
        )
    if mu.dim() != 2:
        raise ValueError(f"mu must be 2D (B, d_z); got {tuple(mu.shape)}")
    return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=-1)


class IBMDLLoss(torch.nn.Module):
    """Variational IB loss with operator-tunable β Lagrangian.

    The loss returns the AVERAGE per-sample KL multiplied by β.

    Args:
        beta: IB Lagrangian (default 0.01). Higher β → tighter bottleneck.
        prior: prior distribution; only "standard_normal" supported in v1.
    """

    def __init__(
        self,
        beta: float = 0.01,
        prior: str = "standard_normal",
    ) -> None:
        super().__init__()
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0; got {beta}")
        if prior != "standard_normal":
            raise NotImplementedError(
                f"only standard_normal prior is supported in v1; got {prior}"
            )
        self.beta = float(beta)
        self.prior = prior

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute β · mean_batch(KL(q(z|frames) || p(z))).

        Args:
            mu: posterior mean ``(B, d_z)``.
            logvar: posterior log-variance ``(B, d_z)``.

        Returns:
            (loss_term, parts_dict) where:
            - loss_term: scalar β-weighted KL (in nats).
            - parts_dict: {"kl_mean", "kl_max", "kl_per_dim_mean"}.
        """
        kl_per_sample = kl_gaussian_to_standard_normal(mu, logvar)
        kl_mean = kl_per_sample.mean()
        loss = self.beta * kl_mean
        parts = {
            "kl_mean": kl_mean.detach(),
            "kl_max": kl_per_sample.max().detach(),
            "kl_per_dim_mean": kl_per_sample.mean().detach() / float(mu.shape[1]),
            "loss_ib": loss.detach(),
        }
        return loss, parts


__all__ = ["IBMDLLoss", "kl_gaussian_to_standard_normal"]
