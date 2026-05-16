# SPDX-License-Identifier: MIT
"""Score-aware loss facade for the Tishby IB-pure L1 scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TishbyIBPureLossWeights:
    """Weights for the research-only IB decomposition."""

    reconstruction: float = 1.0
    beta: float = 0.01
    rate: float = 1.0

    def __post_init__(self) -> None:
        if self.reconstruction < 0:
            raise ValueError("reconstruction weight must be non-negative")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.rate < 0:
            raise ValueError("rate weight must be non-negative")


@dataclass(frozen=True)
class TishbyIBPureLossOutput:
    """Scalar decomposition emitted by the L1 loss facade."""

    total: float
    reconstruction_term: float
    kl_term: float
    rate_term: float
    score_claim: bool = False


class TishbyIBPureScoreAwareLoss:
    """Deterministic scalar IB objective shell.

    The full scorer-routed loss is intentionally not implemented at L1. This
    callable is enough for probes and registry checks to distinguish the IB
    decomposition without loading contest scorers or claiming eval authority.
    """

    def __init__(self, weights: TishbyIBPureLossWeights | None = None) -> None:
        self.weights = weights or TishbyIBPureLossWeights()

    def __call__(
        self,
        *,
        reconstruction_term: float,
        kl_term: float,
        rate_term: float = 0.0,
    ) -> TishbyIBPureLossOutput:
        total = (
            self.weights.reconstruction * reconstruction_term
            + self.weights.beta * kl_term
            + self.weights.rate * rate_term
        )
        return TishbyIBPureLossOutput(
            total=float(total),
            reconstruction_term=float(reconstruction_term),
            kl_term=float(kl_term),
            rate_term=float(rate_term),
            score_claim=False,
        )
