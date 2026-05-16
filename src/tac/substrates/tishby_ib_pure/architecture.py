# SPDX-License-Identifier: MIT
"""Research-only Tishby IB-pure codec scaffold.

This module intentionally stays small at L1: it gives the candidate a real,
importable architecture surface without pretending to be a trained contest
codec. Promotion still requires the real scorer/probe gates documented in the
design memo.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

DEFAULT_BETA = 0.01
DEFAULT_LATENT_DIM = 16
EVAL_HW = (874, 1164)
NUM_PAIRS = 600
NUM_SEGNET_CLASSES = 19


class TishbyIBPurePathVariant(StrEnum):
    """Operationalization path for the IB objective."""

    VIB = "vib"
    MINE = "mine"


@dataclass(frozen=True)
class TishbyIBPureCodecConfig:
    """Configuration for the research-only Tishby IB-pure codec shell."""

    input_dim: int = 64
    latent_dim: int = DEFAULT_LATENT_DIM
    output_dim: int = 5
    beta: float = DEFAULT_BETA
    path_variant: TishbyIBPurePathVariant = TishbyIBPurePathVariant.VIB

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")


class TishbyIBPureCodec:
    """Minimal deterministic codec facade for L1 import/archive tests.

    The full trainable implementation belongs behind the Phase-2 lift. This
    facade is deliberately non-neural so imports, archive plumbing, and
    provenance tools can reason about the candidate without claiming a score.
    """

    def __init__(self, config: TishbyIBPureCodecConfig | None = None) -> None:
        self.config = config or TishbyIBPureCodecConfig()

    def encode_summary(self) -> Mapping[str, Any]:
        """Return a deterministic summary suitable for provenance JSON."""

        return {
            "beta": self.config.beta,
            "input_dim": self.config.input_dim,
            "latent_dim": self.config.latent_dim,
            "output_dim": self.config.output_dim,
            "path_variant": self.config.path_variant.value,
            "research_only": True,
            "score_claim": False,
        }
