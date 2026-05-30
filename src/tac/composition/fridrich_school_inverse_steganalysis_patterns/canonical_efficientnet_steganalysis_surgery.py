# SPDX-License-Identifier: MIT
"""Canonical EfficientNet steganalysis surgery (Yousfi-DDELab deepsteganalysis).

Origin: ``github.com/DDELab/deepsteganalysis`` (DDE Lab Binghamton; last
updated 2025-05-01; PyTorch Lightning). Per CLAUDE.md "Quantizr intelligence"
+ "Exact scorer architectures": *"Yousfi (challenge creator) was Fridrich's
PhD student at Binghamton DDE Lab. EfficientNet steganalysis surgery ->
informed SegNet scorer design."*

The CANONICAL insight (Yousfi-Fridrich-DDELab pipeline):
EfficientNet-B0 through B7 work well for general image classification but
struggle on STEGANALYSIS because the **stride-2 stem** loses half resolution
immediately, destroying the embedding signal which lives in high-frequency
content. The DDELab surgery pattern keeps EfficientNet's backbone but
SURGICALLY modifies the stem:

1. **Stem stride 2 -> stride 1** (preserves spatial resolution).
2. **Replace stem batch-norm with truncated-linear-unit** (preserves
   embedding signal which has small magnitude relative to cover content).
3. **First-block channel doubling** (compensates for the stride change which
   reduces receptive field by 2x).

For the comma-video contest, the canonical SegNet is
``smp.Unet('tu-efficientnet_b2')`` WITHOUT the surgery. Per CLAUDE.md "Exact
scorer architectures": *"vanilla stride-2 stem (no Yousfi surgery)... Key
blind spot: stride-2 stem loses half resolution immediately -> artifacts
below (256,192) invisible."*

This is the **CANONICAL ATTACK VECTOR**: the comma SegNet inherits Yousfi's
EfficientNet-B2 backbone but NOT his surgery. Therefore signal below
(256, 192) is structurally invisible to SegNet. Our pose-axis attack
cascade can exploit this blind spot. Yousfi-Tier-1 pose-axis canonical
helpers (sister landing in flight) operationalize this insight.

5-axis adaptation taxonomy
--------------------------
* **Axis A (contest)** -- detector surgery -> ATTACK VECTOR taxonomy
  (we don't surgically modify our scorer; we surgically modify our perturbation
  pattern to live in scorer blind spot)
* **Axis B (problem space)** -- 5-class softmax -> per-pair score components
* **Axis C (math)** -- stride preservation 1:1 with DDELab surgery insight
* **Axis D (data)** -- 256x256 JPEG QF95 -> 1164x874 contest frames
* **Axis E (video)** -- single-image -> per-pair shared latent

Sister landings
---------------
* CLAUDE.md "Exact scorer architectures" SegNet section (canonical contest
  scorer ``smp.Unet('tu-efficientnet_b2', classes=5, encoder_weights=None)``
  vanilla stride-2 stem with KNOWN blind spot).
* ``tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_comma10k_baseline_lineage``
  (direct ancestry of contest SegNet from Yousfi's comma10k-baseline).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

__all__ = (
    "EfficientNetStemSurgeryStrategy",
    "EfficientNetSurgeryConfig",
    "EfficientNetSurgeryError",
    "compute_blind_spot_resolution_from_stride",
    "CONTEST_SEGNET_STRIDE_2_BLIND_SPOT",
)


class EfficientNetSurgeryError(ValueError):
    """Raised when EfficientNet surgery config violates a canonical invariant."""


class EfficientNetStemSurgeryStrategy(str, Enum):
    """Canonical EfficientNet stem surgery strategies (DDELab Yousfi-Fridrich).

    * ``DDELAB_CANONICAL_STRIDE_1`` -- DDELab canonical surgery: stride-2 ->
      stride-1 + truncated-linear-unit + channel doubling (Yousfi PhD thesis
      canonical pattern).
    * ``VANILLA_STRIDE_2_BASELINE`` -- vanilla EfficientNet stride-2 stem
      WITHOUT surgery; the canonical contest SegNet config; has KNOWN
      blind spot below (256, 192) resolution.
    * ``HYBRID_STRIDE_1_NO_TLU`` -- ablation variant: stride-1 but keep
      batch-norm; tests whether stride or BN is the dominant signal-killer.
    * ``HYBRID_STRIDE_2_WITH_TLU`` -- ablation variant: keep stride-2 but
      add truncated-linear-unit; tests whether TLU alone recovers signal.
    """

    DDELAB_CANONICAL_STRIDE_1 = "ddelab_canonical_stride_1"
    VANILLA_STRIDE_2_BASELINE = "vanilla_stride_2_baseline"
    HYBRID_STRIDE_1_NO_TLU = "hybrid_stride_1_no_tlu"
    HYBRID_STRIDE_2_WITH_TLU = "hybrid_stride_2_with_tlu"


@dataclass(frozen=True)
class EfficientNetSurgeryConfig:
    """Canonical EfficientNet surgery configuration.

    Attributes
    ----------
    strategy
        Which canonical surgery strategy.
    input_resolution
        ``(height, width)`` of the input frame fed to EfficientNet.
        Default ``(384, 512)`` matches CLAUDE.md SegNet input contract
        (``bilinear resize to (512, 384)``).
    target_blind_spot_max
        Per CLAUDE.md "Exact scorer architectures": ``(256, 192)`` is the
        canonical blind-spot resolution below which the contest SegNet
        cannot recover signal. Default matches this canonical value.
    """

    strategy: EfficientNetStemSurgeryStrategy = (
        EfficientNetStemSurgeryStrategy.VANILLA_STRIDE_2_BASELINE
    )
    input_resolution: Tuple[int, int] = (384, 512)
    target_blind_spot_max: Tuple[int, int] = (256, 192)

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, EfficientNetStemSurgeryStrategy):
            raise EfficientNetSurgeryError(
                f"strategy={self.strategy!r} must be EfficientNetStemSurgeryStrategy"
            )
        h, w = self.input_resolution
        if h <= 0 or w <= 0:
            raise EfficientNetSurgeryError(
                f"input_resolution={self.input_resolution} must have positive dims"
            )
        bh, bw = self.target_blind_spot_max
        if bh <= 0 or bw <= 0:
            raise EfficientNetSurgeryError(
                f"target_blind_spot_max={self.target_blind_spot_max} must have positive dims"
            )


CONTEST_SEGNET_STRIDE_2_BLIND_SPOT: Tuple[int, int] = (256, 192)
"""Per CLAUDE.md "Exact scorer architectures" verbatim: *"stride-2 stem
loses half resolution immediately -> artifacts below (256, 192) invisible."*

This is the canonical attack target for Slot RR pose-axis null projection
and sister Yousfi-Tier-1 pose-axis canonical helpers. Any perturbation
that lives ENTIRELY below this resolution is structurally invisible to
the contest SegNet."""


def compute_blind_spot_resolution_from_stride(
    input_resolution: Tuple[int, int],
    stride: int,
) -> Tuple[int, int]:
    """Compute effective blind-spot resolution after stride-N stem.

    Per CLAUDE.md "Exact scorer architectures" canonical math:
    EfficientNet stride-N stem reduces effective spatial resolution by
    factor N in both dimensions. Signal at frequencies above
    ``input_resolution / (2 * stride)`` is structurally lost (Nyquist
    limit + stride decimation).

    Parameters
    ----------
    input_resolution
        ``(height, width)`` of input frame.
    stride
        Stride of the first conv layer (canonical EfficientNet = 2;
        DDELab surgery = 1).

    Returns
    -------
    tuple[int, int]
        ``(blind_spot_height, blind_spot_width)`` = ``(H // (2*stride),
        W // (2*stride))``; signal below this resolution is invisible
        after the stem.

    Raises
    ------
    EfficientNetSurgeryError
        Invalid input.
    """
    h, w = input_resolution
    if h <= 0 or w <= 0:
        raise EfficientNetSurgeryError(
            f"input_resolution={input_resolution} must have positive dims"
        )
    if stride < 1:
        raise EfficientNetSurgeryError(f"stride={stride} must be >= 1")
    # Blind spot = signal below the Nyquist limit after stride decimation.
    # For stride=2: signal below H/4, W/4 is invisible.
    # For stride=1 (surgery): signal below H/2, W/2 is invisible (only
    # standard Nyquist applies; no stride decimation).
    return (h // (2 * stride), w // (2 * stride))
