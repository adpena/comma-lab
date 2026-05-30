# SPDX-License-Identifier: MIT
"""Canonical comma10k-baseline lineage (Yousfi 2023 -> contest SegNet 2026).

Origin: ``github.com/YassineYousfi/comma10k-baseline`` (Jul 6, 2023; 82 stars;
Yousfi's canonical road-segmentation baseline using comma.ai's comma10k
crowdsourced dataset). This is the DIRECT ANCESTRY of the contest SegNet:

* comma10k-baseline ships **U-Net + EfficientNet-B4 backbone** at 874x1164
  full resolution (Stage 2 of 2-stage training; per webfetch verbatim).
* Contest SegNet per CLAUDE.md "Exact scorer architectures":
  ``smp.Unet('tu-efficientnet_b2', classes=5, activation=None,
  encoder_weights=None)`` -- DOWNGRADE from B4 to B2 backbone but SAME
  ``smp.Unet`` framework + SAME 874x1164 native resolution.

The CANONICAL insight (Yousfi-comma transfer):
The contest output dimensions ``1164x874x1200x3 = 3_662_409_600 bytes`` per
Catalog #367 are NOT arbitrary -- they are the comma10k native resolution
that Yousfi's baseline trained on. The contest SegNet inherits both the
``smp.Unet`` framework choice AND the resolution discipline from Yousfi's
2023 baseline.

For our pose-axis attack cascade this means:
1. Per-pair perturbations should be evaluated at **NATIVE 874x1164** not
   the resized 384x512 input PoseNet sees -- because the GROUND-TRUTH
   masks/poses are derived from native-resolution decoded frames.
2. Any architecture we ship that REPLACES the contest SegNet should match
   the canonical ``smp.Unet + EfficientNet`` framework choice, OR document
   the FORK rationale per Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD.
3. The 2-stage training pattern (Stage 1: 437x582 100 epochs, Stage 2:
   874x1164 30 epochs harder-augmentation) is the canonical curriculum
   that Slot FF/YY/AAA/CCC substrate trainers can adopt.

5-axis adaptation taxonomy
--------------------------
* **Axis A (contest)** -- semantic-segmentation pretext -> contest scorer
  (different task but same architectural framework choice)
* **Axis B (problem space)** -- 5-class semantic segmentation -> 5-class
  segmentation (1:1; the contest SegNet output classes mirror comma10k)
* **Axis C (math)** -- 1:1 ``smp.Unet`` framework + same backbone family
* **Axis D (data)** -- comma10k crowdsourced 10k labeled frames -> contest
  ``upstream/videos/0.mkv`` 1200 frames at same native resolution
* **Axis E (video)** -- per-frame -> per-pair shared latent (contest pair
  structure is NEW; baseline was per-frame only)

Sister landings
---------------
* CLAUDE.md "Exact scorer architectures" SegNet section (canonical contest
  scorer + KNOWN architecture).
* Catalog #367 ``check_substrate_inflate_emits_expected_frame_count_or_fail_closed``
  (canonical 1164x874x1200x3 output contract that comes from comma10k).
* ``tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_efficientnet_steganalysis_surgery``
  (sister; explains why the canonical EfficientNet backbone has the
  stride-2 blind spot we can exploit).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

__all__ = (
    "Comma10kBaselineStage",
    "Comma10kTrainingStrategy",
    "Comma10kCurriculumConfig",
    "Comma10kLineageError",
    "compute_resolution_for_stage",
    "CONTEST_NATIVE_RESOLUTION",
    "COMMA10K_BASELINE_BACKBONE",
    "CONTEST_SEGNET_BACKBONE",
)


class Comma10kLineageError(ValueError):
    """Raised when comma10k lineage config violates a canonical invariant."""


CONTEST_NATIVE_RESOLUTION: Tuple[int, int] = (874, 1164)
"""Canonical contest native resolution from comma10k baseline Stage 2.
Per CLAUDE.md "Exact scorer architectures" + Catalog #367: raw output is
``1164 x 874 x 1200 x 3 = 3_662_409_600`` bytes."""


COMMA10K_BASELINE_BACKBONE: str = "tu-efficientnet_b4"
"""Yousfi's canonical baseline backbone for road segmentation (Jul 2023)."""


CONTEST_SEGNET_BACKBONE: str = "tu-efficientnet_b2"
"""Canonical contest SegNet backbone per CLAUDE.md "Exact scorer
architectures": ``smp.Unet('tu-efficientnet_b2', classes=5, activation=None,
encoder_weights=None)``. DOWNGRADE from comma10k baseline B4 to B2 (smaller +
faster but with smaller receptive field)."""


class Comma10kBaselineStage(str, Enum):
    """Canonical 2-stage training schedule (Yousfi comma10k-baseline).

    * ``STAGE_1_437x582_100_EPOCHS`` -- canonical Stage 1: 437x582 resolution
      (half of native), 100 epochs, easier augmentation. Per webfetch
      verbatim: *"Stage 1: 437x582 resolution with efficientnet-b4 backbone
      (100 epochs)"*.
    * ``STAGE_2_874x1164_30_EPOCHS`` -- canonical Stage 2: 874x1164 full
      native resolution, 30 epochs, harder augmentation. Per webfetch:
      *"Stage 2: 874x1164 full resolution with harder augmentation (30
      epochs)"*.
    """

    STAGE_1_437x582_100_EPOCHS = "stage_1_437x582_100_epochs"
    STAGE_2_874x1164_30_EPOCHS = "stage_2_874x1164_30_epochs"


class Comma10kTrainingStrategy(str, Enum):
    """Canonical comma10k training strategy adaptation taxonomy.

    * ``CANONICAL_2_STAGE_YOUSFI`` -- Yousfi's 1:1 canonical 2-stage
      training (Stage 1 half-res 100ep + Stage 2 full-res 30ep).
    * ``SINGLE_STAGE_FULL_RES_BASELINE`` -- ablation: train directly at
      full 874x1164 resolution from scratch. Slower convergence but
      tests whether the curriculum schedule matters.
    * ``FULL_RES_ONLY_FINE_TUNE_FROM_IMAGENET`` -- alternate: start from
      ImageNet pretrained backbone, fine-tune at full res; tests whether
      multi-resolution warm-start matters.
    * ``COMMA_PAIR_LATENT_ADAPTATION`` -- novel adaptation for the contest:
      train per-pair (frame_0, frame_1) shared-latent decoder at full
      native resolution; sister of HNeRV parity discipline L5 (full
      renderer not single-component slot).
    """

    CANONICAL_2_STAGE_YOUSFI = "canonical_2_stage_yousfi"
    SINGLE_STAGE_FULL_RES_BASELINE = "single_stage_full_res_baseline"
    FULL_RES_ONLY_FINE_TUNE_FROM_IMAGENET = "full_res_only_fine_tune_from_imagenet"
    COMMA_PAIR_LATENT_ADAPTATION = "comma_pair_latent_adaptation"


@dataclass(frozen=True)
class Comma10kCurriculumConfig:
    """Canonical comma10k curriculum config.

    Attributes
    ----------
    strategy
        Which canonical strategy to apply.
    backbone
        Canonical backbone to use (default ``tu-efficientnet_b2`` matches
        contest SegNet; comma10k baseline uses ``tu-efficientnet_b4``).
    target_resolution
        Final resolution for full-res training stage.
    """

    strategy: Comma10kTrainingStrategy = Comma10kTrainingStrategy.CANONICAL_2_STAGE_YOUSFI
    backbone: str = CONTEST_SEGNET_BACKBONE
    target_resolution: Tuple[int, int] = CONTEST_NATIVE_RESOLUTION

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, Comma10kTrainingStrategy):
            raise Comma10kLineageError(
                f"strategy={self.strategy!r} must be Comma10kTrainingStrategy"
            )
        if not self.backbone:
            raise Comma10kLineageError("backbone empty")
        h, w = self.target_resolution
        if h <= 0 or w <= 0:
            raise Comma10kLineageError(
                f"target_resolution={self.target_resolution} must have positive dims"
            )


def compute_resolution_for_stage(stage: Comma10kBaselineStage) -> Tuple[int, int]:
    """Return canonical (height, width) for given Yousfi baseline stage.

    1:1 with Yousfi comma10k-baseline README + webfetch report verbatim.

    Parameters
    ----------
    stage
        :class:`Comma10kBaselineStage`.

    Returns
    -------
    tuple[int, int]
        ``(height, width)``.

    Raises
    ------
    Comma10kLineageError
        Unknown stage.
    """
    if stage == Comma10kBaselineStage.STAGE_1_437x582_100_EPOCHS:
        return (437, 582)
    if stage == Comma10kBaselineStage.STAGE_2_874x1164_30_EPOCHS:
        return (874, 1164)
    raise Comma10kLineageError(f"unhandled stage {stage!r}")
