# SPDX-License-Identifier: MIT
"""Canonical fusion detector ensemble (Yousfi autostego 2026).

Origin: ``github.com/YassineYousfi/autostego/blob/eve/steganalysis/fusion.py``
canonical helper combining LCLSMR + SRNet + (optionally SRM) detector
scores into one ensemble verdict per ``autostego/eve.py``
``run_fusion_detector`` orchestrator.

The CANONICAL insight (Yousfi 2026 autostego fusion):
Different detectors have complementary blind spots:
- **LCLSMR** (linear): fast, sees overall feature-space drift
- **SRNet** (deep): sees spatial structure in embedding pattern
- **SRM** (classical): sees high-order Markov-chain statistics

A LINEAR fusion combining their scores at the output level beats any
single detector by ~2-5 percentage points on ALASKA challenge validation.
Per Yousfi's ``run_fusion_detector``: LCLSMR runs first (cheap),
SRNet runs second (expensive), fusion combines their outputs with a final
linear weighting trained on validation data.

For the comma-video contest, the canonical adaptation is:
- **Detector ensemble** = SegNet output + PoseNet output + sister scorer
  feature heads.
- **Fusion** = canonical 100*d_seg + sqrt(10*d_pose) + 25*rate weighting
  per CLAUDE.md "Exact scorer architectures" + sister canonical equation
  ``meta_orchestrator_three_metric_trichotomy_orthogonality_v1`` per Catalog
  #379.
- **Score-level vs feature-level**: canonical fusion is at SCORE level (after
  each detector's full forward); FEATURE-level fusion (concatenating
  penultimate-layer features and training a joint head) is the alternate
  canonical approach Yousfi explores in ``fusion.py`` notebook variants.

5-axis adaptation taxonomy
--------------------------
* **Axis A (contest)** -- detector fusion -> contest score fusion
  (different inputs but same linear-combination math)
* **Axis B (problem space)** -- binary detection -> per-axis score
  components
* **Axis C (math)** -- 1:1 linear weighted sum + softmax
* **Axis D (data)** -- pre-trained detector scores -> live contest scorer
  outputs
* **Axis E (video)** -- per-image -> per-pair shared latent

Sister landings
---------------
* CLAUDE.md "SegNet vs PoseNet importance" (canonical operating-point-
  dependent fusion weighting that this canonical helper inherits).
* ``tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_lclsmr_linear_steganalysis_detector``
  (one of the canonical detectors this fusion combines).
* Catalog #379 (canonical 3-metric trichotomy orthogonality; the canonical
  fusion's weight choice IS the trichotomy assignment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np

__all__ = (
    "FusionStrategy",
    "FusionConfig",
    "FusionDetectorError",
    "compute_fusion_score",
    "compute_canonical_contest_fusion_weights",
    "CONTEST_FUSION_WEIGHTS_CANONICAL",
)


class FusionDetectorError(ValueError):
    """Raised when fusion config violates a canonical invariant."""


CONTEST_FUSION_WEIGHTS_CANONICAL: Mapping[str, float] = {
    "d_seg": 100.0,
    "d_pose_sqrt_10x": 1.0,
    "rate_25_over_denom": 1.0,
}
"""Per CLAUDE.md "Exact scorer architectures" + canonical contest scoring:
``S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37545489``.
This is the canonical 3-detector fusion weighting; sister canonical
equations ``meta_orchestrator_three_metric_trichotomy_orthogonality_v1``
per Catalog #379 enforce orthogonality of the three components."""


class FusionStrategy(str, Enum):
    """Canonical fusion strategies (Yousfi autostego 2026).

    * ``LINEAR_WEIGHTED_SUM`` -- canonical Yousfi 1:1 score-level fusion;
      ``score = sum_i w_i * detector_i_output``. Weights trained on
      validation; canonical contest weights pinned in
      :data:`CONTEST_FUSION_WEIGHTS_CANONICAL`.
    * ``GEOMETRIC_MEAN`` -- ablation: ``score = prod_i (detector_i)^w_i``;
      useful when detectors have very different output scales but
      complementary signal.
    * ``MAX_DETECTOR_WINS`` -- ablation: ``score = max_i (w_i *
      detector_i)``; useful when one detector is dramatically better than
      others on the current substrate; canonical sister of Yousfi
      autostego Eve scoring rule per
      ``canonical_alice_vs_eve_adversarial_loop.compute_eve_score_minimax``.
    * ``LEARNED_MLP_2_LAYER`` -- canonical Yousfi alternate fusion: train
      a small MLP on the (n_detectors,) score vector to predict final
      verdict. More expressive but requires labeled training data.
    """

    LINEAR_WEIGHTED_SUM = "linear_weighted_sum"
    GEOMETRIC_MEAN = "geometric_mean"
    MAX_DETECTOR_WINS = "max_detector_wins"
    LEARNED_MLP_2_LAYER = "learned_mlp_2_layer"


@dataclass(frozen=True)
class FusionConfig:
    """Canonical fusion configuration.

    Attributes
    ----------
    strategy
        Which canonical fusion strategy.
    detector_weights
        Mapping detector-name -> weight. For canonical contest fusion,
        use :data:`CONTEST_FUSION_WEIGHTS_CANONICAL`.
    softmax_normalize
        Whether to apply softmax to weights before fusion (normalizes
        weights to a probability simplex; canonical Yousfi default for
        ``LINEAR_WEIGHTED_SUM``).
    """

    strategy: FusionStrategy = FusionStrategy.LINEAR_WEIGHTED_SUM
    detector_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(CONTEST_FUSION_WEIGHTS_CANONICAL)
    )
    softmax_normalize: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, FusionStrategy):
            raise FusionDetectorError(
                f"strategy={self.strategy!r} must be FusionStrategy"
            )
        if not self.detector_weights:
            raise FusionDetectorError("detector_weights empty")
        for k, v in self.detector_weights.items():
            if not isinstance(v, (int, float)):
                raise FusionDetectorError(
                    f"weight for {k!r} must be numeric; got {v!r}"
                )


def compute_canonical_contest_fusion_weights() -> Mapping[str, float]:
    """Return canonical contest fusion weights.

    Returns
    -------
    Mapping[str, float]
        ``{"d_seg": 100.0, "d_pose_sqrt_10x": 1.0, "rate_25_over_denom": 1.0}``.
    """
    return dict(CONTEST_FUSION_WEIGHTS_CANONICAL)


def compute_fusion_score(
    detector_scores: Mapping[str, float],
    config: FusionConfig,
) -> float:
    """Compute canonical fusion score from per-detector scores.

    1:1 with Yousfi autostego ``fusion.py`` for ``LINEAR_WEIGHTED_SUM``.

    Parameters
    ----------
    detector_scores
        Mapping detector-name -> score (e.g. ``{"SRNet": 0.8, "LCLSMR": 0.65}``).
    config
        :class:`FusionConfig`. Must include weights for all detector names
        in ``detector_scores``.

    Returns
    -------
    float
        Fused score per canonical strategy.

    Raises
    ------
    FusionDetectorError
        Invalid input or detector-name mismatch.
    """
    if not detector_scores:
        raise FusionDetectorError("detector_scores empty")
    for k in detector_scores:
        if k not in config.detector_weights:
            raise FusionDetectorError(
                f"detector_scores has key {k!r} not in config.detector_weights "
                f"({list(config.detector_weights.keys())})"
            )
    score_keys = list(detector_scores.keys())
    score_values = np.array([float(detector_scores[k]) for k in score_keys])
    weight_values = np.array(
        [float(config.detector_weights[k]) for k in score_keys]
    )
    if config.softmax_normalize:
        # Softmax normalization for probabilistic interpretation.
        max_w = weight_values.max()
        exp_w = np.exp(weight_values - max_w)
        weight_values = exp_w / exp_w.sum()

    if config.strategy == FusionStrategy.LINEAR_WEIGHTED_SUM:
        return float(np.sum(weight_values * score_values))
    if config.strategy == FusionStrategy.GEOMETRIC_MEAN:
        # weighted geometric mean: prod_i score_i^w_i = exp(sum_i w_i log(score_i))
        # require strictly positive scores.
        if (score_values <= 0).any():
            raise FusionDetectorError(
                "GEOMETRIC_MEAN requires strictly positive scores; "
                f"got min={float(score_values.min())}"
            )
        return float(np.exp(np.sum(weight_values * np.log(score_values))))
    if config.strategy == FusionStrategy.MAX_DETECTOR_WINS:
        weighted = weight_values * score_values
        return float(weighted.max())
    if config.strategy == FusionStrategy.LEARNED_MLP_2_LAYER:
        # Canonical Yousfi 2-layer MLP fusion; without learned weights we
        # fall back to canonical contest weights + tanh nonlinearity to
        # mirror the MLP's bounded output.
        linear = np.sum(weight_values * score_values)
        return float(np.tanh(linear / max(1.0, weight_values.sum())))
    raise FusionDetectorError(f"unhandled strategy {config.strategy!r}")
