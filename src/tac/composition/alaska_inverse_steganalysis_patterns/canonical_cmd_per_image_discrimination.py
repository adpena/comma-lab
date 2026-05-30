# SPDX-License-Identifier: MIT
"""Canonical CMD (Cross-Match Discrimination) per-image score (Yousfi ALASKA Pattern #5).

Origin: derived from Yousfi's per-image feature-map -> MLP fusion in
``external/alaska_yousfi/src/notebooks/tf_extract_features_color_separated.ipynb``
cells 4 + 10 + ``models.py:32-40`` ``SR_net_feature_extractor_beast``
returning ``(avg, var, min, max)`` 4-stat pool per branch.

The CANONICAL insight (Yousfi 2019 ALASKA-#1-winning):
After per-branch feature extraction, the 4-statistic pool ``(avg, var, min,
max)`` from Layer12 of SRNet gives the MLP detector a compact + powerful
fingerprint of the embedding signal. The MLP discriminates better than a
softmax-head on the raw conv features because the moment-statistics
abstract away spatial layout (which doesn't carry steganalysis signal in
JPEG domain) while preserving the EMBEDDING-LEVEL statistical perturbation.

We call this Cross-Match Discrimination (CMD) per the ALASKA paper's
terminology: ``avg`` = first moment, ``var`` = second moment, ``min``/``max``
= extrema-statistics. The 4-stat fingerprint is the canonical pattern.

5-axis adaptation taxonomy
--------------------------
* **Axis A (contest)** -- JPEG embedding signal -> contest YUV6 perturbation
* **Axis B (problem space)** -- 5-class softmax -> per-pair discrimination
* **Axis C (math)** -- 1:1 (avg, var, min, max) 4-stat tuple
* **Axis D (data)** -- pre-extracted feature-maps -> live computation on
  per-pair latent in the inflate runtime
* **Axis E (video)** -- per-image -> per-pair shared latent

Sister of slot Catalog #205 ``select_inflate_device`` canonical-helper
discipline (CMD score must be computed via the canonical inflate-time
device-selection per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

__all__ = (
    "CMDDiscriminationStrategy",
    "CMDDiscriminationConfig",
    "compute_cmd_per_image_score",
    "CMD_4_STAT_NAMES",
    "CMDDiscriminationError",
)


class CMDDiscriminationError(ValueError):
    """Raised when CMD computation violates a canonical invariant."""


CMD_4_STAT_NAMES: tuple[str, ...] = ("avg", "var", "min", "max")
"""Canonical Yousfi 4-statistic pool order per ``models.py:140`` upstream
``tf.squeeze(tf.stack([avgp, varp, minp, maxp]))``."""


class CMDDiscriminationStrategy(str, Enum):
    """Canonical CMD discrimination strategies.

    * ``MOMENT_4_STAT`` -- Yousfi's canonical (avg, var, min, max)
      4-stat pool from the final conv layer (the ALASKA-#1 pattern).
    * ``MOMENT_2_STAT`` -- (avg, var) only; lighter-weight variant for
      diagnostic-only smokes.
    * ``EXTREMA_2_STAT`` -- (min, max) only; useful when the embedding
      signal is concentrated in tail-statistics.
    * ``RAW_CONV_MEAN`` -- bare ``conv_features.mean(axis=spatial)``
      baseline for ablation (NOT recommended; included for Catalog #303
      cargo-cult-audit comparison).
    """

    MOMENT_4_STAT = "moment_4_stat"
    MOMENT_2_STAT = "moment_2_stat"
    EXTREMA_2_STAT = "extrema_2_stat"
    RAW_CONV_MEAN = "raw_conv_mean"


@dataclass(frozen=True)
class CMDDiscriminationConfig:
    """Canonical CMD config.

    Attributes
    ----------
    strategy
        Which CMD strategy to compute.
    spatial_axes
        Which axes to reduce over (must be a tuple of valid feature-tensor
        axes). Defaults to (-2, -1) for NCHW tensors (reducing H + W;
        keeping batch + channel).
    """

    strategy: CMDDiscriminationStrategy = CMDDiscriminationStrategy.MOMENT_4_STAT
    spatial_axes: tuple[int, ...] = (-2, -1)

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, CMDDiscriminationStrategy):
            raise CMDDiscriminationError(
                f"strategy={self.strategy!r} must be CMDDiscriminationStrategy"
            )
        if not self.spatial_axes:
            raise CMDDiscriminationError("spatial_axes empty")


def compute_cmd_per_image_score(
    feature_tensor: np.ndarray,
    config: CMDDiscriminationConfig,
) -> dict[str, np.ndarray]:
    """Compute canonical CMD score per image.

    1:1 with Yousfi's upstream ``tf.nn.moments(bn, axes=[2,3])`` +
    ``tf.reduce_min/max(bn, axis=[2,3])`` at ``models.py:139-140``.

    Parameters
    ----------
    feature_tensor
        Feature tensor from the penultimate conv layer; shape
        ``(B, C, H, W)`` for NCHW or ``(B, H, W, C)`` for NHWC. The
        ``spatial_axes`` in config selects which axes to reduce.
    config
        :class:`CMDDiscriminationConfig`.

    Returns
    -------
    dict[str, np.ndarray]
        Per-strategy named statistics, e.g. for ``MOMENT_4_STAT``:
        ``{"avg": ..., "var": ..., "min": ..., "max": ...}``. Each value
        has the same shape as ``feature_tensor`` with ``spatial_axes``
        reduced out.

    Raises
    ------
    CMDDiscriminationError
        Invalid feature_tensor or config.
    """
    if not isinstance(feature_tensor, np.ndarray):
        raise CMDDiscriminationError(
            f"feature_tensor must be np.ndarray; got {type(feature_tensor).__name__}"
        )
    if feature_tensor.size == 0:
        raise CMDDiscriminationError("feature_tensor empty")
    axes = config.spatial_axes
    # Validate axes are in range.
    ndim = feature_tensor.ndim
    for ax in axes:
        if not (-ndim <= ax < ndim):
            raise CMDDiscriminationError(
                f"axis={ax} out of range for tensor of ndim={ndim}"
            )
    if config.strategy == CMDDiscriminationStrategy.MOMENT_4_STAT:
        return {
            "avg": feature_tensor.mean(axis=axes),
            "var": feature_tensor.var(axis=axes),
            "min": feature_tensor.min(axis=axes),
            "max": feature_tensor.max(axis=axes),
        }
    if config.strategy == CMDDiscriminationStrategy.MOMENT_2_STAT:
        return {
            "avg": feature_tensor.mean(axis=axes),
            "var": feature_tensor.var(axis=axes),
        }
    if config.strategy == CMDDiscriminationStrategy.EXTREMA_2_STAT:
        return {
            "min": feature_tensor.min(axis=axes),
            "max": feature_tensor.max(axis=axes),
        }
    if config.strategy == CMDDiscriminationStrategy.RAW_CONV_MEAN:
        return {
            "raw_mean": feature_tensor.mean(axis=axes),
        }
    raise CMDDiscriminationError(f"unhandled strategy {config.strategy!r}")
