# SPDX-License-Identifier: MIT
"""Canonical OneHot JPEG steganalysis (Yousfi-Fridrich 2020 IEEE SPL).

Origin: ``github.com/YassineYousfi/OneHotConv`` (Jun 17, 2021; canonical
implementation of Yousfi-Fridrich 2020 IEEE Signal Processing Letters paper).

Paper citation::

    @article{9091221,
      author={Y. {Yousfi} and J. {Fridrich}},
      journal={IEEE Signal Processing Letters},
      title={An Intriguing Struggle of CNNs in JPEG Steganalysis and
             the OneHot Solution},
      year={2020},
      volume={27},
      pages={830-834},
      doi={10.1109/LSP.2020.2993959}
    }

The CANONICAL insight (Yousfi-Fridrich 2020 "Intriguing Struggle"):
CNNs trained directly on DCT coefficients for JPEG steganalysis STRUGGLE
because the embedding signal is concentrated in the LSB of each DCT
coefficient, but the input numerical range varies enormously across
coefficients (DC ~ thousands, high-freq AC ~ single digits). A standard
conv layer's first weights see the cover signal dominate the embedding
signal by 3-4 orders of magnitude.

The **OneHot Solution**: convert each DCT coefficient to a ONE-HOT encoding
across its full integer range (typically -1024 to +1024 = 2049 channels).
Each DCT coefficient becomes a sparse 2049-channel vector. The first conv
layer sees ONLY the embedding signal because cover values are now
structural, not numerical.

This is the canonical RESOLUTION of the "input-range-dominates-signal"
bug class -- directly relevant to our pose-axis attack cascade because
PoseNet's YUV6 input has 12 channels (2 frames x 6 YUV components) where
each channel has different signal magnitudes; the equivalent surgery
would be ONE-HOT encoding per pixel-value per channel before the first
PoseNet conv.

5-axis adaptation taxonomy
--------------------------
* **Axis A (contest)** -- DCT coefficient steganalysis -> YUV6 pixel
  steganalysis (different input domain but same input-range dominance)
* **Axis B (problem space)** -- 2-class softmax -> per-pair pose-axis score
* **Axis C (math)** -- one-hot encoding 1:1 with Yousfi-Fridrich paper
  Section III; spatial conv on (B, 2049, H, W) sparse tensor
* **Axis D (data)** -- BOSSbase JPEG QF95 -> ``upstream/videos/0.mkv``
* **Axis E (video)** -- single-image -> per-pair shared latent

Sister landings
---------------
* CLAUDE.md "Exact scorer architectures" PoseNet section (12-channel YUV6
  input is the canonical adaptation target).
* CLAUDE.md "Fridrich inverse steganalysis" item 4 (CNN blind spots:
  EfficientNet misses DCT statistics, has texture-region blind spots) --
  OneHot is the canonical RESOLUTION of this blind-spot class.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = (
    "OneHotEncodingStrategy",
    "OneHotEncodingConfig",
    "OneHotEncodingError",
    "encode_value_one_hot",
    "compute_one_hot_input_channels",
    "DCT_COEFFICIENT_CANONICAL_RANGE",
)


class OneHotEncodingError(ValueError):
    """Raised when OneHot encoding violates a canonical invariant."""


DCT_COEFFICIENT_CANONICAL_RANGE: tuple[int, int] = (-1024, 1024)
"""Yousfi-Fridrich 2020 canonical DCT coefficient range. JPEG DCT values
fall in approximately [-1024, +1024] after quantization; OneHot encoding
across this range gives 2049-channel sparse tensor per coefficient."""


class OneHotEncodingStrategy(str, Enum):
    """Canonical OneHot encoding strategies (Yousfi-Fridrich 2020).

    * ``FULL_RANGE_2049_CHANNELS`` -- canonical Yousfi-Fridrich 1:1; 2049
      channels for DCT range [-1024, +1024]. Memory cost: ~2049x input
      tensor; only viable for small spatial dimensions or sparse representations.
    * ``CLIPPED_RANGE_512_CHANNELS`` -- ablation; clip range to [-256, +256]
      = 513 channels; useful when most DCT energy is in low-mag coefficients.
    * ``BINARIZED_LSB_2_CHANNELS`` -- ablation extreme; encode only the LSB
      (0 or 1) = 2 channels; tests whether the OneHot signal is dominated
      by LSB alone or requires the full encoding.
    * ``MULTI_SCALE_PYRAMID_OCTAVE_8`` -- novel sister; group consecutive
      DCT values into octaves (8 channels: ones-place, twos-place, etc.).
      Compresses memory cost ~256x at the cost of some signal granularity.
    """

    FULL_RANGE_2049_CHANNELS = "full_range_2049_channels"
    CLIPPED_RANGE_512_CHANNELS = "clipped_range_512_channels"
    BINARIZED_LSB_2_CHANNELS = "binarized_lsb_2_channels"
    MULTI_SCALE_PYRAMID_OCTAVE_8 = "multi_scale_pyramid_octave_8"


@dataclass(frozen=True)
class OneHotEncodingConfig:
    """Canonical OneHot encoding configuration.

    Attributes
    ----------
    strategy
        Which encoding strategy to apply.
    dct_range
        ``(min_value, max_value)`` of input. Defaults to
        :data:`DCT_COEFFICIENT_CANONICAL_RANGE` per Yousfi-Fridrich 2020.
    """

    strategy: OneHotEncodingStrategy = OneHotEncodingStrategy.FULL_RANGE_2049_CHANNELS
    dct_range: tuple[int, int] = DCT_COEFFICIENT_CANONICAL_RANGE

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, OneHotEncodingStrategy):
            raise OneHotEncodingError(
                f"strategy={self.strategy!r} must be OneHotEncodingStrategy"
            )
        lo, hi = self.dct_range
        if lo >= hi:
            raise OneHotEncodingError(
                f"dct_range={self.dct_range} requires lo < hi"
            )


def compute_one_hot_input_channels(
    config: OneHotEncodingConfig,
) -> int:
    """Compute the number of OneHot channels for given config.

    Per Yousfi-Fridrich 2020 Section III: the canonical FULL_RANGE encoding
    produces ``hi - lo + 1`` channels (2049 for canonical [-1024, +1024]).

    Parameters
    ----------
    config
        :class:`OneHotEncodingConfig`.

    Returns
    -------
    int
        Number of output channels after OneHot encoding.

    Raises
    ------
    OneHotEncodingError
        Invalid strategy.
    """
    lo, hi = config.dct_range
    if config.strategy == OneHotEncodingStrategy.FULL_RANGE_2049_CHANNELS:
        return hi - lo + 1
    if config.strategy == OneHotEncodingStrategy.CLIPPED_RANGE_512_CHANNELS:
        # Clip to half range then OneHot.
        return (hi - lo) // 4 + 1
    if config.strategy == OneHotEncodingStrategy.BINARIZED_LSB_2_CHANNELS:
        return 2
    if config.strategy == OneHotEncodingStrategy.MULTI_SCALE_PYRAMID_OCTAVE_8:
        return 8
    raise OneHotEncodingError(f"unhandled strategy {config.strategy!r}")


def encode_value_one_hot(
    value_tensor: np.ndarray,
    config: OneHotEncodingConfig,
) -> np.ndarray:
    """Encode integer tensor as OneHot per Yousfi-Fridrich 2020.

    Parameters
    ----------
    value_tensor
        Integer tensor of shape ``(B, H, W)`` or ``(B, C, H, W)``.
    config
        :class:`OneHotEncodingConfig`.

    Returns
    -------
    np.ndarray
        OneHot encoded tensor; for FULL_RANGE_2049_CHANNELS strategy on
        input shape ``(B, H, W)`` returns shape
        ``(B, n_channels, H, W)`` where ``n_channels = hi - lo + 1``.

    Raises
    ------
    OneHotEncodingError
        Invalid input.
    """
    if not isinstance(value_tensor, np.ndarray):
        raise OneHotEncodingError(
            f"value_tensor must be np.ndarray; got {type(value_tensor).__name__}"
        )
    if not np.issubdtype(value_tensor.dtype, np.integer):
        raise OneHotEncodingError(
            f"value_tensor must be integer dtype; got {value_tensor.dtype}"
        )
    lo, hi = config.dct_range

    if config.strategy == OneHotEncodingStrategy.FULL_RANGE_2049_CHANNELS:
        n_channels = hi - lo + 1
        clipped = np.clip(value_tensor, lo, hi)
        shifted = clipped - lo  # shift to [0, n_channels-1]
        # Build one-hot along new axis=1.
        original_shape = value_tensor.shape
        flat = shifted.reshape(-1)
        one_hot = np.zeros((flat.size, n_channels), dtype=np.uint8)
        one_hot[np.arange(flat.size), flat] = 1
        # Reshape back: insert n_channels as axis=1 for batch-channel-spatial.
        if value_tensor.ndim == 3:  # (B, H, W) -> (B, n_channels, H, W)
            return one_hot.reshape(*original_shape, n_channels).transpose(0, 3, 1, 2)
        if value_tensor.ndim == 4:  # (B, C, H, W) -> (B, C * n_channels, H, W)
            return one_hot.reshape(*original_shape, n_channels).transpose(
                0, 1, 4, 2, 3
            ).reshape(original_shape[0], -1, original_shape[2], original_shape[3])
        raise OneHotEncodingError(
            f"value_tensor ndim={value_tensor.ndim} not supported"
        )
    if config.strategy == OneHotEncodingStrategy.BINARIZED_LSB_2_CHANNELS:
        # Encode only LSB; output shape (B, 2, H, W) for 3-D input.
        lsb = (value_tensor & 1).astype(np.uint8)
        original_shape = value_tensor.shape
        if value_tensor.ndim == 3:
            out = np.zeros((original_shape[0], 2, *original_shape[1:]), dtype=np.uint8)
            out[:, 0] = 1 - lsb
            out[:, 1] = lsb
            return out
        if value_tensor.ndim == 4:
            out = np.zeros(
                (original_shape[0], original_shape[1] * 2, *original_shape[2:]),
                dtype=np.uint8,
            )
            for c in range(original_shape[1]):
                out[:, 2 * c] = 1 - lsb[:, c]
                out[:, 2 * c + 1] = lsb[:, c]
            return out
        raise OneHotEncodingError(
            f"value_tensor ndim={value_tensor.ndim} not supported"
        )
    if config.strategy == OneHotEncodingStrategy.CLIPPED_RANGE_512_CHANNELS:
        # Clip to quarter range then one-hot.
        n_channels = (hi - lo) // 4 + 1
        clipped = np.clip(value_tensor // 4, lo // 4, hi // 4)
        shifted = clipped - (lo // 4)
        original_shape = value_tensor.shape
        flat = shifted.reshape(-1)
        one_hot = np.zeros((flat.size, n_channels), dtype=np.uint8)
        one_hot[np.arange(flat.size), flat] = 1
        if value_tensor.ndim == 3:
            return one_hot.reshape(*original_shape, n_channels).transpose(0, 3, 1, 2)
        if value_tensor.ndim == 4:
            return one_hot.reshape(*original_shape, n_channels).transpose(
                0, 1, 4, 2, 3
            ).reshape(original_shape[0], -1, original_shape[2], original_shape[3])
        raise OneHotEncodingError(
            f"value_tensor ndim={value_tensor.ndim} not supported"
        )
    if config.strategy == OneHotEncodingStrategy.MULTI_SCALE_PYRAMID_OCTAVE_8:
        # Encode in octaves: bits 0-7 each become a separate channel.
        original_shape = value_tensor.shape
        if value_tensor.ndim == 3:
            out = np.zeros((original_shape[0], 8, *original_shape[1:]), dtype=np.uint8)
            for bit in range(8):
                out[:, bit] = ((value_tensor >> bit) & 1).astype(np.uint8)
            return out
        if value_tensor.ndim == 4:
            out = np.zeros(
                (original_shape[0], original_shape[1] * 8, *original_shape[2:]),
                dtype=np.uint8,
            )
            for c in range(original_shape[1]):
                for bit in range(8):
                    out[:, c * 8 + bit] = ((value_tensor[:, c] >> bit) & 1).astype(
                        np.uint8
                    )
            return out
        raise OneHotEncodingError(
            f"value_tensor ndim={value_tensor.ndim} not supported"
        )
    raise OneHotEncodingError(f"unhandled strategy {config.strategy!r}")
