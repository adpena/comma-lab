# SPDX-License-Identifier: MIT
"""Canonical color-separation strategy (Yousfi 2019 ALASKA Pattern #1).

Origin: ``external/alaska_yousfi/src/tools/jpeg_utils.py:50-62`` upstream
``branch_to_slice`` + ``models.py:32-40`` ``SR_net_feature_extractor_beast``
multi-branch loop.

The CANONICAL insight (Yousfi 2019 ALASKA-#1-winning):
Each color channel of a JPEG image carries DIFFERENT steganalysis signal.
Detectors trained on YCrCb (full), Y alone, CrCb alone, Cr alone, Cb alone
yield different feature-map statistics + together (via MLP fusion) beat any
single-channel detector by ~3-5 percentage points on ALASKA validation.

5-axis adaptation taxonomy
--------------------------
* **Axis A (contest)** -- JPEG quantization-noise channels -> contest YUV6
  (4 luma + 2 chroma subsampled per CLAUDE.md "Exact scorer architectures")
* **Axis B (problem space)** -- detection -> generation
  (we minimise distortion not classify; channels still carry distinct signal)
* **Axis C (math)** -- channel-wise slice tensor[:, branch_slice, :, :]
  1:1 with upstream branch_to_slice
* **Axis D (data)** -- 256x256 JPEG -> 1164x874 contest frames + 384x512
  resized per CLAUDE.md scorer input contract
* **Axis E (video)** -- single-image -> per-pair shared latent
  (channel branches still per-frame; pair structure orthogonal)

Sister of slot ``ColorBranchSliceStrategy`` at the COMMA YUV6 surface
(YUV6 = 4 luma planes + 2 chroma subsampled per
``PoseNet.preprocess_input`` upstream).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Tuple

__all__ = (
    "ColorBranchStrategy",
    "ColorBranchSliceStrategy",
    "branch_to_yuv6_channel_slice",
    "SRNET_BRANCH_ORDER",
    "YUV6_CHANNEL_LAYOUT",
)


SRNET_BRANCH_ORDER: Tuple[str, ...] = ("YCrCb", "CrCb", "Y", "Cr", "Cb")
"""Canonical 5-branch order Yousfi's ``SR_net_feature_extractor_beast`` uses
(upstream ``models.py:34``). Order matters because the warm-start checkpoint
loader assigns scope by ``branch+'/LayerN'`` and the MLP fusion takes the
features stacked along axis 0 in this order."""


YUV6_CHANNEL_LAYOUT: Mapping[str, Tuple[int, ...]] = {
    "Y0": (0,),
    "Y1": (1,),
    "Y2": (2,),
    "Y3": (3,),
    "U": (4,),
    "V": (5,),
    "Y_only": (0, 1, 2, 3),
    "UV_only": (4, 5),
    "YUV6_full": (0, 1, 2, 3, 4, 5),
    "Y0_UV": (0, 4, 5),
    "Y123_UV": (1, 2, 3, 4, 5),
}
"""Canonical 6-channel layout for ``contest_auth_eval`` PoseNet input
(per CLAUDE.md "Exact scorer architectures"). PoseNet takes
``rgb_to_yuv6 -> resize -> (B, T*6, H/2, W/2)`` so each frame contributes 6
channels (4 luma planes + 2 chroma subsampled). The 11 named slices below
mirror Yousfi's 5-branch ALASKA pattern but at the COMMA YUV6 contract:

* ``Y0/Y1/Y2/Y3`` -- single-luma-plane branches (sister of Yousfi's ``Y``)
* ``U/V`` -- single-chroma branches (sister of Yousfi's ``Cr``/``Cb``)
* ``Y_only`` -- all-luma (sister of Yousfi's ``Y``)
* ``UV_only`` -- all-chroma (sister of Yousfi's ``CrCb``)
* ``YUV6_full`` -- all 6 channels (sister of Yousfi's ``YCrCb``)
* ``Y0_UV`` + ``Y123_UV`` -- mixed-decomposition branches (NOVEL; tests the
  Y0 vs Y123 hypothesis: which luma plane carries the bulk of PoseNet's signal)
"""


class ColorBranchStrategy(str, Enum):
    """Canonical ALASKA color-separation branches.

    1:1 with Yousfi's ``branch_to_slice`` (``jpeg_utils.py:50-62``).
    The order MUST match ``SRNET_BRANCH_ORDER`` for warm-start compatibility.
    """

    YCRCB = "YCrCb"  # full 3-channel
    CRCB = "CrCb"  # 2-chroma
    Y = "Y"  # luma only
    CR = "Cr"  # red-chroma only
    CB = "Cb"  # blue-chroma only


class ColorBranchSliceStrategy(str, Enum):
    """Canonical COMMA YUV6 branches (sister of :class:`ColorBranchStrategy`
    at the contest scorer surface).

    11 slices mirror Yousfi's 5-branch + add 6 NOVEL slices that exploit
    YUV6's 4-luma-plane structure. Per CLAUDE.md "Exact scorer architectures":
    PoseNet expects 12 channels (2 frames x YUV6); each branch slice is a
    per-frame channel-subset that the canonical pair-constraint batcher
    extends to ``2 x len(slice)`` for the pair.
    """

    Y0 = "Y0"
    Y1 = "Y1"
    Y2 = "Y2"
    Y3 = "Y3"
    U = "U"
    V = "V"
    Y_ONLY = "Y_only"
    UV_ONLY = "UV_only"
    YUV6_FULL = "YUV6_full"
    Y0_UV = "Y0_UV"
    Y123_UV = "Y123_UV"


def branch_to_yuv6_channel_slice(
    branch: ColorBranchSliceStrategy | str,
) -> Tuple[int, ...]:
    """Return the channel-index tuple for ``branch``.

    Canonical helper for the COMMA YUV6 contract; sister of Yousfi's
    ``branch_to_slice(branch)`` at the JPEG YCrCb contract.

    Parameters
    ----------
    branch
        One of :class:`ColorBranchSliceStrategy` (enum) or its string value
        (e.g. ``"Y0_UV"`` / ``"YUV6_full"`` / ``"Y_only"`` / ...).

    Returns
    -------
    tuple[int, ...]
        Channel indices into the YUV6 6-channel layout (Y0/Y1/Y2/Y3/U/V).
        Use as ``yuv6_tensor[:, list(slice), :, :]`` (NCHW) or
        ``yuv6_tensor[..., list(slice)]`` (NHWC).

    Raises
    ------
    KeyError
        If ``branch`` is not a recognized branch name.
    """
    key = branch.value if isinstance(branch, ColorBranchSliceStrategy) else str(branch)
    if key not in YUV6_CHANNEL_LAYOUT:
        raise KeyError(
            f"branch={key!r} not in canonical YUV6 layout; "
            f"valid={list(YUV6_CHANNEL_LAYOUT.keys())}"
        )
    return YUV6_CHANNEL_LAYOUT[key]
