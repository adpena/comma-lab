# SPDX-License-Identifier: MIT
"""F10: YUV6 sublattice geometry primitive.

The contest scorer's PoseNet preprocessing applies ``rgb_to_yuv6`` (per
``upstream/frame_utils.py:50-78``): the RGB frame is projected to 6 channels
at HALF-resolution = ``[Y00, Y10, Y01, Y11, U_sub, V_sub]`` where the four
luma channels are the four 2x2 luma sublattices and the two chroma channels
are the 2x2-averaged U/V components.

Per deep_math §2.2, this is a structured downsampling: each RGB pixel
``(y, x)`` belongs to **exactly one** luma sublattice via the index
``(y % 2, x % 2)``. The four luma sublattices partition the 874x1164
camera-frame into four non-overlapping half-resolution grids.

This primitive decomposes a target RGB frame into the YUV6 sublattice
geometry and computes per-sublattice gradient norms — telling the
bit-allocator which luma sublattice is most score-sensitive (typically
the sublattice with maximal local motion or texture).

Wire-in hooks engaged:

- ``sensitivity_map``: per-luma-sublattice score gradient produces 4
  weighted sensitivity maps for cost-map weighting.
- ``bit_allocator``: budget allocation across the 6 channels of YUV6 can
  use per-sublattice gradient priority.

Cross-references
----------------
- Source: ``upstream/frame_utils.py:50-78`` (structural-code-contract anchor)
- Deep math memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §2.2
- Sister: BT.601 full-range RGB-to-YUV conversion

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tac.xray.base import (
    ComposedXRayPrimitive,
    WireInHook,
    XRayPrimitiveResult,
)

# BT.601 full-range RGB-to-YCbCr coefficients (pinned from frame_utils.py).
BT601_R = (0.299, 0.587, 0.114)
BT601_U = (-0.168736, -0.331264, 0.5)
BT601_V = (0.5, -0.418688, -0.081312)
LUMA_SUBLATTICE_INDICES = ((0, 0), (1, 0), (0, 1), (1, 1))


@dataclass(frozen=True)
class YUV6SublatticeReport:
    """Typed result from :meth:`YUV6SublatticeGeometry.compute`.

    Attributes
    ----------
    rgb_shape : tuple[int, int, int]
        Input RGB shape ``(C, H, W)``.
    yuv6_shape : tuple[int, int, int]
        Output YUV6 shape ``(6, H // 2, W // 2)``.
    per_sublattice_energy : tuple[float, float, float, float]
        L2 energy of each of the 4 luma sublattices (Y00, Y10, Y01, Y11).
    u_sublattice_energy : float
        L2 energy of the U_sub channel.
    v_sublattice_energy : float
        L2 energy of the V_sub channel.
    per_sublattice_fraction : tuple[float, ...]
        Fraction of total luma energy in each of the 4 luma sublattices
        (sum is 1.0 modulo numerical precision).
    """

    rgb_shape: tuple[int, int, int]
    yuv6_shape: tuple[int, int, int]
    per_sublattice_energy: tuple[float, float, float, float]
    u_sublattice_energy: float
    v_sublattice_energy: float
    per_sublattice_fraction: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.rgb_shape) != 3:
            raise ValueError(
                f"rgb_shape must be 3-tuple; got {self.rgb_shape}"
            )
        if len(self.yuv6_shape) != 3:
            raise ValueError(
                f"yuv6_shape must be 3-tuple; got {self.yuv6_shape}"
            )
        if any(e < 0 for e in self.per_sublattice_energy):
            raise ValueError("per_sublattice_energy must be non-negative")
        if len(self.per_sublattice_fraction) != 4:
            raise ValueError("per_sublattice_fraction must have length 4")


class YUV6SublatticeGeometry:
    """F10 canonical primitive: YUV6 sublattice geometry analyzer."""

    @property
    def name(self) -> str:
        return "yuv6_sublattice_geometry"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return ("sensitivity_map", "bit_allocator")

    def compute(
        self,
        target: torch.Tensor,
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Decompose ``target`` RGB frame into YUV6 sublattices.

        Parameters
        ----------
        target : torch.Tensor
            RGB frame of shape ``(3, H, W)`` or ``(N, 3, H, W)`` (batch).
            For batch input, only the first frame is analyzed (callers
            wanting per-frame analysis should iterate themselves).
        """
        if target.dim() == 3:
            x = target
        elif target.dim() == 4:
            x = target[0]
        else:
            raise ValueError(
                f"target must be (3, H, W) or (N, 3, H, W); got shape "
                f"{tuple(target.shape)}"
            )
        if x.shape[0] != 3:
            raise ValueError(
                f"first dim must be 3 (RGB channels); got {x.shape[0]}"
            )
        c, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            # Crop to even-divisible size.
            h2 = (h // 2) * 2
            w2 = (w // 2) * 2
            x = x[:, :h2, :w2]
            h, w = h2, w2

        x = x.float()
        r, g, b = x[0], x[1], x[2]
        y = (
            BT601_R[0] * r + BT601_R[1] * g + BT601_R[2] * b
        )
        u = (
            BT601_U[0] * r + BT601_U[1] * g + BT601_U[2] * b
        )
        v = (
            BT601_V[0] * r + BT601_V[1] * g + BT601_V[2] * b
        )

        # Decompose Y into 4 sublattices by (y%2, x%2).
        sublattices: list[torch.Tensor] = []
        for sy, sx in LUMA_SUBLATTICE_INDICES:
            sublattices.append(y[sy::2, sx::2])
        # 2x2-averaged U and V (block average over 2x2).
        u_blocks = u.reshape(h // 2, 2, w // 2, 2)
        v_blocks = v.reshape(h // 2, 2, w // 2, 2)
        u_sub = u_blocks.mean(dim=(1, 3))
        v_sub = v_blocks.mean(dim=(1, 3))

        per_sub_energy = tuple(
            float((sub**2).sum().item()) for sub in sublattices
        )
        u_energy = float((u_sub**2).sum().item())
        v_energy = float((v_sub**2).sum().item())
        total_luma = sum(per_sub_energy)
        if total_luma > 0:
            per_sub_fraction = tuple(
                e / total_luma for e in per_sub_energy
            )
        else:
            per_sub_fraction = (0.25, 0.25, 0.25, 0.25)

        report = YUV6SublatticeReport(
            rgb_shape=(c, h, w),
            yuv6_shape=(6, h // 2, w // 2),
            per_sublattice_energy=per_sub_energy,
            u_sublattice_energy=u_energy,
            v_sublattice_energy=v_energy,
            per_sublattice_fraction=per_sub_fraction,
        )

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=None,
            archive_sha256=None,
            primitive_value=report,
            evidence_grade="structural-code-contract",
            confidence_band=None,
            composes_with=(
                "bilinear_resize_nullspace",
                "segnet_margin_polytope",
            ),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "bt601_y_coefficients": BT601_R,
                "bt601_u_coefficients": BT601_U,
                "bt601_v_coefficients": BT601_V,
            },
        )

    def project_to_sublattice(
        self,
        rgb: torch.Tensor,
        sublattice_idx: int,
    ) -> torch.Tensor:
        """Project an RGB frame to a single luma sublattice (0..3).

        Returns a 2-D tensor of shape ``(H // 2, W // 2)`` containing the
        luma values at the sublattice positions.
        """
        if sublattice_idx not in (0, 1, 2, 3):
            raise ValueError(
                f"sublattice_idx must be 0..3; got {sublattice_idx}"
            )
        if rgb.dim() == 3:
            x = rgb
        elif rgb.dim() == 4:
            x = rgb[0]
        else:
            raise ValueError(
                f"rgb must be (3, H, W) or (N, 3, H, W); got shape "
                f"{tuple(rgb.shape)}"
            )
        x = x.float()
        r, g, bch = x[0], x[1], x[2]
        y = BT601_R[0] * r + BT601_R[1] * g + BT601_R[2] * bch
        sy, sx = LUMA_SUBLATTICE_INDICES[sublattice_idx]
        return y[sy::2, sx::2]

    def compose_with(self, other: Any) -> Any:
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = [
    "BT601_R",
    "BT601_U",
    "BT601_V",
    "LUMA_SUBLATTICE_INDICES",
    "YUV6SublatticeGeometry",
    "YUV6SublatticeReport",
]
