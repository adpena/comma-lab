# SPDX-License-Identifier: MIT
"""Multi-control hint encoder for mask-conditioned renderers.

The encoder follows the ControlNet-style zero-initialized adapter pattern:
the final projection starts at exactly zero, so enabling the lane is a
baseline no-op until training moves the projection weights.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_HIDDEN_DIMS = (16, 16, 32, 32, 96, 96, 256)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero all parameters in a module and return it."""
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


class _LowRankPointwise(nn.Module):
    """A low-rank 1x1 MLP layer applied independently at each pixel."""

    def __init__(self, in_channels: int, out_channels: int, *, rank: int = 6):
        super().__init__()
        bottleneck = max(1, min(rank, in_channels, out_channels))
        self.reduce = nn.Conv2d(in_channels, bottleneck, 1, bias=True)
        self.project = nn.Conv2d(bottleneck, out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(F.silu(self.reduce(x)))


class MultiControlHintEncoder(nn.Module):
    """Encode 5-class SegNet masks into per-pixel renderer control hints.

    Args:
        in_channels: number of one-hot mask channels.
        hidden_dims: per-pixel MLP channel widths.
        out_channels: feature width emitted for renderer block adapters.
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dims: Sequence[int] | None = None,
        out_channels: int = 256,
    ):
        super().__init__()
        dims = tuple(hidden_dims or DEFAULT_HIDDEN_DIMS)
        if not dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        layers: list[nn.Module] = []
        prev = in_channels
        for dim in dims[:-1]:
            layers.append(_LowRankPointwise(prev, int(dim)))
            layers.append(nn.SiLU(inplace=True))
            prev = int(dim)

        final_hidden = int(dims[-1])
        if final_hidden != out_channels:
            layers.append(_LowRankPointwise(prev, final_hidden))
            layers.append(nn.SiLU(inplace=True))
            prev = final_hidden

        self.final_projection = _LowRankPointwise(prev, out_channels)
        zero_module(self.final_projection.project)
        layers.append(self.final_projection)

        self.net = nn.Sequential(*layers)
        self.control_weight_head = zero_module(nn.Conv2d(out_channels, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _as_one_hot(self, masks: torch.Tensor) -> torch.Tensor:
        if masks.ndim == 3:
            if not masks.dtype.is_floating_point:
                return F.one_hot(masks.long(), num_classes=self.in_channels).permute(0, 3, 1, 2).float()
            if self.in_channels != 1:
                raise ValueError(
                    "floating 3D masks are only valid when in_channels=1; "
                    f"got in_channels={self.in_channels}"
                )
            return masks.unsqueeze(1).float()

        if masks.ndim != 4:
            raise ValueError(
                "masks must have shape (B,H,W) integer labels or "
                f"(B,{self.in_channels},H,W) one-hot/soft masks; got {tuple(masks.shape)}"
            )
        if masks.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} mask channels, got {masks.shape[1]}"
            )
        return masks.float()

    def forward(
        self,
        masks: torch.Tensor,
        *,
        return_weight_map: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self._as_one_hot(masks)
        features = self.net(x)
        if not return_weight_map:
            return features
        weight = self.control_weight_head(features)
        return features, weight

    def control_weight_map(self, masks: torch.Tensor) -> torch.Tensor:
        """Return the per-pixel scalar modulation map."""
        _, weight = self.forward(masks, return_weight_map=True)
        return weight


class ControlSummationAdapter(nn.Module):
    """Add hint features to a renderer block without adding parameters."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, features: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        if hint.shape[-2:] != features.shape[-2:]:
            hint = F.interpolate(hint, size=features.shape[-2:], mode="bilinear", align_corners=False)
        if hint.shape[1] < self.out_channels:
            pad_ch = self.out_channels - hint.shape[1]
            hint = F.pad(hint, (0, 0, 0, 0, 0, pad_ch))
        else:
            hint = hint[:, : self.out_channels]
        return features + hint.to(dtype=features.dtype)
