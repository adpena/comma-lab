# SPDX-License-Identifier: MIT
"""Training-time bottleneck feature masking for Tuna-2 style lanes."""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureMasker(nn.Module):
    """Replace random spatial feature positions with a learnable mask token.

    Mask decisions are per batch item and spatial position, then broadcast
    across channels so a whole bottleneck vector is replaced at selected
    positions. Randomness is generated from ``training_progress`` plus a
    persisted forward counter, making the sequence deterministic under
    checkpoint resume when the module state is saved.
    """

    def __init__(
        self,
        channels: int,
        *,
        p: float = 0.5,
        mask_ratio: float = 0.15,
        apply_in_final_fraction: float = 0.4,
        seed: int = 1729,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = int(channels)
        self.p = float(p)
        self.mask_ratio = float(mask_ratio)
        self.apply_in_final_fraction = float(apply_in_final_fraction)
        self.seed = int(seed)
        self.mask_token = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        self.register_buffer(
            "_forward_counter",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        training_progress: float | torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"FeatureMasker expects BCHW features, got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"FeatureMasker channels={self.channels} but input has {x.shape[1]}"
            )
        if not self.training:
            return x

        progress = self._progress_float(training_progress)
        threshold = 1.0 - self.apply_in_final_fraction
        if progress < threshold or self.p <= 0.0 or self.mask_ratio <= 0.0:
            return x

        p = min(max(self.p, 0.0), 1.0)
        mask_ratio = min(max(self.mask_ratio, 0.0), 1.0)
        counter = int(self._forward_counter.item())
        with torch.no_grad():
            self._forward_counter.add_(1)

        rand_device = x.device if x.device.type in {"cpu", "cuda"} else torch.device("cpu")
        gen = torch.Generator(device=rand_device)
        gen.manual_seed(self._seed(progress, counter))

        apply_draw = torch.rand((), generator=gen, device=rand_device)
        if bool((apply_draw >= p).item()):
            return x

        b, _, h, w = x.shape
        mask = torch.rand((b, 1, h, w), generator=gen, device=rand_device) < mask_ratio
        if mask.device != x.device:
            mask = mask.to(device=x.device)
        token = self.mask_token.to(device=x.device, dtype=x.dtype)
        return torch.where(mask.expand_as(x), token.expand_as(x), x)

    @staticmethod
    def _progress_float(training_progress: float | torch.Tensor) -> float:
        if torch.is_tensor(training_progress):
            progress = float(training_progress.detach().cpu().item())
        else:
            progress = float(training_progress)
        return min(max(progress, 0.0), 1.0)

    def _seed(self, progress: float, counter: int) -> int:
        progress_key = int(round(progress * 1_000_000))
        return (self.seed + progress_key * 1_000_003 + counter * 9_176) % (2**63 - 1)
