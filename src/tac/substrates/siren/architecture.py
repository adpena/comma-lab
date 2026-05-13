"""siren architecture — sinusoidal-init coordinate-based MLP.

L0 SKETCH scaffold per operator approval 2026-05-12. SIREN (Sitzmann et al.,
NeurIPS 2020) is a coordinate-based MLP with **sin activations** and a special
initialization scheme. The first layer uses ``Uniform(-1/fan_in, 1/fan_in) * omega``
with omega=30; downstream layers use ``Uniform(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega)``
with omega=1.0. This initialization preserves the input distribution through
multiple sin activations.

Architecture (council-sketch 2026-05-12; not yet empirical-anchored):

    Coordinate input: (x, y, t)
       x in [-1, 1] normalized horizontal
       y in [-1, 1] normalized vertical
       t = pair_index / num_pairs in [0, 1)
       |
       v
    Frequency encoding (optional but standard for spatial coordinates):
       (x, y, t) -> (sin(2^k * pi * x), cos(2^k * pi * x), ...) for k=0..L
       OR just direct (x, y, t) — first-layer omega=30 handles encoding
       |
       v
    Hidden 1: Linear(3 -> H) -> sin(omega=30 * x)
    Hidden 2: Linear(H -> H) -> sin(omega=1.0 * x)
    ...
    Hidden N: Linear(H -> H) -> sin(omega=1.0 * x)
       |
       v
    Output: Linear(H -> 3) -> sigmoid -> RGB

Council notes:
- Total param target: ~150K (matches Sitzmann's ImageINR experiments at scale)
- Hidden width=128, depth=6 layers gives ~131K params + ~390 output = ~131K
- Hotz's "raw engineering instinct" — analytical-not-learned per the council seat
- EMA decay 0.997 per CLAUDE.md "EMA — non-negotiable" (applied externally)

CLAUDE.md compliance:
- No silent device defaults (caller passes device)
- No scorer loading inside this module
- No /tmp paths
- Reviewable in 30 seconds per L12
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .activation_family import (
    DEFAULT_ACTIVATION_FAMILY,
    ActivationFamilyId,
    apply_activation_family,
    effective_layer_omega,
    normalize_activation_family,
)

_CONTEST_H = 384
_CONTEST_W = 512
_NUM_FRAMES = 1200
_PAIRS = _NUM_FRAMES // 2  # 600 pairs


@dataclass(frozen=True)
class SirenConfig:
    """Static design-time parameters for siren (L0 SKETCH)."""

    hidden_dim: int = 128
    """Width of each MLP hidden layer."""

    num_hidden_layers: int = 6
    """Number of hidden Linear+sin layers (output is a separate Linear)."""

    first_omega: float = 30.0
    """omega_0 for the first layer (Sitzmann signature; NeRF default)."""

    hidden_omega: float = 1.0
    """omega for downstream layers (Sitzmann standard)."""

    num_pairs: int = _PAIRS
    """Number of latent rows (600 for 1200-frame contest video)."""

    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W

    coord_dim: int = 3
    """Input coordinate dimensionality: (x, y, t) — t = pair_index/num_pairs."""

    output_dim: int = 6
    """Output dim: 6 = (rgb_0, rgb_1) concatenated for the pair."""

    activation_family: ActivationFamilyId = DEFAULT_ACTIVATION_FAMILY
    """INR activation family serialized in SRV1 metadata."""

    wire_scale: float = 1.0
    """Positive Gabor/window scale used by the WIRE-style activation."""

    bacon_bandwidth_scale: float = 1.0
    """Positive multiplier for the BACON-style per-layer omega schedule."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "activation_family",
            normalize_activation_family(str(self.activation_family)),
        )
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.first_omega <= 0.0 or self.hidden_omega <= 0.0:
            raise ValueError("omega values must be positive")
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output dimensions must be positive")
        if self.coord_dim != 3:
            raise ValueError("coord_dim must be 3 for (x, y, t)")
        if self.output_dim != 6:
            raise ValueError("output_dim must be 6 for paired RGB output")
        if self.wire_scale <= 0.0:
            raise ValueError("wire_scale must be positive")
        if self.bacon_bandwidth_scale <= 0.0:
            raise ValueError("bacon_bandwidth_scale must be positive")


class _ActivationLayer(nn.Module):
    """Linear + INR activation with SIREN-compatible initialization."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega: float,
        *,
        activation_family: ActivationFamilyId,
        is_first: bool = False,
        layer_index: int = 0,
        wire_scale: float = 1.0,
        bacon_bandwidth_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation_family = normalize_activation_family(activation_family)
        self.init_omega = float(omega)
        self.omega = effective_layer_omega(
            activation_family=self.activation_family,
            base_omega=self.init_omega,
            layer_index=int(layer_index),
            bacon_bandwidth_scale=float(bacon_bandwidth_scale),
        )
        self.wire_scale = float(wire_scale)
        with torch.no_grad():
            if is_first:
                # First layer: Uniform(-1/fan_in, 1/fan_in) (then scaled by omega in forward)
                self.linear.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
            else:
                # Downstream: Uniform(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega)
                bound = math.sqrt(6.0 / in_dim) / max(self.init_omega, 1.0)
                self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_activation_family(
            self.linear(x),
            activation_family=self.activation_family,
            omega=self.omega,
            wire_scale=self.wire_scale,
        )


class SirenSubstrate(nn.Module):
    """SIREN substrate: coordinate-MLP with sin activations.

    Forward signature mirrors sane_hnerv for trainer interop:
        forward(pair_indices) -> (rgb_0, rgb_1), each (B, 3, H, W).

    Internally the model evaluates the MLP at every (x, y, t) in a meshgrid
    for each pair index, producing the full per-pair frame pair in one call.
    """

    def __init__(self, cfg: SirenConfig) -> None:
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        # First layer
        layers.append(
            _ActivationLayer(
                cfg.coord_dim,
                cfg.hidden_dim,
                cfg.first_omega,
                activation_family=cfg.activation_family,
                is_first=True,
                layer_index=0,
                wire_scale=cfg.wire_scale,
                bacon_bandwidth_scale=cfg.bacon_bandwidth_scale,
            )
        )
        # Hidden layers
        for layer_index in range(1, cfg.num_hidden_layers):
            layers.append(
                _ActivationLayer(
                    cfg.hidden_dim,
                    cfg.hidden_dim,
                    cfg.hidden_omega,
                    activation_family=cfg.activation_family,
                    layer_index=layer_index,
                    wire_scale=cfg.wire_scale,
                    bacon_bandwidth_scale=cfg.bacon_bandwidth_scale,
                )
            )
        self.hidden = nn.Sequential(*layers)

        # Output layer (no sin): Linear -> sigmoid in forward
        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / cfg.hidden_dim) / max(cfg.hidden_omega, 1.0)
            self.output_layer.weight.uniform_(-bound, bound)
            self.output_layer.bias.zero_()

        # Pre-cache the spatial meshgrid as a buffer (NOT a parameter)
        ys = torch.linspace(-1.0, 1.0, cfg.output_height)
        xs = torch.linspace(-1.0, 1.0, cfg.output_width)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each
        spatial = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        self.register_buffer("_spatial_coords", spatial)

    def forward(self, pair_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Render frame-pairs at the given pair indices.

        Args:
            pair_indices: ``(B,)`` long tensor in ``[0, num_pairs)``.

        Returns:
            ``(rgb_0, rgb_1)``, each ``(B, 3, H, W)`` in ``[0, 1]``.
        """
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(f"pair_indices out of range [0, {self.cfg.num_pairs})")

        batch = pair_indices.shape[0]
        h, w = self.cfg.output_height, self.cfg.output_width

        # Build (B, H, W, 3) coordinate tensor
        spatial = self._spatial_coords.to(pair_indices.device)  # (H, W, 2)
        spatial = spatial.unsqueeze(0).expand(batch, -1, -1, -1)  # (B, H, W, 2)
        # t in [0, 1)
        t = (pair_indices.float() / float(self.cfg.num_pairs)).view(batch, 1, 1, 1)
        t = t.expand(-1, h, w, 1)  # (B, H, W, 1)
        coords = torch.cat([spatial, t], dim=-1)  # (B, H, W, 3)

        # Flatten to (B*H*W, 3) for MLP
        flat = coords.reshape(-1, self.cfg.coord_dim)
        out = self.hidden(flat)
        out = self.output_layer(out)
        out = torch.sigmoid(out)  # (B*H*W, 6)

        # Reshape: (B, H, W, 6) -> split to (B, 3, H, W) x 2
        out = out.view(batch, h, w, self.cfg.output_dim)
        rgb_0 = out[..., :3].permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
        rgb_1 = out[..., 3:6].permute(0, 3, 1, 2).contiguous()
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def runtime_state_dict_for_archive(self) -> dict[str, torch.Tensor]:
        """Return only trainable MLP tensors for the SRV1 archive.

        ``_spatial_coords`` is a deterministic meshgrid buffer. It is rebuilt
        by ``SirenSubstrate(cfg)`` at inflate time and must not consume archive
        bytes.
        """

        return {
            name: tensor.detach().clone()
            for name, tensor in self.state_dict().items()
            if name != "_spatial_coords"
        }
