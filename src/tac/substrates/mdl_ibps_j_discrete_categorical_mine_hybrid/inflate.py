# SPDX-License-Identifier: MIT
"""inflate — PyTorch runtime for J=MDL-IBPS DISCRETE-CATEGORICAL-MINE-HYBRID.

Catalog #146 (contest-compliant inflate runtime template) + Catalog #205
(canonical select_inflate_device helper) + Catalog #295 (PYTHONPATH
self-containment) compliance scaffold.

This is the SUBSTRATE-INTERNAL inflate logic. The contest-submission
``submissions/<lane>/inflate.py`` will vendor this module + a thin shim
that wires the contest CLI signature (``inflate.sh <archive_dir>
<output_dir> <file_list>``) per Catalog #146.

The L0 SCAFFOLD's submission inflate.py is DEFERRED until Phase 3
follow-on (after MLX-first smoke + Stage 4 Tier-C verdict). This module
is the substrate-level renderer wired into the trainer's export path.

Per CLAUDE.md FORBIDDEN_PATTERNS:
- No silent device defaults; uses canonical ``select_inflate_device`` per Catalog #205
- No scorer load at inflate time (only procedural coord-MLP + FiLM + sigmoid)
- No /tmp paths in persisted artifacts
- File reviewable in 30 seconds per HNeRV parity L12

Sister inflate runtimes:
- ``tac.substrates.c6_e4_mdl_ibps.inflate`` (parent C6 reference)
- ``tac.substrates.coin_pp_implicit_neural_representation.inflate`` (sister K=COIN++)
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from tac.substrates._shared.inflate_runtime import select_inflate_device
from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid import (
    CATEGORICAL_G,
    CATEGORICAL_K,
    EVAL_HW,
    HIDDEN_DIM,
    NUM_HIDDEN_LAYERS,
    NUM_PAIRS,
    POS_DIM,
)


def _sinusoidal_encoding_torch(coords: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positional encoding (NeRF-style) - PyTorch impl."""
    freqs = (2.0 ** torch.arange(POS_DIM, dtype=torch.float32)) * math.pi
    freqs = freqs.to(coords.device)
    scaled = coords.float()[:, :, None] * freqs[None, None, :]
    sins = torch.sin(scaled)
    coss = torch.cos(scaled)
    encoded = torch.cat([sins, coss], dim=-1)
    return encoded.reshape(coords.shape[0], -1)


class CoordMLPBaseTorch(nn.Module):
    """Procedural coord-MLP base - PyTorch reference for inflate."""

    def __init__(self) -> None:
        super().__init__()
        pos_feat_dim = POS_DIM * 2 * 3
        self.linear_first = nn.Linear(pos_feat_dim, HIDDEN_DIM)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(NUM_HIDDEN_LAYERS - 1)]
        )
        self.linear_out = nn.Linear(HIDDEN_DIM, 3)

    def forward(
        self,
        coords: torch.Tensor,
        film_scales: torch.Tensor,
        film_shifts: torch.Tensor,
    ) -> torch.Tensor:
        x = _sinusoidal_encoding_torch(coords)
        h = self.linear_first(x)
        for layer_idx in range(NUM_HIDDEN_LAYERS):
            scale = film_scales[:, layer_idx, :]
            shift = film_shifts[:, layer_idx, :]
            h = h * scale + shift
            h = torch.sin(h)
            if layer_idx < NUM_HIDDEN_LAYERS - 1:
                h = self.hidden_layers[layer_idx](h)
        rgb = torch.sigmoid(self.linear_out(h))
        return rgb


class FilmProjTorch(nn.Module):
    """FiLM projection: one_hot -> (scales, shifts) - PyTorch reference."""

    def __init__(self) -> None:
        super().__init__()
        in_dim = CATEGORICAL_G * CATEGORICAL_K
        out_dim = NUM_HIDDEN_LAYERS * HIDDEN_DIM * 2
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, one_hot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        proj = self.linear(one_hot)
        B = one_hot.shape[0]
        proj = proj.view(B, NUM_HIDDEN_LAYERS, HIDDEN_DIM, 2)
        return proj[..., 0], proj[..., 1]


def categorical_to_one_hot_torch(
    indices: torch.Tensor, K: int = CATEGORICAL_K
) -> torch.Tensor:
    """Categorical index -> flattened one-hot (B, G*K)."""
    if indices.dim() != 2:
        raise ValueError(f"indices must be 2D (B, G); got shape {indices.shape}")
    B, G = indices.shape
    one_hot = torch.zeros(B, G, K, dtype=torch.float32, device=indices.device)
    one_hot.scatter_(2, indices.long().unsqueeze(-1), 1.0)
    return one_hot.view(B, G * K)


def _make_pixel_coords_torch(
    height: int, width: int, t: int, device: torch.device
) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=torch.float32)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_t = torch.full_like(grid_x, float(t))
    return torch.stack(
        [grid_x.flatten(), grid_y.flatten(), grid_t.flatten()], dim=-1
    )


def inflate_substrate(
    base_blob_state_dict: dict[str, Any],
    indices: list[list[int]],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Inflate J=MDL-IBPS archive bytes -> rendered (NUM_PAIRS, 2, 3, H, W) tensor.

    Per Catalog #205: device defaults to canonical ``select_inflate_device``
    output if not specified. NO silent device-selection ternary.

    Args:
        base_blob_state_dict: parsed state_dict from BASE_BLOB; expected keys:
            'film_proj.linear.weight', 'film_proj.linear.bias',
            'coord_mlp.linear_first.weight', 'coord_mlp.linear_first.bias',
            'coord_mlp.hidden_layers.{0..NUM_HIDDEN_LAYERS-2}.weight'/'.bias',
            'coord_mlp.linear_out.weight', 'coord_mlp.linear_out.bias'.
        indices: per-pair categorical indices (NUM_PAIRS x G).
        device: optional explicit device override; default routes through Catalog #205.

    Returns:
        rendered tensor ``(NUM_PAIRS, 2, 3, EVAL_HW[0], EVAL_HW[1])`` in [0, 1].
    """
    if device is None:
        device = select_inflate_device()  # Catalog #205 canonical
    if len(indices) != NUM_PAIRS:
        raise ValueError(f"expected {NUM_PAIRS} pair indices; got {len(indices)}")
    # Construct + load modules
    film_proj = FilmProjTorch().to(device).eval()
    coord_mlp = CoordMLPBaseTorch().to(device).eval()
    # Filter state_dict by prefix
    film_state = {
        k.removeprefix("film_proj."): v
        for k, v in base_blob_state_dict.items()
        if k.startswith("film_proj.")
    }
    coord_state = {
        k.removeprefix("coord_mlp."): v
        for k, v in base_blob_state_dict.items()
        if k.startswith("coord_mlp.")
    }
    film_proj.load_state_dict(film_state, strict=True)
    coord_mlp.load_state_dict(coord_state, strict=True)
    # Render each pair
    height, width = EVAL_HW
    rendered = torch.zeros(NUM_PAIRS, 2, 3, height, width, device=device)
    with torch.no_grad():
        for pair_idx in range(NUM_PAIRS):
            idx_tensor = torch.tensor(
                indices[pair_idx], dtype=torch.long, device=device
            ).unsqueeze(0)  # (1, G)
            one_hot = categorical_to_one_hot_torch(idx_tensor, K=CATEGORICAL_K)
            scales, shifts = film_proj(one_hot)  # (1, L, H)
            for t in (0, 1):
                coords = _make_pixel_coords_torch(height, width, t, device)
                N = coords.shape[0]
                scales_b = scales.expand(N, NUM_HIDDEN_LAYERS, HIDDEN_DIM)
                shifts_b = shifts.expand(N, NUM_HIDDEN_LAYERS, HIDDEN_DIM)
                rgb_flat = coord_mlp(coords, scales_b, shifts_b)  # (N, 3)
                rgb = rgb_flat.view(height, width, 3).permute(2, 0, 1)
                rendered[pair_idx, t] = rgb
    return rendered


__all__ = [
    "CoordMLPBaseTorch",
    "FilmProjTorch",
    "categorical_to_one_hot_torch",
    "inflate_substrate",
]
