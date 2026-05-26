# SPDX-License-Identifier: MIT
"""coin_pp_implicit_neural_representation inflate runtime — contest raw-output contract.

Loads the COINPP1 archive, reconstructs the PyTorch coord-MLP base topology
from the stored state_dict, dequantizes per-pair int8 modulations, runs the
per-pair coord-MLP forward (modulated FiLM scale+shift on hidden layers),
and writes one raw-output ``.raw`` file per contest video (1200 frames of
874×1164 RGB per video).

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device (Catalog #1; CPU/CUDA only via canonical select_inflate_device).
NO MLX at inflate (runtime_dep_closure = torch + brotli only per HNeRV
parity L4).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤200 for
substrate-engineering lanes (coord-MLP forward + FiLM modulation +
sinusoidal positional encoding + sigmoid + bicubic upscale to camera HW +
uint8 cast).

Per Catalog #205 device selection uses canonical ``select_inflate_device``.

L0 SCAFFOLD scope: the PyTorch coord-MLP topology mirrors the MLX module at
``tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer``. The
state_dict key contract is established at MLX-train-time + transferred via
the canonical export bridge per Path 3 cascade.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.substrates._shared.inflate_runtime import (
    raw_output_path,
    select_inflate_device,
)
from tac.substrates.coin_pp_implicit_neural_representation.archive import parse_archive

# Camera resolution required by upstream/evaluate.py contest harness.
CAMERA_H: int = 874
CAMERA_W: int = 1164


def _sinusoidal_positional_encoding(coords: torch.Tensor, pos_dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding per NeRF (Mildenhall 2020) + COIN++.

    Args:
        coords: shape (..., D=3) where D = (x, y, t)
        pos_dim: L; encoding output dim = L * 2 * D
    Returns:
        Encoded coords shape (..., L * 2 * D).
    """
    D = coords.shape[-1]
    L = int(pos_dim)
    freq_bands = (2.0 ** torch.arange(L, dtype=coords.dtype, device=coords.device)) * math.pi
    coords_expanded = coords.unsqueeze(-1)  # (..., D, 1)
    scaled = coords_expanded * freq_bands  # (..., D, L)
    sin_vals = torch.sin(scaled)
    cos_vals = torch.cos(scaled)
    stacked = torch.stack([sin_vals, cos_vals], dim=-1)  # (..., D, L, 2)
    return stacked.reshape(*coords.shape[:-1], D * L * 2)


class CoinPPCoordMLPTorch(nn.Module):
    """PyTorch inflate-time meta-learned modulated coord-MLP.

    Topology mirrors MLX renderer:
        positional_encoding -> input_proj -> hidden_0 -> FiLM_0 -> sin ->
        hidden_1 -> FiLM_1 -> sin -> ... -> output_proj -> sigmoid -> rgb
    """

    def __init__(
        self,
        *,
        mod_dim: int,
        pos_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        self.mod_dim = int(mod_dim)
        self.pos_dim = int(pos_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)

        pos_enc_dim = self.pos_dim * 2 * 3
        self.input_proj = nn.Linear(pos_enc_dim, self.hidden_dim)
        self.hidden = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_hidden_layers)]
        )
        self.film = nn.ModuleList(
            [nn.Linear(self.mod_dim, 2 * self.hidden_dim) for _ in range(self.num_hidden_layers)]
        )
        self.output_proj = nn.Linear(self.hidden_dim, 3)

    def forward(self, coords: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        """Decode (B, H, W, 3) coords + (B, mod_dim) modulation -> (B, H, W, 3) RGB."""
        # Positional encoding -> (B, H, W, pos_enc_dim)
        pos_enc = _sinusoidal_positional_encoding(coords, self.pos_dim)
        # Input projection
        h = self.input_proj(pos_enc)
        # FiLM modulation per hidden layer
        mod_b = modulation.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, mod_dim)
        for i in range(self.num_hidden_layers):
            h = self.hidden[i](h)
            film_out = self.film[i](mod_b)  # (B, 1, 1, 2*hidden_dim)
            scale = film_out[..., : self.hidden_dim] + 1.0  # +1 for identity-init
            shift = film_out[..., self.hidden_dim :]
            h = h * scale + shift
            h = torch.sin(h)
        rgb_logits = self.output_proj(h)
        return torch.sigmoid(rgb_logits)


def _build_coord_grid(H: int, W: int, t_value: float, device: torch.device) -> torch.Tensor:
    """Build (H, W, 3) coord grid; x/y in [-1, 1], t passed through."""
    x = torch.linspace(-1.0, 1.0, W, device=device) if W > 1 else torch.zeros(1, device=device)
    y = torch.linspace(-1.0, 1.0, H, device=device) if H > 1 else torch.zeros(1, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    tt = torch.full((H, W), float(t_value), device=device)
    return torch.stack([xx, yy, tt], dim=-1)


def inflate_one_video(
    archive_bytes: bytes, output_raw_path: Path, *, device: str | None = None
) -> int:
    """Inflate one COINPP1 archive's bytes into one contest ``.raw`` file."""
    arc = parse_archive(archive_bytes)
    render_device = torch.device(select_inflate_device(device))
    modulation_scale = float(arc.meta.get("modulation_scale", 1.0))

    decoder = CoinPPCoordMLPTorch(
        mod_dim=arc.mod_dim,
        pos_dim=arc.pos_dim,
        hidden_dim=arc.hidden_dim,
        num_hidden_layers=arc.num_hidden_layers,
    ).to(render_device)

    # Load state_dict (numpy fp16 arrays -> torch tensors).
    torch_sd: dict[str, torch.Tensor] = {}
    for key, np_arr in arc.base_state_dict.items():
        torch_sd[key] = torch.from_numpy(np_arr.astype("float32"))
    load_result = decoder.load_state_dict(torch_sd, strict=False)
    if set(load_result.missing_keys) or set(load_result.unexpected_keys):
        raise RuntimeError(
            "COINPP1 decoder state_dict mismatch: "
            f"missing={sorted(load_result.missing_keys)} "
            f"unexpected={sorted(load_result.unexpected_keys)}"
        )
    decoder.eval()

    # Dequantize per-pair modulations: int8 in [-128, 127] -> fp32 in [-mod_scale, +mod_scale]
    modulations_fp = torch.from_numpy(
        arc.per_pair_modulations.astype("float32") * (modulation_scale / 127.0)
    ).to(device=render_device)

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    coord_grid_t0 = _build_coord_grid(arc.eval_h, arc.eval_w, -1.0, render_device)
    coord_grid_t1 = _build_coord_grid(arc.eval_h, arc.eval_w, 1.0, render_device)

    n = 0
    with torch.inference_mode(), open(output_raw_path, "wb") as fout:
        for i in range(0, arc.num_pairs, 8):
            j = min(i + 8, arc.num_pairs)
            batch_mod = modulations_fp[i:j]  # (B, mod_dim)
            B = batch_mod.shape[0]
            grid_t0 = coord_grid_t0.unsqueeze(0).expand(B, -1, -1, -1)
            grid_t1 = coord_grid_t1.unsqueeze(0).expand(B, -1, -1, -1)
            rgb_t0 = decoder(grid_t0, batch_mod)  # (B, H, W, 3) in [0, 1]
            rgb_t1 = decoder(grid_t1, batch_mod)
            # NHWC -> NCHW for F.interpolate
            rgb_t0_nchw = rgb_t0.permute(0, 3, 1, 2) * 255.0
            rgb_t1_nchw = rgb_t1.permute(0, 3, 1, 2) * 255.0
            frames = torch.cat([rgb_t0_nchw, rgb_t1_nchw], dim=0)  # (2B, 3, H, W)
            up = F.interpolate(
                frames, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
            )
            frames_u8 = (
                up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            )
            fout.write(frames_u8.tobytes())
            n += int(frames_u8.shape[0])
    return n


def _read_single_member_archive_bytes(archive_dir: Path) -> bytes:
    """Read the single contest archive member, failing on ambiguity."""
    zero_bin = archive_dir / "0.bin"
    x_member = archive_dir / "x"
    present = [p for p in (zero_bin, x_member) if p.is_file()]
    if len(present) != 1:
        if not present:
            raise FileNotFoundError(
                f"expected exactly one archive member at {zero_bin} or {x_member}"
            )
        raise ValueError(f"ambiguous archive members present: {zero_bin} and {x_member}")
    return present[0].read_bytes()


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>`` per Catalog #146."""
    if len(sys.argv) < 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    file_list = file_list_path.read_text(encoding="utf-8").strip().splitlines()
    archive_bytes = _read_single_member_archive_bytes(archive_dir)
    device = select_inflate_device()
    for fname in file_list:
        name = fname.strip()
        if not name:
            continue
        inflate_one_video(archive_bytes, raw_output_path(output_dir, name), device=device)
    return 0


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "CoinPPCoordMLPTorch",
    "_read_single_member_archive_bytes",
    "inflate_one_video",
    "main_cli",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
