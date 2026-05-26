# SPDX-License-Identifier: MIT
"""nirvana_cascading_nerv inflate runtime — contest raw-output contract.

Loads the NIRVANA1 archive, reconstructs the PyTorch hierarchical residual
cascade decoder topology from the stored state_dict, dequantizes per-level
int8 residuals, runs the cascade reconstruction (level0 → upsample →
+level1_residual → ... → final_RGB), and writes one raw-output ``.raw``
file per contest video (1200 frames of 874×1164 RGB per video).

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device (Catalog #1; CPU/CUDA only via canonical select_inflate_device).
NO MLX at inflate (runtime_dep_closure = torch + brotli only per HNeRV
parity L4).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤200 for
substrate-engineering lanes (per-level decoder forward + int8 dequant +
bilinear upsample + residual add + bicubic upscale → camera HW + uint8 cast).

Per Catalog #205 device selection uses canonical ``select_inflate_device``.

L0 SCAFFOLD scope: the PyTorch decoder topology mirrors the MLX module at
``tac.substrates.nirvana_cascading_nerv.mlx_renderer``. The state_dict key
contract is established at MLX-train-time + transferred via the canonical
export bridge per Path 3 cascade.

L0 caveat: this scaffold defines the inflate contract; a paired-empirical
proof via the canonical PR95 #1265 contest-equivalence gate is the next
operator-routable step per the symposium 6-tier dispatch ladder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.substrates._shared.inflate_runtime import (
    raw_output_path,
    select_inflate_device,
)
from tac.substrates.nirvana_cascading_nerv.archive import parse_archive

# Camera resolution required by upstream/evaluate.py contest harness.
CAMERA_H: int = 874
CAMERA_W: int = 1164

# Contest scorer-resolution (matches MLX module EVAL_HW).
DECODER_H_FINAL: int = 384
DECODER_W_FINAL: int = 512


class _NirvanaLevelDecoder(nn.Module):
    """Per-level decoder mirroring MLX module's level decoder.

    Each level: linear (latent → base_channels * h * w) → reshape → sin →
    conv2d (base_channels → base_channels) → sin → conv2d → 3 RGB channels.
    """

    def __init__(
        self,
        *,
        per_pair_latent_dim: int,
        base_channels: int,
        level_h: int,
        level_w: int,
    ) -> None:
        super().__init__()
        self.per_pair_latent_dim = int(per_pair_latent_dim)
        self.base_channels = int(base_channels)
        self.level_h = int(level_h)
        self.level_w = int(level_w)

        self.stem = nn.Linear(
            self.per_pair_latent_dim,
            self.base_channels * self.level_h * self.level_w,
        )
        self.conv1 = nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1)
        self.conv_to_rgb = nn.Conv2d(self.base_channels, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode (B, latent_dim) → (B, 3, H, W) RGB in [0, 1]."""
        B = int(z.shape[0])
        x = self.stem(z).view(B, self.base_channels, self.level_h, self.level_w)
        x = torch.sin(x)
        x = self.conv1(x)
        x = torch.sin(x)
        rgb = torch.sigmoid(self.conv_to_rgb(x))
        return rgb


class NirvanaCascadingDecoderTorch(nn.Module):
    """PyTorch inflate-time hierarchical residual decoder cascade."""

    def __init__(
        self,
        *,
        num_levels: int,
        per_pair_latent_dim: int,
        base_h: int,
        base_w: int,
        base_channels: int,
    ) -> None:
        super().__init__()
        self.num_levels = int(num_levels)
        self.per_pair_latent_dim = int(per_pair_latent_dim)
        self.base_channels = int(base_channels)
        # Only level 0 has a full per-pair decoder; higher levels apply
        # int8-residual additions from the archive
        self.level_0_decoder = _NirvanaLevelDecoder(
            per_pair_latent_dim=per_pair_latent_dim,
            base_channels=base_channels,
            level_h=base_h,
            level_w=base_w,
        )

    def forward(
        self,
        latents: torch.Tensor,
        per_level_residuals_fp: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run cascade: level0 → +residual_1 → ... → final RGB at EVAL_HW.

        Args:
            latents: (B, latent_dim) per-pair latents
            per_level_residuals_fp: list of (H_i, W_i, 3) fp32 residuals
                already dequantized; one per level (level 0 is unused per
                cascade design; levels 1..num_levels-1 carry residuals).

        Returns:
            (B, 3, H_final, W_final) RGB in [0, 1].
        """
        # Level 0: full per-pair decoder
        rgb = self.level_0_decoder(latents)  # (B, 3, base_h, base_w)
        # Levels 1..N-1: upsample + add per-level residual
        for level in range(1, self.num_levels):
            # Upsample ×2 (bilinear, align_corners=False per axis 2 drift discipline)
            rgb = F.interpolate(
                rgb, scale_factor=2, mode="bilinear", align_corners=False
            )
            # Add per-level residual (broadcast across batch)
            # residual_fp shape: (H, W, 3); permute to (1, 3, H, W) for broadcast
            residual = per_level_residuals_fp[level].permute(2, 0, 1).unsqueeze(0)
            rgb = rgb + residual
            rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb


def _dequantize_residuals(
    int8_residuals: list, residual_scale: float
) -> list[torch.Tensor]:
    """Dequantize per-level int8 residuals to fp32 in [-residual_scale, +residual_scale].

    Args:
        int8_residuals: list of (H, W, 3) numpy int8 arrays
        residual_scale: max abs value used at quantization time

    Returns:
        List of (H, W, 3) fp32 torch tensors
    """
    out: list[torch.Tensor] = []
    for arr in int8_residuals:
        # int8 in [-128, 127] → fp32 in [-residual_scale, +residual_scale]
        fp32 = torch.from_numpy(arr.astype("float32")) * (residual_scale / 127.0)
        out.append(fp32)
    return out


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one NIRVANA1 archive's bytes into one contest ``.raw`` file."""
    arc = parse_archive(archive_bytes)
    render_device = torch.device(select_inflate_device(device))

    base_channels = int(arc.meta.get("base_channels", 24))
    residual_scale = float(arc.meta.get("residual_scale", 0.5))
    num_pairs = int(arc.meta.get("num_pairs", 600))

    decoder = NirvanaCascadingDecoderTorch(
        num_levels=arc.num_levels,
        per_pair_latent_dim=arc.per_pair_latent_dim,
        base_h=arc.base_h,
        base_w=arc.base_w,
        base_channels=base_channels,
    ).to(render_device)

    # Load state_dict (numpy fp16 arrays from archive → torch tensors).
    # MLX stores conv weights in NHWC layout (out_ch, kH, kW, in_ch); PyTorch
    # expects NCHW (out_ch, in_ch, kH, kW). Transpose 4-D weight tensors.
    torch_sd: dict[str, torch.Tensor] = {}
    for key, np_arr in arc.decoder_state_dict.items():
        arr_fp32 = np_arr.astype("float32")
        if arr_fp32.ndim == 4:
            arr_fp32 = arr_fp32.transpose(0, 3, 1, 2).copy()
        torch_sd[key] = torch.from_numpy(arr_fp32)

    load_result = decoder.load_state_dict(torch_sd, strict=False)
    if set(load_result.missing_keys) or set(load_result.unexpected_keys):
        raise RuntimeError(
            "NIRVANA1 decoder state_dict mismatch: "
            f"missing={sorted(load_result.missing_keys)} "
            f"unexpected={sorted(load_result.unexpected_keys)}"
        )
    decoder.eval()

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)

    # Dequantize latents: int16 in [-32768, 32767] → fp32 in [-1, 1]
    latents_fp = torch.from_numpy(arc.per_pair_latents.astype("float32") / 32767.0).to(
        device=render_device
    )

    # Dequantize per-level residuals to fp32 (host-side numpy → torch)
    residuals_fp_host = _dequantize_residuals(
        arc.per_level_residuals, residual_scale=residual_scale
    )
    residuals_fp = [r.to(render_device) for r in residuals_fp_host]

    n = 0
    with torch.inference_mode(), open(output_raw_path, "wb") as fout:
        for i in range(0, num_pairs, 16):
            j = min(i + 16, num_pairs)
            batch_latents = latents_fp[i:j]  # (B, latent_dim)
            decoded = decoder(batch_latents, residuals_fp)  # (B, 3, H_final, W_final)
            decoded = decoded * 255.0  # to [0, 255]
            B = decoded.shape[0]
            # Each pair produces 2 frames (per contest contract; pair = consecutive 2)
            # The substrate currently emits one RGB per pair; duplicate it for
            # contest's 2-frames-per-pair contract (canonical sister substrate pattern).
            frames = torch.cat([decoded, decoded], dim=0)  # (2B, 3, H, W)
            up = F.interpolate(
                frames,
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            frames_u8 = (
                up.clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            fout.write(frames_u8.tobytes())
            n += int(frames_u8.shape[0])
    return n


def _read_single_member_archive_bytes(archive_dir: Path) -> bytes:
    """Read the single contest archive member, failing on ambiguity."""
    zero_bin = archive_dir / "0.bin"
    x_member = archive_dir / "x"
    present = [path for path in (zero_bin, x_member) if path.is_file()]
    if len(present) != 1:
        if not present:
            raise FileNotFoundError(
                f"expected exactly one archive member at {zero_bin} or {x_member}"
            )
        raise ValueError(
            f"ambiguous archive members present: {zero_bin} and {x_member}"
        )
    return present[0].read_bytes()


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>`` per Catalog #146."""
    if len(sys.argv) < 4:
        print(
            "usage: inflate.py <archive_dir> <output_dir> <file_list>",
            file=sys.stderr,
        )
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
        inflate_one_video(
            archive_bytes, raw_output_path(output_dir, name), device=device
        )
    return 0


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "DECODER_H_FINAL",
    "DECODER_W_FINAL",
    "NirvanaCascadingDecoderTorch",
    "_read_single_member_archive_bytes",
    "inflate_one_video",
    "main_cli",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
