# SPDX-License-Identifier: MIT
"""DreamerV3 RSSM inflate runtime — contest raw-output contract.

Loads the RSSMC1 archive, reconstructs the PyTorch decoder topology from the
stored state_dict, dequantizes per-pair category indices to one-hot, and
writes one raw-output ``.raw`` file per contest video (1200 frames of
874×1164 RGB per video).

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device (Catalog #1; CPU/CUDA only via canonical select_inflate_device).
NO MLX at inflate (runtime_dep_closure = torch + brotli only per HNeRV
parity L4).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤200 for
substrate-engineering lanes (categorical dequant + decoder forward + bilinear
upscale → camera HW + uint8 cast).

Per Catalog #205 device selection uses canonical ``select_inflate_device``.

L0 SCAFFOLD scope: the PyTorch decoder topology mirrors the MLX module at
``tac.substrates.dreamer_v3_rssm.module.DreamerV3RSSMSubstrateMLX._decoder_forward``.
The state_dict key contract is established at MLX-train-time + transferred
via the canonical export bridge per Path 3 cascade (sister #1251 + #1257).

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
from tac.substrates.dreamer_v3_rssm.archive import parse_archive

# Contest scorer-resolution (matches MLX module EVAL_HW; decoder native output).
DECODER_H: int = 384
DECODER_W: int = 512

# Camera resolution required by upstream/evaluate.py contest harness.
CAMERA_H: int = 874
CAMERA_W: int = 1164


class _RSSMUpsampleBlockTorch(nn.Module):
    """PyTorch mirror of MLX ``_RSSMUpsampleBlock`` (sin + PixelShuffle + skip)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1)
        if in_channels != out_channels:
            self.skip_conv: nn.Module = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        identity = self.skip_conv(identity)
        decoded = self.ps(self.conv(x))
        return torch.sin(decoded + identity)


class DreamerV3RSSMDecoderTorch(nn.Module):
    """PyTorch inflate-time decoder mirroring the MLX module's _decoder_forward.

    State_dict keys MUST match the MLX module's keys (cat_to_continuous,
    stem, blocks.0..5, refine0, refine1, rgb_0, rgb_1). Per-pair logits
    are NOT loaded here (they are reduced to argmax indices stored in the
    archive's indices_blob; inflate reconstructs one-hot from indices).
    """

    def __init__(
        self,
        *,
        num_groups: int,
        num_categories: int,
        decoder_latent_dim: int,
        base_channels: int,
    ) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_categories = int(num_categories)
        self.decoder_latent_dim = int(decoder_latent_dim)
        self.base_channels = int(base_channels)
        self.base_h, self.base_w = 6, 8

        self.cat_to_continuous = nn.Linear(
            self.num_groups * self.num_categories, self.decoder_latent_dim
        )

        C = self.base_channels
        channels = [C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]
        if min(channels) < 1:
            raise ValueError(f"base_channels={C} too small for PR95 channel taper")
        self.channels = channels

        self.stem = nn.Linear(
            self.decoder_latent_dim, channels[0] * self.base_h * self.base_w
        )
        self.blocks = nn.ModuleList(
            [_RSSMUpsampleBlockTorch(channels[i], channels[i + 1]) for i in range(6)]
        )
        final_ch = channels[-1]
        self.refine0 = nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2)
        self.refine1 = nn.Conv2d(final_ch // 2, final_ch, 3, padding=1)
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode (B, G) int category indices → (B, 2, 3, H, W) float RGB in [0, 255]."""
        B, G = int(indices.shape[0]), int(indices.shape[1])
        K = self.num_categories
        # One-hot from indices: scatter into (B, G, K)
        one_hot = F.one_hot(indices.long(), num_classes=K).to(torch.float32)
        flat = one_hot.reshape(B, G * K)
        embedding = self.cat_to_continuous(flat)

        x = self.stem(embedding).view(
            B, self.channels[0], self.base_h, self.base_w
        )
        x = torch.sin(x)
        for block in self.blocks:
            x = block(x)
        refined = self.refine1(self.refine0(x))
        x = x + 0.1 * torch.sin(refined)
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)  # (B, 2, 3, H, W)


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one RSSMC1 archive's bytes into one contest ``.raw`` file."""
    arc = parse_archive(archive_bytes)
    render_device = torch.device(select_inflate_device(device))

    decoder = DreamerV3RSSMDecoderTorch(
        num_groups=arc.num_groups,
        num_categories=arc.num_categories,
        decoder_latent_dim=arc.decoder_latent_dim,
        base_channels=arc.base_channels,
    ).to(render_device)

    # Load state_dict (numpy fp16 arrays from archive → torch tensors).
    # MLX stores conv weights in NHWC layout (out_ch, kH, kW, in_ch); PyTorch
    # expects NCHW layout (out_ch, in_ch, kH, kW). Transpose 4-D weight tensors
    # back to NCHW on load. Mirrors the canonical PR95 inverse of
    # tac.local_acceleration.pr95_hnerv_mlx::_torch_conv_to_mlx.
    torch_sd: dict[str, torch.Tensor] = {}
    for key, np_arr in arc.decoder_state_dict.items():
        arr_fp32 = np_arr.astype("float32")
        if arr_fp32.ndim == 4:
            # NHWC (out_ch, kH, kW, in_ch) → NCHW (out_ch, in_ch, kH, kW)
            arr_fp32 = arr_fp32.transpose(0, 3, 1, 2).copy()
        tensor = torch.from_numpy(arr_fp32)
        torch_sd[key] = tensor

    load_result = decoder.load_state_dict(torch_sd, strict=False)
    if set(load_result.missing_keys) or set(load_result.unexpected_keys):
        raise RuntimeError(
            "RSSMC1 decoder state_dict mismatch: "
            f"missing={sorted(load_result.missing_keys)} "
            f"unexpected={sorted(load_result.unexpected_keys)}"
        )
    decoder.eval()

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = arc.num_pairs
    indices_full = torch.from_numpy(arc.category_indices).to(
        device=render_device, dtype=torch.long
    )

    n = 0
    with torch.inference_mode(), open(output_raw_path, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch_idx = indices_full[i:j]  # (B, G)
            decoded = decoder(batch_idx)  # (B, 2, 3, DECODER_H, DECODER_W)
            B = decoded.shape[0]
            flat = decoded.reshape(B * 2, 3, DECODER_H, DECODER_W)
            up = F.interpolate(
                flat,
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            frames = (
                up.clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            fout.write(frames.tobytes())
            n += B * 2
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
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``.

    Honors the contest's 3-positional-arg inflate.sh contract per Catalog #146.
    """
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
    "DECODER_H",
    "DECODER_W",
    "DreamerV3RSSMDecoderTorch",
    "_read_single_member_archive_bytes",
    "inflate_one_video",
    "main_cli",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
