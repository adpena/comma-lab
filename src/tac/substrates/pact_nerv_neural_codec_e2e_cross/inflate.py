# SPDX-License-Identifier: MIT
"""pact_nerv_neural_codec_e2e_cross inflate runtime - <= 200 LOC; NCEC consumer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .architecture import (
    PactNervNeuralCodecE2ECrossConfig,
    PactNervNeuralCodecE2ECrossSubstrate,
)
from .archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes, output_dir: Path, *, device: str = "cpu"
) -> None:
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    cfg = PactNervNeuralCodecE2ECrossConfig(
        latent_dim_a=int(arc.latents_a.shape[1]),
        latent_dim_b=int(arc.latents_b.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        num_pairs=int(arc.latents_a.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
        hyperprior_hidden=int(meta["hyperprior_hidden"]),
        gate_init_bias=float(meta.get("gate_init_bias", 0.0)),
    )
    model = PactNervNeuralCodecE2ECrossSubstrate(cfg).to(device).eval()
    # Load all 3 state dicts (branch_a, branch_b, gate)
    with torch.no_grad():
        # Re-prefix decoder state dicts for module-name parity
        full_sd: dict[str, torch.Tensor] = {}
        for k, v in arc.decoder_a_state_dict.items():
            full_sd[f"branch_a.{k}"] = v
        for k, v in arc.decoder_b_state_dict.items():
            full_sd[f"branch_b.{k}"] = v
        for k, v in arc.hyperprior_state_dict.items():
            full_sd[f"gate.{k}"] = v
        model.load_state_dict(full_sd, strict=False)
        model.latents_a.copy_(
            arc.latents_a.to(device=device, dtype=model.latents_a.dtype)
        )
        model.latents_b.copy_(
            arc.latents_b.to(device=device, dtype=model.latents_b.dtype)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image  # type: ignore[import-not-found]
    with torch.no_grad():
        for pair_idx in range(cfg.num_pairs):
            idx_tensor = torch.tensor([pair_idx], device=device, dtype=torch.long)
            rgb_0, rgb_1 = model(idx_tensor)
            for off, rgb in ((0, rgb_0), (1, rgb_1)):
                frame_idx = 2 * pair_idx + off
                arr = (rgb[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0)
                arr = arr.round().clip(0, 255).astype("uint8")
                Image.fromarray(arr).save(output_dir / f"{frame_idx}.png")


def main_cli() -> int:
    if len(sys.argv) < 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    file_list = file_list_path.read_text(encoding="utf-8").strip().splitlines()
    for fname in file_list:
        base = Path(fname).stem
        archive_bytes = (archive_dir / "0.bin").read_bytes()
        inflate_one_video(archive_bytes, output_dir / base, device="cpu")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main_cli())
