# SPDX-License-Identifier: MIT
"""pact_nerv_cross_codec_b inflate runtime - <= 200 LOC; CC_B consumer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .architecture import PactNervCrossCodecBConfig, PactNervCrossCodecBSubstrate
from .archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes, output_dir: Path, *, device: str = "cpu"
) -> None:
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    cfg = PactNervCrossCodecBConfig(
        latent_dim=int(arc.latents.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
        pr106_score_table_size=int(arc.score_table_size),
        pose_dim=int(arc.pose_dim),
        ia3_init_delta_std=float(meta.get("ia3_init_delta_std", 0.01)),
        composition_alpha=float(meta.get("composition_alpha", 0.1)),
    )
    model = PactNervCrossCodecBSubstrate(cfg).to(device).eval()
    model.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        model.latents.copy_(
            arc.latents.to(device=device, dtype=model.latents.dtype)
        )
        model.ego_poses.copy_(
            arc.ego_poses.to(device=device, dtype=model.ego_poses.dtype)
        )
        # Load per-pair PR106 score-table indices from score_index_bytes
        import numpy as np
        idx_arr = np.frombuffer(arc.score_index_bytes, dtype=np.uint8)
        idx_tensor = torch.from_numpy(
            idx_arr[: cfg.num_pairs].copy().astype("int64")
        )
        model.score_indices.copy_(idx_tensor.to(device))
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
