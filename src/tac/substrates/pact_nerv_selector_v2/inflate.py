# SPDX-License-Identifier: MIT
"""pact_nerv_selector_v2 inflate runtime - <= 200 LOC; PSV2 monolithic-archive consumer.

Forward path:

1. Read archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> (decoder_state_dict, latents, selector_bytes, meta).
3. Build substrate from ``meta`` + ``palette_size``.
4. ``model.load_state_dict(decoder_state_dict, strict=False)``; copy latents.
5. For each pair index render (rgb_0, rgb_1) + apply selector-conditioned
   deterministic frame-0 transform (FEC6 palette modes) per L1 wire-in.
   L0 SCAFFOLD: omits the selector application (decode-only proof).

L4 budget: <= 200 LOC; <= 2 external deps (torch + brotli).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .architecture import (
    ArithmeticSelectorCoder,
    PactNervSelectorV2Config,
    PactNervSelectorV2Substrate,
    apply_selector_code_to_pair_frames_255,
)
from .archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes,
    output_dir: Path,
    *,
    device: str = "cpu",
) -> None:
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    cfg = PactNervSelectorV2Config(
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
        selector_palette_size=int(arc.palette_size),
    )

    model = PactNervSelectorV2Substrate(cfg).to(device).eval()
    model.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        model.latents.copy_(
            arc.latents.to(device=device, dtype=model.latents.dtype)
        )
    cum_freq = meta.get("selector_cum_freq")
    coder = ArithmeticSelectorCoder(
        int(arc.palette_size),
        cum_freq=tuple(int(v) for v in cum_freq) if isinstance(cum_freq, list) else None,
        precision=int(meta.get("selector_precision", 32)),
    )
    selector_indices = coder.decode(arc.selector_bytes, symbol_count=cfg.num_pairs)

    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image  # type: ignore[import-not-found]
    with torch.no_grad():
        for pair_idx in range(cfg.num_pairs):
            idx_tensor = torch.tensor([pair_idx], device=device, dtype=torch.long)
            rgb_0, rgb_1 = model(idx_tensor)
            pair_frames = torch.cat([rgb_0, rgb_1], dim=0).clamp(0.0, 1.0) * 255.0
            pair_frames = pair_frames.round()
            pair_frames = apply_selector_code_to_pair_frames_255(
                pair_frames,
                int(selector_indices[pair_idx]),
            )
            for off, rgb255 in ((0, pair_frames[0]), (1, pair_frames[1])):
                frame_idx = 2 * pair_idx + off
                arr = rgb255.permute(1, 2, 0).cpu().numpy()
                arr = arr.clip(0, 255).astype("uint8")
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
