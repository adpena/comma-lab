# SPDX-License-Identifier: MIT
"""z6_v2_cargo_cult_unwind inflate runtime — contest-compliant 3-arg entry point.

Per HNeRV parity L4 + CLAUDE.md "Substrate scaffolds MUST be COMPLETE or
RESEARCH-ONLY" non-negotiable + Catalog #146 contest-compliant inflate runtime:
3-arg signature (archive_dir, output_dir, file_list); torch-only dependency
closure (numpy + brotli pulled by archive); ≤200 LOC reviewable in 30 seconds.

This is Z6-v2's OWN canonical inflate runtime per INDIVIDUALLY-FRACTAL — NOT
shared-helper shortcut from PACT-NeRV. Consumes the Z6V2CU1 archive grammar
defined in :mod:`tac.substrates.z6_v2_cargo_cult_unwind.archive`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .architecture import Z6V2Config, Z6V2Substrate
from .archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes, output_dir: Path, *, device: str = "cpu"
) -> None:
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    cfg = Z6V2Config(
        latent_dim=int(arc.latents.shape[1]),
        ego_dim=int(arc.ego_vecs.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        rao_ballard_level_boundary=int(meta.get("rao_ballard_level_boundary", 3)),
        film_generator_depth=int(meta.get("film_generator_depth", 3)),
        film_hidden_width=int(meta.get("film_hidden_width", 80)),
        cooperative_receiver_beta=float(meta.get("cooperative_receiver_beta", 0.5)),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )
    model = Z6V2Substrate(cfg).to(device).eval()
    model.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        model.latents.copy_(
            arc.latents.to(device=device, dtype=model.latents.dtype)
        )
        model.ego_vecs.copy_(
            arc.ego_vecs.to(device=device, dtype=model.ego_vecs.dtype)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image  # type: ignore[import-not-found]
    with torch.no_grad():
        for pair_idx in range(cfg.num_pairs):
            idx_tensor = torch.tensor([pair_idx], device=device, dtype=torch.long)
            rgb_0, rgb_1 = model(idx_tensor)
            for off, rgb in ((0, rgb_0), (1, rgb_1)):
                frame_idx = 2 * pair_idx + off
                arr = (
                    rgb[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                    * 255.0
                )
                arr = arr.round().clip(0, 255).astype("uint8")
                Image.fromarray(arr).save(output_dir / f"{frame_idx}.png")


def main_cli() -> int:
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
    for fname in file_list:
        base = Path(fname).stem
        archive_bytes = (archive_dir / "0.bin").read_bytes()
        inflate_one_video(archive_bytes, output_dir / base, device="cpu")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main_cli())
