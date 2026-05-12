"""balle_renderer inflate runtime — <= 200 LOC waiver per HNeRV L4 NEEDS-WORK (β).

This file is the contest-runtime image of the β substrate. It is imported by
``submissions/balle_renderer/inflate.py`` (one-line passthrough) at packet-build
time. The whole forward path is:

1. Read the archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> ``BalleRendererArchive``.
3. Build the substrate from ``meta`` (no training; deterministic).
4. Load encoder + decoder + hyperprior state_dicts; copy latents + scales.
5. For each pair index i in [0, num_pairs): render (rgb_0, rgb_1); write
   ``output_dir/<base>/<frame_idx>.png``.

L4 budget: <= 200 LOC waiver (council §4.2 β NEEDS-WORK note: GDN forward
adds ~30 LOC over α's 80 LOC). Target ~150 LOC. <= 2 external deps:
``torch`` + ``brotli`` (numpy is the torch transitive). Catalog #146 contract
(<inflate.sh archive_dir output_dir file_list> 3 positional args).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .archive import parse_archive
from .architecture import BalleRendererConfig, BalleRendererSubstrate


def inflate_one_video(
    archive_bytes: bytes,
    output_dir: Path,
    *,
    device: str = "cpu",
) -> None:
    """Inflate one archive's bytes into ``output_dir/<frame_idx>.png`` files.

    Args:
        archive_bytes: raw bytes of the ``0.bin`` member.
        output_dir: where to write per-frame PNGs.
        device: ``"cpu"`` (default, contest-leaderboard CPU axis) or ``"cuda"``.
    """
    arc = parse_archive(archive_bytes)
    meta = arc.meta

    cfg = BalleRendererConfig(
        latent_dim=int(arc.latents.shape[1]),
        hyper_latent_dim=int(arc.scales.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        hyper_mlp_channels=tuple(int(c) for c in meta["hyper_mlp_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
    )

    model = BalleRendererSubstrate(cfg).to(device).eval()

    # Load each state_dict into its sub-module.
    # The β substrate's three blobs are:
    #   - encoder: hyper_analysis.*
    #   - decoder: latent_embed.* + blocks.* + head_rgb_0.* + head_rgb_1.*
    #   - hyperprior: hyper_synthesis.* + w_prior_*
    # We merge them into one state_dict and load with strict=False so the
    # un-named per-pair latents aren't required.
    merged: dict[str, torch.Tensor] = {}
    merged.update({"hyper_analysis." + k: v for k, v in arc.encoder_state_dict.items()})
    merged.update(arc.decoder_state_dict)  # decoder keys are already top-level
    merged.update(arc.hyperprior_state_dict)
    model.load_state_dict(merged, strict=False)

    with torch.no_grad():
        model.latents.copy_(arc.latents.to(device=device, dtype=model.latents.dtype))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-import PIL inside the function to keep this module's import light
    from PIL import Image  # type: ignore[import-not-found]

    with torch.no_grad():
        for pair_idx in range(cfg.num_pairs):
            idx_tensor = torch.tensor([pair_idx], device=device, dtype=torch.long)
            rgb_0, rgb_1, _rate = model(idx_tensor)
            for off, rgb in ((0, rgb_0), (1, rgb_1)):
                frame_idx = 2 * pair_idx + off
                arr = (rgb[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0)
                arr = arr.round().clip(0, 255).astype("uint8")
                Image.fromarray(arr).save(output_dir / f"{frame_idx}.png")


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``.

    Honors the contest's 3-positional-arg inflate.sh contract per Catalog #146.
    """
    if len(sys.argv) < 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])

    file_list = file_list_path.read_text(encoding="utf-8").strip().splitlines()
    for fname in file_list:
        base = Path(fname).stem  # "0" from "0.mkv"
        archive_bytes = (archive_dir / "0.bin").read_bytes()
        inflate_one_video(archive_bytes, output_dir / base, device="cpu")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
