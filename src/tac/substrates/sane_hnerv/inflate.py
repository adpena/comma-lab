"""sane_hnerv inflate runtime — <= 100 LOC; PR101-mirror style.

This file is the contest-runtime image of the substrate. It is imported by
``submissions/sane_hnerv/inflate.py`` (one-line passthrough) at packet-build
time. The whole forward path is:

1. Read the archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> ``(decoder_state_dict, latents, meta)``.
3. Build the substrate from ``meta`` (no training; deterministic).
4. ``model.load_state_dict(decoder_state_dict)``; copy latents.
5. For each pair index i in [0, num_pairs): render (rgb_0, rgb_1); write
   ``output_dir/0/{2*i}.png`` and ``output_dir/0/{2*i+1}.png``.

L4 budget: <= 100 LOC, <= 2 external deps (torch, brotli; numpy is the
torch transitive). Reviewable in 30 seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .archive import parse_archive
from .architecture import SaneHnervConfig, SaneHnervSubstrate


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

    cfg = SaneHnervConfig(
        latent_dim=int(arc.latents.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
    )

    model = SaneHnervSubstrate(cfg).to(device).eval()
    incompat = model.load_state_dict(arc.decoder_state_dict, strict=False)
    missing = set(incompat.missing_keys)
    unexpected = set(incompat.unexpected_keys)
    if missing - {"latents"} or unexpected:
        raise RuntimeError(
            "sane_hnerv archive state_dict mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    with torch.no_grad():
        model.latents.copy_(arc.latents.to(device=device, dtype=model.latents.dtype))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-import PIL inside the function to keep this module's import light
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
