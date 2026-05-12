"""vq_vae inflate runtime — <= 100 LOC; mirrors sane_hnerv style.

Contest-runtime image of the substrate. ``submissions/vq_vae/inflate.py`` will
be a one-line passthrough to ``main_cli`` at packet-build time. Forward path:

1. Read archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> (decoder_sd, indices, meta, K, D).
3. Build the substrate from ``meta`` (deterministic, no training).
4. Load state_dict; the per_pair_features are reconstructed by looking up the
   codebook at the stored indices (no encoder forward needed at inflate time).
5. Decode per-pair via codebook[indices] -> decoder -> RGB; save PNGs.

L4 budget: <= 100 LOC, <= 2 external deps (torch, brotli; numpy is torch transitive).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .architecture import VqVaeConfig, VqVaeSubstrate
from .archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes,
    output_dir: Path,
    *,
    device: str = "cpu",
) -> None:
    """Inflate one archive's bytes into ``output_dir/<frame_idx>.png`` files."""
    arc = parse_archive(archive_bytes)
    meta = arc.meta

    cfg = VqVaeConfig(
        codebook_size=int(arc.codebook_size),
        embedding_dim=int(arc.embedding_dim),
        encoder_hidden=int(meta["encoder_hidden"]),
        decoder_hidden=int(meta["decoder_hidden"]),
        grid_downsample=int(meta["grid_downsample"]),
        num_pairs=int(arc.indices.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )

    model = VqVaeSubstrate(cfg).to(device).eval()
    model.load_state_dict(arc.decoder_state_dict, strict=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image  # type: ignore[import-not-found]

    with torch.no_grad():
        idx_grid_all = arc.indices.to(device=device)  # (num_pairs, 2, h, w)
        codebook = model.codebook  # (K, D)
        for pair_idx in range(cfg.num_pairs):
            for off in (0, 1):
                idx_grid = idx_grid_all[pair_idx, off]  # (h, w)
                # Codebook lookup: gather D-dim embeddings at each cell
                z_q = codebook[idx_grid.view(-1)].view(
                    idx_grid.shape[0], idx_grid.shape[1], cfg.embedding_dim
                ).permute(2, 0, 1).unsqueeze(0)  # (1, D, h, w)
                rgb = model.decoder(z_q)  # (1, 3, H, W)
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
        base = Path(fname).stem
        archive_bytes = (archive_dir / "0.bin").read_bytes()
        inflate_one_video(archive_bytes, output_dir / base, device="cpu")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
