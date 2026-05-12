"""wavelet inflate runtime — <= 100 LOC; mirrors sane_hnerv/inflate.py style.

Contest-runtime image of the substrate. ``submissions/wavelet/inflate.py`` will
be a one-line passthrough to ``main_cli`` at packet-build time. Forward path:

1. Read archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> (synth_sd, film_sd, LL, LH, HL, HH, meta).
3. Build the substrate from ``meta`` (deterministic, no training).
4. Load state_dicts; copy subbands.
5. For each pair index, render (rgb_0, rgb_1); save PNGs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .archive import parse_archive
from .architecture import WaveletConfig, WaveletSubstrate


def inflate_one_video(
    archive_bytes: bytes,
    output_dir: Path,
    *,
    device: str = "cpu",
) -> None:
    """Inflate one archive's bytes into ``output_dir/<frame_idx>.png`` files."""
    arc = parse_archive(archive_bytes)
    meta = arc.meta

    cfg = WaveletConfig(
        coeff_channels=int(arc.LL.shape[1]),
        synthesis_hidden=int(meta["synthesis_hidden"]),
        synthesis_layers=int(meta["synthesis_layers"]),
        num_pairs=int(arc.LL.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )

    model = WaveletSubstrate(cfg).to(device).eval()
    model.synthesis.load_state_dict(arc.synthesis_state_dict, strict=False)
    with torch.no_grad():
        if "film" in arc.film_state_dict:
            model.film.copy_(arc.film_state_dict["film"].to(model.film.dtype))
        model.coeff_ll.copy_(arc.LL.to(device=device, dtype=model.coeff_ll.dtype))
        model.coeff_lh.copy_(arc.LH.to(device=device, dtype=model.coeff_lh.dtype))
        model.coeff_hl.copy_(arc.HL.to(device=device, dtype=model.coeff_hl.dtype))
        model.coeff_hh.copy_(arc.HH.to(device=device, dtype=model.coeff_hh.dtype))

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
