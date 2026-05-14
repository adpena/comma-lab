# SPDX-License-Identifier: MIT
"""wavelet inflate runtime — <= 100 LOC; mirrors sane_hnerv/inflate.py style.

Contest-runtime image of the substrate. ``submissions/wavelet/inflate.py`` will
be a one-line passthrough to ``main_cli`` at packet-build time. Forward path:

1. Read archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> (synth_sd, film_sd, LL, LH, HL, HH, meta).
3. Build the substrate from ``meta`` (deterministic, no training).
4. Load state_dicts; copy subbands.
5. For each pair index, render (rgb_0, rgb_1); append contest raw RGB bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from tac.substrates._shared.inflate_runtime import (
    raw_output_path,
    select_inflate_device,
    write_rgb_pair_to_raw,
)
from .archive import parse_archive
from .architecture import WaveletConfig, WaveletSubstrate


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one archive's bytes into one contest ``.raw`` file."""
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    render_device = select_inflate_device(device)

    cfg = WaveletConfig(
        coeff_channels=int(arc.LL.shape[1]),
        synthesis_hidden=int(meta["synthesis_hidden"]),
        synthesis_layers=int(meta["synthesis_layers"]),
        num_pairs=int(arc.LL.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )

    model = WaveletSubstrate(cfg).to(render_device).eval()
    model.synthesis.load_state_dict(arc.synthesis_state_dict, strict=False)
    with torch.no_grad():
        if "film" in arc.film_state_dict:
            model.film.copy_(
                arc.film_state_dict["film"].to(
                    device=render_device,
                    dtype=model.film.dtype,
                )
            )
        model.coeff_ll.copy_(arc.LL.to(device=render_device, dtype=model.coeff_ll.dtype))
        model.coeff_lh.copy_(arc.LH.to(device=render_device, dtype=model.coeff_lh.dtype))
        model.coeff_hl.copy_(arc.HL.to(device=render_device, dtype=model.coeff_hl.dtype))
        model.coeff_hh.copy_(arc.HH.to(device=render_device, dtype=model.coeff_hh.dtype))

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad():
        with output_raw_path.open("wb") as fh:
            for pair_idx in range(cfg.num_pairs):
                idx_tensor = torch.tensor([pair_idx], device=render_device, dtype=torch.long)
                rgb_0, rgb_1 = model(idx_tensor)
                frames_written += write_rgb_pair_to_raw(
                    fh,
                    rgb_0,
                    rgb_1,
                    input_range="unit",
                )
    return frames_written


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
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    device = select_inflate_device()
    for fname in file_list:
        if not fname.strip():
            continue
        inflate_one_video(archive_bytes, raw_output_path(output_dir, fname), device=device)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
