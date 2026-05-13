"""cool_chic inflate runtime — <= 100 LOC; mirrors sane_hnerv/inflate.py style.

Contest-runtime image of the substrate. ``submissions/cool_chic/inflate.py``
will be a one-line passthrough to ``main_cli`` at packet-build time. Forward path:

1. Read archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> (synth_sd, ar_prior_sd, latents_coarse, latents_fine, meta).
3. Build the substrate from ``meta`` (deterministic, no training).
4. Load state_dicts; copy latents.
5. For each pair index, render (rgb_0, rgb_1); append to one contest ``.raw``
   tensor file.

L4 budget: <= 100 LOC, <= 2 external deps (torch, brotli; numpy is torch transitive).
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
from .architecture import CoolChicConfig, CoolChicSubstrate


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

    cfg = CoolChicConfig(
        latent_channels_coarse=int(arc.latents_coarse.shape[1]),
        latent_channels_fine=int(arc.latents_fine.shape[1]),
        coarse_scale_factor=int(meta["coarse_scale_factor"]),
        fine_scale_factor=int(meta["fine_scale_factor"]),
        synthesis_hidden=int(meta["synthesis_hidden"]),
        synthesis_layers=int(meta["synthesis_layers"]),
        ar_prior_hidden=int(meta["ar_prior_hidden"]),
        num_pairs=int(arc.latents_coarse.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )

    model = CoolChicSubstrate(cfg).to(render_device).eval()
    model.synthesis.load_state_dict(arc.synthesis_state_dict, strict=False)
    model.ar_prior_coarse.load_state_dict(
        {k.replace("coarse.", ""): v for k, v in arc.ar_prior_state_dict.items() if k.startswith("coarse.")},
        strict=False,
    )
    model.ar_prior_fine.load_state_dict(
        {k.replace("fine.", ""): v for k, v in arc.ar_prior_state_dict.items() if k.startswith("fine.")},
        strict=False,
    )
    with torch.no_grad():
        model.latents_coarse.copy_(arc.latents_coarse.to(device=render_device, dtype=model.latents_coarse.dtype))
        model.latents_fine.copy_(arc.latents_fine.to(device=render_device, dtype=model.latents_fine.dtype))

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad():
        with output_raw_path.open("wb") as fh:
            for pair_idx in range(cfg.num_pairs):
                idx_tensor = torch.tensor([pair_idx], device=render_device, dtype=torch.long)
                rgb_0, rgb_1 = model(idx_tensor)
                frames_written += write_rgb_pair_to_raw(fh, rgb_0, rgb_1, input_range="unit")
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
