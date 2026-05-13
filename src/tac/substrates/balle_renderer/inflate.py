"""balle_renderer inflate runtime — <= 200 LOC waiver per HNeRV L4 NEEDS-WORK (β).

This file is the contest-runtime image of the β substrate. It is imported by
``submissions/balle_renderer/inflate.py`` (one-line passthrough) at packet-build
time. The whole forward path is:

1. Read the archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> ``BalleRendererArchive``.
3. Build the substrate from ``meta`` (no training; deterministic).
4. Load encoder + decoder + hyperprior state_dicts; copy latents + scales.
5. For each pair index i in [0, num_pairs): render (rgb_0, rgb_1); append
   frames to one contest ``.raw`` tensor file.

L4 budget: <= 200 LOC waiver (council §4.2 β NEEDS-WORK note: GDN forward
adds ~30 LOC over α's 80 LOC). Target ~150 LOC. <= 2 external deps:
``torch`` + ``brotli`` (numpy is the torch transitive). Catalog #146 contract
(<inflate.sh archive_dir output_dir file_list> 3 positional args).
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
from .architecture import BalleRendererConfig, BalleRendererSubstrate


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one archive's bytes into one contest ``.raw`` file.

    Args:
        archive_bytes: raw bytes of the ``0.bin`` member.
        output_raw_path: where to write the raw tensor stream.
        device: ``"auto"``/``"cpu"``/``"cuda"`` via ``PACT_INFLATE_DEVICE``.
    """
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    render_device = select_inflate_device(device)

    cfg = BalleRendererConfig(
        latent_dim=int(arc.latents.shape[1]),
        hyper_latent_dim=int(arc.scales.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        hyper_mlp_channels=tuple(int(c) for c in meta["hyper_mlp_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        gdn_eps=float(meta.get("gdn_eps", 1e-12)),
        quantize_noise_std=float(meta.get("quantize_noise_std", 0.0)),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
    )

    model = BalleRendererSubstrate(cfg).to(render_device).eval()

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
    incompat = model.load_state_dict(merged, strict=False)
    missing = set(incompat.missing_keys)
    unexpected = set(incompat.unexpected_keys)
    if missing - {"latents"} or unexpected:
        raise RuntimeError(
            "balle_renderer archive state_dict mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )

    with torch.no_grad():
        model.latents.copy_(arc.latents.to(device=render_device, dtype=model.latents.dtype))

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad():
        with output_raw_path.open("wb") as fh:
            for pair_idx in range(cfg.num_pairs):
                idx_tensor = torch.tensor([pair_idx], device=render_device, dtype=torch.long)
                rgb_0, rgb_1, _rate = model(idx_tensor)
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
