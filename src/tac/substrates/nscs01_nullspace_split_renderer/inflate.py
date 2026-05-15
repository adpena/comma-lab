# SPDX-License-Identifier: MIT
"""NSCS01 nullspace-split-renderer inflate runtime — contest raw-output contract.

Loads the NSP1 archive, reconstructs both render heads at their respective
bit-widths + per-pair latents, then per-pair runs both heads forward and
writes raw RGB frames at camera resolution.

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device (Catalog #1; CPU/CUDA only via Catalog #205 canonical helper).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤ 200 for
substrate-engineering lanes (full split-renderer reconstruction + per-pair
forward + raw output).
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
from tac.substrates.nscs01_nullspace_split_renderer.architecture import (
    NullspaceSplitConfig,
    NullspaceSplitRenderer,
)
from tac.substrates.nscs01_nullspace_split_renderer.archive import (
    deserialize_head_state_dicts,
    deserialize_latents,
    parse_archive,
)


def _read_single_member_archive_bytes(archive_dir: Path) -> bytes:
    """Read the single contest archive member, failing on ambiguity."""
    zero_bin = archive_dir / "0.bin"
    x_member = archive_dir / "x"
    present = [p for p in (zero_bin, x_member) if p.is_file()]
    if len(present) != 1:
        if not present:
            raise FileNotFoundError(
                f"expected exactly one archive member at {zero_bin} or {x_member}"
            )
        raise ValueError(
            f"ambiguous archive members present: {zero_bin} and {x_member}"
        )
    return present[0].read_bytes()


def _build_renderer_from_archive_bytes(
    archive_bytes: bytes, device: str
) -> tuple[NullspaceSplitRenderer, torch.Tensor]:
    """Reconstruct the substrate from archive bytes.

    Returns (renderer, latents) ready to render on the given device.
    """
    arc = parse_archive(archive_bytes)
    cfg = NullspaceSplitConfig(
        latent_dim=arc.latent_dim,
        head0_bits=arc.head0_bits,
        head1_bits=arc.head1_bits,
        latent_bits=arc.latent_bits,
        head0_base_channels=arc.head0_base_channels,
        head1_base_channels=arc.head1_base_channels,
        num_pairs=arc.num_pairs,
    )
    renderer = NullspaceSplitRenderer(cfg).to(device)
    head0_sd, head1_sd = deserialize_head_state_dicts(arc)
    head0_sd = {k: v.to(device) for k, v in head0_sd.items()}
    head1_sd = {k: v.to(device) for k, v in head1_sd.items()}
    renderer.frame_0_head.load_state_dict(head0_sd)
    renderer.frame_1_head.load_state_dict(head1_sd)
    latents = deserialize_latents(arc).to(device)
    with torch.no_grad():
        renderer.latents.copy_(latents)
    renderer.eval()
    return renderer, latents


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
    pairs_per_chunk: int = 32,
) -> int:
    """Inflate one NSP1 archive's bytes into one contest ``.raw`` file."""
    render_device = select_inflate_device(device)
    renderer, _ = _build_renderer_from_archive_bytes(archive_bytes, render_device)
    num_pairs = renderer.cfg.num_pairs

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for start in range(0, num_pairs, pairs_per_chunk):
            stop = min(start + pairs_per_chunk, num_pairs)
            idx = torch.arange(start, stop, device=render_device, dtype=torch.long)
            frame_0_batch, frame_1_batch = renderer.reconstruct_pair(idx)
            # Convert from [0, 255] to [0, 1] for write_rgb_pair_to_raw input_range="unit".
            frame_0_unit = frame_0_batch / 255.0
            frame_1_unit = frame_1_batch / 255.0
            for k in range(frame_0_batch.shape[0]):
                frames_written += write_rgb_pair_to_raw(
                    fh,
                    frame_0_unit[k : k + 1],
                    frame_1_unit[k : k + 1],
                    input_range="unit",
                )
    return frames_written


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``."""
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
    archive_bytes = _read_single_member_archive_bytes(archive_dir)
    device = select_inflate_device()
    for fname in file_list:
        name = fname.strip()
        if not name:
            continue
        inflate_one_video(
            archive_bytes,
            raw_output_path(output_dir, name),
            device=device,
        )
    return 0


__all__ = [
    "_build_renderer_from_archive_bytes",
    "_read_single_member_archive_bytes",
    "inflate_one_video",
    "main_cli",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
