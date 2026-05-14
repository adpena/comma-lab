# SPDX-License-Identifier: MIT
"""Scorer-free S2SBS_AR/S2S1 inflate smoke runtime.

Reads ``0.bin`` from the contest archive directory, regenerates each
frame pair via the deterministic base decoder, stuffs payload bytes
into the HF channel, and writes contest-shaped raw RGB files. No
scorer imports, no upstream model weights, CPU-only at L0/L1 scaffold
stage.

LOC budget per HNeRV parity lesson 4: <= 200.

Contest signature: ``inflate.py <archive_dir> <output_dir> <file_list>``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from tac.substrates._shared.inflate_runtime import raw_output_path, write_rgb_pair_to_raw

from .architecture import S2sbsRenderer
from .archive import S2sbsArchive, parse_archive


def _select_device(device: str | None) -> str:
    requested = (device or "cpu").strip().lower()
    if requested != "cpu":
        raise RuntimeError("s2sbs_byte_stuffing L0/L1 inflate is CPU-only")
    return "cpu"


def render_pair(
    archive: S2sbsArchive,
    pair_index: int,
    *,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render one HF-stuffed pair from parsed archive state."""

    render_device = _select_device(device)
    renderer = S2sbsRenderer(archive.config).to(render_device).eval()
    idx = torch.tensor([int(pair_index)], dtype=torch.long, device=render_device)
    payload_rows = tuple(row for row in archive.payloads if row.pair_index == int(pair_index))
    with torch.no_grad():
        return renderer(idx, payload_rows)


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate S2S1 bytes into one deterministic contest-shaped raw file."""

    archive = parse_archive(archive_bytes)
    _select_device(device)
    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = S2sbsRenderer(archive.config).to("cpu").eval()
    payload_by_pair = {row.pair_index: row for row in archive.payloads}
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_index in range(archive.config.num_pairs):
            idx = torch.tensor([pair_index], dtype=torch.long)
            row = payload_by_pair.get(pair_index)
            rows = (row,) if row is not None else ()
            rgb_0, rgb_1 = renderer(idx, rows)
            frames_written += write_rgb_pair_to_raw(fh, rgb_0, rgb_1, input_range="unit")
    return frames_written


def main_cli() -> int:
    if len(sys.argv) < 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    for name in file_list_path.read_text(encoding="utf-8").splitlines():
        clean = name.strip()
        if clean:
            inflate_one_video(archive_bytes, raw_output_path(output_dir, clean), device="cpu")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main_cli())
