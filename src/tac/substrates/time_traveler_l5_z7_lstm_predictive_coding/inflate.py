# SPDX-License-Identifier: MIT
"""Z7-GRU scorer-free inflate runtime scaffold.

Loads a Z7PCWM1 archive, replays the parsed GRU latent sequence, consumes the
Z6-compatible decoder weight stream, and writes contest-shaped raw RGB output.

This is a byte-closed runtime scaffold, not a score authority. It imports no
scorer code and does not make the lane dispatch-ready; the full score-aware
trainer, trained packet, paired exact eval, and Wave N+1 council gates remain
required before promotion.
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
from tac.substrates.time_traveler_l5_z6.architecture import EVAL_HW, _Z6Decoder
from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.archive import (
    Z7PredictiveCodingArchive,
    parse_archive,
    replay_latent_sequence,
)


def _meta_int(meta: dict[str, object], key: str) -> int:
    if key not in meta:
        raise KeyError(f"Z7PCWM1 metadata missing required decoder field {key!r}")
    return int(meta[key])


def _meta_int_tuple(meta: dict[str, object], key: str) -> tuple[int, ...]:
    if key not in meta:
        raise KeyError(f"Z7PCWM1 metadata missing required decoder field {key!r}")
    value = meta[key]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Z7PCWM1 metadata field {key!r} must be a list/tuple")
    return tuple(int(v) for v in value)


def _build_decoder(archive: Z7PredictiveCodingArchive, device: str) -> _Z6Decoder:
    """Build the Z6-compatible decoder used by the Z7 prebuild runtime."""

    meta = archive.meta
    decoder = _Z6Decoder(
        latent_dim=archive.config.latent_dim,
        embed_dim=_meta_int(meta, "decoder_embed_dim"),
        initial_grid_h=_meta_int(meta, "decoder_initial_grid_h"),
        initial_grid_w=_meta_int(meta, "decoder_initial_grid_w"),
        decoder_channels=_meta_int_tuple(meta, "decoder_channels"),
        num_upsample_blocks=_meta_int(meta, "decoder_num_upsample_blocks"),
        output_height=int(meta.get("output_height", EVAL_HW[0])),
        output_width=int(meta.get("output_width", EVAL_HW[1])),
    ).to(device)
    decoder.load_state_dict(archive.decoder_state_dict, strict=True)
    return decoder.eval()


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one Z7PCWM1 archive's bytes into one contest ``.raw`` file."""

    archive = parse_archive(archive_bytes)
    render_device = select_inflate_device(device)
    decoder = _build_decoder(archive, render_device)
    latents = replay_latent_sequence(archive).to(render_device, dtype=torch.float32)

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_idx in range(archive.config.num_pairs):
            z_t = latents[pair_idx : pair_idx + 1]
            rgb_0, rgb_1 = decoder(z_t)
            frames_written += write_rgb_pair_to_raw(
                fh,
                rgb_0,
                rgb_1,
                input_range="unit",
            )
    return frames_written


def _read_single_member_archive_bytes(archive_dir: Path) -> bytes:
    """Read the single contest archive member, failing on ambiguity."""

    zero_bin = archive_dir / "0.bin"
    x_member = archive_dir / "x"
    present = [path for path in (zero_bin, x_member) if path.is_file()]
    if len(present) != 1:
        if not present:
            raise FileNotFoundError(
                f"expected exactly one archive member at {zero_bin} or {x_member}"
            )
        raise ValueError(
            f"ambiguous archive members present: {zero_bin} and {x_member}"
        )
    return present[0].read_bytes()


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
    "_read_single_member_archive_bytes",
    "inflate_one_video",
    "main_cli",
]


if __name__ == "__main__":  # pragma: no cover - CLI smoke
    sys.exit(main_cli())
