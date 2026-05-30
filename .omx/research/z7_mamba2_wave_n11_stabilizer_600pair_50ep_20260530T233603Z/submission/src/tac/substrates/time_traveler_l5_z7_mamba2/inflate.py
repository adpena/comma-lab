# SPDX-License-Identifier: MIT
"""Z7-Mamba-2 scorer-free inflate runtime.

Loads a Z7MCM2 archive, replays the parsed Mamba-2 latent sequence,
consumes the Z6-compatible decoder weight stream, and writes contest-
shaped raw RGB output.

Byte-closed runtime per CLAUDE.md HNeRV parity L4 (Inflate ≤200 LOC
substrate-engineering waiver; this file is ~150 LOC). NO scorer imports
per CLAUDE.md "Strict scorer rule". CUDA/CPU agnostic via canonical
``select_inflate_device`` per Catalog #205.

[verified-against: tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.inflate]
[verified-against: tac.substrates._shared.inflate_runtime canonical helpers]
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
from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.architecture import (
    LatentAffineContextConditioner,
    Z7GruPredictiveCodingConfig,
    normalize_context_conditioning_mode,
)
from tac.substrates.time_traveler_l5_z7_mamba2.archive import (
    Z7Mamba2PredictiveCodingArchive,
    parse_archive,
    replay_latent_sequence_with_context,
)


def _meta_int(meta: dict[str, object], key: str) -> int:
    if key not in meta:
        raise KeyError(f"Z7MCM2 metadata missing required decoder field {key!r}")
    return int(meta[key])


def _meta_int_tuple(meta: dict[str, object], key: str) -> tuple[int, ...]:
    if key not in meta:
        raise KeyError(f"Z7MCM2 metadata missing required decoder field {key!r}")
    value = meta[key]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Z7MCM2 metadata field {key!r} must be a list/tuple")
    return tuple(int(v) for v in value)


def _build_decoder(
    archive: Z7Mamba2PredictiveCodingArchive, device: str
) -> _Z6Decoder:
    """Build the Z6-compatible decoder from archive metadata + weights."""
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
    # Cast fp16 archive weights to fp32 for decoder forward
    # Wave N+9 Slot 1 self-containment fix 2026-05-28: strip the canonical
    # "decoder." prefix that `export_state_dict` adds (per mlx_native.py:735+)
    # before loading into the bare `_Z6Decoder` module which expects
    # unprefixed keys (the prefix represents the field name in the parent
    # PyTorch module that owns the decoder as a child). Empirically verified
    # at landing: without prefix strip, `load_state_dict(strict=True)`
    # raises with Missing+Unexpected keys for all 12 decoder params.
    state_dict_fp32 = {
        (k[len("decoder."):] if k.startswith("decoder.") else k): v.to(torch.float32)
        for k, v in archive.decoder_state_dict.items()
    }
    decoder.load_state_dict(state_dict_fp32, strict=True)
    return decoder.eval()


def _context_condition_latents(
    archive: Z7Mamba2PredictiveCodingArchive,
    latents: torch.Tensor,
    contexts: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    """Apply any runtime-declared Z7 context-conditioned decoder pre-transform."""
    mode = normalize_context_conditioning_mode(
        str(archive.meta.get("context_conditioning_mode", "none"))
    )
    if mode == "none":
        return latents
    if mode != "latent_affine":
        raise ValueError(f"unsupported Z7-Mamba-2 context conditioning mode: {mode}")
    # Adapt Z7-Mamba-2 config to Z7-LSTM/GRU sister LatentAffineContextConditioner.
    adapter_cfg = Z7GruPredictiveCodingConfig(
        latent_dim=archive.config.latent_dim,
        ego_motion_dim=archive.config.ego_motion_dim,
        num_pairs=archive.config.num_pairs,
        context_conditioning_mode="latent_affine",
        context_affine_strength=archive.config.context_affine_strength,
    )
    conditioner = LatentAffineContextConditioner(adapter_cfg).to(device)
    if not archive.encoder_state_dict:
        raise ValueError(
            "latent_affine Z7MCM2 archive missing context conditioner state_dict "
            "in encoder_blob"
        )
    # Wave N+9 Slot 1 self-containment fix 2026-05-28: strip the canonical
    # "context_conditioner." / "conditioner." prefix that export_state_dict
    # adds before loading into the bare `LatentAffineContextConditioner` module
    # (sister of decoder.* + predictor.* prefix-strips above).
    _PREFIXES = ("context_conditioner.", "conditioner.")
    def _strip(k: str) -> str:
        for pfx in _PREFIXES:
            if k.startswith(pfx):
                return k[len(pfx):]
        return k
    state_dict_fp32 = {
        _strip(k): v.to(torch.float32) for k, v in archive.encoder_state_dict.items()
    }
    conditioner.load_state_dict(state_dict_fp32, strict=True)
    conditioner.eval()
    return conditioner(latents, contexts)


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one Z7MCM2 archive's bytes into one contest ``.raw`` file."""
    archive = parse_archive(archive_bytes)
    render_device = select_inflate_device(device)
    decoder = _build_decoder(archive, render_device)
    latents_cpu, contexts_cpu = replay_latent_sequence_with_context(archive)
    latents = latents_cpu.to(render_device, dtype=torch.float32)
    contexts = contexts_cpu.to(render_device, dtype=torch.float32)
    decoder_latents = _context_condition_latents(
        archive,
        latents,
        contexts,
        device=render_device,
    )

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_idx in range(archive.config.num_pairs):
            z_t = decoder_latents[pair_idx : pair_idx + 1]
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
