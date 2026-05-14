"""C6 MDL-IBPS inflate runtime — contest raw-output contract.

Loads the IBPS1 archive, reconstructs the decoder from the stored state_dict,
copies the per-pair latents in, and writes one raw-output ``.raw`` file per
contest video (1200 frames of 874x1164 RGB per video).

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device (Catalog #1; CPU/CUDA only via canonical select_inflate_device).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤200 for
substrate-engineering lanes (encoder/decoder/latent dequant + composition).

Per Catalog #205 device selection uses canonical ``select_inflate_device``.
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
from tac.substrates.c6_e4_mdl_ibps.architecture import (
    EVAL_HW,
    MDLIBPSConfig,
    MDLIBPSSubstrate,
)
from tac.substrates.c6_e4_mdl_ibps.archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one IBPS1 archive's bytes into one contest ``.raw`` file."""
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    render_device = select_inflate_device(device)

    cfg = MDLIBPSConfig(
        latent_dim=int(arc.latents.shape[1]),
        encoder_input_channels=int(meta.get("encoder_input_channels", 3)),
        encoder_sin_freq=float(meta.get("encoder_sin_freq", 30.0)),
        decoder_embed_dim=int(meta["decoder_embed_dim"]),
        decoder_initial_grid_h=int(meta["decoder_initial_grid_h"]),
        decoder_initial_grid_w=int(meta["decoder_initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        decoder_num_upsample_blocks=int(meta["decoder_num_upsample_blocks"]),
        decoder_sin_freq=float(meta.get("decoder_sin_freq", 30.0)),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta.get("output_height", EVAL_HW[0])),
        output_width=int(meta.get("output_width", EVAL_HW[1])),
        beta_ib=float(meta.get("beta_ib", 0.01)),
        latent_init_std=float(meta.get("latent_init_std", 0.02)),
    )

    model = MDLIBPSSubstrate(cfg).to(render_device).eval()

    # Load encoder + decoder state dicts (the substrate stores them in submodules)
    enc_load = model.encoder.load_state_dict(arc.encoder_state_dict, strict=False)
    if set(enc_load.missing_keys) or set(enc_load.unexpected_keys):
        raise RuntimeError(
            "C6 IBPS1 encoder state_dict mismatch: "
            f"missing={sorted(enc_load.missing_keys)} "
            f"unexpected={sorted(enc_load.unexpected_keys)}"
        )
    dec_load = model.decoder.load_state_dict(arc.decoder_state_dict, strict=False)
    if set(dec_load.missing_keys) or set(dec_load.unexpected_keys):
        raise RuntimeError(
            "C6 IBPS1 decoder state_dict mismatch: "
            f"missing={sorted(dec_load.missing_keys)} "
            f"unexpected={sorted(dec_load.unexpected_keys)}"
        )

    with torch.no_grad():
        model.latents.copy_(
            arc.latents.to(device=render_device, dtype=model.latents.dtype)
        )

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad():
        with output_raw_path.open("wb") as fh:
            for pair_idx in range(cfg.num_pairs):
                idx_tensor = torch.tensor(
                    [pair_idx], device=render_device, dtype=torch.long
                )
                # Eval path: frames_for_encoder=None; only decoder forward used.
                rgb_0, rgb_1, _mu, _logvar = model(idx_tensor, frames_for_encoder=None)
                frames_written += write_rgb_pair_to_raw(
                    fh, rgb_0, rgb_1, input_range="unit"
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
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``.

    Honors the contest's 3-positional-arg inflate.sh contract per Catalog #146.
    """
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
            archive_bytes, raw_output_path(output_dir, name), device=device
        )
    return 0


__all__ = [
    "_read_single_member_archive_bytes",
    "inflate_one_video",
    "main_cli",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
