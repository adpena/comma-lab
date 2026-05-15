# SPDX-License-Identifier: MIT
"""ATW codec V1 inflate runtime — contest raw-output contract.

Loads the ATW1 archive, reconstructs encoder + decoder + WZ side-info head
from stored state_dicts, copies in z_residual + scorer_class_prior_table,
and writes one raw-output ``.raw`` file per contest video (1200 frames of
874x1164 RGB per video).

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device (Catalog #1; CPU/CUDA only via canonical select_inflate_device).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤200 for
substrate-engineering lanes (encoder/decoder/wz-head + latent dequant +
WZ side-info reconstruction + composition).

Per Catalog #205 device selection uses canonical ``select_inflate_device``.

The ATW codec inflate path is structurally distinct from Z3/Z4: it
loads the WZ side-info head + scorer_class_prior_table from archive and
reconstructs ``z = z_residual + side_info_head(class_prior_table[pair])``
before the decoder runs. The scorer class prior table is PRECOMPUTED at
compress-time and shipped in the archive — so the inflate path NEVER
loads SegNet or PoseNet weights.
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
from tac.substrates.atw_codec_v1.architecture import (
    EVAL_HW,
    ATWCodec,
    ATWCodecConfig,
)
from tac.substrates.atw_codec_v1.archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one ATW1 archive's bytes into one contest ``.raw`` file."""
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    render_device = select_inflate_device(device)

    cfg = ATWCodecConfig(
        latent_dim=int(arc.latent_residual.shape[1]),
        encoder_input_channels=int(meta.get("encoder_input_channels", 3)),
        encoder_hidden_dim=int(meta.get("encoder_hidden_dim", 64)),
        decoder_embed_dim=int(meta["decoder_embed_dim"]),
        decoder_initial_grid_h=int(meta["decoder_initial_grid_h"]),
        decoder_initial_grid_w=int(meta["decoder_initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        decoder_num_upsample_blocks=int(meta["decoder_num_upsample_blocks"]),
        num_pairs=int(arc.latent_residual.shape[0]),
        output_height=int(meta.get("output_height", EVAL_HW[0])),
        output_width=int(meta.get("output_width", EVAL_HW[1])),
        scorer_class_prior_dim=int(meta.get("_scorer_class_prior_dim",
                                            arc.scorer_class_prior_table.shape[1])),
        wz_head_hidden_dim=int(meta.get("wz_head_hidden_dim", 32)),
        wz_head_enabled=bool(
            meta.get("atw_codec_meta", {}).get("wz_head_enabled", True)
        ),
        ib_kappa_default=float(
            meta.get("atw_codec_meta", {}).get("kappa_ib", 0.0)
        ),
        wz_lambda_default=float(
            meta.get("atw_codec_meta", {}).get("lambda_wz", 1.0)
        ),
        pixel_lambda_default=float(
            meta.get("atw_codec_meta", {}).get("lambda_pixel", 0.0)
        ),
        atw_atick_redlich_form=bool(
            meta.get("atw_codec_meta", {}).get("atick_redlich_form", True)
        ),
        latent_init_std=float(meta.get("latent_init_std", 0.02)),
    )

    model = ATWCodec(cfg).to(render_device).eval()

    enc_load = model.encoder.load_state_dict(arc.encoder_state_dict, strict=False)
    if set(enc_load.missing_keys) or set(enc_load.unexpected_keys):
        raise RuntimeError(
            "ATW1 encoder state_dict mismatch: "
            f"missing={sorted(enc_load.missing_keys)} "
            f"unexpected={sorted(enc_load.unexpected_keys)}"
        )
    dec_load = model.decoder.load_state_dict(arc.decoder_state_dict, strict=False)
    if set(dec_load.missing_keys) or set(dec_load.unexpected_keys):
        raise RuntimeError(
            "ATW1 decoder state_dict mismatch: "
            f"missing={sorted(dec_load.missing_keys)} "
            f"unexpected={sorted(dec_load.unexpected_keys)}"
        )
    if cfg.wz_head_enabled and arc.wz_side_info_head_state_dict:
        wz_load = model.wz_side_info_head.load_state_dict(
            arc.wz_side_info_head_state_dict, strict=False
        )
        if set(wz_load.missing_keys) or set(wz_load.unexpected_keys):
            raise RuntimeError(
                "ATW1 wz_side_info_head state_dict mismatch: "
                f"missing={sorted(wz_load.missing_keys)} "
                f"unexpected={sorted(wz_load.unexpected_keys)}"
            )

    with torch.no_grad():
        # latents stores z_residual at inflate time
        model.latents.copy_(
            arc.latent_residual.to(device=render_device, dtype=model.latents.dtype)
        )
        # scorer_class_prior_table is the side-info source for WZ reconstruction
        model.scorer_class_prior_table.copy_(
            arc.scorer_class_prior_table.to(
                device=render_device,
                dtype=model.scorer_class_prior_table.dtype,
            )
        )

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_idx in range(cfg.num_pairs):
            idx_tensor = torch.tensor(
                [pair_idx], device=render_device, dtype=torch.long
            )
            # Inflate-time path: reconstruct z = z_residual + WZ_head(class_prior).
            z_residual = model.latents[idx_tensor]
            rgb_0, rgb_1 = model.reconstruct_from_wz_residual(
                idx_tensor, z_residual
            )
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
