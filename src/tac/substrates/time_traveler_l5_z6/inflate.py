# SPDX-License-Identifier: MIT
"""Z6 Time-Traveler L5 inflate runtime — contest raw-output contract.

Loads the Z6PCWM1 archive, reconstructs encoder + decoder + FiLM predictor
from their state_dicts, then iterates pair-by-pair:

    z_0 = latent_init
    for t in 1..T:
        z_t_pred = predictor(z_{t-1}, ego_motion[t])
        z_t = z_t_pred + residuals[t]
        rgb_0, rgb_1 = decoder(z_t)

Writes one raw-output ``.raw`` file per contest video (1200 frames of
874x1164 RGB per video).

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device (Catalog #1; CPU/CUDA only via canonical select_inflate_device).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤120 for
substrate-engineering lanes (Z6 SCAFFOLD explicit waiver).

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
from tac.substrates.time_traveler_l5_z6.architecture import (
    EVAL_HW,
    Z6PredictiveCodingConfig,
    Z6PredictiveCodingSubstrate,
)
from tac.substrates.time_traveler_l5_z6.archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one Z6PCWM1 archive's bytes into one contest ``.raw`` file."""
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    render_device = select_inflate_device(device)

    pcwm_meta = meta.get("predictive_coding_world_model_meta", {})
    cfg = Z6PredictiveCodingConfig(
        latent_dim=int(arc.latent_init.shape[0]),
        encoder_input_channels=int(meta.get("encoder_input_channels", 3)),
        encoder_hidden_dim=int(meta.get("encoder_hidden_dim", 64)),
        decoder_embed_dim=int(meta["decoder_embed_dim"]),
        decoder_initial_grid_h=int(meta["decoder_initial_grid_h"]),
        decoder_initial_grid_w=int(meta["decoder_initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        decoder_num_upsample_blocks=int(meta["decoder_num_upsample_blocks"]),
        num_pairs=int(arc.residuals.shape[0]),
        output_height=int(meta.get("output_height", EVAL_HW[0])),
        output_width=int(meta.get("output_width", EVAL_HW[1])),
        predictor_hidden_dim=int(meta.get("predictor_hidden_dim", 64)),
        predictor_film_mlp_hidden_dim=int(
            meta.get("predictor_film_mlp_hidden_dim", 32)
        ),
        predictor_kernel_size=int(pcwm_meta.get("predictor_kernel_size", 3)),
        predictor_ego_motion_dim=int(arc.ego_motion.shape[1]),
        identity_predictor=bool(pcwm_meta.get("identity_predictor", False)),
        latent_init_std=float(meta.get("latent_init_std", 0.02)),
    )

    model = Z6PredictiveCodingSubstrate(cfg).to(render_device).eval()

    # Load encoder + decoder + predictor state dicts
    for sub_name, sub_mod, sd in (
        ("encoder", model.encoder, arc.encoder_state_dict),
        ("decoder", model.decoder, arc.decoder_state_dict),
        ("predictor", model.predictor, arc.predictor_state_dict),
    ):
        load_res = sub_mod.load_state_dict(sd, strict=False)
        if set(load_res.missing_keys) or set(load_res.unexpected_keys):
            raise RuntimeError(
                f"Z6 Z6PCWM1 {sub_name} state_dict mismatch: "
                f"missing={sorted(load_res.missing_keys)} "
                f"unexpected={sorted(load_res.unexpected_keys)}"
            )

    with torch.no_grad():
        model.latent_init.copy_(
            arc.latent_init.to(device=render_device, dtype=model.latent_init.dtype)
        )
        model.residuals.copy_(
            arc.residuals.to(device=render_device, dtype=model.residuals.dtype)
        )
        model.ego_motion_buffer.copy_(
            arc.ego_motion.to(device=render_device, dtype=model.ego_motion_buffer.dtype)
        )

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_idx in range(cfg.num_pairs):
            idx_tensor = torch.tensor(
                [pair_idx], device=render_device, dtype=torch.long
            )
            rgb_0, rgb_1, _z_t = model.reconstruct_pair(idx_tensor)
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
