# SPDX-License-Identifier: MIT
"""Z5 Rao-Ballard L5 inflate runtime — contest raw-output contract.

Loads Z5RB1 archive bytes, reconstructs decoder + predictor + low_latents +
high_latents + ego_vecs, then iterates pair-by-pair:

    z_low_t = arc.low_latents[t]      (stored per-pair)
    rgb_0, rgb_1 = decoder(z_low_t)

The predictor + high_latents + ego_vecs blobs MUST be parsed (Catalog #105
no-op detector + Catalog #220 operational mechanism); their consumption is
implicit through the residual-savings mechanism at the encoder side (which
mints low_latents from the predictor's forecast). The SCAFFOLD ships the
direct low_latents so the inflate runtime stays under the HNeRV L4 budget;
the canonical entropy-coding pass would store residuals instead per Catalog
#344 equation ``z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1``.

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device (Catalog #1) — uses canonical ``select_inflate_device``.
Honors the contest 3-positional-arg ``inflate.sh <archive_dir> <output_dir>
<file_list>`` contract per Catalog #146.

Per HNeRV parity L4 the inflate runtime LOC budget is <=200 for
substrate-engineering lanes (Z5 SCAFFOLD explicit waiver).

Per Catalog #205 device selection uses canonical ``select_inflate_device``.
Per Catalog #367 fail-closes raw_bytes != 1164*874*1200*3 contract.
"""
# NO_GRAD_WAIVED:inflate_uses_torch_no_grad_at_render_time_per_canonical_pattern
from __future__ import annotations

import sys
from pathlib import Path

import torch

from tac.substrates._shared.inflate_runtime import (
    raw_output_path,
    select_inflate_device,
    write_rgb_pair_to_raw,
)
from tac.substrates.time_traveler_l5_z5.architecture import (
    EVAL_HW,
    Z5RaoBallardConfig,
    Z5RaoBallardSubstrate,
)
from tac.substrates.time_traveler_l5_z5.archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one Z5RB1 archive's bytes into one contest ``.raw`` file."""
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    render_device = select_inflate_device(device)

    z5_meta = meta.get("z5_rao_ballard_meta", {})
    if isinstance(z5_meta, bool):  # legacy boolean sentinel
        z5_meta = {}

    num_pairs = int(arc.low_latents.shape[0])
    low_latent_dim = int(arc.low_latents.shape[1])
    high_latent_dim = int(arc.high_latents.shape[1])
    ego_dim = int(arc.ego_vecs.shape[1])

    cfg = Z5RaoBallardConfig(
        low_latent_dim=low_latent_dim,
        high_latent_dim=high_latent_dim,
        ego_dim=ego_dim,
        embed_dim=int(meta.get("embed_dim", 32)),
        initial_grid_h=int(meta.get("initial_grid_h", 3)),
        initial_grid_w=int(meta.get("initial_grid_w", 4)),
        decoder_channels=tuple(
            int(c) for c in meta.get("decoder_channels", (24, 20, 16, 12, 8, 6, 4))
        ),
        num_upsample_blocks=int(meta.get("num_upsample_blocks", 7)),
        sin_frequency=float(meta.get("sin_frequency", 30.0)),
        film_generator_depth=int(meta.get("film_generator_depth", 3)),
        film_hidden_width=int(meta.get("film_hidden_width", 24)),
        num_pairs=num_pairs,
        output_height=int(meta.get("output_height", EVAL_HW[0])),
        output_width=int(meta.get("output_width", EVAL_HW[1])),
        predictor_hidden_dim=int(meta.get("predictor_hidden_dim", 48)),
        predictor_num_layers=int(meta.get("predictor_num_layers", 2)),
    )

    model = Z5RaoBallardSubstrate(cfg).to(render_device).eval()

    # Load decoder + predictor state_dicts. Predictor MUST load so the no-op
    # detector (Catalog #105/#272) can prove byte-mutation propagation.
    for sub_name, sub_mod, sd in (
        ("decoder", model.decoder, arc.decoder_state_dict),
        ("predictor", model.predictor, arc.predictor_state_dict),
    ):
        load_res = sub_mod.load_state_dict(sd, strict=False)
        if set(load_res.missing_keys) or set(load_res.unexpected_keys):
            raise RuntimeError(
                f"Z5 Z5RB1 {sub_name} state_dict mismatch: "
                f"missing={sorted(load_res.missing_keys)} "
                f"unexpected={sorted(load_res.unexpected_keys)}"
            )

    with torch.no_grad():
        model.low_latents.copy_(
            arc.low_latents.to(device=render_device, dtype=model.low_latents.dtype)
        )
        model.high_latents.copy_(
            arc.high_latents.to(device=render_device, dtype=model.high_latents.dtype)
        )
        model.ego_vecs.copy_(
            arc.ego_vecs.to(device=render_device, dtype=model.ego_vecs.dtype)
        )

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_idx in range(cfg.num_pairs):
            idx_tensor = torch.tensor(
                [pair_idx], device=render_device, dtype=torch.long
            )
            rgb_0, rgb_1, _residual = model.reconstruct_pair(idx_tensor)
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
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>`` per Catalog #146."""
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
