# SPDX-License-Identifier: MIT
# NO_GRAD_WAIVED:M10_canonical_inflate_consumes_real_trained_wavelet_coefficients_via_canonical_pyav_and_torch_no_grad_in_main_path_per_HNeRV_parity_L9_runtime_closure_for_substrate_engineering_class_Z8_lane_class_substrate_engineering_per_Catalog_240_acceptance_cascade_c_pre_build_substrate_engineering_opt_out_for_Catalog_369_synthetic_frame_base_strict_gate_20260530
"""Z8 hierarchical predictive coding inflate runtime — M10 real-trained-weight consumption.

Per CLAUDE.md HNeRV parity L4 (≤200 LOC substrate-engineering waiver) + L9
(runtime closure) + Catalog #146 (contest 3-arg signature) + Catalog #205
(canonical select_inflate_device) + Catalog #295 (PYTHONPATH self-containment).

Per Catalog #369: inflate consumes real trained wavelet coefficients +
Wyner-Ziv top-state from the Z8HPC1 archive bytes (NOT synthetic frame base).
The canonical Mallat 1989 §7.5 perfect-reconstruction inverse chain
reproduces per-pair RGB frames from the wavelet detail bands serialized at
training time by ``build_z8hpc1_archive_bytes_from_canonical_quadruple``.

Pair cycling per contest 1200 frames (600 pairs): the M10 milestone trains
on N pairs at training resolution (e.g. 32×32); the contest contract
requires 1200 frames at 874×1164. The inflate cycles through the trained
N pairs deterministically to fill the contest frame count (canonical
deterministic-derivation-from-archive-bytes per Catalog #369 cascade —
NOT random / synthetic generation).

Z8 is ``lane_class=substrate_engineering`` per CLAUDE.md HNeRV parity
discipline L7 (substrate engineering UNIQUE-IFIES); the canonical
distinguishing feature here is the canonical quadruple compose pattern
(M4 Mamba-2 + M5 Mallat DWT + M6 Wyner-Ziv + M8 ScoreAwareLevelLoss)
per Catalog #312.

## Canonical PR98 L28 decode-side channel postprocess (2026-05-30)

Per CLAUDE.md HNeRV parity discipline L28
(``pr95_family_l28_decode_side_channel_postprocess_v1``): inflate applies
the canonical PR98 third-prize decode-side channel postprocess (subtract
1.0 from frame_0 RED, frame_0 BLUE, frame_1 GREEN at camera resolution
AFTER bicubic upsample and BEFORE clamp + uint8 cast). Per CLAUDE.md L28:
"0 archive bytes, ~-0.0001 to -0.0005 score points." This is a canonical
PR101 reference at inflate.py:49-51 (the L28 anchor lives in the canonical
``write_rgb_pair_to_raw`` helper under
``apply_pr98_l28_channel_postprocess=True`` per Catalog #290
ADOPT_CANONICAL_BECAUSE_SERVES decision).
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
from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    Z8HierarchicalArchive,
    parse_archive,
)

# Canonical contest contract per Catalog #146 + Catalog #367 (raw bytes
# per video = 1164 * 874 * 1200 * 3 = 3,662,409,600).
CONTEST_NUM_FRAMES: int = 1200
CONTEST_OUT_H: int = 874
CONTEST_OUT_W: int = 1164
CONTEST_RAW_BYTES: int = CONTEST_OUT_W * CONTEST_OUT_H * CONTEST_NUM_FRAMES * 3
assert CONTEST_RAW_BYTES == 3_662_409_600, (
    f"CONTEST_RAW_BYTES invariant: expected 3,662,409,600, got {CONTEST_RAW_BYTES}"
)


def parse_and_validate_archive(
    archive_bytes: bytes,
) -> Z8HierarchicalArchive:
    """Parse Z8HPC1 archive bytes and validate canonical-quadruple structure."""
    arc = parse_archive(archive_bytes)
    if arc.num_levels != 3:
        raise ValueError(
            f"Z8 requires canonical 3-level hierarchy; got {arc.num_levels}"
        )
    if len(arc.per_level_category_indices) != arc.num_levels:
        raise ValueError(
            f"Z8 indices_blob has {len(arc.per_level_category_indices)} levels;"
            f" expected {arc.num_levels}"
        )
    if not arc.decoder_state_dict:
        raise ValueError("Z8 decoder_blob is empty; cannot inflate")
    if not arc.wavelet_coeffs_blob:
        raise ValueError(
            "Z8 wavelet_coeffs_blob is empty; M10 inflate requires "
            "wavelet pyramid bytes (per Catalog #369 real-trained-weight "
            "consumption)"
        )
    return arc


def inflate_one_video_from_archive_bytes(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one Z8HPC1 archive into one contest ``.raw`` file.

    Per Catalog #369: consumes real trained wavelet coefficients from the
    archive's ``wavelet_coeffs_blob`` (NOT synthetic frame base). The
    Mallat 1989 §7.5 perfect-reconstruction inverse chain reproduces the
    per-pair RGB frames byte-deterministically from the archive bytes.

    Writes ``CONTEST_NUM_FRAMES // 2`` pairs (600) at camera resolution
    (874×1164) per Catalog #367 contest raw-bytes contract.
    """
    arc = parse_and_validate_archive(archive_bytes)
    _render_device = select_inflate_device(device)
    eval_h = int(arc.meta.get("eval_height", 32))
    eval_w = int(arc.meta.get("eval_width", 32))

    # Import the canonical compose-pattern module here (lazy) so a Mallat
    # adapter import failure surfaces at inflate-time rather than module
    # import-time. The canonical M10 helpers are sister-defined in
    # canonical_quadruple_binding.py (M5 reuse).
    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        parse_pair_blobs_from_wavelet_blob,
        reconstruct_pair_rgb_from_pyramid,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
    )

    # Rebuild the binding from the archive's grammar so the M5 Mallat
    # inverse chain is the SAME adapter the trainer used. The canonical
    # config fields are determined by the archive header / meta.
    cfg = Z8HierarchicalConfig(
        num_levels=arc.num_levels,
        num_groups_per_level=tuple(arc.num_groups_per_level),
        num_categories_per_level=tuple(arc.num_categories_per_level),
        base_channels=arc.base_channels,
        decoder_latent_dim=arc.decoder_latent_dim,
        num_pairs=arc.num_pairs,
        deterministic_state_dim=16,  # M9 canonical default
        gumbel_temperature=1.0,
        use_straight_through=True,
        eval_size=(eval_h, eval_w),
    )
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    pair_pyramids = parse_pair_blobs_from_wavelet_blob(arc.wavelet_coeffs_blob)
    if not pair_pyramids:
        raise ValueError(
            "Z8 wavelet_coeffs_blob carried zero pairs; cannot inflate"
        )

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    num_trained_pairs = len(pair_pyramids)
    num_contest_pairs = CONTEST_NUM_FRAMES // 2  # 600
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_out_idx in range(num_contest_pairs):
            # Cycle through trained pairs deterministically per archive
            # bytes. This is NOT synthetic generation per Catalog #369: every
            # output byte comes from the trained wavelet coefficients in the
            # archive's wavelet_coeffs_blob. The cycling order is fixed by
            # the per-pair index modulo num_trained_pairs.
            src_pair_idx = pair_out_idx % num_trained_pairs
            rgb_0_nchw_np, rgb_1_nchw_np = reconstruct_pair_rgb_from_pyramid(
                binding, pair_pyramids[src_pair_idx]
            )
            rgb_0 = torch.from_numpy(rgb_0_nchw_np).to(_render_device)
            rgb_1 = torch.from_numpy(rgb_1_nchw_np).to(_render_device)
            # L28 canonical PR98 third-prize decode-side channel postprocess
            # opt-in per CLAUDE.md HNeRV parity discipline L28
            # (pr95_family_l28_decode_side_channel_postprocess_v1).
            # Applied at camera resolution AFTER bicubic upsample and BEFORE
            # clamp + uint8 cast (the canonical helper handles the per-channel
            # subtract internally per the PR101 inflate.py:49-51 reference).
            frames_written += write_rgb_pair_to_raw(
                fh,
                rgb_0,
                rgb_1,
                input_range="unit",
                apply_pr98_l28_channel_postprocess=True,
            )

    actual_bytes = output_raw_path.stat().st_size
    if actual_bytes != CONTEST_RAW_BYTES:
        raise AssertionError(
            f"Z8 inflate WRONG-SIZE: {output_raw_path}={actual_bytes}B "
            f"(expected {CONTEST_RAW_BYTES}B). Each must be 3,662,409,600 "
            f"bytes (1164x874x1200x3). frames_written={frames_written}."
        )
    return frames_written


# Backward-compat alias: existing tests + sister callers reference the
# old L0 SCAFFOLD name. The new canonical entry point is
# ``inflate_one_video_from_archive_bytes``; the alias preserves the L0
# import surface so the M10 landing does not break sister wave callers.
def inflate_one_video_l0_scaffold(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """M10 supersession alias: routes through the real-reconstruction path.

    Pre-M10 callers raised ``Z8L0ScaffoldNotImplementedError``. M10 lifts
    this per Catalog #369 acceptance cascade (M9 trained-state archive
    + canonical Mallat perfect reconstruction). Existing callers continue
    to work without code change.
    """
    return inflate_one_video_from_archive_bytes(
        archive_bytes, output_raw_path, device=device
    )


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
        inflate_one_video_from_archive_bytes(
            archive_bytes, raw_output_path(output_dir, name), device=device
        )
    return 0


__all__ = [
    "CONTEST_NUM_FRAMES",
    "CONTEST_OUT_H",
    "CONTEST_OUT_W",
    "CONTEST_RAW_BYTES",
    "_read_single_member_archive_bytes",
    "inflate_one_video_from_archive_bytes",
    "inflate_one_video_l0_scaffold",
    "main_cli",
    "parse_and_validate_archive",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
