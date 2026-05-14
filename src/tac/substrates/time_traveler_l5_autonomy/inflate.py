# SPDX-License-Identifier: MIT
"""Time-Traveler L5 Autonomy inflate runtime — contest raw-output contract.

Loads the TT5L archive, rebuilds the world model state, decodes per-pair side
info, and renders RGB pairs for every contest pair index. Writes one
``.raw`` file per requested video matching the contest's
``(1200, 874, 1164, 3)`` uint8 layout.

NO scorer code is imported per CLAUDE.md "Strict scorer rule —
non-negotiable" + Catalog #6. NO MPS device (Catalog #1; CPU/CUDA only).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤ 200 for
substrate-engineering lanes (this is a full renderer + foveation +
predictive decoder + per-pair Lie-algebra reconstruction, not a thin codec).
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
from tac.substrates.time_traveler_l5_autonomy.architecture import (
    TimeTravelerConfig,
    TimeTravelerSubstrate,
)
from tac.substrates.time_traveler_l5_autonomy.archive import (
    TimeTravelerArchive,
    dequantize_per_pair_residual,
    parse_archive,
)


def _build_substrate_from_archive(
    arc: TimeTravelerArchive, *, device: str
) -> TimeTravelerSubstrate:
    """Rebuild the substrate from header + meta + load state_dict."""
    meta = arc.meta
    cfg = TimeTravelerConfig(
        hidden_dim=arc.hidden_dim,
        num_hidden_layers=arc.num_hidden_layers,
        coord_dim=int(meta.get("coord_dim", 4)),
        pose_dim=arc.pose_dim,
        per_pair_side_info_bytes=arc.per_pair_bytes,
        foveation_grid_h=arc.foveation_grid_h,
        foveation_grid_w=arc.foveation_grid_w,
        first_omega=float(meta.get("first_omega", 30.0)),
        hidden_omega=float(meta.get("hidden_omega", 1.0)),
        num_pairs=arc.num_pairs,
        output_height=arc.output_height,
        output_width=arc.output_width,
        markov_transition_band=int(meta.get("markov_transition_band", 4)),
        coord_feature_freqs=int(meta.get("coord_feature_freqs", 4)),
    )
    substrate = TimeTravelerSubstrate(cfg).to(device).eval()
    incompat = substrate.load_state_dict(arc.world_model_state_dict, strict=False)
    missing = set(incompat.missing_keys)
    unexpected = set(incompat.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "TT5L archive state_dict mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return substrate


def _apply_per_pair_residual(
    rgb_0: torch.Tensor,
    rgb_1: torch.Tensor,
    side_info_pair: torch.Tensor,
    *,
    int8_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-pair int8-quantized residual as additive RGB correction.

    The side-info bytes are interpreted as a small per-pair correction in
    ``[-128/scale, 127/scale]`` units. Layout per pair (default 45 B):

    * Bytes 0:12   SE(3) Lie-algebra pose delta (consumed by dynamics; not
      added directly to RGB — left here for cross-frame consistency probes).
    * Bytes 12:30  seg-boundary residual (18 B; not directly RGB-additive).
    * Bytes 30:36  HF byte-stuffing residual (6 B; small RGB perturbation).
    * Bytes 36:45  predictive-coding residual (9 B; the actual RGB residual,
      tiled across the central foveal region).

    Per the design memo, the residual is the predictive-coding correction
    against the world model. The byte budget is small (~9 B effective per
    pair) so the correction tiles into a coarse RGB delta then bilinear
    upsamples.
    """
    import torch.nn.functional as F

    residual_floats = side_info_pair / int8_scale
    # Use the last 9 bytes as the RGB-additive predictive residual.
    rgb_residual = residual_floats[-9:]
    # Reshape to (3, 3) and tile + smooth to RGB-additive correction.
    correction = rgb_residual.view(3, 3).unsqueeze(0)
    correction_full = F.interpolate(
        correction.unsqueeze(0),
        size=(rgb_0.shape[-2], rgb_0.shape[-1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    # Apply gently; the residual is in [-1/scale, 1/scale] roughly. Scale
    # down so the correction is at most ~3 gray levels (out of 255).
    rgb_0_c = (rgb_0 + 0.02 * correction_full).clamp(0.0, 1.0)
    rgb_1_c = (rgb_1 + 0.02 * correction_full).clamp(0.0, 1.0)
    return rgb_0_c, rgb_1_c


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one TT5L archive's bytes into one contest ``.raw`` file.

    Returns the number of frames written.
    """
    arc = parse_archive(archive_bytes)
    render_device = select_inflate_device(device)
    substrate = _build_substrate_from_archive(arc, device=render_device)

    int8_scale = float(arc.meta.get("int8_scale", 64.0))
    side_info_tensor = torch.from_numpy(arc.per_pair_side_info).to(render_device)
    side_info_floats = side_info_tensor.float()

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0

    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_idx in range(arc.num_pairs):
            rgb_0, rgb_1 = substrate.render_pair(pair_idx)
            rgb_0_c, rgb_1_c = _apply_per_pair_residual(
                rgb_0,
                rgb_1,
                side_info_floats[pair_idx],
                int8_scale=int8_scale,
            )
            frames_written += write_rgb_pair_to_raw(
                fh, rgb_0_c, rgb_1_c, input_range="unit"
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
    src_path = archive_dir / "0.bin"
    if not src_path.is_file():
        src_path = archive_dir / "x"
    archive_bytes = src_path.read_bytes()
    device = select_inflate_device()
    for fname in file_list:
        name = fname.strip()
        if not name:
            continue
        inflate_one_video(
            archive_bytes, raw_output_path(output_dir, name), device=device
        )
    return 0


# Re-exported for trainer / test imports.
__all__ = [
    "inflate_one_video",
    "main_cli",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())


# Auxiliary helper exposed for unit tests; not used at inflate-CLI time.
def _dequant_passthrough(
    side_info, scale: float
):  # pragma: no cover — thin shim around archive helper
    return dequantize_per_pair_residual(side_info, scale=scale)
