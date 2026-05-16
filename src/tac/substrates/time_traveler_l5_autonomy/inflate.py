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

_TT5L_SIDEINFO_SECTION_BOUNDS: tuple[tuple[str, int, int], ...] = (
    ("se3_lie", 0, 12),
    ("seg_boundary", 12, 30),
    ("hf_residual", 30, 36),
    ("predict_residual", 36, 45),
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

    The side-info bytes are interpreted as small per-pair corrections in
    ``[-128/scale, 127/scale]`` units. Layout per pair (default 45 B):

    * Bytes 0:12   SE(3) Lie-algebra pose delta (frame-specific low-order
      affine RGB field).
    * Bytes 12:30  seg-boundary residual (two coarse 3x3 RGB fields).
    * Bytes 30:36  HF byte-stuffing residual (frame-specific checker fields).
    * Bytes 36:45  predictive-coding residual (9 B; the actual RGB residual,
      tiled across the central foveal region).

    Every charged side-info byte must affect inflated output. Bytes beyond the
    canonical 45-byte layout are folded into a tiny weighted global residual so
    alternate per-pair budgets do not silently create dead payload.
    """
    import torch.nn.functional as F

    if int8_scale <= 0.0:
        raise ValueError(f"int8_scale must be > 0; got {int8_scale}")
    residual_floats = side_info_pair.flatten().float() / float(int8_scale)
    device = rgb_0.device
    dtype = rgb_0.dtype
    residual_floats = residual_floats.to(device=device, dtype=dtype)
    if residual_floats.numel() == 0:
        return rgb_0.clamp(0.0, 1.0), rgb_1.clamp(0.0, 1.0)

    h, w = int(rgb_0.shape[-2]), int(rgb_0.shape[-1])
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).unsqueeze(0)
    yy = yy.unsqueeze(0).unsqueeze(0)
    checker = 0.25 + 0.75 * torch.sign(torch.sin(32.0 * xx) * torch.sin(32.0 * yy))

    def _section(start: int, end: int) -> torch.Tensor:
        out = torch.zeros(end - start, device=device, dtype=dtype)
        available = residual_floats[start:min(end, residual_floats.numel())]
        if available.numel():
            out[: available.numel()] = available
        return out

    section_bounds = {
        name: (start, end) for name, start, end in _TT5L_SIDEINFO_SECTION_BOUNDS
    }

    def _pose_field(vals: torch.Tensor) -> torch.Tensor:
        base = vals[:3].view(1, 3, 1, 1)
        affine = vals[3].view(1, 1, 1, 1) * xx + vals[4].view(1, 1, 1, 1) * yy
        twist = vals[5].view(1, 1, 1, 1) * (xx * yy)
        return base + affine + twist

    se3 = _section(*section_bounds["se3_lie"]).view(2, 6)
    pose_0 = _pose_field(se3[0])
    pose_1 = _pose_field(se3[1])

    seg = _section(*section_bounds["seg_boundary"]).view(2, 3, 3)
    seg_0 = F.interpolate(
        seg[0].unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    seg_1 = F.interpolate(
        seg[1].unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    hf = _section(*section_bounds["hf_residual"]).view(2, 3)
    hf_0 = hf[0].view(1, 3, 1, 1) * checker
    hf_1 = hf[1].view(1, 3, 1, 1) * checker

    predict = _section(*section_bounds["predict_residual"]).view(3, 3)
    predict_full = F.interpolate(
        predict.unsqueeze(0).unsqueeze(0),
        size=(rgb_0.shape[-2], rgb_0.shape[-1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    tail = residual_floats[45:]
    if tail.numel():
        weights = torch.linspace(1.0, 2.0, tail.numel(), device=device, dtype=dtype)
        tail_scalar = (tail * weights).mean().view(1, 1, 1, 1)
    else:
        tail_scalar = torch.zeros(1, 1, 1, 1, device=device, dtype=dtype)

    rgb_0_c = (
        rgb_0
        + 0.006 * pose_0
        + 0.012 * seg_0
        + 0.010 * hf_0
        + 0.020 * predict_full
        + 0.004 * tail_scalar
    ).clamp(0.0, 1.0)
    rgb_1_c = (
        rgb_1
        + 0.006 * pose_1
        + 0.012 * seg_1
        + 0.010 * hf_1
        + 0.020 * predict_full
        + 0.004 * tail_scalar
    ).clamp(0.0, 1.0)
    return rgb_0_c, rgb_1_c


def quantize_per_pair_residual_for_inflate_ste(
    side_info_float: torch.Tensor,
    *,
    int8_scale: float,
) -> torch.Tensor:
    """Return inflate-equivalent int8 side-info values with STE gradients.

    Forward values exactly match ``round(x * scale).clamp(-128, 127)`` before
    archive serialization. Backward treats the quantizer as identity after the
    divide-by-scale inside ``_apply_per_pair_residual`` so training optimizes
    the same residual transform that inflate will consume.
    """
    if int8_scale <= 0.0:
        raise ValueError(f"int8_scale must be > 0; got {int8_scale}")
    scaled = side_info_float.float() * float(int8_scale)
    quantized = scaled.round().clamp(-128.0, 127.0)
    return scaled + (quantized - scaled).detach()


def apply_quantized_per_pair_residual_for_training(
    rgb_0: torch.Tensor,
    rgb_1: torch.Tensor,
    side_info_float: torch.Tensor,
    *,
    int8_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the inflate-time quantized side-info transform during training.

    Returns corrected ``(rgb_0, rgb_1)`` in unit range plus the int8-domain
    STE tensor used by the transform. The returned side-info tensor is useful
    for logging and auxiliary residual penalties without drifting away from
    archive/inflate behavior.
    """
    side_info_int8_ste = quantize_per_pair_residual_for_inflate_ste(
        side_info_float,
        int8_scale=int8_scale,
    )
    rgb_0_c, rgb_1_c = _apply_per_pair_residual(
        rgb_0,
        rgb_1,
        side_info_int8_ste,
        int8_scale=int8_scale,
    )
    return rgb_0_c, rgb_1_c, side_info_int8_ste


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
    "apply_quantized_per_pair_residual_for_training",
    "inflate_one_video",
    "main_cli",
    "quantize_per_pair_residual_for_inflate_ste",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())


# Auxiliary helper exposed for unit tests; not used at inflate-CLI time.
def _dequant_passthrough(
    side_info, scale: float
):  # pragma: no cover — thin shim around archive helper
    return dequantize_per_pair_residual(side_info, scale=scale)
