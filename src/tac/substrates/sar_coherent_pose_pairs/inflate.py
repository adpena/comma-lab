"""SAR coherent pose-pair substrate inflate runtime — contest raw-output.

Loads the SARC archive, rebuilds the renderer + pose codec state, decodes
per-pair pose codes from sparse rFFT bytes, and renders RGB pairs for every
contest pair index. Writes one ``.raw`` file per requested video matching
the contest's ``(1200, 874, 1164, 3)`` uint8 layout.

NO scorer code is imported per CLAUDE.md "Strict scorer rule —
non-negotiable" + Catalog #6. NO MPS device (Catalog #1; CPU/CUDA only) —
device selection ALWAYS routes through ``select_inflate_device``.

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤ 200 for
substrate-engineering lanes (this is a full renderer + SAR-coherent pose
decoder + per-pair residual; not a thin codec).
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
from tac.substrates.sar_coherent_pose_pairs.architecture import (
    SARCoherentConfig,
    SARCoherentSubstrate,
)
from tac.substrates.sar_coherent_pose_pairs.archive import (
    SARCoherentArchive,
    decode_pose_codec_bytes,
    dequantize_per_pair_residual,
    parse_archive,
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


def _build_substrate_from_archive(
    arc: SARCoherentArchive, *, device: str
) -> SARCoherentSubstrate:
    """Rebuild the substrate from header + meta + load state_dict."""
    meta = arc.meta
    cfg = SARCoherentConfig(
        hidden_dim=arc.hidden_dim,
        num_hidden_layers=arc.num_hidden_layers,
        pose_dim=arc.pose_dim,
        pose_code_dim=arc.pose_code_dim,
        per_pair_residual_bytes=arc.per_pair_residual_bytes,
        first_omega=float(meta.get("first_omega", 30.0)),
        hidden_omega=float(meta.get("hidden_omega", 1.0)),
        coord_feature_freqs=int(meta.get("coord_feature_freqs", 4)),
        sar_topk_keep_fraction=float(meta.get("sar_topk_keep_fraction", 0.10)),
        sar_int16_scale=float(meta.get("sar_int16_scale", 256.0)),
        num_pairs=arc.num_pairs,
        output_height=arc.output_height,
        output_width=arc.output_width,
    )
    substrate = SARCoherentSubstrate(cfg).to(device).eval()
    # Load only the renderer state_dict; pose codec deltas are reconstructed
    # from the sparse rFFT bytes below.
    renderer_sd = {
        k.removeprefix("renderer."): v
        for k, v in arc.renderer_state_dict.items()
        if k.startswith("renderer.")
    }
    incompat = substrate.renderer.load_state_dict(renderer_sd, strict=False)
    missing = set(incompat.missing_keys)
    unexpected = set(incompat.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "SARC archive renderer state_dict mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return substrate


def _rebuild_pose_codes_from_archive(
    arc: SARCoherentArchive, *, device: str
) -> torch.Tensor:
    """Decode sparse rFFT bytes back into ``(num_pairs, pose_code_dim)`` codes."""
    n_rfft_bins = arc.num_pairs // 2 + 1
    int16_scale = float(arc.meta.get("sar_int16_scale", 256.0))
    sparse_coeffs = decode_pose_codec_bytes(
        arc.pose_codec_bytes,
        n_rfft_bins=n_rfft_bins,
        pose_dim=arc.pose_dim,
        int16_scale=int16_scale,
    ).to(device)
    pose_codes = torch.fft.irfft(sparse_coeffs, n=arc.num_pairs, dim=0)
    return pose_codes  # (num_pairs, pose_dim) == (num_pairs, pose_code_dim)


def _apply_per_pair_rgb_residual(
    rgb_0: torch.Tensor,
    rgb_1: torch.Tensor,
    residual_pair: torch.Tensor,
    *,
    int8_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-pair int8-quantized RGB residual as additive correction.

    The residual bytes (default 50 B/pair) are interpreted as a small spatial
    RGB delta. Layout: bytes are reshaped to a small (3, H_low, W_low) tile
    then bilinear upsampled to (3, H, W) and added to BOTH frames of the pair.
    """
    import torch.nn.functional as F

    residual_floats = residual_pair / int8_scale
    n_bytes = residual_floats.shape[0]
    # Reshape to (3, H_low, W_low) where H_low * W_low ≈ n_bytes / 3.
    n_per_channel = n_bytes // 3
    side = max(1, int(round(n_per_channel**0.5)))
    flat = residual_floats[: 3 * side * side]
    if flat.shape[0] < 3:
        # Edge case: trivially small residual; skip correction.
        return rgb_0.clamp(0.0, 1.0), rgb_1.clamp(0.0, 1.0)
    correction = flat.view(3, side, side).unsqueeze(0)
    correction_full = F.interpolate(
        correction,
        size=(rgb_0.shape[-2], rgb_0.shape[-1]),
        mode="bilinear",
        align_corners=False,
    )
    # Apply gently; correction magnitude bounded by ~1/scale ~ 0.015 so the
    # delta is at most ~4 gray levels (out of 255).
    rgb_0_c = (rgb_0 + 0.02 * correction_full).clamp(0.0, 1.0)
    rgb_1_c = (rgb_1 + 0.02 * correction_full).clamp(0.0, 1.0)
    return rgb_0_c, rgb_1_c


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one SARC archive's bytes into one contest ``.raw`` file.

    Returns the number of frames written.
    """
    arc = parse_archive(archive_bytes)
    render_device = select_inflate_device(device)
    substrate = _build_substrate_from_archive(arc, device=render_device)

    int8_scale = float(arc.meta.get("int8_scale", 64.0))
    pose_codes = _rebuild_pose_codes_from_archive(arc, device=render_device)
    residual_tensor = torch.from_numpy(arc.per_pair_residual).to(render_device).float()

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    H, W = arc.output_height, arc.output_width
    coord_grid_xy = substrate._build_coord_grid(torch.device(render_device))

    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_idx in range(arc.num_pairs):
            t = pair_idx / max(1, arc.num_pairs - 1)
            t_col = torch.full((H * W, 1), t, device=render_device, dtype=coord_grid_xy.dtype)
            coords = torch.cat([coord_grid_xy, t_col], dim=-1)
            pose_code = pose_codes[pair_idx]
            out6 = substrate.renderer(coords, pose_code)
            rgb6 = out6.t().reshape(1, 6, H, W)
            rgb_0 = rgb6[:, :3]
            rgb_1 = rgb6[:, 3:]
            rgb_0_c, rgb_1_c = _apply_per_pair_rgb_residual(
                rgb_0, rgb_1, residual_tensor[pair_idx], int8_scale=int8_scale
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


# Re-exported for trainer / test imports.
__all__ = [
    "_read_single_member_archive_bytes",
    "inflate_one_video",
    "main_cli",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())


# Auxiliary helper exposed for unit tests; not used at inflate-CLI time.
def _dequant_passthrough(side_info, scale: float):  # pragma: no cover — thin shim
    return dequantize_per_pair_residual(side_info, scale=scale)
