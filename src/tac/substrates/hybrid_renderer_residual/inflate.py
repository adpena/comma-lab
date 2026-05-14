# SPDX-License-Identifier: MIT
"""hybrid_renderer_residual inflate runtime — <= 200 LOC waiver per HNeRV L4 NEEDS-WORK (γ).

This file is the contest-runtime image of the γ substrate. It is imported by
``submissions/hybrid_renderer_residual/inflate.py`` (one-line passthrough)
at packet-build time. The whole forward path is:

1. Read the archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> ``HybridRendererResidualArchive``.
3. Build the substrate from ``meta`` (no training; deterministic).
4. Load renderer + residual_decoder state_dicts; copy latents AND set the
   sparse residual coefficients at the archived (index, value) positions.
5. For each pair index i in [0, num_pairs): render (rgb_0, rgb_1) =
   renderer + residual; write
   ``output_dir/<base>/<frame_idx>.png``.

L4 budget: <= 200 LOC waiver (council §4.2 γ NEEDS-WORK note: residual
decode adds ~30 LOC over α's 80 LOC). Target ~140 LOC. <= 2 external deps:
``torch`` + ``brotli`` (numpy is the torch transitive). Catalog #146 contract
(<inflate.sh archive_dir output_dir file_list> 3 positional args).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .archive import parse_archive
from .architecture import (
    HybridRendererResidualConfig,
    HybridRendererResidualSubstrate,
)


def _build_full_residual_coeff_matrix(
    indices: torch.Tensor,
    values_int16: torch.Tensor,
    *,
    num_pairs: int,
    basis_dim: int,
    scale: float,
    zero_point: float,
) -> torch.Tensor:
    """Reconstruct the dense (num_pairs, basis_dim) residual coefficient matrix
    by scatter-add of (index, dequantized_value) pairs.
    """
    full = torch.zeros((num_pairs, basis_dim), dtype=torch.float32)
    # Dequantize int16 values -> float
    val_float = (values_int16.to(torch.float32) + 32767.0) * scale + zero_point
    # Scatter values into their indices, per pair
    full.scatter_(1, indices.to(torch.int64), val_float)
    return full


def inflate_one_video(
    archive_bytes: bytes,
    output_dir: Path,
    *,
    device: str = "cpu",
) -> None:
    """Inflate one archive's bytes into ``output_dir/<frame_idx>.png`` files."""
    arc = parse_archive(archive_bytes)
    meta = arc.meta

    cfg = HybridRendererResidualConfig(
        latent_dim=int(arc.latents.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        residual_basis_dim=int(meta["residual_basis_dim"]),
        residual_basis_value_dim=int(meta["residual_basis_value_dim"]),
        residual_coeffs_per_pair=int(meta["residual_coeffs_per_pair"]),
        residual_decoder_hidden=tuple(int(c) for c in meta["residual_decoder_hidden"]),
    )

    model = HybridRendererResidualSubstrate(cfg).to(device).eval()

    # Load renderer + residual_decoder state_dicts together (strict=False
    # so the per-pair learnables aren't required).
    merged: dict[str, torch.Tensor] = {}
    merged.update(arc.renderer_state_dict)
    merged.update(
        {"residual_decoder." + k: v for k, v in arc.residual_decoder_state_dict.items()}
    )
    model.load_state_dict(merged, strict=False)

    with torch.no_grad():
        model.latents.copy_(arc.latents.to(device=device, dtype=model.latents.dtype))

        # Reconstruct the dense residual coefficient matrix from
        # the archived sparse (index, value) pairs.
        rescoeffs = arc.residual_basis_coefficients
        indices = rescoeffs[:, :, 0]
        values_int16 = rescoeffs[:, :, 1].to(torch.int16)
        full = _build_full_residual_coeff_matrix(
            indices,
            values_int16,
            num_pairs=cfg.num_pairs,
            basis_dim=cfg.residual_basis_dim,
            scale=float(meta["residual_quant_scale"]),
            zero_point=float(meta["residual_quant_zero_point"]),
        )
        model.residual_coeff_full.copy_(full.to(device=device))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-import PIL inside the function to keep this module's import light
    from PIL import Image  # type: ignore[import-not-found]

    with torch.no_grad():
        for pair_idx in range(cfg.num_pairs):
            idx_tensor = torch.tensor([pair_idx], device=device, dtype=torch.long)
            rgb_0, rgb_1, _l1 = model(idx_tensor)
            for off, rgb in ((0, rgb_0), (1, rgb_1)):
                frame_idx = 2 * pair_idx + off
                arr = (rgb[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0)
                arr = arr.round().clip(0, 255).astype("uint8")
                Image.fromarray(arr).save(output_dir / f"{frame_idx}.png")


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``.

    Honors the contest's 3-positional-arg inflate.sh contract per Catalog #146.
    """
    if len(sys.argv) < 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])

    file_list = file_list_path.read_text(encoding="utf-8").strip().splitlines()
    for fname in file_list:
        base = Path(fname).stem  # "0" from "0.mkv"
        archive_bytes = (archive_dir / "0.bin").read_bytes()
        inflate_one_video(archive_bytes, output_dir / base, device="cpu")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
