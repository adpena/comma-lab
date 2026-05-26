# SPDX-License-Identifier: MIT
"""faiss_ivf_pq_residual inflate runtime — contest raw-output contract.

Loads the FAISSPQ1 archive, reconstructs per-pair RGB residual via PQ
codebook gather + tile reassemble, adds residual to PR110 fec6 frontier
reconstruction (which MUST be available at inflate time as upstream input
or sibling-archive composition), and writes one raw-output ``.raw`` file
per contest video (1200 frames of 874×1164 RGB per video).

L0 SCAFFOLD scope: this inflate path declares the contest contract. For
standalone (non-PR110-stacked) testing, the residual ADDS to a zero
baseline (passthrough). Phase 5 composition smoke wires the PR110 fec6
frontier reconstruction as upstream sibling-archive input.

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device (Catalog #1; CPU/CUDA only via canonical select_inflate_device).
NO MLX at inflate (runtime_dep_closure = torch + brotli + numpy only per
HNeRV parity L4).

Per Catalog #146 honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤200 for
substrate-engineering lanes (PQ codebook gather + tile reassemble +
bilinear upsample to 384×512 + bicubic upscale to 874×1164 + uint8 cast).

Per Catalog #205 device selection uses canonical ``select_inflate_device``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tac.substrates._shared.inflate_runtime import (
    raw_output_path,
    select_inflate_device,
)
from tac.substrates.faiss_ivf_pq_residual.archive import parse_archive

# Camera resolution required by upstream/evaluate.py contest harness.
CAMERA_H: int = 874
CAMERA_W: int = 1164

# Contest scorer-resolution (matches MLX module EVAL_HW).
DECODER_H_FINAL: int = 384
DECODER_W_FINAL: int = 512

# Frame count per contest video.
FRAMES_PER_VIDEO: int = 1200


def _decode_per_pair_residual(
    codebook: np.ndarray,
    codewords: np.ndarray,
    *,
    tile_h: int,
    tile_w: int,
    residual_scale: float,
) -> np.ndarray:
    """PQ decode + tile reassemble for one pair.

    Args:
        codebook: shape (M, ksub, sub_dim) float32 (shared across pairs)
        codewords: shape (tiles_per_pair, M) uint16 (per-pair indices)
        tile_h: per-tile height
        tile_w: per-tile width
        residual_scale: per-pair residual scale factor

    Returns:
        Per-pair RGB residual shape (DECODER_H_FINAL, DECODER_W_FINAL, 3)
        float32 in [-residual_scale, residual_scale].
    """
    M, ksub, sub_dim = codebook.shape
    tiles_per_pair = codewords.shape[0]
    tile_dim = M * sub_dim
    # Gather per-tile flat vectors from PQ codebook
    tile_vectors = np.zeros((tiles_per_pair, tile_dim), dtype=np.float32)
    for m in range(M):
        m_indices = codewords[:, m]  # (tiles_per_pair,)
        tile_vectors[:, m * sub_dim:(m + 1) * sub_dim] = codebook[m, m_indices, :]
    # Reassemble tiles into frame NHWC
    grid_h = DECODER_H_FINAL // tile_h
    grid_w = DECODER_W_FINAL // tile_w
    assert grid_h * grid_w == tiles_per_pair, (
        f"tiles_per_pair {tiles_per_pair} != grid_h*grid_w {grid_h * grid_w}"
    )
    frame = np.zeros((DECODER_H_FINAL, DECODER_W_FINAL, 3), dtype=np.float32)
    for g in range(tiles_per_pair):
        gi = g // grid_w
        gj = g % grid_w
        tile_3d = tile_vectors[g].reshape(tile_h, tile_w, 3)
        frame[gi * tile_h:(gi + 1) * tile_h, gj * tile_w:(gj + 1) * tile_w, :] = tile_3d
    # Scale to residual range
    return frame * float(residual_scale)


def inflate_one_video(
    archive_path: Path,
    output_path: Path,
    *,
    device: torch.device,
) -> None:
    """Inflate one video's worth of frames from a FAISSPQ1 archive.

    L0 SCAFFOLD posture: residual is decoded standalone (no PR110 fec6 frontier
    upstream wiring). Phase 5 composition smoke adds the PR110 frontier
    reconstruction as sibling-archive input before residual addition.
    """
    archive_bytes = archive_path.read_bytes()
    arch = parse_archive(archive_bytes)

    residual_scale = float(arch.meta.get("residual_scale", 0.5))
    num_pairs = arch.num_pairs

    # Pre-decode all per-pair residuals (CPU; deterministic)
    per_pair_frames_384x512 = np.zeros(
        (num_pairs, DECODER_H_FINAL, DECODER_W_FINAL, 3), dtype=np.float32
    )
    for pair_idx in range(num_pairs):
        per_pair_frames_384x512[pair_idx] = _decode_per_pair_residual(
            arch.codebook,
            arch.per_pair_codewords[pair_idx],
            tile_h=arch.tile_h,
            tile_w=arch.tile_w,
            residual_scale=residual_scale,
        )

    # Frame expansion: 600 pairs → 1200 frames (duplicate each pair to 2 frames
    # per canonical PR110 pair-to-frame mapping)
    if num_pairs * 2 != FRAMES_PER_VIDEO:
        raise ValueError(
            f"num_pairs={num_pairs} expected to map to {FRAMES_PER_VIDEO} frames "
            f"via 2× duplication"
        )
    frames_384x512 = np.repeat(per_pair_frames_384x512, 2, axis=0)

    # Upscale to camera resolution via torch bicubic per canonical contest pipeline
    frames_torch = torch.from_numpy(frames_384x512).to(device=device, dtype=torch.float32)
    # NHWC → NCHW for F.interpolate
    frames_nchw = frames_torch.permute(0, 3, 1, 2)
    camera_nchw = F.interpolate(
        frames_nchw,
        size=(CAMERA_H, CAMERA_W),
        mode="bicubic",
        align_corners=False,
    )
    # NCHW → NHWC for raw byte layout
    camera_nhwc = camera_nchw.permute(0, 2, 3, 1).contiguous()

    # Standalone L0: residual is added to ZERO baseline (passthrough).
    # Phase 5 composition wires PR110 frontier reconstruction here.
    # uint8 cast per canonical Catalog #205 sister rounding.
    camera_uint8 = torch.clamp(torch.round(camera_nhwc), 0, 255).to(torch.uint8)

    # Write raw bytes (NHWC layout per contest evaluate.py)
    output_bytes = camera_uint8.cpu().numpy().tobytes(order="C")
    expected_bytes = FRAMES_PER_VIDEO * CAMERA_H * CAMERA_W * 3
    if len(output_bytes) != expected_bytes:
        raise RuntimeError(
            f"raw output bytes {len(output_bytes)} != expected {expected_bytes}"
        )
    output_path.write_bytes(output_bytes)


def main(argv: list[str] | None = None) -> int:
    """Inflate entry point per Catalog #146 3-positional-arg contest contract.

    Usage: inflate.py <archive_dir> <output_dir> <file_list>
    """
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 3:
        print(
            f"usage: inflate.py <archive_dir> <output_dir> <file_list>; got {len(args)} args",
            file=sys.stderr,
        )
        return 1
    archive_dir = Path(args[0])
    output_dir = Path(args[1])
    file_list = Path(args[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_inflate_device()
    with file_list.open("r") as f:
        for line in f:
            video_name = line.strip()
            if not video_name:
                continue
            archive_path = archive_dir / "0.bin"  # canonical FAISSPQ1 archive
            output_path = raw_output_path(output_dir, video_name)
            inflate_one_video(archive_path, output_path, device=device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
