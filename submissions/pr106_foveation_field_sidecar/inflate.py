#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate PR106 r2 + telescopic foveation field sidecar (research scaffold).

Wire format: magic 0xFC + format_id 0x30 wrapper around PR106 r2's `0.bin`,
where 0.bin itself carries the 0xFE/0x01 PR106 r2 inner sidecar.

Per CLAUDE.md HNeRV parity discipline lesson 4 (inflate ≤ 200 LOC waiver,
``lane_class=substrate_engineering``): inflate runtime LOC = file body only,
this script is reviewable in 30 seconds.

NO scorer load at inflate time (strict-scorer-rule). NO MPS authoritative
selection. NO ``/tmp`` paths. Pure tensor + numpy + brotli decode + bicubic
upsample + Gaussian-pull foveation warp via ``F.grid_sample``.

Invoked as: ``python inflate.py <data_dir>/<base>.bin <output_dir>/<base>.raw``
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

from codec import parse_packed_archive  # type: ignore[import-not-found]
from model import HNeRVDecoder  # type: ignore[import-not-found]
from pr106_inner_sidecar import (  # type: ignore[import-not-found]
    apply_pr106_r2_sidecar_corrections,
    unwrap_pr106_r2_sidecar,
)

FOVEATION_FIELD_MAGIC = 0xFC
FOVEATION_FIELD_FORMAT_ID = 0x30
CAMERA_H, CAMERA_W = 874, 1164


def parse_foveation_archive(blob: bytes) -> tuple[bytes, bytes]:
    """Split (pr106_r2_bytes, foveation_field_blob) from the 0xFC/0x30 wrapper."""
    if len(blob) < 6:
        raise ValueError("archive too short")
    if blob[0] != FOVEATION_FIELD_MAGIC or blob[1] != FOVEATION_FIELD_FORMAT_ID:
        raise ValueError(
            f"foveation wrapper magic/format_id mismatch: 0x{blob[0]:02X} 0x{blob[1]:02X}"
        )
    (pr106_len,) = struct.unpack_from("<I", blob, 2)
    pos = 6
    pr106_bytes = blob[pos : pos + pr106_len]
    pos += pr106_len
    (foveation_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    foveation_blob = blob[pos : pos + foveation_len]
    pos += foveation_len
    if pos != len(blob):
        raise ValueError(f"trailing bytes: pos={pos} total={len(blob)}")
    return bytes(pr106_bytes), bytes(foveation_blob)


def decode_foveation_field_inline(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimal decoder mirroring ``tac.foveation_field.decode_foveation_field``.

    Inflate-side keeps the decoder ≤ 50 LOC so the runtime is reviewable.
    """
    if not blob:
        return (np.zeros((1, 1, 2), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32))
    if blob[0] != 0xFC or blob[1] != 0x30:
        raise ValueError("inner foveation field magic mismatch")
    nf = struct.unpack_from("<H", blob, 2)[0]
    ng = blob[4]
    delta_scale = struct.unpack_from("<f", blob, 5)[0]
    pos = 9
    anchor_centers = np.frombuffer(blob, dtype=np.float16, count=ng * 2, offset=pos).astype(np.float32).reshape(ng, 2)
    pos += ng * 4
    anchor_log_sigma = np.frombuffer(blob, dtype=np.float16, count=ng, offset=pos).astype(np.float32)
    pos += ng * 2
    anchor_log_amp = np.frombuffer(blob, dtype=np.float16, count=ng, offset=pos).astype(np.float32)
    pos += ng * 2
    deltas_len = struct.unpack_from("<I", blob, pos)[0]
    pos += 4
    deltas_payload = blob[pos : pos + deltas_len]
    full = np.empty((nf, ng, 4), dtype=np.float32)
    full[0, :, 0:2] = anchor_centers
    full[0, :, 2] = anchor_log_sigma
    full[0, :, 3] = anchor_log_amp
    if nf > 1 and deltas_len > 0:
        deltas_raw = brotli.decompress(deltas_payload)
        deltas_q = np.frombuffer(deltas_raw, dtype=np.int8).reshape(nf - 1, ng, 4)
        deltas = deltas_q.astype(np.float32) * float(delta_scale)
        for t in range(1, nf):
            full[t] = full[t - 1] + deltas[t - 1]
    return full[:, :, 0:2], full[:, :, 2], full[:, :, 3]


def apply_foveation_warp(rgb: torch.Tensor, centers: torch.Tensor, sigma: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
    """Apply per-frame Gaussian-pull warp (mirrors tac.foveation_field.compute_foveation_warp)."""
    t_dim, _, h, w = rgb.shape
    ys = torch.linspace(0.0, 1.0, h, device=rgb.device)
    xs = torch.linspace(0.0, 1.0, w, device=rgb.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(t_dim, -1, -1, -1)
    delta = torch.zeros_like(base)
    ng = centers.shape[1]
    for g in range(ng):
        c = centers[:, g].unsqueeze(1).unsqueeze(1)
        s = sigma[:, g].unsqueeze(-1).unsqueeze(-1)
        a = amp[:, g].unsqueeze(-1).unsqueeze(-1)
        diff = c - base
        sq_dist = (diff ** 2).sum(dim=-1)
        env = torch.exp(-sq_dist / (2.0 * s ** 2 + 1e-12))
        delta = delta + (a * env).unsqueeze(-1) * diff
    warped_grid = 2.0 * (base + delta) - 1.0
    return F.grid_sample(rgb, warped_grid, mode="bilinear", padding_mode="border", align_corners=False)


def select_inflate_device() -> torch.device:
    policy = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if policy in {"mps", "metal"}:
        raise RuntimeError("PACT_INFLATE_DEVICE=mps is forbidden for auth-eval inflate")
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inflate(src_bin: str, dst_raw: str) -> int:
    blob = Path(src_bin).read_bytes()
    pr106_r2_bytes, foveation_blob = parse_foveation_archive(blob)
    raw_pr106, sidecar_blob = unwrap_pr106_r2_sidecar(pr106_r2_bytes)
    decoder_sd, latents, meta = parse_packed_archive(raw_pr106)
    apply_pr106_r2_sidecar_corrections(latents, sidecar_blob)
    centers_np, log_sigma_np, log_amp_np = decode_foveation_field_inline(foveation_blob)
    device = select_inflate_device()
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    eval_h, eval_w = meta["eval_size"]
    centers = torch.from_numpy(centers_np).to(device)
    sigma = torch.from_numpy(np.exp(log_sigma_np)).to(device)
    amp = torch.from_numpy(np.exp(log_amp_np)).to(device)
    n_frames_field = centers.shape[0]
    print(
        f"[inflate] PR106+foveation_field device={device.type} n_pairs={n_pairs} "
        f"field_frames={n_frames_field} field_n_gauss={centers.shape[1]} "
        f"foveation_bytes={len(foveation_blob)}",
        file=sys.stderr,
    )
    written = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(n_pairs):
            decoded = decoder(latents[i : i + 1]).reshape(2, 3, eval_h, eval_w)
            up = F.interpolate(decoded, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            # Apply foveation warp per-frame; map pair index to field frame.
            f_start = min(i * 2, n_frames_field - 1)
            f_end = min(i * 2 + 2, n_frames_field)
            n_pair_frames = f_end - f_start
            if n_pair_frames > 0:
                up = apply_foveation_warp(
                    up[:n_pair_frames],
                    centers[f_start:f_end],
                    sigma[f_start:f_end],
                    amp[f_start:f_end],
                )
            out = up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            fout.write(out.tobytes())
            written += 2
    print(f"saved {written} frames")
    return written


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
