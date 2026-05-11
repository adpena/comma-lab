#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate PR106 r2 + LAPose inverse-dynamics motion-atom sidecar (research).

Wire format: magic 0xFD + format_id 0x32 (PR106 residual family) around
PR106 r2's `0.bin` + length-prefixed LAPose atom stream blob.

LAPose-inspired motion atoms are typed parameter sets that adjust per-frame
pose corrections OR foveation pulls OR class-token routing of the renderer.
Per the handoff "LAPose-inspired until a paper-faithful inverse-dynamics
encoder and pose head exist; add class/openpilot manifests, calibrate
confidence, and require a charged archive consumer before dispatch."

At inflate time the atoms are decoded and applied to the PR106 r2 base
output. ``score_claim=False`` permanently; ``research_only=True`` at this
scaffold level.

NO scorer load at inflate time (strict-scorer-rule). NO MPS authoritative.
NO ``/tmp`` paths. Per CLAUDE.md HNeRV parity discipline lesson 4 (inflate
≤ 200 LOC waiver, ``lane_class=substrate_engineering``).

Invoked as: ``python inflate.py <data_dir>/<base>.bin <output_dir>/<base>.raw``
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

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

PR106_RESIDUAL_MAGIC = 0xFD
LAPOSE_MOTION_ATOM_FORMAT_ID = 0x32
CAMERA_H, CAMERA_W = 874, 1164

# Atom class codes (mirror tac.lapose_motion_atom_allocator.AtomClass).
ATOM_YAW_RATE = 0
ATOM_PITCH_RATE = 1
ATOM_FOVEATION_PULL = 2
ATOM_CLASS_TOKEN = 3
ATOM_CLASS_PARAM_BYTES = {0: 2, 1: 2, 2: 6, 3: 1}


def parse_lapose_archive(blob: bytes) -> tuple[bytes, bytes]:
    """Split (pr106_r2_bytes, lapose_blob) from the 0xFD/0x32 wrapper."""
    if len(blob) < 6:
        raise ValueError("archive too short")
    magic, format_id, pr106_len = struct.unpack_from("<BBI", blob, 0)
    if magic != PR106_RESIDUAL_MAGIC:
        raise ValueError(f"magic mismatch: 0x{magic:02X}")
    if format_id != LAPOSE_MOTION_ATOM_FORMAT_ID:
        raise ValueError(f"format_id mismatch: 0x{format_id:02X}")
    pos = 6
    pr106_bytes = blob[pos : pos + pr106_len]
    pos += pr106_len
    (lapose_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    lapose_blob = blob[pos : pos + lapose_len]
    pos += lapose_len
    if pos != len(blob):
        raise ValueError(f"trailing bytes: pos={pos} total={len(blob)}")
    return bytes(pr106_bytes), bytes(lapose_blob)


def decode_atom_stream(blob: bytes) -> tuple[int, list[tuple[int, int, bytes]]]:
    """Decode LAPose atom stream. Returns (n_frames, [(class, frame_idx, params), ...])."""
    if not blob:
        return 0, []
    if blob[0] != PR106_RESIDUAL_MAGIC or blob[1] != LAPOSE_MOTION_ATOM_FORMAT_ID:
        raise ValueError("inner lapose magic mismatch")
    pos = 2
    (n_frames,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    (n_atoms,) = struct.unpack_from("<H", blob, pos)
    pos += 2
    atoms = []
    for _ in range(n_atoms):
        atom_class = blob[pos]
        pos += 1
        (frame_idx,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        param_len = blob[pos]
        pos += 1
        if atom_class in ATOM_CLASS_PARAM_BYTES and param_len != ATOM_CLASS_PARAM_BYTES[atom_class]:
            raise ValueError(f"atom_class {atom_class} param_len mismatch: {param_len}")
        params = bytes(blob[pos : pos + param_len])
        pos += param_len
        atoms.append((atom_class, frame_idx, params))
    return n_frames, atoms


def apply_atom_to_frame_pair(frame_pair: torch.Tensor, atom_class: int, params: bytes) -> torch.Tensor:
    """Apply a single atom to a (T, 3, H, W) frame tensor.

    YAW_RATE/PITCH_RATE/CLASS_TOKEN are research_only routes to downstream
    consumers (no RGB effect yet). FOVEATION_PULL produces a Gaussian-pull
    warp; sigma is fixed at 0.1 (full field is the sister lane).
    """
    if atom_class != ATOM_FOVEATION_PULL:
        return frame_pair
    cx, cy, amp = (v * 1e-4 for v in struct.unpack("<hhh", params))
    if abs(amp) < 1e-8:
        return frame_pair
    t_dim, _, h, w = frame_pair.shape
    sigma = 0.1
    ys = torch.linspace(0.0, 1.0, h, device=frame_pair.device)
    xs = torch.linspace(0.0, 1.0, w, device=frame_pair.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(t_dim, -1, -1, -1)
    diff_x = cx - base[..., 0]
    diff_y = cy - base[..., 1]
    env = torch.exp(-(diff_x ** 2 + diff_y ** 2) / (2.0 * sigma ** 2 + 1e-12))
    grid = torch.stack([base[..., 0] + amp * env * diff_x, base[..., 1] + amp * env * diff_y], dim=-1)
    return F.grid_sample(frame_pair, 2.0 * grid - 1.0, mode="bilinear", padding_mode="border", align_corners=False)


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
    pr106_r2_bytes, lapose_blob = parse_lapose_archive(blob)
    raw_pr106, sidecar_blob = unwrap_pr106_r2_sidecar(pr106_r2_bytes)
    decoder_sd, latents, meta = parse_packed_archive(raw_pr106)
    apply_pr106_r2_sidecar_corrections(latents, sidecar_blob)
    n_frames_atom, atoms = decode_atom_stream(lapose_blob)
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
    # Bucket atoms by frame_idx for O(1) lookup.
    atoms_by_frame: dict[int, list[tuple[int, bytes]]] = {}
    for ac, fi, p in atoms:
        atoms_by_frame.setdefault(fi, []).append((ac, p))
    print(
        f"[inflate] PR106+lapose device={device.type} n_pairs={n_pairs} "
        f"atom_n_frames={n_frames_atom} n_atoms={len(atoms)} "
        f"lapose_bytes={len(lapose_blob)}",
        file=sys.stderr,
    )
    written = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(n_pairs):
            decoded = decoder(latents[i : i + 1]).reshape(2, 3, eval_h, eval_w)
            up = F.interpolate(decoded, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            # Apply any atoms targeting either frame of this pair.
            for frame_offset in range(2):
                global_idx = i * 2 + frame_offset
                if global_idx in atoms_by_frame:
                    for atom_class, params in atoms_by_frame[global_idx]:
                        frame_view = up[frame_offset : frame_offset + 1]
                        up = up.clone()
                        up[frame_offset : frame_offset + 1] = apply_atom_to_frame_pair(
                            frame_view, atom_class, params
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
