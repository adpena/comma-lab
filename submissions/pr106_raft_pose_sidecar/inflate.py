#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate PR106 r2 + RAFT optical-flow pose stream sidecar (research scaffold).

Wire format: magic 0xFD + format_id 0x31 (PR106 residual family) around
PR106 r2's `0.bin` + length-prefixed RAFT pose stream blob.

At inflate time, the RAFT model is NOT loaded — the pose stream is read from
the archive as precomputed per-frame-pair 6-DoF poses and written to a
``raft_poses.pt`` sidecar that downstream contest auth-eval picks up via the
PR106 poses-attached pathway. Per CLAUDE.md HNeRV parity discipline lesson 4
(inflate ≤ 200 LOC waiver, ``lane_class=substrate_engineering``).

NO scorer load at inflate time. NO MPS authoritative. NO ``/tmp`` paths.

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

PR106_RESIDUAL_MAGIC = 0xFD
RAFT_POSE_STREAM_FORMAT_ID = 0x31
CAMERA_H, CAMERA_W = 874, 1164
POSE_DIM = 6


def parse_raft_pose_archive(blob: bytes) -> tuple[bytes, bytes]:
    """Split (pr106_r2_bytes, raft_pose_blob) from the 0xFD/0x31 wrapper."""
    if len(blob) < 6:
        raise ValueError("archive too short")
    magic, format_id, pr106_len = struct.unpack_from("<BBI", blob, 0)
    if magic != PR106_RESIDUAL_MAGIC:
        raise ValueError(f"magic mismatch: 0x{magic:02X} != 0x{PR106_RESIDUAL_MAGIC:02X}")
    if format_id != RAFT_POSE_STREAM_FORMAT_ID:
        raise ValueError(f"format_id mismatch: 0x{format_id:02X} != 0x{RAFT_POSE_STREAM_FORMAT_ID:02X}")
    pos = 6
    pr106_bytes = blob[pos : pos + pr106_len]
    pos += pr106_len
    (raft_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    raft_blob = blob[pos : pos + raft_len]
    pos += raft_len
    if pos != len(blob):
        raise ValueError(f"trailing bytes: pos={pos} total={len(blob)}")
    return bytes(pr106_bytes), bytes(raft_blob)


def decode_raft_pose_stream_inline(blob: bytes) -> np.ndarray:
    """Mirror of ``tac.raft_pose_stream.decode_raft_pose_stream``.

    Returns ``(n_pairs, 6)`` float32 pose tensor. Empty input returns shape ``(0, 6)``.
    """
    if not blob:
        return np.zeros((0, POSE_DIM), dtype=np.float32)
    if blob[0] != PR106_RESIDUAL_MAGIC or blob[1] != RAFT_POSE_STREAM_FORMAT_ID:
        raise ValueError("inner raft pose stream magic mismatch")
    pos = 2
    (n_pairs,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    pose_dim = blob[pos]
    pos += 1
    if pose_dim != POSE_DIM:
        raise ValueError(f"pose_dim mismatch: {pose_dim} != {POSE_DIM}")
    anchor_pose = np.frombuffer(blob, dtype=np.float16, count=POSE_DIM, offset=pos).astype(np.float32)
    pos += POSE_DIM * 2
    scales = np.frombuffer(blob, dtype=np.float16, count=POSE_DIM, offset=pos).astype(np.float32)
    pos += POSE_DIM * 2
    (deltas_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    deltas_payload = blob[pos : pos + deltas_len]
    if n_pairs > 1 and deltas_len > 0:
        deltas_raw = brotli.decompress(deltas_payload)
        deltas_q = np.frombuffer(deltas_raw, dtype=np.int8).reshape(n_pairs - 1, POSE_DIM)
        deltas = deltas_q.astype(np.float32) * scales
        poses = np.empty((n_pairs, POSE_DIM), dtype=np.float32)
        poses[0] = anchor_pose
        poses[1:] = anchor_pose + np.cumsum(deltas, axis=0)
    else:
        poses = anchor_pose.reshape(1, POSE_DIM)
    return poses


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
    """Inflate the PR106+RAFT-pose archive to <dst_raw>.

    The RGB frames are written to ``dst_raw`` (binary uint8 RGB stream as in
    the canonical PR106 inflate). The RAFT-derived poses are written to a
    sidecar file ``<dst_raw>.poses.pt`` consumed by the contest auth-eval
    pipeline as the ``poses`` argument (drop-in replacement for PR106's
    ``poses.pt`` member).

    NOTE: the sidecar poses file is a research-only artifact at this scaffold
    level; the score-aware archive grammar that PR106 r2's auth-eval consumes
    is a future deliverable. ``score_claim=False`` permanently per CLAUDE.md.
    """
    blob = Path(src_bin).read_bytes()
    pr106_r2_bytes, raft_blob = parse_raft_pose_archive(blob)
    raw_pr106, sidecar_blob = unwrap_pr106_r2_sidecar(pr106_r2_bytes)
    decoder_sd, latents, meta = parse_packed_archive(raw_pr106)
    apply_pr106_r2_sidecar_corrections(latents, sidecar_blob)
    raft_poses = decode_raft_pose_stream_inline(raft_blob)
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
    print(
        f"[inflate] PR106+raft_pose device={device.type} n_pairs={n_pairs} "
        f"raft_pose_pairs={raft_poses.shape[0]} raft_bytes={len(raft_blob)}",
        file=sys.stderr,
    )
    written = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(n_pairs):
            decoded = decoder(latents[i : i + 1]).reshape(2, 3, eval_h, eval_w)
            up = F.interpolate(decoded, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            out = up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            fout.write(out.tobytes())
            written += 2
    # Side-write the RAFT poses tensor for downstream consumers.
    if raft_poses.size > 0:
        poses_path = Path(dst_raw).with_suffix(".poses.pt")
        torch.save(torch.from_numpy(raft_poses).float(), str(poses_path))
        print(f"[inflate] saved RAFT pose stream → {poses_path}", file=sys.stderr)
    print(f"saved {written} frames")
    return written


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
