# SPDX-License-Identifier: MIT
"""Contest inflate runtime for the capstone VQ-NeRV archive (pure numpy).

Reads the byte-closed capstone ``archive.zip`` member, decodes the render basis
(decoder + FiLM weights + codebook + per-pair VQ indices + stored pose + config)
with NO MLX and NO torch, renders every pair through the pure-numpy reference
:func:`tac.capstone_vq_nerv.numpy_reference.numpy_decode_pair`, upsamples each
384x512 frame to the camera 1164x874, and writes the flat raw-uint8 tensor file
``TensorVideoDataset`` reads (``upstream/frame_utils.py``: shape
``(N, 874, 1164, 3)`` uint8, C-contiguous; pair k -> frames 2k, 2k+1).

CUDA/CPU-agnostic (pure numpy), deterministic, NO scorer loaded (Strict scorer
rule). The render weights (decoder + per-frame FiLM) and the pose stats are read
from the archive's config sidecar — a decoder-only archive is insufficient
because the per-frame FiLM is part of the render.

CLI: ``inflate.py <archive_path> <dst_raw>`` (the anr/PR95 contest convention;
``inflate.sh DATA_DIR OUTPUT_DIR FILE_LIST`` wraps it).
"""

from __future__ import annotations

import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

from tac.capstone_vq_nerv.export import (
    _decode_int8_brotli,
    bit_unpack_vq_indices,
    parse_capstone_archive_bytes,
)
from tac.capstone_vq_nerv.numpy_reference import (
    CapstoneDecodeConfig,
    bilinear_resize_to_nhwc,
    numpy_decode_pair,
)

CAMERA_H, CAMERA_W = 874, 1164  # frame_utils.camera_size = (W=1164, H=874)
RENDER_H, RENDER_W = 384, 512
CAPSTONE_CONFIG_MEMBER = "capstone_config_v1"


def _decode_fp16_blob_dict(blob: bytes) -> dict[str, np.ndarray]:
    """Decode the fp16+brotli name->array section (export ``_fp16_brotli``)."""
    import brotli

    raw = brotli.decompress(blob)
    out: dict[str, np.ndarray] = {}
    off = 0
    while off < len(raw):
        (nlen,) = struct.unpack_from("<H", raw, off)
        off += 2
        name = raw[off : off + nlen].decode("utf-8")
        off += nlen
        (ndim,) = struct.unpack_from("<B", raw, off)
        off += 1
        shape = []
        for _ in range(ndim):
            (d,) = struct.unpack_from("<I", raw, off)
            off += 4
            shape.append(int(d))
        n = int(np.prod(shape)) if shape else 1
        vals = np.frombuffer(raw[off : off + n * 2], dtype=np.float16).astype(np.float32)
        off += n * 2
        out[name] = vals.reshape(shape)
    return out


def _decode_weight_section(blob: bytes, decoder_dtype: str) -> dict[str, np.ndarray]:
    return (
        _decode_int8_brotli(blob)
        if decoder_dtype == "int8"
        else _decode_fp16_blob_dict(blob)
    )


def decode_archive(archive_bytes: bytes, config: dict) -> dict:
    """Parse the capstone archive + config into the numpy render inputs."""
    import brotli

    decoder_dtype = str(config.get("decoder_dtype", "fp16"))
    num_pairs = int(config["num_pairs"])
    codebook_size = int(config["codebook_size"])
    blobs = parse_capstone_archive_bytes(archive_bytes)
    weights = _decode_weight_section(blobs["decoder"], decoder_dtype)
    codebook = _decode_weight_section(blobs["codebook"], decoder_dtype)["codebook"]
    vq_indices = bit_unpack_vq_indices(blobs["index"], num_pairs, codebook_size)
    pose_raw = brotli.decompress(blobs["pose"])
    pose = np.frombuffer(pose_raw, dtype=np.float16).astype(np.float32).reshape(num_pairs, 6)
    cfg = CapstoneDecodeConfig(
        base_channels=int(config["base_channels"]),
        latent_dim=int(config.get("latent_dim", codebook.shape[1])),
        codebook_size=codebook_size,
        base_h=int(config.get("base_h", 6)),
        base_w=int(config.get("base_w", 8)),
        film_enabled=bool(config.get("film_enabled", True)),
        pose_normalize=bool(config.get("pose_normalize", True)),
        pose_mean=tuple(float(v) for v in config.get("pose_mean", (0.0,) * 6)),
        pose_std=tuple(float(v) for v in config.get("pose_std", (1.0,) * 6)),
    )
    return {
        "weights": weights,
        "codebook": codebook,
        "vq_indices": vq_indices,
        "pose": pose,
        "cfg": cfg,
        "num_pairs": num_pairs,
    }


def render_all_camera_frames(decoded: dict, *, batch: int = 16) -> np.ndarray:
    """Render every pair -> camera-resolution uint8 frames ``(N*2, 874, 1164, 3)``."""
    weights, codebook = decoded["weights"], decoded["codebook"]
    vq_indices, pose, cfg = decoded["vq_indices"], decoded["pose"], decoded["cfg"]
    num_pairs = int(decoded["num_pairs"])
    out = np.empty((num_pairs * 2, CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    for s in range(0, num_pairs, batch):
        e = min(s + batch, num_pairs)
        z_q = codebook[vq_indices[s:e]]
        render = numpy_decode_pair(z_q, pose[s:e], weights, cfg)  # (b,2,3,H,W)
        for k in range(2):  # frame0, frame1
            frame_nchw = render[:, k]  # (b,3,H,W)
            frame_nhwc = np.transpose(frame_nchw, (0, 2, 3, 1))  # (b,H,W,3)
            cam = bilinear_resize_to_nhwc(frame_nhwc, CAMERA_H, CAMERA_W)
            cam = np.clip(np.round(cam), 0, 255).astype(np.uint8)
            out[(s + np.arange(e - s)) * 2 + k] = cam
    return out


def _read_archive_and_config(archive_path: Path) -> tuple[bytes, dict]:
    """Read the archive member bytes + config sidecar from the zip (or raw .bin)."""
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            names = set(zf.namelist())
            member = "x" if "x" in names else next(iter(n for n in names if n != CAPSTONE_CONFIG_MEMBER))
            archive_bytes = zf.read(member)
            config = json.loads(zf.read(CAPSTONE_CONFIG_MEMBER).decode("utf-8"))
        return archive_bytes, config
    # Bare ``.bin`` fallback: a sibling ``<stem>.config.json`` carries the config.
    archive_bytes = archive_path.read_bytes()
    config = json.loads(archive_path.with_suffix(".config.json").read_text())
    return archive_bytes, config


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: inflate.py <archive_path> <dst_raw>", file=sys.stderr)
        sys.exit(2)
    archive_path, dst_raw = Path(sys.argv[1]), Path(sys.argv[2])
    archive_bytes, config = _read_archive_and_config(archive_path)
    decoded = decode_archive(archive_bytes, config)
    frames = render_all_camera_frames(decoded)
    dst_raw.parent.mkdir(parents=True, exist_ok=True)
    dst_raw.write_bytes(frames.tobytes(order="C"))
    print(f"Wrote {dst_raw} shape={frames.shape} bytes={dst_raw.stat().st_size}")


if __name__ == "__main__":
    main()
