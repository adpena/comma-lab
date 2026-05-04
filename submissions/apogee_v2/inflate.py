#!/usr/bin/env python
"""Inflate apogee_v2 archive: PR106 HNeRV decoder repacked via water-fill v2.

Reads <src>.bin (apogee_v2 layout: magic 0xA2 + per-tensor codec dispatch +
PR106 latent-brotli unchanged), reconstructs the HNeRV state_dict, runs the
decoder forward at 384x512, bicubic-upsamples to camera resolution (874x1164),
rounds to uint8, writes contiguous (N, H, W, 3) bytes to <dst>.

Codec dispatch:
    codec_id = 0  → PR106 brotli-int8 single-tensor (Linear / bias / small Conv2d)
    codec_id = 1  → tac.water_filling_codec_v2 (OWV2, for block-FP-eligible Conv2d)

Anchor: revival_plan_01_water_filling_codec_v2_pr106_decoder + Lane Ω-W-V3 step 4/4.

Invoked by inflate.sh as:
    python -m submissions.apogee_v2.inflate <data_dir>/<base>.bin <output_dir>/<base>.raw
"""
from __future__ import annotations

import io
import struct
import sys
import zipfile
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

# We import the PR106 model.py + codec.py from src/ (vendored alongside this inflate).
from model import HNeRVDecoder  # type: ignore[import-not-found]
from codec import decode_packed_decoder, decode_fixed_latents  # type: ignore[import-not-found]

# tac is installed in the venv. The water-filling decoder is needed for codec_id=1.
from tac.water_filling_codec_v2 import decode_omega_w_v2  # noqa: E402


CAMERA_H, CAMERA_W = 874, 1164
APOGEE_V2_MAGIC = 0xA2
CODEC_ID_BROTLI_INT8 = 0
CODEC_ID_OWV2 = 1


def _decode_brotli_int8_single_tensor(payload: bytes) -> tuple[str, torch.Tensor]:
    """Inverse of repack_pr106_with_water_filling._encode_brotli_int8_single.
    PR106 quantize_state_dict + encode_decoder embedded for a single tensor.
    Format: brotli-decompress → 4B n_tensors=1 → for each tensor: name_len, name,
    shape_ndim, shape ints, scale float, count, zigzag bytes.
    """
    raw = brotli.decompress(payload)
    buf = io.BytesIO(raw)
    (n_tensors,) = struct.unpack("<I", buf.read(4))
    if n_tensors != 1:
        raise ValueError(f"single-tensor brotli-int8 payload had n_tensors={n_tensors}, expected 1")
    (name_len,) = struct.unpack("<I", buf.read(4))
    name = buf.read(name_len).decode("utf-8")
    (ndim,) = struct.unpack("<I", buf.read(4))
    shape = tuple(struct.unpack(f"<{ndim}I", buf.read(ndim * 4)))
    (scale,) = struct.unpack("<f", buf.read(4))
    (count,) = struct.unpack("<I", buf.read(4))
    zz_u8 = np.frombuffer(buf.read(count), dtype=np.uint8).copy()
    zz_i32 = zz_u8.astype(np.int32)
    i8 = np.where(zz_i32 % 2 == 0, zz_i32 // 2, -(zz_i32 // 2) - 1).astype(np.int8)
    dequant = torch.from_numpy(i8.astype(np.float32)) * float(scale)
    return name, dequant.reshape(shape)


def parse_apogee_v2_archive(archive_bytes: bytes) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    """Parse apogee_v2 0.bin layout.

    Returns (state_dict, latents, meta) — same triple shape as PR106's parse_packed_archive.
    """
    if not archive_bytes or archive_bytes[0] != APOGEE_V2_MAGIC:
        raise ValueError(
            f"apogee_v2 magic mismatch: got 0x{archive_bytes[0]:02X}, expected 0x{APOGEE_V2_MAGIC:02X}"
        )
    pos = 1
    n_codecs = archive_bytes[pos]
    pos += 1
    state_dict: dict[str, torch.Tensor] = {}
    for _ in range(n_codecs):
        codec_id = archive_bytes[pos]; pos += 1
        name_len = archive_bytes[pos]; pos += 1
        name = archive_bytes[pos : pos + name_len].decode("utf-8")
        pos += name_len
        shape_ndim = archive_bytes[pos]; pos += 1  # currently unused (shape is in payload)
        if shape_ndim:
            pos += shape_ndim * 4  # skip shape ints if any
        (payload_len,) = struct.unpack("<I", archive_bytes[pos : pos + 4])
        pos += 4
        payload = archive_bytes[pos : pos + payload_len]
        pos += payload_len
        if codec_id == CODEC_ID_BROTLI_INT8:
            decoded_name, t = _decode_brotli_int8_single_tensor(payload)
            if decoded_name != name:
                raise ValueError(f"name mismatch in brotli-int8 entry: outer={name!r}, inner={decoded_name!r}")
            state_dict[name] = t
        elif codec_id == CODEC_ID_OWV2:
            t = decode_omega_w_v2(payload)
            state_dict[name] = t
        else:
            raise ValueError(f"unknown codec_id={codec_id} for tensor {name!r}")
    # Latent: 4B latent_len + brotli payload
    (latent_len,) = struct.unpack("<I", archive_bytes[pos : pos + 4])
    pos += 4
    latents = decode_fixed_latents(archive_bytes[pos : pos + latent_len])
    if pos + latent_len != len(archive_bytes):
        raise ValueError(
            f"apogee_v2 trailing bytes: pos+latent_len={pos + latent_len}, total={len(archive_bytes)}"
        )
    meta = {"n_pairs": 600, "latent_dim": 28, "base_channels": 36, "eval_size": [384, 512]}
    return state_dict, latents, meta


def inflate(src_bin: str, dst_raw: str) -> int:
    archive_bytes = Path(src_bin).read_bytes()
    decoder_sd, latents, meta = parse_apogee_v2_archive(archive_bytes)

    if not torch.cuda.is_available():
        sys.exit("apogee_v2 inflate requires GPU (per CLAUDE.md MPS-auth-eval-is-NOISE)")
    device = torch.device("cuda")
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

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            B = j - i
            decoded = decoder(latents[i:j])  # (B, 2, 3, eval_h, eval_w)
            flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            frames = (
                up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            )
            fout.write(frames.tobytes())
            n += B * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m submissions.apogee_v2.inflate <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
