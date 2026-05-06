#!/usr/bin/env python
"""Inflate apogee_intN archive: PR106 HNeRV decoder repacked via signed intN block-FP.

Reads <src>.bin (apogee_intN layout: magic 0xA0|bits + per-tensor codec dispatch +
PR106 latent-brotli unchanged), reconstructs the HNeRV state_dict, runs the
decoder forward at 384x512, bicubic-upsamples to camera resolution (874x1164),
rounds to uint8, writes contiguous (N, H, W, 3) bytes to <dst>.

Magic byte encoding: 0xA0 high-nibble + bits low-nibble. The old predicted
score bands for these variants were byte-only planning artifacts; exact int4
T4 eval later showed scorer-basin collapse. Treat this runtime as forensic
unless the exact candidate archive has a SHA-tied distortion model,
scorer-basin parity report, or exact positive CUDA evidence.

Codec dispatch:
    codec_id = 0  → PR106 brotli-int8 single-tensor (Linear / bias / small Conv2d)
    codec_id = 5  → vendored intn_codec.decode_intN_blockfp_from_brotli (block-FP-eligible Conv2d)

Sister of submissions/apogee_v2/inflate.py (which dispatches OWV2 codec_id=1).

Invoked by inflate.sh as:
    python -m submissions.apogee_intN.inflate <data_dir>/<base>.bin <output_dir>/<base>.raw
"""
from __future__ import annotations

import io
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

from codec import decode_fixed_latents, decode_packed_decoder  # type: ignore[import-not-found]  # noqa: E402,F401
from intn_codec import decode_intN_blockfp_from_brotli  # type: ignore[import-not-found]  # noqa: E402
from model import HNeRVDecoder  # type: ignore[import-not-found]  # noqa: E402

CAMERA_H, CAMERA_W = 874, 1164
APOGEE_INTN_MAGIC_HIGH = 0xA0
CODEC_ID_BROTLI_INT8 = 0
CODEC_ID_INTN_BLOCKFP_BROTLI = 5


def _decode_brotli_int8_single_tensor(payload: bytes) -> tuple[str, torch.Tensor]:
    """Inverse of repack_pr106_with_intN_block_fp._encode_brotli_int8_single."""
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


def parse_apogee_intn_archive(archive_bytes: bytes) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    """Parse apogee_intN 0.bin layout.

    Returns (state_dict, latents, meta) — same triple shape as PR106's parse_packed_archive.
    """
    if not archive_bytes:
        raise ValueError("empty archive")
    magic = archive_bytes[0]
    if (magic & 0xF0) != APOGEE_INTN_MAGIC_HIGH or not (4 <= (magic & 0xF) <= 8):
        raise ValueError(
            f"apogee_intN magic mismatch: got 0x{magic:02X}, expected 0xA[4-8]"
        )
    bits = magic & 0xF
    pos = 1
    n_codecs = archive_bytes[pos]
    pos += 1
    state_dict: dict[str, torch.Tensor] = {}
    for _ in range(n_codecs):
        codec_id = archive_bytes[pos]
        pos += 1
        name_len = archive_bytes[pos]
        pos += 1
        name = archive_bytes[pos : pos + name_len].decode("utf-8")
        pos += name_len
        shape_ndim = archive_bytes[pos]
        pos += 1
        if shape_ndim:
            pos += shape_ndim * 4  # skip shape ints if any (currently 0 placeholder)
        (payload_len,) = struct.unpack("<I", archive_bytes[pos : pos + 4])
        pos += 4
        payload = archive_bytes[pos : pos + payload_len]
        pos += payload_len
        if codec_id == CODEC_ID_BROTLI_INT8:
            decoded_name, t = _decode_brotli_int8_single_tensor(payload)
            if decoded_name != name:
                raise ValueError(f"name mismatch in brotli-int8 entry: outer={name!r}, inner={decoded_name!r}")
            state_dict[name] = t
        elif codec_id == CODEC_ID_INTN_BLOCKFP_BROTLI:
            t = decode_intN_blockfp_from_brotli(payload)
            state_dict[name] = t
        else:
            raise ValueError(f"unknown codec_id={codec_id} for tensor {name!r}")
    # Latent: 4B latent_len + brotli payload
    (latent_len,) = struct.unpack("<I", archive_bytes[pos : pos + 4])
    pos += 4
    latents = decode_fixed_latents(archive_bytes[pos : pos + latent_len])
    if pos + latent_len != len(archive_bytes):
        raise ValueError(
            f"apogee_intN trailing bytes: pos+latent_len={pos + latent_len}, total={len(archive_bytes)}"
        )
    meta = {
        "n_pairs": 600, "latent_dim": 28, "base_channels": 36,
        "eval_size": [384, 512], "bits": int(bits),
    }
    return state_dict, latents, meta


def inflate(src_bin: str, dst_raw: str) -> int:
    archive_bytes = Path(src_bin).read_bytes()
    decoder_sd, latents, meta = parse_apogee_intn_archive(archive_bytes)

    if not torch.cuda.is_available():
        sys.exit("apogee_intN inflate requires GPU (per CLAUDE.md MPS-auth-eval-is-NOISE)")
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
    print(f"apogee_intN: bits={meta['bits']}, decoder loaded, running forward...", file=sys.stderr)

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
        sys.exit("Usage: python -m submissions.apogee_intN.inflate <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
