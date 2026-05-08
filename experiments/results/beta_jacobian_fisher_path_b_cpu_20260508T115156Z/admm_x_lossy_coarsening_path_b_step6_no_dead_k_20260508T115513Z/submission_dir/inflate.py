#!/usr/bin/env python
"""Forked PR101 inflate for lossy_coarsening_analytical archive — variant
without K side-info (REVIEW-ENG C2 fix; same reconstruction as the original
variant; the original's K bytes were dead audit metadata).

Wire format (inner blob, single ZIP member 'x'):
    +--------------------------------------------+
    | uint32 LE: decoder_section_total_bytes (D) |
    +--------------------------------------------+
    | byte * 56: per_tensor_fp16_scale (LE half) |
    +--------------------------------------------+
    | byte * (D - 4 - 56): brotli(int8s)         |
    +--------------------------------------------+
    | byte * 15387: latent_blob (PR101 ORIGINAL) |
    +--------------------------------------------+
    | byte * remaining: sidecar_blob (ORIGINAL)  |
    +--------------------------------------------+

Decoder:
1. Read uint32 prefix = D
2. Read 56 bytes scale_fp16[i]
3. brotli-decode the remaining (D - 60) bytes -> flat int8 array
4. Split flat into per-tensor int8 chunks per FIXED_STATE_SCHEMA shapes
5. Use each chunk directly as the recovered coarsened q_i8 (already rounded
   at encode time; no need for K to dequantize)
6. Apply per-tensor fp16 scale to recover float weights:
   recovered_fp32 = recovered_q_i8.astype(fp32) * scale_fp16

Latent + sidecar use the original PR101 codec functions.
"""
import struct
import sys
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from codec import (  # PR101 originals (vendored)
    LATENT_BLOB_LEN,
    N_PAIRS,
    LATENT_DIM,
    BASE_CHANNELS,
    EVAL_SIZE,
    decode_latents_compact,
    apply_latent_sidecar,
)
from model import HNeRVDecoder


def _fixed_state_schema():
    probe = HNeRVDecoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        eval_size=EVAL_SIZE,
    )
    return tuple((name, tuple(t.shape)) for name, t in probe.state_dict().items())


FIXED_STATE_SCHEMA = _fixed_state_schema()
N_TENSORS = len(FIXED_STATE_SCHEMA)
SCALE_SECTION_BYTES = N_TENSORS * 2  # fp16 = 2 bytes each = 56 bytes
PREFIX_BYTES = 4  # uint32 LE


CAMERA_H, CAMERA_W = 874, 1164


def parse_lossy_archive(archive_bytes):
    if len(archive_bytes) < PREFIX_BYTES + SCALE_SECTION_BYTES + LATENT_BLOB_LEN:
        raise ValueError(
            f"archive too short ({len(archive_bytes)} bytes) for lossy_coarsening "
            "no-dead-K format"
        )
    section_total = struct.unpack("<I", archive_bytes[:PREFIX_BYTES])[0]
    if section_total < PREFIX_BYTES + SCALE_SECTION_BYTES:
        raise ValueError(
            f"decoder_section_total ({section_total}) < minimum "
            f"{PREFIX_BYTES + SCALE_SECTION_BYTES}"
        )
    if section_total > len(archive_bytes) - LATENT_BLOB_LEN:
        raise ValueError(
            f"decoder_section_total ({section_total}) leaves no room for "
            f"latent_blob {LATENT_BLOB_LEN}"
        )

    scale_start = PREFIX_BYTES
    scale_end = scale_start + SCALE_SECTION_BYTES
    scales_fp16 = np.frombuffer(archive_bytes[scale_start:scale_end], dtype="<f2")
    if scales_fp16.size != N_TENSORS:
        raise ValueError(
            f"scale section size {scales_fp16.size} != N_TENSORS {N_TENSORS}"
        )

    brotli_start = scale_end
    brotli_end = section_total
    brotli_payload = archive_bytes[brotli_start:brotli_end]
    flat_int8 = np.frombuffer(brotli.decompress(brotli_payload), dtype=np.int8)

    decoder_sd = {}
    cursor = 0
    for idx, (name, shape) in enumerate(FIXED_STATE_SCHEMA):
        nelem = 1
        for d in shape:
            nelem *= d
        if cursor + nelem > flat_int8.size:
            raise ValueError(
                f"flat_int8 underflow at tensor {idx} ({name}): "
                f"need {nelem}, have {flat_int8.size - cursor}"
            )
        chunk = flat_int8[cursor:cursor + nelem].astype(np.int32)
        # Same as the original variant: the stream stores the K-coarsened
        # q_i8 directly. The original variant's comment noted K was kept
        # "for audit/reproducibility" but never used in decoding — the
        # no-dead-k variant simply omits those bytes from the archive.
        reconstructed_q = chunk
        # Dequantize: weight_fp32 = q_i8 * scale_fp16
        weight_fp32 = (reconstructed_q.astype(np.float32) * float(scales_fp16[idx]))
        decoder_sd[name] = torch.from_numpy(weight_fp32.reshape(shape).copy())
        cursor += nelem
    if cursor != flat_int8.size:
        raise ValueError(
            f"flat_int8 leftover {flat_int8.size - cursor} bytes after consuming all tensors"
        )

    latent_start = section_total
    latent_end = latent_start + LATENT_BLOB_LEN
    latent_blob = archive_bytes[latent_start:latent_end]
    sidecar_blob = archive_bytes[latent_end:]
    if not latent_blob:
        raise ValueError("missing latent_blob in lossy archive")

    meta = {
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    return decoder_sd, latents, meta


def inflate(src_bin: str, dst_raw: str):
    with open(src_bin, "rb") as f:
        archive_bytes = f.read()
    decoder_sd, latents, meta = parse_lossy_archive(archive_bytes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W),
                mode="bicubic", align_corners=False,
            )
            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            frames = (
                up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            fout.write(frames.tobytes())
            n += batch * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
