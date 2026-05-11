#!/usr/bin/env python
"""Forked PR101 inflate for fine-tuned A1 archive (no-dead-K wire format).

Wire format (inner blob, single ZIP member 'x'):
    uint32 LE: decoder_section_total_bytes (D)
    byte * (D - 4): encoded decoder blob (PR101 split-Brotli, canonical)
    byte * 15387: latent_blob (PR101 ORIGINAL — preserved from source archive)
    byte * remaining: sidecar_blob (PR101 ORIGINAL)
"""
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from codec import (
    LATENT_BLOB_LEN,
    decode_decoder_compact,
    decode_latents_compact,
    apply_latent_sidecar,
)
from model import HNeRVDecoder

CAMERA_H, CAMERA_W = 874, 1164
EVAL_H, EVAL_W = 384, 512
LATENT_DIM = 28
BASE_CHANNELS = 36
N_PAIRS = 600


def parse_a1_finetuned_archive(archive_bytes: bytes):
    if len(archive_bytes) < 4:
        raise ValueError("archive too short to read decoder section header")
    section_total = struct.unpack_from("<I", archive_bytes, 0)[0]
    if section_total < 4 or section_total > len(archive_bytes):
        raise ValueError(f"bad decoder_section_total {section_total}")
    decoder_blob = archive_bytes[4:section_total]
    latent_blob = archive_bytes[section_total:section_total + LATENT_BLOB_LEN]
    sidecar_blob = archive_bytes[section_total + LATENT_BLOB_LEN:]
    if not decoder_blob or len(latent_blob) != LATENT_BLOB_LEN:
        raise ValueError("bad finetuned-A1 archive layout")
    decoder_sd = decode_decoder_compact(decoder_blob)
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    return decoder_sd, latents


def inflate(src_bin: str, dst_raw: str):
    with open(src_bin, "rb") as f:
        archive_bytes = f.read()
    decoder_sd, latents = parse_a1_finetuned_archive(archive_bytes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = HNeRVDecoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        eval_size=(EVAL_H, EVAL_W),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, N_PAIRS, 16):
            j = min(i + 16, N_PAIRS)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, EVAL_H, EVAL_W)
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
