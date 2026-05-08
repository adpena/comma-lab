#!/usr/bin/env python
"""Forked inflate for cross-paradigm ADMM-x-Op1-finalizer archive.

Wire format (inner blob, single ZIP member 'x'):
    +-----------------------------------------------+
    | 4 bytes: magic = b"CPLX"                      |
    +-----------------------------------------------+
    | 4 bytes: uint32 LE = decoder_section_bytes D  |
    +-----------------------------------------------+
    | 2 bytes: uint16 LE = byte_maps_json_len J     |
    +-----------------------------------------------+
    | J bytes: utf-8 JSON {str(idx): byte_map_str}  |
    +-----------------------------------------------+
    | (D - 10 - J) bytes: Op1 inner blob            |
    |   = PR101 split-Brotli stream concatenation   |
    +-----------------------------------------------+
    | 15387 bytes: PR101 latent_blob (UNCHANGED)    |
    +-----------------------------------------------+
    | remaining: PR101 sidecar_blob (UNCHANGED)     |
    +-----------------------------------------------+

The Op1 inner blob is the raw PR101 split-Brotli stream concatenation produced
by ``encode_decoder_compact(state_dict, effective_byte_maps=byte_maps,
auto_select=False)`` on the dequantized post-ADMM substrate. The byte_maps
JSON is rehydrated to int keys here before passing to ``decode_decoder_compact``.

Latent + sidecar use PR101's authored decoder functions verbatim.

WIRE-DECODER NOTE 2026-05-08: this bypasses ``CodecPipeline`` because the CPL1
wrapper has a JSON-int-key bug (effective_byte_maps int keys → strings → miss
in ``decode_decoder_compact``'s ``idx in effective_byte_maps`` check).
"""
import json
import struct
import sys
from pathlib import Path

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
# Vendored tac module: only PR101 split-Brotli decoder needed (we bypass
# CodecPipeline entirely; see WIRE-DECODER NOTE above).
from tac.pr101_split_brotli_codec import decode_decoder_compact

CPLX_MAGIC = b"CPLX"
CPLX_HEADER_LEN = 8  # 4 magic + 4 length

CAMERA_H, CAMERA_W = 874, 1164


def parse_cross_paradigm_archive(archive_bytes):
    if len(archive_bytes) < CPLX_HEADER_LEN + 2 + LATENT_BLOB_LEN:
        raise ValueError(
            f"archive too short ({len(archive_bytes)} bytes) for cross-paradigm format"
        )
    magic = archive_bytes[:4]
    if magic != CPLX_MAGIC:
        raise ValueError(
            f"bad magic {magic!r}, expected {CPLX_MAGIC!r} (cross-paradigm CPLX wire format)"
        )
    section_total = struct.unpack("<I", archive_bytes[4:8])[0]
    if section_total < CPLX_HEADER_LEN + 2:
        raise ValueError(
            f"decoder_section_total ({section_total}) < minimum {CPLX_HEADER_LEN + 2}"
        )
    if section_total > len(archive_bytes) - LATENT_BLOB_LEN:
        raise ValueError(
            f"decoder_section_total ({section_total}) leaves no room for "
            f"latent_blob {LATENT_BLOB_LEN}"
        )

    bm_json_len = struct.unpack("<H", archive_bytes[8:10])[0]
    bm_json_start = 10
    bm_json_end = bm_json_start + bm_json_len
    if bm_json_end > section_total:
        raise ValueError(
            f"byte_maps json length {bm_json_len} overflows section_total {section_total}"
        )
    bm_str_keyed = json.loads(
        archive_bytes[bm_json_start:bm_json_end].decode("utf-8")
    )
    # Rehydrate int keys (JSON serialization coerces dict[int, str] -> str-keyed).
    effective_byte_maps = {int(k): str(v) for k, v in bm_str_keyed.items()}

    op1_inner_blob = archive_bytes[bm_json_end:section_total]
    decoder_sd = decode_decoder_compact(
        op1_inner_blob, effective_byte_maps=effective_byte_maps
    )

    latent_start = section_total
    latent_end = latent_start + LATENT_BLOB_LEN
    latent_blob = archive_bytes[latent_start:latent_end]
    sidecar_blob = archive_bytes[latent_end:]
    if not latent_blob:
        raise ValueError("missing latent_blob in cross-paradigm archive")

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
    decoder_sd, latents, meta = parse_cross_paradigm_archive(archive_bytes)

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
