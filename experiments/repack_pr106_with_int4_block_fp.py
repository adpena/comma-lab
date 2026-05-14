#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SKETCH: Repack PR106 HNeRV decoder via int4 signed block-FP (parallel to OWV2 repack).

Lane #04 partial-revival empirical comparison vs Lane Ω-W-V3.

Pipeline:
    1. Load PR106 state_dict (from extract_pr106_decoder.py output)
    2. For each Conv2d weight: int4 block-FP (block_size=128) + brotli
    3. For each Linear/bias/small tensor: PR106 brotli-int8 fallback
    4. Pack into apogee_int4 0.bin layout (magic 0xA4)
    5. Wrap in single-member archive.zip

Empirical comparison target:
    PR106 brotli (whole decoder): 186,239 bytes archive
    OWV2 stub (Lane Ω-W-V3): 164,087 bytes archive
    int4 block-FP this script: TARGET <120,000 bytes

Distortion impact UNKNOWN until contest-CUDA dispatch — int4 max_err=0.069
on individual weights may or may not survive PoseNet/SegNet eval.

NOT YET COUNCIL-APPROVED. Sketch lives in experiments/ until council reviews.

Usage:
    .venv/bin/python experiments/repack_pr106_with_int4_block_fp.py \\
        --state-dict experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt \\
        --pr106-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \\
        --out-dir experiments/results/apogee_int4_repack_20260504_claude/
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import torch

PR106_SRC_PATH = Path(__file__).parent / "results" / (
    "public_pr106_belt_and_suspenders_intake_20260504_codex/source/"
    "submissions/belt_and_suspenders/src"
)
sys.path.insert(0, str(PR106_SRC_PATH.resolve()))
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from codec import encode_decoder, quantize_state_dict  # type: ignore[import-not-found]  # noqa: E402
from block_fp_int4_codec_sketch import encode_int4_block_fp  # type: ignore[import-not-found]  # noqa: E402


CODEC_ID_BROTLI_INT8 = 0
CODEC_ID_INT4_BLOCKFP_BROTLI = 2  # 0x02 distinct from OWV2's 0x01


def _is_int4_eligible(name: str, t: torch.Tensor) -> bool:
    """4-D Conv2d weight only (matching OWV2's eligibility criterion)."""
    return t.dim() == 4 and ".weight" in name and t.shape[0] >= 1


def _encode_brotli_int8_single(name: str, t: torch.Tensor) -> bytes:
    sd_one = {name: t}
    q_sd = quantize_state_dict(sd_one)
    return encode_decoder(q_sd)


def _encode_int4_blockfp_with_brotli(t: torch.Tensor, block_size: int = 128) -> bytes:
    """int4 block-FP encode then brotli compress."""
    blob = encode_int4_block_fp(t.to(torch.float32), block_size=block_size)
    return brotli.compress(blob, quality=11)


def _build_apogee_int4_blob(
    encoded: dict[str, tuple[int, bytes]],
    latent_brotli: bytes,
) -> bytes:
    """apogee_int4 0.bin layout (extension of apogee_v2):

        magic(1B) = 0xA4  (apogee_int4 marker)
        n_codecs(1B)
        for each entry:
            codec_id(1B), name_len(1B), name_utf8 bytes,
            shape_ndim(1B), payload_len(4B uint32 LE), payload bytes
        latent_len(4B uint32 LE), latent_brotli bytes
    """
    buf = io.BytesIO()
    buf.write(bytes([0xA4]))
    buf.write(bytes([len(encoded) & 0xFF]))
    for name, (codec_id, payload) in encoded.items():
        buf.write(bytes([codec_id & 0xFF]))
        nb = name.encode("utf-8")
        buf.write(bytes([len(nb) & 0xFF]))
        buf.write(nb)
        buf.write(bytes([0]))  # shape_ndim placeholder
        buf.write(struct.pack("<I", len(payload)))
        buf.write(payload)
    buf.write(struct.pack("<I", len(latent_brotli)))
    buf.write(latent_brotli)
    return buf.getvalue()


def _parse_pr106_packed_for_latents(bin_bytes: bytes) -> tuple[bytes, dict]:
    dec_len = int.from_bytes(bin_bytes[1:4], "little")
    latent_bytes = bin_bytes[4 + dec_len:]
    return latent_bytes, {"dec_len": dec_len}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--pr106-archive", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_dict = torch.load(args.state_dict, map_location="cpu", weights_only=False)
    print(f"[repack-int4] state_dict: {len(state_dict)} tensors")

    encoded: dict[str, tuple[int, bytes]] = {}
    for name, t in state_dict.items():
        if _is_int4_eligible(name, t):
            payload = _encode_int4_blockfp_with_brotli(t, block_size=args.block_size)
            encoded[name] = (CODEC_ID_INT4_BLOCKFP_BROTLI, payload)
            print(f"  [int4] {name}: {len(payload)} bytes")
        else:
            payload = _encode_brotli_int8_single(name, t)
            encoded[name] = (CODEC_ID_BROTLI_INT8, payload)
            print(f"  [brotli-int8] {name}: {len(payload)} bytes")

    with zipfile.ZipFile(args.pr106_archive) as z:
        bin_bytes = z.read("0.bin")
    latent_brotli, _ = _parse_pr106_packed_for_latents(bin_bytes)
    print(f"[repack-int4] harvested latent_brotli: {len(latent_brotli)} bytes")

    new_bin = _build_apogee_int4_blob(encoded, latent_brotli)
    print(f"[repack-int4] apogee_int4 0.bin: {len(new_bin)} bytes")

    archive_path = args.out_dir / "apogee_int4_archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as z:  # DETERMINISTIC_ZIP_OK
        zi = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        z.writestr(zi, new_bin)
    archive_size = archive_path.stat().st_size
    pr106_size = args.pr106_archive.stat().st_size
    delta = archive_size - pr106_size
    score_delta = 25.0 * delta / 37545489.0
    print(f"[repack-int4] wrote {archive_path}: {archive_size} bytes")
    print(f"[repack-int4] PR106: {pr106_size} bytes; delta: {delta:+d} ({100*delta/pr106_size:+.2f}%)")
    print(f"[repack-int4] estimated rate-component score Δ vs PR106: {score_delta:+.6f}")
    print("[repack-int4] (distortion Δ unknown without CUDA dispatch; max_err per int4 weight = 0.069 — "
          "5-14% relative error per weight, may or may not survive eval)")

    metadata = {
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_size,
        "pr106_size_bytes": pr106_size,
        "delta_bytes": delta,
        "rate_component_score_delta_vs_pr106": score_delta,
        "block_size": args.block_size,
        "n_int4_layers": sum(1 for cid, _ in encoded.values() if cid == CODEC_ID_INT4_BLOCKFP_BROTLI),
        "n_brotli_int8_layers": sum(1 for cid, _ in encoded.values() if cid == CODEC_ID_BROTLI_INT8),
        "tag": "[design-validation]",
        "council_status": "NOT_REVIEWED",
        "next_step": "Compare empirically to Lane Ω-W-V3 stub-mode preview (-22,152 bytes / -0.01475 score Δ). If int4 archive is significantly smaller AND empirical contest-CUDA score < 0.20945, this is a Lane #04 standalone success. If distortion blows up, fold int4 into water_filling_codec_v2 with per-channel sensitivity gating (hybrid).",
    }
    metadata_path = args.out_dir / "repack_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[repack-int4] wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
