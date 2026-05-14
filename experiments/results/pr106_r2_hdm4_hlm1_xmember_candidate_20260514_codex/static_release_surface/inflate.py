#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate lane_pr106_latent_sidecar_r2_pr101_grammar archive.

PR106 HNeRV decoder + per-pair latent-correction sidecar with format_id dispatch:

  format_id=0x01 — legacy brotli-compressed (dim u8, delta_q i8) sidecar
  format_id=0x02 — PR101 ranked-Huffman/no-op grammar sidecar (this variant's
                    primary encoding; saves 42 bytes net vs format_id=0x01)

Both format_ids reconstruct the (dims, delta_q) arrays bit-identical, which is
parser/decoder-consumption evidence only. Score components require exact
auth-eval evidence under the scored runtime.

Reads <src>.bin (PR106 wrapper: magic 0xFE + format_id + PR106 bytes verbatim
+ appended sidecar), reconstructs PR106 state_dict + latents, applies the
per-pair (dim, delta) corrections, runs the HNeRV decoder forward at 384x512,
bicubic-upsamples to camera resolution (874x1164), rounds to uint8, and
writes contiguous (N, H, W, 3) bytes to <dst>.

Wire format details: see optimized_variant_manifest.json + the canonical
encoder in ``tac.packet_compiler.pr101_sidecar_grammar``.

Invoked by inflate.sh as:
    python -m submissions.pr106_latent_sidecar_r2_pr101_grammar.inflate <src.bin> <dst.raw>
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
from pr101_grammar import RankedSidecarSchema, decode_ranked_no_op_sidecar  # type: ignore[import-not-found]


CAMERA_H, CAMERA_W = 874, 1164
SIDECAR_MAGIC = 0xFE
SIDECAR_FORMAT_BROTLI = 0x01
SIDECAR_FORMAT_PR101_GRAMMAR = 0x02
SUPPORTED_FORMATS = (SIDECAR_FORMAT_BROTLI, SIDECAR_FORMAT_PR101_GRAMMAR)
DELTA_SCALE = 0.01
NO_OP_DIM = 255
DEFAULT_BATCH_PAIRS = 16

# PR101-grammar schema for this variant: n_pairs=600, n_dims=28,
# deltas=(-2,-1,1,2), huff_min/max=(2,8), no-op sentinel=255.
PR101_SCHEMA = RankedSidecarSchema(
    n_pairs=600,
    n_dims=28,
    deltas=(-2, -1, 1, 2),
    huff_min_len=2,
    huff_max_len=8,
    no_op_sentinel=255,
)


def parse_sidecar_archive(bin_bytes: bytes) -> tuple[int, bytes, bytes, bytes | None]:
    """Slice apart the wrapper and dispatch on format_id.

    Returns ``(format_id, pr106_bytes, sidecar_blob, framing_meta)``.

    For format_id=0x01 (brotli): ``framing_meta`` is ``None``; ``sidecar_blob``
    is the brotli-compressed payload.

    For format_id=0x02 (PR101 grammar): ``framing_meta`` is the bytes
    immediately after the PR101 payload, namely
    ``noop_count(2) | dim_bytes(2) | rank_bytes(1) | noop_rank_bytes(1)``;
    ``sidecar_blob`` is the PR101 grammar payload of length pr101_payload_len.
    """
    if not bin_bytes:
        raise ValueError("empty archive")
    if bin_bytes[0] != SIDECAR_MAGIC:
        raise ValueError(
            f"sidecar magic mismatch: got 0x{bin_bytes[0]:02X}, expected 0x{SIDECAR_MAGIC:02X}"
        )
    format_id = bin_bytes[1]
    if format_id not in SUPPORTED_FORMATS:
        raise ValueError(
            f"sidecar format_id 0x{format_id:02X} not supported; expected one of "
            f"{', '.join(f'0x{f:02X}' for f in SUPPORTED_FORMATS)}"
        )
    pos = 2
    (pr106_len,) = struct.unpack_from("<I", bin_bytes, pos)
    pos += 4
    pr106_bytes = bin_bytes[pos : pos + pr106_len]
    pos += pr106_len

    if format_id == SIDECAR_FORMAT_BROTLI:
        if pos + 2 > len(bin_bytes):
            raise ValueError("sidecar archive truncated before sidecar_len")
        (sidecar_len,) = struct.unpack_from("<H", bin_bytes, pos)
        pos += 2
        sidecar_blob = bin_bytes[pos : pos + sidecar_len]
        pos += sidecar_len
        if pos != len(bin_bytes):
            raise ValueError(
                f"sidecar archive trailing bytes: pos={pos} vs total={len(bin_bytes)}"
            )
        return format_id, pr106_bytes, sidecar_blob, None

    # format_id == SIDECAR_FORMAT_PR101_GRAMMAR
    if pos + 2 > len(bin_bytes):
        raise ValueError("pr101_grammar archive truncated before pr101_payload_len")
    (pr101_payload_len,) = struct.unpack_from("<H", bin_bytes, pos)
    pos += 2
    pr101_payload = bin_bytes[pos : pos + pr101_payload_len]
    pos += pr101_payload_len
    if pos + 6 > len(bin_bytes):
        raise ValueError("pr101_grammar archive truncated before framing meta")
    framing_meta = bin_bytes[pos : pos + 6]
    pos += 6
    if pos != len(bin_bytes):
        raise ValueError(
            f"pr101_grammar archive trailing bytes: pos={pos} vs total={len(bin_bytes)}"
        )
    return format_id, pr106_bytes, pr101_payload, framing_meta


def decode_brotli_sidecar(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode the legacy format_id=0x01 brotli-compressed sidecar payload."""
    raw = brotli.decompress(blob)
    n = struct.unpack_from("<H", raw, 0)[0]
    arr = np.frombuffer(raw[2 : 2 + 2 * n], dtype=np.uint8).reshape(n, 2)
    dim = arr[:, 0]
    delta_q = arr[:, 1].view(np.int8)
    return dim, delta_q


def decode_pr101_grammar_sidecar(
    payload: bytes, framing_meta: bytes
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the format_id=0x02 PR101 ranked-Huffman/no-op grammar sidecar."""
    noop_count, dim_bytes, rank_bytes, noop_rank_bytes = struct.unpack("<HHBB", framing_meta)
    dims, delta_indices = decode_ranked_no_op_sidecar(
        payload,
        schema=PR101_SCHEMA,
        dim_bytes=int(dim_bytes),
        rank_bytes=int(rank_bytes),
        noop_rank_bytes=int(noop_rank_bytes),
        noop_count=int(noop_count),
    )
    # Reconstruct (dim, delta_q) arrays matching format_id=0x01 byte semantics.
    dim_arr = dims.astype(np.int64)
    # Convert no-op sentinel from schema (255) to NO_OP_DIM (also 255).
    dim_arr_u8 = np.where(dim_arr == PR101_SCHEMA.no_op_sentinel, NO_OP_DIM, dim_arr).astype(np.uint8)
    delta_lookup = np.array(PR101_SCHEMA.deltas, dtype=np.int8)
    delta_q_arr = np.zeros(PR101_SCHEMA.n_pairs, dtype=np.int8)
    valid_mask = dim_arr != PR101_SCHEMA.no_op_sentinel
    delta_q_arr[valid_mask] = delta_lookup[delta_indices[valid_mask]]
    return dim_arr_u8, delta_q_arr


def apply_sidecar_corrections(
    latents: torch.Tensor,
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    scale: float = DELTA_SCALE,
) -> torch.Tensor:
    """In-place add per-pair correction to (n, latent_dim) latents tensor."""
    n = latents.shape[0]
    for p in range(n):
        d = int(dim_arr[p])
        if d == NO_OP_DIM:
            continue
        latents[p, d] = latents[p, d] + float(delta_q_arr[p]) * scale
    return latents


def select_inflate_device() -> torch.device:
    """Select an auth-eval-safe inflate device (cuda or cpu; MPS forbidden)."""
    policy = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if policy in {"mps", "metal"}:
        raise RuntimeError(
            "PACT_INFLATE_DEVICE=mps is forbidden for auth-eval inflate; use cpu or cuda"
        )
    if policy not in {"auto", "cpu", "cuda"}:
        raise RuntimeError(
            "PACT_INFLATE_DEVICE must be one of auto, cpu, cuda "
            f"(got {policy!r})"
        )
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def select_batch_pairs() -> int:
    """Return the deterministic decoder batch size for pair forwards."""
    raw = os.environ.get("PACT_INFLATE_BATCH_PAIRS")
    if raw is None:
        return DEFAULT_BATCH_PAIRS
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"PACT_INFLATE_BATCH_PAIRS must be a positive integer (got {raw!r})"
        ) from exc
    if value <= 0:
        raise RuntimeError(
            f"PACT_INFLATE_BATCH_PAIRS must be a positive integer (got {value})"
        )
    return value


def inflate(src_bin: str, dst_raw: str) -> int:
    archive_bytes = Path(src_bin).read_bytes()

    format_id, pr106_bytes, sidecar_blob, framing_meta = parse_sidecar_archive(archive_bytes)
    decoder_sd, latents, meta = parse_packed_archive(pr106_bytes)

    if format_id == SIDECAR_FORMAT_BROTLI:
        if sidecar_blob:
            dim_arr, delta_q_arr = decode_brotli_sidecar(sidecar_blob)
        else:
            dim_arr = np.full(PR101_SCHEMA.n_pairs, NO_OP_DIM, dtype=np.uint8)
            delta_q_arr = np.zeros(PR101_SCHEMA.n_pairs, dtype=np.int8)
    else:  # SIDECAR_FORMAT_PR101_GRAMMAR
        if framing_meta is None:
            raise ValueError("framing_meta missing for format_id=0x02 payload")
        dim_arr, delta_q_arr = decode_pr101_grammar_sidecar(sidecar_blob, framing_meta)

    n_corrections = int((dim_arr != NO_OP_DIM).sum())
    print(
        f"[inflate] format_id=0x{format_id:02X} sidecar applied: "
        f"{n_corrections}/{len(dim_arr)} pairs corrected",
        file=sys.stderr,
    )
    apply_sidecar_corrections(latents, dim_arr, delta_q_arr)

    try:
        device = select_inflate_device()
        batch_pairs = select_batch_pairs()
    except RuntimeError as exc:
        sys.exit(str(exc))
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
        f"[inflate] PR106+sidecar: decoder loaded, device={device.type}, "
        f"batch_pairs={batch_pairs}, running {n_pairs} pair forwards...",
        file=sys.stderr,
    )

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, batch_pairs):
            j = min(i + batch_pairs, n_pairs)
            B = j - i
            decoded = decoder(latents[i:j])  # (B, 2, 3, eval_h, eval_w)
            flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
            )
            frames = (
                up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            )
            fout.write(frames.tobytes())
            n += B * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "Usage: python -m submissions.pr106_latent_sidecar_r2_pr101_grammar.inflate "
            "<src.bin> <dst.raw>"
        )
    inflate(sys.argv[1], sys.argv[2])
