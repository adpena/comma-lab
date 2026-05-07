#!/usr/bin/env python3
"""Markov-1 adaptive arithmetic coding encoder/decoder for PR101 weights.

This implements the SUB-BROTLI deliverable promised by the provable
optimal floor analysis (memo
``feedback_pr101_provable_markov1_optimal_20260507.md``):

  Per-tensor Markov-1 AAC achieves ≤ 152,106 payload bytes
  (Shannon-source-coding-theorem-provable floor for 1-symbol context),
  vs brotli's 162,050 → ~10 KB total archive savings.

Algorithm:
  For each tensor, encode symbols sequentially:
    - First symbol: marginal PMF (Laplace-smoothed)
    - Subsequent: conditional PMF given previous symbol (Laplace-smoothed)
  All counts built adaptively from the prefix; ZERO PMF or transition
  table transmitted. Decoder mirrors exactly.

The encoder ships a payload only — no overhead — beyond per-tensor
length prefixes (4 bytes × 28 = 112 bytes) and the hyperparam alpha
(1 byte). Plus per-tensor scale (fp16 × 28 = 56 bytes). Plus tensor
shape header (8 bytes × 28 = 224 bytes). Total ancillary: ~395 bytes.

CLAUDE.md compliance: pure CPU + numpy + constriction + tac.pr101
quantizer. No scorer load, no GPU. Output bytes are deterministic
(same input → same output bytes).

Usage::

    .venv/bin/python tools/pr101_markov1_aac_codec.py encode \\
        --state-dict-path experiments/.../pr101_decoder_state_dict.pt \\
        --output-archive reports/pr101_markov1_aac.bin

    .venv/bin/python tools/pr101_markov1_aac_codec.py round-trip \\
        --state-dict-path experiments/.../pr101_decoder_state_dict.pt \\
        --output-summary reports/pr101_markov1_aac_round_trip.json
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_markov1_aac_codec.py"
SCHEMA_VERSION = "pr101_markov1_aac_codec.v1"
N_CATEGORIES = 255
DEFAULT_ALPHA = 1.0  # Laplace smoothing


# ---------------------------------------------------------------------------
# Encoder / Decoder primitive (per-tensor)
# ---------------------------------------------------------------------------

def encode_tensor_markov1(
    symbols: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[bytes, dict[str, Any]]:
    """Encode a single tensor's symbols with Markov-1 AAC.

    Returns (encoded_bytes, debug_info).
    """
    import constriction
    if symbols.size == 0:
        return b"", {"n_symbols": 0, "encoded_bytes": 0, "first_symbol": -1}

    # Conditional count tables: P(s_n | s_{n-1}) for s_{n-1} ∈ [0, 255)
    cond_counts = np.zeros((N_CATEGORIES, N_CATEGORIES), dtype=np.float64)
    cond_totals = np.zeros(N_CATEGORIES, dtype=np.float64)

    # Marginal counts for the first symbol
    marg_counts = np.zeros(N_CATEGORIES, dtype=np.float64)
    marg_total = 0.0

    encoder = constriction.stream.queue.RangeEncoder()
    K_alpha = N_CATEGORIES * alpha

    # First symbol: encode at uniform-prior marginal
    first = int(symbols[0])
    pmf0 = np.full(N_CATEGORIES, alpha) / K_alpha  # uniform with smoothing
    pmf0 = pmf0 / pmf0.sum()
    m0 = constriction.stream.model.Categorical(pmf0, perfect=False)
    encoder.encode(np.array([first], dtype=np.int32), m0)
    marg_counts[first] += 1
    marg_total += 1

    # Subsequent symbols: encode with cond pmf
    prev = first
    for n in range(1, symbols.size):
        cur = int(symbols[n])
        # Conditional PMF given prev
        row = cond_counts[prev]
        total_prev = cond_totals[prev]
        cond_pmf = (row + alpha) / (total_prev + K_alpha)
        cond_pmf = cond_pmf / cond_pmf.sum()  # numerical re-normalization
        m_cond = constriction.stream.model.Categorical(cond_pmf, perfect=False)
        encoder.encode(np.array([cur], dtype=np.int32), m_cond)
        # Update conditional counts
        cond_counts[prev, cur] += 1
        cond_totals[prev] += 1
        prev = cur

    encoded_words = encoder.get_compressed()
    encoded_bytes = encoded_words.tobytes()
    return encoded_bytes, {
        "n_symbols": int(symbols.size),
        "first_symbol": first,
        "encoded_bytes": len(encoded_bytes),
        "encoded_words": int(encoded_words.size),
    }


def decode_tensor_markov1(
    encoded_bytes: bytes,
    n_symbols: int,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """Decode a single tensor's symbols from Markov-1 AAC bytes.

    Mirrors encode_tensor_markov1 exactly: first symbol from uniform PMF,
    subsequent from conditional PMF given previous.
    """
    import constriction
    if n_symbols == 0:
        return np.zeros(0, dtype=np.int32)

    # Reconstruct as uint32 numpy array
    if len(encoded_bytes) % 4 != 0:
        raise ValueError(
            f"encoded_bytes length {len(encoded_bytes)} not a multiple of 4"
        )
    encoded_words = np.frombuffer(encoded_bytes, dtype=np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(encoded_words)

    cond_counts = np.zeros((N_CATEGORIES, N_CATEGORIES), dtype=np.float64)
    cond_totals = np.zeros(N_CATEGORIES, dtype=np.float64)
    K_alpha = N_CATEGORIES * alpha

    decoded = np.zeros(n_symbols, dtype=np.int32)

    # First symbol: uniform marginal
    pmf0 = np.full(N_CATEGORIES, alpha) / K_alpha
    pmf0 = pmf0 / pmf0.sum()
    m0 = constriction.stream.model.Categorical(pmf0, perfect=False)
    out0 = decoder.decode(m0, 1)
    decoded[0] = int(out0[0])
    prev = decoded[0]

    # Subsequent: conditional
    for n in range(1, n_symbols):
        row = cond_counts[prev]
        total_prev = cond_totals[prev]
        cond_pmf = (row + alpha) / (total_prev + K_alpha)
        cond_pmf = cond_pmf / cond_pmf.sum()
        m_cond = constriction.stream.model.Categorical(cond_pmf, perfect=False)
        out_n = decoder.decode(m_cond, 1)
        cur = int(out_n[0])
        decoded[n] = cur
        cond_counts[prev, cur] += 1
        cond_totals[prev] += 1
        prev = cur

    return decoded


# ---------------------------------------------------------------------------
# Whole-archive encode / decode (28 tensors)
# ---------------------------------------------------------------------------

def encode_archive_markov1(state_dict: dict, *, alpha: float = DEFAULT_ALPHA) -> tuple[bytes, list[dict]]:
    """Encode all 28 PR101 tensors as concatenated per-tensor Markov-1 streams.

    Wire format:
      header: magic 'M1AAC' (5B) + alpha (1B) + n_tensors (1B) + reserved (1B)
      per_tensor: scale_fp16 (2B) + n_symbols_uint32 (4B) + payload_len_uint32 (4B) + payload bytes

    Returns (archive_bytes, per_tensor_metadata).
    """
    rows: list[dict[str, Any]] = []
    payload_chunks: list[bytes] = []
    per_tensor_headers: list[bytes] = []

    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        symbols = (qt.q_i8.astype(np.int32) + 127).flatten()
        encoded_bytes, debug = encode_tensor_markov1(symbols, alpha=alpha)
        # Per-tensor header: scale (fp16) + n_symbols (uint32) + payload_len (uint32)
        scale_fp16 = np.float16(qt.scale).tobytes()  # 2 bytes
        n_symbols_bytes = struct.pack("<I", int(symbols.size))  # 4 bytes
        payload_len_bytes = struct.pack("<I", len(encoded_bytes))  # 4 bytes
        per_tensor_headers.append(scale_fp16 + n_symbols_bytes + payload_len_bytes)
        payload_chunks.append(encoded_bytes)
        rows.append({
            "name": name,
            "n_symbols": debug["n_symbols"],
            "first_symbol": debug["first_symbol"],
            "payload_bytes": debug["encoded_bytes"],
            "scale": float(qt.scale),
        })

    alpha_byte = struct.pack("<B", int(alpha * 10))  # alpha × 10 in [0, 255]
    header = b"M1AAC" + alpha_byte + struct.pack("<B", 28) + b"\x00"
    archive = (
        header
        + b"".join(per_tensor_headers)
        + b"".join(payload_chunks)
    )
    return archive, rows


def decode_archive_markov1(archive: bytes) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Decode an archive back to per-tensor symbol arrays + scale floats.

    Returns ({tensor_name: int32 symbols}, [metadata]).
    """
    if archive[:5] != b"M1AAC":
        raise ValueError(f"archive missing M1AAC magic: {archive[:5]!r}")
    alpha = archive[5] / 10.0
    n_tensors = archive[6]
    if n_tensors != 28:
        raise ValueError(f"unexpected tensor count {n_tensors}; expected 28")
    cursor = 8  # past magic+alpha+ntensors+reserved
    headers: list[tuple[float, int, int]] = []
    for _ in range(28):
        scale = float(np.frombuffer(archive[cursor:cursor+2], dtype=np.float16)[0])
        cursor += 2
        n_symbols = struct.unpack("<I", archive[cursor:cursor+4])[0]
        cursor += 4
        payload_len = struct.unpack("<I", archive[cursor:cursor+4])[0]
        cursor += 4
        headers.append((scale, n_symbols, payload_len))
    decoded_symbols: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for (name, _shape), (scale, n_symbols, payload_len) in zip(
        FIXED_STATE_SCHEMA, headers, strict=True
    ):
        payload = archive[cursor:cursor+payload_len]
        cursor += payload_len
        symbols = decode_tensor_markov1(payload, n_symbols, alpha=alpha)
        decoded_symbols[name] = symbols
        rows.append({
            "name": name,
            "n_symbols": n_symbols,
            "scale": scale,
        })
    return decoded_symbols, rows


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def _cmd_encode(args) -> int:
    state_dict = torch.load(args.state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit("loaded path is not a dict")
    t0 = time.time()
    archive, rows = encode_archive_markov1(state_dict, alpha=args.alpha)
    elapsed = time.time() - t0
    args.output_archive.parent.mkdir(parents=True, exist_ok=True)
    args.output_archive.write_bytes(archive)
    print(f"encoded {sum(r['n_symbols'] for r in rows):,} symbols in {elapsed:.1f}s")
    print(f"archive size: {len(archive):,} bytes (no zip overhead)")
    print(f"vs brotli+Optuna 178,144 - 16,094 = 162,050 payload: {len(archive) - 162050:+,}")
    return 0


def _cmd_round_trip(args) -> int:
    state_dict = torch.load(args.state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit("loaded path is not a dict")
    t0 = time.time()
    archive, encode_rows = encode_archive_markov1(state_dict, alpha=args.alpha)
    encode_elapsed = time.time() - t0
    t0 = time.time()
    decoded_symbols, decode_rows = decode_archive_markov1(archive)
    decode_elapsed = time.time() - t0
    # Verify byte-faithful round trip
    mismatches = []
    for name, _ in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        original = (qt.q_i8.astype(np.int32) + 127).flatten()
        roundtrip = decoded_symbols[name]
        if not np.array_equal(original, roundtrip):
            n_diff = int((original != roundtrip).sum())
            mismatches.append({
                "name": name,
                "n_elements": int(original.size),
                "n_mismatches": n_diff,
            })

    archive_overhead = 16_094
    summary = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": str(args.state_dict_path),
        "alpha": args.alpha,
        "n_tensors": 28,
        "n_total_symbols": sum(r["n_symbols"] for r in encode_rows),
        "archive_bytes_no_zip": len(archive),
        "archive_bytes_with_zip_overhead": len(archive) + archive_overhead,
        "comparison_brotli_optuna_payload_bytes": 162050,
        "comparison_brotli_optuna_archive_bytes": 178144,
        "savings_vs_brotli_payload": 162050 - len(archive),
        "savings_vs_brotli_archive": 178144 - (len(archive) + archive_overhead),
        "encode_elapsed_seconds": encode_elapsed,
        "decode_elapsed_seconds": decode_elapsed,
        "round_trip_byte_faithful": len(mismatches) == 0,
        "n_mismatched_tensors": len(mismatches),
        "mismatches": mismatches,
        "predicted_provable_markov1_payload_bytes": 152106,
        "actual_payload_minus_predicted": len(archive) - 152106,
        "encode_rows": encode_rows,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Round-trip results written to {args.output_summary}")
    print(f"Encode: {encode_elapsed:.1f}s ; Decode: {decode_elapsed:.1f}s")
    print(f"Archive size (M1AAC, no zip): {len(archive):,} bytes")
    print(f"Archive size (with 16,094 zip overhead estimate): {len(archive) + archive_overhead:,} bytes")
    print(f"vs brotli+Optuna payload (162,050):  {len(archive) - 162050:+,} bytes")
    print(f"vs brotli+Optuna archive (178,144):  {(len(archive) + archive_overhead) - 178144:+,} bytes")
    print(f"vs predicted provable floor (152,106): {len(archive) - 152106:+,} bytes (header overhead)")
    print(f"Round-trip byte-faithful: {len(mismatches) == 0}")
    if mismatches:
        print(f"  MISMATCHES: {mismatches}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enc = sub.add_parser("encode", help="encode a state_dict to M1AAC archive")
    p_enc.add_argument("--state-dict-path", type=Path, required=True)
    p_enc.add_argument("--output-archive", type=Path, required=True)
    p_enc.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p_enc.set_defaults(func=_cmd_encode)

    p_rt = sub.add_parser("round-trip", help="encode + decode + verify byte-faithful")
    p_rt.add_argument("--state-dict-path", type=Path, required=True)
    p_rt.add_argument("--output-summary", type=Path, required=True)
    p_rt.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p_rt.set_defaults(func=_cmd_round_trip)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
