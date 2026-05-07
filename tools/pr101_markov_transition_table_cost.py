#!/usr/bin/env python3
"""Measure PR101 Markov transition-table side-information costs.

The Markov oracle floor is not deployable unless the decoder has the same
transition model. This tool computes several deterministic serializations of
per-tensor Markov-1 transition counts and compresses them with Brotli so the
model-cost claim is reproducible instead of anecdotal.

The output is planning evidence only. It emits no decoder bitstream and no
score-affecting archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_markov_transition_table_cost.py"
SCHEMA_VERSION = "pr101_markov_transition_table_cost.v1"
N_CATEGORIES = 255
ARCHIVE_OVERHEAD_BYTES = 16_094
REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES = 178_144
REFERENCE_MARKOV1_ORACLE_PAYLOAD_BYTES = 152_106
FIRST_SYMBOL_LITERAL_BYTES = len(FIXED_STATE_SCHEMA)


def _entropy_bits(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    bits = 0.0
    for count in counts.values():
        p = count / total
        bits -= p * math.log2(p)
    return bits


def _load_symbols(state_dict_path: Path) -> tuple[str, list[np.ndarray]]:
    input_bytes = state_dict_path.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {state_dict_path} is not a dict")
    tensor_symbols: list[np.ndarray] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        tensor_symbols.append((qt.q_i8.astype(np.int32) + 127).flatten())
    return input_sha256, tensor_symbols


def _transition_tables(tensor_symbols: list[np.ndarray]) -> list[np.ndarray]:
    tables: list[np.ndarray] = []
    for symbols in tensor_symbols:
        table = np.zeros((N_CATEGORIES, N_CATEGORIES), dtype=np.uint32)
        for prev, cur in pairwise(int(v) for v in symbols):
            table[prev, cur] += 1
        tables.append(table)
    return tables


def _markov1_oracle_payload_bytes(tensor_symbols: list[np.ndarray]) -> int:
    bits = 0.0
    for symbols in tensor_symbols:
        if symbols.size == 0:
            continue
        marginal_counts = Counter(int(v) for v in symbols)
        bits += _entropy_bits(marginal_counts)
        if symbols.size == 1:
            continue
        transition_counts = Counter(pairwise(int(v) for v in symbols))
        prev_counts = Counter(int(v) for v in symbols[:-1])
        for prev, n_prev in prev_counts.items():
            conditional = Counter({
                cur: transition_counts[(prev, cur)]
                for cur in range(N_CATEGORIES)
                if transition_counts[(prev, cur)] > 0
            })
            bits += n_prev * _entropy_bits(conditional)
    return math.ceil(bits / 8)


def _dense_u16_blob(tables: list[np.ndarray]) -> bytes:
    max_count = max(int(table.max()) for table in tables)
    if max_count > np.iinfo(np.uint16).max:
        raise ValueError(f"dense_u16 cannot store max transition count {max_count}")
    return b"".join(table.astype("<u2", copy=False).tobytes() for table in tables)


def _dense_u32_blob(tables: list[np.ndarray]) -> bytes:
    return b"".join(table.astype("<u4", copy=False).tobytes() for table in tables)


def _sparse_u16_blob(tables: list[np.ndarray]) -> bytes:
    chunks = [b"S16\x00", struct.pack("<H", len(tables))]
    for tensor_idx, table in enumerate(tables):
        coords = np.argwhere(table > 0)
        chunks.append(struct.pack("<BH", tensor_idx, int(coords.shape[0])))
        for prev, cur in coords:
            count = int(table[prev, cur])
            if count > np.iinfo(np.uint16).max:
                raise ValueError(f"sparse_u16 cannot store transition count {count}")
            chunks.append(struct.pack("<BBH", int(prev), int(cur), count))
    return b"".join(chunks)


def _sparse_varint_blob(tables: list[np.ndarray]) -> bytes:
    def put_varint(value: int, out: bytearray) -> None:
        while value >= 0x80:
            out.append((value & 0x7F) | 0x80)
            value >>= 7
        out.append(value)

    out = bytearray(b"SV1\x00")
    put_varint(len(tables), out)
    for tensor_idx, table in enumerate(tables):
        coords = np.argwhere(table > 0)
        put_varint(tensor_idx, out)
        put_varint(int(coords.shape[0]), out)
        last_flat = 0
        for prev, cur in coords:
            flat = int(prev) * N_CATEGORIES + int(cur)
            put_varint(flat - last_flat, out)
            put_varint(int(table[prev, cur]), out)
            last_flat = flat
    return bytes(out)


def measure_transition_table_cost(state_dict_path: Path) -> dict[str, Any]:
    input_sha256, tensor_symbols = _load_symbols(state_dict_path)
    tables = _transition_tables(tensor_symbols)
    n_nonzero_pairs = int(sum(np.count_nonzero(table) for table in tables))
    max_count = int(max(table.max() for table in tables))
    oracle_payload = _markov1_oracle_payload_bytes(tensor_symbols)

    serializations = []
    for name, builder in [
        ("dense_u16", _dense_u16_blob),
        ("dense_u32", _dense_u32_blob),
        ("sparse_u16", _sparse_u16_blob),
        ("sparse_varint", _sparse_varint_blob),
    ]:
        try:
            blob = builder(tables)
            compressed = brotli.compress(blob, quality=11)
            serializations.append({
                "name": name,
                "raw_bytes": len(blob),
                "brotli_bytes": len(compressed),
                "oracle_payload_plus_brotli_table_bytes": oracle_payload
                + len(compressed),
                "oracle_payload_plus_brotli_table_and_first_symbols_bytes": oracle_payload
                + len(compressed)
                + FIRST_SYMBOL_LITERAL_BYTES,
                "archive_bytes_with_oracle_payload_table_and_zip": oracle_payload
                + len(compressed)
                + ARCHIVE_OVERHEAD_BYTES,
                "archive_bytes_with_oracle_payload_table_first_symbols_and_zip": oracle_payload
                + len(compressed)
                + FIRST_SYMBOL_LITERAL_BYTES
                + ARCHIVE_OVERHEAD_BYTES,
                "delta_archive_vs_brotli_optuna": oracle_payload
                + len(compressed)
                + ARCHIVE_OVERHEAD_BYTES
                - REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
                "delta_archive_with_first_symbols_vs_brotli_optuna": oracle_payload
                + len(compressed)
                + FIRST_SYMBOL_LITERAL_BYTES
                + ARCHIVE_OVERHEAD_BYTES
                - REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
            })
        except ValueError as exc:
            serializations.append({"name": name, "error": str(exc)})
    serializations.sort(key=lambda row: row.get("brotli_bytes", 10**18))

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": str(state_dict_path),
        "input_state_dict_sha256": input_sha256,
        "evidence_grade": "empirical",
        "evidence_semantics": "cpu_markov_transition_table_sideinfo_cost",
        "score_claim": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "model_sideinfo_cost_probe_only",
            "oracle_payload_not_an_actual_coder_bitstream",
            "transition_table_not_wired_into_decoder",
            "no_archive_substitution_performed",
            "missing_exact_cuda_auth_eval",
        ],
        "n_tensors": len(tables),
        "n_categories": N_CATEGORIES,
        "n_nonzero_transition_pairs": n_nonzero_pairs,
        "max_transition_count": max_count,
        "reference_markov1_oracle_payload_bytes": REFERENCE_MARKOV1_ORACLE_PAYLOAD_BYTES,
        "recomputed_markov1_oracle_payload_bytes": oracle_payload,
        "first_symbol_literal_bytes": FIRST_SYMBOL_LITERAL_BYTES,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "comparison_brotli_optuna_archive_bytes": REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
        "serializations": serializations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")

    manifest = measure_transition_table_cost(args.state_dict_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"nonzero transition pairs: {manifest['n_nonzero_transition_pairs']:,}")
    print(f"Markov-1 oracle payload:  {manifest['recomputed_markov1_oracle_payload_bytes']:,}")
    for row in manifest["serializations"]:
        if "error" in row:
            print(f"{row['name']:<14} ERROR {row['error']}")
            continue
        print(
            f"{row['name']:<14} raw={row['raw_bytes']:>9,} "
            f"brotli={row['brotli_bytes']:>8,} "
            f"archive={row['archive_bytes_with_oracle_payload_table_first_symbols_and_zip']:>9,} "
            f"delta={row['delta_archive_with_first_symbols_vs_brotli_optuna']:>+8,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
