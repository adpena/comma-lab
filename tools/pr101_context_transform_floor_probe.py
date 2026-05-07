#!/usr/bin/env python3
"""Context-transform entropy floor probe for PR101 quantized weights.

This tool asks whether deterministic representation transforms can expose
lower entropy to a context coder than the raw PR101 symbol stream:

- identity symbols in [0, 254]
- signed zigzag symbols
- modulo-255 deltas
- high/low nibble byte slicing
- bitplanes
- zero-mask plus nonzero-value categorical split
- abs plus sign split

Every transform here is fixed and invertible without learned side data. The
reported floors are model-class planning evidence only: IID, Markov-1, and
Markov-2 oracle entropies. Context model/table costs are not charged, and no
decoder bitstream or archive is emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
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

TOOL_NAME = "tools/pr101_context_transform_floor_probe.py"
SCHEMA_VERSION = "pr101_context_transform_floor_probe.v1"
EVIDENCE_GRADE = "derivation"
EVIDENCE_SEMANTICS = "cpu_invertible_transform_entropy_floor_probe"
ARCHIVE_OVERHEAD_BYTES = 16_094
REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES = 178_144


@dataclass(frozen=True)
class SymbolStream:
    name: str
    symbols: np.ndarray
    n_categories: int


def _entropy_bits(counts: np.ndarray | Counter[int]) -> float:
    if isinstance(counts, Counter):
        total = sum(counts.values())
        values = counts.values()
    else:
        arr = np.asarray(counts, dtype=np.float64)
        total = float(arr.sum())
        values = arr[arr > 0]
    if total == 0:
        return 0.0
    bits = 0.0
    for count in values:
        if count == 0:
            continue
        p = float(count) / total
        bits -= p * math.log2(p)
    return bits


def _iid_bits(streams: list[SymbolStream]) -> float:
    bits = 0.0
    for stream in streams:
        counts = np.bincount(stream.symbols, minlength=stream.n_categories)
        bits += stream.symbols.size * _entropy_bits(counts)
    return bits


def _markov1_bits(streams: list[SymbolStream]) -> float:
    bits = 0.0
    for stream in streams:
        syms = stream.symbols
        if syms.size == 0:
            continue
        counts = np.bincount(syms, minlength=stream.n_categories)
        bits += _entropy_bits(counts)
        if syms.size == 1:
            continue
        pair_counts = Counter(pairwise(int(v) for v in syms))
        prev_counts = Counter(int(v) for v in syms[:-1])
        for prev, count_prev in prev_counts.items():
            conditional = Counter({
                cur: pair_counts[(prev, cur)]
                for cur in range(stream.n_categories)
                if pair_counts[(prev, cur)] > 0
            })
            bits += count_prev * _entropy_bits(conditional)
    return bits


def _markov2_bits(streams: list[SymbolStream]) -> float:
    bits = 0.0
    for stream in streams:
        syms = stream.symbols
        if syms.size == 0:
            continue
        counts = np.bincount(syms, minlength=stream.n_categories)
        bits += _entropy_bits(counts)
        if syms.size == 1:
            continue
        syms_int = [int(v) for v in syms]
        pairs = list(pairwise(syms_int))
        pair_counts = Counter(pairs)
        prev_counts = Counter(syms_int[:-1])
        first_prev = syms_int[0]
        second = syms_int[1]
        p_second = pair_counts[(first_prev, second)] / prev_counts[first_prev]
        bits += -math.log2(p_second)
        if syms.size == 2:
            continue
        contexts = list(pairwise(syms_int[:-1]))
        context_counts = Counter(contexts)
        triple_counts = Counter(
            (*context, symbol)
            for context, symbol in zip(contexts, syms_int[2:], strict=False)
        )
        for context, count_context in context_counts.items():
            conditional = Counter({
                cur: triple_counts[(*context, cur)]
                for cur in range(stream.n_categories)
                if triple_counts[(*context, cur)] > 0
            })
            bits += count_context * _entropy_bits(conditional)
    return bits


def _signed_zigzag(symbols: np.ndarray) -> np.ndarray:
    signed = symbols.astype(np.int32) - 127
    return np.where(signed >= 0, 2 * signed, -2 * signed - 1).astype(np.int32)


def _nonzero_value_index(values: np.ndarray) -> np.ndarray:
    return np.where(values < 127, values, values - 1).astype(np.int32)


def _load_base_symbols(state_dict_path: Path) -> tuple[str, list[tuple[str, np.ndarray]]]:
    input_bytes = state_dict_path.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {state_dict_path} is not a dict")

    rows: list[tuple[str, np.ndarray]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        symbols = (qt.q_i8.astype(np.int32) + 127).flatten()
        rows.append((name, symbols))
    return input_sha256, rows


def _identity(rows: list[tuple[str, np.ndarray]]) -> list[SymbolStream]:
    return [SymbolStream(name, symbols, 255) for name, symbols in rows]


def _zigzag(rows: list[tuple[str, np.ndarray]]) -> list[SymbolStream]:
    return [SymbolStream(name, _signed_zigzag(symbols), 255) for name, symbols in rows]


def _delta_mod255(rows: list[tuple[str, np.ndarray]]) -> list[SymbolStream]:
    streams: list[SymbolStream] = []
    for name, symbols in rows:
        if symbols.size == 0:
            deltas = symbols
        else:
            deltas = np.empty_like(symbols)
            deltas[0] = symbols[0]
            deltas[1:] = (symbols[1:] - symbols[:-1]) % 255
        streams.append(SymbolStream(name, deltas.astype(np.int32), 255))
    return streams


def _nibbles(rows: list[tuple[str, np.ndarray]]) -> list[SymbolStream]:
    streams: list[SymbolStream] = []
    for name, symbols in rows:
        streams.append(SymbolStream(f"{name}:hi4", (symbols >> 4).astype(np.int32), 16))
        streams.append(SymbolStream(f"{name}:lo4", (symbols & 15).astype(np.int32), 16))
    return streams


def _bitplanes(rows: list[tuple[str, np.ndarray]]) -> list[SymbolStream]:
    streams: list[SymbolStream] = []
    for name, symbols in rows:
        for bit in range(8):
            streams.append(
                SymbolStream(f"{name}:bit{bit}", ((symbols >> bit) & 1).astype(np.int32), 2)
            )
    return streams


def _zero_mask_nonzero_value(rows: list[tuple[str, np.ndarray]]) -> list[SymbolStream]:
    streams: list[SymbolStream] = []
    for name, symbols in rows:
        zero_mask = (symbols == 127).astype(np.int32)
        streams.append(SymbolStream(f"{name}:zero_mask", zero_mask, 2))
        nonzero = symbols[symbols != 127]
        streams.append(
            SymbolStream(f"{name}:nonzero_value", _nonzero_value_index(nonzero), 254)
        )
    return streams


def _abs_sign(rows: list[tuple[str, np.ndarray]]) -> list[SymbolStream]:
    streams: list[SymbolStream] = []
    for name, symbols in rows:
        signed = symbols.astype(np.int32) - 127
        streams.append(SymbolStream(f"{name}:abs", np.abs(signed).astype(np.int32), 128))
        streams.append(SymbolStream(f"{name}:sign", (signed < 0).astype(np.int32), 2))
    return streams


TRANSFORMS = {
    "identity": _identity,
    "signed_zigzag": _zigzag,
    "delta_mod255": _delta_mod255,
    "nibble_split": _nibbles,
    "bitplanes": _bitplanes,
    "zero_mask_nonzero_value": _zero_mask_nonzero_value,
    "abs_sign_split": _abs_sign,
}


def build_transform_floor_report(state_dict_path: Path) -> dict[str, Any]:
    input_sha256, rows = _load_base_symbols(state_dict_path)
    transform_rows: list[dict[str, Any]] = []
    for transform_name, builder in TRANSFORMS.items():
        streams = builder(rows)
        iid_bits = _iid_bits(streams)
        markov1_bits = _markov1_bits(streams)
        markov2_bits = _markov2_bits(streams)
        n_symbols_total = int(sum(stream.symbols.size for stream in streams))
        transform_rows.append({
            "transform": transform_name,
            "invertible_fixed_transform": True,
            "metadata_bytes_charged": 0,
            "n_streams": len(streams),
            "n_symbols_total": n_symbols_total,
            "iid_payload_bytes": math.ceil(iid_bits / 8),
            "markov1_payload_bytes": math.ceil(markov1_bits / 8),
            "markov2_payload_bytes": math.ceil(markov2_bits / 8),
            "iid_archive_bytes": math.ceil(iid_bits / 8) + ARCHIVE_OVERHEAD_BYTES,
            "markov1_archive_bytes": math.ceil(markov1_bits / 8) + ARCHIVE_OVERHEAD_BYTES,
            "markov2_archive_bytes": math.ceil(markov2_bits / 8) + ARCHIVE_OVERHEAD_BYTES,
        })

    identity = next(row for row in transform_rows if row["transform"] == "identity")
    for row in transform_rows:
        row["delta_markov1_payload_vs_identity"] = (
            row["markov1_payload_bytes"] - identity["markov1_payload_bytes"]
        )
        row["delta_markov2_payload_vs_identity"] = (
            row["markov2_payload_bytes"] - identity["markov2_payload_bytes"]
        )
        row["delta_markov1_archive_vs_brotli_optuna"] = (
            row["markov1_archive_bytes"] - REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES
        )
        row["delta_markov2_archive_vs_brotli_optuna"] = (
            row["markov2_archive_bytes"] - REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES
        )

    transform_rows.sort(key=lambda row: row["markov1_payload_bytes"])
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": str(state_dict_path),
        "input_state_dict_sha256": input_sha256,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "entropy_floor_probe_only",
            "model_table_overhead_omitted",
            "no_actual_transform_coder_bitstream",
            "no_archive_substitution_performed",
            "missing_exact_cuda_auth_eval",
        ],
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "comparison_brotli_optuna_archive_bytes": REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
        "transforms": transform_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")

    manifest = build_transform_floor_report(args.state_dict_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(
        f"{'transform':<28} {'streams':>7} {'iid':>10} "
        f"{'markov1':>10} {'markov2':>10} {'m1_vs_brotli':>13}"
    )
    for row in manifest["transforms"]:
        print(
            f"{row['transform']:<28} {row['n_streams']:>7} "
            f"{row['iid_payload_bytes']:>10,} "
            f"{row['markov1_payload_bytes']:>10,} "
            f"{row['markov2_payload_bytes']:>10,} "
            f"{row['delta_markov1_archive_vs_brotli_optuna']:>+13,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
