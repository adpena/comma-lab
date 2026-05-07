#!/usr/bin/env python3
"""Compression-floor ladder for PR101 weights.

Computes Shannon-source-coding-theorem floors at increasing model
complexity, making explicit which source model each practical encoder
approaches:

  floor_iid_per_tensor   = sum_t |X_t| * H(X_t)          [marginal, per-tensor]
  floor_iid_joint_pooled = N * H(X_pooled)               [marginal, single dist]
  floor_markov1          = sum_n H(s_n | s_{n-1})        [1-symbol context]
  floor_markov2          = sum_n H(s_n | s_{n-1}, s_{n-2}) [2-symbol context]
  floor_conditional_tensor = sum_t |X_t| * H(X_t)        [tensor as context]

Shannon's source coding theorem: no uniquely-decodable prefix code can
use fewer bits per symbol than the entropy of the declared source model,
and arithmetic/range coders can approach that model cost closely.
For an iid source the ideal length is N * H(X) bits; for context-aware
sources it becomes an empirical conditional-entropy ladder.

These are model-class floors, not a proof of the global optimum over all
possible compressors. Higher-context rows also omit the cost of transmitting
or learning the model. They are useful as a deterministic research map:
which signal class is worth engineering next?

Pure CPU + numpy. No GPU, no scorer load. No contest score claims.

Usage::

    .venv/bin/python tools/pr101_provable_optimal_floor.py \\
        --state-dict-path experiments/.../pr101_decoder_state_dict.pt \\
        --output reports/pr101_provable_optimal_floor.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_provable_optimal_floor.py"
SCHEMA_VERSION = "pr101_compression_floor_ladder.v2"
N_CATEGORIES = 255
EVIDENCE_GRADE = "derivation"
EVIDENCE_SEMANTICS = "cpu_model_class_entropy_floor_ladder"
REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES = 178_144
REFERENCE_BROTLI_OPTUNA_PAYLOAD_BYTES = 162_050
REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES = 178_181
REFERENCE_PER_TENSOR_AAC_PAYLOAD_BYTES = 161_975
REFERENCE_JOINT_AAC_ARCHIVE_BYTES = 202_999
REFERENCE_JOINT_AAC_PAYLOAD_BYTES = 186_793
REFERENCE_NAIVE_MARKOV1_AAC_ARCHIVE_BYTES = 199_238
REFERENCE_NAIVE_MARKOV1_AAC_PAYLOAD_BYTES = 183_144


def _entropy_bits(counts: np.ndarray | dict) -> float:
    """Shannon entropy of a histogram, in bits per occurrence (returns
    bits per symbol)."""
    if isinstance(counts, dict):
        total = sum(counts.values())
        if total == 0:
            return 0.0
        h = 0.0
        for n in counts.values():
            if n == 0:
                continue
            p = n / total
            h -= p * math.log2(p)
        return h
    arr = np.asarray(counts, dtype=np.float64)
    total = arr.sum()
    if total == 0:
        return 0.0
    nonzero = arr[arr > 0]
    p = nonzero / total
    return float(-np.sum(p * np.log2(p)))


def floor_iid_per_tensor(tensor_data: list[np.ndarray]) -> float:
    """Per-tensor empirical-PMF iid floor in BITS. Each tensor encoded
    against its own empirical distribution. Provable lower bound for
    any per-tensor static encoder when the PMF is free or transmitted."""
    bits = 0.0
    for syms in tensor_data:
        counts = np.bincount(syms, minlength=N_CATEGORIES).astype(np.float64)
        H = _entropy_bits(counts)
        bits += syms.size * H
    return bits


def floor_iid_joint_pooled(tensor_data: list[np.ndarray]) -> float:
    """Joint-pooled iid floor in BITS. All symbols treated as iid from
    one shared empirical PMF. Model-class floor for iid encoders that do
    not distinguish between tensors."""
    pooled = np.concatenate(tensor_data)
    counts = np.bincount(pooled, minlength=N_CATEGORIES).astype(np.float64)
    H = _entropy_bits(counts)
    return pooled.size * H


def floor_markov1_per_tensor(tensor_data: list[np.ndarray]) -> float:
    """1st-order Markov floor in BITS. Each symbol coded conditional on
    the PRIOR symbol within the same tensor. Joint distribution is
    modeled as P(s_n | s_{n-1}) where context resets at tensor
    boundaries.

    Oracle empirical floor for 1-symbol-context encoders. The model-table
    cost is not included.
    """
    bits = 0.0
    for syms in tensor_data:
        # First symbol: encoded with marginal empirical PMF
        if syms.size == 0:
            continue
        marginal_counts = np.bincount(syms, minlength=N_CATEGORIES).astype(np.float64)
        H_marginal = _entropy_bits(marginal_counts)
        bits += H_marginal  # cost of first symbol
        # Subsequent symbols: P(s_n | s_{n-1})
        # Build joint count table
        if syms.size > 1:
            pairs = list(pairwise(syms))
            pair_counter: Counter = Counter(pairs)
            # Conditional entropy: H(Y|X) = sum_x P(x) H(Y | X=x)
            x_counter: Counter = Counter(syms[:-1])
            for x, n_x in x_counter.items():
                # Conditional PMF given X=x
                conditional = {
                    y: pair_counter.get((x, y), 0)
                    for y in range(N_CATEGORIES)
                    if pair_counter.get((x, y), 0) > 0
                }
                H_y_given_x = _entropy_bits(conditional)
                bits += n_x * H_y_given_x
    return bits


def floor_conditional_on_tensor(tensor_data: list[np.ndarray]) -> float:
    """Conditional joint floor: H(X | tensor_id). Treats the tensor
    index itself as side-information. Equivalent to per-tensor iid
    floor since tensor_id is a function of position; here for
    completeness in the ladder."""
    return floor_iid_per_tensor(tensor_data)


def floor_markov2_per_tensor(tensor_data: list[np.ndarray]) -> float:
    """2nd-order Markov oracle floor in BITS.

    Symbols after the first two are coded as P(s_n | s_{n-1}, s_{n-2})
    with context resets at tensor boundaries. The model-table cost is not
    included, so this is a research target rather than a packet budget.
    """
    bits = 0.0
    for syms in tensor_data:
        if syms.size == 0:
            continue
        marginal_counts = np.bincount(syms, minlength=N_CATEGORIES).astype(np.float64)
        H_marginal = _entropy_bits(marginal_counts)
        bits += H_marginal
        if syms.size == 1:
            continue
        pairs_first = Counter(pairwise(syms))
        prev_counts = Counter(syms[:-1])
        first_prev = int(syms[0])
        second = int(syms[1])
        p_second = pairs_first[(first_prev, second)] / prev_counts[first_prev]
        bits += -math.log2(p_second)
        if syms.size == 2:
            continue
        contexts = list(pairwise(syms[:-1]))
        context_counts = Counter(contexts)
        triples = Counter(
            (*context, symbol)
            for context, symbol in zip(contexts, syms[2:], strict=False)
        )
        for context, n_context in context_counts.items():
            conditional = {
                y: triples.get((*context, y), 0)
                for y in range(N_CATEGORIES)
                if triples.get((*context, y), 0) > 0
            }
            bits += n_context * _entropy_bits(conditional)
    return bits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")

    input_bytes = args.state_dict_path.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    state_dict = torch.load(args.state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {args.state_dict_path} is not a dict")
    tensor_data: list[np.ndarray] = []
    for name, _ in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        symbols = (qt.q_i8.astype(np.int32) + 127).flatten()
        tensor_data.append(symbols)

    n_total = sum(t.size for t in tensor_data)

    bits_per_tensor = floor_iid_per_tensor(tensor_data)
    bits_pooled = floor_iid_joint_pooled(tensor_data)
    bits_markov1 = floor_markov1_per_tensor(tensor_data)
    bits_markov2 = floor_markov2_per_tensor(tensor_data)
    bits_conditional_tensor = floor_conditional_on_tensor(tensor_data)

    archive_overhead = 16_094

    floors = [
        {
            "name": "iid_joint_pooled",
            "description": "All symbols treated as iid from a single pooled empirical PMF. Provable iid-pooled floor.",
            "model_table_overhead_included": False,
            "bits": bits_pooled,
            "bytes_payload": math.ceil(bits_pooled / 8),
            "bytes_archive": math.ceil(bits_pooled / 8) + archive_overhead,
        },
        {
            "name": "iid_per_tensor",
            "description": "Each tensor encoded against its own empirical PMF. Provable per-tensor-marginal floor.",
            "model_table_overhead_included": False,
            "bits": bits_per_tensor,
            "bytes_payload": math.ceil(bits_per_tensor / 8),
            "bytes_archive": math.ceil(bits_per_tensor / 8) + archive_overhead,
        },
        {
            "name": "conditional_on_tensor",
            "description": "Same as iid_per_tensor; tensor index treated as side-info.",
            "model_table_overhead_included": False,
            "bits": bits_conditional_tensor,
            "bytes_payload": math.ceil(bits_conditional_tensor / 8),
            "bytes_archive": math.ceil(bits_conditional_tensor / 8) + archive_overhead,
        },
        {
            "name": "markov1_per_tensor",
            "description": "1st-order Markov oracle: P(s_n | s_{n-1}). Context table overhead omitted.",
            "model_table_overhead_included": False,
            "bits": bits_markov1,
            "bytes_payload": math.ceil(bits_markov1 / 8),
            "bytes_archive": math.ceil(bits_markov1 / 8) + archive_overhead,
        },
        {
            "name": "markov2_per_tensor",
            "description": "2nd-order Markov oracle: P(s_n | s_{n-1}, s_{n-2}). Context table overhead omitted.",
            "model_table_overhead_included": False,
            "bits": bits_markov2,
            "bytes_payload": math.ceil(bits_markov2 / 8),
            "bytes_archive": math.ceil(bits_markov2 / 8) + archive_overhead,
        },
    ]
    floors.sort(key=lambda f: f["bits"])

    # Empirical encoders for context (bytes payload only)
    empirical = [
        {"name": "brotli_optuna_optimum", "bytes_payload": REFERENCE_BROTLI_OPTUNA_PAYLOAD_BYTES,
         "bytes_archive": REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES, "ratio_to_per_tensor_floor": None},
        {"name": "per_tensor_aac_zero_pmf", "bytes_payload": REFERENCE_PER_TENSOR_AAC_PAYLOAD_BYTES,
         "bytes_archive": REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES, "ratio_to_per_tensor_floor": None},
        {"name": "joint_aac_zero_pmf", "bytes_payload": REFERENCE_JOINT_AAC_PAYLOAD_BYTES,
         "bytes_archive": REFERENCE_JOINT_AAC_ARCHIVE_BYTES, "ratio_to_per_tensor_floor": None},
        {"name": "naive_markov1_aac_round_trip",
         "bytes_payload": REFERENCE_NAIVE_MARKOV1_AAC_PAYLOAD_BYTES,
         "bytes_archive": REFERENCE_NAIVE_MARKOV1_AAC_ARCHIVE_BYTES,
         "ratio_to_per_tensor_floor": None},
    ]
    iid_pt_payload = math.ceil(bits_per_tensor / 8)
    for e in empirical:
        e["ratio_to_per_tensor_floor"] = e["bytes_payload"] / iid_pt_payload

    manifest = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": str(args.state_dict_path),
        "input_state_dict_sha256": input_sha256,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "entropy_floor_ladder_only",
            "model_table_overhead_omitted_for_context_rows",
            "no_actual_context_coder_bitstream",
            "no_archive_substitution_performed",
            "missing_exact_cuda_auth_eval",
        ],
        "n_total_symbols": n_total,
        "n_categories": N_CATEGORIES,
        "archive_overhead_bytes": archive_overhead,
        "provable_floors": floors,
        "empirical_encoders": empirical,
        "shannon_source_coding_theorem_statement": (
            "For a declared source model, no uniquely-decodable prefix code "
            "can beat that model's entropy in expectation; arithmetic or "
            "range coders can approach the model code length closely. These "
            "rows are model-class floors, not a proof of the unrestricted "
            "global optimum over all possible compressors."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"\nN={n_total:,} symbols, K={N_CATEGORIES}\n")
    print(f"{'PROVABLE FLOOR':<28} {'bits':>15} {'bytes_payload':>15} {'bytes_archive':>15}")
    for f in floors:
        print(f"  {f['name']:<26} {f['bits']:>15,.1f} {f['bytes_payload']:>15,} {f['bytes_archive']:>15,}")
    print(f"\n{'EMPIRICAL ENCODER':<28} {'bytes_payload':>15} {'bytes_archive':>15} {'x iid_per_tensor':>18}")
    for e in empirical:
        print(f"  {e['name']:<26} {e['bytes_payload']:>15,} {e['bytes_archive']:>15,} {e['ratio_to_per_tensor_floor']:>17.4f}x")
    print(f"\n{'INTERPRETATION':<28}")
    iid_pt = next(f for f in floors if f["name"] == "iid_per_tensor")
    markov1 = next(f for f in floors if f["name"] == "markov1_per_tensor")
    print(f"  iid_per_tensor floor:      {iid_pt['bytes_payload']:,} payload / {iid_pt['bytes_archive']:,} archive")
    print(f"  AAC achieves:              161,975 payload / 178,181 archive ({REFERENCE_PER_TENSOR_AAC_PAYLOAD_BYTES/iid_pt['bytes_payload']*100:.2f}% of floor)")
    print(f"  brotli achieves:           162,050 payload / 178,144 archive ({REFERENCE_BROTLI_OPTUNA_PAYLOAD_BYTES/iid_pt['bytes_payload']*100:.2f}% of floor)")
    print(f"  markov1 floor:             {markov1['bytes_payload']:,} payload / {markov1['bytes_archive']:,} archive")
    print(f"  -> markov1 saves {iid_pt['bytes_payload'] - markov1['bytes_payload']:,} bytes vs iid_per_tensor")
    print("  -> No deployable encoder we've measured uses 1-symbol context yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
