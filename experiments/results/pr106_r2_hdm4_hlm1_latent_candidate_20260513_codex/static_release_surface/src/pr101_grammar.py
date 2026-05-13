"""PR101 ranked-Huffman/no-op sidecar grammar — DECODE ONLY.

Inflate-time decoder for the PR101 sidecar grammar applied to the PR106 r2
archive (lane_pr106_latent_sidecar_r2_pr101_grammar, format_id=0x02).

This file is a minimal DECODE-ONLY subset of ``tac.packet_compiler.pr101_sidecar_grammar``.
The full module also exports encoder helpers, centered-delta-uint8 packing, and
self-delimiting split-Brotli; those are unused at inflate time and are excluded
to keep the runtime tree closed under ``numpy`` + ``brotli`` (already present
for the format_id=0x01 path) — NO new external deps.

The decoder reconstructs the (dims, delta_indices) arrays bit-identical to the
brotli-decoded version that the format_id=0x01 path produces. PR106 base bytes
+ HNeRV decoder behaviour are unchanged by construction.

Source-of-truth implementation:
``src/tac/packet_compiler/pr101_sidecar_grammar.py``.
Golden-vector parity is enforced by tests/test_pr101_grammar_runtime_parity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np


@dataclass(frozen=True)
class RankedSidecarSchema:
    """Schema for a ranked Huffman/no-op sidecar over per-pair corrections.

    Mirrors the canonical schema in tac.packet_compiler.pr101_sidecar_grammar.
    PR106 r2 PR101-grammar variant uses:
        n_pairs=600, n_dims=28, deltas=(-2,-1,1,2),
        huff_min_len=2, huff_max_len=8, no_op_sentinel=255.
    """

    n_pairs: int
    n_dims: int
    deltas: tuple[int, ...]
    huff_min_len: int = 2
    huff_max_len: int = 8
    no_op_sentinel: int = 255

    @property
    def kraft_total(self) -> int:
        return 1 << self.huff_max_len


@cache
def _huff_length_vector_count(
    pos: int, remaining: int, *, n_symbols: int, huff_min_len: int, huff_max_len: int
) -> int:
    """Count Huffman length vectors of size ``n_symbols`` over Kraft budget."""
    if pos == n_symbols:
        return int(remaining == 0)
    total = 0
    for length in range(huff_min_len, huff_max_len + 1):
        weight = 1 << (huff_max_len - length)
        if remaining >= weight:
            total += _huff_length_vector_count(
                pos + 1,
                remaining - weight,
                n_symbols=n_symbols,
                huff_min_len=huff_min_len,
                huff_max_len=huff_max_len,
            )
    return total


def _decode_huff_length_rank(rank: int, schema: RankedSidecarSchema) -> np.ndarray:
    """Decode a Huffman-length vector from a non-negative integer rank."""
    n_symbols = len(schema.deltas)
    total = _huff_length_vector_count(
        0,
        schema.kraft_total,
        n_symbols=n_symbols,
        huff_min_len=schema.huff_min_len,
        huff_max_len=schema.huff_max_len,
    )
    if not (0 <= rank < total):
        raise ValueError(f"bad Huffman length-vector rank {rank}; total={total}")
    lengths = np.empty(n_symbols, dtype=np.uint8)
    remaining = schema.kraft_total
    for pos in range(n_symbols):
        emitted = False
        for length in range(schema.huff_min_len, schema.huff_max_len + 1):
            weight = 1 << (schema.huff_max_len - length)
            if remaining < weight:
                continue
            block = _huff_length_vector_count(
                pos + 1,
                remaining - weight,
                n_symbols=n_symbols,
                huff_min_len=schema.huff_min_len,
                huff_max_len=schema.huff_max_len,
            )
            if rank >= block:
                rank -= block
            else:
                lengths[pos] = length
                remaining -= weight
                emitted = True
                break
        if not emitted:
            raise ValueError("bad Huffman length-vector rank (no admissible length)")
    if remaining or rank:
        raise ValueError("bad Huffman length-vector rank (residue not exhausted)")
    return lengths


def _decode_combination_colex(rank: int, n: int, k: int) -> np.ndarray:
    """Decode a co-lex-ranked size-``k`` combination from ``range(n)``."""
    total = math.comb(n, k)
    if not (0 <= rank < total):
        raise ValueError(f"bad combination rank {rank}; C({n},{k})={total}")
    combo = [0] * k
    x = n
    for i in range(k, 0, -1):
        x -= 1
        while math.comb(x, i) > rank:
            x -= 1
        combo[i - 1] = x
        rank -= math.comb(x, i)
    if rank:
        raise ValueError("bad combination rank (residue not exhausted)")
    out = np.array(combo, dtype=np.int64)
    out.sort()
    return out


def _decode_canonical_huffman_n(
    data: bytes, lengths: np.ndarray, n_symbols: int
) -> np.ndarray:
    """Decode exactly ``n_symbols`` symbols from packed Huffman bytes."""
    decode: dict[tuple[int, int], int] = {}
    code = 0
    prev_len = 0
    for sym, length in sorted(
        ((sym, int(length)) for sym, length in enumerate(lengths) if length),
        key=lambda x: (x[1], x[0]),
    ):
        code <<= length - prev_len
        decode[(length, code)] = sym
        code += 1
        prev_len = length

    out = np.empty(n_symbols, dtype=np.int64)
    out_pos = 0
    cur = 0
    cur_len = 0
    for byte in data:
        for shift in range(7, -1, -1):
            cur = (cur << 1) | ((byte >> shift) & 1)
            cur_len += 1
            sym = decode.get((cur_len, cur))
            if sym is not None:
                out[out_pos] = sym
                out_pos += 1
                if out_pos == n_symbols:
                    return out
                cur = 0
                cur_len = 0
    raise ValueError("truncated Huffman payload")


def decode_ranked_no_op_sidecar(
    payload: bytes,
    *,
    schema: RankedSidecarSchema,
    dim_bytes: int,
    rank_bytes: int,
    noop_rank_bytes: int,
    noop_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of tac.packet_compiler.encode_ranked_no_op_sidecar.

    Returns
    -------
    (dims, delta_indices)
        Two ``(schema.n_pairs,)`` int64 arrays. ``dims[i] == schema.no_op_sentinel``
        marks a no-op slot (whose ``delta_indices[i]`` is meaningless / 0).
    """
    if dim_bytes <= 0 or rank_bytes <= 0 or noop_rank_bytes <= 0:
        raise ValueError("dim/rank/noop_rank byte widths must be positive")
    if not (0 <= noop_count <= schema.n_pairs):
        raise ValueError(f"noop_count must be in [0, {schema.n_pairs}]")

    n_valid = schema.n_pairs - noop_count
    expected_min = dim_bytes + rank_bytes + noop_rank_bytes
    if len(payload) < expected_min:
        raise ValueError(
            f"payload too short: got {len(payload)} bytes; "
            f"need at least {expected_min}"
        )
    pos = 0
    dim_blob = payload[pos : pos + dim_bytes]
    pos += dim_bytes
    length_rank_blob = payload[pos : pos + rank_bytes]
    pos += rank_bytes
    huff_bits = payload[pos : len(payload) - noop_rank_bytes]
    noop_rank_blob = payload[len(payload) - noop_rank_bytes :]

    length_rank = int.from_bytes(length_rank_blob, "little")
    lengths = _decode_huff_length_rank(length_rank, schema)

    if n_valid == 0:
        delta_indices_valid = np.empty(0, dtype=np.int64)
    else:
        delta_indices_valid = _decode_canonical_huffman_n(huff_bits, lengths, n_valid)

    noop_rank = int.from_bytes(noop_rank_blob, "little")
    if noop_count > 0:
        noop_pos = _decode_combination_colex(noop_rank, schema.n_pairs, noop_count)
    else:
        noop_pos = np.empty(0, dtype=np.int64)
    valid_mask = np.ones(schema.n_pairs, dtype=bool)
    valid_mask[noop_pos] = False

    dim_value = int.from_bytes(dim_blob, "little")
    dims_valid = np.empty(n_valid, dtype=np.int64)
    for i in range(n_valid):
        dim_value, dims_valid[i] = divmod(dim_value, schema.n_dims)
    if dim_value:
        raise ValueError(
            "trailing dim radix residue (corrupt sidecar or wrong dim_bytes)"
        )

    dims = np.full(schema.n_pairs, schema.no_op_sentinel, dtype=np.int64)
    delta_indices = np.zeros(schema.n_pairs, dtype=np.int64)
    dims[valid_mask] = dims_valid
    delta_indices[valid_mask] = delta_indices_valid
    return dims, delta_indices


__all__ = [
    "RankedSidecarSchema",
    "decode_ranked_no_op_sidecar",
]
