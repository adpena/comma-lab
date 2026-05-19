"""Latent sidecar decoding helpers for the FEC6 PR101 packet."""

from functools import lru_cache
import math

import brotli
import numpy as np
import torch


N_PAIRS = 600
LATENT_DIM = 28
SIDECAR_DELTAS_X100 = np.array(
    [-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10],
    dtype=np.int8,
)
SIDECAR_BASE = 1 + LATENT_DIM * len(SIDECAR_DELTAS_X100)
SIDECAR_PACKED_LEN = 661
SIDECAR_SPLIT_LEN = 656
SIDECAR_HUFF_LEN = 614
SIDECAR_HUFF_ENUM_LEN = 607
SIDECAR_HUFF_COMB_LEN = 609
SIDECAR_NOOP_RANK_PREFIX_LEN = 4
SIDECAR_NOOP_INFER_RANK_LEN = 3
SIDECAR_NOOP_TABLE_LEN = 7
SIDECAR_DIM_PACKED_LEN = 359
SIDECAR_DELTA_HUFF_LENGTHS_LEN = 8
SIDECAR_DELTA_HUFF3_LENGTHS_LEN = 6
SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN = 5
SIDECAR_HUFF_MIN_LEN = 2
SIDECAR_HUFF_MAX_LEN = 8
SIDECAR_HUFF_KRAFT_TOTAL = 1 << SIDECAR_HUFF_MAX_LEN


def unpack_nibbles(data, n):
    """Expand packed low/high nibbles into the first n uint8 symbols."""
    arr = np.frombuffer(data, dtype=np.uint8)
    unpacked = np.empty(arr.size * 2, dtype=np.uint8)
    unpacked[0::2] = arr & 15
    unpacked[1::2] = arr >> 4
    return unpacked[:n]


def unpack_3bit_lengths(data, n, offset):
    """Unpack big-endian 3-bit Huffman lengths plus a fixed offset."""
    out = np.empty(n, dtype=np.uint8)
    bit_pos = 0
    for i in range(n):
        value = 0
        for _ in range(3):
            byte = data[bit_pos // 8]
            value = (value << 1) | ((byte >> (7 - (bit_pos % 8))) & 1)
            bit_pos += 1
        out[i] = value + offset
    return out


def decode_canonical_huffman(data, lengths, n_symbols):
    """Decode exactly n_symbols from a canonical MSB-first Huffman stream."""
    decode = {}
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

    out = np.empty(n_symbols, dtype=np.uint8)
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
    raise ValueError("truncated Huffman sidecar")


def decode_canonical_huffman_all(data, lengths):
    """Decode a complete canonical Huffman stream and reject dangling bits."""
    decode = {}
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

    out = []
    cur = 0
    cur_len = 0
    for byte in data:
        for shift in range(7, -1, -1):
            cur = (cur << 1) | ((byte >> shift) & 1)
            cur_len += 1
            sym = decode.get((cur_len, cur))
            if sym is not None:
                out.append(sym)
                cur = 0
                cur_len = 0
    if cur_len:
        raise ValueError("truncated Huffman sidecar")
    return np.array(out, dtype=np.uint8)


@lru_cache(None)
def huff_length_vector_count(pos, remaining):
    """Count remaining valid canonical length vectors for rank decoding."""
    if pos == len(SIDECAR_DELTAS_X100):
        return int(remaining == 0)
    total = 0
    for length in range(SIDECAR_HUFF_MIN_LEN, SIDECAR_HUFF_MAX_LEN + 1):
        weight = 1 << (SIDECAR_HUFF_MAX_LEN - length)
        if remaining >= weight:
            total += huff_length_vector_count(pos + 1, remaining - weight)
    return total


def decode_huff_length_rank(rank):
    """Recover a canonical Huffman length vector from its colex rank."""
    if rank >= huff_length_vector_count(0, SIDECAR_HUFF_KRAFT_TOTAL):
        raise ValueError("bad Huffman length-vector rank")
    lengths = np.empty(len(SIDECAR_DELTAS_X100), dtype=np.uint8)
    remaining = SIDECAR_HUFF_KRAFT_TOTAL
    for pos in range(lengths.size):
        for length in range(SIDECAR_HUFF_MIN_LEN, SIDECAR_HUFF_MAX_LEN + 1):
            weight = 1 << (SIDECAR_HUFF_MAX_LEN - length)
            if remaining < weight:
                continue
            block = huff_length_vector_count(pos + 1, remaining - weight)
            if rank >= block:
                rank -= block
            else:
                lengths[pos] = length
                remaining -= weight
                break
        else:
            raise ValueError("bad Huffman length-vector rank")
    if remaining or rank:
        raise ValueError("bad Huffman length-vector rank")
    return lengths


def decode_combination_colex(rank, n, k):
    """Decode a k-of-n combination from colexicographic rank."""
    if rank >= math.comb(n, k):
        raise ValueError("bad combination rank")
    combo = [0] * k
    x = n
    for i in range(k, 0, -1):
        x -= 1
        while math.comb(x, i) > rank:
            x -= 1
        combo[i - 1] = x
        rank -= math.comb(x, i)
    if rank:
        raise ValueError("bad combination rank")
    return np.array(combo, dtype=np.int64)


def _packed_dims(value, n_valid, *, error_message):
    """Recover little-endian base-LATENT_DIM sidecar dimensions."""
    dims_valid = np.empty(n_valid, dtype=np.int64)
    for i in range(n_valid):
        value, dims_valid[i] = divmod(value, LATENT_DIM)
    if value:
        raise ValueError(error_message)
    return dims_valid


def _vectors_from_valid(valid_mask, dims_valid, delta_valid):
    """Materialize full per-pair sidecar vectors from valid pair entries."""
    dims = np.full(N_PAIRS, 255, dtype=np.int64)
    codes = np.zeros(N_PAIRS, dtype=np.float32)
    dims[valid_mask] = dims_valid
    codes[valid_mask] = SIDECAR_DELTAS_X100[delta_valid].astype(np.float32)
    return dims, codes


def _vectors_from_choices(choices):
    """Decode legacy dense sidecar choices into full per-pair vectors."""
    valid = choices != 0
    idx = choices[valid] - 1
    dims = np.full(N_PAIRS, 255, dtype=np.int64)
    codes = np.zeros(N_PAIRS, dtype=np.float32)
    dims[valid] = idx // len(SIDECAR_DELTAS_X100)
    codes[valid] = SIDECAR_DELTAS_X100[idx % len(SIDECAR_DELTAS_X100)].astype(np.float32)
    return dims, codes


def _decode_enum_rank_sidecar(raw, arr_size):
    """Decode enum-ranked Huffman sidecar with inferred no-op positions."""
    dim_end = SIDECAR_DIM_PACKED_LEN
    rank_end = dim_end + SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN
    length_rank = int.from_bytes(raw[dim_end:rank_end], "little")
    lengths = decode_huff_length_rank(length_rank)

    noop_rank_start = arr_size - SIDECAR_NOOP_INFER_RANK_LEN
    delta_valid = decode_canonical_huffman_all(
        raw[rank_end:noop_rank_start], lengths
    ).astype(np.int64)
    n_valid = delta_valid.size
    noop_count = N_PAIRS - n_valid
    if noop_count < 0:
        raise ValueError("bad compact Huffman sidecar length")

    noop_rank = int.from_bytes(raw[noop_rank_start:], "little")
    noop_pos = decode_combination_colex(noop_rank, N_PAIRS, noop_count)
    valid_mask = np.ones(N_PAIRS, dtype=bool)
    valid_mask[noop_pos] = False
    if int(valid_mask.sum()) != n_valid:
        raise ValueError("bad compact Huffman sidecar no-op count")

    dims_valid = _packed_dims(
        int.from_bytes(raw[:dim_end], "little"),
        n_valid,
        error_message="bad compact Huffman sidecar dimensions",
    )
    return _vectors_from_valid(valid_mask, dims_valid, delta_valid)


def _decode_comb_rank_sidecar(raw):
    """Decode compact Huffman sidecar with combination-ranked no-op table."""
    noop_count = raw[0]
    noop_rank = int.from_bytes(raw[1:SIDECAR_NOOP_RANK_PREFIX_LEN], "little")
    noop_pos = decode_combination_colex(noop_rank, N_PAIRS, noop_count)
    valid_mask = np.ones(N_PAIRS, dtype=bool)
    valid_mask[noop_pos] = False
    n_valid = int(valid_mask.sum())

    dim_start = SIDECAR_NOOP_RANK_PREFIX_LEN
    dim_end = dim_start + SIDECAR_DIM_PACKED_LEN
    dims_valid = _packed_dims(
        int.from_bytes(raw[dim_start:dim_end], "little"),
        n_valid,
        error_message="bad compact Huffman sidecar dimensions",
    )

    len_start = dim_end
    len_end = len_start + SIDECAR_DELTA_HUFF3_LENGTHS_LEN
    lengths = unpack_3bit_lengths(
        raw[len_start:len_end], len(SIDECAR_DELTAS_X100), 2
    )
    delta_valid = decode_canonical_huffman(
        raw[len_end:], lengths, n_valid
    ).astype(np.int64)
    return _vectors_from_valid(valid_mask, dims_valid, delta_valid)


def _decode_split_sidecar(raw, arr_size):
    """Decode explicit no-op-table sidecars with Huffman or Brotli deltas."""
    noop_count = raw[0]
    noop_pos = np.frombuffer(
        raw[1:1 + 2 * noop_count], dtype="<u2"
    ).astype(np.int64)
    if noop_count * 2 + 1 != SIDECAR_NOOP_TABLE_LEN:
        raise ValueError("bad split sidecar no-op table")
    valid_mask = np.ones(N_PAIRS, dtype=bool)
    valid_mask[noop_pos] = False
    n_valid = int(valid_mask.sum())

    dim_start = SIDECAR_NOOP_TABLE_LEN
    dim_end = dim_start + SIDECAR_DIM_PACKED_LEN
    dims_valid = _packed_dims(
        int.from_bytes(raw[dim_start:dim_end], "little"),
        n_valid,
        error_message="bad split sidecar dimensions",
    )

    if arr_size == SIDECAR_HUFF_LEN:
        len_start = dim_end
        len_end = len_start + SIDECAR_DELTA_HUFF_LENGTHS_LEN
        lengths = unpack_nibbles(raw[len_start:len_end], len(SIDECAR_DELTAS_X100))
        delta_valid = decode_canonical_huffman(
            raw[len_end:], lengths, n_valid
        ).astype(np.int64)
    else:
        packed_delta = brotli.decompress(raw[dim_end:])
        delta_valid = unpack_nibbles(packed_delta, n_valid).astype(np.int64)
    return _vectors_from_valid(valid_mask, dims_valid, delta_valid)


def _decode_packed_choice_sidecar(raw):
    """Decode one big little-endian integer of base-SIDECAR_BASE choices."""
    value = int.from_bytes(raw, "little")
    choices = np.empty(N_PAIRS, dtype=np.int64)
    for i in range(N_PAIRS):
        value, choices[i] = divmod(value, SIDECAR_BASE)
    if value:
        raise ValueError("bad packed latent sidecar")
    return _vectors_from_choices(choices)


def _decode_latent_sidecar_vectors(raw):
    """Dispatch a raw latent sidecar payload to its format-specific decoder."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == SIDECAR_HUFF_ENUM_LEN:
        return _decode_enum_rank_sidecar(raw, arr.size)
    if arr.size == SIDECAR_HUFF_COMB_LEN:
        return _decode_comb_rank_sidecar(raw)
    if arr.size in (SIDECAR_HUFF_LEN, SIDECAR_SPLIT_LEN):
        return _decode_split_sidecar(raw, arr.size)
    if arr.size == SIDECAR_PACKED_LEN:
        return _decode_packed_choice_sidecar(raw)
    if arr.size == N_PAIRS:
        return _vectors_from_choices(arr.astype(np.int64))
    if arr.size == N_PAIRS * 2:
        pairs = arr.reshape(N_PAIRS, 2)
        return pairs[:, 0].astype(np.int64), pairs[:, 1].view(np.int8).astype(np.float32)
    raise ValueError("bad latent sidecar length")


def apply_latent_sidecar(latents, data):
    """Apply archive-local latent corrections without changing decoder math."""
    if not data:
        return latents
    raw = data
    if len(raw) not in (
        SIDECAR_HUFF_ENUM_LEN, SIDECAR_HUFF_COMB_LEN, SIDECAR_HUFF_LEN,
        SIDECAR_SPLIT_LEN, SIDECAR_PACKED_LEN, N_PAIRS, N_PAIRS * 2,
    ):
        raw = brotli.decompress(data)
    dims, codes = _decode_latent_sidecar_vectors(raw)
    valid = dims != 255
    if np.any(dims[valid] >= LATENT_DIM):
        raise ValueError("bad latent sidecar dimension")
    if valid.any():
        row = torch.from_numpy(np.nonzero(valid)[0])
        col = torch.from_numpy(dims[valid])
        delta = torch.from_numpy(codes[valid] / 100.0).to(latents.dtype)
        latents = latents.clone()
        latents[row, col] += delta
    return latents
