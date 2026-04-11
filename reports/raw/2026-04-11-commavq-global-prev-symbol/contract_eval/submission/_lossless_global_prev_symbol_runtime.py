from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import heapq

PREV_SYMBOL_STREAM_MAGIC = b"TPC1"
STREAM_MAGIC = b"TFC1"
UINT16_MAX = 0xFFFF


def _decode_varint(data: bytes, offset: int, *, label: str):
    value = 0
    shift = 0
    cursor = offset
    while True:
        if cursor >= len(data):
            raise ValueError(f"truncated {label}")
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            raise ValueError(f"{label} exceeds supported bounds")


def _build_code_lengths(frequencies: dict[int, int]):
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        symbol = next(iter(frequencies))
        return {symbol: 0}
    nodes = {}
    heap = []
    next_node_id = 0
    for symbol, count in sorted(frequencies.items()):
        nodes[next_node_id] = (None, None, symbol)
        heapq.heappush(heap, (count, symbol, next_node_id))
        next_node_id += 1
    while len(heap) > 1:
        left_count, left_min_symbol, left_id = heapq.heappop(heap)
        right_count, right_min_symbol, right_id = heapq.heappop(heap)
        parent_id = next_node_id
        next_node_id += 1
        nodes[parent_id] = (left_id, right_id, None)
        heapq.heappush(heap, (left_count + right_count, min(left_min_symbol, right_min_symbol), parent_id))
    lengths = {}
    stack = [(heap[0][2], 0)]
    while stack:
        node_id, depth = stack.pop()
        left_id, right_id, symbol = nodes[node_id]
        if symbol is not None:
            lengths[symbol] = depth
            continue
        stack.append((right_id, depth + 1))
        stack.append((left_id, depth + 1))
    return lengths


def _build_canonical_codebook(lengths: dict[int, int]):
    positive_lengths = sorted((length, symbol) for symbol, length in lengths.items() if length > 0)
    if not positive_lengths:
        return {}, {}, {}, (), 0
    code = 0
    previous_length = 0
    first_code_by_length = {}
    first_index_by_length = {}
    count_by_length = {}
    ordered_symbols = []
    for index, (length, symbol) in enumerate(positive_lengths):
        code <<= length - previous_length
        if length not in first_code_by_length:
            first_code_by_length[length] = code
            first_index_by_length[length] = index
        count_by_length[length] = count_by_length.get(length, 0) + 1
        ordered_symbols.append(symbol)
        code += 1
        previous_length = length
    return first_code_by_length, first_index_by_length, count_by_length, tuple(ordered_symbols), max(count_by_length)


def _validate_padding(payload: bytes, *, bits_consumed: int):
    if bits_consumed == len(payload) * 8:
        return
    byte_index = bits_consumed // 8
    bit_offset = bits_consumed % 8
    if bit_offset:
        trailing_mask = (1 << (8 - bit_offset)) - 1
        if payload[byte_index] & trailing_mask:
            raise ValueError("non-zero trailing padding bits")
        byte_index += 1
    if byte_index != len(payload):
        raise ValueError("trailing payload bytes")


def decode_uint16_frequency_stream(encoded: bytes):
    if not encoded.startswith(STREAM_MAGIC):
        raise ValueError("invalid frequency stream header")
    cursor = len(STREAM_MAGIC)
    token_count, cursor = _decode_varint(encoded, cursor, label="token count")
    unique_symbols, cursor = _decode_varint(encoded, cursor, label="unique symbol count")
    payload_size, cursor = _decode_varint(encoded, cursor, label="payload size")
    frequencies = {}
    total = 0
    previous_symbol = 0
    for index in range(unique_symbols):
        delta, cursor = _decode_varint(encoded, cursor, label="frequency header")
        count, cursor = _decode_varint(encoded, cursor, label="frequency header")
        symbol = delta if index == 0 else previous_symbol + delta
        frequencies[symbol] = count
        total += count
        previous_symbol = symbol
    if total != token_count:
        raise ValueError("frequency table does not sum to token count")
    payload = encoded[cursor : cursor + payload_size]
    if len(frequencies) == 1:
        return [next(iter(frequencies))] * token_count
    lengths = _build_code_lengths(frequencies)
    first_code_by_length, first_index_by_length, count_by_length, ordered_symbols, max_code_bits = _build_canonical_codebook(lengths)
    restored = [0] * token_count
    produced = 0
    code = 0
    width = 0
    bits_consumed = 0
    for byte in payload:
        for shift in range(7, -1, -1):
            code = (code << 1) | ((byte >> shift) & 1)
            width += 1
            bits_consumed += 1
            first_code = first_code_by_length.get(width)
            if first_code is None:
                if width > max_code_bits:
                    raise ValueError("payload contains an invalid prefix code")
                continue
            code_offset = code - first_code
            count = count_by_length[width]
            if 0 <= code_offset < count:
                restored[produced] = ordered_symbols[first_index_by_length[width] + code_offset]
                produced += 1
                code = 0
                width = 0
                if produced == token_count:
                    _validate_padding(payload, bits_consumed=bits_consumed)
                    return restored
    raise ValueError("truncated payload")


def decode_uint16_prev_symbol_stream(encoded: bytes):
    if not encoded.startswith(PREV_SYMBOL_STREAM_MAGIC):
        raise ValueError("invalid prev-symbol stream header")
    cursor = len(PREV_SYMBOL_STREAM_MAGIC)
    token_count, cursor = _decode_varint(encoded, cursor, label="token count")
    if token_count == 0:
        context_count, cursor = _decode_varint(encoded, cursor, label="context count")
        if context_count != 0 or cursor != len(encoded):
            raise ValueError("invalid empty prev-symbol stream")
        return []
    first_symbol, cursor = _decode_varint(encoded, cursor, label="first symbol")
    context_count, cursor = _decode_varint(encoded, cursor, label="context count")
    context_sizes = {}
    previous_symbol = 0
    for index in range(context_count):
        delta, cursor = _decode_varint(encoded, cursor, label="context header")
        payload_size, cursor = _decode_varint(encoded, cursor, label="context header")
        symbol = delta if index == 0 else previous_symbol + delta
        context_sizes[symbol] = payload_size
        previous_symbol = symbol
    payload_cursor = cursor
    decoded_contexts = {}
    for symbol, payload_size in context_sizes.items():
        end = payload_cursor + payload_size
        decoded_contexts[symbol] = decode_uint16_frequency_stream(encoded[payload_cursor:end])
        payload_cursor = end
    restored = [first_symbol]
    context_offsets = {symbol: 0 for symbol in decoded_contexts}
    for _ in range(1, token_count):
        previous = restored[-1]
        values = decoded_contexts[previous]
        offset = context_offsets[previous]
        restored.append(values[offset])
        context_offsets[previous] = offset + 1
    return restored


def decode_corpus_global_prev_symbol_position_major(*, encoded_dir: str | Path, output_dir: str | Path):
    root = Path(encoded_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    decoded_chunks = {}
    chunk_offsets = {}

    for record in manifest["records"]:
        chunk_index = int(record["chunk_index"])
        if chunk_index not in decoded_chunks:
            decoded_chunks[chunk_index] = np.asarray(
                decode_uint16_prev_symbol_stream((root / f"chunk_{chunk_index:03d}.tpc").read_bytes()),
                dtype=np.uint16,
            )
            chunk_offsets[chunk_index] = 0
        decoded = decoded_chunks[chunk_index]
        offset = chunk_offsets[chunk_index]
        token_count = int(record["token_count"])
        payload = decoded[offset : offset + token_count]
        chunk_offsets[chunk_index] = offset + token_count
        body = payload[:-1].reshape(128, -1)
        frames = body[:, 1:].T.reshape(-1, 8, 16).astype(np.int16)
        with (target / str(record["file_name"])).open("wb") as handle:
            np.save(handle, frames)
