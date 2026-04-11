from __future__ import annotations

import heapq
from dataclasses import dataclass

STREAM_MAGIC = b"TFC1"
PREV_SYMBOL_STREAM_MAGIC = b"TPC1"
UINT16_MAX = 0xFFFF


@dataclass(frozen=True)
class FrequencyEncodedStream:
    encoded_bytes: bytes
    token_count: int
    unique_symbols: int
    header_bytes: int
    payload_bytes: int
    max_code_bits: int

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if self.unique_symbols < 0:
            raise ValueError("unique_symbols must be non-negative")
        if self.header_bytes <= 0:
            raise ValueError("header_bytes must be positive")
        if self.payload_bytes < 0:
            raise ValueError("payload_bytes must be non-negative")
        if self.header_bytes + self.payload_bytes != len(self.encoded_bytes):
            raise ValueError("header_bytes + payload_bytes must match encoded length")
        if self.max_code_bits < 0:
            raise ValueError("max_code_bits must be non-negative")


@dataclass(frozen=True)
class PrevSymbolEncodedStream:
    encoded_bytes: bytes
    token_count: int
    context_count: int
    header_bytes: int
    payload_bytes: int

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if self.context_count < 0:
            raise ValueError("context_count must be non-negative")
        if self.header_bytes <= 0:
            raise ValueError("header_bytes must be positive")
        if self.payload_bytes < 0:
            raise ValueError("payload_bytes must be non-negative")
        if self.header_bytes + self.payload_bytes != len(self.encoded_bytes):
            raise ValueError("header_bytes + payload_bytes must match encoded length")


@dataclass(frozen=True)
class _ParsedFrequencyStream:
    token_count: int
    frequencies: dict[int, int]
    payload: bytes
    header_bytes: int


@dataclass(frozen=True)
class _CanonicalCodebook:
    encode_table: dict[int, tuple[int, int]]
    first_code_by_length: dict[int, int]
    first_index_by_length: dict[int, int]
    count_by_length: dict[int, int]
    ordered_symbols: tuple[int, ...]
    max_code_bits: int


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for uint16 frequency coding") from exc
    return np


def _normalize_uint16_tokens(tokens):
    np = _require_numpy()

    if isinstance(tokens, np.ndarray):
        array = tokens.reshape(-1)
    else:
        try:
            array = np.asarray(list(tokens))
        except TypeError:
            array = np.asarray([tokens])
        array = array.reshape(-1)
    if array.size == 0:
        return array.astype(np.uint16)
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError("tokens must be uint16-compatible integers")
    signed = array.astype(np.int64, copy=False)
    if np.any(signed < 0) or np.any(signed > UINT16_MAX):
        raise ValueError("tokens must be in uint16 range 0..65535")
    return signed.astype(np.uint16, copy=False)


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint values must be non-negative")
    out = bytearray()
    remaining = value
    while True:
        byte = remaining & 0x7F
        remaining >>= 7
        if remaining:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _decode_varint(data: bytes, offset: int, *, label: str) -> tuple[int, int]:
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


def _build_frequencies(tokens) -> dict[int, int]:
    np = _require_numpy()

    if tokens.size == 0:
        return {}
    unique, counts = np.unique(tokens, return_counts=True)
    return {int(symbol): int(count) for symbol, count in zip(unique.tolist(), counts.tolist())}


def _build_code_lengths(frequencies: dict[int, int]) -> dict[int, int]:
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        symbol = next(iter(frequencies))
        return {symbol: 0}

    nodes: dict[int, tuple[int | None, int | None, int | None]] = {}
    heap: list[tuple[int, int, int]] = []
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
        heapq.heappush(
            heap,
            (left_count + right_count, min(left_min_symbol, right_min_symbol), parent_id),
        )

    lengths: dict[int, int] = {}
    stack = [(heap[0][2], 0)]
    while stack:
        node_id, depth = stack.pop()
        left_id, right_id, symbol = nodes[node_id]
        if symbol is not None:
            lengths[symbol] = depth
            continue
        assert left_id is not None and right_id is not None
        stack.append((right_id, depth + 1))
        stack.append((left_id, depth + 1))
    return lengths


def _build_canonical_codebook(lengths: dict[int, int]) -> _CanonicalCodebook:
    encode_table: dict[int, tuple[int, int]] = {}
    positive_lengths = sorted((length, symbol) for symbol, length in lengths.items() if length > 0)
    if not positive_lengths:
        return _CanonicalCodebook(
            encode_table=encode_table,
            first_code_by_length={},
            first_index_by_length={},
            count_by_length={},
            ordered_symbols=tuple(),
            max_code_bits=0,
        )

    code = 0
    previous_length = 0
    ordered_symbols: list[int] = []
    first_code_by_length: dict[int, int] = {}
    first_index_by_length: dict[int, int] = {}
    count_by_length: dict[int, int] = {}

    for index, (length, symbol) in enumerate(positive_lengths):
        code <<= length - previous_length
        if length not in first_code_by_length:
            first_code_by_length[length] = code
            first_index_by_length[length] = index
        count_by_length[length] = count_by_length.get(length, 0) + 1
        encode_table[symbol] = (code, length)
        ordered_symbols.append(symbol)
        code += 1
        previous_length = length

    return _CanonicalCodebook(
        encode_table=encode_table,
        first_code_by_length=first_code_by_length,
        first_index_by_length=first_index_by_length,
        count_by_length=count_by_length,
        ordered_symbols=tuple(ordered_symbols),
        max_code_bits=max(count_by_length),
    )


def _serialize_header(*, token_count: int, frequencies: dict[int, int], payload_size: int) -> bytes:
    header = bytearray(STREAM_MAGIC)
    header.extend(_encode_varint(token_count))
    header.extend(_encode_varint(len(frequencies)))
    header.extend(_encode_varint(payload_size))

    previous_symbol = 0
    for index, (symbol, count) in enumerate(sorted(frequencies.items())):
        delta = symbol if index == 0 else symbol - previous_symbol
        header.extend(_encode_varint(delta))
        header.extend(_encode_varint(count))
        previous_symbol = symbol
    return bytes(header)


def _parse_header(encoded: bytes | bytearray | memoryview) -> _ParsedFrequencyStream:
    data = bytes(encoded)
    if not data.startswith(STREAM_MAGIC):
        raise ValueError("invalid frequency stream header")

    cursor = len(STREAM_MAGIC)
    token_count, cursor = _decode_varint(data, cursor, label="token count")
    unique_symbols, cursor = _decode_varint(data, cursor, label="unique symbol count")
    payload_size, cursor = _decode_varint(data, cursor, label="payload size")

    if unique_symbols > UINT16_MAX + 1:
        raise ValueError("unique symbol count exceeds uint16 alphabet")
    if token_count == 0 and unique_symbols != 0:
        raise ValueError("empty stream must not declare symbols")
    if token_count == 0 and payload_size != 0:
        raise ValueError("empty stream must not declare payload bytes")
    if token_count > 0 and unique_symbols == 0:
        raise ValueError("non-empty stream must declare symbols")

    frequencies: dict[int, int] = {}
    total = 0
    previous_symbol = 0

    for index in range(unique_symbols):
        delta, cursor = _decode_varint(data, cursor, label="frequency header")
        count, cursor = _decode_varint(data, cursor, label="frequency header")
        symbol = delta if index == 0 else previous_symbol + delta
        if symbol > UINT16_MAX:
            raise ValueError("frequency header symbol exceeds uint16 range")
        if index > 0 and symbol <= previous_symbol:
            raise ValueError("frequency header symbols must be strictly increasing")
        if count <= 0:
            raise ValueError("symbol frequencies must be positive")
        frequencies[symbol] = count
        total += count
        previous_symbol = symbol

    if total != token_count:
        raise ValueError("frequency table does not sum to token count")

    payload_end = cursor + payload_size
    if payload_end > len(data):
        raise ValueError("truncated payload")
    if payload_end < len(data):
        raise ValueError("trailing payload bytes")

    return _ParsedFrequencyStream(
        token_count=token_count,
        frequencies=frequencies,
        payload=data[cursor:payload_end],
        header_bytes=cursor,
    )


def _encode_payload(tokens, *, encode_table: dict[int, tuple[int, int]]) -> bytes:
    if tokens.size == 0 or not encode_table:
        return b""

    out = bytearray()
    bit_buffer = 0
    bit_count = 0

    for symbol in tokens.tolist():
        code, width = encode_table[int(symbol)]
        bit_buffer = (bit_buffer << width) | code
        bit_count += width
        while bit_count >= 8:
            bit_count -= 8
            out.append((bit_buffer >> bit_count) & 0xFF)
            if bit_count:
                bit_buffer &= (1 << bit_count) - 1
            else:
                bit_buffer = 0

    if bit_count:
        out.append((bit_buffer << (8 - bit_count)) & 0xFF)
    return bytes(out)


def _validate_padding(payload: bytes, *, bits_consumed: int) -> None:
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


def _decode_payload(
    payload: bytes,
    *,
    token_count: int,
    canonical: _CanonicalCodebook,
):
    np = _require_numpy()

    restored = np.empty(token_count, dtype=np.uint16)
    produced = 0
    code = 0
    width = 0
    bits_consumed = 0

    for byte in payload:
        for shift in range(7, -1, -1):
            code = (code << 1) | ((byte >> shift) & 1)
            width += 1
            bits_consumed += 1

            first_code = canonical.first_code_by_length.get(width)
            if first_code is None:
                if width > canonical.max_code_bits:
                    raise ValueError("payload contains an invalid prefix code")
                continue
            code_offset = code - first_code
            count = canonical.count_by_length[width]
            if 0 <= code_offset < count:
                symbol_index = canonical.first_index_by_length[width] + code_offset
                restored[produced] = canonical.ordered_symbols[symbol_index]
                produced += 1
                code = 0
                width = 0
                if produced == token_count:
                    _validate_padding(payload, bits_consumed=bits_consumed)
                    return restored

    raise ValueError("truncated payload")


def encode_uint16_frequency_stream(tokens) -> FrequencyEncodedStream:
    normalized = _normalize_uint16_tokens(tokens)
    frequencies = _build_frequencies(normalized)
    lengths = _build_code_lengths(frequencies)
    canonical = _build_canonical_codebook(lengths)
    payload = _encode_payload(normalized, encode_table=canonical.encode_table)
    header = _serialize_header(
        token_count=int(normalized.size),
        frequencies=frequencies,
        payload_size=len(payload),
    )
    encoded = header + payload
    return FrequencyEncodedStream(
        encoded_bytes=encoded,
        token_count=int(normalized.size),
        unique_symbols=len(frequencies),
        header_bytes=len(header),
        payload_bytes=len(payload),
        max_code_bits=canonical.max_code_bits,
    )


def decode_uint16_frequency_stream(encoded: bytes | bytearray | memoryview):
    np = _require_numpy()

    parsed = _parse_header(encoded)
    if parsed.token_count == 0:
        return np.array([], dtype=np.uint16)

    lengths = _build_code_lengths(parsed.frequencies)
    if len(parsed.frequencies) == 1:
        only_symbol = next(iter(parsed.frequencies))
        if parsed.payload:
            raise ValueError("trailing payload bytes")
        return np.full(parsed.token_count, only_symbol, dtype=np.uint16)

    canonical = _build_canonical_codebook(lengths)
    if not canonical.encode_table:
        raise ValueError("frequency table cannot produce a codebook")
    return _decode_payload(
        parsed.payload,
        token_count=parsed.token_count,
        canonical=canonical,
    )


def encode_uint16_frequency_file(source_path: str | Path, encoded_path: str | Path) -> dict[str, object]:
    from pathlib import Path

    np = _require_numpy()
    source = Path(source_path)
    target = Path(encoded_path)
    if source.stat().st_size % 2 != 0:
        raise ValueError(f"token stream must contain an even number of bytes: {source}")
    tokens = np.fromfile(source, dtype=np.uint16)
    encoded = encode_uint16_frequency_stream(tokens)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded.encoded_bytes)
    return {
        "command": "lossless_frequency_encode",
        "source_path": str(source),
        "encoded_path": str(target),
        "token_count": encoded.token_count,
        "unique_symbols": encoded.unique_symbols,
        "header_bytes": encoded.header_bytes,
        "payload_bytes": encoded.payload_bytes,
        "restored_dtype": "uint16",
    }


def decode_uint16_frequency_file(encoded_path: str | Path, restored_path: str | Path) -> str:
    from pathlib import Path

    source = Path(encoded_path)
    target = Path(restored_path)
    tokens = decode_uint16_frequency_stream(source.read_bytes())
    target.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(target)
    return str(target)


def benchmark_uint16_frequency_file(source_path: str | Path, encoded_path: str | Path) -> dict[str, object]:
    from pathlib import Path

    source = Path(source_path)
    if source.stat().st_size % 2 != 0:
        raise ValueError(f"token stream must contain an even number of bytes: {source}")
    encoded = encode_uint16_frequency_file(source, encoded_path)
    original_bytes = source.stat().st_size
    encoded_bytes = Path(encoded["encoded_path"]).stat().st_size
    return {
        "command": "lossless_frequency_benchmark",
        "source_path": str(source),
        "encoded_path": encoded["encoded_path"],
        "token_count": encoded["token_count"],
        "unique_symbols": encoded["unique_symbols"],
        "original_bytes": original_bytes,
        "encoded_bytes": encoded_bytes,
        "compression_ratio": original_bytes / encoded_bytes if encoded_bytes else 0.0,
    }


def benchmark_prev_symbol_frequency_stream(tokens) -> dict[str, object]:
    normalized = _normalize_uint16_tokens(tokens)
    if normalized.size == 0:
        return {
            "command": "lossless_prev_symbol_frequency_benchmark",
            "token_count": 0,
            "context_count": 0,
            "encoded_bytes": 0,
            "compression_ratio": 0.0,
        }

    contexts: dict[int | None, list[int]] = {None: [int(normalized[0])]}
    for previous, current in zip(normalized[:-1], normalized[1:]):
        contexts.setdefault(int(previous), []).append(int(current))

    encoded_bytes = 0
    for stream in contexts.values():
        encoded = encode_uint16_frequency_stream(stream)
        encoded_bytes += len(encoded.encoded_bytes)

    original_bytes = int(normalized.size) * 2
    return {
        "command": "lossless_prev_symbol_frequency_benchmark",
        "token_count": int(normalized.size),
        "context_count": len(contexts),
        "encoded_bytes": encoded_bytes,
        "compression_ratio": original_bytes / encoded_bytes if encoded_bytes else 0.0,
    }


def benchmark_prev_pair_frequency_stream(tokens) -> dict[str, object]:
    normalized = _normalize_uint16_tokens(tokens)
    if normalized.size == 0:
        return {
            "command": "lossless_prev_pair_frequency_benchmark",
            "token_count": 0,
            "context_count": 0,
            "encoded_bytes": 0,
            "compression_ratio": 0.0,
        }

    contexts: dict[tuple[int | None, int], list[int]] = {}
    if normalized.size >= 2:
        contexts[(None, int(normalized[0]))] = [int(normalized[1])]
        for prev_prev, prev, current in zip(normalized[:-2], normalized[1:-1], normalized[2:]):
            contexts.setdefault((int(prev_prev), int(prev)), []).append(int(current))

    encoded_bytes = 0
    for stream in contexts.values():
        encoded = encode_uint16_frequency_stream(stream)
        encoded_bytes += len(encoded.encoded_bytes)

    original_bytes = int(normalized.size) * 2
    return {
        "command": "lossless_prev_pair_frequency_benchmark",
        "token_count": int(normalized.size),
        "context_count": len(contexts),
        "encoded_bytes": encoded_bytes,
        "compression_ratio": original_bytes / encoded_bytes if encoded_bytes else 0.0,
    }


def encode_uint16_prev_symbol_stream(tokens) -> PrevSymbolEncodedStream:
    normalized = _normalize_uint16_tokens(tokens)
    if normalized.size == 0:
        header = bytearray(PREV_SYMBOL_STREAM_MAGIC)
        header.extend(_encode_varint(0))
        header.extend(_encode_varint(0))
        return PrevSymbolEncodedStream(
            encoded_bytes=bytes(header),
            token_count=0,
            context_count=0,
            header_bytes=len(header),
            payload_bytes=0,
        )

    first_symbol = int(normalized[0])
    contexts: dict[int, list[int]] = {}
    for previous, current in zip(normalized[:-1], normalized[1:]):
        contexts.setdefault(int(previous), []).append(int(current))

    payload = bytearray()
    header = bytearray(PREV_SYMBOL_STREAM_MAGIC)
    header.extend(_encode_varint(int(normalized.size)))
    header.extend(_encode_varint(first_symbol))
    header.extend(_encode_varint(len(contexts)))
    previous_symbol = 0
    for index, symbol in enumerate(sorted(contexts)):
        delta = symbol if index == 0 else symbol - previous_symbol
        encoded = encode_uint16_frequency_stream(contexts[symbol])
        header.extend(_encode_varint(delta))
        header.extend(_encode_varint(len(encoded.encoded_bytes)))
        payload.extend(encoded.encoded_bytes)
        previous_symbol = symbol

    encoded_bytes = bytes(header) + bytes(payload)
    return PrevSymbolEncodedStream(
        encoded_bytes=encoded_bytes,
        token_count=int(normalized.size),
        context_count=len(contexts),
        header_bytes=len(header),
        payload_bytes=len(payload),
    )


def decode_uint16_prev_symbol_stream(encoded: bytes | bytearray | memoryview):
    np = _require_numpy()
    data = bytes(encoded)
    if not data.startswith(PREV_SYMBOL_STREAM_MAGIC):
        raise ValueError("invalid prev-symbol stream header")

    cursor = len(PREV_SYMBOL_STREAM_MAGIC)
    token_count, cursor = _decode_varint(data, cursor, label="token count")
    if token_count == 0:
        context_count, cursor = _decode_varint(data, cursor, label="context count")
        if context_count != 0:
            raise ValueError("empty prev-symbol stream must not declare contexts")
        if cursor != len(data):
            raise ValueError("trailing payload bytes")
        return np.array([], dtype=np.uint16)

    first_symbol, cursor = _decode_varint(data, cursor, label="first symbol")
    context_count, cursor = _decode_varint(data, cursor, label="context count")
    if first_symbol > UINT16_MAX:
        raise ValueError("first symbol exceeds uint16 range")
    context_sizes: dict[int, int] = {}
    previous_symbol = 0
    for index in range(context_count):
        delta, cursor = _decode_varint(data, cursor, label="context header")
        payload_size, cursor = _decode_varint(data, cursor, label="context header")
        symbol = delta if index == 0 else previous_symbol + delta
        if symbol > UINT16_MAX:
            raise ValueError("context symbol exceeds uint16 range")
        if index > 0 and symbol <= previous_symbol:
            raise ValueError("context symbols must be strictly increasing")
        context_sizes[symbol] = payload_size
        previous_symbol = symbol

    payload_cursor = cursor
    contexts: dict[int, tuple[int, int]] = {}
    for symbol, payload_size in context_sizes.items():
        end = payload_cursor + payload_size
        if end > len(data):
            raise ValueError("truncated payload")
        contexts[symbol] = (payload_cursor, end)
        payload_cursor = end

    if payload_cursor != len(data):
        raise ValueError("trailing payload bytes")

    decoded_contexts: dict[int, list[int]] = {}
    for symbol, (start, end) in contexts.items():
        decoded = decode_uint16_frequency_stream(data[start:end])
        decoded_contexts[symbol] = decoded.astype(np.uint16).tolist()

    restored = np.empty(token_count, dtype=np.uint16)
    restored[0] = first_symbol
    context_offsets = dict.fromkeys(decoded_contexts, 0)
    for index in range(1, token_count):
        previous = int(restored[index - 1])
        if previous not in decoded_contexts:
            raise ValueError("missing context payload")
        values = decoded_contexts[previous]
        offset = context_offsets[previous]
        if offset >= len(values):
            raise ValueError("context payload exhausted early")
        restored[index] = values[offset]
        context_offsets[previous] = offset + 1

    for symbol, values in decoded_contexts.items():
        if context_offsets[symbol] != len(values):
            raise ValueError("trailing payload bytes")
    return restored


def encode_uint16_prev_symbol_file(source_path: str | Path, encoded_path: str | Path) -> dict[str, object]:
    from pathlib import Path

    np = _require_numpy()
    source = Path(source_path)
    target = Path(encoded_path)
    if source.stat().st_size % 2 != 0:
        raise ValueError(f"token stream must contain an even number of bytes: {source}")
    tokens = np.fromfile(source, dtype=np.uint16)
    encoded = encode_uint16_prev_symbol_stream(tokens)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded.encoded_bytes)
    return {
        "command": "lossless_prev_symbol_encode",
        "source_path": str(source),
        "encoded_path": str(target),
        "token_count": encoded.token_count,
        "context_count": encoded.context_count,
        "header_bytes": encoded.header_bytes,
        "payload_bytes": encoded.payload_bytes,
        "restored_dtype": "uint16",
    }


def decode_uint16_prev_symbol_file(encoded_path: str | Path, restored_path: str | Path) -> str:
    from pathlib import Path

    source = Path(encoded_path)
    target = Path(restored_path)
    tokens = decode_uint16_prev_symbol_stream(source.read_bytes())
    target.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(target)
    return str(target)


def benchmark_prev_symbol_frequency_file(
    source_path: str | Path, *, max_tokens: int | None = None
) -> dict[str, object]:
    from pathlib import Path

    np = _require_numpy()
    source = Path(source_path)
    if source.stat().st_size % 2 != 0:
        raise ValueError(f"token stream must contain an even number of bytes: {source}")
    tokens = np.fromfile(source, dtype=np.uint16)
    if max_tokens is not None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        tokens = tokens[:max_tokens]
    payload = benchmark_prev_symbol_frequency_stream(tokens)
    payload["source_path"] = str(source)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    return payload


def benchmark_prev_pair_frequency_file(source_path: str | Path, *, max_tokens: int | None = None) -> dict[str, object]:
    from pathlib import Path

    np = _require_numpy()
    source = Path(source_path)
    if source.stat().st_size % 2 != 0:
        raise ValueError(f"token stream must contain an even number of bytes: {source}")
    tokens = np.fromfile(source, dtype=np.uint16)
    if max_tokens is not None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        tokens = tokens[:max_tokens]
    payload = benchmark_prev_pair_frequency_stream(tokens)
    payload["source_path"] = str(source)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    return payload


__all__ = [
    "STREAM_MAGIC",
    "FrequencyEncodedStream",
    "benchmark_prev_pair_frequency_file",
    "benchmark_prev_pair_frequency_stream",
    "benchmark_prev_symbol_frequency_file",
    "benchmark_prev_symbol_frequency_stream",
    "benchmark_uint16_frequency_file",
    "decode_uint16_frequency_file",
    "decode_uint16_frequency_stream",
    "decode_uint16_prev_symbol_file",
    "decode_uint16_prev_symbol_stream",
    "encode_uint16_frequency_file",
    "encode_uint16_frequency_stream",
    "encode_uint16_prev_symbol_file",
    "encode_uint16_prev_symbol_stream",
]
