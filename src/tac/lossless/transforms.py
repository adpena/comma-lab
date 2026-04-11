from __future__ import annotations

import lzma
from collections import deque

import numpy as np


def recursive_bisect_frame_order(frame_count: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frame_count == 1:
        return np.array([0], dtype=np.int64)

    order: list[int] = [0, frame_count - 1]
    queue: deque[tuple[int, int]] = deque([(0, frame_count - 1)])
    while queue:
        start, stop = queue.popleft()
        if stop - start <= 1:
            continue
        midpoint = (start + stop) // 2
        order.append(midpoint)
        queue.append((start, midpoint))
        queue.append((midpoint, stop))
    return np.asarray(order, dtype=np.int64)


def apply_frame_order(tokens, order) -> np.ndarray:
    arr = np.asarray(tokens)
    permutation = np.asarray(order, dtype=np.int64).reshape(-1)
    if arr.ndim == 0:
        raise ValueError("tokens must have at least one dimension")
    if arr.shape[0] != permutation.size:
        raise ValueError("frame order length must match the frame dimension")
    if permutation.size and sorted(permutation.tolist()) != list(range(permutation.size)):
        raise ValueError("frame order must be a permutation of 0..frame_count-1")
    return arr[permutation].copy()


def invert_frame_order(tokens, order) -> np.ndarray:
    arr = np.asarray(tokens)
    permutation = np.asarray(order, dtype=np.int64).reshape(-1)
    if arr.ndim == 0:
        raise ValueError("tokens must have at least one dimension")
    if arr.shape[0] != permutation.size:
        raise ValueError("frame order length must match the frame dimension")
    if permutation.size and sorted(permutation.tolist()) != list(range(permutation.size)):
        raise ValueError("frame order must be a permutation of 0..frame_count-1")
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size, dtype=np.int64)
    return arr[inverse].copy()


def apply_frequency_remap(tokens) -> tuple[np.ndarray, dict[int, int]]:
    arr = np.asarray(tokens, dtype=np.uint16).reshape(-1)
    if arr.size == 0:
        return arr.copy(), {}

    unique, counts = np.unique(arr, return_counts=True)
    ranking = sorted(
        ((int(symbol), int(count)) for symbol, count in zip(unique.tolist(), counts.tolist())),
        key=lambda item: (-item[1], item[0]),
    )
    mapping = {symbol: index for index, (symbol, _count) in enumerate(ranking)}
    remapped = np.asarray([mapping[int(token)] for token in arr.tolist()], dtype=np.uint16)
    return remapped, mapping


def invert_frequency_remap(tokens, mapping: dict[int, int]) -> np.ndarray:
    arr = np.asarray(tokens, dtype=np.uint16).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    inverse = {index: symbol for symbol, index in mapping.items()}
    restored = np.asarray([inverse[int(token)] for token in arr.tolist()], dtype=np.uint16)
    return restored


def _zigzag_encode(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def _zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def _encode_temporal_delta(delta: int) -> int:
    encoded = _zigzag_encode(delta)
    if encoded >= 1025:
        encoded += 1
    return encoded


def _decode_temporal_delta(value: int) -> int:
    encoded = value - 1 if value >= 1026 else value
    return _zigzag_decode(encoded)


def temporal_residual_position_major(tokens) -> np.ndarray:
    arr = np.asarray(tokens, dtype=np.uint16).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    if int(arr[-1]) != 1025:
        raise ValueError("position-major stream must end with segment EOT")

    pieces: list[np.ndarray] = []
    start = 0
    while start < arr.size:
        try:
            end = int(np.where(arr[start:] == 1025)[0][0]) + start
        except IndexError as exc:
            raise ValueError("position-major stream is missing segment EOT") from exc
        stream = arr[start:end]
        positions = 128
        if stream.size % positions != 0:
            raise ValueError("position-major stream body must be divisible by 128")
        stream_len = stream.size // positions
        streams = stream.reshape(positions, stream_len)
        out = streams.copy()
        for index in range(positions):
            if int(streams[index, 0]) != 1024:
                raise ValueError("position-major stream must start each position with BOS")
            for offset in range(2, stream_len):
                current = int(streams[index, offset])
                previous = int(streams[index, offset - 1])
                out[index, offset] = _encode_temporal_delta(current - previous)
        pieces.append(out.reshape(-1))
        pieces.append(np.array([1025], dtype=np.uint16))
        start = end + 1
    return np.concatenate(pieces)


def invert_temporal_residual_position_major(tokens) -> np.ndarray:
    arr = np.asarray(tokens, dtype=np.uint16).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    if int(arr[-1]) != 1025:
        raise ValueError("position-major residual stream must end with segment EOT")

    pieces: list[np.ndarray] = []
    start = 0
    while start < arr.size:
        try:
            end = int(np.where(arr[start:] == 1025)[0][0]) + start
        except IndexError as exc:
            raise ValueError("position-major residual stream is missing segment EOT") from exc
        stream = arr[start:end]
        positions = 128
        if stream.size % positions != 0:
            raise ValueError("position-major residual stream body must be divisible by 128")
        stream_len = stream.size // positions
        streams = stream.reshape(positions, stream_len)
        out = streams.copy()
        for index in range(positions):
            if int(streams[index, 0]) != 1024:
                raise ValueError("position-major residual stream must start each position with BOS")
            for offset in range(2, stream_len):
                previous = int(out[index, offset - 1])
                delta = _decode_temporal_delta(int(streams[index, offset]))
                out[index, offset] = previous + delta
        pieces.append(out.reshape(-1))
        pieces.append(np.array([1025], dtype=np.uint16))
        start = end + 1
    return np.concatenate(pieces)


def split_token_bitplanes(tokens, *, low_bits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(tokens, dtype=np.uint16).reshape(-1)
    if low_bits <= 0 or low_bits >= 16:
        raise ValueError("low_bits must be between 1 and 15")
    low_mask = (1 << low_bits) - 1
    low = arr & low_mask
    high = arr >> low_bits
    return high.astype(np.uint16, copy=False), low.astype(np.uint16, copy=False)


def invert_split_token_bitplanes(high_bits, low_bits_values, *, low_bits: int = 5) -> np.ndarray:
    high = np.asarray(high_bits, dtype=np.uint16).reshape(-1)
    low = np.asarray(low_bits_values, dtype=np.uint16).reshape(-1)
    if high.shape != low.shape:
        raise ValueError("high_bits and low_bits_values must have the same shape")
    if low_bits <= 0 or low_bits >= 16:
        raise ValueError("low_bits must be between 1 and 15")
    low_mask = (1 << low_bits) - 1
    if np.any(low > low_mask):
        raise ValueError("low_bits_values exceed the configured bit width")
    restored = (high.astype(np.uint32) << low_bits) | low.astype(np.uint32)
    return restored.astype(np.uint16)


def sustain_attack_position_major(tokens) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(tokens, dtype=np.uint16).reshape(-1)
    if arr.size == 0 or int(arr[-1]) != 1025:
        raise ValueError("position-major stream must end with segment EOT")

    first_values: list[np.ndarray] = []
    hold_masks: list[np.ndarray] = []
    changed_values: list[np.ndarray] = []
    start = 0
    while start < arr.size:
        end = int(np.where(arr[start:] == 1025)[0][0]) + start
        stream = arr[start:end]
        if stream.size % 128 != 0:
            raise ValueError("position-major stream body must be divisible by 128")
        stream_len = stream.size // 128
        streams = stream.reshape(128, stream_len)
        if not np.all(streams[:, 0] == 1024):
            raise ValueError("position-major stream must start each position with BOS")
        first_values.append(streams[:, :2].reshape(-1))
        if stream_len > 2:
            values = streams[:, 2:]
            prev = streams[:, 1:-1]
            hold = (values == prev).astype(np.uint8)
            hold_t = hold.T
            values_t = values.T
            hold_masks.append(hold_t.reshape(-1))
            changed_values.append(values_t[hold_t == 0])
        hold_masks.append(np.array([2], dtype=np.uint8))
        start = end + 1
    return (
        np.concatenate(first_values) if first_values else np.array([], dtype=np.uint16),
        np.concatenate(hold_masks) if hold_masks else np.array([], dtype=np.uint8),
        np.concatenate(changed_values) if changed_values else np.array([], dtype=np.uint16),
    )


def invert_sustain_attack_position_major(first_values, hold_mask, changed_values) -> np.ndarray:
    first = np.asarray(first_values, dtype=np.uint16).reshape(-1)
    hold = np.asarray(hold_mask, dtype=np.uint8).reshape(-1)
    changed = np.asarray(changed_values, dtype=np.uint16).reshape(-1)
    if first.size % 256 != 0:
        raise ValueError("first_values must contain 2 header symbols per position")

    pieces: list[np.ndarray] = []
    segments = first.size // 256
    changed_offset = 0
    mask_segments: list[np.ndarray] = []
    current_mask: list[int] = []
    for value in hold.tolist():
        if value == 2:
            mask_segments.append(np.asarray(current_mask, dtype=np.uint8))
            current_mask = []
            continue
        current_mask.append(value)
    if current_mask:
        raise ValueError("hold_mask must terminate each segment with separator 2")
    if len(mask_segments) != segments:
        raise ValueError("hold_mask segment count does not match first_values")
    for segment_index in range(segments):
        head = first[segment_index * 256 : (segment_index + 1) * 256].reshape(128, 2)
        out = [head]
        mask_segment = mask_segments[segment_index]
        if mask_segment.size % 128 != 0:
            raise ValueError("hold mask body must be divisible by 128")
        future_steps = mask_segment.size // 128
        for step in range(future_steps):
            mask_row = mask_segment[step * 128 : (step + 1) * 128]
            prev = out[-1][:, -1]
            next_row = prev.copy()
            change_positions = np.where(mask_row == 0)[0]
            next_row[change_positions] = changed[changed_offset : changed_offset + len(change_positions)]
            changed_offset += len(change_positions)
            out.append(next_row.reshape(128, 1))
        segment = np.concatenate(out, axis=1).reshape(-1)
        pieces.append(segment)
        pieces.append(np.array([1025], dtype=np.uint16))
    if changed_offset != len(changed):
        raise ValueError("unused changed_values remain")
    return np.concatenate(pieces) if pieces else np.array([], dtype=np.uint16)


def sample_position_major_segments(tokens, *, max_segments: int) -> np.ndarray:
    arr = np.asarray(tokens, dtype=np.uint16).reshape(-1)
    if max_segments <= 0:
        raise ValueError("max_segments must be positive")
    if arr.size == 0 or int(arr[-1]) != 1025:
        raise ValueError("position-major stream must end with segment EOT")

    end_offsets = np.flatnonzero(arr == 1025)
    if end_offsets.size < max_segments:
        raise ValueError("position-major stream does not contain enough segments")
    end = int(end_offsets[max_segments - 1]) + 1
    return arr[:end].copy()


def _lzma_proxy_bytes(payload: bytes, *, preset: int) -> int:
    if not payload:
        return 0
    return len(lzma.compress(payload, preset=preset))


def _serialize_frequency_inverse_map(mapping: dict[int, int]) -> np.ndarray:
    if not mapping:
        return np.array([], dtype=np.uint16)
    inverse = np.empty(len(mapping), dtype=np.uint16)
    for symbol, index in mapping.items():
        inverse[int(index)] = int(symbol)
    return inverse


def _position_major_segment_count(tokens: np.ndarray) -> int:
    return int(np.count_nonzero(tokens == 1025))


def _pack_binary_mask(mask_values: np.ndarray) -> bytes:
    mask = np.asarray(mask_values, dtype=np.uint8).reshape(-1)
    if mask.size == 0:
        return b""
    return np.packbits(mask, bitorder="little").tobytes()


def _pack_bitplanes(values: np.ndarray, *, bit_width: int) -> list[bytes]:
    arr = np.asarray(values, dtype=np.uint16).reshape(-1)
    if bit_width <= 0:
        return []
    planes: list[bytes] = []
    for bit in range(bit_width):
        plane = ((arr >> bit) & 1).astype(np.uint8, copy=False)
        planes.append(_pack_binary_mask(plane))
    return planes


def benchmark_frequency_remap_sample(
    tokens,
    *,
    max_segments: int = 64,
    lzma_preset: int = 6,
) -> dict[str, object]:
    sample = sample_position_major_segments(tokens, max_segments=max_segments)
    remapped, mapping = apply_frequency_remap(sample)
    restored = invert_frequency_remap(remapped, mapping)
    mapping_header = _serialize_frequency_inverse_map(mapping)
    payload_bytes = _lzma_proxy_bytes(remapped.tobytes(), preset=lzma_preset)
    mapping_bytes = int(mapping_header.nbytes)
    encoded_bytes = payload_bytes + mapping_bytes
    return {
        "transform": "frequency_remap",
        "proxy_backend": "lzma",
        "lzma_preset": int(lzma_preset),
        "sample_segments": _position_major_segment_count(sample),
        "sample_tokens": int(sample.size),
        "sample_bytes": int(sample.nbytes),
        "payload_bytes": payload_bytes,
        "mapping_bytes": mapping_bytes,
        "encoded_bytes": encoded_bytes,
        "compression_ratio": (float(sample.nbytes) / float(encoded_bytes)) if encoded_bytes else float("inf"),
        "roundtrip_ok": bool(np.array_equal(restored, sample)),
    }


def benchmark_temporal_residual_sample(
    tokens,
    *,
    max_segments: int = 64,
    lzma_preset: int = 6,
) -> dict[str, object]:
    sample = sample_position_major_segments(tokens, max_segments=max_segments)
    residual = temporal_residual_position_major(sample)
    restored = invert_temporal_residual_position_major(residual)
    encoded_bytes = _lzma_proxy_bytes(residual.tobytes(), preset=lzma_preset)
    return {
        "transform": "temporal_residual",
        "proxy_backend": "lzma",
        "lzma_preset": int(lzma_preset),
        "sample_segments": _position_major_segment_count(sample),
        "sample_tokens": int(sample.size),
        "sample_bytes": int(sample.nbytes),
        "encoded_bytes": encoded_bytes,
        "compression_ratio": (float(sample.nbytes) / float(encoded_bytes)) if encoded_bytes else float("inf"),
        "roundtrip_ok": bool(np.array_equal(restored, sample)),
    }


def benchmark_bitplane_split_sample(
    tokens,
    *,
    max_segments: int = 64,
    low_bits: int = 5,
    lzma_preset: int = 6,
) -> dict[str, object]:
    sample = sample_position_major_segments(tokens, max_segments=max_segments)
    high, low = split_token_bitplanes(sample, low_bits=low_bits)
    restored = invert_split_token_bitplanes(high, low, low_bits=low_bits)
    high_bit_width = max(1, int(np.max(high, initial=0)).bit_length())
    high_plane_bytes = sum(
        _lzma_proxy_bytes(plane, preset=lzma_preset) for plane in _pack_bitplanes(high, bit_width=high_bit_width)
    )
    low_plane_bytes = sum(
        _lzma_proxy_bytes(plane, preset=lzma_preset) for plane in _pack_bitplanes(low, bit_width=low_bits)
    )
    header_bytes = 2
    encoded_bytes = high_plane_bytes + low_plane_bytes + header_bytes
    return {
        "transform": "bitplane_split",
        "proxy_backend": "lzma",
        "lzma_preset": int(lzma_preset),
        "sample_segments": _position_major_segment_count(sample),
        "sample_tokens": int(sample.size),
        "sample_bytes": int(sample.nbytes),
        "low_bits": int(low_bits),
        "high_bit_width": int(high_bit_width),
        "high_plane_bytes": int(high_plane_bytes),
        "low_plane_bytes": int(low_plane_bytes),
        "header_bytes": int(header_bytes),
        "encoded_bytes": int(encoded_bytes),
        "compression_ratio": (float(sample.nbytes) / float(encoded_bytes)) if encoded_bytes else float("inf"),
        "roundtrip_ok": bool(np.array_equal(restored, sample)),
    }


def benchmark_sustain_attack_sample(
    tokens,
    *,
    max_segments: int = 64,
    lzma_preset: int = 6,
) -> dict[str, object]:
    sample = sample_position_major_segments(tokens, max_segments=max_segments)
    first_values, hold_mask, changed_values = sustain_attack_position_major(sample)
    restored = invert_sustain_attack_position_major(first_values, hold_mask, changed_values)
    packed_mask = _pack_binary_mask(hold_mask[hold_mask != 2])
    first_bytes = _lzma_proxy_bytes(first_values.tobytes(), preset=lzma_preset)
    mask_bytes = _lzma_proxy_bytes(packed_mask, preset=lzma_preset)
    changed_bytes = _lzma_proxy_bytes(changed_values.tobytes(), preset=lzma_preset)
    header_bytes = 16
    encoded_bytes = first_bytes + mask_bytes + changed_bytes + header_bytes
    return {
        "transform": "sustain_attack",
        "proxy_backend": "lzma",
        "lzma_preset": int(lzma_preset),
        "sample_segments": _position_major_segment_count(sample),
        "sample_tokens": int(sample.size),
        "sample_bytes": int(sample.nbytes),
        "first_values_bytes": int(first_bytes),
        "packed_mask_bytes": int(mask_bytes),
        "changed_values_bytes": int(changed_bytes),
        "header_bytes": int(header_bytes),
        "encoded_bytes": int(encoded_bytes),
        "compression_ratio": (float(sample.nbytes) / float(encoded_bytes)) if encoded_bytes else float("inf"),
        "roundtrip_ok": bool(np.array_equal(restored, sample)),
    }
