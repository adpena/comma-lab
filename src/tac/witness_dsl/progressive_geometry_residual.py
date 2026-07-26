# SPDX-License-Identifier: MIT
"""Exact progressive geometry residual for an identity-bound semantic predictor.

PBR2 replaces PBR1's one-record-per-changed-cell payload with three counted,
independently applicable strata inside one complete packet:

1. correction states that persist from the preceding pair, represented either
   as target-class row runs or as MPEG-style constant/boundary blocks;
2. exact 8-connected, same-target-class island atoms represented by row spans;
3. singleton sparse-tail events.

The encoder may inspect target labels.  The decoder receives only the counted
packet and the exact predictor it is bound to: no scorer, source video, or
target cache is consulted at decode time.  The packet is nevertheless a
lossless, target-derived encoding of the caller-supplied semantic table.  It is
therefore an encoder-side conditional-entropy measurement only and MUST NOT be
placed in a candidate archive.  Per-stratum digests permit staged application
from a complete packet; they do not make truncated byte prefixes standalone.
This module deliberately makes both restrictions machine-readable.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import lzma
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

PACKET_SCHEMA: Final = "tac.progressive_geometry_residual.v3"
PACKET_MAGIC: Final = b"PBR2"
PACKET_VERSION: Final = 3
SEMANTIC_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
STRATUM_NAMES: Final = ("temporal_boundary", "component_islands", "sparse_tail")

_PREFIX: Final = struct.Struct("<4sIIIII")
_CRC: Final = struct.Struct("<I")
_HEADER_FIELDS: Final = frozenset(
    {
        "schema",
        "version",
        "predictor_contract_id",
        "predictor_renderer_sha256",
        "predictor_program_bytes",
        "predictor_program_sha256",
        "predictor_semantic_bytes",
        "predictor_semantic_sha256",
        "target_semantic_bytes",
        "target_semantic_sha256",
        "n_pairs",
        "height",
        "width",
        "source_pair_start",
        "source_pair_stop_exclusive",
        "source_pair_ids_sha256",
        "semantic_names",
        "semantic_class_ids",
        "strata",
        "separate_dense_target_table_section_bytes",
        "pbr2_is_target_derived",
        "pbr2_target_derived_section_bytes",
        "pbr2_event_count",
        "pbr2_event_density_numerator",
        "pbr2_event_density_denominator",
        "target_derived_residual_promotion_admitted",
        "research_only",
        "artifact_role",
        "candidate_archive_admissible",
        "exact_target_semantic_reconstruction",
        "target_semantic_lineage",
        "pbr2_reconstructs_exact_gt_argmax",
        "reconstructed_target_semantic_bytes",
        "candidate_archive_blocker",
        "generic_apply_requires_external_predictor_semantics",
        "physical_prefix_decode_supported",
        "staged_application_requires_complete_packet",
        "decode_scorer_dependency",
        "score_claim",
        "promotion_eligible",
    }
)
_STRATUM_FIELDS: Final = frozenset(
    {
        "name",
        "order",
        "mode",
        "block_size",
        "codec",
        "raw_bytes",
        "raw_sha256",
        "payload_bytes",
        "payload_sha256",
        "record_count",
        "span_count",
        "corrected_cells",
        "errors_before",
        "errors_after",
        "semantic_sha256_after",
    }
)

_CODEC_ORDER: Final = ("raw", "zlib9", "bz2_9", "lzma6")


class ProgressiveGeometryResidualError(ValueError):
    """Fail-closed packet, identity, coordinate, or geometry error."""


def _sha256(value: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(value)
    return digest.hexdigest()


def _semantic_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value, dtype=np.uint8)).cast("B"))


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ProgressiveGeometryResidualError("header must be finite canonical ASCII JSON") from exc


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ProgressiveGeometryResidualError(f"{label} must be a lowercase SHA-256 digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ProgressiveGeometryResidualError(f"{label} must be a lowercase SHA-256 digest") from exc
    if len(decoded) != 32:
        raise ProgressiveGeometryResidualError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_contract(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProgressiveGeometryResidualError("predictor_contract_id must be non-empty")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProgressiveGeometryResidualError("predictor_contract_id must be ASCII") from exc
    if len(encoded) > 256:
        raise ProgressiveGeometryResidualError("predictor_contract_id is too long")
    return value


def _semantic_array(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or any(int(size) <= 0 for size in array.shape):
        raise ProgressiveGeometryResidualError(f"{label} must have pair x height x width geometry")
    if array.dtype.kind not in ("i", "u"):
        raise ProgressiveGeometryResidualError(f"{label} must contain integer class ids")
    if array.size and (int(array.min()) < 0 or int(array.max()) >= len(SEMANTIC_NAMES)):
        raise ProgressiveGeometryResidualError(f"{label} class ids must be in [0,4]")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _pair_ids(value: Sequence[int], *, n_pairs: int) -> tuple[int, ...]:
    ids = tuple(value)
    if len(ids) != n_pairs or any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in ids):
        raise ProgressiveGeometryResidualError("source_pair_ids must contain one integer per local pair")
    normalized = tuple(int(item) for item in ids)
    if normalized != tuple(range(normalized[0], normalized[0] + n_pairs)):
        raise ProgressiveGeometryResidualError("source_pair_ids must be a contiguous ordered global window")
    if normalized[0] < 0 or normalized[-1] >= 2**32:
        raise ProgressiveGeometryResidualError("source_pair_ids escape uint32 custody")
    return normalized


def _pair_ids_sha256(ids: Sequence[int]) -> str:
    payload = np.asarray(tuple(ids), dtype="<u4").tobytes(order="C")
    return _sha256(payload)


def _write_uvarint(value: int, output: bytearray) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProgressiveGeometryResidualError("unsigned varint value is invalid")
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)


def _read_uvarint(payload: memoryview, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(payload):
            raise ProgressiveGeometryResidualError("residual record is truncated")
        byte = int(payload[offset])
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if shift and byte == 0:
                raise ProgressiveGeometryResidualError("residual varint is not minimally encoded")
            return value, offset
        shift += 7
    raise ProgressiveGeometryResidualError("residual varint exceeds uint64")


def _compress_with(codec: str, raw: bytes) -> bytes:
    if codec == "raw":
        return raw
    if codec == "zlib9":
        return zlib.compress(raw, 9)
    if codec == "bz2_9":
        return bz2.compress(raw, compresslevel=9)
    if codec == "lzma6":
        # Preset 6 is the standard library's bounded-memory default.  It is a
        # wire-format constant, not a scientific tuning parameter.
        return lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
    raise ProgressiveGeometryResidualError("unknown residual entropy codec")


def _compress_canonical(raw: bytes) -> tuple[str, bytes]:
    candidates = [
        (len(encoded := _compress_with(codec, raw)), index, codec, encoded) for index, codec in enumerate(_CODEC_ORDER)
    ]
    _, _, codec, encoded = min(candidates)
    return codec, encoded


def _decompress_exact(codec: str, payload: bytes, *, expected_bytes: int, maximum_bytes: int) -> bytes:
    if expected_bytes < 0 or expected_bytes > maximum_bytes:
        raise ProgressiveGeometryResidualError("residual raw-size custody exceeds geometry bound")
    try:
        if codec == "raw":
            raw = payload
        elif codec == "zlib9":
            decoder = zlib.decompressobj()
            raw = decoder.decompress(payload, expected_bytes + 1)
            if decoder.unconsumed_tail or not decoder.eof or decoder.unused_data:
                raise ProgressiveGeometryResidualError("zlib section is trailing or over-expanding")
        elif codec == "bz2_9":
            decoder = bz2.BZ2Decompressor()
            raw = decoder.decompress(payload, max_length=expected_bytes + 1)
            if not decoder.eof or decoder.unused_data:
                raise ProgressiveGeometryResidualError("bzip2 section is trailing or over-expanding")
        elif codec == "lzma6":
            decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
            raw = decoder.decompress(payload, max_length=expected_bytes + 1)
            if not decoder.eof or decoder.unused_data:
                raise ProgressiveGeometryResidualError("lzma section is trailing or over-expanding")
        else:
            raise ProgressiveGeometryResidualError("unknown residual entropy codec")
    except (OSError, EOFError, lzma.LZMAError, zlib.error) as exc:
        raise ProgressiveGeometryResidualError("residual entropy section is invalid") from exc
    if len(raw) != expected_bytes:
        raise ProgressiveGeometryResidualError("residual raw-size custody differs")
    return raw


def _temporal_codes(predictor: np.ndarray, target: np.ndarray) -> np.ndarray:
    mismatch = predictor != target
    codes = np.zeros_like(predictor, dtype=np.uint8)
    codes[1:] = np.where(
        mismatch[1:] & mismatch[:-1] & (predictor[1:] == predictor[:-1]) & (target[1:] == target[:-1]),
        target[1:] + np.uint8(1),
        np.uint8(0),
    )
    return codes


def _encode_temporal_row_runs(codes: np.ndarray) -> tuple[bytes, int]:
    records = bytearray()
    count = 0
    previous = 0
    _, height, width = codes.shape
    for pair in range(1, codes.shape[0]):
        for row in range(height):
            col = 0
            while col < width:
                code = int(codes[pair, row, col])
                if code == 0:
                    col += 1
                    continue
                start = col
                col += 1
                while col < width and int(codes[pair, row, col]) == code:
                    col += 1
                linear = (pair * height + row) * width + start
                _write_uvarint(linear - previous, records)
                _write_uvarint(col - start, records)
                records.append(code)
                previous = linear
                count += 1
    raw = bytearray()
    _write_uvarint(count, raw)
    raw.extend(records)
    return bytes(raw), count


def _pack_three_bit(values: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(values, dtype=np.uint8).reshape(-1)
    if flat.size and int(flat.max()) > 7:
        raise ProgressiveGeometryResidualError("three-bit temporal code escaped its alphabet")
    bits = np.unpackbits(flat[:, None], axis=1, bitorder="little")[:, :3].reshape(-1)
    return np.packbits(bits, bitorder="little").tobytes()


def _unpack_three_bit(payload: bytes, count: int) -> np.ndarray:
    expected = (count * 3 + 7) // 8
    if len(payload) != expected:
        raise ProgressiveGeometryResidualError("boundary-block code length differs")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    if bits.size > count * 3 and np.any(bits[count * 3 :]):
        raise ProgressiveGeometryResidualError("boundary-block padding bits must be zero")
    triples = bits[: count * 3].reshape(count, 3)
    return np.ascontiguousarray(triples @ np.asarray([1, 2, 4], dtype=np.uint8), dtype=np.uint8)


def _encode_temporal_blocks(codes: np.ndarray, block_size: int) -> tuple[bytes, int]:
    n_pairs, height, width = codes.shape
    block_rows = (height + block_size - 1) // block_size
    block_cols = (width + block_size - 1) // block_size
    body = bytearray()
    count = 0
    previous = 0
    for pair in range(1, n_pairs):
        for row0 in range(0, height, block_size):
            for col0 in range(0, width, block_size):
                block = codes[pair, row0 : row0 + block_size, col0 : col0 + block_size]
                if not np.any(block):
                    continue
                block_index = ((pair - 1) * block_rows + row0 // block_size) * block_cols + col0 // block_size
                _write_uvarint(block_index - previous, body)
                previous = block_index
                unique = np.unique(block)
                if len(unique) == 1 and int(unique[0]) != 0:
                    body.extend((0, int(unique[0])))
                else:
                    body.append(1)
                    body.extend(_pack_three_bit(block))
                count += 1
    raw = bytearray()
    _write_uvarint(block_size, raw)
    _write_uvarint(count, raw)
    raw.extend(body)
    return bytes(raw), count


@dataclass(frozen=True, slots=True)
class _TemporalEncoding:
    mode: str
    block_size: int
    raw: bytes
    codec: str
    payload: bytes
    record_count: int


def _choose_temporal_encoding(
    codes: np.ndarray,
    *,
    mode: Literal["auto", "row_runs", "block_context"],
    block_size: int | None,
) -> _TemporalEncoding:
    candidates: list[_TemporalEncoding] = []
    if mode in ("auto", "row_runs"):
        raw, count = _encode_temporal_row_runs(codes)
        codec, payload = _compress_canonical(raw)
        candidates.append(_TemporalEncoding("row_runs", 0, raw, codec, payload, count))
    if mode in ("auto", "block_context"):
        if block_size is not None:
            sizes = (block_size,)
        else:
            limit = min(codes.shape[1:])
            sizes = tuple(1 << exponent for exponent in range(1, limit.bit_length()) if 1 << exponent <= limit)
        if not sizes or any(size < 2 or size & (size - 1) for size in sizes):
            raise ProgressiveGeometryResidualError("temporal block size must be a power of two >=2")
        for size in sizes:
            raw, count = _encode_temporal_blocks(codes, size)
            codec, payload = _compress_canonical(raw)
            candidates.append(_TemporalEncoding("block_context", size, raw, codec, payload, count))
    if not candidates:
        raise ProgressiveGeometryResidualError("temporal mode is invalid")

    def whole_packet_variable_bytes(item: _TemporalEncoding) -> int:
        # Every other packet field is identical across these encodings.  Count
        # the variable-width metadata too, otherwise a one-byte payload win can
        # select a larger complete packet.
        variable_meta = {
            "mode": item.mode,
            "block_size": item.block_size,
            "codec": item.codec,
            "raw_bytes": len(item.raw),
            "payload_bytes": len(item.payload),
            "record_count": item.record_count,
            "span_count": item.record_count,
        }
        return len(item.payload) + len(_canonical_json(variable_meta))

    return min(
        candidates,
        key=lambda item: (whole_packet_variable_bytes(item), item.mode, item.block_size),
    )


@dataclass(frozen=True, slots=True)
class _Span:
    pair: int
    row: int
    start: int
    stop: int
    target_class: int

    @property
    def cells(self) -> int:
        return self.stop - self.start


def _components_for_class(mask: np.ndarray, *, pair: int, target_class: int) -> list[list[_Span]]:
    spans: list[_Span] = []
    by_row: dict[int, list[int]] = {}
    for row in range(mask.shape[0]):
        cols = np.flatnonzero(mask[row])
        cursor = 0
        while cursor < len(cols):
            start = int(cols[cursor])
            cursor += 1
            while cursor < len(cols) and int(cols[cursor]) == int(cols[cursor - 1]) + 1:
                cursor += 1
            stop = int(cols[cursor - 1]) + 1
            index = len(spans)
            spans.append(_Span(pair, row, start, stop, target_class))
            by_row.setdefault(row, []).append(index)
    parent = list(range(len(spans)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in range(1, mask.shape[0]):
        previous = by_row.get(row - 1, ())
        for current_index in by_row.get(row, ()):
            current = spans[current_index]
            for previous_index in previous:
                prior = spans[previous_index]
                if prior.start > current.stop:
                    break
                if prior.stop < current.start:
                    continue
                union(current_index, previous_index)
    groups: dict[int, list[_Span]] = {}
    for index, span in enumerate(spans):
        groups.setdefault(find(index), []).append(span)
    return [sorted(group, key=lambda span: (span.row, span.start)) for group in groups.values()]


def _extract_islands_and_tail(
    output: np.ndarray, target: np.ndarray
) -> tuple[list[list[_Span]], list[tuple[int, int]]]:
    remaining = output != target
    islands: list[list[_Span]] = []
    tail: list[tuple[int, int]] = []
    _, height, width = output.shape
    frame_cells = height * width
    for pair in range(output.shape[0]):
        for target_class in range(len(SEMANTIC_NAMES)):
            mask = remaining[pair] & (target[pair] == target_class)
            for component in _components_for_class(mask, pair=pair, target_class=target_class):
                cells = sum(span.cells for span in component)
                if cells == 1:
                    span = component[0]
                    tail.append((pair * frame_cells + span.row * width + span.start, target_class))
                else:
                    islands.append(component)
    islands.sort(key=lambda component: (component[0].pair, component[0].row, component[0].start))
    tail.sort()
    return islands, tail


def _encode_islands(islands: Sequence[Sequence[_Span]], *, height: int, width: int) -> tuple[bytes, int]:
    raw = bytearray()
    _write_uvarint(len(islands), raw)
    previous_first = 0
    span_count = 0
    frame_cells = height * width
    for component in islands:
        first = component[0].pair * frame_cells + component[0].row * width + component[0].start
        _write_uvarint(first - previous_first, raw)
        previous_first = first
        raw.append(component[0].target_class)
        _write_uvarint(sum(span.cells for span in component), raw)
        _write_uvarint(len(component), raw)
        previous_span = first
        for span in component:
            linear = span.pair * frame_cells + span.row * width + span.start
            _write_uvarint(linear - previous_span, raw)
            _write_uvarint(span.cells, raw)
            previous_span = linear
            span_count += 1
    return bytes(raw), span_count


def _encode_tail(tail: Sequence[tuple[int, int]]) -> bytes:
    raw = bytearray()
    _write_uvarint(len(tail), raw)
    previous = 0
    for linear, target_class in tail:
        _write_uvarint(linear - previous, raw)
        raw.append(target_class)
        previous = linear
    return bytes(raw)


def _apply_temporal(
    raw: bytes, *, mode: str, block_size: int, output: np.ndarray, predictor: np.ndarray
) -> tuple[int, int]:
    view = memoryview(raw)
    offset = 0
    n_pairs, height, width = output.shape
    total = output.size
    corrected = 0
    if mode == "row_runs":
        count, offset = _read_uvarint(view, offset)
        previous = 0
        for record in range(count):
            delta, offset = _read_uvarint(view, offset)
            length, offset = _read_uvarint(view, offset)
            if offset >= len(view):
                raise ProgressiveGeometryResidualError("temporal row-run code is truncated")
            code = int(view[offset])
            offset += 1
            linear = previous + delta
            pair, rem = divmod(linear, height * width)
            row, col = divmod(rem, width)
            if (
                pair <= 0
                or pair >= n_pairs
                or length <= 0
                or col + length > width
                or code not in range(1, 6)
                or (record and linear <= previous)
            ):
                raise ProgressiveGeometryResidualError("temporal row-run geometry is invalid")
            target_class = np.uint8(code - 1)
            current = predictor[pair, row, col : col + length]
            prior = predictor[pair - 1, row, col : col + length]
            if (
                np.any(current != prior)
                or np.any(current == target_class)
                or np.any(output[pair, row, col : col + length] != current)
            ):
                raise ProgressiveGeometryResidualError("temporal row-run does not belong to the bound predictor")
            output[pair, row, col : col + length] = target_class
            corrected += length
            previous = linear
        record_count = count
    elif mode == "block_context":
        encoded_size, offset = _read_uvarint(view, offset)
        count, offset = _read_uvarint(view, offset)
        if encoded_size != block_size or block_size < 2 or block_size & (block_size - 1):
            raise ProgressiveGeometryResidualError("temporal block-size custody differs")
        block_rows = (height + block_size - 1) // block_size
        block_cols = (width + block_size - 1) // block_size
        maximum_blocks = max(0, n_pairs - 1) * block_rows * block_cols
        previous = 0
        for record in range(count):
            delta, offset = _read_uvarint(view, offset)
            block_index = previous + delta
            if block_index >= maximum_blocks or (record and block_index <= previous) or offset >= len(view):
                raise ProgressiveGeometryResidualError("temporal block index is invalid")
            mode_id = int(view[offset])
            offset += 1
            pair_block, col_block = divmod(block_index, block_cols)
            pair_offset, row_block = divmod(pair_block, block_rows)
            pair = pair_offset + 1
            row0, col0 = row_block * block_size, col_block * block_size
            height_now, width_now = min(block_size, height - row0), min(block_size, width - col0)
            if mode_id == 0:
                if offset >= len(view):
                    raise ProgressiveGeometryResidualError("constant temporal block is truncated")
                code = int(view[offset])
                offset += 1
                codes = np.full((height_now, width_now), code, dtype=np.uint8)
            elif mode_id == 1:
                byte_count = (height_now * width_now * 3 + 7) // 8
                if offset + byte_count > len(view):
                    raise ProgressiveGeometryResidualError("boundary temporal block is truncated")
                codes = _unpack_three_bit(bytes(view[offset : offset + byte_count]), height_now * width_now).reshape(
                    height_now, width_now
                )
                offset += byte_count
            else:
                raise ProgressiveGeometryResidualError("temporal block mode is invalid")
            if not np.any(codes) or np.any(codes > 5):
                raise ProgressiveGeometryResidualError("temporal boundary block has invalid correction codes")
            active = codes != 0
            target_classes = codes - np.uint8(1)
            current = predictor[pair, row0 : row0 + height_now, col0 : col0 + width_now]
            prior = predictor[pair - 1, row0 : row0 + height_now, col0 : col0 + width_now]
            current_output = output[pair, row0 : row0 + height_now, col0 : col0 + width_now]
            if (
                np.any(current[active] != prior[active])
                or np.any(current[active] == target_classes[active])
                or np.any(current_output[active] != current[active])
            ):
                raise ProgressiveGeometryResidualError("temporal boundary block does not belong to the bound predictor")
            current_output[active] = target_classes[active]
            corrected += int(np.count_nonzero(active))
            previous = block_index
        record_count = count
    else:
        raise ProgressiveGeometryResidualError("temporal mode is invalid")
    if offset != len(view) or corrected > total:
        raise ProgressiveGeometryResidualError("temporal section has trailing records")
    return record_count, corrected


def _spans_connected(spans: Sequence[tuple[int, int, int]]) -> bool:
    if not spans:
        return False
    reached = {0}
    changed = True
    while changed:
        changed = False
        for left in tuple(reached):
            row, start, stop = spans[left]
            for right, (other_row, other_start, other_stop) in enumerate(spans):
                if right in reached or abs(row - other_row) != 1:
                    continue
                if other_start <= stop and other_stop >= start:
                    reached.add(right)
                    changed = True
    return len(reached) == len(spans)


def _apply_islands(raw: bytes, *, output: np.ndarray, predictor: np.ndarray) -> tuple[int, int, int]:
    view = memoryview(raw)
    offset = 0
    count, offset = _read_uvarint(view, offset)
    n_pairs, height, width = output.shape
    frame_cells = height * width
    previous_first = 0
    corrected = 0
    total_spans = 0
    for component_index in range(count):
        delta, offset = _read_uvarint(view, offset)
        first = previous_first + delta
        if offset >= len(view):
            raise ProgressiveGeometryResidualError("component class is truncated")
        target_class = int(view[offset])
        offset += 1
        cell_count, offset = _read_uvarint(view, offset)
        span_count, offset = _read_uvarint(view, offset)
        if (
            target_class not in range(5)
            or cell_count < 2
            or span_count < 1
            or (component_index and first <= previous_first)
        ):
            raise ProgressiveGeometryResidualError("component header is invalid")
        component_pair = first // frame_cells
        previous_span = first
        spans: list[tuple[int, int, int]] = []
        actual_cells = 0
        for span_index in range(span_count):
            span_delta, offset = _read_uvarint(view, offset)
            length, offset = _read_uvarint(view, offset)
            linear = previous_span + span_delta
            pair, rem = divmod(linear, frame_cells)
            row, col = divmod(rem, width)
            if (
                pair >= n_pairs
                or pair != component_pair
                or length < 1
                or col + length > width
                or (span_index == 0 and linear != first)
                or (span_index and linear <= previous_span)
            ):
                raise ProgressiveGeometryResidualError("component span geometry is invalid")
            current = output[pair, row, col : col + length]
            baseline = predictor[pair, row, col : col + length]
            if np.any(current != baseline) or np.any(baseline == target_class):
                raise ProgressiveGeometryResidualError("component overlaps prior strata or is inert")
            current[:] = np.uint8(target_class)
            spans.append((row, col, col + length))
            actual_cells += length
            previous_span = linear
        if actual_cells != cell_count or not _spans_connected(spans):
            raise ProgressiveGeometryResidualError("component atom is not one exact 8-connected island")
        corrected += actual_cells
        total_spans += span_count
        previous_first = first
    if offset != len(view):
        raise ProgressiveGeometryResidualError("component section has trailing records")
    return count, total_spans, corrected


def _apply_tail(raw: bytes, *, output: np.ndarray, predictor: np.ndarray) -> tuple[int, int]:
    view = memoryview(raw)
    offset = 0
    count, offset = _read_uvarint(view, offset)
    previous = 0
    frame_cells = output.shape[1] * output.shape[2]
    for event in range(count):
        delta, offset = _read_uvarint(view, offset)
        linear = previous + delta
        if offset >= len(view):
            raise ProgressiveGeometryResidualError("sparse-tail class is truncated")
        target_class = int(view[offset])
        offset += 1
        pair, rem = divmod(linear, frame_cells)
        row, col = divmod(rem, output.shape[2])
        if pair >= output.shape[0] or target_class not in range(5) or (event and linear <= previous):
            raise ProgressiveGeometryResidualError("sparse-tail event geometry is invalid")
        if output[pair, row, col] != predictor[pair, row, col] or predictor[pair, row, col] == target_class:
            raise ProgressiveGeometryResidualError("sparse-tail event overlaps prior strata or is inert")
        output[pair, row, col] = np.uint8(target_class)
        previous = linear
    if offset != len(view):
        raise ProgressiveGeometryResidualError("sparse-tail section has trailing records")
    return count, count


@dataclass(frozen=True, slots=True)
class ProgressiveGeometryResidual:
    """Strictly parsed packet with canonical raw section bytes."""

    header: Mapping[str, Any]
    section_payloads: tuple[bytes, bytes, bytes]
    raw_sections: tuple[bytes, bytes, bytes]


def _section_meta(
    *,
    name: str,
    order: int,
    mode: str,
    block_size: int,
    codec: str,
    raw: bytes,
    payload: bytes,
    record_count: int,
    span_count: int,
    corrected_cells: int,
    errors_before: int,
    errors_after: int,
    semantic_sha256_after: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "order": order,
        "mode": mode,
        "block_size": block_size,
        "codec": codec,
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "payload_bytes": len(payload),
        "payload_sha256": _sha256(payload),
        "record_count": record_count,
        "span_count": span_count,
        "corrected_cells": corrected_cells,
        "errors_before": errors_before,
        "errors_after": errors_after,
        "semantic_sha256_after": semantic_sha256_after,
    }


def build_progressive_geometry_residual(
    *,
    predictor_program: bytes,
    predictor_contract_id: str,
    predictor_renderer_sha256: str,
    predictor_labels: np.ndarray,
    target_labels: np.ndarray,
    source_pair_ids: Sequence[int],
    target_semantic_lineage: Literal["frozen_gt_argmax", "synthetic_fixture"],
    temporal_mode: Literal["auto", "row_runs", "block_context"] = "auto",
    temporal_block_size: int | None = None,
) -> bytes:
    """Build one exact, deterministic, predictor- and pair-bound PBR2 packet."""

    if not isinstance(predictor_program, bytes) or not predictor_program:
        raise ProgressiveGeometryResidualError("predictor_program must be non-empty counted bytes")
    contract = _require_contract(predictor_contract_id)
    renderer_sha = _require_sha256(predictor_renderer_sha256, "predictor_renderer_sha256")
    if target_semantic_lineage not in ("frozen_gt_argmax", "synthetic_fixture"):
        raise ProgressiveGeometryResidualError("target_semantic_lineage is invalid")
    predictor = _semantic_array(predictor_labels, label="predictor_labels")
    target = _semantic_array(target_labels, label="target_labels")
    if predictor.shape != target.shape:
        raise ProgressiveGeometryResidualError("predictor and target semantic geometry differs")
    n_pairs, height, width = (int(item) for item in predictor.shape)
    ids = _pair_ids(source_pair_ids, n_pairs=n_pairs)
    errors_initial = int(np.count_nonzero(predictor != target))

    temporal = _choose_temporal_encoding(
        _temporal_codes(predictor, target),
        mode=temporal_mode,
        block_size=temporal_block_size,
    )
    output = predictor.copy()
    temporal_records, temporal_cells = _apply_temporal(
        temporal.raw,
        mode=temporal.mode,
        block_size=temporal.block_size,
        output=output,
        predictor=predictor,
    )
    errors_after_temporal = int(np.count_nonzero(output != target))

    islands, tail = _extract_islands_and_tail(output, target)
    islands_raw, island_spans = _encode_islands(islands, height=height, width=width)
    islands_codec, islands_payload = _compress_canonical(islands_raw)
    island_count, decoded_spans, island_cells = _apply_islands(islands_raw, output=output, predictor=predictor)
    if decoded_spans != island_spans:
        raise ProgressiveGeometryResidualError("component span accounting changed on parse-back")
    errors_after_islands = int(np.count_nonzero(output != target))

    tail_raw = _encode_tail(tail)
    tail_codec, tail_payload = _compress_canonical(tail_raw)
    tail_count, tail_cells = _apply_tail(tail_raw, output=output, predictor=predictor)
    errors_after_tail = int(np.count_nonzero(output != target))
    if errors_after_tail or not np.array_equal(output, target):
        raise ProgressiveGeometryResidualError("progressive residual did not recover target labels exactly")

    strata = [
        _section_meta(
            name=STRATUM_NAMES[0],
            order=1,
            mode=temporal.mode,
            block_size=temporal.block_size,
            codec=temporal.codec,
            raw=temporal.raw,
            payload=temporal.payload,
            record_count=temporal_records,
            span_count=temporal_records,
            corrected_cells=temporal_cells,
            errors_before=errors_initial,
            errors_after=errors_after_temporal,
            semantic_sha256_after=_semantic_sha256(predictor if temporal_cells == 0 else predictor.copy()),
        ),
        _section_meta(
            name=STRATUM_NAMES[1],
            order=2,
            mode="connected_row_spans_8",
            block_size=0,
            codec=islands_codec,
            raw=islands_raw,
            payload=islands_payload,
            record_count=island_count,
            span_count=island_spans,
            corrected_cells=island_cells,
            errors_before=errors_after_temporal,
            errors_after=errors_after_islands,
            semantic_sha256_after="",
        ),
        _section_meta(
            name=STRATUM_NAMES[2],
            order=3,
            mode="singleton_delta_events",
            block_size=0,
            codec=tail_codec,
            raw=tail_raw,
            payload=tail_payload,
            record_count=tail_count,
            span_count=tail_count,
            corrected_cells=tail_cells,
            errors_before=errors_after_islands,
            errors_after=0,
            semantic_sha256_after=_semantic_sha256(target),
        ),
    ]
    # Recompute staged-application hashes from the real decoder order; never infer them
    # from target accounting or trust the construction scratch state.
    replay = predictor.copy()
    _apply_temporal(
        temporal.raw, mode=temporal.mode, block_size=temporal.block_size, output=replay, predictor=predictor
    )
    strata[0]["semantic_sha256_after"] = _semantic_sha256(replay)
    _apply_islands(islands_raw, output=replay, predictor=predictor)
    strata[1]["semantic_sha256_after"] = _semantic_sha256(replay)
    _apply_tail(tail_raw, output=replay, predictor=predictor)
    if not np.array_equal(replay, target):
        raise ProgressiveGeometryResidualError("fresh progressive parse-back changed target semantics")

    section_payloads = (temporal.payload, islands_payload, tail_payload)
    target_derived_section_bytes = sum(map(len, section_payloads))
    header = {
        "schema": PACKET_SCHEMA,
        "version": PACKET_VERSION,
        "predictor_contract_id": contract,
        "predictor_renderer_sha256": renderer_sha,
        "predictor_program_bytes": len(predictor_program),
        "predictor_program_sha256": _sha256(predictor_program),
        "predictor_semantic_bytes": int(predictor.size),
        "predictor_semantic_sha256": _semantic_sha256(predictor),
        "target_semantic_bytes": int(target.size),
        "target_semantic_sha256": _semantic_sha256(target),
        "n_pairs": n_pairs,
        "height": height,
        "width": width,
        "source_pair_start": ids[0],
        "source_pair_stop_exclusive": ids[-1] + 1,
        "source_pair_ids_sha256": _pair_ids_sha256(ids),
        "semantic_names": list(SEMANTIC_NAMES),
        "semantic_class_ids": list(range(len(SEMANTIC_NAMES))),
        "strata": strata,
        "separate_dense_target_table_section_bytes": 0,
        "pbr2_is_target_derived": True,
        "pbr2_target_derived_section_bytes": target_derived_section_bytes,
        "pbr2_event_count": errors_initial,
        "pbr2_event_density_numerator": errors_initial,
        "pbr2_event_density_denominator": int(predictor.size),
        "target_derived_residual_promotion_admitted": False,
        "research_only": True,
        "artifact_role": "encoder_side_conditional_entropy_measurement",
        "candidate_archive_admissible": False,
        "exact_target_semantic_reconstruction": True,
        "target_semantic_lineage": target_semantic_lineage,
        "pbr2_reconstructs_exact_gt_argmax": target_semantic_lineage == "frozen_gt_argmax",
        "reconstructed_target_semantic_bytes": int(target.size),
        "candidate_archive_blocker": "lossless predictor-conditional target-semantic-table encoding",
        "generic_apply_requires_external_predictor_semantics": True,
        "physical_prefix_decode_supported": False,
        "staged_application_requires_complete_packet": True,
        "decode_scorer_dependency": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    header_bytes = _canonical_json(header)
    prefix = _PREFIX.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        len(header_bytes),
        *(len(section) for section in section_payloads),
    )
    body = header_bytes + b"".join(section_payloads)
    return prefix + body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def decode_progressive_geometry_residual(payload: bytes) -> ProgressiveGeometryResidual:
    """Strictly parse, validate, decompress, and canonicalize a PBR2 packet."""

    if not isinstance(payload, bytes) or len(payload) < _PREFIX.size + _CRC.size:
        raise ProgressiveGeometryResidualError("progressive residual is truncated or not bytes")
    magic, version, header_size, temporal_size, islands_size, tail_size = _PREFIX.unpack_from(payload)
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise ProgressiveGeometryResidualError("progressive residual magic/version mismatch")
    sizes = (temporal_size, islands_size, tail_size)
    expected = _PREFIX.size + header_size + sum(sizes) + _CRC.size
    if len(payload) != expected:
        raise ProgressiveGeometryResidualError("progressive residual length mismatch or trailing bytes")
    body = payload[_PREFIX.size : -_CRC.size]
    (stored_crc,) = _CRC.unpack(payload[-_CRC.size :])
    if stored_crc != (zlib.crc32(body) & 0xFFFFFFFF):
        raise ProgressiveGeometryResidualError("progressive residual CRC mismatch")
    header_bytes = body[:header_size]
    try:
        header = json.loads(header_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProgressiveGeometryResidualError("progressive residual header is invalid") from exc
    if not isinstance(header, dict) or frozenset(header) != _HEADER_FIELDS or _canonical_json(header) != header_bytes:
        raise ProgressiveGeometryResidualError("progressive residual header fields/canonical form differ")
    if header["schema"] != PACKET_SCHEMA or header["version"] != PACKET_VERSION:
        raise ProgressiveGeometryResidualError("progressive residual schema/version differs")
    _require_contract(header["predictor_contract_id"])
    for field in (
        "predictor_renderer_sha256",
        "predictor_program_sha256",
        "predictor_semantic_sha256",
        "target_semantic_sha256",
        "source_pair_ids_sha256",
    ):
        _require_sha256(header[field], field)
    integer_fields = (
        "predictor_program_bytes",
        "predictor_semantic_bytes",
        "target_semantic_bytes",
        "n_pairs",
        "height",
        "width",
        "source_pair_start",
        "source_pair_stop_exclusive",
        "separate_dense_target_table_section_bytes",
        "pbr2_target_derived_section_bytes",
        "pbr2_event_count",
        "pbr2_event_density_numerator",
        "pbr2_event_density_denominator",
        "reconstructed_target_semantic_bytes",
    )
    if any(isinstance(header[name], bool) or not isinstance(header[name], int) for name in integer_fields):
        raise ProgressiveGeometryResidualError("progressive residual integer custody is malformed")
    n_pairs, height, width = header["n_pairs"], header["height"], header["width"]
    semantic_bytes = n_pairs * height * width
    if (
        min(n_pairs, height, width, header["predictor_program_bytes"]) <= 0
        or header["predictor_semantic_bytes"] != semantic_bytes
        or header["target_semantic_bytes"] != semantic_bytes
        or header["source_pair_stop_exclusive"] - header["source_pair_start"] != n_pairs
        or header["semantic_names"] != list(SEMANTIC_NAMES)
        or header["semantic_class_ids"] != list(range(5))
        or header["separate_dense_target_table_section_bytes"] != 0
        or header["pbr2_is_target_derived"] is not True
        or header["target_derived_residual_promotion_admitted"] is not False
        or header["research_only"] is not True
        or header["artifact_role"] != "encoder_side_conditional_entropy_measurement"
        or header["candidate_archive_admissible"] is not False
        or header["exact_target_semantic_reconstruction"] is not True
        or header["target_semantic_lineage"] not in ("frozen_gt_argmax", "synthetic_fixture")
        or header["pbr2_reconstructs_exact_gt_argmax"] is not (header["target_semantic_lineage"] == "frozen_gt_argmax")
        or header["reconstructed_target_semantic_bytes"] != semantic_bytes
        or header["candidate_archive_blocker"] != "lossless predictor-conditional target-semantic-table encoding"
        or header["generic_apply_requires_external_predictor_semantics"] is not True
        or header["physical_prefix_decode_supported"] is not False
        or header["staged_application_requires_complete_packet"] is not True
        or header["decode_scorer_dependency"] is not False
        or header["score_claim"] is not False
        or header["promotion_eligible"] is not False
    ):
        raise ProgressiveGeometryResidualError("progressive residual geometry or no-fake custody differs")
    ids = tuple(range(header["source_pair_start"], header["source_pair_stop_exclusive"]))
    if _pair_ids_sha256(ids) != header["source_pair_ids_sha256"]:
        raise ProgressiveGeometryResidualError("progressive residual pair-coordinate digest differs")
    strata = header["strata"]
    if not isinstance(strata, list) or len(strata) != 3:
        raise ProgressiveGeometryResidualError("progressive residual must contain exactly three strata")
    section_payloads: list[bytes] = []
    raw_sections: list[bytes] = []
    cursor = header_size
    previous_errors: int | None = None
    for index, (meta, size) in enumerate(zip(strata, sizes, strict=True)):
        if not isinstance(meta, dict) or frozenset(meta) != _STRATUM_FIELDS:
            raise ProgressiveGeometryResidualError("progressive residual stratum fields differ")
        if meta["name"] != STRATUM_NAMES[index] or meta["order"] != index + 1:
            raise ProgressiveGeometryResidualError("progressive residual stratum order differs")
        for field in ("raw_sha256", "payload_sha256", "semantic_sha256_after"):
            _require_sha256(meta[field], f"stratum.{field}")
        for field in (
            "block_size",
            "raw_bytes",
            "payload_bytes",
            "record_count",
            "span_count",
            "corrected_cells",
            "errors_before",
            "errors_after",
        ):
            if isinstance(meta[field], bool) or not isinstance(meta[field], int) or meta[field] < 0:
                raise ProgressiveGeometryResidualError("progressive residual stratum accounting is malformed")
        if meta["payload_bytes"] != size or meta["errors_before"] - meta["corrected_cells"] != meta["errors_after"]:
            raise ProgressiveGeometryResidualError("progressive residual stratum accounting does not close")
        if previous_errors is not None and meta["errors_before"] != previous_errors:
            raise ProgressiveGeometryResidualError("progressive residual error ledger is discontinuous")
        previous_errors = meta["errors_after"]
        section = body[cursor : cursor + size]
        cursor += size
        if _sha256(section) != meta["payload_sha256"]:
            raise ProgressiveGeometryResidualError("progressive residual section digest differs")
        raw = _decompress_exact(
            meta["codec"],
            section,
            expected_bytes=meta["raw_bytes"],
            maximum_bytes=semantic_bytes * 16 + 1024,
        )
        if _sha256(raw) != meta["raw_sha256"]:
            raise ProgressiveGeometryResidualError("progressive residual raw-section digest differs")
        section_payloads.append(section)
        raw_sections.append(raw)
    if previous_errors != 0 or cursor != len(body):
        raise ProgressiveGeometryResidualError("progressive residual does not close exact error debt")
    initial_errors = strata[0]["errors_before"]
    if (
        header["pbr2_target_derived_section_bytes"] != sum(sizes)
        or header["pbr2_event_count"] != initial_errors
        or header["pbr2_event_density_numerator"] != initial_errors
        or header["pbr2_event_density_denominator"] != semantic_bytes
    ):
        raise ProgressiveGeometryResidualError("progressive residual target-derived accounting differs")
    return ProgressiveGeometryResidual(
        header=header,
        section_payloads=tuple(section_payloads),  # type: ignore[arg-type]
        raw_sections=tuple(raw_sections),  # type: ignore[arg-type]
    )


def apply_progressive_geometry_residual(
    payload: bytes,
    *,
    predictor_program: bytes,
    predictor_contract_id: str,
    predictor_renderer_sha256: str,
    predictor_labels: np.ndarray,
    source_pair_ids: Sequence[int],
    max_strata: int = 3,
) -> np.ndarray:
    """Apply 0..3 strata from a complete validated packet after identities match."""

    decoded = decode_progressive_geometry_residual(payload)
    header = decoded.header
    predictor = _semantic_array(predictor_labels, label="predictor_labels")
    if isinstance(max_strata, bool) or not isinstance(max_strata, int) or max_strata not in range(4):
        raise ProgressiveGeometryResidualError("max_strata must be one of 0,1,2,3")
    if (
        not isinstance(predictor_program, bytes)
        or len(predictor_program) != header["predictor_program_bytes"]
        or _sha256(predictor_program) != header["predictor_program_sha256"]
    ):
        raise ProgressiveGeometryResidualError("predictor program identity differs from residual custody")
    if _require_contract(predictor_contract_id) != header["predictor_contract_id"]:
        raise ProgressiveGeometryResidualError("predictor contract differs from residual custody")
    if _require_sha256(predictor_renderer_sha256, "predictor_renderer_sha256") != header["predictor_renderer_sha256"]:
        raise ProgressiveGeometryResidualError("predictor renderer identity differs from residual custody")
    ids = _pair_ids(source_pair_ids, n_pairs=header["n_pairs"])
    if ids != tuple(range(header["source_pair_start"], header["source_pair_stop_exclusive"])):
        raise ProgressiveGeometryResidualError("source pair coordinates differ from residual custody")
    if (
        list(predictor.shape) != [header["n_pairs"], header["height"], header["width"]]
        or _semantic_sha256(predictor) != header["predictor_semantic_sha256"]
    ):
        raise ProgressiveGeometryResidualError("predictor semantic stream differs from residual custody")
    output = predictor.copy()
    if max_strata >= 1:
        meta = header["strata"][0]
        records, cells = _apply_temporal(
            decoded.raw_sections[0],
            mode=meta["mode"],
            block_size=meta["block_size"],
            output=output,
            predictor=predictor,
        )
        if (
            records != meta["record_count"]
            or cells != meta["corrected_cells"]
            or _semantic_sha256(output) != meta["semantic_sha256_after"]
        ):
            raise ProgressiveGeometryResidualError("temporal stratum accounting or semantic digest differs")
    if max_strata >= 2:
        meta = header["strata"][1]
        records, spans, cells = _apply_islands(decoded.raw_sections[1], output=output, predictor=predictor)
        if (
            records != meta["record_count"]
            or spans != meta["span_count"]
            or cells != meta["corrected_cells"]
            or _semantic_sha256(output) != meta["semantic_sha256_after"]
        ):
            raise ProgressiveGeometryResidualError("component stratum accounting or semantic digest differs")
    if max_strata >= 3:
        meta = header["strata"][2]
        records, cells = _apply_tail(decoded.raw_sections[2], output=output, predictor=predictor)
        if (
            records != meta["record_count"]
            or cells != meta["corrected_cells"]
            or _semantic_sha256(output) != header["target_semantic_sha256"]
        ):
            raise ProgressiveGeometryResidualError("complete residual target digest differs")
    return output


def packet_accounting(payload: bytes) -> dict[str, Any]:
    """Return exact section and cumulative byte/error accounting."""

    decoded = decode_progressive_geometry_residual(payload)
    header = decoded.header
    cumulative = _PREFIX.size + len(_canonical_json(header))
    strata: list[dict[str, Any]] = []
    for meta in header["strata"]:
        cumulative += meta["payload_bytes"]
        strata.append({**meta, "cumulative_section_bytes_inside_full_packet": cumulative})
    return {
        "schema": PACKET_SCHEMA,
        "packet_bytes": len(payload),
        "packet_sha256": _sha256(payload),
        "packet_prefix_header_bytes": _PREFIX.size,
        "header_bytes": len(_canonical_json(header)),
        "crc_bytes": _CRC.size,
        "strata": strata,
        "initial_error_cells": header["strata"][0]["errors_before"],
        "final_error_cells": header["strata"][-1]["errors_after"],
        "source_pair_start": header["source_pair_start"],
        "source_pair_stop_exclusive": header["source_pair_stop_exclusive"],
        "predictor_program_sha256": header["predictor_program_sha256"],
        "predictor_semantic_sha256": header["predictor_semantic_sha256"],
        "target_semantic_sha256": header["target_semantic_sha256"],
        "separate_dense_target_table_section_bytes": header["separate_dense_target_table_section_bytes"],
        "pbr2_is_target_derived": header["pbr2_is_target_derived"],
        "pbr2_target_derived_section_bytes": header["pbr2_target_derived_section_bytes"],
        "pbr2_event_count": header["pbr2_event_count"],
        "pbr2_event_density_numerator": header["pbr2_event_density_numerator"],
        "pbr2_event_density_denominator": header["pbr2_event_density_denominator"],
        "target_derived_residual_promotion_admitted": header["target_derived_residual_promotion_admitted"],
        "research_only": header["research_only"],
        "artifact_role": header["artifact_role"],
        "candidate_archive_admissible": header["candidate_archive_admissible"],
        "exact_target_semantic_reconstruction": header["exact_target_semantic_reconstruction"],
        "target_semantic_lineage": header["target_semantic_lineage"],
        "pbr2_reconstructs_exact_gt_argmax": header["pbr2_reconstructs_exact_gt_argmax"],
        "reconstructed_target_semantic_bytes": header["reconstructed_target_semantic_bytes"],
        "candidate_archive_blocker": header["candidate_archive_blocker"],
        "generic_apply_requires_external_predictor_semantics": header[
            "generic_apply_requires_external_predictor_semantics"
        ],
        "physical_prefix_decode_supported": header["physical_prefix_decode_supported"],
        "staged_application_requires_complete_packet": header["staged_application_requires_complete_packet"],
        "decode_scorer_dependency": False,
        "score_claim": False,
        "promotion_eligible": False,
    }


__all__ = [
    "PACKET_MAGIC",
    "PACKET_SCHEMA",
    "PACKET_VERSION",
    "ProgressiveGeometryResidual",
    "ProgressiveGeometryResidualError",
    "apply_progressive_geometry_residual",
    "build_progressive_geometry_residual",
    "decode_progressive_geometry_residual",
    "packet_accounting",
]
