"""RC1 lossless adaptive recode of the two RX1 MODEL section bodies.

This file is BOTH the encoder reference and the receiver implementation, on the
``runtime/dx2_cabac_coefficients.py`` precedent.  The builder copies these exact
bytes into the candidate runtime, then proves
``restore_*(apply_*(body, ...)) == body`` with a FRESH decoder.

Mechanism
---------
Both model bodies carry packed signed integer codes of a per-tensor (semantic) or
per-channel (hpac) bit width.  Today those bytes reach the archive through Brotli,
which is a generic byte-level coder: it never sees a code boundary.  RC1 unpacks the
codes and drives each one through a binary tree of adaptive probability bins -- the
same integer 12-bit model and the same carryless range coder the shipped DX2 rider
uses -- so the code alphabet is modelled directly.

The model has no transmitted table: every bin starts at 2048 and updates only from
already-coded bins, so the decoder reproduces the encoder's state exactly.  There is
no float and no device path.  Nothing video-derived crosses into runtime code: the
context set is (tensor index, tree node) or (bit depth, tree node), which is pure
geometry.

Stream layout
-------------
``semantic``: ``b"RC1S" | version | shift | u32 payload_len | metadata | payload``
    ``metadata`` is every NON-code byte of the SM3R body, in body order; ``payload``
    is the range-coded concatenation of every code field's values, in field order.
``hpac``: ``b"RC1H" | version | shift | u32 payload_len | prefix | payload | tail``
    ``prefix`` is the IHS1 magic plus its packed per-channel depth table and ``tail``
    is every byte after the packed weight bitstream, both carried verbatim.
"""

from __future__ import annotations

import struct

import numpy as np

RC1_VERSION = 1
SEMANTIC_MAGIC = b"RC1S"
HPAC_MAGIC = b"RC1H"
RC1_HEADER = struct.Struct("<4sBBI")
PROBABILITY_ONE = 4096
PROBABILITY_INITIAL = 2048


class Rc1CodecError(ValueError):
    """An RC1 stream is malformed or fails its identity control."""


# --------------------------------------------------------------- range coder core


class _RangeEncoder:
    """Carryless 32-bit range encoder (identical to the shipped DX2 encoder)."""

    __slots__ = ("cache", "cache_size", "low", "out", "range")

    def __init__(self) -> None:
        self.low = 0
        self.range = 0xFFFFFFFF
        self.out = bytearray()
        self.cache = 0xFF
        self.cache_size = 0

    def _shift_low(self) -> None:
        if self.low < 0xFF000000 or self.low > 0xFFFFFFFF:
            carry = self.low >> 32
            if self.cache_size:
                self.out.append((self.cache + carry) & 0xFF)
            for _ in range(self.cache_size - 1):
                self.out.append((0xFF + carry) & 0xFF)
            self.cache = (self.low >> 24) & 0xFF
            self.cache_size = 0
        self.cache_size += 1
        self.low = (self.low << 8) & 0xFFFFFFFF

    def encode(self, cumulative_low: int, frequency: int, total: int) -> None:
        unit = self.range // total
        self.low += unit * cumulative_low
        self.range = unit * frequency
        while self.range < (1 << 24):
            self.range <<= 8
            self._shift_low()

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class _RangeDecoder:
    """Integer inverse of :class:`_RangeEncoder`; no float or device path exists."""

    __slots__ = ("buffer", "code", "position", "range")

    def __init__(self, payload: bytes) -> None:
        if len(payload) < 4:
            raise Rc1CodecError("RC1 payload is shorter than its range prefix")
        self.buffer = payload
        self.position = 0
        self.range = 0xFFFFFFFF
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF

    def _byte(self) -> int:
        if self.position < len(self.buffer):
            value = self.buffer[self.position]
            self.position += 1
            return value
        self.position += 1
        return 0

    def decode_frequency(self, total: int) -> int:
        unit = self.range // total
        value = self.code // unit
        return total - 1 if value >= total else value

    def update(self, cumulative_low: int, frequency: int, total: int) -> None:
        unit = self.range // total
        self.code -= unit * cumulative_low
        self.range = unit * frequency
        while self.range < (1 << 24):
            self.range <<= 8
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF


class _BinaryTreeModel:
    """One adaptive binary tree per context group; bins start at 2048 and adapt."""

    __slots__ = ("bins", "shift")

    def __init__(self, shift: int) -> None:
        if not 1 <= shift <= 8:
            raise Rc1CodecError("RC1 adaptation shift is outside 1..8")
        self.shift = shift
        self.bins: dict[tuple[int, int], int] = {}

    def encode(self, encoder: _RangeEncoder, group: int, value: int, bits: int) -> None:
        node = 1
        for index in range(bits - 1, -1, -1):
            bit = (value >> index) & 1
            key = (group, node)
            zero = self.bins.get(key, PROBABILITY_INITIAL)
            if bit:
                encoder.encode(zero, PROBABILITY_ONE - zero, PROBABILITY_ONE)
                zero -= zero >> self.shift
            else:
                encoder.encode(0, zero, PROBABILITY_ONE)
                zero += (PROBABILITY_ONE - zero) >> self.shift
            self.bins[key] = zero
            node = node * 2 + bit

    def decode(self, decoder: _RangeDecoder, group: int, bits: int) -> int:
        node = 1
        value = 0
        for _ in range(bits):
            key = (group, node)
            zero = self.bins.get(key, PROBABILITY_INITIAL)
            bit = 0 if decoder.decode_frequency(PROBABILITY_ONE) < zero else 1
            if bit:
                decoder.update(zero, PROBABILITY_ONE - zero, PROBABILITY_ONE)
                zero -= zero >> self.shift
            else:
                decoder.update(0, zero, PROBABILITY_ONE)
                zero += (PROBABILITY_ONE - zero) >> self.shift
            self.bins[key] = zero
            value = (value << 1) | bit
            node = node * 2 + bit
        return value


# ------------------------------------------------------------------ bit packing


def unpack_signed_codes(payload: bytes, count: int, bits: int) -> np.ndarray:
    """Little-endian bit order, two's complement -- the receiver's own convention."""

    if not 1 <= bits <= 8:
        raise Rc1CodecError(f"invalid code width {bits}")
    byte_count = (count * bits + 7) // 8
    if len(payload) < byte_count:
        raise Rc1CodecError("truncated signed code stream")
    packed = np.frombuffer(payload[:byte_count], dtype=np.uint8)
    stream = np.unpackbits(packed, bitorder="little")[: count * bits]
    stream = stream.reshape(count, bits).astype(np.int32, copy=False)
    unsigned = (stream * (1 << np.arange(bits, dtype=np.int32))).sum(axis=1)
    sign = 1 << (bits - 1)
    return np.where(unsigned >= sign, unsigned - (1 << bits), unsigned).astype(np.int32)


def pack_signed_codes(values: np.ndarray, bits: int) -> bytes:
    """Exact inverse of :func:`unpack_signed_codes`."""

    unsigned = (np.asarray(values, dtype=np.int32) & ((1 << bits) - 1)).astype(np.int32)
    stream = np.zeros((unsigned.size, bits), dtype=np.uint8)
    for index in range(bits):
        stream[:, index] = (unsigned >> index) & 1
    flat = stream.reshape(-1)
    pad = (-flat.size) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(flat, bitorder="little").tobytes()


def _encode_groups(
    groups: list[tuple[int, int, np.ndarray]], shift: int
) -> bytes:
    """Range-code ``(group, bits, values)`` blocks with one shared context bank."""

    model = _BinaryTreeModel(shift)
    encoder = _RangeEncoder()
    for group, bits, values in groups:
        mask = (1 << bits) - 1
        for value in (np.asarray(values, dtype=np.int32) & mask).tolist():
            model.encode(encoder, group, int(value), bits)
    return encoder.finish()


def _decode_groups(
    payload: bytes, plan: list[tuple[int, int, int]], shift: int
) -> list[np.ndarray]:
    """Inverse of :func:`_encode_groups` given ``(group, bits, count)`` blocks."""

    model = _BinaryTreeModel(shift)
    decoder = _RangeDecoder(payload)
    out: list[np.ndarray] = []
    for group, bits, count in plan:
        sign = 1 << (bits - 1)
        span = 1 << bits
        values = np.empty(count, dtype=np.int32)
        for index in range(count):
            raw = model.decode(decoder, group, bits)
            values[index] = raw - span if raw >= sign else raw
        out.append(values)
    return out


# ------------------------------------------------------------------ SM3R geometry

SM3R_MAGIC = b"SM3R"
SM3R_MIXED_MODE = 6
ROW_PRUNE_NAMES = frozenset(
    {"blocks.1.film.weight", "blocks.2.film.weight", "blocks.3.film.weight"}
)


def walk_sm3r(
    read,
    template,
    header: tuple[int, int, int, int],
) -> list[dict]:
    """Drive the SM3R mode-6 layout, calling ``read(kind, length)`` in body order.

    ``read`` returns the bytes for a metadata run and, for a ``codes`` run, is called
    with the run's length so the caller can either consume packed bytes (encode) or
    skip them (decode).  The returned plan lists every field with its kind, byte
    length, and -- for code runs -- ``(group, bits, count)``.
    """

    _version, _mode, keep_percent, _reserved = header
    names = [name for name, value in template.items() if value.ndim >= 2]
    plan: list[dict] = []
    depth_bytes = (len(names) + 1) // 2
    depth_blob = read("depth_table", depth_bytes)
    packed = np.frombuffer(depth_blob, dtype=np.uint8)
    values = np.empty(depth_bytes * 2, dtype=np.uint8)
    values[0::2] = packed & 0xF
    values[1::2] = packed >> 4
    depths = {
        name: int(value)
        for name, value in zip(names, values[: len(names)].tolist(), strict=True)
    }
    if any(depth < 2 or depth > 8 for depth in depths.values()):
        raise Rc1CodecError("SM3R depth table is outside 2..8")
    plan.append({"kind": "depth_table", "length": depth_bytes, "blob": depth_blob})

    group = 0
    for name, value in template.items():
        numel = int(value.numel())
        if value.ndim < 2:
            plan.append(
                {
                    "kind": "meta",
                    "length": numel * 2,
                    "blob": read("fp16_tensor", numel * 2),
                }
            )
            continue
        bits = depths[name]
        if name not in ROW_PRUNE_NAMES:
            scale_count = int(
                value.shape[-1] if name.endswith("embed.weight") else value.shape[0]
            )
            plan.append(
                {
                    "kind": "meta",
                    "length": scale_count * 2,
                    "blob": read("fp16_scales", scale_count * 2),
                }
            )
            length = (numel * bits + 7) // 8
            plan.append(
                {
                    "kind": "codes",
                    "length": length,
                    "group": group,
                    "bits": bits,
                    "count": numel,
                    "blob": read("codes", length),
                }
            )
            group += 1
            continue
        rows = int(value.shape[0])
        mask_bytes = (rows + 7) // 8
        mask_blob = read("prune_mask", mask_bytes)
        plan.append({"kind": "meta", "length": mask_bytes, "blob": mask_blob})
        selected = np.unpackbits(
            np.frombuffer(mask_blob, dtype=np.uint8), bitorder="little"
        )[:rows]
        keep = int(selected.sum())
        expected = max(1, round(rows * keep_percent / 100.0))
        if keep != expected:
            raise Rc1CodecError(f"{name}: kept {keep} rows, header implies {expected}")
        plan.append(
            {"kind": "meta", "length": keep * 2, "blob": read("fp16_scales", keep * 2)}
        )
        count = keep * (numel // rows)
        length = (count * bits + 7) // 8
        plan.append(
            {
                "kind": "codes",
                "length": length,
                "group": group,
                "bits": bits,
                "count": count,
                "blob": read("codes", length),
            }
        )
        group += 1
    return plan


def apply_semantic(body: bytes, template, shift: int) -> bytes:
    """Recode an SM3R mode-6 body into its RC1 rider form."""

    if len(body) < 10 or not body.startswith(SM3R_MAGIC):
        raise Rc1CodecError("not an SM3R body")
    version, mode, keep_percent, reserved = body[4:8]
    if version != 1 or mode != SM3R_MIXED_MODE or reserved != 0:
        raise Rc1CodecError("RC1 semantic rider only covers SM3R v1 mode 6")
    cursor = 10  # 8-byte header + 2-byte selection mask

    def read(_kind: str, length: int) -> bytes:
        nonlocal cursor
        if cursor + length > len(body):
            raise Rc1CodecError("SM3R body ended early")
        chunk = body[cursor : cursor + length]
        cursor += length
        return chunk

    plan = walk_sm3r(read, template, (version, mode, keep_percent, reserved))
    if cursor != len(body):
        raise Rc1CodecError(f"SM3R walk ended at {cursor} of {len(body)}")
    metadata = b"".join(item["blob"] for item in plan if item["kind"] == "meta")
    groups = [
        (
            item["group"],
            item["bits"],
            unpack_signed_codes(item["blob"], item["count"], item["bits"]),
        )
        for item in plan
        if item["kind"] == "codes"
    ]
    payload = _encode_groups(groups, shift)
    return b"".join(
        (
            RC1_HEADER.pack(SEMANTIC_MAGIC, RC1_VERSION, shift, len(payload)),
            body[:10],
            plan[0]["blob"],
            metadata,
            payload,
        )
    )


def restore_semantic(stream: bytes, template) -> bytes:
    """Exact inverse of :func:`apply_semantic`."""

    if len(stream) < RC1_HEADER.size:
        raise Rc1CodecError("truncated RC1 semantic header")
    magic, version, shift, payload_length = RC1_HEADER.unpack_from(stream)
    if magic != SEMANTIC_MAGIC or version != RC1_VERSION:
        raise Rc1CodecError("unsupported RC1 semantic stream")
    offset = RC1_HEADER.size
    head = stream[offset : offset + 10]
    if len(head) != 10 or not head.startswith(SM3R_MAGIC):
        raise Rc1CodecError("RC1 semantic stream lost its SM3R header")
    offset += 10
    body_version, mode, keep_percent, reserved = head[4:8]
    meta_end = len(stream) - payload_length
    if meta_end < offset:
        raise Rc1CodecError("RC1 semantic payload length exceeds the stream")
    cursor = offset

    def read(kind: str, length: int) -> bytes:
        nonlocal cursor
        if kind == "codes":
            return b""
        if cursor + length > meta_end:
            raise Rc1CodecError("RC1 semantic metadata ended early")
        chunk = stream[cursor : cursor + length]
        cursor += length
        return chunk

    plan = walk_sm3r(read, template, (body_version, mode, keep_percent, reserved))
    if cursor != meta_end:
        raise Rc1CodecError(f"RC1 semantic metadata walk ended at {cursor}")
    code_plan = [
        (item["group"], item["bits"], item["count"])
        for item in plan
        if item["kind"] == "codes"
    ]
    decoded = _decode_groups(stream[meta_end:], code_plan, shift)
    pieces = [head]
    index = 0
    for item in plan:
        if item["kind"] == "codes":
            pieces.append(pack_signed_codes(decoded[index], item["bits"]))
            index += 1
        else:
            pieces.append(item["blob"])
    return b"".join(pieces)


# ------------------------------------------------------------------ IHS1 geometry

IHS1_MAGIC = b"IHS1"


def ihs1_geometry(model, hpac_module) -> tuple[list[int], int]:
    """Per-channel weight counts from the deployed HPAC model (pure geometry)."""

    import torch

    compressible = (hpac_module.IntegerConv2d, hpac_module.IntegerLinear)
    counts: list[int] = []
    for module in model.modules():
        if not isinstance(module, compressible):
            continue
        weight = module.weight
        if isinstance(module, hpac_module.IntegerConv2d):
            mask = module.mask.to(torch.bool).expand_as(weight)
            counts.extend(int(mask[i].sum().item()) for i in range(weight.shape[0]))
        else:
            counts.extend(int(weight[i].numel()) for i in range(weight.shape[0]))
    return counts, len(counts)


def _ihs1_depths(body: bytes, channel_count: int) -> tuple[np.ndarray, int]:
    depth_bytes = (channel_count + 1) // 2
    packed = np.frombuffer(
        body[len(IHS1_MAGIC) : len(IHS1_MAGIC) + depth_bytes], dtype=np.uint8
    )
    values = np.empty(depth_bytes * 2, dtype=np.uint8)
    values[0::2] = packed & 0xF
    values[1::2] = packed >> 4
    return values[:channel_count].astype(np.int64), depth_bytes


def apply_hpac(body: bytes, row_counts: list[int], shift: int) -> bytes:
    """Recode an IHS1 body's packed weight bitstream into its RC1 rider form."""

    if not body.startswith(IHS1_MAGIC):
        raise Rc1CodecError("not an IHS1 body")
    depths, depth_bytes = _ihs1_depths(body, len(row_counts))
    if np.any(depths > 8):
        raise Rc1CodecError("IHS1 depth table is outside 0..8")
    total_bits = int(sum(int(b) * c for b, c in zip(depths.tolist(), row_counts, strict=True)))
    weight_offset = len(IHS1_MAGIC) + depth_bytes
    weight_bytes = (total_bits + 7) // 8
    tail_offset = weight_offset + weight_bytes
    if tail_offset > len(body):
        raise Rc1CodecError("IHS1 weight bitstream overruns the body")
    packed = np.frombuffer(body[weight_offset:tail_offset], dtype=np.uint8)
    bitstream = np.unpackbits(packed, bitorder="little")[:total_bits]
    groups: list[tuple[int, int, np.ndarray]] = []
    cursor = 0
    for bits, count in zip(depths.tolist(), row_counts, strict=True):
        bits = int(bits)
        if bits == 0:
            continue
        span = count * bits
        block = bitstream[cursor : cursor + span].reshape(count, bits).astype(np.int32)
        unsigned = (block * (1 << np.arange(bits, dtype=np.int32))).sum(axis=1)
        groups.append((bits, bits, unsigned.astype(np.int32)))
        cursor += span
    if cursor != total_bits:
        raise Rc1CodecError("IHS1 encode walk did not consume the weight bitstream")
    payload = _encode_groups(groups, shift)
    return b"".join(
        (
            RC1_HEADER.pack(HPAC_MAGIC, RC1_VERSION, shift, len(payload)),
            body[:weight_offset],
            payload,
            body[tail_offset:],
        )
    )


def restore_hpac(stream: bytes, row_counts: list[int]) -> bytes:
    """Exact inverse of :func:`apply_hpac`."""

    if len(stream) < RC1_HEADER.size:
        raise Rc1CodecError("truncated RC1 hpac header")
    magic, version, shift, payload_length = RC1_HEADER.unpack_from(stream)
    if magic != HPAC_MAGIC or version != RC1_VERSION:
        raise Rc1CodecError("unsupported RC1 hpac stream")
    body = stream[RC1_HEADER.size :]
    if not body.startswith(IHS1_MAGIC):
        raise Rc1CodecError("RC1 hpac stream lost its IHS1 magic")
    depths, depth_bytes = _ihs1_depths(body, len(row_counts))
    weight_offset = len(IHS1_MAGIC) + depth_bytes
    if weight_offset + payload_length > len(body):
        raise Rc1CodecError("RC1 hpac payload length exceeds the stream")
    payload = body[weight_offset : weight_offset + payload_length]
    tail = body[weight_offset + payload_length :]
    plan = [
        (int(bits), int(bits), count)
        for bits, count in zip(depths.tolist(), row_counts, strict=True)
        if int(bits)
    ]
    decoded = _decode_groups(payload, plan, shift)
    chunks: list[np.ndarray] = []
    for values, bits in zip(decoded, (item[1] for item in plan), strict=True):
        unsigned = (np.asarray(values, dtype=np.int32) & ((1 << bits) - 1)).astype(
            np.int32
        )
        block = np.zeros((unsigned.size, bits), dtype=np.uint8)
        for index in range(bits):
            block[:, index] = (unsigned >> index) & 1
        chunks.append(block.reshape(-1))
    flat = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint8)
    pad = (-flat.size) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    weights = np.packbits(flat, bitorder="little").tobytes()
    return body[:weight_offset] + weights + tail


__all__ = [
    "HPAC_MAGIC",
    "RC1_VERSION",
    "SEMANTIC_MAGIC",
    "Rc1CodecError",
    "apply_hpac",
    "apply_semantic",
    "ihs1_geometry",
    "restore_hpac",
    "restore_semantic",
]
