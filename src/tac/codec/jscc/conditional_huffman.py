# SPDX-License-Identifier: MIT
"""Scorer-conditional JSCC byte allocation plus a deterministic Huffman coder.

The coder is intentionally small: it provides a concrete lossless prototype
that can be used in packet experiments while keeping all scorer-conditioned
metadata non-authoritative. Encoded packets carry code lengths and bytes only;
they do not carry score claims, ranks, or promotion eligibility.
"""

from __future__ import annotations

import dataclasses
import heapq
import math
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

LEGACY_JSCC_HUFFMAN_MAGIC = b"JSCC"
"""Magic bytes for the legacy scorer-conditional Huffman packet format."""

_MAGIC = LEGACY_JSCC_HUFFMAN_MAGIC
_VERSION = 1
_HEADER = ">4sB3xII"
_HEADER_SIZE = struct.calcsize(_HEADER)
_ALPHABET_SIZE = 256
_EVIDENCE_GRADE = "proxy_only_legacy_jscc_huffman"


@dataclasses.dataclass(frozen=True)
class ScorerConditionalSignal:
    """A scorer-relevant conditioning signal, explicitly not a score claim."""

    name: str
    values: tuple[float, ...]
    kind: Literal["sensitivity", "distortion_proxy", "posterior", "feature"] = "feature"
    score_claim: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("signal name must be non-empty")
        if self.score_claim:
            raise ValueError("ScorerConditionalSignal cannot carry score_claim=True")
        if not self.values:
            raise ValueError("signal values must be non-empty")
        if any(not math.isfinite(float(v)) for v in self.values):
            raise ValueError("signal values must be finite")


@dataclasses.dataclass(frozen=True)
class JSCCCodingContext:
    """Context for scorer-conditional archive coding.

    ``score_claim`` and ``promotion_eligible`` are fixed false by validation.
    This lets downstream manifests distinguish packet-construction signals from
    exact-eval results.
    """

    section_name: str
    signals: tuple[ScorerConditionalSignal, ...] = ()
    exact_eval_packet_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    evidence_grade: str = _EVIDENCE_GRADE
    proxy: bool = True
    proxy_only: bool = True

    def __post_init__(self) -> None:
        if not self.section_name:
            raise ValueError("section_name must be non-empty")
        if self.score_claim:
            raise ValueError("JSCCCodingContext cannot carry score_claim=True")
        if self.promotion_eligible:
            raise ValueError("JSCCCodingContext cannot be promotion eligible")
        if self.ready_for_exact_eval_dispatch:
            raise ValueError(
                "JSCCCodingContext cannot be ready_for_exact_eval_dispatch"
            )
        if self.evidence_grade != _EVIDENCE_GRADE:
            raise ValueError(
                f"JSCCCodingContext evidence_grade must be {_EVIDENCE_GRADE!r}"
            )
        if not self.proxy or not self.proxy_only:
            raise ValueError("JSCCCodingContext is proxy-only by construction")

    def manifest_metadata(self) -> dict[str, object]:
        """Return JSON-safe context metadata for packet manifests."""

        return {
            "section_name": self.section_name,
            "format_family": "legacy_jscc_huffman",
            "legacy_huffman_magic": LEGACY_JSCC_HUFFMAN_MAGIC.decode("ascii"),
            "signals": [
                {"name": s.name, "kind": s.kind, "values": list(s.values)}
                for s in self.signals
            ],
            "exact_eval_packet_only": self.exact_eval_packet_only,
            "evidence_grade": _EVIDENCE_GRADE,
            "proxy": True,
            "proxy_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }


@dataclasses.dataclass(frozen=True)
class JSCCSection:
    """A byte-allocation section with a non-authoritative scorer weight."""

    name: str
    raw_bytes: int
    scorer_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("section name must be non-empty")
        if self.raw_bytes < 0:
            raise ValueError("raw_bytes must be non-negative")
        if not math.isfinite(self.scorer_weight) or self.scorer_weight < 0.0:
            raise ValueError("scorer_weight must be finite and non-negative")


@dataclasses.dataclass(frozen=True)
class JSCCEncodedPacket:
    """Encoded bytes plus non-authoritative JSCC metadata."""

    data: bytes
    context: JSCCCodingContext
    original_size: int
    encoded_size: int
    code_lengths: tuple[int, ...]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    evidence_grade: str = _EVIDENCE_GRADE
    proxy: bool = True
    proxy_only: bool = True

    def __post_init__(self) -> None:
        if (
            self.score_claim
            or self.promotion_eligible
            or self.ready_for_exact_eval_dispatch
        ):
            raise ValueError("JSCCEncodedPacket cannot be score/promote authoritative")
        if self.evidence_grade != _EVIDENCE_GRADE:
            raise ValueError(
                f"JSCCEncodedPacket evidence_grade must be {_EVIDENCE_GRADE!r}"
            )
        if not self.proxy or not self.proxy_only:
            raise ValueError("JSCCEncodedPacket is proxy-only by construction")
        if len(self.code_lengths) != _ALPHABET_SIZE:
            raise ValueError("code_lengths must have 256 entries")
        if self.encoded_size != len(self.data):
            raise ValueError("encoded_size must match data length")


def allocate_scorer_conditional_bytes(
    sections: Sequence[JSCCSection],
    *,
    total_bytes: int,
) -> dict[str, int]:
    """Allocate integer bytes by ``raw_bytes * scorer_weight``.

    The allocation is deterministic and sums exactly to ``total_bytes``. It is
    a packet-planning primitive only, not a score/rank authority.
    """

    if total_bytes < 0:
        raise ValueError("total_bytes must be non-negative")
    if not sections:
        raise ValueError("sections must be non-empty")
    if len({s.name for s in sections}) != len(sections):
        raise ValueError("section names must be unique")
    weights = [float(s.raw_bytes) * float(s.scorer_weight) for s in sections]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        base = total_bytes // len(sections)
        alloc = {s.name: base for s in sections}
        for section in sorted(sections, key=lambda item: item.name)[: total_bytes % len(sections)]:
            alloc[section.name] += 1
        return alloc

    raw = [total_bytes * w / total_weight for w in weights]
    floors = [math.floor(v) for v in raw]
    remainder = total_bytes - sum(floors)
    order = sorted(
        range(len(sections)),
        key=lambda i: (-(raw[i] - floors[i]), sections[i].name),
    )
    for i in order[:remainder]:
        floors[i] += 1
    return {section.name: floors[i] for i, section in enumerate(sections)}


def _conditioned_frequencies(
    payload: bytes,
    symbol_prior: Sequence[float] | None,
    prior_strength: float,
) -> list[int]:
    counts = Counter(payload)
    frequencies = [int(counts.get(sym, 0)) for sym in range(_ALPHABET_SIZE)]
    if symbol_prior is None or prior_strength == 0.0:
        return frequencies
    if len(symbol_prior) != _ALPHABET_SIZE:
        raise ValueError("symbol_prior must contain 256 entries")
    if prior_strength < 0.0 or not math.isfinite(prior_strength):
        raise ValueError("prior_strength must be finite and non-negative")
    prior = [float(v) for v in symbol_prior]
    if any((not math.isfinite(v)) or v < 0.0 for v in prior):
        raise ValueError("symbol_prior values must be finite and non-negative")
    prior_sum = sum(prior)
    if prior_sum <= 0.0:
        raise ValueError("symbol_prior must have positive total mass")
    mass = max(1.0, prior_strength * max(1, len(payload)))
    for sym, value in enumerate(prior):
        frequencies[sym] += round(value / prior_sum * mass)
    return frequencies


@dataclasses.dataclass(order=True)
class _HeapNode:
    weight: int
    min_symbol: int
    serial: int
    symbol: int | None = dataclasses.field(compare=False, default=None)
    left: _HeapNode | None = dataclasses.field(compare=False, default=None)
    right: _HeapNode | None = dataclasses.field(compare=False, default=None)


def _huffman_code_lengths(frequencies: Sequence[int]) -> tuple[int, ...]:
    if len(frequencies) != _ALPHABET_SIZE:
        raise ValueError("frequencies must contain 256 entries")
    heap: list[_HeapNode] = []
    serial = 0
    for symbol, weight in enumerate(frequencies):
        weight_i = int(weight)
        if weight_i < 0:
            raise ValueError("frequencies must be non-negative")
        if weight_i > 0:
            heapq.heappush(heap, _HeapNode(weight_i, symbol, serial, symbol=symbol))
            serial += 1
    if not heap:
        return (0,) * _ALPHABET_SIZE
    if len(heap) == 1:
        only = heap[0].symbol
        lengths = [0] * _ALPHABET_SIZE
        assert only is not None
        lengths[only] = 1
        return tuple(lengths)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = _HeapNode(
            left.weight + right.weight,
            min(left.min_symbol, right.min_symbol),
            serial,
            left=left,
            right=right,
        )
        serial += 1
        heapq.heappush(heap, parent)
    lengths = [0] * _ALPHABET_SIZE

    def walk(node: _HeapNode, depth: int) -> None:
        if node.symbol is not None:
            lengths[node.symbol] = max(1, depth)
            return
        assert node.left is not None and node.right is not None
        walk(node.left, depth + 1)
        walk(node.right, depth + 1)

    walk(heap[0], 0)
    return tuple(lengths)


def _canonical_codes(code_lengths: Sequence[int]) -> dict[int, tuple[int, int]]:
    if len(code_lengths) != _ALPHABET_SIZE:
        raise ValueError("code_lengths must contain 256 entries")
    pairs = sorted(
        (int(length), symbol)
        for symbol, length in enumerate(code_lengths)
        if int(length) > 0
    )
    codes: dict[int, tuple[int, int]] = {}
    code = 0
    previous_length = 0
    for length, symbol in pairs:
        if length > 255:
            raise ValueError("Huffman code length exceeds packet format")
        code <<= length - previous_length
        codes[symbol] = (code, length)
        code += 1
        previous_length = length
    return codes


def _write_bits(symbols: bytes, codes: Mapping[int, tuple[int, int]]) -> tuple[bytes, int]:
    out = bytearray()
    current = 0
    used = 0
    bit_length = 0
    for symbol in symbols:
        code, length = codes[symbol]
        for shift in range(length - 1, -1, -1):
            current = (current << 1) | ((code >> shift) & 1)
            used += 1
            bit_length += 1
            if used == 8:
                out.append(current)
                current = 0
                used = 0
    if used:
        out.append(current << (8 - used))
    return bytes(out), bit_length


def _read_bits(payload: bytes, bit_length: int):
    for i in range(bit_length):
        byte = payload[i // 8]
        yield (byte >> (7 - (i % 8))) & 1


class ScorerConditionalHuffmanCoder:
    """Deterministic section-level conditional Huffman prototype."""

    def __init__(self, *, prior_strength: float = 0.0) -> None:
        if prior_strength < 0.0 or not math.isfinite(prior_strength):
            raise ValueError("prior_strength must be finite and non-negative")
        self.prior_strength = float(prior_strength)

    def encode(
        self,
        payload: bytes,
        *,
        context: JSCCCodingContext,
        symbol_prior: Sequence[float] | None = None,
    ) -> JSCCEncodedPacket:
        """Encode ``payload`` with a deterministic conditional codebook."""

        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes-like")
        raw = bytes(payload)
        if not raw:
            lengths = (0,) * _ALPHABET_SIZE
            packet = (
                struct.pack(_HEADER, _MAGIC, _VERSION, 0, 0)
                + bytes(lengths)
            )
            return JSCCEncodedPacket(
                data=packet,
                context=context,
                original_size=0,
                encoded_size=len(packet),
                code_lengths=lengths,
            )

        freqs = _conditioned_frequencies(raw, symbol_prior, self.prior_strength)
        lengths = _huffman_code_lengths(freqs)
        codes = _canonical_codes(lengths)
        bits, bit_length = _write_bits(raw, codes)
        packet = (
            struct.pack(_HEADER, _MAGIC, _VERSION, len(raw), bit_length)
            + bytes(lengths)
            + bits
        )
        return JSCCEncodedPacket(
            data=packet,
            context=context,
            original_size=len(raw),
            encoded_size=len(packet),
            code_lengths=lengths,
        )

    @staticmethod
    def decode(packet: bytes) -> bytes:
        """Decode bytes emitted by :meth:`encode`."""

        if len(packet) < _HEADER_SIZE + _ALPHABET_SIZE:
            raise ValueError("JSCC packet too short")
        magic, version, original_size, bit_length = struct.unpack(
            _HEADER, packet[:_HEADER_SIZE]
        )
        if magic != _MAGIC:
            raise ValueError(f"bad JSCC magic: {magic!r}")
        if version != _VERSION:
            raise ValueError(f"unsupported JSCC version: {version}")
        lengths = tuple(packet[_HEADER_SIZE : _HEADER_SIZE + _ALPHABET_SIZE])
        payload = packet[_HEADER_SIZE + _ALPHABET_SIZE :]
        if bit_length > len(payload) * 8:
            raise ValueError("JSCC packet bit length exceeds payload")
        if original_size == 0:
            if bit_length != 0:
                raise ValueError("empty JSCC packet has nonzero bit length")
            return b""
        codes = _canonical_codes(lengths)
        decode_map = {(length, code): symbol for symbol, (code, length) in codes.items()}
        out = bytearray()
        code = 0
        length = 0
        bits_read = 0
        for bit in _read_bits(payload, bit_length):
            bits_read += 1
            code = (code << 1) | bit
            length += 1
            symbol = decode_map.get((length, code))
            if symbol is not None:
                out.append(symbol)
                if len(out) == original_size:
                    break
                code = 0
                length = 0
        if len(out) != original_size:
            raise ValueError("JSCC packet ended before original_size symbols")
        if bits_read != bit_length:
            raise ValueError("JSCC packet has trailing coded bits")
        return bytes(out)


CONFORMANCE_VECTORS: tuple[dict[str, object], ...] = (
    {
        "name": "tiny_repeated_bytes_v1",
        "payload_hex": "61616161626363",
        "section_name": "vector",
        "prior_strength": 0.0,
        "packet_sha256": "9e93c8219f4c57c60010834cf4f3e6ec62d0797f38bce3fa244d6850340614e0",
        "packet_hex": (
            "4a53434301000000000000070000000a00000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000010202000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000bc0"
        ),
    },
)
"""Pinned deterministic JSCC packet vectors for conformance tests."""


__all__ = [
    "CONFORMANCE_VECTORS",
    "LEGACY_JSCC_HUFFMAN_MAGIC",
    "JSCCCodingContext",
    "JSCCEncodedPacket",
    "JSCCSection",
    "ScorerConditionalHuffmanCoder",
    "ScorerConditionalSignal",
    "allocate_scorer_conditional_bytes",
]
