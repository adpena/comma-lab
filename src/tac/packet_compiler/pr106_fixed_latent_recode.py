"""Lossless PR106 fixed-latent recode primitives.

PR106-style HNeRV packets store the fixed 600x28 latent stream as one Brotli
blob over:

``lo:uint8[600*28] | mins:float16[28] | scales:float16[28] | hi:uint8[600*28]``

The high byte is binary by construction: q is uint8, so first-order deltas are
in [-254, 254] and zigzag(delta) fits in 9 bits.  HLM1 stores low bytes with a
deterministic Brotli choice and stores high-byte ones as a sparse delta-position
stream.  It is pure codec machinery: no scorers, no dispatch, no score
authority.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import brotli
import numpy as np

from tac.repo_io import sha256_bytes

HLM1_MAGIC = b"HLM1"
PR106_FIXED_LATENT_N = 600
PR106_FIXED_LATENT_D = 28
PR106_FIXED_LATENT_TOTAL = PR106_FIXED_LATENT_N * PR106_FIXED_LATENT_D
PR106_FIXED_LATENT_META_BYTES = PR106_FIXED_LATENT_D * 4
PR106_FIXED_LATENT_RAW_BYTES = (
    PR106_FIXED_LATENT_TOTAL
    + PR106_FIXED_LATENT_META_BYTES
    + PR106_FIXED_LATENT_TOTAL
)

DEFAULT_HLM1_BROTLI_CANDIDATES: tuple[tuple[int, int], ...] = tuple(
    (quality, lgwin)
    for quality in (11, 10, 9, 8)
    for lgwin in (24, 22, 20, 18, 16, 14, 12, 10)
)


class PR106FixedLatentRecodeError(ValueError):
    """Raised when an HLM1 latent packet is malformed or unsafe."""


@dataclass(frozen=True)
class BrotliChoice:
    """Deterministic Brotli choice for one HLM1 substream."""

    payload: bytes
    quality: int
    lgwin: int


@dataclass(frozen=True)
class PR106FixedLatentRecode:
    """Encoded HLM1 payload plus proof metadata."""

    payload: bytes
    source_brotli_bytes: int
    source_raw_bytes: int
    source_raw_sha256: str
    decoded_raw_sha256: str
    lo_brotli_bytes: int
    hi_delta_varint_bytes: int
    meta_raw_bytes: int
    packet_overhead_bytes: int
    lo_brotli_quality: int
    lo_brotli_lgwin: int
    hi_nonzero_symbols: int
    hi_nonzero_symbol_count: int

    @property
    def candidate_bytes(self) -> int:
        return len(self.payload)

    @property
    def byte_delta(self) -> int:
        return self.candidate_bytes - self.source_brotli_bytes

    @property
    def rate_positive(self) -> bool:
        return self.byte_delta < 0

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "codec": "pr106_fixed_latent_hlm1",
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "source_brotli_bytes": self.source_brotli_bytes,
            "candidate_bytes": self.candidate_bytes,
            "byte_delta": self.byte_delta,
            "rate_positive": self.rate_positive,
            "source_raw_bytes": self.source_raw_bytes,
            "source_raw_sha256": self.source_raw_sha256,
            "decoded_raw_sha256": self.decoded_raw_sha256,
            "raw_roundtrip_equal": self.source_raw_sha256 == self.decoded_raw_sha256,
            "lo_brotli_bytes": self.lo_brotli_bytes,
            "hi_delta_varint_bytes": self.hi_delta_varint_bytes,
            "meta_raw_bytes": self.meta_raw_bytes,
            "packet_overhead_bytes": self.packet_overhead_bytes,
            "lo_brotli_quality": self.lo_brotli_quality,
            "lo_brotli_lgwin": self.lo_brotli_lgwin,
            "hi_nonzero_symbols": self.hi_nonzero_symbols,
            "hi_nonzero_symbol_count": self.hi_nonzero_symbol_count,
        }


def split_pr106_fixed_latent_raw(raw: bytes) -> tuple[bytes, bytes, bytes]:
    """Split the decoded PR106 fixed latent bytes into lo/meta/hi streams."""
    if len(raw) != PR106_FIXED_LATENT_RAW_BYTES:
        raise PR106FixedLatentRecodeError(
            f"bad PR106 fixed-latent raw bytes: got {len(raw)}, "
            f"expected {PR106_FIXED_LATENT_RAW_BYTES}"
        )
    lo_end = PR106_FIXED_LATENT_TOTAL
    meta_end = lo_end + PR106_FIXED_LATENT_META_BYTES
    return raw[:lo_end], raw[lo_end:meta_end], raw[meta_end:]


def decode_pr106_fixed_latent_raw(payload: bytes) -> bytes:
    """Decode either legacy Brotli fixed latents or HLM1 fixed latents to raw."""
    if payload.startswith(HLM1_MAGIC):
        return decode_hlm1_fixed_latent_raw(payload)
    raw = brotli.decompress(payload)
    split_pr106_fixed_latent_raw(raw)
    return raw


def encode_hlm1_fixed_latents_from_brotli(
    source_brotli: bytes,
    *,
    brotli_candidates: Iterable[tuple[int, int]] = DEFAULT_HLM1_BROTLI_CANDIDATES,
) -> PR106FixedLatentRecode:
    """Encode a legacy PR106 fixed-latent Brotli section as HLM1."""
    raw = decode_pr106_fixed_latent_raw(source_brotli)
    lo, meta, hi = split_pr106_fixed_latent_raw(raw)
    lo_choice = _best_brotli(lo, brotli_candidates)
    hi_delta_varint, hi_nonzero_symbols, hi_nonzero_symbol_count = (
        _encode_sparse_hi_delta_positions(hi)
    )
    payload = _pack_hlm1(
        lo_brotli=lo_choice.payload,
        meta=meta,
        hi_delta_varint=hi_delta_varint,
        hi_nonzero_symbol_count=hi_nonzero_symbol_count,
    )
    decoded = decode_hlm1_fixed_latent_raw(payload)
    source_sha = sha256_bytes(raw)
    decoded_sha = sha256_bytes(decoded)
    if decoded_sha != source_sha:
        raise PR106FixedLatentRecodeError("HLM1 fixed-latent roundtrip mismatch")
    return PR106FixedLatentRecode(
        payload=payload,
        source_brotli_bytes=len(source_brotli),
        source_raw_bytes=len(raw),
        source_raw_sha256=source_sha,
        decoded_raw_sha256=decoded_sha,
        lo_brotli_bytes=len(lo_choice.payload),
        hi_delta_varint_bytes=len(hi_delta_varint),
        meta_raw_bytes=len(meta),
        packet_overhead_bytes=4 + 2 + 2 + 2,
        lo_brotli_quality=lo_choice.quality,
        lo_brotli_lgwin=lo_choice.lgwin,
        hi_nonzero_symbols=hi_nonzero_symbols,
        hi_nonzero_symbol_count=hi_nonzero_symbol_count,
    )


def decode_hlm1_fixed_latent_raw(payload: bytes) -> bytes:
    """Decode an HLM1 fixed-latent payload to the legacy raw byte layout."""
    if not payload.startswith(HLM1_MAGIC):
        raise PR106FixedLatentRecodeError("invalid HLM1 fixed-latent magic")
    cursor = len(HLM1_MAGIC)
    lo_len = _read_u16(payload, cursor, "lo_brotli_len")
    cursor += 2
    hi_delta_len = _read_u16(payload, cursor, "hi_delta_varint_len")
    cursor += 2
    hi_count = _read_u16(payload, cursor, "hi_nonzero_count")
    cursor += 2
    lo_brotli = _read_exact(payload, cursor, lo_len, "lo_brotli")
    cursor += lo_len
    meta = _read_exact(payload, cursor, PR106_FIXED_LATENT_META_BYTES, "meta")
    cursor += PR106_FIXED_LATENT_META_BYTES
    hi_delta_varint = _read_exact(payload, cursor, hi_delta_len, "hi_delta_varint")
    cursor += hi_delta_len
    if cursor != len(payload):
        raise PR106FixedLatentRecodeError("HLM1 fixed-latent payload has trailing bytes")

    try:
        lo = brotli.decompress(lo_brotli)
    except brotli.error as exc:
        raise PR106FixedLatentRecodeError(f"HLM1 Brotli decode failed: {exc}") from exc
    if len(lo) != PR106_FIXED_LATENT_TOTAL:
        raise PR106FixedLatentRecodeError("HLM1 lo stream length mismatch")
    hi = _decode_sparse_hi_delta_positions(hi_delta_varint, hi_count)
    raw = lo + meta + hi
    split_pr106_fixed_latent_raw(raw)
    return raw


def _pack_hlm1(
    *,
    lo_brotli: bytes,
    meta: bytes,
    hi_delta_varint: bytes,
    hi_nonzero_symbol_count: int,
) -> bytes:
    for label, value in (
        ("lo_brotli", lo_brotli),
        ("hi_delta_varint", hi_delta_varint),
    ):
        if label != "hi_delta_varint" and not value:
            raise PR106FixedLatentRecodeError(f"HLM1 {label} must be non-empty")
        if len(value) > 0xFFFF:
            raise PR106FixedLatentRecodeError(
                f"HLM1 {label} too large for u16 length: {len(value)}"
            )
    if not 0 <= int(hi_nonzero_symbol_count) <= 0xFFFF:
        raise PR106FixedLatentRecodeError(
            f"HLM1 hi_nonzero_symbol_count out of u16 range: {hi_nonzero_symbol_count}"
        )
    if len(meta) != PR106_FIXED_LATENT_META_BYTES:
        raise PR106FixedLatentRecodeError(
            f"HLM1 meta bytes mismatch: got {len(meta)}, "
            f"expected {PR106_FIXED_LATENT_META_BYTES}"
        )
    return (
        HLM1_MAGIC
        + len(lo_brotli).to_bytes(2, "little")
        + len(hi_delta_varint).to_bytes(2, "little")
        + int(hi_nonzero_symbol_count).to_bytes(2, "little")
        + lo_brotli
        + meta
        + hi_delta_varint
    )


def _best_brotli(
    payload: bytes,
    candidates: Iterable[tuple[int, int]],
) -> BrotliChoice:
    best: BrotliChoice | None = None
    seen = False
    for quality, lgwin in candidates:
        seen = True
        comp = brotli.compress(
            payload,
            mode=brotli.MODE_GENERIC,
            quality=int(quality),
            lgwin=int(lgwin),
        )
        choice = BrotliChoice(payload=comp, quality=int(quality), lgwin=int(lgwin))
        key = (len(choice.payload), choice.quality, choice.lgwin)
        if best is None or key < (len(best.payload), best.quality, best.lgwin):
            best = choice
    if not seen or best is None:
        raise PR106FixedLatentRecodeError("no Brotli candidates supplied")
    return best


def _encode_sparse_hi_delta_positions(hi: bytes) -> tuple[bytes, int, int]:
    arr = np.frombuffer(hi, dtype=np.uint8)
    if arr.size != PR106_FIXED_LATENT_TOTAL:
        raise PR106FixedLatentRecodeError("HLM1 hi stream length mismatch")
    if np.any((arr != 0) & (arr != 1)):
        raise PR106FixedLatentRecodeError(
            "HLM1 sparse high-byte encoding requires binary hi symbols"
        )
    positions = np.nonzero(arr)[0].astype(np.int64)
    if positions.size == 0:
        return b"", 1, 0
    deltas = np.diff(np.concatenate((np.array([-1], dtype=np.int64), positions)))
    out = bytearray()
    for delta in deltas:
        if delta <= 0:
            raise PR106FixedLatentRecodeError("HLM1 hi positions not strictly increasing")
        if delta <= 254:
            out.append(int(delta))
        elif delta <= 0xFFFF:
            out.append(255)
            out.extend(int(delta).to_bytes(2, "little"))
        else:
            raise PR106FixedLatentRecodeError(f"HLM1 hi delta too large: {delta}")
    return bytes(out), 2, int(positions.size)


def _decode_sparse_hi_delta_positions(payload: bytes, count: int) -> bytes:
    if count < 0:
        raise PR106FixedLatentRecodeError(f"HLM1 negative hi count: {count}")
    hi = np.zeros(PR106_FIXED_LATENT_TOTAL, dtype=np.uint8)
    cursor = 0
    pos = -1
    for _ in range(count):
        if cursor >= len(payload):
            raise PR106FixedLatentRecodeError("truncated HLM1 hi delta stream")
        marker = payload[cursor]
        cursor += 1
        if marker == 255:
            if cursor + 2 > len(payload):
                raise PR106FixedLatentRecodeError("truncated HLM1 extended hi delta")
            delta = int.from_bytes(payload[cursor : cursor + 2], "little")
            cursor += 2
            if delta <= 254:
                raise PR106FixedLatentRecodeError(
                    "non-canonical HLM1 extended hi delta <= 254"
                )
        else:
            delta = int(marker)
        if delta <= 0:
            raise PR106FixedLatentRecodeError("HLM1 hi delta must be positive")
        pos += delta
        if pos >= PR106_FIXED_LATENT_TOTAL:
            raise PR106FixedLatentRecodeError("HLM1 hi position out of range")
        hi[pos] = 1
    if cursor != len(payload):
        raise PR106FixedLatentRecodeError("HLM1 hi delta stream has trailing bytes")
    return hi.tobytes()


def _read_exact(payload: bytes, cursor: int, size: int, label: str) -> bytes:
    end = cursor + size
    if end > len(payload):
        raise PR106FixedLatentRecodeError(f"truncated HLM1 payload at {label}")
    return payload[cursor:end]


def _read_u16(payload: bytes, cursor: int, label: str) -> int:
    return int.from_bytes(_read_exact(payload, cursor, 2, label), "little")
