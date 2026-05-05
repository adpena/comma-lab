"""Deterministic QH0/QM0 record parser and serializer.

This module is intentionally narrow: it preserves the reviewed PR85/QH0 record
order and only rewrites between the two runtime-supported byte layouts (QH0's
hi/lo nibble split + even/odd byte split, and QM0's direct record bytes).

REHYDRATED 2026-05-05 from .recovery_spec.json (preserved at
.recovery_quarantine_20260505T004735Z/src/tac/qh0_record_serializer.recovery_spec.json).
Spec source: bytecode disassembly of compiled .pyc; whitespace + inline comments lost.

PARTIAL REHYDRATION: dataclasses, simple bit-twiddling helpers
(``unsplit_even_odd_bytes``, ``split_even_odd_bytes``, ``pack_hilo_fp4_bytes``,
``unpack_hilo_fp4_bytes``) are reconstructed exactly. The record-set
parser/serializer state machines are deferred to ``NotImplementedError``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_QUARANTINE_SPEC = (
    ".recovery_quarantine_20260505T004735Z/src/tac/qh0_record_serializer.recovery_spec.json"
)


class QH0SerializerError(ValueError):
    """Raised when a QH0/QM0 record set is malformed or unsupported."""

    pass


@dataclass(frozen=True)
class QH0Record:
    """One QH0/QM0 record: tensor name + dtype + shape + raw bytes."""

    name: str
    kind: str  # 'fp4' | 'fp16' | 'int8'
    shape: tuple[int, ...]
    dtype: str
    block_size: int
    payload: bytes
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QH0RecordSet:
    """Parsed QH0/QM0 record set."""

    magic: bytes
    records: Sequence[QH0Record]
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QH0SerializedVariant:
    """One serialized QH0/QM0 variant: name + magic + payload bytes."""

    name: str
    magic: bytes
    payload: bytes
    sha256: str
    extra: Mapping[str, Any] = field(default_factory=dict)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unsplit_even_odd_bytes(packed: bytes) -> bytes:
    """Undo PR85's even/odd byte split."""
    n = len(packed)
    half = (n + 1) // 2
    src = np.frombuffer(packed, dtype=np.uint8)
    out = np.empty(n, dtype=np.uint8)
    out[0::2] = src[:half]
    out[1::2] = src[half:]
    return bytes(out)


def split_even_odd_bytes(data: bytes) -> bytes:
    """Apply PR85's even/odd byte split."""
    n = len(data)
    src = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(n, dtype=np.uint8)
    half = (n + 1) // 2
    out[:half] = src[0::2]
    out[half:] = src[1::2]
    return bytes(out)


def unpack_hilo_fp4_bytes(packed: bytes, packed_len: int) -> bytes:
    """Undo PR85 QH0 high/low nibble split into packed FP4 bytes."""
    if packed_len % 2:
        raise QH0SerializerError(
            f"packed_len must be even for QH0, got {packed_len}"
        )
    half = packed_len // 2
    if 2 * half > len(packed):
        raise QH0SerializerError(
            f"hi/lo packed buffer truncated: have {len(packed)}, need {2 * half}"
        )
    hi_packed = np.frombuffer(packed[:half], dtype=np.uint8)
    lo_packed = np.frombuffer(packed[half : 2 * half], dtype=np.uint8)
    hi = np.empty(half * 2, dtype=np.uint8)
    lo = np.empty(half * 2, dtype=np.uint8)
    hi[0::2] = (hi_packed >> 4) & 15
    hi[1::2] = hi_packed & 15
    lo[0::2] = (lo_packed >> 4) & 15
    lo[1::2] = lo_packed & 15
    out = ((hi[:packed_len] << 4) | lo[:packed_len]).astype(np.uint8)
    return bytes(out)


def pack_hilo_fp4_bytes(packed: bytes) -> bytes:
    """Apply PR85 QH0 high/low nibble split to packed FP4 bytes."""
    n = len(packed)
    if n % 2:
        raise QH0SerializerError(
            f"FP4 packed bytes must have even length for QH0, got {n}"
        )
    src = np.frombuffer(packed, dtype=np.uint8)
    hi = (src >> 4) & 15
    lo = src & 15
    half = n // 2
    hi_packed = ((hi[0::2] << 4) | hi[1::2]).astype(np.uint8)
    lo_packed = ((lo[0::2] << 4) | lo[1::2]).astype(np.uint8)
    out = np.empty(n, dtype=np.uint8)
    out[:half] = hi_packed
    out[half:] = lo_packed
    return bytes(out)


def _module_weight_order(model: Any) -> list[tuple[str, Any]]:
    """Return the canonical (name, module) order for QH0 record placement."""
    ordered: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Embedding)):
            continue
        ordered.append((name, module))
    return ordered


def _require(condition: bool, label: str, *args: Any) -> None:
    if not condition:
        details = ", ".join(repr(a) for a in args)
        raise QH0SerializerError(f"{label}: {details}" if details else label)


def _rehydration_failure(symbol: str) -> NotImplementedError:
    return NotImplementedError(
        f"rehydration incomplete: {symbol} contains the QH0/QM0 record-set "
        f"state machine that pycdc cannot fully decompile; original bytecode "
        f"preserved in {_QUARANTINE_SPEC}"
    )


def parse_qh0_record_set(payload: bytes) -> QH0RecordSet:
    raise _rehydration_failure("parse_qh0_record_set")


def serialize_records(
    records: Sequence[QH0Record], *, magic: bytes = b"QH0"
) -> bytes:
    raise _rehydration_failure("serialize_records")


def build_serialized_variants(
    records: Sequence[QH0Record],
) -> list[QH0SerializedVariant]:
    raise _rehydration_failure("build_serialized_variants")


def prove_decoded_tensor_parity(
    reference: QH0RecordSet, candidate: QH0RecordSet, *, atol: float = 0.0
) -> dict[str, Any]:
    raise _rehydration_failure("prove_decoded_tensor_parity")


def record_set_summary(record_set: QH0RecordSet) -> dict[str, Any]:
    raise _rehydration_failure("record_set_summary")


def choose_byte_win_candidates(
    variants: Sequence[QH0SerializedVariant], *, max_candidates: int | None = None
) -> list[QH0SerializedVariant]:
    raise _rehydration_failure("choose_byte_win_candidates")
