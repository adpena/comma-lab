# SPDX-License-Identifier: MIT
"""Deterministic QH0/QM0 record parser and serializer.

This module is intentionally narrow: it preserves the reviewed PR85/QH0 record
order and only rewrites between the two runtime-supported byte layouts:

* ``QH0``: high/low nibble split for FP4 payloads and even/odd byte split for
  fp16 scale/value tensors.
* ``QM0``: the same records in direct byte order.

It does not introduce a new runtime grammar. Any caller that wants a different
container must prove runtime support separately before dispatch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tac.qh0_renderer_codec import (
    QH0_MAGIC,
    QM0_MAGIC,
    decode_qh0_state_dict,
)
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer

SUPPORTED_OUTPUT_MAGICS = (QH0_MAGIC, QM0_MAGIC)


class QH0SerializerError(ValueError):
    """Raised when a QH0/QM0 record set is malformed or unsupported."""


@dataclass(frozen=True)
class QH0Record:
    """One serialized PR85/QH0 record in both wire layouts."""

    name: str
    category: str
    record_kind: str
    offset: int
    source_nbytes: int
    direct_record: bytes
    qh0_record: bytes
    tensor_shape: tuple[int, ...]
    element_count: int
    kind_byte: int | None


@dataclass(frozen=True)
class QH0RecordSet:
    """Parsed QH0/QM0 record set with canonical per-record byte views."""

    source_magic: str
    source_bytes: int
    source_sha256: str
    records: tuple[QH0Record, ...]


@dataclass(frozen=True)
class QH0SerializedVariant:
    """One complete serialized QH0/QM0 payload variant."""

    variant_id: str
    magic: str
    payload: bytes
    source_magic: str
    same_as_source: bool

    @property
    def name(self) -> str:
        return self.variant_id

    @property
    def payload_bytes(self) -> int:
        return len(self.payload)

    @property
    def payload_sha256(self) -> str:
        return sha256_bytes(self.payload)

    @property
    def sha256(self) -> str:
        return self.payload_sha256


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unsplit_even_odd_bytes(data: bytes) -> bytes:
    """Undo PR85's even/odd byte split."""
    raw = bytes(data)
    half = (len(raw) + 1) // 2
    out = bytearray(len(raw))
    out[0::2] = raw[:half]
    out[1::2] = raw[half:]
    return bytes(out)


def split_even_odd_bytes(data: bytes) -> bytes:
    """Apply PR85's even/odd byte split."""
    raw = bytes(data)
    return bytes(raw[0::2] + raw[1::2])


def unpack_hilo_fp4_bytes(data: bytes, packed_len: int) -> bytes:
    """Undo PR85 QH0 high/low nibble split into packed FP4 bytes."""
    raw = bytes(data)
    if packed_len % 2:
        raise QH0SerializerError(f"packed_len must be even for QH0, got {packed_len}")
    half = packed_len // 2
    if len(raw) < 2 * half:
        raise QH0SerializerError(
            f"hi/lo packed buffer truncated: have {len(raw)}, need {2 * half}"
        )
    hi_packed = np.frombuffer(raw[:half], dtype=np.uint8)
    lo_packed = np.frombuffer(raw[half : 2 * half], dtype=np.uint8)
    hi = np.empty(half * 2, dtype=np.uint8)
    lo = np.empty(half * 2, dtype=np.uint8)
    hi[0::2] = (hi_packed >> 4) & 15
    hi[1::2] = hi_packed & 15
    lo[0::2] = (lo_packed >> 4) & 15
    lo[1::2] = lo_packed & 15
    return bytes(((hi[:packed_len] << 4) | lo[:packed_len]).astype(np.uint8))


def pack_hilo_fp4_bytes(packed: bytes) -> bytes:
    """Apply PR85 QH0 high/low nibble split to packed FP4 bytes."""
    raw = bytes(packed)
    if len(raw) % 2:
        raise QH0SerializerError(
            f"FP4 packed bytes must have even length for QH0, got {len(raw)}"
        )
    src = np.frombuffer(raw, dtype=np.uint8)
    hi = (src >> 4) & 15
    lo = src & 15
    half = len(raw) // 2
    hi_packed = ((hi[0::2] << 4) | hi[1::2]).astype(np.uint8)
    lo_packed = ((lo[0::2] << 4) | lo[1::2]).astype(np.uint8)
    out = np.empty(len(raw), dtype=np.uint8)
    out[:half] = hi_packed
    out[half:] = lo_packed
    return bytes(out)


def parse_qh0_record_set(payload: bytes) -> QH0RecordSet:
    """Parse a QH0/QM0 payload into canonical record bytes without decoding tensors."""
    raw = bytes(payload)
    if len(raw) < 3:
        raise QH0SerializerError("QH0/QM0 payload is shorter than the 3-byte magic")
    magic = raw[:3]
    if magic not in SUPPORTED_OUTPUT_MAGICS:
        raise QH0SerializerError(f"unsupported QH0/QM0 magic: {magic!r}")
    hilo_split = magic == QH0_MAGIC
    pos = 3
    records: list[QH0Record] = []
    covered: set[str] = set()
    probe = build_quantizr_faithful_renderer()

    for name, module in _module_weight_order(probe):
        start = pos
        kind = _read_u8(raw, pos, f"{name}.weight kind")
        pos += 1
        shape = tuple(module.weight.shape)
        numel = int(module.weight.numel())
        if kind == 1:
            blocks = (numel + 31) // 32
            packed_len = (blocks * 32 + 1) // 2
            if hilo_split:
                qh0_packed = _slice(raw, pos, packed_len, f"{name}.weight fp4 hilo")
                pos += packed_len
                qh0_scales = _slice(raw, pos, blocks * 2, f"{name}.weight scales split")
                pos += blocks * 2
                direct_record = (
                    bytes([kind])
                    + unpack_hilo_fp4_bytes(qh0_packed, packed_len)
                    + unsplit_even_odd_bytes(qh0_scales)
                )
                qh0_record = raw[start:pos]
            else:
                packed = _slice(raw, pos, packed_len, f"{name}.weight fp4")
                pos += packed_len
                scales = _slice(raw, pos, blocks * 2, f"{name}.weight scales")
                pos += blocks * 2
                direct_record = raw[start:pos]
                qh0_record = bytes([kind]) + pack_hilo_fp4_bytes(packed) + split_even_odd_bytes(scales)
            record_kind = "fp4"
        elif kind == 0:
            nbytes = numel * 2
            data = _slice(raw, pos, nbytes, f"{name}.weight fp16")
            pos += nbytes
            direct_payload = unsplit_even_odd_bytes(data) if hilo_split else data
            qh0_payload = data if hilo_split else split_even_odd_bytes(data)
            direct_record = bytes([kind]) + direct_payload
            qh0_record = bytes([kind]) + qh0_payload
            record_kind = "fp16"
        else:
            raise QH0SerializerError(f"bad custom model q kind {kind} for {name}")
        records.append(
            _record(
                name=f"{name}.weight",
                category="module_weight",
                record_kind=record_kind,
                offset=start,
                source_nbytes=pos - start,
                direct_record=direct_record,
                qh0_record=qh0_record,
                tensor_shape=shape,
                element_count=numel,
                kind_byte=kind,
            )
        )
        covered.add(f"{name}.weight")

        if getattr(module, "bias", None) is not None:
            start = pos
            shape = tuple(module.bias.shape)
            nbytes = int(module.bias.numel()) * 2
            data = _slice(raw, pos, nbytes, f"{name}.bias fp16")
            pos += nbytes
            records.append(
                _record(
                    name=f"{name}.bias",
                    category="module_bias",
                    record_kind="fp16",
                    offset=start,
                    source_nbytes=pos - start,
                    direct_record=unsplit_even_odd_bytes(data) if hilo_split else data,
                    qh0_record=data if hilo_split else split_even_odd_bytes(data),
                    tensor_shape=shape,
                    element_count=int(module.bias.numel()),
                    kind_byte=None,
                )
            )
            covered.add(f"{name}.bias")

    for key, tensor in probe.state_dict().items():
        if key in covered:
            continue
        start = pos
        kind = _read_u8(raw, pos, f"{key} dense kind")
        pos += 1
        shape = tuple(tensor.shape)
        numel = int(tensor.numel())
        if kind == 2:
            values = _slice(raw, pos, numel, f"{key} int8 values")
            pos += numel
            rows = shape[0] if len(shape) >= 2 else 1
            scales = _slice(raw, pos, rows * 2, f"{key} int8 scales")
            pos += rows * 2
            direct_scales = unsplit_even_odd_bytes(scales) if hilo_split else scales
            qh0_scales = scales if hilo_split else split_even_odd_bytes(scales)
            direct_record = bytes([kind]) + values + direct_scales
            qh0_record = bytes([kind]) + values + qh0_scales
            record_kind = "int8"
        elif kind == 0:
            nbytes = numel * 2
            data = _slice(raw, pos, nbytes, f"{key} fp16")
            pos += nbytes
            direct_payload = unsplit_even_odd_bytes(data) if hilo_split else data
            qh0_payload = data if hilo_split else split_even_odd_bytes(data)
            direct_record = bytes([kind]) + direct_payload
            qh0_record = bytes([kind]) + qh0_payload
            record_kind = "fp16"
        else:
            raise QH0SerializerError(f"bad custom model dense kind {kind} for {key}")
        records.append(
            _record(
                name=key,
                category="dense_state",
                record_kind=record_kind,
                offset=start,
                source_nbytes=pos - start,
                direct_record=direct_record,
                qh0_record=qh0_record,
                tensor_shape=shape,
                element_count=numel,
                kind_byte=kind,
            )
        )

    if pos != len(raw):
        raise QH0SerializerError(
            f"QH0/QM0 payload has trailing bytes after parse: pos={pos} payload={len(raw)}"
        )
    return QH0RecordSet(
        source_magic=magic.decode("ascii"),
        source_bytes=len(raw),
        source_sha256=sha256_bytes(raw),
        records=tuple(records),
    )


def serialize_records(records: Sequence[QH0Record], *, magic: bytes | str = QH0_MAGIC) -> bytes:
    """Serialize parsed records as QH0 or QM0 bytes."""
    magic_bytes = _normalize_magic(magic)
    if magic_bytes == QH0_MAGIC:
        body = b"".join(record.qh0_record for record in records)
    elif magic_bytes == QM0_MAGIC:
        body = b"".join(record.direct_record for record in records)
    else:
        raise QH0SerializerError(f"unsupported output magic: {magic!r}")
    return magic_bytes + body


def build_serialized_variants(
    source: bytes | QH0RecordSet,
) -> tuple[QH0RecordSet, list[QH0SerializedVariant]]:
    """Build QH0 canonical and QM0 direct variants from a source payload."""
    record_set = parse_qh0_record_set(source) if isinstance(source, bytes) else source
    source_magic = record_set.source_magic
    variants: list[QH0SerializedVariant] = []
    for variant_id, magic in (("qh0_canonical", QH0_MAGIC), ("qm0_direct", QM0_MAGIC)):
        payload = serialize_records(record_set.records, magic=magic)
        variants.append(
            QH0SerializedVariant(
                variant_id=variant_id,
                magic=magic.decode("ascii"),
                payload=payload,
                source_magic=source_magic,
                same_as_source=(
                    magic.decode("ascii") == source_magic
                    and sha256_bytes(payload) == record_set.source_sha256
                ),
            )
        )
    return record_set, variants


def prove_decoded_tensor_parity(
    reference: bytes,
    candidate: bytes,
    *,
    device: torch.device | str = "cpu",
    atol: float = 0.0,
) -> dict[str, Any]:
    """Prove two QH0/QM0 payloads decode to identical renderer tensors."""
    reference_state, reference_report = decode_qh0_state_dict(reference, device=device)
    candidate_state, candidate_report = decode_qh0_state_dict(candidate, device=device)
    missing = sorted(set(reference_state) - set(candidate_state))
    extra = sorted(set(candidate_state) - set(reference_state))
    max_abs_diff = 0.0
    mismatched: list[str] = []
    if not missing and not extra:
        for key in sorted(reference_state):
            left = reference_state[key]
            right = candidate_state[key]
            if tuple(left.shape) != tuple(right.shape):
                mismatched.append(key)
                continue
            diff = float((left.float() - right.float()).abs().max().item()) if left.numel() else 0.0
            max_abs_diff = max(max_abs_diff, diff)
            if diff > atol:
                mismatched.append(key)
    return {
        "decoded_tensor_parity": not missing and not extra and not mismatched,
        "device": str(device),
        "atol": float(atol),
        "max_abs_diff": max_abs_diff,
        "missing_keys": missing,
        "extra_keys": extra,
        "mismatched_keys": mismatched,
        "reference_magic": reference_report.magic.decode("ascii"),
        "candidate_magic": candidate_report.magic.decode("ascii"),
        "tensor_count": len(reference_state),
    }


def record_set_summary(record_set: QH0RecordSet) -> dict[str, Any]:
    counts: dict[str, int] = {}
    bytes_by_kind: dict[str, int] = {}
    for record in record_set.records:
        counts[record.record_kind] = counts.get(record.record_kind, 0) + 1
        bytes_by_kind[record.record_kind] = bytes_by_kind.get(record.record_kind, 0) + record.source_nbytes
    return {
        "source_magic": record_set.source_magic,
        "source_bytes": record_set.source_bytes,
        "source_sha256": record_set.source_sha256,
        "record_count": len(record_set.records),
        "record_kind_counts": dict(sorted(counts.items())),
        "source_bytes_by_record_kind": dict(sorted(bytes_by_kind.items())),
    }


def choose_byte_win_candidates(
    rows: Sequence[Mapping[str, Any] | QH0SerializedVariant],
    *,
    require_runtime_compatible: bool = True,
    max_candidates: int | None = None,
) -> list[Mapping[str, Any] | QH0SerializedVariant]:
    """Keep deterministic byte-win candidates and reject no-op/runtime-blocked rows."""
    selected: list[Mapping[str, Any] | QH0SerializedVariant] = []
    if not rows:
        return selected
    if isinstance(rows[0], QH0SerializedVariant):
        variants = [row for row in rows if isinstance(row, QH0SerializedVariant)]
        if not variants:
            return selected
        source_bytes = min(variant.payload_bytes for variant in variants if variant.same_as_source)
        selected = [
            variant
            for variant in variants
            if not variant.same_as_source and variant.payload_bytes < source_bytes
        ]
        selected.sort(key=lambda variant: (variant.payload_bytes, variant.variant_id))
    else:
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            delta = int(row.get("candidate_model_delta_bytes_vs_source", 0))
            if delta >= 0:
                continue
            if row.get("decoded_tensor_parity") is False:
                continue
            runtime = row.get("runtime_compatibility")
            runtime_ok = isinstance(runtime, Mapping) and runtime.get(
                "runtime_can_decode_without_edits"
            ) is True
            if require_runtime_compatible and not runtime_ok:
                continue
            selected.append(row)
        selected.sort(
            key=lambda row: (
                int(row.get("candidate_model_delta_bytes_vs_source", 0)),
                str(row.get("candidate_id", "")),
            )
        )
    return selected[:max_candidates] if max_candidates is not None else selected


def _record(**kwargs: Any) -> QH0Record:
    direct = bytes(kwargs["direct_record"])
    qh0 = bytes(kwargs["qh0_record"])
    if not direct or not qh0:
        raise QH0SerializerError(f"{kwargs['name']} produced an empty record")
    return QH0Record(**{**kwargs, "direct_record": direct, "qh0_record": qh0})


def _module_weight_order(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    ordered: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Embedding)):
            ordered.append((name, module))
    return ordered


def _read_u8(raw: bytes, pos: int, label: str) -> int:
    _require_available(raw, pos, 1, label)
    return int(raw[pos])


def _slice(raw: bytes, pos: int, nbytes: int, label: str) -> bytes:
    _require_available(raw, pos, nbytes, label)
    return raw[pos : pos + nbytes]


def _require_available(raw: bytes, pos: int, nbytes: int, label: str) -> None:
    if nbytes < 0 or pos < 0 or pos + nbytes > len(raw):
        raise QH0SerializerError(
            f"QH0/QM0 payload truncated while reading {label}: "
            f"pos={pos} nbytes={nbytes} payload={len(raw)}"
        )


def _normalize_magic(magic: bytes | str) -> bytes:
    if isinstance(magic, str):
        magic = magic.encode("ascii")
    if magic not in SUPPORTED_OUTPUT_MAGICS:
        raise QH0SerializerError(f"unsupported output magic: {magic!r}")
    return magic


__all__ = [
    "SUPPORTED_OUTPUT_MAGICS",
    "QH0Record",
    "QH0RecordSet",
    "QH0SerializedVariant",
    "QH0SerializerError",
    "build_serialized_variants",
    "choose_byte_win_candidates",
    "pack_hilo_fp4_bytes",
    "parse_qh0_record_set",
    "prove_decoded_tensor_parity",
    "record_set_summary",
    "serialize_records",
    "sha256_bytes",
    "split_even_odd_bytes",
    "unpack_hilo_fp4_bytes",
    "unsplit_even_odd_bytes",
]
