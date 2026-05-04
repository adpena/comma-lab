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
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

from tac.qh0_renderer_codec import (
    QH0_MAGIC,
    QM0_MAGIC,
    QH0CodecError,
    decode_qh0_state_dict,
    reconstruct_qh1_payload,
)
from tac.quantizr_faithful_renderer import JointFrameGenerator, build_quantizr_faithful_renderer


SUPPORTED_OUTPUT_MAGICS = (QH0_MAGIC, QM0_MAGIC)


@dataclass(frozen=True)
class QH0Record:
    """One runtime-ordered QH0/QM0 record with both legal byte layouts."""

    name: str
    category: str
    record_kind: str
    offset: int
    source_nbytes: int
    direct_record: bytes
    qh0_record: bytes
    tensor_shape: tuple[int, ...]
    element_count: int
    kind_byte: int | None = None


@dataclass(frozen=True)
class QH0RecordSet:
    """Parsed QH0/QM0 payload and deterministic serializer inputs."""

    source_magic: str
    source_bytes: int
    source_sha256: str
    records: tuple[QH0Record, ...]
    tensor_count: int


@dataclass(frozen=True)
class QH0SerializedVariant:
    """Serialized model payload candidate."""

    variant_id: str
    magic: str
    payload: bytes
    payload_bytes: int
    payload_sha256: str
    same_as_source: bool


class QH0SerializerError(ValueError):
    """Raised when a QH0/QM0 record stream cannot be serialized safely."""


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest for ``data``."""

    return hashlib.sha256(data).hexdigest()


def unsplit_even_odd_bytes(data: bytes) -> bytes:
    """Undo QH0 even/odd byte split used for fp16 payloads."""

    raw = bytes(data)
    half = (len(raw) + 1) // 2
    out = bytearray(len(raw))
    out[0::2] = raw[:half]
    out[1::2] = raw[half:]
    return bytes(out)


def split_even_odd_bytes(data: bytes) -> bytes:
    """Apply QH0 even/odd byte split used for fp16 payloads."""

    raw = bytes(data)
    return raw[0::2] + raw[1::2]


def unpack_hilo_fp4_bytes(data: bytes, packed_len: int) -> bytes:
    """Undo QH0 high/low nibble split into direct packed FP4 bytes."""

    raw = bytes(data)
    if packed_len < 0 or packed_len % 2:
        raise QH0SerializerError(f"packed_len must be non-negative and even, got {packed_len}")
    if len(raw) != packed_len:
        raise QH0SerializerError(
            f"hilo FP4 slice length mismatch: got={len(raw)} expected={packed_len}"
        )
    half = packed_len // 2
    hi_packed = raw[:half]
    lo_packed = raw[half:]
    hi = bytearray(packed_len)
    lo = bytearray(packed_len)
    for idx, value in enumerate(hi_packed):
        hi[2 * idx] = (value >> 4) & 0x0F
        hi[2 * idx + 1] = value & 0x0F
    for idx, value in enumerate(lo_packed):
        lo[2 * idx] = (value >> 4) & 0x0F
        lo[2 * idx + 1] = value & 0x0F
    return bytes(((hi[idx] << 4) | lo[idx]) & 0xFF for idx in range(packed_len))


def pack_hilo_fp4_bytes(packed: bytes) -> bytes:
    """Apply QH0 high/low nibble split to direct packed FP4 bytes."""

    raw = bytes(packed)
    if len(raw) % 2:
        raise QH0SerializerError(f"packed FP4 bytes must be even, got {len(raw)}")
    hi = [(value >> 4) & 0x0F for value in raw]
    lo = [value & 0x0F for value in raw]
    hi_packed = bytes(((hi[idx] << 4) | hi[idx + 1]) & 0xFF for idx in range(0, len(hi), 2))
    lo_packed = bytes(((lo[idx] << 4) | lo[idx + 1]) & 0xFF for idx in range(0, len(lo), 2))
    return hi_packed + lo_packed


def _module_weight_order(model: JointFrameGenerator) -> list[tuple[str, torch.nn.Module]]:
    ordered: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Embedding)):
            ordered.append((name, module))
    return ordered


def _require(raw: bytes, pos: int, nbytes: int, label: str) -> bytes:
    if pos < 0 or nbytes < 0 or pos + nbytes > len(raw):
        raise QH0SerializerError(
            f"QH0 record stream truncated while reading {label}: "
            f"pos={pos} nbytes={nbytes} payload={len(raw)}"
        )
    return raw[pos : pos + nbytes]


def _fp16_layout_bytes(raw: bytes, pos: int, nbytes: int, *, hilo_split: bool, label: str) -> tuple[bytes, bytes, int]:
    source = _require(raw, pos, nbytes, label)
    direct = unsplit_even_odd_bytes(source) if hilo_split else source
    qh0 = split_even_odd_bytes(direct)
    return direct, qh0, pos + nbytes


def parse_qh0_record_set(payload: bytes | bytearray | memoryview) -> QH0RecordSet:
    """Parse QH0/QM0 bytes into deterministic runtime records."""

    raw = reconstruct_qh1_payload(payload)
    if len(raw) < 3:
        raise QH0SerializerError("QH0/QM0 payload is shorter than the 3-byte magic")
    magic = raw[:3]
    if magic not in SUPPORTED_OUTPUT_MAGICS:
        raise QH0SerializerError(f"unsupported QH0 serializer magic: {magic!r}")
    hilo_split = magic == QH0_MAGIC
    model = build_quantizr_faithful_renderer()
    pos = 3
    records: list[QH0Record] = []
    covered: set[str] = set()

    for module_name, module in _module_weight_order(model):
        start = pos
        kind = int(_require(raw, pos, 1, f"{module_name}.weight.kind")[0])
        pos += 1
        shape = tuple(int(x) for x in module.weight.shape)
        numel = int(module.weight.numel())
        if kind == 1:
            block_size = 32
            blocks = (numel + block_size - 1) // block_size
            packed_len = (blocks * block_size + 1) // 2
            if hilo_split:
                packed_source = _require(raw, pos, packed_len, f"{module_name}.weight.hilo")
                packed = unpack_hilo_fp4_bytes(packed_source, packed_len)
                pos += packed_len
                scales_direct, scales_qh0, pos = _fp16_layout_bytes(
                    raw,
                    pos,
                    blocks * 2,
                    hilo_split=True,
                    label=f"{module_name}.weight.scales",
                )
            else:
                packed = _require(raw, pos, packed_len, f"{module_name}.weight.packed")
                pos += packed_len
                scales_direct, scales_qh0, pos = _fp16_layout_bytes(
                    raw,
                    pos,
                    blocks * 2,
                    hilo_split=False,
                    label=f"{module_name}.weight.scales",
                )
            direct_record = bytes([kind]) + packed + scales_direct
            qh0_record = bytes([kind]) + pack_hilo_fp4_bytes(packed) + scales_qh0
            record_kind = "fp4"
        elif kind == 0:
            direct, qh0, pos = _fp16_layout_bytes(
                raw,
                pos,
                numel * 2,
                hilo_split=hilo_split,
                label=f"{module_name}.weight.fp16",
            )
            direct_record = bytes([kind]) + direct
            qh0_record = bytes([kind]) + qh0
            record_kind = "fp16"
        else:
            raise QH0SerializerError(f"bad QH0 weight kind {kind} for {module_name}.weight")
        records.append(
            _make_record(
                name=f"{module_name}.weight",
                category="module_weight",
                record_kind=record_kind,
                raw=raw,
                start=start,
                end=pos,
                direct_record=direct_record,
                qh0_record=qh0_record,
                tensor_shape=shape,
                element_count=numel,
                kind_byte=kind,
                source_magic=magic,
            )
        )
        covered.add(f"{module_name}.weight")

        bias = getattr(module, "bias", None)
        if bias is not None:
            start = pos
            bias_shape = tuple(int(x) for x in bias.shape)
            bias_numel = int(bias.numel())
            direct, qh0, pos = _fp16_layout_bytes(
                raw,
                pos,
                bias_numel * 2,
                hilo_split=hilo_split,
                label=f"{module_name}.bias",
            )
            records.append(
                _make_record(
                    name=f"{module_name}.bias",
                    category="module_bias",
                    record_kind="fp16_bias",
                    raw=raw,
                    start=start,
                    end=pos,
                    direct_record=direct,
                    qh0_record=qh0,
                    tensor_shape=bias_shape,
                    element_count=bias_numel,
                    kind_byte=None,
                    source_magic=magic,
                )
            )
            covered.add(f"{module_name}.bias")

    for key, tensor in model.state_dict().items():
        if key in covered:
            continue
        start = pos
        kind = int(_require(raw, pos, 1, f"{key}.kind")[0])
        pos += 1
        shape = tuple(int(x) for x in tensor.shape)
        numel = int(tensor.numel())
        if kind == 2:
            q_bytes = _require(raw, pos, numel, f"{key}.q_int8")
            pos += numel
            rows = shape[0] if len(shape) >= 2 else 1
            scales_direct, scales_qh0, pos = _fp16_layout_bytes(
                raw,
                pos,
                rows * 2,
                hilo_split=hilo_split,
                label=f"{key}.scales",
            )
            direct_record = bytes([kind]) + q_bytes + scales_direct
            qh0_record = bytes([kind]) + q_bytes + scales_qh0
            record_kind = "int8_row_scale"
        elif kind == 0:
            direct, qh0, pos = _fp16_layout_bytes(
                raw,
                pos,
                numel * 2,
                hilo_split=hilo_split,
                label=f"{key}.fp16",
            )
            direct_record = bytes([kind]) + direct
            qh0_record = bytes([kind]) + qh0
            record_kind = "fp16_dense"
        else:
            raise QH0SerializerError(f"bad QH0 dense kind {kind} for {key}")
        records.append(
            _make_record(
                name=key,
                category="dense_tensor",
                record_kind=record_kind,
                raw=raw,
                start=start,
                end=pos,
                direct_record=direct_record,
                qh0_record=qh0_record,
                tensor_shape=shape,
                element_count=numel,
                kind_byte=kind,
                source_magic=magic,
            )
        )

    if pos != len(raw):
        raise QH0SerializerError(f"QH0/QM0 payload has trailing bytes: consumed={pos} total={len(raw)}")
    return QH0RecordSet(
        source_magic=magic.decode("ascii"),
        source_bytes=len(raw),
        source_sha256=sha256_bytes(raw),
        records=tuple(records),
        tensor_count=len(records),
    )


def _make_record(
    *,
    name: str,
    category: str,
    record_kind: str,
    raw: bytes,
    start: int,
    end: int,
    direct_record: bytes,
    qh0_record: bytes,
    tensor_shape: tuple[int, ...],
    element_count: int,
    kind_byte: int | None,
    source_magic: bytes,
) -> QH0Record:
    source_record = raw[start:end]
    expected = qh0_record if source_magic == QH0_MAGIC else direct_record
    if source_record != expected:
        raise QH0SerializerError(
            f"internal serializer mismatch for {name}: source layout does not round-trip"
        )
    return QH0Record(
        name=name,
        category=category,
        record_kind=record_kind,
        offset=start,
        source_nbytes=end - start,
        direct_record=bytes(direct_record),
        qh0_record=bytes(qh0_record),
        tensor_shape=tensor_shape,
        element_count=element_count,
        kind_byte=kind_byte,
    )


def serialize_records(records: Sequence[QH0Record], *, magic: bytes) -> bytes:
    """Serialize records as ``QH0`` or ``QM0`` bytes."""

    if magic == QH0_MAGIC:
        return QH0_MAGIC + b"".join(record.qh0_record for record in records)
    if magic == QM0_MAGIC:
        return QM0_MAGIC + b"".join(record.direct_record for record in records)
    raise QH0SerializerError(f"unsupported serializer output magic: {magic!r}")


def build_serialized_variants(payload: bytes | bytearray | memoryview) -> tuple[QH0RecordSet, tuple[QH0SerializedVariant, ...]]:
    """Build deterministic ``QH0`` and ``QM0`` payload variants."""

    raw = reconstruct_qh1_payload(payload)
    record_set = parse_qh0_record_set(raw)
    variants: list[QH0SerializedVariant] = []
    for variant_id, magic in (("qh0_canonical", QH0_MAGIC), ("qm0_direct", QM0_MAGIC)):
        encoded = serialize_records(record_set.records, magic=magic)
        variants.append(
            QH0SerializedVariant(
                variant_id=variant_id,
                magic=magic.decode("ascii"),
                payload=encoded,
                payload_bytes=len(encoded),
                payload_sha256=sha256_bytes(encoded),
                same_as_source=encoded == raw,
            )
        )
    return record_set, tuple(variants)


def prove_decoded_tensor_parity(
    source_payload: bytes,
    candidate_payload: bytes,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Decode two payloads through the reviewed loader and prove tensor equality."""

    source_state, source_report = decode_qh0_state_dict(source_payload, device=device)
    candidate_state, candidate_report = decode_qh0_state_dict(candidate_payload, device=device)
    mismatches: list[dict[str, Any]] = []
    if set(source_state) != set(candidate_state):
        missing = sorted(set(source_state) - set(candidate_state))
        extra = sorted(set(candidate_state) - set(source_state))
        mismatches.append({"kind": "key_set", "missing": missing, "extra": extra})
    for key in sorted(set(source_state) & set(candidate_state)):
        left = source_state[key]
        right = candidate_state[key]
        if tuple(left.shape) != tuple(right.shape):
            mismatches.append(
                {
                    "kind": "shape",
                    "name": key,
                    "source_shape": list(left.shape),
                    "candidate_shape": list(right.shape),
                }
            )
            continue
        if left.dtype != right.dtype:
            mismatches.append(
                {
                    "kind": "dtype",
                    "name": key,
                    "source_dtype": str(left.dtype),
                    "candidate_dtype": str(right.dtype),
                }
            )
            continue
        if not torch.equal(left.cpu(), right.cpu()):
            diff = (left.cpu() - right.cpu()).abs()
            mismatches.append(
                {
                    "kind": "value",
                    "name": key,
                    "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
                    "changed_elements": int((diff != 0).sum().item()),
                }
            )
            if len(mismatches) >= 8:
                break
    return {
        "decoded_tensor_parity": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "source_report": source_report.__dict__,
        "candidate_report": candidate_report.__dict__,
        "source_tensor_count": len(source_state),
        "candidate_tensor_count": len(candidate_state),
    }


def record_set_summary(record_set: QH0RecordSet) -> dict[str, Any]:
    """Return byte accounting for a parsed record set."""

    by_category: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for record in record_set.records:
        by_category[record.category] = by_category.get(record.category, 0) + record.source_nbytes
        by_kind[record.record_kind] = by_kind.get(record.record_kind, 0) + record.source_nbytes
    return {
        "source_magic": record_set.source_magic,
        "source_bytes": record_set.source_bytes,
        "source_sha256": record_set.source_sha256,
        "record_count": len(record_set.records),
        "records_total_bytes": sum(record.source_nbytes for record in record_set.records),
        "magic_bytes": 3,
        "records_plus_magic_bytes": 3 + sum(record.source_nbytes for record in record_set.records),
        "by_category": dict(sorted(by_category.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "top_records_by_bytes": [
            {
                "name": record.name,
                "category": record.category,
                "record_kind": record.record_kind,
                "offset": record.offset,
                "bytes": record.source_nbytes,
                "tensor_shape": list(record.tensor_shape),
                "element_count": record.element_count,
            }
            for record in sorted(record_set.records, key=lambda item: item.source_nbytes, reverse=True)[:16]
        ],
    }


def choose_byte_win_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    require_runtime_compatible: bool = True,
) -> list[Mapping[str, Any]]:
    """Filter candidate rows to real byte wins that are runtime compatible."""

    out: list[Mapping[str, Any]] = []
    for candidate in candidates:
        model_delta = int(candidate.get("candidate_model_delta_bytes_vs_source", 0))
        runtime = candidate.get("runtime_compatibility", {})
        compatible = bool(runtime.get("runtime_can_decode_without_edits", False)) if isinstance(runtime, Mapping) else False
        if model_delta < 0 and (compatible or not require_runtime_compatible):
            out.append(candidate)
    return out
