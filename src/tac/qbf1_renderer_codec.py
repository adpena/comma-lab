# SPDX-License-Identifier: MIT
"""Prototype QBF1 block-FP container for JointFrameGenerator state dicts.

QBF1 is a readiness artifact for an archive-local renderer payload, not a
score-bearing format. It stores a PyTorch ``state_dict`` as deterministic
bytes without pickle or ``torch.load``:

* 4-byte magic ``b"QBF1"`` plus an explicit versioned binary header.
* Canonical JSON metadata with tensor names, shapes, dtypes, block sizes,
  byte counts, and payload offsets.
* Binary per-block float32 scales followed by int8 quantized tensor values.

The parser is intentionally strict. It validates all lengths, offsets, schema
keys, tensor ordering, duplicate names, scale finiteness, and zero-scale block
payloads before exposing records to the decoder.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

QBF1_MAGIC: bytes = b"QBF1"
QBF1_VERSION: int = 1
QBF1_SCHEMA: str = "qbf1.renderer.block_fp.v1"
QBF1_V2_BYTE_PROFILE_SCHEMA: str = "qbf1.renderer.v2_byte_profile.v1"
QBF1_V2_SELF_DESCRIBING_PROFILE_SCHEMA: str = (
    "qbf1.renderer.self_describing_fp4_f16_qv.profile.v1"
)
QBF1_DEFAULT_BLOCK_SIZE: int = 64
QBF1_V2_DEFAULT_BLOCK_SIZES: tuple[int, ...] = (
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    512,
    1024,
    4096,
)
QBF1_V2_REFERENCE_QZS3_NBYTES: int = 59_288
QBF1_QMAX: int = 127

_HEADER_STRUCT = struct.Struct("<4sHHIQ")
_HEADER_NBYTES = _HEADER_STRUCT.size
_FLAGS_NONE = 0
_SCALE_STRUCT = struct.Struct("<f")

_TOP_LEVEL_KEYS = {
    "schema",
    "version",
    "tensor_count",
    "block_fp",
    "byte_accounting",
    "tensors",
}
_BLOCK_FP_KEYS = {"quant_dtype", "scale_dtype", "qmax"}
_BYTE_ACCOUNTING_KEYS = {
    "header_nbytes",
    "metadata_nbytes",
    "payload_nbytes",
    "tensor_payload_nbytes",
    "total_nbytes",
}
_TENSOR_KEYS = {
    "name",
    "shape",
    "dtype",
    "block_size",
    "value_count",
    "num_blocks",
    "quant_dtype",
    "scale_dtype",
    "original_nbytes",
    "scale_nbytes",
    "payload_nbytes",
    "encoded_nbytes",
    "scale_offset",
    "payload_offset",
}
_DTYPE_NBYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}


class QBF1CodecError(ValueError):
    """Raised when a QBF1 payload is malformed or unsupported."""


@dataclass(frozen=True)
class QBF1ByteAccounting:
    """Byte totals for one parsed QBF1 payload."""

    header_nbytes: int
    metadata_nbytes: int
    payload_nbytes: int
    tensor_payload_nbytes: int
    total_nbytes: int


@dataclass(frozen=True)
class QBF1TensorMetadata:
    """Validated per-tensor metadata for a QBF1 record.

    ``scales`` is populated from the binary payload during unpack. The JSON
    metadata records the scale byte range; keeping the values binary avoids a
    text-format dependency for the actual quantization stream.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    block_size: int
    value_count: int
    num_blocks: int
    quant_dtype: str
    scale_dtype: str
    scales: tuple[float, ...]
    original_nbytes: int
    scale_nbytes: int
    payload_nbytes: int
    encoded_nbytes: int
    scale_offset: int
    payload_offset: int


@dataclass(frozen=True)
class QBF1TensorRecord:
    """One tensor's validated metadata plus int8 quantized bytes."""

    metadata: QBF1TensorMetadata
    quantized: bytes


@dataclass(frozen=True)
class QBF1Container:
    """Parsed QBF1 container returned by :func:`unpack_qbf1_container`."""

    tensors: tuple[QBF1TensorRecord, ...]
    byte_accounting: QBF1ByteAccounting
    raw_metadata: Mapping[str, Any]

    def tensor_names(self) -> tuple[str, ...]:
        """Return tensor names in deterministic on-wire order."""

        return tuple(record.metadata.name for record in self.tensors)

    def metadata_by_name(self) -> dict[str, QBF1TensorMetadata]:
        """Return per-tensor metadata keyed by state_dict name."""

        return {record.metadata.name: record.metadata for record in self.tensors}


def pack_qbf1_state_dict(
    state_dict: Mapping[str, Any],
    *,
    block_size: int = QBF1_DEFAULT_BLOCK_SIZE,
) -> bytes:
    """Encode a JointFrameGenerator-compatible state_dict as QBF1 bytes.

    Args:
        state_dict: Mapping from state_dict key to floating tensor. Keys are
            sorted lexicographically before encoding to make output bytes
            deterministic independent of caller insertion order.
        block_size: Number of flattened values sharing one float32 scale.

    Returns:
        A self-describing QBF1 byte container.

    Raises:
        QBF1CodecError: if a tensor name, dtype, value, or block size cannot
            be represented by the prototype format.
    """

    torch = _require_torch()
    if block_size <= 0:
        raise QBF1CodecError(f"block_size must be > 0, got {block_size}")

    payload_parts: list[bytes] = []
    tensor_entries: list[dict[str, Any]] = []
    payload_cursor = 0

    for name in sorted(state_dict):
        _validate_tensor_name(name)
        tensor = state_dict[name]
        if not hasattr(tensor, "detach"):
            raise QBF1CodecError(f"state_dict entry {name!r} is not tensor-like")

        dtype = _normalise_torch_dtype_name(str(tensor.dtype))
        if dtype not in _DTYPE_NBYTES:
            raise QBF1CodecError(
                f"state_dict entry {name!r} has unsupported dtype {tensor.dtype}"
            )

        shape = tuple(int(dim) for dim in tensor.shape)
        value_count = _shape_value_count(shape)
        flat = tensor.detach().cpu().to(torch.float32).contiguous().view(-1)
        if value_count != int(flat.numel()):
            raise QBF1CodecError(f"shape/value count mismatch for tensor {name!r}")
        if value_count and not bool(torch.isfinite(flat).all().item()):
            raise QBF1CodecError(f"state_dict entry {name!r} contains non-finite values")

        scales, quantized = _quantize_flat_tensor(flat, block_size=block_size)
        scale_bytes = _pack_scales(scales)
        scale_offset = payload_cursor
        payload_parts.append(scale_bytes)
        payload_cursor += len(scale_bytes)
        payload_offset = payload_cursor
        payload_parts.append(quantized)
        payload_cursor += len(quantized)

        scale_nbytes = len(scale_bytes)
        payload_nbytes = len(quantized)
        tensor_entries.append(
            {
                "name": name,
                "shape": list(shape),
                "dtype": dtype,
                "block_size": int(block_size),
                "value_count": int(value_count),
                "num_blocks": len(scales),
                "quant_dtype": "int8",
                "scale_dtype": "float32",
                "original_nbytes": int(value_count * _DTYPE_NBYTES[dtype]),
                "scale_nbytes": int(scale_nbytes),
                "payload_nbytes": int(payload_nbytes),
                "encoded_nbytes": int(scale_nbytes + payload_nbytes),
                "scale_offset": int(scale_offset),
                "payload_offset": int(payload_offset),
            }
        )

    payload = b"".join(payload_parts)
    metadata_dict = _build_metadata(tensor_entries, payload_nbytes=len(payload))
    metadata_bytes = _canonical_json_bytes(metadata_dict)
    header = _HEADER_STRUCT.pack(
        QBF1_MAGIC,
        QBF1_VERSION,
        _FLAGS_NONE,
        len(metadata_bytes),
        len(payload),
    )
    return header + metadata_bytes + payload


def unpack_qbf1_container(blob: bytes | bytearray | memoryview) -> QBF1Container:
    """Parse QBF1 bytes without using pickle, ``torch.load``, or tensor code.

    The returned records contain binary int8 payloads and parsed float scales.
    Use :func:`decode_qbf1_state_dict` when reconstructed torch tensors are
    needed.
    """

    data = bytes(blob)
    if len(data) < _HEADER_NBYTES:
        raise QBF1CodecError("QBF1 payload is truncated before header")

    magic, version, flags, metadata_nbytes, payload_nbytes = _HEADER_STRUCT.unpack_from(data)
    if magic != QBF1_MAGIC:
        raise QBF1CodecError(f"bad QBF1 magic: got {magic!r}, expected {QBF1_MAGIC!r}")
    if version != QBF1_VERSION:
        raise QBF1CodecError(
            f"unsupported QBF1 version {version}, expected {QBF1_VERSION}"
        )
    if flags != _FLAGS_NONE:
        raise QBF1CodecError(f"unsupported QBF1 flags {flags}")
    if metadata_nbytes <= 0:
        raise QBF1CodecError("QBF1 metadata length must be positive")

    metadata_start = _HEADER_NBYTES
    metadata_end = metadata_start + metadata_nbytes
    payload_start = metadata_end
    payload_end = payload_start + payload_nbytes
    if metadata_end > len(data):
        raise QBF1CodecError("QBF1 payload is truncated in metadata")
    if payload_end != len(data):
        raise QBF1CodecError(
            "QBF1 payload length mismatch: "
            f"header expects {payload_end} total bytes, got {len(data)}"
        )

    metadata = _load_metadata(data[metadata_start:metadata_end])
    payload = data[payload_start:payload_end]
    accounting = _validate_byte_accounting(
        metadata,
        header_nbytes=_HEADER_NBYTES,
        metadata_nbytes=metadata_nbytes,
        payload_nbytes=payload_nbytes,
        total_nbytes=len(data),
    )
    records = _parse_tensor_records(metadata, payload)
    return QBF1Container(
        tensors=records,
        byte_accounting=accounting,
        raw_metadata=metadata,
    )


def decode_qbf1_state_dict(
    blob: bytes | bytearray | memoryview,
    *,
    device: Any | None = None,
) -> dict[str, Any]:
    """Decode QBF1 bytes into a torch state_dict.

    This helper imports torch lazily and reconstructs tensors from the QBF1
    int8/scale stream. It never calls pickle or ``torch.load``.
    """

    torch = _require_torch()
    container = unpack_qbf1_container(blob)
    decoded: dict[str, Any] = {}
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    target_device = torch.device("cpu" if device is None else device)

    for record in container.tensors:
        meta = record.metadata
        quant_values = [
            byte - 256 if byte >= 128 else byte
            for byte in record.quantized
        ]
        flat_q = torch.tensor(quant_values, dtype=torch.float32)
        out = torch.empty((meta.value_count,), dtype=torch.float32)
        cursor = 0
        for block_index, scale in enumerate(meta.scales):
            start = block_index * meta.block_size
            stop = min(start + meta.block_size, meta.value_count)
            if stop <= start:
                continue
            if scale == 0.0:
                out[start:stop] = 0.0
            else:
                out[start:stop] = flat_q[start:stop] * float(scale)
            cursor = stop
        if cursor != meta.value_count:
            raise QBF1CodecError(
                f"decoder did not cover all values for tensor {meta.name!r}"
            )
        decoded[meta.name] = out.reshape(meta.shape).to(
            dtype=dtype_map[meta.dtype],
            device=target_device,
        )
    return decoded


def qbf1_byte_accounting(blob: bytes | bytearray | memoryview) -> QBF1ByteAccounting:
    """Return validated byte accounting for a QBF1 payload."""

    return unpack_qbf1_container(blob).byte_accounting


def profile_qbf1_v2_renderer_bytes(
    model_or_state: Any | None = None,
    *,
    block_sizes: tuple[int, ...] = QBF1_V2_DEFAULT_BLOCK_SIZES,
    reference_qzs3_nbytes: int | None = None,
    reference_block_size: int = 32,
) -> dict[str, Any]:
    """Profile local QBF1-v2 renderer byte feasibility without score claims.

    This is a byte/readiness screen, not a new score-bearing wire format.  It
    compares three local families:

    * implemented QBF1-v1 int8 + float32-scale payloads;
    * a compact self-describing QBF1-v2 FP4/fp16/QV planning model; and
    * existing QZS3 block-size policies, included only as a reference because
      they already have a pickle-free runtime loader.

    The returned JSON-compatible dict is empirical evidence only.  It never
    promotes, dispatches, or claims score.
    """

    block_sizes = _normalise_block_sizes(block_sizes)
    state = _coerce_renderer_state_dict(model_or_state)

    from tac.quantizr_qzs3_codec import encode_qzs3_state_dict

    reference_payload = encode_qzs3_state_dict(state, block_size=reference_block_size)
    actual_reference_nbytes = len(reference_payload)
    if reference_qzs3_nbytes is None:
        reference_qzs3_nbytes = actual_reference_nbytes
    if reference_qzs3_nbytes <= 0:
        raise QBF1CodecError("reference_qzs3_nbytes must be positive")

    qbf1_v1_candidates = [
        _profile_qbf1_v1_variant(
            state,
            block_size=block_size,
            reference_nbytes=reference_qzs3_nbytes,
        )
        for block_size in block_sizes
    ]
    qbf1_v2_candidates = [
        _profile_qbf1_v2_self_describing_variant(
            state,
            block_size=block_size,
            reference_nbytes=reference_qzs3_nbytes,
        )
        for block_size in block_sizes
    ]
    qzs3_reference_candidates = [
        _profile_existing_qzs3_variant(
            state,
            block_size=block_size,
            reference_nbytes=reference_qzs3_nbytes,
        )
        for block_size in block_sizes
    ]

    best_qbf1_v1 = min(qbf1_v1_candidates, key=_raw_then_deflate_key)
    best_qbf1_v2 = min(qbf1_v2_candidates, key=_raw_then_deflate_key)
    best_qzs3_policy = min(qzs3_reference_candidates, key=_raw_then_deflate_key)
    qbf1_v2_go = bool(
        best_qbf1_v2["beats_reference_raw"]
        and best_qbf1_v2["strict_pickle_free_loader_ready"]
    )

    no_go_reasons: list[str] = []
    if not best_qbf1_v1["beats_reference_raw"]:
        no_go_reasons.append(
            "implemented QBF1-v1 best raw bytes "
            f"{best_qbf1_v1['raw_nbytes']} exceed reference "
            f"{reference_qzs3_nbytes} by {best_qbf1_v1['raw_delta_vs_reference']}"
        )
    if not best_qbf1_v2["beats_reference_raw"]:
        no_go_reasons.append(
            "self-describing QBF1-v2 byte model best raw bytes "
            f"{best_qbf1_v2['raw_nbytes']} exceed reference "
            f"{reference_qzs3_nbytes} by {best_qbf1_v2['raw_delta_vs_reference']}"
        )
    if not best_qbf1_v2["strict_pickle_free_loader_ready"]:
        no_go_reasons.append(
            "QBF1-v2 FP4/fp16/QV layout is profiling-only; no reviewed "
            "decoder or inflate dispatch exists"
        )
    if best_qzs3_policy["beats_reference_raw"]:
        no_go_reasons.append(
            "the only local raw byte win in this screen is an existing QZS3 "
            f"block policy ({best_qzs3_policy['variant_id']}), not a QBF1-v2 "
            "wire-format improvement"
        )

    return {
        "schema": QBF1_V2_BYTE_PROFILE_SCHEMA,
        "evidence_grade": "empirical",
        "score_claim": False,
        "promotion_eligible": False,
        "remote_or_gpu_dispatch": False,
        "reference": {
            "name": f"current_qzs3_b{reference_block_size}",
            "wire_format": "QZS3",
            "reference_block_size": int(reference_block_size),
            "raw_nbytes": int(reference_qzs3_nbytes),
            "actual_local_raw_nbytes": int(actual_reference_nbytes),
            "matches_default_c067_qzs3_reference": (
                int(reference_qzs3_nbytes) == QBF1_V2_REFERENCE_QZS3_NBYTES
            ),
        },
        "block_sizes": list(block_sizes),
        "candidate_families": {
            "qbf1_v1_int8_f32_scales": qbf1_v1_candidates,
            "qbf1_v2_self_describing_fp4_f16_qv": qbf1_v2_candidates,
            "existing_qzs3_block_policy_reference": qzs3_reference_candidates,
        },
        "best": {
            "qbf1_v1": best_qbf1_v1,
            "qbf1_v2_self_describing": best_qbf1_v2,
            "existing_qzs3_block_policy": best_qzs3_policy,
        },
        "readiness": {
            "qbf1_v2_go": qbf1_v2_go,
            "qbf1_v2_dispatchable": False,
            "score_claim": False,
            "promotion_eligible": False,
            "no_go_reasons": no_go_reasons,
            "next_gate": (
                "Only implement a QBF1-v2 decoder/inflate path if a "
                "self-describing layout beats the current QZS3 renderer "
                "slice locally before exact CUDA auth eval is considered."
            ),
        },
    }


def pack_qbf1_renderer_state_dict(
    state_dict: Mapping[str, Any],
    *,
    block_size: int = QBF1_DEFAULT_BLOCK_SIZE,
) -> bytes:
    """Alias for :func:`pack_qbf1_state_dict` with renderer-oriented naming."""

    return pack_qbf1_state_dict(state_dict, block_size=block_size)


def unpack_qbf1_renderer_container(
    blob: bytes | bytearray | memoryview,
) -> QBF1Container:
    """Alias for :func:`unpack_qbf1_container` with renderer-oriented naming."""

    return unpack_qbf1_container(blob)


def decode_qbf1_renderer_state_dict(
    blob: bytes | bytearray | memoryview,
    *,
    device: Any | None = None,
) -> dict[str, Any]:
    """Alias for :func:`decode_qbf1_state_dict` with renderer-oriented naming."""

    return decode_qbf1_state_dict(blob, device=device)


def load_qbf1(
    path: str | bytes | bytearray | memoryview | Any,
    *,
    device: Any | None = None,
) -> Any:
    """Load QBF1 bytes into a Quantizr-faithful JointFrameGenerator.

    The loader is intentionally narrow: QBF1 is a JointFrameGenerator renderer
    container, not a generic checkpoint format. It never calls ``torch.load``.
    """

    blob = _read_payload_input(path)
    state = decode_qbf1_renderer_state_dict(blob, device=device)
    from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer

    target_device = _require_torch().device("cpu" if device is None else device)
    model = build_quantizr_faithful_renderer()
    model.load_state_dict(state, strict=True)
    model.to(target_device).eval()
    return model


def load_qbf1_renderer(
    path: str | bytes | bytearray | memoryview | Any,
    *,
    device: Any | None = None,
) -> Any:
    """Alias for :func:`load_qbf1` with renderer-oriented naming."""

    return load_qbf1(path, device=device)


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a project dependency.
        raise QBF1CodecError("QBF1 tensor encode/decode requires torch") from exc
    return torch


def _coerce_renderer_state_dict(model_or_state: Any | None) -> Mapping[str, Any]:
    if model_or_state is None:
        from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer

        return build_quantizr_faithful_renderer().eval().state_dict()
    if isinstance(model_or_state, Mapping):
        return model_or_state
    if hasattr(model_or_state, "state_dict"):
        return model_or_state.state_dict()
    raise QBF1CodecError("expected a renderer model, state_dict mapping, or None")


def _normalise_block_sizes(block_sizes: tuple[int, ...]) -> tuple[int, ...]:
    if not block_sizes:
        raise QBF1CodecError("block_sizes must not be empty")
    out: list[int] = []
    for raw in block_sizes:
        block_size = int(raw)
        if block_size <= 0 or block_size > 4096:
            raise QBF1CodecError(f"invalid QBF1-v2 profile block size {block_size}")
        if block_size not in out:
            out.append(block_size)
    return tuple(out)


def _raw_then_deflate_key(item: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(item["raw_nbytes"]),
        int(item["deflate9_nbytes"]),
        int(item["block_size"]),
    )


def _profile_payload_bytes(
    *,
    variant_id: str,
    family: str,
    wire_format: str,
    block_size: int,
    raw_bytes: bytes,
    reference_nbytes: int,
    metadata_nbytes: int = 0,
    payload_nbytes: int | None = None,
    strict_pickle_free_loader_ready: bool,
    implemented_loader: bool,
    is_qbf1_v2: bool,
    notes: list[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_nbytes = len(raw_bytes)
    payload_nbytes = raw_nbytes if payload_nbytes is None else int(payload_nbytes)
    out: dict[str, Any] = {
        "variant_id": variant_id,
        "family": family,
        "wire_format": wire_format,
        "block_size": int(block_size),
        "raw_nbytes": int(raw_nbytes),
        "deflate9_nbytes": int(len(zlib.compress(raw_bytes, level=9))),
        "metadata_nbytes": int(metadata_nbytes),
        "payload_nbytes": int(payload_nbytes),
        "raw_delta_vs_reference": int(raw_nbytes - reference_nbytes),
        "beats_reference_raw": raw_nbytes < reference_nbytes,
        "strict_pickle_free_loader_ready": bool(strict_pickle_free_loader_ready),
        "implemented_loader": bool(implemented_loader),
        "deterministic_bytes": True,
        "pickle_free": True,
        "is_qbf1_v2": bool(is_qbf1_v2),
        "score_claim": False,
        "promotion_eligible": False,
        "notes": notes,
    }
    if extra:
        out.update(dict(extra))
    return out


def _profile_qbf1_v1_variant(
    state: Mapping[str, Any],
    *,
    block_size: int,
    reference_nbytes: int,
) -> dict[str, Any]:
    blob = pack_qbf1_state_dict(state, block_size=block_size)
    accounting = qbf1_byte_accounting(blob)
    return _profile_payload_bytes(
        variant_id=f"qbf1_v1_int8_f32_b{block_size}",
        family="qbf1_v1_int8_f32_scales",
        wire_format="QBF1",
        block_size=block_size,
        raw_bytes=blob,
        reference_nbytes=reference_nbytes,
        metadata_nbytes=accounting.metadata_nbytes,
        payload_nbytes=accounting.payload_nbytes,
        strict_pickle_free_loader_ready=True,
        implemented_loader=True,
        is_qbf1_v2=False,
        notes=[
            "implemented strict QBF1 loader path",
            "one int8 byte per value plus float32 block scales and JSON tensor metadata",
        ],
        extra={
            "tensor_payload_nbytes": accounting.tensor_payload_nbytes,
            "header_nbytes": accounting.header_nbytes,
        },
    )


def _profile_qbf1_v2_self_describing_variant(
    state: Mapping[str, Any],
    *,
    block_size: int,
    reference_nbytes: int,
) -> dict[str, Any]:
    from tac.quantizr_qzs3_codec import encode_qzs3_state_dict

    qzs3_payload = encode_qzs3_state_dict(state, block_size=block_size)
    if not qzs3_payload.startswith(b"QZS3"):
        raise QBF1CodecError("internal QBF1-v2 profile expected QZS3-like payload")
    segment_profile = _qbf1_v2_segment_profile(state, block_size=block_size)
    payload_bytes = qzs3_payload[6:]
    if len(payload_bytes) != segment_profile["payload_nbytes"]:
        raise QBF1CodecError("QBF1-v2 profile payload byte accounting mismatch")
    metadata = {
        "schema": QBF1_V2_SELF_DESCRIBING_PROFILE_SCHEMA,
        "wire_format": "QBF1v2-planning-only",
        "block_size": int(block_size),
        "tensor_count": len(segment_profile["tensors"]),
        "segment_bytes": segment_profile["segment_bytes"],
        "payload_nbytes": len(payload_bytes),
        "tensors": segment_profile["tensors"],
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    raw_bytes = b"QB2P" + (b"\x00" * (_HEADER_NBYTES - 4)) + metadata_bytes + payload_bytes
    return _profile_payload_bytes(
        variant_id=f"qbf1_v2_self_describing_fp4_f16_qv_b{block_size}",
        family="qbf1_v2_self_describing_fp4_f16_qv",
        wire_format="QBF1v2-planning-only",
        block_size=block_size,
        raw_bytes=raw_bytes,
        reference_nbytes=reference_nbytes,
        metadata_nbytes=len(metadata_bytes),
        payload_nbytes=len(payload_bytes),
        strict_pickle_free_loader_ready=False,
        implemented_loader=False,
        is_qbf1_v2=True,
        notes=[
            "planning-only compact self-describing FP4/fp16/QV model",
            "uses QZS3-equivalent tensor payload bytes but charges QBF1-style header and tensor metadata",
            "no reviewed decoder or inflate dispatch exists",
        ],
        extra={
            "header_nbytes": _HEADER_NBYTES,
            "segment_bytes": segment_profile["segment_bytes"],
        },
    )


def _profile_existing_qzs3_variant(
    state: Mapping[str, Any],
    *,
    block_size: int,
    reference_nbytes: int,
) -> dict[str, Any]:
    from tac.quantizr_qzs3_codec import encode_qzs3_state_dict

    payload = encode_qzs3_state_dict(state, block_size=block_size)
    return _profile_payload_bytes(
        variant_id=f"existing_qzs3_b{block_size}",
        family="existing_qzs3_block_policy_reference",
        wire_format="QZS3",
        block_size=block_size,
        raw_bytes=payload,
        reference_nbytes=reference_nbytes,
        metadata_nbytes=6,
        payload_nbytes=len(payload) - 6,
        strict_pickle_free_loader_ready=True,
        implemented_loader=True,
        is_qbf1_v2=False,
        notes=[
            "existing QZS3/QZS4-compatible policy reference",
            "included to separate renderer-packer byte wins from QBF1-v2 work",
        ],
    )


def _qbf1_v2_segment_profile(
    state: Mapping[str, Any],
    *,
    block_size: int,
) -> dict[str, Any]:
    torch = _require_torch()
    from tac.quantizr_qzs3_codec import (
        _is_bias_name,
        _is_fp4_weight_name,
        qzs3_qv_specs,
    )

    qv_specs = qzs3_qv_specs()
    segment_bytes = {
        "packed": 0,
        "scales": 0,
        "bias": 0,
        "dense_fp": 0,
        "fp_weight": 0,
        "dense_other": 0,
        "qv": 0,
    }
    tensor_entries: list[dict[str, Any]] = []
    for name, tensor in state.items():
        shape = tuple(int(dim) for dim in tensor.shape)
        value_count = int(tensor.numel())
        if _is_fp4_weight_name(name):
            num_blocks = (value_count + block_size - 1) // block_size
            packed_nbytes = (num_blocks * block_size + 1) // 2
            scale_nbytes = num_blocks * 2
            segment_bytes["packed"] += packed_nbytes
            segment_bytes["scales"] += scale_nbytes
            entry = {
                "name": name,
                "shape": list(shape),
                "kind": "fp4_weight",
                "block_size": int(block_size),
                "value_count": value_count,
                "num_blocks": num_blocks,
                "packed_nbytes": packed_nbytes,
                "scale_nbytes": scale_nbytes,
            }
        elif name.endswith(".weight") and (
            name == "shared_trunk.embedding.weight"
            or name in {"frame1_head.head.weight", "frame2_head.head.weight"}
        ):
            nbytes = value_count * 2
            segment_bytes["fp_weight"] += nbytes
            entry = {
                "name": name,
                "shape": list(shape),
                "kind": "fp16_weight",
                "value_count": value_count,
                "payload_nbytes": nbytes,
            }
        elif _is_bias_name(name):
            nbytes = value_count * 2
            segment_bytes["bias"] += nbytes
            entry = {
                "name": name,
                "shape": list(shape),
                "kind": "fp16_bias",
                "value_count": value_count,
                "payload_nbytes": nbytes,
            }
        elif name in qv_specs:
            bits, per_row = qv_specs[name]
            rows = shape[0] if per_row and len(shape) >= 2 else 1
            nbytes = rows * 4 + (value_count * bits + 7) // 8
            segment_bytes["qv"] += nbytes
            entry = {
                "name": name,
                "shape": list(shape),
                "kind": "qv",
                "bits": int(bits),
                "per_row": bool(per_row),
                "rows": int(rows),
                "value_count": value_count,
                "payload_nbytes": nbytes,
            }
        elif torch.is_floating_point(tensor):
            nbytes = value_count * 2
            segment_bytes["dense_fp"] += nbytes
            entry = {
                "name": name,
                "shape": list(shape),
                "kind": "fp16_dense",
                "value_count": value_count,
                "payload_nbytes": nbytes,
            }
        else:
            nbytes = value_count * tensor.element_size()
            segment_bytes["dense_other"] += nbytes
            entry = {
                "name": name,
                "shape": list(shape),
                "kind": _normalise_torch_dtype_name(str(tensor.dtype)),
                "value_count": value_count,
                "payload_nbytes": nbytes,
            }
        tensor_entries.append(entry)

    return {
        "segment_bytes": segment_bytes,
        "payload_nbytes": sum(segment_bytes.values()),
        "tensors": tensor_entries,
    }


def _read_payload_input(path: str | bytes | bytearray | memoryview | Any) -> bytes:
    if isinstance(path, bytes):
        return path
    if isinstance(path, (bytearray, memoryview)):
        return bytes(path)
    if isinstance(path, str):
        from pathlib import Path

        return Path(path).read_bytes()
    return path.read_bytes()


def _validate_tensor_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise QBF1CodecError("QBF1 tensor names must be non-empty strings")
    if len(name) > 1024:
        raise QBF1CodecError(f"QBF1 tensor name too long: {name!r}")
    if "\x00" in name or "/" in name or "\\" in name:
        raise QBF1CodecError(f"QBF1 tensor name is not archive-local safe: {name!r}")
    if name == "." or name == ".." or ".." in name.split("."):
        raise QBF1CodecError(f"QBF1 tensor name has unsafe path semantics: {name!r}")


def _normalise_torch_dtype_name(dtype_name: str) -> str:
    if dtype_name.startswith("torch."):
        return dtype_name[len("torch."):]
    return dtype_name


def _shape_value_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        if dim < 0:
            raise QBF1CodecError(f"negative tensor shape dimension {dim}")
        count *= dim
    return count


def _quantize_flat_tensor(flat: Any, *, block_size: int) -> tuple[tuple[float, ...], bytes]:
    torch = _require_torch()
    value_count = int(flat.numel())
    if value_count == 0:
        return (), b""

    quantized_parts: list[Any] = []
    scales: list[float] = []
    for start in range(0, value_count, block_size):
        stop = min(start + block_size, value_count)
        block = flat[start:stop]
        max_abs = float(block.abs().max().item())
        if not math.isfinite(max_abs):
            raise QBF1CodecError("cannot quantize non-finite block maximum")
        if max_abs == 0.0:
            scale = 0.0
            q = torch.zeros_like(block, dtype=torch.int8)
        else:
            scale = _float32(max_abs / QBF1_QMAX)
            if scale <= 0.0 or not math.isfinite(scale):
                raise QBF1CodecError("QBF1 scale underflow/overflow")
            q = torch.round(block / scale).clamp(-QBF1_QMAX, QBF1_QMAX).to(torch.int8)
        scales.append(scale)
        quantized_parts.append(q)

    quantized = torch.cat(quantized_parts).to(torch.int16)
    quantized_bytes = bytes(int(value) & 0xFF for value in quantized.tolist())
    return tuple(scales), quantized_bytes


def _float32(value: float) -> float:
    return _SCALE_STRUCT.unpack(_SCALE_STRUCT.pack(float(value)))[0]


def _pack_scales(scales: tuple[float, ...]) -> bytes:
    return b"".join(_SCALE_STRUCT.pack(float(scale)) for scale in scales)


def _build_metadata(
    tensor_entries: list[dict[str, Any]],
    *,
    payload_nbytes: int,
) -> dict[str, Any]:
    tensor_payload_nbytes = sum(int(entry["encoded_nbytes"]) for entry in tensor_entries)
    if tensor_payload_nbytes != payload_nbytes:
        raise QBF1CodecError("internal payload byte accounting mismatch")

    metadata: dict[str, Any] = {
        "schema": QBF1_SCHEMA,
        "version": QBF1_VERSION,
        "tensor_count": len(tensor_entries),
        "block_fp": {
            "quant_dtype": "int8",
            "scale_dtype": "float32",
            "qmax": QBF1_QMAX,
        },
        "byte_accounting": {
            "header_nbytes": _HEADER_NBYTES,
            "metadata_nbytes": 0,
            "payload_nbytes": payload_nbytes,
            "tensor_payload_nbytes": tensor_payload_nbytes,
            "total_nbytes": 0,
        },
        "tensors": tensor_entries,
    }
    previous_len = -1
    for _ in range(8):
        encoded_len = len(_canonical_json_bytes(metadata))
        total_nbytes = _HEADER_NBYTES + encoded_len + payload_nbytes
        accounting = metadata["byte_accounting"]
        accounting["metadata_nbytes"] = encoded_len
        accounting["total_nbytes"] = total_nbytes
        if encoded_len == previous_len:
            return metadata
        previous_len = encoded_len
    raise QBF1CodecError("QBF1 metadata length did not converge")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_metadata(metadata_bytes: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QBF1CodecError("QBF1 metadata is not valid canonical JSON") from exc
    if not isinstance(parsed, dict):
        raise QBF1CodecError("QBF1 metadata must be a JSON object")
    if set(parsed) != _TOP_LEVEL_KEYS:
        raise QBF1CodecError("QBF1 metadata top-level keys do not match schema")
    if parsed["schema"] != QBF1_SCHEMA:
        raise QBF1CodecError(f"unsupported QBF1 schema {parsed['schema']!r}")
    if parsed["version"] != QBF1_VERSION:
        raise QBF1CodecError(f"unsupported QBF1 metadata version {parsed['version']!r}")

    block_fp = parsed["block_fp"]
    if not isinstance(block_fp, dict) or set(block_fp) != _BLOCK_FP_KEYS:
        raise QBF1CodecError("QBF1 block_fp metadata keys do not match schema")
    if block_fp != {"quant_dtype": "int8", "scale_dtype": "float32", "qmax": QBF1_QMAX}:
        raise QBF1CodecError("QBF1 block_fp metadata has unsupported values")

    tensors = parsed["tensors"]
    if not isinstance(tensors, list):
        raise QBF1CodecError("QBF1 tensors metadata must be a list")
    if parsed["tensor_count"] != len(tensors):
        raise QBF1CodecError("QBF1 tensor_count does not match tensors list")
    return parsed


def _validate_byte_accounting(
    metadata: Mapping[str, Any],
    *,
    header_nbytes: int,
    metadata_nbytes: int,
    payload_nbytes: int,
    total_nbytes: int,
) -> QBF1ByteAccounting:
    accounting = metadata["byte_accounting"]
    if not isinstance(accounting, dict) or set(accounting) != _BYTE_ACCOUNTING_KEYS:
        raise QBF1CodecError("QBF1 byte_accounting keys do not match schema")
    expected = {
        "header_nbytes": header_nbytes,
        "metadata_nbytes": metadata_nbytes,
        "payload_nbytes": payload_nbytes,
        "total_nbytes": total_nbytes,
    }
    for key, value in expected.items():
        if accounting.get(key) != value:
            raise QBF1CodecError(
                f"QBF1 byte_accounting[{key!r}]={accounting.get(key)!r} "
                f"does not match {value!r}"
            )
    tensor_payload_nbytes = accounting["tensor_payload_nbytes"]
    if not isinstance(tensor_payload_nbytes, int) or tensor_payload_nbytes < 0:
        raise QBF1CodecError("QBF1 tensor_payload_nbytes must be a non-negative int")
    return QBF1ByteAccounting(
        header_nbytes=header_nbytes,
        metadata_nbytes=metadata_nbytes,
        payload_nbytes=payload_nbytes,
        tensor_payload_nbytes=tensor_payload_nbytes,
        total_nbytes=total_nbytes,
    )


def _parse_tensor_records(
    metadata: Mapping[str, Any],
    payload: bytes,
) -> tuple[QBF1TensorRecord, ...]:
    tensors = metadata["tensors"]
    records: list[QBF1TensorRecord] = []
    cursor = 0
    previous_name: str | None = None

    for raw in tensors:
        if not isinstance(raw, dict) or set(raw) != _TENSOR_KEYS:
            raise QBF1CodecError("QBF1 tensor metadata keys do not match schema")
        name = raw["name"]
        _validate_tensor_name(name)
        if previous_name is not None and name <= previous_name:
            raise QBF1CodecError("QBF1 tensor names must be unique and sorted")
        previous_name = name

        shape = _validate_shape(raw["shape"], name=name)
        value_count = _shape_value_count(shape)
        dtype = raw["dtype"]
        block_size = raw["block_size"]
        num_blocks = raw["num_blocks"]
        scale_nbytes = raw["scale_nbytes"]
        payload_nbytes = raw["payload_nbytes"]
        encoded_nbytes = raw["encoded_nbytes"]
        original_nbytes = raw["original_nbytes"]
        scale_offset = raw["scale_offset"]
        payload_offset = raw["payload_offset"]

        if dtype not in _DTYPE_NBYTES:
            raise QBF1CodecError(f"QBF1 tensor {name!r} has unsupported dtype {dtype!r}")
        for key in (
            "block_size",
            "num_blocks",
            "scale_nbytes",
            "payload_nbytes",
            "encoded_nbytes",
            "original_nbytes",
            "scale_offset",
            "payload_offset",
            "value_count",
        ):
            if not isinstance(raw[key], int) or raw[key] < 0:
                raise QBF1CodecError(f"QBF1 tensor {name!r} has invalid {key}")
        if block_size <= 0:
            raise QBF1CodecError(f"QBF1 tensor {name!r} has invalid block_size")
        expected_blocks = (value_count + block_size - 1) // block_size
        if raw["value_count"] != value_count or num_blocks != expected_blocks:
            raise QBF1CodecError(f"QBF1 tensor {name!r} shape/count metadata mismatch")
        if raw["quant_dtype"] != "int8" or raw["scale_dtype"] != "float32":
            raise QBF1CodecError(f"QBF1 tensor {name!r} has unsupported stream dtypes")
        if original_nbytes != value_count * _DTYPE_NBYTES[dtype]:
            raise QBF1CodecError(f"QBF1 tensor {name!r} original_nbytes mismatch")
        if scale_nbytes != num_blocks * _SCALE_STRUCT.size:
            raise QBF1CodecError(f"QBF1 tensor {name!r} scale_nbytes mismatch")
        if payload_nbytes != value_count:
            raise QBF1CodecError(f"QBF1 tensor {name!r} payload_nbytes mismatch")
        if encoded_nbytes != scale_nbytes + payload_nbytes:
            raise QBF1CodecError(f"QBF1 tensor {name!r} encoded_nbytes mismatch")
        if scale_offset != cursor:
            raise QBF1CodecError(f"QBF1 tensor {name!r} scale_offset mismatch")
        cursor += scale_nbytes
        if payload_offset != cursor:
            raise QBF1CodecError(f"QBF1 tensor {name!r} payload_offset mismatch")
        cursor += payload_nbytes
        if cursor > len(payload):
            raise QBF1CodecError(f"QBF1 tensor {name!r} extends past payload")

        scale_bytes = payload[scale_offset:scale_offset + scale_nbytes]
        quantized = payload[payload_offset:payload_offset + payload_nbytes]
        scales = _unpack_scales(scale_bytes, expected_count=num_blocks, tensor_name=name)
        _validate_quantized_blocks(
            quantized,
            scales=scales,
            block_size=block_size,
            value_count=value_count,
            tensor_name=name,
        )
        records.append(
            QBF1TensorRecord(
                metadata=QBF1TensorMetadata(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    block_size=block_size,
                    value_count=value_count,
                    num_blocks=num_blocks,
                    quant_dtype="int8",
                    scale_dtype="float32",
                    scales=scales,
                    original_nbytes=original_nbytes,
                    scale_nbytes=scale_nbytes,
                    payload_nbytes=payload_nbytes,
                    encoded_nbytes=encoded_nbytes,
                    scale_offset=scale_offset,
                    payload_offset=payload_offset,
                ),
                quantized=quantized,
            )
        )

    if cursor != len(payload):
        raise QBF1CodecError(
            f"QBF1 tensor payload accounting ended at {cursor}, got {len(payload)} bytes"
        )
    accounting_payload = metadata["byte_accounting"]["tensor_payload_nbytes"]
    if accounting_payload != cursor:
        raise QBF1CodecError("QBF1 tensor payload total does not match accounting")
    return tuple(records)


def _validate_shape(raw_shape: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(raw_shape, list):
        raise QBF1CodecError(f"QBF1 tensor {name!r} shape must be a list")
    shape: list[int] = []
    for dim in raw_shape:
        if not isinstance(dim, int) or dim < 0:
            raise QBF1CodecError(f"QBF1 tensor {name!r} has invalid shape dimension")
        shape.append(dim)
    return tuple(shape)


def _unpack_scales(
    scale_bytes: bytes,
    *,
    expected_count: int,
    tensor_name: str,
) -> tuple[float, ...]:
    if len(scale_bytes) != expected_count * _SCALE_STRUCT.size:
        raise QBF1CodecError(f"QBF1 tensor {tensor_name!r} scale byte length mismatch")
    scales: list[float] = []
    for offset in range(0, len(scale_bytes), _SCALE_STRUCT.size):
        (scale,) = _SCALE_STRUCT.unpack_from(scale_bytes, offset)
        scale = float(scale)
        if scale < 0.0 or not math.isfinite(scale):
            raise QBF1CodecError(f"QBF1 tensor {tensor_name!r} has invalid scale")
        scales.append(scale)
    return tuple(scales)


def _validate_quantized_blocks(
    quantized: bytes,
    *,
    scales: tuple[float, ...],
    block_size: int,
    value_count: int,
    tensor_name: str,
) -> None:
    if len(quantized) != value_count:
        raise QBF1CodecError(f"QBF1 tensor {tensor_name!r} quantized length mismatch")
    for block_index, scale in enumerate(scales):
        if scale != 0.0:
            continue
        start = block_index * block_size
        stop = min(start + block_size, value_count)
        if any(byte != 0 for byte in quantized[start:stop]):
            raise QBF1CodecError(
                f"QBF1 tensor {tensor_name!r} has nonzero qint in zero-scale block"
            )


__all__ = [
    "QBF1_DEFAULT_BLOCK_SIZE",
    "QBF1_MAGIC",
    "QBF1_QMAX",
    "QBF1_SCHEMA",
    "QBF1_V2_BYTE_PROFILE_SCHEMA",
    "QBF1_V2_DEFAULT_BLOCK_SIZES",
    "QBF1_V2_REFERENCE_QZS3_NBYTES",
    "QBF1_V2_SELF_DESCRIBING_PROFILE_SCHEMA",
    "QBF1_VERSION",
    "QBF1ByteAccounting",
    "QBF1CodecError",
    "QBF1Container",
    "QBF1TensorMetadata",
    "QBF1TensorRecord",
    "decode_qbf1_renderer_state_dict",
    "decode_qbf1_state_dict",
    "load_qbf1",
    "load_qbf1_renderer",
    "pack_qbf1_renderer_state_dict",
    "pack_qbf1_state_dict",
    "profile_qbf1_v2_renderer_bytes",
    "qbf1_byte_accounting",
    "unpack_qbf1_container",
    "unpack_qbf1_renderer_container",
]
