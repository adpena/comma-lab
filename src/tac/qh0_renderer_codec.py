"""QH0/QM0 renderer codec for PR85-style JointFrameGenerator payloads.

The public PR85/PR89 family stores a Quantizr-faithful
``JointFrameGenerator`` as a compact custom byte stream. This module decodes
that stream without importing the public submission runtime or using pickle.
It is intentionally narrow: the output is the existing
``tac.quantizr_faithful_renderer.JointFrameGenerator`` used by robust_current.

REHYDRATED 2026-05-05 from .recovery_spec.json (preserved at
.recovery_quarantine_20260505T004735Z/src/tac/qh0_renderer_codec.recovery_spec.json).
Spec source: bytecode disassembly of compiled .pyc; whitespace + inline comments lost.

PARTIAL REHYDRATION: Module-level constants, error class, and the simple
nibble/byte-split helpers are reconstructed exactly. The
``reconstruct_qh1_payload`` and ``decode_qh0_state_dict`` record-loop bodies
are stubbed to ``NotImplementedError`` because the bytecode disassembly contains
a long per-record state machine (FP4 / FP16 / int8 record types, hi/lo split,
covered-set bookkeeping) that pycdc cannot fully decompile. The inflate
consumer (``submissions/robust_current/inflate_renderer.py:4358``) catches
``ImportError`` but will surface ``NotImplementedError`` loudly so wrong
masks/weights are not silently produced.
"""
from __future__ import annotations

import hashlib
import lzma
import math
import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from tac.quantizr_faithful_renderer import (
    JointFrameGenerator,
    build_quantizr_faithful_renderer,
)

QH0_MAGIC = b"QH0"
QM0_MAGIC = b"QM0"
QH1_MAGIC = b"QH1"
QH0_SUPPORTED_MAGICS = (QH0_MAGIC, QM0_MAGIC)
QH1_HEADER_STRUCT = struct.Struct("<I")
QH1_SCHEMA = "qh1_record_repack_v1"

FP4_POS_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

_QUARANTINE_SPEC = (
    ".recovery_quarantine_20260505T004735Z/src/tac/qh0_renderer_codec.recovery_spec.json"
)


class QH0CodecError(ValueError):
    """Raised when a QH0/QM0 payload is malformed or unsupported."""

    pass


@dataclass(frozen=True)
class QH0DecodeReport:
    """Record-level diagnostics produced by ``decode_qh0_state_dict``."""

    magic: bytes
    hilo_split: bool
    fp4_count: int
    fp16_count: int
    int8_count: int
    payload_bytes: int
    payload_sha256: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def q_fp4_tensor_count(self) -> int:
        """Compatibility name used by the contest inflate runtime."""
        return self.fp4_count


def unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack high/low 4-bit nibbles in the public PR85 order."""
    if count < 0:
        raise QH0CodecError(f"nibble count must be non-negative, got {count}")
    flat = packed.reshape(-1).to(torch.uint8)
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = (flat >> 4) & 15
    out[1::2] = flat & 15
    return out[:count]


def _require_available(raw: bytes, pos: int, nbytes: int, label: str) -> None:
    if nbytes < 0 or pos < 0 or pos + nbytes > len(raw):
        raise QH0CodecError(
            f"QH0 payload truncated while reading {label}: "
            f"pos={pos} nbytes={nbytes} payload={len(raw)}"
        )


def _read_u8(raw: bytes, pos: int, label: str) -> tuple[int, int]:
    _require_available(raw, pos, 1, label)
    return int(raw[pos]), pos + 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode_qh1_record(data: bytes, codec: str, label: str) -> bytes:
    if codec == "raw":
        return data
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "lzma":
        return lzma.decompress(data)
    if codec == "brotli":
        try:
            import brotli
        except ImportError as exc:
            raise QH0CodecError(
                f"{label}: QH1 brotli record requires brotli"
            ) from exc
        try:
            return brotli.decompress(data)
        except brotli.error as exc:
            raise QH0CodecError(f"{label}: invalid QH1 brotli record") from exc
    raise QH0CodecError(f"{label}: unsupported QH1 record codec {codec!r}")


def _unsplit_bytes_to_tensor(
    raw: bytes,
    pos: int,
    nbytes: int,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device | str,
    label: str,
) -> tuple[torch.Tensor, int]:
    """Undo PR85's even/odd byte split before constructing a tensor."""
    _require_available(raw, pos, nbytes, label)
    half = (nbytes + 1) // 2
    source = np.frombuffer(raw[pos : pos + nbytes], dtype=np.uint8)
    out = np.empty(nbytes, dtype=np.uint8)
    out[0::2] = source[:half]
    out[1::2] = source[half:]
    tensor = torch.frombuffer(bytearray(out.tobytes()), dtype=dtype).clone()
    return tensor.reshape(shape).to(device), pos + nbytes


def _read_tensor_bytes(
    raw: bytes,
    pos: int,
    nbytes: int,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device | str,
    label: str,
) -> tuple[torch.Tensor, int]:
    _require_available(raw, pos, nbytes, label)
    tensor = torch.frombuffer(bytearray(raw[pos : pos + nbytes]), dtype=dtype).clone()
    return tensor.reshape(shape).to(device), pos + nbytes


def _unhilo_packed(
    raw: bytes,
    pos: int,
    packed_len: int,
    *,
    device: torch.device | str,
    label: str,
) -> tuple[torch.Tensor, int]:
    """Undo PR85 QH0 high/low nibble split into packed FP4 bytes."""
    if packed_len % 2:
        raise QH0CodecError(
            f"{label} packed_len must be even for QH0, got {packed_len}"
        )
    half = packed_len // 2
    _require_available(raw, pos, half * 2, label)
    hi_packed = np.frombuffer(raw[pos : pos + half], dtype=np.uint8)
    pos += half
    lo_packed = np.frombuffer(raw[pos : pos + half], dtype=np.uint8)
    pos += half
    hi = np.empty(half * 2, dtype=np.uint8)
    lo = np.empty(half * 2, dtype=np.uint8)
    hi[0::2] = (hi_packed >> 4) & 15
    hi[1::2] = hi_packed & 15
    lo[0::2] = (lo_packed >> 4) & 15
    lo[1::2] = lo_packed & 15
    packed = ((hi[:packed_len] << 4) | lo[:packed_len]).astype(np.uint8)
    tensor = torch.frombuffer(bytearray(packed.tobytes()), dtype=torch.uint8).clone()
    return tensor.to(device), pos


def _dequantize_fp4_from_nibbles(
    nibbles: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    flat_n = math.prod(shape)
    if scales.numel() <= 0:
        raise QH0CodecError("FP4 record has no scales")
    if nibbles.numel() % scales.numel() != 0:
        raise QH0CodecError(
            f"FP4 nibbles/scales mismatch: nibbles={nibbles.numel()} "
            f"scales={scales.numel()}"
        )
    block_size = nibbles.numel() // scales.numel()
    nib = nibbles.view(-1, block_size)
    signs = (nib >> 3).to(torch.int64)
    mag_idx = (nib & 7).to(torch.int64)
    levels = FP4_POS_LEVELS.to(scales.device, torch.float32)
    q = levels[mag_idx]
    q = torch.where(signs.bool(), -q, q)
    dq = q * scales[:, None].to(torch.float32)
    return dq.reshape(-1)[:flat_n].reshape(shape)


def _module_weight_order(model: JointFrameGenerator) -> list[tuple[str, torch.nn.Module]]:
    ordered: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Embedding)):
            continue
        ordered.append((name, module))
    return ordered


def _rehydration_failure(symbol: str) -> NotImplementedError:
    return NotImplementedError(
        f"rehydration incomplete: {symbol} contains the QH0/QM0 per-record "
        f"state machine (FP4/FP16/int8 record dispatch + covered-set "
        f"bookkeeping) that pycdc cannot fully decompile; original bytecode "
        f"preserved in {_QUARANTINE_SPEC}"
    )


def reconstruct_qh1_payload(payload: bytes) -> bytes:
    """Reconstruct original QH0/QM0 bytes from a QH1 record-repack payload.

    QH1 is intentionally lossless: it patches compressed record slices back
    into a stored QH0/QM0 source byte stream, then the normal reviewed QH0
    decoder handles model tensors. It is a byte-packer only, not a renderer
    mutation. Currently a passthrough for non-QH1 magics; QH1 unwrap is
    deferred.
    """
    if len(payload) < 3:
        raise QH0CodecError("QH0 payload is shorter than the 3-byte magic")
    magic = payload[:3]
    if magic in QH0_SUPPORTED_MAGICS:
        return payload
    if magic == QH1_MAGIC:
        raise _rehydration_failure("reconstruct_qh1_payload[QH1 unwrap]")
    raise QH0CodecError(f"unsupported QH0 renderer magic: {magic!r}")


def decode_qh0_state_dict(
    payload: bytes, *, device: torch.device | str
) -> tuple[dict[str, torch.Tensor], QH0DecodeReport]:
    """Decode PR85/PR89 QH0 or QM0 bytes into a JointFrameGenerator state dict.

    This is a reviewed port of the public PR85/PR89 runtime grammar. It
    dispatches between FP4 Conv/Embedding records, FP16 dense records, and
    row-scaled int8 dense records while preserving the QH0 hi/lo and even/odd
    byte transforms exactly.
    """
    if len(payload) < 3:
        raise QH0CodecError("QH0 payload is shorter than the 3-byte magic")
    payload = reconstruct_qh1_payload(payload)
    magic = payload[:3]
    if magic not in QH0_SUPPORTED_MAGICS:
        raise QH0CodecError(f"unsupported QH0 renderer magic: {magic!r}")

    pos = 3
    hilo_split = magic == QH0_MAGIC
    state: dict[str, torch.Tensor] = {}
    covered: set[str] = set()
    fp4_count = 0
    fp16_count = 0
    int8_count = 0
    probe = build_quantizr_faithful_renderer()

    for name, module in _module_weight_order(probe):
        kind, pos = _read_u8(payload, pos, f"{name}.weight kind")
        shape = tuple(module.weight.shape)
        numel = int(module.weight.numel())
        if kind == 1:
            block_size = 32
            blocks = (numel + block_size - 1) // block_size
            packed_len = (blocks * block_size + 1) // 2
            if hilo_split:
                packed, pos = _unhilo_packed(
                    payload,
                    pos,
                    packed_len,
                    device=device,
                    label=f"{name}.weight fp4",
                )
                scales, pos = _unsplit_bytes_to_tensor(
                    payload,
                    pos,
                    blocks * 2,
                    dtype=torch.float16,
                    shape=(blocks,),
                    device=device,
                    label=f"{name}.weight fp4 scales",
                )
            else:
                packed, pos = _read_tensor_bytes(
                    payload,
                    pos,
                    packed_len,
                    dtype=torch.uint8,
                    shape=(packed_len,),
                    device=device,
                    label=f"{name}.weight fp4",
                )
                scales, pos = _read_tensor_bytes(
                    payload,
                    pos,
                    blocks * 2,
                    dtype=torch.float16,
                    shape=(blocks,),
                    device=device,
                    label=f"{name}.weight fp4 scales",
                )
            nibbles = unpack_nibbles(packed, packed.numel() * 2)
            tensor = _dequantize_fp4_from_nibbles(nibbles, scales, shape)
            fp4_count += 1
        elif kind == 0:
            nbytes = numel * 2
            if hilo_split:
                tensor, pos = _unsplit_bytes_to_tensor(
                    payload,
                    pos,
                    nbytes,
                    dtype=torch.float16,
                    shape=shape,
                    device=device,
                    label=f"{name}.weight fp16",
                )
            else:
                tensor, pos = _read_tensor_bytes(
                    payload,
                    pos,
                    nbytes,
                    dtype=torch.float16,
                    shape=shape,
                    device=device,
                    label=f"{name}.weight fp16",
                )
            tensor = tensor.float()
            fp16_count += 1
        else:
            raise QH0CodecError(f"bad custom model q kind {kind} for {name}")

        state[f"{name}.weight"] = tensor.float()
        covered.add(f"{name}.weight")
        if getattr(module, "bias", None) is not None:
            bias_shape = tuple(module.bias.shape)
            nbytes = int(module.bias.numel()) * 2
            if hilo_split:
                bias, pos = _unsplit_bytes_to_tensor(
                    payload,
                    pos,
                    nbytes,
                    dtype=torch.float16,
                    shape=bias_shape,
                    device=device,
                    label=f"{name}.bias fp16",
                )
            else:
                bias, pos = _read_tensor_bytes(
                    payload,
                    pos,
                    nbytes,
                    dtype=torch.float16,
                    shape=bias_shape,
                    device=device,
                    label=f"{name}.bias fp16",
                )
            state[f"{name}.bias"] = bias.float()
            covered.add(f"{name}.bias")
            fp16_count += 1

    for key, tensor in probe.state_dict().items():
        if key in covered:
            continue
        kind, pos = _read_u8(payload, pos, f"{key} dense kind")
        shape = tuple(tensor.shape)
        numel = int(tensor.numel())
        if kind == 2:
            quantized, pos = _read_tensor_bytes(
                payload,
                pos,
                numel,
                dtype=torch.int8,
                shape=shape,
                device=device,
                label=f"{key} int8 values",
            )
            rows = shape[0] if len(shape) >= 2 else 1
            if hilo_split:
                scales, pos = _unsplit_bytes_to_tensor(
                    payload,
                    pos,
                    rows * 2,
                    dtype=torch.float16,
                    shape=(rows,),
                    device=device,
                    label=f"{key} int8 scales",
                )
            else:
                scales, pos = _read_tensor_bytes(
                    payload,
                    pos,
                    rows * 2,
                    dtype=torch.float16,
                    shape=(rows,),
                    device=device,
                    label=f"{key} int8 scales",
                )
            state[key] = (quantized.float() * scales.float()[:, None]).reshape(shape)
            int8_count += 1
        elif kind == 0:
            nbytes = numel * 2
            if hilo_split:
                dense, pos = _unsplit_bytes_to_tensor(
                    payload,
                    pos,
                    nbytes,
                    dtype=torch.float16,
                    shape=shape,
                    device=device,
                    label=f"{key} fp16",
                )
            else:
                dense, pos = _read_tensor_bytes(
                    payload,
                    pos,
                    nbytes,
                    dtype=torch.float16,
                    shape=shape,
                    device=device,
                    label=f"{key} fp16",
                )
            state[key] = dense.float()
            fp16_count += 1
        else:
            raise QH0CodecError(f"bad custom model dense kind {kind} for {key}")

    if pos != len(payload):
        raise QH0CodecError(
            f"QH0 payload has trailing bytes after decode: pos={pos} payload={len(payload)}"
        )
    return state, QH0DecodeReport(
        magic=magic,
        hilo_split=hilo_split,
        fp4_count=fp4_count,
        fp16_count=fp16_count,
        int8_count=int8_count,
        payload_bytes=len(payload),
        payload_sha256=_sha256(payload),
    )


def load_qh0(
    payload: bytes, *, device: torch.device | str
) -> JointFrameGenerator:
    """Load QH0/QM0 bytes into a Quantizr-faithful JointFrameGenerator."""
    state, _report = decode_qh0_state_dict(payload, device=device)
    model = build_quantizr_faithful_renderer()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
