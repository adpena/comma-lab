"""QH0/QM0 renderer codec for PR85-style JointFrameGenerator payloads.

The public PR85/PR89 family stores a Quantizr-faithful
``JointFrameGenerator`` as a compact custom byte stream.  This module decodes
that stream without importing the public submission runtime or using pickle.
It is intentionally narrow: the output is the existing
``tac.quantizr_faithful_renderer.JointFrameGenerator`` used by robust_current.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import struct
import zlib
from dataclasses import dataclass

import numpy as np
import torch

from tac.quantizr_faithful_renderer import JointFrameGenerator, build_quantizr_faithful_renderer


QH0_MAGIC = b"QH0"
QM0_MAGIC = b"QM0"
QH1_MAGIC = b"QH1"
QH0_SUPPORTED_MAGICS = (QH0_MAGIC, QM0_MAGIC)
QH1_HEADER_STRUCT = struct.Struct("<I")
QH1_SCHEMA = "qh1_record_repack_v1"
FP4_POS_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


class QH0CodecError(ValueError):
    """Raised when a QH0/QM0 payload is malformed or unsupported."""


@dataclass(frozen=True)
class QH0DecodeReport:
    """Small decode-custody summary for logs and tests."""

    magic: str
    payload_bytes: int
    consumed_bytes: int
    tensor_count: int
    q_fp4_tensor_count: int
    fp16_tensor_count: int
    int8_dense_tensor_count: int


def unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack high/low 4-bit nibbles in the public PR85 order."""

    if count < 0:
        raise QH0CodecError(f"nibble count must be non-negative, got {count}")
    flat = packed.reshape(-1).to(torch.uint8)
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = (flat >> 4) & 0x0F
    out[1::2] = flat & 0x0F
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
        return lzma.decompress(data)  # type: ignore[name-defined]
    if codec == "brotli":
        try:
            import brotli  # type: ignore
        except ImportError as exc:  # pragma: no cover - clean contest env guard
            raise QH0CodecError(f"{label}: QH1 brotli record requires brotli") from exc
        try:
            return brotli.decompress(data)
        except brotli.error as exc:
            raise QH0CodecError(f"{label}: invalid QH1 brotli record") from exc
    raise QH0CodecError(f"{label}: unsupported QH1 record codec {codec!r}")


def reconstruct_qh1_payload(payload: bytes | bytearray | memoryview) -> bytes:
    """Reconstruct original QH0/QM0 bytes from a QH1 record-repack payload.

    QH1 is intentionally lossless: it patches compressed record slices back into
    a stored QH0/QM0 source byte stream, then the normal reviewed QH0 decoder
    handles model tensors. It is a byte-packer only, not a renderer mutation.
    """

    raw = bytes(payload)
    if not raw.startswith(QH1_MAGIC):
        return raw
    min_len = len(QH1_MAGIC) + QH1_HEADER_STRUCT.size
    if len(raw) < min_len:
        raise QH0CodecError("QH1 payload is shorter than magic/header")
    header_len = QH1_HEADER_STRUCT.unpack_from(raw, len(QH1_MAGIC))[0]
    header_start = len(QH1_MAGIC) + QH1_HEADER_STRUCT.size
    header_end = header_start + header_len
    if header_len <= 0 or header_end > len(raw):
        raise QH0CodecError(f"invalid QH1 header length: {header_len}")
    try:
        header = json.loads(raw[header_start:header_end].decode("utf-8"))
    except Exception as exc:
        raise QH0CodecError("QH1 header is not valid UTF-8 JSON") from exc
    if header.get("schema") != QH1_SCHEMA:
        raise QH0CodecError(f"unsupported QH1 schema: {header.get('schema')!r}")
    base_codec = str(header.get("base_codec", ""))
    base_encoded_bytes = int(header.get("base_encoded_bytes", -1))
    if base_encoded_bytes < 0 or header_end + base_encoded_bytes > len(raw):
        raise QH0CodecError("QH1 base template is truncated")
    base_encoded = raw[header_end : header_end + base_encoded_bytes]
    if str(header.get("base_encoded_sha256", "")) != _sha256(base_encoded):
        raise QH0CodecError("QH1 base encoded SHA-256 mismatch")
    out = bytearray(_decode_qh1_record(base_encoded, base_codec, "QH1 base"))
    if str(header.get("base_decoded_sha256", "")) != _sha256(bytes(out)):
        raise QH0CodecError("QH1 base decoded SHA-256 mismatch")
    expected_bytes = int(header.get("source_bytes", -1))
    if expected_bytes != len(out):
        raise QH0CodecError(
            f"QH1 source length mismatch: header={expected_bytes} base={len(out)}"
        )
    expected_sha = str(header.get("source_sha256", ""))
    body_pos = header_end + base_encoded_bytes
    seen: set[tuple[int, int]] = set()
    records = header.get("records")
    if not isinstance(records, list):
        raise QH0CodecError("QH1 header records must be a list")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise QH0CodecError(f"QH1 record {index} is not an object")
        offset = int(record.get("offset", -1))
        nbytes = int(record.get("nbytes", -1))
        encoded_bytes = int(record.get("encoded_bytes", -1))
        codec = str(record.get("codec", ""))
        label = f"QH1 record {index}"
        if offset < 0 or nbytes < 0 or offset + nbytes > len(out):
            raise QH0CodecError(
                f"{label}: invalid target slice offset={offset} nbytes={nbytes}"
            )
        if encoded_bytes < 0 or body_pos + encoded_bytes > len(raw):
            raise QH0CodecError(f"{label}: encoded slice is truncated")
        key = (offset, offset + nbytes)
        if any(not (key[1] <= old[0] or old[1] <= key[0]) for old in seen):
            raise QH0CodecError(f"{label}: overlapping target slice")
        seen.add(key)
        encoded = raw[body_pos : body_pos + encoded_bytes]
        body_pos += encoded_bytes
        if str(record.get("encoded_sha256", "")) != _sha256(encoded):
            raise QH0CodecError(f"{label}: encoded SHA-256 mismatch")
        decoded = _decode_qh1_record(encoded, codec, label)
        if len(decoded) != nbytes:
            raise QH0CodecError(
                f"{label}: decoded length mismatch: got={len(decoded)} expected={nbytes}"
            )
        if str(record.get("decoded_sha256", "")) != _sha256(decoded):
            raise QH0CodecError(f"{label}: decoded SHA-256 mismatch")
        out[offset : offset + nbytes] = decoded
    if body_pos != len(raw):
        raise QH0CodecError(f"QH1 payload has trailing bytes: consumed={body_pos} total={len(raw)}")
    rebuilt = bytes(out)
    if expected_sha != _sha256(rebuilt):
        raise QH0CodecError("QH1 reconstructed source SHA-256 mismatch")
    if rebuilt[:3] not in QH0_SUPPORTED_MAGICS:
        raise QH0CodecError(f"QH1 reconstructed unsupported magic: {rebuilt[:3]!r}")
    return rebuilt


def _unsplit_bytes_to_tensor(
    raw: bytes,
    pos: int,
    nbytes: int,
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
    device: torch.device | str,
    label: str,
) -> tuple[torch.Tensor, int]:
    """Undo PR85 QH0 high/low nibble split into packed FP4 bytes."""

    if packed_len % 2:
        raise QH0CodecError(f"{label} packed_len must be even for QH0, got {packed_len}")
    half = packed_len // 2
    _require_available(raw, pos, half * 2, label)
    hi_packed = np.frombuffer(raw[pos : pos + half], dtype=np.uint8)
    pos += half
    lo_packed = np.frombuffer(raw[pos : pos + half], dtype=np.uint8)
    pos += half
    hi = np.empty(half * 2, dtype=np.uint8)
    lo = np.empty(half * 2, dtype=np.uint8)
    hi[0::2] = (hi_packed >> 4) & 0x0F
    hi[1::2] = hi_packed & 0x0F
    lo[0::2] = (lo_packed >> 4) & 0x0F
    lo[1::2] = lo_packed & 0x0F
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
            f"FP4 nibbles/scales mismatch: nibbles={nibbles.numel()} scales={scales.numel()}"
        )
    block_size = nibbles.numel() // scales.numel()
    nib = nibbles.view(-1, block_size)
    signs = (nib >> 3).to(torch.int64)
    mag_idx = (nib & 0x7).to(torch.int64)
    levels = FP4_POS_LEVELS.to(scales.device, torch.float32)
    q = levels[mag_idx]
    q = torch.where(signs.bool(), -q, q)
    dq = q * scales[:, None].to(torch.float32)
    return dq.reshape(-1)[:flat_n].reshape(shape)


def _module_weight_order(model: JointFrameGenerator) -> list[tuple[str, torch.nn.Module]]:
    ordered: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Embedding)):
            ordered.append((name, module))
    return ordered


def decode_qh0_state_dict(
    payload: bytes | bytearray | memoryview,
    *,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, torch.Tensor], QH0DecodeReport]:
    """Decode PR85/PR89 QH0 or QM0 bytes into a JointFrameGenerator state dict."""

    raw = reconstruct_qh1_payload(payload)
    if len(raw) < 3:
        raise QH0CodecError("QH0 payload is shorter than the 3-byte magic")
    magic = raw[:3]
    if magic not in QH0_SUPPORTED_MAGICS:
        raise QH0CodecError(f"unsupported QH0 renderer magic: {magic!r}")
    hilo_split = magic == QH0_MAGIC
    pos = 3
    model = build_quantizr_faithful_renderer()
    state: dict[str, torch.Tensor] = {}
    covered: set[str] = set()
    fp4_count = 0
    fp16_count = 0
    int8_count = 0

    for name, module in _module_weight_order(model):
        kind, pos = _read_u8(raw, pos, f"{name}.weight kind")
        shape = tuple(int(x) for x in module.weight.shape)
        numel = int(module.weight.numel())
        if kind == 1:
            block_size = 32
            blocks = (numel + block_size - 1) // block_size
            packed_len = (blocks * block_size + 1) // 2
            if hilo_split:
                packed, pos = _unhilo_packed(raw, pos, packed_len, device, f"{name}.weight")
                scales, pos = _unsplit_bytes_to_tensor(
                    raw, pos, blocks * 2, torch.float16, (blocks,), device, f"{name}.scales"
                )
            else:
                packed, pos = _read_tensor_bytes(
                    raw, pos, packed_len, torch.uint8, (packed_len,), device, f"{name}.packed"
                )
                scales, pos = _read_tensor_bytes(
                    raw, pos, blocks * 2, torch.float16, (blocks,), device, f"{name}.scales"
                )
            nibbles = unpack_nibbles(packed, packed.numel() * 2)
            tensor = _dequantize_fp4_from_nibbles(nibbles, scales, shape)
            fp4_count += 1
        elif kind == 0:
            nbytes = numel * 2
            if hilo_split:
                tensor, pos = _unsplit_bytes_to_tensor(
                    raw, pos, nbytes, torch.float16, shape, device, f"{name}.weight"
                )
            else:
                tensor, pos = _read_tensor_bytes(
                    raw, pos, nbytes, torch.float16, shape, device, f"{name}.weight"
                )
            tensor = tensor.float()
            fp16_count += 1
        else:
            raise QH0CodecError(f"bad QH0 weight kind {kind} for {name}.weight")
        state[f"{name}.weight"] = tensor.float()
        covered.add(f"{name}.weight")

        bias = getattr(module, "bias", None)
        if bias is not None:
            bias_shape = tuple(int(x) for x in bias.shape)
            nbytes = int(bias.numel()) * 2
            if hilo_split:
                bias_tensor, pos = _unsplit_bytes_to_tensor(
                    raw, pos, nbytes, torch.float16, bias_shape, device, f"{name}.bias"
                )
            else:
                bias_tensor, pos = _read_tensor_bytes(
                    raw, pos, nbytes, torch.float16, bias_shape, device, f"{name}.bias"
                )
            state[f"{name}.bias"] = bias_tensor.float()
            covered.add(f"{name}.bias")
            fp16_count += 1

    for key, tensor in model.state_dict().items():
        if key in covered:
            continue
        kind, pos = _read_u8(raw, pos, f"{key} dense kind")
        shape = tuple(int(x) for x in tensor.shape)
        numel = int(tensor.numel())
        if kind == 2:
            q, pos = _read_tensor_bytes(raw, pos, numel, torch.int8, shape, device, f"{key}.q_int8")
            rows = shape[0] if len(shape) >= 2 else 1
            if hilo_split:
                scales, pos = _unsplit_bytes_to_tensor(
                    raw, pos, rows * 2, torch.float16, (rows,), device, f"{key}.scales"
                )
            else:
                scales, pos = _read_tensor_bytes(
                    raw, pos, rows * 2, torch.float16, (rows,), device, f"{key}.scales"
                )
            state[key] = (q.float() * scales.float()[:, None]).reshape(shape)
            int8_count += 1
        elif kind == 0:
            nbytes = numel * 2
            if hilo_split:
                dense, pos = _unsplit_bytes_to_tensor(
                    raw, pos, nbytes, torch.float16, shape, device, key
                )
            else:
                dense, pos = _read_tensor_bytes(raw, pos, nbytes, torch.float16, shape, device, key)
            state[key] = dense.float()
            fp16_count += 1
        else:
            raise QH0CodecError(f"bad QH0 dense kind {kind} for {key}")

    if pos != len(raw):
        raise QH0CodecError(f"QH0 payload has trailing bytes: consumed={pos} total={len(raw)}")
    report = QH0DecodeReport(
        magic=magic.decode("ascii"),
        payload_bytes=len(raw),
        consumed_bytes=pos,
        tensor_count=len(state),
        q_fp4_tensor_count=fp4_count,
        fp16_tensor_count=fp16_count,
        int8_dense_tensor_count=int8_count,
    )
    return state, report


def load_qh0(
    payload: bytes | bytearray | memoryview,
    *,
    device: torch.device | str = "cpu",
) -> JointFrameGenerator:
    """Load QH0/QM0 bytes into a Quantizr-faithful JointFrameGenerator."""

    state, _report = decode_qh0_state_dict(payload, device=device)
    model = build_quantizr_faithful_renderer()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
