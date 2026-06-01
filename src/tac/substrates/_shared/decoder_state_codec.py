# SPDX-License-Identifier: MIT
"""Contest-runtime decoder-state codec portfolio for compact receivers."""

from __future__ import annotations

import io
import json
import lzma
import pickle
import struct
import zlib
from dataclasses import dataclass
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

from tac.substrates._shared.int_stream_codec import (
    pack_fixed_width_uints,
    unpack_fixed_width_uints,
)

DECODER_STATE_CODEC_MAGIC = b"DSC1"
_HEADER = struct.Struct("<4sBI")
_SUPPORTED_COMPRESSORS = ("brotli_q11", "lzma_preset9", "zlib_level9", "none")


@dataclass(frozen=True)
class DecoderStateCodecStats:
    """Byte accounting for one decoder-state encoding."""

    codec: str
    compressor: str
    payload_bytes: int
    envelope_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "codec": self.codec,
            "compressor": self.compressor,
            "payload_bytes": self.payload_bytes,
            "envelope_bytes": self.envelope_bytes,
        }


def serialize_decoder_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    codec: str = "fp16_brotli_legacy",
) -> bytes:
    """Serialize decoder weights through a measured codec portfolio.

    ``fp16_brotli_legacy`` intentionally emits the historic raw brotli-pickle
    payload so old archives remain byte-compatible.  The int8 path is an
    archive-visible codec: tensors are quantized with charged scales, wrapped in
    a codec-tagged envelope, and decoded by the vendored contest runtime.
    """

    normalized = str(codec).strip().lower()
    if normalized in {"fp16", "fp16_brotli", "fp16_brotli_legacy", "legacy"}:
        return _legacy_fp16_brotli(state_dict)
    candidates: list[tuple[bytes, DecoderStateCodecStats]] = []
    if normalized in {"auto", "portfolio_auto", "int8_auto"}:
        for tensor_codec in ("int8_mixed", "fp16_enveloped"):
            payload = _payload_for_codec(state_dict, codec=tensor_codec)
            for compressor in _SUPPORTED_COMPRESSORS:
                candidates.append(
                    _wrap_payload(payload, codec=tensor_codec, compressor=compressor)
                )
    elif normalized in {"int8", "int8_mixed", "int8_mixed_auto"}:
        payload = _payload_for_codec(state_dict, codec="int8_mixed")
        for compressor in _SUPPORTED_COMPRESSORS:
            candidates.append(_wrap_payload(payload, codec="int8_mixed", compressor=compressor))
    elif normalized in {"int4", "int4_mixed", "int4_mixed_auto"}:
        payload = _payload_for_codec(state_dict, codec="int4_mixed")
        for compressor in _SUPPORTED_COMPRESSORS:
            candidates.append(_wrap_payload(payload, codec="int4_mixed", compressor=compressor))
    elif normalized in {"int2", "int2_mixed", "int2_mixed_auto"}:
        payload = _payload_for_codec(state_dict, codec="int2_mixed")
        for compressor in _SUPPORTED_COMPRESSORS:
            candidates.append(_wrap_payload(payload, codec="int2_mixed", compressor=compressor))
    elif normalized == "fp16_enveloped":
        payload = _payload_for_codec(state_dict, codec="fp16_enveloped")
        for compressor in _SUPPORTED_COMPRESSORS:
            candidates.append(_wrap_payload(payload, codec="fp16_enveloped", compressor=compressor))
    else:
        raise ValueError(f"unsupported decoder_state_dict codec: {codec!r}")
    return min(candidates, key=lambda item: len(item[0]))[0]


def deserialize_decoder_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    """Deserialize legacy fp16 or codec-envelope decoder weights."""

    if not blob.startswith(DECODER_STATE_CODEC_MAGIC):
        raw = brotli.decompress(blob)
        sd = pickle.loads(raw)
        if not isinstance(sd, dict):
            raise ValueError("decoder_state_dict legacy blob did not unpickle to a dict")
        return sd
    if len(blob) < _HEADER.size:
        raise ValueError("decoder-state codec envelope too short")
    magic, version, header_len = _HEADER.unpack(blob[: _HEADER.size])
    if magic != DECODER_STATE_CODEC_MAGIC or version != 1:
        raise ValueError("unsupported decoder-state codec envelope")
    header_start = _HEADER.size
    header_end = header_start + int(header_len)
    header = json.loads(blob[header_start:header_end].decode("utf-8"))
    compressed_payload = blob[header_end:]
    payload = _decompress(compressed_payload, compressor=str(header["compressor"]))
    records = pickle.loads(payload)
    if not isinstance(records, dict):
        raise ValueError("decoder-state codec payload did not unpickle to a dict")
    codec = str(header["codec"])
    if codec == "fp16_enveloped":
        return {
            name: torch.from_numpy(np.asarray(value).copy())
            for name, value in records.items()
        }
    if codec == "int8_mixed":
        return {name: _decode_int8_record(record) for name, record in records.items()}
    if codec == "int4_mixed":
        return {
            name: _decode_nbit_record(record, bits=4)
            for name, record in records.items()
        }
    if codec == "int2_mixed":
        return {
            name: _decode_nbit_record(record, bits=2)
            for name, record in records.items()
        }
    raise ValueError(f"unsupported decoder-state payload codec: {codec!r}")


def decoder_state_codec_stats(blob: bytes) -> DecoderStateCodecStats:
    """Return codec byte metadata without decoding tensors."""

    if not blob.startswith(DECODER_STATE_CODEC_MAGIC):
        return DecoderStateCodecStats(
            codec="fp16_brotli_legacy",
            compressor="brotli_q9",
            payload_bytes=len(blob),
            envelope_bytes=0,
        )
    magic, version, header_len = _HEADER.unpack(blob[: _HEADER.size])
    if magic != DECODER_STATE_CODEC_MAGIC or version != 1:
        raise ValueError("unsupported decoder-state codec envelope")
    header_start = _HEADER.size
    header_end = header_start + int(header_len)
    header = json.loads(blob[header_start:header_end].decode("utf-8"))
    return DecoderStateCodecStats(
        codec=str(header["codec"]),
        compressor=str(header["compressor"]),
        payload_bytes=int(header.get("payload_bytes", len(blob) - header_end)),
        envelope_bytes=len(blob),
    )


def _legacy_fp16_brotli(state_dict: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    sd_cpu = {
        name: tensor.detach().to("cpu", dtype=torch.float16).contiguous()
        for name, tensor in state_dict.items()
        if name != "selectors"
    }
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=9))


def _payload_for_codec(state_dict: dict[str, torch.Tensor], *, codec: str) -> bytes:
    if codec == "fp16_enveloped":
        records = {
            name: tensor.detach().to("cpu", dtype=torch.float16).contiguous().numpy()
            for name, tensor in state_dict.items()
            if name != "selectors"
        }
    elif codec == "int8_mixed":
        records = {
            name: _encode_int8_record(tensor.detach().to("cpu", dtype=torch.float32))
            for name, tensor in state_dict.items()
            if name != "selectors"
        }
    elif codec == "int4_mixed":
        records = {
            name: _encode_nbit_record(
                tensor.detach().to("cpu", dtype=torch.float32),
                bits=4,
            )
            for name, tensor in state_dict.items()
            if name != "selectors"
        }
    elif codec == "int2_mixed":
        records = {
            name: _encode_nbit_record(
                tensor.detach().to("cpu", dtype=torch.float32),
                bits=2,
            )
            for name, tensor in state_dict.items()
            if name != "selectors"
        }
    else:
        raise ValueError(f"unsupported payload codec: {codec!r}")
    return pickle.dumps(records, protocol=4)


def _encode_int8_record(tensor: torch.Tensor) -> dict[str, Any]:
    arr = tensor.contiguous().numpy().astype(np.float32, copy=False)
    if arr.size == 0:
        return {"kind": "empty", "shape": list(arr.shape)}
    if arr.ndim >= 2 and arr.shape[0] > 1:
        axis = 0
        reduce_axes = tuple(i for i in range(arr.ndim) if i != axis)
        abs_max = np.max(np.abs(arr), axis=reduce_axes)
        scale = np.where(abs_max > 0.0, abs_max / 127.0, 1.0).astype(np.float16)
        scale32 = scale.astype(np.float32)
        broadcast = [1] * arr.ndim
        broadcast[axis] = arr.shape[axis]
        q = np.round(arr / scale32.reshape(broadcast)).clip(-127, 127).astype(np.int8)
        return {
            "kind": "int8_per_channel_axis0_fp16_scale",
            "shape": list(arr.shape),
            "axis": axis,
            "scale": scale,
            "q": q,
        }
    abs_max = float(np.max(np.abs(arr)))
    scale = np.float16(abs_max / 127.0 if abs_max > 0.0 else 1.0)
    q = np.round(arr / np.float32(scale)).clip(-127, 127).astype(np.int8)
    return {
        "kind": "int8_per_tensor_fp16_scale",
        "shape": list(arr.shape),
        "scale": scale,
        "q": q,
    }


def _decode_int8_record(record: Any) -> torch.Tensor:
    if not isinstance(record, dict):
        raise ValueError("invalid int8 decoder-state record")
    kind = str(record.get("kind"))
    shape = tuple(int(value) for value in record.get("shape", ()))
    if kind == "empty":
        return torch.empty(shape, dtype=torch.float32)
    q = np.asarray(record["q"], dtype=np.int8).reshape(shape)
    if kind == "int8_per_channel_axis0_fp16_scale":
        axis = int(record.get("axis", 0))
        scale = np.asarray(record["scale"], dtype=np.float16).astype(np.float32)
        broadcast = [1] * len(shape)
        broadcast[axis] = shape[axis]
        arr = q.astype(np.float32) * scale.reshape(broadcast)
        return torch.from_numpy(arr.copy())
    if kind == "int8_per_tensor_fp16_scale":
        scale = np.float32(np.asarray(record["scale"], dtype=np.float16))
        return torch.from_numpy((q.astype(np.float32) * scale).copy())
    raise ValueError(f"unsupported int8 decoder-state record kind: {kind!r}")


def _encode_nbit_record(tensor: torch.Tensor, *, bits: int) -> dict[str, Any]:
    if bits not in (2, 4):
        raise ValueError(f"n-bit codec supports int2/int4 only; got {bits}")
    arr = tensor.contiguous().numpy().astype(np.float32, copy=False)
    if arr.size == 0:
        return {"kind": "empty", "shape": list(arr.shape), "bits": bits}
    qmax = (1 << (bits - 1)) - 1
    offset = qmax
    if arr.ndim >= 2 and arr.shape[0] > 1:
        axis = 0
        reduce_axes = tuple(i for i in range(arr.ndim) if i != axis)
        abs_max = np.max(np.abs(arr), axis=reduce_axes)
        scale = np.where(abs_max > 0.0, abs_max / float(qmax), 1.0).astype(
            np.float16
        )
        scale32 = scale.astype(np.float32)
        broadcast = [1] * arr.ndim
        broadcast[axis] = arr.shape[axis]
        q_signed = (
            np.round(arr / scale32.reshape(broadcast))
            .clip(-qmax, qmax)
            .astype(np.int16)
        )
        packed = pack_fixed_width_uints(
            (q_signed.astype(np.int16) + offset).reshape(-1),
            bits=bits,
        )
        return {
            "kind": f"int{bits}_per_channel_axis0_fp16_scale_bitpacked",
            "shape": list(arr.shape),
            "axis": axis,
            "bits": bits,
            "offset": offset,
            "scale": scale,
            "packed_q": packed,
        }
    abs_max = float(np.max(np.abs(arr)))
    scale = np.float16(abs_max / float(qmax) if abs_max > 0.0 else 1.0)
    q_signed = np.round(arr / np.float32(scale)).clip(-qmax, qmax).astype(np.int16)
    packed = pack_fixed_width_uints(q_signed.reshape(-1) + offset, bits=bits)
    return {
        "kind": f"int{bits}_per_tensor_fp16_scale_bitpacked",
        "shape": list(arr.shape),
        "bits": bits,
        "offset": offset,
        "scale": scale,
        "packed_q": packed,
    }


def _decode_nbit_record(record: Any, *, bits: int) -> torch.Tensor:
    if not isinstance(record, dict):
        raise ValueError("invalid n-bit decoder-state record")
    kind = str(record.get("kind"))
    shape = tuple(int(value) for value in record.get("shape", ()))
    if kind == "empty":
        return torch.empty(shape, dtype=torch.float32)
    if int(record.get("bits", bits)) != bits:
        raise ValueError("n-bit decoder-state record bit-width mismatch")
    count = int(np.prod(shape, dtype=np.int64))
    q_unsigned = unpack_fixed_width_uints(
        bytes(record["packed_q"]),
        bits=bits,
        count=count,
    ).astype(np.int16)
    q_signed = (q_unsigned - int(record["offset"])).reshape(shape).astype(np.float32)
    if kind == f"int{bits}_per_channel_axis0_fp16_scale_bitpacked":
        axis = int(record.get("axis", 0))
        scale = np.asarray(record["scale"], dtype=np.float16).astype(np.float32)
        broadcast = [1] * len(shape)
        broadcast[axis] = shape[axis]
        arr = q_signed * scale.reshape(broadcast)
        return torch.from_numpy(arr.copy())
    if kind == f"int{bits}_per_tensor_fp16_scale_bitpacked":
        scale = np.float32(np.asarray(record["scale"], dtype=np.float16))
        return torch.from_numpy((q_signed * scale).copy())
    raise ValueError(f"unsupported n-bit decoder-state record kind: {kind!r}")


def _wrap_payload(
    payload: bytes,
    *,
    codec: str,
    compressor: str,
) -> tuple[bytes, DecoderStateCodecStats]:
    compressed = _compress(payload, compressor=compressor)
    header = {
        "schema": "decoder_state_codec_envelope.v1",
        "codec": codec,
        "compressor": compressor,
        "payload_bytes": len(payload),
        "false_authority": True,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    blob = _HEADER.pack(DECODER_STATE_CODEC_MAGIC, 1, len(header_bytes)) + header_bytes + compressed
    return (
        blob,
        DecoderStateCodecStats(
            codec=codec,
            compressor=compressor,
            payload_bytes=len(payload),
            envelope_bytes=len(blob),
        ),
    )


def _compress(payload: bytes, *, compressor: str) -> bytes:
    if compressor == "brotli_q11":
        return bytes(brotli.compress(payload, quality=11))
    if compressor == "lzma_preset9":
        return lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)
    if compressor == "zlib_level9":
        return zlib.compress(payload, level=9)
    if compressor == "none":
        return payload
    raise ValueError(f"unsupported decoder-state compressor: {compressor!r}")


def _decompress(payload: bytes, *, compressor: str) -> bytes:
    if compressor == "brotli_q11":
        return brotli.decompress(payload)
    if compressor == "lzma_preset9":
        return lzma.decompress(payload)
    if compressor == "zlib_level9":
        return zlib.decompress(payload)
    if compressor == "none":
        return payload
    raise ValueError(f"unsupported decoder-state compressor: {compressor!r}")


__all__ = [
    "DECODER_STATE_CODEC_MAGIC",
    "DecoderStateCodecStats",
    "decoder_state_codec_stats",
    "deserialize_decoder_state_dict",
    "serialize_decoder_state_dict",
]
