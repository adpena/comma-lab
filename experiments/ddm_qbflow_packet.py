# SPDX-License-Identifier: MIT
"""Deterministic packet and NumPy receiver for the QBFLOW rate-first rung.

This module implements the initialized, untrained object named by the QBFLOW
charter.  It is deliberately scorer-free: the receiver produces signed
interfaces, class logits, two RGB frames, and a separate pose head, but it does
not load SegNet, PoseNet, or any contest video-derived table.

All learned tensors and pair latents are counted.  The only free pieces are the
generic coordinate transform, the fixed class-pair incidence matrix, and the
Road self-conditioning computation derived from the decoder's own coarse
field.  This is a real receiver contract, not a distortion or score claim.
"""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import math
import struct
import zipfile
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import brotli
import numpy as np

MAGIC = b"QBF1"
VERSION = 1
SECTION_CONFIG = 1
SECTION_MODEL = 2
SECTION_LATENT_META = 3
SECTION_LATENTS = 4
SECTION_NAMES = {
    SECTION_CONFIG: "config",
    SECTION_MODEL: "model",
    SECTION_LATENT_META: "latent_meta",
    SECTION_LATENTS: "latents",
}
CODEC_BROTLI_Q11 = 1
CODEC_LZMA9E = 2
CODEC_ZLIB9 = 3
CODEC_NAMES = {
    CODEC_BROTLI_Q11: "brotli_q11",
    CODEC_LZMA9E: "lzma9e",
    CODEC_ZLIB9: "zlib9",
}
CODEC_IDS = {value: key for key, value in CODEC_NAMES.items()}

BASE_FEATURE_DIM = 31
COARSE_DIM = 64
COARSE_FEATURE_DIM = 24
FLOW_DIM = 96
FLOW_LAYERS = 4
ALONG_FEATURE_DIM = 8
N_CLASSES = 5
N_INTERFACES = N_CLASSES * (N_CLASSES - 1) // 2
BOUNDARY_FEATURE_DIM = 16
BOUNDARY_LATENT_DIM = 16
INTERIOR_LATENT_DIM = 12
INTERIOR_DIM = 32
N_POSE = 12

_PACKET_HEADER = struct.Struct(">4sBBH")
_SECTION_HEADER = struct.Struct(">BBHII32sI")
_TENSOR_HEADER = struct.Struct(">BBBBfII")
_LATENT_META = struct.Struct(">4sBBBBff")
_LATENT_TABLE_HEADER = struct.Struct(">4sH")
_LATENT_RECORD_HEADER = struct.Struct(">HBBHHI")
_RESET_RECORD_HEADER = struct.Struct(">4sHII")


class QBFLOWPacketError(RuntimeError):
    """Raised when a QBFLOW packet violates the frozen receiver contract."""


@dataclass(frozen=True, slots=True)
class EncodedSection:
    section_id: int
    codec_id: int
    raw_bytes: int
    payload: bytes
    raw_sha256: str

    @property
    def codec_name(self) -> str:
        return CODEC_NAMES[self.codec_id]


@dataclass(frozen=True, slots=True)
class DecodedPacket:
    sections: Mapping[int, bytes]
    section_facts: tuple[dict[str, Any], ...]


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON with a terminal newline."""

    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def architecture_config(*, num_pairs: int, seed: int) -> dict[str, Any]:
    """Return the counted architecture description used by encoder and receiver."""

    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    return {
        "schema": "ddm_qbflow_architecture.v1",
        "seed": int(seed),
        "num_pairs": int(num_pairs),
        "classes": N_CLASSES,
        "signed_interfaces": N_INTERFACES,
        "base_feature_dim": BASE_FEATURE_DIM,
        "coarse_dim": COARSE_DIM,
        "coarse_feature_dim": COARSE_FEATURE_DIM,
        "flow_dim": FLOW_DIM,
        "flow_layers": FLOW_LAYERS,
        "along_tangent_frequencies": [8, 16, 24, 32],
        "boundary_feature_dim": BOUNDARY_FEATURE_DIM,
        "boundary_latent_dim": BOUNDARY_LATENT_DIM,
        "interior_latent_dim": INTERIOR_LATENT_DIM,
        "interior_dim": INTERIOR_DIM,
        "pose_dim": N_POSE,
        "mechanism": {
            "basis_change": (
                "decoder-self-conditioned Road tangent plus trainable step basis and dedicated along-tangent comb"
            ),
            "objective_change": (
                "future scorer-in-loop realized-through-R joint interface/RGB/pose descent; no fixed class paint"
            ),
            "road_conditioning": ("computed from the decoder's own coarse Road probability; no Road mask payload"),
            "pose_separation": (
                "interior head and pair-interior latent feed RGB and pose12 separately from the signed-interface head"
            ),
        },
        "precision_policy": {
            "hidden_mixing": 8,
            "coordinate_and_flow_inputs": 10,
            "interface_rgb_pose_outputs": 12,
            "step_parameters_and_biases": 16,
            "boundary_latents": 10,
            "interior_latents": 12,
            "status": "PREREGISTERED_ROLE_BASED_NOT_SENSITIVITY_MEASURED",
        },
    }


def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    scale = math.sqrt(2.0 / float(fan_in + fan_out))
    return rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)


def _bias(rng: np.random.Generator, width: int, scale: float = 0.01) -> np.ndarray:
    return rng.normal(0.0, scale, size=(width,)).astype(np.float32)


def initialize_params(seed: int) -> dict[str, np.ndarray]:
    """Initialize every learned tensor used by :func:`reference_forward`.

    The random initial values are retained and counted.  They deliberately
    stress the quantized rate surface more honestly than all-zero placeholders.
    """

    rng = np.random.default_rng(int(seed))
    params: dict[str, np.ndarray] = {}

    coarse_in = BASE_FEATURE_DIM + BOUNDARY_LATENT_DIM + INTERIOR_LATENT_DIM
    params["coarse_in_w"] = _xavier(rng, coarse_in, COARSE_DIM)
    params["coarse_in_b"] = _bias(rng, COARSE_DIM)
    params["coarse_res_w"] = _xavier(rng, COARSE_DIM, COARSE_DIM)
    params["coarse_res_b"] = _bias(rng, COARSE_DIM)
    params["coarse_logits_w"] = _xavier(rng, COARSE_DIM, N_CLASSES)
    params["coarse_logits_b"] = _bias(rng, N_CLASSES)
    params["coarse_feat_w"] = _xavier(rng, COARSE_DIM, COARSE_FEATURE_DIM)
    params["coarse_feat_b"] = _bias(rng, COARSE_FEATURE_DIM)

    flow_in = BASE_FEATURE_DIM + COARSE_FEATURE_DIM + 5 + BOUNDARY_LATENT_DIM
    params["flow_in_w"] = _xavier(rng, flow_in, FLOW_DIM)
    params["flow_in_b"] = _bias(rng, FLOW_DIM)
    for index in range(1, FLOW_LAYERS):
        params[f"flow_res_{index}_w"] = _xavier(rng, FLOW_DIM, FLOW_DIM)
        params[f"flow_res_{index}_b"] = _bias(rng, FLOW_DIM)
    params["flow_film_w"] = _xavier(rng, BOUNDARY_LATENT_DIM, FLOW_LAYERS * 2 * FLOW_DIM)
    params["flow_film_b"] = _bias(rng, FLOW_LAYERS * 2 * FLOW_DIM)
    for index in range(FLOW_LAYERS):
        slopes = rng.normal(1.25, 0.08, size=(FLOW_DIM,)).astype(np.float32)
        centers = np.linspace(-0.75, 0.75, FLOW_DIM, dtype=np.float32)
        centers += rng.normal(0.0, 0.015, size=(FLOW_DIM,)).astype(np.float32)
        params[f"step_slope_{index}"] = slopes
        params[f"step_center_{index}"] = centers
    flow_out = N_INTERFACES + BOUNDARY_FEATURE_DIM
    params["flow_head_w"] = _xavier(rng, FLOW_DIM + ALONG_FEATURE_DIM, flow_out)
    params["flow_head_b"] = _bias(rng, flow_out)

    interior_in = BASE_FEATURE_DIM + COARSE_FEATURE_DIM + INTERIOR_LATENT_DIM
    params["interior_in_w"] = _xavier(rng, interior_in, COARSE_DIM)
    params["interior_in_b"] = _bias(rng, COARSE_DIM)
    params["interior_res_w"] = _xavier(rng, COARSE_DIM, COARSE_DIM)
    params["interior_res_b"] = _bias(rng, COARSE_DIM)
    params["interior_head_w"] = _xavier(rng, COARSE_DIM, INTERIOR_DIM)
    params["interior_head_b"] = _bias(rng, INTERIOR_DIM)

    render_in = INTERIOR_DIM + BOUNDARY_FEATURE_DIM + BOUNDARY_LATENT_DIM + INTERIOR_LATENT_DIM
    params["render_in_w"] = _xavier(rng, render_in, COARSE_DIM)
    params["render_in_b"] = _bias(rng, COARSE_DIM)
    params["render_out_w"] = _xavier(rng, COARSE_DIM, 6)
    params["render_out_b"] = _bias(rng, 6)

    pose_in = INTERIOR_DIM + INTERIOR_LATENT_DIM
    params["pose_in_w"] = _xavier(rng, pose_in, INTERIOR_DIM)
    params["pose_in_b"] = _bias(rng, INTERIOR_DIM)
    params["pose_out_w"] = _xavier(rng, INTERIOR_DIM, N_POSE)
    params["pose_out_b"] = _bias(rng, N_POSE)
    validate_param_shapes(params)
    return params


def initialize_latents(seed: int, num_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialize non-degenerate boundary and interior latents for every pair."""

    rng = np.random.default_rng(int(seed) ^ 0x5142464C)
    boundary = rng.normal(0.0, 0.35, size=(int(num_pairs), BOUNDARY_LATENT_DIM)).astype(np.float32)
    interior = rng.normal(0.0, 0.25, size=(int(num_pairs), INTERIOR_LATENT_DIM)).astype(np.float32)
    return boundary, interior


def expected_param_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {
        "coarse_in_w": (
            BASE_FEATURE_DIM + BOUNDARY_LATENT_DIM + INTERIOR_LATENT_DIM,
            COARSE_DIM,
        ),
        "coarse_in_b": (COARSE_DIM,),
        "coarse_res_w": (COARSE_DIM, COARSE_DIM),
        "coarse_res_b": (COARSE_DIM,),
        "coarse_logits_w": (COARSE_DIM, N_CLASSES),
        "coarse_logits_b": (N_CLASSES,),
        "coarse_feat_w": (COARSE_DIM, COARSE_FEATURE_DIM),
        "coarse_feat_b": (COARSE_FEATURE_DIM,),
        "flow_in_w": (
            BASE_FEATURE_DIM + COARSE_FEATURE_DIM + 5 + BOUNDARY_LATENT_DIM,
            FLOW_DIM,
        ),
        "flow_in_b": (FLOW_DIM,),
        "flow_film_w": (BOUNDARY_LATENT_DIM, FLOW_LAYERS * 2 * FLOW_DIM),
        "flow_film_b": (FLOW_LAYERS * 2 * FLOW_DIM,),
        "flow_head_w": (
            FLOW_DIM + ALONG_FEATURE_DIM,
            N_INTERFACES + BOUNDARY_FEATURE_DIM,
        ),
        "flow_head_b": (N_INTERFACES + BOUNDARY_FEATURE_DIM,),
        "interior_in_w": (
            BASE_FEATURE_DIM + COARSE_FEATURE_DIM + INTERIOR_LATENT_DIM,
            COARSE_DIM,
        ),
        "interior_in_b": (COARSE_DIM,),
        "interior_res_w": (COARSE_DIM, COARSE_DIM),
        "interior_res_b": (COARSE_DIM,),
        "interior_head_w": (COARSE_DIM, INTERIOR_DIM),
        "interior_head_b": (INTERIOR_DIM,),
        "render_in_w": (
            INTERIOR_DIM + BOUNDARY_FEATURE_DIM + BOUNDARY_LATENT_DIM + INTERIOR_LATENT_DIM,
            COARSE_DIM,
        ),
        "render_in_b": (COARSE_DIM,),
        "render_out_w": (COARSE_DIM, 6),
        "render_out_b": (6,),
        "pose_in_w": (INTERIOR_DIM + INTERIOR_LATENT_DIM, INTERIOR_DIM),
        "pose_in_b": (INTERIOR_DIM,),
        "pose_out_w": (INTERIOR_DIM, N_POSE),
        "pose_out_b": (N_POSE,),
    }
    for index in range(1, FLOW_LAYERS):
        shapes[f"flow_res_{index}_w"] = (FLOW_DIM, FLOW_DIM)
        shapes[f"flow_res_{index}_b"] = (FLOW_DIM,)
    for index in range(FLOW_LAYERS):
        shapes[f"step_slope_{index}"] = (FLOW_DIM,)
        shapes[f"step_center_{index}"] = (FLOW_DIM,)
    return shapes


def validate_param_shapes(params: Mapping[str, np.ndarray]) -> None:
    expected = expected_param_shapes()
    if set(params) != set(expected):
        missing = sorted(set(expected) - set(params))
        extra = sorted(set(params) - set(expected))
        raise QBFLOWPacketError(f"model tensor set mismatch: missing={missing} extra={extra}")
    for name, shape in expected.items():
        if tuple(np.asarray(params[name]).shape) != shape:
            raise QBFLOWPacketError(
                f"model tensor shape mismatch for {name}: {np.asarray(params[name]).shape} != {shape}"
            )


def parameter_count(params: Mapping[str, np.ndarray]) -> int:
    validate_param_shapes(params)
    return sum(int(np.asarray(value).size) for value in params.values())


def precision_bits(name: str) -> int:
    """Return the preregistered role-based precision for one learned tensor."""

    if name.endswith("_b") or name.startswith("step_"):
        return 16
    if name in {
        "coarse_in_w",
        "flow_in_w",
    }:
        return 10
    if name in {
        "coarse_logits_w",
        "flow_head_w",
        "render_out_w",
        "pose_out_w",
    }:
        return 12
    return 8


def _pack_signed(values: np.ndarray, bits: int) -> bytes:
    if bits not in {8, 10, 12, 16}:
        raise ValueError(f"unsupported precision: {bits}")
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    qmax = (1 << (bits - 1)) - 1
    if flat.size and (int(flat.min()) < -qmax or int(flat.max()) > qmax):
        raise ValueError("signed value exceeds symmetric quantizer range")
    mask = (1 << bits) - 1
    output = bytearray()
    accumulator = 0
    available = 0
    for value in flat.tolist():
        accumulator |= (int(value) & mask) << available
        available += bits
        while available >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        output.append(accumulator & 0xFF)
    return bytes(output)


def _unpack_signed(payload: bytes, count: int, bits: int) -> np.ndarray:
    if bits not in {8, 10, 12, 16}:
        raise QBFLOWPacketError(f"unsupported precision: {bits}")
    expected = (int(count) * bits + 7) // 8
    if len(payload) != expected:
        raise QBFLOWPacketError(f"packed signed byte count mismatch: {len(payload)} != {expected}")
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    output = np.empty(int(count), dtype=np.int32)
    accumulator = 0
    available = 0
    offset = 0
    for index in range(int(count)):
        while available < bits:
            accumulator |= payload[offset] << available
            available += 8
            offset += 1
        value = accumulator & mask
        accumulator >>= bits
        available -= bits
        if value & sign:
            value -= 1 << bits
        output[index] = value
    if accumulator != 0:
        raise QBFLOWPacketError("non-zero packed signed padding")
    return output


def quantize(array: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    value = np.asarray(array, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("cannot quantize non-finite tensor")
    qmax = (1 << (bits - 1)) - 1
    max_abs = float(np.max(np.abs(value), initial=0.0))
    scale = max_abs / qmax if max_abs > 0.0 else 1.0
    codes = np.rint(value / scale).clip(-qmax, qmax).astype(np.int32)
    return codes, float(np.float32(scale))


def dequantize(codes: np.ndarray, scale: float, shape: Iterable[int]) -> np.ndarray:
    return (np.asarray(codes, dtype=np.float32) * np.float32(scale)).reshape(tuple(shape))


def encode_model(params: Mapping[str, np.ndarray]) -> bytes:
    validate_param_shapes(params)
    output = io.BytesIO()
    output.write(b"QBT1")
    output.write(struct.pack(">H", len(params)))
    for name in sorted(params):
        array = np.asarray(params[name], dtype=np.float32)
        bits = precision_bits(name)
        codes, scale = quantize(array, bits)
        packed = _pack_signed(codes, bits)
        name_bytes = name.encode("ascii")
        if len(name_bytes) > 255 or array.ndim > 255:
            raise ValueError("tensor name or rank exceeds packet limit")
        output.write(
            _TENSOR_HEADER.pack(
                len(name_bytes),
                bits,
                array.ndim,
                0,
                scale,
                int(array.size),
                len(packed),
            )
        )
        output.write(name_bytes)
        for dimension in array.shape:
            if not 0 < int(dimension) <= 65535:
                raise ValueError("tensor dimension exceeds packet limit")
            output.write(struct.pack(">H", int(dimension)))
        output.write(packed)
    return output.getvalue()


def decode_model(payload: bytes) -> dict[str, np.ndarray]:
    view = memoryview(payload)
    if len(view) < 6 or bytes(view[:4]) != b"QBT1":
        raise QBFLOWPacketError("model magic mismatch")
    tensor_count = struct.unpack_from(">H", view, 4)[0]
    offset = 6
    params: dict[str, np.ndarray] = {}
    for _ in range(tensor_count):
        if offset + _TENSOR_HEADER.size > len(view):
            raise QBFLOWPacketError("truncated tensor header")
        name_len, bits, ndim, reserved, scale, count, packed_len = _TENSOR_HEADER.unpack_from(view, offset)
        offset += _TENSOR_HEADER.size
        if reserved != 0 or name_len == 0 or ndim == 0:
            raise QBFLOWPacketError("invalid tensor header")
        end_name = offset + name_len
        end_shape = end_name + 2 * ndim
        end_payload = end_shape + packed_len
        if end_payload > len(view):
            raise QBFLOWPacketError("truncated tensor payload")
        try:
            name = bytes(view[offset:end_name]).decode("ascii")
        except UnicodeDecodeError as exc:
            raise QBFLOWPacketError("tensor name is not ASCII") from exc
        shape = tuple(struct.unpack_from(">H", view, end_name + 2 * index)[0] for index in range(ndim))
        if math.prod(shape) != count:
            raise QBFLOWPacketError("tensor shape/count mismatch")
        codes = _unpack_signed(bytes(view[end_shape:end_payload]), count, bits)
        if name in params:
            raise QBFLOWPacketError(f"duplicate tensor: {name}")
        params[name] = dequantize(codes, scale, shape)
        offset = end_payload
    if offset != len(view):
        raise QBFLOWPacketError("trailing model bytes")
    validate_param_shapes(params)
    return params


def encode_latent_meta(boundary: np.ndarray, interior: np.ndarray) -> tuple[bytes, np.ndarray, np.ndarray]:
    boundary_codes, boundary_scale = quantize(boundary, 10)
    interior_codes, interior_scale = quantize(interior, 12)
    payload = _LATENT_META.pack(
        b"QBM1",
        10,
        12,
        BOUNDARY_LATENT_DIM,
        INTERIOR_LATENT_DIM,
        boundary_scale,
        interior_scale,
    )
    return payload, boundary_codes, interior_codes


def decode_latent_meta(payload: bytes) -> dict[str, Any]:
    if len(payload) != _LATENT_META.size:
        raise QBFLOWPacketError("latent metadata byte count mismatch")
    magic, boundary_bits, interior_bits, boundary_dim, interior_dim, boundary_scale, interior_scale = (
        _LATENT_META.unpack(payload)
    )
    if magic != b"QBM1":
        raise QBFLOWPacketError("latent metadata magic mismatch")
    if (
        boundary_bits != 10
        or interior_bits != 12
        or boundary_dim != BOUNDARY_LATENT_DIM
        or interior_dim != INTERIOR_LATENT_DIM
    ):
        raise QBFLOWPacketError("latent metadata changes the v1 ABI")
    return {
        "boundary_bits": boundary_bits,
        "interior_bits": interior_bits,
        "boundary_dim": boundary_dim,
        "interior_dim": interior_dim,
        "boundary_scale": float(boundary_scale),
        "interior_scale": float(interior_scale),
    }


def encode_latent_record(pair_id: int, boundary_codes: np.ndarray, interior_codes: np.ndarray) -> bytes:
    if not 0 <= int(pair_id) <= 65535:
        raise ValueError("pair_id exceeds packet limit")
    boundary = np.asarray(boundary_codes, dtype=np.int32).reshape(-1)
    interior = np.asarray(interior_codes, dtype=np.int32).reshape(-1)
    if boundary.size != BOUNDARY_LATENT_DIM or interior.size != INTERIOR_LATENT_DIM:
        raise ValueError("latent record shape mismatch")
    boundary_payload = _pack_signed(boundary, 10)
    interior_payload = _pack_signed(interior, 12)
    body = boundary_payload + interior_payload
    return (
        _LATENT_RECORD_HEADER.pack(
            int(pair_id),
            10,
            12,
            len(boundary_payload),
            len(interior_payload),
            zlib.crc32(body) & 0xFFFFFFFF,
        )
        + body
    )


def decode_latent_record(payload: bytes) -> tuple[int, np.ndarray, np.ndarray]:
    if len(payload) < _LATENT_RECORD_HEADER.size:
        raise QBFLOWPacketError("truncated latent record")
    pair_id, boundary_bits, interior_bits, boundary_len, interior_len, crc = _LATENT_RECORD_HEADER.unpack_from(payload)
    if boundary_bits != 10 or interior_bits != 12:
        raise QBFLOWPacketError("latent record precision changes the v1 ABI")
    body = payload[_LATENT_RECORD_HEADER.size :]
    if len(body) != boundary_len + interior_len:
        raise QBFLOWPacketError("latent record length mismatch")
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise QBFLOWPacketError("latent record CRC mismatch")
    boundary = _unpack_signed(body[:boundary_len], BOUNDARY_LATENT_DIM, 10)
    interior = _unpack_signed(body[boundary_len:], INTERIOR_LATENT_DIM, 12)
    return int(pair_id), boundary, interior


def encode_latent_table(pair_ids: Iterable[int], boundary_codes: np.ndarray, interior_codes: np.ndarray) -> bytes:
    ids = [int(value) for value in pair_ids]
    output = io.BytesIO()
    output.write(_LATENT_TABLE_HEADER.pack(b"QBL1", len(ids)))
    for pair_id in ids:
        record = encode_latent_record(pair_id, boundary_codes[pair_id], interior_codes[pair_id])
        output.write(struct.pack(">H", len(record)))
        output.write(record)
    return output.getvalue()


def decode_latent_table(payload: bytes) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if len(payload) < _LATENT_TABLE_HEADER.size:
        raise QBFLOWPacketError("truncated latent table")
    magic, count = _LATENT_TABLE_HEADER.unpack_from(payload)
    if magic != b"QBL1":
        raise QBFLOWPacketError("latent table magic mismatch")
    offset = _LATENT_TABLE_HEADER.size
    records: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _ in range(count):
        if offset + 2 > len(payload):
            raise QBFLOWPacketError("truncated latent table record length")
        length = struct.unpack_from(">H", payload, offset)[0]
        offset += 2
        end = offset + length
        if end > len(payload):
            raise QBFLOWPacketError("truncated latent table record")
        pair_id, boundary, interior = decode_latent_record(payload[offset:end])
        if pair_id in records:
            raise QBFLOWPacketError("duplicate pair latent")
        records[pair_id] = (boundary, interior)
        offset = end
    if offset != len(payload):
        raise QBFLOWPacketError("trailing latent table bytes")
    return records


def encode_reset_record(record: bytes) -> dict[str, bytes]:
    """Encode one independently reset variable record through every real coder."""

    pair_id, _boundary, _interior = decode_latent_record(record)
    candidates: dict[str, bytes] = {}
    for codec_name in CODEC_IDS:
        coded = compress(codec_name, record)
        envelope = (
            _RESET_RECORD_HEADER.pack(b"QBR1", pair_id, len(record), zlib.crc32(record) & 0xFFFFFFFF)
            + bytes([CODEC_IDS[codec_name]])
            + coded
        )
        candidates[codec_name] = envelope
    return candidates


def decode_reset_record(payload: bytes) -> bytes:
    if len(payload) < _RESET_RECORD_HEADER.size + 1:
        raise QBFLOWPacketError("truncated reset record")
    magic, pair_id, raw_len, raw_crc = _RESET_RECORD_HEADER.unpack_from(payload)
    codec_id = payload[_RESET_RECORD_HEADER.size]
    if magic != b"QBR1" or codec_id not in CODEC_NAMES:
        raise QBFLOWPacketError("reset record header mismatch")
    raw = decompress(CODEC_NAMES[codec_id], payload[_RESET_RECORD_HEADER.size + 1 :])
    if len(raw) != raw_len or zlib.crc32(raw) & 0xFFFFFFFF != raw_crc:
        raise QBFLOWPacketError("reset record integrity mismatch")
    decoded_pair, _boundary, _interior = decode_latent_record(raw)
    if decoded_pair != pair_id:
        raise QBFLOWPacketError("reset record pair mismatch")
    return raw


def compress(codec_name: str, raw: bytes) -> bytes:
    if codec_name == "brotli_q11":
        return brotli.compress(raw, quality=11)
    if codec_name == "lzma9e":
        return lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    if codec_name == "zlib9":
        return zlib.compress(raw, level=9)
    raise ValueError(f"unknown coder: {codec_name}")


def decompress(codec_name: str, payload: bytes) -> bytes:
    try:
        if codec_name == "brotli_q11":
            return brotli.decompress(payload)
        if codec_name == "lzma9e":
            return lzma.decompress(payload, format=lzma.FORMAT_XZ)
        if codec_name == "zlib9":
            return zlib.decompress(payload)
    except (brotli.error, lzma.LZMAError, zlib.error) as exc:
        raise QBFLOWPacketError(f"{codec_name} decompression failed") from exc
    raise QBFLOWPacketError(f"unknown coder: {codec_name}")


def encode_section_candidates(section_id: int, raw: bytes) -> dict[str, EncodedSection]:
    if section_id not in SECTION_NAMES:
        raise ValueError(f"unknown section: {section_id}")
    output: dict[str, EncodedSection] = {}
    for codec_name, codec_id in CODEC_IDS.items():
        payload = compress(codec_name, raw)
        if decompress(codec_name, payload) != raw:
            raise QBFLOWPacketError(f"coder roundtrip failed: {codec_name}")
        output[codec_name] = EncodedSection(
            section_id=section_id,
            codec_id=codec_id,
            raw_bytes=len(raw),
            payload=payload,
            raw_sha256=sha256_bytes(raw),
        )
    return output


def choose_section(candidates: Mapping[str, EncodedSection]) -> EncodedSection:
    if set(candidates) != set(CODEC_IDS):
        raise ValueError("section coder race is incomplete")
    return min(candidates.values(), key=lambda row: (len(row.payload), row.codec_id))


def pack_packet(sections: Iterable[EncodedSection]) -> bytes:
    ordered = sorted(sections, key=lambda row: row.section_id)
    if len({row.section_id for row in ordered}) != len(ordered):
        raise ValueError("duplicate packet section")
    output = io.BytesIO()
    output.write(_PACKET_HEADER.pack(MAGIC, VERSION, 0, len(ordered)))
    for section in ordered:
        output.write(
            _SECTION_HEADER.pack(
                section.section_id,
                section.codec_id,
                0,
                section.raw_bytes,
                len(section.payload),
                bytes.fromhex(section.raw_sha256),
                zlib.crc32(section.payload) & 0xFFFFFFFF,
            )
        )
        output.write(section.payload)
    return output.getvalue()


def decode_packet(payload: bytes) -> DecodedPacket:
    view = memoryview(payload)
    if len(view) < _PACKET_HEADER.size:
        raise QBFLOWPacketError("truncated packet header")
    magic, version, flags, section_count = _PACKET_HEADER.unpack_from(view)
    if magic != MAGIC or version != VERSION or flags != 0:
        raise QBFLOWPacketError("packet header mismatch")
    offset = _PACKET_HEADER.size
    sections: dict[int, bytes] = {}
    facts: list[dict[str, Any]] = []
    last_section = 0
    for _ in range(section_count):
        if offset + _SECTION_HEADER.size > len(view):
            raise QBFLOWPacketError("truncated section header")
        section_id, codec_id, reserved, raw_len, coded_len, raw_sha, coded_crc = _SECTION_HEADER.unpack_from(
            view, offset
        )
        offset += _SECTION_HEADER.size
        end = offset + coded_len
        if (
            section_id not in SECTION_NAMES
            or codec_id not in CODEC_NAMES
            or reserved != 0
            or section_id <= last_section
            or end > len(view)
        ):
            raise QBFLOWPacketError("invalid section envelope")
        coded = bytes(view[offset:end])
        if zlib.crc32(coded) & 0xFFFFFFFF != coded_crc:
            raise QBFLOWPacketError("section coded CRC mismatch")
        raw = decompress(CODEC_NAMES[codec_id], coded)
        if len(raw) != raw_len or hashlib.sha256(raw).digest() != raw_sha:
            raise QBFLOWPacketError("section raw integrity mismatch")
        sections[section_id] = raw
        facts.append(
            {
                "section_id": section_id,
                "section_name": SECTION_NAMES[section_id],
                "codec": CODEC_NAMES[codec_id],
                "raw_bytes": raw_len,
                "coded_bytes": coded_len,
                "raw_sha256": raw_sha.hex(),
            }
        )
        last_section = section_id
        offset = end
    if offset != len(view):
        raise QBFLOWPacketError("trailing packet bytes")
    return DecodedPacket(sections=sections, section_facts=tuple(facts))


def mutate_counted_section(payload: bytes, section_id: int) -> bytes:
    """Flip one coded-payload bit in the named section without repairing integrity data."""

    view = memoryview(payload)
    if len(view) < _PACKET_HEADER.size:
        raise ValueError("packet is truncated")
    _magic, _version, _flags, section_count = _PACKET_HEADER.unpack_from(view)
    offset = _PACKET_HEADER.size
    for _ in range(section_count):
        header = _SECTION_HEADER.unpack_from(view, offset)
        current_id, _codec, _reserved, _raw_len, coded_len, _sha, _crc = header
        coded_start = offset + _SECTION_HEADER.size
        coded_end = coded_start + coded_len
        if current_id == section_id:
            if coded_len == 0:
                raise ValueError("cannot mutate empty section")
            mutated = bytearray(payload)
            mutated[coded_start + coded_len // 2] ^= 0x01
            return bytes(mutated)
        offset = coded_end
    raise ValueError(f"section not present: {section_id}")


def deterministic_archive(packet_payload: bytes, member_name: str = "0.qbf") -> bytes:
    """Wrap one already-coded packet in deterministic stored-ZIP framing."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, packet_payload, compress_type=zipfile.ZIP_STORED)
    return output.getvalue()


def read_deterministic_archive(archive_payload: bytes, member_name: str = "0.qbf") -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive_payload), "r") as archive:
        if archive.namelist() != [member_name]:
            raise QBFLOWPacketError("archive member contract mismatch")
        return archive.read(member_name)


def _base_features(h: int, w: int, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys = np.linspace(-1.0, 1.0, int(h), dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, int(w), dtype=np.float32)
    y, x = np.meshgrid(ys, xs, indexing="ij")
    t_field = np.full_like(x, np.float32(t))
    road_center = np.float32(0.08) * np.sin(np.float32(np.pi) * t_field)
    road_half = np.float32(0.18) + np.float32(0.62) * (y + 1.0) * 0.5
    road_u = (x - road_center) / np.maximum(road_half, np.float32(0.05))
    road_v = y + np.float32(0.15)
    road_soft = np.exp(-np.square(road_u)).astype(np.float32)
    features: list[np.ndarray] = [x, y, t_field, road_u, road_v, road_soft]
    for frequency in (1.0, 2.0, 4.0, 8.0):
        for coordinate in (x, y):
            phase = np.float32(np.pi * frequency) * coordinate
            features.extend([np.sin(phase), np.cos(phase)])
    for frequency in (1.0, 2.0, 4.0):
        phase = np.float32(np.pi * frequency) * t_field
        features.extend([np.sin(phase), np.cos(phase)])
    features.extend([x * y, road_u * y, x * x])
    stacked = np.stack(features, axis=-1).astype(np.float32)
    if stacked.shape[-1] != BASE_FEATURE_DIM:
        raise AssertionError(f"base feature dimension drifted: {stacked.shape[-1]}")
    return stacked.reshape(-1, BASE_FEATURE_DIM), x, y


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _step(value: np.ndarray, slope: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.tanh(value * slope.reshape(1, -1) - center.reshape(1, -1))


def _linear(value: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Portable deterministic last-axis linear map without BLAS reordering."""

    return np.einsum("...i,ij->...j", value, weight, optimize=False) + bias


def _incidence() -> np.ndarray:
    matrix = np.zeros((N_INTERFACES, N_CLASSES), dtype=np.float32)
    index = 0
    for left in range(N_CLASSES):
        for right in range(left + 1, N_CLASSES):
            matrix[index, left] = 1.0
            matrix[index, right] = -1.0
            index += 1
    return matrix


def reference_forward(
    params: Mapping[str, np.ndarray],
    boundary_latent: np.ndarray,
    interior_latent: np.ndarray,
    *,
    pair_id: int,
    num_pairs: int,
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    """Execute the real QBFLOW NumPy receiver on one coordinate grid.

    The Road frame is decoder-self-conditioned: the first pass emits a coarse
    Road probability, finite differences generate a tangent, and the second
    pass emits ten signed class interfaces plus renderer conditioning.  RGB and
    pose are separate heads of that same counted object.
    """

    validate_param_shapes(params)
    # Float64 accumulation is the portable NumPy reference. The packet stores
    # quantized float32 values, but platform BLAS float32 reduction order is not
    # the receiver verdict surface.
    p = {name: np.asarray(value, dtype=np.float64) for name, value in params.items()}
    boundary = np.asarray(boundary_latent, dtype=np.float64).reshape(BOUNDARY_LATENT_DIM)
    interior = np.asarray(interior_latent, dtype=np.float64).reshape(INTERIOR_LATENT_DIM)
    if not 0 <= int(pair_id) < int(num_pairs):
        raise ValueError("pair_id is outside num_pairs")
    t = -1.0 + 2.0 * float(pair_id) / max(1, int(num_pairs) - 1)
    base32, x32, y32 = _base_features(height, width, t)
    base = base32.astype(np.float64)
    x = x32.astype(np.float64)
    y = y32.astype(np.float64)
    points = base.shape[0]
    boundary_tiled = np.broadcast_to(boundary, (points, BOUNDARY_LATENT_DIM))
    interior_tiled = np.broadcast_to(interior, (points, INTERIOR_LATENT_DIM))

    coarse_input = np.concatenate([base, boundary_tiled, interior_tiled], axis=1)
    coarse = np.tanh(_linear(coarse_input, p["coarse_in_w"], p["coarse_in_b"]))
    coarse = np.tanh(coarse + _linear(coarse, p["coarse_res_w"], p["coarse_res_b"]))
    coarse_logits = _linear(coarse, p["coarse_logits_w"], p["coarse_logits_b"])
    coarse_features = np.tanh(_linear(coarse, p["coarse_feat_w"], p["coarse_feat_b"]))

    road_probability = _softmax(coarse_logits)[:, 0].reshape(height, width)
    gy, gx = np.gradient(road_probability.astype(np.float32))
    norm = np.sqrt(gx * gx + gy * gy) + np.float32(1e-6)
    tangent_x = (-gy / norm).astype(np.float32)
    tangent_y = (gx / norm).astype(np.float32)
    road_condition = np.stack([road_probability, gx, gy, tangent_x, tangent_y], axis=-1).reshape(points, 5)

    flow_input = np.concatenate([base, coarse_features, road_condition, boundary_tiled], axis=1)
    film = _linear(boundary, p["flow_film_w"], p["flow_film_b"])
    film = film.reshape(FLOW_LAYERS, 2, FLOW_DIM)
    flow = _linear(flow_input, p["flow_in_w"], p["flow_in_b"])
    flow = _step(
        flow * (1.0 + 0.1 * np.tanh(film[0, 0])) + 0.1 * film[0, 1],
        p["step_slope_0"],
        p["step_center_0"],
    )
    for index in range(1, FLOW_LAYERS):
        proposal = _linear(flow, p[f"flow_res_{index}_w"], p[f"flow_res_{index}_b"])
        proposal = proposal * (1.0 + 0.1 * np.tanh(film[index, 0])) + 0.1 * film[index, 1]
        flow = _step(
            flow + proposal,
            p[f"step_slope_{index}"],
            p[f"step_center_{index}"],
        )

    u_tangent = x * tangent_x + y * tangent_y
    along: list[np.ndarray] = []
    for frequency in (8.0, 16.0, 24.0, 32.0):
        phase = np.float32(np.pi * frequency) * u_tangent
        along.extend([np.sin(phase), np.cos(phase)])
    along_features = np.stack(along, axis=-1).reshape(points, ALONG_FEATURE_DIM)
    flow_output = _linear(
        np.concatenate([flow, along_features], axis=1),
        p["flow_head_w"],
        p["flow_head_b"],
    )
    signed_interfaces = flow_output[:, :N_INTERFACES]
    boundary_features = np.tanh(flow_output[:, N_INTERFACES:])
    class_logits = coarse_logits + _linear(signed_interfaces, _incidence().astype(np.float64), np.zeros(N_CLASSES))

    interior_input = np.concatenate([base, coarse_features, interior_tiled], axis=1)
    interior_state = np.tanh(_linear(interior_input, p["interior_in_w"], p["interior_in_b"]))
    interior_state = np.tanh(interior_state + _linear(interior_state, p["interior_res_w"], p["interior_res_b"]))
    interior_features = np.tanh(_linear(interior_state, p["interior_head_w"], p["interior_head_b"]))

    render_input = np.concatenate([interior_features, boundary_features, boundary_tiled, interior_tiled], axis=1)
    render_state = np.tanh(_linear(render_input, p["render_in_w"], p["render_in_b"]))
    rgb_pair = 1.0 / (1.0 + np.exp(-_linear(render_state, p["render_out_w"], p["render_out_b"])))
    pooled_interior = np.mean(interior_features, axis=0)
    pose_input = np.concatenate([pooled_interior, interior], axis=0)
    pose_state = np.tanh(_linear(pose_input, p["pose_in_w"], p["pose_in_b"]))
    pose12 = _linear(pose_state, p["pose_out_w"], p["pose_out_b"])

    outputs = {
        "signed_interfaces": signed_interfaces.reshape(height, width, N_INTERFACES).astype(np.float32),
        "class_logits": class_logits.reshape(height, width, N_CLASSES).astype(np.float32),
        "rgb_pair": rgb_pair.reshape(height, width, 6).astype(np.float32),
        "pose12": np.asarray(pose12, dtype=np.float32),
        "coarse_road_probability": road_probability.astype(np.float32),
        "road_tangent": np.stack([tangent_x, tangent_y], axis=-1).astype(np.float32),
    }
    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise QBFLOWPacketError("receiver produced non-finite output")
    return outputs


__all__ = [
    "BOUNDARY_LATENT_DIM",
    "CODEC_IDS",
    "INTERIOR_LATENT_DIM",
    "SECTION_CONFIG",
    "SECTION_LATENTS",
    "SECTION_LATENT_META",
    "SECTION_MODEL",
    "SECTION_NAMES",
    "DecodedPacket",
    "EncodedSection",
    "QBFLOWPacketError",
    "architecture_config",
    "canonical_json_bytes",
    "choose_section",
    "decode_latent_meta",
    "decode_latent_record",
    "decode_latent_table",
    "decode_model",
    "decode_packet",
    "decode_reset_record",
    "deterministic_archive",
    "encode_latent_meta",
    "encode_latent_record",
    "encode_latent_table",
    "encode_model",
    "encode_reset_record",
    "encode_section_candidates",
    "expected_param_shapes",
    "initialize_latents",
    "initialize_params",
    "mutate_counted_section",
    "pack_packet",
    "parameter_count",
    "precision_bits",
    "read_deterministic_archive",
    "reference_forward",
    "sha256_bytes",
]
