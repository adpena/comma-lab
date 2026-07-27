# SPDX-License-Identifier: MIT
"""Exact G105 public-wire quantization surface for NumPy verdicts and MLX QAT.

G105 does not use the trainer's legacy arbitrary-scale int8 realization.  Its
public packet chooses one power-of-two exponent per stacked wire tensor, stores
weights as symmetric int8, stores biases and the palette as symmetric int16,
and stores one shared-exponent int16 Y1 code.  This module deliberately does
not reimplement that authority.

The NumPy surface calls the canonical G105 compiler, emits one canonical Y1
wire family, parses those exact bytes back, and exposes only the parsed values.
The MLX surface places those parsed values in the forward pass with an identity
straight-through estimator.  Consequently:

* the forward value is exactly the public G105 dequantization;
* exponent selection, symmetric range refusal, dtype, stacking, and rounding
  are owned by G105 rather than a second approximation;
* the backward derivative with respect to every admitted source tensor is the
  identity; and
* each call reopens the current source values, so an optimizer step cannot
  silently keep a stale exponent plan.

This is an encoder/trainer integration surface.  It creates no archive, score,
candidate, or pointer claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    VARIANT_ID,
    ExactV9SemanticRootY1ProgramV1,
    V9RuntimeConfigV1,
    V9TensorDTypeV1,
    Y1WireCodecV1,
    compile_from_y1_state,
    encode_packet,
    encode_packet_y1_variant,
    parse_packet,
)

SCHEMA: Final = "tac.g105_public_wire_quantization_surface.v1"
QUANTIZATION_ABI: Final = "tac.g105_pow2_int8_weight_int16_bias_palette_y1.v1"
AUTHORITY: Final = "canonical_G105_compile_encode_parse"
PAIR_COUNT_N600: Final = 600

_LOWER_SHA256 = frozenset("0123456789abcdef")


class G105PublicWireQuantizationError(ValueError):
    """The exact public-wire quantization surface failed closed."""


def _sha256(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise G105PublicWireQuantizationError("quantization receipt is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in _LOWER_SHA256 for character in value):
        raise G105PublicWireQuantizationError(f"{name} must be a lowercase SHA-256")
    return value


def _little_float32(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not array.size or not np.isfinite(array).all():
        raise G105PublicWireQuantizationError(f"{name} must be finite and nonempty")
    return np.ascontiguousarray(array, dtype="<f4")


def _hash_named_float32_state(
    config: V9RuntimeConfigV1,
    params: dict[str, np.ndarray],
    y1_code: np.ndarray,
) -> str:
    """Content-address the exact float32 source consumed by canonical G105."""

    digest = hashlib.sha256()
    config_bytes = _canonical_json(config.to_dict())
    digest.update(struct.pack(">I", len(config_bytes)))
    digest.update(config_bytes)
    for name in sorted(params):
        name_bytes = name.encode("ascii")
        array = _little_float32(params[name], name=name)
        digest.update(struct.pack(">H", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack(">B", array.ndim))
        digest.update(struct.pack(f">{array.ndim}I", *array.shape))
        digest.update(struct.pack(">Q", array.nbytes))
        digest.update(array.tobytes(order="C"))
    code = _little_float32(y1_code, name="y1_code")
    digest.update(struct.pack(">H", len(b"y1_code")))
    digest.update(b"y1_code")
    digest.update(struct.pack(">B", code.ndim))
    digest.update(struct.pack(f">{code.ndim}I", *code.shape))
    digest.update(struct.pack(">Q", code.nbytes))
    digest.update(code.tobytes(order="C"))
    return digest.hexdigest()


def _source_keys_for_wire_tensor(
    wire_name: str,
    *,
    hidden_layer_count: int,
) -> tuple[str, ...]:
    if wire_name in {
        "hidden.weight",
        "hidden.bias",
        "film_pl.weight",
        "film_pl.bias",
        "concat_pl.weight",
        "concat_pl.bias",
    }:
        prefix, suffix = wire_name.split(".", maxsplit=1)
        return tuple(f"{prefix}.{layer_index}.{suffix}" for layer_index in range(hidden_layer_count))
    return (wire_name,)


def _dequantized_sha256(values: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(values, dtype="<f4").tobytes(order="C"))


@dataclass(frozen=True, slots=True)
class G105PublicWireTensorPlanV1:
    """One canonical G105 wire tensor and its trainer-layout source keys."""

    wire_name: str
    source_keys: tuple[str, ...]
    dtype: V9TensorDTypeV1
    shape: tuple[int, ...]
    scale_exponent: int
    quantized_bytes: int
    quantized_sha256: str
    dequantized_sha256: str

    def __post_init__(self) -> None:
        if type(self.wire_name) is not str or not self.wire_name or not self.wire_name.isascii():
            raise G105PublicWireQuantizationError("wire tensor name must be nonempty ASCII")
        if (
            type(self.source_keys) is not tuple
            or not self.source_keys
            or any(type(key) is not str or not key for key in self.source_keys)
        ):
            raise G105PublicWireQuantizationError("wire tensor source keys are malformed")
        if type(self.dtype) is not V9TensorDTypeV1:
            raise G105PublicWireQuantizationError("wire tensor dtype is not typed")
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(value) is not int or value <= 0 for value in self.shape)
        ):
            raise G105PublicWireQuantizationError("wire tensor shape is malformed")
        if type(self.scale_exponent) is not int or not -64 <= self.scale_exponent <= 63:
            raise G105PublicWireQuantizationError("wire tensor exponent is outside G105's canonical range")
        expected_itemsize = 1 if self.dtype is V9TensorDTypeV1.INT8 else 2
        if type(self.quantized_bytes) is not int or self.quantized_bytes != math.prod(self.shape) * expected_itemsize:
            raise G105PublicWireQuantizationError("wire tensor byte count disagrees with shape and dtype")
        _require_sha256(self.quantized_sha256, name="quantized tensor")
        _require_sha256(self.dequantized_sha256, name="dequantized tensor")
        expected_dtype = V9TensorDTypeV1.INT8 if self.wire_name.endswith(".weight") else V9TensorDTypeV1.INT16_LE
        if self.wire_name == "y1_code":
            expected_dtype = V9TensorDTypeV1.INT16_LE
        if self.dtype is not expected_dtype:
            raise G105PublicWireQuantizationError(f"{self.wire_name} dtype does not match the G105 public ABI")

    @property
    def symmetric_limit(self) -> int:
        return 127 if self.dtype is V9TensorDTypeV1.INT8 else 32767

    @property
    def step(self) -> float:
        return math.ldexp(1.0, self.scale_exponent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wire_name": self.wire_name,
            "source_keys": list(self.source_keys),
            "dtype": self.dtype.name,
            "shape": list(self.shape),
            "scale_exponent": self.scale_exponent,
            "symmetric_limit": self.symmetric_limit,
            "step": self.step,
            "quantized_bytes": self.quantized_bytes,
            "quantized_sha256": self.quantized_sha256,
            "dequantized_sha256": self.dequantized_sha256,
        }


@dataclass(frozen=True, slots=True)
class G105PublicWireQuantizationReceiptV1:
    """Typed receipt binding one source state to exact parsed G105 bytes."""

    source_state_sha256: str
    packet_bytes: int
    packet_sha256: str
    y1_wire_codec: Y1WireCodecV1
    tensor_plans: tuple[G105PublicWireTensorPlanV1, ...]
    y1_plan: G105PublicWireTensorPlanV1
    parse_reencode_identical: bool

    def __post_init__(self) -> None:
        _require_sha256(self.source_state_sha256, name="source state")
        _require_sha256(self.packet_sha256, name="G105 packet")
        if type(self.packet_bytes) is not int or self.packet_bytes <= 0:
            raise G105PublicWireQuantizationError("G105 packet byte count must be positive")
        if type(self.y1_wire_codec) is not Y1WireCodecV1:
            raise G105PublicWireQuantizationError("Y1 wire codec is not typed")
        if (
            type(self.tensor_plans) is not tuple
            or not self.tensor_plans
            or any(type(plan) is not G105PublicWireTensorPlanV1 for plan in self.tensor_plans)
        ):
            raise G105PublicWireQuantizationError("tensor plans are malformed")
        if type(self.y1_plan) is not G105PublicWireTensorPlanV1:
            raise G105PublicWireQuantizationError("Y1 plan is malformed")
        if self.y1_plan.wire_name != "y1_code":
            raise G105PublicWireQuantizationError("Y1 plan has the wrong wire name")
        if self.parse_reencode_identical is not True:
            raise G105PublicWireQuantizationError("G105 packet did not parse/re-encode identically")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "quantization_abi": QUANTIZATION_ABI,
            "authority": AUTHORITY,
            "variant_id": VARIANT_ID,
            "source_state_sha256": self.source_state_sha256,
            "packet_bytes": self.packet_bytes,
            "packet_sha256": self.packet_sha256,
            "y1_wire_codec": self.y1_wire_codec.name,
            "tensor_plans": [plan.to_dict() for plan in self.tensor_plans],
            "y1_plan": self.y1_plan.to_dict(),
            "parse_reencode_identical": self.parse_reencode_identical,
            "full_n600": True,
            "requires_fresh_surface_each_optimizer_forward": True,
            "candidate_or_score_claim": False,
            "score_claim": False,
            "pointer_moved": False,
        }

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_json(self._identity_payload()))

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class G105PublicWireQuantizationSurfaceV1:
    """The exact parsed NumPy authority for one G105 public-wire realization."""

    program: ExactV9SemanticRootY1ProgramV1
    packet: bytes = field(repr=False)
    receipt: G105PublicWireQuantizationReceiptV1

    def __post_init__(self) -> None:
        if type(self.program) is not ExactV9SemanticRootY1ProgramV1:
            raise G105PublicWireQuantizationError("surface program is not the exact G105 type")
        if type(self.packet) is not bytes or _sha256(self.packet) != self.receipt.packet_sha256:
            raise G105PublicWireQuantizationError("surface packet disagrees with its receipt")
        if len(self.packet) != self.receipt.packet_bytes:
            raise G105PublicWireQuantizationError("surface packet length disagrees with its receipt")
        if encode_packet(self.program) != self.packet:
            raise G105PublicWireQuantizationError("surface program does not re-emit its packet")

    @property
    def params(self) -> dict[str, np.ndarray]:
        return self.program.params

    @property
    def y1_code(self) -> np.ndarray:
        return self.program.y1_code


@dataclass(frozen=True, slots=True)
class G105PublicWireMLXStateV1:
    """Differentiable MLX values whose forward is the exact parsed G105 state."""

    params: dict[str, Any]
    y1_code: Any
    receipt: G105PublicWireQuantizationReceiptV1


def compile_g105_public_wire_quantization_surface_numpy(
    *,
    config: V9RuntimeConfigV1,
    params: dict[str, np.ndarray],
    y1_code: np.ndarray,
    y1_wire_codec: Y1WireCodecV1 = Y1WireCodecV1.RAW_I16_LE,
) -> G105PublicWireQuantizationSurfaceV1:
    """Compile, encode, parse, and expose the exact public G105 realization."""

    if type(config) is not V9RuntimeConfigV1:
        raise G105PublicWireQuantizationError("config must be V9RuntimeConfigV1")
    if type(params) is not dict:
        raise G105PublicWireQuantizationError("params must be an exact dictionary")
    if type(y1_wire_codec) is not Y1WireCodecV1:
        raise G105PublicWireQuantizationError("Y1 wire codec must be typed")

    source_params = {name: _little_float32(values, name=name) for name, values in params.items()}
    source_y1 = _little_float32(y1_code, name="y1_code")
    program = compile_from_y1_state(
        config=config,
        params=source_params,
        y1_code=source_y1,
    )
    expected_source_keys = set(program.params)
    if set(source_params) != expected_source_keys:
        missing = sorted(expected_source_keys - set(source_params))
        extra = sorted(set(source_params) - expected_source_keys)
        raise G105PublicWireQuantizationError(f"semantic parameter census differs: missing={missing}, extra={extra}")

    packet = encode_packet_y1_variant(program, y1_wire_codec)
    parsed = parse_packet(packet)
    reencoded = encode_packet(parsed)
    if reencoded != packet or parsed.y1_wire_codec is not y1_wire_codec:
        raise G105PublicWireQuantizationError("canonical G105 packet changed under parse/re-encode")

    plans = tuple(
        G105PublicWireTensorPlanV1(
            wire_name=tensor.name,
            source_keys=_source_keys_for_wire_tensor(
                tensor.name,
                hidden_layer_count=config.hidden_layer_count,
            ),
            dtype=tensor.dtype,
            shape=tensor.shape,
            scale_exponent=tensor.scale_exponent,
            quantized_bytes=len(tensor.data),
            quantized_sha256=_sha256(tensor.data),
            dequantized_sha256=_dequantized_sha256(tensor.dequantized),
        )
        for tensor in parsed.tensors
    )
    y1_quantized = np.ascontiguousarray(parsed.y1_code_q, dtype="<i2")
    y1_plan = G105PublicWireTensorPlanV1(
        wire_name="y1_code",
        source_keys=("y1_code",),
        dtype=V9TensorDTypeV1.INT16_LE,
        shape=tuple(int(value) for value in y1_quantized.shape),
        scale_exponent=parsed.y1_code_scale_exponent,
        quantized_bytes=y1_quantized.nbytes,
        quantized_sha256=_sha256(y1_quantized.tobytes(order="C")),
        dequantized_sha256=_dequantized_sha256(parsed.y1_code),
    )
    receipt = G105PublicWireQuantizationReceiptV1(
        source_state_sha256=_hash_named_float32_state(
            config,
            source_params,
            source_y1,
        ),
        packet_bytes=len(packet),
        packet_sha256=_sha256(packet),
        y1_wire_codec=y1_wire_codec,
        tensor_plans=plans,
        y1_plan=y1_plan,
        parse_reencode_identical=True,
    )
    return G105PublicWireQuantizationSurfaceV1(
        program=parsed,
        packet=packet,
        receipt=receipt,
    )


def g105_public_wire_quantize_ste_mlx(
    *,
    config: V9RuntimeConfigV1,
    params: dict[str, Any],
    y1_code: Any,
    y1_wire_codec: Y1WireCodecV1 = Y1WireCodecV1.RAW_I16_LE,
) -> G105PublicWireMLXStateV1:
    """Return exact G105 public values in forward and identity gradients in MLX.

    This function intentionally performs a small device-to-host synchronization
    over the candidate-owned semantic state on every invocation.  That is the
    correctness boundary: the canonical G105 compiler selects value-dependent
    exponents, so caching a plan across optimizer updates could silently score a
    different semantic Y1.  Call this outside ``mx.compile`` and rebuild it for
    every optimizer forward or every exact checkpoint-verdict forward.
    """

    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - exercised on non-MLX hosts
        raise G105PublicWireQuantizationError("MLX is required for the G105 public-wire STE surface") from exc

    if type(params) is not dict:
        raise G105PublicWireQuantizationError("MLX params must be an exact dictionary")
    numpy_params = {name: np.asarray(values, dtype=np.float32) for name, values in params.items()}
    numpy_y1 = np.asarray(y1_code, dtype=np.float32)
    authority = compile_g105_public_wire_quantization_surface_numpy(
        config=config,
        params=numpy_params,
        y1_code=numpy_y1,
        y1_wire_codec=y1_wire_codec,
    )

    result: dict[str, Any] = {}
    parsed_params = authority.params
    if set(params) != set(parsed_params):
        raise G105PublicWireQuantizationError("MLX semantic parameter census differs from parsed G105")
    for name, source in params.items():
        source_f32 = source.astype(mx.float32)
        exact_forward = mx.array(parsed_params[name], dtype=mx.float32)
        result[name] = source_f32 + mx.stop_gradient(exact_forward - source_f32)
    source_y1_f32 = y1_code.astype(mx.float32)
    exact_y1 = mx.array(authority.y1_code, dtype=mx.float32)
    quantized_y1 = source_y1_f32 + mx.stop_gradient(exact_y1 - source_y1_f32)
    return G105PublicWireMLXStateV1(
        params=result,
        y1_code=quantized_y1,
        receipt=authority.receipt,
    )


__all__ = [
    "AUTHORITY",
    "PAIR_COUNT_N600",
    "QUANTIZATION_ABI",
    "SCHEMA",
    "G105PublicWireMLXStateV1",
    "G105PublicWireQuantizationError",
    "G105PublicWireQuantizationReceiptV1",
    "G105PublicWireQuantizationSurfaceV1",
    "G105PublicWireTensorPlanV1",
    "compile_g105_public_wire_quantization_surface_numpy",
    "g105_public_wire_quantize_ste_mlx",
]
