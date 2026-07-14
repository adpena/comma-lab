# SPDX-License-Identifier: MIT
"""Deterministic whole-teacher decision-quotient student.

The student is a throughput *means*: it maps a realized-through-R RGB frame to
the four coordinates of the five-class centered-logit quotient.  Neither this
module nor its local measurements are evaluator or score authority.  Exact
teacher calls are deliberately absent from this module.

NumPy-fp32 is the deterministic reference.  Torch and MLX helpers import their
frameworks lazily, so importing this module never initializes Metal or a scorer.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal

import numpy as np

from tac.scorer_surrogate.frozen_replay_convex_head import (
    ReplayAssignment,
    deterministic_replay_assignments,
)

RESEARCH_ONLY: Final[bool] = True
AUTHORITY_SCOPE: Final[str] = "training_gradient_throughput_means_not_score"
MEASUREMENT_AXIS: Final[str] = (
    "[n600 macOS-MLX advisory; NumPy-fp32 reference; no score authority]"
)
CACHE_SCHEMA: Final[str] = "whole_teacher_distilled_student_cache.v2"
TEACHER_SOURCE_CUSTODY_SCHEMA: Final[str] = (
    "whole_teacher_distilled_student_teacher_source_custody.v1"
)
PARAMETER_SCHEMA: Final[str] = "whole_teacher_distilled_student_parameters.v1"
PARAMETER_BLOB_MAGIC: Final[bytes] = b"WTDSv1\x00"
FIT_DRIVER_STATUS: Final[str] = "EXECUTABLE_CACHED_ONLY_MLX_AUTOGRAD_V1"
FIT_STATE_SCHEMA: Final[str] = "whole_teacher_distilled_student_fit_state.v1"
# v2 adds separate, content-hashed NumPy-authority and MLX-advisory repeat
# streams.  A v1 consumer could otherwise mistake the primary stream for the
# old MLX-only repeat surface.
FIT_RESULT_SCHEMA: Final[str] = "whole_teacher_distilled_student_fit_measurement.v2"
FIT_CHECKPOINT_INTERVAL_STEPS: Final[int] = 40

# CONFIG-PREREGISTERED, UNMEASURED baseline optimizer.  These values are not an
# empirical winner and carry no admission authority; the measured n600 gate
# decides whether any fitted size is useful.
FIT_LEARNING_RATE: Final[float] = 1.0e-3
FIT_ADAM_BETA1: Final[float] = 0.9
FIT_ADAM_BETA2: Final[float] = 0.999
FIT_ADAM_EPSILON: Final[float] = float(np.finfo(np.float32).eps)
FIT_WEIGHT_DECAY: Final[float] = 0.0
FIT_VALUE_WEIGHT: Final[float] = 1.0
FIT_VJP_RELATIVE_L2_WEIGHT: Final[float] = 1.0
FIT_VJP_COSINE_WEIGHT: Final[float] = 1.0

RGB_CHANNELS: Final[int] = 3
CLASS_COUNT: Final[int] = 5
QUOTIENT_DIM: Final[int] = CLASS_COUNT - 1
N600: Final[int] = 600
TRAIN_COUNT: Final[int] = 480
HELDOUT_COUNT: Final[int] = 120
FRAME_SHAPE: Final[tuple[int, ...]] = (1, RGB_CHANNELS, 384, 512)
QUOTIENT_SHAPE: Final[tuple[int, ...]] = (1, QUOTIENT_DIM, 384, 512)
COSTATE_SHAPE: Final[tuple[int, ...]] = FRAME_SHAPE
LABEL_SHAPE: Final[tuple[int, ...]] = (384, 512)
SOURCE_KIND: Final[str] = "rendered_v9_replay"
R_OPERATOR_IDENTITY: Final[str] = (
    "tac.cuda_levelset_training.contest_r:float32_nhwc_0_255_bicubic_874x1164_"
    "round_ste_clamp_0_255_bilinear_384x512_align_corners_false"
)
SEGNET_ARCHITECTURE_IDENTITY: Final[str] = (
    "upstream.modules.SegNet:smp.Unet_tu-efficientnet_b2_classes5_activation_none"
)
SCALAR_OBJECTIVE_IDENTITY: Final[str] = (
    "mean_cross_entropy_of_zero_sum_helmert_lifted_centered_logit_quotient4"
)
COSTATE_SURFACE_IDENTITY: Final[str] = (
    "gradient_wrt_post_contest_r_float32_rgb_nchw_0_255"
)
CACHE_GENERATION_AXIS: Final[str] = (
    "[n600 cache-generation; actual-R; frozen CPU-torch SegNet; no score authority]"
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_R_OPERATOR_SOURCE_PATH: Final[str] = "src/tac/cuda_levelset_training.py"
_SEGNET_SOURCE_PATH: Final[str] = "upstream/modules.py"
_SEGNET_WEIGHTS_PATH: Final[str] = "upstream/models/segnet.safetensors"
_RENDERER_SOURCE_PATH: Final[str] = "tools/dash_comb_probe_n600.py"
_REPLAY_HARNESS_SOURCE_PATH: Final[str] = "tools/probe_frozen_replay_convex_head.py"
_CHECKPOINT_SOURCE_PATHS: Final[tuple[str, ...]] = (
    "experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_BEST.npz",
    "experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_ckpt_stageOctave1_ep251.npz",
    "experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_mlx.npz",
)

CHECKPOINTS: Final[tuple[tuple[str, int], ...]] = (
    ("v9_ep150_ema_best", 150),
    ("v9_ep251_stage_octave1", 251),
    ("v9_ep275_ema_final", 275),
)

# One pointwise stem followed by one depthwise-separable block.  These are
# preregistered configuration sizes, not empirical winners.
STUDENT_SIZE_WIDTHS: Final[dict[str, tuple[int, int]]] = {
    "tiny": (8, 8),
    "small": (12, 16),
    "medium": (16, 24),
}


def _helmert_basis_5x4() -> np.ndarray:
    """Build the fixed orthonormal Helmert contrast basis in float32."""

    basis = np.zeros((CLASS_COUNT, QUOTIENT_DIM), dtype=np.float32)
    for column in range(QUOTIENT_DIM):
        denominator = np.float32(math.sqrt((column + 1) * (column + 2)))
        basis[: column + 1, column] = np.float32(1.0) / denominator
        basis[column + 1, column] = np.float32(-(column + 1)) / denominator
    basis.setflags(write=False)
    return basis


HELMERT_BASIS_5X4: Final[np.ndarray] = _helmert_basis_5x4()
HELMERT_BASIS_SHA256: Final[str] = hashlib.sha256(
    np.ascontiguousarray(HELMERT_BASIS_5X4).tobytes()
).hexdigest()


class StudentContractError(ValueError):
    """A student, cache, metric, or economics invariant failed closed."""


@dataclass(frozen=True)
class StudentArchitecture:
    """Fully explicit architecture for the RGB-to-quotient student."""

    size: Literal["tiny", "small", "medium"]
    widths: tuple[int, int]
    kernel_size: int
    include_coordinates: bool
    frame_value_scale: float

    def validate(self) -> None:
        if self.size not in STUDENT_SIZE_WIDTHS:
            raise StudentContractError(f"unknown student size {self.size!r}")
        if tuple(self.widths) != STUDENT_SIZE_WIDTHS[self.size]:
            raise StudentContractError("student widths drifted from the named size layout")
        if self.kernel_size != 3:
            raise StudentContractError("the sealed student kernel_size must be 3")
        if self.include_coordinates is not True:
            raise StudentContractError("the sealed student requires deterministic XY coordinates")
        if not math.isfinite(self.frame_value_scale) or self.frame_value_scale != 255.0:
            raise StudentContractError("the realized-through-R frame scale must be exactly 255")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["widths"] = list(self.widths)
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> StudentArchitecture:
        required = {
            "size",
            "widths",
            "kernel_size",
            "include_coordinates",
            "frame_value_scale",
        }
        if set(value) != required:
            raise StudentContractError("student architecture keys drifted")
        try:
            architecture = cls(
                size=value["size"],
                widths=tuple(value["widths"]),
                kernel_size=value["kernel_size"],
                include_coordinates=value["include_coordinates"],
                frame_value_scale=value["frame_value_scale"],
            )
        except (KeyError, TypeError) as exc:
            raise StudentContractError("invalid student architecture payload") from exc
        architecture.validate()
        return architecture


@dataclass(frozen=True)
class CachedStudentFitPolicy:
    """Typed cached-only optimizer and Sobolev objective contract.

    The coefficients are a preregistered neutral baseline, not measured
    hyperparameter superiority.  A future policy change changes ``policy_sha``
    and therefore cannot resume an older state silently.
    """

    architecture: StudentArchitecture
    seed: int
    fit_epochs: int
    optimizer: Literal["explicit_adamw"]
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    value_weight: float
    vjp_relative_l2_weight: float
    vjp_cosine_weight: float
    checkpoint_interval_steps: int

    def validate(self) -> None:
        self.architecture.validate()
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise StudentContractError("fit policy seed must be a nonnegative integer")
        if isinstance(self.fit_epochs, bool) or not isinstance(self.fit_epochs, int) or self.fit_epochs < 1:
            raise StudentContractError("fit policy epochs must be an integer >= 1")
        if self.optimizer != "explicit_adamw":
            raise StudentContractError("fit policy optimizer must be explicit_adamw")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise StudentContractError("fit learning_rate must be finite > 0")
        if not math.isfinite(self.beta1) or not 0.0 <= self.beta1 < 1.0:
            raise StudentContractError("fit beta1 must be finite in [0,1)")
        if not math.isfinite(self.beta2) or not 0.0 <= self.beta2 < 1.0:
            raise StudentContractError("fit beta2 must be finite in [0,1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise StudentContractError("fit epsilon must be finite > 0")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise StudentContractError("fit weight_decay must be finite >= 0")
        weights = (
            self.value_weight,
            self.vjp_relative_l2_weight,
            self.vjp_cosine_weight,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in weights) or sum(weights) <= 0.0:
            raise StudentContractError("fit objective weights must be finite, nonnegative, and nonzero")
        if (
            isinstance(self.checkpoint_interval_steps, bool)
            or not isinstance(self.checkpoint_interval_steps, int)
            or self.checkpoint_interval_steps < 1
        ):
            raise StudentContractError("fit checkpoint interval must be an integer >= 1")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "architecture": self.architecture.to_dict(),
            "seed": self.seed,
            "fit_epochs": self.fit_epochs,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "value_weight": self.value_weight,
            "vjp_relative_l2_weight": self.vjp_relative_l2_weight,
            "vjp_cosine_weight": self.vjp_cosine_weight,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "constant_provenance": (
                "CONFIG-PREREGISTERED UNMEASURED neutral baseline; n600 value/VJP and charged-wall "
                "gates retain all authority"
            ),
            "teacher_calls_allowed": 0,
            "synthetic_fallback_allowed": False,
            "cpu_backend_allowed": False,
        }


def cached_student_fit_policy(
    *, architecture: StudentArchitecture, seed: int, fit_epochs: int
) -> CachedStudentFitPolicy:
    """Construct the sealed typed fit policy used by the harness entrypoint."""

    policy = CachedStudentFitPolicy(
        architecture=architecture,
        seed=seed,
        fit_epochs=fit_epochs,
        optimizer="explicit_adamw",
        learning_rate=FIT_LEARNING_RATE,
        beta1=FIT_ADAM_BETA1,
        beta2=FIT_ADAM_BETA2,
        epsilon=FIT_ADAM_EPSILON,
        weight_decay=FIT_WEIGHT_DECAY,
        value_weight=FIT_VALUE_WEIGHT,
        vjp_relative_l2_weight=FIT_VJP_RELATIVE_L2_WEIGHT,
        vjp_cosine_weight=FIT_VJP_COSINE_WEIGHT,
        checkpoint_interval_steps=FIT_CHECKPOINT_INTERVAL_STEPS,
    )
    policy.validate()
    return policy


def architecture_for_size(size: str) -> StudentArchitecture:
    """Return the sealed layout for ``tiny``, ``small``, or ``medium``."""

    if size not in STUDENT_SIZE_WIDTHS:
        raise StudentContractError(f"unknown student size {size!r}")
    return StudentArchitecture(
        size=size,  # type: ignore[arg-type]
        widths=STUDENT_SIZE_WIDTHS[size],
        kernel_size=3,
        include_coordinates=True,
        frame_value_scale=255.0,
    )


def _as_finite_fp32(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise StudentContractError(f"{name} contains nonfinite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _normalized_axis(length: int) -> np.ndarray:
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise StudentContractError("coordinate axis length must be an integer >= 1")
    index = np.arange(length, dtype=np.float32)
    return np.asarray((np.float32(2.0) * index + np.float32(1.0)) / length - 1.0, dtype=np.float32)


def coordinate_channels_numpy(batch: int, height: int, width: int) -> np.ndarray:
    """Return deterministic pixel-center ``(y,x)`` coordinates in ``[-1,1]``."""

    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise StudentContractError("batch must be an integer >= 1")
    y = _normalized_axis(height).reshape(1, 1, height, 1)
    x = _normalized_axis(width).reshape(1, 1, 1, width)
    y_full = np.broadcast_to(y, (batch, 1, height, width))
    x_full = np.broadcast_to(x, (batch, 1, height, width))
    return np.ascontiguousarray(np.concatenate((y_full, x_full), axis=1), dtype=np.float32)


def quotient4_from_logits5_numpy(logits5: Any, *, class_axis: int = 1) -> np.ndarray:
    """Project five logits onto the orthonormal centered-logit quotient."""

    logits = _as_finite_fp32(logits5, name="logits5")
    axis = int(class_axis)
    if not -logits.ndim <= axis < logits.ndim:
        raise StudentContractError("class_axis is outside logits rank")
    axis %= logits.ndim
    if logits.shape[axis] != CLASS_COUNT:
        raise StudentContractError("logits5 class axis must have length five")
    moved = np.moveaxis(logits, axis, -1)
    centered = np.asarray(
        moved - np.mean(moved, axis=-1, keepdims=True, dtype=np.float32),
        dtype=np.float32,
    )
    quotient = np.zeros((*centered.shape[:-1], QUOTIENT_DIM), dtype=np.float32)
    for coordinate in range(QUOTIENT_DIM):
        for klass in range(CLASS_COUNT):
            quotient[..., coordinate] += centered[..., klass] * HELMERT_BASIS_5X4[klass, coordinate]
    return np.ascontiguousarray(np.moveaxis(quotient, -1, axis), dtype=np.float32)


def logits5_from_quotient4_numpy(quotient4: Any, *, class_axis: int = 1) -> np.ndarray:
    """Lift quotient coordinates to their unique zero-sum five-logit representative."""

    quotient = _as_finite_fp32(quotient4, name="quotient4")
    axis = int(class_axis)
    if not -quotient.ndim <= axis < quotient.ndim:
        raise StudentContractError("class_axis is outside quotient rank")
    axis %= quotient.ndim
    if quotient.shape[axis] != QUOTIENT_DIM:
        raise StudentContractError("quotient4 class axis must have length four")
    moved = np.moveaxis(quotient, axis, -1)
    logits = np.zeros((*moved.shape[:-1], CLASS_COUNT), dtype=np.float32)
    for klass in range(CLASS_COUNT):
        for coordinate in range(QUOTIENT_DIM):
            logits[..., klass] += moved[..., coordinate] * HELMERT_BASIS_5X4[klass, coordinate]
    return np.ascontiguousarray(np.moveaxis(logits, -1, axis), dtype=np.float32)


def _parameter_shapes(architecture: StudentArchitecture) -> tuple[tuple[str, tuple[int, ...]], ...]:
    architecture.validate()
    stem_width, block_width = architecture.widths
    input_channels = RGB_CHANNELS + 2
    kernel = architecture.kernel_size
    return (
        ("stem.weight", (input_channels, stem_width)),
        ("stem.bias", (stem_width,)),
        ("block.depthwise.weight", (stem_width, kernel, kernel)),
        ("block.depthwise.bias", (stem_width,)),
        ("block.pointwise.weight", (stem_width, block_width)),
        ("block.pointwise.bias", (block_width,)),
        ("head.weight", (block_width, QUOTIENT_DIM)),
        ("head.bias", (QUOTIENT_DIM,)),
    )


def parameter_layout_sha256(architecture: StudentArchitecture) -> str:
    """Hash exact parameter names, order, shapes, dtype, and architecture."""

    payload = {
        "schema": PARAMETER_SCHEMA,
        "architecture": architecture.to_dict(),
        "parameters": [
            {"name": name, "shape": list(shape), "dtype": "<f4"}
            for name, shape in _parameter_shapes(architecture)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fan_uniform(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    bound = math.sqrt(3.0 / fan_in)
    return np.ascontiguousarray(rng.uniform(-bound, bound, size=shape).astype(np.float32))


def initialize_student_parameters(
    architecture: StudentArchitecture, *, seed: int
) -> dict[str, np.ndarray]:
    """Create a deterministic PCG64/Xavier-uniform float32 parameter mapping."""

    architecture.validate()
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise StudentContractError("seed must be a nonnegative integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    stem_width, block_width = architecture.widths
    kernel = architecture.kernel_size
    parameters = {
        "stem.weight": _fan_uniform(rng, (RGB_CHANNELS + 2, stem_width), RGB_CHANNELS + 2),
        "stem.bias": np.zeros((stem_width,), dtype=np.float32),
        "block.depthwise.weight": _fan_uniform(
            rng, (stem_width, kernel, kernel), kernel * kernel
        ),
        "block.depthwise.bias": np.zeros((stem_width,), dtype=np.float32),
        "block.pointwise.weight": _fan_uniform(
            rng, (stem_width, block_width), stem_width
        ),
        "block.pointwise.bias": np.zeros((block_width,), dtype=np.float32),
        "head.weight": _fan_uniform(rng, (block_width, QUOTIENT_DIM), block_width),
        "head.bias": np.zeros((QUOTIENT_DIM,), dtype=np.float32),
    }
    return validate_student_parameters(architecture, parameters)


def validate_student_parameters(
    architecture: StudentArchitecture, parameters: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Return a copied canonical float32 mapping or reject any layout drift."""

    expected = _parameter_shapes(architecture)
    if set(parameters) != {name for name, _shape in expected}:
        raise StudentContractError("student parameter names drifted from the sealed layout")
    result: dict[str, np.ndarray] = {}
    for name, shape in expected:
        array = _as_finite_fp32(parameters[name], name=name)
        if array.shape != shape:
            raise StudentContractError(f"student parameter {name} has shape {array.shape}, expected {shape}")
        result[name] = array.copy()
    return result


def serialize_student_parameters(
    architecture: StudentArchitecture, parameters: Mapping[str, Any]
) -> bytes:
    """Serialize parameters in a deterministic, framework-independent binary layout."""

    canonical = validate_student_parameters(architecture, parameters)
    entries = []
    body = bytearray()
    for name, shape in _parameter_shapes(architecture):
        array = np.asarray(canonical[name], dtype="<f4", order="C")
        raw = array.tobytes(order="C")
        entries.append(
            {
                "name": name,
                "shape": list(shape),
                "dtype": "<f4",
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        body.extend(raw)
    header = {
        "schema": PARAMETER_SCHEMA,
        "architecture": architecture.to_dict(),
        "layout_sha256": parameter_layout_sha256(architecture),
        "entries": entries,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return PARAMETER_BLOB_MAGIC + struct.pack(">Q", len(header_bytes)) + header_bytes + bytes(body)


def deserialize_student_parameters(payload: bytes) -> tuple[StudentArchitecture, dict[str, np.ndarray]]:
    """Restore and hash-check the explicit parameter blob."""

    if not isinstance(payload, bytes) or not payload.startswith(PARAMETER_BLOB_MAGIC):
        raise StudentContractError("unknown student parameter blob")
    prefix = len(PARAMETER_BLOB_MAGIC)
    if len(payload) < prefix + 8:
        raise StudentContractError("truncated student parameter blob")
    (header_length,) = struct.unpack(">Q", payload[prefix : prefix + 8])
    header_start = prefix + 8
    header_end = header_start + header_length
    if header_end > len(payload):
        raise StudentContractError("truncated student parameter header")
    try:
        header = json.loads(payload[header_start:header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StudentContractError("invalid student parameter header") from exc
    if header.get("schema") != PARAMETER_SCHEMA:
        raise StudentContractError("student parameter schema drifted")
    architecture = StudentArchitecture.from_mapping(header.get("architecture", {}))
    if header.get("layout_sha256") != parameter_layout_sha256(architecture):
        raise StudentContractError("student parameter layout hash drifted")
    expected = _parameter_shapes(architecture)
    entries = header.get("entries")
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise StudentContractError("student parameter entry count drifted")
    cursor = header_end
    parameters: dict[str, np.ndarray] = {}
    for entry, (expected_name, expected_shape) in zip(entries, expected, strict=True):
        if (
            entry.get("name") != expected_name
            or entry.get("shape") != list(expected_shape)
            or entry.get("dtype") != "<f4"
        ):
            raise StudentContractError("student parameter entry metadata drifted")
        nbytes = int(np.prod(expected_shape, dtype=np.int64)) * np.dtype("<f4").itemsize
        if entry.get("nbytes") != nbytes or cursor + nbytes > len(payload):
            raise StudentContractError("student parameter entry is truncated")
        raw = payload[cursor : cursor + nbytes]
        if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
            raise StudentContractError(f"student parameter hash drifted for {expected_name}")
        parameters[expected_name] = np.frombuffer(raw, dtype="<f4").reshape(expected_shape).copy()
        cursor += nbytes
    if cursor != len(payload):
        raise StudentContractError("student parameter blob has trailing bytes")
    return architecture, validate_student_parameters(architecture, parameters)


def _pointwise_forward_numpy(value: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, _channels, height, width = value.shape
    output = np.zeros((batch, weight.shape[1], height, width), dtype=np.float32)
    for input_channel in range(weight.shape[0]):
        for output_channel in range(weight.shape[1]):
            output[:, output_channel] += value[:, input_channel] * weight[input_channel, output_channel]
    output += bias.reshape(1, -1, 1, 1)
    return output


def _pointwise_vjp_numpy(cotangent: np.ndarray, weight: np.ndarray) -> np.ndarray:
    batch, _channels, height, width = cotangent.shape
    result = np.zeros((batch, weight.shape[0], height, width), dtype=np.float32)
    for input_channel in range(weight.shape[0]):
        for output_channel in range(weight.shape[1]):
            result[:, input_channel] += cotangent[:, output_channel] * weight[input_channel, output_channel]
    return result


def _depthwise_forward_numpy(value: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, channels, height, width = value.shape
    kernel = weight.shape[-1]
    pad = kernel // 2
    padded = np.pad(value, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    output = np.zeros((batch, channels, height, width), dtype=np.float32)
    for kh in range(kernel):
        for kw in range(kernel):
            output += padded[:, :, kh : kh + height, kw : kw + width] * weight[:, kh, kw].reshape(
                1, channels, 1, 1
            )
    output += bias.reshape(1, channels, 1, 1)
    return output


def _depthwise_vjp_numpy(cotangent: np.ndarray, weight: np.ndarray) -> np.ndarray:
    batch, channels, height, width = cotangent.shape
    kernel = weight.shape[-1]
    pad = kernel // 2
    padded = np.zeros((batch, channels, height + 2 * pad, width + 2 * pad), dtype=np.float32)
    for kh in range(kernel):
        for kw in range(kernel):
            padded[:, :, kh : kh + height, kw : kw + width] += cotangent * weight[
                :, kh, kw
            ].reshape(1, channels, 1, 1)
    return np.ascontiguousarray(padded[:, :, pad : pad + height, pad : pad + width])


def _validate_frame(frame_nchw: Any) -> np.ndarray:
    frame = _as_finite_fp32(frame_nchw, name="frame_nchw")
    if frame.ndim != 4 or frame.shape[0] < 1 or frame.shape[1] != RGB_CHANNELS:
        raise StudentContractError("frame_nchw must have shape (N,3,H,W)")
    if frame.shape[2] < 1 or frame.shape[3] < 1:
        raise StudentContractError("frame_nchw spatial dimensions must be positive")
    return frame


def _forward_with_cache_numpy(
    frame_nchw: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame = _validate_frame(frame_nchw)
    params = validate_student_parameters(architecture, parameters)
    frame_unit = np.asarray(frame / np.float32(architecture.frame_value_scale), dtype=np.float32)
    coords = coordinate_channels_numpy(frame.shape[0], frame.shape[2], frame.shape[3])
    student_input = np.concatenate((frame_unit, coords), axis=1)
    stem_pre = _pointwise_forward_numpy(student_input, params["stem.weight"], params["stem.bias"])
    stem = np.tanh(stem_pre).astype(np.float32)
    depthwise_pre = _depthwise_forward_numpy(
        stem,
        params["block.depthwise.weight"],
        params["block.depthwise.bias"],
    )
    depthwise = np.tanh(depthwise_pre).astype(np.float32)
    block_pre = _pointwise_forward_numpy(
        depthwise,
        params["block.pointwise.weight"],
        params["block.pointwise.bias"],
    )
    block = np.tanh(block_pre).astype(np.float32)
    quotient = _pointwise_forward_numpy(block, params["head.weight"], params["head.bias"])
    if not np.isfinite(quotient).all():
        raise StudentContractError("student forward emitted nonfinite quotient values")
    return np.ascontiguousarray(quotient), {
        "student_input": student_input,
        "stem": stem,
        "depthwise": depthwise,
        "block": block,
    }


def student_forward_numpy(
    frame_nchw: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    """Run the fixed-order NumPy-fp32 student reference."""

    quotient, _cache = _forward_with_cache_numpy(frame_nchw, architecture, parameters)
    return quotient


def student_input_vjp_numpy(
    frame_nchw: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
    output_cotangent: Any,
) -> np.ndarray:
    """Return ``J_student(frame)^T output_cotangent`` in NumPy-fp32."""

    frame = _validate_frame(frame_nchw)
    params = validate_student_parameters(architecture, parameters)
    quotient, cache = _forward_with_cache_numpy(frame, architecture, params)
    cotangent = _as_finite_fp32(output_cotangent, name="output_cotangent")
    if cotangent.shape != quotient.shape:
        raise StudentContractError("output cotangent shape does not match student quotient")
    gradient = _pointwise_vjp_numpy(cotangent, params["head.weight"])
    gradient *= np.float32(1.0) - cache["block"] * cache["block"]
    gradient = _pointwise_vjp_numpy(gradient, params["block.pointwise.weight"])
    gradient *= np.float32(1.0) - cache["depthwise"] * cache["depthwise"]
    gradient = _depthwise_vjp_numpy(gradient, params["block.depthwise.weight"])
    gradient *= np.float32(1.0) - cache["stem"] * cache["stem"]
    gradient = _pointwise_vjp_numpy(gradient, params["stem.weight"])
    gradient = gradient[:, :RGB_CHANNELS] / np.float32(architecture.frame_value_scale)
    if not np.isfinite(gradient).all():
        raise StudentContractError("student input VJP emitted nonfinite values")
    return np.ascontiguousarray(gradient, dtype=np.float32)


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted, dtype=np.float32)
    return np.asarray(exponential / np.sum(exponential, axis=1, keepdims=True), dtype=np.float32)


def quotient_cross_entropy_cotangent_numpy(quotient4: Any, labels_hw: Any) -> np.ndarray:
    """Return ``d mean(CE(Bz,y)) / dz`` for a quotient field."""

    quotient = _as_finite_fp32(quotient4, name="quotient4")
    if quotient.ndim != 4 or quotient.shape[1] != QUOTIENT_DIM:
        raise StudentContractError("quotient4 must have shape (N,4,H,W)")
    labels = np.asarray(labels_hw)
    if labels.ndim == 2 and quotient.shape[0] == 1:
        labels = labels[None, ...]
    if labels.shape != (quotient.shape[0], quotient.shape[2], quotient.shape[3]):
        raise StudentContractError("labels do not match quotient geometry")
    if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= CLASS_COUNT)):
        raise StudentContractError("labels must be integer class ids in [0,4]")
    logits = logits5_from_quotient4_numpy(quotient)
    residual = _softmax_numpy(logits)
    flat = residual.transpose(0, 2, 3, 1).reshape(-1, CLASS_COUNT)
    flat[np.arange(flat.shape[0]), labels.reshape(-1).astype(np.int64)] -= np.float32(1.0)
    residual = flat.reshape(quotient.shape[0], quotient.shape[2], quotient.shape[3], CLASS_COUNT)
    residual = residual.transpose(0, 3, 1, 2)
    residual /= np.float32(labels.size)
    return quotient4_from_logits5_numpy(residual)


def student_ce_input_vjp_numpy(
    frame_nchw: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
    labels_hw: Any,
) -> np.ndarray:
    """Return the student's mean-CE input costate through its quotient lift."""

    quotient = student_forward_numpy(frame_nchw, architecture, parameters)
    cotangent = quotient_cross_entropy_cotangent_numpy(quotient, labels_hw)
    return student_input_vjp_numpy(frame_nchw, architecture, parameters, cotangent)


def _torch_parameter_tensors(parameters: Mapping[str, Any], *, device: Any, dtype: Any) -> dict[str, Any]:
    import torch

    return {
        name: value.to(device=device, dtype=dtype)
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(np.asarray(value), device=device, dtype=dtype)
        for name, value in parameters.items()
    }


def student_forward_torch(
    frame_nchw: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
) -> Any:
    """Torch-CPU/debug forward with the identical parameter layout."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise StudentContractError("Torch is unavailable for the debug surface") from exc

    architecture.validate()
    frame = frame_nchw if isinstance(frame_nchw, torch.Tensor) else torch.as_tensor(frame_nchw)
    if frame.ndim != 4 or frame.shape[1] != RGB_CHANNELS or not frame.is_floating_point():
        raise StudentContractError("Torch frame must be floating (N,3,H,W)")
    if not bool(torch.isfinite(frame).all()):
        raise StudentContractError("Torch frame contains nonfinite values")
    frame = frame.to(dtype=torch.float32)
    params = _torch_parameter_tensors(parameters, device=frame.device, dtype=frame.dtype)
    validate_shapes = _parameter_shapes(architecture)
    if set(params) != {name for name, _shape in validate_shapes} or any(
        tuple(params[name].shape) != shape for name, shape in validate_shapes
    ):
        raise StudentContractError("Torch parameter layout drifted")
    coords = torch.as_tensor(
        coordinate_channels_numpy(int(frame.shape[0]), int(frame.shape[2]), int(frame.shape[3])),
        device=frame.device,
        dtype=frame.dtype,
    )
    value = torch.cat((frame / architecture.frame_value_scale, coords), dim=1)
    value = torch.tanh(
        functional.conv2d(value, params["stem.weight"].T[:, :, None, None], params["stem.bias"])
    )
    value = torch.tanh(
        functional.conv2d(
            value,
            params["block.depthwise.weight"][:, None, :, :],
            params["block.depthwise.bias"],
            padding=architecture.kernel_size // 2,
            groups=architecture.widths[0],
        )
    )
    value = torch.tanh(
        functional.conv2d(
            value,
            params["block.pointwise.weight"].T[:, :, None, None],
            params["block.pointwise.bias"],
        )
    )
    return functional.conv2d(
        value,
        params["head.weight"].T[:, :, None, None],
        params["head.bias"],
    )


def student_input_vjp_torch(
    frame_nchw: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
    output_cotangent: Any,
) -> Any:
    """Torch debug VJP; returned tensor is detached from the temporary graph."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise StudentContractError("Torch is unavailable for the debug surface") from exc

    frame = frame_nchw if isinstance(frame_nchw, torch.Tensor) else torch.as_tensor(frame_nchw)
    frame = frame.detach().clone().to(dtype=torch.float32).requires_grad_(True)
    output = student_forward_torch(frame, architecture, parameters)
    cotangent = output_cotangent if isinstance(output_cotangent, torch.Tensor) else torch.as_tensor(
        output_cotangent
    )
    cotangent = cotangent.to(device=output.device, dtype=output.dtype)
    if cotangent.shape != output.shape or not bool(torch.isfinite(cotangent).all()):
        raise StudentContractError("Torch output cotangent is invalid")
    (gradient,) = torch.autograd.grad(output, frame, grad_outputs=cotangent)
    return gradient.detach()


def quotient4_from_logits5_torch(logits5: Any, *, class_axis: int = 1) -> Any:
    """Torch differentiable twin of :func:`quotient4_from_logits5_numpy`."""

    import torch

    if logits5.shape[class_axis] != CLASS_COUNT:
        raise StudentContractError("Torch logits class axis must have length five")
    basis = torch.as_tensor(
        HELMERT_BASIS_5X4.copy(), device=logits5.device, dtype=logits5.dtype
    )
    moved = torch.movedim(logits5, class_axis, -1)
    centered = moved - moved.mean(dim=-1, keepdim=True)
    return torch.movedim(centered @ basis, -1, class_axis)


def logits5_from_quotient4_torch(quotient4: Any, *, class_axis: int = 1) -> Any:
    """Torch differentiable zero-sum lift of four quotient coordinates."""

    import torch

    if quotient4.shape[class_axis] != QUOTIENT_DIM:
        raise StudentContractError("Torch quotient class axis must have length four")
    basis = torch.as_tensor(
        HELMERT_BASIS_5X4.copy(), device=quotient4.device, dtype=quotient4.dtype
    )
    moved = torch.movedim(quotient4, class_axis, -1)
    return torch.movedim(moved @ basis.T, -1, class_axis)


def _mlx_pointwise(value: Any, weight: Any, bias: Any, mx: Any) -> Any:
    output = mx.zeros((*value.shape[:-1], weight.shape[1]), dtype=mx.float32)
    for input_channel in range(weight.shape[0]):
        output = output + value[..., input_channel : input_channel + 1] * weight[
            input_channel : input_channel + 1, :
        ]
    return output + bias.reshape(1, 1, 1, -1)


def student_parameters_mlx(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Lazily convert the explicit parameter mapping to an MLX parameter tree."""

    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise StudentContractError("MLX is unavailable") from exc
    return {name: value if type(value).__module__.startswith("mlx") else mx.array(value) for name, value in parameters.items()}


def student_forward_mlx(
    frame_nchw: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
) -> Any:
    """Differentiable MLX forward; importing the module does not initialize MLX."""

    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise StudentContractError("MLX is unavailable") from exc
    architecture.validate()
    frame = frame_nchw if type(frame_nchw).__module__.startswith("mlx") else mx.array(frame_nchw)
    if len(frame.shape) != 4 or int(frame.shape[1]) != RGB_CHANNELS:
        raise StudentContractError("MLX frame must have shape (N,3,H,W)")
    params = student_parameters_mlx(parameters)
    expected = _parameter_shapes(architecture)
    if set(params) != {name for name, _shape in expected} or any(
        tuple(int(v) for v in params[name].shape) != shape for name, shape in expected
    ):
        raise StudentContractError("MLX parameter layout drifted")
    coords_np = coordinate_channels_numpy(int(frame.shape[0]), int(frame.shape[2]), int(frame.shape[3]))
    coords = mx.array(coords_np.transpose(0, 2, 3, 1))
    value = mx.concatenate((mx.transpose(frame, (0, 2, 3, 1)) / architecture.frame_value_scale, coords), axis=-1)
    value = mx.tanh(_mlx_pointwise(value, params["stem.weight"], params["stem.bias"], mx))
    pad = architecture.kernel_size // 2
    padded = mx.pad(value, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    height, width = int(value.shape[1]), int(value.shape[2])
    depthwise = mx.zeros_like(value)
    for kh in range(architecture.kernel_size):
        for kw in range(architecture.kernel_size):
            depthwise = depthwise + padded[:, kh : kh + height, kw : kw + width, :] * params[
                "block.depthwise.weight"
            ][:, kh, kw]
    value = mx.tanh(depthwise + params["block.depthwise.bias"].reshape(1, 1, 1, -1))
    value = mx.tanh(
        _mlx_pointwise(
            value,
            params["block.pointwise.weight"],
            params["block.pointwise.bias"],
            mx,
        )
    )
    quotient = _mlx_pointwise(value, params["head.weight"], params["head.bias"], mx)
    return mx.transpose(quotient, (0, 3, 1, 2))


def student_input_vjp_mlx(
    frame_nchw: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
    output_cotangent: Any,
) -> Any:
    """Differentiable MLX input VJP through the small student."""

    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise StudentContractError("MLX is unavailable") from exc
    frame = frame_nchw if type(frame_nchw).__module__.startswith("mlx") else mx.array(frame_nchw)
    cotangent = (
        output_cotangent
        if type(output_cotangent).__module__.startswith("mlx")
        else mx.array(output_cotangent)
    )
    output, (gradient,) = mx.vjp(
        lambda value: student_forward_mlx(value, architecture, parameters),
        (frame,),
        (cotangent,),
    )
    if output.shape != cotangent.shape:
        raise StudentContractError("MLX output cotangent shape does not match student output")
    return gradient


def student_value_fit_loss_mlx(
    frame_nchw: Any,
    teacher_quotient4: Any,
    boundary_mask: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
) -> Any:
    """MLX boundary-restricted centered-quotient MSE used by a future fit loop.

    This is intentionally a value-fit primitive, not a VJP-admission shortcut.
    The exact held-out input-VJP gate remains mandatory after fitting.
    """

    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise StudentContractError("MLX is unavailable") from exc
    target = (
        teacher_quotient4
        if type(teacher_quotient4).__module__.startswith("mlx")
        else mx.array(teacher_quotient4)
    )
    mask = boundary_mask if type(boundary_mask).__module__.startswith("mlx") else mx.array(boundary_mask)
    prediction = student_forward_mlx(frame_nchw, architecture, parameters)
    if prediction.shape != target.shape:
        raise StudentContractError("MLX teacher quotient shape does not match student output")
    if len(mask.shape) == 2:
        mask = mask[None, None, :, :]
    elif len(mask.shape) == 3:
        mask = mask[:, None, :, :]
    if (
        len(mask.shape) != 4
        or int(mask.shape[0]) != int(prediction.shape[0])
        or int(mask.shape[1]) != 1
        or tuple(int(v) for v in mask.shape[2:]) != tuple(int(v) for v in prediction.shape[2:])
    ):
        raise StudentContractError("MLX boundary mask does not match N,H,W")
    mask = mask.astype(mx.float32)
    denominator = mx.maximum(mx.sum(mask) * QUOTIENT_DIM, mx.array(1.0, dtype=mx.float32))
    return mx.sum(mx.square(prediction - target) * mask) / denominator


def student_value_fit_value_and_grad_mlx(
    frame_nchw: Any,
    teacher_quotient4: Any,
    boundary_mask: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
) -> tuple[Any, Any]:
    """Return the MLX value-fit loss and gradient over the explicit parameter tree."""

    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise StudentContractError("MLX is unavailable") from exc
    parameter_tree = student_parameters_mlx(parameters)
    return mx.value_and_grad(
        lambda tree: student_value_fit_loss_mlx(
            frame_nchw,
            teacher_quotient4,
            boundary_mask,
            architecture,
            tree,
        )
    )(parameter_tree)


def boundary_mask_from_labels_numpy(labels_hw: Any) -> np.ndarray:
    """Return the exact four-neighbour semantic boundary mask.

    No dilation radius or learned threshold is introduced: a cell is selected
    iff at least one in-frame axial neighbour has another class.
    """

    labels = np.asarray(labels_hw)
    if labels.ndim == 2:
        labels = labels[None, ...]
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise StudentContractError("boundary labels must have integer shape (N,H,W) or (H,W)")
    if np.any((labels < 0) | (labels >= CLASS_COUNT)):
        raise StudentContractError("boundary labels must contain class ids in [0,4]")
    mask = np.zeros(labels.shape, dtype=np.bool_)
    vertical = labels[:, 1:, :] != labels[:, :-1, :]
    horizontal = labels[:, :, 1:] != labels[:, :, :-1]
    mask[:, 1:, :] |= vertical
    mask[:, :-1, :] |= vertical
    mask[:, :, 1:] |= horizontal
    mask[:, :, :-1] |= horizontal
    if not np.any(mask):
        raise StudentContractError("boundary-restricted fit requires at least one semantic boundary")
    return mask


def logits5_from_quotient4_mlx(quotient4: Any) -> Any:
    """MLX differentiable Helmert lift with the core's single basis custody."""

    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise StudentContractError("MLX is unavailable") from exc
    if len(quotient4.shape) != 4 or int(quotient4.shape[1]) != QUOTIENT_DIM:
        raise StudentContractError("MLX quotient must have shape (N,4,H,W)")
    basis = mx.array(HELMERT_BASIS_5X4.copy())
    moved = mx.transpose(quotient4, (0, 2, 3, 1))
    return mx.transpose(moved @ mx.transpose(basis), (0, 3, 1, 2))


def _mean_quotient_cross_entropy_mlx(quotient4: Any, labels_nhw: Any, mx: Any) -> Any:
    logits = logits5_from_quotient4_mlx(quotient4)
    labels = labels_nhw
    if len(labels.shape) == 2 and int(logits.shape[0]) == 1:
        labels = labels[None, :, :]
    if tuple(int(v) for v in labels.shape) != (
        int(logits.shape[0]),
        int(logits.shape[2]),
        int(logits.shape[3]),
    ):
        raise StudentContractError("MLX labels do not match quotient geometry")
    moved = mx.transpose(logits, (0, 2, 3, 1))
    maximum = mx.max(moved, axis=-1, keepdims=True)
    logsumexp = mx.log(mx.sum(mx.exp(moved - maximum), axis=-1)) + maximum[..., 0]
    one_hot = (labels[..., None] == mx.arange(CLASS_COUNT)).astype(mx.float32)
    selected = mx.sum(moved * one_hot, axis=-1)
    return mx.mean(logsumexp - selected)


def student_ce_input_vjp_mlx(
    frame_nchw: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
    labels_hw: Any,
) -> Any:
    """Return MLX-autograd ``d mean CE(B q_student,y) / d frame``."""

    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise StudentContractError("MLX is unavailable") from exc
    frame = frame_nchw if type(frame_nchw).__module__.startswith("mlx") else mx.array(frame_nchw)
    labels = labels_hw if type(labels_hw).__module__.startswith("mlx") else mx.array(labels_hw)
    return mx.grad(
        lambda value: _mean_quotient_cross_entropy_mlx(
            student_forward_mlx(value, architecture, parameters), labels, mx
        )
    )(frame)


def _student_sobolev_objective_mlx(
    *,
    mx: Any,
    frame_nchw: Any,
    teacher_quotient4: Any,
    teacher_input_costate: Any,
    labels_nhw: Any,
    boundary_mask_nhw: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
    policy: CachedStudentFitPolicy,
) -> Any:
    """Scalar value+input-VJP objective differentiated only by MLX autograd."""

    prediction = student_forward_mlx(frame_nchw, architecture, parameters)
    mask = boundary_mask_nhw
    if len(mask.shape) == 3:
        mask = mask[:, None, :, :]
    mask = mask.astype(mx.float32)
    floor = mx.array(policy.epsilon, dtype=mx.float32)
    value_difference = (prediction - teacher_quotient4) * mask
    teacher_boundary = teacher_quotient4 * mask
    value_relative_mse = mx.sum(mx.square(value_difference)) / mx.maximum(
        mx.sum(mx.square(teacher_boundary)), floor
    )

    student_costate = student_ce_input_vjp_mlx(
        frame_nchw,
        architecture,
        parameters,
        labels_nhw,
    )
    # Training is boundary-restricted on both value and Sobolev terms.  The
    # post-fit gate below separately compares the unmasked full input vector.
    student_costate_boundary = student_costate * mask
    teacher_costate_boundary = teacher_input_costate * mask
    costate_difference = student_costate_boundary - teacher_costate_boundary
    teacher_square = mx.sum(mx.square(teacher_costate_boundary))
    vjp_relative_l2_square = mx.sum(mx.square(costate_difference)) / mx.maximum(
        teacher_square, floor
    )
    dot = mx.sum(student_costate_boundary * teacher_costate_boundary)
    norm_product = mx.sqrt(
        mx.sum(mx.square(student_costate_boundary)) * teacher_square
    )
    cosine_debt = 1.0 - dot / mx.maximum(norm_product, floor)
    return (
        policy.value_weight * value_relative_mse
        + policy.vjp_relative_l2_weight * vjp_relative_l2_square
        + policy.vjp_cosine_weight * cosine_debt
    )


class _MlxAutogradBackend:
    """Real Metal/MLX backend with explicit, serializable AdamW state."""

    name = "mlx"

    def __init__(self, mx: Any) -> None:
        self.mx = mx
        self.measurement_axis = MEASUREMENT_AXIS
        self.hardware_descriptor = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "mlx_version": str(getattr(mx, "__version__", "unknown")),
            "default_device": str(mx.default_device()),
        }
        self.hardware_fingerprint_sha256 = _semantic_sha256(self.hardware_descriptor)

    @classmethod
    def load(cls) -> _MlxAutogradBackend:
        try:
            import mlx.core as mx
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise StudentContractError("UNMEASURED_BLOCKED-MLX: MLX is unavailable") from exc
        device = mx.default_device()
        if getattr(device, "type", None) != mx.gpu:
            raise StudentContractError(
                "UNMEASURED_BLOCKED-MLX-METAL: fit requires an active MLX GPU; CPU fallback is forbidden"
            )
        return cls(mx)

    def parameters_from_numpy(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        return {name: self.mx.array(np.asarray(value, dtype=np.float32)) for name, value in parameters.items()}

    def zeros_like(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        return {name: self.mx.zeros_like(value) for name, value in parameters.items()}

    def tree_to_numpy(self, tree: Mapping[str, Any]) -> dict[str, np.ndarray]:
        self.mx.eval(*tree.values())
        result = {
            name: np.ascontiguousarray(np.asarray(value), dtype=np.float32)
            for name, value in tree.items()
        }
        if not all(np.isfinite(value).all() for value in result.values()):
            raise StudentContractError("MLX tree contains nonfinite values")
        return result

    def _objective_and_grad(
        self,
        parameters: Mapping[str, Any],
        *,
        frame: np.ndarray,
        teacher_quotient: np.ndarray,
        teacher_costate: np.ndarray,
        labels: np.ndarray,
        boundary_mask: np.ndarray,
        policy: CachedStudentFitPolicy,
    ) -> tuple[Any, Mapping[str, Any]]:
        mx = self.mx
        frame_mx = mx.array(frame)
        quotient_mx = mx.array(teacher_quotient)
        costate_mx = mx.array(teacher_costate)
        labels_mx = mx.array(labels)
        boundary_mx = mx.array(boundary_mask)

        def objective(tree: Mapping[str, Any]) -> Any:
            return _student_sobolev_objective_mlx(
                mx=mx,
                frame_nchw=frame_mx,
                teacher_quotient4=quotient_mx,
                teacher_input_costate=costate_mx,
                labels_nhw=labels_mx,
                boundary_mask_nhw=boundary_mx,
                architecture=policy.architecture,
                parameters=tree,
                policy=policy,
            )

        loss, gradients = mx.value_and_grad(objective)(parameters)
        mx.eval(loss, *gradients.values())
        return loss, gradients

    def warmup_train(
        self,
        parameters: Mapping[str, Any],
        *,
        row: Mapping[str, np.ndarray],
        boundary_mask: np.ndarray,
        policy: CachedStudentFitPolicy,
    ) -> None:
        self._objective_and_grad(
            parameters,
            frame=row["rendered_frame"],
            teacher_quotient=row["teacher_quotient4"],
            teacher_costate=row["teacher_input_costate"],
            labels=row["labels"],
            boundary_mask=boundary_mask,
            policy=policy,
        )

    def train_step(
        self,
        parameters: Mapping[str, Any],
        first_moment: Mapping[str, Any],
        second_moment: Mapping[str, Any],
        *,
        optimizer_step: int,
        row: Mapping[str, np.ndarray],
        boundary_mask: np.ndarray,
        policy: CachedStudentFitPolicy,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, float]:
        started = time.perf_counter()
        loss, gradients = self._objective_and_grad(
            parameters,
            frame=row["rendered_frame"],
            teacher_quotient=row["teacher_quotient4"],
            teacher_costate=row["teacher_input_costate"],
            labels=row["labels"],
            boundary_mask=boundary_mask,
            policy=policy,
        )
        step = optimizer_step + 1
        beta1_power = policy.beta1**step
        beta2_power = policy.beta2**step
        updated_parameters: dict[str, Any] = {}
        updated_first: dict[str, Any] = {}
        updated_second: dict[str, Any] = {}
        for name in parameters:
            gradient = gradients[name]
            first = policy.beta1 * first_moment[name] + (1.0 - policy.beta1) * gradient
            second = policy.beta2 * second_moment[name] + (1.0 - policy.beta2) * self.mx.square(
                gradient
            )
            first_hat = first / (1.0 - beta1_power)
            second_hat = second / (1.0 - beta2_power)
            decayed = parameters[name] * (1.0 - policy.learning_rate * policy.weight_decay)
            updated_parameters[name] = decayed - policy.learning_rate * first_hat / (
                self.mx.sqrt(second_hat) + policy.epsilon
            )
            updated_first[name] = first
            updated_second[name] = second
        self.mx.eval(
            loss,
            *updated_parameters.values(),
            *updated_first.values(),
            *updated_second.values(),
        )
        loss_value = float(np.asarray(loss))
        if not math.isfinite(loss_value):
            raise StudentContractError("MLX Sobolev fit emitted a nonfinite loss")
        # Fail closed on nonfinite optimizer state before any checkpoint calls it durable.
        self.tree_to_numpy(updated_parameters)
        self.tree_to_numpy(updated_first)
        self.tree_to_numpy(updated_second)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return updated_parameters, updated_first, updated_second, loss_value, elapsed_ms

    def predict_and_vjp(
        self,
        parameters: Mapping[str, Any],
        *,
        frame: np.ndarray,
        labels: np.ndarray,
        architecture: StudentArchitecture,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        mx = self.mx
        frame_mx = mx.array(frame)
        labels_mx = mx.array(labels)
        started = time.perf_counter()
        quotient = student_forward_mlx(frame_mx, architecture, parameters)
        mx.eval(quotient)
        forward_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        input_vjp = student_ce_input_vjp_mlx(frame_mx, architecture, parameters, labels_mx)
        mx.eval(input_vjp)
        forward_vjp_ms = (time.perf_counter() - started) * 1000.0
        quotient_numpy = np.ascontiguousarray(np.asarray(quotient), dtype=np.float32)
        vjp_numpy = np.ascontiguousarray(np.asarray(input_vjp), dtype=np.float32)
        if not np.isfinite(quotient_numpy).all() or not np.isfinite(vjp_numpy).all():
            raise StudentContractError("MLX heldout forward/VJP emitted nonfinite values")
        return quotient_numpy, vjp_numpy, forward_ms, forward_vjp_ms


def _load_mlx_backend() -> _MlxAutogradBackend:
    """Single monkeypatchable loader; production always returns real MLX autograd."""

    return _MlxAutogradBackend.load()


def _vector_fidelity(reference: Any, candidate: Any) -> dict[str, float | int]:
    ref = _as_finite_fp32(reference, name="reference").astype(np.float64).reshape(-1)
    cand = _as_finite_fp32(candidate, name="candidate").astype(np.float64).reshape(-1)
    if ref.shape != cand.shape or ref.size == 0:
        raise StudentContractError("fidelity vectors must be nonempty and shape-matched")
    ref_norm = float(np.linalg.norm(ref))
    cand_norm = float(np.linalg.norm(cand))
    if ref_norm == 0.0 or cand_norm == 0.0:
        raise StudentContractError(
            "cosine/relative-L2 fidelity is undefined for a zero teacher or student vector"
        )
    dot = float(np.dot(ref, cand))
    cosine = float(np.clip(dot / (ref_norm * cand_norm), -1.0, 1.0))
    relative_l2 = float(np.linalg.norm(cand - ref) / ref_norm)
    return {
        "compared_elements": int(ref.size),
        "cosine": cosine,
        "relative_l2": relative_l2,
        "reference_l2": ref_norm,
        "candidate_l2": cand_norm,
    }


def forward_pair_metrics(
    assignment_id: str, teacher_quotient4: Any, student_quotient4: Any
) -> dict[str, Any]:
    """Measure one pair's quotient value fidelity and decision preservation."""

    if not isinstance(assignment_id, str) or not assignment_id:
        raise StudentContractError("assignment_id must be a nonempty string")
    teacher = _as_finite_fp32(teacher_quotient4, name="teacher_quotient4")
    student = _as_finite_fp32(student_quotient4, name="student_quotient4")
    if teacher.shape != student.shape or teacher.ndim != 4 or teacher.shape[1] != QUOTIENT_DIM:
        raise StudentContractError("forward quotient fields must be matching (N,4,H,W)")
    vector = _vector_fidelity(teacher, student)
    error = np.abs(student.astype(np.float64) - teacher.astype(np.float64))
    teacher_logits = logits5_from_quotient4_numpy(teacher)
    student_logits = logits5_from_quotient4_numpy(student)
    sign_matches = []
    for left in range(CLASS_COUNT):
        for right in range(left + 1, CLASS_COUNT):
            sign_matches.append(
                np.sign(teacher_logits[:, left] - teacher_logits[:, right])
                == np.sign(student_logits[:, left] - student_logits[:, right])
            )
    margin_sign_agreement = float(np.mean(np.stack(sign_matches, axis=0)))
    argmax_disagreement = float(
        np.mean(np.argmax(teacher_logits, axis=1) != np.argmax(student_logits, axis=1))
    )
    return {
        "assignment_id": assignment_id,
        **vector,
        "mean_absolute_error": float(np.mean(error)),
        "max_absolute_error": float(np.max(error)),
        "margin_sign_agreement": margin_sign_agreement,
        "argmax_disagreement": argmax_disagreement,
    }


def vjp_pair_metrics(
    assignment_id: str,
    teacher_input_costate: Any,
    student_input_costate: Any,
    *,
    boundary_mask: Any | None = None,
) -> dict[str, Any]:
    """Measure decisive full-vector input-VJP fidelity plus optional boundary diagnostic."""

    if not isinstance(assignment_id, str) or not assignment_id:
        raise StudentContractError("assignment_id must be a nonempty string")
    teacher = _as_finite_fp32(teacher_input_costate, name="teacher_input_costate")
    student = _as_finite_fp32(student_input_costate, name="student_input_costate")
    if teacher.shape != student.shape or teacher.ndim != 4 or teacher.shape[1] != RGB_CHANNELS:
        raise StudentContractError("input costates must be matching (N,3,H,W)")
    result: dict[str, Any] = {"assignment_id": assignment_id, **_vector_fidelity(teacher, student)}
    if boundary_mask is not None:
        mask = np.asarray(boundary_mask, dtype=np.bool_)
        if mask.ndim == 2 and teacher.shape[0] == 1:
            mask = mask[None, ...]
        if mask.shape != (teacher.shape[0], teacher.shape[2], teacher.shape[3]) or not np.any(mask):
            raise StudentContractError("boundary mask must be nonempty and match N,H,W")
        selected = np.broadcast_to(mask[:, None], teacher.shape)
        boundary = _vector_fidelity(teacher[selected], student[selected])
        result["boundary_diagnostic"] = boundary
    return result


def _aggregate_pair_rows(rows: Sequence[Mapping[str, Any]], *, kind: str) -> dict[str, Any]:
    if not rows:
        raise StudentContractError("at least one pair metric row is required")
    ids = [row.get("assignment_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise StudentContractError("pair metric assignment ids must be unique nonempty strings")
    required = ("relative_l2", "cosine")
    if kind == "forward":
        required += (
            "mean_absolute_error",
            "max_absolute_error",
            "margin_sign_agreement",
            "argmax_disagreement",
        )
    for row in rows:
        if any(key not in row or not math.isfinite(float(row[key])) for key in required):
            raise StudentContractError("pair metric row is incomplete or nonfinite")
    summary: dict[str, Any] = {"pair_count": len(rows), "kind": kind}
    lower_is_worse = {"cosine", "margin_sign_agreement"}
    for metric in required:
        values = [float(row[metric]) for row in rows]
        index = int(np.argmin(values) if metric in lower_is_worse else np.argmax(values))
        summary[f"mean_{metric}"] = float(np.mean(values, dtype=np.float64))
        summary[f"worst_{metric}"] = values[index]
        summary[f"worst_{metric}_assignment_id"] = ids[index]
    return summary


def aggregate_forward_pair_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate forward metrics without hiding a single bad worst pair."""

    return _aggregate_pair_rows(rows, kind="forward")


def aggregate_vjp_pair_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate decisive full-vector VJP metrics without averaging away failure."""

    return _aggregate_pair_rows(rows, kind="vjp")


@dataclass(frozen=True)
class CustodyFile:
    """One content-bound source or sidecar file named by semantic custody."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class TeacherSourceCustody:
    """Validated semantic authority for every cached quotient and costate."""

    custody_sha256: str
    r_operator_sha256: str
    scorer_sha256: str
    scalar_objective_sha256: str
    replay_source_sha256: str
    generation_sha256: str
    helmert_basis_sha256: str
    post_r_input_surface_sha256: str
    generation_axis: str
    repository_files: tuple[CustodyFile, ...]
    bundle_sidecars: tuple[CustodyFile, ...]


@dataclass(frozen=True)
class CacheArtifact:
    path: str
    bytes: int
    sha256: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class CacheRow:
    assignment_id: str
    pair_index: int
    checkpoint_index: int
    checkpoint_name: str
    checkpoint_epoch: int
    split: Literal["train", "heldout"]
    artifacts: Mapping[str, CacheArtifact]


@dataclass(frozen=True)
class ValidatedN600Cache:
    """Hash-verified, exact-assignment n600 bundle safe for cached-only fitting."""

    bundle_root: Path
    manifest_sha256: str
    teacher_source_custody: TeacherSourceCustody
    rows: tuple[CacheRow, ...]

    @property
    def train_rows(self) -> tuple[CacheRow, ...]:
        return tuple(row for row in self.rows if row.split == "train")

    @property
    def heldout_rows(self) -> tuple[CacheRow, ...]:
        return tuple(row for row in self.rows if row.split == "heldout")

    def load_row(self, pair_index: int) -> dict[str, np.ndarray]:
        """Load one already-hash-verified row and recheck dtype/shape."""

        matches = [row for row in self.rows if row.pair_index == pair_index]
        if len(matches) != 1:
            raise StudentContractError(f"pair_index {pair_index} is absent or duplicated")
        row = matches[0]
        result: dict[str, np.ndarray] = {}
        for name, artifact in row.artifacts.items():
            path = self.bundle_root / artifact.path
            array = np.load(path, allow_pickle=False)
            if not isinstance(array, np.ndarray):
                raise StudentContractError(f"cache artifact {name} is not a standalone NPY array")
            if array.dtype.str != artifact.dtype or tuple(array.shape) != artifact.shape:
                raise StudentContractError(f"cache artifact {name} header drifted after validation")
            result[name] = np.asarray(array)
        return result


def _expected_assignments() -> tuple[ReplayAssignment, ...]:
    return deterministic_replay_assignments(
        n_pairs=N600,
        checkpoint_names=tuple(name for name, _epoch in CHECKPOINTS),
        holdout_period=5,
        seed=455,
    )


_ARTIFACT_SPECS: Final[dict[str, tuple[str, tuple[int, ...]]]] = {
    "rendered_frame": ("<f4", FRAME_SHAPE),
    "teacher_quotient4": ("<f4", QUOTIENT_SHAPE),
    "teacher_input_costate": ("<f4", COSTATE_SHAPE),
    "labels": ("<i8", LABEL_SHAPE),
}


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _post_r_input_surface_sha256(r_operator_sha256: str) -> str:
    return _semantic_sha256(
        {
            "surface_identity": COSTATE_SURFACE_IDENTITY,
            "tensor_shape": list(FRAME_SHAPE),
            "dtype": "<f4",
            "layout": "NCHW_RGB",
            "value_domain": [0.0, 255.0],
            "gradient_units": (
                "d_mean_ce_per_post_r_rgb_code_value_where_one_unit_equals_one_of_255"
            ),
            "r_operator_sha256": r_operator_sha256,
        }
    )


def _parse_content_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StudentContractError(f"{name} sha256 is invalid")
    if len(set(value)) == 1 or value == hashlib.sha256(b"").hexdigest():
        raise StudentContractError(f"{name} sha256 is a generic placeholder")
    return value


def _safe_relative_path(value: Any, *, name: str, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise StudentContractError(f"{name} path must be nonempty and relative")
    pure = PurePosixPath(value)
    if ".." in pure.parts or (suffix is not None and pure.suffix != suffix):
        raise StudentContractError(f"{name} path is unsafe or has the wrong suffix")
    return value


def _parse_custody_file(
    value: Any,
    *,
    name: str,
    expected_path: str | None = None,
    suffix: str | None = None,
) -> CustodyFile:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise StudentContractError(f"{name} file custody keys drifted")
    path = _safe_relative_path(value["path"], name=name, suffix=suffix)
    if expected_path is not None and path != expected_path:
        raise StudentContractError(f"{name} source path drifted")
    byte_count = value["bytes"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise StudentContractError(f"{name} bytes must be a positive integer")
    sha256 = _parse_content_sha256(value["sha256"], name=name)
    return CustodyFile(path=path, bytes=byte_count, sha256=sha256)


def _verify_semantic_hash(value: Mapping[str, Any], *, key: str, name: str) -> str:
    observed = _parse_content_sha256(value.get(key), name=name)
    body = {field: item for field, item in value.items() if field != key}
    if observed != _semantic_sha256(body):
        raise StudentContractError(f"{name} semantic hash drifted")
    return observed


def _expected_assignment_sha256() -> str:
    return _semantic_sha256([assignment.to_dict() for assignment in _expected_assignments()])


def _parse_teacher_source_custody(value: Any) -> TeacherSourceCustody:
    """Validate semantic authority, not merely the raw tensor byte hashes."""

    required = {
        "schema",
        "r_operator",
        "frozen_segnet",
        "scalar_objective",
        "replay_source",
        "generation",
        "custody_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise StudentContractError("teacher source custody keys drifted")
    if value["schema"] != TEACHER_SOURCE_CUSTODY_SCHEMA:
        raise StudentContractError("teacher source custody schema drifted")

    r_operator = value["r_operator"]
    r_required = {
        "identity",
        "source",
        "camera_hw",
        "output_hw",
        "up_interpolation",
        "quantization",
        "down_interpolation",
        "align_corners",
        "input_surface",
        "output_surface",
        "r_operator_sha256",
    }
    if not isinstance(r_operator, Mapping) or set(r_operator) != r_required:
        raise StudentContractError("R operator custody keys drifted")
    if (
        r_operator["identity"] != R_OPERATOR_IDENTITY
        or r_operator["camera_hw"] != [874, 1164]
        or r_operator["output_hw"] != [384, 512]
        or r_operator["up_interpolation"] != "bicubic"
        or r_operator["quantization"] != "round_ste_then_clamp_0_255"
        or r_operator["down_interpolation"] != "bilinear"
        or r_operator["align_corners"] is not False
        or r_operator["input_surface"] != "float32_rgb_nhwc_0_255"
        or r_operator["output_surface"] != "post_r_float32_rgb_nhwc_0_255"
    ):
        raise StudentContractError("actual R operator semantics drifted")
    r_source = _parse_custody_file(
        r_operator["source"],
        name="R operator source",
        expected_path=_R_OPERATOR_SOURCE_PATH,
        suffix=".py",
    )
    r_operator_sha256 = _verify_semantic_hash(
        r_operator, key="r_operator_sha256", name="R operator"
    )

    scorer = value["frozen_segnet"]
    scorer_required = {
        "architecture_identity",
        "architecture_source",
        "weights",
        "frozen",
        "class_count",
        "input_surface",
        "logit_surface",
        "preprocess_identity",
        "scorer_sha256",
    }
    if not isinstance(scorer, Mapping) or set(scorer) != scorer_required:
        raise StudentContractError("frozen SegNet custody keys drifted")
    if (
        scorer["architecture_identity"] != SEGNET_ARCHITECTURE_IDENTITY
        or scorer["frozen"] is not True
        or scorer["class_count"] != CLASS_COUNT
        or scorer["input_surface"] != "post_r_float32_rgb_nchw_0_255_384x512"
        or scorer["logit_surface"] != "float32_logits_nchw_5x384x512"
        or scorer["preprocess_identity"]
        != "same_state_last_frame_then_bilinear_384x512_align_corners_false"
    ):
        raise StudentContractError("frozen SegNet architecture or preprocessing drifted")
    segnet_source = _parse_custody_file(
        scorer["architecture_source"],
        name="SegNet architecture source",
        expected_path=_SEGNET_SOURCE_PATH,
        suffix=".py",
    )
    segnet_weights = _parse_custody_file(
        scorer["weights"],
        name="SegNet weights",
        expected_path=_SEGNET_WEIGHTS_PATH,
        suffix=".safetensors",
    )
    scorer_sha256 = _verify_semantic_hash(
        scorer, key="scorer_sha256", name="frozen SegNet scorer"
    )

    objective = value["scalar_objective"]
    objective_required = {
        "identity",
        "target",
        "quotient_basis_version",
        "quotient_basis_sha256",
        "logit_lift",
        "loss",
        "reduction",
        "label_semantics",
        "class_count",
        "costate_surface",
        "costate_units",
        "teacher_value_artifact",
        "teacher_costate_artifact",
        "same_scalar_for_value_and_costate",
        "post_r_input_surface_sha256",
        "r_operator_sha256",
        "scorer_sha256",
        "scalar_objective_sha256",
    }
    if not isinstance(objective, Mapping) or set(objective) != objective_required:
        raise StudentContractError("scalar objective custody keys drifted")
    if (
        objective["identity"] != SCALAR_OBJECTIVE_IDENTITY
        or objective["target"] != "centered_logit_decision_quotient_4d"
        or objective["quotient_basis_version"] != "orthonormal_helmert_5x4_v1"
        or objective["quotient_basis_sha256"] != HELMERT_BASIS_SHA256
        or objective["logit_lift"] != "zero_sum_helmert_lift_4_to_5"
        or objective["loss"] != "cross_entropy"
        or objective["reduction"] != "mean_over_batch_height_width"
        or objective["label_semantics"] != "same_replay_integer_class_ids_0_to_4"
        or objective["class_count"] != CLASS_COUNT
        or objective["costate_surface"] != COSTATE_SURFACE_IDENTITY
        or objective["costate_units"]
        != "d_mean_ce_per_post_r_rgb_code_value_where_one_unit_equals_one_of_255"
        or objective["teacher_value_artifact"] != "teacher_quotient4"
        or objective["teacher_costate_artifact"] != "teacher_input_costate"
        or objective["same_scalar_for_value_and_costate"] is not True
        or objective["post_r_input_surface_sha256"]
        != _post_r_input_surface_sha256(r_operator_sha256)
        or objective["r_operator_sha256"] != r_operator_sha256
        or objective["scorer_sha256"] != scorer_sha256
    ):
        raise StudentContractError(
            "teacher objective/reduction/label/basis/costate surface semantics drifted"
        )
    scalar_objective_sha256 = _verify_semantic_hash(
        objective, key="scalar_objective_sha256", name="scalar objective"
    )

    replay = value["replay_source"]
    replay_required = {
        "source_kind",
        "renderer_identity",
        "renderer_source",
        "replay_harness_source",
        "renderer_config",
        "source_manifest",
        "upstream_source_manifest",
        "checkpoint_custody",
        "assignment_sha256",
        "r_operator_sha256",
        "scorer_sha256",
        "replay_source_sha256",
    }
    if not isinstance(replay, Mapping) or set(replay) != replay_required:
        raise StudentContractError("replay source custody keys drifted")
    if (
        replay["source_kind"] != SOURCE_KIND
        or replay["renderer_identity"]
        != "tools.dash_comb_probe_n600.Renderer_plus_task455_differentiable_chart"
        or replay["assignment_sha256"] != _expected_assignment_sha256()
        or replay["r_operator_sha256"] != r_operator_sha256
        or replay["scorer_sha256"] != scorer_sha256
    ):
        raise StudentContractError("replay renderer/source/assignment semantics drifted")
    renderer_source = _parse_custody_file(
        replay["renderer_source"],
        name="renderer source",
        expected_path=_RENDERER_SOURCE_PATH,
        suffix=".py",
    )
    replay_harness_source = _parse_custody_file(
        replay["replay_harness_source"],
        name="replay harness source",
        expected_path=_REPLAY_HARNESS_SOURCE_PATH,
        suffix=".py",
    )
    renderer_config = _parse_custody_file(
        replay["renderer_config"], name="renderer config", suffix=".json"
    )
    source_manifest = _parse_custody_file(
        replay["source_manifest"], name="source manifest", suffix=".json"
    )
    upstream_source_manifest = _parse_custody_file(
        replay["upstream_source_manifest"],
        name="upstream source manifest",
        suffix=".json",
    )
    raw_checkpoints = replay["checkpoint_custody"]
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != len(CHECKPOINTS):
        raise StudentContractError("replay checkpoint custody must preserve all three states")
    checkpoint_files: list[CustodyFile] = []
    checkpoint_hashes: set[str] = set()
    for raw_checkpoint, (expected_name, expected_epoch), expected_path in zip(
        raw_checkpoints, CHECKPOINTS, _CHECKPOINT_SOURCE_PATHS, strict=True
    ):
        if not isinstance(raw_checkpoint, Mapping) or set(raw_checkpoint) != {
            "checkpoint_name",
            "checkpoint_epoch",
            "source",
        }:
            raise StudentContractError("replay checkpoint custody keys drifted")
        if (
            raw_checkpoint["checkpoint_name"] != expected_name
            or raw_checkpoint["checkpoint_epoch"] != expected_epoch
        ):
            raise StudentContractError("replay checkpoint identity/epoch drifted")
        checkpoint_file = _parse_custody_file(
            raw_checkpoint["source"],
            name=f"checkpoint {expected_name}",
            expected_path=expected_path,
            suffix=".npz",
        )
        checkpoint_files.append(checkpoint_file)
        checkpoint_hashes.add(checkpoint_file.sha256)
    if len(checkpoint_hashes) != len(CHECKPOINTS):
        raise StudentContractError("replay checkpoint hashes must be distinct")
    replay_source_sha256 = _verify_semantic_hash(
        replay, key="replay_source_sha256", name="replay source"
    )

    generation = value["generation"]
    generation_required = {
        "axis",
        "teacher_backend",
        "teacher_device",
        "rendered_state_count",
        "mps_used",
        "synthetic_used",
        "source_video_substitute_used",
        "command_sha256",
        "environment_sha256",
        "receipt",
        "generation_sha256",
    }
    if not isinstance(generation, Mapping) or set(generation) != generation_required:
        raise StudentContractError("cache generation custody keys drifted")
    if (
        generation["axis"] != CACHE_GENERATION_AXIS
        or generation["teacher_backend"] != "torch"
        or generation["teacher_device"] != "cpu"
        or generation["rendered_state_count"] != N600
        or generation["mps_used"] is not False
        or generation["synthetic_used"] is not False
        or generation["source_video_substitute_used"] is not False
    ):
        raise StudentContractError("cache generation axis/backend/provenance drifted")
    _parse_content_sha256(generation["command_sha256"], name="generation command")
    _parse_content_sha256(generation["environment_sha256"], name="generation environment")
    generation_receipt = _parse_custody_file(
        generation["receipt"], name="generation receipt", suffix=".json"
    )
    generation_sha256 = _verify_semantic_hash(
        generation, key="generation_sha256", name="cache generation"
    )

    custody_sha256 = _verify_semantic_hash(
        value, key="custody_sha256", name="teacher source custody"
    )
    repository_files = (
        r_source,
        segnet_source,
        segnet_weights,
        renderer_source,
        replay_harness_source,
        *checkpoint_files,
    )
    bundle_sidecars = (
        renderer_config,
        source_manifest,
        upstream_source_manifest,
        generation_receipt,
    )
    if len({item.path for item in bundle_sidecars}) != len(bundle_sidecars):
        raise StudentContractError("teacher custody bundle sidecar paths must be unique")
    return TeacherSourceCustody(
        custody_sha256=custody_sha256,
        r_operator_sha256=r_operator_sha256,
        scorer_sha256=scorer_sha256,
        scalar_objective_sha256=scalar_objective_sha256,
        replay_source_sha256=replay_source_sha256,
        generation_sha256=generation_sha256,
        helmert_basis_sha256=HELMERT_BASIS_SHA256,
        post_r_input_surface_sha256=_post_r_input_surface_sha256(
            r_operator_sha256
        ),
        generation_axis=CACHE_GENERATION_AXIS,
        repository_files=repository_files,
        bundle_sidecars=bundle_sidecars,
    )


def _parse_artifact(value: Any, *, name: str) -> CacheArtifact:
    if not isinstance(value, Mapping):
        raise StudentContractError(f"cache artifact {name} must be a mapping")
    required = {"path", "bytes", "sha256", "dtype", "shape"}
    if set(value) != required:
        raise StudentContractError(f"cache artifact {name} keys drifted")
    path = value["path"]
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise StudentContractError(f"cache artifact {name} path must be bundle-relative")
    pure = PurePosixPath(path)
    if ".." in pure.parts or pure.suffix != ".npy":
        raise StudentContractError(f"cache artifact {name} path is unsafe or not .npy")
    byte_count = value["bytes"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise StudentContractError(f"cache artifact {name} bytes must be a positive integer")
    sha = _parse_content_sha256(value["sha256"], name=f"cache artifact {name}")
    expected_dtype, expected_shape = _ARTIFACT_SPECS[name]
    if value["dtype"] != expected_dtype or tuple(value["shape"]) != expected_shape:
        raise StudentContractError(f"cache artifact {name} dtype/shape drifted")
    return CacheArtifact(path, byte_count, sha, expected_dtype, expected_shape)


def validate_n600_cache_manifest_structure(manifest: Mapping[str, Any]) -> tuple[CacheRow, ...]:
    """Validate exact n600 assignments and raw-tensor custody metadata only.

    This structural function is not fit authority.  The strict
    :func:`validate_n600_cache_manifest` additionally hashes and inspects every
    referenced file.
    """

    required_top_level = {
        "schema",
        "source_kind",
        "cohort_count",
        "train_count",
        "heldout_count",
        "checkpoint_epochs",
        "teacher_source_custody",
        "rows",
    }
    if not isinstance(manifest, Mapping) or manifest.get("schema") != CACHE_SCHEMA:
        raise StudentContractError("unknown whole-teacher cache schema")
    if set(manifest) not in (required_top_level, required_top_level | {"charged_timing_inputs"}):
        raise StudentContractError("whole-teacher cache top-level keys drifted")
    if manifest.get("source_kind") != SOURCE_KIND:
        raise StudentContractError("source-video or non-rendered replay cache substitution refused")
    if manifest.get("cohort_count") != N600:
        raise StudentContractError("whole-teacher cache must contain exactly n600 rows")
    if manifest.get("train_count") != TRAIN_COUNT or manifest.get("heldout_count") != HELDOUT_COUNT:
        raise StudentContractError("whole-teacher cache must preserve the sealed 480/120 split")
    if manifest.get("checkpoint_epochs") != [epoch for _name, epoch in CHECKPOINTS]:
        raise StudentContractError("whole-teacher cache checkpoint epochs drifted")
    _parse_teacher_source_custody(manifest.get("teacher_source_custody"))
    raw_rows = manifest.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != N600:
        raise StudentContractError("whole-teacher cache rows must have length 600")
    expected_by_pair = {row.pair_index: row for row in _expected_assignments()}
    epochs = {index: epoch for index, (_name, epoch) in enumerate(CHECKPOINTS)}
    rows: list[CacheRow] = []
    assignment_ids: set[str] = set()
    pair_indices: set[int] = set()
    paths: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise StudentContractError("cache row must be a mapping")
        required = {
            "assignment_id",
            "pair_index",
            "checkpoint_index",
            "checkpoint_name",
            "checkpoint_epoch",
            "split",
            "source_kind",
            "artifacts",
        }
        if set(raw) != required:
            raise StudentContractError("cache row keys drifted")
        assignment_id = raw["assignment_id"]
        pair_index = raw["pair_index"]
        if not isinstance(assignment_id, str) or not assignment_id or assignment_id in assignment_ids:
            raise StudentContractError("cache assignment ids must be unique nonempty strings")
        if isinstance(pair_index, bool) or not isinstance(pair_index, int) or pair_index not in expected_by_pair:
            raise StudentContractError("cache pair index is outside the sealed n600 cohort")
        if pair_index in pair_indices:
            raise StudentContractError("cache duplicated a replay pair")
        expected = expected_by_pair[pair_index]
        if (
            raw["checkpoint_index"] != expected.checkpoint_index
            or raw["checkpoint_name"] != expected.checkpoint_name
            or raw["checkpoint_epoch"] != epochs[expected.checkpoint_index]
            or raw["split"] != expected.split
        ):
            raise StudentContractError("cache replay assignment drifted from the sealed helper")
        if raw["source_kind"] != SOURCE_KIND:
            raise StudentContractError("cache row is not a rendered V9 replay state")
        raw_artifacts = raw["artifacts"]
        if not isinstance(raw_artifacts, Mapping) or set(raw_artifacts) != set(_ARTIFACT_SPECS):
            raise StudentContractError("cache row lacks a required raw tensor artifact")
        artifacts = {name: _parse_artifact(raw_artifacts[name], name=name) for name in _ARTIFACT_SPECS}
        row_paths = {artifact.path for artifact in artifacts.values()}
        if len(row_paths) != len(artifacts) or paths.intersection(row_paths):
            raise StudentContractError("raw cache artifact paths must be unique per assignment")
        paths.update(row_paths)
        assignment_ids.add(assignment_id)
        pair_indices.add(pair_index)
        rows.append(
            CacheRow(
                assignment_id=assignment_id,
                pair_index=pair_index,
                checkpoint_index=expected.checkpoint_index,
                checkpoint_name=expected.checkpoint_name,
                checkpoint_epoch=epochs[expected.checkpoint_index],
                split=expected.split,
                artifacts=artifacts,
            )
        )
    if pair_indices != set(range(N600)):
        raise StudentContractError("cache does not cover each sealed replay pair exactly once")
    rows.sort(key=lambda row: row.pair_index)
    return tuple(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_n600_cache_manifest(
    manifest: Mapping[str, Any], *, bundle_root: str | Path
) -> ValidatedN600Cache:
    """Hash and header-verify every raw tensor in the exact n600 bundle."""

    rows = validate_n600_cache_manifest_structure(manifest)
    teacher_source_custody = _parse_teacher_source_custody(
        manifest["teacher_source_custody"]
    )
    root = Path(bundle_root).resolve()
    if not root.is_dir():
        raise StudentContractError("n600 cache bundle_root must be an existing directory")
    for source in teacher_source_custody.repository_files:
        path = (_REPO_ROOT / source.path).resolve()
        try:
            path.relative_to(_REPO_ROOT)
        except ValueError as exc:
            raise StudentContractError("teacher repository source escapes the repository") from exc
        if not path.is_file() or path.stat().st_size != source.bytes:
            raise StudentContractError(f"teacher repository source {source.path} byte count drifted")
        if _file_sha256(path) != source.sha256:
            raise StudentContractError(f"teacher repository source {source.path} sha256 drifted")
    for sidecar in teacher_source_custody.bundle_sidecars:
        path = (root / sidecar.path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise StudentContractError("teacher custody sidecar escapes bundle_root") from exc
        if not path.is_file() or path.stat().st_size != sidecar.bytes:
            raise StudentContractError(f"teacher custody sidecar {sidecar.path} byte count drifted")
        if _file_sha256(path) != sidecar.sha256:
            raise StudentContractError(f"teacher custody sidecar {sidecar.path} sha256 drifted")
    for row in rows:
        for name, artifact in row.artifacts.items():
            path = (root / artifact.path).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise StudentContractError(f"cache artifact {name} escapes bundle_root") from exc
            if not path.is_file() or path.stat().st_size != artifact.bytes:
                raise StudentContractError(f"cache artifact {name} is missing or byte count drifted")
            if _file_sha256(path) != artifact.sha256:
                raise StudentContractError(f"cache artifact {name} sha256 drifted")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if not isinstance(array, np.ndarray):
                raise StudentContractError(f"cache artifact {name} must be standalone NPY")
            if array.dtype.str != artifact.dtype or tuple(array.shape) != artifact.shape:
                raise StudentContractError(f"cache artifact {name} NPY header drifted")
    canonical_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return ValidatedN600Cache(
        bundle_root=root,
        manifest_sha256=hashlib.sha256(canonical_manifest).hexdigest(),
        teacher_source_custody=teacher_source_custody,
        rows=rows,
    )


def _canonical_payload_sha256(value: Any) -> str:
    return _semantic_sha256(value)


def _array_mapping_sha256(arrays: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _fit_contract_hashes(
    *,
    policy: CachedStudentFitPolicy,
    validated_cache: ValidatedN600Cache,
    manifest_path: Path,
) -> dict[str, str]:
    return {
        "policy_sha256": _canonical_payload_sha256(policy.to_dict()),
        "source_sha256": _file_sha256(Path(__file__)),
        "cache_manifest_sha256": validated_cache.manifest_sha256,
        "cache_manifest_file_sha256": _file_sha256(manifest_path),
        "teacher_source_custody_sha256": (
            validated_cache.teacher_source_custody.custody_sha256
        ),
        "scalar_objective_sha256": (
            validated_cache.teacher_source_custody.scalar_objective_sha256
        ),
        "parameter_layout_sha256": parameter_layout_sha256(policy.architecture),
    }


def _validate_fit_row_arrays(value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = {
        "rendered_frame",
        "teacher_quotient4",
        "teacher_input_costate",
        "labels",
    }
    if set(value) != required:
        raise StudentContractError("cached fit row lacks an exact required tensor")
    frame = _as_finite_fp32(value["rendered_frame"], name="rendered_frame")
    quotient = _as_finite_fp32(value["teacher_quotient4"], name="teacher_quotient4")
    costate = _as_finite_fp32(value["teacher_input_costate"], name="teacher_input_costate")
    labels = np.asarray(value["labels"])
    if frame.ndim != 4 or frame.shape[0] != 1 or frame.shape[1] != RGB_CHANNELS:
        raise StudentContractError("cached rendered frame must have shape (1,3,H,W)")
    if np.any((frame < 0.0) | (frame > 255.0)):
        raise StudentContractError(
            "cached rendered frame must remain on the post-R float32 RGB 0..255 surface"
        )
    if quotient.shape != (1, QUOTIENT_DIM, frame.shape[2], frame.shape[3]):
        raise StudentContractError("cached teacher quotient geometry drifted")
    if costate.shape != frame.shape:
        raise StudentContractError("cached teacher input costate geometry drifted")
    if labels.shape != frame.shape[2:] or not np.issubdtype(labels.dtype, np.integer):
        raise StudentContractError("cached labels must have integer shape (H,W)")
    if np.any((labels < 0) | (labels >= CLASS_COUNT)):
        raise StudentContractError("cached labels contain an invalid class id")
    return {
        "rendered_frame": frame,
        "teacher_quotient4": quotient,
        "teacher_input_costate": costate,
        "labels": np.ascontiguousarray(labels, dtype=np.int64),
    }


def _fit_state_arrays(
    *,
    backend: Any,
    parameters: Mapping[str, Any],
    first_moment: Mapping[str, Any],
    second_moment: Mapping[str, Any],
    best_parameters: Mapping[str, np.ndarray],
    train_loss_history: Sequence[float],
    train_step_ms_history: Sequence[float],
) -> dict[str, np.ndarray]:
    current = backend.tree_to_numpy(parameters)
    first = backend.tree_to_numpy(first_moment)
    second = backend.tree_to_numpy(second_moment)
    arrays: dict[str, np.ndarray] = {}
    for prefix, tree in (
        ("parameter", current),
        ("adam_first_moment", first),
        ("adam_second_moment", second),
        ("best_parameter", best_parameters),
    ):
        for name, value in tree.items():
            arrays[f"{prefix}__{name}"] = np.ascontiguousarray(value, dtype=np.float32)
    arrays["history__train_loss"] = np.asarray(train_loss_history, dtype=np.float64)
    arrays["history__train_step_ms"] = np.asarray(train_step_ms_history, dtype=np.float64)
    return arrays


def _parameter_tree_from_state_arrays(
    arrays: Mapping[str, Any],
    *,
    prefix: str,
    architecture: StudentArchitecture,
) -> dict[str, np.ndarray]:
    names = [name for name, _shape in _parameter_shapes(architecture)]
    keys = {f"{prefix}__{name}" for name in names}
    if not keys.issubset(arrays):
        raise StudentContractError(f"resume state lacks complete {prefix} tensors")
    tree = {name: arrays[f"{prefix}__{name}"] for name in names}
    if prefix in {"parameter", "best_parameter"}:
        return validate_student_parameters(architecture, tree)
    result: dict[str, np.ndarray] = {}
    for name, shape in _parameter_shapes(architecture):
        value = _as_finite_fp32(tree[name], name=f"{prefix}__{name}")
        if value.shape != shape:
            raise StudentContractError(f"resume optimizer tensor {prefix}__{name} drifted")
        result[name] = value
    return result


def _fit_checkpoint_metadata(
    *,
    stage: str,
    hashes: Mapping[str, str],
    arrays: Mapping[str, np.ndarray],
    architecture: StudentArchitecture,
    parameters: Mapping[str, np.ndarray],
    best_parameters: Mapping[str, np.ndarray],
    optimizer_step: int,
    next_epoch: int,
    next_train_offset: int,
    epoch_loss_sum: float,
    epoch_loss_count: int,
    best_objective: float | None,
    best_epoch: int | None,
    terminal_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema": FIT_STATE_SCHEMA,
        "stage": stage,
        "hashes": dict(hashes),
        "optimizer": "explicit_adamw",
        "optimizer_step": optimizer_step,
        "next_epoch": next_epoch,
        "next_train_offset": next_train_offset,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_loss_count": epoch_loss_count,
        "best_objective": best_objective,
        "best_epoch": best_epoch,
        "state_arrays_sha256": _array_mapping_sha256(arrays),
        "current_parameters_sha256": hashlib.sha256(
            serialize_student_parameters(architecture, parameters)
        ).hexdigest(),
        "best_parameters_sha256": hashlib.sha256(
            serialize_student_parameters(architecture, best_parameters)
        ).hexdigest(),
        "teacher_calls": 0,
        "backend": "mlx",
        "numerical_reference": "numpy_fp32",
        "measurement_axis": MEASUREMENT_AXIS,
        "research_only": True,
        "score_claim": False,
    }
    if terminal_result is not None:
        metadata["terminal_result"] = terminal_result
        metadata["terminal_result_sha256"] = _canonical_payload_sha256(terminal_result)
    return metadata


def _emit_fit_checkpoint(
    *,
    checkpoint_callback: Any,
    stage: str,
    hashes: Mapping[str, str],
    backend: Any,
    architecture: StudentArchitecture,
    parameters: Mapping[str, Any],
    first_moment: Mapping[str, Any],
    second_moment: Mapping[str, Any],
    best_parameters: Mapping[str, np.ndarray],
    optimizer_step: int,
    next_epoch: int,
    next_train_offset: int,
    epoch_loss_sum: float,
    epoch_loss_count: int,
    best_objective: float | None,
    best_epoch: int | None,
    train_loss_history: Sequence[float],
    train_step_ms_history: Sequence[float],
    terminal_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    arrays = _fit_state_arrays(
        backend=backend,
        parameters=parameters,
        first_moment=first_moment,
        second_moment=second_moment,
        best_parameters=best_parameters,
        train_loss_history=train_loss_history,
        train_step_ms_history=train_step_ms_history,
    )
    current_numpy = _parameter_tree_from_state_arrays(
        arrays, prefix="parameter", architecture=architecture
    )
    metadata = _fit_checkpoint_metadata(
        stage=stage,
        hashes=hashes,
        arrays=arrays,
        architecture=architecture,
        parameters=current_numpy,
        best_parameters=best_parameters,
        optimizer_step=optimizer_step,
        next_epoch=next_epoch,
        next_train_offset=next_train_offset,
        epoch_loss_sum=epoch_loss_sum,
        epoch_loss_count=epoch_loss_count,
        best_objective=best_objective,
        best_epoch=best_epoch,
        terminal_result=terminal_result,
    )
    checkpoint_callback(stage, optimizer_step, {"arrays": arrays, "metadata": metadata})
    return metadata


def _restore_fit_state(
    *,
    resume_state: Mapping[str, Any],
    expected_hashes: Mapping[str, str],
    architecture: StudentArchitecture,
    backend: Any,
    fit_epochs: int,
    train_count: int,
) -> dict[str, Any]:
    if set(resume_state) != {"arrays", "metadata"}:
        raise StudentContractError("resume_state requires exact arrays and metadata mappings")
    arrays = resume_state["arrays"]
    metadata = resume_state["metadata"]
    if not isinstance(arrays, Mapping) or not isinstance(metadata, Mapping):
        raise StudentContractError("resume_state arrays/metadata must be mappings")
    if metadata.get("schema") != FIT_STATE_SCHEMA or metadata.get("hashes") != dict(expected_hashes):
        raise StudentContractError("resume source/cache/policy/layout hashes drifted")
    if metadata.get("backend") != "mlx" or metadata.get("numerical_reference") != "numpy_fp32":
        raise StudentContractError("resume backend or numerical authority drifted")
    if metadata.get("state_arrays_sha256") != _array_mapping_sha256(arrays):
        raise StudentContractError("resume state array hash drifted")
    parameters_numpy = _parameter_tree_from_state_arrays(
        arrays, prefix="parameter", architecture=architecture
    )
    first_numpy = _parameter_tree_from_state_arrays(
        arrays, prefix="adam_first_moment", architecture=architecture
    )
    second_numpy = _parameter_tree_from_state_arrays(
        arrays, prefix="adam_second_moment", architecture=architecture
    )
    best_parameters = _parameter_tree_from_state_arrays(
        arrays, prefix="best_parameter", architecture=architecture
    )
    if metadata.get("current_parameters_sha256") != hashlib.sha256(
        serialize_student_parameters(architecture, parameters_numpy)
    ).hexdigest():
        raise StudentContractError("resume current parameter hash drifted")
    if metadata.get("best_parameters_sha256") != hashlib.sha256(
        serialize_student_parameters(architecture, best_parameters)
    ).hexdigest():
        raise StudentContractError("resume best parameter hash drifted")
    loss_history = np.asarray(arrays.get("history__train_loss"), dtype=np.float64)
    time_history = np.asarray(arrays.get("history__train_step_ms"), dtype=np.float64)
    if (
        loss_history.ndim != 1
        or time_history.ndim != 1
        or loss_history.shape != time_history.shape
        or not np.isfinite(loss_history).all()
        or not np.isfinite(time_history).all()
        or np.any(time_history < 0.0)
    ):
        raise StudentContractError("resume training history is invalid")
    optimizer_step = metadata.get("optimizer_step")
    next_epoch = metadata.get("next_epoch")
    next_offset = metadata.get("next_train_offset")
    epoch_count = metadata.get("epoch_loss_count")
    epoch_sum = metadata.get("epoch_loss_sum")
    if (
        isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step != len(loss_history)
    ):
        raise StudentContractError("resume optimizer step disagrees with preserved history")
    if (
        isinstance(next_epoch, bool)
        or not isinstance(next_epoch, int)
        or not 0 <= next_epoch <= fit_epochs
        or isinstance(next_offset, bool)
        or not isinstance(next_offset, int)
        or not 0 <= next_offset < train_count
    ):
        raise StudentContractError("resume epoch/offset is outside the typed fit schedule")
    if next_epoch == fit_epochs and next_offset != 0:
        raise StudentContractError("completed fit resume must start measurement at offset zero")
    if (
        isinstance(epoch_count, bool)
        or not isinstance(epoch_count, int)
        or epoch_count != next_offset
        or not math.isfinite(float(epoch_sum))
    ):
        raise StudentContractError("resume partial-epoch accumulator drifted")
    best_objective = metadata.get("best_objective")
    best_epoch = metadata.get("best_epoch")
    if best_objective is not None and (
        not math.isfinite(float(best_objective)) or float(best_objective) < 0.0
    ):
        raise StudentContractError("resume best objective is invalid")
    if best_epoch is not None and (
        isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or not 0 <= best_epoch < fit_epochs
    ):
        raise StudentContractError("resume best epoch is invalid")
    terminal_result = metadata.get("terminal_result")
    if terminal_result is not None:
        if metadata.get("terminal_result_sha256") != _canonical_payload_sha256(terminal_result):
            raise StudentContractError("resume terminal result hash drifted")
        if (
            terminal_result.get("schema") != FIT_RESULT_SCHEMA
            or terminal_result.get("n_pairs") != N600
            or terminal_result.get("backend") != "mlx"
            or terminal_result.get("teacher_calls") != 0
        ):
            raise StudentContractError("resume terminal result authority drifted")
    return {
        "parameters": backend.parameters_from_numpy(parameters_numpy),
        "first_moment": backend.parameters_from_numpy(first_numpy),
        "second_moment": backend.parameters_from_numpy(second_numpy),
        "best_parameters": best_parameters,
        "optimizer_step": optimizer_step,
        "next_epoch": next_epoch,
        "next_offset": next_offset,
        "epoch_loss_sum": float(epoch_sum),
        "epoch_loss_count": epoch_count,
        "best_objective": None if best_objective is None else float(best_objective),
        "best_epoch": best_epoch,
        "train_loss_history": loss_history.tolist(),
        "train_step_ms_history": time_history.tolist(),
        "terminal_result": terminal_result,
    }


def _metric_timing_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise StudentContractError("timing summary requires finite nonnegative observations")
    return {
        "count": int(array.size),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "max_ms": float(np.max(array)),
    }


def _update_output_stream_digest(
    digest: Any,
    *,
    assignment_id: str,
    tensor_name: str,
    value: Any,
) -> None:
    array = _as_finite_fp32(value, name=tensor_name)
    digest.update(assignment_id.encode())
    digest.update(tensor_name.encode())
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(array).tobytes())


def _load_json_without_duplicate_keys(path: Path, *, name: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StudentContractError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    except StudentContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StudentContractError(f"{name} is unreadable") from exc


def _teacher_timing_inputs_from_manifest(
    manifest: Mapping[str, Any],
    *,
    bundle_root: Path,
    teacher_source_custody: TeacherSourceCustody,
    backend_impl: Any,
) -> dict[str, Any]:
    """Load a byte-verified, same-axis exact-teacher timing receipt."""

    value = manifest.get("charged_timing_inputs")
    if value is None:
        return {
            "status": "UNMEASURED_BLOCKED_MISSING_VERIFIED_TEACHER_TIMING_RECEIPT",
            "exact_teacher_forward_ms": None,
            "exact_teacher_forward_input_vjp_ms": None,
            "teacher_timing_receipt": None,
            "teacher_timing_receipt_sha256": None,
            "raw_observations": None,
            "measurement_axis": None,
            "hardware_fingerprint_sha256": None,
        }
    if not isinstance(value, Mapping) or set(value) != {"schema", "receipt"}:
        raise StudentContractError("charged teacher timing input schema drifted")
    if value["schema"] != "whole_teacher_distilled_student_charged_timing_inputs.v2":
        raise StudentContractError("unknown charged teacher timing schema")
    receipt_file = _parse_custody_file(
        value["receipt"], name="teacher timing receipt", suffix=".json"
    )
    receipt_path = (bundle_root / receipt_file.path).resolve()
    try:
        receipt_path.relative_to(bundle_root)
    except ValueError as exc:
        raise StudentContractError("teacher timing receipt escapes bundle_root") from exc
    if not receipt_path.is_file() or receipt_path.stat().st_size != receipt_file.bytes:
        raise StudentContractError("teacher timing receipt byte count drifted")
    if _file_sha256(receipt_path) != receipt_file.sha256:
        raise StudentContractError("teacher timing receipt sha256 drifted")
    receipt = _load_json_without_duplicate_keys(
        receipt_path, name="teacher timing receipt"
    )
    required = {
        "schema",
        "measurement_axis",
        "hardware_fingerprint_sha256",
        "teacher_source_custody_sha256",
        "r_operator_sha256",
        "scorer_sha256",
        "scalar_objective_sha256",
        "post_r_input_surface_sha256",
        "warmup_excluded",
        "observation_count",
        "exact_teacher_forward_ms",
        "exact_teacher_forward_input_vjp_ms",
        "raw_observations",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise StudentContractError("teacher timing receipt keys drifted")
    if receipt["schema"] != "whole_teacher_distilled_student_teacher_timing_receipt.v1":
        raise StudentContractError("teacher timing receipt schema drifted")
    hardware_fingerprint = _parse_content_sha256(
        receipt["hardware_fingerprint_sha256"], name="teacher timing hardware fingerprint"
    )
    if (
        receipt["measurement_axis"] != getattr(backend_impl, "measurement_axis", None)
        or hardware_fingerprint
        != getattr(backend_impl, "hardware_fingerprint_sha256", None)
    ):
        raise StudentContractError("teacher/student timing axis or hardware fingerprint drifted")
    if (
        receipt["teacher_source_custody_sha256"]
        != teacher_source_custody.custody_sha256
        or receipt["r_operator_sha256"] != teacher_source_custody.r_operator_sha256
        or receipt["scorer_sha256"] != teacher_source_custody.scorer_sha256
        or receipt["scalar_objective_sha256"]
        != teacher_source_custody.scalar_objective_sha256
        or receipt["post_r_input_surface_sha256"]
        != teacher_source_custody.post_r_input_surface_sha256
    ):
        raise StudentContractError("teacher timing receipt semantic custody drifted")
    if receipt["warmup_excluded"] is not True:
        raise StudentContractError("teacher timing receipt must exclude first-use warmup")
    observation_count = receipt["observation_count"]
    if (
        isinstance(observation_count, bool)
        or not isinstance(observation_count, int)
        or observation_count < 1
    ):
        raise StudentContractError("teacher timing observation_count must be positive")
    timings = (
        receipt["exact_teacher_forward_ms"],
        receipt["exact_teacher_forward_input_vjp_ms"],
    )
    if not all(math.isfinite(float(item)) and float(item) > 0.0 for item in timings):
        raise StudentContractError("charged teacher timings must be finite and positive")
    raw_observations = _parse_custody_file(
        receipt["raw_observations"],
        name="teacher timing raw observations",
        suffix=".json",
    )
    raw_path = (bundle_root / raw_observations.path).resolve()
    try:
        raw_path.relative_to(bundle_root)
    except ValueError as exc:
        raise StudentContractError("teacher timing observations escape bundle_root") from exc
    if not raw_path.is_file() or raw_path.stat().st_size != raw_observations.bytes:
        raise StudentContractError("teacher timing observations byte count drifted")
    if _file_sha256(raw_path) != raw_observations.sha256:
        raise StudentContractError("teacher timing observations sha256 drifted")
    raw = _load_json_without_duplicate_keys(
        raw_path, name="teacher timing raw observations"
    )
    raw_required = {
        "schema",
        "measurement_axis",
        "hardware_fingerprint_sha256",
        "teacher_source_custody_sha256",
        "r_operator_sha256",
        "scorer_sha256",
        "scalar_objective_sha256",
        "post_r_input_surface_sha256",
        "warmup_excluded",
        "summary_statistic",
        "rows",
    }
    if not isinstance(raw, Mapping) or set(raw) != raw_required:
        raise StudentContractError("teacher timing raw observation keys drifted")
    if raw["schema"] != "whole_teacher_distilled_student_teacher_timing_raw.v1":
        raise StudentContractError("teacher timing raw observation schema drifted")
    if (
        raw["measurement_axis"] != receipt["measurement_axis"]
        or raw["hardware_fingerprint_sha256"] != hardware_fingerprint
        or raw["teacher_source_custody_sha256"]
        != teacher_source_custody.custody_sha256
        or raw["r_operator_sha256"] != teacher_source_custody.r_operator_sha256
        or raw["scorer_sha256"] != teacher_source_custody.scorer_sha256
        or raw["scalar_objective_sha256"]
        != teacher_source_custody.scalar_objective_sha256
        or raw["post_r_input_surface_sha256"]
        != teacher_source_custody.post_r_input_surface_sha256
        or raw["warmup_excluded"] is not True
        or raw["summary_statistic"] != "median_ms"
    ):
        raise StudentContractError("teacher timing raw semantic/axis custody drifted")
    rows = raw["rows"]
    if not isinstance(rows, list) or len(rows) != observation_count:
        raise StudentContractError("teacher timing raw observation count drifted")
    forward_observations: list[float] = []
    forward_vjp_observations: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "sample_index",
            "exact_teacher_forward_ms",
            "exact_teacher_forward_input_vjp_ms",
        }:
            raise StudentContractError("teacher timing raw row keys drifted")
        if row["sample_index"] != index:
            raise StudentContractError("teacher timing raw sample indices drifted")
        row_timings = (
            row["exact_teacher_forward_ms"],
            row["exact_teacher_forward_input_vjp_ms"],
        )
        if not all(math.isfinite(float(item)) and float(item) > 0.0 for item in row_timings):
            raise StudentContractError("teacher timing raw row must be finite and positive")
        forward_observations.append(float(row_timings[0]))
        forward_vjp_observations.append(float(row_timings[1]))
    derived_timings = (
        float(np.median(np.asarray(forward_observations, dtype=np.float64))),
        float(np.median(np.asarray(forward_vjp_observations, dtype=np.float64))),
    )
    if timings != derived_timings:
        raise StudentContractError(
            "teacher timing receipt summaries do not equal raw median observations"
        )
    return {
        "status": "CONTENT_BOUND_MATCHED_AXIS_RECEIPT_VERIFIED",
        "exact_teacher_forward_ms": float(timings[0]),
        "exact_teacher_forward_input_vjp_ms": float(timings[1]),
        "teacher_timing_receipt": asdict(receipt_file),
        "teacher_timing_receipt_sha256": receipt_file.sha256,
        "raw_observations": asdict(raw_observations),
        "measurement_axis": receipt["measurement_axis"],
        "hardware_fingerprint_sha256": hardware_fingerprint,
        "observation_count": observation_count,
        "summary_statistic": "median_ms",
    }


def fit_measure_cached_student(
    manifest_path: str | Path,
    bundle_root: str | Path,
    student_size: str,
    seed: int,
    fit_epochs: int,
    checkpoint_callback: Any,
    resume_state: Mapping[str, Any] | None,
    scratch_dir: str | Path,
    backend: str,
    numerical_reference: str,
) -> dict[str, Any]:
    """Fit and measure the cached n600 student using only real MLX autograd.

    Strict cache validation occurs before MLX import.  The function never calls
    a teacher, creates synthetic labels, or relabels a CPU/Torch result as MLX.
    All optimizer and best-parameter state is emitted through the harness's
    atomic, stage-distinct callback and hash-verified on resume.
    """

    path = Path(manifest_path)
    if not path.is_file():
        raise StudentContractError("BLOCKED-DATA-CUSTODY: cache manifest is absent")
    try:
        manifest = _load_json_without_duplicate_keys(path, name="cache manifest")
    except StudentContractError as exc:
        raise StudentContractError("BLOCKED-DATA-CUSTODY: cache manifest is unreadable") from exc
    validated_cache = validate_n600_cache_manifest(manifest, bundle_root=bundle_root)
    architecture = architecture_for_size(student_size)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise StudentContractError("seed must be a nonnegative integer")
    if isinstance(fit_epochs, bool) or not isinstance(fit_epochs, int) or fit_epochs < 1:
        raise StudentContractError("fit_epochs must be an integer >= 1")
    if checkpoint_callback is None or not callable(checkpoint_callback):
        raise StudentContractError("checkpoint_callback must be callable")
    if resume_state is not None and not isinstance(resume_state, Mapping):
        raise StudentContractError("resume_state must be a mapping or None")
    scratch = Path(scratch_dir)
    if not scratch.is_dir():
        raise StudentContractError("scratch_dir must be an existing durable task directory")
    if backend != "mlx":
        raise StudentContractError("whole-teacher fit backend must be MLX; no Torch fallback")
    if numerical_reference != "numpy_fp32":
        raise StudentContractError("whole-teacher numerical reference must be numpy_fp32")
    policy = cached_student_fit_policy(
        architecture=architecture,
        seed=seed,
        fit_epochs=fit_epochs,
    )
    if (
        len(validated_cache.rows) != N600
        or len(validated_cache.train_rows) != TRAIN_COUNT
        or len(validated_cache.heldout_rows) != HELDOUT_COUNT
    ):
        raise StudentContractError("validated cache cohort/split count drifted before fit")
    backend_impl = _load_mlx_backend()
    if getattr(backend_impl, "name", None) != "mlx":
        raise StudentContractError("fit backend cannot be relabeled as MLX")
    if getattr(backend_impl, "measurement_axis", None) != MEASUREMENT_AXIS:
        raise StudentContractError("MLX backend measurement axis drifted")
    _parse_content_sha256(
        getattr(backend_impl, "hardware_fingerprint_sha256", None),
        name="MLX backend hardware fingerprint",
    )
    hashes = _fit_contract_hashes(
        policy=policy,
        validated_cache=validated_cache,
        manifest_path=path,
    )

    train_rows = tuple(sorted(validated_cache.train_rows, key=lambda row: row.pair_index))
    if resume_state is None:
        initial_numpy = initialize_student_parameters(architecture, seed=seed)
        parameters = backend_impl.parameters_from_numpy(initial_numpy)
        first_moment = backend_impl.zeros_like(parameters)
        second_moment = backend_impl.zeros_like(parameters)
        state: dict[str, Any] = {
            "parameters": parameters,
            "first_moment": first_moment,
            "second_moment": second_moment,
            "best_parameters": {name: value.copy() for name, value in initial_numpy.items()},
            "optimizer_step": 0,
            "next_epoch": 0,
            "next_offset": 0,
            "epoch_loss_sum": 0.0,
            "epoch_loss_count": 0,
            "best_objective": None,
            "best_epoch": None,
            "train_loss_history": [],
            "train_step_ms_history": [],
            "terminal_result": None,
        }
        _emit_fit_checkpoint(
            checkpoint_callback=checkpoint_callback,
            stage="initialized",
            hashes=hashes,
            backend=backend_impl,
            architecture=architecture,
            parameters=state["parameters"],
            first_moment=state["first_moment"],
            second_moment=state["second_moment"],
            best_parameters=state["best_parameters"],
            optimizer_step=0,
            next_epoch=0,
            next_train_offset=0,
            epoch_loss_sum=0.0,
            epoch_loss_count=0,
            best_objective=None,
            best_epoch=None,
            train_loss_history=(),
            train_step_ms_history=(),
        )
    else:
        state = _restore_fit_state(
            resume_state=resume_state,
            expected_hashes=hashes,
            architecture=architecture,
            backend=backend_impl,
            fit_epochs=fit_epochs,
            train_count=len(train_rows),
        )
        if state["terminal_result"] is not None:
            return dict(state["terminal_result"])

    # Symmetric first-use compilation warmup; no optimizer update and no timing
    # row enters the charged accounting.
    warmup_arrays = _validate_fit_row_arrays(
        validated_cache.load_row(train_rows[state["next_offset"]].pair_index)
    )
    warmup_boundary = boundary_mask_from_labels_numpy(warmup_arrays["labels"])
    backend_impl.warmup_train(
        state["parameters"],
        row=warmup_arrays,
        boundary_mask=warmup_boundary,
        policy=policy,
    )

    for epoch in range(state["next_epoch"], fit_epochs):
        start_offset = state["next_offset"] if epoch == state["next_epoch"] else 0
        for offset in range(start_offset, len(train_rows)):
            assignment = train_rows[offset]
            row = _validate_fit_row_arrays(validated_cache.load_row(assignment.pair_index))
            boundary = boundary_mask_from_labels_numpy(row["labels"])
            (
                state["parameters"],
                state["first_moment"],
                state["second_moment"],
                loss_value,
                step_ms,
            ) = backend_impl.train_step(
                state["parameters"],
                state["first_moment"],
                state["second_moment"],
                optimizer_step=state["optimizer_step"],
                row=row,
                boundary_mask=boundary,
                policy=policy,
            )
            state["optimizer_step"] += 1
            state["epoch_loss_sum"] += loss_value
            state["epoch_loss_count"] += 1
            state["train_loss_history"].append(loss_value)
            state["train_step_ms_history"].append(step_ms)
            state["next_epoch"] = epoch
            state["next_offset"] = offset + 1
            if state["next_offset"] == len(train_rows):
                state["next_epoch"] = epoch + 1
                state["next_offset"] = 0
            if (
                state["optimizer_step"] % policy.checkpoint_interval_steps == 0
                and state["next_offset"] != 0
            ):
                _emit_fit_checkpoint(
                    checkpoint_callback=checkpoint_callback,
                    stage="fit_progress",
                    hashes=hashes,
                    backend=backend_impl,
                    architecture=architecture,
                    parameters=state["parameters"],
                    first_moment=state["first_moment"],
                    second_moment=state["second_moment"],
                    best_parameters=state["best_parameters"],
                    optimizer_step=state["optimizer_step"],
                    next_epoch=state["next_epoch"],
                    next_train_offset=state["next_offset"],
                    epoch_loss_sum=state["epoch_loss_sum"],
                    epoch_loss_count=state["epoch_loss_count"],
                    best_objective=state["best_objective"],
                    best_epoch=state["best_epoch"],
                    train_loss_history=state["train_loss_history"],
                    train_step_ms_history=state["train_step_ms_history"],
                )

        if state["epoch_loss_count"] != len(train_rows):
            raise StudentContractError("fit epoch did not cover the exact train split once")
        epoch_objective = state["epoch_loss_sum"] / state["epoch_loss_count"]
        improved = state["best_objective"] is None or epoch_objective < state["best_objective"]
        if improved:
            state["best_objective"] = epoch_objective
            state["best_epoch"] = epoch
            state["best_parameters"] = backend_impl.tree_to_numpy(state["parameters"])
        state["epoch_loss_sum"] = 0.0
        state["epoch_loss_count"] = 0
        _emit_fit_checkpoint(
            checkpoint_callback=checkpoint_callback,
            stage=f"fit_epoch_{epoch + 1:04d}",
            hashes=hashes,
            backend=backend_impl,
            architecture=architecture,
            parameters=state["parameters"],
            first_moment=state["first_moment"],
            second_moment=state["second_moment"],
            best_parameters=state["best_parameters"],
            optimizer_step=state["optimizer_step"],
            next_epoch=state["next_epoch"],
            next_train_offset=0,
            epoch_loss_sum=0.0,
            epoch_loss_count=0,
            best_objective=state["best_objective"],
            best_epoch=state["best_epoch"],
            train_loss_history=state["train_loss_history"],
            train_step_ms_history=state["train_step_ms_history"],
        )
        if improved:
            _emit_fit_checkpoint(
                checkpoint_callback=checkpoint_callback,
                stage="best_fit",
                hashes=hashes,
                backend=backend_impl,
                architecture=architecture,
                parameters=state["parameters"],
                first_moment=state["first_moment"],
                second_moment=state["second_moment"],
                best_parameters=state["best_parameters"],
                optimizer_step=state["optimizer_step"],
                next_epoch=state["next_epoch"],
                next_train_offset=0,
                epoch_loss_sum=0.0,
                epoch_loss_count=0,
                best_objective=state["best_objective"],
                best_epoch=state["best_epoch"],
                train_loss_history=state["train_loss_history"],
                train_step_ms_history=state["train_step_ms_history"],
            )

    if state["best_objective"] is None or state["best_epoch"] is None:
        raise StudentContractError("fit completed without a finite best parameter state")
    best_tree = backend_impl.parameters_from_numpy(state["best_parameters"])
    all_rows = tuple(sorted(validated_cache.rows, key=lambda row: row.pair_index))
    first_measurement = _validate_fit_row_arrays(
        validated_cache.load_row(all_rows[0].pair_index)
    )
    # Symmetric MLX and NumPy first-use warmups excluded from timing.
    backend_impl.predict_and_vjp(
        best_tree,
        frame=first_measurement["rendered_frame"],
        labels=first_measurement["labels"],
        architecture=architecture,
    )
    student_forward_numpy(first_measurement["rendered_frame"], architecture, state["best_parameters"])
    student_ce_input_vjp_numpy(
        first_measurement["rendered_frame"],
        architecture,
        state["best_parameters"],
        first_measurement["labels"],
    )

    per_pair: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    vjp_rows: list[dict[str, Any]] = []
    parity_forward_rows: list[dict[str, Any]] = []
    parity_vjp_rows: list[dict[str, Any]] = []
    boundary_vjp_rows: list[dict[str, Any]] = []
    forward_ms_rows: list[float] = []
    forward_vjp_ms_rows: list[float] = []
    first_forward_digest = hashlib.sha256()
    first_vjp_digest = hashlib.sha256()
    first_combined_digest = hashlib.sha256()
    first_mlx_forward_digest = hashlib.sha256()
    first_mlx_vjp_digest = hashlib.sha256()
    first_mlx_combined_digest = hashlib.sha256()
    for assignment in all_rows:
        row = _validate_fit_row_arrays(validated_cache.load_row(assignment.pair_index))
        boundary = boundary_mask_from_labels_numpy(row["labels"])
        mlx_quotient, mlx_vjp, forward_ms, forward_vjp_ms = backend_impl.predict_and_vjp(
            best_tree,
            frame=row["rendered_frame"],
            labels=row["labels"],
            architecture=architecture,
        )
        numpy_quotient = student_forward_numpy(
            row["rendered_frame"], architecture, state["best_parameters"]
        )
        numpy_vjp = student_ce_input_vjp_numpy(
            row["rendered_frame"], architecture, state["best_parameters"], row["labels"]
        )
        forward = forward_pair_metrics(
            assignment.assignment_id,
            row["teacher_quotient4"],
            numpy_quotient,
        )
        vjp = vjp_pair_metrics(
            assignment.assignment_id,
            row["teacher_input_costate"],
            numpy_vjp,
            boundary_mask=boundary,
        )
        parity_forward = forward_pair_metrics(
            assignment.assignment_id,
            numpy_quotient,
            mlx_quotient,
        )
        parity_vjp = vjp_pair_metrics(
            assignment.assignment_id,
            numpy_vjp,
            mlx_vjp,
        )
        boundary_metric = {"assignment_id": assignment.assignment_id, **vjp["boundary_diagnostic"]}
        forward_rows.append(forward)
        vjp_rows.append(vjp)
        parity_forward_rows.append(parity_forward)
        parity_vjp_rows.append(parity_vjp)
        boundary_vjp_rows.append(boundary_metric)
        forward_ms_rows.append(forward_ms)
        forward_vjp_ms_rows.append(forward_vjp_ms)
        for digest in (first_forward_digest, first_combined_digest):
            _update_output_stream_digest(
                digest,
                assignment_id=assignment.assignment_id,
                tensor_name="student_quotient4",
                value=numpy_quotient,
            )
        for digest in (first_vjp_digest, first_combined_digest):
            _update_output_stream_digest(
                digest,
                assignment_id=assignment.assignment_id,
                tensor_name="student_input_vjp",
                value=numpy_vjp,
            )
        for digest in (first_mlx_forward_digest, first_mlx_combined_digest):
            _update_output_stream_digest(
                digest,
                assignment_id=assignment.assignment_id,
                tensor_name="student_quotient4",
                value=mlx_quotient,
            )
        for digest in (first_mlx_vjp_digest, first_mlx_combined_digest):
            _update_output_stream_digest(
                digest,
                assignment_id=assignment.assignment_id,
                tensor_name="student_input_vjp",
                value=mlx_vjp,
            )
        per_pair.append(
            {
                "assignment_id": assignment.assignment_id,
                "pair_index": assignment.pair_index,
                "checkpoint_name": assignment.checkpoint_name,
                "checkpoint_epoch": assignment.checkpoint_epoch,
                "split": assignment.split,
                "forward_numpy_fp32_authority": forward,
                "full_input_vjp_numpy_fp32_decisive": vjp,
                "boundary_input_vjp_diagnostic": boundary_metric,
                "mlx_vs_numpy_forward": parity_forward,
                "mlx_vs_numpy_input_vjp": parity_vjp,
                "timing_ms": {
                    "student_forward": forward_ms,
                    "student_forward_input_vjp": forward_vjp_ms,
                },
            }
        )

    # A second full n600 NumPy-authority + MLX-advisory evaluation pass is
    # untimed and excluded from every charged C_S row. Hashing both ordered
    # quotient+VJP byte streams proves cohort determinism on the verdict path
    # as well as the prospective execution path; a canary or terminal-result
    # reread is insufficient.
    second_forward_digest = hashlib.sha256()
    second_vjp_digest = hashlib.sha256()
    second_combined_digest = hashlib.sha256()
    second_mlx_forward_digest = hashlib.sha256()
    second_mlx_vjp_digest = hashlib.sha256()
    second_mlx_combined_digest = hashlib.sha256()
    for assignment in all_rows:
        row = _validate_fit_row_arrays(validated_cache.load_row(assignment.pair_index))
        repeat_numpy_quotient = student_forward_numpy(
            row["rendered_frame"], architecture, state["best_parameters"]
        )
        repeat_numpy_vjp = student_ce_input_vjp_numpy(
            row["rendered_frame"], architecture, state["best_parameters"], row["labels"]
        )
        repeat_quotient, repeat_vjp, _ignored_forward_ms, _ignored_forward_vjp_ms = (
            backend_impl.predict_and_vjp(
                best_tree,
                frame=row["rendered_frame"],
                labels=row["labels"],
                architecture=architecture,
            )
        )
        for digest in (second_forward_digest, second_combined_digest):
            _update_output_stream_digest(
                digest,
                assignment_id=assignment.assignment_id,
                tensor_name="student_quotient4",
                value=repeat_numpy_quotient,
            )
        for digest in (second_vjp_digest, second_combined_digest):
            _update_output_stream_digest(
                digest,
                assignment_id=assignment.assignment_id,
                tensor_name="student_input_vjp",
                value=repeat_numpy_vjp,
            )
        for digest in (second_mlx_forward_digest, second_mlx_combined_digest):
            _update_output_stream_digest(
                digest,
                assignment_id=assignment.assignment_id,
                tensor_name="student_quotient4",
                value=repeat_quotient,
            )
        for digest in (second_mlx_vjp_digest, second_mlx_combined_digest):
            _update_output_stream_digest(
                digest,
                assignment_id=assignment.assignment_id,
                tensor_name="student_input_vjp",
                value=repeat_vjp,
            )
    numpy_repeat_verified = bool(
        first_forward_digest.digest() == second_forward_digest.digest()
        and first_vjp_digest.digest() == second_vjp_digest.digest()
        and first_combined_digest.digest() == second_combined_digest.digest()
    )
    mlx_repeat_verified = bool(
        first_mlx_forward_digest.digest() == second_mlx_forward_digest.digest()
        and first_mlx_vjp_digest.digest() == second_mlx_vjp_digest.digest()
        and first_mlx_combined_digest.digest() == second_mlx_combined_digest.digest()
    )
    deterministic_repeat = {
        "scope": "full_ordered_n600_numpy_fp32_authority_forward_and_input_vjp_stream",
        "numerical_authority": "numpy_fp32",
        "pair_count": len(all_rows),
        "timed": False,
        "charged_student_timing_includes_repeat": False,
        "first_forward_sha256": first_forward_digest.hexdigest(),
        "second_forward_sha256": second_forward_digest.hexdigest(),
        "forward_equal": first_forward_digest.digest() == second_forward_digest.digest(),
        "first_input_vjp_sha256": first_vjp_digest.hexdigest(),
        "second_input_vjp_sha256": second_vjp_digest.hexdigest(),
        "input_vjp_equal": first_vjp_digest.digest() == second_vjp_digest.digest(),
        "first_combined_sha256": first_combined_digest.hexdigest(),
        "second_combined_sha256": second_combined_digest.hexdigest(),
        "combined_equal": first_combined_digest.digest() == second_combined_digest.digest(),
        "authority_verified": numpy_repeat_verified,
        "mlx_advisory": {
            "scope": "full_ordered_n600_mlx_advisory_forward_and_input_vjp_stream",
            "pair_count": len(all_rows),
            "timed": False,
            "charged_student_timing_includes_repeat": False,
            "first_forward_sha256": first_mlx_forward_digest.hexdigest(),
            "second_forward_sha256": second_mlx_forward_digest.hexdigest(),
            "forward_equal": (
                first_mlx_forward_digest.digest() == second_mlx_forward_digest.digest()
            ),
            "first_input_vjp_sha256": first_mlx_vjp_digest.hexdigest(),
            "second_input_vjp_sha256": second_mlx_vjp_digest.hexdigest(),
            "input_vjp_equal": first_mlx_vjp_digest.digest() == second_mlx_vjp_digest.digest(),
            "first_combined_sha256": first_mlx_combined_digest.hexdigest(),
            "second_combined_sha256": second_mlx_combined_digest.hexdigest(),
            "combined_equal": (
                first_mlx_combined_digest.digest() == second_mlx_combined_digest.digest()
            ),
            "advisory_verified": mlx_repeat_verified,
        },
        "all_required_streams_equal": numpy_repeat_verified and mlx_repeat_verified,
    }

    heldout_ids = {row.assignment_id for row in validated_cache.heldout_rows}
    heldout_forward = [row for row in forward_rows if row["assignment_id"] in heldout_ids]
    heldout_vjp = [row for row in vjp_rows if row["assignment_id"] in heldout_ids]
    heldout_boundary = [row for row in boundary_vjp_rows if row["assignment_id"] in heldout_ids]
    teacher_timings = _teacher_timing_inputs_from_manifest(
        manifest,
        bundle_root=validated_cache.bundle_root,
        teacher_source_custody=validated_cache.teacher_source_custody,
        backend_impl=backend_impl,
    )
    best_parameter_blob = serialize_student_parameters(
        architecture, state["best_parameters"]
    )
    forward_n600 = aggregate_forward_pair_metrics(forward_rows)
    heldout_forward_summary = aggregate_forward_pair_metrics(heldout_forward)
    vjp_n600 = aggregate_vjp_pair_metrics(vjp_rows)
    heldout_vjp_summary = aggregate_vjp_pair_metrics(heldout_vjp)
    parity_forward_n600 = aggregate_forward_pair_metrics(parity_forward_rows)
    parity_vjp_n600 = aggregate_vjp_pair_metrics(parity_vjp_rows)
    result: dict[str, Any] = {
        "schema": FIT_RESULT_SCHEMA,
        "fit_driver_status": FIT_DRIVER_STATUS,
        "n_pairs": len(all_rows),
        "train_pairs": len(train_rows),
        "heldout_pairs": len(validated_cache.heldout_rows),
        "teacher_calls": 0,
        "backend": "mlx",
        "numerical_reference": "numpy_fp32",
        "measurement_axis": MEASUREMENT_AXIS,
        "hardware_descriptor": dict(backend_impl.hardware_descriptor),
        "hardware_fingerprint_sha256": backend_impl.hardware_fingerprint_sha256,
        "student_size": student_size,
        "measured_tier": "training_gradient",
        "student_anchor_cadence": None,
        "fit_epochs": fit_epochs,
        "fit_steps": state["optimizer_step"],
        "fit_policy": policy.to_dict(),
        "hashes": hashes,
        "best_epoch": state["best_epoch"],
        "best_objective": state["best_objective"],
        "best_parameters_sha256": hashlib.sha256(best_parameter_blob).hexdigest(),
        "best_parameters_blob_bytes": len(best_parameter_blob),
        "parameter_layout_sha256": hashes["parameter_layout_sha256"],
        "source_custody_sha256": (
            validated_cache.teacher_source_custody.custody_sha256
        ),
        "teacher_source_custody_sha256": (
            validated_cache.teacher_source_custody.custody_sha256
        ),
        "cache_manifest_sha256": validated_cache.manifest_sha256,
        "cache_manifest_file_sha256": hashes["cache_manifest_file_sha256"],
        "quotient_basis_sha256": (
            validated_cache.teacher_source_custody.helmert_basis_sha256
        ),
        "post_r_input_surface_sha256": (
            validated_cache.teacher_source_custody.post_r_input_surface_sha256
        ),
        "durable_best_parameters_blob": {
            "status": "PROBE_MUST_MATERIALIZE_FROM_COMPLETION_CHECKPOINT",
            "sha256": hashlib.sha256(best_parameter_blob).hexdigest(),
            "bytes": len(best_parameter_blob),
            "path": None,
        },
        "source_custody": {
            "teacher_source_custody_sha256": (
                validated_cache.teacher_source_custody.custody_sha256
            ),
            "cache_manifest_sha256": validated_cache.manifest_sha256,
            "cache_manifest_file_sha256": hashes["cache_manifest_file_sha256"],
            "r_operator_sha256": validated_cache.teacher_source_custody.r_operator_sha256,
            "scorer_sha256": validated_cache.teacher_source_custody.scorer_sha256,
            "scalar_objective_sha256": (
                validated_cache.teacher_source_custody.scalar_objective_sha256
            ),
            "quotient_basis_sha256": (
                validated_cache.teacher_source_custody.helmert_basis_sha256
            ),
            "helmert_basis_sha256": (
                validated_cache.teacher_source_custody.helmert_basis_sha256
            ),
            "post_r_input_surface_sha256": (
                validated_cache.teacher_source_custody.post_r_input_surface_sha256
            ),
        },
        "forward_fidelity": {
            "n600": forward_n600,
            "heldout_n120": heldout_forward_summary,
        },
        "vjp_fidelity_decisive_full_vector": {
            "n600": vjp_n600,
            "heldout_n120": heldout_vjp_summary,
        },
        "vjp_fidelity_boundary_diagnostic": {
            "n600": aggregate_vjp_pair_metrics(boundary_vjp_rows),
            "heldout_n120": aggregate_vjp_pair_metrics(heldout_boundary),
        },
        "framework_parity": {
            "primary_teacher_fidelity_uses_mlx": False,
            "primary_teacher_fidelity_authority": "numpy_fp32",
            "mlx_role": "advisory_parity_timing_and_repeat_only",
            "mlx_vs_numpy_forward_n600": parity_forward_n600,
            "mlx_vs_numpy_input_vjp_n600": parity_vjp_n600,
        },
        "deterministic_repeat": deterministic_repeat,
        "deterministic_repeat_verified": bool(
            deterministic_repeat["all_required_streams_equal"]
        ),
        "charged_timings": {
            "warmup_excluded": True,
            "student_forward": _metric_timing_summary(forward_ms_rows),
            "student_forward_input_vjp": _metric_timing_summary(forward_vjp_ms_rows),
            "anchor_update": _metric_timing_summary(state["train_step_ms_history"]),
            "exact_teacher_inputs": teacher_timings,
            "teacher_timing_receipt_sha256": teacher_timings[
                "teacher_timing_receipt_sha256"
            ],
            "teacher_timing_receipt_path": (
                None
                if teacher_timings["teacher_timing_receipt"] is None
                else teacher_timings["teacher_timing_receipt"]["path"]
            ),
            "student_teacher_axis_matched": (
                teacher_timings["measurement_axis"] == MEASUREMENT_AXIS
            ),
            "fully_charged_economics_ready": (
                teacher_timings["status"]
                == "CONTENT_BOUND_MATCHED_AXIS_RECEIPT_VERIFIED"
            ),
            "student_anchor_cadence_required_from_typed_DSL": True,
            "inherited_487_speed_claim": False,
        },
        "fit_loss": {
            "count": len(state["train_loss_history"]),
            "first": state["train_loss_history"][0],
            "last": state["train_loss_history"][-1],
            "minimum": float(min(state["train_loss_history"])),
        },
        "per_pair": per_pair,
        "authority": {
            "research_only": True,
            "means_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "teacher_regeneration": False,
            "synthetic_fallback": False,
            "cpu_fallback": False,
            "full_input_vjp_is_decisive": True,
            "primary_teacher_fidelity_numerical_authority": "numpy_fp32",
            "mlx_outputs_used_for_primary_teacher_gate": False,
            "boundary_input_vjp_is_diagnostic_only": True,
        },
    }
    _emit_fit_checkpoint(
        checkpoint_callback=checkpoint_callback,
        stage="heldout_measurement",
        hashes=hashes,
        backend=backend_impl,
        architecture=architecture,
        parameters=state["parameters"],
        first_moment=state["first_moment"],
        second_moment=state["second_moment"],
        best_parameters=state["best_parameters"],
        optimizer_step=state["optimizer_step"],
        next_epoch=fit_epochs,
        next_train_offset=0,
        epoch_loss_sum=0.0,
        epoch_loss_count=0,
        best_objective=state["best_objective"],
        best_epoch=state["best_epoch"],
        train_loss_history=state["train_loss_history"],
        train_step_ms_history=state["train_step_ms_history"],
        terminal_result=result,
    )
    _emit_fit_checkpoint(
        checkpoint_callback=checkpoint_callback,
        stage="completion",
        hashes=hashes,
        backend=backend_impl,
        architecture=architecture,
        parameters=state["parameters"],
        first_moment=state["first_moment"],
        second_moment=state["second_moment"],
        best_parameters=state["best_parameters"],
        optimizer_step=state["optimizer_step"],
        next_epoch=fit_epochs,
        next_train_offset=0,
        epoch_loss_sum=0.0,
        epoch_loss_count=0,
        best_objective=state["best_objective"],
        best_epoch=state["best_epoch"],
        train_loss_history=state["train_loss_history"],
        train_step_ms_history=state["train_step_ms_history"],
        terminal_result=result,
    )
    return result


def surrogate_economics(
    *,
    tier: Literal["forward_advisory", "training_gradient"],
    student_anchor_cadence: int,
    student_step_ms: float,
    exact_teacher_step_ms: float,
    anchor_update_ms: float,
    tier_gate_passed: bool,
) -> dict[str, Any]:
    """Apply the fully charged cadence law for one independently gated tier."""

    if tier not in {"forward_advisory", "training_gradient"}:
        raise StudentContractError("unknown student economics tier")
    if (
        isinstance(student_anchor_cadence, bool)
        or not isinstance(student_anchor_cadence, int)
        or student_anchor_cadence < 1
    ):
        raise StudentContractError("student_anchor_cadence must be an integer >= 1")
    costs = (student_step_ms, exact_teacher_step_ms, anchor_update_ms)
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in costs):
        raise StudentContractError("student economics costs must be finite and nonnegative")
    if exact_teacher_step_ms <= 0.0:
        raise StudentContractError("exact_teacher_step_ms must be positive")
    if not isinstance(tier_gate_passed, bool):
        raise StudentContractError("tier_gate_passed must be boolean")
    charged = float(student_step_ms) + (
        float(exact_teacher_step_ms) + float(anchor_update_ms)
    ) / student_anchor_cadence
    kill_fraction = 1.0 - charged / float(exact_teacher_step_ms)
    component_pays = charged < float(exact_teacher_step_ms)
    return {
        "tier": tier,
        "student_anchor_cadence": student_anchor_cadence,
        "student_step_ms": float(student_step_ms),
        "exact_teacher_step_ms": float(exact_teacher_step_ms),
        "anchor_update_ms": float(anchor_update_ms),
        "charged_ms_per_step": charged,
        "teacher_slice_kill_fraction": kill_fraction,
        "component_pays": component_pays,
        "tier_gate_passed": tier_gate_passed,
        "pays": bool(component_pays and tier_gate_passed),
        "inclusive_95_kill_feasible": bool(kill_fraction >= 0.95 and tier_gate_passed),
        "score_or_pointer_authority": False,
    }


def cadence_composition(
    *,
    student_anchor_cadence: int,
    inner_costate_reuse_cadence: int | None,
) -> dict[str, Any]:
    """Keep student refresh K independent from optional #487 inner reuse K<=2."""

    if (
        isinstance(student_anchor_cadence, bool)
        or not isinstance(student_anchor_cadence, int)
        or student_anchor_cadence < 1
    ):
        raise StudentContractError("student_anchor_cadence must be an integer >= 1")
    if inner_costate_reuse_cadence is not None and (
        isinstance(inner_costate_reuse_cadence, bool)
        or not isinstance(inner_costate_reuse_cadence, int)
        or not 1 <= inner_costate_reuse_cadence <= 2
    ):
        raise StudentContractError("inner_costate_reuse_cadence must be None, 1, or 2")
    return {
        "student_anchor_cadence": student_anchor_cadence,
        "inner_costate_reuse_cadence": inner_costate_reuse_cadence,
        "student_cadence_capped_by_inner_controller": False,
        "inherited_speed_claim": False,
        "economics_requires_separately_measured_charged_costs": True,
    }


__all__ = [
    "AUTHORITY_SCOPE",
    "CACHE_GENERATION_AXIS",
    "CACHE_SCHEMA",
    "CHECKPOINTS",
    "CLASS_COUNT",
    "COSTATE_SHAPE",
    "COSTATE_SURFACE_IDENTITY",
    "FIT_DRIVER_STATUS",
    "FIT_RESULT_SCHEMA",
    "FIT_STATE_SCHEMA",
    "FRAME_SHAPE",
    "HELDOUT_COUNT",
    "HELMERT_BASIS_5X4",
    "HELMERT_BASIS_SHA256",
    "MEASUREMENT_AXIS",
    "N600",
    "PARAMETER_SCHEMA",
    "QUOTIENT_DIM",
    "QUOTIENT_SHAPE",
    "RESEARCH_ONLY",
    "R_OPERATOR_IDENTITY",
    "SCALAR_OBJECTIVE_IDENTITY",
    "SEGNET_ARCHITECTURE_IDENTITY",
    "SOURCE_KIND",
    "STUDENT_SIZE_WIDTHS",
    "TEACHER_SOURCE_CUSTODY_SCHEMA",
    "TRAIN_COUNT",
    "CacheArtifact",
    "CacheRow",
    "CachedStudentFitPolicy",
    "CustodyFile",
    "StudentArchitecture",
    "StudentContractError",
    "TeacherSourceCustody",
    "ValidatedN600Cache",
    "aggregate_forward_pair_metrics",
    "aggregate_vjp_pair_metrics",
    "architecture_for_size",
    "boundary_mask_from_labels_numpy",
    "cached_student_fit_policy",
    "cadence_composition",
    "coordinate_channels_numpy",
    "deserialize_student_parameters",
    "fit_measure_cached_student",
    "forward_pair_metrics",
    "initialize_student_parameters",
    "logits5_from_quotient4_mlx",
    "logits5_from_quotient4_numpy",
    "logits5_from_quotient4_torch",
    "parameter_layout_sha256",
    "quotient4_from_logits5_numpy",
    "quotient4_from_logits5_torch",
    "quotient_cross_entropy_cotangent_numpy",
    "serialize_student_parameters",
    "student_ce_input_vjp_mlx",
    "student_ce_input_vjp_numpy",
    "student_forward_mlx",
    "student_forward_numpy",
    "student_forward_torch",
    "student_input_vjp_mlx",
    "student_input_vjp_numpy",
    "student_input_vjp_torch",
    "student_parameters_mlx",
    "student_value_fit_loss_mlx",
    "student_value_fit_value_and_grad_mlx",
    "surrogate_economics",
    "validate_n600_cache_manifest",
    "validate_n600_cache_manifest_structure",
    "validate_student_parameters",
    "vjp_pair_metrics",
]
