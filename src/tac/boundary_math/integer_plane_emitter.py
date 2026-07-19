# SPDX-License-Identifier: MIT
"""Deterministic two-plane uint8 witness emitter for the v10 C2 build.

The module is intentionally narrower than a trainer.  Frozen solved structure
is expanded by one explicitly named quotient-residual parameter group, rounded
once at the scorer-plane boundary, and (when requested) lifted through the
certified factor-2 integer lattice.  It has no score, launch, archive, or
candidate-admission authority.

Authority is NumPy-fp32 plus exact uint8 bytes.  Torch uses the repository's
saturation-aware :class:`~tac.quantization.Uint8STE`; MLX is imported lazily and
uses the equivalent clip/stop-gradient expression.  The receiver expansion is
pair-independent and shape-parallel by construction: there is no cross-pair
state or recurrence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Final, Literal

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)

SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
RGB_CHANNELS: Final = 3
PLANE_COUNT: Final = 2
FROZEN_SEGNET_HEAD_FILE_SHA256: Final = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
MEASURED_U4_SINGULAR_VALUES: Final = (
    3.128376325627011,
    2.1542713872702617,
    2.024707869857505,
    1.796263835653701,
)
FROZEN_SEGNET_HEAD_WEIGHT_SHA256: Final = "f89b12dc58bd5a311ed62aea8136772a2facaa6cbdcad5ede558a03eb863e9f9"
FROZEN_U4_F64_SHA256: Final = "1d62c1fe316214dd7b370e52c0927015c8ca4dca91bde26a2c88c68bdb6b3f62"
FROZEN_PAIR_COEFFICIENTS_F64_SHA256: Final = "86b784d04062249530499d8f6a4cc05a3423eddf0d85321ff690673053443f51"
PAIR_INDICES: Final = tuple(combinations(range(5), 2))


class IntegerPlaneEmitterError(ValueError):
    """Fail-closed C2 contract violation."""


class MLXUnavailableError(RuntimeError):
    """MLX execution is unavailable; no parity may be inferred."""


def _immutable_f32(value: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.float32:
        raise IntegerPlaneEmitterError(f"{name} must have exact dtype float32")
    if not np.isfinite(raw).all():
        raise IntegerPlaneEmitterError(f"{name} must contain only finite values")
    out = np.array(raw, dtype=np.float32, copy=True, order="C")
    out.setflags(write=False)
    return out


def _immutable_u8(value: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.uint8:
        raise IntegerPlaneEmitterError(f"{name} must have exact dtype uint8")
    out = np.array(raw, dtype=np.uint8, copy=True, order="C")
    out.setflags(write=False)
    return out


def _immutable_f64(value: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.float64:
        raise IntegerPlaneEmitterError(f"{name} must have exact dtype float64")
    if not np.isfinite(raw).all():
        raise IntegerPlaneEmitterError(f"{name} must contain only finite values")
    out = np.array(raw, dtype="<f8", copy=True, order="C")
    out.setflags(write=False)
    return out


def _raw_f64_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _raw_f32_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise IntegerPlaneEmitterError(f"{name} must be a lowercase SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise IntegerPlaneEmitterError(f"{name} must be a lowercase SHA-256 hex digest") from exc
    if value != value.lower():
        raise IntegerPlaneEmitterError(f"{name} must be lowercase")
    return value


@dataclass(frozen=True)
class IntegerPlaneGeometry:
    """Frozen evaluator geometry.  Alternate geometries are not C2."""

    scorer_height: int = SCORER_HEIGHT
    scorer_width: int = SCORER_WIDTH
    channels: int = RGB_CHANNELS
    plane_count: int = PLANE_COUNT
    camera_height: int = CAMERA_HEIGHT
    camera_width: int = CAMERA_WIDTH

    def __post_init__(self) -> None:
        actual = (
            self.scorer_height,
            self.scorer_width,
            self.channels,
            self.plane_count,
            self.camera_height,
            self.camera_width,
        )
        expected = (
            SCORER_HEIGHT,
            SCORER_WIDTH,
            RGB_CHANNELS,
            PLANE_COUNT,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
        )
        if actual != expected or any(type(value) is not int for value in actual):
            raise IntegerPlaneEmitterError(f"C2 geometry is sealed to {expected}")

    @property
    def scorer_shape(self) -> tuple[int, int, int]:
        return self.scorer_height, self.scorer_width, self.channels

    @property
    def camera_shape(self) -> tuple[int, int, int]:
        return self.camera_height, self.camera_width, self.channels


def deterministic_coordinate_basis(residual_width: int) -> np.ndarray:
    """Return a deterministic non-Fourier polynomial coordinate dictionary."""

    if type(residual_width) is not int or residual_width < 1 or residual_width > 64:
        raise IntegerPlaneEmitterError("residual_width must be an integer in [1,64]")
    yy = np.linspace(-1.0, 1.0, SCORER_HEIGHT, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, SCORER_WIDTH, dtype=np.float32)[None, :]
    x = np.broadcast_to(xx, (SCORER_HEIGHT, SCORER_WIDTH))
    y = np.broadcast_to(yy, (SCORER_HEIGHT, SCORER_WIDTH))
    features: list[np.ndarray] = [
        np.ones_like(x),
        x,
        y,
        x * y,
        x * x,
        y * y,
    ]
    degree = 3
    while len(features) < residual_width:
        for x_power in range(degree + 1):
            y_power = degree - x_power
            if x_power + y_power <= 2:
                continue
            features.append(np.asarray((x**x_power) * (y**y_power), dtype=np.float32))
            if len(features) == residual_width:
                break
        degree += 1
    out = np.stack(features[:residual_width], axis=-1).astype(np.float32, copy=False)
    # Fixed per-feature scale prevents high-degree monomials from becoming an
    # implicit capacity change while preserving the span.
    norms = np.sqrt(np.mean(out * out, axis=(0, 1), dtype=np.float32), dtype=np.float32)
    out = np.divide(out, np.maximum(norms, np.float32(1e-6)), dtype=np.float32)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class StructuredEmitterState:
    """Immutable solved base and frozen coordinate topology."""

    base: np.ndarray
    coordinate_basis: np.ndarray
    topology_id: str = "quotient_residual_polynomial_basis_v1"
    pair_parallel: bool = True
    cross_pair_autoregression: bool = False

    def __post_init__(self) -> None:
        base = _immutable_f32(self.base, name="base")
        basis = _immutable_f32(self.coordinate_basis, name="coordinate_basis")
        expected_tail = (PLANE_COUNT, SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS)
        if base.ndim != 5 or base.shape[1:] != expected_tail or base.shape[0] < 1:
            raise IntegerPlaneEmitterError(f"base must have shape [N,{expected_tail}]")
        if basis.ndim != 3 or basis.shape[:2] != (SCORER_HEIGHT, SCORER_WIDTH):
            raise IntegerPlaneEmitterError("coordinate_basis must have shape [384,512,K]")
        if not 1 <= basis.shape[2] <= 64:
            raise IntegerPlaneEmitterError("coordinate basis width must be in [1,64]")
        if not isinstance(self.topology_id, str) or not self.topology_id:
            raise IntegerPlaneEmitterError("topology_id must be nonempty")
        if self.pair_parallel is not True or self.cross_pair_autoregression is not False:
            raise IntegerPlaneEmitterError("C2 expansion must be pair-parallel with no cross-pair autoregression")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "coordinate_basis", basis)

    @classmethod
    def from_base(cls, base: np.ndarray, *, residual_width: int) -> StructuredEmitterState:
        return cls(base=base, coordinate_basis=deterministic_coordinate_basis(residual_width))

    @property
    def pair_count(self) -> int:
        return int(self.base.shape[0])

    @property
    def residual_width(self) -> int:
        return int(self.coordinate_basis.shape[2])

    @property
    def topology_sha256(self) -> str:
        return _sha256_array(self.coordinate_basis)


@dataclass(frozen=True)
class QuotientResidualState:
    """Every and only trainable C2 array: independent codes plus shared head."""

    pair_plane_codes: np.ndarray
    shared_rgb_head: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        codes = _immutable_f32(self.pair_plane_codes, name="pair_plane_codes")
        head = _immutable_f32(self.shared_rgb_head, name="shared_rgb_head")
        if codes.ndim != 3 or codes.shape[1] != PLANE_COUNT or codes.shape[2] < 1:
            raise IntegerPlaneEmitterError("pair_plane_codes must have shape [N,2,K]")
        if head.shape != (codes.shape[2], RGB_CHANNELS):
            raise IntegerPlaneEmitterError("shared_rgb_head must have shape [K,3]")
        if type(self.seed) is not int or self.seed < 0 or self.seed >= 2**64:
            raise IntegerPlaneEmitterError("seed must be an integer in [0,2**64)")
        object.__setattr__(self, "pair_plane_codes", codes)
        object.__setattr__(self, "shared_rgb_head", head)

    @classmethod
    def fresh(
        cls,
        structured: StructuredEmitterState,
        *,
        seed: int,
        scale: float = 0.01,
    ) -> QuotientResidualState:
        if type(seed) is not int or seed < 0 or seed >= 2**64:
            raise IntegerPlaneEmitterError("seed must be an integer in [0,2**64)")
        if not np.isfinite(scale) or scale < 0.0:
            raise IntegerPlaneEmitterError("scale must be finite and nonnegative")
        rng = np.random.Generator(np.random.PCG64(seed))
        codes = rng.standard_normal(
            (structured.pair_count, PLANE_COUNT, structured.residual_width), dtype=np.float32
        ) * np.float32(scale)
        head = rng.standard_normal((structured.residual_width, RGB_CHANNELS), dtype=np.float32) * np.float32(scale)
        return cls(codes.astype(np.float32), head.astype(np.float32), seed)

    @classmethod
    def deleted(cls, structured: StructuredEmitterState, *, seed: int = 0) -> QuotientResidualState:
        return cls(
            np.zeros((structured.pair_count, PLANE_COUNT, structured.residual_width), dtype=np.float32),
            np.zeros((structured.residual_width, RGB_CHANNELS), dtype=np.float32),
            seed,
        )


@dataclass(frozen=True)
class CapacitySignature:
    pair_count: int
    plane_count: int
    residual_width: int
    code_parameters: int
    head_parameters: int
    total_parameters: int
    topology_sha256: str

    @classmethod
    def from_states(cls, structured: StructuredEmitterState, residual: QuotientResidualState) -> CapacitySignature:
        _validate_state_pair(structured, residual)
        codes = int(residual.pair_plane_codes.size)
        head = int(residual.shared_rgb_head.size)
        return cls(
            structured.pair_count,
            PLANE_COUNT,
            structured.residual_width,
            codes,
            head,
            codes + head,
            structured.topology_sha256,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _validate_state_pair(structured: StructuredEmitterState, residual: QuotientResidualState) -> None:
    if not isinstance(structured, StructuredEmitterState):
        raise IntegerPlaneEmitterError("structured must be StructuredEmitterState")
    if not isinstance(residual, QuotientResidualState):
        raise IntegerPlaneEmitterError("residual must be QuotientResidualState")
    expected = (structured.pair_count, PLANE_COUNT, structured.residual_width)
    if residual.pair_plane_codes.shape != expected:
        raise IntegerPlaneEmitterError(f"residual codes must have shape {expected}")


def numpy_precursor(structured: StructuredEmitterState, residual: QuotientResidualState) -> np.ndarray:
    """Expand all pairs/planes in one deterministic, independent contraction."""

    _validate_state_pair(structured, residual)
    perturbation = np.zeros_like(structured.base, dtype=np.float32)
    # A fixed scalar-feature accumulation order is shared by NumPy/Torch/MLX.
    # This avoids backend-specific GEMM reduction order at uint8 half-ties.
    for feature_index in range(structured.residual_width):
        amplitude = np.multiply(
            residual.pair_plane_codes[:, :, feature_index, None],
            residual.shared_rgb_head[feature_index][None, None, :],
            dtype=np.float32,
        )
        term = np.multiply(
            structured.coordinate_basis[None, None, :, :, feature_index, None],
            amplitude[:, :, None, None, :],
            dtype=np.float32,
        )
        perturbation = np.add(perturbation, term, dtype=np.float32)
    out = np.add(structured.base, perturbation, dtype=np.float32)
    if out.dtype != np.float32 or not np.isfinite(out).all():
        raise IntegerPlaneEmitterError("NumPy precursor violated fp32/finite contract")
    return out


def numpy_uint8(
    structured: StructuredEmitterState,
    residual: QuotientResidualState,
    *,
    require_distinct_planes: bool = True,
) -> np.ndarray:
    precursor = numpy_precursor(structured, residual)
    clipped = np.clip(precursor, np.float32(0.0), np.float32(255.0))
    out = np.rint(clipped).astype(np.uint8)
    return validate_scorer_uint8(out, require_distinct_planes=require_distinct_planes)


def _torch_validate_float_tensor(value: object, *, name: str, shape: tuple[int, ...]) -> None:
    import torch

    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise IntegerPlaneEmitterError(f"{name} must be a torch.float32 Tensor")
    if tuple(value.shape) != shape:
        raise IntegerPlaneEmitterError(f"{name} must have shape {shape}")
    if not bool(torch.isfinite(value).all().item()):
        raise IntegerPlaneEmitterError(f"{name} must contain only finite values")


def torch_precursor(
    structured: StructuredEmitterState,
    pair_plane_codes: object,
    shared_rgb_head: object,
) -> object:
    """Torch differentiable precursor; only the two supplied tensors are trainable."""

    import torch

    code_shape = (structured.pair_count, PLANE_COUNT, structured.residual_width)
    head_shape = (structured.residual_width, RGB_CHANNELS)
    _torch_validate_float_tensor(pair_plane_codes, name="pair_plane_codes", shape=code_shape)
    _torch_validate_float_tensor(shared_rgb_head, name="shared_rgb_head", shape=head_shape)
    assert isinstance(pair_plane_codes, torch.Tensor)
    assert isinstance(shared_rgb_head, torch.Tensor)
    device = pair_plane_codes.device
    if shared_rgb_head.device != device:
        raise IntegerPlaneEmitterError("Torch residual tensors must share one device")
    base = torch.as_tensor(np.array(structured.base), dtype=torch.float32, device=device)
    basis = torch.as_tensor(np.array(structured.coordinate_basis), dtype=torch.float32, device=device)
    perturbation = torch.zeros_like(base)
    for feature_index in range(structured.residual_width):
        amplitude = pair_plane_codes[:, :, feature_index, None] * shared_rgb_head[feature_index][None, None, :]
        perturbation = perturbation + (basis[None, None, :, :, feature_index, None] * amplitude[:, :, None, None, :])
    return base + perturbation


def torch_uint8(
    structured: StructuredEmitterState,
    pair_plane_codes: object,
    shared_rgb_head: object,
) -> object:
    """Integer-valued float32 Torch tensor with saturation-aware gradients."""

    from tac.quantization import Uint8STE

    return Uint8STE.apply(torch_precursor(structured, pair_plane_codes, shared_rgb_head))


def torch_uint8_bytes(
    structured: StructuredEmitterState,
    pair_plane_codes: object,
    shared_rgb_head: object,
    *,
    require_distinct_planes: bool = True,
) -> np.ndarray:
    rounded = torch_uint8(structured, pair_plane_codes, shared_rgb_head)
    out = rounded.detach().cpu().numpy().astype(np.uint8, copy=False)
    return validate_scorer_uint8(out, require_distinct_planes=require_distinct_planes)


def _load_mlx() -> object:
    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MLXUnavailableError("MLX is not importable; parity is unmeasured") from exc
    return mx


def mlx_precursor(
    structured: StructuredEmitterState,
    pair_plane_codes: object,
    shared_rgb_head: object,
) -> object:
    """Lazy MLX expansion.  Evaluation errors remain explicit environment blockers."""

    mx = _load_mlx()
    expected_codes = (structured.pair_count, PLANE_COUNT, structured.residual_width)
    expected_head = (structured.residual_width, RGB_CHANNELS)
    if tuple(getattr(pair_plane_codes, "shape", ())) != expected_codes:
        raise IntegerPlaneEmitterError(f"MLX pair_plane_codes must have shape {expected_codes}")
    if tuple(getattr(shared_rgb_head, "shape", ())) != expected_head:
        raise IntegerPlaneEmitterError(f"MLX shared_rgb_head must have shape {expected_head}")
    if getattr(pair_plane_codes, "dtype", None) != mx.float32 or getattr(shared_rgb_head, "dtype", None) != mx.float32:
        raise IntegerPlaneEmitterError("MLX residual tensors must have dtype float32")
    try:
        codes_finite = mx.all(mx.isfinite(pair_plane_codes))
        head_finite = mx.all(mx.isfinite(shared_rgb_head))
        mx.eval(codes_finite, head_finite)
        codes_finite_value = bool(np.asarray(codes_finite).reshape(()))
        head_finite_value = bool(np.asarray(head_finite).reshape(()))
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MLXUnavailableError("MLX finite-value validation is unavailable; parity is unmeasured") from exc
    if not codes_finite_value or not head_finite_value:
        raise IntegerPlaneEmitterError("MLX residual tensors must contain only finite values")
    try:
        base = mx.array(np.array(structured.base), dtype=mx.float32)
        basis = mx.array(np.array(structured.coordinate_basis), dtype=mx.float32)
        perturbation = mx.zeros_like(base)
        for feature_index in range(structured.residual_width):
            amplitude = pair_plane_codes[:, :, feature_index, None] * shared_rgb_head[feature_index][None, None, :]
            perturbation = perturbation + (
                basis[None, None, :, :, feature_index, None] * amplitude[:, :, None, None, :]
            )
        return base + perturbation
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MLXUnavailableError("MLX array evaluation is unavailable; parity is unmeasured") from exc


def mlx_uint8(
    structured: StructuredEmitterState,
    pair_plane_codes: object,
    shared_rgb_head: object,
) -> object:
    """MLX clip-round STE: derivative of clip, exact rounded forward value."""

    mx = _load_mlx()
    precursor = mlx_precursor(structured, pair_plane_codes, shared_rgb_head)
    try:
        clipped = mx.clip(precursor, 0.0, 255.0)
        return clipped + mx.stop_gradient(mx.round(clipped) - clipped)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MLXUnavailableError("MLX STE execution unavailable; parity is unmeasured") from exc


def validate_scorer_uint8(value: np.ndarray, *, require_distinct_planes: bool = True) -> np.ndarray:
    out = _immutable_u8(value, name="scorer planes")
    if out.ndim != 5 or out.shape[0] < 1 or out.shape[1:] != (
        PLANE_COUNT,
        SCORER_HEIGHT,
        SCORER_WIDTH,
        RGB_CHANNELS,
    ):
        raise IntegerPlaneEmitterError("scorer planes must have shape [N,2,384,512,3]")
    if require_distinct_planes:
        collapsed = [
            pair_index for pair_index in range(out.shape[0]) if np.array_equal(out[pair_index, 0], out[pair_index, 1])
        ]
        if collapsed:
            raise IntegerPlaneEmitterError(f"copied/collapsed scorer planes refused for pairs {collapsed}")
    return out


@lru_cache(maxsize=1)
def factor2_operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.build(
        camera_h=CAMERA_HEIGHT,
        camera_w=CAMERA_WIDTH,
        scorer_h=SCORER_HEIGHT,
        scorer_w=SCORER_WIDTH,
    )


@dataclass(frozen=True)
class LatticePlaneProof:
    pair_index: int
    plane_index: int
    scorer_sha256: str
    camera_sha256: str
    denominator: int
    numerator_equal_values: int
    scorer_values: int
    numerator_exact: bool
    certified_exact: bool

    def __post_init__(self) -> None:
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise IntegerPlaneEmitterError("lattice proof pair_index must be a nonnegative integer")
        if type(self.plane_index) is not int or self.plane_index not in range(PLANE_COUNT):
            raise IntegerPlaneEmitterError("lattice proof plane_index must be 0 or 1")
        _require_sha256(self.scorer_sha256, name="scorer_sha256")
        _require_sha256(self.camera_sha256, name="camera_sha256")
        if type(self.denominator) is not int or self.denominator <= 0:
            raise IntegerPlaneEmitterError("lattice proof denominator must be a positive integer")
        expected_values = SCORER_HEIGHT * SCORER_WIDTH * RGB_CHANNELS
        if self.scorer_values != expected_values or self.numerator_equal_values != expected_values:
            raise IntegerPlaneEmitterError("lattice proof must cover every scorer value exactly")
        if self.numerator_exact is not True or self.certified_exact is not True:
            raise IntegerPlaneEmitterError("lattice proof must carry two successful exactness verdicts")


@dataclass(frozen=True)
class LatticeBatchProof:
    scorer_planes: np.ndarray
    camera_planes: np.ndarray
    rows: tuple[LatticePlaneProof, ...]

    def __post_init__(self) -> None:
        scorers = validate_scorer_uint8(self.scorer_planes)
        cameras = _immutable_u8(self.camera_planes, name="camera_planes")
        if cameras.ndim != 5 or cameras.shape[0] != scorers.shape[0] or cameras.shape[1:] != (
            PLANE_COUNT,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            RGB_CHANNELS,
        ):
            raise IntegerPlaneEmitterError("camera_planes must have shape [N,2,874,1164,3]")
        if not isinstance(self.rows, tuple) or len(self.rows) != cameras.shape[0] * PLANE_COUNT:
            raise IntegerPlaneEmitterError("one lattice proof row is required per pair/plane")
        operator = factor2_operator()
        pair_planes = (
            (pair_index, plane_index)
            for pair_index in range(scorers.shape[0])
            for plane_index in range(PLANE_COUNT)
        )
        for row_index, (pair_index, plane_index) in enumerate(pair_planes):
            row = self.rows[row_index]
            if not isinstance(row, LatticePlaneProof):
                raise IntegerPlaneEmitterError("lattice proof rows must be LatticePlaneProof values")
            if (row.pair_index, row.plane_index) != (pair_index, plane_index):
                raise IntegerPlaneEmitterError("lattice proof rows must use canonical pair/plane order")
            scorer = scorers[pair_index, plane_index]
            camera = cameras[pair_index, plane_index]
            verification = verify_factor2_uint8_scorer_plane(operator, camera, scorer)
            expected = (
                _sha256_array(scorer),
                _sha256_array(camera),
                verification.denominator,
                verification.numerator_equal_values,
                verification.scorer_values,
                verification.numerator_exact,
                verification.certified_exact,
            )
            actual = (
                row.scorer_sha256,
                row.camera_sha256,
                row.denominator,
                row.numerator_equal_values,
                row.scorer_values,
                row.numerator_exact,
                row.certified_exact,
            )
            if actual != expected:
                raise IntegerPlaneEmitterError("lattice proof row does not bind the supplied scorer/camera bytes")
        object.__setattr__(self, "scorer_planes", scorers)
        object.__setattr__(self, "camera_planes", cameras)


def realize_all_factor2(scorer_planes: np.ndarray) -> LatticeBatchProof:
    """Independently realize and prove every pair/plane exact integer target."""

    targets = validate_scorer_uint8(scorer_planes, require_distinct_planes=True)
    operator = factor2_operator()
    cameras = np.empty(
        (targets.shape[0], PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS),
        dtype=np.uint8,
    )
    rows: list[LatticePlaneProof] = []
    for pair_index in range(targets.shape[0]):
        for plane_index in range(PLANE_COUNT):
            target = targets[pair_index, plane_index]
            camera = realize_factor2_uint8_scorer_plane(operator, target)
            verification = verify_factor2_uint8_scorer_plane(operator, camera, target)
            if not verification.numerator_exact or not verification.certified_exact:
                raise IntegerPlaneEmitterError(
                    f"factor-2 numerator proof failed at pair={pair_index}, plane={plane_index}"
                )
            cameras[pair_index, plane_index] = camera
            rows.append(
                LatticePlaneProof(
                    pair_index=pair_index,
                    plane_index=plane_index,
                    scorer_sha256=_sha256_array(target),
                    camera_sha256=_sha256_array(camera),
                    denominator=verification.denominator,
                    numerator_equal_values=verification.numerator_equal_values,
                    scorer_values=verification.scorer_values,
                    numerator_exact=verification.numerator_exact,
                    certified_exact=verification.certified_exact,
                )
            )
    return LatticeBatchProof(targets, cameras, tuple(rows))


@dataclass(frozen=True)
class SignFixedU4Basis:
    """Custodied left ``U4`` class/logit basis of the centered five-class head.

    SVD signs are fixed by the first maximum-magnitude coordinate of each right
    singular vector.  The corresponding left columns are the named ``U4``.
    This matches the frozen diagnostic byte custody and keeps feature-space V4
    explicitly separate from class/logit-space U4.
    """

    u4: np.ndarray
    right_vectors: np.ndarray
    singular_values: np.ndarray
    centered_weight: np.ndarray
    pair_difference_map: np.ndarray
    pair_coefficients: np.ndarray
    frozen_head_sha256: str

    def __post_init__(self) -> None:
        u4 = _immutable_f64(self.u4, name="u4")
        right = _immutable_f64(self.right_vectors, name="right_vectors")
        singular = _immutable_f64(self.singular_values, name="singular_values")
        centered = _immutable_f64(self.centered_weight, name="centered_weight")
        pair_map = _immutable_f64(self.pair_difference_map, name="pair_difference_map")
        pair_coefficients = _immutable_f64(self.pair_coefficients, name="pair_coefficients")
        _require_sha256(self.frozen_head_sha256, name="frozen_head_sha256")
        if u4.shape != (5, 4) or right.shape != (4, 144):
            raise IntegerPlaneEmitterError("U4 basis has invalid rank-four geometry")
        if (
            singular.shape != (4,)
            or centered.shape != (5, 144)
            or pair_map.shape != (10, 5)
            or pair_coefficients.shape != (10, 4)
        ):
            raise IntegerPlaneEmitterError("U4 custody arrays have invalid geometry")
        if np.linalg.matrix_rank(centered.astype(np.float64)) != 4:
            raise IntegerPlaneEmitterError("centered frozen head must have exact numerical rank four")
        if not np.allclose(
            singular.astype(np.float64),
            np.asarray(MEASURED_U4_SINGULAR_VALUES),
            rtol=0.0,
            atol=5e-12,
        ):
            raise IntegerPlaneEmitterError("U4 singular values disagree with measured custody")
        for vector in right:
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] <= 0.0:
                raise IntegerPlaneEmitterError("U4 right singular vectors violate sign rule")
        if not np.allclose(u4.T @ u4, np.eye(4), rtol=0.0, atol=5e-14):
            raise IntegerPlaneEmitterError("U4 columns must be orthonormal")
        if not np.allclose(u4.sum(axis=0), 0.0, rtol=0.0, atol=5e-14):
            raise IntegerPlaneEmitterError("U4 columns must lie in the centered-logit subspace")
        reconstructed = (u4 * singular[None, :]) @ right
        if not np.allclose(reconstructed, centered, rtol=0.0, atol=8e-15):
            raise IntegerPlaneEmitterError("U4 basis does not reconstruct the centered head")
        expected_map = np.zeros((10, 5), dtype=np.float64)
        for row, (class_i, class_j) in enumerate(PAIR_INDICES):
            expected_map[row, class_i] = 1.0
            expected_map[row, class_j] = -1.0
        if not np.array_equal(pair_map, expected_map):
            raise IntegerPlaneEmitterError("pair-difference map must contain all ten class pairs")
        if not np.allclose(pair_coefficients, pair_map @ u4, rtol=0.0, atol=5e-15):
            raise IntegerPlaneEmitterError("pair coefficients must be D10x5 @ U4")
        if _raw_f64_sha256(u4) != FROZEN_U4_F64_SHA256:
            raise IntegerPlaneEmitterError("U4 float64 byte custody mismatch")
        if _raw_f64_sha256(pair_coefficients) != FROZEN_PAIR_COEFFICIENTS_F64_SHA256:
            raise IntegerPlaneEmitterError("U4 pair-coefficient float64 byte custody mismatch")
        object.__setattr__(self, "u4", u4)
        object.__setattr__(self, "right_vectors", right)
        object.__setattr__(self, "singular_values", singular)
        object.__setattr__(self, "centered_weight", centered)
        object.__setattr__(self, "pair_difference_map", pair_map)
        object.__setattr__(self, "pair_coefficients", pair_coefficients)

    @classmethod
    def from_head_weight(
        cls,
        weight: np.ndarray,
        *,
        frozen_head_sha256: str,
    ) -> SignFixedU4Basis:
        if _require_sha256(frozen_head_sha256, name="frozen_head_sha256") != (FROZEN_SEGNET_HEAD_FILE_SHA256):
            raise IntegerPlaneEmitterError("frozen SegNet head file SHA custody mismatch")
        raw = np.asarray(weight)
        if raw.dtype != np.float32 or raw.shape != (5, 16, 3, 3):
            raise IntegerPlaneEmitterError("head weight must have exact shape (5,16,3,3) and float32")
        if not np.isfinite(raw).all():
            raise IntegerPlaneEmitterError("head weight must be finite")
        if _raw_f32_sha256(raw) != FROZEN_SEGNET_HEAD_WEIGHT_SHA256:
            raise IntegerPlaneEmitterError("frozen SegNet head tensor byte custody mismatch")
        rows = raw.reshape(5, 144).astype(np.float64)
        centered = rows - rows.mean(axis=0, keepdims=True)
        left, singular, right = np.linalg.svd(centered, full_matrices=False)
        scale = max(float(singular[0]), 1.0)
        rank = int(np.count_nonzero(singular > np.finfo(np.float64).eps * scale * 64.0))
        if rank != 4:
            raise IntegerPlaneEmitterError(f"centered frozen head rank must be four, got {rank}")
        left4 = left[:, :4].copy()
        right4 = right[:4].copy()
        for index in range(4):
            pivot = int(np.argmax(np.abs(right4[index])))
            if right4[index, pivot] < 0.0:
                right4[index] *= -1.0
                left4[:, index] *= -1.0
        pair_map = np.zeros((10, 5), dtype=np.float64)
        for row, (class_i, class_j) in enumerate(PAIR_INDICES):
            pair_map[row, class_i] = 1.0
            pair_map[row, class_j] = -1.0
        return cls(
            u4=left4.astype(np.float64),
            right_vectors=right4.astype(np.float64),
            singular_values=singular[:4].astype(np.float64),
            centered_weight=centered.astype(np.float64),
            pair_difference_map=pair_map,
            pair_coefficients=pair_map @ left4,
            frozen_head_sha256=frozen_head_sha256,
        )

    def raw_four_coordinates(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits)
        if values.dtype != np.float32 or values.shape[-1] != 5 or not np.isfinite(values).all():
            raise IntegerPlaneEmitterError("logits must be finite float32 with last dimension five")
        return np.subtract(values[..., 1:], values[..., :1], dtype=np.float32)

    def u4_coordinates(self, head_features: np.ndarray) -> np.ndarray:
        """Project features into U4 coordinates through V4*Sigma, never /Sigma."""

        features = np.asarray(head_features)
        if features.dtype != np.float32 or features.shape[-1] != 144 or not np.isfinite(features).all():
            raise IntegerPlaneEmitterError("head_features must be finite float32 [...,144]")
        projected = features.astype(np.float64) @ self.right_vectors.T
        return projected * self.singular_values

    def centered_logits(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits)
        if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise IntegerPlaneEmitterError("logits must have dtype float32 or float64")
        if values.shape[-1] != 5 or not np.isfinite(values).all():
            raise IntegerPlaneEmitterError("logits must be finite with last dimension five")
        values64 = values.astype(np.float64, copy=False)
        return values64 - values64.mean(axis=-1, keepdims=True)

    def logit_u4_coordinates(self, logits: np.ndarray) -> np.ndarray:
        """Return ``q = centered_logits @ U4`` in float64 custody."""

        return self.centered_logits(logits) @ self.u4

    def centered_logits_from_u4(self, coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinates)
        if values.dtype != np.float64 or values.shape[-1] != 4 or not np.isfinite(values).all():
            raise IntegerPlaneEmitterError("U4 coordinates must be finite float64 [...,4]")
        return values @ self.u4.T

    def all_pair_margins_from_u4(self, coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinates)
        if values.dtype != np.float64 or values.shape[-1] != 4 or not np.isfinite(values).all():
            raise IntegerPlaneEmitterError("U4 coordinates must be finite float64 [...,4]")
        return values @ self.pair_coefficients.T

    def all_pair_margins(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits)
        if values.dtype != np.float32 or values.shape[-1] != 5 or not np.isfinite(values).all():
            raise IntegerPlaneEmitterError("logits must be finite float32 with last dimension five")
        return values.astype(np.float64) @ self.pair_difference_map.T

    def coordinates(self, head_features: np.ndarray, *, divide_by_sigma: bool = False) -> np.ndarray:
        if divide_by_sigma:
            raise IntegerPlaneEmitterError("sigma normalization is forbidden for U4")
        return self.u4_coordinates(head_features)

    @property
    def u4_sha256(self) -> str:
        return _raw_f64_sha256(self.u4)

    @property
    def pair_coefficients_sha256(self) -> str:
        return _raw_f64_sha256(self.pair_coefficients)


BasisArm = Literal["raw_centered", "sign_fixed_u4_pair_margin"]


@dataclass(frozen=True)
class FixedCapacityBasisAB:
    """Two objective views sharing one immutable emitter capacity exactly."""

    structured: StructuredEmitterState
    residual: QuotientResidualState
    u4: SignFixedU4Basis
    hard_oracle_requested: bool = True

    def __post_init__(self) -> None:
        _validate_state_pair(self.structured, self.residual)
        if not isinstance(self.u4, SignFixedU4Basis):
            raise IntegerPlaneEmitterError("u4 must be SignFixedU4Basis")
        if self.hard_oracle_requested is not True:
            raise IntegerPlaneEmitterError("both C2 basis arms must request the hard oracle")

    @property
    def capacity_signature(self) -> CapacitySignature:
        return CapacitySignature.from_states(self.structured, self.residual)

    def objective_view(self, arm: BasisArm, logits: np.ndarray) -> np.ndarray:
        if arm == "raw_centered":
            return self.u4.raw_four_coordinates(logits)
        if arm == "sign_fixed_u4_pair_margin":
            coordinates = self.u4.logit_u4_coordinates(logits)
            return self.u4.all_pair_margins_from_u4(coordinates)
        raise IntegerPlaneEmitterError(f"unknown basis arm {arm!r}")

    def require_same_emitted_bytes(self, raw_arm: np.ndarray, u4_arm: np.ndarray) -> None:
        raw = validate_scorer_uint8(raw_arm)
        transformed = validate_scorer_uint8(u4_arm)
        if not np.array_equal(raw, transformed):
            raise IntegerPlaneEmitterError("basis-only A/B arms must emit identical build-time bytes")

    def refuse_capacity_change_until_verdict(self, proposed: CapacitySignature) -> None:
        """Keep capacity sealed while this BUILD has no verifier-issued verdict API.

        C2 deliberately does not accept caller-minted booleans, hashes, or receipt
        lookalikes.  A later measurement landing may add a verifier that reads and
        validates governed raw/U4 receipt bytes; until then every mutation refuses.
        """

        if not isinstance(proposed, CapacitySignature):
            raise IntegerPlaneEmitterError("proposed must be CapacitySignature")
        if proposed == self.capacity_signature:
            return
        raise IntegerPlaneEmitterError(
            "capacity mutation is blocked: no verifier-issued measured basis verdict API is landed"
        )


@dataclass(frozen=True)
class EncodeOnlyVJPGuidance:
    """Hash-bound proposal metadata that is forbidden from decoder/admission use."""

    manifest_sha256: str
    vjp_sha256: str
    pair_ids: tuple[int, ...]
    purpose: str = "encode_only_proposal_or_trust_region_guidance"

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_sha256, name="manifest_sha256")
        _require_sha256(self.vjp_sha256, name="vjp_sha256")
        if (
            not isinstance(self.pair_ids, tuple)
            or not self.pair_ids
            or any(type(value) is not int or value < 0 for value in self.pair_ids)
            or len(set(self.pair_ids)) != len(self.pair_ids)
        ):
            raise IntegerPlaneEmitterError("pair_ids must be unique nonnegative integers")
        if self.purpose != "encode_only_proposal_or_trust_region_guidance":
            raise IntegerPlaneEmitterError("VJP guidance purpose is sealed to encode-only")

    def proposal_metadata(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "vjp_sha256": self.vjp_sha256,
            "pair_ids": list(self.pair_ids),
            "purpose": self.purpose,
            "decoder_serializable": False,
            "candidate_admission_authority": False,
        }

    def decoder_payload(self) -> bytes:
        raise IntegerPlaneEmitterError("encode-only VJP guidance has no decoder serialization surface")

    def admit_candidate(self) -> None:
        raise IntegerPlaneEmitterError("encode-only VJP guidance cannot admit a candidate")


def rgb_pair_to_yuv6(rgb_pair_nhwc: np.ndarray) -> np.ndarray:
    """Frozen PoseNet RGB-pair transform to `[N,12,H/2,W/2]` float32.

    Four luma samples and one 2x2-averaged U/V pair are emitted per RGB plane,
    then the two plane channel groups are concatenated exactly as DistortionNet.
    """

    raw = np.asarray(rgb_pair_nhwc)
    if raw.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
        raise IntegerPlaneEmitterError("RGB pair must have dtype uint8 or float32")
    if raw.ndim != 5 or raw.shape[1] != PLANE_COUNT or raw.shape[-1] != RGB_CHANNELS:
        raise IntegerPlaneEmitterError("RGB pair must have shape [N,2,H,W,3]")
    if raw.shape[2] < 2 or raw.shape[3] < 2:
        raise IntegerPlaneEmitterError("RGB pair spatial geometry must be at least 2x2")
    values = raw.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise IntegerPlaneEmitterError("RGB pair must contain finite values")
    height = (values.shape[2] // 2) * 2
    width = (values.shape[3] // 2) * 2
    rgb = values[:, :, :height, :width]
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = np.clip(
        red * np.float32(0.299) + green * np.float32(0.587) + blue * np.float32(0.114),
        np.float32(0.0),
        np.float32(255.0),
    )
    u = np.clip(
        (blue - luma) / np.float32(1.772) + np.float32(128.0),
        np.float32(0.0),
        np.float32(255.0),
    )
    v = np.clip(
        (red - luma) / np.float32(1.402) + np.float32(128.0),
        np.float32(0.0),
        np.float32(255.0),
    )
    u_sub = (u[..., 0::2, 0::2] + u[..., 1::2, 0::2] + u[..., 0::2, 1::2] + u[..., 1::2, 1::2]) * np.float32(0.25)
    v_sub = (v[..., 0::2, 0::2] + v[..., 1::2, 0::2] + v[..., 0::2, 1::2] + v[..., 1::2, 1::2]) * np.float32(0.25)
    yuv6 = np.stack(
        (
            luma[..., 0::2, 0::2],
            luma[..., 1::2, 0::2],
            luma[..., 0::2, 1::2],
            luma[..., 1::2, 1::2],
            u_sub,
            v_sub,
        ),
        axis=2,
    )
    return yuv6.reshape(raw.shape[0], 12, height // 2, width // 2).astype(np.float32, copy=False)


__all__ = [
    "CAMERA_HEIGHT",
    "CAMERA_WIDTH",
    "FROZEN_PAIR_COEFFICIENTS_F64_SHA256",
    "FROZEN_SEGNET_HEAD_FILE_SHA256",
    "FROZEN_SEGNET_HEAD_WEIGHT_SHA256",
    "FROZEN_U4_F64_SHA256",
    "MEASURED_U4_SINGULAR_VALUES",
    "PLANE_COUNT",
    "SCORER_HEIGHT",
    "SCORER_WIDTH",
    "CapacitySignature",
    "EncodeOnlyVJPGuidance",
    "FixedCapacityBasisAB",
    "IntegerPlaneEmitterError",
    "IntegerPlaneGeometry",
    "LatticeBatchProof",
    "LatticePlaneProof",
    "MLXUnavailableError",
    "QuotientResidualState",
    "SignFixedU4Basis",
    "StructuredEmitterState",
    "deterministic_coordinate_basis",
    "factor2_operator",
    "mlx_precursor",
    "mlx_uint8",
    "numpy_precursor",
    "numpy_uint8",
    "realize_all_factor2",
    "rgb_pair_to_yuv6",
    "torch_precursor",
    "torch_uint8",
    "torch_uint8_bytes",
    "validate_scorer_uint8",
]
