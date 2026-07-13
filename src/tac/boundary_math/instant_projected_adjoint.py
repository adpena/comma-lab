# SPDX-License-Identifier: MIT
"""OSS-faithful INSTANT input-adjoint projection for frozen pointwise Conv2d.

This is an adaptation, not a copy, of the algorithm in the official MIT
INSTANT repository (``github.com/hieu-trannn/INSTANT``), specifically its
``LinearSVDOp`` and 1x1-Conv2d registration path.  The corresponding paper is:

Tuan-Kiet Doan, Trung-Hieu Tran, Enzo Tartaglione, Nikola Simidjievski, and
Van-Tam Nguyen (2026), *INSTANT: Compressing Gradients and Activations for
Resource-Efficient Training*, ICLR 2026, OpenReview ``P2q6Y7UweV``.

INSTANT keeps the ordinary forward exact.  In backward it projects the output
cotangent on its smaller matrix axis.  For ``G in R^(L x C_out)`` and frozen
pointwise weight ``W in R^(C_out x C_in)``:

* channel axis (``C_out <= L``): ``(G Q.T) (Q W)``;
* spatial axis (``L < C_out``): ``Q.T ((Q G) W)``.

Only ungrouped 1x1 convolutions are eligible, matching the official computer-
vision wrapper.  Other convolutions remain exact and fail closed if passed to
the projected primitive.  NumPy float64 is the deterministic reference; Torch
is the measured execution path.  MLX parity covers the projection algebra and
does not claim an optimized Metal convolution kernel.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import tempfile
import threading
import weakref
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tac.witness_dsl.scorer_gradient_policy import ProviderCostateEvaluation

RESEARCH_ONLY = True

ProjectionAxis = Literal["channels", "spatial"]


def _fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _canonicalize_signs(rows: np.ndarray) -> np.ndarray:
    result = np.array(rows, dtype=np.float64, copy=True)
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    return result


def _validate_energy_target(value: float) -> float:
    target = float(value)
    if not np.isfinite(target) or not 0.0 < target <= 1.0:
        raise ValueError("energy_target must be finite and in (0, 1]")
    return target


@dataclass(frozen=True)
class AdaptiveProjectorCalibration:
    """Content-bound projector calibrated from exact output cotangents."""

    axis: ProjectionAxis
    basis: np.ndarray  # (rank, projected_dimension), float64
    singular_values: np.ndarray
    base_rank: int
    rank: int
    energy_target: float
    retained_energy: float
    oversampling: int
    calibration_samples: int
    channels: int
    output_hw: tuple[int, int]
    source_fingerprint: str
    calibration_fingerprint: str

    def __post_init__(self) -> None:
        dimension = self.channels if self.axis == "channels" else self.output_hw[0] * self.output_hw[1]
        if self.basis.shape != (self.rank, dimension):
            raise ValueError("basis shape does not match adaptive projection metadata")
        if self.basis.dtype != np.float64 or self.singular_values.dtype != np.float64:
            raise TypeError("calibration arrays must be NumPy float64")
        if self.base_rank < 1 or self.rank < self.base_rank or self.rank > dimension:
            raise ValueError("invalid calibrated rank")
        # macOS Accelerate may set spurious fp status flags around a valid
        # finite matmul; the explicit finite check remains authoritative.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            gram = self.basis @ self.basis.T
        if not np.isfinite(gram).all() or not np.allclose(gram, np.eye(self.rank), rtol=0.0, atol=1e-10):
            raise ValueError("calibration basis is not finite orthonormal")

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "tac.instant_adaptive_projector.v1",
            "authority": "numpy-float64",
            "axis": self.axis,
            "base_rank": self.base_rank,
            "rank": self.rank,
            "energy_target": self.energy_target,
            "retained_energy": self.retained_energy,
            "oversampling": self.oversampling,
            "calibration_samples": self.calibration_samples,
            "channels": self.channels,
            "output_hw": list(self.output_hw),
            "source_fingerprint": self.source_fingerprint,
            "calibration_fingerprint": self.calibration_fingerprint,
            "eligibility": "frozen ungrouped Conv2d with kernel_size=(1,1)",
            "citation": "Doan et al. (2026), INSTANT, OpenReview:P2q6Y7UweV",
            "oss_reference": "https://github.com/hieu-trannn/INSTANT (MIT)",
        }


def calibrate_adaptive_projector_numpy(
    cotangents: np.ndarray,
    *,
    energy_target: float = 0.95,
    oversampling: int = 5,
) -> AdaptiveProjectorCalibration:
    """Calibrate INSTANT's smaller-axis SVD from NCHW or SNCHW samples.

    ``energy_target=0.95`` and ``oversampling=5`` are the official computer-
    vision example defaults.  The final rank is the smallest energy rank plus
    oversampling, capped at the projected dimension; no guessed rank ladder is
    involved.
    """

    values = np.asarray(cotangents, dtype=np.float64)
    if values.ndim == 4:
        values = values[None]
    if values.ndim != 5:
        raise ValueError("cotangents must have shape NCHW or SNCHW")
    samples, batch, channels, height, width = values.shape
    if batch != 1:
        raise ValueError("isolated INSTANT calibration requires batch N=1")
    if samples < 1 or channels < 1 or height < 1 or width < 1:
        raise ValueError("cotangent geometry must be nonempty")
    if not np.isfinite(values).all():
        raise FloatingPointError("cotangent calibration bank contains nonfinite values")
    target = _validate_energy_target(energy_target)
    if not isinstance(oversampling, int) or isinstance(oversampling, bool) or oversampling < 0:
        raise ValueError("oversampling must be an integer >= 0")

    spatial = height * width
    axis: ProjectionAxis = "channels" if channels <= spatial else "spatial"
    if axis == "channels":
        matrix = values[:, 0].transpose(0, 2, 3, 1).reshape(samples * spatial, channels)
        _left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
        vectors = right_t
        dimension = channels
    else:
        matrix = values[:, 0].transpose(2, 3, 0, 1).reshape(spatial, samples * channels)
        left, singular, _right_t = np.linalg.svd(matrix, full_matrices=False)
        vectors = left.T
        dimension = spatial
    energy = singular.astype(np.float64) ** 2
    total = float(energy.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("cannot calibrate an all-zero or nonfinite cotangent bank")
    cumulative = np.cumsum(energy) / total
    base_rank = int(np.searchsorted(cumulative, target, side="left") + 1)
    rank = min(dimension, base_rank + oversampling)
    basis = _canonicalize_signs(vectors[:rank])
    retained = float(cumulative[min(rank, cumulative.size) - 1])
    source_fingerprint = _fingerprint(values)
    calibration_fingerprint = _fingerprint(basis, singular.astype(np.float64))
    return AdaptiveProjectorCalibration(
        axis=axis,
        basis=basis,
        singular_values=singular.astype(np.float64),
        base_rank=base_rank,
        rank=rank,
        energy_target=target,
        retained_energy=retained,
        oversampling=oversampling,
        calibration_samples=samples,
        channels=channels,
        output_hw=(height, width),
        source_fingerprint=source_fingerprint,
        calibration_fingerprint=calibration_fingerprint,
    )


def _validate_projection_geometry(value: Any, calibration: AdaptiveProjectorCalibration) -> None:
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError("projected cotangent must be NCHW with N=1")
    if tuple(value.shape[1:]) != (calibration.channels, *calibration.output_hw):
        raise ValueError("cotangent geometry differs from calibration custody")


def project_cotangent_numpy(
    cotangent: np.ndarray, calibration: AdaptiveProjectorCalibration
) -> np.ndarray:
    value = np.asarray(cotangent)
    _validate_projection_geometry(value, calibration)
    if not np.isfinite(value).all():
        raise FloatingPointError("cotangent contains nonfinite values")
    channels = calibration.channels
    spatial = calibration.output_hw[0] * calibration.output_hw[1]
    matrix = value.reshape(1, channels, spatial).transpose(0, 2, 1)
    q = calibration.basis.astype(value.dtype, copy=False)
    projected = matrix @ q.T @ q if calibration.axis == "channels" else q.T @ (q @ matrix)
    return projected.transpose(0, 2, 1).reshape(value.shape)


def project_cotangent_torch(cotangent: Any, calibration: AdaptiveProjectorCalibration) -> Any:
    import torch

    _validate_projection_geometry(cotangent, calibration)
    if not bool(torch.isfinite(cotangent).all()):
        raise FloatingPointError("cotangent contains nonfinite values")
    channels = calibration.channels
    spatial = calibration.output_hw[0] * calibration.output_hw[1]
    matrix = cotangent.reshape(1, channels, spatial).permute(0, 2, 1)
    q = torch.as_tensor(calibration.basis, dtype=cotangent.dtype, device=cotangent.device)
    projected = (
        matrix @ q.transpose(0, 1) @ q
        if calibration.axis == "channels"
        else q.transpose(0, 1) @ (q @ matrix)
    )
    return projected.permute(0, 2, 1).reshape_as(cotangent)


def project_cotangent_mlx(cotangent: Any, calibration: AdaptiveProjectorCalibration) -> Any:
    import mlx.core as mx

    _validate_projection_geometry(cotangent, calibration)
    channels = calibration.channels
    spatial = calibration.output_hw[0] * calibration.output_hw[1]
    matrix = cotangent.reshape((1, channels, spatial)).transpose(0, 2, 1)
    q = mx.array(calibration.basis, dtype=cotangent.dtype)
    if calibration.axis == "channels":
        projected = mx.matmul(mx.matmul(matrix, q.T), q)
    else:
        projected = mx.matmul(q.T, mx.matmul(q, matrix))
    return projected.transpose(0, 2, 1).reshape(cotangent.shape)


def pointwise_input_adjoint_numpy(
    output_cotangent: np.ndarray,
    weight: np.ndarray,
    calibration: AdaptiveProjectorCalibration,
) -> np.ndarray:
    """NumPy-float64 reference for projected frozen 1x1 input adjoints."""

    grad = np.asarray(output_cotangent)
    matrix_weight = np.asarray(weight)
    if matrix_weight.ndim == 4:
        if tuple(matrix_weight.shape[2:]) != (1, 1):
            raise ValueError("INSTANT OSS computer-vision path only covers 1x1 Conv2d")
        matrix_weight = matrix_weight[:, :, 0, 0]
    if matrix_weight.ndim != 2 or matrix_weight.shape[0] != calibration.channels:
        raise ValueError("pointwise weight geometry differs from calibration")
    _validate_projection_geometry(grad, calibration)
    spatial = calibration.output_hw[0] * calibration.output_hw[1]
    matrix = grad.reshape(1, calibration.channels, spatial).transpose(0, 2, 1)
    q = calibration.basis.astype(np.result_type(grad, matrix_weight), copy=False)
    if calibration.axis == "channels":
        input_matrix = (matrix @ q.T) @ (q @ matrix_weight)
    else:
        input_matrix = q.T @ ((q @ matrix) @ matrix_weight)
    return input_matrix.transpose(0, 2, 1).reshape(
        1, matrix_weight.shape[1], *calibration.output_hw
    )


@dataclass
class ProjectionProof:
    backward_calls: int = 0
    channel_axis_calls: int = 0
    spatial_axis_calls: int = 0
    dense_conv2d_input_calls: int = 0


def instant_pointwise_conv2d(
    x: Any,
    weight: Any,
    bias: Any,
    calibration: AdaptiveProjectorCalibration,
    *,
    proof: ProjectionProof | None = None,
) -> Any:
    """Exact Torch forward with the OSS adaptive low-rank input adjoint."""

    import torch
    import torch.nn.functional as functional

    if x.ndim != 4 or weight.ndim != 4 or tuple(weight.shape[2:]) != (1, 1):
        raise ValueError("INSTANT OSS computer-vision path requires an ungrouped 1x1 Conv2d")
    if weight.requires_grad or (bias is not None and bias.requires_grad):
        raise ValueError("INSTANT input-adjoint projection requires frozen weight and bias")
    if x.shape[0] != 1 or weight.shape[1] != x.shape[1] or weight.shape[0] != calibration.channels:
        raise ValueError("pointwise convolution geometry differs from calibration")
    exact_output = functional.conv2d(x, weight, bias)
    if tuple(exact_output.shape[-2:]) != calibration.output_hw:
        raise ValueError("pointwise output geometry differs from calibration")

    class Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x_: Any, weight_: Any, exact_: Any) -> Any:
            ctx.save_for_backward(weight_)
            return exact_.clone()

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, None, None]:
            (weight_,) = ctx.saved_tensors
            channels = calibration.channels
            spatial = calibration.output_hw[0] * calibration.output_hw[1]
            matrix = grad_output.reshape(1, channels, spatial).permute(0, 2, 1)
            q = torch.as_tensor(calibration.basis, dtype=grad_output.dtype, device=grad_output.device)
            matrix_weight = weight_[:, :, 0, 0]
            if calibration.axis == "channels":
                input_matrix = (matrix @ q.transpose(0, 1)) @ (q @ matrix_weight)
                if proof is not None:
                    proof.channel_axis_calls += 1
            else:
                input_matrix = q.transpose(0, 1) @ ((q @ matrix) @ matrix_weight)
                if proof is not None:
                    proof.spatial_axis_calls += 1
            if proof is not None:
                proof.backward_calls += 1
            grad_input = input_matrix.permute(0, 2, 1).reshape(
                1, matrix_weight.shape[1], *calibration.output_hw
            )
            return grad_input, None, None

    return Function.apply(x, weight, exact_output.detach())


_PROVIDER_MANIFEST_SCHEMA = "tac.instant_projected_adjoint_provider.v1"


@dataclass(frozen=True)
class InstantMechanismProof:
    """Non-authority proof that the projected mechanism actually executed."""

    derivation_digest: str
    frame_sha256: str
    costate_sha256: str
    objective_context_fingerprint: str
    scorer_state_sha256: str
    provider_manifest_sha256: str
    execution_source_sha256: str
    objective_callable_sha256: str
    eligible_layer_identities: tuple[tuple[str, str], ...]
    calibration_sha256: tuple[tuple[str, str], ...]
    backward_calls: tuple[tuple[str, int], ...]
    evaluated_at_step: int
    exact_forward_equal: bool

    def recompute_derivation_digest(self) -> str:
        payload = {
            "frame_sha256": self.frame_sha256,
            "costate_sha256": self.costate_sha256,
            "objective_context_fingerprint": self.objective_context_fingerprint,
            "scorer_state_sha256": self.scorer_state_sha256,
            "provider_manifest_sha256": self.provider_manifest_sha256,
            "execution_source_sha256": self.execution_source_sha256,
            "objective_callable_sha256": self.objective_callable_sha256,
            "eligible_layer_identities": list(self.eligible_layer_identities),
            "calibration_sha256": list(self.calibration_sha256),
            "backward_calls": list(self.backward_calls),
            "evaluated_at_step": self.evaluated_at_step,
            "exact_forward_equal": self.exact_forward_equal,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class InstantProviderCostateEvaluation(ProviderCostateEvaluation):
    """Provider evaluation constructible only after an internal derivation."""

    derivation_digest: str = ""
    mechanism_proof: InstantMechanismProof | None = None
    execution_capability: Any = field(default=None, repr=False, compare=False)


_CAPABILITY_ISSUER = object()


class _InstantExecutionCapability:
    """Opaque, identity-bound evidence that one provider execution completed."""

    __slots__ = ("__weakref__",)

    def __new__(cls, issuer: object | None = None) -> _InstantExecutionCapability:
        if issuer is not _CAPABILITY_ISSUER:
            raise TypeError("INSTANT execution capabilities are provider-issued only")
        return super().__new__(cls)


@dataclass(frozen=True)
class _InstantCapabilityRecord:
    evaluation_ref: weakref.ReferenceType[InstantProviderCostateEvaluation]
    provider_ref: weakref.ReferenceType[InstantProjectedAdjointProvider]
    proof_ref: weakref.ReferenceType[InstantMechanismProof]
    derivation_digest: str
    frame_sha256: str
    costate_sha256: str
    objective_context_fingerprint: str
    scorer_state_sha256: str
    provider_manifest_sha256: str
    evaluated_at_step: int


_CAPABILITY_LOCK = threading.Lock()
_CAPABILITY_REGISTRY: weakref.WeakKeyDictionary[
    _InstantExecutionCapability, _InstantCapabilityRecord
] = weakref.WeakKeyDictionary()


def verify_instant_provider_evaluation_origin(
    evaluation: InstantProviderCostateEvaluation,
) -> tuple[bool, str]:
    """Verify that ``evaluation`` is the exact live object issued by a provider.

    Public proof fields remain useful audit data, but they cannot establish
    origin by authenticating one another.  The opaque capability is registered
    only after provider execution completes and is bound to the exact returned
    evaluation object, proof object, and content hashes.  Copies, reconstructed
    dataclasses, and in-place costate mutation therefore fail closed.
    """

    if not isinstance(evaluation, InstantProviderCostateEvaluation):
        return False, "INSTANT evaluation has the wrong specialized type"
    capability = evaluation.execution_capability
    if not isinstance(capability, _InstantExecutionCapability):
        return False, "INSTANT provider-issued execution capability is missing"
    with _CAPABILITY_LOCK:
        record = _CAPABILITY_REGISTRY.get(capability)
    if record is None:
        return False, "INSTANT execution capability is not registered"
    if record.evaluation_ref() is not evaluation:
        return False, "INSTANT execution capability does not bind this exact evaluation object"
    if record.provider_ref() is None:
        return False, "INSTANT issuing provider is no longer live"
    proof = evaluation.mechanism_proof
    if proof is None or record.proof_ref() is not proof:
        return False, "INSTANT execution capability proof identity mismatch"
    try:
        costate_sha256 = _array_content_sha256(evaluation.costate)
    except (TypeError, ValueError) as exc:
        return False, f"INSTANT capability costate cannot be hashed: {exc}"
    expected = (
        evaluation.derivation_digest,
        evaluation.frame_sha256,
        costate_sha256,
        evaluation.objective_context_fingerprint,
        proof.scorer_state_sha256,
        evaluation.provider_custody_sha256,
        evaluation.evaluated_at_step,
    )
    registered = (
        record.derivation_digest,
        record.frame_sha256,
        record.costate_sha256,
        record.objective_context_fingerprint,
        record.scorer_state_sha256,
        record.provider_manifest_sha256,
        record.evaluated_at_step,
    )
    if expected != registered:
        return False, "INSTANT provider-issued execution capability content mismatch"
    return True, "INSTANT provider-issued execution capability verified"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _torch_tensor_payload(value: Any) -> bytes:
    array = value.detach().cpu().contiguous().numpy()
    metadata = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(metadata).to_bytes(8, "big") + metadata + np.ascontiguousarray(array).tobytes()


def _torch_module_identity(name: str, module: Any) -> str:
    payload = {
        "name": name,
        "type": f"{type(module).__module__}.{type(module).__qualname__}",
        "kernel_size": list(module.kernel_size),
        "stride": list(module.stride),
        "padding": list(module.padding),
        "dilation": list(module.dilation),
        "groups": module.groups,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    digest.update(_torch_tensor_payload(module.weight))
    if module.bias is not None:
        digest.update(_torch_tensor_payload(module.bias))
    return digest.hexdigest()


def _eligible_convolutions(scorer: Any) -> dict[str, Any]:
    import torch

    return {
        name: module
        for name, module in scorer.named_modules()
        if isinstance(module, torch.nn.Conv2d)
        and tuple(module.kernel_size) == (1, 1)
        and tuple(module.stride) == (1, 1)
        and tuple(module.padding) == (0, 0)
        and tuple(module.dilation) == (1, 1)
        and module.groups == 1
    }


def _scorer_state_sha256(scorer: Any) -> str:
    digest = hashlib.sha256()
    modules = tuple(scorer.named_modules())
    digest.update(
        json.dumps(
            [(name, f"{type(module).__module__}.{type(module).__qualname__}", bool(module.training)) for name, module in modules],
            separators=(",", ":"),
        ).encode()
    )
    for name, value in sorted(scorer.state_dict().items()):
        digest.update(name.encode())
        digest.update(_torch_tensor_payload(value))
    return digest.hexdigest()


def _callable_sha256(function: Callable[..., Any]) -> str:
    code = getattr(function, "__code__", None)
    if code is None:
        raise ValueError("INSTANT objective must be a source-identifiable Python callable")
    digest = hashlib.sha256()
    digest.update(str(getattr(function, "__module__", "")).encode())
    digest.update(str(getattr(function, "__qualname__", "")).encode())
    digest.update(code.co_code)
    digest.update(repr(code.co_consts).encode())
    try:
        digest.update(inspect.getsource(function).encode())
    except (OSError, TypeError):
        pass
    return digest.hexdigest()


def _array_content_sha256(value: Any) -> str:
    return hashlib.sha256(_torch_tensor_payload(value)).hexdigest()


def _require_exact_forward_equal(projected_output: Any, dense_output: Any) -> None:
    import torch

    if not bool(torch.equal(projected_output.detach(), dense_output.detach())):
        raise ValueError("INSTANT projected mechanism changed the exact scorer forward")


def _require_projection_proofs(proofs: Mapping[str, ProjectionProof]) -> None:
    for name, proof in proofs.items():
        if (
            proof.backward_calls < 1
            or proof.dense_conv2d_input_calls != 0
            or proof.channel_axis_calls + proof.spatial_axis_calls != proof.backward_calls
        ):
            raise ValueError(f"INSTANT mechanism proof missing or invalid: {name}")


_PINNED_POINTWISE_PRIMITIVE = instant_pointwise_conv2d
_PINNED_POINTWISE_PRIMITIVE_SHA256 = _callable_sha256(_PINNED_POINTWISE_PRIMITIVE)
_PINNED_EXACT_FORWARD_GUARD = _require_exact_forward_equal
_PINNED_EXACT_FORWARD_GUARD_SHA256 = _callable_sha256(_PINNED_EXACT_FORWARD_GUARD)
_PINNED_PROJECTION_PROOF_GUARD = _require_projection_proofs
_PINNED_PROJECTION_PROOF_GUARD_SHA256 = _callable_sha256(_PINNED_PROJECTION_PROOF_GUARD)
_FORBIDDEN_INSTANCE_EXECUTION_OVERRIDES = frozenset(
    {
        "eligible_layer_identities",
        "evaluate",
        "scorer_state_sha256",
        "_load_verified_bank",
        "_objective_fingerprint",
        "_projected_context",
        "_require_frozen_eval_scorer",
        "_verify_projected_primitive",
    }
)


class InstantProjectedAdjointProvider:
    """Research-only owner of scorer, objective, and hash-verified INSTANT bank.

    ``evaluate`` deliberately has no costate argument.  It re-verifies every
    provider input, performs the exact scorer forward, executes projected
    pointwise backwards, and emits a digest-bound typed evaluation only after
    every eligible layer supplies a mechanism proof.
    """

    research_only = True

    def __init__(
        self,
        *,
        scorer: Any,
        objective: Callable[[Any, Any], Any],
        objective_context: Any,
        expected_objective_context_fingerprint: str,
        provider_manifest_path: str | Path,
        expected_provider_manifest_sha256: str,
        expected_provider_manifest_size_bytes: int,
    ) -> None:
        if type(self) is not InstantProjectedAdjointProvider:
            raise ValueError("INSTANT provider subclasses are not an admitted execution surface")
        self._scorer = scorer
        self._objective = objective
        self._objective_context = objective_context
        self._expected_objective_context_fingerprint = expected_objective_context_fingerprint
        self._manifest_path = Path(provider_manifest_path)
        self._expected_manifest_sha256 = expected_provider_manifest_sha256
        self._expected_manifest_size = expected_provider_manifest_size_bytes
        self._execution_path = Path(__file__).resolve()
        self._execution_source_sha256 = _sha256_file(self._execution_path)
        self._objective_callable_sha256 = _callable_sha256(objective)
        self._verify_projected_primitive()
        self._require_frozen_eval_scorer()
        if self.scorer_state_sha256(scorer) != getattr(objective_context, "scorer_sha256", None):
            raise ValueError("INSTANT scorer state does not match objective-context scorer custody")
        self._load_verified_bank()

    @staticmethod
    def scorer_state_sha256(scorer: Any) -> str:
        return _scorer_state_sha256(scorer)

    @staticmethod
    def eligible_layer_identities(scorer: Any) -> dict[str, str]:
        return {name: _torch_module_identity(name, module) for name, module in _eligible_convolutions(scorer).items()}

    def _require_frozen_eval_scorer(self) -> None:
        if any(module.training for module in self._scorer.modules()):
            raise ValueError("INSTANT scorer must be in eval mode")
        if any(parameter.requires_grad for parameter in self._scorer.parameters()):
            raise ValueError("INSTANT scorer parameters must be frozen")

    @staticmethod
    def _verify_projected_primitive() -> None:
        if instant_pointwise_conv2d is not _PINNED_POINTWISE_PRIMITIVE:
            raise ValueError("INSTANT projected primitive identity changed")
        if _callable_sha256(_PINNED_POINTWISE_PRIMITIVE) != _PINNED_POINTWISE_PRIMITIVE_SHA256:
            raise ValueError("INSTANT projected primitive implementation changed")
        if _require_exact_forward_equal is not _PINNED_EXACT_FORWARD_GUARD:
            raise ValueError("INSTANT exact-forward guard identity changed")
        if _callable_sha256(_PINNED_EXACT_FORWARD_GUARD) != _PINNED_EXACT_FORWARD_GUARD_SHA256:
            raise ValueError("INSTANT exact-forward guard implementation changed")
        if _require_projection_proofs is not _PINNED_PROJECTION_PROOF_GUARD:
            raise ValueError("INSTANT projection-proof guard identity changed")
        if _callable_sha256(_PINNED_PROJECTION_PROOF_GUARD) != _PINNED_PROJECTION_PROOF_GUARD_SHA256:
            raise ValueError("INSTANT projection-proof guard implementation changed")

    def _objective_fingerprint(self) -> str:
        fingerprint = getattr(self._objective_context, "fingerprint", None)
        if not callable(fingerprint):
            raise ValueError("objective_context must expose fingerprint()")
        actual = fingerprint()
        if actual != self._expected_objective_context_fingerprint:
            raise ValueError("INSTANT objective/context fingerprint changed")
        return actual

    def _load_verified_bank(self) -> tuple[dict[str, AdaptiveProjectorCalibration], dict[str, str], str]:
        path = self._manifest_path
        if not path.is_file():
            raise ValueError(f"INSTANT provider manifest is missing: {path}")
        before = _file_identity(path)
        if before[2] != self._expected_manifest_size:
            raise ValueError("INSTANT provider manifest byte count changed")
        manifest_sha = _sha256_file(path)
        if before != _file_identity(path):
            raise ValueError("INSTANT provider manifest changed while hashing")
        if manifest_sha != self._expected_manifest_sha256:
            raise ValueError("INSTANT provider manifest SHA-256 changed")
        try:
            manifest = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"INSTANT provider manifest is unreadable: {exc}") from exc
        if manifest.get("schema") != _PROVIDER_MANIFEST_SCHEMA or not isinstance(manifest.get("layers"), dict):
            raise ValueError("INSTANT provider manifest schema is invalid")
        eligible = _eligible_convolutions(self._scorer)
        if set(manifest["layers"]) != set(eligible) or not eligible:
            raise ValueError("INSTANT calibration bank does not exactly cover eligible layers")
        root = path.parent.resolve()
        bank: dict[str, AdaptiveProjectorCalibration] = {}
        hashes: dict[str, str] = {}
        for name, module in eligible.items():
            record = manifest["layers"][name]
            if not isinstance(record, dict):
                raise ValueError(f"INSTANT calibration record is invalid: {name}")
            identity = _torch_module_identity(name, module)
            if record.get("module_identity_sha256") != identity:
                raise ValueError(f"INSTANT eligible-layer identity mismatch: {name}")
            calibration_path = (root / str(record.get("calibration_path", ""))).resolve()
            if calibration_path.parent != root or not calibration_path.is_file():
                raise ValueError(f"INSTANT calibration path is invalid: {name}")
            calibration_before = _file_identity(calibration_path)
            calibration_sha = _sha256_file(calibration_path)
            calibration = load_calibration(calibration_path)
            if calibration_before != _file_identity(calibration_path):
                raise ValueError(f"INSTANT calibration changed while loading: {name}")
            if calibration_sha != record.get("calibration_sha256"):
                raise ValueError(f"INSTANT calibration SHA-256 mismatch: {name}")
            if calibration.channels != module.out_channels:
                raise ValueError(f"INSTANT calibration channel geometry mismatch: {name}")
            bank[name] = calibration
            hashes[name] = calibration_sha
        return bank, hashes, manifest_sha

    @contextlib.contextmanager
    def _projected_context(
        self,
        bank: Mapping[str, AdaptiveProjectorCalibration],
        proofs: Mapping[str, ProjectionProof],
    ) -> Iterator[None]:
        eligible = _eligible_convolutions(self._scorer)
        originals = {name: module.forward for name, module in eligible.items()}
        try:
            for name, module in eligible.items():
                calibration = bank[name]
                proof = proofs[name]

                def projected(
                    value: Any,
                    *,
                    bound=module,
                    bound_calibration=calibration,
                    bound_proof=proof,
                    bound_primitive=_PINNED_POINTWISE_PRIMITIVE,
                ) -> Any:
                    return bound_primitive(
                        value,
                        bound.weight,
                        bound.bias,
                        bound_calibration,
                        proof=bound_proof,
                    )

                module.forward = projected
            yield
        finally:
            for name, module in eligible.items():
                module.forward = originals[name]

    def evaluate(self, *, frame: Any, evaluated_at_step: int) -> InstantProviderCostateEvaluation:
        import torch

        if type(self) is not InstantProjectedAdjointProvider:
            raise ValueError("INSTANT provider subclasses are not an admitted execution surface")
        execution_overrides = sorted(
            _FORBIDDEN_INSTANCE_EXECUTION_OVERRIDES.intersection(self.__dict__)
        )
        if execution_overrides:
            raise ValueError(
                f"INSTANT provider execution surface has instance overrides: {execution_overrides}"
            )
        if InstantProjectedAdjointProvider._projected_context is not _PINNED_PROJECTED_CONTEXT:
            raise ValueError("INSTANT provider execution context identity changed")
        if _callable_sha256(_PINNED_PROJECTED_CONTEXT) != _PINNED_PROJECTED_CONTEXT_SHA256:
            raise ValueError("INSTANT provider execution context implementation changed")
        if not isinstance(evaluated_at_step, int) or isinstance(evaluated_at_step, bool) or evaluated_at_step < 0:
            raise ValueError("evaluated_at_step must be an integer >= 0")
        if not isinstance(frame, torch.Tensor) or frame.ndim != 4 or frame.shape[0] != 1:
            raise ValueError("INSTANT frame must be a batch-one NCHW torch.Tensor")
        if not bool(torch.isfinite(frame).all()):
            raise ValueError("INSTANT frame must be finite")
        self._verify_projected_primitive()
        self._require_frozen_eval_scorer()
        objective_fingerprint = self._objective_fingerprint()
        if _callable_sha256(self._objective) != self._objective_callable_sha256:
            raise ValueError("INSTANT objective execution identity changed")
        if _sha256_file(self._execution_path) != self._execution_source_sha256:
            raise ValueError("INSTANT execution source identity changed")
        state_before = _scorer_state_sha256(self._scorer)
        if state_before != getattr(self._objective_context, "scorer_sha256", None):
            raise ValueError("INSTANT scorer state changed")
        bank, calibration_hashes, manifest_sha = self._load_verified_bank()
        eligible_identities = self.eligible_layer_identities(self._scorer)

        dense_frame = frame.detach().clone().requires_grad_(True)
        dense_output = self._scorer(dense_frame)
        proofs = {name: ProjectionProof() for name in bank}
        projected_frame = frame.detach().clone().requires_grad_(True)
        with _PINNED_PROJECTED_CONTEXT(self, bank, proofs):
            projected_output = self._scorer(projected_frame)
            _PINNED_EXACT_FORWARD_GUARD(projected_output, dense_output)
            loss = self._objective(projected_output, self._objective_context)
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not bool(torch.isfinite(loss)):
                raise ValueError("INSTANT objective must produce one finite Torch scalar")
            costate = torch.autograd.grad(loss, projected_frame)[0].detach()

        if tuple(costate.shape) != tuple(frame.shape) or not bool(torch.isfinite(costate).all()):
            raise ValueError("INSTANT internally derived costate is nonfinite or frame-shape mismatched")
        _PINNED_PROJECTION_PROOF_GUARD(proofs)
        if _scorer_state_sha256(self._scorer) != state_before:
            raise ValueError("INSTANT scorer state changed during derivation")

        frame_sha = _array_content_sha256(frame)
        costate_sha = _array_content_sha256(costate)
        proof_payload = {
            "frame_sha256": frame_sha,
            "costate_sha256": costate_sha,
            "objective_context_fingerprint": objective_fingerprint,
            "scorer_state_sha256": state_before,
            "provider_manifest_sha256": manifest_sha,
            "execution_source_sha256": self._execution_source_sha256,
            "objective_callable_sha256": self._objective_callable_sha256,
            "eligible_layer_identities": sorted(eligible_identities.items()),
            "calibration_sha256": sorted(calibration_hashes.items()),
            "backward_calls": sorted((name, proof.backward_calls) for name, proof in proofs.items()),
            "evaluated_at_step": evaluated_at_step,
            "exact_forward_equal": True,
        }
        derivation_digest = hashlib.sha256(
            json.dumps(proof_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        mechanism_proof = InstantMechanismProof(
            derivation_digest=derivation_digest,
            frame_sha256=frame_sha,
            costate_sha256=costate_sha,
            objective_context_fingerprint=objective_fingerprint,
            scorer_state_sha256=state_before,
            provider_manifest_sha256=manifest_sha,
            execution_source_sha256=self._execution_source_sha256,
            objective_callable_sha256=self._objective_callable_sha256,
            eligible_layer_identities=tuple(sorted(eligible_identities.items())),
            calibration_sha256=tuple(sorted(calibration_hashes.items())),
            backward_calls=tuple(sorted((name, proof.backward_calls) for name, proof in proofs.items())),
            evaluated_at_step=evaluated_at_step,
            exact_forward_equal=True,
        )
        capability = _InstantExecutionCapability(_CAPABILITY_ISSUER)
        evaluation = InstantProviderCostateEvaluation(
            costate=costate,
            frame_sha256=frame_sha,
            objective_context_fingerprint=objective_fingerprint,
            provider_custody_sha256=manifest_sha,
            evaluated_at_step=evaluated_at_step,
            derivation_digest=derivation_digest,
            mechanism_proof=mechanism_proof,
            execution_capability=capability,
        )
        with _CAPABILITY_LOCK:
            _CAPABILITY_REGISTRY[capability] = _InstantCapabilityRecord(
                evaluation_ref=weakref.ref(evaluation),
                provider_ref=weakref.ref(self),
                proof_ref=weakref.ref(mechanism_proof),
                derivation_digest=derivation_digest,
                frame_sha256=frame_sha,
                costate_sha256=costate_sha,
                objective_context_fingerprint=objective_fingerprint,
                scorer_state_sha256=state_before,
                provider_manifest_sha256=manifest_sha,
                evaluated_at_step=evaluated_at_step,
            )
        return evaluation


_PINNED_PROJECTED_CONTEXT = InstantProjectedAdjointProvider._projected_context
_PINNED_PROJECTED_CONTEXT_SHA256 = _callable_sha256(_PINNED_PROJECTED_CONTEXT)


def save_calibration(path: str | Path, calibration: AdaptiveProjectorCalibration) -> None:
    """Atomically preserve one content-addressed calibration stage."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez(
                handle,
                basis=calibration.basis,
                singular_values=calibration.singular_values,
                metadata=np.asarray(json.dumps(calibration.metadata(), sort_keys=True)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_calibration(path: str | Path) -> AdaptiveProjectorCalibration:
    with np.load(path, allow_pickle=False) as payload:
        basis = np.asarray(payload["basis"], dtype=np.float64)
        singular = np.asarray(payload["singular_values"], dtype=np.float64)
        metadata = json.loads(str(payload["metadata"].item()))
    if metadata.get("schema") != "tac.instant_adaptive_projector.v1":
        raise ValueError("unsupported INSTANT calibration schema")
    fingerprint = _fingerprint(basis, singular)
    if fingerprint != metadata.get("calibration_fingerprint"):
        raise ValueError("INSTANT calibration fingerprint mismatch")
    return AdaptiveProjectorCalibration(
        axis=metadata["axis"],
        basis=basis,
        singular_values=singular,
        base_rank=int(metadata["base_rank"]),
        rank=int(metadata["rank"]),
        energy_target=float(metadata["energy_target"]),
        retained_energy=float(metadata["retained_energy"]),
        oversampling=int(metadata["oversampling"]),
        calibration_samples=int(metadata["calibration_samples"]),
        channels=int(metadata["channels"]),
        output_hw=tuple(metadata["output_hw"]),
        source_fingerprint=str(metadata["source_fingerprint"]),
        calibration_fingerprint=str(metadata["calibration_fingerprint"]),
    )


__all__ = [
    "RESEARCH_ONLY",
    "AdaptiveProjectorCalibration",
    "InstantMechanismProof",
    "InstantProjectedAdjointProvider",
    "InstantProviderCostateEvaluation",
    "ProjectionProof",
    "calibrate_adaptive_projector_numpy",
    "instant_pointwise_conv2d",
    "load_calibration",
    "pointwise_input_adjoint_numpy",
    "project_cotangent_mlx",
    "project_cotangent_numpy",
    "project_cotangent_torch",
    "save_calibration",
    "verify_instant_provider_evaluation_origin",
]
