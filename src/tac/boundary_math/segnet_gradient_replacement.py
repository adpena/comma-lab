# SPDX-License-Identifier: MIT
"""Input-costate injection for replacing a frozen SegNet backward pass.

This module does *not* approximate SegNet.  It implements the exact chain-rule
seam that a learned student or a refreshed cache must satisfy.  If a renderer
produces ``x(theta)`` and a detached provider supplies an input costate
``lambda_hat`` with the same shape as ``x``, then

    L_inject(theta) = sum(stopgrad(lambda_hat) * x(theta))

has parameter gradient ``J_x(theta).T @ lambda_hat``.  Consequently an exact
``lambda_hat = d L_teacher / d x`` reproduces the frozen teacher's parameter
gradient without retaining the teacher graph.  Approximate providers are
admitted elsewhere only after fail-closed agreement and real-teacher step
checks; forward/logit agreement alone is intentionally absent from this API.

NumPy owns the framework-independent faithfulness metrics.  Torch is imported
inside the Torch helper and MLX is imported inside the MLX helper so importing
this module on a CPU-only/headless host cannot initialize a Metal device.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

_METRIC_EPS = 1.0e-12


@dataclass(frozen=True)
class CostateAgreementMetrics:
    """Measured agreement between a candidate and a real-teacher input costate.

    ``None`` metrics mean that a fail-closed prerequisite was not satisfied
    (shape, finiteness, non-empty mask, or non-zero reference norm).  Callers
    must check :attr:`valid` before comparing thresholds.
    """

    shape_match: bool
    finite: bool
    compared_elements: int
    cosine_similarity: float | None
    relative_l2_error: float | None
    norm_ratio: float | None
    reasons: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return (
            self.shape_match
            and self.finite
            and self.compared_elements > 0
            and self.cosine_similarity is not None
            and self.relative_l2_error is not None
            and self.norm_ratio is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_match": self.shape_match,
            "finite": self.finite,
            "compared_elements": self.compared_elements,
            "cosine_similarity": self.cosine_similarity,
            "relative_l2_error": self.relative_l2_error,
            "norm_ratio": self.norm_ratio,
            "reasons": list(self.reasons),
            "valid": self.valid,
        }


@dataclass(frozen=True)
class TeacherStepCheck:
    """Real-teacher loss check for one candidate step.

    ``reference_loss`` is the teacher loss after a same-step update made with
    the exact teacher costate.  Regret is candidate minus reference, so a
    negative value is allowed and is not clipped away.
    """

    current_loss: float
    candidate_loss: float
    reference_loss: float
    finite: bool
    decreases_teacher_loss: bool
    regret: float | None
    objective_context_fingerprint: str
    anchor_frame_sha256: str
    candidate_frame_sha256: str
    reference_frame_sha256: str
    provider_custody_sha256: str
    evaluated_at_step: int
    candidate_frame: Any
    reference_frame: Any

    def passes(self, *, max_regret: float) -> bool:
        return (
            self.finite
            and self.decreases_teacher_loss
            and self.regret is not None
            and self.regret <= max_regret
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_loss": self.current_loss,
            "candidate_loss": self.candidate_loss,
            "reference_loss": self.reference_loss,
            "finite": self.finite,
            "decreases_teacher_loss": self.decreases_teacher_loss,
            "regret": self.regret,
            "objective_context_fingerprint": self.objective_context_fingerprint,
            "anchor_frame_sha256": self.anchor_frame_sha256,
            "candidate_frame_sha256": self.candidate_frame_sha256,
            "reference_frame_sha256": self.reference_frame_sha256,
            "provider_custody_sha256": self.provider_custody_sha256,
            "evaluated_at_step": self.evaluated_at_step,
        }


def _broadcast_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast a boolean annulus mask without guessing channel semantics.

    Common scorer layouts are supported explicitly: full-shape masks, NHWC
    masks lacking the final channel, NCHW masks lacking channel axis 1, and a
    spatial ``(H, W)`` mask.  Other broadcastable layouts use NumPy's ordinary
    rules; incompatible masks are rejected by the caller.
    """

    candidates: list[np.ndarray] = [mask]
    if mask.ndim + 1 == len(shape):
        candidates.append(np.expand_dims(mask, axis=-1))
        if len(shape) >= 3:
            candidates.append(np.expand_dims(mask, axis=1))
    if mask.ndim == 2 and len(shape) >= 2 and tuple(mask.shape) == tuple(shape[-2:]):
        candidates.append(mask.reshape((1,) * (len(shape) - 2) + mask.shape))

    for candidate in candidates:
        try:
            return np.broadcast_to(candidate, shape).astype(bool, copy=False)
        except ValueError:
            continue
    raise ValueError(f"mask shape {mask.shape} is not broadcastable to costate shape {shape}")


def measure_costate_agreement(
    teacher_costate: Any,
    candidate_costate: Any,
    *,
    mask: Any | None = None,
    eps: float = _METRIC_EPS,
) -> CostateAgreementMetrics:
    """Measure candidate input-costate fidelity with pure NumPy arithmetic.

    Metrics are accumulated in float64 even when training uses float32.  Shape,
    finiteness, empty-mask, and zero-reference-norm failures return an invalid
    metric record rather than laundering undefined values into a threshold.
    """

    ref = np.asarray(teacher_costate)
    cand = np.asarray(candidate_costate)
    if ref.shape != cand.shape:
        return CostateAgreementMetrics(
            shape_match=False,
            finite=bool(np.isfinite(ref).all() and np.isfinite(cand).all()),
            compared_elements=0,
            cosine_similarity=None,
            relative_l2_error=None,
            norm_ratio=None,
            reasons=(f"shape mismatch: teacher={ref.shape}, candidate={cand.shape}",),
        )

    reasons: list[str] = []
    finite = bool(np.isfinite(ref).all() and np.isfinite(cand).all())
    if not finite:
        reasons.append("teacher or candidate costate contains a nonfinite value")
        return CostateAgreementMetrics(
            shape_match=True,
            finite=False,
            compared_elements=int(ref.size),
            cosine_similarity=None,
            relative_l2_error=None,
            norm_ratio=None,
            reasons=tuple(reasons),
        )

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        try:
            selected = _broadcast_mask(mask_arr, tuple(ref.shape))
        except ValueError as exc:
            return CostateAgreementMetrics(
                shape_match=True,
                finite=True,
                compared_elements=0,
                cosine_similarity=None,
                relative_l2_error=None,
                norm_ratio=None,
                reasons=(str(exc),),
            )
        ref = ref[selected]
        cand = cand[selected]

    compared = int(ref.size)
    if compared == 0:
        return CostateAgreementMetrics(
            shape_match=True,
            finite=True,
            compared_elements=0,
            cosine_similarity=None,
            relative_l2_error=None,
            norm_ratio=None,
            reasons=("mask selected zero costate elements",),
        )

    ref64 = np.asarray(ref, dtype=np.float64).reshape(-1)
    cand64 = np.asarray(cand, dtype=np.float64).reshape(-1)
    ref_norm = float(np.linalg.norm(ref64))
    cand_norm = float(np.linalg.norm(cand64))
    if not np.isfinite(ref_norm) or ref_norm <= eps:
        reasons.append("real-teacher costate norm is zero or numerically undefined")
        return CostateAgreementMetrics(
            shape_match=True,
            finite=True,
            compared_elements=compared,
            cosine_similarity=None,
            relative_l2_error=None,
            norm_ratio=None,
            reasons=tuple(reasons),
        )

    cosine = float(np.dot(ref64, cand64) / (ref_norm * max(cand_norm, eps)))
    # Roundoff can produce 1 + a few ulps, which is not a meaningful cosine.
    cosine = float(np.clip(cosine, -1.0, 1.0))
    rel_l2 = float(np.linalg.norm(cand64 - ref64) / ref_norm)
    norm_ratio = float(cand_norm / ref_norm)
    return CostateAgreementMetrics(
        shape_match=True,
        finite=True,
        compared_elements=compared,
        cosine_similarity=cosine,
        relative_l2_error=rel_l2,
        norm_ratio=norm_ratio,
        reasons=(),
    )


def evaluate_teacher_step(
    *,
    current_loss: float,
    candidate_loss: float,
    reference_loss: float,
    objective_context_fingerprint: str,
    anchor_frame: Any,
    candidate_frame: Any,
    reference_frame: Any,
    provider_custody_sha256: str,
    evaluated_at_step: int,
) -> TeacherStepCheck:
    """Build a provenance-bound real-teacher one-step check.

    The scalar losses are inadmissible without hashes tying them to the exact
    objective, anchor, provider candidate, teacher-reference candidate, and
    provider bytes.  This prevents a favorable check from another pair, stage,
    loss, or checkpoint from being replayed as current evidence.
    """

    anchor_frame_sha256 = array_content_sha256(anchor_frame)
    candidate_frame_sha256 = array_content_sha256(candidate_frame)
    reference_frame_sha256 = array_content_sha256(reference_frame)
    frame_arrays = tuple(
        _as_numpy_array(value) for value in (anchor_frame, candidate_frame, reference_frame)
    )
    if any(array.shape != frame_arrays[0].shape for array in frame_arrays[1:]):
        raise ValueError("teacher step anchor/candidate/reference frames must have identical shapes")
    if not all(np.isfinite(array).all() for array in frame_arrays):
        raise ValueError("teacher step anchor/candidate/reference frames must be finite")
    bindings = {
        "objective_context_fingerprint": objective_context_fingerprint,
        "anchor_frame_sha256": anchor_frame_sha256,
        "candidate_frame_sha256": candidate_frame_sha256,
        "reference_frame_sha256": reference_frame_sha256,
        "provider_custody_sha256": provider_custody_sha256,
    }
    for name, value in bindings.items():
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    if (
        not isinstance(evaluated_at_step, int)
        or isinstance(evaluated_at_step, bool)
        or evaluated_at_step < 0
    ):
        raise ValueError("evaluated_at_step must be an integer >= 0")

    current = float(current_loss)
    candidate = float(candidate_loss)
    reference = float(reference_loss)
    finite = bool(np.isfinite([current, candidate, reference]).all())
    return TeacherStepCheck(
        current_loss=current,
        candidate_loss=candidate,
        reference_loss=reference,
        finite=finite,
        decreases_teacher_loss=bool(finite and candidate < current),
        regret=(float(candidate - reference) if finite else None),
        objective_context_fingerprint=objective_context_fingerprint,
        anchor_frame_sha256=anchor_frame_sha256,
        candidate_frame_sha256=candidate_frame_sha256,
        reference_frame_sha256=reference_frame_sha256,
        provider_custody_sha256=provider_custody_sha256,
        evaluated_at_step=evaluated_at_step,
        candidate_frame=candidate_frame,
        reference_frame=reference_frame,
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _as_numpy_array(value: Any) -> np.ndarray:
    array_like = value
    if hasattr(array_like, "detach"):
        array_like = array_like.detach()
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    if hasattr(array_like, "numpy") and not isinstance(array_like, np.ndarray):
        array_like = array_like.numpy()
    return np.asarray(array_like)


def array_content_sha256(value: Any) -> str:
    """Strong content hash over an array's dtype, shape, and C-order bytes.

    The metadata prefix prevents byte-identical buffers with different shapes
    or dtypes from sharing an anchor identity.  Torch-like CPU tensors are
    detached before conversion without importing Torch at module import time.
    """

    array = _as_numpy_array(value)
    if array.dtype.hasobject:
        raise ValueError("object arrays cannot be content-addressed as frame evidence")
    contiguous = np.ascontiguousarray(array)
    metadata = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def relative_frame_displacement(anchor_frame: Any, current_frame: Any) -> float:
    """Return ``||current-anchor||_2 / ||anchor||_2`` or ``inf`` on invalid input."""

    anchor = np.asarray(anchor_frame)
    current = np.asarray(current_frame)
    if anchor.shape != current.shape or not (
        np.isfinite(anchor).all() and np.isfinite(current).all()
    ):
        return float("inf")
    anchor64 = np.asarray(anchor, dtype=np.float64)
    current64 = np.asarray(current, dtype=np.float64)
    denom = float(np.linalg.norm(anchor64.reshape(-1)))
    if denom <= _METRIC_EPS:
        return float("inf")
    return float(np.linalg.norm((current64 - anchor64).reshape(-1)) / denom)


def costate_injection_loss_numpy(frame: Any, costate: Any) -> np.floating[Any]:
    """NumPy value of the canonical injection functional (no autograd implied)."""

    x = np.asarray(frame)
    lam = np.asarray(costate)
    if x.shape != lam.shape:
        raise ValueError(f"frame/costate shape mismatch: {x.shape} != {lam.shape}")
    if not (np.isfinite(x).all() and np.isfinite(lam).all()):
        raise ValueError("frame and costate must be finite")
    return np.sum(x * lam)


def costate_injection_loss_torch(frame: Any, costate: Any) -> Any:
    """Torch injection functional with the provider costate explicitly detached."""

    import torch

    if not isinstance(frame, torch.Tensor) or not isinstance(costate, torch.Tensor):
        raise TypeError("frame and costate must both be torch.Tensor")
    if frame.shape != costate.shape:
        raise ValueError(f"frame/costate shape mismatch: {frame.shape} != {costate.shape}")
    if frame.device != costate.device:
        raise ValueError(f"frame/costate device mismatch: {frame.device} != {costate.device}")
    if not bool(torch.isfinite(frame).all()) or not bool(torch.isfinite(costate).all()):
        raise ValueError("frame and costate must be finite")
    return torch.sum(frame * costate.detach())


def costate_injection_loss_mlx(frame: Any, costate: Any) -> Any:
    """MLX injection functional; importing this module never imports MLX.

    Device execution remains the caller's responsibility.  This function is
    intentionally tiny so a future live-trainer integration can compose it
    without inventing a second loss or changing the provider contract.
    """

    import mlx.core as mx

    if tuple(frame.shape) != tuple(costate.shape):
        raise ValueError(f"frame/costate shape mismatch: {frame.shape} != {costate.shape}")
    return mx.sum(frame * mx.stop_gradient(costate))


__all__ = [
    "CostateAgreementMetrics",
    "TeacherStepCheck",
    "array_content_sha256",
    "costate_injection_loss_mlx",
    "costate_injection_loss_numpy",
    "costate_injection_loss_torch",
    "evaluate_teacher_step",
    "measure_costate_agreement",
    "relative_frame_displacement",
]
