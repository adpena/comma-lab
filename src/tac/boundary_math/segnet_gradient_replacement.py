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
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_METRIC_EPS = 1.0e-12


@dataclass(frozen=True, init=False)
class TerminalObjectiveProviderMode:
    """Fail-closed descriptor for a non-costate terminal-objective provider.

    SFESS can search a sealed exact-objective table, but it cannot supply the
    frame-shaped input costate required by this module's injection seam. The
    descriptor is intentionally non-configurable: an attempted live-gradient
    use always retains the full frozen teacher.
    """

    mode: str = "sfess_cached_k_subset"
    objective_surface: str = "terminal_exact_through_r"
    produces_costate: bool = False
    research_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False
    live_gradient_fallback: str = "full_teacher"
    cache_failure_action: str = "refuse"
    requires_rederived_objective_context_fingerprint: bool = True
    requires_rederived_frame_or_state_fingerprint: bool = True
    requires_rederived_provider_fingerprint: bool = True
    max_evidence_age_queries: int = 0
    requires_finite_objective: bool = True
    live_admission_requires_real_teacher_regret_gate: bool = True

    def resolve_live_gradient(self) -> str:
        """Return the only legal action for a live-gradient request."""

        return self.live_gradient_fallback


SFESS_CACHED_K_SUBSET_PROVIDER_MODE = TerminalObjectiveProviderMode()

YOPO_FIRST_LAYER_SPLIT_PATH = "encoder.model.blocks[0]"
_YOPO_BANK_SCHEMA = "yopo_first_layer_costate_bank_v1"


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
        return self.finite and self.decreases_teacher_loss and self.regret is not None and self.regret <= max_regret

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
    frame_arrays = tuple(_as_numpy_array(value) for value in (anchor_frame, candidate_frame, reference_frame))
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
    if not isinstance(evaluated_at_step, int) or isinstance(evaluated_at_step, bool) or evaluated_at_step < 0:
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
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


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
    if anchor.shape != current.shape or not (np.isfinite(anchor).all() and np.isfinite(current).all()):
        return float("inf")
    anchor64 = np.asarray(anchor, dtype=np.float64)
    current64 = np.asarray(current, dtype=np.float64)
    denom = float(np.linalg.norm(anchor64.reshape(-1)))
    if denom <= _METRIC_EPS:
        return float("inf")
    return float(np.linalg.norm((current64 - anchor64).reshape(-1)) / denom)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_stat_fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    """Mutation-relevant stat fields; deliberately excludes read-updated atime."""

    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _module_state_sha256(module: Any) -> str:
    """Hash live state bytes, names, dtypes, and shapes without disk custody."""

    try:
        state = module.state_dict()
    except AttributeError as exc:
        raise ValueError("YOPO scorer must expose state_dict for live-state binding") from exc
    digest = hashlib.sha256()
    for name in sorted(state):
        value = _as_numpy_array(state[name])
        if value.dtype.hasobject or not bool(np.isfinite(value).all()):
            raise ValueError(f"YOPO live scorer state {name!r} is nonfinite or unsupported")
        digest.update(name.encode("utf-8"))
        digest.update(array_content_sha256(value).encode("ascii"))
    return digest.hexdigest()


def _require_yopo_frozen_eval_scorer(segnet: Any) -> None:
    """Refuse a YOPO scorer that can learn or mutate normalization state.

    A stored first-layer adjoint is meaningful only for one frozen scorer.  A
    top-level ``eval()`` flag is insufficient because a child BatchNorm module
    can be put back into train mode independently, so inspect every module.
    """

    try:
        modules = tuple(segnet.modules())
        parameters = tuple(segnet.parameters())
    except AttributeError as exc:
        raise ValueError("YOPO scorer must expose modules() and parameters()") from exc
    if not modules:
        raise ValueError("YOPO scorer has no modules to validate")
    if any(bool(module.training) for module in modules):
        raise ValueError("YOPO scorer must be in eval mode for every module")
    if any(bool(parameter.requires_grad) for parameter in parameters):
        raise ValueError("YOPO scorer parameters must all be frozen")


def _require_yopo_state_unchanged(*, before: str, after: str, phase: str) -> None:
    """Fail closed if a supposedly frozen scorer changed during a provider phase."""

    if after != before:
        raise ValueError(f"YOPO scorer state changed during {phase}")


def _require_sha256(name: str, value: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")


def _yopo_topology_payload(segnet: Any) -> dict[str, Any]:
    """Validate the frozen EfficientNet-B2 cut and return its identity payload.

    The cut is after ``blocks[0]``, not after the stem: the upstream encoder
    exposes that tensor as its first feature.  The concrete stage map prevents
    a seemingly compatible timm/SMP upgrade from silently changing the cut.
    """

    try:
        encoder = segnet.encoder.model
        blocks = encoder.blocks
        stage_out_idx = dict(encoder._stage_out_idx)
        feature_info = list(encoder.feature_info.info)
        conv_stem = encoder.conv_stem
        bn1 = encoder.bn1
        block0 = blocks[0]
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("YOPO first-layer cut topology is unavailable") from exc
    if not callable(conv_stem) or not callable(bn1) or not callable(block0):
        raise ValueError("YOPO prefix modules must be callable")
    expected_stage_out_idx = {1: 0, 2: 1, 3: 2, 5: 3, 7: 4}
    if stage_out_idx != expected_stage_out_idx:
        raise ValueError(
            "YOPO first-layer cut topology mismatch: expected canonical "
            f"_stage_out_idx={expected_stage_out_idx}, got {stage_out_idx}"
        )
    expected_modules = ["blocks.0", "blocks.1", "blocks.2", "blocks.4", "blocks.6"]
    modules = [str(row.get("module")) for row in feature_info]
    if modules != expected_modules:
        raise ValueError(f"YOPO first-layer feature topology mismatch: expected {expected_modules}, got {modules}")
    return {
        "split_module_path": YOPO_FIRST_LAYER_SPLIT_PATH,
        "prefix_module_paths": [
            "encoder.model.conv_stem",
            "encoder.model.bn1",
            YOPO_FIRST_LAYER_SPLIT_PATH,
        ],
        "stage_out_idx": expected_stage_out_idx,
        "feature_modules": expected_modules,
        "cut_output_is_single_tensor": True,
        "decoder_drops_only_raw_input": True,
    }


def yopo_first_layer_split_identity(segnet: Any) -> str:
    """Return the content identity of the exact frozen-SegNet YOPO cut."""

    payload = _yopo_topology_payload(segnet)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _yopo_first_layer_prefix_torch(segnet: Any, frame: Any) -> Any:
    """Evaluate exactly conv-stem -> bn1 -> blocks[0], never a bypassed stem."""

    _yopo_topology_payload(segnet)
    encoder = segnet.encoder.model
    return encoder.blocks[0](encoder.bn1(encoder.conv_stem(frame)))


@dataclass(frozen=True)
class YopoFirstLayerBank:
    """Content-bound detached first-layer adjoint from a teacher refresh."""

    p1: np.ndarray
    objective_context_fingerprint: str
    scorer_fingerprint: str
    anchor_frame_sha256: str
    split_identity_sha256: str
    live_segnet_state_sha256: str
    source_step: int

    def metadata(self) -> dict[str, Any]:
        p1 = np.asarray(self.p1)
        if p1.dtype.hasobject or not bool(np.isfinite(p1).all()):
            raise ValueError("YOPO p1 must be a finite non-object array")
        for name, value in (
            ("objective_context_fingerprint", self.objective_context_fingerprint),
            ("scorer_fingerprint", self.scorer_fingerprint),
            ("anchor_frame_sha256", self.anchor_frame_sha256),
            ("split_identity_sha256", self.split_identity_sha256),
            ("live_segnet_state_sha256", self.live_segnet_state_sha256),
        ):
            _require_sha256(name, value)
        if not isinstance(self.source_step, int) or isinstance(self.source_step, bool) or self.source_step < 0:
            raise ValueError("YOPO source_step must be an integer >= 0")
        return {
            "schema": _YOPO_BANK_SCHEMA,
            "objective_context_fingerprint": self.objective_context_fingerprint,
            "scorer_fingerprint": self.scorer_fingerprint,
            "anchor_frame_sha256": self.anchor_frame_sha256,
            "split_module_path": YOPO_FIRST_LAYER_SPLIT_PATH,
            "split_identity_sha256": self.split_identity_sha256,
            "live_segnet_state_sha256": self.live_segnet_state_sha256,
            "source_step": self.source_step,
            "p1_sha256": array_content_sha256(p1),
            "p1_shape": list(p1.shape),
            "p1_dtype": p1.dtype.str,
            "units": "teacher_loss_per_split_activation_unit",
        }


def capture_yopo_first_layer_bank(
    *,
    segnet: Any,
    anchor_frame: Any,
    teacher_loss_fn: Any,
    objective_context_fingerprint: str,
    scorer_fingerprint: str,
    evaluated_at_step: int,
) -> tuple[YopoFirstLayerBank, Any]:
    """Run one full teacher refresh and bank both exact input and first-layer costates."""

    import torch

    for name, value in (
        ("objective_context_fingerprint", objective_context_fingerprint),
        ("scorer_fingerprint", scorer_fingerprint),
    ):
        _require_sha256(name, value)
    if not isinstance(evaluated_at_step, int) or isinstance(evaluated_at_step, bool) or evaluated_at_step < 0:
        raise ValueError("YOPO evaluated_at_step must be an integer >= 0")
    if not isinstance(anchor_frame, torch.Tensor):
        raise TypeError("YOPO anchor_frame must be a torch.Tensor")
    if not bool(torch.isfinite(anchor_frame).all()):
        raise ValueError("YOPO anchor_frame must be finite")
    _require_yopo_frozen_eval_scorer(segnet)
    state_before_teacher = _module_state_sha256(segnet)
    frame = anchor_frame.detach().requires_grad_(True)
    captured: list[Any] = []

    def capture(_module: Any, _args: Any, output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise ValueError("YOPO blocks[0] output must be a single tensor")
        captured.append(output)

    _yopo_topology_payload(segnet)
    handle = segnet.encoder.model.blocks[0].register_forward_hook(capture)
    try:
        teacher_loss = teacher_loss_fn(segnet(frame))
    finally:
        handle.remove()
    if not isinstance(teacher_loss, torch.Tensor) or teacher_loss.ndim != 0:
        raise ValueError("YOPO teacher_loss_fn must return one scalar torch.Tensor")
    if not bool(torch.isfinite(teacher_loss)) or len(captured) != 1:
        raise ValueError("YOPO teacher refresh did not produce one finite cut activation")
    try:
        exact_input_costate, p1 = torch.autograd.grad(teacher_loss, (frame, captured[0]))
    except RuntimeError:
        # BatchNorm-like state mutation can make autograd fail before the
        # ordinary post-refresh comparison.  Reclassify that as custody loss
        # when the frozen scorer state changed, never as a recoverable torch
        # implementation detail.
        _require_yopo_state_unchanged(
            before=state_before_teacher,
            after=_module_state_sha256(segnet),
            phase="teacher forward/backward",
        )
        raise
    _require_yopo_state_unchanged(
        before=state_before_teacher,
        after=_module_state_sha256(segnet),
        phase="teacher forward/backward",
    )
    if not bool(torch.isfinite(exact_input_costate).all()) or not bool(torch.isfinite(p1).all()):
        raise ValueError("YOPO teacher refresh produced a nonfinite costate")
    # This is the non-negotiable refresh canary.  A matching module path is not
    # enough: the cut must actually lie on the teacher's differentiable route.
    state_before_canary = _module_state_sha256(segnet)
    canary_frame = anchor_frame.detach().requires_grad_(True)
    canary_z1 = _yopo_first_layer_prefix_torch(segnet, canary_frame)
    canary_input_costate = torch.autograd.grad(canary_z1, canary_frame, grad_outputs=p1)[0]
    _require_yopo_state_unchanged(
        before=state_before_canary,
        after=_module_state_sha256(segnet),
        phase="refresh prefix canary",
    )
    if not torch.equal(canary_input_costate, exact_input_costate) or (
        array_content_sha256(canary_input_costate) != array_content_sha256(exact_input_costate)
    ):
        raise ValueError(
            "YOPO refresh cut canary failed: J_prefix(anchor)^T p1 does not equal the full-teacher input costate"
        )
    bank = YopoFirstLayerBank(
        p1=p1.detach().cpu().contiguous().numpy(),
        objective_context_fingerprint=objective_context_fingerprint,
        scorer_fingerprint=scorer_fingerprint,
        anchor_frame_sha256=array_content_sha256(frame),
        split_identity_sha256=yopo_first_layer_split_identity(segnet),
        live_segnet_state_sha256=_module_state_sha256(segnet),
        source_step=evaluated_at_step,
    )
    bank.metadata()
    return bank, exact_input_costate.detach()


def write_yopo_first_layer_bank(path: str | Path, bank: YopoFirstLayerBank) -> str:
    """Atomically persist a content-addressed NPZ bank and return its SHA-256."""

    destination = Path(path)
    metadata = bank.metadata()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle,
                p1=np.ascontiguousarray(bank.p1),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return _sha256_file(destination)


def load_yopo_first_layer_bank(
    path: str | Path,
    *,
    expected_bank_sha256: str,
    objective_context_fingerprint: str,
    scorer_fingerprint: str,
    expected_split_identity_sha256: str,
    expected_anchor_frame_sha256: str | None = None,
    expected_source_step: int | None = None,
) -> YopoFirstLayerBank:
    """Re-hash and validate a YOPO bank on every provider evaluation."""

    for name, value in (
        ("expected_bank_sha256", expected_bank_sha256),
        ("objective_context_fingerprint", objective_context_fingerprint),
        ("scorer_fingerprint", scorer_fingerprint),
        ("expected_split_identity_sha256", expected_split_identity_sha256),
    ):
        _require_sha256(name, value)
    if expected_anchor_frame_sha256 is not None:
        _require_sha256("expected_anchor_frame_sha256", expected_anchor_frame_sha256)
    bank_path = Path(path)
    if not bank_path.is_file():
        raise ValueError(f"YOPO bank is missing: {bank_path}")
    before = _file_stat_fingerprint(bank_path)
    actual_bank_sha256 = _sha256_file(bank_path)
    if actual_bank_sha256 != expected_bank_sha256:
        raise ValueError("YOPO bank SHA-256 changed")
    try:
        with np.load(bank_path, allow_pickle=False) as archive:
            if set(archive.files) != {"p1", "metadata_json"}:
                raise ValueError("YOPO bank members are invalid")
            p1 = np.asarray(archive["p1"])
            metadata_raw = archive["metadata_json"]
            metadata = json.loads(str(metadata_raw.item()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"YOPO bank cannot be parsed safely: {exc}") from exc
    after = _file_stat_fingerprint(bank_path)
    if before != after or _sha256_file(bank_path) != actual_bank_sha256:
        raise ValueError("YOPO bank changed while it was verified and parsed")
    required = {
        "schema",
        "objective_context_fingerprint",
        "scorer_fingerprint",
        "anchor_frame_sha256",
        "split_module_path",
        "split_identity_sha256",
        "live_segnet_state_sha256",
        "source_step",
        "p1_sha256",
        "p1_shape",
        "p1_dtype",
        "units",
    }
    if set(metadata) != required or metadata.get("schema") != _YOPO_BANK_SCHEMA:
        raise ValueError("YOPO bank metadata schema is invalid")
    if metadata["split_module_path"] != YOPO_FIRST_LAYER_SPLIT_PATH:
        raise ValueError("YOPO bank split module path changed")
    if metadata["objective_context_fingerprint"] != objective_context_fingerprint:
        raise ValueError("YOPO bank objective/context fingerprint mismatch")
    if metadata["scorer_fingerprint"] != scorer_fingerprint:
        raise ValueError("YOPO bank scorer fingerprint mismatch")
    if expected_anchor_frame_sha256 is not None and metadata["anchor_frame_sha256"] != expected_anchor_frame_sha256:
        raise ValueError("YOPO bank anchor frame fingerprint mismatch")
    if metadata["split_identity_sha256"] != expected_split_identity_sha256:
        raise ValueError("YOPO bank split identity mismatch")
    if expected_source_step is not None and metadata["source_step"] != expected_source_step:
        raise ValueError("YOPO bank source step mismatch")
    bank = YopoFirstLayerBank(
        p1=p1,
        objective_context_fingerprint=str(metadata["objective_context_fingerprint"]),
        scorer_fingerprint=str(metadata["scorer_fingerprint"]),
        anchor_frame_sha256=str(metadata["anchor_frame_sha256"]),
        split_identity_sha256=str(metadata["split_identity_sha256"]),
        live_segnet_state_sha256=str(metadata["live_segnet_state_sha256"]),
        source_step=metadata["source_step"],
    )
    if bank.metadata() != metadata:
        raise ValueError("YOPO bank p1 metadata does not match its bytes")
    return bank


def yopo_first_layer_costate_torch(
    *,
    segnet: Any,
    current_frame: Any,
    bank_path: str | Path,
    expected_bank_sha256: str,
    objective_context_fingerprint: str,
    scorer_fingerprint: str,
    current_step: int,
    expected_split_identity_sha256: str,
    expected_anchor_frame_sha256: str,
    expected_source_step: int,
    max_staleness_steps: int,
) -> tuple[Any, dict[str, Any]]:
    """Compute ``J_prefix(current_x)^T p1`` with a freshly verified bank."""

    import torch

    if not isinstance(current_step, int) or isinstance(current_step, bool) or current_step < 0:
        raise ValueError("YOPO current_step must be an integer >= 0")
    if not isinstance(expected_source_step, int) or isinstance(expected_source_step, bool) or expected_source_step < 0:
        raise ValueError("YOPO expected_source_step must be an integer >= 0")
    if not isinstance(max_staleness_steps, int) or isinstance(max_staleness_steps, bool) or max_staleness_steps < 0:
        raise ValueError("YOPO max_staleness_steps must be an integer >= 0")
    if current_step < expected_source_step:
        raise ValueError("YOPO bank source step is in the future")
    if current_step - expected_source_step > max_staleness_steps:
        raise ValueError("YOPO bank is stale for the declared max_staleness_steps")
    if not isinstance(current_frame, torch.Tensor):
        raise TypeError("YOPO current_frame must be a torch.Tensor")
    if not bool(torch.isfinite(current_frame).all()):
        raise ValueError("YOPO current_frame must be finite")
    _require_yopo_frozen_eval_scorer(segnet)
    if current_step == expected_source_step and array_content_sha256(current_frame) != expected_anchor_frame_sha256:
        raise ValueError("YOPO same-step current frame must match the bank anchor frame")
    current_split_identity = yopo_first_layer_split_identity(segnet)
    if current_split_identity != expected_split_identity_sha256:
        raise ValueError("YOPO current scorer split identity mismatch")
    bank = load_yopo_first_layer_bank(
        bank_path,
        expected_bank_sha256=expected_bank_sha256,
        objective_context_fingerprint=objective_context_fingerprint,
        scorer_fingerprint=scorer_fingerprint,
        expected_split_identity_sha256=expected_split_identity_sha256,
        expected_anchor_frame_sha256=expected_anchor_frame_sha256,
        expected_source_step=expected_source_step,
    )
    state_before_prefix_vjp = _module_state_sha256(segnet)
    if state_before_prefix_vjp != bank.live_segnet_state_sha256:
        raise ValueError("YOPO live SegNet state changed since the bank refresh")
    frame = current_frame.detach().requires_grad_(True)
    z1 = _yopo_first_layer_prefix_torch(segnet, frame)
    p1 = torch.as_tensor(bank.p1, device=z1.device, dtype=z1.dtype)
    if tuple(p1.shape) != tuple(z1.shape):
        raise ValueError("YOPO bank p1 shape does not match current split activation")
    if not bool(torch.isfinite(p1).all()):
        raise ValueError("YOPO bank p1 is nonfinite")
    try:
        costate = torch.autograd.grad(z1, frame, grad_outputs=p1)[0].detach()
    except RuntimeError:
        _require_yopo_state_unchanged(
            before=state_before_prefix_vjp,
            after=_module_state_sha256(segnet),
            phase="current prefix VJP",
        )
        raise
    _require_yopo_state_unchanged(
        before=state_before_prefix_vjp,
        after=_module_state_sha256(segnet),
        phase="current prefix VJP",
    )
    if tuple(costate.shape) != tuple(frame.shape) or not bool(torch.isfinite(costate).all()):
        raise ValueError("YOPO first-layer VJP is nonfinite or frame-shape mismatched")
    metadata = bank.metadata()
    metadata.update(
        {
            "bank_sha256": expected_bank_sha256,
            "current_frame_sha256": array_content_sha256(frame),
            "current_step": current_step,
            "pixel_costate_sha256": array_content_sha256(costate),
            "pixel_costate_units": "teacher_loss_per_rendered_frame_unit",
        }
    )
    return costate, metadata


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
    "SFESS_CACHED_K_SUBSET_PROVIDER_MODE",
    "YOPO_FIRST_LAYER_SPLIT_PATH",
    "CostateAgreementMetrics",
    "TeacherStepCheck",
    "TerminalObjectiveProviderMode",
    "YopoFirstLayerBank",
    "array_content_sha256",
    "capture_yopo_first_layer_bank",
    "costate_injection_loss_mlx",
    "costate_injection_loss_numpy",
    "costate_injection_loss_torch",
    "evaluate_teacher_step",
    "load_yopo_first_layer_bank",
    "measure_costate_agreement",
    "relative_frame_displacement",
    "write_yopo_first_layer_bank",
    "yopo_first_layer_costate_torch",
    "yopo_first_layer_split_identity",
]
