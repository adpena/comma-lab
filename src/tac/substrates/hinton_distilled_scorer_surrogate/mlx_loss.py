# SPDX-License-Identifier: MIT
"""Canonical Hinton-distilled KL T=2.0 ``custom_loss_fn`` for MLX long training.

This module provides the canonical implementation of the Hinton-Vinyals-Dean
2014 "Distilling the Knowledge in a Neural Network" distillation paradigm
for the substrate-class-shift Top-1 candidate per the DQS1-ASYMPTOTIC-FLOOR
ranking (`.omx/research/dqs1_asymptotic_floor_substrate_class_shift_prioritization_20260525.md`)
and AAA T4 §6.5 + §12.4 Tier 2A spec.

Canonical math (Hinton-Vinyals-Dean 2014 eq. 1 + 2):

  soft_labels      = softmax(teacher_logits / T)
  student_softmax  = softmax(student_logits / T)
  L_distill        = T**2 * KL(student_softmax || soft_labels)

The ``T**2`` factor (Hinton 2014 §2.1) preserves the gradient magnitude as
temperature varies because the soft-label cross-entropy gradients scale as
``1/T**2``; multiplying the loss by ``T**2`` makes the distillation loss
gradient comparable in magnitude to a hard-label cross-entropy gradient
that uses the same hyperparameters.

The default temperature ``T = 2.0`` follows:
  * The Quantizr canonical 0.33 [contest-CUDA] anchor (CLAUDE.md "Quantizr
    intelligence — verified competitive data" — `kl_on_logits()` with T=2.0
    for SegNet during specific training phases).
  * Probe 6 W=2 + Probe 7 W=6 CCC anchors (commits 685fe6726 +
    `combined_tier_1_ccc_ext_probes_uniward_per_class_plus_hinton_kl_temporal_context_landed_20260525.md`).
  * AAA T4 §6.5 + §12.4 Tier 2A cascade table (commit a951a11f9).

The teacher logits surface is captured by a structured
``TeacherLogitsProvider`` protocol so this module remains independent of any
specific upstream SegNet / PoseNet MLX adapter. The canonical MLX SegNet
adapter (``tac.local_acceleration.mlx_scorer_adapters``) is the production
teacher; ``MockTeacherLogitsProvider`` is provided here as a deterministic,
mathematically well-formed surrogate so the first MLX long-training
validation can land at $0 without requiring upstream SegNet weights on disk.

Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable +
Catalog #192 + Catalog #1: every value emitted from this module is part of
``[macOS-MLX research-signal]`` axis evidence and carries no contest-score
authority. ``score_claim``, ``promotion_eligible``,
``rank_or_kill_eligible``, and ``ready_for_exact_eval_dispatch`` defaults
remain ``False`` for any artifact that consumes this loss; the canonical
MLX false-authority labels in ``tac.local_acceleration.pr95_hnerv_mlx_long_training``
(``PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY``) apply unchanged.

Canonical references:
  * Hinton, Vinyals, Dean 2014: "Distilling the Knowledge in a Neural
    Network" (arXiv:1503.02531).
  * CLAUDE.md "Quantizr intelligence — verified competitive data
    (2026-04-21)": KL T=2.0 SegNet distill empirical anchor.
  * AAA T4 §6.5 + §12.4 Tier 2A (commit a951a11f9): distortion-axis
    substrate-class-shift cascade.
  * DQS1-ASYMPTOTIC-FLOOR Top-1 ranking (commit acf1661ca + sister
    `.omx/research/dqs1_asymptotic_floor_substrate_class_shift_prioritization_20260525.md`).
  * Slot 1 canonical infrastructure: ``tac.local_acceleration.pr95_hnerv_mlx_long_training``.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable
from typing import Any, Protocol

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

# MLX is required for this canonical loss but is imported lazily so the
# package remains importable on machines without an MLX runtime (the test
# fixtures + CLI then fail-closed at the first MLX-bound call with a clear
# RuntimeError instead of an ImportError at module-load time).
try:  # pragma: no cover — guard exercised only on non-MLX hosts.
    import mlx.core as mx  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]

# Canonical defaults per Hinton 2014 + Quantizr empirical anchor.
DEFAULT_DISTILLATION_TEMPERATURE: float = 2.0
DEFAULT_SEGNET_CLASSES: int = 5  # canonical SegNet output classes per
# `upstream/modules.py` (smp.Unet(classes=5)).
# Tiny numerical floor for log + division stability in KL across MLX float32.
_NUMERIC_FLOOR: float = 1.0e-12


def _require_mlx() -> None:
    if mx is None:  # pragma: no cover — non-MLX hosts (Linux x86_64 CI etc.).
        raise RuntimeError(
            "hinton_distilled_scorer_surrogate.mlx_loss requires MLX; install "
            "`mlx` (Apple Silicon only) or invoke from a macOS-ARM64 host. The "
            "canonical fallback is the sister cathedral-consumer surface which "
            "consumes the [macOS-MLX research-signal] axis tag downstream."
        )


def softmax_with_temperature(logits: Any, temperature: float) -> Any:
    """Compute ``softmax(logits / T)`` numerically stable across MLX float32.

    Args:
        logits: MLX array of shape ``(..., num_classes)``.
        temperature: Temperature scalar ``T > 0`` (Hinton 2014).

    Returns:
        MLX array of same shape as ``logits`` whose last axis sums to 1.
    """
    _require_mlx()
    if temperature <= 0.0:
        raise ValueError(
            f"temperature must be > 0 (Hinton 2014 canonical T=2.0); got {temperature}"
        )
    scaled = logits / temperature
    # Numerical stability: subtract per-row max before exp (canonical
    # max-trick for log-sum-exp).
    row_max = mx.max(scaled, axis=-1, keepdims=True)
    exp = mx.exp(scaled - row_max)
    denom = mx.sum(exp, axis=-1, keepdims=True)
    return exp / (denom + _NUMERIC_FLOOR)


def kl_divergence_between_softmax(student_probs: Any, teacher_probs: Any) -> Any:
    """Compute mean ``KL(student || teacher)`` across leading dims.

    Both ``student_probs`` and ``teacher_probs`` must be the output of
    :func:`softmax_with_temperature` (or any other proper softmax) — i.e.
    last-axis probability vectors summing to 1.

    Returns a scalar MLX array (mean across all leading dims) per the
    canonical PyTorch ``F.kl_div(..., reduction='batchmean')`` semantics.
    """
    _require_mlx()
    # KL(P || Q) = sum_i P_i * (log P_i - log Q_i)
    # Numerically stable: clamp both to floor before log.
    p = mx.maximum(student_probs, _NUMERIC_FLOOR)
    q = mx.maximum(teacher_probs, _NUMERIC_FLOOR)
    per_element = student_probs * (mx.log(p) - mx.log(q))
    per_row_kl = mx.sum(per_element, axis=-1)
    return mx.mean(per_row_kl)


def hinton_distilled_kl_t2_loss(
    student_logits: Any,
    teacher_logits: Any,
    temperature: float = DEFAULT_DISTILLATION_TEMPERATURE,
) -> Any:
    """Canonical Hinton-distilled KL loss with the T**2 gradient-magnitude factor.

    Implements Hinton-Vinyals-Dean 2014 eq. 1 + 2 + §2.1::

        soft_labels     = softmax(teacher_logits / T)
        student_softmax = softmax(student_logits / T)
        L               = T**2 * KL(student_softmax || soft_labels)

    Args:
        student_logits: MLX array of shape ``(..., num_classes)``. The smaller-
            capacity surrogate's pre-softmax outputs.
        teacher_logits: MLX array of same shape as ``student_logits``. The
            canonical contest SegNet (or sister scorer) pre-softmax logits.
            By construction this is computed with ``mx.stop_gradient`` if the
            teacher is not being trained; callers that want pure teacher
            distillation should pass ``mx.stop_gradient(teacher_logits)``.
        temperature: Distillation temperature ``T``. Default ``2.0`` per
            Hinton 2014 + Quantizr canonical anchor.

    Returns:
        Scalar MLX array (loss value). Combined with ``mx.value_and_grad``
        in the canonical training step at
        ``tac.local_acceleration.pr95_hnerv_mlx_long_training.MLXLongTrainingPipeline.training_step``.
    """
    _require_mlx()
    if temperature <= 0.0:
        raise ValueError(
            f"temperature must be > 0 (Hinton 2014 canonical T=2.0); got {temperature}"
        )
    teacher_soft = softmax_with_temperature(teacher_logits, temperature)
    student_soft = softmax_with_temperature(student_logits, temperature)
    kl = kl_divergence_between_softmax(student_soft, teacher_soft)
    return kl * (temperature * temperature)


# ---------------------------------------------------------------------------
# Teacher logits provider protocol + canonical mock for $0 macOS-MLX smoke
# ---------------------------------------------------------------------------


class TeacherLogitsProvider(Protocol):
    """Structural type for any teacher-logits source.

    Implementations include:
      * Production: ``tac.local_acceleration.mlx_scorer_adapters.MLXSegNetAdapter``
        wrapped to return pre-softmax logits (read-only; never trained).
      * Mock: :class:`MockTeacherLogitsProvider` for $0 smoke validation
        when upstream SegNet weights are not staged on disk.
      * Sister cathedral-consumer adapters (per Catalog #335 contract).

    The contract: given an RGB frame batch ``frames_bhwc`` (shape
    ``(B, H, W, 3)`` in ``[0, 1]`` per the Slot 1 canonical normalization)
    and a target ``num_classes``, return a teacher logits MLX array of
    shape ``(B, H_logits, W_logits, num_classes)``. The H_logits / W_logits
    need NOT match the input H / W; they only need to match the student's
    logits shape so the KL term is well-defined.
    """

    num_classes: int

    def teacher_logits(self, frames_bhwc: Any) -> Any:
        """Return teacher logits MLX array (shape sister of student logits)."""


@dataclasses.dataclass(frozen=True)
class MockTeacherLogitsProvider:
    """Deterministic, mathematically well-formed teacher logits.

    This canonical mock generates teacher logits as a fixed nonlinear
    transform of the input frame statistics (per-batch deterministic +
    per-class differentiated). It is NOT the production teacher; it exists
    so the $0 macOS-MLX long-training smoke can land WITHOUT requiring
    upstream SegNet weights staged on disk.

    Why this is structurally appropriate as a foundation surface:
      * The KL T=2.0 math is identical regardless of whether the teacher is
        the contest SegNet or a deterministic surrogate; the foundation
        smoke proves the training loop + canonical Provenance routing +
        convergence verdict emission work end-to-end.
      * Sister Slot 3 dispatch packaging (canonical paid-dispatch surface)
        consumes the contest SegNet teacher; the MOCK is replaced 1-line
        by passing the production provider through the same
        ``make_hinton_custom_loss_fn`` factory.
      * Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog
        #192: this mock is part of the ``[macOS-MLX research-signal]`` axis
        evidence and produces NO contest-score authority — only loss-curve
        convergence telemetry that can feed the next local queue-owned proof.

    The mock is sensitivity-bearing on the student decoder output: the
    teacher logits depend on the decoded frame (through ``frames_bhwc``),
    so as the student decoder learns to match its targets, the teacher
    logits drift in a consistent direction, and the KL is non-trivial.

    Determinism: the mock is a pure tensor function with no internal RNG;
    the same frames produce the same logits across runs (canonical
    determinism per CLAUDE.md "Beauty, simplicity, and developer
    experience" + Catalog #305 observability "diff-able across runs").
    """

    num_classes: int = DEFAULT_SEGNET_CLASSES
    seed_offset: float = 0.07
    spatial_downsample_factor: int = 4

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2 for KL distillation to be defined; "
                f"got {self.num_classes}"
            )
        if self.spatial_downsample_factor < 1:
            raise ValueError(
                f"spatial_downsample_factor must be >= 1; got {self.spatial_downsample_factor}"
            )

    def teacher_logits(self, frames_bhwc: Any) -> Any:
        """Compute teacher logits from RGB frames.

        Args:
            frames_bhwc: MLX float32 array of shape ``(B, H, W, 3)`` in
                ``[0, 1]``.

        Returns:
            MLX float32 array of shape ``(B, H', W', num_classes)`` where
            ``H' = H // spatial_downsample_factor`` and similarly for W'.
            The teacher is a pure deterministic function of the input
            frame statistics; produces nontrivial gradients to drive
            convergence of the KL distillation loss.
        """
        _require_mlx()
        # Average-pool the input to (B, H', W', 3) — canonical
        # spatial downsample matching SegNet's stride-2 stem aggregation.
        if self.spatial_downsample_factor == 1:
            pooled = frames_bhwc
        else:
            # MLX has no direct avg_pool over NHWC at the time of writing
            # this canonical surrogate; emulate via reshape + mean (any
            # B, H, W with H % factor == 0 and W % factor == 0). Slot 1
            # canonical pipeline normalizes targets to (H, W) divisible by
            # the canonical eval size so this constraint holds in
            # production smoke.
            b, h, w, c = frames_bhwc.shape
            f = self.spatial_downsample_factor
            if h % f != 0 or w % f != 0:
                raise ValueError(
                    f"spatial dims {(h, w)} must be divisible by "
                    f"spatial_downsample_factor={f}"
                )
            h2 = h // f
            w2 = w // f
            # (B, h2, f, w2, f, C)
            reshaped = mx.reshape(frames_bhwc, (b, h2, f, w2, f, c))
            pooled = mx.mean(reshaped, axis=(2, 4))
        # Per-class deterministic projection: each class has a unique
        # nonlinear transform of (R, G, B) so per-class logits differ.
        # The seed_offset shifts contributions so adjacent classes are
        # not collinear.
        # logits[b, h, w, k] = cos((k * seed_offset + R + 0.5*G + 0.25*B) * pi)
        red = pooled[..., 0:1]
        green = pooled[..., 1:2]
        blue = pooled[..., 2:3]
        # Build per-class scaling tensor of shape (num_classes,) so the
        # final logits have shape (B, H', W', num_classes).
        # Each class uses a unique offset and weighting.
        class_offsets = mx.array(
            [float(k) * self.seed_offset for k in range(self.num_classes)],
            dtype=mx.float32,
        )
        # Compute per-class logits by broadcasting.
        # Shape (B, H', W', 1) × ()-broadcast offsets → (B, H', W', K)
        base = red + 0.5 * green + 0.25 * blue  # (B, H', W', 1)
        scaled = base + class_offsets  # broadcast (1, ..., K)
        # Bound to a sensible range for stable softmax via cos.
        logits = mx.cos(scaled * 3.14159265)
        return logits


# ---------------------------------------------------------------------------
# Canonical real-SegNet teacher cache for the HINTON-MLX-BUNDLE-2026-05-25
# REAL-TEACHER REFIRE per the MVP-first phasing non-negotiable.
#
# This cache is the canonical falsification surface for the mock-vs-real
# teacher cargo-cult: the existing :class:`MockTeacherLogitsProvider`
# produces deterministic cosine logits, while the real upstream PyTorch
# SegNet (loaded via :func:`tac.scorer.load_default_scorers`) produces the
# actual contest-scoring logits. The cache holds pre-computed real SegNet
# logits for every video frame so the per-batch ``teacher_logits_for_indices``
# lookup is O(1) MLX indexing instead of N PyTorch forwards per epoch.
#
# Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable +
# Catalog #192: the cache holds [macOS-MLX research-signal] axis data and
# carries NO contest-score authority on its own. The convergence verdict
# from the smoke is research signal only; promotion still requires paired
# CPU+CUDA contest-hardware verify per Catalog #205.
#
# The canonical scorer-response dataset pattern at
# :mod:`tac.optimization.scorer_response_dataset` is the long-term
# canonicalization surface for this kind of pre-computed teacher logits
# corpus. The cache here is the per-session in-memory equivalent so the
# first MLX long-training validation lands at $0 without requiring the
# operator to first stage an HF dataset.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RealSegNetTeacherLogitsCache:
    """Pre-computed real upstream-PyTorch SegNet logits indexed by frame index.

    Wraps an MLX float32 array of shape ``(T, H, W, K)`` where:
      * ``T`` = total frame count (matches the Slot 1 pair iterator's
        ``frame_count``);
      * ``H``, ``W`` = canonical SegNet input size ``(384, 512)`` per
        :mod:`upstream.modules` SegNet.preprocess_input;
      * ``K`` = canonical SegNet output classes (5 per the canonical
        ``smp.Unet(classes=5)`` config).

    The cache is built ONCE pre-training by :func:`build_real_segnet_teacher_cache`
    (one PyTorch SegNet forward per video frame, no gradient, CPU) and then
    indexed by batch ``indices`` during training. This makes the per-batch
    teacher logits lookup pure MLX (no PyTorch round-trip per step), which
    keeps the MLX training loop fast even with the real-SegNet teacher
    active.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": this
    cache and any artifact derived from it MUST carry the canonical MLX
    false-authority labels (``score_claim=False``, ``promotion_eligible=False``,
    ``ready_for_exact_eval_dispatch=False``, ``rank_or_kill_eligible=False``).
    """

    teacher_logits_thwk: Any  # MLX float32 (T, H, W, K)
    frame_count: int
    height: int
    width: int
    num_classes: int
    upstream_segnet_safetensors_sha256: str | None = None
    cache_build_seconds: float | None = None

    def __post_init__(self) -> None:
        _require_mlx()
        if self.frame_count < 1:
            raise ValueError(f"frame_count must be >= 1; got {self.frame_count}")
        if self.height < 1 or self.width < 1:
            raise ValueError(
                f"height and width must be positive; got ({self.height}, {self.width})"
            )
        if self.num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2 for KL distillation; got {self.num_classes}"
            )

    def teacher_logits_for_indices(self, indices: Any) -> Any:
        """Look up real SegNet teacher logits via MLX integer indexing.

        Args:
            indices: MLX int32 array of shape ``(B,)``.

        Returns:
            MLX float32 array of shape ``(B, H, W, K)``.
        """
        _require_mlx()
        return self.teacher_logits_thwk[indices]


def build_real_segnet_teacher_cache(
    frames_thwc_uint8: Any,  # numpy (T, H, W, 3) uint8
    *,
    upstream_dir: Any = "upstream",
    device: str = "cpu",
    cache_dtype: Any = None,  # MLX dtype; defaults to float32 inside
    segnet_safetensors_sha256: str | None = None,
) -> RealSegNetTeacherLogitsCache:
    """Compute real upstream-PyTorch SegNet logits for every video frame.

    Loads the canonical contest SegNet via :func:`tac.scorer.load_default_scorers`,
    constructs a fake 2-frame stack per frame (SegNet's ``preprocess_input``
    only uses the LAST frame so the first slot is ignored), runs ONE
    PyTorch SegNet forward per frame (no gradient, eval mode, CPU by
    default to avoid CUDA staging cost on macOS), then returns the
    canonical :class:`RealSegNetTeacherLogitsCache`.

    Args:
        frames_thwc_uint8: numpy array of shape ``(T, H, W, 3)`` uint8 RGB
            frames (matches the Slot 1 ``MLXPairIterator._frames_np``
            buffer layout).
        upstream_dir: Path to the upstream repo containing
            ``models/segnet.safetensors``. Default ``"upstream"``.
        device: PyTorch device for the SegNet forward. Default ``"cpu"``
            (no MPS — MPS forwards through SegNet produce 2× distortion drift
            per CLAUDE.md "MPS auth eval is NOISE"; CPU is the canonical
            authoritative substrate for the teacher cache).
        cache_dtype: MLX dtype for the cache. Default ``mx.float32``.
        segnet_safetensors_sha256: Optional sha256 of the SegNet safetensors
            file for canonical Catalog #229 PV traceability.

    Returns:
        :class:`RealSegNetTeacherLogitsCache` carrying the canonical
        ``(T, H, W, K)`` MLX teacher logits array.
    """
    _require_mlx()
    import time as _time

    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]

    from tac.scorer import load_default_scorers

    if frames_thwc_uint8.ndim != 4 or frames_thwc_uint8.shape[-1] != 3:
        raise ValueError(
            f"frames_thwc_uint8 must be (T, H, W, 3); got shape "
            f"{frames_thwc_uint8.shape!r}"
        )
    if frames_thwc_uint8.dtype != np.uint8:
        raise ValueError(
            f"frames_thwc_uint8 must be uint8; got {frames_thwc_uint8.dtype!r}"
        )

    t0 = _time.time()
    _, segnet = load_default_scorers(upstream_dir, device=device)
    segnet.eval()
    # Build (T, 2, 3, H, W) float32 tensor in the same 0..255 RGB scale used
    # by upstream DistortionNet.preprocess_input. SegNet has no internal
    # image normalization beyond resize/last-frame selection, so dividing by
    # 255 here would silently measure a different teacher distribution.
    frames_f32 = frames_thwc_uint8.astype(np.float32)  # (T, H, W, 3)
    # Per upstream/modules.py SegNet.preprocess_input: x = x[:, -1, ...] then
    # interpolate to (segnet_model_input_size[1], segnet_model_input_size[0]).
    # The pipeline targets are already at canonical (384, 512) per
    # CANONICAL_EVAL_SIZE so the interpolation is a no-op on the spatial
    # dims; the only transform left is NHWC -> NCHW + last-frame slicing.
    teacher_logits_chunks = []
    chunk_size = 16  # batch SegNet forwards in chunks of 16 to keep CPU
    # memory bounded
    with torch.inference_mode():
        for start in range(0, frames_f32.shape[0], chunk_size):
            end = min(start + chunk_size, frames_f32.shape[0])
            chunk = frames_f32[start:end]  # (b, H, W, 3)
            chunk_nchw = np.transpose(chunk, (0, 3, 1, 2))  # (b, 3, H, W)
            # Build (b, 2, 3, H, W) for SegNet's 2-frame input contract.
            stacked = np.stack([chunk_nchw, chunk_nchw], axis=1)
            x = torch.from_numpy(stacked).to(device)
            # Per upstream/modules.py SegNet.debug_run: callers MUST invoke
            # `x = self.preprocess_input(x)` before `self(x)`. SegNet.forward
            # does NOT call preprocess_input internally.
            x_pre = segnet.preprocess_input(x)  # (b, 3, 384, 512)
            logits = segnet(x_pre)  # (b, 5, 384, 512)
            # NHWC + numpy then convert to MLX float32 below.
            logits_np = logits.detach().cpu().numpy()
            teacher_logits_chunks.append(np.transpose(logits_np, (0, 2, 3, 1)))
    teacher_logits_np = np.concatenate(teacher_logits_chunks, axis=0)  # (T, 384, 512, 5)
    cache_dt = cache_dtype if cache_dtype is not None else mx.float32
    teacher_logits_mx = mx.array(teacher_logits_np, dtype=cache_dt)
    cache_seconds = _time.time() - t0
    return RealSegNetTeacherLogitsCache(
        teacher_logits_thwk=teacher_logits_mx,
        frame_count=int(teacher_logits_np.shape[0]),
        height=int(teacher_logits_np.shape[1]),
        width=int(teacher_logits_np.shape[2]),
        num_classes=int(teacher_logits_np.shape[3]),
        upstream_segnet_safetensors_sha256=segnet_safetensors_sha256,
        cache_build_seconds=cache_seconds,
    )


# ---------------------------------------------------------------------------
# Canonical custom_loss_fn factory matching SubstrateAdapterScaffold contract
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HintonDistilledKLLossResult:
    """Telemetry-bearing snapshot of one ``custom_loss_fn`` invocation.

    Carries enough provenance for the canonical Slot 1 telemetry row to
    surface the distillation-specific signal alongside the MSE
    reconstruction term. The Slot 1 ``StageTelemetryRow.loss`` field
    captures the SCALAR combined loss; this struct provides the
    decomposition so the operator can inspect whether reconstruction or
    distillation dominates.
    """

    combined_loss: float
    reconstruction_mse: float
    distillation_kl_t2_loss: float
    distillation_temperature: float
    distillation_weight: float
    student_logits_shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class HintonMlxCustomLossFnConfig:
    """Configuration for :func:`make_hinton_custom_loss_fn`.

    Args:
        distillation_weight: Scalar ``λ`` such that the combined loss is
            ``L = mse + λ * L_distill``. The default ``0.5`` balances
            reconstruction (RGB MSE — the Slot 1 canonical foundation
            objective) against distillation (Hinton KL T=2.0 — this
            module's added scorer-aware signal).
        temperature: Hinton distillation temperature. Default 2.0 per
            Quantizr canonical anchor + Probe 6 + Probe 7 empirical anchors.
        student_head_out_channels: Number of student-side classes the
            student decoder emits as logits. Defaults to
            :data:`DEFAULT_SEGNET_CLASSES` = 5 (matching canonical SegNet).
        teacher_provider: Canonical teacher-logits source per
            :class:`TeacherLogitsProvider` protocol. Defaults to
            :class:`MockTeacherLogitsProvider` for $0 smoke.
        evidence_grade: Canonical MLX evidence grade. Defaults to
            ``macOS-MLX-research-signal`` per CLAUDE.md "MLX
            portable-local-substrate authority" non-negotiable. NOT
            operator-overridable per the non-negotiable.
    """

    distillation_weight: float = 0.5
    temperature: float = DEFAULT_DISTILLATION_TEMPERATURE
    student_head_out_channels: int = DEFAULT_SEGNET_CLASSES
    teacher_provider: TeacherLogitsProvider | None = None
    real_teacher_cache: RealSegNetTeacherLogitsCache | None = None
    evidence_grade: str = EVIDENCE_GRADE_MLX

    def __post_init__(self) -> None:
        if self.real_teacher_cache is not None and self.real_teacher_cache.num_classes != self.student_head_out_channels:
            raise ValueError(
                f"real_teacher_cache.num_classes ({self.real_teacher_cache.num_classes}) "
                f"must match student_head_out_channels ({self.student_head_out_channels})"
            )
        if self.distillation_weight < 0.0:
            raise ValueError(
                f"distillation_weight must be >= 0 (set to 0.0 to disable "
                f"distillation and recover pure MSE); got {self.distillation_weight}"
            )
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0 (Hinton 2014 canonical T=2.0); got {self.temperature}"
            )
        if self.student_head_out_channels < 2:
            raise ValueError(
                f"student_head_out_channels must be >= 2; got {self.student_head_out_channels}"
            )
        if self.evidence_grade != EVIDENCE_GRADE_MLX:
            # CLAUDE.md "MLX portable-local-substrate authority" non-negotiable:
            # MLX evidence MUST be tagged macOS-MLX research-signal. This
            # invariant is enforced at construction time so a future
            # refactor cannot silently downgrade the axis tag.
            raise ValueError(
                f"evidence_grade must be {EVIDENCE_GRADE_MLX!r} per "
                f"CLAUDE.md 'MLX portable-local-substrate authority' "
                f"non-negotiable; got {self.evidence_grade!r}"
            )


def _student_logits_from_decoded(
    decoded_bhwc: Any,
    config: HintonMlxCustomLossFnConfig,
) -> Any:
    """Project decoded RGB frames into student logits via a fixed linear head.

    For the foundation MLX smoke we do NOT add a learnable student head;
    instead the student "logits" are produced by a deterministic per-pixel
    projection that depends ONLY on the decoded RGB (so gradients flow
    through the decoder weights but not through any new trainable params).
    This keeps the foundation smoke surface minimal while still exercising
    the canonical Hinton KL T=2.0 contract end-to-end.

    The projection reuses the teacher provider on the decoded frame path
    while the stopped teacher target in :func:`make_hinton_custom_loss_fn`
    is computed from the target frame path. A future sister wave can swap
    this for a learnable student head (additional trainable parameters) to
    extend the paradigm; for now the canonical contract surface is the loss
    function itself, not the student head architecture.

    Returns student logits of shape
    ``(B, H // teacher_downsample, W // teacher_downsample, num_classes)``.
    """
    _require_mlx()
    provider = config.teacher_provider
    assert provider is not None, "teacher_provider required (resolved by factory)"
    # Use the provider's projection on the DECODED frames so the gradient
    # flows from KL -> student_logits -> decoded -> decoder parameters.
    # The stopped teacher target is computed from the TARGET frames below,
    # which keeps the KL term non-self-referential.
    return provider.teacher_logits(decoded_bhwc)


def make_hinton_custom_loss_fn(
    config: HintonMlxCustomLossFnConfig | None = None,
) -> Callable[[Any, Any, Any], Any]:
    """Build a canonical Hinton-distilled KL T=2.0 ``custom_loss_fn``.

    The returned callable matches the
    :class:`tac.local_acceleration.pr95_hnerv_mlx_long_training.SubstrateAdapterScaffold.custom_loss_fn`
    signature ``(bundle, indices, targets_batch) -> mx.array`` so the
    operator can pass the resulting callable directly into a
    ``SubstrateAdapterScaffold`` or subclass
    :class:`tac.local_acceleration.pr95_hnerv_mlx_long_training.MLXLongTrainingPipeline`
    and override :meth:`loss_fn` to delegate to it.

    The combined loss is::

        L = mse(decoded_frame_0, targets) + λ * T**2 * KL(student || teacher)

    where:
      * ``decoded_frame_0 = bundle(indices)[:, 0]`` is the canonical Slot
        1 reconstruction (in [0, 255] then normalized to [0, 1]).
      * ``mse`` is the Slot 1 canonical RGB MSE term (same definition as
        :meth:`MLXLongTrainingPipeline.loss_fn`).
      * ``student`` = ``softmax(student_logits_from(decoded_frame_0) / T)``.
      * ``teacher`` = ``softmax(teacher_provider.teacher_logits(targets_batch) / T)``.

    The student consumes the DECODED frame while the stopped teacher consumes
    the TARGET frame. This avoids the self-KL false positive where student
    and teacher logits are identical functions of the same decoded tensor.

    Args:
        config: Hinton loss configuration. If ``None``, uses the
            canonical defaults (T=2.0, distillation_weight=0.5,
            num_classes=5, MockTeacherLogitsProvider, evidence_grade
            ``macOS-MLX-research-signal``).

    Returns:
        A callable ``custom_loss_fn(bundle, indices, targets_batch)`` that
        returns a scalar MLX loss suitable for ``mx.value_and_grad``.
    """
    _require_mlx()
    if config is None:
        config = HintonMlxCustomLossFnConfig()
    # Resolve teacher provider after config construction so the default
    # is constructed only when needed.
    if config.teacher_provider is None:
        config = dataclasses.replace(
            config,
            teacher_provider=MockTeacherLogitsProvider(
                num_classes=config.student_head_out_channels,
            ),
        )

    def _custom_loss_fn(bundle: Any, indices: Any, targets_batch: Any) -> Any:
        decoded = bundle(indices)
        # Canonical Slot 1 normalization: (B, 2, 3, H, W) in [0, 255] →
        # frame_0 (B, 3, H, W) normalized to [0, 1] → NHWC.
        decoded_f0 = decoded[:, 0] / 255.0
        decoded_bhwc = mx.transpose(decoded_f0, (0, 2, 3, 1))
        # Reconstruction MSE (sister of the Slot 1 canonical loss_fn).
        diff = decoded_bhwc - targets_batch
        mse = mx.mean(diff * diff)
        # Hinton KL T=2.0 distillation term.
        # Student-side: always the MLX-native projection on decoded frames
        # (pure-MLX, gradient-bearing path from KL -> decoder weights).
        student_logits = _student_logits_from_decoded(decoded_bhwc, config)
        # Teacher-side: prefer the real-SegNet cache (canonical contest
        # teacher) when wired; else fall back to the mock provider on the
        # target frames (foundation $0 smoke surface). The cache lookup is
        # O(1) MLX indexing per Catalog #305 observability + queryable
        # post-hoc.
        if config.real_teacher_cache is not None:
            teacher_logits = config.real_teacher_cache.teacher_logits_for_indices(
                indices
            )
        else:
            teacher_logits = config.teacher_provider.teacher_logits(targets_batch)
        # Stop the gradient on the teacher path so distillation matches the
        # Hinton 2014 canonical pattern (teacher provides soft targets, not
        # gradient signal): the teacher is either a pure tensor function
        # (mock) or a pre-computed cache (real SegNet) — both are gradient-
        # blocked here as defense-in-depth.
        teacher_logits_stopped = mx.stop_gradient(teacher_logits)
        distill = hinton_distilled_kl_t2_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits_stopped,
            temperature=config.temperature,
        )
        combined = mse + config.distillation_weight * distill
        return combined

    return _custom_loss_fn


def custom_loss_fn_canonical_signature_hash() -> str:
    """Stable hash for the canonical ``custom_loss_fn`` contract.

    Returned as a hex sha256 over the canonical signature string so future
    sister waves can verify the contract is unchanged across the loss-fn
    factory + the Slot 1 SubstrateAdapterScaffold field type.
    """
    sig = (
        "custom_loss_fn(bundle: Any, indices: Any, targets_batch: Any) -> Any "
        f"| evidence_grade={EVIDENCE_GRADE_MLX} "
        f"| evidence_tag={EVIDENCE_TAG_MLX} "
        "| canonical=hinton_distilled_kl_t2_loss "
        "| student_logits=provider(decoded_frame_0) "
        "| teacher_logits=stop_gradient(provider(targets_batch)) "
        "| temperature=DEFAULT_DISTILLATION_TEMPERATURE=2.0 "
        "| num_classes=DEFAULT_SEGNET_CLASSES=5"
    )
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()
