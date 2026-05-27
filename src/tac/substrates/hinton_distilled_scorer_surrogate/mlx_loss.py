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


def pose_distillation_mse_loss(
    student_pose: Any,
    teacher_pose: Any,
    *,
    per_dim_scale: Any = None,
) -> Any:
    """Canonical pose-distillation MSE loss (the POSE axis sister of KL T=2.0).

    The contest PoseNet distortion is the mean-squared-error between the two
    poses' first 6 dims (``upstream/modules.py`` PoseNet.compute_distortion:
    ``(out1[h.name][..., :h.out//2] - out2[h.name][..., :h.out//2]).pow(2).mean``).
    So the pose distillation target is NOT a softmax/KL (pose is a continuous
    ego-motion vector, NOT a class distribution); it is a direct MSE between the
    student's predicted pose and the REAL PoseNet's teacher pose. Distilling the
    student toward the teacher pose (gradient-blocked) pulls the renderer toward
    frames whose REAL PoseNet pose matches the target's.

    PER-DIM SCALING (why ``per_dim_scale`` matters): the real PoseNet pose dims
    have wildly different magnitudes — empirically dim 0 (a depth/forward
    translation term) has mean ~34 and std ~0.6 while the 5 rotation/lateral
    dims have std ~0.01. A raw MSE is dominated ENTIRELY by dim 0's offset, so
    the 5 informative ego-motion dims receive ~no gradient AND the loss scale
    (O(180)) swamps the recon (O(0.006)) + SegNet-KL (O(3)) terms — at weight
    0.5 the pose gradient destroys reconstruction (empirically recon_mse 0.006
    -> 0.244). Standardizing the squared error by the teacher's per-dim std
    makes each pose dim contribute proportionally to its informativeness (a
    Mahalanobis-like distance) AND yields a scale-stable loss O(1) comparable to
    the other terms. This is the canonical scorer-bound pose objective.

    Args:
        student_pose: MLX float32 ``(B, pose_dims)`` predicted pose
            (gradient-bearing through the learnable pose head -> renderer).
        teacher_pose: MLX float32 ``(B, pose_dims)`` real-PoseNet teacher pose
            (caller must pass ``mx.stop_gradient(...)`` to block the teacher).
        per_dim_scale: optional MLX float32 ``(pose_dims,)`` per-dim divisor
            applied to the error (the canonical choice is the teacher per-dim
            std, supplied by the teacher cache via ``per_dim_scale``). ``None``
            recovers the raw unstandardized MSE.

    Returns:
        Scalar MLX array — ``mean(((student - teacher) / scale) ** 2)``
        (raw ``mean((student - teacher) ** 2)`` when ``per_dim_scale`` is None).
    """
    _require_mlx()
    diff = student_pose - teacher_pose
    if per_dim_scale is not None:
        diff = diff / mx.maximum(per_dim_scale, _NUMERIC_FLOOR)
    return mx.mean(diff * diff)


# ---------------------------------------------------------------------------
# Canonical real-PoseNet teacher cache (the POSE axis sister of
# RealSegNetTeacherLogitsCache). Holds the REAL contest PoseNet's pose for every
# pair, pre-computed gradient-free pre-training, indexed by PAIR index.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RealPoseNetTeacherCache:
    """Pre-computed real upstream-PyTorch PoseNet pose indexed by PAIR index.

    Wraps an MLX float32 array ``(num_pairs, pose_dims)`` where each row is the
    REAL contest PoseNet's pose (first ``pose_dims`` of the 12-dim pose head) on
    the corresponding pair's TWO TARGET frames. Built ONCE pre-training (one
    PoseNet forward per pair, gradient-free, CPU per CLAUDE.md "MPS auth eval is
    NOISE") then indexed by batch ``idx`` during training — pure MLX lookup, no
    PyTorch round-trip per step.

    Satisfies the
    :class:`tac.substrates._shared.mlx_score_aware.bundle.PoseScorerTeacherProvider`
    protocol (``pose_dims`` + ``teacher_pose_for_indices``).

    Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192: holds
    ``[macOS-MLX research-signal]`` data; NO contest-score authority on its own.
    """

    teacher_pose_np: Any  # MLX float32 (num_pairs, pose_dims)
    num_pairs: int
    pose_dims: int
    #: MLX float32 ``(pose_dims,)`` per-dim std of the teacher poses, used as the
    #: canonical ``per_dim_scale`` divisor in :func:`pose_distillation_mse_loss`
    #: so the dominant-magnitude pose dim does not swamp the loss (see that
    #: function's PER-DIM SCALING note). Computed by the builder; ``None`` falls
    #: back to raw unstandardized MSE.
    per_dim_scale: Any = None
    upstream_posenet_safetensors_sha256: str | None = None
    cache_build_seconds: float | None = None

    def __post_init__(self) -> None:
        _require_mlx()
        if self.num_pairs < 1:
            raise ValueError(f"num_pairs must be >= 1; got {self.num_pairs}")
        if self.pose_dims < 1:
            raise ValueError(f"pose_dims must be >= 1; got {self.pose_dims}")
        if tuple(self.teacher_pose_np.shape) != (self.num_pairs, self.pose_dims):
            raise ValueError(
                "teacher_pose_np must have shape "
                f"({self.num_pairs}, {self.pose_dims}); got "
                f"{self.teacher_pose_np.shape!r}"
            )
        if self.per_dim_scale is not None and tuple(
            self.per_dim_scale.shape
        ) != (self.pose_dims,):
            raise ValueError(
                f"per_dim_scale must have shape ({self.pose_dims},); got "
                f"{self.per_dim_scale.shape!r}"
            )

    def teacher_pose_for_indices(self, indices: Any) -> Any:
        """Look up real PoseNet teacher pose via MLX integer indexing.

        Args:
            indices: MLX int32 ``(B,)`` pair-index batch.

        Returns:
            MLX float32 ``(B, pose_dims)``.
        """
        _require_mlx()
        return self.teacher_pose_np[indices]


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
    # CASCADE B 2026-05-26 Path A extension per sister Hinton-MLX bundle
    # `lane_hinton_mlx_first_local_pivot_20260526` Path A reactivation
    # criterion. When None (default), the canonical loss falls through to
    # the sister deterministic-projection back-compat path. When set, the
    # learnable 1x1-conv student head is invoked at every loss step (Path A);
    # the head's ~20 trainable params are managed by a sibling MLX optimizer
    # external to this dataclass (see canonical training loop extension at
    # `tools/run_hinton_mlx_long_training_smoke.py`'s
    # ``--learnable-student-head`` branch and the CASCADE B landing memo).
    learnable_student_head: LearnableConv1x1StudentHead | None = None

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
        if (
            self.learnable_student_head is not None
            and self.learnable_student_head.num_classes != self.student_head_out_channels
        ):
            raise ValueError(
                f"learnable_student_head.num_classes ({self.learnable_student_head.num_classes}) "
                f"must match student_head_out_channels ({self.student_head_out_channels})"
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


# ---------------------------------------------------------------------------
# Path A learnable 1x1-conv student head (CASCADE B HINTON KL-DISTILL CATALYST
# DISTORTION-ATTACK 2026-05-26 self-protection per the sister Path A
# reactivation criterion enumerated in
# `lane_hinton_mlx_first_local_pivot_20260526` commit `dfc1d11de`).
#
# Sister predecessor's empirical anchor: the deterministic-projection student
# head saturated KL T=2.0 loss at ~3.03 across 1000ep on the real upstream
# SegNet teacher cache. Per Catalog #307: IMPLEMENTATION-LEVEL falsification;
# Hinton paradigm INTACT. Path A canonical reactivation = learnable 1x1-conv
# student head with ~20 trainable params (3 input RGB channels x 5 SegNet
# classes + 5 bias = 20).
#
# This is the canonical APPEND-ONLY extension per Catalog #110/#113. The
# sister deterministic projection at `_student_logits_from_decoded` remains
# the back-compat default; Path A is opt-in via
# `HintonMlxCustomLossFnConfig.learnable_student_head is not None`.
#
# The learnable head holds its own MLX trainable parameters (an ``mx.array``
# weight + bias) and computes ``logits = decoded_bhwc @ W + b`` as the
# canonical 1x1-conv equivalent in NHWC. Gradients flow into both the
# decoder (via the standard ``nn.value_and_grad(self._bundle, loss_fn)`` path)
# AND the learnable head's weights (via a sibling ``mx.value_and_grad`` on
# the head; see the canonical training loop extension at
# `tools/run_hinton_mlx_long_training_smoke.py`'s ``--learnable-student-head``
# branch and the runtime composition documented in the landing memo).
#
# Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable +
# Catalog #192: the learnable head is part of the [macOS-MLX research-signal]
# axis evidence inherited from `EVIDENCE_GRADE_MLX` via the existing
# `HintonMlxCustomLossFnConfig.__post_init__` invariant; the field carries no
# contest-score authority on its own.
#
# Per CLAUDE.md "MLX portable-local-substrate authority" + the sister
# `numpy_pytorch_parity_proof.json` BYTE_STABLE_BY_DEFAULT proof: the head's
# ~20 float32 parameters export via the canonical
# `tac.local_acceleration.mlx_to_pytorch_export.export_mlx_state_dict_to_torch_pt`
# bridge with the same byte-level invariants.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LearnableConv1x1StudentHead:
    """Canonical Path A learnable 1x1-conv student head.

    A minimal MLX-native learnable mapping ``(R, G, B) -> K-class logits``
    that extends the canonical Hinton-distilled scorer surrogate's student
    side to break the deterministic-projection saturation point empirically
    confirmed at KL T=2.0 ~3.03 in sister `lane_hinton_mlx_first_local_pivot_20260526`
    (commit `dfc1d11de`).

    Parameters:
      * ``weight`` (mx.array, shape ``(in_channels, num_classes)``): 1x1
        convolution weight; ``in_channels`` is always 3 (RGB) per the
        canonical Slot 1 normalization.
      * ``bias`` (mx.array, shape ``(num_classes,)``): per-class bias.

    Total trainable parameter count: ``in_channels * num_classes + num_classes``
    = ``3 * 5 + 5`` = 20 for the canonical SegNet 5-class config.

    Note: this dataclass is NOT frozen because MLX arrays are mutable
    references; the canonical training loop extension uses
    ``mx.value_and_grad`` on the head's ``__call__`` method to compute
    gradients and updates the params in-place via the sibling optimizer.

    Sister back-compat: when ``HintonMlxCustomLossFnConfig.learnable_student_head``
    is ``None``, ``_student_logits_from_decoded`` falls through to the
    canonical deterministic projection (sister 1000ep run is reproducible
    bit-for-bit when this field is unset).
    """

    weight: Any  # mx.array of shape (in_channels=3, num_classes)
    bias: Any  # mx.array of shape (num_classes,)
    num_classes: int = DEFAULT_SEGNET_CLASSES

    def __post_init__(self) -> None:
        _require_mlx()
        if self.num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2 for KL distillation; got {self.num_classes}"
            )
        # Validate weight + bias shapes vs num_classes (defense-in-depth so
        # a misconfigured head cannot silently produce shape-mismatched KL).
        if self.weight.shape[-1] != self.num_classes:
            raise ValueError(
                f"weight last dim must equal num_classes={self.num_classes}; "
                f"got weight.shape={self.weight.shape!r}"
            )
        if self.bias.shape[-1] != self.num_classes:
            raise ValueError(
                f"bias last dim must equal num_classes={self.num_classes}; "
                f"got bias.shape={self.bias.shape!r}"
            )

    def __call__(self, decoded_bhwc: Any) -> Any:
        """Apply the canonical 1x1-conv: logits[b,h,w,k] = sum_c decoded[b,h,w,c] * W[c,k] + b[k].

        Args:
            decoded_bhwc: MLX float32 array of shape ``(B, H, W, 3)`` in
                ``[0, 1]`` per the Slot 1 canonical normalization.

        Returns:
            MLX float32 array of shape ``(B, H, W, num_classes)``.
        """
        _require_mlx()
        # Canonical 1x1-conv in NHWC: einsum-equivalent matmul on the
        # channel axis. mx supports this via a simple matmul because the
        # last axis of decoded_bhwc and the first axis of weight align.
        logits = mx.einsum("bhwc,ck->bhwk", decoded_bhwc, self.weight) + self.bias
        return logits

    def parameters_dict(self) -> dict[str, Any]:
        """Return the canonical parameter dict for optimizer + parity proof.

        Keys match the canonical MLX state-dict layout consumed by
        :func:`tac.local_acceleration.mlx_to_pytorch_export.export_mlx_state_dict_to_torch_pt`:
        ``{"learnable_student_head.weight": ..., "learnable_student_head.bias": ...}``.

        This is the surface the sister numpy<->PyTorch parity proof
        extends to cover the new head's ~20 params alongside the canonical
        decoder's 228,958 params.
        """
        return {
            "learnable_student_head.weight": self.weight,
            "learnable_student_head.bias": self.bias,
        }


def build_learnable_student_head(
    *,
    num_classes: int = DEFAULT_SEGNET_CLASSES,
    in_channels: int = 3,
    seed: int = 0,
    init_scale: float = 0.1,
) -> LearnableConv1x1StudentHead:
    """Construct a canonical :class:`LearnableConv1x1StudentHead` with
    deterministic initialization.

    Args:
        num_classes: Number of SegNet output classes. Default 5 per the
            canonical contest SegNet.
        in_channels: Number of input channels (3 for RGB; the canonical
            Slot 1 pipeline normalizes targets to NHWC RGB float32 in [0, 1]).
        seed: Deterministic seed for weight initialization per CLAUDE.md
            "Beauty, simplicity, and developer experience" + Catalog #305
            observability "diff-able across runs".
        init_scale: Standard deviation of the per-weight Gaussian
            initialization. Default 0.1 (small enough that initial logits
            land near the linear regime of softmax-with-T=2.0).

    Returns:
        :class:`LearnableConv1x1StudentHead` with weight + bias initialized
        from a deterministic Gaussian. The 20-param head is small enough that
        the deterministic init is itself the canonical fixture.
    """
    _require_mlx()
    if num_classes < 2:
        raise ValueError(
            f"num_classes must be >= 2 for KL distillation; got {num_classes}"
        )
    if in_channels < 1:
        raise ValueError(f"in_channels must be >= 1; got {in_channels}")
    if init_scale <= 0.0:
        raise ValueError(f"init_scale must be > 0; got {init_scale}")
    # Deterministic init via mx.random.key + mx.random.normal so the head
    # is reproducible bit-for-bit across runs with the same seed.
    rng_key = mx.random.key(seed)
    key_w, key_b = mx.random.split(rng_key)
    weight = mx.random.normal(
        shape=(in_channels, num_classes), key=key_w
    ) * init_scale
    # Bias initialized small + offset per class so initial logits are not
    # all-zero (which would produce a uniform softmax and zero KL gradient
    # on a uniform teacher).
    bias = mx.random.normal(
        shape=(num_classes,), key=key_b
    ) * (init_scale * 0.5)
    return LearnableConv1x1StudentHead(
        weight=weight,
        bias=bias,
        num_classes=num_classes,
    )


# ---------------------------------------------------------------------------
# Canonical PoseNet pose-distillation student head (MLX-HARNESS-POSENET-TEACHER-
# BINDING 2026-05-27). Sister of the SegNet ``LearnableConv1x1StudentHead`` for
# the POSE axis — the dominant-at-frontier scorer component (per CLAUDE.md
# "SegNet vs PoseNet importance — operating-point dependent": below
# pose_avg ~ 2.5e-4 the pose marginal exceeds SegNet's; at the ~0.192 frontier
# pose is ~2.71x more important by marginal-value-per-byte).
#
# Why a DIFFERENT student shape than the SegNet head: the contest PoseNet emits
# a GLOBAL per-pair 6-dim pose vector (NOT per-pixel logits), and pose
# distortion is MSE on the first 6 pose dims (NOT argmax-disagreement / KL). So
# the pose student maps the DECODED FRAME PAIR -> a 6-dim pose vector and is
# distilled toward the REAL PoseNet's pose on the TARGET pair via MSE.
#
# Why a learnable head (NOT full-PoseNet backprop): identical to the SegNet
# finding — backprop through the full ported MLX FastViT PoseNet composed with
# the renderer's PixelShuffle/bilinear backward NaNs in MLX's second-order
# autograd (the first-order grad-to-input IS finite; the second-order
# composition is not). The learnable pose head gives a FINITE, genuinely
# scorer-bound gradient: it learns decoded-pair-RGB -> real-PoseNet-pose, so the
# renderer is pulled toward frames whose REAL PoseNet pose matches the target's,
# NOT toward a pixel-MSE-redundant direction.
#
# Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192: the
# head is [macOS-MLX research-signal]; it carries no contest-score authority on
# its own. Its ~few-hundred float32 params export via the canonical
# MLX->PyTorch bridge with the same byte-level invariants as the SegNet head.
# ---------------------------------------------------------------------------


#: Canonical contest pose dimensionality used for distortion (first 6 of the
#: PoseNet ``pose`` head per ``upstream/modules.py`` ``compute_distortion``
#: ``out[..., : h.out // 2]`` with the 12-dim pose head).
DEFAULT_POSE_DIMS: int = 6
#: Canonical pooled-grid resolution for the pose student's per-frame spatial
#: feature. PoseNet pose is a global ego-motion estimate; a coarse 4x4 grid of
#: per-channel means is sufficient signal for the linear pose projection while
#: keeping the head tiny + the gradient finite.
DEFAULT_POSE_POOL_GRID: int = 4


@dataclasses.dataclass
class LearnablePoseStudentHead:
    """Canonical learnable pose-distillation student head.

    Maps a DECODED FRAME PAIR ``(rgb_0, rgb_1)`` (each ``(B, H, W, 3)`` in
    ``[0, 1]``) to a ``(B, pose_dims)`` pose vector via a coarse spatial pool +
    linear projection. Distilled (MSE) toward the REAL PoseNet's pose on the
    pair's TARGET frames.

    Architecture (deliberately minimal so the gradient is finite + the head is
    cheap to train jointly with the renderer):

      * Per frame: average-pool the ``(B, H, W, 3)`` RGB to a
        ``(B, grid, grid, 3)`` coarse grid (canonical global-ego-motion
        feature), then flatten to ``(B, grid*grid*3)``.
      * Concatenate both frames' pooled features -> ``(B, 2*grid*grid*3)``.
      * Linear projection ``feature @ weight + bias`` -> ``(B, pose_dims)``.

    Parameters:
      * ``weight`` (mx.array, shape ``(feature_dim, pose_dims)``) where
        ``feature_dim = 2 * grid * grid * 3``.
      * ``bias`` (mx.array, shape ``(pose_dims,)``).

    For the canonical ``grid=4`` + ``pose_dims=6``: ``feature_dim = 96`` and
    total params ``= 96 * 6 + 6 = 582``.

    Not frozen (MLX arrays are mutable references); the harness trains the head
    jointly via a sibling ``mx.value_and_grad`` step, identical to the SegNet
    head.
    """

    weight: Any  # mx.array (feature_dim, pose_dims)
    bias: Any  # mx.array (pose_dims,)
    pose_dims: int = DEFAULT_POSE_DIMS
    pool_grid: int = DEFAULT_POSE_POOL_GRID

    def __post_init__(self) -> None:
        _require_mlx()
        if self.pose_dims < 1:
            raise ValueError(f"pose_dims must be >= 1; got {self.pose_dims}")
        if self.pool_grid < 1:
            raise ValueError(f"pool_grid must be >= 1; got {self.pool_grid}")
        if self.weight.shape[-1] != self.pose_dims:
            raise ValueError(
                f"weight last dim must equal pose_dims={self.pose_dims}; "
                f"got weight.shape={self.weight.shape!r}"
            )
        if self.bias.shape[-1] != self.pose_dims:
            raise ValueError(
                f"bias last dim must equal pose_dims={self.pose_dims}; "
                f"got bias.shape={self.bias.shape!r}"
            )
        expected_feat = 2 * self.pool_grid * self.pool_grid * 3
        if self.weight.shape[0] != expected_feat:
            raise ValueError(
                f"weight first dim must equal 2*pool_grid^2*3={expected_feat} "
                f"(both frames, coarse {self.pool_grid}x{self.pool_grid} RGB "
                f"pool); got weight.shape={self.weight.shape!r}"
            )

    def _pool_frame(self, rgb_bhwc: Any) -> Any:
        """Average-pool ``(B, H, W, 3)`` -> flattened ``(B, grid*grid*3)``.

        Uses adaptive averaging via reshape+mean to a ``pool_grid`` x
        ``pool_grid`` grid. Spatial dims need not be divisible by ``pool_grid``;
        the canonical path crops to the largest divisible extent (a sub-pixel
        loss negligible for a global ego-motion feature).
        """
        _require_mlx()
        b, h, w, c = rgb_bhwc.shape
        g = self.pool_grid
        # Crop to the largest h, w divisible by g (canonical coarse pool).
        h2 = (h // g) * g
        w2 = (w // g) * g
        cropped = rgb_bhwc[:, :h2, :w2, :]
        fh = h2 // g
        fw = w2 // g
        if fh < 1 or fw < 1:
            raise ValueError(
                f"spatial dims {(h, w)} are too small for pool_grid={g}; "
                "need at least one pixel per pooled cell."
            )
        # (B, g, fh, g, fw, C) -> mean over the within-cell axes.
        reshaped = mx.reshape(cropped, (b, g, fh, g, fw, c))
        pooled = mx.mean(reshaped, axis=(2, 4))  # (B, g, g, C)
        return mx.reshape(pooled, (b, g * g * c))

    def forward_with_params(
        self,
        rgb_0_bhwc: Any,
        rgb_1_bhwc: Any,
        params: dict[str, Any],
    ) -> Any:
        """Map a decoded pair to pose using an explicit parameter dict.

        The sibling optimizer in the MLX score-aware adapter differentiates
        with respect to ``{"weight": ..., "bias": ...}`` rather than mutating
        ``self`` inside the gradient closure. Keeping this helper on the head
        prevents the adapter from depending on private pooling internals.
        """
        _require_mlx()
        f0 = self._pool_frame(rgb_0_bhwc)
        f1 = self._pool_frame(rgb_1_bhwc)
        feat = mx.concatenate([f0, f1], axis=-1)
        return feat @ params["weight"] + params["bias"]

    def __call__(self, rgb_0_bhwc: Any, rgb_1_bhwc: Any) -> Any:
        """Map the decoded frame pair to a ``(B, pose_dims)`` pose vector.

        Args:
            rgb_0_bhwc: decoded frame 0 ``(B, H, W, 3)`` in ``[0, 1]``.
            rgb_1_bhwc: decoded frame 1 ``(B, H, W, 3)`` in ``[0, 1]``.

        Returns:
            MLX float32 ``(B, pose_dims)`` predicted pose.
        """
        return self.forward_with_params(
            rgb_0_bhwc,
            rgb_1_bhwc,
            {"weight": self.weight, "bias": self.bias},
        )

    def parameters_dict(self) -> dict[str, Any]:
        """Canonical parameter dict for optimizer + MLX->PyTorch export."""
        return {
            "learnable_pose_student_head.weight": self.weight,
            "learnable_pose_student_head.bias": self.bias,
        }


def build_learnable_pose_student_head(
    *,
    pose_dims: int = DEFAULT_POSE_DIMS,
    pool_grid: int = DEFAULT_POSE_POOL_GRID,
    seed: int = 0,
    init_scale: float = 0.05,
) -> LearnablePoseStudentHead:
    """Construct a canonical :class:`LearnablePoseStudentHead` deterministically.

    Args:
        pose_dims: Number of pose dims the head emits (default 6 — the contest
            pose distortion uses the first 6 of the 12-dim PoseNet pose head).
        pool_grid: Coarse spatial pool resolution per frame (default 4x4).
        seed: Deterministic init seed (Catalog #305 diff-able-across-runs).
        init_scale: Gaussian init stddev. Default 0.05 — small so initial pose
            predictions land near zero (the teacher pose vectors are O(0.1)
            ego-motion values, so a small init keeps the first MSE gradient
            well-scaled).

    Returns:
        :class:`LearnablePoseStudentHead` with deterministic weight + bias.
    """
    _require_mlx()
    if pose_dims < 1:
        raise ValueError(f"pose_dims must be >= 1; got {pose_dims}")
    if pool_grid < 1:
        raise ValueError(f"pool_grid must be >= 1; got {pool_grid}")
    if init_scale <= 0.0:
        raise ValueError(f"init_scale must be > 0; got {init_scale}")
    feature_dim = 2 * pool_grid * pool_grid * 3
    rng_key = mx.random.key(seed)
    key_w, key_b = mx.random.split(rng_key)
    weight = mx.random.normal(shape=(feature_dim, pose_dims), key=key_w) * init_scale
    bias = mx.random.normal(shape=(pose_dims,), key=key_b) * (init_scale * 0.5)
    return LearnablePoseStudentHead(
        weight=weight,
        bias=bias,
        pose_dims=pose_dims,
        pool_grid=pool_grid,
    )


def _student_logits_from_decoded(
    decoded_bhwc: Any,
    config: HintonMlxCustomLossFnConfig,
) -> Any:
    """Project decoded RGB frames into student logits.

    Two canonical paths (selected by ``config.learnable_student_head``):

    1. **Path A learnable 1x1-conv** (default when
       ``config.learnable_student_head is not None``): applies the canonical
       :class:`LearnableConv1x1StudentHead` to ``decoded_bhwc``. Gradients
       flow through both the decoder weights AND the learnable head's
       weights via the sister sibling-optimizer pattern documented in the
       canonical training loop extension.

    2. **Deterministic projection** (default when
       ``config.learnable_student_head is None``): falls through to the
       canonical ``teacher_provider.teacher_logits(decoded_bhwc)`` deterministic
       cosine projection. This is the sister back-compat path that preserves
       bit-for-bit reproducibility of the sister 1000ep run.

    The choice between paths is determined ENTIRELY by
    ``config.learnable_student_head``; the runtime never blends.

    Returns student logits of shape ``(B, H, W, num_classes)`` for Path A
    (1x1-conv preserves spatial dims) OR
    ``(B, H // teacher_downsample, W // teacher_downsample, num_classes)``
    for the deterministic projection path (sister behavior preserved).
    """
    _require_mlx()
    # Path A learnable 1x1-conv (CASCADE B 2026-05-26 extension)
    if config.learnable_student_head is not None:
        return config.learnable_student_head(decoded_bhwc)
    # Sister back-compat deterministic projection
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
