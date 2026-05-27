# SPDX-License-Identifier: MIT
"""The substrate-specific axis passed into the MLX score-aware harness.

Separation of concerns: this module owns ONLY the canonical-vs-unique BOUNDARY
contract (Catalog #290). Everything in :class:`RendererBundle` is the
substrate's UNIQUE axis; the rest of the harness package (device gate / loss /
adapter / portability / orchestrator) is substrate-AGNOSTIC.

A substrate satisfies the harness by passing a ``RendererBundle`` describing
its MLX renderer + real-video targets + (optional) extra-loss callback +
(optional) distillation weight. The ``MlxRenderer`` Protocol documents the two
canonical forward conventions a renderer may expose.

[verified-against: tac.substrates.dreamer_v3_rssm.module.DreamerV3RSSMSubstrateMLX call_b2chw_255 reference]
[verified-against: tac.substrates.atw_v2_cooperative_receiver_v2.mlx_renderer reconstruct_pair_nchw01 reference]
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
)

#: The two canonical renderer forward conventions the harness auto-detects.
FORWARD_CONVENTIONS: frozenset[str] = frozenset(
    {"reconstruct_pair_nchw01", "call_b2chw_255"}
)


@runtime_checkable
class MlxRenderer(Protocol):
    """Documentation Protocol for a harness-compatible MLX renderer.

    A renderer MUST be differentiable by MLX ``value_and_grad`` — i.e. it is an
    ``mlx.nn.Module`` (or exposes ``.parameters()`` + ``.update()``) — AND
    expose ONE of the two canonical forwards (selected via the bundle's
    ``forward_convention``):

    * ``reconstruct_pair(idx) -> (rgb_0, rgb_1)`` each ``(B, 3, H, W)`` in
      ``[0, 1]`` (the Z6 / atw_v2 / faiss / coin_pp convention), OR
    * ``__call__(idx) -> (B, 2, 3, H, W)`` in ``[0, 255]`` (the dreamer / z8
      HNeRV convention).

    This Protocol is ``runtime_checkable`` for ``.parameters()`` presence only;
    the forward method name varies by convention so it is validated at call
    time by the loss module rather than by ``isinstance``.
    """

    def parameters(self) -> Any: ...


@runtime_checkable
class ScorerTeacherProvider(Protocol):
    """Structural type for a REAL contest-scorer teacher (gradient-blocked).

    The C6 IBPS / DreamerV3 RSSM scorer-blindness lesson (per CLAUDE.md
    "HNeRV / leaderboard-implementation parity discipline" L1 + Catalog #164):
    a reconstruction-MSE proxy that never binds the contest SegNet/PoseNet
    converges the decoder's pixels (loss ~0.005) while the scorer collapses
    (seg 0.52, pose 185). The fix is a distillation term whose TEACHER target
    is the REAL contest SegNet's per-pixel class distribution (NOT a fixed
    cosine of pixel means).

    A ``ScorerTeacherProvider`` returns, for a batch of pair indices, the
    teacher logits the student head is distilled toward. The teacher is
    gradient-blocked by the loss (``mx.stop_gradient``); only the student head
    + renderer carry gradient. Implementations:

      * Production / canonical: a precomputed real-MLX-SegNet teacher logits
        cache indexed by pair index (one SegNet forward per contest SegNet
        frame, gradient-free, built ONCE pre-training). This is the canonical
        scorer-bound surrogate per Catalog #164 + the C6 IBPS lesson.
      * Mock (EXPLICIT opt-in only): the deterministic-cosine
        ``MockTeacherLogitsProvider`` — scorer-BLIND, used ONLY for a $0
        no-real-SegNet smoke and gated behind ``allow_mock_scorer_teacher``.

    The contract: ``teacher_logits_for_indices(idx)`` returns an MLX float32
    array ``(B, H', W', num_classes)`` matching the student head's logits shape
    so the Hinton-KL term is well-defined.
    """

    num_classes: int

    def teacher_logits_for_indices(self, idx: Any) -> Any:
        """Return teacher logits ``(B, H', W', num_classes)`` for pair batch ``idx``."""


@runtime_checkable
class PoseScorerTeacherProvider(Protocol):
    """Structural type for a REAL contest-PoseNet teacher (gradient-blocked).

    The POSE axis sister of :class:`ScorerTeacherProvider`. PoseNet is the
    DOMINANT scorer component at the frontier (per CLAUDE.md "SegNet vs PoseNet
    importance — operating-point dependent": below pose_avg ~ 2.5e-4 the pose
    marginal exceeds SegNet's; at the ~0.192 frontier pose is ~2.71x more
    important by marginal-value-per-byte). The SegNet-only verification run
    drifted pose +10.6 precisely because no pose teacher was wired.

    Unlike SegNet (per-pixel class logits, KL-distilled), the contest PoseNet
    emits a GLOBAL per-pair 6-dim ego-motion pose vector and its distortion is
    MSE on the first 6 dims. So a ``PoseScorerTeacherProvider`` returns, for a
    batch of pair indices, the teacher pose the learnable pose-student head is
    distilled toward (via MSE). The teacher is gradient-blocked by the loss
    (``mx.stop_gradient``); only the pose head + renderer carry gradient.

    Canonical implementation: a precomputed real-MLX-PoseNet teacher pose cache
    indexed by pair index (one PoseNet forward per pair's TWO target frames,
    gradient-free, built ONCE pre-training) per Catalog #164 + the C6 IBPS /
    DreamerV3 lesson.

    The contract: ``teacher_pose_for_indices(idx)`` returns an MLX float32 array
    ``(B, pose_dims)`` matching the pose head's output shape so the pose-MSE
    distillation term is well-defined.
    """

    pose_dims: int

    def teacher_pose_for_indices(self, idx: Any) -> Any:
        """Return teacher pose ``(B, pose_dims)`` for pair batch ``idx``."""


@dataclass
class RendererBundle:
    """Substrate-specific renderer + targets + optional extra-loss callback.

    The canonical-vs-unique boundary: everything in this bundle is the
    substrate's UNIQUE axis; the harness owns everything else (AGNOSTIC).

    Attributes:
        model: the MLX renderer. MUST be an ``mlx.nn.Module`` (or expose
            ``.parameters()`` + ``.update()`` so MLX ``value_and_grad`` can
            differentiate it) AND expose ONE of:
              * ``reconstruct_pair(idx) -> (rgb_0, rgb_1)`` with each
                ``(B, 3, H, W)`` in ``[0, 1]`` (the Z6 / atw_v2 convention), OR
              * ``__call__(idx) -> (B, 2, 3, H, W)`` in ``[0, 255]`` (the
                dreamer / z8 HNeRV convention).
            The harness auto-detects the convention via ``forward_convention``.
        target_rgb_0: MLX float32 ``(num_pairs, H, W, 3)`` in ``[0, 1]``.
        target_rgb_1: MLX float32 ``(num_pairs, H, W, 3)`` in ``[0, 1]``.
        num_pairs: total trainable pair count.
        forward_convention: ``"reconstruct_pair_nchw01"`` (model returns
            ``(rgb_0, rgb_1)`` NCHW in ``[0, 1]``) or ``"call_b2chw_255"``
            (model returns ``(B, 2, 3, H, W)`` in ``[0, 255]``).
        extra_loss_terms: optional callback ``(model, idx) -> {name: scalar}``
            for the variant's UNIQUE extra terms (residual L2 / commitment /
            MINE). Each scalar is weighted by ``loss_weights[name]`` (default
            weight from ``extra_loss_weights``). The harness adds them to the
            reconstruction + score-aware terms.
        extra_loss_weights: default Lagrangian weights for ``extra_loss_terms``
            keys.
        distillation_weight: weight ``lambda`` on the gradient-reachable
            Hinton-KL T=2.0 score-aware surrogate term. ``0.0`` disables it
            (pure reconstruction). Default ``0.0`` so a substrate opts INTO the
            scorer surrogate explicitly.
        scorer_teacher: the REAL contest-scorer teacher (a
            :class:`ScorerTeacherProvider`). When set AND
            ``distillation_weight > 0`` the distill term BINDS THE REAL SCORER:
            student =
            ``learnable_student_head(decoded_frame_{segnet_teacher_frame_index})``;
            teacher =
            ``stop_gradient(scorer_teacher.teacher_logits_for_indices(idx))``.
            Gradient flows KL -> student_head(decoded) -> renderer params, so
            the renderer is pulled toward frames whose real-SegNet class
            distribution matches the target's — the canonical scorer-binding
            per Catalog #164 + the C6 IBPS / DreamerV3 lesson. This is the
            PRODUCTION path; ``learnable_student_head`` MUST also be set.
        learnable_student_head: the gradient-bearing student head
            (:class:`LearnableConv1x1StudentHead`) mapping decoded RGB
            ``(B, H, W, 3)`` -> class logits ``(B, H, W, num_classes)``. Its
            ~20 params train jointly via the adapter's sibling optimizer. REQUIRED
            when ``scorer_teacher`` is set.
        allow_mock_scorer_teacher: EXPLICIT opt-in to the scorer-BLIND
            deterministic-cosine ``MockTeacherLogitsProvider`` fallback. Default
            ``False`` — the loss FAILS CLOSED when ``distillation_weight > 0``
            and no real ``scorer_teacher`` is wired, so the C6 IBPS scorer-blind
            trap cannot recur silently. Set ``True`` ONLY for a $0 no-real-SegNet
            smoke that explicitly accepts the result is reconstruction-proxy
            (NOT scorer-bound).
        segnet_teacher_frame_index: pair-frame index used by the real SegNet
            distillation term. Default ``1`` matches upstream
            ``SegNet.preprocess_input`` slicing ``x[:, -1, ...]`` from the
            two-frame contest pair. ``0`` is allowed only for deliberate
            frame-0 research probes; the default must stay contest-aligned.
        distillation_temperature: Hinton-KL temperature ``T`` (default 2.0).
        distillation_num_classes: SegNet surrogate class count (default 5).
        pose_distillation_weight: weight ``lambda_pose`` on the gradient-reachable
            POSE-MSE score-aware surrogate term. ``0.0`` disables it. PoseNet is
            DOMINANT at the frontier (per CLAUDE.md "SegNet vs PoseNet
            importance"); a frontier-targeting candidate should bind BOTH the
            SegNet (``distillation_weight``) AND the PoseNet
            (``pose_distillation_weight``) teachers.
        pose_scorer_teacher: the REAL contest-PoseNet teacher (a
            :class:`PoseScorerTeacherProvider`). When set AND
            ``pose_distillation_weight > 0`` the pose term BINDS THE REAL
            POSENET: student = ``learnable_pose_student_head(decoded_0,
            decoded_1)``; teacher =
            ``stop_gradient(pose_scorer_teacher.teacher_pose_for_indices(idx))``.
            Gradient flows pose-MSE -> pose_head(decoded pair) -> renderer params.
            ``learnable_pose_student_head`` MUST also be set.
        learnable_pose_student_head: the gradient-bearing pose head
            (:class:`tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss.LearnablePoseStudentHead`)
            mapping the decoded frame pair -> ``(B, pose_dims)``. REQUIRED when
            ``pose_scorer_teacher`` is set; trains jointly via the adapter's
            sibling optimizer (identical to the SegNet head).
        pose_dims: contest pose dimensionality (default 6 — the first 6 of the
            12-dim PoseNet pose head, matching ``compute_distortion``).
        allow_segnet_only_research: EXPLICIT opt-in to bind ONLY the SegNet
            teacher (``distillation_weight > 0`` with a real ``scorer_teacher``)
            WITHOUT a PoseNet teacher. Default ``False`` — the bundle FAILS
            CLOSED so a SegNet-bound candidate that does NOT also bind PoseNet is
            REFUSED (PoseNet is dominant at the frontier; the SegNet-only
            verification run drifted pose +10.6). Set ``True`` ONLY for
            deliberate SegNet-axis research that explicitly accepts the pose axis
            is unbound.
        export_state_dict_fn: optional ``(model, path) -> None`` PyTorch-export
            bridge; threaded into the adapter's ``export_state_dict``.
        export_archive_fn: optional ``(model, output_dir) -> (path, sha, bytes)``
            numpy-portable archive builder; threaded into the adapter's
            ``export_archive``.
    """

    model: Any
    target_rgb_0: Any
    target_rgb_1: Any
    num_pairs: int
    forward_convention: str = "call_b2chw_255"
    extra_loss_terms: Callable[[Any, Any], Mapping[str, Any]] | None = None
    extra_loss_weights: Mapping[str, float] = field(default_factory=dict)
    distillation_weight: float = 0.0
    scorer_teacher: Any | None = None
    learnable_student_head: Any | None = None
    allow_mock_scorer_teacher: bool = False
    segnet_teacher_frame_index: int = 1
    distillation_temperature: float = 2.0
    distillation_num_classes: int = 5
    pose_distillation_weight: float = 0.0
    pose_scorer_teacher: Any | None = None
    learnable_pose_student_head: Any | None = None
    pose_dims: int = 6
    allow_segnet_only_research: bool = False
    export_state_dict_fn: Callable[[Any, Path], None] | None = None
    export_archive_fn: (
        Callable[[Any, Path], tuple[Path, str, int] | None] | None
    ) = None

    def __post_init__(self) -> None:
        if self.forward_convention not in FORWARD_CONVENTIONS:
            raise MlxScoreAwareHarnessError(
                f"forward_convention must be one of {sorted(FORWARD_CONVENTIONS)}; "
                f"got {self.forward_convention!r}"
            )
        if self.num_pairs < 1:
            raise MlxScoreAwareHarnessError(
                f"num_pairs must be >= 1; got {self.num_pairs}"
            )
        if self.distillation_weight < 0.0:
            raise MlxScoreAwareHarnessError(
                f"distillation_weight must be >= 0 (0.0 disables); got "
                f"{self.distillation_weight}"
            )
        if self.distillation_temperature <= 0.0:
            raise MlxScoreAwareHarnessError(
                f"distillation_temperature must be > 0; got "
                f"{self.distillation_temperature}"
            )
        if self.distillation_num_classes < 1:
            raise MlxScoreAwareHarnessError(
                f"distillation_num_classes must be >= 1; got "
                f"{self.distillation_num_classes}"
            )
        if self.segnet_teacher_frame_index not in (0, 1):
            raise MlxScoreAwareHarnessError(
                "segnet_teacher_frame_index must be 0 or 1; got "
                f"{self.segnet_teacher_frame_index}. Default 1 matches "
                "upstream SegNet.preprocess_input last-frame slicing."
            )
        if self.pose_distillation_weight < 0.0:
            raise MlxScoreAwareHarnessError(
                f"pose_distillation_weight must be >= 0 (0.0 disables); got "
                f"{self.pose_distillation_weight}"
            )
        if self.pose_dims < 1:
            raise MlxScoreAwareHarnessError(
                f"pose_dims must be >= 1; got {self.pose_dims}"
            )
        # C6 IBPS / DreamerV3 scorer-blindness fail-closed (Catalog #164):
        # if a distillation term is active it MUST bind the real scorer via
        # ``scorer_teacher`` + ``learnable_student_head`` UNLESS the caller
        # EXPLICITLY opts into the scorer-blind mock with
        # ``allow_mock_scorer_teacher=True``. This structurally extincts the
        # "decoder reconstructs pixels but SegNet/PoseNet collapse" failure
        # mode that bit C6 IBPS (105.15) + DreamerV3 RSSM (advisory 95.7).
        if self.distillation_weight > 0.0:
            has_real = self.scorer_teacher is not None
            if has_real and self.learnable_student_head is None:
                raise MlxScoreAwareHarnessError(
                    "scorer_teacher is set but learnable_student_head is None; "
                    "the real-scorer-bound distillation requires a "
                    "gradient-bearing student head (per Catalog #164). Build one "
                    "via tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss."
                    "build_learnable_student_head(num_classes=<K>)."
                )
            if not has_real and not self.allow_mock_scorer_teacher:
                raise MlxScoreAwareHarnessError(
                    "distillation_weight > 0 but no real scorer_teacher is wired "
                    "AND allow_mock_scorer_teacher is False. A distillation term "
                    "without a real SegNet/PoseNet teacher is SCORER-BLIND (the "
                    "C6 IBPS / DreamerV3 RSSM failure mode: decoder reconstructs "
                    "pixels but SegNet/PoseNet collapse). Either (a) wire a real "
                    "scorer_teacher + learnable_student_head per Catalog #164, OR "
                    "(b) set allow_mock_scorer_teacher=True to EXPLICITLY accept "
                    "the scorer-blind mock for a $0 no-real-SegNet smoke (the "
                    "result is reconstruction-proxy, NOT scorer-bound)."
                )
        # POSE axis fail-closed (the dominant-at-frontier scorer component): a
        # pose distillation term MUST bind the REAL PoseNet via
        # ``pose_scorer_teacher`` + ``learnable_pose_student_head``. There is no
        # pose mock (pose is a continuous ego-motion vector, not a class
        # distribution, so the SegNet pixel-cosine mock has no pose analogue);
        # a pose distill term without a real teacher is unconditionally refused.
        if self.pose_distillation_weight > 0.0:
            if self.pose_scorer_teacher is None:
                raise MlxScoreAwareHarnessError(
                    "pose_distillation_weight > 0 but no real pose_scorer_teacher "
                    "is wired. Pose distillation requires a REAL PoseNet teacher "
                    "(there is no scorer-blind pose mock — pose is a continuous "
                    "ego-motion vector). Build one via "
                    "tac.substrates._shared.mlx_score_aware.build_mlx_posenet_pair_teacher."
                )
            if self.learnable_pose_student_head is None:
                raise MlxScoreAwareHarnessError(
                    "pose_scorer_teacher is set but learnable_pose_student_head "
                    "is None; the real-pose-bound distillation requires a "
                    "gradient-bearing pose head (per Catalog #164). Build one via "
                    "tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss."
                    "build_learnable_pose_student_head(pose_dims=<D>)."
                )
        # FRONTIER both-scorer invariant: PoseNet is dominant at the ~0.192
        # frontier (CLAUDE.md "SegNet vs PoseNet importance"). A SegNet-bound
        # candidate that does NOT also bind PoseNet is REFUSED unless the caller
        # EXPLICITLY opts into SegNet-only research. This structurally extincts
        # the "bind SegNet, leave pose drifting (+10.6)" half-foundation that the
        # SegNet-only verification run exhibited.
        segnet_bound = (
            self.distillation_weight > 0.0 and self.scorer_teacher is not None
        )
        pose_bound = (
            self.pose_distillation_weight > 0.0
            and self.pose_scorer_teacher is not None
        )
        if segnet_bound and not pose_bound and not self.allow_segnet_only_research:
            raise MlxScoreAwareHarnessError(
                "the bundle binds the REAL SegNet teacher but NOT a PoseNet "
                "teacher. PoseNet is DOMINANT at the contest frontier (per "
                "CLAUDE.md 'SegNet vs PoseNet importance — operating-point "
                "dependent': below pose_avg ~ 2.5e-4 the pose marginal exceeds "
                "SegNet's; the SegNet-only verification run drifted pose +10.6). "
                "Either (a) ALSO wire pose_scorer_teacher + "
                "learnable_pose_student_head + pose_distillation_weight > 0 "
                "(the canonical frontier-binding path), OR (b) set "
                "allow_segnet_only_research=True to EXPLICITLY accept a "
                "SegNet-axis-only research run (the pose axis is unbound)."
            )


__all__ = [
    "FORWARD_CONVENTIONS",
    "MlxRenderer",
    "PoseScorerTeacherProvider",
    "RendererBundle",
    "ScorerTeacherProvider",
]
