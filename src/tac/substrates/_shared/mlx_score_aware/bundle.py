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
from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
    DISTILLATION_OBJECTIVE_KL_T2,
    VALID_DISTILLATION_OBJECTIVES,
)

#: The two canonical renderer forward conventions the harness auto-detects.
FORWARD_CONVENTIONS: frozenset[str] = frozenset(
    {"reconstruct_pair_nchw01", "call_b2chw_255"}
)

#: Canonical normalization modes for the ``recon_pixel_weight`` channel.
#: ``"mean"`` preserves the loss SCALE (convex re-distribution of the same total
#: magnitude); ``"none"`` applies the raw map (caller owns the scale).
_RECON_PIXEL_WEIGHT_NORMALIZE_MODES: frozenset[str] = frozenset({"mean", "none"})
_POSE_STUDENT_INPUT_PREPROCESS_MODES: frozenset[str] = frozenset(
    {"rgb", "pr95_yuv6"}
)
_POSE_DISTILLATION_LOSS_MODES: frozenset[str] = frozenset({"mse", "huber"})

_SUBSTRATE_METADATA_FORBIDDEN_AUTHORITY_KEYS: frozenset[str] = frozenset(
    {
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
        "promotable",
        "score_claim_valid",
    }
)


def _reject_metadata_authority_keys(value: Any, path: str) -> None:
    """Reject score/readiness keys anywhere inside substrate metadata."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise MlxScoreAwareHarnessError(
                    "substrate_artifact_metadata keys must be non-empty str; "
                    f"got {key!r} at {path}"
                )
            child_path = f"{path}.{key}"
            if key in _SUBSTRATE_METADATA_FORBIDDEN_AUTHORITY_KEYS:
                raise MlxScoreAwareHarnessError(
                    "substrate_artifact_metadata cannot carry canonical "
                    f"authority/readiness key {key!r} at {child_path}; use "
                    "the canonical TrainingArtifact fields as the single "
                    "custody surface."
                )
            _reject_metadata_authority_keys(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_metadata_authority_keys(child, f"{path}[{index}]")


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
        segnet_distillation_objective: SegNet scorer-teacher objective. Defaults
            to KL T=2.0; boundary objectives route through the same scorer-bound
            loss and sibling-head optimizer.
        segnet_student_live_calibration_weight: extra sibling-head-only
            calibration weight against the REAL SegNet logits of the decoded
            candidate frame. The cached target teacher supplies the objective
            direction (match target semantics); this live term keeps the tiny
            student head calibrated to the real scorer response on the current
            renderer output so the renderer does not optimize an unfaithful
            surrogate. ``0.0`` disables it for legacy ablations; HiNeRV/SNeRV
            long-run launchers default it on when real SegNet is bound.
        segnet_direct_live_distillation_weight: optional renderer-gradient
            weight against the REAL ported SegNet logits of the decoded
            candidate frame. Unlike ``segnet_student_live_calibration_weight``
            (which updates only the tiny student head), this term backprops
            through ``teacher_logits_for_frames_nhwc01(decoded_frame)`` to the
            renderer pixels and matches the target-frame real SegNet logits.
            It is default-off because it is heavier and remains MLX-local
            false-authority evidence, but it is the scorer-faithful antidote to
            all-one-class SegNet masks when the student surrogate is too weak.
        segnet_direct_live_class_histogram_weight: relative weight for the
            differentiable class-measure tether inside the direct-live SegNet
            loss. The exact upstream ``d_seg`` is an argmax-flip rate, but a
            collapsed renderer can still reduce per-pixel hinge loss by moving
            one dominant class while leaving the global class measure wrong.
            This term matches the candidate soft class histogram to the target
            argmax histogram from the real SegNet cache, making class collapse a
            train-time loss rather than export-only telemetry. Default-off
            because the first bounded HiNeRV probes showed histogram-only
            pressure can preserve soft class mass while stalling hard argmax
            escape.
        segnet_direct_live_class_balanced_hinge_weight: relative weight for a
            bootstrap-only class-balanced Crammer-Singer hinge. It averages the
            per-pixel argmax hinge within each target class and then averages
            across occupied target classes, so minority but score-relevant
            classes cannot be swamped by the dominant road/undrivable mass.
            This is a training escape control, not an evaluation reweighting:
            the exact upstream ``d_seg`` remains uniform over pixels.
        segnet_direct_live_class_balanced_ce_weight: relative weight for a
            class-balanced hard-target cross-entropy over the real SegNet
            target argmax. This is sharper than the hinge in one-class collapse
            basins because crushed target probabilities receive larger
            gradients; it is still exactly scoped to upstream SegNet's
            last-frame argmax decision surface.
        segnet_tau_boundary: boundary-band temperature for boundary-aware SegNet
            objectives.
        segnet_hinge_margin: Crammer-Singer margin buffer for the
            ``boundary_argmax_hinge`` objective.
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
        substrate_artifact_metadata: optional JSON-safe substrate metadata
            threaded into the canonical long-training artifact. This is for
            non-authority facts such as backend lineage, math fidelity class,
            or substrate-local blockers. Canonical readiness/score authority
            fields are refused here so downstream consumers do not grow a
            second stale readiness reader.
        recon_pixel_weight: OPTIONAL canonical per-pixel reconstruction-loss
            weight map (the codex-named ``recon_pixel_weight`` channel). An MLX
            float32 spatial map ``(H, W)`` / ``(H, W, C)`` / ``(1, H, W, C)``,
            a per-pair map ``(N, H, W, C)``, or a per-pair/per-frame map
            ``(N, 2, H, W, C)`` with ``C in {1, 3}``, non-negative, matching
            the decoded frame's ``(H, W)``. When set, the recon MSE is
            re-weighted PER PIXEL by this map BEFORE the spatial mean
            (``mean(w * (rgb - gt)^2) / mean(w)``) so the renderer spends its
            capacity on the pixels the map deems score-relevant. The
            canonical source is the FULL-GRID measured SegNet input-gradient
            saliency ``|∂L_seg/∂pixel|`` from
            :mod:`tac.substrates.uniward_per_pixel_distortion.full_grid_segnet_response_cost_map`
            (sister #1587: CONTEST_RELEVANT at moderate seg degradation), but
            ANY valid non-negative map is accepted (e.g. inverse-S-UNIWARD
            texture for the A/B comparison, or a joint P18/P19 per-pair map).
            ``None`` DISABLES it — the recon term is the canonical UNIFORM
            ``mean((rgb - gt)^2)``, BYTE-IDENTICAL to existing runs
            (Catalog #290 opt-in default-OFF). Static maps are applied to BOTH
            frame_0 and frame_1; ``(N,2,H,W,C)`` maps may protect/spend
            different pixels per pair/frame. The weight is gradient-blocked
            (``mx.stop_gradient``) so only the renderer carries gradient.
        recon_pixel_weight_normalize: how the weight map is normalized before
            re-weighting. ``"mean"`` (canonical default) preserves the loss
            SCALE — ``mean(w * sq) / mean(w)`` so the weighted loss is a
            convex re-distribution of the SAME total magnitude as the uniform
            loss (the recon_weight Lagrangian coefficient stays comparable
            across A/B arms). ``"none"`` applies the raw map (caller owns the
            scale). Default ``"mean"``.
        eval_roundtrip_ste_enabled: opt-in PR95 train/eval surface simulation:
            decoded frames pass through native-MLX bicubic camera resize,
            bilinear return to scorer resolution, and uint8 round/clamp via
            STE BEFORE reconstruction and score-teacher student losses. Default
            ``False`` keeps existing substrates byte-stable; HiNeRV long
            campaigns opt in because PR95's frontier stack trained against this
            exact byte-realized scorer surface.
        eval_roundtrip_camera_hw: camera-resolution ``(H, W)`` for the
            roundtrip. Default ``(874, 1164)`` matches upstream
            ``camera_size``; tests may use smaller positive dimensions.
        pose_student_input_preprocess: input surface for the learnable PoseNet
            student head. ``"rgb"`` preserves legacy behavior. ``"pr95_yuv6"``
            feeds decoded frames through canonical differentiable PR95
            RGB->YUV6 before the pose student, matching the source-faithful
            PoseNet preprocessing lane while retaining finite surrogate
            gradients.
        pose_distillation_loss: pose surrogate loss used for the train-time
            gradient. ``"mse"`` is the exact legacy PoseNet-teacher MSE.
            ``"huber"`` uses an MSE-matched Huber penalty after the canonical
            per-dim scaling: quadratic for small errors and bounded-gradient
            linear outside ``pose_distillation_huber_delta``. This is an
            explicit pose-protected training mode for HiNeRV long runs that
            show catastrophic PoseNet-teacher startup spikes; it still exposes
            the raw MSE telemetry and remains false-authority MLX evidence.
        pose_distillation_huber_delta: positive robust-loss transition point
            used only when ``pose_distillation_loss == "huber"``.
        scorer_input_distribution_guard_weight: optional differentiable guard
            against the observed compact-carrier value-domain collapse where a
            byte-closed receiver emits saturated/out-of-distribution RGB and
            then both SegNet/PoseNet collapse. The guard matches decoded RGB
            per-channel mean/std/dynamic range plus soft saturation mass,
            direct SegNet frame-1 RGB fit, and the PR95 PoseNet YUV6 pair
            mean/std/dynamic range/direct fit plus temporal-delta distribution
            and fit to the real-video targets. It is a train-time Lagrangian
            term, not a score authority claim.
        scorer_input_distribution_guard_saturation_margin: byte-domain edge
            band in normalized RGB units. ``0.02`` means the soft saturation
            term tracks mass near ``<= 0.02`` or ``>= 0.98``.
        scorer_input_distribution_guard_temperature: positive logistic
            temperature for the soft saturation mass. Smaller is sharper.
        scorer_input_contrast_floor_weight: optional train-time hinge against
            scorer-domain contrast collapse on the exact upstream domains:
            SegNet's last-frame RGB tensor and PoseNet's two-frame YUV6 tensor.
            Unlike the distribution guard, this does not pull toward a
            particular image; it only refuses the flat-input basin by requiring
            candidate per-channel std to clear a reference-relative floor.
        scorer_input_contrast_floor_segnet_min_std_ratio: minimum candidate /
            reference std ratio for SegNet's last-frame RGB scorer input.
        scorer_input_contrast_floor_posenet_yuv6_min_std_ratio: minimum
            candidate / reference std ratio for PoseNet's concatenated two-frame
            YUV6 scorer input.
        source_pair_indices: optional local-target-row -> source-video-pair
            mapping. When set, ``num_pairs`` is the hydrated target row count
            and each local row decodes the corresponding source model/latent
            row. Reconstruction and teacher caches remain indexed by local
            rows, while renderer calls use these source pair IDs. This is the
            first-class hard-pair hydration contract for full-video models
            such as HiNeRV, where archive/runtime pair rows stay global.
        train_time_section_byte_metrics: optional callback
            ``(model, idx, loss_weights) -> mapping`` that returns archive or
            section-byte telemetry during training. The callback is the
            MLX-first portable byte-cap bridge: SNeRV and HiNeRV can expose
            receiver-packet/predicted-section bytes without exporting a full
            archive every step, and the shared dual-ascent controller can price
            those bytes against the upstream fixed waterline. Accepted shapes:
            ``{"archive_bytes": int, "section_bytes": {"decoder": int}}`` or
            direct numeric ``section_name -> bytes`` rows. The values are
            telemetry and training pressure only, never score authority.
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
    segnet_distillation_objective: str = DISTILLATION_OBJECTIVE_KL_T2
    segnet_student_live_calibration_weight: float = 0.0
    segnet_direct_live_distillation_weight: float = 0.0
    segnet_direct_live_class_histogram_weight: float = 0.0
    segnet_direct_live_class_balanced_hinge_weight: float = 0.0
    segnet_direct_live_class_balanced_ce_weight: float = 0.0
    segnet_tau_boundary: float = 1.0
    segnet_hinge_margin: float = 1.0
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
    substrate_artifact_metadata: Mapping[str, Any] = field(default_factory=dict)
    recon_pixel_weight: Any | None = None
    recon_pixel_weight_normalize: str = "mean"
    eval_roundtrip_ste_enabled: bool = False
    eval_roundtrip_camera_hw: tuple[int, int] = (874, 1164)
    pose_student_input_preprocess: str = "rgb"
    pose_distillation_loss: str = "mse"
    pose_distillation_huber_delta: float = 1.0
    scorer_input_distribution_guard_weight: float = 0.0
    scorer_input_distribution_guard_saturation_margin: float = 0.02
    scorer_input_distribution_guard_temperature: float = 0.01
    scorer_input_contrast_floor_weight: float = 0.0
    scorer_input_contrast_floor_segnet_min_std_ratio: float = 0.5
    scorer_input_contrast_floor_posenet_yuv6_min_std_ratio: float = 0.5
    source_pair_indices: tuple[int, ...] | None = None
    train_time_section_byte_metrics: (
        Callable[[Any, Any, Mapping[str, float]], Mapping[str, Any]] | None
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
        if self.source_pair_indices is not None:
            normalized_source_indices: list[int] = []
            seen_source_indices: set[int] = set()
            try:
                raw_source_pair_indices = tuple(self.source_pair_indices)
            except TypeError as exc:
                raise MlxScoreAwareHarnessError(
                    "source_pair_indices must be a finite sequence of integer "
                    "source video pair ids"
                ) from exc
            if len(raw_source_pair_indices) != int(self.num_pairs):
                raise MlxScoreAwareHarnessError(
                    "source_pair_indices length must equal num_pairs so every "
                    "hydrated target row has exactly one source model row; got "
                    f"{len(raw_source_pair_indices)} indices for num_pairs="
                    f"{self.num_pairs}"
                )
            for raw in raw_source_pair_indices:
                try:
                    value = int(raw)
                except (TypeError, ValueError) as exc:
                    raise MlxScoreAwareHarnessError(
                        "source_pair_indices must contain integer source pair ids"
                    ) from exc
                if value < 0:
                    raise MlxScoreAwareHarnessError(
                        f"source_pair_indices must be non-negative; got {value}"
                    )
                if value in seen_source_indices:
                    raise MlxScoreAwareHarnessError(
                        "source_pair_indices must not contain duplicates; "
                        f"duplicate source pair id {value}"
                    )
                seen_source_indices.add(value)
                normalized_source_indices.append(value)
            self.source_pair_indices = tuple(normalized_source_indices)
        if not isinstance(self.substrate_artifact_metadata, Mapping):
            raise MlxScoreAwareHarnessError(
                "substrate_artifact_metadata must be a Mapping; got "
                f"{type(self.substrate_artifact_metadata).__name__}"
            )
        _reject_metadata_authority_keys(
            self.substrate_artifact_metadata,
            "substrate_artifact_metadata",
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
        if self.segnet_distillation_objective not in VALID_DISTILLATION_OBJECTIVES:
            raise MlxScoreAwareHarnessError(
                "segnet_distillation_objective must be one of "
                f"{VALID_DISTILLATION_OBJECTIVES!r}; got "
                f"{self.segnet_distillation_objective!r}"
            )
        if self.segnet_student_live_calibration_weight < 0.0:
            raise MlxScoreAwareHarnessError(
                "segnet_student_live_calibration_weight must be >= 0; got "
                f"{self.segnet_student_live_calibration_weight}"
            )
        if self.segnet_direct_live_distillation_weight < 0.0:
            raise MlxScoreAwareHarnessError(
                "segnet_direct_live_distillation_weight must be >= 0; got "
                f"{self.segnet_direct_live_distillation_weight}"
            )
        if self.segnet_direct_live_class_histogram_weight < 0.0:
            raise MlxScoreAwareHarnessError(
                "segnet_direct_live_class_histogram_weight must be >= 0; got "
                f"{self.segnet_direct_live_class_histogram_weight}"
            )
        if self.segnet_direct_live_class_balanced_hinge_weight < 0.0:
            raise MlxScoreAwareHarnessError(
                "segnet_direct_live_class_balanced_hinge_weight must be >= 0; got "
                f"{self.segnet_direct_live_class_balanced_hinge_weight}"
            )
        if self.segnet_direct_live_class_balanced_ce_weight < 0.0:
            raise MlxScoreAwareHarnessError(
                "segnet_direct_live_class_balanced_ce_weight must be >= 0; got "
                f"{self.segnet_direct_live_class_balanced_ce_weight}"
            )
        if self.segnet_tau_boundary <= 0.0:
            raise MlxScoreAwareHarnessError(
                f"segnet_tau_boundary must be > 0; got {self.segnet_tau_boundary}"
            )
        if self.segnet_hinge_margin <= 0.0:
            raise MlxScoreAwareHarnessError(
                f"segnet_hinge_margin must be > 0; got {self.segnet_hinge_margin}"
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
        if self.recon_pixel_weight_normalize not in _RECON_PIXEL_WEIGHT_NORMALIZE_MODES:
            raise MlxScoreAwareHarnessError(
                "recon_pixel_weight_normalize must be one of "
                f"{sorted(_RECON_PIXEL_WEIGHT_NORMALIZE_MODES)}; got "
                f"{self.recon_pixel_weight_normalize!r}"
            )
        if self.pose_student_input_preprocess not in _POSE_STUDENT_INPUT_PREPROCESS_MODES:
            raise MlxScoreAwareHarnessError(
                "pose_student_input_preprocess must be one of "
                f"{sorted(_POSE_STUDENT_INPUT_PREPROCESS_MODES)}; got "
                f"{self.pose_student_input_preprocess!r}"
            )
        if self.pose_distillation_loss not in _POSE_DISTILLATION_LOSS_MODES:
            raise MlxScoreAwareHarnessError(
                "pose_distillation_loss must be one of "
                f"{sorted(_POSE_DISTILLATION_LOSS_MODES)}; got "
                f"{self.pose_distillation_loss!r}"
            )
        if self.pose_distillation_huber_delta <= 0.0:
            raise MlxScoreAwareHarnessError(
                "pose_distillation_huber_delta must be > 0; got "
                f"{self.pose_distillation_huber_delta}"
            )
        if self.scorer_input_distribution_guard_weight < 0.0:
            raise MlxScoreAwareHarnessError(
                "scorer_input_distribution_guard_weight must be >= 0; got "
                f"{self.scorer_input_distribution_guard_weight}"
            )
        if (
            self.train_time_section_byte_metrics is not None
            and not callable(self.train_time_section_byte_metrics)
        ):
            raise MlxScoreAwareHarnessError(
                "train_time_section_byte_metrics must be callable when set; got "
                f"{type(self.train_time_section_byte_metrics).__name__}"
            )
        if not (0.0 < self.scorer_input_distribution_guard_saturation_margin < 0.5):
            raise MlxScoreAwareHarnessError(
                "scorer_input_distribution_guard_saturation_margin must be in "
                "(0, 0.5); got "
                f"{self.scorer_input_distribution_guard_saturation_margin}"
            )
        if self.scorer_input_distribution_guard_temperature <= 0.0:
            raise MlxScoreAwareHarnessError(
                "scorer_input_distribution_guard_temperature must be > 0; got "
                f"{self.scorer_input_distribution_guard_temperature}"
            )
        if self.scorer_input_contrast_floor_weight < 0.0:
            raise MlxScoreAwareHarnessError(
                "scorer_input_contrast_floor_weight must be >= 0; got "
                f"{self.scorer_input_contrast_floor_weight}"
            )
        if self.scorer_input_contrast_floor_segnet_min_std_ratio <= 0.0:
            raise MlxScoreAwareHarnessError(
                "scorer_input_contrast_floor_segnet_min_std_ratio must be > 0; got "
                f"{self.scorer_input_contrast_floor_segnet_min_std_ratio}"
            )
        if self.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio <= 0.0:
            raise MlxScoreAwareHarnessError(
                "scorer_input_contrast_floor_posenet_yuv6_min_std_ratio must be > 0; got "
                f"{self.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio}"
            )
        try:
            cam_h, cam_w = self.eval_roundtrip_camera_hw
        except (TypeError, ValueError) as exc:
            raise MlxScoreAwareHarnessError(
                "eval_roundtrip_camera_hw must be a 2-tuple (H, W); got "
                f"{self.eval_roundtrip_camera_hw!r}"
            ) from exc
        if int(cam_h) <= 0 or int(cam_w) <= 0:
            raise MlxScoreAwareHarnessError(
                "eval_roundtrip_camera_hw entries must be positive; got "
                f"{self.eval_roundtrip_camera_hw!r}"
            )
        self.eval_roundtrip_camera_hw = (int(cam_h), int(cam_w))
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
        live_segnet_candidate_frame_terms_active = bool(
            (
                self.distillation_weight > 0.0
                and self.segnet_student_live_calibration_weight > 0.0
            )
            or self.segnet_direct_live_distillation_weight > 0.0
        )
        if live_segnet_candidate_frame_terms_active:
            has_real = self.scorer_teacher is not None
            live_fn = getattr(
                self.scorer_teacher,
                "teacher_logits_for_frames_nhwc01",
                None,
            )
            has_empty_live_adapter = (
                hasattr(self.scorer_teacher, "live_segnet_adapter")
                and self.scorer_teacher.live_segnet_adapter is None
            )
            if not has_real or not callable(live_fn) or has_empty_live_adapter:
                raise MlxScoreAwareHarnessError(
                    "live SegNet candidate-frame terms require "
                    "a real SegNet teacher that can evaluate decoded "
                    "candidate frames via teacher_logits_for_frames_nhwc01. "
                    "Use build_mlx_segnet_pair_teacher or set the "
                    "live calibration/direct weights to 0 for a legacy ablation."
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
