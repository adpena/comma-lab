# SPDX-License-Identifier: MIT
"""Gradient-reachable MLX score-aware Lagrangian (separation of concerns).

This module owns ONLY the loss math: reconstruction MSE + the optional
gradient-reachable Hinton-distilled KL T=2.0 scorer surrogate + optional
substrate-specific extra terms. It is substrate-AGNOSTIC: the renderer forward
convention is decoded via :func:`decode_frames_nhwc01` so the loss never
assumes a fixed model signature.

The score-aware term is the canonical Hinton-distilled surrogate per
CLAUDE.md "eval_roundtrip -- NON-NEGOTIABLE" + Catalog #164 sister discipline:
the production teacher is the real MLX SegNet logits cache on the contest
SegNet frame (default pair frame 1, matching upstream ``x[:, -1, ...]``), the
student is a learnable head on the decoded frame, and gradient flows KL ->
decoded -> renderer params. The explicit mock path is allowed only for
scorer-blind smoke tests.

[verified-against: tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss.hinton_distilled_kl_t2_loss canonical scorer surrogate]
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from tac.substrates._shared.mlx_score_aware.device_gate import (
    require_mlx_for_harness,
)

if TYPE_CHECKING:
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle


def decode_frames_nhwc01(bundle: RendererBundle, idx: Any) -> tuple[Any, Any]:
    """Decode ``(rgb_0, rgb_1)`` as NHWC ``[0, 1]`` regardless of model convention.

    Returns two MLX float32 arrays each ``(B, H, W, 3)`` in ``[0, 1]``, ready
    for MSE against the canonical NHWC ``[0, 1]`` targets.

    Args:
        bundle: the substrate RendererBundle.
        idx: MLX int32 ``(B,)`` pair-index batch.

    Returns:
        ``(rgb_0, rgb_1)`` NHWC float32 each ``(B, H, W, 3)`` in ``[0, 1]``.
    """
    mx = require_mlx_for_harness()
    model = bundle.model
    if bundle.forward_convention == "reconstruct_pair_nchw01":
        result = model.reconstruct_pair(idx)
        # The renderer may return (rgb_0, rgb_1) or (rgb_0, rgb_1, z); take the
        # first two. Each is (B, 3, H, W) in [0, 1].
        rgb_0 = result[0]
        rgb_1 = result[1]
        rgb_0 = mx.transpose(rgb_0, (0, 2, 3, 1))
        rgb_1 = mx.transpose(rgb_1, (0, 2, 3, 1))
        return rgb_0, rgb_1
    # call_b2chw_255: model(idx) -> (B, 2, 3, H, W) in [0, 255].
    pair = model(idx)
    pair01 = pair / 255.0
    rgb_0 = mx.transpose(pair01[:, 0], (0, 2, 3, 1))
    rgb_1 = mx.transpose(pair01[:, 1], (0, 2, 3, 1))
    return rgb_0, rgb_1


def score_aware_loss(
    bundle: RendererBundle,
    idx: Any,
    *,
    recon_weight: float = 1.0,
    loss_weights: Mapping[str, float] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compute the gradient-reachable MLX score-aware Lagrangian.

    The combined loss is::

        L = recon_weight * (mse(rgb_0, gt_0) + mse(rgb_1, gt_1))
            + distillation_weight * T**2 * KL(student || teacher)
            + sum_k extra_weight[k] * extra_term_k

    The reconstruction MSE is over the canonical NHWC ``[0, 1]`` frames. The
    optional score-aware term is the canonical Hinton-distilled KL T=2.0
    surrogate (gradient-reachable from KL -> decoded frame -> renderer params)
    per CLAUDE.md "eval_roundtrip" + Catalog #164 sister discipline.

    Args:
        bundle: the substrate RendererBundle.
        idx: MLX int32 ``(B,)`` pair-index batch.
        recon_weight: Lagrangian weight on the reconstruction MSE term.
        loss_weights: optional per-name overrides for the extra-loss terms.

    Returns:
        ``(total_loss_scalar, parts_dict)`` where ``parts_dict`` has scalar
        component values for telemetry (``total`` / ``recon`` / ``distill`` /
        per-extra).
    """
    mx = require_mlx_for_harness()
    weights = dict(bundle.extra_loss_weights)
    if loss_weights:
        weights.update({k: float(v) for k, v in loss_weights.items()})

    rgb_0, rgb_1 = decode_frames_nhwc01(bundle, idx)
    gt_0 = bundle.target_rgb_0[idx]
    gt_1 = bundle.target_rgb_1[idx]
    mse_0 = mx.mean((rgb_0 - gt_0) ** 2)
    mse_1 = mx.mean((rgb_1 - gt_1) ** 2)
    recon = mse_0 + mse_1
    total = recon_weight * recon
    parts: dict[str, Any] = {"recon": recon}

    if bundle.distillation_weight > 0.0:
        from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
            hinton_distilled_kl_t2_loss,
        )

        if bundle.scorer_teacher is not None:
            # PRODUCTION path (Catalog #164 + C6 IBPS / DreamerV3 lesson): the
            # distill term BINDS THE REAL SCORER. The student is the learnable
            # 1x1-conv head on the DECODED contest SegNet frame
            # (gradient-bearing: KL -> head(decoded) -> renderer params); the
            # teacher is the REAL contest SegNet's per-pixel class distribution
            # on this pair's TARGET SegNet frame (gradient-blocked). Backprop
            # through the FULL ported SegNet would
            # be ideal but produces NaN gradients in MLX's second-order autograd
            # composition with the renderer's PixelShuffle/bilinear backward;
            # the learnable-head-distilled-from-real-SegNet-teacher surrogate
            # gives a FINITE, genuinely scorer-bound gradient (the head learns
            # decoded-RGB -> real-SegNet-class-logits, so the renderer gradient
            # is pulled toward what the real scorer rewards, NOT toward a fixed
            # cosine of pixel means).
            head = bundle.learnable_student_head
            if head is None:  # defensive; bundle.__post_init__ already enforces.
                raise ValueError(
                    "scorer_teacher set without learnable_student_head; "
                    "RendererBundle.__post_init__ should have rejected this."
                )
            seg_rgb = rgb_1 if bundle.segnet_teacher_frame_index == 1 else rgb_0
            student_logits = head(seg_rgb)
            teacher_logits = mx.stop_gradient(
                bundle.scorer_teacher.teacher_logits_for_indices(idx)
            )
            distill = hinton_distilled_kl_t2_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                temperature=bundle.distillation_temperature,
            )
        else:
            # SCORER-BLIND mock fallback — reachable ONLY when
            # ``allow_mock_scorer_teacher=True`` (bundle.__post_init__ fails
            # closed otherwise). The MockTeacherLogitsProvider is a fixed cosine
            # of RGB pixel means with NO SegNet weights; the distill gradient is
            # ~parallel to the recon gradient (scorer-blind). Kept for $0
            # no-real-SegNet smokes that explicitly accept reconstruction-proxy.
            from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
                MockTeacherLogitsProvider,
            )

            provider = MockTeacherLogitsProvider(
                num_classes=bundle.distillation_num_classes,
            )
            student_logits = provider.teacher_logits(rgb_0)
            teacher_logits = mx.stop_gradient(provider.teacher_logits(gt_0))
            distill = hinton_distilled_kl_t2_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                temperature=bundle.distillation_temperature,
            )
        total = total + bundle.distillation_weight * distill
        parts["distill"] = distill

    if bundle.extra_loss_terms is not None:
        extra = bundle.extra_loss_terms(bundle.model, idx)
        for name, term in extra.items():
            w = float(weights.get(name, 1.0))
            total = total + w * term
            parts[name] = term

    parts["total"] = total
    return total, parts


def build_mlx_segnet_pair_teacher(
    bundle: RendererBundle,
    *,
    upstream_dir: Any = "upstream",
    device: str = "cpu",
) -> Any:
    """Build a real-MLX-SegNet per-pair teacher cache for the harness.

    The canonical scorer-bound teacher per Catalog #164 + the C6 IBPS /
    DreamerV3 RSSM scorer-blindness lesson. Loads the real upstream PyTorch
    SegNet, ports it to MLX (pure-MLX op graph), runs ONE gradient-free SegNet
    forward per pair's TARGET SegNet frame, and caches the per-pixel class
    logits indexed by PAIR index. The cache satisfies the
    :class:`tac.substrates._shared.mlx_score_aware.bundle.ScorerTeacherProvider`
    protocol (``num_classes`` + ``teacher_logits_for_indices``) so it threads
    directly into ``RendererBundle.scorer_teacher``.

    This is the teacher target the learnable student head is distilled toward;
    the renderer gradient then flows KL -> head(decoded) -> renderer, binding
    the renderer to the REAL SegNet's class boundaries (NOT a pixel-cosine).

    NOTE on resolution: the SegNet logits are at SegNet's canonical
    ``(384, 512)`` output. The learnable student head preserves the decoded
    frame's spatial dims, so the bundle's targets MUST be ``(384, 512)`` for
    the student/teacher shapes to align (the canonical contest eval size).

    Args:
        bundle: the harness RendererBundle. Its
            ``segnet_teacher_frame_index`` selects which target frame supplies
            teacher logits; default ``1`` matches upstream SegNet last-frame
            slicing. Targets MUST be NHWC ``[0, 1]`` at SegNet size.
        upstream_dir: path to the upstream repo (contains the SegNet weights).
        device: PyTorch device for the SegNet weight load + MLX port (``cpu``
            per CLAUDE.md "MPS auth eval is NOISE" — no MPS for the teacher).

    Returns:
        a :class:`RealSegNetTeacherLogitsCache` keyed by PAIR index (so its
        ``teacher_logits_for_indices(idx)`` aligns with the harness batch).

    Raises:
        MlxScoreAwareHarnessError: targets are not at SegNet's ``(384, 512)``.
    """
    import numpy as np

    from tac.local_acceleration.mlx_scorer_adapters import MLXSegNetAdapter
    from tac.scorer import load_default_scorers
    from tac.substrates._shared.mlx_score_aware.device_gate import (
        MlxScoreAwareHarnessError,
    )
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        RealSegNetTeacherLogitsCache,
    )

    mx = require_mlx_for_harness()
    tgt = (
        bundle.target_rgb_1
        if bundle.segnet_teacher_frame_index == 1
        else bundle.target_rgb_0
    )
    n_pairs, h, w, _c = tgt.shape
    if (h, w) != (384, 512):
        raise MlxScoreAwareHarnessError(
            f"build_mlx_segnet_pair_teacher requires targets at SegNet size "
            f"(384, 512) for student/teacher shape alignment; got ({h}, {w}). "
            "Decode the harness targets at the canonical contest eval size."
        )
    _posenet, segnet = load_default_scorers(str(upstream_dir), device=device)
    segnet.eval()
    mlx_segnet = MLXSegNetAdapter(segnet)
    # One gradient-free SegNet forward per pair target SegNet frame, chunked to
    # keep memory bounded. SegNet preprocess expects RGB in 0..255 (no internal
    # /255 per the upstream cache builder convention), so scale the [0,1]
    # target up.
    chunk = 16
    logits_chunks = []
    for start in range(0, n_pairs, chunk):
        end = min(start + chunk, n_pairs)
        x = tgt[start:end] * 255.0  # (b, 384, 512, 3) MLX
        out = mx.stop_gradient(mlx_segnet(x))  # (b, 384, 512, K) MLX
        mx.eval(out)
        logits_chunks.append(np.array(out).astype(np.float32))
    logits_np = np.concatenate(logits_chunks, axis=0)  # (n_pairs, 384, 512, K)
    return RealSegNetTeacherLogitsCache(
        teacher_logits_thwk=mx.array(logits_np),
        frame_count=int(logits_np.shape[0]),
        height=int(logits_np.shape[1]),
        width=int(logits_np.shape[2]),
        num_classes=int(logits_np.shape[3]),
    )


__all__ = [
    "build_mlx_segnet_pair_teacher",
    "decode_frames_nhwc01",
    "score_aware_loss",
]
