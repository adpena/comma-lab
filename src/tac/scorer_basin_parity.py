# SPDX-License-Identifier: MIT
"""Scorer basin parity gate - local readiness evidence for apogee_intN candidates.

Implements one of three accepted predispatch evidence types in
``tools/predispatch_sanity.py`` Gate 6 (apogee_evidence_semantics):

  - ``contest_cuda_exact_eval_positive``    (only after a real dispatch)
  - ``contest_faithful_distortion_model``   (extensive multi-anchor regression)
  - ``scorer_basin_parity_gate``            (THIS - the cheapest evidence path)

A basin-parity probe asks: do the QUANTIZED weights remain in the same
loss basin as the LOSSLESS reference? If yes, the scorer's output on the
quantized reconstruction stays close to the scorer's output on the lossless
reconstruction. If no - the basin has shifted (e.g. apogee_int4's 700x pose
collapse) - the candidate must not be dispatched.

The probe is composed of two evidence axes:

  1. **Distortion-delta axis** - actual scorer (PoseNet + SegNet) called on
     reconstructed frames from BOTH weight points, scored against the same
     ground-truth frames. ``pose_dist_delta`` and ``seg_dist_delta`` are the
     primary fail criteria.

  2. **Hessian-trace axis** - Hutchinson trace estimator of d(scorer_loss)
     /d(weight) at both points. A blown-up trace ratio means the local
     curvature at the quantized point is dramatically different from the
     lossless point, which is direct evidence of a basin shift.

Both axes must pass for the gate to pass. CPU-only is acceptable because the
gate is a basin-geometry probe, not a contest-CUDA score claim. Per CLAUDE.md
per-tag discipline, evidence emitted by this module MUST be tagged
``[scorer-basin-parity:CPU]`` (or ``[scorer-basin-parity:CUDA]`` if
``--device cuda`` is used) - NEVER ``[contest-CUDA]``.

Strict-scorer rule note: this is a COMPRESS-time / pre-dispatch readiness tool,
not an inflate-time path. Loading scorer state-dicts here is permitted; the
inflate path of every shipped archive must remain free of scorer load.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Default thresholds (tunable; rationale documented next to each constant)
# ---------------------------------------------------------------------------

# PR106 anchor pose_avg = 3.4e-5; int4 (FALSIFIED at 1.43) had pose 2.37e-2
# (~700x collapse). int8 (near-lossless) had pose 3.375e-5. We set the
# pose-delta threshold at 1.0e-3 - about 30x PR106 - so any candidate whose
# basin shift produces apogee_int4-style pose collapse will fail, while
# near-lossless candidates (int8, int7, int6 mild) pass through.
DEFAULT_POSE_DIST_DELTA_THRESHOLD = 1.0e-3

# PR106 anchor seg_avg = 6.78e-4; int4 had seg 8.69e-3 (~13x). Threshold
# 5.0e-3 admits about 7x growth - easily passing PR106-class candidates and
# strictly rejecting int4-style basin collapse.
DEFAULT_SEG_DIST_DELTA_THRESHOLD = 5.0e-3

# Hessian-trace ratio. We compute log10(trace_quantized / trace_lossless) and
# require its absolute value to stay below 1.0, i.e. the curvature scale at the
# quantized point is within 10x of the lossless point. A 100x curvature
# explosion is a textbook basin shift.
DEFAULT_HESSIAN_LOG_RATIO_TOLERANCE = 1.0

# Absolute ceilings prevent a "bad anchor versus bad candidate" comparison
# from passing just because both reconstructions live in the same bad basin.
DEFAULT_ABSOLUTE_POSE_CEILING = 1.0e-2
DEFAULT_ABSOLUTE_SEG_CEILING = 2.0e-2


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParityReport:
    """Structured output of :func:`compute_scorer_basin_parity`."""

    pose_dist_lossless: float
    pose_dist_quantized: float
    pose_dist_delta: float
    seg_dist_lossless: float
    seg_dist_quantized: float
    seg_dist_delta: float

    hessian_trace_lossless: float
    hessian_trace_quantized: float
    hessian_log_ratio: float  # log10(trace_quantized / trace_lossless)

    basin_parity_passed: bool
    pose_threshold: float
    seg_threshold: float
    hessian_log_ratio_tolerance: float

    absolute_pose_ceiling: float
    absolute_seg_ceiling: float

    n_probes: int
    n_hessian_samples: int
    anchor_frame_shas: tuple[str, ...]
    device: str
    computed_utc: str

    failure_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Hessian trace estimator (Hutchinson)
# ---------------------------------------------------------------------------


def hutchinson_trace_estimate(
    loss_fn: Callable[[], torch.Tensor],
    parameters: Sequence[torch.Tensor],
    *,
    n_samples: int = 4,
    generator: torch.Generator | None = None,
) -> float:
    """Estimate Tr(H) where H is the Hessian of ``loss_fn`` w.r.t. ``parameters``.

    Uses Hutchinson's trick: for Rademacher vectors v_i with E[v_i v_i^T] = I,
    Tr(H) = E[v^T H v]. We compute v^T H v as a Hessian-vector product via two
    backwards (one for grad, one for grad-of-grad-dot-v).

    The function is dependency-light and works on CPU.
    """

    if not parameters:
        return 0.0

    sample_traces: list[float] = []
    for _ in range(n_samples):
        # Sample Rademacher (+/-1) vectors
        vs = [
            (
                torch.randint(
                    0,
                    2,
                    p.shape,
                    generator=generator,
                    dtype=torch.int8,
                    device=p.device,
                )
                * 2
                - 1
            ).to(
                dtype=p.dtype,
            )
            for p in parameters
        ]

        # First backward: compute grad
        for p in parameters:
            if p.grad is not None:
                p.grad = None
        loss = loss_fn()
        grads = torch.autograd.grad(loss, parameters, create_graph=True, allow_unused=True)
        # Replace any None grads with zeros (parameters not on the graph this pass)
        grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, parameters, strict=True)
        ]

        # Form g . v (scalar)
        gv = torch.zeros((), device=parameters[0].device, dtype=parameters[0].dtype)
        for g, v in zip(grads, vs, strict=True):
            gv = gv + (g * v).sum()

        # Second backward: H v has structure d(g.v)/d(theta) = H v
        hv_list = torch.autograd.grad(gv, parameters, retain_graph=False, allow_unused=True)
        hv_list = [
            h if h is not None else torch.zeros_like(p)
            for h, p in zip(hv_list, parameters, strict=True)
        ]

        # v^T H v = sum_i v_i . (Hv)_i
        vhv = 0.0
        for h, v in zip(hv_list, vs, strict=True):
            vhv += float((h * v).sum().item())
        sample_traces.append(vhv)

    return float(sum(sample_traces) / max(1, len(sample_traces)))


# ---------------------------------------------------------------------------
# Distortion / parity computation
# ---------------------------------------------------------------------------


def _build_pose_loss(
    pose_out_quantized: dict[str, torch.Tensor],
    pose_out_target: dict[str, torch.Tensor],
) -> torch.Tensor:
    """MSE over first 6 pose dims (matches PoseNet.compute_distortion semantics)."""

    parts = []
    for head_name, target in pose_out_target.items():
        out_q = pose_out_quantized[head_name]
        # PoseNet.compute_distortion uses ``[..., : h.out // 2]`` and out=12 so [:,:6]
        last_dim = target.shape[-1]
        half = last_dim // 2
        diff = (out_q[..., :half] - target[..., :half]).pow(2)
        parts.append(diff.mean())
    return torch.stack(parts).sum()


def _build_seg_loss(
    seg_logits_quantized: torch.Tensor,
    seg_logits_target: torch.Tensor,
) -> torch.Tensor:
    """Soft proxy of SegNet argmax-disagreement: KL on softmax outputs.

    The hard argmax is non-differentiable, but for Hessian-curvature probing
    we only need a smooth surrogate of the local landscape. KL(target ||
    quantized) is the canonical smooth measure of agreement on logits and
    gives a meaningful curvature signal even if the argmax labels match.
    """

    log_q = torch.log_softmax(seg_logits_quantized, dim=1)
    p_t = torch.softmax(seg_logits_target, dim=1)
    return torch.nn.functional.kl_div(log_q, p_t, reduction="none").sum(dim=1).mean()


def _run_scorer(
    posenet: nn.Module,
    segnet: nn.Module,
    pred_frames: torch.Tensor,
    gt_frames: torch.Tensor,
) -> tuple[float, float]:
    """Run scorer on (pred, gt) and return (pose_dist, seg_dist).

    Args:
        posenet, segnet: scorer modules (already loaded + .eval()).
        pred_frames, gt_frames: (B, T=2, H=874, W=1164, 3) uint8.

    Returns scalar means matching upstream evaluate.py's compute_distortion.
    """

    # Replicate DistortionNet.preprocess_input
    import einops

    def _preprocess(x_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # B, T, H, W, C -> B, T, C, H, W (float)
        x = einops.rearrange(x_uint8, "b t h w c -> b t c h w").float()
        return posenet.preprocess_input(x), segnet.preprocess_input(x)

    with torch.inference_mode():
        pose_in_p, seg_in_p = _preprocess(pred_frames)
        pose_in_g, seg_in_g = _preprocess(gt_frames)

        pose_out_p = posenet(pose_in_p)
        pose_out_g = posenet(pose_in_g)
        seg_out_p = segnet(seg_in_p)
        seg_out_g = segnet(seg_in_g)

        pose_dist = posenet.compute_distortion(pose_out_p, pose_out_g)
        seg_dist = segnet.compute_distortion(seg_out_p, seg_out_g)

    return float(pose_dist.mean().item()), float(seg_dist.mean().item())


def _hessian_probe(
    posenet: nn.Module,
    segnet: nn.Module,
    decoder: nn.Module,
    decoder_state: dict[str, torch.Tensor],
    latents: torch.Tensor,
    gt_frames: torch.Tensor,
    *,
    n_samples: int,
    seed: int,
    n_probes_for_hessian: int = 2,
) -> float:
    """Compute Hutchinson trace of d(seg KL + pose MSE)/d(decoder_weights).

    The decoder is treated as the trainable surface; its weights are
    perturbed (in the Hessian sense) and we measure the curvature of the
    scorer-on-reconstruction loss. Latents are held FIXED (frozen) - this
    isolates the basin geometry around the weight point.

    Uses only the first ``n_probes_for_hessian`` latent pairs because Hessian
    trace estimation is O(n_pairs * n_samples * 2 backwards) and this is a
    CPU-only probe.
    """

    decoder = decoder.train()  # need grads
    decoder.load_state_dict(decoder_state)
    for p in decoder.parameters():
        p.requires_grad = True

    # Use a small number of latent pairs for the Hessian probe (curvature
    # estimate is robust and we want CPU runtime under a few minutes).
    z = latents[:n_probes_for_hessian].detach()
    gt = gt_frames[:n_probes_for_hessian]

    import einops
    import torch.nn.functional as F

    cam_h, cam_w = int(gt_frames.shape[2]), int(gt_frames.shape[3])

    def _loss_fn() -> torch.Tensor:
        # Reconstruct frames from decoder
        decoded = decoder(z)  # (B, 2, 3, eval_h, eval_w)
        B = decoded.shape[0]
        eval_h, eval_w = decoded.shape[-2], decoded.shape[-1]
        flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
        up = F.interpolate(
            flat, size=(cam_h, cam_w), mode="bicubic", align_corners=False
        )
        # Reshape to (B, T=2, H, W, C) uint8-equivalent (don't .round() - we
        # need a smooth surface for Hessian)
        recon = up.clamp(0, 255).reshape(B, 2, 3, cam_h, cam_w)
        recon = einops.rearrange(recon, "b t c h w -> b t h w c")

        gt_local = gt.float()
        # Pose and seg preprocessing
        pose_in_r, seg_in_r = (
            posenet.preprocess_input(einops.rearrange(recon, "b t h w c -> b t c h w")),
            segnet.preprocess_input(einops.rearrange(recon, "b t h w c -> b t c h w")),
        )
        with torch.no_grad():
            pose_in_g, seg_in_g = (
                posenet.preprocess_input(
                    einops.rearrange(gt_local, "b t h w c -> b t c h w")
                ),
                segnet.preprocess_input(
                    einops.rearrange(gt_local, "b t h w c -> b t c h w")
                ),
            )
            pose_out_g = posenet(pose_in_g)
            seg_out_g = segnet(seg_in_g)

        pose_out_r = posenet(pose_in_r)
        seg_out_r = segnet(seg_in_r)

        pose_term = _build_pose_loss(pose_out_r, pose_out_g)
        seg_term = _build_seg_loss(seg_out_r, seg_out_g)
        return pose_term + seg_term

    # Pick a small subset of decoder params for the Hessian probe to keep
    # CPU runtime bounded. We pick the rgb_0 + rgb_1 + refine layers - the
    # last few that touch the output and are most sensitive to quantization.
    target_layer_prefixes = ("rgb_0", "rgb_1", "refine")
    params = [
        p
        for n, p in decoder.named_parameters()
        if any(n.startswith(prefix) for prefix in target_layer_prefixes)
    ]
    if not params:
        # Fallback: take first 2 parameters
        params = [p for _, p in list(decoder.named_parameters())[:2]]

    gen = torch.Generator(device=latents.device)
    gen.manual_seed(seed)
    trace = hutchinson_trace_estimate(
        _loss_fn, params, n_samples=n_samples, generator=gen
    )
    return abs(trace)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reconstruct_frames(
    decoder: nn.Module,
    decoder_state: dict[str, torch.Tensor],
    latents: torch.Tensor,
    *,
    eval_size: tuple[int, int] | None = None,
    camera_size: tuple[int, int] = (874, 1164),
    n_pairs: int | None = None,
) -> torch.Tensor:
    """Run decoder forward at ``eval_size``, bicubic up to ``camera_size``.

    If ``eval_size`` is None, the decoder must expose ``.eval_size``.
    Returns ``(N, 2, camera_size[0], camera_size[1], 3)`` uint8 tensor - the
    same shape as upstream's ground-truth batches before einops rearrange.
    """

    import torch.nn.functional as F

    decoder.load_state_dict(decoder_state)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    if n_pairs is None:
        n_pairs = latents.shape[0]
    n_pairs = min(n_pairs, latents.shape[0])

    out_frames = []
    if eval_size is None:
        eval_size = tuple(decoder.eval_size)  # type: ignore[attr-defined]
    eval_h, eval_w = eval_size
    cam_h, cam_w = camera_size

    with torch.inference_mode():
        for i in range(0, n_pairs, 8):
            j = min(i + 8, n_pairs)
            decoded = decoder(latents[i:j])  # (B, 2, 3, eval_h, eval_w)
            B = decoded.shape[0]
            flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(cam_h, cam_w), mode="bicubic", align_corners=False)
            frames = (
                up.clamp(0, 255)
                .reshape(B, 2, 3, cam_h, cam_w)
                .permute(0, 1, 3, 4, 2)  # (B, 2, H, W, 3)
                .round()
                .to(torch.uint8)
            )
            out_frames.append(frames)
    return torch.cat(out_frames, dim=0)


def compute_scorer_basin_parity(
    quantized_state_dict: dict[str, torch.Tensor],
    lossless_state_dict: dict[str, torch.Tensor],
    decoder: nn.Module,
    posenet: nn.Module,
    segnet: nn.Module,
    latents: torch.Tensor,
    gt_frames: torch.Tensor,
    *,
    n_probes: int = 10,
    pose_threshold: float = DEFAULT_POSE_DIST_DELTA_THRESHOLD,
    seg_threshold: float = DEFAULT_SEG_DIST_DELTA_THRESHOLD,
    hessian_log_ratio_tolerance: float = DEFAULT_HESSIAN_LOG_RATIO_TOLERANCE,
    absolute_pose_ceiling: float = DEFAULT_ABSOLUTE_POSE_CEILING,
    absolute_seg_ceiling: float = DEFAULT_ABSOLUTE_SEG_CEILING,
    n_hessian_samples: int = 4,
    n_hessian_pairs: int = 2,
    anchor_frame_shas: Sequence[str] = (),
    device: str = "cpu",
    seed: int = 1234,
) -> ParityReport:
    """Run the basin parity probe and return a structured :class:`ParityReport`.

    Args:
        quantized_state_dict: candidate (intN-quantized) decoder weights.
        lossless_state_dict: PR106 lossless reference decoder weights.
        decoder: instantiated HNeRVDecoder (or compatible nn.Module). The
            module is mutated by ``load_state_dict`` calls.
        posenet, segnet: instantiated, loaded scorers in ``.eval()`` mode.
        latents: ``(N_pairs, latent_dim)`` decoded latents (shared between the
            two state dicts in the apogee_intN pipeline).
        gt_frames: ``(n_probes, 2, H, W, 3)`` uint8 GT frame pairs.
        n_probes: how many latent pairs to use for distortion measurement.
        absolute_pose_ceiling, absolute_seg_ceiling: maximum allowed absolute
            scorer distances for both the lossless anchor and the candidate.
        n_hessian_samples: number of Hutchinson samples per Hessian trace.
        n_hessian_pairs: number of latent pairs to use for the Hessian probe.
        anchor_frame_shas: optional SHA-256 of GT frame bytes for forensic ref.
        device: "cpu" or "cuda" - the report is tagged with this string.
        seed: RNG seed for Rademacher samples.

    Returns:
        ParityReport.
    """

    if latents.shape[0] < n_probes:
        raise ValueError(
            f"latents has {latents.shape[0]} pairs but n_probes={n_probes}"
        )
    if gt_frames.shape[0] < n_probes:
        raise ValueError(
            f"gt_frames has {gt_frames.shape[0]} pairs but n_probes={n_probes}"
        )

    # Camera size = GT frame spatial size (lets stubs use small frames).
    cam_h, cam_w = int(gt_frames.shape[2]), int(gt_frames.shape[3])

    # ------------------------------------------------------------------
    # Distortion-delta axis
    # ------------------------------------------------------------------
    pred_lossless = reconstruct_frames(
        decoder,
        lossless_state_dict,
        latents[:n_probes],
        camera_size=(cam_h, cam_w),
        n_pairs=n_probes,
    )
    pose_lossless, seg_lossless = _run_scorer(
        posenet, segnet, pred_lossless, gt_frames[:n_probes]
    )

    pred_quantized = reconstruct_frames(
        decoder,
        quantized_state_dict,
        latents[:n_probes],
        camera_size=(cam_h, cam_w),
        n_pairs=n_probes,
    )
    pose_quantized, seg_quantized = _run_scorer(
        posenet, segnet, pred_quantized, gt_frames[:n_probes]
    )

    pose_delta = pose_quantized - pose_lossless
    seg_delta = seg_quantized - seg_lossless

    # ------------------------------------------------------------------
    # Hessian-trace axis (small probe; CPU-bounded)
    # ------------------------------------------------------------------
    trace_lossless = _hessian_probe(
        posenet,
        segnet,
        decoder,
        lossless_state_dict,
        latents[:n_hessian_pairs].clone(),
        gt_frames[:n_hessian_pairs].clone(),
        n_samples=n_hessian_samples,
        seed=seed,
        n_probes_for_hessian=n_hessian_pairs,
    )
    trace_quantized = _hessian_probe(
        posenet,
        segnet,
        decoder,
        quantized_state_dict,
        latents[:n_hessian_pairs].clone(),
        gt_frames[:n_hessian_pairs].clone(),
        n_samples=n_hessian_samples,
        seed=seed,
        n_probes_for_hessian=n_hessian_pairs,
    )

    if trace_lossless < 1e-12 and trace_quantized < 1e-12:
        # Both effectively zero curvature - degenerate but parity-preserving.
        log_ratio = 0.0
    elif trace_lossless < 1e-12:
        # Lossless flat, quantized non-flat -> suspicious; large positive ratio
        log_ratio = math.copysign(10.0, trace_quantized)
    else:
        log_ratio = math.log10(max(trace_quantized, 1e-12) / max(trace_lossless, 1e-12))

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    failure_reasons: list[str] = []
    if pose_delta > pose_threshold:
        failure_reasons.append(
            f"pose_dist_delta {pose_delta:.3e} exceeds threshold {pose_threshold:.3e}"
        )
    if pose_lossless > absolute_pose_ceiling:
        failure_reasons.append(
            f"pose_dist_lossless {pose_lossless:.3e} exceeds absolute ceiling "
            f"{absolute_pose_ceiling:.3e}"
        )
    if pose_quantized > absolute_pose_ceiling:
        failure_reasons.append(
            f"pose_dist_quantized {pose_quantized:.3e} exceeds absolute ceiling "
            f"{absolute_pose_ceiling:.3e}"
        )
    if seg_delta > seg_threshold:
        failure_reasons.append(
            f"seg_dist_delta {seg_delta:.3e} exceeds threshold {seg_threshold:.3e}"
        )
    if seg_lossless > absolute_seg_ceiling:
        failure_reasons.append(
            f"seg_dist_lossless {seg_lossless:.3e} exceeds absolute ceiling "
            f"{absolute_seg_ceiling:.3e}"
        )
    if seg_quantized > absolute_seg_ceiling:
        failure_reasons.append(
            f"seg_dist_quantized {seg_quantized:.3e} exceeds absolute ceiling "
            f"{absolute_seg_ceiling:.3e}"
        )
    if abs(log_ratio) > hessian_log_ratio_tolerance:
        failure_reasons.append(
            f"hessian_log10_ratio |{log_ratio:.2f}| exceeds tolerance "
            f"{hessian_log_ratio_tolerance:.2f}"
        )

    passed = not failure_reasons

    return ParityReport(
        pose_dist_lossless=pose_lossless,
        pose_dist_quantized=pose_quantized,
        pose_dist_delta=pose_delta,
        seg_dist_lossless=seg_lossless,
        seg_dist_quantized=seg_quantized,
        seg_dist_delta=seg_delta,
        hessian_trace_lossless=trace_lossless,
        hessian_trace_quantized=trace_quantized,
        hessian_log_ratio=log_ratio,
        basin_parity_passed=passed,
        pose_threshold=pose_threshold,
        seg_threshold=seg_threshold,
        hessian_log_ratio_tolerance=hessian_log_ratio_tolerance,
        absolute_pose_ceiling=absolute_pose_ceiling,
        absolute_seg_ceiling=absolute_seg_ceiling,
        n_probes=n_probes,
        n_hessian_samples=n_hessian_samples,
        anchor_frame_shas=tuple(anchor_frame_shas),
        device=device,
        computed_utc=_dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        failure_reasons=tuple(failure_reasons),
    )


__all__ = [
    "DEFAULT_HESSIAN_LOG_RATIO_TOLERANCE",
    "DEFAULT_POSE_DIST_DELTA_THRESHOLD",
    "DEFAULT_SEG_DIST_DELTA_THRESHOLD",
    "ParityReport",
    "compute_scorer_basin_parity",
    "hutchinson_trace_estimate",
    "reconstruct_frames",
]
