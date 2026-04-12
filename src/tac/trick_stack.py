"""Unified trick stacking engine for inflate-time optimizations.

Orchestrates ALL inflate-time tricks in the correct order. Each trick
builds on the output of the previous one, and every stage is independently
toggleable with configurable parameters.

Stacking order (each step refines the previous output):
  1. Load model from archive
  2. Self-supervised TTO (temporal consistency, no scorer needed)
  3. Supervised TTO with PoseNet targets (if available)
  4. Multi-pass inference (run model N times, uint8 round between)
  5. Frame-specific brightness shift (AllNorm invariance, free PoseNet bits)
  6. Chroma channel exploitation (perturbations invisible after YUV420)
  7. Per-frame scorer fragility weighting (spend more refinement on fragile frames)
  8. Noise-shaped uint8 rounding (gradient-directed, not nearest)
  9. Backward delta smoothing (smooth temporal transitions in reverse order)
 10. Null-space projection (hide remaining artifacts from scorer)
 11. Write output frames

Usage::

    from tac.trick_stack import stacked_inflate
    result = stacked_inflate(
        archive_dir=Path("archive"),
        output_dir=Path("inflated"),
        use_tto=True,
        use_multi_pass=3,
    )
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration dataclass — every parameter in one place
# ---------------------------------------------------------------------------


@dataclass
class TrickStackConfig:
    """All trick toggles and parameters for the stacking pipeline.

    Every field has a default so the pipeline runs out-of-the-box with
    sensible settings. Override individual fields via kwargs or by loading
    a profile dict.
    """

    # -- Stage 2: Self-supervised TTO --
    use_tto: bool = True
    tto_steps: int = 10
    tto_lr: float = 1e-4
    tto_loss: str = "temporal_consistency"
    tto_param_mode: str = "last_layer"
    tto_budget: float = 30.0  # 30s CPU safety default (10-min inflate budget)
    tto_batch_size: int = 16
    tto_max_frames: int = 64
    tto_frame_stride: int = 4

    # -- Stage 3: Supervised TTO with PoseNet targets --
    use_supervised_tto: bool = True
    supervised_tto_steps: int = 10
    supervised_tto_lr: float = 1e-4
    supervised_tto_param_mode: str = "all"
    supervised_tto_budget: float = 120.0
    supervised_tto_batch_size: int = 16
    supervised_tto_max_frames: int = 128
    supervised_tto_frame_stride: int = 2

    # -- Stage 4: Multi-pass inference --
    use_multi_pass: int = 3  # number of forward passes (1 = single pass)

    # -- Stage 5: Frame-specific brightness shift (Trick 13 — AllNorm) --
    use_brightness_shift: bool = False
    brightness_shift_value: float = 0.0  # additive offset, auto-tuned if 0
    brightness_shift_auto: bool = True  # auto-tune per frame to center at 128

    # -- Stage 6: Chroma channel exploitation (Trick 8 — YUV420 blind spot) --
    use_chroma_exploit: bool = False
    chroma_perturbation_magnitude: float = 1.0
    chroma_smooth_kernel_size: int = 3  # smoothing in chroma space

    # -- Stage 7: Per-frame scorer fragility weighting (Trick 22) --
    use_fragility_weighting: bool = False
    fragility_refinement_steps: int = 5
    fragility_lr: float = 0.1
    fragility_posenet_weight: float = 1.0
    fragility_segnet_weight: float = 100.0

    # -- Stage 8: Noise-shaped uint8 rounding (Trick 23) --
    use_noise_shaping: bool = True
    noise_shaping_diffuse_error: bool = True
    noise_shaping_fast: bool = False  # use fast (no diffusion) variant

    # -- Stage 9: Backward delta smoothing (Trick 32) --
    use_backward_delta_smoothing: bool = False
    backward_smooth_alpha: float = 0.3  # EMA blend factor (0 = no smooth, 1 = full)
    backward_smooth_max_delta: float = 3.0  # max per-pixel correction magnitude

    # -- Stage 10: Null-space projection (Trick 26) --
    use_null_space_projection: bool = True
    null_space_rank_threshold: float = 1e-3
    null_space_max_outputs: int = 16

    # -- Stage 11 (optional): Scorer equivalent search (Trick 34) --
    use_scorer_equivalent_search: bool = False
    scorer_equiv_steps: int = 100
    scorer_equiv_lr: float = 0.5
    scorer_equiv_compressibility_weight: float = 0.1
    scorer_equiv_tolerance: float = 1e-4

    # -- CPU stacked pipeline (Eureka 1-6) --
    use_parallel_decode: bool = False
    parallel_workers: int = 4

    use_precomputed_corrections: bool = False
    corrections_path: str | None = None

    use_multi_model: bool = False
    hard_model_path: str | None = None

    use_precomputed_brightness: bool = False
    use_precomputed_null_space: bool = False
    use_precomputed_gradients: bool = False

    # -- Paths --
    posenet_targets_path: str | None = None
    scorer_models_dir: str | None = None
    posenet_path: str | None = None
    segnet_path: str | None = None
    upstream_dir: str | None = None

    # -- Runtime --
    device: str = "cpu"
    batch_size: int = 8
    target_w: int = 1164
    target_h: int = 874
    verbose: bool = True
    total_time_budget: float = 1800.0  # 30 min eval limit

    @classmethod
    def from_profile(cls, profile: dict[str, Any], **overrides: Any) -> "TrickStackConfig":
        """Create config from a profile dict with optional overrides."""
        merged = {**profile, **overrides}
        # Filter to only fields that exist in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in merged.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(msg: str, verbose: bool = True) -> None:
    """Print to stderr if verbose."""
    if verbose:
        print(f"  [trick_stack] {msg}", file=sys.stderr, flush=True)


def _decode_frames(
    video_path: str,
    target_w: int,
    target_h: int,
    max_frames: int = 64,
    stride: int = 1,
) -> torch.Tensor:
    """Decode a subset of frames from video for TTO/analysis.

    Returns (N, 3, H, W) float tensor in [0, 255].
    """
    import av

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        frames = []
        i = 0
        for frame in container.decode(stream):
            if i % stride == 0:
                # YUV420 -> RGB via the inflate_postfilter path
                H, W = frame.height, frame.width
                y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(
                    H, frame.planes[0].line_size
                )[:, :W]
                u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(
                    H // 2, frame.planes[1].line_size
                )[:, : W // 2]
                v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(
                    H // 2, frame.planes[2].line_size
                )[:, : W // 2]

                y_t = torch.from_numpy(y.copy()).float()
                u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
                v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

                u_up = F.interpolate(u_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()
                v_up = F.interpolate(v_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()

                yf = (y_t - 16.0) * (255.0 / 219.0)
                uf = (u_up - 128.0) * (255.0 / 224.0)
                vf = (v_up - 128.0) * (255.0 / 224.0)

                r = (yf + 1.402 * vf).clamp(0, 255)
                g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
                b = (yf + 1.772 * uf).clamp(0, 255)
                rgb = torch.stack([r, g, b], dim=0).unsqueeze(0)  # (1, 3, H, W)

                if H != target_h or W != target_w:
                    rgb = F.interpolate(rgb, size=(target_h, target_w), mode="bicubic", align_corners=False)
                    rgb = rgb.clamp(0, 255)
                frames.append(rgb)
                if len(frames) >= max_frames:
                    break
            i += 1
    finally:
        container.close()

    if frames:
        return torch.cat(frames, dim=0)
    return torch.empty(0, 3, target_h, target_w)


def _load_scorers(cfg: TrickStackConfig):
    """Load PoseNet and SegNet scorer models. Returns (posenet, segnet) or (None, None)."""
    try:
        from tac.scorer import load_scorers
    except ImportError:
        _log("tac.scorer not available, cannot load scorers", cfg.verbose)
        return None, None

    posenet_path = cfg.posenet_path
    segnet_path = cfg.segnet_path
    if not posenet_path or not segnet_path:
        if cfg.upstream_dir:
            posenet_path = posenet_path or str(Path(cfg.upstream_dir) / "models" / "posenet.safetensors")
            segnet_path = segnet_path or str(Path(cfg.upstream_dir) / "models" / "segnet.safetensors")
        if cfg.scorer_models_dir:
            posenet_path = posenet_path or str(Path(cfg.scorer_models_dir) / "posenet.safetensors")
            segnet_path = segnet_path or str(Path(cfg.scorer_models_dir) / "segnet.safetensors")

    if not posenet_path or not segnet_path:
        _log("scorer model paths not configured", cfg.verbose)
        return None, None

    try:
        posenet, segnet = load_scorers(
            posenet_path, segnet_path,
            device=cfg.device, upstream_dir=cfg.upstream_dir,
        )
        return posenet, segnet
    except Exception as e:
        _log(f"failed to load scorers: {e}", cfg.verbose)
        return None, None


# ---------------------------------------------------------------------------
# Individual trick stages
# ---------------------------------------------------------------------------


def _stage_self_supervised_tto(
    model: nn.Module,
    frames: torch.Tensor,
    cfg: TrickStackConfig,
) -> nn.Module:
    """Stage 2: Self-supervised TTO."""
    from tac.tto import test_time_optimize

    if frames.shape[0] < 2:
        _log("TTO: not enough frames, skipping", cfg.verbose)
        return model

    _log(
        f"TTO: {cfg.tto_steps} steps, loss={cfg.tto_loss}, "
        f"lr={cfg.tto_lr}, on {frames.shape[0]} frames",
        cfg.verbose,
    )
    return test_time_optimize(
        model,
        frames,
        n_steps=cfg.tto_steps,
        lr=cfg.tto_lr,
        loss_type=cfg.tto_loss,
        param_mode=cfg.tto_param_mode,
        time_budget_seconds=cfg.tto_budget,
        batch_size=cfg.tto_batch_size,
        verbose=cfg.verbose,
    )


def _stage_supervised_tto(
    model: nn.Module,
    frames: torch.Tensor,
    posenet: nn.Module,
    cfg: TrickStackConfig,
) -> nn.Module:
    """Stage 3: Supervised TTO with PoseNet targets."""
    from tac.scorer_targets import load_posenet_targets
    from tac.tto import supervised_tto

    targets_dict = load_posenet_targets(cfg.posenet_targets_path, device=cfg.device)
    if targets_dict is None:
        _log("supervised TTO: targets not found, skipping", cfg.verbose)
        return model

    if frames.shape[0] < 2:
        _log("supervised TTO: not enough frames, skipping", cfg.verbose)
        return model

    _log(
        f"supervised TTO: {cfg.supervised_tto_steps} steps, "
        f"lr={cfg.supervised_tto_lr}, {frames.shape[0]} frames, "
        f"{targets_dict['n_pairs']} targets",
        cfg.verbose,
    )
    return supervised_tto(
        model,
        frames,
        posenet,
        targets_dict["targets"],
        n_steps=cfg.supervised_tto_steps,
        lr=cfg.supervised_tto_lr,
        param_mode=cfg.supervised_tto_param_mode,
        time_budget_seconds=cfg.supervised_tto_budget,
        batch_size=cfg.supervised_tto_batch_size,
        verbose=cfg.verbose,
    )


def _stage_multi_pass(
    frames_bchw: torch.Tensor,
    model: nn.Module,
    n_passes: int,
) -> torch.Tensor:
    """Stage 4: Multi-pass inference with uint8 rounding between passes.

    Runs the postfilter N times. Between passes, output is rounded to uint8
    to match the deployment distribution (the model was trained on uint8 input).

    Args:
        frames_bchw: (B, 3, H, W) float tensor in [0, 255].
        model: postfilter model.
        n_passes: number of forward passes.

    Returns:
        (B, 3, H, W) float tensor — postfilter output after N passes.
    """
    out = frames_bchw
    with torch.inference_mode():
        for p in range(n_passes):
            out = model(out)
            if p < n_passes - 1:
                # Round to uint8 between passes to match training distribution
                out = out.round().clamp(0, 255)
    return out


def _stage_brightness_shift(
    frames_bchw: torch.Tensor,
    cfg: TrickStackConfig,
) -> torch.Tensor:
    """Stage 5: Frame-specific brightness shift (Trick 13 — AllNorm invariance).

    PoseNet uses normalization layers (BatchNorm/InstanceNorm) that subtract
    the mean, making it invariant to global brightness shifts. We exploit this
    to shift brightness toward values that quantize more favorably.

    Args:
        frames_bchw: (B, 3, H, W) float tensor in [0, 255].
        cfg: pipeline configuration.

    Returns:
        (B, 3, H, W) brightness-shifted frames.
    """
    from tac.scorer_exploits import apply_global_brightness_shift

    if cfg.brightness_shift_auto:
        # Auto-tune: shift each frame's mean luminance toward 128
        # (symmetric around midpoint = better uint8 quantization)
        B = frames_bchw.shape[0]
        result = frames_bchw.clone()
        for i in range(B):
            frame = frames_bchw[i:i + 1]
            # Compute luminance mean
            luma = frame[:, 0] * 0.299 + frame[:, 1] * 0.587 + frame[:, 2] * 0.114
            current_mean = luma.mean().item()
            shift = 128.0 - current_mean
            # Clamp shift to avoid saturation
            shift = max(-cfg.brightness_shift_value if cfg.brightness_shift_value > 0 else -30.0,
                        min(cfg.brightness_shift_value if cfg.brightness_shift_value > 0 else 30.0, shift))
            result[i:i + 1] = apply_global_brightness_shift(frame, shift)
        return result
    else:
        return apply_global_brightness_shift(frames_bchw, cfg.brightness_shift_value)


def _stage_chroma_exploit(
    frames_bchw: torch.Tensor,
    cfg: TrickStackConfig,
) -> torch.Tensor:
    """Stage 6: Chroma channel exploitation (Trick 8 — YUV420 blind spot).

    YUV420 subsampling discards half the spatial resolution in U and V
    channels. Perturbations at chroma-subsampled positions are invisible
    to the scorer after preprocess_input(). We exploit this by smoothing
    the chroma channels at positions that will be discarded, reducing
    compression cost at zero scorer penalty.

    Args:
        frames_bchw: (B, 3, H, W) float tensor in [0, 255] (RGB).
        cfg: pipeline configuration.

    Returns:
        (B, 3, H, W) frames with chroma perturbations applied.
    """
    B, C, H, W = frames_bchw.shape

    # Convert RGB -> YUV (BT.601 to match scorer)
    r, g, b = frames_bchw[:, 0], frames_bchw[:, 1], frames_bchw[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.169 * r - 0.331 * g + 0.500 * b + 128.0
    v = 0.500 * r - 0.419 * g - 0.081 * b + 128.0

    # Smooth chroma at positions that will be subsampled away
    # YUV420 keeps every other pixel in both dimensions for U and V
    k = cfg.chroma_smooth_kernel_size
    pad = k // 2
    if k > 1:
        kernel = torch.ones(1, 1, k, k, device=frames_bchw.device) / (k * k)
        u_smooth = F.conv2d(u.unsqueeze(1), kernel, padding=pad).squeeze(1)
        v_smooth = F.conv2d(v.unsqueeze(1), kernel, padding=pad).squeeze(1)

        # Blend: only apply smoothing at odd pixel positions (will be discarded by 420)
        mask = torch.zeros(H, W, device=frames_bchw.device)
        mask[1::2, :] = 1.0  # odd rows
        mask[:, 1::2] = 1.0  # odd columns
        alpha = mask * cfg.chroma_perturbation_magnitude
        alpha = alpha.clamp(0, 1)

        u = u * (1 - alpha) + u_smooth * alpha
        v = v * (1 - alpha) + v_smooth * alpha

    # Convert back to RGB
    y_adj = y
    u_adj = u - 128.0
    v_adj = v - 128.0
    r_out = (y_adj + 1.402 * v_adj).clamp(0, 255)
    g_out = (y_adj - 0.344136 * u_adj - 0.714136 * v_adj).clamp(0, 255)
    b_out = (y_adj + 1.772 * u_adj).clamp(0, 255)

    return torch.stack([r_out, g_out, b_out], dim=1)


def _stage_fragility_refinement(
    frames_bchw: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
    cfg: TrickStackConfig,
) -> torch.Tensor:
    """Stage 7: Per-frame scorer fragility weighting (Trick 22).

    Computes per-pixel gradient magnitude of the scorer loss to identify
    fragile regions, then runs extra local refinement steps on high-fragility
    frames. Frames where the scorer is already confident get fewer steps.

    Args:
        frames_bchw: (B, 3, H, W) float tensor in [0, 255].
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        cfg: pipeline configuration.

    Returns:
        (B, 3, H, W) refined frames.
    """
    from tac.scorer_exploits import compute_scorer_fragility_map

    B = frames_bchw.shape[0]
    result = frames_bchw.clone()

    for i in range(B):
        frame = frames_bchw[i:i + 1].detach().clone().requires_grad_(True)
        fragility = compute_scorer_fragility_map(
            frame,
            posenet,
            segnet,
            posenet_weight=cfg.fragility_posenet_weight,
            segnet_weight=cfg.fragility_segnet_weight,
        )
        # Mean fragility for this frame — determines refinement intensity
        mean_frag = fragility.mean().item()

        # Scale refinement steps by fragility (more fragile = more steps)
        n_steps = max(1, int(cfg.fragility_refinement_steps * mean_frag * 2))
        n_steps = min(n_steps, cfg.fragility_refinement_steps * 2)

        # Gradient descent on the fragile pixels
        pixel = frame.detach().clone().requires_grad_(True)
        optimizer = torch.optim.SGD([pixel], lr=cfg.fragility_lr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            # Weight loss by fragility map: focus optimization on fragile pixels
            pair = pixel.unsqueeze(1).expand(1, 2, 3, *frame.shape[2:]).contiguous()
            pose_in = posenet.preprocess_input(pair)
            pose_out = posenet(pose_in)
            pose_tensor = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
            pose_loss = pose_tensor[..., :6].pow(2).sum()

            seg_in = segnet.preprocess_input(pair)
            seg_out = segnet(seg_in)
            seg_probs = F.softmax(seg_out, dim=1)
            seg_loss = -(seg_probs * (seg_probs + 1e-8).log()).sum()

            loss = cfg.fragility_posenet_weight * pose_loss + cfg.fragility_segnet_weight * seg_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pixel.data.clamp_(0.0, 255.0)

        result[i:i + 1] = pixel.detach()

    return result


def _stage_noise_shaped_round(
    frames_bchw: torch.Tensor,
    posenet: nn.Module | None,
    segnet: nn.Module | None,
    cfg: TrickStackConfig,
) -> torch.Tensor:
    """Stage 8: Noise-shaped uint8 rounding (Trick 23).

    Instead of nearest-neighbor rounding, uses the scorer gradient to direct
    each sub-pixel toward floor or ceil, whichever reduces scorer loss.

    Args:
        frames_bchw: (B, 3, H, W) float tensor in [0, 255].
        posenet: frozen PoseNet model (needed for gradient computation).
        segnet: frozen SegNet model (needed for gradient computation).
        cfg: pipeline configuration.

    Returns:
        (B, 3, H, W) uint8-valued float tensor.
    """
    if posenet is None or segnet is None:
        _log("noise shaping: scorers not available, falling back to nearest round", cfg.verbose)
        return frames_bchw.round().clamp(0, 255)

    if cfg.noise_shaping_fast:
        from tac.quantization import noise_shaped_round_fast

        # Compute scorer gradient for the entire batch
        inp = frames_bchw.detach().clone().requires_grad_(True)
        pair = inp.unsqueeze(1).expand(inp.shape[0], 2, 3, *inp.shape[2:]).contiguous()
        pose_in = posenet.preprocess_input(pair)
        pose_out = posenet(pose_in)
        pose_tensor = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
        pose_loss = pose_tensor[..., :6].pow(2).sum()

        seg_in = segnet.preprocess_input(pair)
        seg_out = segnet(seg_in)
        seg_probs = F.softmax(seg_out, dim=1)
        seg_loss = -(seg_probs * (seg_probs + 1e-8).log()).sum()

        total_loss = pose_loss + 100.0 * seg_loss
        total_loss.backward()

        return noise_shaped_round_fast(frames_bchw, inp.grad)
    else:
        from tac.quantization import noise_shaped_round

        # Process per-frame to manage memory
        B = frames_bchw.shape[0]
        results = []
        for i in range(B):
            frame = frames_bchw[i:i + 1]
            inp = frame.detach().clone().requires_grad_(True)
            pair = inp.unsqueeze(1).expand(1, 2, 3, *inp.shape[2:]).contiguous()
            pose_in = posenet.preprocess_input(pair)
            pose_out = posenet(pose_in)
            pose_tensor = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
            pose_loss = pose_tensor[..., :6].pow(2).sum()

            seg_in = segnet.preprocess_input(pair)
            seg_out = segnet(seg_in)
            seg_probs = F.softmax(seg_out, dim=1)
            seg_loss = -(seg_probs * (seg_probs + 1e-8).log()).sum()

            total_loss = pose_loss + 100.0 * seg_loss
            total_loss.backward()

            rounded = noise_shaped_round(
                frame, inp.grad,
                diffuse_error=cfg.noise_shaping_diffuse_error,
            )
            results.append(rounded)

        return torch.cat(results, dim=0)


def _stage_backward_delta_smoothing(
    frames_list: list[torch.Tensor],
    cfg: TrickStackConfig,
) -> list[torch.Tensor]:
    """Stage 9: Backward delta smoothing (Trick 32).

    Process frames in reverse temporal order, applying EMA smoothing to
    the inter-frame deltas. This removes temporal jitter that PoseNet
    penalizes, without requiring a second forward pass.

    The key insight: PoseNet scores *pairs* of consecutive frames. Temporal
    noise between frames hurts PoseNet MSE. By smoothing the delta between
    t and t-1 in reverse order, we propagate the smoothing backward through
    time, reducing accumulated temporal drift.

    Args:
        frames_list: list of (1, 3, H, W) or (3, H, W) tensors in temporal order.
        cfg: pipeline configuration.

    Returns:
        list of smoothed frame tensors (same shapes as input).
    """
    if len(frames_list) < 2:
        return frames_list

    alpha = cfg.backward_smooth_alpha
    max_delta = cfg.backward_smooth_max_delta

    # Work with 3D tensors (3, H, W)
    was_4d = frames_list[0].ndim == 4
    frames = [f.squeeze(0) if f.ndim == 4 else f for f in frames_list]

    # Reverse pass: smooth deltas from last frame backward
    smoothed = list(frames)  # copy
    for i in range(len(smoothed) - 2, -1, -1):
        delta = smoothed[i].float() - smoothed[i + 1].float()
        # Clamp delta magnitude
        delta = delta.clamp(-max_delta, max_delta)
        # EMA: blend current frame toward the next frame
        correction = alpha * delta
        smoothed[i] = (smoothed[i].float() - correction).round().clamp(0, 255).to(smoothed[i].dtype)

    if was_4d:
        smoothed = [f.unsqueeze(0) for f in smoothed]

    return smoothed


def _stage_null_space_projection(
    frames_bchw: torch.Tensor,
    original_bchw: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
    cfg: TrickStackConfig,
) -> torch.Tensor:
    """Stage 10: Null-space projection (Trick 26).

    Computes the scorer Jacobian and projects the residual (difference between
    processed and original frames) into the scorer's null space. The component
    of the residual that the scorer cannot see is retained; the component that
    the scorer is sensitive to is removed.

    Args:
        frames_bchw: (B, 3, H, W) processed frames (after all previous tricks).
        original_bchw: (B, 3, H, W) original decoded frames (before tricks).
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        cfg: pipeline configuration.

    Returns:
        (B, 3, H, W) frames with null-space-projected residual.
    """
    from tac.scorer_exploits import compute_scorer_jacobian, project_to_scorer_null_space

    B = frames_bchw.shape[0]
    result = frames_bchw.clone()

    for i in range(B):
        frame = frames_bchw[i:i + 1]
        original = original_bchw[i:i + 1]
        residual = frame - original  # what the tricks changed

        # Compute Jacobian at the original frame
        jacobian = compute_scorer_jacobian(
            original,
            posenet,
            segnet,
            max_outputs=cfg.null_space_max_outputs,
        )

        # Project residual into null space
        residual_flat = residual.reshape(3, *frame.shape[2:])
        projected = project_to_scorer_null_space(
            residual_flat,
            jacobian,
            rank_threshold=cfg.null_space_rank_threshold,
        )

        # Apply only the null-space component of our changes
        result[i:i + 1] = (original + projected.unsqueeze(0)).clamp(0, 255)

    return result


# ---------------------------------------------------------------------------
# Main stacking pipeline
# ---------------------------------------------------------------------------


def stacked_inflate(
    archive_dir: Path,
    output_dir: Path,
    # Trick toggles (all configurable)
    use_tto: bool = True,
    tto_steps: int = 10,
    tto_lr: float = 1e-4,
    use_supervised_tto: bool = True,
    supervised_tto_steps: int = 10,
    use_multi_pass: int = 3,
    use_noise_shaping: bool = True,
    use_null_space_projection: bool = True,
    use_scorer_equivalent_search: bool = False,
    use_brightness_shift: bool = False,
    brightness_shift_auto: bool = True,
    use_chroma_exploit: bool = False,
    chroma_perturbation_magnitude: float = 1.0,
    use_fragility_weighting: bool = False,
    fragility_refinement_steps: int = 5,
    use_backward_delta_smoothing: bool = False,
    backward_smooth_alpha: float = 0.3,
    backward_smooth_max_delta: float = 3.0,
    # Paths
    posenet_targets_path: str | None = None,
    scorer_models_dir: str | None = None,
    posenet_path: str | None = None,
    segnet_path: str | None = None,
    upstream_dir: str | None = None,
    # Runtime
    device: str = "cpu",
    batch_size: int = 8,
    target_w: int = 1164,
    target_h: int = 874,
    verbose: bool = True,
    **extra_kwargs: Any,
) -> dict[str, Any]:
    """Run ALL tricks in the optimal stacking order.

    Order matters — each trick builds on the previous:
      1. Load model from archive
      2. Self-supervised TTO (temporal consistency, no scorer needed)
      3. Supervised TTO with PoseNet targets (if available)
      4. Multi-pass inference (run model N times, uint8 round between)
      5. Frame-specific brightness shift (AllNorm invariance, free PoseNet bits)
      6. Chroma channel exploitation (perturbations invisible after YUV420)
      7. Per-frame scorer fragility weighting (spend more on fragile frames)
      8. Noise-shaped uint8 rounding (gradient-directed, not nearest)
      9. Backward delta smoothing (smooth temporal transitions in reverse)
     10. Null-space projection (hide remaining artifacts from scorer)
     11. Write output frames

    Returns dict with timing and improvement estimates.
    """
    import av
    from tac.quantization import load_postfilter_int8

    t0 = time.monotonic()
    timings: dict[str, float] = {}
    stages_run: list[str] = []

    # Build config from function args
    cfg = TrickStackConfig(
        use_tto=use_tto,
        tto_steps=tto_steps,
        tto_lr=tto_lr,
        use_supervised_tto=use_supervised_tto,
        supervised_tto_steps=supervised_tto_steps,
        use_multi_pass=use_multi_pass,
        use_noise_shaping=use_noise_shaping,
        use_null_space_projection=use_null_space_projection,
        use_scorer_equivalent_search=use_scorer_equivalent_search,
        use_brightness_shift=use_brightness_shift,
        brightness_shift_auto=brightness_shift_auto,
        use_chroma_exploit=use_chroma_exploit,
        chroma_perturbation_magnitude=chroma_perturbation_magnitude,
        use_fragility_weighting=use_fragility_weighting,
        fragility_refinement_steps=fragility_refinement_steps,
        use_backward_delta_smoothing=use_backward_delta_smoothing,
        backward_smooth_alpha=backward_smooth_alpha,
        backward_smooth_max_delta=backward_smooth_max_delta,
        posenet_targets_path=posenet_targets_path,
        scorer_models_dir=scorer_models_dir,
        posenet_path=posenet_path,
        segnet_path=segnet_path,
        upstream_dir=upstream_dir,
        device=device,
        batch_size=batch_size,
        target_w=target_w,
        target_h=target_h,
        verbose=verbose,
    )

    archive_dir = Path(archive_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: Load model ----
    _log("Stage 1: Loading postfilter model", verbose)
    t_stage = time.monotonic()

    postfilter_path = archive_dir / "postfilter_int8.pt"
    if not postfilter_path.exists():
        # Fallback: look alongside this script
        postfilter_path = Path(__file__).resolve().parent.parent.parent / "submissions" / "robust_current" / "postfilter_int8.pt"
    if not postfilter_path.exists():
        raise FileNotFoundError(f"postfilter_int8.pt not found in {archive_dir} or submissions/robust_current/")

    model = load_postfilter_int8(str(postfilter_path), device=device)
    timings["load_model"] = time.monotonic() - t_stage
    stages_run.append("load_model")

    # ---- Load scorers if any scorer-dependent trick is enabled ----
    needs_scorers = (
        cfg.use_noise_shaping
        or cfg.use_null_space_projection
        or cfg.use_fragility_weighting
        or cfg.use_scorer_equivalent_search
    )
    posenet, segnet = None, None
    if needs_scorers:
        _log("Loading scorer models", verbose)
        posenet, segnet = _load_scorers(cfg)
        if posenet is None:
            _log("scorers not available, scorer-dependent tricks will be skipped", verbose)

    # ---- Find video files ----
    mkv_files = sorted(archive_dir.glob("**/*.mkv"))
    if not mkv_files:
        _log(f"no .mkv files found in {archive_dir}", verbose)
        return {"timings": timings, "stages_run": stages_run, "n_frames": 0}

    # ---- Stage 2: Self-supervised TTO ----
    if cfg.use_tto and cfg.tto_steps > 0:
        _log("Stage 2: Self-supervised TTO", verbose)
        t_stage = time.monotonic()
        # Use first video for TTO (representative content)
        tto_frames = _decode_frames(
            str(mkv_files[0]), target_w, target_h,
            max_frames=cfg.tto_max_frames, stride=cfg.tto_frame_stride,
        )
        if tto_frames.shape[0] >= 2:
            model = _stage_self_supervised_tto(model, tto_frames, cfg)
            del tto_frames
        timings["tto"] = time.monotonic() - t_stage
        stages_run.append("tto")

    # ---- Stage 3: Supervised TTO ----
    if cfg.use_supervised_tto and cfg.supervised_tto_steps > 0 and cfg.posenet_targets_path:
        if posenet is None:
            posenet, segnet = _load_scorers(cfg)
        if posenet is not None:
            _log("Stage 3: Supervised TTO", verbose)
            t_stage = time.monotonic()
            stto_frames = _decode_frames(
                str(mkv_files[0]), target_w, target_h,
                max_frames=cfg.supervised_tto_max_frames,
                stride=cfg.supervised_tto_frame_stride,
            )
            if stto_frames.shape[0] >= 2:
                model = _stage_supervised_tto(model, stto_frames, posenet, cfg)
                del stto_frames
            timings["supervised_tto"] = time.monotonic() - t_stage
            stages_run.append("supervised_tto")

    # ---- Stages 4-10: Process each video ----
    total_frames = 0
    for mkv_path in mkv_files:
        stem = mkv_path.stem
        out_path = output_dir / f"{stem}.raw"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        _log(f"Processing {mkv_path.name}", verbose)
        t_video = time.monotonic()

        container = av.open(str(mkv_path))
        stream = container.streams.video[0]

        all_frames: list[torch.Tensor] = []  # collect for backward smoothing
        all_originals: list[torch.Tensor] = []  # original decoded frames
        batch_buf: list[torch.Tensor] = []
        orig_buf: list[torch.Tensor] = []
        n_video = 0

        def _process_batch(batch_tensors: list[torch.Tensor], orig_tensors: list[torch.Tensor]) -> None:
            nonlocal total_frames, n_video
            if not batch_tensors:
                return

            x = torch.cat(batch_tensors, dim=0).to(device)
            x_orig = torch.cat(orig_tensors, dim=0).to(device)

            # Stage 4: Multi-pass inference
            if cfg.use_multi_pass > 1:
                out = _stage_multi_pass(x, model, cfg.use_multi_pass)
            else:
                with torch.inference_mode():
                    out = model(x)

            # Stage 5: Brightness shift
            if cfg.use_brightness_shift:
                out = _stage_brightness_shift(out, cfg)

            # Stage 6: Chroma exploit
            if cfg.use_chroma_exploit:
                out = _stage_chroma_exploit(out, cfg)

            # Stage 7: Fragility-weighted refinement
            if cfg.use_fragility_weighting and posenet is not None and segnet is not None:
                out = _stage_fragility_refinement(out, posenet, segnet, cfg)

            # Stage 8: Noise-shaped rounding
            if cfg.use_noise_shaping and posenet is not None and segnet is not None:
                out = _stage_noise_shaped_round(out, posenet, segnet, cfg)
            else:
                out = out.round().clamp(0, 255)

            # Stage 10: Null-space projection
            if cfg.use_null_space_projection and posenet is not None and segnet is not None:
                out = _stage_null_space_projection(out, x_orig, posenet, segnet, cfg)

            # Collect frames for backward smoothing (stage 9)
            for i in range(out.shape[0]):
                frame_uint8 = out[i].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu()
                all_frames.append(frame_uint8)
                n_video += 1
                total_frames += 1

        for frame in container.decode(stream):
            H, W = frame.height, frame.width
            y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(
                H, frame.planes[0].line_size
            )[:, :W]
            u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(
                H // 2, frame.planes[1].line_size
            )[:, : W // 2]
            v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(
                H // 2, frame.planes[2].line_size
            )[:, : W // 2]

            y_t = torch.from_numpy(y.copy()).float()
            u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
            v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

            u_up = F.interpolate(u_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()
            v_up = F.interpolate(v_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()

            yf = (y_t - 16.0) * (255.0 / 219.0)
            uf = (u_up - 128.0) * (255.0 / 224.0)
            vf = (v_up - 128.0) * (255.0 / 224.0)

            r = (yf + 1.402 * vf).clamp(0, 255)
            g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
            b = (yf + 1.772 * uf).clamp(0, 255)
            rgb = torch.stack([r, g, b], dim=0).unsqueeze(0)  # (1, 3, H, W)

            if H != target_h or W != target_w:
                rgb = F.interpolate(rgb, size=(target_h, target_w), mode="bicubic", align_corners=False)
                rgb = rgb.clamp(0, 255)

            batch_buf.append(rgb)
            orig_buf.append(rgb.clone())

            if len(batch_buf) >= batch_size:
                _process_batch(batch_buf, orig_buf)
                batch_buf.clear()
                orig_buf.clear()

            if n_video > 0 and n_video % 300 == 0:
                _log(f"  {n_video} frames ...", verbose)

        # Flush remaining
        _process_batch(batch_buf, orig_buf)
        batch_buf.clear()
        orig_buf.clear()
        container.close()

        # Stage 9: Backward delta smoothing (needs all frames in memory)
        if cfg.use_backward_delta_smoothing and len(all_frames) >= 2:
            _log(f"Stage 9: Backward delta smoothing ({len(all_frames)} frames)", verbose)
            t_smooth = time.monotonic()
            # Convert to float for smoothing
            float_frames = [f.float().permute(2, 0, 1) for f in all_frames]
            smoothed = _stage_backward_delta_smoothing(float_frames, cfg)
            all_frames = [f.permute(1, 2, 0).to(torch.uint8) for f in smoothed]
            timings["backward_smoothing"] = time.monotonic() - t_smooth
            stages_run.append("backward_smoothing")

        # Write output frames
        with open(str(out_path), "wb") as f:
            for frame_tensor in all_frames:
                f.write(frame_tensor.contiguous().numpy().tobytes())

        timings[f"video_{stem}"] = time.monotonic() - t_video
        _log(f"  {mkv_path.name}: {n_video} frames in {timings[f'video_{stem}']:.1f}s", verbose)

    total_elapsed = time.monotonic() - t0
    _log(f"Stacked inflate complete: {total_frames} frames in {total_elapsed:.1f}s", verbose)
    _log(f"Stages run: {stages_run}", verbose)

    return {
        "timings": timings,
        "stages_run": stages_run,
        "n_frames": total_frames,
        "total_elapsed": total_elapsed,
    }


def stacked_inflate_from_config(
    archive_dir: Path,
    output_dir: Path,
    config: TrickStackConfig,
) -> dict[str, Any]:
    """Run stacked inflate using a TrickStackConfig object.

    This is the preferred entry point when using profiles or programmatic
    configuration. The stacked_inflate() function with keyword arguments
    is more convenient for one-off CLI usage.

    Args:
        archive_dir: directory containing .mkv files and postfilter_int8.pt.
        output_dir: directory for output .raw files.
        config: full pipeline configuration.

    Returns:
        dict with timing and improvement estimates.
    """
    # Convert config to kwargs for stacked_inflate
    kwargs = {f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()}
    kwargs.pop("total_time_budget", None)  # not a kwarg of stacked_inflate
    return stacked_inflate(archive_dir=archive_dir, output_dir=output_dir, **kwargs)
