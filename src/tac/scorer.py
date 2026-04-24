"""Scorer interface for task-aware codec post-filters.

A Scorer wraps one or more frozen perception networks and provides:
  - Forward pass: compute distortion between filtered and ground-truth frames
  - Saliency: compute per-pixel gradient magnitude for loss weighting
  - Score formula: combine distortion terms into the final competition metric

The Scorer is frozen — its parameters are never updated. The post-filter
learns to minimize the Scorer's output.

Example::

    scorer = Scorer.from_comma_challenge(
        posenet_path="models/posenet.safetensors",
        segnet_path="models/segnet.safetensors",
    )
    pose_dist, seg_dist = scorer.distortion(filtered_pair, gt_pair)
    score = scorer.score(pose_dist, seg_dist, rate)
    saliency = scorer.posenet_saliency(gt_frames)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Protocol

import torch


class Scorer(Protocol):
    """Protocol for a frozen scorer that evaluates frame quality."""

    def distortion(
        self,
        filtered: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> tuple[float, float]:
        """Compute (pose_distortion, seg_distortion) for a frame pair.

        Args:
            filtered: (1, 2, H, W, 3) uint8 or float filtered pair
            ground_truth: (1, 2, H, W, 3) uint8 or float GT pair

        Returns:
            (pose_dist, seg_dist) as floats
        """
        ...

    def score(
        self,
        pose_dist: float,
        seg_dist: float,
        rate: float,
    ) -> float:
        """Compute the competition score from distortion + rate.

        Default formula: 100 * seg_dist + sqrt(10 * pose_dist) + 25 * rate
        """
        ...

    def posenet_saliency(
        self,
        gt_frames: list[torch.Tensor],
        device: str = "cpu",
    ) -> torch.Tensor:
        """Compute per-pixel PoseNet gradient saliency on GT frames.

        Returns: (N, H, W) float tensor of gradient magnitudes.
        """
        ...


def detect_device() -> torch.device:
    """Auto-detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_scorers(
    posenet_path: str | Path,
    segnet_path: str | Path,
    device: str | torch.device | None = None,
    upstream_dir: str | Path | None = None,
) -> tuple:
    """Load frozen PoseNet and SegNet scorer models.

    This is the single most-used function across all experiment scripts.
    Loads from safetensors, freezes parameters, moves to device.

    Args:
        posenet_path: path to posenet.safetensors
        segnet_path: path to segnet.safetensors
        device: target device (auto-detected if None)
        upstream_dir: if set, adds to sys.path for importing modules.py

    Returns:
        (posenet, segnet) tuple, both frozen and on device
    """
    if device is None:
        device = detect_device()
    device = torch.device(device) if isinstance(device, str) else device

    # Ensure upstream modules are importable
    if upstream_dir:
        p = str(Path(upstream_dir))
        if p not in sys.path:
            sys.path.insert(0, p)

    from modules import PoseNet, SegNet
    from safetensors.torch import load_file

    # Load on CPU first, then move — avoids 2x peak VRAM from constructor
    # allocating zero tensors on device then immediately replacing with loaded weights
    posenet = PoseNet().eval()
    segnet = SegNet().eval()
    posenet.load_state_dict(load_file(str(posenet_path), device="cpu"))
    segnet.load_state_dict(load_file(str(segnet_path), device="cpu"))
    posenet = posenet.to(device)
    segnet = segnet.to(device)

    for p in posenet.parameters():
        p.requires_grad = False
    for p in segnet.parameters():
        p.requires_grad = False

    return posenet, segnet


def load_default_scorers(
    upstream_dir: str | Path,
    device: str | torch.device | None = None,
) -> tuple:
    """Load frozen PoseNet and SegNet from standard upstream model paths.

    Convenience wrapper around load_scorers() that constructs the canonical
    model paths (upstream/models/posenet.safetensors, upstream/models/segnet.safetensors).

    Args:
        upstream_dir: path to the upstream repository root.
        device: target device (auto-detected if None).

    Returns:
        (posenet, segnet) tuple, both frozen and on device.
    """
    upstream = Path(upstream_dir)
    return load_scorers(
        posenet_path=upstream / "models" / "posenet.safetensors",
        segnet_path=upstream / "models" / "segnet.safetensors",
        device=device,
        upstream_dir=upstream_dir,
    )


def load_differentiable_scorers(
    upstream_dir: str | Path,
    device: str | torch.device | None = None,
) -> tuple:
    """Load frozen scorers with differentiable PoseNet preprocessing.

    Combines load_default_scorers() + make_scorers_differentiable() into a single
    call. Use this whenever you need to backpropagate through PoseNet (TTO,
    constrained generation, sensitivity analysis, any gradient-based optimization).

    Without this, PoseNet gradients are ZERO due to upstream rgb_to_yuv6 having
    @torch.no_grad(). This convenience function eliminates the class of bugs where
    a script loads scorers but forgets to patch them.

    Args:
        upstream_dir: path to the upstream repository root.
        device: target device (auto-detected if None).

    Returns:
        (posenet, segnet) tuple, both frozen, on device, with differentiable preprocessing.
    """
    posenet, segnet = load_default_scorers(upstream_dir, device=device)
    make_scorers_differentiable(posenet, segnet)
    return posenet, segnet


def make_scorers_differentiable(posenet: torch.nn.Module, segnet: torch.nn.Module) -> None:
    """Patch frozen scorers for differentiable optimization (TTO, constrained gen).

    The upstream PoseNet.preprocess_input calls rgb_to_yuv6() which is decorated
    with @torch.no_grad(), killing all gradients through the YUV color space
    conversion. This function replaces it with a differentiable version that
    faithfully reproduces the upstream BT.601 math without the no_grad barrier.

    Also patches AllNorm.forward to use reshape() instead of view() for
    robustness with non-contiguous tensors from einops rearrange.

    CRITICAL: Without this patch, PoseNet gradients are ZERO during TTO.
    Every optimization that backprops through PoseNet MUST call this first.

    Extracted from training.py:_patch_scorers_for_training for reuse in
    TTO, constrained generation, and any gradient-based scorer optimization.

    References:
        - BT.601 YUV conversion: ITU-R BT.601-7
        - 4:2:0 chroma subsampling: average of 2x2 block
        - Bug discovered by skunkworks council adversarial review (2026-04-15)
    """
    import types

    import einops

    # Patch AllNorm to not break gradients on non-contiguous tensors
    for module in list(posenet.modules()) + list(segnet.modules()):
        if type(module).__name__ == "AllNorm":

            def _patched_forward(self, x):
                return self.bn(x.reshape(-1, 1)).reshape(x.shape)

            module.forward = types.MethodType(_patched_forward, module)

    # Differentiable rgb_to_yuv6: full-range BT.601 with 4:2:0 subsampling.
    # Matches upstream frame_utils.py rgb_to_yuv6 exactly, minus @torch.no_grad.
    def _rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
        H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
        H2, W2 = H // 2, W // 2
        rgb = rgb_chw[..., :, : 2 * H2, : 2 * W2]
        R = rgb[..., 0, :, :]
        G = rgb[..., 1, :, :]
        B = rgb[..., 2, :, :]
        # BT.601 full-range luma/chroma
        Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
        U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
        V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
        # 4:2:0 chroma subsampling (average of 2x2 block)
        U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
        V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
        y00 = Y[..., 0::2, 0::2]
        y10 = Y[..., 1::2, 0::2]
        y01 = Y[..., 0::2, 1::2]
        y11 = Y[..., 1::2, 1::2]
        return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)

    # PoseNet expects (B, 12, 192, 256) after preprocess: resize to scorer
    # input size, then YUV 4:2:0 conversion, then rearrange for 2-frame input.
    try:
        from modules import segnet_model_input_size
    except ImportError:
        segnet_model_input_size = (512, 384)  # (W, H) default

    def _diff_preprocess(self, x):
        batch_size, seq_len_local = x.shape[0], x.shape[1]
        x = einops.rearrange(x, "b t c h w -> (b t) c h w", b=batch_size, t=seq_len_local, c=3)
        x = torch.nn.functional.interpolate(
            x,
            size=(segnet_model_input_size[1], segnet_model_input_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        yuv = _rgb_to_yuv6_diff(x)
        return einops.rearrange(yuv, "(b t) c h w -> b (t c) h w", b=batch_size, t=seq_len_local, c=6).contiguous()

    posenet.preprocess_input = types.MethodType(_diff_preprocess, posenet)


def comma_score(pose_dist: float, seg_dist: float, rate: float) -> float:
    """Comma.ai video compression challenge score formula.

    score = 100 * segnet_distortion + sqrt(10 * posenet_distortion) + 25 * rate

    Lower is better.
    """
    return 100.0 * seg_dist + math.sqrt(10.0 * pose_dist) + 25.0 * rate


def compute_proxy_score(
    frames: torch.Tensor,
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    rate: float = 0.0,
    batch_size: int = 16,
    eval_roundtrip: bool = True,
) -> dict:
    """Compute proxy score matching the official scorer formula.

    Evaluates SegNet hard disagreement and PoseNet MSE on non-overlapping
    pairs, then combines them using the competition score formula.

    Args:
        frames: (N, H, W, 3) float tensor of candidate frames.
        gt_frames: list of (H, W, 3) uint8 GT frames.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        device: computation device.
        rate: rate term (archive_size / uncompressed_size).
        batch_size: pairs per forward pass.
        eval_roundtrip: if True, simulates the official scorer's
            resolution round-trip (384->874->384) for more faithful proxy.

    Returns:
        dict with score, pose, seg, rate, and per-term contributions.
    """
    import torch.nn.functional as F

    from tac.camera import CAMERA_H, CAMERA_W, SEGNET_INPUT_H, SEGNET_INPUT_W

    N = frames.shape[0]
    P = N // 2
    total_pose, total_seg, n_pairs = 0.0, 0.0, 0

    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)

        cand_pairs, gt_pairs = [], []
        for k in range(start, end):
            cand_pairs.append(torch.stack([frames[2 * k], frames[2 * k + 1]], dim=0))
            gt_pairs.append(torch.stack([
                gt_frames[2 * k].float(), gt_frames[2 * k + 1].float(),
            ], dim=0))

        cand_t = torch.stack(cand_pairs).to(device)
        gt_t = torch.stack(gt_pairs).to(device)

        cand_chw = cand_t.permute(0, 1, 4, 2, 3).contiguous()
        gt_chw = gt_t.permute(0, 1, 4, 2, 3).contiguous()

        B, T, C, H, W = cand_chw.shape
        if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
            cand_flat = cand_chw.reshape(B * T, C, H, W)
            gt_flat = gt_chw.reshape(B * T, C, H, W)
            cand_flat = F.interpolate(
                cand_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear", align_corners=False,
            )
            gt_flat = F.interpolate(
                gt_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear", align_corners=False,
            )
            cand_chw = cand_flat.reshape(B, T, C, SEGNET_INPUT_H, SEGNET_INPUT_W)
            gt_chw = gt_flat.reshape(B, T, C, SEGNET_INPUT_H, SEGNET_INPUT_W)

        cand_chw = cand_chw.round().clamp(0, 255)

        if eval_roundtrip:
            flat = cand_chw.reshape(-1, *cand_chw.shape[2:])
            flat = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W),
                mode="bilinear", align_corners=False,
            )
            flat = flat.round().clamp(0, 255)
            flat = F.interpolate(
                flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear", align_corners=False,
            )
            cand_chw = flat.reshape(B, T, *flat.shape[1:])

        with torch.no_grad():
            fp_in = posenet.preprocess_input(cand_chw)
            gp_in = posenet.preprocess_input(gt_chw)
            fp_out = posenet(fp_in)
            gp_out = posenet(gp_in)
            pose_mse = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean(dim=-1)
            total_pose += pose_mse.sum().item()

            fs_in = segnet.preprocess_input(cand_chw)
            gs_in = segnet.preprocess_input(gt_chw)
            fs_out = segnet(fs_in)
            gs_out = segnet(gs_in)
            diff = (fs_out.argmax(dim=1) != gs_out.argmax(dim=1)).float()
            seg_disagree = diff.mean(dim=tuple(range(1, diff.ndim)))
            total_seg += seg_disagree.sum().item()
            n_pairs += B

    avg_pose = total_pose / max(n_pairs, 1)
    avg_seg = total_seg / max(n_pairs, 1)
    pose_term = math.sqrt(max(0.0, 10.0 * avg_pose))
    score = 100.0 * avg_seg + pose_term + 25.0 * rate

    return {
        "score": score,
        "pose": avg_pose,
        "seg": avg_seg,
        "rate": rate,
        "pose_contribution": pose_term,
        "seg_contribution": 100.0 * avg_seg,
        "rate_contribution": 25.0 * rate,
        "n_pairs": n_pairs,
    }


def extract_gt_masks(
    gt_frames: list[torch.Tensor],
    segnet: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Extract SegNet argmax masks from GT frames.

    Args:
        gt_frames: list of (H, W, 3) uint8 tensors.
        segnet: frozen SegNet model.
        device: computation device.
        batch_size: frames per forward pass.

    Returns:
        (N, seg_H, seg_W) long tensor of class indices.
    """
    import torch.nn.functional as F

    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    masks = []
    for i in range(0, len(gt_frames), batch_size):
        batch = gt_frames[i:i + batch_size]
        frames_t = torch.stack(batch).float().to(device)
        frames_chw = frames_t.permute(0, 3, 1, 2).contiguous()

        _, _, H, W = frames_chw.shape
        if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
            frames_chw = F.interpolate(
                frames_chw, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear", align_corners=False,
            )

        seg_in_btchw = frames_chw.unsqueeze(1)
        seg_in = segnet.preprocess_input(seg_in_btchw)
        with torch.no_grad():
            seg_out = segnet(seg_in)
        mask = seg_out.argmax(dim=1)
        masks.append(mask.cpu())

    return torch.cat(masks, dim=0).long()


def extract_gt_pose_targets(
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Extract PoseNet targets from GT frames using non-overlapping pairs.

    For N frames, produces N//2 pose targets: pair(0,1), pair(2,3), ...

    Args:
        gt_frames: list of (H, W, 3) uint8 tensors.
        posenet: frozen PoseNet model.
        device: computation device.
        batch_size: pairs per forward pass.

    Returns:
        (P, 6) float tensor of pose targets, P = len(gt_frames) // 2.
    """
    import torch.nn.functional as F

    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    N = len(gt_frames)
    P = N // 2
    targets = []

    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        batch_pairs = []
        for k in range(start, end):
            f0 = gt_frames[2 * k].float()
            f1 = gt_frames[2 * k + 1].float()
            pair = torch.stack([f0, f1], dim=0)
            batch_pairs.append(pair)

        pairs = torch.stack(batch_pairs).to(device)
        pairs_chw = pairs.permute(0, 1, 4, 2, 3).contiguous()

        B, T, C, H, W = pairs_chw.shape
        if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
            pairs_flat = pairs_chw.reshape(B * T, C, H, W)
            pairs_flat = F.interpolate(
                pairs_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear", align_corners=False,
            )
            pairs_chw = pairs_flat.reshape(B, T, C, SEGNET_INPUT_H, SEGNET_INPUT_W)

        posenet_in = posenet.preprocess_input(pairs_chw)
        with torch.no_grad():
            posenet_out = posenet(posenet_in)
        pose = posenet_out["pose"][..., :6]
        targets.append(pose.cpu())

    return torch.cat(targets, dim=0).float()


def score_sensitivity(pose_dist: float) -> dict[str, float]:
    """Compute marginal sensitivities at the current operating point.

    Returns dict with d(score)/d(seg), d(score)/d(pose), d(score)/d(rate),
    and the leverage ratio seg/pose.
    """
    d_seg = 100.0
    d_pose = math.sqrt(10.0) / (2.0 * math.sqrt(pose_dist)) if pose_dist > 0 else float("inf")
    d_rate = 25.0  # d(score)/d(rate) = 25 from the formula: score = ... + 25 * rate
    return {
        "d_score_d_seg": d_seg,
        "d_score_d_pose": d_pose,
        "d_score_d_rate": d_rate,
        "seg_pose_leverage": d_seg / d_pose if d_pose > 0 else float("inf"),
    }
