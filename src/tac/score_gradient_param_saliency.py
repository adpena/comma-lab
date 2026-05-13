"""Parameter-space score-gradient saliency for HNeRV-class decoders.

Per Track 4 reactivation criterion #1 (memory:
``feedback_track_4_uniward_stc_hessian_a1_landed_20260509.md``): replace the
``mean(theta^2)`` Fisher proxy with the **true** score-gradient saliency
``S(theta_i) = E_{(x,y)} [|d(score)/d(theta_i)|^2]`` computed via autograd
hooks on ``upstream/evaluate.py``'s SegNet+PoseNet.

Why this matters
----------------
A1 was trained with score-gradient supervision, so ``mean(theta^2)`` is
**anti-correlated** with score saliency on this substrate (parameters with
large magnitude are exactly the ones the score-gradient pushed AWAY from
zero because they were score-relevant). Coarsening "low-Fisher-by-mean(theta^2)"
tensors hits the directions the score-gradient training identified as
score-relevant in the orthogonal sense. Track 4 v1's best candidate lost
+0.0058 score [contest-CPU GHA] proving the proxy is wrong on this substrate.

The right per-parameter importance is the **score-gradient Fisher diagonal**::

    S(theta_i) = E_{(x,y)} [(d(L_score(x,y; theta)) / d(theta_i))^2]

where ``L_score`` is a differentiable surrogate of the contest score:

    L_score = lambda_seg * KL_distill(SegNet(decoder(z))||SegNet(GT)) +
              lambda_pose * MSE(PoseNet(decoder(z))[..., :6], PoseNet(GT)[..., :6])

This is the canonical Cramer-Rao Fisher information for the joint score axis
(NOT the upstream argmax-distortion which is non-differentiable; we use the
KL-distill surrogate per ``tac.losses.kl_distill_segnet_only``).

Output
------
``compute_score_gradient_param_saliency()`` returns a ``dict[str, float]`` with
one scalar per state-dict tensor: the per-tensor MEAN of squared parameter
gradients aggregated over all sampled pairs. Use this dict the same way
``build_uniward_stc_hessian_a1_v1.compute_fisher_proxy()`` is used (drop-in
replacement; the only difference is **mathematical correctness**).

Tag
---
The saliency dict itself is not a score claim; it is a per-tensor priority
heuristic. CPU computation on M5 Max is tagged ``[saliency-prior]`` (research
signal). The downstream archive byte-anchor is independent and gets its own
``[contest-CPU]`` / ``[contest-CUDA]`` tag from the actual auth eval.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

DEFAULT_CPU_POSE_OPERATING_POINT = 3.0e-5


def pose_distortion_score_derivative(
    pose_distortion: float = DEFAULT_CPU_POSE_OPERATING_POINT,
    *,
    eps: float = 1.0e-12,
) -> float:
    """Return ``d sqrt(10*d_pose) / d d_pose`` at ``pose_distortion``.

    This is the local contest-score marginal for pose distortion. At the
    medal-band CPU floor (roughly 3e-5), the derivative is about 289, not
    about 1. Keeping this explicit prevents score-gradient saliency from
    silently underweighting the pose axis.
    """
    d = max(float(pose_distortion), float(eps))
    return float(np.sqrt(10.0) / (2.0 * np.sqrt(d)))


DEFAULT_CPU_POSE_SCORE_WEIGHT = pose_distortion_score_derivative()


def _add_repo_paths(repo_root: Path) -> None:
    """Ensure both ``upstream`` and ``src`` are importable.

    The PoseNet/SegNet/frame_utils trio ships in ``upstream/``; adding it to
    ``sys.path`` mirrors the upstream evaluator's own bootstrap.
    """
    for candidate in (repo_root / "upstream", repo_root / "src"):
        s = str(candidate)
        if s not in sys.path:
            sys.path.insert(0, s)


def _read_pairs_from_video(
    video_path: Path,
    n_pairs: int,
    *,
    eval_h: int = 384,
    eval_w: int = 512,
    camera_h: int = 874,
    camera_w: int = 1164,
) -> torch.Tensor:
    """Decode the first ``2 * n_pairs`` frames from ``video_path``.

    Returns a uint8 tensor with shape ``(n_pairs, 2, camera_h, camera_w, 3)``
    matching the upstream evaluator's ``(B, seq_len=2, H, W, C)`` layout. The
    raw frames stay at camera resolution; the caller is responsible for any
    resize / colorspace conversion.
    """
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")
    try:
        import av
    except ImportError as exc:  # pragma: no cover - dependency failure surfaces in caller
        raise RuntimeError(
            "PyAV is required for score-gradient saliency. Install with "
            "`uv pip install av`."
        ) from exc
    from frame_utils import yuv420_to_rgb  # type: ignore[import-not-found]

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        frames: list[torch.Tensor] = []
        for frame in container.decode(stream):
            frames.append(yuv420_to_rgb(frame))
            if len(frames) >= 2 * n_pairs:
                break
    finally:
        container.close()

    if len(frames) < 2 * n_pairs:
        raise RuntimeError(
            f"Video {video_path} has only {len(frames)} decoded frames; need "
            f"{2 * n_pairs} for n_pairs={n_pairs}."
        )

    pairs = []
    for i in range(n_pairs):
        f0 = frames[2 * i]
        f1 = frames[2 * i + 1]
        if f0.shape[:2] != (camera_h, camera_w) or f1.shape[:2] != (camera_h, camera_w):
            raise RuntimeError(
                f"Frame {i} shape {f0.shape} != expected ({camera_h}, {camera_w}, 3)"
            )
        pairs.append(torch.stack([f0, f1], dim=0))
    return torch.stack(pairs, dim=0)  # (n_pairs, 2, H, W, 3) uint8


def _decode_to_pairs(
    decoder: torch.nn.Module,
    latents: torch.Tensor,
    *,
    eval_h: int,
    eval_w: int,
    camera_h: int,
    camera_w: int,
    batch_size: int,
) -> torch.Tensor:
    """Run the HNeRV decoder over ``latents``, upsample to camera resolution.

    Mirrors the inflate.py post-decode pipeline (bicubic upsample +
    per-frame channel offsets) but stays in float32 + DIFFERENTIABLE so the
    SegNet/PoseNet backward pass can flow back through the decoder.

    Returns a tensor with shape ``(N, 2, camera_h, camera_w, 3)`` in [0, 255].
    """
    n = latents.shape[0]
    out_chunks: list[torch.Tensor] = []
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        batch = j - i
        decoded = decoder(latents[i:j])  # (batch, 2, 3, eval_h, eval_w)
        flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
        up = F.interpolate(flat, size=(camera_h, camera_w), mode="bicubic", align_corners=False)
        up = up.reshape(batch, 2, 3, camera_h, camera_w)
        # inflate.py applies channel offsets to mimic inflated raw uint8 path:
        #   up[:, 0, 0].sub_(1.0); up[:, 0, 2].sub_(1.0); up[:, 1, 1].sub_(1.0)
        # We replicate that mathematically without in-place ops to keep the
        # autograd graph clean for backward.
        offsets = torch.zeros_like(up)
        offsets[:, 0, 0, :, :] = -1.0
        offsets[:, 0, 2, :, :] = -1.0
        offsets[:, 1, 1, :, :] = -1.0
        up = up + offsets
        # Channels-last to match (B, S, H, W, C) eval/scorer layout.
        up = up.permute(0, 1, 3, 4, 2).clamp(0, 255)
        out_chunks.append(up)
    return torch.cat(out_chunks, dim=0)


def _surrogate_score_loss(
    pred_pairs: torch.Tensor,
    gt_pairs: torch.Tensor,
    *,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    seg_distill_temperature: float,
    lambda_seg: float,
    lambda_pose: float,
) -> torch.Tensor:
    """Differentiable surrogate of the contest score over a batch of pairs.

    SegNet's ``compute_distortion`` uses ``argmax`` (non-differentiable);
    we replace it with a Hinton-style KL distillation between SegNet logits
    of the predicted reconstruction and SegNet logits on the GT pair (the
    canonical surrogate per ``tac.losses.kl_distill_segnet_only`` and
    council-Round-2 finding 10 in
    ``experiments/train_score_gradient_pr101_finetune.py``).

    PoseNet's ``compute_distortion`` is already differentiable (MSE on first 6
    pose dims); we use exactly that.
    """
    pred_in = pred_pairs.permute(0, 1, 4, 2, 3).float()  # (B, 2, 3, H, W)
    gt_in = gt_pairs.permute(0, 1, 4, 2, 3).float()

    # PoseNet path — preprocess + forward + MSE on first 6 pose dims.
    pose_pred_in = posenet.preprocess_input(pred_in)
    pose_gt_in = posenet.preprocess_input(gt_in.detach())
    pose_pred_out = posenet(pose_pred_in)["pose"]
    with torch.no_grad():
        pose_gt_out = posenet(pose_gt_in)["pose"]
    pose_loss = (pose_pred_out[..., :6] - pose_gt_out[..., :6]).pow(2).mean()

    # SegNet path — Hinton KL distill on logits at the canonical T.
    seg_pred_in = segnet.preprocess_input(pred_in)
    seg_gt_in = segnet.preprocess_input(gt_in.detach())
    seg_pred_logits = segnet(seg_pred_in)
    with torch.no_grad():
        seg_gt_logits = segnet(seg_gt_in)
    T = float(seg_distill_temperature)
    seg_loss = (
        F.kl_div(
            F.log_softmax(seg_pred_logits / T, dim=1),
            F.softmax(seg_gt_logits / T, dim=1),
            reduction="none",
        )
        .sum(dim=1)
        .mean()
        * (T * T)
    )

    return lambda_seg * seg_loss + lambda_pose * pose_loss


def compute_score_gradient_param_saliency(
    *,
    decoder: torch.nn.Module,
    latents: torch.Tensor,
    gt_pairs_uint8: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device | str = "cpu",
    eval_h: int = 384,
    eval_w: int = 512,
    camera_h: int = 874,
    camera_w: int = 1164,
    forward_batch_size: int = 8,
    saliency_batch_size: int = 4,
    seg_distill_temperature: float = 2.0,
    lambda_seg: float = 100.0,
    lambda_pose: float = DEFAULT_CPU_POSE_SCORE_WEIGHT,
    progress: bool = False,
) -> dict[str, float]:
    """Compute per-tensor ``E[|d(score)/d(theta)|^2]`` via autograd.

    Streams the latent batches through the decoder + scorers, accumulating
    parameter-gradient squared sums per tensor, then returns the per-tensor
    mean (sum / n_params / n_samples). Returned scalars are positive floats
    in arbitrary units; relative ranking is the load-bearing signal.

    Parameters
    ----------
    decoder
        HNeRVDecoder-class module on ``device`` with ``requires_grad=True``
        on every parameter the caller wants ranked.
    latents
        Latent tensor from the A1 archive (e.g. shape ``(600, 28)``). Must be
        on ``device``.
    gt_pairs_uint8
        Ground-truth pairs ``(N, 2, camera_h, camera_w, 3)`` uint8. Must
        match ``latents.shape[0]`` along dim 0.
    posenet, segnet
        Differentiable scorers from ``tac.scorer.load_differentiable_scorers``.
        Their parameters must have ``requires_grad=False``.
    forward_batch_size
        Inner-loop decoder batch when accumulating gradients.
    saliency_batch_size
        Progress/checkpoint chunk size. The Fisher diagonal itself is always
        accumulated with one backward pass per sample so gradients are squared
        before any cross-sample averaging can cancel them.
    seg_distill_temperature
        T in the KL distill surrogate (canonical T=2 per Hinton 2014 +
        council Round 2 finding 10).
    lambda_seg, lambda_pose
        Score-axis weights. Default mirrors the contest score:
        ``score = 100 * seg_distort + sqrt(10 * pose_distort) + 25 * rate``.
        Around the A1 operating point (seg_avg ~0.001, pose_avg ~3e-5), the
        marginal d(score)/d(seg_distort) = 100 and
        d(score)/d(pose_distort) = sqrt(10) / (2 * sqrt(pose_distort)) ≈ 289.
        The default pose coefficient uses that local derivative. Callers may
        override it only when they are deliberately measuring a normalized
        surrogate rather than the local contest-score axis.
    progress
        If True, print a per-batch progress line to stderr.

    Returns
    -------
    dict[str, float]
        ``{name: per_tensor_mean_squared_gradient}`` — one entry per
        decoder parameter that has ``requires_grad=True``.
    """
    if latents.shape[0] != gt_pairs_uint8.shape[0]:
        raise ValueError(
            f"latents.shape[0]={latents.shape[0]} != "
            f"gt_pairs_uint8.shape[0]={gt_pairs_uint8.shape[0]}"
        )
    if saliency_batch_size <= 0:
        raise ValueError(f"saliency_batch_size must be positive, got {saliency_batch_size}")
    if forward_batch_size <= 0:
        raise ValueError(f"forward_batch_size must be positive, got {forward_batch_size}")

    device = torch.device(device) if isinstance(device, str) else device
    latents = latents.to(device).detach()  # detach so grad doesn't accumulate on latents
    gt_pairs_uint8 = gt_pairs_uint8.to(device)

    # Make sure decoder params require grad; collect for accumulation.
    grad_sq_sum: dict[str, torch.Tensor] = {}
    grad_n_samples: dict[str, int] = {}
    for name, param in decoder.named_parameters():
        param.requires_grad_(True)
        grad_sq_sum[name] = torch.zeros_like(param.detach(), device=device)
        grad_n_samples[name] = 0

    # Freeze scorer params.
    for p in posenet.parameters():
        p.requires_grad_(False)
    for p in segnet.parameters():
        p.requires_grad_(False)

    n = latents.shape[0]
    n_batches = (n + saliency_batch_size - 1) // saliency_batch_size
    for b_idx, i in enumerate(range(0, n, saliency_batch_size)):
        j = min(i + saliency_batch_size, n)
        chunk_loss_sum = 0.0

        # True Fisher diagonal requires E[g_i^2], not (E[g_i])^2. A single
        # backward over a batch squares the batch-averaged gradient and can
        # cancel opposite-signed sample gradients before squaring. Accumulate
        # one sample at a time; `saliency_batch_size` remains a progress chunk
        # size, not a mathematical batching parameter.
        for sample_idx in range(i, j):
            sample_latents = latents[sample_idx : sample_idx + 1]
            sample_gt = gt_pairs_uint8[sample_idx : sample_idx + 1].float()

            pred_pairs = _decode_to_pairs(
                decoder,
                sample_latents,
                eval_h=eval_h,
                eval_w=eval_w,
                camera_h=camera_h,
                camera_w=camera_w,
                batch_size=min(forward_batch_size, 1),
            )

            loss = _surrogate_score_loss(
                pred_pairs,
                sample_gt,
                posenet=posenet,
                segnet=segnet,
                seg_distill_temperature=seg_distill_temperature,
                lambda_seg=lambda_seg,
                lambda_pose=lambda_pose,
            )

            decoder.zero_grad(set_to_none=True)
            loss.backward()
            chunk_loss_sum += float(loss.item())

            with torch.no_grad():
                for name, param in decoder.named_parameters():
                    if param.grad is None:
                        continue
                    grad_sq_sum[name] += param.grad.detach().pow(2)
                    grad_n_samples[name] += 1

        if progress:
            chunk_n = max(j - i, 1)
            print(
                f"  [score-grad-saliency] batch {b_idx + 1}/{n_batches} "
                f"mean_loss={chunk_loss_sum / chunk_n:.6f} pairs={i}-{j}",
                file=sys.stderr,
                flush=True,
            )

    saliency: dict[str, float] = {}
    for name in grad_sq_sum:
        n_params = grad_sq_sum[name].numel()
        n_samples = max(grad_n_samples[name], 1)
        # Per-parameter mean squared gradient, then per-tensor mean. This is
        # the per-tensor analogue of E[(dL/dtheta_i)^2] needed by the bit
        # allocator (which receives ONE saliency scalar per tensor).
        per_param_mean_sq = grad_sq_sum[name].sum().item() / (n_params * n_samples)
        saliency[name] = float(per_param_mean_sq)
    decoder.zero_grad(set_to_none=True)
    return saliency


def build_score_gradient_saliency_for_a1_archive(
    *,
    decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    repo_root: Path,
    device: str = "cpu",
    n_pairs: int | None = None,
    forward_batch_size: int = 8,
    saliency_batch_size: int = 4,
    progress: bool = False,
) -> dict[str, float]:
    """High-level convenience: build decoder + load scorers + compute saliency.

    Loads PoseNet/SegNet from ``repo_root/upstream/models/`` and decodes the
    first ``2 * n_pairs`` frames from ``repo_root/upstream/videos/0.mkv`` as
    GT. ``n_pairs`` defaults to ``min(latents.shape[0], 600)``.
    """
    _add_repo_paths(repo_root)
    from tac.scorer import load_differentiable_scorers

    n_pairs = int(n_pairs if n_pairs is not None else min(latents.shape[0], 600))
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")

    # Build the decoder fresh so its parameter names exactly match the
    # state-dict keys (which is also what the bit allocator expects). Import
    # the A1 runtime model by absolute file path so a generic ``model`` module
    # never leaks into or shadows the caller's import graph.
    model_path = repo_root / (
        "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
        "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/submission_dir/src/model.py"
    )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"A1 HNeRVDecoder source missing for score-gradient saliency: {model_path}"
        )
    spec = importlib.util.spec_from_file_location("_tac_a1_score_gradient_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import A1 HNeRVDecoder from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        HNeRVDecoder = module.HNeRVDecoder
    except AttributeError as exc:
        raise ImportError(f"{model_path} does not define HNeRVDecoder") from exc

    decoder = HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=(384, 512))
    decoder.load_state_dict(decoder_state_dict)
    decoder = decoder.to(device).train()

    posenet, segnet = load_differentiable_scorers(repo_root / "upstream", device=device)
    posenet.eval()
    segnet.eval()

    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    if not video_path.is_file():
        raise FileNotFoundError(
            f"GT video for score-gradient saliency missing: {video_path}. "
            f"Run upstream/download_and_remux.sh first."
        )
    gt_pairs = _read_pairs_from_video(video_path, n_pairs).to(device)

    return compute_score_gradient_param_saliency(
        decoder=decoder,
        latents=latents[:n_pairs],
        gt_pairs_uint8=gt_pairs,
        posenet=posenet,
        segnet=segnet,
        device=device,
        forward_batch_size=forward_batch_size,
        saliency_batch_size=saliency_batch_size,
        progress=progress,
    )
