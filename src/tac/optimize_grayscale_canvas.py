"""Lane AL — Analog Latent canvas optimizer.

EUREKA insight (gpt-5.5 grand council 2026-04-29 11am):
    The grayscale mask is NOT compressed argmax segmentation. It is a 1-channel
    ANALOG LATENT CANVAS whose values feed a Gaussian-softmax LUT into the
    SegMap / MaskRenderer pipeline. The boundary pixels are SOFT class
    probabilities — optimize the gray values directly so those soft
    probabilities produce a better SegNet/PoseNet score, while AV1 still sees a
    smooth monochrome video with no rate cost.

What this module provides
-------------------------
``optimize_grayscale_canvas`` — SGD loop over a per-pixel grayscale latent that:

* Initializes from Lane MM's class-argmax encoding (so we start at Lane MM's
  baseline).
* Defines ``gray_logits`` as an ``nn.Parameter`` (real-valued); the forward
  uses a clamped sigmoid mapping to [0, 255] with a straight-through
  estimator (STE) so backprop sees the smooth gray-value path even though
  the renderer only sees integer pixels.
* Forward chain:

      gray_logits  ─sigmoid·255─►  gray_continuous  ─STE round─►  gray_int
                                              │
                                              └───►  Gaussian softmax LUT
                                                     (sigma=15) on gray_continuous
                                                     ► (B, 5, H, W) soft class probs
                                                     ► expectation over class
                                                       embeddings (= soft mask)
                                                     ► frozen renderer (Lane A)
                                                     ► (B, T, H, W, 3) RGB pair
                                                     ► eval_roundtrip + noise_std
                                                     ► frozen PoseNet + SegNet
                                                     ► loss = 100·seg_dist
                                                       + sqrt(10·pose_dist)

* No rate term in the loss (AV1 + 1-channel grayscale layout holds rate
  constant by construction; Lane AL keeps the same archive layout as
  Lane MM).
* Adam optimizer; lr=1e-2 by default; ~100-300 steps.

CLAUDE.md compliance
--------------------
* ``eval_roundtrip=True`` is the default; setting False raises (matches the
  CLAUDE.md non-negotiable that every TTO/training path must roundtrip).
* ``noise_std=0.5`` default — the mid-point of the canonical noise schedule.
* CUDA-required default — passing ``device='mps'`` raises (forbidden default
  trap).
* Optimizes a COMPRESS-TIME artifact only; no scorer is loaded at inflate
  time. ``gray_int`` is what gets shipped through ffmpeg ► grayscale.mkv.
* The renderer is FROZEN (loaded from Lane A's renderer.bin); only
  ``gray_logits`` receives gradient.
* Returns the final ``gray_int`` (uint8) along with per-step proxy metrics
  for the lane script's heartbeat / smoke-eval cadence.

Reuses (does not modify):
    tac.mask_grayscale_lut.create_gaussian_softmax_lut + CLASS_TO_GRAY
    tac.scorer.load_differentiable_scorers
    tac.losses.scorer_forward_pair
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.losses import scorer_forward_pair
from tac.mask_grayscale_lut import (
    CLASS_TO_GRAY,
    LUT_DEFAULT_SIGMA,
    NUM_CLASSES,
    encode_masks_grayscale,
)
from tac.preflight import PreflightError


# ── module constants (mirror submissions/robust_current/inflate_renderer.py) ──
SCORER_INPUT_H = 384
SCORER_INPUT_W = 512
CAMERA_H = 874
CAMERA_W = 1164


@dataclass
class OptimizeConfig:
    """Knobs for the Lane AL SGD loop.

    Attributes:
        steps: Number of Adam steps over the full frame stack.
        lr: Adam learning rate on ``gray_logits``.
        sigma: Gaussian-LUT temperature (matches Lane MM default).
        noise_std: Roundtrip noise scale used by the proxy. ``0.5`` matches
            the canonical CLAUDE.md non-negotiable.
        eval_roundtrip: If False, raises — present so the caller has to be
            explicit about violating the non-negotiable.
        seg_weight: Multiplier on SegNet distortion in the loss. Default
            100 matches the contest formula at 1:1.
        pose_weight: Multiplier inside the sqrt term on PoseNet distortion.
            Default 10 matches the contest formula.
        batch_size: Number of pose-pairs per Adam step. Each pair is two
            consecutive frames; total frames per step = 2·batch_size.
        log_every: Emit a metric snapshot every N steps.
        smoke_eval_callback: Optional callable invoked with
            ``(step, gray_int_uint8_tensor, metrics_dict)`` every
            ``smoke_eval_every`` steps. The callback may write the gray
            tensor to disk and trigger an out-of-process auth eval.
        smoke_eval_every: Cadence for ``smoke_eval_callback`` in steps.
            Defaults to 0 (disabled — the lane script uses the in-process
            smoke at the end instead).
        seed: RNG seed for the noise injection during eval_roundtrip.
    """

    steps: int = 200
    lr: float = 1e-2
    sigma: float = LUT_DEFAULT_SIGMA
    noise_std: float = 0.5
    eval_roundtrip: bool = True
    seg_weight: float = 100.0
    pose_weight: float = 10.0
    batch_size: int = 8
    log_every: int = 10
    smoke_eval_callback: Optional[Callable[[int, torch.Tensor, dict], None]] = None
    smoke_eval_every: int = 0
    seed: int = 0xA1A1


def _validate_device(device: torch.device | str) -> torch.device:
    """CUDA-required default; reject MPS per CLAUDE.md non-negotiable."""
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "mps":
        raise PreflightError(
            "Lane AL refuses device='mps' — MPS auth/proxy gap is 23x for "
            "PoseNet (CLAUDE.md non-negotiable, 2026-04-25 measurement)."
        )
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise PreflightError(
            "Lane AL requires CUDA; torch.cuda.is_available() is False. "
            # `--device cpu` is permitted ONLY for smoke tests where "
            # deterministic-bytes acceptable (i.e., the archive is not
            # being shipped). Production lane scripts always pin CUDA.
            "Use --device cpu only when deterministic-bytes acceptable "
            "(smoke tests; the contest archive is NOT being shipped)."
        )
    return dev


def _gaussian_softmax_soft(
    gray_continuous: torch.Tensor,
    sigma: float,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Gaussian softmax LUT in the gray-value direction.

    Mirrors ``tac.mask_grayscale_lut.create_gaussian_softmax_lut`` but operates
    on a real-valued gray tensor instead of integer indices. The LUT is the
    256-row softmax over class targets at integer gray values; we evaluate the
    same formula at fractional gray values so gradients can flow.

    Args:
        gray_continuous: (..., H, W) float tensor in [0, 255].
        sigma: Gaussian temperature.
        targets: (NUM_CLASSES,) float tensor of class gray targets, on the
            same device/dtype as ``gray_continuous``.

    Returns:
        (..., NUM_CLASSES, H, W) soft probability tensor.
    """
    # broadcast: gray_continuous (..., H, W) → (..., 1, H, W); targets (5,) →
    # (5, 1, 1). distance shape (..., 5, H, W).
    g = gray_continuous.unsqueeze(-3)
    t = targets.view(-1, 1, 1)
    sq_dist = (g - t) ** 2
    bell = torch.exp(-sq_dist / (2.0 * sigma * sigma))
    return F.softmax(bell, dim=-3)


def _soft_embedding_lookup(
    soft_probs: torch.Tensor,
    embedding: nn.Embedding,
) -> torch.Tensor:
    """Compute expectation over class embeddings.

    embedding(class_id) gives a (embed_dim,) vector per class. The soft
    analogue is ``sum_c p_c · embedding.weight[c]`` per pixel.

    Args:
        soft_probs: (B, NUM_CLASSES, H, W) soft probabilities (sum_c=1).
        embedding: nn.Embedding(num_classes, embed_dim) — frozen.

    Returns:
        (B, embed_dim, H, W) soft embedding tensor.
    """
    # embedding.weight: (num_classes, embed_dim).
    weight = embedding.weight  # (C, D)
    # einsum: (B, C, H, W) × (C, D) → (B, D, H, W)
    return torch.einsum("bchw,cd->bdhw", soft_probs, weight)


class _STERoundClamp(torch.autograd.Function):
    """Round + clamp to [0, 255] in forward; identity gradient in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round().clamp(0.0, 255.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


def _ste_round_clamp(x: torch.Tensor) -> torch.Tensor:
    return _STERoundClamp.apply(x)


class _STEClamp(torch.autograd.Function):
    """Clamp to [0, 255] in forward; identity gradient in backward.

    Without STE, sigmoid-based parameterization saturates at the
    Lane MM initialization (targets {0, 64, 128, 192, 255} put most
    pixels at the saturating ends, where sigmoid' ≈ 0 and gradient
    vanishes). Operating directly in gray space + STE clamp gives
    Adam a clean signal: the contest-canonical pixel range [0, 255]
    is small enough that lr=1e-2 produces ~0.1 px/step updates.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 255.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


def _ste_clamp(x: torch.Tensor) -> torch.Tensor:
    return _STEClamp.apply(x)


def _gray_logits_to_continuous(gray_logits: torch.Tensor) -> torch.Tensor:
    """STE-clamped identity → gray_continuous ∈ [0, 255] in forward.

    The parameter is operated on directly in gray-value space (no
    sigmoid); STE keeps the clamp invisible to the optimizer so the
    Lane MM initialization (saturating ends 0 / 255) does not zero
    out the gradient.
    """
    return _ste_clamp(gray_logits)


def initialize_gray_logits(
    class_ids: torch.Tensor,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Build the SGD parameter from Lane MM's class-argmax encoding.

    The parameter lives directly in gray-value space [0, 255]. The
    initial forward pass reproduces Lane MM's canonical grayscale
    targets exactly (no sigmoid roundoff).

    Args:
        class_ids: int64 (N, H, W) tensor of class indices.
        device: target device for the resulting parameter.

    Returns:
        float32 (N, H, W) ``nn.Parameter`` that can be passed straight to
        an Adam optimizer.
    """
    if class_ids.dtype != torch.int64:
        raise ValueError(
            f"class_ids must be int64, got {class_ids.dtype}. "
            "Cast with .long() if needed."
        )
    gray_uint8 = encode_masks_grayscale(class_ids)  # (N, H, W) uint8
    init = gray_uint8.to(torch.float32)  # exact Lane MM target gray values
    return nn.Parameter(init.to(device).contiguous())


def _eval_roundtrip_with_noise(
    rgb_btchw: torch.Tensor,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Match the canonical roundtrip used in compute_proxy_score.

    rgb at (B, T, 3, H_scorer, W_scorer) → upsample to camera res →
    add noise (modeling AV1 quantization) → clamp/round →
    downsample back to scorer res. This is the SAME schema used in
    ``tac.scorer.compute_proxy_score`` with eval_roundtrip=True.
    """
    B, T, C, H, W = rgb_btchw.shape
    flat = rgb_btchw.reshape(B * T, C, H, W)
    flat = F.interpolate(
        flat, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False,
    )
    if noise_std > 0:
        noise = torch.empty_like(flat).normal_(
            mean=0.0, std=noise_std, generator=generator,
        )
        flat = flat + noise
    # CRITICAL: bare .round() has ZERO gradient → severs backprop chain →
    # silent training freeze. Lane DARTS-S V1 burned 5h on Vast.ai 4090 with
    # this exact bug at src/tac/segmap_renderer.py:281. Use Uint8STE for
    # proper STE behavior (forward = clamp+round, backward = identity).
    # Council A 2026-04-29 PM, .omx/research/council_darts_s_freeze_audit.
    from tac.quantization import Uint8STE
    flat = Uint8STE.apply(flat)
    flat = F.interpolate(
        flat, size=(SCORER_INPUT_H, SCORER_INPUT_W),
        mode="bilinear", align_corners=False,
    )
    return flat.reshape(B, T, C, SCORER_INPUT_H, SCORER_INPUT_W)


def _render_pair_from_logits(
    gray_logits_pair: torch.Tensor,
    sigma: float,
    targets: torch.Tensor,
    embedding: nn.Embedding,
    renderer_forward_from_embedding: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Forward through the soft-embedding path → renderer → (B, 3, H, W).

    Args:
        gray_logits_pair: (B, T, H, W) float — the SGD parameter slice.
        sigma: Gaussian-LUT temperature.
        targets: (NUM_CLASSES,) float on the same device as logits.
        embedding: frozen renderer embedding (Lane A's MaskRenderer.embedding).
        renderer_forward_from_embedding: callable that accepts a soft
            embedding tensor of shape (B*T, embed_dim, H, W) and returns
            (B*T, 3, H, W) RGB. Used to monkey-patch the discrete-only
            renderer with the soft path. Caller wires this to the renderer
            being trained (frozen weights, soft input).

    Returns:
        (B, T, 3, H, W) float in [0, 255].
    """
    B, T, H, W = gray_logits_pair.shape
    gray_continuous = _gray_logits_to_continuous(gray_logits_pair)  # (B,T,H,W)
    soft_probs = _gaussian_softmax_soft(
        gray_continuous, sigma=sigma, targets=targets,
    )  # (B, T, NUM_CLASSES, H, W)
    soft_probs = soft_probs.reshape(B * T, NUM_CLASSES, H, W)
    soft_embed = _soft_embedding_lookup(soft_probs, embedding)
    rgb = renderer_forward_from_embedding(soft_embed)  # (B*T, 3, H, W)
    return rgb.reshape(B, T, 3, H, W)


def optimize_grayscale_canvas(
    init_class_ids: torch.Tensor,
    gt_pairs_btchw: torch.Tensor,
    embedding: nn.Embedding,
    renderer_forward_from_embedding: Callable[[torch.Tensor], torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
    cfg: OptimizeConfig,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, list[dict]]:
    """Optimize per-pixel grayscale latent against the contest scorer.

    Args:
        init_class_ids: int64 (N, H, W) tensor of starting class ids.
            ``N`` is the total number of mask frames (e.g., 1200).
            Initialized from the Lane MM class-argmax encoding.
        gt_pairs_btchw: float (P, 2, 3, H, W) GT pair tensor in [0, 255]
            at SCORER_INPUT_H × SCORER_INPUT_W resolution. P = N // 2 with
            non-overlapping pairing (matches upstream evaluate.py).
        embedding: frozen nn.Embedding from the renderer.
        renderer_forward_from_embedding: see ``_render_pair_from_logits``.
        posenet, segnet: frozen scorers (load_differentiable_scorers
            output).
        cfg: OptimizeConfig.
        device: training device — must be CUDA in production; CPU allowed
            for smoke tests with the [advisory only] caveat.

    Returns:
        (gray_int_uint8, metrics_log) where:
            gray_int_uint8: uint8 (N, H, W) — the final optimized grayscale
                canvas, ready for ffmpeg ► grayscale.mkv encoding.
            metrics_log: list of dicts with per-step ``step``, ``loss``,
                ``pose``, ``seg`` proxy metrics.

    Raises:
        PreflightError: device is MPS, eval_roundtrip is False, or required
            inputs have inconsistent shapes.
    """
    if not cfg.eval_roundtrip:
        raise PreflightError(
            "Lane AL refuses eval_roundtrip=False (CLAUDE.md non-negotiable: "
            "proxy/auth gap is 2-11x without roundtrip)."
        )
    dev = _validate_device(device)

    if init_class_ids.dim() != 3:
        raise ValueError(
            f"init_class_ids must be (N, H, W); got {tuple(init_class_ids.shape)}"
        )
    N, H, W = init_class_ids.shape
    if N % 2 != 0:
        raise ValueError(
            f"init_class_ids has odd N={N}; pose pairs are non-overlapping "
            "and require an even frame count."
        )
    P = N // 2
    if gt_pairs_btchw.shape[:2] != (P, 2):
        raise ValueError(
            f"gt_pairs_btchw must be (P={P}, 2, 3, H, W); "
            f"got {tuple(gt_pairs_btchw.shape)}"
        )
    if gt_pairs_btchw.shape[2] != 3:
        raise ValueError(
            f"gt_pairs_btchw channel dim must be 3 (RGB); got {gt_pairs_btchw.shape[2]}"
        )

    gray_logits = initialize_gray_logits(init_class_ids, device=dev)
    targets = torch.tensor(
        [CLASS_TO_GRAY[c] for c in range(NUM_CLASSES)],
        dtype=torch.float32, device=dev,
    )
    optimizer = torch.optim.Adam([gray_logits], lr=cfg.lr)
    embedding = embedding.to(dev).eval()
    for p in embedding.parameters():
        p.requires_grad = False

    # Reproducible noise for eval_roundtrip.
    if dev.type == "cuda":
        gen = torch.Generator(device=dev)
    else:
        gen = torch.Generator()
    gen.manual_seed(cfg.seed)

    # gt_pairs_btchw stays on CPU until per-batch slice; keeps memory bounded.
    metrics_log: list[dict] = []

    for step in range(cfg.steps):
        # Random batch of pair indices each step.
        pair_idx = torch.randperm(P, generator=gen if dev.type != "cuda" else None)[
            : cfg.batch_size
        ]
        # gt_pairs slice: (B, 2, 3, H, W)
        gt_batch = gt_pairs_btchw[pair_idx].to(dev, non_blocking=True)

        # Map pair index to two consecutive frame indices in init_class_ids
        # (non-overlapping: pair k = frames 2k, 2k+1).
        flat_frame_idx = torch.stack(
            [pair_idx * 2, pair_idx * 2 + 1], dim=1
        ).reshape(-1)
        gray_logits_pair = gray_logits[flat_frame_idx].reshape(
            cfg.batch_size, 2, H, W,
        )

        rgb_btchw = _render_pair_from_logits(
            gray_logits_pair=gray_logits_pair,
            sigma=cfg.sigma,
            targets=targets,
            embedding=embedding,
            renderer_forward_from_embedding=renderer_forward_from_embedding,
        )
        # Apply STE round/clamp to match what AV1 + ffmpeg + downstream
        # roundtrip will produce. STE keeps gradient flowing through the
        # rounding for the optimizer's benefit.
        rgb_btchw = _ste_round_clamp(rgb_btchw)

        rgb_btchw = _eval_roundtrip_with_noise(
            rgb_btchw, noise_std=cfg.noise_std, generator=gen,
        )

        # Scorer forward expects (B, T, C, H, W) at scorer resolution.
        fp_out, fs_out = scorer_forward_pair(rgb_btchw, posenet, segnet)
        with torch.no_grad():
            gp_out, gs_out = scorer_forward_pair(gt_batch, posenet, segnet)

        # PoseNet MSE on first-6 dims (matches upstream evaluate.py).
        pose_dist = (
            (fp_out["pose"][..., :6] - gp_out["pose"][..., :6])
            .pow(2)
            .mean()
        )
        # SegNet differentiable surrogate: cross-entropy of soft predicted
        # probs against GT-argmax class. The contest metric (argmax
        # disagreement rate) is non-differentiable; CE is the canonical
        # surrogate (see tac/losses.py kl_distill_segnet_only). Here we
        # use plain CE because the GT logits are detached.
        gt_classes = gs_out.argmax(dim=1).detach()  # (B, H_seg, W_seg)
        seg_dist = F.cross_entropy(fs_out, gt_classes, reduction="mean")

        loss = (
            cfg.seg_weight * seg_dist
            + torch.sqrt((cfg.pose_weight * pose_dist).clamp(min=1e-12))
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step % cfg.log_every == 0) or (step == cfg.steps - 1):
            metrics_log.append(
                {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "pose": float(pose_dist.detach().item()),
                    "seg": float(seg_dist.detach().item()),
                }
            )
        if (
            cfg.smoke_eval_callback is not None
            and cfg.smoke_eval_every > 0
            and step > 0
            and step % cfg.smoke_eval_every == 0
        ):
            with torch.no_grad():
                gray_int_full = (
                    _ste_round_clamp(_gray_logits_to_continuous(gray_logits))
                    .to(torch.uint8)
                    .cpu()
                )
            cfg.smoke_eval_callback(step, gray_int_full, metrics_log[-1])

    with torch.no_grad():
        gray_int_full = (
            _gray_logits_to_continuous(gray_logits)
            .round()
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .cpu()
        )

    return gray_int_full, metrics_log


__all__ = [
    "OptimizeConfig",
    "initialize_gray_logits",
    "optimize_grayscale_canvas",
    "_gaussian_softmax_soft",
    "_soft_embedding_lookup",
    "_ste_clamp",
    "_ste_round_clamp",
    "_gray_logits_to_continuous",
    "_render_pair_from_logits",
]
