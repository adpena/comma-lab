# SPDX-License-Identifier: MIT
"""Training loop for task-aware codec post-filters.

Provides a class-based Trainer that composes:
  - QAT (Quantization-Aware Training) via FakeQuant STE
  - EMA (Exponential Moving Average) weight averaging
  - Best-checkpoint int8 selection (the key mechanism)
  - Saliency-weighted reconstruction loss
  - Optional boundary-weighted SegNet STE loss

Usage::

    from tac.training import Trainer, TrainConfig
    from tac.architectures import build_postfilter

    config = TrainConfig(hidden=64, epochs=1000, alpha=20)
    model = build_postfilter("standard", hidden=config.hidden)
    trainer = Trainer(model, config)
    trainer.fit(comp_pairs, gt_pairs, posenet, segnet, sal_weights)
"""

from __future__ import annotations

import atexit
import json
import math
import signal
import time as _time_module
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, model_validator

from .data import pair_from_frames, pair_start_indices, saliency_for_pair
from .kl_config import DistillationPolicy, distillation_policy_sha256, normalize_distillation_policy
from .losses import (
    dual_saliency_reconstruction_loss,
    eval_scorer_loss,
    feature_matching_loss,
    focal_segnet_ste_loss,
    frequency_aware_loss,
    kl_distill_scorer_loss,
    posenet_embedding_loss,
    saliency_reconstruction_loss,

    scorer_forward_pair,
    scorer_loss,
    scorer_loss_cached,
    scorer_loss_pcgrad,
    scorer_loss_pcgrad_cached,
    segnet_kl_divergence_loss,
    segnet_ste_loss,
    temperature_scorer_loss,
)
from .quantization import quantize_state_dict


class TrainConfig(BaseModel):
    """Validated training hyperparameters.

    Uses pydantic for runtime validation — catches misconfiguration before
    burning GPU hours on a broken run.
    """

    model_config = {"frozen": True}  # immutable after creation

    # Architecture
    hidden: int = Field(64, ge=4, le=512, description="Hidden channel width")
    kernel: int = Field(3, ge=1, le=7, description="Convolution kernel size (must be odd)")
    variant: str = Field("standard", description="Architecture variant name")

    # Training schedule
    epochs: int = Field(1000, ge=1, le=50000)
    alpha: float = Field(20.0, ge=0.0, description="Scorer loss weight")
    sal_lambda: float = Field(1.0, ge=0.0, description="Saliency reconstruction weight")
    lr: float = Field(5e-4, gt=0.0, le=1.0)
    warmup_epochs: int = Field(10, ge=0)
    ema_decay: float = Field(0.997, ge=0.9, le=0.9999)
    grad_clip: float = Field(1.0, gt=0.0)
    pairs_per_epoch: int | None = Field(None, ge=1)
    scheduler: Literal["cosine", "cosine_restart"] = "cosine"
    restart_t0: int = Field(200, ge=1)
    restart_tmult: int = Field(2, ge=1)

    # SegNet boundary attack
    boundary_weight: float = Field(1.0, ge=0.0)
    use_ste_segnet: bool = False

    # SegNet headroom unlocking
    loss_mode: Literal[
        "standard", "temperature", "focal_ste", "kl_distill", "pcgrad",
        "feature_match", "segnet_kl", "posenet_embedding",
    ] = "standard"
    kl_distill_scope: Literal["none", "segnet_aux", "primary_scorer"] = Field(
        "none",
        description=(
            "Disambiguates loss_mode='kl_distill'. 'segnet_aux' means standard "
            "scorer loss plus SegNet-only KL auxiliary. 'primary_scorer' is the "
            "legacy kl_distill_scorer_loss path, which is banned for promotion "
            "and requires allow_banned_primary_kl_distill=True."
        ),
    )
    allow_banned_primary_kl_distill: bool = Field(
        False,
        description=(
            "Explicit forensic opt-in for the legacy primary KL-distill scorer "
            "loss. This path collapsed PoseNet in authoritative evals and must "
            "not be used for promotion candidates."
        ),
    )
    promotion_eligible: bool = Field(
        True,
        description=(
            "Whether this config may be treated as promotion-eligible. Legacy "
            "primary KL-distill configs must set this False."
        ),
    )
    forensic_reason: str | None = Field(
        None,
        description=(
            "Required explanation for forensic-only distillation families "
            "such as primary scorer KL, legacy SegNet-KL, and JBL."
        ),
    )
    temperature_start: float = Field(1.0, gt=0.0)
    temperature_end: float = Field(0.05, gt=0.0)
    temp_schedule: str = Field(
        "exponential",
        pattern=r"^(linear|exponential)$",
        description="Temperature decay: 'linear' or 'exponential' (recommended)",
    )
    focal_gamma: float = Field(2.0, ge=0.0)
    segnet_loss_weight: float = Field(100.0, ge=0.0)
    use_dual_saliency: bool = False
    alpha_seg: float = Field(200.0, ge=0.0)

    # Training dynamics
    accum_steps: int = Field(4, ge=1, le=64)
    eval_every: int = Field(5, ge=1, description="Evaluate int8 checkpoint every N epochs")
    hard_frame_ratio: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of training pairs to oversample from hardest SegNet frames. "
        "0.0 = uniform sampling, 0.5 = half hard / half uniform.",
    )
    error_replay_every: int = Field(
        0,
        ge=0,
        description="Recompute hard-frame weights using current model output every N epochs. "
        "0 = static (compute once at start). 200 = adaptive every 200 epochs.",
    )
    boundary_anneal: bool = Field(
        False,
        description="Couple boundary_weight to temperature: "
        "increases boundary attention as T decreases (maintains gradient pressure)",
    )
    use_swa: bool = Field(
        False,
        description="Stochastic Weight Averaging over final 20% of training. Wider minima → better int8 robustness.",
    )
    adaptive_rebalance: bool = Field(
        False,
        description="Enable adaptive weight rebalancing from "
        "src/tac/adaptive.py. Derives segnet_weight and boundary_weight "
        "from current (pose, seg) at each eval epoch.",
    )
    rebalance_every: int = Field(50, ge=1, description="Epochs between adaptive weight updates")
    boundary_fraction: float = Field(0.05, gt=0.0, lt=1.0, description="Measured boundary pixel fraction (beta)")
    # Intentionally 0.0 for the competition: we have exactly one video, so
    # holding out pairs would waste signal. The int8 eval loop already provides
    # checkpoint selection without a held-out split. Set to 0.25 for proper
    # generalization estimates on multi-video datasets.
    eval_holdout: float = Field(
        0.0,
        ge=0.0,
        le=0.5,
        description="Fraction of pairs held out for eval. "
        "0.0 = contest mode (train+eval on all pairs). "
        "0.25 = production mode (25% held-out eval split).",
    )
    use_lsq: bool = Field(False, description="Enable Learned Step Size Quantization")
    use_entropy_bottleneck: bool = Field(
        False,
        description="Enable train-time entropy bottleneck rate regularization.",
    )
    eb_lambda: float = Field(0.0, ge=0.0, description="Entropy bottleneck rate loss weight")
    eb_num_channels: int = Field(64, ge=1, description="Entropy bottleneck latent channels")

    # Eval-matched roundtrip (MANDATORY for auth-faithful training)
    eval_roundtrip: bool = Field(
        True,
        description="Simulate contest eval resize chain (384→874→uint8→384) in scorer loss. "
        "WITHOUT this, proxy-auth gap is 2-6x on PoseNet. "
        "This has caused EVERY wasted training run in this project.",
    )
    roundtrip_noise_std: float = Field(
        0.5,
        ge=0.0,
        description="Gaussian noise std after STE quantization in roundtrip. "
        "0.5 = Hotz fix for proxy-auth drift. 0.0 = no noise. "
        "Default 0.5 — matches train_distill.py and all other training scripts.",
    )

    # Yousfi council tricks
    even_frame_skip_seg: bool = Field(
        False,
        description="Trick 3: skip SegNet loss on even frames. "
        "SegNet only evaluates odd frames, so even frames only need "
        "PoseNet pair fidelity. Reduces multi-task conflict surface.",
    )
    use_frequency_loss: bool = Field(
        False,
        description="Trick 2: add wavelet frequency-domain shaping loss. "
        "Penalizes mid-frequency deviation more than low/high frequency, "
        "matching PoseNet's texture sensitivity profile.",
    )
    frequency_loss_weight: float = Field(0.1, ge=0.0, le=10.0, description="Weight for frequency-domain loss (Trick 2)")

    # Migrated legacy techniques
    use_kalman: bool = Field(
        False,
        description="Use Kalman weight filter instead of EMA. "
        "Inverse-variance weighted: filters out oscillation noise. "
        "Migrated from experiments/train_postfilter_kalman.py.",
    )
    kalman_process_noise: float = Field(1e-6, gt=0.0, description="Kalman filter process noise")
    kalman_obs_noise_base: float = Field(1e-4, gt=0.0, description="Kalman filter observation noise baseline")
    kalman_obs_noise_scale: float = Field(10.0, ge=0.0, description="Kalman filter observation noise scale factor")
    band_lambda: float = Field(0.0, ge=0.0, description="Band-orthogonality loss weight for counterpoint ensemble")

    # Vanishing-point saliency prior (Exploit #5)
    use_vp_saliency: bool = Field(
        False,
        description="Weight per-pixel loss by vanishing-point Gaussian prior. "
        "Pixels near the VP (where PoseNet tz is most sensitive) get higher "
        "gradient weight. Pixels far from VP (sky corners) get lower weight.",
    )
    vp_saliency_sigma: float = Field(
        40.0, gt=0.0,
        description="Gaussian spread for VP saliency map in scorer-resolution pixels.",
    )
    vp_saliency_min_weight: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Minimum weight for pixels far from the vanishing point.",
    )
    vp_saliency_horizon_boost: float = Field(
        2.0, ge=1.0,
        description="Multiplicative boost for the horizon band (sky/road boundary).",
    )
    decor_lambda: float = Field(0.0, ge=0.0, description="Output decorrelation loss weight for counterpoint ensemble")
    posenet_embedding_layer: str = Field(
        "summary",
        description="PoseNet layer for embedding loss: 'summary' (512-d) or 'stages.2' (256-d)",
    )
    posenet_embedding_weight: float = Field(
        0.5, ge=0.0,
        description="Weight for PoseNet embedding loss when loss_mode='posenet_embedding'",
    )

    # Learnable loss weights (Yousfi architectural pass)
    learn_loss_weights: bool = Field(
        False,
        description="Learn segnet/posenet loss weights via log-space nn.Parameters. "
        "When enabled, w_seg = exp(log_w_seg) and w_pose = exp(log_w_pose) are "
        "optimized with a separate LR group (10x base LR, no weight decay).",
    )
    init_log_w_seg: float = Field(
        math.log(100.0),
        description="Initial log(w_seg) for learnable loss weights.",
    )
    init_log_w_pose: float = Field(
        math.log(10.0),
        description="Initial log(w_pose) for learnable loss weights.",
    )

    # Adaptive boundary weight (per-epoch, scorer-feedback-driven)
    adaptive_boundary: bool = Field(
        False,
        description="Adjust boundary_weight each eval epoch based on SegNet distortion. "
        "If seg is improving, reduce boundary weight; if stagnating, increase it.",
    )

    # Optimizer (previously hardcoded)
    weight_decay: float = Field(1e-4, ge=0.0, description="AdamW weight decay")
    eta_min: float = Field(1e-4, ge=0.0, description="CosineAnnealingLR minimum learning rate")

    # Camera / geometry priors (for DepthAwareMotionPredictor and domain solvers)
    # These are passed through to modules that need them. Values follow
    # tac.camera conventions: depth in meters, focal in pixels, etc.
    depth_priors: dict[int, float] | None = Field(
        None,
        description="Per-class depth priors in meters. Keys are class indices (0-4). "
        "None = use defaults from tac.camera.DEPTH_PRIORS_METERS.",
    )
    focal_length: tuple[float, float] | None = Field(
        None,
        description="Camera focal length (fx, fy) in pixels. "
        "None = use defaults from tac.camera.COMMA_INTRINSICS.",
    )
    principal_point: tuple[float, float] | None = Field(
        None,
        description="Camera principal point (cx, cy) in pixels from top-left. "
        "None = use defaults from tac.camera.COMMA_INTRINSICS.",
    )
    camera_height: float | None = Field(
        None,
        description="Camera height above road in meters. "
        "None = use default from tac.camera.COMMA_EXTRINSICS.",
    )

    # Resumption
    resume_from: str | None = None

    # Wall-clock timeout (seconds). When training has been running for this long,
    # save a checkpoint and exit cleanly. Set to 0 to disable.
    # Kaggle P100 kernels have a 12h limit — use 39600 (11h) for safety margin.
    wall_clock_timeout: int = Field(
        0,
        ge=0,
        description="Max wall-clock seconds before emergency save + clean exit. "
        "0 = no limit. 39600 = 11h (for 12h Kaggle kernels).",
    )

    # ── Council C OOM-class deep fixes (DF2 + DF3) ──────────────────────
    # Memory: .omx/research/council_oom_class_deep_fix_20260429.md.
    # The 21 GB OOM observed across Lane SC++/SA/SO on Modal A10G 22 GB
    # comes from PoseNet FastViT-T12 stage-1 self-attention map
    # (B × heads × N² × 4 bytes, N=12288 at 384×512 scorer input).
    # bf16 halves that allocation; per-pair scorer chunking divides it
    # by chunk_size. Both fixes are required for SegMap-class training
    # to fit on a 24 GB 4090 with the canonical batch size.
    bf16: bool = Field(
        False,
        description="Enable bf16 autocast around the SegMapTrainer forward "
        "(DF2 of council OOM-class deep fix). Cuts the dominant FastViT "
        "attention-map allocation by ~50%. CUDA-only; raises if requested "
        "without CUDA. bf16 (NOT fp16) is intentional: bf16 has the same "
        "exponent range as fp32 so no GradScaler is needed and KL-distill "
        "softmax math stays well-conditioned for T >= 1.5.",
    )
    scorer_chunk: int = Field(
        0,
        ge=0,
        description="Per-pair scorer chunk size (DF3 of council OOM-class "
        "deep fix). 0 = no chunking (legacy unchunked behaviour). N>0 = "
        "split each mini-batch's scorer_forward_pair call into chunks of "
        "N pairs along the batch dim. Cuts per-call attention-map memory "
        "by chunk_size. Apply to BOTH the gradient-flowing rendered branch "
        "and the no_grad GT branch. Council C recommendation: B*N <= 8 for "
        "RTX 4090 24 GB / A10G 22 GB.",
    )

    # ── KL-distill weight (Round 7 Defect #2 fix, 2026-04-29 PM) ────────
    # Memory: feedback_silent_default_bug_class_findings_20260429.md.
    # Council Round 7 §6.2 caught train_segmap.py's --kl-distill-weight
    # 0.002 default being silently DROPPED because TrainConfig had no
    # corresponding field — the conditional plumbing at
    # experiments/train_segmap.py:191 (`if "kl_distill_weight" in fields`)
    # was a no-op, and src/tac/segmap_renderer.py:667 hard-coded
    # `0.002 * kl_loss`. A future operator passing 0.001/0.01 (KL
    # sensitivity sweep) would be silently ignored. This field closes
    # the silent-default override loop: train_segmap.py now ALWAYS
    # threads args.kl_distill_weight into the config, and the trainer
    # uses self.config.kl_distill_weight instead of the literal 0.002.
    kl_distill_weight: float = Field(
        0.002,
        ge=0.0,
        description="KL-distill auxiliary loss weight applied to the "
        "SegNet-only KL term in SegMapTrainer (and any other "
        "kl_distill-mode trainer). Lane G v3 canonical = 0.002. Round 7 "
        "Defect #2 fix made this operator-controllable instead of a "
        "hard-coded literal in segmap_renderer.py.",
    )
    kl_distill_temperature: float = Field(
        2.0,
        gt=0.0,
        description=(
            "Temperature for SegNet-only KL auxiliary helpers. Kept separate "
            "from temperature_start/temperature_end so fixed-temperature KL "
            "auxiliary configs do not masquerade as primary temperature "
            "annealing schedules."
        ),
    )

    # Output
    output_dir: str = "experiments/postfilter_weights"
    tag: str = Field("untitled", min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-]+$")

    @model_validator(mode="after")
    def _validate_config(self) -> TrainConfig:
        if self.kernel % 2 == 0:
            raise ValueError(f"kernel must be odd, got {self.kernel}")
        if self.warmup_epochs >= self.epochs:
            raise ValueError(f"warmup_epochs ({self.warmup_epochs}) must be < epochs ({self.epochs})")
        if self.temperature_end > self.temperature_start:
            raise ValueError("temperature_end must be <= temperature_start")
        if self.loss_mode == "pcgrad" and self.accum_steps > 1:
            import warnings

            warnings.warn(
                f"pcgrad with accum_steps={self.accum_steps}: gradient conflict detection "
                f"only runs on first microbatch of each window. Consider accum_steps=1 "
                f"for full-strength non-opposing guarantee.",
                stacklevel=2,
            )
        if self.loss_mode == "kl_distill":
            if self.kl_distill_scope == "none":
                raise ValueError(
                    "loss_mode='kl_distill' is ambiguous and banned as a silent "
                    "primary loss. Set kl_distill_scope='segnet_aux' for the "
                    "SegNet-only auxiliary path, or kl_distill_scope='primary_scorer' "
                    "with allow_banned_primary_kl_distill=True and "
                    "promotion_eligible=False for a forensic legacy run."
                )
            if self.kl_distill_scope == "primary_scorer":
                if not self.allow_banned_primary_kl_distill:
                    raise ValueError(
                        "primary kl_distill_scorer_loss is banned unless "
                        "allow_banned_primary_kl_distill=True. Prior authoritative "
                        "evals collapsed PoseNet; use segnet_aux for future KL work."
                    )
                if self.promotion_eligible:
                    raise ValueError(
                        "primary kl_distill_scorer_loss configs must set "
                        "promotion_eligible=False. This loss mode is forensic-only."
                    )
            if self.temperature_start < 2.0:
                raise ValueError(
                    f"kl_distill requires temperature_start >= 2.0 (Hinton: anneal 5.0→1.0). "
                    f"Got {self.temperature_start}. Use --temperature-start 5.0 --temperature-end 1.0"
                )
            if self.temperature_end < 0.1:
                raise ValueError(
                    f"kl_distill requires temperature_end >= 0.1 (below 0.1 is numerically unstable). "
                    f"Got {self.temperature_end}. Use --temperature-end 0.2 for aggressive argmax pressure"
                )
        if self.loss_mode == "segnet_kl":
            if self.kl_distill_scope != "segnet_aux":
                raise ValueError(
                    "loss_mode='segnet_kl' is a legacy SegNet auxiliary KL-like "
                    "path and must set kl_distill_scope='segnet_aux'."
                )
            if self.promotion_eligible:
                raise ValueError(
                    "loss_mode='segnet_kl' is not promotion-eligible until "
                    "migrated to kl_distill_segnet_only with exact CUDA "
                    "component gates; set promotion_eligible=False for a "
                    "forensic/debug run."
                )
        self.distillation_policy()
        if self.learn_loss_weights and self.adaptive_boundary:
            import warnings

            warnings.warn(
                "learn_loss_weights and adaptive_boundary are both enabled. "
                "Both mechanisms adjust loss weighting, which can cause conflicting "
                "updates. Consider using only one: learn_loss_weights (gradient-based) "
                "or adaptive_boundary (heuristic feedback-driven).",
                stacklevel=2,
            )
        return self

    def distillation_policy(self) -> DistillationPolicy:
        """Return the frozen KL/distillation policy represented by this config."""

        source = self.model_dump() if hasattr(self, "model_dump") else dict(vars(self))
        # A zero-weight KL configuration is useful for controlled no-op
        # equivalence tests, but it is not an active distillation policy.
        if (
            source.get("loss_mode") == "kl_distill"
            and source.get("kl_distill_scope") == "segnet_aux"
            and float(source.get("kl_distill_weight", 0.0) or 0.0) == 0.0
        ):
            source.update(
                {
                    "family": "none",
                    "scope": "none",
                    "kl_distill_scope": "none",
                    "weight": 0.0,
                    "kl_distill_weight": 0.0,
                }
            )
        return normalize_distillation_policy(source)

    def distillation_policy_provenance(self) -> dict:
        return self.distillation_policy().to_provenance()

    def distillation_policy_sha256(self) -> str:
        return distillation_policy_sha256(self.distillation_policy())


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                # Codex finding 2 hardening: if a module was added to the
                # model AFTER EMA construction (e.g. a late-bound entropy
                # bottleneck), seed its shadow entry from the live tensor
                # instead of KeyError-ing. This keeps EMA correct without
                # requiring every call site to remember the registration
                # ordering invariant.
                if k not in self.shadow:
                    self.shadow[k] = v.clone().detach()
                    continue
                if not v.is_floating_point():
                    # Non-float buffers (e.g. BN num_batches_tracked, int masks):
                    # EMA decay on integers produces garbage — copy directly instead.
                    self.shadow[k].copy_(v)
                else:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """Load EMA weights into model."""
        model.load_state_dict(self.shadow)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


class SWA:
    """Stochastic Weight Averaging — averages EMA shadow snapshots over time.

    Takes periodic snapshots of the EMA shadow dict and computes a running
    average. Applied on top of EMA for wider minima (better int8 quantization).

    Usage: call update(ema_obj) each epoch in the final 20% of training.
    Call apply(ema_obj) at the end of training to replace EMA weights with
    the SWA average before final checkpoint save.
    """

    def __init__(self):
        self.avg: dict[str, torch.Tensor] | None = None
        self.count = 0

    def update(self, ema):
        """Snapshot the EMA shadow weights into the running average."""
        shadow = ema.shadow if hasattr(ema, "shadow") else ema
        if self.avg is None:
            self.avg = {k: v.clone() for k, v in shadow.items()}
            self.count = 1
        else:
            self.count += 1
            for k in self.avg:
                if k in shadow:
                    self.avg[k] += (shadow[k] - self.avg[k]) / self.count

    def apply(self, ema):
        """Replace the EMA shadow with the SWA average. Call before final save."""
        if self.avg is None:
            return
        shadow = ema.shadow if hasattr(ema, "shadow") else ema
        for k in self.avg:
            if k in shadow:
                shadow[k].copy_(self.avg[k])
        print(f"[SWA] Applied average of {self.count} snapshots to EMA shadow")


class KalmanWeightFilter:
    """Per-parameter scalar Kalman filter as an alternative to EMA.

    Migrated from experiments/train_postfilter_kalman.py (KalmanWeightFilter).
    Inverse-variance weighted averaging: parameters with low observation noise
    (stable across epochs) get more weight. Parameters with high observation
    noise (noisy/oscillating across epochs) get less weight.

    Uses per-tensor scalar sigma^2 (uncertainty) rather than per-element tensors
    for efficiency. Observation noise is estimated from the L2 norm of the weight
    delta since last update -- large deltas suggest oscillation / high obs noise.

    This is strictly more information-theoretic than EMA: EMA is the special case
    where obs_noise is constant and equal to (1 - decay) * process_noise.
    In practice, EMA performed comparably for the comma competition.
    """

    def __init__(
        self,
        model: nn.Module,
        process_noise: float = 1e-6,
        obs_noise_base: float = 1e-4,
        obs_noise_scale: float = 10.0,
    ):
        self.process_noise = process_noise
        self.obs_noise_base = obs_noise_base
        self.obs_noise_scale = obs_noise_scale

        # Shadow state (the filtered estimate)
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # Per-tensor scalar sigma^2 (cheaper than per-element, empirically sufficient)
        self.sigma2: dict[str, float] = {k: 1.0 for k in self.state.keys()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if not v.dtype.is_floating_point:
                self.state[k].copy_(v)
                continue
            z = v.detach()
            # Prediction step: sigma^2 grows by process noise
            sigma2 = self.sigma2[k] + self.process_noise
            # Observation noise: proportional to weight delta magnitude
            # (high delta = oscillating / noisy update = high obs noise)
            delta = (z - self.state[k]).pow(2).mean().item()
            obs_noise = self.obs_noise_base + self.obs_noise_scale * delta
            # Kalman gain
            K = sigma2 / (sigma2 + obs_noise)
            # Update state and uncertainty
            self.state[k].mul_(1 - K).add_(z, alpha=K)
            self.sigma2[k] = (1 - K) * sigma2

    def apply(self, model: nn.Module):
        model.load_state_dict(self.state)

    def copy_to(self, model: nn.Module):
        """Alias for apply() -- matches legacy experiment interface."""
        self.apply(model)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.state.items()}


class Trainer:
    """QAT+EMA trainer with best-checkpoint int8 selection.

    The key mechanism: after each epoch, load EMA weights, quantize to int8,
    evaluate on the scorer, and save if it's the best int8 score so far.
    This finds the rare epochs where quantization-friendly weight distributions
    produce good deployed performance.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        device: str | torch.device = "cpu",
    ):
        self.model = model.to(device)
        # P2: channels_last memory format for faster conv2d on MPS
        if str(device) == "mps":
            self.model = self.model.to(memory_format=torch.channels_last)
        self.config = config
        self.device = device
        self.distillation_policy = config.distillation_policy()
        self.distillation_policy_provenance = self.distillation_policy.to_provenance()
        self.distillation_policy_sha256 = distillation_policy_sha256(self.distillation_policy)
        # Codex finding 2 fix (2026-04-28):
        #   The entropy bottleneck MUST be installed on the model BEFORE the
        #   EMA snapshot is taken, otherwise EMA.shadow lacks the
        #   ``entropy_bottleneck.*`` keys and the first ``EMA.update(self.model)``
        #   would KeyError on the new keys (or, depending on the EMA impl,
        #   silently skip them and produce stale EMA weights).
        self.entropy_bottleneck: nn.Module | None = None
        self._entropy_bottleneck_handle = None

        if config.use_entropy_bottleneck:
            from .entropy_bottleneck import EntropyBottleneck

            renderer = getattr(self.model, "renderer", self.model)
            bottleneck = getattr(renderer, "bottleneck", None)
            if bottleneck is None:
                raise ValueError(
                    "use_entropy_bottleneck=True requires a model with a "
                    "bottleneck module"
                )
            self.entropy_bottleneck = EntropyBottleneck(
                num_channels=config.eb_num_channels,
            ).to(device)
            renderer.add_module("entropy_bottleneck", self.entropy_bottleneck)

            def _eb_hook(_module, _inputs, output):
                y_hat, _bits = self.entropy_bottleneck(output)
                return y_hat

            self._entropy_bottleneck_handle = bottleneck.register_forward_hook(_eb_hook)
            print(
                f"[trainer] Entropy bottleneck enabled "
                f"(channels={config.eb_num_channels}, lambda={config.eb_lambda})"
            )

        # EMA constructed AFTER entropy bottleneck is installed so its shadow
        # contains all the keys that ``self.model.state_dict()`` will produce
        # during training (codex finding 2).
        self.ema = EMA(self.model, decay=config.ema_decay)
        self.best_scorer = float("inf")
        self.best_epoch = -1
        self._patched_scorers = False
        self._start_wall_time = _time_module.monotonic()

        # Kaggle safety: warn if on Kaggle with no timeout set
        if config.wall_clock_timeout <= 0 and Path("/kaggle").exists():
            import warnings

            warnings.warn(
                "Running on Kaggle without wall_clock_timeout! "
                "Set wall_clock_timeout=39600 (11h) to avoid losing training state "
                "when the 12h kernel limit hits. Use --profile kaggle_p100_dilated.",
                stacklevel=2,
            )
        self._current_epoch = 0
        self._last_eval_pose = None
        self._last_eval_seg = None
        self._baseline_pose = None
        self._baseline_seg = None
        self._last_replay_scorer = float("inf")
        self._plateau_window: list[float] = []
        self._plateau_reduced = False

        # Adaptive weights (council_v2_adaptive profile)
        self._adaptive = None
        if getattr(config, "adaptive_rebalance", False):
            from .adaptive import AdaptiveWeights

            beta = getattr(config, "boundary_fraction", 0.05)
            self._adaptive = AdaptiveWeights(boundary_fraction=beta)
            print(f"[trainer] Adaptive weights enabled (beta={beta})")

        if self.entropy_bottleneck is not None:
            eb_param_ids = {id(p) for p in self.entropy_bottleneck.parameters()}
            main_params = [p for p in model.parameters() if id(p) not in eb_param_ids]
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": main_params, "lr": config.lr, "weight_decay": config.weight_decay},
                    {
                        "params": list(self.entropy_bottleneck.parameters()),
                        "lr": 1e-3,
                        "weight_decay": 0.0,
                    },
                ]
            )
        else:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # LSQ: Learned Step Size Quantization
        self._lsq_scales: dict[str, nn.Module] | None = None
        if config.use_lsq:
            from .quantization import apply_lsq

            self._lsq_scales = apply_lsq(self.model)
            if self._lsq_scales:
                lsq_params = []
                for lsq_mod in self._lsq_scales.values():
                    lsq_params.extend(lsq_mod.parameters())
                self.optimizer.add_param_group(
                    {
                        "params": lsq_params,
                        "lr": config.lr * 5,
                        "weight_decay": 0.0,
                    }
                )
                print(f"[trainer] LSQ enabled: {len(self._lsq_scales)} learned scales, lr={config.lr * 5:.6f}")

        # Learnable loss weights (log-space for unconstrained optimization)
        self._log_w_seg: nn.Parameter | None = None
        self._log_w_pose: nn.Parameter | None = None
        if config.learn_loss_weights:
            self._log_w_seg = nn.Parameter(torch.tensor(config.init_log_w_seg))
            self._log_w_pose = nn.Parameter(torch.tensor(config.init_log_w_pose))
            self.optimizer.add_param_group(
                {
                    "params": [self._log_w_seg, self._log_w_pose],
                    "lr": config.lr * 10,
                    "weight_decay": 0.0,
                }
            )
            print(
                f"[trainer] Learnable loss weights enabled: "
                f"w_seg={math.exp(config.init_log_w_seg):.1f}, "
                f"w_pose={math.exp(config.init_log_w_pose):.1f}, "
                f"lr={config.lr * 10:.6f}"
            )

        # Adaptive boundary state
        self._adaptive_boundary_weight = config.boundary_weight
        self._prev_seg_distortion: float | None = None

        # VP saliency prior: pre-compute the spatial weight map once
        self._vp_saliency_map: torch.Tensor | None = None
        if config.use_vp_saliency:
            from .camera import vanishing_point_saliency, FRAME_H, FRAME_W

            self._vp_saliency_map = vanishing_point_saliency(
                H=FRAME_H,
                W=FRAME_W,
                sigma=config.vp_saliency_sigma,
                min_weight=config.vp_saliency_min_weight,
                horizon_boost=config.vp_saliency_horizon_boost,
            ).to(device)
            print(
                f"[trainer] VP saliency enabled: sigma={config.vp_saliency_sigma}, "
                f"min={config.vp_saliency_min_weight}, horizon_boost={config.vp_saliency_horizon_boost}"
            )

        if config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=config.eta_min
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=config.restart_t0, T_mult=config.restart_tmult, eta_min=config.eta_min
            )

        # Resume from checkpoint if specified (must come after optimizer/scheduler init)
        if config.resume_from and Path(config.resume_from).exists():
            self.load_training_state(config.resume_from)

        # Emergency save on signals and exit — never lose training state
        self._emergency_registered = False
        self._register_emergency_save()

    def _wall_clock_exceeded(self) -> bool:
        """Check if wall-clock timeout has been exceeded.

        Returns True if the timeout is set and the elapsed training time
        exceeds the configured limit. This triggers a clean save-and-exit.
        """
        timeout = self.config.wall_clock_timeout
        if timeout <= 0:
            return False
        elapsed = _time_module.monotonic() - self._start_wall_time
        return elapsed >= timeout

    def _wall_clock_remaining(self) -> float:
        """Seconds remaining before wall-clock timeout. inf if no timeout set."""
        timeout = self.config.wall_clock_timeout
        if timeout <= 0:
            return float("inf")
        elapsed = _time_module.monotonic() - self._start_wall_time
        return max(0.0, timeout - elapsed)

    def _register_emergency_save(self):
        """Register signal handlers and atexit for crash-proof state saving."""
        if self._emergency_registered:
            return
        self._emergency_registered = True

        def _emergency_save(reason: str):
            try:
                print(f"\n[trainer] EMERGENCY SAVE ({reason}) at epoch {self._current_epoch}")
                self.save_training_state()
                print("[trainer] Emergency save complete.")
            except Exception as e:
                print(f"[trainer] Emergency save FAILED: {e}")

        def _signal_handler(signum, frame):
            _emergency_save(f"signal {signum}")
            raise SystemExit(1)

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                signal.signal(sig, _signal_handler)
            except (OSError, ValueError):
                pass  # some signals can't be caught in threads

        atexit.register(_emergency_save, "atexit")

    def save_training_state(self, path: str | Path | None = None):
        """Save full training state for resumption.

        Uses atomic write (tmp + rename) so a crash mid-save cannot corrupt
        or delete the checkpoint.
        """
        if path is None:
            path = Path(self.config.output_dir) / f"training_state_{self.config.tag}.pt"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema_shadow": self.ema.shadow,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self._current_epoch,
                "best_scorer": self.best_scorer,
                "best_epoch": self.best_epoch,
                "plateau_reduced": self._plateau_reduced,
                "distillation_policy": self.distillation_policy_provenance,
                "distillation_policy_sha256": self.distillation_policy_sha256,
            },
            tmp_path,
        )
        tmp_path.rename(path)  # atomic on POSIX

    def _save_training_cost(self, final_epoch: int) -> None:
        """Save training cost and replicability metadata alongside the checkpoint.

        Writes training_cost_{tag}.json to the output directory. This is
        informational only — never blocks training.
        """
        try:
            elapsed = _time_module.monotonic() - self._start_wall_time
            device_str = str(self.device)

            # Detect platform
            if "cuda" in device_str:
                platform_name = "cloud"
                try:
                    gpu_name = torch.cuda.get_device_name(0).lower()
                except Exception:
                    gpu_name = "cuda"
            elif "mps" in device_str:
                platform_name = "local"
                gpu_name = "mps"
            else:
                platform_name = "local"
                gpu_name = "cpu"

            training_cost = {
                "epochs_completed": final_epoch + 1,
                "wall_clock_seconds": round(elapsed, 2),
                "wall_clock_hours": round(elapsed / 3600, 4),
                "device": device_str,
                "platform": platform_name,
                "gpu": gpu_name,
                "best_scorer": self.best_scorer if self.best_scorer < float("inf") else None,
                "best_epoch": self.best_epoch,
                "tag": self.config.tag,
                "hidden": self.config.hidden,
                "architecture": getattr(self.config, "architecture", "unknown"),
            }

            # Try to add cost estimate from cost_tracker
            try:
                from .cost_tracker import CostRecord
                cost = CostRecord.from_run(platform_name, gpu_name, elapsed)
                training_cost["cost"] = cost.to_dict()
            except ImportError:
                training_cost["cost"] = {
                    "rate_per_hour": 0.0,
                    "total_cost": 0.0,
                    "is_free_tier": True,
                }

            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            cost_path = out_dir / f"training_cost_{self.config.tag}.json"
            cost_path.write_text(json.dumps(training_cost, indent=2) + "\n")
            print(f"[trainer] Cost metadata saved: {cost_path}")
        except Exception as exc:
            # Cost tracking is informational — never block training
            print(f"[trainer] WARNING: failed to save training cost: {exc}")

    def load_training_state(self, path: str | Path):
        """Resume training from a saved state."""
        try:
            state = torch.load(str(path), map_location=self.device, weights_only=True)
        except Exception:
            print(
                "WARNING: weights_only=True failed for checkpoint (likely optimizer state "
                "contains non-tensor objects). Falling back to weights_only=False. "
                "Only load checkpoints you trust."
            )
            state = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.ema.shadow = {k: v.to(self.device) for k, v in state["ema_shadow"].items()}
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self._current_epoch = state.get("epoch", 0)
        self.best_scorer = state.get("best_scorer", float("inf"))
        self.best_epoch = state.get("best_epoch", -1)
        self._plateau_reduced = state.get("plateau_reduced", False)
        print(f"[trainer] Resumed from epoch {self._current_epoch}, best {self.best_scorer:.4f}")

    @staticmethod
    def _patch_scorers_for_training(posenet, segnet):
        """Monkey-patch upstream scorer models for differentiable training.

        The upstream PoseNet.preprocess_input uses rgb_to_yuv6 decorated with
        @torch.no_grad(), which kills gradients through the color space conversion.
        We replace it with a differentiable version that faithfully reproduces
        the upstream math: full-range BT.601, 4:2:0 chroma subsampling, resize
        to scorer input size, and proper einops rearrange.

        AllNorm.forward uses .view() which we replace with .reshape() for
        robustness with non-contiguous tensors. Note: AllNorm is BatchNorm1d(1)
        on flattened post-backbone features — it does NOT provide pixel-level
        brightness invariance (this was disproven 2026-04-11).

        This is REQUIRED for training — without it, PoseNet gradients are zero.
        """
        import types

        import einops

        # Patch AllNorm to not break gradients
        for module in list(posenet.modules()) + list(segnet.modules()):
            if type(module).__name__ == "AllNorm":

                def _patched_forward(self, x):
                    return self.bn(x.reshape(-1, 1)).reshape(x.shape)

                module.forward = types.MethodType(_patched_forward, module)

        # Differentiable rgb_to_yuv6: full-range BT.601 with 4:2:0 subsampling
        # Matches upstream frame_utils.py rgb_to_yuv6 exactly, minus @torch.no_grad
        def _rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
            H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
            H2, W2 = H // 2, W // 2
            rgb = rgb_chw[..., :, : 2 * H2, : 2 * W2]
            R = rgb[..., 0, :, :]
            G = rgb[..., 1, :, :]
            B = rgb[..., 2, :, :]
            Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
            U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
            V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
            U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
            V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
            y00 = Y[..., 0::2, 0::2]
            y10 = Y[..., 1::2, 0::2]
            y01 = Y[..., 0::2, 1::2]
            y11 = Y[..., 1::2, 1::2]
            return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)

        # Get scorer input size from upstream module
        # PoseNet expects (B, 12, 192, 256) after preprocess
        try:
            from modules import segnet_model_input_size
        except ImportError:
            segnet_model_input_size = (512, 384)  # (W, H) default

        def _diff_preprocess(self, x):
            batch_size, seq_len_local = x.shape[0], x.shape[1]
            x = einops.rearrange(x, "b t c h w -> (b t) c h w", b=batch_size, t=seq_len_local, c=3)
            # Resize to scorer input size (bilinear, matching upstream)
            x = nn.functional.interpolate(
                x,
                size=(segnet_model_input_size[1], segnet_model_input_size[0]),
                mode="bilinear",
                align_corners=False,
            )
            # Differentiable YUV conversion with 4:2:0 subsampling
            yuv = _rgb_to_yuv6_diff(x)
            return einops.rearrange(yuv, "(b t) c h w -> b (t c) h w", b=batch_size, t=seq_len_local, c=6).contiguous()

        posenet.preprocess_input = types.MethodType(_diff_preprocess, posenet)

    @property
    def _effective_boundary_weight(self) -> float:
        """Return the boundary_weight, accounting for adaptive boundary if enabled."""
        if self.config.adaptive_boundary:
            return self._adaptive_boundary_weight
        return self.config.boundary_weight

    @property
    def _is_pair_aware(self) -> bool:
        """Check if the model expects 6-channel pair input."""
        from .architectures import PairAwarePostFilter

        return isinstance(self.model, PairAwarePostFilter)

    def _apply_filter_to_pair(self, comp_pair: torch.Tensor) -> torch.Tensor:
        """Apply the post-filter to both frames of a pair.

        Input: (1, 2, H, W, 3) uint8
        Output: (1, 2, H, W, 3) float [0, 255]

        For pair-aware models: concatenates both frames (6ch) and runs
        the model twice — once for each frame with the other as context.
        For standard models: processes each frame independently (3ch).
        """
        B, T, H, W, C = comp_pair.shape

        if self._is_pair_aware:
            # Pair-aware: each frame sees the other as context
            f0 = comp_pair[:, 0].float().permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
            f1 = comp_pair[:, 1].float().permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
            # Frame 0 correction: target=f0, context=f1
            inp0 = torch.cat([f0, f1], dim=1)  # (B, 6, H, W)
            out0 = self.model(inp0)  # (B, 3, H, W)
            # Frame 1 correction: target=f1, context=f0
            inp1 = torch.cat([f1, f0], dim=1)  # (B, 6, H, W)
            out1 = self.model(inp1)  # (B, 3, H, W)
            # Reassemble pair
            result = torch.stack(
                [
                    out0.permute(0, 2, 3, 1),  # (B, H, W, 3)
                    out1.permute(0, 2, 3, 1),
                ],
                dim=1,
            )  # (B, 2, H, W, 3)
            return result
        else:
            # Standard: process each frame independently
            frames_bchw = comp_pair.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
            filtered_bchw = self.model(frames_bchw)
            return filtered_bchw.permute(0, 2, 3, 1).contiguous().reshape(B, T, H, W, C)

    def _evaluate_int8(
        self,
        comp_pairs,
        gt_pairs,
        posenet,
        segnet,
        subsample: int = 2,
    ) -> float:
        """Evaluate EMA model after int8 quantization.

        This is the best-checkpoint selection mechanism.
        Uses subsample=2 (300/600 pairs) for faithful checkpoint selection.

        Assumes B=1 per pair (each comp_pairs[idx] is a single pair).
        Accumulates mean-per-pair and divides by pair count, which is
        equivalent to the global mean only when B=1.
        """
        orig_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Load EMA weights
        self.ema.apply(self.model)

        # Simulate int8 quantization — per-channel for better precision
        q_state = quantize_state_dict(self.model.state_dict(), per_channel=True)
        self.model.load_state_dict(q_state)
        self.model.eval()

        total_p, total_s, count = 0.0, 0.0, 0
        # P1: autocast for CUDA and MPS (fp16 on scorers)
        use_autocast = (str(self.device).startswith("cuda") and torch.cuda.is_available()) or str(self.device) == "mps"
        autocast_device = "cuda" if str(self.device).startswith("cuda") else "mps"
        autocast_ctx = torch.amp.autocast(autocast_device, enabled=use_autocast)
        with torch.no_grad(), autocast_ctx:
            for idx in range(0, len(comp_pairs), subsample):
                filtered = self._apply_filter_to_pair(comp_pairs[idx])
                # uint8 round-trip: matches official inflate → scorer pipeline
                filtered = filtered.round().clamp(0, 255).to(torch.uint8).float()
                score, pd, sd = eval_scorer_loss(filtered, gt_pairs[idx], posenet, segnet)
                total_p += pd
                total_s += sd
                count += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        count = max(count, 1)
        scorer = 100.0 * (total_s / count) + math.sqrt(10.0 * (total_p / count))

        # Restore original training weights
        self.model.load_state_dict(orig_state)
        self.model.train()
        return scorer

    def _save_checkpoint(self, epoch: int, scorer: float):
        """Save best int8 checkpoint with atomic writes.

        Note on the apparent duplication with quantize_state_dict() in
        quantization.py: that function returns a DEQUANTIZED state dict
        (for in-memory proxy eval — round-trip simulation). This loop
        produces the PACKED int8 archive format that inflate-time code
        deserializes (name+".q" int8 tensor + name+".s" fp32 scale
        tensor + meta dict). They look similar but produce different
        artifacts. Calling quantize_state_dict here would also discard
        the meta + config_fingerprint that the inflate-side validator
        depends on.
        """
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = self.config.tag

        fp32_path = out_dir / f"postfilter_{tag}_best_fp32.pt"
        int8_path = out_dir / f"postfilter_{tag}_best_int8.pt"
        meta_path = out_dir / f"postfilter_{tag}_best_meta.json"

        # Codex finding 2 fix: training-only ``entropy_bottleneck.*`` keys
        # are NOT part of the deployed renderer (the bottleneck is replaced
        # by quantization at inflate). Strip them from BOTH the fp32 and
        # int8 archives so that a deployment loader doesn't see phantom
        # entries (and so the int8 archive size doesn't include them).
        ema_state = {
            k: v for k, v in self.ema.state_dict().items()
            if not k.startswith("entropy_bottleneck.")
            and "entropy_bottleneck." not in k
        }

        # Save EMA fp32 (atomic)
        fp32_tmp = fp32_path.with_suffix(".pt.tmp")
        torch.save(ema_state, fp32_tmp)
        fp32_tmp.rename(fp32_path)

        # Save int8 per-channel (atomic) — better precision for multi-channel convs
        int8_state = {}
        for name, param in self.ema.shadow.items():
            if name.startswith("entropy_bottleneck.") or "entropy_bottleneck." in name:
                continue  # codex finding 2: training-only key
            p = param.detach().cpu().float()
            if p.ndim >= 2 and "weight" in name:
                # Per-channel: scale per output channel (dim 0)
                flat = p.reshape(p.shape[0], -1)
                scale = flat.abs().amax(dim=1) / 127.0
                scale = scale.clamp(min=1e-10)
                q = (p / scale.reshape(-1, *([1] * (p.ndim - 1)))).round().clamp(-128, 127).to(torch.int8)
                int8_state[name + ".q"] = q
                int8_state[name + ".s"] = scale
            else:
                # Bias or 1D: per-tensor
                scale = p.abs().max() / 127.0
                if scale.item() < 1e-10:
                    scale = torch.tensor(1.0, device=p.device)
                int8_state[name + ".q"] = (p / scale).round().clamp(-128, 127).to(torch.int8)
                int8_state[name + ".s"] = scale
        int8_state["__meta__"] = {
            "variant": self.config.variant,
            "hidden": self.config.hidden,
            "kernel": self.config.kernel,
            "alpha": self.config.alpha,
            "distillation_policy": self.distillation_policy_provenance,
            "distillation_policy_sha256": self.distillation_policy_sha256,
        }
        # Distribution shift guard: save encode config fingerprint so inflate
        # can warn if config.env changed after training (2026-04-11).
        int8_state["config_fingerprint"] = {
            "crf": getattr(self.config, "crf", 34),
            "color_matrix": getattr(self.config, "color_matrix", "bt709"),
            "codec": getattr(self.config, "codec", "libsvtav1"),
            "scale_w": getattr(self.config, "scale_w", 524),
            "scale_h": getattr(self.config, "scale_h", 394),
        }
        int8_tmp = int8_path.with_suffix(".pt.tmp")
        torch.save(int8_state, int8_tmp)
        int8_tmp.rename(int8_path)

        meta_tmp = meta_path.with_suffix(".json.tmp")
        meta_tmp.write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "scorer": scorer,
                    "pose": getattr(self, "_last_eval_pose", None),
                    "seg": getattr(self, "_last_eval_seg", None),
                    "fp32_path": str(fp32_path),
                    "int8_path": str(int8_path),
                    "int8_size": int8_path.stat().st_size,
                    "meta": int8_state["__meta__"],
                    "config": {
                        "variant": self.config.variant,
                        "hidden": self.config.hidden,
                        "loss_mode": self.config.loss_mode,
                        "kl_distill_scope": self.config.kl_distill_scope,
                        "kl_distill_weight": self.config.kl_distill_weight,
                        "kl_distill_temperature": self.config.kl_distill_temperature,
                        "allow_banned_primary_kl_distill": self.config.allow_banned_primary_kl_distill,
                        "promotion_eligible": self.config.promotion_eligible,
                        "boundary_weight": self.config.boundary_weight,
                        "segnet_loss_weight": self.config.segnet_loss_weight,
                        "hard_frame_ratio": self.config.hard_frame_ratio,
                        "temperature_start": self.config.temperature_start,
                        "temperature_end": self.config.temperature_end,
                        "temp_schedule": self.config.temp_schedule,
                        "alpha": self.config.alpha,
                    },
                    "distillation_policy": self.distillation_policy_provenance,
                    "distillation_policy_sha256": self.distillation_policy_sha256,
                    "baseline_pose": getattr(self, "_baseline_pose", None),
                    "baseline_seg": getattr(self, "_baseline_seg", None),
                },
                indent=2,
            )
        )
        meta_tmp.rename(meta_path)

        # Durable backup — survives MPS SIGKILL which can't be caught
        # Use output_dir-relative path so it works on cloud platforms too
        import shutil

        backup_dir = out_dir / ".backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_tmp = backup_dir / f"postfilter_{tag}_best_int8.pt.tmp"
        shutil.copy2(int8_path, backup_tmp)
        backup_tmp.rename(backup_dir / f"postfilter_{tag}_best_int8.pt")

    def fit(
        self,
        comp_pairs: list[torch.Tensor],
        gt_pairs: list[torch.Tensor],
        posenet,
        segnet,
        sal_weights: torch.Tensor,
        boundary_masks: list[torch.Tensor] | None = None,
        callback: Callable[[int, float, float, float, float], None] | None = None,
    ):
        """Run the full training loop.

        Args:
            comp_pairs: list of (1, 2, H, W, 3) compressed pairs
            gt_pairs: list of (1, 2, H, W, 3) ground-truth pairs
            posenet: frozen PoseNet
            segnet: frozen SegNet
            sal_weights: (N, 1, H, W) saliency weights
            boundary_masks: optional list of (H, W) boundary masks per pair
            callback: optional (epoch, loss, pose, seg, scorer) callback
        """
        cfg = self.config

        # Guard: fit() only supports standard and boundary-weighted loss modes
        if cfg.loss_mode not in ("standard",) and not cfg.use_ste_segnet:
            raise NotImplementedError(
                f"loss_mode='{cfg.loss_mode}' requires fit_lazy(). "
                "fit() only supports 'standard' and boundary-weighted (use_ste_segnet) modes."
            )
        if getattr(cfg, "adaptive_rebalance", False):
            raise NotImplementedError(
                "adaptive_rebalance requires fit_lazy(). fit() does not support adaptive weights."
            )
        if cfg.use_swa:
            import warnings

            warnings.warn(
                "use_swa=True has no effect in fit(). SWA is only implemented in fit_lazy(). "
                "Either switch to fit_lazy() or set use_swa=False.",
                stacklevel=2,
            )

        # Patch scorer models for differentiable training (CRITICAL: without this,
        # PoseNet gradients are zero due to upstream @torch.no_grad on rgb_to_yuv6)
        if not self._patched_scorers:
            self._patch_scorers_for_training(posenet, segnet)
            self._patched_scorers = True

        n_pairs = len(comp_pairs)

        pairs_per_epoch = cfg.pairs_per_epoch or n_pairs
        use_boundary = cfg.use_ste_segnet and boundary_masks is not None

        print(
            f"[trainer] {cfg.epochs} epochs, {pairs_per_epoch} pairs/ep, "
            f"h={cfg.hidden}, alpha={cfg.alpha}, device={self.device}"
        )
        if use_boundary:
            print(f"[trainer] SegNet STE + boundary weighting ({self._effective_boundary_weight}x)")
        if self._current_epoch > 0:
            print(f"[trainer] Resuming from epoch {self._current_epoch}")

        for epoch in range(self._current_epoch, cfg.epochs):
            self._current_epoch = epoch
            self.model.train()
            total_loss, total_pose, total_seg = 0.0, 0.0, 0.0

            # Warmup LR
            if epoch < cfg.warmup_epochs:
                lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

            indices = torch.randperm(n_pairs)[:pairs_per_epoch]
            for idx in indices:
                idx = idx.item()
                filtered = self._apply_filter_to_pair(comp_pairs[idx])

                # Scorer loss (fit() only supports standard and boundary-weighted modes;
                # temperature/focal_ste/kl_distill require fit_lazy())
                if use_boundary:
                    bm = boundary_masks[idx] if boundary_masks else None
                    loss, pd, sd = segnet_ste_loss(
                        filtered,
                        gt_pairs[idx],
                        posenet,
                        segnet,
                        boundary_mask=bm,
                        boundary_weight=self._effective_boundary_weight,
                    )
                else:
                    loss, pd, sd = scorer_loss(
                        filtered,
                        gt_pairs[idx],
                        posenet,
                        segnet,
                    )

                # Saliency reconstruction
                B, T, H, W, C = filtered.shape
                filtered_bchw = filtered[:, 1].permute(0, 3, 1, 2)
                comp_bchw = comp_pairs[idx][:, 1].float().permute(0, 3, 1, 2)
                sal_idx = min(idx * 2 + 1, sal_weights.shape[0] - 1)
                sal_w_pair = sal_weights[sal_idx : sal_idx + 1]
                # Exploit #5: VP saliency prior — multiply per-pixel sal weights
                # by the vanishing-point Gaussian map so VP-region pixels get
                # stronger gradient signal in the reconstruction loss.
                if self._vp_saliency_map is not None:
                    vp_map = self._vp_saliency_map
                    if vp_map.shape[-2:] != sal_w_pair.shape[-2:]:
                        vp_map = nn.functional.interpolate(
                            vp_map, size=sal_w_pair.shape[-2:],
                            mode="bilinear", align_corners=False,
                        )
                    sal_w_pair = sal_w_pair * vp_map
                sal_recon = saliency_reconstruction_loss(filtered_bchw, comp_bchw, sal_w_pair)

                total = loss + cfg.sal_lambda * sal_recon
                if self.entropy_bottleneck is not None and cfg.eb_lambda > 0.0:
                    total = total + cfg.eb_lambda * self.entropy_bottleneck.rate_loss()

                # 2026-04-28 deep hardening pass 3 dimension 3: assert finite
                # loss BEFORE backward. NaN/inf propagates silently through
                # backward + optimizer.step, corrupting weights for the rest
                # of training. Fail loud at the first non-finite step so the
                # operator can see exactly which loss term blew up. Cheap:
                # one host-device sync per step but only on .item() of a 0-d
                # tensor.
                _loss_val = total.item()
                if not (_loss_val == _loss_val and abs(_loss_val) != float("inf")):
                    raise RuntimeError(
                        f"[Trainer] non-finite loss at epoch step: total={_loss_val} "
                        f"loss={loss.item()} pd={pd} sd={sd}. "
                        f"Aborting before backward to preserve weights."
                    )

                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                self.ema.update(self.model)

                total_loss += loss.item()
                total_pose += pd
                total_seg += sd

            if epoch >= cfg.warmup_epochs:
                self.scheduler.step()

            n = len(indices)
            avg_loss = total_loss / n
            avg_pose = total_pose / n
            avg_seg = total_seg / n

            # Best-checkpoint int8 evaluation — gated by eval_every (matches fit_lazy)
            epoch_frac = epoch / max(cfg.epochs - 1, 1)
            if epoch_frac >= 0.8:
                eval_freq = 1
            elif epoch_frac >= 0.2:
                eval_freq = max(2, cfg.eval_every // 2)
            else:
                eval_freq = cfg.eval_every
            is_eval_epoch = (epoch + 1) % eval_freq == 0 or epoch == cfg.epochs - 1 or epoch == 0

            if is_eval_epoch:
                scorer_val = self._evaluate_int8(comp_pairs, gt_pairs, posenet, segnet)

                if scorer_val < self.best_scorer:
                    self.best_scorer = scorer_val
                    self.best_epoch = epoch
                    self._save_checkpoint(epoch, scorer_val)
                    print(f"  ** NEW BEST: ep {epoch}, scorer {scorer_val:.4f} **")
            else:
                scorer_val = self.best_scorer  # carry forward for logging

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"[ep {epoch:4d}] loss={avg_loss:.4f} pose={avg_pose:.6f} "
                f"seg={avg_seg:.6f} scorer={scorer_val:.4f} best={self.best_scorer:.4f} lr={lr:.6f}"
            )

            if callback:
                callback(epoch, avg_loss, avg_pose, avg_seg, scorer_val)

            # Save training state every 50 epochs for crash recovery
            if epoch % 50 == 0 and epoch > 0:
                self.save_training_state()

            # Wall-clock timeout: save and exit cleanly before platform kills us
            if self._wall_clock_exceeded():
                remaining = self._wall_clock_remaining()
                elapsed = _time_module.monotonic() - self._start_wall_time
                print(
                    f"\n[trainer] WALL-CLOCK TIMEOUT at epoch {epoch} "
                    f"(elapsed {elapsed / 3600:.1f}h, limit {self.config.wall_clock_timeout / 3600:.1f}h)"
                )
                self.save_training_state()
                print(f"[trainer] Timeout exit. Best: {self.best_scorer:.4f} at epoch {self.best_epoch}")
                return self.best_scorer

        self.save_training_state()
        self._save_training_cost(epoch)
        print(f"[trainer] Complete. Best: {self.best_scorer:.4f} at epoch {self.best_epoch}")
        return self.best_scorer

    def fit_lazy(
        self,
        comp_frames: list[torch.Tensor],
        gt_frames: list[torch.Tensor],
        posenet,
        segnet,
        raw_saliency: torch.Tensor,
        subsample: int = 8,
    ):
        """Memory-efficient training using lazy pair construction.

        This is the partner agent's proven pattern that survives MPS memory
        pressure for 1000+ epoch runs. Instead of pre-building all 600 pairs
        in memory (~12GB), pairs are constructed on-the-fly from frame lists.

        Args:
            comp_frames: list of (H, W, 3) uint8 compressed frames (on CPU)
            gt_frames: list of (H, W, 3) uint8 ground-truth frames (on CPU)
            posenet: frozen PoseNet (on device)
            segnet: frozen SegNet (on device)
            raw_saliency: (N, H, W) raw saliency map (on CPU)
            subsample: train on 1/subsample of pairs per epoch
        """
        if not self._patched_scorers:
            self._patch_scorers_for_training(posenet, segnet)
            self._patched_scorers = True

        # NOTE: channels_last on scorers is DISABLED. The upstream PoseNet/SegNet
        # use AllNorm (BatchNorm1d(1) on flattened features) whose backward pass
        # calls .view() which is incompatible with channels_last strides on MPS. The model (postfilter)
        # still uses channels_last for speed; scorers stay in standard layout.

        cfg = self.config
        all_pair_starts = pair_start_indices(len(comp_frames))
        n_total = len(all_pair_starts)
        eval_pair_starts: list[int] = []

        # Train/eval split controlled by eval_holdout:
        #   0.0  = contest mode: train+eval on ALL pairs (maximize signal from 1 video)
        #   0.25 = production mode: last 25% held out (proper generalization estimate)
        if cfg.eval_holdout > 0:
            eval_size = max(1, int(n_total * cfg.eval_holdout))
            train_pair_starts = all_pair_starts[:-eval_size]
            eval_pair_starts = all_pair_starts[-eval_size:]
            split_label = f"train {len(train_pair_starts)} / eval {len(eval_pair_starts)} (held-out)"
        else:
            train_pair_starts = all_pair_starts
            eval_pair_starts = all_pair_starts
            split_label = f"all {n_total} pairs (contest mode, eval=train)"
        n_train = len(train_pair_starts)
        train_size = max(1, n_train // subsample)

        print(
            f"[trainer-lazy] {cfg.epochs} epochs, {train_size}/{n_train} pairs/ep, "
            f"{split_label}, h={cfg.hidden}, alpha={cfg.alpha}, device={self.device}"
        )
        print("[trainer-lazy] Frames on CPU, pairs built on-the-fly (MPS-safe)")
        if cfg.loss_mode != "standard":
            print(f"[trainer-lazy] Loss mode: {cfg.loss_mode}")
        if cfg.even_frame_skip_seg:
            print("[trainer-lazy] Trick 3: Even-frame SegNet skip ENABLED")
        if cfg.use_frequency_loss:
            print(f"[trainer-lazy] Trick 2: Frequency loss ENABLED (weight={cfg.frequency_loss_weight})")
        if cfg.sal_lambda == 0:
            print("[trainer-lazy] Saliency reconstruction DISABLED (sal_lambda=0)")

        # P3: Pre-stage frames on MPS (unified memory — no actual copy, just marks as MPS tensors)
        if str(self.device) == "mps":
            if isinstance(gt_frames, list):
                gt_frames = [f.to(self.device) for f in gt_frames]
            else:
                gt_frames = gt_frames.to(self.device)
            if comp_frames is not None:
                if isinstance(comp_frames, list):
                    comp_frames = [f.to(self.device) for f in comp_frames]
                else:
                    comp_frames = comp_frames.to(self.device)
            print("[trainer-lazy] P3: Frames pre-staged on MPS (unified memory)")

        # P0: Precompute GT scorer outputs (constant — frames and scorers are frozen)
        # This eliminates ~50% of scorer forward passes in the training loop.
        import torch.nn.functional as _F

        from .losses import _hwc_to_chw

        print("[trainer-lazy] P0: Precomputing GT scorer cache...")
        gt_scorer_cache = {}
        with torch.no_grad():
            for start in train_pair_starts:
                gt_pair = pair_from_frames(gt_frames, start).to(self.device)
                gx = _hwc_to_chw(gt_pair)
                gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)
                gt_scorer_cache[start] = {
                    "pose_6": gp_out["pose"][..., :6].cpu(),
                    "seg_soft": _F.softmax(gs_out, dim=1).cpu(),
                }
                del gt_pair, gx, gp_out, gs_out
        cache_bytes = sum(
            v["pose_6"].numel() * v["pose_6"].element_size() + v["seg_soft"].numel() * v["seg_soft"].element_size()
            for v in gt_scorer_cache.values()
        )
        print(f"[trainer-lazy] P0: Cached {len(gt_scorer_cache)} GT scorer outputs ({cache_bytes / 1e6:.1f}MB)")
        if cfg.eval_roundtrip:
            print(f"[trainer-lazy] eval_roundtrip=True (noise_std={cfg.roundtrip_noise_std}) — "
                  "GT cache BYPASSED, roundtrip active on both pred+GT")
        else:
            print("[trainer-lazy] eval_roundtrip=False — WARNING: proxy-auth gap 2-6x on PoseNet")

        # Precompute boundary masks if needed (dual saliency OR kl_distill with boundary weighting)
        self._boundary_masks = None
        needs_boundary = cfg.use_dual_saliency or cfg.boundary_weight > 1.0 or cfg.adaptive_boundary
        if needs_boundary:
            from .losses import compute_boundary_mask

            print("[trainer-lazy] Computing SegNet boundary masks for dual saliency...")
            self._boundary_masks = {}
            for start in train_pair_starts:
                gt_pair = pair_from_frames(gt_frames, start)
                mask = compute_boundary_mask(gt_pair, segnet, device=self.device)
                self._boundary_masks[start] = mask
            avg_frac = sum(m.mean().item() for m in self._boundary_masks.values()) / len(self._boundary_masks)
            print(
                f"[trainer-lazy] Boundary masks: {len(self._boundary_masks)} pairs, avg {avg_frac:.4f} ({avg_frac * 100:.2f}%)"
            )

        # Hard-frame curriculum: precompute per-pair SegNet disagreement for weighted sampling
        hard_frame_weights = None

        def _compute_hard_frame_weights(use_model: bool = False, label: str = "init"):
            """Compute weighted sampling from per-pair SegNet difficulty.

            Args:
                use_model: if True, run current model on compressed frames first (error replay).
                    if False, measure raw compressed vs GT (static curriculum).
                label: log label for this recomputation.
            """
            from .losses import eval_scorer_loss

            print(
                f"[trainer-lazy] Computing hard-frame weights ({label}, "
                f"{'model output' if use_model else 'raw compressed'})..."
            )
            pair_difficulties = []
            self.model.eval()
            with torch.no_grad():
                for start in train_pair_starts:
                    gt_pair = pair_from_frames(gt_frames, start).to(self.device)
                    comp_pair = pair_from_frames(comp_frames, start).to(self.device)
                    if use_model:
                        # Error replay: measure difficulty on model's CURRENT output
                        filtered = self._apply_filter_to_pair(comp_pair)
                        filtered = filtered.round().clamp(0, 255).to(torch.uint8).float()
                        _, _, seg_d = eval_scorer_loss(filtered, gt_pair, posenet, segnet)
                    else:
                        _, _, seg_d = eval_scorer_loss(comp_pair, gt_pair, posenet, segnet)
                    pair_difficulties.append(seg_d)
            self.model.train()
            difficulties = torch.tensor(pair_difficulties)
            # Store raw ranks for adaptive ratio ramping
            ranks = torch.argsort(torch.argsort(difficulties)).float()
            normalized_ranks = ranks / max(ranks.max().item(), 1.0)
            self._hard_frame_ranks = normalized_ranks + 0.01  # epsilon for nonzero
            self._hard_frame_avg_diff = difficulties.mean().item()
            # Compute initial weights at current ratio
            return _apply_hard_frame_ratio(cfg.hard_frame_ratio)

        def _apply_hard_frame_ratio(ratio: float) -> torch.Tensor:
            """Apply power-law weighting to cached ranks with given ratio."""
            ranks = self._hard_frame_ranks
            weights = ranks ** max(ratio, 0.01)
            weights = weights / weights.sum()
            return weights

        if cfg.hard_frame_ratio > 0:
            hard_frame_weights = _compute_hard_frame_weights(use_model=False, label="init")

        if self._current_epoch > 0:
            print(f"[trainer-lazy] Resuming from epoch {self._current_epoch}")

        for epoch in range(self._current_epoch, cfg.epochs):
            self._current_epoch = epoch
            self.model.train()
            total_loss, total_pose, total_seg = 0.0, 0.0, 0.0

            if epoch < cfg.warmup_epochs:
                lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

            # Adaptive weight rebalance (once per rebalance_every epochs, not per step)
            if self._adaptive and self._last_eval_pose is not None and epoch % getattr(cfg, "rebalance_every", 50) == 0:
                if cfg.loss_mode in ("standard", "pcgrad", "feature_match"):
                    # Pareto MRS condition: w_seg = 200 * sqrt(10 * pose)
                    # No temperature — standard/pcgrad loss has no temperature parameter.
                    result = self._adaptive.rebalance_standard(
                        eval_pose=self._last_eval_pose,
                        eval_seg=self._last_eval_seg or 0.01,
                    )
                else:
                    # KL distill mode (DEPRECATED — formula is vacuous, see adaptive.py)
                    progress = epoch / max(cfg.epochs - 1, 1)
                    if cfg.temp_schedule == "exponential" and cfg.temperature_start > cfg.temperature_end:
                        _T = cfg.temperature_start * (cfg.temperature_end / cfg.temperature_start) ** progress
                    else:
                        _T = cfg.temperature_start + progress * (cfg.temperature_end - cfg.temperature_start)
                    result = self._adaptive.rebalance(
                        eval_pose=self._last_eval_pose,
                        eval_seg=self._last_eval_seg or 0.01,
                        temperature=_T,
                    )
                self._cached_sw = result["segnet_weight"]
                self._cached_bw = result["boundary_weight"]
                if epoch % (getattr(cfg, "rebalance_every", 50) * 5) == 0:
                    summary = result.get("diagnostics", {}).get("summary", "")
                    print(f"[adaptive] ep={epoch} sw={self._cached_sw:.1f} bw={self._cached_bw:.1f} {summary}")

            # Adaptive hard_frame_ratio ramp: 0.1 → target over first 50% of training
            # (DeepSeek recommendation: uniform exploration early, aggressive exploitation late)
            if cfg.hard_frame_ratio > 0:
                ramp_progress = min(1.0, epoch / max(cfg.epochs * 0.5, 1))
                sin2_progress = math.sin(math.pi / 2 * ramp_progress) ** 2
                effective_hfr = 0.1 + sin2_progress * (cfg.hard_frame_ratio - 0.1)
            else:
                effective_hfr = 0.0

            # Error replay: recompute hard-frame weights using current model output
            if (
                hard_frame_weights is not None
                and cfg.error_replay_every > 0
                and epoch > 0
                and epoch % cfg.error_replay_every == 0
            ):
                improvement = self._last_replay_scorer - self.best_scorer
                if improvement > 0.002 or self._last_replay_scorer == float("inf"):
                    hard_frame_weights = _compute_hard_frame_weights(use_model=True, label=f"error-replay-ep{epoch}")
                    self._last_replay_scorer = self.best_scorer
                else:
                    print(
                        f"[trainer-lazy] Skipping error replay ep={epoch} (improvement={improvement:.4f} < threshold)"
                    )

            # Weighted or uniform sampling of training pairs
            if hard_frame_weights is not None and effective_hfr > 0:
                # Recompute weights with ramped ratio (cheap: just re-exponentiate cached ranks)
                hard_frame_weights = _apply_hard_frame_ratio(effective_hfr)
                perm = torch.multinomial(hard_frame_weights, min(train_size, n_train), replacement=False)
            else:
                perm = torch.randperm(n_train)[:train_size]

            accum = cfg.accum_steps
            self.optimizer.zero_grad(set_to_none=True)
            epoch_grad_norms: list[float] = []

            _oom_count = 0
            for step, pair_idx in enumerate(perm):
                start = train_pair_starts[pair_idx.item()]

                # Build pair on-the-fly (CPU → device)
                try:
                    comp_pair = pair_from_frames(comp_frames, start).to(self.device)
                    gt_pair = pair_from_frames(gt_frames, start).to(self.device)
                except torch.cuda.OutOfMemoryError:
                    _oom_count += 1
                    if _oom_count <= 3:
                        print(f"[trainer-lazy] CUDA OOM loading pair (step {step}), skipping")
                    torch.cuda.empty_cache()
                    continue

                # Apply filter
                try:
                    filtered = self._apply_filter_to_pair(comp_pair)
                except torch.cuda.OutOfMemoryError:
                    _oom_count += 1
                    if _oom_count <= 3:
                        print(f"[trainer-lazy] CUDA OOM in forward (step {step}), skipping")
                    del comp_pair, gt_pair
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # Apply eval-matched roundtrip (simulate contest eval resize chain)
                if cfg.eval_roundtrip:
                    from tac.renderer import simulate_eval_roundtrip
                    from tac.camera import CAMERA_H, CAMERA_W
                    # filtered is (B, 2, H, W, 3) HWC — convert to CHW for roundtrip
                    B_f, T_f, H_f, W_f, C_f = filtered.shape
                    flat_chw = filtered.reshape(B_f * T_f, H_f, W_f, C_f).permute(0, 3, 1, 2).contiguous()
                    flat_chw = simulate_eval_roundtrip(
                        flat_chw, target_h=CAMERA_H, target_w=CAMERA_W,
                        noise_std=cfg.roundtrip_noise_std,
                    )
                    # Back to HWC: (B*T, C, H, W) → (B*T, H, W, C) → (B, T, H, W, C)
                    filtered = flat_chw.permute(0, 2, 3, 1).contiguous().reshape(B_f, T_f, H_f, W_f, C_f)

                    # GT roundtrip (no noise, no grad)
                    with torch.no_grad():
                        gt_chw = gt_pair.reshape(B_f * T_f, H_f, W_f, C_f).permute(0, 3, 1, 2).contiguous()
                        gt_chw = simulate_eval_roundtrip(
                            gt_chw, target_h=CAMERA_H, target_w=CAMERA_W,
                            noise_std=0.0,
                        )
                        gt_pair = gt_chw.permute(0, 2, 3, 1).contiguous().reshape(B_f, T_f, H_f, W_f, C_f)

                # P0: Load cached GT scorer outputs (avoids redundant GT forward pass)
                # Skip GT cache when eval_roundtrip is on — cached values were
                # computed without roundtrip, so they don't match the roundtripped
                # filtered output. Force recomputation through the roundtripped gt_pair.
                _cached_gt = None if cfg.eval_roundtrip else gt_scorer_cache.get(start)
                if _cached_gt is not None:
                    _gt_pose_6 = _cached_gt["pose_6"].to(self.device)
                    _gt_seg_soft = _cached_gt["seg_soft"].to(self.device)
                else:
                    _gt_pose_6 = None
                    _gt_seg_soft = None

                # Scorer loss (configurable mode)
                if cfg.loss_mode == "temperature":
                    # Anneal temperature from start to end over training
                    progress = epoch / max(cfg.epochs - 1, 1)
                    if cfg.temp_schedule == "exponential" and cfg.temperature_start > cfg.temperature_end:
                        temp = cfg.temperature_start * (cfg.temperature_end / cfg.temperature_start) ** progress
                    else:
                        temp = cfg.temperature_start + progress * (cfg.temperature_end - cfg.temperature_start)
                    loss, pd, sd = temperature_scorer_loss(
                        filtered,
                        gt_pair,
                        posenet,
                        segnet,
                        temperature=temp,
                    )
                elif cfg.loss_mode == "focal_ste":
                    loss, pd, sd = focal_segnet_ste_loss(
                        filtered,
                        gt_pair,
                        posenet,
                        segnet,
                        gamma=cfg.focal_gamma,
                    )
                elif cfg.loss_mode == "kl_distill":
                    if (
                        cfg.kl_distill_scope != "primary_scorer"
                        or not cfg.allow_banned_primary_kl_distill
                        or cfg.promotion_eligible
                    ):
                        raise RuntimeError(
                            "Trainer.fit_lazy would execute the legacy primary "
                            "kl_distill_scorer_loss path, which is banned for "
                            "promotion after PoseNet collapse. Use the SegNet-only "
                            "auxiliary API instead, or set "
                            "kl_distill_scope='primary_scorer', "
                            "allow_banned_primary_kl_distill=True, and "
                            "promotion_eligible=False for a forensic run."
                        )
                    # Hinton-style KL distillation: T anneals from 5.0 → 0.5
                    progress = epoch / max(cfg.epochs - 1, 1)
                    if cfg.temp_schedule == "exponential" and cfg.temperature_start > cfg.temperature_end:
                        temp = cfg.temperature_start * (cfg.temperature_end / cfg.temperature_start) ** progress
                    else:
                        temp = cfg.temperature_start + progress * (cfg.temperature_end - cfg.temperature_start)
                    bm = self._boundary_masks.get(start) if self._boundary_masks else None

                    # Adaptive or static weights
                    # Use cached adaptive weights (computed once per epoch, not per step)
                    sw = getattr(self, "_cached_sw", cfg.segnet_loss_weight)
                    bw = getattr(self, "_cached_bw", self._effective_boundary_weight)
                    if not self._adaptive:
                        # Static boundary anneal for non-adaptive mode
                        if cfg.boundary_anneal and temp > 0:
                            bw = self._effective_boundary_weight * min(3.0, cfg.temperature_start / max(temp, cfg.temperature_end))

                    loss, pd, sd = kl_distill_scorer_loss(
                        filtered,
                        gt_pair,
                        posenet,
                        segnet,
                        temperature=temp,
                        boundary_mask=bm,
                        boundary_weight=bw,
                        segnet_weight=sw,
                    )
                elif cfg.loss_mode == "pcgrad":
                    # Non-opposing gradient: decouple PoseNet and SegNet
                    sw = getattr(self, "_cached_sw", cfg.segnet_loss_weight)
                    is_first_microbatch = step % accum == 0
                    if _gt_pose_6 is not None:
                        # P0: use cached GT scorer outputs
                        loss, pd, sd, _conflict = scorer_loss_pcgrad_cached(
                            filtered,
                            _gt_pose_6,
                            _gt_seg_soft,
                            posenet,
                            segnet,
                            segnet_weight=sw,
                            do_projection=is_first_microbatch,
                        )
                    else:
                        loss, pd, sd, _conflict = scorer_loss_pcgrad(
                            filtered,
                            gt_pair,
                            posenet,
                            segnet,
                            segnet_weight=sw,
                            do_projection=is_first_microbatch,
                        )
                    # Council requirement: log conflict frequency per epoch
                    if is_first_microbatch:
                        self._epoch_pcgrad_total = getattr(self, "_epoch_pcgrad_total", 0) + 1
                        if _conflict:
                            self._epoch_pcgrad_conflicts = getattr(self, "_epoch_pcgrad_conflicts", 0) + 1
                elif cfg.loss_mode == "feature_match":
                    # Trick 1: intermediate PoseNet feature matching
                    loss, pd, sd = feature_matching_loss(
                        filtered,
                        gt_pair,
                        posenet,
                        segnet,
                        segnet_weight=cfg.segnet_loss_weight,
                    )
                elif cfg.loss_mode == "segnet_kl":
                    # SegNet KL-divergence: direct semantic preservation
                    # Standard loss + KL term on SegNet logits
                    sw = getattr(self, "_cached_sw", cfg.segnet_loss_weight)
                    if _gt_pose_6 is not None:
                        loss, pd, sd = scorer_loss_cached(
                            filtered, _gt_pose_6, _gt_seg_soft,
                            posenet, segnet,
                        )
                    else:
                        loss, pd, sd = scorer_loss(
                            filtered, gt_pair, posenet, segnet,
                        )
                    # Compute GT SegNet log-probs for KL loss
                    # gt_pair is (B,T,H,W,C) but segnet.preprocess_input expects (B,T,C,H,W)
                    with torch.no_grad():
                        _gt_seg_input = segnet.preprocess_input(_hwc_to_chw(gt_pair))
                        _gt_seg_logits = segnet(_gt_seg_input)
                        _gt_log_probs = torch.nn.functional.log_softmax(_gt_seg_logits, dim=1)
                    kl_loss, _kl_val, _kl_disagree = segnet_kl_divergence_loss(
                        _gt_log_probs, filtered, segnet,
                    )
                    loss = loss + sw * kl_loss
                elif cfg.loss_mode == "posenet_embedding":
                    # PoseNet embedding: perceptual loss on internal features
                    if _gt_pose_6 is not None:
                        loss, pd, sd = scorer_loss_cached(
                            filtered, _gt_pose_6, _gt_seg_soft,
                            posenet, segnet,
                        )
                    else:
                        loss, pd, sd = scorer_loss(
                            filtered, gt_pair, posenet, segnet,
                        )
                    emb_term = posenet_embedding_loss(
                        gt_pair, filtered, posenet,
                        layer=cfg.posenet_embedding_layer,
                    )
                    loss = loss + cfg.posenet_embedding_weight * emb_term
                else:
                    _eff_bw = self._effective_boundary_weight
                    if _eff_bw > 1.0 and self._boundary_masks is not None:
                        bm = self._boundary_masks.get(start)
                        loss, pd, sd = segnet_ste_loss(
                            filtered,
                            gt_pair,
                            posenet,
                            segnet,
                            boundary_mask=bm,
                            boundary_weight=_eff_bw,
                        )
                    elif _gt_pose_6 is not None:
                        # P0: use cached GT scorer outputs (standard loss)
                        loss, pd, sd = scorer_loss_cached(
                            filtered,
                            _gt_pose_6,
                            _gt_seg_soft,
                            posenet,
                            segnet,
                        )
                    else:
                        loss, pd, sd = scorer_loss(filtered, gt_pair, posenet, segnet)

                # Saliency reconstruction (frame 1 only)
                sal_w = saliency_for_pair(raw_saliency, start, cfg.alpha, self.device)
                filtered_bchw = filtered[:, 1].permute(0, 3, 1, 2)
                comp_bchw = comp_pair[:, 1].float().permute(0, 3, 1, 2)

                # Exploit #5: VP saliency prior — modulate per-pixel sal weights
                sal_w_frame = sal_w[1:2]
                if self._vp_saliency_map is not None:
                    vp_map = self._vp_saliency_map
                    if vp_map.shape[-2:] != sal_w_frame.shape[-2:]:
                        vp_map = nn.functional.interpolate(
                            vp_map, size=sal_w_frame.shape[-2:],
                            mode="bilinear", align_corners=False,
                        )
                    sal_w_frame = sal_w_frame * vp_map

                if cfg.use_dual_saliency and hasattr(self, "_boundary_masks") and self._boundary_masks is not None:
                    bm = self._boundary_masks.get(start)
                    sal_recon = dual_saliency_reconstruction_loss(
                        filtered_bchw,
                        comp_bchw,
                        posenet_sal=sal_w_frame,
                        segnet_boundary=bm,
                        alpha_pose=cfg.alpha,
                        alpha_seg=cfg.alpha_seg,
                    )
                else:
                    sal_recon = saliency_reconstruction_loss(filtered_bchw, comp_bchw, sal_w_frame)

                # Trick 3: Even-indexed pair SegNet skip — for even-indexed pairs,
                # SegNet contribution is halved. With standard pair layout
                # (starts at 0,2,4...) frame_t1 is always odd, so the old check
                # `(start + 1) % 2 == 0` never fired. Fixed: use pair_idx.
                if cfg.even_frame_skip_seg and pair_idx % 2 == 0:
                    # BUG FIX: This used to halve the ENTIRE loss including PoseNet,
                    # not just SegNet. PoseNet is extremely sensitive (29x regression
                    # history) and should never have reduced gradients. Since the loss
                    # components are entangled in scorer_loss, we cannot cleanly skip
                    # only SegNet here. The safe action: skip this pair entirely for
                    # SegNet purposes but keep PoseNet at full strength. Since we
                    # cannot separate them, we disable this trick with a clear error.
                    raise ValueError(
                        "even_frame_skip_seg=True is broken: it halves PoseNet loss "
                        "on 50% of training pairs. This was the root cause of PoseNet "
                        "degradation. Disabled until loss functions return per-component "
                        "values. Set even_frame_skip_seg=False."
                    )

                # Trick 2: Frequency-domain wavelet loss (additive)
                if cfg.use_frequency_loss and cfg.frequency_loss_weight > 0:
                    freq_loss = frequency_aware_loss(filtered, gt_pair)
                    loss = loss + cfg.frequency_loss_weight * freq_loss

                # Apply learnable loss weights if enabled
                if self._log_w_seg is not None and self._log_w_pose is not None:
                    w_seg = torch.exp(self._log_w_seg)
                    w_pose = torch.exp(self._log_w_pose)
                    # Reweight: scorer loss already combines pose+seg; we scale
                    # the scorer loss by w_pose (dominant PoseNet signal) and add
                    # a w_seg multiplier on the saliency reconstruction which
                    # correlates with SegNet boundary fidelity.
                    loss = w_pose * loss
                    sal_recon = w_seg * sal_recon

                total_unscaled = loss + cfg.sal_lambda * sal_recon
                if self.entropy_bottleneck is not None and cfg.eb_lambda > 0.0:
                    total_unscaled = total_unscaled + cfg.eb_lambda * self.entropy_bottleneck.rate_loss()
                total = total_unscaled / accum
                # Deep hardening pass 3 dim 3: assert finite loss before backward.
                _loss_val = total.item()
                if not (_loss_val == _loss_val and abs(_loss_val) != float("inf")):
                    raise RuntimeError(
                        f"[Trainer-lazy] non-finite loss at step {step}: "
                        f"total={_loss_val} loss={loss.item()} sal={sal_recon.item()}. "
                        f"Aborting before backward to preserve weights."
                    )
                try:
                    total.backward()
                except torch.cuda.OutOfMemoryError:
                    _oom_count += 1
                    if _oom_count <= 3:
                        print(f"[trainer-lazy] CUDA OOM in backward (step {step}), skipping")
                    del comp_pair, gt_pair, filtered, filtered_bchw, comp_bchw, sal_w
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                total_loss += loss.item()
                total_pose += pd
                total_seg += sd

                # Gradient accumulation: step every accum_steps
                if (step + 1) % accum == 0 or (step + 1) == len(perm):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    epoch_grad_norms.append(grad_norm.item())
                    self.optimizer.step()
                    self.ema.update(self.model)
                    self.optimizer.zero_grad(set_to_none=True)

                # Free pair memory immediately
                del comp_pair, gt_pair, filtered, filtered_bchw, comp_bchw, sal_w, _gt_pose_6, _gt_seg_soft, _cached_gt

            if _oom_count > 0 and (epoch == 0 or epoch % 50 == 0):
                print(f"[trainer-lazy] OOM events this epoch: {_oom_count}")

            if epoch >= cfg.warmup_epochs:
                self.scheduler.step()

            # SWA: snapshot weights in final 20% of training for wider minima
            if cfg.use_swa and epoch >= int(cfg.epochs * 0.8):
                if not hasattr(self, "_swa"):
                    self._swa = SWA()
                    print(f"[trainer-lazy] SWA started at epoch {epoch}")
                # Bug #1 fix: pass EMA state_dict, not the raw model
                self._swa.update(self.ema)

            # Empty CUDA cache once per epoch (not per step — avoids sync stalls)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            n = len(perm)
            avg_loss = total_loss / n
            avg_pose = total_pose / n
            avg_seg = total_seg / n

            # Best-checkpoint int8 evaluation — three-phase frequency
            epoch_frac = epoch / max(cfg.epochs - 1, 1)
            if epoch_frac >= 0.8:
                eval_freq = 1
            elif epoch_frac >= 0.2:
                eval_freq = max(2, cfg.eval_every // 2)
            else:
                eval_freq = cfg.eval_every
            is_eval_epoch = (epoch + 1) % eval_freq == 0 or epoch == cfg.epochs - 1 or epoch == 0
            if is_eval_epoch:
                scorer_val = self._evaluate_int8_lazy(
                    comp_frames,
                    gt_frames,
                    posenet,
                    segnet,
                    eval_pair_starts=eval_pair_starts,
                )
            else:
                scorer_val = self.best_scorer  # reuse last known

            # Adaptive boundary: adjust boundary_weight based on SegNet feedback
            if cfg.adaptive_boundary and is_eval_epoch and self._last_eval_seg is not None:
                cur_seg = self._last_eval_seg
                if self._prev_seg_distortion is not None:
                    seg_delta = cur_seg - self._prev_seg_distortion
                    if seg_delta < -1e-6:
                        # SegNet improving: reduce boundary pressure (diminishing returns)
                        self._adaptive_boundary_weight *= 0.9
                    elif seg_delta > 1e-6:
                        # SegNet stagnating/regressing: increase boundary pressure
                        self._adaptive_boundary_weight *= 1.1
                    # Clamp to reasonable range
                    self._adaptive_boundary_weight = max(0.1, min(500.0, self._adaptive_boundary_weight))
                self._prev_seg_distortion = cur_seg

            if scorer_val < self.best_scorer:
                self.best_scorer = scorer_val
                self.best_epoch = epoch
                self._save_checkpoint(epoch, scorer_val)
                print(f"  ** NEW BEST: ep {epoch}, scorer {scorer_val:.4f} **")

            lr = self.optimizer.param_groups[0]["lr"]
            eval_tag = "*" if is_eval_epoch else " "
            # PCGrad conflict telemetry (council requirement — per-epoch, resets each epoch)
            conflict_str = ""
            if cfg.loss_mode == "pcgrad":
                ep_total = getattr(self, "_epoch_pcgrad_total", 0)
                ep_conflicts = getattr(self, "_epoch_pcgrad_conflicts", 0)
                if ep_total > 0:
                    conflict_str = f" conflict={ep_conflicts}/{ep_total}={ep_conflicts / ep_total:.0%}"
                # Reset for next epoch
                self._epoch_pcgrad_total = 0
                self._epoch_pcgrad_conflicts = 0
            print(
                f"[ep {epoch:4d}]{eval_tag} loss={avg_loss:.4f} pose={avg_pose:.6f} "
                f"seg={avg_seg:.6f} scorer={scorer_val:.4f} best={self.best_scorer:.4f} lr={lr:.6f}{conflict_str}"
            )

            # LR plateau detection for standard loss
            if is_eval_epoch and cfg.loss_mode in ("standard", "pcgrad"):
                self._plateau_window.append(scorer_val)
                if len(self._plateau_window) > 20:
                    self._plateau_window.pop(0)
                if len(self._plateau_window) >= 20 and not self._plateau_reduced and epoch > cfg.epochs * 0.3:
                    recent_best = min(self._plateau_window)
                    window_start_best = min(self._plateau_window[:10])
                    if recent_best >= window_start_best - 0.005:
                        for pg in self.optimizer.param_groups:
                            pg["lr"] *= 0.5
                        self._plateau_reduced = True
                        print(
                            f"[trainer-lazy] LR plateau detected ep={epoch}, halving LR to {self.optimizer.param_groups[0]['lr']:.6f}"
                        )

            # JSONL telemetry log — structured data for council analysis
            if is_eval_epoch:
                telemetry_path = Path(cfg.output_dir) / f"telemetry_{cfg.tag}.jsonl"
                telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                import time as _time

                with open(telemetry_path, "a") as tf:
                    avg_grad_norm = sum(epoch_grad_norms) / max(len(epoch_grad_norms), 1)
                    entry = {
                        "epoch": epoch,
                        "scorer": round(scorer_val, 6),
                        "eval_pose": self._last_eval_pose,
                        "eval_seg": self._last_eval_seg,
                        "train_pose": round(avg_pose, 8),
                        "train_seg": round(avg_seg, 8),
                        "train_loss": round(avg_loss, 6),
                        "avg_grad_norm": round(avg_grad_norm, 6),
                        "lr": round(lr, 8),
                        "best_scorer": round(self.best_scorer, 6),
                        "best_epoch": self.best_epoch,
                        "ts": _time.time(),
                        "loss_mode": cfg.loss_mode,
                        "variant": cfg.variant,
                    }
                    # Adaptive weight diagnostics
                    if hasattr(self, "_cached_sw"):
                        entry["adaptive_sw"] = round(self._cached_sw, 4)
                    if hasattr(self, "_cached_bw"):
                        entry["adaptive_bw"] = round(self._cached_bw, 2)
                    # Learnable loss weight diagnostics
                    if self._log_w_seg is not None:
                        entry["learned_w_seg"] = round(math.exp(self._log_w_seg.item()), 4)
                        entry["learned_w_pose"] = round(math.exp(self._log_w_pose.item()), 4)
                    # Adaptive boundary diagnostics
                    if cfg.adaptive_boundary:
                        entry["adaptive_boundary_weight"] = round(self._adaptive_boundary_weight, 4)
                    # Proxy hardening diagnostics
                    if hasattr(self, "_proxy_confidence"):
                        entry["proxy_confidence"] = self._proxy_confidence
                    if hasattr(self, "_corrected_scorer"):
                        entry["corrected_scorer"] = self._corrected_scorer
                    if hasattr(self, "_baseline_pose") and self._baseline_pose:
                        entry["baseline_pose"] = self._baseline_pose
                    tf.write(json.dumps(entry) + "\n")

            # Save training state every 50 epochs for crash recovery
            if epoch % 50 == 0 and epoch > 0:
                self.save_training_state()

            # Wall-clock timeout: save and exit cleanly before platform kills us
            if self._wall_clock_exceeded():
                elapsed = _time_module.monotonic() - self._start_wall_time
                print(
                    f"\n[trainer-lazy] WALL-CLOCK TIMEOUT at epoch {epoch} "
                    f"(elapsed {elapsed / 3600:.1f}h, limit {self.config.wall_clock_timeout / 3600:.1f}h)"
                )
                self.save_training_state()
                print(f"[trainer-lazy] Timeout exit. Best: {self.best_scorer:.4f} at epoch {self.best_epoch}")
                return self.best_scorer

        # Apply SWA if it was active — average replaces EMA, then re-evaluate
        if cfg.use_swa and hasattr(self, "_swa") and self._swa.count > 0:
            self._swa.apply(self.ema)
            # Re-evaluate with SWA-averaged weights to see if it's better
            if eval_pair_starts is not None:
                swa_scorer = self._evaluate_int8_lazy(comp_frames, gt_frames, posenet, segnet, eval_pair_starts)
                if swa_scorer < self.best_scorer:
                    print(f"  ** SWA IMPROVED: {self.best_scorer:.4f} -> {swa_scorer:.4f} **")
                    self.best_scorer = swa_scorer
                    self.best_epoch = cfg.epochs
                    self._save_checkpoint(cfg.epochs, swa_scorer)
                else:
                    print(f"[SWA] No improvement ({swa_scorer:.4f} vs best {self.best_scorer:.4f})")

        # Final save
        self.save_training_state()
        self._save_training_cost(cfg.epochs - 1)
        print(f"[trainer-lazy] Complete. Best: {self.best_scorer:.4f} at epoch {self.best_epoch}")
        return self.best_scorer

    def _evaluate_int8_lazy(
        self,
        comp_frames,
        gt_frames,
        posenet,
        segnet,
        eval_pair_starts: list[int] | None = None,
    ) -> float:
        """Evaluate EMA model after int8 quantization on HELD-OUT pairs only.

        Uses eval_pair_starts (set by fit_lazy's train/eval partition).
        These pairs are NEVER seen during training — no leakage.
        """
        orig_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.ema.apply(self.model)
        q_state = quantize_state_dict(self.model.state_dict(), per_channel=True)
        self.model.load_state_dict(q_state)
        self.model.eval()

        if eval_pair_starts is None:
            # Fallback: use last 25% as eval (matches fit_lazy partition)
            all_starts = pair_start_indices(len(comp_frames))
            eval_size = max(1, len(all_starts) // 4)
            eval_pair_starts = all_starts[-eval_size:]

        total_p, total_s, count = 0.0, 0.0, 0

        # P1: autocast for CUDA and MPS (fp16 on scorers)
        use_autocast = (str(self.device).startswith("cuda") and torch.cuda.is_available()) or str(self.device) == "mps"
        autocast_device = "cuda" if str(self.device).startswith("cuda") else "mps"
        autocast = torch.amp.autocast(autocast_device, enabled=use_autocast)
        with torch.no_grad(), autocast:
            for start in eval_pair_starts:
                comp_pair = pair_from_frames(comp_frames, start).to(self.device)
                gt_pair = pair_from_frames(gt_frames, start).to(self.device)
                filtered = self._apply_filter_to_pair(comp_pair)
                # uint8 round-trip: matches official inflate → scorer pipeline exactly
                filtered = filtered.round().clamp(0, 255).to(torch.uint8).float()
                score, pd, sd = eval_scorer_loss(filtered, gt_pair, posenet, segnet)
                total_p += pd
                total_s += sd
                count += 1
                del comp_pair, gt_pair, filtered
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        count = max(count, 1)
        avg_p = total_p / count
        avg_s = total_s / count
        scorer = 100.0 * avg_s + math.sqrt(10.0 * avg_p)

        # Store per-component for telemetry (meta.json, regression alarm)
        self._last_eval_pose = round(avg_p, 8)
        self._last_eval_seg = round(avg_s, 8)

        # Baseline watermark: record raw compressed distortion at first eval
        if not hasattr(self, "_baseline_pose") or self._baseline_pose is None:
            self._baseline_pose = round(avg_p, 8)
            self._baseline_seg = round(avg_s, 8)
            print(f"[eval] Baseline watermark: pose={avg_p:.6f}, seg={avg_s:.6f}")

        # Proxy confidence: high when pose is in a credible range relative to baseline.
        # Drops when pose regresses (ratio > 2) OR improves implausibly (ratio < 0.05).
        # Normal training drives pose from baseline toward ~0.3x baseline over hundreds of epochs.
        proxy_confidence = 1.0
        # R41 fix: `truthy and >0` short-circuits when _baseline_pose=0.0 (falsy).
        # The first `> 0` already handles None via getattr default + comparison guard.
        _bp = getattr(self, "_baseline_pose", None)
        if _bp is not None and _bp > 0:
            pose_ratio = avg_p / _bp

            if pose_ratio > 2.0:
                # PoseNet regressing — proxy may be masking damage
                proxy_confidence = max(0.1, 1.0 / pose_ratio)
                print(
                    f"  !! PROXY CONFIDENCE LOW (regression): pose={avg_p:.6f} is "
                    f"{pose_ratio:.1f}x baseline — authoritative eval recommended !!"
                )
            elif pose_ratio < 0.05:
                # PoseNet suspiciously good — may be noise floor or measurement artifact
                proxy_confidence = 0.5
                print(
                    f"  !! PROXY CONFIDENCE MODERATE: pose={avg_p:.6f} is "
                    f"{pose_ratio:.2f}x baseline (suspiciously low) !!"
                )

            # Regression alarm: warn if PoseNet regresses 3x from baseline
            if pose_ratio > 3.0:
                print(
                    f"  !! POSENET REGRESSION ALARM: {avg_p:.6f} is {pose_ratio:.1f}x baseline {self._baseline_pose:.6f} !!"
                )
            if pose_ratio > 5.0:
                print(f"  !! CRITICAL: PoseNet {pose_ratio:.0f}x baseline — checkpoint NOT saved !!")
                self.model.load_state_dict(orig_state)
                self.model.train()
                return float("inf")  # prevent promotion of regressed checkpoint

        # Store proxy confidence for telemetry
        self._proxy_confidence = round(proxy_confidence, 4)

        # Corrected score estimate using proxy correction factors (α_p, α_s)
        # These are calibrated from authoritative eval runs. Default 1.0 = no correction.
        alpha_p = getattr(self, "_proxy_alpha_p", 1.0)
        alpha_s = getattr(self, "_proxy_alpha_s", 1.0)
        corrected_scorer = 100.0 * (alpha_s * avg_s) + math.sqrt(10.0 * (alpha_p * avg_p))
        self._corrected_scorer = round(corrected_scorer, 6)
        if abs(corrected_scorer - scorer) > 0.01:
            print(
                f"  [proxy-correction] raw={scorer:.4f} corrected={corrected_scorer:.4f} "
                f"(α_p={alpha_p:.2f}, α_s={alpha_s:.2f})"
            )

        self.model.load_state_dict(orig_state)
        self.model.train()
        return scorer


# ── Re-exports (Lane S V2) ──────────────────────────────────────────────
# The scorer-loss convergence detector logically belongs in this module
# but is implemented as a top-level file (`tac.scorer_loss_convergence_detector`)
# because `src/tac/training.py` exists as a flat module rather than a
# package. Re-export it here so callers can use the canonical
# `from tac.training import ScorerLossConvergenceDetector` import path.
from tac.scorer_loss_convergence_detector import ScorerLossConvergenceDetector  # noqa: E402,F401
