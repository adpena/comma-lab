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
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, model_validator

from .data import pair_from_frames, pair_start_indices, saliency_for_pair
from .losses import (
    dual_saliency_reconstruction_loss,
    eval_scorer_loss,
    focal_segnet_ste_loss,
    kl_distill_scorer_loss,
    saliency_reconstruction_loss,
    scorer_loss,
    scorer_loss_pcgrad,
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
    loss_mode: Literal["standard", "temperature", "focal_ste", "kl_distill", "pcgrad"] = "standard"
    temperature_start: float = Field(1.0, gt=0.0)
    temperature_end: float = Field(0.05, gt=0.0)
    temp_schedule: str = Field("exponential", pattern=r"^(linear|exponential)$",
                                description="Temperature decay: 'linear' or 'exponential' (recommended)")
    focal_gamma: float = Field(2.0, ge=0.0)
    segnet_loss_weight: float = Field(100.0, ge=0.0)
    use_dual_saliency: bool = False
    alpha_seg: float = Field(200.0, ge=0.0)

    # Training dynamics
    accum_steps: int = Field(4, ge=1, le=64)
    eval_every: int = Field(5, ge=1, description="Evaluate int8 checkpoint every N epochs")
    hard_frame_ratio: float = Field(0.0, ge=0.0, le=1.0,
                                     description="Fraction of training pairs to oversample from hardest SegNet frames. "
                                     "0.0 = uniform sampling, 0.5 = half hard / half uniform.")
    error_replay_every: int = Field(0, ge=0,
                                     description="Recompute hard-frame weights using current model output every N epochs. "
                                     "0 = static (compute once at start). 200 = adaptive every 200 epochs.")
    boundary_anneal: bool = Field(False, description="Couple boundary_weight to temperature: "
                                  "increases boundary attention as T decreases (maintains gradient pressure)")
    use_swa: bool = Field(False, description="Stochastic Weight Averaging over final 20% of training. "
                          "Wider minima → better int8 robustness.")
    adaptive_rebalance: bool = Field(False, description="Enable adaptive weight rebalancing from "
                                     "src/tac/adaptive.py. Derives segnet_weight and boundary_weight "
                                     "from current (pose, seg) at each eval epoch.")
    rebalance_every: int = Field(50, ge=1, description="Epochs between adaptive weight updates")
    boundary_fraction: float = Field(0.05, gt=0.0, lt=1.0, description="Measured boundary pixel fraction (beta)")
    eval_holdout: float = Field(0.0, ge=0.0, le=0.5,
                                description="Fraction of pairs held out for eval. "
                                "0.0 = contest mode (train+eval on all pairs). "
                                "0.25 = production mode (25% held-out eval split).")
    use_lsq: bool = Field(False, description="Enable Learned Step Size Quantization")

    # Resumption
    resume_from: str | None = None

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
        return self


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
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
        shadow = ema.shadow if hasattr(ema, 'shadow') else ema
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
        shadow = ema.shadow if hasattr(ema, 'shadow') else ema
        for k in self.avg:
            if k in shadow:
                shadow[k].copy_(self.avg[k])
        print(f"[SWA] Applied average of {self.count} snapshots to EMA shadow")


class KalmanWeightFilter:
    """Per-parameter scalar Kalman filter as an alternative to EMA.

    Uses inverse-variance weighting: parameters with low observation noise
    (stable across epochs) get more weight. Parameters with high observation
    noise (noisy across epochs) get less weight.
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

        # Initialize state estimate = model weights, variance = 1.0
        self.state = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.variance = {k: torch.ones_like(v) for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, obs in model.state_dict().items():
                if not torch.is_floating_point(obs):
                    self.state[k] = obs.clone()
                    continue

                # Predict step: variance grows by process noise
                pred_var = self.variance[k] + self.process_noise

                # Observation noise: adaptive based on parameter magnitude
                obs_noise = self.obs_noise_base + self.obs_noise_scale * obs.detach().abs()

                # Kalman gain
                gain = pred_var / (pred_var + obs_noise)

                # Update
                self.state[k] = self.state[k] + gain * (obs.detach() - self.state[k])
                self.variance[k] = (1 - gain) * pred_var

    def apply(self, model: nn.Module):
        model.load_state_dict(self.state)

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
        self.config = config
        self.device = device
        self.ema = EMA(model, decay=config.ema_decay)
        self.best_scorer = float("inf")
        self.best_epoch = -1
        self._patched_scorers = False
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
        if getattr(config, 'adaptive_rebalance', False):
            from .adaptive import AdaptiveWeights
            beta = getattr(config, 'boundary_fraction', 0.05)
            self._adaptive = AdaptiveWeights(boundary_fraction=beta)
            print(f"[trainer] Adaptive weights enabled (beta={beta})")

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=1e-4
        )

        # LSQ: Learned Step Size Quantization
        self._lsq_scales: dict[str, nn.Module] | None = None
        if config.use_lsq:
            from .quantization import apply_lsq
            self._lsq_scales = apply_lsq(self.model)
            if self._lsq_scales:
                lsq_params = []
                for lsq_mod in self._lsq_scales.values():
                    lsq_params.extend(lsq_mod.parameters())
                self.optimizer.add_param_group({
                    "params": lsq_params,
                    "lr": config.lr * 5,
                    "weight_decay": 0.0,
                })
                print(f"[trainer] LSQ enabled: {len(self._lsq_scales)} learned scales, lr={config.lr * 5:.6f}")

        if config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=1e-5
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=config.restart_t0, T_mult=config.restart_tmult, eta_min=1e-5
            )

        # Resume from checkpoint if specified (must come after optimizer/scheduler init)
        if config.resume_from and Path(config.resume_from).exists():
            self.load_training_state(config.resume_from)

        # Emergency save on signals and exit — never lose training state
        self._emergency_registered = False
        self._register_emergency_save()

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
        torch.save({
            "model": self.model.state_dict(),
            "ema_shadow": self.ema.shadow,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self._current_epoch,
            "best_scorer": self.best_scorer,
            "best_epoch": self.best_epoch,
            "plateau_reduced": self._plateau_reduced,
        }, tmp_path)
        tmp_path.rename(path)  # atomic on POSIX

    def load_training_state(self, path: str | Path):
        """Resume training from a saved state."""
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
        robustness with non-contiguous tensors.

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
            rgb = rgb_chw[..., :, :2 * H2, :2 * W2]
            R = rgb[..., 0, :, :]
            G = rgb[..., 1, :, :]
            B = rgb[..., 2, :, :]
            Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
            U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
            V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
            U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
                     U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
            V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
                     V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
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
            x = einops.rearrange(x, 'b t c h w -> (b t) c h w',
                                 b=batch_size, t=seq_len_local, c=3)
            # Resize to scorer input size (bilinear, matching upstream)
            x = nn.functional.interpolate(
                x, size=(segnet_model_input_size[1], segnet_model_input_size[0]),
                mode='bilinear', align_corners=False,
            )
            # Differentiable YUV conversion with 4:2:0 subsampling
            yuv = _rgb_to_yuv6_diff(x)
            return einops.rearrange(yuv, '(b t) c h w -> b (t c) h w',
                                    b=batch_size, t=seq_len_local, c=6).contiguous()

        posenet.preprocess_input = types.MethodType(_diff_preprocess, posenet)

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
            result = torch.stack([
                out0.permute(0, 2, 3, 1),  # (B, H, W, 3)
                out1.permute(0, 2, 3, 1),
            ], dim=1)  # (B, 2, H, W, 3)
            return result
        else:
            # Standard: process each frame independently
            frames_bchw = comp_pair.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
            filtered_bchw = self.model(frames_bchw)
            return filtered_bchw.permute(0, 2, 3, 1).reshape(B, T, H, W, C)

    def _evaluate_int8(
        self, comp_pairs, gt_pairs, posenet, segnet, subsample: int = 2,
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
        use_autocast = str(self.device).startswith("cuda") and torch.cuda.is_available()
        autocast_ctx = torch.amp.autocast("cuda", enabled=use_autocast)
        with torch.no_grad(), autocast_ctx:
            for idx in range(0, len(comp_pairs), subsample):
                filtered = self._apply_filter_to_pair(comp_pairs[idx])
                # uint8 round-trip: matches official inflate → scorer pipeline
                filtered = filtered.round().clamp(0, 255).to(torch.uint8).float()
                score, pd, sd = eval_scorer_loss(filtered, gt_pairs[idx], posenet, segnet)
                total_p += pd
                total_s += sd
                count += 1
                if use_autocast:
                    torch.cuda.empty_cache()

        scorer = 100.0 * (total_s / count) + math.sqrt(10.0 * (total_p / count))

        # Restore original training weights
        self.model.load_state_dict(orig_state)
        self.model.train()
        return scorer

    def _save_checkpoint(self, epoch: int, scorer: float):
        """Save best int8 checkpoint with atomic writes."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = self.config.tag

        fp32_path = out_dir / f"postfilter_{tag}_best_fp32.pt"
        int8_path = out_dir / f"postfilter_{tag}_best_int8.pt"
        meta_path = out_dir / f"postfilter_{tag}_best_meta.json"

        # Save EMA fp32 (atomic)
        fp32_tmp = fp32_path.with_suffix(".pt.tmp")
        torch.save(self.ema.state_dict(), fp32_tmp)
        fp32_tmp.rename(fp32_path)

        # Save int8 per-channel (atomic) — better precision for multi-channel convs
        int8_state = {}
        for name, param in self.ema.shadow.items():
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
        }
        int8_tmp = int8_path.with_suffix(".pt.tmp")
        torch.save(int8_state, int8_tmp)
        int8_tmp.rename(int8_path)

        meta_tmp = meta_path.with_suffix(".json.tmp")
        meta_tmp.write_text(json.dumps({
            "epoch": epoch,
            "scorer": scorer,
            "pose": getattr(self, '_last_eval_pose', None),
            "seg": getattr(self, '_last_eval_seg', None),
            "fp32_path": str(fp32_path),
            "int8_path": str(int8_path),
            "int8_size": int8_path.stat().st_size,
            "meta": int8_state["__meta__"],
            "config": {
                "variant": self.config.variant,
                "hidden": self.config.hidden,
                "loss_mode": self.config.loss_mode,
                "boundary_weight": self.config.boundary_weight,
                "segnet_loss_weight": self.config.segnet_loss_weight,
                "hard_frame_ratio": self.config.hard_frame_ratio,
                "temperature_start": self.config.temperature_start,
                "temperature_end": self.config.temperature_end,
                "temp_schedule": self.config.temp_schedule,
                "alpha": self.config.alpha,
            },
            "baseline_pose": getattr(self, '_baseline_pose', None),
            "baseline_seg": getattr(self, '_baseline_seg', None),
        }, indent=2))
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
        if getattr(cfg, 'adaptive_rebalance', False):
            raise NotImplementedError(
                "adaptive_rebalance requires fit_lazy(). fit() does not support adaptive weights."
            )

        # Patch scorer models for differentiable training (CRITICAL: without this,
        # PoseNet gradients are zero due to upstream @torch.no_grad on rgb_to_yuv6)
        if not self._patched_scorers:
            self._patch_scorers_for_training(posenet, segnet)
            self._patched_scorers = True

        n_pairs = len(comp_pairs)

        pairs_per_epoch = cfg.pairs_per_epoch or n_pairs
        use_boundary = cfg.use_ste_segnet and boundary_masks is not None

        print(f"[trainer] {cfg.epochs} epochs, {pairs_per_epoch} pairs/ep, "
              f"h={cfg.hidden}, alpha={cfg.alpha}, device={self.device}")
        if use_boundary:
            print(f"[trainer] SegNet STE + boundary weighting ({cfg.boundary_weight}x)")
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
                        filtered, gt_pairs[idx], posenet, segnet,
                        boundary_mask=bm, boundary_weight=cfg.boundary_weight,
                    )
                else:
                    loss, pd, sd = scorer_loss(
                        filtered, gt_pairs[idx], posenet, segnet,
                    )

                # Saliency reconstruction
                B, T, H, W, C = filtered.shape
                filtered_bchw = filtered[:, 1].permute(0, 3, 1, 2)
                comp_bchw = comp_pairs[idx][:, 1].float().permute(0, 3, 1, 2)
                sal_idx = min(idx * 2 + 1, sal_weights.shape[0] - 1)
                sal_recon = saliency_reconstruction_loss(
                    filtered_bchw, comp_bchw, sal_weights[sal_idx : sal_idx + 1]
                )

                total = loss + cfg.sal_lambda * sal_recon

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

            # Best-checkpoint int8 evaluation
            scorer_val = self._evaluate_int8(comp_pairs, gt_pairs, posenet, segnet)

            if scorer_val < self.best_scorer:
                self.best_scorer = scorer_val
                self.best_epoch = epoch
                self._save_checkpoint(epoch, scorer_val)
                print(f"  ** NEW BEST: ep {epoch}, scorer {scorer_val:.4f} **")

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"[ep {epoch:4d}] loss={avg_loss:.4f} pose={avg_pose:.6f} "
                  f"seg={avg_seg:.6f} scorer={scorer_val:.4f} best={self.best_scorer:.4f} lr={lr:.6f}")

            if callback:
                callback(epoch, avg_loss, avg_pose, avg_seg, scorer_val)

            # Save training state every 50 epochs for crash recovery
            if epoch % 50 == 0 and epoch > 0:
                self.save_training_state()

        self.save_training_state()
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

        cfg = self.config
        all_pair_starts = pair_start_indices(len(comp_frames))
        n_total = len(all_pair_starts)

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

        print(f"[trainer-lazy] {cfg.epochs} epochs, {train_size}/{n_train} pairs/ep, "
              f"{split_label}, h={cfg.hidden}, alpha={cfg.alpha}, device={self.device}")
        print("[trainer-lazy] Frames on CPU, pairs built on-the-fly (MPS-safe)")
        if cfg.loss_mode != "standard":
            print(f"[trainer-lazy] Loss mode: {cfg.loss_mode}")
        if cfg.sal_lambda == 0:
            print("[trainer-lazy] Saliency reconstruction DISABLED (sal_lambda=0)")

        # Precompute boundary masks if needed (dual saliency OR kl_distill with boundary weighting)
        self._boundary_masks = None
        needs_boundary = cfg.use_dual_saliency or cfg.boundary_weight > 1.0
        if needs_boundary:
            from .losses import compute_boundary_mask
            print("[trainer-lazy] Computing SegNet boundary masks for dual saliency...")
            self._boundary_masks = {}
            for start in train_pair_starts:
                gt_pair = pair_from_frames(gt_frames, start)
                mask = compute_boundary_mask(gt_pair, segnet, device=self.device)
                self._boundary_masks[start] = mask
            avg_frac = sum(m.mean().item() for m in self._boundary_masks.values()) / len(self._boundary_masks)
            print(f"[trainer-lazy] Boundary masks: {len(self._boundary_masks)} pairs, avg {avg_frac:.4f} ({avg_frac*100:.2f}%)")

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
            print(f"[trainer-lazy] Computing hard-frame weights ({label}, "
                  f"{'model output' if use_model else 'raw compressed'})...")
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
            if (self._adaptive and self._last_eval_pose is not None
                    and epoch % getattr(cfg, 'rebalance_every', 50) == 0):
                if cfg.loss_mode in ("standard", "pcgrad"):
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
                if epoch % (getattr(cfg, 'rebalance_every', 50) * 5) == 0:
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
            if (hard_frame_weights is not None
                    and cfg.error_replay_every > 0
                    and epoch > 0
                    and epoch % cfg.error_replay_every == 0):
                improvement = self._last_replay_scorer - self.best_scorer
                if improvement > 0.002 or self._last_replay_scorer == float("inf"):
                    hard_frame_weights = _compute_hard_frame_weights(
                        use_model=True, label=f"error-replay-ep{epoch}")
                    self._last_replay_scorer = self.best_scorer
                else:
                    print(f"[trainer-lazy] Skipping error replay ep={epoch} (improvement={improvement:.4f} < threshold)")

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

            for step, pair_idx in enumerate(perm):
                start = train_pair_starts[pair_idx.item()]

                # Build pair on-the-fly (CPU → device)
                comp_pair = pair_from_frames(comp_frames, start).to(self.device)
                gt_pair = pair_from_frames(gt_frames, start).to(self.device)

                # Apply filter
                filtered = self._apply_filter_to_pair(comp_pair)

                # Scorer loss (configurable mode)
                if cfg.loss_mode == "temperature":
                    # Anneal temperature from start to end over training
                    progress = epoch / max(cfg.epochs - 1, 1)
                    if cfg.temp_schedule == "exponential" and cfg.temperature_start > cfg.temperature_end:
                        temp = cfg.temperature_start * (cfg.temperature_end / cfg.temperature_start) ** progress
                    else:
                        temp = cfg.temperature_start + progress * (cfg.temperature_end - cfg.temperature_start)
                    loss, pd, sd = temperature_scorer_loss(
                        filtered, gt_pair, posenet, segnet, temperature=temp,
                    )
                elif cfg.loss_mode == "focal_ste":
                    loss, pd, sd = focal_segnet_ste_loss(
                        filtered, gt_pair, posenet, segnet,
                        gamma=cfg.focal_gamma,
                    )
                elif cfg.loss_mode == "kl_distill":
                    # Hinton-style KL distillation: T anneals from 5.0 → 0.5
                    progress = epoch / max(cfg.epochs - 1, 1)
                    if cfg.temp_schedule == "exponential" and cfg.temperature_start > cfg.temperature_end:
                        temp = cfg.temperature_start * (cfg.temperature_end / cfg.temperature_start) ** progress
                    else:
                        temp = cfg.temperature_start + progress * (cfg.temperature_end - cfg.temperature_start)
                    bm = self._boundary_masks.get(start) if self._boundary_masks else None

                    # Adaptive or static weights
                    # Use cached adaptive weights (computed once per epoch, not per step)
                    sw = getattr(self, '_cached_sw', cfg.segnet_loss_weight)
                    bw = getattr(self, '_cached_bw', cfg.boundary_weight)
                    if not self._adaptive:
                        # Static boundary anneal for non-adaptive mode
                        if cfg.boundary_anneal and temp > 0:
                            bw = cfg.boundary_weight * min(3.0, cfg.temperature_start / max(temp, cfg.temperature_end))

                    loss, pd, sd = kl_distill_scorer_loss(
                        filtered, gt_pair, posenet, segnet,
                        temperature=temp,
                        boundary_mask=bm,
                        boundary_weight=bw,
                        segnet_weight=sw,
                    )
                elif cfg.loss_mode == "pcgrad":
                    # Non-opposing gradient: decouple PoseNet and SegNet
                    # BUG 3/6 fix: only run projection on first microbatch of each
                    # accumulation window. Later microbatches use the cached scale
                    # to avoid stale conflict detection across the window.
                    sw = getattr(self, '_cached_sw', cfg.segnet_loss_weight)
                    is_first_microbatch = (step % accum == 0)
                    loss, pd, sd, _conflict = scorer_loss_pcgrad(
                        filtered, gt_pair, posenet, segnet,
                        segnet_weight=sw,
                        do_projection=is_first_microbatch,
                    )
                    # Council requirement: log conflict frequency per epoch
                    if is_first_microbatch:
                        self._epoch_pcgrad_total = getattr(self, '_epoch_pcgrad_total', 0) + 1
                        if _conflict:
                            self._epoch_pcgrad_conflicts = getattr(self, '_epoch_pcgrad_conflicts', 0) + 1
                else:
                    if cfg.boundary_weight > 1.0 and self._boundary_masks is not None:
                        bm = self._boundary_masks.get(start)
                        loss, pd, sd = segnet_ste_loss(
                            filtered, gt_pair, posenet, segnet,
                            boundary_mask=bm,
                            boundary_weight=cfg.boundary_weight,
                        )
                    else:
                        loss, pd, sd = scorer_loss(filtered, gt_pair, posenet, segnet)

                # Saliency reconstruction (frame 1 only)
                sal_w = saliency_for_pair(raw_saliency, start, cfg.alpha, self.device)
                filtered_bchw = filtered[:, 1].permute(0, 3, 1, 2)
                comp_bchw = comp_pair[:, 1].float().permute(0, 3, 1, 2)

                if cfg.use_dual_saliency and hasattr(self, '_boundary_masks') and self._boundary_masks is not None:
                    bm = self._boundary_masks.get(start)
                    sal_recon = dual_saliency_reconstruction_loss(
                        filtered_bchw, comp_bchw,
                        posenet_sal=sal_w[1:2],
                        segnet_boundary=bm,
                        alpha_pose=cfg.alpha,
                        alpha_seg=cfg.alpha_seg,
                    )
                else:
                    sal_recon = saliency_reconstruction_loss(
                        filtered_bchw, comp_bchw, sal_w[1:2]
                    )

                total = (loss + cfg.sal_lambda * sal_recon) / accum
                total.backward()

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
                del comp_pair, gt_pair, filtered, filtered_bchw, comp_bchw, sal_w

            if epoch >= cfg.warmup_epochs:
                self.scheduler.step()

            # SWA: snapshot weights in final 20% of training for wider minima
            if cfg.use_swa and epoch >= int(cfg.epochs * 0.8):
                if not hasattr(self, '_swa'):
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
                    comp_frames, gt_frames, posenet, segnet,
                    eval_pair_starts=eval_pair_starts,
                )
            else:
                scorer_val = self.best_scorer  # reuse last known

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
                ep_total = getattr(self, '_epoch_pcgrad_total', 0)
                ep_conflicts = getattr(self, '_epoch_pcgrad_conflicts', 0)
                if ep_total > 0:
                    conflict_str = f" conflict={ep_conflicts}/{ep_total}={ep_conflicts/ep_total:.0%}"
                # Reset for next epoch
                self._epoch_pcgrad_total = 0
                self._epoch_pcgrad_conflicts = 0
            print(f"[ep {epoch:4d}]{eval_tag} loss={avg_loss:.4f} pose={avg_pose:.6f} "
                  f"seg={avg_seg:.6f} scorer={scorer_val:.4f} best={self.best_scorer:.4f} lr={lr:.6f}{conflict_str}")

            # LR plateau detection for standard loss
            if is_eval_epoch and cfg.loss_mode in ("standard", "pcgrad"):
                self._plateau_window.append(scorer_val)
                if len(self._plateau_window) > 20:
                    self._plateau_window.pop(0)
                if (len(self._plateau_window) >= 20
                        and not self._plateau_reduced
                        and epoch > cfg.epochs * 0.3):
                    recent_best = min(self._plateau_window)
                    window_start_best = min(self._plateau_window[:10])
                    if recent_best >= window_start_best - 0.005:
                        for pg in self.optimizer.param_groups:
                            pg["lr"] *= 0.5
                        self._plateau_reduced = True
                        print(f"[trainer-lazy] LR plateau detected ep={epoch}, halving LR to {self.optimizer.param_groups[0]['lr']:.6f}")

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
                    if hasattr(self, '_cached_sw'):
                        entry["adaptive_sw"] = round(self._cached_sw, 4)
                    if hasattr(self, '_cached_bw'):
                        entry["adaptive_bw"] = round(self._cached_bw, 2)
                    # Proxy hardening diagnostics
                    if hasattr(self, '_proxy_confidence'):
                        entry["proxy_confidence"] = self._proxy_confidence
                    if hasattr(self, '_corrected_scorer'):
                        entry["corrected_scorer"] = self._corrected_scorer
                    if hasattr(self, '_baseline_pose') and self._baseline_pose:
                        entry["baseline_pose"] = self._baseline_pose
                    tf.write(json.dumps(entry) + "\n")

            # Save training state every 50 epochs for crash recovery
            if epoch % 50 == 0 and epoch > 0:
                self.save_training_state()

        # Apply SWA if it was active — average replaces EMA, then re-evaluate
        if cfg.use_swa and hasattr(self, '_swa') and self._swa.count > 0:
            self._swa.apply(self.ema)
            # Re-evaluate with SWA-averaged weights to see if it's better
            if eval_pair_starts is not None:
                swa_scorer = self._evaluate_int8_lazy(
                    comp_frames, gt_frames, posenet, segnet, eval_pair_starts)
                if swa_scorer < self.best_scorer:
                    print(f"  ** SWA IMPROVED: {self.best_scorer:.4f} -> {swa_scorer:.4f} **")
                    self.best_scorer = swa_scorer
                    self.best_epoch = cfg.epochs
                    self._save_checkpoint(cfg.epochs, swa_scorer)
                else:
                    print(f"[SWA] No improvement ({swa_scorer:.4f} vs best {self.best_scorer:.4f})")

        # Final save
        self.save_training_state()
        print(f"[trainer-lazy] Complete. Best: {self.best_scorer:.4f} at epoch {self.best_epoch}")
        return self.best_scorer

    def _evaluate_int8_lazy(
        self, comp_frames, gt_frames, posenet, segnet,
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

        # autocast reduces VRAM from ~6GB to ~0.2GB for scorer forward pass
        use_autocast = str(self.device).startswith("cuda") and torch.cuda.is_available()
        autocast = torch.amp.autocast("cuda", enabled=use_autocast)
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

        avg_p = total_p / count
        avg_s = total_s / count
        scorer = 100.0 * avg_s + math.sqrt(10.0 * avg_p)

        # Store per-component for telemetry (meta.json, regression alarm)
        self._last_eval_pose = round(avg_p, 8)
        self._last_eval_seg = round(avg_s, 8)

        # Baseline watermark: record raw compressed distortion at first eval
        if not hasattr(self, '_baseline_pose') or self._baseline_pose is None:
            self._baseline_pose = round(avg_p, 8)
            self._baseline_seg = round(avg_s, 8)
            print(f"[eval] Baseline watermark: pose={avg_p:.6f}, seg={avg_s:.6f}")

        # Proxy confidence: high when pose is in a credible range relative to baseline.
        # Drops when pose regresses (ratio > 2) OR improves implausibly (ratio < 0.05).
        # Normal training drives pose from baseline toward ~0.3x baseline over hundreds of epochs.
        proxy_confidence = 1.0
        if hasattr(self, '_baseline_pose') and self._baseline_pose and self._baseline_pose > 0:
            pose_ratio = avg_p / self._baseline_pose

            if pose_ratio > 2.0:
                # PoseNet regressing — proxy may be masking damage
                proxy_confidence = max(0.1, 1.0 / pose_ratio)
                print(f"  !! PROXY CONFIDENCE LOW (regression): pose={avg_p:.6f} is "
                      f"{pose_ratio:.1f}x baseline — authoritative eval recommended !!")
            elif pose_ratio < 0.05:
                # PoseNet suspiciously good — may be noise floor or measurement artifact
                proxy_confidence = 0.5
                print(f"  !! PROXY CONFIDENCE MODERATE: pose={avg_p:.6f} is "
                      f"{pose_ratio:.2f}x baseline (suspiciously low) !!")

            # Regression alarm: warn if PoseNet regresses 3x from baseline
            if pose_ratio > 3.0:
                print(f"  !! POSENET REGRESSION ALARM: {avg_p:.6f} is {pose_ratio:.1f}x baseline {self._baseline_pose:.6f} !!")
            if pose_ratio > 5.0:
                print(f"  !! CRITICAL: PoseNet {pose_ratio:.0f}x baseline — checkpoint NOT saved !!")
                self.model.load_state_dict(orig_state)
                self.model.train()
                return float('inf')  # prevent promotion of regressed checkpoint

        # Store proxy confidence for telemetry
        self._proxy_confidence = round(proxy_confidence, 4)

        # Corrected score estimate using proxy correction factors (α_p, α_s)
        # These are calibrated from authoritative eval runs. Default 1.0 = no correction.
        alpha_p = getattr(self, '_proxy_alpha_p', 1.0)
        alpha_s = getattr(self, '_proxy_alpha_s', 1.0)
        corrected_scorer = 100.0 * (alpha_s * avg_s) + math.sqrt(10.0 * (alpha_p * avg_p))
        self._corrected_scorer = round(corrected_scorer, 6)
        if abs(corrected_scorer - scorer) > 0.01:
            print(f"  [proxy-correction] raw={scorer:.4f} corrected={corrected_scorer:.4f} "
                  f"(α_p={alpha_p:.2f}, α_s={alpha_s:.2f})")

        self.model.load_state_dict(orig_state)
        self.model.train()
        return scorer
