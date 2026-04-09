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

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from .data import pair_from_frames, pair_start_indices, saliency_for_pair, load_raw_saliency, SEQ_LEN
from .losses import scorer_loss, segnet_ste_loss, saliency_reconstruction_loss
from .quantization import save_int8, load_int8, quantize_state_dict


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    hidden: int = 64
    kernel: int = 3
    variant: str = "standard"
    epochs: int = 1000
    alpha: float = 20.0
    sal_lambda: float = 1.0
    lr: float = 5e-4
    warmup_epochs: int = 10
    ema_decay: float = 0.997
    grad_clip: float = 1.0
    pairs_per_epoch: int | None = None  # None = all pairs
    scheduler: str = "cosine"  # cosine | cosine_restart
    restart_t0: int = 200
    restart_tmult: int = 2

    # SegNet boundary attack
    boundary_weight: float = 1.0  # >1.0 enables boundary weighting
    use_ste_segnet: bool = False

    # Output
    output_dir: str = "experiments/postfilter_weights"
    tag: str = "untitled"


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

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=1e-4
        )

        if config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=1e-6
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=config.restart_t0, T_mult=config.restart_tmult, eta_min=1e-6
            )

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
        import einops
        import types

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

    def _apply_filter_to_pair(self, comp_pair: torch.Tensor) -> torch.Tensor:
        """Apply the post-filter to both frames of a pair.

        Input: (1, 2, H, W, 3) uint8
        Output: (1, 2, H, W, 3) float [0, 255]
        """
        B, T, H, W, C = comp_pair.shape
        frames_bchw = comp_pair.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
        filtered_bchw = self.model(frames_bchw)
        return filtered_bchw.permute(0, 2, 3, 1).reshape(B, T, H, W, C)

    def _evaluate_int8(
        self, comp_pairs, gt_pairs, posenet, segnet, subsample: int = 4,
    ) -> float:
        """Evaluate EMA model after int8 quantization.

        This is the best-checkpoint selection mechanism.
        """
        orig_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Load EMA weights
        self.ema.apply(self.model)

        # Simulate int8 quantization
        q_state = {}
        for name, param in self.model.state_dict().items():
            p = param.detach().float()
            scale = p.abs().max() / 127.0
            if scale.item() < 1e-10:
                scale = torch.tensor(1.0, device=p.device)
            q = (p / scale).round().clamp(-128, 127) * scale
            q_state[name] = q
        self.model.load_state_dict(q_state)
        self.model.eval()

        total_p, total_s, count = 0.0, 0.0, 0
        with torch.no_grad():
            for idx in range(0, len(comp_pairs), subsample):
                filtered = self._apply_filter_to_pair(comp_pairs[idx])
                loss, pd, sd = scorer_loss(filtered, gt_pairs[idx], posenet, segnet)
                total_p += pd
                total_s += sd
                count += 1

        scorer = 100.0 * (total_s / count) + math.sqrt(10.0 * (total_p / count))

        # Restore original training weights
        self.model.load_state_dict(orig_state)
        self.model.train()
        return scorer

    def _save_checkpoint(self, epoch: int, scorer: float):
        """Save best int8 checkpoint."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = self.config.tag

        fp32_path = out_dir / f"postfilter_{tag}_best_fp32.pt"
        int8_path = out_dir / f"postfilter_{tag}_best_int8.pt"
        meta_path = out_dir / f"postfilter_{tag}_best_meta.json"

        # Save EMA fp32
        torch.save(self.ema.state_dict(), fp32_path)

        # Save int8
        int8_state = {}
        for name, param in self.ema.shadow.items():
            p = param.detach().cpu().float()
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
        torch.save(int8_state, int8_path)

        meta_path.write_text(json.dumps({
            "epoch": epoch,
            "scorer": scorer,
            "fp32_path": str(fp32_path),
            "int8_path": str(int8_path),
            "int8_size": int8_path.stat().st_size,
            "meta": int8_state["__meta__"],
        }, indent=2))

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
        # Patch scorer models for differentiable training (CRITICAL: without this,
        # PoseNet gradients are zero due to upstream @torch.no_grad on rgb_to_yuv6)
        if not self._patched_scorers:
            self._patch_scorers_for_training(posenet, segnet)
            self._patched_scorers = True

        n_pairs = len(comp_pairs)
        cfg = self.config
        pairs_per_epoch = cfg.pairs_per_epoch or n_pairs
        use_boundary = cfg.use_ste_segnet and boundary_masks is not None

        print(f"[trainer] {cfg.epochs} epochs, {pairs_per_epoch} pairs/ep, "
              f"h={cfg.hidden}, alpha={cfg.alpha}, device={self.device}")
        if use_boundary:
            print(f"[trainer] SegNet STE + boundary weighting ({cfg.boundary_weight}x)")

        for epoch in range(cfg.epochs):
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

                # Scorer loss
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

                self.optimizer.zero_grad()
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
        pair_starts = pair_start_indices(len(comp_frames))
        n_pairs = len(pair_starts)
        train_size = max(1, n_pairs // subsample)

        print(f"[trainer-lazy] {cfg.epochs} epochs, {train_size}/{n_pairs} pairs/ep, "
              f"h={cfg.hidden}, alpha={cfg.alpha}, device={self.device}")
        print(f"[trainer-lazy] Frames on CPU, pairs built on-the-fly (MPS-safe)")

        for epoch in range(cfg.epochs):
            self.model.train()
            total_loss, total_pose, total_seg = 0.0, 0.0, 0.0

            if epoch < cfg.warmup_epochs:
                lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

            # Random subset of pair indices
            perm = torch.randperm(n_pairs)[:train_size]

            for step, pair_idx in enumerate(perm):
                start = pair_starts[pair_idx.item()]

                # Build pair on-the-fly (CPU → device)
                comp_pair = pair_from_frames(comp_frames, start).to(self.device)
                gt_pair = pair_from_frames(gt_frames, start).to(self.device)

                # Apply filter
                filtered = self._apply_filter_to_pair(comp_pair)

                # Scorer loss
                loss, pd, sd = scorer_loss(filtered, gt_pair, posenet, segnet)

                # Saliency reconstruction (frame 1 only)
                sal_w = saliency_for_pair(raw_saliency, start, cfg.alpha, self.device)
                filtered_bchw = filtered[:, 1].permute(0, 3, 1, 2)
                comp_bchw = comp_pair[:, 1].float().permute(0, 3, 1, 2)
                sal_recon = saliency_reconstruction_loss(
                    filtered_bchw, comp_bchw, sal_w[1:2]  # frame 1 saliency only
                )

                total = loss + cfg.sal_lambda * sal_recon

                self.optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                self.ema.update(self.model)

                total_loss += loss.item()
                total_pose += pd
                total_seg += sd

                # Free pair memory immediately
                del comp_pair, gt_pair, filtered, filtered_bchw, comp_bchw, sal_w

            if epoch >= cfg.warmup_epochs:
                self.scheduler.step()

            n = len(perm)
            avg_loss = total_loss / n
            avg_pose = total_pose / n
            avg_seg = total_seg / n

            # Best-checkpoint int8 evaluation (subsample for speed)
            scorer_val = self._evaluate_int8_lazy(
                comp_frames, gt_frames, posenet, segnet
            )

            if scorer_val < self.best_scorer:
                self.best_scorer = scorer_val
                self.best_epoch = epoch
                self._save_checkpoint(epoch, scorer_val)
                print(f"  ** NEW BEST: ep {epoch}, scorer {scorer_val:.4f} **")

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"[ep {epoch:4d}] loss={avg_loss:.4f} pose={avg_pose:.6f} "
                  f"seg={avg_seg:.6f} scorer={scorer_val:.4f} best={self.best_scorer:.4f} lr={lr:.6f}")

        print(f"[trainer-lazy] Complete. Best: {self.best_scorer:.4f} at epoch {self.best_epoch}")
        return self.best_scorer

    def _evaluate_int8_lazy(
        self, comp_frames, gt_frames, posenet, segnet, subsample: int = 4,
    ) -> float:
        """Evaluate EMA model after int8 quantization, using lazy pair loading."""
        orig_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.ema.apply(self.model)
        q_state = quantize_state_dict(self.model.state_dict())
        self.model.load_state_dict(q_state)
        self.model.eval()

        pair_starts = pair_start_indices(len(comp_frames))
        total_p, total_s, count = 0.0, 0.0, 0

        with torch.no_grad():
            for i, start in enumerate(pair_starts):
                if i % subsample != 0:
                    continue
                comp_pair = pair_from_frames(comp_frames, start).to(self.device)
                gt_pair = pair_from_frames(gt_frames, start).to(self.device)
                filtered = self._apply_filter_to_pair(comp_pair)
                loss, pd, sd = scorer_loss(filtered, gt_pair, posenet, segnet)
                total_p += pd
                total_s += sd
                count += 1
                del comp_pair, gt_pair, filtered

        import math
        scorer = 100.0 * (total_s / count) + math.sqrt(10.0 * (total_p / count))

        self.model.load_state_dict(orig_state)
        self.model.train()
        return scorer
