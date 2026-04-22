#!/usr/bin/env python3
"""Quantization-Aware Training (QAT) fine-tuning for FP4 deployment.

Implements progressive quantization following Bit-by-Bit (ICLR 2026)
and Compute-Optimal QAT (Apple, Sep 2025) findings:

  Phase A: INT8 warm-up  — 50 epochs, lets weights migrate toward
           quantization-friendly regions at coarser granularity first.
  Phase B: FP4 fine-tune — 250 epochs with our FP4 codebook, cosine
           decay from 0.1x base LR to zero.

Research-backed design choices:
  - Progressive bit-reduction beats direct 4-bit QAT (Bit-by-Bit, ICLR 2026)
  - QAT fraction 30% of total training for <200K-param models (Apple, Sep 2025)
  - Per-block FP4 with block_size=32 (our codebook matches BOF4 structure)
  - STE gradient estimator (PEGE offers 1-2% but adds complexity)
  - Mixed precision: Embedding + FiLM Linear stay in FP16 (negligible rate,
    disproportionate quality impact). Only Conv2d/ConvTranspose2d in FP4.
  - eval_roundtrip=True + noise_std=0.5 in loss (mandatory per our findings)
  - Scorer loss matches training: hinge SegNet + PoseNet MSE

Usage (after overnight float training completes):
    PYTHONPATH=src:upstream:$PWD python experiments/qat_finetune.py \\
        --checkpoint experiments/results/overnight_small_renderer/distill_best.pt \\
        --upstream upstream/ \\
        --device cuda \\
        --output-dir experiments/results/qat_fp4

References:
    [1] Compute-Optimal QAT, Apple, Sep 2025, arXiv:2509.22935
    [2] Bit-by-Bit Progressive QAT, ICLR 2026, arXiv:2604.07888
    [3] BOF4: Block-Wise Optimal Float, May 2025, arXiv:2505.06653
    [4] NeuroQuant: Variable-Rate Neural Video, ICLR 2025, arXiv:2502.11729
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Path setup
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

_UPSTREAM_CANDIDATES = [
    Path(os.environ.get("TAC_UPSTREAM_DIR", "")),
    Path(os.environ.get("UPSTREAM_ROOT", "")),
    _ROOT / "upstream",
]
UPSTREAM_ROOT: Path | None = None
for _p in _UPSTREAM_CANDIDATES:
    if _p and (_p / "modules.py").exists():
        UPSTREAM_ROOT = _p
        sys.path.insert(0, str(_p))
        break


@dataclass
class QATConfig:
    """All hyperparameters for QAT fine-tuning."""

    # Architecture (must match the float checkpoint)
    embed_dim: int = 6
    base_ch: int = 24
    mid_ch: int = 32
    motion_hidden: int = 32
    depth: int = 1
    pose_dim: int = 6
    max_flow_px: float = 20.0
    max_residual: float = 20.0
    use_dsconv: bool = True

    # Paths
    checkpoint_path: str = ""
    upstream_dir: str = "upstream/"
    output_dir: str = "experiments/results/qat_fp4"

    # QAT schedule — research-backed (Apple 2025, Bit-by-Bit 2026)
    int8_warmup_epochs: int = 50       # Phase A: INT8 warm-up
    fp4_epochs: int = 250              # Phase B: FP4 fine-tune
    base_lr: float = 3e-5             # 0.1x of phase2 LR in float training
    lr_min_ratio: float = 0.0         # cosine decay to zero (Apple 2025)
    warmup_steps: int = 20            # re-warmup at QAT start (Apple 2025)
    batch_size: int = 4               # scorer-limited
    grad_clip: float = 1.0

    # FP4 quantization
    fp4_block_size: int = 32          # weights per scale group
    mixed_precision: bool = True      # keep Embedding + FiLM Linear in FP16

    # Scorer loss (must match float training)
    seg_weight: float = 100.0
    pose_weight: float = 10.0
    segnet_loss_mode: str = "hinge"
    hinge_margin: float = 0.5
    eval_roundtrip: bool = True
    noise_std: float = 0.5

    # Monitoring
    eval_every: int = 25
    log_every: int = 5
    checkpoint_every: int = 50
    device: str = "cuda"
    seed: int = 42


def create_model(cfg: QATConfig, device: torch.device) -> nn.Module:
    """Create AsymmetricPairGenerator matching the float checkpoint."""
    from tac.renderer import AsymmetricPairGenerator

    model = AsymmetricPairGenerator(
        num_classes=5,
        embed_dim=cfg.embed_dim,
        base_ch=cfg.base_ch,
        mid_ch=cfg.mid_ch,
        motion_hidden=cfg.motion_hidden,
        depth=cfg.depth,
        max_flow_px=cfg.max_flow_px,
        max_residual=cfg.max_residual,
        pose_dim=cfg.pose_dim,
        use_dsconv=cfg.use_dsconv,
    )
    return model.to(device)


def load_float_checkpoint(model: nn.Module, path: str, device: torch.device) -> None:
    """Load the float checkpoint from distillation training."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    print(f"  Loaded float checkpoint from {path}")


def apply_int8_fake_quant(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """Wrap Conv2d weights with INT8 FakeQuant STE (Phase A warm-up).

    Returns list of (module, param_name) for later removal.
    """
    from tac.quantization import FakeQuantSTE

    class Int8Parametrize(nn.Module):
        def forward(self, weight: torch.Tensor) -> torch.Tensor:
            return FakeQuantSTE.apply(weight)

    wrapped = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.utils.parametrize.register_parametrization(
                module, "weight", Int8Parametrize()
            )
            wrapped.append((module, "weight"))
    return wrapped


def remove_parametrizations(wrapped: list[tuple[nn.Module, str]]) -> None:
    """Remove all registered parametrizations."""
    for module, param_name in wrapped:
        if nn.utils.parametrize.is_parametrized(module, param_name):
            nn.utils.parametrize.remove_parametrizations(module, param_name)


def apply_fp4_fake_quant(
    model: nn.Module,
    block_size: int = 32,
    mixed_precision: bool = True,
) -> list[tuple[nn.Module, str]]:
    """Wrap quantizable layers with FP4 FakeQuant STE (Phase B).

    Mixed precision (research-backed):
      - Conv2d, ConvTranspose2d: FP4 (bulk of parameters, most benefit)
      - Embedding: FP4 (only 30 params, but large impact on class boundaries)
      - Linear (FiLM): SKIP if mixed_precision=True (small layers, high sensitivity)

    Args:
        model: the renderer to wrap
        block_size: FP4 block size (32 = default, good for small models)
        mixed_precision: if True, skip FiLM Linear layers

    Returns:
        list of (module, param_name) for cleanup
    """
    from tac.fp4_quantize import FP4Parametrize, DEFAULT_CODEBOOK

    wrapped = []
    for name, module in model.named_modules():
        # Skip FiLM layers in mixed-precision mode (film_bottleneck, film_decoder)
        if mixed_precision and isinstance(module, nn.Linear):
            if "film" in name.lower():
                continue

        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Embedding)):
            if hasattr(module, "weight") and module.weight.ndim >= 2:
                nn.utils.parametrize.register_parametrization(
                    module,
                    "weight",
                    FP4Parametrize(DEFAULT_CODEBOOK.clone(), block_size),
                )
                wrapped.append((module, "weight"))

    n_fp4 = len(wrapped)
    n_skip = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear) and "film" in _.lower())
    print(f"  FP4 parametrized: {n_fp4} layers, skipped: {n_skip} FiLM Linear layers")
    return wrapped


def compute_scorer_loss(
    model: nn.Module,
    masks_even: torch.Tensor,
    masks_odd: torch.Tensor,
    gt_frames_even: torch.Tensor,
    gt_frames_odd: torch.Tensor,
    poses: torch.Tensor | None,
    posenet: nn.Module,
    segnet: nn.Module,
    cfg: QATConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the contest-faithful scorer loss.

    Matches train_distill.py Phase 2 loss exactly:
      loss = seg_weight * hinge_loss + pose_weight * pose_mse

    With eval_roundtrip + noise_std for contest fidelity.
    """
    from tac.renderer import simulate_eval_roundtrip
    from tac.camera import CAMERA_H, CAMERA_W

    B = masks_even.shape[0]

    # Forward pass through renderer
    pair_kwargs = {}
    if poses is not None and hasattr(model, "pose_dim") and model.pose_dim > 0:
        pair_kwargs["pose"] = poses
    # Handle QAT wrapper
    renderer = model.base if hasattr(model, "base") else model
    pairs = renderer(masks_even, masks_odd, **pair_kwargs)  # (B, 2, H, W, 3)

    pred_even = pairs[:, 0]  # (B, H, W, 3)
    pred_odd = pairs[:, 1]

    # Flatten to (2B, H, W, 3) for scorer
    pred_all = torch.cat([pred_even, pred_odd], dim=0)
    gt_all = torch.cat([gt_frames_even, gt_frames_odd], dim=0)

    # eval_roundtrip: simulate contest resize chain with noise
    if cfg.eval_roundtrip:
        pred_chw = pred_all.permute(0, 3, 1, 2)
        pred_chw = simulate_eval_roundtrip(
            pred_chw, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=cfg.noise_std,
        )
        pred_for_loss = pred_chw.permute(0, 2, 3, 1)

        gt_chw = gt_all.permute(0, 3, 1, 2)
        gt_chw = simulate_eval_roundtrip(
            gt_chw, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=cfg.noise_std,
        )
        gt_for_loss = gt_chw.permute(0, 2, 3, 1)
    else:
        pred_for_loss = pred_all
        gt_for_loss = gt_all

    # SegNet loss (hinge)
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W
    pred_seg_chw = pred_for_loss.permute(0, 3, 1, 2)
    gt_seg_chw = gt_for_loss.permute(0, 3, 1, 2)

    pred_seg_in = segnet.preprocess_input(pred_seg_chw.unsqueeze(1))
    gt_seg_in = segnet.preprocess_input(gt_seg_chw.unsqueeze(1))
    pred_seg_logits = segnet(pred_seg_in)
    gt_seg_argmax = segnet(gt_seg_in).argmax(dim=1)

    if cfg.segnet_loss_mode == "hinge":
        # Hinge loss: penalize pixels at risk of argmax flip
        correct = pred_seg_logits.gather(1, gt_seg_argmax.unsqueeze(1)).squeeze(1)
        mask_inf = torch.zeros_like(pred_seg_logits)
        mask_inf.scatter_(1, gt_seg_argmax.unsqueeze(1), float("-inf"))
        runner_up = (pred_seg_logits + mask_inf).max(dim=1).values
        seg_loss = F.relu(cfg.hinge_margin - (correct - runner_up)).mean()
    else:
        seg_loss = F.cross_entropy(pred_seg_logits, gt_seg_argmax)

    # PoseNet loss (MSE on 6D pose)
    pred_pose_chw = pred_for_loss.permute(0, 3, 1, 2)
    gt_pose_chw = gt_for_loss.permute(0, 3, 1, 2)

    # Build pairs for PoseNet: (B, 2, C, H, W)
    pred_pose_pairs = torch.stack([
        pred_pose_chw[:B], pred_pose_chw[B:]
    ], dim=1)
    gt_pose_pairs = torch.stack([
        gt_pose_chw[:B], gt_pose_chw[B:]
    ], dim=1)

    pred_pose_in = posenet.preprocess_input(pred_pose_pairs)
    gt_pose_in = posenet.preprocess_input(gt_pose_pairs)
    pred_pose_out = posenet(pred_pose_in)["pose"][..., :6]
    gt_pose_out = posenet(gt_pose_in)["pose"][..., :6]
    pose_loss = (pred_pose_out - gt_pose_out).pow(2).mean()

    # Combined loss
    total = cfg.seg_weight * seg_loss + cfg.pose_weight * pose_loss

    metrics = {
        "seg_loss": seg_loss.item(),
        "pose_loss": pose_loss.item(),
        "total_loss": total.item(),
    }
    return total, metrics


def cosine_lr(step: int, total_steps: int, base_lr: float, min_ratio: float, warmup: int) -> float:
    """Cosine LR with linear warmup (Apple 2025 QAT schedule)."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def evaluate_fp4_quality(
    model: nn.Module,
    masks: torch.Tensor,
    gt_frames: list,
    poses: torch.Tensor | None,
    device: torch.device,
    n_pairs: int = 20,
) -> dict[str, float]:
    """Quick quality check: generate frames and compute distortion via upstream scorer."""
    sys.path.insert(0, str(UPSTREAM_ROOT))
    from modules import DistortionNet

    dn = DistortionNet().eval().to(device)
    dn.load_state_dicts(
        str(UPSTREAM_ROOT / "models" / "posenet.safetensors"),
        str(UPSTREAM_ROOT / "models" / "segnet.safetensors"),
        str(device),
    )

    renderer = model.base if hasattr(model, "base") else model
    renderer.eval()

    pd_list, sd_list = [], []
    with torch.inference_mode():
        for i in range(min(n_pairs, len(gt_frames) // 2)):
            m_t = masks[2 * i : 2 * i + 1].to(device=device, dtype=torch.int64)
            m_t1 = masks[2 * i + 1 : 2 * i + 2].to(device=device, dtype=torch.int64)
            p = poses[i : i + 1].to(device) if poses is not None else None
            kwargs = {"pose": p} if p is not None else {}
            pair = renderer(m_t, m_t1, **kwargs)
            chw = pair[0].permute(0, 3, 1, 2).float()
            cam = F.interpolate(chw, size=(874, 1164), mode="bilinear", align_corners=False)
            cam = cam.round().clamp(0, 255)

            gt_p = torch.stack([
                torch.from_numpy(gt_frames[2 * i]).float(),
                torch.from_numpy(gt_frames[2 * i + 1]).float(),
            ]).unsqueeze(0).to(device)
            comp_p = cam.permute(0, 2, 3, 1).unsqueeze(0)
            pd, sd = dn.compute_distortion(comp_p, gt_p)
            pd_list.append(pd.item())
            sd_list.append(sd.item())

    renderer.train()
    del dn
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    avg_pose = sum(pd_list) / max(len(pd_list), 1)
    avg_seg = sum(sd_list) / max(len(sd_list), 1)
    distortion = 100 * avg_seg + math.sqrt(10 * avg_pose)

    return {"pose_d": avg_pose, "seg_d": avg_seg, "distortion": distortion}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QAT fine-tuning: float checkpoint → FP4-robust model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to float checkpoint (.pt)")
    parser.add_argument("--upstream", default="upstream/")
    parser.add_argument("--output-dir", default="experiments/results/qat_fp4")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])

    # Architecture (must match float training)
    parser.add_argument("--base-ch", type=int, default=24)
    parser.add_argument("--mid-ch", type=int, default=32)
    parser.add_argument("--pose-dim", type=int, default=6)
    parser.add_argument("--use-dsconv", action="store_true", default=True)

    # QAT schedule
    parser.add_argument("--int8-warmup-epochs", type=int, default=50)
    parser.add_argument("--fp4-epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=4)

    # Control
    parser.add_argument("--no-mixed-precision", action="store_true",
                        help="Quantize ALL layers including FiLM (not recommended)")
    parser.add_argument("--skip-int8-warmup", action="store_true",
                        help="Skip INT8 warm-up phase (direct FP4)")

    args = parser.parse_args()

    cfg = QATConfig(
        checkpoint_path=args.checkpoint,
        upstream_dir=args.upstream,
        output_dir=args.output_dir,
        device=args.device,
        base_ch=args.base_ch,
        mid_ch=args.mid_ch,
        pose_dim=args.pose_dim,
        use_dsconv=args.use_dsconv,
        int8_warmup_epochs=0 if args.skip_int8_warmup else args.int8_warmup_epochs,
        fp4_epochs=args.fp4_epochs,
        base_lr=args.lr,
        batch_size=args.batch_size,
        mixed_precision=not args.no_mixed_precision,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "qat_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("=" * 70)
    print("QAT FINE-TUNING: Float → FP4-Robust")
    print("=" * 70)
    print(f"  Float checkpoint: {cfg.checkpoint_path}")
    print(f"  Schedule: {cfg.int8_warmup_epochs} INT8 warm-up + {cfg.fp4_epochs} FP4")
    print(f"  LR: {cfg.base_lr} (cosine → 0)")
    print(f"  Mixed precision: {cfg.mixed_precision} (FiLM layers in FP16)")
    print(f"  Device: {cfg.device}")
    print()

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading data...")

    # GT video
    from tac.eval.auth_eval import AuthEvaluator
    evaluator = AuthEvaluator(
        upstream_dir=Path(cfg.upstream_dir) if UPSTREAM_ROOT is None else UPSTREAM_ROOT,
        device=cfg.device,
    )
    evaluator.load_scorers()
    gt_frames = evaluator.decode_gt_video("0.mkv")
    masks = evaluator.extract_masks(gt_frames, batch_size=4)
    print(f"  GT: {len(gt_frames)} frames, masks: {masks.shape}")

    # Scorers (differentiable)
    from tac.scorer import load_differentiable_scorers
    upstream_dir = str(UPSTREAM_ROOT) if UPSTREAM_ROOT else cfg.upstream_dir
    posenet, segnet = load_differentiable_scorers(upstream_dir, device=device)

    # GT poses
    poses = None
    for poses_path in [
        Path("experiments/results/gt_poses.pt"),
        Path(cfg.upstream_dir) / "gt_poses.pt",
    ]:
        if poses_path.exists():
            poses = torch.load(str(poses_path), map_location="cpu", weights_only=True)
            if isinstance(poses, dict):
                poses = poses.get("poses", poses.get("gt_poses"))
            poses = poses.float()
            print(f"  Poses: {poses.shape}")
            break

    # ── Create model and load float weights ───────────────────────────
    print("Creating model...")
    model = create_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters")

    load_float_checkpoint(model, cfg.checkpoint_path, device)

    # ── Baseline quality (float, no quantization) ─────────────────────
    print("\nBaseline quality (float32):")
    baseline = evaluate_fp4_quality(model, masks, gt_frames, poses, device)
    print(f"  pose_d={baseline['pose_d']:.5f} seg_d={baseline['seg_d']:.5f} "
          f"distortion={baseline['distortion']:.3f}")

    # ── Prepare training data (pre-extract GT scorer features) ────────
    n_pairs = masks.shape[0] // 2
    gt_frames_tensor = torch.stack([
        torch.from_numpy(f).float() for f in gt_frames
    ])  # (N, H, W, 3)

    best_distortion = float("inf")
    best_state = None
    history = []

    # ══════════════════════════════════════════════════════════════════
    # PHASE A: INT8 Warm-Up (progressive quantization)
    # Research: Bit-by-Bit (ICLR 2026) — coarse quant first, then fine
    # ══════════════════════════════════════════════════════════════════
    if cfg.int8_warmup_epochs > 0:
        print(f"\n{'='*70}")
        print(f"PHASE A: INT8 Warm-Up ({cfg.int8_warmup_epochs} epochs)")
        print(f"{'='*70}")

        int8_wrapped = apply_int8_fake_quant(model)
        print(f"  INT8 parametrized: {len(int8_wrapped)} layers")

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr, weight_decay=1e-4)
        total_steps = cfg.int8_warmup_epochs

        model.train()
        for epoch in range(cfg.int8_warmup_epochs):
            lr = cosine_lr(epoch, total_steps, cfg.base_lr, 0.1, cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Sample random batch of pairs
            idx = torch.randperm(n_pairs)[:cfg.batch_size]
            m_even = masks[idx * 2].to(device, dtype=torch.int64)
            m_odd = masks[idx * 2 + 1].to(device, dtype=torch.int64)
            gt_even = gt_frames_tensor[idx * 2].to(device)
            gt_odd = gt_frames_tensor[idx * 2 + 1].to(device)
            p = poses[idx].to(device) if poses is not None else None

            optimizer.zero_grad()
            loss, metrics = compute_scorer_loss(
                model, m_even, m_odd, gt_even, gt_odd, p,
                posenet, segnet, cfg, device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if epoch % cfg.log_every == 0:
                print(f"  [INT8] ep {epoch:>4d}/{total_steps} | "
                      f"loss={metrics['total_loss']:.4f} "
                      f"(seg={metrics['seg_loss']:.4f} pose={metrics['pose_loss']:.4f}) "
                      f"lr={lr:.2e}")

            if epoch > 0 and epoch % cfg.eval_every == 0:
                q = evaluate_fp4_quality(model, masks, gt_frames, poses, device, n_pairs=10)
                print(f"  [INT8] eval: distortion={q['distortion']:.3f}")
                history.append({"phase": "int8", "epoch": epoch, **q})

        # Remove INT8 parametrizations before Phase B
        remove_parametrizations(int8_wrapped)
        print("  INT8 parametrizations removed")

        del optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # PHASE B: FP4 Fine-Tune (target quantization)
    # Research: Apple 2025 — cosine decay to zero, no cooldown phase
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"PHASE B: FP4 Fine-Tune ({cfg.fp4_epochs} epochs)")
    print(f"{'='*70}")

    fp4_wrapped = apply_fp4_fake_quant(
        model,
        block_size=cfg.fp4_block_size,
        mixed_precision=cfg.mixed_precision,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr, weight_decay=1e-4)
    total_steps = cfg.fp4_epochs

    model.train()
    for epoch in range(cfg.fp4_epochs):
        lr = cosine_lr(epoch, total_steps, cfg.base_lr, cfg.lr_min_ratio, cfg.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Sample random batch
        idx = torch.randperm(n_pairs)[:cfg.batch_size]
        m_even = masks[idx * 2].to(device, dtype=torch.int64)
        m_odd = masks[idx * 2 + 1].to(device, dtype=torch.int64)
        gt_even = gt_frames_tensor[idx * 2].to(device)
        gt_odd = gt_frames_tensor[idx * 2 + 1].to(device)
        p = poses[idx].to(device) if poses is not None else None

        optimizer.zero_grad()
        loss, metrics = compute_scorer_loss(
            model, m_even, m_odd, gt_even, gt_odd, p,
            posenet, segnet, cfg, device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if epoch % cfg.log_every == 0:
            print(f"  [FP4] ep {epoch:>4d}/{total_steps} | "
                  f"loss={metrics['total_loss']:.4f} "
                  f"(seg={metrics['seg_loss']:.4f} pose={metrics['pose_loss']:.4f}) "
                  f"lr={lr:.2e}")

        if epoch > 0 and epoch % cfg.eval_every == 0:
            q = evaluate_fp4_quality(model, masks, gt_frames, poses, device, n_pairs=15)
            print(f"  [FP4] eval: distortion={q['distortion']:.3f} "
                  f"(pose={q['pose_d']:.5f} seg={q['seg_d']:.5f})")
            history.append({"phase": "fp4", "epoch": epoch, **q})

            if q["distortion"] < best_distortion:
                best_distortion = q["distortion"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  [FP4] ★ NEW BEST: distortion={best_distortion:.3f}")

        if epoch > 0 and epoch % cfg.checkpoint_every == 0:
            ckpt_path = out_dir / f"qat_epoch_{epoch}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "phase": "fp4",
                "distortion": metrics["total_loss"],
            }, ckpt_path)

    # ── Remove FP4 parametrizations and restore best weights ──────────
    remove_parametrizations(fp4_wrapped)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model (distortion={best_distortion:.3f})")
    else:
        print("\nNo improvement found during QAT — using final weights")

    # ── Final quality check ───────────────────────────────────────────
    print("\nFinal quality (QAT-trained, float inference):")
    final_float = evaluate_fp4_quality(model, masks, gt_frames, poses, device)
    print(f"  pose_d={final_float['pose_d']:.5f} seg_d={final_float['seg_d']:.5f} "
          f"distortion={final_float['distortion']:.3f}")

    # ── Export FP4 binary ─────────────────────────────────────────────
    print("\nExporting FP4 binary...")
    from tac.renderer_export import export_asymmetric_checkpoint_fp4

    fp4_path = out_dir / "renderer_fp4.bin"
    export_asymmetric_checkpoint_fp4(model, fp4_path)
    fp4_size = fp4_path.stat().st_size
    print(f"  FP4 export: {fp4_size:,} bytes ({fp4_size/1024:.1f} KB)")

    # Save float checkpoint too (for pose TTO)
    float_path = out_dir / "qat_best_float.pt"
    torch.save({"model_state_dict": model.state_dict()}, float_path)

    # ── Quality after actual FP4 round-trip ───────────────────────────
    print("\nQuality after FP4 round-trip (actual deployment):")
    from tac.renderer_export import load_asymmetric_checkpoint_fp4
    model_fp4 = load_asymmetric_checkpoint_fp4(fp4_path.read_bytes(), device=str(device))
    model_fp4.eval()

    # Quick distortion check
    fp4_quality = evaluate_fp4_quality(model_fp4, masks, gt_frames, poses, device)
    print(f"  pose_d={fp4_quality['pose_d']:.5f} seg_d={fp4_quality['seg_d']:.5f} "
          f"distortion={fp4_quality['distortion']:.3f}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"QAT SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline (float):    distortion={baseline['distortion']:.3f}")
    print(f"  QAT-trained (float): distortion={final_float['distortion']:.3f}")
    print(f"  FP4 round-trip:      distortion={fp4_quality['distortion']:.3f}")
    print(f"  FP4 degradation:     {fp4_quality['distortion'] - baseline['distortion']:+.3f}")
    print(f"  FP4 binary size:     {fp4_size:,} bytes ({fp4_size/1024:.1f} KB)")
    rate_estimate = 25 * (fp4_size + 280000 + 8000) / 37545489
    print(f"  Estimated archive:   {fp4_size + 280000 + 8000:,} bytes")
    print(f"  Estimated rate:      {rate_estimate:.4f}")
    print(f"  Projected score:     {fp4_quality['distortion'] + rate_estimate:.3f}")
    print(f"  Output: {out_dir}")

    # Save results
    results = {
        "baseline": baseline,
        "qat_float": final_float,
        "fp4_roundtrip": fp4_quality,
        "fp4_degradation": fp4_quality["distortion"] - baseline["distortion"],
        "fp4_size_bytes": fp4_size,
        "n_params": n_params,
        "config": asdict(cfg),
        "history": history,
    }
    (out_dir / "qat_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_dir / 'qat_results.json'}")


if __name__ == "__main__":
    main()
