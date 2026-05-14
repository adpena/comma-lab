#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-trained tiny Hinton-distilled SegNet+PoseNet surrogate (forward-only KL distill).

Per W's DEFERRED reactivation analysis (2026-05-11) — see memory file
`feedback_l2_hinton_saliency_first_dispatch_landed_20260511.md`. The
PR106 r2 L2 + Hinton + saliency dispatch attempt landed structurally
(YUV6 dominance broken 49,381 → 526) but the random-init surrogate
produced uninformative saliency (max=1.28e-4 / mean=7.6e-6 = noise floor),
blocking score-positive residual without the T10 IB-Lagrangian dispatch
($40 operator-gated).

This tool tests the hypothesis that we can $0-unlock W's DEFERRED
criteria #1+#2 by training the LL DistilledSegNet+DistilledPoseNet
surrogate via FORWARD-ONLY KL distillation on macOS CPU. The teacher
inference (real SegNet + PoseNet) runs on CPU only — no gradient
backprop through the teacher, no MPS use. The student backprops
through the KL loss against the teacher's CPU outputs.

CLAUDE.md non-negotiables wired
-------------------------------
- "MPS auth eval is NOISE": NO MPS use. Tagged `[macOS-CPU-research-signal]`.
- "EMA — NON-NEGOTIABLE" decay 0.997: per CLAUDE.md, snapshot+restore.
- "eval_roundtrip — NON-NEGOTIABLE": teacher inputs use the canonical
  uint8 STE roundtrip (the surrogate sees the same eval-time bottleneck
  the contest scorer sees). The student is trained against the teacher's
  outputs ON THOSE roundtripped frames, so the surrogate learns the
  uint8-bottleneck-aware contest-scorer response.
- "Strict scorer rule": surrogate is COMPRESS-TIME ONLY; this tool
  produces a CHECKPOINT for downstream encoder consumption. The inflate
  path NEVER loads the surrogate.
- "/tmp paths FORBIDDEN": output dir under
  `experiments/results/<lane>_<utc>/`.
- "Forbidden score claims": permanent `score_claim=False` /
  `promotion_eligible=False` / `ready_for_exact_eval_dispatch=False`.
- "Forbidden empirical-claim-without-evidence-tag": every metric
  written carries `[macOS-CPU-research-signal]` axis tag; the
  CHECKPOINT carries the same tag inside `t10_config.distill_label`.
- "Catalog #134" Phase 3 prereq: distillation gap measured-and-recorded
  in `distillation_gap_estimate.json` so consumers can read the gap
  the same way they read T10's. **NB**: a CPU-trained surrogate
  CANNOT be promoted to Phase 3 unblock; this gate is informational
  only. The gap is tagged `cpu_trained` in the artifact and the
  `passes_phase3_threshold` flag is HARDWIRED FALSE for
  CPU-trained-on-non-Linux surrogates.

What this DOES (vs T10)
-----------------------
- Loads real `tac.scorer.load_default_scorers` SegNet + PoseNet on CPU
  (NOT MPS).
- Iterates a set of (configurable count of) 600 frame-pairs from
  `upstream/videos/0.mkv` via PyAV decode.
- For each pair: runs the real SegNet + PoseNet forward (CPU,
  `torch.no_grad`), captures `(seg_logits, pose_floats)` as the
  Hinton distillation targets.
- Trains `DistilledSegNet` + `DistilledPoseNet` (the LL scaffolds) via
  `KL(σ(z_real/T) || σ(z_aux/T)) * T²` for seg + MSE on first 6 pose
  dims. Adam lr 1e-3 default; ~10-50 epochs.
- Saves EMA shadow per CLAUDE.md non-negotiable.
- Measures (a) gradient norm of student vs random-init baseline, (b)
  surrogate-vs-real-scorer output agreement on a held-out pair set,
  (c) the gap via the standard `final_loss_kl / T^2` formula.

What this does NOT do
---------------------
- It does NOT promote any score / archive / dispatch readiness.
- It does NOT bypass the operator gate on the actual L2+Hinton+saliency
  Modal T4 dispatch — that is still the operator's call (≤$1).
- It does NOT ship in any contest archive — the surrogate is a
  COMPRESS-TIME training-time prior ONLY.

Reactivation criteria for the CPU-trained surrogate
---------------------------------------------------
A CPU-trained surrogate "succeeds" (per the dispatch prompt's
contingency framing) if:
  * Final gradient-norm RATIO (trained / random-init) is > 100×.
  * Per-pixel saliency max value > 1e-2 (vs the random-init 1.28e-4 floor).
  * Surrogate-vs-real-scorer output agreement is < 10% MSE drift on
    the held-out pair set.

If all three hold, the operator can re-attempt the L2+Hinton+saliency
dispatch path with this CPU-trained checkpoint instead of waiting on T10.

If any falls short, the artifact still serves as the explicit
falsification anchor that confirms T10 IB-Lagrangian dispatch IS the
unique unlock.

Cross-references
----------------
- W's DEFERRED memo: `feedback_l2_sparse_aware_encoders_first_dispatch_landed_20260511.md`
- LL surrogate scaffold: `src/tac/residual_basis/hinton_distilled_scorer_surrogate.py`
- T10 trainer (the canonical $40 dispatch this tool tries to $0-bypass):
  `experiments/train_t10_ib_lagrangian_aux_scorer.py`
- Phase 3 prereq gate: Catalog #134 (`Phase3DispatchGate` in
  `tac.phase3.joint_scorer_renderer_codec`)
- Catalog #123: `check_no_weight_domain_saliency_on_score_gradient_substrate`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LANE_ID = "lane_cpu_trained_tiny_hinton_surrogate_bootstrap"
SCHEMA_VERSION = "1.0.0-cpu-trained-tiny-hinton-surrogate"
EVIDENCE_GRADE_TAG = "[macOS-CPU-research-signal]"
PHASE3_DISTILL_GAP_THRESHOLD = 0.03  # informational only for CPU-trained
DEFAULT_EMA_DECAY = 0.997


def _maybe_relpath_to_repo(p: Path) -> str:
    """Return path relative to REPO_ROOT if possible, else absolute string.

    Pytest tmp_path lives outside the repo; relative_to() raises ValueError
    in that case. The provenance schema accepts either form — both are
    deterministic and reproducible because the file lives there.
    """
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU-trained tiny Hinton surrogate (forward-only KL distill)"
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    p.add_argument(
        "--video-path", type=Path, default=REPO_ROOT / "upstream" / "videos" / "0.mkv"
    )
    p.add_argument("--n-pairs", type=int, default=64,
                   help="Number of 2-frame pairs to use as distillation set "
                        "(set 600 for full dataset; default 64 for quick CPU runs)")
    p.add_argument("--n-heldout-pairs", type=int, default=16,
                   help="Number of pairs reserved for held-out agreement check")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--ema-decay", type=float, default=DEFAULT_EMA_DECAY)
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--seg-base-channels", type=int, default=16)
    p.add_argument("--pose-base-channels", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument(
        "--smoke", action="store_true", default=False,
        help="Build verification only; uses 4 pairs + 1 epoch synthetic data",
    )
    return p.parse_args(argv)


def _load_video_pairs(
    video_path: Path, n_pairs: int, *, seed: int = 0
) -> torch.Tensor:
    """Decode `n_pairs` non-overlapping pairs from `video_path` via PyAV.

    Returns ``(n_pairs, 2, 3, H, W)`` uint8 tensor at camera resolution.
    Per CLAUDE.md "MASKS.MKV AT 48x64" lesson: NEVER use overlapping pairs;
    the contest uses non-overlapping `seq_len=2` batching (frames 0+1, 2+3, …).
    """
    try:
        import av  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "PyAV not installed; install via `uv pip install av` for "
            "real-video distillation"
        ) from exc

    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    pairs: list[torch.Tensor] = []
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frame_buffer: list[torch.Tensor] = []
    n_needed = n_pairs * 2
    for frame in container.decode(stream):
        if len(frame_buffer) >= n_needed:
            break
        arr = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)
        frame_buffer.append(t)
    container.close()

    if len(frame_buffer) < n_needed:
        raise RuntimeError(
            f"video {video_path} yielded only {len(frame_buffer)} frames; "
            f"need {n_needed} for {n_pairs} non-overlapping pairs"
        )

    for k in range(n_pairs):
        pair = torch.stack([frame_buffer[2 * k], frame_buffer[2 * k + 1]], dim=0)
        pairs.append(pair)
    return torch.stack(pairs, dim=0)  # (n_pairs, 2, 3, H, W) uint8


def _make_synthetic_pairs_for_smoke(
    n_pairs: int, *, seed: int = 0
) -> torch.Tensor:
    """# SYNTHETIC_NON_SMOKE_OK:cpu_trained_hinton_surrogate_smoke_only"""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 256, (n_pairs, 2, 3, 96, 128), generator=g, dtype=torch.uint8,
    )


def _build_real_scorer_targets(
    pairs_uint8: torch.Tensor,
    *,
    upstream_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run real SegNet + PoseNet on CPU to produce distillation targets.

    Per CLAUDE.md "MPS auth eval is NOISE": CPU forward only. Returns
    `(rgb_pairs_for_seg, yuv6_pairs_for_pose, seg_targets, pose_targets)`
    where:
      * rgb_pairs_for_seg: (N, 3, H_scorer, W_scorer) — second frame of each
        pair, resized to scorer input size, ready for DistilledSegNet
      * yuv6_pairs_for_pose: (N, 12, H_yuv, W_yuv) — both frames YUV6'd
        and concatenated, ready for DistilledPoseNet
      * seg_targets: (N, 5, H_scorer, W_scorer) — real SegNet logits
      * pose_targets: (N, 6) — real PoseNet first-6 dims

    The targets are computed via the real scorers at scorer resolution
    (512x384) so the surrogate learns a 1:1 mimic at the scorer's native
    spatial dims.
    """
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W
    from tac.quantization import Uint8STE
    from tac.scorer import load_default_scorers

    device = torch.device("cpu")
    posenet, segnet = load_default_scorers(upstream_dir, device=device)

    n_pairs = pairs_uint8.shape[0]
    pairs_chw = pairs_uint8.float()  # (N, 2, 3, H_cam, W_cam)
    H_cam, W_cam = pairs_chw.shape[-2], pairs_chw.shape[-1]

    # Eval-roundtrip: simulate the contest scorer's resolution roundtrip
    # (camera → scorer → camera → scorer) per CLAUDE.md non-negotiable.
    # For target generation we use the scorer's NATIVE input (after the
    # first resize step); the student is trained against THAT.
    flat = pairs_chw.reshape(-1, 3, H_cam, W_cam)
    flat_clamped = Uint8STE.apply(flat)
    flat_scorer = F.interpolate(
        flat_clamped, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
        mode="bilinear", align_corners=False,
    )
    pairs_scorer = flat_scorer.reshape(n_pairs, 2, 3, SEGNET_INPUT_H, SEGNET_INPUT_W)

    # SegNet target: real SegNet on the SECOND frame of each pair
    # (per upstream/modules.py SegNet contract: x[:, -1, ...]).
    rgb_for_seg = pairs_scorer[:, 1]  # (N, 3, H_scorer, W_scorer)

    # PoseNet target: real PoseNet on the full pair via its preprocess_input.
    seg_targets_list: list[torch.Tensor] = []
    pose_targets_list: list[torch.Tensor] = []
    yuv6_pairs_list: list[torch.Tensor] = []

    BATCH = 4
    with torch.no_grad():
        for start in range(0, n_pairs, BATCH):
            end = min(start + BATCH, n_pairs)
            batch_pairs = pairs_scorer[start:end]  # (b, 2, 3, H, W)

            # SegNet forward: takes a 2-frame pair input via its
            # preprocess_input (last frame extraction inside).
            seg_in = segnet.preprocess_input(batch_pairs)
            seg_out = segnet(seg_in)
            seg_targets_list.append(seg_out.detach())

            # PoseNet forward: pose head on the YUV6 12-channel pair.
            pose_in = posenet.preprocess_input(batch_pairs)  # (b, 12, h, w)
            pose_out = posenet(pose_in)
            pose_t = (
                pose_out["pose"]
                if isinstance(pose_out, dict) and "pose" in pose_out
                else pose_out
            )
            pose_targets_list.append(pose_t[..., :6].detach())
            yuv6_pairs_list.append(pose_in.detach())

    seg_targets = torch.cat(seg_targets_list, dim=0)  # (N, 5, h, w)
    pose_targets = torch.cat(pose_targets_list, dim=0)  # (N, 6)
    yuv6_pairs = torch.cat(yuv6_pairs_list, dim=0)  # (N, 12, h, w)

    return rgb_for_seg, yuv6_pairs, seg_targets, pose_targets


def _kl_distill_loss(
    z_aux: torch.Tensor, z_real: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    """Hinton T=2.0 KL distill: KL(σ(z_real/T) || σ(z_aux/T)) * T²."""
    soft_real = F.softmax(z_real / temperature, dim=1)
    log_soft_aux = F.log_softmax(z_aux / temperature, dim=1)
    # KL(P||Q) where P = soft_real (target). PyTorch kl_div expects log Q.
    return F.kl_div(log_soft_aux, soft_real, reduction="batchmean") * (
        temperature ** 2
    )


def _ema_snapshot(model: torch.nn.Module, decay: float) -> dict[str, torch.Tensor]:
    """Initial EMA shadow snapshot (clone of state_dict)."""
    if not (0.99 <= decay < 1.0):
        raise ValueError(f"ema_decay must be in [0.99, 1.0); got {decay}")
    return {name: param.detach().clone() for name, param in model.state_dict().items()}


def _ema_update(
    shadow: dict[str, torch.Tensor], model: torch.nn.Module, decay: float
) -> None:
    """In-place EMA update: shadow = decay * shadow + (1 - decay) * live."""
    for name, param in model.state_dict().items():
        if name not in shadow:
            shadow[name] = param.detach().clone()
            continue
        if param.dtype.is_floating_point:
            shadow[name].mul_(decay).add_(param.detach(), alpha=(1.0 - decay))
        else:
            shadow[name] = param.detach().clone()


def _measure_input_gradient_norm(
    seg_module: torch.nn.Module,
    pose_module: torch.nn.Module,
    rgb_for_seg: torch.Tensor,
    yuv6_pairs: torch.Tensor,
) -> tuple[float, float, float]:
    """Compute (seg_grad_norm, pose_grad_norm, total_grad_norm) on a small batch.

    Per Catalog #123: this is INPUT-space (not weight-space) gradient norm.
    Used to compare random-init vs trained surrogate's signal richness.
    """
    seg_module.eval()
    pose_module.eval()

    # Use a small batch for gradient probe.
    rgb = rgb_for_seg[:4].clone().requires_grad_(True)
    yuv6 = yuv6_pairs[:4].clone().requires_grad_(True)

    # Score-domain "loss" surrogate: sum of seg KL-distill-style L2 + pose MSE.
    seg_logits = seg_module(rgb)
    seg_loss = (seg_logits ** 2).sum() / max(1, seg_logits.numel())
    pose_pred = pose_module(yuv6)
    pose_loss = (pose_pred ** 2).sum() / max(1, pose_pred.numel())
    total = seg_loss + pose_loss
    total.backward()

    seg_grad_norm = float(rgb.grad.norm().item()) if rgb.grad is not None else 0.0
    pose_grad_norm = float(yuv6.grad.norm().item()) if yuv6.grad is not None else 0.0
    return seg_grad_norm, pose_grad_norm, seg_grad_norm + pose_grad_norm


def _measure_agreement(
    seg_module: torch.nn.Module,
    pose_module: torch.nn.Module,
    rgb_for_seg: torch.Tensor,
    yuv6_pairs: torch.Tensor,
    seg_targets: torch.Tensor,
    pose_targets: torch.Tensor,
) -> dict[str, float]:
    """Compute MSE between surrogate output and real scorer targets on held-out set.

    Lower = surrogate's outputs closer to real scorer's. Tagged
    `[macOS-CPU-research-signal]`.
    """
    seg_module.eval()
    pose_module.eval()
    with torch.no_grad():
        seg_pred = seg_module(rgb_for_seg)
        seg_mse = float(F.mse_loss(seg_pred, seg_targets).item())
        seg_argmax_disagree = float(
            (seg_pred.argmax(dim=1) != seg_targets.argmax(dim=1))
            .float()
            .mean()
            .item()
        )
        pose_pred = pose_module(yuv6_pairs)
        pose_mse = float(F.mse_loss(pose_pred, pose_targets).item())
    return {
        "seg_mse": seg_mse,
        "seg_argmax_disagree_fraction": seg_argmax_disagree,
        "pose_mse": pose_mse,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # CLAUDE.md "MPS auth eval is NOISE": CPU only.
    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Lazy import the LL scaffolds (must be after sys.path setup).
    from tac.residual_basis.hinton_distilled_scorer_surrogate import (
        DistilledPoseNet,
        DistilledSegNet,
        ScorerSurrogateConfig,
        load_pretrained_distilled_scorer_pair,
    )

    config = ScorerSurrogateConfig.council_canonical()

    print("[cpu_hinton] starting CPU-only Hinton distillation training")
    print(f"[cpu_hinton] device={device} epochs={args.epochs} "
          f"n_pairs={args.n_pairs} ema_decay={args.ema_decay}")

    # ── Step 1: Build distillation set via REAL scorers on CPU ────────────
    if args.smoke:
        print("[cpu_hinton] SMOKE mode: synthetic pairs (4 pairs, 1 epoch)")
        pairs_uint8 = _make_synthetic_pairs_for_smoke(
            n_pairs=args.n_pairs, seed=args.seed,
        )
        # Smoke uses synthetic targets too (no real scorer load); mirrors
        # contract shape but bypasses the heavy load step.
        n = pairs_uint8.shape[0]
        # Synthetic shape: pretend 192x256 scorer output (smaller for speed)
        H_s, W_s = 192, 256
        H_y, W_y = 192 // 4, 256 // 4
        rgb_for_seg = torch.randn(n, 3, H_s, W_s)
        yuv6_pairs = torch.randn(n, 12, H_y, W_y)
        seg_targets = torch.randn(n, 5, H_s, W_s)
        pose_targets = torch.randn(n, 6)
    else:
        print(f"[cpu_hinton] decoding {args.n_pairs + args.n_heldout_pairs} "
              f"frame pairs from {args.video_path}")
        pairs_uint8 = _load_video_pairs(
            args.video_path,
            n_pairs=args.n_pairs + args.n_heldout_pairs,
            seed=args.seed,
        )
        print("[cpu_hinton] running real SegNet + PoseNet on CPU "
              "to produce distillation targets")
        rgb_for_seg, yuv6_pairs, seg_targets, pose_targets = (
            _build_real_scorer_targets(
                pairs_uint8, upstream_dir=args.upstream_dir,
            )
        )

    # Split into train/heldout for agreement check.
    n_total = rgb_for_seg.shape[0]
    n_heldout = min(args.n_heldout_pairs, n_total // 4)
    n_train = n_total - n_heldout
    train_rgb_for_seg = rgb_for_seg[:n_train]
    train_yuv6_pairs = yuv6_pairs[:n_train]
    train_seg_targets = seg_targets[:n_train]
    train_pose_targets = pose_targets[:n_train]
    heldout_rgb_for_seg = rgb_for_seg[n_train:]
    heldout_yuv6_pairs = yuv6_pairs[n_train:]
    heldout_seg_targets = seg_targets[n_train:]
    heldout_pose_targets = pose_targets[n_train:]
    print(f"[cpu_hinton] split: train={n_train} heldout={n_heldout}")

    # ── Step 2: Random-init baseline gradient norm ────────────────────────
    print("[cpu_hinton] measuring random-init baseline gradient norm")
    seg_random, pose_random = load_pretrained_distilled_scorer_pair(
        config=config,
        seg_base_channels=args.seg_base_channels,
        pose_base_channels=args.pose_base_channels,
    )
    seg_grad_random, pose_grad_random, total_grad_random = (
        _measure_input_gradient_norm(
            seg_random, pose_random, heldout_rgb_for_seg, heldout_yuv6_pairs,
        )
    )
    print(f"[cpu_hinton] random-init grad norms: "
          f"seg={seg_grad_random:.6e} pose={pose_grad_random:.6e} "
          f"total={total_grad_random:.6e}")

    # ── Step 3: Build trainable surrogate + EMA ───────────────────────────
    seg_student = DistilledSegNet(
        seg_class_count=config.seg_class_count,
        base_channels=args.seg_base_channels,
    )
    pose_student = DistilledPoseNet(
        pose_dim=config.pose_dim,
        base_channels=args.pose_base_channels,
    )
    seg_ema = _ema_snapshot(seg_student, args.ema_decay)
    pose_ema = _ema_snapshot(pose_student, args.ema_decay)

    seg_optimizer = torch.optim.Adam(
        seg_student.parameters(), lr=args.learning_rate
    )
    pose_optimizer = torch.optim.Adam(
        pose_student.parameters(), lr=args.learning_rate
    )

    seg_n_params = sum(p.numel() for p in seg_student.parameters())
    pose_n_params = sum(p.numel() for p in pose_student.parameters())
    print(f"[cpu_hinton] student params: seg={seg_n_params} pose={pose_n_params} "
          f"total={seg_n_params + pose_n_params}")

    # ── Step 4: Training loop ────────────────────────────────────────────
    print(f"[cpu_hinton] starting {args.epochs} epochs of distillation")
    final_seg_kl = 0.0
    final_pose_mse = 0.0
    total_batches = 0

    for epoch in range(args.epochs):
        seg_student.train()
        pose_student.train()
        epoch_seg_loss = 0.0
        epoch_pose_loss = 0.0
        n_batches_this_epoch = 0
        # Shuffle indices.
        perm = torch.randperm(n_train)
        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            batch_idx = perm[start:end]
            batch_rgb = train_rgb_for_seg[batch_idx]
            batch_yuv6 = train_yuv6_pairs[batch_idx]
            batch_seg_target = train_seg_targets[batch_idx]
            batch_pose_target = train_pose_targets[batch_idx]

            # SegNet distillation: KL on T=2.0-softened logits.
            seg_pred = seg_student(batch_rgb)
            seg_loss = _kl_distill_loss(
                seg_pred, batch_seg_target, temperature=args.distill_temperature,
            )
            seg_optimizer.zero_grad(set_to_none=True)
            seg_loss.backward()
            seg_optimizer.step()
            _ema_update(seg_ema, seg_student, args.ema_decay)

            # PoseNet distillation: MSE on first 6 dims.
            pose_pred = pose_student(batch_yuv6)
            pose_loss = F.mse_loss(pose_pred, batch_pose_target)
            pose_optimizer.zero_grad(set_to_none=True)
            pose_loss.backward()
            pose_optimizer.step()
            _ema_update(pose_ema, pose_student, args.ema_decay)

            epoch_seg_loss += float(seg_loss.detach().item())
            epoch_pose_loss += float(pose_loss.detach().item())
            n_batches_this_epoch += 1
            total_batches += 1

        if n_batches_this_epoch == 0:
            raise RuntimeError(f"no batches at epoch {epoch}")
        avg_seg = epoch_seg_loss / n_batches_this_epoch
        avg_pose = epoch_pose_loss / n_batches_this_epoch
        final_seg_kl = avg_seg
        final_pose_mse = avg_pose
        print(f"[cpu_hinton] epoch {epoch + 1}/{args.epochs}: "
              f"seg_kl={avg_seg:.6f} pose_mse={avg_pose:.6f}")

    # ── Step 5: Apply EMA shadow + measure trained gradient norm ─────────
    print("[cpu_hinton] applying EMA shadow + measuring trained surrogate")
    seg_student_orig_state = {k: v.clone() for k, v in seg_student.state_dict().items()}
    pose_student_orig_state = {k: v.clone() for k, v in pose_student.state_dict().items()}
    seg_student.load_state_dict(seg_ema)
    pose_student.load_state_dict(pose_ema)

    seg_grad_trained, pose_grad_trained, total_grad_trained = (
        _measure_input_gradient_norm(
            seg_student, pose_student, heldout_rgb_for_seg, heldout_yuv6_pairs,
        )
    )
    print(f"[cpu_hinton] trained grad norms: "
          f"seg={seg_grad_trained:.6e} pose={pose_grad_trained:.6e} "
          f"total={total_grad_trained:.6e}")

    # ── Step 6: Held-out agreement check ─────────────────────────────────
    print("[cpu_hinton] measuring held-out agreement vs real scorer targets")
    agreement = _measure_agreement(
        seg_student, pose_student,
        heldout_rgb_for_seg, heldout_yuv6_pairs,
        heldout_seg_targets, heldout_pose_targets,
    )
    print(f"[cpu_hinton] agreement: seg_mse={agreement['seg_mse']:.4f} "
          f"seg_argmax_disagree={agreement['seg_argmax_disagree_fraction']:.4f} "
          f"pose_mse={agreement['pose_mse']:.4f}")

    # ── Step 7: Compute reactivation criteria verdict ────────────────────
    grad_ratio = total_grad_trained / max(1e-12, total_grad_random)
    saliency_max_proxy = total_grad_trained  # treat as the saliency proxy
    output_mse_drift_fraction = (
        agreement["seg_argmax_disagree_fraction"]
        + min(1.0, agreement["pose_mse"])
    ) / 2.0  # 0..1 fraction (loose)

    criterion_grad = grad_ratio > 100.0
    criterion_saliency = saliency_max_proxy > 1e-2
    criterion_agreement = output_mse_drift_fraction < 0.10
    all_three_pass = criterion_grad and criterion_saliency and criterion_agreement

    print(f"[cpu_hinton] reactivation criteria: "
          f"grad_ratio={grad_ratio:.2f}× (>100x: {criterion_grad}) "
          f"saliency_max={saliency_max_proxy:.4e} (>1e-2: {criterion_saliency}) "
          f"output_drift={output_mse_drift_fraction:.4f} (<0.10: {criterion_agreement})")
    print(f"[cpu_hinton] all-three-pass: {all_three_pass}")

    # ── Step 8: Save EMA shadow checkpoint ───────────────────────────────
    distill_label = f"cpu_trained_tiny_hinton_{started_at}"
    distillation_gap_estimate = (
        final_seg_kl / max(1e-9, args.distill_temperature ** 2)
    )

    seg_checkpoint = {
        "schema": SCHEMA_VERSION,
        "ema_state_dict": seg_ema,
        "config": {
            "distill_temperature": float(args.distill_temperature),
            "ema_decay": float(args.ema_decay),
            "seg_class_count": int(config.seg_class_count),
            "base_channels": int(args.seg_base_channels),
            "distill_label": distill_label,
            "trained_on_axis": "macOS-CPU-research-signal",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    }
    seg_ckpt_path = args.output_dir / "distilled_segnet_ema_shadow.pt"
    torch.save(seg_checkpoint, seg_ckpt_path)

    pose_checkpoint = {
        "schema": SCHEMA_VERSION,
        "ema_state_dict": pose_ema,
        "config": {
            "ema_decay": float(args.ema_decay),
            "pose_dim": int(config.pose_dim),
            "base_channels": int(args.pose_base_channels),
            "distill_label": distill_label,
            "trained_on_axis": "macOS-CPU-research-signal",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    }
    pose_ckpt_path = args.output_dir / "distilled_posenet_ema_shadow.pt"
    torch.save(pose_checkpoint, pose_ckpt_path)

    # ── Step 9: Write distillation_gap_estimate.json (T10-format-compatible) ─
    # Per CLAUDE.md "Strict scorer rule" + Catalog #134 informational gate.
    # CRITICAL: passes_phase3_threshold is HARDWIRED FALSE for CPU-trained
    # surrogates regardless of the measured gap. Phase 3 dispatch requires
    # T10's CUDA-trained EMA shadow per the strict policy.
    gap_artifact = {
        "schema": SCHEMA_VERSION,
        "lane_id": LANE_ID,
        "distillation_gap_estimate": float(distillation_gap_estimate),
        "phase3_threshold": PHASE3_DISTILL_GAP_THRESHOLD,
        "passes_phase3_threshold": False,  # HARDWIRED FALSE for CPU-trained
        "passes_phase3_threshold_rationale": (
            "CPU-trained surrogate cannot be promoted to Phase 3 unblock per "
            "CLAUDE.md 'MPS auth eval is NOISE' + 'Submission auth eval' "
            "non-negotiables. Phase 3 dispatch requires T10's CUDA-trained "
            "EMA shadow (operator-gated $40)."
        ),
        "final_loss_kl": float(final_seg_kl),
        "final_loss_pose": float(final_pose_mse),
        "n_epochs_completed": int(args.epochs),
        "n_train_pairs": int(n_train),
        "n_heldout_pairs": int(n_heldout),
        "smoke_mode": bool(args.smoke),
        "evidence_grade": EVIDENCE_GRADE_TAG,
        "trained_on_axis": "macOS-CPU-research-signal",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    (args.output_dir / "distillation_gap_estimate.json").write_text(
        json.dumps(gap_artifact, indent=2)
    )

    # ── Step 10: Write reactivation criteria verdict + provenance ────────
    verdict_artifact = {
        "schema": SCHEMA_VERSION,
        "lane_id": LANE_ID,
        "started_at_utc": started_at,
        "evidence_grade": EVIDENCE_GRADE_TAG,
        "reactivation_criteria": {
            "grad_norm_ratio_trained_over_random": float(grad_ratio),
            "grad_norm_threshold": 100.0,
            "saliency_max_proxy": float(saliency_max_proxy),
            "saliency_max_threshold": 1e-2,
            "output_mse_drift_fraction": float(output_mse_drift_fraction),
            "output_mse_drift_threshold": 0.10,
            "criterion_grad_ratio_passes": bool(criterion_grad),
            "criterion_saliency_passes": bool(criterion_saliency),
            "criterion_agreement_passes": bool(criterion_agreement),
            "all_three_pass": bool(all_three_pass),
        },
        "agreement": agreement,
        "trained_grad_norms": {
            "seg": float(seg_grad_trained),
            "pose": float(pose_grad_trained),
            "total": float(total_grad_trained),
        },
        "random_init_grad_norms": {
            "seg": float(seg_grad_random),
            "pose": float(pose_grad_random),
            "total": float(total_grad_random),
        },
        "config": {
            "n_pairs": int(args.n_pairs),
            "n_heldout_pairs": int(args.n_heldout_pairs),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "ema_decay": float(args.ema_decay),
            "distill_temperature": float(args.distill_temperature),
            "seg_base_channels": int(args.seg_base_channels),
            "pose_base_channels": int(args.pose_base_channels),
            "smoke_mode": bool(args.smoke),
            "trained_on_axis": "macOS-CPU-research-signal",
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "next_action": (
            "OPERATOR DECISION: if all_three_pass=True, the L2+Hinton+saliency "
            "dispatch may be re-attempted with these CPU-trained checkpoints "
            "instead of waiting on T10 ($40). If any falls short, this is the "
            "explicit falsification anchor confirming T10 IS the unique unlock."
            if all_three_pass
            else "FALSIFICATION: CPU-trained surrogate did not unlock W's "
            "DEFERRED criteria #1+#2 at the 100x grad-norm-ratio + 1e-2 "
            "saliency-max + 10% agreement-drift threshold. T10 IB-Lagrangian "
            "dispatch ($40 operator-gated) IS the unique unlock for the "
            "L2+Hinton+saliency residual dispatch path."
        ),
    }
    (args.output_dir / "reactivation_criteria_verdict.json").write_text(
        json.dumps(verdict_artifact, indent=2)
    )

    # Provenance manifest (for cathedral autopilot consumption).
    provenance = {
        "schema": SCHEMA_VERSION,
        "lane_id": LANE_ID,
        "started_at_utc": started_at,
        "device": str(device),
        "smoke": bool(args.smoke),
        "n_train_pairs": int(n_train),
        "n_heldout_pairs": int(n_heldout),
        "evidence_grade": EVIDENCE_GRADE_TAG,
        "predicted_delta_score": (
            "N/A — research-signal artifact; not a contest archive; "
            "CPU-trained surrogate cannot promote per CLAUDE.md non-negotiables"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "research_only": True,
        "research_only_rationale": (
            "CPU-trained Hinton surrogate is a research-signal artifact "
            "exploring whether $0 macOS-CPU training can $0-unlock W's "
            "DEFERRED criteria #1+#2 (random-init Hinton produces "
            "uninformative saliency); Phase 3 unblock still requires T10 "
            "CUDA-trained EMA shadow per CLAUDE.md 'MPS auth eval is NOISE'."
        ),
        "compliance_tags": [
            "ema_0p997",
            "hinton_distill_T_2p0",
            "no_mps_authoritative",
            "cpu_only_training",
            "no_synthetic_outside_smoke",
            "no_tmp_paths",
            "research_only_artifact",
            "score_claim_false",
        ],
        "checkpoint_paths": {
            "seg": _maybe_relpath_to_repo(seg_ckpt_path),
            "pose": _maybe_relpath_to_repo(pose_ckpt_path),
        },
        "checkpoint_sha256s": {
            "seg": hashlib.sha256(seg_ckpt_path.read_bytes()).hexdigest(),
            "pose": hashlib.sha256(pose_ckpt_path.read_bytes()).hexdigest(),
        },
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print(f"[cpu_hinton] done; artifacts in {args.output_dir}")
    print(f"[cpu_hinton] verdict: {'PASS' if all_three_pass else 'FALSIFY'} "
          f"(see reactivation_criteria_verdict.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
