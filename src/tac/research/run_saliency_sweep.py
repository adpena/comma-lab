#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
Run all saliency-weighted post-filter training variants in a single process.
Loads data and scorers ONCE, then trains all variants sequentially.

Usage:
  PYTHONUNBUFFERED=1 uv run --with torch --with safetensors --with timm --with einops \
         --with segmentation-models-pytorch --with av --with numpy \
         python -u experiments/run_saliency_sweep.py
"""
import gc
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tac.architectures import PostFilter
from tac.data import build_pairs, decode_archive, decode_video, SEQ_LEN as seq_len
from tac.losses import scorer_forward_pair
from tac.scorer import detect_device, load_scorers
from tac.proxy_eval import _default_paths
from tac.quantization import normalize_postfilter_meta, save_int8 as save_model_int8

PROJECT = Path(__file__).resolve().parent.parent.parent.parent  # src/tac/research -> project root
_PROJECT, _UPSTREAM, VIDEOS_DIR, _LIVE_ARCHIVE, _LEGACY_ARCHIVE = _default_paths()
MODELS_DIR = _UPSTREAM / "models"
ARCHIVE_ZIP = _LEGACY_ARCHIVE
OUTPUT_DIR = PROJECT / "experiments" / "postfilter_weights"
SALIENCY_PATH = PROJECT / "experiments" / "masks" / "posenet_saliency.npy"
DEVICE = detect_device()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def apply_filter_to_pair(model, comp_pair, device):
    """Apply post-filter to a frame pair. Input: (1,2,H,W,3) uint8. Output: (1,2,H,W,3) float."""
    B, T, H, W, C = comp_pair.shape
    x = comp_pair.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous().to(device)
    with torch.no_grad():
        y = model(x)
    return y.permute(0, 2, 3, 1).reshape(B, T, H, W, C).clamp(0, 255)


def compute_pair_loss(filtered_pair, gt_pair, posenet, segnet):
    """Compute scorer loss. Returns (loss, pose_dist, seg_dist)."""
    fx = filtered_pair.float().permute(0, 1, 4, 2, 3).contiguous()
    gx = gt_pair.float().permute(0, 1, 4, 2, 3).contiguous()
    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)
    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean().item()
    pred_soft = F.softmax(fs_out, dim=1)
    gt_soft = F.softmax(gs_out, dim=1)
    seg_dist = (1.0 - (pred_soft * gt_soft).sum(dim=1).mean()).item()
    loss = 100.0 * seg_dist + math.sqrt(10.0 * pose_dist)
    return loss, pose_dist, seg_dist


def compute_combined_loss(filtered_pair, gt_pair, comp_pair, posenet, segnet, sal_weights, sal_lambda):
    """Compute combined loss with saliency reconstruction term."""
    loss, pose_dist, seg_dist = compute_pair_loss(filtered_pair, gt_pair, posenet, segnet)
    scorer_loss_val = loss
    # Saliency reconstruction loss: penalize deviation from GT in salient regions
    sal_recon = ((filtered_pair - gt_pair.float()).pow(2) * sal_weights.unsqueeze(0).unsqueeze(-1)).mean()
    total_loss = torch.tensor(loss, device=filtered_pair.device, requires_grad=True) + sal_lambda * sal_recon
    return total_loss, scorer_loss_val, pose_dist, seg_dist, sal_recon.item()

VARIANTS = [
    {"alpha": 5,  "hidden": 16, "tag": "saliency_alpha5"},
    {"alpha": 10, "hidden": 16, "tag": "saliency_alpha10"},
    {"alpha": 20, "hidden": 16, "tag": "saliency_alpha20"},
    {"alpha": 10, "hidden": 32, "tag": "saliency_alpha10_h32"},
]

N_EPOCHS = 50
TRAIN_SUBSAMPLE = 16  # train on ~37 pairs per epoch
EVAL_SUBSAMPLE = 12   # evaluate on ~50 pairs
ACCUM_STEPS = 4
SAL_LAMBDA = 0.1


def train_variant(tag, alpha, hidden, comp_pairs, gt_pairs, sal_all,
                  posenet, segnet, n_pairs, eval_indices, baseline_loss,
                  baseline_pose, baseline_seg):
    """Train a single variant and return results dict."""
    print(f"\n{'#' * 70}")
    print(f"# VARIANT: {tag} (alpha={alpha}, hidden={hidden})")
    print(f"{'#' * 70}")

    # Build per-pair saliency weights for this alpha
    n_frames = sal_all.shape[0]
    sal_weighted = 1.0 + alpha * sal_all  # (n_frames, 1, H, W)
    sal_pairs = []
    for i in range(0, n_frames - 1, seq_len):
        if i + seq_len > n_frames:
            break
        sal_pairs.append(sal_weighted[i:i + seq_len])

    model = PostFilter(hidden=hidden, kernel=3).to(DEVICE)
    param_count = count_params(model)
    print(f"[{tag}] Model: {param_count} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5
    )

    train_size = max(1, n_pairs // TRAIN_SUBSAMPLE)
    print(f"[{tag}] Training: {N_EPOCHS} epochs, {train_size} pairs/epoch")
    print(f"{'epoch':>5} {'total':>10} {'scorer':>10} {'pose':>12} {'seg':>12} {'lr':>10}")
    print("-" * 65)

    best_loss = float("inf")
    best_state = None

    for epoch in range(N_EPOCHS):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()

        epoch_loss = 0.0
        epoch_scorer = 0.0
        epoch_pose = 0.0
        epoch_seg = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            total_loss, scorer_loss, pd, sd, sal_recon = compute_combined_loss(
                filtered, gt_pairs[idx], comp_pairs[idx],
                posenet, segnet, sal_pairs[idx], SAL_LAMBDA
            )
            (total_loss / ACCUM_STEPS).backward()

            epoch_loss += total_loss.item()
            epoch_scorer += scorer_loss
            epoch_pose += pd
            epoch_seg += sd

            if (step_i + 1) % ACCUM_STEPS == 0 or (step_i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        avg_scorer = epoch_scorer / len(indices)
        avg_pose = epoch_pose / len(indices)
        avg_seg = epoch_seg / len(indices)

        if avg_scorer < best_loss:
            best_loss = avg_scorer
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = epoch_loss / len(indices)
            print(f"{epoch + 1:>5} {avg_loss:>10.4f} {avg_scorer:>10.4f} "
                  f"{avg_pose:>12.6f} {avg_seg:>12.6f} {lr:>10.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    n_eval = len(eval_indices)
    print(f"[{tag}] Final eval on {n_eval} pairs...")
    model.eval()
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            _, pd, sd = compute_pair_loss(filtered, gt_pairs[idx], posenet, segnet)
            total_pose += pd
            total_seg += sd
    final_pose = total_pose / n_eval
    final_seg = total_seg / n_eval
    final_loss = 100.0 * final_seg + math.sqrt(10.0 * final_pose)

    delta = final_loss - baseline_loss
    pose_delta = final_pose - baseline_pose
    seg_delta = final_seg - baseline_seg
    print(f"[{tag}] Result: loss={final_loss:.4f} (delta={delta:+.4f}) "
          f"pose={final_pose:.6f} ({pose_delta:+.6f}) "
          f"seg={final_seg:.6f} ({seg_delta:+.6f})")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = normalize_postfilter_meta(hidden, 3, alpha)

    fp32_path = OUTPUT_DIR / f"postfilter_{tag}_fp32.pt"
    torch.save(model.state_dict(), fp32_path)

    int8_path = OUTPUT_DIR / f"postfilter_{tag}_int8.pt"
    int8_size = save_model_int8(model, int8_path, meta=meta)
    print(f"[{tag}] Saved: {int8_path} ({int8_size} bytes)")

    del model, optimizer, scheduler, sal_weighted, sal_pairs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "tag": tag,
        "baseline_loss": baseline_loss,
        "final_loss": final_loss,
        "delta": delta,
        "pose": final_pose,
        "seg": final_seg,
        "pose_delta": pose_delta,
        "seg_delta": seg_delta,
    }


def main():
    print(f"[sweep] Device: {DEVICE}")
    print("[sweep] Loading scorers...")
    posenet, segnet = load_scorers(DEVICE)

    print("[sweep] Decoding compressed archive...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    print(f"[sweep] Compressed frames: {len(comp_frames)}")

    print("[sweep] Decoding ground truth...")
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    print(f"[sweep] GT frames: {len(gt_frames)}")

    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]

    # Load raw saliency (without alpha weighting -- each variant applies its own alpha)
    print("[sweep] Loading saliency map...")
    sal_raw = np.load(str(SALIENCY_PATH))
    sal_t = torch.from_numpy(sal_raw).float()
    if sal_t.shape[0] < n:
        pad = sal_t[-1:].expand(n - sal_t.shape[0], -1, -1)
        sal_t = torch.cat([sal_t, pad], dim=0)
    sal_t = sal_t[:n].unsqueeze(1).to(DEVICE)  # (n_frames, 1, H, W)

    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = len(comp_pairs)
    print(f"[sweep] {n_pairs} frame pairs")

    del comp_frames, gt_frames
    gc.collect()

    comp_pairs = [p.to(DEVICE) for p in comp_pairs]
    gt_pairs = [p.to(DEVICE) for p in gt_pairs]

    # Compute baseline
    eval_indices = list(range(0, n_pairs, EVAL_SUBSAMPLE))
    n_eval = len(eval_indices)
    print(f"\n[sweep] Computing baseline on {n_eval}/{n_pairs} pairs...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            _, pd, sd = compute_pair_loss(comp_pairs[idx].float(), gt_pairs[idx], posenet, segnet)
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_eval
    baseline_seg = total_seg / n_eval
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[sweep] Baseline: loss={baseline_loss:.4f}, "
          f"pose={baseline_pose:.6f}, seg={baseline_seg:.6f}")

    # Train all variants
    results = []
    for v in VARIANTS:
        try:
            r = train_variant(
                v["tag"], v["alpha"], v["hidden"],
                comp_pairs, gt_pairs, sal_t,
                posenet, segnet, n_pairs, eval_indices,
                baseline_loss, baseline_pose, baseline_seg
            )
            results.append(r)
        except Exception as e:
            print(f"FAILED: {v['tag']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"tag": v["tag"], "error": str(e)})

    # Comparison table
    print(f"\n\n{'=' * 90}")
    print("COMPARISON TABLE: Saliency-Weighted Post-Filter Variants")
    print(f"{'=' * 90}")
    print(f"{'Variant':<25} {'Loss':>8} {'Delta':>8} {'Pose':>12} {'Seg':>12} "
          f"{'Pose_d':>10} {'Seg_d':>10}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['tag']:<25} FAILED: {r['error']}")
        else:
            print(f"{r['tag']:<25} {r['final_loss']:>8.4f} {r['delta']:>+8.4f} "
                  f"{r['pose']:>12.6f} {r['seg']:>12.6f} "
                  f"{r['pose_delta']:>+10.6f} {r['seg_delta']:>+10.6f}")

    print(f"\nBaseline (no filter): loss={baseline_loss:.4f}")
    print("Current best (h16 canonical): loss~2.05 (from prior experiments)")
    print("\nWeights saved to: experiments/postfilter_weights/postfilter_saliency_alpha*_int8.pt")


if __name__ == "__main__":
    main()
