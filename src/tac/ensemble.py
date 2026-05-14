# SPDX-License-Identifier: MIT
"""Per-pair ensemble selection: best-of-N frame selection.

Given N candidate frame sets from different generators/configs, selects the
best-scoring version of each pair independently and assembles the final
frame tensor. This is a score-free improvement: it always helps or is neutral.

The selection criterion is the proxy score per pair: the pair with the lowest
combined pose + seg distortion wins.

Usage::

    from tac.ensemble import select_best_pairs

    # candidates: list of (N, H, W, 3) tensors from different generators
    best_frames = select_best_pairs(
        candidates=[renderer_frames, tto_frames, warp_frames],
        gt_frames=gt_frames,
        posenet=posenet,
        segnet=segnet,
        device=device,
    )
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from tac.scorer import comma_score

logger = logging.getLogger(__name__)


def score_pairs(
    frames: torch.Tensor,
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Compute per-pair proxy score for a frame tensor.

    Args:
        frames: (N, H, W, 3) float tensor of candidate frames.
        gt_frames: list of (H, W, 3) uint8 GT frames.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        device: computation device.
        batch_size: pairs per forward pass.

    Returns:
        (P,) float tensor of per-pair scores (lower is better), P = N // 2.
    """
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    N = frames.shape[0]
    P = N // 2
    pair_scores = torch.zeros(P)

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
            cand_flat = F.interpolate(cand_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                                      mode="bilinear", align_corners=False)
            gt_flat = F.interpolate(gt_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                                    mode="bilinear", align_corners=False)
            cand_chw = cand_flat.reshape(B, T, C, SEGNET_INPUT_H, SEGNET_INPUT_W)
            gt_chw = gt_flat.reshape(B, T, C, SEGNET_INPUT_H, SEGNET_INPUT_W)

        cand_chw = cand_chw.round().clamp(0, 255)

        with torch.no_grad():
            # PoseNet per-pair MSE
            fp_in = posenet.preprocess_input(cand_chw)
            gp_in = posenet.preprocess_input(gt_chw)
            fp_out = posenet(fp_in)
            gp_out = posenet(gp_in)
            pose_mse = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean(dim=-1)

            # SegNet per-pair disagreement
            fs_in = segnet.preprocess_input(cand_chw)
            gs_in = segnet.preprocess_input(gt_chw)
            fs_out = segnet(fs_in)
            gs_out = segnet(gs_in)
            diff = (fs_out.argmax(dim=1) != gs_out.argmax(dim=1)).float()
            seg_disagree = diff.mean(dim=tuple(range(1, diff.ndim)))

            # Combined per-pair score using official formula (evaluate.py line 92)
            for i in range(B):
                pose_val = pose_mse[i].item()
                seg_val = seg_disagree[i].item()
                pair_scores[start + i] = comma_score(pose_val, seg_val, rate=0.0)

    return pair_scores


def select_best_pairs(
    candidates: list[torch.Tensor],
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[torch.Tensor, dict]:
    """Select best-scoring pair from N candidates for each pair position.

    Args:
        candidates: list of K (N, H, W, 3) float tensors from different generators.
        gt_frames: list of (H, W, 3) uint8 GT frames.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        device: computation device.
        batch_size: pairs per scoring batch.

    Returns:
        (best_frames, stats) where:
            best_frames: (N, H, W, 3) float tensor assembled from best pairs.
            stats: dict with selection statistics.
    """
    K = len(candidates)
    if K == 0:
        raise ValueError("Need at least one candidate")
    if K == 1:
        return candidates[0], {"n_candidates": 1, "selections": {0: candidates[0].shape[0] // 2}}

    N = candidates[0].shape[0]
    P = N // 2
    for i, cand in enumerate(candidates):
        if cand.shape != candidates[0].shape:
            raise ValueError(
                f"Candidate {i} shape {cand.shape} != candidate 0 shape {candidates[0].shape}"
            )

    # Score each candidate
    logger.info("[ensemble] Scoring %d candidates (%d pairs each)...", K, P)
    all_scores = []
    for i, cand in enumerate(candidates):
        scores = score_pairs(cand, gt_frames, posenet, segnet, device, batch_size)
        all_scores.append(scores)
        logger.info("  Candidate %d: mean_score=%.4f, min=%.4f, max=%.4f",
                    i, scores.mean(), scores.min(), scores.max())

    # Stack: (K, P) and take argmin per pair
    score_matrix = torch.stack(all_scores, dim=0)  # (K, P)
    best_idx = score_matrix.argmin(dim=0)  # (P,) — which candidate is best per pair

    # Assemble best frames
    best_frames = torch.zeros_like(candidates[0])
    selections = {i: 0 for i in range(K)}

    for k in range(P):
        winner = best_idx[k].item()
        best_frames[2 * k] = candidates[winner][2 * k]
        best_frames[2 * k + 1] = candidates[winner][2 * k + 1]
        selections[winner] += 1

    # Compute score of the assembled result
    ensemble_scores = torch.zeros(P)
    for k in range(P):
        winner = best_idx[k].item()
        ensemble_scores[k] = all_scores[winner][k]

    individual_means = [s.mean().item() for s in all_scores]
    ensemble_mean = ensemble_scores.mean().item()

    stats = {
        "n_candidates": K,
        "n_pairs": P,
        "selections": selections,
        "individual_mean_scores": individual_means,
        "ensemble_mean_score": ensemble_mean,
        "improvement_over_best_individual": min(individual_means) - ensemble_mean,
    }

    logger.info("[ensemble] Selection: %s", selections)
    logger.info("[ensemble] Individual means: %s", [f'{s:.4f}' for s in individual_means])
    logger.info("[ensemble] Ensemble mean: %.4f (improvement: %+.4f)",
                ensemble_mean, stats['improvement_over_best_individual'])

    return best_frames, stats
