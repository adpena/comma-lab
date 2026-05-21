# SPDX-License-Identifier: MIT
"""Legacy TAC post-filter proxy evaluation.

This helper mirrors the contest component formula for TAC post-filter
experiments, but it is not the authoritative submission evaluator.  It does
not execute the byte-closed contest path ``archive.zip -> inflate.sh ->
inflated/*.raw -> upstream/evaluate.py`` and must not be used for promotion,
leaderboard, rank/kill, or public score claims.

Usage as library:
    from tac.evaluate import canonical_score
    result = canonical_score(
        int8_path="weights/best.pt",
        archive_path="submission/archive.zip",
        gt_video_path="videos/0.mkv",
        gt_dir="videos/",
        models_dir="models/",
    )

Usage as CLI:
    python -m tac.evaluate --checkpoint weights/best.pt --upstream /path/to/upstream
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch

from .models import ScoreResult


def canonical_score(
    int8_path: str | Path,
    archive_path: str | Path,
    gt_video_path: str | Path,
    gt_dir: str | Path,
    models_dir: str | Path,
    upstream_dir: str | Path | None = None,
    variant: str = "standard",
    hidden: int = 64,
    kernel: int = 3,
    device: str | None = None,
) -> ScoreResult:
    """Evaluate a TAC post-filter under the legacy proxy harness.

    All paths are explicit parameters — nothing hardcoded.
    The component definitions mirror the official formula:
    - Hard argmax SegNet disagreement
    - MSE on first 6 PoseNet outputs
    - Rate = archive_size / uncompressed_size
    - Score = 100*seg + sqrt(10*pose) + 25*rate

    This bypasses the official ``inflate.sh``/``upstream/evaluate.py`` custody
    path, so the return value is proxy evidence only.
    """
    from .architectures import PairAwarePostFilter, build_postfilter
    from .data import build_pairs, decode_archive, decode_video
    from .quantization import load_int8
    from .scorer import detect_device, load_scorers

    t_start = time.monotonic()

    if device is None:
        device = str(detect_device())
    if upstream_dir is None:
        upstream_dir = str(Path(models_dir).parent)

    # Load model
    model = build_postfilter(variant, hidden=hidden, kernel=kernel)
    load_int8(int8_path, model, device=device)
    model = model.eval().to(device)
    t_model = time.monotonic()

    # Load data
    comp_frames = decode_archive(str(archive_path))
    gt_frames = decode_video(str(gt_video_path))
    assert len(comp_frames) == len(gt_frames), f"Frame count mismatch: {len(comp_frames)} comp vs {len(gt_frames)} GT"
    t_data = time.monotonic()

    # Load scorers
    posenet, segnet = load_scorers(
        Path(models_dir) / "posenet.safetensors",
        Path(models_dir) / "segnet.safetensors",
        device=device,
        upstream_dir=str(upstream_dir),
    )

    # Compute rate
    archive_size = Path(archive_path).stat().st_size
    gt_dir = Path(gt_dir)
    uncompressed_size = sum(f.stat().st_size for f in gt_dir.rglob("*") if f.is_file())
    rate = archive_size / uncompressed_size

    # Build pairs and evaluate — ALL pairs, no subsampling
    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    assert len(comp_pairs) == len(gt_pairs)

    is_pair_aware = isinstance(model, PairAwarePostFilter)
    total_pose, total_seg, n_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for cp, gp in zip(comp_pairs, gt_pairs, strict=True):
            cp = cp.to(device)
            gp = gp.to(device)

            B, T, H, W, C = cp.shape
            if is_pair_aware:
                f0 = cp[:, 0].float().permute(0, 3, 1, 2).contiguous()
                f1 = cp[:, 1].float().permute(0, 3, 1, 2).contiguous()
                out0 = model(torch.cat([f0, f1], dim=1))
                out1 = model(torch.cat([f1, f0], dim=1))
                filtered_pair = torch.stack([out0.permute(0, 2, 3, 1), out1.permute(0, 2, 3, 1)], dim=1)
            else:
                frames = cp.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
                filtered = model(frames)
                filtered_pair = filtered.permute(0, 2, 3, 1).reshape(B, T, H, W, C)

            # uint8 round-trip: matches official inflate → disk → scorer pipeline
            filtered_pair = filtered_pair.round().clamp(0, 255).to(torch.uint8).float()

            # Scorer input format: (B, T, C, H, W)
            fx = filtered_pair.float().permute(0, 1, 4, 2, 3).contiguous()
            gx = gp.float().permute(0, 1, 4, 2, 3).contiguous()

            # PoseNet: per-sample MSE on first 6 outputs
            fp_in = posenet.preprocess_input(fx)
            gp_in = posenet.preprocess_input(gx)
            fp_out = posenet(fp_in)
            gp_out = posenet(gp_in)
            pose_per_sample = (
                (fp_out["pose"][..., :6] - gp_out["pose"][..., :6])
                .pow(2)
                .mean(dim=tuple(range(1, fp_out["pose"].ndim)))
            )

            # SegNet: hard argmax disagreement
            fs_in = segnet.preprocess_input(fx)
            gs_in = segnet.preprocess_input(gx)
            fs_out = segnet(fs_in)
            gs_out = segnet(gs_in)
            diff = (fs_out.argmax(dim=1) != gs_out.argmax(dim=1)).float()
            seg_per_sample = diff.mean(dim=tuple(range(1, diff.ndim)))

            total_pose += pose_per_sample.sum().item()
            total_seg += seg_per_sample.sum().item()
            n_samples += B

    t_score = time.monotonic()

    avg_pose = total_pose / n_samples
    avg_seg = total_seg / n_samples
    score = 100.0 * avg_seg + math.sqrt(10.0 * avg_pose) + 25.0 * rate

    t_total = time.monotonic() - t_start
    timing = {
        "model_load_s": round(t_model - t_start, 2),
        "data_load_s": round(t_data - t_model, 2),
        "scoring_s": round(t_score - t_data, 2),
        "total_s": round(t_total, 2),
    }
    print(
        f"[eval] timing: model={timing['model_load_s']}s, "
        f"data={timing['data_load_s']}s, scoring={timing['scoring_s']}s, "
        f"total={timing['total_s']}s"
    )

    # ScoreResult imported at module level (L30); no need to re-import.
    return ScoreResult(
        score=score,
        pose=avg_pose,
        seg=avg_seg,
        rate=rate,
        rate_contribution=25.0 * rate,
        pose_contribution=math.sqrt(10.0 * avg_pose),
        seg_contribution=100.0 * avg_seg,
        n_samples=n_samples,
        archive=str(archive_path),
        checkpoint=str(int8_path),
        timing=timing,
    )


def find_checkpoints(
    weights_dir: str | Path,
    tag_pattern: str = "",
) -> list[dict]:
    """Find all checkpoint metadata files, optionally filtered by tag pattern."""
    results = []
    for path in sorted(Path(weights_dir).glob("*_best_meta.json")):
        if tag_pattern and tag_pattern not in path.stem:
            continue
        try:
            data = json.loads(path.read_text())
            data["meta_path"] = str(path)
            results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(results, key=lambda d: d.get("scorer", float("inf")))


def average_top_k_checkpoints(
    weights_dir: str | Path,
    tag_pattern: str = "",
    k: int = 3,
    output_path: str | Path | None = None,
) -> dict:
    """Average the top-K checkpoints by scorer and save as int8."""
    from .quantization import save_int8_from_state_dict

    checkpoints = find_checkpoints(weights_dir, tag_pattern)
    top_k = [c for c in checkpoints if os.path.exists(c.get("fp32_path", ""))][:k]

    if not top_k:
        raise ValueError(f"No checkpoints found matching '{tag_pattern}' in {weights_dir}")

    print(f"Averaging top-{len(top_k)} checkpoints:")
    for c in top_k:
        print(f"  epoch {c['epoch']}, scorer {c['scorer']:.4f}")

    states = [torch.load(c["fp32_path"], map_location="cpu", weights_only=True) for c in top_k]
    avg = {}
    for key in states[0]:
        tensors = [s[key].float() for s in states if key in s]
        if tensors:
            avg[key] = torch.stack(tensors).mean(dim=0)

    if output_path is None:
        output_path = Path(weights_dir) / f"postfilter_{tag_pattern}_top{len(top_k)}_avg_int8.pt"

    meta = top_k[0].get("meta", {})
    size = save_int8_from_state_dict(avg, output_path, meta=meta)

    return {
        "source_epochs": [c["epoch"] for c in top_k],
        "source_scorers": [c["scorer"] for c in top_k],
        "avg_scorer": sum(c["scorer"] for c in top_k) / len(top_k),
        "int8_path": str(output_path),
        "int8_size": size,
    }


if __name__ == "__main__":
    """CLI: python -m tac.evaluate --checkpoint path/to/int8.pt --upstream /path/to/upstream"""
    import argparse

    parser = argparse.ArgumentParser(description="Canonical tac evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to int8 checkpoint")
    parser.add_argument("--upstream", required=True, help="Path to upstream repo")
    parser.add_argument(
        "--archive",
        default=None,
        help="Path to archive.zip (default: upstream/../submissions/robust_current/archive.zip)",
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--variant", default="standard")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    upstream = Path(args.upstream)
    archive = Path(args.archive) if args.archive else upstream.parent / "submissions" / "robust_current" / "archive.zip"

    print("Legacy TAC post-filter proxy evaluation (non-authoritative)")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  archive:    {archive}")
    print(f"  upstream:   {upstream}")
    print()

    result = canonical_score(
        int8_path=args.checkpoint,
        archive_path=archive,
        gt_video_path=upstream / "videos" / "0.mkv",
        gt_dir=upstream / "videos",
        models_dir=upstream / "models",
        upstream_dir=upstream,
        variant=args.variant,
        hidden=args.hidden,
        device=args.device,
    )

    print(f"{'=' * 60}")
    print(f"  CANONICAL SCORE:      {result.score:.4f}")
    print(f"  SegNet contribution:  {result.seg_contribution:.4f}  (seg={result.seg:.8f})")
    print(f"  PoseNet contribution: {result.pose_contribution:.4f}  (pose={result.pose:.8f})")
    print(f"  Rate contribution:    {result.rate_contribution:.4f}  (rate={result.rate:.8f})")
    print(f"  N samples:            {result.n_samples}")
    print(f"{'=' * 60}")
