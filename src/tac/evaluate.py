"""Canonical evaluation for task-aware codec post-filters.

ONE evaluation path. Matches the official scorer exactly.
No ambiguity, no archive mismatches, no sampling shortcuts.

Usage:
    from tac.evaluate import canonical_score
    result = canonical_score("path/to/int8.pt")  # that's it
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).parent.parent.parent
UPSTREAM = REPO / "workspace" / "upstream" / "comma_video_compression_challenge"
SUBMISSION_ARCHIVE = REPO / "submissions" / "robust_current" / "archive.zip"
GT_VIDEO = UPSTREAM / "videos" / "0.mkv"
GT_DIR = UPSTREAM / "videos"
MODELS_DIR = UPSTREAM / "models"


def canonical_score(
    int8_path: str | Path,
    variant: str = "standard",
    hidden: int = 64,
    kernel: int = 3,
    device: str | None = None,
) -> dict[str, float]:
    """The ONE canonical evaluation function.

    Matches the official evaluate.py exactly:
    - Loads int8 checkpoint
    - Decodes the SUBMISSION archive (same as what gets scored)
    - Decodes GT video
    - Runs official PoseNet + SegNet distortion computation
    - Computes rate from submission archive size
    - Returns score = 100*seg + sqrt(10*pose) + 25*rate

    This is the single source of truth for all proxy scoring.
    """
    from .architectures import build_postfilter
    from .data import build_pairs, decode_archive, decode_video
    from .quantization import load_int8
    from .scorer import detect_device, load_scorers

    if device is None:
        device = str(detect_device())

    # Load model
    model = build_postfilter(variant, hidden=hidden, kernel=kernel)
    load_int8(int8_path, model, device=device)
    model = model.eval().to(device)

    # Load data — ALWAYS from the submission archive
    comp_frames = decode_archive(str(SUBMISSION_ARCHIVE))
    gt_frames = decode_video(str(GT_VIDEO))
    assert len(comp_frames) == len(gt_frames), (
        f"Frame count mismatch: {len(comp_frames)} comp vs {len(gt_frames)} GT"
    )

    # Load scorers
    posenet, segnet = load_scorers(
        MODELS_DIR / "posenet.safetensors",
        MODELS_DIR / "segnet.safetensors",
        device=device,
        upstream_dir=str(UPSTREAM),
    )

    # Compute rate — matches official formula exactly
    archive_size = Path(SUBMISSION_ARCHIVE).stat().st_size
    uncompressed_size = sum(f.stat().st_size for f in GT_DIR.rglob("*") if f.is_file())
    rate = archive_size / uncompressed_size

    # Build pairs and evaluate — ALL pairs, no subsampling
    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    assert len(comp_pairs) == len(gt_pairs)

    total_pose, total_seg, n_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for cp, gp in zip(comp_pairs, gt_pairs, strict=True):
            cp = cp.to(device)
            gp = gp.to(device)

            # Apply filter to both frames
            B, T, H, W, C = cp.shape
            frames = cp.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
            filtered = model(frames)
            filtered_pair = filtered.permute(0, 2, 3, 1).reshape(B, T, H, W, C)

            # Convert to scorer input format: (B, T, C, H, W)
            fx = filtered_pair.float().permute(0, 1, 4, 2, 3).contiguous()
            gx = gp.float().permute(0, 1, 4, 2, 3).contiguous()

            # PoseNet: MSE on first 6 outputs (matches upstream compute_distortion)
            fp_in = posenet.preprocess_input(fx)
            gp_in = posenet.preprocess_input(gx)
            fp_out = posenet(fp_in)
            gp_out = posenet(gp_in)
            pose_per_sample = (
                (fp_out["pose"][..., :6] - gp_out["pose"][..., :6])
                .pow(2)
                .mean(dim=tuple(range(1, fp_out["pose"].ndim)))
            )

            # SegNet: hard argmax disagreement (matches upstream compute_distortion)
            fs_in = segnet.preprocess_input(fx)
            gs_in = segnet.preprocess_input(gx)
            fs_out = segnet(fs_in)
            gs_out = segnet(gs_in)
            diff = (fs_out.argmax(dim=1) != gs_out.argmax(dim=1)).float()
            seg_per_sample = diff.mean(dim=tuple(range(1, diff.ndim)))

            total_pose += pose_per_sample.sum().item()
            total_seg += seg_per_sample.sum().item()
            n_samples += B

    avg_pose = total_pose / n_samples
    avg_seg = total_seg / n_samples
    score = 100.0 * avg_seg + math.sqrt(10.0 * avg_pose) + 25.0 * rate

    return {
        "score": score,
        "pose": avg_pose,
        "seg": avg_seg,
        "rate": rate,
        "rate_contribution": 25.0 * rate,
        "pose_contribution": math.sqrt(10.0 * avg_pose),
        "seg_contribution": 100.0 * avg_seg,
        "n_samples": n_samples,
        "archive": str(SUBMISSION_ARCHIVE),
        "checkpoint": str(int8_path),
    }


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
    """CLI: python -m tac.evaluate path/to/int8.pt"""
    path = sys.argv[1] if len(sys.argv) > 1 else str(
        REPO / "submissions" / "robust_current" / "postfilter_int8.pt"
    )
    hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    print(f"Canonical evaluation: {path}")
    print(f"  variant=standard, hidden={hidden}, kernel=3")
    print()

    result = canonical_score(path, hidden=hidden)

    print(f"{'=' * 60}")
    print(f"  CANONICAL SCORE:     {result['score']:.4f}")
    print(f"  SegNet contribution: {result['seg_contribution']:.4f}  (seg={result['seg']:.8f})")
    print(f"  PoseNet contribution:{result['pose_contribution']:.4f}  (pose={result['pose']:.8f})")
    print(f"  Rate contribution:   {result['rate_contribution']:.4f}  (rate={result['rate']:.8f})")
    print(f"  N samples:           {result['n_samples']}")
    print(f"  Archive:             {result['archive']}")
    print(f"{'=' * 60}")
