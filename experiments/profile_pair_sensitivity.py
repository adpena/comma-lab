#!/usr/bin/env python
"""Lane W: per-pair sensitivity profiler.

Loads a renderer + poses + masks (typically Lane A's anchor artifacts),
renders all 600 contest pairs, runs PoseNet + SegNet on the (gen, gt)
pairs, and records each pair's contribution to the score formula
``100 * seg_dist + sqrt(10 * pose_dist)``.

The output is a ``(N_pairs,) float32`` weight tensor where the top-K
hardest pairs (highest contribution) are weighted ``--hard-weight``
and the rest stay at ``1.0``. The downstream consumer is
``train_renderer.py --pair-loss-weights`` (Lane S Self-Compression),
which scales BOTH the scorer loss AND the SC Lagrangian gradient on
each step by ``pair_loss_weights[pair_idx_int]`` so the per-channel
learnable bit-depth allocation is steered to protect the hardest
pairs.

Memory-grounded premise:
- ``feedback_overfit_is_the_goal``: single-video contest, memorisation
  is optimal.
- ``feedback_posenet_tracking``: average distortion hides the heavy
  tail; per-pair tracking is the only correct view.
- ``project_council_shower_thoughts`` + ``feedback_curriculum_must_use_full_score``:
  hard-pair curriculum must use the FULL score formula, not PoseNet
  alone (PoseNet is solved at our operating point — SegNet dominates).

CLAUDE.md compliance:
- Default ``--device cuda`` (no MPS/CPU fallback). MPS PoseNet drift is
  23x; sensitivity ranking from MPS would be noise.
- Strict-scorer-rule: this is a COMPRESS-time profiling tool, never
  invoked at inflate. The scorer load is bounded to this script.
- All artefacts written to ``--output`` deterministically; sidecar
  JSON captures the full provenance for the downstream training run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Constants (must match upstream/evaluate.py)
NUM_FRAMES = 1200
NUM_PAIRS = NUM_FRAMES // 2  # 600 contest pairs (non-overlapping seq_len=2)
SEG_W, SEG_H = 512, 384
OUT_W, OUT_H = 1164, 874


def _ensure_paths(repo_root: Path, upstream: Path) -> None:
    """Make tac + upstream importable regardless of CWD."""
    for p in (str(repo_root / "src"), str(upstream)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _yuv420_to_rgb(frame) -> torch.Tensor:
    """BT.601 limited range YUV420 -> RGB uint8 (matches auth_eval_renderer)."""
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(
        H, frame.planes[0].line_size
    )[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(
        H // 2, frame.planes[1].line_size
    )[:, :W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(
        H // 2, frame.planes[2].line_size
    )[:, :W // 2]

    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

    u_up = F.interpolate(u_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()

    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)

    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


def _decode_video(mkv_path: Path) -> torch.Tensor:
    """Decode a YUV420 .mkv to (N, 3, H, W) uint8 tensor."""
    import av

    container = av.open(str(mkv_path))
    stream = container.streams.video[0]
    frames: list[torch.Tensor] = []
    for frame in container.decode(stream):
        rgb = _yuv420_to_rgb(frame)  # (H, W, 3) uint8
        frames.append(rgb.permute(2, 0, 1))  # (3, H, W)
    container.close()
    return torch.stack(frames, dim=0)


def _decode_masks_mkv(mkv_path: Path, device: str) -> torch.Tensor:
    """Decode an AV1-monochrome masks.mkv -> (N, SEG_H, SEG_W) long tensor.

    Mirrors submissions/robust_current/inflate_renderer.py::_load_masks_from_archive
    decoding (5-class grayscale: pixel = class * (255 // 4) = class * 63).
    """
    import av

    container = av.open(str(mkv_path))
    stream = container.streams.video[0]
    masks_list: list[torch.Tensor] = []
    for frame in container.decode(stream):
        # gray planar — first plane holds Y/luma
        H, W = frame.height, frame.width
        arr = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(
            H, frame.planes[0].line_size
        )[:, :W]
        # Round to nearest class (0..4) inverting the *63 quantization.
        m = torch.from_numpy(arr.copy()).float() / 63.0
        m = m.round().clamp(0, 4).to(torch.long)
        masks_list.append(m)
    container.close()
    masks = torch.stack(masks_list, dim=0)
    if masks.shape[-2:] != (SEG_H, SEG_W):
        # Up/down-sample to canonical SegNet resolution if encoder used a
        # smaller native size (rare; Lane A uses full SEG_H x SEG_W).
        masks = F.interpolate(
            masks.float().unsqueeze(1),
            size=(SEG_H, SEG_W),
            mode="nearest",
        ).squeeze(1).long()
    return masks


def _load_scorers(upstream: Path, device: str) -> tuple[nn.Module, nn.Module]:
    from modules import PoseNet, SegNet  # type: ignore[import-not-found]
    from safetensors.torch import load_file

    posenet = PoseNet().eval().to(device)
    segnet = SegNet().eval().to(device)
    posenet.load_state_dict(load_file(str(upstream / "models" / "posenet.safetensors"), device=device))
    segnet.load_state_dict(load_file(str(upstream / "models" / "segnet.safetensors"), device=device))
    for p in posenet.parameters():
        p.requires_grad = False
    for p in segnet.parameters():
        p.requires_grad = False
    return posenet, segnet


def _generate_pair(
    model: nn.Module,
    masks_t: torch.Tensor,
    masks_t1: torch.Tensor,
    pose: torch.Tensor | None,
    device: str,
) -> torch.Tensor:
    """Render ONE pair (B=1) -> (2, 3, OUT_H, OUT_W) uint8 tensor."""
    kwargs: dict = {}
    if pose is not None:
        kwargs["pose"] = pose
    # AsymmetricPairGenerator returns (B, 2, H, W, 3) HWC float in [0,255]
    pairs = model(masks_t, masks_t1, **kwargs)  # (1, 2, H, W, 3)
    out = []
    for frame_pos in range(2):
        frame_hwc = pairs[0, frame_pos]
        frame_chw = frame_hwc.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        frame_up = F.interpolate(
            frame_chw, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False
        )
        out.append(frame_up.round().clamp(0, 255).to(torch.uint8).squeeze(0))
    return torch.stack(out, dim=0)


def profile(
    checkpoint: Path,
    poses_path: Path | None,
    masks_mkv: Path,
    video_mkv: Path,
    upstream: Path,
    device: str,
    output: Path,
    top_k: int,
    hard_weight: float,
    batch_size: int,
) -> dict:
    """Run the per-pair sensitivity sweep and write pair_weights.pt + JSON."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required (--device cuda) — MPS PoseNet drift is 23x, "
            "MPS sensitivity ranking would be noise. See "
            "feedback_mps_cuda_drift_critical."
        )

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_paths(repo_root, upstream)

    # Late imports (after sys.path fixup)
    from tac.eval.auth_eval import AuthEvaluator  # noqa: F401 (parity check only)

    print(f"[lane-w-profile] device={device}")
    print(f"[lane-w-profile] checkpoint={checkpoint}")
    print(f"[lane-w-profile] poses={poses_path}")
    print(f"[lane-w-profile] masks_mkv={masks_mkv}")
    print(f"[lane-w-profile] video_mkv={video_mkv}")
    print(f"[lane-w-profile] top_k={top_k} hard_weight={hard_weight}")

    t_total = time.monotonic()

    # Load scorers
    t0 = time.monotonic()
    posenet, segnet = _load_scorers(upstream, device)
    print(f"[lane-w-profile] scorers loaded in {time.monotonic() - t0:.1f}s")

    # Load renderer (reuse the canonical loader path used by inflate_renderer)
    t0 = time.monotonic()
    inflate_path = repo_root / "submissions" / "robust_current"
    if str(inflate_path) not in sys.path:
        sys.path.insert(0, str(inflate_path))
    from inflate_renderer import _load_renderer  # type: ignore[import-not-found]

    model = _load_renderer(str(checkpoint), device).eval()
    has_film = bool(getattr(model, "pose_dim", 0))
    print(
        f"[lane-w-profile] renderer loaded in {time.monotonic() - t0:.1f}s "
        f"(pose_dim={getattr(model, 'pose_dim', 0)})"
    )

    # Load poses if model is FiLM-conditioned
    poses: torch.Tensor | None = None
    if has_film:
        if poses_path is None or not poses_path.exists():
            raise FileNotFoundError(
                f"FiLM renderer (pose_dim={model.pose_dim}) requires --poses; "
                f"got {poses_path}. See feedback_film_eval_no_poses_critical."
            )
        from tac.submission_archive import load_optimized_poses
        poses = load_optimized_poses(
            poses_path, pose_dim=model.pose_dim, expected_n_pairs=NUM_PAIRS
        ).to(device)
        print(f"[lane-w-profile] poses loaded: {tuple(poses.shape)}")

    # Decode GT video + masks
    t0 = time.monotonic()
    gt_frames = _decode_video(video_mkv).to(device)  # (N, 3, OUT_H, OUT_W) uint8
    print(f"[lane-w-profile] GT video decoded: {tuple(gt_frames.shape)} in {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    masks = _decode_masks_mkv(masks_mkv, device).to(device)  # (N, SEG_H, SEG_W) long
    print(f"[lane-w-profile] masks decoded: {tuple(masks.shape)} in {time.monotonic() - t0:.1f}s")

    n_frames = min(gt_frames.shape[0], masks.shape[0])
    n_frames -= n_frames % 2  # align to non-overlapping pairs
    n_pairs = n_frames // 2
    print(f"[lane-w-profile] aligned: n_frames={n_frames} n_pairs={n_pairs}")

    if n_pairs != NUM_PAIRS:
        print(
            f"[lane-w-profile] WARNING: n_pairs={n_pairs} != contest standard "
            f"{NUM_PAIRS}; weights tensor will have {n_pairs} entries."
        )

    # Per-pair distortion sweep
    pose_dists = torch.zeros(n_pairs, dtype=torch.float64)
    seg_dists = torch.zeros(n_pairs, dtype=torch.float64)

    torch.manual_seed(42)
    t0 = time.monotonic()
    with torch.inference_mode():
        for pi in range(n_pairs):
            f0, f1 = 2 * pi, 2 * pi + 1
            mask_t = masks[f0:f0 + 1]
            mask_t1 = masks[f1:f1 + 1]
            pose = poses[pi:pi + 1] if poses is not None else None

            gen_pair = _generate_pair(model, mask_t, mask_t1, pose, device)
            # gen_pair: (2, 3, OUT_H, OUT_W) uint8

            gen_t = gen_pair[0].float().unsqueeze(0)
            gen_t1 = gen_pair[1].float().unsqueeze(0)
            gt_t = gt_frames[f0].float().unsqueeze(0)
            gt_t1 = gt_frames[f1].float().unsqueeze(0)

            # PoseNet — first 6 dims, MSE
            gen_p = torch.stack([gen_t, gen_t1], dim=1)  # (1, 2, 3, H, W)
            gt_p = torch.stack([gt_t, gt_t1], dim=1)
            p_in_gen = posenet.preprocess_input(gen_p)
            p_in_gt = posenet.preprocess_input(gt_p)
            po_gen = posenet(p_in_gen)["pose"][..., :6]
            po_gt = posenet(p_in_gt)["pose"][..., :6]
            pose_dists[pi] = (po_gen - po_gt).pow(2).mean().item()

            # SegNet — argmax disagreement on LAST frame
            seg_gen = segnet.preprocess_input(gen_t1.unsqueeze(1))
            seg_gt = segnet.preprocess_input(gt_t1.unsqueeze(1))
            sl_gen = segnet(seg_gen).argmax(dim=1)
            sl_gt = segnet(seg_gt).argmax(dim=1)
            seg_dists[pi] = (sl_gen != sl_gt).float().mean().item()

            if (pi + 1) % 50 == 0:
                print(f"[lane-w-profile]   pair {pi + 1}/{n_pairs}", flush=True)
    print(f"[lane-w-profile] sweep done in {time.monotonic() - t0:.1f}s")

    # Per-pair contribution to total score (rate term cancels — same archive).
    # contrib_i = (100 * seg_i + sqrt(10 * pose_i)) — rank by this scalar.
    contrib = 100.0 * seg_dists + (10.0 * pose_dists).sqrt()

    # Build weights: default 1.0, top-K get hard_weight
    weights = torch.ones(n_pairs, dtype=torch.float32)
    k = max(0, min(int(top_k), n_pairs))
    if k > 0:
        topk_idx = torch.topk(contrib, k, largest=True).indices
        weights[topk_idx] = float(hard_weight)
        hardest_indices = sorted(topk_idx.tolist())
    else:
        hardest_indices = []

    # Persist
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, output)
    print(f"[lane-w-profile] wrote {output} (shape={tuple(weights.shape)})")

    sidecar = output.with_suffix(output.suffix + ".meta.json")
    summary = {
        "schema_version": 1,
        "lane": "W",
        "n_pairs": int(n_pairs),
        "top_k": k,
        "hard_weight": float(hard_weight),
        "device": device,
        "checkpoint": str(checkpoint),
        "poses_path": str(poses_path) if poses_path else None,
        "masks_mkv": str(masks_mkv),
        "video_mkv": str(video_mkv),
        "upstream": str(upstream),
        "hardest_pair_indices": hardest_indices,
        "stats": {
            "pose_mean": float(pose_dists.mean().item()),
            "pose_max": float(pose_dists.max().item()),
            "pose_p99": float(pose_dists.kthvalue(max(1, int(n_pairs * 0.99))).values.item()),
            "seg_mean": float(seg_dists.mean().item()),
            "seg_max": float(seg_dists.max().item()),
            "seg_p99": float(seg_dists.kthvalue(max(1, int(n_pairs * 0.99))).values.item()),
            "contrib_mean": float(contrib.mean().item()),
            "contrib_max": float(contrib.max().item()),
            "contrib_top_k_mean": float(contrib[topk_idx].mean().item()) if k > 0 else 0.0,
            "weight_floor": 1.0,
            "weight_ceiling": float(hard_weight),
        },
        "per_pair_pose_dist": pose_dists.tolist(),
        "per_pair_seg_dist": seg_dists.tolist(),
        "per_pair_contrib": contrib.tolist(),
        "elapsed_s": time.monotonic() - t_total,
    }
    sidecar.write_text(json.dumps(summary, indent=2))
    print(f"[lane-w-profile] wrote sidecar {sidecar}")
    print(
        f"[lane-w-profile] DONE  total={time.monotonic() - t_total:.1f}s  "
        f"contrib_mean={summary['stats']['contrib_mean']:.4f}  "
        f"top{k}_mean={summary['stats']['contrib_top_k_mean']:.4f}"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Lane W per-pair sensitivity profiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Renderer checkpoint (.bin or .pt) — typically Lane A's renderer.bin")
    p.add_argument("--poses", type=Path, default=None,
                   help="Optimized poses.pt (REQUIRED if renderer has FiLM)")
    p.add_argument("--masks-mkv", type=Path, required=True,
                   help="Pre-encoded masks.mkv (Lane A's archive masks)")
    p.add_argument("--video-mkv", type=Path, required=True,
                   help="GT video (upstream/videos/0.mkv)")
    p.add_argument("--upstream", type=Path, default=Path("upstream"),
                   help="Upstream repo root (modules.py, models/, videos/)")
    p.add_argument("--output", type=Path,
                   default=Path("experiments/results/lane_w_profile/pair_weights.pt"),
                   help="Output path for the (N_pairs,) float32 weights tensor")
    p.add_argument("--top-k", type=int, default=30,
                   help="Number of hardest pairs to up-weight")
    p.add_argument("--hard-weight", type=float, default=5.0,
                   help="Weight applied to the top-K hardest pairs (others = 1.0)")
    p.add_argument("--device", default="cuda",
                   help="Torch device (CUDA REQUIRED — MPS drift is 23x)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Pair batch size (currently unused; reserved for future)")
    args = p.parse_args(argv)

    profile(
        checkpoint=args.checkpoint.resolve(),
        poses_path=args.poses.resolve() if args.poses else None,
        masks_mkv=args.masks_mkv.resolve(),
        video_mkv=args.video_mkv.resolve(),
        upstream=args.upstream.resolve(),
        device=args.device,
        output=args.output.resolve(),
        top_k=int(args.top_k),
        hard_weight=float(args.hard_weight),
        batch_size=int(args.batch_size),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
