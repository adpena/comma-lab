#!/usr/bin/env python
"""Lane PA helper: initialize SegMap.frame_affine_embedding from PoseNet output.

EUREKA #1 (grand council session, 2026-04-29):
    PoseNet outputs 6 numbers per frame pair. SegMap consumes a 6-DOF affine
    embedding per frame. SAME 6 NUMBERS, different hats. Instead of (a) running
    expensive pose-space TTO over the renderer or (b) initialising the affine
    embedding to N(0, 0.01) and hoping training discovers the right pose, we
    BAKE the frozen-PoseNet outputs directly into the embedding via a small
    linear map fit on a calibration subset.

Pipeline:
    1. Load anchor masks + GT video pairs (600 non-overlapping pairs).
    2. Run frozen PoseNet on each pair -> (600, 6) pose vectors (the first
       6 dims; PoseNet outputs 12-dim pose head, scorer reads first 6).
    3. Build a FRESH SegMap with the standard hyperparams.
    4. Construct the per-frame affine target by replicating each pose vector
       across the pair (frames 2*i and 2*i+1 share pose i).
    5. Fit a 6x6 affine map A such that A @ pose ~= "embedding raw value":
       this inverts SegMap's tanh + scaling by computing
           raw = arctanh( clamp((scaled - 0) / max_delta, -0.99, 0.99) )
       so the SegMap's forward (which applies tanh + scale) reproduces the
       desired displacement.
    6. Save a fresh SegMap checkpoint with the seeded affine_embedding.
       Saved alongside as `segmap_inference.pt` so the lane shell can pack it
       with block_fp_codec like the standard SegMap path.

CLAUDE.md compliance:
    * STRICT-SCORER-RULE: PoseNet runs at COMPRESS time only (not inflate).
    * eval_roundtrip simulation NOT needed here (we are computing GT poses,
      not training; the SegMap that consumes the seeded embedding goes
      through the canonical SegMapTrainer roundtrip in the lane shell).
    * --device cuda required (PoseNet on MPS drifts 23x).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "upstream") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "upstream"))


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        print(
            "[init-segmap-from-posenet] WARNING: --device cpu — PoseNet output "
            "will DIFFER from contest-CUDA (MPS drift was 23x; CPU is unverified). "
            "Use only for smoke tests.",
            file=sys.stderr,
        )
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: --device cuda requested but torch.cuda.is_available() is False."
        )
    return torch.device("cuda")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-video", type=str, default="upstream/videos/0.mkv")
    p.add_argument("--upstream", type=str, default="upstream")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--block-hidden", type=int, default=24)
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--max-frame-index", type=int, default=1200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--pose-scale",
        type=float,
        default=1.0,
        help="Multiplier on PoseNet output before fitting; tune if PoseNet "
             "output range exceeds the affine_max_* clamps.",
    )
    return p.parse_args()


def _load_gt_pairs(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    """Load (n_pairs, 2, H, W, 3) uint8 GT pair tensor for PoseNet input."""
    from tac.data import load_gt_video

    n_frames = args.max_frame_index
    gt_frames = load_gt_video(Path(args.gt_video), n_frames=n_frames)
    if isinstance(gt_frames, list):
        gt_tensor = torch.stack(gt_frames, dim=0)
    else:
        gt_tensor = gt_frames
    if gt_tensor.ndim != 4:
        raise RuntimeError(
            f"GT video tensor ndim={gt_tensor.ndim} (expected 4). "
            f"shape={tuple(gt_tensor.shape)}"
        )
    # gt_tensor is (N, C, H, W) or (N, H, W, C). The contest convention
    # (PoseNet.preprocess_input) expects (B, T, C, H, W) - so we permute to
    # (N, C, H, W) if needed and reshape into (N//2, 2, C, H, W).
    if gt_tensor.shape[-1] == 3 and gt_tensor.shape[-3] != 3:
        # HWC -> CHW
        gt_tensor = gt_tensor.permute(0, 3, 1, 2).contiguous()
    n = gt_tensor.shape[0]
    if n % 2 != 0:
        raise RuntimeError(f"Frame count {n} is odd; cannot form non-overlapping pairs.")
    half = n // 2
    return gt_tensor.view(half, 2, *gt_tensor.shape[-3:])


def _run_posenet(gt_pairs: torch.Tensor, posenet, device: torch.device,
                 batch: int = 8) -> torch.Tensor:
    """Forward all GT pairs through frozen PoseNet -> (n_pairs, 6) pose tensor."""
    n_pairs = gt_pairs.shape[0]
    out = torch.zeros(n_pairs, 6, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, n_pairs, batch):
            end = min(start + batch, n_pairs)
            chunk = gt_pairs[start:end].to(device, dtype=torch.float32)
            # PoseNet expects (B, T, C, H, W); preprocess_input handles the
            # rgb_to_yuv6 + interp + reshape internally.
            posenet_in = posenet.preprocess_input(chunk)
            head_out = posenet(posenet_in)
            # `pose` head is 12-dim; first 6 is what compute_distortion uses.
            pose6 = head_out["pose"][..., :6].detach().cpu().float()
            out[start:end] = pose6
    return out


def _seed_affine_embedding(
    seg_model,  # SegMap
    poses_per_pair: torch.Tensor,  # (n_pairs, 6)
    pose_scale: float,
) -> dict:
    """Bake the per-pair pose into seg_model.frame_affine_embedding.weight.

    SegMap stores a (max_frame_index, 6) embedding. The forward path in
    `_build_affine_latent_channel` applies::

        zoom    = 1.0 + max_zoom_delta * tanh(emb[:, 0])
        aspect  = max_aspect_delta * tanh(emb[:, 1])
        shear_x = max_shear * tanh(emb[:, 2])
        shear_y = max_shear * tanh(emb[:, 3])
        trans_x = max_translation * tanh(emb[:, 4])
        trans_y = max_translation * tanh(emb[:, 5])

    So to seed an embedding that produces a given (zoom_delta, aspect_delta,
    shear_x, shear_y, trans_x, trans_y) target, we INVERT the tanh + scale:

        emb[:, k] = arctanh( clamp(target[k] / max_k, -0.99, 0.99) )

    We use the PoseNet pose vector directly as the (zoom_delta, ..., trans_y)
    target after a uniform `pose_scale` multiplier — the empirical observation
    from the council session is that PoseNet's first 6 outputs lie in roughly
    the same range as the SegMap affine deltas (~[-0.1, +0.1]), so a
    pose_scale=1.0 is the right starting point.
    """
    n_pairs = poses_per_pair.shape[0]
    max_frame_index = seg_model.frame_affine_embedding.weight.shape[0]
    expected_pairs = max_frame_index // 2
    if n_pairs != expected_pairs:
        raise ValueError(
            f"poses_per_pair has {n_pairs} rows but SegMap expects "
            f"{expected_pairs} pairs (max_frame_index={max_frame_index})."
        )

    # The 6 max-* clamps from SegMap.__init__.
    clamps = torch.tensor(
        [
            seg_model.max_zoom_delta,
            seg_model.max_aspect_delta,
            seg_model.max_shear,
            seg_model.max_shear,
            seg_model.max_translation,
            seg_model.max_translation,
        ],
        dtype=torch.float32,
    )
    # PoseNet outputs are in scorer-native units; the council EUREKA bets that
    # a uniform per-channel rescaling places them inside the affine clamps.
    # Pose-scale defaults to 1.0; operators can override if the empirical pose
    # range overflows.
    scaled = poses_per_pair * pose_scale  # (n_pairs, 6)
    # Per-channel rescale: divide each column by its clamp so the result is
    # in the [-1, +1] tanh domain.
    targets_in_tanh_domain = scaled / clamps.unsqueeze(0)
    # Clamp to a safe range and invert via arctanh.
    targets_in_tanh_domain = torch.clamp(targets_in_tanh_domain, -0.99, 0.99)
    raw_per_pair = torch.atanh(targets_in_tanh_domain)  # (n_pairs, 6)
    # Replicate across both frames in the pair (frames 2*i and 2*i+1 share pose i).
    raw_per_frame = raw_per_pair.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 6)
    # raw_per_frame: (2*n_pairs, 6). Pad to max_frame_index with zeros if needed.
    out = torch.zeros(max_frame_index, 6, dtype=torch.float32)
    out[: raw_per_frame.shape[0]] = raw_per_frame

    with torch.no_grad():
        seg_model.frame_affine_embedding.weight.copy_(out)

    # Diagnostic stats: how saturated is the seeded embedding? An entry near
    # ±5 means we hit the arctanh asymptote (target was outside the clamp).
    saturated = (raw_per_frame.abs() > 3.0).float().mean().item()
    return {
        "n_pairs": int(n_pairs),
        "n_frames_seeded": int(raw_per_frame.shape[0]),
        "saturated_fraction": saturated,
        "raw_abs_max": float(raw_per_frame.abs().max().item()),
        "raw_abs_mean": float(raw_per_frame.abs().mean().item()),
    }


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)

    from tac.scorer import load_differentiable_scorers
    from tac.segmap_renderer import SegMap

    posenet, _segnet = load_differentiable_scorers(args.upstream, device=device)
    posenet.eval()
    for prm in posenet.parameters():
        prm.requires_grad_(False)

    print("[init-segmap-from-posenet] loading GT pairs...", flush=True)
    gt_pairs = _load_gt_pairs(args, device)
    print(f"[init-segmap-from-posenet] GT pairs shape: {tuple(gt_pairs.shape)}",
          flush=True)

    print("[init-segmap-from-posenet] running frozen PoseNet on all pairs...",
          flush=True)
    poses_per_pair = _run_posenet(gt_pairs, posenet, device)
    print(
        f"[init-segmap-from-posenet] PoseNet -> {tuple(poses_per_pair.shape)} "
        f"(min={poses_per_pair.min().item():.3f}, "
        f"max={poses_per_pair.max().item():.3f}, "
        f"mean={poses_per_pair.mean().item():.3f})",
        flush=True,
    )

    seg_model = SegMap(
        hidden=args.hidden,
        block_hidden=args.block_hidden,
        num_blocks=args.num_blocks,
        max_frame_index=args.max_frame_index,
    )
    diagnostics = _seed_affine_embedding(seg_model, poses_per_pair, args.pose_scale)
    print(
        f"[init-segmap-from-posenet] seeded embedding: "
        f"saturated={diagnostics['saturated_fraction']:.3%}, "
        f"raw_abs_max={diagnostics['raw_abs_max']:.3f}, "
        f"raw_abs_mean={diagnostics['raw_abs_mean']:.3f}",
        flush=True,
    )

    # Save a fresh inference state dict so the lane shell can pack via
    # block_fp_codec.pack_payload_tar_xz like the standard SegMap path.
    inference_state = {k: v.detach().cpu().clone() for k, v in seg_model.state_dict().items()}
    inference_path = output_dir / "segmap_inference.pt"
    torch.save(inference_state, inference_path)

    # Also save the raw poses so the lane shell can use them as `optimized_poses.pt`
    # if desired (or the operator can drop them — the embedding already encodes
    # the pose information).
    poses_path = output_dir / "posenet_poses.pt"
    torch.save(poses_per_pair, poses_path)

    summary = {
        "pose_diagnostics": diagnostics,
        "n_pose_pairs": int(poses_per_pair.shape[0]),
        "pose_dim": int(poses_per_pair.shape[1]),
        "pose_scale": float(args.pose_scale),
        "device": str(device),
        "hidden": int(args.hidden),
        "block_hidden": int(args.block_hidden),
        "num_blocks": int(args.num_blocks),
        "max_frame_index": int(args.max_frame_index),
    }
    (output_dir / "init_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[init-segmap-from-posenet] wrote {inference_path}, {poses_path}, init_summary.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
