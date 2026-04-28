"""Lane SI: profile PoseNet + SegNet saliency on Lane A's rendered frames.

Outputs a torch.save() dict:
    {
        "posenet_saliency": (H, W) float32,  # gradient norm per pixel
        "segnet_saliency":  (H, W) float32,
        "combined":         (H, W) float32,  # normalised sum of both
    }

The maps live on the camera grid (874x1164) so they align with mask
frames produced by mask_codec / inflate_renderer.

CRITICAL CLAUDE.md rules respected:
- Scorer access is COMPRESS-TIME ONLY (this script runs at compress-time).
- --device cuda is REQUIRED. No MPS / CPU fallback.
- Real per-pair pose loss + per-pixel softmax-entropy proxy for SegNet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Repo plumbing: the upstream package lives under upstream/ and ships
# `frame_utils`, `modules` as top-level imports.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "upstream"))
sys.path.insert(0, str(REPO / "src"))

from tac.saliency_inversion import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    compute_pixel_saliency,
)


def _require_cuda(device: str) -> torch.device:
    """Block MPS/CPU. CLAUDE.md non-negotiable: scorer measurements on CUDA only."""
    if device != "cuda":
        raise SystemExit(
            f"FATAL: --device must be 'cuda' (got {device!r}). MPS auth eval "
            "is NOISE (23x PoseNet drift verified 2026-04-25). See "
            "feedback_mps_cuda_drift_critical."
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: CUDA unavailable. Lane SI saliency profiling requires CUDA "
            "(scorer numerics differ across backends; MPS would invalidate the map)."
        )
    return torch.device("cuda")


def _load_scorers(upstream_dir: Path, device: torch.device):
    from safetensors.torch import load_file
    from modules import PoseNet, SegNet  # type: ignore[import-not-found]

    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(
        load_file(str(upstream_dir / "models" / "posenet.safetensors"), device=str(device))
    )
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(
        load_file(str(upstream_dir / "models" / "segnet.safetensors"), device=str(device))
    )
    return posenet, segnet


def _decode_video(video_path: Path, max_frames: int | None = None) -> torch.Tensor:
    """Decode a .mkv into ``(N, H, W, 3)`` uint8 RGB frames via PyAV.

    Returns the camera-resolution frames (H=874, W=1164). The caller is
    responsible for batching into pairs.
    """
    import av  # lazy: keeps import-time cheap for tests
    from frame_utils import yuv420_to_rgb  # type: ignore[import-not-found]

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames: list[torch.Tensor] = []
    for i, frame in enumerate(container.decode(stream)):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(yuv420_to_rgb(frame))
    container.close()
    return torch.stack(frames, dim=0)


def _build_pairs(frames: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """Convert ``(N, H, W, 3)`` uint8 → ``(P, 2, 3, H, W)`` float32 pairs.

    Uses non-overlapping pairs (matching upstream evaluate.py: seq_len=2).
    """
    n = frames.shape[0] // 2
    n = min(n, n_pairs)
    if n == 0:
        raise SystemExit("FATAL: video has < 2 frames; cannot form pairs.")
    pairs = []
    for k in range(n):
        a = frames[2 * k]
        b = frames[2 * k + 1]
        # (H, W, 3) → (3, H, W)
        pair = torch.stack([a, b], dim=0).permute(0, 3, 1, 2).float()
        pairs.append(pair)
    return torch.stack(pairs, dim=0)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Profile PoseNet + SegNet pixel saliency on a video."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Lane A renderer.bin (used only as provenance for the map)",
    )
    p.add_argument(
        "--poses",
        type=Path,
        required=True,
        help="Lane A optimized_poses.pt (provenance)",
    )
    p.add_argument(
        "--masks-mkv",
        type=Path,
        required=True,
        help="Lane A masks.mkv (provenance — saliency map will be aligned to this grid)",
    )
    p.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Source video (.mkv) to compute saliency over",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pt path: dict{posenet_saliency, segnet_saliency, combined}",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help="Device — CUDA required (MPS / CPU forbidden, see CLAUDE.md)",
    )
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=REPO / "upstream",
        help="upstream/ directory containing models/{posenet,segnet}.safetensors",
    )
    p.add_argument(
        "--n-pairs",
        type=int,
        default=64,
        help="Pairs to sample for saliency (more = smoother but slower)",
    )
    p.add_argument(
        "--reduce",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Per-pixel reduction across pairs",
    )

    args = p.parse_args()

    # Strict input validation (FAIL LOUD, not silently default)
    for name, path in (
        ("checkpoint", args.checkpoint),
        ("poses", args.poses),
        ("masks-mkv", args.masks_mkv),
        ("video", args.video),
    ):
        if not path.is_file():
            raise SystemExit(f"FATAL: --{name} file not found: {path}")

    device = _require_cuda(args.device)
    print(f"[lane-si] device={device} torch={torch.__version__}", flush=True)

    t0 = time.time()
    posenet, segnet = _load_scorers(args.upstream_dir, device)
    print(f"[lane-si] scorers loaded in {time.time() - t0:.2f}s", flush=True)

    t0 = time.time()
    raw = _decode_video(args.video, max_frames=2 * args.n_pairs)
    print(
        f"[lane-si] decoded {raw.shape[0]} frames "
        f"({raw.shape[1]}x{raw.shape[2]}) in {time.time() - t0:.2f}s",
        flush=True,
    )
    if raw.shape[1:3] != (CAMERA_H, CAMERA_W):
        raise SystemExit(
            f"FATAL: video frame size {tuple(raw.shape[1:3])} != camera grid "
            f"({CAMERA_H}, {CAMERA_W})."
        )

    pairs = _build_pairs(raw, args.n_pairs)
    print(f"[lane-si] formed {pairs.shape[0]} non-overlapping pairs", flush=True)

    t0 = time.time()
    pose_sal = compute_pixel_saliency(
        posenet,
        pairs,
        output_shape=(CAMERA_H, CAMERA_W),
        reduce=args.reduce,
    )
    print(
        f"[lane-si] PoseNet saliency: shape={tuple(pose_sal.shape)} "
        f"min={pose_sal.min().item():.3e} max={pose_sal.max().item():.3e} "
        f"in {time.time() - t0:.2f}s",
        flush=True,
    )

    t0 = time.time()
    seg_sal = compute_pixel_saliency(
        segnet,
        pairs,
        output_shape=(CAMERA_H, CAMERA_W),
        reduce=args.reduce,
    )
    print(
        f"[lane-si] SegNet saliency: shape={tuple(seg_sal.shape)} "
        f"min={seg_sal.min().item():.3e} max={seg_sal.max().item():.3e} "
        f"in {time.time() - t0:.2f}s",
        flush=True,
    )

    # Combined map: normalise each to [0, 1] then sum.
    def _norm01(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        rng = (x.max() - x.min()).clamp_min(1e-12)
        return (x - x.min()) / rng

    combined = _norm01(pose_sal) + _norm01(seg_sal)
    combined = _norm01(combined)

    out = {
        "posenet_saliency": pose_sal.detach().cpu().float(),
        "segnet_saliency": seg_sal.detach().cpu().float(),
        "combined": combined.detach().cpu().float(),
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "poses": str(args.poses),
            "masks_mkv": str(args.masks_mkv),
            "video": str(args.video),
            "n_pairs": int(pairs.shape[0]),
            "reduce": args.reduce,
            "torch_version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(
        f"[lane-si] saved {args.output} "
        f"({args.output.stat().st_size} bytes)",
        flush=True,
    )
    print(json.dumps({"output": str(args.output), "n_pairs": int(pairs.shape[0])}))


if __name__ == "__main__":
    main()
