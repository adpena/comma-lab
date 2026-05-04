#!/usr/bin/env python3
"""Prepare renderer-output and GT-pair tensors for SJ-KL residual building.

This is a build-only utility. It does not load scorers, does not dispatch GPU
jobs, and does not make score claims. It converts a rendered frame tensor plus
the official GT video into the exact tensor contract consumed by
``experiments/build_sjkl_residual.py``:

``renderer_target_slot_chw.pt``: ``(P, 3, H, W)`` float32
``gt_pairs_btchw.pt``: ``(P, 2, 3, H, W)`` uint8

The current robust_current SJ-KL runtime corrects JointFrameGenerator pair slot
0 only, so the production target slot defaults to 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from tac.data import decode_video


SCHEMA = "sjkl_pair_tensor_prep_v1"
TOOL = "experiments/prepare_sjkl_pair_tensors.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def normalize_renderer_target_slot(
    frames: torch.Tensor,
    *,
    target_slot: int = 0,
    max_pairs: int | None = None,
) -> torch.Tensor:
    """Return renderer target frames as ``(P, 3, H, W)`` float32."""
    if target_slot != 0:
        raise ValueError("current robust_current SJ-KL runtime supports only target_slot=0")
    if frames.ndim != 4:
        raise ValueError(f"renderer frames must be rank-4, got {tuple(frames.shape)}")

    if frames.shape[-1] == 3:
        # Full frame stream: (N,H,W,3). Select pair slot then CHW.
        n_frames = int(frames.shape[0])
        if n_frames % 2 != 0:
            raise ValueError(f"renderer frame count must be even, got {n_frames}")
        selected = frames[target_slot::2]
        if max_pairs is not None:
            if max_pairs <= 0:
                raise ValueError(f"max_pairs must be positive, got {max_pairs}")
            selected = selected[:max_pairs]
        out = selected.permute(0, 3, 1, 2).contiguous()
    elif frames.shape[1] == 3:
        # Already target-slot frames: (P,3,H,W).
        selected = frames
        if max_pairs is not None:
            if max_pairs <= 0:
                raise ValueError(f"max_pairs must be positive, got {max_pairs}")
            selected = selected[:max_pairs]
        out = selected.contiguous()
    else:
        raise ValueError(
            "renderer frames must be either (N,H,W,3) or (P,3,H,W); "
            f"got {tuple(frames.shape)}"
        )

    return out.to(dtype=torch.float32, copy=False).cpu().contiguous()


def build_gt_pairs_btchw(frames_hwc: list[torch.Tensor], *, n_pairs: int) -> torch.Tensor:
    """Return GT pairs as ``(P,2,3,H,W)`` uint8."""
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")
    needed = n_pairs * 2
    if len(frames_hwc) < needed:
        raise ValueError(f"GT frames too short: need {needed}, got {len(frames_hwc)}")
    pairs: list[torch.Tensor] = []
    for pair_idx in range(n_pairs):
        f0 = frames_hwc[2 * pair_idx]
        f1 = frames_hwc[2 * pair_idx + 1]
        if f0.ndim != 3 or f0.shape[-1] != 3 or f1.shape != f0.shape:
            raise ValueError("GT frames must be matching (H,W,3) RGB tensors")
        pair = torch.stack([f0, f1], dim=0).permute(0, 3, 1, 2).contiguous()
        pairs.append(pair)
    return torch.stack(pairs, dim=0).to(dtype=torch.uint8, copy=False).cpu().contiguous()


def write_manifest(
    *,
    output_dir: Path,
    renderer_source: Path,
    gt_video: Path,
    renderer_tensor_path: Path,
    gt_pairs_path: Path,
    target_slot: int,
    renderer_tensor: torch.Tensor,
    gt_pairs: torch.Tensor,
) -> dict[str, Any]:
    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "build_tensor_prep_only",
        "target_slot": int(target_slot),
        "runtime_target": "robust_current JointFrameGenerator pair slot 0 / fake1",
        "inputs": {
            "renderer_source": {
                "path": str(renderer_source),
                "bytes": int(renderer_source.stat().st_size),
                "sha256": _sha256_file(renderer_source),
            },
            "gt_video": {
                "path": str(gt_video),
                "bytes": int(gt_video.stat().st_size),
                "sha256": _sha256_file(gt_video),
            },
        },
        "outputs": {
            "renderer_target_slot_chw": {
                "path": str(renderer_tensor_path),
                "bytes": int(renderer_tensor_path.stat().st_size),
                "sha256": _sha256_file(renderer_tensor_path),
                "shape": list(renderer_tensor.shape),
                "dtype": str(renderer_tensor.dtype),
            },
            "gt_pairs_btchw": {
                "path": str(gt_pairs_path),
                "bytes": int(gt_pairs_path.stat().st_size),
                "sha256": _sha256_file(gt_pairs_path),
                "shape": list(gt_pairs.shape),
                "dtype": str(gt_pairs.dtype),
            },
        },
        "next_command_template": (
            "PYTHONPATH=src:upstream .venv/bin/python experiments/build_sjkl_residual.py "
            f"--renderer-output {renderer_tensor_path} --gt-pairs {gt_pairs_path} "
            f"--target-slot {target_slot} --out {output_dir / 'sjkl.bin'} "
            f"--manifest {output_dir / 'sjkl_manifest.json'} --device cuda"
        ),
    }
    (output_dir / "sjkl_pair_tensor_prep_manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer-frames", type=Path, required=True)
    parser.add_argument("--gt-video", type=Path, default=Path("upstream/videos/0.mkv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-slot", type=int, default=0)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"output directory is non-empty; pass --force: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    renderer_frames = torch.load(args.renderer_frames, map_location="cpu", weights_only=True)
    renderer_target = normalize_renderer_target_slot(
        renderer_frames,
        target_slot=args.target_slot,
        max_pairs=args.max_pairs,
    )
    n_pairs = int(renderer_target.shape[0])
    _p, _c, h, w = renderer_target.shape

    gt_frames = decode_video(args.gt_video, target_h=h, target_w=w, max_frames=n_pairs * 2)
    gt_pairs = build_gt_pairs_btchw(gt_frames, n_pairs=n_pairs)

    renderer_out = args.output_dir / "renderer_target_slot_chw.pt"
    gt_out = args.output_dir / "gt_pairs_btchw.pt"
    torch.save(renderer_target, renderer_out)
    torch.save(gt_pairs, gt_out)
    manifest = write_manifest(
        output_dir=args.output_dir,
        renderer_source=args.renderer_frames,
        gt_video=args.gt_video,
        renderer_tensor_path=renderer_out,
        gt_pairs_path=gt_out,
        target_slot=args.target_slot,
        renderer_tensor=renderer_target,
        gt_pairs=gt_pairs,
    )
    print(json.dumps({
        "schema": manifest["schema"],
        "output_dir": str(args.output_dir),
        "n_pairs": n_pairs,
        "renderer_target_bytes": manifest["outputs"]["renderer_target_slot_chw"]["bytes"],
        "gt_pairs_bytes": manifest["outputs"]["gt_pairs_btchw"]["bytes"],
        "score_claim": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
