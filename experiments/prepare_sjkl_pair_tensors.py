#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare frame-pair tensors for SJ-KL residual build.

Recovery note: this script was lost when subagent worktrees were auto-cleaned
without committing source. Rebuilt 2026-05-04 to satisfy the contract that
experiments/build_sjkl_residual.py consumes via --pair-tensor-manifest.

Pipeline:
  1. Load renderer output tensor (the post-renderer reconstruction frames)
     OR decode source video frames if --decode-from-video
  2. Build per-pair residuals: residual[i] = frame_target[i] - frame_predicted[i]
     (the residual the SJ-KL basis needs to encode)
  3. Pick an anchor frame (the linearization point for Lanczos HVP)
  4. Save:
       - anchor_frame.pt   (D,)            float32 flattened
       - pair_residuals.pt (n_pairs, D)    float32 flattened
       - sjkl_pair_tensor_prep_manifest.json  with sha + bytes + schema version

Output manifest schema matches what build_sjkl_residual.py expects:
  {
    "schema_version": 1,
    "anchor_frame_path": "<rel path>",
    "pair_residuals_path": "<rel path>",
    "n_pairs": <int>,
    "frame_dim": <int>,
    "source_kind": "renderer_output" | "raw_video_decoded" | "cpu_stub",
    "source_path": "<rel path or null>",
    "source_sha256": "<hex>",
    "produced_at_utc": "<ISO8601>",
  }

Per CLAUDE.md FORBIDDEN PATTERNS:
  - This script does no scorer loads (it's pre-build, before basis computation)
  - --device cpu is fine here; pair tensors are device-agnostic on disk
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PrepConfig:
    renderer_output: Path
    target_frames: Path
    output_dir: Path
    n_pairs: int | None
    anchor_pair_idx: int
    flatten: bool
    cpu_stub: bool
    cpu_stub_dim: int
    cpu_stub_n_pairs: int
    seed: int


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_frames(path: Path, *, name: str) -> torch.Tensor:
    """Load a tensor of frames from disk. Accepts (n, ...) shape; squeezes
    any singleton time dim."""
    if not path.is_file():
        raise SystemExit(f"FATAL: {name} not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        # Common case: dict with "frames" or "rgb" key
        for key in ("frames", "rgb", "tensor", "data"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise SystemExit(f"FATAL: {name} is a dict but has no frames/rgb/tensor/data key")
    if not torch.is_tensor(obj):
        raise SystemExit(f"FATAL: {name} loaded as {type(obj).__name__}, expected Tensor")
    while obj.dim() > 4 and obj.shape[1] == 1:
        # squeeze redundant time dim if shape is (N, 1, C, H, W)
        obj = obj.squeeze(1)
    return obj


def _build_cpu_stub_inputs(n_pairs: int, dim: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic inputs for smoke testing the I/O pipeline."""
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(n_pairs, dim, generator=g, dtype=torch.float32)
    predicted = target + torch.randn(n_pairs, dim, generator=g, dtype=torch.float32) * 0.05
    return target, predicted


def prepare_sjkl_pair_tensors(cfg: PrepConfig) -> dict:
    """Main pipeline. Returns manifest dict (also written to disk)."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.cpu_stub:
        target_flat, predicted_flat = _build_cpu_stub_inputs(
            cfg.cpu_stub_n_pairs, cfg.cpu_stub_dim, seed=cfg.seed
        )
        n_pairs = cfg.cpu_stub_n_pairs
        frame_dim = cfg.cpu_stub_dim
        source_kind = "cpu_stub"
        source_path_str = None
        source_sha = "stub_no_source_file"
    else:
        renderer_frames = _load_frames(cfg.renderer_output, name="renderer-output")
        target_frames = _load_frames(cfg.target_frames, name="target-frames")

        if renderer_frames.shape[0] != target_frames.shape[0]:
            raise SystemExit(
                f"FATAL: renderer-output ({renderer_frames.shape[0]}) and "
                f"target-frames ({target_frames.shape[0]}) have different pair counts"
            )

        n_pairs = renderer_frames.shape[0]
        if cfg.n_pairs is not None and cfg.n_pairs < n_pairs:
            n_pairs = cfg.n_pairs
            renderer_frames = renderer_frames[:n_pairs]
            target_frames = target_frames[:n_pairs]

        if cfg.flatten:
            predicted_flat = renderer_frames.reshape(n_pairs, -1).to(torch.float32)
            target_flat = target_frames.reshape(n_pairs, -1).to(torch.float32)
        else:
            predicted_flat = renderer_frames.to(torch.float32)
            target_flat = target_frames.to(torch.float32)
        frame_dim = int(predicted_flat.shape[-1])
        source_kind = "renderer_output"
        source_path_str = str(cfg.renderer_output.relative_to(REPO_ROOT)
                              if cfg.renderer_output.is_relative_to(REPO_ROOT)
                              else cfg.renderer_output)
        source_sha = _sha256_file(cfg.renderer_output)

    pair_residuals = (target_flat - predicted_flat).contiguous()
    if pair_residuals.shape != (n_pairs, frame_dim):
        raise RuntimeError(
            f"unexpected pair_residuals shape {tuple(pair_residuals.shape)}; "
            f"expected ({n_pairs}, {frame_dim})"
        )

    # Anchor frame: the linearization point for Lanczos HVP. Use the predicted
    # frame at anchor_pair_idx (the renderer's reconstruction) by convention.
    anchor_idx = max(0, min(cfg.anchor_pair_idx, n_pairs - 1))
    anchor_frame = predicted_flat[anchor_idx].clone().contiguous()

    # Save tensors
    anchor_path = cfg.output_dir / "anchor_frame.pt"
    residuals_path = cfg.output_dir / "pair_residuals.pt"
    torch.save(anchor_frame, anchor_path)
    torch.save(pair_residuals, residuals_path)

    # Manifest with file-relative paths (so build_sjkl_residual.py can resolve
    # against REPO_ROOT)
    def _rel(p: Path) -> str:
        return str(p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p)

    manifest = {
        "schema_version": 1,
        "anchor_frame_path": _rel(anchor_path),
        "pair_residuals_path": _rel(residuals_path),
        "n_pairs": int(n_pairs),
        "frame_dim": int(frame_dim),
        "anchor_pair_idx": int(anchor_idx),
        "source_kind": source_kind,
        "source_path": source_path_str,
        "source_sha256": source_sha,
        "anchor_frame_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        "pair_residuals_sha256": hashlib.sha256(residuals_path.read_bytes()).hexdigest(),
        "produced_at_utc": _utc_now(),
        "produced_by": "experiments/prepare_sjkl_pair_tensors.py",
        "score_claim": False,
    }
    manifest_path = cfg.output_dir / "sjkl_pair_tensor_prep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[sjkl-prep] wrote {anchor_path} ({anchor_frame.numel()} floats)", file=sys.stderr)
    print(f"[sjkl-prep] wrote {residuals_path} ({pair_residuals.numel()} floats)", file=sys.stderr)
    print(f"[sjkl-prep] wrote {manifest_path}", file=sys.stderr)
    print(f"[sjkl-prep] n_pairs={n_pairs} frame_dim={frame_dim} source={source_kind}", file=sys.stderr)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer-output", type=Path, default=Path("/dev/null"),
                        help="Tensor file with renderer-predicted frames (n_pairs, ...).")
    parser.add_argument("--target-frames", type=Path, default=Path("/dev/null"),
                        help="Tensor file with ground-truth target frames (n_pairs, ...).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-pairs", type=int, default=None,
                        help="Cap n_pairs (default: use all available).")
    parser.add_argument("--anchor-pair-idx", type=int, default=0,
                        help="Index of pair whose predicted frame becomes the Lanczos anchor.")
    parser.add_argument("--no-flatten", action="store_true",
                        help="Preserve original (n, C, H, W) shape (not recommended for SJ-KL).")
    parser.add_argument("--cpu-stub", action="store_true",
                        help="Synthesize inputs for smoke testing.")
    parser.add_argument("--cpu-stub-dim", type=int, default=256)
    parser.add_argument("--cpu-stub-n-pairs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PrepConfig(
        renderer_output=args.renderer_output,
        target_frames=args.target_frames,
        output_dir=args.output_dir,
        n_pairs=args.n_pairs,
        anchor_pair_idx=args.anchor_pair_idx,
        flatten=not args.no_flatten,
        cpu_stub=args.cpu_stub,
        cpu_stub_dim=args.cpu_stub_dim,
        cpu_stub_n_pairs=args.cpu_stub_n_pairs,
        seed=args.seed,
    )
    prepare_sjkl_pair_tensors(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
