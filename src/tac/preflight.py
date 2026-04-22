"""Preflight pipeline validator — catches integration mismatches before GPU burns.

Every bug in this project was at a boundary between components:
  - Masks at wrong resolution (48x64 vs 384x512 → score 103 vs 2)
  - Poses optimized against wrong masks (fresh vs CRF50 → 27x PoseNet regression)
  - Archive missing artifacts (119KB vs 338KB → 0.108 rate error)
  - eval_roundtrip defaulting False (proxy-auth drift 11x)
  - FP4 without QAT (26x PoseNet degradation)

This module validates the ENTIRE pipeline configuration before any
expensive operation (training, TTO, eval). Run it first, always.

Usage:
    from tac.preflight import preflight_check
    preflight_check(
        renderer_path="renderer.bin",
        masks_path="masks.mkv",
        poses_path="optimized_poses.pt",
    )
"""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import torch


class PreflightError(Exception):
    """A preflight check failed — do NOT proceed."""
    pass


class PreflightWarning:
    """A preflight check raised a concern but is not fatal."""
    def __init__(self, msg: str):
        self.msg = msg


def preflight_check(
    renderer_path: str | Path | None = None,
    masks_path: str | Path | None = None,
    poses_path: str | Path | None = None,
    archive_path: str | Path | None = None,
    expected_n_frames: int = 1200,
    expected_n_pairs: int = 600,
    expected_seg_h: int = 384,
    expected_seg_w: int = 512,
    verbose: bool = True,
) -> list[PreflightWarning]:
    """Run all preflight checks. Raises PreflightError on fatal issues.

    Returns list of warnings (non-fatal concerns).
    """
    warnings: list[PreflightWarning] = []
    checks_passed = 0
    checks_total = 0

    def _pass(msg: str) -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        checks_passed += 1
        if verbose:
            print(f"  [PASS] {msg}")

    def _fail(msg: str) -> None:
        nonlocal checks_total
        checks_total += 1
        if verbose:
            print(f"  [FAIL] {msg}")
        raise PreflightError(msg)

    def _warn(msg: str) -> None:
        nonlocal checks_total
        checks_total += 1
        warnings.append(PreflightWarning(msg))
        if verbose:
            print(f"  [WARN] {msg}")

    if verbose:
        print("=" * 60)
        print("PREFLIGHT CHECK")
        print("=" * 60)

    # ── Renderer checks ──────────────────────────────────────────
    if renderer_path:
        renderer_path = Path(renderer_path)
        if not renderer_path.exists():
            _fail(f"Renderer not found: {renderer_path}")

        raw = renderer_path.read_bytes()
        magic = raw[:4]

        if magic == b"ASYM":
            header_len = struct.unpack("<I", raw[4:8])[0]
            import json
            header = json.loads(raw[8:8 + header_len])
            pose_dim = header.get("pose_dim", 0)
            base_ch = header.get("base_ch", "?")
            dsconv = header.get("use_dsconv", False)
            _pass(f"Renderer: ASYM, pose_dim={pose_dim}, base_ch={base_ch}, dsconv={dsconv}, {len(raw):,}B")

            if pose_dim == 0:
                _warn("Renderer has pose_dim=0 — FiLM conditioning disabled, poses will have no effect")
            if pose_dim > 0 and poses_path is None:
                _warn("Renderer has pose_dim>0 but no poses_path provided — will use zero poses")
        elif magic == b"FP4A":
            _pass(f"Renderer: FP4A, {len(raw):,}B")
            _warn("FP4 renderer — verify QAT was used during training (post-hoc QAT degrades 3-26x)")
        else:
            _warn(f"Renderer: unknown format (magic={magic}), assuming PyTorch .pt")

    # ── Mask checks ──────────────────────────────────────────────
    if masks_path:
        masks_path = Path(masks_path)
        if not masks_path.exists():
            _fail(f"Masks not found: {masks_path}")

        if masks_path.suffix in (".mkv", ".mp4"):
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0",
                 str(masks_path)],
                capture_output=True, text=True, timeout=10,
            )
            if probe.returncode == 0:
                parts = probe.stdout.strip().split(",")
                w, h = int(parts[0]), int(parts[1])
                size = masks_path.stat().st_size

                if h == expected_seg_h and w == expected_seg_w:
                    _pass(f"Masks: {w}x{h} (native resolution), {size:,}B")
                elif expected_seg_h % h == 0 and expected_seg_w % w == 0:
                    scale = expected_seg_h // h
                    _warn(f"Masks at 1/{scale} resolution ({w}x{h}), will upsample to {expected_seg_w}x{expected_seg_h}")
                else:
                    _fail(f"Masks resolution {w}x{h} is not a clean factor of {expected_seg_w}x{expected_seg_h}")
            else:
                _warn("Could not probe mask video with ffprobe")
        elif masks_path.suffix == ".pt":
            m = torch.load(str(masks_path), weights_only=True)
            if m.shape[1] != expected_seg_h or m.shape[2] != expected_seg_w:
                _warn(f"Masks shape {m.shape} — expected (N, {expected_seg_h}, {expected_seg_w})")
            else:
                _pass(f"Masks: {m.shape}, .pt format")

    # ── Pose checks ──────────────────────────────────────────────
    if poses_path:
        poses_path = Path(poses_path)
        if not poses_path.exists():
            _fail(f"Poses not found: {poses_path}")

        p = torch.load(str(poses_path), weights_only=True)
        if p.shape[0] != expected_n_pairs:
            _fail(f"Poses shape {p.shape} — expected ({expected_n_pairs}, 6). "
                  f"Wrong number of pairs.")
        if p.shape[1] != 6:
            _fail(f"Poses shape {p.shape} — expected (N, 6). Wrong pose dimension.")
        _pass(f"Poses: {p.shape}, dtype={p.dtype}")

        if p.abs().max() > 100:
            _warn(f"Poses max value {p.abs().max():.1f} — unusually large, may indicate wrong scale")
        if p.abs().mean() < 0.001:
            _warn(f"Poses mean abs {p.abs().mean():.6f} — near zero, may not have been optimized")

    # ── Pose-mask consistency ────────────────────────────────────
    if poses_path and masks_path and masks_path.suffix in (".mkv", ".mp4"):
        _warn("Cannot verify poses were optimized against THESE masks (not fresh SegNet). "
              "Use --masks flag in optimize_poses.py to ensure match.")

    # ── Archive checks ───────────────────────────────────────────
    if archive_path:
        archive_path = Path(archive_path)
        if not archive_path.exists():
            _fail(f"Archive not found: {archive_path}")

        try:
            from tac.submission_archive import validate_archive, RENDERER_SUBMISSION_MANIFEST
            result = validate_archive(archive_path, RENDERER_SUBMISSION_MANIFEST, strict=False)
            if result.valid:
                _pass(f"Archive: {result.archive_bytes:,}B, rate={result.rate_term:.4f}, valid")
            else:
                for err in result.errors:
                    _fail(f"Archive: {err}")
                for w in result.warnings:
                    _warn(f"Archive: {w}")
        except Exception as e:
            _warn(f"Archive validation failed: {e}")

    # ── Summary ──────────────────────────────────────────────────
    if verbose:
        print(f"\n  {checks_passed}/{checks_total} checks passed, {len(warnings)} warnings")
        if warnings:
            print("  Warnings:")
            for w in warnings:
                print(f"    - {w.msg}")
        print("=" * 60)

    return warnings


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preflight pipeline validator")
    parser.add_argument("--renderer", type=str, default=None)
    parser.add_argument("--masks", type=str, default=None)
    parser.add_argument("--poses", type=str, default=None)
    parser.add_argument("--archive", type=str, default=None)
    args = parser.parse_args()

    try:
        preflight_check(
            renderer_path=args.renderer,
            masks_path=args.masks,
            poses_path=args.poses,
            archive_path=args.archive,
        )
    except PreflightError as e:
        print(f"\nPREFLIGHT FAILED: {e}")
        sys.exit(1)
