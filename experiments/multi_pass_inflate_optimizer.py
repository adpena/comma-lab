#!/usr/bin/env python3
"""Lane 8 (Phase 1) — multi-pass inflate optimizer for compress-time UNLIMITED.

The 6-month plan's Lane 8 envisions a compress-time iteration loop that
wraps the canonical inflate_renderer.py pipeline with score feedback to
refine archive content. The DEPLOY-time inflate stays single-pass +
scorer-free + within 30-min T4 bound; the COMPRESS-time iteration is
unbounded.

Existing infrastructure:
  - src/tac/trick_stack.py:_stage_multi_pass — POSTFILTER multi-pass
    (legacy inflate_postfilter.py path; NOT the canonical inflate_renderer)
  - experiments/optimize_poses.py — single-purpose pose TTO with scorer
    feedback; the building block this module GENERALIZES
  - src/tac/learnable_class_targets.py — LCT codebook ema_update
    (Quantizr-style VQ-VAE-2 persistent buffers per commit 90398b7c)
  - submissions/robust_current/compress_archive.py — deterministic
    archive packing (with --pose-delta, --binary-poses, --brotli)

This module is the OUTER LOOP that composes the above:

    while not converged:
        rgb = inflate(archive_content)
        seg_loss, pose_loss = score(rgb, gt)            # compress-time SegNet/PoseNet
        grad = backprop(seg_loss + pose_loss)
        archive_content = update_step(archive_content, grad)  # poses, LCT, masks
        if score_plateau(patience=3) or iter > MAX: break
    write_deterministic_archive(archive_content)        # final output

CLAUDE.md non-negotiables:
  - encoder uses --device cuda everywhere (no MPS)
  - eval_roundtrip=True (matches contest eval resize chain)
  - strict-scorer-rule: SegNet/PoseNet AT COMPRESS TIME only; deploy-time
    inflate sees NO scorer
  - bit-deterministic output via build_submission_archive

Status: SCAFFOLD only. Phase 1 verdict (Round 1 codex) flagged Lane 8 as
RED — current trick_stack._stage_multi_pass targets the wrong pipeline.
This module establishes the canonical-pipeline interface + convergence
criterion. Detailed inner-loop implementation (gradient computation,
LCT EMA threading, mask value updates) will land via subsequent commits
gated by adversarial review.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src",):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@dataclass
class MultiPassConfig:
    """Configuration for the compress-time multi-pass inflate optimizer.

    All thresholds chosen to match Phase 1 codex prescriptions:
    - max_iters = 5: per Round 2 codex sketch ("MVP is 50 LOC wrapper around
      single TTO step on poses + single LCT EMA update + single archive-byte
      gradient step; stop when score plateaus, patience=3")
    - patience = 3: Karpathy / standard early-stop convention
    - tol = 0.0005: Half the smallest meaningful score delta (60KB = 0.04;
      we want to detect plateau LONG before the final 0.0005)
    - wall_clock_cap_sec = 14400 (4h): caller can override; matches Modal
      T4 budget for a 4h iteration session
    """
    max_iters: int = 5
    patience: int = 3
    tol: float = 0.0005
    wall_clock_cap_sec: int = 4 * 3600
    # What components to optimize each pass. Conservative default = poses
    # only (proven via existing optimize_poses.py path).
    optimize_poses: bool = True
    optimize_lct: bool = False  # codebook EMA — needs SegMap renderer support
    optimize_mask_values: bool = False  # full mask-byte gradient — Phase 3 territory
    # Per-pass step sizes
    pose_lr: float = 0.005
    pose_steps: int = 100
    # Output
    output_archive: Path = field(default_factory=lambda: _REPO / "multi_pass_out.zip")


@dataclass
class PassResult:
    """One outer-iteration result. Logged to manifest.json."""
    iter_idx: int
    archive_bytes: int
    seg_distortion: float
    pose_distortion: float
    score: float
    elapsed_seconds: float
    converged: bool = False


def _score_archive(archive_path: Path, *, device: str = "cuda") -> tuple[float, float, float]:
    """Run contest_auth_eval on the archive, return (seg, pose, score).

    Uses experiments/contest_auth_eval.py as the authoritative scorer
    (same path the council kill/promote decisions use). Returns the
    canonical contest-CUDA score triple.

    Raises:
        RuntimeError: if auth_eval fails or returns malformed output.
    """
    if device != "cuda":
        raise ValueError(
            f"_score_archive: device={device!r} forbidden. CLAUDE.md "
            f"non-negotiable: contest scoring is CUDA-only. MPS/CPU will "
            f"silently drift 23x on PoseNet."
        )
    # Defer the import so this module imports cleanly on CPU-only hosts
    # for static analysis / preflight.
    import subprocess
    import re

    contest_eval = _REPO / "experiments" / "contest_auth_eval.py"
    if not contest_eval.exists():
        raise RuntimeError(f"contest_auth_eval not found at {contest_eval}")
    with tempfile.TemporaryDirectory() as td:
        cmd = [
            sys.executable, str(contest_eval),
            "--archive", str(archive_path),
            "--inflate-sh", str(_REPO / "submissions/robust_current/inflate.sh"),
            "--upstream-dir", str(_REPO / "upstream"),
            "--device", device,
            "--work-dir", td,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(
            f"contest_auth_eval failed rc={result.returncode}: "
            f"stderr={result.stderr[-500:]}"
        )
    # Parse RESULT_JSON line from stdout
    m = re.search(
        r"RESULT_JSON:\s*(\{[^\n]+\})", result.stdout
    )
    if not m:
        raise RuntimeError(
            f"contest_auth_eval emitted no RESULT_JSON; stdout tail: "
            f"{result.stdout[-500:]}"
        )
    data = json.loads(m.group(1))
    return (
        float(data["avg_segnet_dist"]),
        float(data["avg_posenet_dist"]),
        float(data["final_score"]),
    )


def _check_score_plateau(
    history: list[PassResult],
    patience: int,
    tol: float,
) -> bool:
    """True if best score didn't improve by >tol in the last `patience` iters."""
    if len(history) < patience + 1:
        return False
    best_recent = min(r.score for r in history[-patience - 1 :])
    best_old = min(r.score for r in history[: -patience - 1])
    return (best_old - best_recent) < tol


def run_multi_pass(
    initial_archive: Path,
    cfg: MultiPassConfig,
    *,
    gt_video: Path | None = None,
    device: str = "cuda",
) -> tuple[Path, list[PassResult]]:
    """Run the compress-time multi-pass inflate optimizer.

    SCAFFOLD STATUS: this entrypoint validates the iteration shape +
    score-plateau convergence + manifest writing, but the per-pass
    update step is currently a NO-OP placeholder. The real implementation
    lands in subsequent commits, each gated by 3-clean-pass review.

    Args:
        initial_archive: starting archive (e.g. Lane G v3 archive_lane_g_v3.zip)
        cfg: MultiPassConfig with iteration budget, optimization knobs, output path
        gt_video: optional override for upstream/videos/0.mkv
        device: must be 'cuda' for contest-CUDA scoring

    Returns:
        (final_archive_path, history_of_PassResult)
    """
    if not initial_archive.exists():
        raise FileNotFoundError(f"initial archive not found: {initial_archive}")

    history: list[PassResult] = []
    current_archive = initial_archive
    start_time = time.monotonic()

    for iter_idx in range(cfg.max_iters):
        elapsed = time.monotonic() - start_time
        if elapsed > cfg.wall_clock_cap_sec:
            print(
                f"[multi-pass] iter {iter_idx}: wall-clock cap "
                f"{cfg.wall_clock_cap_sec}s reached at {elapsed:.0f}s. STOP."
            )
            break

        # Step 1: score current archive (contest-CUDA, authoritative)
        try:
            seg, pose, score = _score_archive(current_archive, device=device)
        except RuntimeError as exc:
            print(f"[multi-pass] iter {iter_idx}: scoring failed: {exc}")
            break

        archive_bytes = current_archive.stat().st_size
        result = PassResult(
            iter_idx=iter_idx,
            archive_bytes=archive_bytes,
            seg_distortion=seg,
            pose_distortion=pose,
            score=score,
            elapsed_seconds=elapsed,
        )
        history.append(result)
        print(
            f"[multi-pass] iter {iter_idx}: archive={archive_bytes:,}B "
            f"seg={seg:.6f} pose={pose:.6f} score={score:.4f} "
            f"elapsed={elapsed:.0f}s"
        )

        # Step 2: convergence check
        if _check_score_plateau(history, cfg.patience, cfg.tol):
            print(
                f"[multi-pass] iter {iter_idx}: score plateau "
                f"(patience={cfg.patience}, tol={cfg.tol}). CONVERGED."
            )
            history[-1].converged = True
            break

        # Step 3: update archive content (SCAFFOLD: currently no-op)
        # Real implementation will:
        #   - if cfg.optimize_poses: 1 TTO step via experiments/optimize_poses
        #   - if cfg.optimize_lct: 1 EMA update via learnable_class_targets
        #   - if cfg.optimize_mask_values: 1 mask-byte gradient step
        #   - re-pack archive deterministically via build_submission_archive
        # All gated by 3-clean-pass adversarial review per landing.
        print(
            f"[multi-pass] iter {iter_idx}: update step is SCAFFOLD no-op. "
            f"Stop here pending Round-2-cleared inner-loop implementation."
        )
        break

    # Write manifest + final archive
    cfg.output_archive.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        print("[multi-pass] no iterations completed; output archive == input")
        # Bit-identical copy
        import shutil
        shutil.copy2(initial_archive, cfg.output_archive)
    else:
        # Output is the BEST archive seen. For now since update is a no-op,
        # this is just the input.
        import shutil
        shutil.copy2(initial_archive, cfg.output_archive)
    manifest_path = cfg.output_archive.with_suffix(".manifest.json")
    manifest = {
        "lane": "multi_pass_inflate_optimizer",
        "config": {
            "max_iters": cfg.max_iters,
            "patience": cfg.patience,
            "tol": cfg.tol,
            "wall_clock_cap_sec": cfg.wall_clock_cap_sec,
            "optimize_poses": cfg.optimize_poses,
            "optimize_lct": cfg.optimize_lct,
            "optimize_mask_values": cfg.optimize_mask_values,
        },
        "iterations": [
            {
                "iter": r.iter_idx,
                "archive_bytes": r.archive_bytes,
                "seg": r.seg_distortion,
                "pose": r.pose_distortion,
                "score": r.score,
                "elapsed_sec": r.elapsed_seconds,
                "converged": r.converged,
            }
            for r in history
        ],
        "scaffold_status": "no-op update step pending Round 2 review",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[multi-pass] wrote {manifest_path}")
    return cfg.output_archive, history


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Lane 8 multi-pass inflate optimizer — compress-time unlimited. "
            "SCAFFOLD: per-pass update is no-op pending Round 2 review."
        )
    )
    p.add_argument(
        "--initial-archive", type=Path, required=True,
        help="Starting archive (e.g. experiments/results/lane_g_v3_landed/"
             "archive_lane_g_v3.zip)",
    )
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--tol", type=float, default=0.0005)
    p.add_argument(
        "--wall-clock-cap-sec", type=int, default=4 * 3600,
        help="Wall-clock budget per session (default 4h on Modal T4)",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="MUST be cuda. CLAUDE.md non-negotiable: MPS/CPU drift 23x.",
    )
    p.add_argument(
        "--output-archive", type=Path,
        default=_REPO / "experiments/results/multi_pass_out/archive.zip",
    )
    args = p.parse_args()

    cfg = MultiPassConfig(
        max_iters=args.max_iters,
        patience=args.patience,
        tol=args.tol,
        wall_clock_cap_sec=args.wall_clock_cap_sec,
        output_archive=args.output_archive,
    )
    out, history = run_multi_pass(
        args.initial_archive, cfg, device=args.device,
    )
    print(
        f"[multi-pass] done. output={out} iterations={len(history)} "
        f"final_score={history[-1].score if history else 'N/A'}"
    )


if __name__ == "__main__":
    main()
