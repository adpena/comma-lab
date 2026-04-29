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
        seg, pose, score = score(archive)                # contest-CUDA auth
        if score_plateau(patience): break
        poses, lct = update_step(poses, lct)             # 1 TTO step + 1 EMA
        archive = repack_deterministic(poses, lct, ...)  # ZipInfo+writestr
    write_deterministic_archive(best_archive_seen)       # final output

CLAUDE.md non-negotiables:
  - encoder uses --device cuda everywhere (no MPS)
  - eval_roundtrip=True (matches contest eval resize chain)
  - strict-scorer-rule: SegNet/PoseNet AT COMPRESS TIME only; deploy-time
    inflate sees NO scorer
  - bit-deterministic output via _deterministic_zip_write (codex R5-r6 #5;
    same fixed-timestamp ZipInfo + writestr pattern as
    experiments/results/lane_a_brotli/build_brotli_from_lane_a.py)

Status (2026-04-29): Phase 1 council Y3→G3 promotion. Inner loop MVP wired —
1 TTO step on poses + 1 LCT EMA update + deterministic archive re-pack per
outer iteration, score-plateau convergence with the existing
MultiPassConfig knobs. Remote / GPU integration of optimize_poses_batch
lives behind an injectable callback so the loop is unit-testable on a
synthetic archive without GPU.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src",):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic ZIP write — codex R5-r6 #5
# Mirrors experiments/results/lane_a_brotli/build_brotli_from_lane_a.py so
# multi-pass re-packs produce byte-identical archives across reruns / hosts.
# This file's filename does not match the `experiments/build*.py` glob the
# preflight scanner walks today, but we still use the deterministic helper
# because the rate term feedback loop REQUIRES bit-stable bytes.
# DETERMINISTIC_ZIP_OK
# ─────────────────────────────────────────────────────────────────────────────
_DET_ZIP_DATE_TIME = (2026, 4, 29, 0, 0, 0)
_DET_ZIP_EXTERNAL_ATTR = (0o644 & 0xFFFF) << 16
_DET_ZIP_CREATE_SYSTEM = 3  # Unix


def _deterministic_zip_write(
    z: zipfile.ZipFile,
    arcname: str,
    data: bytes,
    *,
    compress_type: int = zipfile.ZIP_DEFLATED,
    compresslevel: int = 9,
) -> None:
    """Write ``data`` into ``z`` as ``arcname`` with FIXED metadata.

    Fixed date_time + external_attr + create_system means the zip's
    central directory is byte-identical given identical inputs. The
    rate-term feedback loop relies on this.
    """
    info = zipfile.ZipInfo(filename=arcname, date_time=_DET_ZIP_DATE_TIME)
    info.compress_type = compress_type
    info.external_attr = _DET_ZIP_EXTERNAL_ATTR
    info.create_system = _DET_ZIP_CREATE_SYSTEM
    z.writestr(info, data, compress_type=compress_type, compresslevel=compresslevel)


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


# ─────────────────────────────────────────────────────────────────────────────
# Score function — contest-CUDA, strict-scorer-rule compliant.
# This runs at COMPRESS time (inside the multi-pass loop), NOT at inflate
# time. The deploy-time inflate path has zero scorer access.
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Convergence — score-plateau early stop with `patience` consecutive iters
# of improvement < tol.
# ─────────────────────────────────────────────────────────────────────────────


def _check_score_plateau(
    history: list[PassResult],
    patience: int,
    tol: float,
) -> bool:
    """True when the last `patience` iters all failed to improve best by >= tol.

    The improvement is measured against the best score BEFORE the patience
    window. Once `patience` consecutive non-improvements land, the loop
    stops at the iter that triggered the patience gate.

    Edge cases:
        - history shorter than patience+1 → False (need at least one
          baseline iter plus `patience` non-improvers)
        - lower score is better (matches contest convention)
    """
    if len(history) < patience + 1:
        return False
    best_old = min(r.score for r in history[: -patience])
    recent_best = min(r.score for r in history[-patience:])
    return (best_old - recent_best) < tol


# ─────────────────────────────────────────────────────────────────────────────
# Inner loop — extract → 1 TTO step + 1 LCT EMA update → deterministic re-pack.
# The default `_default_inner_step` calls into experiments/optimize_poses.py
# and tac.learnable_class_targets; tests inject a stub callback so the loop
# can be exercised without GPU.
# ─────────────────────────────────────────────────────────────────────────────


def _extract_archive(archive_path: Path, dest_dir: Path) -> dict[str, bytes]:
    """Read every entry of `archive_path` into a name → bytes dict.

    The dict keeps insertion order matching the zip's central-directory
    order so the re-pack can sort deterministically without losing the
    audit trail.
    """
    contents: dict[str, bytes] = {}
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            with zf.open(info) as fh:
                contents[info.filename] = fh.read()
    return contents


def _repack_archive(
    contents: dict[str, bytes],
    output_path: Path,
) -> Path:
    """Re-pack `contents` into a deterministic zip.

    Entries are sorted by arcname so the on-disk order is stable across
    reruns. Combined with `_deterministic_zip_write`'s fixed metadata,
    this produces byte-identical archives given identical inputs (which is
    what the rate-term feedback loop requires).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_names = sorted(contents.keys())
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in sorted_names:
            _deterministic_zip_write(zf, name, contents[name])
    return output_path


def _default_inner_step(
    contents: dict[str, bytes],
    cfg: MultiPassConfig,
    *,
    device: str,
    work_dir: Path,
) -> dict[str, bytes]:
    """Real inner step: 1 TTO call on poses + 1 LCT EMA update.

    This is the production path. It calls into
    ``experiments/optimize_poses.py:optimize_poses_batch`` for the pose
    update and ``tac.learnable_class_targets.LearnableClassTargets.ema_update``
    for the codebook update. The function is deliberately thin —
    the heavy lifting lives in those two callees, which the wider
    skunkworks council and CLAUDE.md non-negotiables already gate.

    Tests inject a lighter stub via ``run_multi_pass(inner_step_fn=...)``
    so the loop wiring can be verified without GPU. Production callers
    use this default, which requires CUDA + an upstream renderer +
    PoseNet/SegNet weights — same prerequisites as the underlying
    optimize_poses.py CLI.
    """
    # Production wiring: the council Y3→G3 spec is one outer iter ==
    # one call to optimize_poses_batch with `pose_steps` inner steps
    # (the default 100 matches the existing MultiPassConfig). Caller
    # is expected to have the renderer + scorers already loaded; this
    # MVP doesn't try to bootstrap them in-process — that's a Phase 2
    # remote-orchestration concern. Until then, the GPU path raises a
    # clear NotImplementedError and tests use injected stubs.
    raise NotImplementedError(
        "_default_inner_step: production GPU wiring lands as a Phase 2 "
        "follow-up (remote orchestration of optimize_poses_batch + "
        "segmap_renderer LCT extraction). For Phase 1 MVP the loop "
        "control flow + deterministic re-pack are exercised via the "
        "`inner_step_fn` injection point in `run_multi_pass()`."
    )


def run_multi_pass(
    initial_archive: Path,
    cfg: MultiPassConfig,
    *,
    gt_video: Path | None = None,
    device: str = "cuda",
    score_fn: Callable[[Path, str], tuple[float, float, float]] | None = None,
    inner_step_fn: Callable[[dict[str, bytes], MultiPassConfig, Any], dict[str, bytes]] | None = None,
) -> tuple[Path, list[PassResult]]:
    """Run the compress-time multi-pass inflate optimizer.

    Council Y3→G3 spec (2026-04-29): each outer iter runs one TTO step
    on poses + one LCT EMA update + a deterministic archive re-pack.
    Convergence is score plateau over `patience` iters within `tol`,
    capped at `max_iters` and `wall_clock_cap_sec`.

    Args:
        initial_archive: starting archive (e.g. Lane G v3 archive_lane_g_v3.zip)
        cfg: MultiPassConfig with iteration budget, optimization knobs, output path
        gt_video: optional override for upstream/videos/0.mkv
        device: must be 'cuda' for contest-CUDA scoring
        score_fn: override for contest-CUDA scoring (tests inject a deterministic
            stub so the loop runs without GPU)
        inner_step_fn: override for the per-iter update step (tests inject a
            stub that mutates `optimized_poses.pt` deterministically; production
            path will land in Phase 2 once the GPU orchestration is in place)

    Returns:
        (final_archive_path, history_of_PassResult)
    """
    if not initial_archive.exists():
        raise FileNotFoundError(f"initial archive not found: {initial_archive}")

    score = score_fn if score_fn is not None else _score_archive
    inner_step = inner_step_fn if inner_step_fn is not None else _default_inner_step

    history: list[PassResult] = []
    start_time = time.monotonic()

    cfg.output_archive.parent.mkdir(parents=True, exist_ok=True)
    work_root = cfg.output_archive.parent / f"{cfg.output_archive.stem}_work"
    work_root.mkdir(parents=True, exist_ok=True)

    # The "current" archive is what we score this iter; the next iter
    # rebuilds from `current_contents` after the inner step mutates it.
    current_archive = initial_archive
    current_contents: dict[str, bytes] = _extract_archive(initial_archive, work_root)
    best_archive: Path = initial_archive
    best_score: float = float("inf")

    converged = False
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
            seg, pose, score_val = score(current_archive, device)
        except RuntimeError as exc:
            print(f"[multi-pass] iter {iter_idx}: scoring failed: {exc}")
            break

        archive_bytes = current_archive.stat().st_size
        result = PassResult(
            iter_idx=iter_idx,
            archive_bytes=archive_bytes,
            seg_distortion=seg,
            pose_distortion=pose,
            score=score_val,
            elapsed_seconds=elapsed,
        )
        history.append(result)
        print(
            f"[multi-pass] iter {iter_idx}: archive={archive_bytes:,}B "
            f"seg={seg:.6f} pose={pose:.6f} score={score_val:.4f} "
            f"elapsed={elapsed:.0f}s"
        )

        # Track best archive seen — this is what the loop returns as the
        # final output. Tied scores keep the earlier (smaller iter_idx)
        # archive (no benefit to the tie-breaker; first-seen is reproducible).
        if score_val < best_score:
            best_score = score_val
            best_archive = current_archive

        # Step 2: convergence check (after at least patience+1 iters)
        if _check_score_plateau(history, cfg.patience, cfg.tol):
            print(
                f"[multi-pass] iter {iter_idx}: score plateau "
                f"(patience={cfg.patience}, tol={cfg.tol}). CONVERGED."
            )
            history[-1].converged = True
            converged = True
            break

        # Step 3: don't bother updating if this was the last iter — score
        # would never feed back into a decision.
        if iter_idx == cfg.max_iters - 1:
            break

        # Step 4: inner update step. 1 TTO step on poses + 1 LCT EMA update
        # produces a NEW `current_contents` dict; the deterministic re-pack
        # writes the next-iter archive.
        current_contents = inner_step(
            current_contents,
            cfg,
            {"device": device, "work_dir": work_root, "iter_idx": iter_idx},
        )
        next_archive = work_root / f"iter_{iter_idx + 1}.zip"
        current_archive = _repack_archive(current_contents, next_archive)

    # Final output: the best archive seen, re-packed to the canonical
    # output path (so the destination is always written even if no iters
    # ran or if the best was the input itself).
    if best_archive == initial_archive:
        # Bit-identical copy of the input. shutil.copy2 preserves
        # metadata; the archive bytes are what the rate term cares about,
        # not metadata, but copy2 is the intent-clear API.
        shutil.copy2(initial_archive, cfg.output_archive)
    else:
        shutil.copy2(best_archive, cfg.output_archive)

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
            "pose_lr": cfg.pose_lr,
            "pose_steps": cfg.pose_steps,
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
        "converged": converged,
        "best_score": best_score if history else None,
        "best_archive_iter": int(min(range(len(history)), key=lambda i: history[i].score))
            if history else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[multi-pass] wrote {manifest_path}")
    return cfg.output_archive, history


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Lane 8 multi-pass inflate optimizer — compress-time unlimited. "
            "Phase 1 MVP: control flow + deterministic re-pack landed; the "
            "production GPU inner step lands as Phase 2."
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
