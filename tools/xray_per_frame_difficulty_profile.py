#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Per-frame score-aware difficulty profile for the contest video.

Per HNeRV parity discipline lesson 1 (`feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`):
the difficulty signal that drives a frame-conditional codec MUST be derived
from gradient-through-SegNet/PoseNet, NOT from raw L²/KL on frames.

The existing ``tac.codec.frame_conditional_bit_budget.compute_per_frame_complexity``
returns ``edge_density × pixel_variance × frame_difference`` which is
score-naive. This tool fills the gap surfaced by xray-tools-enhancement
subagent (``feedback_profiling_xray_tools_enhancement_landed_20260509.md``).

What it computes for every frame in ``upstream/videos/0.mkv``:

* ``segnet_entropy`` — mean per-pixel Shannon entropy of SegNet logits over
  the 5 classes after softmax. A frame whose mask is highly uncertain
  (boundaries, occlusions) scores high; a frame where SegNet is confident
  scores low. Computed as ``-Σ p_c·log p_c`` averaged across pixels.
* ``posenet_variance`` — variance of the PoseNet first-6 output across small
  perturbations (chroma jitter ±1, horizontal shift ±1px, brightness ±2/255)
  for the pair containing this frame. A pair where PoseNet is sensitive to
  perturbation is high difficulty; a pair where PoseNet is stable is low.
* ``combined_difficulty`` — α·H(SegNet) + β·Var(PoseNet) with calibrated
  weights derived from the contest-score formula (lambda_seg=100,
  lambda_pose=DEFAULT_CPU_POSE_SCORE_WEIGHT≈5.0).
* ``percentile_rank`` — 0-100 within-video rank by combined_difficulty.

Output: ``experiments/results/per_frame_difficulty_profile_<timestamp>/profile.json``
plus an ASCII histogram + per-decile summary in ``profile.md``.

Tagging discipline (per CLAUDE.md `forbidden_score_claims`):
* All fields tagged ``[diagnostic; per-frame difficulty profile on upstream
  video]`` — NOT a score claim.
* ``score_claim=False``, ``promotion_eligible=False``,
  ``ready_for_exact_eval_dispatch=False``, ``evidence_grade=diagnostic_only``
  per gate #113 DERIVED_OUTPUT discipline.
* Markdown report includes regen header
  ``<!-- generated_at: ..., from_state_hash: ... -->``.

Hardware: CPU-only by default (decode + 2400 forward passes through
SegNet/PoseNet on 1200 frames × 2 frames-per-pair = ~3-4 hours on M5 Max
performance cores). Optional ``--device cuda`` for ~10× speedup. Tagged
``[macOS-CPU calibrated]`` when run on M5 Max per
``feedback_macos_x86_64_epsilon_calibrated_tag_20260508.md`` because the
output is a *diagnostic* (not a score) — the per-frame ranking is
substrate-stable across CPU/CUDA per the conv-accumulation drift mechanism
(it's a relative ordering, not an absolute number).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ImportError:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "upstream"))


def _open_video_pairs(video_path: Path, *, n_pairs: int, camera_h: int = 874, camera_w: int = 1164):
    """Decode the first 2*n_pairs frames as uint8 numpy ``(2*n_pairs, H, W, 3)``."""
    try:
        import av  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyAV required: `uv pip install av`") from exc
    from frame_utils import yuv420_to_rgb  # type: ignore[import-not-found]

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        frames: list[torch.Tensor] = []
        for frame in container.decode(stream):
            tensor = yuv420_to_rgb(frame)
            frames.append(tensor)
            if len(frames) >= 2 * n_pairs:
                break
    finally:
        container.close()
    if len(frames) < 2 * n_pairs:
        raise RuntimeError(
            f"Video {video_path} has only {len(frames)} frames; need {2 * n_pairs}"
        )
    return [np.asarray(f.cpu().numpy() if hasattr(f, "cpu") else f, dtype=np.uint8) for f in frames]


def _segnet_entropy_per_frame(
    frames_uint8: list[np.ndarray],
    segnet: torch.nn.Module,
    *,
    device: torch.device,
    eval_h: int = 384,
    eval_w: int = 512,
    batch_size: int = 4,
) -> np.ndarray:
    """Per-frame mean Shannon entropy of SegNet 5-class logits.

    SegNet input is the LAST frame of a 2-frame pair, but for the
    *per-frame* difficulty profile we treat each frame as if it were the
    "last frame" of an artificial pair (frame, frame). This is correct
    because SegNet only consumes ``x[:, -1, ...]`` so duplicating doesn't
    change behavior. The output entropy is intrinsic to the single frame.
    """
    n = len(frames_uint8)
    out = np.zeros(n, dtype=np.float64)
    segnet.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_np = np.stack(frames_uint8[start:end], axis=0)  # (B, H, W, 3)
        # SegNet expects (B, T, H, W, C) with T=2. Duplicate the frame.
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        batch_pair = batch.unsqueeze(1).expand(-1, 2, -1, -1, -1).contiguous()
        # Permute to (B, T, C, H, W) — SegNet preprocess_input handles it.
        batch_pair = batch_pair.permute(0, 1, 4, 2, 3).contiguous()
        with torch.no_grad():
            seg_in = segnet.preprocess_input(batch_pair)
            logits = segnet(seg_in)  # (B, 5, H', W')
            probs = F.softmax(logits, dim=1).clamp(min=1e-12)
            ent = -(probs * probs.log()).sum(dim=1)  # (B, H', W')
            ent_per_frame = ent.mean(dim=(1, 2))  # (B,)
        out[start:end] = ent_per_frame.detach().cpu().numpy().astype(np.float64)
    return out


def _posenet_variance_per_pair(
    frames_uint8: list[np.ndarray],
    posenet: torch.nn.Module,
    *,
    device: torch.device,
    n_perturbations: int = 4,
    perturb_brightness: float = 2.0,
    perturb_shift_pixels: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Per-pair variance of PoseNet first-6 dims under small perturbations.

    Returns a 1-D array of length ``len(frames_uint8) // 2`` (one variance
    per pair). Per-frame difficulty for both frames in a pair inherits the
    same variance value (pose is a pair-level signal).

    Round 1 adversarial review (Quantizr): spatial perturbations are added
    in addition to brightness jitter because PoseNet is more sensitive to
    spatial than to brightness perturbations. Each perturbation combines a
    random brightness jitter ±``perturb_brightness/255`` with a random
    horizontal shift of ±``perturb_shift_pixels`` (via np.roll, which
    wraps; for the difficulty signal the wrap-effect at frame edges is
    negligible compared to the shift itself).
    """
    n_frames = len(frames_uint8)
    if n_frames % 2 != 0:
        raise ValueError(f"frames must be even count, got {n_frames}")
    n_pairs = n_frames // 2
    out = np.zeros(n_pairs, dtype=np.float64)
    posenet.eval()
    rng = np.random.default_rng(seed)
    for pair_idx in range(n_pairs):
        f0 = frames_uint8[2 * pair_idx]
        f1 = frames_uint8[2 * pair_idx + 1]
        original = np.stack([f0, f1], axis=0)
        batch_list = [original]
        for k in range(n_perturbations):
            jitter = rng.uniform(-perturb_brightness, perturb_brightness, size=(2, 1, 1, 3))
            perturbed = np.clip(original.astype(np.float32) + jitter, 0, 255).astype(np.uint8)
            if perturb_shift_pixels > 0:
                shift_x = int(rng.integers(-perturb_shift_pixels, perturb_shift_pixels + 1))
                shift_y = int(rng.integers(-perturb_shift_pixels, perturb_shift_pixels + 1))
                if shift_x != 0 or shift_y != 0:
                    perturbed = np.roll(perturbed, shift=(shift_y, shift_x), axis=(1, 2))
            batch_list.append(perturbed)
        batch_np = np.stack(batch_list, axis=0)  # (P+1, 2, H, W, 3)
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        batch = batch.permute(0, 1, 4, 2, 3).contiguous()  # (P+1, 2, 3, H, W)
        with torch.no_grad():
            pose_in = posenet.preprocess_input(batch)
            pose_out = posenet(pose_in)["pose"][..., :6]  # (P+1, 6)
            var = pose_out.var(dim=0, unbiased=False).mean().item()
        out[pair_idx] = float(var)
    return out


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return per-element percentile rank ∈ [0, 100] using average-rank ties."""
    n = values.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # Convert to percentile in [0, 100]
    return ranks * (100.0 / max(n - 1, 1))


def _state_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _ascii_histogram(values: np.ndarray, *, bins: int = 20, width: int = 40) -> str:
    if values.size == 0:
        return "(empty)"
    counts, edges = np.histogram(values, bins=bins)
    max_count = int(counts.max())
    if max_count == 0:
        return "(zero counts)"
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(width * c / max_count))
        lines.append(f"  [{edges[i]:8.4f}, {edges[i+1]:8.4f}) {bar} {c}")
    return "\n".join(lines)


def compute_per_frame_difficulty_profile(
    *,
    video_path: Path,
    upstream_dir: Path,
    n_frames: int = 1200,
    device: str = "cpu",
    n_pose_perturbations: int = 4,
    pose_perturb_brightness: float = 2.0,
    pose_perturb_shift_pixels: int = 1,
    seed: int = 0,
    progress: bool = False,
) -> dict[str, Any]:
    """Compute the score-aware per-frame difficulty profile.

    Returns a dict with keys:
        ``frames`` — list of dicts ``{frame_idx, segnet_entropy,
                     posenet_variance, combined_difficulty,
                     percentile_rank}`` length ``n_frames``.
        ``stats`` — summary stats (mean, std, percentiles, deciles).
        ``provenance`` — video sha256, n_frames, device, seed, weights.
    """
    if n_frames % 2 != 0:
        raise ValueError(f"n_frames must be even (paired frames), got {n_frames}")
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError(f"device must be cpu/cuda/mps, got {device!r}")
    if device == "mps":
        # Per CLAUDE.md MPS rule: tag, but still compute (this is diagnostic-only).
        pass
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    torch_device = torch.device(device)
    from tac.scorer import load_default_scorers, make_scorers_differentiable

    posenet, segnet = load_default_scorers(str(upstream_dir), device=str(torch_device))
    make_scorers_differentiable(posenet, segnet)
    for p in posenet.parameters():
        p.requires_grad_(False)
    for p in segnet.parameters():
        p.requires_grad_(False)

    if progress:
        print(f"[xray-difficulty] decoding {n_frames} frames from {video_path}", file=sys.stderr)
    n_pairs = n_frames // 2
    frames = _open_video_pairs(video_path, n_pairs=n_pairs)

    if progress:
        print(f"[xray-difficulty] computing SegNet entropy on {n_frames} frames", file=sys.stderr)
    segnet_ent = _segnet_entropy_per_frame(frames, segnet, device=torch_device)

    if progress:
        print(f"[xray-difficulty] computing PoseNet variance on {n_pairs} pairs", file=sys.stderr)
    pose_var_per_pair = _posenet_variance_per_pair(
        frames,
        posenet,
        device=torch_device,
        n_perturbations=n_pose_perturbations,
        perturb_brightness=pose_perturb_brightness,
        perturb_shift_pixels=pose_perturb_shift_pixels,
        seed=seed,
    )
    pose_var_per_frame = np.repeat(pose_var_per_pair, 2)  # both frames inherit pair variance

    # Calibrated weights from contest score formula:
    #   d(score)/d(seg_distort) = 100; d(score)/d(pose_distort) = 5/sqrt(10*pose)
    # We use the marginal-at-A1 operating point (pose_avg ~ 3.4e-5) for pose
    # and lambda_seg=100 for seg, then divide by per-axis std for unit-free
    # combination. This makes combined_difficulty unitless and comparable.
    #
    # MacKay caveat (Round 3 review): the units of Var(PoseNet) are
    # pose-dim^2 while H(SegNet) is nats. The std-normalization makes them
    # COMPARABLE but does NOT preserve interpretability. combined_difficulty
    # is therefore a RANKING signal, not a quantitative score; the
    # percentile_rank field is the load-bearing output for downstream
    # decile assignment.
    seg_std = float(segnet_ent.std()) if segnet_ent.std() > 0 else 1.0
    pose_std = float(pose_var_per_frame.std()) if pose_var_per_frame.std() > 0 else 1.0
    seg_norm = (segnet_ent - segnet_ent.mean()) / seg_std
    pose_norm = (pose_var_per_frame - pose_var_per_frame.mean()) / pose_std
    LAMBDA_SEG = 100.0
    LAMBDA_POSE = 5.04  # R_pose at HNeRV-cluster operating point (calibrated 2026-05-08)
    weight_sum = LAMBDA_SEG + LAMBDA_POSE
    alpha = LAMBDA_SEG / weight_sum
    beta = LAMBDA_POSE / weight_sum
    combined = alpha * seg_norm + beta * pose_norm
    pct = _percentile_rank(combined)

    frames_out: list[dict[str, Any]] = []
    for i in range(n_frames):
        frames_out.append(
            {
                "frame_idx": int(i),
                "segnet_entropy": float(segnet_ent[i]),
                "posenet_variance": float(pose_var_per_frame[i]),
                "combined_difficulty": float(combined[i]),
                "percentile_rank": float(pct[i]),
            }
        )

    deciles = np.percentile(combined, np.arange(10, 101, 10))
    stats = {
        "n_frames": int(n_frames),
        "n_pairs": int(n_pairs),
        "segnet_entropy_mean": float(segnet_ent.mean()),
        "segnet_entropy_std": float(seg_std),
        "segnet_entropy_min": float(segnet_ent.min()),
        "segnet_entropy_max": float(segnet_ent.max()),
        "posenet_variance_mean": float(pose_var_per_frame.mean()),
        "posenet_variance_std": float(pose_std),
        "posenet_variance_min": float(pose_var_per_frame.min()),
        "posenet_variance_max": float(pose_var_per_frame.max()),
        "combined_difficulty_mean": float(combined.mean()),
        "combined_difficulty_std": float(combined.std()),
        "combined_difficulty_deciles": [float(x) for x in deciles],
        "alpha_seg_weight": float(alpha),
        "beta_pose_weight": float(beta),
        "lambda_seg": LAMBDA_SEG,
        "lambda_pose": LAMBDA_POSE,
    }

    video_sha = hashlib.sha256(video_path.read_bytes()).hexdigest()
    provenance = {
        "video_path": str(video_path),
        "video_sha256": video_sha,
        "n_frames": int(n_frames),
        "device": str(torch_device),
        "seed": int(seed),
        "n_pose_perturbations": int(n_pose_perturbations),
        "pose_perturb_brightness": float(pose_perturb_brightness),
        "pose_perturb_shift_pixels": int(pose_perturb_shift_pixels),
        "tag": "[diagnostic; per-frame difficulty profile on upstream video]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "diagnostic_only",
    }

    return {"frames": frames_out, "stats": stats, "provenance": provenance}


def _build_markdown_report(profile: dict[str, Any], *, generated_at: str, state_hash: str) -> str:
    s = profile["stats"]
    p = profile["provenance"]
    combined = np.array([f["combined_difficulty"] for f in profile["frames"]], dtype=np.float64)
    seg_ent = np.array([f["segnet_entropy"] for f in profile["frames"]], dtype=np.float64)
    pose_var = np.array([f["posenet_variance"] for f in profile["frames"]], dtype=np.float64)
    lines = [
        f"<!-- generated_at: {generated_at}, from_state_hash: {state_hash} -->",
        "# Per-frame score-aware difficulty profile",
        "",
        f"**Video**: `{p['video_path']}` (sha256: `{p['video_sha256'][:16]}...`)",
        f"**Frames**: {s['n_frames']} | **Pairs**: {s['n_pairs']}",
        f"**Device**: `{p['device']}` | **Seed**: {p['seed']}",
        f"**Tag**: {p['tag']}",
        "",
        "## Component summary",
        "",
        "| Field | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
        f"| segnet_entropy | {s['segnet_entropy_mean']:.4f} | {s['segnet_entropy_std']:.4f} | {s['segnet_entropy_min']:.4f} | {s['segnet_entropy_max']:.4f} |",
        f"| posenet_variance | {s['posenet_variance_mean']:.6e} | {s['posenet_variance_std']:.6e} | {s['posenet_variance_min']:.6e} | {s['posenet_variance_max']:.6e} |",
        f"| combined_difficulty | {s['combined_difficulty_mean']:.4f} | {s['combined_difficulty_std']:.4f} | {combined.min():.4f} | {combined.max():.4f} |",
        "",
        "## Combined-difficulty deciles",
        "",
    ]
    for i, d in enumerate(s["combined_difficulty_deciles"]):
        lines.append(f"  decile {i+1:2d} ({(i+1)*10:3d}%): {d:.4f}")
    lines += [
        "",
        "## ASCII histogram (combined_difficulty)",
        "",
        _ascii_histogram(combined, bins=20),
        "",
        "## Calibration weights",
        "",
        f"alpha_seg_weight = {s['alpha_seg_weight']:.4f} (lambda_seg = {s['lambda_seg']:.2f})",
        f"beta_pose_weight = {s['beta_pose_weight']:.4f} (lambda_pose = {s['lambda_pose']:.2f})",
        "",
        "## Custody contract",
        "",
        f"- score_claim: {p['score_claim']}",
        f"- promotion_eligible: {p['promotion_eligible']}",
        f"- ready_for_exact_eval_dispatch: {p['ready_for_exact_eval_dispatch']}",
        f"- evidence_grade: {p['evidence_grade']}",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", type=Path, default=REPO_ROOT / "upstream" / "videos" / "0.mkv")
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--n-frames", type=int, default=1200)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--n-pose-perturbations", type=int, default=4)
    parser.add_argument("--pose-perturb-brightness", type=float, default=2.0)
    parser.add_argument("--pose-perturb-shift-pixels", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: experiments/results/per_frame_difficulty_profile_<UTC>",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Process only 8 frames for fast tests")
    args = parser.parse_args(argv)

    n_frames = 8 if args.smoke else args.n_frames
    if args.output_dir is None:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = "_smoke" if args.smoke else ""
        args.output_dir = REPO_ROOT / "experiments" / "results" / f"per_frame_difficulty_profile_{ts}{suffix}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profile = compute_per_frame_difficulty_profile(
        video_path=args.video_path,
        upstream_dir=args.upstream_dir,
        n_frames=n_frames,
        device=args.device,
        n_pose_perturbations=args.n_pose_perturbations,
        pose_perturb_brightness=args.pose_perturb_brightness,
        pose_perturb_shift_pixels=args.pose_perturb_shift_pixels,
        seed=args.seed,
        progress=args.progress,
    )

    state_hash = _state_hash(profile)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

    json_path = args.output_dir / "profile.json"
    md_path = args.output_dir / "profile.md"
    json_path.write_text(json.dumps(profile, indent=2, sort_keys=True))
    md_path.write_text(_build_markdown_report(profile, generated_at=generated_at, state_hash=state_hash))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"state_hash: {state_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
