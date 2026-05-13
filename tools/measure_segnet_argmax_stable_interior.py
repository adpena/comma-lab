#!/usr/bin/env python
"""φ1 SABOR boundary audit — SegNet argmax-stable interior measurement.

First-principles hypothesis (Grand Council 2026-05-13):
    SegNet output is argmax(logits) over 5 classes — only LOGIT ORDERING
    matters, not magnitudes. Pixels INTERIOR to large argmax-stable regions
    are "free bytes" — their RGB values can be perturbed without changing
    SegNet output.

Question: What fraction of pixels on the contest video's 600 last-pair frames
are stable under RGB perturbations of amplitude ε ∈ {1, 2, 4, 8, 16, 32}?

If ≥50% are stable: substantial free-byte capacity for SABOR substrate.
If <10%: SABOR is dominated by other approaches.

Method (two measurements, both [macOS-CPU advisory] per CLAUDE.md axis discipline):

  (A) Logit-margin proxy (cheap; all 600 frames):
      For each pixel, margin = logit[argmax] - logit[second_argmax]. Larger
      margin = more robust to perturbation. We report margin-quantile maps
      and stable-fraction(margin > threshold) for several thresholds. This
      is a per-frame closed-form measurement — no extra SegNet forwards.

  (B) Empirical perturbation (faithful; stratified subset of 60 frames):
      For each ε, add iid uniform noise of amplitude ε to RGB, re-run SegNet,
      measure argmax-disagreement-rate. K=2 perturbations per ε.
      Stratification: every 10th frame of the 600. Reports both expected
      stable_pixel_fraction(ε) across perturbations and stricter
      all-samples-stable fraction for capacity estimates.

Per-class breakdown + spatial-distribution (cluster vs scattered) reported
for both measurements.

CLAUDE.md compliance:
  - [macOS-CPU advisory] tags on every score-related claim (per Catalog #192).
  - No /tmp paths (artifacts under experiments/results/lane_sabor_*/).
  - No KILL verdicts.
  - Strict-scorer-rule respected: scorer is for AUDIT-TIME measurement only;
    NEVER part of an inflate.sh runtime.
  - $0 GPU spend.

Outputs:
  experiments/results/lane_sabor_boundary_audit_20260513_<UTC>/
    stable_pixel_capacity.json       — machine-readable per-frame + aggregate stats
    margin_quantile_summary.json     — logit-margin distribution per frame
    per_class_breakdown.json         — fraction of pixels in each of 5 classes
    spatial_distribution.json        — stable-region cluster statistics
    sample_argmax_frame_000.npy      — saved argmax maps for spot-check (10 frames)
    sample_margin_frame_000.npy      — saved margin maps for spot-check (10 frames)

Free-byte-capacity estimate:
    A "stable interior" pixel can be replaced with ANY of 256 R, 256 G, 256 B
    values without affecting SegNet output. If the model class for byte-
    stuffing uses a uniform-fill (gray + small dither), capacity per stable
    pixel is reported conservatively as 1 bit/pixel, moderately as
    log2(2ε) bits on one channel, and aggressively as 3*log2(2ε) bits with
    all channels independent.

Usage:
    .venv/bin/python tools/measure_segnet_argmax_stable_interior.py
        [--video-path upstream/videos/0.mkv]
        [--output-dir experiments/results/lane_sabor_boundary_audit_<UTC>]
        [--n-frames 600]
        [--n-perturbation-samples 2]
        [--perturbation-subset-stride 10]
        [--margin-thresholds 0.5,1.0,2.0,4.0,8.0,16.0]
        [--epsilon-list 1,2,4,8,16,32]
        [--num-threads 4]
        [--seed 17]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "upstream"))

# Upstream scorer modules
from frame_utils import camera_size, segnet_model_input_size, seq_len  # noqa: E402
from modules import SegNet  # noqa: E402  (path-injection)
from safetensors.torch import load_file  # noqa: E402

# 5-class label mapping for SegNet's 5-class output. The exact mapping is
# determined by the SegNet checkpoint training labels; we use generic
# class_<n> identifiers in the report to avoid mislabeling. Empirically the
# largest-area class on the contest video maps to road/drivable surface;
# next-largest typically maps to lane-markings or undrivable.
CLASS_NAMES = ["class_0", "class_1", "class_2", "class_3", "class_4"]


def _utc_stamp() -> str:
    return _dt.datetime.now(tz=_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(_json_safe(obj), indent=2, allow_nan=False), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-path", default="upstream/videos/0.mkv", help="contest video path")
    p.add_argument("--output-dir", default=None,
                   help="output directory (default: experiments/results/lane_sabor_boundary_audit_<UTC>)")
    p.add_argument("--n-frames", type=int, default=600,
                   help="number of last-pair frames to process (max 600)")
    p.add_argument("--n-perturbation-samples", type=int, default=2,
                   help="K random perturbations per epsilon (empirical measurement)")
    p.add_argument("--perturbation-subset-stride", type=int, default=10,
                   help="frame stride for empirical perturbation subset (10 → 60 frames)")
    p.add_argument("--margin-thresholds", default="0.5,1.0,2.0,4.0,8.0,16.0",
                   help="logit-margin thresholds for proxy stable-fraction")
    p.add_argument("--epsilon-list", default="1,2,4,8,16,32",
                   help="RGB perturbation amplitudes (uint8 units) for empirical measurement")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--save-spot-check-frames", type=int, default=10,
                   help="number of argmax/margin frames to save to .npy")
    return p


# ---------- video iterator -------------------------------------------------

def _pair_hwc_to_tchw(pair_hwc: list[torch.Tensor]) -> torch.Tensor:
    """Convert one upstream AV pair from HWC uint8 frames to TCHW float RGB.

    This mirrors ``upstream.frame_utils.AVVideoDataset`` before
    ``DistortionNet.preprocess_input`` rearranges ``B,T,H,W,C`` to
    ``B,T,C,H,W``. Keeping the 2-frame pair intact avoids bypassing
    ``SegNet.preprocess_input`` and protects the non-overlapping pair contract.
    """
    if len(pair_hwc) != seq_len:
        raise ValueError(f"expected {seq_len} frames per pair, got {len(pair_hwc)}")
    for i, frame in enumerate(pair_hwc):
        if frame.dtype != torch.uint8:
            raise TypeError(f"decoded frame {i} must be uint8, got {frame.dtype}")
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"decoded frame {i} must be HWC RGB, got {tuple(frame.shape)}")
        expected_hw = (camera_size[1], camera_size[0])
        if tuple(frame.shape[:2]) != expected_hw:
            raise ValueError(
                f"decoded frame {i} shape {tuple(frame.shape[:2])} != "
                f"upstream camera HxW {expected_hw}"
            )
    return torch.stack(pair_hwc, dim=0).permute(0, 3, 1, 2).float().contiguous()


def _decode_video_pairs_rgb_tchw(video_path: Path, n_pairs: int):
    """Decode video and yield non-overlapping RGB pairs as ``(idx, T,C,H,W)``.

    ``upstream.evaluate`` uses ``AVVideoDataset`` on CPU/macOS, which groups
    consecutive frames into non-overlapping pairs and sends tensors through
    ``DistortionNet`` as ``B,T,H,W,C`` uint8. This iterator reproduces that
    pairing while returning a float ``T,C,H,W`` pair for local scorer calls.
    """
    import av
    from frame_utils import yuv420_to_rgb

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    pair_buf = []
    n_yielded = 0
    pair_idx = 0
    for frame in container.decode(stream):
        rgb_hwc = yuv420_to_rgb(frame)  # (H, W, 3) uint8
        pair_buf.append(rgb_hwc)
        if len(pair_buf) == seq_len:
            yield pair_idx, _pair_hwc_to_tchw(pair_buf)
            pair_idx += 1
            n_yielded += 1
            pair_buf = []
            if n_yielded >= n_pairs:
                break
    container.close()


# ---------- SegNet inference helpers ---------------------------------------

@torch.inference_mode()
def _segnet_logits_from_pair(segnet: torch.nn.Module, rgb_pair_tchw: torch.Tensor) -> torch.Tensor:
    """Run SegNet through the upstream 5D pair preprocessing contract.

    ``rgb_pair_tchw`` is one non-overlapping pair with shape ``(2,3,874,1164)``
    in uint8 RGB value range, stored as float for perturbations. Returns logits
    with shape ``(5,384,512)`` on CPU.
    """
    if rgb_pair_tchw.ndim != 4 or tuple(rgb_pair_tchw.shape[:2]) != (seq_len, 3):
        raise ValueError(f"expected rgb_pair_tchw=(2,3,H,W), got {tuple(rgb_pair_tchw.shape)}")
    x_pair = rgb_pair_tchw.unsqueeze(0)  # (1, 2, 3, H, W)
    x_seg = segnet.preprocess_input(x_pair)
    expected_hw = (segnet_model_input_size[1], segnet_model_input_size[0])
    if x_seg.ndim != 4 or tuple(x_seg.shape) != (1, 3, *expected_hw):
        raise RuntimeError(
            f"SegNet.preprocess_input contract drift: expected (1,3,{expected_hw[0]},{expected_hw[1]}), "
            f"got {tuple(x_seg.shape)}"
        )
    out = segnet(x_seg)
    if out.ndim != 4 or out.shape[0] != 1 or out.shape[1] != 5:
        raise RuntimeError(f"SegNet forward contract drift: expected (1,5,H,W), got {tuple(out.shape)}")
    return out[0].contiguous()  # (5, 384, 512)


def _logit_margin(logits: torch.Tensor) -> torch.Tensor:
    """logits: (5, H, W) → margin: (H, W). Margin = logit[top1] - logit[top2]."""
    top2 = logits.topk(2, dim=0).values  # (2, H, W)
    return (top2[0] - top2[1]).contiguous()


def _argmax_map(logits: torch.Tensor) -> torch.Tensor:
    """logits: (5, H, W) → argmax: (H, W) uint8."""
    return logits.argmax(dim=0).to(torch.uint8)


# ---------- per-frame metric collection -----------------------------------

def _per_class_fraction(argmax_hw: torch.Tensor, n_classes: int = 5) -> list[float]:
    total = float(argmax_hw.numel())
    out = []
    for c in range(n_classes):
        cnt = float((argmax_hw == c).sum().item())
        out.append(cnt / total)
    return out


def _per_class_stable_fraction(argmax_hw: torch.Tensor, stable_mask: torch.Tensor,
                                n_classes: int = 5) -> list[float]:
    """Of pixels in class c, what fraction are stable?"""
    out = []
    for c in range(n_classes):
        in_class = (argmax_hw == c)
        denom = float(in_class.sum().item())
        if denom == 0.0:
            out.append(float("nan"))
            continue
        stable_in_class = float((in_class & stable_mask).sum().item())
        out.append(stable_in_class / denom)
    return out


def _spatial_cluster_stats(stable_mask: torch.Tensor) -> dict:
    """Heuristic: compute (a) total stable fraction, (b) "interior fraction"
    via morphological erosion proxy (a pixel is "deeply interior" iff all 4
    nearest neighbors are also stable), and (c) "scattered" fraction
    (stable but at least one neighbor unstable → fringe).
    """
    m = stable_mask.bool()  # (H, W)
    H, W = m.shape
    pad = torch.zeros(H + 2, W + 2, dtype=torch.bool)
    pad[1:-1, 1:-1] = m
    interior = (
        pad[1:-1, 1:-1]
        & pad[:-2, 1:-1]   # up
        & pad[2:, 1:-1]    # down
        & pad[1:-1, :-2]   # left
        & pad[1:-1, 2:]    # right
    )
    total = float(m.numel())
    return {
        "stable_fraction": float(m.sum().item()) / total,
        "interior_fraction": float(interior.sum().item()) / total,
        "fringe_fraction": float((m & ~interior).sum().item()) / total,
    }


def _empirical_stability_from_masks(
    argmax_hw: torch.Tensor,
    stable_masks: list[torch.Tensor],
    n_classes: int = 5,
) -> dict:
    """Summarize empirical argmax stability over one or more perturbations.

    The mean-per-perturbation fraction estimates expected stability under the
    sampled noise distribution. The all-samples fraction is stricter and is the
    capacity-driving number because byte stuffing needs a pixel to survive all
    sampled perturbation probes, not merely most of them.
    """
    if not stable_masks:
        raise ValueError("at least one stable mask is required")
    stable_all = torch.ones_like(argmax_hw, dtype=torch.bool)
    stable_observation_count = 0.0
    total_observation_count = 0.0
    per_class_stable_counts = [0.0] * n_classes
    per_class_total_counts = [0.0] * n_classes

    for stable in stable_masks:
        if stable.shape != argmax_hw.shape:
            raise ValueError(f"stable mask shape {tuple(stable.shape)} != argmax shape {tuple(argmax_hw.shape)}")
        stable_bool = stable.bool()
        stable_all &= stable_bool
        stable_observation_count += float(stable_bool.sum().item())
        total_observation_count += float(stable_bool.numel())
        for c in range(n_classes):
            in_class = (argmax_hw == c)
            denom = float(in_class.sum().item())
            if denom == 0.0:
                continue
            per_class_stable_counts[c] += float((in_class & stable_bool).sum().item())
            per_class_total_counts[c] += denom

    all_samples_stable_count = float(stable_all.sum().item())
    pixel_count = float(argmax_hw.numel())
    per_class_stable_fraction = [
        (
            per_class_stable_counts[c] / per_class_total_counts[c]
            if per_class_total_counts[c] > 0.0 else float("nan")
        )
        for c in range(n_classes)
    ]
    cluster_stats = _spatial_cluster_stats(stable_all)
    return {
        "K": len(stable_masks),
        "stable_fraction_mean_per_perturbation": stable_observation_count / total_observation_count,
        "stable_fraction_all_samples": all_samples_stable_count / pixel_count,
        "interior_fraction_all_samples": cluster_stats["interior_fraction"],
        "fringe_fraction_all_samples": cluster_stats["fringe_fraction"],
        "per_class_stable_fraction_mean_per_perturbation": per_class_stable_fraction,
        "_stable_observation_count": stable_observation_count,
        "_total_observation_count": total_observation_count,
        "_all_samples_stable_count": all_samples_stable_count,
        "_pixel_count": pixel_count,
        "_per_class_stable_counts": per_class_stable_counts,
        "_per_class_total_counts": per_class_total_counts,
    }


# ---------- main routine ---------------------------------------------------

def main():
    args = _build_arg_parser().parse_args()
    if int(args.n_frames) <= 0:
        raise SystemExit("--n-frames must be positive")
    if int(args.n_perturbation_samples) <= 0:
        raise SystemExit("--n-perturbation-samples must be positive")
    if int(args.perturbation_subset_stride) <= 0:
        raise SystemExit("--perturbation-subset-stride must be positive")
    torch.set_num_threads(int(args.num_threads))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    run_started_at_utc = _dt.datetime.now(tz=_dt.UTC).isoformat()

    margin_thresholds = [float(s) for s in args.margin_thresholds.split(",") if s.strip()]
    epsilon_list = [float(s) for s in args.epsilon_list.split(",") if s.strip()]

    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "experiments" / "results" /
        f"lane_sabor_boundary_audit_20260513_{_utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = (REPO_ROOT / args.video_path).resolve()
    assert video_path.exists(), f"video missing: {video_path}"

    # Load SegNet on CPU
    device = torch.device("cpu")
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(REPO_ROOT / "upstream" / "models" / "segnet.safetensors"),
                                      device="cpu"))

    H_RES, W_RES = segnet_model_input_size[1], segnet_model_input_size[0]
    n_classes = 5
    n_frames = int(args.n_frames)
    perturbation_subset_stride = int(args.perturbation_subset_stride)
    save_spot_check = int(args.save_spot_check_frames)

    perturbation_subset_idx = set(range(0, n_frames, perturbation_subset_stride))

    print(f"[sabor-audit] video={video_path}")
    print(f"[sabor-audit] output={output_dir}")
    print(f"[sabor-audit] n_frames={n_frames} subset_stride={perturbation_subset_stride}"
          f" → {len(perturbation_subset_idx)} empirical-perturbation frames")
    print(f"[sabor-audit] margin_thresholds={margin_thresholds}")
    print(f"[sabor-audit] epsilon_list={epsilon_list} (RGB uint8 units)")
    print(f"[sabor-audit] K perturbation samples per ε = {args.n_perturbation_samples}")
    print("[sabor-audit] tag: [macOS-CPU advisory] only — proxy for boundary-only-renderer feasibility")
    print()

    # Accumulators
    per_frame_records = []
    # Aggregate stable fraction over all frames per (margin_threshold) and per (epsilon)
    agg_margin_stable_counts = dict.fromkeys(margin_thresholds, 0)
    agg_margin_stable_total = 0
    agg_class_pixel_counts = [0] * n_classes  # over all clean argmax maps
    agg_pixel_total = 0
    agg_epsilon_stable_observation_counts = dict.fromkeys(epsilon_list, 0.0)
    agg_epsilon_observation_totals = dict.fromkeys(epsilon_list, 0.0)
    agg_epsilon_all_samples_stable_counts = dict.fromkeys(epsilon_list, 0.0)
    agg_epsilon_pixel_totals = dict.fromkeys(epsilon_list, 0.0)
    agg_epsilon_per_class_stable_counts = {e: [0.0] * n_classes for e in epsilon_list}
    agg_epsilon_per_class_total_counts = {e: [0.0] * n_classes for e in epsilon_list}
    agg_epsilon_frames_measured = dict.fromkeys(epsilon_list, 0)

    t_start = time.time()

    for pair_idx, rgb_pair_tchw in _decode_video_pairs_rgb_tchw(video_path, n_frames):
        t_frame_start = time.time()

        # rgb_pair_tchw shape: (T=2, C=3, H=874, W=1164) float in [0, 255]
        rgb_clean_pair = rgb_pair_tchw.clamp(0.0, 255.0)

        # Clean forward — runs through SegNet.preprocess_input (last-frame + bilinear)
        logits_clean = _segnet_logits_from_pair(segnet, rgb_clean_pair)
        argmax_clean = _argmax_map(logits_clean)  # (H, W) uint8
        margin = _logit_margin(logits_clean)  # (H, W) float

        # Margin-proxy stable masks
        margin_records = {}
        for thr in margin_thresholds:
            stable_mask = (margin > thr)
            cluster_stats = _spatial_cluster_stats(stable_mask)
            per_class_stable = _per_class_stable_fraction(argmax_clean, stable_mask, n_classes)
            margin_records[thr] = {
                "stable_fraction": cluster_stats["stable_fraction"],
                "interior_fraction": cluster_stats["interior_fraction"],
                "fringe_fraction": cluster_stats["fringe_fraction"],
                "per_class_stable_fraction": per_class_stable,
            }
            agg_margin_stable_counts[thr] += int(stable_mask.sum().item())

        agg_margin_stable_total += int(margin.numel())

        # Per-class breakdown on clean argmax
        per_class = _per_class_fraction(argmax_clean, n_classes)
        for c in range(n_classes):
            agg_class_pixel_counts[c] += int((argmax_clean == c).sum().item())
        agg_pixel_total += int(argmax_clean.numel())

        # Margin distribution summary
        margin_flat = margin.flatten()
        q_p = torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        q_vals = torch.quantile(margin_flat, q_p).tolist()
        margin_quantiles = dict(
            zip([f"q{int(p * 100):02d}" for p in q_p.tolist()], q_vals, strict=True)
        )

        # Empirical perturbation measurement (only on subset frames)
        eps_records = {}
        if pair_idx in perturbation_subset_idx:
            for eps in epsilon_list:
                K = int(args.n_perturbation_samples)
                stable_masks = []
                for _k in range(K):
                    # iid uniform noise in [-eps, +eps] on the last frame.
                    # SegNet ignores the first frame by contract, but preserving
                    # the pair shape catches accidental 3D/4D shortcut regressions.
                    rgb_pert_pair = rgb_clean_pair.clone()
                    noise = (torch.rand_like(rgb_pert_pair[-1]) * 2.0 - 1.0) * eps
                    rgb_pert_pair[-1] = (rgb_pert_pair[-1] + noise).clamp(0.0, 255.0)
                    logits_pert = _segnet_logits_from_pair(segnet, rgb_pert_pair)
                    argmax_pert = _argmax_map(logits_pert)
                    stable_masks.append(argmax_pert == argmax_clean)
                eps_record_internal = _empirical_stability_from_masks(argmax_clean, stable_masks, n_classes)
                eps_records[eps] = {
                    k: v for k, v in eps_record_internal.items()
                    if not k.startswith("_")
                }
                agg_epsilon_stable_observation_counts[eps] += eps_record_internal["_stable_observation_count"]
                agg_epsilon_observation_totals[eps] += eps_record_internal["_total_observation_count"]
                agg_epsilon_all_samples_stable_counts[eps] += eps_record_internal["_all_samples_stable_count"]
                agg_epsilon_pixel_totals[eps] += eps_record_internal["_pixel_count"]
                for c in range(n_classes):
                    agg_epsilon_per_class_stable_counts[eps][c] += eps_record_internal["_per_class_stable_counts"][c]
                    agg_epsilon_per_class_total_counts[eps][c] += eps_record_internal["_per_class_total_counts"][c]
                agg_epsilon_frames_measured[eps] += 1

        # Save spot-check artifacts for first N frames
        if pair_idx < save_spot_check:
            np.save(output_dir / f"sample_argmax_frame_{pair_idx:03d}.npy",
                    argmax_clean.cpu().numpy().astype(np.uint8))
            np.save(output_dir / f"sample_margin_frame_{pair_idx:03d}.npy",
                    margin.cpu().numpy().astype(np.float32))

        per_frame_records.append({
            "pair_idx": pair_idx,
            "per_class_fraction": per_class,
            "margin_quantiles": margin_quantiles,
            "margin_records": {str(k): v for k, v in margin_records.items()},
            "empirical_perturbation_records": ({str(k): v for k, v in eps_records.items()}
                                                if eps_records else None),
            "wall_clock_sec": time.time() - t_frame_start,
        })

        if (pair_idx + 1) % 50 == 0 or pair_idx < 5:
            elapsed = time.time() - t_start
            rate = (pair_idx + 1) / elapsed
            eta = (n_frames - pair_idx - 1) / max(rate, 1e-6)
            print(f"[sabor-audit] frame {pair_idx + 1}/{n_frames}  "
                  f"elapsed={elapsed:.0f}s  rate={rate:.2f} fps  eta={eta:.0f}s")

    # Aggregate results
    print(f"\n[sabor-audit] done: {len(per_frame_records)} frames in {time.time() - t_start:.0f}s")

    aggregate_margin = {}
    for thr in margin_thresholds:
        agg = (
            float("nan")
            if agg_margin_stable_total == 0
            else agg_margin_stable_counts[thr] / agg_margin_stable_total
        )
        aggregate_margin[str(thr)] = agg

    aggregate_class_dist = {
        CLASS_NAMES[c]: (agg_class_pixel_counts[c] / max(agg_pixel_total, 1))
        for c in range(n_classes)
    }

    aggregate_epsilon = {}
    for eps in epsilon_list:
        obs_denom = agg_epsilon_observation_totals[eps]
        pixel_denom = agg_epsilon_pixel_totals[eps]
        if obs_denom == 0 or pixel_denom == 0:
            mean_stable_frac = float("nan")
            all_samples_stable_frac = float("nan")
        else:
            mean_stable_frac = agg_epsilon_stable_observation_counts[eps] / obs_denom
            all_samples_stable_frac = agg_epsilon_all_samples_stable_counts[eps] / pixel_denom
        n_count = agg_epsilon_frames_measured[eps]
        per_class_avg = []
        for c in range(n_classes):
            class_denom = agg_epsilon_per_class_total_counts[eps][c]
            per_class_avg.append(
                agg_epsilon_per_class_stable_counts[eps][c] / class_denom
                if class_denom > 0.0 else float("nan")
            )
        aggregate_epsilon[str(eps)] = {
            "stable_fraction": all_samples_stable_frac,
            "stable_fraction_all_samples": all_samples_stable_frac,
            "stable_fraction_mean_per_perturbation": mean_stable_frac,
            "n_frames_measured": n_count,
            "per_class_stable_fraction": dict(zip(CLASS_NAMES, per_class_avg, strict=True)),
        }

    # Free-byte capacity estimates per ε
    # Conservative: 1 channel x 1 bit per stable pixel (replace with mid-gray +/- dither)
    # Aggressive: 3 channels x 8 bits per stable pixel (independent RGB perturbation tolerated)
    h, w = H_RES, W_RES
    pixels_per_frame = h * w
    free_byte_capacity_per_frame = {}
    for eps in epsilon_list:
        stable_frac = aggregate_epsilon[str(eps)]["stable_fraction"]
        if stable_frac != stable_frac:  # nan
            free_byte_capacity_per_frame[str(eps)] = {
                "conservative_bytes_per_frame_1ch_1bit": float("nan"),
                "moderate_bytes_per_frame_1ch_log2_2eps": float("nan"),
                "aggressive_bytes_per_frame_3ch_log2_2eps": float("nan"),
            }
            continue
        stable_pixels = stable_frac * pixels_per_frame
        # Conservative: each stable pixel can carry 1 bit (sign flip on the L
        # channel) without changing argmax. 1 bit per pixel → 1/8 byte/pixel.
        cons = stable_pixels * (1.0 / 8.0)
        # Moderate: each stable pixel can carry log2(2ε) bits on ONE channel.
        # 2ε levels (the perturbation range we measured robust to).
        levels = max(2 * eps, 2.0)
        bits_per_pixel_1ch = float(np.log2(levels))
        mod = stable_pixels * bits_per_pixel_1ch / 8.0
        # Aggressive: 3 channels independent.
        agg_bytes = stable_pixels * 3.0 * bits_per_pixel_1ch / 8.0
        free_byte_capacity_per_frame[str(eps)] = {
            "stable_pixels": float(stable_pixels),
            "pixels_per_frame": float(pixels_per_frame),
            "stable_fraction": float(stable_frac),
            "conservative_bytes_per_frame_1ch_1bit": float(cons),
            "moderate_bytes_per_frame_1ch_log2_2eps": float(mod),
            "aggressive_bytes_per_frame_3ch_log2_2eps": float(agg_bytes),
        }

    aggregate_record = {
        "schema_version": "sabor_boundary_audit_v1",
        "evidence_grade": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "research_only": True,
        "lane_id": "lane_sabor_boundary_audit_20260513",
        "lane_class": "research_substrate_audit",
        "target_modes": ["research_substrate"],
        "deployment_target": "local_macos_cpu_advisory_only",
        "video_path": str(video_path),
        "video_sha256": _sha256_path(video_path),
        "segnet_safetensors_sha256": _sha256_path(REPO_ROOT / "upstream" / "models" / "segnet.safetensors"),
        "n_frames": len(per_frame_records),
        "resized_HxW": [H_RES, W_RES],
        "pixels_per_frame": H_RES * W_RES,
        "margin_thresholds": margin_thresholds,
        "epsilon_list": epsilon_list,
        "n_perturbation_samples_per_eps_K": int(args.n_perturbation_samples),
        "perturbation_subset_stride": perturbation_subset_stride,
        "perturbation_subset_n_frames_requested": len(perturbation_subset_idx),
        "perturbation_subset_n_frames_measured": max(agg_epsilon_frames_measured.values()) if epsilon_list else 0,
        "aggregate_margin_proxy_stable_fraction": aggregate_margin,
        "aggregate_clean_class_distribution": aggregate_class_dist,
        "aggregate_epsilon_empirical_stable_fraction": aggregate_epsilon,
        "free_byte_capacity_per_frame": free_byte_capacity_per_frame,
        "wall_clock_seconds_total": time.time() - t_start,
        "torch_version": torch.__version__,
        "torch_num_threads": int(args.num_threads),
        "seed": int(args.seed),
        "utc_start": run_started_at_utc,
        "utc_end": _dt.datetime.now(tz=_dt.UTC).isoformat(),
        # Wire-in declarations per CLAUDE.md Catalog #125 (subagent coherence-by-default):
        "wire_in_hooks": {
            "sensitivity_map": "N/A — audit is per-pixel argmax-stability scan; the output JSON IS the sensitivity map for SABOR substrate construction",
            "pareto_constraint": "N/A — research-only audit, no Pareto-binding edge until SABOR substrate built",
            "bit_allocator": "N/A — pre-substrate audit; bit-allocator wires into the future SABOR codec, not the audit",
            "cathedral_autopilot_dispatch_hook": "N/A — research-only macOS-CPU advisory; no dispatch artifact",
            "continual_learning_posterior_update": "N/A — no empirical anchor (advisory-only)",
            "probe_disambiguator": "N/A — single hypothesis (boundary-only-renderer-byte-capacity); empirical result is the disambiguator itself",
        },
    }

    # Write all output JSONs
    _write_json(output_dir / "stable_pixel_capacity.json", aggregate_record)
    _write_json(output_dir / "per_frame_records.json", {"records": per_frame_records})

    margin_quantile_summary = [r["margin_quantiles"] for r in per_frame_records]
    _write_json(
        output_dir / "margin_quantile_summary.json",
        {"per_frame_margin_quantiles": margin_quantile_summary},
    )

    _write_json(output_dir / "per_class_breakdown.json", {
        "aggregate_clean_class_distribution": aggregate_class_dist,
        "per_frame_class_fraction": [r["per_class_fraction"] for r in per_frame_records],
        "aggregate_per_eps_per_class_stable_fraction": {
            str(eps): aggregate_epsilon[str(eps)]["per_class_stable_fraction"]
            for eps in epsilon_list
        },
    })

    _write_json(output_dir / "spatial_distribution.json", {
        "per_frame_margin_records": [r["margin_records"] for r in per_frame_records],
        "per_frame_empirical_records": [r["empirical_perturbation_records"]
                                          for r in per_frame_records
                                          if r["empirical_perturbation_records"] is not None],
    })

    # Build_manifest.json per CLAUDE.md custody contract
    _write_json(output_dir / "build_manifest.json", {
        "lane_id": "lane_sabor_boundary_audit_20260513",
        "target_modes": ["research_substrate"],
        "custody_status": "transient-allowed",
        "deployment_target": "local_macos_cpu_advisory_only",
        "evidence_grade": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "research_only": True,
        "generated_at": _dt.datetime.now(tz=_dt.UTC).isoformat(),
        "video_path": str(video_path),
        "video_sha256": _sha256_path(video_path),
        "segnet_safetensors_sha256": _sha256_path(REPO_ROOT / "upstream" / "models" / "segnet.safetensors"),
        "n_frames": len(per_frame_records),
        "n_perturbation_frames_requested": len(perturbation_subset_idx),
        "n_perturbation_frames_measured": max(agg_epsilon_frames_measured.values()) if epsilon_list else 0,
        "blockers": [
            "no_byte_closed_runtime_packet_built",
            "research_audit_only_no_dispatch_required",
        ],
        "cleared_blockers": [],
        "next_required_actions": [
            "operator + council decision: proceed to SABOR substrate build OR deprioritize per stable-fraction verdict",
        ],
    })

    # Print summary table
    print(f"\n{'=' * 72}")
    print("AGGREGATE SUMMARY  [macOS-CPU advisory]")
    print(f"{'=' * 72}")
    print(f"Frames measured:                {len(per_frame_records)}")
    print(f"Perturbation subset:            {max(agg_epsilon_frames_measured.values()) if epsilon_list else 0} frames "
          f"(every {perturbation_subset_stride}th)")
    print()
    print("CLEAN CLASS DISTRIBUTION (fraction of pixels per class):")
    for name, frac in aggregate_class_dist.items():
        print(f"  {name:>16s}: {frac:6.3f}")
    print()
    print("LOGIT-MARGIN PROXY STABLE FRACTION (margin > threshold):")
    for thr in margin_thresholds:
        print(f"  margin > {thr:5.2f}: {aggregate_margin[str(thr)]:6.3f}")
    print()
    print("EMPIRICAL ε-STABLE FRACTION (uniform iid RGB noise in [-ε, +ε]):")
    for eps in epsilon_list:
        rec = aggregate_epsilon[str(eps)]
        print(f"  ε={eps:5.1f}:  all_samples={rec['stable_fraction_all_samples']:6.3f}  "
              f"mean_per_perturb={rec['stable_fraction_mean_per_perturbation']:6.3f}  "
              f"(n_frames={rec['n_frames_measured']})")
    print()
    print("FREE-BYTE CAPACITY ESTIMATES PER FRAME (at each ε):")
    for eps in epsilon_list:
        rec = free_byte_capacity_per_frame[str(eps)]
        if rec["conservative_bytes_per_frame_1ch_1bit"] != rec["conservative_bytes_per_frame_1ch_1bit"]:
            continue  # nan
        print(f"  ε={eps:5.1f}:  conservative={rec['conservative_bytes_per_frame_1ch_1bit']:8.0f}B  "
              f"moderate={rec['moderate_bytes_per_frame_1ch_log2_2eps']:8.0f}B  "
              f"aggressive={rec['aggressive_bytes_per_frame_3ch_log2_2eps']:8.0f}B")
    print(f"\nArtifacts: {output_dir}")
    print()


if __name__ == "__main__":
    main()
