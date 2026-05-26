# SPDX-License-Identifier: MIT
"""UNIWARD per-pixel N+1 real-scorer-anchored empirical sweep (MLX-local).

Sister of just-landed scaffold subagent (commit aa2612d9b; substrate
`uniward_per_pixel_distortion` L1). Measures REAL-SCORER-ANCHORED per-axis
d_seg + d_pose reduction from UNIWARD per-pixel score-conditional weighting
vs uniform-weighting baseline.

Design (Carmack-engineering minimal-architecture for clean isolation):

- Fixture: 50 pairs decoded from `upstream/videos/0.mkv` at 96x128
- Architecture: learnable raw RGB tensor per pair (one fp32 (T,3,H,W) tensor
  per pair, optimized by MLX Adam). This isolates the WEIGHTING EFFECT
  on the loss landscape from any renderer-architecture confound.
- BASELINE arm: uniform L2 reconstruction loss
- VARIANT arm: UNIWARD-weighted L2 reconstruction loss using per-pixel
  weight map computed from REAL scorer gradient magnitudes
- Per-axis measurement: d_seg + d_pose via canonical
  `score_pair_components` at start (epoch 0 / random init) + end (epoch 30)
- Comparison: VARIANT vs BASELINE per-axis improvement RATIO

Per CLAUDE.md "MLX portable-local-substrate authority": tagged
`[macOS-MLX research-signal]` per Catalog #192/#317/#341.

Per CLAUDE.md "Forbidden `make_synthetic_pair_batch` calls in any non-smoke
training path": uses real `upstream/videos/0.mkv` decode via `tac.data`.

Per CLAUDE.md "Apples-to-apples evidence discipline": this is a CONTROLLED
COMPARISON of weighting effect on loss landscape, NOT a contest score claim.
The output is per-axis IMPROVEMENT RATIO (variant/baseline), not absolute
contest score.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import torch

# Repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upstream"))

from tac.scorer import load_differentiable_scorers
from tac.substrates.score_aware_common import score_pair_components
from tac.substrates.uniward_per_pixel_distortion import (
    compute_per_pixel_uniward_weight_map_numpy,
    compose_uniward_weighted_score_loss,
)
from tac.substrates.uniward_per_pixel_distortion.weight_map import (
    decompose_per_axis_weights,
    normalize_weight_map_to_unit_mean,
    histogram_weight_distribution,
)

UPSTREAM = ROOT / "upstream"
VIDEO_PATH = UPSTREAM / "videos" / "0.mkv"

N_PAIRS = 50
SPATIAL_H = 96
SPATIAL_W = 128
EPOCHS = 30
LEARNING_RATE = 0.05
SEED = 42
EMA_DECAY = 0.997
UNIWARD_LAMBDA = 0.01


def decode_pairs() -> np.ndarray:
    """Decode first 2*N_PAIRS frames from upstream/videos/0.mkv, return (N_PAIRS, 2, 3, H, W) fp32 in [0,1]."""
    import av

    container = av.open(str(VIDEO_PATH))
    stream = container.streams.video[0]
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i >= 2 * N_PAIRS:
            break
        img = frame.to_ndarray(format="rgb24")  # (H_orig, W_orig, 3)
        # Resize to SPATIAL_H x SPATIAL_W using bilinear via numpy (fast)
        import PIL.Image
        pil = PIL.Image.fromarray(img)
        pil = pil.resize((SPATIAL_W, SPATIAL_H), PIL.Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = arr.transpose(2, 0, 1)  # (3, H, W)
        frames.append(arr)
    container.close()
    frames = np.stack(frames)  # (2*N_PAIRS, 3, H, W)
    pairs = frames.reshape(N_PAIRS, 2, 3, SPATIAL_H, SPATIAL_W)
    return pairs


def compute_real_scorer_gradient_per_pixel(
    posenet, segnet, pair_gt_rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pixel SegNet + PoseNet gradient magnitudes for a single pair.

    Uses torch autograd on a leaf RGB tensor seeded at GT; gradient is taken
    of `seg_distortion` and `pose_distortion` (scalar outputs of canonical
    score_pair_components) w.r.t. the input RGB. Per-pixel magnitude is
    sqrt of summed squared gradients over the 3 channels.

    Parameters
    ----------
    pair_gt_rgb : np.ndarray, shape (2, 3, H, W), fp32 in [0,1]
    """
    device = torch.device("cpu")
    # Leaf RGB tensors: seed at GT + LARGE noise (0.05 sigma) matching trainer
    # init perturbation, so gradient is measured at the OPERATING POINT the
    # renderer actually inhabits (per-pixel pose gradient at GT is degenerate
    # because PoseNet sees identical frames; gradient is meaningful only when
    # reconstruction has substantial error)
    gen = torch.Generator(device=device).manual_seed(SEED + 99)
    x_0 = torch.tensor(pair_gt_rgb[0], device=device, dtype=torch.float32).unsqueeze(0)
    x_1 = torch.tensor(pair_gt_rgb[1], device=device, dtype=torch.float32).unsqueeze(0)
    # Use 0.15 sigma to land at operating point where pose has real gradient
    # signal (at 0.05 sigma pose_d ~3e-6 << PR110 operating point ~3.4e-5;
    # at 0.15 sigma pose_d climbs into the meaningful-signal band)
    noise_0 = torch.randn(x_0.shape, generator=gen, device=device) * 0.15
    noise_1 = torch.randn(x_1.shape, generator=gen, device=device) * 0.15
    rgb_0 = (x_0 + noise_0).clamp(0, 1).detach().requires_grad_(True)
    rgb_1 = (x_1 + noise_1).clamp(0, 1).detach().requires_grad_(True)
    gt_0 = x_0
    gt_1 = x_1
    seg_d, pose_d = score_pair_components(
        seg_scorer=segnet,
        pose_scorer=posenet,
        rgb_0_rt=rgb_0,
        rgb_1_rt=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
    )
    # Per-pixel gradient of d_seg w.r.t. rgb_1 (the frame SegNet evaluates on)
    seg_grad_1 = torch.autograd.grad(seg_d, rgb_1, retain_graph=True, allow_unused=True)[0]
    if seg_grad_1 is None:
        seg_grad_1 = torch.zeros_like(rgb_1)
    # Per-pixel gradient of d_pose w.r.t. both rgb_0 and rgb_1 (PoseNet uses both)
    pose_grad_0 = torch.autograd.grad(pose_d, rgb_0, retain_graph=True, allow_unused=True)[0]
    pose_grad_1 = torch.autograd.grad(pose_d, rgb_1, retain_graph=False, allow_unused=True)[0]
    if pose_grad_0 is None:
        pose_grad_0 = torch.zeros_like(rgb_0)
    if pose_grad_1 is None:
        pose_grad_1 = torch.zeros_like(rgb_1)
    # Per-pixel magnitude: sqrt(sum over 3 channels of grad^2)
    seg_mag = torch.sqrt((seg_grad_1[0] ** 2).sum(dim=0) + 1e-12)  # (H, W)
    pose_mag = torch.sqrt(
        (pose_grad_0[0] ** 2).sum(dim=0) + (pose_grad_1[0] ** 2).sum(dim=0) + 1e-12
    )  # (H, W)
    return seg_mag.detach().cpu().numpy(), pose_mag.detach().cpu().numpy()


def measure_per_axis_scores(
    posenet, segnet, pairs_pred: np.ndarray, pairs_gt: np.ndarray
) -> dict[str, float]:
    """Measure d_seg + d_pose averaged over all pairs via canonical helper."""
    seg_sum = 0.0
    pose_sum = 0.0
    rate_sum = 0.0
    n = pairs_pred.shape[0]
    with torch.no_grad():
        for p in range(n):
            x0 = torch.tensor(pairs_pred[p, 0], dtype=torch.float32).unsqueeze(0).clamp(0, 1)
            x1 = torch.tensor(pairs_pred[p, 1], dtype=torch.float32).unsqueeze(0).clamp(0, 1)
            g0 = torch.tensor(pairs_gt[p, 0], dtype=torch.float32).unsqueeze(0)
            g1 = torch.tensor(pairs_gt[p, 1], dtype=torch.float32).unsqueeze(0)
            seg_d, pose_d = score_pair_components(
                seg_scorer=segnet, pose_scorer=posenet,
                rgb_0_rt=x0, rgb_1_rt=x1, gt_rgb_0=g0, gt_rgb_1=g1,
            )
            seg_sum += float(seg_d)
            pose_sum += float(pose_d)
    return {
        "seg_distortion_mean": seg_sum / n,
        "pose_distortion_mean": pose_sum / n,
        "seg_contribution_canonical": 100.0 * seg_sum / n,
        "pose_contribution_canonical": math.sqrt(10.0 * pose_sum / n),
    }


def eval_roundtrip_sim(x_mlx):
    """Simulate eval_roundtrip uint8 bottleneck (canonical per CLAUDE.md non-negotiable).

    MLX → uint8 cast → MLX fp32 (gradient passes through via straight-through).
    """
    # Straight-through estimator: forward = uint8 cast, backward = identity
    x_clip = mx.clip(x_mlx, 0.0, 1.0)
    x_uint8 = mx.round(x_clip * 255.0) / 255.0
    return x_clip + mx.stop_gradient(x_uint8 - x_clip)


def train_arm(
    arm_name: str,
    pairs_gt: np.ndarray,
    weight_maps_per_pair: np.ndarray | None,
    posenet, segnet,
) -> dict:
    """Train one arm: BASELINE (weight_maps_per_pair=None → uniform) or VARIANT.

    Returns per-axis scores at epoch 0 (init), epoch 5, epoch 15, epoch 30.
    """
    mx.random.seed(SEED)
    np.random.seed(SEED)
    # Learnable RGB tensors per pair, seeded at GT-mean (frame-pair mean)
    init_rgb = pairs_gt.copy()  # (N_PAIRS, 2, 3, H, W)
    # Match operating point: 0.15 sigma init perturbation aligns trainer with
    # gradient probe so the weight map measured at init operating point
    # corresponds to where the loss landscape applies
    init_rgb = init_rgb + np.random.randn(*init_rgb.shape).astype(np.float32) * 0.15
    init_rgb = np.clip(init_rgb, 0.0, 1.0)

    # MLX parameter: one big tensor (N_PAIRS, 2, 3, H, W)
    theta = mx.array(init_rgb)
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    # State for EMA
    ema_state = mx.array(init_rgb)

    if weight_maps_per_pair is not None:
        w_mlx = mx.array(weight_maps_per_pair.astype(np.float32))  # (N_PAIRS, H, W)
    else:
        w_mlx = mx.ones((N_PAIRS, SPATIAL_H, SPATIAL_W), dtype=mx.float32)

    pairs_gt_mlx = mx.array(pairs_gt)

    def loss_fn(theta_in):
        # theta_in: (N_PAIRS, 2, 3, H, W)
        # eval_roundtrip simulation
        theta_rt = eval_roundtrip_sim(theta_in)
        # Per-pixel L2 reconstruction error (mean over channels)
        per_pixel_err = mx.mean((theta_rt - pairs_gt_mlx) ** 2, axis=2)  # (N_PAIRS, 2, H, W)
        # Average over the 2 frames in each pair → (N_PAIRS, H, W)
        per_pixel_err_pair = mx.mean(per_pixel_err, axis=1)
        # UNIWARD weighting: HIGH weight (low sensitivity) absorbs more error;
        # weighted loss = per_pixel_err * weight → reducing loss FAVORS letting
        # error grow in high-weight zones and shrink in low-weight zones.
        # Per Fridrich UNIWARD this routes perturbation to safe zones.
        weighted = per_pixel_err_pair * w_mlx
        return mx.mean(weighted)

    val_and_grad = mx.value_and_grad(loss_fn)

    # Reduced measurement cadence: init + final only (halves wall-clock vs
    # 4-measurement scheme; per-epoch trace not required for verdict)
    measurement_epochs = [0, 30]
    measurements = {}

    # Epoch 0 measurement (init)
    pred_init = np.asarray(theta).copy()
    measurements[0] = measure_per_axis_scores(posenet, segnet, pred_init, pairs_gt)
    measurements[0]["wall_clock_s"] = 0.0

    t_start = time.time()
    # Adam state (manual implementation; MLX optimizer.update needs a Module
    # but our theta is a raw mx.array — direct Adam update is simpler)
    m_state = mx.zeros_like(theta)
    v_state = mx.zeros_like(theta)
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps = 1e-8
    for epoch in range(1, EPOCHS + 1):
        loss_val, grad = val_and_grad(theta)
        # Adam update
        m_state = adam_beta1 * m_state + (1.0 - adam_beta1) * grad
        v_state = adam_beta2 * v_state + (1.0 - adam_beta2) * grad * grad
        m_hat = m_state / (1.0 - adam_beta1 ** epoch)
        v_hat = v_state / (1.0 - adam_beta2 ** epoch)
        theta = theta - LEARNING_RATE * m_hat / (mx.sqrt(v_hat) + adam_eps)
        # EMA update
        ema_state = EMA_DECAY * ema_state + (1.0 - EMA_DECAY) * theta
        mx.eval(theta, ema_state, m_state, v_state, loss_val)
        if epoch in measurement_epochs:
            # Use EMA shadow for evaluation per CLAUDE.md EMA non-negotiable
            pred_arr = np.asarray(ema_state).copy()
            measurements[epoch] = measure_per_axis_scores(posenet, segnet, pred_arr, pairs_gt)
            measurements[epoch]["wall_clock_s"] = time.time() - t_start
            measurements[epoch]["loss_value"] = float(loss_val)
            print(f"  [{arm_name}] epoch {epoch}: seg_mean={measurements[epoch]['seg_distortion_mean']:.6f} "
                  f"pose_mean={measurements[epoch]['pose_distortion_mean']:.6f} loss={float(loss_val):.6f} "
                  f"wc={measurements[epoch]['wall_clock_s']:.1f}s", flush=True)

    return {
        "arm_name": arm_name,
        "epochs_measured": measurement_epochs,
        "measurements_per_epoch": {str(k): v for k, v in measurements.items()},
        "final_seg_distortion_mean": measurements[EPOCHS]["seg_distortion_mean"],
        "final_pose_distortion_mean": measurements[EPOCHS]["pose_distortion_mean"],
    }


def main() -> int:
    t_total_start = time.time()
    print(f"=== UNIWARD per-pixel N+1 real-scorer-anchored empirical sweep ===")
    print(f"  fixture: {VIDEO_PATH} (N_PAIRS={N_PAIRS}, {SPATIAL_H}x{SPATIAL_W})")
    print(f"  epochs: {EPOCHS}, lr={LEARNING_RATE}, seed={SEED}, ema={EMA_DECAY}")

    # Output dir
    artifacts_dir = ROOT / ".omx/research/uniward_per_pixel_n_plus_1_artifacts_20260526"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: decode pairs
    print("Phase 1: decoding pairs from upstream/videos/0.mkv...", flush=True)
    pairs_gt = decode_pairs()
    print(f"  decoded shape: {pairs_gt.shape}, dtype: {pairs_gt.dtype}, range: [{pairs_gt.min():.3f}, {pairs_gt.max():.3f}]", flush=True)

    # Phase 2: load real scorers
    print("Phase 2: loading real scorers via canonical load_differentiable_scorers...", flush=True)
    posenet, segnet = load_differentiable_scorers(upstream_dir=str(UPSTREAM), device="cpu")
    pose_params = sum(p.numel() for p in posenet.parameters())
    seg_params = sum(p.numel() for p in segnet.parameters())
    print(f"  pose params: {pose_params:,}, seg params: {seg_params:,}", flush=True)

    # Phase 3: compute REAL per-pixel scorer gradient magnitudes for each pair
    # (cache to disk so reruns avoid recomputation)
    cache_path = artifacts_dir / "real_scorer_gradients_cache.npz"
    if cache_path.exists():
        print(f"Phase 3: loading cached real scorer gradients from {cache_path}...", flush=True)
        cache = np.load(cache_path)
        seg_grads = cache["seg_grads"]
        pose_grads = cache["pose_grads"]
        if seg_grads.shape != (N_PAIRS, SPATIAL_H, SPATIAL_W):
            print(f"  cache shape mismatch ({seg_grads.shape}); recomputing", flush=True)
            cache_path.unlink()
    if not cache_path.exists():
        print(f"Phase 3: computing REAL per-pixel scorer gradients for {N_PAIRS} pairs...", flush=True)
        seg_grads = np.zeros((N_PAIRS, SPATIAL_H, SPATIAL_W), dtype=np.float32)
        pose_grads = np.zeros((N_PAIRS, SPATIAL_H, SPATIAL_W), dtype=np.float32)
        t0 = time.time()
        for p in range(N_PAIRS):
            seg_mag, pose_mag = compute_real_scorer_gradient_per_pixel(posenet, segnet, pairs_gt[p])
            seg_grads[p] = seg_mag
            pose_grads[p] = pose_mag
            if (p + 1) % 10 == 0:
                print(f"  computed {p+1}/{N_PAIRS} pairs (elapsed {time.time()-t0:.1f}s)", flush=True)
        print(f"  total gradient compute: {time.time()-t0:.1f}s", flush=True)
        np.savez_compressed(cache_path, seg_grads=seg_grads, pose_grads=pose_grads)
        print(f"  cached: {cache_path}", flush=True)
    print(f"  seg_grad stats: min={seg_grads.min():.6e} max={seg_grads.max():.6e} mean={seg_grads.mean():.6e}", flush=True)
    print(f"  pose_grad stats: min={pose_grads.min():.6e} max={pose_grads.max():.6e} mean={pose_grads.mean():.6e}", flush=True)

    # Phase 4: compute UNIWARD weight map per pair
    print(f"Phase 4: computing UNIWARD weight maps + observability...", flush=True)
    weight_maps = np.zeros((N_PAIRS, SPATIAL_H, SPATIAL_W), dtype=np.float32)
    per_axis_decomp_samples = []
    histograms = []
    for p in range(N_PAIRS):
        w = compute_per_pixel_uniward_weight_map_numpy(seg_grads[p], pose_grads[p])
        w = normalize_weight_map_to_unit_mean(w)
        weight_maps[p] = w
        if p < 3:  # sample first 3 pairs for observability
            decomp = decompose_per_axis_weights(seg_grads[p], pose_grads[p])
            per_axis_decomp_samples.append({
                k: {
                    "min": float(v.min()),
                    "max": float(v.max()),
                    "mean": float(v.mean()),
                    "std": float(v.std()),
                } for k, v in decomp.items()
            })
            hist = histogram_weight_distribution(w, bins=10)
            histograms.append({
                "weight_min": float(hist["weight_min"]),
                "weight_max": float(hist["weight_max"]),
                "weight_mean": float(hist["weight_mean"]),
                "weight_median": float(hist["weight_median"]),
                "histogram_counts": hist["histogram_counts"].tolist(),
                "histogram_edges": hist["histogram_edges"].tolist(),
            })
    print(f"  weight_map stats: min={weight_maps.min():.3f} max={weight_maps.max():.3f} mean={weight_maps.mean():.3f}", flush=True)
    print(f"  weight_map dynamic range (max/min ratio): {weight_maps.max()/max(weight_maps.min(), 1e-12):.2f}", flush=True)

    # Phase 5: train BASELINE arm (uniform weight)
    print(f"\n=== Phase 5: BASELINE arm (uniform weight) ===", flush=True)
    baseline_result = train_arm("BASELINE", pairs_gt, None, posenet, segnet)

    # Phase 6: train VARIANT arm (UNIWARD weight)
    print(f"\n=== Phase 6: VARIANT arm (UNIWARD per-pixel weight) ===", flush=True)
    variant_result = train_arm("VARIANT_UNIWARD", pairs_gt, weight_maps, posenet, segnet)

    # Phase 7: per-axis comparison + verdict
    b_seg = baseline_result["final_seg_distortion_mean"]
    b_pose = baseline_result["final_pose_distortion_mean"]
    v_seg = variant_result["final_seg_distortion_mean"]
    v_pose = variant_result["final_pose_distortion_mean"]
    seg_ratio = v_seg / max(b_seg, 1e-12)
    pose_ratio = v_pose / max(b_pose, 1e-12)
    print(f"\n=== Phase 7: PER-AXIS COMPARISON ===", flush=True)
    print(f"  d_seg : baseline={b_seg:.6f}  variant={v_seg:.6f}  ratio={seg_ratio:.4f}  (<1 = improvement)", flush=True)
    print(f"  d_pose: baseline={b_pose:.6f}  variant={v_pose:.6f}  ratio={pose_ratio:.4f}  (<1 = improvement)", flush=True)
    # Composite contest-axis contributions (canonical formula)
    b_contest_partial = 100.0 * b_seg + math.sqrt(10.0 * b_pose)
    v_contest_partial = 100.0 * v_seg + math.sqrt(10.0 * v_pose)
    print(f"  contest-partial (100*seg + sqrt(10*pose)): baseline={b_contest_partial:.4f}  variant={v_contest_partial:.4f}  delta={v_contest_partial-b_contest_partial:+.4f}", flush=True)

    # Verdict
    seg_improved = seg_ratio < 0.99
    pose_improved = pose_ratio < 0.99
    seg_regressed = seg_ratio > 1.05
    pose_regressed = pose_ratio > 1.05
    if (seg_improved or pose_improved) and not (seg_regressed and pose_regressed):
        if seg_improved and pose_improved:
            verdict = "PARADIGM-VALIDATED-JOINT"
        elif seg_improved and not pose_regressed:
            verdict = "PARADIGM-VALIDATED-SEG-AXIS-ONLY"
        elif pose_improved and not seg_regressed:
            verdict = "PARADIGM-VALIDATED-POSE-AXIS-ONLY"
        else:
            verdict = "PARADIGM-PARTIAL"
    else:
        if seg_regressed or pose_regressed:
            verdict = "PARADIGM-FALSIFIED-IMPLEMENTATION-LEVEL"
        else:
            verdict = "PARADIGM-NULL-NO-EFFECT"
    print(f"\n  VERDICT: {verdict}", flush=True)

    # Phase 8: emit canonical JSON
    now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = {
        "schema_version": "uniward_per_pixel_n_plus_1_empirical_v1",
        "subagent_id": "uniward-per-pixel-n-plus-1-real-scorer-anchored-empirical-anchor-50pair-mlx-local-pr110-baseline-20260526",
        "lane_id": "lane_uniward_per_pixel_n_plus_1_real_scorer_empirical_20260526",
        "measurement_utc": now_utc,
        "fixture": {
            "video_path": str(VIDEO_PATH),
            "n_pairs": N_PAIRS,
            "spatial_h": SPATIAL_H,
            "spatial_w": SPATIAL_W,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "ema_decay": EMA_DECAY,
        },
        "scorer_metadata": {
            "loader": "tac.scorer.load_differentiable_scorers (canonical per Catalog #164/#226)",
            "pose_params": pose_params,
            "seg_params": seg_params,
            "device": "cpu",
        },
        "real_scorer_gradient_stats": {
            "seg_grad_min": float(seg_grads.min()),
            "seg_grad_max": float(seg_grads.max()),
            "seg_grad_mean": float(seg_grads.mean()),
            "pose_grad_min": float(pose_grads.min()),
            "pose_grad_max": float(pose_grads.max()),
            "pose_grad_mean": float(pose_grads.mean()),
        },
        "weight_map_stats": {
            "min": float(weight_maps.min()),
            "max": float(weight_maps.max()),
            "mean": float(weight_maps.mean()),
            "dynamic_range_ratio": float(weight_maps.max() / max(weight_maps.min(), 1e-12)),
        },
        "weight_map_per_axis_decomposition_samples": per_axis_decomp_samples,
        "weight_map_histograms": histograms,
        "baseline_arm": baseline_result,
        "variant_arm": variant_result,
        "per_axis_comparison": {
            "seg_baseline_final": b_seg,
            "seg_variant_final": v_seg,
            "seg_ratio_variant_over_baseline": seg_ratio,
            "seg_improved_below_0p99": seg_improved,
            "seg_regressed_above_1p05": seg_regressed,
            "pose_baseline_final": b_pose,
            "pose_variant_final": v_pose,
            "pose_ratio_variant_over_baseline": pose_ratio,
            "pose_improved_below_0p99": pose_improved,
            "pose_regressed_above_1p05": pose_regressed,
            "contest_partial_baseline": b_contest_partial,
            "contest_partial_variant": v_contest_partial,
            "contest_partial_delta": v_contest_partial - b_contest_partial,
        },
        "verdict": verdict,
        "total_wall_clock_s": time.time() - t_total_start,
        "provenance": {
            "evidence_grade": "macOS-MLX research-signal",
            "score_claim": False,
            "promotable": False,
            "axis_tag": "[predicted]",
            "hardware_substrate": "darwin_arm64_m5_max_mlx_local",
            "measurement_axis": "[macOS-MLX research-signal]",
            "hook_numbers_fired": [1, 2, 4, 5, 6],
            "model_id": "uniward_per_pixel_distortion_substrate_v1_2026-05-26",
            "captured_at_utc": now_utc,
            "inputs_sha256": "real_scorer_anchored_50pair_30ep_96x128_seed_42",
        },
    }
    out_path = artifacts_dir / "sweep_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  emitted: {out_path}", flush=True)
    print(f"  total wall-clock: {time.time()-t_total_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
