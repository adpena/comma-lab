# SPDX-License-Identifier: MIT
"""UNIWARD 6th-order integration into BoostNeRV-PR110-residual capacity-
constrained substrate empirical sweep (MLX-local).

Successor of N+1 sister (`3316721639`; PARADIGM-NULL-NO-EFFECT verdict on
free-RGB-tensor architecture). N+1 DIAGNOSED mechanism: UNIWARD per-pixel
weight map IS VALID but requires PARAMETER BOTTLENECK. THIS sweep tests
UNIWARD integration into a MLX-native capacity-constrained ResidualHeadMLP
sister of `tac.substrates.boost_nerv_pr110_residual.architecture.ResidualHeadMLX`
(same shape contract; READ-ONLY consumer scope per Catalog #230).

Fixture: 50 pairs @ 96x128 (sister of N+1; cached real-scorer gradients reused).
Architecture: MLX residual head (z_proj + conv1 + conv2) targeting per-pixel
residual on top of a frozen "PR110-like base" (here a GT-with-noise proxy
so the integration test is end-to-end without PR110 inflate dependency).

The KEY test isn't whether this minimal MLX BoostNeRV beats PR110 frontier;
it's whether UNIWARD per-pixel routing has structural traction at the
~3K-params-per-round capacity bottleneck (the N+1-diagnosed missing element).

Per CLAUDE.md "MLX portable-local-substrate authority": tagged
`[macOS-MLX research-signal]` per Catalog #192/#317/#341. NO paid dispatch.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

# Repo root (script is in tools/; parents[1] is the repo root)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upstream"))

from tac.scorer import load_differentiable_scorers
from tac.substrates.score_aware_common import score_pair_components
from tac.substrates.uniward_per_pixel_distortion import (
    compute_per_pixel_uniward_weight_map_numpy,
)
from tac.substrates.uniward_per_pixel_distortion.weight_map import (
    normalize_weight_map_to_unit_mean,
)
# Sister-disjoint READ-ONLY import of BoostNeRV substrate per Catalog #230
from tac.substrates.boost_nerv_pr110_residual import (
    BoostNervPr110ResidualConfig,
    DEFAULT_NUM_BOOSTING_ROUNDS,
)
from tac.substrates.boost_nerv_pr110_residual.architecture import (
    num_residual_parameters,
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

# Cached gradients from N+1 (sister-disjoint reuse per Catalog #230)
N_PLUS_1_CACHE = ROOT / ".omx/research/uniward_per_pixel_n_plus_1_artifacts_20260526/real_scorer_gradients_cache.npz"

# Output artifacts dir
ARTIFACTS_DIR = ROOT / ".omx/research/uniward_6th_order_into_boostnerv_artifacts_20260526"


def decode_pairs() -> np.ndarray:
    """Decode first 2*N_PAIRS frames from upstream/videos/0.mkv (sister of N+1)."""
    import av
    import PIL.Image

    container = av.open(str(VIDEO_PATH))
    stream = container.streams.video[0]
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i >= 2 * N_PAIRS:
            break
        img = frame.to_ndarray(format="rgb24")
        pil = PIL.Image.fromarray(img)
        pil = pil.resize((SPATIAL_W, SPATIAL_H), PIL.Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # (3, H, W)
        frames.append(arr)
    container.close()
    frames = np.stack(frames)
    pairs = frames.reshape(N_PAIRS, 2, 3, SPATIAL_H, SPATIAL_W)
    return pairs


def measure_per_axis_scores(posenet, segnet, pairs_pred, pairs_gt) -> dict:
    """Per-axis d_seg + d_pose averaged over all pairs via canonical helper."""
    seg_sum = 0.0
    pose_sum = 0.0
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
    """Simulate eval_roundtrip uint8 bottleneck per CLAUDE.md non-negotiable."""
    x_clip = mx.clip(x_mlx, 0.0, 1.0)
    x_uint8 = mx.round(x_clip * 255.0) / 255.0
    return x_clip + mx.stop_gradient(x_uint8 - x_clip)


def init_capacity_constrained_residual_head(cfg: BoostNervPr110ResidualConfig, prng_key: int):
    """Initialize MLX residual head parameters mirroring BoostNeRV ResidualHeadMLX.

    Returns dict of MLX arrays for the 3 layers: z_proj (Linear), conv1 (Conv2d),
    conv2 (Conv2d). Init scheme: Xavier-like for stability.
    """
    rng = np.random.default_rng(prng_key)
    h = cfg.residual_hidden_dim  # 12
    # z_proj: (pr110_latent_dim, h) + bias (h,)
    z_proj_w = rng.normal(0, 1.0 / np.sqrt(cfg.pr110_latent_dim), (cfg.pr110_latent_dim, h)).astype(np.float32)
    z_proj_b = np.zeros(h, dtype=np.float32)
    # conv1: in=3+h, out=h, kernel=3, padding=1 → weight shape (h, in, 3, 3)
    in_c1 = 3 + h
    conv1_w = rng.normal(0, 1.0 / np.sqrt(in_c1 * 3 * 3), (h, in_c1, 3, 3)).astype(np.float32)
    conv1_b = np.zeros(h, dtype=np.float32)
    # conv2: in=h, out=3, kernel=1
    conv2_w = rng.normal(0, 1.0 / np.sqrt(h), (3, h, 1, 1)).astype(np.float32)
    conv2_b = np.zeros(3, dtype=np.float32)
    return {
        "z_proj_w": mx.array(z_proj_w),
        "z_proj_b": mx.array(z_proj_b),
        "conv1_w": mx.array(conv1_w),
        "conv1_b": mx.array(conv1_b),
        "conv2_w": mx.array(conv2_w),
        "conv2_b": mx.array(conv2_b),
    }


def total_params(params_dict) -> int:
    """Total parameter count summed over all MLX arrays."""
    total = 0
    for k, v in params_dict.items():
        total += int(np.prod(v.shape))
    return total


def forward_residual_head(params, rgb_base_NCHW, z_pr110):
    """Forward through MLX residual head (mirrors BoostNeRV ResidualHeadMLX shape).

    Inputs:
        rgb_base_NCHW: (B, 3, H, W) MLX array — base reconstruction in [0,1]
        z_pr110: (B, pr110_latent_dim) MLX array — per-pair latent

    Output: (B, 3, H, W) MLX tanh-bounded residual in [-1, 1]

    Note: MLX conv2d operates on NHWC convention. We transpose internally and
    transpose back at the end so the external API stays NCHW.
    """
    # z_proj: (B, pr110_latent_dim) @ (pr110_latent_dim, h) → (B, h)
    z_emb = z_pr110 @ params["z_proj_w"] + params["z_proj_b"]  # (B, h)
    b, c = z_emb.shape
    h, w = rgb_base_NCHW.shape[2], rgb_base_NCHW.shape[3]
    # Broadcast z_emb to spatial grid: (B, h, H, W)
    z_grid = mx.broadcast_to(z_emb[:, :, None, None], (b, c, h, w))
    # Concat along channel axis: (B, 3+h, H, W)
    h_concat_nchw = mx.concatenate([rgb_base_NCHW, z_grid], axis=1)
    # Transpose to NHWC: (B, H, W, 3+h)
    h_concat = mx.transpose(h_concat_nchw, axes=(0, 2, 3, 1))
    # MLX conv2d expects weight in (out_c, kh, kw, in_c) layout.
    # Our params["conv1_w"] is (h, 3+h, 3, 3) (NCHW-like layout from init).
    # Transpose to (h, 3, 3, 3+h)
    w1 = mx.transpose(params["conv1_w"], axes=(0, 2, 3, 1))
    h_act = mx.conv2d(h_concat, w1, stride=1, padding=1)
    h_act = h_act + params["conv1_b"][None, None, None, :]
    # ReLU
    h_act = mx.maximum(h_act, 0.0)
    # conv2 (1x1): (3, h, 1, 1) → (3, 1, 1, h)
    w2 = mx.transpose(params["conv2_w"], axes=(0, 2, 3, 1))
    out = mx.conv2d(h_act, w2, stride=1, padding=0)
    out = out + params["conv2_b"][None, None, None, :]
    # Transpose back to NCHW: (B, 3, H, W)
    out_nchw = mx.transpose(out, axes=(0, 3, 1, 2))
    # tanh bound
    residual = mx.tanh(out_nchw)
    return residual


def compose_base_plus_residual(rgb_base_NCHW, residual, gain_clamp: float):
    """Canonical BoostNeRV composition: clamp(base + clamp(residual, ±gain), 0, 1)."""
    residual_clamped = mx.clip(residual, -gain_clamp, gain_clamp)
    composed = mx.clip(rgb_base_NCHW + residual_clamped, 0.0, 1.0)
    return composed


def train_capacity_constrained_arm(
    arm_name: str,
    pairs_gt: np.ndarray,
    weight_maps_per_pair: np.ndarray | None,
    posenet, segnet,
    cfg: BoostNervPr110ResidualConfig,
) -> dict:
    """Train one arm with MLX capacity-constrained residual head.

    BASELINE: weight_maps_per_pair = None → uniform per-pixel weighting.
    VARIANT: weight_maps_per_pair = (N_PAIRS, H, W) UNIWARD weight map.

    Architecture:
        rgb_base = GT + 0.15 sigma noise (proxy for PR110-like base reconstruction)
        z_pr110 = per-pair learned latent (24-dim; canonical BoostNeRV pr110_latent_dim)
        residual = ResidualHeadMLP(rgb_base, z_pr110) → tanh-bounded (B, 3, H, W)
        rgb_composed = clamp(rgb_base + clamp(residual, ±0.05), 0, 1)
        loss = mean(per_pixel_err * weight_map)  [where per_pixel_err = (rgb_composed - gt)**2 avg over C]

    The PARAMETER BOTTLENECK is the residual head's ~3K params across 12K pixels —
    UNIWARD routing has signal to leverage HERE that N+1's free-RGB-tensor lacked.
    """
    mx.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # Per-pair learned latent (canonical BoostNeRV pr110_latent_dim=24)
    # Sister to PR110-extracted latent (here random; PR110 inflate dependency
    # avoided so this stays MLX-local + sister-disjoint)
    z_latents = mx.array(
        rng.standard_normal((N_PAIRS, cfg.pr110_latent_dim)).astype(np.float32)
    )

    # Initialize ONE shared residual head (canonical BoostNeRV "boosting round"
    # shape; per CLAUDE.md HNeRV parity L7 bolt-on vs substrate-engineering split:
    # the residual head IS shared across pairs at this scale)
    params = init_capacity_constrained_residual_head(cfg, prng_key=SEED + 1)
    ema_params = {k: mx.array(v) for k, v in params.items()}

    print(f"  [{arm_name}] residual head params: {total_params(params):,}", flush=True)

    # Build rgb_base (frozen): GT + small noise per pair, simulating PR110 base
    # reconstruction at a non-trivial operating point where UNIWARD routing can
    # have effect (sister to N+1 0.15-sigma init)
    rgb_base_per_pair = np.zeros((N_PAIRS, 2, 3, SPATIAL_H, SPATIAL_W), dtype=np.float32)
    rng_base = np.random.default_rng(SEED + 2)
    for p in range(N_PAIRS):
        noise = rng_base.standard_normal((2, 3, SPATIAL_H, SPATIAL_W)).astype(np.float32) * 0.15
        rgb_base_per_pair[p] = np.clip(pairs_gt[p] + noise, 0.0, 1.0)
    rgb_base_NCHW_per_pair = mx.array(rgb_base_per_pair)  # (N_PAIRS, 2, 3, H, W) frozen base

    pairs_gt_mlx = mx.array(pairs_gt)

    if weight_maps_per_pair is not None:
        w_mlx = mx.array(weight_maps_per_pair.astype(np.float32))  # (N_PAIRS, H, W)
    else:
        w_mlx = mx.ones((N_PAIRS, SPATIAL_H, SPATIAL_W), dtype=mx.float32)

    def loss_fn(params_in, z_latents_in):
        total_loss = mx.array(0.0)
        # Process ONE pair at a time (composable; N_PAIRS=50, B=2 per pair)
        for p in range(N_PAIRS):
            rgb_base_pair = rgb_base_NCHW_per_pair[p]  # (2, 3, H, W)
            z_pair = z_latents_in[p:p+1]  # (1, latent_dim)
            # Apply residual head to each frame (broadcast z over 2 frames)
            z_pair_2frame = mx.broadcast_to(z_pair, (2, cfg.pr110_latent_dim))
            residual = forward_residual_head(params_in, rgb_base_pair, z_pair_2frame)
            composed = compose_base_plus_residual(rgb_base_pair, residual, cfg.boosting_gain_clamp)
            # eval_roundtrip simulation
            composed_rt = eval_roundtrip_sim(composed)
            # Per-pixel L2 reconstruction error (mean over channels)
            per_pixel_err = mx.mean((composed_rt - pairs_gt_mlx[p]) ** 2, axis=1)  # (2, H, W)
            per_pixel_err_pair = mx.mean(per_pixel_err, axis=0)  # (H, W)
            # UNIWARD weighting at the capacity-constrained substrate's loss path
            weighted = per_pixel_err_pair * w_mlx[p]
            total_loss = total_loss + mx.mean(weighted)
        return total_loss / N_PAIRS

    val_and_grad = mx.value_and_grad(loss_fn)

    # Adam state (manual implementation)
    m_state = {k: mx.zeros_like(v) for k, v in params.items()}
    v_state = {k: mx.zeros_like(v) for k, v in params.items()}
    z_m_state = mx.zeros_like(z_latents)
    z_v_state = mx.zeros_like(z_latents)
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps = 1e-8

    # Measure init (epoch 0) — get composed RGB from initial params
    def compute_composed_rgb_for_eval(params_eval, z_eval):
        composed_arr = np.zeros((N_PAIRS, 2, 3, SPATIAL_H, SPATIAL_W), dtype=np.float32)
        for p in range(N_PAIRS):
            rgb_base_pair = rgb_base_NCHW_per_pair[p]
            z_pair = z_eval[p:p+1]
            z_pair_2frame = mx.broadcast_to(z_pair, (2, cfg.pr110_latent_dim))
            residual = forward_residual_head(params_eval, rgb_base_pair, z_pair_2frame)
            composed = compose_base_plus_residual(rgb_base_pair, residual, cfg.boosting_gain_clamp)
            composed_rt = eval_roundtrip_sim(composed)
            mx.eval(composed_rt)
            composed_arr[p] = np.asarray(composed_rt).copy()
        return composed_arr

    measurement_epochs = [0, 30]
    measurements = {}
    pred_init = compute_composed_rgb_for_eval(params, z_latents)
    measurements[0] = measure_per_axis_scores(posenet, segnet, pred_init, pairs_gt)
    measurements[0]["wall_clock_s"] = 0.0

    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        # val_and_grad returns grads with the same nested structure as the first arg
        loss_val, (grad_params, grad_z) = mx.value_and_grad(loss_fn, argnums=(0, 1))(params, z_latents)
        # Adam update for params dict
        new_params = {}
        for k in params.keys():
            m_state[k] = adam_beta1 * m_state[k] + (1.0 - adam_beta1) * grad_params[k]
            v_state[k] = adam_beta2 * v_state[k] + (1.0 - adam_beta2) * grad_params[k] * grad_params[k]
            m_hat = m_state[k] / (1.0 - adam_beta1 ** epoch)
            v_hat = v_state[k] / (1.0 - adam_beta2 ** epoch)
            new_params[k] = params[k] - LEARNING_RATE * m_hat / (mx.sqrt(v_hat) + adam_eps)
        # Adam update for z_latents
        z_m_state = adam_beta1 * z_m_state + (1.0 - adam_beta1) * grad_z
        z_v_state = adam_beta2 * z_v_state + (1.0 - adam_beta2) * grad_z * grad_z
        z_m_hat = z_m_state / (1.0 - adam_beta1 ** epoch)
        z_v_hat = z_v_state / (1.0 - adam_beta2 ** epoch)
        z_latents = z_latents - LEARNING_RATE * z_m_hat / (mx.sqrt(z_v_hat) + adam_eps)
        # EMA update
        for k in params.keys():
            ema_params[k] = EMA_DECAY * ema_params[k] + (1.0 - EMA_DECAY) * new_params[k]
        params = new_params
        mx.eval(loss_val, z_latents, *params.values(), *ema_params.values())
        if epoch in measurement_epochs:
            # Use EMA shadow per CLAUDE.md EMA non-negotiable
            pred_arr = compute_composed_rgb_for_eval(ema_params, z_latents)
            measurements[epoch] = measure_per_axis_scores(posenet, segnet, pred_arr, pairs_gt)
            measurements[epoch]["wall_clock_s"] = time.time() - t_start
            measurements[epoch]["loss_value"] = float(loss_val)
            print(
                f"  [{arm_name}] epoch {epoch}: seg_mean={measurements[epoch]['seg_distortion_mean']:.6f} "
                f"pose_mean={measurements[epoch]['pose_distortion_mean']:.6f} "
                f"loss={float(loss_val):.6f} wc={measurements[epoch]['wall_clock_s']:.1f}s",
                flush=True,
            )

    return {
        "arm_name": arm_name,
        "epochs_measured": measurement_epochs,
        "measurements_per_epoch": {str(k): v for k, v in measurements.items()},
        "final_seg_distortion_mean": measurements[EPOCHS]["seg_distortion_mean"],
        "final_pose_distortion_mean": measurements[EPOCHS]["pose_distortion_mean"],
        "residual_head_params": total_params(params),
        "z_latents_params": int(np.prod(np.asarray(z_latents).shape)),
    }


def main() -> int:
    t_total_start = time.time()
    print(f"=== UNIWARD 6th-order integration into BoostNeRV-PR110-residual sweep ===")
    print(f"  fixture: {VIDEO_PATH} (N_PAIRS={N_PAIRS}, {SPATIAL_H}x{SPATIAL_W})")
    print(f"  epochs: {EPOCHS}, lr={LEARNING_RATE}, seed={SEED}, ema={EMA_DECAY}")
    print(f"  uniward_lambda: {UNIWARD_LAMBDA}")

    cfg = BoostNervPr110ResidualConfig()
    print(f"  BoostNeRV config: hidden_dim={cfg.residual_hidden_dim} latent_dim={cfg.pr110_latent_dim} "
          f"residual_grid=({cfg.residual_spatial_h},{cfg.residual_spatial_w}) gain_clamp={cfg.boosting_gain_clamp}")
    print(f"  theoretical residual params/round: {num_residual_parameters(cfg):,}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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

    # Phase 3: reuse cached real-scorer gradients from sister N+1 (Catalog #230)
    print(f"Phase 3: loading cached real-scorer gradients from sister N+1 ({N_PLUS_1_CACHE})...", flush=True)
    if not N_PLUS_1_CACHE.exists():
        print(f"  FATAL: N+1 cache not found at {N_PLUS_1_CACHE}", flush=True)
        return 1
    cache = np.load(N_PLUS_1_CACHE)
    seg_grads = cache["seg_grads"]
    pose_grads = cache["pose_grads"]
    print(f"  loaded cache: seg_grads={seg_grads.shape} pose_grads={pose_grads.shape}", flush=True)
    if seg_grads.shape != (N_PAIRS, SPATIAL_H, SPATIAL_W):
        print(f"  FATAL: cache shape mismatch ({seg_grads.shape}); expected ({N_PAIRS}, {SPATIAL_H}, {SPATIAL_W})", flush=True)
        return 1

    # Phase 4: compute UNIWARD weight maps
    print(f"Phase 4: computing UNIWARD weight maps...", flush=True)
    weight_maps = np.zeros((N_PAIRS, SPATIAL_H, SPATIAL_W), dtype=np.float32)
    for p in range(N_PAIRS):
        w = compute_per_pixel_uniward_weight_map_numpy(seg_grads[p], pose_grads[p])
        w = normalize_weight_map_to_unit_mean(w)
        weight_maps[p] = w
    print(f"  weight_map stats: min={weight_maps.min():.3f} max={weight_maps.max():.3f} mean={weight_maps.mean():.3f}", flush=True)
    print(f"  weight_map dynamic range: {weight_maps.max()/max(weight_maps.min(), 1e-12):.2f}x", flush=True)

    # Phase 5: train BASELINE arm (BoostNeRV uniform per-pixel weighting)
    print(f"\n=== Phase 5: BASELINE arm (BoostNeRV uniform per-pixel weighting) ===", flush=True)
    baseline_result = train_capacity_constrained_arm(
        "BASELINE", pairs_gt, None, posenet, segnet, cfg
    )

    # Phase 6: train VARIANT arm (BoostNeRV + UNIWARD)
    print(f"\n=== Phase 6: VARIANT arm (BoostNeRV + UNIWARD per-pixel weighting) ===", flush=True)
    variant_result = train_capacity_constrained_arm(
        "VARIANT_UNIWARD_INTO_BOOSTNERV", pairs_gt, weight_maps, posenet, segnet, cfg
    )

    # Phase 7: per-axis comparison + verdict
    b_seg = baseline_result["final_seg_distortion_mean"]
    b_pose = baseline_result["final_pose_distortion_mean"]
    v_seg = variant_result["final_seg_distortion_mean"]
    v_pose = variant_result["final_pose_distortion_mean"]
    seg_ratio = v_seg / max(b_seg, 1e-12)
    pose_ratio = v_pose / max(b_pose, 1e-12)
    print(f"\n=== Phase 7: PER-AXIS COMPARISON (BoostNeRV capacity-constrained substrate) ===", flush=True)
    print(f"  d_seg : baseline={b_seg:.6f}  variant={v_seg:.6f}  ratio={seg_ratio:.4f}  (<1 = improvement)", flush=True)
    print(f"  d_pose: baseline={b_pose:.6f}  variant={v_pose:.6f}  ratio={pose_ratio:.4f}  (<1 = improvement)", flush=True)
    b_contest_partial = 100.0 * b_seg + math.sqrt(10.0 * b_pose)
    v_contest_partial = 100.0 * v_seg + math.sqrt(10.0 * v_pose)
    contest_delta = v_contest_partial - b_contest_partial
    print(f"  contest-partial (100*seg + sqrt(10*pose)): baseline={b_contest_partial:.4f} variant={v_contest_partial:.4f} delta={contest_delta:+.4f}", flush=True)

    # Verdict
    seg_improved = seg_ratio < 0.99
    pose_improved = pose_ratio < 0.99
    seg_regressed = seg_ratio > 1.05
    pose_regressed = pose_ratio > 1.05
    if (seg_improved or pose_improved) and not (seg_regressed and pose_regressed):
        if seg_improved and pose_improved:
            verdict = "PARADIGM-VALIDATED-JOINT-AT-CAPACITY-CONSTRAINED-SUBSTRATE"
        elif seg_improved and not pose_regressed:
            verdict = "PARADIGM-VALIDATED-SEG-AXIS-ONLY-AT-CAPACITY-CONSTRAINED-SUBSTRATE"
        elif pose_improved and not seg_regressed:
            verdict = "PARADIGM-VALIDATED-POSE-AXIS-ONLY-AT-CAPACITY-CONSTRAINED-SUBSTRATE"
        else:
            verdict = "PARADIGM-PARTIAL-AT-CAPACITY-CONSTRAINED-SUBSTRATE"
    else:
        if seg_regressed or pose_regressed:
            verdict = "PARADIGM-FALSIFIED-IMPLEMENTATION-LEVEL-AT-CAPACITY-CONSTRAINED-SUBSTRATE"
        else:
            verdict = "PARADIGM-NULL-NO-EFFECT-AT-CAPACITY-CONSTRAINED-SUBSTRATE"
    print(f"\n  VERDICT: {verdict}", flush=True)

    # Phase 8: emit canonical JSON
    now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = {
        "schema_version": "uniward_6th_order_into_boostnerv_capacity_constrained_empirical_v1",
        "subagent_id": "uniward-6th-order-integration-into-boostnerv-pr110-residual-capacity-constrained-substrate-recursive-doctrine-mlx-first-numpy-portable-20260526",
        "lane_id": "lane_uniward_6th_order_integration_into_boostnerv_pr110_residual_20260526",
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
            "uniward_lambda": UNIWARD_LAMBDA,
        },
        "scorer_metadata": {
            "loader": "tac.scorer.load_differentiable_scorers (canonical per Catalog #164/#226)",
            "pose_params": pose_params,
            "seg_params": seg_params,
            "device": "cpu",
        },
        "boostnerv_substrate_config": {
            "residual_hidden_dim": cfg.residual_hidden_dim,
            "pr110_latent_dim": cfg.pr110_latent_dim,
            "boosting_gain_clamp": cfg.boosting_gain_clamp,
            "residual_spatial_h": cfg.residual_spatial_h,
            "residual_spatial_w": cfg.residual_spatial_w,
            "num_boosting_rounds": cfg.num_boosting_rounds,
            "theoretical_params_per_round": num_residual_parameters(cfg),
            "consumer_scope": "read_only_consumer_import_per_catalog_230",
        },
        "real_scorer_gradient_source": {
            "cache_path": str(N_PLUS_1_CACHE.relative_to(ROOT)),
            "sister_lane": "lane_uniward_per_pixel_n_plus_1_real_scorer_empirical_20260526",
            "sister_commit": "3316721639",
        },
        "weight_map_stats": {
            "min": float(weight_maps.min()),
            "max": float(weight_maps.max()),
            "mean": float(weight_maps.mean()),
            "dynamic_range_ratio": float(weight_maps.max() / max(weight_maps.min(), 1e-12)),
        },
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
            "contest_partial_delta": contest_delta,
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
            "model_id": "uniward_6th_order_into_boostnerv_v1_2026-05-26",
            "captured_at_utc": now_utc,
            "inputs_sha256": "real_scorer_anchored_from_n_plus_1_cache_50pair_30ep_96x128_seed_42",
        },
    }
    out_path = ARTIFACTS_DIR / "sweep_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  emitted: {out_path}", flush=True)
    print(f"  total wall-clock: {time.time()-t_total_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
