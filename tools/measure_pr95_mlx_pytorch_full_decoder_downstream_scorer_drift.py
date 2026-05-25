#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Measure MLX-to-PyTorch drift propagation through the full PR95 HNeRV decoder
all the way through the canonical SegNet + PoseNet scorer to the contest score
formula.

Task: #1258 PR95-MLX-FULL-DECODER-DOWNSTREAM-SCORER-DRIFT-MEASUREMENT
Lane: lane_pr95_mlx_full_decoder_downstream_scorer_drift_measurement_20260525
Evidence grade: [macOS-MLX research-signal] per CLAUDE.md "MLX portable-local-substrate
authority". Per Catalog #1 (MPS noise) + Catalog #192 (macOS-CPU advisory not
promotable) + Catalog #287/#323 canonical Provenance: every emitted row carries
score_claim=False + promotable=False + ready_for_exact_eval_dispatch=False +
axis_tag="[macOS-MLX research-signal]".

Operational scope: empirically VERIFY the theoretical Selfcomp+MacKay analysis
that 3.05e-5 Conv2d boundary drift propagates as < 0.001 contest-score-units
through the full scorer pipeline. Goal = operational closure of the drift
question per T3 grand council ACTIVE EXPLORATION TERTIARY priority
(.omx/research/t3_grand_council_active_exploration_conv2d_drift_unexplored_paths_landed_20260525.md
operator-routable #3 = "Sister Yousfi full-decoder downstream scorer test").

Pipeline measured per task spec:

  Stage 1: HNeRVDecoder forward (MLX vs PyTorch from a single archive's
           state_dict via the Slot 1 canonical bridge).
  Stage 2: uint8 RGB quantization at inflate ((rgb * 1).clip(0, 255).round()
           .astype(uint8); HNeRVDecoder already outputs [0, 255] so no scale
           factor needed; the contest inflate.sh pipeline does cast to uint8).
  Stage 3: SegNet forward (canonical upstream.modules.SegNet on RGB frame
           pair -> 5-class logits -> argmax).
  Stage 4: PoseNet forward (canonical upstream.modules.PoseNet on RGB frame
           pair -> rgb_to_yuv6 -> 12-channel -> 6-vec pose).
  Stage 5: contest score aggregation (S = 100*d_seg + sqrt(10*d_pose) +
           rate term constant; we measure only d_seg + d_pose drift; rate
           term is unchanged because archive bytes are identical between
           runs).

NOTE: this is NOT a contest-score measurement. It is a MLX-vs-PyTorch drift
propagation measurement on a single archive's bytes. The PyTorch and MLX
HNeRVDecoder share the same state_dict bytes (per Slot 1 export bridge); they
differ only in framework-arithmetic order. Therefore both produce the SAME
ground-truth (contest score) within the framework drift bound. The objective is
to characterize the drift bound at every stage of the pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "upstream") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "upstream"))

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX  # noqa: E402,I001


# ----------------------------------------------------------------------------
# Canonical formula constants (per CLAUDE.md "Apples-to-apples evidence
# discipline" + .omx/state/canonical_equations_registry).
# ----------------------------------------------------------------------------
CANONICAL_SEG_MULTIPLIER = 100.0
CANONICAL_POSE_SQRT_INNER = 10.0
CANONICAL_RATE_MULTIPLIER = 25.0
CANONICAL_RATE_DENOM_BYTES = 37_545_489

# Per task spec: theoretical Selfcomp+MacKay analysis predicts
# aggregate contest-score drift well below 0.001 (frontier-relevant precision).
CONTEST_SCORE_PRECISION_LO = 0.001  # below = BELOW_SCORER_PRECISION
CONTEST_SCORE_PRECISION_HI = 0.01  # above = ABOVE_SCORER_PRECISION


@dataclass(frozen=True)
class StageDriftMeasurement:
    """Canonical drift measurement for a single pipeline stage."""

    stage_name: str
    metric: str
    max_abs: float
    mean_abs: float
    rms: float
    extra: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "metric": self.metric,
            "max_abs": float(self.max_abs),
            "mean_abs": float(self.mean_abs),
            "rms": float(self.rms),
            "extra": dict(self.extra),
        }


def _canonical_provenance(*, checkpoint_mode: str, n_pairs: int) -> dict[str, Any]:
    """Catalog #287/#323 canonical Provenance for every emitted row."""
    return {
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "score_claim": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "axis_tag": EVIDENCE_TAG_MLX,
        "hardware_substrate": "darwin_arm64_apple_silicon",
        "frame_count_basis": int(n_pairs * 2),
        "pair_count_basis": int(n_pairs),
        "checkpoint_mode": checkpoint_mode,
        "captured_at_utc": datetime.now(UTC).isoformat(),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_int64_array(values: np.ndarray) -> str:
    data = np.ascontiguousarray(values, dtype=np.int64).tobytes()
    return hashlib.sha256(data).hexdigest()


def _aggregate_drift_max_abs(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if diff.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(diff.max()),
        float(diff.mean()),
        float(np.sqrt((diff * diff).mean())),
    )


# ----------------------------------------------------------------------------
# Stage 1: HNeRVDecoder forward (MLX vs PyTorch from same state_dict).
# ----------------------------------------------------------------------------
def _render_pair_batch(
    *,
    state_dict_mlx: Mapping[str, np.ndarray],
    latents: np.ndarray,
    meta: Mapping[str, Any],
    start_pair: int,
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Render N pairs through MLX vs PyTorch HNeRVDecoder forward.

    Returns (rgb_mlx_uint8_range, rgb_pytorch_uint8_range, mlx_seconds, torch_seconds).
    Both arrays have shape (N, 2, 3, H, W) in [0, 255] range, float32 (NOT yet
    uint8-quantized).
    """
    import importlib.util

    import mlx.core as mx
    import torch

    # Load canonical PR 95 HNeRVDecoder from submissions/a1/src/model.py
    # (canonical reference per Slot 1 export bridge); the submissions/ tree is
    # not a Python package, so we load by file path.
    model_path = REPO_ROOT / "submissions" / "a1" / "src" / "model.py"
    spec = importlib.util.spec_from_file_location(
        "pr95_a1_canonical_model", model_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load HNeRVDecoder from {model_path}")
    canonical_model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical_model_mod)
    HNeRVDecoder = canonical_model_mod.HNeRVDecoder
    from tac.local_acceleration.pr95_hnerv_mlx import (
        HNeRVDecoderMLX,
        load_pytorch_state_dict_into_mlx,
    )

    pair_count = int(latents.shape[0])
    if start_pair < 0 or start_pair >= pair_count:
        raise ValueError(f"start_pair out of range: {start_pair} for {pair_count} pairs")
    end_pair = min(start_pair + n_pairs, pair_count)
    indices = np.arange(start_pair, end_pair, dtype=np.int64)
    z_np = latents[indices].astype(np.float32, copy=True)

    latent_dim = int(meta["latent_dim"])
    base_channels = int(meta["base_channels"])
    eval_size = tuple(int(dim) for dim in meta["eval_size"])

    torch_model = HNeRVDecoder(
        latent_dim=latent_dim,
        base_channels=base_channels,
        eval_size=eval_size,
    ).eval()
    torch_state_dict = {
        name: torch.from_numpy(value.astype(np.float32, copy=True))
        for name, value in state_dict_mlx.items()
    }
    torch_model.load_state_dict(torch_state_dict)

    # IMPORTANT: pin MLX device to CPU explicitly per the canonical Slot 1
    # bridge (compare_pr95_public_archive_forward_with_pytorch). The MLX
    # default device on macOS Apple Silicon is GPU (Metal) which has
    # different drift than CPU. We measure CPU-vs-CPU per Slot 1's canonical
    # anchor 3.05e-5.
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mlx_model = HNeRVDecoderMLX(
            latent_dim=latent_dim,
            base_channels=base_channels,
            eval_size=eval_size,
        )
        load_pytorch_state_dict_into_mlx(mlx_model, state_dict_mlx)

        t0 = time.perf_counter()
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(z_np)).detach().cpu().numpy()
        torch_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        mlx_out = mlx_model(mx.array(z_np))
        mx.eval(mlx_out)
        mlx_out_np = np.asarray(mlx_out)
        mlx_elapsed = time.perf_counter() - t0
    finally:
        mx.set_default_device(previous_device)

    return mlx_out_np, torch_out, mlx_elapsed, torch_elapsed, indices


def _measure_stage1_decoder_drift(
    *,
    rgb_mlx: np.ndarray,
    rgb_torch: np.ndarray,
) -> StageDriftMeasurement:
    max_abs, mean_abs, rms = _aggregate_drift_max_abs(rgb_mlx, rgb_torch)
    return StageDriftMeasurement(
        stage_name="stage_1_hnerv_decoder_forward",
        metric="per_pixel_rgb_float32_in_0_255_range",
        max_abs=max_abs,
        mean_abs=mean_abs,
        rms=rms,
        extra={
            "tensor_shape": [int(d) for d in rgb_mlx.shape],
            "value_range": [float(min(rgb_mlx.min(), rgb_torch.min())),
                            float(max(rgb_mlx.max(), rgb_torch.max()))],
        },
    )


# ----------------------------------------------------------------------------
# Stage 2: uint8 RGB quantization at inflate.
# ----------------------------------------------------------------------------
def _quantize_rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Canonical contest inflate uint8 cast: clip + round + cast."""
    return np.clip(rgb, 0.0, 255.0).round().astype(np.uint8)


def _measure_stage2_uint8_drift(
    *,
    rgb_mlx: np.ndarray,
    rgb_torch: np.ndarray,
) -> StageDriftMeasurement:
    uint8_mlx = _quantize_rgb_to_uint8(rgb_mlx)
    uint8_torch = _quantize_rgb_to_uint8(rgb_torch)
    diff = np.abs(uint8_mlx.astype(np.int32) - uint8_torch.astype(np.int32))
    flipped_pixels = int((diff > 0).sum())
    total_pixels = int(diff.size)
    histogram: dict[str, int] = {}
    for level in range(min(int(diff.max()) + 1, 6)):
        histogram[f"diff_{level}_levels"] = int((diff == level).sum())
    return StageDriftMeasurement(
        stage_name="stage_2_uint8_quantization_at_inflate",
        metric="per_pixel_uint8_level_difference",
        max_abs=float(diff.max()) if diff.size else 0.0,
        mean_abs=float(diff.mean()) if diff.size else 0.0,
        rms=float(np.sqrt((diff.astype(np.float64) ** 2).mean())) if diff.size else 0.0,
        extra={
            "tensor_shape": [int(d) for d in uint8_mlx.shape],
            "flipped_pixels": flipped_pixels,
            "total_pixels": total_pixels,
            "flipped_fraction": float(flipped_pixels) / float(total_pixels)
            if total_pixels
            else 0.0,
            "histogram_levels_0_to_5": histogram,
            "selfcomp_mackay_predicted_uint8_levels_max_abs": 0.0078,
            "theoretical_prediction": "ZERO uint8 flips (sub-quantization)",
        },
    )


# ----------------------------------------------------------------------------
# Stage 3 + 4: SegNet + PoseNet forward (uses canonical upstream modules
# loaded with paired MLX adapter).
# ----------------------------------------------------------------------------
def _build_scorer_pair(*, posenet_sd_path: Path, segnet_sd_path: Path) -> tuple[Any, Any, Any, Any]:
    """Build canonical PoseNet + SegNet + their paired MLX adapters."""
    from modules import PoseNet, SegNet  # type: ignore
    from safetensors.torch import load_file

    from tac.local_acceleration.mlx_scorer_adapters import (
        torch_posenet_to_mlx,
        torch_segnet_to_mlx,
    )

    segnet = SegNet().eval()
    segnet.load_state_dict(load_file(str(segnet_sd_path), device="cpu"))
    posenet = PoseNet().eval()
    posenet.load_state_dict(load_file(str(posenet_sd_path), device="cpu"))

    segnet_mlx = torch_segnet_to_mlx(segnet)
    posenet_mlx = torch_posenet_to_mlx(posenet)
    return posenet, segnet, posenet_mlx, segnet_mlx


def _run_segnet_pair(
    *,
    segnet_torch: Any,
    segnet_mlx: Any,
    rgb_pair_btchw: np.ndarray,  # (B, T=2, C=3, H_render=384, W_render=512), [0,255] float32
) -> tuple[np.ndarray, np.ndarray]:
    """Run SegNet on a pair. Returns (torch_logits, mlx_logits) of shape (B, 5, H, W)."""
    import torch
    from frame_utils import segnet_model_input_size  # type: ignore

    from tac.local_acceleration.mlx_scorer_adapters import run_mlx_segnet_nchw

    # SegNet.preprocess_input slices last frame + interpolates to (384, 512)
    # via canonical bilinear (HNeRVDecoder already outputs 384x512, so the
    # interpolate is a no-op here).
    target_h, target_w = segnet_model_input_size[1], segnet_model_input_size[0]
    last_frame_t = torch.from_numpy(rgb_pair_btchw[:, -1, ...].astype(np.float32, copy=True))
    if last_frame_t.shape[-2] != target_h or last_frame_t.shape[-1] != target_w:
        last_frame_t = torch.nn.functional.interpolate(
            last_frame_t, size=(target_h, target_w), mode="bilinear"
        )
    with torch.no_grad():
        torch_logits = segnet_torch(last_frame_t).detach().cpu().numpy()
    last_frame_np = last_frame_t.detach().cpu().numpy()
    mlx_logits = run_mlx_segnet_nchw(segnet_mlx, last_frame_np)
    return torch_logits, mlx_logits


def _run_posenet_pair(
    *,
    posenet_torch: Any,
    posenet_mlx: Any,
    rgb_pair_btchw: np.ndarray,  # (B, T=2, C=3, H_render=384, W_render=512), [0,255] float32
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run PoseNet on a pair. Returns ({pose: torch_pose}, {pose: mlx_pose})."""
    import einops
    import torch
    from frame_utils import rgb_to_yuv6, segnet_model_input_size  # type: ignore

    from tac.local_acceleration.mlx_scorer_adapters import run_mlx_posenet_nchw

    target_h, target_w = segnet_model_input_size[1], segnet_model_input_size[0]
    x_t = torch.from_numpy(rgb_pair_btchw.astype(np.float32, copy=True))
    batch_size, seq_len = x_t.shape[0], x_t.shape[1]
    # canonical preprocess (mirror of PoseNet.preprocess_input)
    x_t = einops.rearrange(
        x_t, "b t c h w -> (b t) c h w", b=batch_size, t=seq_len, c=3
    )
    if x_t.shape[-2] != target_h or x_t.shape[-1] != target_w:
        x_t = torch.nn.functional.interpolate(
            x_t, size=(target_h, target_w), mode="bilinear"
        )
    x_yuv6 = einops.rearrange(
        rgb_to_yuv6(x_t),
        "(b t) c h w -> b (t c) h w",
        b=batch_size,
        t=seq_len,
        c=6,
    )

    with torch.no_grad():
        torch_out = posenet_torch(x_yuv6)
    torch_out_np = {k: v.detach().cpu().numpy() for k, v in torch_out.items()}
    mlx_out_np = run_mlx_posenet_nchw(posenet_mlx, x_yuv6.detach().cpu().numpy())
    return torch_out_np, mlx_out_np


def _measure_stage3_segnet_drift(
    *,
    torch_logits: np.ndarray,
    mlx_logits: np.ndarray,
) -> StageDriftMeasurement:
    max_abs, mean_abs, rms = _aggregate_drift_max_abs(torch_logits, mlx_logits)
    torch_argmax = torch_logits.argmax(axis=1)
    mlx_argmax = mlx_logits.argmax(axis=1)
    argmax_flips = int((torch_argmax != mlx_argmax).sum())
    total_pixels = int(torch_argmax.size)
    return StageDriftMeasurement(
        stage_name="stage_3_segnet_forward",
        metric="per_pixel_logit_max_abs_plus_argmax_flip_count",
        max_abs=max_abs,
        mean_abs=mean_abs,
        rms=rms,
        extra={
            "logits_shape": [int(d) for d in torch_logits.shape],
            "argmax_flip_pixels": argmax_flips,
            "total_pixels": total_pixels,
            "argmax_flip_fraction": float(argmax_flips) / float(total_pixels)
            if total_pixels
            else 0.0,
            "theoretical_prediction": "extremely unlikely argmax flips at <3.05e-5 logit drift",
        },
    )


def _measure_stage4_posenet_drift(
    *,
    torch_pose: dict[str, np.ndarray],
    mlx_pose: dict[str, np.ndarray],
) -> StageDriftMeasurement:
    # PoseNet hydra returns {pose: (B, 12)}; only first 6 dims are used per
    # PoseNet.compute_distortion (out[..., : h.out // 2]).
    torch_p = torch_pose["pose"][..., :6]
    mlx_p = mlx_pose["pose"][..., :6]
    max_abs, mean_abs, rms = _aggregate_drift_max_abs(torch_p, mlx_p)
    rel_drift = float(
        (np.abs(torch_p - mlx_p) / (np.abs(torch_p) + 1e-9)).mean()
    )
    return StageDriftMeasurement(
        stage_name="stage_4_posenet_forward",
        metric="per_coord_pose_6vec_max_abs",
        max_abs=max_abs,
        mean_abs=mean_abs,
        rms=rms,
        extra={
            "pose_shape": [int(d) for d in torch_p.shape],
            "mean_relative_drift": rel_drift,
            "theoretical_prediction": "~3e-5 relative drift (~3e-5 contest-score units)",
        },
    )


# ----------------------------------------------------------------------------
# Stage 5: Contest score aggregation.
# ----------------------------------------------------------------------------
def _compute_distortion_seg(torch_logits: np.ndarray, mlx_logits: np.ndarray) -> tuple[float, float, float]:
    """Mirror of SegNet.compute_distortion across an N-pair batch.

    Returns (d_seg_torch_vs_torch, d_seg_mlx_vs_mlx, d_seg_pairwise_torch_vs_mlx).
    The first two are placeholders (always 0); the third is the cross-framework
    distortion (NOT the contest's pair-of-frames distortion; this is a MLX vs
    PyTorch drift propagation measurement, not a contest score).
    """
    torch_argmax = torch_logits.argmax(axis=1)
    mlx_argmax = mlx_logits.argmax(axis=1)
    diff = (torch_argmax != mlx_argmax).astype(np.float64)
    # Reduce over (H, W) per pair
    diff_per_pair = diff.reshape(diff.shape[0], -1).mean(axis=1)
    return 0.0, 0.0, float(diff_per_pair.mean())


def _compute_distortion_pose(
    torch_pose: dict[str, np.ndarray],
    mlx_pose: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    """Mirror of PoseNet.compute_distortion."""
    t = torch_pose["pose"][..., :6]
    m = mlx_pose["pose"][..., :6]
    # MSE per pair (mean over the 6-vec)
    mse_per_pair = ((t - m) ** 2).mean(axis=-1)
    return 0.0, 0.0, float(mse_per_pair.mean())


def _aggregate_contest_score_drift(
    *,
    segnet_torch_logits: np.ndarray,
    segnet_mlx_logits: np.ndarray,
    posenet_torch_pose: dict[str, np.ndarray],
    posenet_mlx_pose: dict[str, np.ndarray],
) -> StageDriftMeasurement:
    """Compute the contest-score drift due to MLX vs PyTorch framework drift.

    The contest score formula is S = 100*d_seg + sqrt(10*d_pose) + rate term.
    Rate term is INVARIANT (same archive bytes), so the framework drift's
    contest-score-units impact is bounded by:

      |delta S| <= 100 * d_seg_cross_framework + sqrt(10) * |sqrt(d_pose_T) - sqrt(d_pose_M)|

    Where d_seg_cross_framework is the SegNet argmax disagreement rate between
    MLX and PyTorch (NOT the contest's pair-disagreement rate).
    """
    _, _, d_seg_cross = _compute_distortion_seg(segnet_torch_logits, segnet_mlx_logits)
    _, _, d_pose_cross = _compute_distortion_pose(posenet_torch_pose, posenet_mlx_pose)

    # Per Selfcomp+MacKay: the seg contribution scales as 100 * d_seg_cross_framework.
    # The pose contribution scales as sqrt(10 * |d_pose_T - d_pose_M|) (the
    # non-linear contest formula amplifies near zero but vanishes asymptotically).
    # We report d_pose_cross as the MSE of (torch_pose - mlx_pose).
    seg_contest_delta = CANONICAL_SEG_MULTIPLIER * d_seg_cross
    pose_contest_delta = float(np.sqrt(CANONICAL_POSE_SQRT_INNER * d_pose_cross))
    aggregate_delta = seg_contest_delta + pose_contest_delta

    if aggregate_delta < CONTEST_SCORE_PRECISION_LO:
        verdict = "BELOW_SCORER_PRECISION"
    elif aggregate_delta < CONTEST_SCORE_PRECISION_HI:
        verdict = "AT_SCORER_PRECISION_BOUNDARY"
    else:
        verdict = "ABOVE_SCORER_PRECISION"

    return StageDriftMeasurement(
        stage_name="stage_5_contest_score_aggregation",
        metric="aggregate_contest_score_drift_due_to_mlx_vs_pytorch_framework",
        max_abs=aggregate_delta,
        mean_abs=aggregate_delta,  # single scalar aggregate
        rms=aggregate_delta,
        extra={
            "d_seg_cross_framework": d_seg_cross,
            "d_pose_cross_framework_mse": d_pose_cross,
            "seg_contest_delta_units": seg_contest_delta,
            "pose_contest_delta_units": pose_contest_delta,
            "aggregate_contest_delta_units": aggregate_delta,
            "verdict_per_stage": verdict,
            "precision_lo": CONTEST_SCORE_PRECISION_LO,
            "precision_hi": CONTEST_SCORE_PRECISION_HI,
            "formula": (
                "deltaS <= 100*d_seg_cross + sqrt(10*d_pose_cross_MSE) "
                "(rate term invariant: identical archive bytes between runs)"
            ),
            "theoretical_prediction_selfcomp_mackay": (
                "BELOW_SCORER_PRECISION (~3e-5 contest-score units; negligible "
                "vs 0.001 frontier-relevant precision)"
            ),
        },
    )


# ----------------------------------------------------------------------------
# Main measurement driver.
# ----------------------------------------------------------------------------
def measure_full_decoder_downstream_drift(
    *,
    archive_zip: Path,
    start_pair: int,
    n_pairs: int,
    seed: int,
    posenet_sd_path: Path,
    segnet_sd_path: Path,
    checkpoint_mode: str,
    scorer_input_mode: str,
) -> dict[str, Any]:
    """Measure 5-stage MLX-vs-PyTorch drift propagation. Returns canonical
    machine-readable manifest."""
    from tac.local_acceleration.pr95_hnerv_mlx import (
        parse_pr95_public_archive_zip,
    )

    packet = parse_pr95_public_archive_zip(archive_zip)
    total_pairs = int(packet.latents.shape[0])
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")
    if start_pair < 0 or start_pair >= total_pairs:
        raise ValueError(f"start_pair out of range: {start_pair} for {total_pairs} pairs")
    n_pairs = min(n_pairs, total_pairs - start_pair)

    # Stage 1 + 2: render N pairs through both frameworks.
    (
        rgb_mlx,
        rgb_torch,
        mlx_decode_sec,
        torch_decode_sec,
        pair_indices,
    ) = _render_pair_batch(
        state_dict_mlx=packet.state_dict,
        latents=packet.latents,
        meta=packet.meta,
        start_pair=start_pair,
        n_pairs=n_pairs,
        seed=seed,
    )

    stage_1 = _measure_stage1_decoder_drift(rgb_mlx=rgb_mlx, rgb_torch=rgb_torch)
    stage_2 = _measure_stage2_uint8_drift(rgb_mlx=rgb_mlx, rgb_torch=rgb_torch)
    if scorer_input_mode == "contest_uint8":
        scorer_rgb_mlx = _quantize_rgb_to_uint8(rgb_mlx).astype(np.float32)
        scorer_rgb_torch = _quantize_rgb_to_uint8(rgb_torch).astype(np.float32)
    elif scorer_input_mode == "decoder_float":
        scorer_rgb_mlx = rgb_mlx.astype(np.float32, copy=False)
        scorer_rgb_torch = rgb_torch.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported scorer_input_mode: {scorer_input_mode}")

    # Stage 3 + 4: SegNet + PoseNet on the rendered pairs. We feed the FLOAT
    # arrays that the selected boundary makes visible to the scorer. Default is
    # contest_uint8 because inflate writes uint8 RGB before auth scoring. The
    # decoder_float mode is diagnostic only and cannot be used as contest-path
    # closure evidence.
    posenet_torch, segnet_torch, posenet_mlx, segnet_mlx = _build_scorer_pair(
        posenet_sd_path=posenet_sd_path,
        segnet_sd_path=segnet_sd_path,
    )

    # Process in batches to control memory: ~5-10 pairs at a time for the
    # SegNet (37MB) + PoseNet (53MB) forward passes.
    BATCH = 4
    seg_torch_chunks: list[np.ndarray] = []
    seg_mlx_chunks: list[np.ndarray] = []
    pose_torch_chunks: list[dict[str, np.ndarray]] = []
    pose_mlx_chunks: list[dict[str, np.ndarray]] = []
    for start in range(0, n_pairs, BATCH):
        end = min(start + BATCH, n_pairs)
        rgb_torch_batch = scorer_rgb_torch[start:end]
        # SegNet pair
        seg_t, seg_m = _run_segnet_pair(
            segnet_torch=segnet_torch,
            segnet_mlx=segnet_mlx,
            # IMPORTANT: SegNet's torch reference must be fed the SAME float32
            # frames it would see at contest eval; we feed PyTorch's HNeRV
            # output (rgb_torch[start:end]) to BOTH torch_segnet (giving the
            # baseline) and again, separately, MLX's rgb output to the MLX
            # scorer. But this would conflate decoder drift with scorer drift.
            # Correct approach per task spec: feed the SAME RGB pair to both
            # SegNet/PoseNet so the drift we measure is the SCORER drift on a
            # fixed input. Then in Stage 5 we measure the combined effect.
            #
            # For decoder + scorer end-to-end propagation we feed PyTorch
            # decoder's RGB to PyTorch SegNet, MLX decoder's RGB to MLX
            # SegNet, and measure their disagreement.
            rgb_pair_btchw=rgb_torch_batch,
        )
        # MLX path: feed MLX decoder's RGB to MLX SegNet (separate forward).
        _seg_t_on_mlx_path, seg_m_mlx_path = _run_segnet_pair(
            segnet_torch=segnet_torch,
            segnet_mlx=segnet_mlx,
            rgb_pair_btchw=scorer_rgb_mlx[start:end],
        )
        # The "PyTorch reference" is segnet_torch on PyTorch's HNeRV output.
        # The "MLX path" is segnet_mlx on MLX's HNeRV output. We compare
        # these two end-to-end outputs.
        seg_torch_chunks.append(seg_t)  # torch decoder -> torch segnet
        seg_mlx_chunks.append(seg_m_mlx_path)  # mlx decoder -> mlx segnet

        pose_t, pose_m = _run_posenet_pair(
            posenet_torch=posenet_torch,
            posenet_mlx=posenet_mlx,
            rgb_pair_btchw=rgb_torch_batch,
        )
        _pose_t_on_mlx_path, pose_m_mlx_path = _run_posenet_pair(
            posenet_torch=posenet_torch,
            posenet_mlx=posenet_mlx,
            rgb_pair_btchw=scorer_rgb_mlx[start:end],
        )
        pose_torch_chunks.append(pose_t)
        pose_mlx_chunks.append(pose_m_mlx_path)

    seg_torch_all = np.concatenate(seg_torch_chunks, axis=0)
    seg_mlx_all = np.concatenate(seg_mlx_chunks, axis=0)
    pose_torch_all = {
        k: np.concatenate([c[k] for c in pose_torch_chunks], axis=0)
        for k in pose_torch_chunks[0]
    }
    pose_mlx_all = {
        k: np.concatenate([c[k] for c in pose_mlx_chunks], axis=0)
        for k in pose_mlx_chunks[0]
    }

    stage_3 = _measure_stage3_segnet_drift(
        torch_logits=seg_torch_all, mlx_logits=seg_mlx_all
    )
    stage_4 = _measure_stage4_posenet_drift(
        torch_pose=pose_torch_all, mlx_pose=pose_mlx_all
    )
    stage_5 = _aggregate_contest_score_drift(
        segnet_torch_logits=seg_torch_all,
        segnet_mlx_logits=seg_mlx_all,
        posenet_torch_pose=pose_torch_all,
        posenet_mlx_pose=pose_mlx_all,
    )

    aggregate_verdict = stage_5.extra["verdict_per_stage"]

    provenance = _canonical_provenance(checkpoint_mode=checkpoint_mode, n_pairs=n_pairs)
    posenet_sha256 = _sha256_file(posenet_sd_path)
    segnet_sha256 = _sha256_file(segnet_sd_path)
    return {
        "schema": "pr95_mlx_pytorch_full_decoder_downstream_scorer_drift_v1",
        "schema_version": "pr95_mlx_pytorch_full_decoder_downstream_scorer_drift_v1",
        "lane_id": "lane_pr95_mlx_full_decoder_downstream_scorer_drift_measurement_20260525",
        "task_id": "#1258",
        "archive_zip": str(archive_zip),
        "archive_zip_sha256": packet.archive_zip_sha256,
        "posenet_sha256": posenet_sha256,
        "segnet_sha256": segnet_sha256,
        "scorer_components": {
            "posenet_sha256": posenet_sha256,
            "segnet_sha256": segnet_sha256,
        },
        "n_pairs_actual": int(n_pairs),
        "covered_pair_window": [int(pair_indices.min()), int(pair_indices.max()) + 1],
        "covered_pair_indices_sha256": _sha256_int64_array(pair_indices),
        "seed": int(seed),
        "checkpoint_mode": checkpoint_mode,
        "scorer_input_mode": scorer_input_mode,
        "mlx_decode_seconds": mlx_decode_sec,
        "torch_decode_seconds": torch_decode_sec,
        "stages": [
            stage_1.as_dict(),
            stage_2.as_dict(),
            stage_3.as_dict(),
            stage_4.as_dict(),
            stage_5.as_dict(),
        ],
        "aggregate_verdict": aggregate_verdict,
        "aggregate_contest_score_drift_units": stage_5.max_abs,
        "selfcomp_mackay_theoretical_prediction_verified": (
            aggregate_verdict == "BELOW_SCORER_PRECISION"
        ),
        "carmack_mvp_first_step_2_prediction": "BELOW_SCORER_PRECISION",
        "carmack_mvp_first_step_2_falsified": (
            aggregate_verdict != "BELOW_SCORER_PRECISION"
        ),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "axis_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "provenance": provenance,
        "blockers": [
            "macos_mlx_research_signal_not_contest_authority",
            "requires_paired_contest_cpu_plus_cuda_for_score_claim",
            "drift_measurement_is_implementation_parity_not_scorer_response",
            *(
                ["scorer_input_mode_decoder_float_not_contest_path"]
                if scorer_input_mode != "contest_uint8"
                else []
            ),
        ],
        "operator_routable_per_verdict": {
            "BELOW_SCORER_PRECISION": (
                "Local MLX-vs-PyTorch downstream drift is below the configured "
                "frontier-precision threshold for this sampled PR95-class path. "
                "Use this as engineering-bridge evidence only; keep exact "
                "contest CPU/CUDA anchors for promotion, score claims, and any "
                "hardware-sensitive kill/rank decision."
            ),
            "AT_SCORER_PRECISION_BOUNDARY": (
                "Continue 5-priority queue. Investigate SECONDARY (Boyd ADMM "
                "stacked Kahan+FP64) + QUATERNARY (Daubechies extended sweep)."
            ),
            "ABOVE_SCORER_PRECISION": (
                "Escalate to paired exact-hardware reference measurement before "
                "using local MLX rows for any spend-triage bridge decision."
            ),
        }[aggregate_verdict],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure MLX-to-PyTorch drift propagation through full PR95 "
            "HNeRVDecoder -> SegNet/PoseNet -> contest score formula. "
            "[macOS-MLX research-signal] per CLAUDE.md."
        )
    )
    parser.add_argument(
        "--archive-zip",
        type=Path,
        default=Path(
            ".omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z"
            "/matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f"
            "/pr95_packaged_submission/archive.zip"
        ),
        help="Path to PR 95 single-member archive ZIP (default = Slot 1 anchor).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=100,
        help=(
            "Number of frame pairs to measure (default=100; 600 = canonical "
            "contest video full set; 100 reasonable wall-clock balance)."
        ),
    )
    parser.add_argument(
        "--start-pair",
        type=int,
        default=0,
        help=(
            "First pair index covered by the measurement window. Use with "
            "--n-pairs to bind drift proof to a scorer-response pair window."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--posenet-sd",
        type=Path,
        default=Path("upstream/models/posenet.safetensors"),
        help="Path to canonical PoseNet safetensors.",
    )
    parser.add_argument(
        "--segnet-sd",
        type=Path,
        default=Path("upstream/models/segnet.safetensors"),
        help="Path to canonical SegNet safetensors.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        type=str,
        default="trained_archive",
        choices=("trained_archive", "random_init"),
        help="Checkpoint provenance label for the manifest row.",
    )
    parser.add_argument(
        "--scorer-input-mode",
        type=str,
        default="contest_uint8",
        choices=("contest_uint8", "decoder_float"),
        help=(
            "Boundary passed into SegNet/PoseNet. contest_uint8 mirrors auth "
            "inflate output; decoder_float is diagnostic scorer drift only."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Default = "
            "experiments/results/pr95_mlx_pytorch_full_decoder_downstream_drift_<utc>/results.json"
        ),
    )
    parser.add_argument(
        "--register-canonical-equation",
        action="store_true",
        help=(
            "Register mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1 "
            "from the emitted result JSON. This appends to the canonical equations "
            "ledger and does not grant score or dispatch authority."
        ),
    )
    parser.add_argument(
        "--register-from-result-json",
        type=Path,
        default=None,
        help=(
            "Register mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1 "
            "from an existing result JSON and skip measurement."
        ),
    )
    args = parser.parse_args(argv)

    if args.register_from_result_json is not None:
        from tac.canonical_equations import (
            build_mlx_pytorch_drift_equation_from_result_json,
            register_canonical_equation,
        )

        result_json = args.register_from_result_json
        if not result_json.is_absolute():
            result_json = REPO_ROOT / result_json
        if not result_json.is_file():
            print(f"ERROR: result JSON not found: {result_json}", file=sys.stderr)
            return 2
        equation = build_mlx_pytorch_drift_equation_from_result_json(result_json)
        register_canonical_equation(
            equation,
            agent="codex",
            subagent_id="codex_mlx_downstream_drift_equation_registration_20260525",
            notes="Registered from existing full-decoder downstream drift result JSON; false-authority preserved.",
        )
        print(f"Registered canonical equation: {equation.equation_id}")
        print(f"Source result JSON: {result_json}")
        return 0

    if args.output_json is None:
        utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_json = (
            REPO_ROOT
            / "experiments"
            / "results"
            / f"pr95_mlx_pytorch_full_decoder_downstream_drift_{utc}"
            / "results.json"
        )

    archive_zip = args.archive_zip
    if not archive_zip.is_absolute():
        archive_zip = REPO_ROOT / archive_zip
    posenet_sd = args.posenet_sd
    if not posenet_sd.is_absolute():
        posenet_sd = REPO_ROOT / posenet_sd
    segnet_sd = args.segnet_sd
    if not segnet_sd.is_absolute():
        segnet_sd = REPO_ROOT / segnet_sd

    if not archive_zip.is_file():
        print(f"ERROR: archive ZIP not found: {archive_zip}", file=sys.stderr)
        return 2
    if not posenet_sd.is_file():
        print(f"ERROR: PoseNet safetensors not found: {posenet_sd}", file=sys.stderr)
        return 2
    if not segnet_sd.is_file():
        print(f"ERROR: SegNet safetensors not found: {segnet_sd}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    manifest = measure_full_decoder_downstream_drift(
        archive_zip=archive_zip,
        start_pair=args.start_pair,
        n_pairs=args.n_pairs,
        seed=args.seed,
        posenet_sd_path=posenet_sd,
        segnet_sd_path=segnet_sd,
        checkpoint_mode=args.checkpoint_mode,
        scorer_input_mode=args.scorer_input_mode,
    )
    elapsed = time.perf_counter() - started
    manifest["wall_clock_seconds"] = elapsed

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"Wrote {args.output_json}")
    print(f"Aggregate verdict: {manifest['aggregate_verdict']}")
    print(f"Aggregate contest-score drift: {manifest['aggregate_contest_score_drift_units']:.6e}")
    print(f"Wall clock: {elapsed:.1f}s")
    if args.register_canonical_equation:
        from tac.canonical_equations import (
            build_mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1,
            register_canonical_equation,
        )

        equation = build_mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1(
            manifest,
            source_artifact=str(args.output_json),
        )
        register_canonical_equation(
            equation,
            agent="codex",
            subagent_id="codex_mlx_downstream_drift_equation_registration_20260525",
            notes="Registered from full-decoder downstream drift CLI; false-authority preserved.",
        )
        print(f"Registered canonical equation: {equation.equation_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
