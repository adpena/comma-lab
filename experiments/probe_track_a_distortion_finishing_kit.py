# SPDX-License-Identifier: MIT
"""Track-A DISTORTION finishing-kit MEASUREMENT probe (base_ch=20 HNeRV basin).

$0 CPU, REAL frozen contest scorer + REAL ``frame_utils.yuv420_to_rgb`` GT
(NO MPS, NO paid spend). Measures the four Track-A inflate-side distortion
bolt-ons on the base_ch=20 basin fork-point:

  A. PR98 channel-bias RE-FIT — re-derive the per-(frame,channel) constant bias
     that minimizes the REAL-scorer ``100*d_seg + sqrt(10*d_pose)`` on the
     basin render-vs-GT (the canonical PR101 constants are substrate-specific).
  B. T10 affine color correction — fit a per-channel affine (scale+bias) on the
     camera-res frames the scorer actually reads, measure gain BEYOND PR98.
  C. S12 resize-null preimage — measure the certified-invisible fraction + the
     real byte savings of filling the null region with compressible values, and
     CONFIRM the zero-distortion certification holds through the eval roundtrip.
  D. LeverD margin-conditional seg-repair — measure |B| flip concentration +
     conditional B/flip + the waterfill NET ΔS on the basin's REAL flip set; emit
     a measured GO/NO-GO verdict.

Authority: ``[contest-CPU advisory] NON-PROMOTABLE``. Every number is a
frozen-CPU advisory measurement on a mid-basin fork-point; no byte-closed
archive row, no ``upstream/evaluate.py`` row. The frontier is UNMOVED. This is a
MEANS (a fit + go/no-go measurement) toward the END (a lower exact score on the
converged decoder); it moves no row.

The scorer reads CAMERA-RES uint8 frames ``(B, 2, H, W, 3)`` (per the vendored
``score.evaluate_decoder``): decoder(z) -> (B,2,3,384,512) float -> bicubic↑ to
(874,1164) -> permute to BHWC -> clamp/round/uint8 -> ``distortion_net``. The
distortion-kit postproc operates on the camera-res frames at exactly that point
(the canonical inflate-side hook), so the fit minimizes what the scorer sees.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_EVAL_H, _EVAL_W = 384, 512
_CAMERA_H, _CAMERA_W = 874, 1164
_ADVISORY = "[contest-CPU advisory] NON-PROMOTABLE"

_BASIN = Path("experiments/results/forkpoints/basin_bc20_20260612T121523Z")


# ---------------------------------------------------------------------------
# Rendering + GT + scorer plumbing (1:1 with the vendored score.evaluate_decoder)
# ---------------------------------------------------------------------------
def _decoded_to_camera(decoded_native: torch.Tensor) -> torch.Tensor:
    """Bicubic ↑ (384,512) -> (874,1164) — 1:1 with vendored ``score._decoded_to_camera``."""
    return F.interpolate(
        decoded_native, size=(_CAMERA_H, _CAMERA_W), mode="bicubic", align_corners=False
    )


def _render_camera_pairs(
    decoder: torch.nn.Module, latents: torch.Tensor, n_pairs: int, *, batch: int = 8
) -> torch.Tensor:
    """Render basin pairs to CAMERA-RES FLOAT frames ``(n, 2, H, W, 3)`` (pre-clamp/round).

    Returns the FLOAT camera frames (so the affine/bias fit operates on the same
    continuous signal the scorer's round() snaps). The caller clamps/rounds/casts
    to uint8 (1:1 with the eval) AFTER any postproc."""
    decoder.eval()
    out_pairs = []
    with torch.inference_mode():
        for i in range(0, n_pairs, batch):
            b = min(batch, n_pairs - i)
            z = latents[i : i + b]
            decoded = decoder(z)  # (b,2,3,384,512) float [0,255]
            flat = decoded.reshape(b * 2, 3, _EVAL_H, _EVAL_W)
            up = _decoded_to_camera(flat)  # (b*2,3,874,1164)
            bhwc = up.reshape(b, 2, 3, _CAMERA_H, _CAMERA_W).permute(0, 1, 3, 4, 2)
            out_pairs.append(bhwc.contiguous())
    return torch.cat(out_pairs, dim=0)  # (n,2,H,W,3) float


def _decode_gt_pairs(video_path: str | Path, n_pairs: int) -> torch.Tensor:
    """Decode the first ``n_pairs`` GT pairs via canonical ``yuv420_to_rgb``.

    Returns ``(n, 2, H, W, 3)`` uint8. PyAV rgb24 is FORBIDDEN (phantom pose);
    ONLY ``frame_utils.yuv420_to_rgb`` per CLAUDE.md."""
    import av
    from frame_utils import yuv420_to_rgb

    cont = av.open(str(video_path))
    frames = []
    need = n_pairs * 2
    for frame in cont.decode(cont.streams.video[0]):
        frames.append(yuv420_to_rgb(frame))
        if len(frames) >= need:
            break
    cont.close()
    gt = torch.stack(frames[:need])  # (2n,H,W,3) uint8
    return gt.reshape(n_pairs, 2, _CAMERA_H, _CAMERA_W, 3)


def _frames_to_uint8(cam_float: torch.Tensor) -> torch.Tensor:
    """Clamp/round/cast camera FLOAT frames to uint8 — 1:1 with the eval cast."""
    return cam_float.clamp(0, 255).round().to(torch.uint8)


@torch.inference_mode()
def _measure_d_seg_d_pose(
    net: Any, gt_pairs_u8: torch.Tensor, cand_pairs_u8: torch.Tensor, *, batch: int = 8
) -> dict[str, float]:
    """REAL frozen-scorer d_seg / d_pose over the pairs (mean), 1:1 with
    ``distortion_net.compute_distortion``. Inputs are uint8 ``(n,2,H,W,3)``."""
    n = gt_pairs_u8.shape[0]
    seg_total = 0.0
    pose_total = 0.0
    for i in range(0, n, batch):
        b = min(batch, n - i)
        pose_d, seg_d = net.compute_distortion(
            gt_pairs_u8[i : i + b], cand_pairs_u8[i : i + b]
        )
        seg_total += float(seg_d.sum().item())
        pose_total += float(pose_d.sum().item())
    return {"d_seg": seg_total / n, "d_pose": pose_total / n}


@torch.inference_mode()
def _measure_bias_affine(
    net: Any,
    gt_pairs_u8: torch.Tensor,
    cam_float: torch.Tensor,
    scale: np.ndarray | None,
    bias: np.ndarray | None,
    *,
    batch: int = 8,
) -> dict[str, float]:
    """MEMORY-BOUNDED real-scorer d_seg/d_pose for the affine ``scale*x - bias``.

    Applies the per-(frame,channel) transform PER-BATCH (never clones the full
    camera tensor) so peak memory is one 8-pair batch — this is the fix for the
    silent-OOM the whole-tensor-clone path hit at n>=24. ``scale=None`` -> 1,
    ``bias=None`` -> 0 (the identity / baseline)."""
    n = gt_pairs_u8.shape[0]
    seg_total = 0.0
    pose_total = 0.0
    for i in range(0, n, batch):
        b = min(batch, n - i)
        chunk = cam_float[i : i + b].clone()  # (b,2,H,W,3) float — ONE batch only
        if scale is not None or bias is not None:
            for fr in range(2):
                for ch in range(3):
                    s = 1.0 if scale is None else float(scale[fr, ch])
                    bb = 0.0 if bias is None else float(bias[fr, ch])
                    if s == 1.0 and bb == 0.0:
                        continue
                    chunk[:, fr, :, :, ch] = chunk[:, fr, :, :, ch] * s - bb
        cand = chunk.clamp(0, 255).round().to(torch.uint8)
        pose_d, seg_d = net.compute_distortion(gt_pairs_u8[i : i + b], cand)
        seg_total += float(seg_d.sum().item())
        pose_total += float(pose_d.sum().item())
        del chunk, cand
    return {"d_seg": seg_total / n, "d_pose": pose_total / n}


def _seg_score(d_seg: float, d_pose: float) -> float:
    """The DISTORTION half of the contest score (rate excluded — these bolt-ons
    are 0-archive-byte except S12 which is measured separately)."""
    return 100.0 * d_seg + (10.0 * d_pose) ** 0.5


# ---------------------------------------------------------------------------
# Color-statistics fits (analytic) — minimize the real scorer-input mismatch.
#
# The scorer reads the camera-res frame. Aligning the rendered frame's per-channel
# statistics to the GT's reduces the systematic color-space bias the decoder
# carries (the L28/PR98 mechanism). We derive the fit ANALYTICALLY from the
# render-vs-GT channel statistics (cheap), then MEASURE the real-scorer gain of
# the derived fit + the canonical PR98 anchor + a tiny integer refinement around
# the analytic bias (so the verdict is real-scorer-measured, not just analytic).
# ---------------------------------------------------------------------------
def _channel_stats(cam_float: torch.Tensor, gt_u8: torch.Tensor) -> dict[str, Any]:
    """Per-(frame,channel) means + the bias/affine that aligns render->GT.

    bias[fr,ch]   = mean(render) - mean(GT)         (subtract to match the GT mean)
    scale[fr,ch]  = std(GT) / std(render)           (variance match)
    affine_bias   = mean(render) - scale*mean(GT)   (so scale*x - affine_bias ~ GT)
    Computed on the camera FLOAT render and uint8 GT.
    """
    gt_f = gt_u8.float()
    r_mean = cam_float.mean(dim=(0, 2, 3))  # (2,3) over pairs,H,W
    g_mean = gt_f.mean(dim=(0, 2, 3))
    r_std = cam_float.std(dim=(0, 2, 3))
    g_std = gt_f.std(dim=(0, 2, 3))
    bias = (r_mean - g_mean).cpu().numpy()  # (2,3) subtract this to match GT mean
    scale = (g_std / r_std.clamp_min(1e-6)).cpu().numpy()  # (2,3)
    affine_bias = (r_mean.cpu().numpy() - scale * g_mean.cpu().numpy())  # x*scale - affine_bias ~ GT
    return {
        "render_mean": r_mean.cpu().numpy(),
        "gt_mean": g_mean.cpu().numpy(),
        "render_std": r_std.cpu().numpy(),
        "gt_std": g_std.cpu().numpy(),
        "mean_align_bias": bias,
        "std_align_scale": scale,
        "affine_bias": affine_bias,
    }


# ---------------------------------------------------------------------------
# A. PR98 channel-bias RE-FIT (analytic mean-align + measure-confirm)
# ---------------------------------------------------------------------------
def refit_pr98_bias(
    net: Any,
    gt_u8: torch.Tensor,
    cam_float: torch.Tensor,
    *,
    refine: bool = True,
    base_distortion: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Re-fit the per-(frame,channel) constant bias minimizing real d_seg+d_pose.

    The canonical PR98 form subtracts 1.0 from frame_0 R, frame_0 B, frame_1 G.
    Those constants are PR101-specific. We RE-DERIVE the bias analytically as the
    per-(frame,channel) mean-alignment ``mean(render) - mean(GT)`` and MEASURE the
    REAL-scorer distortion score of: (1) base, (2) canonical PR98 (-1 on the 3
    canonical slots), (3) the analytic mean-align bias, (4) a BOUNDED single-pass
    ±1 integer refinement around the best seed (so the chosen constants are
    real-scorer-measured, not just analytic). SegNet reads ONLY frame_1 so a
    frame_0 bias can move ONLY d_pose; a frame_1 bias moves BOTH.

    MEMORY: every measurement applies the bias per-batch (``_measure_bias_affine``)
    — never clones the whole camera tensor (the n>=24 OOM fix)."""
    if base_distortion is None:
        base_distortion = _measure_bias_affine(net, gt_u8, cam_float, None, None)
    base_score = _seg_score(base_distortion["d_seg"], base_distortion["d_pose"])

    stats = _channel_stats(cam_float, gt_u8)
    analytic_bias = stats["mean_align_bias"]  # (2,3)

    candidates: dict[str, np.ndarray] = {"zero": np.zeros((2, 3))}
    canon = np.zeros((2, 3))
    canon[0, 0] = 1.0  # frame_0 R -1
    canon[0, 2] = 1.0  # frame_0 B -1
    canon[1, 1] = 1.0  # frame_1 G -1
    candidates["canonical_pr98"] = canon
    candidates["analytic_mean_align"] = analytic_bias.copy()
    candidates["analytic_rounded"] = np.round(analytic_bias)

    measured: dict[str, dict[str, float]] = {}
    for name, bias in candidates.items():
        if name == "zero":
            m = base_distortion
        else:
            m = _measure_bias_affine(net, gt_u8, cam_float, None, bias)
        measured[name] = {
            "d_seg": m["d_seg"],
            "d_pose": m["d_pose"],
            "distortion_score": _seg_score(m["d_seg"], m["d_pose"]),
            "bias_frame_channel": bias.tolist(),
        }

    best_name = min(measured, key=lambda k: measured[k]["distortion_score"])
    best_bias = np.asarray(candidates[best_name], dtype=np.float64).copy()
    best_score = measured[best_name]["distortion_score"]
    refine_trace = []
    if refine:
        # BOUNDED single coordinate pass: each of 6 slots, try ±1 once, keep if it helps.
        for fr in range(2):
            for ch in range(3):
                for step in (-1.0, 1.0):
                    trial = best_bias.copy()
                    trial[fr, ch] += step
                    m = _measure_bias_affine(net, gt_u8, cam_float, None, trial)
                    sc = _seg_score(m["d_seg"], m["d_pose"])
                    if sc < best_score - 1e-12:
                        best_score, best_bias = sc, trial
                        refine_trace.append(
                            {"frame": fr, "channel": ch, "bias": trial.tolist(), "score": sc}
                        )
                        break  # one accepted step per slot (bounded)
    fm = _measure_bias_affine(net, gt_u8, cam_float, None, best_bias)
    final_score = _seg_score(fm["d_seg"], fm["d_pose"])

    return {
        "base_d_seg": base_distortion["d_seg"],
        "base_d_pose": base_distortion["d_pose"],
        "base_distortion_score": base_score,
        "channel_stats": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()},
        "candidates_measured": measured,
        "best_seed_candidate": best_name,
        "best_bias_frame_channel": best_bias.tolist(),
        "refine_trace": refine_trace,
        "fit_d_seg": fm["d_seg"],
        "fit_d_pose": fm["d_pose"],
        "fit_distortion_score": final_score,
        "joint_delta_vs_base": final_score - base_score,
        "joint_fit_distortion_score": final_score,  # back-compat alias
    }


# ---------------------------------------------------------------------------
# B. T10 affine color correction (analytic std-align + measure-confirm)
# ---------------------------------------------------------------------------
def fit_t10_affine(
    net: Any, gt_u8: torch.Tensor, cam_float: torch.Tensor, pr98_bias: np.ndarray
) -> dict[str, Any]:
    """Fit a per-(frame,channel) affine ``x -> scale*x - bias`` (2nd-order beyond
    PR98's pure bias), measured on the REAL scorer.

    FINDING (the n=4 smoke): the closed-form mean/std-alignment OVER-corrects (it
    aligns the FULL channel statistics, but the render already matches the scorer's
    argmax well at d_seg~0.0037, so a full-mean shift moves the input AWAY from the
    decision boundary). The score lever is therefore the SMALL local affine around
    the PR98-refit operating point, NOT the global stat-match. We SEED from PR98
    (scale=1, the re-fit bias) and refine scale per slot by SMALL steps around 1.0;
    we ALSO measure the closed-form analytic affine as an HONEST comparison row (it
    is reported, not selected, so the over-correction is visible). MEMORY: per-batch
    transform (``_measure_bias_affine``)."""
    pr98_m = _measure_bias_affine(net, gt_u8, cam_float, None, pr98_bias)
    pr98_score = _seg_score(pr98_m["d_seg"], pr98_m["d_pose"])

    stats = _channel_stats(cam_float, gt_u8)
    analytic_scale = stats["std_align_scale"]  # (2,3)
    analytic_bias = stats["affine_bias"]  # (2,3) so scale*x - bias ~ GT
    # HONEST comparison row: the closed-form stat-match (expected to over-correct).
    am = _measure_bias_affine(net, gt_u8, cam_float, analytic_scale, analytic_bias)
    analytic_score = _seg_score(am["d_seg"], am["d_pose"])

    # SEED from PR98 (scale=1, the re-fit bias) — the small-local-affine operating
    # point. Refine scale per slot by SMALL steps; re-derive the bias for each scale
    # so the channel MEAN stays at the PR98-refit operating point (bias = pr98_bias
    # adjusted by the scale's effect on the render mean).
    rmean = np.asarray(stats["render_mean"])  # (2,3) render channel means
    best_scale = np.ones((2, 3), dtype=np.float64)
    best_bias = np.asarray(pr98_bias, dtype=np.float64).copy()
    best_score = pr98_score
    scale_steps = (-0.02, -0.01, 0.01, 0.02)
    for fr in range(2):
        for ch in range(3):
            for step in scale_steps:
                trial_s = best_scale.copy()
                trial_s[fr, ch] += step
                # keep the post-transform channel mean equal to the PR98 operating
                # point: scale*mean - bias' = 1*mean - pr98_bias  =>
                # bias' = pr98_bias + (scale-1)*mean
                trial_b = best_bias.copy()
                trial_b[fr, ch] = float(
                    pr98_bias[fr, ch] + (trial_s[fr, ch] - 1.0) * rmean[fr, ch]
                )
                m = _measure_bias_affine(net, gt_u8, cam_float, trial_s, trial_b)
                sc = _seg_score(m["d_seg"], m["d_pose"])
                if sc < best_score - 1e-12:
                    best_score, best_scale, best_bias = sc, trial_s, trial_b
                    break  # bounded: one accepted step per slot

    fm = _measure_bias_affine(net, gt_u8, cam_float, best_scale, best_bias)
    joint_score = _seg_score(fm["d_seg"], fm["d_pose"])

    return {
        "pr98_only_distortion_score": pr98_score,
        "analytic_affine_score": analytic_score,
        "analytic_affine_over_corrects": analytic_score > pr98_score,
        "best_scale_frame_channel": best_scale.tolist(),
        "best_bias_frame_channel": best_bias.tolist(),
        "affine_d_seg": fm["d_seg"],
        "affine_d_pose": fm["d_pose"],
        "affine_distortion_score": joint_score,
        "affine_delta_vs_pr98": joint_score - pr98_score,
        "channel_stats": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()},
    }


# ---------------------------------------------------------------------------
# C. S12 resize-null preimage — invisible fraction + byte savings + zero-distortion
# ---------------------------------------------------------------------------
def probe_s12(
    net: Any, gt_u8: torch.Tensor, cam_float: torch.Tensor, *, n_proof: int = 4
) -> dict[str, Any]:
    """Measure the certified-invisible (resize-null) fraction + the byte savings of
    filling it with compressible values + confirm ZERO distortion through the real
    eval roundtrip.

    HONEST SCOPE: this HNeRV substrate stores decoder weights + latents, NOT
    frames — so the S12 byte savings only materialize IF a per-frame residual
    section is shipped. We report (a) the universal invisible-fraction +
    per-frame brotli savings (the would-be gain), and (b) the zero-distortion
    CERTIFICATION on the REAL scorer (so the fill is provably safe to apply to
    ANY stored-frame sidecar / residual section)."""
    from tac.optimization.resize_null_preimage import (
        ResizeProjector,
        apply_tier1_zero_weight_fill,
        zero_weight_pixel_mask,
    )

    projector = ResizeProjector.build(camera_h=_CAMERA_H, camera_w=_CAMERA_W)
    mask = zero_weight_pixel_mask(
        camera_h=_CAMERA_H, camera_w=_CAMERA_W,
        scorer_h=projector.scorer_h, scorer_w=projector.scorer_w,
    )
    invisible_frac = float(mask.mean())

    n_proof = min(n_proof, cam_float.shape[0])
    # Slice to the proof window BEFORE the numpy conversion (memory-bounded).
    cam_u8 = _frames_to_uint8(cam_float[:n_proof]).cpu().numpy()  # (n_proof,2,H,W,3)
    n = cam_u8.shape[0]
    # Apply the tier-1 fill to every frame; collect byte savings + max projection
    # residual (the CERTIFIED-exact proof: zero-weight pixels carry 0 resize weight).
    filled = cam_u8.copy()
    bytes_before_tot = 0
    bytes_after_tot = 0
    max_resid = 0.0
    n_frames_proofed = 0
    for p in range(min(n_proof, n)):
        for fr in range(2):
            frame = cam_u8[p, fr]  # (H,W,3)
            out, proof = apply_tier1_zero_weight_fill(
                frame, strategy="measured_best", mask=mask, projector=projector
            )
            filled[p, fr] = out
            bytes_before_tot += proof.bytes_before["brotli"]
            bytes_after_tot += proof.bytes_after["brotli"]
            max_resid = max(max_resid, float(proof.max_abs_projection_residual))
            n_frames_proofed += 1

    # Confirm the FILLED frames score IDENTICALLY on the REAL scorer (zero distortion).
    proof_slice = slice(0, n)
    base_score = _measure_d_seg_d_pose(net, gt_u8[proof_slice], _frames_to_uint8(cam_float[proof_slice]))
    filled_u8 = torch.from_numpy(filled)
    filled_score = _measure_d_seg_d_pose(net, gt_u8[proof_slice], filled_u8)

    byte_reduction = bytes_before_tot - bytes_after_tot
    pct = (byte_reduction / bytes_before_tot * 100.0) if bytes_before_tot else 0.0
    return {
        "invisible_fraction_of_camera_frame": invisible_frac,
        "n_frames_proofed": n_frames_proofed,
        "per_frame_brotli_bytes_before": bytes_before_tot,
        "per_frame_brotli_bytes_after": bytes_after_tot,
        "per_frame_brotli_byte_reduction": byte_reduction,
        "per_frame_brotli_pct_reduction": pct,
        "max_abs_projection_residual": max_resid,
        "certified_zero_distortion_proof": max_resid <= 0.0,
        "real_scorer_d_seg_before": base_score["d_seg"],
        "real_scorer_d_seg_after_fill": filled_score["d_seg"],
        "real_scorer_d_pose_before": base_score["d_pose"],
        "real_scorer_d_pose_after_fill": filled_score["d_pose"],
        "real_scorer_d_seg_unchanged": abs(base_score["d_seg"] - filled_score["d_seg"]) < 1e-9,
        "real_scorer_d_pose_unchanged": abs(base_score["d_pose"] - filled_score["d_pose"]) < 1e-9,
        "honest_note": (
            "HNeRV substrate stores decoder+latents, NOT frames. The per-frame "
            "byte reduction is the WOULD-BE gain on a stored-frame residual "
            "section; the zero-distortion certification is universal and means "
            "the fill is safe on any frame-carrying section."
        ),
    }


# ---------------------------------------------------------------------------
# D. LeverD margin-conditional seg-repair — GO/NO-GO
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _segnet_argmax_margin_one(net: Any, frames_u8_1: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """SegNet argmax + top-2 margin for ONE pair ``(1,2,H,W,3)`` uint8.

    Returns ``(argmax_flat (384*512,), margin_flat (384*512,))`` numpy — the
    per-pair SegNet output the decoder regenerates for free. Batched per-pair so
    no full-n logit tensor is ever held (the OOM fix)."""
    posenet_in, segnet_in = net.preprocess_input(frames_u8_1)  # last-frame, resized
    seg_out = net.segnet(segnet_in)  # (1,5,384,512)
    top2 = torch.topk(seg_out, k=2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).abs()  # (1,384,512)
    return (
        seg_out.argmax(dim=1).reshape(-1).cpu().numpy(),
        margin.reshape(-1).cpu().numpy(),
    )


def assess_lever_d(
    net: Any, gt_u8: torch.Tensor, cam_float: torch.Tensor, *, tau: float = 0.5
) -> dict[str, Any]:
    """Measure |B| flip concentration + conditional B/flip + the waterfill NET ΔS
    against the FRONTIER archive bytes, and emit a measured GO/NO-GO verdict.

    Per the witness probe: the per-flip break-even (1.27 B/flip) is necessary but
    NOT sufficient — the binding term is the absolute FLIP COUNT priced against the
    FRONTIER archive (not credited against the residual's own bytes). We reproduce
    that economics on the basin's REAL flip set, BATCHED per-pair (memory-bounded)."""
    from tac.boundary_math.margin_conditional_residual import (
        WATERLINE_BYTES_PER_FLIP,
        boundary_set_from_margin,
        measure_code_cost,
    )

    n = gt_u8.shape[0]
    total_flips = 0
    per_pair_bpf = []
    per_pair_cond_pos_below = []
    boundary_fracs = []
    for p in range(n):
        gt_pair = gt_u8[p : p + 1]
        cand_pair = _frames_to_uint8(cam_float[p : p + 1].clone())
        gt_argmax, _gt_margin = _segnet_argmax_margin_one(net, gt_pair)
        cand_argmax, cand_margin = _segnet_argmax_margin_one(net, cand_pair)
        flip_idx = np.where(gt_argmax != cand_argmax)[0]
        total_flips += int(flip_idx.size)
        B = boundary_set_from_margin(cand_margin, tau)
        boundary_fracs.append(float(B.mean()))
        if flip_idx.size == 0:
            continue
        target_cls = gt_argmax[flip_idx]  # the GT label = the scorer's GT output
        cost = measure_code_cost(cand_margin, flip_idx, target_cls, tau=tau)
        per_pair_bpf.append(cost.bytes_per_flip)
        per_pair_cond_pos_below.append(cost.bytes_per_flip < cost.unconditional_bytes_per_flip)

    flips_per_pair = total_flips / n
    d_seg = total_flips / (n * 384 * 512)
    cond_bpf = float(np.mean(per_pair_bpf)) if per_pair_bpf else float("inf")
    uncond_clears = bool(cond_bpf < WATERLINE_BYTES_PER_FLIP)

    # Scaled residual bytes over 600 pairs (the witness-probe killer term).
    total_flips_600 = flips_per_pair * 600.0
    residual_bytes_600 = total_flips_600 * cond_bpf

    # Frontier-relative economics (price the residual section against the frontier).
    frontier_bytes = _read_frontier_bytes()
    # Seg drop if ALL flips fixed (isolated): 100 * d_seg.
    seg_drop_isolated = 100.0 * d_seg
    rate_cost_residual = 25.0 * residual_bytes_600 / 37_545_489.0
    net_delta_s_full_sidecar = -seg_drop_isolated + rate_cost_residual

    # GO requires: conditional clears break-even AND the residual section priced
    # against the frontier is NET-negative ΔS (i.e. the seg win exceeds the rate
    # cost of the ADDED section). On a basin base the flip count is large.
    go = bool(uncond_clears and net_delta_s_full_sidecar < 0.0)
    verdict = "GO" if go else "NO-GO"
    return {
        "tau": tau,
        "flips_per_pair": flips_per_pair,
        "d_seg": d_seg,
        "mean_boundary_fraction": float(np.mean(boundary_fracs)) if boundary_fracs else 0.0,
        "conditional_bytes_per_flip": cond_bpf,
        "waterline_bytes_per_flip": float(WATERLINE_BYTES_PER_FLIP),
        "conditional_clears_break_even": uncond_clears,
        "conditional_below_unconditional_frac": (
            float(np.mean(per_pair_cond_pos_below)) if per_pair_cond_pos_below else 0.0
        ),
        "scaled_total_flips_600pairs": total_flips_600,
        "scaled_residual_bytes_600pairs": residual_bytes_600,
        "frontier_archive_bytes": frontier_bytes,
        "seg_drop_isolated_if_all_fixed": seg_drop_isolated,
        "rate_cost_of_residual_section": rate_cost_residual,
        "net_delta_s_full_sidecar_vs_frontier": net_delta_s_full_sidecar,
        "verdict": verdict,
        "rationale": (
            "GO requires conditional B/flip < break-even AND the residual section "
            "priced against the FRONTIER archive is net-negative ΔS. On a basin "
            "base the half-million flips make the ADDED section dominate the seg "
            "win (the witness-probe flip-count crux); the in-training margin-weighted "
            "seg loss banks the d_seg at ZERO added bytes (the §E.2 hybrid)."
        ),
    }


def _read_frontier_bytes() -> int:
    """Read the contest-CPU frontier archive bytes from the canonical pointer
    (SoT — never hardcoded). Falls back to the documented 177,169 B only if the
    pointer is unreadable."""
    try:
        ptr = json.loads(Path(".omx/state/canonical_frontier_pointer.json").read_text())
        node = ptr.get("our_local_frontier_contest_cpu", {})
        extra = node.get("extra", {}) if isinstance(node, dict) else {}
        if isinstance(extra, dict) and extra.get("archive_bytes"):
            return int(extra["archive_bytes"])
        if isinstance(node, dict) and node.get("archive_bytes"):
            return int(node["archive_bytes"])
    except Exception:
        pass
    return 177_169  # documented pointer fallback (bolton inventory memo)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _load_basin_decoder() -> tuple[torch.nn.Module, torch.Tensor, dict[str, Any]]:
    from tac.torch_vehicle.driver import import_vendored_bundle

    v = import_vendored_bundle()
    arch = (_BASIN / "best" / "best_archive.bin").read_bytes()
    dec_sd, latents, meta = v.parse_archive(arch)
    dec = v.HNeRVDecoder(
        latent_dim=meta.get("latent_dim", 28),
        base_channels=meta.get("base_channels", 20),
        eval_size=(_EVAL_H, _EVAL_W),
    )
    dec.load_state_dict({k: vv for k, vv in dec_sd.items()})
    dec.eval()
    return dec, latents, meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pairs", type=int, default=48, help="pairs to fit/measure on (CPU; 48~quick, 600~full)")
    p.add_argument("--n-proof", type=int, default=4, help="frames for the S12 byte/zero-distortion proof")
    p.add_argument("--tau", type=float, default=0.5, help="LeverD boundary margin threshold")
    p.add_argument("--out", type=Path, default=None, help="JSON output path")
    p.add_argument("--skip", nargs="*", default=[], choices=["A", "B", "C", "D"], help="skip sub-probes")
    args = p.parse_args(argv)

    t0 = time.time()
    from tac.score_aware_loop.targets import load_frozen_distortion_net
    from tac.torch_vehicle.vendored_imports import import_vendored

    data = import_vendored("data")
    video_path = data.get_default_video_path()
    net = load_frozen_distortion_net(device="cpu")
    dec, latents, meta = _load_basin_decoder()
    n = min(args.n_pairs, int(latents.shape[0]))
    print(f"[probe] loaded scorer+basin in {time.time()-t0:.1f}s; n_pairs={n} meta={meta}", flush=True)

    cam_float = _render_camera_pairs(dec, latents[:n], n)
    gt_u8 = _decode_gt_pairs(video_path, n)
    print(f"[probe] rendered+decoded {n} pairs in {time.time()-t0:.1f}s", flush=True)

    base = _measure_bias_affine(net, gt_u8, cam_float, None, None)
    result: dict[str, Any] = {
        "authority": _ADVISORY,
        "n_pairs": n,
        "basin_fork_point": str(_BASIN),
        "basin_meta": meta,
        "baseline_d_seg": base["d_seg"],
        "baseline_d_pose": base["d_pose"],
        "baseline_distortion_score": _seg_score(base["d_seg"], base["d_pose"]),
    }
    print(f"[probe] baseline d_seg={base['d_seg']:.6f} d_pose={base['d_pose']:.6f}", flush=True)

    out_path = args.out
    if out_path is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_path = Path(f".omx/research/track_a_distortion_finishing_kit_probe_{stamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _flush_partial() -> None:
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    _flush_partial()  # write baseline immediately so a crash still leaves data

    if "A" not in args.skip:
        ta = time.time()
        try:
            result["A_pr98_refit"] = refit_pr98_bias(net, gt_u8, cam_float, base_distortion=base)
            print(f"[probe] A PR98 re-fit done in {time.time()-ta:.1f}s "
                  f"delta={result['A_pr98_refit']['joint_delta_vs_base']:.6f}", flush=True)
        except Exception as exc:
            result["A_pr98_refit_error"] = repr(exc)
            print(f"[probe] A FAILED: {exc!r}", flush=True)
        _flush_partial()

    if "B" not in args.skip and "A_pr98_refit" in result:
        tb = time.time()
        try:
            pr98_bias = np.asarray(result["A_pr98_refit"]["best_bias_frame_channel"], dtype=np.float64)
            result["B_t10_affine"] = fit_t10_affine(net, gt_u8, cam_float, pr98_bias)
            print(f"[probe] B T10 affine done in {time.time()-tb:.1f}s "
                  f"delta_vs_pr98={result['B_t10_affine']['affine_delta_vs_pr98']:.6f}", flush=True)
        except Exception as exc:
            result["B_t10_affine_error"] = repr(exc)
            print(f"[probe] B FAILED: {exc!r}", flush=True)
        _flush_partial()

    if "C" not in args.skip:
        tc = time.time()
        try:
            result["C_s12"] = probe_s12(net, gt_u8, cam_float, n_proof=args.n_proof)
            print(f"[probe] C S12 done in {time.time()-tc:.1f}s "
                  f"invisible={result['C_s12']['invisible_fraction_of_camera_frame']:.4f} "
                  f"pct_reduction={result['C_s12']['per_frame_brotli_pct_reduction']:.2f}", flush=True)
        except Exception as exc:
            result["C_s12_error"] = repr(exc)
            print(f"[probe] C FAILED: {exc!r}", flush=True)
        _flush_partial()

    if "D" not in args.skip:
        td = time.time()
        try:
            result["D_lever_d"] = assess_lever_d(net, gt_u8, cam_float, tau=args.tau)
            print(f"[probe] D LeverD done in {time.time()-td:.1f}s "
                  f"verdict={result['D_lever_d']['verdict']}", flush=True)
        except Exception as exc:
            result["D_lever_d_error"] = repr(exc)
            print(f"[probe] D FAILED: {exc!r}", flush=True)
        _flush_partial()

    result["elapsed_seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[probe] wrote {out_path} ({result['elapsed_seconds']}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
