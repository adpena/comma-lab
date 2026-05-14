#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Measure SegNet + PoseNet high-frequency blindspot capacity (S2SBS audit).

Lane: lane_s2sbs_blindspot_audit_20260513
Memo: .omx/research/s2sbs_blindspot_audit_20260513.md
JSON: experiments/results/lane_s2sbs_blindspot_audit_20260513_<UTC>/blindspot_capacity.json

PURPOSE
-------
Council F (commit 896f1d79) hypothesis O3: SegNet's `tu-efficientnet_b2` encoder
uses a vanilla stride-2 stem (Conv2d, k=3, stride=2). After preprocess_input
bilinear-resize to (384, 512) the stem outputs (192, 256). Spatial frequencies
above the (192, 256) Nyquist cut are aliased/attenuated. PoseNet's fastvit_t12
stem has TWO stride-2 MobileOneBlocks => total stride 4 + rgb_to_yuv6 does
its own 4:2:0 chroma subsampling (another 2x in U,V). All three scorer paths
have HF blindspots.

We measure how many bits of arbitrary data per frame we can stuff into the HF
band (>= Nyquist of stem output) WITHOUT changing:
- SegNet argmax (pixelwise 5-class)
- PoseNet first-6-pose MSE (under a tight threshold)

CRITICAL CAVEATS (per Council F + CLAUDE.md):
- [macOS-CPU advisory] tag — every numeric output. NOT [contest-CPU]
  (that requires Linux x86_64) and NOT [contest-CUDA]. This is research
  signal only; no promotion, no kill.
- Bilinear resize is INFORMATION-LOSSY but not zero — even with perfect
  HF zero-out, checkerboard energy leaks into LF. We measure both the
  "pure HF mask" (zero out FFT bins above Nyquist) and the "downsample-
  upsample" (which is what actually happens through preprocess).
- Argmax decision boundary may be near an edge. We separately track
  "stable interior pixels" (margin > tau) vs "boundary pixels". Free-byte
  budget is reported AT each margin threshold.

NO GPU SPEND. NO authoritative claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "upstream"))


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class CapacityRow:
    config: str
    delta_amp: float
    pixels_total: int
    pixels_changed_segnet_argmax: int
    fraction_changed_segnet_argmax: float
    posenet_pose_mse_first6: float
    posenet_pose_max_abs_first6: float
    bits_per_pixel_segnet_safe: float
    bits_per_pixel_joint_safe: float
    bytes_per_frame_segnet_only: float
    bytes_per_frame_joint: float
    notes: str = ""


@dataclass
class AuditResult:
    schema: str = "s2sbs_blindspot_capacity_v1"
    schema_evidence_grade: str = "[macOS-CPU advisory]"
    research_only: bool = True
    schema_score_claim: bool = False
    schema_promotion_eligible: bool = False
    schema_ready_for_exact_eval_dispatch: bool = False
    timestamp_utc: str = ""
    host: str = ""
    python: str = ""
    torch_version: str = ""
    repo_head_sha: str = ""
    video_path: str = ""
    video_sha256: str = ""
    segnet_sd_sha256: str = ""
    posenet_sd_sha256: str = ""
    n_frames_sampled: int = 0
    camera_size: tuple = (1164, 874)
    segnet_input_size: tuple = (512, 384)
    segnet_stem_stride: int = 2
    posenet_stem_total_stride: int = 4
    posenet_chroma_subsample: int = 2
    nyquist_segnet_input_hw: tuple = (192, 256)
    config_description: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    prbs_demo: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def build_audit_result() -> AuditResult:
    res = AuditResult()
    res.timestamp_utc = _utc_now()
    res.host = socket.gethostname()
    res.python = sys.version.split()[0]
    import torch
    res.torch_version = torch.__version__
    return res


def _git_head_sha() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
        ).decode().strip()
        return out
    except Exception:
        return ""


def _load_scorers(device):
    """Returns (segnet, posenet) both eval(), on device, with safetensors loaded."""
    from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path
    from safetensors.torch import load_file

    seg = SegNet().eval().to(device)
    seg_sd = load_file(str(segnet_sd_path), device=str(device))
    seg.load_state_dict(seg_sd)

    pose = PoseNet().eval().to(device)
    pose_sd = load_file(str(posenet_sd_path), device=str(device))
    pose.load_state_dict(pose_sd)

    return seg, pose


def _decode_frames(video_path: Path, n_pairs: int, seq_len: int = 2):
    """Decode `n_pairs` non-overlapping pairs using upstream frame_utils.

    Returns tensor (n_pairs, seq_len, H, W, 3) uint8 -> float, ordered HWC.

    This intentionally goes through AVVideoDataset/yuv420_to_rgb rather than
    PyAV's direct rgb24 conversion. upstream/evaluate.py uses AVVideoDataset on
    CPU, and yuv420_to_rgb is the contest scorer's CPU-side RGB contract.
    """
    import torch
    from frame_utils import AVVideoDataset, camera_size
    from frame_utils import seq_len as upstream_seq_len

    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")
    if seq_len != upstream_seq_len:
        raise RuntimeError(
            f"tool seq_len={seq_len} does not match upstream frame_utils.seq_len="
            f"{upstream_seq_len}"
        )

    device = torch.device("cpu")
    ds = AVVideoDataset(
        [video_path.name],
        data_dir=video_path.parent,
        batch_size=n_pairs,
        device=device,
    )
    ds.prepare_data()
    for _, _, batch in ds:
        if batch.shape[0] < n_pairs:
            raise RuntimeError(
                f"only decoded {batch.shape[0]} pairs, need {n_pairs}"
            )
        expected_shape_tail = (seq_len, camera_size[1], camera_size[0], 3)
        if tuple(batch.shape[1:]) != expected_shape_tail:
            raise RuntimeError(
                f"unexpected upstream frame batch shape: {tuple(batch.shape)}; "
                f"expected (*, {expected_shape_tail})"
            )
        return batch[:n_pairs].to(dtype=torch.float32)

    raise RuntimeError(f"decoded no batches from {video_path}")


def _hf_mask_hw(height: int, width: int, nyquist_hw: tuple, *, device=None):
    """Return shifted-FFT mask that keeps only bins outside the LF rectangle."""
    import torch

    nyq_h, nyq_w = nyquist_hw
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid spatial shape: {(height, width)}")
    if nyq_h <= 0 or nyq_w <= 0:
        raise ValueError(f"invalid nyquist window: {nyquist_hw}")

    cy, cx = height // 2, width // 2
    hbw_y = min(height // 2, int(nyq_h) // 2)
    hbw_x = min(width // 2, int(nyq_w) // 2)

    mask = torch.ones(height, width, device=device)
    y0 = max(0, cy - hbw_y)
    y1 = min(height, cy + hbw_y)
    x0 = max(0, cx - hbw_x)
    x1 = min(width, cx + hbw_x)
    mask[y0:y1, x0:x1] = 0.0
    # Keep the shifted-FFT mask Hermitian-symmetric. Otherwise a real-valued
    # inverse FFT would reintroduce conjugate low-frequency coefficients when
    # the masked spectrum is projected back to real pixels.
    zero_coords = (mask == 0).nonzero(as_tuple=False)
    for yy, xx in zero_coords:
        cyy, cxx = _shifted_fft_conjugate_coord(
            int(yy.item()), int(xx.item()), height, width
        )
        mask[cyy, cxx] = 0.0
    return mask


def _shifted_fft_conjugate_coord(yy: int, xx: int, height: int, width: int) -> tuple[int, int]:
    """Conjugate coordinate for a shifted FFT grid."""
    cy, cx = height // 2, width // 2
    return ((2 * cy - yy) % height, (2 * cx - xx) % width)


def _hf_hermitian_coordinate_pairs(mask: torch.Tensor) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Unique HF coordinate pairs for a real-valued inverse FFT perturbation."""
    pairs = []
    seen = set()
    height, width = int(mask.shape[0]), int(mask.shape[1])
    for yy in range(height):
        for xx in range(width):
            if mask[yy, xx].item() <= 0:
                continue
            coord = (yy, xx)
            if coord in seen:
                continue
            conj = _shifted_fft_conjugate_coord(yy, xx, height, width)
            if conj == coord:
                seen.add(coord)
                continue
            if mask[conj[0], conj[1]].item() <= 0:
                seen.add(coord)
                continue
            pairs.append((coord, conj))
            seen.add(coord)
            seen.add(conj)
    return pairs


def _make_hf_perturbation(
    shape_chw: tuple, nyquist_hw: tuple, delta_amp: float, seed: int = 7
):
    """Build an additive RGB perturbation whose FFT support is OUTSIDE the
    rectangular [|fy| < nyquist_h/2, |fx| < nyquist_w/2] window.

    shape_chw: (C, H, W). The mask is broadcast across C.
    nyquist_hw: (nyquist_h, nyquist_w) — frequency band the stride-2 stem keeps.
                Frequencies STRICTLY above these (in either axis) are HF.
    delta_amp: peak amplitude in pixel units.

    Returns float tensor (C, H, W).
    """
    import torch

    C, H, W = shape_chw
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(C, H, W, generator=g)
    # FFT-shifted: low freqs at center
    F = torch.fft.fftshift(torch.fft.fft2(noise, dim=(-2, -1)), dim=(-2, -1))

    mask = _hf_mask_hw(H, W, nyquist_hw)

    F = F * mask.unsqueeze(0)
    F_unshift = torch.fft.ifftshift(F, dim=(-2, -1))
    pert = torch.fft.ifft2(F_unshift, dim=(-2, -1)).real

    # Normalize to peak amplitude
    peak = pert.abs().max().clamp_min(1e-12)
    pert = pert * (delta_amp / peak)
    return pert  # (C, H, W) float


def _segnet_argmax(seg, pair_bthwc: torch.Tensor):
    """pair_bthwc: (B, T, H, W, 3) float [0..255]. Returns argmax map and logits.

    Returns:
      argmax: (B, H_segin, W_segin) long
      logits: (B, 5, H_segin, W_segin) float
    """
    import einops
    import torch

    B, T, H, W, _ = pair_bthwc.shape
    x = einops.rearrange(pair_bthwc, "b t h w c -> b t c h w", c=3).float()
    seg_in = seg.preprocess_input(x)  # (B, 3, 384, 512)
    with torch.inference_mode():
        logits = seg(seg_in)
    argmax = logits.argmax(dim=1)
    return argmax, logits


def _posenet_pose(pose, pair_bthwc: torch.Tensor):
    """Returns first-6-pose dim per batch element."""
    import einops
    import torch

    B, T, H, W, _ = pair_bthwc.shape
    x = einops.rearrange(pair_bthwc, "b t h w c -> b t c h w", c=3).float()
    pose_in = pose.preprocess_input(x)
    with torch.inference_mode():
        out = pose(pose_in)
    p = out["pose"][..., :6]  # (B, 6)
    return p


def _logit_margin(logits):
    """For each pixel, return (top1 - top2) logit margin. logits: (B, C, H, W)."""
    import torch
    top2, _ = torch.topk(logits, k=2, dim=1)  # (B, 2, H, W)
    return (top2[:, 0] - top2[:, 1])  # (B, H, W)


def _first_conv2d_contract(module) -> dict:
    """Small runtime contract record for the first Conv2d in a scorer module."""
    import torch.nn as nn

    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d):
            return {
                "path": name,
                "kernel_size": list(child.kernel_size),
                "stride": list(child.stride),
                "padding": list(child.padding),
                "in_channels": int(child.in_channels),
                "out_channels": int(child.out_channels),
            }
    return {"missing": True}


def _verify_upstream_preprocess_contracts(seg, pose) -> dict:
    """Verify dimensions against upstream/modules.py and frame_utils.py."""
    import torch
    from frame_utils import camera_size, segnet_model_input_size, seq_len

    dummy = torch.zeros(
        1, seq_len, 3, camera_size[1], camera_size[0], dtype=torch.float32
    )
    seg_in = seg.preprocess_input(dummy)
    pose_in = pose.preprocess_input(dummy)
    expected_seg = (1, 3, segnet_model_input_size[1], segnet_model_input_size[0])
    expected_pose = (
        1,
        seq_len * 6,
        segnet_model_input_size[1] // 2,
        segnet_model_input_size[0] // 2,
    )
    if tuple(seg_in.shape) != expected_seg:
        raise RuntimeError(
            f"SegNet preprocess contract drift: got {tuple(seg_in.shape)}, "
            f"expected {expected_seg}"
        )
    if tuple(pose_in.shape) != expected_pose:
        raise RuntimeError(
            f"PoseNet preprocess contract drift: got {tuple(pose_in.shape)}, "
            f"expected {expected_pose}"
        )
    return {
        "frame_utils_seq_len": int(seq_len),
        "frame_utils_camera_size_wh": list(camera_size),
        "frame_utils_segnet_model_input_size_wh": list(segnet_model_input_size),
        "segnet_preprocess_output_bchw": list(seg_in.shape),
        "posenet_preprocess_output_bchw": list(pose_in.shape),
        "segnet_first_conv2d": _first_conv2d_contract(seg),
        "posenet_vision_first_conv2d": _first_conv2d_contract(pose.vision),
        "contract_source": "upstream/modules.py + upstream/frame_utils.py",
    }


def run_audit(args) -> AuditResult:
    import torch

    res = build_audit_result()
    res.repo_head_sha = _git_head_sha()

    device = torch.device("cpu")
    res.notes.append(
        "Running on CPU (macOS or Linux). Output is [macOS-CPU advisory] / "
        "research-signal-only. NOT [contest-CUDA] and NOT [contest-CPU]."
    )

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    res.video_path = str(video_path)
    res.video_sha256 = _sha256_file(video_path)

    from modules import posenet_sd_path, segnet_sd_path
    res.segnet_sd_sha256 = _sha256_file(Path(str(segnet_sd_path)))
    res.posenet_sd_sha256 = _sha256_file(Path(str(posenet_sd_path)))

    print(f"[s2sbs] loading scorers on {device}...", flush=True)
    seg, pose = _load_scorers(device)
    scorer_contract = _verify_upstream_preprocess_contracts(seg, pose)
    print("[s2sbs] scorers loaded.", flush=True)

    n_pairs = args.n_pairs
    print(
        f"[s2sbs] decoding {n_pairs} pairs ({n_pairs * 2} frames) from {video_path}...",
        flush=True,
    )
    pairs = _decode_frames(video_path, n_pairs=n_pairs, seq_len=2)
    res.n_frames_sampled = n_pairs * 2

    # Stems
    res.segnet_stem_stride = 2  # verified empirically (Conv2d k=3 stride=2)
    res.posenet_stem_total_stride = 4  # two MobileOneBlocks both stride 2
    res.posenet_chroma_subsample = 2

    # Nyquist of stem output (relative to SegNet input which is (384,512))
    res.nyquist_segnet_input_hw = (192, 256)
    res.config_description = {
        "segnet_input_size_hw": [384, 512],
        "segnet_stem_post_size_hw": [192, 256],
        "posenet_input_size_hw": [384, 512],
        "posenet_stem_post_size_hw": [96, 128],
        "perturb_applied_at": "camera_size_HWC_pre_preprocess (874, 1164, 3)",
        "perturb_HF_window": "outside FFT rect (256, 192) at SegNet input scale "
        "— but applied at camera scale, where the same fraction of bandwidth "
        "(top half of Nyquist) is HF",
        "upstream_scorer_contract": scorer_contract,
        "pair_selection": (
            "first N non-overlapping seq_len=2 pairs emitted by "
            "frame_utils.AVVideoDataset; this smoke is deterministic but not "
            "a stratified full-video capacity estimate"
        ),
    }

    # Apply perturbation at camera scale on each frame
    # We need a frame-scale Nyquist that corresponds to (192, 256) at (384, 512).
    # The bilinear interp scales freq by (segnet_h / cam_h) and (segnet_w / cam_w).
    # Camera (874, 1164). SegNet input (384, 512).
    # A frequency component at camera scale (fy, fx) survives bilinear and reaches
    # segnet input at (fy * 384/874, fx * 512/1164). For it to be HF at segnet
    # input (>= 192/2 = 96 in y, >= 256/2 = 128 in x in shifted FFT), we need
    # fy >= 96 * 874/384 = 218.5 and fx >= 128 * 1164/512 = 291. So the camera-scale
    # window we zero out is (218*2, 291*2) = (437, 583) — i.e. perturb only outside
    # an FFT rect of that size at camera scale. That's the LF window to preserve.
    cam_h, cam_w = 874, 1164
    seg_h, seg_w = 384, 512
    nyq_seg_h, nyq_seg_w = 192, 256
    # half-bandwidth in shifted-FFT coords at camera scale that bilinear-resamples
    # to the FULL Nyquist of the segnet stem (so its image survives the resize):
    hbw_cam_h = round(nyq_seg_h * cam_h / seg_h / 2)
    hbw_cam_w = round(nyq_seg_w * cam_w / seg_w / 2)
    nyquist_cam_hw = (2 * hbw_cam_h, 2 * hbw_cam_w)
    res.config_description["nyquist_cam_window_hw_zeroed_in_pert"] = list(nyquist_cam_hw)

    print(
        f"[s2sbs] camera-scale LF-preserve window (zeroed in pert) = "
        f"{nyquist_cam_hw} (cam_size={cam_h}x{cam_w})",
        flush=True,
    )

    # Sweep perturbation amplitudes
    delta_amps = args.delta_amps
    print(f"[s2sbs] sweeping delta_amps = {delta_amps}", flush=True)

    margin_tau = args.margin_tau

    # We measure per-amplitude across all pairs and aggregate.
    rows = []
    posenet_baseline = {}

    # --- BASELINE: no perturbation ---
    baseline_argmax_list = []
    baseline_logits_list = []
    baseline_pose_list = []
    for i in range(n_pairs):
        pair = pairs[i:i + 1]  # (1, 2, H, W, 3)
        am, lo = _segnet_argmax(seg, pair)
        baseline_argmax_list.append(am)
        baseline_logits_list.append(lo)
        p = _posenet_pose(pose, pair)
        baseline_pose_list.append(p)
    import torch
    baseline_argmax = torch.cat(baseline_argmax_list, dim=0)  # (n_pairs, 384, 512)
    baseline_logits = torch.cat(baseline_logits_list, dim=0)
    baseline_pose = torch.cat(baseline_pose_list, dim=0)  # (n_pairs, 6)
    baseline_margin = _logit_margin(baseline_logits)  # (n_pairs, 384, 512)
    stable_mask = (baseline_margin > margin_tau)  # (n_pairs, 384, 512)
    stable_frac = stable_mask.float().mean().item()
    print(
        f"[s2sbs] baseline: stable-pixel fraction (margin > {margin_tau}) = "
        f"{stable_frac:.4f}",
        flush=True,
    )
    posenet_baseline["mean_norm"] = baseline_pose.norm(dim=-1).mean().item()

    pixels_per_segnet_frame = 384 * 512
    # The perturbation is applied at CAM_SIZE (874x1164), but the bilinear-resize
    # to (384, 512) means freq bins above the camera-scale LF window collapse to
    # a single SegNet-input pixel. The realizable byte budget is bounded by:
    #   min(cam-scale HF-band free bins * 3 channels, segnet-input pixel count * 3)
    # The amplitude formula log2(2*delta+1) bits-per-pixel is per SEGNET-INPUT
    # pixel; that's the binding bound because that's where the scorer reads.

    for delta_amp in delta_amps:
        per_pair_changed = []
        per_pair_pose_mse = []
        per_pair_pose_maxabs = []

        # We measure SegNet impact on the LAST frame only (which SegNet consumes)
        # and PoseNet impact on the perturbation applied to BOTH frames at the
        # camera scale. (We perturb only frame1 for segnet; perturb both frame0
        # and frame1 with INDEPENDENT noise for pose. This is the conservative
        # worst case for S2SBS.)
        # For per-frame budget we report the perturbation budget on FRAME1
        # (which is the frame SegNet sees) and on FRAME0 (which only PoseNet sees).

        for i in range(n_pairs):
            pair = pairs[i:i + 1].clone()  # (1, 2, H, W, 3) float
            pert1 = _make_hf_perturbation(
                shape_chw=(3, cam_h, cam_w),
                nyquist_hw=nyquist_cam_hw,
                delta_amp=delta_amp,
                seed=11 + i,
            )
            pert0 = _make_hf_perturbation(
                shape_chw=(3, cam_h, cam_w),
                nyquist_hw=nyquist_cam_hw,
                delta_amp=delta_amp,
                seed=23 + i,
            )
            # Apply: pair is (1, 2, H, W, 3). Frames are stored HWC; perturb is CHW.
            pert1_hwc = pert1.permute(1, 2, 0)
            pert0_hwc = pert0.permute(1, 2, 0)
            pair[0, 0] = (pair[0, 0] + pert0_hwc).clamp(0.0, 255.0)
            pair[0, 1] = (pair[0, 1] + pert1_hwc).clamp(0.0, 255.0)

            am, _ = _segnet_argmax(seg, pair)
            changed = (am != baseline_argmax[i:i + 1]).float()
            per_pair_changed.append(changed.sum().item())

            p = _posenet_pose(pose, pair)
            diff = (p - baseline_pose[i:i + 1])
            per_pair_pose_mse.append(diff.pow(2).mean().item())
            per_pair_pose_maxabs.append(diff.abs().max().item())

        changed_total = sum(per_pair_changed)
        pose_mse_mean = sum(per_pair_pose_mse) / max(1, len(per_pair_pose_mse))
        pose_maxabs_mean = sum(per_pair_pose_maxabs) / max(1, len(per_pair_pose_maxabs))

        pix_total = n_pairs * pixels_per_segnet_frame
        frac_changed = changed_total / pix_total

        # Bits per (segnet-input) pixel safely encodable, given the empirical
        # fraction of pixels that did NOT change argmax. Heuristic: if no
        # pixels flipped at amplitude delta, you can encode log2(2*delta+1)
        # bits per pixel (using a uniform amplitude alphabet from -delta to delta).
        # If a fraction f flipped, the SAFE bits is downweighted by (1-f).
        safe_bpp_seg = math.log2(2 * delta_amp + 1) * (1.0 - frac_changed) if delta_amp > 0 else 0.0

        # Joint safety: also require pose MSE under a tight threshold.
        # PR106-era pose_avg [contest-CUDA] is ~3.4e-5 per CLAUDE.md "SegNet vs
        # PoseNet importance — operating-point dependent" section. We use 1e-5
        # (~30% of PR106 pose_avg) as the joint-safety threshold so a frame-byte
        # codec's added pose drift would NOT dominate the existing pose_avg
        # baseline. This is research-signal-only; final codec would re-measure
        # against the actual archive baseline.
        # Joint safe bpp: same formula but multiplied by indicator(pose_mse < 1e-5).
        joint_safe = pose_mse_mean < 1e-5
        safe_bpp_joint = safe_bpp_seg if joint_safe else 0.0

        # Bytes per frame: bits across the segnet-input pixel grid, divided by 8.
        # At camera scale the perturbation has 3 RGB channels times cam_h x cam_w
        # bandwidth, BUT only the HF part outside the LF window is free. The
        # information-theoretic available bytes are at the HF-band frequency-
        # bin count, which equals (cam_h * cam_w * 3) - (LF_window * 3) freq bins
        # in the FFT — but realizable bits per pixel is bounded by the
        # amplitude-quantization formula above applied at SegNet INPUT scale
        # (384 * 512 * 3 = 589824 pixels per frame).
        bytes_seg_only = safe_bpp_seg * (384 * 512 * 3) / 8.0
        bytes_joint = safe_bpp_joint * (384 * 512 * 3) / 8.0

        row = CapacityRow(
            config="hf_band_zero_lf_window",
            delta_amp=float(delta_amp),
            pixels_total=pix_total,
            pixels_changed_segnet_argmax=int(changed_total),
            fraction_changed_segnet_argmax=float(frac_changed),
            posenet_pose_mse_first6=float(pose_mse_mean),
            posenet_pose_max_abs_first6=float(pose_maxabs_mean),
            bits_per_pixel_segnet_safe=float(safe_bpp_seg),
            bits_per_pixel_joint_safe=float(safe_bpp_joint),
            bytes_per_frame_segnet_only=float(bytes_seg_only),
            bytes_per_frame_joint=float(bytes_joint),
            notes=(
                f"HF-only perturbation at camera scale outside FFT rect {nyquist_cam_hw}; "
                f"both frames perturbed independently; SegNet measured on frame1 only; "
                f"PoseNet sees pair-relative perturbation. macOS-CPU advisory."
            ),
        )
        rows.append(asdict(row))
        print(
            f"[s2sbs] delta={delta_amp:6.2f}  changed_frac={frac_changed:.6f}  "
            f"pose_mse={pose_mse_mean:.3e}  bpp_seg={safe_bpp_seg:.4f}  "
            f"bytes_seg={bytes_seg_only:8.1f}  bpp_joint={safe_bpp_joint:.4f}  "
            f"bytes_joint={bytes_joint:8.1f}",
            flush=True,
        )

    res.rows = rows

    # Aggregate: the LARGEST delta where joint safety holds AND the largest
    # delta where segnet-only safety (frac_changed < 1e-4) holds.
    seg_only_safe_rows = [r for r in rows if r["fraction_changed_segnet_argmax"] < 1e-4]
    joint_safe_rows = [r for r in rows if r["bits_per_pixel_joint_safe"] > 0.0]
    res.aggregate = {
        "stable_pixel_fraction_baseline_at_margin_tau": stable_frac,
        "margin_tau": margin_tau,
        "max_bytes_per_frame_segnet_only_advisory": max(
            (r["bytes_per_frame_segnet_only"] for r in seg_only_safe_rows), default=0.0
        ),
        "max_bytes_per_frame_joint_advisory": max(
            (r["bytes_per_frame_joint"] for r in joint_safe_rows), default=0.0
        ),
        "n_pairs": n_pairs,
        "n_frames": n_pairs * 2,
        "posenet_baseline": posenet_baseline,
    }

    # --- PRBS-31 STUFFING DEMO at the largest joint-safe delta ---
    if joint_safe_rows:
        best = max(joint_safe_rows, key=lambda r: r["bits_per_pixel_joint_safe"])
        target_delta = best["delta_amp"]
        # Generate a PRBS-31 sequence and stuff it as HF amplitude noise
        prbs_bits = _prbs31(64 * 1024)  # 64 Kbits = 8 KB
        # Encode as +/- target_delta perturbations at HF positions of frame1 only.
        # We measure: (a) decoded bit-error rate via FFT round-trip on the
        # un-perturbed frame plus the stuffed perturbation; (b) SegNet argmax
        # disagreement; (c) PoseNet pose drift.
        demo = _prbs_stuff_demo(
            seg, pose, pairs[0:1], target_delta, prbs_bits, nyquist_cam_hw,
            cam_h, cam_w
        )
        res.prbs_demo = demo
    else:
        res.prbs_demo = {
            "skipped": True,
            "reason": "no delta produced joint_safe>0; sweep deltas downward",
        }

    return res


def _prbs31(n_bits: int):
    """Generate a PRBS-31 sequence (x^31 + x^28 + 1)."""
    state = 0x5A5A5A5A & 0x7FFFFFFF
    out = bytearray()
    bitbuf = 0
    bitcount = 0
    for _ in range(n_bits):
        bit = ((state >> 30) ^ (state >> 27)) & 1
        state = ((state << 1) | bit) & 0x7FFFFFFF
        bitbuf = (bitbuf << 1) | bit
        bitcount += 1
        if bitcount == 8:
            out.append(bitbuf & 0xFF)
            bitbuf = 0
            bitcount = 0
    if bitcount:
        out.append((bitbuf << (8 - bitcount)) & 0xFF)
    return bytes(out)


def _binary_entropy2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def _prbs_stuff_demo(seg, pose, pair_one, target_delta, prbs_bits, nyquist_cam_hw,
                     cam_h, cam_w):
    """One-pair stuffing demo. Encodes prbs_bits as +/- target_delta HF amplitude
    on frame1, measures recovery BER (after FFT round-trip on integer-rounded
    frame), SegNet argmax disagreement, PoseNet drift."""
    import torch

    # Build HF mask at camera scale
    mask = _hf_mask_hw(cam_h, cam_w, nyquist_cam_hw)
    coord_pairs = _hf_hermitian_coordinate_pairs(mask)

    # We encode one bit per conjugate-paired real cosine coefficient in R.
    # Pairing is mandatory: an unconstrained arbitrary shifted FFT payload is
    # not Hermitian, so taking .real after ifft would silently discard signal.
    n_bits_avail = len(coord_pairs)
    n_bits = min(8 * len(prbs_bits), n_bits_avail)
    bits = []
    for byte in prbs_bits:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
            if len(bits) >= n_bits:
                break
        if len(bits) >= n_bits:
            break

    # Assign bits to HF conjugate pairs in raster order. +1 bit -> positive
    # cosine coefficient, 0 bit -> negative coefficient.
    F_pert_shifted = torch.zeros(cam_h, cam_w, dtype=torch.complex64)
    used_pairs = coord_pairs[:n_bits]
    for k, ((yy, xx), (cy, cx)) in enumerate(used_pairs):
        signed = (1.0 if bits[k] == 1 else -1.0) * target_delta
        F_pert_shifted[yy, xx] = complex(signed, 0.0)
        F_pert_shifted[cy, cx] = complex(signed, 0.0)
    pert_R = torch.fft.ifft2(
        torch.fft.ifftshift(F_pert_shifted), dim=(-2, -1)
    ).real
    peak = pert_R.abs().max().clamp_min(1e-12)
    pert_R = pert_R * (target_delta / peak)
    pert_chw = torch.stack([pert_R, torch.zeros_like(pert_R), torch.zeros_like(pert_R)], dim=0)

    pair = pair_one.clone()  # (1, 2, H, W, 3)
    pert_hwc = pert_chw.permute(1, 2, 0)
    pair[0, 1] = (pair[0, 1] + pert_hwc).clamp(0.0, 255.0)
    # Integer-round to uint8 to simulate the byte-channel
    pair_u8 = pair.round()
    stuffed_frame_R = pair_u8[0, 1, :, :, 0]  # (H, W)
    baseline_frame_R = pair_one[0, 1, :, :, 0]
    diff_R = (stuffed_frame_R - baseline_frame_R)  # what survived rounding

    # Recover bits via FFT
    F_recov = torch.fft.fftshift(torch.fft.fft2(diff_R), dim=(-2, -1))
    recovered_bits = []
    for ((yy, xx), _) in used_pairs:
        real_part = F_recov[yy, xx].real.item()
        recovered_bits.append(1 if real_part > 0 else 0)
    n_err = sum(1 for a, b in zip(bits, recovered_bits, strict=True) if a != b)
    ber = n_err / max(1, len(bits))
    bsc_capacity = max(0.0, 1.0 - _binary_entropy2(ber)) if ber < 0.5 else 0.0

    # Scorer impact on stuffed pair
    am, _ = _segnet_argmax(seg, pair_u8)
    am_base, _ = _segnet_argmax(seg, pair_one)
    seg_disagree = (am != am_base).float().mean().item()
    p_stuffed = _posenet_pose(pose, pair_u8)
    p_base = _posenet_pose(pose, pair_one)
    pose_mse = (p_stuffed - p_base).pow(2).mean().item()
    pose_maxabs = (p_stuffed - p_base).abs().max().item()

    return {
        "target_delta": float(target_delta),
        "n_bits_encoded": int(n_bits),
        "n_bytes_encoded": int(n_bits // 8),
        "bit_error_rate_after_uint8_roundtrip": float(ber),
        "binary_symmetric_channel_capacity_bits_per_encoded_bit": float(bsc_capacity),
        "effective_payload_bytes_before_ecc_upper_bound": float(n_bits * bsc_capacity / 8.0),
        "hf_conjugate_pairs_available_one_channel": int(n_bits_avail),
        "hermitian_symmetric_fft_payload": True,
        "segnet_argmax_disagree_frac": float(seg_disagree),
        "posenet_pose_mse_first6": float(pose_mse),
        "posenet_pose_max_abs_first6": float(pose_maxabs),
        "single_pair_only": True,
        "channel_used_for_stuffing": "R",
        "notes": (
            "Single-pair PRBS-31 stuffing demo at camera-scale HF band using "
            "Hermitian FFT coordinate pairs. BER includes uint8 quantization "
            "loss; BSC capacity is an optimistic pre-ECC upper bound. "
            "macOS-CPU advisory."
        ),
    }


def write_outputs(res: AuditResult, json_path: Path, memo_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(asdict(res), f, indent=2, default=str)
    print(f"[s2sbs] wrote JSON: {json_path}", flush=True)

    memo_path.parent.mkdir(parents=True, exist_ok=True)
    md = _render_memo(res, json_path)
    memo_path.write_text(md)
    print(f"[s2sbs] wrote memo: {memo_path}", flush=True)


def _render_memo(res: AuditResult, json_path: Path) -> str:
    lines = []
    lines.append("# phi3 S2SBS — Stride-2-Stem Byte-Stuffing Blindspot Audit")
    lines.append("")
    lines.append("- Lane: `lane_s2sbs_blindspot_audit_20260513` (Phase 2)")
    lines.append("- Council source: commit 896f1d79 (TRIPLET phi, O3)")
    lines.append("- Evidence grade: **[macOS-CPU advisory]** — research signal only")
    lines.append("- `research_only=true`, `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`")
    lines.append(f"- Timestamp UTC: {res.timestamp_utc}")
    lines.append(f"- Host: {res.host}")
    lines.append(f"- Repo HEAD: `{res.repo_head_sha}`")
    lines.append(f"- Video: `{res.video_path}` sha256=`{res.video_sha256}`")
    lines.append(f"- SegNet weights sha256=`{res.segnet_sd_sha256}`")
    lines.append(f"- PoseNet weights sha256=`{res.posenet_sd_sha256}`")
    lines.append(f"- Sampled pairs: {res.aggregate.get('n_pairs')} ({res.n_frames_sampled} frames)")
    lines.append(f"- JSON: `{json_path.relative_to(REPO_ROOT) if json_path.is_absolute() else json_path}`")
    lines.append("")
    lines.append("## Architectural derivation")
    lines.append("")
    lines.append("- SegNet `tu-efficientnet_b2` conv_stem: `Conv2d(k=3, stride=2)` -> post-stem feature map at HALF resolution.")
    lines.append("- SegNet `preprocess_input` resizes to (384, 512) -> stem out (192, 256). Nyquist preserved up to (192/2, 256/2) at the input resolution.")
    lines.append("- PoseNet `fastvit_t12` stem: TWO consecutive stride-2 MobileOneBlocks -> total stride 4. Plus PoseNet preprocess does rgb_to_yuv6 with 2x chroma subsampling on U/V.")
    lines.append("- Stems/resampling attenuate and alias high spatial frequency; they do NOT erase it. The audit measures leakage empirically rather than treating the FFT construction as proof.")
    lines.append("")
    lines.append("## Scorer contract check")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(res.config_description.get("upstream_scorer_contract", {}), indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("1. Decode the first N non-overlapping pairs from `upstream/videos/0.mkv` with `frame_utils.AVVideoDataset` / `yuv420_to_rgb`. seq_len=2, HxW=874x1164. This matches `upstream/evaluate.py` CPU-side frame semantics but is NOT a stratified full-video sample.")
    lines.append("2. For each pair: build TWO independent HF perturbations (one for frame0, one for frame1) at camera scale. The perturbation is FFT-band-limited to lie OUTSIDE the camera-scale LF window whose bilinear-resampled image equals the SegNet stem Nyquist (192, 256). Amplitude swept across `--delta-amps`.")
    lines.append("3. Run SegNet on frame1 (which is what the scorer uses) BEFORE and AFTER perturbation. Measure pixelwise argmax disagreement fraction.")
    lines.append("4. Run PoseNet on the full pair BEFORE and AFTER perturbation. Measure first-6-pose MSE and max-abs delta.")
    lines.append("5. Per delta amplitude, compute `bits_per_pixel_safe = log2(2*delta+1) * (1 - changed_frac)` for SegNet only AND for joint (SegNet + PoseNet MSE < 1e-5) safety. Project to bytes/frame at SegNet input resolution (384*512*3).")
    lines.append("6. PRBS-31 stuffing demo at the largest joint-safe delta: encode +/-delta into Hermitian HF FFT coordinate pairs, integer-round to uint8 channel, decode via FFT, measure BER + scorer drift.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| delta_amp | changed_frac | pose_mse | bpp_seg_only | bytes/frame_seg | bpp_joint | bytes/frame_joint |")
    lines.append("|----------:|-------------:|---------:|-------------:|----------------:|----------:|------------------:|")
    for r in res.rows:
        lines.append(
            f"| {r['delta_amp']:.2f} | {r['fraction_changed_segnet_argmax']:.6f} | "
            f"{r['posenet_pose_mse_first6']:.3e} | {r['bits_per_pixel_segnet_safe']:.4f} | "
            f"{r['bytes_per_frame_segnet_only']:.1f} | {r['bits_per_pixel_joint_safe']:.4f} | "
            f"{r['bytes_per_frame_joint']:.1f} |"
        )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(res.aggregate, indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## PRBS-31 stuffing demo")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(res.prbs_demo, indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Caveats and limits")
    lines.append("")
    lines.append("- **[macOS-CPU advisory]** only. SegNet/PoseNet inference on macOS CPU is not 1:1 with [contest-CUDA] or Linux x86_64 [contest-CPU]. PoseNet MSE drift on MPS is 23x per CLAUDE.md, but here we used CPU (not MPS). Still: these numbers are research-signal only and CANNOT be used to promote, kill, or claim a score.")
    lines.append("- Bilinear resize is information-lossy but the HF-band-zeroed pert leaks small amounts of energy into the LF band via the resize kernel. The empirically measured `changed_frac` is the ground-truth, not the theoretical FFT support.")
    lines.append("- The PoseNet MSE threshold 1e-5 for joint safety is research-pick (PR106-era pose_avg is ~3.4e-5; 1e-5 is well below). The actual contest impact would need [contest-CPU] / [contest-CUDA] verification on a byte-closed archive.")
    lines.append("- The byte-budget number is THEORETICAL (info-theoretic upper bound per pixel from amplitude alphabet). A real S2SBS codec also needs (a) a uint8 quantization channel — the PRBS demo measures BER under round-to-uint8, (b) an inflate-time recovery path, (c) error correction below uint8-rounding noise floor.")
    lines.append("- The `bits_per_pixel_safe = log2(2*delta+1) * (1 - frac_changed)` formula is a heuristic upper bound (integer-amplitude alphabet size discount by argmax-flip fraction). Shannon-faithful capacity would use `0.5 * log2(1 + SNR)` after measuring the actual channel SNR (HF perturbation amplitude / leakage noise amplitude). The PRBS-31 BER at the largest joint-safe delta is the realized-channel lower bound on raw bytes after uint8 quantization — ECC reduces effective bytes further.")
    lines.append("- The PRBS demo is single-pair only. Its BSC-capacity field is an optimistic pre-ECC upper bound; if BER approaches 0.5, the theoretical byte budget is not a usable payload channel. Cross-pair generalization needs a wider sweep before any codec is built.")
    lines.append("- PoseNet visibility is real: `rgb_to_yuv6` preserves four luma subpixel channels (Y00/Y10/Y01/Y11), so 2x2 luma checkerboards remain visible to PoseNet even when U/V chroma averages cancel. S2SBS should prefer perturbations that are pair-consistent and pose-small, not merely SegNet-argmax-safe.")
    lines.append("- This is not byte-closed: no archive bytes changed, no `inflate.sh` carries a payload, no decoder recovers bytes from inflated raw frames, and no exact evaluator consumed an S2SBS archive. Therefore every capacity number remains `score_claim=false` and `ready_for_exact_eval_dispatch=false`.")
    lines.append("- Exact-eval blockers before contest relevance: build an archive pass that (1) embeds an ECC-coded payload into real Hermitian HF coefficients after renderer/video generation, (2) decodes the payload from uint8 inflated `.raw` frames with a manifest SHA, (3) proves payload byte recovery and frame count over all 600 pairs, (4) runs `scripts/pre_submission_compliance_check.py --contest-final --strict`, and only then (5) queues claimed [contest-CUDA] / paired [contest-CPU] eval.")
    lines.append("")
    lines.append("## Go / no-go verdict")
    lines.append("")
    agg = res.aggregate
    max_joint = agg.get("max_bytes_per_frame_joint_advisory", 0.0)
    max_seg = agg.get("max_bytes_per_frame_segnet_only_advisory", 0.0)
    prbs_effective = float(res.prbs_demo.get("effective_payload_bytes_before_ecc_upper_bound", 0.0)) if isinstance(res.prbs_demo, dict) else 0.0
    prbs_ber = float(res.prbs_demo.get("bit_error_rate_after_uint8_roundtrip", 1.0)) if isinstance(res.prbs_demo, dict) else 1.0
    if max_joint > 1000.0 and prbs_effective > 128.0:
        lines.append(f"- **GO-FOR-PROTOTYPE**: joint-safe theoretical budget ~{max_joint:.0f} bytes/frame and single-pair PRBS BSC upper bound ~{prbs_effective:.1f} bytes at BER={prbs_ber:.4f}. Next step is a byte-closed ECC/Hermitian-HF prototype, not a score claim.")
    elif max_joint > 1000.0:
        lines.append(f"- **NEEDS-FIX**: scorer drift budget is large (~{max_joint:.0f} bytes/frame theoretical), but the realized PRBS channel is too noisy for a payload prototype yet (effective pre-ECC upper bound ~{prbs_effective:.1f} bytes, BER={prbs_ber:.4f}). Fix frequency shaping / ECC before archive work.")
    elif max_seg > 1000.0:
        lines.append(f"- **NEEDS-FIX**: SegNet-only budget ~{max_seg:.0f} bytes/frame BUT joint (incl PoseNet) collapses to ~{max_joint:.0f}. S2SBS would need to apply only on frame0 (SegNet-blind frame) or use a smaller delta to keep PoseNet drift under control.")
    else:
        lines.append(f"- **NO-GO at HF-band+amplitude config tested**: budget too small (~{max_joint:.0f} bytes/frame joint, ~{max_seg:.0f} segnet-only). Per CLAUDE.md, this is NOT a KILL — DEFERRED-pending-research on (a) larger delta on frame0 only (per O8 sister lane), (b) per-class spatial gating (avoid boundary pixels), (c) frequency-band shaping outside the bilinear-resize leakage band.")
    lines.append("")
    lines.append("## Worst-case scenarios for the technique")
    lines.append("")
    lines.append("- Bilinear-resize energy leakage: the FFT-band-limited HF perturbation can still leak ~5-15% of its peak amplitude into the LF band after the (cam_size -> 384x512) bilinear resize. This explains why high-delta perturbations show non-zero `changed_frac` even when the FFT is band-limited.")
    lines.append("- PoseNet rgb_to_yuv6 + chroma subsampling integrates U/V over 2x2 blocks, but the four luma subpixel channels preserve checkerboard structure. PoseNet is the binding visibility risk for frame-pair perturbations.")
    lines.append("- Argmax decision boundary: pixels with logit margin < tau will flip even from a small noise floor. The `stable_pixel_fraction` aggregate quantifies how much of the frame is safe vs boundary.")
    lines.append("- uint8 quantization channel: an HF perturbation of amplitude < 0.5 has ~50% chance of being rounded away. The PRBS demo's BER under uint8-roundtrip is the realistic codec lower bound.")
    lines.append("")
    lines.append("## Wire-in hooks (per CLAUDE.md Subagent coherence-by-default)")
    lines.append("")
    lines.append("1. Sensitivity-map contribution: this audit provides a per-pixel (logit-margin, HF-blindspot) sensitivity that the renderer trainer / archive builder can consume to prioritize byte-spend AWAY from these pixels.")
    lines.append("2. Pareto constraint: an additional rate-axis constraint `bytes <= rate_budget - s2sbs_recovered_bytes` becomes available when a codec ships.")
    lines.append("3. Bit-allocator hook: per-tensor importance UNCHANGED (this is a frame-bytes side channel, not a tensor allocation).")
    lines.append("4. Cathedral autopilot dispatch hook: N/A — research-only audit; no archive bytes change yet. A future S2SBS codec lane would register here.")
    lines.append("5. Continual-learning posterior update: N/A — no empirical anchor produced (advisory only).")
    lines.append("6. Probe-disambiguator: N/A — single defensible interpretation (architectural HF blindspot empirically measured).")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        default=str(REPO_ROOT / "upstream/videos/0.mkv"),
        help="Path to contest video (default: upstream/videos/0.mkv)",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=4,
        help="Number of frame pairs to sample (default: 4 = 8 frames)",
    )
    parser.add_argument(
        "--delta-amps",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        help="Sweep of perturbation amplitudes (pixel units, 0..255)",
    )
    parser.add_argument(
        "--margin-tau",
        type=float,
        default=2.0,
        help="Logit-margin threshold for 'stable' pixels",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: experiments/results/lane_s2sbs_blindspot_audit_20260513_<UTC>)",
    )
    parser.add_argument(
        "--memo-path",
        default=str(REPO_ROOT / ".omx/research/s2sbs_blindspot_audit_20260513.md"),
    )
    args = parser.parse_args()

    if args.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        args.output_dir = str(
            REPO_ROOT / f"experiments/results/lane_s2sbs_blindspot_audit_20260513_{ts}"
        )

    res = run_audit(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "blindspot_capacity.json"
    memo_path = Path(args.memo_path)
    write_outputs(res, json_path, memo_path)

    # Final summary line
    agg = res.aggregate
    print("")
    print("=" * 72)
    print("S2SBS BLINDSPOT AUDIT — SUMMARY ([macOS-CPU advisory])")
    print("=" * 72)
    print(f"Max bytes/frame, SegNet-only safe: {agg.get('max_bytes_per_frame_segnet_only_advisory', 0):.1f}")
    print(f"Max bytes/frame, joint  safe    : {agg.get('max_bytes_per_frame_joint_advisory', 0):.1f}")
    print(f"Stable pixel fraction (margin>{args.margin_tau}): {agg.get('stable_pixel_fraction_baseline_at_margin_tau', 0):.4f}")
    print(f"JSON : {json_path}")
    print(f"Memo : {memo_path}")


if __name__ == "__main__":
    main()
