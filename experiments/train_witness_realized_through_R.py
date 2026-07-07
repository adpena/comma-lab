# SPDX-License-Identifier: MIT
"""CAPSTONE LEVER #2 — the never-done WITNESS trained THROUGH the round-trip R with
BOTH frozen scorers IN-LOOP, optimizing the REALIZED (post-R, post-scorer) d_seg AND
d_pose JOINTLY.

THE RECKONING (FEED-ah, 2026-06-25): every prior witness/symbolic d_seg number this
session was DIRECT-PARTITION (argmax-equality between a generator's argmax and the GT
SegNet argmax). FEED-ah MEASURED that this is a ~170-350x MIRAGE vs the axis the contest
actually scores: paint the EXACT partition L* as palette RGB, run the REAL frozen SegNet
THROUGH the round-trip R, and the realized d_seg is 0.005-0.008 (seg_term 0.64 = 3.4x the
whole pointer). The task-space SPLIT (partition store + separate pose carrier) is TRIPLY
DOMINATED at the byte-closed exact level (S~1.1) because SegNet AND PoseNet read the SAME
RGB and the split pays for appearance twice (palette for seg, carrier for pose) and loses
both. The frontier HNeRV wins precisely by JOINT coding: one full-RGB code gets realized
d_seg 5.6e-4 + d_pose 1.6e-5 from ONE shared appearance code at 177KB.

THE LEVER (never implemented): train ONE RGB-output witness so that, THROUGH R, the
RE-DERIVED SegNet argmax matches the target partition AND the PoseNet pose matches the GT
pose, JOINTLY. The round-trip R and the frozen scorers are IN THE TRAINING LOSS (not the
direct argmax). This is the only arm consistent with the frontier physics. The synergy:
the boundary MOTION is the pose -- one RGB code serves SegNet (boundary) + PoseNet
(parallax) together.

THE PHYSICS (the realized axis):
  witness(coords, per-pair-per-frame code) -> RGB pair (frame0 pose-carrying, frame1
  SegNet-read) -> R [bicubic up 384->874-ish camera -> uint8-STE -> bilinear down to
  384x512] -> frozen SegNet argmax (d_seg) + frozen PoseNet YUV6 (d_pose). Backprop the
  REALIZED loss into the witness.

COMPUTE-SUBSTRATE LAW (CLAUDE.md): MPS is a VALID fp32 TRAINING-GRADIENT device (the
104x scorer speedup), NEVER a score authority. The d_seg/d_pose VERDICT comes from the
FROZEN CPU-torch SegNet argmax + PoseNet (NEVER MPS). EMA + eval_roundtrip non-negotiable.

NO-FAKE: the realized d_seg = argmax-disagreement RATE between the FROZEN CPU SegNet on
the rendered-through-R frame1 and the GT SegNet argmax (L*). The realized d_pose =
pose-MSE between the FROZEN CPU PoseNet on the rendered-through-R pair and the GT pose.
A witness that learns to render RGB whose SegNet-re-derived partition does NOT survive R
will show a HIGH realized d_seg -- the loss is on the realized axis, the verdict is the
exact frozen-scorer quantity. No surrogate is the verdict.

BORROWED-SUBSTRATE (CLAUDE.md NO-FAKE #7):
  * BORROWED: the differentiable eval-roundtrip + autograd rgb_to_yuv6
    (tac.differentiable_eval_roundtrip, replicating PR#95 findings A+B); the frozen
    contest SegNet+PoseNet (upstream); the GT-decode + authority chain
    (tac.boundary_math.seg_core + the feedy probe's CPU verdict path); Fourier-feature
    coord-INR (Tancik 2020) + FiLM-per-instance modulation (Perez 2018) + HNeRV
    amortized-decoder framing.
  * OURS-ORIGINAL: training a NON-RGB-vehicle coord-INR to OUTPUT RGB whose REALIZED
    (post-R, post-frozen-SegNet-re-derive) argmax partition survives R AND whose
    PoseNet-re-derived pose matches GT, JOINTLY in one shared code -- the round-trip-
    survival lever R_surv with BOTH scorers in-loop, the joint realized-d_seg+d_pose
    loss, the synergy of one code for both scorers. This vehicle is our coord-INR, NOT
    PR95's HNeRV imported wholesale (anti-reskin).

Evidence: training gradient [macOS-MPS training-gradient]; the d_seg/d_pose VERDICT is
[contest-CPU advisory] (frozen CPU-torch direct-scorer mirror of evaluate.py over the
measured pair subset). promotion_eligible=False; NO score claim beyond advisory; pointer
UNMOVED unless a real byte-closed sub-0.19110 exact-eval row lands. Disk: SSD tier / repo
results, NEVER /tmp durable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Camera-native + scorer-input resolutions (from upstream/frame_utils.py).
CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512
RATE_DENOM = 37_545_489.0
FRONTIER = 0.19110
SUB015 = 0.15
N_CLASSES = 5

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
_FOURIER_SEED = 0


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field_name: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field_name}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# The witness: coord-INR (Fourier features + FiLM-per-(pair,frame)) -> RGB.
# RENDERS at a chosen render resolution (a small INR-friendly grid), then R
# upsamples to camera-res. Per (pair, frame) FiLM code carries the appearance +
# parallax. This is OUR coord-INR vehicle (anti-reskin), now OUTPUTTING RGB.
# ---------------------------------------------------------------------------
def deterministic_fourier_B(n_fourier: int, fourier_sigma: float) -> np.ndarray:
    rng = np.random.default_rng(_FOURIER_SEED)
    return (rng.standard_normal((2, n_fourier)) * fourier_sigma).astype(np.float32)


class RGBWitness(nn.Module):
    """g(coords, code[pair, frame]) -> RGB in [0,255] at the render grid.

    Input feature per pixel = [sin/cos(coords @ B_iso)]  (2*n_fourier).
    Base MLP (shared) + per-(pair,frame) FiLM modulation on each hidden layer.
    Output head -> 3 channels -> 255*sigmoid -> RGB in [0,255].

    The number of conditioning codes is num_pairs * 2 (one per frame of each pair),
    so frame0 and frame1 of a pair have DIFFERENT codes -> the witness can render
    real inter-frame motion (parallax) that PoseNet reads, while sharing the base
    appearance MLP (the joint code). This is the synergy: one MLP, per-frame codes.
    """

    def __init__(
        self,
        num_pairs: int,
        n_fourier: int = 24,
        hidden_dim: int = 128,
        n_hidden: int = 4,
        mod_dim: int = 48,
        fourier_sigma: float = 8.0,
    ) -> None:
        super().__init__()
        self.num_pairs = num_pairs
        self.n_fourier = n_fourier
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.mod_dim = mod_dim
        B = torch.from_numpy(deterministic_fourier_B(n_fourier, fourier_sigma))
        self.register_buffer("_fourier_B", B)
        in_feat = 2 * n_fourier
        self.in_feat = in_feat
        # Two codes per pair (frame0, frame1) -> parallax.
        self.code = nn.Parameter(torch.zeros(num_pairs * 2, mod_dim))
        self.in_proj = nn.Linear(in_feat, hidden_dim)
        self.film = nn.Linear(mod_dim, 2 * hidden_dim * n_hidden)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)])
        self.out = nn.Linear(hidden_dim, 3)

    def build_feats(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (P, 2) in [-1, 1] -> (P, 2*n_fourier) sin/cos Fourier features."""
        proj = coords @ self._fourier_B  # (P, n_fourier)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, coord_feats: torch.Tensor, code_idx: int) -> torch.Tensor:
        """coord_feats: (P, in_feat); returns (P, 3) RGB in [0, 255]."""
        h = F.relu(self.in_proj(coord_feats))
        m = self.code[code_idx]
        film = self.film(m).reshape(self.n_hidden, 2, self.hidden_dim)
        for li, layer in enumerate(self.hidden):
            scale = 1.0 + film[li, 0]
            shift = film[li, 1]
            h = F.relu(layer(h) * scale + shift)
        rgb = torch.sigmoid(self.out(h)) * 255.0
        return rgb


# ---------------------------------------------------------------------------
# EMA (CLAUDE.md non-negotiable). Decay 0.997 (Quantizr). Shadow used at eval.
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.997) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.is_floating_point()}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        orig = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        sd = model.state_dict()
        for k in self.shadow:
            sd[k].copy_(self.shadow[k])
        return orig

    def restore(self, model: nn.Module, orig: dict[str, torch.Tensor]) -> None:
        sd = model.state_dict()
        for k, v in orig.items():
            sd[k].copy_(v)


# ---------------------------------------------------------------------------
# The round-trip R (differentiable, in the training loss). render-grid RGB ->
# bicubic up to camera -> uint8-STE -> the scorer preprocess does its own
# bilinear down. We apply R = bicubic-up-to-camera + uint8-STE here, then feed
# the camera-res frame to the (differentiable) scorer preprocess which bilinears
# to 384x512. This matches evaluate.py: inflate emits camera-res uint8 frames;
# the scorer.preprocess_input bilinears them to 384x512.
# ---------------------------------------------------------------------------
def render_through_R(
    witness: RGBWitness,
    coord_feats: torch.Tensor,
    code_idx: int,
    render_h: int,
    render_w: int,
    *,
    ste_round: bool = True,
) -> torch.Tensor:
    """Witness -> (render_h, render_w, 3) RGB -> bicubic up to camera -> uint8-STE.

    Returns a (1, 3, CAMERA_H, CAMERA_W) float tensor in [0, 255] (the camera-res
    uint8-rounded frame the scorer will consume), gradients preserved.
    """
    rgb_flat = witness(coord_feats, code_idx)  # (render_h*render_w, 3)
    rgb = rgb_flat.reshape(render_h, render_w, 3).permute(2, 0, 1).unsqueeze(0)  # (1,3,h,w)
    up = F.interpolate(rgb, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
    up = up.clamp(0.0, 255.0)
    if ste_round:
        # Uint8 STE: forward rounds, backward identity.
        rounded = torch.round(up)
        up = up + (rounded - up).detach()
    return up


# ---------------------------------------------------------------------------
# Frozen-scorer realized losses (differentiable, training-gradient device).
# ---------------------------------------------------------------------------
def realized_seg_loss(
    segnet,
    frame1_cam: torch.Tensor,   # (1, 3, CAMERA_H, CAMERA_W) float [0,255], grad
    gt_argmax: torch.Tensor,    # (SEG_H, SEG_W) long  -- the GT SegNet argmax (L*)
    gt_margin: torch.Tensor | None = None,  # (SEG_H, SEG_W) float -- shallow-margin weight
    hinge_weight: float = 4.0,
) -> torch.Tensor:
    """Realized seg loss = margin-weighted CE(SegNet(rendered frame1) logits, L*).

    The SegNet sees the RENDERED-THROUGH-R RGB, so the gradient teaches the witness
    to render RGB whose SegNet-RE-DERIVED argmax matches L*. This is the realized
    axis (not direct argmax-equality).
    """
    # SegNet.preprocess_input expects (B, T, C, H, W) and takes x[:, -1]; build a
    # (1, 1, 3, H, W) so the last (only) frame IS frame1.
    x = frame1_cam.unsqueeze(1)  # (1, 1, 3, CAMERA_H, CAMERA_W)
    seg_in = segnet.preprocess_input(x)  # (1, 3, SEG_H, SEG_W) bilinear
    logits = segnet(seg_in)  # (1, 5, SEG_H, SEG_W)
    ce = F.cross_entropy(logits, gt_argmax.unsqueeze(0), reduction="none")  # (1, H, W)
    if gt_margin is not None:
        w = 1.0 + hinge_weight * torch.exp(-gt_margin.clamp(min=0.0)).unsqueeze(0)
        return (ce * w).mean()
    return ce.mean()


def realized_pose_loss(
    posenet,
    f0_cam: torch.Tensor,   # (1, 3, CAMERA_H, CAMERA_W) float [0,255], grad
    f1_cam: torch.Tensor,   # (1, 3, CAMERA_H, CAMERA_W) float [0,255], grad
    gt_pose: torch.Tensor,  # (half,) the GT pose (first 3 dims), no grad
) -> torch.Tensor:
    """Realized pose loss = MSE(PoseNet(rendered pair) pose, GT pose), first half dims.

    PoseNet reads BOTH rendered frames as YUV6 (rgb_to_yuv6 must be differentiable,
    monkey-patched at the call site). The gradient teaches the witness to render a
    pair whose PoseNet-re-derived pose matches GT -> realized d_pose.
    """
    pair = torch.cat([f0_cam.unsqueeze(1), f1_cam.unsqueeze(1)], dim=1)  # (1,2,3,H,W)
    pose_in = posenet.preprocess_input(pair)  # (1, 12, SEG_H, SEG_W) YUV6
    out = posenet(pose_in)
    pose = out["pose"] if isinstance(out, dict) else out  # (1, 12)
    half = None
    for hh in posenet.hydra.heads:
        if hh.name == "pose":
            half = hh.out // 2
            break
    if half is None:
        half = pose.shape[-1] // 2
    return F.mse_loss(pose[0, :half], gt_pose)


# ---------------------------------------------------------------------------
# Authority verdict (FROZEN CPU-torch, NEVER MPS) -- mirrors the feedy probe
# + upstream/modules.py exactly. This is the d_seg/d_pose VERDICT.
# ---------------------------------------------------------------------------
def cpu_verdict_d_seg(segnet_cpu, frame1_cam_uint8: np.ndarray, gt_argmax_np: np.ndarray) -> float:
    """Realized d_seg = argmax-disagreement of frozen CPU SegNet on the rendered
    frame1 vs L* (GT argmax). Exact frozen-scorer quantity."""
    r = np.asarray(frame1_cam_uint8)
    pair = torch.from_numpy(np.stack([r, r], axis=0)[None]).float()  # (1,2,H,W,3)
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()  # (1,2,3,H,W)
    with torch.inference_mode():
        seg_in = segnet_cpu.preprocess_input(xp)
        logits = segnet_cpu(seg_in)
        realized = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
    return float(np.count_nonzero(realized != gt_argmax_np)) / gt_argmax_np.size


def cpu_verdict_d_pose(posenet_cpu, f0_uint8: np.ndarray, f1_uint8: np.ndarray, gt_pose_np: np.ndarray) -> float:
    """Realized d_pose = pose-MSE of frozen CPU PoseNet on the rendered pair vs GT pose."""
    import einops

    pair = torch.from_numpy(np.stack([f0_uint8, f1_uint8], axis=0)[None]).float()  # (1,2,H,W,3)
    x = einops.rearrange(pair, "b t h w c -> b t c h w").float()
    with torch.inference_mode():
        pose_in = posenet_cpu.preprocess_input(x)
        out = posenet_cpu(pose_in)
        pose = out["pose"] if isinstance(out, dict) else out
        half = None
        for hh in posenet_cpu.hydra.heads:
            if hh.name == "pose":
                half = hh.out // 2
                break
        if half is None:
            half = pose.shape[-1] // 2
        p = pose[0, :half].cpu().numpy().astype(np.float64)
    return float(np.mean((p - gt_pose_np) ** 2))


# ---------------------------------------------------------------------------
# GT precompute: decode pairs, build L* + margin + GT pose on frozen CPU scorers.
# ---------------------------------------------------------------------------
@dataclass
class GTData:
    n_pairs: int
    gt_f0: list[np.ndarray]
    gt_f1: list[np.ndarray]
    lstars: list[np.ndarray]     # (SEG_H, SEG_W) int64 -- GT SegNet argmax of frame1
    margins: list[np.ndarray]    # (SEG_H, SEG_W) float -- shallow-margin
    gt_poses: list[np.ndarray]   # (half,) float64


def precompute_gt(n_pairs: int) -> tuple[GTData, Any, Any]:
    """Decode GT pairs + build L*/margin/pose on the frozen CPU scorers (authority)."""
    from tac.boundary_math.seg_core import (
        decode_gt_frame1_pairs,
        load_real_segnet,
        segnet_argmax_and_margin,
    )

    seg_cpu = load_real_segnet("cpu")
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # upstream

    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet_cpu = dn.posenet
    for p in posenet_cpu.parameters():
        p.requires_grad = False

    gt_f0, gt_f1, lstars, margins, gt_poses = [], [], [], [], []
    for _idx, f0, f1 in decode_gt_frame1_pairs(n_pairs=n_pairs):
        f0 = np.asarray(f0)
        f1 = np.asarray(f1)
        lstar, margin = segnet_argmax_and_margin(seg_cpu, f1)
        gt_f0.append(f0)
        gt_f1.append(f1)
        lstars.append(np.asarray(lstar).astype(np.int64))
        margins.append(np.asarray(margin).astype(np.float32))
        gt_poses.append(cpu_verdict_pose_raw(posenet_cpu, f0, f1))
    gt = GTData(
        n_pairs=len(lstars), gt_f0=gt_f0, gt_f1=gt_f1,
        lstars=lstars, margins=margins, gt_poses=gt_poses,
    )
    return gt, seg_cpu, posenet_cpu


def cpu_verdict_pose_raw(posenet_cpu, f0_uint8: np.ndarray, f1_uint8: np.ndarray) -> np.ndarray:
    """The raw GT pose (first half dims) from frozen CPU PoseNet."""
    import einops

    pair = torch.from_numpy(np.stack([f0_uint8, f1_uint8], axis=0)[None]).float()
    x = einops.rearrange(pair, "b t h w c -> b t c h w").float()
    with torch.inference_mode():
        pose_in = posenet_cpu.preprocess_input(x)
        out = posenet_cpu(pose_in)
        pose = out["pose"] if isinstance(out, dict) else out
        half = None
        for hh in posenet_cpu.hydra.heads:
            if hh.name == "pose":
                half = hh.out // 2
                break
        if half is None:
            half = pose.shape[-1] // 2
        return pose[0, :half].cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Train + eval driver.
# ---------------------------------------------------------------------------
def _build_render_coords(h: int, w: int) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def _render_full_frame_for_verdict(
    witness: RGBWitness, coord_feats_dev: torch.Tensor, code_idx: int,
    render_h: int, render_w: int,
) -> np.ndarray:
    """Render -> R -> camera-res uint8 numpy frame (HWC) for the CPU verdict.

    This is the SAME R the loss uses (bicubic up + round), then to uint8 numpy.
    """
    with torch.no_grad():
        cam = render_through_R(witness, coord_feats_dev, code_idx, render_h, render_w, ste_round=True)
    arr = cam[0].permute(1, 2, 0).clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    return arr


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    _refuse_tmp(out_dir, "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_device = args.train_device
    if train_device == "mps":
        from tac.torch_mps_compat import patch_scorer_for_mps
        patch_scorer_for_mps()
    if train_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--train-device cuda but CUDA not available")
    if train_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--train-device mps but MPS not available")

    # --- 1. GT precompute on frozen CPU scorers (authority) ---
    t0 = time.time()
    gt, seg_cpu, posenet_cpu = precompute_gt(args.num_pairs)
    P = gt.n_pairs
    gt_secs = time.time() - t0
    print(json.dumps({"stage": "gt_precompute", "n_pairs": P, "secs": round(gt_secs, 1)}), flush=True)

    # --- 2. Differentiable scorers for the training-gradient device ---
    # Monkey-patch rgb_to_yuv6 globally (PR#95 finding B) so PoseNet gradient flows.
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally, unpatch_upstream_yuv6
    yuv_token = patch_upstream_yuv6_globally()

    # Build a SECOND set of scorers on the train device (the CPU ones stay frozen authority).
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # upstream
    dn_dev = DistortionNet().eval()
    dn_dev.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    seg_dev = dn_dev.segnet.to(train_device)
    pose_dev = dn_dev.posenet.to(train_device)
    for p in seg_dev.parameters():
        p.requires_grad = False
    for p in pose_dev.parameters():
        p.requires_grad = False

    # --- 3. Witness + EMA + optimizer ---
    witness = RGBWitness(
        num_pairs=P, n_fourier=args.n_fourier, hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden, mod_dim=args.mod_dim, fourier_sigma=args.fourier_sigma,
    ).to(train_device)
    ema = EMA(witness, decay=args.ema_decay)
    opt = torch.optim.AdamW(witness.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    render_h, render_w = args.render_h, args.render_w
    coords_np = _build_render_coords(render_h, render_w)
    coords_dev = torch.from_numpy(coords_np).to(train_device)
    with torch.no_grad():
        coord_feats_dev = witness.build_feats(coords_dev)  # (P_px, in_feat) fixed

    # Move GT targets to device tensors (long argmax + margin).
    lstar_dev = [torch.from_numpy(l).long().to(train_device) for l in gt.lstars]
    margin_dev = [torch.from_numpy(m).float().to(train_device) for m in gt.margins]
    pose_dev_t = [torch.from_numpy(p).float().to(train_device) for p in gt.gt_poses]

    def eval_verdict(use_ema: bool) -> dict[str, float]:
        """Frozen CPU-torch authority verdict: realized d_seg + realized d_pose."""
        orig = ema.apply_to(witness) if use_ema else None
        witness.eval()
        d_segs, d_poses = [], []
        try:
            for pi in range(P):
                f0 = _render_full_frame_for_verdict(witness, coord_feats_dev, 2 * pi, render_h, render_w)
                f1 = _render_full_frame_for_verdict(witness, coord_feats_dev, 2 * pi + 1, render_h, render_w)
                d_segs.append(cpu_verdict_d_seg(seg_cpu, f1, gt.lstars[pi]))
                d_poses.append(cpu_verdict_d_pose(posenet_cpu, f0, f1, gt.gt_poses[pi]))
        finally:
            if orig is not None:
                ema.restore(witness, orig)
            witness.train()
        return {"d_seg": float(np.mean(d_segs)), "d_pose": float(np.mean(d_poses))}

    # --- 4. Baseline verdict (untrained witness) ---
    base_verdict = eval_verdict(use_ema=False)
    print(json.dumps({"stage": "baseline_verdict", **{k: round(v, 6) for k, v in base_verdict.items()}}), flush=True)

    # --- 5. Training loop (through-R, both scorers in-loop) ---
    history: list[dict[str, Any]] = []
    best = {"d_seg": 1.0, "d_pose": 1e9, "implied_S": 1e9, "epoch": 0}
    rng = np.random.default_rng(args.seed)
    w_seg = args.w_seg
    w_pose = args.w_pose
    t_train = time.time()
    for ep in range(1, args.epochs + 1):
        ep_loss = 0.0
        order = rng.permutation(P)
        for pi_np in order:
            pi = int(pi_np)
            # Render frame0 + frame1 through R (differentiable).
            f0_cam = render_through_R(witness, coord_feats_dev, 2 * pi, render_h, render_w, ste_round=True)
            f1_cam = render_through_R(witness, coord_feats_dev, 2 * pi + 1, render_h, render_w, ste_round=True)
            seg_l = realized_seg_loss(seg_dev, f1_cam, lstar_dev[pi], margin_dev[pi], args.hinge_weight)
            pose_l = realized_pose_loss(pose_dev, f0_cam, f1_cam, pose_dev_t[pi])
            loss = w_seg * seg_l + w_pose * pose_l
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(witness)
            ep_loss += float(loss.detach().cpu())
        if ep % args.eval_every == 0 or ep == 1 or ep == args.epochs:
            # Live verdict ALWAYS (the trained weights' realized axis; EMA lags ~1000 steps
            # at decay 0.997). The promotable verdict is EMA, but for a direction-check
            # smoke the live verdict shows whether the through-R lever drives the realized
            # axis at all -- EMA catches up later in a long run.
            v_live = eval_verdict(use_ema=False)
            v = eval_verdict(use_ema=True)
            implied_S = implied_score_from_verdict(v["d_seg"], v["d_pose"], args.witness_bytes)
            implied_S_live = implied_score_from_verdict(v_live["d_seg"], v_live["d_pose"], args.witness_bytes)
            row = {
                "epoch": ep, "train_loss": round(ep_loss / P, 4),
                "d_seg": round(v["d_seg"], 6), "d_pose": round(v["d_pose"], 6),
                "implied_S": round(implied_S, 4),
                "d_seg_live": round(v_live["d_seg"], 6), "d_pose_live": round(v_live["d_pose"], 6),
                "implied_S_live": round(implied_S_live, 4),
                "wall_s": round(time.time() - t_train, 1),
            }
            # Track best by EMA implied-S (the promotable verdict).
            if implied_S < best["implied_S"]:
                best = {"d_seg": v["d_seg"], "d_pose": v["d_pose"], "implied_S": implied_S, "epoch": ep}
            # Also track the best LIVE (the lever-works direction signal).
            if implied_S_live < best.get("implied_S_live", 1e9):
                best["implied_S_live"] = implied_S_live
                best["d_seg_live"] = v_live["d_seg"]
                best["d_pose_live"] = v_live["d_pose"]
                best["epoch_live"] = ep
            history.append(row)
            print(json.dumps(row), flush=True)

    unpatch_upstream_yuv6(yuv_token)

    # --- 6. Final verdict + assemble result ---
    final = history[-1] if history else {}
    blob = _quantize_witness_blob(witness)
    result = {
        "subagent": "train_witness_realized_through_R_lever2_20260625",
        "utc": _utc(),
        "evidence_grade_train": "[macOS-MPS training-gradient]" if train_device == "mps" else f"[{train_device} training-gradient]",
        "evidence_grade_verdict": "[contest-CPU advisory]",
        "promotion_eligible": False, "score_claim": False, "ready_for_exact_eval_dispatch": False,
        "axis": "REALIZED (post-R, post-frozen-scorer) -- the actually-scored axis",
        "config": {
            "num_pairs": P, "epochs": args.epochs, "render_h": render_h, "render_w": render_w,
            "n_fourier": args.n_fourier, "hidden_dim": args.hidden_dim, "n_hidden": args.n_hidden,
            "mod_dim": args.mod_dim, "fourier_sigma": args.fourier_sigma,
            "lr": args.lr, "weight_decay": args.weight_decay, "ema_decay": args.ema_decay,
            "w_seg": w_seg, "w_pose": w_pose, "hinge_weight": args.hinge_weight,
            "train_device": train_device, "seed": args.seed,
        },
        "baseline_verdict": {k: round(v, 6) for k, v in base_verdict.items()},
        "final_verdict": {"d_seg": final.get("d_seg"), "d_pose": final.get("d_pose")},
        "best_verdict": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best.items()},
        "history": history,
        "quantized_witness_blob": blob,
        # The reference anchors FEED-ah measured (the naive-realization baseline + frontier).
        "naive_realization_d_seg_baseline": "0.005-0.008 (FEED-ah: paint EXACT L* -> SegNet through R)",
        "frontier_realized": {"d_seg": 5.6e-4, "d_pose": 1.6e-5, "bytes": 177000, "S": 0.19110},
        "gt_precompute_secs": round(gt_secs, 1),
    }
    (out_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    # Save EMA witness checkpoint (the inference weights).
    orig = ema.apply_to(witness)
    torch.save({"state_dict": witness.state_dict(), "config": result["config"]},
               out_dir / "witness_ema.pt")
    ema.restore(witness, orig)
    return result


def implied_score_from_verdict(d_seg: float, d_pose: float, witness_bytes: int) -> float:
    """The implied exact S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/RATE_DENOM.

    This is the COMPOSED implied S from the measured realized halves + the witness
    weight bytes (the rate the trained generator would cost). Advisory until byte-closed.
    """
    seg_term = 100.0 * d_seg
    pose_term = math.sqrt(10.0 * max(d_pose, 0.0))
    rate_term = 25.0 * witness_bytes / RATE_DENOM
    return seg_term + pose_term + rate_term


def _quantize_witness_blob(witness: RGBWitness) -> dict[str, Any]:
    """int8-quantize + brotli the witness weights (base) and per-frame codes (mod)."""
    import brotli

    base_chunks: list[bytes] = []
    code_chunk = b""
    n_params = 0
    for name, arr in witness.state_dict().items():
        if name == "_fourier_B":
            continue  # deterministic, free table
        a = arr.detach().cpu().numpy().astype(np.float32)
        n_params += a.size
        s = float(np.abs(a).max()) + 1e-8
        q = np.clip(np.round(a / s * 127.0), -127, 127).astype(np.int8)
        if name == "code":
            code_chunk = q.tobytes()
        else:
            base_chunks.append(q.tobytes())
    base_raw = b"".join(base_chunks)
    base_brotli = len(brotli.compress(base_raw, quality=11))
    code_brotli = len(brotli.compress(code_chunk, quality=11)) if code_chunk else 0
    return {
        "n_params": int(n_params),
        "base_int8_brotli_bytes": base_brotli,
        "code_int8_brotli_bytes": code_brotli,
        "total_quantized_blob_bytes": base_brotli + code_brotli,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAPSTONE LEVER #2: witness trained THROUGH R, both scorers in-loop")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-pairs", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    # render grid (the witness output resolution before R upsamples to camera)
    ap.add_argument("--render-h", type=int, default=192)
    ap.add_argument("--render-w", type=int, default=256)
    # witness capacity
    ap.add_argument("--n-fourier", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--n-hidden", type=int, default=4)
    ap.add_argument("--mod-dim", type=int, default=48)
    ap.add_argument("--fourier-sigma", type=float, default=8.0)
    # opt
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ema-decay", type=float, default=0.997)
    ap.add_argument("--hinge-weight", type=float, default=4.0)
    # realized-loss weights (seg vs pose). Default pose-weighted per the operating-point flip.
    ap.add_argument("--w-seg", type=float, default=1.0)
    ap.add_argument("--w-pose", type=float, default=1.0)
    # for the implied-S rate term (advisory)
    ap.add_argument("--witness-bytes", type=int, default=80000)
    ap.add_argument("--train-device", choices=["mps", "cuda", "cpu"], default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # (#254) P0 admission guard: refuse a RAW heavy launch that skipped the governed admission gate
    # (a concurrent >128 GB run CRASHED the box). ADVISORY (warn only) until enforce is armed; when
    # armed, REFUSES (exit 7) unless launched via a governed path. Runs AFTER argparse (--help stays
    # free); before any heavy allocation.
    from tac.admission_guard import assert_governed_admission
    assert_governed_admission("train_witness_realized_through_R")

    result = run_train(args)
    print("\n=== THROUGH-R WITNESS RESULT (realized axis) ===")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
