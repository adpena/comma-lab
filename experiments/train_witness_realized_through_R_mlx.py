# SPDX-License-Identifier: MIT
"""CAPSTONE LEVER #2 — MLX (mx-framework) port of the through-R WITNESS trainer.

The torch reference is ``experiments/train_witness_realized_through_R.py``. This
module mirrors its REALIZED-axis physics ENTIRELY in MLX (mx) so the inner loop
runs on the M5 Max GPU at FP32 with unified-memory saturation:

  witness(coords, per-(pair,frame) code) -> RGB at scorer res
    -> R [bicubic up to camera -> bilinear down to 384x512 -> uint8-STE]
    -> frame1 -> frozen MLX SegNet -> realized seg CE (vs L*)
    -> pair  -> rgb_to_yuv6_mlx -> frozen MLX PoseNet -> realized pose MSE (vs GT pose)
  mx.value_and_grad over (witness params) -> AdamW step -> EMA.

WHY MLX (operator directive 2026-06-25): the torch path is impractical — torch
n600 = ~935 s/ep (MPS) and ~25 s/ep (CPU at n16); AND torch-MPS drifts 23x on
pose (NEVER authority). The MLX scorer port is FP32-accurate: the through-R
gradient's MLX-CPU-vs-MLX-GPU cosine is 0.99996 (smoke 2026-06-25), and the
SegNet MLX-vs-torch per-logit drift bound is ~0.038 max / ~0.002 mean. MLX GPU
gives a usable+accurate GRADIENT, fast, with everything resident in unified mem.

COMPUTE-SUBSTRATE LAW (CLAUDE.md "MLX portable-local-substrate authority" +
"MPS NEVER authority"): MLX (CPU or GPU) is a VALID fp32 TRAINING-GRADIENT
device, NEVER a score authority. The d_seg/d_pose VERDICT comes from the FROZEN
CPU-torch SegNet argmax + PoseNet (NEVER MLX, NEVER MPS). The MLX scorer is the
fast accurate gradient + advisory; the verdict is recomputed on CPU-torch. EMA +
eval_roundtrip (R) are non-negotiable and in-loop.

NO-FAKE: the realized d_seg VERDICT = argmax-disagreement RATE between the FROZEN
CPU-torch SegNet on the MLX-rendered-through-R frame1 and L* (GT argmax). The
realized d_pose VERDICT = pose-MSE of the FROZEN CPU-torch PoseNet on the
rendered pair vs GT pose. The MLX scorer trains the witness; the CPU-torch scorer
is the verdict. No surrogate is the verdict.

BORROWED-SUBSTRATE (CLAUDE.md NO-FAKE #7):
  * BORROWED: the MLX SegNet/PoseNet port (tac.local_acceleration.mlx_scorer_*),
    the canonical MLX R + rgb_to_yuv6 primitives (pr95_hnerv_mlx_training:
    apply_eval_roundtrip_nhwc, rgb_to_yuv6_mlx), the frozen contest scorers +
    GT-decode + CPU authority (seg_core + upstream modules), Fourier-feature
    coord-INR (Tancik 2020) + FiLM (Perez 2018) + HNeRV amortized framing.
  * OURS-ORIGINAL (same vehicle as the torch trainer, now on MLX): a NON-RGB-
    vehicle coord-INR OUTPUTTING RGB whose REALIZED (post-R, post-frozen-scorer)
    argmax partition survives R AND whose PoseNet pose matches GT, JOINTLY in one
    shared code -- the round-trip-survival lever with BOTH scorers in-loop, the
    joint realized d_seg+d_pose loss, ported to mx for throughput. NOT a PR95
    HNeRV reskin (anti-reskin: our coord-INR, not their decoder).

Evidence: training gradient [macOS-MLX training-gradient]; the d_seg/d_pose
VERDICT is [contest-CPU advisory] (frozen CPU-torch mirror of evaluate.py over
the measured pair subset). promotion_eligible=False; NO score claim beyond
advisory; pointer UNMOVED unless a real byte-closed sub-0.19110 exact-eval row
lands. Disk: SSD/repo results, NEVER /tmp durable.
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

# Camera-native + scorer-input resolutions (from upstream/frame_utils.py).
CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512
RATE_DENOM = 37_545_489.0
FRONTIER = 0.19110
N_CLASSES = 5

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
_FOURIER_SEED = 0


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field_name: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field_name}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# Deterministic Fourier B (shared with the torch trainer for parity).
# ---------------------------------------------------------------------------
def deterministic_fourier_B(n_fourier: int, fourier_sigma: float) -> np.ndarray:
    rng = np.random.default_rng(_FOURIER_SEED)
    return (rng.standard_normal((2, n_fourier)) * fourier_sigma).astype(np.float32)


# ---------------------------------------------------------------------------
# The witness: an mlx.nn.Module coord-INR (Fourier feats + per-(pair,frame) FiLM)
# -> RGB at scorer res (NHWC). OUR coord-INR vehicle (anti-reskin), on MLX.
# ---------------------------------------------------------------------------
def build_witness_module(
    num_pairs: int,
    n_fourier: int,
    hidden_dim: int,
    n_hidden: int,
    mod_dim: int,
    fourier_sigma: float,
):
    import mlx.core as mx
    import mlx.nn as nn

    class RGBWitnessMLX(nn.Module):
        """g(coord_feats, code_idx) -> (P_px, 3) RGB in [0,255]."""

        def __init__(self) -> None:
            super().__init__()
            self.num_pairs = num_pairs
            self.n_fourier = n_fourier
            self.hidden_dim = hidden_dim
            self.n_hidden = n_hidden
            self.mod_dim = mod_dim
            in_feat = 2 * n_fourier
            self.in_feat = in_feat
            B = deterministic_fourier_B(n_fourier, fourier_sigma)
            # Fixed (non-trainable) Fourier projection, held as a plain constant.
            self._B = mx.array(B)
            self.code = mx.zeros((num_pairs * 2, mod_dim))
            self.in_proj = nn.Linear(in_feat, hidden_dim)
            self.film = nn.Linear(mod_dim, 2 * hidden_dim * n_hidden)
            self.hidden = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
            self.out = nn.Linear(hidden_dim, 3)

        def build_feats(self, coords: Any) -> Any:
            proj = coords @ self._B  # (P, n_fourier)
            return mx.concatenate([mx.sin(proj), mx.cos(proj)], axis=-1)

        def __call__(self, coord_feats: Any, code_idx: int) -> Any:
            h = nn.relu(self.in_proj(coord_feats))
            m = self.code[code_idx]
            film = mx.reshape(self.film(m), (self.n_hidden, 2, self.hidden_dim))
            for li, layer in enumerate(self.hidden):
                scale = 1.0 + film[li, 0]
                shift = film[li, 1]
                h = nn.relu(layer(h) * scale + shift)
            rgb = mx.sigmoid(self.out(h)) * 255.0
            return rgb

    return RGBWitnessMLX()


# ---------------------------------------------------------------------------
# Render through R (mx, differentiable). Witness -> (h,w,3) RGB at scorer res ->
# apply_eval_roundtrip_nhwc (bicubic up to camera -> bilinear down to scorer ->
# uint8 STE). Returns (1, h, w, 3) NHWC camera-equivalent scorer-res frame.
# ---------------------------------------------------------------------------
def render_through_R_mlx(witness, coord_feats, code_idx: int, render_h: int, render_w: int):
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import apply_eval_roundtrip_nhwc

    rgb_flat = witness(coord_feats, code_idx)  # (h*w, 3)
    rgb = mx.reshape(rgb_flat, (1, render_h, render_w, 3))  # NHWC
    # R: render-grid -> camera (bicubic) -> scorer-res (bilinear) -> uint8 STE.
    r = apply_eval_roundtrip_nhwc(rgb, output_hw=(SEG_H, SEG_W), ste_round=True)
    return r  # (1, SEG_H, SEG_W, 3)


# ---------------------------------------------------------------------------
# GT precompute (FROZEN CPU-torch authority -- identical to the torch trainer).
# ---------------------------------------------------------------------------
@dataclass
class GTData:
    n_pairs: int
    gt_f0: list[np.ndarray]
    gt_f1: list[np.ndarray]
    lstars: list[np.ndarray]
    margins: list[np.ndarray]
    gt_poses: list[np.ndarray]


def precompute_gt(n_pairs: int) -> tuple[GTData, Any, Any]:
    import torch

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
        gt_poses.append(_cpu_pose_raw(posenet_cpu, f0, f1))
    gt = GTData(n_pairs=len(lstars), gt_f0=gt_f0, gt_f1=gt_f1, lstars=lstars, margins=margins, gt_poses=gt_poses)
    return gt, seg_cpu, posenet_cpu


def _cpu_pose_raw(posenet_cpu, f0_uint8: np.ndarray, f1_uint8: np.ndarray) -> np.ndarray:
    import einops
    import torch

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
# CPU-torch VERDICT (authority, NEVER MLX). Mirrors the torch trainer exactly.
# ---------------------------------------------------------------------------
def cpu_verdict_d_seg(segnet_cpu, frame1_uint8: np.ndarray, gt_argmax_np: np.ndarray) -> float:
    import torch

    r = np.asarray(frame1_uint8)
    pair = torch.from_numpy(np.stack([r, r], axis=0)[None]).float()  # (1,2,H,W,3)
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        seg_in = segnet_cpu.preprocess_input(xp)
        logits = segnet_cpu(seg_in)
        realized = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
    return float(np.count_nonzero(realized != gt_argmax_np)) / gt_argmax_np.size


def cpu_verdict_d_pose(posenet_cpu, f0_uint8: np.ndarray, f1_uint8: np.ndarray, gt_pose_np: np.ndarray) -> float:
    p = _cpu_pose_raw(posenet_cpu, f0_uint8, f1_uint8)
    return float(np.mean((p - gt_pose_np) ** 2))


# ---------------------------------------------------------------------------
# MLX EMA (decay 0.997, Quantizr). Shadow over the flattened param tree.
# ---------------------------------------------------------------------------
class MlxEMA:
    def __init__(self, model, decay: float = 0.997) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        self.decay = decay
        self._mx = mx
        self.shadow = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}

    def update(self, model) -> None:
        from mlx.utils import tree_flatten

        d = self.decay
        for k, v in tree_flatten(model.parameters()):
            self.shadow[k] = self.shadow[k] * d + v * (1.0 - d)

    def shadow_tree(self):
        from mlx.utils import tree_unflatten

        return tree_unflatten(list(self.shadow.items()))


# ---------------------------------------------------------------------------
# Implied S (advisory; identical formula to the torch trainer + evaluate.py).
# ---------------------------------------------------------------------------
def implied_score_from_verdict(d_seg: float, d_pose: float, witness_bytes: int) -> float:
    seg_term = 100.0 * d_seg
    pose_term = math.sqrt(10.0 * max(d_pose, 0.0))
    rate_term = 25.0 * witness_bytes / RATE_DENOM
    return seg_term + pose_term + rate_term


def quantize_witness_blob(model) -> dict[str, Any]:
    """int8 + brotli the witness params (base) and per-frame codes (mod)."""
    import brotli
    from mlx.utils import tree_flatten

    base_chunks: list[bytes] = []
    code_chunk = b""
    n_params = 0
    for name, arr in tree_flatten(model.parameters()):
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            continue
        n_params += a.size
        s = float(np.abs(a).max()) + 1e-8
        q = np.clip(np.round(a / s * 127.0), -127, 127).astype(np.int8)
        if name.endswith("code"):
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


def _build_render_coords(h: int, w: int) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Realized loss (mx, through both frozen MLX scorers). The training signal.
# ---------------------------------------------------------------------------
def make_loss_fn(adapter, render_h: int, render_w: int):
    """Returns loss_fn(model, coord_feats, code0, code1, lstar_oh, margin, pose_tgt,
    w_seg, w_pose, hinge) -> scalar mx loss. Closes over the frozen MLX adapter."""
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    def _yuv6_pair_nhwc(f0, f1):
        # f0,f1: (1,H,W,3). Build (1,2,H,W,3) -> yuv6 (1,2,h2,w2,6) -> (1,h2,w2,12).
        pair = mx.stack([f0[0], f1[0]], axis=0)[None]  # (1,2,H,W,3)
        yuv = rgb_to_yuv6_mlx(pair)  # (1,2,h2,w2,6)
        b, t, h2, w2, c6 = yuv.shape
        return mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (b, h2, w2, t * c6))

    def loss_fn(model, coord_feats, code0, code1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge):
        f0 = render_through_R_mlx(model, coord_feats, code0, render_h, render_w)
        f1 = render_through_R_mlx(model, coord_feats, code1, render_h, render_w)
        # Realized seg CE (margin-weighted) on frame1.
        seg_logits = adapter.segnet(f1)  # (1,H,W,5)
        logsum = mx.logsumexp(seg_logits, axis=-1)
        tgt = mx.sum(seg_logits * lstar_oh, axis=-1)
        ce = logsum - tgt  # (1,H,W)
        w = 1.0 + hinge * mx.exp(-mx.clip(margin, 0.0, 1e9))
        seg_l = mx.mean(ce * w[None])
        # Realized pose MSE on the pair.
        yuv_nhwc = _yuv6_pair_nhwc(f0, f1)
        pose = adapter.posenet(yuv_nhwc)["pose"][..., : pose_tgt.shape[-1]]
        pose_l = mx.mean(mx.square(pose[0] - pose_tgt))
        return w_seg * seg_l + w_pose * pose_l

    return loss_fn


def _render_uint8_for_verdict(model, coord_feats, code_idx: int, render_h: int, render_w: int) -> np.ndarray:
    import mlx.core as mx

    r = render_through_R_mlx(model, coord_feats, code_idx, render_h, render_w)  # (1,H,W,3)
    mx.eval(r)
    arr = np.asarray(r[0], dtype=np.float32)
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )

    out_dir = Path(args.out_dir)
    _refuse_tmp(out_dir, "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    device = args.mlx_device  # "gpu" or "cpu"

    # --- 1. GT precompute on frozen CPU scorers (authority) ---
    t0 = time.time()
    gt, seg_cpu, posenet_cpu = precompute_gt(args.num_pairs)
    P = gt.n_pairs
    gt_secs = time.time() - t0
    print(json.dumps({"stage": "gt_precompute", "n_pairs": P, "secs": round(gt_secs, 1)}), flush=True)

    render_h, render_w = args.render_h, args.render_w
    coords_np = _build_render_coords(render_h, render_w)

    result_history: list[dict[str, Any]] = []
    best = {"d_seg": 1.0, "d_pose": 1e9, "implied_S": 1e9, "epoch": 0}

    with temporary_mlx_device(device):
        # --- 2. Frozen MLX scorers on the train device (gradient device) ---
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")

        # --- 3. Witness + EMA + AdamW ---
        model = build_witness_module(
            num_pairs=P, n_fourier=args.n_fourier, hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden, mod_dim=args.mod_dim, fourier_sigma=args.fourier_sigma,
        )
        mx.eval(model.parameters())
        ema = MlxEMA(model, decay=args.ema_decay)
        opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

        coords = mx.array(coords_np)
        coord_feats = model.build_feats(coords)  # fixed (P_px, in_feat)
        mx.eval(coord_feats)

        # GT targets as mx (one-hot L*, margin, pose).
        lstar_oh = []
        margins_mx = []
        poses_mx = []
        for pi in range(P):
            lo = np.eye(N_CLASSES, dtype=np.float32)[gt.lstars[pi]]  # (H,W,5)
            lstar_oh.append(mx.array(lo[None]))  # (1,H,W,5)
            margins_mx.append(mx.array(gt.margins[pi]))  # (H,W)
            poses_mx.append(mx.array(gt.gt_poses[pi].astype(np.float32)))  # (half,)

        loss_fn = make_loss_fn(adapter, render_h, render_w)
        value_and_grad = nn.value_and_grad(model, loss_fn)

        def eval_verdict(use_ema: bool) -> dict[str, float]:
            """FROZEN CPU-torch authority verdict on the MLX-rendered frames."""
            saved = None
            if use_ema:
                saved = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
                model.update(ema.shadow_tree())
                mx.eval(model.parameters())
            d_segs, d_poses = [], []
            try:
                for pi in range(P):
                    f0 = _render_uint8_for_verdict(model, coord_feats, 2 * pi, render_h, render_w)
                    f1 = _render_uint8_for_verdict(model, coord_feats, 2 * pi + 1, render_h, render_w)
                    d_segs.append(cpu_verdict_d_seg(seg_cpu, f1, gt.lstars[pi]))
                    d_poses.append(cpu_verdict_d_pose(posenet_cpu, f0, f1, gt.gt_poses[pi]))
            finally:
                if saved is not None:
                    model.update(tree_unflatten(list(saved.items())))
                    mx.eval(model.parameters())
            return {"d_seg": float(np.mean(d_segs)), "d_pose": float(np.mean(d_poses))}

        base_verdict = eval_verdict(use_ema=False)
        print(json.dumps({"stage": "baseline_verdict", **{k: round(v, 6) for k, v in base_verdict.items()}}), flush=True)

        # --- 4. Training loop (through-R, both MLX scorers in-loop) ---
        rng = np.random.default_rng(args.seed)
        w_seg = mx.array(float(args.w_seg))
        w_pose = mx.array(float(args.w_pose))
        hinge = mx.array(float(args.hinge_weight))
        t_train = time.time()
        ep_walls: list[float] = []
        for ep in range(1, args.epochs + 1):
            ep_t0 = time.time()
            ep_loss = 0.0
            order = rng.permutation(P)
            for pi_np in order:
                pi = int(pi_np)
                loss, grads = value_and_grad(
                    model, coord_feats, 2 * pi, 2 * pi + 1,
                    lstar_oh[pi], margins_mx[pi], poses_mx[pi], w_seg, w_pose, hinge,
                )
                opt.update(model, grads)
                mx.eval(model.parameters(), opt.state)
                ema.update(model)
                ep_loss += float(loss)
            ep_walls.append(time.time() - ep_t0)
            if ep % args.eval_every == 0 or ep == 1 or ep == args.epochs:
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
                    "ep_wall_s": round(ep_walls[-1], 2),
                    "wall_s": round(time.time() - t_train, 1),
                }
                if implied_S < best["implied_S"]:
                    best = {"d_seg": v["d_seg"], "d_pose": v["d_pose"], "implied_S": implied_S, "epoch": ep}
                if implied_S_live < best.get("implied_S_live", 1e9):
                    best["implied_S_live"] = implied_S_live
                    best["d_seg_live"] = v_live["d_seg"]
                    best["d_pose_live"] = v_live["d_pose"]
                    best["epoch_live"] = ep
                result_history.append(row)
                print(json.dumps(row), flush=True)

        blob = quantize_witness_blob(model)
        # Save the EMA witness (inference weights).
        model.update(ema.shadow_tree())
        mx.eval(model.parameters())
        ema_params = {k: np.asarray(v, dtype=np.float32) for k, v in tree_flatten(model.parameters())}
        np.savez(out_dir / "witness_ema_mlx.npz", **ema_params)

    mean_ep = float(np.mean(ep_walls)) if ep_walls else 0.0
    median_ep = float(np.median(ep_walls)) if ep_walls else 0.0
    final = result_history[-1] if result_history else {}
    result = {
        "subagent": "train_witness_realized_through_R_mlx_lever2_20260625",
        "utc": _utc(),
        "evidence_grade_train": "[macOS-MLX training-gradient]",
        "evidence_grade_verdict": "[contest-CPU advisory]",
        "promotion_eligible": False, "score_claim": False, "ready_for_exact_eval_dispatch": False,
        "axis": "REALIZED (post-R, post-frozen-scorer) -- the actually-scored axis",
        "mlx_device": device,
        "config": {
            "num_pairs": P, "epochs": args.epochs, "render_h": render_h, "render_w": render_w,
            "n_fourier": args.n_fourier, "hidden_dim": args.hidden_dim, "n_hidden": args.n_hidden,
            "mod_dim": args.mod_dim, "fourier_sigma": args.fourier_sigma,
            "lr": args.lr, "weight_decay": args.weight_decay, "ema_decay": args.ema_decay,
            "w_seg": args.w_seg, "w_pose": args.w_pose, "hinge_weight": args.hinge_weight,
            "seed": args.seed,
        },
        "throughput": {
            "mean_ep_wall_s": round(mean_ep, 3),
            "median_ep_wall_s": round(median_ep, 3),
            "n_pairs": P, "epochs": args.epochs,
        },
        "baseline_verdict": {k: round(v, 6) for k, v in base_verdict.items()},
        "final_verdict": {"d_seg": final.get("d_seg"), "d_pose": final.get("d_pose")},
        "best_verdict": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best.items()},
        "history": result_history,
        "quantized_witness_blob": blob,
        "gt_precompute_secs": round(gt_secs, 1),
        "frontier_realized": {"d_seg": 5.6e-4, "d_pose": 1.6e-5, "bytes": 177000, "S": FRONTIER},
    }
    (out_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAPSTONE LEVER #2 (MLX): witness through R, both MLX scorers in-loop")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-pairs", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--render-h", type=int, default=192)
    ap.add_argument("--render-w", type=int, default=256)
    ap.add_argument("--n-fourier", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--n-hidden", type=int, default=4)
    ap.add_argument("--mod-dim", type=int, default=48)
    ap.add_argument("--fourier-sigma", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ema-decay", type=float, default=0.997)
    ap.add_argument("--hinge-weight", type=float, default=4.0)
    ap.add_argument("--w-seg", type=float, default=1.0)
    ap.add_argument("--w-pose", type=float, default=1.0)
    ap.add_argument("--witness-bytes", type=int, default=80000)
    ap.add_argument("--mlx-device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    result = run_train(args)
    print("\n=== THROUGH-R WITNESS RESULT (MLX, realized axis) ===")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
