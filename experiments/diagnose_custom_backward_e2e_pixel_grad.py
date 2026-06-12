#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""DECISIVE localization: where does the n600 pose divergence come from?

The per-layer random-data gradient cosine is 1.0 for ALL 12 strided-grouped
shapes (test_metal_grouped_conv_backward.py, 54 passing). Yet the n600 training
run diverged on the POSE axis. This diagnostic computes the FULL end-to-end
``dL/d(pixels)`` cotangent on REAL render data through three gradient sources and
compares them DIRECTLY (cosine + relmax), separately for the SegNet-only loss and
the PoseNet-only loss:

  * torch-CPU authority      : TorchScorerBridge (the gradient authority)
  * mlx_gpu REFERENCE backward: TAC_MLX_CUSTOM_GROUPED_BACKWARD=0 (loop fallback)
  * mlx_gpu CUSTOM backward   : TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 (the suspect)

Decisive logic (NO-FAKE):
  * If CUSTOM and REFERENCE agree with each other (cosine ~1.0) but BOTH disagree
    with torch-CPU on POSE, the divergence is the mlx_gpu FORWARD fidelity (the
    documented ~2.76e-4 pose drift), NOT the custom backward -> kernel EXONERATED.
  * If CUSTOM disagrees with REFERENCE on POSE (lower cosine / higher relmax), the
    custom backward has a real pose-shape gradient bug -> kernel CONVICTED.

Authority: exact d_seg/d_pose come from torch-CPU. The pixel-grad comparison runs
the MLX bridge twice (custom + reference) on the SAME data; NO MPS; $0; local.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _relmax(a: np.ndarray, b: np.ndarray) -> float:
    # |a - b|_max / |b|_max  (b is the reference)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def _build_render_np(bundle, idx):
    """Decode the carrier render for the selected pairs as a numpy (B,2,3,h,w).

    The bundle is an MLX module returning an mx.array; decode on the MLX CPU so
    the render numerics are device-stable, then return numpy for both bridges.
    """
    import mlx.core as mx

    idx_mx = mx.array(np.asarray(idx, dtype=np.int32))
    out = bundle(idx_mx)
    mx.eval(out)
    return np.asarray(out).astype(np.float32)


def _mlx_pixel_grad(net, seg_t, pose_t, render_n2chw_np, idx, *, which, custom_backward):
    """Return (seg_pixel_grad, pose_pixel_grad) np arrays from the mlx_gpu bridge.

    ``which`` selects 'seg' or 'pose' isolation by zeroing the other weight.
    We call loss_and_pixel_grad twice: once seg-only (pose_weight=0) and once
    pose-only (seg_weight=0) so the two cotangents are cleanly separated.
    """
    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1" if custom_backward else "0"
    os.environ.setdefault("MLX_METAL_GPU_ARCH", "applegpu_g15")
    import mlx.core as mx

    from tac.mlx_pr95_port.mlx_gpu_score_bridge import MLXGpuScorerBridge

    render_mx = mx.array(render_n2chw_np.astype(np.float32))
    idx_t = torch.as_tensor(idx, dtype=torch.long)

    grads = {}
    for tag, sw, pw in (("seg", 100.0, 0.0), ("pose", 0.0, 1.0)):
        bridge = MLXGpuScorerBridge(
            net, seg_t, pose_t,
            seg_loss_form="ce_seg_loss", seg_weight=sw, pose_weight=pw,
            eval_roundtrip=True, device_type="gpu",
        )
        res = bridge.loss_and_pixel_grad(render_mx, idx_t)
        g = np.asarray(res.pixel_cotangent)
        grads[tag] = g
        del bridge
    return grads["seg"], grads["pose"]


def _torch_pixel_grad(net, seg_t, pose_t, render_n2chw_np, idx):
    import mlx.core as mx

    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    render_mx = mx.array(render_n2chw_np.astype(np.float32))
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    grads = {}
    for tag, sw, pw in (("seg", 100.0, 0.0), ("pose", 0.0, 1.0)):
        bridge = TorchScorerBridge(
            net, seg_t, pose_t,
            seg_loss_form="ce_seg_loss", seg_weight=sw, pose_weight=pw,
            eval_roundtrip=True,
        )
        res = bridge.loss_and_pixel_grad(render_mx, idx_t)
        grads[tag] = np.asarray(res.pixel_cotangent)
        del bridge
    return grads["seg"], grads["pose"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--base-channels", type=int, default=20)
    ap.add_argument("--carrier", default="stored_latent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets-cache", default="experiments/results/capstone_gt_targets_cache")
    ap.add_argument("--out-json", default="experiments/results/custom_backward_e2e_pixel_grad_diag.json")
    ap.add_argument("--train-steps", type=int, default=0,
                    help="train the bundle this many mlx_gpu CUSTOM-backward steps "
                         "first, so the comparison is on a partially-trained render "
                         "(the regime where n600 diverged), not just init.")
    args = ap.parse_args()
    torch.set_num_threads(min(6, torch.get_num_threads()))

    from tac.capstone_vq_nerv.vq_nerv_bundle import CapstoneVqNervBundle, CapstoneVqNervConfig
    from tac.score_aware_loop.targets import build_gt_targets, load_frozen_distortion_net

    net = load_frozen_distortion_net(device="cpu")
    cache = Path(args.targets_cache) / f"gt_targets_n{args.max_pairs}.pt"
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        seg_t, pose_t = blob["seg"][: args.max_pairs], blob["pose"][: args.max_pairs]
    else:
        seg_t, pose_t, _ = build_gt_targets(net, max_pairs=args.max_pairs, device="cpu")
        seg_t, pose_t = seg_t[: args.max_pairs], pose_t[: args.max_pairs]

    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=args.max_pairs, base_channels=args.base_channels,
            carrier=args.carrier, seed=args.seed,
        )
    )
    # Optionally TRAIN the bundle first (mlx_gpu custom backward) so the render
    # carries the partially-trained structure of the n600-divergence regime.
    if args.train_steps > 0:
        os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1"
        os.environ.setdefault("MLX_METAL_GPU_ARCH", "applegpu_g15")
        from tac.capstone_vq_nerv.capstone_trainer import CapstoneTrainConfig, CapstoneTrainer
        from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

        bridge = TorchScorerBridge(net, seg_t, pose_t, seg_loss_form="ce_seg_loss",
                                   seg_weight=100.0, pose_weight=1.0, eval_roundtrip=True)
        cfg = CapstoneTrainConfig(epochs=1, batch_size=args.batch_size, seed=args.seed,
                                  scorer_backend="mlx_gpu", authority_recheck_every=0,
                                  muon_lr=0.03, grad_clip=50.0, grad_clip_muon=50.0)
        trainer = CapstoneTrainer(bundle, bridge, pose_t.float().cpu().numpy(), cfg)
        rng = np.random.RandomState(args.seed)
        for step in range(args.train_steps):
            sub = rng.permutation(args.max_pairs)[: args.batch_size]
            trainer.step(sub, lr_scale=1.0)
        print(f"trained {args.train_steps} mlx_gpu custom-backward steps", flush=True)
        del trainer, bridge

    idx = np.arange(args.batch_size)
    render_np = _build_render_np(bundle, idx)
    print(f"render shape {render_np.shape}  range [{render_np.min():.2f}, {render_np.max():.2f}]", flush=True)

    print("=== torch-CPU authority pixel grad (seg-only, pose-only) ===", flush=True)
    t_seg, t_pose = _torch_pixel_grad(net, seg_t, pose_t, render_np, idx)
    print(f"  seg |grad|_max={np.abs(t_seg).max():.4e}  pose |grad|_max={np.abs(t_pose).max():.4e}", flush=True)

    print("=== mlx_gpu REFERENCE backward pixel grad ===", flush=True)
    r_seg, r_pose = _mlx_pixel_grad(net, seg_t, pose_t, render_np, idx, which="both", custom_backward=False)
    print(f"  seg |grad|_max={np.abs(r_seg).max():.4e}  pose |grad|_max={np.abs(r_pose).max():.4e}", flush=True)

    print("=== mlx_gpu CUSTOM backward pixel grad ===", flush=True)
    c_seg, c_pose = _mlx_pixel_grad(net, seg_t, pose_t, render_np, idx, which="both", custom_backward=True)
    print(f"  seg |grad|_max={np.abs(c_seg).max():.4e}  pose |grad|_max={np.abs(c_pose).max():.4e}", flush=True)

    cmp = {
        # The DECISIVE comparison: custom-vs-reference (both mlx_gpu, isolate the
        # kernel) AND each-vs-torch (the forward-fidelity floor).
        "seg_custom_vs_reference_cos": _cos(c_seg, r_seg),
        "seg_custom_vs_reference_relmax": _relmax(c_seg, r_seg),
        "pose_custom_vs_reference_cos": _cos(c_pose, r_pose),
        "pose_custom_vs_reference_relmax": _relmax(c_pose, r_pose),
        "seg_reference_vs_torch_cos": _cos(r_seg, t_seg),
        "seg_custom_vs_torch_cos": _cos(c_seg, t_seg),
        "pose_reference_vs_torch_cos": _cos(r_pose, t_pose),
        "pose_custom_vs_torch_cos": _cos(c_pose, t_pose),
    }
    print("\n=== COMPARISON (cotangent on render pixels) ===")
    for k, v in cmp.items():
        print(f"  {k:42s} {v:.8f}")

    # Verdict: convict the kernel ONLY if custom diverges from reference on POSE.
    pose_kernel_ok = cmp["pose_custom_vs_reference_cos"] > 0.9999 and cmp["pose_custom_vs_reference_relmax"] < 1e-3
    seg_kernel_ok = cmp["seg_custom_vs_reference_cos"] > 0.9999 and cmp["seg_custom_vs_reference_relmax"] < 1e-3
    if pose_kernel_ok and seg_kernel_ok:
        verdict = "KERNEL_GRAD_MATCHES_REFERENCE_e2e_real_data_pose_divergence_is_mlx_forward_fidelity"
    elif not pose_kernel_ok:
        verdict = "KERNEL_CONVICTED_pose_pixel_grad_diverges_from_reference"
    else:
        verdict = "KERNEL_SEG_GRAD_DIVERGES_unexpected"
    print(f"\nVERDICT: {verdict}")

    out = {"config": vars(args), "comparison": cmp, "verdict": verdict,
           "grad_max": {
               "torch_seg": float(np.abs(t_seg).max()), "torch_pose": float(np.abs(t_pose).max()),
               "ref_seg": float(np.abs(r_seg).max()), "ref_pose": float(np.abs(r_pose).max()),
               "custom_seg": float(np.abs(c_seg).max()), "custom_pose": float(np.abs(c_pose).max()),
           }}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
