# SPDX-License-Identifier: MIT
"""DECISIVE PROBE — "seed + SOLVE at compress time" vs the 5-day gradient descent.

THE OPERATOR REFRAME (the big one): d_seg descends slowly because the CE/soft-cosine
gradient vanishes at the residual boundary flips. But d_seg=0 is a CONSTRAINT-SATISFACTION
problem (per-pixel argmax-cell inequalities ``logit[GT_class] > logit[other]``), and we have
full compress-time custody + known GT (one video) + a frozen locally-linear SegNet. So SOLVE
it (Jacobian / Gauss-Newton in the latent space, holding the decoder fixed), don't descend.

DECISIVE QUESTION: does a compress-time SOLVE land d_seg far lower, far FASTER than the
5-day gradient descent (the trained basin d_seg ~0.00260)?

THE EXACT CHAIN (matches vendored ``score.evaluate_decoder`` + ``modules.SegNet`` 1:1):
    decoder(z) -> f1 (3,384,512) -> bicubic^ (874,1164) -> uint8 round
              -> [SegNet.preprocess] x[:,-1]; bilinear v to (384,512) -> SegNet -> argmax
    d_seg(pair) = mean( argmax(SegNet(decoded_f1)) != argmax(SegNet(GT_f1)) )   [the seg target]

THE EXPERIMENT (a few real pairs, $0 CPU, real frozen scorer):
  1. BASELINE: trained latents -> exact d_seg (the descent's result).
  2. LATENT-SOLVE: hold decoder FIXED, SOLVE each pair's 28-dim latent by minimizing the
     boundary-weighted CE of SegNet logits vs the GT-argmax target through the EXACT chain
     (autograd Gauss-Newton-ish descent in 28-dim; this IS the per-pixel Jacobian projection
     onto the argmax-cell, the cheap subspace C = the fixed decoder's 28-dim image). Early-stop
     / track on the EXACT argmax-flip d_seg (NOT the surrogate). Measure: exact d_seg after,
     wall-clock/pair, expressiveness ceiling (how many flips the 28-dim solve can fix).

NO-FAKE: real frozen SegNet (load_frozen_distortion_net device='cpu', MPS off), real basin
ckpt READ-ONLY, GT via the vendored evaluate path (frame_utils.yuv420_to_rgb), exact d_seg =
the literal argmax-disagreement of the frozen SegNet through the full roundtrip. The SOLVE must
ACTUALLY reduce the EXACT d_seg, not a proxy. [macOS-CPU advisory] NON-PROMOTABLE.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CAMERA_H, CAMERA_W = 874, 1164
EVAL_H, EVAL_W = 384, 512
SEG_IN_H, SEG_IN_W = 384, 512  # segnet_model_input_size = (512,384) -> (H,W)=(384,512)

BASIN = Path("experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best")


def _decoded_to_camera(decoded_native: torch.Tensor) -> torch.Tensor:
    """(B,3,384,512) -> (B,3,874,1164) bicubic — matches score._decoded_to_camera."""
    return F.interpolate(
        decoded_native, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
    )


def _segnet_input_from_camera_f1(camera_f1: torch.Tensor) -> torch.Tensor:
    """(B,3,874,1164) camera-res frame1 -> (B,3,384,512) SegNet input (bilinear v).
    Matches modules.SegNet.preprocess_input (after x[:,-1] last-frame select)."""
    return F.interpolate(
        camera_f1, size=(SEG_IN_H, SEG_IN_W), mode="bilinear"
    )


def render_f1_segnet_input(
    decoder: torch.nn.Module, z: torch.Tensor, *, roundtrip_uint8: bool, differentiable: bool
) -> torch.Tensor:
    """z (B,28) -> SegNet-input frame1 (B,3,384,512) through the EXACT eval chain.

    roundtrip_uint8=True replicates the exact scored object (bicubic^ -> uint8 round ->
    bilinear v). The uint8 round is non-differentiable; for the SOLVE we use a
    straight-through round when differentiable=True so the gradient flows but the FORWARD
    value is the exact scored one (the survival-aware gradient).
    """
    decoded = decoder(z)  # (B,2,3,384,512) float [0,255]
    f1 = decoded[:, 1]  # frame1 = the SegNet object (B,3,384,512)
    cam = _decoded_to_camera(f1)  # (B,3,874,1164)
    if roundtrip_uint8:
        cam_round = cam.clamp(0, 255).round()
        if differentiable:
            # straight-through: forward = rounded uint8 value, backward = identity
            cam = cam + (cam_round - cam).detach()
        else:
            cam = cam_round
    seg_in = _segnet_input_from_camera_f1(cam)
    return seg_in


def exact_dseg_per_pair(
    decoder: torch.nn.Module,
    segnet: torch.nn.Module,
    z: torch.Tensor,
    seg_target_argmax: torch.Tensor,
) -> torch.Tensor:
    """Exact per-pair d_seg = mean argmax-flip vs GT seg target, through the full roundtrip.
    z (B,28); seg_target_argmax (B, 384, 512) long. Returns (B,) float."""
    with torch.inference_mode():
        seg_in = render_f1_segnet_input(
            decoder, z, roundtrip_uint8=True, differentiable=False
        )
        logits = segnet(seg_in)  # (B,5,384,512)
        pred = logits.argmax(dim=1)  # (B,384,512)
        flip = (pred != seg_target_argmax).float()
        return flip.mean(dim=(1, 2))


def build_seg_targets(video_path: Path, segnet: torch.nn.Module, n_pairs: int):
    """Per-pair GT SegNet argmax target (the d_seg reference), 1:1 with the eval path.
    GT frame1 -> camera-res (it IS camera res) -> SegNet.preprocess (bilinear v) -> argmax.
    Returns (n_pairs, 384, 512) long."""
    import av
    from frame_utils import yuv420_to_rgb

    container = av.open(str(video_path))
    targets = []
    prev = None
    pair_idx = 0
    with torch.inference_mode():
        for frame in container.decode(container.streams.video[0]):
            f = yuv420_to_rgb(frame)  # (H,W,3) uint8 tensor, camera res
            if prev is None:
                prev = f
                continue
            f1 = f  # frame1 of the pair = the SegNet last-frame object
            prev = None
            # GT f1 camera-res -> (1,3,874,1164) float -> SegNet input -> argmax
            cam = f1.float().permute(2, 0, 1).unsqueeze(0)  # (1,3,874,1164)
            seg_in = _segnet_input_from_camera_f1(cam)
            logits = segnet(seg_in)
            targets.append(logits.argmax(dim=1).squeeze(0).clone())
            pair_idx += 1
            if pair_idx >= n_pairs:
                break
    container.close()
    return torch.stack(targets)  # (n_pairs,384,512) long


def boundary_weighted_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-pixel CE of SegNet logits vs the hard GT-argmax target, weighted toward the
    decision boundary (small top-2 margin) where the flips actually live. This is the
    differentiable surrogate for the argmax-cell inequality; its gradient through the
    fixed decoder IS the Gauss-Newton step that projects the latent toward the cell.
    logits (B,5,384,512); target (B,384,512) long. Returns scalar."""
    ce = F.cross_entropy(logits, target, reduction="none")  # (B,384,512)
    # boundary weight: emphasize where the model is near a flip (top1-top2 margin small)
    top2 = logits.topk(2, dim=1).values  # (B,2,384,512)
    margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)  # (B,384,512)
    w = torch.exp(-margin / 2.0)  # near-boundary -> ~1, deep-interior -> ~0
    return (ce * (0.25 + w)).mean()


def latent_solve(
    decoder: torch.nn.Module,
    segnet: torch.nn.Module,
    z0: torch.Tensor,
    seg_target: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> tuple[torch.Tensor, dict]:
    """SOLVE the per-pair 28-dim latent (decoder FIXED) to land the SegNet argmax in the
    GT cell, through the exact eval chain. Tracks the EXACT argmax-flip d_seg (not the
    surrogate) and returns the best-by-exact-d_seg latent. z0 (B,28); seg_target (B,384,512).
    """
    z = z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    best_z = z0.detach().clone()
    best_dseg = exact_dseg_per_pair(decoder, segnet, z0, seg_target)  # (B,)
    history = []
    for step in range(steps):
        opt.zero_grad()
        seg_in = render_f1_segnet_input(
            decoder, z, roundtrip_uint8=True, differentiable=True
        )
        logits = segnet(seg_in)
        loss = boundary_weighted_ce(logits, seg_target)
        loss.backward()
        opt.step()
        # track exact d_seg; keep per-pair best
        cur = exact_dseg_per_pair(decoder, segnet, z, seg_target)  # (B,)
        improved = cur < best_dseg
        best_z[improved] = z.detach()[improved]
        best_dseg = torch.minimum(best_dseg, cur)
        history.append(
            {"step": step, "loss": float(loss.item()),
             "mean_exact_dseg": float(cur.mean().item()),
             "best_mean_exact_dseg": float(best_dseg.mean().item())}
        )
    return best_z, {"best_dseg": best_dseg, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--out", type=str,
                    default="experiments/results/probe_seed_solve_dseg_20260617.json")
    args = ap.parse_args()

    os.environ.setdefault("COMMA_CHALLENGE_ROOT", os.path.abspath("upstream"))
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(int(args.threads))

    # sys.path for the vendored frame_utils (yuv420_to_rgb) + av
    from tac.torch_vehicle.vendored_imports import import_vendored
    import_vendored("data")  # sets sys.path + COMMA_CHALLENGE_ROOT, applies yuv6 patch

    from tac.score_aware_loop.targets import load_frozen_distortion_net
    from tac.torch_vehicle.configurable_taper_decoder import (
        ConfigurableTaperHNeRVDecoder, vendored_taper,
    )

    t0 = time.time()
    dnet = load_frozen_distortion_net(device="cpu")
    segnet = dnet.segnet.eval()
    for p in segnet.parameters():
        p.requires_grad_(False)
    print(f"[load] frozen net in {time.time()-t0:.1f}s", flush=True)

    dec = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=vendored_taper(20)
    )
    sd = torch.load(BASIN / "best_ema_decoder.pt", map_location="cpu")
    dec.load_state_dict(sd, strict=True)
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    latents = torch.load(BASIN / "best_ema_latents.pt", map_location="cpu")  # (600,28)

    video_path = Path("upstream/videos/0.mkv")
    n = int(args.n_pairs)
    t1 = time.time()
    seg_target = build_seg_targets(video_path, segnet, n)  # (n,384,512)
    print(f"[targets] {n} GT seg targets in {time.time()-t1:.1f}s "
          f"(={(time.time()-t1)/n:.2f}s/pair)", flush=True)

    z0 = latents[:n].clone()

    # ---- BASELINE: trained latents, exact d_seg through the full roundtrip ----
    t2 = time.time()
    base_dseg = exact_dseg_per_pair(dec, segnet, z0, seg_target)  # (n,)
    base_eval_s = time.time() - t2
    print(f"[baseline] trained-latent exact d_seg mean={base_dseg.mean().item():.6f} "
          f"(per-pair {[round(float(x),5) for x in base_dseg]}) in {base_eval_s:.1f}s",
          flush=True)

    # ---- LATENT-SOLVE: decoder fixed, solve the 28-dim latent ----
    t3 = time.time()
    solved_z, info = latent_solve(
        dec, segnet, z0, seg_target, steps=int(args.steps), lr=float(args.lr)
    )
    solve_s = time.time() - t3
    solved_dseg = info["best_dseg"]  # (n,)
    print(f"[latent-solve] best exact d_seg mean={solved_dseg.mean().item():.6f} "
          f"(per-pair {[round(float(x),5) for x in solved_dseg]}) in {solve_s:.1f}s "
          f"(={solve_s/n:.2f}s/pair, {solve_s/max(1,args.steps):.2f}s/step)", flush=True)

    # ---- expressiveness ceiling: how many flips did the 28-dim solve fix? ----
    base_total = float(base_dseg.mean().item())
    solved_total = float(solved_dseg.mean().item())
    reduction = (base_total - solved_total) / max(base_total, 1e-12)

    out = {
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE — exact frozen SegNet, no MPS",
        "n_pairs": n, "steps": int(args.steps), "lr": float(args.lr),
        "baseline_mean_dseg": base_total,
        "baseline_per_pair_dseg": [float(x) for x in base_dseg],
        "baseline_eval_seconds": base_eval_s,
        "latent_solve_mean_dseg": solved_total,
        "latent_solve_per_pair_dseg": [float(x) for x in solved_dseg],
        "latent_solve_seconds": solve_s,
        "latent_solve_seconds_per_pair": solve_s / n,
        "latent_solve_seconds_per_step": solve_s / max(1, args.steps),
        "dseg_relative_reduction": reduction,
        "trained_basin_full_dseg": 0.002600919948890805,
        "solve_history_pair_mean": info["history"],
        "wall_clock_total_s": time.time() - t0,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"[verdict] baseline {base_total:.6f} -> latent-solve {solved_total:.6f} "
          f"({reduction*100:+.1f}% rel) | wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
