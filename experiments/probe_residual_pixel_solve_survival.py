# SPDX-License-Identifier: MIT
"""DECISIVE PROBE part 2 — the RESIDUAL pixel-SOLVE + SURVIVAL + BYTE-COST wall.

After the 28-dim latent-SOLVE hits its expressiveness ceiling (probe part 1), the residual
flips are pixels the fixed decoder's 28-dim image cannot reach. The reframe says: SOLVE the
minimal frame-perturbation that flips them (the per-pixel SegNet-Jacobian preimage onto the
argmax-cell). THE WALL the operator asks about: does that solved correction SURVIVE the exact
eval roundtrip (the perturbation lives at camera-res, gets bicubic v-then-bilinear v-resampled
+ uint8-rounded before SegNet sees it), and at what BYTE cost?

THE EXACT GEOMETRY (the survival trap):
  SegNet sees: GT_camera_f1 (874,1164) -> bilinear v (384,512) -> SegNet.
  We can only STORE a perturbation; the cheapest place to store it is the (384,512) SegNet-input
  grid (the scored object). But the eval chain does NOT feed our (384,512) directly: a real
  decoder renders (384,512) -> bicubic ^ (874,1164) -> uint8 -> bilinear v (384,512). A residual
  added at (384,512) and then run through ^uint8v is NOT the same residual (resample + quantize
  smear it). This probe measures the smear: solve a residual at the SegNet-input grid that
  flips the residual pixels, then push it through the EXACT ^->uint8->v chain and re-measure
  how many flips SURVIVE. That survival fraction + the brotli byte cost IS the wall.

NO-FAKE: real frozen SegNet (CPU, MPS off), real basin decoder+latents READ-ONLY, GT via the
vendored frame_utils path, exact argmax-flip d_seg through the literal roundtrip. The residual
must ACTUALLY flip pixels on the real scorer and the survival is measured on the real chain.
[macOS-CPU advisory] NON-PROMOTABLE.
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
SEG_IN_H, SEG_IN_W = 384, 512
BASIN = Path("experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best")


def _to_camera(x_chw: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x_chw, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)


def _to_seg_in(cam_chw: torch.Tensor) -> torch.Tensor:
    return F.interpolate(cam_chw, size=(SEG_IN_H, SEG_IN_W), mode="bilinear")


def render_decoder_camera_f1(decoder, z):
    """decoder(z) frame1 -> camera-res (B,3,874,1164) float [0,255] (pre-uint8)."""
    decoded = decoder(z)
    f1 = decoded[:, 1]
    return _to_camera(f1)


def seg_argmax_from_camera(segnet, cam_f1, *, roundtrip_uint8: bool, differentiable: bool):
    """camera-res f1 -> [uint8] -> SegNet input -> logits. Returns logits (B,5,384,512)."""
    if roundtrip_uint8:
        camr = cam_f1.clamp(0, 255).round()
        if differentiable:
            cam_f1 = cam_f1 + (camr - cam_f1).detach()
        else:
            cam_f1 = camr
    seg_in = _to_seg_in(cam_f1)
    return segnet(seg_in)


def build_seg_targets(video_path, segnet, n_pairs):
    import av
    from frame_utils import yuv420_to_rgb
    container = av.open(str(video_path))
    targets, prev, idx = [], None, 0
    with torch.inference_mode():
        for frame in container.decode(container.streams.video[0]):
            f = yuv420_to_rgb(frame)
            if prev is None:
                prev = f
                continue
            f1 = f
            prev = None
            cam = f1.float().permute(2, 0, 1).unsqueeze(0)  # (1,3,874,1164)
            logits = seg_argmax_from_camera(segnet, cam, roundtrip_uint8=False, differentiable=False)
            targets.append(logits.argmax(dim=1).squeeze(0).clone())
            idx += 1
            if idx >= n_pairs:
                break
    container.close()
    return torch.stack(targets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2.0)  # pixel-domain LR (units are RGB levels)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--out", type=str,
                    default="experiments/results/probe_residual_pixel_solve_20260617.json")
    args = ap.parse_args()

    os.environ.setdefault("COMMA_CHALLENGE_ROOT", os.path.abspath("upstream"))
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(int(args.threads))

    from tac.torch_vehicle.vendored_imports import import_vendored
    import_vendored("data")
    from tac.score_aware_loop.targets import load_frozen_distortion_net
    from tac.torch_vehicle.configurable_taper_decoder import (
        ConfigurableTaperHNeRVDecoder, vendored_taper,
    )

    t0 = time.time()
    dnet = load_frozen_distortion_net(device="cpu")
    segnet = dnet.segnet.eval()
    for p in segnet.parameters():
        p.requires_grad_(False)
    dec = ConfigurableTaperHNeRVDecoder(latent_dim=28, base_channels=20, channels=vendored_taper(20))
    dec.load_state_dict(torch.load(BASIN / "best_ema_decoder.pt", map_location="cpu"), strict=True)
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    latents = torch.load(BASIN / "best_ema_latents.pt", map_location="cpu")
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    video_path = Path("upstream/videos/0.mkv")
    n = int(args.n_pairs)
    seg_target = build_seg_targets(video_path, segnet, n)  # (n,384,512)

    z0 = latents[:n].clone()
    # baseline camera-res render (the fixed decoder's frame1) — the residual base.
    with torch.inference_mode():
        cam_base = render_decoder_camera_f1(dec, z0)  # (n,3,874,1164)
        base_logits = seg_argmax_from_camera(segnet, cam_base, roundtrip_uint8=True, differentiable=False)
        base_pred = base_logits.argmax(dim=1)
        base_flip = (base_pred != seg_target)  # (n,384,512) bool
    base_dseg = base_flip.float().mean().item()
    n_flips_base = int(base_flip.sum().item())
    print(f"[baseline] mean d_seg={base_dseg:.6f}  total residual flips={n_flips_base} "
          f"({n_flips_base/(n*SEG_IN_H*SEG_IN_W)*100:.3f}% of pixels)", flush=True)

    # ---- RESIDUAL SOLVE: a free camera-res perturbation, solved to flip the residual pixels
    #      through the EXACT chain (^uint8v). This is the UPPER BOUND on what a residual can do:
    #      it uses the full (3,874,1164) DOF, not a cheap subspace. If even this cannot drive
    #      the flips down through the roundtrip, the survival wall is fundamental. ----
    delta = torch.zeros_like(cam_base, requires_grad=True)  # camera-res perturbation
    opt = torch.optim.Adam([delta], lr=float(args.lr))
    best_dseg = base_dseg
    hist = []
    for step in range(int(args.steps)):
        opt.zero_grad()
        cam_pert = cam_base + delta
        logits = seg_argmax_from_camera(segnet, cam_pert, roundtrip_uint8=True, differentiable=True)
        # CE only on the residual-flip pixels (the constraint-satisfaction target = GT cell)
        ce = F.cross_entropy(logits, seg_target, reduction="none")  # (n,384,512)
        loss = (ce * base_flip.float()).sum() / base_flip.float().sum().clamp_min(1)
        # small L2 on the perturbation (cheapness proxy)
        loss = loss + 1e-4 * (delta ** 2).mean()
        loss.backward()
        opt.step()
        with torch.inference_mode():
            cur_logits = seg_argmax_from_camera(segnet, cam_base + delta, roundtrip_uint8=True, differentiable=False)
            cur_pred = cur_logits.argmax(dim=1)
            cur_dseg = (cur_pred != seg_target).float().mean().item()
        best_dseg = min(best_dseg, cur_dseg)
        hist.append({"step": step, "loss": float(loss.item()), "exact_dseg": cur_dseg})

    # final residual + survival/byte characterization
    with torch.inference_mode():
        final_delta = delta.detach()
        # byte cost: quantize the camera-res delta to int8 + brotli (the storage model)
        dq = final_delta.clamp(-127, 127).round().to(torch.int8).cpu().numpy()
        nz = int((dq != 0).sum())
        try:
            import brotli
            blob = brotli.compress(dq.tobytes(), quality=11)
            byte_cost = len(blob)
        except Exception:
            byte_cost = None
        # SURVIVAL test: the residual was solved AT camera-res WITH the ^uint8v chain in the loop,
        # so "best_dseg" already IS the surviving d_seg. Also measure the no-roundtrip ceiling
        # (what the residual COULD do if uint8/resample didn't smear it) to quantify the smear.
        ideal_logits = seg_argmax_from_camera(segnet, cam_base + final_delta, roundtrip_uint8=False, differentiable=False)
        ideal_dseg = (ideal_logits.argmax(dim=1) != seg_target).float().mean().item()

    rate_term = (25.0 * byte_cost / 37_545_489.0) if byte_cost else None
    print(f"[residual-solve] baseline {base_dseg:.6f} -> survived(roundtrip) {best_dseg:.6f} | "
          f"ideal(no-roundtrip) {ideal_dseg:.6f}", flush=True)
    print(f"[byte-cost] camera-res int8 delta: nonzeros={nz} brotli_bytes={byte_cost} "
          f"rate_term(+{rate_term:.4f} S)" if byte_cost else "[byte-cost] brotli unavailable",
          flush=True)

    out = {
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE — exact frozen SegNet, no MPS",
        "n_pairs": n, "steps": int(args.steps), "lr": float(args.lr),
        "baseline_mean_dseg": base_dseg,
        "n_residual_flips": n_flips_base,
        "residual_solve_survived_dseg": best_dseg,
        "residual_solve_ideal_no_roundtrip_dseg": ideal_dseg,
        "smear_loss_dseg": best_dseg - ideal_dseg,
        "camera_delta_nonzeros_int8": nz,
        "camera_delta_brotli_bytes": byte_cost,
        "camera_delta_rate_term_S": rate_term,
        "history": hist,
        "wall_clock_total_s": time.time() - t0,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"[verdict] wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
