#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""REAL-LOSS per-step MPS PoseNet drift diagnostic (H_opbias test).

The prior single-step diag (``diag_mps_posenet_drift.py``) found the MPS PoseNet
per-step forward + input-gradient near-identical (cos 1.0, relmax ~2e-4) — BUT it
used a PROXY loss (sum of pose^2), RANDOM input, B=4. The caveat: the real basin
loss is ``seg_weight*CE(seg) + pose_weight*sqrt(10*MSE(pose6))`` on REAL frames
near the operating point. This diag closes that gap: it loads the TRAINED base_ch=20
decoder (from the prior MPS A/B's CPU arm — 30 epochs at the real operating point),
renders the REAL frames, applies the exact PR95 eval-roundtrip, and measures the
per-step INPUT-GRADIENT dL/dF (the exact surface where the training drift was
measured) CPU vs MPS, under:

  (1) the FULL basin loss (seg_weight=100*CE + pose_weight=1*sqrt(10*MSE)),
  (2) the SEG-only loss (the path validated bit-identical),
  (3) the POSE-only loss (the suspected drift path).

If the POSE-only and FULL dL/dF relmax on REAL frames is still ~2e-4 (the proxy
diag's number) -> the per-step gradient is essentially correct under the real loss
too -> the training divergence is CHAOS, not per-step bias (H_chaos). If the
real-loss relmax is MATERIALLY worse (e.g. >1e-2) -> a real bias (H_opbias);
report which head carries it.

Authority: [macOS-CPU advisory], $0, correctness-only (single-step, contention-
immune). NON-PROMOTABLE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, "upstream")

from tac.torch_mps_compat import patch_scorer_for_mps  # noqa: E402
from tac.score_aware_loop.targets import load_frozen_distortion_net  # noqa: E402
from tac.torch_vehicle.driver import import_vendored_bundle  # noqa: E402

from tac.torch_vehicle.driver import _EVAL_H, _EVAL_W  # noqa: E402  vehicle decoder eval size (384, 512)


def _cos(a, b):
    a = a.detach().float().cpu().flatten()
    b = b.detach().float().cpu().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _relmax(a, b):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    denom = a.abs().max().clamp_min(1e-8)
    return (a - b).abs().max().item() / denom.item()


def _build_decoder(state, base_channels, latent_dim, device):
    b = import_vendored_bundle()
    dec = b.HNeRVDecoder(latent_dim=latent_dim, base_channels=base_channels,
                         eval_size=(_EVAL_H, _EVAL_W)).to(device)
    dec.load_state_dict({k: v.to(device) for k, v in state["decoder"].items()})
    dec.eval()
    return dec


def _roundtrip(decoded_pair, B, device):
    """PR95 eval-roundtrip: bicubic-up 874x1164 -> bilinear-down 384x512 -> STE round,
    matching driver._train_one_epoch lines 348-355."""
    flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    clamped = bhwc.clamp(0, 255)
    rounded = clamped.round()
    return clamped + (rounded - clamped).detach()


def _frame_grad(net, frames_leaf, seg_targets, pose_targets, *, mode, seg_w, pose_w):
    """Backprop the chosen loss to frames_leaf.grad (the dL/dF cotangent)."""
    posenet_in, segnet_in = net.preprocess_input(frames_leaf)
    seg_out = net.segnet(segnet_in)
    pose_out = net.posenet(posenet_in)["pose"][:, :6]
    seg_l = F.cross_entropy(seg_out, seg_targets)
    pose_mse = F.mse_loss(pose_out, pose_targets)
    pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
    if mode == "full":
        loss = seg_w * seg_l + pose_w * pose_l
    elif mode == "seg":
        loss = seg_w * seg_l
    elif mode == "pose":
        loss = pose_w * pose_l
    else:
        raise ValueError(mode)
    seg_l_v, pose_mse_v, pose_l_v = float(seg_l.detach()), float(pose_mse.detach()), float(pose_l.detach())
    loss.backward()
    return frames_leaf.grad.detach().clone(), seg_l_v, pose_mse_v, pose_l_v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state",
                    default="experiments/results/torch_vehicle_mps_descent_ab/arm_cpu/torch_vehicle_checkpoint_state.pt")
    ap.add_argument("--targets",
                    default="experiments/results/torch_vehicle_mps_descent_ab/gt_targets_cache/gt_targets_n48.pt")
    ap.add_argument("--base-channels", type=int, default=20)
    ap.add_argument("--latent-dim", type=int, default=28)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seg-weight", type=float, default=100.0)
    ap.add_argument("--pose-weight", type=float, default=1.0)
    ap.add_argument("--out-json",
                    default="experiments/results/torch_vehicle_chaos_control_ab/real_loss_diag.json")
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        print("[FATAL] torch-MPS unavailable — use a Mac.")
        return 2
    patch_scorer_for_mps()
    torch.manual_seed(0)

    state = torch.load(args.state, map_location="cpu", weights_only=False)
    tgt = torch.load(args.targets, map_location="cpu", weights_only=False)
    latents = state["latents"]                       # (48, 28) trained
    seg_targets_all = tgt["seg"].long()              # (48, 384, 512)
    pose_targets_all = tgt["pose"].float()           # (48, 6)
    B = min(args.batch_size, latents.shape[0])
    idx = torch.arange(B)

    net_cpu = load_frozen_distortion_net(device="cpu")
    net_mps = load_frozen_distortion_net(device="mps")
    dec_cpu = _build_decoder(state, args.base_channels, args.latent_dim, "cpu")
    dec_mps = _build_decoder(state, args.base_channels, args.latent_dim, "mps")

    # Render the REAL frames at the trained operating point, on each device.
    with torch.no_grad():
        decoded_cpu = dec_cpu(latents[idx].to("cpu"))
        frames_cpu_val = _roundtrip(decoded_cpu, B, "cpu").detach()
        decoded_mps = dec_mps(latents[idx].to("mps"))
        frames_mps_val = _roundtrip(decoded_mps, B, "mps").detach()
    torch.mps.synchronize()

    print("=== render parity (decoder forward, REAL frames) ===")
    print(f"  frames cos={_cos(frames_cpu_val, frames_mps_val):.6f}  "
          f"relmax={_relmax(frames_cpu_val, frames_mps_val):.4e}")

    seg_t_cpu = seg_targets_all[idx].to("cpu")
    pose_t_cpu = pose_targets_all[idx].to("cpu")
    seg_t_mps = seg_targets_all[idx].to("mps")
    pose_t_mps = pose_targets_all[idx].to("mps")

    rows = {}
    for mode in ("full", "seg", "pose"):
        # CPU authority cotangent (use the SAME CPU frame values on both so we
        # isolate the SCORER-backend drift, not the decoder render drift).
        fc = frames_cpu_val.clone().requires_grad_(True)
        gc, segc, posemc, poselc = _frame_grad(
            net_cpu, fc, seg_t_cpu, pose_t_cpu, mode=mode,
            seg_w=args.seg_weight, pose_w=args.pose_weight)
        # MPS cotangent on the SAME (CPU-rendered) frame values moved to MPS — so
        # any difference is the MPS SCORER gradient, not the MPS decoder render.
        fm = frames_cpu_val.clone().to("mps").requires_grad_(True)
        gm, segm, posemm, poselm = _frame_grad(
            net_mps, fm, seg_t_mps, pose_t_mps, mode=mode,
            seg_w=args.seg_weight, pose_w=args.pose_weight)
        torch.mps.synchronize()
        cos = _cos(gc, gm)
        rmax = _relmax(gc, gm)
        rows[mode] = {"cos": cos, "relmax": rmax,
                      "cpu_seg_l": segc, "cpu_pose_mse": posemc, "cpu_pose_l": poselc,
                      "grad_norm_cpu": float(gc.float().norm()),
                      "grad_norm_mps": float(gm.float().norm())}
        print(f"=== dL/dF [{mode:>4}] loss (REAL frames, same CPU pixels) ===")
        print(f"  cos={cos:.6f}  relmax={rmax:.4e}  "
              f"|g|_cpu={rows[mode]['grad_norm_cpu']:.4e} |g|_mps={rows[mode]['grad_norm_mps']:.4e}")

    full_relmax = rows["full"]["relmax"]
    pose_relmax = rows["pose"]["relmax"]
    seg_relmax = rows["seg"]["relmax"]
    verdict = (
        "H_chaos SUPPORTED (per-step side): the REAL-loss dL/dF relmax is small "
        f"(full={full_relmax:.2e}, pose={pose_relmax:.2e}) — consistent with the proxy "
        "diag's ~2e-4; the per-step MPS gradient is essentially correct under the real "
        "loss, so the training gap is CHAOS not per-step bias."
        if (full_relmax < 1e-2 and pose_relmax < 1e-2)
        else f"H_opbias INDICATED: REAL-loss dL/dF relmax is large (full={full_relmax:.2e}, "
             f"pose={pose_relmax:.2e}, seg={seg_relmax:.2e}) — a real per-step bias; "
             "the dominant head is the patch target."
    )
    print(f"\n>>> {verdict}")

    out = {"config": vars(args), "render_cos": _cos(frames_cpu_val, frames_mps_val),
           "render_relmax": _relmax(frames_cpu_val, frames_mps_val),
           "per_mode": rows, "verdict": verdict,
           "authority": "[macOS-CPU advisory] single-step correctness; NON-PROMOTABLE."}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
