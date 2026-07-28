#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_da1 D2 -- frame_0 2.7 MB ROLE decomposition: the A7 coupling curve fc1 named but did not fire.

frame_0 has TWO consumers: (1) PoseNet fidelity (pose-read frame), (2) copy-base for frame_1 seg.
At WebP Q in {1,5,10,20} (method 6), MEASURE all three:
  (a) SUPPORT growth: SegNet(copy(crushed f0)) argmax vs GT f1 lstars -> flip count / d_seg. (real SegNet)
  (b) d_pose: MSE(PoseNet(crushed_f0, gt_f1)[:6], PoseNet(gt_f0, gt_f1)[:6]) -- the pose collateral of
      crushing ONLY frame_0, using the real f1 for the 2nd frame (bounds it; real gen_f1 differs 2nd-order).
  (c) range(A) split: energy of the crush error (crushed_f0 - gt_f0) INSIDE vs OUTSIDE range(A) -- how
      much of the frame_0 bytes pay for scorer-invisible (ker A) detail.
  + webp bytes per Q (the composed rate term).

Reads gt_f0/gt_f1 (camera-res) mmap + lstars. ONE scorer job at a time. Resumable per-Q (partial JSON).
`[macOS-CPU advisory]` -- real WebP + frozen SegNet/PoseNet; NOT a byte-closed evaluate.py row.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA = "ddm_da1_d2_frame0_coupling.v1"
COPY_UNCRUSHED_DSEG = 0.00864212883843316  # baseline: copy on UNCRUSHED f0 (self-check at Q=inf)


def _load_segnet(repo_root: Path, weights: Path):
    sys.path.insert(0, str(repo_root / "upstream"))
    import torch
    from modules import SegNet
    from safetensors.torch import load_file
    net = SegNet().eval()
    net.load_state_dict(load_file(str(weights)))
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 8)))
    return net


def _load_posenet(repo_root: Path, weights: Path):
    sys.path.insert(0, str(repo_root / "upstream"))
    import torch
    from modules import PoseNet
    from safetensors.torch import load_file
    net = PoseNet().eval()
    net.load_state_dict(load_file(str(weights)))
    torch.set_grad_enabled(False)
    return net


def _segnet_argmax(net, frames_bhwc_uint8):
    import einops
    import torch
    x = torch.from_numpy(frames_bhwc_uint8).float()[:, None]  # (B,1,H,W,C)
    x = einops.rearrange(x, "b t h w c -> b t c h w")
    with torch.inference_mode():
        inp = net.preprocess_input(x)
        out = net(inp)
        return out.argmax(dim=1).cpu().numpy()


def _posenet_pose6(net, pair_bt_hwc_uint8):
    """pair_bt_hwc_uint8: (B,2,H,W,C) uint8 -> (B,6) first-6 pose head outputs."""
    import einops
    import torch
    x = torch.from_numpy(pair_bt_hwc_uint8).float()  # (B,2,H,W,C)
    x = einops.rearrange(x, "b t h w c -> b t c h w")
    with torch.inference_mode():
        inp = net.preprocess_input(x)
        out = net(inp)  # dict with 'pose'
        pose = out["pose"]
        return pose[..., : pose.shape[-1] // 2].cpu().numpy()  # (B,6)


def _webp_roundtrip(frame_hwc, quality, method):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(frame_hwc).save(b, "WEBP", quality=quality, method=method)
    nbytes = b.tell()
    b.seek(0)
    dec = np.asarray(Image.open(b).convert("RGB"), dtype=np.uint8)
    return dec, nbytes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-frames", type=Path, required=True, help="gt_n600.npz with gt_f0/gt_f1/lstars")
    ap.add_argument("--repo-root", type=Path, default=Path("/Users/adpena/Projects/pact"))
    ap.add_argument("--segnet-weights", type=Path, required=True)
    ap.add_argument("--posenet-weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--qualities", type=str, default="1,5,10,20")
    ap.add_argument("--method", type=int, default=6)
    ap.add_argument("--max-pairs", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=50)
    args = ap.parse_args(argv)
    t0 = time.time()

    from tac.boundary_math.range_a_projection import apply_projection

    gt = np.load(str(args.gt_frames), mmap_mode="r")
    lstars = gt["lstars"]  # (600,384,512) int64
    P = min(args.max_pairs, gt["gt_f0"].shape[0])
    qualities = [int(q) for q in args.qualities.split(",")]

    seg = _load_segnet(args.repo_root, args.segnet_weights)
    pose = _load_posenet(args.repo_root, args.posenet_weights)

    # reference pose on GT pair (per-pair pose6), computed once
    print(f"[d2] computing GT-pair pose reference (P={P})...", flush=True)
    pose_gt = np.zeros((P, 6), dtype=np.float64)
    for s in range(0, P, args.chunk):
        e = min(s + args.chunk, P)
        f0 = np.asarray(gt["gt_f0"][s:e])
        f1 = np.asarray(gt["gt_f1"][s:e])
        pair = np.stack([f0, f1], axis=1)  # (b,2,H,W,C)
        pose_gt[s:e] = _posenet_pose6(pose, pair)
    print(f"[d2] pose_gt done ({time.time()-t0:.0f}s)", flush=True)

    # resume: load existing partial out
    results = {}
    if args.out.exists():
        try:
            prev = json.loads(args.out.read_text())
            results = prev.get("per_quality", {})
        except Exception:
            results = {}

    total_sites = P * lstars.shape[1] * lstars.shape[2]

    for q in qualities:
        key = str(q)
        if key in results:
            print(f"[d2] Q={q} already done, skip", flush=True)
            continue
        flips = 0
        webp_bytes = 0
        dpose_sum = 0.0
        e_in = 0.0   # range(A) energy of crush error
        e_out = 0.0  # ker(A) energy
        e_tot = 0.0
        for s in range(0, P, args.chunk):
            e = min(s + args.chunk, P)
            f0 = np.asarray(gt["gt_f0"][s:e]).astype(np.uint8)
            f1 = np.asarray(gt["gt_f1"][s:e]).astype(np.uint8)
            # crush f0
            crushed = np.empty_like(f0)
            for i in range(f0.shape[0]):
                dec, nb = _webp_roundtrip(f0[i], q, args.method)
                crushed[i] = dec
                webp_bytes += nb
            # (a) support growth: SegNet on copy(crushed f0) vs lstars
            am = _segnet_argmax(seg, crushed)
            gt_ls = np.asarray(lstars[s:e])
            flips += int((am != gt_ls).sum())
            # (b) d_pose: PoseNet(crushed_f0, gt_f1) vs pose_gt
            pair_gen = np.stack([crushed, f1], axis=1)
            pg = _posenet_pose6(pose, pair_gen)
            dpose_sum += float(((pg - pose_gt[s:e]) ** 2).mean(axis=1).sum())
            # (c) range(A) split of crush error
            err = crushed.astype(np.float64) - f0.astype(np.float64)  # (b,H,W,C)
            perr = apply_projection(err, out_dtype=np.float64, compute_dtype=np.float64)
            e_in += float((perr ** 2).sum())
            e_out += float(((err - perr) ** 2).sum())
            e_tot += float((err ** 2).sum())
            print(f"[d2] Q={q} pairs {s}-{e} ({time.time()-t0:.0f}s) flips={flips}", flush=True)
        d_seg = flips / total_sites
        d_pose = dpose_sum / P
        results[key] = {
            "webp_quality": q,
            "webp_method": args.method,
            "webp_bytes_total": webp_bytes,
            "webp_bytes_per_frame": webp_bytes / P,
            "S_rate_term_frame0": 25.0 * webp_bytes / 37_545_489,
            "flip_sites": flips,
            "d_seg_support": d_seg,
            "d_seg_vs_uncrushed_copy": d_seg / COPY_UNCRUSHED_DSEG,
            "d_pose": d_pose,
            "pose_term_sqrt10": float((10.0 * d_pose) ** 0.5),
            "rangeA_energy_inside": e_in,
            "kerA_energy_outside": e_out,
            "total_crush_err_energy": e_tot,
            "frac_scorer_invisible_kerA": e_out / max(1e-12, e_tot),
        }
        # write partial after each Q (resumable)
        out = {
            "schema": SCHEMA,
            "evidence_axis": "[macOS-CPU advisory] REAL WebP + frozen SegNet/PoseNet on cached GT frames; NOT a byte-closed evaluate.py row",
            "pairs": P,
            "copy_uncrushed_dseg_reference": COPY_UNCRUSHED_DSEG,
            "note_pose": "d_pose = collateral of crushing ONLY frame_0 (PoseNet(crushed_f0,gt_f1) vs PoseNet(gt_f0,gt_f1)); read through pose_plane_proximity law; cb1 warns repaint pose sign is class-dependent (Lane +22.7 / MyCar -0.18)",
            "per_quality": results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"[d2] Q={q} DONE d_seg={d_seg:.6f} ({d_seg/COPY_UNCRUSHED_DSEG:.2f}x uncrushed) d_pose={d_pose:.6f} kerA_invis={results[key]['frac_scorer_invisible_kerA']:.3f} bytes={webp_bytes} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[d2] ALL DONE ({time.time()-t0:.0f}s) -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
