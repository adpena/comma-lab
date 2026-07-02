# SPDX-License-Identifier: MIT
"""VCM TEMPORAL KEYFRAME-STREAM CODING — the proper (non-toy) MPEG/HEVC/AV1 revisit.

Adversarial overturn (operator 2026-07-02) of the anti-MPEG negative: the rate track
ALREADY measured temporal HEVC ~4x cheaper than per-frame WebP. This tool does it
PROPERLY on the REAL keyframe stream: real x265 / SVT-AV1 / VP9 at proper GOP + quality,
temporal (I+P[+B]) coding of the CORRELATED keyframe stream, vs the per-frame still
baseline. Grounded in MPEG-AI Part 2 "Video Coding for Machines" (ISO/IEC 23888-2, DIS
2026): code the pixel stream FOR the machine task (PoseNet), guided by d_pose not PSNR.

Two axes:
  RATE (fast, no scorer): temporal-stream bytes for {13-keyframe, 40-keyframe} streams
    across codec x CRF x GOP x B-frames, + still (WebP/AVIF/J2K) baseline -> rate table.
  D_POSE (opt --measure-dpose, 13 keyframes only): decode the stream, warp each decoded
    keyframe by its ego-twist, measure d_pose through frozen CPU-torch PoseNet (NEVER MPS)
    -> the RD point (temporal-stream rate vs machine-task distortion at the keyframes).

``[macOS-CPU advisory]`` ONLY; pointer 0.19110 UNMOVED; score_claim=False. Byte numbers
are real codecs on real ``gt_f0`` frames; the decoder libs (libx265/libsvtav1/libvpx) are
FREE (rule-118, like brotli) but the STREAM bytes are COUNTED in archive.zip.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--n-pairs", type=int, default=0)
    ap.add_argument("--strides", default="47,15", help="47 -> 13 kf (reach); 15 -> 40 kf (short-reach)")
    ap.add_argument("--resolution", default="384x512", help="keyframe store resolution WxH (or 'native')")
    ap.add_argument("--codecs", default="x265,svtav1,vp9")
    ap.add_argument("--crfs", default="28,34,40,46,52")
    ap.add_argument("--gop", type=int, default=9999, help="9999 => single-GOP (1 I-frame, rest inter)")
    ap.add_argument("--bframes", type=int, default=2)
    ap.add_argument("--measure-dpose", action="store_true", help="decode + d_pose on the 13 keyframes")
    ap.add_argument("--s-t", type=float, default=0.044)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from tac.boundary_math import keyframe_codec as kc

    t0 = time.time()
    cache = (REPO / args.cache) if not Path(args.cache).is_absolute() else Path(args.cache)
    z = np.load(cache, allow_pickle=False)
    gt_f0 = np.asarray(z["gt_f0"])
    poses = np.asarray(z["gt_poses"], dtype=np.float64)
    P = gt_f0.shape[0] if not args.n_pairs else min(args.n_pairs, gt_f0.shape[0])
    NAT_H, NAT_W = gt_f0.shape[1], gt_f0.shape[2]

    if args.resolution == "native":
        rh, rw = NAT_H, NAT_W
    else:
        w, h = args.resolution.split("x")
        rh, rw = int(h), int(w)

    posenet = geom = None
    if args.measure_dpose:
        import torch
        from modules import DistortionNet, posenet_sd_path, segnet_sd_path

        from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom
        dn = DistortionNet().eval()
        dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
        posenet = dn.posenet
        for p_ in posenet.parameters():
            p_.requires_grad = False
        geom = GroundHomographyGeom.eon(native_hw=(NAT_H, NAT_W), pitch=0.0)

    def _dpose_on_decoded(decoded_native_kfs, kf_idx):
        from experiments.train_witness_realized_through_R_mlx import cpu_verdict_d_pose_batch
        from tac.boundary_math.warp_real_luma_frame0 import (
            warp_frame0_uint8_numpy, xi_from_pose_calibration,
        )
        f0s, f1s, gps = [], [], []
        for src, p in zip(decoded_native_kfs, kf_idx):
            xi = xi_from_pose_calibration(poses[p], args.s_t, 0.0, 0.0, whole_ground=True)
            f0s.append(src); f1s.append(warp_frame0_uint8_numpy(src, xi, geom)); gps.append(poses[p])
        return float(np.mean(cpu_verdict_d_pose_batch(posenet, f0s, f1s, gps)))

    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    crfs = [int(c) for c in args.crfs.split(",") if c.strip()]
    results = {}
    for stride in [int(s) for s in args.strides.split(",")]:
        kf_idx = list(range(0, P, stride))
        n = len(kf_idx)
        # native keyframes -> store resolution (downsample), keep native copy for warp/d_pose.
        kfs_store = [kc.downsample_only(gt_f0[i], rw, rh) if (rh, rw) != (NAT_H, NAT_W)
                     else kc._as_u8_hwc(gt_f0[i]) for i in kf_idx]
        row = {"n_keyframes": n, "store_hw": [rh, rw], "temporal": {}, "still": {}}
        # ---- still baseline (per-frame) ----
        for still in ("webp", "avif", "j2k"):
            for q in (30, 50, 75):
                tot = 0
                ok = True
                for k in kfs_store:
                    if still == "webp":
                        b = kc.webp_bytes(k, q)
                    elif still == "avif":
                        b = kc.avif_bytes(k, q)
                    else:
                        b = kc.jpeg2000_bytes(k)
                    if b is None:
                        ok = False; break
                    tot += b
                if ok:
                    row["still"][f"{still}_q{q}"] = {"bytes": tot, "rate": round(kc.rate_from_bytes(tot), 5)}
                if still == "j2k":
                    break  # j2k quality fixed here
        # ---- temporal streams ----
        for codec in codecs:
            for crf in crfs:
                try:
                    if args.measure_dpose:
                        nb, dec = kc.encode_decode_video_stream(
                            kfs_store, codec=codec, crf=crf, gop=args.gop, bframes=args.bframes)
                        dec_native = [kc.resize_roundtrip(d, rw, rh) if (rh, rw) != (NAT_H, NAT_W)
                                      else d for d in dec]
                        # decoded store-res -> upsample to native for warp/PoseNet
                        dec_native = [kc.downsample_only(d, NAT_W, NAT_H, interp="cubic")
                                      if (rh, rw) != (NAT_H, NAT_W) else d for d in dec]
                        dp = _dpose_on_decoded(dec_native, kf_idx)
                    else:
                        nb = kc.encode_video_stream_bytes(
                            kfs_store, codec=codec, crf=crf, gop=args.gop, bframes=args.bframes)
                        dp = None
                    row["temporal"][f"{codec}_crf{crf}"] = {
                        "bytes": nb, "rate": round(kc.rate_from_bytes(nb), 5),
                        "d_pose_at_keyframes": (round(dp, 4) if dp is not None else None),
                        "pose_sqrt10": (round(float(np.sqrt(10 * dp)), 4) if dp is not None else None),
                    }
                    dpt = f" d_pose={dp:.3f}" if dp is not None else ""
                    print(f"  [stride{stride} n{n}] {codec}_crf{crf}: {nb} B  rate={kc.rate_from_bytes(nb):.5f}{dpt}", flush=True)
                except kc.KeyframeCodecError as e:
                    row["temporal"][f"{codec}_crf{crf}"] = {"error": str(e)[:200]}
                    print(f"  [stride{stride}] {codec}_crf{crf} FAILED: {str(e)[:120]}", flush=True)
        results[f"stride{stride}"] = row

    out = {
        "tool": "tools/measure_keyframe_vcm_rate.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS-CPU advisory / research-signal] (NOT a contest score)",
        "score_claim": False, "promotable": False, "frontier_pointer": "UNMOVED 0.19110",
        "grounding": "MPEG-AI Part 2 Video Coding for Machines (ISO/IEC 23888-2, DIS 2026): code FOR PoseNet, metric d_pose not PSNR",
        "n_pairs": P, "native_hw": [NAT_H, NAT_W], "store_hw": [rh, rw],
        "gop": args.gop, "bframes": args.bframes, "s_t": args.s_t,
        "results": results,
        "elapsed_secs": round(time.time() - t0, 1),
    }
    out_path = (Path(args.out) if args.out
                else (REPO / f"experiments/results/keyframe_vcm_rate_{args.resolution}/results.json"))
    _refuse_tmp(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[written] {out_path}  ({out['elapsed_secs']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
