#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""$0 BINDINGNESS probe for the ARM-C #524 Lane stride-2 skip-band lever.

QUESTION (charter): does enabling the lever change the skip-path-visible energy objective on a
CACHED checkpoint's REAL render — i.e. does the term BIND (nonzero value + nonzero gradient on
the real inputs) rather than sit counted-but-inert? This is a BINDINGNESS proof, NOT a d_seg
claim (the d_seg effect is RUN-GATED; duty-to-measure A/B).

Mechanism: regenerate the witness's REAL frames from a prior run's EMA deploy checkpoint via
the CANONICAL numpy-fp32 oracle (``numpy_oracle_reference_frames`` on the byte-closed +
dequantized blob — the SAME weights the shipped inflate renders), resize f1 to the SegNet-input
domain (the exact surface the training lever reads), and evaluate:

  * ``term_on``    — the lever's loss value on the real render (numpy reference, bit-matching
                     the MLX branch); the OFF path is structurally 0 (the trainer branch is
                     gated on ``skipband_w > 0``), so term_on != 0 <=> enabling changes L.
  * ``grad``       — the CLOSED-FORM gradient of the term w.r.t. the render luma
                     (``skipband_term_grad_np``): nonzero in-band norm <=> enabling changes
                     dL/d(render) (the training gradient) on the real inputs.
  * ``sb_energy``  — witness vs GT skip-band energy on the lane band (the DIAGNOSTIC deficit
                     the lever supervises; the fractal memo §5 predicts the witness under-
                     carries Lane skip-band detail).

Axis: [macOS-CPU advisory] numpy-fp32 oracle on real cached weights + real GT; research_only;
score_claim=false. Pair subset stated in the output (bindingness needs existence, not n600).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", type=Path, required=True,
                    help="prior run dir with levelset_witness_ema_mlx.npz (READ-ONLY)")
    ap.add_argument("--gt-cache", type=Path, required=True,
                    help="GT npz (gt_f1 camera frames + lstars), e.g. gt_n96.npz")
    ap.add_argument("--n-pairs", type=int, default=24,
                    help="pair subset for the bindingness proof (<=96; stated in output)")
    ap.add_argument("--dilate", type=int, default=2, help="lane-band dilation (trainer default)")
    ap.add_argument("--out-json", type=Path, required=True, help="durable output (never /tmp)")
    args = ap.parse_args(argv)

    if str(args.out_json).startswith("/tmp"):
        raise SystemExit("--out-json must be durable (never /tmp) per CLAUDE.md")

    import levelset_byte_close_and_eval as bce
    from tac.boundary_math.lane_skipband import (
        lane_band_mask_half,
        luma_bt601,
        skip_band_detail,
        skipband_term_grad_np,
        skipband_term_np,
    )
    from tac.optimization.frame1_seg_safe_pose_atoms import _resize_map

    params, cfg = bce._load_levelset_ckpt(args.ckpt_dir, None)
    so = bce.detect_self_orient(cfg, {"freq_across": 32.0, "freq_along": 4.0, "tau": 4.0, "iters": 4})
    blob, _binfo = bce.build_levelset_blob(params, cfg, so, pose_sidecar=None)
    manifest, dq_params, code, lane_pairs, _pc = bce._dequant_blob(blob)

    gt = np.load(args.gt_cache, allow_pickle=False)
    n = min(int(args.n_pairs), int(gt["n_pairs"]), int(cfg["n_pairs"]))
    print(json.dumps({"stage": "render", "n_pairs": n,
                      "note": "numpy-fp32 oracle on the byte-closed dequantized weights"}), flush=True)
    frames, _argmax = bce.numpy_oracle_reference_frames(dq_params, code, manifest, n,
                                                        lane_pairs=lane_pairs)

    seg_h, seg_w = np.asarray(gt["lstars"][0]).shape                    # (384, 512)
    rows = []
    for pi in range(n):
        wit_f1_cam = np.asarray(frames[2 * pi + 1], np.float32)          # (874,1164,3) witness f1
        gt_f1_cam = np.asarray(gt["gt_f1"][pi], np.float32)
        wit_rs = np.stack([_resize_map(wit_f1_cam[:, :, c], seg_h, seg_w) for c in range(3)], -1)
        gt_rs = np.stack([_resize_map(gt_f1_cam[:, :, c], seg_h, seg_w) for c in range(3)], -1)
        lstar = np.asarray(gt["lstars"][pi])
        sb_gt = skip_band_detail(luma_bt601(gt_rs))
        mask = lane_band_mask_half(lstar, dilate=int(args.dilate), lane_class=1)
        band_px = float(mask.sum())
        if band_px == 0:
            rows.append({"pair": pi, "band_px": 0, "note": "no lane band in this frame"})
            continue
        term = skipband_term_np(wit_rs, sb_gt, mask)
        g = skipband_term_grad_np(wit_rs, sb_gt, mask)
        sb_wit = skip_band_detail(luma_bt601(wit_rs))
        e_wit = float(np.sum(mask * sb_wit ** 2) / band_px)
        e_gt = float(np.sum(mask * sb_gt ** 2) / band_px)
        # in-band vs out-of-band gradient mass (locality: the term only pulls the lane band)
        band_full = np.repeat(np.repeat(mask, 2, 0), 2, 1)               # full-res support (+halo)
        g_in = float(np.sqrt(np.sum((g * band_full) ** 2)))
        g_all = float(np.sqrt(np.sum(g ** 2)))
        rows.append({"pair": pi, "band_px": int(band_px), "term_on": term, "term_off": 0.0,
                     "grad_l2": g_all, "grad_l2_in_band": g_in, "grad_max_abs": float(np.max(np.abs(g))),
                     "sb_energy_witness": e_wit, "sb_energy_gt": e_gt,
                     "sb_energy_deficit_ratio": (e_wit / e_gt if e_gt > 0 else None)})

    scored = [r for r in rows if r.get("band_px", 0) > 0]
    binds = bool(scored) and all(r["term_on"] > 0 and r["grad_l2"] > 0 for r in scored)
    summary = {
        "schema": "lane_skipband_bindingness.v1",
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ckpt_dir": str(args.ckpt_dir), "npz": cfg.get("npz_name"),
        "gt_cache": str(args.gt_cache), "n_pairs_scored": len(scored),
        "axis": "[macOS-CPU advisory] numpy-fp32 oracle; research_only; score_claim=false",
        "verdict_scope": (f"BINDINGNESS on a {len(scored)}-pair real-render subset — existence "
                          "proof of a live term+gradient, NOT a d_seg claim, NOT n600 evidence"),
        "binds_when_enabled": binds,
        "term_on_mean": float(np.mean([r["term_on"] for r in scored])) if scored else None,
        "grad_l2_mean": float(np.mean([r["grad_l2"] for r in scored])) if scored else None,
        "sb_energy_witness_mean": float(np.mean([r["sb_energy_witness"] for r in scored])) if scored else None,
        "sb_energy_gt_mean": float(np.mean([r["sb_energy_gt"] for r in scored])) if scored else None,
        "per_pair": rows,
        "note": ("term_off is structurally 0 (the trainer branch is gated on skipband_w>0 => "
                 "byte-identical OFF path); binds_when_enabled == (term_on>0 AND grad_l2>0 on "
                 "every scored pair)"),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("binds_when_enabled", "n_pairs_scored",
                                              "term_on_mean", "grad_l2_mean",
                                              "sb_energy_witness_mean", "sb_energy_gt_mean")}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
