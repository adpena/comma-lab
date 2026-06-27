# SPDX-License-Identifier: MIT
"""THE GATE — does the 587x FIELD-LEVEL R-survival transfer to the REALIZED SegNet argmax?

The 587x was measured on ``argmax(R(phi))`` (field level). The contest scores
``SegNet(R(RGB(phi))).argmax`` — SegNet RE-segments the RGB, and the RGB passes through
softmax(phi/T)@palette + sigmoid. This probe measures whether the smooth (1-Lipschitz) SDF
boundary still yields a LOW + R-STABLE realized SegNet d_seg, isolating the RENDER composition
from the training difficulty by using the IDEAL SDF (reproduces L* exactly) + a natural per-class
palette (mean GT color per class — the in-distribution best case).

Decisive numbers (per pair, frozen CPU-torch SegNet authority):
  field_postR        = argmax(R_field(phi)) vs L*            (the 587x quantity, reference)
  realized_dseg_R    = SegNet(R(RGB)).argmax vs L*           (THE SCORED quantity)
  realized_dseg_noR  = SegNet(RGB@camera, uint8).argmax vs L* (R-isolation: flips from re-seg alone)
  r_added_segnet     = realized_dseg_R - realized_dseg_noR    (flips R adds at the SegNet output)

VERDICT logic: if realized_dseg_R is LOW (ideal SDF + natural palette lands in SegNet's L*
pre-image with R-stability) -> the vehicle is SOUND (the smoke's 0.507 is a training/capacity
issue). If realized_dseg_R is HIGH (~chance) -> SegNet does NOT read the palette frame ->
the 1-Lipschitz quantity must be the RGB MARGIN SegNet sees, NOT latent phi -> reformulate.
Swept across T (softmax temperature) + texture level. [macOS-CPU advisory]; non-promotable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="level-set field->realized SegNet R-survival transfer probe")
    ap.add_argument("--num-pairs", type=int, default=4)
    ap.add_argument("--temps", type=str, default="0.5,0.1,0.03")
    ap.add_argument("--tex-bound", type=float, default=0.3,
                    help="bounded GT-residual texture arm: rgb logit += clip(gt_resid, -B, B).")
    ap.add_argument("--out-json", type=Path, default=REPO / ".omx" / "research" / "levelset_transfer_probe.json")
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F

    from tac.boundary_math.lever_b_levelset_generator import apply_R_to_fields, signed_distance_fields
    from tac.boundary_math.seg_core import decode_gt_frame1_pairs, load_real_segnet, segnet_argmax_and_margin
    from train_witness_realized_through_R_mlx import _torch_R_to_camera_uint8, cpu_verdict_d_seg_batch

    seg = load_real_segnet("cpu")
    temps = [float(t) for t in args.temps.split(",")]

    # gather L* + GT f1 (downsampled to L* res for the per-class palette mean).
    items = []
    for _idx, _f0, f1 in decode_gt_frame1_pairs(n_pairs=args.num_pairs):
        lstar, _m = segnet_argmax_and_margin(seg, np.asarray(f1))  # (384,512)
        h, w = lstar.shape
        f1t = torch.from_numpy(np.asarray(f1, np.float32)).permute(2, 0, 1)[None]
        f1_small = F.interpolate(f1t, size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
        items.append((lstar.astype(np.int64), f1_small.astype(np.float32)))
    print(json.dumps({"stage": "gathered", "n_pairs": len(items), "shape": list(items[0][0].shape)}), flush=True)

    rows = []
    for ti, (lstar, gt_small) in enumerate(items):
        h, w = lstar.shape
        phi = signed_distance_fields(lstar, 5)  # (h,w,5) ideal SDF, argmax==L*
        # natural per-class palette in LOGIT space (so sigmoid(palette)*255 -> mean color).
        pal = np.zeros((5, 3), np.float32)
        for k in range(5):
            m = lstar == k
            pal[k] = (gt_small[m].mean(0) if m.any() else np.array([127.0, 127.0, 127.0], np.float32))
        pal_logit = _logit(pal / 255.0)  # (5,3)
        # field-level reference (the 587x quantity).
        rfield = apply_R_to_fields(phi)
        h2, w2, _ = rfield.shape
        ys = np.linspace(0, h - 1, h2).round().astype(np.int64); xs = np.linspace(0, w - 1, w2).round().astype(np.int64)
        l2 = lstar[np.ix_(ys, xs)]
        field_postR = float(np.count_nonzero(rfield.argmax(-1) != l2)) / l2.size
        # GT-residual texture (logit-space) for the bounded-texture arm.
        base_color = pal[lstar]  # (h,w,3) piecewise-constant natural color
        gt_resid_logit = _logit(np.clip(gt_small, 1, 254) / 255.0) - _logit(np.clip(base_color, 1, 254) / 255.0)

        for T in temps:
            ex = np.exp((phi - phi.max(-1, keepdims=True)) / T)
            soft = ex / ex.sum(-1, keepdims=True)              # (h,w,5)
            base_logit = soft @ pal_logit                      # (h,w,3)
            for tex_name, tex in (("tex0", 0.0), ("texB", np.clip(gt_resid_logit, -args.tex_bound, args.tex_bound))):
                rgb = (1.0 / (1.0 + np.exp(-(base_logit + tex)))) * 255.0  # (h,w,3) the trainer's render
                cam = _torch_R_to_camera_uint8(rgb.astype(np.float32))     # camera uint8 (the R)
                d_R = cpu_verdict_d_seg_batch(seg, [cam], [lstar])[0]      # realized SCORED d_seg
                # no-R isolation: SegNet on the camera-res render WITHOUT the bicubic+uint8 R
                # (upsample float -> still goes through SegNet.preprocess bilinear to 384).
                cam_noR = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[None]
                cam_noR = F.interpolate(cam_noR, size=(874, 1164), mode="bilinear", align_corners=False)
                cam_noR = cam_noR[0].permute(1, 2, 0).numpy().astype(np.uint8)
                d_noR = cpu_verdict_d_seg_batch(seg, [cam_noR], [lstar])[0]
                rows.append({"pair": ti, "T": T, "tex": tex_name, "field_postR": field_postR,
                             "realized_dseg_R": d_R, "realized_dseg_noR": d_noR,
                             "r_added_segnet": d_R - d_noR})
        print(json.dumps({"stage": "pair", "i": ti, "field_postR": round(field_postR, 6),
                          "best_realized_R": round(min(r["realized_dseg_R"] for r in rows if r["pair"] == ti), 4)}), flush=True)

    def _agg(sel):
        return {
            "realized_dseg_R": float(np.mean([r["realized_dseg_R"] for r in sel])),
            "realized_dseg_noR": float(np.mean([r["realized_dseg_noR"] for r in sel])),
            "r_added_segnet": float(np.mean([r["r_added_segnet"] for r in sel])),
            "field_postR": float(np.mean([r["field_postR"] for r in sel])),
        }

    by_cfg = {}
    for T in temps:
        for tex in ("tex0", "texB"):
            sel = [r for r in rows if r["T"] == T and r["tex"] == tex]
            by_cfg[f"T{T}_{tex}"] = _agg(sel)
    best = min(by_cfg.items(), key=lambda kv: kv[1]["realized_dseg_R"])
    field_ref = float(np.mean([r["field_postR"] for r in rows]))
    result = {
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "axis": "[macOS-CPU advisory] transfer probe; promotion_eligible=false",
        "n_pairs": len(items), "by_config": by_cfg,
        "best_config": best[0], "best_realized_dseg_R": best[1]["realized_dseg_R"],
        "field_level_postR_reference": field_ref,
        "transfer_ratio_best_over_field": best[1]["realized_dseg_R"] / max(field_ref, 1e-9),
        "VERDICT": ("TRANSFERS (vehicle sound; ideal-SDF+palette is read by SegNet R-stably)"
                    if best[1]["realized_dseg_R"] < 0.05 else
                    "DOES_NOT_TRANSFER (SegNet re-seg of palette frame is OOD/unstable; "
                    "reformulate: regularize the realized RGB-margin to 1-Lipschitz, not latent phi)"),
    }
    out = Path(args.out_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("\n=== LEVEL-SET TRANSFER PROBE (field 587x -> realized SegNet argmax) ===")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
