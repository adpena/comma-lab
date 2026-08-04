# SPDX-License-Identifier: MIT
"""ddm_bz1 -- the SEG carriage linchpin: what does the offset field realize through a LEGAL receiver?

ph1 SS6 and et1 both priced the block16 offset field (57,809 B LZMA1) but measured its seg gain
by TRANSLATING THE ARGMAX FIELD DIRECTLY (ph1) or by a FROZEN-SEGNET-GUIDED PAINT (et1, eta=0.5267).
Neither is what a legal inflate can do:

  * translating the argmax field needs the argmax field -- ph1 SS6: "a real receiver has tokens,
    renders through R, and the argmax EMERGES ... it is an UPPER BOUND on the carrier";
  * the SegNet-guided paint needs SegNet at inflate (FORBIDDEN) OR ships the painted pixels (a
    much larger, unpriced rate) -- the object priced (offsets) is not the object realized (pixels).

The ONE thing a receiver CAN do deterministically from the offset field alone is TRANSLATE THE
RENDERED FRAME_1 RGB BLOCKS by the per-block offsets and let SegNet re-argmax the translated RGB.
That is legal (no scorer at inflate: the translation is deterministic) and cheap (57,809 B).  Its
seg gain is UNMEASURED and is the number that decides whether the offset field is a real seg
carrier or an argmax-field mirage.

This probe measures, per pair, the DETERMINISTIC-RGB-TRANSLATION eta:
    eta_det = (flips_baseline - flips_after_RGB_translation) / n_described
against et1's SegNet-paint eta on the same pairs.  If eta_det << 0.5267 the offset field cannot
carry the seg gain deterministically and the composed row's seg leg is a mirage; if eta_det is
close, the offset field is the legal cheap carrier and the row's seg economics hold.

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=False.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from ddm_et1_ph1_block16_on_our_vehicle import solve_blocks, translate_blocks  # noqa: E402
from ddm_js1_staging_discriminator import (  # noqa: E402
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    S_PER_FLIP,
    Scorer,
    decode_gt_frames,
    resize_to_scorer,
    seq_len,
)


def translate_rgb_blocks(img_s: np.ndarray, off: np.ndarray, block: int) -> np.ndarray:
    """Deterministic per-block translation of a scorer-lattice RGB frame (H,W,3), border-clamped,
    using the SAME sampling convention as translate_blocks (out[y,x] = img[y+dy, x+dx])."""
    nby, nbx = SEG_H // block, SEG_W // block
    out = img_s.copy()
    for bi in range(nby):
        for bj in range(nbx):
            dy, dx = int(off[bi * nbx + bj][0]), int(off[bi * nbx + bj][1])
            if dy == 0 and dx == 0:
                continue
            ys, ye, xs, xe = bi * block, (bi + 1) * block, bj * block, (bj + 1) * block
            yy = np.clip(np.arange(ys, ye) + dy, 0, SEG_H - 1)
            xx = np.clip(np.arange(xs, xe) + dx, 0, SEG_W - 1)
            out[ys:ye, xs:xe] = img_s[np.ix_(yy, xx)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, required=True)
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--pairs-npy", type=Path, required=True)
    ap.add_argument("--argmax-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only-pairs", type=str, required=True)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--rmax", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    all_pairs = np.load(args.pairs_npy).tolist()
    pairs = [int(x) for x in args.only_pairs.split(",")]
    miss = [p for p in pairs if p not in all_pairs]
    if miss:
        raise SystemExit(f"--only-pairs {miss} not in --pairs-npy")

    raw = np.memmap(args.sub_dir / "inflated" / "0.raw", dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))
    # lstar/lgt are computed live from the scorer below; the argmax cache path is kept as a required
    # arg for interface parity with the other bz1 harnesses but is not consumed here.
    wanted = set()
    for p in pairs:
        wanted.update({seq_len * p, seq_len * p + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    sc = Scorer(args.threads)
    print(f"[detseg] scorer ready t={time.time()-t0:.1f}s", flush=True)

    rows = []
    for n, p in enumerate(pairs):
        tp = time.time()
        dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])
        lstar = sc.seg_argmax(dec)
        lgt = sc.seg_argmax(gt)
        flips0 = int((lstar != lgt).sum())

        # solve the offsets on the LABEL field (as ph1/et1 do), zero-biased tie-break
        off = solve_blocks(lstar, lgt, args.block, args.rmax)
        target = translate_blocks(lstar, off.reshape(-1, 2), args.block)
        band = target != lstar
        nd = flips0 - int((target != lgt).sum())          # label-space ceiling (n_described)

        # ---- LEGAL DETERMINISTIC RECEIVER: translate the RENDERED frame_1 RGB blocks ----------
        # frame_1 on the scorer lattice, translated per block, then re-argmaxed through SegNet.
        f1_s = resize_to_scorer(dec[1])[0].permute(1, 2, 0).numpy()          # (384,512,3) float
        f1_s_t = translate_rgb_blocks(f1_s, off.reshape(-1, 2), args.block)
        f1_cam_t = _scorer_to_camera_bilinear(f1_s_t)                        # (CAM_H,CAM_W,3)
        pair_det = np.stack([dec[0], f1_cam_t]).astype(np.uint8)
        lam_det = sc.seg_argmax(pair_det)
        fa_det = int((lam_det != lgt).sum())
        eta_det = ((flips0 - fa_det) / nd) if nd else None

        # For reference: the label-space ceiling realized EXACTLY (translate argmax directly).
        # This is ph1/et1's UPPER BOUND -- what the field buys if RGB translation reproduced it.
        lam_label = target
        fa_label = int((lam_label != lgt).sum())
        eta_label_ceiling = ((flips0 - fa_label) / nd) if nd else None       # == 1.0 by construction

        rec = {
            "pair": int(p), "flips_before": flips0, "n_described": nd,
            "band_px": int(band.sum()),
            "flips_after_det_rgb_translate": fa_det,
            "eta_deterministic_rgb_translate": eta_det,
            "eta_label_ceiling": eta_label_ceiling,
            "seg_S_gain_det": (flips0 - fa_det) * S_PER_FLIP,
            "seg_S_ceiling": nd * S_PER_FLIP,
        }
        rows.append(rec)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"schema": "ddm_bz1_deterministic_seg_realize.v1",
                   "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                   "score_claim": False, "promotion_eligible": False,
                   "note": "eta_det = LEGAL deterministic-RGB-translation seg realization; "
                           "compare to et1 SegNet-paint eta=0.5267 (which needs SegNet at inflate "
                           "or ships painted pixels).",
                   "S_per_flip": S_PER_FLIP, "rows": rows},
                  open(args.out, "w"), indent=1)
        print(f"[detseg] pair {p:3d} ({n+1}/{len(pairs)}) flips0 {flips0:5d} nd {nd:5d} "
              f"| eta_det {eta_det if eta_det is None else round(eta_det,4)} "
              f"(paint eta ref 0.5267) [{time.time()-tp:.1f}s]", flush=True)

    print(f"[detseg] done t={time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


def _scorer_to_camera_bilinear(img_s: np.ndarray) -> np.ndarray:
    """Upsample a (384,512,3) scorer-lattice frame to the camera plane by bilinear resize -- the
    inverse of resize_to_scorer's downsample.  This is the deterministic realization of a translated
    scorer-lattice frame back into camera pixels (a receiver-legal op, no scorer)."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(img_s).permute(2, 0, 1)[None].float()
    up = F.interpolate(t, size=(CAM_H, CAM_W), mode="bilinear", align_corners=False)
    return np.clip(np.round(up[0].permute(1, 2, 0).numpy()), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
