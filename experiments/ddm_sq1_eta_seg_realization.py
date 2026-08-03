#!/usr/bin/env python
"""ddm_sq1 Job 1 -- measure eta_seg (gp1 falsifier F1) through the REAL receiver path.

gp1 priced DESCRIPTIONS: "band pixel p should be class c", 367,523 B, bound -0.17466 S.
It explicitly did NOT build the mechanism that turns that description into an RGB change the
frozen SegNet actually argmaxes differently.  This script measures that mechanism.

    eta_net = (flips_before - flips_after) / n_described        [whole-frame accounting]

Everything is scored through the ACTUAL contest scorer object (`upstream.modules.DistortionNet`)
on the ACTUAL decoded camera-resolution frames emitted by the shipped archive.  No
reimplementation of the resize operator `D`, no proxy render, no MPS.

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.

Pre-registration: `.omx/research/ddm_sq1_eta_seg_and_hinge_ab_20260803.md` §1 (committed
BEFORE this script was run).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"
for _p in (str(UPSTREAM), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import einops  # noqa: E402

from frame_utils import camera_size, seq_len, yuv420_to_rgb  # noqa: E402
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402

CAM_W, CAM_H = camera_size          # 1164, 874
SEG_W, SEG_H = 512, 384             # segnet_model_input_size, (W, H)
N_PAIRS_TOTAL = 600
N_CLASSES = 5
# comma10k CANONICAL order (MEASURED, CLAUDE.md -- never re-derive by luma sort)
CLASS_NAMES = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]
ROAD, LANE = 0, 1

# --- denominators, all named (never a bare delta) -----------------------------------------
LIVE_BEST_S = 0.826496209256714     # v4d_cx1_pj2ix2 report.txt, recomputed from components
PR130_FLOOR_S = 0.172141            # the BAR (lessons-only lineage; never ours)
GAP_S = LIVE_BEST_S - PR130_FLOOR_S
BASE_D_POSE = 0.0025513987495742437
S_PER_FLIP = 100.0 / (N_PAIRS_TOTAL * SEG_H * SEG_W)
RADII = (1, 2, 3, 5, 8, 13, 21, 34, 55)
DS_DDPOSE = 5.0 / np.sqrt(10.0 * BASE_D_POSE)   # CURRENT slope, never a shelf price (K3)


# ============================================================================================
# geometry of D: which camera pixels does each scorer pixel read?
# ============================================================================================
def bilinear_support_indices(n_out: int, n_in: int) -> np.ndarray:
    """torch bilinear align_corners=False support: src = (dst+0.5)*scale - 0.5.

    Returns (n_out, 2) int array of the two contributing input indices per output index.
    """
    scale = n_in / n_out
    src = (np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5
    lo = np.floor(src).astype(np.int64)
    hi = lo + 1
    return np.stack([np.clip(lo, 0, n_in - 1), np.clip(hi, 0, n_in - 1)], axis=1)


ROW_SUP = bilinear_support_indices(SEG_H, CAM_H)   # (384, 2)
COL_SUP = bilinear_support_indices(SEG_W, CAM_W)   # (512, 2)


def _assert_private_support() -> dict:
    """m86: scale 2.276 > 2 => each scorer pixel owns a PRIVATE 2x2 camera block.

    FAIL-CLOSED, not merely reported: `scorer_mask_to_camera` and
    `realize_scorer_paint_to_camera` both assume that editing one scorer pixel's camera block
    cannot perturb a neighbouring scorer pixel.  If that ever stopped holding, every eta below
    would be silently contaminated -- the m50 vacuity class.  So it is asserted here.
    """
    rows = np.concatenate([ROW_SUP[:, 0], ROW_SUP[:, 1]])
    cols = np.concatenate([COL_SUP[:, 0], COL_SUP[:, 1]])
    row_priv = rows.size == np.unique(rows).size
    col_priv = cols.size == np.unique(cols).size
    if not (row_priv and col_priv):
        raise RuntimeError(
            "D's 2x2 supports are NOT private (rows_private=%s cols_private=%s); band-local "
            "paint would leak across scorer pixels and every eta would be contaminated."
            % (row_priv, col_priv)
        )
    touched = np.unique(rows).size * np.unique(cols).size
    return {
        "rows_private": bool(row_priv),
        "cols_private": bool(col_priv),
        "camera_px_touched_by_D": int(touched),
        "camera_px_total": int(CAM_H * CAM_W),
        "frac_blind_to_scorers": float(1.0 - touched / (CAM_H * CAM_W)),
    }


def scorer_mask_to_camera(mask: np.ndarray) -> np.ndarray:
    """Expand a (384,512) bool scorer mask to the (874,1164) camera pixels D reads for it."""
    cam = np.zeros((CAM_H, CAM_W), dtype=bool)
    for a in range(2):
        r = ROW_SUP[:, a]
        for b in range(2):
            c = COL_SUP[:, b]
            cam[r[:, None], c[None, :]] |= mask
    return cam


def scorer_labels_to_camera(mask: np.ndarray, labels: np.ndarray) -> tuple:
    """Return (cam_idx_flat, cam_labels) for the camera pixels of the masked scorer pixels."""
    ii, jj = np.nonzero(mask)
    lab = labels[ii, jj]
    idx_list, lab_list = [], []
    for a in range(2):
        r = ROW_SUP[ii, a]
        for b in range(2):
            c = COL_SUP[jj, b]
            idx_list.append(r.astype(np.int64) * CAM_W + c.astype(np.int64))
            lab_list.append(lab)
    return np.concatenate(idx_list), np.concatenate(lab_list)


# ============================================================================================
# the free band: pixels within r px of the DECODER'S OWN label boundary (gp1 F4)
# ============================================================================================
def boundary(lab: np.ndarray) -> np.ndarray:
    """4-neighbour label boundary -- VERBATIM from experiments/ddm_gp1_free_band_and_net.py.

    Reused (not re-derived) so the band this unit realizes is byte-for-byte the SAME object
    gp1 priced at 367,523 B; the F1 threshold 0.583 is only meaningful against gp1's band.
    """
    b = np.zeros(lab.shape, dtype=bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Chebyshev dilation by r via iterated 3x3 OR -- gp1's convention."""
    out = mask.copy()
    for _ in range(r):
        acc = out.copy()
        acc[:-1, :] |= out[1:, :]
        acc[1:, :] |= out[:-1, :]
        acc[:, :-1] |= out[:, 1:]
        acc[:, 1:] |= out[:, :-1]
        cur = acc.copy()
        cur[:-1, :] |= acc[1:, :]
        cur[1:, :] |= acc[:-1, :]
        out = cur
    return out


def label_boundary_band(labels: np.ndarray, r: int) -> np.ndarray:
    """gp1's FREE band: pixels within r of the DECODER'S OWN label boundary. Zero bytes."""
    b = boundary(labels)
    return b if r <= 0 else dilate(b, r)


# ============================================================================================
# scorer driving -- the real DistortionNet, nothing else
# ============================================================================================
class Scorer:
    def __init__(self, threads: int):
        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        self.net = DistortionNet().eval()
        self.net.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))

    @staticmethod
    def _to_bthwc(pair_u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(pair_u8))[None]      # (1,2,H,W,3) uint8
        return einops.rearrange(x, "b t h w c -> b t c h w").float()

    @torch.inference_mode()
    def seg_argmax(self, pair_u8: np.ndarray) -> np.ndarray:
        x = self._to_bthwc(pair_u8)
        out = self.net.segnet(self.net.segnet.preprocess_input(x))     # (1,5,384,512)
        return out.argmax(dim=1)[0].numpy().astype(np.uint8)

    @torch.inference_mode()
    def pose_out(self, pair_u8: np.ndarray):
        x = self._to_bthwc(pair_u8)
        return self.net.posenet(self.net.posenet.preprocess_input(x))

    def d_pose(self, out_gt, out_cmp) -> float:
        return float(self.net.posenet.compute_distortion(out_gt, out_cmp)[0])


# ============================================================================================
# GT decode -- canonical yuv420_to_rgb ONLY (PyAV rgb24 manufactures ~100x phantom pose)
# ============================================================================================
def decode_gt_frames(mkv: Path, wanted: set[int]) -> dict:
    import av

    out, hi = {}, max(wanted)
    container = av.open(str(mkv))
    stream = container.streams.video[0]
    for idx, frame in enumerate(container.decode(stream)):
        if idx in wanted:
            out[idx] = yuv420_to_rgb(frame).numpy()
        if idx >= hi:
            break
    container.close()
    return out


# ============================================================================================
# realization rungs
# ============================================================================================
def paint_class_anchor(dec_f1: np.ndarray, band: np.ndarray, tgt: np.ndarray,
                       lstar: np.ndarray) -> np.ndarray:
    """L1 -- RECEIVER-LEGAL: paint band camera pixels with the decoded frame's OWN per-class
    anchor colour for the shipped target class.

    Anchor for class c = mean decoded camera RGB over the camera pixels of scorer pixels that
    are labelled c by the decoder AND lie OUTSIDE the band (stable interior).  Uses only
    (decoded RGB, decoded L*, shipped label) -- zero scorer weights, zero GT pixels.
    """
    edited = dec_f1.copy()
    flat = edited.reshape(-1, 3)
    anchors = {}
    for c in range(N_CLASSES):
        interior = (lstar == c) & (~band)
        if interior.sum() < 16:
            interior = lstar == c
        if interior.sum() == 0:
            continue
        cam = scorer_mask_to_camera(interior)
        anchors[c] = dec_f1[cam].reshape(-1, 3).mean(axis=0)
    idx, lab = scorer_labels_to_camera(band, tgt)
    for c, a in anchors.items():
        sel = idx[lab == c]
        if sel.size:
            flat[sel] = np.round(a).astype(np.uint8)
    return edited


def paste_truth(dec_f1: np.ndarray, gt_f1: np.ndarray, band: np.ndarray) -> np.ndarray:
    """L0 -- CONTENT ORACLE: the true camera pixels, but only inside the band."""
    edited = dec_f1.copy()
    cam = scorer_mask_to_camera(band)
    edited[cam] = gt_f1[cam]
    return edited


# ============================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, required=True)
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--pairs-npy", type=Path, required=True)
    ap.add_argument("--argmax-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N selected pairs")
    args = ap.parse_args()

    t0 = time.time()
    pairs = np.load(args.pairs_npy).tolist()
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[sq1] {len(pairs)} pairs: {pairs}", flush=True)

    geom = _assert_private_support()
    print(f"[sq1] D geometry: {geom}", flush=True)

    raw = np.memmap(args.sub_dir / "inflated" / "0.raw", dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))
    cx1 = np.load(args.argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gtc = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")

    wanted = set()
    for p in pairs:
        wanted.update({seq_len * p, seq_len * p + 1})
    print(f"[sq1] decoding {len(wanted)} GT frames (canonical yuv420_to_rgb)...", flush=True)
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    print(f"[sq1] GT decode done  t={time.time()-t0:.1f}s", flush=True)

    sc = Scorer(args.threads)
    print(f"[sq1] scorer loaded   t={time.time()-t0:.1f}s", flush=True)

    rows = []
    for n, p in enumerate(pairs):
        tp = time.time()
        dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])

        lstar = sc.seg_argmax(dec)
        lgt = sc.seg_argmax(gt)

        rec = {
            "pair": int(p),
            # ---- positive controls C2 / C3 -------------------------------------------------
            "C2_lstar_matches_cache": bool((lstar == np.asarray(cx1[p])).all()),
            "C3_lgt_matches_cache": bool((lgt == np.asarray(gtc[p])).all()),
        }

        flips0_map = lstar != lgt
        flips0 = int(flips0_map.sum())
        rec["flips_before"] = flips0

        pose_gt = sc.pose_out(gt)
        rec["d_pose_before"] = sc.d_pose(pose_gt, sc.pose_out(dec))

        def run(tag, edited_f1, described, do_pose=False):
            """Score one realization rung.  `described` is a bool scorer-lattice mask.

            n_described = the flips the description ADDRESSES (in-band flips) -- gp1's own
            denominator for eta.  eta_net divides the WHOLE-FRAME flip reduction by it, so
            collateral damage outside the band is charged against the rung; eta_inband_raw is
            the diagnostic that ignores collateral.  The gap between them IS the collateral.
            """
            pair_e = np.stack([dec[0], edited_f1])
            lam = sc.seg_argmax(pair_e)
            fa = int((lam != lgt).sum())
            n_desc_flips = int((flips0_map & described).sum())
            fixed_in = int((flips0_map & described & (lam == lgt)).sum())
            r = {
                f"{tag}_n_described": n_desc_flips,
                f"{tag}_flips_after": fa,
                f"{tag}_eta_net": ((flips0 - fa) / n_desc_flips) if n_desc_flips else None,
                f"{tag}_eta_inband_raw": (fixed_in / n_desc_flips) if n_desc_flips else None,
                f"{tag}_band_px": int(described.sum()),
            }
            if do_pose:
                r[f"{tag}_d_pose_after"] = sc.d_pose(pose_gt, sc.pose_out(pair_e))
            return r

        # ---- C4 positive control: full-frame GT paste MUST give eta=1, flips_after=0 -------
        full = np.ones((SEG_H, SEG_W), dtype=bool)
        rec.update(run("P0_fullpaste", gt[1], full, do_pose=True))

        # ---- the locality curve: truth pasted only inside the r-band ----------------------
        # gp1's A3 is r=1, A4 is r=2.  The wider radii probe WHERE eta crosses 0 and 0.583 --
        # i.e. how much of the frame a band-local realization must own before it stops being
        # anti-productive.  That crossover, not eta(r=1) alone, is what re-prices the ladder.
        bands = {}
        for r_ in RADII:
            b = label_boundary_band(lstar, r_)
            bands[r_] = b
            rec[f"band_r{r_}_frac"] = float(b.mean())
            rec[f"band_r{r_}_capture"] = float((flips0_map & b).sum() / max(flips0, 1))
            rec.update(run(f"L0_r{r_}", paste_truth(dec[1], gt[1], b), b,
                           do_pose=(r_ == 1)))

        # ---- the receiver-legal rung ------------------------------------------------------
        b1 = bands[1]
        rec.update(run("L1_r1", paint_class_anchor(dec[1], b1, lgt, lstar), b1, do_pose=True))

        # ---- F2: the Road<->Lane restriction ----------------------------------------------
        rl = b1 & (((lstar == ROAD) & (lgt == LANE)) | ((lstar == LANE) & (lgt == ROAD)))
        rec.update(run("L0_RL", paste_truth(dec[1], gt[1], rl), rl))
        rec.update(run("L1_RL", paint_class_anchor(dec[1], rl, lgt, lstar), rl, do_pose=True))

        rows.append(rec)
        print(f"[sq1] pair {p:3d} ({n+1}/{len(pairs)})  flips {flips0:5d}  "
              f"eta_net L0r1 {rec['L0_r1_eta_net']:.4f}  L1r1 {rec['L1_r1_eta_net']:.4f}  "
              f"P0 {rec['P0_fullpaste_eta_net']:.4f}  [{time.time()-tp:.1f}s]", flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"schema": "ddm_sq1_eta_seg.v1",
                       "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                       "score_claim": False, "promotion_eligible": False,
                       "pointer": "0.1910828242 [contest-CPU] UNMOVED",
                       "D_geometry": geom,
                       "denominators": {"live_best_S": LIVE_BEST_S, "floor_S": PR130_FLOOR_S,
                                        "gap_S": GAP_S, "S_per_flip": S_PER_FLIP,
                                        "dS_ddpose_current": float(DS_DDPOSE)},
                       "pairs": pairs, "rows": rows}, f, indent=1)

    print(f"[sq1] DONE {len(rows)} pairs  t={time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
