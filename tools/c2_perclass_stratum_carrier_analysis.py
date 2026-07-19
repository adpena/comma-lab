# SPDX-License-Identifier: MIT
"""c2 per-class × per-stratum carrier analysis — residual decomposition + separatrix
asymmetry + sensitivity drivers + seed+residual feasibility smokes (n600, $0 local).

Extends tools/necessity_dseg_calibration.py (b625194009): at its measured min-S knee
(eps=0 lossless partition + static ds16 hood-tex seed, d_seg_real 0.01328 n600), answer:

  decomp  Which (class × stratum) buckets carry the residual d_seg, with per-bucket
          confusion (gt->pred; the erasure direction) AND static-vs-dynamic stability
          (consecutive-frame persistence + occupancy of the disagreement mask).
          n600, resumable JSONL + packbits mask memmap.
  slope   Separatrix asymmetry at n600 from the CACHED bit-exact margin field
          (gt_n600.npz 'margins'): one-sided margin profile m(k), k=1..6 px from each
          class-pair boundary, per side -> slope + asymmetry ratio + shallow side.
          Extends eq separatrix_asymmetry_t_subpixel_boundary_localizer_v1 (n6 -> n600).
          $0, numpy-only (no SegNet forward).
  sens    Margin-deficit VJP at DISAGREEING pixels of the palette render through the
          REAL frozen SegNet (incl. exact bilinear resize): decompose the cure gradient
          onto luma BT.601 (span{ell}) vs its Euclidean orthogonal complement ker(ell)
          vs spatial locality/side/coherence -> which driver the carrier must supply.
          NOTE (corrected 2026-07-19, Task #570 / #564 §4): this active split is the
          Euclidean span{ell}/ker(ell) split (lines 345-374, `gl = gr @ LUMA_HAT`), NOT
          a U/V analysis-covector split. span{U-row,V-row} = (1,1,1)^perp differs from
          ker(ell) by a 30.27914784 deg principal angle (projector distance 0.504213367),
          so this output must NOT be relabeled "U/V" or "chroma sensitivity". See the
          fixture src/tac/tests/test_yuv6_analysis_covectors_vs_primal_luma_null_20260719.py.
  smoke   Generator-variant + oracle-seed feasibility on a stride subset (LABELLED
          n<600): one-sided vs symmetric boundary-contrast band (0 B), blur (0 B),
          GT-texture oracles (counted bytes, brotli): global ds16 / Movable-crop /
          boundary-band -> d_seg/byte per bucket's candidate carrier.

Axes / honesty: [macOS-CPU advisory] frozen CPU-torch fp32 SegNet on the bit-exact
cached GT argmax (gt_n600.npz). research_only; score_claim=false; promotable=false.
The pointer (0.19108) moves only via upstream/evaluate.py on exact archive bytes.

Usage:
  .venv/bin/python tools/c2_perclass_stratum_carrier_analysis.py --stage decomp
  .venv/bin/python tools/c2_perclass_stratum_carrier_analysis.py --stage temporal
  .venv/bin/python tools/c2_perclass_stratum_carrier_analysis.py --stage slope
  .venv/bin/python tools/c2_perclass_stratum_carrier_analysis.py --stage sens
  .venv/bin/python tools/c2_perclass_stratum_carrier_analysis.py --stage smoke --variant blur2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

GT_N600 = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
CAL_DIR = "experiments/results/necessity_dseg_calibration_20260715"
OUT_DIR = "experiments/results/c2_perclass_stratum_20260716"
CLASSES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
STRATA = ("saddle", "edge", "near", "far")
H_S, W_S = 384, 512
H_C, W_C = 874, 1164
N_RATE = 37_545_489
AXIS = "[macOS-CPU advisory] frozen CPU-torch fp32; bit-exact cached GT (gt_n600.npz)"
LUMA = np.array([0.299, 0.587, 0.114])
LUMA_HAT = LUMA / np.linalg.norm(LUMA)

sys.path.insert(0, "tools")
from necessity_dseg_calibration import (  # noqa: E402
    _hood_seed,
    _load_segnet,
    _pair_name,
    _realize_frame,
    _segnet_argmax,
    _strata_masks,
)


def _load_gt(keys: tuple[str, ...]) -> dict:
    z = np.load(GT_N600)
    return {k: z[k] for k in keys}


def _palette() -> np.ndarray:
    with open(os.path.join(CAL_DIR, "palette.json")) as fh:
        pj = json.load(fh)
    return np.array(pj[f"palette_{pj['winner']}_rgb"], np.uint8)


def _stratum_field(lab: np.ndarray):
    """Per-pixel stratum id (0 saddle,1 edge,2 near,3 far) + boundary pair_id map."""
    smask, bmask, pair_id, near = _strata_masks(lab)
    strat = np.full(lab.shape, 3, np.uint8)
    strat[near] = 2
    strat[bmask] = 1
    strat[smask] = 0
    return strat, pair_id


# ---------------------------------------------------------------------------
# Stage DECOMP — n600 per-(class x stratum) confusion + disagreement masks
# ---------------------------------------------------------------------------
def stage_decomp(limit: int, batch: int) -> None:
    rows_path = os.path.join(OUT_DIR, "decomp_rows.jsonl")
    masks_path = os.path.join(OUT_DIR, "dis_masks_packbits.u8")
    done: set[int] = set()
    if os.path.exists(rows_path):
        for line in open(rows_path):
            try:
                done.add(json.loads(line)["frame"])
            except (json.JSONDecodeError, KeyError):
                pass
    g = _load_gt(("lstars", "gt_f1"))
    lstars = g["lstars"]
    n = lstars.shape[0]
    masks = np.memmap(masks_path, dtype=np.uint8, mode="r+" if os.path.exists(masks_path)
                      else "w+", shape=(n, H_S * W_S // 8))
    hood = _hood_seed(lstars, g["gt_f1"])[0]
    del g
    palette = _palette()
    todo = [i for i in range(n) if i not in done][:limit]
    if not todo:
        print(f"[c2] decomp: all {n} frames done")
        return
    model = _load_segnet()
    t0 = time.time()
    with open(rows_path, "a") as fh:
        for k in range(0, len(todo), batch):
            idx = todo[k:k + batch]
            cams = []
            for i in idx:
                _, cam = _realize_frame(lstars[i], 0.0, palette, hood=hood)
                cams.append(cam)
            preds = _segnet_argmax(model, np.stack(cams))
            for j, i in enumerate(idx):
                lab = lstars[i]
                pred = preds[j]
                dis = pred != lab
                masks[i] = np.packbits(dis)
                strat, pair_id = _stratum_field(lab)
                # confusion per stratum: counts[(stratum, gt, pred)]
                conf: dict[str, int] = {}
                ss, gg, pp = strat[dis], lab[dis], pred[dis]
                key = ss.astype(np.int64) * 25 + gg * 5 + pp
                for kk, cc in zip(*np.unique(key, return_counts=True), strict=True):
                    s_, g_, p_ = int(kk) // 25, (int(kk) % 25) // 5, int(kk) % 5
                    conf[f"{STRATA[s_]}|{CLASSES[g_]}|{CLASSES[p_]}"] = int(cc)
                # edge+near disagreements by (pair, side-class): nearest-boundary pair
                side: dict[str, int] = {}
                en = dis & (strat <= 2) & (strat >= 1)
                if en.any():
                    from scipy.ndimage import distance_transform_edt
                    bm = pair_id != 255
                    inds = distance_transform_edt(~bm, return_distances=False,
                                                  return_indices=True)
                    near_pair = pair_id[inds[0], inds[1]]
                    codes = near_pair[en].astype(np.int64) * 5 + lab[en]
                    for kk, cc in zip(*np.unique(codes, return_counts=True), strict=True):
                        pc, sc = int(kk) // 5, int(kk) % 5
                        if pc != 255:
                            side[f"{_pair_name(pc)}|{CLASSES[sc]}"] = int(cc)
                fh.write(json.dumps({"frame": int(i), "dseg": float(dis.mean()),
                                     "conf": conf, "pair_side": side}) + "\n")
            fh.flush()
            masks.flush()
            el = time.time() - t0
            print(f"[c2] decomp {k + len(idx)}/{len(todo)} ({el:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# Stage TEMPORAL — static-vs-dynamic from the stored masks (numpy only)
# ---------------------------------------------------------------------------
def stage_temporal() -> None:
    g = _load_gt(("lstars",))
    lstars = g["lstars"]
    n = lstars.shape[0]
    masks = np.memmap(os.path.join(OUT_DIR, "dis_masks_packbits.u8"), dtype=np.uint8,
                      mode="r", shape=(n, H_S * W_S // 8))
    occ = np.zeros((H_S, W_S), np.float64)
    dis_prev = None
    strat_prev = None
    lab_prev = None
    # per-(class x stratum): [n_dis, n_persist_next, occ_sum]
    acc = np.zeros((5, 4, 3), np.float64)
    dis_all = []
    for i in range(n):
        dis = np.unpackbits(masks[i]).reshape(H_S, W_S).astype(bool)
        dis_all.append(dis)
        occ += dis
    occ /= n
    for i in range(n):
        dis = dis_all[i]
        lab = lstars[i]
        strat, _ = _stratum_field(lab)
        if dis_prev is not None:
            pass
        nxt = dis_all[i + 1] if i + 1 < n else None
        for c in range(5):
            for s in range(4):
                m = dis & (lab == c) & (strat == s)
                cnt = int(m.sum())
                if not cnt:
                    continue
                acc[c, s, 0] += cnt
                if nxt is not None:
                    acc[c, s, 1] += int((m & nxt).sum())
                acc[c, s, 2] += float(occ[m].sum())
        dis_prev, strat_prev, lab_prev = dis, strat, lab  # noqa: F841
        if i % 100 == 0:
            print(f"[c2] temporal {i}/{n}", flush=True)
    out = {"scope": f"n600; {AXIS}",
           "model": ("persist_next = P(same-pixel disagrees at frame i+1 | disagrees at i); "
                     "occ = per-pixel disagreement frequency over n600 (static error sites "
                     "have high occ); buckets keyed by GT class x stratum at frame i"),
           "buckets": {}}
    for c in range(5):
        for s in range(4):
            nd = acc[c, s, 0]
            if nd < 100:
                continue
            out["buckets"][f"{CLASSES[c]}|{STRATA[s]}"] = {
                "dis_px_total": int(nd),
                "frac_of_all_dis": float(nd / acc[:, :, 0].sum()),
                "dseg_contribution": float(nd / (n * H_S * W_S)),
                "persist_next_frac": float(acc[c, s, 1] / nd),
                "mean_occupancy": float(acc[c, s, 2] / nd),
            }
    with open(os.path.join(OUT_DIR, "temporal.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
# Stage SLOPE — separatrix asymmetry: one-sided margin profile per pair (cached)
# ---------------------------------------------------------------------------
def stage_slope(stride: int) -> None:
    from scipy.ndimage import distance_transform_edt

    g = _load_gt(("lstars", "margins"))
    lstars, margins = g["lstars"], g["margins"]
    frames = list(range(0, lstars.shape[0], stride))
    kmax = 6
    bins = np.linspace(0.0, 12.0, 241)
    hist = np.zeros((25, 5, kmax, 240), np.int64)  # (pair, side-class, dist-1, bin)
    for i in frames:
        lab = lstars[i]
        m = margins[i]
        _, pair_id = _stratum_field(lab)
        bm = pair_id != 255
        dist, inds = distance_transform_edt(~bm, return_indices=True)
        near_pair = pair_id[inds[0], inds[1]]
        sel = (dist >= 0.5) & (dist < kmax + 0.5) & (near_pair != 255)
        kbin = np.clip(np.round(dist[sel]).astype(np.int64) - 1, 0, kmax - 1)
        pc = near_pair[sel].astype(np.int64)
        sc = lab[sel]
        mb = np.clip(np.digitize(m[sel], bins) - 1, 0, 239)
        np.add.at(hist, (pc, sc, kbin, mb), 1)
    centers = (bins[:-1] + bins[1:]) / 2

    def med(h):
        c = h.cumsum()
        if c[-1] == 0:
            return None
        return float(centers[np.searchsorted(c, c[-1] / 2)])

    out = {"scope": f"stride-{stride} ({len(frames)} frames of n600); cached bit-exact "
                    f"margins; {AXIS}",
           "model": ("one-sided margin profile m_med(k), k=1..6 px euclid from the "
                     "class-pair boundary, keyed by (nearest-boundary pair, pixel's own "
                     "GT class = side); slope = (m(3)-m(1))/2; shallow side = smaller "
                     "slope AND smaller m(1) -> where flips are cheap (erasure side)"),
           "pairs": {}}
    for code in range(25):
        a, b = code // 5, code % 5
        if a >= b:
            continue
        tot = int(hist[code].sum())
        if tot < 5000:
            continue
        row = {"n_px": tot, "sides": {}}
        for sc in (a, b):
            prof = [med(hist[code, sc, k]) for k in range(kmax)]
            if prof[0] is None or prof[2] is None:
                continue
            row["sides"][CLASSES[sc]] = {
                "m_med_k": [p if p is not None else -1 for p in prof],
                "m1": prof[0],
                "slope_1to3": (prof[2] - prof[0]) / 2.0,
            }
        if len(row["sides"]) == 2:
            (na, ra), (nb, rb) = row["sides"].items()
            sa, sb = ra["slope_1to3"], rb["slope_1to3"]
            shallow = na if (sa < sb) else nb
            row["asym_slope_ratio_max_over_min"] = float(max(sa, sb) / max(min(sa, sb), 1e-6))
            row["shallow_side"] = shallow
            row["m1_ratio"] = float(max(ra["m1"], rb["m1"]) / max(min(ra["m1"], rb["m1"]), 1e-6))
            row["m1_shallow_side"] = na if ra["m1"] < rb["m1"] else nb
        out["pairs"][_pair_name(code)] = row
    with open(os.path.join(OUT_DIR, "slope.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
# Stage SENS — margin-deficit VJP at disagreeing pixels of the palette render
# ---------------------------------------------------------------------------
def stage_sens(frame_stride: int, per_frame: int, seed: int) -> None:
    import torch
    import torch.nn.functional as tfun

    g = _load_gt(("lstars", "gt_f1"))
    lstars = g["lstars"]
    hood = _hood_seed(lstars, g["gt_f1"])[0]
    del g
    palette = _palette()
    model = _load_segnet()
    rng = np.random.default_rng(seed)
    n = lstars.shape[0]
    masks = np.memmap(os.path.join(OUT_DIR, "dis_masks_packbits.u8"), dtype=np.uint8,
                      mode="r", shape=(n, H_S * W_S // 8))
    rows = []
    for i in range(0, n, frame_stride):
        lab = lstars[i]
        dis = np.unpackbits(masks[i]).reshape(H_S, W_S).astype(bool)
        if not dis.any():
            continue
        strat, pair_id = _stratum_field(lab)
        _, cam = _realize_frame(lab, 0.0, palette, hood=hood)
        x_cam = torch.from_numpy(cam).permute(2, 0, 1).float().unsqueeze(0)
        x_cam.requires_grad_(True)
        x_s = tfun.interpolate(x_cam, size=(H_S, W_S), mode="bilinear")
        logits = model(x_s)[0]
        pred = logits.argmax(dim=0).numpy()
        # stratified picks: per (class, stratum) up to per_frame total
        picks = []
        order = [(c, s) for s in range(4) for c in range(5)]
        rng.shuffle(order)
        for c, s in order:
            m = dis & (lab == c) & (strat == s) & (pred != lab)
            ys, xs = np.where(m)
            if len(ys):
                j = rng.integers(len(ys))
                picks.append((int(ys[j]), int(xs[j]), c, s))
            if len(picks) >= per_frame:
                break
        cam_lab_up = None
        for py, px, c, s in picks:
            p_ = int(pred[py, px])
            cure = logits[c, py, px] - logits[p_, py, px]  # deficit: <0 at flip
            if x_cam.grad is not None:
                x_cam.grad = None
            cure.backward(retain_graph=True)
            gr = x_cam.grad[0].permute(1, 2, 0).detach().numpy()  # (H_C,W_C,3)
            e_tot = float((gr ** 2).sum())
            if e_tot <= 0:
                continue
            gl = gr @ LUMA_HAT
            e_luma = float((gl ** 2).sum())
            # locality around the camera-projected pixel
            cy, cx = int(py * H_C / H_S), int(px * W_C / W_S)
            loc = {}
            for r in (4, 12, 36):
                y0, y1 = max(cy - r, 0), min(cy + r + 1, H_C)
                x0, x1 = max(cx - r, 0), min(cx + r + 1, W_C)
                loc[f"e_frac_r{r}"] = float((gr[y0:y1, x0:x1] ** 2).sum() / e_tot)
            # side split at camera res (NEAREST-upsampled GT label)
            if cam_lab_up is None:
                import cv2
                cam_lab_up = cv2.resize(lab.astype(np.uint8), (W_C, H_C),
                                        interpolation=cv2.INTER_NEAREST)
            m_gt = cam_lab_up == c
            m_pr = cam_lab_up == p_
            e_gt = float((gr[m_gt] ** 2).sum() / e_tot)
            e_pr = float((gr[m_pr] ** 2).sum() / e_tot)
            gl_gt, gl_pr = gl[m_gt], gl[m_pr]
            coh_gt = float(abs(gl_gt.sum()) / (np.abs(gl_gt).sum() + 1e-12))
            coh_pr = float(abs(gl_pr.sum()) / (np.abs(gl_pr).sum() + 1e-12))
            rows.append({
                "frame": i, "y": py, "x": px, "gt": CLASSES[c], "pred": CLASSES[p_],
                "stratum": STRATA[s], "deficit_margin": float(-cure.item()),
                "luma_energy_frac": e_luma / e_tot,
                "chroma_energy_frac": 1.0 - e_luma / e_tot,
                **loc,
                "e_frac_gt_side": e_gt, "e_frac_pred_side": e_pr,
                "flat_coherence_gt_side": coh_gt, "flat_coherence_pred_side": coh_pr,
                "gt_side_shift_sign": float(np.sign(gl_gt.sum())),
                "pred_side_shift_sign": float(np.sign(gl_pr.sum())),
            })
        del logits, x_s, x_cam
        print(f"[c2] sens frame {i}: {len(picks)} samples (total {len(rows)})", flush=True)
    out = {"scope": f"frame stride {frame_stride}, {len(rows)} samples; VJP of "
                    f"logit_gt - logit_pred at DISAGREEING px of the eps0+hood palette "
                    f"render, through the REAL frozen SegNet incl. exact resize; {AXIS}",
           "rows": rows}
    with open(os.path.join(OUT_DIR, "sens.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[c2] sens: {len(rows)} rows written")


# ---------------------------------------------------------------------------
# Stage SMOKE — carrier-variant feasibility (stride subset, LABELLED)
# ---------------------------------------------------------------------------
def _band_masks(lab: np.ndarray, kpx: int = 2):
    """Per-class one-sided boundary band (<=kpx inside class) + nearest pair partner."""
    from scipy.ndimage import distance_transform_edt

    _, pair_id = _stratum_field(lab)
    bm = pair_id != 255
    dist, inds = distance_transform_edt(~bm, return_indices=True)
    near_pair = pair_id[inds[0], inds[1]]
    band = (dist <= kpx) & (near_pair != 255)
    return band, near_pair


def _apply_band_contrast(frame: np.ndarray, lab: np.ndarray, palette: np.ndarray,
                         beta: float, side_classes: set[int] | None, kpx: int = 2
                         ) -> np.ndarray:
    """Push band-pixel values AWAY from the pair partner palette (margin-restoring).

    side_classes=None -> symmetric (both sides); else only pixels whose GT class is in
    side_classes get the push (one-sided carrier). 0 counted bytes (constants only).
    """
    import cv2

    band, near_pair = _band_masks(lab, kpx)
    out = frame.astype(np.int16).copy()
    band_up = cv2.resize(band.astype(np.uint8), (W_C, H_C),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    lab_up = cv2.resize(lab.astype(np.uint8), (W_C, H_C), interpolation=cv2.INTER_NEAREST)
    pair_up = cv2.resize(near_pair, (W_C, H_C), interpolation=cv2.INTER_NEAREST)
    for c in range(5):
        m = band_up & (lab_up == c)
        if side_classes is not None:
            if c not in side_classes:
                continue
        if not m.any():
            continue
        pc = pair_up[m].astype(np.int64)
        partner = np.where(pc // 5 == c, pc % 5, pc // 5)
        delta = palette[c].astype(np.int16) - palette[partner].astype(np.int16)
        out[m] = out[m] + (beta * delta).astype(np.int16)
    return np.clip(out, 0, 255).astype(np.uint8)


def stage_smoke(variant: str, stride: int, beta: float) -> None:
    import brotli
    import cv2

    g = _load_gt(("lstars", "gt_f1"))
    lstars, gt_f1 = g["lstars"], g["gt_f1"]
    hood = _hood_seed(lstars, gt_f1)[0]
    palette = _palette()
    model = _load_segnet()
    n = lstars.shape[0]
    frames = list(range(0, n, stride))
    rows = []
    extra_bytes_n600 = 0
    t0 = time.time()
    for fi, i in enumerate(frames):
        lab = lstars[i]
        _, cam = _realize_frame(lab, 0.0, palette, hood=hood)
        by = 0
        if variant.startswith("blur"):
            sigma = float(variant[4:])
            cam = cv2.GaussianBlur(cam, (0, 0), sigma)
        elif variant in ("oneside_shallow", "oneside_deep", "symmetric",
                         "oneside_lane", "oneside_movable"):
            # shallow sides from slope.json erasure reading: fine classes Lane, Movable
            side = {"oneside_shallow": {1, 3}, "oneside_deep": {0, 2, 4},
                    "symmetric": None, "oneside_lane": {1},
                    "oneside_movable": {3}}[variant]
            cam = _apply_band_contrast(cam, lab, palette, beta, side)
        elif variant == "movable_meancolor":
            # per-frame mean RGB of the Movable class (3 B/frame counted seed)
            lab_up = cv2.resize(lab.astype(np.uint8), (W_C, H_C),
                                interpolation=cv2.INTER_NEAREST)
            mv = lab_up == 3
            if mv.any():
                small_gt = gt_f1[i]
                mean_rgb = np.clip(np.round(small_gt[mv].mean(axis=0)), 0, 255
                                   ).astype(np.uint8)
                cam[mv] = mean_rgb
                by = 3
        elif variant == "tex_global_ds16":
            small = cv2.resize(gt_f1[i], (W_C // 16, H_C // 16), interpolation=cv2.INTER_AREA)
            by = len(brotli.compress(small.tobytes(), quality=11))
            cam = cv2.resize(small, (W_C, H_C), interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(lab.astype(np.uint8), (W_C, H_C),
                           interpolation=cv2.INTER_NEAREST) == 4
            hood0, texd = hood
            mm = m & hood0
            cam[mm] = texd[mm]
        elif variant == "tex_movable_ds8":
            lab_up = cv2.resize(lab.astype(np.uint8), (W_C, H_C),
                                interpolation=cv2.INTER_NEAREST)
            mv = lab_up == 3
            if mv.any():
                ys, xs = np.where(mv)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                crop = gt_f1[i][y0:y1, x0:x1]
                small = cv2.resize(crop, (max((x1 - x0) // 8, 1), max((y1 - y0) // 8, 1)),
                                   interpolation=cv2.INTER_AREA)
                by = len(brotli.compress(small.tobytes(), quality=11)) + 8
                up = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                sub = cam[y0:y1, x0:x1]
                sub[mv[y0:y1, x0:x1]] = up[mv[y0:y1, x0:x1]]
                cam[y0:y1, x0:x1] = sub
        elif variant == "tex_band_ds4":
            band, _ = _band_masks(lab, 3)
            small_gt = cv2.resize(gt_f1[i], (W_S, H_S), interpolation=cv2.INTER_LINEAR)
            bys, bxs = np.where(band)
            vals = small_gt[bys, bxs]
            # ds4 along the band ordering (oracle byte proxy: every 4th value kept)
            by = len(brotli.compress(vals[::4].tobytes(), quality=11))
            band_up = cv2.resize(band.astype(np.uint8), (W_C, H_C),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            gt_up_vals = cv2.resize(small_gt, (W_C, H_C), interpolation=cv2.INTER_LINEAR)
            cam[band_up] = gt_up_vals[band_up]
        else:
            raise SystemExit(f"unknown variant {variant}")
        pred = _segnet_argmax(model, cam[None])[0]
        dis = pred != lab
        strat, _ = _stratum_field(lab)
        bucket = {}
        for c in range(5):
            for s in range(4):
                cnt = int((dis & (lab == c) & (strat == s)).sum())
                if cnt:
                    bucket[f"{CLASSES[c]}|{STRATA[s]}"] = cnt
        rows.append({"frame": int(i), "dseg": float(dis.mean()), "bytes": int(by),
                     "bucket": bucket})
        if fi % 20 == 0:
            print(f"[c2] smoke {variant} {fi}/{len(frames)} ({time.time() - t0:.0f}s)",
                  flush=True)
    # bytes at full n600 for counted variants (no inference needed) - skip: report subset
    dseg = float(np.mean([r["dseg"] for r in rows]))
    agg: dict[str, int] = {}
    for r in rows:
        for k, v in r["bucket"].items():
            agg[k] = agg.get(k, 0) + v
    out = {"scope": f"stride-{stride} subset ({len(frames)} frames of n600) — LABELLED "
                    f"SUBSET, not n600 evidence; reason: carrier-variant SMOKE ranking "
                    f"only; winner re-measured at n600 before any verdict; {AXIS}",
           "variant": variant, "beta": beta,
           "dseg_subset": dseg,
           "bytes_per_frame_mean": float(np.mean([r["bytes"] for r in rows])),
           "bytes_n600_extrapolated": float(np.mean([r["bytes"] for r in rows]) * n),
           "bucket_px": dict(sorted(agg.items(), key=lambda kv: -kv[1])),
           "extra_bytes_note": extra_bytes_n600}
    with open(os.path.join(OUT_DIR, f"smoke_{variant}_b{beta}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("variant", "beta", "dseg_subset",
                                          "bytes_per_frame_mean")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True,
                    choices=("decomp", "temporal", "slope", "sens", "smoke"))
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--frame-stride", type=int, default=50)
    ap.add_argument("--per-frame", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="blur2")
    ap.add_argument("--beta", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.stage == "decomp":
        stage_decomp(args.limit, args.batch)
    elif args.stage == "temporal":
        stage_temporal()
    elif args.stage == "slope":
        stage_slope(args.stride)
    elif args.stage == "sens":
        stage_sens(args.frame_stride, args.per_frame, args.seed)
    else:
        stage_smoke(args.variant, args.stride, args.beta)


if __name__ == "__main__":
    main()
