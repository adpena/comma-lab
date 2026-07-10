# SPDX-License-Identifier: MIT
"""NEG-AUDIT re-tests C1 (directional transfer ceiling) + E5 (fire sR vs real-ckpt flips).
$0, CPU, cached-authority artifacts + canonical measure_through_r. Advisory / NON-PROMOTABLE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/Projects/pact")
for p in (REPO, REPO / "src", REPO / "upstream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

GT = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
SR = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600_sR.npz"
MAPS = REPO / "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/annulus_live_maps/maps_BEST_ep300.npz"
STRIDE, NPAIRS_SUB = 37, 16
IDX = list(range(0, STRIDE * NPAIRS_SUB, STRIDE))  # [0,37,...,555]
TAU = 0.5  # sR build tau / trainer --margin-saliency-tau default

def pearson(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0

def topk_jaccard(score, mask_bool, k=0.05):
    thr = np.quantile(score.ravel(), 1.0 - k)
    hi = score.ravel() >= thr
    m = mask_bool.ravel()
    inter = np.logical_and(hi, m).sum()
    union = np.logical_or(hi, m).sum()
    return float(inter / union) if union > 0 else 0.0

def auc_score_vs_binary(score, y):
    # rank-based AUC = P(score(flip) > score(non-flip)); subsample non-flips for tractability
    s = score.ravel()
    y = y.ravel().astype(bool)
    pos = s[y]
    neg = s[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    if neg.size > 400_000:
        neg = rng.choice(neg, 400_000, replace=False)
    if pos.size > 400_000:
        pos = rng.choice(pos, 400_000, replace=False)
    # Mann-Whitney U via rank
    allv = np.concatenate([pos, neg])
    # average-rank (ties handled) for a Mann-Whitney-U AUC
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    starts = csum - counts
    avg = (starts + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[: pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))

def flipmass_at_topk(score, mask_bool, k):
    thr = np.quantile(score.ravel(), 1.0 - k)
    hi = score.ravel() >= thr
    m = mask_bool.ravel()
    tot = m.sum()
    return float(np.logical_and(hi, m).sum() / tot) if tot > 0 else 0.0

def main():
    out = {"axis": "[macOS-CPU advisory . NON-PROMOTABLE] . subset(16 strided pairs, non-authority)",
           "pointer": "0.19110 UNMOVED (MEANS)", "idx_stride": STRIDE, "idx": IDX}

    z = np.load(GT, mmap_mode="r")
    L = np.asarray(z["lstars"][IDX]).astype(np.int64)        # (16,384,512)
    MG = np.asarray(z["margins"][IDX]).astype(np.float32)    # (16,384,512)
    zr = np.load(SR, mmap_mode="r")
    sR = np.asarray(zr["sR"][IDX]).astype(np.float32)        # (16,384,512)
    m = np.load(MAPS)
    am = m["argmax"].astype(np.int64)                        # (16,384,512) realized verdict argmax
    gtm_maps = m["gt_margin"].astype(np.float32)

    # --- verify alignment (maps gt_margin must match cached margins at strided idx) ---
    align_ok = bool(np.allclose(gtm_maps, MG, atol=1e-2))
    out["alignment_maps_gtmargin_vs_cached"] = {"ok": align_ok,
        "maxdiff": float(np.abs(gtm_maps - MG).max())}

    # ================= E5: FIRE sR vs a REAL checkpoint's realized flips =================
    flip = (am != L)                                         # (16,384,512) bool realized flip field
    flip_frac = float(flip.mean())
    w = np.exp(-MG / TAU).astype(np.float32)                 # the PRIMARY fragility factor (already in-loop)
    gmag = np.abs(np.gradient(MG, axis=1)) + np.abs(np.gradient(MG, axis=2))  # |grad margin|

    e5 = {
        "checkpoint": "mod32cap BEST_ep300 (self-orient ON, chroma, w_pose=0), realized verdict argmax",
        "subset_realized_d_seg": round(flip_frac, 6),
        "n_flips": int(flip.sum()), "total_px": int(flip.size),
        # does sR predict the ACTUAL flips? (the first-fire question)
        "sR_vs_flip": {
            "pearson": round(pearson(sR, flip.astype(np.float32)), 4),
            "auc": round(auc_score_vs_binary(sR, flip), 4),
            "top5pct_jaccard": round(topk_jaccard(sR, flip, 0.05), 4),
            "flipmass_captured_top5pct": round(flipmass_at_topk(sR, flip, 0.05), 4),
            "flipmass_captured_top10pct": round(flipmass_at_topk(sR, flip, 0.10), 4),
        },
        # baselines for apples-to-apples
        "w_fragility_vs_flip": {  # the PRIMARY factor already in the loss (sR is the SECONDARY multiplier)
            "pearson": round(pearson(w, flip.astype(np.float32)), 4),
            "auc": round(auc_score_vs_binary(w, flip), 4),
            "top5pct_jaccard": round(topk_jaccard(w, flip, 0.05), 4),
            "flipmass_captured_top5pct": round(flipmass_at_topk(w, flip, 0.05), 4),
        },
        "neg_margin_vs_flip": {  # -margin = simplest flip-proneness signal
            "auc": round(auc_score_vs_binary(-MG, flip), 4),
            "flipmass_captured_top5pct": round(flipmass_at_topk(-MG, flip, 0.05), 4),
        },
        # sR geometry cross-checks (reproduce the anchor's theta-independent measurements)
        "sR_geometry": {
            "sR_vs_margin_pearson": round(pearson(sR, MG), 4),          # anchor: -0.323
            "sR_vs_gradmargin_pearson": round(pearson(sR, gmag), 4),    # anchor: +0.272
            "sR_vs_w_pearson": round(pearson(sR, w), 4),
        },
        "texprox_anchor_baseline": {"pearson_vs_sR": -0.033, "top5pct_jaccard_vs_sR": 0.024, "chance_jaccard": 0.026},
        "verdict_scope": "INSTANCE (16 strided pairs, ONE checkpoint) — first characterization row, NOT an n600 verdict",
    }
    out["E5_fire_sR"] = e5

    # ================= C1: directional transfer CEILING via measure_through_r =================
    # realize the PERFECT direct partition (GT L*) as palette RGB -> R -> frozen SegNet -> realized d_seg.
    # per-class mean RGB palette computed at render-grid (bilinear-down gt_f1), the fair naive-realization floor.
    import torch

    from tac.through_r import measure_through_r
    SEG_HW = (384, 512)
    frames = []
    gt_f1 = z["gt_f1"]  # (600,874,1164,3) uint8 (lazy)
    for k, i in enumerate(IDX):
        t = torch.from_numpy(np.asarray(gt_f1[i])).float().permute(2, 0, 1)[None]
        d = torch.nn.functional.interpolate(t, size=SEG_HW, mode="bilinear", align_corners=False)
        rg = d[0].permute(1, 2, 0).contiguous().numpy()  # (384,512,3) float [0,255]
        lab = L[k]                                        # (384,512)
        palette = np.zeros((5, 3), np.float32)
        for c in range(5):
            msk = lab == c
            palette[c] = rg[msk].mean(axis=0) if msk.any() else 0.0
        painted = palette[lab]                            # (384,512,3) float
        frames.append(painted.astype(np.float32))
    res = measure_through_r(frames, lstars=L, pairs="n600", input_space="render-grid",
                            allow_subset_reason="C1 transfer-ceiling calibration (16 strided pairs, non-authority)")
    out["C1_transfer_ceiling"] = {
        "realize": "GT L* palette-painted (per-class mean RGB @render-grid) -> R(bicubic-up->uint8->down) -> frozen CPU-torch SegNet",
        "perfect_direct_partition_d_seg": 0.0,
        "realized_d_seg_ceiling_F": round(res.agg_dseg, 6),
        "per_class_realized": {k: round(v, 6) for k, v in res.per_class_dseg.items()},
        "direct_partition_n600_iso_control": 0.007476,   # smoke_result.json (existing)
        "direct_partition_n600_directional": 0.003679,   # smoke_result.json (existing)
        "direct_partition_delta_pct": round(100 * (0.003679 - 0.007476) / 0.007476, 1),
        "interpretation": ("F is the realized-axis floor of a PERFECT (d_seg=0) direct partition. "
                           "If F >= the directional direct number, the -48% direct improvement lives "
                           "BELOW the realized floor -> the direct-partition axis is a mirage under naive realization."),
        "caveat": "palette=per-class-mean-RGB naive realization; a TRAINED-through-R RGB witness may beat it (chroma-slack). This is a CEILING on the naive realization, not a hard floor for the capstone.",
        "verdict_scope": "INSTANCE (16 strided pairs); reproduces FEED-ah's ~0.005-0.008 with the canonical harness",
    }

    print(json.dumps(out, indent=2))
    Path("/private/tmp/claude-501/-Users-adpena-Projects-pact/89ff112f-013d-43b5-b949-2a6d43b650c3/scratchpad/negaudit_result.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
