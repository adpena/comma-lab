# SPDX-License-Identifier: MIT
"""JITTER PREDICTABILITY PRECURSOR: is the bulk per-frame SegNet jitter compressible?

THE QUESTION (FEED-kb / a6b1a8b08 open door). a6b1a8b08 confirmed the clean-canonical
+ exact-pose warp prior MISSES a bulk residual (~0.0029 bulk d_seg, 2.4x the 1.23e-3
budget): the per-frame frozen-SegNet jitter. M4 found storing it UNIFORMLY as a sparse
dither costs ~178KB/600 = rate 0.1185 (~= PR95's whole archive) -> S~0.26, does NOT
close. The remaining OPEN DOOR (M4's NOT-PESSIMISTIC caveat): a TRAINED content-aware
generator that emits the jitter from a COMPACT CODE could be far cheaper -- IF the jitter
is PREDICTABLE / COMPRESSIBLE from local content. If the jitter is ~WHITE SegNet decision
noise (no exploitable structure), even a trained generator cannot beat the naive store and
sub-0.15 via this path needs rethinking.

This $0 CPU probe BOUNDS that door (NO GPU, NO training run beyond a tiny CPU logistic/
binning probe). It measures, on the per-frame bulk flip set (warp-prior argmax != GT
per-frame argmax, on bulk classes Road/Undriv/MyCar, exact-pose VOTE prior, window +/-2):

  M1 STRUCTURE      spatial + temporal autocorrelation + clustering of the flip map.
                    Structured/clustered -> compressible; white/isolated -> incompressible.
  M2 MARGIN DECOMP  flip-rate vs SegNet argmax-margin; the irreducible-coin-flip fraction
                    (near-zero margin) vs the moderate/predictable fraction.
  M3 PREDICTABILITY (DECISIVE, CROSS-VALIDATED) a tiny CPU logistic probe trained on a
                    TRAIN block of pairs and evaluated on a HELD-OUT block, in three feature
                    buckets: FREE-at-decode (geometry+vote), +MARGIN, +CONTENT (texture/color).
                    Reports held-out AUC, H(flip), H(flip|features), mutual information, and
                    the crux: I(flip; content | margin) -- can content predict WHICH ambiguous
                    boundary pixels flip, BEYOND merely knowing where the boundaries are?
  M4 TARGETED WATERFILL (operator overturn reframe) the 0.118 dither stored the WHOLE bulk
                    UNIFORMLY. Test the SAFE/UNSAFE partition: clamp the high-margin
                    temporally-stable SAFE bulk for FREE (generalizes the #139 hood-clamp to
                    all classes), store ONLY the thin UNSAFE annulus. Report SAFE fraction,
                    the targeted annulus rate vs 0.118, and budget closure -- BOTH for an
                    ORACLE (GT-margin) annulus AND a DECODE-FREE (warp-prior-boundary) annulus.
  RATE + VERDICT    conditioned jitter rate vs 0.1185 (the naive dither); OPEN/PARTIAL/CLOSED.

AUTHORITY / HONESTY FIREWALL (CLAUDE.md):
  * ``[macOS advisory / CPU-torch research-signal]`` ONLY. NOT a contest score. Pointer
    0.19110 UNMOVED. score_claim / promotable / ready_for_exact_eval_dispatch = False. MEANS.
  * d_seg = REAL argmax-disagreement vs the cached FROZEN CPU-torch SegNet argmax ``lstars``
    (``measure_segnet_argmax`` = the same preprocess/last-frame/bilinear-resize contract
    ``upstream/evaluate.py`` uses). Exact CPU-torch, NEVER MPS. A NO-FAKE self-check asserts
    ``SegNet(gt_f1) == lstars`` exactly and ABORTS rather than report a fabricated number.
  * CROSS-VALIDATION is MANDATORY: predictability is reported on a HELD-OUT block of pairs
    (contiguous split = honest, no temporal-adjacency leakage). Train-set predictability is
    NEVER reported as the answer. An interleaved split (leaky upper bound) is reported only
    as a labelled robustness side-number.
  * DECODE-REALIZABILITY is flagged per feature: FREE-at-decode (warp-prior geometry, vote
    confidence, position) vs CONTENT/ORACLE (SegNet margin, GT RGB texture/color) which a
    decoder does NOT have unless it stores/represents content. The CONTENT probe is the
    UPPER BOUND on predictability; the FREE probe is what a pure-geometry decoder achieves.

rule-118: warp/vote/homography/expmap + the polynomial annulus rasterizer are FREE
deterministic geometry (inflate.py, uncounted). The per-pair 6-DOF pose is COUNTED-but-
EXISTING. The stored jitter code / annulus residual is COUNTED. The probe itself ships
NOTHING; it only bounds whether a learned code COULD be cheap. NOT FORBIDDEN: honest
predictability measurement, NOT a smuggled per-frame argmax table.

Reuses a513372a/a23062c4/a95b0ad6/a6b1a8b08 machinery (homography/warp_labels/compose/
exact-pose). sklearn is absent -> logistic regression is a self-contained numpy IRLS;
conditional entropy also cross-checked non-parametrically by margin binning.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.measure_pose_warp_dseg import (  # noqa: E402
    CLASS_NAMES,
    NATIVE_H,
    NATIVE_W,
    SCREW_REGIME,
    intrinsics_at,
    warp_labels,
    _target_grid,
)
from tools.measure_screw_warp_through_R import fit_calibration_within_pair  # noqa: E402
from tools.measure_clean_canonical_warp_through_R import (  # noqa: E402
    compose_path_H,
    rgb_at,
    BUDGET,
    BULK_CLASSES,
)
from tools.measure_budget_gate_overturn import build_exact_step_poses  # noqa: E402

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
B0 = 37_545_489  # contest archive normalizer (25*bytes/B0 = rate term)
NAIVE_DITHER_RATE = 0.1185  # M4/FEED-kb uniform whole-bulk store (the door-closed reference)
BULK_IDX = [CLASS_NAMES.index(c) for c in BULK_CLASSES]  # [0, 2, 4]


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


# --------------------------------------------------------------------------- #
# exact-pose VOTE prediction (per regime) + vote confidence (FREE at decode).
# --------------------------------------------------------------------------- #
def vote_predict(seg_cache, t, window_radius, exact_step_poses, K_seg, Kinv_seg, grid_seg,
                 fit_params, n_frames):
    """Return (vote_am[regime], vote_conf[regime]) for ground/rotonly/identity at target t.

    vote_am = per-pixel majority argmax over warped neighbour argmaxes (the warp prior).
    vote_conf = winning-class vote fraction in [0,1] (FREE at decode; low conf = neighbours
    disagree = ambiguous). identity regime = unwarped (the static-scene/hood case)."""
    SEG_H, SEG_W = seg_cache.shape[1], seg_cache.shape[2]
    vote_am, vote_conf = {}, {}
    for regime in ("ground", "rotonly", "identity"):
        votes = np.zeros((5, SEG_H, SEG_W), dtype=np.float64)
        nv = 0
        for g in range(max(0, t - window_radius), min(n_frames, t + window_radius + 1)):
            if g == t:
                continue
            if regime == "identity":
                lab = seg_cache[g]
            else:
                Hc = compose_path_H(g, t, exact_step_poses, K_seg, Kinv_seg, fit_params, regime)
                lab, valid = warp_labels(seg_cache[g], Hc, grid_seg)
                lab = np.where(valid, lab, seg_cache[g])
            for c in range(5):
                votes[c] += (lab == c)
            nv += 1
        s = votes.sum(0)
        vote_am[regime] = votes.argmax(0).astype(np.int64)
        vote_conf[regime] = (votes.max(0) / np.maximum(s, 1.0)).astype(np.float64)
    return vote_am, vote_conf


def bulk_flip_map(vote_am, tgt):
    """Per-pixel flip indicator (target-keyed, matching a6b1a8b08 M4): a bulk pixel flips if
    its TARGET class is a bulk class and the regime-routed vote prediction disagrees."""
    H, W = tgt.shape
    flip = np.zeros((H, W), dtype=bool)
    for c in BULK_IDX:
        r = SCREW_REGIME[c]
        flip |= (tgt == c) & (vote_am[r] != c)
    return flip


# --------------------------------------------------------------------------- #
# self-contained numpy logistic regression (ridge IRLS / Newton). No sklearn.
# --------------------------------------------------------------------------- #
def fit_logistic_irls(X, y, l2=5.0, iters=40, tol=1e-7):
    """Ridge IRLS / damped Newton. Guards against near-separable divergence (clip w, halve
    overshooting steps, keep last finite iterate)."""
    n, k = X.shape
    w = np.zeros(k, dtype=np.float64)
    reg = l2 * np.ones(k)
    reg[0] = 0.0  # no ridge on the bias column
    y = y.astype(np.float64)
    W_CLIP = 30.0
    for _ in range(iters):
        z = np.clip(X @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        Wv = np.maximum(p * (1.0 - p), 1e-6)
        grad = X.T @ (p - y) + reg * w
        H = X.T @ (X * Wv[:, None])
        H[np.diag_indices_from(H)] += reg
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break
        # damped step with simple backtracking on the penalized neg-log-likelihood
        def nll(wv):
            zz = np.clip(X @ wv, -30.0, 30.0)
            return float(np.logaddexp(0.0, zz).sum() - (y * zz).sum() + 0.5 * (reg * wv * wv).sum())
        f0 = nll(w)
        damp = 1.0
        w_new = w
        for _bt in range(8):
            cand = np.clip(w - damp * step, -W_CLIP, W_CLIP)
            if np.all(np.isfinite(cand)) and nll(cand) <= f0 + 1e-9:
                w_new = cand
                break
            damp *= 0.5
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def cross_entropy_bits(p, y):
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    ce_nats = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(ce_nats.mean() / math.log(2.0))


def binary_entropy_bits(q):
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return float(-(q * math.log2(q) + (1.0 - q) * math.log2(1.0 - q)))


def auc_score(scores, y):
    y = np.asarray(y)
    n1 = int(y.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# --------------------------------------------------------------------------- #
# feature columns (FREE vs MARGIN vs CONTENT buckets).
# --------------------------------------------------------------------------- #
FEATS = [
    # name, bucket, is_continuous
    ("row_norm", "free", True),
    ("col_norm", "free", True),
    ("is_hood_prior", "free", False),
    ("vote_conf_ground", "free", True),
    ("vote_conf_rotonly", "free", True),
    ("vote_conf_identity", "free", True),
    ("sdf_prior", "free", True),
    ("margin", "margin", True),
    ("texture_grad", "content", True),
    ("rgb_r", "content", True),
    ("rgb_g", "content", True),
    ("rgb_b", "content", True),
    ("sdf_gt", "content", True),
]
FNAMES = [f[0] for f in FEATS]
BUCKET = {f[0]: f[1] for f in FEATS}
CONT = {f[0]: f[2] for f in FEATS}


def design_matrix(F, cols, mean, std):
    """Build [bias | standardized-continuous / raw-binary] for the named cols."""
    n = F["row_norm"].shape[0]
    X = [np.ones(n, dtype=np.float64)]
    for c in cols:
        v = F[c].astype(np.float64)
        if CONT[c]:
            v = (v - mean[c]) / (std[c] if std[c] > 1e-9 else 1.0)
        X.append(v)
    return np.stack(X, 1)


def run_probe(Ftr, ytr, Fte, yte, cols, l2=2.0):
    mean = {c: float(Ftr[c].mean()) for c in cols if CONT[c]}
    std = {c: float(Ftr[c].std()) for c in cols if CONT[c]}
    Xtr = design_matrix(Ftr, cols, mean, std)
    Xte = design_matrix(Fte, cols, mean, std)
    w = fit_logistic_irls(Xtr, ytr, l2=l2)
    pte = _sigmoid(Xte @ w)
    return {
        "cols": list(cols),
        "test_auc": auc_score(pte, yte),
        "test_ce_bits": cross_entropy_bits(pte, yte),
        "test_flip_rate": float(yte.mean()),
        "Hflip_marginal_bits": binary_entropy_bits(float(yte.mean())),
        "_pte": pte,  # popped before json
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--comma-pose",
                    default="experiments/results/pose_feasibility_probe/comma2k19_gt_pose_raw.npz")
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = all in cache")
    ap.add_argument("--window-radius", type=int, default=2)
    ap.add_argument("--train-frac", type=float, default=0.60, help="contiguous train fraction (held-out = rest)")
    ap.add_argument("--train-sample", type=int, default=2_000_000, help="max train pixels for logistic fit")
    ap.add_argument("--eval-sample", type=int, default=5_000_000, help="max held-out pixels for AUC/CE")
    ap.add_argument("--selfcheck-pairs", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax
    from tac.boundary_math.seg_core import load_real_segnet

    t0 = time.time()
    cache = (REPO / args.cache) if not Path(args.cache).is_absolute() else Path(args.cache)
    z = np.load(cache, allow_pickle=False)
    gt_f0 = np.asarray(z["gt_f0"]); gt_f1 = np.asarray(z["gt_f1"])
    lstars = np.asarray(z["lstars"], dtype=np.int64)
    margins = np.asarray(z["margins"], dtype=np.float32)
    posenet_poses = np.asarray(z["gt_poses"], dtype=np.float64)
    P_cache = lstars.shape[0]
    P = P_cache if not args.n_pairs else min(args.n_pairs, P_cache)
    gt_f0, gt_f1, lstars, margins, posenet_poses = (gt_f0[:P], gt_f1[:P], lstars[:P],
                                                    margins[:P], posenet_poses[:P])
    SEG_H, SEG_W = lstars.shape[1], lstars.shape[2]
    NAT_H, NAT_W = gt_f0.shape[1], gt_f0.shape[2]
    assert (NAT_H, NAT_W) == (NATIVE_H, NATIVE_W)
    n_frames = 2 * P

    cp = np.load((REPO / args.comma_pose) if not Path(args.comma_pose).is_absolute()
                 else Path(args.comma_pose), allow_pickle=True)
    pos = np.asarray(cp["frame_positions"], dtype=np.float64)
    ori = np.asarray(cp["frame_orientations"], dtype=np.float64)

    seg = load_real_segnet("cpu")

    def seg_argmax(frame_uint8_native):
        am, _ = measure_segnet_argmax(seg, np.asarray(frame_uint8_native, dtype=np.float64))
        return am

    # ---- NO-FAKE self-check ----
    scn = min(args.selfcheck_pairs, P)
    sc = {"pairs_checked": scn, "exact_matches": 0, "max_disagree_px": 0}
    for p in range(scn):
        am = seg_argmax(gt_f1[p])
        nd = int(np.count_nonzero(am != lstars[p]))
        sc["max_disagree_px"] = max(sc["max_disagree_px"], nd)
        sc["exact_matches"] += int(nd == 0)
    sc["PASS"] = bool(sc["exact_matches"] == scn and sc["max_disagree_px"] == 0)
    if not sc["PASS"]:
        raise SystemExit(f"NO-FAKE self-check FAILED (max_disagree_px={sc['max_disagree_px']}).")

    # ---- per-global-frame SegNet argmax cache ----
    print(f"[jitter] caching per-frame SegNet argmax for {n_frames} frames (CPU)...", flush=True)
    seg_cache = np.zeros((n_frames, SEG_H, SEG_W), dtype=np.int64)
    for g in range(n_frames):
        seg_cache[g] = seg_argmax(rgb_at(gt_f0, gt_f1, g))
        if (g + 1) % 48 == 0 or g == n_frames - 1:
            print(f"  ...{g + 1}/{n_frames}", flush=True)
    lstar0 = seg_cache[0::2]

    # ---- exact step poses + convention validation + calibration ----
    exact_step_poses = build_exact_step_poses(pos, ori, n_frames)
    K_seg = intrinsics_at(SEG_W, SEG_H); Kinv_seg = np.linalg.inv(K_seg); grid_seg = _target_grid(SEG_H, SEG_W)
    comma_within = np.stack([exact_step_poses[2 * p] for p in range(P)], 0)
    fit_comma = fit_calibration_within_pair(lstar0, lstars, comma_within, K_seg, Kinv_seg, grid_seg)
    fit_pn = fit_calibration_within_pair(lstar0, lstars, posenet_poses, K_seg, Kinv_seg, grid_seg)
    convention_ok = bool(fit_comma["fit_roadplane_dseg"] <= fit_pn["fit_roadplane_dseg"] * 1.05)
    fit_params = (fit_comma["s_t"], fit_comma["s_r"], fit_comma["pitch"])
    print(f"[jitter] convention_ok={convention_ok} comma={fit_comma['fit_roadplane_dseg']:.5f} "
          f"posenet={fit_pn['fit_roadplane_dseg']:.5f}", flush=True)

    # ============================================================ #
    # per-pair pass: flip map + features + structure accumulators.
    # ============================================================ #
    print(f"[jitter] flip map + features over {P} pairs (window=+/-{args.window_radius})...", flush=True)
    feat_cols = {c: [] for c in FNAMES}
    flip_list = []; pair_list = []
    # structure accumulators
    spatial_obs_adj = 0; spatial_flip = 0          # flips with >=1 flip 4-neighbour
    spatial_lag1_num = 0.0
    bulk_px_total = 0; flip_total = 0
    flip_label_counts = np.zeros(5, dtype=np.int64)
    margins_at_flips = []
    # temporal (hood = MyCar identity, ego-free) per fixed pixel across consecutive frames
    hood_flip_seq = []   # list of (flip_map_hood) per global frame index for temporal autocorr
    # full-frame per-pixel flip maps kept compactly (bool) for spatial/temporal at native seg-res
    # (n96: 96 maps * 384*512 bool = ~18MB, fine)
    flip_maps = np.zeros((P, SEG_H, SEG_W), dtype=bool)
    bulk_masks = np.zeros((P, SEG_H, SEG_W), dtype=bool)
    rows_norm = (np.arange(SEG_H)[:, None] / (SEG_H - 1)).repeat(SEG_W, 1)
    cols_norm = (np.arange(SEG_W)[None, :] / (SEG_W - 1)).repeat(SEG_H, 0)

    for p in range(P):
        t = 2 * p + 1
        tgt = lstars[p]
        vote_am, vote_conf = vote_predict(seg_cache, t, args.window_radius, exact_step_poses,
                                          K_seg, Kinv_seg, grid_seg, fit_params, n_frames)
        flip = bulk_flip_map(vote_am, tgt)
        bulk_mask = np.isin(tgt, BULK_IDX)
        flip_maps[p] = flip; bulk_masks[p] = bulk_mask
        bulk_px_total += int(bulk_mask.sum()); flip_total += int(flip.sum())
        for c in range(5):
            flip_label_counts[c] += int(((tgt == c) & flip).sum())
        margins_at_flips.append(margins[p][flip])

        # --- features on bulk candidate pixels ---
        idx = np.where(bulk_mask)
        # seg-res target RGB + texture (content)
        rgb_nat = gt_f1[p].astype(np.float64)
        gray_nat = rgb_nat.mean(2)
        zy, zx = SEG_H / NAT_H, SEG_W / NAT_W
        rgb_seg = np.stack([ndimage.zoom(rgb_nat[:, :, k], (zy, zx), order=1) for k in range(3)], 2)
        gray_seg = ndimage.zoom(gray_nat, (zy, zx), order=1)
        gx = ndimage.sobel(gray_seg, axis=1); gy = ndimage.sobel(gray_seg, axis=0)
        tex = np.hypot(gx, gy)
        # SDF to GT boundary (content) and to PRIOR boundary (free; id_vote partition)
        prior = vote_am["identity"]
        # cap distances (EDT returns inf for a degenerate single-class / no-boundary frame).
        sdf_gt = np.minimum(np.nan_to_num(ndimage.distance_transform_edt(~_boundary(tgt)),
                                          posinf=64.0), 64.0)
        sdf_prior = np.minimum(np.nan_to_num(ndimage.distance_transform_edt(~_boundary(prior)),
                                             posinf=64.0), 64.0)
        is_hood_prior = (prior == 4).astype(np.float64)

        feat_cols["row_norm"].append(rows_norm[idx])
        feat_cols["col_norm"].append(cols_norm[idx])
        feat_cols["is_hood_prior"].append(is_hood_prior[idx])
        feat_cols["vote_conf_ground"].append(vote_conf["ground"][idx])
        feat_cols["vote_conf_rotonly"].append(vote_conf["rotonly"][idx])
        feat_cols["vote_conf_identity"].append(vote_conf["identity"][idx])
        feat_cols["sdf_prior"].append(sdf_prior[idx])
        feat_cols["margin"].append(margins[p][idx].astype(np.float64))
        feat_cols["texture_grad"].append(tex[idx])
        feat_cols["rgb_r"].append(rgb_seg[:, :, 0][idx])
        feat_cols["rgb_g"].append(rgb_seg[:, :, 1][idx])
        feat_cols["rgb_b"].append(rgb_seg[:, :, 2][idx])
        feat_cols["sdf_gt"].append(sdf_gt[idx])
        flip_list.append(flip[idx])
        pair_list.append(np.full(idx[0].shape[0], p, dtype=np.int32))
        if (p + 1) % 16 == 0 or p == P - 1:
            print(f"  ...{p + 1}/{P}", flush=True)

    F = {c: np.concatenate(feat_cols[c]).astype(np.float32) for c in FNAMES}
    del feat_cols
    y_all = np.concatenate(flip_list).astype(np.float64)
    pair_all = np.concatenate(pair_list)
    bulk_flip_fraction = flip_total / max(bulk_px_total, 1)
    margins_at_flips = np.concatenate(margins_at_flips) if margins_at_flips else np.array([0.0])

    # ============================================================ #
    # M1 STRUCTURE: spatial clustering + temporal autocorrelation.
    # ============================================================ #
    structure = run_structure(flip_maps, bulk_masks, seg_cache, lstars, P, SEG_H, SEG_W)

    # ============================================================ #
    # M2 MARGIN decomposition: flip rate vs margin; irreducible vs predictable.
    # ============================================================ #
    margin_decomp = run_margin_decomp(F["margin"], y_all)

    # ============================================================ #
    # M3 PREDICTABILITY (cross-validated, contiguous held-out split).
    # ============================================================ #
    predict = run_predictability(F, y_all, pair_all, P, args, rng)

    # ============================================================ #
    # M4 TARGETED SAFE/UNSAFE WATERFILL (operator overturn).
    # ============================================================ #
    waterfill = run_waterfill(F, y_all, pair_all, P, bulk_px_total, flip_total,
                              n_frames, args.train_frac)

    elapsed = round(time.time() - t0, 1)
    out = {
        "tool": "tools/measure_jitter_predictability.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS advisory / CPU-torch research-signal]",
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110 (advisory; not a contest score)",
        "cache": str(cache.relative_to(REPO)) if str(cache).startswith(str(REPO)) else str(cache),
        "n_pairs": P, "seg_hw": [SEG_H, SEG_W], "window_radius": args.window_radius,
        "no_fake_selfcheck_segnet_gt_f1_eq_lstars": sc,
        "comma2k19_convention_validation": {
            "comma_within_fit_roadplane_dseg": fit_comma["fit_roadplane_dseg"],
            "posenet_within_fit_roadplane_dseg": fit_pn["fit_roadplane_dseg"],
            "VALIDATED": convention_ok},
        "flip_set": {
            "bulk_flip_fraction": bulk_flip_fraction,
            "a6b1a8b08_M4_reference": 0.00297,
            "bulk_px_total": bulk_px_total, "flip_total": flip_total,
            "flip_label_distribution": {CLASS_NAMES[c]: int(flip_label_counts[c]) for c in range(5)},
            "margin_at_flips": {"mean": float(margins_at_flips.mean()),
                                "median": float(np.median(margins_at_flips)),
                                "p90": float(np.percentile(margins_at_flips, 90))},
            "definition": ("target-keyed bulk flip (a6b1a8b08 M4): target class in {Road,Undriv,MyCar} "
                           "AND regime-routed exact-pose VOTE prediction disagrees. This is the residual "
                           "after the BEST geometric (clean-canonical exact-pose) prior."),
        },
        "M1_structure": structure,
        "M2_margin_decomposition": margin_decomp,
        "M3_predictability_cross_validated": predict,
        "M4_targeted_safe_unsafe_waterfill": waterfill,
        "rule_118": {
            "FREE_generic_in_inflate": "warp/vote/homography/expmap + polynomial annulus rasterizer",
            "COUNTED_existing": "per-pair 6-DOF pose (stored for d_pose; +0 marginal)",
            "COUNTED": "stored jitter code / unsafe-annulus residual",
            "not_forbidden": "honest predictability measurement, NOT a smuggled per-frame argmax table",
        },
        "assumptions": {
            "PROVEN": "flip map = real argmax-disagreement vs frozen CPU-torch SegNet (lstars); "
                      "AUC/CE/entropy on HELD-OUT pairs (contiguous split); byte counts via combinatorial "
                      "sparse-set + binary entropy (= optimal arithmetic-code length).",
            "DECODE_REALIZABILITY": "FREE features (warp-prior vote conf, position, prior-boundary SDF) are "
                                    "decode-computable; CONTENT/ORACLE features (SegNet margin, GT RGB "
                                    "texture/color, GT-boundary SDF) require representing content -> the "
                                    "CONTENT probe is an UPPER BOUND on predictability, the FREE probe is "
                                    "the pure-geometry-decoder achievable.",
            "CAVEATS": ["candidate set = GT-bulk pixels (matches M4; ~99.7% == prior-bulk); "
                        "raw temporal autocorr is ego-motion-confounded except the static hood (reported "
                        "separately, clean); margin-keyed/annulus byte costs are entropy LOWER BOUNDS "
                        "(real coder overhead higher); the rate counts ONLY the bulk jitter (lane + movables "
                        "+ canonical keyframe are ADDITIONAL); camera-res, excludes sub-874 aliasing."],
        },
        "elapsed_secs": elapsed,
    }
    # final OPEN-DOOR verdict
    out["OPEN_DOOR_VERDICT"] = open_door_verdict(out)

    out_dir = (Path(args.out_dir) if args.out_dir
               else (REPO / f"experiments/results/jitter_predictability_n{P}_r{args.window_radius}"))
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))

    _print_summary(out)
    print(f"\n[written] {out_path} (elapsed {elapsed}s)")
    return 0


def _boundary(lab):
    """4-neighbour class-change boundary mask."""
    b = np.zeros(lab.shape, dtype=bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b


# --------------------------------------------------------------------------- #
def run_structure(flip_maps, bulk_masks, seg_cache, lstars, P, H, W):
    """Spatial clustering (4-neighbour adjacency vs random) + temporal autocorrelation."""
    # spatial: among flip pixels, fraction with >=1 flip 4-neighbour, vs random expectation.
    obs_adj = 0; n_flip = 0; lag1_num = 0.0; lag1_den = 0
    for p in range(P):
        f = flip_maps[p]
        nf = int(f.sum())
        if nf == 0:
            continue
        n_flip += nf
        nbr = np.zeros(f.shape, dtype=np.int32)
        nbr[:-1, :] += f[1:, :]; nbr[1:, :] += f[:-1, :]
        nbr[:, :-1] += f[:, 1:]; nbr[:, 1:] += f[:, :-1]
        obs_adj += int((f & (nbr > 0)).sum())
        # lag-1 spatial autocorr (Moran-ish): corr(f, neighbour-mean) over bulk pixels
        bm = bulk_masks[p]
        fv = f[bm].astype(np.float64)
        nm = (nbr[bm] / 4.0)
        if fv.std() > 0 and nm.std() > 0:
            lag1_num += np.corrcoef(fv, nm)[0, 1]; lag1_den += 1
    q = n_flip / max(sum(int(b.sum()) for b in bulk_masks), 1)
    exp_adj_frac = 1.0 - (1.0 - q) ** 4  # random map, same density
    obs_adj_frac = obs_adj / max(n_flip, 1)

    # temporal: STATIC HOOD (MyCar identity regime, ego-free) -- the clean signal.
    # hood flip = target MyCar AND identity-vote != MyCar. Track per fixed pixel across frames.
    hood_lag1 = []
    hood_persist = []  # P(flip_{g+1} | flip_g) vs base rate, at fixed hood pixels
    prev_hf = None; prev_hood_mask = None
    base_hood_flips = 0; base_hood_px = 0; cond_num = 0; cond_den = 0
    for p in range(P):
        tgt = lstars[p]
        hoodmask = (tgt == 4)
        hf = flip_maps[p] & hoodmask
        base_hood_flips += int(hf.sum()); base_hood_px += int(hoodmask.sum())
        if prev_hf is not None:
            common = hoodmask & prev_hood_mask
            a = prev_hf[common].astype(np.float64); b = hf[common].astype(np.float64)
            if a.std() > 0 and b.std() > 0:
                hood_lag1.append(float(np.corrcoef(a, b)[0, 1]))
            # P(flip now | flip prev) at fixed hood pixels
            pf = prev_hf & common
            cond_den += int(pf.sum()); cond_num += int((pf & hf).sum())
        prev_hf = hf; prev_hood_mask = hoodmask
    hood_base_rate = base_hood_flips / max(base_hood_px, 1)
    cond_persist = (cond_num / cond_den) if cond_den else None

    return {
        "spatial_clustering": {
            "flip_density_bulk": q,
            "observed_adjacency_fraction": obs_adj_frac,
            "random_adjacency_fraction": exp_adj_frac,
            "clustering_ratio_obs_over_random": (obs_adj_frac / exp_adj_frac) if exp_adj_frac else None,
            "moran_lag1_spatial_autocorr_mean": (lag1_num / lag1_den) if lag1_den else None,
            "note": "ratio>>1 + positive Moran -> flips spatially clustered (compressible). "
                    "~1 + ~0 Moran -> isolated/white.",
        },
        "temporal_autocorr_static_hood": {
            "hood_flip_base_rate": hood_base_rate,
            "lag1_autocorr_mean_fixed_hood_px": float(np.mean(hood_lag1)) if hood_lag1 else None,
            "P_flip_next_given_flip_now": cond_persist,
            "persistence_ratio_over_base": (cond_persist / hood_base_rate)
            if (cond_persist and hood_base_rate) else None,
            "note": "static hood = ego-free clean temporal signal. P(flip|flip_prev)>>base -> the SAME "
                    "boundary pixels recur (temporally compressible). ~base -> white per-frame noise.",
        },
    }


def run_margin_decomp(margin, y):
    """Flip rate vs SegNet argmax-margin; irreducible coin-flip fraction vs predictable."""
    edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0, 1e9]
    rows = []
    total_flips = float(y.sum())
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (margin >= lo) & (margin < hi)
        n = int(m.sum()); nf = float(y[m].sum())
        rows.append({"margin_lo": lo, "margin_hi": (hi if hi < 1e8 else None),
                     "n_bulk_px": n, "flip_rate": (nf / n) if n else None,
                     "flip_share_of_all_flips": (nf / total_flips) if total_flips else None})
    # irreducible coin-flip band: pixels near the SegNet decision boundary where flip rate ~ a
    # coin flip. Use the lowest-margin band's flip rate as the local coin-flip probability.
    near0 = (margin < 0.05)
    near0_rate = float(y[near0].mean()) if int(near0.sum()) else None
    low = (margin < 0.5)
    low_rate = float(y[low].mean()) if int(low.sum()) else None
    low_share = float(y[low].sum() / total_flips) if total_flips else None
    return {
        "bins": rows,
        "near_zero_margin_band_margin_lt_0p05": {
            "flip_rate": near0_rate, "n_px": int(near0.sum()),
            "share_of_all_flips": float(y[near0].sum() / total_flips) if total_flips else None,
            "interpretation": "flip rate here ~ the LOCAL coin-flip probability (genuine SegNet boundary "
                              "stochasticity). If high (~0.3-0.5) these are irreducible; the SHARE tells "
                              "how much of the jitter is irreducible coin-flips vs moderate-margin (systematic).",
        },
        "low_margin_band_margin_lt_0p5": {"flip_rate": low_rate, "share_of_all_flips": low_share},
        "moderate_high_margin_flip_share": (1.0 - low_share) if low_share is not None else None,
        "note": "if most flips are at MODERATE margin (not near-zero) they are potentially systematic/"
                "predictable; if concentrated at near-zero margin they are irreducible coin-flips.",
    }


def run_predictability(F, y, pair_all, P, args, rng):
    """Cross-validated logistic probe: FREE vs +MARGIN vs +CONTENT. Contiguous held-out split."""
    n_train_pairs = max(1, int(round(P * args.train_frac)))
    tr_mask = pair_all < n_train_pairs
    te_mask = ~tr_mask

    def subsample(mask, cap):
        idx = np.where(mask)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        return idx

    tr_idx = subsample(tr_mask, args.train_sample)
    te_idx = subsample(te_mask, args.eval_sample)
    Ftr = {c: F[c][tr_idx] for c in FNAMES}; ytr = y[tr_idx]
    Fte = {c: F[c][te_idx] for c in FNAMES}; yte = y[te_idx]

    free = [c for c in FNAMES if BUCKET[c] == "free"]
    free_margin = free + [c for c in FNAMES if BUCKET[c] == "margin"]
    content = free_margin + [c for c in FNAMES if BUCKET[c] == "content"]
    margin_only = [c for c in FNAMES if BUCKET[c] == "margin"]

    probes = {}
    for name, cols in [("free_geometry", free), ("free_plus_margin", free_margin),
                       ("full_content", content), ("margin_only", margin_only)]:
        r = run_probe(Ftr, ytr, Fte, yte, cols)
        r.pop("_pte", None)
        probes[name] = r

    Hflip = probes["full_content"]["Hflip_marginal_bits"]
    ce_free = probes["free_geometry"]["test_ce_bits"]
    ce_fm = probes["free_plus_margin"]["test_ce_bits"]
    ce_content = probes["full_content"]["test_ce_bits"]
    mi_total = Hflip - ce_content                 # I(flip; all features)
    mi_content_given_margin = ce_fm - ce_content  # I(flip; content | free+margin) -- the crux
    mi_free = Hflip - ce_free

    # crux: among LOW-MARGIN (ambiguous) pixels, can CONTENT predict which flip?
    low_te = Fte["margin"] < 0.5
    low_crux = None
    if int(low_te.sum()) > 1000 and yte[low_te].sum() > 10:
        Ftr_low = {c: Ftr[c][Ftr["margin"] < 0.5] for c in FNAMES}
        ytr_low = ytr[Ftr["margin"] < 0.5]
        Fte_low = {c: Fte[c][low_te] for c in FNAMES}
        yte_low = yte[low_te]
        if ytr_low.sum() > 10:
            r_free = run_probe(Ftr_low, ytr_low, Fte_low, yte_low, free)
            r_content = run_probe(Ftr_low, ytr_low, Fte_low, yte_low,
                                  free + [c for c in FNAMES if BUCKET[c] == "content"])
            r_free.pop("_pte", None); r_content.pop("_pte", None)
            low_crux = {
                "low_margin_flip_rate_heldout": float(yte_low.mean()),
                "free_only_auc": r_free["test_auc"], "free_only_ce_bits": r_free["test_ce_bits"],
                "plus_content_auc": r_content["test_auc"], "plus_content_ce_bits": r_content["test_ce_bits"],
                "Hflip_lowmargin_bits": binary_entropy_bits(float(yte_low.mean())),
                "content_MI_within_lowmargin_bits": r_free["test_ce_bits"] - r_content["test_ce_bits"],
                "interpretation": "AMONG ambiguous low-margin pixels: does CONTENT tell which way the "
                                  "coin lands? AUC~0.5 + ~0 MI -> irreducible coin-flips (door CLOSED). "
                                  "AUC>>0.5 + MI>0 -> content predicts the outcome (door OPEN).",
            }

    # interleaved (leaky) robustness side-number
    inter_tr = (pair_all % 2 == 0); inter_te = ~inter_tr
    ii_tr = subsample(inter_tr, args.train_sample); ii_te = subsample(inter_te, args.eval_sample)
    r_inter = run_probe({c: F[c][ii_tr] for c in FNAMES}, y[ii_tr],
                        {c: F[c][ii_te] for c in FNAMES}, y[ii_te], content)
    r_inter.pop("_pte", None)

    return {
        "split": {"type": "contiguous_held_out", "train_pairs": [0, n_train_pairs],
                  "test_pairs": [n_train_pairs, P], "train_px": int(len(tr_idx)),
                  "test_px": int(len(te_idx)),
                  "note": "contiguous = honest (no temporal-adjacency leakage between train/test pairs)."},
        "Hflip_marginal_bits": Hflip,
        "probes_heldout": probes,
        "mutual_information_bits": {
            "I_flip_free_geometry": mi_free,
            "I_flip_all_features": mi_total,
            "I_flip_content_given_margin_THE_CRUX": mi_content_given_margin,
            "note": "I(flip; content | margin) is the EXPLOITABLE-BEYOND-boundary-localization signal: "
                    "can content predict the flip beyond merely knowing where boundaries/low-margin are?",
        },
        "low_margin_crux_heldout": low_crux,
        "interleaved_leaky_robustness": {"full_content_auc": r_inter["test_auc"],
                                         "full_content_ce_bits": r_inter["test_ce_bits"],
                                         "note": "LEAKY upper bound (temporally adjacent frames in both "
                                                 "splits); contiguous is the honest number."},
    }


def run_waterfill(F, y, pair_all, P, bulk_px_total, flip_total, n_frames, train_frac):
    """Operator overturn: clamp SAFE bulk free, store ONLY the UNSAFE annulus.
    ORACLE annulus = GT-margin < tau. FREE annulus = near prior-boundary / low vote-conf.
    Targeted rate vs 0.1185; budget closure. Temporal CV of the safe property."""
    margin = F["margin"]; sdf_prior = F["sdf_prior"]
    vc = np.minimum.reduce([F["vote_conf_ground"], F["vote_conf_rotonly"], F["vote_conf_identity"]])
    N = bulk_px_total
    Ftot = float(flip_total)
    scale600 = 600.0 / n_frames

    def store_bytes(U, Fu):
        """combinatorial sparse-set position bits + 1.5 label bits/flip -> bytes scaled to 600."""
        if Fu <= 0 or U <= 0:
            return 0.0
        bits_pos = Fu * math.log2(math.e * U / Fu) if Fu < U else float(U)
        bits = bits_pos + 1.5 * Fu
        return (bits / 8.0) * scale600

    def waterfill_curve(unsafe_score, thresholds, kind, decode_free):
        """unsafe = unsafe_score < thr (lower score = more unsafe). Sweep thr."""
        rows = []
        for thr in thresholds:
            unsafe = unsafe_score < thr
            U = int(unsafe.sum())
            Fu = float(y[unsafe].sum())
            Fsafe = Ftot - Fu
            safe_frac = (N - U) / N
            residual_dseg = Fsafe / N                       # safe flips not stored -> bulk d_seg
            bytes_t = store_bytes(U, Fu)
            rate = 25.0 * bytes_t / B0
            rows.append({
                "threshold": float(thr), "safe_fraction": safe_frac, "unsafe_fraction": U / N,
                "flip_capture": (Fu / Ftot) if Ftot else None,
                "residual_bulk_dseg": residual_dseg,
                "residual_within_budget": bool(residual_dseg <= BUDGET),
                "targeted_store_bytes_per600": int(round(bytes_t)),
                "targeted_rate_term": rate,
                "S_bulk_only": 100.0 * residual_dseg + rate,
            })
        # the waterfill operating point: smallest unsafe set whose residual <= budget
        feas = [r for r in rows if r["residual_within_budget"]]
        best = min(feas, key=lambda r: r["targeted_rate_term"]) if feas else None
        # the S-OPTIMAL point: min over the curve of S_bulk_only (d_seg residual + rate tradeoff)
        s_opt = min(rows, key=lambda r: r["S_bulk_only"]) if rows else None
        return {"kind": kind, "decode_free": decode_free, "curve": rows,
                "waterfill_operating_point_residual_le_budget": best,
                "S_optimal_operating_point": s_opt}

    # ORACLE annulus: GT-margin. thresholds spanning the low tail.
    oracle = waterfill_curve(margin, [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
                             "oracle_GT_margin", decode_free=False)
    # FREE annulus 1: near prior boundary (sdf_prior small = near boundary = unsafe).
    # unsafe_score = sdf_prior, unsafe = sdf_prior < thr (within thr px of a prior boundary).
    free_sdf = waterfill_curve(sdf_prior, [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 1e9],
                               "free_prior_boundary_sdf", decode_free=True)
    # FREE annulus 2: low vote-confidence (neighbours disagree = unsafe). unsafe = vc < thr.
    # invert sense: low vc is unsafe -> use unsafe_score = vc.
    free_vc = waterfill_curve(vc, [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.01],
                              "free_low_vote_confidence", decode_free=True)

    # naive uniform whole-bulk full close (M4 reference reproduction): store ALL flips over all bulk.
    naive_bytes = store_bytes(N, Ftot)
    naive_rate = 25.0 * naive_bytes / B0

    # temporal CV of the SAFE property: is "safe" (margin>=tau) on TRAIN frames still safe (low flip
    # rate) on HELD-OUT frames? -> flip rate within the safe set on held-out frames at the oracle
    # operating tau.
    n_tr = int(round(P * train_frac))
    te = pair_all >= n_tr
    op = oracle["waterfill_operating_point_residual_le_budget"]
    cv = None
    if op is not None:
        tau = op["threshold"]
        safe_te = te & (margin >= tau)
        unsafe_te = te & (margin < tau)
        cv = {
            "oracle_operating_tau": tau,
            "heldout_safe_flip_rate": float(y[safe_te].mean()) if int(safe_te.sum()) else None,
            "heldout_unsafe_flip_rate": float(y[unsafe_te].mean()) if int(unsafe_te.sum()) else None,
            "heldout_safe_fraction_of_bulk": float(safe_te.sum() / te.sum()) if int(te.sum()) else None,
            "interpretation": "if held-out SAFE flip rate stays ~0 the clamp generalizes (safe genuinely "
                              "free); the COST lives in the unsafe set.",
        }

    return {
        "naive_uniform_whole_bulk": {"full_close_bytes_per600": int(round(naive_bytes)),
                                     "rate_term": naive_rate, "M4_reference_rate": NAIVE_DITHER_RATE},
        "oracle_GT_margin_annulus": oracle,
        "free_prior_boundary_annulus": free_sdf,
        "free_low_vote_confidence_annulus": free_vc,
        "temporal_cross_validation_of_safe_property": cv,
        "note": ("ORACLE annulus needs GT margin (decoder lacks it unless content is represented) -> a "
                 "LOWER BOUND on targeted cost. FREE annuli (prior-boundary / vote-confidence) are "
                 "decode-computable -> the HONEST targeted cost. Compare targeted_rate to naive 0.1185."),
    }


def open_door_verdict(out):
    pr = out["M3_predictability_cross_validated"]
    wf = out["M4_targeted_safe_unsafe_waterfill"]
    crux = pr.get("low_margin_crux_heldout") or {}
    content_auc = pr["probes_heldout"]["full_content"]["test_auc"]
    free_auc = pr["probes_heldout"]["free_geometry"]["test_auc"]
    mi_cgm = pr["mutual_information_bits"]["I_flip_content_given_margin_THE_CRUX"]
    Hflip = pr["Hflip_marginal_bits"]
    ce_content = pr["probes_heldout"]["full_content"]["test_ce_bits"]
    ce_free = pr["probes_heldout"]["free_geometry"]["test_ce_bits"]
    # conditioned rate = content/free CE relative to marginal, scaled off the naive 0.1185
    cond_rate_content = NAIVE_DITHER_RATE * (ce_content / Hflip) if Hflip > 0 else None
    cond_rate_free = NAIVE_DITHER_RATE * (ce_free / Hflip) if Hflip > 0 else None
    low_content_auc = crux.get("plus_content_auc")
    low_mi = crux.get("content_MI_within_lowmargin_bits")

    # targeted (decode-free) best S-optimal point
    def best_pt(keys, field):
        vals = []
        for k in keys:
            op = wf[k].get(field)
            if op:
                vals.append(op)
        return vals

    free_keys = ("free_prior_boundary_annulus", "free_low_vote_confidence_annulus")
    free_sopt = best_pt(free_keys, "S_optimal_operating_point")
    targeted_free_Sbulk = min((o["S_bulk_only"] for o in free_sopt), default=None)
    targeted_free_rate = min((o["targeted_rate_term"] for o in free_sopt), default=None)
    oracle_sopt = wf["oracle_GT_margin_annulus"].get("S_optimal_operating_point")
    targeted_oracle_Sbulk = oracle_sopt["S_bulk_only"] if oracle_sopt else None

    # ---- AXIS 1: white vs structured (does ANYTHING predict the residual jitter?) ----
    st = out["M1_structure"]
    clustering = st["spatial_clustering"]["clustering_ratio_obs_over_random"]
    hood_persist_ratio = st["temporal_autocorr_static_hood"]["persistence_ratio_over_base"]
    structured = bool((clustering or 0) > 3.0 or (low_content_auc or free_auc or 0.5) > 0.70)
    axis1 = ("STRUCTURED (clustered/persistent/predictable, NOT white)" if structured
             else "~WHITE (isolated, unpredictable)")

    # ---- AXIS 2: does CONTENT add predictive power beyond FREE geometry? (the trained-generator edge) ----
    content_edge_auc = (low_content_auc - crux.get("free_only_auc", low_content_auc)) \
        if (low_content_auc is not None and crux.get("free_only_auc") is not None) else None
    content_helps = bool((low_mi is not None and low_mi > 0.02) or
                         (content_edge_auc is not None and content_edge_auc > 0.05))
    axis2 = ("CONTENT ADDS predictive power beyond free geometry -> a trained content generator helps"
             if content_helps else
             "CONTENT adds ~NOTHING beyond free warp-prior geometry -> a trained content generator gives "
             "little edge over a deterministic free-annulus store")

    # ---- AXIS 3: does the targeted waterfill close sub-0.15 for the bulk? ----
    closes = bool(targeted_free_Sbulk is not None and targeted_free_Sbulk < 0.15)
    axis3 = (f"targeted decode-FREE waterfill S_bulk_best={targeted_free_Sbulk} "
             f"({'CLOSES' if closes else 'does NOT close'} sub-0.15 for the bulk alone; lane+movables additional)")

    # ---- overall door ----
    if not structured:
        door = "CLOSED"
    elif content_helps and closes:
        door = "OPEN"
    elif (targeted_free_rate is not None and targeted_free_rate < 0.6 * NAIVE_DITHER_RATE) or content_helps:
        door = "PARTIAL"
    else:
        door = "PARTIAL_predictable_but_uncompressible_enough"

    return {
        "AXIS1_white_vs_structured": axis1,
        "AXIS2_content_generator_edge": axis2,
        "AXIS3_targeted_waterfill_closure": axis3,
        "content_full_heldout_auc": content_auc,
        "free_geometry_heldout_auc": free_auc,
        "I_flip_content_given_margin_bits": mi_cgm,
        "low_margin_free_auc": crux.get("free_only_auc"),
        "low_margin_content_auc": low_content_auc,
        "low_margin_content_edge_auc": content_edge_auc,
        "low_margin_content_MI_bits": low_mi,
        "spatial_clustering_ratio": clustering,
        "hood_temporal_persistence_ratio": hood_persist_ratio,
        "conditioned_rate_estimate": {
            "content_upper_bound": cond_rate_content,
            "free_geometry": cond_rate_free,
            "naive_uniform": NAIVE_DITHER_RATE,
            "method": "NAIVE_DITHER_RATE * (heldout CE(flip|features)/H(flip)); per-pixel arithmetic-code "
                      "length; content=upper-bound-if-content-available, free=pure-geometry-decoder.",
        },
        "targeted_waterfill": {
            "decode_free_best_rate": targeted_free_rate,
            "decode_free_best_S_bulk": targeted_free_Sbulk,
            "oracle_GT_margin_best_S_bulk": targeted_oracle_Sbulk,
            "naive_uniform_rate": NAIVE_DITHER_RATE,
        },
        "door": door,
        "summary_template": ("OPEN = a TRAINED CONTENT generator compresses the jitter enough to close "
                             "sub-0.15 (GPU run worth it); PARTIAL = structured/predictable but content "
                             "adds little OR rate only modestly reduced; CLOSED = ~white SegNet noise. "
                             "Decisive axes: (1) white vs structured, (2) content edge beyond free "
                             "geometry, (3) targeted waterfill S_bulk vs 0.15."),
    }


def _print_summary(out):
    print("\n[jitter] ===== SUMMARY =====")
    fs = out["flip_set"]
    print(f"  bulk flip fraction = {fs['bulk_flip_fraction']:.5f} (M4 ref 0.00297); "
          f"flip margin median {fs['margin_at_flips']['median']:.3f}")
    st = out["M1_structure"]
    sc = st["spatial_clustering"]; tc = st["temporal_autocorr_static_hood"]
    print(f"  [M1 structure] spatial clustering ratio (obs/random) = {sc['clustering_ratio_obs_over_random']:.2f}; "
          f"Moran lag1 = {sc['moran_lag1_spatial_autocorr_mean']:.3f}")
    print(f"               hood temporal P(flip|flip_prev)={tc['P_flip_next_given_flip_now']} "
          f"base={tc['hood_flip_base_rate']:.4f} ratio={tc['persistence_ratio_over_base']}")
    md = out["M2_margin_decomposition"]
    n0 = md["near_zero_margin_band_margin_lt_0p05"]
    print(f"  [M2 margin] near-0 (m<0.05) flip rate={n0['flip_rate']} share={n0['share_of_all_flips']}; "
          f"moderate/high-margin flip share={md['moderate_high_margin_flip_share']}")
    pr = out["M3_predictability_cross_validated"]
    p = pr["probes_heldout"]
    print(f"  [M3 predictability HELD-OUT AUC] free={p['free_geometry']['test_auc']:.3f} "
          f"+margin={p['free_plus_margin']['test_auc']:.3f} +content={p['full_content']['test_auc']:.3f}")
    print(f"      H(flip)={pr['Hflip_marginal_bits']:.4f}b  CE(flip|content)={p['full_content']['test_ce_bits']:.4f}b  "
          f"I(flip;content|margin)={pr['mutual_information_bits']['I_flip_content_given_margin_THE_CRUX']:.4f}b")
    lc = pr.get("low_margin_crux_heldout")
    if lc:
        print(f"      [CRUX low-margin] flip_rate={lc['low_margin_flip_rate_heldout']:.3f} "
              f"free_auc={lc['free_only_auc']:.3f} content_auc={lc['plus_content_auc']:.3f} "
              f"MI={lc['content_MI_within_lowmargin_bits']:.4f}b")
    wf = out["M4_targeted_safe_unsafe_waterfill"]
    print(f"  [M4 waterfill] naive uniform rate={wf['naive_uniform_whole_bulk']['rate_term']:.4f} "
          f"(M4 ref {NAIVE_DITHER_RATE})")
    for k, lab in [("oracle_GT_margin_annulus", "oracle"),
                   ("free_prior_boundary_annulus", "free-bndry"),
                   ("free_low_vote_confidence_annulus", "free-voteconf")]:
        op = wf[k].get("waterfill_operating_point_residual_le_budget")
        if op:
            print(f"      {lab:13s} targeted rate={op['targeted_rate_term']:.4f} "
                  f"safe_frac={op['safe_fraction']:.3f} capture={op['flip_capture']:.3f} "
                  f"S_bulk={op['S_bulk_only']:.4f}")
        else:
            print(f"      {lab:13s} NO feasible residual<=budget operating point")
    cv = wf.get("temporal_cross_validation_of_safe_property")
    if cv:
        print(f"      [CV safe] heldout safe flip rate={cv['heldout_safe_flip_rate']} "
              f"(safe_frac {cv['heldout_safe_fraction_of_bulk']})")
    v = out["OPEN_DOOR_VERDICT"]
    print(f"\n  >>> OPEN-DOOR VERDICT: {v['door']}")
    print(f"      AXIS1 {v['AXIS1_white_vs_structured']}")
    print(f"      AXIS2 {v['AXIS2_content_generator_edge']}")
    print(f"      AXIS3 {v['AXIS3_targeted_waterfill_closure']}")


if __name__ == "__main__":
    raise SystemExit(main())
