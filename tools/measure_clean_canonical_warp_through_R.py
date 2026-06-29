# SPDX-License-Identifier: MIT
"""CLEAN-CANONICAL warp d_seg THROUGH R + ground-frame RATE gate (the v2 budget gate).

THE QUESTION (a23062c4 / DAG FEED-jq named this as the decisive next $0 step).
a23062c4 (``tools/measure_screw_warp_through_R.py``) warped the NOISY previous
frame ``gt_f0`` through the contest R operator and got a BULK d_seg of ~0.0048
(~4x the 1.23e-3 budget). It attributed the floor to the **inter-frame SegNet
per-frame JITTER FLOOR (~0.008)** that a single-neighbour warp cannot predict,
and named the decisive open question:

    How much of that ~0.008 floor is (a) POSE-EXPLAINABLE -- removed by warping a
    CLEAN, temporally-aggregated, jitter-free canonical instead of one noisy
    neighbour -- vs (b) GENUINE per-frame SegNet noise that MUST be stored
    per-frame (the thing PR95 captures by per-pair storage, reaching ~6e-4)?

This tool builds a CLEAN canonical and runs that gate.

METHOD -- the clean canonical (TEST-1, the BUDGET gate, d_seg THROUGH R).
The driving scene EVOLVES (new road appears, lanes curve), so a single global
canonical is wrong; we build a LOCAL sliding-window canonical. For each target
pair p (target = the f1 frame, global index t=2p+1):

  1. take a short window of NEIGHBOUR frames {t-R .. t+R} EXCLUDING t itself
     (the target's own jitter must NOT leak into the canonical -- else trivial);
  2. warp every window frame's NATIVE RGB into the target frame t's view, using
     the screw per-class-regime homography composed along the per-frame ego-motion
     chain (Road->ground plane homography, sky->rotation-only, hood->identity);
  3. per-pixel MEDIAN over the (valid) warped views -> a DENOISED canonical RGB
     (the inter-frame boundary jitter averages out IF it is random per-frame);
  4. push the denoised canonical RGB through the contest R chain
     (warp@874 -> uint8@874 -> scorer bilinear-down 384 ; bicubic-up is identity
     for a camera-res witness, per a23062c4) -> the FROZEN CPU-torch SegNet ->
     argmax -> d_seg vs the target's GT ``lstars[p]`` (= ``SegNet(gt_f1[p])``).

We measure the clean-canonical-warp d_seg (TOTAL + PER-CLASS, through R) and
compare it to THREE references:
  * a23062c4's PREV-FRAME-warp (re-measured here as the single-source baseline,
    same cache + same calibration -> apples-to-apples): the ~0.0048 bulk;
  * a PER-FRAME-EXACT floor: warp/render the frame's OWN partition -> R. At native
    res this is 0 (SegNet(gt_f1)==lstars exactly); the meaningful per-frame-exact
    *carrier* floor is FEED-jk's single-SDF lane @render-192 = 5.9e-4 (cited);
  * a pre-R argmax-VOTE cross-check (warp each neighbour's cached argmax into t,
    per-pixel majority vote) -- isolates the jitter-averaging effect from RGB blur.

THE DECOMPOSITION (the decisive output):
  source_jitter (pose/aggregation-explainable, REMOVABLE) = prevwarp - canonical
  genuine_target_jitter (must STORE per-frame)            = canonical - perframe_exact
If the clean-canonical BULK drops under 1.23e-3 the v2 "bulk needs NO INR" thesis
holds (store one canonical + warp, trained INR only for the Lane). If it stays
well above budget, the residual is target-side per-frame jitter that MUST be
stored, and we QUANTIFY it per-class.

METHOD -- TEST-2 (the RATE gate, bytes; closes FEED-jm's correction).
FEED-jm measured lane centerlines at ~65KB/600 IMAGE-SPACE iid (adjacent-frame
lane IoU 0.284 -> ego-motion kills image-space temporal redundancy). The claim
under test: ego-motion-COMPENSATED (ground-frame) coding recovers that redundancy
-> far fewer bytes. We MEASURE, on the lane occupancy sequence:
  (i)   iid per-frame bytes (the FEED-jm-style baseline),
  (ii)  image-space temporal-delta bytes (NO ego-comp),
  (iii) ground-frame ego-compensated delta bytes (warp prev lane into current via
        the ground homography, then delta) -- and the adjacent IoU image-space vs
        ground-aligned. The pose stream is REUSED (already stored for d_pose; ~0
        marginal). Verdict: does (iii) approach the 0.5-5KB ground-frame target?

AUTHORITY / HONESTY FIREWALL (CLAUDE.md):
  * ``[macOS advisory / CPU-torch research-signal]`` ONLY. NOT a contest score.
    Canonical frontier pointer 0.19110 UNMOVED. score_claim/promotable=False.
  * d_seg = REAL argmax-disagreement vs the cached frozen CPU-torch SegNet argmax
    ``lstars`` (``measure_segnet_argmax`` = the SAME preprocess_input/last-frame/
    bilinear-resize contract ``upstream/evaluate.py`` uses). Exact CPU-torch,
    NEVER MPS. A NO-FAKE self-check asserts ``SegNet(gt_f1) == lstars`` exactly,
    and the tool ABORTS rather than report a fabricated number if it fails.
  * PROVEN: the measured through-R / pre-R d_seg numbers + the measured byte
    counts. INFERRED (flagged in JSON ``assumptions``): the raw PoseNet 6-vector
    column physics; the 3 fitted global calibration scalars; the INTER-PAIR
    ego-motion proxy (0.5*(pose[p]+pose[p+1]) -- only WITHIN-pair poses are stored)
    and the per-step homography composition over the window (small-motion approx).
  * It warps GT RGB (not a shipped witness RGB) -> bounds the deterministic bulk;
    the authority is realized-through-R inside the witness INR + exact CPU/CUDA
    eval on byte-closed bytes -- NOT this advisory probe. This is a MEANS.

rule-118: the plane-induced homography + expmap + per-frame-step composition +
window-median + R chain is a FREE deterministic geometric algorithm (expandable
inside inflate.py, uncounted). The per-pair 6-DOF pose is COUNTED-but-EXISTING
(already stored for d_pose). The static scene descriptor (n, d, hood-mask,
calibration, + the few stored canonical keyframes) is COUNTED. NOT FORBIDDEN:
honest geometry, not a smuggled per-frame argmax/warp table.

Reuses a513372a's screw-homography machinery (``tools/measure_pose_warp_dseg.py``)
and a23062c4's RGB-warp + within-pair calibration (``measure_screw_warp_through_R``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

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
    regime_homography,
    warp_labels,
    _target_grid,
)
from tools.measure_screw_warp_through_R import (  # noqa: E402
    warp_rgb,
    _to_uint8,
    fit_calibration_within_pair,
)

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
BUDGET = 1.23e-3                     # the v2 d_seg TOTAL budget (100*d_seg ~ 0.123)
PERFRAME_EXACT_CARRIER_FLOOR = 5.9e-4  # FEED-jk single-SDF lane @render-192 (CITED, not re-measured)
BULK_CLASSES = ["Road", "Undriv", "MyCar"]  # deterministic-screw bulk (ground+sky+hood)


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


# --------------------------------------------------------------------------- #
# Interleaved global frame sequence + per-frame-step ego-motion poses.
# Pairs are non-overlapping seq_len=2: pair p -> global frames {2p (f0), 2p+1 (f1)}.
# step_pose[g] = relative ego-motion frame g -> g+1.
#   g even  (=2p)   : WITHIN-pair f0->f1 = gt_poses[p]            (EXACT, stored).
#   g odd   (=2p+1) : INTER-pair  f1[p]->f0[p+1] = 0.5*(p,p+1)    (PROXY, not stored).
# --------------------------------------------------------------------------- #
def build_step_poses(gt_poses: np.ndarray) -> np.ndarray:
    P = gt_poses.shape[0]
    n_steps = 2 * P - 1
    steps = np.zeros((n_steps, 6), dtype=np.float64)
    for g in range(n_steps):
        if g % 2 == 0:                      # within-pair g=2p
            steps[g] = gt_poses[g // 2]
        else:                               # inter-pair g=2p+1
            p = g // 2
            steps[g] = 0.5 * (gt_poses[p] + gt_poses[p + 1]) if (p + 1) < P else gt_poses[p]
    return steps


def rgb_at(gt_f0, gt_f1, g: int) -> np.ndarray:
    """Global-frame RGB: even g -> f0[g//2], odd g -> f1[g//2]."""
    return gt_f0[g // 2] if (g % 2 == 0) else gt_f1[g // 2]


def compose_path_H(g_src: int, g_tgt: int, step_poses, K, Kinv, params, regime) -> np.ndarray:
    """Homography mapping frame g_src coords -> frame g_tgt coords (composed per step).

    Each per-step ``regime_homography(step_poses[g], ...)`` maps g->g+1. We compose
    along the path. INFERRED: composition of per-step plane-induced homographies
    approximates the homography of the composed motion (exact only in the small-motion
    limit). The window is kept small (<= +/-R steps) to bound this error.
    """
    if g_src == g_tgt or regime == "identity":
        return np.eye(3, dtype=np.float64)
    lo, hi = (g_src, g_tgt) if g_src < g_tgt else (g_tgt, g_src)
    H = np.eye(3, dtype=np.float64)
    for g in range(lo, hi):
        Hstep = regime_homography(step_poses[g], K, Kinv, params, regime)  # g -> g+1
        H = Hstep @ H                                                       # lo -> hi
    return H if g_src < g_tgt else np.linalg.inv(H)


# --------------------------------------------------------------------------- #
# RGB inverse warp returning a VALIDITY mask (NO persist fallback) -- so the
# canonical median uses ONLY geometrically-valid views per pixel.
# --------------------------------------------------------------------------- #
def warp_rgb_masked(src_hwc: np.ndarray, H: np.ndarray, tgt_grid: np.ndarray):
    Hh, Ww, C = src_hwc.shape
    flat = src_hwc.astype(np.float64).reshape(-1, C)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Hinv = np.linalg.inv(H)
        src_h = Hinv @ tgt_grid
        z = src_h[2]
        su = src_h[0] / z
        sv = src_h[1] / z
    valid = (np.isfinite(su) & np.isfinite(sv) & (z > 0)
             & (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1))
    su_c = np.clip(su, 0.0, Ww - 1)
    sv_c = np.clip(sv, 0.0, Hh - 1)
    x0 = np.floor(su_c).astype(np.int64); y0 = np.floor(sv_c).astype(np.int64)
    x1 = np.minimum(x0 + 1, Ww - 1); y1 = np.minimum(y0 + 1, Hh - 1)
    wx = (su_c - x0)[:, None]; wy = (sv_c - y0)[:, None]
    Ia = flat[y0 * Ww + x0]; Ib = flat[y0 * Ww + x1]
    Ic = flat[y1 * Ww + x0]; Id = flat[y1 * Ww + x1]
    top = Ia * (1.0 - wx) + Ib * wx
    bot = Ic * (1.0 - wx) + Id * wx
    sampled = (top * (1.0 - wy) + bot * wy).reshape(Hh, Ww, C)
    return sampled, valid.reshape(Hh, Ww)


def build_canonical_rgb(gt_f0, gt_f1, t, window_radius, step_poses,
                        K_nat, Kinv_nat, grid_nat, params, regime, n_frames, f0_persist):
    """Median over the (valid) warped neighbour views -> denoised canonical RGB at t.

    Excludes the target frame t. All-invalid pixels fall back to f0_persist (the
    natural persist = the target's own previous frame). Returns uint8 canonical.
    """
    views = []
    masks = []
    for g in range(max(0, t - window_radius), min(n_frames, t + window_radius + 1)):
        if g == t:
            continue
        Hc = compose_path_H(g, t, step_poses, K_nat, Kinv_nat, params, regime)
        rgb, valid = warp_rgb_masked(rgb_at(gt_f0, gt_f1, g).astype(np.float64), Hc, grid_nat)
        views.append(np.where(valid[:, :, None], rgb, np.nan))
        masks.append(valid)
    if not views:
        return _to_uint8(f0_persist.astype(np.float64))
    stack = np.stack(views, 0)                       # (nv, H, W, 3) with NaN where invalid
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(stack, axis=0)            # per-pixel median over valid views
    allnan = ~np.isfinite(med).any(axis=2)
    med[allnan] = f0_persist.astype(np.float64)[allnan]
    med = np.where(np.isfinite(med), med, f0_persist.astype(np.float64))
    return _to_uint8(med)


def _finalize(ne, tot):
    out = {"total": sum(ne) / max(sum(tot), 1)}
    all_tot = max(sum(tot), 1)
    area = {}
    for c in range(5):
        out[CLASS_NAMES[c]] = (ne[c] / tot[c]) if tot[c] else None
        out[CLASS_NAMES[c] + "_contrib"] = ne[c] / all_tot
        area[CLASS_NAMES[c]] = tot[c] / all_tot
    out["_area"] = area
    return out


def _bulk(d):
    return sum(d[c + "_contrib"] for c in BULK_CLASSES)


# =========================================================================== #
# TEST-1: the BUDGET gate (clean-canonical warp d_seg through R).
# =========================================================================== #
def run_test1(gt_f0, gt_f1, lstars, gt_poses, seg, measure_segnet_argmax,
              window_radius, n_pairs):
    P = n_pairs
    n_frames = 2 * P
    step_poses = build_step_poses(gt_poses)
    SEG_H, SEG_W = lstars.shape[1], lstars.shape[2]
    K_nat = intrinsics_at(NATIVE_W, NATIVE_H); Kinv_nat = np.linalg.inv(K_nat)
    grid_nat = _target_grid(NATIVE_H, NATIVE_W)
    K_seg = intrinsics_at(SEG_W, SEG_H); Kinv_seg = np.linalg.inv(K_seg)
    grid_seg = _target_grid(SEG_H, SEG_W)

    def seg_argmax(frame_uint8_native):
        am, _ = measure_segnet_argmax(seg, np.asarray(frame_uint8_native, dtype=np.float64))
        return am

    # ---- per-global-frame cached argmax (through R, unwarped). Reused by the
    #      identity-regime prev-warp AND the pre-R argmax-vote cross-check. ----
    print(f"[clean-canon] caching per-frame SegNet argmax for {n_frames} frames...", flush=True)
    seg_cache = np.zeros((n_frames, SEG_H, SEG_W), dtype=np.int64)
    for g in range(n_frames):
        seg_cache[g] = seg_argmax(rgb_at(gt_f0, gt_f1, g))
        if (g + 1) % 32 == 0 or g == n_frames - 1:
            print(f"  ...{g + 1}/{n_frames}", flush=True)
    # lstar0[p] = seg_cache[2p] = SegNet(f0); selfcheck: seg_cache[2p+1] == lstars[p]
    cache_selfcheck = int(sum(int(np.array_equal(seg_cache[2 * p + 1], lstars[p])) for p in range(P)))

    # ---- within-pair calibration (lstar0 -> lstars on Road+Lane) ----
    lstar0 = np.stack([seg_cache[2 * p] for p in range(P)], 0)
    fit = fit_calibration_within_pair(lstar0, lstars, gt_poses, K_seg, Kinv_seg, grid_seg)
    params = (fit["s_t"], fit["s_r"], fit["pitch"])
    print(f"[clean-canon] calibration fit: {fit}", flush=True)

    # accumulators (per target class c)
    naive_ne = [0] * 5; naive_tot = [0] * 5           # persist (prev frame, no warp)
    prevwarp_ne = [0] * 5; prevwarp_tot = [0] * 5     # a23062c4 single-source through-R
    canon_ne = [0] * 5; canon_tot = [0] * 5           # clean-canonical window-median through-R
    prevote_ne = [0] * 5; prevote_tot = [0] * 5       # pre-R single-source label warp
    vote_ne = [0] * 5; vote_tot = [0] * 5             # pre-R window argmax majority vote

    print(f"[clean-canon] through-R pass over {P} pairs (window_radius={window_radius})...", flush=True)
    for p in range(P):
        t = 2 * p + 1
        f0 = gt_f0[p].astype(np.float64)
        tgt = lstars[p]

        # ---------- through-R argmaxes, per regime ----------
        # prev-frame (single source f0) through R, per regime (a23062c4 style)
        prev_am = {"identity": seg_cache[2 * p]}      # f0 unwarped == cached
        for regime in ("ground", "rotonly"):
            Hp = regime_homography(gt_poses[p], K_nat, Kinv_nat, params, regime)  # f0(2p)->f1(2p+1)
            prev_am[regime] = seg_argmax(_to_uint8(warp_rgb(f0, Hp, grid_nat)))
        # clean canonical (window median) through R, per regime
        canon_am = {}
        for regime in ("ground", "rotonly", "identity"):
            cano = build_canonical_rgb(gt_f0, gt_f1, t, window_radius, step_poses,
                                       K_nat, Kinv_nat, grid_nat, params, regime,
                                       n_frames, f0)
            canon_am[regime] = seg_argmax(cano)

        # ---------- pre-R label-space (cross-check), per regime ----------
        # single-source label warp of seg_cache[2p]
        prevote_am = {"identity": seg_cache[2 * p]}
        for regime in ("ground", "rotonly"):
            Hp = regime_homography(gt_poses[p], K_seg, Kinv_seg, params, regime)
            pr, valid = warp_labels(seg_cache[2 * p], Hp, grid_seg)
            prevote_am[regime] = np.where(valid, pr, seg_cache[2 * p])
        # window majority-vote per regime (warp each neighbour argmax into t @ seg res)
        vote_am = {}
        for regime in ("ground", "rotonly", "identity"):
            votes = np.full((5, SEG_H, SEG_W), 0.0)
            for g in range(max(0, t - window_radius), min(n_frames, t + window_radius + 1)):
                if g == t:
                    continue
                if regime == "identity":
                    lab = seg_cache[g]
                else:
                    Hc = compose_path_H(g, t, step_poses, K_seg, Kinv_seg, params, regime)
                    lab, valid = warp_labels(seg_cache[g], Hc, grid_seg)
                    lab = np.where(valid, lab, seg_cache[g])
                for c in range(5):
                    votes[c] += (lab == c)
            vote_am[regime] = votes.argmax(0).astype(np.int64)

        # ---------- score per target class c under regime(c) ----------
        for c in range(5):
            r = SCREW_REGIME[c]
            m = (tgt == c)
            nc = int(m.sum())
            if not nc:
                continue
            naive_ne[c] += int(((seg_cache[2 * p] != c) & m).sum()); naive_tot[c] += nc
            prevwarp_ne[c] += int(((prev_am[r] != c) & m).sum()); prevwarp_tot[c] += nc
            canon_ne[c] += int(((canon_am[r] != c) & m).sum()); canon_tot[c] += nc
            prevote_ne[c] += int(((prevote_am[r] != c) & m).sum()); prevote_tot[c] += nc
            vote_ne[c] += int(((vote_am[r] != c) & m).sum()); vote_tot[c] += nc
        if (p + 1) % 8 == 0 or p == P - 1:
            print(f"  ...{p + 1}/{P}", flush=True)

    naive = _finalize(naive_ne, naive_tot)
    prevwarp = _finalize(prevwarp_ne, prevwarp_tot)
    canon = _finalize(canon_ne, canon_tot)
    prevote = _finalize(prevote_ne, prevote_tot)
    vote = _finalize(vote_ne, vote_tot)

    prevwarp_bulk = _bulk(prevwarp)
    canon_bulk = _bulk(canon)
    vote_bulk = _bulk(vote)
    prevote_bulk = _bulk(prevote)

    # ---- the decisive decomposition of the ~0.008 inter-frame floor ----
    # TWO parallel tracks (kept apples-to-apples within each space; do NOT subtract
    # across spaces). The pre-R label-VOTE track isolates jitter-averaging without the
    # RGB-median blur confound -> it is the FAIR upper bound on what a clean canonical
    # CAN remove. The through-R RGB track is the literal "warp a clean RGB canonical
    # through R" (pessimistic: median-of-misaligned-RGB blurs boundaries; a real witness
    # renders a SHARP partition, not a blurred RGB).
    src_removed_rgb = prevwarp_bulk - canon_bulk        # through-R RGB space
    src_removed_vote = prevote_bulk - vote_bulk         # pre-R label space (no blur)
    best_clean_bulk = min(canon_bulk, vote_bulk)        # best achievable clean canonical
    genuine_target_jitter = best_clean_bulk - PERFRAME_EXACT_CARRIER_FLOOR  # must-store residual
    pose_explainable_frac_vote = (src_removed_vote / prevote_bulk) if prevote_bulk else None
    pose_explainable_frac_rgb = (src_removed_rgb / prevwarp_bulk) if prevwarp_bulk else None

    # per-class genuine-jitter residual (what v2 must store/train beyond the lane)
    per_class_residual = {}
    for c in range(5):
        nm = CLASS_NAMES[c]
        per_class_residual[nm] = {
            "naive_persist": naive[nm],
            "prevwarp_through_R": prevwarp[nm],
            "clean_canonical_through_R": canon[nm],
            "canonical_vs_prevwarp_delta": (
                (canon[nm] - prevwarp[nm]) if (canon[nm] is not None and prevwarp[nm] is not None) else None),
            "contrib_clean_canonical": canon[nm + "_contrib"],
            "area": canon["_area"][nm],
            "regime": SCREW_REGIME[c],
        }

    verdict_bulk_free = bool(best_clean_bulk <= BUDGET)
    return {
        "window_radius": window_radius,
        "calibration_fit_within_pair": fit,
        "cache_selfcheck_seg_cache_f1_eq_lstars": {"matches": cache_selfcheck, "P": P,
                                                   "PASS": bool(cache_selfcheck == P)},
        "through_R_naive_persist": naive,
        "through_R_prevframe_warp": prevwarp,
        "through_R_clean_canonical": canon,
        "preR_prevframe_label_warp": prevote,
        "preR_clean_canonical_vote": vote,
        "bulk_terms": {
            "prevframe_warp_bulk": prevwarp_bulk,
            "clean_canonical_bulk": canon_bulk,
            "preR_prevframe_label_bulk": prevote_bulk,
            "preR_clean_canonical_vote_bulk": vote_bulk,
            "a23062c4_prevframe_bulk_reference": 0.0048,
            "budget_d_seg_total": BUDGET,
            "perframe_exact_carrier_floor_FEEDjk": PERFRAME_EXACT_CARRIER_FLOOR,
        },
        "DECOMPOSITION_of_interframe_floor": {
            "through_R_RGB_track": {
                "prevframe_warp_bulk": prevwarp_bulk,
                "clean_canonical_bulk": canon_bulk,
                "source_jitter_removed": src_removed_rgb,
                "pose_explainable_fraction": pose_explainable_frac_rgb,
                "caveat": "median-of-misaligned-RGB blurs boundaries -> pessimistic; can be NEGATIVE (blur hurts).",
            },
            "preR_label_VOTE_track": {
                "prevframe_label_bulk": prevote_bulk,
                "clean_canonical_vote_bulk": vote_bulk,
                "source_jitter_removed": src_removed_vote,
                "pose_explainable_fraction": pose_explainable_frac_vote,
                "note": "isolates jitter-averaging without RGB blur -> FAIR upper bound on removable source jitter.",
            },
            "best_clean_canonical_bulk": best_clean_bulk,
            "genuine_target_jitter_must_store": genuine_target_jitter,
            "perframe_exact_carrier_floor": PERFRAME_EXACT_CARRIER_FLOOR,
            "note": ("source_jitter_removed = prevframe - clean_canonical (pose/aggregation explainable, "
                     "removable). genuine_target_jitter = best_clean_canonical - perframe_exact_floor "
                     "(target-side per-frame SegNet jitter that NO warp captures -> must store per-frame, "
                     "the thing PR95 stores by per-pair partition storage)."),
        },
        "per_class_residual": per_class_residual,
        "VERDICT": {
            "best_clean_canonical_bulk": best_clean_bulk,
            "through_R_RGB_canonical_bulk": canon_bulk,
            "preR_vote_canonical_bulk": vote_bulk,
            "bulk_fits_under_budget": verdict_bulk_free,
            "bulk_over_budget_factor": (best_clean_bulk / BUDGET) if BUDGET else None,
            "clean_canonical_beats_prevframe_warp_RGB": bool(canon_bulk < prevwarp_bulk),
            "clean_canonical_vote_beats_prevframe_label": bool(vote_bulk < prevote_bulk),
            "summary": (
                "BULK FREE via clean-canonical warp (under budget): store one local canonical + "
                "warp; trained INR is lane-only."
                if verdict_bulk_free else
                "BULK NOT free via clean-canonical warp: even the best-case clean canonical "
                "(blur-free argmax vote) leaves a bulk residual ABOVE budget; the remainder is "
                "target-side per-frame jitter that MUST be stored per-frame; v2 needs per-frame "
                "bulk correction beyond the lane (quantified per-class)."),
        },
    }


# =========================================================================== #
# TEST-2: the RATE gate (ground-frame lane bytes vs image-space iid).
# =========================================================================== #
def _packbits_bytes(mask_bool: np.ndarray) -> bytes:
    return np.packbits(mask_bool.ravel()).tobytes()


def _comp(data: bytes, brotli_mod) -> int:
    z = len(zlib.compress(data, 9))
    if brotli_mod is not None:
        try:
            return min(z, len(brotli_mod.compress(data, quality=11)))
        except Exception:
            return z
    return z


def run_test2(lstars, lstar0_cache, gt_poses, n_pairs, scale_to=600):
    """Ground-frame ego-compensated lane coding vs image-space iid (measured bytes)."""
    try:
        import brotli as brotli_mod
    except Exception:
        brotli_mod = None
    P = n_pairs
    SEG_H, SEG_W = lstars.shape[1], lstars.shape[2]
    K_seg = intrinsics_at(SEG_W, SEG_H); Kinv_seg = np.linalg.inv(K_seg)
    grid_seg = _target_grid(SEG_H, SEG_W)
    step_poses = build_step_poses(gt_poses)

    # interleaved lane-occupancy sequence (2P consecutive frames)
    n_frames = 2 * P
    occ = np.zeros((n_frames, SEG_H, SEG_W), dtype=bool)
    for p in range(P):
        occ[2 * p] = (lstar0_cache[p] == 1)
        occ[2 * p + 1] = (lstars[p] == 1)

    # within-pair calibration on lane (reuse Road+Lane fit)
    fit = fit_calibration_within_pair(lstar0_cache, lstars, gt_poses, K_seg, Kinv_seg, grid_seg)
    params = (fit["s_t"], fit["s_r"], fit["pitch"])

    # (i) iid per-frame
    iid = sum(_comp(_packbits_bytes(occ[g]), brotli_mod) for g in range(n_frames))
    # (ii) image-space temporal XOR delta
    img_delta = _comp(_packbits_bytes(occ[0]), brotli_mod)
    for g in range(1, n_frames):
        img_delta += _comp(_packbits_bytes(occ[g] ^ occ[g - 1]), brotli_mod)
    # (iii) ground-frame ego-compensated delta: warp occ[g-1] -> g (ground), XOR
    grd_delta = _comp(_packbits_bytes(occ[0]), brotli_mod)
    iou_img, iou_grd = [], []
    for g in range(1, n_frames):
        Hc = compose_path_H(g - 1, g, step_poses, K_seg, Kinv_seg, params, "ground")
        warped, valid = warp_labels(occ[g - 1].astype(np.int64), Hc, grid_seg)
        warped_occ = np.where(valid, warped, occ[g - 1].astype(np.int64)).astype(bool)
        grd_delta += _comp(_packbits_bytes(occ[g] ^ warped_occ), brotli_mod)
        # IoU corroboration
        a, b = occ[g], occ[g - 1]
        u = (a | b).sum()
        if u:
            iou_img.append((a & b).sum() / u)
        ug = (occ[g] | warped_occ).sum()
        if ug:
            iou_grd.append((occ[g] & warped_occ).sum() / ug)

    sc = scale_to / n_frames
    return {
        "brotli_available": brotli_mod is not None,
        "n_frames_measured": n_frames,
        "scaled_to_frames": scale_to,
        "lane_repr": "occupancy mask (lstars==1) -- NOT centerline; FEED-jm's 65KB was centerline-based.",
        "bytes_measured": {
            "iid_per_frame": iid,
            "image_space_xor_delta": img_delta,
            "ground_frame_egocomp_delta": grd_delta,
        },
        "bytes_scaled_to_600": {
            "iid_per_frame": int(round(iid * sc)),
            "image_space_xor_delta": int(round(img_delta * sc)),
            "ground_frame_egocomp_delta": int(round(grd_delta * sc)),
        },
        "ratio_groundframe_over_iid": (grd_delta / iid) if iid else None,
        "ratio_groundframe_over_imagedelta": (grd_delta / img_delta) if img_delta else None,
        "adjacent_lane_IoU_image_space": float(np.mean(iou_img)) if iou_img else None,
        "adjacent_lane_IoU_ground_aligned": float(np.mean(iou_grd)) if iou_grd else None,
        "FEEDjm_image_space_anchor_bytes_per_600": 65000,
        "groundframe_target_bytes_per_600": [500, 5000],
        "pose_stream_marginal": "~0 (6 floats/frame already stored for d_pose); ground-frame coding adds only the static descriptor + keyframe canonical bytes.",
        "verdict_note": ("ground-frame ego-compensated delta vs image-space iid: the ratio + the "
                         "image-vs-ground-aligned IoU jump test FEED-jm's claim that ego-motion "
                         "(not lane non-redundancy) kills image-space temporal redundancy. "
                         "Occupancy-mask bytes are an upper bound vs a structured centerline/spline."),
        "calibration_fit": fit,
        "assumptions": {
            "inter_pair_pose_proxy": "0.5*(pose[p]+pose[p+1]) for inter-pair steps (only within-pair stored).",
            "no_global_cumulative_raster": "uses consecutive-frame ego-comp deltas (drift-robust), NOT a global bird's-eye raster (which would drift over the clip).",
            "occupancy_not_centerline": "lane as binary occupancy mask; a centerline/spline would code differently (the ratio is the robust signal).",
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = all in cache")
    ap.add_argument("--window-radius", type=int, default=2,
                    help="canonical window = +/- this many GLOBAL frames around the target (excl. target).")
    ap.add_argument("--test", choices=["1", "2", "both"], default="both")
    ap.add_argument("--selfcheck-pairs", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax
    from tac.boundary_math.seg_core import load_real_segnet

    t0 = time.time()
    cache = (REPO / args.cache) if not Path(args.cache).is_absolute() else Path(args.cache)
    z = np.load(cache, allow_pickle=False)
    gt_f0 = np.asarray(z["gt_f0"]); gt_f1 = np.asarray(z["gt_f1"])
    lstars = np.asarray(z["lstars"], dtype=np.int64)
    poses = np.asarray(z["gt_poses"], dtype=np.float64)
    P_cache = lstars.shape[0]
    P = P_cache if not args.n_pairs else min(args.n_pairs, P_cache)
    gt_f0, gt_f1, lstars, poses = gt_f0[:P], gt_f1[:P], lstars[:P], poses[:P]
    SEG_H, SEG_W = lstars.shape[1], lstars.shape[2]
    NAT_H, NAT_W = gt_f0.shape[1], gt_f0.shape[2]
    assert (NAT_H, NAT_W) == (NATIVE_H, NATIVE_W), f"native {NAT_H}x{NAT_W} != {NATIVE_H}x{NATIVE_W}"

    seg = load_real_segnet("cpu")

    def seg_argmax(frame_uint8_native):
        am, _ = measure_segnet_argmax(seg, np.asarray(frame_uint8_native, dtype=np.float64))
        return am

    # ---- NO-FAKE self-check: SegNet(gt_f1) MUST reproduce lstars exactly ----
    scn = min(args.selfcheck_pairs, P)
    selfcheck = {"pairs_checked": scn, "exact_matches": 0, "max_disagree_px": 0}
    for p in range(scn):
        am = seg_argmax(gt_f1[p])
        ndiff = int(np.count_nonzero(am != lstars[p]))
        selfcheck["max_disagree_px"] = max(selfcheck["max_disagree_px"], ndiff)
        if ndiff == 0:
            selfcheck["exact_matches"] += 1
    selfcheck["PASS"] = bool(selfcheck["exact_matches"] == scn and selfcheck["max_disagree_px"] == 0)
    if not selfcheck["PASS"]:
        raise SystemExit(
            f"NO-FAKE self-check FAILED: SegNet(gt_f1) != lstars "
            f"(max_disagree_px={selfcheck['max_disagree_px']}). Aborting rather than reporting a "
            "fabricated number.")

    out = {
        "tool": "tools/measure_clean_canonical_warp_through_R.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS advisory / CPU-torch research-signal]",
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110 (advisory measurement; not a contest score)",
        "cache": str(cache.relative_to(REPO)) if str(cache).startswith(str(REPO)) else str(cache),
        "n_pairs": P, "seg_hw": [SEG_H, SEG_W], "native_hw": [NAT_H, NAT_W],
        "no_fake_selfcheck_segnet_gt_f1_eq_lstars": selfcheck,
        "rule_118": {
            "FREE_generic_in_inflate": "plane-induced homography + expmap + per-step composition + window-median + R chain",
            "COUNTED_existing": "per-pair 6-DOF pose (already stored for d_pose; +0 marginal)",
            "COUNTED": "static scene descriptor (n,d,hood-mask,calibration) + stored canonical keyframe bytes (TEST-2)",
            "not_forbidden": "honest geometry, not a smuggled per-frame argmax/warp table",
        },
        "assumptions": {
            "PROVEN": "through-R/pre-R d_seg = real argmax-disagreement vs frozen CPU-torch SegNet argmax (lstars); byte counts are measured zlib/brotli sizes.",
            "INFERRED_pose_columns": "raw PoseNet 6-vector [fwd,lat,vert,r0,r1,r2]; col0 dominant forward.",
            "INFERRED_inter_pair_pose": "0.5*(pose[p]+pose[p+1]) -- only WITHIN-pair poses are stored; inter-pair step is a constant-velocity proxy.",
            "INFERRED_H_composition": "per-step plane-induced homographies composed over the window (small-motion approximation; window kept small).",
            "calibration": "3 global scalars (s_t,s_r,pitch) fit in label space (lstar0->lstars), applied to RGB.",
            "warps_GT_RGB": "denoises GT RGB (not a shipped witness RGB); bounds the deterministic bulk. Authority = realized-through-R inside the witness INR + exact CPU/CUDA eval.",
            "camera_res_R": "models a camera-res warp witness; excludes sub-874 bicubic-up aliasing of a low-capacity INR.",
        },
    }

    test1 = None
    if args.test in ("1", "both"):
        test1 = run_test1(gt_f0, gt_f1, lstars, poses, seg, measure_segnet_argmax,
                          args.window_radius, P)
        out["TEST1_budget_gate"] = test1

    if args.test in ("2", "both"):
        # lstar0 cache (SegNet of f0) needed for the interleaved lane f0 frames.
        print(f"[clean-canon] TEST-2: computing lstar0 (SegNet f0) for {P} pairs...", flush=True)
        lstar0_cache = np.stack([seg_argmax(gt_f0[p]) for p in range(P)], 0)
        out["TEST2_rate_gate"] = run_test2(lstars, lstar0_cache, poses, P)

    out["elapsed_secs"] = round(time.time() - t0, 1)

    out_dir = (Path(args.out_dir) if args.out_dir
               else (REPO / f"experiments/results/clean_canonical_warp_n{P}_r{args.window_radius}"))
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2))

    # ---- console summary ----
    if test1 is not None:
        bt = test1["bulk_terms"]; dc = test1["DECOMPOSITION_of_interframe_floor"]; vd = test1["VERDICT"]
        print("\n[clean-canon] === TEST-1 budget gate (through-R, n=%d, window=+/-%d) ===" % (P, args.window_radius))
        print(f"  {'class':8s} {'regime':9s} {'area':>7s} {'naive':>9s} {'prevwarp':>9s} {'canon':>9s}")
        for c in range(5):
            nm = CLASS_NAMES[c]
            nv = test1["through_R_naive_persist"][nm]
            pv = test1["through_R_prevframe_warp"][nm]
            cv = test1["through_R_clean_canonical"][nm]
            f = lambda x: (x if x is not None else float('nan'))
            print(f"  {nm:8s} {SCREW_REGIME[c]:9s} {test1['through_R_clean_canonical']['_area'][nm]:>7.3f} "
                  f"{f(nv):>9.4f} {f(pv):>9.4f} {f(cv):>9.4f}")
        rgbt = dc["through_R_RGB_track"]; vt = dc["preR_label_VOTE_track"]
        print(f"\n  [through-R RGB] prevwarp bulk={rgbt['prevframe_warp_bulk']:.5f} -> canon bulk={rgbt['clean_canonical_bulk']:.5f}"
              f"  (removed {rgbt['source_jitter_removed']:+.5f}, blur-confounded)")
        print(f"  [pre-R  vote ] prevlabel bulk={vt['prevframe_label_bulk']:.5f} -> vote bulk={vt['clean_canonical_vote_bulk']:.5f}"
              f"  (removed {vt['source_jitter_removed']:+.5f} = {(vt['pose_explainable_fraction'] or 0)*100:.0f}%)")
        print(f"  a23062c4 prevframe ref=0.0048   budget={BUDGET:.2e}   perframe-exact floor={PERFRAME_EXACT_CARRIER_FLOOR:.1e}")
        print(f"  best clean-canonical bulk={dc['best_clean_canonical_bulk']:.5f}  "
              f"genuine target jitter (must store)={dc['genuine_target_jitter_must_store']:+.5f}")
        print(f"  VERDICT: bulk_under_budget={vd['bulk_fits_under_budget']}  factor={vd['bulk_over_budget_factor']:.1f}x")
    if "TEST2_rate_gate" in out:
        t2 = out["TEST2_rate_gate"]
        print("\n[clean-canon] === TEST-2 rate gate (lane bytes, scaled to 600) ===")
        b = t2["bytes_scaled_to_600"]
        print(f"  iid per-frame        : {b['iid_per_frame']:>8d} B  (FEED-jm anchor ~65000 B)")
        print(f"  image-space xor delta: {b['image_space_xor_delta']:>8d} B")
        print(f"  ground-frame egocomp : {b['ground_frame_egocomp_delta']:>8d} B  "
              f"(target 500-5000 B)")
        print(f"  IoU image={t2['adjacent_lane_IoU_image_space']:.3f}  ground-aligned={t2['adjacent_lane_IoU_ground_aligned']:.3f}")
        print(f"  ratio ground/iid={t2['ratio_groundframe_over_iid']:.3f}")

    print(f"\n[written] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
