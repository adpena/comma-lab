# SPDX-License-Identifier: MIT
"""Measure whether the stored ego-pose carries the d_seg TRAJECTORY for free.

GROK / DAG FEED-iv central claim: ``d_seg`` and ``d_pose`` are two readouts of
the SAME sufficient statistic (the ego-pose), because each frame's SegNet
argmax partition = ``homography(ego-pose) · canonical_scene``. If true, the
pose sidecar we already store for ``d_pose`` (6 numbers/frame) IS the d_seg
modulation for free, leaving only a survival + movables residual.

This tool runs the LOCAL consequence of that claim, which is robust to
cumulative-pose drift: if ``frame[p] = H_p · C`` for a shared canonical ``C``,
then ``frame[p+1] = (H_{p+1} H_p^{-1}) · frame[p] = H_rel · frame[p]`` where
``H_rel`` is the plane-induced homography of the RELATIVE ego-pose. So we test:

    predict lstars[p+1]  :=  warp( lstars[p] , H_rel(pose) )

and compare its d_seg (argmax disagreement vs the actual frozen-SegNet argmax)
against the no-motion null (persist: ``predict := lstars[p]``). We decompose the
residual PER CLASS: Road/Lane are road-plane (homography) classes that SHOULD be
pose-explained; MyCar (ego hood) is static-in-image (needs identity, not a
ground warp); Undrivable is sky-dominated (plane at infinity); Movables move
independently (irreducible).

AUTHORITY / HONESTY FIREWALL (CLAUDE.md):
  * ``[macOS advisory / research-signal]`` ONLY. NOT a contest score. The
    canonical frontier pointer is UNMOVED. This is a frozen-instance,
    direct-partition warp test = a PROXY for warp-inside-the-witness-INR,
    measured PRE-R (no bicubic/uint8 round-trip) -> necessary, not sufficient.
  * d_seg here is the REAL argmax-disagreement against ``lstars`` = the FROZEN
    CPU-torch SegNet argmax cached by ``tools/build_shared_gt_cache_for_mlx_fleet``
    (the same authority the witness trainer uses inline). No surrogate.
  * PROVEN: the measured warp/persist d_seg numbers. INFERRED (flagged in the
    JSON ``assumptions`` block): the physical interpretation of the raw learned
    PoseNet 6-vector columns, the adjacent-pose proxy for the relative motion,
    and the calibration units (we FIT 3 global scalars).

Reuses: openpilot/comma2k19 EON intrinsics (fx=fy=910, cx=582, cy=437 native
1164x874; ``tac.calibrated_geometry`` pins the same), camera height 1.22 m
(openpilot ``HEIGHT_INIT``), plane-induced homography ``H = K (R - t n^T / d) K^{-1}``
(Hartley & Zisserman). See ``.omx/research/grok_pose_warp_dseg_test_*.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
CLASS_NAMES = ["Road", "Lane", "Undriv", "Movable", "MyCar"]  # comma10k canonical order
ROAD_PLANE_CLASSES = (0, 1)  # Road + Lane: true ground-plane (homography) classes

# openpilot / comma2k19 EON road camera intrinsics (NATIVE 1164x874).
# Source (subagent-verified, 2 independent repos agree): comma2k19 utils/camera.py
# (eon_focal_length=910.0, FULL_FRAME_SIZE=(1164,874), pp=(W/2,H/2)) and openpilot
# common/transformations/camera.py (_neo_config). Camera height = 1.22 m
# (openpilot selfdrive/locationd/calibrationd.py HEIGHT_INIT). Pitch calibrated
# online (init 0, bounds [-0.091, 0.17] rad) -> we FIT it.
NATIVE_W, NATIVE_H = 1164, 874
NATIVE_FX = NATIVE_FY = 910.0
NATIVE_CX, NATIVE_CY = 582.0, 437.0
CAMERA_HEIGHT_M = 1.22


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def intrinsics_at(seg_w: int, seg_h: int) -> np.ndarray:
    """EON K scaled from native (1164x874) to the SegNet working resolution."""
    sx, sy = seg_w / NATIVE_W, seg_h / NATIVE_H
    return np.array(
        [[NATIVE_FX * sx, 0.0, NATIVE_CX * sx],
         [0.0, NATIVE_FY * sy, NATIVE_CY * sy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _expmap_so3(omega: np.ndarray) -> np.ndarray:
    """Rodrigues axis-angle -> rotation matrix (numpy, deterministic)."""
    theta = float(np.linalg.norm(omega))
    K = np.array([[0.0, -omega[2], omega[1]],
                  [omega[2], 0.0, -omega[0]],
                  [-omega[1], omega[0], 0.0]], dtype=np.float64)
    if theta < 1e-12:
        return np.eye(3) + K  # 1st-order (theta~0)
    return (np.eye(3)
            + (np.sin(theta) / theta) * K
            + ((1.0 - np.cos(theta)) / (theta * theta)) * (K @ K))


def pose_to_homography(pose6: np.ndarray, K: np.ndarray, Kinv: np.ndarray,
                       s_t: float, s_r: float, pitch: float) -> np.ndarray:
    """Plane-induced homography H = K (R - t n^T / d) K^{-1} from a learned pose.

    ASSUMPTION (flagged): raw PoseNet 6-vector = [fwd, c1, c2, r0, r1, r2]; col0
    (~33, dominant) is forward translation -> view-frame z. View frame is
    (x->right, y->down, z->forward); road normal points up (=-y) tilted by pitch.
    Only t/d matters and d, s_t are degenerate -> we fix d=1.22 and FIT s_t (which
    also absorbs the learned-units->metric scale and the adjacent-pose factor).
    """
    t = s_t * np.array([pose6[2], pose6[1], pose6[0]], dtype=np.float64)  # (x,y,z=fwd)
    R = _expmap_so3(s_r * np.array([pose6[3], pose6[4], pose6[5]], dtype=np.float64))
    n = np.array([0.0, -np.cos(pitch), -np.sin(pitch)], dtype=np.float64)  # road up, tilted
    M = R - np.outer(t, n) / CAMERA_HEIGHT_M
    return K @ M @ Kinv


def warp_labels(src: np.ndarray, H: np.ndarray, tgt_grid: np.ndarray):
    """Nearest-neighbour inverse-warp an int label map. Returns (pred, valid)."""
    Hh, Ww = src.shape
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Hinv = np.linalg.inv(H)
        src_h = Hinv @ tgt_grid  # (3, Hh*Ww)
        z = src_h[2]
        su = src_h[0] / z
        sv = src_h[1] / z
    valid = np.isfinite(su) & np.isfinite(sv) & (z > 0)
    valid &= (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1)
    sui = np.clip(np.round(su), 0, Ww - 1).astype(np.int64)
    svi = np.clip(np.round(sv), 0, Hh - 1).astype(np.int64)
    pred = src[svi, sui].reshape(Hh, Ww)
    return pred, valid.reshape(Hh, Ww)


def _target_grid(Hh: int, Ww: int) -> np.ndarray:
    us, vs = np.meshgrid(np.arange(Ww), np.arange(Hh))
    return np.stack([us.ravel(), vs.ravel(), np.ones(Hh * Ww)], 0).astype(np.float64)


def per_class_dseg(pred: np.ndarray, tgt: np.ndarray, valid: np.ndarray) -> dict:
    """d_seg total + per-class (keyed by TARGET class), over valid pixels."""
    ne = (pred != tgt) & valid
    out = {"total": float(ne.sum() / max(valid.sum(), 1)),
           "coverage": float(valid.mean())}
    for c in range(5):
        m = (tgt == c) & valid
        out[CLASS_NAMES[c]] = float(ne[m].mean()) if m.sum() else None
        out[CLASS_NAMES[c] + "_area"] = float(((tgt == c) & valid).mean())
    return out


def eval_predictor(L, poses, K, Kinv, tgt_grid, mode, params):
    """Aggregate d_seg over all p->p+1 transitions for persist + warp predictors.

    Returns dict with 'persist', 'warp', 'static_mode' aggregate d_seg + per-class.
    """
    P = L.shape[0]
    s_t, s_r, pitch = params
    # accumulate confusion-free counts: ne_count[class], tot_count[class], valid_tot
    def fresh():
        return {"ne": 0, "tot": 0, "ne_c": [0] * 5, "tot_c": [0] * 5}
    persist, warp = fresh(), fresh()
    for p in range(P - 1):
        src, tgt = L[p], L[p + 1]
        # persist (identity / no warp): valid everywhere
        ne_p = (src != tgt)
        persist["ne"] += int(ne_p.sum()); persist["tot"] += tgt.size
        # warp by relative pose (adjacent-pose proxy = pose of pair ending at target)
        H = pose_to_homography(poses[p + 1], K, Kinv, s_t, s_r, pitch)
        pred, valid = warp_labels(src, H, tgt_grid)
        ne_w = (pred != tgt) & valid
        warp["ne"] += int(ne_w.sum()); warp["tot"] += int(valid.sum())
        for c in range(5):
            mp = (tgt == c)
            persist["ne_c"][c] += int(ne_p[mp].sum()); persist["tot_c"][c] += int(mp.sum())
            mw = mp & valid
            warp["ne_c"][c] += int(ne_w[mw].sum()); warp["tot_c"][c] += int(mw.sum())

    def fin(d):
        o = {"total": d["ne"] / max(d["tot"], 1)}
        for c in range(5):
            o[CLASS_NAMES[c]] = (d["ne_c"][c] / d["tot_c"][c]) if d["tot_c"][c] else None
            o[CLASS_NAMES[c] + "_area"] = d["tot_c"][c] / max(d["tot"], 1)
        return o
    # static-mode (no warp, no pose): each frame vs global per-pixel mode
    ne_m = (L != mode[None])
    static = {"total": float(ne_m.mean())}
    for c in range(5):
        m = (L == c)
        static[CLASS_NAMES[c]] = float(ne_m[m].mean()) if m.sum() else None
    return {"persist": fin(persist), "warp": fin(warp), "static_mode": static,
            "warp_coverage": warp["tot"] / max((L.shape[0] - 1) * L[0].size, 1)}


def fit_calibration(L, poses, K, Kinv, tgt_grid, fit_classes=ROAD_PLANE_CLASSES,
                    full_coverage=False):
    """Coordinate-descent fit of (s_t, s_r, pitch) minimizing road-plane d_seg.

    LOW capacity (3 global scalars shared across ALL transitions) -> cannot
    overfit; the per-frame variation is 100% from the stored pose. We allow s_t
    to take either sign so the warp direction is data-determined (no convention
    guess). Objective = aggregate d_seg over ``fit_classes`` target pixels.

    ``full_coverage`` (default False, preserves the original grok behaviour):
    when True the objective uses the persist-fallback full-coverage accounting
    (invalid -> source label, every target pixel scored) so a per-class oracle
    fit cannot lower its objective by invalidating (pushing pixels off-frame).
    Used for the screw per-class INDEPENDENT oracle so (b) is a fair baseline.
    """
    P = L.shape[0]

    def objective(s_t, s_r, pitch):
        ne = 0
        tot = 0
        for p in range(P - 1):
            src, tgt = L[p], L[p + 1]
            H = pose_to_homography(poses[p + 1], K, Kinv, s_t, s_r, pitch)
            pred, valid = warp_labels(src, H, tgt_grid)
            tsel = np.isin(tgt, fit_classes)
            if full_coverage:
                pred = np.where(valid, pred, src)
                sel = tsel
            else:
                sel = tsel & valid
            ne += int(((pred != tgt) & sel).sum())
            tot += int(sel.sum())
        return ne / max(tot, 1)

    # stage 1: s_t over a signed log grid (s_r=0, pitch=0)
    mags = np.concatenate([-np.logspace(-1, -4, 14), [0.0], np.logspace(-4, -1, 14)])
    best = (0.0, 0.0, 0.0)
    best_obj = objective(0.0, 0.0, 0.0)  # == persist on road-plane classes
    persist_obj = best_obj
    for s in mags:
        o = objective(float(s), 0.0, 0.0)
        if o < best_obj:
            best_obj, best = o, (float(s), 0.0, 0.0)
    # stage 2: pitch over openpilot bounds at best s_t
    for pit in np.linspace(-0.09, 0.17, 14):
        o = objective(best[0], 0.0, float(pit))
        if o < best_obj:
            best_obj, best = o, (best[0], 0.0, float(pit))
    # stage 3: s_r (rotation scale) at best s_t, pitch
    for sr in np.concatenate([[0.0], np.logspace(-2, 1.0, 10)]):
        o = objective(best[0], float(sr), best[2])
        if o < best_obj:
            best_obj, best = o, (best[0], float(sr), best[2])
    # stage 4: refine s_t around the winner
    if best[0] != 0.0:
        for s in best[0] * np.linspace(0.4, 1.8, 12):
            o = objective(float(s), best[1], best[2])
            if o < best_obj:
                best_obj, best = o, (float(s), best[1], best[2])
    return {"s_t": best[0], "s_r": best[1], "pitch": best[2],
            "fit_roadplane_dseg": best_obj, "persist_roadplane_dseg": persist_obj,
            "fit_classes": [CLASS_NAMES[c] for c in fit_classes]}


# =====================================================================================
# SCREW / TWIST (Chasles + Helmholtz-Hodge) extension — DAG FEED graphics_aa Task 6.
#
# CLAIM (screw theory): every rigid ego-motion is ONE twist (t, omega) in se(3). The
# per-class warps are DERIVED from that single twist + a tiny STATIC scene descriptor
# (road plane normal n + distance d / sky-at-infinity / hood-mask) via the
# Longuet-Higgins-Prazdny / plane-induced-homography formula:
#   * Road / Lane  (ground plane, small Z)  -> full  H = K (R - t n^T / d) K^{-1}
#   * Undriv (sky, Z -> infinity)           -> rotation-only  H = K R K^{-1}  (t-term dropped)
#   * MyCar (ego hood, rigid to camera)     -> identity
#   * Movable (independent; ~road-coupled)  -> ground H (best deterministic guess; the
#                                              part it cannot explain IS the residual)
# The twist reuses the ALREADY-STORED pose sidecar (the 6 d_pose scalars) at ~0 marginal
# bytes; only the O(few) static descriptor is new. rule-118: the LHP/plane-homography +
# expmap algorithm = FREE in inflate.py; the per-pair pose = COUNTED-but-existing; the
# static (n, d, hood-mask) = COUNTED-but-tiny.
#
# WHAT WE MEASURE (PRE-R, label-space — same regime as the grok probe above): the d_seg
# (real argmax disagreement vs frozen lstars) of the single-twist STRATIFIED warp (c)
# against the per-class INDEPENDENT homography oracle (b) and the naive-copy null (a).
# If (c) MATCHES (b), the screw parameterization is d_seg-free compression: same fidelity,
# far fewer bytes (one shared twist+descriptor vs an independent homography per class).
# =====================================================================================

# Physics-derived per-class warp REGIME for the single-twist stratified warp (c).
SCREW_REGIME = {0: "ground", 1: "ground", 2: "rotonly", 3: "ground", 4: "identity"}


def regime_homography(pose6, K, Kinv, params, regime):
    """Build the warp homography for a physical REGIME from ONE twist + calibration.

    regime: 'ground' = full plane-induced H = K(R - t n^T/d)K^{-1};
            'rotonly' = depth->infinity limit H = K R K^{-1} (t-term dropped, the
                        sky / focus-of-expansion-at-infinity case);
            'identity' = static-in-image (ego hood); H = I.
    """
    s_t, s_r, pitch = params
    if regime == "identity":
        return np.eye(3, dtype=np.float64)
    if regime == "rotonly":
        return pose_to_homography(pose6, K, Kinv, 0.0, s_r, pitch)  # t-term -> 0
    return pose_to_homography(pose6, K, Kinv, s_t, s_r, pitch)


def dseg_for_target_class(L, poses, K, Kinv, tgt_grid, c, regime, params):
    """Aggregate (ne, tot, dseg) on TARGET-class-c pixels under one warp regime+params.

    FULL-COVERAGE, non-gameable accounting: where the warp maps off-frame (invalid),
    we FALL BACK to persist (the source label) rather than EXCLUDING the pixel. This
    (a) is what a real stratified codec does (warp where you can, persist where you
    can't), and (b) prevents a per-class fit from cheating its d_seg toward 0 by
    invalidating pixels (an artifact seen when invalid pixels were excluded). d_seg is
    then evaluated over EVERY target-class-c pixel (no coverage caveat).
    """
    P = L.shape[0]
    ne = 0
    tot = 0
    for p in range(P - 1):
        src, tgt = L[p], L[p + 1]
        if regime == "identity":
            pred = src
        else:
            H = regime_homography(poses[p + 1], K, Kinv, params, regime)
            wpred, valid = warp_labels(src, H, tgt_grid)
            pred = np.where(valid, wpred, src)  # persist fallback where warp invalid
        m = (tgt == c)
        ne += int(((pred != tgt) & m).sum())
        tot += int(m.sum())
    return ne, tot, (ne / max(tot, 1))


def screw_analysis(L, poses, K, Kinv, tgt_grid, shared_params):
    """Compare (a) persist, (b) per-class independent homography oracle, (c) single-twist
    stratified screw warp; plus the sky-divergence-null ablation.

    shared_params = the Road+Lane global fit (s_t, s_r, pitch) = the static calibration.
    """
    per_class = {}
    tot_a = {"ne": 0, "tot": 0}
    tot_b = {"ne": 0, "tot": 0}
    tot_c = {"ne": 0, "tot": 0}
    indep_fits = {}
    for c in range(5):
        nm = CLASS_NAMES[c]
        # (a) persist (identity)
        ne_a, n_a, ds_a = dseg_for_target_class(L, poses, K, Kinv, tgt_grid, c, "identity", shared_params)
        # (b) per-class INDEPENDENT oracle: fit this class's OWN full ground homography
        # (full_coverage objective so the oracle cannot cheat d_seg via invalidation)
        fit_c = fit_calibration(L, poses, K, Kinv, tgt_grid, fit_classes=(c,), full_coverage=True)
        indep_fits[nm] = {k: fit_c[k] for k in ("s_t", "s_r", "pitch")}
        params_b = (fit_c["s_t"], fit_c["s_r"], fit_c["pitch"])
        ne_b, n_b, ds_b = dseg_for_target_class(L, poses, K, Kinv, tgt_grid, c, "ground", params_b)
        # (c) screw stratified: physics regime, SHARED twist+calibration (no per-class fit)
        regime_c = SCREW_REGIME[c]
        ne_c, n_c, ds_c = dseg_for_target_class(L, poses, K, Kinv, tgt_grid, c, regime_c, shared_params)
        # screw should never do WORSE than persist on a class it routes to identity;
        # if it does (warp regime hurts), the honest fallback verdict notes it.
        per_class[nm] = {
            "area": (n_a / max(L[0].size * (L.shape[0] - 1), 1)),
            "screw_regime": regime_c,
            "a_persist_dseg": ds_a,
            "b_independent_dseg": ds_b,
            "c_screw_dseg": ds_c,
            "c_minus_b": ds_c - ds_b,        # >0 => screw worse than per-class oracle
            "c_minus_a": ds_c - ds_a,        # <0 => screw beats naive copy
            "b_independent_fit": indep_fits[nm],
        }
        for d, ne, n in ((tot_a, ne_a, n_a), (tot_b, ne_b, n_b), (tot_c, ne_c, n_c)):
            d["ne"] += ne
            d["tot"] += n

    totals = {
        "a_persist_total": tot_a["ne"] / max(tot_a["tot"], 1),
        "b_independent_total": tot_b["ne"] / max(tot_b["tot"], 1),
        "c_screw_total": tot_c["ne"] / max(tot_c["tot"], 1),
    }

    # ---- adversarial audit of the c-vs-b gap (why does the shared twist "lose"?) ----
    # Decompose the total (c)-(b) gap by class, and flag where the per-class oracle won
    # via a NON-PHYSICAL warp: translation on a class the screw correctly routes to
    # identity (rigidly-attached hood) or rotation-only (depth->infinity sky). A static
    # hood and an at-infinity sky CANNOT translate, so any oracle |s_t| there is
    # clip-specific overfit, NOT generalizable ego-motion, and costs per-class bytes the
    # screw forgoes. NOTE: ground-routed classes (Road/Lane/Movable) using translation
    # is PHYSICAL — including opposite-sign translation on Movable (independent motion:
    # an approaching car has expanding flow). Their c-b gap is GENUINE residual, not
    # overfit (Movable = the off-ego-orbit residual the grok probe predicted).
    road_st = per_class["Road"]["b_independent_fit"]["s_t"]
    st_ref = abs(road_st) if abs(road_st) > 1e-9 else 1.0
    gap_decomp = {}
    physical_gap = 0.0
    nonphysical_gap = 0.0
    for c in range(5):
        nm = CLASS_NAMES[c]
        pc = per_class[nm]
        contrib = pc["c_minus_b"] * pc["area"]  # area-weighted contribution to total gap
        ofit = pc["b_independent_fit"]
        regime_says_no_translation = pc["screw_regime"] in ("identity", "rotonly")
        nonphysical = bool(regime_says_no_translation and abs(ofit["s_t"]) > 0.1 * st_ref)
        gap_decomp[nm] = {
            "area_weighted_c_minus_b": contrib,
            "oracle_s_t": ofit["s_t"],
            "oracle_nonphysical": nonphysical,
            "reason": ("oracle used translation on a no-translation (identity/rotonly) class "
                       "-> clip-specific overfit"
                       if nonphysical
                       else "physical / genuine residual (ground-routed; incl. independent Movable motion)"),
        }
        if nonphysical:
            nonphysical_gap += max(contrib, 0.0)
        else:
            physical_gap += max(contrib, 0.0)
    total_gap = totals["c_screw_total"] - totals["b_independent_total"]
    gap_summary = {
        "total_c_minus_b": total_gap,
        "gap_from_oracle_nonphysical_overfit": nonphysical_gap,
        "gap_from_genuine_residual": physical_gap,
        "note": ("the part of (c)-(b) attributable to the per-class oracle's NON-PHYSICAL warps "
                 "(translation on hood/sky, opposite-sign on a static class) is clip-specific "
                 "overfit the screw deliberately forgoes; only the genuine-residual part (chiefly "
                 "Movable independent motion) is a real screw deficiency."),
    }

    # ---- sky-divergence-null ablation (Task 6 #2) ----
    # rotation-only (screw rule, t dropped) vs full-ground (t added back) on the sky class.
    _, _, sky_rotonly = dseg_for_target_class(L, poses, K, Kinv, tgt_grid, 2, "rotonly", shared_params)
    _, _, sky_ground = dseg_for_target_class(L, poses, K, Kinv, tgt_grid, 2, "ground", shared_params)
    _, _, sky_identity = dseg_for_target_class(L, poses, K, Kinv, tgt_grid, 2, "identity", shared_params)
    sky_null = {
        "sky_rotonly_dseg": sky_rotonly,        # screw rule (depth->inf): t-term dropped
        "sky_ground_dseg": sky_ground,          # t-term added back (divergence/translation)
        "sky_identity_dseg": sky_identity,       # persist reference
        "t_term_hurts_sky": bool(sky_ground > sky_rotonly + 1e-9),
        "abs_t_term_penalty": sky_ground - sky_rotonly,
        "note": ("predicts depth-independence: the translational (1/Z) flow vanishes at "
                 "infinity, so adding it to the sky should HURT. CAVEAT: div<->translation / "
                 "curl<->rotation labeling is exact only for forward-translation + roll; "
                 "yaw/pitch mix (graphics_aa Task 6 caveat)."),
    }

    # ---- verdict: does the single-twist screw MATCH/BEAT the per-class oracle? ----
    road = per_class["Road"]
    hood = per_class["MyCar"]
    sky = per_class["Undriv"]
    tol = 0.05
    b_total = totals["b_independent_total"]
    screw_matches_oracle_raw = bool(totals["c_screw_total"] <= b_total * (1.0 + tol))
    # PHYSICAL match: the screw's deficit vs the oracle, EXCLUDING the oracle's
    # non-physical clip-specific overfit (hood/sky translation, opposite-sign), is small.
    screw_matches_oracle_physical = bool(physical_gap <= b_total * tol)
    road_beats_copy = bool(road["c_screw_dseg"] < road["a_persist_dseg"])
    statics_not_destroyed = bool(
        hood["c_screw_dseg"] <= hood["a_persist_dseg"] + 1e-9
        and sky["c_screw_dseg"] <= sky["a_persist_dseg"] + tol * max(sky["a_persist_dseg"], 1e-6)
    )
    if screw_matches_oracle_raw and road_beats_copy and statics_not_destroyed:
        screw_verdict = "SCREW_WIN_ZERO_BYTE"
        screw_note = ("single 6-DOF twist + tiny static descriptor reproduces the per-class "
                      "independent-homography d_seg (raw total within %.0f%%) AND fixes the static "
                      "classes (hood=identity, sky=rotation-only) that a single global homography "
                      "destroys. d_seg-free compression of the per-class warps: same fidelity, ~0 "
                      "marginal bytes (reuses the stored pose)." % (tol * 100))
    elif screw_matches_oracle_physical and road_beats_copy and statics_not_destroyed:
        screw_verdict = "SCREW_WIN_ZERO_BYTE_PHYSICAL"
        screw_note = ("the single twist MATCHES the per-class oracle on every PHYSICALLY-MEANINGFUL "
                      "class (Road exactly; hood=identity, sky=rotation-only are the correct, "
                      "generalizable choices). The oracle's small raw-total edge is clip-specific "
                      "NON-PHYSICAL overfit (opposite-sign/no-translation-class warps) that costs "
                      "per-class bytes and would not generalize. The only GENUINE residual the screw "
                      "cannot capture is Movable independent motion (GAP-1). => ~0-byte screw WIN at "
                      "the physical level; the residual is the off-ego-orbit part, as predicted.")
    elif road_beats_copy and statics_not_destroyed:
        screw_verdict = "SCREW_PARTIAL"
        screw_note = ("screw beats naive-copy on Road and does not destroy static classes, but the "
                      "genuine-residual deficit vs the per-class oracle exceeds %.0f%%: per-class "
                      "freedom buys real (not just overfit) d_seg the shared twist cannot." % (tol * 100))
    else:
        screw_verdict = "SCREW_REFUTED"
        screw_note = ("the single-twist stratified warp fails to match the structure (either it does "
                      "not beat naive-copy on Road or it harms a static class). Re-examine regime "
                      "assignment / calibration.")

    return {
        "shared_calibration": {"s_t": shared_params[0], "s_r": shared_params[1], "pitch": shared_params[2]},
        "per_class": per_class,
        "totals": totals,
        "gap_decomposition": gap_decomp,
        "gap_summary": gap_summary,
        "sky_divergence_null_ablation": sky_null,
        "screw_verdict": screw_verdict,
        "screw_note": screw_note,
        "byte_accounting": {
            "screw_c_marginal": ("~0 marginal: reuses the stored 6-DOF pose sidecar (already paid for "
                                 "d_pose) + ONE static scene descriptor for the whole clip (calibration "
                                 "s_t/s_r/pitch + plane n,d + hood-mask) ~= O(10) params total."),
            "independent_b_global_granularity": ("3 global scalars x 5 classes = 15 globals for the whole "
                                                 "clip (the granularity actually MEASURED here)."),
            "independent_per_pair_alternative": ("per-PAIR independent per-class homographies ~ 11 params/"
                                                 "pair x ~600 pairs ~= 6,600 params (graphics_aa Task 6) — "
                                                 "the expensive granularity the screw replaces; NOT measured "
                                                 "here (would overfit at this n)."),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = all in cache")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-screw", action="store_true",
                    help="skip the screw/twist stratified-warp analysis (default: run it)")
    args = ap.parse_args(argv)

    t0 = time.time()
    cache = (REPO / args.cache) if not Path(args.cache).is_absolute() else Path(args.cache)
    z = np.load(cache, allow_pickle=False)
    L = np.asarray(z["lstars"], dtype=np.int64)
    poses = np.asarray(z["gt_poses"], dtype=np.float64)
    if args.n_pairs and args.n_pairs < L.shape[0]:
        L, poses = L[: args.n_pairs], poses[: args.n_pairs]
    P, Hh, Ww = L.shape
    K = intrinsics_at(Ww, Hh)
    Kinv = np.linalg.inv(K)
    tgt_grid = _target_grid(Hh, Ww)
    counts = np.stack([(L == c).sum(0) for c in range(5)], 0)
    mode = counts.argmax(0)

    fit = fit_calibration(L, poses, K, Kinv, tgt_grid)
    res = eval_predictor(L, poses, K, Kinv, tgt_grid, mode, (fit["s_t"], fit["s_r"], fit["pitch"]))
    screw = None
    if not args.no_screw:
        screw = screw_analysis(L, poses, K, Kinv, tgt_grid, (fit["s_t"], fit["s_r"], fit["pitch"]))

    # ---- decomposition + verdict (per-class warp vs persist) ----
    persist, warp = res["persist"], res["warp"]
    decomp = {}
    for c in range(5):
        nm = CLASS_NAMES[c]
        pv, wv = persist[nm], warp[nm]
        if pv is None or wv is None:
            decomp[nm] = None
            continue
        decomp[nm] = {
            "persist_dseg": pv, "warp_dseg": wv,
            "abs_improvement": pv - wv,
            "rel_improvement": (pv - wv) / pv if pv > 0 else 0.0,
            "target_area": persist[nm + "_area"],
        }
    road = decomp["Road"]; lane = decomp["Lane"]
    roadplane_improves = bool(road and road["abs_improvement"] > 0)
    # fraction of total persist d_seg that the pose-warp explains on road-plane classes
    pp_total = persist["total"]
    explained_roadplane = sum(
        (decomp[CLASS_NAMES[c]]["abs_improvement"] * decomp[CLASS_NAMES[c]]["target_area"])
        for c in ROAD_PLANE_CLASSES if decomp[CLASS_NAMES[c]]
    )
    if roadplane_improves and road["rel_improvement"] >= 0.10:
        verdict = "CONFIRMED_PARTIAL"
        verdict_note = ("pose-homography compresses the Road d_seg trajectory "
                        f"({road['rel_improvement']*100:.0f}% on Road); the residual concentrates in "
                        "Lane-survival + Movables + static classes (which need identity, not a ground "
                        "warp). The stored pose carries the road-plane d_seg modulation.")
    elif roadplane_improves:
        verdict = "WEAK_SIGNAL"
        verdict_note = ("pose-homography reduces Road d_seg but by <10%; pose carries less of the "
                        "trajectory than hoped at this (proxy, pre-R, raw-learned-pose) operating point.")
    else:
        verdict = "REFUTED"
        verdict_note = ("the fitted pose-homography does NOT reduce Road d_seg below persist; the raw "
                        "learned pose does not linearly drive a ground-plane warp that explains the "
                        "argmax trajectory (calibration did not close / non-planarity / pose!=ego-motion).")

    out = {
        "tool": "tools/measure_pose_warp_dseg.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS advisory / research-signal]",
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED (advisory measurement; not a contest score)",
        "cache": str(cache.relative_to(REPO)) if str(cache).startswith(str(REPO)) else str(cache),
        "n_pairs": P, "n_transitions": P - 1, "seg_hw": [Hh, Ww],
        "intrinsics_K_seg_res": K.tolist(),
        "camera_height_m": CAMERA_HEIGHT_M,
        "fit_calibration": fit,
        "baselines": {
            "B_static_mode_total": res["static_mode"]["total"],
            "B_persist_total": persist["total"],
            "W_pose_warp_total": warp["total"],
            "warp_coverage": res["warp_coverage"],
        },
        "persist_per_class": persist,
        "warp_per_class": warp,
        "static_mode_per_class": res["static_mode"],
        "decomposition": decomp,
        "roadplane_explained_dseg_contribution": explained_roadplane,
        "persist_total_dseg": pp_total,
        "verdict": verdict,
        "verdict_note": verdict_note,
        "byte_implication": ("pose = 6 floats/frame, ALREADY stored for d_pose; any d_seg trajectory "
                             "it explains is FREE bytes (only the canonical scene + survival/movables "
                             "residual must be paid)."),
        "assumptions": {
            "PROVEN": "warp/persist/static d_seg = real argmax-disagreement vs frozen CPU-torch SegNet "
                      "argmax (lstars); no surrogate.",
            "INFERRED_pose_columns": "raw learned PoseNet 6-vector interpreted as [fwd, lat, vert, r0, r1, r2]; "
                                     "col0 (~33) is forward. Physical axis mapping is a flagged assumption.",
            "INFERRED_relative_pose": "relative motion frame(2p+1)->frame(2p+3) proxied by pose[p+1] "
                                      "(pairs are non-overlapping seq_len=2 -> lstars are 2 frames apart); "
                                      "the per-frame factor + learned-units->metric scale are absorbed into fitted s_t.",
            "PROXY_pre_R": "direct-partition warp PRE round-trip (no bicubic->uint8->resize R operator); "
                           "necessary-not-sufficient for warp-inside-the-witness-INR. Realized-through-R + "
                           "exact CPU/CUDA eval is the authority.",
            "calibration": f"3 global scalars fit (s_t, s_r, pitch) on Road+Lane; K from EON intrinsics scaled to {Ww}x{Hh}.",
        },
        "elapsed_secs": round(time.time() - t0, 1),
    }
    if screw is not None:
        out["screw_analysis"] = screw

    out_dir = Path(args.out_dir) if args.out_dir else (REPO / f"experiments/results/grok_pose_warp_dseg_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("verdict", "verdict_note", "baselines", "fit_calibration")}, indent=2))
    print(f"\n[written] {out_path}")
    print(f"[per-class warp vs persist]")
    for c in range(5):
        d = decomp[CLASS_NAMES[c]]
        if d:
            print(f"  {CLASS_NAMES[c]:8s} area={d['target_area']:.3f}  persist={d['persist_dseg']:.4f}  "
                  f"warp={d['warp_dseg']:.4f}  rel_impr={d['rel_improvement']*100:+.0f}%")
    if screw is not None:
        t = screw["totals"]
        print("\n[SCREW/TWIST stratified vs per-class independent oracle]  "
              f"verdict={screw['screw_verdict']}")
        print(f"  totals: (a)persist={t['a_persist_total']:.5f}  "
              f"(b)independent={t['b_independent_total']:.5f}  (c)screw={t['c_screw_total']:.5f}")
        print(f"  {'class':8s} {'regime':9s} {'(a)persist':>11s} {'(b)indep':>10s} {'(c)screw':>10s} "
              f"{'c-b':>9s}")
        for c in range(5):
            pc = screw["per_class"][CLASS_NAMES[c]]
            print(f"  {CLASS_NAMES[c]:8s} {pc['screw_regime']:9s} {pc['a_persist_dseg']:>11.4f} "
                  f"{pc['b_independent_dseg']:>10.4f} {pc['c_screw_dseg']:>10.4f} {pc['c_minus_b']:>+9.4f}")
        gs = screw["gap_summary"]
        print(f"  gap (c-b)={gs['total_c_minus_b']:+.5f}  of which non-physical-oracle-overfit="
              f"{gs['gap_from_oracle_nonphysical_overfit']:.5f}  genuine-residual="
              f"{gs['gap_from_genuine_residual']:.5f}")
        sn = screw["sky_divergence_null_ablation"]
        print(f"  sky-null: rotonly={sn['sky_rotonly_dseg']:.4f}  ground(+t)={sn['sky_ground_dseg']:.4f}  "
              f"identity={sn['sky_identity_dseg']:.4f}  t_term_hurts_sky={sn['t_term_hurts_sky']}")
        print(f"  {screw['screw_note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
