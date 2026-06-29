# SPDX-License-Identifier: MIT
"""WATERFILL THROUGH R: does the decode-free targeted-waterfill's MODELED bulk-only win
survive when the witness is REALIZED and pushed through the contest R operator?

THE QUESTION (DAG FEED-kd open). FEED-kd (``tools/measure_jitter_predictability.py`` M4,
``experiments/results/jitter_predictability_n96_r2``) MODELED a decode-free TARGETED
WATERFILL: store only the thin "unsafe" annulus (near the warp-prior boundary / low vote
confidence), leave the high-margin "safe" bulk un-stored. In the PRE-R FLIP-SET MODEL it
gave bulk-only S ~= 0.108 (free prior-boundary annulus tau=2: safe_frac 0.969, modeled
residual 0.000236, rate 0.0844) -- a 1.4x rate cut vs the naive whole-bulk store (0.118),
DETERMINISTIC. This tool VALIDATES that modeled win THROUGH the contest R operator.

THREE conditioning sets (coordinator scope), all measured against the SAME through-R flip map:
  (a) DECODE-FREE GEOMETRY: prior-boundary SDF / vote-confidence (the original scope, FREE).
  (b) + WARPED-STORED-CANONICAL-MARGIN: SegNet margin computed ONCE on the canonical at
      COMPRESS time, warped per-frame by the stored pose (FREE warp; COUNTED one-field store
      ~KB). COMPLIANT: NO SegNet at decode (Yousfi PR#35 / strict-scorer non-negotiable --
      shipping SegNet's ~73MB would dwarf the rate). Proxy used here = margin OF the warped
      canonical (margin is smooth -> ~= warp OF the margin; flagged). PLUS an ORACLE per-frame
      GT-margin annulus as the strict UPPER BOUND.
  (c) STRONG-PREDICTOR edge -> separate tool measure_content_edge_strong.py (logistic floor
      was MI=0.0002b; does a non-overflowing MLP extract materially more?).

TWO realization regimes (the decisive through-R question):
  MEASUREMENT A (idealized correction = the FEED-kd model, through-R safe region). Residual =
    through-R flips OUTSIDE the annulus; the annulus is ASSUMED perfectly corrected. rate =
    REAL clustering-aware coder (zlib/brotli on the per-annulus-pixel symbol stream; the iid
    sparse-set "LB" is reported too but it IGNORES the measured 78x clustering so it OVER-states
    rate). The S-optimal point is the through-R bulk-only S to compare to the modeled 0.108.
  MEASUREMENT B (realized correction). At a sweep of taus, build the witness RGB = SHARP
    canonical (safe) with GT RGB content PASTED at the annulus (OPTIMISTIC: best a renderer
    could do) -> R -> SegNet -> REAL residual. Also a per-class flat-color paste (PESSIMISTIC
    bracket). If B ~= A, the override realizes through R; if B >> A, a render/seam penalty eats
    the win. The realized through-R bulk-only S = min_tau [100*B_residual(tau) + real_rate(tau)].

AUTHORITY / HONESTY FIREWALL (CLAUDE.md): ``[macOS advisory / CPU-torch research-signal]``
ONLY. NOT a contest score. Pointer 0.19110 UNMOVED. score_claim/promotable/ready=False. MEANS.
d_seg = REAL argmax-disagreement vs the frozen CPU-torch SegNet argmax ``lstars`` (NEVER MPS);
a NO-FAKE self-check asserts SegNet(gt_f1)==lstars and ABORTS otherwise. BULK-ONLY is BULK-ONLY
(Road/Undriv/MyCar): Lane long-tail wall + Movables + canonical keyframe STACK ON TOP. It
warps/pastes GT RGB (not a shipped INR) -> bounds the deterministic realization; authority =
realized-through-R inside the witness INR + exact CPU/CUDA eval on byte-closed bytes.

rule-118: homography+expmap+compose+window-SHARP+R+boundary-dilation+margin-warp = FREE
deterministic geometry (uncounted, may use up to the 30-min decode budget). The per-pair 6-DOF
pose is COUNTED-but-EXISTING. Stored = unsafe-annulus flip corrections + canonical (+ the one
canonical margin field for (b)). Annulus POSITIONS are FREE (decoder recomputes the boundary).
FORBIDDEN: SegNet at inflate (so (b) WARPS a stored margin; the GT-margin annulus is an UB).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
from scipy import ndimage

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

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
    _target_grid,
)
from tools.measure_screw_warp_through_R import fit_calibration_within_pair  # noqa: E402
from tools.measure_clean_canonical_warp_through_R import (  # noqa: E402
    compose_path_H,
    rgb_at,
    BUDGET,
    BULK_CLASSES,
)
from tools.measure_budget_gate_overturn import (  # noqa: E402
    build_exact_step_poses,
    _warp_window,
    _vote_argmax,
    _sharp_canonical_rgb,
)

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
B0 = 37_545_489  # contest archive normalizer (25*bytes/B0 = rate term)
NAIVE_DITHER_RATE = 0.1185  # FEED-kb uniform whole-bulk store (the door-closed reference)
MODEL_FREE_PRIOR_SOPT_BULK = 0.10808769483983477  # FEED-kd modeled free-prior tau=2 S_bulk
MODEL_FREE_PRIOR_SOPT_RES = 0.00023628235806761314
MODEL_FREE_PRIOR_SOPT_RATE = 0.08445945903307345
BULK_IDX = [CLASS_NAMES.index(c) for c in BULK_CLASSES]  # [0, 2, 4]


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def _boundary(lab):
    b = np.zeros(lab.shape, dtype=bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b


def store_bytes_lb(U, Fu, scale600):
    """iid sparse-set estimate: bits to say WHICH Fu of U annulus pixels flip + 1.5 label
    bits/flip. IGNORES the measured 78x clustering -> OVER-states rate (NOT a true LB)."""
    if Fu <= 0 or U <= 0:
        return 0.0
    bits_pos = Fu * math.log2(math.e * U / Fu) if Fu < U else float(U)
    return ((bits_pos + 1.5 * Fu) / 8.0) * scale600


def real_coder_bytes(unsafe_mask, flip_all, cls_all, scale600):
    """REAL clustering-aware coder (the HONEST rate). Per annulus pixel, symbol = 0 (warp
    correct) else stored class label+1; zlib/brotli on the (pair, raster)-ordered stream
    exploits the 78x clustering. Annulus POSITIONS are FREE (decoder recomputes the boundary)."""
    if not unsafe_mask.any():
        return 0.0
    sym = np.zeros(int(unsafe_mask.sum()), dtype=np.uint8)
    fl = flip_all[unsafe_mask]
    sym[fl] = cls_all[unsafe_mask][fl].astype(np.uint8) + 1
    raw = sym.tobytes()
    z = len(zlib.compress(raw, 9))
    try:
        import brotli
        z = min(z, len(brotli.compress(raw, quality=11)))
    except Exception:
        pass
    return z * scale600


def _vote_conf(lv, lvd):
    nv, Hs, Ws = lv.shape
    votes = np.zeros((5, Hs, Ws), dtype=np.float64)
    for k in range(nv):
        vk = lvd[k]
        for c in range(5):
            votes[c] += (lv[k] == c) & vk
    s = votes.sum(0)
    return votes.max(0) / np.maximum(s, 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--comma-pose",
                    default="experiments/results/pose_feasibility_probe/comma2k19_gt_pose_raw.npz")
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = all in cache")
    ap.add_argument("--window-radius", type=int, default=2)
    ap.add_argument("--realize-taus-sdf", default="0.0,2.0,5.0,8.0,12.0,1000000.0",
                    help="prior-boundary SDF taus for the Measurement-B realization sweep "
                         "(0=no-override realized; 1e6=full-GT sanity, must give residual~0).")
    ap.add_argument("--selfcheck-pairs", type=int, default=4)
    ap.add_argument("--save-seg-cache", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    realize_taus = [float(x) for x in args.realize_taus_sdf.split(",")]

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
    fwd_times = []

    def seg_full(frame_uint8_native):
        tt = time.time()
        am, mg = measure_segnet_argmax(seg, np.asarray(frame_uint8_native, dtype=np.float64))
        fwd_times.append(time.time() - tt)
        return am, mg

    # ---- NO-FAKE self-check ----
    scn = min(args.selfcheck_pairs, P)
    sc = {"pairs_checked": scn, "exact_matches": 0, "max_disagree_px": 0}
    for p in range(scn):
        am, _ = seg_full(gt_f1[p])
        nd = int(np.count_nonzero(am != lstars[p]))
        sc["max_disagree_px"] = max(sc["max_disagree_px"], nd)
        sc["exact_matches"] += int(nd == 0)
    sc["PASS"] = bool(sc["exact_matches"] == scn and sc["max_disagree_px"] == 0)
    if not sc["PASS"]:
        raise SystemExit(f"NO-FAKE self-check FAILED (max_disagree_px={sc['max_disagree_px']}).")

    # ---- per-global-frame SegNet argmax cache ----
    print(f"[wf-R] caching per-frame SegNet argmax for {n_frames} frames (CPU)...", flush=True)
    seg_cache = np.zeros((n_frames, SEG_H, SEG_W), dtype=np.int64)
    for g in range(n_frames):
        seg_cache[g], _ = seg_full(rgb_at(gt_f0, gt_f1, g))
        if (g + 1) % 48 == 0 or g == n_frames - 1:
            print(f"  ...{g + 1}/{n_frames}", flush=True)
    cache_selfcheck = int(sum(int(np.array_equal(seg_cache[2 * p + 1], lstars[p])) for p in range(P)))
    if args.save_seg_cache:
        scp = (REPO / args.save_seg_cache) if not Path(args.save_seg_cache).is_absolute() else Path(args.save_seg_cache)
        _refuse_tmp(scp); scp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(scp, seg_cache=seg_cache.astype(np.int8))

    # ---- exact step poses + calibration ----
    exact_step_poses = build_exact_step_poses(pos, ori, n_frames)
    K_nat = intrinsics_at(NAT_W, NAT_H); Kinv_nat = np.linalg.inv(K_nat); grid_nat = _target_grid(NAT_H, NAT_W)
    K_seg = intrinsics_at(SEG_W, SEG_H); Kinv_seg = np.linalg.inv(K_seg); grid_seg = _target_grid(SEG_H, SEG_W)
    lstar0 = seg_cache[0::2]
    comma_within = np.stack([exact_step_poses[2 * p] for p in range(P)], 0)
    fit = fit_calibration_within_pair(lstar0, lstars, comma_within, K_seg, Kinv_seg, grid_seg)
    fit_pn = fit_calibration_within_pair(lstar0, lstars, posenet_poses, K_seg, Kinv_seg, grid_seg)
    convention_ok = bool(fit["fit_roadplane_dseg"] <= fit_pn["fit_roadplane_dseg"] * 1.05)
    fit_params = (fit["s_t"], fit["s_r"], fit["pitch"])
    print(f"[wf-R] convention_ok={convention_ok} comma={fit['fit_roadplane_dseg']:.5f} "
          f"posenet={fit_pn['fit_roadplane_dseg']:.5f}", flush=True)

    # accumulators
    flip_list, sdf_list, vc_list, mg_gt_list, mg_canon_list, cls_list = [], [], [], [], [], []
    bulk_px_total = 0; noov_flip_total = 0
    noov_ne = [0] * 5; noov_tot = [0] * 5
    # Measurement B realization sweep accumulators (per sdf tau)
    B = {tau: {"gtp": 0} for tau in realize_taus}
    B_flat = {tau: 0 for tau in realize_taus}  # flat-color render of the unsafe annulus (pessimistic bracket)
    canon_margin_store_bytes = None

    ys_n2s = (np.arange(NAT_H) * SEG_H / NAT_H).astype(np.int64).clip(0, SEG_H - 1)
    xs_n2s = (np.arange(NAT_W) * SEG_W / NAT_W).astype(np.int64).clip(0, SEG_W - 1)
    # per-class mean RGB for the pessimistic flat-color paste (compress-time stat)
    class_mean_rgb = np.full((5, 3), 127.0)
    cnt = np.zeros(5)
    for p in range(min(P, 24)):
        for c in range(5):
            mn = (lstars[p] == c)[ys_n2s][:, xs_n2s]
            if mn.any():
                class_mean_rgb[c] = class_mean_rgb[c] * 0 + (class_mean_rgb[c] * cnt[c] +
                                    gt_f1[p][mn].astype(np.float64).sum(0)) / (cnt[c] + int(mn.sum()))
                cnt[c] += int(mn.sum())

    print(f"[wf-R] through-R pass over {P} pairs (window=+/-{args.window_radius}, "
          f"realize taus={realize_taus})...", flush=True)
    for p in range(P):
        t = 2 * p + 1
        f0 = gt_f0[p].astype(np.float64)
        tgt = lstars[p]
        canon_rgb, vote_am, vote_conf_r = {}, {}, {}
        for regime in ("ground", "rotonly", "identity"):
            rv, lv, lvd, ndt = _warp_window(gt_f0, gt_f1, seg_cache, t, args.window_radius,
                                            exact_step_poses, K_nat, Kinv_nat, grid_nat,
                                            K_seg, Kinv_seg, grid_seg, fit_params, regime, n_frames)
            if rv is None:
                canon_rgb[regime] = gt_f0[p]; vote_am[regime] = seg_cache[2 * p]
                vote_conf_r[regime] = np.ones((SEG_H, SEG_W)); continue
            vote = _vote_argmax(lv, lvd)
            canon_rgb[regime] = _sharp_canonical_rgb(rv, lv, lvd, ndt, vote, f0)
            vote_am[regime] = vote; vote_conf_r[regime] = _vote_conf(lv, lvd)

        # through-R no-override argmax + margin (margin = the canonical-margin proxy for (b))
        noov_am, noov_mg = {}, {}
        for r in ("ground", "rotonly", "identity"):
            noov_am[r], noov_mg[r] = seg_full(canon_rgb[r])

        prior = vote_am["identity"]
        sdf_prior = np.minimum(np.nan_to_num(ndimage.distance_transform_edt(~_boundary(prior)),
                                             posinf=64.0), 64.0)
        vc = np.minimum.reduce([vote_conf_r["ground"], vote_conf_r["rotonly"], vote_conf_r["identity"]])
        gt_marg = margins[p]
        # regime-routed canonical margin (compliant-b feature proxy) + flip map
        canon_marg = np.zeros((SEG_H, SEG_W), dtype=np.float32)
        flip_noov = np.zeros((SEG_H, SEG_W), dtype=bool)
        bulk_mask = np.isin(tgt, BULK_IDX)
        for c in BULK_IDX:
            r = SCREW_REGIME[c]; m = (tgt == c)
            if not m.any():
                continue
            canon_marg[m] = noov_mg[r][m]
            fl = (noov_am[r] != c) & m
            flip_noov |= fl
            noov_ne[c] += int(fl.sum()); noov_tot[c] += int(m.sum())
        bulk_px_total += int(bulk_mask.sum()); noov_flip_total += int(flip_noov.sum())
        idx = np.where(bulk_mask)
        flip_list.append(flip_noov[idx]); sdf_list.append(sdf_prior[idx].astype(np.float32))
        vc_list.append(vc[idx].astype(np.float32)); mg_gt_list.append(gt_marg[idx].astype(np.float32))
        mg_canon_list.append(canon_marg[idx].astype(np.float32)); cls_list.append(tgt[idx].astype(np.int8))

        # one-canonical-margin store cost (compliant-b): quantize identity-canonical margin to 4 bits + zlib
        if canon_margin_store_bytes is None:
            mq = np.clip(noov_mg["identity"] / 8.0 * 15.0, 0, 15).astype(np.uint8)  # margins ~0..>8 -> 16 levels
            b = len(zlib.compress(mq.tobytes(), 9))
            try:
                import brotli
                b = min(b, len(brotli.compress(mq.tobytes(), quality=11)))
            except Exception:
                pass
            canon_margin_store_bytes = b

        # Measurement B realization sweep -- ONE faithful witness per tau (per-pixel regime-routed):
        #   GT everywhere EXCEPT safe-bulk pixels use THEIR class's regime canonical (the un-stored
        #   safe region). unsafe annulus + Lane + Movables = GT (handled by other v2 components ->
        #   isolates the BULK-correction realization). tau=inf -> no safe-bulk -> GT -> residual 0
        #   (clean mechanism check). B_flat: also render the unsafe bulk annulus as flat class color.
        tgt_nat = tgt[ys_n2s][:, xs_n2s]
        for tau in realize_taus:
            comp = gt_f1[p].astype(np.float64).copy()
            comp_flat = gt_f1[p].astype(np.float64).copy()
            for c in BULK_IDX:
                r = SCREW_REGIME[c]
                safe_c = (tgt == c) & (sdf_prior >= tau)        # seg-res un-stored safe bulk of class c
                ann_c = (tgt == c) & (sdf_prior < tau)          # seg-res stored unsafe annulus of class c
                safe_nat = safe_c[ys_n2s][:, xs_n2s]
                ann_nat = ann_c[ys_n2s][:, xs_n2s]
                comp[safe_nat] = canon_rgb[r].astype(np.float64)[safe_nat]
                comp_flat[safe_nat] = canon_rgb[r].astype(np.float64)[safe_nat]
                comp_flat[ann_nat] = class_mean_rgb[c]
            am_g, _ = seg_full(comp)
            am_f, _ = seg_full(comp_flat)
            for c in BULK_IDX:
                m = (tgt == c)
                B[tau]["gtp"] += int(((am_g != c) & m).sum())
                B_flat[tau] += int(((am_f != c) & m).sum())
        if (p + 1) % 8 == 0 or p == P - 1:
            print(f"  ...{p + 1}/{P}", flush=True)

    flip_all = np.concatenate(flip_list); sdf_all = np.concatenate(sdf_list)
    vc_all = np.concatenate(vc_list); mg_gt_all = np.concatenate(mg_gt_list)
    mg_canon_all = np.concatenate(mg_canon_list); cls_all = np.concatenate(cls_list)
    N = bulk_px_total; Ftot = float(noov_flip_total); scale600 = 600.0 / n_frames
    noov_bulk = sum(noov_ne[c] for c in BULK_IDX) / max(sum(noov_tot[c] for c in BULK_IDX), 1)
    canon_margin_rate = 25.0 * (canon_margin_store_bytes or 0) / B0  # ONE field; ~KB

    def curve(score, thresholds, count_margin_store=False):
        rows = []
        for thr in thresholds:
            unsafe = score < thr
            U = int(unsafe.sum()); Fin = float(flip_all[unsafe].sum()); Fout = Ftot - Fin
            residual = Fout / N
            rate_real = 25.0 * real_coder_bytes(unsafe, flip_all, cls_all, scale600) / B0
            rate_lb = 25.0 * store_bytes_lb(U, Fin, scale600) / B0
            if count_margin_store:
                rate_real += canon_margin_rate; rate_lb += canon_margin_rate
            rows.append({"threshold": float(thr), "safe_fraction": (N - U) / N,
                         "flip_capture_through_R": (Fin / Ftot) if Ftot else None,
                         "residual_bulk_dseg_through_R": residual,
                         "residual_within_budget": bool(residual <= BUDGET),
                         "rate_term_realcoder": rate_real, "rate_term_iid_lb": rate_lb,
                         "S_bulk_realcoder": 100.0 * residual + rate_real,
                         "S_bulk_iid_lb": 100.0 * residual + rate_lb})
        s_opt = min(rows, key=lambda r: r["S_bulk_realcoder"]) if rows else None
        return {"curve": rows, "S_optimal_realcoder": s_opt}

    sdf_taus = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0]
    vc_taus = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.01]
    mg_taus = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    a_sdf = curve(sdf_all, sdf_taus); a_vc = curve(vc_all, vc_taus)
    b_canon = curve(mg_canon_all, mg_taus, count_margin_store=True)
    b_oracle = curve(mg_gt_all, mg_taus, count_margin_store=False)

    # Measurement B realized curve (GT-paste residual + real-coder rate at each sdf tau)
    realized_rows = []
    for tau in realize_taus:
        unsafe = sdf_all < tau
        rate_real = 25.0 * real_coder_bytes(unsafe, flip_all, cls_all, scale600) / B0
        resA = (Ftot - float(flip_all[unsafe].sum())) / N
        resB = B[tau]["gtp"] / max(N, 1)
        resF = B_flat[tau] / max(N, 1)
        row = {"tau_sdf": tau, "residual_A_idealized": resA, "residual_B_GTpaste": resB,
               "residual_B_flat_color": resF, "rate_term_realcoder": rate_real,
               "S_bulk_A_idealized": 100.0 * resA + rate_real,
               "S_bulk_B_realized_GTpaste": 100.0 * resB + rate_real,
               "S_bulk_B_flat_color": 100.0 * resF + rate_real}
        realized_rows.append(row)
    realized_best = min(realized_rows, key=lambda r: r["S_bulk_B_realized_GTpaste"])

    fwd_arr = np.array(fwd_times)
    elapsed = round(time.time() - t0, 1)
    out = {
        "tool": "tools/measure_waterfill_through_R.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS advisory / CPU-torch research-signal]",
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110 (advisory; not a contest score)",
        "cache": str(cache.relative_to(REPO)) if str(cache).startswith(str(REPO)) else str(cache),
        "n_pairs": P, "seg_hw": [SEG_H, SEG_W], "window_radius": args.window_radius,
        "no_fake_selfcheck_segnet_gt_f1_eq_lstars": sc,
        "cache_selfcheck_seg_cache_f1_eq_lstars": {"matches": cache_selfcheck, "P": P, "PASS": bool(cache_selfcheck == P)},
        "comma2k19_convention_validation": {"comma": fit["fit_roadplane_dseg"],
                                            "posenet": fit_pn["fit_roadplane_dseg"], "VALIDATED": convention_ok},
        "through_R_no_override_baseline": {
            "bulk_dseg": noov_bulk,
            "per_class": {CLASS_NAMES[c]: (noov_ne[c] / noov_tot[c]) if noov_tot[c] else None for c in BULK_IDX},
            "overturn_n96_exact_SHARP_ref": 0.0036929448445638022,
            "note": "SHARP-exact clean-canonical through R, NO correction (0%-stored baseline). Sanity vs overturn.",
        },
        "flip_set_through_R": {"bulk_px_total": N, "noov_flip_total": noov_flip_total,
                              "noov_bulk_flip_fraction": Ftot / max(N, 1)},
        "MEASUREMENT_A_idealized_correction": {
            "note": ("residual = through-R flips OUTSIDE annulus (annulus assumed perfectly corrected = the "
                     "FEED-kd model); rate = REAL clustering-aware coder (honest) + iid LB (over-states)."),
            "a_decode_free_prior_sdf": a_sdf,
            "a_decode_free_vote_conf": a_vc,
            "b_compliant_warped_canonical_margin": b_canon,
            "b_oracle_per_frame_GT_margin_UPPER_BOUND": b_oracle,
        },
        "MEASUREMENT_B_realized_correction_GTpaste_sweep": {
            "note": ("witness = SHARP canonical (safe) + GT RGB pasted at the decode-free sdf annulus "
                     "(OPTIMISTIC realization) -> R -> SegNet. flat-color = pessimistic bracket. "
                     "realized S_bulk = 100*B_residual + real-coder rate."),
            "rows": realized_rows,
            "realized_best_GTpaste": realized_best,
        },
        "compliant_b_margin_store": {
            "one_canonical_margin_field_bytes_4bit_zlib": canon_margin_store_bytes,
            "rate_contribution_one_field": canon_margin_rate,
            "note": ("(b) COMPLIANT cost = store ONE canonical margin field (4-bit quantized, smooth -> "
                     "compresses well), warp it per-frame (free). b_compliant curve rate INCLUDES this. "
                     "Multiple sliding keyframes -> x(#keyframes). NO SegNet at decode (Yousfi PR#35)."),
        },
        "decode_budget_feasibility": {
            "cpu_segnet_forward_sec_mean": float(fwd_arr.mean()) if len(fwd_arr) else None,
            "cpu_segnet_forward_sec_p90": float(np.percentile(fwd_arr, 90)) if len(fwd_arr) else None,
            "est_cpu_sec_for_600_forwards": float(fwd_arr.mean() * 600) if len(fwd_arr) else None,
            "note": ("decode does NOT run SegNet (forbidden); this times the COMPRESS-time SegNet only, for "
                     "reference. The decode mechanism (warp + paste/render) is far cheaper than a SegNet forward."),
        },
        "MODEL_COMPARISON": {
            "FEEDkd_modeled_free_prior_Sopt_bulk": MODEL_FREE_PRIOR_SOPT_BULK,
            "FEEDkd_modeled_residual": MODEL_FREE_PRIOR_SOPT_RES, "FEEDkd_modeled_rate": MODEL_FREE_PRIOR_SOPT_RATE,
            "naive_whole_bulk_rate": NAIVE_DITHER_RATE,
            "through_R_A_decode_free_sdf_Sopt": a_sdf["S_optimal_realcoder"],
            "through_R_A_compliant_margin_Sopt": b_canon["S_optimal_realcoder"],
            "through_R_A_oracle_margin_Sopt": b_oracle["S_optimal_realcoder"],
            "through_R_B_realized_best": realized_best,
            "interpretation": ("MEASUREMENT A (idealized) tests safe-region R-survival + rate; MEASUREMENT B "
                               "(realized) tests whether the override actually corrects through R. The honest "
                               "through-R bulk-only S is B (realized). A-vs-B gap = the realization penalty; "
                               "(a)-vs-(b) gap = the margin-conditioning value. Both are FINDINGS, not verdicts."),
        },
        "rule_118": {
            "FREE": "homography+expmap+compose+window-SHARP+R+boundary-dilation+margin-warp (uncounted, <=30min decode)",
            "COUNTED_existing": "per-pair 6-DOF pose (stored for d_pose)",
            "COUNTED": "unsafe-annulus flip corrections + canonical scene + ONE canonical margin field (for b)",
            "FREE_annulus_positions": "decoder recomputes the warped-canonical boundary; only flip SET + labels stored",
            "FORBIDDEN": "SegNet at inflate -> (b) warps a stored margin; GT-margin annulus is an UPPER BOUND",
        },
        "assumptions": {
            "PROVEN": "through-R d_seg = real argmax-disagreement vs frozen CPU-torch SegNet (lstars); "
                      "no-override SHARP bulk cross-checks the overturn exact_SHARP_bulk.",
            "BULK_ONLY": "Road/Undriv/MyCar only; Lane wall + Movables + canonical keyframe STACK ON TOP.",
            "B_GTpaste_optimistic": "GT-paste = best a renderer could do (true content); flat-color = pessimistic. "
                                    "the shippable INR render is between them. annulus pasted at native via "
                                    "nearest-upsample of the seg-res mask (slight seam misalignment).",
            "b_margin_proxy": "(b) feature = margin OF the warped canonical (free from the noov SegNet); the "
                              "COMPLIANT decode uses warp OF the stored canonical margin ~= this (margin smooth).",
            "rate": "real-coder = honest clustering-aware (zlib/brotli, FREE annulus positions); iid LB over-states.",
            "CANDIDATE_SET": "GT-bulk pixels (matches FEED-kd; ~99.7% == prior-bulk).",
        },
        "elapsed_secs": elapsed,
    }

    out_dir = (Path(args.out_dir) if args.out_dir
               else (REPO / f"experiments/results/waterfill_through_R_n{P}_r{args.window_radius}"))
    _refuse_tmp(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2, default=float))

    print("\n[wf-R] ===== SUMMARY (n=%d, through R, BULK-ONLY, advisory) =====" % P)
    print(f"  no-override SHARP through-R bulk = {noov_bulk:.5f} (overturn ref 0.00369)")
    for tag, cv in [("a decode-free sdf", a_sdf), ("a vote-conf", a_vc),
                    ("b compliant margin", b_canon), ("b ORACLE margin UB", b_oracle)]:
        s = cv["S_optimal_realcoder"]
        print(f"  [A-ideal {tag:18s}] tau={s['threshold']:<5g} res={s['residual_bulk_dseg_through_R']:.6f} "
              f"rate={s['rate_term_realcoder']:.4f} (LB {s['rate_term_iid_lb']:.4f}) S_bulk={s['S_bulk_realcoder']:.5f}")
    print(f"  [b margin store] 1 field = {canon_margin_store_bytes} B -> rate +{canon_margin_rate:.5f}")
    print("  [B realized GT-paste sweep]:")
    for r in realized_rows:
        extra = (f" flat_res={r.get('residual_B_flat_color'):.6f}" if "residual_B_flat_color" in r else "")
        print(f"     tau={r['tau_sdf']:<5g} A_res={r['residual_A_idealized']:.6f} B_res={r['residual_B_GTpaste']:.6f} "
              f"rate={r['rate_term_realcoder']:.4f} S_A={r['S_bulk_A_idealized']:.5f} "
              f"S_B={r['S_bulk_B_realized_GTpaste']:.5f}{extra}")
    rb = realized_best
    print(f"  >>> REALIZED best (GT-paste): tau={rb['tau_sdf']} S_bulk={rb['S_bulk_B_realized_GTpaste']:.5f} "
          f"(res {rb['residual_B_GTpaste']:.6f} + rate {rb['rate_term_realcoder']:.4f})")
    print(f"  MODEL was: free-prior S_bulk=0.108 (res 0.000236 rate 0.0844); naive rate=0.1185")
    print(f"\n[written] {out_dir/'results.json'} (elapsed {elapsed}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
