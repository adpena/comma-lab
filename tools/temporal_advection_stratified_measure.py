# SPDX-License-Identifier: MIT
"""TEMPORAL / ADVECTION lens (projection unification lens 5): per-STRATUM xi-transport.

Prior art (PROACTIVE RECALL, do not re-derive):
  * ``tools/measure_pose_warp_dseg.py`` — pair->pair pose-homography label warp,
    class-level verdicts (grok memo 20260629T181000Z: Road +15-17%; hood needs identity;
    sky needs rotation-only) and the single-twist SCREW stratified warp
    (screw memo 20260629T192609Z: SCREW_WIN_ZERO_BYTE_PHYSICAL — matches the per-class
    oracle on every physical class at ~0 marginal bytes).
  * ``src/tac/boundary_math/stratified_depth_warp.py`` (#365) — off-plane parallax DOF.
  * ``src/tac/boundary_math/ego_xi_trajectory.py`` — PoseNet 6-vec is xi up-to-affine.

THIS tool measures what those did NOT: the **Morse-Smale STRATUM decomposition** of the
transport (CELL interiors / EDGE separatrices / SADDLE junctions — the V9-CGauge carrier
strata per ``projection_unification_and_eight_lenses_20260715.md`` lens 5) plus the
**trajectory RATE amortization** (naive per-frame partition coding vs project-once +
transport-by-xi + irreducible residual), at n600.

What is measured per transition p -> p+1 (lstars are the frame_1 argmax of each
non-overlapping pair, 2 frames apart; xi proxy = pose[p+1], calibration absorbed into
the 3 fitted globals exactly as in the grok/screw probes):

  1. Predictors: PERSIST (identity transport), GROUND (single ground homography,
     persist fallback), SCREW (single-twist stratified per-class-regime composite:
     hood=identity, sky=rotation-only, ground classes=full plane homography).
  2. Per-stratum label agreement on the TARGET partition's strata:
     CELL = non-boundary pixels; EDGE = boundary pixels not in a saddle plaquette;
     SADDLE = pixels of 2x2 plaquettes containing >=3 distinct classes.
  3. EDGE geometric transport: per class-pair separatrix crack sites, EDT distance
     from target sites to the predicted same-pair separatrix (the delta(s) offset
     residual the curve coder would store) — fraction within 0/1/2 px + mean.
  4. SADDLE transport: target saddle plaquettes matched to predicted saddles of the
     SAME class-signature within radius 2 px.
  5. RATE (zlib-9 conditional-coding proxy, labeled PROXY): naive = sum zlib(L_p);
     trajectory = zlib(L_0) + sum residual_bytes(pred_p -> L_{p+1}) + xi bytes
     (6 fp16/pair — ALREADY banked for d_pose => marginal 0) + O(3) calibration
     scalars. residual_bytes = min(sentinel-plane zlib, packbits(mask)+values zlib).

AUTHORITY / HONESTY FIREWALL: ``[macOS advisory / research-signal]`` ONLY; d_seg-space
agreement vs the frozen CPU-torch SegNet argmax cache (real, no surrogate) but PRE-R
and label-space; the zlib figures are a CONDITIONAL-CODING PROXY, not archive bytes.
NOT a contest score; the canonical frontier pointer is UNMOVED. score_claim=false,
promotable=false.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import measure_pose_warp_dseg as MW  # noqa: E402  (reused oracle-parity machinery)

CLASS_NAMES = MW.CLASS_NAMES
SCREW_REGIME = MW.SCREW_REGIME  # {0:ground,1:ground,2:rotonly,3:ground,4:identity}
STRATA = ("cell", "edge", "saddle")
PREDICTORS = ("persist", "ground", "screw")


# --------------------------------------------------------------------------- #
# Strata extraction (Morse-Smale complex of the argmax partition, pixel grid).
# --------------------------------------------------------------------------- #
def boundary_mask(L: np.ndarray) -> np.ndarray:
    """Pixels with any 4-neighbour of a different class (the separatrix band)."""
    b = np.zeros(L.shape, dtype=bool)
    b[:-1, :] |= L[:-1, :] != L[1:, :]
    b[1:, :] |= L[1:, :] != L[:-1, :]
    b[:, :-1] |= L[:, :-1] != L[:, 1:]
    b[:, 1:] |= L[:, 1:] != L[:, :-1]
    return b


def saddle_plaquettes(L: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float, tuple[int, ...]]]]:
    """2x2 plaquettes with >=3 distinct classes = the saddle 0-cells (triple junctions).

    Returns (saddle_pixel_mask, list of (row_center, col_center, sorted class tuple)).
    """
    a = L[:-1, :-1]
    b = L[:-1, 1:]
    c = L[1:, :-1]
    d = L[1:, 1:]
    corners = np.stack([a, b, c, d], axis=0)  # (4, H-1, W-1)
    s = np.sort(corners, axis=0)
    ndistinct = 1 + (np.diff(s, axis=0) != 0).sum(axis=0)
    sad = ndistinct >= 3  # (H-1, W-1)
    mask = np.zeros(L.shape, dtype=bool)
    mask[:-1, :-1] |= sad
    mask[:-1, 1:] |= sad
    mask[1:, :-1] |= sad
    mask[1:, 1:] |= sad
    ys, xs = np.nonzero(sad)
    pts = []
    for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
        cls = tuple(sorted({int(L[y, x]), int(L[y, x + 1]), int(L[y + 1, x]), int(L[y + 1, x + 1])}))
        pts.append((y + 0.5, x + 0.5, cls))
    return mask, pts


def strata_masks(L: np.ndarray) -> dict[str, np.ndarray]:
    b = boundary_mask(L)
    sad, _ = saddle_plaquettes(L)
    return {"cell": ~b & ~sad, "edge": b & ~sad, "saddle": sad}


def pair_crack_sites(L: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Per class-pair separatrix crack sites: bool (H, W) marking the left/top pixel of
    each horizontal/vertical crack between classes (a, b), keyed by sorted (a, b)."""
    out: dict[tuple[int, int], np.ndarray] = {}
    H, W = L.shape
    # horizontal cracks (col j | j+1)
    dh = L[:, :-1] != L[:, 1:]
    ys, xs = np.nonzero(dh)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
        k = tuple(sorted((int(L[y, x]), int(L[y, x + 1]))))
        m = out.get(k)
        if m is None:
            m = np.zeros((H, W), dtype=bool)
            out[k] = m
        m[y, x] = True
    # vertical cracks (row i | i+1)
    dv = L[:-1, :] != L[1:, :]
    ys, xs = np.nonzero(dv)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
        k = tuple(sorted((int(L[y, x]), int(L[y + 1, x]))))
        m = out.get(k)
        if m is None:
            m = np.zeros((H, W), dtype=bool)
            out[k] = m
        m[y, x] = True
    return out


# --------------------------------------------------------------------------- #
# Predictors (decode-realizable composites; persist fallback everywhere).
# --------------------------------------------------------------------------- #
def predict_ground(src: np.ndarray, pose6: np.ndarray, K, Kinv, tgt_grid, params) -> np.ndarray:
    H = MW.pose_to_homography(pose6, K, Kinv, *params)
    pred, valid = MW.warp_labels(src, H, tgt_grid)
    return np.where(valid, pred, src)


def predict_screw(src: np.ndarray, pose6: np.ndarray, K, Kinv, tgt_grid, params) -> np.ndarray:
    """Single-twist stratified composite (decode-realizable: routes by the WARPED-IN
    class regime; hood static wins where the source says hood).

    Priority: source-hood identity > ground-warped ground-class > rot-warped sky >
    ground-warped anything > persist."""
    s_t, s_r, pitch = params
    Hg = MW.pose_to_homography(pose6, K, Kinv, s_t, s_r, pitch)
    Hr = MW.pose_to_homography(pose6, K, Kinv, 0.0, s_r, pitch)  # rotation-only (sky)
    pg, vg = MW.warp_labels(src, Hg, tgt_grid)
    pr, vr = MW.warp_labels(src, Hr, tgt_grid)
    pred = src.copy()  # persist fallback
    ground_in = vg & np.isin(pg, (0, 1, 3))
    pred = np.where(ground_in, pg, pred)
    sky_in = vr & (pr == 2) & ~ground_in
    pred = np.where(sky_in, pr, pred)
    hood = src == 4  # rigidly-attached hood: identity regime always wins
    pred = np.where(hood, src, pred)
    return pred


# --------------------------------------------------------------------------- #
# Rate proxy (zlib-9 conditional coding).
# --------------------------------------------------------------------------- #
def zbytes(a: np.ndarray) -> int:
    return len(zlib.compress(np.ascontiguousarray(a, dtype=np.uint8).tobytes(), 9))


def residual_bytes(pred: np.ndarray, tgt: np.ndarray) -> int:
    mism = pred != tgt
    plane = tgt.astype(np.uint8).copy()
    plane[~mism] = 255  # sentinel where transport already supplies the label
    b_plane = zbytes(plane)
    b_split = zbytes(np.packbits(mism)) + zbytes(tgt[mism].astype(np.uint8))
    return min(b_plane, b_split)


# --------------------------------------------------------------------------- #
# Main measurement loop.
# --------------------------------------------------------------------------- #
def run(cache: Path, n_pairs: int | None, fit_pairs: int, out_dir: Path) -> dict:
    t0 = time.time()
    z = np.load(cache, mmap_mode="r")
    L_all = np.asarray(z["lstars"])
    poses = np.asarray(z["gt_poses"], np.float64)
    if n_pairs is not None:
        L_all = L_all[:n_pairs]
        poses = poses[:n_pairs]
    L = L_all.astype(np.uint8)
    P, Hh, Ww = L.shape
    K = MW.intrinsics_at(Ww, Hh)
    Kinv = np.linalg.inv(K)
    tgt_grid = MW._target_grid(Hh, Ww)

    # --- calibration: 3 global scalars on Road+Lane, subset of transitions ---
    nfit = min(fit_pairs, P)
    fit = MW.fit_calibration(L[:nfit].astype(np.int64), poses[:nfit], K, Kinv, tgt_grid,
                             fit_classes=MW.ROAD_PLANE_CLASSES, full_coverage=True)
    params = (fit["s_t"], fit["s_r"], fit["pitch"])

    # --- accumulators ---
    ag = {pr: {s: [0, 0] for s in STRATA} for pr in PREDICTORS}          # [ne, tot]
    ag_total = {pr: [0, 0] for pr in PREDICTORS}
    resid_strata = {pr: dict.fromkeys(STRATA, 0) for pr in PREDICTORS}      # mismatch px per stratum
    edge_geo = {pr: {"n": 0, "d0": 0, "d1": 0, "d2": 0, "dsum": 0.0, "dropped_pairs": 0}
                for pr in PREDICTORS}
    sad_match = {pr: {"n_tgt": 0, "matched": 0} for pr in PREDICTORS}
    rate = {"naive": zbytes(L[0]), "traj": {pr: zbytes(L[0]) for pr in PREDICTORS}}
    stratum_px = dict.fromkeys(STRATA, 0)

    for p in range(P - 1):
        src, tgt = L[p], L[p + 1]
        rate["naive"] += zbytes(tgt)
        preds = {
            "persist": src,
            "ground": predict_ground(src, poses[p + 1], K, Kinv, tgt_grid, params),
            "screw": predict_screw(src, poses[p + 1], K, Kinv, tgt_grid, params),
        }
        sm = strata_masks(tgt)
        for s in STRATA:
            stratum_px[s] += int(sm[s].sum())
        tgt_cracks = pair_crack_sites(tgt)
        _, tgt_sad = saddle_plaquettes(tgt)

        for pr, pred in preds.items():
            ne = pred != tgt
            ag_total[pr][0] += int(ne.sum())
            ag_total[pr][1] += tgt.size
            for s in STRATA:
                m = sm[s]
                ag[pr][s][0] += int(ne[m].sum())
                ag[pr][s][1] += int(m.sum())
                resid_strata[pr][s] += int((ne & m).sum())
            rate["traj"][pr] += residual_bytes(pred, tgt)

            # edge geometric transport: distance from target crack sites to the
            # predicted same-pair separatrix (delta(s) offset residual).
            pred_cracks = pair_crack_sites(pred)
            eg = edge_geo[pr]
            for k, tmask in tgt_cracks.items():
                pmask = pred_cracks.get(k)
                nt = int(tmask.sum())
                if pmask is None or not pmask.any():
                    eg["dropped_pairs"] += 1
                    eg["n"] += nt
                    eg["dsum"] += float(nt * max(Hh, Ww))  # pair absent -> max penalty
                    continue
                dist = ndimage.distance_transform_edt(~pmask)
                dv = dist[tmask]
                eg["n"] += nt
                eg["d0"] += int((dv <= 0.0).sum())
                eg["d1"] += int((dv <= 1.0).sum())
                eg["d2"] += int((dv <= 2.0).sum())
                eg["dsum"] += float(dv.sum())

            # saddle transport: same-signature match within 2 px.
            _, pred_sad = saddle_plaquettes(pred)
            by_sig: dict[tuple[int, ...], list[tuple[float, float]]] = {}
            for y, x, sig in pred_sad:
                by_sig.setdefault(sig, []).append((y, x))
            trees = {sig: cKDTree(np.asarray(v)) for sig, v in by_sig.items()}
            sm_ = sad_match[pr]
            for y, x, sig in tgt_sad:
                sm_["n_tgt"] += 1
                tr = trees.get(sig)
                if tr is not None and tr.query([y, x], k=1)[0] <= 2.0:
                    sm_["matched"] += 1

    # --- finalize ---
    n_tr = P - 1
    xi_bytes_gross = P * 6 * 2  # fp16 x 6/pair (the dxi sidecar shape)
    out = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS advisory / research-signal] PRE-R label-space; zlib rate = PROXY",
        "score_claim": False,
        "promotable": False,
        "cache": str(cache),
        "n_pairs": P,
        "n_transitions": n_tr,
        "grid_hw": [Hh, Ww],
        "calibration": {**fit, "fit_pairs": nfit, "note": "3 global scalars (s_t,s_r,pitch) "
                        "fit on Road+Lane full-coverage, subset of transitions; per-transition "
                        "variation is 100% from the stored pose (grok probe discipline)"},
        "stratum_target_px_share": {s: stratum_px[s] / max(sum(stratum_px.values()), 1)
                                    for s in STRATA},
        "per_stratum_agreement": {
            pr: {s: (1.0 - ag[pr][s][0] / max(ag[pr][s][1], 1)) for s in STRATA}
            for pr in PREDICTORS},
        "per_stratum_mismatch_rate": {
            pr: {s: ag[pr][s][0] / max(ag[pr][s][1], 1) for s in STRATA}
            for pr in PREDICTORS},
        "total_dseg_transport": {pr: ag_total[pr][0] / max(ag_total[pr][1], 1)
                                 for pr in PREDICTORS},
        "residual_px_stratum_share": {
            pr: {s: resid_strata[pr][s] / max(sum(resid_strata[pr].values()), 1)
                 for s in STRATA} for pr in PREDICTORS},
        "edge_geometric": {
            pr: {
                "frac_d0": edge_geo[pr]["d0"] / max(edge_geo[pr]["n"], 1),
                "frac_d_le1": edge_geo[pr]["d1"] / max(edge_geo[pr]["n"], 1),
                "frac_d_le2": edge_geo[pr]["d2"] / max(edge_geo[pr]["n"], 1),
                "mean_offset_px": edge_geo[pr]["dsum"] / max(edge_geo[pr]["n"], 1),
                "dropped_pairs": edge_geo[pr]["dropped_pairs"],
            } for pr in PREDICTORS},
        "saddle_transport": {
            pr: {"n_target_saddles": sad_match[pr]["n_tgt"],
                 "matched_frac_r2_same_signature":
                     sad_match[pr]["matched"] / max(sad_match[pr]["n_tgt"], 1)}
            for pr in PREDICTORS},
        "rate_proxy_zlib9": {
            "naive_per_frame_bytes_total": rate["naive"],
            "trajectory_bytes_total": {
                pr: {"partition0_plus_residuals": rate["traj"][pr],
                     "xi_sidecar_gross_bytes": xi_bytes_gross,
                     "xi_marginal_bytes": 0,
                     "calibration_scalars": 3,
                     "amortization_ratio_vs_naive":
                         rate["naive"] / max(rate["traj"][pr], 1)}
                for pr in PREDICTORS},
            "note": "residual = min(sentinel-plane zlib9, packbits(mask)+values zlib9); "
                    "xi marginal 0 because the 6 fp16/pair dxi sidecar is ALREADY stored "
                    "for d_pose (R1 #238).",
        },
        "elapsed_sec": time.time() - t0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path,
                    default=REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--n-pairs", type=int, default=None,
                    help="limit pairs (smoke); default = ALL pairs in the cache (n600)")
    ap.add_argument("--fit-pairs", type=int, default=100,
                    help="transitions used for the 3-scalar calibration fit")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    res = run(args.cache, args.n_pairs, args.fit_pairs, args.out_dir)
    print(json.dumps({k: res[k] for k in
                      ("n_pairs", "calibration", "per_stratum_agreement",
                       "total_dseg_transport", "edge_geometric", "saddle_transport",
                       "rate_proxy_zlib9", "elapsed_sec")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
