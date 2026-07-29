#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_of1 — the two gc6 $0 coherence probes (pure array analysis, NO scorer).

Both probes read data already on disk: ru1's flip atlas
(``atlas_flat.npz``: per-flip pair,y,x,gt/realized class,m_def,gap12,gt_margin,
dist_bin,gt_flicker) and the gt_n600 GT argmax cache (``lstars`` memmap). NO
SegNet/PoseNet forward runs, NO training, NO launches (pb1 owns the scorer slot).

PROBE offset (P2C-OF) — is the deep residual band a COHERENT 1-D offset field?
  Reconstruct the flip mask per pair, connected-component it (the flip *bands*),
  measure per-band PCA arclength + band thickness tau, the within-band thickness
  autocorrelation length L along arclength, and the DECISIVE number: predicted
  flips-fixed per offset-basis DOF = total_flips / total_DOF (DOF = sum over
  bands of ceil(arclen / L)) vs ru1's +24 flips/quantum aimed-edit currency.
  Falsifier: autocorr length <= ~3 px OR conditional ~= marginal entropy =>
  incoherent jitter, offset-field solve DEAD at FORMULATION scope. Stratified by
  m_def band (deep>0.25 = the offset field's claimed home) and by class-pair
  (the Fridrich lane-corridor stored-stream exception is Lane<->Road specific).

PROBE flicker (W1-COH) — does GT flicker flip PHASE-COHERENTLY by region?
  Per pair, flicker field = (lstars[p] != lstars[nb]); connected-component it;
  per component measure the majority-transition fraction (phase agreement);
  the implied per-region phase-bit budget in bytes (1 bit/region-instance upper
  bound); the pessimistic support-transmission cost (flicker-mask entropy); and
  the tail re-price B/err = phase_bytes / flicker_flips_fixed vs the 1.2731
  water. Falsifier: area-weighted phase agreement < ~0.8 (incoherent) =>
  region-phase pricing DEAD at FORMULATION scope.

Axis: [macOS-CPU advisory]. score_claim=false, promotion_eligible=false.
Consumers: pb1 P2c round-2 / attack-search arm / E2 tree N3-N4 / r7 flicker
channel / c1 waterfill. Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
H, W = 384, 512
N_PAIRS = 600
STRUCT8 = np.ones((3, 3), dtype=int)
CLASS_NAMES = ("Road", "Lane", "Undriv", "Movable", "MyCar")
WATER_B_PER_ERR = 1.2731  # gc6/box: incumbent per-error rate floor (bytes/error)
RU1_FLIPS_PER_QUANTUM = 24.0  # ru1 best-of-8 single-quantum aimed-edit yield


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atlas", required=True, type=Path)
    ap.add_argument("--gt-cache", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--probe", choices=("offset", "flicker", "both"),
                    default="both")
    ap.add_argument("--maxlag", type=int, default=12,
                    help="max arclength lag for the autocorr sweep (px)")
    return ap.parse_args()


def _load_lstars(gt_cache: Path):
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    return open_stored_npy_memmap(gt_cache, "lstars")


def _entropy_bits(counts: np.ndarray) -> float:
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    return float(-(p * np.log2(p)).sum())


def _pca_arclen(yy: np.ndarray, xx: np.ndarray) -> int:
    """Arclength (px) of a component along its PCA major axis."""
    a = yy.size
    if a < 3:
        return int(a)
    pts = np.stack([yy, xx], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = pts.T @ pts / a
    _, evec = np.linalg.eigh(cov)
    proj = pts @ evec[:, 1]
    return max(1, round(float(proj.max() - proj.min())) + 1)


def _thickness_seq(yy: np.ndarray, xx: np.ndarray):
    """Per-arclength-bin thickness sequence tau(s) along the PCA major axis."""
    a = yy.size
    if a < 3:
        return None
    pts = np.stack([yy, xx], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = pts.T @ pts / a
    _, evec = np.linalg.eigh(cov)
    s = pts @ evec[:, 1]
    s = s - s.min()
    nb = int(np.floor(s.max())) + 1
    if nb < 3:
        return None
    return np.bincount(np.floor(s).astype(int), minlength=nb).astype(np.float64)


def probe_offset(atlas, lstars, maxlag: int) -> dict:
    """P2C-OF: connected flip-band geometry + offset-field coherence."""
    pair = atlas["pair"]
    order = np.argsort(pair, kind="stable")
    pair_s = pair[order]
    ys_s = atlas["y"][order].astype(np.int64)
    xs_s = atlas["x"][order].astype(np.int64)
    md_s = atlas["m_def"][order].astype(np.float64)
    gtc_s = atlas["gt_class"][order].astype(np.int64)
    rlc_s = atlas["realized_class"][order].astype(np.int64)
    dist_s = atlas["dist_bin"][order].astype(np.int64)
    bounds = np.searchsorted(pair_s, np.arange(N_PAIRS + 1))

    # band geometry, stratified by m_def band
    strata = {"all": [], "deep": [], "shallow": []}  # each: (area, arclen)
    # class-pair band geometry (undirected gt<->realized), Lane<->Road tracked
    classpair_area: dict[tuple[int, int], list[int]] = {}
    classpair_arclen: dict[tuple[int, int], list[int]] = {}
    # within-band thickness autocorr (num/den over arclength lags)
    ac_num = np.zeros(maxlag + 1)
    ac_den = np.zeros(maxlag + 1)
    ac_num_d = np.zeros(maxlag + 1)
    ac_den_d = np.zeros(maxlag + 1)
    # conditional-entropy pair counts of quantized thickness along arclength
    tmax = 16
    joint = np.zeros((tmax + 1, tmax + 1), dtype=np.int64)
    marg = np.zeros(tmax + 1, dtype=np.int64)

    total_boundary_px = 0
    total_flips = 0
    dist_hist = np.zeros(3, dtype=np.int64)

    for p in range(N_PAIRS):
        gt = np.asarray(lstars[p], dtype=np.int64)
        # GT inter-class boundary pixel count = total contour arclength proxy
        bnd = np.zeros((H, W), dtype=bool)
        bnd[:-1, :] |= gt[:-1, :] != gt[1:, :]
        bnd[1:, :] |= gt[1:, :] != gt[:-1, :]
        bnd[:, :-1] |= gt[:, :-1] != gt[:, 1:]
        bnd[:, 1:] |= gt[:, 1:] != gt[:, :-1]
        total_boundary_px += int(bnd.sum())

        lo, hi = bounds[p], bounds[p + 1]
        fy, fx = ys_s[lo:hi], xs_s[lo:hi]
        if fy.size == 0:
            continue
        total_flips += fy.size
        dist_hist += np.bincount(dist_s[lo:hi], minlength=3)
        fmd = md_s[lo:hi]
        fgt, frl = gtc_s[lo:hi], rlc_s[lo:hi]

        flip = np.zeros((H, W), dtype=bool)
        flip[fy, fx] = True
        md_map = np.zeros((H, W), dtype=np.float64)
        md_map[fy, fx] = fmd
        # per-pixel unordered class-pair key for class-pair stratification
        cpkey = np.minimum(fgt, frl) * 5 + np.maximum(fgt, frl)
        cp_map = np.full((H, W), -1, dtype=np.int64)
        cp_map[fy, fx] = cpkey

        lab, ncomp = ndimage.label(flip, structure=STRUCT8)
        if ncomp == 0:
            continue
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            sub = lab[sl] == i
            yy, xx = np.nonzero(sub)
            area = int(yy.size)
            arclen = _pca_arclen(yy, xx)
            mean_md = float(md_map[sl][sub].mean())
            strata["all"].append((area, arclen))
            cat = "deep" if mean_md > 0.25 else "shallow"
            strata[cat].append((area, arclen))
            # class-pair (majority key of the band)
            keys = cp_map[sl][sub]
            kmaj = np.bincount(keys[keys >= 0]).argmax() if (keys >= 0).any() \
                else -1
            if kmaj >= 0:
                ck = (int(kmaj) // 5, int(kmaj) % 5)
                classpair_area.setdefault(ck, []).append(area)
                classpair_arclen.setdefault(ck, []).append(arclen)
            # within-band thickness autocorr + conditional entropy
            tau = _thickness_seq(yy, xx)
            if tau is not None:
                tc = tau - tau.mean()
                v = float((tc * tc).mean())
                if v > 0:
                    for lag in range(min(maxlag, tau.size - 1) + 1):
                        prod = tc[:tau.size - lag] * tc[lag:]
                        ac_num[lag] += prod.sum()
                        ac_den[lag] += prod.size
                        if mean_md > 0.25:
                            ac_num_d[lag] += prod.sum()
                            ac_den_d[lag] += prod.size
                tq = np.clip(tau.astype(np.int64), 0, tmax)
                marg += np.bincount(tq, minlength=tmax + 1)
                flat = tq[:-1] * (tmax + 1) + tq[1:]
                joint += np.bincount(
                    flat, minlength=(tmax + 1) ** 2).reshape(
                        tmax + 1, tmax + 1)

    def agg(rows):
        a = np.array([r[0] for r in rows], dtype=np.float64)
        length = np.array([r[1] for r in rows], dtype=np.float64)
        tot = a.sum()
        return a, length, tot

    def dof_report(a, length, tot):
        out = {}
        for lname, lval in (("L1", 1.0), ("L3", 3.0)):
            dof = np.ceil(length / lval)
            dof = np.maximum(dof, 1.0).sum()
            out[f"flips_per_dof_{lname}"] = float(tot / dof) if dof else 0.0
        # most-generous: one constant offset per whole band
        out["flips_per_dof_perband_const"] = float(tot / len(a)) if len(a) else 0.0
        return out

    strat_stats = {}
    for cat, rows in strata.items():
        if not rows:
            continue
        a, length, tot = agg(rows)
        strat_stats[cat] = {
            "n_components": int(a.size),
            "total_flips": int(tot),
            "area_mean": float(a.mean()),
            "area_p50": float(np.median(a)),
            "area_p90": float(np.percentile(a, 90)),
            "area_p99": float(np.percentile(a, 99)),
            "arclen_mean": float(length.mean()),
            "arclen_p90": float(np.percentile(length, 90)),
            "thickness_tau_mean": float(tot / length.sum()),
            "mass_frac_area_ge16": float(a[a >= 16].sum() / tot),
            "mass_frac_area_ge32": float(a[a >= 32].sum() / tot),
            "singleton_mass_frac": float(a[a == 1].sum() / tot),
            **dof_report(a, length, tot),
        }

    def ac_L(num, den):
        ac = num / np.maximum(den, 1)
        ac = ac / ac[0] if ac[0] != 0 else ac
        below = np.where(ac[1:] < 1.0 / np.e)[0]
        Lc = int(below[0] + 1) if below.size else maxlag + 1
        return ac.tolist(), Lc

    ac_all, L_all = ac_L(ac_num, ac_den)
    ac_deep, L_deep = ac_L(ac_num_d, ac_den_d)

    h_marg = _entropy_bits(marg.astype(np.float64))
    h_joint = _entropy_bits(joint.astype(np.float64).ravel())
    h_cond = h_joint - _entropy_bits(joint.sum(axis=1).astype(np.float64))

    classpairs = []
    for ck, alist in sorted(classpair_area.items(),
                            key=lambda kv: -sum(kv[1])):
        a = np.array(alist, dtype=np.float64)
        length = np.array(classpair_arclen[ck], dtype=np.float64)
        tot = a.sum()
        classpairs.append({
            "class_pair": f"{CLASS_NAMES[ck[0]]}<->{CLASS_NAMES[ck[1]]}",
            "total_flips": int(tot),
            "n_bands": int(a.size),
            "arclen_mean": float(length.mean()),
            "arclen_p90": float(np.percentile(length, 90)),
            "area_mean": float(a.mean()),
            "flips_per_dof_perband_const": float(tot / a.size),
        })

    return {
        "schema": "ddm_of1_offset_coherence.v1",
        "evidence_axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "total_flips": int(total_flips),
        "total_gt_boundary_px": int(total_boundary_px),
        "mean_band_thickness_flips_per_boundary_px": float(
            total_flips / total_boundary_px),
        "dist_bin_frac": {
            "on_gt_boundary": float(dist_hist[0] / total_flips),
            "near_3px": float(dist_hist[1] / total_flips),
            "interior": float(dist_hist[2] / total_flips),
            "within_3px_cumulative": float(
                (dist_hist[0] + dist_hist[1]) / total_flips),
        },
        "band_geometry_by_mdef": strat_stats,
        "within_band_thickness_autocorr": {
            "all_lag0_to_maxlag": ac_all,
            "deep_lag0_to_maxlag": ac_deep,
            "autocorr_length_px_all": L_all,
            "autocorr_length_px_deep": L_deep,
            "note": "L = first lag where autocorr < 1/e",
        },
        "offset_field_entropy_bits_per_node": {
            "marginal_H0": float(h_marg),
            "conditional_H1_given_prev": float(h_cond),
            "coherence_mutual_info": float(h_marg - h_cond),
            "note": ("thickness quantized to integer px; conditional ~= "
                     "marginal => white field / no coherence gain"),
        },
        "ru1_flips_per_quantum_calibration": RU1_FLIPS_PER_QUANTUM,
        "class_pairs": classpairs,
    }


def probe_flicker(atlas, lstars, maxlag: int) -> dict:
    """W1-COH: flicker connected-component phase coherence + tail re-price."""
    md = atlas["m_def"].astype(np.float64)
    flick_flag = atlas["gt_flicker"].astype(bool)
    total_flips = int(md.size)
    total_flicker_flips = int(flick_flag.sum())
    deep = md >= 1.0
    deep_flicker_flips = int(flick_flag[deep].sum())

    ncomp_per_pair = np.zeros(N_PAIRS, dtype=np.int64)
    comp_areas = []
    comp_agree = []  # (area, majority_transition_fraction)
    total_flicker_px = 0
    mask_entropy_bits_total = 0.0

    for p in range(N_PAIRS):
        nb = p + 1 if p + 1 < N_PAIRS else p - 1
        gp = np.asarray(lstars[p], dtype=np.int8)
        gn = np.asarray(lstars[nb], dtype=np.int8)
        flick = gp != gn
        nfl = int(flick.sum())
        total_flicker_px += nfl
        # flicker-mask transmission cost (pessimistic support bound)
        rho = nfl / (H * W)
        if 0.0 < rho < 1.0:
            hbit = -(rho * np.log2(rho) + (1 - rho) * np.log2(1 - rho))
            mask_entropy_bits_total += hbit * H * W
        lab, ncomp = ndimage.label(flick, structure=STRUCT8)
        ncomp_per_pair[p] = ncomp
        if ncomp == 0:
            continue
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            sub = lab[sl] == i
            area = int(sub.sum())
            gpp = gp[sl][sub].astype(np.int32)
            gnn = gn[sl][sub].astype(np.int32)
            trans = gpp * 5 + gnn
            _, cnts = np.unique(trans, return_counts=True)
            comp_areas.append(area)
            comp_agree.append((area, cnts.max() / area))

    comp_areas = np.array(comp_areas, dtype=np.float64)
    pa = np.array(comp_agree, dtype=np.float64)  # (n,2)
    areas, agrees = pa[:, 0], pa[:, 1]
    aw_agree = float((areas * agrees).sum() / areas.sum())
    big = areas >= 4
    aw_agree_big = float(
        (areas[big] * agrees[big]).sum() / areas[big].sum())

    total_region_instances = int(ncomp_per_pair.sum())
    phase_bytes = total_region_instances / 8.0  # 1 bit/region-instance upper bnd
    support_bytes_pessimistic = mask_entropy_bits_total / 8.0

    # flips fixed = coherence * flicker-flip mass (area-weighted coherence)
    fixed_all = aw_agree * total_flicker_flips
    fixed_deep = aw_agree * deep_flicker_flips
    b_per_err_all = phase_bytes / fixed_all if fixed_all else float("inf")
    b_per_err_deep = phase_bytes / fixed_deep if fixed_deep else float("inf")
    dseg_reach_all = fixed_all / (H * W * N_PAIRS)
    dseg_reach_deep = fixed_deep / (H * W * N_PAIRS)

    return {
        "schema": "ddm_of1_flicker_phase.v1",
        "evidence_axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "total_flips": total_flips,
        "total_flicker_flips": total_flicker_flips,
        "flicker_flip_frac": float(total_flicker_flips / total_flips),
        "deep_tail_m_def_ge1p0_flips": int(deep.sum()),
        "deep_tail_flicker_flips": deep_flicker_flips,
        "deep_tail_flicker_frac": float(deep_flicker_flips / int(deep.sum())),
        "flicker_pixels_per_pair_mean": float(total_flicker_px / N_PAIRS),
        "flicker_frame_frac": float(total_flicker_px / (N_PAIRS * H * W)),
        "components_per_pair": {
            "mean": float(ncomp_per_pair.mean()),
            "p50": float(np.median(ncomp_per_pair)),
            "p90": float(np.percentile(ncomp_per_pair, 90)),
        },
        "component_area": {
            "n_components": int(comp_areas.size),
            "mean": float(comp_areas.mean()),
            "p50": float(np.median(comp_areas)),
            "p90": float(np.percentile(comp_areas, 90)),
            "p99": float(np.percentile(comp_areas, 99)),
            "max": float(comp_areas.max()),
            "singleton_frac": float((comp_areas == 1).mean()),
        },
        "phase_agreement": {
            "unweighted_mean": float(agrees.mean()),
            "area_weighted_mean": aw_agree,
            "area_weighted_mean_area_ge4": aw_agree_big,
            "area_weighted_frac_ge_0p8": float(
                areas[agrees >= 0.8].sum() / areas.sum()),
            "falsifier_threshold": 0.8,
            "falsifier_fired": bool(aw_agree < 0.8),
        },
        "phase_bit_budget": {
            "total_region_instances": total_region_instances,
            "phase_bytes_1bit_per_region": float(phase_bytes),
            "support_bytes_pessimistic_mask_entropy": float(
                support_bytes_pessimistic),
            "note": ("phase_bytes is the INCREMENTAL cost assuming the region "
                     "support is receiver-derivable; if the support must be "
                     "transmitted, support_bytes_pessimistic dominates and the "
                     "channel is DEAD -> derivability is the binding "
                     "precondition, not coherence"),
        },
        "tail_reprice": {
            "water_b_per_err": WATER_B_PER_ERR,
            "flicker_flips_fixed_all": float(fixed_all),
            "flicker_flips_fixed_deep_tail": float(fixed_deep),
            "b_per_err_all_flicker": float(b_per_err_all),
            "b_per_err_deep_tail_paying_all_regions": float(b_per_err_deep),
            "dseg_reach_all_flicker": float(dseg_reach_all),
            "dseg_reach_deep_tail": float(dseg_reach_deep),
            "beats_water_all": bool(b_per_err_all < WATER_B_PER_ERR),
            "beats_water_deep": bool(b_per_err_deep < WATER_B_PER_ERR),
        },
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lstars = _load_lstars(args.gt_cache)
    atlas = np.load(args.atlas)

    if args.probe in ("offset", "both"):
        t0 = time.time()
        rep = probe_offset(atlas, lstars, args.maxlag)
        rep["wall_seconds"] = round(time.time() - t0, 2)
        out = args.out_dir / "offset_field_coherence_receipt.json"
        out.write_text(json.dumps(rep, indent=1))
        print(f"[offset] {rep['wall_seconds']}s -> {out}")
        print(f"  autocorr L(all)={rep['within_band_thickness_autocorr']['autocorr_length_px_all']}px "
              f"L(deep)={rep['within_band_thickness_autocorr']['autocorr_length_px_deep']}px")
        print(f"  flips/DOF all: L1={rep['band_geometry_by_mdef']['all']['flips_per_dof_L1']:.2f} "
              f"perband={rep['band_geometry_by_mdef']['all']['flips_per_dof_perband_const']:.2f} "
              f"deep perband={rep['band_geometry_by_mdef']['deep']['flips_per_dof_perband_const']:.2f} "
              f"(vs ru1 {RU1_FLIPS_PER_QUANTUM}/quantum)")

    if args.probe in ("flicker", "both"):
        t0 = time.time()
        rep = probe_flicker(atlas, lstars, args.maxlag)
        rep["wall_seconds"] = round(time.time() - t0, 2)
        out = args.out_dir / "flicker_phase_coherence_receipt.json"
        out.write_text(json.dumps(rep, indent=1))
        print(f"[flicker] {rep['wall_seconds']}s -> {out}")
        print(f"  phase agreement area-weighted={rep['phase_agreement']['area_weighted_mean']:.3f} "
              f"(falsifier<0.8 fired={rep['phase_agreement']['falsifier_fired']})")
        print(f"  phase bytes={rep['phase_bit_budget']['phase_bytes_1bit_per_region']:.0f} "
              f"B/err all={rep['tail_reprice']['b_per_err_all_flicker']:.4f} "
              f"deep={rep['tail_reprice']['b_per_err_deep_tail_paying_all_regions']:.4f} "
              f"(water {WATER_B_PER_ERR})")


if __name__ == "__main__":
    main()
