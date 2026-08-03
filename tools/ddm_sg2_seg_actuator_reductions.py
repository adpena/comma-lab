# SPDX-License-Identifier: MIT
"""ddm_sg2 -- second anchor for the margin-density law, and the seg ACTUATOR leverage map.

Everything here is a $0 reduction of already-cached arrays. **No SegNet or PoseNet forward or
backward is fired.** The n600 evaluator slot is not touched. `score_claim=false`.

WHAT THIS ANSWERS
-----------------
`ddm_tl1` measured, from one artifact and one statistic, the source-margin density

    rho = 0.0282 +/- 0.0003 flat on t in [0, 0.2]      ==>   d_seg ~= rho * r

and deliberately did NOT mint it as a law, because it had exactly ONE anchor. This tool supplies
the owed second anchor along the three axes that are reachable without a scorer pass, and then
prices the actuator that moves `r`.

  T1  rho per FRAME (600 quasi-independent scenes, not one pooled number).  The pooled value is a
      mean; per `m88` a mean over a skewed population can be an artifact of a few frames.  If rho
      is a law it must hold frame-by-frame.
  T2  rho on the two DISJOINT cached partitions of the pair set (strided n200 / heldout n400).
      Labeled honestly: a partition of the same producer, so this is a population-variance check,
      NOT an independent producer.
  T3  rho predicted by a LOCAL COAREA estimator -- a genuinely different statistic computed from
      boundary-crossing pairs only, never from the global histogram.  For a field growing linearly
      away from its zero set, a 1-D scan gives

          rho = (1/N) * sum_over_crossings 2 / g_c ,      g_c = |dm/dx| at the crossing,

      and for a locally linear field the two pixels straddling a crossing give g_c = m_p + m_q
      EXACTLY.  This predicts a GLOBAL CDF slope out to t=0.15 from PURELY LOCAL boundary data, so
      agreement validates the linear-growth mechanism the "no wall" claim rests on -- it is not a
      restatement of the histogram.

  T4  the ACTUATOR.  `sR` (cached, n600) is the through-R fragility-weighted input sensitivity
      S_R(i) = |d(sum_p w_p margin_p)/dx_i|, w = exp(-margin/tau).  The reach the render delivers is
      r ~ sum_i S_R(i)*|delta_i|, so the seg actuator has exactly TWO knobs:
          (1) shrink |delta|      -- costs bytes
          (2) move |delta| off high-S_R pixels -- costs ZERO bytes
      Knob (2)'s ceiling is set by how CONCENTRATED S_R is.  This measures that concentration
      exactly (Lorenz curve), plus the blind fraction and the S_R-vs-margin coupling.

CAVEAT CARRIED ON EVERY sR NUMBER
---------------------------------
The cached `sR` is `clip(S_R / percentile(S_R,99), 0, 1)` -- per-frame normalized AND CLIPPED, so
the top ~1% of pixels are saturated at 1.0 and the per-frame absolute scale is NOT recoverable.
Consequence, stated in the direction it biases: clipping REMOVES tail mass, so every concentration
number below is a LOWER BOUND on the true concentration, and every byte-neutral ceiling derived
from it is a LOWER BOUND on the true ceiling.  Cross-frame absolute comparison is refused.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "experiments/results/mlx_fleet_gt_cache"
GT_N600 = CACHE / "gt_n600.npz"
SR_N600 = CACHE / "gt_n600_sR.npz"
GT_HELDOUT = CACHE / "gt_heldout_n400.npz"
GT_STRIDED = CACHE / "gt_strided_n200.npz"

# tl1's pooled anchor, quoted for comparison (NOT re-used as an input anywhere).
TL1_RHO = 0.0282
# The interval tl1 declared the law over, and which contains both endpoints of the seg gap.
LAW_LO, LAW_HI = 0.0, 0.2
# Operating point + floor, DERIVED by tl1 from the gap decomposition (quoted, not re-derived).
T_STAR_LIVE = 0.153053
T_STAR_FLOOR = 0.010518


def _load_member(path: Path, key: str) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    if key not in z.files:
        raise KeyError(f"{path.name} has no member {key!r} (has {z.files})")
    return z[key]


# ---------------------------------------------------------------------------------------------
# T1/T2 -- rho by frame and by partition
# ---------------------------------------------------------------------------------------------
def rho_of_block(margins: np.ndarray, lo: float = LAW_LO, hi: float = LAW_HI) -> float:
    """Pooled density over [lo,hi] for a (P,H,W) margin block: P(lo<=m<hi)/(hi-lo)."""
    n = margins.size
    cnt = int(np.count_nonzero((margins >= lo) & (margins < hi)))
    return cnt / n / (hi - lo)


def rho_per_frame(margins: np.ndarray, lo: float = LAW_LO, hi: float = LAW_HI) -> np.ndarray:
    """Per-frame density -> (P,).  Each frame is a different scene => a quasi-independent anchor."""
    per_px = margins.shape[1] * margins.shape[2]
    out = np.empty(margins.shape[0], dtype=np.float64)
    for i in range(margins.shape[0]):
        m = margins[i]
        out[i] = np.count_nonzero((m >= lo) & (m < hi)) / per_px / (hi - lo)
    return out


def cdf_at(margins: np.ndarray, ts: list[float]) -> dict[str, float]:
    """Exact P(margin < t) for each t, computed by counting (no interpolation)."""
    n = margins.size
    return {f"{t:g}": int(np.count_nonzero(margins < t)) / n for t in ts}


# ---------------------------------------------------------------------------------------------
# T3 -- the local coarea estimator (a DIFFERENT statistic, boundary-local only)
# ---------------------------------------------------------------------------------------------
def coarea_rho(margins: np.ndarray, lstars: np.ndarray) -> dict:
    """Predict rho from boundary-crossing pairs alone.

    Along a grid line, m dips to 0 at each class crossing and (locally) grows linearly with slope
    g on each side.  Pixels of that row with m < t number 2t/g per crossing, so

        rho = (1/N_px) * sum_crossings 2/g .

    For a locally linear field the straddling pair gives g = m_p + m_q exactly.  Computed
    independently along x and along y; the two must agree if the model holds.

    No upper cutoff on g is applied (a steep crossing simply contributes little).  The only
    exclusion is the DEGENERATE g == 0 end -- an exact tie on both sides -- and those are COUNTED
    and reported so the exclusion is visible rather than silent.
    """
    P, H, W = margins.shape
    n_px = P * H * W
    res = {}
    for axis, name in ((2, "x"), (1, "y")):
        tot = 0.0
        n_cross = 0
        n_deg = 0
        for i in range(P):
            m = margins[i].astype(np.float64)
            lab = lstars[i]
            if axis == 2:
                diff = lab[:, 1:] != lab[:, :-1]
                g = m[:, 1:] + m[:, :-1]
            else:
                diff = lab[1:, :] != lab[:-1, :]
                g = m[1:, :] + m[:-1, :]
            gc = g[diff]
            n_cross += int(gc.size)
            good = gc > 0
            n_deg += int(gc.size - np.count_nonzero(good))
            gc = gc[good]
            tot += float(np.sum(2.0 / gc))
        res[f"rho_coarea_{name}"] = tot / n_px
        res[f"crossings_{name}"] = n_cross
        res[f"degenerate_zero_g_{name}"] = n_deg
    res["rho_coarea_mean"] = 0.5 * (res["rho_coarea_x"] + res["rho_coarea_y"])
    res["exclusion_note"] = "no upper cutoff on g; only exact g==0 crossings excluded (counted above)"
    return res


# ---------------------------------------------------------------------------------------------
# T4 -- the actuator: how concentrated is the through-R leverage?
# ---------------------------------------------------------------------------------------------
def sr_concentration(sr: np.ndarray, *, quantiles: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50)) -> dict:
    """Lorenz curve of S_R mass, per frame then aggregated.

    Returns, for each q: `mass_in_top_q` (share of total S_R carried by the top q of pixels) and
    `ceiling_bottom_q` = mean(S_R) / mean(S_R over the bottom q of pixels) -- the byte-NEUTRAL
    factor by which the delivered reach would fall if the same total |delta| could be relocated
    onto the least-sensitive q of render pixels.  That is a CEILING, not a realizable gain.
    """
    P = sr.shape[0]
    n_px = sr.shape[1] * sr.shape[2]
    top_mass = {q: [] for q in quantiles}
    bot_ceiling = {q: [] for q in quantiles}
    blind = {thr: [] for thr in (0.001, 0.01, 0.05, 0.10)}
    means = []
    sat = []
    for i in range(P):
        v = np.sort(sr[i].ravel().astype(np.float64))  # ascending
        tot = float(v.sum())
        mean = tot / n_px
        means.append(mean)
        sat.append(float(np.count_nonzero(sr[i] >= 1.0)) / n_px)
        if tot <= 0:
            continue
        csum = np.cumsum(v)
        for q in quantiles:
            k = max(1, round(q * n_px))
            bottom_sum = float(csum[k - 1])
            top_mass[q].append((tot - float(csum[n_px - k - 1])) / tot)
            bot_mean = bottom_sum / k
            bot_ceiling[q].append(mean / bot_mean if bot_mean > 0 else float("inf"))
        for thr in blind:
            blind[thr].append(float(np.count_nonzero(sr[i] < thr)) / n_px)

    def _stat(a):
        a = np.asarray([x for x in a if np.isfinite(x)], dtype=np.float64)
        if a.size == 0:
            return None
        return {"median": float(np.median(a)), "mean": float(a.mean()),
                "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}

    return {
        "n_frames": P,
        "sR_mean_per_frame": _stat(means),
        "saturated_at_1p0_frac": _stat(sat),
        "mass_in_top_q": {f"{q:g}": _stat(top_mass[q]) for q in quantiles},
        "byte_neutral_ceiling_bottom_q": {f"{q:g}": _stat(bot_ceiling[q]) for q in quantiles},
        "blind_frac_below": {f"{t:g}": _stat(blind[t]) for t in blind},
    }


def sr_vs_margin(sr: np.ndarray, margins: np.ndarray, *, stride: int = 5) -> dict:
    """Coupling between where the scorer is FRAGILE (low margin, output side) and where it is
    LEVERAGED (high S_R, input side).  Both live on the same 384x512 grid but are different
    objects, so this is a spatial-coupling measurement, not a per-pixel identity.

    Reported as conditional mean S_R per margin bucket + a Spearman rank correlation on a
    strided subsample (stride keeps this cheap and avoids a 118M-element argsort).
    """
    edges = sorted({0.0, 1e-3, 1e-2, 0.05, 0.1, T_STAR_LIVE, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf})
    P = sr.shape[0]
    sums = np.zeros(len(edges) - 1, dtype=np.float64)
    cnts = np.zeros(len(edges) - 1, dtype=np.int64)
    xs, ys = [], []
    for i in range(P):
        m = margins[i].ravel()
        s = sr[i].ravel().astype(np.float64)
        idx = np.digitize(m, edges) - 1
        np.clip(idx, 0, len(edges) - 2, out=idx)
        sums += np.bincount(idx, weights=s, minlength=len(edges) - 1)
        cnts += np.bincount(idx, minlength=len(edges) - 1)
        if i % 20 == 0:  # subsample frames too, for the rank correlation only
            xs.append(m[::stride])
            ys.append(s[::stride])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    spearman = float((rx * ry).sum() / math.sqrt((rx * rx).sum() * (ry * ry).sum()))
    overall = float(sums.sum() / max(cnts.sum(), 1))
    tot_mass = float(sums.sum())
    tot_px = int(cnts.sum())
    buckets = [
        {"margin_lo": float(edges[k]), "margin_hi": float(edges[k + 1]),
         "n": int(cnts[k]),
         "px_share": float(cnts[k] / tot_px) if tot_px else None,
         "mean_sR": (float(sums[k] / cnts[k]) if cnts[k] else None),
         "ratio_to_overall": (float(sums[k] / cnts[k] / overall) if cnts[k] and overall else None),
         "leverage_mass_share": (float(sums[k] / tot_mass) if tot_mass else None)}
        for k in range(len(edges) - 1)
    ]
    # The decisive roll-up: how much of the through-R INPUT leverage sits on pixels whose own
    # OUTPUT margin puts them structurally out of reach of flipping.  tl1's "0.42% scored-active"
    # is an OUTPUT-side statistic; this is the INPUT-side one the actuator actually faces.
    def _mass_below(t):
        return float(sum(b["leverage_mass_share"] for b in buckets if b["margin_hi"] <= t))

    def _px_below(t):
        return float(sum(b["px_share"] for b in buckets if b["margin_hi"] <= t))

    rollup = {}
    for t in (0.153053, 0.25, 2.0, 4.0):
        pxs, mss = _px_below(t), _mass_below(t)
        rollup[f"margin_lt_{t:g}"] = {
            "px_share": pxs, "leverage_mass_share": mss,
            "leverage_concentration_ratio": (mss / pxs) if pxs else None,
        }
    rollup["margin_ge_4_leverage_mass_share"] = 1.0 - _mass_below(4.0)
    return {
        "spearman_margin_vs_sR": spearman,
        "spearman_n": int(x.size),
        "overall_mean_sR": overall,
        "conditional_mean_sR_by_margin_bucket": buckets,
        "leverage_rollup": rollup,
    }


# ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ddm_sg2 $0 reductions: rho second anchor + seg actuator leverage.")
    ap.add_argument("--task", choices=("rho", "coarea", "actuator", "all"), default="all")
    ap.add_argument("--num-pairs", type=int, default=None, help="limit frames (default: all cached)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    out: dict = {"tool": "ddm_sg2_seg_actuator_reductions", "scorer_fired": False,
                 "score_claim": False, "axis": "[macOS-CPU advisory] $0 cached-array reduction"}

    if args.task in ("rho", "all"):
        m = _load_member(GT_N600, "margins")
        if args.num_pairs:
            m = m[: args.num_pairs]
        pooled = rho_of_block(m)
        pf = rho_per_frame(m)
        out["T1_rho"] = {
            "n_frames": int(m.shape[0]),
            "pooled_rho_0_to_0p2": pooled,
            "tl1_pooled_rho": TL1_RHO,
            "pooled_vs_tl1_rel": pooled / TL1_RHO - 1.0,
            "per_frame": {
                "median": float(np.median(pf)), "mean": float(pf.mean()), "std": float(pf.std()),
                "cv": float(pf.std() / pf.mean()),
                "min": float(pf.min()), "max": float(pf.max()),
                "p05": float(np.percentile(pf, 5)), "p95": float(np.percentile(pf, 95)),
                "max_over_min": float(pf.max() / pf.min()),
                "frac_within_10pct_of_pooled": float(np.mean(np.abs(pf / pooled - 1) < 0.10)),
                "frac_within_25pct_of_pooled": float(np.mean(np.abs(pf / pooled - 1) < 0.25)),
            },
            "cdf_exact": cdf_at(m, [1e-4, 1e-3, 1e-2, 0.1, T_STAR_FLOOR, T_STAR_LIVE, 0.25, 2.0]),
        }
        del m

        parts = {}
        for name, path in (("strided_n200", GT_STRIDED), ("heldout_n400", GT_HELDOUT)):
            if path.exists():
                mm = _load_member(path, "margins")
                parts[name] = {"n_frames": int(mm.shape[0]), "rho_0_to_0p2": rho_of_block(mm)}
                del mm
        out["T2_partitions"] = {
            "note": "DISJOINT pair partitions of the SAME producer (stride-3 subset + complement); "
                    "this is a population-variance check, NOT an independent producer.",
            **parts,
        }

    if args.task in ("coarea", "all"):
        n = args.num_pairs or 600
        m = _load_member(GT_N600, "margins")[:n]
        lab = _load_member(GT_N600, "lstars")[:n].astype(np.int16)
        ca = coarea_rho(m, lab)
        pooled = rho_of_block(m)
        ca["rho_histogram_same_block"] = pooled
        ca["coarea_over_histogram"] = ca["rho_coarea_mean"] / pooled
        ca["x_over_y_isotropy"] = ca["rho_coarea_x"] / ca["rho_coarea_y"]
        out["T3_coarea"] = ca
        del m, lab

    if args.task in ("actuator", "all"):
        sr = _load_member(SR_N600, "sR")
        if args.num_pairs:
            sr = sr[: args.num_pairs]
        out["T4_actuator"] = sr_concentration(sr)
        m = _load_member(GT_N600, "margins")[: sr.shape[0]]
        out["T4_actuator"]["coupling"] = sr_vs_margin(sr, m)
        out["T4_actuator"]["clip_caveat"] = (
            "sR is clip(S_R/p99,0,1): top ~1% saturated, per-frame scale unrecoverable. "
            "Clipping removes tail mass => concentration and ceilings below are LOWER BOUNDS.")
        del sr, m

    print(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
