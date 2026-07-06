#!/usr/bin/env python3
"""$0 gate for the LADDER island-birth lever: is lane/movable WINNABLE via a
difficulty homotopy? (read-only; cached GT argmax; no training, no GPU.)

The island-birth failure (measured, memory `why_mod32cap_baseline_has_zero_lane_
movable_islands`): CE won't birth a <1%-area class. LADDER's fix — make the hard
thing winnable in an easier variant, birth it, anneal back — needs the difficulty
gradient to (a) actually make the rare class BIRTHABLE (coherent + above the
CE-birth area threshold), at (b) an acceptable ANNEAL-DEBT (distance to the true
target), along (c) a CONTINUOUS path (no cliff that loses the birthed island).

This probe constructs the homotopy by morphological DILATION of the rare-class GT
mask (radius r: the eased target lets class c claim any pixel within r of a true
class-c pixel) and measures, per rare class, over all 600 frames:

  * area_frac(r)     — eased class area (does it cross the CE-birth threshold ~1-2%?)
  * coherence(r)     — mean largest-connected-component fraction of the class mask
                       (a fat coherent band a coord-INR can cheaply represent, vs speckle)
  * anneal_debt(r)   — mean pixel-disagreement of eased target vs true GT
                       (= how far the anneal must travel back; a d_seg-like distance)
  * step_debt(r)     — disagreement between consecutive radii (path CONTINUITY;
                       a cliff here = the birthed island vanishes in one anneal step)

Verdict: GO if there is a radius r* where the class becomes birthable
(area ≥ ~1.5% AND coherence high) while anneal_debt stays modest and step_debt
stays small (continuous path). NO-GO if area/coherence never reach birthable
(representation wall — dilation can't make a coherent island) or step_debt shows a
cliff (discontinuous — the homotopy would lose the island).

Run: .venv/bin/python tools/lane_winnability_homotopy_smoke.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

_REPO = Path(__file__).resolve().parents[1]
_CACHE = _REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
_OUT = _REPO / "experiments/results/lane_winnability_homotopy_smoke"

# comma10k canonical order (memory: SegNet class index order)
_CLASSES = {0: "road", 1: "lane", 2: "undrivable", 3: "movable", 4: "my_car"}
_RARE = [1, 3]                       # lane, movable
_RADII = [0, 1, 2, 3, 4, 6, 8, 12]
_BIRTH_AREA = 0.015                  # CE-birth threshold (~1.5% area)
_BIRTH_COH = 0.35                    # largest-CC must be ≥35% of the class mask


def _dilate_eased(lab: np.ndarray, c: int, r: int) -> np.ndarray:
    """Isotropic homotopy: class c claims any pixel within r of a true-c pixel."""
    if r == 0:
        return lab
    mask = ndimage.binary_dilation(lab == c, iterations=r)
    out = lab.copy()
    out[mask] = c
    return out


def _curve_eased(lab: np.ndarray, c: int, L: int, w: int = 1) -> np.ndarray:
    """CURVE-PRIOR homotopy (the optimal lane easing): bridge the dash gaps ALONG
    the lane direction, not isotropically. Lane dashes are near-collinear along the
    (near-vertical) lane curve toward the vanishing point, so a vertical CLOSING of
    length L merges the dashes into ONE coherent band while barely widening it — the
    class stays ON its curve (LADDER: ease toward winnable WITHOUT leaving the target
    manifold). w = small isotropic width so the birthed band is a few px thick.
    L=0,w=0 → the true target (dashes)."""
    if L == 0 and w == 0:
        return lab
    m = lab == c
    if L > 0:
        vline = np.ones((2 * L + 1, 1), dtype=bool)      # vertical structuring element
        m = ndimage.binary_closing(m, structure=vline)   # connect dashes along the curve
    if w > 0:
        m = ndimage.binary_dilation(m, iterations=w)     # minimal cross-curve width
    out = lab.copy()
    out[m] = c
    return out


def _coherence(mask: np.ndarray) -> float:
    """Largest connected component as a fraction of the class mask (1.0 = one blob)."""
    tot = int(mask.sum())
    if tot == 0:
        return 0.0
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return float(sizes.max()) / tot


def main() -> int:
    if not _CACHE.exists():
        raise SystemExit(f"cache not found: {_CACHE}")
    z = np.load(_CACHE)
    lstars = np.asarray(z["lstars"])                 # (600,384,512) int
    n = lstars.shape[0]
    prize = {_CLASSES[c]: float((lstars == c).mean()) for c in range(5)}
    print("=== prize (raw GT area fraction; through-R scored prize is ≤ this) ===")
    for k, v in prize.items():
        print(f"  {k:12s} {v:.4f}")

    results: dict = {"prize_raw_area": prize, "birth_area": _BIRTH_AREA,
                     "birth_coherence": _BIRTH_COH, "radii": _RADII, "classes": {}}
    for c in _RARE:
        name = _CLASSES[c]
        prev = None
        curve = []
        print(f"\n=== {name} (class {c}) winnability homotopy ===")
        print(f"  {'r':>3} {'area':>7} {'coher':>7} {'anneal_debt':>12} {'step_debt':>10}  birthable")
        for r in _RADII:
            eased = np.empty_like(lstars)
            areas = np.empty(n)
            cohs = np.empty(n)
            debts = np.empty(n)
            steps = np.empty(n) if prev is not None else None
            for i in range(n):
                e = _dilate_eased(lstars[i], c, r)
                eased[i] = e
                m = e == c
                areas[i] = m.mean()
                cohs[i] = _coherence(m)
                debts[i] = (e != lstars[i]).mean()
                if prev is not None:
                    steps[i] = (e != prev[i]).mean()
            area, coh = float(areas.mean()), float(cohs.mean())
            debt = float(debts.mean())
            step = float(steps.mean()) if steps is not None else 0.0
            birthable = area >= _BIRTH_AREA and coh >= _BIRTH_COH
            print(f"  {r:>3} {area:>7.4f} {coh:>7.3f} {debt:>12.4f} {step:>10.4f}  "
                  f"{'YES' if birthable else 'no'}")
            curve.append({"r": r, "area": area, "coherence": coh,
                          "anneal_debt": debt, "step_debt": step, "birthable": birthable})
            prev = eased
        # verdict per class
        birth_radii = [row for row in curve if row["birthable"]]
        cont = all(row["step_debt"] < 0.02 for row in curve[1:])   # no cliff (<2%/step)
        if birth_radii:
            r_star = birth_radii[0]
            verdict = ("GO" if cont else "GO-BUT-DISCONTINUOUS")
            note = (f"birthable at r={r_star['r']} (area {r_star['area']:.3f}, "
                    f"coh {r_star['coherence']:.2f}), anneal_debt {r_star['anneal_debt']:.3f}, "
                    f"path {'continuous' if cont else 'HAS A CLIFF'}")
        else:
            verdict = "NO-GO-REPRESENTATION-WALL"
            note = ("dilation never makes the class a coherent birthable island "
                    f"(max area {max(row['area'] for row in curve):.3f}, "
                    f"max coh {max(row['coherence'] for row in curve):.2f}) — "
                    "the wall is representational, not curricular")
        print(f"  VERDICT[{name}]: {verdict} — {note}")
        results["classes"][name] = {"curve": curve, "verdict": verdict, "note": note}

    # ── CURVE-PRIOR homotopy for LANE (the optimal easing: bridge dash gaps ALONG
    #    the curve, don't dilate isotropically) ─────────────────────────────────
    LANE = 1
    curve_L = [8, 6, 4, 3, 2, 1, 0]                       # easy(connected)→hard(dashes)
    prev = None
    curve = []
    print("\n=== lane (class 1) CURVE-PRIOR homotopy (vertical closing, w=1) ===")
    print(f"  {'L':>3} {'area':>7} {'coher':>7} {'anneal_debt':>12} {'step_debt':>10}  birthable")
    for L in curve_L:
        eased = np.empty_like(lstars)
        areas = np.empty(n); cohs = np.empty(n); debts = np.empty(n)
        steps = np.empty(n) if prev is not None else None
        for i in range(n):
            e = _curve_eased(lstars[i], LANE, L, w=1)
            eased[i] = e
            m = e == LANE
            areas[i] = m.mean(); cohs[i] = _coherence(m); debts[i] = (e != lstars[i]).mean()
            if prev is not None:
                steps[i] = (e != prev[i]).mean()
        area, coh = float(areas.mean()), float(cohs.mean())
        debt = float(debts.mean()); step = float(steps.mean()) if steps is not None else 0.0
        birthable = area >= _BIRTH_AREA and coh >= _BIRTH_COH
        print(f"  {L:>3} {area:>7.4f} {coh:>7.3f} {debt:>12.4f} {step:>10.4f}  "
              f"{'YES' if birthable else 'no'}")
        curve.append({"L": L, "area": area, "coherence": coh,
                      "anneal_debt": debt, "step_debt": step, "birthable": birthable})
        prev = eased
    birth = [row for row in curve if row["birthable"]]
    cont = all(row["step_debt"] < 0.02 for row in curve[1:])
    if birth:
        r0 = birth[0]
        verdict = "GO" if cont else "GO-BUT-DISCONTINUOUS"
        note = (f"birthable at L={r0['L']} (area {r0['area']:.3f}, coh {r0['coherence']:.2f}), "
                f"anneal_debt {r0['anneal_debt']:.3f}, path {'continuous' if cont else 'HAS A CLIFF'}")
    else:
        verdict = "NO-GO"
        note = (f"even curve-closing does not make lane a coherent birthable band "
                f"(max coh {max(row['coherence'] for row in curve):.2f})")
    print(f"  VERDICT[lane curve-prior]: {verdict} — {note}")
    print(f"  (vs isotropic lane verdict: {results['classes']['lane']['verdict']})")
    results["classes"]["lane_curve_prior"] = {"curve": curve, "verdict": verdict, "note": note}

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "result.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {_OUT / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
