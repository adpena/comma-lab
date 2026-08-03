"""ddm_pc2 — per-EDGE decomposition of the seg residual (region-adjacency graph view).

Reduces the ``ddm_ru1`` per-flip atlas to the EDGE view of the seg residual, and joins it
against the two references that bound our position from below: the oracle-R achievable floor
at the live render grid (``probe_PA`` / #210) and the exact per-cell solve (``ddm_sg1``).

WHY AN EDGE VIEW.  Charging flips by GT class splits ONE separatrix across TWO rows: the
per-class table's "Road 44%" and "Lane 30%" are largely the SAME pixels on the Road<->Lane
tie locus, counted from opposite sides.  ``probe_PA`` measured that d_seg factorizes over
pairwise tie-loci on the region-adjacency graph (Road = hub, zero interior flips); SPEC_v8 §1
makes the edge-centric decomposition binding.  This module is the measurement that view needs.

MEASURED OUTPUT (tb1 ep399 endpoint, n600):
  * Road participates in 87.8% of all 458,738 flips (as GT side or realized side).
  * Road<->Lane alone = 49.2% of flips = 22.1% of the total remaining gap to PR130.
  * 93.9% of flips lie ON the GT boundary; 0.058% are interior -- the residual is codim-1.

Axis: ``[macOS-CPU advisory]`` NON-PROMOTABLE.  ``score_claim=false``.  $0 -- reads cached
arrays only, fires NO scorer pass.  A positive control reproduces the source receipt's
taxonomy and flicker fractions to ``absdiff = 0.0`` before any number is read off the atlas;
it refuses (rc=2) if that control fails, so a corrupted or re-typed atlas cannot be reported
as a clean decomposition.

Usage::

    .venv/bin/python experiments/ddm_pc2_edge_decomp.py            # human-readable tables
    .venv/bin/python experiments/ddm_pc2_edge_decomp.py --json     # machine-readable

Memo: ``.omx/research/ddm_pc2_perclass_road_edges_20260802.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- custody (SSD tier; these are read-only inputs, never written by this module) ---------
RU1 = Path("/Volumes/VertigoDataTier/pact/ddm_ru1_20260729")
SG1 = Path("/Volumes/VertigoDataTier/pact/ddm_sg1_20260731")

CLS: tuple[str, ...] = ("Road", "Lane", "Undriv", "Movable", "MyCar")
"""CLAUDE.md canonical comma10k order, MEASURED.  NEVER re-derive by luma-sort."""

_SG1_NAME = {"Road": "Road", "Lane": "Lane", "Undriv": "Undrivable",
             "Movable": "Movable", "MyCar": "MyCar"}

# ``dist_bin`` taxonomy as emitted by ddm_ru1; verified against the receipt at runtime.
_DIST_BIN = {0: "on_gt_boundary", 1: "near_3px", 2: "interior"}

# Gap to the PR130 demonstrated floor -- tac.canonical_equations
# gap_decomposition_against_floor_20260802 fed PR130's measured 190,952 B.
GAP_TO_FLOOR_S = 0.7263025

# reports/levelset_oracle_R_floor_n600_20260701.json :: g1_realframe_R_384, n600.
# The live vehicle renders 384x512 (ddm_tr1_runtime.SEG_H/SEG_W), so @384 is the
# apples-to-apples row.  @192 is the OLDER witness default and must NOT be used here.
ORACLE_R_384_S = 0.0009099748399522569 * 100.0

# probe_PA_paintfloor_perclass_20260708 RESULT 1 (share of composite) + RESULT 2
# (flip-destination matrix), both @ render grid 384x512, n600.
_PA_CLASS_SHARE = {"Road": .437, "Lane": .163, "Undriv": .182,
                   "Movable": .104, "MyCar": .114}
_PA_DESTINATION = {
    "Road": {"Lane": .41, "Undriv": .25, "MyCar": .23, "Movable": .10},
    "Lane": {"Road": .99},
    "Undriv": {"Road": .64, "Movable": .36},
    "Movable": {"Undriv": .57, "Road": .43},
    "MyCar": {"Road": .99},
}


class Pc2ControlError(RuntimeError):
    """The apparatus-validity control failed; no decomposition may be reported."""


def _edge_key(a: str, b: str) -> str:
    return "<->".join(sorted((a, b)))


def oracle_edge_profile() -> dict[str, float]:
    """Per-EDGE oracle-R@384 flip mass in S-units, from probe_PA's RAG matrix.

    DERIVED, not measured: probe_PA published rounded percentages, so each edge carries
    roughly +/-3%.  Sufficient to rank edges and to support the extreme ratios; NOT a
    substitute for a direct per-edge oracle pass.
    """
    per_class = {c: _PA_CLASS_SHARE[c] * ORACLE_R_384_S for c in CLS}
    out: dict[str, float] = {}
    for src, dests in _PA_DESTINATION.items():
        for dst, frac in dests.items():
            out[_edge_key(src, dst)] = out.get(_edge_key(src, dst), 0.0) + per_class[src] * frac
    return out


def _positive_control(db: np.ndarray, fl: np.ndarray) -> dict[str, Any]:
    """Reproduce the ru1 receipt's own taxonomy + flicker fractions. Refuses on drift.

    This is the confound-self-protection leg: an empty, re-typed, or partially-written
    atlas would otherwise reduce silently to a clean-looking table (VACUITY == PASS).
    """
    rec = json.loads((RU1 / "atlas_analysis_receipt.json").read_text())
    checks: dict[str, Any] = {}
    for b, name in _DIST_BIN.items():
        got = float((db == b).mean())
        ref = float(rec["taxonomy"][name]["frac"])
        checks[name] = {"computed": got, "receipt": ref, "absdiff": abs(got - ref)}
    got_fl = float(fl.mean())
    ref_fl = float(rec["gt_flicker_at_flips"]["overall_frac"])
    checks["gt_flicker"] = {"computed": got_fl, "receipt": ref_fl,
                            "absdiff": abs(got_fl - ref_fl)}
    worst = max(c["absdiff"] for c in checks.values())
    checks["worst_absdiff"] = worst
    checks["passed"] = worst < 1e-12
    if not checks["passed"]:
        raise Pc2ControlError(
            f"ru1 atlas does not reproduce its own receipt (worst absdiff {worst:.3e}); "
            "refusing to report a decomposition off an unvalidated instrument")
    return checks


def decompose() -> dict[str, Any]:
    """The full reduction: 5x5 confusion, undirected edges, class/node shares, joins."""
    z = np.load(RU1 / "atlas_flat.npz", allow_pickle=True)
    gt = z["gt_class"].astype(np.int64)
    rz = z["realized_class"].astype(np.int64)
    db = z["dist_bin"].astype(np.int64)
    fl = z["gt_flicker"].astype(np.int64)
    md = z["m_def"].astype(np.float64)
    gm = z["gt_margin"].astype(np.float64)
    n = int(gt.size)

    out: dict[str, Any] = {
        "schema": "ddm_pc2_edge_decomp.v1",
        "evidence_axis": "[macOS-CPU advisory]",
        "score_claim": False, "promotion_eligible": False, "research_only": True,
        "endpoint": "tb1 ep399 (ru1 + sg1 are the SAME endpoint; 458,738 vs 458,621 flips)",
        "n_flips": n,
        "positive_control": _positive_control(db, fl),
    }

    conf = np.zeros((5, 5), dtype=np.int64)
    for i in range(5):
        for j in range(5):
            conf[i, j] = int(((gt == i) & (rz == j)).sum())
    out["directed_counts_gt_by_realized"] = conf.tolist()
    out["per_class_gt_charged_share"] = {CLS[i]: float(conf[i].sum() / n) for i in range(5)}
    # A flip involves exactly two distinct classes, so node shares sum to 2.0 -- no
    # double counting within a flip.
    out["per_class_node_participation_share"] = {
        CLS[i]: float(((gt == i) | (rz == i)).sum() / n) for i in range(5)}

    sg1 = json.loads((SG1 / "sg1_typing_receipt.json").read_text())
    dseg_s = float(sg1["renderer_endpoint"]["d_seg"]) * 100.0
    oracle = oracle_edge_profile()

    edges: list[dict[str, Any]] = []
    for i in range(5):
        for j in range(i + 1, 5):
            mask = ((gt == i) & (rz == j)) | ((gt == j) & (rz == i))
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            share = cnt / n
            live_s = share * dseg_s
            key = _edge_key(CLS[i], CLS[j])
            orc = oracle.get(key)
            lo, hi = int(conf[i, j]), int(conf[j, i])
            edges.append({
                "edge": key, "n": cnt, "share_of_all_flips": share,
                f"n_{CLS[i]}_to_{CLS[j]}": lo, f"n_{CLS[j]}_to_{CLS[i]}": hi,
                "asymmetry": (max(lo, hi) / min(lo, hi)) if min(lo, hi) else None,
                "live_S": live_s,
                "oracle_R384_S": orc,
                "ratio_vs_oracle": (live_s / orc) if orc else None,
                "headroom_S": (live_s - orc) if orc else None,
                "pct_of_total_gap": (100.0 * (live_s - orc) / GAP_TO_FLOOR_S) if orc else None,
                "on_gt_bnd_frac": float((db[mask] == 0).mean()),
                "near_3px_frac": float((db[mask] == 1).mean()),
                "interior_frac": float((db[mask] == 2).mean()),
                "flicker_frac": float(fl[mask].mean()),
                "m_def_med": float(np.median(md[mask])),
                "gt_margin_med": float(np.median(gm[mask])),
                "frac_m_def_below_0p25": float((md[mask] < 0.25).mean()),
            })
    edges.sort(key=lambda e: -e["n"])
    out["edges"] = edges

    road = [e for e in edges if "Road" in e["edge"]]
    out["road_incident"] = {
        "n": sum(e["n"] for e in road),
        "share_of_all_flips": sum(e["n"] for e in road) / n,
        "headroom_S": sum(e["headroom_S"] or 0.0 for e in road),
        "pct_of_total_gap": 100.0 * sum(e["headroom_S"] or 0.0 for e in road) / GAP_TO_FLOOR_S,
    }
    lost = int(conf[0].sum() - conf[0, 0])
    gained = int(conf[:, 0].sum() - conf[0, 0])
    out["road_area_bias"] = {"lost_px": lost, "gained_px": gained,
                             "net_px": gained - lost,
                             "note": "net > 0 => the vehicle OVER-paints Road"}

    # --- per-class join against both reference floors -------------------------------
    s_per_flip = dseg_s / int(sg1["renderer_endpoint"]["total_flip"])
    per_class = {}
    for c in CLS:
        row = sg1["per_class"][_SG1_NAME[c]]
        live = row["renderer_errors"] * s_per_flip
        solve = row["exact_solve_concede"] * s_per_flip
        orc = _PA_CLASS_SHARE[c] * ORACLE_R_384_S
        per_class[c] = {
            "live_S": live, "oracle_R384_S": orc, "exact_solve_S": solve,
            "ratio_vs_oracle": live / orc, "ratio_vs_exact_solve": live / solve,
            "gt_area_frac": row["gt_pixels"] / sum(
                v["gt_pixels"] for v in sg1["per_class"].values()),
            "within_class_err_rate": row["renderer_err_rate"],
            "px_per_frame": row["gt_pixels"] / 600.0,
        }
    out["per_class_join"] = per_class

    # --- the sub-cell minority-averaging regularity ---------------------------------
    # Live token lattice: 24x32 cells over a 384x512 render => 16x16 = 256 scored px/cell,
    # described by code_width=4 numbers.  ddm_tr1_runtime.py:274-288 + selector.sec.
    cell_px = (384 // 24) * (512 // 32)
    area = np.array([per_class[c]["gt_area_frac"] for c in CLS])
    err = np.array([per_class[c]["within_class_err_rate"] for c in CLS])
    slope, icpt = np.polyfit(np.log10(area), np.log10(err), 1)
    out["area_error_law"] = {
        "scored_px_per_token_cell": cell_px, "code_width": 4,
        "px_per_cell": {c: per_class[c]["px_per_frame"] / 768.0 for c in CLS},
        "loglog_slope": float(slope), "loglog_intercept_coeff": float(10 ** icpt),
        "pearson_r_log": float(np.corrcoef(np.log10(area), np.log10(err))[0, 1]),
        "n_points": 5,
        "verdict_scope": "INSTANCE (5 classes, one vehicle, one endpoint) -- a strong "
                         "regularity, NOT a registered law; area and thinness are "
                         "confounded across these five points and the fit cannot "
                         "separate them (both readings imply the same cure).",
    }
    return out


def _print_tables(d: dict[str, Any]) -> None:
    pc = d["per_class_join"]
    print(f"positive control: worst absdiff {d['positive_control']['worst_absdiff']:.3e} "
          f"(PASS={d['positive_control']['passed']})   n_flips={d['n_flips']:,}")
    print("\n--- per class, vs two reference floors (S-units) " + "-" * 46)
    print(f"{'class':9s} {'live':>8s} {'orcR@384':>9s} {'x':>6s} {'exactSlv':>9s} {'x':>7s} "
          f"{'area%':>7s} {'errRate':>8s} {'px/cell':>8s}")
    for c in CLS:
        r = pc[c]
        print(f"{c:9s} {r['live_S']:8.5f} {r['oracle_R384_S']:9.5f} "
              f"{r['ratio_vs_oracle']:6.2f} {r['exact_solve_S']:9.5f} "
              f"{r['ratio_vs_exact_solve']:7.2f} {r['gt_area_frac']*100:6.2f}% "
              f"{r['within_class_err_rate']*100:7.3f}% "
              f"{d['area_error_law']['px_per_cell'][c]:8.1f}")
    law = d["area_error_law"]
    print(f"  err_rate ~ {law['loglog_intercept_coeff']:.3e} * area^({law['loglog_slope']:.3f})"
          f"   Pearson r(log,log) = {law['pearson_r_log']:.4f}  (n={law['n_points']})")

    print("\n--- per EDGE (region-adjacency graph) " + "-" * 57)
    print(f"{'edge':22s} {'share':>7s} {'live':>8s} {'orcR@384':>9s} {'x':>6s} "
          f"{'%gap':>6s} {'onBnd':>6s} {'nr3px':>6s} {'flick':>6s} {'md<.25':>7s} {'asym':>6s}")
    for e in d["edges"]:
        if e["n"] < 1000:
            continue
        print(f"{e['edge']:22s} {e['share_of_all_flips']*100:6.2f}% {e['live_S']:8.5f} "
              f"{e['oracle_R384_S']:9.5f} {e['ratio_vs_oracle']:6.2f} "
              f"{e['pct_of_total_gap']:5.1f}% {e['on_gt_bnd_frac']*100:5.1f}% "
              f"{e['near_3px_frac']*100:5.1f}% {e['flicker_frac']*100:5.1f}% "
              f"{e['frac_m_def_below_0p25']*100:6.1f}% {e['asymmetry'] or 0:5.2f}x")
    ri, ab = d["road_incident"], d["road_area_bias"]
    print(f"\nRoad-incident edges: {ri['share_of_all_flips']*100:.1f}% of ALL flips · "
          f"headroom {ri['headroom_S']:.5f} S = {ri['pct_of_total_gap']:.1f}% of the total gap")
    print(f"Road NODE participation "
          f"{d['per_class_node_participation_share']['Road']*100:.1f}%  ·  "
          f"net area bias {ab['net_px']:+,} px ({ab['note']})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)
    try:
        d = decompose()
    except Pc2ControlError as exc:
        print(f"[REFUSED] {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[BLOCKED] custody input missing: {exc}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(d, indent=1))
    else:
        _print_tables(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
