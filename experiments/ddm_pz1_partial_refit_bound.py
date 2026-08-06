#!/usr/bin/env python
"""ddm_pz1 — can a pose RE-FIT rescue the ll1 window solve?  A matched bound.

THE QUESTION THIS ANSWERS
-------------------------
If the naive bolt-on regresses d_pose, ``ra1`` §5.3 / ``sg2`` convert the item to
"re-fit the pose sidecar against the solved raster" (~1.76-1.96 h wall, ~11.2
CPU-hours).  The charter requires that conversion be RANKED, not assumed.  The
decisive question is whether the perturbation is something the pose parameters
can absorb at all, or whether it raises the achievable NOISE FLOOR.

THE MATCHED DESIGN (m85: matched-base control BEFORE composing)
---------------------------------------------------------------
Re-fitting only the SOLVED raster and comparing it to the SHIPPED base is a
rigged comparison -- one side gets a fresh optimisation the other never got.  So
the SAME partial re-fit is run on BOTH rasters over the SAME grid:

    d_pose(base,   shipped params)   <- the live value
    d_pose(base,   argmin over grid) <- base at the same level of optimisation
    d_pose(solved, shipped params)   <- the naive bolt-on
    d_pose(solved, argmin over grid) <- solved at the same level of optimisation

The load-bearing comparison is the LAST TWO ROWS against each other.  If
``refit(solved) > refit(base)`` at matched optimisation, the floor genuinely rose
and more DOF cannot recover what was lost -- they can only chase a worse optimum.

SCOPE, STATED HONESTLY
----------------------
The grid is the DISCRETE part of the shipped v4d grammar that is exhaustively
searchable per pair: ``st_idx`` (the 11-entry ST_GRID) x ``sel`` (0/1).  The
continuous DOF (``p_best`` 6-vector, ``ab`` photometric pair) are NOT re-fitted.
So this is a LOWER bound on full-re-fit recovery, and a partial-re-fit failure is
only PARTIAL evidence against a full re-fit.  The matched-base leg is what makes
it informative anyway: it measures whether the base was already AT its grid
optimum, which calibrates how much these DOF were carrying in the first place.

axis: [macOS-CPU advisory] NON-PROMOTABLE.  score_claim=false.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _strided_pair_selection_scope(idx: np.ndarray, population: int) -> dict[str, object]:
    return {
        "schema": "subset_scope.v1",
        "n": int(len(idx)),
        "population": int(population),
        "selection_mode": "strided_linspace",
        "pair_indices": [int(v) for v in idx],
        "selection_rule": "unique(round(linspace(0, population - 1, requested_pairs)))",
        "axis_bias_caveat": (
            "strided advisory subset; no population claim or n600 conclusion follows without "
            "a governing subset/population bias check"
        ),
        "population_claim": False,
    }


def _install_paths(sub: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for p in (str(sub), str(root / "upstream"), str(root / "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _best_over_grid(dec, p3v2, posenet, i: int, f1: np.ndarray,
                    target: np.ndarray, n_st: int) -> tuple[float, int, int]:
    """Exhaustive argmin of d_pose over (st_idx, sel) using the SHIPPED composer."""
    st0, sel0 = int(dec.st_idx[i]), int(dec.sel[i])
    best = (float("inf"), st0, sel0)
    try:
        for st in range(n_st):
            for sel in (0, 1):
                dec.st_idx[i], dec.sel[i] = st, sel
                f0 = dec.f0(i, f1)
                d = p3v2.d_pose_u8(posenet, f0, f1, target)
                if d < best[0]:
                    best = (d, st, sel)
    finally:
        dec.st_idx[i], dec.sel[i] = st0, sel0
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True, type=Path)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sub = args.submission_dir.resolve()
    _install_paths(sub)

    import ddm_p3v2_optimal_form_pose_resolve as p3v2  # noqa: E402
    from inflate_runner import Decoder  # noqa: E402
    from tac.optimization import ddm_tr1_runtime as repo_tr1  # noqa: E402

    posenet, _ = p3v2.load_posenet()
    targets = p3v2.load_targets(600)
    dec = Decoder(sub / "archive")
    n_st = int(np.asarray(dec.st_vals).size)
    n = int(dec.n_pairs)
    idx = np.unique(np.linspace(0, n - 1, args.pairs).round().astype(int))
    print(f"[pz1] base={sub.name} pairs={len(idx)} grid={n_st} st x 2 sel", flush=True)

    rows, t0 = [], time.time()
    for k, i in enumerate(idx):
        i = int(i)
        tgt = targets[i]
        f1b = repo_tr1.render_frame1_camera_uint8(dec.packet, i, window_solve=False)
        f1s = repo_tr1.render_frame1_camera_uint8(dec.packet, i, window_solve=True)

        ship_b = p3v2.d_pose_u8(posenet, dec.f0(i, f1b), f1b, tgt)
        ship_s = p3v2.d_pose_u8(posenet, dec.f0(i, f1s), f1s, tgt)
        refit_b, stb, selb = _best_over_grid(dec, p3v2, posenet, i, f1b, tgt, n_st)
        refit_s, sts, sels = _best_over_grid(dec, p3v2, posenet, i, f1s, tgt, n_st)

        rows.append({
            "pair": i,
            "shipped_st": int(dec.st_idx[i]), "shipped_sel": int(dec.sel[i]),
            "d_pose_base_shipped": ship_b, "d_pose_base_refit": refit_b,
            "base_refit_st": stb, "base_refit_sel": selb,
            "d_pose_solved_shipped": ship_s, "d_pose_solved_refit": refit_s,
            "solved_refit_st": sts, "solved_refit_sel": sels,
        })
        print(f"[pz1] {k + 1:3d}/{len(idx)} pair {i:4d} | base {ship_b:.8f}"
              f"/refit {refit_b:.8f} | solved {ship_s:.8f}/refit {refit_s:.8f}"
              f" | {time.time() - t0:6.1f}s", flush=True)

    bs = np.array([r["d_pose_base_shipped"] for r in rows])
    br = np.array([r["d_pose_base_refit"] for r in rows])
    ss = np.array([r["d_pose_solved_shipped"] for r in rows])
    sr = np.array([r["d_pose_solved_refit"] for r in rows])

    penalty_naive = float(ss.mean() - bs.mean())
    penalty_matched = float(sr.mean() - br.mean())
    recovered = 1.0 - (penalty_matched / penalty_naive) if penalty_naive != 0 else float("nan")

    summary = {
        "base": sub.name,
        "n_pairs": len(idx),
        "pairs": [int(v) for v in idx],
        "pair_selection": _strided_pair_selection_scope(idx, n),
        "grid": {"n_st": n_st, "n_sel": 2},
        "mean_d_pose_base_shipped": float(bs.mean()),
        "mean_d_pose_base_refit": float(br.mean()),
        "mean_d_pose_solved_shipped": float(ss.mean()),
        "mean_d_pose_solved_refit": float(sr.mean()),
        "base_already_at_grid_optimum_frac": float(np.mean(bs <= br + 1e-15)),
        "penalty_naive_bolt_on": penalty_naive,
        "penalty_after_matched_partial_refit": penalty_matched,
        "fraction_of_penalty_recovered_by_partial_refit": float(recovered),
        "matched_refit_ratio": float(sr.mean() / br.mean()) if br.mean() > 0 else None,
        "rows": rows,
        "scope": ("partial re-fit over the DISCRETE shipped grammar (st_idx x sel) only; "
                  "p_best and ab NOT re-fitted => LOWER bound on full-re-fit recovery"),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print("\n[pz1] ===== MATCHED PARTIAL RE-FIT =====")
    print(f"[pz1] base   shipped {bs.mean():.8f} -> refit {br.mean():.8f}")
    print(f"[pz1] solved shipped {ss.mean():.8f} -> refit {sr.mean():.8f}")
    print(f"[pz1] base already at grid optimum on "
          f"{100 * summary['base_already_at_grid_optimum_frac']:.1f}% of pairs")
    print(f"[pz1] penalty naive bolt-on      : {penalty_naive:+.8f}")
    print(f"[pz1] penalty after matched refit: {penalty_matched:+.8f}")
    print(f"[pz1] fraction recovered by partial re-fit: "
          f"{100 * summary['fraction_of_penalty_recovered_by_partial_refit']:.1f}%")
    print(f"[pz1] matched refit ratio solved/base: {summary['matched_refit_ratio']}")
    print(f"[pz1] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
