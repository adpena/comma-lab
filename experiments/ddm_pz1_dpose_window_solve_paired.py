#!/usr/bin/env python
"""ddm_pz1 — PAIRED d_pose under the ddm_ll1 window solve, frozen-PoseNet authority.

WHY THIS AND NOT A BRACKET
--------------------------
``ddm_sg2`` priced the ll1 rung with an UNMEASURED pose risk against a DERIVED
+18.8-32.0% relative-d_pose tolerance, noting the two available anchors differ by
~7,300x and refusing to interpolate.  That refusal was right.  This measures the
quantity directly instead, on the live base, through the canonical frozen-PoseNet
path (``ddm_p3v2_optimal_form_pose_resolve.d_pose_u8``) that the pj2 solver itself
uses -- so the number is on the same instrument that produced the shipped solve.

DESIGN
------
PAIRED: base and solved d_pose are computed on the SAME pairs from the SAME
renders, differing only by ``window_solve``.  The paired ratio is far more robust
than either mean, which matters because per-pair d_pose is right-skewed
(``m88``: top 1% of pairs carry 70.33% of pj2's win).

THE m88 GUARD IS EXECUTED, NOT CITED: the subset's mean base d_pose is reported
against the known n600 population value 0.00255143 (= pj2/cx1 pose 0.1597320
squared over 10).  If that ratio is far from 1 the subset is a DIFFERENT
POPULATION and the extrapolation is refused in the output itself.

axis: [macOS-CPU advisory] NON-PROMOTABLE.  score_claim=false.  This is a
frozen-PoseNet forward on our own frames, NOT upstream/evaluate.py; it decides
whether to SPEND the n600 slot, it does not substitute for it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

#: pj2 / cx1 n600 population value: pose contribution 0.1597320 = sqrt(10*d_pose)
POPULATION_D_POSE: float = 0.00255143
#: the n600 rate/seg context needed to turn Delta d_pose into Delta S
CX1_SEG, CX1_POSE_CONTRIB, CX1_RATE = 0.4311790, 0.1597320, 0.2355862


def _strided_pair_selection_scope(
    idx: np.ndarray,
    population: int,
    *,
    subset_over_population: float,
) -> dict[str, object]:
    return {
        "schema": "subset_scope.v1",
        "n": int(len(idx)),
        "population": int(population),
        "selection_mode": "strided_linspace",
        "pair_indices": [int(v) for v in idx],
        "selection_rule": "unique(round(linspace(0, population - 1, requested_pairs)))",
        "axis_bias_caveat": (
            "strided advisory subset; m88 subset/population guard must be read before any "
            "population or n600 interpretation"
        ),
        "governing_ratio": {
            "name": "base_d_pose",
            "subset_over_population": float(subset_over_population),
        },
        "population_claim": False,
    }


def _install_paths(sub: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for p in (str(sub), str(root / "upstream"), str(root / "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True, type=Path)
    ap.add_argument("--pairs", type=int, default=32)
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
    n = int(dec.n_pairs)
    idx = np.unique(np.linspace(0, n - 1, args.pairs).round().astype(int))
    print(f"[pz1] base={sub.name} strided pairs={len(idx)} of {n}", flush=True)

    rows = []
    t0 = time.time()
    for k, i in enumerate(idx):
        f1b = repo_tr1.render_frame1_camera_uint8(dec.packet, int(i), window_solve=False)
        f1s = repo_tr1.render_frame1_camera_uint8(dec.packet, int(i), window_solve=True)
        f0b = dec.f0(int(i), f1b)
        f0s = dec.f0(int(i), f1s)
        db = p3v2.d_pose_u8(posenet, f0b, f1b, targets[int(i)])
        ds = p3v2.d_pose_u8(posenet, f0s, f1s, targets[int(i)])
        rows.append({"pair": int(i), "d_pose_base": db, "d_pose_solved": ds,
                     "delta": ds - db, "ratio": ds / db if db > 0 else float("inf")})
        print(f"[pz1] {k + 1:3d}/{len(idx)} pair {int(i):4d} | base {db:.8f} "
              f"-> solved {ds:.8f} | x{ds / db if db > 0 else float('nan'):7.3f} "
              f"| {time.time() - t0:6.1f}s", flush=True)

    base = np.array([r["d_pose_base"] for r in rows])
    sol = np.array([r["d_pose_solved"] for r in rows])
    ratios = sol / np.maximum(base, 1e-30)

    # ---- m88 guard, EXECUTED -------------------------------------------
    subset_mean = float(base.mean())
    m88_ratio = subset_mean / POPULATION_D_POSE
    m88_representative = bool(0.5 <= m88_ratio <= 2.0)

    # ---- Delta S: the VERDICT uses the scorer convention ----------------
    # upstream/evaluate.py averages d_pose ITSELF across pairs and only then
    # applies sqrt(10*.), so the axis verdict is the ratio of MEANS
    # (sol.mean()/base.mean()).  The median of per-pair ratios is a
    # distribution DIAGNOSTIC only and can disagree with the verdict IN SIGN
    # under skew (rt1 measured x1.809 median-of-ratios vs x0.431 scorer
    # convention on the same 7 pairs; law:
    # pose_aggregation_is_mean_of_dpose_never_mean_of_ratios_20260816).
    mean_ratio = float(sol.mean() / base.mean())      # scorer convention: THE VERDICT
    median_ratio = float(np.median(ratios))           # diagnostic only, never the verdict
    ds_pose_mean = CX1_POSE_CONTRIB * (math.sqrt(mean_ratio) - 1.0)
    ds_pose_median = CX1_POSE_CONTRIB * (math.sqrt(median_ratio) - 1.0)

    # tolerance from sg2: the seg win the rung must defend
    tol = {}
    for label, seg_win in (("ll1_measured", 0.0144), ("rms_scaled", 0.0238)):
        tol[label] = {
            "seg_win_S": seg_win,
            "max_tolerable_ratio": float(
                ((CX1_POSE_CONTRIB + seg_win) / CX1_POSE_CONTRIB) ** 2),
        }

    summary = {
        "base": sub.name,
        "pairs_measured": [int(v) for v in idx],
        "n_pairs_measured": len(idx),
        "pair_selection": _strided_pair_selection_scope(
            idx, n, subset_over_population=m88_ratio
        ),
        "m88_guard": {
            "subset_mean_base_d_pose": subset_mean,
            "population_d_pose": POPULATION_D_POSE,
            "subset_over_population": m88_ratio,
            "subset_is_representative": m88_representative,
            "note": ("ratio far from 1 means the strided subset is a DIFFERENT "
                     "population and no n600 extrapolation may be made from it"),
        },
        "d_pose_base_mean": subset_mean,
        "d_pose_solved_mean": float(sol.mean()),
        "mean_ratio": mean_ratio,
        "median_ratio": median_ratio,
        "min_ratio": float(ratios.min()),
        "max_ratio": float(ratios.max()),
        "frac_pairs_worse": float(np.mean(sol > base)),
        "delta_S_pose_from_mean_ratio": ds_pose_mean,
        "delta_S_pose_from_median_ratio": ds_pose_median,
        "pose_aggregation_verdict_basis": "ratio_of_means_scorer_convention",
        "delta_S_pose_median_is_diagnostic_only": True,
        "tolerance": tol,
        "rows": rows,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print("\n[pz1] ===== d_pose PAIRED SUMMARY =====")
    print(f"[pz1] m88 guard: subset mean base d_pose {subset_mean:.8f} vs "
          f"population {POPULATION_D_POSE:.8f} = {m88_ratio:.3f}x "
          f"=> representative: {m88_representative}")
    print(f"[pz1] d_pose  base {subset_mean:.8f} -> solved {sol.mean():.8f}")
    print(f"[pz1] ratio   mean {mean_ratio:.4f}  median {median_ratio:.4f}  "
          f"min {ratios.min():.4f}  max {ratios.max():.4f}")
    print(f"[pz1] pairs made worse: {100 * summary['frac_pairs_worse']:.1f}%")
    print(f"[pz1] Delta S(pose) from mean ratio   : {ds_pose_mean:+.5f}  (VERDICT: scorer convention)")
    print(f"[pz1] Delta S(pose) from median ratio : {ds_pose_median:+.5f}  (diagnostic only, never the verdict)")
    for label, t in tol.items():
        print(f"[pz1] tolerance {label}: seg win {t['seg_win_S']:.4f} "
              f"survives only if ratio <= {t['max_tolerable_ratio']:.4f}")
    print(f"[pz1] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
