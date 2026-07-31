# SPDX-License-Identifier: MIT
"""ddm_gd1 (#817) — audit + fix for the A1 realized-gate estimator (the 36-pair instrument).

WHAT THIS IS
------------
``experiments/train_tr1_partition_renderer_mlx.py::realized_gate`` reports
``realized_gate_dseg_mean = np.mean(dsegs)`` over a 36-pair gate set built by
``resolve_gate_ids(600)`` as ``BLOCK(447,448,449,450) + 32 SRS-without-replacement
from the other 596 (numpy default_rng(0))``.

That is an UNWEIGHTED mean over a NON-PROBABILITY sample: the 4 deterministic block
pairs carry weight 4/36 = 11.11% in the estimator but only 4/600 = 0.667% in the n600
quantity the estimator stands in for — a **16.67x over-weight**. The estimator is
therefore biased for the n600 mean BY DESIGN, and (measured, ddm_gc14) its bias MOVED
across the burn (-1.10e-6 -> +1.52e-5), overstating the window_02 descent by 7.2%.

EXACT DECOMPOSITION (DERIVED, algebra below, unit-tested in
``src/tac/tests/test_gd1_gate_estimator.py``)
With B = block (n_B=4), S = the SRS draw (n_S=32), O = the 596 off-block pairs,
N = 600, and X any per-pair quantity:

    X_unweighted = (n_B/36) Xbar_B + (n_S/36) Xbar_S            <- what ships today
    X_ht         = (n_B/N)  Xbar_B + (|O|/N) Xbar_S             <- Horvitz-Thompson
    X_pop        = (n_B/N)  Xbar_B + (|O|/N) Xbar_O             <- the n600 truth

    X_unweighted - X_pop
        = (n_B/36 - n_B/N) * (Xbar_B - Xbar_S)        [BLOCK OVER-WEIGHT: deterministic,
                                                       removable EXACTLY by re-weighting]
        + (|O|/N) * (Xbar_S - Xbar_O)                 [SRS SAMPLING ERROR: mean-zero over
                                                       the seed, FIXED for seed 0, removable
                                                       only by anchoring to periodic n600]

The first coefficient is 4/36 - 4/600 = 0.1044444...; the second is 596/600 = 0.9933333.
The HT estimator is unbiased for the n600 mean in expectation over the SRS draw and
removes the first (deterministic, drift-carrying) term exactly, at ZERO extra scorer
cost -- it is pure arithmetic on the SAME 36 renders.

WHAT IT MEASURES HERE
---------------------
Per-pair d_seg is NOT logged by the trainer (only mean + per-pair max; and
``full_confirm`` discards its own ``all_dsegs`` vector after reducing to mean/max), so
the realized bias split cannot be recomputed from existing telemetry. This tool
therefore measures the decomposition on SCORER-FREE per-pair sensitivity proxies read
from the frozen GT cache (``margins`` = the SegNet margin field, which CLAUDE.md records
as the Fisher surrogate at Pearson 0.978; plus boundary-pixel density and per-class
composition from ``lstars``). Those proxies drive per-pair d_seg (100% of flips live in
the small-margin annulus, QA74), so the SIGN and RELATIVE MAGNITUDE of the block-vs-SRS
differential transfer; the absolute d_seg bias does not, and is never claimed.

axis: [macOS-CPU advisory] -- scorer-FREE (no SegNet/PoseNet forward, no MLX, no Metal,
no paid dispatch). score_claim=false. research_only.

USAGE
-----
    .venv/bin/python tools/gd1_gate_estimator_audit.py \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
        --out .omx/research/ddm_gd1_gate_estimator_audit_20260731.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.ddm_gd1_gate_estimator import (
    GateDesign,
    bias_decomposition,
    neyman_stratified_gate,
    srs_standard_error,
)

DEFAULT_GT_CACHE = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def resolve_live_gate_design(n_pairs: int = 600) -> tuple[GateDesign, str]:
    """Read the LIVE gate geometry from the trainer itself (never a hand-copy).

    The trainer's module-level imports are stdlib + numpy only (mlx is imported lazily
    inside functions), so importing it is cheap and cannot start a scorer.
    """
    from experiments.train_tr1_partition_renderer_mlx import (
        GATE_BLOCK_PAIRS,
        resolve_gate_ids,
    )

    ids = resolve_gate_ids(n_pairs)
    block = tuple(int(x) for x in GATE_BLOCK_PAIRS)
    srs = tuple(int(x) for x in ids if int(x) not in set(block))
    return GateDesign(n_population=n_pairs, block_ids=block, srs_ids=srs), "imported_from_trainer"


def per_pair_proxies(gt_cache: Path) -> dict[str, np.ndarray]:
    """Scorer-free per-pair sensitivity proxies from the frozen GT cache.

    MEMORY (honest): an ``NpzFile`` member is fully materialised on first access, so the
    two source arrays dominate peak RSS -- ``margins`` (600,384,512) float32 = 472 MB and
    ``lstars`` (600,384,512) int64 = 944 MB, ~1.4 GB resident. The per-pair chunking
    bounds only the TRANSIENT intermediates (the float32 copy and the three bool edge
    masks, ~O(chunk) each) -- it does NOT bound the source load. Measured peak RSS for a
    full 600-pair run is reported in the receipt as ``peak_rss_mib``. A live burn may own
    most of RAM, so this stays well under any admission floor but is not O(chunk).
    """
    z = np.load(gt_cache)
    n = int(z["lstars"].shape[0])
    taus = (0.25, 0.5, 1.0, 2.0)
    out: dict[str, list[float]] = {f"fp_mass_tau{t}": [] for t in taus}
    out["mean_margin"] = []
    out["boundary_frac"] = []
    out["lane_frac"] = []

    margins = z["margins"]
    lstars = z["lstars"]
    chunk = 25
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        m = np.asarray(margins[c0:c1], dtype=np.float32)
        ls = np.asarray(lstars[c0:c1]).astype(np.uint8, copy=False)
        for t in taus:
            out[f"fp_mass_tau{t}"].extend((m < t).mean(axis=(1, 2)).tolist())
        out["mean_margin"].extend(m.mean(axis=(1, 2)).tolist())
        # 4-neighbour inter-class edge density = the codim-1 separatrix length per pair.
        horiz = ls[:, :, 1:] != ls[:, :, :-1]
        vert = ls[:, 1:, :] != ls[:, :-1, :]
        edge = np.zeros(ls.shape, dtype=bool)
        edge[:, :, 1:] |= horiz
        edge[:, :, :-1] |= horiz
        edge[:, 1:, :] |= vert
        edge[:, :-1, :] |= vert
        out["boundary_frac"].extend(edge.mean(axis=(1, 2)).tolist())
        out["lane_frac"].extend((ls == 1).mean(axis=(1, 2)).tolist())
        del m, ls, horiz, vert, edge
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-cache", default=DEFAULT_GT_CACHE)
    ap.add_argument("--out", default=".omx/research/ddm_gd1_gate_estimator_audit_20260731.json")
    ap.add_argument("--strata", type=int, default=4)
    args = ap.parse_args(argv)

    import resource

    design, design_source = resolve_live_gate_design(600)
    proxies = per_pair_proxies(Path(args.gt_cache))
    _ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mib = _ru / (1024 * 1024) if sys.platform == "darwin" else _ru / 1024

    rows: dict[str, Any] = {}
    for name, x in proxies.items():
        dec = bias_decomposition(design, x)
        se = srs_standard_error(design, x)
        rows[name] = {
            "pop_mean": dec.pop_mean,
            "gate_unweighted": dec.unweighted,
            "gate_ht": dec.horvitz_thompson,
            "total_design_error": dec.total_error,
            "total_design_error_rel_pct": 100.0 * dec.total_error / dec.pop_mean,
            "block_overweight_term": dec.block_overweight_term,
            "srs_sampling_term": dec.srs_sampling_term,
            "removable_fraction_of_error": dec.removable_fraction,
            "block_mean": dec.block_mean,
            "srs_mean": dec.srs_mean,
            "offblock_mean": dec.offblock_mean,
            "srs_se_of_mean": se,
            "srs_se_rel_pct": 100.0 * se / dec.pop_mean,
        }

    strat = neyman_stratified_gate(
        design, proxies["fp_mass_tau1.0"], n_strata=int(args.strata)
    )

    receipt = {
        "schema": "ddm_gd1_gate_estimator_audit.v1",
        "arm": "ddm_gd1_undecided_defaults (#817)",
        "axis": "[macOS-CPU advisory] scorer-free (no SegNet/PoseNet forward)",
        "score_claim": False,
        "promotion_eligible": False,
        "research_only": True,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "gt_cache": str(args.gt_cache),
        "gate_design_source": design_source,
        "peak_rss_mib": round(peak_rss_mib, 1),
        "gate": {
            "n_population": design.n_population,
            "block_ids": list(design.block_ids),
            "srs_ids": list(design.srs_ids),
            "n_gate": design.n_gate,
            "block_weight_in_estimator": design.n_block / design.n_gate,
            "block_weight_in_population": design.n_block / design.n_population,
            "block_overweight_factor": (design.n_block / design.n_gate)
            / (design.n_block / design.n_population),
            "block_overweight_coefficient": design.block_overweight_coefficient,
        },
        "proxy_rows": rows,
        "neyman_stratified_redesign": strat,
        "honesty": {
            "measured": "per-pair scorer-free sensitivity proxies from the frozen GT cache",
            "derived": "the exact estimator decomposition (algebra, unit-tested)",
            "not_measured": (
                "the realized per-pair d_seg split -- the trainer logs only mean+max for the "
                "gate and discards full_confirm's own per-pair vector, so the realized bias "
                "cannot be recomputed from existing telemetry"
            ),
            "verdict_scope": "INSTANCE (this gate design, this GT cache, this vehicle)",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("gate", "neyman_stratified_redesign")}, indent=1))
    for name, r in rows.items():
        print(
            f"{name:20s} pop={r['pop_mean']:.6g} unw={r['gate_unweighted']:.6g} "
            f"ht={r['gate_ht']:.6g} err={r['total_design_error']:+.4g} "
            f"({r['total_design_error_rel_pct']:+.2f}%) removable={r['removable_fraction_of_error']:.1%} "
            f"srs_se={r['srs_se_of_mean']:.4g} ({r['srs_se_rel_pct']:.2f}%)"
        )
    print(f"\nreceipt -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
