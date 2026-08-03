#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_ms8 -- the s_t menu as an RD-optimal codebook, MEASURED not asserted.

WHY THIS EXISTS
---------------
``ST_GRID`` (``experiments/ddm_pfs1_ep_warp_pose_solve.py:61``, vendored into the
receiver ``pfs1_warp_receiver.py``) is an 11-entry scalar codebook for the
per-pair ground-plane translation scale ``s_t``.  Its MEASURED occupancy on the
shipped n600 solution is::

    idx      0     1     2     3     4     5     6     7     8     9    10
    value  .000  .005  .010  .020  .030  .044  .060  .080  .120  .160  .240
    count     0     0     0     0     0     0    22   364   156    58     0

Six of eleven codewords are never used and 60.67% of the mass sits on ONE
INTERIOR entry.  ``ddm_pw1``'s discriminator ("is the bound binding?") correctly
answers NO here -- the mass is interior, so this is not a clipping menu.  But
that discriminator never asks the second question: **is the codebook
well-designed for its own measured distribution?**  A codebook with 6 dead
codewords and 4 live ones covering the entire support is a candidate for
resolution starvation at the mode, which is an RD defect and not a clipping one.

Separately (the "start is the lever" axis): ``s_t`` is chosen ONCE, by the D1
grid search, against the D1 objective, and then FROZEN.  MEASURED here:
``d2_ep_solve``'s ``s_t`` equals ``ST_GRID[d1 s_t_idx]`` on 600/600 pairs, and
``ddm_v4d_build_composed_archive._st_coded_from_base`` copies the coded index
stream verbatim out of the base archive.  So the index was never revisited after
the pose, the photometric (a,b), the two-plane selector and the rolling-shutter
beta were solved.  Re-selecting the index against the FINAL composed objective
is a realizable receiver-legal move that costs ZERO alphabet change.

WHAT THIS TOOL MEASURES
-----------------------
``--mode curves`` evaluates, for every pair, the realized ``d_pose`` of the
SHIPPED reconstruction at each candidate ``s_t`` in ``S_EVAL`` -- everything else
(pose, a, b, selector, beta) held at the shipped value, through the same
frozen-CPU-PoseNet path the receiver runs.  One f1 render per pair, reused for
every candidate.  Emits per-pair curves.

``--mode design`` consumes the curves and races codebooks:

  CTRL   the shipped index (positive control: must reproduce ``d_final``)
  RESEL  argmin over the INCUMBENT 11-entry grid -- zero table bytes, because
         the alphabet is unchanged and the receiver already hardcodes it
  DESIGN k-medoid + local-swap optimal K-subset of ``S_EVAL`` for each K, which
         is the entropy-unconstrained RD-optimal codebook restricted to measured
         support points
  FULL   argmin over all of ``S_EVAL`` (the instrument's own upper bound)

Every arm is priced through the REAL coder (``ddm_r7_token_coder``) plus the
manifest cost of shipping a fitted table, and reported in composed-S units.

AXIS
----
``[macOS-CPU frozen-PoseNet advisory]``, ``score_claim=false``,
``promotion_eligible=false``.  No training, no paid dispatch, no pointer
mutation.  The composed-S column is a PREDICTION whose fidelity anchor on this
exact vehicle is the QA78 v4d gate residual (1.8e-6) and the pw1 gate residual
(2.5e-6); it is still a prediction and is labelled one.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SCHEMA = "ddm_ms8_st_codebook_race.v1"

N_PAIRS = 600
ARCHIVE_DENOM = 37_545_489.0

# The incumbent codebook, vendored verbatim so this tool refuses if the live
# table ever moves under it (checked in ``_assert_incumbent``).
INCUMBENT_ST_GRID: tuple[float, ...] = (
    0.0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24,
)

# The evaluation support.  DERIVED, not picked for convenience:
#   * every incumbent codeword (so RESEL is exactly representable and the
#     positive control lands on a measured column);
#   * midpoints of every incumbent cell that carries mass, plus quarter points
#     inside the dominant cell, so a refinement win is representable;
#   * 0.0 and 0.30 -- BOTH ends past the incumbent's occupied region, so this
#     instrument can detect clipping in either direction instead of assuming
#     the incumbent's support was right (the pw1 lesson, applied to myself).
S_EVAL: tuple[float, ...] = (
    0.0, 0.005, 0.01, 0.02, 0.03, 0.044,
    0.05, 0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.10, 0.11, 0.12,
    0.14, 0.16, 0.20, 0.24, 0.30,
)


def _utc() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _assert_incumbent() -> None:
    """Refuse if the live ST_GRID has drifted from the vendored copy."""
    sys.path.insert(0, str(REPO / "experiments"))
    import ddm_pfs1_ep_warp_pose_solve as pfs1
    live = tuple(float(x) for x in pfs1.ST_GRID)
    if live != INCUMBENT_ST_GRID:
        raise SystemExit(
            f"live ST_GRID {live} != vendored incumbent {INCUMBENT_ST_GRID}; "
            "this tool's occupancy and RESEL arms would be measuring a "
            "different codebook than the one that ships")


def contribution(d_pose_mean: float) -> float:
    return float(np.sqrt(10.0 * float(d_pose_mean)))


# --------------------------------------------------------------------------- #
# mode: curves
# --------------------------------------------------------------------------- #
def run_curves(args: argparse.Namespace) -> None:
    for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_tv] = "1"
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    import torch

    torch.set_num_threads(1)
    import ddm_v4c_resolve as v4c

    _assert_incumbent()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shipped = {int(r["pair"]): r for r in
               (json.loads(ln) for ln in
                args.final_jsonl.read_text().splitlines() if ln.strip())}
    if len(shipped) != N_PAIRS:
        raise SystemExit(f"expected {N_PAIRS} shipped rows, got {len(shipped)}")

    oracle = v4c.build_oracle(args.base, s_r=1.0)
    comp = v4c.StaticComposer(oracle)

    def score(pose, s_t, sel, f1_u8, f1_f, tp, a, b, g):
        """Realized d_pose -- mirrors ``inflate_runner_v4d.Decoder.f0`` and
        ``ddm_v4d_resolve._beta_select`` exactly."""
        beta = g * (1.0 if pose[5] >= 0.0 else -1.0)
        wg_t, wf_t = comp.warps(f1_f, pose, s_t, 1.0 - beta / 2.0)
        wg_b, wf_b = comp.warps(f1_f, pose, s_t, 1.0 + beta / 2.0)
        f0_t = np.where(comp.far[..., None], wf_t, wg_t) if sel else wg_t
        f0_b = np.where(comp.far[..., None], wf_b, wg_b) if sel else wg_b
        f0 = (1.0 - comp.alpha_row) * f0_t + comp.alpha_row * f0_b
        if a != 1.0 or b != 0.0:
            f0 = a * f0 + b
        p6 = comp.o.p3v2.pose6_u8(comp.o.posenet, comp.recv._to_uint8(f0), f1_u8)
        return float(np.mean((p6 - tp) ** 2))

    seq = [p for p in range(min(args.pairs, N_PAIRS))
           if p % args.nshards == args.shard]
    jl = args.out_dir / f"ms8_curves_shard{args.shard}.jsonl"
    cache = {int(json.loads(ln)["pair"]) for ln in
             (jl.read_text().splitlines() if jl.exists() else []) if ln.strip()}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[ms8 curves] shard {args.shard}/{args.nshards} pairs={len(seq)} "
          f"cached={len(cache)} |S_EVAL|={len(S_EVAL)}", flush=True)

    done = 0
    for pidx in seq:
        if pidx in cache:
            continue
        if (time.time() - t0) > args.max_minutes * 60.0:
            print(f"[ms8 curves] wall cap at pair {pidx}; rerun to resume",
                  flush=True)
            break
        sh = shipped[pidx]
        pose = np.asarray(sh["p"], np.float64)
        a, b = float(sh["a"]), float(sh["b"])
        sel = int(sh["selector"])
        g = float(sh["beta_mag"])
        s_ship = float(v4c._d2_row(pidx)["s_t"])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        curve = [score(pose, float(s), sel, f1_u8, f1_f, tp, a, b, g)
                 for s in S_EVAL]
        # POSITIVE CONTROL: the shipped s_t is itself a member of S_EVAL, so the
        # canary is read straight off the curve -- no separate code path that
        # could agree for the wrong reason.
        j_ship = int(np.argmin(np.abs(np.asarray(S_EVAL) - s_ship)))
        if abs(S_EVAL[j_ship] - s_ship) > 1e-12:
            raise SystemExit(
                f"pair {pidx}: shipped s_t {s_ship} is not a member of S_EVAL")
        rec = {
            "pair": int(pidx),
            "s_shipped": s_ship,
            "j_shipped": j_ship,
            "d_shipped_reported": float(sh["d_final"]),
            "d_ctrl": float(curve[j_ship]),
            "canary_abs_err": abs(float(curve[j_ship]) - float(sh["d_final"])),
            "curve": [float(x) for x in curve],
            "selector": sel, "beta_mag": g,
        }
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        done += 1
        if done % 10 == 0 or done <= 3:
            el = time.time() - t0
            print(f"[ms8 s{args.shard} {done}/{len(seq)}] pair {pidx} "
                  f"ctrl {rec['d_ctrl']:.6f} canary {rec['canary_abs_err']:.2e} "
                  f"min {min(curve):.6f} ({el:.0f}s, {el/max(done,1):.1f}s/pair)",
                  flush=True)
    fj.close()
    print(f"[ms8 curves] shard {args.shard} done={done} "
          f"total={len(cache)+done} {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- #
# codebook design
# --------------------------------------------------------------------------- #
# max_k C(21,k) = C(21,10) = 352,716, so this cap makes EVERY codebook size on
# the 21-point support a certified GLOBAL optimum rather than a local one.
# Measured cost of the full exhaustive ladder: see the receipt's wall time.
_EXHAUSTIVE_CAP = 400_000


def subset_search_mode(n_support: int, k: int) -> str:
    """Which search ``kmedoid_subset`` will use -- so an arm can be LABELLED.

    ``exhaustive`` is a certified global optimum over the measured support.
    ``greedy_swap_local`` is a local optimum (greedy start + single-swap
    descent from several starts) and must never be called "the RD-optimal
    codebook" without that qualifier.
    """
    if k >= n_support:
        return "exhaustive"
    n_sub = 1
    for t in range(k):
        n_sub = n_sub * (n_support - t) // (t + 1)
        if n_sub > _EXHAUSTIVE_CAP:
            return "greedy_swap_local"
    return "exhaustive"


def kmedoid_subset(curves: np.ndarray, k: int, *, restrict=None,
                   seeds: int = 3) -> tuple[int, ...]:
    """Best K-subset of the support minimising ``sum_i min_j curves[i, j]``.

    This is the entropy-unconstrained RD-optimal scalar codebook restricted to
    the measured support (a k-medoid / facility-location problem).  Greedy
    forward selection, then exhaustive single-swap local search to a local
    optimum, from several starts.  For small K the greedy start is provably
    within (1-1/e) of optimal on the equivalent coverage form; the swap phase
    then removes the remaining slack.  Exhaustive is used whenever the number of
    subsets is small enough to be certified OPTIMAL rather than local.
    """
    n, m = curves.shape
    cand = list(range(m)) if restrict is None else sorted(restrict)
    if k >= len(cand):
        return tuple(cand)

    def cost(sub) -> float:
        return float(curves[:, list(sub)].min(axis=1).sum())

    if subset_search_mode(len(cand), k) == "exhaustive":
        best = min(itertools.combinations(cand, k), key=cost)
        return tuple(sorted(best))

    best_sub, best_cost = None, float("inf")
    rng = np.random.default_rng(20260802)
    for s in range(seeds):
        if s == 0:
            sub: list[int] = []
            for _ in range(k):
                sub.append(min((j for j in cand if j not in sub),
                               key=lambda j: cost([*sub, j])))
        else:
            sub = list(rng.choice(cand, size=k, replace=False))
        cur = cost(sub)
        improved = True
        while improved:
            improved = False
            for pos in range(k):
                for j in cand:
                    if j in sub:
                        continue
                    trial = list(sub)
                    trial[pos] = j
                    c = cost(trial)
                    if c < cur - 1e-15:
                        sub, cur, improved = trial, c, True
        if cur < best_cost:
            best_sub, best_cost = tuple(sorted(sub)), cur
    assert best_sub is not None
    return best_sub


# The SHIPPED index coder (``ddm_r7_token_coder._codes``) admits only
# ``2 <= levels <= 16``, so a codebook wider than 16 entries is NOT
# representable on the live vehicle at any price.  That is a hard boundary of
# the design space, not a tuning choice, and it is reported rather than
# silently worked around.
R7_MAX_LEVELS = 16


def st_stream_bytes(idx: np.ndarray, levels: int) -> int | None:
    """Bytes the REAL shipped coder spends on this index stream.

    Returns ``None`` when the alphabet exceeds what the shipped coder accepts:
    such an arm is a DISTORTION-ONLY reference, never a priced candidate.
    """
    if not 2 <= int(levels) <= R7_MAX_LEVELS:
        return None
    sys.path.insert(0, str(REPO / "experiments"))
    from ddm_r7_token_coder import encode_token_codes
    codes = np.ascontiguousarray(idx, np.uint8).reshape(len(idx), 1, 1, 1)
    return len(encode_token_codes(codes, levels=int(levels), codec="auto"))


def table_manifest_bytes(table: tuple[float, ...] | None) -> int:
    """Bytes a fitted table costs in the manifest.

    A fitted table is VIDEO-DERIVED, so under contest rule 118 it is COUNTED --
    it cannot ride free in the receiver the way the generic incumbent ladder
    does.  Measured the same way the builder writes it: the compact JSON
    encoding of the extra key, then DEFLATE (``manifest.json`` is a DEFLATE
    member, ``ddm_v4d_build_composed_archive.py:47``).  Reported as the
    STANDALONE delta; the byte-close arm measures the true archive delta.
    """
    if table is None:
        return 0
    import zlib
    blob = json.dumps({"st_grid": list(table)}, separators=(",", ":")).encode()
    return len(zlib.compress(blob, 9))


def run_design(args: argparse.Namespace) -> None:
    _assert_incumbent()
    rows: dict[int, dict] = {}
    for path in sorted(args.out_dir.glob("ms8_curves_shard*.jsonl")):
        for ln in path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                rows[int(r["pair"])] = r
    missing = [i for i in range(N_PAIRS) if i not in rows]
    if missing and not args.allow_partial:
        raise SystemExit(f"curves incomplete: {len(rows)}/{N_PAIRS}, "
                         f"first missing {missing[:5]}")
    pairs = sorted(rows)
    n = len(pairs)
    curves = np.asarray([rows[p]["curve"] for p in pairs], np.float64)
    j_ship = np.asarray([rows[p]["j_shipped"] for p in pairs], np.int64)
    d_ctrl = curves[np.arange(n), j_ship]
    canary = np.asarray([rows[p]["canary_abs_err"] for p in pairs])
    sev = np.asarray(S_EVAL, np.float64)
    inc_cols = [int(np.argmin(np.abs(sev - c))) for c in INCUMBENT_ST_GRID]
    if len(set(inc_cols)) != len(inc_cols):
        raise SystemExit("two incumbent codewords collapsed onto one S_EVAL "
                         "column; the shipped-index reconstruction would be "
                         "ambiguous")

    base_mean = float(d_ctrl.mean())
    base_contrib = contribution(base_mean)
    # incumbent index (position within INCUMBENT_ST_GRID) of the shipped choice
    inc_of_col = {c: k for k, c in enumerate(inc_cols)}
    idx_ship = np.asarray([inc_of_col[int(j)] for j in j_ship], np.uint8)
    bytes_ship = st_stream_bytes(idx_ship, len(INCUMBENT_ST_GRID))

    def price(sub_cols, *, table_is_fitted: bool, label: str,
              search: str = "given") -> dict:
        sub = list(sub_cols)
        loc = curves[:, sub].argmin(axis=1)
        d = curves[:, sub][np.arange(n), loc]
        idx = loc.astype(np.uint8)
        levels = max(len(sub), 2)
        b_stream = st_stream_bytes(idx, levels)
        table = tuple(float(sev[c]) for c in sub)
        b_table = table_manifest_bytes(table if table_is_fitted else None)
        d_mean = float(d.mean())
        d_contrib = contribution(d_mean)
        shippable = b_stream is not None
        d_bytes = (b_stream + b_table) - bytes_ship if shippable else None
        rec = {
            "arm": label,
            "k": len(sub),
            "table": list(table),
            "table_is_fitted": bool(table_is_fitted),
            "codebook_search": search,
            "shippable_through_live_coder": shippable,
            "occupancy": np.bincount(idx, minlength=len(sub)).tolist(),
            "d_pose_mean": d_mean,
            "pose_contribution": d_contrib,
            "d_pose_delta_vs_ctrl": d_mean - base_mean,
            "stream_bytes": b_stream,
            "table_bytes": b_table,
            "delta_bytes_vs_shipped": None if d_bytes is None else int(d_bytes),
            "delta_S_pose": d_contrib - base_contrib,
            "delta_S_rate": None if d_bytes is None
                            else 25.0 * d_bytes / ARCHIVE_DENOM,
            "delta_S_total": None if d_bytes is None
                             else (d_contrib - base_contrib)
                             + 25.0 * d_bytes / ARCHIVE_DENOM,
            "pairs_moved": int((idx != idx_ship).sum())
                           if len(sub) == len(INCUMBENT_ST_GRID)
                           and list(sub) == inc_cols else None,
        }
        return rec

    arms = [price(inc_cols, table_is_fitted=False, label="RESEL_incumbent11")]
    live_cols = sorted({int(c) for c in inc_cols})
    for k in args.k_values:
        if k > len(S_EVAL):
            continue
        sub = kmedoid_subset(curves, k)
        arms.append(price(sub, table_is_fitted=True, label=f"DESIGN_k{k}",
                          search=subset_search_mode(len(S_EVAL), k)))
    arms.append(price(list(range(len(S_EVAL))), table_is_fitted=True,
                      label=f"FULL_k{len(S_EVAL)}"))
    # Control: the incumbent alphabet restricted to the cells it actually used.
    # This IS a fitted table -- WHICH cells survive is a fact about this video,
    # so rule 118 counts it even though every codeword came from the generic
    # ladder.  Only the untouched full incumbent ladder rides free.
    used = sorted({int(inc_cols[k]) for k in np.unique(idx_ship)})
    arms.append(price(used, table_is_fitted=True,
                      label=f"RESEL_livecells{len(used)}"))

    # DECOMPOSITION -- a headline composite is an unmeasured number.  Split the
    # move into the two mechanisms the charter names:
    #   SELECTION  (#882) : same 11 codewords, a better per-pair index
    #   PLACEMENT  (#873) : the extra buy from relocating the codewords
    best_ship = min((a for a in arms if a["delta_S_total"] is not None),
                    key=lambda r: (round(r["delta_S_total"], 9), r["k"]))
    resel = next(a for a in arms if a["arm"] == "RESEL_incumbent11")
    resel_cols = inc_cols
    resel_idx = curves[:, resel_cols].argmin(axis=1)
    best_cols = [int(np.argmin(np.abs(sev - c))) for c in best_ship["table"]]
    best_idx = curves[:, best_cols].argmin(axis=1)
    s_ship_val = sev[np.asarray(j_ship)]
    s_best_val = np.asarray([sev[best_cols[j]] for j in best_idx])
    d_best = curves[:, best_cols][np.arange(n), best_idx]
    gain = d_ctrl - d_best
    order = np.argsort(-gain)
    cum = np.cumsum(gain[order]) / max(gain.sum(), 1e-300)
    decomposition = {
        "winning_arm": best_ship["arm"],
        "selection_only_delta_S": resel["delta_S_total"],
        "placement_extra_delta_S": best_ship["delta_S_total"]
                                   - resel["delta_S_total"],
        "pairs_whose_index_moves_within_incumbent": int(
            (resel_idx != idx_ship).sum()),
        "pairs_whose_s_moves_at_all": int((s_best_val != s_ship_val).sum()),
        "pairs_landing_outside_the_incumbent_codewords": int(
            sum(1 for s in s_best_val
                if not any(abs(s - c) < 1e-12 for c in INCUMBENT_ST_GRID))),
        "abs_s_move_mean": float(np.abs(s_best_val - s_ship_val).mean()),
        "abs_s_move_max": float(np.abs(s_best_val - s_ship_val).max()),
        "pairs_that_regress": int((gain < 0).sum()),
        "gain_share_top_1pct_pairs": float(cum[max(int(0.01 * n) - 1, 0)]),
        "gain_share_top_10pct_pairs": float(cum[max(int(0.10 * n) - 1, 0)]),
    }

    occ_ship = np.bincount(idx_ship, minlength=len(INCUMBENT_ST_GRID)).tolist()
    receipt = {
        "schema": SCHEMA, "utc": _utc(), "n_pairs": n,
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "pointer_moved": False,
        "s_eval": list(S_EVAL),
        "incumbent_grid": list(INCUMBENT_ST_GRID),
        "incumbent_occupancy": occ_ship,
        "incumbent_dead_codewords": int(sum(1 for c in occ_ship if c == 0)),
        "incumbent_mode_share": max(occ_ship) / float(n),
        "canary_max_abs_err": float(canary.max()),
        "canary_exact_pairs": int((canary == 0.0).sum()),
        "ctrl_d_pose_mean": base_mean,
        "ctrl_pose_contribution": base_contrib,
        "ctrl_stream_bytes": bytes_ship,
        "clip_low_selected": int((curves.argmin(axis=1) == 0).sum()),
        "clip_high_selected": int(
            (curves.argmin(axis=1) == len(S_EVAL) - 1).sum()),
        "arms": arms,
        "live_cols": live_cols,
        "decomposition": decomposition,
    }
    out = args.out_dir / "ms8_design_receipt.json"
    out.write_text(json.dumps(receipt, indent=1) + "\n")

    if args.emit_override:
        # Emit the winning SHIPPABLE arm as a builder override.  Ties on
        # delta_S_total are broken toward the SMALLER codebook (fewer manifest
        # bytes, and a smaller alphabet is never worse through the coder).
        if len(pairs) != N_PAIRS:
            raise SystemExit("--emit-override requires the full n600 curve set")
        ship = [a for a in arms if a["delta_S_total"] is not None]
        if not ship:
            raise SystemExit("no arm is shippable through the live coder")
        best = min(ship, key=lambda r: (round(r["delta_S_total"], 9), r["k"]))
        cols = [int(np.argmin(np.abs(sev - c))) for c in best["table"]]
        loc = curves[:, cols].argmin(axis=1)
        doc = {
            "schema": "ddm_ms8_st_override.v1", "utc": _utc(),
            "arm": best["arm"],
            "st_grid": [float(sev[c]) for c in cols],
            "st_idx": [int(x) for x in loc],
            "advisory_d_pose_mean": best["d_pose_mean"],
            "advisory_pose_contribution": best["pose_contribution"],
            "ctrl_d_pose_mean": base_mean,
            "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
            "note": "byte-close through ddm_v4d_build_composed_archive.py "
                    "--st-override is the byte authority; the composed S here "
                    "is a PREDICTION.",
        }
        ov = args.out_dir / "ms8_st_override.json"
        ov.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"override ({best['arm']}) -> {ov}")

    print(f"\n=== ms8 codebook race, n={n} "
          f"[macOS-CPU frozen-PoseNet advisory] ===")
    print(f"canary: max |d_ctrl - d_shipped| = {canary.max():.3e} "
          f"({int((canary == 0.0).sum())}/{n} EXACT)")
    print(f"CTRL d_pose {base_mean:.8f}  contribution {base_contrib:.6f}  "
          f"stream {bytes_ship} B")
    print(f"incumbent occupancy {occ_ship}  dead={sum(1 for c in occ_ship if c == 0)}"
          f"  mode share {max(occ_ship)/n:.2%}")
    print(f"argmin at S_EVAL[0]={S_EVAL[0]}: {receipt['clip_low_selected']} pairs; "
          f"at S_EVAL[-1]={S_EVAL[-1]}: {receipt['clip_high_selected']} pairs "
          "(both ends past the incumbent support -> clipping check)")
    hdr = (f"{'arm':<22}{'K':>3}{'d_pose':>13}{'dS_pose':>11}"
           f"{'dBytes':>8}{'dS_rate':>11}{'dS_total':>11}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for a in sorted(arms, key=lambda r: (r["delta_S_total"] is None,
                                         r["delta_S_total"] or 0.0)):
        if a["delta_S_total"] is None:
            print(f"{a['arm']:<22}{a['k']:>3}{a['d_pose_mean']:>13.8f}"
                  f"{a['delta_S_pose']:>11.6f}"
                  f"{'  NOT SHIPPABLE (live coder caps levels at 16)':>30}")
            continue
        print(f"{a['arm']:<22}{a['k']:>3}{a['d_pose_mean']:>13.8f}"
              f"{a['delta_S_pose']:>11.6f}{a['delta_bytes_vs_shipped']:>8d}"
              f"{a['delta_S_rate']:>11.6f}{a['delta_S_total']:>11.6f}")
    print(f"\nDECOMPOSITION of {decomposition['winning_arm']}:")
    print(f"  SELECTION only (#882, incumbent codewords, better index): "
          f"dS {decomposition['selection_only_delta_S']:+.6f}  "
          f"({decomposition['pairs_whose_index_moves_within_incumbent']} pairs "
          "re-index)")
    print(f"  PLACEMENT extra (#873, relocated codewords):              "
          f"dS {decomposition['placement_extra_delta_S']:+.6f}")
    print(f"  {decomposition['pairs_landing_outside_the_incumbent_codewords']}"
          f"/{n} pairs land on an s_t the incumbent could not express; "
          f"mean |ds| {decomposition['abs_s_move_mean']:.5f}, max "
          f"{decomposition['abs_s_move_max']:.5f}")
    print(f"  gain concentration: top 1% of pairs "
          f"{decomposition['gain_share_top_1pct_pairs']:.1%}, top 10% "
          f"{decomposition['gain_share_top_10pct_pairs']:.1%}; "
          f"{decomposition['pairs_that_regress']} pairs regress")
    print(f"\nreceipt -> {out}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("curves", "design"), required=True)
    ap.add_argument("--base", default="celldrop50")
    ap.add_argument("--final-jsonl", type=Path,
                    default=Path("/Volumes/VertigoDataTier/pact/"
                                 "ddm_v4d_20260731/pw1/final_pw1.jsonl"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/Volumes/VertigoDataTier/pact/ddm_ms8_20260802"))
    # n600 IS the evidence bar (CLAUDE.md "allergic to non-n600-scale"); the
    # default is the full set and a subset must be asked for explicitly.
    ap.add_argument("--pairs", type=int, default=N_PAIRS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=600.0)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--emit-override", action="store_true",
                    help="write ms8_st_override.json for the winning shippable "
                         "arm (requires the full n600 curve set)")
    ap.add_argument("--k-values", type=int, nargs="*",
                    default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "curves":
        run_curves(args)
    else:
        run_design(args)


if __name__ == "__main__":
    main()
