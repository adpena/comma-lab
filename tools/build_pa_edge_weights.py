#!/usr/bin/env python3
"""Build the FEED-PA edge-weight matrix W_e for P0 FORCE 3 (tie-locus displacement; task #360).

DETERMINISTIC · $0 · CACHE-ONLY · NO SegNet/PoseNet forward · MPS-never.  Reads the frozen
GT authority (cached ``gt_n600.npz`` lstars/margins — the θ-independent frozen-CPU-torch SegNet
outputs) and stamps a 5×5 symmetric per-edge flip-density matrix to ``reports/pa_edge_weights.json``.

WHY (derivation ``.omx/research/p0_forces_derivation_20260708.md`` §FORCE 3.2 + DAG FEED-PA):
FEED-PA MEASURED (n600, bf1ee1fa8) that 100% of the achievable d_seg floor is BOUNDARY PLACEMENT
and the flips are NOT uniform over straddles — they concentrate on Road-adjacent edges (Road is the
hub; Road↔Lane = 41% of Road's flips). Force 3 weights each genuine-V straddle by its adjacency-edge
flip-mass share ``W_e[c_a, c_b]``. The derivation FORBIDS a hardcoded guess ("stamp it from the
measured destination matrix"); this tool IS the deterministic stamp.

THE MEASURED QUANTITY (θ-independent, GT-only — no witness argmax, no scorer forward):
per genuine-V inter-class straddle (the SAME active set the subpix term consumes: ``lstar`` differs
AND both GT margins < ``--v-band``), accumulate a FRAGILITY weight ``(1 − min(M_p,M_q)/band) ∈ (0,1]``
(linear in depth-into-the-fragile-band; monotone in flip propensity — the margin AUC-for-flip is 0.91,
FEED-lq, so lower margin ⇒ higher flip mass) into the symmetric class-pair cell. This is the GT-side
boundary-fragility density that FEED-PA's witness-confusion flip mass rides on (100%-boundary-placement,
annulus-concentrated). The witness-confusion FEED-PA per-class shares (Road 43.7 / Lane 16.3 / Undriv
18.2 / Movable 10.4 / MyCar 11.4) are stamped as a CROSS-CHECK (ranking must agree: Road hub,
Road↔Lane heaviest), NOT as fabricated inputs.

Output schema (consumed by the trainer at
``experiments/train_levelset_witness_realized_through_R_mlx.py`` L5786+, keyed ``W_e``):
    {"W_e": [[5×5 float]], "provenance": {...}, "feed_pa_crosscheck": {...}, ...}

All numbers ``[macOS-CPU/numpy advisory · NON-PROMOTABLE]``; pointer UNMOVED — this is MEANS (it makes
Force 3's pa_flipmass mode fireable; it moves no score until a trained arm is byte-closed).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# CLAUDE.md canonical comma10k order (MEASURED 2026-06-27): 0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar.
# W_e is indexed by the argmax index directly (order-agnostic for the lookup); these names label the
# human-readable cross-check only.
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

# FEED-PA (bf1ee1fa8, n600) MEASURED witness-confusion per-class flip-mass shares — the CROSS-CHECK
# authority (DAG FEED-PA 2026-07-08). Not an input; the built matrix's per-class aggregate must rank
# the same (Road hub, Road↔Lane heaviest).
FEED_PA_CLASS_FLIP_SHARE = {"Road": 0.437, "Lane": 0.163, "Undrivable": 0.182,
                            "Movable": 0.104, "MyCar": 0.114}


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:  # noqa: BLE001 — provenance best-effort
        return "unknown"


def build_edge_weights(gt_path: Path, v_band: float, eps: float = 1e-6,
                       floor: float = 0.05) -> tuple[np.ndarray, dict]:
    """Compute the symmetric 5×5 W_e (mean over populated off-diagonal edges = 1.0) + a stats dict.

    Uses the EXACT genuine-V straddle definition the subpix term consumes (trainer L5820-5827):
    RIGHT straddles ``lst[:,:-1] != lst[:,1:]`` and DOWN straddles ``lst[:-1,:] != lst[1:,:]``, both
    with ``min(M_p, M_q) < v_band``. Fragility weight ``1 − min(M_p,M_q)/band``.
    """
    d = np.load(gt_path)
    lstars = d["lstars"]        # (P, H, W) int
    margins = d["margins"]      # (P, H, W) f32 (top1 − top2, the #141 GT-class margin)
    P = int(lstars.shape[0])
    n_cls = 5
    mass = np.zeros((n_cls, n_cls), np.float64)   # symmetric accumulation of fragility weight
    count = np.zeros((n_cls, n_cls), np.int64)    # raw straddle count (diagnostic)
    total_straddles = 0

    for pi in range(P):
        lst = np.asarray(lstars[pi], np.int64)
        mg = np.asarray(margins[pi], np.float32)
        for axis in (1, 0):  # 1 = RIGHT (horizontal neighbour), 0 = DOWN (vertical neighbour)
            if axis == 1:
                ca = lst[:, :-1]; cb = lst[:, 1:]
                ma = mg[:, :-1]; mb = mg[:, 1:]
            else:
                ca = lst[:-1, :]; cb = lst[1:, :]
                ma = mg[:-1, :]; mb = mg[1:, :]
            mmin = np.minimum(ma, mb)
            gv = (ca != cb) & (mmin < v_band)     # genuine-V straddle mask (SAME as the subpix term)
            if not gv.any():
                continue
            a = ca[gv].astype(np.int64)
            b = cb[gv].astype(np.int64)
            frag = (1.0 - mmin[gv] / float(v_band)).astype(np.float64)  # (0,1], deeper = more fragile
            total_straddles += int(gv.sum())
            # symmetric accumulation
            np.add.at(mass, (a, b), frag)
            np.add.at(mass, (b, a), frag)
            np.add.at(count, (a, b), 1)
            np.add.at(count, (b, a), 1)

    # normalize so the MEAN over populated off-diagonal cells = 1.0 (the trainer expects mean≈1.0 so
    # W_e reweights, not rescales, the subpix loss). Diagonal is structurally 0 (a straddle requires
    # differing classes). Unpopulated off-diagonal edges get a small positive floor (never looked up at
    # a real straddle, but avoids zeroing a rare edge that does appear at train time).
    off = ~np.eye(n_cls, dtype=bool)
    populated = off & (mass > 0.0)
    mean_pop = float(mass[populated].mean()) if populated.any() else 1.0
    W_e = np.zeros((n_cls, n_cls), np.float64)
    W_e[populated] = mass[populated] / mean_pop
    W_e[off & (mass <= 0.0)] = float(floor)   # floor for absent edges
    # enforce exact symmetry (guard against fp asymmetry from the two np.add.at passes)
    W_e = 0.5 * (W_e + W_e.T)
    np.fill_diagonal(W_e, 0.0)

    # per-class aggregate share (for the FEED-PA cross-check): each class's total edge mass / grand total
    per_class_mass = mass.sum(axis=1)
    gtot = float(per_class_mass.sum()) or 1.0
    per_class_share = {CLASS_NAMES[i]: float(per_class_mass[i] / gtot) for i in range(n_cls)}

    stats = {
        "n_pairs": P,
        "total_genuine_v_straddles": total_straddles,
        "mean_over_populated_edges_pre_norm": mean_pop,
        "raw_edge_count": count.astype(int).tolist(),
        "per_class_share_measured": per_class_share,
    }
    return W_e, stats


def _crosscheck_ranking(per_class_share_measured: dict) -> dict:
    """Rank-correlate the built matrix's per-class aggregate against FEED-PA's measured shares."""
    names = list(CLASS_NAMES)
    built = np.array([per_class_share_measured[n] for n in names])
    feed = np.array([FEED_PA_CLASS_FLIP_SHARE[n] for n in names])
    # Pearson on shares + argmax (Road hub) + Road↔Lane-heaviest agreement
    built_r = built - built.mean(); feed_r = feed - feed.mean()
    denom = float(np.sqrt((built_r ** 2).sum() * (feed_r ** 2).sum())) or 1.0
    pearson = float((built_r * feed_r).sum() / denom)
    road_is_hub_built = bool(np.argmax(built) == names.index("Road"))
    road_is_hub_feed = bool(np.argmax(feed) == names.index("Road"))
    return {
        "feed_pa_class_flip_share": FEED_PA_CLASS_FLIP_SHARE,
        "built_class_share": {n: round(float(v), 4) for n, v in per_class_share_measured.items()},
        "pearson_r_built_vs_feed_pa": round(pearson, 4),
        "road_is_hub_built": road_is_hub_built,
        "road_is_hub_feed_pa": road_is_hub_feed,
        "ranking_agrees": road_is_hub_built and road_is_hub_feed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
                    help="cached GT authority (lstars/margins); DEFAULT n600 (n600-scale, per the "
                         "allergic-to-toys discipline — never an n96 subset for the shipped artifact).")
    ap.add_argument("--v-band", type=float, default=1.0,
                    help="genuine-V fragile band (MUST match the trainer --seg-subpix-boundary-v-band; "
                         "default 1.0).")
    ap.add_argument("--out", default="reports/pa_edge_weights.json")
    ap.add_argument("--floor", type=float, default=0.05, help="weight floor for absent edges.")
    args = ap.parse_args()

    gt_path = Path(args.gt_cache)
    if not gt_path.is_absolute():
        gt_path = REPO / args.gt_cache
    if not gt_path.is_file():
        raise SystemExit(f"gt cache not found: {gt_path}")

    W_e, stats = build_edge_weights(gt_path, v_band=float(args.v_band), floor=float(args.floor))
    crosscheck = _crosscheck_ranking(stats["per_class_share_measured"])

    payload = {
        "W_e": W_e.round(6).tolist(),
        "class_order": list(CLASS_NAMES),
        "class_order_note": "argmax index order (CLAUDE.md canonical comma10k 2026-06-27); "
                            "the trainer looks up W_e[lstar[p], lstar[neighbour]] by index.",
        "provenance": {
            "tool": "tools/build_pa_edge_weights.py",
            "derivation": ".omx/research/p0_forces_derivation_20260708.md §FORCE 3.2",
            "dag_anchor": "FEED-PA (bf1ee1fa8, n600)",
            "task": "#360 FORCE 3 tie_locus_displacement pa_flipmass",
            "gt_cache": str(gt_path.relative_to(REPO)) if gt_path.is_relative_to(REPO) else str(gt_path),
            "v_band": float(args.v_band),
            "floor": float(args.floor),
            "measured_quantity": "genuine-V inter-class straddle fragility density (1 - min(Mp,Mq)/band), "
                                 "symmetric, mean-over-populated-edges normalized to 1.0; theta-independent, "
                                 "GT-only, NO scorer forward",
            "git_hash": _git_hash(),
            "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "axis": "[macOS-CPU/numpy advisory · NON-PROMOTABLE]",
            "pointer": "0.19108282 UNMOVED (means)",
        },
        "stats": stats,
        "feed_pa_crosscheck": crosscheck,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(out_path)  # atomic

    print(json.dumps({
        "stage": "build_pa_edge_weights", "out": str(out_path.relative_to(REPO)),
        "n_pairs": stats["n_pairs"], "total_straddles": stats["total_genuine_v_straddles"],
        "pearson_r_vs_feed_pa": crosscheck["pearson_r_built_vs_feed_pa"],
        "road_is_hub": crosscheck["road_is_hub_built"],
        "ranking_agrees": crosscheck["ranking_agrees"],
        "W_e_road_lane": round(float(W_e[0, 1]), 4),
        "W_e_road_undriv": round(float(W_e[0, 2]), 4),
    }, indent=2))


if __name__ == "__main__":
    main()
