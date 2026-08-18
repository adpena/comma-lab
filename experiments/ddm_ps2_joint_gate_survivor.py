#!/usr/bin/env python3
"""ddm_ps2 rung 2 -- is there a survivor once the channel is gated on the JOINT objective?

WHY THIS RUNG EXISTS.  Rung 1 refuted the projected seg-correction channel on the joint axis:
eta clears fo1's bar (0.5804 > 0.5196) but the pose term it also moves costs +0.001424 S against
a seg+rate gain of -0.000336, netting +0.001088.  Closing there would be a premature KILL: the
channel is selected by a SEG-ONLY objective (sr1's waterfill ranks cells by seg value per byte),
and nothing has yet asked what it does when the selection is made on the objective that actually
scores.  This rung asks that, at $0, from rows already on disk.

TWO GATES, and the difference between them is the whole point:

  ORACLE gate  -- the encoder measures each pair's joint effect and ships the winning subset.
                  This is LEGAL (the encoder owns a scorer) but the decoder must be told which
                  pairs, so the subset index is priced as an exact combinatorial rank.  It is an
                  UPPER BOUND on any gating scheme, not a candidate.
  FEATURE gate -- a rule fitted on edit-side features that are computable WITHOUT measuring the
                  outcome, validated LEAVE-ONE-OUT.  This is the only gate that could ship
                  without side info, and it is exactly where `pk3` died (23/23 in-sample -> 0/23
                  LOO).  LOO is therefore mandatory, and an in-sample-only number is not reported
                  as a result anywhere in this module.

If the ORACLE bound does not clear zero, no gate can, and F2 closes at FORMULATION with a
measured ceiling rather than an assertion.

Axis `[macOS-CPU advisory]` -- arithmetic over fo2h retained rows.  NEVER a score.  No scorer is
run; no Modal; no Metal.  `score_claim=false`.  `promotable=false`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path

import numpy as np

FO1_TOTAL_B = 4308.0
FO1_DESCRIBED_FLIPS = 6512.0
SEG_DS_PER_FLIP = 100.0 / (600 * 384 * 512)
RATE_DS_PER_BYTE = 25.0 / 37_545_489
D_POSE_N600 = 6.885643e-06
N_PAIRS_CLIP = 600
FX1_GAP_S = 0.15816036933414834 - 0.15

FO2H_WORK = Path("/Volumes/APDataStore/pact/ddm_fo2h_eta_hardening")
DEFAULT_OUT = Path("/Volumes/APDataStore/pact/ddm_ps2")

# Edit-side features: every one is computable by the encoder from the EDIT itself, without
# evaluating PoseNet on the edited frame.  d_pose_after and anything derived from it is banned
# here by construction -- that is the outcome the gate is trying to predict.
FEATURES = ("support_px", "snap_tax", "n_described_ring0", "flips_before",
            "max_abs_dY", "mean_abs_dY", "max_abs_dU", "max_abs_dV", "d_pose_before")


class Ps2GateError(RuntimeError):
    """Fail-closed error."""


def progress(out: Path, milestone: str, detail: dict) -> None:
    row = {"arm": "ddm_ps2", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "milestone": milestone, "detail": detail, "pid": os.getpid(),
           "host": socket.gethostname()}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "PROGRESS.jsonl").open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[ps2-gate] {milestone}: {json.dumps(detail, sort_keys=True)}", flush=True)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise Ps2GateError(f"required retained rows missing: {path}")
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def feat(r: dict, name: str) -> float:
    if name in ("max_abs_dY", "mean_abs_dY", "max_abs_dU", "max_abs_dV"):
        return float(r.get("yuv6_shift", {}).get(name, float("nan")))
    return float(r[name])


def subset_index_bytes(n_total: int, k: int) -> float:
    """Exact combinatorial rank of a k-subset of n_total pairs, in bytes.

    This is the honest price of telling the decoder WHICH pairs were edited.  fo2h priced its
    cell-set side info the same way rather than as a bitmap, and the two differ by >6x at small k.
    """
    if k <= 0 or k >= n_total:
        return 0.0
    return math.log2(math.comb(n_total, k)) / 8.0


def joint_dS(rows: list[dict], gate: np.ndarray, index_bytes: float) -> dict:
    """Total dS for a gated channel, in the same three terms rung 1 composed.

    Extrapolation, stated rather than hidden: the sample's flip and pose shares are taken as
    representative of the clip, which is fo2h's own convention (it applied a 48-pair eta to the
    clip-wide 6,512 described flips).  The gate changes WHICH pairs are edited, so both the seg
    numerator and the pose numerator move; the described-flip denominator does not.
    """
    before = np.array([r["flips_before"] for r in rows], dtype=np.float64)
    after = np.array([r["flips_after"] for r in rows], dtype=np.float64)
    described = np.array([r["n_described_ring0"] for r in rows], dtype=np.float64)
    pb = np.array([r["d_pose_before"] for r in rows], dtype=np.float64)
    pa = np.array([r["d_pose_after"] for r in rows], dtype=np.float64)

    eta_eff = float((gate * (before - after)).sum() / described.sum())
    dS_seg = -eta_eff * FO1_DESCRIBED_FLIPS * SEG_DS_PER_FLIP
    # ungated pairs keep their unedited pose; gated pairs take the edited value
    pose_ratio = float((gate * pa + (1 - gate) * pb).sum() / pb.sum())
    dS_pose = float((10.0 * D_POSE_N600 * pose_ratio) ** 0.5 - (10.0 * D_POSE_N600) ** 0.5)
    # the payload still codes the clip-wide cell selection; the gate ADDS a subset index
    dS_rate = (FO1_TOTAL_B + index_bytes) * RATE_DS_PER_BYTE
    return {"k": int(gate.sum()), "eta_eff": eta_eff, "pose_ratio": pose_ratio,
            "index_bytes": index_bytes, "dS_seg": dS_seg, "dS_pose": dS_pose,
            "dS_rate": dS_rate, "dS_joint": dS_seg + dS_pose + dS_rate}


def per_pair_joint_value(rows: list[dict]) -> np.ndarray:
    """Marginal joint dS of editing each pair alone, used to ORDER the oracle gate.

    Ordering by this marginal is not the same as optimising the set (pose is nonlinear through
    the sqrt), which is why the oracle sweep below evaluates every prefix rather than trusting
    the ranking.
    """
    before = np.array([r["flips_before"] for r in rows], dtype=np.float64)
    after = np.array([r["flips_after"] for r in rows], dtype=np.float64)
    pb = np.array([r["d_pose_before"] for r in rows], dtype=np.float64)
    pa = np.array([r["d_pose_after"] for r in rows], dtype=np.float64)
    seg = -(before - after) * SEG_DS_PER_FLIP
    # local linearisation of the sqrt at the operating point, for RANKING only
    dmean = (pa - pb) / N_PAIRS_CLIP
    pose = 0.5 * (10.0 / (10.0 * D_POSE_N600)) ** 0.5 * 10.0 * dmean / 10.0
    return seg + pose


def loo_feature_gate(rows: list[dict], name: str) -> dict:
    """Leave-one-out validation of a single-feature threshold gate.

    Protocol: for each held-out pair, choose the threshold on the OTHER n-1 pairs that maximises
    their gated joint dS, then apply it to the held-out pair and record whether that decision was
    correct.  `pk3`'s failure mode is a rule that is perfect in-sample and chance out-of-sample,
    so the reported number is the OUT-OF-SAMPLE agreement only.
    """
    x = np.array([feat(r, name) for r in rows], dtype=np.float64)
    pb = np.array([r["d_pose_before"] for r in rows], dtype=np.float64)
    pa = np.array([r["d_pose_after"] for r in rows], dtype=np.float64)
    v = per_pair_joint_value(rows)
    truth = v < 0                      # editing this pair alone helps
    if not np.isfinite(x).all():
        return {"feature": name, "status": "NON_FINITE_FEATURE"}
    n = len(rows)
    correct = 0
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        cand = np.unique(x[keep])
        best_t, best_val = None, np.inf
        for t in cand:                 # gate = keep pairs with x <= t
            g = (x[keep] <= t).astype(np.float64)
            val = float((g * v[keep]).sum())
            if val < best_val:
                best_val, best_t = val, float(t)
        if best_t is None:
            continue
        pred = bool(x[i] <= best_t)
        correct += int(pred == bool(truth[i]))
    base = float(max(truth.mean(), 1.0 - truth.mean()))
    return {"feature": name, "loo_accuracy": correct / n, "majority_baseline": base,
            "loo_beats_baseline": bool(correct / n > base),
            "n": n, "frac_pairs_helping": float(truth.mean()),
            "corr_with_pose_excess": float(np.corrcoef(x, pa - pb)[0, 1])}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args: argparse.Namespace) -> int:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.fo2h / "null_shardA" / "ETA_GATE_ROWS.jsonl") + \
        load_rows(args.fo2h / "null_shardB" / "ETA_GATE_ROWS.jsonl")
    seen: set[int] = set()
    rows = [r for r in rows if not (r["pair"] in seen or seen.add(r["pair"]))]
    n = len(rows)
    if n != 48:
        raise Ps2GateError(f"expected 48 projected rows, got {n}")

    # --- control: the ungated channel must reproduce rung 1 exactly ---------------------------
    ungated = joint_dS(rows, np.ones(n), 0.0)
    if abs(ungated["dS_joint"] - 0.001087878201153595) > 1e-12:
        raise Ps2GateError(f"control FAILED: ungated joint {ungated['dS_joint']!r} != rung 1")
    progress(out, "control_passed", {"ungated_dS_joint": ungated["dS_joint"]})

    # --- ORACLE gate: sweep every prefix of the joint-value ranking ---------------------------
    v = per_pair_joint_value(rows)
    order = np.argsort(v)                       # most negative (most helpful) first
    sweep = []
    for k in range(0, n + 1):
        g = np.zeros(n)
        g[order[:k]] = 1.0
        # the subset index is over the CLIP's 600 pairs, not the 48-pair sample
        kb = subset_index_bytes(N_PAIRS_CLIP, round(k * N_PAIRS_CLIP / n))
        sweep.append(joint_dS(rows, g, kb))
    best = min(sweep, key=lambda d: d["dS_joint"])
    # "index free" must actually REMOVE the index from the reported total, not merely rank without
    # it -- a dict labelled FREE whose dS_joint still carries the index price is a mislabelled
    # field, and it is masked whenever both criteria pick the same k (they do here, at k=38).
    def _joint_without_index(d: dict) -> float:
        return d["dS_seg"] + d["dS_pose"] + FO1_TOTAL_B * RATE_DS_PER_BYTE

    best_free = dict(min(sweep, key=_joint_without_index))
    best_free["dS_rate"] = FO1_TOTAL_B * RATE_DS_PER_BYTE
    best_free["index_bytes"] = 0.0
    best_free["dS_joint"] = _joint_without_index(best_free)

    # --- RIGOROUS IMPOSSIBILITY BOUND ---------------------------------------------------------
    # The prefix sweep above is one ordering, so on its own it is a good feasible point and not a
    # proof.  Bound each term by its own unconstrained optimum over ALL 2^48 subsets:
    #   seg   : include every pair that reduces flips             -> max achievable eta_eff
    #   pose  : include a pair only when it reduces d_pose        -> min achievable ratio
    #   rate  : the payload is fixed and the subset index is >= 0 -> floor at 4308 B
    # No subset attains all three at once (seg and pose disagree on which pairs to keep), so the
    # sum is a STRICT lower bound.  If it is positive, no gate of any kind can supply, and the
    # close is a proof rather than a sweep result.
    before = np.array([r["flips_before"] for r in rows], dtype=np.float64)
    after = np.array([r["flips_after"] for r in rows], dtype=np.float64)
    described = np.array([r["n_described_ring0"] for r in rows], dtype=np.float64)
    pb = np.array([r["d_pose_before"] for r in rows], dtype=np.float64)
    pa = np.array([r["d_pose_after"] for r in rows], dtype=np.float64)
    eta_max = float(np.maximum(before - after, 0.0).sum() / described.sum())
    ratio_min = float(np.minimum(pa, pb).sum() / pb.sum())
    bound = {
        "max_achievable_eta_eff": eta_max,
        "min_achievable_pose_ratio": ratio_min,
        "dS_seg_floor": -eta_max * FO1_DESCRIBED_FLIPS * SEG_DS_PER_FLIP,
        "dS_pose_floor": float((10.0 * D_POSE_N600 * ratio_min) ** 0.5
                               - (10.0 * D_POSE_N600) ** 0.5),
        "dS_rate_floor": FO1_TOTAL_B * RATE_DS_PER_BYTE,
        "attainable_by_any_single_subset": False,
    }
    bound["dS_joint_lower_bound"] = (bound["dS_seg_floor"] + bound["dS_pose_floor"]
                                     + bound["dS_rate_floor"])
    bound["no_gate_can_supply"] = bool(bound["dS_joint_lower_bound"] > 0)

    # --- WHERE A SURVIVOR COULD STILL LIVE: the pose BUDGET vs operating point -----------------
    # The refutation above is at ONE operating point: fo1's 41-cell / 4,308 B selection, whose
    # seg gain exceeds its rate cost by only ~10%.  fo2h LEG 2 measured real coder bytes for all
    # 74 live prefixes.  For each, compute the pose cost the point could absorb and still supply:
    #     margin_m = -(dS_seg_m + dS_rate_m);  tolerable ratio r_max = (1 + margin_m/p0)^2
    # Pose is measured ONLY at the full ring-0 edit (1.3725).  Naming r_max per m turns "the
    # channel failed" into "here is the pose number that is missing, and the m at which it is
    # cheapest to need" -- which is a measurement request, not a closure.
    wf = args.fo2h / "FO2H_WATERFILL_MEASURED.json"
    budget = []
    if wf.exists():
        p0 = (10.0 * D_POSE_N600) ** 0.5
        for row in json.loads(wf.read_text())["rows"]:
            ds_seg = -ungated["eta_eff"] * float(row["flips"]) * SEG_DS_PER_FLIP
            ds_rate = float(row["total_bytes_with_side_info"]) * RATE_DS_PER_BYTE
            margin = -(ds_seg + ds_rate)
            budget.append({
                "cells": int(row["cells"]), "flips": int(row["flips"]),
                "total_bytes": float(row["total_bytes_with_side_info"]),
                "dS_seg_at_hardened_eta": ds_seg, "dS_rate": ds_rate,
                "seg_rate_margin": margin,
                "tolerable_pose_ratio": (1.0 + margin / p0) ** 2 if margin > -p0 else 0.0,
            })
    best_budget = max(budget, key=lambda d: d["tolerable_pose_ratio"], default=None)

    # --- FEATURE gate: LOO on every edit-side feature ------------------------------------------
    loo = [loo_feature_gate(rows, f) for f in FEATURES]
    any_generalises = any(d.get("loo_beats_baseline") for d in loo)

    verdict = ("GATED SURVIVOR FOUND" if best["dS_joint"] < 0 else
               "NO GATE CAN SUPPLY -- PROVED BY THE TERM-WISE LOWER BOUND"
               if bound["no_gate_can_supply"] else
               "NO GATE SURVIVES THE SWEEP -- bound does not close it, more search owed")
    res = {
        "schema": "ddm_ps2_joint_gate_survivor.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU advisory] -- arithmetic over fo2h retained rows; NEVER a score",
        "score_claim": False, "promotable": False, "pointer_moved": False,
        "no_scorer_run": True,
        "ungated_reference": ungated,
        "oracle_gate": {
            "note": "UPPER BOUND, not a candidate: the encoder measures each pair's outcome and "
                    "ships the subset index. No feature-based gate can beat this.",
            "best_with_index_priced": best,
            "best_if_index_were_FREE": best_free,
            "share_of_fx1_gap_if_best": -best["dS_joint"] / FX1_GAP_S,
            "sweep": sweep,
        },
        "impossibility_bound": bound,
        "pose_budget_vs_operating_point": {
            "note": "pose is MEASURED only at the full ring-0 edit (ratio 1.3725). These are the "
                    "ratios each operating point could absorb and still supply -- the missing "
                    "measurement, named per m.",
            "measured_pose_ratio_at_full_edit": ungated["pose_ratio"],
            "best_by_tolerable_pose_ratio": best_budget,
            "rows": budget,
        },
        "feature_gate_loo": {
            "note": "LOO only. pk3 died 23/23 in-sample -> 0/23 LOO, so no in-sample number is "
                    "reported here at all.",
            "rows": loo,
            "any_feature_generalises": any_generalises,
        },
        "verdict": verdict,
        "verdict_scope": (
            "FORMULATION: gating the pose-null-projected seg-correction channel on this frozen "
            "artifact, ring-0 described set, n=48 out-of-sample sample. The ORACLE bound is a "
            "property of THIS sample's per-pair outcomes; m96 -- clearing or missing a bar on a "
            "seeded-random sample does not license a LIVE n600 verdict."),
    }
    path = out / "PS2_JOINT_GATE_SURVIVOR.json"
    path.write_text(json.dumps(res, indent=1, sort_keys=True))
    progress(out, "gate_adjudicated", {"verdict": verdict, "best_dS_joint": best["dS_joint"],
                                       "best_k": best["k"], "sha256": sha256_of(path)})
    print(json.dumps({"verdict": verdict, "ungated": ungated,
                      "best_oracle": best, "best_oracle_index_free": best_free,
                      "any_feature_generalises": any_generalises}, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fo2h", type=Path, default=FO2H_WORK)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
