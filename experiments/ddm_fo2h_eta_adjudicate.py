#!/usr/bin/env python3
"""ddm_fo2h LEG 1 -- adjudicate the hardened eta against fo1's FROZEN break-even.

WHAT THIS PAYS.  fo1 priced sr1's waterfilled seg-correction channel with a real, round-trip
verified coder and found it a supplier by -0.000505 S -- entirely conditional on eta = 0.6111,
which pn2 measured on n=12 seeded-random pairs and explicitly refused to call a level:

    "the eta advantage fell +15.6% (n=4) -> +14.2% (n=7) -> +11.7% (n=10) -> +8.1% (n=12) ...
     Quote the direction and the order of magnitude, never a level."   -- pn2 memo s3

The entire supplier margin therefore lives inside a regressing n=12 number.  This module hardens
it on a sample that is DISJOINT from pn2's by construction and adjudicates against fo1's frozen
break-even.

PRE-REGISTERED ADJUDICATION, against the FROZEN bar 0.5196321126365346 (fo1's break-even on its
measured 4,308 B; frozen BEFORE this arm ran and NOT recomputed from anything this arm produces):

    eta_pooled AND its lower spread edge > bar   -> SUPPLIER CONFIRMED-HARDENED
    eta_pooled > bar but the spread STRADDLES it -> INDETERMINATE-MORE-N (report the n required)
    eta_pooled <= bar                            -> SUPPLIER REFUTED-AT-HARDENED-ETA
                                                    (fo1's verdict was an n=12 artifact)

GOALPOST DISCIPLINE.  LEG 2 of this same arm re-optimizes the cell selection and produces a LOWER
break-even.  Adjudicating against that number instead would be this arm marking its own homework,
so the primary verdict is against the FROZEN bar and the re-optimized bar is reported only as a
secondary "how much wider the margin gets if the selection is also fixed".

SPREAD (the falsifier-band law, `seed_ensemble_falsifier_band_v1`).  That law's content is: one
draw is not the population, so calibrate the band on an ENSEMBLE and refuse a verdict whose bar
the spread STRADDLES.  Its sigma form is realized here on the two independent equal-size shards
this arm ran (`sigma = |eta_A - eta_B| / sqrt(2)` for a shard, /sqrt(2) again for their pooled
mean), and cross-checked by a pair-level bootstrap of the pooled ratio estimator.  Two
independent spread estimators are reported; if they disagree the weaker (wider) one governs,
because a spread estimate that is itself uncertain must not be allowed to license a verdict.

WHY THE POOLED ETA IS A RATIO OF SUMS, NOT A MEAN OF RATIOS.  eta enters the S arithmetic as
(flips recovered)/(flips described) over the whole clip, so the estimator that matches the
contest arithmetic is sum(before-after)/sum(described).  A mean of per-pair etas weights a pair
describing 5 flips the same as one describing 200.  Both are reported; the pooled one governs.

Axis `[macOS-CPU advisory]` -- NEVER a score.  `score_claim=false`.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

from tac.canonical_equations.seed_ensemble_falsifier_band_20260817 import (
    band_boundary_straddled,
    seed_sigma_est,
)

# --- FROZEN pins, cited from the stores, never recomputed here --------------------------------
FO1_BREAKEVEN_ETA = 0.5196321126365346   # fo1 s5 -- the bar, frozen before this arm ran
FO1_TOTAL_B = 4308.0                     # fo1's measured round-trip-verified bytes
PN2_ETA_PROJECTED_N12 = 0.6111           # the number being hardened
PN2_ETA_UNPROJECTED_N12 = 0.5651
PN2_PAIRS = (33, 66, 81, 89, 280, 299, 322, 353, 410, 438, 474, 538)
SEG_DS_PER_FLIP = 100.0 / (600 * 384 * 512)
RATE_DS_PER_BYTE = 25.0 / 37_545_489
BASE_S = 0.15959729295498598
GAP_S = BASE_S - 0.15
D_POSE_N600 = 6.885643e-06               # hv1 contest-CUDA n600 pose term (pn2 s5)

DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_fo2h_eta_hardening")
PN2_WORK = Path("/Volumes/APDataStore/pact/ddm_pn2")
# rt1's retained PROJECTED arm on pn2's 12 pairs -- the rows behind pn2's 0.6111.  Named
# explicitly rather than derived from PN2_WORK.parent, which reads like an accident.
PN2_NULL_ROWS = Path("/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816"
                     "/eta_gate_null/ETA_GATE_ROWS.jsonl")


class Fo2hError(RuntimeError):
    """Fail-closed error."""


def progress(work: Path, milestone: str, detail: dict) -> None:
    row = {"arm": "ddm_fo2h", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "milestone": milestone, "detail": detail, "pid": os.getpid(),
           "host": socket.gethostname()}
    work.mkdir(parents=True, exist_ok=True)
    with (work / "PROGRESS.jsonl").open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[fo2h] {milestone}: {json.dumps(detail, sort_keys=True)}", flush=True)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def pooled_eta(rows: list[dict]) -> float:
    b = sum(r["flips_before"] for r in rows)
    a = sum(r["flips_after"] for r in rows)
    d = sum(r["n_described_ring0"] for r in rows)
    return (b - a) / d if d else float("nan")


def pose_agg_ratio(rows: list[dict]) -> float:
    """upstream/evaluate.py aggregates d_pose as a MEAN OVER PAIRS and only then takes the sqrt,
    so the aggregate ratio is mean(after)/mean(before) -- never a mean or median of per-pair
    ratios, which weights a pair carrying no pose the same as one carrying the axis."""
    db = np.array([r["d_pose_before"] for r in rows], dtype=np.float64)
    da = np.array([r["d_pose_after"] for r in rows], dtype=np.float64)
    return float(da.mean() / db.mean()) if db.mean() else float("nan")


def bootstrap_spread(rows: list[dict], n_boot: int, seed: int) -> dict:
    """Pair-level bootstrap of the pooled ratio estimator: 'would another draw of these n pairs
    have cleared the bar?'  Resampling PAIRS is the right unit because the pair is the sampling
    unit -- the pixels inside a pair are not independent draws."""
    rng = np.random.default_rng(seed)
    b = np.array([r["flips_before"] for r in rows], dtype=np.float64)
    a = np.array([r["flips_after"] for r in rows], dtype=np.float64)
    d = np.array([r["n_described_ring0"] for r in rows], dtype=np.float64)
    n = len(rows)
    idx = rng.integers(0, n, size=(n_boot, n))
    num = (b - a)[idx].sum(axis=1)
    den = d[idx].sum(axis=1)
    est = num / den
    return {"n_boot": n_boot, "seed": seed,
            "sd": float(est.std(ddof=1)),
            "p02_5": float(np.percentile(est, 2.5)),
            "p16": float(np.percentile(est, 16.0)),
            "p50": float(np.percentile(est, 50.0)),
            "p84": float(np.percentile(est, 84.0)),
            "p97_5": float(np.percentile(est, 97.5)),
            "frac_below_bar": float((est <= FO1_BREAKEVEN_ETA).mean())}


def cumulative_curve(rows: list[dict], seed: int) -> list[dict]:
    """Pooled eta as rows land, in a SHUFFLED order.

    Deliberately not the on-disk order: the shards were dispatched in ascending pair index, which
    is TIME order in the clip, so a cumulative curve in that order would read scene drift as
    convergence (the m88 prefix genus wearing an 'incremental' costume)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    out = []
    for k in range(1, len(rows) + 1):
        sub = [rows[i] for i in order[:k]]
        out.append({"n": k, "pooled_eta": pooled_eta(sub)})
    return out


def matched_advantage_curve(free_rows: list[dict], null_rows: list[dict],
                            pairs: list[int], seed: int) -> list[dict]:
    """Cumulative null-vs-free eta ADVANTAGE (%) over a shuffled pair order.

    This is the direct continuation of pn2's regression series.  Shuffled, not on-disk order,
    for the same reason as `cumulative_curve`: pair index is time in the clip.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    fmap = {r["pair"]: r for r in free_rows}
    nmap = {r["pair"]: r for r in null_rows}
    out = []
    for k in range(1, len(pairs) + 1):
        sub = [pairs[i] for i in order[:k]]
        ef = pooled_eta([fmap[p] for p in sub])
        en = pooled_eta([nmap[p] for p in sub])
        out.append({"n": k, "pooled_eta_free": ef, "pooled_eta_null": en,
                    "advantage_pct": (100.0 * (en - ef) / ef) if ef else None})
    return out


def net_dS(eta: float, flips: float, total_B: float) -> float:
    return -eta * flips * SEG_DS_PER_FLIP + total_B * RATE_DS_PER_BYTE


def adjudicate(args: argparse.Namespace) -> int:
    work = args.work
    new_rows: list[dict] = []
    shards = {}
    for name in args.shards:
        rws = load_rows(work / name / "ETA_GATE_ROWS.jsonl")
        shards[name] = rws
        new_rows.extend(rws)
    seen: set[int] = set()
    dedup: list[dict] = []
    for r in new_rows:
        if r["pair"] not in seen:
            seen.add(r["pair"])
            dedup.append(r)
    new_rows = sorted(dedup, key=lambda r: r["pair"])
    if not new_rows:
        raise Fo2hError("no NEW rows landed -- refusing to emit an adjudication")

    overlap = sorted({r["pair"] for r in new_rows} & set(PN2_PAIRS))
    if overlap:
        raise Fo2hError(f"NEW sample overlaps pn2's n=12 at {overlap} -- not out-of-sample")

    pn2_null = load_rows(PN2_NULL_ROWS)
    eta_new = pooled_eta(new_rows)
    pooled_all = sorted(new_rows + pn2_null, key=lambda r: r["pair"])
    eta_pooled_60 = pooled_eta(pooled_all)

    # --- spread estimator 1: the falsifier-band law on the two independent equal-size shards ---
    shard_names = [n for n in args.shards if n.startswith("null_shard")]
    shard_etas = {n: pooled_eta(shards[n]) for n in shard_names if shards[n]}
    band = {}
    if len(shard_etas) == 2:
        (na, ea), (nb, eb) = sorted(shard_etas.items())
        sigma_shard = seed_sigma_est(ea, eb)
        n_a, n_b = len(shards[na]), len(shards[nb])
        # sigma of the mean of two equal-size shards; unequal sizes are reported, not smoothed
        sigma_pooled = sigma_shard / np.sqrt(2.0)
        band = {
            "law": "seed_ensemble_falsifier_band_v1 (ensemble-calibrate, then check straddle)",
            "shard_a": {"name": na, "n": n_a, "pooled_eta": ea},
            "shard_b": {"name": nb, "n": n_b, "pooled_eta": eb},
            "shards_equal_size": n_a == n_b,
            "sigma_est_per_shard": float(sigma_shard),
            "sigma_est_pooled": float(sigma_pooled),
            "bar_straddled_by_shard_spread": bool(
                band_boundary_straddled(FO1_BREAKEVEN_ETA, ea, eb)),
            "note": "n=2 sigma is itself wide (~76% rel. error, the law's own caveat)",
        }

    # --- spread estimator 2: pair-level bootstrap of the pooled ratio estimator ----------------
    boot = bootstrap_spread(new_rows, args.n_boot, args.boot_seed)
    boot60 = bootstrap_spread(pooled_all, args.n_boot, args.boot_seed) if pn2_null else None

    # --- the governing lower edge: the WIDER (more conservative) of the two estimators ---------
    edges = [boot["p16"]]
    if band:
        edges.append(eta_new - band["sigma_est_pooled"])
    lower_1sigma = float(min(edges))
    lower_2sigma = float(min([boot["p02_5"]]
                            + ([eta_new - 2 * band["sigma_est_pooled"]] if band else [])))

    if eta_new <= FO1_BREAKEVEN_ETA:
        verdict = "SUPPLIER REFUTED-AT-HARDENED-ETA"
    elif lower_1sigma > FO1_BREAKEVEN_ETA:
        verdict = "SUPPLIER CONFIRMED-HARDENED"
    else:
        verdict = "INDETERMINATE-MORE-N"

    # n required for the 1-sigma lower edge to clear the bar, if it does not already.
    n_required = None
    if verdict == "INDETERMINATE-MORE-N" and eta_new > FO1_BREAKEVEN_ETA:
        sd_n = boot["sd"] * np.sqrt(len(new_rows))       # sd scales ~1/sqrt(n)
        need = (sd_n / (eta_new - FO1_BREAKEVEN_ETA)) ** 2
        n_required = int(np.ceil(need))

    flips = 6512.0
    joint = {}
    for tag, e in (("new_only", eta_new), ("pooled_60", eta_pooled_60),
                   ("pn2_n12", PN2_ETA_PROJECTED_N12), ("lower_1sigma", lower_1sigma)):
        joint[tag] = {"eta": e, "net_dS_at_fo1_bytes": net_dS(e, flips, FO1_TOTAL_B),
                      "share_of_gap_closed": -net_dS(e, flips, FO1_TOTAL_B) / GAP_S}

    pose = {"aggregate_ratio_new": pose_agg_ratio(new_rows),
            "pairs_pose_improved_new": int(sum(1 for r in new_rows
                                               if (r.get("d_pose_ratio") or 9e9) < 1.0)),
            "n_new": len(new_rows),
            "convention": "mean(d_pose_after)/mean(d_pose_before), the evaluate.py aggregation",
            "delta_S_pose": ((10.0 * D_POSE_N600 * pose_agg_ratio(new_rows)) ** 0.5
                             - (10.0 * D_POSE_N600) ** 0.5)}

    matched = {}
    free_rows = load_rows(work / "free_matched16" / "ETA_GATE_ROWS.jsonl")
    if free_rows:
        fp = {r["pair"] for r in free_rows}
        null_sub = [r for r in new_rows if r["pair"] in fp]
        common = sorted(fp & {r["pair"] for r in null_sub})
        cset = set(common)
        fr = [r for r in free_rows if r["pair"] in cset]
        nr = [r for r in null_sub if r["pair"] in cset]
        if common:
            e_f, e_n = pooled_eta(fr), pooled_eta(nr)
            per = {p: (next(x["eta_net"] for x in nr if x["pair"] == p)
                       - next(x["eta_net"] for x in fr if x["pair"] == p)) for p in common}
            matched = {
                "n_matched": len(common), "pairs": common,
                "pooled_eta_unprojected": e_f, "pooled_eta_projected": e_n,
                "delta_eta": e_n - e_f,
                "pct_change": 100.0 * (e_n - e_f) / e_f if e_f else None,
                "pairs_projection_raised_eta": int(sum(1 for v in per.values() if v > 0)),
                "pose_ratio_unprojected": pose_agg_ratio(fr),
                "pose_ratio_projected": pose_agg_ratio(nr),
                "per_pair_delta_eta": per,
                "pn2_n12_reference": {"unprojected": PN2_ETA_UNPROJECTED_N12,
                                      "projected": PN2_ETA_PROJECTED_N12},
                # pn2's regression series (+15.6% -> +14.2% -> +11.7% -> +8.1% at n=4/7/10/12)
                # was in THIS quantity -- the null-vs-free ADVANTAGE -- not in eta itself.  A
                # cumulative curve of eta alone would answer a different question than the one
                # pn2 left open, so the advantage gets its own curve on the new sample.
                "pn2_advantage_regression_series_pct": [15.6, 14.2, 11.7, 8.1],
                "cumulative_advantage_pct_shuffled": matched_advantage_curve(
                    fr, nr, common, args.boot_seed),
                "advantage_floor_note":
                    "even if the advantage regressed to ZERO, the projected eta would fall back "
                    "to the UNPROJECTED level, which pn2 measured at 0.5651 -- still above the "
                    "0.5196 bar.  The advantage regressing is therefore not by itself a threat "
                    "to the supplier verdict; it would be a threat to the MECHANISM claim.",
            }

    rec = {
        "schema": "ddm_fo2h_eta_adjudication.v1",
        "axis": "[macOS-CPU advisory] frozen CPU-torch SegNet+PoseNet -- NEVER a score",
        "score_claim": False, "promotable": False, "pointer_moved": False,
        "frozen_bar": FO1_BREAKEVEN_ETA,
        "bar_provenance": "fo1 s5 break-even on its measured round-trip-verified 4,308 B; "
                          "FROZEN before this arm ran and NOT recomputed from this arm's LEG 2",
        "sampling": {
            "n_new": len(new_rows),
            "pairs_new": [r["pair"] for r in new_rows],
            "disjoint_from_pn2_n12": True,
            "pn2_pairs_excluded": list(PN2_PAIRS),
            "method": "seeded random choice over the population with pn2's 12 removed "
                      "(m96: never a [:n] prefix; m88: never a scene block)",
            "shard_n": {k: len(v) for k, v in shards.items()},
        },
        "eta": {
            "pooled_new_out_of_sample": eta_new,
            "pooled_with_pn2_n60": eta_pooled_60,
            "per_pair_mean_new": float(np.mean([r["eta_net"] for r in new_rows])),
            "per_pair_sd_new": float(np.std([r["eta_net"] for r in new_rows], ddof=1))
            if len(new_rows) > 1 else None,
            "per_pair_min": float(min(r["eta_net"] for r in new_rows)),
            "per_pair_max": float(max(r["eta_net"] for r in new_rows)),
            "pn2_n12_projected": PN2_ETA_PROJECTED_N12,
            "estimator_note": "pooled = sum(before-after)/sum(described), the ratio the S "
                              "arithmetic uses; the per-pair mean is reported but does not govern",
            "pooled_60_caveat":
                "the n=60 figure pools THIS arm's rows with rt1's retained eta_gate_null rows. "
                "Solver config matches exactly (steps 30, lr 6.0, eval_every 2, focus_weight "
                "500, radius 1, starts 2, mode null), but rt1's receipt does NOT record its "
                "torch thread count, and thread count is part of the forward instrument (et4: "
                "reduction order can flip argmax ties). This arm ran threads=4, the tool's "
                "default. The out-of-sample n=48 figure is the PRIMARY number precisely "
                "because it carries no such assumption.",
        },
        "spread": {"falsifier_band_law": band, "bootstrap_new": boot,
                   "bootstrap_pooled_60": boot60,
                   "governing_lower_1sigma": lower_1sigma,
                   "governing_lower_2sigma": lower_2sigma,
                   "governing_rule": "the WIDER of the two estimators governs -- a spread "
                                     "estimate that is itself uncertain may not license a verdict"},
        "verdict": verdict,
        "n_required_for_1sigma_clearance": n_required,
        "joint_arithmetic_at_fo1_bytes": joint,
        "pose_leg": pose,
        "matched_AB_new_sample": matched,
        "cumulative_eta_curve_shuffled": cumulative_curve(new_rows, args.boot_seed),
        "verdict_scope": "INSTANCE: hv1 ep0634 base, ring-0 described set, r=1 support, "
                         "pose-null-constrained realization, this solver budget, n as reported; "
                         "m96: a seeded-random sample may REFUTE a bar; clearing one does not "
                         "license a LIVE n600 verdict",
        "rows_new": new_rows,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (work / "FO2H_ETA_ADJUDICATION.json").write_text(
        json.dumps(rec, indent=2, sort_keys=True) + "\n")
    progress(work, "leg1-adjudicated", {
        "n_new": len(new_rows), "eta_new": eta_new, "eta_pooled_60": eta_pooled_60,
        "lower_1sigma": lower_1sigma, "bar": FO1_BREAKEVEN_ETA, "verdict": verdict})
    print(json.dumps({k: v for k, v in rec.items() if k != "rows_new"}, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--shards", nargs="+",
                    default=["null_shardA", "null_shardB"],
                    help="out dirs holding the PROJECTED arm's rows")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--boot-seed", type=int, default=20260817)
    return adjudicate(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
