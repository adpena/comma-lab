#!/usr/bin/env python3
"""ddm_ps2 -- compose the THIRD term fo2h measured but never added to its verdict.

WHAT THIS PAYS.  `fo2h` hardened the projected seg-correction channel's eta on an n=48
out-of-sample seeded-random sample and returned SUPPLIER CONFIRMED-HARDENED.  That verdict is
adjudicated by `net_dS(eta, flips, total_B) = -eta*flips*SEG_DS_PER_FLIP + total_B*RATE_DS_PER_BYTE`
-- seg gain plus rate cost, and NOTHING ELSE.  The same run measured a pose leg
(`aggregate_ratio_new` 1.3725, `delta_S_pose` +0.00142) and reported it BESIDE the verdict rather
than inside it.  The two numbers have never been added together in any document.

They must be added, because the channel edits pixels that PoseNet reads: total
    dS = dS_seg + dS_rate + dS_pose
and the pose term at n=48 is 4.2x the size of the seg+rate term with the opposite sign.

WHAT THIS FIXES IN THE FRAMING.  `na9` F2 reports the projection as removing "0.010423 S of pose
cost ... 1.09x the entire remaining gap", from pn2's matched A/B (unprojected x4.6089 vs projected
x0.7935).  That is a delta against the UNPROJECTED arm.  The shipping baseline is not the
unprojected edit -- it is NO EDIT, whose pose cost is exactly zero.  Measured against the shipping
baseline the projected arm is a pose COST, not a saving, whenever its ratio exceeds 1.0.  This is
`a_delta_without_its_baseline_is_unanchored_and_baselines_move_20260803` on the pose axis.

DISCIPLINE THIS ADDS.  fo2h bootstrapped the eta leg (20,000 resamples, shard sigma, refuse on
straddle) and applied NO spread estimator to the pose leg, which is now the binding term.  The
pose aggregate is a ratio of means over a heavy-tailed sample (fo2h's own concentration note: one
pair owns 33.5% of the excess), so a point estimate cannot license a verdict.  Same estimator,
same discipline, applied to the term that binds.

RE-DERIVE, DON'T CONFIRM.  Every fo2h figure this module cites is recomputed from the retained
per-pair rows and asserted against the published value; a mismatch is fail-closed.

NO SCORER IS RUN.  This is arithmetic over rows already on disk: $0, no Modal, no Metal, no torch.

Axis `[macOS-CPU advisory]` -- NEVER a score.  `score_claim=false`.  `promotable=false`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np

# --- FROZEN pins, cited from the stores, never recomputed here --------------------------------
# fo2h's own constants, reused verbatim so the control assertions are meaningful.
FO1_BREAKEVEN_ETA = 0.5196321126365346   # fo1 s5 break-even on its measured 4,308 B
FO1_TOTAL_B = 4308.0                     # fo1's measured round-trip-verified bytes
SEG_DS_PER_FLIP = 100.0 / (600 * 384 * 512)
RATE_DS_PER_BYTE = 25.0 / 37_545_489
D_POSE_N600 = 6.885643e-06               # hv1 contest-CUDA n600 pose term (pn2 s5, via fo2h)
# The channel describes 6,512 seg flips CLIP-WIDE (fo1/sr1 41-cell selection, fo2h LEG 2 table).
# eta is the fraction of THOSE it fixes, so the seg gain is eta*6512 -- never the flip count of
# whatever sample eta was estimated on.  Conflating the two inflates the seg gain ~4.5x.
FO1_DESCRIBED_FLIPS = 6512.0

# The LIVE object this arm is chartered against (ddm_ps2 charter, 2026-08-18).
FX1_S = 0.15816036933414834              # fx1 frontier, archive sha 65c75d7f...
FX1_GAP_S = FX1_S - 0.15                 # 0.00816036933414834
FX1_POSE_CONTRIB = (10.0 * D_POSE_N600) ** 0.5   # sqrt(10*d_pose); charter quotes 0.008295

# fo2h published figures -- asserted, not trusted.
FO2H_PUBLISHED = {
    "pooled_eta_n48": 0.5804404628592759,
    "pose_ratio_n48": 1.3725406813262484,
    "delta_S_pose_n48": 0.0014235579773669155,
    "net_dS_n48": -0.00033567977621332037,
    "matched_eta_projected_n16": 0.5673758865248227,
    "matched_eta_unprojected_n16": 0.541033434650456,
    "matched_pose_projected_n16": 1.5663033764141072,
    "matched_pose_unprojected_n16": 6.9563141262017885,
}

FO2H_WORK = Path("/Volumes/APDataStore/pact/ddm_fo2h_eta_hardening")
DEFAULT_OUT = Path("/Volumes/APDataStore/pact/ddm_ps2")
TOL = 1e-12


class Ps2Error(RuntimeError):
    """Fail-closed error."""


def progress(out: Path, milestone: str, detail: dict) -> None:
    row = {"arm": "ddm_ps2", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "milestone": milestone, "detail": detail, "pid": os.getpid(),
           "host": socket.gethostname()}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "PROGRESS.jsonl").open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[ps2] {milestone}: {json.dumps(detail, sort_keys=True)}", flush=True)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise Ps2Error(f"required retained rows missing: {path}")
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def dedup_by_pair(rows: list[dict]) -> list[dict]:
    seen: set[int] = set()
    out: list[dict] = []
    for r in rows:
        if r["pair"] not in seen:
            seen.add(r["pair"])
            out.append(r)
    return out


def pooled_eta(rows: list[dict]) -> float:
    """sum(before-after)/sum(described) -- the ratio the S arithmetic uses (fo2h's estimator)."""
    b = sum(r["flips_before"] for r in rows)
    a = sum(r["flips_after"] for r in rows)
    d = sum(r["n_described_ring0"] for r in rows)
    return (b - a) / d if d else float("nan")


def pose_agg_ratio(rows: list[dict]) -> float:
    """mean(after)/mean(before) -- evaluate.py aggregates d_pose as a mean over pairs and only
    then takes the sqrt, so the aggregate ratio is a ratio of means, never a mean of ratios."""
    db = np.array([r["d_pose_before"] for r in rows], dtype=np.float64)
    da = np.array([r["d_pose_after"] for r in rows], dtype=np.float64)
    m = db.mean()
    return float(da.mean() / m) if m else float("nan")


def dS_pose(ratio: float) -> float:
    """The pose term moves as sqrt: dS = sqrt(10*d_pose*ratio) - sqrt(10*d_pose)."""
    return float((10.0 * D_POSE_N600 * ratio) ** 0.5 - (10.0 * D_POSE_N600) ** 0.5)


def net_dS_seg_rate(eta: float, flips: float, total_B: float) -> float:
    """fo2h's verdict arithmetic, reproduced exactly: seg gain + rate cost, no pose."""
    return -eta * flips * SEG_DS_PER_FLIP + total_B * RATE_DS_PER_BYTE


def pose_ratio_break_even(seg_rate_dS: float) -> float:
    """The pose ratio at which the channel exactly breaks even once pose is counted.

    Solve dS_pose(r) + seg_rate_dS = 0 for r:
        sqrt(10*d*r) = sqrt(10*d) - seg_rate_dS      (seg_rate_dS is negative when supplying)
        r = (1 - seg_rate_dS/sqrt(10*d))^2
    """
    p0 = (10.0 * D_POSE_N600) ** 0.5
    root = 1.0 - seg_rate_dS / p0
    return float(root * root) if root > 0 else 0.0


def bootstrap_pose(rows: list[dict], n_boot: int, seed: int) -> dict:
    """Pair-level bootstrap of the pose ratio-of-means, matching fo2h's eta bootstrap protocol.

    The pose aggregate is a ratio of means over a heavy-tailed sample, so resampling pairs is the
    estimator whose spread governs -- exactly the discipline fo2h applied to eta and not to pose.
    """
    db = np.array([r["d_pose_before"] for r in rows], dtype=np.float64)
    da = np.array([r["d_pose_after"] for r in rows], dtype=np.float64)
    n = len(rows)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ratios = da[idx].mean(axis=1) / db[idx].mean(axis=1)
    return {"n_boot": n_boot, "seed": seed, "sd": float(ratios.std(ddof=1)),
            "p02_5": float(np.percentile(ratios, 2.5)),
            "p16": float(np.percentile(ratios, 16.0)),
            "p50": float(np.percentile(ratios, 50.0)),
            "p84": float(np.percentile(ratios, 84.0)),
            "p97_5": float(np.percentile(ratios, 97.5)),
            "_draws": ratios}


def cumulative_curves(rows: list[dict], seed: int) -> list[dict]:
    """Cumulative pooled eta AND pose ratio vs n on ONE seeded shuffle.

    A cumulative curve on the natural pair order would be a scene-block artifact (m88); the
    shuffle is what makes the n-axis mean sample size rather than scene position.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    out = []
    for k in range(1, len(rows) + 1):
        sub = [rows[i] for i in order[:k]]
        r = pose_agg_ratio(sub)
        out.append({"n": k, "pooled_eta": pooled_eta(sub), "pose_ratio": r,
                    "delta_S_pose": dS_pose(r)})
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_close(name: str, got: float, want: float) -> None:
    if abs(got - want) > TOL:
        raise Ps2Error(f"control FAILED: {name} re-derived {got!r} != published {want!r}")


def adjudicate(args: argparse.Namespace) -> int:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    null_rows = dedup_by_pair(
        load_rows(args.fo2h / "null_shardA" / "ETA_GATE_ROWS.jsonl")
        + load_rows(args.fo2h / "null_shardB" / "ETA_GATE_ROWS.jsonl"))
    free_rows = dedup_by_pair(load_rows(args.fo2h / "free_matched16" / "ETA_GATE_ROWS.jsonl"))
    if len(null_rows) != 48:
        raise Ps2Error(f"expected 48 projected rows, got {len(null_rows)}")
    if len(free_rows) != 16:
        raise Ps2Error(f"expected 16 matched unprojected rows, got {len(free_rows)}")

    # --- CONTROLS: re-derive every fo2h figure this arm leans on -----------------------------
    eta48 = pooled_eta(null_rows)
    pose48 = pose_agg_ratio(null_rows)
    seg_rate48 = net_dS_seg_rate(eta48, FO1_DESCRIBED_FLIPS, FO1_TOTAL_B)

    matched_pairs = sorted(r["pair"] for r in free_rows)
    null_matched = [r for r in null_rows if r["pair"] in set(matched_pairs)]
    if sorted(r["pair"] for r in null_matched) != matched_pairs:
        raise Ps2Error("matched A/B pair sets differ between arms")

    # Every published figure gets a re-derivation bound to it HERE, so a key cannot be reported
    # as "re-derived" without an assertion actually running -- the VACUITY==PASS cure applied to
    # this arm's own control (an earlier draft listed net_dS_n48 as checked while asserting
    # nothing, and that hid a wrong flip-count constant).
    rederived = {
        "pooled_eta_n48": eta48,
        "pose_ratio_n48": pose48,
        "delta_S_pose_n48": dS_pose(pose48),
        "net_dS_n48": seg_rate48,
        "matched_eta_projected_n16": pooled_eta(null_matched),
        "matched_eta_unprojected_n16": pooled_eta(free_rows),
        "matched_pose_projected_n16": pose_agg_ratio(null_matched),
        "matched_pose_unprojected_n16": pose_agg_ratio(free_rows),
    }
    if set(rederived) != set(FO2H_PUBLISHED):
        raise Ps2Error(f"control coverage gap: {set(FO2H_PUBLISHED) ^ set(rederived)}")
    for name, got in sorted(rederived.items()):
        _assert_close(name, got, FO2H_PUBLISHED[name])
    progress(out, "controls_passed", {"n_null": len(null_rows), "n_free": len(free_rows),
                                      "asserted": sorted(rederived)})

    # --- THE COMPOSITION fo2h never performed -------------------------------------------------
    joint48 = seg_rate48 + dS_pose(pose48)
    boot = bootstrap_pose(null_rows, args.n_boot, args.seed)
    draws = boot.pop("_draws")
    joint_draws = seg_rate48 + np.array([dS_pose(float(r)) for r in draws])
    r_be = pose_ratio_break_even(seg_rate48)

    # matched A/B, both arms priced on the SHIPPING baseline (no edit), not on each other
    mp, mu = (FO2H_PUBLISHED["matched_pose_projected_n16"],
              FO2H_PUBLISHED["matched_pose_unprojected_n16"])
    matched = {
        "n": 16, "pairs": matched_pairs,
        "eta_projected": pooled_eta(null_matched), "eta_unprojected": pooled_eta(free_rows),
        "eta_advantage_pct": 100.0 * (pooled_eta(null_matched) - pooled_eta(free_rows))
                             / pooled_eta(free_rows),
        "pose_ratio_projected": mp, "pose_ratio_unprojected": mu,
        "dS_pose_projected_vs_ship": dS_pose(mp),
        "dS_pose_unprojected_vs_ship": dS_pose(mu),
        "pose_cost_removed_by_projection": dS_pose(mu) - dS_pose(mp),
        "residual_pose_cost_after_projection": dS_pose(mp),
    }

    verdict = ("F2 REFUTED-ON-THE-JOINT-AXIS" if joint48 > 0 else "F2 NET SUPPLIER")
    res = {
        "schema": "ddm_ps2_f2_joint_adjudication.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU advisory] -- arithmetic over fo2h retained rows; NEVER a score",
        "score_claim": False, "promotable": False, "pointer_moved": False,
        "provenance": {
            "rows_from": str(args.fo2h),
            "no_scorer_run": True,
            "sampling_law": "INHERITED from fo2h: seeded-random over the population with pn2's 12 "
                            "removed; disjoint from pn2 by construction; never a [:n] prefix "
                            "(m96) and never a scene block (m88). This arm re-uses that sample "
                            "and draws no new one.",
            "controls": "every fo2h figure below was re-derived from the retained per-pair rows "
                        "and asserted equal to the published value (fail-closed)",
        },
        "n48_projected_arm": {
            "n": 48, "pooled_eta": eta48, "fo1_break_even_eta": FO1_BREAKEVEN_ETA,
            "eta_clears_bar": bool(eta48 > FO1_BREAKEVEN_ETA),
            "dS_seg_plus_rate": seg_rate48,
            "pose_ratio": pose48, "dS_pose": dS_pose(pose48),
            "dS_joint": joint48,
            "pose_cost_over_seg_gain": abs(dS_pose(pose48) / seg_rate48),
            "share_of_fx1_gap": -joint48 / FX1_GAP_S,
        },
        "pose_spread": {
            "note": "fo2h bootstrapped eta and applied NO spread estimator to pose. Pose is the "
                    "binding term once composed, so the same discipline is applied to it here.",
            "bootstrap_pose_ratio": boot,
            "joint_dS_p02_5": float(np.percentile(joint_draws, 2.5)),
            "joint_dS_p16": float(np.percentile(joint_draws, 16.0)),
            "joint_dS_p50": float(np.percentile(joint_draws, 50.0)),
            "joint_dS_p84": float(np.percentile(joint_draws, 84.0)),
            "joint_dS_p97_5": float(np.percentile(joint_draws, 97.5)),
            "frac_draws_net_supplier": float((joint_draws < 0).mean()),
            "pose_ratio_break_even": r_be,
            "measured_ratio_over_break_even": pose48 / r_be,
            "frac_draws_pose_ratio_below_break_even": float((draws < r_be).mean()),
        },
        "matched_AB_n16_priced_on_shipping_baseline": matched,
        "cumulative_curves_projected_arm": cumulative_curves(null_rows, args.seed),
        "verdict": verdict,
        "verdict_scope": (
            "FORMULATION: post-hoc pose-null-projected seg-correction overlays on the frozen "
            "hv1/rr4-lineage artifact, ring-0 described set, r=1 support, this solver budget, "
            "n=48 out-of-sample. NOT a FAMILY verdict on projection as a mechanism -- the "
            "matched A/B confirms the mechanism removes most of the pose cost; it does not "
            "remove enough. m96: a seeded-random sample may REFUTE a bar; clearing one does not "
            "license a LIVE n600 verdict."),
    }
    path = out / "PS2_F2_JOINT_ADJUDICATION.json"
    path.write_text(json.dumps(res, indent=1, sort_keys=True))
    progress(out, "adjudicated", {"verdict": verdict, "dS_joint": joint48,
                                  "sha256": sha256_of(path)})
    print(json.dumps({k: res[k] for k in ("verdict", "n48_projected_arm")}, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fo2h", type=Path, default=FO2H_WORK,
                   help="fo2h retained work root holding the eta-gate rows")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="ddm_ps2 payload root")
    p.add_argument("--n-boot", type=int, default=20000,
                   help="bootstrap resamples (fo2h used 20000 on eta)")
    p.add_argument("--seed", type=int, default=20260818)
    return adjudicate(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
