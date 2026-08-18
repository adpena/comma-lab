"""ddm_sa3 -- does the frame-0 actuator have the authority to cancel keep01's collapse?

THE QUESTION, MADE SHARP
------------------------
sa2 cancelled 99.9417% of the S2 semantic edit's pose damage and admitted.  sa2's own
memo then named the ladder as the real prize and flagged the honest risk: keep01 is a
**26x** pose collapse where S2 was 6.7x, "a different and much harder ask, and it should
be measured, not assumed".  This module measures it.

Derived against the LIVE sz1 pointer (S 0.15771357797660338 @ 179,930 B), the admit
arithmetic gives:

    row      damage (CPU d_pose)   damage vs S2   required cancellation
    S2            8.3908e-4            1.00x          99.9261%
    keep01        3.7265e-3            4.44x          99.9479%

So keep01 needs MORE cancellation than has ever been achieved (99.9417%) against 4.44x
MORE damage.  Both moves are adverse.  There is no reason to expect it to clear, and
equally no reason to assume it cannot -- 236 of sa2's 600 pairs ended *better* than base,
so the actuator was not saturated on S2.  Only a solve answers it.

WHAT IS AND IS NOT REDUCED
--------------------------
SCOPE is reduced: a seeded-RANDOM subset of pairs, never a prefix (prefix bias on the
pose axis measures 2.5-4.2x HARDER than the population, which would make a NO-GO here
exactly the false-negative shape).  A representativeness control reports the subset's
uncompensated damage against the n600 measured damage so the reader can see whether the
subset is a credible estimator.

MECHANISM is NOT reduced: the identical solver, the identical exact receiver realization
at full resolution, the identical frozen CPU-torch PoseNet, the identical descent ladder.
This is sa2's instrument pointed at a different edited frame_1.

AXIS ``[macOS-CPU advisory, frozen CPU-torch PoseNet]`` -- score_claim=false.  Not a
score, and it moves no pointer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_sa2_compensated_semantic_edit import (
    CAMERA_H,
    CAMERA_W,
    PAIR_COUNT,
    RR4Frame0Surface,
    decode_gt_pairs,
    load_posenet,
    solve_pair,
)

SA1: Final = Path("/Volumes/APDataStore/pact/ddm_sa1")
ADVISORY: Final = SA1 / "advisory_n600_cpu"
BASE_RAW: Final = ADVISORY / "rr4_base/attempt_0002/work/inflated/0.raw"

# --- the candidate under test, with its MEASURED n600 advisory row -------------------
ROWS: Final = {
    "sm3r_keep01": {
        "raw": ADVISORY / "sm3r_keep01/attempt_0006/work/inflated/0.raw",
        "archive_bytes": 178_272,
        "d_pose_n600": 3.87399e-3,
        "d_seg_delta_n600": 4.77e-6,
        "receipt": ADVISORY / "sm3r_keep01/attempt_0006/contest_auth_eval.json",
    },
    "sm3r_keep87": {
        "raw": ADVISORY / "sm3r_keep87/attempt_0001/work/inflated/0.raw",
        "archive_bytes": 181_031,
        "d_pose_n600": None,
        "d_seg_delta_n600": None,
        "receipt": ADVISORY / "sm3r_keep87/attempt_0001/contest_auth_eval.json",
    },
}

D_POSE_BASE_CPU: Final = 1.474661e-04
SZ1_SCORE: Final = 0.15771357797660338
SZ1_POSE_S: Final = 0.008294576541331089
SZ1_D_POSE: Final = (SZ1_POSE_S**2) / 10.0
SZ1_ARCHIVE_BYTES: Final = 179_930
FX2_TAIL_CREDIT: Final = -711  # measured: sz1's token tail is edit-independent
RATE_DENOMINATOR: Final = 37_545_489
ADMIT_BAR_S: Final = -3.5e-6
AXIS: Final = "[macOS-CPU advisory, frozen CPU-torch PoseNet]"


class SA3AuthorityError(RuntimeError):
    """An authority-probe precondition failed."""


def required_cancellation(archive_bytes: int, d_seg_delta: float, damage: float) -> dict[str, Any]:
    """How much of the damage must be cancelled for this row to clear the bar."""
    delta_bytes = archive_bytes + FX2_TAIL_CREDIT - SZ1_ARCHIVE_BYTES
    rate_s = 25.0 * delta_bytes / RATE_DENOMINATOR
    seg_s = 100.0 * d_seg_delta
    budget = -(rate_s + seg_s) + ADMIT_BAR_S
    allowed_d_pose = ((SZ1_POSE_S + budget) ** 2) / 10.0
    max_residual = allowed_d_pose - SZ1_D_POSE
    return {
        "delta_bytes_vs_sz1": delta_bytes,
        "rate_s": rate_s,
        "seg_s": seg_s,
        "pose_budget_s": budget,
        "max_residual_d_pose": max_residual,
        "uncompensated_damage": damage,
        "required_cancellation": 1.0 - max_residual / damage,
    }


def load_raw(path: Path) -> np.memmap:
    if not path.is_file():
        raise SA3AuthorityError(f"inflated raw missing: {path}")
    expected = PAIR_COUNT * 2 * CAMERA_H * CAMERA_W * 3
    if path.stat().st_size != expected:
        raise SA3AuthorityError(
            f"raw {path} is {path.stat().st_size} B, expected {expected}"
        )
    return np.memmap(
        path, dtype=np.uint8, mode="r", shape=(PAIR_COUNT * 2, CAMERA_H, CAMERA_W, 3)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", default="sm3r_keep01", choices=sorted(ROWS))
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--out", type=Path, default=Path("/Volumes/APDataStore/pact/ddm_sa3/keep01_authority")
    )
    args = parser.parse_args(argv)

    row = ROWS[args.row]
    if row["d_pose_n600"] is None:
        raise SA3AuthorityError(f"{args.row} has no measured n600 pose row to price against")

    damage_n600 = row["d_pose_n600"] - D_POSE_BASE_CPU
    need = required_cancellation(
        row["archive_bytes"], row["d_seg_delta_n600"], damage_n600
    )
    print(f"row                     : {args.row}")
    print(f"delta bytes vs sz1      : {need['delta_bytes_vs_sz1']:+d}")
    print(f"rate credit             : {need['rate_s']:+.6e} S")
    print(f"seg cost                : {need['seg_s']:+.6e} S")
    print(f"uncompensated damage    : {damage_n600:.6e} d_pose  "
          f"({damage_n600 / (9.865438e-4 - D_POSE_BASE_CPU):.2f}x S2's)")
    print(f"max residual allowed    : {need['max_residual_d_pose']:.6e} d_pose")
    print(f"REQUIRED cancellation   : {need['required_cancellation'] * 100:.4f}%")
    print("sa2 ACHIEVED on S2      : 99.9417%\n")

    base_raw, edited_raw = load_raw(BASE_RAW), load_raw(row["raw"])

    rng = np.random.default_rng(args.seed)
    pairs = np.sort(rng.choice(PAIR_COUNT, size=args.pairs, replace=False))
    print(f"seeded-RANDOM pairs (never a prefix): {pairs.tolist()}\n")

    surface = RR4Frame0Surface.load()
    posenet = load_posenet()
    gt = decode_gt_pairs([int(p) for p in pairs])

    out = args.out / args.row
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, pair in enumerate(pairs):
        pair = int(pair)
        root = out / f"pair_{pair:04d}"
        root.mkdir(parents=True, exist_ok=True)
        record = solve_pair(
            surface=surface,
            posenet=posenet,
            pair=pair,
            master_base=np.asarray(base_raw[2 * pair + 1]),
            master_edited=np.asarray(edited_raw[2 * pair + 1]),
            gt_pair=gt[index],
            root=root,
        )
        results.append(record)
        dp = record["pose"]
        print(
            f"  pair {pair:3d}  base {dp['d_pose_pair_base']:.6e}  "
            f"event {dp['d_pose_pair_event_uncompensated']:.6e}  "
            f"final {dp['d_pose_pair_compensated']:.6e}  "
            f"cancel {dp['d_pose_damage_cancellation_fraction'] * 100:7.3f}%  "
            f"({time.perf_counter() - started:.0f}s)"
        )

    base = np.array([r["pose"]["d_pose_pair_base"] for r in results], dtype=np.float64)
    event = np.array(
        [r["pose"]["d_pose_pair_event_uncompensated"] for r in results], dtype=np.float64
    )
    final = np.array([r["pose"]["d_pose_pair_compensated"] for r in results], dtype=np.float64)
    subset_damage = float((event - base).mean())
    subset_residual = float((final - base).mean())
    cancellation = 1.0 - subset_residual / subset_damage

    # Price the subset residual through the SAME admit arithmetic, scaled by the
    # representativeness ratio so the estimator's own bias is visible.
    ratio = subset_damage / damage_n600
    projected_residual = subset_residual / ratio if ratio else float("nan")
    pose_s = ((10.0 * (SZ1_D_POSE + projected_residual)) ** 0.5) - (
        (10.0 * SZ1_D_POSE) ** 0.5
    )
    net = need["rate_s"] + need["seg_s"] + pose_s

    summary = {
        "schema": "ddm_sa3_actuator_authority.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "row": args.row,
        "seed": args.seed,
        "pairs": [int(p) for p in pairs],
        "n_pairs": len(pairs),
        "requirement": need,
        "subset": {
            "mean_base_d_pose": float(base.mean()),
            "mean_uncompensated_d_pose": float(event.mean()),
            "mean_compensated_d_pose": float(final.mean()),
            "mean_damage": subset_damage,
            "mean_residual": subset_residual,
            "cancellation": cancellation,
            "pairs_better_than_base": int((final < base).sum()),
        },
        "representativeness": {
            "n600_damage": damage_n600,
            "subset_damage": subset_damage,
            "subset_over_n600": ratio,
            "note": (
                "A ratio far from 1.0 means the subset is not a credible estimator of "
                "the population mean and the projection below must not be believed."
            ),
        },
        "projection": {
            "scaled_residual_d_pose": projected_residual,
            "pose_s": pose_s,
            "net_delta_s": net,
            "projected_score": SZ1_SCORE + net,
            "admits": bool(net < ADMIT_BAR_S),
            "admit_bar_s": ADMIT_BAR_S,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out / "AUTHORITY.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"\n=== {args.row}: n={len(pairs)} seeded-random pairs ===")
    print(f"subset damage / n600 damage : {ratio:.4f}  (representativeness control)")
    print(f"ACHIEVED cancellation       : {cancellation * 100:.4f}%")
    print(f"REQUIRED cancellation       : {need['required_cancellation'] * 100:.4f}%")
    print(f"pairs better than base      : {summary['subset']['pairs_better_than_base']} of {len(pairs)}")
    print(f"projected net delta S       : {net:+.6e}  -> S {SZ1_SCORE + net:.9f}")
    print(f"admits (bar {ADMIT_BAR_S:+.1e})     : {summary['projection']['admits']}")
    print(f"\nreport -> {out / 'AUTHORITY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
