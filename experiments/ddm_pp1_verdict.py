#!/usr/bin/env python3
"""ddm_pp1: assemble the arm's verdict from its own measured receipts.

Reads only the JSON/JSONL this arm wrote and recomputes every score-unit number from
components (never a stored headline), so the memo's tables and the receipts cannot
drift.  Scorer-free: it does no measurement of its own.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import collections
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

#: MOVE 49, the base this arm stands on.  Recomputed, never trusted as a headline.
POINTER_BYTES = 179_153
POINTER_D_SEG_T4 = 0.00010287
POINTER_D_POSE_T4 = 4.55e-06
RATE_DENOMINATOR = 37_545_489
CELLS_TOTAL = 600 * 384 * 512
S_PER_CELL = 100.0 / CELLS_TOTAL
S_PER_BYTE = 25.0 / RATE_DENOMINATOR
N_PAIRS = 600
ADMIT_BAR = -2e-5


def pointer_score() -> float:
    return (
        100.0 * POINTER_D_SEG_T4
        + math.sqrt(10.0 * POINTER_D_POSE_T4)
        + 25.0 * POINTER_BYTES / RATE_DENOMINATOR
    )


def read_jsonl(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="/Volumes/APDataStore/pact/ddm_pp1")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    store = Path(args.store)

    base = json.loads((store / "base/POSE_BASE_MOVE49.json").read_text())
    per_pair = np.load(store / "base/pose_base_move49.npy")
    controls = json.loads((store / "base/CONTROLS.json").read_text())
    tolerance = json.loads((store / "base/BASE_TOLERANCE.json").read_text())
    fee = json.loads((store / "base/RATE_FEE.json").read_text())

    screen = read_jsonl(str(store / "segscreen/seg_screen_*.jsonl"))
    refine = read_jsonl(str(store / "refine/refine_moves_*.jsonl"))
    floor = read_jsonl(str(store / "floor/floor_rows_*.jsonl"))
    search_b = read_jsonl(str(store / "searchB/search_b_*.jsonl"))

    mass = float(per_pair.sum())
    order = np.argsort(-per_pair)
    top12 = [int(p) for p in order[:12]]
    pose_marginal_per_pair = 5.0 / math.sqrt(10.0 * float(per_pair.mean())) / N_PAIRS

    by_pair = collections.defaultdict(list)
    for row in screen:
        by_pair[row["pair"]].append(row)
    neutral = [row for row in screen if row["d_cells"] <= 0]

    fee_n1 = fee["summary"]["1"]
    rate_s_n1 = fee_n1["shipped_mean"] * S_PER_BYTE
    # The economic break-even the ARM must clear, derived at THIS move: a seg-neutral
    # candidate pays only the container fee, so the pose credit must cover the bar plus
    # that fee.
    required_sum_d_pose = (abs(ADMIT_BAR) + rate_s_n1) / pose_marginal_per_pair
    top12_mass = float(per_pair[order[:12]].sum())

    verdict: dict[str, Any] = {
        "schema": "ddm_pp1_verdict.v1",
        "axis": (
            "d_pose [macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT, n600]; "
            "d_seg [frozen CPU-torch SegNet argmax, DALI GT]; bytes EXACT through the "
            "shipped SM1S mixer and the shipped Brotli container"
        ),
        "score_claim": False,
        "pointer": {
            "archive_bytes": POINTER_BYTES,
            "d_seg_t4": POINTER_D_SEG_T4,
            "d_pose_t4": POINTER_D_POSE_T4,
            "score_recomputed": pointer_score(),
        },
        "base": {
            "d_pose_mean": base["d_pose_mean"],
            "instrument_ratio_vs_t4_print": base["instrument_ratio_vs_t4_print"],
            "d_pose_median": base["d_pose_median"],
            "d_pose_max": base["d_pose_max"],
            "max_over_median": base["d_pose_max"] / base["d_pose_median"],
            "top12_pairs": top12,
            "top12_share": top12_mass / mass,
            "pose_marginal_S_per_unit_pair_d_pose": pose_marginal_per_pair,
        },
        "controls": {
            "all_pass": controls["all_pass"],
            "rows": [
                {k: row[k] for k in row if k != "population"} for row in controls["rows"]
            ],
        },
        "base_tolerance": {
            "batch1_repeat_exact_all": tolerance["batch1_repeat_exact_all"],
            "max_rel_gap_on_treated_pairs": max(
                row["gap_rel"] for row in tolerance["rows"]
                if row["n600_batch8"] >= 4.6e-5
            ),
            "max_abs_gap": max(
                abs(row["batch1_first"] - row["n600_batch8"]) for row in tolerance["rows"]
            ),
            "gate_used": 3.29e-4,
        },
        "rate_fee": {
            "base_member_bytes": fee["base_member_bytes"],
            "container_shape": fee["container_shape"],
            "container_search_run": False,
            "summary": fee["summary"],
        },
        "seg_screen": {
            "candidates": len(screen),
            "pairs": len(by_pair),
            "seg_neutral": len(neutral),
            "seg_neutral_fraction": len(neutral) / max(len(screen), 1),
            "pairs_with_zero_neutral": sum(
                1 for p in by_pair if all(r["d_cells"] > 0 for r in by_pair[p])
            ),
            "per_pair": [
                {
                    "pair": p,
                    "base_flips": by_pair[p][0]["base_flips"],
                    "n": len(by_pair[p]),
                    "min_d_cells": min(r["d_cells"] for r in by_pair[p]),
                    "n_neutral": sum(1 for r in by_pair[p] if r["d_cells"] <= 0),
                }
                for p in sorted(by_pair)
            ],
            "neutral_moves": [
                {k: r[k] for k in ("pair", "dim", "old", "new", "d_cells")}
                for r in neutral
            ],
        },
        "economics": {
            "admit_bar_S": ADMIT_BAR,
            "rate_fee_N1_mean_bytes": fee_n1["shipped_mean"],
            "rate_fee_N1_S": rate_s_n1,
            "required_sum_d_pose_fall_at_seg_neutral": required_sum_d_pose,
            "top12_pose_mass": top12_mass,
            "required_fraction_of_top12_mass": required_sum_d_pose / top12_mass,
            "charter_falsifier_fraction": 0.05,
        },
    }

    rows = []
    for row in refine:
        d_s_rate = fee_n1["shipped_mean"] * S_PER_BYTE
        # Identity control on the seg arithmetic: the stored leg must BE the cell count
        # times the exchange, or the table and the receipt have drifted apart.
        if abs(row["dS_seg"] - row["d_cells"] * S_PER_CELL) > 1e-18:
            raise SystemExit(
                f"{row['key']}: stored dS_seg {row['dS_seg']} != d_cells "
                f"{row['d_cells']} x {S_PER_CELL}"
            )
        rows.append({
            "key": row["key"],
            "pair": row["pair"],
            "move": row["move"],
            "d_cells": row["d_cells"],
            "d_pose_base": row["d_pose_base"],
            "d_pose_stale": row["d_pose_stale"],
            "d_pose_resolved": row["d_pose_resolved"],
            "resolved_over_base": row["resolved_over_base"],
            "base_gap_rel_vs_n600": row["base_gap_rel_vs_n600"],
            "dS_seg": row["dS_seg"],
            "dS_pose_resolved": row["dS_pose_resolved"],
            "dS_seg_plus_pose": row["dS_seg_plus_pose"],
            "dS_with_mean_rate_fee": row["dS_seg_plus_pose"] + d_s_rate,
            "admits_before_rate": row["dS_seg_plus_pose"] < ADMIT_BAR,
            "admits_with_rate": (row["dS_seg_plus_pose"] + d_s_rate) < ADMIT_BAR,
        })
    verdict["actuator_a"] = {
        "rows": sorted(rows, key=lambda r: r["dS_seg_plus_pose"]),
        "n": len(rows),
        "n_admitting_before_rate": sum(1 for r in rows if r["admits_before_rate"]),
        "n_admitting_with_rate": sum(1 for r in rows if r["admits_with_rate"]),
        "best_seg_neutral": min(
            (r for r in rows if r["d_cells"] <= 0),
            key=lambda r: r["resolved_over_base"], default=None,
        ),
        "best_any": min(rows, key=lambda r: r["dS_seg_plus_pose"], default=None),
    }

    verdict["floor_probe"] = {
        "n": len(floor),
        "rows": [
            {k: r[k] for k in (
                "pair", "size", "repeat", "d_cells", "stale_over_base",
                "resolved_over_base",
            )}
            for r in floor
        ],
        "resolved_over_base_min": min(
            (r["resolved_over_base"] for r in floor), default=None
        ),
        "resolved_over_base_max": max(
            (r["resolved_over_base"] for r in floor), default=None
        ),
        "stale_over_base_max": max((r["stale_over_base"] for r in floor), default=None),
        "n_below_base": sum(1 for r in floor if r["resolved_over_base"] < 1.0),
    }

    verdict["actuator_b"] = {
        "n": len(search_b),
        "rows": [
            {k: r[k] for k in (
                "pair", "cell", "old", "new", "d_cells", "d_pose_base",
                "d_pose_stale", "d_pose_resolved", "resolved_over_base",
                "dS_seg", "dS_pose_resolved",
            )}
            for r in search_b
        ],
        "n_seg_neutral": sum(1 for r in search_b if r["d_cells"] <= 0),
        "best_resolved_over_base": min(
            (r["resolved_over_base"] for r in search_b), default=None
        ),
    }

    compose_path = store / "compose/COMPOSE_88.json"
    if compose_path.exists():
        compose = json.loads(compose_path.read_text())
        verdict["composition"] = {
            "pair": compose["pair"],
            "additivity": compose["additivity"],
            "rows": [
                {k: r[k] for k in (
                    "label", "code_moves", "token_edits", "d_cells", "d_pose_base",
                    "d_pose_resolved", "resolved_over_base", "dS_seg",
                    "dS_pose_resolved", "dS_seg_plus_pose",
                )}
                for r in compose["rows"]
            ],
        }

    falsifier_rows = [r for r in rows if r["d_cells"] <= 0]
    neutral_b = [r for r in search_b if r["d_cells"] <= 0]
    best_fall = (
        1.0 - min(r["resolved_over_base"] for r in falsifier_rows)
        if falsifier_rows else 0.0
    )
    best_fall_b = (
        1.0 - min(r["resolved_over_base"] for r in neutral_b) if neutral_b else 0.0
    )
    verdict["falsifier"] = {
        "statement": (
            "if the resolved per-pair d_pose on the treated pairs does not fall by "
            ">= 5 % at seg-neutral admission, the actuator has no supply on the hard "
            "pairs and the family closes there"
        ),
        "seg_neutral_rows_measured_a": len(falsifier_rows),
        "seg_neutral_rows_measured_b": len(neutral_b),
        "best_fall_at_seg_neutral_a": best_fall,
        "best_fall_at_seg_neutral_b": best_fall_b,
        "best_fall_at_seg_neutral": max(best_fall, best_fall_b),
        "threshold": 0.05,
        "fired": max(best_fall, best_fall_b) < 0.05,
        "shortfall_x": 0.05 / max(best_fall, best_fall_b, 1e-12),
    }

    out = Path(args.out) if args.out else store / "VERDICT.json"
    out.write_text(json.dumps(verdict, indent=1, sort_keys=True))
    print(json.dumps({
        "out": str(out),
        "falsifier_fired": verdict["falsifier"]["fired"],
        "best_fall_at_seg_neutral_a": best_fall,
        "best_fall_at_seg_neutral_b": best_fall_b,
        "seg_neutral_of": f"{len(neutral)}/{len(screen)}",
        "n_admitting_with_rate": verdict["actuator_a"]["n_admitting_with_rate"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
