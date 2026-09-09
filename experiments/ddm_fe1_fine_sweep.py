#!/usr/bin/env python3
"""ddm_fe1: a finer subset sweep around the admission's optimum, priced by real builds.

The semantic section's cost is a container-break fee, not a per-code price, so the
archive size is genuinely non-monotonic in the edit set -- the coarse sweep measured
181,305 B at 15 pairs and 181,405 B at 27.  Choosing the subset whose REAL archive is
smallest is rate engineering on exact, deterministic bytes, not fitting to noise: the
bytes selected here are the bytes that ship.  Every cut is built and parsed back.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_fe1_admit_and_build as ab
import ddm_fe1_frame_embedding_search as fe1
import ddm_up3_carrier_splice as up3

CELL_COUNT = ab.CELL_COUNT


def score_subset(section, body, live_carrier, subset, base_pose, verify=True):
    codes = section.codes.astype(np.int64).copy()
    carrier = live_carrier.copy()
    for row in subset:
        codes[int(row["pair"])] = np.asarray(row["final_row"], dtype=np.int64)
        carrier[int(row["pair"])] = np.asarray(row["resolved_codes"], dtype=np.int32)
    built = ab.build_candidate_archive(section, codes, carrier, verify=verify)
    cells = sum(int(r["cells"]) for r in subset)
    pose_mean = base_pose + sum(
        r["d_pose_resolved"] - r["d_pose_base"] for r in subset
    ) / fe1.N_PAIRS
    d_seg_new = (fe1.LIVE_D_SEG_CELLS - cells) / CELL_COUNT
    leg0 = math.sqrt(10.0 * base_pose)
    dS = (
        100.0 * (d_seg_new - fe1.LIVE_D_SEG_LOCAL)
        + (math.sqrt(10.0 * pose_mean) - leg0)
        + (built["archive_size"] - fe1.LIVE_ARCHIVE_BYTES) * fe1.RATE_PER_BYTE
    )
    return built, cells, pose_mean, d_seg_new, dS


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-rows", default=str(fe1.WORK / "admission/pose/pose_rows.jsonl"))
    ap.add_argument("--out-dir", default=str(fe1.WORK / "candidate"))
    ap.add_argument("--base-mean-d-pose", type=float, default=fe1.LIVE_D_POSE)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    fe1._set_threads(args.threads)

    rows = [
        json.loads(line)
        for line in Path(args.pose_rows).read_text().splitlines()
        if line.strip()
    ]
    section = fe1.load_semantic_section()
    body = up3.parse_shipped_body(fe1.LIVE_TREE, verify_sha=False)
    live_carrier = np.asarray(body.codes, dtype=np.int32)
    base_pose = float(args.base_mean_d_pose)
    leg0 = math.sqrt(10.0 * base_pose)

    for row in rows:
        pose_delta = row["d_pose_resolved"] - row["d_pose_base"]
        row["dS_seg"] = -row["cells"] * 100.0 / CELL_COUNT
        row["dS_pose"] = math.sqrt(10.0 * (base_pose + pose_delta / fe1.N_PAIRS)) - leg0
        row["value"] = row["dS_seg"] + row["dS_pose"]
    ranked = sorted(rows, key=lambda r: r["value"])

    def _codes(section, subset):
        codes = section.codes.astype(np.int64).copy()
        for row in subset:
            codes[int(row["pair"])] = np.asarray(row["final_row"], dtype=np.int64)
        return codes.astype(np.int8)

    def _carrier(live_carrier, subset):
        carrier = live_carrier.copy()
        for row in subset:
            carrier[int(row["pair"])] = np.asarray(row["resolved_codes"], dtype=np.int32)
        return carrier

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def consider(subset, label):
        nonlocal best
        built, cells, pose_mean, d_seg_new, dS = score_subset(
            section, body, live_carrier, subset, base_pose
        )
        entry = {
            "label": label,
            "pairs": len(subset),
            "cells": cells,
            "archive_bytes": built["archive_size"],
            "archive_sha256": built["archive_sha256"],
            "d_archive_bytes": built["archive_size"] - fe1.LIVE_ARCHIVE_BYTES,
            "semantic_container": built["semantic_container"],
            "d_seg_local": d_seg_new,
            "d_pose_mean": pose_mean,
            "dS_seg": 100.0 * (d_seg_new - fe1.LIVE_D_SEG_LOCAL),
            "dS_pose": math.sqrt(10.0 * pose_mean) - leg0,
            "dS_rate": (built["archive_size"] - fe1.LIVE_ARCHIVE_BYTES) * fe1.RATE_PER_BYTE,
            "dS_total": dS,
            "admits": bool(dS < ab.ADMIT_BAR),
            "pairs_included": sorted(int(r["pair"]) for r in subset),
        }
        results.append(entry)
        print(
            f"  {label:<22} {len(subset):>3} pairs / {cells:>3} cells: "
            f"{built['archive_size']} B ({entry['d_archive_bytes']:+d}) -> dS {dS:+.6e}"
            f"{'  ADMITS' if entry['admits'] else ''}",
            flush=True,
        )
        if best is None or dS < best["dS_total"]:
            best = entry
            (out_dir / "archive.zip").write_bytes(built["archive_bytes"])
            np.save(out_dir / "frame_embed_codes.npy", _codes(section, subset))
            np.save(out_dir / "carrier_codes.npy", _carrier(live_carrier, subset))
        return entry

    print("prefix cuts (ranked by per-pair value):")
    for count in range(8, len(ranked) + 1):
        consider(ranked[:count], f"prefix-{count}")

    print("\nleave-one-out around the best prefix:")
    anchor = [r for r in ranked if r["pair"] in set(best["pairs_included"])]
    for row in list(anchor):
        subset = [r for r in anchor if r["pair"] != row["pair"]]
        if subset:
            consider(subset, f"best-minus-{row['pair']}")

    print("\nadd-one-back to the best:")
    chosen = set(best["pairs_included"])
    for row in ranked:
        if row["pair"] in chosen:
            continue
        consider([r for r in ranked if r["pair"] in chosen] + [row], f"best-plus-{row['pair']}")

    result = {
        "schema": "ddm_fe1_fine_sweep.v1",
        "axis": (
            "d_seg [macOS-CPU advisory, jg1/sj1 instrument, DALI GT]; d_pose "
            "[cpu_torch fp32, n600 base]; bytes EXACT through real archive builds"
        ),
        "score_claim": False,
        "pointer_archive_bytes": fe1.LIVE_ARCHIVE_BYTES,
        "pointer_archive_sha256": fe1.LIVE_ARCHIVE_SHA256,
        "pointer_score_t4": fe1.LIVE_SCORE_T4,
        "base_mean_d_pose": base_pose,
        "admit_bar": ab.ADMIT_BAR,
        "cuts": results,
        "best": best,
        "projected_score_t4": fe1.LIVE_SCORE_T4 + best["dS_total"],
    }
    (out_dir / "FINE_SWEEP.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"best": {k: best[k] for k in ("label", "pairs", "cells", "archive_bytes", "dS_total")},
                      "projected_score_t4": result["projected_score_t4"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
