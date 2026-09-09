#!/usr/bin/env python3
"""ddm_fe1: race the re-based FiLM set against the ITEM 5 neutral move, on pass 4.

Three things this must not get wrong, and does not:

* the ITEM 5 move is only usable if it is STILL seg-neutral on the new field -- verified
  separately (pair 331, 21 -> 21 flips on sj1's pass-4 field, and 331 is not one of the
  112 pairs pass 4 edits, so the two sets are disjoint);
* it still changes that pair's render, so it still owes a carrier re-solve and a pose
  leg -- measured here, with the unmoved-render control, not assumed free;
* -68 and -61 do NOT add.  §13.2 measured the container delta to be a one-sample
  lottery, so every combination is priced by its own REAL build and nothing is summed.
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

import ddm_br1_pose_basis_reorientation as br1
import ddm_fe1_admit_and_build as ab
import ddm_fe1_frame_embedding_search as fe1
import ddm_fe1_rebase as rb
import ddm_jg5_pose_resolve_on_edited_renders as jg5
import ddm_up3_carrier_splice as up3

CELL_COUNT = ab.CELL_COUNT


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pointer-tree", required=True)
    ap.add_argument("--token-field", required=True)
    ap.add_argument("--base-raw", required=True)
    ap.add_argument("--rebase", required=True, help="REBASE.json from ddm_fe1_rebase")
    ap.add_argument("--item5", default=str(fe1.WORK / "item5/ITEM5.json"))
    ap.add_argument("--base-cells", type=int, required=True)
    ap.add_argument("--base-mean-d-pose", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    fe1._set_threads(args.threads)

    tree = Path(args.pointer_tree).resolve()
    base_bytes = (tree / "archive.zip").stat().st_size
    rebase = json.loads(Path(args.rebase).read_text())
    survivors = list(rebase["survivors"])
    neutral = json.loads(Path(args.item5).read_text())["chosen_moves"][0]

    body = fe1.load_body(with_raw=False, verify_shas=False)
    body.tokens = rb.load_field(Path(args.token_field))
    raw = fe1._open_raw(Path(args.base_raw))

    # ---- the neutral move's own pose leg on THIS tree ---------------------------
    pair = int(neutral["pair"])
    row = body.section.codes[pair].astype(np.int64).copy()
    row[int(neutral["dim"])] = int(neutral["new"])
    fe1.set_pair_codes(body, pair, row)
    frame = fe1.render_pair(body, pair)[0]
    fe1.restore_pair_codes(body, pair)
    base_inst = rb._instrument(raw, tree)
    import ddm_fe1_pose_price as price

    moved_inst = rb._instrument(price._MemoryOverlayRaw(raw, {pair: frame}), tree)
    codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    threshold = jg5.materiality_dd_threshold(args.base_mean_d_pose)
    d_base = float(br1.evaluate_codes(base_inst, pair, codes[pair][None])[0])
    control = jg5.refine_pair(base_inst, pair, codes[pair], dd_threshold=threshold)
    d_stale = float(br1.evaluate_codes(moved_inst, pair, codes[pair][None])[0])
    refined = jg5.refine_pair(moved_inst, pair, codes[pair], dd_threshold=threshold)
    neutral_row = {
        "pair": pair,
        "dim": int(neutral["dim"]),
        "final_row": row.tolist(),
        "cells_new": 0,
        "d_pose_base": d_base,
        "d_pose_base_resolved": float(control["final_d_pose"]),
        "d_pose_stale": d_stale,
        "d_pose_resolved": float(refined["final_d_pose"]),
        "resolved_codes": [int(c) for c in refined["codes"]],
        "seg_neutral": True,
    }
    print(
        f"neutral pair {pair}: d_pose {d_base:.4e} (control "
        f"{control['final_d_pose']:.4e}) -> stale {d_stale:.4e} -> resolved "
        f"{refined['final_d_pose']:.4e}",
        flush=True,
    )

    # ---- the race ---------------------------------------------------------------
    section = fe1.load_semantic_section(
        archive_path=tree / "archive.zip", runtime_dir=tree / "runtime"
    )
    shipped = up3.parse_shipped_body(tree, verify_sha=False)
    live_carrier = np.asarray(shipped.codes, dtype=np.int32)
    base_pose = float(args.base_mean_d_pose)
    leg0 = math.sqrt(10.0 * base_pose)

    for entry in survivors:
        delta = entry["d_pose_resolved"] - entry["d_pose_base"]
        entry["dS_seg"] = -entry["cells_new"] * 100.0 / CELL_COUNT
        entry["dS_pose"] = math.sqrt(10.0 * (base_pose + delta / fe1.N_PAIRS)) - leg0
        entry["value"] = entry["dS_seg"] + entry["dS_pose"]
    ranked = sorted(survivors, key=lambda e: e["value"])

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def price(subset, label):
        nonlocal best
        codes_t = section.codes.astype(np.int64).copy()
        carrier = live_carrier.copy()
        for entry in subset:
            codes_t[int(entry["pair"])] = np.asarray(entry["final_row"], dtype=np.int64)
            carrier[int(entry["pair"])] = np.asarray(entry["resolved_codes"], dtype=np.int32)
        built = ab.build_candidate_archive(
            section, codes_t, carrier, tree_dir=tree, verify=True
        )
        cells = sum(int(e["cells_new"]) for e in subset)
        pose_mean = base_pose + sum(
            e["d_pose_resolved"] - e["d_pose_base"] for e in subset
        ) / fe1.N_PAIRS
        d_seg_new = (args.base_cells - cells) / CELL_COUNT
        dS = (
            100.0 * (d_seg_new - args.base_cells / CELL_COUNT)
            + (math.sqrt(10.0 * pose_mean) - leg0)
            + (built["archive_size"] - base_bytes) * fe1.RATE_PER_BYTE
        )
        entry = {
            "label": label,
            "pairs": len(subset),
            "cells": cells,
            "archive_bytes": built["archive_size"],
            "archive_sha256": built["archive_sha256"],
            "d_archive_bytes": built["archive_size"] - base_bytes,
            "semantic_container": built["semantic_container"],
            "d_seg_local": d_seg_new,
            "d_pose_mean": pose_mean,
            "dS_seg": 100.0 * (d_seg_new - args.base_cells / CELL_COUNT),
            "dS_pose": math.sqrt(10.0 * pose_mean) - leg0,
            "dS_rate": (built["archive_size"] - base_bytes) * fe1.RATE_PER_BYTE,
            "dS_total": dS,
            "admits": bool(dS < ab.ADMIT_BAR),
            "includes_neutral": any(int(e["pair"]) == pair for e in subset),
            "pairs_included": sorted(int(e["pair"]) for e in subset),
        }
        results.append(entry)
        print(
            f"  {label:<26} {len(subset):>3}p/{cells:>3}c: {built['archive_size']} B "
            f"({entry['d_archive_bytes']:+d}) -> dS {dS:+.6e}"
            f"{'  ADMITS' if entry['admits'] else ''}",
            flush=True,
        )
        if best is None or dS < best["dS_total"]:
            best = entry
            (out_dir / "archive.zip").write_bytes(built["archive_bytes"])
            np.save(out_dir / "frame_embed_codes.npy", codes_t.astype(np.int8))
            np.save(out_dir / "carrier_codes.npy", carrier)
        return entry

    print("\nFiLM prefix cuts:")
    for count in range(1, len(ranked) + 1):
        price(ranked[:count], f"film-{count}")
    print("\nthe three MAIN asked for:")
    price([neutral_row], "neutral-alone")
    film_best = [e for e in ranked if e["pair"] in set(best["pairs_included"])] or ranked[:1]
    price([*film_best, neutral_row], "film-best-plus-neutral")
    print("\nneighbourhood around the winner:")
    anchor = [e for e in [*ranked, neutral_row] if e["pair"] in set(best["pairs_included"])]
    for entry in list(anchor):
        subset = [e for e in anchor if e["pair"] != entry["pair"]]
        if subset:
            price(subset, f"best-minus-{entry['pair']}")
    taken = set(best["pairs_included"])
    for entry in [*ranked, neutral_row]:
        if int(entry["pair"]) in taken:
            continue
        anchor2 = [e for e in [*ranked, neutral_row] if e["pair"] in taken]
        price([*anchor2, entry], f"best-plus-{entry['pair']}")

    report = {
        "schema": "ddm_fe1_pass4_race.v1",
        "axis": (
            "d_seg [macOS-CPU advisory, jg1/sj1 instrument, DALI GT]; d_pose "
            "[cpu_torch fp32]; bytes EXACT through real builds on sj1 pass 4"
        ),
        "score_claim": False,
        "pointer": {"tree": str(tree), "bytes": base_bytes},
        "base_cells": args.base_cells,
        "base_mean_d_pose": base_pose,
        "neutral_move": neutral_row,
        "cuts": results,
        "best": best,
    }
    (out_dir / "RACE.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"best": {k: best[k] for k in ("label", "pairs", "cells", "archive_bytes", "d_archive_bytes", "dS_total", "includes_neutral")}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
