#!/usr/bin/env python3
"""ddm_fe1 final rebuild: re-price the surviving zero-distortion move on a new tree.

By the time this runs, the only thing left of this arm's candidate is ONE seg-neutral
``frame_embed`` code change (pair 331) whose entire value is 61 bytes of container draw.
§13.2 measured that draw to be a ONE-SAMPLE LOTTERY over the perturbed payload, so it
**re-samples whenever the payload's neighbours change** -- and cmp1 changes both the hpac
section and the token tail under the same brotli container.  There is therefore no carry
here at all: the move is re-priced by a REAL build on the new tree, and

    **if the re-priced draw does not clear the admit bar, nothing ships.**

That kill rule is the point of this script.  A 61-byte win that came from a draw has no
claim to survive a re-deal, and shipping it on the strength of the old draw would be
exactly the borrowed-number failure this campaign extincts.

Three builds are recorded either way -- the move alone, the best surviving FiLM cut, and
the two together -- so the closed row carries its own evidence.
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
import ddm_fe1_pose_price as price
import ddm_fe1_rebase as rb
import ddm_jg5_pose_resolve_on_edited_renders as jg5
import ddm_up3_carrier_splice as up3

CELL_COUNT = ab.CELL_COUNT


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pointer-tree", required=True)
    ap.add_argument("--token-field", required=True)
    ap.add_argument("--base-raw", required=True, help="the new pointer's own decode")
    ap.add_argument("--race", required=True, help="RACE.json from the pass-4 race")
    ap.add_argument("--rebase", required=True, help="REBASE.json, for the FiLM survivors")
    ap.add_argument("--base-cells", type=int, required=True)
    ap.add_argument("--base-mean-d-pose", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--admit-bar", type=float, default=ab.ADMIT_BAR)
    ap.add_argument("--film-top", type=int, default=1, help="how many FiLM survivors to race alongside")
    args = ap.parse_args(argv)
    fe1._set_threads(args.threads)

    tree = Path(args.pointer_tree).resolve()
    base_bytes = (tree / "archive.zip").stat().st_size
    race = json.loads(Path(args.race).read_text())
    neutral = dict(race["neutral_move"])
    rebase = json.loads(Path(args.rebase).read_text())
    survivors = sorted(rebase["survivors"], key=lambda e: -int(e["cells_new"]))

    # ---- sections: the semantic must not have moved ----------------------------
    old = rb.split_sections(fe1.LIVE_ARCHIVE, fe1.LIVE_TREE)
    new = rb.split_sections(tree / "archive.zip", tree)
    for name in ("hpac", "semantic", "carrier", "tail"):
        print(f"  {name:>9}: {len(old[name])} -> {len(new[name])}  identical={old[name] == new[name]}")
    if old["semantic"] != new["semantic"]:
        raise fe1.Fe1Error("the new pointer changed the SEMANTIC section: this is a re-run")

    body = fe1.load_body(with_raw=False, verify_shas=False)
    body.tokens = rb.load_field(Path(args.token_field))
    raw = fe1._open_raw(Path(args.base_raw))

    # ---- the null build must reproduce the new pointer exactly -----------------
    section = fe1.load_semantic_section(
        archive_path=tree / "archive.zip", runtime_dir=tree / "runtime"
    )
    shipped = up3.parse_shipped_body(tree, verify_sha=False)
    live_carrier = np.asarray(shipped.codes, dtype=np.int32)
    null = ab.build_candidate_archive(
        section, section.codes.astype(np.int64), live_carrier, tree_dir=tree, verify=True
    )
    if null["archive_size"] != base_bytes:
        raise fe1.Fe1Error(
            f"null build is {null['archive_size']} B, the pointer is {base_bytes} B"
        )
    print(f"  null build: {null['archive_size']} B, byte-identical to the new pointer")

    # ---- the move's pose leg, re-solved from THIS tree's coefficients ----------
    pair = int(neutral["pair"])
    row = np.asarray(neutral["final_row"], dtype=np.int64)
    fe1.set_pair_codes(body, pair, row)
    frame = fe1.render_pair(body, pair)[0]
    fe1.restore_pair_codes(body, pair)
    base_inst = rb._instrument(raw, tree)
    moved_inst = rb._instrument(price._MemoryOverlayRaw(raw, {pair: frame}), tree)
    codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    threshold = jg5.materiality_dd_threshold(args.base_mean_d_pose)
    neutral["d_pose_base"] = float(br1.evaluate_codes(base_inst, pair, codes[pair][None])[0])
    control = jg5.refine_pair(base_inst, pair, codes[pair], dd_threshold=threshold)
    neutral["d_pose_base_resolved"] = float(control["final_d_pose"])
    neutral["d_pose_stale"] = float(br1.evaluate_codes(moved_inst, pair, codes[pair][None])[0])
    refined = jg5.refine_pair(moved_inst, pair, codes[pair], dd_threshold=threshold)
    neutral["d_pose_resolved"] = float(refined["final_d_pose"])
    neutral["resolved_codes"] = [int(c) for c in refined["codes"]]
    print(
        f"  pair {pair}: d_pose {neutral['d_pose_base']:.4e} (control "
        f"{neutral['d_pose_base_resolved']:.4e}) -> stale {neutral['d_pose_stale']:.4e} "
        f"-> resolved {neutral['d_pose_resolved']:.4e}"
    )

    base_pose = float(args.base_mean_d_pose)
    leg0 = math.sqrt(10.0 * base_pose)
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def price_set(subset, label):
        nonlocal best
        codes_t = section.codes.astype(np.int64).copy()
        carrier = live_carrier.copy()
        for entry in subset:
            codes_t[int(entry["pair"])] = np.asarray(entry["final_row"], dtype=np.int64)
            carrier[int(entry["pair"])] = np.asarray(entry["resolved_codes"], dtype=np.int32)
        built = ab.build_candidate_archive(
            section, codes_t, carrier, tree_dir=tree, verify=True
        )
        cells = sum(int(e.get("cells_new", 0)) for e in subset)
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
            "pairs": sorted(int(e["pair"]) for e in subset),
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
            "admits": bool(dS < args.admit_bar),
        }
        results.append(entry)
        print(
            f"  {label:<24} {len(subset)}p/{cells}c: {built['archive_size']} B "
            f"({entry['d_archive_bytes']:+d}) -> dS {dS:+.6e}"
            f"{'  ADMITS' if entry['admits'] else '  below bar'}",
            flush=True,
        )
        if best is None or dS < best["dS_total"]:
            best = entry
            (out_dir / "archive.zip").write_bytes(built["archive_bytes"])
            np.save(out_dir / "frame_embed_codes.npy", codes_t.astype(np.int8))
            np.save(out_dir / "carrier_codes.npy", carrier)
        return entry

    print("\nthe three builds, recorded either way:")
    price_set([neutral], "neutral-alone")
    film = survivors[: args.film_top]
    if film:
        price_set(film, f"film-best-{len(film)}")
        price_set([*film, neutral], "film-plus-neutral")

    verdict = "SHIP" if best and best["admits"] else "CLOSE"
    report = {
        "schema": "ddm_fe1_cmp1_rebuild.v1",
        "axis": (
            "d_seg [macOS-CPU advisory, jg1/sj1 instrument, DALI GT]; d_pose "
            "[cpu_torch fp32]; bytes EXACT through real builds on the NEW tree"
        ),
        "score_claim": False,
        "pointer": {"tree": str(tree), "bytes": base_bytes, "sha256": null["archive_sha256"]},
        "null_build_byte_identical": True,
        "base_cells": args.base_cells,
        "base_mean_d_pose": base_pose,
        "admit_bar": args.admit_bar,
        "neutral_move": neutral,
        "builds": results,
        "best": best,
        "verdict": verdict,
        "kill_rule": (
            "the container draw RE-SAMPLES when the payload's neighbours change; if the "
            "re-priced draw does not clear the admit bar, nothing ships and the row closes"
        ),
    }
    (out_dir / "CMP1_REBUILD.json").write_text(json.dumps(report, indent=1))
    print(f"\nVERDICT: {verdict}")
    if verdict == "CLOSE":
        print(
            "  the re-deal did not clear the bar. Nothing ships; the three builds above "
            "are the closed row's evidence."
        )
        return 3
    print(f"  best: {best['label']} {best['archive_bytes']} B, dS {best['dS_total']:+.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
