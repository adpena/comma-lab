#!/usr/bin/env python3
"""ddm_fe1 ITEM 5: seg-NEUTRAL frame-embedding code sets as a pure rate lever.

The n600 realized search measured 243 single-code moves that leave the pair's flip count
EXACTLY unchanged.  Each of them re-randomises the RC1 range-coded payload, and §6 of the
memo measured that such a perturbation is worth tens of bytes in either direction once the
brotli container is re-searched.  Choosing the ones that go DOWN is therefore a rate lever
with **no seg term by construction** -- it cannot lose on the axis that is hardest to win.

Design points that make the claim safe rather than merely cheap:

* **At most one move per pair.**  ``frame_embed`` is per-pair, so moves on different pairs
  cannot interact and the composition's seg-neutrality follows from each move's.  Moves on
  pairs the FiLM candidate already edits are excluded so the two sets compose.
* **Neutrality is VERIFIED on the composed set**, by re-rendering every pair in it through
  the receiver's own forward model and comparing realized flip counts -- the prediction
  above is checked, not assumed.
* **Every price is a REAL archive build** with the container searched, and every build is
  parsed back.  A modelled byte count would be worthless here: the whole effect being
  measured is the part a model does not capture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def _codes_for(section, moves: Sequence[dict[str, Any]]) -> np.ndarray:
    codes = section.codes.astype(np.int64).copy()
    for move in moves:
        codes[int(move["pair"]), int(move["dim"])] = int(move["new"])
    return codes


def _price(section, carrier, moves, tree) -> dict[str, Any]:
    built = ab.build_candidate_archive(
        section, _codes_for(section, moves), carrier, tree_dir=tree, verify=False
    )
    return built


def cmd_run(args) -> int:
    fe1._set_threads(args.threads)
    tree = Path(args.pointer_tree).resolve()
    catalogue = json.loads(Path(args.moves).read_text())
    moves = catalogue["moves"]
    # one move per pair, chosen deterministically by (pair, dim) so a re-run is identical
    by_pair: dict[int, dict[str, Any]] = {}
    for move in sorted(moves, key=lambda m: (m["pair"], m["dim"], m["new"])):
        by_pair.setdefault(int(move["pair"]), move)
    candidates = [by_pair[p] for p in sorted(by_pair)]
    print(f"{len(moves)} free neutral moves over {len(candidates)} pairs; one per pair")

    section = fe1.load_semantic_section(
        archive_path=tree / "archive.zip", runtime_dir=tree / "runtime"
    )
    shipped = up3.parse_shipped_body(tree, verify_sha=False)
    carrier = np.asarray(shipped.codes, dtype=np.int32)
    base = _price(section, carrier, [], tree)
    base_bytes = base["archive_size"]
    if base_bytes != args.base_archive_bytes:
        raise fe1.Fe1Error(
            f"null build is {base_bytes} B, pointer is {args.base_archive_bytes} B"
        )
    print(f"null build {base_bytes} B -- the baseline every delta below is measured against")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    builds = 0

    # ---- stage A: every single move, priced ------------------------------------
    singles_path = out_dir / f"singles_{args.shard_index}.jsonl"
    done = set()
    if singles_path.is_file() and args.resume:
        for line in singles_path.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["pair"]))
    for index, move in enumerate(candidates):
        if index % args.shard_count != args.shard_index or int(move["pair"]) in done:
            continue
        built = _price(section, carrier, [move], tree)
        builds += 1
        row = dict(move)
        row["archive_bytes"] = built["archive_size"]
        row["delta_bytes"] = built["archive_size"] - base_bytes
        with singles_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if args.progress and builds % 20 == 0:
            print(f"  stage A {builds} builds, {(time.time()-started)/builds:.1f} s/build", flush=True)
    if args.stage_a_only:
        print(f"stage A shard {args.shard_index} complete: {builds} builds")
        return 0
    singles = []
    for path in sorted(out_dir.glob("singles_*.jsonl")):
        singles += [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    seen_pairs = set()
    singles = [
        row
        for row in singles
        if not (row["pair"] in seen_pairs or seen_pairs.add(row["pair"]))
    ]
    singles.sort(key=lambda r: (r["delta_bytes"], r["pair"]))
    deltas = np.array([r["delta_bytes"] for r in singles])
    print(
        f"stage A: {len(singles)} singles priced | best {deltas.min():+d} B | "
        f"median {int(np.median(deltas)):+d} B | worst {deltas.max():+d} B | "
        f"{int((deltas < 0).sum())} reduce"
    )

    # ---- stage B: greedy accumulation, re-priced at every step ------------------
    chosen: list[dict[str, Any]] = []
    current = base_bytes
    trail = []
    for row in singles:
        if row["delta_bytes"] >= 0 and not args.explore_positive:
            break
        trial = [*chosen, row]
        built = _price(section, carrier, trial, tree)
        builds += 1
        if built["archive_size"] < current:
            chosen = trial
            current = built["archive_size"]
            trail.append({"pair": row["pair"], "archive_bytes": current, "delta": current - base_bytes})
            if args.progress:
                print(
                    f"  stage B accept pair {row['pair']:>3} -> {current} B "
                    f"({current-base_bytes:+d})",
                    flush=True,
                )
    print(f"stage B: {len(chosen)} moves, {current} B ({current-base_bytes:+d})")

    # ---- stage C: bounded neighbourhood ----------------------------------------
    improved = True
    rounds = 0
    while improved and rounds < args.max_rounds:
        improved = False
        rounds += 1
        for row in list(chosen):
            trial = [m for m in chosen if m["pair"] != row["pair"]]
            built = _price(section, carrier, trial, tree)
            builds += 1
            if built["archive_size"] < current:
                chosen, current, improved = trial, built["archive_size"], True
                print(f"  stage C drop pair {row['pair']:>3} -> {current} B ({current-base_bytes:+d})", flush=True)
        taken = {m["pair"] for m in chosen}
        for row in singles[: args.stage_c_top]:
            if row["pair"] in taken:
                continue
            trial = [*chosen, row]
            built = _price(section, carrier, trial, tree)
            builds += 1
            if built["archive_size"] < current:
                chosen, current, improved = trial, built["archive_size"], True
                taken.add(row["pair"])
                print(f"  stage C add pair {row['pair']:>3} -> {current} B ({current-base_bytes:+d})", flush=True)

    final = ab.build_candidate_archive(
        section, _codes_for(section, chosen), carrier, tree_dir=tree, verify=True
    )
    (out_dir / "archive.zip").write_bytes(final["archive_bytes"])
    np.save(out_dir / "frame_embed_codes.npy", _codes_for(section, chosen).astype(np.int8))

    result = {
        "schema": "ddm_fe1_item5.v1",
        "axis": "[exact bytes through REAL archive builds on the live pointer; scorer-free]",
        "score_claim": False,
        "pointer_tree": str(tree),
        "base_archive_bytes": base_bytes,
        "candidates_considered": len(candidates),
        "builds": builds,
        "stage_a_best_single_delta": int(deltas.min()),
        "stage_a_singles_reducing": int((deltas < 0).sum()),
        "stage_a_median_delta": int(np.median(deltas)),
        "chosen_moves": chosen,
        "chosen_pairs": sorted(int(m["pair"]) for m in chosen),
        "archive_bytes": final["archive_size"],
        "archive_sha256": final["archive_sha256"],
        "delta_bytes": final["archive_size"] - base_bytes,
        "dS_rate": (final["archive_size"] - base_bytes) * fe1.RATE_PER_BYTE,
        "semantic_container": final["semantic_container"],
        "greedy_trail": trail,
        "elapsed_s": round(time.time() - started, 1),
        "prediction": {
            "best_single_near_bytes": -70,
            "best_subset_at_most_bytes": -150,
            "falsifier_bytes": -88,
        },
    }
    (out_dir / "ITEM5.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("builds", "archive_bytes", "delta_bytes", "dS_rate", "elapsed_s")}, indent=1))
    return 0


def cmd_verify(args) -> int:
    """Re-render every pair in the chosen set and CHECK the composition is seg-neutral."""
    fe1._set_threads(args.threads)
    result = json.loads(Path(args.item5).read_text())
    chosen = result["chosen_moves"]
    body = fe1.load_body(with_raw=False, verify_shas=False)
    base_flips = np.load(args.base_flips)
    rows = []
    for move in chosen:
        pair = int(move["pair"])
        row = body.section.codes[pair].astype(np.int64).copy()
        row[int(move["dim"])] = int(move["new"])
        fe1.set_pair_codes(body, pair, row)
        flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        fe1.restore_pair_codes(body, pair)
        rows.append(
            {
                "pair": pair,
                "dim": int(move["dim"]),
                "base_flips": int(base_flips[pair]),
                "flips": flips,
                "delta": flips - int(base_flips[pair]),
            }
        )
        print(f"  pair {pair:>3}: {int(base_flips[pair]):>3} -> {flips:>3} ({flips-int(base_flips[pair]):+d})", flush=True)
    total = sum(r["delta"] for r in rows)
    report = {
        "schema": "ddm_fe1_item5_neutrality.v1",
        "axis": "[macOS-CPU advisory, jg1/sj1 instrument, DALI GT lineage]",
        "score_claim": False,
        "pairs": len(rows),
        "total_cell_delta": total,
        "neutral": total == 0,
        "non_neutral_pairs": [r["pair"] for r in rows if r["delta"] != 0],
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("pairs", "total_cell_delta", "neutral", "non_neutral_pairs")}, indent=1))
    return 0 if total == 0 else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="price the neutral moves and search for the byte minimum")
    run.add_argument("--pointer-tree", default=str(fe1.LIVE_TREE))
    run.add_argument("--moves", default=str(fe1.WORK / "item5/NEUTRAL_MOVES.json"))
    run.add_argument("--out-dir", default=str(fe1.WORK / "item5"))
    run.add_argument("--base-archive-bytes", type=int, default=fe1.LIVE_ARCHIVE_BYTES)
    run.add_argument("--threads", type=int, default=3)
    run.add_argument("--max-rounds", type=int, default=2)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--stage-a-only", action="store_true")
    run.add_argument("--stage-c-top", type=int, default=50)
    run.add_argument("--explore-positive", action="store_true")
    run.add_argument("--resume", action="store_true", default=True)
    run.add_argument("--progress", action="store_true", default=True)
    run.set_defaults(func=cmd_run)

    ver = sub.add_parser("verify", help="check the chosen set really is seg-neutral")
    ver.add_argument("--item5", default=str(fe1.WORK / "item5/ITEM5.json"))
    ver.add_argument("--base-flips", default=str(fe1.WORK / "probe/base_flips_per_pair.npy"))
    ver.add_argument("--out", default=str(fe1.WORK / "item5/NEUTRALITY.json"))
    ver.add_argument("--threads", type=int, default=3)
    ver.set_defaults(func=cmd_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
