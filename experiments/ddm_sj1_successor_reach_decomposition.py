#!/usr/bin/env python3
"""Decompose a ddm_sj1 SUCCESSOR pass's reach and run its pre-registered controls.

Pass 6 and pass 7 each answered the same question with a script named after themselves,
and pass 8 would have made a third.  The question is not pass-specific, so this module is
named for the SHAPE instead: given the prior round's base/after fields and rows, and this
round's, it splits this round's reach into the three columns the arm's own law predicts
and runs the two controls the pre-registration named.

    carryover   a repaired position the PRIOR round also accepted but its Lagrange sweep
                DROPPED.  It owes nothing to the object change: the pair's plane, and
                therefore its render, is byte-identical to what the prior search saw.
    fresh_new   a NEW position on a pair whose plane CHANGED between the two base fields.
                This column is the re-opening claim, and only this column.
    stale_new   a NEW position on a pair whose plane did NOT change.  The render is a
                per-pair function of that pair's own plane and index, so with striding
                and batch matched this column must be EMPTY.  A nonzero value falsifies
                [[repair_family_exhausts_per_object_20260910]] and is reported as a
                falsification, never absorbed into the headline.

Controls, both pre-registered before the search ran:
  F11  every position the prior round accepted and dropped must be re-found, on every
       stale pair, with zero pairs disagreeing.
  F14  stale_new must be exactly zero.

Axis [macOS-CPU advisory, jg1 instrument, DALI GT lineage].  No scorer runs here; this is
set arithmetic on fields and accepted-move rows.  No score is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

N_PAIRS, EVAL_H, EVAL_W = 600, 384, 512


class ReachDecompositionError(RuntimeError):
    """An input is not the object the decomposition needs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_field(path: Path) -> dict[int, np.ndarray]:
    """An edit npz as {pair: plane}; a pair ABSENT is not 'unchanged', it is absent."""
    planes: dict[int, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as blob:
        for key in blob.files:
            plane = np.asarray(blob[key], dtype=np.uint8)
            if plane.shape != (EVAL_H, EVAL_W):
                raise ReachDecompositionError(f"plane {key} has shape {plane.shape}")
            planes[int(key)] = plane
    return planes


def load_rows(paths: list[Path]) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[int(row["pair"])] = row
    if len(rows) != N_PAIRS:
        raise ReachDecompositionError(f"rows cover {len(rows)} pairs, not {N_PAIRS}")
    return rows


def position_key(move: object) -> tuple[int, int, int]:
    """The canonical identity of one accepted token move: (token_y, token_x, new_class).

    A search row's ``accepted`` entries are DICTS carrying the move plus its own outcome
    (``repaired``, ``old_class``, the site that moved).  Two rounds find "the same
    position" when they set the same token cell to the same class; the outcome fields are
    per-round and must NOT enter the key, or a re-found position whose repair count
    differed would read as a miss.  ``tuple(move)`` on a dict yields its KEY tuple, which
    is identical for every move -- a control that would collapse 218 distinct moves into
    130 and silently pass.  MEASURED against pass 7's own stored expectation: this key
    re-finds 154 of 154, while (token_x, token_y, new_class), (site_y, site_x, new_class)
    and (token_y, token_x, old_class) find 0, 3 and 0.

    A pre-registration that stored the triple directly is accepted in that shape too, so
    an older round's expectation stays readable without rewriting it.
    """
    if isinstance(move, dict):
        return (int(move["token_y"]), int(move["token_x"]), int(move["new_class"]))
    if isinstance(move, (list, tuple)) and len(move) == 3:
        return (int(move[0]), int(move[1]), int(move[2]))
    raise ReachDecompositionError(f"accepted move has an unreadable shape: {move!r}")


def accepted_positions(rows: dict[int, dict]) -> dict[int, set[tuple[int, int, int]]]:
    """The accepted moves per pair, as canonical position keys."""
    out: dict[int, set[tuple[int, int, int]]] = {}
    for pair, row in rows.items():
        keys = {position_key(move) for move in row["accepted"]}
        if len(keys) != len(row["accepted"]):
            raise ReachDecompositionError(
                f"pair {pair}: {len(row['accepted'])} accepted moves collapse to {len(keys)} "
                "position keys; the key is not an identity on this round's rows"
            )
        out[pair] = keys
    return out


def changed_pairs(prior_base: dict[int, np.ndarray], current_base: dict[int, np.ndarray]) -> set[int]:
    """Pairs whose PLANE differs between the two base fields, i.e. whose render moved.

    Both fields are cumulative edit npzs over the same original body tokens, so a pair
    present in one and absent from the other genuinely differs; comparing only the
    intersection would silently call such a pair unchanged.
    """
    moved: set[int] = set()
    for pair in set(prior_base) | set(current_base):
        a, b = prior_base.get(pair), current_base.get(pair)
        if a is None or b is None or not np.array_equal(a, b):
            moved.add(pair)
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-base", type=Path, required=True,
                        help="the PRIOR round's input field npz")
    parser.add_argument("--prior-rows", type=Path, nargs="+", required=True,
                        help="the PRIOR round's rows_shard_*.jsonl")
    parser.add_argument("--prior-kept-pairs", type=Path, required=True,
                        help="the PRIOR round's admitted kept_pairs.json")
    parser.add_argument("--current-base", type=Path, required=True,
                        help="THIS round's input field npz")
    parser.add_argument("--current-rows", type=Path, nargs="+", required=True,
                        help="THIS round's rows_shard_*.jsonl")
    parser.add_argument("--current-field-after", type=Path, required=True,
                        help="THIS round's merged field_after.npz")
    parser.add_argument("--expectation", type=Path, required=True,
                        help="the F11 expectation written BEFORE the search")
    parser.add_argument("--residual-entering", type=int, required=True)
    parser.add_argument("--prior-label", default="prior")
    parser.add_argument("--current-label", default="current")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prior_base = load_field(args.prior_base)
    current_base = load_field(args.current_base)
    prior_rows = load_rows(list(args.prior_rows))
    current_rows = load_rows(list(args.current_rows))
    expectation = json.loads(args.expectation.read_text())

    prior_accepted = accepted_positions(prior_rows)
    current_accepted = accepted_positions(current_rows)
    kept = set(json.loads(args.prior_kept_pairs.read_text()))
    dropped = {pair: moves for pair, moves in prior_accepted.items()
               if moves and pair not in kept}

    moved = changed_pairs(prior_base, current_base)
    stale = set(range(N_PAIRS)) - moved

    carryover_tokens = fresh_tokens = stale_new_tokens = 0
    carryover_cells = fresh_cells = stale_new_cells = 0
    stale_new_by_pair: dict[str, list] = {}
    for pair, moves in current_accepted.items():
        if not moves:
            continue
        repaired = float(current_rows[pair]["flips_repaired"])
        share = repaired / len(moves) if moves else 0.0
        prior_moves = dropped.get(pair, set())
        for move in moves:
            if move in prior_moves:
                carryover_tokens += 1
                carryover_cells += share
            elif pair in moved:
                fresh_tokens += 1
                fresh_cells += share
            else:
                stale_new_tokens += 1
                stale_new_cells += share
                stale_new_by_pair.setdefault(str(pair), []).append(list(move))

    # F11: every DROPPED prior position must be re-found, on every stale pair.
    expected_by_pair = {int(k): {position_key(m) for m in v}
                        for k, v in expectation["expected_refound_positions_by_pair"].items()}
    refound = 0
    disagreeing: list[int] = []
    for pair, wanted in expected_by_pair.items():
        found = wanted & current_accepted.get(pair, set())
        refound += len(found)
        if found != wanted:
            disagreeing.append(pair)
    expected_total = sum(len(v) for v in expected_by_pair.values())

    reach_cells = sum(float(row["flips_repaired"]) for row in current_rows.values())
    reach_tokens = sum(int(row["tokens_changed"]) for row in current_rows.values())
    pairs_edited = sum(1 for row in current_rows.values() if row["tokens_changed"] > 0)
    after = load_field(args.current_field_after)
    field_pairs_edited = sum(
        1 for pair, plane in after.items()
        if not np.array_equal(plane, current_base.get(pair, plane))
        or pair not in current_base
    )

    document = {
        "schema": "ddm_sj1_successor_reach_decomposition.v1",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage]",
        "prior_label": args.prior_label,
        "current_label": args.current_label,
        "inputs": {
            "prior_base": {"path": str(args.prior_base), "sha256": sha256_file(args.prior_base)},
            "current_base": {"path": str(args.current_base), "sha256": sha256_file(args.current_base)},
            "current_field_after": {"path": str(args.current_field_after),
                                    "sha256": sha256_file(args.current_field_after)},
            "expectation": {"path": str(args.expectation), "sha256": sha256_file(args.expectation)},
        },
        "residual_entering": args.residual_entering,
        "reach_cells": reach_cells,
        "reach_tokens": reach_tokens,
        "pairs_edited": pairs_edited,
        "fraction_of_residual": reach_cells / args.residual_entering,
        "decomposition": {
            "moved_pairs": len(moved),
            "stale_pairs": len(stale),
            "stale_pairs_carrying_prior_dropped_positions": len(dropped),
            "carryover_tokens": carryover_tokens,
            "carryover_cells": carryover_cells,
            "fresh_tokens": fresh_tokens,
            "fresh_cells": fresh_cells,
            "fresh_cells_per_moved_pair": (fresh_cells / len(moved)) if moved else None,
            "stale_new_tokens": stale_new_tokens,
            "stale_new_cells": stale_new_cells,
            "stale_new_positions_by_pair": stale_new_by_pair,
        },
        "F11_prior_dropped_position_control": {
            "expected_positions": expected_total,
            "refound": refound,
            "fraction": (refound / expected_total) if expected_total else None,
            "pairs_disagreeing": len(disagreeing),
            "pairs_disagreeing_list": sorted(disagreeing),
            "verdict": "PASS" if expected_total and refound == expected_total and not disagreeing
                       else "FAIL",
        },
        "F14_reach_law_control": {
            "statement": ("not one NEW repair may exist on a pair whose render did not move "
                          "([[repair_family_exhausts_per_object_20260910]])"),
            "stale_new_tokens": stale_new_tokens,
            "verdict": "PASS" if stale_new_tokens == 0 else "FALSIFIED",
            "if_falsified": ("a new position on a stale pair means the family does NOT exhaust per "
                             "object; report it as a falsification of the arm's own law, never as "
                             "extra yield"),
        },
        "F2_no_silent_revert": {
            "pairs_claimed_by_rows": pairs_edited,
            "pairs_edited_in_field": field_pairs_edited,
            "verdict": "PASS" if pairs_edited == field_pairs_edited else "FAIL",
        },
        "score_claim": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in document.items() if k != "inputs"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
