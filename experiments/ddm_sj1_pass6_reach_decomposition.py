#!/usr/bin/env python3
"""Decompose ddm_sj1's pass-6 reach and run the pre-registered stale-pair control (F11).

Pass 6 re-runs a repair family this arm CLOSED at pass 5 (80 admitted cells of 12,614 =
0.634 %, under its own pre-registered 1 %/pass rule).  Re-running it is legitimate only
because the OBJECT changed underneath: moves 38/39/40/42 rewrote 365 of the 600 token
planes.  That makes exactly one honest question, and this module answers it by
measurement rather than by assertion:

    of the cells pass 6 repairs, how many owe their existence to the changed render,
    and how many are simply pass 5's own accepted moves that the Lagrange sweep dropped
    and never shipped?

Three columns, from set arithmetic on the fields themselves:

  carryover   a repaired position pass 5 also accepted.  Floor 157 -- pass 5 accepted 235
              moves on 126 pairs but only 78 positions on 42 pairs entered the pointer.
              This column owes NOTHING to the object change.
  fresh_new   a NEW position on one of the 365 pairs whose plane changed since move 37.
              This column is the re-opening claim, and only this column.
  stale_new   a NEW position on one of the 235 pairs whose plane did NOT change.  The
              render is a per-pair function of that pair's own plane and index, so those
              renders are BYTE-IDENTICAL to the ones pass 5 searched; this column should
              be ~0 and anything in it is the control's residual, not signal.

F11, pre-registered before the number (2026-09-10T14:54:51Z): on the STALE pairs pass 5
accepted moves on and the sweep then dropped, pass 6 must re-find the same positions.
This is rp1's control-group shape (307 unchanged pairs transferred 2,401 of 2,401 tokens).

No scorer runs here, no archive is built, no pointer is written: this reads four token
fields and one pass ledger and does set arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "src")]
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

N_PAIRS = 600
AXIS = "[macOS-CPU advisory / exact token-field set arithmetic, no scorer]"


def load_field(path: Path) -> np.ndarray:
    """Load a 600-plane token field, refusing anything that is not exactly that."""
    with np.load(path, allow_pickle=False) as data:
        missing = {str(i) for i in range(N_PAIRS)} - set(data.files)
        if missing:
            raise SystemExit(f"{path} is not a {N_PAIRS}-plane field ({len(missing)} absent)")
        return np.stack([data[str(i)] for i in range(N_PAIRS)])


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass5-base", type=Path, required=True,
                        help="move 37 field: the body pass 5 proposed against")
    parser.add_argument("--pass5-after", type=Path, required=True,
                        help="pass 5's field_after.npz (its ACCEPTED moves, before admission)")
    parser.add_argument("--pass6-base", type=Path, required=True,
                        help="move 42 field: the body pass 6 proposed against (the shipped one)")
    parser.add_argument("--pass6-after", type=Path, required=True,
                        help="pass 6's field_after.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    b37 = load_field(args.pass5_base)
    a5 = load_field(args.pass5_after)
    b42 = load_field(args.pass6_base)
    a6 = load_field(args.pass6_after)

    # Positions each pass CHANGED, as boolean planes.  A "position" is one token cell.
    p5_moves = a5 != b37
    p6_moves = a6 != b42
    # A pair is FRESH when its plane moved between the two proposal bodies.
    plane_moved = (b37 != b42).reshape(N_PAIRS, -1).any(axis=1)
    fresh = np.flatnonzero(plane_moved)
    stale = np.flatnonzero(~plane_moved)

    # Carryover is a position BOTH passes moved.  It is deliberately position-level and not
    # value-level: pass 5 proved that position reachable, whatever token it landed on.
    #
    # One narrow over-count is accepted on purpose: on a FRESH pair whose pass-5 move WAS
    # admitted, the position already differs from the move-37 base, so if pass 6 moves it
    # again to a third value the position is counted as carryover rather than as new.  That
    # biases the split AGAINST the re-opening claim, which is the direction a claim should
    # be biased in.
    carryover = p5_moves & p6_moves
    new = p6_moves & ~p5_moves
    fresh_mask = np.zeros(N_PAIRS, dtype=bool)
    fresh_mask[fresh] = True

    per_pair_new = new.reshape(N_PAIRS, -1).sum(axis=1)
    per_pair_carry = carryover.reshape(N_PAIRS, -1).sum(axis=1)

    # F11: stale pairs pass 5 accepted moves on, whose moves the sweep dropped (so the
    # pass-6 base still equals the pass-5 base there).  Pass 6 must re-find them.
    p5_per_pair = p5_moves.reshape(N_PAIRS, -1).sum(axis=1)
    control_pairs = [int(p) for p in stale if p5_per_pair[p] > 0]
    refound = int(sum(int(per_pair_carry[p]) for p in control_pairs))
    offered = int(sum(int(p5_per_pair[p]) for p in control_pairs))
    control_fraction = (refound / offered) if offered else None
    control_missed = [
        {"pair": int(p), "pass5_positions": int(p5_per_pair[p]), "refound": int(per_pair_carry[p])}
        for p in control_pairs
        if int(per_pair_carry[p]) != int(p5_per_pair[p])
    ]

    doc = {
        "schema": "ddm_sj1_pass6_reach_decomposition.v1",
        "score_claim": False,
        "axis": AXIS,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in (
                ("pass5_base_move37", args.pass5_base),
                ("pass5_after", args.pass5_after),
                ("pass6_base_move42", args.pass6_base),
                ("pass6_after", args.pass6_after),
            )
        },
        "pairs": {"fresh": int(fresh.size), "stale": int(stale.size)},
        "pass5": {
            "accepted_positions": int(p5_moves.sum()),
            "accepted_pairs": int((p5_per_pair > 0).sum()),
        },
        "pass6_positions": {
            "total": int(p6_moves.sum()),
            "carryover": int(carryover.sum()),
            "new_on_fresh_pairs": int(per_pair_new[fresh_mask].sum()),
            "new_on_stale_pairs": int(per_pair_new[~fresh_mask].sum()),
        },
        "per_pair_rate": {
            "fresh_positions_per_pair": float(p6_moves.reshape(N_PAIRS, -1).sum(axis=1)[fresh_mask].mean())
            if fresh.size else None,
            "stale_positions_per_pair": float(p6_moves.reshape(N_PAIRS, -1).sum(axis=1)[~fresh_mask].mean())
            if stale.size else None,
        },
        "F11_stale_pair_control": {
            "why_strict_identity_is_the_right_test": (
                "--pass-index is ledger-only (it reaches no selector, seed or ordering), the shard "
                "striding and batch are the same 5/8 pass 5 used, and the render is a per-pair "
                "function of that pair's own plane and index.  A stale pair therefore presents pass "
                "6 with a byte-identical search, so it must accept the SAME positions -- not merely "
                "the same count."
            ),
            "control_pairs": control_pairs,
            "positions_offered": offered,
            "positions_refound": refound,
            "fraction": control_fraction,
            "pairs_disagreeing": control_missed,
            "verdict": (
                "PASS" if offered and refound == offered
                else "VACUOUS (no stale pair carries a pass-5 accepted move)" if not offered
                else "FAIL"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("inputs",)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
